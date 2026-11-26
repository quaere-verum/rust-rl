use super::{distribution::RecurrentPolicyDistribution, structured::ConditionalBatch};
use crate::{
    core::{
        action::{Action, ActionArgument, ConditionalAction},
        stateful_on_policy::{ConditionalOutcome, DistributionSnapshot},
        structured::EntityId,
    },
    nn::{device, structured::ConditionalPolicyLaw, to_vec, vector},
    objectives::{
        mdpo::{MdpoTargets, mdpo_objective, normalized_occupancy_weights},
        on_policy::{ppo_clipped_surrogate, ppo_ratio_and_kl},
        vmpo::{dual_objective, policy_objective, temperature_weights, top_half_indices_into},
    },
};
use burn::tensor::{Int, Tensor, TensorData, activation::softplus};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

const WAIT: ConditionalAction = ConditionalAction {
    operation: 0,
    argument: None,
};
const FIRST: ConditionalAction = ConditionalAction {
    operation: 1,
    argument: Some(ActionArgument::Entity(EntityId { kind: 0, id: 17 })),
};
const SECOND: ConditionalAction = ConditionalAction {
    operation: 1,
    argument: Some(ActionArgument::Entity(EntityId { kind: 0, id: 3 })),
};
const DISABLED: ConditionalAction = ConditionalAction {
    operation: 2,
    argument: None,
};
const KEYS: [ConditionalAction; 3] = [WAIT, FIRST, SECOND];
const CHOSEN: [usize; 4] = [1, 0, 2, 1];
const LATENTS: [Option<Vec<f32>>; 4] = [const { None }; 4];
const ADVANTAGES: [f32; 4] = [3., -2., 1.4, -0.5];
const OLD: [[f64; 3]; 4] = [
    [0.60, 0.10, 0.30],
    [0.75, 0.18, 0.07],
    [0.30, 0.20, 0.50],
    [0.34, 0.36, 0.30],
];
const X: f64 = 0.4;
const Y: f64 = -0.24;
const OPERATIONS: [f64; 4] = [0.6, -0.7, 0., 0.4];
const TARGETS: [f64; 4] = [-0.3, 0.9, -1.1, 0.];

fn actions() -> [Action; 4] {
    CHOSEN.map(|row| Action::Conditional(Arc::new(KEYS[row])))
}

fn old_logs() -> Tensor<1> {
    vector(&std::array::from_fn::<_, 4, _>(|row| {
        OLD[row][CHOSEN[row]].ln() as f32
    }))
}

fn behavior() -> ConditionalBatch {
    let snapshots = OLD.map(|probabilities| {
        let mut outcomes = vec![ConditionalOutcome {
            action: DISABLED,
            log_prob: None,
        }];
        outcomes.extend(KEYS.into_iter().zip(probabilities).map(|(action, p)| {
            ConditionalOutcome {
                action,
                log_prob: Some(p.ln() as f32),
            }
        }));
        DistributionSnapshot::conditional(outcomes).unwrap()
    });
    ConditionalBatch::from_snapshots(&snapshots, &device()).unwrap()
}

fn tracked(value: f64) -> Tensor<1> {
    vector(&[value as f32]).require_grad()
}

fn current(x: &Tensor<1>, y: &Tensor<1>) -> ConditionalBatch {
    let rows = OPERATIONS
        .into_iter()
        .zip(TARGETS)
        .map(|(operation, target)| {
            let operation = x.clone() + operation;
            let target = y.clone() + target;
            let wait = softplus(operation.clone(), 1.).neg();
            let choose = softplus(operation.neg(), 1.).neg();
            let first = choose.clone() - softplus(target.clone().neg(), 1.);
            let second = choose - softplus(target, 1.);
            // Inference snapshots sort by semantic key. Current model order need not match.
            ConditionalPolicyLaw::new(
                vec![FIRST, SECOND, WAIT],
                Tensor::cat(vec![first, second, wait], 0),
            )
            .unwrap()
        })
        .collect();
    ConditionalBatch::from_laws(rows).unwrap()
}

fn probabilities(x: f64, y: f64) -> [[f64; 3]; 4] {
    let sigmoid = |logit: f64| 1. / (1. + (-logit).exp());
    std::array::from_fn(|row| {
        let operation = sigmoid(x + OPERATIONS[row]);
        let target = sigmoid(y + TARGETS[row]);
        [
            1. - operation,
            operation * target,
            operation * (1. - target),
        ]
    })
}

fn ratios(p: &[[f64; 3]; 4]) -> [f64; 4] {
    std::array::from_fn(|row| p[row][CHOSEN[row]] / OLD[row][CHOSEN[row]])
}

fn entropy(row: &[f64; 3]) -> f64 {
    -row.iter().map(|&p| p * p.ln()).sum::<f64>()
}

fn kl(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    p.iter().zip(q).map(|(&p, &q)| p * (p / q).ln()).sum()
}

fn number(tensor: Tensor<1>) -> f64 {
    f64::from(tensor.into_scalar::<f32>())
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 4e-6,
        "actual={actual} expected={expected}"
    );
}

fn loss_and_derivatives(
    loss: Tensor<1>,
    first: (&Tensor<1>, f64),
    second: (&Tensor<1>, f64),
    reference: impl Fn(f64, f64) -> f64,
) {
    let (x, a) = first;
    let (y, b) = second;
    let step = 1e-5;
    close(number(loss.clone()), reference(a, b));
    let gradients = loss.backward();
    let da = (reference(a + step, b) - reference(a - step, b)) / (2. * step);
    let db = (reference(a, b + step) - reference(a, b - step)) / (2. * step);
    assert!(
        da.abs() > 1e-5 && db.abs() > 1e-5,
        "fixture needs both derivatives"
    );
    close(number(x.grad(&gradients).unwrap()), da);
    close(number(y.grad(&gradients).unwrap()), db);
}

#[test]
fn structured_ppo_clips_joint_likelihood_ratios_and_differentiates_entropy() {
    let (x, y) = (tracked(X), tracked(Y));
    let current = current(&x, &y);
    let (logp, cache) = current.log_prob_with_cache(&actions(), &LATENTS).unwrap();
    let (ratio, kl_proxy) = ppo_ratio_and_kl(logp, old_logs());
    let expected_ratios = ratios(&probabilities(X, Y));
    assert!(expected_ratios[0] > 1.2 && expected_ratios[1] < 0.8);
    assert!(expected_ratios[2..].iter().all(|r| (0.8..1.2).contains(r)));
    for (actual, expected) in to_vec(ratio.clone())
        .unwrap()
        .into_iter()
        .zip(expected_ratios)
    {
        close(f64::from(actual), expected);
    }
    close(
        number(kl_proxy),
        expected_ratios.iter().map(|r| r - 1. - r.ln()).sum::<f64>() / 4.,
    );
    let entropy_bonus = current
        .entropy_with_cache(cache, &mut ChaCha8Rng::seed_from_u64(0))
        .unwrap()
        .mean();
    let loss = ppo_clipped_surrogate(ratio, vector(&ADVANTAGES), 0.2) - entropy_bonus * 0.17;
    loss_and_derivatives(loss, (&x, X), (&y, Y), |x, y| {
        let p = probabilities(x, y);
        let rows = ratios(&p)
            .into_iter()
            .zip(ADVANTAGES)
            .map(|(ratio, advantage)| {
                let a = f64::from(advantage);
                (ratio * a).min(ratio.clamp(0.8, 1.2) * a)
            })
            .sum::<f64>();
        -(rows + 0.17 * p.iter().map(entropy).sum::<f64>()) / 4.
    });
}

#[test]
fn structured_vmpo_weights_top_joint_scores_and_differentiates_forward_kl_and_duals() {
    let (x, y) = (tracked(X), tracked(Y));
    let current = current(&x, &y);
    let (eta, alpha) = (0.8_f64, 0.29_f64);
    let mut top = Vec::new();
    top_half_indices_into(&ADVANTAGES, &mut top);
    assert_eq!(top, [0, 2]);
    let top_advantages = top.iter().map(|&i| ADVANTAGES[i]).collect::<Vec<_>>();
    let (weights, maximum) = temperature_weights(&top_advantages, eta as f32);
    let indices = Tensor::<1, Int>::from_data(
        TensorData::new(
            top.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [top.len()],
        ),
        &device(),
    );
    let selected = current
        .log_prob(&actions(), &LATENTS)
        .unwrap()
        .select(0, indices);
    let forward = current.kl(&behavior()).unwrap().mean();
    let p = probabilities(X, Y);
    let reference_kl = p.iter().zip(OLD).map(|(p, q)| kl(p, &q)).sum::<f64>() / 4.;
    let reverse = p.iter().zip(OLD).map(|(p, q)| kl(&q, p)).sum::<f64>() / 4.;
    assert!((reference_kl - reverse).abs() > 1e-3);
    let forward_number = number(forward.clone());
    close(forward_number, reference_kl);
    let loss = policy_objective(weights, selected, forward, alpha as f32);
    loss_and_derivatives(loss, (&x, X), (&y, Y), |x, y| {
        let p = probabilities(x, y);
        let high_weight = (3. / eta).exp();
        let lower_weight = (f64::from(ADVANTAGES[2]) / eta).exp();
        let cross_entropy = -(high_weight * p[0][1].ln() + lower_weight * p[2][2].ln())
            / (high_weight + lower_weight);
        cross_entropy + alpha * p.iter().zip(OLD).map(|(p, q)| kl(p, &q)).sum::<f64>() / 4.
    });

    let (log_eta, log_alpha) = (tracked(-0.3), tracked(0.4));
    let loss = dual_objective(
        log_eta.clone(),
        log_alpha.clone(),
        &top_advantages,
        maximum,
        0.02,
        0.11,
        forward_number as f32,
    );
    loss_and_derivatives(loss, (&log_eta, -0.3), (&log_alpha, 0.4), |a, b| {
        let eta = a.exp().ln_1p() + 1e-6;
        let alpha = b.exp().ln_1p() + 1e-6;
        let mean_exponential = ((3. / eta).exp() + (f64::from(ADVANTAGES[2]) / eta).exp()) / 2.;
        eta * (0.02 + mean_exponential.ln()) + alpha * (0.11 - reference_kl)
    });
}

#[test]
fn structured_mdpo_uses_unclipped_joint_ratios_and_occupancy_weighted_kl() {
    let (x, y) = (tracked(X), tracked(Y));
    let current = current(&x, &y);
    let occupancy = [1., 0.32, 0.10, 0.72];
    let weights = normalized_occupancy_weights(&occupancy).unwrap();
    let coefficient = 1.7;
    let output = mdpo_objective(
        &current,
        &behavior(),
        MdpoTargets {
            actions: &actions(),
            latents: &LATENTS,
            old_log_probs: &old_logs(),
            weights: &vector(&weights),
            advantages: &vector(&ADVANTAGES),
        },
        coefficient as f32,
    )
    .unwrap();
    let reference = |x, y| {
        let p = probabilities(x, y);
        let r = ratios(&p);
        let total = occupancy.iter().map(|&w| f64::from(w)).sum::<f64>();
        let weights = occupancy.map(|w| f64::from(w) / total);
        let surrogate = (0..4)
            .map(|i| weights[i] * r[i] * f64::from(ADVANTAGES[i]))
            .sum::<f64>();
        let divergence = (0..4).map(|i| weights[i] * kl(&p[i], &OLD[i])).sum::<f64>();
        (surrogate, divergence)
    };
    let (surrogate, divergence) = reference(X, Y);
    close(number(output.surrogate), surrogate);
    close(number(output.kl), divergence);
    loss_and_derivatives(output.loss, (&x, X), (&y, Y), |x, y| {
        let (surrogate, divergence) = reference(x, y);
        -surrogate + coefficient * divergence
    });
}
