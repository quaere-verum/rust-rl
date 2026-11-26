use super::*;
use crate::{
    core::action::Action,
    data::rollout::{BootstrappedBatch, OnPolicyMetadata},
    nn::matrix,
};
use crate::{
    data::rollout::OnPolicyBatch,
    nn::{distributions::Distribution, finite_loss, to_vec},
};

fn categorical(logits: &[f32], masks: &[bool], n: usize) -> Distribution {
    Distribution::Categorical {
        logits: matrix(logits.to_vec(), n, logits.len() / n),
        masks: masks.to_vec(),
    }
}
fn batch_for(old: &Distribution, actions: &[usize], advantages: &[f32]) -> OnPolicyBatch {
    let actions: Vec<_> = actions.iter().map(|&a| Action::Discrete(a)).collect();
    let latents = vec![None; actions.len()];
    let lp = to_vec(old.log_prob(&actions, &latents).unwrap()).unwrap();
    OnPolicyBatch {
        observations: vec![crate::core::env::Observation::new(vec![1.]); actions.len()],
        actions,
        latents,
        returns: vec![0.; advantages.len()],
        discount_weights: vec![1.; advantages.len()],
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: lp,
            values: vec![0.; advantages.len()],
            advantages: advantages.to_vec(),
        }),
        episode_count: 1,
    }
}
fn softmax64(x: &[f64], mask: &[bool]) -> Vec<f64> {
    let max = x
        .iter()
        .zip(mask)
        .filter(|(_, m)| **m)
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<_> = x
        .iter()
        .zip(mask)
        .map(|(x, m)| if *m { (x - max).exp() } else { 0. })
        .collect();
    let sum: f64 = p.iter().sum();
    p.iter().map(|p| p / sum).collect()
}
fn oracle_loss(
    x: &[f64],
    old: &[f64],
    masks: &[bool],
    actions: &[usize],
    adv: &[f64],
    weights: &[f64],
    beta: f64,
) -> f64 {
    let k = x.len() / actions.len();
    let mut loss = 0.;
    for i in 0..actions.len() {
        let p = softmax64(&x[i * k..(i + 1) * k], &masks[i * k..(i + 1) * k]);
        let q = softmax64(&old[i * k..(i + 1) * k], &masks[i * k..(i + 1) * k]);
        let kl: f64 = p
            .iter()
            .zip(&q)
            .filter(|(p, _)| **p > 0.)
            .map(|(p, q)| p * (p / q).ln())
            .sum();
        loss += weights[i] * (-p[actions[i]] / q[actions[i]] * adv[i] + beta * kl);
    }
    loss
}

#[test]
fn unclipped_surrogate_and_reverse_kl_gradients_match_independent_f64() {
    let old = [-1., 0.7, 2., 0.2, -0.9, 1., 0.5, -0.2, 0.8];
    let x = [0.6, -0.2, 3., -0.3, 0.5, 0.2, 0.1, 0.9, -0.7];
    let masks = [true, true, false, true, true, true, false, true, true];
    let actions = [0, 2, 1];
    let adv = [1.3, -0.8, 0.];
    let behavior = categorical(&old, &masks, 3);
    let mut b = batch_for(&behavior, &actions, &adv);
    b.discount_weights = vec![1., 2., 4.];
    for beta in [0., 2.5] {
        let input = matrix(x.to_vec(), 3, 3).require_grad();
        let current = Distribution::Categorical {
            logits: input.clone(),
            masks: masks.to_vec(),
        };
        let objective =
            MirrorObjective::new(&b, behavior.clone(), MdpoWeighting::DiscountedOccupancy).unwrap();
        let result = objective.evaluate(&current, beta).unwrap();
        let reference = oracle_loss(
            &x.map(f64::from),
            &old.map(f64::from),
            &masks,
            &actions,
            &adv.map(f64::from),
            &[1. / 7., 2. / 7., 4. / 7.],
            beta as f64,
        );
        assert!((finite_loss(&result.loss).unwrap() - reference).abs() < 2e-6);
        let grad = to_vec(input.grad(&result.loss.backward()).unwrap()).unwrap();
        for j in 0..x.len() {
            let mut plus = x.map(f64::from);
            let mut minus = plus;
            let eps = 1e-5;
            plus[j] += eps;
            minus[j] -= eps;
            let f = |z: &[f64]| {
                oracle_loss(
                    z,
                    &old.map(f64::from),
                    &masks,
                    &actions,
                    &adv.map(f64::from),
                    &[1. / 7., 2. / 7., 4. / 7.],
                    beta as f64,
                )
            };
            let numerical = (f(&plus) - f(&minus)) / (2. * eps);
            assert!(
                (grad[j] as f64 - numerical).abs() < 3e-6,
                "j={j} grad={} oracle={numerical}",
                grad[j]
            );
            if !masks[j] {
                assert_eq!(grad[j], 0.);
            }
        }
    }
}

#[test]
fn kl_direction_asymmetry_and_zero_then_nonzero_gradient() {
    let masks = [true, true, true];
    let old = categorical(&[0.1_f32.ln(), 0.3_f32.ln(), 0.6_f32.ln()], &masks, 1);
    let input = matrix(vec![0.6_f32.ln(), 0.25_f32.ln(), 0.15_f32.ln()], 1, 3).require_grad();
    let new = Distribution::Categorical {
        logits: input.clone(),
        masks: masks.to_vec(),
    };
    let reverse = new.kl(&old).unwrap();
    let forward = old.kl(&new).unwrap();
    let expected = 0.6_f64 * (0.6_f64 / 0.1).ln()
        + 0.25_f64 * (0.25_f64 / 0.3).ln()
        + 0.15_f64 * (0.15_f64 / 0.6).ln();
    assert!((finite_loss(&reverse).unwrap() - expected).abs() < 2e-6);
    assert!((finite_loss(&reverse).unwrap() - finite_loss(&forward).unwrap()).abs() > 0.1);
    assert!(
        to_vec(input.grad(&reverse.backward()).unwrap())
            .unwrap()
            .iter()
            .any(|x| x.abs() > 0.01)
    );
    let equal = matrix(vec![0.1_f32.ln(), 0.3_f32.ln(), 0.6_f32.ln()], 1, 3).require_grad();
    let same = Distribution::Categorical {
        logits: equal.clone(),
        masks: masks.to_vec(),
    };
    assert!(
        to_vec(
            equal
                .grad(&same.kl(&old).unwrap().sum().backward())
                .unwrap()
        )
        .unwrap()
        .iter()
        .all(|x| x.abs() < 1e-6)
    );
}

#[test]
fn exact_gaussian_kl_is_invariant_under_shared_affine_tanh_transform() {
    for bounds in [None, Some((vec![-2., 1.], vec![4., 5.]))] {
        let new = Distribution::Gaussian {
            mean: matrix(vec![0.4, -0.8], 1, 2),
            log_std: matrix(vec![0.2, -0.3], 1, 2),
            bounds: bounds.clone(),
        };
        let old = Distribution::Gaussian {
            mean: matrix(vec![-0.5, 0.1], 1, 2),
            log_std: matrix(vec![-0.1, 0.4], 1, 2),
            bounds,
        };
        let expected =
            (-0.1_f64 - 0.2 + ((0.4_f64).exp() + 0.9_f64.powi(2)) / (2. * (-0.2_f64).exp()) - 0.5)
                + (0.4_f64 + 0.3 + ((-0.6_f64).exp() + (-0.9_f64).powi(2)) / (2. * 0.8_f64.exp())
                    - 0.5);
        assert!((finite_loss(&new.kl(&old).unwrap()).unwrap() - expected).abs() < 1e-6);
    }
}

#[test]
fn singleton_mask_has_zero_loss_and_gradients_and_support_mismatch_rejects() {
    let input = matrix(vec![999., -2., -999.], 1, 3).require_grad();
    let mask = [false, true, false];
    let old = categorical(&[0., 0., 0.], &mask, 1);
    let b = batch_for(&old, &[1], &[0.]);
    let current = Distribution::Categorical {
        logits: input.clone(),
        masks: mask.to_vec(),
    };
    let result = MirrorObjective::new(&b, old.clone(), MdpoWeighting::Uniform)
        .unwrap()
        .evaluate(&current, 3.)
        .unwrap();
    assert_eq!(finite_loss(&result.loss).unwrap(), 0.);
    assert_eq!(
        to_vec(input.grad(&result.loss.backward()).unwrap()).unwrap(),
        vec![0.; 3]
    );
    assert!(
        current
            .kl(&categorical(&[0., 0., 0.], &[true, true, false], 1))
            .is_err()
    );
}

#[test]
fn exponential_tabular_oracle_matches_gradient_optimization_and_grid_search() {
    let old_p = [0.2_f64, 0.3, 0.5];
    let q = [-1.0_f64, 0.4, 1.2];
    let t = 0.7_f64;
    let z: f64 = old_p.iter().zip(q).map(|(p, q)| p * (t * q).exp()).sum();
    let oracle: Vec<_> = old_p
        .iter()
        .zip(q)
        .map(|(p, q)| p * (t * q).exp() / z)
        .collect();
    let score = |p: &[f64]| {
        p.iter().zip(q).map(|(p, q)| p * q).sum::<f64>()
            - p.iter()
                .zip(old_p)
                .filter(|(p, _)| **p > 0.)
                .map(|(p, o)| p * (p / o).ln())
                .sum::<f64>()
                / t
    };
    assert!(score(&oracle) > score(&old_p));
    for a in 1..100 {
        for b in 1..100 - a {
            let p = [
                a as f64 / 100.,
                b as f64 / 100.,
                (100 - a - b) as f64 / 100.,
            ];
            assert!(score(&oracle) >= score(&p) - 1e-12);
        }
    }
    // Enumerate each old action once and weight its advantage by 3*pi_old: the empirical surrogate is exact.
    let logits: Vec<_> = old_p.iter().map(|p| p.ln() as f32).collect();
    let old = categorical(&logits.repeat(3), &[true; 9], 3);
    let adv: Vec<_> = old_p
        .iter()
        .zip(q)
        .map(|(p, q)| (3. * p * q) as f32)
        .collect();
    let b = batch_for(&old, &[0, 1, 2], &adv);
    let target = MirrorObjective::new(&b, old, MdpoWeighting::Uniform).unwrap();
    let mut x = matrix(logits, 1, 3).require_grad();
    for _ in 0..500 {
        let new = Distribution::Categorical {
            logits: x.clone().repeat_dim(0, 3),
            masks: vec![true; 9],
        };
        let loss = target.evaluate(&new, (1. / t) as f32).unwrap().loss;
        let g = x.grad(&loss.backward()).unwrap();
        x = (x - g * 0.15).detach().require_grad();
    }
    let fit = to_vec(burn::tensor::activation::softmax(x, 1)).unwrap();
    for (p, o) in fit.iter().zip(&oracle) {
        assert!((*p as f64 - o).abs() < 2e-5, "{fit:?} vs {oracle:?}");
    }
}

#[test]
fn extreme_unclipped_ratio_returns_meaningful_numerical_error() {
    let old = categorical(&[-100., 0.], &[true, true], 1);
    let b = batch_for(&old, &[0], &[1.]);
    let target = MirrorObjective::new(&b, old, MdpoWeighting::Uniform).unwrap();
    assert!(matches!(
        target.evaluate(&categorical(&[0., -1.], &[true, true], 1), 1.),
        Err(crate::Error::Numerical(_))
    ));
}
