#![cfg(feature = "neural")]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::categorical_dpg::{CategoricalDpg, CategoricalDpgConfig},
    core::{checkpoint::Checkpoint, transition::Transition},
    data::replay::ReplayItem,
    nn::{matrix, to_vec, vector},
    objectives::{categorical::CategoricalSupport, risk::Risk},
    prelude::*,
};

fn learner(risk: Risk, support: CategoricalSupport) -> CategoricalDpg {
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let mut critic = QValue::new(1, 1, 3, &[], &mut rng).unwrap();
    critic.network.layers[0].weight =
        Param::from_tensor(matrix(vec![0., 0., 0., -1., 0.5, 2.], 2, 3));
    critic.network.layers[0].bias = Some(Param::from_tensor(vector(&[
        0.2_f32.ln(),
        0.3_f32.ln(),
        0.5_f32.ln(),
    ])));
    let mut config = CategoricalDpgConfig::new(support);
    config.training.target_noise = 0.;
    config.training.policy_delay = 1;
    config.risk = risk;
    // The target action is exactly zero at the initial model boundary.
    let mut policy = policy;
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.])));
    CategoricalDpg::new(policy, critic, config, 17).unwrap()
}

fn replay() -> ReplayBatch {
    ReplayBatch {
        items: [
            (0.5_f32, 0., true, false),
            (-1., 0.5, false, true),
            (0.5, 0.25, false, false),
        ]
        .into_iter()
        .map(|(reward, discount, terminated, truncated)| ReplayItem {
            transition: Transition {
                observation: Observation::new(vec![0.]),
                action: Action::Continuous(vec![0.]),
                reward: reward.into(),
                next_observation: Observation::new(vec![0.]),
                terminated,
                truncated,
            },
            discount,
        })
        .collect(),
    }
}

#[test]
fn learner_projects_mixed_terminal_and_physical_time_targets() {
    let support = CategoricalSupport::uniform(-2., 2., 3).unwrap();
    let mut learner = learner(Risk::Expected, support);
    let observed = learner.update(&replay()).unwrap().metrics["critic_loss"];
    let targets = [[0., 0.75, 0.25], [0.35, 0.65, 0.], [0., 0.675, 0.325]];
    let expected = -targets
        .into_iter()
        .flatten()
        .zip([0.2_f64, 0.3, 0.5].into_iter().cycle())
        .map(|(weight, predicted)| weight * predicted.ln())
        .sum::<f64>()
        / 3.;
    assert!(
        (observed - expected).abs() < 3e-6,
        "{observed} != {expected}"
    );
}

#[test]
fn expected_return_and_fractional_cvar_preserve_action_gradients() {
    let support = CategoricalSupport::uniform(-2., 2., 3).unwrap();
    let obs = [Observation::new(vec![0.])];
    for (risk, expected_at_zero) in [
        (Risk::Expected, 0.6),
        (Risk::Cvar { alpha: 0.4 }, -1.),
        (Risk::Cvar { alpha: 1. }, 0.6),
    ] {
        let learner = learner(risk, support.clone());
        let evaluate = |a| {
            learner
                .action_utility((&obs).into(), matrix(vec![a], 1, 1))
                .unwrap()
                .into_scalar::<f32>()
        };
        assert!((evaluate(0.) - expected_at_zero).abs() < 1e-6);
        let action = matrix(vec![0.2], 1, 1).require_grad();
        let utility = learner
            .action_utility((&obs).into(), action.clone())
            .unwrap();
        let gradient = action
            .grad(&utility.sum().backward())
            .unwrap()
            .into_scalar::<f32>();
        let finite_difference = (evaluate(0.201) - evaluate(0.199)) / 0.002;
        assert!(gradient > 0.);
        assert!(
            (gradient - finite_difference).abs() < 3e-4,
            "{gradient} != {finite_difference}"
        );
    }
}

#[test]
fn softmax_probabilities_handle_extreme_and_offset_logits() {
    let support = CategoricalSupport::uniform(-2., 2., 3).unwrap();
    let logits = matrix(vec![1000., 0., -1000., 1000., 1000., 1000.], 2, 3);
    let probabilities = to_vec(support.probabilities(logits).unwrap()).unwrap();
    assert_eq!(&probabilities[..3], &[1., 0., 0.]);
    for mass in &probabilities[3..] {
        assert!((mass - 1. / 3.).abs() < 1e-7);
    }
    assert!(
        support
            .actor_utility(matrix(vec![1., 2.], 1, 2), &Risk::Expected)
            .is_err()
    );
}

#[test]
fn categorical_variance_and_gradient_are_stable_under_a_common_return_offset() {
    let evaluate = |offset| {
        let support = CategoricalSupport::uniform(offset, offset + 1., 2).unwrap();
        let logits = matrix(vec![0., 0.], 1, 2).require_grad();
        let utility = support
            .actor_utility(logits.clone(), &Risk::MeanVariance { penalty: 2. })
            .unwrap();
        let gradient = to_vec(logits.grad(&utility.clone().backward()).unwrap()).unwrap();
        (utility.into_scalar::<f32>() - offset, gradient)
    };
    let unshifted = evaluate(0.);
    let shifted = evaluate(10_000.);
    assert_eq!(unshifted, (0., vec![-0.25, 0.25]));
    assert_eq!(shifted, unshifted);
}

#[test]
fn bounded_neural_rare_loss_learning_moves_toward_the_analytic_mean() {
    let start = std::time::Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let mut critic = QValue::new(1, 1, 51, &[], &mut rng).unwrap();
    critic.network.layers[0].weight = Param::from_tensor(matrix(vec![0.; 102], 2, 51));
    critic.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.; 51])));
    let support = CategoricalSupport::uniform(-500., 0., 51).unwrap();
    let mut config = CategoricalDpgConfig::new(support);
    config.training.critic.learning_rate = 0.03;
    config.training.tau = 1.;
    config.training.target_noise = 0.;
    config.training.policy_delay = 20;
    let mut learner = CategoricalDpg::new(policy, critic, config, 17).unwrap();
    let observation = Observation::new(vec![0.]);
    // Fixed empirical reward frequencies for q = .02 and a one-step discount of .9.
    let batch = ReplayBatch {
        items: (0..50)
            .map(|i| ReplayItem {
                transition: Transition {
                    observation: observation.clone(),
                    action: Action::Continuous(vec![0.]),
                    reward: if i == 0 { -50_f32 } else { 0_f32 }.into(),
                    next_observation: observation.clone(),
                    terminated: false,
                    truncated: false,
                },
                discount: 0.9,
            })
            .collect(),
    };
    let mean = |learner: &CategoricalDpg| {
        learner
            .action_utility(
                std::slice::from_ref(&observation).into(),
                matrix(vec![0.], 1, 1),
            )
            .unwrap()
            .into_scalar::<f32>()
    };
    let initial = mean(&learner);
    for _ in 0..120 {
        learner.update(&batch).unwrap();
    }
    let final_mean = mean(&learner);
    let exact = -10.;
    println!(
        "rare-loss neural integration (120 x 50 replay rows): mean {initial:.3} -> {final_mean:.3}, exact {exact}, {:.3}s",
        start.elapsed().as_secs_f64()
    );
    assert!((final_mean - exact).abs() < (initial - exact).abs() * 0.7);
}

#[test]
fn checkpoint_restores_probabilities_and_the_next_update() {
    let support = CategoricalSupport::uniform(-2., 2., 3).unwrap();
    let mut original = learner(Risk::Cvar { alpha: 0.4 }, support.clone());
    let batch = replay();
    original.update(&batch).unwrap();
    let record = original.checkpoint().unwrap();
    let mut restored = learner(Risk::Expected, support);
    restored.restore(&record).unwrap();

    let obs = [Observation::new(vec![0.2])];
    let probabilities = |l: &CategoricalDpg| {
        to_vec(
            l.probabilities((&obs).into(), matrix(vec![-0.3], 1, 1))
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(probabilities(&original), probabilities(&restored));
    assert!(matches!(restored.risk(), Risk::Cvar { alpha } if *alpha == 0.4));
    assert_eq!(
        original.update(&batch).unwrap().metrics,
        restored.update(&batch).unwrap().metrics
    );
    assert_eq!(probabilities(&original), probabilities(&restored));
    let mut mismatched = learner(
        Risk::Expected,
        CategoricalSupport::uniform(-3., 3., 3).unwrap(),
    );
    assert!(mismatched.restore(&record).is_err());
}
