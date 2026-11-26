#![cfg(feature = "neural")]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    imitation::{BehaviorCloning, CloningConfig, Demonstrations},
    prelude::*,
};
#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

fn policy() -> Policy {
    let mut p = Policy::new(
        1,
        ActionSpace::Discrete { count: 3 },
        &[],
        &mut ChaCha8Rng::seed_from_u64(71),
    )
    .unwrap();
    p.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![0.; 3], 1, 3));
    p.network.layers[0].bias = Some(Param::from_tensor(rl::nn::vector(&[0.; 3])));
    p
}
fn data(xs: &[f32]) -> Demonstrations {
    Demonstrations::new(
        ActionSpace::Discrete { count: 3 },
        xs.iter()
            .map(|&x| Observation::dense(vec![x], Some(vec![true, true, false])))
            .collect(),
        xs.iter()
            .map(|&x| Action::Discrete(usize::from(x > 0.)))
            .collect(),
    )
    .unwrap()
}
#[test]
fn masked_demonstrations_fit_held_out_inputs_and_resume_shuffle_and_adam() {
    let train = data(&[-1., 1., -2., 2., -0.8, 0.8]);
    let held_out = data(&[-0.7, 0.7]);
    let config = CloningConfig {
        optimizer: OptimizerConfig {
            learning_rate: 0.03,
            ..Default::default()
        },
        batch_size: 4,
    };
    let mut learner = BehaviorCloning::new(policy(), config.clone(), 72).unwrap();
    assert!((learner.evaluate(&held_out).unwrap() - 2_f64.ln()).abs() < 1e-6);
    for _ in 0..40 {
        learner.fit_epoch(&train).unwrap();
    }
    // Initial 40-epoch attempt reached NLL .13868 with the correct linear rule.
    // One bounded extension retains the original .12 acceptance threshold.
    assert!(learner.evaluate(&held_out).unwrap() < 2_f64.ln());
    for _ in 0..40 {
        learner.fit_epoch(&train).unwrap();
    }
    let validation = learner.evaluate(&held_out).unwrap();
    assert!(
        validation < 0.12,
        "held-out NLL={validation}; weights={:?}",
        rl::nn::to_vec(learner.policy().network.layers[0].weight.val()).unwrap()
    );
    let record = learner.checkpoint().unwrap();
    let mut restored = BehaviorCloning::new(policy(), config, 999).unwrap();
    restored.restore(&record).unwrap();
    assert_eq!(
        learner.fit_epoch(&train).unwrap().metrics,
        restored.fit_epoch(&train).unwrap().metrics
    );
    assert_eq!(
        checkpoint_eq::normalized(&learner.checkpoint().unwrap()),
        checkpoint_eq::normalized(&restored.checkpoint().unwrap())
    );
    let actor = rl::nn::policy::PolicyActor::new(learner.policy()).unwrap();
    let actions = actor
        .act(
            &[Observation::dense(vec![0.7], Some(vec![true, true, false]))],
            ActMode::Deterministic,
            &mut ChaCha8Rng::seed_from_u64(8),
        )
        .unwrap();
    assert_eq!(actions, [Action::Discrete(1)]);
    let dir = tempfile::tempdir().unwrap();
    rl::train::checkpoint::export_policy(
        dir.path().join("actor"),
        &learner.policy(),
        actor.action_space(),
    )
    .unwrap();
    let loaded = rl::train::checkpoint::load_policy(
        dir.path().join("actor"),
        &policy(),
        actor.action_space(),
    )
    .unwrap();
    use burn::module::Module;
    let value = Value::new(1, &[], &mut ChaCha8Rng::seed_from_u64(19)).unwrap();
    let _fine_tuning = Ppo::new(
        ActorCritic {
            policy: loaded.unfreeze().train(),
            value,
        },
        Default::default(),
        20,
    )
    .unwrap();
}
#[test]
fn continuous_likelihood_inverts_affine_tanh_without_clipping_demonstrations() {
    let space = ActionSpace::Box {
        low: vec![-2., 1.],
        high: vec![4., 5.],
    };
    let mut p =
        GaussianPolicy::new(1, space.clone(), &[], &mut ChaCha8Rng::seed_from_u64(91)).unwrap();
    p.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![0.; 4], 1, 4));
    p.network.layers[0].bias = Some(Param::from_tensor(rl::nn::vector(&[0.; 4])));
    let actions = vec![Action::Continuous(vec![
        1. + 3. * 0.2_f32.tanh(),
        3. + 2. * (-0.3_f32).tanh(),
    ])];
    let data =
        Demonstrations::new(space.clone(), vec![Observation::new(vec![0.])], actions).unwrap();
    let learner = BehaviorCloning::new(p, Default::default(), 92).unwrap();
    let expected: f64 = [(0.2_f64, 3_f64), (-0.3, 2.)]
        .into_iter()
        .map(|(z, s)| {
            0.5 * z * z
                + 0.5 * (2. * std::f64::consts::PI).ln()
                + s.ln()
                + (1. - z.tanh().powi(2)).ln()
        })
        .sum();
    assert!((learner.evaluate(&data).unwrap() - expected).abs() < 2e-6);
    assert!(
        Demonstrations::new(
            space.clone(),
            vec![Observation::new(vec![0.])],
            vec![Action::Continuous(vec![4., 3.])]
        )
        .is_err()
    );
    assert!(
        Demonstrations::new(
            space,
            vec![Observation::new(vec![0.])],
            vec![Action::Continuous(vec![4.1, 3.])]
        )
        .is_err()
    );
}
#[test]
fn forbidden_demonstrated_actions_and_mismatched_spaces_reject() {
    assert!(
        Demonstrations::new(
            ActionSpace::Discrete { count: 2 },
            vec![Observation::dense(vec![1.], Some(vec![true, false]))],
            vec![Action::Discrete(1)]
        )
        .is_err()
    );
    let mut learner = BehaviorCloning::new(policy(), Default::default(), 1).unwrap();
    let data = Demonstrations::new(
        ActionSpace::Discrete { count: 2 },
        vec![Observation::new(vec![1.])],
        vec![Action::Discrete(1)],
    )
    .unwrap();
    assert!(learner.fit_epoch(&data).is_err());
}

#[test]
fn strict_interior_demonstrations_remain_invertible_near_an_asymmetric_endpoint() {
    let data = Demonstrations::new(
        ActionSpace::Box {
            low: vec![0.],
            high: vec![1.],
        },
        vec![Observation::new(vec![0.])],
        vec![Action::Continuous(vec![1e-30])],
    )
    .unwrap();
    let policy = GaussianPolicy::new(
        1,
        data.action_space().clone(),
        &[],
        &mut ChaCha8Rng::seed_from_u64(99),
    )
    .unwrap();
    assert!(
        BehaviorCloning::new(policy, Default::default(), 100)
            .unwrap()
            .evaluate(&data)
            .unwrap()
            .is_finite()
    );
}

#[test]
fn continuous_policy_fits_held_out_actions_and_malformed_restore_is_atomic() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let make_data = |xs: &[f32]| {
        Demonstrations::new(
            space.clone(),
            xs.iter().map(|&x| Observation::new(vec![x])).collect(),
            xs.iter()
                .map(|&x| Action::Continuous(vec![0.5 * x]))
                .collect(),
        )
        .unwrap()
    };
    let train = make_data(&[-1., 1., -2., 2.]);
    let held_out = make_data(&[-0.75, 0.75]);
    let p = GaussianPolicy::new(1, space, &[], &mut ChaCha8Rng::seed_from_u64(107)).unwrap();
    let mut learner = BehaviorCloning::new(
        p,
        CloningConfig {
            optimizer: OptimizerConfig {
                learning_rate: 0.02,
                ..Default::default()
            },
            batch_size: 4,
        },
        108,
    )
    .unwrap();
    let before = learner.evaluate(&held_out).unwrap();
    for _ in 0..100 {
        learner.fit_epoch(&train).unwrap();
    }
    assert!(learner.evaluate(&held_out).unwrap() < before - 0.5);
    let record = learner.checkpoint().unwrap();
    for case in 0..3 {
        let mut bad = record.clone();
        match case {
            0 => bad.metadata["optimizer_owners"] = serde_json::json!([]),
            1 => bad.metadata["trainability"][0] = serde_json::json!(false),
            _ => bad.metadata["epochs"] = serde_json::json!(1000),
        }
        assert!(learner.restore(&bad).is_err());
        assert_eq!(
            checkpoint_eq::normalized(&learner.checkpoint().unwrap()),
            checkpoint_eq::normalized(&record)
        );
    }
}

#[test]
fn checkpoint_rejects_a_template_with_different_frozen_layers() {
    let mut p = policy();
    p.network.layers[0].bias = p.network.layers[0]
        .bias
        .take()
        .map(|b| b.set_require_grad(false));
    let mut source = BehaviorCloning::new(p, Default::default(), 109).unwrap();
    source.fit_epoch(&data(&[-1., 1.])).unwrap();
    let record = source.checkpoint().unwrap();
    let mut different = BehaviorCloning::new(policy(), Default::default(), 110).unwrap();
    assert!(different.restore(&record).is_err());
    source.restore(&record).unwrap();
}
