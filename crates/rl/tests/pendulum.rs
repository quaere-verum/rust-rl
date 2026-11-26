use rl::{
    core::{
        action::Action,
        checkpoint::Snapshot,
        env::{ClipAction, Env},
    },
    envs::pendulum::Pendulum,
};

#[test]
fn reset_and_lifecycle_use_owned_seeded_stream() {
    let mut env = Pendulum::new(4);
    assert!(env.step(&Action::Continuous(vec![0.])).is_err());
    let first = env.reset(Some(71)).unwrap().observation;
    assert_eq!(first, env.reset(Some(71)).unwrap().observation);
    assert_ne!(first, env.reset(None).unwrap().observation);
    assert_eq!(first.dense_values().len(), 3);
    assert!(first.dense_values()[2].abs() <= 1.);
}

#[test]
fn reward_uses_pre_transition_state_and_applied_torque() {
    let mut env = Pendulum::from_state(0, [0., 0.]).unwrap();
    let step = env.step(&Action::Continuous(vec![2.])).unwrap();
    assert!((step.reward.total() + 0.004).abs() < 1e-7);
    assert!((step.observation.dense_values()[0] - (0.015_f64.cos() as f32)).abs() < 1e-7);
    assert!((step.observation.dense_values()[1] - (0.015_f64.sin() as f32)).abs() < 1e-7);
    assert!((step.observation.dense_values()[2] - 0.3).abs() < 1e-7);
    assert!(!step.ended());
}

#[test]
fn raw_gaussian_actions_are_clipped_by_adapter() {
    let mut physical = Pendulum::from_state(1, [0.7, 7.99]).unwrap();
    let before = serde_json::to_vec(&physical).unwrap();
    assert!(physical.step(&Action::Continuous(vec![7.])).is_err());
    assert_eq!(before, serde_json::to_vec(&physical).unwrap());
    let mut clipped = ClipAction::new(physical.clone()).unwrap();
    let a = clipped.step(&Action::Continuous(vec![7.])).unwrap();
    let b = physical.step(&Action::Continuous(vec![2.])).unwrap();
    assert_eq!(a.observation, b.observation);
    assert_eq!(a.reward, b.reward);
    assert_eq!(a.observation.dense_values()[2], 8.);
}

#[test]
fn snapshot_roundtrips_exact_state_and_random_stream() {
    let mut env = Pendulum::from_state(15, [2.123456789012345, -0.123456789012345]).unwrap();
    let bytes = serde_json::to_vec(&env).unwrap();
    let mut restored: Pendulum = serde_json::from_slice(&bytes).unwrap();
    restored.validate_snapshot().unwrap();
    for action in [0.2, -2., 1.2345, 0.] {
        let a = env.step(&Action::Continuous(vec![action])).unwrap();
        let b = restored.step(&Action::Continuous(vec![action])).unwrap();
        assert_eq!(a.observation, b.observation);
        assert_eq!(a.reward, b.reward);
    }
    assert_eq!(
        serde_json::to_vec(&env).unwrap(),
        serde_json::to_vec(&restored).unwrap()
    );
    assert_eq!(
        env.reset(None).unwrap().observation,
        restored.reset(None).unwrap().observation
    );
    let mut corrupt = serde_json::to_value(&env).unwrap();
    corrupt["state"][0] = serde_json::json!("NaN");
    assert!(serde_json::from_value::<Pendulum>(corrupt).is_err());
}

#[test]
fn invalid_state_and_actions_are_rejected_without_mutation() {
    assert!(Pendulum::from_state(0, [f64::NAN, 0.]).is_err());
    assert!(Pendulum::from_state(0, [0., 8.01]).is_err());
    let mut env = Pendulum::from_state(0, [0.5, 0.1]).unwrap();
    let before = serde_json::to_vec(&env).unwrap();
    for action in [
        Action::Discrete(0),
        Action::Continuous(vec![]),
        Action::Continuous(vec![f32::NAN]),
    ] {
        assert!(env.step(&action).is_err());
        assert_eq!(before, serde_json::to_vec(&env).unwrap());
    }
}

#[test]
fn reset_distribution_is_uniform_with_independent_coordinates() {
    let mut env = Pendulum::new(47);
    let mut sums = [0.0_f64; 2];
    let mut squares = [0.0_f64; 2];
    for _ in 0..4096 {
        let o = env.reset(None).unwrap().observation;
        let x = [
            f64::from(o.dense_values()[1]).atan2(f64::from(o.dense_values()[0])),
            f64::from(o.dense_values()[2]),
        ];
        for i in 0..2 {
            sums[i] += x[i];
            squares[i] += x[i] * x[i];
        }
    }
    for (i, bound) in [std::f64::consts::PI, 1.].into_iter().enumerate() {
        let mean = sums[i] / 4096.;
        let variance = squares[i] / 4096. - mean * mean;
        assert!(mean.abs() < bound * 0.04);
        assert!((variance - bound * bound / 3.).abs() < bound * bound * 0.03);
    }
}

#[test]
#[cfg(feature = "neural")]
fn gaussian_ppo_collects_clipped_actions_and_resets_at_v1_limit() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::prelude::*;
    let environments = (0..4)
        .map(|seed| ClipAction::new(Pendulum::v1(seed).unwrap()).unwrap())
        .collect();
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(environments).unwrap(), 91).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 16,
            normalize_advantages: false,
            ..Default::default()
        },
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(92);
    let models = ActorCritic {
        policy: Policy::new(3, ActionSpace::Unbounded { dimensions: 1 }, &[16], &mut rng).unwrap(),
        value: Value::new(3, &[16], &mut rng).unwrap(),
    };
    let optimizer = OptimizerConfig {
        max_grad_norm: None,
        ..Default::default()
    };
    let mut ppo = Ppo::new_with_joint_gradient_clipping(
        models,
        PpoConfig {
            policy: optimizer.clone(),
            value: optimizer,
            epochs: 1,
            minibatch_size: 32,
            target_kl: None,
            ..Default::default()
        },
        93,
        0.5,
        0.5,
    )
    .unwrap()
    .with_minibatch_advantage_normalization(true);
    let summary = trainer
        .train(&mut ppo, Budget::vector_steps(208), None)
        .unwrap();
    assert_eq!(summary.counters.env_steps, 832);
    assert_eq!(summary.counters.episodes, 4);
    assert_eq!(summary.counters.updates, 26);
    assert!(
        summary
            .last_update
            .unwrap()
            .metrics
            .values()
            .all(|x| x.is_finite())
    );
}
