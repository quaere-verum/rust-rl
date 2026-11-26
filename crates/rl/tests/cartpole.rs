use rl::{
    core::{
        action::Action,
        checkpoint::Snapshot,
        env::{Env, TimeLimit},
    },
    envs::cartpole::CartPole,
};

#[test]
fn seeded_reset_is_reproducible_and_steps_require_a_live_episode() {
    let mut env = CartPole::new(4);
    assert!(env.step(&Action::Discrete(0)).is_err());
    let first = env.reset(Some(71)).unwrap().observation;
    assert_eq!(first.dense_values().len(), 4);
    assert!(
        first
            .dense_values()
            .iter()
            .all(|x| *x >= -0.05 && *x <= 0.05)
    );
    let second = env.reset(Some(71)).unwrap().observation;
    assert_eq!(first, second);
    let third = env.reset(None).unwrap().observation;
    assert_ne!(second, third);
}

#[test]
fn invalid_actions_do_not_change_state_or_random_stream() {
    let mut env = CartPole::from_state(9, [0.0, 0.0, 0.0, 0.0]).unwrap();
    let before = serde_json::to_vec(&env).unwrap();
    assert!(env.step(&Action::Discrete(2)).is_err());
    assert!(env.step(&Action::Continuous(vec![0.0])).is_err());
    assert_eq!(before, serde_json::to_vec(&env).unwrap());
    let right = env.step(&Action::Discrete(1)).unwrap();
    assert_eq!(right.reward.total(), 1.0);
    assert!(!right.ended());
    assert!(right.observation.dense_values()[1] > 0.0);
    assert!(right.observation.dense_values()[3] < 0.0);
}

#[test]
fn terminal_reward_and_boundary_priority() {
    let mut env = CartPole::from_state(1, [2.399, 0.1, 0., 0.]).unwrap();
    let step = env.step(&Action::Discrete(1)).unwrap();
    assert!(step.terminated && !step.truncated);
    assert_eq!(step.reward.total(), 1.0);
    assert!(env.step(&Action::Discrete(1)).is_err());

    let mut timed = TimeLimit::new(CartPole::new(1), 1).unwrap();
    timed.reset(Some(1)).unwrap();
    timed.inner = CartPole::from_state(1, [2.399, 0.1, 0., 0.]).unwrap();
    let step = timed.step(&Action::Discrete(1)).unwrap();
    assert!(step.terminated && !step.truncated);
    assert!(timed.step(&Action::Discrete(1)).is_err());
}

#[test]
fn snapshot_preserves_exact_physical_state_and_rng() {
    let mut env = CartPole::from_state(
        15,
        [
            0.03123456789012345,
            0.01234567890123456,
            -0.01876543210987654,
            0.,
        ],
    )
    .unwrap();
    let bytes = serde_json::to_vec(&env).unwrap();
    let mut restored: CartPole = serde_json::from_slice(&bytes).unwrap();
    restored.validate_snapshot().unwrap();
    for action in [0, 1, 1, 0, 1] {
        let a = env.step(&Action::Discrete(action)).unwrap();
        let b = restored.step(&Action::Discrete(action)).unwrap();
        assert_eq!(a.observation, b.observation);
        assert_eq!(
            (a.reward, a.terminated, a.truncated),
            (b.reward, b.terminated, b.truncated)
        );
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
    assert!(serde_json::from_value::<CartPole>(corrupt).is_err());
}

#[test]
fn controlled_state_rejects_invalid_live_states() {
    assert!(CartPole::from_state(0, [f64::NAN, 0., 0., 0.]).is_err());
    assert!(CartPole::from_state(0, [2.5, 0., 0., 0.]).is_err());
    assert!(CartPole::from_state(0, [0., 0., 0.21, 0.]).is_err());
}

#[test]
fn reset_samples_have_expected_uniform_moments() {
    let mut env = CartPole::new(47);
    let mut sums = [0.0_f64; 4];
    let mut squares = [0.0_f64; 4];
    for _ in 0..4096 {
        let observation = env.reset(None).unwrap().observation;
        for (i, &value) in observation.dense_values().iter().enumerate() {
            assert!((-0.05..=0.05).contains(&value));
            sums[i] += value as f64;
            squares[i] += (value as f64).powi(2);
        }
    }
    for i in 0..4 {
        let mean = sums[i] / 4096.;
        let variance = squares[i] / 4096. - mean.powi(2);
        assert!(mean.abs() < 0.002);
        assert!((variance - 0.05_f64.powi(2) / 3.).abs() < 0.00008);
    }
}

#[test]
fn numerical_overflow_does_not_mutate_the_episode() {
    let mut env = CartPole::from_state(9, [0., 0., 0.1, f32::MAX as f64]).unwrap();
    let before = serde_json::to_vec(&env).unwrap();
    assert!(env.step(&Action::Discrete(1)).is_err());
    assert_eq!(before, serde_json::to_vec(&env).unwrap());
}

#[test]
#[cfg(feature = "neural")]
fn categorical_ppo_can_collect_reset_and_update_on_cartpole() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::prelude::*;
    let environments = (0..4).map(|seed| CartPole::v1(seed).unwrap()).collect();
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
        policy: Policy::new(4, ActionSpace::Discrete { count: 2 }, &[16], &mut rng).unwrap(),
        value: Value::new(4, &[16], &mut rng).unwrap(),
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
        .train(&mut ppo, Budget::vector_steps(128), None)
        .unwrap();
    assert_eq!(summary.counters.env_steps, 512);
    assert!(summary.counters.episodes > 0);
    assert_eq!(summary.counters.updates, 16);
    assert!(
        summary
            .last_update
            .unwrap()
            .metrics
            .values()
            .all(|x| x.is_finite())
    );
}
