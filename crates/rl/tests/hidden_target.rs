use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{core::checkpoint::Snapshot, envs::hidden_target::HiddenTarget, prelude::*};

fn action(value: f32) -> Action {
    Action::Continuous(vec![value])
}
fn decide(env: &mut HiddenTarget, value: f32) -> Step {
    for _ in 0..env.delay() {
        let step = env.step(&action(0.)).unwrap();
        assert_eq!(step.reward.total(), 0.);
        assert!(!step.ended());
    }
    env.step(&action(value)).unwrap()
}

#[test]
fn seeded_reset_is_balanced_independent_and_none_uses_only_owned_rng() {
    let mut a = HiddenTarget::new(4, Some(2), false, 19).unwrap();
    let mut b = HiddenTarget::new(4, Some(2), false, 999).unwrap();
    for _ in 0..7 {
        a.reset(None).unwrap();
    }
    let mut negative = 0;
    for seed in 1000..1200 {
        let expected = if seed % 2 == 0 { -1. } else { 1. };
        assert_eq!(
            a.reset(Some(seed)).unwrap().observation,
            b.reset(Some(seed)).unwrap().observation
        );
        assert_eq!(
            a.reset(Some(seed)).unwrap().observation.dense_values()[0],
            expected
        );
        negative += usize::from(expected < 0.);
    }
    assert_eq!(negative, 100);
    assert_eq!(
        a.reset(Some(47)).unwrap().observation,
        b.reset(Some(47)).unwrap().observation
    );
    let mut reference = ChaCha8Rng::seed_from_u64(47);
    let mut unrelated = ChaCha8Rng::seed_from_u64(100);
    for _ in 0..64 {
        let expected = if reference.random_range(0..2usize) == 0 {
            -1.
        } else {
            1.
        };
        let _ = unrelated.random::<f64>();
        let observation = a.reset(None).unwrap().observation;
        assert_eq!(observation, b.reset(None).unwrap().observation);
        assert_eq!(observation.dense_values()[0], expected);
        let copy: HiddenTarget = serde_json::from_value(serde_json::to_value(&a).unwrap()).unwrap();
        copy.validate_snapshot().unwrap();
        assert_eq!(
            serde_json::to_value(&copy).unwrap(),
            serde_json::to_value(&a).unwrap()
        );
    }
}

#[test]
fn oracle_constant_and_uniform_rewards_match_independent_quadratic_math() {
    for delay in [2, 8, 128] {
        for seed in [1000, 1001] {
            let mut env = HiddenTarget::new(delay, None, false, 1).unwrap();
            let target = env.reset(Some(seed)).unwrap().observation.dense_values()[0];
            let oracle = decide(&mut env, target);
            assert!(oracle.terminated);
            assert!(!oracle.truncated);
            assert_eq!(oracle.reward.total(), 0.);
        }
    }
    for constant in [-2., -1.25, -0.5, 0., 0.5, 1.25, 2.] {
        let mut mean = 0f64;
        for seed in [0, 1] {
            let mut env = HiddenTarget::new(3, Some(1), false, 99).unwrap();
            env.reset(Some(seed)).unwrap();
            mean += decide(&mut env, constant).reward.total() as f64 / 2.;
        }
        assert!((mean + 1. + (constant as f64).powi(2)).abs() < 1e-12);
        assert!(mean <= -1.);
    }
    // Simpson's rule integrates this quadratic exactly under Uniform[-2,2].
    let mut uniform_mean = 0.;
    for (sample, weight) in [(-2., 1.), (0., 4.), (2., 1.)] {
        for seed in [0, 1] {
            let mut env = HiddenTarget::new(2, None, false, 1).unwrap();
            env.reset(Some(seed)).unwrap();
            uniform_mean += weight * decide(&mut env, sample).reward.total() as f64 / 12.;
        }
    }
    assert!((uniform_mean + 7. / 3.).abs() < 1e-12);
}

#[test]
fn earlier_actions_and_marker_do_not_store_or_reveal_the_hidden_target() {
    for fully_observed in [false, true] {
        let mut left = HiddenTarget::new(6, Some(3), fully_observed, 9).unwrap();
        let mut right = HiddenTarget::new(6, Some(3), fully_observed, 9).unwrap();
        let initial_left = left.reset(Some(0)).unwrap().observation;
        let initial_right = right.reset(Some(1)).unwrap().observation;
        assert_eq!(initial_left.dense_values(), vec![-1., 1., 0., 0., 0.]);
        assert_eq!(initial_right.dense_values(), vec![1., 1., 0., 0., 0.]);
        assert!(initial_left.dense_mask().is_none());
        for tick in 1..=6 {
            let a = left.step(&action(-2.)).unwrap();
            let b = right.step(&action(2.)).unwrap();
            assert!(!a.ended() && !b.ended());
            assert_eq!(a.reward.total(), 0.);
            assert_eq!(b.reward.total(), 0.);
            assert_eq!(a.observation.dense_values()[1], 0.);
            assert_eq!(a.observation.dense_values()[2], f32::from(tick == 6));
            assert_eq!(a.observation.dense_values()[3], f32::from(tick == 3));
            assert_eq!(a.observation.dense_values()[4], tick as f32 / 6.);
            assert_eq!(a.observation == b.observation, !fully_observed);
            assert_eq!(
                a.observation.dense_values()[0],
                if fully_observed { -1. } else { 0. }
            );
            assert_eq!(
                b.observation.dense_values()[0],
                if fully_observed { 1. } else { 0. }
            );
        }
        assert_eq!(left.step(&action(0.)).unwrap().reward.total(), -1.);
        assert_eq!(right.step(&action(0.)).unwrap().reward.total(), -1.);
    }
}

#[test]
fn invalid_actions_and_terminal_steps_are_pure_errors_and_rewards_are_finite() {
    let mut env = HiddenTarget::new(2, Some(1), false, 19).unwrap();
    let initial = serde_json::to_value(&env).unwrap();
    assert!(env.step(&action(0.)).is_err());
    assert_eq!(initial, serde_json::to_value(&env).unwrap());
    env.reset(Some(0)).unwrap();
    for invalid in [
        Action::Discrete(0),
        Action::Continuous(vec![]),
        Action::Continuous(vec![0., 1.]),
        action(f32::NAN),
        action(f32::INFINITY),
        action(f32::NEG_INFINITY),
        action(-2.01),
        action(2.01),
    ] {
        let before = serde_json::to_value(&env).unwrap();
        assert!(env.step(&invalid).is_err());
        assert_eq!(before, serde_json::to_value(&env).unwrap());
    }
    assert_eq!(
        env.action_space(),
        &ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.]
        }
    );
    let final_step = decide(&mut env, 2.);
    assert_eq!(final_step.reward.total(), -9.);
    assert!(final_step.reward.total().is_finite());
    assert!(final_step.terminated && !final_step.truncated);
    let ended = serde_json::to_value(&env).unwrap();
    assert!(env.step(&action(0.)).is_err());
    assert_eq!(ended, serde_json::to_value(&env).unwrap());
    env.reset(Some(1)).unwrap();
    assert_eq!(decide(&mut env, -2.).reward.total(), -9.);
}

#[test]
fn snapshots_preserve_mid_episode_boundary_and_future_reset_streams() {
    let mut original = HiddenTarget::new(8, Some(4), false, 29).unwrap();
    original.reset(None).unwrap();
    for _ in 0..4 {
        original.step(&action(1.5)).unwrap();
    }
    let saved = serde_json::to_value(&original).unwrap();
    let mut restored: HiddenTarget = serde_json::from_value(saved.clone()).unwrap();
    restored.validate_snapshot().unwrap();
    for value in [-1.7, 0.25, 1., -0.5, 0.7] {
        let a = original.step(&action(value)).unwrap();
        let b = restored.step(&action(value)).unwrap();
        assert_eq!(a.observation, b.observation);
        assert_eq!(a.reward.total().to_bits(), b.reward.total().to_bits());
        assert_eq!(a.terminated, b.terminated);
        assert_eq!(
            serde_json::to_value(&original).unwrap(),
            serde_json::to_value(&restored).unwrap()
        );
    }
    for _ in 0..32 {
        assert_eq!(
            original.reset(None).unwrap().observation,
            restored.reset(None).unwrap().observation
        );
    }
    for corruption in 0..9 {
        let mut invalid = saved.clone();
        match corruption {
            0 => invalid["version"] = 2.into(),
            1 => invalid["delay"] = 1.into(),
            2 => invalid["delay"] = 129.into(),
            3 => invalid["tick"] = 9.into(),
            4 => invalid["target"] = 0.into(),
            5 => invalid["target"] = 2.into(),
            6 => invalid["ended"] = true.into(),
            7 => invalid["marker"] = 0.into(),
            _ => invalid["space"]["Box"]["high"] = serde_json::json!([3.]),
        }
        let invalid: HiddenTarget = serde_json::from_value(invalid).unwrap();
        assert!(invalid.validate_snapshot().is_err());
    }
}

#[test]
fn constructor_bounds_match_the_declared_controlled_horizons() {
    for delay in [0, 1, 129, usize::MAX] {
        assert!(HiddenTarget::new(delay, None, false, 1).is_err());
    }
    for marker in [0, 8, 9, usize::MAX] {
        assert!(HiddenTarget::new(8, Some(marker), false, 1).is_err());
    }
    HiddenTarget::new(2, Some(1), true, 1)
        .unwrap()
        .validate_snapshot()
        .unwrap();
    HiddenTarget::new(128, Some(127), false, 1)
        .unwrap()
        .validate_snapshot()
        .unwrap();
}
