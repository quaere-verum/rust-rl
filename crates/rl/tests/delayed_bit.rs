use rl::{envs::delayed_bit::DelayedBit, prelude::*};

#[test]
fn exact_oracle_and_memoryless_bound_with_no_external_action_memory() {
    for marker in [None, Some(3)] {
        let mut memoryless_success = 0;
        for seed in 1000..1200 {
            let mut env = DelayedBit::new(6, marker, false, seed).unwrap();
            let cue = env.reset(Some(seed)).unwrap().observation.dense_values()[0];
            let correct = usize::from(cue > 0.);
            for t in 0..6 {
                assert!(env.step(&Action::Discrete(1)).is_err());
                let step = env.step(&Action::Discrete(0)).unwrap();
                assert!(!step.ended());
                assert_eq!(step.reward.total(), 0.);
                if t == 2 {
                    assert_eq!(
                        step.observation.dense_values()[3],
                        f32::from(marker.is_some())
                    );
                }
            }
            let step = env.step(&Action::Discrete(correct)).unwrap();
            assert!(step.terminated);
            assert_eq!(step.reward.total(), 1.);
            assert!(env.step(&Action::Discrete(correct)).is_err());
            memoryless_success += usize::from(correct == 0);
        }
        assert_eq!(memoryless_success, 100);
    }
}

#[test]
fn decision_observations_are_identical_except_in_fully_observed_control() {
    for full in [false, true] {
        let mut outcomes = Vec::new();
        for seed in [1000, 1001] {
            let mut env = DelayedBit::new(6, Some(3), full, seed).unwrap();
            env.reset(Some(seed)).unwrap();
            for _ in 0..5 {
                env.step(&Action::Discrete(0)).unwrap();
            }
            outcomes.push(env.step(&Action::Discrete(0)).unwrap().observation);
        }
        assert_eq!(outcomes[0] == outcomes[1], !full);
    }
}
