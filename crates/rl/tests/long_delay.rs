use rl::{envs::delayed_bit::DelayedBit, prelude::*};

#[test]
fn long_delay_has_one_real_episode_and_preserves_the_public_nonterminal_marker() {
    for seed in [2000, 2001] {
        let mut env = DelayedBit::new(8192, Some(4096), false, 11).unwrap();
        let initial = env.reset(Some(seed)).unwrap().observation;
        assert_eq!(
            initial.dense_values()[0],
            if seed % 2 == 0 { -1. } else { 1. }
        );
        for t in 0..8192 {
            let s = env.step(&Action::Discrete(0)).unwrap();
            assert!(!s.terminated && !s.truncated);
            assert_eq!(s.reward.total(), 0.);
            assert_eq!(s.observation.dense_values()[0], 0.);
            assert_eq!(s.observation.dense_values()[3], f32::from(t + 1 == 4096));
            if t == 4095 {
                let restored: DelayedBit =
                    serde_json::from_slice(&serde_json::to_vec(&env).unwrap()).unwrap();
                restored.validate_snapshot().unwrap();
            }
        }
        let result = env.step(&Action::Discrete((seed % 2) as usize)).unwrap();
        assert!(result.terminated);
        assert_eq!(result.reward.total(), 1.);
        assert!(env.step(&Action::Discrete(0)).is_err());
    }
    assert!(DelayedBit::new(16385, None, false, 1).is_err());
}
