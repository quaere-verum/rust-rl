use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{core::transition::Transition, data::replay::ReplayItem, prelude::*};
#[test]
fn reused_sampling_preserves_the_uniform_draws_and_fields() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    ] {
        let mut replay = ReplayBuffer::new(8, 1, space.clone()).unwrap();
        for r in 0..5 {
            replay
                .push(&ReplayItem {
                    transition: Transition {
                        observation: Observation::new(vec![r as f32]),
                        action: if matches!(space, ActionSpace::Discrete { .. }) {
                            Action::Discrete(1)
                        } else {
                            Action::Continuous(vec![0.5])
                        },
                        reward: (r as f32).into(),
                        next_observation: Observation::new(vec![r as f32 + 1.]),
                        terminated: true,
                        truncated: false,
                    },
                    discount: 0.,
                })
                .unwrap();
        }
        let mut a = ChaCha8Rng::seed_from_u64(3);
        let mut b = a.clone();
        let mut scratch = ReplayBatch { items: vec![] };
        for n in [64, 64, 16, 32] {
            let expected = replay.sample(n, &mut a).unwrap();
            replay.sample_into(n, &mut b, &mut scratch).unwrap();
            assert_eq!(
                serde_json::to_string(&expected).unwrap(),
                serde_json::to_string(&scratch).unwrap()
            );
        }
    }
}
