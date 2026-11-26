//! Allocation instrumentation lives only in this benchmark executable; the library forbids unsafe.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{core::transition::Transition, data::replay::ReplayItem, prelude::*};
use std::hint::black_box;

#[path = "support/counting_allocator.rs"]
mod counting_allocator;
use counting_allocator::snapshot;

fn allocations(f: impl FnOnce()) -> u64 {
    let (before, _) = snapshot();
    f();
    snapshot().0 - before
}
fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let item = ReplayItem {
        transition: Transition {
            observation: Observation::new(vec![1., 0.]),
            action: Action::Continuous(vec![0.]),
            reward: 1.0_f32.into(),
            next_observation: Observation::new(vec![0., 1.]),
            terminated: true,
            truncated: false,
        },
        discount: 0.,
    };
    let (_, before) = snapshot();
    let mut replay = ReplayBuffer::new(100_000, 2, space.clone()).unwrap();
    let bytes = snapshot().1 - before;
    for _ in 0..100_000 {
        replay.push(&item).unwrap();
    }
    let insert = allocations(|| {
        for _ in 0..1000 {
            replay.push(black_box(&item)).unwrap();
        }
    });
    black_box(replay.sample(64, &mut rng).unwrap());
    let sample = allocations(|| {
        black_box(replay.sample(64, &mut rng).unwrap());
    });
    let mut scratch = ReplayBatch::default();
    replay.sample_into(64, &mut rng, &mut scratch).unwrap();
    let reused = allocations(|| replay.sample_into(64, &mut rng, &mut scratch).unwrap());
    println!("Reused sample64 after warmup: {reused} allocations");
    let actor = ActorCritic {
        policy: Policy::new(2, space, &[32, 32], &mut rng).unwrap(),
        value: Value::new(2, &[32], &mut rng).unwrap(),
    };
    let obs = vec![Observation::new(vec![1., 0.]); 64];
    for _ in 0..10 {
        black_box(actor.act(&obs, ActMode::Deterministic, &mut rng).unwrap());
    }
    let inference = allocations(|| {
        black_box(actor.act(&obs, ActMode::Deterministic, &mut rng).unwrap());
    });
    let mut behavior = RolloutBehavior::default();
    actor
        .sample_rollout_into((&obs).into(), &mut rng, &mut behavior)
        .unwrap();
    let behavior_allocations = allocations(|| {
        actor
            .sample_rollout_into((&obs).into(), &mut rng, &mut behavior)
            .unwrap();
    });
    println!(
        "Replay capacity=100000, obs=2, action=1: live heap allocation bytes={bytes}; 1000 inserts={insert} allocations; sample64={sample} allocations; actor64 action-only after warmup={inference} allocations; actor64 reused behavior plus value={behavior_allocations} allocations"
    );
}
