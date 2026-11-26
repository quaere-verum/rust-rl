#![cfg(feature = "diagnostics")]
#[path = "support/controlled/environment.rs"]
mod environment;
use environment::ContextLottery;
use rl::{diagnostics::evaluation::BranchingEnv, prelude::*};
fn action(a: f32) -> Action {
    Action::Continuous(vec![a])
}
#[test]
fn reward_is_terminal_and_later_actions_do_not_change_commitment() {
    for context in [-1, 1] {
        let mut a = ContextLottery::new(context, 7);
        assert_eq!(
            a.reset(None).unwrap().observation.dense_values(),
            vec![0., 0., context as f32]
        );
        let first = a.step(&action(0.5)).unwrap();
        assert_eq!(first.reward.total(), 0.);
        assert!(!first.ended());
        assert_eq!(
            first.observation.dense_values(),
            vec![0.25, 0.5, context as f32]
        );
        let mut b = a.clone();
        for time in 2..=4 {
            let sa = a.step(&action(0.)).unwrap();
            let sb = b.step(&action(1.)).unwrap();
            assert_eq!(sa.reward.total(), sb.reward.total());
            assert_eq!(sa.observation, sb.observation);
            assert_eq!(sa.terminated, time == 4);
            assert!(!sa.truncated);
            if time < 4 {
                assert_eq!(sa.reward.total(), 0.);
            } else {
                let high = if context == 1 { 2. } else { 1.25 };
                assert!(
                    (-1.5..=-0.5).contains(&sa.reward.total())
                        || (high - 0.5..=high + 0.5).contains(&sa.reward.total())
                );
            }
        }
        assert!(a.step(&action(0.)).is_err());
    }
}
#[test]
fn branching_reseeds_future_without_resetting_physical_state() {
    let mut source = ContextLottery::new(-1, 99);
    source.reset(None).unwrap();
    source.step(&action(0.5)).unwrap();
    let mut rewards = vec![];
    for seed in 100..116 {
        let mut branch = source.branch(seed).unwrap();
        assert_eq!(branch.branch_state().unwrap(), vec![1., 0.5, -1.]);
        for _ in 0..2 {
            branch.step(&action(0.)).unwrap();
        }
        rewards.push(branch.step(&action(0.)).unwrap().reward.total());
    }
    assert!(
        rewards.windows(2).any(|p| p[0] != p[1]),
        "future seeds must replace RNG"
    );
    let mut a = source.branch(100).unwrap();
    let mut b = source.branch(100).unwrap();
    for _ in 0..3 {
        assert_eq!(
            a.step(&action(0.)).unwrap().reward.total(),
            b.step(&action(1.)).unwrap().reward.total()
        );
    }
}
