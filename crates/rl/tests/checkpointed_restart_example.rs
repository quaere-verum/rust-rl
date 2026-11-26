#[path = "support/restart_branch.rs"]
mod restart_branch;
use restart_branch::Branch;
use rl::prelude::*;

#[test]
fn delayed_branch_has_known_optimum_and_terminal_only_rewards() {
    for seed in 0..20 {
        let mut env = Branch::new();
        env.reset(Some(seed)).unwrap();
        for _ in 0..Branch::DEPTH {
            let step = env.step(&Action::Discrete(1)).unwrap();
            assert_eq!(step.reward.total(), 0.);
            assert!(!step.ended());
        }
        let observation = env.observation();
        let correct = usize::from(observation.dense_values()[Branch::DEPTH + 1] > 0.);
        let mut wrong = env.clone();
        let terminal = env.step(&Action::Discrete(correct)).unwrap();
        assert_eq!(terminal.reward.total(), 1.);
        assert!(terminal.terminated);
        assert_eq!(
            wrong
                .step(&Action::Discrete(1 - correct))
                .unwrap()
                .reward
                .total(),
            0.
        );
        assert!(env.step(&Action::Discrete(0)).is_err());
        env.reset(None).unwrap();
        assert_eq!(env.step(&Action::Discrete(0)).unwrap().reward.total(), 0.05);
    }
}

#[test]
fn branch_clones_and_snapshots_preserve_exact_future_and_clock() {
    let mut env = Branch::new();
    env.reset(Some(91)).unwrap();
    env.step(&Action::Discrete(1)).unwrap();
    let mut copy: Branch = serde_json::from_slice(&serde_json::to_vec(&env).unwrap()).unwrap();
    copy.validate_snapshot().unwrap();
    for _ in 1..Branch::DEPTH {
        assert_eq!(
            env.step(&Action::Discrete(1)).unwrap().observation,
            copy.step(&Action::Discrete(1)).unwrap().observation
        );
    }
    assert_eq!(env.observation(), copy.observation());
}
