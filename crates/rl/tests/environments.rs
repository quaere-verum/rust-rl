use rl::{
    core::{action::Action, env::Env},
    envs::lqg::{Lqg, LqgConfig},
};
#[test]
fn one_step_riccati_and_terminal_cost_match_direct_minimization() {
    let c = LqgConfig::scalar(1, 1., 1., 1., 1., 1., 2., 0., 10.).unwrap();
    let gains = c.riccati().unwrap();
    assert!((gains.gains[0][(0, 0)] - 0.5).abs() < 1e-12);
    let mut env = Lqg::new(c, 1).unwrap();
    env.reset(None).unwrap();
    let step = env.step(&Action::Continuous(vec![-1.])).unwrap();
    assert_eq!(step.reward.total(), -6.);
    assert!(step.terminated);
    assert!(env.step(&Action::Continuous(vec![0.])).is_err());
}
