#![cfg(feature = "neural")]

use std::num::NonZeroU16;

use burn::{module::Param, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    nn::{critics::scalar::QValue, matrix, policy::Policy, to_vec, vector},
    objectives::mpo::sampled_current_policy_continuation,
};

#[test]
fn continuous_expectation_is_detached_and_uses_the_declared_current_actor() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut policy =
        Policy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![2.], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[1.])));
    let mut target = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    target.network.layers[0].weight = Param::from_tensor(matrix(vec![1., 2.], 2, 1));
    target.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.5])));
    let states = [Observation::new(vec![-2.]), Observation::new(vec![3.])];
    let run = |seed| -> Tensor<1> {
        sampled_current_policy_continuation(
            &policy,
            &target,
            states.as_slice().into(),
            NonZeroU16::new(512).unwrap(),
            &mut ChaCha8Rng::seed_from_u64(seed),
        )
        .unwrap()
    };
    let bootstrap = run(97);
    assert!(!bootstrap.is_tracked());
    let values = to_vec(bootstrap).unwrap();
    assert_eq!(values, to_vec(run(97)).unwrap());
    for (value, expected) in values.iter().zip([-7.5, 17.5]) {
        assert!((value - expected).abs() < 0.25);
    }
}

#[test]
fn saturated_box_policy_can_bootstrap_from_a_finite_executed_action() {
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let mut policy = Policy::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[f32::MAX * 0.75])));
    let mut target = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    target.network.layers[0].weight = Param::from_tensor(matrix(vec![0., 1.], 2, 1));
    let states = [Observation::new(vec![0.])];
    let value = sampled_current_policy_continuation(
        &policy,
        &target,
        states.as_slice().into(),
        NonZeroU16::new(1).unwrap(),
        &mut rng,
    )
    .unwrap();
    assert_eq!(to_vec(value).unwrap(), vec![1.]);
}
