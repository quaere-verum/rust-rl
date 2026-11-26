#![cfg(feature = "neural")]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::ActionSpace,
        env::{Observation, ObservationBatch},
    },
    nn::{
        distributions::Distribution, gaussian_policy::GaussianPolicy, matrix, network::Mlp,
        policy::StochasticPolicy, to_vec, vector,
    },
};

#[test]
fn state_dependent_mean_and_deviation_share_a_custom_network_contract() {
    let mut network = Mlp::new(1, &[], 4, &mut ChaCha8Rng::seed_from_u64(17)).unwrap();
    network.layers[0].weight = Param::from_tensor(matrix(vec![1., 2., -1., 1.], 1, 4));
    network.layers[0].bias = Some(Param::from_tensor(vector(&[0.25, -0.5, 0., -1.])));
    let space = ActionSpace::Box {
        low: vec![-2., 1.],
        high: vec![4., 5.],
    };
    let policy = GaussianPolicy::from_network(network, space).unwrap();
    let observations = [Observation::new(vec![1.]), Observation::new(vec![-1.])];
    let Distribution::Gaussian {
        mean,
        log_std,
        bounds,
    } = policy
        .distribution(ObservationBatch::all(&observations))
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(to_vec(mean).unwrap(), vec![1.25, 1.5, -0.75, -2.5]);
    assert_eq!(to_vec(log_std).unwrap(), vec![-1., 0., 1., -2.]);
    assert_eq!(bounds, Some((vec![-2., 1.], vec![4., 5.])));
}

#[test]
fn gaussian_policy_rejects_discrete_or_unrepresentable_affine_actions() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![f32::MAX * 0.75],
            high: vec![f32::MAX],
        },
        ActionSpace::Box {
            low: vec![0.],
            high: vec![f32::from_bits(1)],
        },
    ] {
        assert!(GaussianPolicy::new(1, space, &[], &mut ChaCha8Rng::seed_from_u64(17)).is_err());
    }
    let wrong = Mlp::new(1, &[], 1, &mut ChaCha8Rng::seed_from_u64(18)).unwrap();
    assert!(GaussianPolicy::from_network(wrong, ActionSpace::Unbounded { dimensions: 1 }).is_err());
}
