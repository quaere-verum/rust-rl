#![cfg(feature = "neural")]

use burn::{module::Module, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, RolloutActor, RolloutBehavior},
        env::{Observation, ObservationBatch},
    },
    nn::{
        critics::scalar::Value,
        device,
        distributions::Distribution,
        network::{Mlp, Network},
        observations,
        policy::{ActorCritic, Policy, StochasticPolicy},
        target::polyak_update,
        to_vec,
    },
};

#[test]
fn squashed_gaussian_density_includes_affine_and_tanh_jacobians() {
    let dist = Distribution::Gaussian {
        mean: Tensor::from_data([[0.0_f32]], &device()),
        log_std: Tensor::from_data([[0.0_f32]], &device()),
        bounds: Some((vec![-2.], vec![2.])),
    };
    let logp = dist
        .log_prob(&[Action::Continuous(vec![0.])], &[Some(vec![0.])])
        .unwrap()
        .into_scalar::<f32>();
    assert!((logp - (-1.6120857)).abs() < 1e-5);
    let near = dist
        .log_prob(&[Action::Continuous(vec![2.])], &[Some(vec![20.])])
        .unwrap()
        .into_scalar::<f32>();
    assert!(near.is_finite());
}

#[test]
fn masked_categorical_sampling_and_likelihood_agree() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let policy = Policy::new(2, ActionSpace::Discrete { count: 3 }, &[8], &mut rng).unwrap();
    let obs = vec![Observation::dense(vec![1., 2.], Some(vec![false, true, false])); 32];
    let dist = policy.valid().distribution((&obs).into()).unwrap();
    let mut samples = RolloutBehavior::default();
    dist.sample_behavior_into(ActMode::Sample, &mut rng, &mut samples)
        .unwrap();
    assert!(
        samples
            .actions
            .iter()
            .zip(&samples.log_probs)
            .all(|(a, lp)| *a == Action::Discrete(1) && lp.abs() < 1e-6)
    );
    let bad = [Observation::dense(vec![1., 2.], Some(vec![false; 3]))];
    assert!(policy.distribution((&bad).into()).is_err());
}

#[test]
fn selected_observations_pack_in_lane_order_without_copying_observations() {
    let obs = [
        Observation::new(vec![1., 2.]),
        Observation::new(vec![3., 4.]),
        Observation::new(vec![5., 6.]),
    ];
    let view = ObservationBatch::selected(&obs, &[2, 0, 2]).unwrap();
    assert!(std::ptr::eq(view.get(0).unwrap(), &obs[2]));
    assert_eq!(
        to_vec(observations(view, 2).unwrap()).unwrap(),
        [5., 6., 1., 2., 5., 6.]
    );
}

#[test]
fn actor_critic_action_only_does_not_evaluate_the_value_network() {
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let actor = ActorCritic {
        policy: Policy::new(2, ActionSpace::Discrete { count: 2 }, &[4], &mut rng).unwrap(),
        // A nonscalar network makes any accidental value evaluation fail.
        value: Value {
            network: Mlp::new(2, &[4], 2, &mut rng).unwrap(),
        },
    };
    let obs = [Observation::new(vec![0., 1.])];
    let mut actions = Vec::new();
    actor
        .act_into((&obs).into(), ActMode::Sample, &mut rng, &mut actions)
        .unwrap();
    assert_eq!(actions.len(), 1);
    assert!(
        actor
            .sample_rollout_into((&obs).into(), &mut rng, &mut RolloutBehavior::default())
            .is_err()
    );
}

#[test]
fn target_interpolation_and_seeded_initialization_are_reproducible() {
    let mut a = ChaCha8Rng::seed_from_u64(9);
    let mut b = a.clone();
    let online = Mlp::new(2, &[8], 1, &mut a).unwrap();
    let same = Mlp::new(2, &[8], 1, &mut b).unwrap();
    let x = Tensor::from_data([[1.0_f32, 2.]], &device());
    assert_eq!(
        online.valid().forward(x.clone()).into_scalar::<f32>(),
        same.valid().forward(x.clone()).into_scalar::<f32>()
    );
    let target = Mlp::new(2, &[8], 1, &mut a).unwrap().freeze();
    let copied = polyak_update(target, &online, 1.).unwrap();
    assert!(
        (copied.forward(x.clone()).into_scalar::<f32>()
            - online.valid().forward(x).into_scalar::<f32>())
        .abs()
            < 1e-6
    );
}

#[test]
fn gaussian_kl_is_zero_for_identical_parameters() {
    let dist = Distribution::Gaussian {
        mean: Tensor::from_data([[1.0_f32, 2.]], &device()),
        log_std: Tensor::from_data([[0.0_f32, -1.]], &device()),
        bounds: None,
    };
    assert!(dist.kl(&dist).unwrap().into_scalar::<f32>().abs() < 1e-6);
}
