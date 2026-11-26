#![cfg(feature = "neural")]

use burn::{module::Param, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::ddpg::{Ddpg, DdpgConfig},
    core::{action::ActionSpace, checkpoint::Checkpoint, env::Observation},
    nn::{
        critics::{
            residual::{ResidualQ, ResidualScale},
            scalar::{ActionValue, QValue},
        },
        matrix,
        policy::{Deterministic, DeterministicPolicy},
        to_vec, vector,
    },
};

fn actor(parameters: &[f32; 6]) -> Deterministic {
    let space = ActionSpace::Box {
        low: vec![-2., -1.],
        high: vec![2., 3.],
    };
    let mut model = Deterministic::new(2, space, &[], &mut ChaCha8Rng::seed_from_u64(1)).unwrap();
    model.network.layers[0].weight = Param::from_tensor(matrix(parameters[..4].to_vec(), 2, 2));
    model.network.layers[0].bias = Some(Param::from_tensor(vector(&parameters[4..])));
    model
}

fn inner() -> QValue {
    let mut model = QValue::new(2, 2, 1, &[], &mut ChaCha8Rng::seed_from_u64(2)).unwrap();
    model.network.layers[0].weight = Param::from_tensor(matrix(vec![2., -3., 1., 2.], 4, 1));
    model.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.1])));
    model
}

#[test]
fn complete_actor_parameter_gradient_matches_fixed_anchor_finite_differences() {
    let states = [
        Observation::new(vec![1., 0.3]),
        Observation::new(vec![-0.4, 2.]),
    ];
    let obs = states.as_slice().into();
    let parameters = [0.2, 0.1, -0.5, 0.25, 0.3, -0.6];
    let policy = actor(&parameters);
    let critic = ResidualQ::new(inner(), &policy, ResidualScale::Diagonal(vec![0.5, 2.])).unwrap();
    let objective = |actor: &Deterministic| -> Tensor<1> {
        critic
            .values(obs, actor.actions(obs).unwrap())
            .unwrap()
            .mean()
    };
    let grads = objective(&policy).backward();
    let mut gradient = to_vec(policy.network.layers[0].weight.val().grad(&grads).unwrap()).unwrap();
    gradient.extend(
        to_vec(
            policy.network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
                .grad(&grads)
                .unwrap(),
        )
        .unwrap(),
    );
    assert!(gradient.iter().any(|g| g.abs() > 0.5));
    let step = 0.001;
    for (i, actual) in gradient.into_iter().enumerate() {
        let mut left = parameters;
        let mut right = parameters;
        left[i] -= step;
        right[i] += step;
        let low = to_vec(objective(&actor(&left))).unwrap()[0];
        let high = to_vec(objective(&actor(&right))).unwrap()[0];
        let expected = (high - low) / (2. * step);
        assert!(
            (actual - expected).abs() < 0.001,
            "{i}: {actual} versus {expected}"
        );
    }
}

#[test]
fn refresh_is_explicit_and_checkpoint_restores_the_actual_anchor_and_scale() {
    let states = [Observation::new(vec![1., 0.3])];
    let obs = states.as_slice().into();
    let actions = matrix(vec![0.4, 1.1], 1, 2);
    let old = actor(&[0.; 6]);
    let current = actor(&[0.4; 6]);
    let critic = ResidualQ::new(inner(), &old, ResidualScale::Scalar(0.25)).unwrap();
    let mut learner = Ddpg::new(current, critic, DdpgConfig::default(), 3).unwrap();
    let evaluate = |model: &ResidualQ| to_vec(model.values(obs, actions.clone()).unwrap()).unwrap();
    let before = evaluate(&learner.critic());
    assert_eq!(before, evaluate(&learner.target_critic()));
    let anchor = learner.policy();
    learner.refresh_residual_anchor(&anchor).unwrap();
    let after = evaluate(&learner.critic());
    assert_ne!(before, after);
    assert_eq!(after, evaluate(&learner.target_critic()));
    let saved = learner.checkpoint().unwrap();
    let template = ResidualQ::new(inner(), &old, ResidualScale::Scalar(2.)).unwrap();
    let mut restored = Ddpg::new(old, template, DdpgConfig::default(), 4).unwrap();
    restored.restore(&saved).unwrap();
    for model in [restored.critic(), restored.target_critic()] {
        assert_eq!(evaluate(&model), after);
        assert_eq!(model.scales().unwrap(), vec![0.25, 0.25]);
        assert_eq!(
            model.anchor_identity().unwrap(),
            learner.critic().anchor_identity().unwrap()
        );
    }
}

#[test]
fn diagonal_scale_is_positive_and_matches_the_action_coordinates() {
    let policy = actor(&[0.; 6]);
    for scale in [
        ResidualScale::Scalar(0.),
        ResidualScale::Scalar(f32::NAN),
        ResidualScale::Diagonal(vec![1.]),
        ResidualScale::Diagonal(vec![1., -1.]),
    ] {
        assert!(ResidualQ::new(inner(), &policy, scale).is_err());
    }
}
