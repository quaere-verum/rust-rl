use super::*;
use crate::{
    core::{action::ActionSpace, env::Observation},
    nn::{gaussian_policy::GaussianPolicy, matrix, to_vec, vector},
};
use burn::module::Param;

#[test]
fn categorical_nll_has_exact_masked_linear_gradient_and_no_variance_gradient() {
    let mut policy = Policy::new(
        1,
        ActionSpace::Discrete { count: 3 },
        &[],
        &mut ChaCha8Rng::seed_from_u64(2),
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.; 3], 1, 3));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.; 3])));
    let data = Demonstrations::new(
        policy.action_space().clone(),
        [-1., 1.]
            .into_iter()
            .map(|x| Observation::dense(vec![x], Some(vec![true, true, false])))
            .collect(),
        vec![Action::Discrete(0), Action::Discrete(1)],
    )
    .unwrap();
    let loss = BatchScratch::default()
        .loss(&policy, &data, &[0, 1])
        .unwrap();
    assert!((finite_loss(&loss).unwrap() - 2_f64.ln()).abs() < 1e-6);
    let grads = loss.backward();
    assert_eq!(
        to_vec(policy.network.layers[0].weight.val().grad(&grads).unwrap()).unwrap(),
        [0.5, -0.5, 0.]
    );
    assert_eq!(
        to_vec(
            policy.network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
                .grad(&grads)
                .unwrap()
        )
        .unwrap(),
        [0., 0., 0.]
    );
    assert!(policy.log_std.val().grad(&grads).is_none());
}
#[test]
fn continuous_nll_gradients_match_independent_normal_calculus() {
    let space = ActionSpace::Box {
        low: vec![-2., 1.],
        high: vec![4., 5.],
    };
    let mut policy =
        GaussianPolicy::new(1, space.clone(), &[], &mut ChaCha8Rng::seed_from_u64(3)).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.; 4], 1, 4));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.; 4])));
    let data = Demonstrations::new(
        space,
        vec![Observation::new(vec![0.])],
        vec![Action::Continuous(vec![
            1. + 3. * 0.2_f32.tanh(),
            3. + 2. * (-0.3_f32).tanh(),
        ])],
    )
    .unwrap();
    let loss = BatchScratch::default().loss(&policy, &data, &[0]).unwrap();
    let grads = loss.backward();
    let actual = to_vec(
        policy.network.layers[0]
            .bias
            .as_ref()
            .unwrap()
            .val()
            .grad(&grads)
            .unwrap(),
    )
    .unwrap();
    for (&a, expected) in actual.iter().zip([-0.2, 0.3, 0.96, 0.91]) {
        assert!((a - expected).abs() < 2e-6);
    }
}

#[test]
fn evaluation_weights_short_final_batch_by_examples_and_does_not_consume_rng() {
    let mut policy = Policy::new(
        1,
        ActionSpace::Discrete { count: 2 },
        &[],
        &mut ChaCha8Rng::seed_from_u64(4),
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![1., -1.], 1, 2));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0.])));
    let data = Demonstrations::new(
        policy.action_space().clone(),
        [0., 1., 2.]
            .into_iter()
            .map(|x| Observation::new(vec![x]))
            .collect(),
        vec![
            Action::Discrete(0),
            Action::Discrete(0),
            Action::Discrete(1),
        ],
    )
    .unwrap();
    let learner = BehaviorCloning::new(
        policy,
        CloningConfig {
            batch_size: 2,
            ..Default::default()
        },
        5,
    )
    .unwrap();
    let rng = learner.rng.clone();
    let expected = (2_f64.ln() + (-2_f64).exp().ln_1p() + 4_f64.exp().ln_1p()) / 3.;
    assert!((learner.evaluate(&data).unwrap() - expected).abs() < 1e-6);
    assert_eq!(learner.rng, rng);
}

#[derive(burn::module::Module, Debug)]
struct BufferedPolicy {
    inner: Policy,
    buffer: burn::module::RunningState<Tensor<1>>,
}
impl StochasticPolicy for BufferedPolicy {
    fn action_space(&self) -> &ActionSpace {
        self.inner.action_space()
    }
    fn distribution(
        &self,
        obs: ObservationBatch<'_>,
    ) -> Result<crate::nn::distributions::Distribution> {
        self.buffer.update(vector(&[f32::NAN]));
        self.inner.distribution(obs)
    }
}
#[test]
fn invalid_pending_running_buffer_stays_on_private_candidate() {
    use crate::core::checkpoint::Checkpoint;
    let inner = Policy::new(
        1,
        ActionSpace::Discrete { count: 2 },
        &[],
        &mut ChaCha8Rng::seed_from_u64(112),
    )
    .unwrap();
    let policy = BufferedPolicy {
        inner,
        buffer: burn::module::RunningState::new(vector(&[0.])),
    };
    let mut learner = BehaviorCloning::new(policy, Default::default(), 113).unwrap();
    let before = learner.checkpoint().unwrap();
    let data = Demonstrations::new(
        ActionSpace::Discrete { count: 2 },
        vec![Observation::new(vec![1.])],
        vec![Action::Discrete(1)],
    )
    .unwrap();
    assert!(learner.fit_epoch(&data).is_err());
    assert_eq!(to_vec(learner.policy.buffer.value_sync()).unwrap(), [0.]);
    assert!(learner.checkpoint().is_err());
    learner.restore(&before).unwrap();
    learner.checkpoint().unwrap();
}

#[test]
fn constructors_reject_nonfinite_pending_buffers_before_storing_a_policy() {
    let make = || {
        let policy = BufferedPolicy {
            inner: Policy::new(
                1,
                ActionSpace::Discrete { count: 2 },
                &[],
                &mut ChaCha8Rng::seed_from_u64(114),
            )
            .unwrap(),
            buffer: burn::module::RunningState::new(vector(&[0.])),
        };
        policy.buffer.update(vector(&[f32::NAN]));
        policy
    };
    assert!(BehaviorCloning::new(make(), Default::default(), 115).is_err());
    assert!(crate::nn::policy::PolicyActor::new(make()).is_err());
}
