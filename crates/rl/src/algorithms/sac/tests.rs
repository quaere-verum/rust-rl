use super::*;
use crate::core::{checkpoint::Checkpoint, env::Observation};
use crate::nn::{matrix, vector};
use burn::module::{Module, Param};

#[test]
fn fixed_noise_actor_gradient_flows_through_both_frozen_critics_actions_only() {
    let mean = matrix(vec![0.2], 1, 1).require_grad();
    let log_std = matrix(vec![0.5_f32.ln()], 1, 1).require_grad();
    let d = Distribution::Gaussian {
        mean: mean.clone(),
        log_std: log_std.clone(),
        bounds: Some((vec![-2.], vec![4.])),
    };
    let sample = d
        .reparameterize_with_noise_and_sampled_score(matrix(vec![0.4], 1, 1))
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(871);
    let mut first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let mut second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    for (q, slope, bias) in [(&mut first, 2., 10.), (&mut second, -3., 0.)] {
        q.network.layers[0].weight = Param::from_tensor(matrix(vec![0., slope], 2, 1));
        q.network.layers[0].bias = Some(Param::from_tensor(vector(&[bias])));
    }
    let obs = [Observation::new(vec![0.])];
    let q1 = first
        .valid()
        .freeze()
        .values((&obs[..]).into(), sample.actions.clone())
        .unwrap()
        .squeeze_dim(1);
    let q2 = second
        .valid()
        .freeze()
        .values((&obs[..]).into(), sample.actions)
        .unwrap()
        .squeeze_dim(1);
    let gradients = objective::actor_loss(sample.log_probs, q1, q2, 0.25)
        .unwrap()
        .backward();
    let t = 0.4_f64.tanh();
    let du = 0.5 * t + 9. * (1. - t * t);
    let dlogstd = -0.25 + du * 0.2;
    assert!((to_vec(mean.grad(&gradients).unwrap()).unwrap()[0] as f64 - du).abs() < 2e-6);
    assert!((to_vec(log_std.grad(&gradients).unwrap()).unwrap()[0] as f64 - dlogstd).abs() < 2e-6);
    for q in [&first, &second] {
        assert!(q.network.layers[0].weight.val().grad(&gradients).is_none());
        assert!(
            q.network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
                .grad(&gradients)
                .is_none()
        );
    }
}

#[test]
fn asymmetric_saturated_actions_remain_inside_box_in_collection_and_pathwise_sampling() {
    let space = ActionSpace::Box {
        low: vec![0.1, -0.3],
        high: vec![0.3, -0.1],
    };
    let mut rng = ChaCha8Rng::seed_from_u64(872);
    let mut policy = GaussianPolicy::new(1, space.clone(), &[], &mut rng).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.; 4], 1, 4));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[-30., 30., 0., 0.])));
    let first = QValue::new(1, 2, 1, &[], &mut rng).unwrap();
    let second = QValue::new(1, 2, 1, &[], &mut rng).unwrap();
    let learner = Sac::new(policy, first, second, Default::default(), 873).unwrap();
    let obs = [Observation::new(vec![1.])];
    for mode in [ActMode::Sample, ActMode::Deterministic] {
        for action in learner.act(&obs, mode, &mut rng).unwrap() {
            space.check(&action, None).unwrap();
        }
    }
    let d = gaussian(&learner.policy, (&obs[..]).into()).unwrap();
    let sample = d
        .reparameterize_with_noise_and_sampled_score(matrix(vec![0., 0.], 1, 2))
        .unwrap();
    space
        .check(&Action::Continuous(to_vec(sample.actions).unwrap()), None)
        .unwrap();
    assert!(to_vec(sample.log_probs).unwrap()[0].is_finite());
}

#[test]
fn frozen_critics_are_rejected_without_an_autodiff_panic() {
    let mut rng = ChaCha8Rng::seed_from_u64(874);
    let policy =
        GaussianPolicy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng).unwrap();
    let first = QValue::new(1, 1, 1, &[], &mut rng).unwrap().freeze();
    let second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    assert!(Sac::new(policy, first, second, Default::default(), 875).is_err());
}

#[test]
fn terminal_or_zero_discount_does_not_sample_successor_rng() {
    for gamma in [0., 0.9] {
        let mut rng = ChaCha8Rng::seed_from_u64(876);
        let policy =
            GaussianPolicy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng)
                .unwrap();
        let first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        let second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        let mut learner = Sac::new(
            policy,
            first,
            second,
            SacConfig {
                gamma,
                ..Default::default()
            },
            877,
        )
        .unwrap();
        let before = learner.checkpoint().unwrap().metadata["target_rng"].clone();
        let batch = ReplayBatch {
            items: vec![crate::data::replay::ReplayItem {
                transition: crate::core::transition::Transition {
                    observation: Observation::new(vec![1.]),
                    action: Action::Continuous(vec![0.]),
                    reward: 2.0_f32.into(),
                    next_observation: Observation::new(vec![f32::MAX]),
                    terminated: gamma > 0.,
                    truncated: gamma == 0.,
                },
                discount: 0.,
            }],
        };
        let report = learner.update(&batch).unwrap();
        assert_eq!(report.metrics["target_mean"], 2.);
        assert_eq!(learner.checkpoint().unwrap().metadata["target_rng"], before);
    }
}

#[test]
fn nonzero_entropy_targets_match_independent_normal_density_with_affine_jacobian() {
    use rand::Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(881);
    let mut policy = GaussianPolicy::new(
        1,
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![4.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0., 0.], 1, 2));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0.])));
    let mut first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let mut second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    for (q, bias) in [(&mut first, 3.), (&mut second, 1.)] {
        q.network.layers[0].weight = Param::from_tensor(matrix(vec![0., 0.], 2, 1));
        q.network.layers[0].bias = Some(Param::from_tensor(vector(&[bias])));
    }
    let learner = Sac::new(
        policy,
        first,
        second,
        SacConfig {
            gamma: 0.5,
            temperature: Temperature::Fixed(0.2),
            ..Default::default()
        },
        882,
    )
    .unwrap();
    let batch = ReplayBatch {
        items: [true, false, false]
            .into_iter()
            .map(|terminated| crate::data::replay::ReplayItem {
                transition: crate::core::transition::Transition {
                    observation: Observation::new(vec![1.]),
                    action: Action::Continuous(vec![0.]),
                    reward: 2.0_f32.into(),
                    next_observation: Observation::new(vec![1.]),
                    terminated,
                    truncated: !terminated,
                },
                discount: if terminated { 0. } else { 0.5 },
            })
            .collect(),
    };
    let prepared = PreparedReplayBatch::new(&batch, learner.policy.action_space()).unwrap();
    let actual = to_vec(
        learner
            .targets(&prepared, &mut learner.target_rng.clone())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(actual[0], 2.);
    let mut noise_rng = learner.target_rng.clone();
    for &value in &actual[1..] {
        let z = noise_rng.sample::<f32, _>(rand_distr::StandardNormal) as f64;
        let lp = -0.5 * z * z
            - 0.5 * (2. * std::f64::consts::PI).ln()
            - 3_f64.ln()
            - (1. - z.tanh().powi(2)).ln();
        let expected = 2. + 0.5 * (1. - 0.2 * lp);
        assert!((value as f64 - expected).abs() < 1e-6);
    }
}

#[derive(Module, Debug)]
struct ConditionalQ {
    negative: Param<Tensor<2>>,
    positive: Param<Tensor<2>>,
}
impl ActionValue for ConditionalQ {
    fn action_dim(&self) -> usize {
        1
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        let parameter = if obs.iter().next().unwrap().dense_values()[0] < 0. {
            &self.negative
        } else {
            &self.positive
        };
        Ok(actions.matmul(parameter.val()))
    }
}
#[test]
fn conditional_custom_critic_restores_its_own_sparse_adam_history() {
    let mut rng = ChaCha8Rng::seed_from_u64(883);
    let policy =
        GaussianPolicy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng).unwrap();
    let q = || ConditionalQ {
        negative: Param::from_tensor(matrix(vec![0.5], 1, 1)),
        positive: Param::from_tensor(matrix(vec![0.25], 1, 1)),
    };
    let mut learner = Sac::new(policy, q(), q(), Default::default(), 884).unwrap();
    for sign in [-1., 1.] {
        let batch = ReplayBatch {
            items: vec![crate::data::replay::ReplayItem {
                transition: crate::core::transition::Transition {
                    observation: Observation::new(vec![sign]),
                    action: Action::Continuous(vec![0.5]),
                    reward: 1.0_f32.into(),
                    next_observation: Observation::new(vec![sign]),
                    terminated: true,
                    truncated: false,
                },
                discount: 0.,
            }],
        };
        learner.update(&batch).unwrap();
    }
    let record = learner.checkpoint().unwrap();
    let steps =
        crate::nn::checkpoint::audit_adam_optimizer(&record, "first", "first", None, 2).unwrap();
    assert_eq!(steps.steps.len(), 2);
    assert!(steps.steps.values().all(|&n| n == 1));
    learner.restore(&record).unwrap();
}

#[test]
fn checkpoint_rejects_changed_trainability_and_the_earlier_unbound_envelope() {
    let make = |freeze_bias| {
        let mut rng = ChaCha8Rng::seed_from_u64(887);
        let mut policy =
            GaussianPolicy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng)
                .unwrap();
        if freeze_bias {
            policy.network.layers[0].bias = policy.network.layers[0]
                .bias
                .take()
                .map(|p| p.set_require_grad(false));
        }
        let first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        let second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        Sac::new(policy, first, second, Default::default(), 888).unwrap()
    };
    let mut source = make(true);
    let record = source.checkpoint().unwrap();
    source.restore(&record).unwrap();
    assert!(make(false).restore(&record).is_err());
    let mut earlier = record;
    earlier.algorithm = "sac-v1".into();
    earlier.metadata["version"] = serde_json::json!(1);
    earlier
        .metadata
        .as_object_mut()
        .unwrap()
        .remove("trainability");
    assert!(source.restore(&earlier).is_err());
}
