#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    nn::{critics::scalar::StateValue, to_vec},
    prelude::*,
};

fn fixture(space: ActionSpace) -> (ActorCritic, OnPolicyBatch) {
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let models = ActorCritic {
        policy: Policy::new(1, space, &[], &mut rng).unwrap(),
        value: Value::new(1, &[], &mut rng).unwrap(),
    };
    let observations: Vec<_> = (0..8)
        .map(|i| Observation::new(vec![i as f32 / 8.]))
        .collect();
    let mut behavior = RolloutBehavior::default();
    models
        .sample_rollout_into((&observations).into(), &mut rng, &mut behavior)
        .unwrap();
    let batch = OnPolicyBatch {
        episode_count: 2,
        observations,
        actions: behavior.actions,
        latents: behavior.latents,
        returns: vec![0.5; 8],
        discount_weights: vec![1., 0.8, 0.6, 0.4, 1., 0.8, 0.6, 0.4],
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: behavior.log_probs,
            values: behavior.values,
            advantages: vec![1., -0.5, 0., 2., -1., 0.25, 1.5, -0.75],
        }),
    };
    (models, batch)
}
fn config(m: usize, t: f32) -> MdpoConfig {
    MdpoConfig {
        policy: OptimizerConfig {
            learning_rate: 0.03,
            max_grad_norm: None,
        },
        value: OptimizerConfig {
            learning_rate: 0.01,
            max_grad_norm: None,
        },
        inner_steps: m,
        value_epochs: 2,
        value_minibatch_size: 3,
        mirror_step: MirrorStepSchedule::Constant(t),
        ..Default::default()
    }
}
fn probabilities(learner: &Mdpo, batch: &OnPolicyBatch) -> Vec<f32> {
    to_vec(
        learner
            .policy()
            .distribution((&batch.observations).into())
            .unwrap()
            .log_prob(&batch.actions, &batch.latents)
            .unwrap(),
    )
    .unwrap()
}
fn record(learner: &Mdpo) -> serde_json::Value {
    let record = learner.checkpoint().unwrap();
    let pack = |bytes: Vec<u8>| {
        let reader =
            burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes)).unwrap();
        let scalars = format!("{:?}", reader.scalars());
        let metadata = reader.metadata().clone();
        let tensors: std::collections::BTreeMap<_, _> = reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|t| {
                let (name, dtype, shape, id, bytes) = t.into_parts().unwrap();
                (
                    name,
                    serde_json::json!([format!("{dtype:?}"), shape.to_vec(), id, bytes.to_vec()]),
                )
            })
            .collect();
        serde_json::json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
    };
    let models: std::collections::BTreeMap<_, _> = record
        .models
        .into_iter()
        .map(|(name, m)| {
            let mut json = serde_json::to_value(m).unwrap();
            let bytes = serde_json::from_value(json["bytes"].clone()).unwrap();
            json["bytes"] = pack(bytes);
            (name, json)
        })
        .collect();
    let optimizers: std::collections::BTreeMap<_, _> = record
        .optimizers
        .into_iter()
        .map(|(name, bytes)| (name, pack(bytes)))
        .collect();
    serde_json::json!({"metadata":record.metadata,"models":models,"optimizers":optimizers})
}
#[test]
fn supports_all_action_families_and_exact_optimizer_counts() {
    for space in [
        ActionSpace::Discrete { count: 3 },
        ActionSpace::Unbounded { dimensions: 2 },
        ActionSpace::Box {
            low: vec![-2., 0.],
            high: vec![1., 3.],
        },
    ] {
        let (models, batch) = fixture(space);
        let mut learner = Mdpo::new(models, config(10, 0.5), 4).unwrap();
        let before = probabilities(&learner, &batch);
        let report = learner.update(&batch).unwrap();
        assert_eq!(report.updates, 16); // ten actor + two passes * ceil(8/3) critic
        assert_eq!(learner.updates(), 16);
        assert_eq!(learner.outer_iterations(), 1);
        assert_eq!(report.metrics["actor_updates"], 10.);
        assert_eq!(report.metrics["critic_updates"], 6.);
        assert_eq!(report.metrics["mirror_step"], 0.5);
        assert_eq!(report.metrics["kl_coefficient"], 2.);
        assert!(report.metrics.values().all(|x| x.is_finite()));
        assert!(report.metrics["kl_new_behavior"] > 0.);
        assert_ne!(before, probabilities(&learner, &batch));
        assert!(matches!(
            learner.collection_actor(),
            OnPolicyActorRef::Bootstrapped(_)
        ));
    }
}

#[test]
fn stale_and_invalid_batches_are_rejected_without_any_mutation() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut learner = Mdpo::new(models, config(3, 0.5), 4).unwrap();
    let before = record(&learner);
    let mut invalid = batch.clone();
    if let OnPolicyMetadata::Bootstrapped(b) = &mut invalid.metadata {
        b.log_probs[0] += 0.01;
    }
    assert!(learner.update(&invalid).is_err());
    assert_eq!(record(&learner), before);
    invalid = batch.clone();
    invalid.actions[0] = Action::Discrete(99);
    assert!(learner.update(&invalid).is_err());
    assert_eq!(record(&learner), before);
    invalid = batch.clone();
    invalid.observations[0].as_dense_mut().unwrap().action_mask = Some(vec![false; 3]);
    assert!(learner.update(&invalid).is_err());
    assert_eq!(record(&learner), before);
    invalid = batch.clone();
    invalid.returns[0] = f32::NAN;
    assert!(learner.update(&invalid).is_err());
    assert_eq!(record(&learner), before);
    learner.update(&batch).unwrap();
    let after = record(&learner);
    assert!(learner.update(&batch).is_err());
    assert_eq!(record(&learner), after);
}

#[test]
fn first_step_is_independent_of_penalty_but_later_steps_are_regularized() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut weak = Mdpo::new(models.clone(), config(1, 10.), 4).unwrap();
    let mut strong = Mdpo::new(models.clone(), config(1, 0.01), 4).unwrap();
    weak.update(&batch).unwrap();
    strong.update(&batch).unwrap();
    let one = probabilities(&weak, &batch);
    for (a, b) in one.iter().zip(probabilities(&strong, &batch)) {
        assert!((a - b).abs() < 2e-6, "{a} vs {b}");
    }
    let mut weak = Mdpo::new(models.clone(), config(10, 10.), 4).unwrap();
    let mut strong = Mdpo::new(models, config(10, 0.03), 4).unwrap();
    let w = weak.update(&batch).unwrap();
    let s = strong.update(&batch).unwrap();
    assert!(w.metrics["kl_new_behavior"] > s.metrics["kl_new_behavior"] * 2.);
    assert_ne!(one, probabilities(&strong, &batch));
}

#[test]
fn critic_passes_and_return_targets_cannot_change_completed_actor_update() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut a = Mdpo::new(models.clone(), config(4, 0.5), 4).unwrap();
    let mut c = config(4, 0.5);
    c.value_epochs = 7;
    let mut b = Mdpo::new(models, c, 4).unwrap();
    let mut other = batch.clone();
    other.returns.fill(9.);
    a.update(&batch).unwrap();
    b.update(&other).unwrap();
    assert_eq!(probabilities(&a, &batch), probabilities(&b, &batch));
    assert_ne!(
        to_vec(a.value().values((&batch.observations).into()).unwrap()).unwrap(),
        to_vec(b.value().values((&batch.observations).into()).unwrap()).unwrap()
    );
}

#[test]
fn schedule_validation_endpoints_and_outer_progression() {
    for t in [0., -1., f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        assert!(MirrorStepSchedule::Constant(t).value(0).is_err());
    }
    let schedule = MirrorStepSchedule::Linear {
        start: 1.,
        end: 0.2,
        duration: 2,
    };
    assert_eq!(schedule.value(0).unwrap(), 1.);
    assert!((schedule.value(1).unwrap() - 0.6).abs() < 1e-6);
    assert_eq!(schedule.value(u64::MAX).unwrap(), 0.2);
    assert!(
        MirrorStepSchedule::Linear {
            start: 1.,
            end: 0.,
            duration: 2
        }
        .value(0)
        .is_err()
    );
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut c = config(3, 1.);
    c.mirror_step = schedule;
    let mut l = Mdpo::new(models, c, 2).unwrap();
    assert_eq!(l.update(&batch).unwrap().metrics["mirror_step"], 1.);
    let mut next = batch.clone();
    let lp = probabilities(&l, &next);
    if let OnPolicyMetadata::Bootstrapped(b) = &mut next.metadata {
        b.log_probs = lp;
    }
    assert!((l.update(&next).unwrap().metrics["mirror_step"] - 0.6).abs() < 1e-6);
    assert_eq!(l.outer_iterations(), 2);
}

#[test]
fn checkpoint_continuation_and_atomic_rejection() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut c = config(4, 0.5);
    c.mirror_step = MirrorStepSchedule::Linear {
        start: 0.5,
        end: 0.1,
        duration: 9,
    };
    let mut a = Mdpo::new(models.clone(), c, 33).unwrap();
    a.update(&batch).unwrap();
    let saved = a.checkpoint().unwrap();
    let mut b = Mdpo::new(models, MdpoConfig::default(), 99).unwrap();
    b.restore(&saved).unwrap();
    let mut next = batch.clone();
    let lp = probabilities(&a, &next);
    if let OnPolicyMetadata::Bootstrapped(b) = &mut next.metadata {
        b.log_probs = lp;
    }
    let ar = a.update(&next).unwrap();
    let br = b.update(&next).unwrap();
    assert_eq!(ar.metrics, br.metrics);
    assert_eq!(record(&a), record(&b));
    let before = record(&b);
    let mut bad = saved.clone();
    bad.metadata["config"]["inner_steps"] = 0.into();
    assert!(b.restore(&bad).is_err());
    assert_eq!(record(&b), before);
    let mut bad = saved.clone();
    bad.optimizers.remove("value");
    assert!(b.restore(&bad).is_err());
    assert_eq!(record(&b), before);
    let mut bad = saved.clone();
    bad.models.remove("policy");
    assert!(b.restore(&bad).is_err());
    assert_eq!(record(&b), before);
    let mut bad = saved;
    bad.metadata["space"] = serde_json::to_value(ActionSpace::Discrete { count: 5 }).unwrap();
    assert!(b.restore(&bad).is_err());
    assert_eq!(record(&b), before);
}

#[test]
fn occupancy_validation_and_uniform_equivalence() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut c = config(4, 0.5);
    c.weighting = MdpoWeighting::DiscountedOccupancy;
    let mut learner = Mdpo::new(models.clone(), c.clone(), 2).unwrap();
    let before = record(&learner);
    for weights in [vec![0.; 8], vec![-1.; 8]] {
        let mut b = batch.clone();
        b.discount_weights = weights;
        assert!(learner.update(&b).is_err());
        assert_eq!(record(&learner), before);
    }
    let mut uniform = batch.clone();
    uniform.discount_weights.fill(1.);
    let mut other = Mdpo::new(models.clone(), config(4, 0.5), 2).unwrap();
    learner.update(&uniform).unwrap();
    other.update(&uniform).unwrap();
    assert_eq!(
        probabilities(&learner, &uniform),
        probabilities(&other, &uniform)
    );
    let mut weighted = Mdpo::new(models, c, 2).unwrap();
    weighted.update(&batch).unwrap();
    assert_ne!(
        probabilities(&weighted, &batch),
        probabilities(&other, &batch)
    );
}

#[test]
fn inner_loop_matches_manual_adam_with_one_frozen_reference_and_fixed_advantages() {
    use burn::{module::Module, optim::GradientsParams};
    use rl::nn::{target::frozen_copy, vector};
    let (mut manual, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let c = config(6, 0.4);
    let mut optimizer = c.policy.build().unwrap();
    let mut learner = Mdpo::new(manual.clone(), c.clone(), 4).unwrap();
    let behavior = frozen_copy(&manual.policy)
        .distribution((&batch.observations).into())
        .unwrap();
    let old = behavior.log_prob(&batch.actions, &batch.latents).unwrap();
    let advantages = vector(&batch.bootstrapped().unwrap().advantages);
    for _ in 0..c.inner_steps {
        let current = manual
            .policy
            .distribution((&batch.observations).into())
            .unwrap();
        let ratio = (current.log_prob(&batch.actions, &batch.latents).unwrap() - old.clone()).exp();
        let loss =
            (ratio * advantages.clone()).mean().neg() + current.kl(&behavior).unwrap().mean() / 0.4;
        let grads = GradientsParams::from_grads(loss.backward(), &manual.policy);
        manual.policy = optimizer.step(c.policy.learning_rate, manual.policy, grads);
    }
    let report = learner.update(&batch).unwrap();
    let expected = to_vec(
        manual
            .policy
            .valid()
            .distribution((&batch.observations).into())
            .unwrap()
            .log_prob(&batch.actions, &batch.latents)
            .unwrap(),
    )
    .unwrap();
    for (a, b) in expected.iter().zip(probabilities(&learner, &batch)) {
        assert!((a - b).abs() < 2e-6);
    }
    let final_distribution = learner
        .policy()
        .distribution((&batch.observations).into())
        .unwrap();
    let kl = final_distribution
        .kl(&behavior)
        .unwrap()
        .mean()
        .into_scalar::<f32>() as f64;
    assert!((report.metrics["kl_new_behavior"] - kl).abs() < 1e-7);
}

#[test]
fn corrupt_models_optimizer_moments_and_shapes_are_rejected_atomically() {
    use rl::{core::checkpoint::LearnerRecord, nn::checkpoint::RecordBurnExt};
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut learner = Mdpo::new(models, config(3, 0.5), 4).unwrap();
    learner.update(&batch).unwrap();
    let good = learner.checkpoint().unwrap();
    let before = record(&learner);
    let mut bad = good.clone();
    let reader = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(
        bad.optimizers["policy"].clone(),
    ))
    .unwrap();
    let scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    let i = tensors
        .iter()
        .position(|t| t.name.ends_with(".moment_1"))
        .unwrap();
    tensors.remove(i);
    let mut writer = burn_pack::Writer::new(tensors);
    for (k, v) in scalars {
        writer = writer.with_scalar(&k, v);
    }
    for (k, v) in metadata {
        writer = writer.with_metadata(&k, &v);
    }
    bad.optimizers
        .insert("policy".into(), writer.into_bytes().unwrap().to_vec());
    assert!(learner.restore(&bad).is_err());
    assert_eq!(record(&learner), before);
    let mut json = serde_json::to_value(&good).unwrap();
    json["models"]["policy"]["bytes"] = serde_json::json!([1, 2, 3]);
    let bad: LearnerRecord = serde_json::from_value(json).unwrap();
    assert!(learner.restore(&bad).is_err());
    assert_eq!(record(&learner), before);
    let mut policy = learner.policy();
    policy.network.layers[0].weight =
        burn::module::Param::from_tensor(rl::nn::matrix(vec![f32::NAN; 3], 1, 3));
    let mut bad = good.clone();
    bad.model("policy", &policy).unwrap();
    assert!(learner.restore(&bad).is_err());
    assert_eq!(record(&learner), before);
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let wrong = ActorCritic {
        policy: Policy::new(1, ActionSpace::Discrete { count: 3 }, &[4], &mut rng).unwrap(),
        value: Value::new(1, &[], &mut rng).unwrap(),
    };
    assert!(
        Mdpo::new(wrong, config(3, 0.5), 2)
            .unwrap()
            .restore(&good)
            .is_err()
    );
}

#[test]
fn policy_snapshots_and_cached_behavior_are_independent_of_live_running_buffers() {
    use burn::{
        module::{Module, RunningState},
        tensor::Tensor,
    };
    use rl::nn::distributions::Distribution;
    #[derive(Module, Debug)]
    struct BufferedPolicy {
        inner: Policy,
        running: RunningState<Tensor<1>>,
    }
    impl StochasticPolicy for BufferedPolicy {
        fn action_space(&self) -> &ActionSpace {
            self.inner.action_space()
        }
        fn distribution(&self, obs: ObservationBatch<'_>) -> Result<Distribution> {
            if self.inner.network.layers[0].weight.val().is_require_grad() {
                let next = self.running.value_sync() + 1.;
                self.running.update(next);
            }
            self.inner.distribution(obs)
        }
    }
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let models = ActorCritic {
        policy: BufferedPolicy {
            inner: models.policy,
            running: RunningState::new(rl::nn::vector(&[0.])),
        },
        value: models.value,
    };
    let mut learner = Mdpo::new(models, config(5, 0.5), 2).unwrap();
    let frozen = learner.policy();
    let distribution = frozen.distribution((&batch.observations).into()).unwrap();
    let old = to_vec(
        distribution
            .log_prob(&batch.actions, &batch.latents)
            .unwrap(),
    )
    .unwrap();
    learner.update(&batch).unwrap();
    assert_eq!(frozen.running.value_sync().into_scalar::<f32>(), 0.);
    assert_eq!(
        learner.policy().running.value_sync().into_scalar::<f32>(),
        5.
    );
    assert_eq!(
        old,
        to_vec(
            frozen
                .distribution((&batch.observations).into())
                .unwrap()
                .log_prob(&batch.actions, &batch.latents)
                .unwrap()
        )
        .unwrap()
    );
    let snapshot = learner.policy();
    snapshot.running.update(rl::nn::vector(&[99.]));
    assert_eq!(
        learner.policy().running.value_sync().into_scalar::<f32>(),
        5.
    );
}

#[test]
fn nonfinite_optimizer_moments_are_rejected_without_partial_restore() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut learner = Mdpo::new(models, config(3, 0.5), 4).unwrap();
    learner.update(&batch).unwrap();
    let mut bad = learner.checkpoint().unwrap();
    let before = record(&learner);
    let reader = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(
        bad.optimizers["policy"].clone(),
    ))
    .unwrap();
    let scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    let tensor = tensors.remove(0);
    let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
    let mut data = bytes.to_vec();
    data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    tensors.push(burn_pack::Tensor::new(
        name,
        dtype,
        shape,
        id,
        burn_pack::Bytes::from_bytes_vec(data),
    ));
    let mut writer = burn_pack::Writer::new(tensors);
    for (k, v) in scalars {
        writer = writer.with_scalar(&k, v);
    }
    for (k, v) in metadata {
        writer = writer.with_metadata(&k, &v);
    }
    bad.optimizers
        .insert("policy".into(), writer.into_bytes().unwrap().to_vec());
    assert!(
        learner.restore(&bad).is_err(),
        "a NaN Adam moment must not enter the live learner"
    );
    assert_eq!(record(&learner), before);
}

fn rewrite_optimizer(
    record: &mut rl::core::checkpoint::LearnerRecord,
    name: &str,
    edit: impl FnOnce(
        &mut Vec<burn_pack::Tensor>,
        &mut std::collections::BTreeMap<String, burn_pack::Scalar>,
    ),
) {
    let reader = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(
        record.optimizers[name].clone(),
    ))
    .unwrap();
    let mut scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    edit(&mut tensors, &mut scalars);
    let mut writer = burn_pack::Writer::new(tensors);
    for (k, v) in scalars {
        writer = writer.with_scalar(&k, v);
    }
    for (k, v) in metadata {
        writer = writer.with_metadata(&k, &v);
    }
    record
        .optimizers
        .insert(name.into(), writer.into_bytes().unwrap().to_vec());
}
#[test]
fn review_restore_rejects_empty_swapped_and_missing_parameter_optimizer_groups() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let fresh = Mdpo::new(models.clone(), config(3, 0.5), 4)
        .unwrap()
        .checkpoint()
        .unwrap();
    let mut learner = Mdpo::new(models, config(3, 0.5), 4).unwrap();
    learner.update(&batch).unwrap();
    let good = learner.checkpoint().unwrap();
    let before = record(&learner);
    let mut empty = good.clone();
    empty
        .optimizers
        .insert("policy".into(), fresh.optimizers["policy"].clone());
    assert!(
        learner.restore(&empty).is_err(),
        "trained Adam state silently reset to empty"
    );
    assert_eq!(record(&learner), before);
    let mut swapped = good.clone();
    swapped
        .optimizers
        .insert("policy".into(), good.optimizers["value"].clone());
    assert!(
        learner.restore(&swapped).is_err(),
        "critic Adam states accepted as actor states"
    );
    assert_eq!(record(&learner), before);
    let mut missing = good;
    rewrite_optimizer(&mut missing, "policy", |tensors, scalars| {
        let prefix = format!("{}.", tensors[0].param_id.unwrap());
        tensors.retain(|t| !t.name.starts_with(&prefix));
        scalars.retain(|key, _| !key.starts_with(&prefix));
    });
    assert!(
        learner.restore(&missing).is_err(),
        "whole Adam parameter group silently removed"
    );
    assert_eq!(record(&learner), before);
}
#[test]
fn review_restore_rejects_invalid_adam_times_atomically() {
    let (models, batch) = fixture(ActionSpace::Discrete { count: 3 });
    let mut learner = Mdpo::new(models, config(3, 0.5), 4).unwrap();
    learner.update(&batch).unwrap();
    let good = learner.checkpoint().unwrap();
    let before = record(&learner);
    for time in [0, 4, usize::MAX, i32::MAX as usize] {
        let mut bad = good.clone();
        rewrite_optimizer(&mut bad, "policy", |_, scalars| {
            let key = scalars
                .keys()
                .find(|k| k.ends_with(".time"))
                .unwrap()
                .clone();
            scalars.insert(key, burn_pack::Scalar::from(time));
        });
        assert!(
            learner.restore(&bad).is_err(),
            "invalid Adam time accepted: {time}"
        );
        assert_eq!(record(&learner), before);
    }
}
#[test]
fn review_finite_but_unrepresentable_adam_gradient_squares_reject_before_step() {
    for clipping in [None, Some(1.)] {
        let (models, mut batch) = fixture(ActionSpace::Discrete { count: 3 });
        if let OnPolicyMetadata::Bootstrapped(metadata) = &mut batch.metadata {
            for a in &mut metadata.advantages {
                *a *= 1e25;
            }
        }
        let mut c = config(1, 0.5);
        c.policy.max_grad_norm = clipping;
        let mut learner = Mdpo::new(models, c, 4).unwrap();
        let before = record(&learner);
        assert!(
            matches!(learner.update(&batch), Err(Error::Numerical(_))),
            "finite gradients overflowing Adam's f32 squares must reject"
        );
        assert_eq!(record(&learner), before);
    }
}
