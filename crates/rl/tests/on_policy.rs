#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        ppo::{Ppo, PpoConfig},
        reinforce::{Reinforce, ReinforceConfig},
        vmpo::{Vmpo, VmpoConfig},
    },
    core::{
        action::ActionSpace,
        actor::{EpisodeActor, EpisodeBehavior, OnPolicyActorRef, RolloutActor, RolloutBehavior},
        env::Observation,
        learner::{Learner, OnPolicyLearner},
    },
    data::rollout::{BootstrappedBatch, OnPolicyBatch, OnPolicyMetadata},
    nn::{
        critics::scalar::Value,
        policy::{ActorCritic, Policy},
    },
};
fn fixture(space: ActionSpace) -> (ActorCritic, OnPolicyBatch) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let models = ActorCritic {
        policy: Policy::new(1, space, &[8], &mut rng).unwrap(),
        value: Value::new(1, &[8], &mut rng).unwrap(),
    };
    let obs = vec![Observation::new(vec![1.])];
    let mut out = RolloutBehavior::default();
    models
        .sample_rollout_into((&obs).into(), &mut rng, &mut out)
        .unwrap();
    let b = OnPolicyBatch {
        episode_count: 1,
        observations: obs,
        actions: out.actions,
        latents: out.latents,
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: out.log_probs,
            values: out.values,
            advantages: vec![1.],
        }),
        returns: vec![1.],
        discount_weights: vec![1.],
    };
    (models, b)
}
#[test]
fn all_on_policy_learners_update_discrete_and_continuous_batches() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    ] {
        let (m, b) = fixture(space.clone());
        let mut p = Ppo::new(
            m,
            PpoConfig {
                epochs: 1,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        assert_eq!(p.update(&b).unwrap().updates, 1);
        let (m, mut b) = fixture(space.clone());
        b.metadata = OnPolicyMetadata::Episodic {
            baseline_values: Some(b.bootstrapped().unwrap().values.clone()),
        };
        let mut r = Reinforce::new(m, ReinforceConfig::default(), 3).unwrap();
        assert_eq!(r.update(&b).unwrap().updates, 1);
        let (m, b) = fixture(space);
        let mut v = Vmpo::new(
            m,
            VmpoConfig {
                epochs: 1,
                ..Default::default()
            },
            3,
        )
        .unwrap();
        assert_eq!(v.update(&b).unwrap().updates, 1);
    }
}
#[test]
fn reinforce_rejects_incomplete_returns() {
    let (m, b) = fixture(ActionSpace::Discrete { count: 2 });
    assert!(
        Reinforce::new(m, Default::default(), 0)
            .unwrap()
            .update(&b)
            .is_err()
    );
}

#[test]
fn bootstrapped_algorithms_reject_episodic_metadata_before_update() {
    let (models, mut batch) = fixture(ActionSpace::Discrete { count: 2 });
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: None,
    };
    let mut ppo = Ppo::new(models.clone(), Default::default(), 5).unwrap();
    let mut vmpo = Vmpo::new(models, Default::default(), 5).unwrap();
    assert!(ppo.update(&batch).is_err());
    assert!(vmpo.update(&batch).is_err());
    assert_eq!(ppo.updates(), 0);
    assert_eq!(vmpo.updates(), 0);
    assert!(matches!(
        ppo.collection_actor(),
        OnPolicyActorRef::Bootstrapped(_)
    ));
    assert!(matches!(
        vmpo.collection_actor(),
        OnPolicyActorRef::Bootstrapped(_)
    ));
}

#[test]
fn reinforce_without_baseline_never_forwards_the_value_model() {
    let (mut models, mut batch) = fixture(ActionSpace::Discrete { count: 2 });
    models.value.network =
        rl::nn::network::Mlp::new(1, &[4], 2, &mut ChaCha8Rng::seed_from_u64(90)).unwrap();
    let mut learner = Reinforce::new(
        models,
        ReinforceConfig {
            baseline: false,
            ..Default::default()
        },
        9,
    )
    .unwrap();
    let mut out = EpisodeBehavior::default();
    learner
        .sample_episode_into(
            (&batch.observations).into(),
            &mut ChaCha8Rng::seed_from_u64(7),
            &mut out,
        )
        .unwrap();
    assert!(out.baseline_values.is_none());
    assert_eq!(out.actions.len(), batch.len());
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: None,
    };
    assert_eq!(learner.update(&batch).unwrap().updates, 1);
    assert!(matches!(
        learner.collection_actor(),
        OnPolicyActorRef::Episodic(_)
    ));
}

#[test]
fn reinforce_with_baseline_requires_and_collects_current_values() {
    let (models, mut batch) = fixture(ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    });
    let mut learner = Reinforce::new(models, ReinforceConfig::default(), 9).unwrap();
    let mut out = EpisodeBehavior::default();
    learner
        .sample_episode_into(
            (&batch.observations).into(),
            &mut ChaCha8Rng::seed_from_u64(7),
            &mut out,
        )
        .unwrap();
    assert_eq!(out.baseline_values.as_ref().unwrap().len(), batch.len());
    assert!(out.latents.iter().all(Option::is_some));
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: None,
    };
    assert!(learner.update(&batch).is_err());
    assert_eq!(learner.updates(), 0);
}

#[test]
fn disabled_reinforce_entropy_does_not_advance_training_rng() {
    use rl::core::checkpoint::Checkpoint;
    let (models, mut batch) = fixture(ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    });
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: None,
    };
    let mut learner = Reinforce::new(
        models,
        ReinforceConfig {
            baseline: false,
            entropy: 0.,
            ..Default::default()
        },
        51,
    )
    .unwrap();
    let before = learner.checkpoint().unwrap().metadata["rng"].clone();
    learner.update(&batch).unwrap();
    assert_eq!(before, learner.checkpoint().unwrap().metadata["rng"]);
}

#[test]
fn ppo_early_stop_does_not_draw_entropy_rng_after_likelihood() {
    use rl::core::checkpoint::Checkpoint;
    let (models, mut batch) = fixture(ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    });
    if let OnPolicyMetadata::Bootstrapped(metadata) = &mut batch.metadata {
        metadata.log_probs[0] += 10.;
    }
    let config = PpoConfig {
        epochs: 1,
        minibatch_size: 1,
        target_kl: Some(0.001),
        entropy: 1.,
        ..Default::default()
    };
    let mut with_entropy = Ppo::new(models.clone(), config.clone(), 41).unwrap();
    let mut without_entropy = Ppo::new(
        models,
        PpoConfig {
            entropy: 0.,
            ..config
        },
        41,
    )
    .unwrap();
    assert!(with_entropy.update(&batch).unwrap().early_stopped);
    assert!(without_entropy.update(&batch).unwrap().early_stopped);
    assert_eq!(with_entropy.updates(), 0);
    assert_eq!(
        with_entropy.checkpoint().unwrap().metadata["rng"],
        without_entropy.checkpoint().unwrap().metadata["rng"]
    );
}

#[test]
fn ppo_zero_entropy_does_not_draw_bounded_gaussian_samples() {
    use rl::core::checkpoint::Checkpoint;
    let (models, batch) = fixture(ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    });
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            minibatch_size: 1,
            target_kl: None,
            entropy: 0.,
            ..Default::default()
        },
        59,
    )
    .unwrap();
    let before = learner.checkpoint().unwrap().metadata["rng"].clone();
    assert_eq!(learner.update(&batch).unwrap().updates, 1);
    assert_eq!(learner.checkpoint().unwrap().metadata["rng"], before);
}

#[test]
fn learner_value_inspection_returns_independent_running_state() {
    use burn::{
        module::{Module, RunningState},
        tensor::Tensor,
    };
    use rl::{core::env::ObservationBatch, nn::critics::scalar::StateValue};
    #[derive(Module, Debug)]
    struct BufferedValue {
        running: RunningState<Tensor<1>>,
    }
    impl StateValue for BufferedValue {
        fn values(&self, obs: ObservationBatch<'_>) -> rl::Result<Tensor<1>> {
            Ok(self.running.value_sync().repeat_dim(0, obs.len()))
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(82);
    let learner = Ppo::new(
        ActorCritic {
            policy: Policy::new(1, ActionSpace::Discrete { count: 2 }, &[4], &mut rng).unwrap(),
            value: BufferedValue {
                running: RunningState::new(rl::nn::vector(&[3.])),
            },
        },
        PpoConfig::default(),
        23,
    )
    .unwrap();
    let snapshot = learner.value();
    snapshot.running.update(rl::nn::vector(&[99.]));
    assert_eq!(snapshot.running.value_sync().into_scalar::<f32>(), 99.);
    assert_eq!(
        learner.value().running.value_sync().into_scalar::<f32>(),
        3.
    );
}

#[test]
fn rejected_on_policy_restore_preserves_models_counters_and_rng() {
    use rl::core::checkpoint::Checkpoint;
    let (models, batch) = fixture(ActionSpace::Discrete { count: 2 });
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            ..Default::default()
        },
        31,
    )
    .unwrap();
    learner.update(&batch).unwrap();
    let original = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&original).unwrap();
    let mut invalid_config = original.clone();
    invalid_config.metadata["config"]["epochs"] = 0.into();
    assert!(learner.restore(&invalid_config).is_err());
    assert_eq!(
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        before
    );
    let mut missing_optimizer = original;
    missing_optimizer.optimizers.remove("value");
    assert!(learner.restore(&missing_optimizer).is_err());
    assert_eq!(
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        before
    );
}

#[test]
fn vmpo_restore_rejects_invalid_dual_configuration_without_partial_changes() {
    use rl::core::checkpoint::Checkpoint;
    let (models, _) = fixture(ActionSpace::Discrete { count: 2 });
    let mut learner = Vmpo::new(models, VmpoConfig::default(), 71).unwrap();
    let mut record = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    record.metadata["config"]["temperature_constraint"] = (-1.).into();
    assert!(learner.restore(&record).is_err());
    assert_eq!(
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        before
    );
}

#[test]
fn ppo_clips_positive_but_not_negative_advantages_at_large_ratio() {
    use burn::{module::Param, tensor::Tensor};
    let (mut models, mut b) = fixture(ActionSpace::Discrete { count: 2 });
    for layer in &mut models.policy.network.layers {
        let dims = layer.weight.val().dims();
        layer.weight = Param::from_tensor(Tensor::zeros(dims, &rl::nn::device()));
        if let Some(bias) = &mut layer.bias {
            *bias = Param::from_tensor(Tensor::zeros(bias.val().dims(), &rl::nn::device()));
        }
    }
    b.metadata = OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
        log_probs: vec![0.25_f32.ln()],
        values: vec![0.],
        advantages: vec![1.],
    });
    let mut learner = Ppo::new(
        models.clone(),
        PpoConfig {
            epochs: 1,
            target_kl: None,
            ..Default::default()
        },
        0,
    )
    .unwrap();
    assert!((learner.update(&b).unwrap().metrics["policy_loss"] + 1.2).abs() < 1e-6);
    if let OnPolicyMetadata::Bootstrapped(metadata) = &mut b.metadata {
        metadata.advantages[0] = -1.;
    }
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            target_kl: None,
            ..Default::default()
        },
        0,
    )
    .unwrap();
    assert!((learner.update(&b).unwrap().metrics["policy_loss"] - 2.).abs() < 1e-6);
}
#[test]
fn vmpo_accepts_its_high_dimensional_saturated_behavior_but_rejects_changed_likelihoods() {
    use burn::{module::Param, tensor::Tensor};
    use rl::nn::{device, vector};
    let mut init = ChaCha8Rng::seed_from_u64(1);
    let mut policy = Policy::new(
        1,
        ActionSpace::Box {
            low: vec![-1.; 32],
            high: vec![1.; 32],
        },
        &[],
        &mut init,
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(Tensor::zeros([1, 32], &device()));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[20.; 32])));
    let models = ActorCritic {
        policy,
        value: Value::new(1, &[], &mut init).unwrap(),
    };
    let obs = vec![Observation::new(vec![0.])];
    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let mut behavior = RolloutBehavior::default();
    models
        .sample_rollout_into((&obs).into(), &mut rng, &mut behavior)
        .unwrap();
    let batch = OnPolicyBatch {
        episode_count: 1,
        observations: obs,
        actions: behavior.actions,
        latents: behavior.latents,
        returns: vec![1.],
        discount_weights: vec![1.],
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: behavior.log_probs,
            values: behavior.values,
            advantages: vec![1.],
        }),
    };
    let mut valid = Vmpo::new(
        models.clone(),
        VmpoConfig {
            epochs: 1,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    valid.update(&batch).unwrap();
    let mut changed = batch.clone();
    if let OnPolicyMetadata::Bootstrapped(b) = &mut changed.metadata {
        b.log_probs[0] += 0.1;
    }
    let mut invalid = Vmpo::new(
        models,
        VmpoConfig {
            epochs: 1,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    assert!(invalid.update(&changed).is_err());
    assert_eq!(invalid.updates(), 0);
}
