#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::reppo::{Reppo, ReppoConfig},
    objectives::hl_gauss::HlGaussConfig,
    prelude::*,
};

fn config() -> ReppoConfig {
    ReppoConfig {
        epochs: 2,
        minibatch_size: 2,
        hl_gauss: HlGaussConfig {
            min: -30.,
            max: 10.,
            bins: 21,
            sigma_ratio: 0.75,
        },
        ..Default::default()
    }
}
fn build(
    config: ReppoConfig,
) -> (
    Reppo<Policy, QValue>,
    OnPolicyTrainer<SyncVectorEnv<rl::envs::lqg::Lqg>>,
) {
    use rl::envs::lqg::{Lqg, LqgConfig};
    let env = Lqg::new(
        LqgConfig::scalar(4, 0.8, 0.4, 1., 0.1, 1., 1., 0.05, 2.).unwrap(),
        11,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let p = Policy::new(2, env.action_space().clone(), &[8], &mut rng).unwrap();
    let q = QValue::new(2, 1, config.hl_gauss.bins, &[8], &mut rng).unwrap();
    let learner = Reppo::new(p, q, config, 5).unwrap();
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 6).unwrap();
    let trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 4,
            gamma: 0.9,
            ..Default::default()
        },
    )
    .unwrap();
    (learner, trainer)
}

#[test]
fn continuous_reppo_uses_the_standard_trainer_and_reports_required_diagnostics() {
    let (mut learner, mut trainer) = build(config());
    let summary = trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    assert_eq!(summary.counters.env_steps, 4);
    assert_eq!(summary.counters.updates, 4);
    assert_eq!(learner.updates(), 4);
    let report = summary.last_update.unwrap();
    for key in [
        "hl_gauss_loss",
        "predicted_q",
        "soft_target_mean",
        "entropy",
        "kl_behavior_current",
        "alpha",
        "beta",
        "kl_constrained_fraction",
        "hl_gauss_out_of_support_fraction",
    ] {
        assert!(report.metrics[key].is_finite(), "{key}");
    }
    assert_eq!(report.metrics["target_constructions"], 1.);
    assert_eq!(learner.config().hl_gauss.bins, 21);
}

#[test]
fn entropy_pressure_increases_below_target_and_decreases_when_satisfied() {
    for (target, increase) in [(100., true), (-100., false)] {
        let (mut learner, mut trainer) = build(ReppoConfig {
            entropy_target: target,
            ..config()
        });
        let before = learner.multipliers().0;
        trainer
            .train(&mut learner, Budget::vector_steps(4), None)
            .unwrap();
        assert_eq!(learner.multipliers().0 > before, increase);
    }
}

#[test]
fn historical_rollout_and_invalid_temporal_metadata_fail_without_state_mutation() {
    let (mut learner, trainer) = build(config());
    let mut collector = trainer.collector().clone();
    let mut rollout = RolloutBuffer::default();
    collector
        .collect_pathwise_into(&learner, 4, &mut rollout)
        .unwrap();
    let batch = rollout.to_batch(0.9, None, false).unwrap();
    let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    let mut malformed = batch.clone();
    if let OnPolicyMetadata::Pathwise(p) = &mut malformed.metadata {
        p.rewards.pop();
    }
    assert!(learner.update(&malformed).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    let mut malformed = batch.clone();
    if let OnPolicyMetadata::Pathwise(p) = &mut malformed.metadata {
        p.next_observations[0].as_dense_mut().unwrap().values[0] = f32::NAN;
    }
    assert!(learner.update(&malformed).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    learner.update(&batch).unwrap();
    let after = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    assert!(learner.update(&batch).is_err());
    assert_eq!(
        after,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
}

#[test]
fn another_collection_invalidates_the_prior_fragment_even_without_an_update() {
    let (mut learner, trainer) = build(config());
    let mut collector = trainer.collector().clone();
    let mut rollout = RolloutBuffer::default();
    collector
        .collect_pathwise_into(&learner, 2, &mut rollout)
        .unwrap();
    let prior = rollout.to_batch(0.9, None, false).unwrap();
    collector
        .collect_pathwise_into(&learner, 2, &mut rollout)
        .unwrap();
    let fresh = rollout.to_batch(0.9, None, false).unwrap();
    let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    assert!(learner.update(&prior).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    learner.update(&fresh).unwrap();
}

#[test]
fn config_rejects_nonfinite_duals_lambda_and_scalar_critic_substitution() {
    for config in [
        ReppoConfig {
            initial_alpha: 0.,
            ..config()
        },
        ReppoConfig {
            initial_beta: f32::INFINITY,
            ..config()
        },
        ReppoConfig {
            lambda: 1.1,
            ..config()
        },
        ReppoConfig {
            minibatch_size: 0,
            ..config()
        },
    ] {
        assert!(config.validate().is_err());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let p = Policy::new(1, ActionSpace::Unbounded { dimensions: 1 }, &[], &mut rng).unwrap();
    let q = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    assert!(Reppo::new(p, q, config(), 1).is_err());
}

#[test]
fn checkpoint_restores_dual_optimizers_rng_and_next_full_training_cycle() {
    use rl::train::checkpoint::{load_run, save_run};
    let (mut learner, mut trainer) = build(config());
    trainer
        .train(&mut learner, Budget::vector_steps(5), None)
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    save_run(dir.path().join("run"), &learner, &trainer).unwrap();
    let (mut restored, _) = build(config());
    let mut resumed: OnPolicyTrainer<SyncVectorEnv<rl::envs::lqg::Lqg>> =
        load_run(dir.path().join("run"), &mut restored).unwrap();
    let a = trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let b = resumed
        .train(&mut restored, Budget::vector_steps(4), None)
        .unwrap();
    assert_eq!(learner.updates(), restored.updates());
    assert_eq!(learner.multipliers(), restored.multipliers());
    assert_eq!(
        a.last_update.unwrap().metrics,
        b.last_update.unwrap().metrics
    );
    assert_eq!(
        serde_json::to_value(trainer).unwrap(),
        serde_json::to_value(resumed).unwrap()
    );
    assert_eq!(
        learner.checkpoint().unwrap().metadata,
        restored.checkpoint().unwrap().metadata
    );
}

#[test]
fn rejected_incomplete_or_invalid_checkpoint_is_atomic() {
    let (mut learner, mut trainer) = build(config());
    trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let record = learner.checkpoint().unwrap();
    assert_eq!(record.algorithm, "reppo");
    let before = serde_json::to_value(&record).unwrap();
    let mut invalid = record.clone();
    invalid.optimizers.remove("entropy_dual");
    assert!(learner.restore(&invalid).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    let mut invalid = record.clone();
    invalid.metadata["config"]["initial_alpha"] = serde_json::json!(-1.);
    assert!(learner.restore(&invalid).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    let mut invalid = record;
    invalid.metadata["space"] = serde_json::json!({"Unbounded":{"dimensions":1}});
    assert!(learner.restore(&invalid).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
}

#[test]
fn restore_rejects_optimizer_records_owned_by_another_model_atomically() {
    let (mut learner, mut trainer) = build(config());
    trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let record = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    for (left, right) in [
        ("entropy_dual", "kl_dual"),
        ("policy", "critic"),
        ("entropy_dual", "policy"),
        ("kl_dual", "critic"),
    ] {
        let mut invalid = record.clone();
        invalid
            .optimizers
            .insert(left.into(), record.optimizers[right].clone());
        invalid
            .optimizers
            .insert(right.into(), record.optimizers[left].clone());
        assert!(
            learner.restore(&invalid).is_err(),
            "accepted optimizer swap {left}/{right}"
        );
        assert_eq!(
            before,
            serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
        );
    }
}

#[test]
fn restore_rejects_missing_collection_revision_atomically() {
    let (mut learner, trainer) = build(config());
    let mut collector = trainer.collector().clone();
    let mut rollout = RolloutBuffer::default();
    collector
        .collect_pathwise_into(&learner, 2, &mut rollout)
        .unwrap();
    collector
        .collect_pathwise_into(&learner, 2, &mut rollout)
        .unwrap();
    let mut record = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    record
        .metadata
        .as_object_mut()
        .unwrap()
        .remove("collection_revision");
    assert!(learner.restore(&record).is_err());
    assert_eq!(
        before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
}

#[test]
fn restore_rejects_empty_trained_dual_moments_but_accepts_initial_state() {
    let (mut learner, mut trainer) = build(config());
    let initial = learner.checkpoint().unwrap();
    learner.restore(&initial).unwrap();
    trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let record = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    for name in ["entropy_dual", "kl_dual"] {
        let mut invalid = record.clone();
        invalid
            .optimizers
            .insert(name.into(), initial.optimizers[name].clone());
        assert!(
            learner.restore(&invalid).is_err(),
            "accepted missing trained {name} moments"
        );
        assert_eq!(
            before,
            serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
        );
    }
}

#[test]
fn custom_critic_builds_next_state_targets_once_across_all_epochs() {
    use burn::{module::Module, tensor::Tensor};
    use rl::nn::critics::scalar::ActionValue;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Module, Debug)]
    struct CountingCritic {
        q: QValue,
        #[module(skip)]
        target_calls: Arc<AtomicUsize>,
    }
    impl ActionValue for CountingCritic {
        fn action_dim(&self) -> usize {
            1
        }
        fn output_dim(&self) -> usize {
            self.q.output_dim()
        }
        fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
            if obs.iter().all(|o| o.dense_values()[0] == 9.) {
                self.target_calls.fetch_add(1, Ordering::Relaxed);
            }
            self.q.values(obs, actions)
        }
    }
    impl ReppoCritic for CountingCritic {}
    #[derive(Clone)]
    struct Truncated(ActionSpace);
    impl Env for Truncated {
        fn observation_space(&self) -> ObservationSpace {
            ObservationSpace::Dense(1)
        }
        fn action_space(&self) -> &ActionSpace {
            &self.0
        }
        fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
            Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
        }
        fn step(&mut self, _: &Action) -> Result<Step> {
            Ok(Step {
                observation: Observation::new(vec![9.]),
                reward: 1.0_f32.into(),
                terminated: false,
                truncated: true,
            })
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let cfg = ReppoConfig {
        epochs: 4,
        ..config()
    };
    let policy = Policy::new(1, space.clone(), &[4], &mut rng).unwrap();
    let target_calls = Arc::new(AtomicUsize::new(0));
    let q = CountingCritic {
        q: QValue::new(1, 1, cfg.hl_gauss.bins, &[4], &mut rng).unwrap(),
        target_calls: target_calls.clone(),
    };
    let mut learner = Reppo::new(policy, q, cfg, 2).unwrap();
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![Truncated(space)]).unwrap(), 3).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 4,
            ..Default::default()
        },
    )
    .unwrap();
    trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    assert_eq!(learner.updates(), 8);
    assert_eq!(target_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn kl_multiplier_increases_on_a_violation_and_decreases_under_slack() {
    for (rate, limit, increase) in [(0.1, 1e-8, true), (1e-30, 0.5, false)] {
        let mut cfg = config();
        cfg.epochs = 4;
        cfg.policy.learning_rate = rate;
        cfg.kl_limit = limit;
        let (mut learner, mut trainer) = build(cfg);
        let initial = learner.multipliers().1;
        let summary = trainer
            .train(&mut learner, Budget::vector_steps(4), None)
            .unwrap();
        assert_eq!(learner.multipliers().1 > initial, increase);
        if increase {
            assert!(summary.last_update.unwrap().metrics["kl_constrained_fraction"] > 0.);
        }
    }
}
