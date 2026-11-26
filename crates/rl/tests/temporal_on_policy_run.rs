#![cfg(feature = "neural")]

#[path = "support/resume_fixture.rs"]
mod resume_fixture;

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use resume_fixture::ResumeFixture;
use rl::data::temporal_on_policy::TemporalOnPolicyLearner;
use rl::nn::temporal_actor::TemporalActor;
use rl::{
    algorithms::{
        mdpo::MdpoConfig, ppo::PpoConfig, recurrent_critic::RecurrentUpdateMode,
        recurrent_mdpo::RecurrentMdpo, recurrent_ppo::RecurrentPpo,
        recurrent_reinforce::RecurrentReinforce, recurrent_vmpo::RecurrentVmpo,
        reinforce::ReinforceConfig, vmpo::VmpoConfig,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        checkpoint::{Checkpoint, Snapshot},
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
        stateful_on_policy::StatefulOnPolicyActor,
    },
    data::{temporal::SequenceConfig, temporal_on_policy::TemporalBatchKind},
    envs::delayed_bit::DelayedBit,
    nn::{temporal_policy::TemporalPolicy, temporal_value::TemporalValue},
    prelude::*,
    train::{
        checkpoint::{load_run, load_run_portable, save_run},
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};

type Trainer = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>;
fn learner(seed: u64) -> RecurrentPpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentPpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
        PpoConfig {
            epochs: 2,
            minibatch_size: 4,
            target_kl: None,
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}

#[test]
fn numerical_update_modes_reflect_model_capability() -> rl::Result<()> {
    use RecurrentUpdateMode::{OptimizedExact, Reference};
    let mut recurrent = learner(11);
    recurrent.set_update_mode(OptimizedExact);
    assert!(recurrent.supports_update_mode(OptimizedExact));
    assert_eq!(recurrent.update_mode(), OptimizedExact);

    let mut rng = ChaCha8Rng::seed_from_u64(13);
    let mut observation_only = RecurrentPpo::new(
        recurrent.policy(),
        TemporalValue::new(recurrent.policy().spec().clone(), None, &[], &mut rng)?,
        PpoConfig::default(),
        17,
    )?;
    observation_only.set_update_mode(OptimizedExact);
    assert!(!observation_only.supports_update_mode(OptimizedExact));
    assert_eq!(observation_only.requested_update_mode(), OptimizedExact);
    assert_eq!(observation_only.update_mode(), Reference);

    let mut baseline_free = RecurrentReinforce::new(
        recurrent.policy(),
        None,
        ReinforceConfig {
            baseline: false,
            ..Default::default()
        },
        19,
    )?;
    baseline_free.set_update_mode(OptimizedExact);
    assert!(!baseline_free.supports_update_mode(OptimizedExact));
    assert_eq!(baseline_free.update_mode(), Reference);
    Ok(())
}

fn build(seed: u64, delays: &[usize]) -> (RecurrentPpo, Trainer) {
    let l = learner(seed);
    let env = SyncVectorEnv::new(
        delays
            .iter()
            .enumerate()
            .map(|(i, &d)| DelayedBit::new(d, Some(d / 2), false, 100 + i as u64).unwrap())
            .collect(),
    )
    .unwrap();
    let cfg = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 2,
            gamma: 0.99,
            gae_lambda: Some(0.95),
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 1,
            learn_len: 2,
            n_step: 1,
        },
        capacity: 8192,
        max_fresh: 1024,
        ..Default::default()
    };
    let t = TemporalOnPolicyTrainer::new(env, &l, cfg, 47).unwrap();
    (l, t)
}
fn weights(model: &impl Module) -> Vec<f32> {
    struct Values(Vec<f32>);
    impl ModuleVisitor for Values {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0
                .extend(p.val().to_data().try_to_vec::<f32>().unwrap());
        }
    }
    let mut out = Values(vec![]);
    model.visit(&mut out);
    out.0
}
fn assert_current_states(t: &Trainer) {
    for lane in 0..t.collector().len() {
        if t.collector().lane_needs_reset(lane) == Some(true) {
            continue;
        }
        let actor = t.collector().lane_actor(lane).unwrap();
        let id = t.collector().lane_episode(lane).unwrap();
        let episode = t.rollout().store().episode(id).unwrap();
        let mut state = actor.initial_state().unwrap();
        for (time, obs) in episode.observations.iter().enumerate() {
            let previous = time.checked_sub(1).map(|i| PreviousInput {
                factual: None,
                action: &episode.steps[i].action,
                reward: episode.steps[i].reward.total(),
            });
            state = actor
                .assimilate(
                    &state,
                    HistoryInput {
                        observation: obs,
                        previous,
                    },
                )
                .unwrap();
        }
        assert_eq!(t.collector().lane_state(lane), Some(&state));
    }
}

#[test]
fn recurrent_ppo_restores_current_actor_and_learns_only_the_fresh_suffix() {
    let mut learner = learner(11);
    let environment = ResumeFixture::new(DelayedBit::new(3, None, false, 100).unwrap(), Some(3));
    let env = SyncVectorEnv::new(vec![environment]).unwrap();
    let config = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 4,
            gamma: 0.99,
            gae_lambda: Some(0.95),
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 1,
            learn_len: 2,
            n_step: 1,
        },
        ..Default::default()
    };
    let mut trainer = TemporalOnPolicyTrainer::new(env, &learner, config, 47).unwrap();
    for _ in 0..4 {
        trainer.collect_tick(&learner).unwrap();
    }
    let original = trainer.collector().lane_episode(0).unwrap();
    let prefix = trainer.rollout().store().episode(original).unwrap().clone();
    let updated = trainer.update(&mut learner).unwrap();
    assert!(updated.updates > 0);
    assert_eq!(learner.revision(), 1);

    trainer.collect_tick(&learner).unwrap();
    let continuation = trainer.collector().lane_episode(0).unwrap();
    assert_ne!(continuation, original);
    assert_eq!(trainer.collector().lane_revision(0), Some(1));
    assert_eq!(trainer.collector().lane_timestep(0), Some(4));
    let actor = trainer.collector().lane_actor(0).unwrap();
    let mut state = actor.initial_state().unwrap();
    for time in 0usize..=3 {
        state = actor
            .assimilate(
                &state,
                HistoryInput {
                    observation: &prefix.observations[time],
                    previous: time.checked_sub(1).map(|index| PreviousInput {
                        factual: None,
                        action: &prefix.steps[index].action,
                        reward: prefix.steps[index].reward.total(),
                    }),
                },
            )
            .unwrap();
    }
    let expected = actor
        .sample_on_policy(
            &state,
            &prefix.observations[3],
            ActMode::Deterministic,
            &mut ChaCha8Rng::seed_from_u64(9),
        )
        .unwrap();
    let batch = trainer.seal(&learner).unwrap();
    assert_eq!(batch.rows.len(), 1);
    let row = &batch.rows[0];
    assert_eq!(row.key.episode_id, continuation);
    assert_eq!(row.key.timestep, 3);
    assert_eq!(row.distribution, expected.distribution);
    assert_eq!(row.value, expected.value);
    let sample = &batch.sequences.samples[row.sample];
    assert_eq!(sample.learn_start, 3);
    assert_eq!(sample.loss_mask, [false, false, false, true]);
    assert!(sample.steps[3].terminated);
    assert_eq!(row.return_, Some(sample.steps[3].reward.total()));
    assert!(trainer.update(&mut learner).unwrap().updates > 0);
    trainer.validate_snapshot().unwrap();
}

#[test]
fn checkpointed_ppo_modes_match_across_truncation_and_snapshot_restore() {
    type Restart = ResumeFixture<TimeLimit<DelayedBit>>;
    type RestartTrainer = TemporalOnPolicyTrainer<SyncVectorEnv<Restart>>;
    let make = |mode| {
        let mut model = learner(29);
        model.set_update_mode(mode);
        let envs = [(3, 8), (5, 4)]
            .into_iter()
            .enumerate()
            .map(|(lane, (delay, limit))| {
                ResumeFixture::new(
                    TimeLimit::new(
                        DelayedBit::new(delay, None, false, 11 + lane as u64).unwrap(),
                        limit,
                    )
                    .unwrap(),
                    Some(3),
                )
            })
            .collect();
        let trainer = TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(envs).unwrap(),
            &model,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 4,
                    normalize_advantages: false,
                    ..Default::default()
                },
                sequence: SequenceConfig {
                    burn_in: 1,
                    learn_len: 2,
                    n_step: 1,
                },
                ..Default::default()
            },
            54,
        )
        .unwrap();
        (model, trainer)
    };
    let (mut reference, mut left) = make(RecurrentUpdateMode::Reference);
    let (mut optimized, mut right) = make(RecurrentUpdateMode::OptimizedExact);
    for _ in 0..4 {
        left.collect_tick(&reference).unwrap();
        right.collect_tick(&optimized).unwrap();
    }
    assert_eq!(
        left.update(&mut reference).unwrap().metrics,
        right.update(&mut optimized).unwrap().metrics
    );
    left.collect_tick(&reference).unwrap();
    right.collect_tick(&optimized).unwrap();

    let collector = right.collector();
    let truncated_successor = collector
        .lane_actor(1)
        .unwrap()
        .state_value(
            collector.lane_state(1).unwrap(),
            &collector.observations()[1],
        )
        .unwrap()
        .unwrap();

    let left_batch = left.seal(&reference).unwrap();
    let right_batch = right.seal(&optimized).unwrap();
    assert_eq!(left_batch.rows, right_batch.rows);
    assert_eq!(left_batch.sequences, right_batch.sequences);
    assert_eq!(right_batch.rows.len(), 2);
    for row in &right_batch.rows {
        assert_eq!(row.key.timestep, 3);
        let sample = &right_batch.sequences.samples[row.sample];
        let step = &sample.steps[row.time];
        assert_eq!(sample.loss_mask, [false, false, false, true]);
        assert_eq!(
            row.next_value,
            Some(if step.truncated {
                truncated_successor
            } else {
                0.
            })
        );
        assert!(step.terminated || step.truncated);
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("checkpointed-ppo");
    save_run(&path, &optimized, &right).unwrap();
    let mut restored = learner(75);
    let mut loaded: RestartTrainer = load_run(&path, &mut restored).unwrap();
    for round in 0..4 {
        if round > 0 {
            left.collect_tick(&reference).unwrap();
            right.collect_tick(&optimized).unwrap();
            loaded.collect_tick(&restored).unwrap();
        }
        let expected = left.update(&mut reference).unwrap();
        assert_eq!(
            expected.metrics,
            right.update(&mut optimized).unwrap().metrics
        );
        assert_eq!(
            expected.metrics,
            loaded.update(&mut restored).unwrap().metrics
        );
        assert_eq!(weights(&reference.policy()), weights(&restored.policy()));
        assert_eq!(weights(&reference.value()), weights(&restored.value()));
        assert_eq!(right.counters(), loaded.counters());
        for lane in 0..2 {
            assert_eq!(
                left.collector().lane_state(lane),
                loaded.collector().lane_state(lane)
            );
        }
    }
    loaded.validate_snapshot().unwrap();
}

#[test]
fn checkpointed_vmpo_mdpo_and_reinforce_fit_complete_fresh_suffixes() {
    fn exercise(mut model: impl TemporalOnPolicyLearner<TemporalActor>) {
        let env = ResumeFixture::new(DelayedBit::new(3, None, false, 71).unwrap(), Some(3));
        let episodic = model.collection_kind() == TemporalBatchKind::Episodic;
        let mut trainer = TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(vec![env]).unwrap(),
            &model,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 4,
                    episode_guard: episodic.then_some(4),
                    gae_lambda: (!episodic).then_some(0.95),
                    gamma: 0.5,
                    normalize_advantages: false,
                },
                sequence: SequenceConfig {
                    burn_in: 1,
                    learn_len: 2,
                    n_step: 1,
                },
                ..Default::default()
            },
            47,
        )
        .unwrap();
        for _ in 0..4 {
            trainer.collect_tick(&model).unwrap();
        }
        trainer.update(&mut model).unwrap();
        trainer.collect_tick(&model).unwrap();
        let batch = trainer.seal(&model).unwrap();
        assert_eq!(batch.rows.len(), 1);
        assert_eq!(batch.rows[0].key.timestep, 3);
        assert_eq!(batch.rows[0].discount_weight, 0.125);
        let sample = &batch.sequences.samples[batch.rows[0].sample];
        assert_eq!(batch.rows[0].return_, Some(sample.steps[3].reward.total()));
        assert_eq!(sample.loss_mask, [false, false, false, true]);
        let report = trainer.update(&mut model).unwrap();
        assert!(report.updates > 0);
        trainer.validate_snapshot().unwrap();
    }

    let pair = learner(19);
    let mut vmpo = RecurrentVmpo::new(
        pair.policy(),
        pair.value(),
        VmpoConfig {
            epochs: 1,
            ..Default::default()
        },
        22,
    )
    .unwrap();
    vmpo.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    exercise(vmpo);
    let mut mdpo = RecurrentMdpo::new(
        pair.policy(),
        pair.value(),
        MdpoConfig {
            inner_steps: 1,
            value_epochs: 1,
            ..Default::default()
        },
        22,
    )
    .unwrap();
    mdpo.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    exercise(mdpo);
    for baseline in [false, true] {
        let mut reinforce = RecurrentReinforce::new(
            pair.policy(),
            baseline.then(|| pair.value()),
            ReinforceConfig {
                baseline,
                ..Default::default()
            },
            22,
        )
        .unwrap();
        reinforce.set_update_mode(RecurrentUpdateMode::OptimizedExact);
        exercise(reinforce);
    }
}

#[test]
fn updates_reconcile_both_memories_without_physical_episode_reset() {
    let (mut l, mut t) = build(11, &[19, 23, 29, 31]);
    for _ in 0..2 {
        t.collect_tick(&l).unwrap();
    }
    let ids: Vec<_> = (0..4).map(|i| t.collector().lane_episode(i)).collect();
    let report = t.update(&mut l).unwrap();
    assert!(report.updates > 0);
    assert_eq!(l.revision(), 1);
    for (i, id) in ids.into_iter().enumerate() {
        assert_eq!(t.collector().lane_episode(i), id);
        assert_eq!(t.collector().lane_timestep(i), Some(2));
        assert_eq!(t.collector().lane_revision(i), Some(1));
    }
    assert_eq!(t.rollout().fresh_len(), 0);
    assert_eq!(t.collector().compaction_count(), 4);
    assert_current_states(&t);
    t.validate_snapshot().unwrap();
}

#[test]
fn partial_rollout_and_sealed_batch_restore_exact_future_trajectories_and_updates() {
    check_partial_rollout_and_sealed_batch_restore(RecurrentUpdateMode::Reference);
}

#[test]
fn optimized_ppo_restores_partial_and_sealed_rollouts_exactly() {
    check_partial_rollout_and_sealed_batch_restore(RecurrentUpdateMode::OptimizedExact);
}

fn check_partial_rollout_and_sealed_batch_restore(mode: RecurrentUpdateMode) {
    for sealed in [false, true] {
        let (mut l, mut t) = build(11, &[7, 11, 13, 17]);
        l.set_update_mode(mode);
        for _ in 0..3 {
            t.collect_tick(&l).unwrap();
        }
        if sealed {
            t.seal(&l).unwrap();
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run.json");
        save_run(&path, &l, &t).unwrap();
        let mut other_l = learner(99);
        let mut other: Trainer = load_run(&path, &mut other_l).unwrap();
        assert_eq!(other_l.update_mode(), mode);
        for round in 0..5 {
            if !sealed || round > 0 {
                for _ in 0..2 {
                    t.collect_tick(&l).unwrap();
                    other.collect_tick(&other_l).unwrap();
                }
            }
            let left = t.update(&mut l).unwrap();
            let right = other.update(&mut other_l).unwrap();
            assert_eq!(left.updates, right.updates);
            assert_eq!(left.metrics, right.metrics);
            assert_eq!(weights(&l.policy()), weights(&other_l.policy()));
            assert_eq!(weights(&l.value()), weights(&other_l.value()));
            assert_eq!(t.counters(), other.counters());
            for lane in 0..4 {
                assert_eq!(
                    t.collector().lane_state(lane),
                    other.collector().lane_state(lane)
                );
                assert_eq!(
                    t.collector().lane_episode(lane),
                    other.collector().lane_episode(lane)
                );
            }
            assert_current_states(&t);
            assert_current_states(&other);
        }
    }
}

fn save_corrupt_snapshot(
    path: &std::path::Path,
    l: &RecurrentPpo,
    t: &Trainer,
    snapshot: serde_json::Value,
) {
    // Keep the genuine run type and all envelope fields; only replace the
    // deliberate corruption and recompute the transport checksum.
    save_run(path, l, t).unwrap();
    let file = path.join("checkpoint.json");
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    envelope["snapshot"] = snapshot;
    let bytes = serde_json::to_vec(&envelope).unwrap();
    let checksum = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    });
    std::fs::write(&file, &bytes).unwrap();
    std::fs::write(
        path.join("COMPLETE"),
        format!("{}:{checksum:016x}", bytes.len()),
    )
    .unwrap();
}
#[test]
fn corrupt_memory_and_collection_binding_fail_before_learner_installation() {
    let (mut l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    t.update(&mut l).unwrap();
    let saved = serde_json::to_value(&t).unwrap();
    let mut altered = saved.clone();
    let memory = &mut altered["collector"]["agents"][0]["state"]["policy"][0];
    *memory = serde_json::json!(memory.as_f64().unwrap() + 0.125);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corrupt.json");
    save_corrupt_snapshot(&path, &l, &t, altered);
    let mut destination = learner(47);
    let before = destination.checkpoint().unwrap();
    assert!(load_run::<Trainer>(&path, &mut destination).is_err());
    assert!(
        serde_json::to_value(before).unwrap()
            == serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
        "failed restore changed learner"
    );
    let mut altered = saved;
    altered["config"]["sequence"]["n_step"] = 2.into();
    let config_path = dir.path().join("bad-config.json");
    save_corrupt_snapshot(&config_path, &l, &t, altered);
    assert!(load_run::<Trainer>(&config_path, &mut destination).is_err());
}

#[test]
fn restored_rollout_cannot_substitute_its_resource_limits() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let saved = serde_json::to_value(&t).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for field in ["max_fresh", "max_metadata_bytes", "max_resident_bytes"] {
        let mut altered = saved.clone();
        let limit = if field == "max_resident_bytes" {
            &mut altered["rollout"]["store"]["storage"]["FullPrefix"][field]
        } else {
            &mut altered["rollout"][field]
        };
        *limit = (limit.as_u64().expect("serialized resource cap") + 1).into();
        let path = dir.path().join(field);
        save_corrupt_snapshot(&path, &l, &t, altered);
        let mut destination = learner(47);
        assert!(
            load_run::<Trainer>(&path, &mut destination).is_err(),
            "{field}"
        );
    }
}

#[test]
fn long_episode_survives_more_than_128_fresh_policy_updates() {
    let (mut l, mut t) = build(11, &[1024]);
    for _ in 0..131 {
        t.collect_tick(&l).unwrap();
        t.update(&mut l).unwrap();
    }
    assert_eq!(t.counters().episodes, 0);
    assert_eq!(t.collector().lane_episode(0), Some(0));
    assert_eq!(t.collector().lane_timestep(0), Some(131));
    assert_eq!(l.revision(), 131);
    assert_eq!(t.collector().compaction_count(), 131);
    assert_current_states(&t);
    t.validate_snapshot().unwrap();
}

#[test]
fn completed_update_cannot_reuse_empty_data_and_next_collection_remains_usable() {
    let (mut l, mut t) = build(11, &[9]);
    t.collect_tick(&l).unwrap();
    t.update(&mut l).unwrap();
    let before = serde_json::to_value(&t).unwrap();
    assert!(t.update(&mut l).is_err());
    assert_eq!(serde_json::to_value(&t).unwrap(), before);
    t.collect_tick(&l).unwrap();
    t.update(&mut l).unwrap();
    assert_eq!(l.revision(), 2);
}

#[test]
fn deserialized_wrong_finite_memory_is_rejected_before_the_next_transition() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut encoded = serde_json::to_value(&t).unwrap();
    let memory = &mut encoded["collector"]["agents"][0]["state"]["policy"][0];
    *memory = serde_json::json!(memory.as_f64().unwrap() + 0.125);
    let mut restored: Trainer = serde_json::from_value(encoded).unwrap();
    let before = restored.counters().env_steps;
    assert!(
        restored.collect_tick(&l).is_err(),
        "a finite but incorrect memory advanced the environment"
    );
    assert_eq!(restored.counters().env_steps, before);
}

#[test]
fn forged_finite_collected_values_fail_before_installing_learner_parameters() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let original = serde_json::to_value(&t).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let good_path = dir.path().join("valid.json");
    save_run(&good_path, &l, &t).unwrap();
    let mut control = learner(99);
    let _: Trainer = load_run(&good_path, &mut control).expect("uncorrupted control must restore");
    for field in ["value", "next_value"] {
        let path = dir.path().join(format!("bad-{field}.json"));
        let mut encoded = original.clone();
        let number = encoded["rollout"]["fresh"][0][field].as_f64().unwrap();
        encoded["rollout"]["fresh"][0][field] = serde_json::json!(number + 0.25);
        save_corrupt_snapshot(&path, &l, &t, encoded);
        let mut destination = learner(99);
        let before = destination.checkpoint().unwrap();
        assert!(
            load_run::<Trainer>(&path, &mut destination).is_err(),
            "forged finite {field} was restored"
        );
        assert!(
            serde_json::to_value(before).unwrap()
                == serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
            "invalid metadata changed destination parameters"
        );
    }
}

#[test]
fn forged_reset_flag_cannot_turn_a_nonterminal_transition_into_an_episode_boundary() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut snapshot = serde_json::to_value(&t).unwrap();
    assert_eq!(snapshot["collector"]["driver"]["needs_reset"][0], false);
    snapshot["collector"]["driver"]["needs_reset"][0] = true.into();
    let forged: Trainer = serde_json::from_value(snapshot.clone()).unwrap();
    assert!(
        forged.validate_snapshot().is_err(),
        "a forged reset flag created an artificial boundary"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("forged-reset.json");
    save_corrupt_snapshot(&path, &l, &t, snapshot);
    let mut destination = learner(99);
    let before = destination.checkpoint().unwrap();
    assert!(load_run::<Trainer>(&path, &mut destination).is_err());
    assert!(
        serde_json::to_value(before).unwrap()
            == serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
        "forged boundary changed learner"
    );
}

#[test]
fn finite_driver_observation_must_match_the_recorded_current_observation() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut snapshot = serde_json::to_value(&t).unwrap();
    let value = &mut snapshot["collector"]["driver"]["observations"][0]["values"][0];
    *value = serde_json::json!(value.as_f64().unwrap() + 0.125);
    let forged: Trainer = serde_json::from_value(snapshot).unwrap();
    assert!(
        forged.validate_snapshot().is_err(),
        "driver and recorded observations diverged undetected"
    );
}

#[test]
fn forged_lane_begin_flag_is_rejected_before_duplicate_episode_begin() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut snapshot = serde_json::to_value(&t).unwrap();
    snapshot["collector"]["agents"][0]["begun"] = false.into();
    let forged: Trainer = serde_json::from_value(snapshot).unwrap();
    assert!(
        forged.validate_snapshot().is_err(),
        "retained episode accepted a false begin flag"
    );
}

#[test]
fn forged_pending_decision_is_not_a_resumable_trainer_boundary() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut rollout = t.rollout().clone();
    let actor = t.collector().lane_actor(0).unwrap();
    let sample = actor
        .sample_on_policy(
            t.collector().lane_state(0).unwrap(),
            &t.collector().observations()[0],
            ActMode::Sample,
            &mut ChaCha8Rng::seed_from_u64(81),
        )
        .unwrap();
    rollout
        .stage_decision(
            rl::core::recording::RecordingKey {
                episode_id: t.collector().lane_episode(0).unwrap(),
                timestep: 1,
            },
            actor.revision(),
            &sample,
        )
        .unwrap();
    rollout.validate_snapshot().unwrap(); // standalone staging is valid
    let mut snapshot = serde_json::to_value(&t).unwrap();
    snapshot["rollout"] = serde_json::to_value(rollout).unwrap();
    let forged: Trainer = serde_json::from_value(snapshot).unwrap();
    assert!(
        forged.validate_snapshot().is_err(),
        "half a collector tick was treated as resumable"
    );
}

#[test]
fn forged_fragment_clock_cannot_change_the_next_rollout_boundary() {
    let (l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    let mut snapshot = serde_json::to_value(&t).unwrap();
    snapshot["collection_ticks"] = 0.into();
    let forged: Trainer = serde_json::from_value(snapshot).unwrap();
    assert!(
        forged.validate_snapshot().is_err(),
        "fragment clock diverged from committed transitions"
    );
}

fn perturb_memory(snapshot: &mut serde_json::Value, role: &str, ulps: u32) {
    let value = &mut snapshot["collector"]["agents"][0]["state"][role][0];
    let original = value.as_f64().unwrap() as f32;
    *value = serde_json::json!(f32::from_bits(original.to_bits() + ulps));
}

#[test]
fn portable_import_is_explicit_bounded_and_canonicalizes_only_runtime_memory() {
    // Cover fresh records, a sealed pending batch and an updated continuing episode.
    for boundary in 0..3 {
        let (mut l, mut t) = build(11, &[19, 23, 29, 31]);
        t.collect_tick(&l).unwrap();
        if boundary == 1 {
            t.seal(&l).unwrap();
        } else if boundary == 2 {
            t.update(&mut l).unwrap();
        }
        let original = serde_json::to_value(&t).unwrap();
        for role in ["policy", "value"] {
            for ulps in [1, 4] {
                let mut altered = original.clone();
                perturb_memory(&mut altered, role, ulps);
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("foreign-runtime");
                save_corrupt_snapshot(&path, &l, &t, altered);
                let mut destination = learner(47);
                let before = serde_json::to_value(destination.checkpoint().unwrap()).unwrap();
                assert!(load_run::<Trainer>(&path, &mut destination).is_err());
                assert_eq!(
                    serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
                    before
                );
                let (mut native, report): (Trainer, _) =
                    load_run_portable(&path, &mut destination).unwrap();
                assert_eq!(report.version, 1);
                assert_eq!(
                    report
                        .lanes
                        .iter()
                        .flat_map(|lane| &lane.memories)
                        .map(|m| m.adjusted_values)
                        .sum::<usize>(),
                    1
                );
                // Everything except the deliberately perturbed memory is preserved.
                assert_eq!(serde_json::to_value(&native).unwrap(), original);
                native.validate_snapshot().unwrap();
                assert_current_states(&native);
                let native_path = dir.path().join("native");
                save_run(&native_path, &destination, &native).unwrap();
                let mut strict_destination = learner(99);
                let strict: Trainer = load_run(&native_path, &mut strict_destination).unwrap();
                assert_eq!(serde_json::to_value(&strict).unwrap(), original);
                // A subsequent update uses ordinary strict semantics and valid provenance.
                if native.pending_batch().is_some() || native.rollout().fresh_len() > 0 {
                    native.update(&mut destination).unwrap();
                } else {
                    native.collect_tick(&destination).unwrap();
                }
            }
        }
    }
}

#[test]
fn portable_import_rejects_just_over_bound_large_and_structural_corruptions_atomically() {
    let (mut l, mut t) = build(11, &[19]);
    t.collect_tick(&l).unwrap();
    t.update(&mut l).unwrap();
    let original = serde_json::to_value(&t).unwrap();
    for corruption in 0..11 {
        let mut altered = original.clone();
        match corruption {
            0 => perturb_memory(&mut altered, "policy", 5),
            1 => perturb_memory(&mut altered, "value", 5),
            2 | 3 => {
                let role = if corruption == 2 { "policy" } else { "value" };
                let value = &mut altered["collector"]["agents"][0]["state"][role][0];
                *value = serde_json::json!(value.as_f64().unwrap() + 0.125);
            }
            4 => altered["collector"]["agents"][0]["state"]["policy"] = serde_json::json!([0.]),
            5 => {
                altered["collector"]["agents"][0]["actor"]["metadata"]["revision"] =
                    serde_json::json!(999)
            }
            6 => {
                altered["collector"]["driver"]["observations"][0]["values"][0] =
                    serde_json::json!(0.25)
            }
            7 => {
                altered["rollout"]["store"]["episodes"][0]["steps"][0]["reward"] =
                    serde_json::json!(0.125)
            }
            8 => altered["version"] = serde_json::json!(999),
            9 => {
                altered["collector"]["agents"][0]["actor"]["metadata"]["policy"]["hidden_dim"] =
                    serde_json::json!(4)
            }
            _ => {
                let action = &mut altered["rollout"]["store"]["episodes"][0]["steps"][0]["action"];
                let original: Action = serde_json::from_value(action.clone()).unwrap();
                let Action::Discrete(index) = original else {
                    unreachable!()
                };
                *action = serde_json::to_value(Action::Discrete(1 - index)).unwrap();
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupted");
        save_corrupt_snapshot(&path, &l, &t, altered);
        let mut destination = learner(47);
        let before = serde_json::to_value(destination.checkpoint().unwrap()).unwrap();
        assert!(
            load_run_portable::<Trainer>(&path, &mut destination).is_err(),
            "accepted corruption {corruption}"
        );
        assert_eq!(
            serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
            before,
            "partial installation for corruption {corruption}"
        );
    }
}
