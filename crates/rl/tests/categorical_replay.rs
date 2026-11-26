//! Optional recording must preserve collection arithmetic and factual ancestry.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::ActionSpace,
        actor::ActMode,
        categorical_behavior::CategoricalStatefulActor,
        checkpoint::Snapshot,
        env::{EpisodeStart, Observation},
        stateful::{HistoryInput, HistorySpec, StatefulActor},
    },
    data::{
        categorical_replay::CategoricalReplay,
        temporal::{EpisodeStore, SequenceConfig, SequenceSampler},
    },
    envs::delayed_bit::DelayedBit,
    nn::{temporal_categorical_actor::TemporalCategoricalActor, temporal_policy::TemporalPolicy},
};

fn actor(epsilon: f64, revision: u64) -> TemporalCategoricalActor {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    TemporalCategoricalActor::new(
        TemporalPolicy::new(spec, Some(4), &[], &mut ChaCha8Rng::seed_from_u64(12)).unwrap(),
        epsilon,
        revision,
    )
    .unwrap()
}
fn environments() -> SyncVectorEnv<DelayedBit> {
    SyncVectorEnv::new(vec![
        DelayedBit::new(2, None, false, 101).unwrap(),
        DelayedBit::new(3, None, false, 102).unwrap(),
    ])
    .unwrap()
}
fn replay(capacity: usize) -> CategoricalReplay {
    CategoricalReplay::new(capacity, 5, ActionSpace::Discrete { count: 2 }, 2).unwrap()
}

#[test]
fn recorded_sampler_is_the_same_draw_and_likelihood_with_typed_source() {
    let model = actor(0.2, 9);
    let obs = Observation::dense(vec![1., 0., 1., 0., 1.], Some(vec![true, true]));
    let state = model
        .assimilate(
            &model.initial_state().unwrap(),
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let mut ordinary = ChaCha8Rng::seed_from_u64(7);
    let mut recorded = ordinary.clone();
    for _ in 0..40 {
        let a = model
            .decide(&state, &obs, ActMode::Sample, &mut ordinary)
            .unwrap();
        let b = model
            .sample_with_behavior(&state, &obs, &mut recorded)
            .unwrap();
        assert_eq!(a.action, b.decision.action);
        assert_eq!(a.log_prob, b.decision.log_prob);
        b.behavior
            .validate(model.action_space(), &obs, &b.decision)
            .unwrap();
        assert_eq!(b.behavior.revision, 9);
        assert_eq!(b.behavior.source, model.behavior_source());
    }
    assert_eq!(
        serde_json::to_value(ordinary).unwrap(),
        serde_json::to_value(recorded).unwrap()
    );
    assert_eq!(model.behavior_source(), actor(0.2, 90).behavior_source());
    assert_ne!(model.behavior_source(), actor(0.1, 9).behavior_source());
}

#[test]
fn opt_in_recording_preserves_factual_tapes_and_builds_checked_padded_batches() {
    let mut basic = StatefulCollector::new(environments(), actor(0.2, 0), 73).unwrap();
    let mut recording = StatefulCollector::new(environments(), actor(0.2, 0), 73).unwrap();
    let mut plain = EpisodeStore::new(64, 5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rich = replay(64);
    for t in 0..14 {
        if t == 2 {
            basic.publish(actor(0.1, 1)).unwrap();
            recording.publish(actor(0.1, 1)).unwrap();
        }
        let a = basic
            .tick(&[true, true], ActMode::Sample, &mut plain)
            .unwrap();
        let b = recording
            .tick_recorded(&[true, true], ActMode::Sample, &mut rich)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(
            serde_json::to_value(&plain).unwrap(),
            serde_json::to_value(rich.store()).unwrap()
        );
        assert_eq!(basic.observations(), recording.observations());
    }
    rich.validate_snapshot().unwrap();
    assert_eq!(rich.recorded_rows(), plain.len());
    let mut sampler = SequenceSampler::new(91);
    let sequences = sampler
        .sample(
            rich.store(),
            &SequenceConfig {
                burn_in: 1,
                learn_len: 2,
                n_step: 1,
            },
            8,
        )
        .unwrap();
    let batch = rich.batch(sequences.clone()).unwrap();
    assert!(rich.batch_with_limit(sequences.clone(), 1).is_err());
    let mut broken = serde_json::to_value(&rich).unwrap();
    let row = broken["rows"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .find(|rows| !rows.as_array().unwrap().is_empty())
        .unwrap();
    row[0]["revision"] = 999.into();
    assert!(serde_json::from_value::<CategoricalReplay>(broken).is_err());
    let mut signed_zero = sequences.clone();
    assert_eq!(signed_zero.samples[0].observations[0].dense_values()[2], 0.);
    signed_zero.samples[0].observations[0]
        .as_dense_mut()
        .unwrap()
        .values[2] = -0.;
    assert!(
        rich.batch(signed_zero).is_err(),
        "factual observation matching must preserve float bits"
    );
    batch
        .validate(&HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap())
        .unwrap();
    assert!(
        batch.sources.iter().any(Option::is_none),
        "endpoints/padding have no behavior"
    );
    let mut forged = sequences;
    forged.samples[0].observations[0]
        .as_dense_mut()
        .unwrap()
        .values[0] += 0.1;
    assert!(
        rich.batch(forged).is_err(),
        "a matching key cannot authenticate a different observation history"
    );
    let mut forged = batch;
    let i = forged.behavior.iter().position(|p| *p > 0.).unwrap();
    forged.behavior[i] *= 0.5;
    assert!(
        forged
            .validate(&HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap())
            .is_err()
    );
}

#[test]
fn bounded_eviction_and_snapshot_restore_preserve_only_retained_distributions() {
    let mut collector = StatefulCollector::new(environments(), actor(0.15, 0), 73).unwrap();
    let mut rich = replay(8);
    for _ in 0..30 {
        collector
            .tick_recorded(&[true, true], ActMode::Sample, &mut rich)
            .unwrap();
        assert_eq!(rich.recorded_rows(), rich.store().len());
        assert!(rich.recorded_rows() <= 8);
        rich.validate_snapshot().unwrap();
    }
    let mut resumed: CategoricalReplay =
        serde_json::from_value(serde_json::to_value(&rich).unwrap()).unwrap();
    resumed.validate_snapshot().unwrap();
    let mut restored: StatefulCollector<SyncVectorEnv<DelayedBit>, TemporalCategoricalActor> =
        serde_json::from_value(serde_json::to_value(&collector).unwrap()).unwrap();
    restored.validate_snapshot().unwrap();
    for _ in 0..8 {
        collector
            .tick_recorded(&[true, true], ActMode::Sample, &mut rich)
            .unwrap();
        restored
            .tick_recorded(&[true, true], ActMode::Sample, &mut resumed)
            .unwrap();
        assert_eq!(
            serde_json::to_value(&rich).unwrap(),
            serde_json::to_value(&resumed).unwrap()
        );
        assert_eq!(collector.observations(), restored.observations());
    }
}

#[test]
fn an_uncommitted_draw_cannot_become_a_valid_snapshot_by_serialization() {
    use rl::{
        core::recording::{RecordingKey, StatefulRecording},
        core::temporal::TemporalSink,
    };
    let model = actor(0.1, 0);
    let obs = Observation::dense(vec![1.; 5], Some(vec![true, false]));
    let state = model
        .assimilate(
            &model.initial_state().unwrap(),
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let mut store = replay(8);
    store
        .begin_episode(7, &EpisodeStart::fresh(obs.clone()))
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    store
        .decide(
            RecordingKey {
                episode_id: 7,
                timestep: 0,
            },
            &model,
            &state,
            &obs,
            ActMode::Sample,
            &mut rng,
        )
        .unwrap();
    assert!(store.validate_snapshot().is_err());
    assert!(
        serde_json::to_value(&store).is_err(),
        "serialization silently discarded an incomplete sampling transaction"
    );
    store.abandon_episode(7).unwrap();
    store.validate_snapshot().unwrap();
}

#[test]
fn ignored_active_tapes_do_not_accumulate_or_fabricate_behavior() {
    let env = SyncVectorEnv::new(vec![DelayedBit::new(5, None, false, 101).unwrap()]).unwrap();
    let mut collector = StatefulCollector::new(env, actor(0.1, 0), 73).unwrap();
    let mut store = CategoricalReplay::new(2, 5, ActionSpace::Discrete { count: 2 }, 1).unwrap();
    let mut saw_ignored = false;
    for _ in 0..18 {
        collector
            .tick_recorded(&[true], ActMode::Sample, &mut store)
            .unwrap();
        saw_ignored |= store.store().ignored_count() > 0;
        assert_eq!(store.recorded_rows(), store.store().len());
        assert!(store.recorded_rows() <= 2);
        store.validate_snapshot().unwrap();
    }
    assert!(saw_ignored);
}

#[test]
fn resetting_an_ended_lane_preserves_completed_tape_and_behavior() {
    let env = SyncVectorEnv::new(vec![DelayedBit::new(2, None, false, 101).unwrap()]).unwrap();
    let mut collector = StatefulCollector::new(env, actor(0.1, 0), 73).unwrap();
    let mut store = replay(8);
    for _ in 0..3 {
        collector
            .tick_recorded(&[true], ActMode::Sample, &mut store)
            .unwrap();
    }
    assert_eq!(store.recorded_rows(), 3);
    collector.reset_all(&mut store).unwrap();
    store.validate_snapshot().unwrap();
    assert_eq!(store.recorded_rows(), store.store().len());
}

#[test]
fn batch_compatible_actors_have_identical_sampling_source() {
    let plus = actor(0., 0);
    let minus = actor(-0., 0);
    assert!(!plus.batch_compatible(&minus) || plus.behavior_source() == minus.behavior_source());
}

#[test]
fn changed_likelihood_bits_reject_before_factual_commit() {
    use rl::{
        core::recording::{RecordingKey, StatefulRecording},
        core::temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    };
    let model = actor(0.1, 0);
    let obs = Observation::dense(vec![1.; 5], Some(vec![true, false]));
    let state = model
        .assimilate(
            &model.initial_state().unwrap(),
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let mut store = replay(8);
    store
        .begin_episode(7, &EpisodeStart::fresh(obs.clone()))
        .unwrap();
    let decision = store
        .decide(
            RecordingKey {
                episode_id: 7,
                timestep: 0,
            },
            &model,
            &state,
            &obs,
            ActMode::Sample,
            &mut ChaCha8Rng::seed_from_u64(8),
        )
        .unwrap();
    assert_eq!(decision.log_prob.unwrap().to_bits(), 0f32.to_bits());
    let transition = TemporalTransition {
        episode_id: 7,
        timestep: 0,
        observation: obs.clone(),
        action: decision.action,
        reward: 0.0_f32.into(),
        next_observation: obs,
        terminated: false,
        truncated: false,
        behavior: BehaviorMetadata {
            revision: 0,
            log_prob: Some(-0.),
        },
    };
    assert!(store.push(&transition).is_err());
    assert!(store.store().is_empty());
}

#[test]
fn direct_recording_checks_factual_observation_bits_before_sampling() {
    use rl::{
        core::recording::{RecordingKey, StatefulRecording},
        core::temporal::TemporalSink,
    };
    let model = actor(0.1, 0);
    let mut obs = Observation::dense(vec![0.; 5], Some(vec![true, false]));
    let mut replay = replay(8);
    replay
        .begin_episode(7, &EpisodeStart::fresh(obs.clone()))
        .unwrap();
    obs.as_dense_mut().unwrap().values[0] = -0.;
    let state = model
        .assimilate(
            &model.initial_state().unwrap(),
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let before = serde_json::to_value(&rng).unwrap();
    assert!(
        replay
            .decide(
                RecordingKey {
                    episode_id: 7,
                    timestep: 0
                },
                &model,
                &state,
                &obs,
                ActMode::Sample,
                &mut rng
            )
            .is_err()
    );
    assert_eq!(serde_json::to_value(rng).unwrap(), before);
    replay.validate_snapshot().unwrap();
}
