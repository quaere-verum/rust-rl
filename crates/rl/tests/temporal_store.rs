use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        env::{EpisodeStart, Observation, PublicHistory, ResumeContext},
    },
    data::temporal::{
        BehaviorMetadata, EpisodeStore, ResourceLimits, SequenceBatch, SequenceConfig,
        SequenceSampler, TemporalSink, TemporalStorage, TemporalTransition, sample_at,
    },
};

fn space() -> ActionSpace {
    ActionSpace::Discrete { count: 2 }
}

fn observation(value: f32) -> Observation {
    Observation::dense(vec![value], Some(vec![true, true]))
}

fn transition(id: u64, t: usize, terminated: bool, truncated: bool) -> TemporalTransition {
    TemporalTransition {
        episode_id: id,
        timestep: t,
        observation: observation(t as f32),
        action: Action::Discrete(t % 2),
        reward: ((t + 1) as f32).into(),
        next_observation: observation((t + 1) as f32),
        terminated,
        truncated,
        behavior: BehaviorMetadata {
            revision: 7,
            log_prob: Some(-0.5),
        },
    }
}

fn episode(store: &mut EpisodeStore, id: u64, len: usize, ending: Option<bool>) {
    store
        .begin_episode(id, &EpisodeStart::fresh(observation(0.)))
        .unwrap();
    for t in 0..len {
        let last = t + 1 == len;
        store
            .push(&transition(
                id,
                t,
                last && ending == Some(true),
                last && ending == Some(false),
            ))
            .unwrap();
    }
}

#[test]
fn fragment_appends_keep_one_complete_prefix_and_the_actual_final_observation() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    store
        .begin_episode(10, &EpisodeStart::fresh(observation(0.)))
        .unwrap();
    store.push(&transition(10, 0, false, false)).unwrap();
    store.push(&transition(10, 1, false, true)).unwrap();
    let e = store.episode(10).unwrap();
    assert_eq!(store.len(), 2);
    assert_eq!(
        e.observations
            .iter()
            .map(|o| o.dense_values()[0])
            .collect::<Vec<_>>(),
        [0., 1., 2.]
    );
    assert_eq!(
        e.steps.iter().map(|s| s.reward.total()).collect::<Vec<_>>(),
        [1., 2.]
    );
    assert!(e.steps[1].truncated);
    assert!(!e.steps[1].terminated);
    assert!(store.push(&transition(10, 2, false, false)).is_err());
    store.validate_snapshot().unwrap();
}

#[test]
fn resumed_context_is_retained_but_never_sampled_as_fresh_replay() {
    let mut history = PublicHistory::new(observation(0.));
    for t in 0..3 {
        history = history
            .append(Action::Discrete(t % 2), 100., observation((t + 1) as f32))
            .unwrap();
    }
    let start = EpisodeStart {
        observation: observation(3.),
        context: Some(ResumeContext {
            history,
            physical_timestep: 3,
        }),
    };
    let config = SequenceConfig {
        burn_in: 2,
        learn_len: 1,
        n_step: 2,
    };
    let limits = ResourceLimits {
        max_transitions: 2,
        max_resident_bytes: 16_384,
        max_anchor_bytes: 0,
        max_anchors: 0,
        max_anchor_payload_bytes: 64,
        max_dimension_rank: 4,
    };
    for storage in [
        TemporalStorage::ExactEpisodes,
        TemporalStorage::FullPrefix(limits),
    ] {
        let mut store = EpisodeStore::with_storage(2, 1, space(), storage).unwrap();
        store.begin_episode(4, &start).unwrap();
        assert_eq!(store.len(), 0);
        assert_eq!(store.active_timestep(4), Some(3));
        store.push(&transition(4, 3, false, false)).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.eligible_starts(2).unwrap(), 0);
        assert!(sample_at(&store, 4, 2, &config).is_err());
        store.push(&transition(4, 4, false, true)).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.eligible_starts(2).unwrap(), 2);
        let json = serde_json::to_value(&store).unwrap();
        let mut old_version = json.clone();
        old_version["version"] = serde_json::json!(2);
        assert!(serde_json::from_value::<EpisodeStore>(old_version).is_err());
        let restored: EpisodeStore = serde_json::from_value(json).unwrap();
        let sample = sample_at(&restored, 4, 3, &config).unwrap();
        assert_eq!(sample.trajectory_start, 3);
        assert_eq!(sample.loss_mask, [false, false, false, true, false]);
        assert!(sample.target_at(2, 0.5, 2).is_err());
        let target = sample.target_at(3, 0.5, 2).unwrap();
        assert_eq!(target.reward, 6.5);
        assert_eq!(target.discount, 0.25);
        assert_eq!(target.bootstrap_index, 5);
        let batch = SequenceSampler::new(9)
            .sample(&restored, &config, 32)
            .unwrap();
        assert!(batch.samples.iter().all(|s| s.learn_start >= 3));
    }
}

#[test]
fn malformed_append_does_not_change_the_episode() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    episode(&mut store, 1, 1, None);
    let before = serde_json::to_value(&store).unwrap();
    for bad in 0..6 {
        let mut t = transition(1, 1, false, false);
        match bad {
            0 => t.timestep = 3,
            1 => t.observation.as_dense_mut().unwrap().values[0] = 90.,
            2 => t.behavior.log_prob = Some(f32::NAN),
            3 => t.reward = f32::INFINITY.into(),
            4 => t.action = Action::Discrete(2),
            _ => t.next_observation.as_dense_mut().unwrap().values.push(1.),
        }
        assert!(store.push(&t).is_err(), "malformation {bad}");
        assert_eq!(serde_json::to_value(&store).unwrap(), before);
    }
    store.push(&transition(1, 1, true, false)).unwrap();
}

#[test]
fn revision_changes_are_per_transition_provenance_and_do_not_split_an_episode() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    episode(&mut store, 1, 1, None);
    let mut next = transition(1, 1, false, false);
    next.behavior.revision = 3;
    next.behavior.log_prob = None;
    store.push(&next).unwrap();
    let mut last = transition(1, 2, true, false);
    last.behavior.revision = 9;
    store.push(&last).unwrap();
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 1,
    };
    let sample = sample_at(&store, 1, 1, &config).unwrap();
    assert_eq!(
        sample
            .steps
            .iter()
            .map(|s| s.behavior.revision)
            .collect::<Vec<_>>(),
        [7, 3, 9]
    );
    assert_eq!(sample.observations.len(), 4);
    sample.validate(&config).unwrap();
    store.validate().unwrap();
}

#[test]
fn whole_episode_eviction_keeps_no_suffix_of_an_evicted_partial_episode() {
    let mut store = EpisodeStore::new(3, 1, space()).unwrap();
    episode(&mut store, 1, 2, None);
    episode(&mut store, 2, 2, Some(true));
    assert!(store.episode(1).is_none());
    assert_eq!(store.len(), 2);
    assert_eq!(store.ignored_count(), 1);
    store.push(&transition(1, 2, false, false)).unwrap();
    store.push(&transition(1, 3, true, false)).unwrap();
    assert_eq!(store.len(), 2);
    assert_eq!(store.ignored_count(), 0);
    assert!(
        sample_at(
            &store,
            1,
            0,
            &SequenceConfig {
                burn_in: 0,
                learn_len: 1,
                n_step: 1
            }
        )
        .is_err()
    );
    assert!(
        store
            .begin_episode(1, &EpisodeStart::fresh(observation(0.)))
            .is_err()
    );
    episode(&mut store, 3, 1, Some(true));
    store.validate().unwrap();
}

#[test]
fn active_identity_limit_bounds_empty_episodes_and_ignored_bookkeeping() {
    let mut store = EpisodeStore::new(2, 1, space()).unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(observation(0.)))
        .unwrap();
    store
        .begin_episode(2, &EpisodeStart::fresh(observation(0.)))
        .unwrap();
    let before = serde_json::to_value(&store).unwrap();
    assert!(
        store
            .begin_episode(3, &EpisodeStart::fresh(observation(0.)))
            .is_err()
    );
    assert_eq!(serde_json::to_value(&store).unwrap(), before);
    store.abandon_episode(1).unwrap();
    store
        .begin_episode(3, &EpisodeStart::fresh(observation(0.)))
        .unwrap();
    assert!(
        store
            .begin_episode(2, &EpisodeStart::fresh(observation(0.)))
            .is_err()
    );
    store.validate().unwrap();
}

#[test]
fn abandoning_an_ignored_episode_releases_its_bounded_identity() {
    let mut store = EpisodeStore::new(2, 1, space()).unwrap();
    episode(&mut store, 1, 2, None);
    store.push(&transition(1, 2, false, false)).unwrap();
    assert!(store.episode(1).is_none());
    assert_eq!(store.ignored_count(), 1);
    store.abandon_episode(1).unwrap();
    assert_eq!(store.ignored_count(), 0);
    assert!(store.push(&transition(1, 3, true, false)).is_err());
    episode(&mut store, 2, 1, Some(true));
    assert!(
        store
            .begin_episode(1, &EpisodeStart::fresh(observation(0.)))
            .is_err()
    );
}

#[test]
fn a_long_ignored_episode_does_not_grow_replay_memory() {
    let mut store = EpisodeStore::new(4, 1, space()).unwrap();
    episode(&mut store, 1, 5, None);
    let small = serde_json::to_vec(&store).unwrap().len();
    for t in 5..10_000 {
        store.push(&transition(1, t, false, false)).unwrap();
    }
    assert_eq!(store.len(), 0);
    assert_eq!(store.episodes().count(), 0);
    assert_eq!(store.ignored_count(), 1);
    assert!(serde_json::to_vec(&store).unwrap().len() <= small + 8);
    store.push(&transition(1, 10_000, false, true)).unwrap();
    assert_eq!(store.ignored_count(), 0);
    store.validate().unwrap();
}

#[test]
fn sample_contains_the_full_prefix_burn_tail_loss_window_and_nstep_context() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 8, Some(true));
    let config = SequenceConfig {
        burn_in: 2,
        learn_len: 2,
        n_step: 3,
    };
    let sample = sample_at(&store, 1, 3, &config).unwrap();
    assert_eq!(sample.episode_id, 1);
    assert_eq!(sample.learn_start, 3);
    assert_eq!(sample.burn_start, 1);
    assert_eq!(
        sample
            .observations
            .iter()
            .map(|o| o.dense_values()[0])
            .collect::<Vec<_>>(),
        [0., 1., 2., 3., 4., 5., 6., 7.]
    );
    assert_eq!(
        sample.loss_mask,
        [false, false, false, true, true, false, false]
    );
    let target = sample.target_at(3, 0.5, 3).unwrap();
    assert_eq!(target.reward, 8.);
    assert_eq!(target.discount, 0.125);
    assert_eq!(target.bootstrap_index, 6);
    sample.validate(&config).unwrap();
}

#[test]
fn terminal_and_truncated_targets_end_at_their_own_final_observation() {
    let config = SequenceConfig {
        burn_in: 8,
        learn_len: 2,
        n_step: 5,
    };
    for (id, terminal, discount) in [(1, true, 0.), (2, false, 0.25)] {
        let mut store = EpisodeStore::new(8, 1, space()).unwrap();
        episode(&mut store, id, 3, Some(terminal));
        let sample = sample_at(&store, id, 1, &config).unwrap();
        assert_eq!(sample.burn_start, 0);
        assert_eq!(sample.loss_mask, [false, true, true]);
        let target = sample.target_at(1, 0.5, 5).unwrap();
        assert_eq!(target.reward, 3.5);
        assert_eq!(target.discount, discount);
        assert_eq!(target.bootstrap_index, 3);
        assert_eq!(
            sample.observations[target.bootstrap_index].dense_values(),
            [3.]
        );
    }
}

#[test]
fn live_partial_tails_with_insufficient_nstep_context_are_not_learning_positions() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    episode(&mut store, 1, 3, None);
    let config = SequenceConfig {
        burn_in: 0,
        learn_len: 4,
        n_step: 2,
    };
    let sample = sample_at(&store, 1, 0, &config).unwrap();
    assert_eq!(sample.loss_mask, [true, true, false]);
    assert!(sample.target_at(2, 0.5, 2).is_err());
    assert!(sample_at(&store, 1, 2, &config).is_err());
    let mut sampler = SequenceSampler::new(10);
    let batch = sampler.sample(&store, &config, 100).unwrap();
    assert!(batch.samples.iter().all(|s| s.learn_start < 2));
}

#[test]
fn batch_major_padding_preserves_final_tokens_without_padding_losses_or_resets() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 5, Some(true));
    episode(&mut store, 2, 2, Some(true));
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 2,
    };
    let batch = SequenceBatch {
        samples: vec![
            sample_at(&store, 1, 2, &config).unwrap(),
            sample_at(&store, 2, 0, &config).unwrap(),
        ],
        config,
    };
    let layout = batch.padded_layout().unwrap();
    assert_eq!((layout.batch, layout.time, layout.width), (2, 6, 1));
    assert_eq!(
        layout.observations,
        [0., 1., 2., 3., 4., 5., 0., 1., 2., 0., 0., 0.]
    );
    assert_eq!(
        layout.valid_mask,
        [
            true, true, true, true, true, true, true, true, true, false, false, false
        ]
    );
    assert_eq!(
        layout.learning_mask,
        [
            false, false, true, true, false, false, true, true, false, false, false, false
        ]
    );
    assert_eq!(
        layout.reset_mask,
        [
            true, false, false, false, false, false, true, false, false, false, false, false
        ]
    );
}

#[test]
fn samples_are_owned_and_sampler_snapshot_continues_the_same_random_stream() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 6, Some(true));
    episode(&mut store, 2, 4, Some(false));
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 3,
    };
    let mut sampler = SequenceSampler::new(33);
    let owned = sampler.sample(&store, &config, 3).unwrap();
    let frozen = serde_json::to_value(&owned).unwrap();
    let mut resumed: SequenceSampler =
        serde_json::from_slice(&serde_json::to_vec(&sampler).unwrap()).unwrap();
    resumed.validate_snapshot().unwrap();
    assert_eq!(
        serde_json::to_value(sampler.sample(&store, &config, 20).unwrap()).unwrap(),
        serde_json::to_value(resumed.sample(&store, &config, 20).unwrap()).unwrap()
    );
    episode(&mut store, 3, 12, Some(true));
    assert!(store.episode(1).is_none());
    assert_eq!(serde_json::to_value(&owned).unwrap(), frozen);
    owned.validate().unwrap();
}

#[test]
fn snapshots_reject_unknown_versions_and_inconsistent_episode_topology() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    episode(&mut store, 1, 2, None);
    let encoded = serde_json::to_value(&store).unwrap();
    for bad in 0..5 {
        let mut value = encoded.clone();
        match bad {
            0 => value["version"] = 99.into(),
            1 => value["len"] = 0.into(),
            2 => value["episodes"][0]["observations"] = serde_json::json!([]),
            3 => value["episodes"][0]["steps"][0]["terminated"] = true.into(),
            _ => value["highest_episode_id"] = 0.into(),
        }
        assert!(
            serde_json::from_value::<EpisodeStore>(value).is_err(),
            "corruption {bad} must be rejected before hot store operations"
        );
    }
    let mut sampler = serde_json::to_value(SequenceSampler::new(4)).unwrap();
    sampler["version"] = 99.into();
    let sampler: SequenceSampler = serde_json::from_value(sampler).unwrap();
    assert!(sampler.validate_snapshot().is_err());
}

#[test]
fn malformed_sequences_and_invalid_budgets_are_rejected_without_rng_consumption() {
    let mut store = EpisodeStore::new(8, 1, space()).unwrap();
    episode(&mut store, 1, 3, Some(true));
    let config = SequenceConfig {
        burn_in: 0,
        learn_len: 2,
        n_step: 1,
    };
    let mut sampler = SequenceSampler::new(9);
    let before = serde_json::to_value(&sampler).unwrap();
    assert!(sampler.sample(&store, &config, 0).is_err());
    assert!(
        sampler
            .sample(
                &store,
                &SequenceConfig {
                    n_step: 0,
                    ..config
                },
                1
            )
            .is_err()
    );
    assert!(
        sampler
            .sample(
                &store,
                &SequenceConfig {
                    learn_len: usize::MAX,
                    n_step: 2,
                    ..config
                },
                1
            )
            .is_err()
    );
    assert_eq!(serde_json::to_value(&sampler).unwrap(), before);
    let sample = sample_at(&store, 1, 1, &config).unwrap();
    let mut invalid = sample.clone();
    invalid.loss_mask[0] = true;
    assert!(invalid.validate(&config).is_err());
    let mut invalid = sample.clone();
    invalid.observations.pop();
    assert!(invalid.validate(&config).is_err());
    assert!(sample.target_at(3, 0.5, 1).is_err());
    assert!(sample.target_at(1, f32::NAN, 1).is_err());
}

fn assert_reference_population(store: &EpisodeStore, population: &[(u64, usize)], n_step: usize) {
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step,
    };
    assert_eq!(store.eligible_starts(n_step).unwrap(), population.len());
    let seed = 923;
    let mut sampler = SequenceSampler::new(seed);
    let mut reference_rng = ChaCha8Rng::seed_from_u64(seed);
    let actual = sampler.sample(store, &config, 8_000).unwrap();
    let mut counts = vec![0usize; population.len()];
    for sample in actual.samples {
        let rank = reference_rng.random_range(0..population.len());
        assert_eq!((sample.episode_id, sample.learn_start), population[rank]);
        assert_eq!(
            sample,
            sample_at(store, population[rank].0, population[rank].1, &config).unwrap()
        );
        counts[rank] += 1;
    }
    let expected = 8_000f64 / population.len() as f64;
    for count in counts {
        assert!(
            (count as f64 - expected).abs() < 6. * expected.sqrt(),
            "biased eligible-start count {count}"
        );
    }
    store.validate_snapshot().unwrap();
}

#[test]
fn weighted_sampling_matches_reference_after_live_growth_end_abandon_and_eviction() {
    let mut store = EpisodeStore::new(18, 1, space()).unwrap();
    episode(&mut store, 1, 3, Some(true));
    episode(&mut store, 2, 4, None);
    episode(&mut store, 3, 2, Some(false));
    episode(&mut store, 4, 1, None);
    assert_reference_population(
        &store,
        &[(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)],
        3,
    );
    store.push(&transition(2, 4, false, false)).unwrap();
    store.push(&transition(4, 1, true, false)).unwrap();
    assert_reference_population(
        &store,
        &[
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
        ],
        3,
    );
    store.abandon_episode(2).unwrap();
    episode(&mut store, 5, 12, Some(true));
    assert!(store.episode(1).is_none());
    assert_reference_population(
        &store,
        &[
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
        ],
        3,
    );
    let restored: EpisodeStore =
        serde_json::from_value(serde_json::to_value(&store).unwrap()).unwrap();
    assert_reference_population(
        &restored,
        &[
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (5, 10),
            (5, 11),
        ],
        13,
    );
}

#[test]
fn arbitrary_horizons_exclude_complete_live_tails_without_caching_configs() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 1, None);
    episode(&mut store, 2, 2, Some(true));
    episode(&mut store, 3, 3, None);
    episode(&mut store, 4, 0, None);
    for (horizon, count) in [(1, 6), (2, 4), (3, 3), (4, 2), (usize::MAX, 2), (1, 6)] {
        assert_eq!(store.eligible_starts(horizon).unwrap(), count);
    }
    assert!(store.eligible_starts(0).is_err());
    assert_reference_population(&store, &[(2, 0), (2, 1), (3, 0)], 3);
    assert_reference_population(&store, &[(2, 0), (2, 1)], 4);
}

#[test]
fn reusable_batches_match_owned_sampling_and_preserve_output_and_rng_on_error() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 6, Some(true));
    episode(&mut store, 2, 3, None);
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 2,
    };
    let mut owned = SequenceSampler::new(31);
    let mut reusable = SequenceSampler::new(31);
    let mut out = SequenceBatch {
        samples: Vec::new(),
        config,
    };
    for size in [8, 8, 3, 10] {
        reusable
            .sample_into(&store, &config, size, &mut out)
            .unwrap();
        assert_eq!(out, owned.sample(&store, &config, size).unwrap());
    }
    let before_batch = out.clone();
    let before_rng = serde_json::to_value(&reusable).unwrap();
    let overflowing = SequenceConfig {
        learn_len: usize::MAX,
        n_step: 1,
        ..config
    };
    assert!(
        reusable
            .sample_into(&store, &overflowing, 20, &mut out)
            .is_err()
    );
    assert_eq!(out, before_batch);
    assert_eq!(serde_json::to_value(&reusable).unwrap(), before_rng);
    episode(&mut store, 3, 12, Some(false));
    assert_eq!(out, before_batch);
}

#[test]
fn restore_rejects_duplicate_order_and_active_index_topology_corruption() {
    let mut store = EpisodeStore::new(16, 1, space()).unwrap();
    episode(&mut store, 1, 2, None);
    episode(&mut store, 2, 1, Some(true));
    let encoded = serde_json::to_value(&store).unwrap();
    for bad in 0..5 {
        let mut value = encoded.clone();
        match bad {
            0 => value["episodes"][1] = value["episodes"][0].clone(),
            1 => value["episodes"].as_array_mut().unwrap().swap(0, 1),
            2 => value["active"]["1"]["next_timestep"] = 5.into(),
            3 => value["active"]["1"]["ignored"] = true.into(),
            _ => value["active"]["2"] = serde_json::json!({"next_timestep": 1, "ignored": false}),
        }
        assert!(
            serde_json::from_value::<EpisodeStore>(value).is_err(),
            "corruption {bad}"
        );
    }
}
