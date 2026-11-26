use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        anchors::*,
        env::{EpisodeStart, Observation},
    },
    data::temporal::*,
};
use std::collections::BTreeMap;

fn obs(t: usize) -> Observation {
    Observation::dense(vec![t as f32], Some(vec![true, true]))
}
fn step(id: u64, t: usize, ended: bool) -> TemporalTransition {
    TemporalTransition {
        episode_id: id,
        timestep: t,
        observation: obs(t),
        action: Action::Discrete(t % 2),
        reward: (t as f32).into(),
        next_observation: obs(t + 1),
        terminated: false,
        truncated: ended,
        behavior: BehaviorMetadata::default(),
    }
}
fn anchors(id: u64, b: usize) -> AnchorSet {
    AnchorSet {
        version: 2,
        anchors: [1, 2]
            .into_iter()
            .map(|key| {
                let representation_id = RepresentationId(key);
                (
                    representation_id,
                    StateAnchor {
                        version: 2,
                        episode_id: id,
                        observation_index: b,
                        phase: AnchorPhase::BeforeObservation,
                        representation_id,
                        model_identity: [key as u8; 32],
                        generation: 0,
                        schema: AnchorSchema {
                            codec: 1,
                            codec_version: 1,
                            representation_identity: [9; 32],
                            dimensions: vec![1],
                        },
                        payload: (b as f32).to_le_bytes().to_vec(),
                        exact_ancestry: true,
                        source: AnchorSource::ResetReconstruction,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>(),
    }
}
fn config(cap: usize) -> SegmentedConfig {
    SegmentedConfig {
        segment_steps: 8,
        minimum_context: 16,
        required_representations: vec![RepresentationId(1), RepresentationId(2)],
        limits: ResourceLimits {
            max_transitions: cap,
            max_resident_bytes: 2_000_000,
            max_anchor_bytes: 100_000,
            max_anchors: 256,
            max_anchor_payload_bytes: 64,
            max_dimension_rank: 4,
        },
    }
}

#[test]
fn a_continuing_episode_keeps_recent_anchored_history_past_hot_capacity() {
    let mut store = EpisodeStore::with_storage(
        64,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::Segmented(config(64)),
    )
    .unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    for t in 0..2_048 {
        store
            .push_with_anchors(
                &step(1, t, false),
                ((t + 1) % 8 == 0).then(|| anchors(1, t + 1)),
            )
            .unwrap();
        assert!(store.len() <= 64);
        assert_eq!(store.active_timestep(1), Some(t + 1));
    }
    let retained = store.episode(1).unwrap();
    assert!(retained.base_timestep > 1_900);
    assert_eq!(
        retained.previous.as_ref().unwrap().reward,
        (retained.base_timestep - 1) as f32
    );
    assert_eq!(retained.end_timestep(), 2_048);
    store.validate().unwrap();
}

fn store(capacity: usize) -> EpisodeStore {
    EpisodeStore::with_storage(
        capacity,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::Segmented(config(capacity)),
    )
    .unwrap()
}
fn append(store: &mut EpisodeStore, id: u64, start: usize, end: usize, finish: bool) {
    for t in start..end {
        store
            .push_with_anchors(
                &step(id, t, finish && t + 1 == end),
                ((t + 1) % 8 == 0).then(|| anchors(id, t + 1)),
            )
            .unwrap();
    }
}
fn sequences() -> SequenceConfig {
    SequenceConfig {
        burn_in: 4,
        learn_len: 6,
        n_step: 3,
    }
}
fn requirements() -> AnchorRequirements {
    AnchorRequirements {
        required: [1, 2]
            .into_iter()
            .map(|key| AnchorRequirement {
                representation_id: RepresentationId(key),
                schema: None,
                model_identity: Some([key as u8; 32]),
                require_exact: true,
                min_generation: None,
                max_generation: None,
            })
            .collect(),
    }
}

#[test]
fn fair_cuts_keep_eight_continuing_lanes_and_a_fresh_terminal_tail() {
    let mut store = store(512);
    for id in 1..=8 {
        store
            .begin_episode(id, &EpisodeStart::fresh(obs(0)))
            .unwrap();
    }
    for t in 0..2_048 {
        for id in 1..=8 {
            store
                .push_with_anchors(
                    &step(id, t, id == 1 && t == 2_047),
                    ((t + 1) % 8 == 0).then(|| anchors(id, t + 1)),
                )
                .unwrap();
        }
        if t % 128 == 0 {
            store.validate().unwrap();
        }
    }
    assert!(store.len() <= 512);
    for id in 1..=8 {
        let tape = store.episode(id).unwrap();
        assert_eq!(tape.end_timestep(), 2_048);
        assert!(tape.steps.len() >= 48, "lane {id} lost its fair context");
        assert!(!store.is_ignored(id));
    }
    append(&mut store, 2, 2_048, 2_200, false);
    let terminal = store.episode(1).unwrap();
    assert!(terminal.ended());
    assert_eq!(terminal.observations.last(), Some(&obs(2_048)));
    assert!(terminal.steps.len() >= 16);
    let stats = store.maintenance_stats();
    assert!(
        stats.cuts > 0
            && stats.trimmed_transitions > 15_000
            && stats.moved_descriptors > stats.cuts
    );
    assert!(stats.seconds.is_finite());
    store.validate().unwrap();
}

#[test]
fn missing_anchors_and_bad_envelopes_fail_before_changing_history_or_clock() {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    for t in 0..64 {
        store.push(&step(1, t, false)).unwrap();
    }
    let before = serde_json::to_value(&store).unwrap();
    let usage = store.resource_usage();
    assert!(store.push(&step(1, 64, false)).is_err());
    assert_eq!(serde_json::to_value(&store).unwrap(), before);
    assert_eq!(store.resource_usage(), usage);
    for bad in 0..5 {
        let mut set = anchors(1, 65);
        match bad {
            0 => {
                set.anchors.remove(&RepresentationId(2));
            }
            1 => {
                set.anchors
                    .get_mut(&RepresentationId(1))
                    .unwrap()
                    .observation_index = 64;
            }
            2 => {
                set.anchors.get_mut(&RepresentationId(1)).unwrap().payload = vec![0; 65];
            }
            3 => {
                set.anchors
                    .get_mut(&RepresentationId(1))
                    .unwrap()
                    .schema
                    .dimensions = vec![usize::MAX, 2];
            }
            _ => {
                set.anchors.get_mut(&RepresentationId(1)).unwrap().version = 1;
            }
        }
        assert!(
            store
                .push_with_anchors(&step(1, 64, false), Some(set))
                .is_err()
        );
        assert_eq!(serde_json::to_value(&store).unwrap(), before);
    }
    store.install_anchors(anchors(1, 8)).unwrap();
    store.push(&step(1, 64, false)).unwrap();
    assert_eq!(store.episode(1).unwrap().base_timestep, 8);
    assert_eq!(store.active_timestep(1), Some(65));
    store.validate().unwrap();
}

#[test]
fn anchored_windows_use_actual_previous_input_and_match_full_prefix_targets() {
    let mut bounded = store(64);
    let limits = config(2_100).limits;
    let mut exact = EpisodeStore::with_storage(
        2_100,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::FullPrefix(limits),
    )
    .unwrap();
    bounded
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    exact
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    for t in 0..2_048 {
        let transition = step(1, t, t == 2_047);
        bounded
            .push_with_anchors(&transition, ((t + 1) % 8 == 0).then(|| anchors(1, t + 1)))
            .unwrap();
        exact.push(&transition).unwrap();
    }
    let sequence = sequences();
    let batch = SequenceSampler::new(83)
        .sample(&bounded, &sequence, 256)
        .unwrap();
    for sample in &batch.samples {
        assert!(sample.history_start > 0);
        assert!(sample.learn_start >= 4 && sample.learn_start < 12);
        assert!(sample.observations.len() <= 21);
        assert_eq!(
            sample.previous.as_ref().unwrap().reward,
            (sample.history_start - 1) as f32
        );
        assert_eq!(
            sample.previous.as_ref().unwrap().action,
            Action::Discrete((sample.history_start - 1) % 2)
        );
        assert_eq!(sample.observations[0], obs(sample.history_start));
        let SequenceOrigin::Anchored(set) = &sample.origin else {
            panic!("suffix was reset");
        };
        assert_eq!(set.observation_index(), sample.history_start);
        let reference = sample_at(&exact, 1, sample.absolute_learn_start(), &sequence).unwrap();
        for (local, &learning) in sample.loss_mask.iter().enumerate() {
            if !learning {
                continue;
            }
            let actual = sample.target_at(local, 0.9, 3).unwrap();
            let expected = reference
                .target_at(sample.history_start + local, 0.9, 3)
                .unwrap();
            assert_eq!(actual.reward, expected.reward);
            assert_eq!(actual.discount, expected.discount);
            assert_eq!(
                sample.observations[actual.bootstrap_index],
                reference.observations[expected.bootstrap_index]
            );
        }
        sample.validate(&sequence).unwrap();
    }
    assert!(
        batch
            .padded_layout()
            .unwrap()
            .reset_mask
            .iter()
            .all(|&reset| !reset)
    );
    let mut corrupt = batch.samples[0].clone();
    corrupt.origin = SequenceOrigin::Reset;
    assert!(corrupt.validate(&sequence).is_err());
}

#[test]
fn uniform_keyed_sampling_filters_wrong_target_identity_and_enforces_atomic_limits() {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut store, 1, 0, 200, false);
    let sequence = sequences();
    let request = requirements();
    let tape = store.episode(1).unwrap();
    let first = tape.base_timestep + 4;
    let end = tape.end_timestep() - 2;
    assert_eq!(
        store
            .eligible_starts_with(&sequence, &request, &SampleLimits::default())
            .unwrap(),
        end - first
    );
    let mut sampler = SequenceSampler::new(42);
    let mut reference = ChaCha8Rng::seed_from_u64(42);
    let mut out = SequenceBatch {
        samples: Vec::new(),
        config: sequence,
    };
    sampler
        .sample_into_with(
            &store,
            &sequence,
            4_000,
            &request,
            &SampleLimits::default(),
            &mut out,
        )
        .unwrap();
    let mut counts = vec![0usize; end - first];
    for sample in &out.samples {
        let expected = first + reference.random_range(0..end - first);
        assert_eq!(sample.absolute_learn_start(), expected);
        counts[expected - first] += 1;
    }
    let expected = 4_000. / counts.len() as f64;
    assert!(
        counts
            .iter()
            .all(|&count| (count as f64 - expected).abs() < 6. * expected.sqrt())
    );
    let before = out.clone();
    let before_rng = serde_json::to_value(&sampler).unwrap();
    let mut wrong = request.clone();
    wrong.required[1].model_identity = Some([1; 32]);
    assert_eq!(
        store
            .eligible_starts_with(&sequence, &wrong, &SampleLimits::default())
            .unwrap(),
        0
    );
    assert!(
        sampler
            .sample_into_with(
                &store,
                &sequence,
                2,
                &wrong,
                &SampleLimits::default(),
                &mut out
            )
            .is_err()
    );
    for limits in [
        SampleLimits {
            max_sample_tokens: 1,
            ..Default::default()
        },
        SampleLimits {
            max_batch_tokens: 1,
            ..Default::default()
        },
        SampleLimits {
            max_sample_payload_bytes: 1,
            ..Default::default()
        },
        SampleLimits {
            max_batch_payload_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(
            sampler
                .sample_into_with(&store, &sequence, 2, &request, &limits, &mut out)
                .is_err()
        );
    }
    assert_eq!(serde_json::to_value(&sampler).unwrap(), before_rng);
    assert_eq!(out, before);
}

#[test]
fn v2_snapshots_continue_windows_and_rng_and_reject_corrupted_boundaries() {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut store, 1, 0, 200, false);
    let encoded = serde_json::to_value(&store).unwrap();
    assert_eq!(encoded["version"], 2);
    let restored: EpisodeStore = serde_json::from_value(encoded.clone()).unwrap();
    let mut sampler = SequenceSampler::new(98);
    sampler.sample(&store, &sequences(), 7).unwrap();
    let mut resumed: SequenceSampler =
        serde_json::from_value(serde_json::to_value(&sampler).unwrap()).unwrap();
    assert_eq!(
        sampler.sample(&store, &sequences(), 50).unwrap(),
        resumed.sample(&restored, &sequences(), 50).unwrap()
    );
    for bad in 0..7 {
        let mut value = encoded.clone();
        match bad {
            0 => value["episodes"][0]["base_timestep"] = 0.into(),
            1 => value["episodes"][0]["previous"] = serde_json::Value::Null,
            2 => value["episodes"][0]["anchors"][0]["version"] = 1.into(),
            3 => value["episodes"][0]["anchors"][0]["anchors"]["1"]["observation_index"] = 0.into(),
            4 => value["episodes"][0]["anchors"][0]["anchors"]
                .as_object_mut()
                .unwrap()
                .remove("2")
                .map(|_| ())
                .unwrap(),
            5 => value["storage"]["Segmented"]["limits"]["max_anchor_bytes"] = 1.into(),
            _ => value["active"]["1"]["next_timestep"] = 3.into(),
        }
        assert!(
            serde_json::from_value::<EpisodeStore>(value).is_err(),
            "corruption {bad}"
        );
    }
}

#[test]
fn full_prefix_and_anchor_limits_error_atomically_and_approximation_cannot_be_laundered() {
    let mut store = EpisodeStore::with_storage(
        4,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::FullPrefix(config(4).limits),
    )
    .unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    for t in 0..4 {
        store.push(&step(1, t, false)).unwrap();
    }
    let before = serde_json::to_value(&store).unwrap();
    assert!(store.push(&step(1, 4, false)).is_err());
    assert_eq!(serde_json::to_value(&store).unwrap(), before);
    assert!(!store.is_ignored(1));
    let mut set = anchors(1, 4);
    let anchor = set.anchors.get_mut(&RepresentationId(1)).unwrap();
    anchor.source = AnchorSource::Continued {
        parent_identity: [0; 32],
        parent_model_identity: [2; 32],
        parent_generation: 0,
        parent_exact: true,
    };
    assert!(set.validate().is_err());
    set.anchors
        .get_mut(&RepresentationId(1))
        .unwrap()
        .exact_ancestry = false;
    set.validate().unwrap();
    store.install_anchors(set).unwrap();
    store.validate().unwrap();
}

#[test]
fn an_unmasked_sample_validates_a_previous_action_absent_from_its_current_steps() {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut store, 1, 0, 200, false);
    let mut sample = SequenceSampler::new(3)
        .sample(&store, &sequences(), 1)
        .unwrap()
        .samples
        .remove(0);
    for observation in &mut sample.observations {
        observation.as_dense_mut().unwrap().action_mask = None;
    }
    for step in &mut sample.steps {
        step.action = Action::Discrete(0);
    }
    sample.previous.as_mut().unwrap().action = Action::Discrete(1);
    sample.validate(&sequences()).unwrap();
}

#[test]
fn an_older_completed_tail_does_not_starve_a_new_full_prefix_episode() {
    let mut store = EpisodeStore::with_storage(
        2,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::FullPrefix(config(2).limits),
    )
    .unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    store.push(&step(1, 0, false)).unwrap();
    store.push(&step(1, 1, true)).unwrap();
    store
        .begin_episode(2, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    store.push(&step(2, 0, false)).unwrap();
    assert!(store.episode(1).is_none());
    assert_eq!(store.active_timestep(2), Some(1));
    assert_eq!(store.episode(2).unwrap().base_timestep, 0);
    store.validate().unwrap();
}

#[test]
fn descriptor_growth_interval_cannot_exceed_the_hot_capacity() {
    let mut oversized = config(64);
    oversized.segment_steps = 65;
    assert!(
        EpisodeStore::with_storage(
            64,
            1,
            ActionSpace::Discrete { count: 2 },
            TemporalStorage::Segmented(oversized)
        )
        .is_err()
    );
}

#[test]
fn declared_approximation_schema_generation_and_resource_limits_are_enforced() {
    let mut bounded = store(64);
    bounded
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    for t in 0..200 {
        let set = ((t + 1) % 8 == 0).then(|| {
            let mut set = anchors(1, t + 1);
            for anchor in set.anchors.values_mut() {
                anchor.exact_ancestry = false;
                anchor.source = AnchorSource::Continued {
                    parent_identity: [0; 32],
                    parent_model_identity: anchor.model_identity,
                    parent_generation: 0,
                    parent_exact: false,
                };
            }
            set
        });
        bounded.push_with_anchors(&step(1, t, false), set).unwrap();
    }
    let mut request = requirements();
    assert_eq!(
        bounded
            .eligible_starts_with(&sequences(), &request, &SampleLimits::default())
            .unwrap(),
        0
    );
    for source in &mut request.required {
        source.require_exact = false;
    }
    assert!(
        bounded
            .eligible_starts_with(&sequences(), &request, &SampleLimits::default())
            .unwrap()
            > 0
    );
    request.required[0].min_generation = Some(1);
    assert_eq!(
        bounded
            .eligible_starts_with(&sequences(), &request, &SampleLimits::default())
            .unwrap(),
        0
    );
    request.required[0].min_generation = None;
    let mut future = serde_json::to_value(&bounded).unwrap();
    for set in future["episodes"][0]["anchors"].as_array_mut().unwrap() {
        for anchor in set["anchors"].as_object_mut().unwrap().values_mut() {
            anchor["generation"] = 1.into();
        }
    }
    let future: EpisodeStore = serde_json::from_value(future).unwrap();
    for source in &mut request.required {
        source.max_generation = Some(0);
    }
    assert_eq!(
        future
            .eligible_starts_with(&sequences(), &request, &SampleLimits::default())
            .unwrap(),
        0
    );
    for source in &mut request.required {
        source.max_generation = None;
    }
    let mut schema = anchors(1, 0).anchors[&RepresentationId(1)].schema.clone();
    schema.representation_identity = [8; 32];
    request.required[0].schema = Some(schema);
    assert_eq!(
        bounded
            .eligible_starts_with(&sequences(), &request, &SampleLimits::default())
            .unwrap(),
        0
    );

    let mut limited = config(64);
    limited.limits.max_anchors = 2;
    limited.limits.max_anchor_bytes = 8;
    let mut store = EpisodeStore::with_storage(
        64,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::Segmented(limited),
    )
    .unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut store, 1, 0, 15, false);
    let before = serde_json::to_value(&store).unwrap();
    assert!(
        store
            .push_with_anchors(&step(1, 15, false), Some(anchors(1, 16)))
            .is_err()
    );
    assert_eq!(serde_json::to_value(&store).unwrap(), before);
    assert_eq!(store.resource_usage().anchors, 2);
    store.validate().unwrap();

    let mut empty = store_for_small_resident_budget();
    let before = serde_json::to_value(&empty).unwrap();
    assert!(empty.push(&step(1, 0, false)).is_err());
    assert_eq!(serde_json::to_value(&empty).unwrap(), before);
    empty.validate().unwrap();
}

fn store_for_small_resident_budget() -> EpisodeStore {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    let mut value = serde_json::to_value(&store).unwrap();
    value["storage"]["Segmented"]["limits"]["max_resident_bytes"] =
        store.resource_usage().resident_bytes.into();
    serde_json::from_value(value).unwrap()
}

#[test]
fn impossible_descriptor_reservation_is_rejected_as_a_budget_error_before_allocation() {
    let capacity = usize::MAX / 2;
    let mut configuration = config(capacity);
    configuration.segment_steps = capacity;
    configuration.limits.max_resident_bytes = 4_096;
    let mut store = EpisodeStore::with_storage(
        capacity,
        1,
        ActionSpace::Discrete { count: 2 },
        TemporalStorage::Segmented(configuration),
    )
    .unwrap();
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    let before = serde_json::to_value(&store).unwrap();
    assert!(
        matches!(store.push(&step(1, 0, false)), Err(rl::Error::Config(_))),
        "resident budget must reject before Vec reservation"
    );
    assert_eq!(serde_json::to_value(&store).unwrap(), before);
}

#[test]
fn v2_restores_allocation_history_at_a_tight_resident_boundary() {
    let mut calibration = store(64);
    calibration
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut calibration, 1, 0, 56, false);
    let reserve = 65 * std::mem::size_of::<rl::core::env::Observation>()
        + 64 * std::mem::size_of::<TemporalStep>()
        + 6;
    for adjustment in [-1isize, 1] {
        let mut cfg = config(64);
        cfg.limits.max_resident_bytes = (calibration.resource_usage().resident_bytes + reserve)
            .checked_add_signed(adjustment)
            .unwrap();
        let mut original = EpisodeStore::with_storage(
            64,
            1,
            ActionSpace::Discrete { count: 2 },
            TemporalStorage::Segmented(cfg),
        )
        .unwrap();
        original
            .begin_episode(1, &EpisodeStart::fresh(obs(0)))
            .unwrap();
        append(&mut original, 1, 0, 56, false);
        let mut cloned = original.clone();
        cloned.validate().unwrap();
        let value = serde_json::to_value(&original).unwrap();
        let mut restored: EpisodeStore = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(original.resource_usage(), restored.resource_usage());
        for t in 56..70 {
            let transition = step(1, t, false);
            let supplied = ((t + 1) % 8 == 0).then(|| anchors(1, t + 1));
            let expected = original
                .push_with_anchors(&transition, supplied.clone())
                .is_ok();
            assert_eq!(
                restored
                    .push_with_anchors(&transition, supplied.clone())
                    .is_ok(),
                expected
            );
            assert_eq!(
                cloned.push_with_anchors(&transition, supplied).is_ok(),
                expected
            );
            assert_eq!(
                serde_json::to_value(&original).unwrap(),
                serde_json::to_value(&restored).unwrap()
            );
            assert_eq!(original.resource_usage(), restored.resource_usage());
            if !expected {
                break;
            }
        }
        for corruption in 0..3 {
            let mut bad = value.clone();
            match corruption {
                0 => {
                    bad.as_object_mut().unwrap().remove("allocation_capacities");
                }
                1 => bad["allocation_capacities"]["1"]["observations"] = 0.into(),
                _ => bad["allocation_capacities"]["1"]["steps"] = usize::MAX.into(),
            }
            assert!(serde_json::from_value::<EpisodeStore>(bad).is_err());
        }
    }
}

#[test]
fn incoming_anchor_spare_capacity_is_canonical_and_huge_batches_fail_before_reservation() {
    let mut store = store(64);
    store
        .begin_episode(1, &EpisodeStart::fresh(obs(0)))
        .unwrap();
    append(&mut store, 1, 0, 7, false);
    let mut set = anchors(1, 8);
    for anchor in set.anchors.values_mut() {
        anchor.payload.reserve(1_000);
        anchor.schema.dimensions.reserve(100);
    }
    store
        .push_with_anchors(&step(1, 7, false), Some(set))
        .unwrap();
    for anchor in store.episode(1).unwrap().anchors[0].anchors.values() {
        assert_eq!(anchor.payload.capacity(), anchor.payload.len());
        assert_eq!(
            anchor.schema.dimensions.capacity(),
            anchor.schema.dimensions.len()
        );
    }
    let mut sampler = SequenceSampler::new(34);
    let mut output = sampler.sample(&store, &sequences(), 1).unwrap();
    let frozen = output.clone();
    let random = serde_json::to_value(&sampler).unwrap();
    for limits in [
        SampleLimits {
            max_batch_tokens: 2,
            ..Default::default()
        },
        SampleLimits {
            max_batch_payload_bytes: 2,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            sampler.sample_into_with(
                &store,
                &sequences(),
                usize::MAX / 2,
                &AnchorRequirements::default(),
                &limits,
                &mut output
            ),
            Err(rl::Error::Config(_))
        ));
    }
    assert_eq!(output, frozen);
    assert_eq!(serde_json::to_value(&sampler).unwrap(), random);
}
