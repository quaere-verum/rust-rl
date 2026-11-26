use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::data::{RunIndex, SelectedInput, Selection};
use std::{collections::BTreeMap, path::Path};

fn writer(path: &Path) -> Recorder {
    Recorder::create(
        path,
        RunManifest::new(
            "index",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap()
}

fn context(revision: u64, role: &str, set: Option<&str>) -> RecordContext {
    let mut context = RecordContext::new("index", revision);
    context.identity.model_role = role.into();
    context.probe_set = set.map(str::to_owned);
    context
}

fn probe(revision: u64, set: &str, value: f32) -> Chunk {
    ProbeSet::new(
        set,
        ProbePopulation::FixedReference,
        "fixture",
        vec![Observation::new(vec![value])],
        None,
        ActionSpace::Discrete { count: 2 },
    )
    .unwrap()
    .record(context(revision, "online", Some(set)))
    .unwrap()
}

fn activation(revision: u64, role: &str, set: Option<&str>) -> Chunk {
    Chunk {
        context: context(revision, role, set),
        record: Record::Activations(ActivationRecord {
            network: "policy".into(),
            layers: BTreeMap::from([("hidden".into(), "values".into())]),
            ..Default::default()
        }),
        arrays: BTreeMap::from([("values".into(), Array::f32(vec![1, 1], vec![1.]).unwrap())]),
    }
}

fn metric(revision: u64) -> Chunk {
    Chunk {
        context: context(revision, "online", None),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::from([("loss".into(), 1.)]),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    }
}

#[test]
fn local_reference_shares_index_without_reopening_and_pins_live_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer.append(&metric(1)).unwrap();
    let mut run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut reference = rl_viewer::comparison::Comparison::new();

    // Reference initialization should use already validated metadata even if
    // the index cannot be reopened. Artifact availability is checked on use.
    let index = path.join("index.jsonl");
    let parked = path.join("index-test-parked.jsonl");
    std::fs::rename(&index, &parked).unwrap();
    let result = reference.reset_reference_to(&run);
    std::fs::rename(&parked, &index).unwrap();
    result.expect("same-run reference must not reopen the index");
    let pinned = reference.reference_run().unwrap();
    assert_eq!(
        pinned.recording().entries().as_ptr(),
        run.recording().entries().as_ptr()
    );
    assert_eq!(pinned.revisions().as_ptr(), run.revisions().as_ptr());

    writer.append(&metric(3)).unwrap();
    assert_eq!(run.refresh().unwrap(), 1);
    assert_eq!(run.revisions(), &[1, 3]);
    assert_eq!(pinned.revisions(), &[1]);
    assert_eq!(pinned.recording().entries().len(), 1);
    assert_eq!(run.recording().entries().len(), 2);
    assert_ne!(
        pinned.recording().entries().as_ptr(),
        run.recording().entries().as_ptr()
    );
    writer.finish().unwrap();
}

#[test]
fn loader_shared_wake_preserves_send_only_callback_support() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    // Cell is Send but not Sync: sharing the wake callback must not tighten
    // Loader's existing constructor contract.
    let local = std::cell::Cell::new(0usize);
    let loader = rl_viewer::data::Loader::new(1024, move || {
        local.set(local.get() + 1);
        observed.store(local.get(), Ordering::SeqCst);
    })
    .unwrap();
    let first = loader.wake_callback();
    let second = loader.wake_callback();
    std::thread::scope(|scope| {
        scope.spawn(move || first());
        scope.spawn(move || second());
    });
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

fn snapshot(revision: u64, role: &str) -> Chunk {
    Chunk {
        context: context(revision, role, None),
        record: Record::ModelSnapshot(ModelSnapshotRecord {
            id: format!("{role}-{revision}"),
            architecture: "fixture".into(),
            configuration: serde_json::json!({}),
            models: BTreeMap::from([(
                "policy".into(),
                ModelArtifact {
                    bytes: "weights".into(),
                    signature: "fixture".into(),
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
            capabilities: vec![],
        }),
        arrays: BTreeMap::from([("weights".into(), Array::u8(vec![1], vec![1]).unwrap())]),
    }
}

fn evaluation() -> EvaluationMetadata {
    EvaluationMetadata {
        conditioning: Conditioning::InitialScenario,
        initial_action: ActionCondition::CurrentPolicy,
        continuation_policy: "policy".into(),
        stochastic_policy: false,
        discount: 0.9,
        independent_future_streams: true,
        seed_methodology: "explicit seeds".into(),
        truncation_handling: "reject".into(),
        complete_returns: true,
    }
}

fn diagnostic(kind: RecordKind, revision: u64) -> Chunk {
    let mut chunk = activation(revision, "online", Some("fixed"));
    chunk.arrays.extend([
        ("actions".into(), Array::u64(vec![1], vec![0]).unwrap()),
        ("returns".into(), Array::f32(vec![1], vec![1.]).unwrap()),
        ("seeds".into(), Array::u64(vec![1], vec![7]).unwrap()),
        (
            "observations".into(),
            Array::f32(vec![2, 1], vec![0., 1.]).unwrap(),
        ),
        ("terminated".into(), Array::u8(vec![1], vec![1]).unwrap()),
        ("truncated".into(), Array::u8(vec![1], vec![0]).unwrap()),
    ]);
    chunk.record = match kind {
        RecordKind::Policy => Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        RecordKind::Critic => Record::Critic(CriticRecord {
            representation: CriticRepresentation::Scalar {
                values: "returns".into(),
            },
            action_condition: ActionCondition::StateValue,
            actions: None,
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        RecordKind::Activations => chunk.record,
        RecordKind::Empirical => Record::Empirical(EmpiricalRecord {
            returns: "returns".into(),
            seeds: "seeds".into(),
            evaluation: evaluation(),
            probe_row: Some(0),
            initial_action: None,
            conditioning_observation: None,
            conditioning_mask: None,
            simulator_state: None,
        }),
        RecordKind::Trajectory => Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode".into(),
            environment: "generic".into(),
            observations: "observations".into(),
            actions: "actions".into(),
            rewards: "returns".into(),
            terminated: "terminated".into(),
            truncated: "truncated".into(),
            simulation_times: None,
            action_masks: None,
            frames: BTreeMap::new(),
            evaluation: evaluation(),
        }),
        _ => panic!("fixture expects a diagnostic kind"),
    };
    chunk
}

#[test]
fn probe_predecessor_uses_revision_before_append_order() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer.append(&probe(21, "fixed", 21.)).unwrap();
    writer.append(&probe(1, "fixed", 1.)).unwrap();
    writer.append(&probe(41, "fixed", 41.)).unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(index.probe_set("fixed", 37), Some(0));
    assert_eq!(index.probe_set("fixed", 0), None);
}

#[test]
fn requested_revision_can_be_between_records_and_keeps_anchored_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer.append(&activation(1, "online", None)).unwrap();
    writer.append(&activation(41, "online", None)).unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let anchored = SelectedInput::TrajectoryStep {
        sequence: 19,
        step: 7,
    };
    let mut selection = Selection {
        revision: 1,
        input: anchored.clone(),
        ..Default::default()
    };
    selection.select_revision(&index, 1).unwrap();
    selection.select_revision(&index, 41).unwrap();
    selection.select_revision(&index, 37).unwrap();
    assert_eq!(selection.revision, 37);
    assert_eq!(selection.input, anchored);
    assert!(selection.select_revision(&index, 0).is_err());
    assert!(selection.select_revision(&index, 42).is_err());
    assert_eq!(selection.revision, 37);
}

#[test]
fn empty_recording_has_no_selection_range_or_candidates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let _writer = writer(&path);
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert!(Selection::default().select_revision(&index, 0).is_err());
    assert!(
        index
            .candidates(RecordKind::Activations, "online", None, u64::MAX)
            .next()
            .is_none()
    );
    assert!(index.snapshot_revisions("online").is_empty());
}

#[test]
fn model_availability_requires_its_own_role_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer.append(&snapshot(2, "target")).unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut selection = Selection {
        revision: 2,
        ..Default::default()
    };
    assert!(!index.has_model_for(selection.revision, &selection.model_role));
    selection.model_role = "target".into();
    assert!(index.has_model_for(selection.revision, &selection.model_role));
}

#[test]
fn dense_metrics_do_not_hide_sparse_modality_role_or_population_predecessors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    for revision in 0..=42 {
        writer.append(&metric(revision)).unwrap();
    }
    writer.append(&probe(1, "fixed", 1.)).unwrap(); // 43
    writer
        .append(&activation(1, "online", Some("fixed")))
        .unwrap(); // 44
    writer.append(&probe(21, "fixed", 21.)).unwrap(); // 45
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 46
    writer
        .append(&activation(41, "online", Some("fixed")))
        .unwrap(); // 47
    writer
        .append(&activation(21, "target", Some("fixed")))
        .unwrap(); // 48
    writer
        .append(&activation(21, "online", Some("visited")))
        .unwrap(); // 49
    writer.append(&activation(10, "online", None)).unwrap(); // 50
    // Indexing must remain usable even when none of the tensor chunks can load.
    for entry in Recording::open(&path, ReadLimits::default())
        .unwrap()
        .entries()
    {
        std::fs::remove_file(path.join(&entry.file)).unwrap();
    }
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "online", Some("fixed"), 37)
            .collect::<Vec<_>>(),
        [46, 44]
    );
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "online", None, 37)
            .collect::<Vec<_>>(),
        [49, 46, 50, 44]
    );
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "target", None, 37)
            .collect::<Vec<_>>(),
        [48]
    );
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "online", Some("visited"), 37)
            .collect::<Vec<_>>(),
        [49]
    );
    assert!(
        index
            .candidates(RecordKind::Activations, "online", None, 0)
            .next()
            .is_none()
    );
    assert!(
        index
            .candidates(RecordKind::Activations, "absent", None, 37)
            .next()
            .is_none()
    );
    assert!(
        index
            .candidates(RecordKind::Policy, "online", None, 37)
            .next()
            .is_none()
    );
    assert!(
        index
            .candidates(RecordKind::Activations, "online", Some("absent"), 37)
            .next()
            .is_none()
    );
}

#[test]
fn measurement_probe_identity_uses_its_own_sequence_and_revision() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 0
    writer.append(&probe(21, "fixed", 1.)).unwrap(); // 1
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 2
    writer.append(&probe(21, "fixed", 2.)).unwrap(); // 3
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 4
    writer.append(&probe(1, "fixed", 3.)).unwrap(); // 5
    writer
        .append(&activation(10, "online", Some("fixed")))
        .unwrap(); // 6
    writer.append(&probe(0, "fixed", 4.)).unwrap(); // 7
    writer
        .append(&activation(21, "online", Some("other")))
        .unwrap(); // 8
    writer.append(&activation(21, "online", None)).unwrap(); // 9
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(index.probe_for_record(0), None);
    assert_eq!(index.probe_for_record(1), Some(1));
    assert_eq!(index.probe_for_record(2), Some(1));
    assert_eq!(index.probe_for_record(3), Some(3));
    assert_eq!(index.probe_for_record(4), Some(3));
    assert_eq!(index.probe_for_record(6), Some(5));
    assert_eq!(index.probe_for_record(8), None);
    assert_eq!(index.probe_for_record(9), None);
    assert_eq!(index.probe_for_record(usize::MAX), None);
    assert_eq!(index.probe_set("fixed", 21), Some(3));
}

#[test]
fn live_append_updates_sorted_indices_without_loading_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer
        .append(&activation(41, "online", Some("fixed")))
        .unwrap(); // 0
    writer.append(&probe(41, "fixed", 41.)).unwrap(); // 1
    let mut index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    writer.append(&probe(1, "fixed", 1.)).unwrap(); // 2
    writer
        .append(&activation(1, "online", Some("fixed")))
        .unwrap(); // 3
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 4
    writer.append(&snapshot(13, "target")).unwrap(); // 5
    for entry in Recording::open(&path, ReadLimits::default())
        .unwrap()
        .entries()
        .iter()
        .skip(2)
    {
        std::fs::remove_file(path.join(&entry.file)).unwrap();
    }
    assert_eq!(index.refresh().unwrap(), 4);
    assert_eq!(index.refresh().unwrap(), 0);
    assert_eq!(index.revisions(), [1, 13, 21, 41]);
    assert_eq!(index.snapshot_revisions("target"), [13]);
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "online", None, 37)
            .collect::<Vec<_>>(),
        [4, 3]
    );
    assert_eq!(
        index
            .candidates(RecordKind::Activations, "online", None, 41)
            .collect::<Vec<_>>(),
        [0, 4, 3]
    );
    assert_eq!(index.probe_for_record(4), Some(2));
    assert_eq!(index.probe_for_record(0), None);
    assert_eq!(index.probe_set("fixed", 37), Some(2));
    assert!(!index.has_model_for(13, "target"));
}

#[test]
fn timeline_markers_include_only_diagnostics_and_separate_role_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path).with_snapshot_retention(1).unwrap();
    for (kind, revision) in [
        (RecordKind::Policy, 11),
        (RecordKind::Critic, 21),
        (RecordKind::Activations, 31),
        (RecordKind::Empirical, 41),
        (RecordKind::Trajectory, 51),
        (RecordKind::Policy, 11),
    ] {
        writer.append(&diagnostic(kind, revision)).unwrap();
    }
    writer.append(&metric(1)).unwrap();
    writer.append(&probe(2, "fixed", 1.)).unwrap();
    writer
        .append(&Chunk {
            context: context(3, "online", None),
            record: Record::Event(EventRecord {
                category: "fixture".into(),
                severity: "info".into(),
                message: "fixture".into(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    writer.append(&snapshot(5, "online")).unwrap();
    let mut index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert!(index.has_model_for(5, "online"));
    assert!(!index.has_model_for(5, "target"));
    writer.append(&snapshot(7, "online")).unwrap();
    writer.append(&snapshot(9, "target")).unwrap();
    index.refresh().unwrap();
    assert_eq!(index.snapshot_revisions("online"), [5, 7]);
    assert_eq!(index.snapshot_revisions("target"), [9]);
    assert!(index.snapshot_revisions("absent").is_empty());
    assert!(!index.has_model_for(5, "online"));
    assert!(index.has_model_for(7, "online"));
    assert!(index.has_model_for(9, "target"));
    assert!(!index.has_model_for(8, "online"));
    let mut selection = Selection {
        revision: 7,
        ..Default::default()
    };
    selection.select_revision(&index, 7).unwrap();
    selection.select_revision(&index, 9).unwrap();
}

#[test]
fn exclusive_candidate_cursor_continues_within_one_revision_without_revisiting_records() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    for (revision, role, set) in [
        (21, "online", "fixed"),   // 0
        (1, "online", "fixed"),    // 1: older revision, greater sequence
        (21, "online", "fixed"),   // 2
        (21, "online", "visited"), // 3
        (21, "target", "fixed"),   // 4
        (41, "online", "fixed"),   // 5
        (21, "online", "fixed"),   // 6: older revision appended last
    ] {
        writer
            .append(&activation(revision, role, Some(set)))
            .unwrap();
    }
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(
        index
            .candidates_before(RecordKind::Activations, "online", None, (21, 6))
            .collect::<Vec<_>>(),
        [3, 2, 0, 1]
    );
    for (before, expected) in [
        ((21, 6), vec![2, 0, 1]),
        ((21, 2), vec![0, 1]),
        ((21, 0), vec![1]),
        ((0, 0), vec![]),
        ((u64::MAX, u64::MAX), vec![5, 6, 2, 0, 1]),
    ] {
        assert_eq!(
            index
                .candidates_before(RecordKind::Activations, "online", Some("fixed"), before)
                .collect::<Vec<_>>(),
            expected,
            "exclusive cursor {before:?}"
        );
    }
    assert!(
        index
            .candidates_before(RecordKind::Activations, "absent", None, (21, 6))
            .next()
            .is_none()
    );
    assert!(
        index
            .candidates_before(RecordKind::Activations, "online", Some("absent"), (21, 6))
            .next()
            .is_none()
    );
}

#[test]
fn candidate_frontier_counts_late_older_records_even_when_its_newest_tuple_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = writer(&path);
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 0
    writer
        .append(&activation(41, "online", Some("fixed")))
        .unwrap(); // 1
    writer
        .append(&activation(21, "target", Some("fixed")))
        .unwrap(); // 2
    writer
        .append(&activation(10, "online", Some("visited")))
        .unwrap(); // 3
    writer.append(&metric(5)).unwrap(); // 4
    let mut index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", None, 37),
        (2, Some((21, 0)))
    );
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("fixed"), 37),
        (1, Some((21, 0)))
    );
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("fixed"), 41),
        (2, Some((41, 1)))
    );
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", None, 0),
        (0, None)
    );
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "absent", None, 37),
        (0, None)
    );
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("absent"), 37),
        (0, None)
    );

    writer
        .append(&activation(1, "online", Some("fixed")))
        .unwrap(); // 5
    writer.append(&metric(37)).unwrap(); // 6
    index.refresh().unwrap();
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("fixed"), 37),
        (2, Some((21, 0)))
    );
    writer.append(&metric(38)).unwrap(); // 7
    index.refresh().unwrap();
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("fixed"), 37),
        (2, Some((21, 0)))
    );
    writer
        .append(&activation(21, "online", Some("fixed")))
        .unwrap(); // 8
    index.refresh().unwrap();
    assert_eq!(
        index.candidate_frontier(RecordKind::Activations, "online", Some("fixed"), 37),
        (3, Some((21, 8)))
    );
}
