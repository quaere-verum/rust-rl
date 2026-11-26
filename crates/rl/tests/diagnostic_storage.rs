#![cfg(feature = "diagnostics")]
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use std::{collections::BTreeMap, fs, io::Write};

fn manifest() -> RunManifest {
    RunManifest::new(
        "run-a",
        "experiment",
        "synthetic",
        2,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        0.9,
    )
    .unwrap()
}
fn chunk(revision: u64) -> Chunk {
    let mut arrays = BTreeMap::new();
    arrays.insert(
        "values".into(),
        Array::f32(vec![2, 2], vec![1., -0., 3., 4.]).unwrap(),
    );
    Chunk {
        context: RecordContext::new("run-a", revision),
        record: Record::Activations(ActivationRecord {
            network: "policy".into(),
            layers: BTreeMap::from([("layers.0.pre".into(), "values".into())]),
            summaries: BTreeMap::new(),
            ..Default::default()
        }),
        arrays,
    }
}

#[test]
fn committed_arrays_round_trip_and_loading_one_does_not_read_missing_others() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let mut writer = Recorder::create(&path, manifest(), StorageConfig::default()).unwrap();
    writer.append(&chunk(1)).unwrap();
    writer.append(&chunk(2)).unwrap();
    writer.finish().unwrap();
    let reader = Recording::open(&path, ReadLimits::default()).unwrap();
    assert!(reader.complete());
    assert_eq!(reader.entries().len(), 2);
    let loaded = reader.load(0).unwrap();
    assert_eq!(
        loaded.arrays["values"].as_f32().unwrap(),
        &[1., -0., 3., 4.]
    );
    assert_eq!(
        loaded.arrays["values"].as_f32().unwrap()[1].to_bits(),
        (-0.0f32).to_bits()
    );
    fs::remove_file(path.join(&reader.entries()[1].file)).unwrap();
    assert!(reader.load(1).is_err());
    assert!(reader.load(0).is_ok());
}

#[test]
fn live_refresh_ignores_partial_index_tail_and_detects_corrupt_chunk() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let mut writer = Recorder::create(&path, manifest(), StorageConfig::default()).unwrap();
    let mut reader = Recording::open(&path, ReadLimits::default()).unwrap();
    writer.append(&chunk(4)).unwrap();
    reader.refresh().unwrap();
    assert_eq!(reader.entries().len(), 1);
    assert!(!reader.complete());
    let mut index = fs::OpenOptions::new()
        .append(true)
        .open(path.join("index.jsonl"))
        .unwrap();
    index.write_all(b"{\"interrupted").unwrap();
    reader.refresh().unwrap();
    assert!(
        reader
            .issues()
            .iter()
            .any(|issue| issue.contains("incomplete"))
    );
    assert!(reader.load(0).is_ok());
    let file = path.join(&reader.entries()[0].file);
    let mut bytes = fs::read(&file).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&file, bytes).unwrap();
    assert!(reader.load(0).is_err());
}

#[test]
fn schema_dimensions_paths_and_allocation_limits_are_validated() {
    assert!(Array::f32(vec![2, 2], vec![1.]).is_err());
    assert!(Array::f64(vec![usize::MAX, 2], vec![]).is_err());
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let mut writer = Recorder::create(&path, manifest(), StorageConfig::default()).unwrap();
    writer.append(&chunk(0)).unwrap();
    let reader = Recording::open(
        &path,
        ReadLimits {
            max_chunk_bytes: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(reader.load(0).is_err());
    let mut data: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
    data["schema_version"] = 999.into();
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec(&data).unwrap(),
    )
    .unwrap();
    assert!(Recording::open(&path, ReadLimits::default()).is_err());
}

#[test]
fn disabled_recording_does_not_evaluate_lazy_payload_and_budget_skips_are_visible() {
    let mut recorder = OptionalRecorder::disabled();
    recorder
        .record(|| -> rl::Result<Chunk> { panic!("disabled recorder evaluated probe") })
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let mut writer = Recorder::create(
        &path,
        manifest(),
        StorageConfig {
            max_total_bytes: 16_384,
            max_chunk_bytes: 1024,
            ..Default::default()
        },
    )
    .unwrap();
    let mut large = chunk(1);
    large.arrays.insert(
        "values".into(),
        Array::f32(vec![1000, 2], vec![0.; 2000]).unwrap(),
    );
    assert!(matches!(
        writer.append(&large).unwrap(),
        AppendOutcome::Skipped { .. }
    ));
    writer.append(&chunk(2)).unwrap();
    writer.finish().unwrap();
    let reader = Recording::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(reader.status().unwrap().skipped_records, 1);
    assert!(reader.entries().iter().any(|entry| entry.revision == 2));
}

#[test]
fn run_identity_and_dangling_array_references_are_rejected_before_commit() {
    let temp = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        temp.path().join("run"),
        manifest(),
        StorageConfig::default(),
    )
    .unwrap();
    let mut data = chunk(3);
    data.context.identity.run_id = "wrong-run".into();
    assert!(writer.append(&data).is_err());
    data.context.identity.run_id = "run-a".into();
    data.arrays.clear();
    assert!(writer.append(&data).is_err());
}

#[test]
fn every_principal_record_type_survives_a_multi_checkpoint_recording() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("all-records");
    let mut writer = Recorder::create(&path, manifest(), StorageConfig::default()).unwrap();
    let evaluation = EvaluationMetadata {
        conditioning: Conditioning::InitialScenario,
        initial_action: ActionCondition::CurrentPolicy,
        continuation_policy: "policy".into(),
        stochastic_policy: false,
        discount: 0.9,
        independent_future_streams: true,
        seed_methodology: "explicit seeds".into(),
        truncation_handling: "reject".into(),
        complete_returns: true,
    };
    let records = vec![
        Record::Metrics(MetricRecord {
            values: BTreeMap::from([("loss".into(), 1.25)]),
            unavailable: BTreeMap::new(),
        }),
        Record::Event(EventRecord {
            category: "numerical".into(),
            severity: "warning".into(),
            message: "fixture".into(),
        }),
        Record::ProbeSet(ProbeSetRecord {
            id: "fixed".into(),
            population: ProbePopulation::FixedReference,
            methodology: "hand selected".into(),
            observations: "obs".into(),
            action_masks: None,
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::new(),
        }),
        Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        Record::Critic(CriticRecord {
            representation: CriticRepresentation::Quantiles {
                values: "obs".into(),
                locations: "grid".into(),
                masses: "grid".into(),
            },
            action_condition: ActionCondition::Fixed,
            actions: Some("actions".into()),
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        chunk(0).record,
        Record::Empirical(EmpiricalRecord {
            returns: "returns".into(),
            seeds: "ids".into(),
            evaluation: evaluation.clone(),
            probe_row: Some(0),
            initial_action: None,
            conditioning_observation: None,
            conditioning_mask: None,
            simulator_state: None,
        }),
        Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode".into(),
            environment: "generic".into(),
            observations: "obs".into(),
            actions: "one-action".into(),
            rewards: "reward".into(),
            terminated: "flags".into(),
            truncated: "zero-flags".into(),
            simulation_times: None,
            action_masks: None,
            frames: BTreeMap::new(),
            evaluation,
        }),
        Record::ModelSnapshot(ModelSnapshotRecord {
            id: "k-2".into(),
            architecture: "external".into(),
            configuration: serde_json::json!({}),
            capabilities: vec![],
            models: BTreeMap::from([(
                "policy".into(),
                ModelArtifact {
                    bytes: "model".into(),
                    signature: "opaque fixture".into(),
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
        }),
    ];
    for revision in [2, 5] {
        for record in &records {
            let mut data = chunk(revision);
            data.record = record.clone();
            data.arrays.extend([
                (
                    "obs".into(),
                    Array::f32(vec![2, 2], vec![0., 1., 2., 3.]).unwrap(),
                ),
                (
                    "actions".into(),
                    Array::f32(vec![2, 1], vec![0., 1.]).unwrap(),
                ),
                (
                    "one-action".into(),
                    Array::f32(vec![1, 1], vec![0.]).unwrap(),
                ),
                ("reward".into(), Array::f64(vec![1], vec![2.]).unwrap()),
                ("flags".into(), Array::u8(vec![1], vec![1]).unwrap()),
                ("zero-flags".into(), Array::u8(vec![1], vec![0]).unwrap()),
                (
                    "ids".into(),
                    Array::u64(vec![2], vec![u64::MAX, 7]).unwrap(),
                ),
                (
                    "returns".into(),
                    Array::f64(vec![2], vec![-3.25, 7.125]).unwrap(),
                ),
                (
                    "grid".into(),
                    Array::f32(vec![2], vec![0.25, 0.75]).unwrap(),
                ),
                (
                    "model".into(),
                    Array::u8(vec![8192], vec![42; 8192]).unwrap(),
                ),
            ]);
            assert!(matches!(
                writer.append(&data).unwrap(),
                AppendOutcome::Committed { .. }
            ));
        }
    }
    writer.finish().unwrap();
    let reader = Recording::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(reader.entries().len(), 18);
    for (index, entry) in reader.entries().iter().enumerate() {
        let data = reader.load(index).unwrap();
        assert_eq!(data.record.kind(), entry.kind);
        assert_eq!(data.arrays["ids"].as_u64().unwrap(), &[u64::MAX, 7]);
        assert_eq!(data.arrays["returns"].as_f64().unwrap(), &[-3.25, 7.125]);
        assert!(
            entry.bytes < data.array_bytes() as u64,
            "repeated binary payload should compress"
        );
    }
}

#[test]
fn schedules_handle_skipped_revisions_without_inventing_intermediate_models() {
    use rl::diagnostics::schedule::*;
    let schedule = RecordingSchedule::default();
    assert!(!schedule.due(None, 19).unwrap().probes);
    assert!(schedule.due(Some(19), 23).unwrap().probes);
    assert!(!schedule.due(Some(23), 39).unwrap().probes);
    assert!(schedule.due(Some(39), 205).unwrap().snapshots);
    assert!(schedule.due(Some(30), 29).is_err());
}
