use rl::core::action::ActionSpace;
use rl::diagnostics::{schema::*, storage::*};
use rl_viewer::data::{CacheKey, ChunkCache, RunIndex, SelectedInput, Selection};
use std::{collections::BTreeMap, sync::Arc};

fn data(revision: u64) -> Chunk {
    Chunk {
        context: RecordContext::new("viewer-test", revision),
        record: Record::Activations(ActivationRecord {
            network: "policy".into(),
            layers: BTreeMap::from([("layers.0.post".into(), "values".into())]),
            summaries: BTreeMap::new(),
            ..Default::default()
        }),
        arrays: BTreeMap::from([(
            "values".into(),
            Array::f32(vec![2, 4], vec![revision as f32; 8]).unwrap(),
        )]),
    }
}
#[test]
fn opening_an_index_is_lazy_and_training_selection_preserves_environment_time() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let manifest = RunManifest::new(
        "viewer-test",
        "test",
        "ppo",
        2,
        ActionSpace::Discrete { count: 2 },
        0.9,
    )
    .unwrap();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.append(&data(2)).unwrap();
    writer.append(&data(7)).unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(index.revisions(), &[2, 7]);
    std::fs::remove_file(path.join(&index.recording().entries()[1].file)).unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut selection = Selection {
        revision: 2,
        input: SelectedInput::TrajectoryStep {
            sequence: 42,
            step: 9,
        },
        ..Default::default()
    };
    selection.select_revision(&index, 7).unwrap();
    assert_eq!(
        selection.input,
        SelectedInput::TrajectoryStep {
            sequence: 42,
            step: 9
        }
    );
    assert!(!index.has_model_for(7, "online"));
    assert!(!index.has_model_for(7, &selection.model_role));
    assert!(index.recording().load(0).is_ok());
    assert!(index.recording().load(1).is_err());
}
#[test]
fn cache_evicts_old_chunks_and_never_retains_a_chunk_larger_than_its_budget() {
    let root = Arc::new(std::path::PathBuf::from("fixture"));
    let key = |sequence| CacheKey {
        root: root.clone(),
        sequence,
    };
    let chunk = Arc::new(data(1));
    let charge = ChunkCache::charge(&chunk);
    let mut cache = ChunkCache::new(charge * 2);
    assert!(cache.insert(key(1), chunk.clone()));
    assert!(cache.insert(key(2), chunk.clone()));
    assert!(cache.get(&key(1)).is_some());
    assert!(cache.insert(key(3), chunk.clone()));
    assert!(cache.get(&key(2)).is_none());
    assert!(cache.resident_bytes() <= charge * 2);
    let mut small = ChunkCache::new(charge - 1);
    assert!(!small.insert(key(1), chunk));
    assert_eq!(small.resident_bytes(), 0);
}

#[test]
fn moving_training_time_keeps_the_source_record_and_rematches_its_input() {
    use rl::{core::env::Observation, diagnostics::probes::ProbeSet};
    use rl_viewer::plot_data::{matching_input, source_observation};
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let space = ActionSpace::Discrete { count: 2 };
    let mut writer = Recorder::create(
        &path,
        RunManifest::new("anchor", "test", "test", 1, space.clone(), 0.9).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for (k, values) in [(2, [10., 20.]), (7, [20., 10.])] {
        let set = ProbeSet::new(
            "reused",
            ProbePopulation::PolicyVisited,
            "fixture",
            values
                .into_iter()
                .map(|v| Observation::new(vec![v]))
                .collect(),
            None,
            space.clone(),
        )
        .unwrap();
        writer
            .append(&set.record(RecordContext::new("anchor", k)).unwrap())
            .unwrap();
    }
    writer.finish().unwrap();
    let index = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut selection = Selection {
        revision: 2,
        ..Default::default()
    };
    selection.select_probe(&index, "reused", 0);
    let anchored = selection.input.clone();
    selection.select_revision(&index, 7).unwrap();
    assert_eq!(selection.input, anchored);
    let SelectedInput::ProbeRow { sequence, row, .. } = selection.input else {
        panic!()
    };
    let source = index
        .recording()
        .load(index.by_sequence(sequence).unwrap())
        .unwrap();
    let target = index
        .recording()
        .load(index.probe_set("reused", 7).unwrap())
        .unwrap();
    assert_eq!(
        source_observation(&source, row).unwrap().dense_values(),
        [10.]
    );
    assert_eq!(matching_input(&source, row, &target).unwrap(), Some(1));
}
