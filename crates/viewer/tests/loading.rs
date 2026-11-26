use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use rl_viewer::data::*;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

fn fixture(count: u64) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "loading",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for revision in 0..count {
        writer
            .append(&Chunk {
                context: RecordContext::new("loading", revision),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), revision as f64)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
    }
    writer.finish().unwrap();
    (dir, RunIndex::open(path, ReadLimits::default()).unwrap())
}

fn wait_for_completion(loader: &mut Loader) {
    let deadline = Instant::now() + Duration::from_secs(3);
    let before = loader.completed_loads;
    while loader.completed_loads == before {
        loader.poll();
        assert!(Instant::now() < deadline, "reader did not complete");
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn one_pass_preserves_independent_reads_and_deduplicates_shared_dependencies() {
    let (_dir, run) = fixture(4);
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    loader.begin_pass();
    for record in [0, 1, 0, 2, 1] {
        assert!(loader.request(&run, record).unwrap().is_none());
    }
    loader.finish_pass().unwrap();
    assert_eq!(loader.pending(), 3);
    wait_for_completion(&mut loader);
    assert!(loader.cache.get(&run.key(0).unwrap()).is_some());
    assert_eq!(
        loader.pending(),
        2,
        "poll must not dispatch work before a new pass publishes its demands"
    );
    loader.begin_pass();
    loader.request(&run, 1).unwrap();
    loader.request(&run, 2).unwrap();
    loader.finish_pass().unwrap();
    wait_for_completion(&mut loader);
    assert!(loader.cache.get(&run.key(1).unwrap()).is_some());
}

#[test]
fn fifo_deduplication_and_queue_bound_preserve_the_first_current_dependency() {
    let (_dir, run) = fixture(70);
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    loader.begin_pass();
    for record in [0, 1, 0, 2, 1] {
        loader.request(&run, record).unwrap();
    }
    loader.finish_pass().unwrap();
    assert_eq!(loader.pending(), 3);
    wait_for_completion(&mut loader);
    assert!(loader.cache.get(&run.key(0).unwrap()).is_some());
    assert!(loader.cache.get(&run.key(1).unwrap()).is_none());
    loader.begin_pass();
    for record in 1..70 {
        loader.request(&run, record).unwrap();
    }
    loader.finish_pass().unwrap();
    assert_eq!(
        loader.pending(),
        32,
        "one active read plus the remaining admitted desires"
    );
    wait_for_completion(&mut loader);
    assert!(loader.cache.get(&run.key(1).unwrap()).is_some());
}

#[test]
fn standalone_poll_drains_dependencies_once_in_fifo_order() {
    let (_dir, run) = fixture(3);
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    for record in [0, 1, 0, 2, 1] {
        loader.request(&run, record).unwrap();
    }
    assert_eq!(loader.pending(), 3);
    let deadline = Instant::now() + Duration::from_secs(3);
    while loader.pending() != 0 {
        loader.poll();
        assert!(Instant::now() < deadline, "standalone reader did not drain");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(loader.completed_loads, 3);
    for record in 0..3 {
        assert!(loader.request(&run, record).unwrap().is_some());
    }
}

#[test]
fn failed_immutable_keys_retry_only_after_explicit_invalidation() {
    let (_dir, run) = fixture(1);
    let path = run.root().join(&run.recording().entries()[0].file);
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    loader.request(&run, 0).unwrap();
    wait_for_completion(&mut loader);
    assert!(loader.request(&run, 0).is_err());
    std::fs::write(&path, bytes).unwrap();
    assert!(
        loader.request(&run, 0).is_err(),
        "repair must not trigger per-frame stat/retry loops"
    );
    assert!(loader.retry_failed(&run.key(0).unwrap()));
    assert!(!loader.retry_failed(&run.key(0).unwrap()));
    assert!(loader.request(&run, 0).unwrap().is_none());
    wait_for_completion(&mut loader);
    assert!(loader.request(&run, 0).unwrap().is_some());
    loader.invalidate();
    assert_eq!(loader.cache.resident_bytes(), 0);
}

#[test]
fn metadata_retirement_blocks_a_previously_cached_snapshot_without_file_stat() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retention");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "retention",
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
    .with_snapshot_retention(1)
    .unwrap();
    let snapshot = |revision| Chunk {
        context: RecordContext::new("retention", revision),
        record: Record::ModelSnapshot(ModelSnapshotRecord {
            id: format!("k{revision}"),
            architecture: "fixture".into(),
            configuration: serde_json::json!({}),
            capabilities: vec![],
            models: BTreeMap::from([(
                "policy".into(),
                ModelArtifact {
                    bytes: "weights".into(),
                    signature: "fixture".into(),
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
        }),
        arrays: BTreeMap::from([("weights".into(), Array::u8(vec![1], vec![1]).unwrap())]),
    };
    writer.append(&snapshot(1)).unwrap();
    let mut run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    loader.request(&run, 0).unwrap();
    wait_for_completion(&mut loader);
    assert!(loader.request(&run, 0).unwrap().is_some());
    writer.append(&snapshot(2)).unwrap();
    run.refresh().unwrap();
    assert!(
        loader.request(&run, 0).is_err(),
        "retention metadata is checked before a raw cache hit"
    );
    loader.request(&run, 1).unwrap();
    wait_for_completion(&mut loader);
    assert!(loader.request(&run, 1).unwrap().is_some());
}
