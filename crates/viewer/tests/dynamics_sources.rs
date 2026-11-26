use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use rl_viewer::{
    data::{Loader, RunIndex},
    dynamics::source::{CatalogLimits, TrajectoryCatalog},
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

fn writer(path: &std::path::Path) -> Recorder {
    Recorder::create(
        path,
        RunManifest::new(
            "sources",
            "sources",
            "fixture",
            2,
            ActionSpace::Unbounded { dimensions: 1 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap()
}

fn trajectory(revision: u64, role: &str, conditional: bool, caption: &str) -> Chunk {
    let mut context = RecordContext::new("sources", revision);
    context.identity.model_role = role.into();
    let id = format!("episode-{revision}");
    context.trajectory = Some(id.clone());
    Chunk {
        context,
        record: Record::Trajectory(TrajectoryRecord {
            render: None,
            id,
            environment: "custom".into(),
            observations: "s".into(),
            actions: "a".into(),
            rewards: "r".into(),
            terminated: "term".into(),
            truncated: "trunc".into(),
            simulation_times: None,
            action_masks: None,
            frames: BTreeMap::new(),
            evaluation: EvaluationMetadata {
                conditioning: if conditional {
                    Conditioning::Observation {
                        methodology: "branched observation".into(),
                    }
                } else {
                    Conditioning::InitialScenario
                },
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: caption.into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "explicit environment and policy seeds".into(),
                truncation_handling: "reject incomplete episodes".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "s".into(),
                Array::f32(vec![3, 2], vec![0., 0., 1., 0., 1., 1.]).unwrap(),
            ),
            ("a".into(), Array::f32(vec![2, 1], vec![0.1, 0.2]).unwrap()),
            ("r".into(), Array::f32(vec![2], vec![1., 2.]).unwrap()),
            ("term".into(), Array::u8(vec![2], vec![0, 1]).unwrap()),
            ("trunc".into(), Array::u8(vec![2], vec![0, 0]).unwrap()),
        ]),
    }
}

fn metric(revision: u64) -> Chunk {
    Chunk {
        context: RecordContext::new("sources", revision),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::from([("loss".into(), 0.5)]),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    }
}

fn settle(catalog: &mut TrajectoryCatalog, run: &RunIndex, loader: &mut Loader) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        loader.begin_pass();
        loader.poll();
        catalog.poll(run, loader).unwrap();
        loader.finish_pass().unwrap();
        if !catalog.pending() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "trajectory catalog did not settle"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn authored_learner_snapshots_share_coverage_without_coalescing_other_policies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    for (revision, policy) in [(0, "sources@0"), (3, "sources@3"), (5, "Online learner")] {
        recorder
            .append(&trajectory(revision, "online", false, policy))
            .unwrap();
    }
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    let mut positions = catalog
        .groups()
        .iter()
        .map(|group| group.positions.as_slice())
        .collect::<Vec<_>>();
    positions.sort_unstable();
    assert_eq!(positions, [&[0, 3][..], &[5]]);
}

#[test]
fn catalog_completes_uncached_population_without_intermediate_paints() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    for revision in 0..12 {
        recorder
            .append(&trajectory(revision, "online", false, "policy"))
            .unwrap();
    }
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let mut loader = Loader::new(8 * 1024 * 1024, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    catalog.poll(&run, &mut loader).unwrap();
    // No poll/paint drives any of the remaining eleven uncached reads.
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let completed = catalog.metrics();
    assert_eq!(completed.generation, catalog.generation());
    assert_eq!(completed.submitted_generations, 1);
    assert_eq!(completed.started_generations, 1);
    assert_eq!(completed.completed_generations, 1);
    assert_eq!(completed.cancelled_generations, 0);
    assert_eq!(completed.read_attempts, 12);
    assert_eq!(completed.decoded_sources, 12);
    assert!(completed.submitted_at <= completed.started_at);
    assert!(completed.started_at <= completed.last_progress_at);
    assert!(completed.last_progress_at <= completed.completed_at);
    assert_eq!(
        completed.worker_duration,
        Some(completed.completed_at.unwrap() - completed.started_at.unwrap())
    );
    // A sufficiently fast worker may finish during the initial submission
    // poll. The source-local gate test proves delayed consumption explicitly.
    assert_eq!(catalog.progress(), (12, 12));
    loader.poll();
    catalog.poll(&run, &mut loader).unwrap();
    assert_eq!(catalog.progress(), (12, 12));
    assert_eq!(catalog.entries().len(), 12);
    let consumed = catalog.metrics();
    assert!(consumed.consumed_at.unwrap() >= completed.completed_at.unwrap());
    for _ in 0..20 {
        catalog.poll(&run, &mut loader).unwrap();
    }
    assert_eq!(
        catalog.metrics(),
        consumed,
        "idle paint must not restart work"
    );
}

#[test]
fn catalog_retry_supersedes_discovery_and_rechecks_disappeared_sources() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    for revision in 0..12 {
        recorder
            .append(&trajectory(revision, "online", false, "policy"))
            .unwrap();
    }
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let file = path.join(&run.recording().entries()[3].file);
    let saved = std::fs::read(&file).unwrap();
    let mut loader = Loader::new(8 << 20, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    catalog.poll(&run, &mut loader).unwrap();
    let generation = catalog.generation();
    std::fs::remove_file(&file).unwrap();
    catalog.retry(&run, &mut loader).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert!(catalog.generation() > generation);
    assert_eq!(catalog.rejected(), 1);
    assert_eq!(catalog.entries().len(), 11);
    assert!(!catalog.entries().iter().any(|entry| entry.revision == 3));
    std::fs::write(file, saved).unwrap();
    catalog.poll(&run, &mut loader).unwrap();
    assert_eq!(
        catalog.entries().len(),
        11,
        "failed generation remains stable until retry"
    );
    catalog.retry(&run, &mut loader).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.rejected(), 0);
    assert_eq!(
        catalog
            .entries()
            .iter()
            .map(|entry| entry.revision)
            .collect::<Vec<_>>(),
        (0..12).collect::<Vec<_>>()
    );
}

#[test]
fn sparse_evaluation_revisions_group_actual_roles_and_protocols() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    for k in 0..=12 {
        recorder.append(&metric(k)).unwrap();
    }
    recorder
        .append(&trajectory(2, "online", false, "policy"))
        .unwrap();
    recorder
        .append(&trajectory(2, "online", false, "policy"))
        .unwrap();
    let mut incomplete_batch_metadata = trajectory(8, "online", false, "policy");
    if let Record::Trajectory(t) = &mut incomplete_batch_metadata.record {
        t.evaluation.complete_returns = false;
    }
    recorder.append(&incomplete_batch_metadata).unwrap();
    recorder
        .append(&trajectory(4, "online", true, "policy"))
        .unwrap();
    recorder
        .append(&trajectory(6, "reference", false, "policy"))
        .unwrap();
    recorder
        .append(&trajectory(10, "online", false, "policy k10"))
        .unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.entries().len(), 5);
    assert_eq!(catalog.groups().len(), 3);
    let group = catalog
        .groups()
        .iter()
        .find(|g| g.protocol.unconditional() && g.positions == [2, 8])
        .unwrap();
    assert_eq!(group.episodes, 3);
    assert_eq!(
        catalog
            .groups()
            .iter()
            .filter(|g| !g.protocol.unconditional())
            .count(),
        1
    );
    assert!(catalog.entries().iter().all(|s| s.role == "online"));
    assert_eq!(
        catalog.entries().iter().filter(|s| s.revision == 2).count(),
        2
    );
    assert_ne!(catalog.entries()[0].key, catalog.entries()[1].key);
    assert_eq!(catalog.omitted(), 0);
    assert!(catalog.errors().is_empty());
}

#[test]
fn unrelated_metrics_keep_catalog_but_late_older_trajectory_rebuilds_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    recorder
        .append(&trajectory(8, "online", false, "policy"))
        .unwrap();
    let mut run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    let generation = catalog.generation();
    recorder.append(&metric(10)).unwrap();
    run.refresh().unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.generation(), generation);
    assert_eq!(catalog.groups()[0].positions, [8]);
    recorder
        .append(&trajectory(2, "online", false, "policy"))
        .unwrap();
    run.refresh().unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert!(catalog.generation() > generation);
    assert_eq!(catalog.groups()[0].positions, [2, 8]);
    recorder.finish().unwrap();
}

#[test]
fn bounded_catalog_discloses_newest_evidence_window_and_metadata_admission() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    for revision in [1, 3, 5, 7] {
        recorder
            .append(&trajectory(revision, "online", false, "policy"))
            .unwrap();
    }
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(8 * 1024 * 1024, || {}).unwrap();
    let limits = CatalogLimits {
        max_sources: 2,
        ..Default::default()
    };
    let mut catalog = TrajectoryCatalog::new(&run, "online", limits).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.groups()[0].positions, [5, 7]);
    assert_eq!(catalog.omitted(), 2);
    assert_eq!(catalog.metrics().read_attempts, 2);
    assert_eq!(catalog.metrics().decoded_sources, 2);
    assert!(catalog.retained_bytes() <= limits.max_bytes);
    let mut tiny = TrajectoryCatalog::new(
        &run,
        "online",
        CatalogLimits {
            max_bytes: 64,
            ..limits
        },
    )
    .unwrap();
    settle(&mut tiny, &run, &mut loader);
    assert!(tiny.entries().is_empty());
    assert!(!tiny.errors().is_empty());
    assert_eq!(tiny.metrics().read_attempts, 2);
    assert_eq!(tiny.metrics().decoded_sources, 2);
    assert_eq!(tiny.metrics().completed_generations, 1);
}

#[test]
fn no_trajectory_evidence_means_no_positions_or_coverage_claim() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    recorder.append(&metric(100)).unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert!(catalog.groups().is_empty());
    assert!(catalog.entries().is_empty());
    assert!(!catalog.pending());
    assert_eq!(catalog.metrics().submitted_generations, 0);
    assert_eq!(catalog.metrics().read_attempts, 0);
    assert_eq!(catalog.metrics().completed_at, None);
}

#[test]
fn catalog_missing_source_counts_attempt_without_decode_and_recovers_only_on_retry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    recorder
        .append(&trajectory(1, "online", false, "policy"))
        .unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let file = path.join(&run.recording().entries()[0].file);
    let saved = std::fs::read(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    let mut loader = Loader::new(8 << 20, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.rejected(), 1);
    assert_eq!(catalog.metrics().read_attempts, 1);
    assert_eq!(catalog.metrics().decoded_sources, 0);
    let failed = catalog.metrics();
    std::fs::write(file, saved).unwrap();
    for _ in 0..20 {
        catalog.poll(&run, &mut loader).unwrap();
    }
    assert_eq!(catalog.metrics(), failed);
    catalog.retry(&run, &mut loader).unwrap();
    assert_eq!(catalog.metrics().completed_at, None);
    assert_eq!(catalog.metrics().consumed_at, None);
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.entries().len(), 1);
    assert_eq!(catalog.rejected(), 0);
    assert_eq!(catalog.metrics().submitted_generations, 2);
    assert_eq!(catalog.metrics().completed_generations, 2);
    assert_eq!(catalog.metrics().read_attempts, 2);
    assert_eq!(catalog.metrics().decoded_sources, 1);
}

#[test]
fn different_recorded_environments_do_not_share_a_coverage_protocol() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut recorder = writer(&path);
    recorder
        .append(&trajectory(1, "online", false, "policy"))
        .unwrap();
    let mut other = trajectory(2, "online", false, "policy");
    if let Record::Trajectory(t) = &mut other.record {
        t.environment = "different simulator".into();
    }
    recorder.append(&other).unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
    settle(&mut catalog, &run, &mut loader);
    assert_eq!(catalog.groups().len(), 2);
    assert_eq!(catalog.groups()[0].positions, [1]);
    assert_eq!(catalog.groups()[1].positions, [2]);
}
