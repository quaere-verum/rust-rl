use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{data::RunIndex, learning::LearningWorker};
use std::{
    collections::BTreeMap,
    sync::{Arc, mpsc},
    time::Duration,
};

fn recording_open() -> (tempfile::TempDir, RunIndex, Recorder) {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("run");
    let space = ActionSpace::Box {
        low: vec![-1., -3.],
        high: vec![1., 3.],
    };
    let m = RunManifest::new("learning", "history", "fixture", 1, space.clone(), 0.9).unwrap();
    let mut w = Recorder::create(&path, m, StorageConfig::default()).unwrap();
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "stable population",
        vec![Observation::new(vec![0.]), Observation::new(vec![1.])],
        None,
        space,
    )
    .unwrap();
    let mut c = RecordContext::new("learning", 1);
    c.probe_set = Some("fixed".into());
    w.append(&set.record(c.clone()).unwrap()).unwrap();
    for (revision, values) in [(1, vec![0., 0., 0., 0.]), (7, vec![0., 3., 0., 3.])] {
        c.identity.learner_update_revision = revision;
        c.counters.updates = revision;
        w.append(&Chunk {
            context: c.clone(),
            record: Record::Policy(PolicyRecord {
                actions: "a".into(),
                parameters: PolicyParameters::Deterministic,
                entropy: None,
                entropy_method: None,
                masks: None,
                stochastic: false,
            }),
            arrays: BTreeMap::from([("a".into(), Array::f64(vec![2, 2], values).unwrap())]),
        })
        .unwrap();
    }
    for (revision, returns) in [(1, vec![1., 3., 5.]), (4, vec![7.])] {
        let n = returns.len();
        w.append(&Chunk {
            context: RecordContext::new("learning", revision),
            record: Record::Empirical(EmpiricalRecord {
                returns: "returns".into(),
                seeds: "seeds".into(),
                probe_row: None,
                initial_action: None,
                conditioning_observation: None,
                conditioning_mask: None,
                simulator_state: None,
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "learner evaluation".into(),
                    stochastic_policy: false,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "independent explicit futures".into(),
                    truncation_handling: "true terminal".into(),
                    complete_returns: true,
                },
            }),
            arrays: BTreeMap::from([
                ("returns".into(), Array::f64(vec![n], returns).unwrap()),
                (
                    "seeds".into(),
                    Array::u64(vec![n], (0..n as u64).collect()).unwrap(),
                ),
            ]),
        })
        .unwrap();
    }
    // A training metric never supplies evaluation performance or critic error.
    w.append(&Chunk {
        context: RecordContext::new("learning", 5),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::from([("loss".into(), 0.)]),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    })
    .unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    (d, run, w)
}
fn recording() -> (tempfile::TempDir, RunIndex) {
    let (d, run, mut w) = recording_open();
    w.finish().unwrap();
    (d, run)
}
fn worker() -> (LearningWorker, mpsc::Receiver<()>) {
    let (tx, rx) = mpsc::channel();
    (
        LearningWorker::new(move || {
            let _ = tx.send(());
        })
        .unwrap(),
        rx,
    )
}
fn settle(worker: &mut LearningWorker, rx: &mpsc::Receiver<()>) {
    while worker.pending() {
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
    }
}

#[test]
fn history_uses_actual_revisions_frozen_earliest_anchor_and_missing_measurement_gaps() {
    let (_d, run) = recording();
    let (mut worker, rx) = worker();
    assert!(worker.request(&run, "online").unwrap());
    settle(&mut worker, &rx);
    let f = worker.frame().unwrap();
    let a = f.anchor.as_ref().unwrap();
    assert_eq!(a.revision, 1);
    assert_eq!(a.rows, 2);
    assert_eq!(a.normalization, vec![2., 6.]);
    assert_eq!(
        f.evaluations
            .iter()
            .filter(|p| p.value.is_some())
            .map(|p| (p.revision, p.count, p.value.unwrap()))
            .collect::<Vec<_>>(),
        vec![(1, 3, 3.), (4, 1, 7.)]
    );
    assert!(
        f.policy_changes
            .iter()
            .any(|p| p.revision == 4 && p.value.is_none() && p.reason.is_some())
    );
    assert!(f.evaluations.iter().all(|p| p.revision != 5));
    assert!(f.critic_errors.iter().all(|p| p.value.is_none()));
    let metrics = worker.metrics();
    eprintln!(
        "shared probe history: reads={} decodes={}",
        metrics.reads, metrics.decodes
    );
    assert_eq!(metrics.reads, 6);
    assert!(
        metrics.decodes < metrics.reads,
        "shared original probes should decode once while every logical read remains"
    );
    assert!(
        (f.policy_changes
            .iter()
            .find(|p| p.revision == 7)
            .unwrap()
            .value
            .unwrap()
            - 0.353_553_390_593_273_8)
            .abs()
            < 1e-14
    );
}

#[test]
fn unchanged_requests_do_not_read_again_and_retained_frames_stay_charged_after_clear() {
    let (_d, run) = recording();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    let held = worker.frame().unwrap().clone();
    let m = worker.metrics();
    assert!(!worker.request(&run, "online").unwrap());
    assert_eq!(worker.metrics().reads, m.reads);
    worker.clear();
    assert!(worker.frame().is_none());
    assert!(worker.metrics().owned_bytes > 0);
    assert!(worker.metrics().owned_bytes <= 64 * 1024 * 1024);
    let weak = Arc::downgrade(&held);
    drop(held);
    assert!(weak.upgrade().is_none());
}

#[test]
fn newer_recording_request_supersedes_old_work_and_source_deletion_invalidates_a_frame() {
    let (_d, run) = recording();
    let (_next, next) = recording();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    worker.request(&next, "online").unwrap();
    settle(&mut worker, &rx);
    assert_eq!(worker.frame().unwrap().root, next.root().clone());
    let entry = next
        .recording()
        .entries()
        .iter()
        .find(|e| e.kind == RecordKind::Empirical)
        .unwrap();
    std::fs::remove_file(next.root().join(&entry.file)).unwrap();
    assert!(worker.recheck());
    settle(&mut worker, &rx);
    assert!(worker.frame().is_none());
    assert!(worker.error().is_some());
}

#[test]
fn frontier_refresh_keeps_frozen_anchor_and_ignores_unrelated_training_metrics() {
    let (_d, mut run, mut w) = recording_open();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    let anchor = worker.frame().unwrap().anchor.as_ref().unwrap().sequence;
    w.append(&Chunk {
        context: RecordContext::new("learning", 8),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::new(),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::new(),
    })
    .unwrap();
    run.refresh().unwrap();
    assert!(!worker.request(&run, "online").unwrap());
    let earlier = Chunk {
        context: {
            let mut c = RecordContext::new("learning", 0);
            c.probe_set = Some("fixed".into());
            c
        },
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        arrays: BTreeMap::from([("a".into(), Array::f64(vec![2, 2], vec![0.; 4]).unwrap())]),
    };
    w.append(&earlier).unwrap();
    run.refresh().unwrap();
    assert!(worker.request(&run, "online").unwrap());
    settle(&mut worker, &rx);
    assert_eq!(
        worker.frame().unwrap().anchor.as_ref().unwrap().sequence,
        anchor
    );
}

#[test]
fn retry_after_pruning_keeps_other_measurements_and_discloses_the_missing_source() {
    let (_d, run) = recording();
    let entry = run
        .recording()
        .entries()
        .iter()
        .find(|e| e.kind == RecordKind::Empirical && e.revision == 1)
        .unwrap();
    std::fs::remove_file(run.root().join(&entry.file)).unwrap();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    let frame = worker.frame().unwrap();
    assert!(
        frame
            .evaluations
            .iter()
            .any(|p| p.revision == 4 && p.value == Some(7.))
    );
    assert!(
        frame
            .evaluations
            .iter()
            .any(|p| p.revision == 1 && p.value.is_none() && p.reason.is_some())
    );
}

#[test]
fn readmission_after_an_unavailable_anchor_does_not_silently_choose_a_later_policy() {
    let (_d, mut run, mut w) = recording_open();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    let sequence = worker.frame().unwrap().anchor.as_ref().unwrap().sequence;
    let source = run
        .recording()
        .entries()
        .iter()
        .find(|e| e.sequence == sequence)
        .unwrap();
    std::fs::remove_file(run.root().join(&source.file)).unwrap();
    assert!(worker.recheck());
    settle(&mut worker, &rx);
    assert!(worker.frame().is_none());
    worker.retry();
    // Refresh the relevant frontier while the old immutable reference is pinned.
    let source = run
        .recording()
        .entries()
        .iter()
        .position(|e| e.kind == RecordKind::Empirical && e.revision == 4)
        .unwrap();
    let mut c = run.recording().load(source).unwrap();
    c.context.identity.learner_update_revision = 9;
    c.context.counters.updates = 9;
    w.append(&c).unwrap();
    run.refresh().unwrap();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    let frame = worker.frame().unwrap();
    assert_eq!(frame.anchor.as_ref().unwrap().sequence, sequence);
    assert!(frame.policy_changes.iter().all(|p| p.value.is_none()));
}

#[test]
fn bounded_source_window_discloses_omissions_and_owner_leases_refuse_excess_overlap() {
    let (directory, mut run, mut writer) = recording_open();
    let index = run
        .recording()
        .entries()
        .iter()
        .position(|e| e.kind == RecordKind::Empirical && e.revision == 4)
        .unwrap();
    let mut sample = run.recording().load(index).unwrap();
    for revision in 10..525 {
        sample.context.identity.learner_update_revision = revision;
        sample.context.counters.updates = revision;
        writer.append(&sample).unwrap();
    }
    run.refresh().unwrap();
    let (mut worker, rx) = worker();
    worker.request(&run, "online").unwrap();
    settle(&mut worker, &rx);
    writer.finish().unwrap();
    drop(writer);
    let frame = worker.frame().unwrap();
    assert_eq!(frame.omitted_sources, 7);
    assert!(frame.retained_bytes <= 2 * 1024 * 1024);
    assert_eq!(
        frame
            .evaluations
            .iter()
            .filter(|p| p.value.is_some())
            .count(),
        512
    );
    // All decoded evidence from this first window is dropped before new runs;
    // external summaries keep their descriptor/summary leases until final drop.
    let mut held = vec![frame.clone()];
    let mut directories = vec![directory];
    let mut refused = false;
    for _ in 0..12 {
        let (d, next) = recording();
        directories.push(d);
        match worker.request(&next, "online") {
            Ok(_) => {
                settle(&mut worker, &rx);
                if let Some(f) = worker.frame() {
                    held.push(f.clone());
                } else {
                    refused = true;
                    break;
                }
            }
            Err(_) => {
                refused = true;
                break;
            }
        }
        assert!(worker.metrics().owned_bytes <= 64 * 1024 * 1024);
    }
    assert!(
        refused,
        "unbounded retained frame overlap must not escape ownership admission"
    );
    worker.clear();
    drop(held);
    assert_eq!(worker.metrics().owned_bytes, 0);
}

#[test]
fn unsupported_manifest_admission_settles_until_explicit_retry_instead_of_repeating_on_paint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut manifest = RunManifest::new(
        "large-metadata",
        "large-metadata",
        "fixture",
        1,
        ActionSpace::Discrete { count: 2 },
        0.9,
    )
    .unwrap();
    manifest
        .metadata
        .insert("oversized".into(), "x".repeat(140 * 1024));
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let (mut worker, _rx) = worker();
    assert!(!worker.request(&run, "online").unwrap());
    assert!(worker.error().is_some());
    let counters = worker.metrics();
    assert!(!worker.request(&run, "online").unwrap());
    assert_eq!(worker.metrics().submitted, counters.submitted);
    assert_eq!(worker.metrics().reads, 0);
}
