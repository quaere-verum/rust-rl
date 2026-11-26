#[path = "support/dynamics.rs"]
mod fixture;
use rl::diagnostics::storage::ReadLimits;
use rl_viewer::{
    data::{Loader, RunIndex},
    dynamics::{
        ActionAxis, Axes,
        source::{CatalogLimits, TrajectoryCatalog},
        work::{DomainSpec, DynamicsWorker, HitView, Projection, Request, SourceSet, WorkLimits},
    },
};
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

fn sources() -> (tempfile::TempDir, Arc<SourceSet>) {
    let (dir, _, _) = fixture::saved(fixture::bounded(3));
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut loader = Loader::new(1 << 20, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "target", CatalogLimits::default()).unwrap();
    while catalog.pending() {
        loader.poll();
        catalog.poll(&run, &mut loader).unwrap();
        if catalog.pending() {
            rx.recv_timeout(Duration::from_secs(10)).unwrap();
        }
    }
    let sources = SourceSet::new(
        &run,
        catalog.entries(),
        catalog.generation(),
        &WorkLimits::default(),
    )
    .unwrap();
    (dir, sources)
}
fn request(sources: Arc<SourceSet>) -> Request {
    Request {
        selected: sources.sources()[0].key.clone(),
        sources,
        revision: 17,
        axes: Axes { x: 0, y: 1 },
        cells: [2, 2],
        action: ActionAxis::Continuous(0),
        projection: Projection::StateAction,
        reference: None,
        domain: DomainSpec::default(),
    }
}

#[test]
fn availability_is_idle_coalesced_and_detects_deletion_without_rebuilding() {
    let (_dir, run, sources) = multiple_sources_with_run(&[17, 37], false, &WorkLimits::default());
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame_arc().unwrap();
    let counters = worker.counters();
    assert!(worker.request_availability(&frame, &sources));
    assert!(!worker.request_availability(&frame, &sources));
    assert!(!worker.pending());
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(!worker.pending());
    assert_eq!(worker.poll_availability(&frame, &sources), Some(Ok(())));
    assert_eq!(worker.poll_availability(&frame, &sources), None);
    assert!(rx.try_recv().is_err());
    assert!(Arc::ptr_eq(&frame, &worker.frame_arc().unwrap()));
    assert_eq!(worker.counters(), counters);
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    assert!(worker.request_availability(&frame, &sources));
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let error = worker
        .poll_availability(&frame, &sources)
        .unwrap()
        .unwrap_err();
    assert!(!error.is_empty());
    assert!(error.chars().count() <= 512);
    assert_eq!(worker.counters(), counters);
    assert!(Arc::ptr_eq(&frame, &worker.frame_arc().unwrap()));
}

#[test]
fn availability_checks_pinned_sources_outside_the_current_window() {
    let (_dir, run, all) = multiple_sources_with_run(&[17, 37, 57], false, &WorkLimits::default());
    let initial = SourceSet::new(&run, &all.sources()[..1], 1, &WorkLimits::default()).unwrap();
    let current = SourceSet::new(&run, &all.sources()[2..], 2, &WorkLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(initial)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let reference = worker.frame().unwrap().state.reference.clone();
    let mut next = request(current.clone());
    next.revision = 57;
    next.reference = Some(reference.clone());
    worker.request(next).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame_arc().unwrap();
    assert!(Arc::ptr_eq(&reference, &frame.state.reference));
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    assert!(worker.request_availability(&frame, &current));
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(worker.poll_availability(&frame, &current), Some(Ok(())));
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    assert!(worker.request_availability(&frame, &current));
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(
        worker
            .poll_availability(&frame, &current)
            .unwrap()
            .unwrap_err()
            .contains("fixed-reference source unavailable")
    );
}

#[test]
fn exact_click_worker_preserves_terminal_ties_threshold_and_supersession() {
    let (_dir, run, sources) = multiple_sources_with_run(&[17], false, &WorkLimits::default());
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame_arc().unwrap();
    let view = HitView {
        axes: Axes { x: 0, y: 1 },
        bounds: [[0., 2.], [0., 2.]],
        chart: eframe::egui::Rect::from_min_max(
            eframe::egui::pos2(0., 0.),
            eframe::egui::pos2(40., 40.),
        ),
    };
    for (pointer, expected) in [
        ([0., 0.], Some(4)),   // Exact terminal observation.
        ([10., 40.], Some(0)), // Equal distance chooses the earliest index.
        ([-15., 40.], None),   // Original threshold is strict.
        ([-14.99, 40.], Some(0)),
    ] {
        worker
            .request_hit(&frame, sources.clone(), view, pointer.into())
            .unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(worker.poll_hit(&frame, view), expected);
    }
    worker
        .request_hit(&frame, sources.clone(), view, [0., 40.].into())
        .unwrap();
    worker
        .request_hit(&frame, sources.clone(), view, [40., 0.].into())
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut selected = None;
    while worker.pending() {
        rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
            .unwrap();
        selected = worker.poll_hit(&frame, view).or(selected);
    }
    assert_eq!(selected, Some(3));
    worker
        .request_hit(&frame, sources.clone(), view, [0., 0.].into())
        .unwrap();
    while worker.pending() {
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let resized = HitView {
            chart: view.chart.expand(1.),
            ..view
        };
        assert_eq!(worker.poll_hit(&frame, resized), None);
    }
    let changed = SourceSet::new(
        &run,
        sources.sources(),
        sources.generation() + 1,
        &WorkLimits::default(),
    )
    .unwrap();
    assert!(
        worker
            .request_hit(&frame, changed, view, [0., 0.].into())
            .is_err()
    );
    assert!(
        worker
            .request_hit(
                &frame,
                sources.clone(),
                HitView {
                    axes: Axes { x: 1, y: 0 },
                    ..view
                },
                [0., 0.].into()
            )
            .is_err()
    );
    worker
        .request_hit(&frame, sources, view, [0., 0.].into())
        .unwrap();
    worker.retire();
    assert_eq!(worker.poll_hit(&frame, view), None);
}

#[test]
fn repeated_exact_points_choose_first_and_deleted_sources_do_not_answer_clicks() {
    let (_dir, run, sources) = multiple_sources_with_observations(
        &[17],
        false,
        &WorkLimits::default(),
        Some(vec![
            0., 0., 10., 1., 0., 11., 0., 0., 12., 2., 2., 13., 0., 0., 14.,
        ]),
    );
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame_arc().unwrap();
    let view = HitView {
        axes: Axes { x: 0, y: 1 },
        bounds: [[0., 2.], [0., 2.]],
        chart: eframe::egui::Rect::from_min_max([0., 0.].into(), [40., 40.].into()),
    };
    worker
        .request_hit(&frame, sources.clone(), view, [0., 40.].into())
        .unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(worker.poll_hit(&frame, view), Some(0));
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    worker
        .request_hit(&frame, sources, view, [0., 40.].into())
        .unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(worker.poll_hit(&frame, view), None);
}

#[test]
fn separately_retained_histories_cannot_escape_the_old_frame_lease_budget() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut histories = Vec::new();
    let mut req = request(sources);
    for cells in [2, 3, 4] {
        req.cells = [cells, cells];
        worker.request(req.clone()).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
        histories.push(worker.frame().unwrap().state.coverage.clone());
    }
    req.cells = [5, 5];
    assert!(
        worker
            .request(req.clone())
            .unwrap_err()
            .to_string()
            .contains("leases")
    );
    histories.clear();
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert_eq!(
        worker.frame().unwrap().state.coverage.display_grid_cells,
        25
    );
}
#[test]
fn action_only_reuses_state_reward_and_exact_cursor() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources);
    assert!(worker.request(req.clone()).unwrap());
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.state.occupancy.counts, [1, 1, 0, 2]);
    assert_eq!(
        frame.state.reference.state_bounds(2),
        Some([0., 2.]),
        "declared coordinate domains take precedence"
    );
    assert_eq!(
        frame.state.reference.state_extrema(2),
        Some([10., 14.]),
        "observed extrema remain available separately"
    );
    assert_eq!(
        frame.selected.episode.project(4, req.axes).unwrap(),
        [0., 2.]
    );
    assert_eq!(
        frame.selected.episode.action_value(4, req.action).unwrap(),
        None
    );
    let state = frame.state.clone();
    let selected = frame.selected.clone();
    req.action = ActionAxis::Continuous(2);
    assert!(worker.request(req.clone()).unwrap());
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert!(Arc::ptr_eq(&state, &frame.state));
    assert!(Arc::ptr_eq(&selected, &frame.selected));
    assert_eq!(
        frame.action_trace[0].value,
        rl_viewer::dynamics::ActionValue::Continuous(20.)
    );
    assert_eq!(frame.geometry3[0].points[0][2], 20.);
    assert!(!worker.request(req).unwrap());
    let before = worker.counters();
    for _ in 0..100 {
        worker.poll();
        let _ = worker.frame();
    }
    assert_eq!(worker.counters(), before);
}

#[test]
fn superseded_run_axes_revision_action_and_retirement_never_publish_stale() {
    use std::sync::Mutex;
    let (_dir, sources) = sources();
    let (_other_dir, other_sources) = self::sources();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let reader_calls = calls.clone();
    let (wake_tx, wake_rx) = mpsc::channel();
    let mut worker = DynamicsWorker::with_reader(
        WorkLimits::default(),
        move || {
            let _ = wake_tx.send(());
        },
        move |source, limit| {
            if reader_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            source.load_if_within_charge(limit)
        },
    )
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let mut replaced = request(sources);
    replaced.axes = Axes { x: 1, y: 2 };
    replaced.action = ActionAxis::Continuous(2);
    worker.request(replaced).unwrap();
    worker.retire(); // consumes desired, despite an active read and future wake.
    let mut latest = request(other_sources);
    latest.action = ActionAxis::Continuous(1);
    worker.request(latest.clone()).unwrap();
    release_tx.send(()).unwrap();
    wake_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.selected.key, latest.selected);
    assert_eq!(frame.action, latest.action);
    assert_eq!(frame.state.revision, latest.revision);
    assert_eq!(worker.counters().cancelled, 1);
    assert_eq!(worker.counters().state_builds, 1);
    assert_eq!(worker.counters().source_reads, 2);
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    assert!(!worker.pending());
}

#[test]
fn vanished_source_and_failed_preparation_are_stable_until_explicit_retry() {
    use std::sync::Mutex;
    let (dir, sources) = sources();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let (wake_tx, wake_rx) = mpsc::channel();
    let mut worker = DynamicsWorker::with_reader(
        WorkLimits::default(),
        move || {
            let _ = wake_tx.send(());
        },
        move |source, limit| {
            let loaded = source.load_if_within_charge(limit)?;
            entered_tx.send(()).unwrap();
            release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            Ok(loaded)
        },
    )
    .unwrap();
    let req = request(sources.clone());
    worker.request(req.clone()).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    // Delete only this fixture's real Recorder-produced source.
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let file = dir
        .path()
        .join("run")
        .join(&run.recording().entries()[0].file);
    std::fs::remove_file(file).unwrap();
    release_tx.send(()).unwrap();
    wake_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(worker.frame().is_none());
    assert!(worker.error().unwrap().contains("missing"));
    assert!(sources.check_available().is_err());
    let before = worker.counters();
    for _ in 0..10 {
        assert!(!worker.request(req.clone()).unwrap());
        worker.poll();
    }
    assert_eq!(before, worker.counters());
    worker.retry();
    assert!(worker.request(req).unwrap());
    wake_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(worker.error().is_some());
    assert_eq!(worker.counters().source_reads, 1);
}

#[test]
fn retirement_after_queued_answer_drops_it_and_reentry_rebuilds() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let req = request(sources);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap(); // Answer ready; not polled.
    worker.retire();
    worker.poll();
    assert!(worker.frame().is_none());
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(worker.frame().is_some());
    assert_eq!(worker.counters().state_builds, 2);
}

#[test]
fn sampled_geometry_retains_real_adjacent_edges_and_exact_endpoints() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let limits = WorkLimits {
        max_segments: 2,
        max_trace_points: 2,
        ..Default::default()
    };
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let req = request(sources);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let f = worker.frame().unwrap();
    assert_eq!(
        f.selected
            .path
            .segments
            .iter()
            .map(|s| s.t)
            .collect::<Vec<_>>(),
        [0, 3]
    );
    for edge in &f.selected.path.segments {
        assert_eq!(
            edge.points,
            [
                f.selected.episode.project(edge.t, req.axes).unwrap(),
                f.selected.episode.project(edge.t + 1, req.axes).unwrap()
            ]
        );
    }
    assert_eq!(
        f.selected.path.end,
        f.selected.episode.project(4, req.axes).unwrap()
    );
    assert_eq!(
        f.action_trace.iter().map(|s| s.t).collect::<Vec<_>>(),
        [0, 3]
    );
    assert_eq!(f.geometry3.iter().map(|s| s.t).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(
        f.geometry3.capacity(),
        f.geometry3.len(),
        "selected geometry has no uncharged spare capacity"
    );
    assert!(
        f.background
            .paths
            .iter()
            .all(|p| p.segments.capacity() == p.segments.len())
    );
    assert_eq!(f.state.displayed_segments, 2);
    assert_eq!(f.state.population_decisions, 4);
    assert_eq!(f.state.occupancy.included, 4);
}

#[test]
fn owner_envelope_overflow_admission_and_external_frame_leases_are_bounded() {
    assert!(
        WorkLimits {
            max_read_bytes: usize::MAX,
            ..Default::default()
        }
        .envelope_bytes()
        .is_err()
    );
    assert!(
        WorkLimits {
            max_trace_points: usize::MAX,
            ..Default::default()
        }
        .envelope_bytes()
        .is_err()
    );
    assert!(
        WorkLimits {
            max_owner_bytes: 1,
            ..Default::default()
        }
        .envelope_bytes()
        .is_err()
    );
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let limits = WorkLimits::default();
    let expected = limits.envelope_bytes().unwrap();
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    assert_eq!(worker.envelope_bytes(), expected);
    let mut req = request(sources);
    let mut held = vec![];
    for component in 0..3 {
        req.action = ActionAxis::Continuous(component);
        worker.request(req.clone()).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
        let frame = worker.frame_arc().unwrap();
        assert!(frame.retained_bytes <= WorkLimits::default().max_frame_bytes);
        held.push(frame);
    }
    req.action = ActionAxis::Continuous(0);
    assert!(
        worker
            .request(req.clone())
            .unwrap_err()
            .to_string()
            .contains("leases")
    );
    held.clear();
    assert!(worker.request(req).unwrap());
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
}

fn multiple_sources(
    revisions: &[u64],
    empirical: bool,
    limits: &WorkLimits,
) -> (tempfile::TempDir, Arc<SourceSet>) {
    let (dir, _, sources) = multiple_sources_with_run(revisions, empirical, limits);
    (dir, sources)
}
fn multiple_sources_with_run(
    revisions: &[u64],
    empirical: bool,
    limits: &WorkLimits,
) -> (tempfile::TempDir, RunIndex, Arc<SourceSet>) {
    multiple_sources_with_observations(revisions, empirical, limits, None)
}
fn multiple_sources_with_observations(
    revisions: &[u64],
    empirical: bool,
    limits: &WorkLimits,
    observations: Option<Vec<f64>>,
) -> (tempfile::TempDir, RunIndex, Arc<SourceSet>) {
    multiple_sources_with_episode_observations(
        revisions,
        empirical,
        limits,
        observations.map(|values| vec![values; revisions.len()]),
        None,
    )
}
fn multiple_sources_with_episode_observations(
    revisions: &[u64],
    empirical: bool,
    limits: &WorkLimits,
    observations: Option<Vec<Vec<f64>>>,
    actions: Option<Vec<Vec<f64>>>,
) -> (tempfile::TempDir, RunIndex, Arc<SourceSet>) {
    use rl::diagnostics::{
        schema::Record,
        storage::{Recorder, StorageConfig},
    };
    let dir = tempfile::tempdir().unwrap();
    let space = fixture::bounded(3);
    let mut manifest = fixture::manifest(space.clone());
    if empirical {
        manifest.environment_descriptor = None;
    }
    let mut recorder =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (i, &revision) in revisions.iter().enumerate() {
        let mut chunk = fixture::chunk(&space);
        if let Some(observations) = &observations {
            chunk.arrays.insert(
                "obs".into(),
                rl::diagnostics::schema::Array::f64(
                    vec![observations[i].len() / 3, 3],
                    observations[i].clone(),
                )
                .unwrap(),
            );
        }
        let decisions = observations
            .as_ref()
            .map_or(4, |values| values[i].len() / 3 - 1);
        if decisions != 4 {
            use rl::diagnostics::schema::Array;
            chunk.arrays.insert(
                "actions".into(),
                Array::f64(
                    vec![decisions, 3],
                    (0..decisions)
                        .flat_map(|t| (0..3).map(move |j| (10 * j + t) as f64))
                        .collect(),
                )
                .unwrap(),
            );
            chunk.arrays.insert(
                "rewards".into(),
                Array::f64(vec![decisions], vec![1.; decisions]).unwrap(),
            );
            let mut terminated = vec![0; decisions];
            terminated[decisions - 1] = 1;
            chunk.arrays.insert(
                "terminated".into(),
                Array::u8(vec![decisions], terminated).unwrap(),
            );
            chunk.arrays.insert(
                "truncated".into(),
                Array::u8(vec![decisions], vec![0; decisions]).unwrap(),
            );
            chunk.arrays.insert(
                "times".into(),
                Array::f64(
                    vec![decisions + 1],
                    (0..=decisions).map(|t| t as f64).collect(),
                )
                .unwrap(),
            );
        }
        if let Some(actions) = &actions {
            chunk.arrays.insert(
                "actions".into(),
                rl::diagnostics::schema::Array::f64(vec![decisions, 3], actions[i].clone())
                    .unwrap(),
            );
        }
        chunk.context.identity.learner_update_revision = revision;
        chunk.context.counters.updates = revision;
        chunk.context.trajectory = Some(format!("episode-{i}"));
        let Record::Trajectory(record) = &mut chunk.record else {
            unreachable!()
        };
        record.id = format!("episode-{i}");
        record.evaluation.continuation_policy = "fixed protocol".into();
        recorder.append(&chunk).unwrap();
    }
    recorder.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut loader = Loader::new(4 << 20, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "target", CatalogLimits::default()).unwrap();
    while catalog.pending() {
        loader.poll();
        catalog.poll(&run, &mut loader).unwrap();
        if catalog.pending() {
            rx.recv_timeout(Duration::from_secs(10)).unwrap();
        }
    }
    let sources = SourceSet::new(&run, catalog.entries(), catalog.generation(), limits).unwrap();
    (dir, run, sources)
}

#[test]
fn reference_uses_all_admitted_episodes_and_exact_internal_terminal_extrema() {
    let observations = vec![
        vec![
            -7., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 0., 2., 14.,
        ],
        vec![
            0., 0., 10., 1., -90., 11., 1., 1., 12., 2., 2., 13., 0., 2., 14.,
        ],
        vec![
            0., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 100., 80., 1000.,
        ],
    ];
    let (_dir, _, sources) = multiple_sources_with_episode_observations(
        &[17, 37, 57],
        true,
        &WorkLimits::default(),
        Some(observations),
        Some(vec![
            vec![0., 10., 20., 1., 11., 21., 2., 12., 22., 3., 13., 23.],
            vec![0., 10., 20., 1., 11., -90., 2., 12., 22., 3., 13., 23.],
            vec![0., 10., 20., 1., 11., 21., 2., 12., 99., 3., 13., 23.],
        ]),
    );
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(
        frame.state.reference.grid.grid.bounds(),
        [[-7., 100.], [-90., 80.]]
    );
    assert_eq!(frame.state.reference.source_keys.len(), 3);
    assert_eq!(frame.state.history_decisions, 12);
    assert_eq!(frame.state.reference.state_extrema(2), Some([10., 1000.]));
    assert_eq!(frame.state.reference.state_bounds(2), Some([10., 1000.]));
    assert_eq!(frame.state.reference.action_bounds(2), Some([-90., 99.]));
    assert_eq!(frame.state.reference.episodes(), 3);
    assert_eq!(frame.state.reference.decisions(), 12);
    assert_eq!(
        worker.counters().source_reads,
        3,
        "a fitting population must decode once across both passes"
    );
    assert_eq!(worker.counters().decoded_hits, 2);
    assert!(worker.counters().decoded_cache_bytes > 0);
    assert!(
        worker.counters().decoded_cache_bytes <= WorkLimits::default().max_episode_bytes as u64
    );
}

#[test]
fn explicit_domain_identity_defeats_cached_reference_fallback() {
    let (_dir, run, all) = multiple_sources_with_episode_observations(
        &[17, 37, 57],
        true,
        &WorkLimits::default(),
        Some(vec![
            vec![
                0., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 0., 2., 14.,
            ],
            vec![
                0., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 0., 2., 14.,
            ],
            vec![
                0., 0., 10., 1., 0., 11., 1., 1., 12., 2., 2., 13., 100., 200., 14.,
            ],
        ]),
        None,
    );
    let initial = SourceSet::new(&run, &all.sources()[..2], 1, &WorkLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(initial)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let original = worker.frame().unwrap().state.reference.clone();
    assert_eq!(original.grid.grid.bounds(), [[0., 2.], [0., 2.]]);
    let mut req = request(all);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(Arc::ptr_eq(
        &original,
        &worker.frame().unwrap().state.reference
    ));
    req.domain = DomainSpec::automatic(1);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let recomputed = worker.frame().unwrap().state.reference.clone();
    assert!(!Arc::ptr_eq(&original, &recomputed));
    assert_eq!(recomputed.source_keys.len(), 3);
    assert_eq!(recomputed.grid.grid.bounds(), [[0., 100.], [0., 200.]]);
    assert_eq!(recomputed.domain(), req.domain);
    req.domain = DomainSpec::manual([[-10., 10.], [-20., 20.]]).unwrap();
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(
        frame.state.reference.grid.grid.bounds(),
        [[-10., 10.], [-20., 20.]]
    );
    assert_eq!(frame.state.reference.grid.empirical, [false, false]);
    assert_eq!(frame.state.reference.state_extrema(2), Some([10., 14.]));
    req.reference = Some(original);
    assert!(
        worker.request(req).is_err(),
        "an old explicit reference cannot cross domains"
    );
}

#[test]
fn superseded_extrema_pass_never_publishes_a_partial_or_obsolete_domain() {
    let (_dir, sources) = multiple_sources(&[17, 37, 57], true, &WorkLimits::default());
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = std::sync::Mutex::new(release_rx);
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::with_reader(
        WorkLimits::default(),
        move || {
            let _ = tx.send(());
        },
        move |source, limit| {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 2 {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            source.load_if_within_charge(limit)
        },
    )
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(worker.frame().is_none());
    assert_eq!(worker.counters().state_builds, 0);
    let mut latest = request(sources);
    latest.domain = DomainSpec::manual([[-1., 1.], [-1., 1.]]).unwrap();
    worker.request(latest.clone()).unwrap();
    release_tx.send(()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.state.reference.domain(), latest.domain);
    assert_eq!(
        frame.state.reference.grid.grid.bounds(),
        [[-1., 1.], [-1., 1.]]
    );
    assert_eq!(frame.state.reference.episodes(), 3);
    assert_eq!(frame.state.reference.decisions(), 12);
    assert_eq!(worker.counters().cancelled, 1);
    assert_eq!(worker.counters().state_builds, 1);
    assert_eq!(
        worker.counters().source_reads,
        4,
        "completed immutable decodes survive cancellation; interrupted source is read again"
    );
}

#[test]
fn small_decoded_cache_uses_two_bounded_read_phases_without_losing_evidence() {
    let (_dir, run, sources) =
        multiple_sources_with_run(&[17, 37, 57], true, &WorkLimits::default());
    let chunk = run.recording().load(0).unwrap();
    let episode = rl_viewer::dynamics::prepare_episode(
        &chunk,
        run.recording().manifest(),
        &Default::default(),
    )
    .unwrap();
    let total_charge = (0..3)
        .map(|i| run.recording().load(i).unwrap().retained_charge())
        .sum::<usize>();
    let limits = WorkLimits {
        max_episode_bytes: episode.retained_bytes(),
        max_total_read_bytes: total_charge,
        ..Default::default()
    };
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(limits.clone(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(worker.error().is_none(), "{:?}", worker.error());
    assert_eq!(worker.frame().unwrap().state.reference.source_keys.len(), 3);
    assert_eq!(worker.frame().unwrap().state.history_decisions, 12);
    assert_eq!(worker.counters().source_reads, 5);
    assert_eq!(
        worker.counters().decoded_cache_bytes,
        0,
        "metadata must be charged as well as episode arrays"
    );
    assert!(worker.counters().source_read_bytes <= (2 * total_charge) as u64);
    let (tx, rx) = mpsc::channel();
    let mut denied = DynamicsWorker::new(
        WorkLimits {
            max_total_read_bytes: total_charge - 1,
            ..limits
        },
        move || {
            let _ = tx.send(());
        },
    )
    .unwrap();
    denied.request(request(sources)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    denied.poll();
    assert!(denied.frame().is_none());
    assert!(denied.error().unwrap().contains("budget"));
}

#[test]
fn adopted_reference_action_change_reuses_state_and_revision_changes_keep_grid() {
    let (_dir, sources) = multiple_sources(&[17, 37, 57], true, &WorkLimits::default());
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources.clone());
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let state = worker.frame().unwrap().state.clone();
    req.reference = Some(state.reference.clone());
    req.action = ActionAxis::Continuous(2);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(Arc::ptr_eq(&state, &worker.frame().unwrap().state));
    let reads_before_revision = worker.counters().source_reads;
    req.revision = 57;
    req.selected = sources.sources()[2].key.clone();
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert!(Arc::ptr_eq(&state.reference, &frame.state.reference));
    assert_eq!(
        frame
            .state
            .coverage
            .points
            .iter()
            .map(|p| p.revision)
            .collect::<Vec<_>>(),
        [17, 37, 57]
    );
    assert_eq!(
        frame
            .state
            .coverage
            .points
            .iter()
            .map(|p| p.cumulative)
            .collect::<Vec<_>>(),
        [3, 3, 3]
    );
    assert_eq!(frame.state.population_episodes, 1);
    assert_eq!(frame.state.history_episodes, 3);
    assert_eq!(
        worker.counters().source_reads - reads_before_revision,
        0,
        "a fitting decoded population must survive revision switching"
    );
    assert!(
        std::ptr::eq(&*state.coverage.points, &*frame.state.coverage.points),
        "revision switching must share immutable history storage"
    );
}

#[test]
fn oversized_populations_are_deterministically_sampled_and_disclosed() {
    let limits = WorkLimits {
        max_sources: 2,
        max_decisions: 8,
        ..Default::default()
    };
    let (_dir, sources) = multiple_sources(&[17, 37, 57, 77, 97], false, &limits);
    assert_eq!(
        sources
            .sources()
            .iter()
            .map(|s| s.revision)
            .collect::<Vec<_>>(),
        [77, 97]
    );
    assert_eq!(sources.omitted_sources(), 3);
    assert_eq!(sources.omitted_decisions(), 12);
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources);
    req.revision = 77;
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert_eq!(worker.frame().unwrap().state.history_decisions, 8);
}

#[test]
fn last_request_wins_for_revision_axes_and_action_without_reentry() {
    use std::sync::Mutex;
    let (_dir, sources) = multiple_sources(&[17, 37, 57], false, &WorkLimits::default());
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release_rx = Mutex::new(release_rx);
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let reader_calls = calls.clone();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::with_reader(
        WorkLimits::default(),
        move || {
            let _ = tx.send(());
        },
        move |source, limit| {
            if reader_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            source.load_if_within_charge(limit)
        },
    )
    .unwrap();
    worker.request(request(sources.clone())).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    for revision in [37, 17, 57, 37] {
        let mut req = request(sources.clone());
        req.revision = revision;
        req.selected = sources
            .sources()
            .iter()
            .find(|s| s.revision == revision)
            .unwrap()
            .key
            .clone();
        req.axes = Axes { x: 1, y: 2 };
        req.action = ActionAxis::Continuous(2);
        req.projection = Projection::State([2, 1, 0]);
        worker.request(req).unwrap();
    }
    release_tx.send(()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.state.revision, 37);
    assert_eq!(frame.action, ActionAxis::Continuous(2));
    assert_eq!(frame.selected.path.start, [0., 10.]);
    assert_eq!(frame.geometry3[0].points[0], [10., 0., 0.]);
    assert_eq!(worker.counters().state_builds, 1);
    assert_eq!(worker.counters().cancelled, 1);
    assert_eq!(worker.counters().submitted, 5);
}

#[test]
fn small_worker_caps_reject_long_or_wide_exact_arrays_without_partial_cursor_data() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let limits = WorkLimits {
        max_episode_bytes: 1,
        ..Default::default()
    };
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(worker.frame().is_none());
    assert!(worker.error().unwrap().contains("budget"));
}

#[test]
fn live_source_generation_extends_coverage_without_moving_reference_bounds() {
    let (_dir, run, all) = multiple_sources_with_run(&[17, 37, 57], true, &WorkLimits::default());
    let initial = SourceSet::new(&run, &all.sources()[..2], 1, &WorkLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(initial)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let reference = worker.frame().unwrap().state.reference.clone();
    assert_eq!(worker.frame().unwrap().state.coverage.points.len(), 2);
    let mut req = request(all);
    req.reference = Some(reference.clone());
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.state.coverage.points.len(), 3);
    assert!(Arc::ptr_eq(&reference, &frame.state.reference));
    assert_eq!(reference.source_keys.len(), 2);
    assert_eq!(worker.counters().state_builds, 2);
}

#[test]
fn long_wide_recorded_episode_keeps_exact_arbitrary_cursor_with_bounded_geometry() {
    use rl::diagnostics::{
        schema::{Array, Record},
        storage::{Recorder, StorageConfig},
    };
    let dir = tempfile::tempdir().unwrap();
    let space = fixture::bounded(48);
    let manifest = fixture::manifest(space.clone());
    let transitions = 20_000;
    let mut chunk = fixture::chunk(&space);
    chunk.arrays.insert(
        "obs".into(),
        Array::f64(
            vec![transitions + 1, 3],
            (0..=transitions)
                .flat_map(|t| [t as f64 / 10_000., 0.5, 1.])
                .collect(),
        )
        .unwrap(),
    );
    chunk.arrays.insert(
        "actions".into(),
        Array::f64(
            vec![transitions, 48],
            (0..transitions)
                .flat_map(|t| (0..48).map(move |j| j as f64 + (t % 3) as f64))
                .collect(),
        )
        .unwrap(),
    );
    chunk.arrays.insert(
        "rewards".into(),
        Array::f64(vec![transitions], vec![1.; transitions]).unwrap(),
    );
    let mut terminated = vec![0; transitions];
    terminated[transitions - 1] = 1;
    chunk.arrays.insert(
        "terminated".into(),
        Array::u8(vec![transitions], terminated).unwrap(),
    );
    chunk.arrays.insert(
        "truncated".into(),
        Array::u8(vec![transitions], vec![0; transitions]).unwrap(),
    );
    chunk.arrays.remove("times");
    let Record::Trajectory(record) = &mut chunk.record else {
        unreachable!()
    };
    record.simulation_times = None;
    let mut recorder =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    recorder.append(&chunk).unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut loader = Loader::new(32 << 20, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut catalog = TrajectoryCatalog::new(&run, "target", CatalogLimits::default()).unwrap();
    while catalog.pending() {
        loader.poll();
        catalog.poll(&run, &mut loader).unwrap();
        if catalog.pending() {
            rx.recv_timeout(Duration::from_secs(10)).unwrap();
        }
    }
    let limits = WorkLimits {
        max_segments: 32,
        max_trace_points: 16,
        ..Default::default()
    };
    let sources = SourceSet::new(&run, catalog.entries(), 1, &limits).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources.clone());
    req.action = ActionAxis::Continuous(47);
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let f = worker.frame().unwrap();
    assert_eq!(f.selected.episode.action_axes().len(), 48);
    assert_eq!(f.selected.path.segments.len(), 32);
    assert_eq!(f.geometry3.len(), 32);
    assert_eq!(f.action_trace.len(), 16);
    assert_eq!(f.selected.reward_trace.len(), 16);
    assert_eq!(
        f.selected
            .episode
            .action_value(19_999, ActionAxis::Continuous(47))
            .unwrap(),
        Some(rl_viewer::dynamics::ActionValue::Continuous(48.))
    );
    assert_eq!(
        f.selected.episode.step(20_000).unwrap().observation,
        [2., 0.5, 1.]
    );
    assert_eq!(f.selected.episode.step(20_000).unwrap().reward, None);
    assert_eq!(f.state.population_decisions, 20_000);
    assert!(f.retained_bytes < WorkLimits::default().max_frame_bytes);
    let frame = worker.frame_arc().unwrap();
    let view = HitView {
        axes: Axes { x: 0, y: 1 },
        bounds: [[0., 2.], [0., 2.]],
        chart: eframe::egui::Rect::from_min_max([0., 0.].into(), [20_000., 100.].into()),
    };
    // Index 12_345 is absent from the 32-edge display sample. Both it and the
    // final observation must still be selectable by the full exact worker scan.
    for step in [12_345, 20_000] {
        worker
            .request_hit(&frame, sources.clone(), view, [step as f32, 75.].into())
            .unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(worker.poll_hit(&frame, view), Some(step));
    }
}

#[test]
fn reconstructing_equivalent_source_handles_does_not_invalidate_exact_state_identity() {
    let (_dir, run, sources) = multiple_sources_with_run(&[17, 37], false, &WorkLimits::default());
    let same = SourceSet::new(
        &run,
        sources.sources(),
        sources.generation(),
        &WorkLimits::default(),
    )
    .unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(sources)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(!worker.request(request(same)).unwrap());
}

#[test]
fn unavailable_empirical_reference_outside_sliding_window_cannot_publish_cached_state() {
    let (dir, run, all) = multiple_sources_with_run(&[17, 37, 57], true, &WorkLimits::default());
    let initial = SourceSet::new(&run, &all.sources()[..2], 1, &WorkLimits::default()).unwrap();
    let later = SourceSet::new(&run, &all.sources()[2..], 2, &WorkLimits::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    worker.request(request(initial)).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let reference = worker.frame().unwrap().state.reference.clone();
    assert_eq!(reference.grid.empirical, [true, true]);
    reference.check_available().unwrap();
    assert!(reference.retained_bytes() > 0);
    assert!(reference.retained_bytes() <= WorkLimits::default().max_source_set_bytes);
    assert!(worker.frame().unwrap().state.retained_bytes >= reference.retained_bytes());
    let bounds = reference.grid.grid.bounds();
    let original = reference.source_keys[1].clone();
    assert!(!later.sources().iter().any(|source| source.key == original));
    let mut req = request(later.clone());
    req.revision = 57;
    req.reference = Some(reference.clone());
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(Arc::ptr_eq(
        &reference,
        &worker.frame().unwrap().state.reference
    ));
    let builds = worker.counters().state_builds;
    // Delete the second pinned source, outside the current admitted set.
    // Checking only the first reference source cannot detect this deletion.
    let index = run.by_sequence(original.sequence).unwrap();
    std::fs::remove_file(
        dir.path()
            .join("run")
            .join(&run.recording().entries()[index].file),
    )
    .unwrap();
    later.check_available().unwrap();
    assert!(
        reference
            .check_available()
            .unwrap_err()
            .to_string()
            .contains("reference")
    );
    req.action = ActionAxis::Continuous(2);
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(
        worker.frame().is_none(),
        "missing pinned reference must prevent publication even when current source state is cached"
    );
    assert!(worker.error().unwrap().contains("reference"));
    assert_eq!(worker.counters().state_builds, builds);
    assert_eq!(
        reference.grid.grid.bounds(),
        bounds,
        "missing reference must never move the grid"
    );
}
#[test]
fn background_projection_reuses_selection_and_state_but_tracks_each_action_component() {
    let limits = WorkLimits {
        max_segments: 4,
        ..Default::default()
    };
    let (_dir, sources) = multiple_sources(&[17, 17, 63], true, &limits);
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(limits, move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources.clone());
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let state = worker.frame().unwrap().state.clone();
    let background = worker.frame().unwrap().background.clone();
    assert_eq!(background.paths.len(), 2);
    assert_eq!(background.eligible_segments, 6);
    assert_eq!(background.displayed_segments, 4);
    for path in &background.paths {
        assert_eq!(
            path.segments.iter().map(|s| s.t).collect::<Vec<_>>(),
            [0, 2]
        );
        for edge in &path.segments {
            assert_eq!(edge.points[0][2], edge.t as f64);
            assert_eq!(edge.points[1][2], (edge.t + 1) as f64);
        }
    }
    let before = worker.counters();
    req.reference = Some(state.reference.clone());
    req.selected = sources.sources()[1].key.clone();
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(Arc::ptr_eq(
        &background,
        &worker.frame().unwrap().background
    ));
    assert_eq!(
        before.projection_builds,
        worker.counters().projection_builds
    );
    for component in [1, 2] {
        req.action = ActionAxis::Continuous(component);
        worker.request(req.clone()).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
        let frame = worker.frame().unwrap();
        assert!(Arc::ptr_eq(&state, &frame.state));
        assert!(Arc::ptr_eq(&state.coverage, &frame.state.coverage));
        assert_eq!(
            frame.background.paths[0].segments[0].points[0][2],
            (10 * component) as f64
        );
    }
    assert_eq!(before.source_reads, worker.counters().source_reads);
    drop(background);
    req.projection = Projection::State([0, 1, 2]);
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let background = worker.frame().unwrap().background.clone();
    assert_eq!(background.eligible_segments, 8);
    assert_eq!(background.bounds[2], [10., 14.]);
    req.action = ActionAxis::Continuous(0);
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    assert!(Arc::ptr_eq(
        &background,
        &worker.frame().unwrap().background
    ));
}

#[test]
fn independent_projection_leases_are_charged() {
    let (_dir, sources) = sources();
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources);
    let mut held = Vec::new();
    for component in 0..3 {
        req.action = ActionAxis::Continuous(component);
        worker.request(req.clone()).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.poll();
        held.push(worker.frame().unwrap().background.clone());
    }
    req.action = ActionAxis::Continuous(0);
    assert!(
        worker
            .request(req.clone())
            .unwrap_err()
            .to_string()
            .contains("leases")
    );
    held.clear();
    assert!(worker.request(req).unwrap());
}

#[test]
fn revision_evidence_aggregates_unequal_episodes_and_excluded_visits_without_terminal_actions() {
    let observations = vec![
        vec![0., 0., 7., 3., 0., 7., 2., 2., 20.],
        vec![1.5, 1.5, 7., -5., -5., 20.],
        vec![0., 0., 7., 1.5, 0., 7., 5., 5., 7., 2., 2., 20.],
        vec![0., 0., 7., 2., 2., 20.],
    ];
    let (_dir, _, sources) = multiple_sources_with_episode_observations(
        &[17, 17, 63, 97],
        true,
        &WorkLimits::default(),
        Some(observations),
        None,
    );
    let (tx, rx) = mpsc::channel();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
        let _ = tx.send(());
    })
    .unwrap();
    let mut req = request(sources);
    req.domain = DomainSpec::manual([[0., 2.], [0., 2.]]).unwrap();
    worker.request(req.clone()).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(
        frame
            .state
            .coverage
            .points
            .iter()
            .map(|p| (
                p.revision,
                p.episodes,
                p.decisions,
                p.included,
                p.excluded,
                p.occupied,
                p.new_cells,
                p.cumulative
            ))
            .collect::<Vec<_>>(),
        [
            (17, 2, 3, 2, 1, 2, 2, 2),
            (63, 1, 3, 2, 1, 2, 1, 3),
            (97, 1, 1, 1, 0, 1, 0, 3)
        ]
    );
    assert_eq!(frame.background.eligible_segments, 1);
    assert_eq!(frame.background.paths.len(), 2);
    assert_eq!(frame.background.paths[1].start, [1.5, 1.5, 0.]);
    assert_eq!(
        frame.background.paths[1].end,
        frame.background.paths[1].start
    );
    assert!(frame.background.paths[1].segments.is_empty());
    assert_eq!(frame.background.bounds, [[0., 2.], [0., 2.], [0., 2.]]);
    let reference = frame.state.reference.clone();
    req.reference = Some(reference);
    req.projection = Projection::State([0, 1, 2]);
    worker.request(req).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker.poll();
    let frame = worker.frame().unwrap();
    assert_eq!(frame.background.bounds, [[0., 2.], [0., 2.], [7., 20.]]);
    assert_eq!(frame.background.paths[1].segments.len(), 1);
    assert_eq!(frame.background.paths[1].end, [-5., -5., 20.]);
}

#[test]
fn projection_path_metadata_is_preflighted_before_allocation() {
    let limits = WorkLimits {
        max_sources: 10,
        max_displayed_episodes: 10,
        max_segments: 2,
        max_trace_points: 2,
        max_grid_cells: 1,
        max_frame_bytes: 1024,
        ..Default::default()
    };
    assert!(
        limits
            .envelope_bytes()
            .unwrap_err()
            .to_string()
            .contains("geometry")
    );
    assert!(WorkLimits::default().envelope_bytes().unwrap() <= 512 * 1024 * 1024);
}
