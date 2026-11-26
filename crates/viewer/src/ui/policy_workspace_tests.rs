use super::*;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        probes::ProbeSet,
        storage::{Recorder, StorageConfig},
    },
};
use std::collections::BTreeMap;
fn fixture() -> (tempfile::TempDir, egui::Context, Laboratory) {
    fixture_visits(false)
}
fn fixture_visits(include_visits: bool) -> (tempfile::TempDir, egui::Context, Laboratory) {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let mut manifest =
        RunManifest::new("policy-lifecycle", "ordinary", "PPO", 2, space.clone(), 0.9).unwrap();
    manifest.environment = "lqg".into();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("policy-lifecycle", 1);
    context.probe_set = Some("fixed-grid".into());
    writer
        .append(
            &ProbeSet::new(
                "fixed-grid",
                ProbePopulation::FixedReference,
                "actual complete inputs",
                vec![
                    Observation::new(vec![0., 0.]),
                    Observation::new(vec![0.2, 0.3]),
                ],
                None,
                space,
            )
            .unwrap()
            .record(context.clone())
            .unwrap(),
        )
        .unwrap();
    for revision in [1, 5] {
        context.identity.learner_update_revision = revision;
        context.counters.updates = revision;
        writer
            .append(&Chunk {
                context: context.clone(),
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Categorical {
                        probabilities: "p".into(),
                    },
                    stochastic: false,
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays: BTreeMap::from([
                    ("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
                    (
                        "p".into(),
                        Array::f64(vec![2, 2], vec![0.7, 0.3, 0.4, 0.6]).unwrap(),
                    ),
                ]),
            })
            .unwrap();
    }
    if include_visits {
        let mut visited = context.clone();
        visited.probe_set = Some("actual-visits".into());
        writer
            .append(
                &ProbeSet::new(
                    "actual-visits",
                    ProbePopulation::PolicyVisited,
                    "actual fixture decision observations",
                    vec![
                        Observation::new(vec![0., 0.]),
                        Observation::new(vec![0.2, 0.3]),
                        Observation::new(vec![999., 999.]),
                    ],
                    None,
                    ActionSpace::Discrete { count: 2 },
                )
                .unwrap()
                .record(visited)
                .unwrap(),
            )
            .unwrap();
    }
    writer.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app =
        Laboratory::from_context(&ctx, dir.path().join("run").display().to_string()).unwrap();
    app.select_view(View::PolicyMap);
    (dir, ctx, app)
}
fn render(app: &mut Laboratory, ctx: &egui::Context, events: Vec<egui::Event>) -> egui::FullOutput {
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 900.),
            )),
            events,
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    out.textures_delta.clear();
    assert!(app.notice().is_none(), "{:?}", app.notice());
    out
}
fn settle(app: &mut Laboratory, ctx: &egui::Context) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while !app.policy_pane.ready() {
        let mut out = render(app, ctx, vec![]);
        out.textures_delta.clear();
        assert!(Instant::now() < deadline, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// Manual release measurement using a complete recording. This remains ignored
/// in CI because the public demonstration data is intentionally not in Git.
#[test]
#[ignore = "requires a complete local public recording"]
fn measure_real_reference_switches() {
    let path = std::env::var("RL_REFERENCE_BENCH_RUN").expect("RL_REFERENCE_BENCH_RUN");
    let revision: u64 = std::env::var("RL_REFERENCE_BENCH_REVISION")
        .expect("RL_REFERENCE_BENCH_REVISION")
        .parse()
        .unwrap();
    let earlier: u64 = std::env::var("RL_REFERENCE_BENCH_EARLIER")
        .expect("RL_REFERENCE_BENCH_EARLIER")
        .parse()
        .unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path).unwrap();
    app.select_view(View::PolicyMap);
    app.select_revision(revision).unwrap();
    app.workspace_defaults = None;
    app.comparison.set_reference_role("online").unwrap();
    app.policy_pane.reference_enabled = true;
    app.policy_pane.reference_initialized = true;
    app.policy_pane.reference_automatic = false;
    let wait_for = |app: &mut Laboratory, expected: u64, start: Instant| {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut publication = None;
        loop {
            let mut out = render(app, &ctx, vec![]);
            let shapes = std::mem::take(&mut out.shapes);
            std::hint::black_box(ctx.tessellate(shapes, out.pixels_per_point));
            if app.policy_frame().is_some_and(|frame| {
                frame.requested_revision == revision
                    && frame.grid.current.revision == revision
                    && frame
                        .grid
                        .reference
                        .as_ref()
                        .is_some_and(|r| r.revision == expected)
            }) && !app.policy_pending()
            {
                publication.get_or_insert_with(|| start.elapsed());
                if app.snapshot_preparation_idle() {
                    break publication.unwrap();
                }
            }
            assert!(
                Instant::now() < deadline,
                "reference {expected} did not settle: {:?}",
                app.policy_pane.status()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    };
    app.comparison.set_reference_revision(earlier).unwrap();
    wait_for(&mut app, earlier, Instant::now());
    let mut csv = String::from(
        "round,reference_revision,publication_ms,settled_ms,restores,grid_batches,reused_grids,restore_ms,inference_ms,preparation_ms,prepared_peak_bytes\n",
    );
    for round in 0..20 {
        let target = if round % 2 == 0 { revision } else { earlier };
        let before = app.policy_metrics();
        let start = Instant::now();
        app.comparison.set_reference_revision(target).unwrap();
        let published = wait_for(&mut app, target, start).as_secs_f64() * 1000.;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        let after = app.policy_metrics();
        if target == revision {
            let grid = &app.policy_frame().unwrap().grid;
            for (row, current) in grid.values.iter().enumerate() {
                assert_eq!(current.to_bits(), grid.reference_at(row).unwrap().to_bits());
            }
        }
        csv.push_str(&format!(
            "{round},{target},{published:.3},{elapsed:.3},{},{},{},{:.3},{:.3},{:.3},{}\n",
            after.restores - before.restores,
            after.grid_batches - before.grid_batches,
            after.reused_grids - before.reused_grids,
            (after.restore_micros - before.restore_micros) as f64 / 1000.,
            (after.inference_micros - before.inference_micros) as f64 / 1000.,
            (after.preparation_micros - before.preparation_micros) as f64 / 1000.,
            after.prepared_peak
        ));
        eprintln!(
            "Reference switch {}/20: {published:.1} ms published, {elapsed:.1} ms settled",
            round + 1
        );
    }
    if let Some(path) = std::env::var_os("RL_REFERENCE_BENCH_OUTPUT") {
        std::fs::write(path, &csv).unwrap();
    }
    print!("{csv}");
}

#[test]
fn automatic_provider_policy_overlay_uses_the_shared_probe_population() {
    let (directory, _, original) = fixture();
    let original = original.run.as_ref().unwrap().recording();
    let root = directory.path().join("provider-run");
    let mut writer =
        Recorder::create(&root, original.manifest().clone(), StorageConfig::default()).unwrap();
    for i in 0..original.entries().len() {
        writer.append(&original.load(i).unwrap()).unwrap();
    }
    let mut context = RecordContext::new("policy-lifecycle", 1);
    context.identity.model_role = "reference:benchmark".into();
    for (set, population, observations) in [
        (
            "fixed-grid",
            ProbePopulation::FixedReference,
            [[0., 0.], [0.2, 0.3]],
        ),
        (
            "actual-visits",
            ProbePopulation::PolicyVisited,
            [[1., 1.], [2., 2.]],
        ),
    ] {
        context.probe_set = Some(set.into());
        writer
            .append(
                &ProbeSet::new(
                    set,
                    population,
                    "provider population",
                    observations
                        .map(|values| Observation::new(values.to_vec()))
                        .to_vec(),
                    None,
                    ActionSpace::Discrete { count: 2 },
                )
                .unwrap()
                .record(context.clone())
                .unwrap(),
            )
            .unwrap();
        writer
            .append(&Chunk {
                context: context.clone(),
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Deterministic,
                    stochastic: false,
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays: BTreeMap::from([("a".into(), Array::u64(vec![2], vec![1, 0]).unwrap())]),
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, root.display().to_string()).unwrap();
    assert!(matches!(
        &app.selection.input,
        SelectedInput::ProbeRow { set, sequence: 0, .. } if set == "fixed-grid"
    ));
    app.comparison
        .set_reference_role("reference:benchmark")
        .unwrap();
    app.comparison.set_reference_revision(1).unwrap();
    app.select_view(View::PolicyMap);
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.policy_pane.config.is_none() {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(app.policy_pane.reference_enabled);
    #[cfg(feature = "screenshots")]
    {
        app.policy_pane.capture_configured = true;
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let point = loop {
        render(&mut app, &ctx, vec![]);
        if let Some(point) = app.policy_frame().and_then(|frame| frame.selected.clone()) {
            break point;
        }
        assert!(Instant::now() < deadline, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    };
    assert_eq!(point.reference_chosen_action, Some(1));
    assert!(matches!(
        &point.reference_recorded_input,
        Some(SelectedInput::ProbeRow { set, row: 0, .. }) if set == "fixed-grid"
    ));
}

#[test]
fn oversized_automatic_reference_keeps_learner_and_explicit_choice_reports_error() {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let mut manifest = RunManifest::new(
        "bounded-reference",
        "ordinary",
        "PPO",
        2,
        space.clone(),
        0.9,
    )
    .unwrap();
    manifest.environment = "lqg".into();
    let root = dir.path().join("run");
    let mut writer = Recorder::create(&root, manifest, StorageConfig::default()).unwrap();
    for (role, rows) in [("online", 512), ("reference:benchmark", 2049)] {
        let mut context = RecordContext::new("bounded-reference", 1);
        context.identity.model_role = role.into();
        context.probe_set = Some("fixed-grid".into());
        let observations = (0..rows)
            .map(|i| Observation::new(vec![i as f32 / 2049., 0.]))
            .collect();
        writer
            .append(
                &ProbeSet::new(
                    "fixed-grid",
                    ProbePopulation::FixedReference,
                    "bounded matching fixture",
                    observations,
                    None,
                    space.clone(),
                )
                .unwrap()
                .record(context.clone())
                .unwrap(),
            )
            .unwrap();
        writer
            .append(&Chunk {
                context,
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Deterministic,
                    stochastic: false,
                    masks: None,
                    entropy: None,
                    entropy_method: None,
                }),
                arrays: BTreeMap::from([(
                    "a".into(),
                    Array::u64(vec![rows], vec![0; rows]).unwrap(),
                )]),
            })
            .unwrap();
    }
    writer.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, root.display().to_string()).unwrap();
    app.comparison
        .set_reference_role("reference:benchmark")
        .unwrap();
    app.comparison.set_reference_revision(1).unwrap();
    app.select_view(View::PolicyMap);
    settle(&mut app, &ctx);
    assert!(!app.policy_pane.reference_enabled);
    assert!(app.policy_frame().unwrap().grid.reference.is_none());

    click(&mut app, &ctx, "Reference");
    let deadline = Instant::now() + Duration::from_secs(15);
    while !app.policy_pane.terminal {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < deadline, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(app.policy_pane.reference_enabled);
    assert!(
        app.policy_pane
            .status()
            .unwrap()
            .contains("row-check budget")
    );
}

fn text_position(shape: &egui::Shape, text: &str) -> Option<egui::Pos2> {
    match shape {
        egui::Shape::Text(t) if t.galley.text() == text => {
            Some(t.pos + t.galley.rect.center().to_vec2())
        }
        egui::Shape::Vec(v) => v.iter().find_map(|s| text_position(s, text)),
        _ => None,
    }
}
pub(super) fn click(app: &mut Laboratory, ctx: &egui::Context, text: &str) {
    let mut out = render(app, ctx, vec![]);
    let position = out
        .shapes
        .iter()
        .find_map(|s| text_position(&s.shape, text))
        .unwrap();
    out.textures_delta.clear();
    for pressed in [true, false] {
        let mut out = render(
            app,
            ctx,
            vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        out.textures_delta.clear();
    }
}
#[test]
fn policy_all_departure_paths_release_pending_completed_and_committed_owners() {
    for departure in 0..6 {
        let (_dir, ctx, mut app) = fixture();
        settle(&mut app, &ctx);
        assert!(app.interactive.policy_metrics().prepared_bytes > 0);
        let before = app.interactive.retirement_status().0;
        match departure {
            0 => app.select_view(View::Network),
            1 => app.open_run(0),
            2 => app.clear_recording(),
            3 => app.select_view(View::Value),
            4 => click(&mut app, &ctx, "Network"),
            5 => {
                app.error = Some("test read failure".into());
                let mut warm = ctx.run_ui(Default::default(), |ui| app.show(ui));
                warm.textures_delta.clear();
                let mut out = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1440., 900.),
                        )),
                        ..Default::default()
                    },
                    |ui| app.show(ui),
                );
                let position = out
                    .shapes
                    .iter()
                    .find_map(|s| text_position(&s.shape, "Retry recording reads"));
                out.textures_delta.clear();
                let position = position.expect("Retry recording reads button");
                for pressed in [true, false] {
                    let mut out = ctx.run_ui(
                        egui::RawInput {
                            screen_rect: Some(egui::Rect::from_min_size(
                                egui::Pos2::ZERO,
                                egui::vec2(1440., 900.),
                            )),
                            events: vec![
                                egui::Event::PointerMoved(position),
                                egui::Event::PointerButton {
                                    pos: position,
                                    button: egui::PointerButton::Primary,
                                    pressed,
                                    modifiers: egui::Modifiers::NONE,
                                },
                            ],
                            ..Default::default()
                        },
                        |ui| app.show(ui),
                    );
                    out.textures_delta.clear();
                }
            }
            _ => unreachable!(),
        }
        assert!(
            app.interactive.retirement_status().0 > before,
            "departure {departure}"
        );
        assert_eq!(app.policy_pane.texture_bytes(), 0);
        let deadline = Instant::now() + Duration::from_secs(15);
        while app.interactive.policy_metrics().prepared_bytes != 0 {
            app.interactive.housekeeping();
            assert!(Instant::now() < deadline, "departure {departure}");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(app.interactive.policy_frame().is_none());
    }
}
#[test]
#[cfg(feature = "screenshots")]
fn policy_capture_waits_for_real_grid_exact_pin_reference_and_latest_revision() {
    let (dir, ctx, mut app) = fixture();
    app.capture = Some(dir.path().join("not-dispatched-to-native.png"));
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut saw_pending = false;
    loop {
        let mut out = render(&mut app, &ctx, vec![]);
        let capture = out.viewport_output.values().any(|v| {
            v.commands
                .iter()
                .any(|c| matches!(c, egui::ViewportCommand::Screenshot(_)))
        });
        out.textures_delta.clear();
        if capture {
            let f = app.interactive.policy_frame().unwrap();
            assert!(f.selected.is_some());
            assert_eq!(f.requested_revision, app.selection.revision);
            assert!(!app.interactive.policy_pending());
            break;
        }
        saw_pending = true;
        assert!(Instant::now() < deadline, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(saw_pending);
    assert!(
        app.policy_capture_problem(Some("grid"), None)
            .unwrap()
            .contains("grid")
    );
    // A stale ready flag cannot dispatch while an exact replacement point is pending.
    let mut request = app.policy_pane.request.clone().unwrap();
    request.pin = crate::policy::prepared::PolicyPin::Recorded {
        key: request.current.identity().key.clone(),
        probe_key: request.current.identity().probe_key.clone().unwrap(),
        row: 1,
    };
    let expected = app
        .interactive
        .policy_frame()
        .unwrap()
        .grid
        .input(1)
        .unwrap();
    app.interactive
        .policy_frame()
        .unwrap()
        .grid
        .selection_intent(1)
        .unwrap()
        .apply(app.run.as_ref().unwrap(), &mut app.selection)
        .unwrap();
    app.policy_pane.pin_input = Some(expected.clone());
    app.policy_pane.pin = Some(request.pin.clone());
    app.policy_pane.request = Some(request.clone());
    app.interactive.submit_policy(request).unwrap();
    assert!(!app.policy_capture_ready());
    app.capture = Some(dir.path().join("exact-pin.png"));
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let mut out = render(&mut app, &ctx, vec![]);
        let capture = out.viewport_output.values().any(|v| {
            v.commands
                .iter()
                .any(|c| matches!(c, egui::ViewportCommand::Screenshot(_)))
        });
        out.textures_delta.clear();
        if capture {
            assert_eq!(
                app.interactive
                    .policy_frame()
                    .unwrap()
                    .selected
                    .as_ref()
                    .unwrap()
                    .input,
                expected
            );
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    app.policy_pane.reference_enabled = true;
    assert!(
        !app.policy_capture_ready(),
        "enabled reference has no matching answer yet"
    );
    app.select_revision(1).unwrap();
    app.capture = Some(dir.path().join("reference.png"));
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let mut out = render(&mut app, &ctx, vec![]);
        let capture = out.viewport_output.values().any(|v| {
            v.commands
                .iter()
                .any(|c| matches!(c, egui::ViewportCommand::Screenshot(_)))
        });
        out.textures_delta.clear();
        if capture {
            let f = app.interactive.policy_frame().unwrap();
            assert_eq!(f.requested_revision, 1);
            assert!(f.selected.as_ref().unwrap().reference_value.is_some());
            break;
        }
        assert!(Instant::now() < deadline, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
#[cfg(feature = "screenshots")]
fn policy_pending_discovery_clears_previous_terminal_capture_state() {
    let (_directory, context, mut app) = fixture();
    settle(&mut app, &context);
    let run = app.run.clone().unwrap();
    app.policy_pane.ready = false;
    app.policy_pane.terminal = true;
    app.policy_pane.status = Some("Previous input was unavailable".into());
    app.workspace_defaults = Some((
        crate::presentation::WorkspaceDefaults::new(&run, &app.selection),
        app.selection.input.clone(),
        app.comparison.reference_interaction_serial(),
    ));
    let mut result = Ok(());
    let mut output = context.run_ui(egui::RawInput::default(), |ui| {
        result = app.policy_workspace(ui, &run);
    });
    output.textures_delta.clear();
    result.unwrap();
    assert_eq!(app.policy_pane.status(), Some("Loading…"));
    assert!(
        !app.policy_pane.terminal,
        "pending work must clear a previous terminal publication state"
    );
    app.workspace_defaults = None;
    assert!(!app.interactive.pending() && !app.policy_pending() && app.loader.pending() == 0);
    assert_eq!(
        app.policy_capture_problem(None, None),
        None,
        "a transient Loading state must not become an idle capture failure"
    );
}

#[test]
#[cfg(feature = "screenshots")]
fn policy_unavailable_capture_requires_exact_nonempty_status_and_idle_terminal_state() {
    let (_dir, ctx, mut app) = fixture();
    settle(&mut app, &ctx);
    let expected =
        "invalid configuration: Exact reference requires matching authored scientific contexts";
    app.policy_pane.ready = false;
    app.policy_pane.terminal = true;
    app.policy_pane.status = Some(expected.into());
    assert!(app.policy_capture_ready_for(Some(expected)));
    assert_eq!(app.policy_capture_problem(None, Some(expected)), None);
    for expectation in [None, Some("different failure")] {
        assert!(!app.policy_capture_ready_for(expectation));
        assert_eq!(
            app.policy_capture_problem(None, expectation).as_deref(),
            Some(expected)
        );
    }
    for empty in ["", " "] {
        assert!(!app.policy_capture_ready_for(Some(empty)));
        assert!(app.policy_capture_problem(None, Some(empty)).is_some());
    }
    app.policy_pane.terminal = false;
    assert!(!app.policy_capture_ready_for(Some(expected)));
    assert_eq!(app.policy_capture_problem(None, Some(expected)), None);
    app.policy_pane.terminal = true;
    let mut request = app.policy_pane.request.clone().unwrap();
    request.pin = crate::policy::prepared::PolicyPin::Recorded {
        key: request.current.identity().key.clone(),
        probe_key: request.current.identity().probe_key.clone().unwrap(),
        row: 1,
    };
    app.interactive.submit_policy(request).unwrap();
    assert!(!app.policy_capture_ready_for(Some(expected)));
    assert_eq!(app.policy_capture_problem(None, Some(expected)), None);
    app.interactive.retire_session();
    let end = Instant::now() + Duration::from_secs(10);
    while app.interactive.pending() {
        app.interactive.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    let run = app.run.clone().unwrap();
    app.loader.cache.clear();
    assert!(app.loader.request(&run, 0).unwrap().is_none());
    assert!(!app.policy_capture_ready_for(Some(expected)));
    assert_eq!(app.policy_capture_problem(None, Some(expected)), None);
    while app.loader.pending() > 0 {
        app.loader.poll();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    app.workspace_defaults = Some((
        crate::presentation::WorkspaceDefaults::new(&run, &app.selection),
        app.selection.input.clone(),
        app.comparison.reference_interaction_serial(),
    ));
    assert!(!app.policy_capture_ready_for(Some(expected)));
    assert_eq!(app.policy_capture_problem(None, Some(expected)), None);
    app.workspace_defaults = None;
    assert!(app.policy_capture_ready_for(Some(expected)));
    app.policy_pane.ready = true;
    assert!(!app.policy_capture_ready_for(Some(expected)));
    assert!(app.policy_capture_problem(None, Some(expected)).is_some());
}

#[test]
fn policy_early_enabled_visits_wait_for_discovery_then_prepare_once() {
    let (_dir, ctx, mut app) = fixture_visits(true);
    #[cfg(feature = "screenshots")]
    {
        app.policy_pane.capture_configured = true;
    }
    app.policy_pane.visits_enabled = true;
    settle(&mut app, &ctx);
    let frame = app.policy_frame().unwrap();
    assert_eq!(
        frame
            .grid
            .visits
            .as_ref()
            .expect("enabled real visits must be prepared")
            .filtered_count,
        2
    );
    let grid = std::sync::Arc::as_ptr(&frame.grid);
    let before = app.policy_metrics();
    for _ in 0..10 {
        render(&mut app, &ctx, vec![]);
    }
    assert_eq!(
        std::sync::Arc::as_ptr(&app.policy_frame().unwrap().grid),
        grid
    );
    assert_eq!(app.policy_metrics().grid_batches, before.grid_batches);
    assert_eq!(app.policy_metrics().prepared_bytes, before.prepared_bytes);
    let (_missing_dir, missing_ctx, mut missing) = fixture_visits(false);
    #[cfg(feature = "screenshots")]
    {
        missing.policy_pane.capture_configured = true;
    }
    missing.policy_pane.visits_enabled = true;
    settle(&mut missing, &missing_ctx);
    assert!(missing.policy_pane.visit_search_done);
    assert!(missing.policy_frame().unwrap().grid.visits.is_none());
    assert!(missing.policy_pane.ready());
}

#[test]
fn final_exhausted_visits_restart_at_later_checkpoint_and_stay_idle_when_settled() {
    for available in [true, false] {
        let (_dir, ctx, mut app) = fixture_visits(available);
        #[cfg(feature = "screenshots")]
        {
            app.policy_pane.capture_configured = true;
        }
        app.select_revision(1).unwrap();
        render(&mut app, &ctx, vec![]);
        settle(&mut app, &ctx);
        for _ in 0..10 {
            render(&mut app, &ctx, vec![]);
        }
        assert!(app.policy_pane.visit_search_done);
        assert!(app.policy_pane.visits.is_none());
        app.select_revision(5).unwrap();
        render(&mut app, &ctx, vec![]);
        settle(&mut app, &ctx);
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.policy_pane.visits.is_none() && !app.policy_pane.visit_search_done {
            render(&mut app, &ctx, vec![]);
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            app.policy_pane.visits.is_some(),
            available,
            "later revision discovery"
        );
        if available {
            click(&mut app, &ctx, "Visited states");
            settle(&mut app, &ctx);
            let visits = app.policy_frame().unwrap().grid.visits.as_ref().unwrap();
            assert_eq!(visits.filtered_count, 2);
            assert_eq!(visits.revision, 5);
        } else {
            assert!(app.policy_pane.visit_search_done);
        }
        let before = app.policy_metrics();
        let checked = app.policy_pane.visit_checked;
        let cursor = app.policy_pane.visit_cursor;
        let grid = Arc::as_ptr(&app.policy_frame().unwrap().grid);
        for _ in 0..10 {
            render(&mut app, &ctx, vec![]);
        }
        assert_eq!(app.policy_pane.visit_checked, checked);
        assert_eq!(app.policy_pane.visit_cursor, cursor);
        assert_eq!(app.policy_metrics().grid_batches, before.grid_batches);
        assert_eq!(app.policy_metrics().prepared_bytes, before.prepared_bytes);
        assert_eq!(Arc::as_ptr(&app.policy_frame().unwrap().grid), grid);
    }
}

#[test]
fn policy_pending_status_keeps_the_main_plot_coordinate_transform_fixed() {
    let (_dir, ctx, mut app) = fixture();
    settle(&mut app, &ctx);
    let environment = rl::diagnostics::visualization::EnvironmentSemantics::new(
        app.run.as_ref().unwrap().recording().manifest(),
    );
    let frame = app.interactive.policy_frame();
    let shown_checkpoint = format!(
        "Showing checkpoint {}",
        frame.unwrap().grid.current.revision
    );
    let mut paint = |status: Option<&str>, ready: bool| {
        app.policy_pane.status = status.map(str::to_owned);
        app.policy_pane.ready = ready;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1440., 900.),
                )),
                ..Default::default()
            },
            |ui| {
                app.policy_pane.show(ui, environment, frame);
            },
        );
        output.textures_delta.clear();
        let has_text = |label| {
            output
                .shapes
                .iter()
                .any(|shape| text_position(&shape.shape, label).is_some())
        };
        let rect =
            *egui_plot::PlotMemory::load(&ctx, crate::policy::view::PolicyPane::main_plot_id())
                .unwrap()
                .transform()
                .frame();
        (
            rect,
            has_text("Showing previous complete frame · Loading…"),
            has_text(&shown_checkpoint),
            has_text("Showing previous complete frame · Recorded samples · surface unavailable"),
        )
    };
    paint(None, true);
    let (baseline, _, _, _) = paint(None, true);
    let (pending, transient, redundant_checkpoint, _) = paint(Some("Loading…"), false);
    assert_eq!(
        pending, baseline,
        "pending pin must not move its coordinate transform"
    );
    assert!(
        !transient,
        "pending redraws must omit the retained-frame caveat"
    );
    assert!(
        !redundant_checkpoint,
        "a matching checkpoint needs no warning"
    );
    let (unavailable, _, _, persistent) =
        paint(Some("Recorded samples · surface unavailable"), false);
    assert_eq!(unavailable, baseline);
    assert!(persistent, "a missing surface needs an explanation");
    assert_eq!(paint(None, false).0, baseline);
    assert!(!paint(None, false).1);
    assert_eq!(paint(None, true).0, baseline);
}

#[test]
#[cfg(feature = "screenshots")]
fn policy_native_hover_hook_uses_the_ordinary_exact_pin_position() {
    let (_dir, ctx, mut app) = fixture();
    settle(&mut app, &ctx);
    let mut input = egui::RawInput::default();
    app.policy_capture_hover(&ctx, &mut input, None);
    assert!(input.events.is_empty());
    app.policy_capture_hover(&ctx, &mut input, Some(false));
    assert!(matches!(
        input.events.as_slice(),
        [egui::Event::PointerGone]
    ));
    input.events.clear();
    app.policy_capture_hover(&ctx, &mut input, Some(true));
    let pin = app.policy_pane.pinned_input().unwrap();
    let evidence = app.policy_capture_evidence();
    assert_eq!(
        evidence["selected"]["input"],
        serde_json::json!(pin.dense_values())
    );
    assert_eq!(evidence["requested_revision"], app.selection.revision);
    assert_eq!(
        evidence["rows"],
        app.policy_frame().unwrap().grid.values.len()
    );
    let config = app.policy_pane.config().unwrap();
    let memory =
        egui_plot::PlotMemory::load(&ctx, crate::policy::view::PolicyPane::main_plot_id()).unwrap();
    let expected = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(
            pin.dense_values()[config.x] as f64,
            pin.dense_values()[config.y.unwrap()] as f64,
        ));
    assert!(matches!(input.events.as_slice(), [egui::Event::PointerMoved(p)] if *p == expected));
    app.select_view(View::Network);
    input.events.clear();
    app.policy_capture_hover(&ctx, &mut input, Some(true));
    assert!(input.events.is_empty());
}

#[test]
fn ordinary_ui_admission_rejection_is_coherent_and_retires_to_zero() {
    let (_dir, ctx, mut app) = fixture();
    let reservation = app
        .interactive
        .reserve_policy_view(crate::budgets::POLICY_PREPARED - 1024)
        .unwrap();
    // Admission is attempted after asynchronous cache discovery has selected
    // Interactive. Its first paint need not finish that unrelated discovery.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let mut out = render(&mut app, &ctx, vec![]);
        out.textures_delta.clear();
        if app.policy_mode() == Some(PolicyMode::Interactive) {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!app.policy_pane.ready());
    assert!(app.policy_pane.config().is_none());
    assert!(app.interactive.policy_frame().is_none());
    assert!(app.policy_pane.status().unwrap().contains("4 MiB"));
    assert!(app.interactive.policy_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    drop(reservation);
    settle(&mut app, &ctx);
    assert!(app.policy_pane.owned_state_bytes() <= app.policy_pane.state_reservation_bytes());
    app.select_view(View::Network);
    let deadline = Instant::now() + Duration::from_secs(10);
    while app.interactive.policy_metrics().prepared_bytes != 0 {
        app.interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(app.policy_pane.state_reservation_bytes(), 0);
}

#[test]
fn policy_pending_1d_pin_has_no_fabricated_output_marker() {
    let (_dir, ctx, mut app) = fixture();
    settle(&mut app, &ctx);
    click(&mut app, &ctx, "Slice");
    settle(&mut app, &ctx);
    let grid = app.policy_frame().unwrap().grid.clone();
    let pending = crate::policy::prepared::PreparedPolicyFrame {
        grid,
        selected: None,
        requested_revision: app.selection.revision,
    };
    app.policy_pane.pin_input = Some(Observation::new(vec![0.1, 0.]));
    let manifest = app.run.as_ref().unwrap().recording().manifest().clone();
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 900.),
            )),
            ..Default::default()
        },
        |ui| {
            app.policy_pane.show(
                ui,
                rl::diagnostics::visualization::EnvironmentSemantics::new(&manifest),
                Some(&pending),
            );
        },
    );
    out.textures_delta.clear();
    fn marker(s: &egui::Shape) -> bool {
        match s {
            egui::Shape::Circle(c) => c.fill == egui::Color32::WHITE && (c.radius - 6.).abs() < 0.1,
            egui::Shape::Vec(v) => v.iter().any(marker),
            _ => false,
        }
    }
    assert!(
        !out.shapes.iter().any(|s| marker(&s.shape)),
        "pending exact output must not be drawn at y=0"
    );
}

#[test]
fn exact_snapshot_header_and_content_publish_in_one_paint_with_reference_discovery() {
    use crate::policy::{
        prepared::{PolicyPin, PolicySource},
        slice::SliceConfig,
    };
    let (_dir, run) = crate::interactive::value_worker_tests::fixture();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::PolicyMap);
    let source = PolicySource::model(&run, 0, Some("actor")).unwrap();
    let input = Observation::new(vec![0.25]);
    let mut config = SliceConfig::new(source.semantics(), &input, &[]).unwrap();
    config.resolution = [16, 16];
    app.policy_pane.config = Some(config.clone());
    #[cfg(feature = "screenshots")]
    {
        app.policy_pane.capture_configured = true;
    }
    app.policy_pane.pin_input = Some(input.clone());
    app.policy_pane.pin = Some(PolicyPin::Synthetic(input.clone()));
    app.policy_pane.reference_enabled = true;
    app.policy_pane.dirty = true;
    app.comparison.set_reference_revision(0).unwrap();
    let end = Instant::now() + Duration::from_secs(15);
    loop {
        let out = render(&mut app, &ctx, vec![]);
        if let Some(frame) = app.policy_frame() {
            let positions = app.snapshot_catalog.as_ref().unwrap().positions();
            let n = positions
                .iter()
                .position(|&k| k == frame.grid.current.revision)
                .unwrap()
                + 1;
            let label = format!("{n} / {}", positions.len());
            assert!(
                out.shapes
                    .iter()
                    .any(|s| text_position(&s.shape, &label).is_some()),
                "header must name the complete frame painted in this pass"
            );
            assert_eq!(
                frame.selected.as_ref().unwrap().value,
                frame.grid.current.revision as f64
            );
        }
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending()) && app.policy_pane.ready() {
            break;
        }
        assert!(Instant::now() < end, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        app.snapshot_catalog.as_ref().unwrap().positions(),
        vec![0, 1, 2]
    );
    app.select_revision(2).unwrap();
    let mut observed = false;
    while !observed {
        let out = render(&mut app, &ctx, vec![]);
        let frame = app.policy_frame().unwrap();
        let label = format!("{} / 3", frame.grid.current.revision + 1);
        assert!(
            out.shapes
                .iter()
                .any(|s| text_position(&s.shape, &label).is_some())
        );
        observed = frame.grid.current.revision == 2;
        assert_eq!(frame.selected.as_ref().unwrap().input, input);
        assert_eq!(app.policy_pane.config.as_ref().unwrap(), &config);
        assert!(Instant::now() < end, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        app.policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value,
        Some(0.)
    );
    app.select_view(View::Network);
    assert!(app.snapshot_catalog.is_none());
}

#[test]
fn value_snapshot_header_matches_complete_grid_point_and_learned_overlay() {
    use crate::value::{
        prepared::{ValuePin, ValueSource},
        slice::ValueSlice,
        view::CriticChoice,
    };
    use rl::diagnostics::value::{ValueCriticKind, ValueOutput};
    let (_dir, run) = crate::interactive::value_worker_tests::fixture();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    let source = ValueSource::model(&run, 0, "q", &run.recording().load(0).unwrap()).unwrap();
    let input = Observation::new(vec![0.25]);
    let mut config = ValueSlice::new(source.semantics(), &input, &[]).unwrap();
    config.resolution = [16, 1];
    app.value_view.config = Some(config.clone());
    app.value_view.pin = Some(ValuePin::State(input.clone()));
    app.value_view.choice = Some(CriticChoice {
        name: "q".into(),
        twin: None,
        kind: ValueCriticKind::ScalarQ,
    });
    app.value_view.output = Some(ValueOutput::ExpectedReturn);
    app.value_view.dirty = true;
    #[cfg(feature = "screenshots")]
    {
        app.value_view.capture_configured = true;
    }
    let end = Instant::now() + Duration::from_secs(15);
    loop {
        let out = render(&mut app, &ctx, vec![]);
        if let Some(f) = app.value_frame() {
            let positions = app.snapshot_catalog.as_ref().unwrap().positions();
            let n = positions
                .iter()
                .position(|&k| k == f.grid.current().identity().revision)
                .unwrap()
                + 1;
            assert!(
                out.shapes.iter().any(|s| text_position(
                    &s.shape,
                    &format!("{n} / {}", positions.len())
                )
                .is_some())
            );
        }
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
            && app.value_frame().is_some()
        {
            break;
        }
        assert!(Instant::now() < end, "{:?}", app.value_view.status());
        std::thread::sleep(Duration::from_millis(2));
    }
    app.select_revision(2).unwrap();
    loop {
        let out = render(&mut app, &ctx, vec![]);
        let f = app.value_frame().unwrap();
        let revision = f.grid.current().identity().revision;
        assert!(
            out.shapes
                .iter()
                .any(|s| text_position(&s.shape, &format!("{} / 3", revision + 1)).is_some())
        );
        assert_eq!(f.selected.as_ref().unwrap().input, input);
        assert_eq!(
            f.selected.as_ref().unwrap().value,
            0.25 + 2. * revision as f64
        );
        assert!(Instant::now() < end, "{:?}", app.value_view.status());
        if revision == 2 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

// Real serialized snapshots and genuine measurements deliberately have different times.
fn snapshot_transition_fixture(
    policy_only: bool,
) -> (tempfile::TempDir, egui::Context, Laboratory) {
    use rl::diagnostics::{
        models::{CriticInspectionRequest, InspectionRequest, ModelBundle, ModelLimits},
        value::*,
    };
    let (_original_dir, original) = crate::interactive::value_worker_tests::fixture();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        original.recording().manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for revision in [0, 120, 240] {
        let mut chunk = original.recording().load(0).unwrap();
        let mut d = declaration(&chunk).unwrap().unwrap();
        chunk.context.identity.learner_update_revision = revision;
        chunk.context.counters.updates = revision;
        d.revision = revision;
        chunk.arrays.remove(VALUE_METADATA_KEY);
        if policy_only {
            if let Record::ModelSnapshot(s) = &mut chunk.record {
                s.configuration["models"]
                    .as_object_mut()
                    .unwrap()
                    .remove("q");
                s.models.remove("q");
            }
        } else {
            attach_declaration(&mut chunk, &d).unwrap();
        }
        writer.append(&chunk).unwrap();
        if revision == 120 {
            let model = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
            let observations = vec![Observation::new(vec![0.25])];
            let inspected = model
                .inspect(&InspectionRequest::new("actor", observations.clone()))
                .unwrap();
            writer.append(&inspected.probes).unwrap();
            writer.append(&inspected.policy).unwrap();
            if !policy_only {
                let inspected = model
                    .inspect_critics(&CriticInspectionRequest {
                        critics: vec!["q".into()],
                        observations,
                        fixed_actions: Some(vec![rl::core::action::Action::Continuous(vec![0.])]),
                    })
                    .unwrap();
                writer.append(&inspected.probes).unwrap();
                writer.append(&inspected.critics["q"]).unwrap();
            }
        }
    }
    writer.finish().unwrap();
    let ctx = egui::Context::default();
    let app = Laboratory::from_context(&ctx, dir.path().join("run").display().to_string()).unwrap();
    (dir, ctx, app)
}

#[test]
fn review_fix_uninitialized_policy_preserves_requested_snapshot() {
    uninitialized_snapshot_selection(View::PolicyMap);
}

#[test]
fn ordinary_snapshot_routes_warm_without_publishing_and_idle_without_repeat_work() {
    for view in [View::PolicyMap, View::Value] {
        let (_dir, ctx, mut app) = snapshot_transition_fixture(false);
        app.select_view(view);
        app.select_revision(0).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            render(&mut app, &ctx, vec![]);
            if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
                && !app.interactive.pending()
                && if view == View::PolicyMap {
                    app.policy_frame().is_some()
                } else {
                    app.value_frame().is_some()
                }
            {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        // Give the ordinary post-settlement warmup its dispatch/completion paints.
        for _ in 0..30 {
            render(&mut app, &ctx, vec![]);
            std::thread::sleep(Duration::from_millis(2));
        }
        let before = app.interactive.policy_metrics();
        for _ in 0..10 {
            render(&mut app, &ctx, vec![]);
        }
        let idle = app.interactive.policy_metrics();
        assert_eq!(
            (before.restores, before.grid_batches, before.point_batches),
            (idle.restores, idle.grid_batches, idle.point_batches)
        );
        assert_eq!(app.selection.revision, 0);
        assert_eq!(
            if view == View::PolicyMap {
                app.policy_frame().unwrap().grid.current.revision
            } else {
                app.value_frame()
                    .unwrap()
                    .grid
                    .current()
                    .identity()
                    .revision
            },
            0
        );
        app.select_revision(120).unwrap();
        loop {
            render(&mut app, &ctx, vec![]);
            let revision = if view == View::PolicyMap {
                app.policy_frame().unwrap().grid.current.revision
            } else {
                app.value_frame()
                    .unwrap()
                    .grid
                    .current()
                    .identity()
                    .revision
            };
            if revision == 120 {
                break;
            }
            assert_eq!(revision, 0);
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            app.interactive.policy_metrics().restores,
            before.restores,
            "next exact bundle is warm"
        );
        if view == View::PolicyMap {
            assert_eq!(
                app.interactive.policy_metrics().grid_batches,
                before.grid_batches,
                "the next policy grid was prepared before navigation"
            );
        }
    }
}

#[test]
fn value_training_round_trip_reuses_the_compatible_prepared_frame() {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(false);
    app.select_view(View::Value);
    app.select_revision(120).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_preparation_idle()
            && app
                .value_frame()
                .is_some_and(|f| f.requested_revision == 120)
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let original_grid = std::sync::Arc::as_ptr(&app.value_frame().unwrap().grid) as usize;
    app.select_view(View::Training);
    render(&mut app, &ctx, vec![]);
    let before = app.value_metrics();
    app.select_view(View::Value);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_preparation_idle()
            && app
                .value_frame()
                .is_some_and(|f| f.requested_revision == 120)
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let after = app.value_metrics();
    assert_eq!(
        std::sync::Arc::as_ptr(&app.value_frame().unwrap().grid) as usize,
        original_grid
    );
    assert_eq!(
        (after.restores, after.grid_batches, after.point_batches),
        (before.restores, before.grid_batches, before.point_batches)
    );

    // A checkpoint selected while Training is visible must not acquire the
    // retained frame's old scientific identity.
    app.select_view(View::Training);
    app.select_revision(240).unwrap();
    app.select_view(View::Value);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_preparation_idle()
            && app
                .value_frame()
                .is_some_and(|f| f.requested_revision == 240)
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let current = app.value_frame().unwrap();
    assert_eq!(current.grid.current().identity().revision, 240);
    assert!(app.value_metrics().grid_batches > after.grid_batches);
}

#[test]
fn backward_navigation_prefetches_the_previous_policy_without_publishing_it() {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(false);
    app.select_view(View::PolicyMap);
    app.select_revision(240).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_preparation_idle()
            && app
                .policy_frame()
                .is_some_and(|f| f.grid.current.revision == 240)
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    app.select_revision(120).unwrap();
    loop {
        render(&mut app, &ctx, vec![]);
        if app
            .policy_frame()
            .is_some_and(|f| f.grid.current.revision == 120)
            && !app.policy_pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    // Dispatch the bounded speculative request, then wait for it to settle.
    for _ in 0..30 {
        render(&mut app, &ctx, vec![]);
        std::thread::sleep(Duration::from_millis(2));
    }
    while !app.snapshot_preparation_idle() {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(app.selection.revision, 120);
    assert_eq!(app.policy_frame().unwrap().grid.current.revision, 120);
    let before = app.policy_metrics();
    app.select_revision(0).unwrap();
    loop {
        render(&mut app, &ctx, vec![]);
        if app
            .policy_frame()
            .is_some_and(|f| f.grid.current.revision == 0)
            && !app.policy_pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let after = app.policy_metrics();
    assert_eq!(after.restores, before.restores);
    assert_eq!(after.grid_batches, before.grid_batches);
}
#[test]
fn review_fix_uninitialized_value_preserves_requested_snapshot() {
    uninitialized_snapshot_selection(View::Value);
}
fn uninitialized_snapshot_selection(view: View) {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(false);
    app.select_view(view);
    app.select_revision(240).unwrap();
    assert!(app.policy_pane.config.is_none() && app.value_view.config.is_none());
    let end = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        assert_eq!(
            app.selection.revision, 240,
            "initialization must not seek to the sole measurement at120"
        );
        let actual = if view == View::PolicyMap {
            app.policy_frame().map(|f| f.grid.current.revision)
        } else {
            app.value_frame()
                .map(|f| f.grid.current().identity().revision)
        };
        if let Some(actual) = actual {
            assert_eq!(actual, 240);
            break;
        }
        assert!(
            Instant::now() < end,
            "policy={:?}, value={:?}",
            app.policy_pane.status(),
            app.value_view.status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn review_fix_panel_switch_does_not_show_retired_policy_counter_in_value() {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(true);
    app.select_view(View::PolicyMap);
    app.select_revision(240).unwrap();
    settle(&mut app, &ctx);
    let end = Instant::now() + Duration::from_secs(15);
    while app.snapshot_catalog.as_ref().is_none_or(|c| c.pending()) {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    app.select_revision(240).unwrap();
    while app
        .policy_frame()
        .is_none_or(|f| f.grid.current.revision != 240)
    {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    let out = render(&mut app, &ctx, vec![]);
    assert!(
        out.shapes
            .iter()
            .any(|s| text_position(&s.shape, "3 / 3").is_some())
    );
    app.select_view(View::Value);
    loop {
        let out = render(&mut app, &ctx, vec![]);
        assert!(app.value_frame().is_none());
        assert!(
            !out.shapes.iter().any(|s| ["1 / 3", "2 / 3", "3 / 3"]
                .iter()
                .any(|t| text_position(&s.shape, t).is_some())),
            "retired Policy publication must never label Value"
        );
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending()) {
            break;
        }
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn policy_value_switch_reuses_exact_catalogue_restoration_without_relabelling() {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(false);
    app.select_view(View::PolicyMap);
    app.select_revision(240).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
            && app
                .policy_frame()
                .is_some_and(|f| f.grid.current.revision == 240)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let selected = app.selection.input.clone();
    let policy_grid = Arc::downgrade(&app.policy_frame().unwrap().grid);
    let before = app.interactive.policy_metrics();
    app.select_view(View::Value);
    assert!(
        app.snapshot_catalog
            .as_ref()
            .unwrap()
            .positions()
            .is_empty(),
        "prior Policy eligibility must not label Value"
    );
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
            && app
                .value_frame()
                .is_some_and(|f| f.grid.current().identity().revision == 240)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "value={:?}",
            app.value_view.status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let after = app.interactive.policy_metrics();
    assert_eq!(app.selection.input, selected);
    assert!(
        after.restores - before.restores <= 1,
        "catalogue should reuse prior whole-snapshot restoration"
    );
    assert!(after.snapshot_verifications - before.snapshot_verifications >= 3);
    assert!(after.prepared_peak <= crate::budgets::POLICY_PREPARED);
    app.select_view(View::PolicyMap);
    assert!(
        app.snapshot_catalog
            .as_ref()
            .unwrap()
            .positions()
            .is_empty(),
        "prior Value eligibility must not label Policy"
    );
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
            && app
                .policy_frame()
                .is_some_and(|f| f.grid.current.revision == 240)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "policy={:?}",
            app.policy_pane.status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let returned = app.interactive.policy_metrics();
    assert_eq!(app.selection.input, selected);
    let original = policy_grid
        .upgrade()
        .expect("the compatible Policy grid should remain cached");
    assert!(Arc::ptr_eq(&original, &app.policy_frame().unwrap().grid));
    assert_eq!(
        returned.restores, after.restores,
        "a compatible cached Policy return must not restore a model"
    );
    assert_eq!(
        returned.grid_batches, after.grid_batches,
        "a compatible cached Policy return must not repeat grid inference"
    );
    assert!(returned.snapshot_verifications - after.snapshot_verifications >= 3);
    assert!(returned.prepared_peak <= crate::budgets::POLICY_PREPARED);
    drop(original);

    app.select_view(View::Value);
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.snapshot_catalog.as_ref().is_some_and(|c| !c.pending())
            && app
                .value_frame()
                .is_some_and(|f| f.grid.current().identity().revision == 240)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    app.select_revision(120).unwrap();
    loop {
        render(&mut app, &ctx, vec![]);
        if app
            .value_frame()
            .is_some_and(|f| f.grid.current().identity().revision == 120)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let before_changed_return = app.interactive.policy_metrics();
    app.select_view(View::PolicyMap);
    loop {
        render(&mut app, &ctx, vec![]);
        if app
            .policy_frame()
            .is_some_and(|f| f.grid.current.revision == 120)
            && !app.interactive.pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        app.interactive.policy_metrics().grid_batches > before_changed_return.grid_batches,
        "a changed checkpoint cannot reuse the old grid"
    );
}

#[test]
fn review_fix_live_policy_loss_keeps_complete_shapes_and_actionable_error() {
    let (_dir, ctx, mut app) = snapshot_transition_fixture(true);
    app.select_view(View::PolicyMap);
    app.select_revision(240).unwrap();
    settle(&mut app, &ctx);
    let end = Instant::now() + Duration::from_secs(15);
    while app.snapshot_catalog.as_ref().is_none_or(|c| c.pending()) {
        render(&mut app, &ctx, vec![]);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    let old = app.policy_frame().unwrap().clone();
    let before = render(&mut app, &ctx, vec![]);
    fn markers(shape: &egui::Shape, points: &mut Vec<egui::Pos2>) {
        match shape {
            egui::Shape::Circle(c)
                if c.fill == egui::Color32::WHITE && (c.radius - 6.).abs() < 0.1 =>
            {
                points.push(c.center)
            }
            egui::Shape::Vec(v) => {
                for s in v {
                    markers(s, points);
                }
            }
            _ => {}
        }
    }
    let mut old_markers = vec![];
    for s in &before.shapes {
        markers(&s.shape, &mut old_markers);
    }
    assert!(
        !old_markers.is_empty(),
        "fixture must paint its real selected marker"
    );
    let run = app.run.as_ref().unwrap();
    let index = run.by_sequence(old.grid.current.key.sequence).unwrap();
    let source_path = run.root().join(&run.recording().entries()[index].file);
    let source_bytes = std::fs::read(&source_path).unwrap();
    std::fs::remove_file(&source_path).unwrap();
    app.live = true;
    app.last_refresh = Instant::now() - Duration::from_secs(2);
    let out = render(&mut app, &ctx, vec![]);
    let held = app.policy_frame().unwrap();
    assert!(Arc::ptr_eq(&old.grid, &held.grid));
    assert!(Arc::ptr_eq(
        old.selected.as_ref().unwrap(),
        held.selected.as_ref().unwrap()
    ));
    assert_eq!(
        held.selected.as_ref().unwrap().input,
        old.selected.as_ref().unwrap().input
    );
    let mut held_markers = vec![];
    for s in &out.shapes {
        markers(&s.shape, &mut held_markers);
    }
    assert_eq!(
        held_markers, old_markers,
        "held selected marker must still be painted"
    );
    let reason = app.interactive.policy_error().unwrap().to_owned();
    click(&mut app, &ctx, "Unavailable");
    let out = render(&mut app, &ctx, vec![]);
    assert!(
        out.shapes
            .iter()
            .any(|s| text_position(&s.shape, &reason).is_some()),
        "the complete unavailable reason remains accessible"
    );
    assert!(
        out.shapes
            .iter()
            .any(|s| text_position(&s.shape, "Retry").is_some()),
        "missing Retry"
    );
    assert!(app.interactive.policy_error().is_some());
    assert!(!app.policy_pane.ready());
    std::fs::write(&source_path, source_bytes).unwrap();
    click(&mut app, &ctx, "Retry");
    settle(&mut app, &ctx);
    let recovered = app.policy_frame().unwrap();
    assert_eq!(recovered.grid.current, old.grid.current);
    assert_eq!(
        recovered.selected.as_ref().unwrap().input,
        old.selected.as_ref().unwrap().input
    );
    assert_eq!(
        recovered.selected.as_ref().unwrap().value,
        old.selected.as_ref().unwrap().value
    );
}

#[test]
fn value_reference_switches_model_to_recorded_oracle_and_back_without_stale_choices() {
    use crate::policy::slice::SliceMode;
    use crate::value::{
        prepared::{ValuePin, ValueSource},
        slice::{ValueCoordinate, ValueSlice},
        view::CriticChoice,
    };
    use rl::diagnostics::value::{ValueCriticKind, ValueOutput};
    let (_dir, run) = crate::reference::tests::model_switch_recording();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    let source = ValueSource::model(&run, 6, "value", &run.recording().load(6).unwrap()).unwrap();
    let input = Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
    let mut config = ValueSlice::new(source.semantics(), &input, &[]).unwrap();
    config.mode = SliceMode::OneDimensional;
    config.x = ValueCoordinate::DiscreteAction;
    config.y = None;
    app.value_view.config = Some(config);
    app.value_view.pin = Some(ValuePin::State(input));
    app.value_view.choice = Some(CriticChoice {
        name: "value".into(),
        twin: None,
        kind: ValueCriticKind::DiscreteQ,
    });
    app.value_view.output = Some(ValueOutput::ExpectedReturn);
    app.value_view.reference_enabled = true;
    app.value_view.dirty = true;
    #[cfg(feature = "screenshots")]
    {
        app.value_view.capture_configured = true;
    }
    app.comparison.reset_reference_to(&run).unwrap();
    fn settle_reference(app: &mut Laboratory, ctx: &egui::Context, name: &str) {
        let end = Instant::now() + Duration::from_secs(15);
        loop {
            render(app, ctx, vec![]);
            if app.value_view.ready()
                && app.value_frame().is_some_and(|f| {
                    f.grid
                        .reference()
                        .is_some_and(|s| s.identity().name == name)
                        && f.grid.reference_error.is_none()
                })
            {
                break;
            }
            assert!(
                Instant::now() < end,
                "status {:?}; reference {:?}; search {:?}",
                app.value_view.status,
                app.value_frame().map(|f| &f.grid.reference_error),
                app.value_view
                    .value_search
                    .as_ref()
                    .and_then(|s| s.last_rejection.as_ref())
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn click_topmost(app: &mut Laboratory, ctx: &egui::Context, text: &str) {
        let out = render(app, ctx, vec![]);
        let position = out
            .shapes
            .iter()
            .rev()
            .find_map(|s| text_position(&s.shape, text))
            .unwrap_or_else(|| panic!("missing {text}"));
        for pressed in [true, false] {
            render(
                app,
                ctx,
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            );
        }
    }
    settle_reference(&mut app, &ctx, "value");
    assert!(
        app.value_view
            .reference_choices
            .iter()
            .any(|c| c.name == "value")
    );
    // Exercise the shared selector while Value holds prepared model choices.
    click(&mut app, &ctx, "Reference source");
    click_topmost(&mut app, &ctx, "online");
    click_topmost(&mut app, &ctx, "oracle");
    assert_eq!(app.comparison.reference_role(), "oracle");
    settle_reference(&mut app, &ctx, "q");
    assert!(app.value_view.reference_choices.is_empty());
    assert!(app.value_view.reference_policies.is_empty());
    app.comparison.set_reference_role("online").unwrap();
    let out = render(&mut app, &ctx, vec![]);
    assert!(
        !out.shapes
            .iter()
            .any(|s| text_position(&s.shape, "Verified optimal reference").is_some())
    );
    settle_reference(&mut app, &ctx, "value");
    assert!(
        app.value_view
            .reference_choices
            .iter()
            .any(|c| c.name == "value")
    );
}

#[test]
#[cfg(feature = "screenshots")]
#[ignore = "requires the complete local alpha execution recording"]
fn real_alpha_same_input_reference_selects_action_value() {
    let path = std::env::var("RL_REFERENCE_BENCH_RUN").expect("real recording path");
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path).unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.value_view.terminal
            && let Some(r) = &app.value_view.request
        {
            eprintln!(
                "admission: resolution={:?}, frame={:?}, metadata={}, empirical={}, view={}/{}, metrics={:?}",
                r.slice.resolution,
                crate::value::prepared::publication_charge(r),
                r.metadata_bytes(),
                r.empirical.is_some(),
                app.value_view.owned_state_bytes(),
                app.value_view.state_reservation_bytes(),
                app.value_metrics()
            );
        }
        assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
        if app.value_view.ready() && !app.value_pending() {
            break;
        }
        assert!(
            Instant::now() < end,
            "status={:?} ready={} pending={} frame={:?} metrics={:?}",
            app.value_view.status,
            app.value_view.ready(),
            app.value_pending(),
            app.value_frame()
                .map(|f| (f.requested_revision, f.grid.current().identity().revision)),
            app.value_metrics()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let frame = app.value_frame().unwrap();
    assert_eq!(
        frame.grid.reference().unwrap().kind(),
        rl::diagnostics::value::ValueCriticKind::ScalarQ
    );
    assert!(
        frame.grid.reference_error.is_none(),
        "{:?}",
        frame.grid.reference_error
    );
    assert!(frame.selected.as_ref().unwrap().reference_value.is_some());
    let empirical = frame
        .selected
        .as_ref()
        .unwrap()
        .empirical
        .as_ref()
        .expect("matched conditional evidence");
    assert_eq!(empirical.condition.summary.count, 512);
    assert!(frame.grid.slice.resolution[0] >= 32 && frame.grid.slice.resolution[1] >= 32);
    assert!(app.value_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    assert!(app.value_view.owned_state_bytes() <= app.value_view.state_reservation_bytes());
    let output = render(&mut app, &ctx, vec![]);
    assert!(
        output
            .shapes
            .iter()
            .any(|shape| text_position(&shape.shape, "Reference value").is_some())
    );
    assert!(
        output
            .shapes
            .iter()
            .any(|shape| text_position(&shape.shape, "Empirical").is_some())
    );
    fn numeric_positions(shape: &egui::Shape, positions: &mut Vec<egui::Pos2>) {
        match shape {
            egui::Shape::Text(t) if t.galley.text().parse::<f64>().is_ok() => {
                positions.push(t.pos + t.galley.rect.center().to_vec2());
            }
            egui::Shape::Vec(shapes) => {
                for shape in shapes {
                    numeric_positions(shape, positions);
                }
            }
            _ => {}
        }
    }
    let curve =
        egui_plot::PlotMemory::load(&ctx, egui::Id::unique("value-selected-action-curve")).unwrap();
    let plot = *curve.transform().frame();
    let mut numeric = Vec::new();
    for shape in &output.shapes {
        numeric_positions(&shape.shape, &mut numeric);
    }
    let ticks = numeric
        .iter()
        .filter(|p| p.x < plot.left() && p.y >= plot.top() && p.y <= plot.bottom())
        .count();
    assert!(
        ticks >= 2,
        "action-value axis needs multiple readable ticks, got {ticks}"
    );
    let mut compact = None;
    for _ in 0..3 {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1024., 640.),
                )),
                ..Default::default()
            },
            |ui| app.show(ui),
        );
        output.textures_delta.clear();
        compact = Some(output);
    }
    let compact = compact.unwrap();
    let position = |label| {
        compact
            .shapes
            .iter()
            .find_map(|shape| {
                text_position(&shape.shape, label).filter(|point| shape.clip_rect.contains(*point))
            })
            .unwrap_or_else(|| panic!("missing compact control {label}"))
    };
    let map = position("Map");
    let comparison = position("Compare value");
    assert!(
        (map.y - comparison.y).abs() < 10.,
        "Value comparison controls should share the local toolbar: map={map:?}, comparison={comparison:?}"
    );
    assert!(
        compact
            .shapes
            .iter()
            .any(|shape| text_position(&shape.shape, "Return")
                .is_some_and(|position| shape.clip_rect.contains(position))),
        "conditional return axis is clipped"
    );
    fn prefix_position(shape: &egui::Shape, prefix: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(t) if t.galley.text().starts_with(prefix) => {
                Some(t.pos + t.galley.rect.center().to_vec2())
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| prefix_position(s, prefix)),
            _ => None,
        }
    }
    for prefix in ["Mean ", "SD ", "CVaR₁₀ "] {
        assert!(
            compact
                .shapes
                .iter()
                .any(|shape| prefix_position(&shape.shape, prefix)
                    .is_some_and(|position| shape.clip_rect.contains(position))),
            "compact conditional statistic {prefix} is clipped"
        );
    }
    app.select_revision(19599).unwrap();
    loop {
        render(&mut app, &ctx, vec![]);
        assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
        if app.value_view.ready()
            && !app.value_pending()
            && app
                .value_frame()
                .is_some_and(|frame| frame.requested_revision == 19599)
        {
            break;
        }
        assert!(
            Instant::now() < end,
            "status={:?} ready={} pending={} frame={:?} metrics={:?}",
            app.value_view.status,
            app.value_view.ready(),
            app.value_pending(),
            app.value_frame()
                .map(|f| (f.requested_revision, f.grid.current().identity().revision)),
            app.value_metrics()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        app.value_frame()
            .unwrap()
            .grid
            .current()
            .identity()
            .revision,
        19599
    );
    click(&mut app, &ctx, "Compare value");
    loop {
        let output = render(&mut app, &ctx, vec![]);
        if output
            .shapes
            .iter()
            .any(|shape| text_position(&shape.shape, "Recorded diagnostics").is_some())
        {
            assert!(
                output.shapes.iter().any(|shape| {
                    text_position(&shape.shape, "Approximate numerical benchmark").is_some()
                }),
                "the comparison menu must disclose the numerical reference quality"
            );
            break;
        }
        assert!(
            Instant::now() < end,
            "recorded diagnostics did not appear in comparison menu"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn recorded_state_value_controller_comparison_preserves_exact_probe_and_uses_oracle_max_q() {
    use crate::{
        policy::slice::SliceMode,
        value::{
            prepared::{ValuePin, ValueSource},
            slice::{ValueCoordinate, ValueSlice},
            view::CriticChoice,
        },
    };
    use rl::{
        core::action::Action,
        diagnostics::value::{ComparisonCondition, ValueCriticKind, ValueOutput},
    };
    let (_dir, run) = crate::reference::tests::state_value_reference_recording();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    let key = run.key(0).unwrap();
    let probes = app
        .resolver
        .lease_record(
            &run,
            &key,
            std::sync::Arc::new(run.recording().load(0).unwrap()),
        )
        .unwrap();
    let critic_key = run.key(2).unwrap();
    let critic = app
        .resolver
        .lease_record(
            &run,
            &critic_key,
            std::sync::Arc::new(run.recording().load(2).unwrap()),
        )
        .unwrap();
    let current = ValueSource::recorded(
        &run,
        &crate::resolver::ResolvedRecord {
            key: critic_key,
            probe_key: key.clone(),
            chunk: critic,
            probes: probes.clone(),
            row: 1,
        },
        "value",
    )
    .unwrap();
    let input = crate::policy::prepared::exact_observation(&probes, 1).unwrap();
    let mut config = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
    config.mode = SliceMode::OneDimensional;
    config.x = ValueCoordinate::Observation(0);
    config.y = None;
    config.state.mode = SliceMode::OneDimensional;
    config.state.x = 0;
    config.state.y = None;
    app.selection = Selection {
        revision: 0,
        model_role: "online".into(),
        input: SelectedInput::ProbeRow {
            set: "online".into(),
            row: 1,
            sequence: 0,
        },
    };
    app.workspace_defaults = None;
    app.value_view.config = Some(config);
    app.value_view.pin = Some(ValuePin::Probe {
        source: crate::policy::prepared::VisitSource {
            key: key.clone(),
            chunk: probes,
        },
        row: 1,
        action: None,
    });
    app.value_view.anchor_key = Some((key.clone(), 1));
    app.value_view.choice = Some(CriticChoice {
        name: "value".into(),
        twin: None,
        kind: ValueCriticKind::StateValue,
    });
    app.value_view.output = Some(ValueOutput::ExpectedReturn);
    app.value_view.reference_enabled = true;
    app.value_view.condition = Some(ComparisonCondition::ControllerValues);
    app.value_view.dirty = true;
    #[cfg(feature = "screenshots")]
    {
        app.value_view.capture_configured = true;
    }
    app.comparison.reset_reference_to(&run).unwrap();
    app.comparison.set_reference_role("oracle").unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        render(&mut app, &ctx, vec![]);
        assert!(
            !app.value_view.terminal || app.value_view.ready(),
            "{:?}",
            app.value_view.status
        );
        if app.value_view.ready() {
            break;
        }
        assert!(Instant::now() < end, "{:?}", app.value_view.status);
        std::thread::sleep(Duration::from_millis(2));
    }
    let frame = app.value_frame().unwrap();
    assert_eq!(frame.grid.current().kind(), ValueCriticKind::StateValue);
    assert_eq!(
        frame.grid.reference().unwrap().kind(),
        ValueCriticKind::DiscreteQ
    );
    assert_eq!(frame.grid.reference_values, vec![Some(1.), Some(3.)]);
    assert!(frame.grid.reference_error.is_none());
    let point = frame.selected.as_ref().unwrap();
    assert_eq!(point.input, input);
    assert_eq!(point.input.dense_values()[4], 0.375);
    assert_eq!(point.recorded_key.as_ref(), Some(&key));
    assert_eq!(point.recorded_row, Some(1));
    assert_eq!(point.recorded_revision, Some(0));
    assert_eq!(point.value, -3.);
    assert_eq!(point.reference_value, Some(3.));
    assert_eq!(point.reference_action, Some(Action::Discrete(1)));
    assert!(point.policy_conditioned);
    assert!(point.reference_error.is_none());
    let request = app.value_view.request.as_ref().unwrap();
    assert!(matches!(
        request.pin,
        ValuePin::Probe {
            action: None,
            row: 1,
            ..
        }
    ));
    let mut fixed = request.clone();
    let ValuePin::Probe { action, .. } = &mut fixed.pin else {
        panic!()
    };
    *action = Some(Action::Discrete(0));
    assert!(fixed.validate().is_err());
    let mut q_pin = request.clone();
    q_pin.current = frame.grid.reference().unwrap().clone();
    assert!(q_pin.validate().is_err());
    let mut fixed_condition = request.clone();
    std::sync::Arc::make_mut(&mut fixed_condition.slice).conditioning =
        crate::value::slice::ActionConditioning::Fixed(Action::Discrete(0));
    assert!(fixed_condition.validate().is_err());
    let mut action_axis = request.clone();
    std::sync::Arc::make_mut(&mut action_axis.slice).x = ValueCoordinate::DiscreteAction;
    assert!(action_axis.validate().is_err());
    let mut test_resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
    for history in [false, true] {
        let mut probe = run.recording().load(0).unwrap();
        let Record::ProbeSet(p) = &mut probe.record else {
            panic!()
        };
        if history {
            p.histories.insert("past".into(), "past".into());
            probe.arrays.insert(
                "past".into(),
                Array::f32(vec![2, 1, 5], vec![0.; 10]).unwrap(),
            );
        } else {
            p.fixed_actions = Some("fixed".into());
            probe
                .arrays
                .insert("fixed".into(), Array::u64(vec![2], vec![0, 0]).unwrap());
        }
        probe.validate().unwrap();
        let mut conditioned = request.clone();
        let ValuePin::Probe { source, .. } = &mut conditioned.pin else {
            panic!()
        };
        source.chunk = test_resolver
            .lease_record(&run, &key, std::sync::Arc::new(probe))
            .unwrap();
        assert!(conditioned.validate().is_err(), "history={history}");
    }
    // The original Same complete input comparison still rejects V versus Q.
    let mut same_input = request.clone();
    same_input.condition = ComparisonCondition::SameInput;
    let accounting = std::sync::Arc::new(crate::policy::prepared::Accounting::default());
    let owner = crate::value::prepared::RequestOwner::new(same_input, &accounting).unwrap();
    let crate::value::prepared::Completion::Grid(same_grid) = crate::value::prepared::execute(
        &crate::value::prepared::Work::Grid(owner),
        &accounting,
        || false,
        &mut |_, _, _| panic!(),
    )
    .unwrap()
    .unwrap() else {
        panic!()
    };
    assert!(same_grid.reference_error.is_some());
    assert!(same_grid.reference_values.is_empty());
    assert_eq!(same_grid.values, vec![-2., -3.]);
}

#[cfg(feature = "screenshots")]
#[test]
fn recorded_state_value_reference_2d_admits_only_current_population_and_shared_metadata() {
    let inputs = (0..169)
        .flat_map(|i| [(i % 13) as f32 / 12., (i / 13) as f32 / 12., 0., 0., 0.375])
        .collect();
    let (_dir, run) = crate::reference::tests::state_value_reference_recording_with_inputs(
        inputs, 43_264, 16_384,
    );
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    app.value_view.capture_configured = true;
    app.value_view.reference_enabled = true;
    app.value_view.condition = Some(rl::diagnostics::value::ComparisonCondition::ControllerValues);
    app.value_view.display = crate::policy::view::Display::Reference;
    app.comparison.reset_reference_to(&run).unwrap();
    app.comparison.set_reference_role("oracle").unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut paints = 0;
    let mut axes_set = false;
    loop {
        render(&mut app, &ctx, vec![]);
        paints += 1;
        if !axes_set && app.value_view.request.is_some() {
            app.value_view
                .set_axes(
                    crate::value::slice::ValueCoordinate::Observation(0),
                    Some(crate::value::slice::ValueCoordinate::Observation(1)),
                )
                .unwrap();
            axes_set = true;
        }
        assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
        if axes_set && app.value_capture_ready() && app.loader.pending() == 0 {
            break;
        }
        assert!(Instant::now() < deadline, "{:?}", app.value_view.status);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(paints > 1);
    let frame = app.value_frame().unwrap();
    assert_eq!(frame.grid.values.len(), 169);
    assert_eq!(frame.grid.reference_values.len(), 169);
    assert!(frame.grid.reference_values.iter().all(Option::is_some));
    assert_eq!(
        frame.grid.slice.mode,
        crate::policy::slice::SliceMode::TwoDimensional
    );
    assert!(app.value_view.main_rect.is_some());
    assert!(app.value_view.color_limits().is_some());
    assert!(
        app.value_view.owned_state_bytes() + app.value_view.drawing_bytes(frame)
            <= app.value_view.state_reservation_bytes()
    );
    for _ in 0..3 {
        render(&mut app, &ctx, vec![]);
        assert!(app.value_capture_ready());
        assert!(!app.value_view.terminal);
    }
    assert!(app.value_capture_error().is_none());
    // A later drawing rejection retains its complete frame for the interactive
    // view, but must fail capture rather than wait forever or accept that image.
    app.value_view.terminal = true;
    app.value_view.ready = false;
    app.value_view.status = Some("Value drawing exceeds shared view admission".into());
    assert!(!app.value_capture_ready());
    assert_eq!(
        app.value_capture_error().as_deref(),
        app.value_view.status()
    );
    app.value_view.terminal = false;
    assert!(app.value_capture_error().is_none());
    app.value_view.terminal = true;
    app.loader.cache.clear();
    assert!(app.loader.request(&run, 0).unwrap().is_none());
    assert!(app.value_capture_error().is_none());
    while app.loader.pending() > 0 {
        app.loader.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(app.value_capture_error().is_some());
    let mut request = app.value_view.request.clone().unwrap();
    let crate::value::prepared::ValuePin::Probe { row, .. } = &mut request.pin else {
        panic!()
    };
    *row = 1;
    app.interactive.submit_value(request).unwrap();
    assert!(app.interactive.value_pending());
    assert!(app.value_capture_error().is_none());
}

#[test]
fn decision_loss_value_route_resolves_exact_policy_across_frames_and_withdraws_missing_oracle() {
    decision_loss_value_route(true, false, false, false);
}
#[test]
fn decision_loss_value_route_sparse_critic_does_not_constrain_policy_timeline() {
    decision_loss_value_route(false, false, false, false);
}
#[test]
fn decision_loss_value_route_169_row_2d_population_and_trajectory_inspector() {
    decision_loss_value_route(false, true, false, false);
}
#[test]
fn decision_loss_optional_corrupt_actor_does_not_disable_ordinary_value() {
    decision_loss_value_route(false, false, true, false);
}
#[cfg(feature = "screenshots")]
#[test]
fn decision_loss_capture_cold_start_off_probe_trajectory_admits_without_critic_prediction() {
    decision_loss_value_route(false, true, false, true);
}
fn decision_loss_value_route(
    measurements_at_ten: bool,
    grid: bool,
    corrupt_actor: bool,
    cold: bool,
) {
    use crate::value::{prepared::ValuePin, view::CriticChoice};
    use rl::diagnostics::value::{ValueCriticKind, ValueOutput};
    let (_dir, run) =
        crate::reference::tests::decision_tests::fixture_population(measurements_at_ten, grid);
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
    let r = crate::reference::tests::decision_tests::request(&run, &mut resolver, 2, 6);
    if corrupt_actor {
        std::fs::write(
            run.root().join(&run.recording().entries()[6].file),
            b"corrupt actor",
        )
        .unwrap();
    }
    app.selection = Selection {
        revision: 0,
        model_role: "online".into(),
        input: SelectedInput::ProbeRow {
            set: "online".into(),
            row: 0,
            sequence: 0,
        },
    };
    app.workspace_defaults = None;
    app.value_view.config = Some((*r.slice).clone());
    app.value_view.pin = Some(r.pin.clone());
    app.value_view.anchor_key = Some((run.key(0).unwrap(), 0));
    app.value_view.choice = Some(CriticChoice {
        name: "q".into(),
        twin: None,
        kind: ValueCriticKind::DiscreteQ,
    });
    app.value_view.output = Some(ValueOutput::ExpectedReturn);
    app.value_view.reference_enabled = true;
    app.value_view.dirty = true;
    #[cfg(feature = "screenshots")]
    {
        app.value_view.capture_configured = true;
    }
    app.comparison.reset_reference_to(&run).unwrap();
    app.comparison.set_reference_role("oracle").unwrap();
    #[cfg(feature = "screenshots")]
    if cold {
        app.value_view.capture_decision_loss = true;
        app.selection.revision = 10;
        let sequence = run
            .recording()
            .entries()
            .iter()
            .find(|e| e.kind == RecordKind::Trajectory)
            .unwrap()
            .sequence;
        crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep { sequence, step: 0 })
            .apply(&run, &mut app.selection)
            .unwrap();
        app.value_view.anchor_key = None;
        let deadline = Instant::now() + Duration::from_secs(15);
        let mut paints = 0;
        loop {
            render(&mut app, &ctx, vec![]);
            paints += 1;
            assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
            if app.value_view.ready() && app.value_view.output == Some(ValueOutput::DecisionLoss) {
                break;
            }
            assert!(Instant::now() < deadline, "{:?}", app.value_view.status);
            assert!(!app.value_capture_ready());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(paints > 1);
        let f = app.value_frame().unwrap();
        assert_eq!(f.grid.output(), &ValueOutput::DecisionLoss);
        assert!(f.grid.values.is_empty());
        assert!((f.selected.as_ref().unwrap().value - 1.).abs() < 1e-6);
        assert_eq!(
            f.selected
                .as_ref()
                .unwrap()
                .current_policy()
                .unwrap()
                .identity()
                .revision,
            10
        );
        assert_eq!(
            f.selected
                .as_ref()
                .unwrap()
                .recorded_key
                .as_ref()
                .unwrap()
                .sequence,
            sequence
        );
        assert!(app.value_view.inspector_rect.is_some());
        assert!(app.value_capture_ready());
        return;
    }
    #[cfg(not(feature = "screenshots"))]
    assert!(!cold);
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut paints = 0;
    loop {
        render(&mut app, &ctx, vec![]);
        paints += 1;
        assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
        if app.value_view.ready()
            && app
                .value_view
                .request
                .as_ref()
                .is_some_and(|r| corrupt_actor || r.decision_loss_available())
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "status={:?} output={:?} selection={} request={:?} frame={:?} error={:?}",
            app.value_view.status,
            app.value_view.output,
            app.selection.revision,
            app.value_view
                .request
                .as_ref()
                .map(|r| (&r.output, r.requested_revision)),
            app.value_frame()
                .map(|f| (f.grid.output(), f.requested_revision)),
            app.notice()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(paints > 1);
    if corrupt_actor {
        assert_eq!(
            app.value_frame().unwrap().grid.output(),
            &ValueOutput::ExpectedReturn
        );
        assert!(
            app.value_view
                .request
                .as_ref()
                .unwrap()
                .current_policy
                .is_none()
        );
        assert!(
            !app.value_view
                .request
                .as_ref()
                .unwrap()
                .decision_loss_available()
        );
        let cached = app.value_view.unavailable_policy.clone().unwrap();
        for _ in 0..5 {
            render(&mut app, &ctx, vec![]);
            assert!(app.value_view.ready());
            assert_eq!(app.value_view.unavailable_policy.as_ref(), Some(&cached));
        }
        return;
    }
    let oracle_info = app
        .value_view
        .request
        .as_ref()
        .unwrap()
        .reference_value
        .as_ref()
        .unwrap()
        .reference_info
        .clone();
    if grid {
        app.value_view
            .set_axes(
                crate::value::slice::ValueCoordinate::Observation(0),
                Some(crate::value::slice::ValueCoordinate::Observation(1)),
            )
            .unwrap();
    }
    app.value_view.output = Some(ValueOutput::DecisionLoss);
    app.value_view.dirty = true;
    app.value_view.ready = false;
    for (k, expected) in [(0, 2.25), (10, 0.75)] {
        app.select_revision(k).unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            render(&mut app, &ctx, vec![]);
            assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
            if app.value_view.ready()
                && app.value_frame().is_some_and(|f| {
                    f.grid.output() == &ValueOutput::DecisionLoss && f.requested_revision == k
                })
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "status={:?} output={:?} selection={} request={:?} frame={:?} error={:?}",
                app.value_view.status,
                app.value_view.output,
                app.selection.revision,
                app.value_view
                    .request
                    .as_ref()
                    .map(|r| (&r.output, r.requested_revision)),
                app.value_frame()
                    .map(|f| (f.grid.output(), f.requested_revision)),
                app.notice()
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(std::sync::Arc::ptr_eq(
            &oracle_info,
            &app.value_view
                .request
                .as_ref()
                .unwrap()
                .reference_value
                .as_ref()
                .unwrap()
                .reference_info
        ));
        let f = app.value_frame().unwrap();
        assert!((f.selected.as_ref().unwrap().value - expected).abs() < 1e-6);
        assert_eq!(
            f.selected
                .as_ref()
                .unwrap()
                .current_policy()
                .unwrap()
                .identity()
                .revision,
            k
        );
        assert!(f.grid.reference_values.is_empty());
        assert_eq!(f.grid.values.len(), if grid { 169 } else { 2 });
        assert!(app.value_view.color_limits().unwrap()[1] < 4.);
    }
    let grid = app.value_frame().unwrap().grid.clone();
    app.value_view.pin = Some(ValuePin::State(Observation::dense(
        vec![0., 0., 0., 0., 0.5],
        Some(vec![true, true]),
    )));
    app.value_view.dirty = true;
    app.value_view.ready = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.value_view.ready() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "status={:?} output={:?} selection={} request={:?} frame={:?} error={:?}",
            app.value_view.status,
            app.value_view.output,
            app.selection.revision,
            app.value_view
                .request
                .as_ref()
                .map(|r| (&r.output, r.requested_revision)),
            app.value_frame()
                .map(|f| (f.grid.output(), f.requested_revision)),
            app.notice()
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(std::sync::Arc::ptr_eq(
        &grid,
        &app.value_frame().unwrap().grid
    ));
    assert!((app.value_frame().unwrap().selected.as_ref().unwrap().value - 1.).abs() < 1e-6);
    let trajectory = run
        .recording()
        .entries()
        .iter()
        .find(|e| e.kind == RecordKind::Trajectory)
        .unwrap()
        .sequence;
    crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
        sequence: trajectory,
        step: 0,
    })
    .apply(&run, &mut app.selection)
    .unwrap();
    app.value_view.dirty = true;
    app.value_view.ready = false;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.value_view.ready() {
            break;
        }
        assert!(!app.value_view.terminal, "{:?}", app.value_view.status);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(app.value_frame().unwrap().grid.values.is_empty());
    assert!(app.value_view.inspector_rect.is_some());
    assert!((app.value_frame().unwrap().selected.as_ref().unwrap().value - 1.).abs() < 1e-6);
    #[cfg(feature = "screenshots")]
    assert!(app.value_capture_ready());
    app.value_view.reference_enabled = false;
    app.value_view.dirty = true;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.value_view.terminal {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(app.value_view.output, Some(ValueOutput::DecisionLoss));
    assert!(!app.value_view.ready());
    assert!(app.value_frame().is_none());
    assert!(
        app.value_view
            .status
            .as_ref()
            .unwrap()
            .contains("Decision loss")
    );
}

#[test]
fn decision_loss_interactive_cold_start_off_probe_offers_output_without_critic_prediction() {
    decision_loss_interactive_cold_start(Some("oracle"), false);
}
#[test]
fn decision_loss_interactive_cold_start_absent_or_unverified_oracle_has_no_output() {
    decision_loss_interactive_cold_start(None, false);
    decision_loss_interactive_cold_start(Some("online"), false);
    decision_loss_interactive_cold_start(Some("oracle"), true);
}
fn decision_loss_interactive_cold_start(reference_role: Option<&str>, mismatched_oracle: bool) {
    use crate::value::view::CriticChoice;
    use rl::diagnostics::value::{ValueCriticKind, ValueOutput};
    let (_dir, run) = crate::reference::tests::decision_tests::fixture_population(true, true);
    let (_ref_dir, reference_run) =
        crate::reference::tests::decision_tests::fixture_population_context(
            true,
            true,
            mismatched_oracle,
        );
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    app.select_view(View::Value);
    let sequence = run
        .recording()
        .entries()
        .iter()
        .find(|e| e.kind == RecordKind::Trajectory)
        .unwrap()
        .sequence;
    app.selection = Selection {
        revision: 10,
        model_role: "online".into(),
        input: SelectedInput::TrajectoryStep { sequence, step: 0 },
    };
    app.workspace_defaults = None;
    app.value_view.choice = Some(CriticChoice {
        name: "q".into(),
        twin: None,
        kind: ValueCriticKind::DiscreteQ,
    });
    app.value_view.output = Some(ValueOutput::ExpectedReturn);
    app.value_view.reference_enabled = reference_role.is_some();
    app.value_view.dirty = true;
    #[cfg(feature = "screenshots")]
    {
        app.value_view.capture_configured = true;
        app.value_view.capture_decision_loss = false;
    }
    app.comparison.reset_reference_to(&reference_run).unwrap();
    if let Some(role) = reference_role {
        app.comparison.set_reference_role(role).unwrap();
    }
    let admitted = reference_role == Some("oracle") && !mismatched_oracle;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        let available = app
            .value_view
            .request
            .as_ref()
            .is_some_and(|r| r.decision_loss_available());
        if app.value_view.terminal && (available || !admitted) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "status={:?}, capability={available}",
            app.value_view.status
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(app.value_view.output, Some(ValueOutput::ExpectedReturn));
    assert!(!app.value_view.ready());
    assert!(app.value_frame().is_none());
    if !admitted {
        let deadline = Instant::now() + Duration::from_secs(15);
        while app.loader.pending() > 0 || app.interactive.pending() {
            render(&mut app, &ctx, vec![]);
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        let activity = |app: &Laboratory| {
            let m = app.value_metrics();
            (
                app.loader.completed_loads,
                m.grid_batches,
                m.point_batches,
                m.restores,
            )
        };
        let settled = activity(&app);
        for _ in 0..5 {
            render(&mut app, &ctx, vec![]);
        }
        assert_eq!(activity(&app), settled);
        assert!(
            !app.value_view
                .request
                .as_ref()
                .is_some_and(|r| r.decision_loss_available())
        );
        return;
    }
    assert!(
        app.value_view
            .status
            .as_deref()
            .unwrap()
            .contains("unavailable")
    );
    let request = app.value_view.request.as_ref().unwrap();
    assert!(request.capability_only);
    let mut ordinary = request.clone();
    ordinary.capability_only = false;
    assert!(!request.same_grid(&ordinary));
    let metrics = app.value_metrics();
    assert_eq!(
        (
            metrics.grid_batches,
            metrics.point_batches,
            metrics.restores
        ),
        (0, 0, 0)
    );
    click(&mut app, &ctx, "Expected return");
    click(&mut app, &ctx, "Decision loss");
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        render(&mut app, &ctx, vec![]);
        if app.value_view.ready() && app.value_view.output == Some(ValueOutput::DecisionLoss) {
            break;
        }
        assert!(Instant::now() < deadline, "{:?}", app.value_view.status);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!app.value_view.terminal);
    let f = app.value_frame().unwrap();
    assert_eq!(f.grid.output(), &ValueOutput::DecisionLoss);
    assert!(f.grid.values.is_empty());
    assert!((f.selected.as_ref().unwrap().value - 1.).abs() < 1e-6);
    assert_eq!(
        f.selected
            .as_ref()
            .unwrap()
            .current_policy()
            .unwrap()
            .identity()
            .revision,
        10
    );
    assert_eq!(
        f.selected
            .as_ref()
            .unwrap()
            .recorded_key
            .as_ref()
            .unwrap()
            .sequence,
        sequence
    );
    assert!(app.value_view.inspector_rect.is_some());
}
