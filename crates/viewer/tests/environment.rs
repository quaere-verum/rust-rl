//! Environment workspace regression tests; no built-in environment is required.
#[path = "support/policy_ui.rs"]
mod support;
use rl_viewer::ui::View;

#[test]
fn environment_startup_hook_selects_its_own_workspace() {
    assert_eq!(
        format!("{:?}", View::from_startup_hook("environment")),
        "Environment"
    );
}

use eframe::egui;
use rl::{
    core::action::ActionSpace,
    diagnostics::{render::*, schema::*, storage::*},
};
use rl_viewer::{
    data::{CacheKey, Loader, RunIndex, SelectedInput},
    dynamics::source::{CatalogLimits, TrajectoryCatalog},
    environment::{SceneIdentity, SceneTransform, SceneWorker},
    ui::Laboratory,
};
use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

fn scene() -> RenderScene {
    RenderScene {
        version: 1,
        bounds: RenderBounds {
            min: [0., 0.],
            max: [2., 1.],
        },
        background: [0, 0, 0, 255],
        items: vec![
            RenderItem {
                layer: 2,
                primitive: RenderPrimitive::Text {
                    position: [1., 0.5],
                    text: "custom scene".into(),
                    color: [255; 4],
                    size: 16.,
                },
            },
            RenderItem {
                layer: -1,
                primitive: RenderPrimitive::Circle {
                    center: [1., 0.5],
                    radius: 0.2,
                    fill: Some([200, 50, 50, 255]),
                    stroke: None,
                },
            },
        ],
    }
}
fn trajectory(revision: u64, rendered: bool) -> Chunk {
    let mut context = RecordContext::new("environment-test", revision);
    context.trajectory = Some(format!("episode-{revision}"));
    let mut chunk = Chunk {
        context,
        record: Record::Trajectory(TrajectoryRecord {
            id: format!("episode-{revision}"),
            environment: "arbitrary-custom".into(),
            observations: "s".into(),
            actions: "a".into(),
            rewards: "r".into(),
            terminated: "term".into(),
            truncated: "trunc".into(),
            simulation_times: None,
            action_masks: None,
            frames: BTreeMap::new(),
            render: None,
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "custom-policy".into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "test seeds".into(),
                truncation_handling: "true termination".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "s".into(),
                Array::f32(vec![3, 1], vec![0., 1., 2.]).unwrap(),
            ),
            ("a".into(), Array::u64(vec![2], vec![0, 0]).unwrap()),
            ("r".into(), Array::f32(vec![2], vec![0., 1.]).unwrap()),
            ("term".into(), Array::u8(vec![2], vec![0, 1]).unwrap()),
            ("trunc".into(), Array::u8(vec![2], vec![0, 0]).unwrap()),
        ]),
    };
    if rendered {
        let bytes = serde_json::to_vec(&scene()).unwrap();
        let mut payload = Vec::new();
        let mut offsets = vec![0];
        for _ in 0..3 {
            payload.extend(&bytes);
            offsets.push(payload.len() as u64);
        }
        chunk.arrays.insert(
            "render-data".into(),
            Array::u8(vec![payload.len()], payload).unwrap(),
        );
        chunk.arrays.insert(
            "render-offsets".into(),
            Array::u64(vec![4], offsets).unwrap(),
        );
        if let Record::Trajectory(t) = &mut chunk.record {
            t.render = Some(RenderTrack {
                version: 1,
                offsets: "render-offsets".into(),
                data: "render-data".into(),
            });
        }
    }
    chunk
}
fn fixture(rendered: bool) -> (tempfile::TempDir, RunIndex) {
    fixture_with_episodes(rendered, false)
}
fn fixture_with_episodes(rendered: bool, second: bool) -> (tempfile::TempDir, RunIndex) {
    fixture_with_index_gap(rendered, second, true)
}
fn fixture_with_index_gap(
    rendered: bool,
    second: bool,
    indexed_gap: bool,
) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        RunManifest::new(
            "environment-test",
            "custom",
            "custom",
            1,
            ActionSpace::Discrete { count: 1 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    writer.append(&trajectory(1, rendered)).unwrap();
    if second {
        let mut other = trajectory(1, rendered);
        other.context.trajectory = Some("episode-B".into());
        if let Record::Trajectory(record) = &mut other.record {
            record.id = "episode-B".into();
        }
        writer.append(&other).unwrap();
    }
    if indexed_gap {
        writer
            .append(&Chunk {
                context: RecordContext::new("environment-test", 2),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("return".into(), 1.)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
    }
    writer.append(&trajectory(3, rendered)).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    (dir, run)
}
fn identity(step: usize) -> SceneIdentity {
    SceneIdentity {
        session: 1,
        key: CacheKey {
            root: Arc::new(std::path::PathBuf::from("custom")),
            sequence: 0,
        },
        revision: 1,
        role: "online".into(),
        step,
    }
}
#[test]
fn same_episode_keeps_published_scene_and_readouts_until_next_is_ready() {
    let mut chunk = trajectory(1, true);
    if let Record::Trajectory(record) = &mut chunk.record {
        record.frames.insert("position".into(), "s".into());
    }
    let chunk = Arc::new(chunk);
    let readouts = vec![rl_viewer::environment::ReadoutSpec {
        key: "s".into(),
        component: 0,
        label: "Position".into(),
    }];
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker
        .submit(identity(0), &chunk, readouts.clone())
        .unwrap();
    wait(&mut worker);
    assert_eq!(worker.frame().unwrap().readouts[0].1, 0.);
    worker.submit(identity(1), &chunk, readouts).unwrap();
    let displayed = worker.frame().expect("previous scene must remain visible");
    assert_eq!(displayed.identity.step, 0);
    assert_eq!(displayed.readouts[0].1, 0.);
    wait(&mut worker);
    let displayed = worker.frame().unwrap();
    assert_eq!(displayed.identity.step, 1);
    assert_eq!(displayed.readouts[0].1, 1.);
    worker.submit(identity(3), &chunk, vec![]).unwrap();
    assert!(worker.frame().is_some());
    wait(&mut worker);
    assert!(worker.frame().is_none());
    assert!(worker.error().is_some());
}
fn wait(worker: &mut SceneWorker) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline, "scene worker stalled");
        std::thread::sleep(Duration::from_millis(1));
    }
    worker.poll();
}
#[test]
fn scene_transform_preserves_aspect_y_and_letterboxing() {
    let t = SceneTransform::new(
        &scene().bounds,
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200., 200.)),
    )
    .unwrap();
    assert_eq!(t.point([0., 0.]), egui::pos2(0., 150.));
    assert_eq!(t.point([2., 1.]), egui::pos2(200., 50.));
    assert_eq!(t.radius(0.25), 25.);
}
#[test]
fn selected_scene_is_decoded_once_and_terminal_state_is_supported() {
    let chunk = Arc::new(trajectory(1, true));
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker.submit(identity(2), &chunk, vec![]).unwrap();
    wait(&mut worker);
    assert_eq!(worker.frame().unwrap().identity.step, 2);
    assert_eq!(
        worker.frame().unwrap().scene.as_ref().unwrap().items[0].layer,
        -1
    );
    let decodes = worker.metrics().decodes;
    for _ in 0..20 {
        worker.submit(identity(2), &chunk, vec![]).unwrap();
        worker.poll();
    }
    assert_eq!(worker.metrics().decodes, decodes);
    let held = worker.frame().unwrap().clone();
    worker.retire();
    assert!(worker.metrics().owned_bytes > 0);
    drop(held);
    assert_eq!(
        worker.metrics().owned_bytes,
        rl_viewer::environment::CONTROL_BYTES
    );
}
#[test]
fn same_key_raw_replacement_reprepares_and_withdraws_the_previous_owner_scene() {
    let original = Arc::new(trajectory(1, true));
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker.submit(identity(0), &original, vec![]).unwrap();
    wait(&mut worker);
    let decodes = worker.metrics().decodes;
    let mut changed = trajectory(1, true);
    let mut replacement_scene = scene();
    replacement_scene.background = [42, 17, 23, 255];
    let bytes = serde_json::to_vec(&replacement_scene).unwrap();
    changed.arrays.insert(
        "render-data".into(),
        Array::u8(vec![bytes.len() * 3], bytes.repeat(3)).unwrap(),
    );
    changed.arrays.insert(
        "render-offsets".into(),
        Array::u64(vec![4], (0..4).map(|i| (i * bytes.len()) as u64).collect()).unwrap(),
    );
    let replacement = Arc::new(changed);
    worker.submit(identity(0), &replacement, vec![]).unwrap();
    assert!(
        worker.frame().is_none(),
        "a different raw owner must withdraw the old scene immediately"
    );
    wait(&mut worker);
    assert_eq!(worker.metrics().decodes, decodes + 1);
    assert_eq!(
        worker.frame().unwrap().scene.as_ref().unwrap().background,
        replacement_scene.background
    );

    // A changed nonselected offset must also invalidate the old proof.
    let mut bad = trajectory(1, true);
    let bytes = serde_json::to_vec(&scene()).unwrap().repeat(2);
    let end = bytes.len() as u64;
    bad.arrays.insert(
        "render-data".into(),
        Array::u8(vec![bytes.len()], bytes).unwrap(),
    );
    bad.arrays.insert(
        "render-offsets".into(),
        Array::u64(vec![4], vec![0, end / 2, end, end]).unwrap(),
    );
    worker.submit(identity(1), &Arc::new(bad), vec![]).unwrap();
    assert!(worker.frame().is_none());
    wait(&mut worker);
    assert!(worker.frame().is_none());
    assert!(worker.error().unwrap().contains("invalid render offset"));
}

#[test]
fn successful_preparation_and_idle_proof_do_not_retain_raw_payload() {
    let chunk = Arc::new(trajectory(1, true));
    let weak = Arc::downgrade(&chunk);
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker.submit(identity(0), &chunk, vec![]).unwrap();
    wait(&mut worker);
    drop(chunk);
    assert!(weak.upgrade().is_none());
    assert!(worker.frame().is_some());
    worker.retire();
    assert_eq!(
        worker.metrics().owned_bytes,
        rl_viewer::environment::CONTROL_BYTES
    );
}
#[test]
fn wrong_identity_and_invalid_step_never_publish_while_absent_scenes_allow_numerical_playback() {
    let mut worker = SceneWorker::new(|| {}).unwrap();
    let chunk = Arc::new(trajectory(1, true));
    let mut wrong = identity(0);
    wrong.revision = 2;
    assert!(worker.submit(wrong, &chunk, vec![]).is_err());
    worker.submit(identity(3), &chunk, vec![]).unwrap();
    wait(&mut worker);
    assert!(worker.frame().is_none());
    assert!(worker.error().is_some());
    worker.retire();
    worker
        .submit(identity(0), &Arc::new(trajectory(1, false)), vec![])
        .unwrap();
    wait(&mut worker);
    let frame = worker.frame().expect("numerical playback without a scene");
    assert!(frame.scene.is_none());
    assert_eq!(frame.identity.step, 0);
}
#[test]
fn rapid_scrub_and_retirement_cannot_publish_stale_scenes() {
    let mut worker = SceneWorker::new(|| {}).unwrap();
    let chunk = Arc::new(trajectory(1, true));
    for step in 0..100 {
        worker.submit(identity(step % 3), &chunk, vec![]).unwrap();
    }
    worker.submit(identity(2), &chunk, vec![]).unwrap();
    wait(&mut worker);
    assert_eq!(worker.frame().unwrap().identity.step, 2);
    worker.retire();
    std::thread::sleep(Duration::from_millis(20));
    worker.poll();
    assert!(worker.frame().is_none());
    assert!(!worker.pending());
    assert!(worker.metrics().peak_bytes <= rl_viewer::environment::MAX_OWNER_BYTES);
}
#[test]
fn catalog_render_presence_comes_from_validated_records() {
    for rendered in [false, true] {
        let (_dir, run) = fixture(rendered);
        let mut catalog = TrajectoryCatalog::new(&run, "online", CatalogLimits::default()).unwrap();
        let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            catalog.poll(&run, &mut loader).unwrap();
            if !catalog.pending() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(catalog.entries().len(), 2);
        assert!(
            catalog
                .entries()
                .iter()
                .all(|e| e.has_render_scenes == rendered)
        );
    }
}
fn paint(ctx: &egui::Context, app: &mut Laboratory) {
    let mut output = ctx.run_ui(egui::RawInput::default(), |ui| app.show(ui));
    output.textures_delta.clear();
}
#[test]
fn environment_snaps_missing_checkpoint_and_shares_terminal_selection() {
    let (dir, _run) = fixture(true);
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, dir.path().to_string_lossy().into()).unwrap();
    app.select_view(View::Environment);
    app.select_revision(2).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !app.environment_available() {
        paint(&ctx, &mut app);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    for _ in 0..5 {
        paint(&ctx, &mut app);
    }
    assert_eq!(app.selection().revision, 1);
    app.select_revision(3).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !app.environment_ready() {
        paint(&ctx, &mut app);
        assert!(Instant::now() < deadline, "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(2));
    }
    let sequence = match app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("not shared episode input"),
    };
    app.select_environment_step(sequence, 2).unwrap();
    paint(&ctx, &mut app);
    assert_eq!(
        app.selection().input,
        SelectedInput::TrajectoryStep { sequence, step: 2 }
    );
    app.select_view(View::PolicyMap);
    assert_eq!(
        app.selection().input,
        SelectedInput::TrajectoryStep { sequence, step: 2 }
    );
}

#[test]
fn scene_decoder_rejects_corrupt_and_oversized_recorded_frames() {
    let mut worker = SceneWorker::new(|| {}).unwrap();
    let mut malformed = trajectory(1, true);
    if let Record::Trajectory(t) = &mut malformed.record {
        t.render.as_mut().unwrap().version = 999;
    }
    worker
        .submit(identity(0), &Arc::new(malformed), vec![])
        .unwrap();
    wait(&mut worker);
    assert!(worker.frame().is_none());
    assert!(worker.error().is_some());
    worker.retire();
    let mut oversized = trajectory(1, true);
    let data = vec![b' '; 65 * 1024 * 3];
    oversized.arrays.insert(
        "render-data".into(),
        Array::u8(vec![data.len()], data).unwrap(),
    );
    oversized.arrays.insert(
        "render-offsets".into(),
        Array::u64(vec![4], vec![0, 65 * 1024, 130 * 1024, 195 * 1024]).unwrap(),
    );
    worker
        .submit(identity(0), &Arc::new(oversized), vec![])
        .unwrap();
    wait(&mut worker);
    assert!(worker.frame().is_none());
    assert!(worker.error().is_some());
}
#[test]
fn ordinary_recording_has_numerical_environment_playback() {
    let (dir, _run) = fixture(false);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    assert!(h.app.environment_available());
    h.click("Observations");
    assert!(h.text.iter().any(|(text, _)| text == "Public observations"));
    h.click("Observations");
    assert!(h.text.iter().any(|(text, _)| text == "Episode rewards"));
    assert!(h.text.iter().any(|(text, _)| text == "Decision step"));
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text.contains("Executed action"))
    );
    h.app.select_environment_step(0, 2).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text.contains("Environment terminated"))
    );
    assert!(h.text.iter().any(|(text, _)| text.contains("1.000")));
}

#[test]
fn numerical_episode_plots_recorded_frame_channels_on_physical_time_and_selects_exact_state() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = RunManifest::new(
        "environment-test",
        "custom",
        "custom",
        1,
        ActionSpace::Discrete { count: 1 },
        0.9,
    )
    .unwrap();
    let mut chunk = trajectory(1, false);
    chunk.arrays.insert(
        "height".into(),
        Array::f64(vec![3, 1], vec![10., 5., 0.]).unwrap(),
    );
    chunk.arrays.insert(
        "time".into(),
        Array::f32(vec![3], vec![0., 0.04, 0.10]).unwrap(),
    );
    let Record::Trajectory(record) = &mut chunk.record else {
        unreachable!()
    };
    record.frames.insert("altitude".into(), "height".into());
    record.simulation_times = Some("time".into());
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..3 {
        h.frame(vec![]);
    }
    assert!(h.has("Episode telemetry"));
    assert!(h.has("altitude"));
    assert!(h.has("Simulation time"));
    let sequence = match h.app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("expected episode"),
    };
    let plot_id = egui::Id::unique((
        "environment-frame",
        run.root().as_ref(),
        sequence,
        "altitude",
        0usize,
    ));
    let transform = egui_plot::PlotMemory::load(&h.ctx, plot_id)
        .unwrap()
        .transform();
    let x = transform
        .position_from_point(&egui_plot::PlotPoint::new(0.04, 5.))
        .x;
    h.click_at(egui::pos2(x, transform.frame().center().y));
    wait_ui(&mut h, |app| {
        app.selection().input == SelectedInput::TrajectoryStep { sequence, step: 1 }
            && app.environment_ready()
    });
}

#[test]
fn narrow_numerical_episode_keeps_frame_telemetry_available() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = RunManifest::new(
        "environment-test",
        "custom",
        "custom",
        1,
        ActionSpace::Discrete { count: 1 },
        0.9,
    )
    .unwrap();
    let mut chunk = trajectory(1, false);
    chunk.arrays.insert(
        "height".into(),
        Array::f64(vec![3, 1], vec![10., 5., 0.]).unwrap(),
    );
    let Record::Trajectory(record) = &mut chunk.record else {
        unreachable!()
    };
    record.frames.insert("altitude".into(), "height".into());
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(720., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..3 {
        h.frame(vec![]);
    }
    assert!(h.has("Episode telemetry"));
    assert!(h.has("altitude"));
    h.frame(vec![
        egui::Event::PointerMoved(egui::pos2(680., 700.)),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0., -240.),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Unknown,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    for _ in 0..15 {
        h.frame(vec![]);
    }
    let telemetry = h
        .text
        .iter()
        .find(|(label, _)| label == "Episode telemetry")
        .unwrap()
        .1;
    assert!(
        telemetry.top() >= 0. && telemetry.bottom() < h.size.y,
        "telemetry heading must scroll into view"
    );
    let sequence = match h.app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("expected episode"),
    };
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let plot_id = egui::Id::unique((
        "environment-frame",
        run.root().as_ref(),
        sequence,
        "altitude",
        0usize,
    ));
    let transform = egui_plot::PlotMemory::load(&h.ctx, plot_id)
        .unwrap()
        .transform();
    let x = transform
        .position_from_point(&egui_plot::PlotPoint::new(1., 5.))
        .x;
    let point = egui::pos2(x, transform.frame().center().y);
    assert!(
        point.y >= 0. && point.y < h.size.y,
        "selected telemetry point must be visible"
    );
    h.click_at(point);
    wait_ui(&mut h, |app| {
        app.selection().input == SelectedInput::TrajectoryStep { sequence, step: 1 }
            && app.environment_ready()
    });
}

#[test]
fn numerical_playback_reads_the_final_physical_clock_and_rewards_without_a_scene() {
    let mut chunk = trajectory(1, false);
    chunk.arrays.insert(
        "trajectory/physical_durations.v1".into(),
        Array::u64(vec![2], vec![2, 3]).unwrap(),
    );
    chunk.arrays.insert(
        "trajectory/discounted_step_rewards.v1".into(),
        Array::f64(vec![2], vec![0., 1.]).unwrap(),
    );
    chunk.arrays.insert(
        "time".into(),
        Array::f64(vec![3], vec![0., 0.04, 0.1]).unwrap(),
    );
    let Record::Trajectory(record) = &mut chunk.record else {
        unreachable!()
    };
    record.simulation_times = Some("time".into());
    chunk.validate().unwrap();

    let chunk = Arc::new(chunk);
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker.submit(identity(2), &chunk, vec![]).unwrap();
    wait(&mut worker);
    let frame = worker.frame().expect("numerical trajectory frame");
    assert!(frame.scene.is_none());
    assert_eq!(frame.playback.observations.values, [2.]);
    assert!(frame.playback.executed_action.is_none());
    assert!(frame.playback.reward.is_none());
    assert_eq!(frame.playback.progress.observed_reward, 1.);
    assert_eq!(frame.playback.progress.primitive_steps, 5);
    assert_eq!(frame.playback.progress.simulation_time, Some(0.1));
    assert!(matches!(
        frame.playback.end,
        Some(rl_viewer::environment::EpisodeEnd::Terminated)
    ));
}
#[test]
fn decoded_frames_are_charged_until_all_external_owners_release() {
    let mut worker = SceneWorker::new(|| {}).unwrap();
    let chunk = Arc::new(trajectory(1, true));
    let mut held = Vec::new();
    for session in 0..8 {
        let mut id = identity(0);
        id.session = session;
        worker.submit(id, &chunk, vec![]).unwrap();
        wait(&mut worker);
        held.push(worker.frame().unwrap().clone());
    }
    worker.retire();
    assert_eq!(
        worker.metrics().owned_bytes,
        rl_viewer::environment::CONTROL_BYTES
            + held.len() * rl_viewer::environment::MAX_SCENE_BYTES
            + rl_viewer::environment::MAX_TELEMETRY_BYTES
    );
    held.clear();
    assert_eq!(
        worker.metrics().owned_bytes,
        rl_viewer::environment::CONTROL_BYTES
    );
}

#[test]
fn excessive_readout_capacity_is_rejected_before_pinning() {
    let mut worker = SceneWorker::new(|| {}).unwrap();
    let readouts = Vec::with_capacity(100_000);
    assert!(
        worker
            .submit(identity(0), &Arc::new(trajectory(1, true)), readouts)
            .is_err()
    );
    assert_eq!(
        worker.metrics().owned_bytes,
        rl_viewer::environment::CONTROL_BYTES
    );
}
#[test]
fn huge_projection_is_unavailable_instead_of_changing_line_slope() {
    let mut input = scene();
    input.items.push(RenderItem {
        layer: 0,
        primitive: RenderPrimitive::Line {
            from: [-1e9, 1e12],
            to: [0., 0.],
            stroke: RenderStroke {
                color: [255; 4],
                width: 1.,
            },
        },
    });
    assert!(
        SceneTransform::for_scene(
            &input,
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200., 200.))
        )
        .is_err()
    );
    input.bounds.max = [f64::MIN_POSITIVE / 100., f64::MIN_POSITIVE / 100.];
    assert!(
        SceneTransform::for_scene(
            &input,
            egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200., 200.))
        )
        .is_err()
    );
}
#[test]
fn missing_immutable_file_withdraws_a_cached_scene_asynchronously() {
    let (dir, run) = fixture(true);
    let chunk = Arc::new(run.recording().load(0).unwrap());
    let mut worker = SceneWorker::new(|| {}).unwrap();
    worker.submit(identity(0), &chunk, vec![]).unwrap();
    wait(&mut worker);
    let path = dir
        .path()
        .join("run")
        .join(&run.recording().entries()[0].file);
    std::fs::remove_file(&path).unwrap();
    worker.check_path(path).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while worker.error().is_none() {
        worker.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(worker.frame().is_none());
    assert!(worker.error().unwrap().contains("missing"));
}

fn wait_ui(h: &mut support::Harness, condition: impl Fn(&Laboratory) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        h.frame(vec![]);
        if condition(&h.app) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "Environment did not settle: {:?}",
            h.app.environment_status()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn terminal_environment_state_opens_state_value_panel() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::diagnostics::models::SnapshotBuilder;
    use rl::nn::critics::scalar::Value;

    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest = RunManifest::new(
        "environment-test",
        "custom",
        "custom",
        1,
        space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut builder =
        SnapshotBuilder::new("value", RecordContext::new("environment-test", 1), 1, space).unwrap();
    builder
        .value(
            "value",
            &Value::new(1, &[], &mut ChaCha8Rng::seed_from_u64(1)).unwrap(),
        )
        .unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    let mut episode = trajectory(1, true);
    episode
        .arrays
        .insert("a".into(), Array::f32(vec![2, 1], vec![0., 0.]).unwrap());
    writer.append(&episode).unwrap();
    writer.finish().unwrap();

    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    let SelectedInput::TrajectoryStep { sequence, .. } = h.app.selection().input else {
        panic!("missing episode")
    };
    h.app.select_environment_step(sequence, 2).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    h.app.select_view(View::Value);
    wait_ui(&mut h, |app| app.value_view().ready());
    let point = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.final_observation);
    assert_eq!(point.recorded_row, Some(2));
    assert!(point.action.is_none() && point.observed_return.is_none());
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text.contains("Final recorded observation"))
    );
}

#[test]
fn next_state_keeps_scene_and_selection_until_new_scene_is_ready() {
    let (dir, _) = fixture(true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    let pos = h.find("Next state");
    h.frame(vec![
        egui::Event::PointerMoved(pos),
        support::button(pos, true),
    ]);
    h.frame(vec![support::button(pos, false)]);
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 0
        }
    );
    assert!(h.text.iter().any(|(text, _)| text == "custom scene"));
    wait_ui(&mut h, |app| {
        app.environment_ready()
            && app.selection().input
                == SelectedInput::TrajectoryStep {
                    sequence: 0,
                    step: 1,
                }
    });
    assert!(h.text.iter().any(|(text, _)| text == "custom scene"));
}

#[test]
fn episode_playback_keeps_scene_visible_until_terminal_state() {
    let (dir, _) = fixture(true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    h.click("Play episode");
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut previous = 0;
    loop {
        std::thread::sleep(Duration::from_millis(10));
        h.frame(vec![]);
        assert!(h.text.iter().any(|(text, _)| text == "custom scene"));
        let SelectedInput::TrajectoryStep { step, .. } = h.app.selection().input else {
            panic!("missing trajectory")
        };
        assert!(step >= previous && step <= previous + 1);
        previous = step;
        if step == 2 && h.app.environment_ready() {
            break;
        }
        assert!(Instant::now() < deadline, "playback stalled");
    }
    h.frame(vec![]);
    assert!(h.text.iter().any(|(text, _)| text == "Final state"));
}

#[test]
fn external_step_change_supersedes_pending_manual_step() {
    let (dir, _) = fixture(true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    let pos = h.find("Next state");
    h.frame(vec![
        egui::Event::PointerMoved(pos),
        support::button(pos, true),
    ]);
    h.frame(vec![support::button(pos, false)]);
    h.app.select_environment_step(0, 2).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 2
        }
    );
}

#[test]
fn global_play_skips_checkpoints_without_rendered_episodes() {
    let (dir, _) = fixture(true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    h.click("Play");
    assert!(
        h.app.training_is_playing(),
        "global Play was cancelled by the next paint"
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        h.frame(vec![]);
        assert_ne!(
            h.app.selection().revision,
            2,
            "playback entered a checkpoint without scenes"
        );
        if h.app.selection().revision == 3 && h.app.environment_ready() {
            break;
        }
        assert!(Instant::now() < deadline, "playback stalled");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn environment_navigation_counts_distinct_rendered_checkpoints() {
    let (dir, _) = fixture_with_episodes(true, true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    h.frame(vec![]);
    assert!(h.text.iter().any(|(text, _)| text == "1 / 2"));
    h.click("›");
    wait_ui(&mut h, |app| {
        app.selection().revision == 3 && app.environment_ready()
    });
    h.frame(vec![]);
    assert!(h.text.iter().any(|(text, _)| text == "2 / 2"));
    h.click("‹");
    wait_ui(&mut h, |app| {
        app.selection().revision == 1 && app.environment_ready()
    });
}

#[test]
fn episode_combobox_recovers_from_previous_missing_source_without_retry() {
    let (dir, run) = fixture_with_episodes(true, true);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 0
        }
    );
    let path = dir
        .path()
        .join("run")
        .join(&run.recording().entries()[0].file);
    std::fs::remove_file(path).unwrap();
    // Use the normal live source recheck, including its asynchronous worker result.
    wait_ui(&mut h, |app| {
        app.environment_status()
            .is_some_and(|s| s.contains("missing"))
    });
    h.click("episode-1");
    h.click("episode-B");
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 1,
            step: 0
        }
    );
    wait_ui(&mut h, Laboratory::environment_ready);
    assert!(h.app.environment_status().is_none());
}

#[test]
fn unindexed_checkpoint_snaps_to_rendered_neighbor_after_discovery() {
    let (dir, run) = fixture_with_index_gap(true, false, false);
    assert_eq!(run.revisions(), &[1, 3]);
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(2).unwrap();
    wait_ui(&mut h, Laboratory::environment_available);
    wait_ui(&mut h, Laboratory::environment_ready);
    assert_eq!(h.app.selection().revision, 1);
    h.click("›");
    wait_ui(&mut h, Laboratory::environment_ready);
    assert_eq!(h.app.selection().revision, 3);
}

#[test]
fn benchmark_playback_selects_the_matched_scenario_and_keeps_an_independent_cursor() {
    use rl::diagnostics::{
        reference::{
            ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceInformation, ReferenceKind,
            ReferenceLifetime, ReferenceObjective, ReferenceQuality, ReferenceSpec,
            ReferenceTarget,
        },
        value::ValueConventions,
        visualization::{
            ActionDescriptor, ClockDescriptor, DiscreteChoice, ENVIRONMENT_DESCRIPTOR_VERSION,
            EnvironmentDescriptor, FrameChannel, FrameReference, FrameRepresentation,
            FrameVisibility, ObservationField, ObservationType, QuantityLabel, RewardDescriptor,
            VisualizationPreferences,
        },
    };
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 1 };
    let mut manifest =
        RunManifest::new("environment-test", "pair", "test", 1, space.clone(), 0.9).unwrap();
    let clock = |label: &str| QuantityLabel {
        label: label.into(),
        unit: None,
    };
    let channels = [
        ("height", "Altitude"),
        ("speed", "Velocity"),
        ("pitch", "Attitude"),
    ];
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: vec![ObservationField {
            id: "state".into(),
            label: "State".into(),
            unit: None,
            description: None,
            kind: ObservationType::Numerical,
            range: None,
            coordinate: false,
        }],
        actions: ActionDescriptor::Discrete {
            choices: vec![DiscreteChoice {
                label: "Action".into(),
                meaning: None,
            }],
        },
        clocks: ClockDescriptor {
            learner_revision: clock("Revision"),
            decision_step: clock("Decision"),
            simulation: None,
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: channels
            .iter()
            .map(|(name, label)| FrameChannel {
                name: (*name).into(),
                label: (*label).into(),
                width: 1,
                unit: None,
                visibility: FrameVisibility::DiagnosticOnly,
                representation: Some(FrameRepresentation::Scalar),
            })
            .collect(),
        preferences: VisualizationPreferences {
            trajectories: channels
                .iter()
                .map(|(name, _)| FrameReference {
                    channel: (*name).into(),
                    index: 0,
                })
                .collect(),
            ..Default::default()
        },
    });
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let seed_key = "trajectory/scenario_seed.v1";
    let mut learner = trajectory(1, true);
    let add_frames = |chunk: &mut Chunk| {
        let Record::Trajectory(record) = &mut chunk.record else {
            unreachable!()
        };
        for (name, _) in channels {
            record.frames.insert(name.into(), name.into());
            chunk.arrays.insert(
                name.into(),
                Array::f64(vec![3, 1], vec![2., 1., 0.]).unwrap(),
            );
        }
    };
    add_frames(&mut learner);
    learner
        .arrays
        .insert(seed_key.into(), Array::u64(vec![1], vec![11]).unwrap());
    writer.append(&learner).unwrap();

    let spec = ReferenceSpec {
        id: "paired-controller".into(),
        context: ReferenceContext {
            environment: "arbitrary-custom".into(),
            observation_dim: 1,
            typed_observation_space: None,
            action_space: space,
            conventions: ValueConventions {
                dynamics: "two successive decisions".into(),
                observation: "elapsed decisions".into(),
                action: "one discrete action".into(),
                reward_units: "reward".into(),
                reward_convention: "environment reward".into(),
                discount: 0.9,
                horizon: "environment termination after two decisions".into(),
                terminal_treatment: "no continuation after termination".into(),
                action_constraints: "single legal action".into(),
            },
            objective: ReferenceObjective::ExpectedDiscountedReturn,
            horizon: ReferenceHorizon::Finite { decisions: 2 },
            environment_config: BTreeMap::from([("horizon".into(), ConfigValue::Integer(2))]),
            markov_observation: true,
            decision_time_coordinate: Some(0),
        },
        quality: ReferenceQuality {
            kind: ReferenceKind::Heuristic,
            method: "select the only available action".into(),
            assumptions: Vec::new(),
        },
        target: ReferenceTarget::BenchmarkPolicy,
        lifetime: ReferenceLifetime::Fixed,
        information: ReferenceInformation::LearnerObservation,
        uncertainty: None,
    };
    let specification = serde_json::to_vec(&spec).unwrap();
    for seed in [79, 11] {
        let mut benchmark = trajectory(1, true);
        add_frames(&mut benchmark);
        let role = spec.model_role();
        let id = format!("benchmark-{seed}");
        benchmark.context.identity.model_role = role.clone();
        benchmark.context.trajectory = Some(id.clone());
        let Record::Trajectory(record) = &mut benchmark.record else {
            unreachable!()
        };
        record.id = id;
        record.evaluation.continuation_policy = role;
        benchmark
            .arrays
            .insert(seed_key.into(), Array::u64(vec![1], vec![seed]).unwrap());
        benchmark.arrays.insert(
            "metadata/reference-provider.v1".into(),
            Array::u8(vec![specification.len()], specification.clone()).unwrap(),
        );
        writer.append(&benchmark).unwrap();
    }
    writer.finish().unwrap();

    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    h.click("Compare matched benchmark");
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(h.text.iter().any(|(text, _)| text.contains("scenario 11")));
    assert!(!h.text.iter().any(|(text, _)| text.contains("scenario 79")));
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text == "Benchmark state 0 of 2"),
        "{:?}",
        h.text
    );
    let title = |name: &str| h.text.iter().find(|(text, _)| text == name).unwrap().1;
    let altitude = title("Altitude");
    let velocity = title("Velocity");
    let attitude = title("Attitude");
    assert!(
        altitude.left() > h.size.x / 2.
            && altitude.top() < velocity.top()
            && velocity.top() < attitude.top()
    );
    assert!(
        attitude.bottom() < h.size.y,
        "matched telemetry must fit the desktop viewport"
    );
    let scene_title = |name: &str| {
        h.text
            .iter()
            .find(|(text, rect)| text == name && rect.left() < h.size.x / 2.)
            .unwrap()
            .1
    };
    assert!(scene_title("Learner").top() < scene_title("Benchmark").top());
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text.starts_with("Cumulative reward"))
    );
    h.click("Details");
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text.starts_with("Primitive steps"))
    );
    h.click("Details");
    h.size = egui::vec2(1280., 800.);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(
        h.text
            .iter()
            .any(|(text, rect)| text == "Attitude" && rect.bottom() < h.size.y)
    );
    h.size = egui::vec2(1024., 640.);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    h.frame(vec![
        egui::Event::PointerMoved(egui::pos2(900., 550.)),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0., -700.),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Unknown,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    for _ in 0..15 {
        h.frame(vec![]);
    }
    assert!(
        h.text
            .iter()
            .any(|(text, rect)| text == "Attitude" && rect.top() >= 0. && rect.bottom() < h.size.y),
        "matched telemetry must remain reachable in a compact viewport"
    );
    h.click("Next");
    wait_ui(&mut h, Laboratory::environment_ready);
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text == "Benchmark state 1 of 2")
    );
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 0
        }
    );
}

#[test]
fn rendered_episode_shows_recorded_observation_telemetry_with_the_scene() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = RunManifest::new(
        "environment-test",
        "custom",
        "custom",
        1,
        ActionSpace::Discrete { count: 1 },
        0.9,
    )
    .unwrap();
    let mut chunk = trajectory(1, true);
    chunk.arrays.insert(
        "s".into(),
        Array::f32(vec![11, 1], (0..=10).map(|step| step as f32).collect()).unwrap(),
    );
    chunk
        .arrays
        .insert("a".into(), Array::u64(vec![10], vec![0; 10]).unwrap());
    chunk
        .arrays
        .insert("r".into(), Array::f32(vec![10], vec![1.; 10]).unwrap());
    chunk.arrays.insert(
        "term".into(),
        Array::u8(vec![10], [vec![0; 9], vec![1]].concat()).unwrap(),
    );
    chunk
        .arrays
        .insert("trunc".into(), Array::u8(vec![10], vec![0; 10]).unwrap());
    let bytes = serde_json::to_vec(&scene()).unwrap();
    chunk.arrays.insert(
        "render-data".into(),
        Array::u8(vec![bytes.len() * 11], bytes.repeat(11)).unwrap(),
    );
    chunk.arrays.insert(
        "render-offsets".into(),
        Array::u64(
            vec![12],
            (0..=11).map(|i| (i * bytes.len()) as u64).collect(),
        )
        .unwrap(),
    );
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..3 {
        h.frame(vec![]);
    }
    assert!(h.has("Episode telemetry"));
    assert!(h.has("obs[0]"));
    assert!(h.has("Decision step"));
    let axis = h
        .text
        .iter()
        .find(|(text, _)| text == "Decision step")
        .unwrap()
        .1;
    let ticks = h
        .text
        .iter()
        .filter(|(text, rect)| {
            text.parse::<f64>().is_ok()
                && rect.left() > h.size.x / 2.
                && rect.bottom() <= axis.top() + 1.
                && rect.bottom() >= axis.top() - 30.
        })
        .count();
    assert!(
        ticks >= 3,
        "telemetry time axis must have readable numerical ticks: {:?}",
        h.text
    );
    let revision = h.app.selection().revision;
    let sequence = match h.app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("expected selected episode"),
    };
    let plot_id = egui::Id::unique((
        "environment-observation",
        run.root().as_ref(),
        sequence,
        0usize,
    ));
    let memory = egui_plot::PlotMemory::load(&h.ctx, plot_id).unwrap();
    let transform = memory.transform();
    let x = transform
        .position_from_point(&egui_plot::PlotPoint::new(10., 10.))
        .x;
    h.click_at(egui::pos2(x, transform.frame().center().y));
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.app.selection().input != (SelectedInput::TrajectoryStep { sequence, step: 10 })
        || !h.app.environment_ready()
    {
        h.frame(vec![]);
        assert!(
            Instant::now() < deadline,
            "telemetry selection did not publish"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(h.app.selection().revision, revision);
    assert_eq!(
        h.app.selection().input,
        SelectedInput::TrajectoryStep { sequence, step: 10 }
    );
}

#[test]
fn rendered_episode_includes_recorded_continuous_action_telemetry() {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Box {
        low: vec![0.],
        high: vec![1.],
    };
    let manifest = RunManifest::new("environment-test", "custom", "custom", 1, space, 0.9).unwrap();
    let mut chunk = trajectory(1, true);
    chunk
        .arrays
        .insert("a".into(), Array::f32(vec![2, 1], vec![0.2, 0.8]).unwrap());
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Environment);
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..3 {
        h.frame(vec![]);
    }
    assert!(h.has("action[0]"));
    let sequence = match h.app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("expected selected episode"),
    };
    let plot_id = egui::Id::unique(("environment-action", run.root().as_ref(), sequence, 0usize));
    assert!(egui_plot::PlotMemory::load(&h.ctx, plot_id).is_some());
}

#[test]
fn rendered_episode_pairs_physical_frames_and_selects_the_recorded_state() {
    use rl::diagnostics::visualization::{
        ActionDescriptor, ClockDescriptor, DiscreteChoice, ENVIRONMENT_DESCRIPTOR_VERSION,
        EnvironmentDescriptor, FrameChannel, FrameRepresentation, FrameVisibility,
        ObservationField, ObservationType, QuantityLabel, RewardDescriptor,
        VisualizationPreferences,
    };
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 1 };
    let mut manifest =
        RunManifest::new("environment-test", "custom", "custom", 1, space, 0.9).unwrap();
    let clock = |label: &str| QuantityLabel {
        label: label.into(),
        unit: None,
    };
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: vec![ObservationField {
            id: "state".into(),
            label: "State".into(),
            unit: None,
            description: None,
            kind: ObservationType::Numerical,
            range: None,
            coordinate: false,
        }],
        actions: ActionDescriptor::Discrete {
            choices: vec![DiscreteChoice {
                label: "Action".into(),
                meaning: None,
            }],
        },
        clocks: ClockDescriptor {
            learner_revision: clock("Revision"),
            decision_step: clock("Decision"),
            simulation: Some(clock("Time")),
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: [("x", "Horizontal position"), ("height", "Altitude")]
            .into_iter()
            .map(|(name, label)| FrameChannel {
                name: name.into(),
                label: label.into(),
                width: 1,
                unit: Some("m".into()),
                visibility: FrameVisibility::DiagnosticOnly,
                representation: Some(FrameRepresentation::Scalar),
            })
            .collect(),
        preferences: VisualizationPreferences::default(),
    });
    let mut chunk = trajectory(1, true);
    chunk.arrays.insert(
        "physical-x".into(),
        Array::f64(vec![3, 1], vec![-2., -1., 0.]).unwrap(),
    );
    chunk.arrays.insert(
        "physical-height".into(),
        Array::f64(vec![3, 1], vec![10., 5., 0.]).unwrap(),
    );
    chunk.arrays.insert(
        "time".into(),
        Array::f64(vec![3], vec![0., 0.04, 0.10]).unwrap(),
    );
    let Record::Trajectory(record) = &mut chunk.record else {
        unreachable!()
    };
    record.frames.insert("x".into(), "physical-x".into());
    record
        .frames
        .insert("height".into(), "physical-height".into());
    record.simulation_times = Some("time".into());
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(1280., 800.));
    h.app.select_view(View::Environment);
    h.app.select_revision(1).unwrap();
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(h.has("Recorded trajectory"));
    assert!(h.has("Horizontal position [m]"));
    assert!(h.has("Altitude [m]"));
    let sequence = match h.app.selection().input {
        SelectedInput::TrajectoryStep { sequence, .. } => sequence,
        _ => panic!("expected episode"),
    };
    let plot_id = egui::Id::unique((
        "environment-frame-trajectory",
        run.root().as_ref(),
        sequence,
    ));
    let transform = egui_plot::PlotMemory::load(&h.ctx, plot_id)
        .unwrap()
        .transform();
    let bounds_before = (transform.bounds().min(), transform.bounds().max());
    let position = transform.position_from_point(&egui_plot::PlotPoint::new(-1., 5.));
    h.click_at(position);
    wait_ui(&mut h, |app| {
        app.selection().input == SelectedInput::TrajectoryStep { sequence, step: 1 }
            && app.environment_ready()
    });
    for _ in 0..3 {
        h.frame(vec![]);
    }
    let transform = egui_plot::PlotMemory::load(&h.ctx, plot_id)
        .unwrap()
        .transform();
    assert_eq!(
        (transform.bounds().min(), transform.bounds().max()),
        bounds_before
    );
    fn has_selected_marker(shape: &egui::Shape, expected: egui::Pos2) -> bool {
        match shape {
            egui::Shape::Circle(circle) => {
                circle.fill == rl_viewer::theme::SELECTED && circle.center.distance(expected) < 2.
            }
            egui::Shape::Vec(shapes) => shapes
                .iter()
                .any(|shape| has_selected_marker(shape, expected)),
            _ => false,
        }
    }
    let expected = transform.position_from_point(&egui_plot::PlotPoint::new(-1., 5.));
    assert!(
        h.shapes
            .iter()
            .any(|shape| has_selected_marker(shape, expected)),
        "selected marker must use the exact current frame values"
    );
    h.size = egui::vec2(1024., 640.);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    let title = h
        .text
        .iter()
        .find(|(text, rect)| text == "Recorded trajectory" && rect.center().x < 512.)
        .unwrap()
        .1;
    let y_axis = h
        .text
        .iter()
        .find(|(text, rect)| text == "Altitude [m]" && rect.center().x < 512.)
        .unwrap()
        .1;
    let x_axis = h
        .text
        .iter()
        .find(|(text, rect)| text == "Horizontal position [m]" && rect.center().x < 512.)
        .unwrap()
        .1;
    assert!(
        !title.intersects(y_axis),
        "compact trajectory title overlaps the Y axis"
    );
    assert!(
        x_axis.bottom() <= h.size.y && y_axis.bottom() <= h.size.y,
        "compact trajectory labels are clipped"
    );
}

#[test]
fn rendered_episode_with_five_channels_keeps_time_ticks_visible() {
    use rl::diagnostics::visualization::{
        ActionComponent, ActionDescriptor, ClockDescriptor, ENVIRONMENT_DESCRIPTOR_VERSION,
        EnvironmentDescriptor, ObservationField, ObservationType, QuantityLabel, RewardDescriptor,
        VisualizationPreferences,
    };
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Box {
        low: vec![0.],
        high: vec![1.],
    };
    let mut manifest =
        RunManifest::new("environment-test", "custom", "custom", 4, space, 0.9).unwrap();
    let clock = |label: &str| QuantityLabel {
        label: label.into(),
        unit: None,
    };
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: ["Altitude", "Vertical velocity", "Attitude", "Fuel"]
            .iter()
            .enumerate()
            .map(|(i, label)| ObservationField {
                id: format!("o{i}"),
                label: (*label).into(),
                unit: None,
                description: None,
                kind: ObservationType::Numerical,
                range: None,
                coordinate: false,
            })
            .collect(),
        actions: ActionDescriptor::Continuous {
            components: vec![ActionComponent {
                label: "Throttle".into(),
                unit: None,
                meaning: None,
            }],
            preferred_component: Some(0),
        },
        clocks: ClockDescriptor {
            learner_revision: clock("Revision"),
            decision_step: clock("Decision step"),
            simulation: None,
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![],
        preferences: VisualizationPreferences::default(),
    });
    let mut chunk = trajectory(1, true);
    chunk.arrays.insert(
        "s".into(),
        Array::f32(
            vec![3, 4],
            vec![0., 0., 0., 0., 3.5, 3.5, 3.5, 3.5, 7., 7., 7., 7.],
        )
        .unwrap(),
    );
    chunk
        .arrays
        .insert("a".into(), Array::f32(vec![2, 1], vec![0.2, 0.8]).unwrap());
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let mut h = support::Harness::new(&dir, egui::vec2(1280., 800.));
    h.app.select_view(View::Environment);
    wait_ui(&mut h, Laboratory::environment_ready);
    for _ in 0..6 {
        h.frame(vec![]);
    }
    assert!(h.has("Throttle"));
    let axis = h
        .text
        .iter()
        .find(|(text, _)| text == "Decision step")
        .unwrap()
        .1;
    let ticks = h
        .text
        .iter()
        .filter(|(text, rect)| {
            text.parse::<f64>().is_ok()
                && rect.left() > h.size.x / 2.
                && rect.bottom() <= axis.top() + 1.
                && rect.bottom() >= axis.top() - 30.
        })
        .count();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let plot_id = egui::Id::unique(("environment-action", run.root().as_ref(), 0u64, 0usize));
    let memory = egui_plot::PlotMemory::load(&h.ctx, plot_id).unwrap();
    assert!(
        ticks >= 3,
        "time ticks must be visible above the last axis: frame={:?}, bounds={:?}, text={:?}",
        memory.transform().frame(),
        memory.transform().bounds(),
        h.text
    );
    let left_edges = (0..4)
        .map(|index| {
            let id =
                egui::Id::unique(("environment-observation", run.root().as_ref(), 0u64, index));
            egui_plot::PlotMemory::load(&h.ctx, id)
                .unwrap()
                .transform()
                .frame()
                .left()
        })
        .chain(std::iter::once(memory.transform().frame().left()))
        .collect::<Vec<_>>();
    assert!(
        left_edges
            .iter()
            .all(|left| (left - left_edges[0]).abs() < 1.),
        "time axes must align: {left_edges:?}"
    );
    for index in 0..4 {
        let id = egui::Id::unique(("environment-observation", run.root().as_ref(), 0u64, index));
        let memory = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
        let plot = *memory.transform().frame();
        let ticks = h
            .text
            .iter()
            .filter(|(text, rect)| {
                text.parse::<f64>().is_ok()
                    && rect.right() <= plot.left()
                    && rect.left() >= plot.left() - 60.
                    && rect.center().y >= plot.top()
                    && rect.center().y <= plot.bottom()
            })
            .count();
        assert!(
            ticks >= 2,
            "telemetry chart {index} needs at least two value ticks: {plot:?}, text={:?}",
            h.text
        );
    }
}
