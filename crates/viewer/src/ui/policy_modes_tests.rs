use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        models::{ModelBundle, ModelLimits, SnapshotBuilder},
        policy_cache::{CacheLimits, PolicyCache},
        policy_precompute::{PrecomputeOptions, precompute},
        policy_slice::{PolicySliceSpec, SliceConfig},
        policy_surface::evaluate_policy_batch,
        storage::{Recorder, StorageConfig},
        visualization::EnvironmentSemantics,
    },
    nn::policy::{Deterministic, Policy},
};

fn fixture(
    cached: bool,
    prune: bool,
    same_revision: bool,
) -> (tempfile::TempDir, egui::Context, Laboratory) {
    family_fixture(cached, prune, same_revision, 0)
}
fn family_fixture(
    cached: bool,
    prune: bool,
    same_revision: bool,
    family: usize,
) -> (tempfile::TempDir, egui::Context, Laboratory) {
    let dir = tempfile::tempdir().unwrap();
    let space = match family {
        1 => ActionSpace::Discrete { count: 3 },
        2 => ActionSpace::Box {
            low: vec![-2., -1.],
            high: vec![3., 4.],
        },
        _ => ActionSpace::Unbounded { dimensions: 2 },
    };
    let manifest = RunManifest::new("modes", "test", "ordinary", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (seed, revision) in [3, if same_revision { 3 } else { 7 }]
        .into_iter()
        .enumerate()
    {
        let mut builder = SnapshotBuilder::new(
            &format!("saved-{seed}"),
            RecordContext::new("modes", revision),
            2,
            space.clone(),
        )
        .unwrap();
        if family == 0 {
            builder
                .deterministic(
                    "actor",
                    &Deterministic::new(
                        2,
                        space.clone(),
                        &[4],
                        &mut ChaCha8Rng::seed_from_u64(seed as u64),
                    )
                    .unwrap(),
                )
                .unwrap();
        } else {
            builder
                .policy(
                    "actor",
                    &Policy::new(
                        2,
                        space.clone(),
                        &[4],
                        &mut ChaCha8Rng::seed_from_u64(seed as u64),
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        let snapshot = builder.finish().unwrap();
        let model = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
        let mut evaluated = evaluate_policy_batch(
            &model,
            "actor",
            &[
                Observation::new(vec![0., 0.]),
                Observation::new(vec![0.5, -0.25]),
            ],
        )
        .unwrap();
        // These fixture observations are explicitly unconditioned. Inspection
        // also reports full effective support; retain the actual None input.
        if let Record::ProbeSet(p) = &mut evaluated.probes.record
            && let Some(key) = p.action_masks.take()
        {
            evaluated.probes.arrays.remove(&key);
        }
        if let Record::Policy(p) = &mut evaluated.policy.record
            && let Some(key) = p.masks.take()
        {
            evaluated.policy.arrays.remove(&key);
        }
        writer.append(&evaluated.probes).unwrap();
        writer.append(&evaluated.policy).unwrap();
        writer.append(&snapshot).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    if cached {
        let mut config = SliceConfig::new(
            EnvironmentSemantics::new(run.recording().manifest()),
            &Observation::new(vec![0., 0.]),
            &[],
        )
        .unwrap();
        config.resolution = [8, 8];
        let spec = PolicySliceSpec::from_config("online", "actor", &config);
        let cache = PolicyCache::open_or_create(
            dir.path().join("run-policy-cache"),
            run.recording(),
            CacheLimits::default(),
        )
        .unwrap();
        precompute(
            run.recording(),
            &cache,
            &[spec],
            &PrecomputeOptions::default(),
            |_| {},
        )
        .unwrap();
    }
    if prune {
        for entry in run
            .recording()
            .entries()
            .iter()
            .filter(|e| e.kind == RecordKind::ModelSnapshot)
        {
            std::fs::remove_file(run.root().join(&entry.file)).unwrap();
        }
    }
    let ctx = egui::Context::default();
    let app = Laboratory::from_context(&ctx, run.root().display().to_string()).unwrap();
    (dir, ctx, app)
}

#[test]
fn all_saved_policy_families_export_play_components_and_return_to_interactive() {
    for family in 0..3 {
        let (dir, ctx, mut app) = family_fixture(false, false, false, family);
        settle(&mut app, &ctx);
        let mut spec = app.export_policy_slice().unwrap();
        spec.resolution = [8, 8];
        app.configure_policy_demo_slice(&spec).unwrap();
        settle(&mut app, &ctx);
        let expected = app.policy_frame().unwrap().grid.values.clone();
        let exported = app.export_policy_slice().unwrap();
        assert_eq!(exported, spec);
        let run = app.run.as_ref().unwrap();
        let cache = PolicyCache::open_or_create(
            dir.path().join("run-policy-cache"),
            run.recording(),
            CacheLimits::default(),
        )
        .unwrap();
        precompute(
            run.recording(),
            &cache,
            &[exported],
            &PrecomputeOptions::default(),
            |_| {},
        )
        .unwrap();
        app.set_policy_mode(PolicyMode::Prepared);
        settle(&mut app, &ctx);
        assert_eq!(app.policy_frame().unwrap().grid.values, expected);
        let before = app.policy_metrics();
        for source in [2, 5, 2, 5] {
            app.select_prepared_source(source).unwrap();
            settle(&mut app, &ctx);
            assert_eq!(
                app.policy_frame().unwrap().grid.current.key.sequence,
                source
            );
        }
        app.configure_policy_demo_component(1, 9).unwrap();
        settle(&mut app, &ctx);
        let selected = app.policy_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(selected.cached_row, Some(9));
        assert!(selected.value.is_finite());
        if family == 2 {
            assert_eq!(selected.plot.as_ref().unwrap().domain, [-1., 4.]);
        }
        assert_eq!(app.policy_metrics().restores, before.restores);
        assert_eq!(app.policy_metrics().grid_batches, before.grid_batches);
        app.set_policy_mode(PolicyMode::Interactive);
        settle(&mut app, &ctx);
        assert_eq!(app.export_policy_slice().unwrap(), spec);
        assert_eq!(app.policy_frame().unwrap().grid.values, expected);
    }
}

#[test]
fn prepared_replacement_waits_for_worker_ack_across_departures() {
    use crate::policy::cache::{PreparedLimits, PreparedProvider};
    use std::sync::{
        Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    struct Release(Arc<(Mutex<bool>, Condvar)>);
    impl Drop for Release {
        fn drop(&mut self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
    }
    for departure in 0..4 {
        let (dir, ctx, mut app) = fixture(true, false, false);
        let blocked = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Release(blocked.clone());
        let entered = Arc::new(AtomicBool::new(false));
        let signal = entered.clone();
        app.policy_execution.provider = Some(
            PreparedProvider::new(
                app.run.as_ref().unwrap().clone(),
                dir.path().join("run-policy-cache"),
                PreparedLimits::default(),
                move || {
                    signal.store(true, Ordering::Release);
                    let (lock, wake) = &*blocked;
                    let mut ready = lock.lock().unwrap();
                    while !*ready {
                        ready = wake.wait(ready).unwrap();
                    }
                },
            )
            .unwrap(),
        );
        app.set_policy_mode(PolicyMode::Prepared);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !entered.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        match departure {
            0 => {
                app.select_view(View::Network);
                app.select_view(View::PolicyMap);
            }
            1 => app.open_run(0),
            2 => {
                app.clear_recording();
                app.open_run(0);
            }
            3 => super::policy_workspace_tests::click(&mut app, &ctx, "Reload"),
            _ => unreachable!(),
        }
        // The old operation is deterministically held, independent of storage
        // speed. Repeated UI paints must not admit any replacement worker.
        for _ in 0..10 {
            paint(&mut app, &ctx);
            assert!(
                app.policy_execution.provider.is_none(),
                "departure {departure} admitted a replacement before retirement"
            );
            assert_eq!(app.policy_metrics().restores, 0);
        }
        drop(release);
        settle(&mut app, &ctx);
        assert_eq!(app.policy_mode(), Some(PolicyMode::Prepared));
    }
}

#[test]
fn prepared_replacement_keeps_external_frame_owners_charged() {
    let (_dir, ctx, mut app) = fixture(true, false, false);
    settle(&mut app, &ctx);
    let external = app.policy_frame().unwrap().clone();
    app.select_view(View::Network);
    app.select_view(View::PolicyMap);
    for _ in 0..10 {
        paint(&mut app, &ctx);
        assert!(app.policy_execution.provider.is_none());
    }
    drop(external);
    settle(&mut app, &ctx);
    assert!(app.prepared_policy_metrics().unwrap().retained_bytes > 0);
}
fn paint(app: &mut Laboratory, ctx: &egui::Context) {
    let mut out = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 800.),
            )),
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    out.textures_delta.clear();
    assert!(app.notice().is_none(), "{:?}", app.notice());
}
fn settle(app: &mut Laboratory, ctx: &egui::Context) {
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        paint(app, ctx);
        if app.policy_pane.ready() {
            break;
        }
        assert!(Instant::now() < until, "{:?}", app.policy_pane.status());
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn prepared_policy_mode_and_slice_share_a_toolbar() {
    fn labels(shape: &egui::Shape, output: &mut Vec<(String, f32)>) {
        match shape {
            egui::Shape::Text(text) => output.push((text.galley.text().into(), text.pos.y)),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|shape| labels(shape, output)),
            _ => {}
        }
    }
    let (_dir, ctx, mut app) = fixture(true, true, true);
    settle(&mut app, &ctx);
    let output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1200., 800.),
            )),
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    let mut text = Vec::new();
    for shape in &output.shapes {
        labels(&shape.shape, &mut text);
    }
    let y = |label: &str| text.iter().find(|(value, _)| value == label).unwrap().1;
    assert!((y("Mapping") - y("Slice")).abs() < 8., "{text:?}");
    assert!(text.iter().any(|(value, _)| value == "Reload"));
    super::policy_workspace_tests::click(&mut app, &ctx, "Evolution");
    assert!(app.policy_evolution);
    super::policy_workspace_tests::click(&mut app, &ctx, "Mapping");
    settle(&mut app, &ctx);
    assert!(!app.policy_evolution);
    assert_eq!(app.policy_mode(), Some(PolicyMode::Prepared));
    assert_eq!(app.policy_metrics().grid_batches, 0);
}

#[test]
fn compatible_cache_defaults_to_prepared_and_never_restores_pruned_models() {
    let (_dir, ctx, mut app) = fixture(true, true, true);
    settle(&mut app, &ctx);
    assert_eq!(app.policy_mode(), Some(PolicyMode::Prepared));
    assert_eq!(app.policy_metrics().restores, 0);
    #[cfg(feature = "screenshots")]
    assert_eq!(app.policy_capture_problem(Some("grid"), None), None);
    let last = app.policy_frame().unwrap().grid.current.key.sequence;
    app.select_prepared_source(2).unwrap();
    settle(&mut app, &ctx);
    assert_eq!(app.policy_frame().unwrap().grid.current.key.sequence, 2);
    assert_ne!(last, 2);
    assert_eq!(app.selection.revision, 3);
    let count = app.prepared_policy_metrics().unwrap().preparations;
    for _ in 0..10 {
        paint(&mut app, &ctx);
    }
    assert_eq!(app.prepared_policy_metrics().unwrap().preparations, count);
    assert_eq!(app.policy_metrics().grid_batches, 0);
}

#[test]
fn exported_interactive_slice_round_trips_into_prepared_and_back() {
    let (dir, ctx, mut app) = fixture(false, false, false);
    settle(&mut app, &ctx);
    let exported = app.export_policy_slice().unwrap();
    assert_eq!(
        exported
            .to_config(EnvironmentSemantics::new(
                app.run.as_ref().unwrap().recording().manifest()
            ))
            .unwrap(),
        *app.policy_pane.config().unwrap()
    );
    let before = app.policy_frame().unwrap().grid.values.clone();
    let run = app.run.as_ref().unwrap();
    let cache = PolicyCache::open_or_create(
        dir.path().join("run-policy-cache"),
        run.recording(),
        CacheLimits::default(),
    )
    .unwrap();
    precompute(
        run.recording(),
        &cache,
        std::slice::from_ref(&exported),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    app.set_policy_mode(PolicyMode::Prepared);
    settle(&mut app, &ctx);
    assert_eq!(app.policy_frame().unwrap().grid.values, before);
    let restores = app.policy_metrics().restores;
    for _ in 0..10 {
        app.set_policy_mode(PolicyMode::Interactive);
        app.set_policy_mode(PolicyMode::Prepared);
    }
    settle(&mut app, &ctx);
    assert_eq!(app.policy_metrics().restores, restores);
    app.set_policy_mode(PolicyMode::Interactive);
    settle(&mut app, &ctx);
    assert_eq!(app.export_policy_slice().unwrap(), exported);
}

#[test]
fn explicitly_missing_prepared_data_has_no_automatic_inference_fallback() {
    let (_dir, ctx, mut app) = fixture(false, false, false);
    app.set_policy_mode(PolicyMode::Prepared);
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        paint(&mut app, &ctx);
        if app.policy_pane.terminal {
            break;
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(app.policy_mode(), Some(PolicyMode::Prepared));
    assert_eq!(app.policy_metrics().restores, 0);
    assert!(app.policy_frame().is_none());
}

#[test]
fn switching_to_interactive_keeps_an_exact_same_revision_prepared_source() {
    let (_dir, ctx, mut app) = fixture(true, false, true);
    settle(&mut app, &ctx);
    app.select_prepared_source(2).unwrap();
    settle(&mut app, &ctx);
    let values = app.policy_frame().unwrap().grid.values.clone();
    app.set_policy_mode(PolicyMode::Interactive);
    settle(&mut app, &ctx);
    assert_eq!(app.policy_frame().unwrap().grid.current.key.sequence, 2);
    assert_eq!(app.policy_frame().unwrap().grid.values, values);
}

#[test]
fn selecting_a_disjoint_cached_slice_refits_its_bounds() {
    let (dir, ctx, mut app) = fixture(true, false, false);
    let run = app.run.as_ref().unwrap();
    let cache = PolicyCache::open(
        dir.path().join("run-policy-cache"),
        run.recording(),
        CacheLimits::default(),
    )
    .unwrap();
    let mut spec = cache.catalog().unwrap().entries[0].slice.clone();
    spec.ranges = vec![[100., 200.], [100., 200.]];
    precompute(
        run.recording(),
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    let far = cache
        .catalog()
        .unwrap()
        .entries
        .iter()
        .find(|e| e.slice.ranges[0][0] == 100.)
        .unwrap()
        .slice_sha256
        .clone();
    settle(&mut app, &ctx);
    let near = cache
        .catalog()
        .unwrap()
        .entries
        .iter()
        .find(|e| e.slice.ranges[0][0] < 0.)
        .unwrap()
        .slice_sha256
        .clone();
    app.select_prepared_slice(&near).unwrap();
    settle(&mut app, &ctx);
    app.select_prepared_slice(&far).unwrap();
    settle(&mut app, &ctx);
    let memory =
        egui_plot::PlotMemory::load(&ctx, crate::policy::view::PolicyPane::main_plot_id()).unwrap();
    assert!(memory.bounds().max()[0] >= 200., "{:?}", memory.bounds());
    assert!(memory.bounds().min()[0] >= 90., "{:?}", memory.bounds());
}

#[test]
fn reference_off_restores_the_current_plot_and_export_saves_complete_json() {
    use super::policy_workspace_tests::click;
    let (_dir, ctx, mut app) = fixture(true, false, false);
    settle(&mut app, &ctx);
    click(&mut app, &ctx, "Off");
    click(&mut app, &ctx, "actor / online · k=3 · source 2");
    settle(&mut app, &ctx);
    app.policy_pane
        .set_display(crate::policy::view::Display::Difference);
    click(&mut app, &ctx, "actor · source 2");
    click(&mut app, &ctx, "Off");
    settle(&mut app, &ctx);
    assert_eq!(
        app.policy_pane.display,
        crate::policy::view::Display::Current
    );
    app.set_policy_mode(PolicyMode::Interactive);
    settle(&mut app, &ctx);
    click(&mut app, &ctx, "Slice settings");
    click(&mut app, &ctx, "Export slice…");
    assert!(app.policy_execution.export_open);
    let specs: Vec<PolicySliceSpec> =
        serde_json::from_str(&app.policy_execution.export_text).unwrap();
    assert_eq!(specs, [app.export_policy_slice().unwrap()]);
    let output = app
        .run
        .as_ref()
        .unwrap()
        .root()
        .with_file_name("exported-test-slices.json");
    app.policy_execution.export_path = output.display().to_string();
    click(&mut app, &ctx, "Save file");
    assert_eq!(
        rl::diagnostics::policy_precompute::read_slice_specs(output).unwrap(),
        specs
    );
    app.policy_execution.export_open = false;
    let config = app.policy_pane.config.as_mut().unwrap();
    config.ranges[0] = [-3., 3.];
    config.range_sources[0] = rl::diagnostics::policy_slice::RangeSource::UserConfigured;
    app.policy_pane.dirty = true;
    app.policy_pane.ready = false;
    settle(&mut app, &ctx);
    click(&mut app, &ctx, "Slice settings");
    click(&mut app, &ctx, "Export slice…");
    let exported: Vec<PolicySliceSpec> =
        serde_json::from_str(&app.policy_execution.export_text).unwrap();
    assert_eq!(exported.len(), 2);
    assert_eq!(exported[1], app.export_policy_slice().unwrap());
}

#[test]
fn incomplete_cache_does_not_substitute_an_adjacent_source() {
    let (dir, ctx, mut app) = fixture(true, false, false);
    let run = app.run.as_ref().unwrap();
    let cache = PolicyCache::open(
        dir.path().join("run-policy-cache"),
        run.recording(),
        CacheLimits::default(),
    )
    .unwrap();
    let missing = cache
        .catalog()
        .unwrap()
        .entries
        .into_iter()
        .find(|e| e.source.sequence == 5)
        .unwrap();
    std::fs::remove_dir_all(cache.root().join("frames").join(missing.key)).unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        paint(&mut app, &ctx);
        if app.policy_pane.terminal {
            break;
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(app.policy_mode(), Some(PolicyMode::Prepared));
    assert_eq!(app.selection.revision, 7);
    assert!(app.policy_frame().is_none());
    assert_eq!(app.policy_metrics().restores, 0);
    app.select_prepared_source(2).unwrap();
    settle(&mut app, &ctx);
    assert_eq!(app.policy_frame().unwrap().grid.current.key.sequence, 2);
}

#[test]
fn run_catalog_does_not_report_a_valid_policy_cache_as_a_broken_recording() {
    let (dir, _ctx, _app) = fixture(true, false, false);
    let (entries, issues) = crate::data::catalog(dir.path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert!(issues.is_empty(), "{issues:?}");
}
