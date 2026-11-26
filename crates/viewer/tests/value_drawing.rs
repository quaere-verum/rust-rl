//! Focused allocation evidence. Cases serialize internally because the allocator is process global.
//! Measurement runs additionally use --test-threads=1; ordinary CI may use multiple threads.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[allow(dead_code)]
#[path = "support/value_ui.rs"]
mod families;
#[path = "support/policy_ui.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;
use eframe::egui;
use rl_viewer::{ui::View, value::slice::ValueCoordinate};
use std::time::{Duration, Instant};
static DRAWING_CASE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn policy_sparse_points_use_bounded_color_batches_and_finite_hit_radius() {
    use rl::{
        core::{action::ActionSpace, env::Observation},
        diagnostics::{probes::ProbeSet, schema::*, storage::*},
    };
    let _serial = DRAWING_CASE.lock().unwrap();
    let d = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest =
        RunManifest::new("sparse-policy", "ordinary", "typed", 2, space.clone(), 0.9).unwrap();
    let mut w = Recorder::create(d.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("sparse-policy", 7);
    context.probe_set = Some("fixed".into());
    let observations = (0..4096)
        .map(|i| {
            Observation::new(vec![
                (i % 64) as f32 / 63. - 0.5,
                (i / 64) as f32 / 63. - 0.5,
            ])
        })
        .collect();
    w.append(
        &ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "4096 exact typed input rows",
            observations,
            None,
            space,
        )
        .unwrap()
        .record(context.clone())
        .unwrap(),
    )
    .unwrap();
    w.append(&Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            stochastic: false,
            masks: None,
            entropy: None,
            entropy_method: None,
        }),
        arrays: std::collections::BTreeMap::from([(
            "a".into(),
            Array::f64(
                vec![4096, 1],
                (0..4096).map(|i| (i as f64 / 4095.) * 2. - 1.).collect(),
            )
            .unwrap(),
        )]),
    })
    .unwrap();
    w.finish().unwrap();
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    h.settle();
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert_eq!(h.app.policy_frame().unwrap().grid.population_count, 4096);
    let original = h
        .app
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    let plot =
        egui_plot::PlotMemory::load(&h.ctx, rl_viewer::policy::view::PolicyPane::main_plot_id())
            .unwrap();
    let empty = plot
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(0.45, -1.4));
    h.click_at(empty);
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        original
    );
    let allowance =
        h.app.policy_pane().state_reservation_bytes() - h.app.policy_pane().owned_state_bytes();
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "policy sparse 4096 rows: calls={} peak={} allowance={allowance}",
        measured.calls, measured.peak
    );
    assert!(
        measured.calls < 2048,
        "point drawing allocated per row: {}",
        measured.calls
    );
    assert!(
        measured.peak <= allowance as u64,
        "drawing peak {} exceeds reserved {allowance}",
        measured.peak
    );
}

#[test]
fn sparse_draw_allocations_fit_admission_and_retire_on_departure() {
    let _serial = DRAWING_CASE.lock().unwrap();
    // Admission includes retained source metadata as well as drawing. This
    // population fits the existing lease; the oversized case below must reject.
    let d = families::sparse_populations(2048, 2);
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if h.app.value_frame().is_some() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{:?}",
            h.app.value_view().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    h.app
        .value_view_mut()
        .set_axes(ValueCoordinate::Observation(0), None)
        .unwrap();
    loop {
        h.frame(vec![]);
        if h.app.value_view().ready() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{:?}",
            h.app.value_view().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert_eq!(h.app.value_frame().unwrap().grid.values.len(), 2048);
    let allowance =
        h.app.value_view().state_reservation_bytes() - h.app.value_view().owned_state_bytes();
    let drawing = h
        .app
        .value_view()
        .drawing_bytes(h.app.value_frame().unwrap());
    assert!(drawing <= allowance);
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "admitted sparse current=2048: calls={} requested={} live={} peak={} drawing_bound={drawing} drawing_allowance={allowance}",
        measured.calls, measured.requested, measured.live, measured.peak
    );
    assert!(measured.peak <= drawing as u64);
    assert!(
        measured.peak <= allowance as u64,
        "draw peak {} exceeds view allowance {allowance}",
        measured.peak
    );
    assert!(
        measured.calls < 2048,
        "compatible sparse points must be batched; {} allocations",
        measured.calls
    );
    drop(output);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert!(h.app.value_frame().is_none());
}

#[test]
fn oversized_sparse_draw_is_rejected_before_allocation_and_retires() {
    let _serial = DRAWING_CASE.lock().unwrap();
    // Shared metadata lets 4096 rows fit this lease. The largest ordinarily
    // admitted population still exceeds the drawing bound.
    const ROWS: usize = 8192;
    let d = families::sparse_populations(ROWS, 2);
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    // Choose the target axes as soon as source metadata is ready, before an
    // oversized default 2D frame can consume the old/new publication budget.
    let deadline = Instant::now() + Duration::from_secs(20);
    while h
        .app
        .value_view_mut()
        .set_axes(ValueCoordinate::Observation(0), None)
        .is_err()
    {
        h.frame(vec![]);
        assert!(Instant::now() < deadline, "source metadata unavailable");
        std::thread::sleep(Duration::from_millis(2));
    }
    wait(&mut h, |h| {
        !h.app.value_pending()
            && h.app.value_frame().is_some_and(|f| {
                f.grid.slice.mode == rl_viewer::policy::slice::SliceMode::OneDimensional
            })
    });
    for _ in 0..4 {
        h.frame(vec![]);
    }
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.values.len(), ROWS);
    let drawing = h.app.value_view().drawing_bytes(f);
    let allowance =
        h.app.value_view().state_reservation_bytes() - h.app.value_view().owned_state_bytes();
    assert!(
        drawing > allowance,
        "drawing {drawing} must exceed {allowance}"
    );
    assert!(h.has("Value drawing exceeds shared view admission"));
    assert!(!h.app.value_view().ready());
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "rejected sparse current={ROWS}: calls={} peak={} drawing_bound={drawing} drawing_allowance={allowance}",
        measured.calls, measured.peak
    );
    assert!(
        measured.peak < 128 * 1024,
        "rejection must precede sample plot collection"
    );
    drop(output);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert!(h.app.value_frame().is_none());
}

#[test]
fn independent_reference_rejection_allocates_no_sample_plot_and_retires() {
    let _serial = DRAWING_CASE.lock().unwrap();
    let d = families::sparse_populations(2, 8192);
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    wait(&mut h, |h| h.app.value_view().ready());
    assert_eq!(
        h.app.value_view().slice().unwrap().state.ranges[0],
        [-1., 1.]
    );
    h.click_last("Reference");
    wait(&mut h, |h| h.app.value_view().ready());
    h.set_reference_request(3);
    wait(&mut h, |h| {
        !h.app.value_pending()
            && h.app.value_frame().is_some_and(|f| {
                f.grid
                    .reference_series
                    .as_ref()
                    .is_some_and(|s| s.values.len() == 8192)
            })
    });
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(h.has("Value drawing exceeds shared view admission"));
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.values.len(), 2);
    assert_eq!(f.reference_requested_revision, Some(3));
    assert_eq!(f.grid.reference().unwrap().identity().revision, 3);
    let drawing = h.app.value_view().drawing_bytes(f);
    let allowance =
        h.app.value_view().state_reservation_bytes() - h.app.value_view().owned_state_bytes();
    assert!(drawing > allowance);
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "rejected sparse current=2 reference=8192: calls={} requested={} live={} peak={} drawing_bound={drawing} drawing_allowance={allowance}",
        measured.calls, measured.requested, measured.live, measured.peak
    );
    assert!(
        measured.peak < 128 * 1024,
        "rejection must precede reference plot collection: {}",
        measured.peak
    );
    assert!(!h.app.value_view().ready());
    drop(output);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert!(h.app.value_frame().is_none());
}
fn wait(h: &mut support::Harness, done: impl Fn(&support::Harness) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if done(h) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{:?}",
            h.app.value_view().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn zoomed_reference_curve_clips_dash_expansion_before_allocation() {
    let _serial = DRAWING_CASE.lock().unwrap();
    let d = families::family("quantile", true);
    let mut h = support::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    wait(&mut h, |h| h.app.value_view().ready());
    h.click("Slice");
    wait(&mut h, |h| h.app.value_view().ready());
    h.click_last("Reference");
    wait(&mut h, |h| h.app.value_view().ready());
    for _ in 0..4 {
        h.frame(vec![]);
    }
    let allowance =
        h.app.value_view().state_reservation_bytes() - h.app.value_view().owned_state_bytes();
    let mut memory =
        egui_plot::PlotMemory::load(&h.ctx, rl_viewer::value::view::ValueView::main_plot_id())
            .unwrap();
    memory.set_bounds(egui_plot::PlotBounds::from_min_max(
        [-1., -1e-7],
        [1., 1e-7],
    ));
    memory.store(&h.ctx, rl_viewer::value::view::ValueView::main_plot_id());
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "zoomed model reference: calls={} requested={} live={} peak={} drawing_allowance={allowance}",
        measured.calls, measured.requested, measured.live, measured.peak
    );
    assert!(
        measured.peak <= allowance as u64,
        "zoom-dependent dash expansion exceeded existing admission: {} > {allowance}",
        measured.peak
    );
    assert!(
        h.app.value_view().ready(),
        "clipped ordinary curve should fit: {:?}",
        h.app.value_view().status()
    );
    assert!(!output.shapes.is_empty());
}

#[test]
fn promoted_value_population_has_bounded_idle_drawing_and_retires() {
    let _serial = DRAWING_CASE.lock().unwrap();
    let d = families::projection_population("quantile", 512, "overlap");
    let mut h = support::Harness::new(&d, egui::vec2(1440., 1100.));
    h.app.select_view(View::Value);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if h.app.value_view().ready() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{:?}",
            h.app.value_view().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    for _ in 0..4 {
        h.frame(vec![]);
    }
    let g = &h.app.value_frame().unwrap().grid;
    assert_eq!(g.population_count, 512);
    assert_eq!(g.distinct_positions, 2);
    assert!(g.promote_selected());
    let idle = format!("{:?}", h.app.value_metrics());
    let allowance =
        h.app.value_view().state_reservation_bytes() - h.app.value_view().owned_state_bytes();
    let drawing = h
        .app
        .value_view()
        .drawing_bytes(h.app.value_frame().unwrap());
    let before = allocation::start();
    let mut output = h.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
            time: Some(h.time + 0.1),
            ..Default::default()
        },
        |ui| h.app.show(ui),
    );
    let measured = allocation::finish(before);
    output.textures_delta.clear();
    eprintln!(
        "promoted 512 rows: calls={} peak={} drawing={drawing} allowance={allowance}",
        measured.calls, measured.peak
    );
    assert!(measured.calls < 2048);
    assert!(measured.peak <= drawing as u64 && measured.peak <= allowance as u64);
    assert_eq!(format!("{:?}", h.app.value_metrics()), idle);
    drop(output);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
    assert!(h.app.value_frame().is_none());
}
