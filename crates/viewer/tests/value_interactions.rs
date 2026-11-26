#[path = "support/policy_ui.rs"]
mod support;
use eframe::egui;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
    nn::critics::scalar::Value,
};
use rl_viewer::{ui::View, value::view::ValueView};
use std::time::{Duration, Instant};
use support::Harness;
fn fixture(dim: usize) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest =
        RunManifest::new("value-ui", "ordinary", "PPO", dim, space.clone(), 0.9).unwrap();
    let mut w =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut ctx = RecordContext::new("value-ui", 3);
    ctx.probe_set = Some("fixed".into());
    w.append(
        &ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "actual inputs",
            vec![Observation::new(vec![0.; dim])],
            None,
            space.clone(),
        )
        .unwrap()
        .record(ctx)
        .unwrap(),
    )
    .unwrap();
    for rev in [3, 7] {
        let mut b = SnapshotBuilder::new(
            "actual",
            RecordContext::new("value-ui", rev),
            dim,
            space.clone(),
        )
        .unwrap();
        let value = Value::new(dim, &[4], &mut ChaCha8Rng::seed_from_u64(rev)).unwrap();
        b.value("baseline", &value).unwrap();
        w.append(&b.finish().unwrap()).unwrap();
    }
    w.finish().unwrap();
    dir
}
fn open(dim: usize, width: f32) -> (tempfile::TempDir, Harness) {
    let d = fixture(dim);
    let mut h = Harness::new(&d, egui::vec2(width, 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    (d, h)
}
fn settle(h: &mut Harness) {
    let until = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if h.app.value_view().ready() {
            h.frame(vec![]);
            break;
        }
        assert!(
            Instant::now() < until,
            "status={:?} metrics={:?} view={} grid={:?}",
            h.app.value_view().status(),
            h.app.value_metrics(),
            h.app.value_view().state_reservation_bytes(),
            h.app.value_frame().map(|f| (
                f.grid.values.len(),
                f.grid.slice.resolution,
                f.grid.current().identity().clone()
            ))
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}
#[test]
fn value_route_uses_shared_navigation_and_compact_selected_inspector() {
    assert_eq!(View::from_startup_hook("value"), View::Value);
    assert_eq!(View::from_startup_hook("policy"), View::PolicyMap);
    assert_eq!(View::from_startup_hook("network"), View::Network);
    let (_d, mut h) = open(3, 1440.);
    assert!(h.has("Value") && h.has("Selected") && h.has("Expected return"));
    assert!(h.has("Value map · Expected return"));
    let p = h.app.value_view().main_rect.unwrap();
    let i = h.app.value_view().inspector_rect.unwrap();
    assert!((280.0..=320.0).contains(&i.width()), "{i:?}");
    assert!(p.width() > i.width() * 2.5);
    assert!(i.right() <= 1440. && i.bottom() <= 900.);
    for destination in ["Policy", "Value", "Network"] {
        assert!(h.text.iter().any(|(text, _)| text == destination));
    }
    let selected_input = h.app.selection().input.clone();
    let revision = h.app.selection().revision;
    h.click("Policy");
    // Policy mode discovery and source loading run in the background. A click's
    // three paint frames need not finish them on a slower CI runner.
    let deadline = Instant::now() + Duration::from_secs(20);
    while h.app.policy_pane().config().is_none() {
        h.frame(vec![]);
        assert!(
            Instant::now() < deadline,
            "Policy configuration did not finish: {:?}",
            h.app.policy_pane().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    h.frame(vec![]);
    assert!(h.app.policy_pane().config().is_some());
    assert!(h.has("Action") && !h.has("Expected return"));
    assert!(h.app.value_frame().is_none());
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert_eq!(h.app.selection().revision, revision);
    assert_eq!(h.app.selection().input, selected_input);
}
#[test]
fn one_dimensional_controls_hide_y_and_scalar_has_no_distribution() {
    let (_d, mut h) = open(3, 1440.);
    h.click("Slice");
    settle(&mut h);
    assert!(h.app.value_view().slice().unwrap().y.is_none());
    assert!(!h.has("Y axis"));
    assert!(!h.has("Return profile") && !h.has("Density") && !h.has("Probability"));
}
#[test]
fn narrow_layout_stacks_and_large_fixed_editor_starts_collapsed() {
    let (_d, h) = open(32, 700.);
    let p = h.app.value_view().main_rect.unwrap();
    assert!(p.right() <= 700. && p.bottom() <= 900.);
    assert!(h.has("Fixed inputs"));
    assert!(h.app.value_view().inspector_rect.is_none());
}
#[test]
fn exact_off_grid_pin_reuses_grid_and_route_departure_retires_view_texture() {
    let (_d, mut h) = open(3, 1440.);
    let grid = h.app.value_frame().unwrap().grid.clone();
    let input = Observation::new(vec![0.12345, 0.23456, 0.]);
    h.app
        .value_view_mut()
        .pin_input(input.clone(), None)
        .unwrap();
    settle(&mut h);
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        input
    );
    assert!(std::sync::Arc::ptr_eq(
        &grid,
        &h.app.value_frame().unwrap().grid
    ));
    assert_eq!(h.app.value_view().texture_uploads, 1);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
}
#[test]
fn checkpoint_change_preserves_exact_pin_and_scale() {
    let (_d, mut h) = open(3, 1440.);
    let input = Observation::new(vec![0.12345, 0.23456, 0.]);
    h.app
        .value_view_mut()
        .pin_input(input.clone(), None)
        .unwrap();
    settle(&mut h);
    let scale = h.app.value_view().color_limits();
    h.app.select_revision(3).unwrap();
    settle(&mut h);
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        input
    );
    assert_eq!(h.app.value_view().color_limits(), scale);
}
#[test]
fn shared_layout_bounds_and_physical_cell_mapping_are_stable() {
    let l = rl_viewer::landscape::layout(1200., 700., true);
    assert!(!l.stacked);
    assert!((l.main_width / (l.main_width + l.selected_width) - 0.73).abs() < 0.001);
    assert_eq!(
        rl_viewer::landscape::cell_index([0.5, 1.], [[0., 1.], [0., 2.]], [3, 3]),
        Some(4)
    );
    assert_eq!(
        rl_viewer::landscape::cell_index([-0.01, 1.], [[0., 1.], [0., 2.]], [3, 3]),
        None
    );
    assert_eq!(
        rl_viewer::landscape::numeric_bounds([2., 5.].into_iter(), true),
        [-5., 5.]
    );
    let _ = ValueView::main_plot_id();
}
#[test]
fn automatic_value_resolution_stays_fixed_across_real_checkpoint_replacements() {
    let (_d, mut h) = open(3, 1440.);
    let resolution = h.app.value_frame().unwrap().grid.slice.resolution;
    for revision in [3, 7, 3, 7] {
        h.app.select_revision(revision).unwrap();
        settle(&mut h);
        let frame = h.app.value_frame().unwrap();
        assert_eq!(frame.grid.current().identity().revision, revision);
        assert_eq!(
            frame.grid.slice.resolution, resolution,
            "revision={revision}"
        );
        assert!(h.app.value_metrics().prepared_peak <= 4 * 1024 * 1024);
    }
}

#[test]
fn ordinary_value_hides_advanced_numerics_and_keeps_linear_grid() {
    let (_d, mut h) = open(3, 1440.);
    assert!(
        !h.text
            .iter()
            .any(|(s, r)| s == "Advanced" && r.top() > h.find("Value").y),
        "advanced parameters belong in contextual settings"
    );
    assert!(!h.has("Samples") && !h.has("Fixed-input tolerance"));
    {
        let manager = h.ctx.tex_manager();
        let manager = manager.read();
        let texture = manager
            .allocated()
            .find(|(_, texture)| texture.name == "Value")
            .unwrap()
            .1;
        assert_eq!(texture.options, egui::TextureOptions::LINEAR);
        assert_eq!(texture.size, [48, 48]);
    }
    // Automatic admission leaves room for a replacement frame and its successor.
    assert_eq!(h.app.value_frame().unwrap().grid.slice.resolution, [48, 48]);
    let before = h.app.value_metrics();
    let uploads = h.app.value_view().texture_uploads;
    for _ in 0..12 {
        h.frame(vec![]);
    }
    let after = h.app.value_metrics();
    // Neighbor model warmup can finish after the visible frame is ready.
    // Idle repaint must not reevaluate the grid or selected input.
    assert_eq!(
        (after.grid_batches, after.point_batches),
        (before.grid_batches, before.point_batches)
    );
    assert_eq!(h.app.value_view().texture_uploads, uploads);
    assert!(after.prepared_peak <= 4 * 1024 * 1024);
}
#[test]
fn automatic_reference_and_visits_keep_two_real_successors_admitted_and_idle() {
    let original = families::family("twin", true);
    let run = rl_viewer::data::RunIndex::open(original.path().join("run"), ReadLimits::default())
        .unwrap();
    let d = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        d.path().join("run"),
        run.recording().manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for index in 0..run.recording().entries().len() {
        writer
            .append(&run.recording().load(index).unwrap())
            .unwrap();
    }
    // A third actual serialized snapshot identity, with deliberately unchanged
    // weights. This is a controlled publication fixture, never a trained demo.
    let mut third = run.recording().load(2).unwrap();
    let mut declaration = rl::diagnostics::value::declaration(&third)
        .unwrap()
        .unwrap();
    third.context.identity.learner_update_revision = 11;
    third.context.counters.updates = 11 - run.recording().manifest().starting_revision;
    declaration.revision = 11;
    third
        .arrays
        .remove(rl::diagnostics::value::VALUE_METADATA_KEY);
    rl::diagnostics::value::attach_declaration(&mut third, &declaration).unwrap();
    writer.append(&third).unwrap();
    writer.finish().unwrap();
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    h.click("Visited states");
    settle(&mut h);
    h.click_last("Reference");
    settle(&mut h);
    set_reference_request(&mut h, 3);
    settle(&mut h);
    let input = h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    let limits = h.app.value_view().color_limits();
    for revision in [7, 3, 7, 11, 7] {
        let start = Instant::now();
        h.app.select_revision(revision).unwrap();
        settle(&mut h);
        let frame = h.app.value_frame().unwrap();
        assert_eq!(frame.grid.current().identity().revision, revision);
        assert_eq!(frame.grid.reference().unwrap().identity().revision, 3);
        assert_eq!(frame.grid.slice.resolution, [48, 48]);
        assert_eq!(frame.selected.as_ref().unwrap().input, input);
        assert!(frame.grid.visits.is_some());
        assert_eq!(h.app.value_view().color_limits(), limits);
        let latency = start.elapsed();
        let before = h.app.value_metrics();
        let uploads = h.app.value_view().texture_uploads;
        for _ in 0..12 {
            h.frame(vec![]);
        }
        let after = h.app.value_metrics();
        assert_eq!(
            (after.restores, after.grid_batches, after.point_batches),
            (before.restores, before.grid_batches, before.point_batches)
        );
        assert_eq!(h.app.value_view().texture_uploads, uploads);
        eprintln!(
            "APP revision={revision} resolution=48x48 latency_us={} uploads={uploads} retained={} peak={} idle_grid_batches_delta={} idle_point_batches_delta={}",
            latency.as_micros(),
            after.prepared_bytes,
            after.prepared_peak,
            after.grid_batches - before.grid_batches,
            after.point_batches - before.point_batches
        );
        assert!(after.prepared_peak <= 4 * 1024 * 1024);
    }
    h.app.select_view(View::PolicyMap);
    assert!(
        h.app.value_frame().is_none(),
        "Value publication must be cleared on the workspace switch"
    );
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    // The related Policy workspace may retain a charged catalogue or a
    // compatible Policy frame. Leaving both workspaces must retire them too.
    h.app.select_view(View::Dynamics);
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.app.value_metrics().prepared_bytes != 0 {
        h.frame(vec![]);
        assert!(Instant::now() < deadline, "all prepared owners must retire");
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[path = "support/value_ui.rs"]
mod families;
use rl::diagnostics::value::{ValueOutput, ValuePrediction};
use rl_viewer::value::{prepared::PreparedValueProfile, slice::ValueCoordinate as C};
fn open_family(kind: &str, known: bool) -> (tempfile::TempDir, Harness) {
    let d = families::family(kind, known);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    (d, h)
}
#[test]
fn legacy_typed_critics_are_discovered_with_distinct_immutable_choices() {
    let d = families::discovery(false, false, false, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.values, vec![11., 13.]);
    assert_eq!(f.grid.current().identity().name, "Recorded critic #2");
    assert!(f.grid.current().declaration().is_none());
    assert_eq!(
        f.selected.as_ref().unwrap().input,
        Observation::dense(vec![0.; 3], Some(vec![true, false, true, false]))
    );
    assert!(!h.has("Utility"));
    h.click("Recorded critic #2");
    h.click("Recorded critic #1");
    settle(&mut h);
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.current().identity().name, "Recorded critic #1");
    assert_ne!(f.grid.values, vec![11., 13.]);
    h.click_last("Reference");
    settle(&mut h);
    assert!(
        h.app
            .value_frame()
            .unwrap()
            .grid
            .reference_values
            .iter()
            .all(Option::is_none)
    );
}
#[test]
fn masked_source_switches_model_samples_and_back_without_losing_held_frames() {
    let d = families::discovery(true, true, false, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let held = h.app.value_frame().unwrap().grid.clone();
    assert!(held.regular());
    for (x, y, model) in [
        (C::Observation(0), None, false),
        (C::Observation(0), Some(C::DiscreteAction), false),
        (C::DiscreteAction, None, true),
    ] {
        h.app.value_view_mut().set_axes(x, y).unwrap();
        settle(&mut h);
        let f = h.app.value_frame().unwrap();
        assert_eq!(f.grid.regular(), model);
        assert_eq!((f.grid.slice.x, f.grid.slice.y), (x, y));
        assert!(!f.grid.values.is_empty());
        assert_eq!(
            f.selected.as_ref().unwrap().input.dense_mask(),
            Some(&[true, false, true, false][..])
        );
        if !model {
            assert!(f.grid.rows.iter().all(|r| *r == 0));
            assert_eq!(f.grid.values.len(), if y.is_some() { 2 } else { 1 });
        }
    }
    assert_eq!(held.values.len(), 2);
    h.app.select_view(View::PolicyMap);
    h.frame(vec![]);
    assert_eq!(h.app.value_view().state_reservation_bytes(), 0);
    assert_eq!(
        held.input(0).unwrap().dense_mask(),
        Some(&[true, false, true, false][..])
    );
}
#[test]
fn retained_legacy_model_falls_back_to_unnamed_samples_without_claiming_the_model_name() {
    let d = families::discovery(true, false, false, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    assert!(h.app.value_frame().unwrap().grid.regular());
    assert_eq!(
        h.app.value_frame().unwrap().grid.current().identity().name,
        "actions"
    );
    h.app
        .value_view_mut()
        .set_axes(C::Observation(0), None)
        .unwrap();
    settle(&mut h);
    let g = &h.app.value_frame().unwrap().grid;
    assert!(!g.regular());
    assert_eq!(g.values, vec![13.]);
    assert_eq!(g.current().identity().name, "Recorded critic #3");
    assert!(g.current().declaration().is_none());
    assert!(h.has("Recorded critic #3"));
    h.app
        .value_view_mut()
        .set_axes(C::DiscreteAction, None)
        .unwrap();
    settle(&mut h);
    assert!(h.app.value_frame().unwrap().grid.regular());
    assert_eq!(
        h.app.value_frame().unwrap().grid.current().identity().name,
        "actions"
    );
}
#[test]
fn history_bound_input_uses_genuine_recorded_critic_with_retained_model_present() {
    let d = families::discovery(true, true, true, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let f = h.app.value_frame().unwrap();
    assert!(!f.grid.regular());
    assert_eq!(f.grid.rows, vec![0, 0]);
    assert_eq!(f.selected.as_ref().unwrap().recorded_row, Some(0));
}
#[test]
fn masked_slice_without_compatible_samples_is_explicitly_unavailable() {
    let d = families::discovery(true, true, false, false);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let held = h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .clone();
    h.app
        .value_view_mut()
        .set_axes(C::Observation(0), None)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        h.frame(vec![]);
        if h.app
            .value_view()
            .status()
            .is_some_and(|s| s.contains("recorded"))
            && !h.app.value_pending()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{:?}",
            h.app.value_view().status()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(std::sync::Arc::ptr_eq(
        h.app.value_frame().unwrap().selected.as_ref().unwrap(),
        &held
    ));
    assert!(!h.app.value_view().ready());
    assert!(h.has("Showing previous complete frame"));
}
fn hover_plot(h: &mut Harness, xy: [f64; 2]) -> String {
    let p = egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id())
        .unwrap()
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(xy[0], xy[1]));
    h.frame(vec![egui::Event::PointerMoved(p)]);
    for _ in 0..12 {
        h.frame(vec![]);
    }
    assert!(
        !h.tooltip_text.is_empty(),
        "tooltip paint missing at {xy:?} screen={p:?} main={:?} text={:?}",
        h.app.value_view().main_rect,
        h.text
    );
    h.tooltip_text
        .iter()
        .map(|(s, _)| s.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}
#[test]
fn comparison_tooltip_and_selected_disclose_actual_sources_continuations_and_actions() {
    use rl::core::action::Action;
    let (_d, mut h) = open_family("quantile", true);
    h.app
        .value_view_mut()
        .pin_input(
            Observation::new(vec![0.; 3]),
            Some(Action::Continuous(vec![0.125, 0.375])),
        )
        .unwrap();
    settle(&mut h);
    h.click_last("Reference");
    settle(&mut h);
    set_reference_request(&mut h, 5);
    settle(&mut h);
    assert!(
        !h.text.iter().any(|(s, _)| s == "k"),
        "reference popup reopened after the updated source finished loading"
    );
    let text = hover_plot(&mut h, [0.31, 0.27]);
    for expected in [
        "Current",
        "distribution",
        "online",
        "actual k=7 / requested k=7",
        "actual k=3 / requested k=5",
        "controller-7",
        "controller-3",
        "deterministic",
        "action[0] = 0.125",
        "action[1] = 0.375",
        "Reference",
        "difference",
    ] {
        assert!(text.contains(expected), "missing {expected}: {text}");
    }
    assert!(!text.contains("Some(") && !text.contains("Continuous("));
    let p = h
        .text
        .iter()
        .rev()
        .find(|(s, r)| {
            s == "Expected return"
                && h.app
                    .value_view()
                    .inspector_rect
                    .unwrap()
                    .contains(r.center())
        })
        .unwrap()
        .1
        .center();
    h.frame(vec![egui::Event::PointerMoved(p)]);
    for _ in 0..12 {
        h.frame(vec![]);
    }
    let text = h
        .tooltip_text
        .iter()
        .map(|(s, _)| s.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        text.contains("actual k=3 / requested k=5") && text.contains("controller-3"),
        "selected tooltip missing at {p:?}, inspector={:?}, pointer layer={:?}, text={text:?}, paint={:?}",
        h.app.value_view().inspector_rect,
        h.ctx.layer_id_at(p),
        h.text
    );
    assert!(text.contains("action[1] = 0.375"), "{text}");
}
#[test]
fn actor_hover_discloses_only_real_off_slice_projection() {
    use rl::core::action::Action;
    let (_d, mut h) = open_family("quantile", true);
    h.app
        .value_view_mut()
        .pin_input(
            Observation::new(vec![0.; 3]),
            Some(Action::Continuous(vec![0., 3.])),
        )
        .unwrap();
    h.app
        .value_view_mut()
        .set_axes(C::Observation(0), Some(C::Action(0)))
        .unwrap();
    settle(&mut h);
    let g = &h.app.value_frame().unwrap().grid;
    let index = 32;
    let Action::Continuous(a) = g.policy_overlay[index].as_ref().unwrap() else {
        panic!()
    };
    let xy = [g.positions[index][0] as f64, a[0] as f64];
    let action = a.clone();
    let text = hover_plot(&mut h, xy);
    assert!(
        text.contains("component projection") && text.contains("fixed remaining"),
        "{text}"
    );
    assert!(
        text.contains("action[1]") && !text.contains("stochastic representative"),
        "{text}"
    );
    h.app
        .value_view_mut()
        .pin_input(
            Observation::new(vec![0.; 3]),
            Some(Action::Continuous(action)),
        )
        .unwrap();
    settle(&mut h);
    let text = hover_plot(&mut h, xy);
    assert!(
        !text.contains("component projection") && !text.contains("stochastic representative"),
        "{text}"
    );
}
#[test]
fn stochastic_actor_hover_identifies_the_complete_representative_action() {
    use rl::core::action::Action;
    let d = families::family_with_actor("quantile", true, 16, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.app.value_view().slice().is_none() {
        h.frame(vec![]);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    h.app
        .value_view_mut()
        .pin_input(
            Observation::new(vec![0.; 3]),
            Some(Action::Continuous(vec![0., 3.])),
        )
        .unwrap();
    settle(&mut h);
    h.app
        .value_view_mut()
        .set_axes(C::Observation(0), Some(C::Action(0)))
        .unwrap();
    settle(&mut h);
    let g = &h.app.value_frame().unwrap().grid;
    let Action::Continuous(a) = g.policy_overlay[32].as_ref().unwrap() else {
        panic!()
    };
    let xy = [g.positions[32][0] as f64, a[0] as f64];
    let text = hover_plot(&mut h, xy);
    assert!(
        text.contains("Stochastic representative action") && text.contains("component projection"),
        "{text}"
    );
    assert!(
        text.contains("action[0]") && text.contains("action[1]"),
        "{text}"
    );
}
#[test]
fn independent_reference_controller_sample_hover_keeps_actual_action_and_excludes_difference() {
    let d = families::discrete_populations();
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    h.app
        .value_view_mut()
        .set_axes(C::Observation(0), None)
        .unwrap();
    settle(&mut h);
    h.click_last("Reference");
    settle(&mut h);
    set_reference_request(&mut h, 3);
    settle(&mut h);
    h.app.value_view_mut().set_color_limits([-2., 2.]).unwrap();
    // Manual colour limits remove the provisional-scale row; use current control positions.
    h.frame(vec![]);
    h.click("Slice settings");
    h.click("Fit axes");
    h.click("Slice settings");
    h.frame(vec![]);
    let g = &h.app.value_frame().unwrap().grid;
    let series = g.reference_series.as_ref().unwrap();
    assert_eq!(series.values.len(), 2);
    assert_eq!(g.reference_values.len(), 1);
    let xy = [series.positions[0][0] as f64, series.values[0]];
    let action = series.actions[0].as_ref().unwrap().clone();
    let row = series.rows[0];
    let text = hover_plot(&mut h, xy);
    assert!(
        text.contains("Reference controller samples · independent inputs"),
        "{text}"
    );
    assert!(text.contains("excluded from paired differences"), "{text}");
    assert!(text.contains(&format!("Recorded row {row}")), "{text}");
    let rl::core::action::Action::Discrete(index) = action else {
        panic!()
    };
    assert!(text.contains(&format!("index {index}")), "{text}");
}
#[test]
fn td3_default_uses_actual_declared_index_zero_name_and_outputs_both_critics() {
    let (_d, mut h) = open_family("twin", true);
    let g = &h.app.value_frame().unwrap().grid;
    assert_eq!(g.current().critic_names(), &["omega", "alpha"]);
    assert!(matches!(g.output(), ValueOutput::Q1));
    assert!(h.has("Q1") && h.has("Disagreement"));
    h.click_last("Reference");
    settle(&mut h);
    let inspector = h.app.value_view().inspector_rect.unwrap();
    assert!(
        !h.text
            .iter()
            .any(|(s, r)| s == "Output" && inspector.contains(r.center())),
        "selected Q1 is already named by its critic card"
    );
    assert_eq!(
        h.text
            .iter()
            .filter(|(s, r)| s == "Q1" && inspector.contains(r.center()))
            .count(),
        1
    );
    assert!(h.has("Q2") && h.has("Difference"));
    h.click("Q1");
    h.click("Min(Q1, Q2)");
    settle(&mut h);
    assert!(matches!(
        h.app.value_frame().unwrap().grid.output(),
        ValueOutput::Minimum
    ));
}
#[test]
fn quantile_parameters_and_full_ordered_profile_and_reference_are_real() {
    let (_d, mut h) = open_family("quantile", true);
    assert!(h.has("Return profile") && h.has("CDF"));
    h.click("Expected return");
    h.click("Quantile");
    settle(&mut h);
    assert!(matches!(
        h.app.value_frame().unwrap().grid.output(),
        ValueOutput::Quantile { .. }
    ));
    assert!(h.has("p ="));
    h.click_last("Reference");
    settle(&mut h);
    let p = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert!(p.reference_value.is_some());
    assert!(matches!(
        p.reference_profile,
        Some(PreparedValueProfile::Quantiles { .. })
    ));
}
#[test]
fn conditional_profile_remains_reachable_when_the_window_shrinks() {
    let (_d, mut h) = open_family("quantile", true);
    h.click_last("Reference");
    settle(&mut h);
    let selected = h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    for size in [egui::vec2(1280., 720.), egui::vec2(1920., 1080.)] {
        h.size = size;
        settle(&mut h);
        let inspector = h.app.value_view().inspector_rect.unwrap();
        assert!(
            (400.0..=530.0).contains(&inspector.width()),
            "{inspector:?}"
        );
        assert!(
            inspector.bottom() <= size.y + 1. && inspector.right() <= size.x + 1.,
            "{inspector:?}"
        );
        let return_axis = h
            .text
            .iter()
            .find(|(s, r)| s == "Return" && r.center().x > inspector.left())
            .unwrap()
            .1;
        assert!(
            return_axis.bottom() <= size.y && return_axis.top() >= inspector.top(),
            "CDF return axis is clipped: {return_axis:?}; inspector {inspector:?}"
        );
        assert_eq!(
            h.app
                .value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .input,
            selected
        );
    }
    h.size = egui::vec2(700., 720.);
    settle(&mut h);
    assert!(h.app.value_view().inspector_rect.is_none());
    h.click("Selected");
    settle(&mut h);
    let inspector = h.app.value_view().inspector_rect.unwrap();
    assert!(inspector.bottom() <= 721.);
    let pointer = inspector.left_top() + egui::vec2(8., 30.);
    for _ in 0..4 {
        h.frame(vec![
            egui::Event::PointerMoved(pointer),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                phase: egui::TouchPhase::Move,
                delta: egui::vec2(0., -180.),
                modifiers: egui::Modifiers::NONE,
                source: egui::MouseWheelSource::Unknown,
            },
        ]);
    }
    let return_axis = h.text.iter().find(|(s, _)| s == "Return").unwrap().1;
    assert!(
        return_axis.bottom() <= 720. && return_axis.top() >= inspector.top(),
        "CDF must remain reachable by scrolling: {return_axis:?}"
    );
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        selected
    );
}
#[test]
fn full_cdf_draws_atomic_steps_without_band_aggregation() {
    fn check(shapes: &[egui::Shape], rect: egui::Rect) -> usize {
        let mut steps = 0;
        for shape in shapes {
            match shape {
                egui::Shape::Band(b) => assert!(
                    !rect.contains_rect(b.visual_bounding_rect()),
                    "atomic CDF must not become an interpolated band"
                ),
                egui::Shape::Path(p)
                    if p.points.len() > 2 && p.points.iter().all(|p| rect.contains(*p)) =>
                {
                    if matches!(p.stroke.color, egui::epaint::ColorMode::Solid(c) if c == rl_viewer::theme::CURRENT)
                    {
                        for pair in p.points.windows(2) {
                            assert!(
                                pair[0].x == pair[1].x || pair[0].y == pair[1].y,
                                "CDF must retain horizontal/vertical atomic steps"
                            );
                        }
                        steps += p.points.len();
                    }
                }
                egui::Shape::Vec(v) => steps += check(v, rect),
                _ => {}
            }
        }
        steps
    }
    let d = families::family_with_heads("quantile", true, 16);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let rect = h.app.value_view().main_rect.unwrap();
    h.click_at(rect.center() + egui::vec2(100., 50.));
    settle(&mut h);
    let count = match &h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .profile
    {
        Some(PreparedValueProfile::Quantiles { cdf, .. }) => cdf.len(),
        _ => panic!("missing CDF"),
    };
    assert_eq!(
        check(&h.shapes, h.app.value_view().inspector_rect.unwrap()),
        count
    );
    h.click_last("Reference");
    settle(&mut h);
    assert_eq!(
        check(&h.shapes, h.app.value_view().inspector_rect.unwrap()),
        count
    );
}
#[test]
fn main_value_ticks_leave_room_for_negative_fractional_labels() {
    let (_d, mut h) = open_family("quantile", true);
    let id = ValueView::main_plot_id();
    let mut memory = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
    memory.set_bounds(egui_plot::PlotBounds::from_min_max([-3., -1.], [-1., 1.]));
    memory.auto_bounds = egui::Vec2b::FALSE;
    memory.store(&h.ctx, id);
    h.frame(vec![]);
    h.frame(vec![]);
    let frame = *egui_plot::PlotMemory::load(&h.ctx, id)
        .unwrap()
        .transform()
        .frame();
    let mut ticks: Vec<_> = h
        .text
        .iter()
        .filter(|(s, r)| {
            s.parse::<f64>().is_ok()
                && r.top() >= frame.bottom()
                && r.top() < frame.bottom() + 18.
                && r.center().x >= frame.left()
                && r.center().x <= frame.right()
        })
        .map(|(_, r)| *r)
        .collect();
    ticks.sort_by(|a, b| a.left().total_cmp(&b.left()));
    assert!(ticks.len() >= 3);
    for pair in ticks.windows(2) {
        assert!(
            pair[1].center().x - pair[0].center().x >= 90.,
            "crowded ticks: {pair:?}"
        );
    }
}
#[test]
fn moment_cards_preserve_signed_variance_and_actual_utility_without_density() {
    let (_d, mut h) = open_family("moments", true);
    assert!(h.has("Signed variance") && h.has("Negative variance") && h.has("Declared utility"));
    let variance = h
        .text
        .iter()
        .find(|(s, _)| s.starts_with("Signed variance"))
        .unwrap()
        .1
        .center();
    h.frame(vec![egui::Event::PointerMoved(variance)]);
    for _ in 0..12 {
        h.frame(vec![]);
    }
    assert!(
        h.tooltip_text
            .iter()
            .any(|(s, _)| s.contains("Raw second moment 3.")),
        "{:?}",
        h.tooltip_text
    );
    assert!(!h.has("Return profile") && !h.has("Density"));
    let p = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert!(matches!(
        p.profile,
        Some(PreparedValueProfile::Moments {
            utility: Some(2.),
            ..
        })
    ));
}
#[test]
fn ddqn_bars_keep_actual_legal_categories_and_no_continuous_identity() {
    let (_d, mut h) = open_family("discrete", true);
    let g = &h.app.value_frame().unwrap().grid;
    assert_eq!(g.categories, vec![0, 2]);
    assert_eq!(g.values.len(), 2);
    assert!(h.has("Action category") && h.has("Legal greedy action"));
    assert_eq!(h.app.value_view().texture_bytes(), 0);
    assert!(matches!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .prediction,
        ValuePrediction::Discrete { .. }
    ));
    let id = ValueView::main_plot_id();
    let mut memory = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
    let rect = *memory.transform().frame();
    for value in std::iter::once(0.).chain(g.values.iter().copied()) {
        let p = memory
            .transform()
            .position_from_point(&egui_plot::PlotPoint::new(0., value));
        assert!(
            p.y > rect.top() + 5. && p.y < rect.bottom() - 5.,
            "bar endpoint or zero baseline clipped: {value} {p:?} {rect:?}"
        );
    }
    let zoom = egui_plot::PlotBounds::from_min_max([-0.5, 0.25], [2.5, 0.75]);
    memory.set_bounds(zoom);
    memory.auto_bounds = egui::Vec2b::FALSE;
    memory.store(&h.ctx, id);
    h.frame(vec![]);
    assert_eq!(
        *egui_plot::PlotMemory::load(&h.ctx, id).unwrap().bounds(),
        zoom
    );
}
#[test]
fn unknown_scientific_conventions_disable_value_reference_without_losing_policy_overlay() {
    let (_d, mut h) = open_family("quantile", false);
    h.click_last("Reference");
    settle(&mut h);
    let g = &h.app.value_frame().unwrap().grid;
    assert!(g.reference_error.is_some());
    assert!(g.reference_values.is_empty());
    assert!(!g.reference_policy_overlay.is_empty());
    assert!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_profile
            .is_none()
    );
}
#[test]
fn all_joint_axis_types_and_exact_actions_retain_complete_inputs() {
    let (_d, mut h) = open_family("twin", true);
    for (x, y) in [
        (C::Observation(0), Some(C::Action(0))),
        (C::Action(1), Some(C::Observation(2))),
        (C::Action(1), Some(C::Action(0))),
        (C::Action(0), None),
    ] {
        h.app.value_view_mut().set_axes(x, y).unwrap();
        settle(&mut h);
        let g = &h.app.value_frame().unwrap().grid;
        assert_eq!(g.slice.x, x);
        assert_eq!(g.slice.y, y);
        assert_eq!(g.input(0).unwrap().dense_values().len(), 3);
        assert!(
            g.actions
                .iter()
                .all(|a| matches!(a,Some(rl::core::action::Action::Continuous(v))if v.len()==2))
        );
        let grid = g.clone();
        h.app
            .value_view_mut()
            .select_cell(&grid, grid.values.len() / 2)
            .unwrap();
        settle(&mut h);
        assert!(matches!(
            h.app.value_view().slice().unwrap().conditioning,
            rl_viewer::value::slice::ActionConditioning::Fixed(_)
        ));
    }
}
#[test]
fn visit_marker_precedes_grid_pin_and_coincident_decisions_keep_individual_returns() {
    let (_d, mut h) = open_family("twin", true);
    h.click("Visited states");
    settle(&mut h);
    let frame = h.app.value_frame().unwrap();
    assert!(frame.grid.visits.as_ref().unwrap().markers.len() >= 4);
    let p = egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id())
        .unwrap()
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(0., 0.));
    h.click_at(p);
    settle(&mut h);
    let first = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert!(first.observed_return.is_some());
    assert!(h.has("Return-to-go") && h.has("γ=0.9"));
    let identity = (first.recorded_key.clone(), first.recorded_row);
    h.click_at(p);
    settle(&mut h);
    let second = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert_ne!(identity, (second.recorded_key.clone(), second.recorded_row));
    assert_eq!(second.recorded_revision, Some(3));
    assert_eq!(h.app.value_frame().unwrap().requested_revision, 7);
}
#[test]
fn comparison_difference_is_zero_centered_and_locked_through_checkpoint_playback() {
    let (_d, mut h) = open_family("quantile", true);
    h.click_last("Reference");
    settle(&mut h);
    h.click("Difference");
    h.frame(vec![]);
    let range = h.app.value_view().color_limits().unwrap();
    assert_eq!(range[0], -range[1]);
    h.app.select_revision(3).unwrap();
    settle(&mut h);
    assert_eq!(h.app.value_view().color_limits(), Some(range));
}
#[test]
fn repaint_does_no_model_restoration_or_numerical_preparation_and_retains_texture() {
    let (_d, mut h) = open_family("quantile", true);
    let deadline = Instant::now() + Duration::from_secs(20);
    while !h.app.snapshot_preparation_idle() {
        h.frame(vec![]);
        assert!(
            Instant::now() < deadline,
            "catalog discovery did not settle"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let before = h.app.value_metrics();
    let uploads = h.app.value_view().texture_uploads;
    for _ in 0..10 {
        h.frame(vec![]);
    }
    let after = h.app.value_metrics();
    assert_eq!(
        (
            before.restores,
            before.grid_batches,
            before.point_batches,
            before.preparation_micros
        ),
        (
            after.restores,
            after.grid_batches,
            after.point_batches,
            after.preparation_micros
        )
    );
    assert_eq!(h.app.value_view().texture_uploads, uploads);
    assert!(h.app.value_view().owned_state_bytes() <= h.app.value_view().state_reservation_bytes());
}
#[test]
fn illegal_category_gap_has_no_numerical_hover_or_pin() {
    let (_d, mut h) = open_family("discrete", true);
    let memory = egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id()).unwrap();
    let pos = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(1., 0.));
    h.frame(vec![egui::Event::PointerMoved(pos)]);
    assert!(
        !h.tooltip_text
            .iter()
            .any(|(text, _)| text.contains("actual k=")),
        "an illegal category gap must not invent a numerical hover source"
    );
    let before = h.app.value_metrics().point_batches;
    h.click_at(pos);
    h.frame(vec![]);
    assert_eq!(h.app.value_metrics().point_batches, before);
}
#[test]
fn sparse_recorded_value_uses_points_and_does_not_pin_a_missing_middle_input() {
    let d = families::sparse();
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    h.click("Slice");
    settle(&mut h);
    let f = h.app.value_frame().unwrap();
    assert!(!f.grid.regular());
    assert_eq!(f.grid.values.len(), 2);
    let before = f.selected.as_ref().unwrap().input.clone();
    let memory = egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id()).unwrap();
    let rect = *memory.transform().frame();
    fn dense(shapes: &[egui::Shape], r: egui::Rect) -> bool {
        shapes.iter().any(|s|match s{egui::Shape::Path(p)=>p.points.len()>1&&matches!(p.stroke.color,egui::epaint::ColorMode::Solid(c)if c==rl_viewer::theme::CURRENT)&&p.points.iter().all(|p|r.contains(*p)),egui::Shape::Vec(v)=>dense(v,r),_=>false})
    }
    assert!(!dense(&h.shapes, rect));
    let p = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(0., 0.));
    h.click_at(p);
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        before
    );
    assert!(
        !h.tooltip_text
            .iter()
            .any(|(text, _)| text.contains("actual k=")),
        "unmeasured inputs must not invent a numerical hover source"
    );
}
#[test]
fn maximum_full_selected_current_and_reference_profiles_fit_shared_admission() {
    use rl::core::action::Action;
    use rl_viewer::{
        data::RunIndex,
        interactive::Interactive,
        value::{
            prepared::{ValuePin, ValueRequest, ValueSource},
            slice::{ActionConditioning, ValueSlice},
        },
    };
    let d = families::family_with_heads("quantile", true, 4096);
    let run = RunIndex::open(d.path().join("run"), ReadLimits::default()).unwrap();
    let chunk = run.recording().load(2).unwrap();
    let source = ValueSource::model(&run, 2, "distribution", &chunk).unwrap();
    let input = Observation::new(vec![0.; 3]);
    let mut slice = ValueSlice::new(source.semantics(), &input, &[]).unwrap();
    slice.resolution = [2, 2];
    slice.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![0.; 2]));
    let mut r = ValueRequest::new(
        source.clone(),
        ValueOutput::ExpectedReturn,
        slice,
        ValuePin::Input {
            observation: input,
            action: Some(Action::Continuous(vec![0.; 2])),
        },
    )
    .unwrap();
    r.reference_value = Some(source);
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_value(r).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        worker.poll_value().unwrap();
        assert!(worker.value_error().is_none(), "{:?}", worker.value_error());
        if worker.value_frame().is_some_and(|f| f.selected.is_some()) {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let p = worker.value_frame().unwrap().selected.as_ref().unwrap();
    for profile in [&p.profile, &p.reference_profile] {
        assert!(matches!(profile,Some(PreparedValueProfile::Quantiles{cdf,..})if cdf.len()==8192));
    }
    assert!(worker.value_metrics().prepared_peak <= 4 * 1024 * 1024);
    assert!(worker.value_metrics().scratch_peak <= 4 * 1024 * 1024);
}
#[test]
fn value_only_reference_remains_available_without_any_policy_model() {
    let (_d, mut h) = open_family("state", true);
    h.click_last("Reference");
    settle(&mut h);
    let f = h.app.value_frame().unwrap();
    assert!(f.selected.as_ref().unwrap().reference_value.is_some());
    assert!(f.grid.reference_error.is_none());
    assert!(f.grid.reference_policy().is_none());
    assert!(f.grid.reference_policy_overlay.iter().all(Option::is_none));
    assert!(h.has("Difference"));
}
#[test]
fn no_critic_route_is_truthful_and_never_fabricates_selected_values() {
    let d = support::fixture(3, 0, true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if h.app.value_view().status().is_some_and(|s| s != "Loading…") && !h.app.value_pending()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(h.app.value_frame().is_none());
    assert!(h.has("Value"));
    assert!(
        h.app.value_view().inspector_rect.is_none(),
        "an absent prediction does not reserve a Selected column"
    );
    assert!(
        h.app
            .value_view()
            .status()
            .is_some_and(|s| s.contains("No value model") || s.contains("No compatible"))
    );
    assert!(!h.has("Return profile") && !h.has("Return-to-go"));
}
#[test]
fn one_dimensional_actor_uses_a_vertical_indicator_and_exact_min_label() {
    let (_d, mut h) = open_family("twin", true);
    h.app.value_view_mut().set_axes(C::Action(0), None).unwrap();
    settle(&mut h);
    let r = *egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id())
        .unwrap()
        .transform()
        .frame();
    fn vertical(s: &[egui::Shape], r: egui::Rect) -> bool {
        s.iter().any(|s| match s {
            egui::Shape::LineSegment { points, stroke } => {
                stroke.color == rl_viewer::theme::CURRENT
                    && (points[0].x - points[1].x).abs() < 0.1
                    && (points[0].y - points[1].y).abs() > r.height() * 0.8
            }
            egui::Shape::Path(p) => matches!(p.stroke.color,egui::epaint::ColorMode::Solid(c)if c==rl_viewer::theme::CURRENT) && p.points.len()>1 && (p.points[0].x-p.points[p.points.len()-1].x).abs()<0.1 && (p.points[0].y-p.points[p.points.len()-1].y).abs()>r.height()*0.8,
            egui::Shape::Vec(v) => vertical(v, r),
            _ => false,
        })
    }
    assert!(vertical(&h.shapes, r));
    h.click("Q1");
    assert!(h.has("Min(Q1, Q2)"));
}
#[test]
fn selecting_another_real_masked_state_recenters_fixed_chips_and_legal_categories() {
    let (_d, mut h) = open_family("discrete", true);
    h.click("Input");
    for _ in 0..5 {
        h.frame(vec![]);
    }
    let prefix = h.text.iter().find(|(s, _)| s == "state ").unwrap().1;
    let p = h
        .text
        .iter()
        .filter(|(s, r)| {
            s == "0"
                && r.center().x > prefix.right()
                && (r.center().y - prefix.center().y).abs() < 3.
        })
        .min_by(|a, b| a.1.left().total_cmp(&b.1.left()))
        .unwrap()
        .1
        .center();
    h.frame(vec![
        egui::Event::PointerMoved(p),
        support::button(p, true),
        support::button(p, false),
        support::button(p, true),
        support::button(p, false),
    ]);
    h.frame(vec![
        egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        },
        egui::Event::Text("1".into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    settle(&mut h);
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::ProbeRow { row: 1, .. }
    ));
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.slice.state.fixed, vec![0.25, 0.5, 0.75]);
    assert_eq!(f.grid.categories, vec![1, 3]);
    assert_eq!(
        f.selected.as_ref().unwrap().input.dense_mask(),
        Some(&[false, true, false, true][..])
    );
}

#[test]
fn sparse_reference_population_is_admitted_independently_of_current_grid() {
    let d = families::sparse_populations(2, 8192);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    assert_eq!(
        h.app.value_view().slice().unwrap().state.ranges[0],
        [-1., 1.]
    );
    h.click_last("Reference");
    settle(&mut h);
    set_reference_request(&mut h, 3);
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        h.frame(vec![]);
        if !h.app.value_pending()
            && h.app.value_frame().is_some_and(|f| {
                f.reference_requested_revision == Some(3)
                    && f.grid
                        .reference()
                        .is_some_and(|r| r.identity().revision == 3)
                    && f.grid.reference_series.is_some()
            })
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "status={:?} frame={:?}",
            h.app.value_view().status(),
            h.app.value_frame().map(|f| (
                f.reference_requested_revision,
                f.grid.reference().map(|r| r.identity()),
                f.grid.reference_series.as_ref().map(|s| s.values.len())
            ))
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(h.app.value_frame().unwrap().grid.values.len(), 2);
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .grid
            .reference_series
            .as_ref()
            .unwrap()
            .values
            .len(),
        8192
    );
    assert!(
        !h.app.value_view().ready(),
        "oversized independent reference drawing must be rejected before collection"
    );
    assert!(h.has("Value drawing exceeds shared view admission"));
}

#[test]
fn reference_only_sparse_hover_uses_its_own_requested_predecessor_revision() {
    let d = families::sparse_populations(2, 2);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    assert_eq!(
        h.app.value_view().slice().unwrap().state.ranges[0],
        [-1., 1.]
    );
    h.click("Slice");
    settle(&mut h);
    h.click_last("Reference");
    settle(&mut h);
    // Ordinary Comparison slider value is editable; current global time remains k=7.
    set_reference_request(&mut h, 5);
    settle(&mut h);
    h.app.value_view_mut().set_color_limits([-1., 1.]).unwrap();
    h.frame(vec![]);
    h.click("Slice settings");
    h.click("Fit axes");
    h.click("Slice settings");
    h.frame(vec![]);
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.requested_revision, 7);
    assert_eq!(f.grid.reference().unwrap().identity().revision, 3);
    assert_eq!(f.reference_requested_revision, Some(5));
    let s = f.grid.reference_series.as_ref().unwrap();
    let pos = egui_plot::PlotMemory::load(&h.ctx, ValueView::main_plot_id())
        .unwrap()
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(
            s.positions[1][0] as f64,
            s.values[1],
        ));
    h.frame(vec![egui::Event::PointerMoved(pos)]);
    for _ in 0..12 {
        h.frame(vec![]);
    }
    assert!(h.has("Reference sample"), "{:?}", h.text);
    assert!(h.has("actual k=3 / requested k=5"), "{:?}", h.text);
    assert!(!h.has("actual k=3 / requested k=7"));
}
fn set_reference_request(h: &mut Harness, revision: u64) {
    h.set_reference_request(revision);
}

#[test]
fn reference_requested_time_is_metadata_when_the_immutable_source_is_unchanged() {
    use rl::core::action::Action;
    use rl_viewer::{
        data::RunIndex,
        interactive::Interactive,
        value::{
            prepared::{ValuePin, ValueRequest, ValueSource},
            slice::{ActionConditioning, ValueSlice},
        },
    };
    let d = families::family("quantile", true);
    let run = RunIndex::open(d.path().join("run"), ReadLimits::default()).unwrap();
    let current =
        ValueSource::model(&run, 2, "distribution", &run.recording().load(2).unwrap()).unwrap();
    let reference =
        ValueSource::model(&run, 1, "distribution", &run.recording().load(1).unwrap()).unwrap();
    let input = Observation::new(vec![0.; 3]);
    let mut slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
    slice.resolution = [2, 2];
    slice.conditioning = ActionConditioning::Fixed(Action::Continuous(vec![0.; 2]));
    let mut request = ValueRequest::new(
        current,
        ValueOutput::ExpectedReturn,
        slice,
        ValuePin::Input {
            observation: input,
            action: Some(Action::Continuous(vec![0.; 2])),
        },
    )
    .unwrap();
    request.reference_value = Some(reference);
    request.reference_requested_revision = Some(3);
    let mut worker = Interactive::new(|| {}).unwrap();
    worker.submit_value(request.clone()).unwrap();
    // Before the grid/selected completions are polled, newest request metadata wins.
    request.reference_requested_revision = Some(5);
    worker.submit_value(request.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        worker.poll_value().unwrap();
        assert!(worker.value_error().is_none(), "{:?}", worker.value_error());
        if worker.value_frame().is_some_and(|f| f.selected.is_some()) && !worker.value_pending() {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let grid = worker.value_frame().unwrap().grid.clone();
    assert_eq!(
        worker.value_frame().unwrap().reference_requested_revision,
        Some(5)
    );
    let selected = worker.value_frame().unwrap().selected.clone().unwrap();
    let before = worker.value_metrics();
    request.reference_requested_revision = Some(6);
    worker.submit_value(request).unwrap();
    worker.poll_value().unwrap();
    let after = worker.value_metrics();
    assert!(
        worker.value_frame().is_some(),
        "metadata-only reference time must retain the frame"
    );
    assert_eq!(
        worker.value_frame().unwrap().reference_requested_revision,
        Some(6)
    );
    assert!(
        std::sync::Arc::ptr_eq(&grid, &worker.value_frame().unwrap().grid),
        "requested reference time must not invalidate unchanged numerical inputs"
    );
    assert!(std::sync::Arc::ptr_eq(
        &selected,
        worker.value_frame().unwrap().selected.as_ref().unwrap()
    ));
    assert_eq!(
        (before.grid_batches, before.point_batches, before.restores),
        (after.grid_batches, after.point_batches, after.restores)
    );
    assert!(!worker.value_pending());
}

#[test]
fn action_value_map_and_distribution_have_coordinated_panel_surfaces() {
    fn surfaces(shapes: &[egui::Shape], out: &mut Vec<egui::Rect>) {
        for shape in shapes {
            match shape {
                egui::Shape::Rect(rect) if rect.fill == rl_viewer::theme::SURFACE => {
                    out.push(rect.rect)
                }
                egui::Shape::Vec(shapes) => surfaces(shapes, out),
                _ => {}
            }
        }
    }
    let d = families::bounded_family("quantile", true);
    let mut h = Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle(&mut h);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !h.has("Recorded diagnostics") {
        h.frame(vec![]);
        assert!(
            Instant::now() < deadline,
            "recorded diagnostic did not load"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let control_y = |label: &str| {
        h.text
            .iter()
            .find(|(text, _)| text == label)
            .unwrap()
            .1
            .center()
            .y
    };
    assert!((control_y("Map") - control_y("Recorded diagnostics")).abs() < 4.);
    assert!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .action_curve
            .is_some()
    );
    h.frame(vec![]);
    let mut panels = Vec::new();
    surfaces(&h.shapes, &mut panels);
    let main = h.app.value_view().main_rect.unwrap();
    let side = h.app.value_view().inspector_rect.unwrap();
    assert!(
        panels
            .iter()
            .any(|r| main.contains(r.center()) && r.width() > main.width() * 0.8)
    );
    assert!(
        panels
            .iter()
            .any(|r| side.contains(r.center()) && r.width() > side.width() * 0.8)
    );
    assert!(main.bottom() <= h.size.y + 1. && side.bottom() <= h.size.y + 1.);
    assert!(side.right() <= h.size.x + 1., "side={side:?}");
    let curve =
        egui_plot::PlotMemory::load(&h.ctx, egui::Id::unique("value-selected-action-curve"))
            .unwrap();
    let plot = *curve.transform().frame();
    let legend = h
        .text
        .iter()
        .find(|(text, _)| text == "Learned critic")
        .unwrap()
        .1;
    assert!(
        legend.left() >= plot.right(),
        "plot={plot:?}, legend={legend:?}"
    );
    for label in [
        "Learned critic",
        "Selected action",
        "Grid maximum (sampled)",
    ] {
        let bounds = h.text.iter().find(|(text, _)| text == label).unwrap().1;
        assert!(
            bounds.right() <= h.size.x,
            "label={label} bounds={bounds:?}"
        );
    }
    h.click("Recorded diagnostics");
    assert!(h.has("Reference diagnostics"));
}

#[test]
fn compact_desktop_keeps_the_three_value_plots_visible() {
    let d = families::bounded_family("quantile", true);
    let mut h = Harness::new(&d, egui::vec2(1024., 640.));
    h.app.select_view(View::Value);
    settle(&mut h);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(h.has("Action-value function"));
    assert!(h.has("Action-value function · selected input · Expected return"));
    assert!(h.has("Value map"));
    assert!(h.has("Conditional return distribution"));
    let axis = h.text.iter().find(|(text, _)| text == "Return").unwrap().1;
    assert!(
        axis.bottom() <= h.size.y,
        "return axis is clipped: {axis:?}"
    );
    for prefix in ["Mean ", "SD ", "CVaR₁₀ "] {
        let (label, bounds) = h
            .text
            .iter()
            .find(|(text, _)| text.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing compact return statistic {prefix}: {:?}", h.text));
        assert!(
            bounds.bottom() <= h.size.y,
            "compact statistic {label} is clipped: {bounds:?}"
        );
    }
    let curve =
        egui_plot::PlotMemory::load(&h.ctx, egui::Id::unique("value-selected-action-curve"))
            .unwrap();
    let plot = *curve.transform().frame();
    let ticks = h
        .text
        .iter()
        .filter(|(text, rect)| {
            text.parse::<f64>().is_ok()
                && rect.right() <= plot.left()
                && rect.center().y >= plot.top()
                && rect.center().y <= plot.bottom()
        })
        .count();
    assert!(
        ticks >= 2,
        "compact action-value axis needs readable ticks: frame={plot:?}, bounds={:?}, labels={:?}",
        curve.transform().bounds(),
        h.text
            .iter()
            .filter(|(text, _)| text.parse::<f64>().is_ok())
            .collect::<Vec<_>>()
    );
    assert!(
        !h.has("Difference"),
        "compact display options should be grouped"
    );
    h.click("Current");
    assert!(
        h.has("Difference"),
        "the display menu must retain the difference option"
    );
}

#[test]
fn shorter_desktop_keeps_the_conditional_return_axis_inside_the_window() {
    let d = families::bounded_family("quantile", true);
    let mut h = Harness::new(&d, egui::vec2(1024., 580.));
    h.app.select_view(View::Value);
    settle(&mut h);
    for _ in 0..4 {
        h.frame(vec![]);
    }
    assert!(h.has("Conditional return distribution"));
    let axis = h.text.iter().find(|(text, _)| text == "Return").unwrap().1;
    assert!(
        axis.bottom() <= h.size.y,
        "return axis is clipped: {axis:?}"
    );
}

#[test]
fn conditional_distribution_can_expand_without_changing_selected_input() {
    let (_directory, mut h) = open_family("quantile", true);
    let selected = h.app.selection().input.clone();
    let revision = h.app.selection().revision;
    h.click("Expand distribution");
    settle(&mut h);
    assert!(h.app.value_view().main_rect.is_none());
    let expanded = h.app.value_view().inspector_rect.unwrap();
    assert!(expanded.width() > h.size.x * 0.8, "{expanded:?}");
    assert!(h.has("CDF"));
    assert!(h.has("Conditional return distribution"));
    let axis = h.text.iter().find(|(text, _)| text == "Return").unwrap().1;
    let ticks = h
        .text
        .iter()
        .filter(|(text, rect)| {
            text.parse::<f64>().is_ok()
                && rect.left() > expanded.left() + 100.
                && rect.bottom() <= axis.top() + 1.
                && rect.bottom() >= axis.top() - 30.
        })
        .count();
    assert!(
        ticks >= 2,
        "expanded return axis needs numerical ticks: {:?}",
        h.text
    );
    assert_eq!(h.app.selection().input, selected);
    assert_eq!(h.app.selection().revision, revision);
    h.click("Restore layout");
    settle(&mut h);
    assert!(h.app.value_view().main_rect.is_some());
}
