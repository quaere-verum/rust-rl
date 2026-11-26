#[path = "support/policy_ui.rs"]
mod support;
use eframe::egui;
use rl_viewer::policy::{slice::SliceMode, view::PolicyPane};
use support::*;

#[test]
fn deterministic_policy_shows_the_matching_critic_action_curve() {
    use rl::{
        core::action::Action,
        diagnostics::{
            models::{CriticInspectionRequest, InspectionRequest, ModelBundle, ModelLimits},
            storage::ReadLimits,
            value::{ValueOutput, ValuePrediction},
        },
    };
    use rl_viewer::data::RunIndex;
    let dir = policy_with_critic_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    assert!(
        h.text
            .iter()
            .any(|(text, _)| text == "Action-value · selected input"),
        "the same-checkpoint critic curve should be visible: {:?}",
        h.text
    );
    assert!(
        !h.text.iter().any(|(text, _)| text == "Selected"),
        "the action-value panel should not repeat the generic selected heading"
    );
    assert!(
        h.text
            .iter()
            .any(|(text, rect)| text == "Policy cross-section" && rect.bottom() <= h.size.y)
    );
    assert!(
        h.text.iter().any(|(text, _)| text == "Q [reward]"),
        "the critic axis must carry its declared units"
    );
    let frame = h.app.policy_frame().unwrap();
    let selected = frame.selected.as_ref().unwrap();
    let curve = selected.action_value_curve().unwrap();
    assert_eq!(curve.critic, "critic");
    assert_eq!(curve.component, 0);
    assert_eq!(curve.points.len(), 65);
    assert_eq!(curve.points.first().unwrap()[0], -1.);
    assert_eq!(curve.points.last().unwrap()[0], 1.);
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let index = run.by_sequence(frame.grid.current.key.sequence).unwrap();
    let model = ModelBundle::load(
        &run.recording().load(index).unwrap(),
        ModelLimits::default(),
    )
    .unwrap();
    let actor = model
        .inspect(&InspectionRequest::new(
            "actor",
            vec![selected.input.clone()],
        ))
        .unwrap()
        .actions
        .remove(0);
    let Action::Continuous(actor) = actor else {
        panic!("continuous actor expected")
    };
    assert_eq!(curve.actor_action, f64::from(actor[0]));
    let mut actions = [0, 32, 64]
        .into_iter()
        .map(|i| {
            let mut action = actor.clone();
            action[0] = curve.points[i][0] as f32;
            Action::Continuous(action)
        })
        .collect::<Vec<_>>();
    actions.push(Action::Continuous(actor));
    let result = model
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["critic".into()],
            observations: vec![selected.input.clone(); actions.len()],
            fixed_actions: Some(actions),
        })
        .unwrap();
    for (row, index) in [0, 32, 64].into_iter().enumerate() {
        let expected = ValuePrediction::from_critic(&result.critics["critic"], row, None)
            .unwrap()
            .scalar(&ValueOutput::ExpectedReturn, None)
            .unwrap();
        assert!((curve.points[index][1] - expected).abs() < 1e-5);
    }
    let actor_value = ValuePrediction::from_critic(&result.critics["critic"], 3, None)
        .unwrap()
        .scalar(&ValueOutput::ExpectedReturn, None)
        .unwrap();
    assert!((curve.actor_value - actor_value).abs() < 1e-5);
}

#[test]
fn policy_hides_action_value_when_the_critic_is_ambiguous() {
    let dir = policy_with_ambiguous_critics_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.action_value_curve().is_none());
    assert!(
        !h.text
            .iter()
            .any(|(text, _)| text == "Action-value · selected input")
    );
}

#[test]
fn policy_action_value_tracks_the_selected_historical_snapshot() {
    let dir = policy_with_critic_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    let first = h.app.policy_frame().unwrap();
    let revision = first.requested_revision;
    let value = first
        .selected
        .as_ref()
        .unwrap()
        .action_value_curve()
        .unwrap()
        .points[32][1];
    let next = if revision == 3 { 7 } else { 3 };
    h.app.select_revision(next).unwrap();
    h.settle();
    let second = h.app.policy_frame().unwrap();
    assert_eq!(second.requested_revision, next);
    assert_eq!(second.grid.current.revision, next);
    let next_value = second
        .selected
        .as_ref()
        .unwrap()
        .action_value_curve()
        .unwrap()
        .points[32][1];
    assert_ne!(value.to_bits(), next_value.to_bits());
}

#[test]
fn policy_action_value_keeps_a_shared_vertical_range_across_checkpoints() {
    fn highest_tick(h: &Harness) -> f64 {
        let title = h
            .text
            .iter()
            .find(|(s, _)| s == "Action-value · selected input")
            .unwrap()
            .1;
        let below = h
            .text
            .iter()
            .find(|(s, _)| s == "Policy cross-section")
            .unwrap()
            .1;
        h.text
            .iter()
            .filter_map(|(s, rect)| {
                (rect.center().x > h.size.x * 0.6
                    && rect.center().y > title.bottom()
                    && rect.center().y < below.top())
                .then(|| s.parse::<f64>().ok())
                .flatten()
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }
    let dir = policy_with_critic_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    h.app.select_revision(7).unwrap();
    h.settle();
    let initial = highest_tick(&h);
    assert!(
        initial > 5.,
        "fixture should start with the higher critic scale"
    );
    h.app.select_revision(3).unwrap();
    h.settle();
    let after = highest_tick(&h);
    assert!(
        after >= initial - 0.2,
        "the scale shrank from {initial} to {after}"
    );
}

#[test]
fn policy_action_value_tracks_an_exact_new_input_without_rebuilding_the_map() {
    let dir = policy_with_critic_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    let first = h.app.policy_frame().unwrap();
    let grid = std::sync::Arc::as_ptr(&first.grid);
    let previous = first
        .selected
        .as_ref()
        .unwrap()
        .action_value_curve()
        .unwrap()
        .actor_value;
    let batches = h.app.policy_metrics().grid_batches;
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let target = egui_plot::PlotPoint::new(0.37, -0.42);
    h.click_at(memory.transform().position_from_point(&target));
    h.settle();
    let frame = h.app.policy_frame().unwrap();
    assert_eq!(std::sync::Arc::as_ptr(&frame.grid), grid);
    assert_eq!(h.app.policy_metrics().grid_batches, batches);
    let point = frame.selected.as_ref().unwrap();
    assert!((point.input.dense_values()[0] - 0.37).abs() < 0.01);
    assert!((point.input.dense_values()[1] + 0.42).abs() < 0.01);
    assert_ne!(
        previous.to_bits(),
        point.action_value_curve().unwrap().actor_value.to_bits()
    );
}

#[test]
fn compact_policy_keeps_display_controls_and_both_plot_axes_visible() {
    let dir = policy_with_critic_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1024., 640.));
    h.settle();
    h.click("Reference");
    h.settle();
    assert!(
        !h.has("Difference"),
        "compact display choices should be grouped"
    );
    assert!(
        h.text.iter().any(|(text, rect)| text == "obs[1]"
            && rect.center().x > h.size.x * 0.5
            && rect.bottom() <= h.size.y),
        "the supporting plot axis must remain inside the window: {:?}",
        h.text
    );
    assert!(
        h.text.iter().any(|(s, _)| s == "Q") && h.text.iter().any(|(s, _)| s == "reward"),
        "compact critic axes must show their quantity and units without overlap"
    );
    let title = h
        .text
        .iter()
        .find(|(s, _)| s == "Action-value · selected input")
        .unwrap()
        .1;
    let below = h
        .text
        .iter()
        .find(|(s, _)| s == "Policy cross-section")
        .unwrap()
        .1;
    let ticks = h
        .text
        .iter()
        .filter(|(s, rect)| {
            s.parse::<f64>().is_ok()
                && rect.left() >= title.left()
                && rect.right() < title.left() + 64.
                && rect.center().y > title.bottom()
                && rect.center().y < below.top()
        })
        .count();
    assert!(
        ticks >= 2,
        "compact critic plot needs at least two readable value ticks, got {ticks}: {:?}",
        h.text
    );
    h.click("Current");
    h.click_last("Difference");
    h.settle();
    assert_eq!(
        h.app.policy_pane().display,
        rl_viewer::policy::view::Display::Difference
    );
}

#[test]
fn returning_from_training_reuses_the_compatible_policy_frame() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1280., 800.));
    h.settle();
    assert!(!h.text.iter().any(|(text, _)| text.contains("actual k=")));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !h.app.snapshot_preparation_idle() {
        h.frame(vec![]);
        assert!(
            std::time::Instant::now() < deadline,
            "catalogue did not settle"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let before = h.app.policy_metrics();
    let revision = h.app.policy_frame().unwrap().requested_revision;
    h.app.select_view(rl_viewer::ui::View::Training);
    h.frame(vec![]);
    h.app.select_view(rl_viewer::ui::View::PolicyMap);
    h.settle();
    assert_eq!(h.app.policy_frame().unwrap().requested_revision, revision);
    let after = h.app.policy_metrics();
    assert_eq!(after.restores, before.restores);
    assert_eq!(after.grid_batches, before.grid_batches);

    h.app.select_view(rl_viewer::ui::View::Training);
    h.app
        .select_revision(if revision == 3 { 7 } else { 3 })
        .unwrap();
    h.app.select_view(rl_viewer::ui::View::PolicyMap);
    h.settle();
    let changed = h.app.policy_frame().unwrap();
    assert_ne!(changed.requested_revision, revision);
    assert_eq!(changed.grid.current.revision, changed.requested_revision);
    assert!(h.app.policy_metrics().grid_batches > after.grid_batches);
}

#[test]
fn compact_categorical_distribution_has_readable_action_and_probability_ticks() {
    let dir = fixture(3, 8, false);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    for narrow in [false, true] {
        if narrow {
            h.size = egui::vec2(700., 900.);
            h.settle();
            h.click("Selected");
            h.settle();
            let center = h.app.policy_pane().inspector_rect.unwrap().center();
            h.frame(vec![
                egui::Event::PointerMoved(center),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0., -500.),
                    phase: egui::TouchPhase::Move,
                    source: egui::MouseWheelSource::Unknown,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        for _ in 0..4 {
            h.frame(vec![]);
        }
        let memory =
            egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
        let plot = *memory.transform().frame();
        let x: Vec<_> = h
            .text
            .iter()
            .filter_map(|(s, r)| {
                let v = s.parse::<u32>().ok()?;
                (v > 0
                    && v < 8
                    && r.center().x >= plot.left()
                    && r.center().x <= plot.right()
                    && r.center().y >= plot.bottom()
                    && r.center().y < plot.bottom() + 30.)
                    .then_some(v)
            })
            .collect();
        assert_eq!(
            x,
            vec![1, 2, 3, 4, 5, 6, 7],
            "all compact action identities must be readable: {x:?}"
        );
        let y = h
            .text
            .iter()
            .filter(|(s, r)| {
                s.parse::<f64>().is_ok_and(|v| v > 0. && v <= 1.)
                    && r.center().y > plot.top()
                    && r.center().y < plot.bottom()
                    && r.center().x < plot.left()
                    && r.center().x > plot.left() - 45.
            })
            .count();
        assert!(
            y >= 2,
            "probability scale needs nonzero numerical labels: {:?}",
            h.text
        );
        let labels: Vec<_> = h
            .text
            .iter()
            .filter(|(s, r)| {
                s.parse::<u32>().is_ok_and(|v| v < 8)
                    && r.center().x >= plot.left()
                    && r.center().x <= plot.right()
                    && r.center().y >= plot.bottom()
                    && r.center().y < plot.bottom() + 30.
            })
            .map(|(_, r)| *r)
            .collect();
        assert!(
            labels
                .iter()
                .all(|r| r.left() >= 0. && r.right() <= h.size.x && r.bottom() <= h.size.y)
        );
        assert!(
            labels.windows(2).all(|r| r[0].right() < r[1].left()),
            "category labels overlap"
        );
    }
}
fn assert_distribution_y_ticks(h: &Harness) {
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    let plot = *memory.transform().frame();
    let labels: Vec<_> = h
        .text
        .iter()
        .filter(|(s, r)| {
            s.parse::<f64>().is_ok_and(|v| v > 0.)
                && r.center().y > plot.top()
                && r.center().y < plot.bottom()
                && r.center().x < plot.left()
                && r.center().x > plot.left() - 45.
        })
        .map(|(_, r)| *r)
        .collect();
    assert!(
        labels.len() >= 2,
        "nonzero density/CDF ticks are required: {:?}",
        h.text
    );
    assert!(
        labels
            .iter()
            .all(|r| r.left() >= 0. && r.right() <= h.size.x && r.bottom() <= h.size.y)
    );
}
#[test]
fn pointer_modes_axes_action_exact_click_drag_hover_and_playback() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let pin = h.app.policy_pane().pinned_input().unwrap().clone();
    h.click("Slice");
    h.settle();
    assert_eq!(
        h.app.policy_pane().config().unwrap().mode,
        SliceMode::OneDimensional
    );
    assert_eq!(h.app.policy_pane().config().unwrap().resolution[0], 256);
    assert!(!h.has("Y axis"));
    h.click("Map");
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &pin);
    // A real off-grid click changes only the two plotted input coordinates.
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let point = egui_plot::PlotPoint::new(0.12345, -0.23456);
    let pixel = memory.transform().position_from_point(&point);
    h.click_at(pixel);
    h.settle();
    let input = h.app.policy_pane().pinned_input().unwrap().clone();
    assert!((input.dense_values()[0] - 0.12345).abs() < 0.01);
    assert!((input.dense_values()[1] + 0.23456).abs() < 0.01);
    assert_eq!(input.dense_values()[2], pin.dense_values()[2]);
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input,
        input
    );
    let grid = std::sync::Arc::as_ptr(&h.app.policy_frame().unwrap().grid);
    let uploads = h.app.policy_pane().texture_uploads;
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let start = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(
            input.dense_values()[0] as f64,
            input.dense_values()[1] as f64,
        ));
    let end = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(0.35, 0.25));
    h.frame(vec![egui::Event::PointerMoved(start), button(start, true)]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.settle();
    assert!((h.app.policy_pane().pinned_input().unwrap().dense_values()[0] - 0.35).abs() < 0.01);
    assert_eq!(
        std::sync::Arc::as_ptr(&h.app.policy_frame().unwrap().grid),
        grid
    );
    assert_eq!(h.app.policy_pane().texture_uploads, uploads);
    h.frame(vec![egui::Event::PointerMoved(pixel)]);
    h.time += 1.;
    h.frame(vec![]);
    assert!(h.has("Output"));
    assert!(
        h.has("obs[2] = 0.0"),
        "exact complete hover input: {:?}",
        h.text
    );
    assert!(h.has("actor · k=7"));
    assert_eq!(h.app.policy_pane().texture_uploads, uploads);
    let input = h.app.policy_pane().pinned_input().unwrap().clone();
    h.click("Slice settings");
    assert!(h.has("X axis"), "first axis context must be open");
    h.click_last("obs[0]");
    h.click_last("obs[2]");
    if h.has("X axis") {
        h.click("Slice settings");
    }
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    h.click("Slice settings");
    assert!(h.has("X axis"), "second axis context must be open");
    h.click_last("obs[2]");
    h.click_last("obs[0]");
    if h.has("X axis") {
        h.click("Slice settings");
    }
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    h.click("Slice");
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    h.click("Map");
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let bounds = *memory.bounds();
    let center = memory.transform().frame().center();
    let end = center + egui::vec2(40., 20.);
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.frame(vec![]);
    let manual = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
        .unwrap()
        .bounds()
        .to_owned();
    assert_ne!(bounds, manual);
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    let limits = h.app.policy_pane().color_limits();
    h.app.select_revision(3).unwrap();
    h.settle();
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &input);
    assert_eq!(
        egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
            .unwrap()
            .bounds()
            .to_owned(),
        manual
    );
    assert_eq!(h.app.policy_pane().color_limits(), limits);
}

#[test]
fn pointer_axis_swap_action_fixed_edit_search_reference_and_visits() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice settings");
    h.click_last("obs[0]");
    h.click_last("obs[1]");
    if h.has("X axis") {
        h.click("Slice settings");
    }
    h.settle();
    let c = h.app.policy_pane().config().unwrap();
    assert_eq!((c.x, c.y), (1, Some(0)));
    h.click("action[0]");
    h.click_last("action[1]");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().action, 1);
    h.click("Fixed inputs");
    let numeric = h.find("0.00");
    h.frame(vec![
        egui::Event::PointerMoved(numeric),
        button(numeric, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(
        numeric + egui::vec2(20., 0.),
    )]);
    h.frame(vec![button(numeric + egui::vec2(20., 0.), false)]);
    h.settle();
    assert_ne!(h.app.policy_pane().config().unwrap().fixed[2], 0.);
    assert_eq!(
        h.app.policy_pane().config().unwrap().fixed[2],
        h.app.policy_pane().pinned_input().unwrap().dense_values()[2]
    );
    h.click("Fixed inputs");
    h.click("Reference");
    h.settle();
    assert!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value
            .is_some()
    );
    h.click("Difference");
    h.settle();
    assert_eq!(
        h.app.policy_pane().display,
        rl_viewer::policy::view::Display::Difference
    );
    h.click("Visited states");
    h.settle();
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .visits
            .as_ref()
            .unwrap()
            .filtered_count,
        0,
        "fixed input moved away from actual visits"
    );
    let dir = fixture(10, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Fixed inputs");
    h.click("Search inputs");
    h.frame(vec![egui::Event::Text("obs[9]".into())]);
    h.frame(vec![]);
    assert!(h.has("obs[9]"), "{:?}", h.text);
    assert!(!h.has("obs[8]"));
    h.frame(vec![egui::Event::Key {
        key: egui::Key::Space,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert!(!h.app.training_is_playing(), "text editor owns keyboard");
}

#[test]
fn categorical_chart_pointer_zoom_reaches_action_79_and_action_identity_survives_time() {
    let dir = fixture(3, 80, false);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let id = PolicyPane::distribution_plot_id();
    let memory = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
    let center = memory.transform().frame().center();
    for _ in 0..30 {
        if egui_plot::PlotMemory::load(&h.ctx, id)
            .unwrap()
            .bounds()
            .max()[0]
            > 80.
        {
            break;
        }
        h.frame(vec![
            egui::Event::PointerMoved(center),
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(-300., 0.),
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                modifiers: egui::Modifiers::NONE,
            },
        ]);
    }
    for _ in 0..12 {
        h.frame(vec![]);
    }
    let memory = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
    assert!(memory.bounds().max()[0] > 79., "{:?}", memory.bounds());
    let pos = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(79., 0.02));
    h.frame(vec![egui::Event::PointerMoved(pos)]);
    h.time += 1.;
    h.frame(vec![]);
    assert!(h.has("invalid"), "{:?}", h.text);
    h.click("0");
    let popup = h
        .text
        .iter()
        .rev()
        .find(|(s, _)| s == "5")
        .unwrap()
        .1
        .center();
    h.frame(vec![
        egui::Event::PointerMoved(popup),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0., -10000.),
            phase: egui::TouchPhase::Move,
            source: egui::MouseWheelSource::Unknown,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    h.frame(vec![]);
    h.click_last("79");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().action, 79);
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .value,
        0.
    );
    h.app.select_revision(3).unwrap();
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().action, 79);
}

#[test]
fn sparse_click_keeps_real_identity_and_explicit_controller_path_is_accessible() {
    let dir = fixture(3, 80, false);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let position = memory
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(0.21, 0.31));
    h.click_at(position);
    h.settle();
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values(),
        vec![0., 0., 0.]
    );
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::ProbeRow { row: 0, .. }
    ));
    h.click("Reference");
    h.click("Reference source");
    let revision = h.text.windows(2).find(|pair| pair[0].0 == "k").unwrap()[1]
        .1
        .center();
    edit_number_at(&mut h, revision, "7");
    h.settle();
    if !h.has("Recorded policy") {
        h.click("Reference source");
    }
    h.click("Recorded policy");
    h.click_last("Recorded controller");
    h.settle();
    assert!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference
            .as_ref()
            .unwrap()
            .explicit_controller
    );
    h.click("Reference");
    h.settle();
    assert!(h.app.policy_frame().unwrap().grid.reference.is_none());
}

fn edit_number(h: &mut Harness, label: &str, value: &str) {
    let pos = h.find(label);
    edit_number_at(h, pos, value);
}
fn edit_number_at(h: &mut Harness, pos: egui::Pos2, value: &str) {
    h.click_at(pos);
    let ctrl = egui::Modifiers {
        ctrl: true,
        ..Default::default()
    };
    h.frame(vec![
        egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: ctrl,
        },
        egui::Event::Text(value.into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    h.frame(vec![]);
}
#[test]
fn ordinary_settings_admit_maximum_grids_and_keep_edited_resolution_by_mode() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Reference");
    h.settle();
    h.click("Slice settings");
    edit_number(&mut h, "128", "256");
    edit_number(&mut h, "128", "256");
    h.click("Slice settings");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().resolution, [256, 256]);
    assert_eq!(h.app.policy_frame().unwrap().grid.values.len(), 65536);
    assert!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference_at(65535)
            .is_some()
    );
    assert!(h.app.policy_metrics().prepared_peak <= 4 * 1024 * 1024);
    h.click("Slice");
    h.settle();
    h.click("Slice settings");
    edit_number(&mut h, "256", "4096");
    h.click("Slice settings");
    h.settle();
    assert_eq!(h.app.policy_frame().unwrap().grid.values.len(), 4096);
    h.click("Map");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().resolution, [256, 256]);
    h.click("Slice");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().resolution[0], 4096);
    println!(
        "ordinary max-resolution peak={} reserved_ui={} owned_ui={}",
        h.app.policy_metrics().prepared_peak,
        h.app.policy_pane().state_reservation_bytes(),
        h.app.policy_pane().owned_state_bytes()
    );
}
#[test]
fn categorical_fixed_editor_only_exposes_declared_codes() {
    let dir = fixture(3, 0, true);
    let path = dir.path().join("run/manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["environment_descriptor"] = serde_json::json!({"version":1,"observations":[{"id":"x","label":"Position","kind":"Numerical","range":[-1,1],"coordinate":true},{"id":"y","label":"Time","kind":"Numerical","range":[-1,1],"coordinate":true},{"id":"regime","label":"Regime","kind":{"Categorical":{"categories":[{"code":0,"label":"Low"},{"code":0.5,"label":"High"}]}},"coordinate":false}],"actions":{"Continuous":{"components":[{"label":"Effort"},{"label":"Other effort"}],"preferred_component":0}},"clocks":{"learner_revision":{"label":"Revision"},"decision_step":{"label":"Step"}},"reward":{"label":"Reward"}});
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Fixed inputs");
    h.click("Low");
    h.click_last("High");
    h.settle();
    assert_eq!(h.app.policy_pane().config().unwrap().fixed[2], 0.5);
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values()[2],
        0.5
    );
}

#[test]
fn unsupported_recorded_fixed_edit_never_relabels_old_output_as_the_new_complete_input() {
    let dir = fixture(3, 80, false);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let actual = h
        .app
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    let selected = h.app.selection().input.clone();
    h.click("Fixed inputs");
    edit_number(&mut h, "0.00", "0.5");
    h.click("Fixed inputs");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        h.frame(vec![]);
        assert!(
            h.app
                .policy_frame()
                .and_then(|f| f.selected.as_ref())
                .is_some_and(|p| p.input == actual),
            "the held complete output must retain its original exact input"
        );
        if !h.app.policy_pending()
            && h.app
                .policy_pane()
                .status()
                .is_some_and(|s| s.contains("unsupported"))
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            h.app.policy_pane().status()
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_ne!(h.app.policy_pane().pinned_input().unwrap(), &actual);
    assert_eq!(h.app.selection().input, selected);
    assert!(!h.app.policy_pane().ready());
    assert!(h.has("0.013"), "the complete held output remains visible");
    assert!(h.has("Showing previous complete frame"));
    assert!(h.has(&format!(
        "Showing checkpoint {}",
        h.app.policy_frame().unwrap().grid.current.revision
    )));
    assert!(
        h.has("Previous input"),
        "a held prediction must visibly identify its previous complete input"
    );
    assert!(h.text.iter().any(|(s, _)| s.contains("unsupported")));
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(point.input, actual);
    assert_main_circle(
        &h,
        [
            actual.dense_values()[0] as f64,
            actual.dense_values()[1] as f64,
        ],
        egui::Color32::WHITE,
        6.,
    );
}

fn plot_pixel(h: &Harness, point: [f64; 2]) -> egui::Pos2 {
    egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
        .unwrap()
        .transform()
        .position_from_point(&egui_plot::PlotPoint::from(point))
}
fn circles(shapes: &[egui::Shape]) -> Vec<egui::epaint::CircleShape> {
    shapes
        .iter()
        .flat_map(|s| match s {
            egui::Shape::Circle(c) => vec![*c],
            egui::Shape::Vec(v) => circles(v),
            _ => vec![],
        })
        .collect()
}
fn assert_main_circle(h: &Harness, point: [f64; 2], color: egui::Color32, radius: f32) {
    let expected = plot_pixel(h, point);
    assert!(
        circles(&h.shapes).iter().any(|c| c.fill == color
            && (c.radius - radius).abs() < 0.1
            && c.center.distance(expected) < 0.3),
        "missing rendered circle {point:?} {color:?} radius={radius}"
    );
}
fn enable_recorded_controller(h: &mut Harness) {
    enable_review_reference(h);
    if !h.has("Recorded policy") {
        h.click("Reference source");
    }
    h.click("Recorded policy");
    h.click_last("Recorded controller");
    h.settle();
    assert!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference
            .as_ref()
            .unwrap()
            .explicit_controller
    );
}
fn enable_review_reference(h: &mut Harness) {
    h.click("Reference");
    h.settle();
    h.click("Reference source");
    let revision = h
        .text
        .windows(2)
        .find(|pair| pair[0].0 == "k")
        .expect("reference revision editor follows its k label")[1]
        .1
        .center();
    edit_number_at(h, revision, "3");
    h.settle();
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference
            .as_ref()
            .unwrap()
            .revision,
        3
    );
}
fn colored_segments(shapes: &[egui::Shape], rect: egui::Rect, color: egui::Color32) -> usize {
    shapes
        .iter()
        .map(|s| match s {
            egui::Shape::Path(p)
                if p.points.len() > 1
                    && matches!(p.stroke.color, egui::epaint::ColorMode::Solid(c) if c == color)
                    && p.points.iter().all(|p| rect.contains(*p)) =>
            {
                1
            }
            egui::Shape::LineSegment { points, stroke }
                if stroke.color == color && points.iter().all(|p| rect.contains(*p)) =>
            {
                1
            }
            egui::Shape::Vec(v) => colored_segments(v, rect, color),
            _ => 0,
        })
        .sum()
}
#[test]
fn final_regular_current_keeps_sparse_recorded_reference_and_difference_as_points() {
    let reference = rl_viewer::theme::REFERENCE;
    let current = rl_viewer::policy::view::CURRENT;
    for matches in [2, 1] {
        let dir = mixed_reference_fixture(matches);
        let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
        h.settle();
        h.click("Slice");
        h.settle();
        enable_recorded_controller(&mut h);
        let g = &h.app.policy_frame().unwrap().grid;
        assert!(g.regular());
        assert!(
            g.selection_intent(0).is_none(),
            "synthetic input has no fake recorded identity"
        );
        let samples: Vec<_> = (0..g.values.len())
            .filter_map(|i| {
                g.reference_at(i).map(|r| {
                    (
                        g.input(i).unwrap().dense_values()[0] as f64,
                        r,
                        g.values[i] - r,
                    )
                })
            })
            .collect();
        assert_eq!(
            samples.len(),
            matches,
            "slice={:?} first={:?} last={:?}",
            g.config,
            g.input(0),
            g.input(g.values.len() - 1)
        );
        for &(x, r, _) in &samples {
            assert_main_circle(&h, [x, r], reference, 4.);
        }
        let rect = *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
            .unwrap()
            .transform()
            .frame();
        assert_eq!(
            colored_segments(&h.shapes, rect, reference),
            0,
            "recorded reference must not bridge unsupported inputs"
        );
        assert!(
            colored_segments(&h.shapes, rect, current) > 0,
            "retained current remains a curve"
        );
        h.click("Difference");
        h.settle();
        for &(x, _, d) in &samples {
            assert_main_circle(&h, [x, d], current, 4.);
        }
        let rect = *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
            .unwrap()
            .transform()
            .frame();
        assert_eq!(
            colored_segments(&h.shapes, rect, current),
            0,
            "sparse difference must not bridge unsupported inputs"
        );
    }
}
fn distribution_bounds(h: &Harness) -> egui_plot::PlotBounds {
    *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
        .unwrap()
        .bounds()
}
fn assert_density_visible(h: &Harness, x: std::ops::RangeInclusive<f64>) {
    let p = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    let peak = p
        .plot
        .iter()
        .chain(p.reference_plot.iter())
        .flat_map(|p| p.samples.iter())
        .map(|p| p[1])
        .fold(0., f64::max);
    let bounds = distribution_bounds(h);
    assert!(
        bounds.max()[1] >= peak,
        "automatic density clipped: bounds={bounds:?} peak={peak}"
    );
    assert_eq!(bounds.range_x(), x, "action domain must remain stable");
}
fn select_second_gaussian_row(h: &mut Harness) {
    let value = h.app.policy_frame().unwrap().grid.values[1];
    let pixel = plot_pixel(h, [0.02f32 as f64, value]);
    h.click_at(pixel);
    h.settle();
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::ProbeRow { row: 1, .. }
    ));
}
#[test]
fn final_automatic_density_expands_after_current_pin_change() {
    let dir = varying_gaussian_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    let x = distribution_bounds(&h).range_x();
    assert!(distribution_bounds(&h).max()[1] < 0.5);
    select_second_gaussian_row(&mut h);
    assert_density_visible(&h, x);
}
#[test]
fn final_automatic_density_expands_after_current_checkpoint_change() {
    let dir = varying_gaussian_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let x = distribution_bounds(&h).range_x();
    h.app.select_revision(3).unwrap();
    h.settle();
    assert_density_visible(&h, x);
}
#[test]
fn final_automatic_density_expands_for_same_reference_source_at_new_pin() {
    let dir = varying_gaussian_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    enable_review_reference(&mut h);
    let x = distribution_bounds(&h).range_x();
    let key = h
        .app
        .policy_frame()
        .unwrap()
        .grid
        .reference
        .as_ref()
        .unwrap()
        .key
        .clone();
    assert!(distribution_bounds(&h).max()[1] < 2.);
    select_second_gaussian_row(&mut h);
    assert_eq!(
        h.app
            .policy_frame()
            .unwrap()
            .grid
            .reference
            .as_ref()
            .unwrap()
            .key,
        key
    );
    assert_density_visible(&h, x);
}
#[test]
fn final_manual_density_bounds_survive_a_taller_selected_distribution() {
    let dir = varying_gaussian_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    let center = memory.transform().frame().center();
    let end = center + egui::vec2(15., 10.);
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.frame(vec![]);
    let manual = distribution_bounds(&h);
    select_second_gaussian_row(&mut h);
    assert_eq!(
        distribution_bounds(&h),
        manual,
        "deliberate X/Y navigation must survive a taller peak"
    );
}
#[test]
fn review_sparse_1d_renders_both_exact_populations_and_omits_unmatched_reference() {
    let dir = comparison_fixture(None);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    enable_recorded_controller(&mut h);
    let g = &h.app.policy_frame().unwrap().grid;
    assert!(!g.regular());
    assert_eq!(g.values.len(), 3);
    assert_eq!(g.reference_at(0), Some(0.9));
    assert_eq!(g.reference_at(1), Some(0.5));
    assert_eq!(g.reference_at(2), None);
    for (x, y) in [(0., 0.8), (0.02f32 as f64, 0.2), (0.5, 0.6)] {
        assert_main_circle(&h, [x, y], rl_viewer::policy::view::CURRENT, 4.);
    }
    for (x, y) in [(0., 0.9), (0.02f32 as f64, 0.5)] {
        assert_main_circle(&h, [x, y], rl_viewer::theme::REFERENCE, 4.);
    }
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let plot = *memory.transform().frame();
    let reference_points = circles(&h.shapes)
        .into_iter()
        .filter(|c| c.fill == rl_viewer::theme::REFERENCE && plot.contains(c.center))
        .count();
    assert_eq!(
        reference_points, 2,
        "unmatched reference input must not invent a point"
    );
    fn line_in(shapes: &[egui::Shape], rect: egui::Rect) -> bool {
        shapes.iter().any(|s| match s {
            egui::Shape::Path(p) => {
                p.points.len() > 1
                    && matches!(p.stroke.color, egui::epaint::ColorMode::Solid(color) if [rl_viewer::policy::view::CURRENT, egui::Color32::from_rgb(174,187,207)].contains(&color))
                    && p.points.iter().all(|p| rect.contains(*p))
            }
            egui::Shape::Vec(v) => line_in(v, rect),
            _ => false,
        })
    }
    assert!(
        !line_in(&h.shapes, plot),
        "sparse populations must not be connected by interpolation"
    );
    h.click("Difference");
    h.settle();
    assert_main_circle(&h, [0., -0.1], rl_viewer::policy::view::CURRENT, 4.);
    assert_main_circle(
        &h,
        [0.02f32 as f64, -0.3],
        rl_viewer::policy::view::CURRENT,
        4.,
    );
    assert!(
        !circles(&h.shapes)
            .iter()
            .any(|c| c.fill == rl_viewer::theme::REFERENCE && plot.contains(c.center))
    );
}

#[test]
fn deterministic_selected_action_comparison_requires_a_matched_reference() {
    let dir = comparison_fixture(None);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    enable_recorded_controller(&mut h);
    let selected = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(selected.value, 0.8);
    assert_eq!(selected.reference_value, Some(0.9));
    assert!(h.has("Action comparison"));
    let unmatched = plot_pixel(&h, [0.5, 0.6]);
    h.click_at(unmatched);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        h.frame(vec![]);
        if !h.app.policy_pending()
            && h.app
                .policy_frame()
                .and_then(|frame| frame.selected.as_ref())
                .is_some_and(|point| point.input.dense_values()[0] == 0.5)
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "unmatched input did not publish"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    h.frame(vec![]);
    assert!(
        h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value
            .is_none()
    );
    assert!(!h.has("Action comparison"));
}
#[test]
fn review_sparse_difference_click_selects_the_visible_real_row_and_marker() {
    let dir = comparison_fixture(None);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    h.click("Slice");
    h.settle();
    enable_recorded_controller(&mut h);
    let row_one = plot_pixel(&h, [0.02f32 as f64, 0.2]);
    h.click_at(row_one);
    h.settle();
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::ProbeRow { row: 1, .. }
    ));
    h.click("Difference");
    h.settle();
    assert!(h.has("Difference · action[0]"));
    // Explicit Fit isolates hit testing from the independently covered quantity-bounds defect.
    h.click("Slice settings");
    h.click("Fit");
    h.settle();
    let pixel = plot_pixel(&h, [0., -0.1]);
    h.click_at(pixel);
    h.settle();
    assert!(
        matches!(
            h.app.selection().input,
            rl_viewer::data::SelectedInput::ProbeRow { row: 0, .. }
        ),
        "clicking visible difference row0 selected {:?}",
        h.app.selection().input
    );
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!((point.value - 0.8).abs() < 1e-7);
    assert!((point.reference_value.unwrap() - 0.9).abs() < 1e-7);
    assert_main_circle(&h, [0., -0.1], egui::Color32::WHITE, 6.);
}
#[test]
fn review_regular_1d_quantity_bounds_marker_and_reference_mode_are_coherent() {
    use rl_viewer::policy::view::Display;
    let dir = positive_model_fixture();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    enable_review_reference(&mut h);
    let reference_display = h
        .text
        .iter()
        .filter(|(s, _)| s == "Reference")
        .nth(1)
        .unwrap()
        .1
        .center();
    h.click_at(reference_display);
    h.settle();
    assert_eq!(h.app.policy_pane().display, Display::Reference);
    h.click("Slice");
    h.settle();
    assert_eq!(
        h.app.policy_pane().display,
        Display::Current,
        "1D normalizes its unsupported reference-only display"
    );
    let rect = *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
        .unwrap()
        .transform()
        .frame();
    assert!(
        colored_segments(&h.shapes, rect, rl_viewer::theme::REFERENCE) > 1,
        "model-to-model reference retains its dashed curve"
    );
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let center = memory.transform().frame().center();
    let end = center + egui::vec2(30., 0.);
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.frame(vec![]);
    let x_before = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
        .unwrap()
        .bounds()
        .range_x();
    h.click("Difference");
    h.settle();
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    assert_eq!(
        memory.bounds().range_x(),
        x_before,
        "quantity fitting must retain manual X bounds"
    );
    assert!(
        memory.bounds().min()[1] < 0.,
        "positive current bounds must not hide negative differences"
    );
    let g = &h.app.policy_frame().unwrap().grid;
    let i = (0..g.values.len())
        .min_by(|&a, &b| {
            g.difference_at(a)
                .unwrap()
                .total_cmp(&g.difference_at(b).unwrap())
        })
        .unwrap();
    let delta = g.difference_at(i).unwrap();
    assert!(
        delta < -1e-4,
        "fixture must have a nonzero negative difference: {delta}"
    );
    let x = g.input(i).unwrap().dense_values()[0] as f64;
    let pixel = plot_pixel(&h, [x, delta]);
    h.click_at(pixel);
    h.settle();
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.value > 0. && point.reference_value.unwrap() > 0.);
    let expected = [
        point.input.dense_values()[0] as f64,
        point.difference.unwrap(),
    ];
    assert_main_circle(&h, expected, egui::Color32::WHITE, 6.);
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).unwrap();
    let center = memory.transform().frame().center();
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(
        center + egui::vec2(20., 10.),
    )]);
    h.frame(vec![button(center + egui::vec2(20., 10.), false)]);
    h.frame(vec![]);
    let bounds = *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
        .unwrap()
        .bounds();
    h.app.select_revision(3).unwrap();
    h.settle();
    assert_eq!(
        *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id())
            .unwrap()
            .bounds(),
        bounds,
        "same quantity retains navigation across checkpoints"
    );
}
#[test]
fn review_gaussian_reference_activation_fits_both_peaks_without_resetting_action_navigation() {
    let dir = comparison_fixture(Some(0.25f64.ln()));
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let current = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(!current.plot.as_ref().unwrap().is_cdf);
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    assert!(memory.bounds().max()[1] < 0.5);
    let center = memory.transform().frame().center();
    let end = center + egui::vec2(20., 0.);
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.frame(vec![]);
    let horizontal = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
        .unwrap()
        .bounds()
        .range_x();
    enable_review_reference(&mut h);
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    let peak = point
        .reference_plot
        .as_ref()
        .unwrap()
        .samples
        .iter()
        .map(|p| p[1])
        .fold(0f64, f64::max);
    assert!(peak > 1.5);
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    assert!(
        memory.bounds().max()[1] >= peak,
        "reference density clipped: bounds={:?}, peak={peak}",
        memory.bounds()
    );
    assert_eq!(
        memory.bounds().range_x(),
        horizontal,
        "reference activation retains action viewport"
    );
    h.click("Slice settings");
    h.click("Fit");
    h.settle();
    assert!(
        egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
            .unwrap()
            .bounds()
            .max()[1]
            >= peak
    );
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    let center = memory.transform().frame().center();
    let end = center + egui::vec2(15., 10.);
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(end)]);
    h.frame(vec![button(end, false)]);
    h.frame(vec![]);
    let manual = *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
        .unwrap()
        .bounds();
    h.app.select_revision(3).unwrap();
    h.settle();
    assert_eq!(
        *egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
            .unwrap()
            .bounds(),
        manual,
        "same density/reference keeps deliberate action-chart navigation across checkpoints"
    );
}
#[test]
fn review_gaussian_density_cdf_transitions_fit_y_and_keep_the_horizontal_domain() {
    let dir = comparison_fixture(Some(-30.));
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    assert_distribution_y_ticks(&h);
    let horizontal = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id())
        .unwrap()
        .bounds()
        .range_x();
    enable_review_reference(&mut h);
    assert_distribution_y_ticks(&h);
    let point = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
    assert!(point.plot.as_ref().unwrap().is_cdf && point.reference_plot.as_ref().unwrap().is_cdf);
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    assert!(
        memory.bounds().max()[1] >= 1.,
        "CDF clipped at old density height {:?}",
        memory.bounds()
    );
    assert_eq!(memory.bounds().range_x(), horizontal);
    h.click("Reference");
    h.settle();
    assert!(
        !h.app
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .plot
            .as_ref()
            .unwrap()
            .is_cdf
    );
    let memory = egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).unwrap();
    assert!(memory.bounds().max()[1] < 0.5);
    assert_eq!(memory.bounds().range_x(), horizontal);
}
