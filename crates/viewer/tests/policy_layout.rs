#[path = "support/policy_ui.rs"]
mod support;
use eframe::egui;
use support::*;

#[test]
fn compact_shell_keeps_tabs_and_reference_controls_separate() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1024., 640.));
    h.settle();
    let rect = |label: &str| h.text.iter().find(|(text, _)| text == label).unwrap().1;
    assert!(!rect("Environment").intersects(rect("Reference source")));
    assert!(rect("Environment").right() <= 1024. && rect("Reference source").right() <= 1024.);
    assert!(rect("Environment").top() > rect("Reference source").bottom());
}

#[test]
fn policy_mode_and_mapping_controls_share_a_desktop_toolbar() {
    let dir = fixture(3, 0, true);
    for width in [1280., 1440.] {
        let mut h = Harness::new(&dir, egui::vec2(width, 900.));
        h.settle();
        let position = |label: &str| {
            h.text
                .iter()
                .find(|(text, _)| text == label)
                .unwrap()
                .1
                .center()
                .y
        };
        assert!(
            (position("Mapping") - position("Map")).abs() < 4.,
            "Mapping and Map should share a row at width {width}"
        );
        h.click("Evolution");
        assert!(h.has("Mapping") && h.text.iter().all(|(text, _)| text != "Map"));
        h.click("Mapping");
        h.settle();
        assert!(h.has("Map") && h.app.policy_pane().ready());
    }
}

#[test]
fn policy_map_and_supporting_plots_have_distinct_panel_surfaces() {
    fn surfaces(shapes: &[egui::Shape], out: &mut Vec<egui::Rect>) {
        for shape in shapes {
            match shape {
                egui::Shape::Rect(rect) if rect.fill == rl_viewer::theme::SURFACE => {
                    out.push(rect.rect);
                }
                egui::Shape::Vec(shapes) => surfaces(shapes, out),
                _ => {}
            }
        }
    }
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let mut panels = Vec::new();
    surfaces(&h.shapes, &mut panels);
    let pane = h.app.policy_pane();
    let main = pane.main_rect.unwrap();
    let side = pane.inspector_rect.unwrap();
    assert!(
        panels
            .iter()
            .any(|r| main.contains(r.center()) && r.width() > main.width() * 0.8)
    );
    let supporting: Vec<_> = panels
        .iter()
        .filter(|r| side.contains(r.center()) && r.width() > side.width() * 0.8 && r.height() > 30.)
        .collect();
    assert!(supporting.len() >= 2, "panels={panels:?}");
    let bottom = supporting.iter().map(|r| r.bottom()).fold(0., f32::max);
    assert!(
        side.bottom() - bottom < 35.,
        "side={side:?}, panels={supporting:?}"
    );
}

#[test]
fn closed_inspector_reclaims_the_entire_landscape_width() {
    let closed = rl_viewer::landscape::layout(1200., 700., false);
    assert_eq!(closed.main_width, 1200.);
    assert_eq!(closed.selected_width, 0.);
    let open = rl_viewer::landscape::layout(1200., 700., true);
    assert!((open.main_width / (open.main_width + open.selected_width) - 0.73).abs() < 0.001);
    let narrow = rl_viewer::landscape::layout(700., 700., true);
    assert!(narrow.stacked && narrow.plot_height < 400.);
}
#[test]
fn negative_only_values_have_no_artificial_zero_boundary() {
    assert_eq!(
        rl_viewer::landscape::numeric_bounds([-20., -10.].into_iter(), false),
        [-20., -10.]
    );
    let low = rl_viewer::landscape::color(-20., [-20., -10.]);
    let mid = rl_viewer::landscape::color(-15., [-20., -10.]);
    let high = rl_viewer::landscape::color(-10., [-20., -10.]);
    assert!(
        low.b() < mid.b() && mid.b() < high.b(),
        "sequential blue lightness must rise through the entire negative range: {low:?} {mid:?} {high:?}"
    );
}
#[test]
fn automatic_policy_grid_uses_admitted_128_and_continuous_linear_texture() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    assert_eq!(
        h.app.policy_frame().unwrap().grid.config.resolution,
        [128, 128]
    );
    let manager = h.ctx.tex_manager();
    let manager = manager.read();
    let texture = manager
        .allocated()
        .find(|(_, texture)| texture.name == "Policy")
        .unwrap()
        .1;
    assert_eq!(texture.options, egui::TextureOptions::LINEAR);
}
#[test]
fn explicit_semantics_keep_probability_difference_and_category_colors_honest() {
    use rl_viewer::landscape::{LandscapeScale, Palette};
    let p = LandscapeScale::new([-5., 5.], Palette::Probability);
    assert_eq!(p.range, [0., 1.]);
    assert_eq!(p.color(0.), p.color(-1.));
    let d = LandscapeScale::new([-1., 3.], Palette::Difference);
    assert_eq!(d.range, [-3., 3.]);
    assert_eq!(d.color(0.), rl_viewer::theme::SIGNED_ZERO);
    assert_ne!(d.color(0.), egui::Color32::TRANSPARENT);
    assert_ne!(d.color(0.), rl_viewer::theme::BG);
    let signed = LandscapeScale::new([-2., 4.], Palette::SignedAction);
    assert_eq!(signed.range, [-4., 4.]);
    let categorical = LandscapeScale::new([0., 7.], Palette::Categorical);
    assert_ne!(categorical.color(0.), categorical.color(1.));
    assert_eq!(categorical.color(1.), categorical.color(1.2));
    assert_eq!(d.color(f64::NAN), egui::Color32::TRANSPARENT);
}
#[test]
fn native_legend_geometry_is_vertical_adjacent_and_inside_main_card() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let pane = h.app.policy_pane();
    let frame =
        *egui_plot::PlotMemory::load(&h.ctx, rl_viewer::policy::view::PolicyPane::main_plot_id())
            .unwrap()
            .transform()
            .frame();
    let ticks: Vec<_> = h
        .text
        .iter()
        .filter(|(s, r)| {
            s.parse::<f64>().is_ok()
                && r.left() > frame.right()
                && r.right() < pane.main_rect.unwrap().right()
        })
        .collect();
    assert!(
        ticks.len() >= 5,
        "five adjacent numerical legend labels: {ticks:?}"
    );
    let top = ticks
        .iter()
        .map(|(_, r)| r.center().y)
        .fold(f32::INFINITY, f32::min);
    let bottom = ticks.iter().map(|(_, r)| r.center().y).fold(0., f32::max);
    assert!(bottom - top > frame.height() * 0.9);
    assert!(pane.main_rect.unwrap().right() < pane.inspector_rect.unwrap().left());
}
fn assert_numerical_axis_ticks(h: &mut Harness) {
    // Responsive axes use previous-pass geometry; finish the collapse animation too.
    for _ in 0..4 {
        h.frame(vec![]);
    }
    let memory =
        egui_plot::PlotMemory::load(&h.ctx, rl_viewer::policy::view::PolicyPane::main_plot_id())
            .unwrap();
    let frame = *memory.transform().frame();
    for axis in [0, 1] {
        let ticks = h
            .text
            .iter()
            .filter(|(s, rect)| {
                s.parse::<f64>().is_ok_and(|v| v.is_finite() && v != 0.)
                    && if axis == 0 {
                        rect.center().x > frame.left()
                            && rect.center().x < frame.right()
                            && rect.center().y > frame.bottom()
                            && rect.center().y < frame.bottom() + 30.
                    } else {
                        rect.center().y > frame.top()
                            && rect.center().y < frame.bottom()
                            && rect.center().x < frame.left()
                            && rect.center().x > frame.left() - 55.
                    }
            })
            .count();
        assert!(
            ticks >= 2,
            "axis={axis} nonzero ticks={ticks}, frame={frame:?}; numeric={:?}",
            h.text
                .iter()
                .filter(|(s, _)| s.parse::<f64>().is_ok())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn ordinary_route_has_minimal_wide_and_narrow_layout() {
    let dir = fixture(3, 0, true);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    assert_numerical_axis_ticks(&mut h);
    for label in [
        "Policy",
        "Value",
        "Network",
        "Map",
        "Slice",
        "Action",
        "Slice settings",
        "Fixed inputs",
        "Selected",
        "action[0]",
    ] {
        assert!(h.has(label), "{label}");
    }
    for forbidden in [
        "Output mode",
        "Selected input",
        "Probe row",
        "Policy & value",
        "Action distribution",
        "Explore",
        "Advanced",
        "Layer",
    ] {
        assert!(!h.has(forbidden), "{forbidden}");
    }
    let pane = h.app.policy_pane();
    let main = pane.main_rect.unwrap();
    let selected = pane.inspector_rect.unwrap();
    assert!(main.right() < selected.left());
    assert!(
        (480.0..=525.0).contains(&selected.width()),
        "main={main:?} selected={selected:?}"
    );
    assert!(
        main.bottom() <= 901. && selected.right() <= 1441.,
        "main={main:?}, selected={selected:?}"
    );
    let pin = h.app.policy_pane().pinned_input().unwrap().clone();
    h.click("Slice settings");
    assert!(h.has("X axis") && h.has("Y axis"));
    h.click("Slice settings");
    h.click("Fixed inputs");
    assert!(h.has("obs[2]"));
    h.click("Fixed inputs");
    assert_eq!(h.app.policy_pane().pinned_input().unwrap(), &pin);
    h.size = egui::vec2(1920., 1080.);
    h.settle();
    let pane = h.app.policy_pane();
    let main = pane.main_rect.unwrap();
    let selected = pane.inspector_rect.unwrap();
    assert!(
        (650.0..=700.0).contains(&selected.width()),
        "selected={selected:?}"
    );
    assert!(main.right() < selected.left() && selected.right() <= 1921.);
    h.size = egui::vec2(700., 900.);
    h.settle();
    assert_numerical_axis_ticks(&mut h);
    let pane = h.app.policy_pane();
    assert!(pane.main_rect.unwrap().right() <= 701.);
    assert!(h.has("Selected"));
    assert!(pane.inspector_rect.is_none());
    h.click("Selected");
    h.settle();
    assert_numerical_axis_ticks(&mut h);
    let pane = h.app.policy_pane();
    assert!(pane.inspector_rect.unwrap().bottom() <= 901.);
    assert!(
        pane.main_rect.unwrap().bottom() <= pane.inspector_rect.unwrap().top(),
        "main={:?} selected={:?}",
        pane.main_rect,
        pane.inspector_rect
    );
}
#[test]
fn masked_ordinary_route_keeps_full_policy_only_mask_and_sparse_samples() {
    let dir = fixture(3, 80, false);
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let f = h.app.policy_frame().unwrap();
    assert!(!f.grid.regular());
    assert_eq!(f.grid.input(0).unwrap().dense_mask().unwrap().len(), 80);
    assert!(!f.selected.as_ref().unwrap().input.dense_mask().unwrap()[79]);
    assert!(h.has("Action distribution"), "{:?}", h.text);
    assert!(h.has("surface unavailable"));
}

#[test]
fn startup_route_maps_to_the_three_supported_panels() {
    use rl_viewer::ui::View;
    for route in ["", "policy", "policy-map", "explore", "returns", "unknown"] {
        assert_eq!(View::from_startup_hook(route), View::PolicyMap, "{route}");
    }
    assert_eq!(View::from_startup_hook("value"), View::Value);
    assert_eq!(View::from_startup_hook("network"), View::Network);
}

#[test]
fn stochastic_model_distribution_has_a_readable_side_panel() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        core::{action::ActionSpace, env::Observation},
        diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*},
        nn::gaussian_policy::GaussianPolicy,
    };
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let manifest =
        RunManifest::new("policy-width", "ordinary", "PPO", 2, space.clone(), 0.9).unwrap();
    let mut recorder =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("policy-width", 3);
    context.probe_set = Some("fixed".into());
    recorder
        .append(
            &ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "inputs",
                vec![
                    Observation::new(vec![0., 0.]),
                    Observation::new(vec![1., 1.]),
                ],
                None,
                space.clone(),
            )
            .unwrap()
            .record(context)
            .unwrap(),
        )
        .unwrap();
    let model =
        GaussianPolicy::new(2, space.clone(), &[4], &mut ChaCha8Rng::seed_from_u64(8)).unwrap();
    let mut snapshot =
        SnapshotBuilder::new("policy", RecordContext::new("policy-width", 3), 2, space).unwrap();
    snapshot.state_gaussian("actor", &model).unwrap();
    recorder.append(&snapshot.finish().unwrap()).unwrap();
    recorder.finish().unwrap();
    let mut h = Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    let pane = h.app.policy_pane();
    let main = pane.main_rect.unwrap();
    let inspector = pane.inspector_rect.unwrap();
    assert!(inspector.width() >= 400., "{inspector:?}");
    assert!(main.right() < inspector.left());
    assert!(inspector.right() <= h.size.x + 1.);
    assert!(h.has("Action distribution"));
    assert!(h.has("Policy map"));
}
