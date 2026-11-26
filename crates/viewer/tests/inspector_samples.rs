#[path = "support/policy_ui.rs"]
mod policy;
#[path = "support/value_ui.rs"]
#[allow(dead_code)]
mod value;
use eframe::egui;
use rl::{
    core::action::Action,
    diagnostics::{schema::*, storage::*},
};
use rl_viewer::{
    policy::{slice::SliceMode, view::PolicyPane},
    ui::View,
};
use std::time::{Duration, Instant};

#[cfg(feature = "screenshots")]
#[test]
fn explicit_capture_axis_is_rejected_for_categorical_history_population() {
    // Environment hooks are process-global. Exercise the actual native startup
    // hook in an isolated test process rather than mutating this process's env.
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "categorical_policy_has_no_physical_axes_and_navigates_original_history_rows",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("RL_VIEWER_POLICY_X", "0")
        .output()
        .unwrap();
    let message = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        !result.status.success(),
        "an explicit invalid numerical axis must be rejected"
    );
    assert!(
        message.contains("slice axes must be distinct numerical coordinates"),
        "{message}"
    );
}

fn categorical_manifest(dir: &tempfile::TempDir, actions: usize) {
    let path = dir.path().join("run/manifest.json");
    let mut m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let observations: Vec<_> = (0..3).map(|i| serde_json::json!({"id":format!("c{i}"),"label":format!("Category {i}"),"kind":{"Categorical":{"categories":[{"code":-0.4,"label":"Left"},{"code":0.4,"label":"Right"},{"code":0.,"label":"Zero"},{"code":0.25,"label":"Quarter"},{"code":0.5,"label":"Half"},{"code":0.75,"label":"Three quarters"},{"code":999.,"label":"Terminal"}]}},"coordinate":false})).collect();
    let choices: Vec<_> = (0..actions)
        .map(|i| serde_json::json!({"label":format!("Choice {i}")}))
        .collect();
    m["environment_descriptor"] = serde_json::json!({"version":1,"observations":observations,"actions":{"Discrete":{"choices":choices}},"clocks":{"learner_revision":{"label":"Revision"},"decision_step":{"label":"Step"}},"reward":{"label":"Reward"}});
    if actions == 0 {
        m["environment_descriptor"]["actions"] = serde_json::json!({"Continuous":{"components":[{"label":"Effort"},{"label":"Other effort"}],"preferred_component":0}});
    }
    std::fs::write(path, serde_json::to_vec(&m).unwrap()).unwrap();
}
#[test]
fn twelve_masked_history_rows_at_one_numerical_position_promote_selected_without_losing_rows() {
    let base = policy::fixture(3, 8, false);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for i in 0..recording.entries().len() {
        let mut chunk = recording.load(i).unwrap();
        if !matches!(chunk.record, Record::ProbeSet(_) | Record::Policy(_)) {
            continue;
        }
        for array in chunk.arrays.values_mut() {
            let width = array.len() / 2;
            let mut shape = array.shape().to_vec();
            shape[0] = 12;
            *array = match array.dtype() {
                DType::F32 => Array::f32(shape, array.as_f32().unwrap()[..width].repeat(12)),
                DType::F64 => Array::f64(shape, array.as_f64().unwrap()[..width].repeat(12)),
                DType::U64 => Array::u64(shape, array.as_u64().unwrap()[..width].repeat(12)),
                DType::U8 => Array::u8(shape, array.as_u8().unwrap()[..width].repeat(12)),
            }
            .unwrap();
        }
        if let Record::ProbeSet(p) = &mut chunk.record {
            p.population = ProbePopulation::ImportedHistory;
            p.histories.insert("past".into(), "past".into());
            chunk.arrays.insert(
                p.row_ids.clone(),
                Array::u64(vec![12], (0..12).collect()).unwrap(),
            );
            chunk.arrays.insert(
                "past".into(),
                Array::f32(vec![12, 2, 3], (0..72).map(|i| i as f32).collect()).unwrap(),
            );
        }
        writer.append(&chunk).unwrap();
    }
    writer.finish().unwrap();
    let mut h = policy::Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    assert_eq!(h.app.policy_frame().unwrap().grid.population_count, 12);
    assert_eq!(h.app.policy_frame().unwrap().grid.positions.len(), 12);
    assert!(
        h.app.policy_pane().main_rect.unwrap().height() < 180.,
        "one coordinate must be compact even with12 original rows"
    );
    assert!(h.app.policy_pane().inspector_rect.unwrap().width() > 1300.);
    for row in 0..12 {
        let f = h.app.policy_frame().unwrap();
        let p = f.selected.as_ref().unwrap();
        assert!(
            matches!(&p.recorded_input, Some(rl_viewer::data::SelectedInput::ProbeRow { row:r,.. }) if *r==row)
        );
        assert!(p.input.dense_mask().unwrap().iter().any(|v| !*v));
        assert!(
            matches!(f.grid.recorded_probes().unwrap().record,Record::ProbeSet(ref p) if !p.histories.is_empty())
        );
        if row < 11 {
            h.click("Next sample");
            h.settle();
        }
    }
}
fn settle_value(h: &mut policy::Harness) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        h.frame(vec![]);
        if h.app.value_view().ready() {
            h.frame(vec![]);
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
fn categorical_policy_has_no_physical_axes_and_navigates_original_history_rows() {
    let base = policy::fixture(3, 8, false);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for i in 0..recording.entries().len() {
        let mut chunk = recording.load(i).unwrap();
        if let Record::ProbeSet(p) = &mut chunk.record {
            p.population = ProbePopulation::ImportedHistory;
            p.histories.insert("past".into(), "past".into());
            chunk.arrays.insert(
                "past".into(),
                Array::f32(
                    vec![2, 2, 3],
                    vec![0., 0., 0., 0., 0., 0., 1., 1., 1., 2., 2., 2.],
                )
                .unwrap(),
            );
        }
        writer.append(&chunk).unwrap();
    }
    writer.finish().unwrap();
    categorical_manifest(&dir, 8);
    let mut h = policy::Harness::new(&dir, egui::vec2(1440., 900.));
    h.settle();
    assert_eq!(
        h.app.policy_pane().config().unwrap().mode,
        SliceMode::SamplesOnly
    );
    let old = h
        .app
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    assert_eq!(h.app.policy_frame().unwrap().grid.population_count, 2);
    assert!(h.app.policy_frame().unwrap().grid.positions.is_empty());
    assert!(egui_plot::PlotMemory::load(&h.ctx, PolicyPane::main_plot_id()).is_none());
    assert!(h.has("Entropy"));
    assert!(h.has("Probability"));
    assert!(h.has("Recorded history inputs"));
    assert!(!h.has("X axis"));
    h.click("Next sample");
    h.settle();
    let f = h.app.policy_frame().unwrap();
    assert_eq!(
        f.selected.as_ref().unwrap().input.dense_values(),
        vec![0.5; 3]
    );
    assert_eq!(
        f.selected.as_ref().unwrap().input.dense_mask(),
        old.dense_mask()
    );
    assert!(matches!(
        f.selected.as_ref().unwrap().recorded_input,
        Some(rl_viewer::data::SelectedInput::ProbeRow { row: 1, .. })
    ));
    assert!(h.has("Sample 2 of 2"));
    let metrics = h.app.policy_metrics();
    for _ in 0..10 {
        h.frame(vec![]);
    }
    assert_eq!(h.app.policy_metrics().point_batches, metrics.point_batches);
    assert_eq!(h.app.policy_metrics().restores, 0);
}
#[test]
fn categorical_value_promotes_one_masked_q_chart_with_original_history() {
    let dir = value::discovery(true, true, true, true);
    categorical_manifest(&dir, 4);
    let mut h = policy::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle_value(&mut h);
    let f = h.app.value_frame().unwrap();
    assert_eq!(f.grid.slice.mode, SliceMode::SamplesOnly);
    assert_eq!(f.grid.population_count, 1);
    assert!(f.grid.positions.is_empty() && f.grid.values.is_empty());
    assert_eq!(
        f.selected.as_ref().unwrap().input.dense_mask(),
        Some(&[true, false, true, false][..])
    );
    assert!(h.has("Action values") && h.has("Choice"));
    assert!(!h.has("X axis"));
    let plot =
        egui_plot::PlotMemory::load(&h.ctx, egui::Id::unique("value-selected-actions")).unwrap();
    let pos = plot
        .transform()
        .position_from_point(&egui_plot::PlotPoint::new(2., 0.));
    h.click_at(pos);
    settle_value(&mut h);
    let selected = h.app.value_frame().unwrap().selected.as_ref().unwrap();
    assert_eq!(selected.action, Some(Action::Discrete(2)));
    assert_eq!(selected.recorded_row, Some(0));
    assert_eq!(h.app.value_metrics().restores, 0);
}

#[test]
fn hidden_illegal_reference_q_outliers_do_not_flatten_visible_bars() {
    use rl::core::env::Observation;
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = value::family("discrete", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for index in [1, 2] {
        let model =
            ModelBundle::load(&recording.load(index).unwrap(), ModelLimits::default()).unwrap();
        let mut result = model
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["actions".into()],
                observations: vec![
                    Observation::dense(vec![0.; 3], Some(vec![true, false, true, false])),
                    Observation::dense(vec![0.25, 0.5, 0.75], Some(vec![false, true, false, true])),
                ],
                fixed_actions: None,
            })
            .unwrap();
        let critic = result.critics.get_mut("actions").unwrap();
        let Record::Critic(c) = &critic.record else {
            panic!()
        };
        let CriticRepresentation::Discrete { values } = &c.representation else {
            panic!()
        };
        critic.arrays.insert(
            values.clone(),
            Array::f32(
                vec![2, 4],
                if index == 1 {
                    vec![1., 1e9, 2., 1e9, 1e9, 0., 1e9, 1.5]
                } else {
                    // Current invalid actions have actual gray bars, including Q=2.
                    vec![0., 0., 1., 0., 2., 1., 0., 1.]
                },
            )
            .unwrap(),
        );
        w.append(&result.probes).unwrap();
        w.append(critic).unwrap();
    }
    w.finish().unwrap();
    categorical_manifest(&dir, 4);
    let mut h = policy::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle_value(&mut h);
    let id = egui::Id::unique("value-selected-actions");
    let mut plot = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
    plot.set_bounds(egui_plot::PlotBounds::from_min_max(
        [-0.25, -1.],
        [3.25, 3.],
    ));
    plot.store(&h.ctx, id);
    h.click_last("Reference");
    settle_value(&mut h);
    h.click("Reference source");
    let k = h.text.iter().find(|(s, _)| s == "k").unwrap().1;
    let revision = h
        .text
        .iter()
        .find(|(s, r)| {
            s.parse::<u64>().is_ok()
                && r.left() > k.right()
                && (r.center().y - k.center().y).abs() < 4.
        })
        .unwrap()
        .1
        .center();
    h.click_at(revision);
    h.frame(vec![
        egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        },
        egui::Event::Text("3".into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    h.click("Reference source");
    settle_value(&mut h);
    for row in 0..2 {
        if row > 0 {
            h.click("Next sample");
            settle_value(&mut h);
        }
        let selected = h.app.value_frame().unwrap().selected.as_ref().unwrap();
        assert_eq!(selected.recorded_row, Some(row));
        let rl::diagnostics::value::ValuePrediction::Discrete { values, legal } =
            selected.reference_prediction.as_ref().unwrap()
        else {
            panic!()
        };
        assert!(
            values
                .iter()
                .zip(legal)
                .any(|(&v, &valid)| !valid && v == 1e9)
        );
        let plot = egui_plot::PlotMemory::load(&h.ctx, id).unwrap();
        let bounds = *plot.transform().bounds();
        eprintln!("masked reference row={row}: bounds={bounds:?}");
        assert!(
            bounds.min()[1] > -0.5 && bounds.max()[1] >= 2. && bounds.max()[1] < 2.5,
            "hidden illegal reference values must not flatten visible Q=0..2 bars"
        );
        assert_eq!(bounds.min()[0], -0.25);
        assert_eq!(bounds.max()[0], 3.25);
    }
}

#[test]
fn identical_categorical_value_states_keep_distinct_original_history_rows_and_masks() {
    use rl::core::env::Observation;
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = value::family("discrete", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let model = ModelBundle::load(&recording.load(2).unwrap(), ModelLimits::default()).unwrap();
    let mut result = model
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["actions".into()],
            observations: (0..3)
                .map(|i| {
                    Observation::dense(
                        vec![0.; 3],
                        Some(if i == 2 {
                            vec![false, true, false, true]
                        } else {
                            vec![true, false, true, false]
                        }),
                    )
                })
                .collect(),
            fixed_actions: None,
        })
        .unwrap();
    if let Record::ProbeSet(p) = &mut result.probes.record {
        p.population = ProbePopulation::ImportedHistory;
        p.histories.insert("past".into(), "past".into());
    }
    result.probes.arrays.insert(
        "past".into(),
        Array::f32(vec![3, 2, 3], (0..18).map(|i| i as f32).collect()).unwrap(),
    );
    let critic = result.critics.get_mut("actions").unwrap();
    if let Record::Critic(c) = &critic.record
        && let CriticRepresentation::Discrete { values } = &c.representation
    {
        // Distinct typed recorded outputs bind the selected history row. These
        // are controlled fixture values, never a trained-model demonstration.
        critic.arrays.insert(
            values.clone(),
            Array::f32(
                vec![3, 4],
                vec![1., 0., 2., 0., 3., 0., 4., 0., 0., 5., 0., 6.],
            )
            .unwrap(),
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    w.append(&result.probes).unwrap();
    w.append(&result.critics["actions"]).unwrap();
    w.finish().unwrap();
    categorical_manifest(&dir, 4);
    let mut h = policy::Harness::new(&dir, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle_value(&mut h);
    let plot_id = egui::Id::unique("value-selected-actions");
    let mut plot = egui_plot::PlotMemory::load(&h.ctx, plot_id).unwrap();
    plot.set_bounds(egui_plot::PlotBounds::from_min_max(
        [-0.25, -1.],
        [3.25, 3.],
    ));
    plot.store(&h.ctx, plot_id);
    for row in 1..3 {
        h.click("Next sample");
        settle_value(&mut h);
        let f = h.app.value_frame().unwrap();
        assert_eq!(f.grid.population_count, 3);
        let p = f.selected.as_ref().unwrap();
        assert_eq!(p.recorded_row, Some(row));
        assert_eq!(p.value, if row == 1 { 4. } else { 6. });
        let plot = egui_plot::PlotMemory::load(&h.ctx, egui::Id::unique("value-selected-actions"))
            .unwrap();
        assert!(
            plot.transform().bounds().max()[1] >= p.value,
            "selected Q must remain visible after original-row navigation"
        );
        assert_eq!(plot.transform().bounds().min()[0], -0.25);
        assert_eq!(plot.transform().bounds().max()[0], 3.25);
        assert_eq!(p.input.dense_values(), vec![0.; 3]);
        assert_eq!(
            p.input.dense_mask(),
            Some(if row == 2 {
                &[false, true, false, true][..]
            } else {
                &[true, false, true, false][..]
            })
        );
        assert!(h.has(&format!("Sample {} of 3", row + 1)));
    }
    assert_eq!(h.app.value_metrics().restores, 0);
}

#[test]
fn deterministic_continuous_and_gaussian_selected_capabilities_are_distinct() {
    for gaussian in [false, true] {
        let d = policy::comparison_fixture(gaussian.then_some(0.));
        let mut h = policy::Harness::new(&d, egui::vec2(1440., 900.));
        h.settle();
        let p = h.app.policy_frame().unwrap().selected.as_ref().unwrap();
        if gaussian {
            assert_eq!(p.action_statistics.unwrap().standard_deviation, 1.);
            assert!(h.has("Action mean") && h.has("std"));
            assert!(h.has("Density") && h.has("Current"));
            assert!(
                egui_plot::PlotMemory::load(&h.ctx, PolicyPane::distribution_plot_id()).is_some()
            );
        } else {
            assert!(p.action_statistics.is_none() && p.plot.is_none() && p.kl.is_none());
            assert!(h.has("action[0]"));
            assert!(!h.has("Action distribution") && !h.has("Entropy") && !h.has("density"));
        }
    }
}

#[test]
fn samples_only_quantile_output_and_probability_edits_preserve_exact_conditioning() {
    use rl::diagnostics::value::ValueOutput;
    let d = value::sparse();
    categorical_manifest(&d, 0);
    let mut h = policy::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle_value(&mut h);
    assert_eq!(
        h.app.value_frame().unwrap().grid.slice.mode,
        SliceMode::SamplesOnly
    );
    let before = h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .input
        .clone();
    let action = h
        .app
        .value_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .action
        .clone();
    h.click("Expected return");
    h.click("Quantile");
    settle_value(&mut h);
    let prefix = h.text.iter().find(|(s, _)| s.starts_with("p =")).unwrap().1;
    let pos = h
        .text
        .iter()
        .find(|(s, r)| {
            s.parse::<f64>().is_ok()
                && r.left() >= prefix.right()
                && (r.center().y - prefix.center().y).abs() < 3.
        })
        .unwrap()
        .1
        .center();
    h.frame(vec![
        egui::Event::PointerMoved(pos),
        policy::button(pos, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(pos + egui::vec2(15., 0.))]);
    h.frame(vec![policy::button(pos + egui::vec2(15., 0.), false)]);
    settle_value(&mut h);
    let f = h.app.value_frame().unwrap();
    let ValueOutput::Quantile { probability } = f.grid.output() else {
        panic!()
    };
    assert_ne!(*probability, 0.5);
    let p = f.selected.as_ref().unwrap();
    assert_eq!(p.input, before);
    assert_eq!(p.action, action);
    assert_eq!(
        p.value,
        p.prediction
            .ordered_quantiles()
            .unwrap()
            .quantile(*probability)
            .unwrap()
    );
    assert!(h.has("Return profile") && h.has("Raw quantile details"));
    assert!(f.grid.positions.is_empty());
    h.click("Next sample");
    settle_value(&mut h);
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values(),
        vec![0.4, 0., 0.]
    );
    assert_eq!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .action,
        action
    );
}

#[test]
fn samples_only_keeps_distinct_legacy_critic_choices_and_absent_reference_capability() {
    let d = value::discovery(false, false, false, true);
    categorical_manifest(&d, 4);
    let mut h = policy::Harness::new(&d, egui::vec2(1440., 900.));
    h.app.select_view(View::Value);
    settle_value(&mut h);
    assert_eq!(
        h.app.value_frame().unwrap().grid.current().identity().name,
        "Recorded critic #2"
    );
    h.click("Recorded critic #2");
    h.click("Recorded critic #1");
    settle_value(&mut h);
    assert_eq!(
        h.app.value_frame().unwrap().grid.current().identity().name,
        "Recorded critic #1"
    );
    h.click_last("Reference");
    settle_value(&mut h);
    assert!(
        h.app
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value
            .is_none()
    );
}

#[test]
fn value_visible_projection_promotes_overlap_and_filtered_populations_only() {
    use rl_viewer::value::slice::ValueCoordinate;
    for kind in ["discrete", "quantile"] {
        let mut cdf_heights = Vec::new();
        for pattern in ["overlap", "filtered", "spread"] {
            let d = value::projection_population(kind, 12, pattern);
            let mut h = policy::Harness::new(&d, egui::vec2(1440., 1100.));
            h.app.select_view(View::Value);
            settle_value(&mut h);
            h.app
                .value_view_mut()
                .set_axes(
                    ValueCoordinate::Observation(0),
                    Some(ValueCoordinate::Observation(1)),
                )
                .unwrap();
            settle_value(&mut h);
            let g = &h.app.value_frame().unwrap().grid;
            assert_eq!(g.population_count, 12);
            assert_eq!(
                g.positions.len(),
                if pattern == "filtered" {
                    2
                } else if kind == "discrete" && pattern == "overlap" {
                    8
                } else {
                    12
                }
            );
            let promoted = pattern != "spread";
            assert_eq!(
                h.app.value_view().main_rect.unwrap().height() < 180.,
                promoted,
                "{kind}/{pattern} compact plane"
            );
            assert_eq!(
                h.app.value_view().inspector_rect.unwrap().width() > 1300.,
                promoted,
                "{kind}/{pattern} inspector"
            );
            assert!(h.has(if kind == "discrete" {
                "Action values"
            } else {
                "Return profile"
            }));
            if kind == "quantile" {
                let profile = h
                    .text
                    .iter()
                    .find(|(s, _)| s == "Return profile")
                    .unwrap()
                    .1;
                let details = h
                    .text
                    .iter()
                    .find(|(s, _)| s == "Raw quantile details")
                    .unwrap()
                    .1;
                let inspector = h.app.value_view().inspector_rect.unwrap();
                // A borderless CDF has no background rectangle. Its actual
                // curve clip measures the usable plot frame independently of
                // headings, warnings, axis labels and decorative surfaces.
                h.time += 0.1;
                let mut output = h.ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, h.size)),
                        time: Some(h.time),
                        ..Default::default()
                    },
                    |ui| h.app.show(ui),
                );
                output.textures_delta.clear();
                fn cdf_curve(shape: &egui::Shape, top: f32, bottom: f32) -> Option<&[egui::Pos2]> {
                    match shape {
                        egui::Shape::Path(path)
                            if path.points.len() > 2
                                && matches!(path.stroke.color, egui::epaint::ColorMode::Solid(color) if color == rl_viewer::theme::CURRENT)
                                && path
                                    .points
                                    .iter()
                                    .all(|point| point.y > top && point.y < bottom) =>
                        {
                            Some(&path.points)
                        }
                        egui::Shape::Vec(shapes) => shapes
                            .iter()
                            .find_map(|shape| cdf_curve(shape, top, bottom)),
                        _ => None,
                    }
                }
                let (plot, curve, vertices) = output
                    .shapes
                    .iter()
                    .find_map(|shape| {
                        cdf_curve(&shape.shape, profile.bottom(), details.top()).map(|points| {
                            (
                                shape.clip_rect,
                                egui::Rect::from_points(points),
                                points.len(),
                            )
                        })
                    })
                    .expect("painted complete CDF curve between its heading and details");
                assert!(
                    plot.expand(1. / h.ctx.pixels_per_point())
                        .contains_rect(curve),
                    "CDF clipped for {pattern}: curve {curve:?} / plot {plot:?}"
                );
                let Some(rl_viewer::value::prepared::PreparedValueProfile::Quantiles {
                    cdf, ..
                }) = &h
                    .app
                    .value_frame()
                    .unwrap()
                    .selected
                    .as_ref()
                    .unwrap()
                    .profile
                else {
                    panic!("quantile fixture must retain its complete CDF");
                };
                assert_eq!(
                    vertices,
                    cdf.len(),
                    "all actual CDF step vertices remain painted for {pattern}"
                );
                // Plot paint snaps to physical pixels; the enclosing layout
                // rectangle can retain a fractional edge before that rounding.
                let rounded_inspector = inspector.expand(1. / h.ctx.pixels_per_point());
                assert!(
                    rounded_inspector.contains_rect(plot),
                    "CDF inspector {pattern}: {plot:?} / {inspector:?}"
                );
                assert!(
                    inspector.contains_rect(details),
                    "CDF details must remain visible"
                );
                let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, h.size);
                assert!(screen.contains_rect(plot) && screen.contains_rect(details));
                assert_eq!(
                    plot.height() >= 300.,
                    promoted,
                    "CDF geometry {pattern}: {plot:?}"
                );
                cdf_heights.push((pattern, plot.height()));
            }
            let before = h.app.value_metrics();
            for _ in 0..10 {
                h.frame(vec![]);
            }
            assert_eq!(
                format!("{:?}", h.app.value_metrics()),
                format!("{before:?}"),
                "idle work {kind}/{pattern}"
            );
            if kind == "discrete" && pattern == "spread" {
                h.app
                    .value_view_mut()
                    .set_axes(
                        ValueCoordinate::DiscreteAction,
                        Some(ValueCoordinate::Observation(0)),
                    )
                    .unwrap();
                settle_value(&mut h);
                assert!(h.app.value_view().main_rect.is_some());
                assert!(!h.app.value_frame().unwrap().grid.promote_selected());
                assert!(
                    !h.has("Action values"),
                    "actual main category chart suppresses duplicate"
                );
                h.app
                    .value_view_mut()
                    .set_axes(ValueCoordinate::DiscreteAction, None)
                    .unwrap();
                settle_value(&mut h);
                assert!(h.app.value_frame().unwrap().grid.promote_selected());
                assert!(h.app.value_view().main_rect.is_none());
                assert!(
                    h.has("Action values"),
                    "selected chart remains when main chart is absent"
                );
            }
            if kind == "discrete" && pattern == "overlap" {
                for row in 0..12 {
                    let f = h.app.value_frame().unwrap();
                    let p = f.selected.as_ref().unwrap();
                    assert_eq!(p.recorded_row, Some(row));
                    let mask = if row % 3 == 2 {
                        vec![false, true, false, true]
                    } else {
                        vec![true, false, true, false]
                    };
                    assert_eq!(p.input.dense_mask(), Some(mask.as_slice()));
                    assert_eq!(p.value, (row * 4 + if row % 3 == 2 { 4 } else { 3 }) as f64);
                    assert!(
                        f.grid
                            .recorded_population()
                            .unwrap()
                            .histories
                            .contains_key("past")
                    );
                    assert!(f.grid.population_selection(row).is_some());
                    if row < 11 {
                        h.click("Next sample");
                        settle_value(&mut h);
                    }
                }
            }
        }
        if kind == "quantile" {
            let ordinary = cdf_heights
                .iter()
                .find(|(pattern, _)| *pattern == "spread")
                .unwrap()
                .1;
            assert!(ordinary <= 280.);
            for (pattern, height) in cdf_heights {
                if pattern != "spread" {
                    assert!(
                        height >= 1.25 * ordinary,
                        "promoted CDF {pattern}: {height} vs {ordinary}"
                    );
                }
            }
        }
    }
}
