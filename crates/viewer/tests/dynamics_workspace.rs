#[path = "support/dynamics.rs"]
mod fixture;
#[path = "support/policy_ui.rs"]
mod support;

use eframe::egui;
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*, visualization::*},
};
use rl_viewer::{
    dynamics::{ActionAxis, ActionValue, work::Projection},
    ui::View,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn recording(space: ActionSpace, numeric: usize) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let mut m = fixture::manifest(space.clone());
    if numeric == 2 {
        m.environment_descriptor.as_mut().unwrap().observations[2].kind =
            ObservationType::Categorical {
                categories: vec![Category {
                    code: 10.,
                    label: "phase".into(),
                }],
            };
    }
    let mut w = Recorder::create(d.path().join("run"), m, StorageConfig::default()).unwrap();
    for revision in [17, 63] {
        let mut c = fixture::chunk(&space);
        c.context.identity.model_role = "online".into();
        c.context.identity.learner_update_revision = revision;
        c.context.counters.updates = revision;
        if revision == 63 {
            if let Record::Trajectory(record) = &mut c.record {
                record.evaluation.complete_returns = false;
            }
            c.arrays
                .insert("terminated".into(), Array::u8(vec![4], vec![0; 4]).unwrap());
            c.arrays.insert(
                "truncated".into(),
                Array::u8(vec![4], vec![0, 0, 0, 1]).unwrap(),
            );
        }
        w.append(&c).unwrap();
    }
    w.append(&Chunk {
        context: RecordContext::new("dynamics", 80),
        record: Record::Metrics(MetricRecord {
            values: std::collections::BTreeMap::from([("loss".into(), 0.1)]),
            unavailable: Default::default(),
        }),
        arrays: Default::default(),
    })
    .unwrap();
    w.finish().unwrap();
    d
}
fn protocol_recording() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(1);
    let mut w = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    for (revision, caption) in [(17, "first policy"), (63, "alternate policy")] {
        let mut c = fixture::chunk(&space);
        c.context.identity.model_role = "online".into();
        c.context.identity.learner_update_revision = revision;
        c.context.counters.updates = revision;
        if let Record::Trajectory(record) = &mut c.record {
            record.evaluation.continuation_policy = caption.into();
        }
        w.append(&c).unwrap();
    }
    w.finish().unwrap();
    d
}
fn one_decision_recording() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut w = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    let mut c = fixture::chunk(&space);
    c.context.identity.model_role = "online".into();
    c.arrays.insert(
        "obs".into(),
        Array::f64(vec![2, 3], vec![0., 0., 0., 1., 1., 1.]).unwrap(),
    );
    c.arrays
        .insert("actions".into(), Array::f64(vec![1, 1], vec![10.]).unwrap());
    c.arrays
        .insert("rewards".into(), Array::f64(vec![1], vec![2.]).unwrap());
    c.arrays
        .insert("terminated".into(), Array::u8(vec![1], vec![1]).unwrap());
    c.arrays
        .insert("truncated".into(), Array::u8(vec![1], vec![0]).unwrap());
    c.arrays
        .insert("times".into(), Array::f64(vec![2], vec![0., 1.]).unwrap());
    w.append(&c).unwrap();
    w.finish().unwrap();
    d
}
fn long_spike_recording() -> tempfile::TempDir {
    const T: usize = 5000;
    const SPIKE: usize = 2499;
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(1);
    let mut w = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    let mut c = fixture::chunk(&space);
    c.context.identity.model_role = "online".into();
    c.arrays.insert(
        "obs".into(),
        Array::f64(vec![T + 1, 3], vec![1.; (T + 1) * 3]).unwrap(),
    );
    let mut values = vec![1.; T];
    values[SPIKE] = 100.;
    c.arrays.insert(
        "actions".into(),
        Array::f64(vec![T, 1], values.clone()).unwrap(),
    );
    c.arrays
        .insert("rewards".into(), Array::f64(vec![T], values).unwrap());
    let mut done = vec![0; T];
    done[T - 1] = 1;
    c.arrays
        .insert("terminated".into(), Array::u8(vec![T], done).unwrap());
    c.arrays
        .insert("truncated".into(), Array::u8(vec![T], vec![0; T]).unwrap());
    c.arrays.insert(
        "times".into(),
        Array::f64(vec![T + 1], (0..=T).map(|t| t as f64).collect()).unwrap(),
    );
    w.append(&c).unwrap();
    w.finish().unwrap();
    d
}
fn edge_sampling_recording() -> tempfile::TempDir {
    const T: usize = 9000;
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(1);
    let mut w = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    let mut c = fixture::chunk(&space);
    c.context.identity.model_role = "online".into();
    c.arrays.insert(
        "obs".into(),
        Array::f64(vec![T + 1, 3], vec![1.; (T + 1) * 3]).unwrap(),
    );
    c.arrays.insert(
        "actions".into(),
        Array::f64(vec![T, 1], vec![0.; T]).unwrap(),
    );
    c.arrays
        .insert("rewards".into(), Array::f64(vec![T], vec![0.; T]).unwrap());
    let mut done = vec![0; T];
    done[T - 1] = 1;
    c.arrays
        .insert("terminated".into(), Array::u8(vec![T], done).unwrap());
    c.arrays
        .insert("truncated".into(), Array::u8(vec![T], vec![0; T]).unwrap());
    c.arrays.insert(
        "times".into(),
        Array::f64(vec![T + 1], (0..=T).map(|t| t as f64).collect()).unwrap(),
    );
    w.append(&c).unwrap();
    w.finish().unwrap();
    d
}
fn long_label_recording() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(3);
    let mut manifest = fixture::manifest(space.clone());
    let descriptor = manifest.environment_descriptor.as_mut().unwrap();
    for (i, field) in descriptor.observations.iter_mut().enumerate() {
        field.label = format!("LQG coordinate {i} [model units]");
    }
    if let ActionDescriptor::Continuous { components, .. } = &mut descriptor.actions {
        for (i, field) in components.iter_mut().enumerate() {
            field.label = format!("Control component {i} [model units]");
        }
    }
    let mut w = Recorder::create(d.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut c = fixture::chunk(&space);
    c.context.identity.model_role = "online".into();
    c.context.trajectory = Some("evaluation-long-trajectory-sequence".into());
    if let Record::Trajectory(record) = &mut c.record {
        record.id = "evaluation-long-trajectory-sequence".into();
    }
    w.append(&c).unwrap();
    w.finish().unwrap();
    d
}
fn four_revision_recording() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(1);
    let mut w = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    for revision in 1..=4 {
        let mut c = fixture::chunk(&space);
        c.context.identity.model_role = "online".into();
        c.context.identity.learner_update_revision = revision;
        c.context.counters.updates = revision;
        w.append(&c).unwrap();
    }
    w.finish().unwrap();
    d
}
fn open(d: &tempfile::TempDir, size: egui::Vec2) -> support::Harness {
    let mut h = support::Harness::new(d, size);
    h.app.select_view(View::Dynamics);
    h
}
fn settle(h: &mut support::Harness) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        h.frame(vec![]);
        if h.app.dynamics_ready() {
            h.frame(vec![]);
            return;
        }
        assert!(
            Instant::now() < deadline,
            "Dynamics did not settle: {:?}",
            h.text
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn every_action_component_updates_timeline_and_state_action_without_rebuilding_state() {
    for space in [
        fixture::bounded(3),
        ActionSpace::Unbounded { dimensions: 3 },
    ] {
        let d = recording(space, 3);
        let mut h = open(&d, egui::vec2(1440., 900.));
        settle(&mut h);
        assert_eq!(
            h.app.dynamics_frame().unwrap().action,
            ActionAxis::Continuous(2)
        );
        assert!(h.has("Motor 2 [N]"));
        h.click("3D");
        h.click("Spatial settings");
        h.click("State–action");
        h.click("Spatial settings");
        settle(&mut h);
        let state = h.app.dynamics_frame().unwrap().state.clone();
        let selected = h.app.dynamics_frame().unwrap().selected.clone();
        for j in 0..3 {
            let current = h.app.dynamics_frame().unwrap().action;
            let ActionAxis::Continuous(current) = current else {
                panic!()
            };
            h.click(&format!("Motor {current} [N]"));
            for axis in 0..3 {
                assert!(h.has(&format!("Motor {axis} [N]")));
            }
            h.click_last(&format!("Motor {j} [N]"));
            settle(&mut h);
            let f = h.app.dynamics_frame().unwrap();
            assert_eq!(f.action, ActionAxis::Continuous(j));
            assert_eq!(f.projection, Projection::StateAction);
            assert_eq!(
                f.action_trace[1].value,
                ActionValue::Continuous((10 * j + 1) as f64)
            );
            assert_eq!(f.geometry3[0].points[0][2], (10 * j) as f64);
            assert!(
                f.geometry3.iter().all(|s| s.t < 3),
                "terminal gets no action"
            );
            assert!(Arc::ptr_eq(&state, &f.state));
            assert!(Arc::ptr_eq(&selected, &f.selected));
        }
    }
}

#[test]
fn two_numeric_coordinates_disable_state3_and_keep_categorical_actions_as_steps() {
    let d = recording(ActionSpace::Discrete { count: 3 }, 2);
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    h.click("3D");
    h.click("Spatial settings");
    h.frame(vec![]);
    assert!(h.has("Three numerical state coordinates required"));
    assert!(h.has("Categorical actions have no continuous 3D axis"));
    h.click("Spatial settings");
    assert_eq!(h.app.dynamics_frame().unwrap().projection, Projection::None);
    assert!(h.has("Action ID"));
    let f = h.app.dynamics_frame().unwrap();
    assert_eq!(f.action, ActionAxis::Discrete);
    assert_eq!(f.action_trace[1].value, ActionValue::Discrete(2));
    h.click("Next step");
    assert!(
        h.has("choice 2"),
        "{:?}",
        h.text.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );
}

#[test]
fn sparse_evaluation_revision_cursor_modes_terminal_and_idle_paints_are_linked() {
    let d = recording(fixture::bounded(1), 3);
    let mut h = open(&d, egui::vec2(1920., 1080.));
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 63);
    assert!(h.has("truncated prefix"));
    h.click("Spatial settings");
    assert!(h.has("Evaluation k=63"));
    h.click("Spatial settings");
    let frame = h.app.dynamics_frame().unwrap().state.clone();
    let builds = h.app.dynamics_render_builds();
    for _ in 0..10 {
        h.frame(vec![]);
    }
    assert_eq!(h.app.dynamics_render_builds(), builds);
    assert!(Arc::ptr_eq(&frame, &h.app.dynamics_frame().unwrap().state));
    h.click("Flow");
    h.click("Coverage");
    h.click("Occupancy");
    assert!(Arc::ptr_eq(&frame, &h.app.dynamics_frame().unwrap().state));
    h.click("Final state");
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::TrajectoryStep { step: 4, .. }
    ));
    assert!(h.has("— no action") && h.has("— no reward"));
    h.click("Decision details");
    h.frame(vec![]);
    assert!(h.has("Undiscounted prefix reward sum"));
    h.app.select_revision(17).unwrap();
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 17);
    assert_eq!(
        h.app.dynamics_frame().unwrap().state.coverage.points.len(),
        2
    );
    assert!(Arc::ptr_eq(
        &frame.reference,
        &h.app.dynamics_frame().unwrap().state.reference
    ));
}

#[test]
fn desktop_and_normal_window_keep_three_chart_bounds_visible() {
    let d = recording(fixture::bounded(1), 3);
    for size in [egui::vec2(1920., 1080.), egui::vec2(1200., 800.)] {
        let mut h = open(&d, size);
        settle(&mut h);
        let bounds = h.app.dynamics_plot_bounds();
        assert_eq!(bounds.len(), 3);
        assert!(bounds[0].right() < bounds[1].left(), "{bounds:?}");
        assert!(bounds[1].bottom() < bounds[2].top(), "{bounds:?}");
        for rect in bounds {
            assert!(
                rect.is_positive()
                    && rect.min.x >= 0.
                    && rect.max.x <= size.x
                    && rect.max.y <= size.y,
                "{rect:?} in {size:?}"
            );
        }
        assert!(h.has("Episode timeline") && h.has("Evaluation coverage"));
        for label in ["Policy", "Value", "Network", "Dynamics"] {
            assert!(h.has(label));
        }
        assert!(!h.has("Explore") && !h.has("Learning Progress"));
    }
}

#[test]
fn episode_framing_and_terminal_reward_keep_recorded_values_accessible() {
    let d = tempfile::tempdir().unwrap();
    let space = fixture::bounded(3);
    let mut writer = Recorder::create(
        d.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    let mut chunk = fixture::chunk(&space);
    chunk.context.identity.model_role = "online".into();
    chunk.arrays.insert(
        "rewards".into(),
        Array::f64(vec![4], vec![-0.1, -0.2, 0.1, 100.]).unwrap(),
    );
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    let mut h = open(&d, egui::vec2(1280., 800.));
    settle(&mut h);
    assert!(h.has("Reward · terminal 100.000 outside scale"));
    let state = h.app.dynamics_frame().unwrap().state.clone();
    let builds = h.app.dynamics_render_builds();
    h.click("Fit episode");
    h.click("Fit episode");
    assert!(Arc::ptr_eq(&state, &h.app.dynamics_frame().unwrap().state));
    assert_eq!(builds, h.app.dynamics_render_builds());
    h.click("Final state");
    h.click("Previous step");
    assert!(
        h.has("100.000"),
        "the exact terminal reward remains in the selected transition"
    );
}

#[test]
fn requested_initial_revision_and_numeric_axis_change_keep_recorded_episode() {
    let d = recording(fixture::bounded(3), 3);
    let mut h = open(&d, egui::vec2(1440., 900.));
    h.app.select_revision(17).unwrap();
    settle(&mut h);
    let first = h.app.dynamics_frame().unwrap();
    assert_eq!(first.state.revision, 17);
    assert_eq!(
        first.state.key.axes(),
        rl_viewer::dynamics::Axes { x: 1, y: 0 }
    );
    let sequence = first.selected.key.sequence;
    let rewards = first.selected.reward_trace.clone();
    h.click("Spatial settings");
    h.click_last("Position 1 [m]");
    assert!(h.has("Position 2 [m]"));
    h.click_last("Position 2 [m]");
    settle(&mut h);
    let next = h.app.dynamics_frame().unwrap();
    assert_eq!(next.state.revision, 17);
    assert_eq!(
        next.state.key.axes(),
        rl_viewer::dynamics::Axes { x: 2, y: 0 }
    );
    assert_eq!(next.selected.key.sequence, sequence);
    assert_eq!(next.selected.reward_trace, rewards);
}

#[test]
fn state_3d_axes_remain_distinct_when_xy_changes_and_view_toggles() {
    let d = recording(fixture::bounded(1), 3);
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    h.click("3D");
    h.click("Spatial settings");
    h.click("State");
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().projection,
        Projection::State([1, 0, 2])
    );
    h.click_last("Position 1 [m]");
    h.click_last("Position 2 [m]");
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().projection,
        Projection::State([2, 0, 1])
    );
    h.click("Spatial settings");
    h.click_last("Position 0 [m]");
    h.click_last("Position 1 [m]");
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().projection,
        Projection::State([2, 1, 0])
    );
    h.click("2D");
    settle(&mut h);
    h.click("Spatial settings");
    h.click_last("Position 2 [m]");
    h.click_last("Position 0 [m]");
    settle(&mut h);
    h.click("3D");
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().projection,
        Projection::State([0, 1, 2])
    );
}

#[test]
fn shortcut_entry_preserves_the_requested_checkpoint_and_uses_the_exact_or_prior_episode() {
    let directory = tempfile::tempdir().unwrap();
    let space = fixture::bounded(1);
    let mut writer = Recorder::create(
        directory.path().join("run"),
        fixture::manifest(space.clone()),
        StorageConfig::default(),
    )
    .unwrap();
    for (revision, caption) in [(17, "first policy"), (63, "alternate policy")] {
        let mut chunk = fixture::chunk(&space);
        chunk.context.identity.model_role = "online".into();
        chunk.context.identity.learner_update_revision = revision;
        chunk.context.counters.updates = revision;
        if let Record::Trajectory(record) = &mut chunk.record {
            record.evaluation.continuation_policy = caption.into();
        }
        writer.append(&chunk).unwrap();
    }
    writer
        .append(
            &rl::diagnostics::probes::ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "exact fixed input",
                vec![rl::core::env::Observation::new(vec![0.; 3])],
                None,
                space,
            )
            .unwrap()
            .record(RecordContext::new("dynamics", 63))
            .unwrap(),
        )
        .unwrap();
    writer.finish().unwrap();
    for (requested, actual, sequence) in [(63, 63, 1), (40, 17, 0)] {
        let mut h = support::Harness::new(&directory, egui::vec2(1440., 900.));
        h.app.select_revision(requested).unwrap();
        assert_eq!(
            h.app.selection().input,
            rl_viewer::data::SelectedInput::ProbeRow {
                set: "fixed".into(),
                row: 0,
                sequence: 2,
            },
            "Opening the recording chooses its genuine fixed probe without capture helpers"
        );
        h.frame(vec![egui::Event::Key {
            key: egui::Key::Num5,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::CTRL,
        }]);
        settle(&mut h);
        let frame = h.app.dynamics_frame().unwrap();
        assert_eq!(
            h.app.selection().revision,
            requested,
            "Opening Dynamics must preserve the shared learner clock"
        );
        assert_eq!(
            h.app.selection().input,
            rl_viewer::data::SelectedInput::ProbeRow {
                set: "fixed".into(),
                row: 0,
                sequence: 2,
            },
            "Inspecting an episode must preserve the fixed input until a decision is selected"
        );
        assert_eq!(frame.state.revision, actual);
        assert_eq!(frame.selected.key.sequence, sequence);
        assert_eq!(
            frame
                .selected
                .episode
                .context()
                .identity
                .learner_update_revision,
            actual
        );
        assert_eq!(
            frame.state.population_episodes, 1,
            "Distinct continuation protocols must remain separate populations"
        );
        h.click("Spatial settings");
        assert!(h.has(&format!("Evaluation k={actual}")));
        h.click("Spatial settings");
    }
}

#[test]
fn protocol_switch_retires_old_reference_before_new_preparation() {
    let d = protocol_recording();
    let mut h = open(&d, egui::vec2(1440., 900.));
    h.app.select_revision(17).unwrap();
    settle(&mut h);
    let old = h.app.dynamics_frame().unwrap().state.reference.clone();
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 17);
    h.click("Spatial settings");
    h.click("first policy");
    assert!(h.has("alternate policy · 2"));
    h.click_last("alternate policy · 2");
    settle(&mut h);
    let f = h.app.dynamics_frame().unwrap();
    assert_eq!(f.state.revision, 63);
    assert!(!Arc::ptr_eq(&old, &f.state.reference));
    h.click("Spatial settings");
    h.click("alternate policy");
    h.click_last("first policy · 1");
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().state.revision, 17);
}

#[test]
fn one_decision_unbounded_action_has_visible_state_action_point_without_edges() {
    let d = one_decision_recording();
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    h.click("3D");
    h.click("Spatial settings");
    h.click("State–action");
    h.click("Spatial settings");
    settle(&mut h);
    let f = h.app.dynamics_frame().unwrap();
    assert_eq!(f.projection, Projection::StateAction);
    assert!(f.geometry3.is_empty());
    assert_eq!(f.background.paths.len(), 1);
    assert!(f.background.paths[0].segments.is_empty());
    assert_eq!(f.background.paths[0].start, f.background.paths[0].end);
    assert_eq!(f.background.eligible_segments, 0);
    assert_eq!(
        f.selected.episode.action_value(0, f.action).unwrap(),
        Some(ActionValue::Continuous(10.))
    );
    assert!(
        h.has("t = 0"),
        "one-decision action point is outside visible 3D bounds"
    );
    h.click("Final state");
    assert!(h.has("Terminal state · no action coordinate"));
}

#[test]
fn exact_unsampled_spike_cursor_is_scaled_into_timeline() {
    const SPIKE: usize = 2499;
    let d = long_spike_recording();
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    let frame = h.app.dynamics_frame().unwrap();
    assert!(
        !frame
            .selected
            .reward_trace
            .iter()
            .any(|p| p[0] as usize == SPIKE)
    );
    assert!(!frame.action_trace.iter().any(|p| p.t == SPIKE));
    h.click("Spatial settings");
    assert!(h.has("reward 4096 samples · action 4096 samples · 5000 transitions"));
    h.click("Spatial settings");
    let outer = h.app.dynamics_plot_bounds()[1];
    let plot = egui::Rect::from_min_max(
        outer.min + egui::vec2(52., 18.),
        outer.max - egui::vec2(56., 37.),
    );
    let x = plot.left() + plot.width() * SPIKE as f32 / 5000.;
    h.click_at(egui::pos2(x, plot.center().y));
    assert!(matches!(
        h.app.selection().input,
        rl_viewer::data::SelectedInput::TrajectoryStep { step: SPIKE, .. }
    ));
    let golden = rl_viewer::theme::EMPIRICAL;
    assert!(h.shapes.iter().any(|shape|matches!(shape,egui::Shape::Circle(c)
        if c.fill==golden&&c.radius>=3.7&&(c.center.x-x).abs()<2.&&(c.center.y-plot.top()).abs()<2.)),
        "exact unsampled reward marker should be visible at the top of the rescaled timeline");
    let cyan = rl_viewer::theme::CURRENT;
    assert!(
        h.shapes
            .iter()
            .any(|shape| matches!(shape,egui::Shape::Circle(c)
        if c.fill==cyan&&c.radius>=3.7&&(c.center.x-x).abs()<2.&&(c.center.y-plot.top()).abs()<2.)),
        "exact unsampled action marker should be visible at the top of the rescaled timeline"
    );
}

#[test]
fn selected_path_edge_sampling_is_disclosed_in_2d_and_state_action_3d() {
    let d = edge_sampling_recording();
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().selected.path.segments.len(),
        8192
    );
    h.click("Spatial settings");
    assert!(h.has("Selected 2D edges 8192"));
    h.click("Spatial settings");
    h.click("3D");
    h.click("Spatial settings");
    h.click("State–action");
    settle(&mut h);
    assert_eq!(h.app.dynamics_frame().unwrap().geometry3.len(), 8192);
    assert!(h.has("3D edges 8192") && h.has("9000 transitions"));
    h.click("Spatial settings");
}

#[test]
fn long_descriptor_labels_keep_all_three_plots_inside_desktop_and_normal_windows() {
    let d = long_label_recording();
    for size in [egui::vec2(1920., 1080.), egui::vec2(1440., 900.)] {
        let mut h = open(&d, size);
        settle(&mut h);
        assert_eq!(h.app.dynamics_plot_bounds().len(), 3);
        for rect in h.app.dynamics_plot_bounds() {
            assert!(
                rect.min.x >= 0. && rect.max.x <= size.x && rect.max.y <= size.y,
                "{rect:?} exceeds {size:?}"
            );
        }
        assert!(h.has("Evaluation coverage") && h.has("Episode timeline"));
    }
}

#[test]
fn coverage_integer_ticks_mark_actual_revision_positions() {
    let d = four_revision_recording();
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    assert_eq!(
        h.app.dynamics_frame().unwrap().state.coverage.points.len(),
        4
    );
    let outer = h.app.dynamics_plot_bounds()[2];
    let plot = egui::Rect::from_min_max(
        outer.min + egui::vec2(48., 18.),
        outer.max - egui::vec2(14., 37.),
    );
    for revision in 1..=4 {
        let expected_x = plot.left() + plot.width() * (revision - 1) as f32 / 3.;
        assert!(
            h.text
                .iter()
                .any(|(text, rect)| text == &revision.to_string()
                    && rect.center().y > plot.bottom()
                    && (rect.center().x - expected_x).abs() < 3.),
            "revision {revision} tick must sit at its actual x coordinate"
        );
    }
}

#[test]
fn rotating_context_geometry_reuses_worker_frame_and_projection() {
    let d = recording(fixture::bounded(3), 3);
    let mut h = open(&d, egui::vec2(1440., 900.));
    settle(&mut h);
    h.click("3D");
    settle(&mut h);
    let background = h.app.dynamics_frame().unwrap().background.clone();
    let counters = h.app.dynamics_counters();
    let center = h.app.dynamics_plot_bounds()[0].center();
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    h.frame(vec![
        egui::Event::PointerMoved(center),
        button(center, true),
    ]);
    h.frame(vec![egui::Event::PointerMoved(
        center + egui::vec2(60., 20.),
    )]);
    h.frame(vec![button(center + egui::vec2(60., 20.), false)]);
    assert_eq!(counters, h.app.dynamics_counters());
    assert!(Arc::ptr_eq(
        &background,
        &h.app.dynamics_frame().unwrap().background
    ));
}
