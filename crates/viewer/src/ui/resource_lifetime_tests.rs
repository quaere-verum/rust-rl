use super::*;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        probes::ProbeSet,
        storage::{Recorder, StorageConfig},
    },
};

fn fixture() -> (tempfile::TempDir, egui::Context, Laboratory) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let space = ActionSpace::Discrete { count: 2 };
    let mut manifest =
        RunManifest::new("resources", "fixture", "PPO", 1, space.clone(), 0.9).unwrap();
    manifest.environment = "lqg".into();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("resources", 1);
    context.probe_set = Some("fixed-grid".into());
    writer
        .append(
            &ProbeSet::new(
                "fixed-grid",
                ProbePopulation::FixedReference,
                "stable input",
                vec![Observation::new(vec![1.])],
                None,
                space,
            )
            .unwrap()
            .record(context.clone())
            .unwrap(),
        )
        .unwrap();
    writer.finish().unwrap();
    let context = egui::Context::default();
    let mut app = Laboratory::from_context(&context, path.display().to_string()).unwrap();
    app.live = false;
    let run = app.run.as_ref().unwrap();
    for index in 0..run.recording().entries().len() {
        assert!(app.loader.cache.insert(
            run.key(index).unwrap(),
            Arc::new(run.recording().load(index).unwrap())
        ));
    }
    (directory, context, app)
}

#[test]
fn clear_recording_releases_numerical_data_and_preserves_selected_source() {
    let (_directory, _context, mut app) = fixture();
    let selected = app.selection.clone();
    assert!(app.loader.cache.resident_bytes() > 0);
    app.clear_recording();
    assert!(app.run.is_none());
    assert_eq!(app.loader.cache.resident_bytes(), 0);
    assert_eq!(app.resolver.resident_bytes(), 0);
    assert_eq!(app.selection.input, selected.input);
    assert_eq!(app.selection.revision, selected.revision);
    assert_eq!(app.selection.model_role, selected.model_role);
    assert!(app.workspace_defaults.is_none());
    assert_eq!(app.comparison.resident_bytes(), 0);
}

#[test]
fn switching_between_supported_panels_retires_inference_but_reselecting_does_not() {
    let (_directory, _context, mut app) = fixture();
    for source in [
        View::PolicyMap,
        View::Value,
        View::Network,
        View::Dynamics,
        View::Environment,
    ] {
        for target in [
            View::PolicyMap,
            View::Value,
            View::Network,
            View::Dynamics,
            View::Environment,
        ] {
            app.select_view(source);
            let before = app.interactive.retirement_status().0;
            app.select_view(target);
            assert_eq!(
                app.interactive.retirement_status().0 > before,
                source != target
            );
        }
    }
}

#[test]
fn successful_run_open_retires_even_without_a_panel_change() {
    let (_directory, _context, mut app) = fixture();
    for view in [
        View::PolicyMap,
        View::Value,
        View::Network,
        View::Dynamics,
        View::Environment,
    ] {
        app.select_view(view);
        let before = app.interactive.retirement_status().0;
        app.open_run(0);
        assert_eq!(app.view, view);
        assert!(app.interactive.retirement_status().0 > before);
    }
}

#[test]
fn header_navigation_retires_the_previous_panel_session() {
    let (_directory, context, mut app) = fixture();
    let run = app.run.take().unwrap();
    for (label, target) in [
        ("Value", View::Value),
        ("Network", View::Network),
        ("Dynamics", View::Dynamics),
    ] {
        app.select_view(View::PolicyMap);
        let before = app.interactive.retirement_status().0;
        let mut output = context.run_ui(egui::RawInput::default(), |ui| {
            app.toolbar(ui, &run).unwrap();
        });
        let position = output.shapes.iter().find_map(|shape| match &shape.shape {
            egui::Shape::Text(text) if text.galley.text() == label => {
                Some(text.pos + text.galley.rect.center().to_vec2())
            }
            _ => None,
        });
        output.textures_delta.clear();
        let position = position.expect("current panel button");
        for pressed in [true, false] {
            let mut output = context.run_ui(
                egui::RawInput {
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
                |ui| {
                    app.toolbar(ui, &run).unwrap();
                },
            );
            output.textures_delta.clear();
        }
        assert_eq!(app.view, target);
        assert!(app.interactive.retirement_status().0 > before);
    }
}
