use eframe::egui;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{
    data::SelectedInput,
    network::representations::RepresentationFrame,
    ui::{Laboratory, View},
};
use std::{collections::BTreeMap, time::Duration};

fn paint(app: &mut Laboratory, ctx: &egui::Context, events: Vec<egui::Event>) -> egui::FullOutput {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            events,
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    output.textures_delta.clear();
    output
}

fn ready(app: &mut Laboratory, ctx: &egui::Context) -> RepresentationFrame {
    let started = std::time::Instant::now();
    loop {
        paint(app, ctx, Vec::new());
        if app.network_ready()
            && let Some(frame) = app.network_representation_frame()
        {
            return frame.clone();
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            app.notice()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn current_points(shape: &egui::Shape, positions: &mut Vec<egui::Pos2>) {
    match shape {
        egui::Shape::Circle(circle)
            if circle.fill == rl_viewer::theme::CURRENT && circle.radius == 4. =>
        {
            positions.push(circle.center);
        }
        egui::Shape::Vec(shapes) => {
            for shape in shapes {
                current_points(shape, positions);
            }
        }
        _ => {}
    }
}

fn write_fixture(path: &std::path::Path, compact: Option<[[u64; 2]; 2]>) {
    let action_space = ActionSpace::Discrete { count: 2 };
    let manifest = RunManifest::new(
        "latent",
        "representation",
        "test",
        1,
        action_space.clone(),
        0.9,
    )
    .unwrap();
    let mut writer = Recorder::create(path, manifest, StorageConfig::default()).unwrap();
    for (slot, (revision, order, offset)) in [(2, [0, 1, 2], 0.), (8, [2, 0, 1], 2.)]
        .into_iter()
        .enumerate()
    {
        let mut context = RecordContext::new("latent", revision);
        context.probe_set = Some("fixed".into());
        let probes = ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "same fixed state inputs",
            order
                .iter()
                .map(|&row| Observation::new(vec![row as f32]))
                .collect(),
            None,
            action_space.clone(),
        )
        .unwrap()
        .with_row_ids(order.iter().map(|&row| row as u64 + 10).collect())
        .unwrap()
        .record(context.clone())
        .unwrap();
        writer.append(&probes).unwrap();
        let features = [[-1., 0.2], [0., 1.], [1., -1.]];
        let mut activation = ActivationRecord {
            inputs: ActivationInputs::Observations,
            network: "encoder".into(),
            layers: BTreeMap::from([("hidden".into(), "hidden".into())]),
            ..Default::default()
        };
        let mut arrays = BTreeMap::from([(
            "hidden".into(),
            Array::f32(
                vec![3, 2],
                order
                    .into_iter()
                    .flat_map(|row| [features[row][0] + offset, features[row][1]])
                    .collect(),
            )
            .unwrap(),
        )]);
        if let Some(compact) = compact {
            activation
                .selected_neurons
                .insert("hidden".into(), "neuron_ids".into());
            activation.source_widths.insert("hidden".into(), 4);
            arrays.insert(
                "neuron_ids".into(),
                Array::u64(vec![2], compact[slot].to_vec()).unwrap(),
            );
        }
        writer
            .append(&Chunk {
                context,
                record: Record::Activations(activation),
                arrays,
            })
            .unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn recorded_representation_reuses_earlier_axes_and_aligns_reordered_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    write_fixture(&path, None);
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Network);
    let frame = ready(&mut app, &ctx);
    assert_eq!(frame.revision, 8);
    assert_eq!(frame.basis_revision, Some(2));
    assert_eq!(frame.row_ids, vec![12, 10, 11]);
    assert!((frame.cka.unwrap() - 1.).abs() < 1e-9);
    let reference = frame.reference_points.as_ref().unwrap();
    let shift = [
        frame.points[0][0] - reference[0][0],
        frame.points[0][1] - reference[0][1],
    ];
    assert!((shift[0].hypot(shift[1]) - 2.).abs() < 1e-6);
    for (point, anchor) in frame.points.iter().zip(reference) {
        assert!((point[0] - anchor[0] - shift[0]).abs() < 1e-6);
        assert!((point[1] - anchor[1] - shift[1]).abs() < 1e-6);
    }
    let mut markers = Vec::new();
    for shape in paint(&mut app, &ctx, Vec::new()).shapes {
        current_points(&shape.shape, &mut markers);
    }
    assert_eq!(markers.len(), 3);
    let position = markers[1];
    for pressed in [true, false] {
        paint(
            &mut app,
            &ctx,
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
    assert_eq!(
        app.selection().input,
        SelectedInput::ProbeRow {
            sequence: 2,
            set: "fixed".into(),
            row: 1
        }
    );
    let input = app.selection().input.clone();
    app.select_revision(2).unwrap();
    let earlier = ready(&mut app, &ctx);
    assert_eq!(earlier.revision, 2);
    assert_eq!(earlier.row_for_input(&input), Some(0));
    assert_eq!(app.selection().input, input);
}

#[test]
fn different_recorded_feature_ids_have_only_checkpoint_local_axes() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    write_fixture(&path, Some([[0, 1], [2, 3]]));
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Network);
    let frame = ready(&mut app, &ctx);
    assert_eq!(frame.feature_ids, vec![2, 3]);
    assert_eq!(frame.source_width, 4);
    assert_eq!(frame.points.len(), 3);
    assert_eq!(frame.basis_revision, Some(8));
    assert!(frame.reference_points.is_none());
    assert!(frame.cka.is_none());
    assert!(
        frame
            .comparison_note
            .as_deref()
            .is_some_and(|note| note.contains("feature identities"))
    );
}

#[cfg(feature = "screenshots")]
#[test]
#[ignore = "requires the genuine diagnostic_session reppo recording and capture environment"]
fn genuine_reppo_restores_encoder_on_fixed_reference_actions() {
    let recording = std::env::var("RL_NETWORK_REPPO_RUN").unwrap();
    assert_eq!(
        std::env::var("RL_VIEWER_NETWORK_SOURCE").as_deref(),
        Ok("historical")
    );
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, recording).unwrap();
    app.select_view(View::Network);
    let frame = ready(&mut app, &ctx);
    assert!(frame.historical);
    assert!(frame.historical_fixed_reference);
    assert_eq!(frame.layer, "encoder.features");
    assert_eq!(frame.population, "cartpole-resets");
    assert!(frame.revision > 0);
    assert_eq!(frame.revision, frame.requested_revision);
    assert_eq!(frame.basis_revision, Some(0));
    assert_eq!(frame.actions.len(), frame.row_ids.len());
    assert_eq!(
        frame.reference_points.as_ref().unwrap().len(),
        frame.row_ids.len()
    );
    assert!(frame.cka.is_some_and(|value| (0. ..=1.).contains(&value)));
    println!(
        "REPPO k{} · {} · {} inputs · CKA {:.6}",
        frame.revision,
        frame.layer,
        frame.row_ids.len(),
        frame.cka.unwrap()
    );
}

#[test]
fn small_representation_population_shows_pca_and_activation_heatmap_together() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    write_fixture(&path, None);
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Network);
    ready(&mut app, &ctx);
    let output = paint(&mut app, &ctx, Vec::new());
    fn contains(shape: &egui::Shape, value: &str) -> bool {
        match shape {
            egui::Shape::Text(text) => text.galley.text().contains(value),
            egui::Shape::Vec(shapes) => shapes.iter().any(|shape| contains(shape, value)),
            _ => false,
        }
    }
    assert!(
        output
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "PC 1"))
    );
    assert!(
        output
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "Feature ID"))
    );
}
