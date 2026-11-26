//! Publication timing regressions use the real worker and shared transport.
use super::*;
use crate::network_test_fixture as fixtures;

fn paint(app: &mut Laboratory, ctx: &egui::Context) -> egui::FullOutput {
    paint_at(app, ctx, egui::vec2(1920., 1080.))
}
fn paint_at(app: &mut Laboratory, ctx: &egui::Context, size: egui::Vec2) -> egui::FullOutput {
    paint_events(app, ctx, size, Vec::new())
}
fn paint_events(
    app: &mut Laboratory,
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            events,
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    output.textures_delta.clear();
    output
}

fn representation_and_snapshot_fixture(dense: bool) -> tempfile::TempDir {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{diagnostics::models::SnapshotBuilder, nn::network::Mlp};
    let (dir, mut writer) = fixtures::recorder();
    let mut context = RecordContext::new("network", 0);
    context.probe_set = Some("fixed".into());
    writer
        .append(&Chunk {
            context,
            record: Record::Activations(ActivationRecord {
                inputs: ActivationInputs::Observations,
                network: "actor".into(),
                layers: std::collections::BTreeMap::from([("hidden".into(), "hidden".into())]),
                ..Default::default()
            }),
            arrays: std::collections::BTreeMap::from([(
                "hidden".into(),
                Array::f32(vec![12, 2], (0..24).map(|v| v as f32 * 0.1).collect()).unwrap(),
            )]),
        })
        .unwrap();
    let snapshot = if dense {
        fixtures::snapshot(0, "online", &["actor"])
    } else {
        let space = rl::core::action::ActionSpace::Unbounded { dimensions: 1 };
        let mut builder =
            SnapshotBuilder::new("opaque", RecordContext::new("network", 0), 3, space).unwrap();
        let model = Mlp::new(3, &[4], 1, &mut ChaCha8Rng::seed_from_u64(4)).unwrap();
        builder
            .opaque("opaque", &model, "no geometry adapter")
            .unwrap();
        builder.finish().unwrap()
    };
    writer.append(&snapshot).unwrap();
    writer.finish().unwrap();
    dir
}

#[test]
fn geometry_is_preferred_when_dense_snapshots_and_representations_coexist() {
    let dir = representation_and_snapshot_fixture(true);
    let ctx = egui::Context::default();
    let mut app =
        Laboratory::from_context(&ctx, dir.path().join("run").display().to_string()).unwrap();
    app.select_view(View::Network);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() {
        paint(&mut app, &ctx);
        assert!(Instant::now() < deadline, "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        app.network_frame().is_some(),
        "a compatible dense snapshot should open in Geometry"
    );
    assert!(app.network_representation_frame().is_none());
}

#[test]
fn representations_remain_the_default_when_snapshots_are_not_inspectable() {
    let dir = representation_and_snapshot_fixture(false);
    let ctx = egui::Context::default();
    let mut app =
        Laboratory::from_context(&ctx, dir.path().join("run").display().to_string()).unwrap();
    app.select_view(View::Network);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() {
        paint(&mut app, &ctx);
        assert!(Instant::now() < deadline, "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(app.network_representation_frame().is_some());
    assert!(app.network_frame().is_none());
}

#[test]
fn explicit_network_mode_selection_is_preserved_after_default_discovery() {
    fn text_position(shape: &egui::Shape, label: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(text) if text.galley.text() == label => {
                Some(text.pos + text.galley.rect.center().to_vec2())
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| text_position(s, label)),
            _ => None,
        }
    }
    fn click(app: &mut Laboratory, ctx: &egui::Context, label: &str) {
        let output = paint(app, ctx);
        let position = output
            .shapes
            .iter()
            .find_map(|s| text_position(&s.shape, label))
            .unwrap();
        for pressed in [true, false] {
            paint_events(
                app,
                ctx,
                egui::vec2(1920., 1080.),
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
    }
    let dir = representation_and_snapshot_fixture(true);
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let input = app.selection.input.clone();
    click(&mut app, &ctx, "Representations");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() || app.network_representation_frame().is_none() {
        paint(&mut app, &ctx);
        assert!(Instant::now() < deadline, "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(2));
    }
    for _ in 0..3 {
        paint(&mut app, &ctx);
    }
    assert!(app.network_representation_frame().is_some());
    assert_eq!(app.selection.input, input);
    click(&mut app, &ctx, "Geometry");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() || app.network_frame().is_none() {
        paint(&mut app, &ctx);
        assert!(Instant::now() < deadline, "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(app.selection.input, input);
}

#[test]
fn slice_axis_dropdown_remains_selectable_and_preserves_the_full_recorded_input() {
    fn text_center(shape: &egui::Shape, label: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(text) if text.galley.text() == label => {
                Some(text.pos + text.galley.rect.center().to_vec2())
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| text_center(shape, label)),
            _ => None,
        }
    }
    fn click_text(app: &mut Laboratory, ctx: &egui::Context, label: &str) -> egui::FullOutput {
        let output = paint(app, ctx);
        // Popup text is painted last; choose it over underlying plot axis labels.
        let position = output
            .shapes
            .iter()
            .rev()
            .find_map(|shape| text_center(&shape.shape, label))
            .unwrap_or_else(|| panic!("click target {label:?} is not painted"));
        let mut clicked = None;
        for pressed in [true, false] {
            clicked = Some(paint_events(
                app,
                ctx,
                egui::vec2(1920., 1080.),
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            ));
        }
        clicked.unwrap()
    }

    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let original_input = app.selection.input.clone();
    let original_snapshot = app.network_frame().unwrap().snapshot.clone();
    let original_output = app.network_frame().unwrap().geometry.output_values.clone();
    assert_eq!(app.network_frame().unwrap().geometry.slice_anchor, [0., 0.]);
    click_text(&mut app, &ctx, "Slice");
    click_text(&mut app, &ctx, "obs[0]");
    let dropdown = paint(&mut app, &ctx);
    assert!(
        dropdown
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "Slice direction")),
        "opening an axis dropdown must keep its Slice context mounted"
    );
    let selected = click_text(&mut app, &ctx, "obs[2]");
    // Until the header publishes the changed request, only placeholders may
    // appear under the new axes, never the previous coordinate geometry.
    if app.network_frame().unwrap().geometry.slice_anchor == [0., 0.] {
        assert!(
            !selected
                .shapes
                .iter()
                .any(|shape| contains(&shape.shape, "E=")),
            "an incompatible old frame must not be painted under the requested axes"
        );
    }
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        paint(&mut app, &ctx);
        if app.network_ready() && app.network_frame().unwrap().geometry.slice_anchor == [1., 0.] {
            break;
        }
        assert!(
            Instant::now() < end,
            "the requested X=obs[2], Y=obs[1] frame did not publish"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let frame = app.network_frame().unwrap();
    // Fixture row zero is [0, 0, 1]: the distinct X/Y axes changed, while
    // inference continues to receive the same complete recorded observation.
    assert_eq!(frame.geometry.slice_anchor, [1., 0.]);
    assert_eq!(frame.geometry.output_values, original_output);
    assert_eq!(frame.snapshot.key, original_snapshot.key);
    assert_eq!(frame.snapshot.revision, original_snapshot.revision);
    assert_eq!(app.selection.input, original_input);
}

#[test]
fn narrow_finite_panes_keep_every_layer_error_readable_without_horizontal_scroll() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let expected = app.network_frame().unwrap().geometry.comparisons.len();
    // 1280 physical pixels at 150% scale leaves about 853 logical pixels.
    let output = paint_at(&mut app, &ctx, egui::vec2(853., 900.));
    let mut errors = 0;
    for clipped in &output.shapes {
        if let egui::Shape::Text(text) = &clipped.shape
            && text.galley.text().starts_with("E=")
        {
            errors += 1;
            let rect = text.galley.rect.translate(text.pos.to_vec2());
            assert!(
                clipped.clip_rect.contains_rect(rect),
                "layer error clipped: {rect:?} outside {:?}",
                clipped.clip_rect
            );
        }
    }
    assert_eq!(
        errors, expected,
        "every layer retains its exact error readout"
    );
}

#[test]
fn compact_desktop_keeps_all_geometry_path_summaries_visible() {
    let dir = representation_and_snapshot_fixture(true);
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let expected = app.network_frame().unwrap().geometry.paths.len();
    let mut output = paint_at(&mut app, &ctx, egui::vec2(1024., 640.));
    for _ in 0..3 {
        output = paint_at(&mut app, &ctx, egui::vec2(1024., 640.));
    }
    let mut summaries = 0;
    let mut x_labels = 0;
    for clipped in &output.shapes {
        if let egui::Shape::Text(text) = &clipped.shape
            && (text.galley.text().starts_with("variance ") || text.galley.text() == "obs[0]")
        {
            let rect = text.galley.rect.translate(text.pos.to_vec2());
            assert!(
                rect.bottom() <= 640. && clipped.clip_rect.contains_rect(rect),
                "path summary clipped: {rect:?} outside {:?}",
                clipped.clip_rect
            );
            summaries += usize::from(text.galley.text().starts_with("variance "));
            x_labels += usize::from(text.galley.text() == "obs[0]");
        }
    }
    assert_eq!(
        summaries, expected,
        "every representation path must remain visible"
    );
    assert!(x_labels >= 2, "both geometry plots need their X axis label");
}

#[test]
fn ordinary_desktop_keeps_geometry_axis_labels_and_path_summaries_visible() {
    let dir = representation_and_snapshot_fixture(true);
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let expected = app.network_frame().unwrap().geometry.paths.len();
    let mut output = paint_at(&mut app, &ctx, egui::vec2(1440., 960.));
    for _ in 0..3 {
        output = paint_at(&mut app, &ctx, egui::vec2(1440., 960.));
    }
    let mut summaries = 0;
    let mut x_labels = 0;
    for clipped in &output.shapes {
        if let egui::Shape::Text(text) = &clipped.shape {
            let label = text.galley.text();
            if label.starts_with("variance ") || label == "obs[0]" {
                let rect = text.galley.rect.translate(text.pos.to_vec2());
                assert!(
                    rect.bottom() <= 960. && clipped.clip_rect.contains_rect(rect),
                    "{label} clipped: {rect:?} outside {:?}",
                    clipped.clip_rect
                );
                summaries += usize::from(label.starts_with("variance "));
                x_labels += usize::from(label == "obs[0]");
            }
        }
    }
    assert_eq!(summaries, expected);
    assert!(x_labels >= 2, "both geometry plots need their X axis label");
}

#[test]
fn changed_finite_input_reserves_panels_without_relabelling_old_geometry() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    app.network_pane.t = 0.25;
    // The request is submitted in the workspace, after the only publication
    // point in the header. This paint must retain layout, not old geometry.
    let output = paint(&mut app, &ctx);
    for title in [
        "Local geometry",
        "Propagation by layer",
        "Local nonlinearity",
        "Representation paths",
    ] {
        assert!(
            output
                .shapes
                .iter()
                .any(|shape| contains(&shape.shape, title)),
            "{title} must reserve its place during preparation"
        );
    }
    assert!(
        !output
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "E=")),
        "incompatible finite geometry must not be painted"
    );
    assert_eq!(app.network_frame().unwrap().geometry.t, 1.);
}
fn ready_app(ctx: &egui::Context, dir: &std::path::Path) -> Laboratory {
    let mut app = Laboratory::from_context(ctx, dir.join("run").display().to_string()).unwrap();
    app.select_view(View::Network);
    app.select_revision(0).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() || app.snapshot_catalog.as_ref().is_none_or(|c| c.pending()) {
        paint(&mut app, ctx);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    app
}
fn contains(shape: &egui::Shape, needle: &str) -> bool {
    match shape {
        egui::Shape::Text(t) => t.galley.text().contains(needle),
        egui::Shape::Vec(shapes) => shapes.iter().any(|s| contains(s, needle)),
        _ => false,
    }
}
#[test]
fn checkpoint_preparation_keeps_all_three_plots_visible() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    app.select_revision(100).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let output = paint(&mut app, &ctx);
        for title in [
            "Local geometry",
            "Propagation by layer",
            "Local nonlinearity",
            "log s",
            "Representation paths",
        ] {
            assert!(
                output.shapes.iter().any(|s| contains(&s.shape, title)),
                "{title} disappeared while preparing a checkpoint"
            );
        }
        if app.network_ready() {
            assert_eq!(app.network_frame().unwrap().snapshot.revision, 100);
            break;
        }
        assert_eq!(app.network_frame().unwrap().snapshot.revision, 0);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn finite_controls_keep_the_fixed_map_and_idle_frames_do_not_resubmit() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let original_map = app.network_frame().unwrap().geometry.neighborhood.clone();
    let original_input = app.selection.input.clone();
    app.network_pane.t = 0.25;
    app.network_pane.layer = 3; // The real fixture's output, beyond hidden 2.
    paint(&mut app, &ctx);
    let end = Instant::now() + Duration::from_secs(10);
    while !app.network_ready() {
        paint(&mut app, &ctx);
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(2));
    }
    let frame = app.network_frame().unwrap();
    assert_eq!(frame.geometry.t, 0.25);
    assert!(Arc::ptr_eq(&original_map, &frame.geometry.neighborhood));
    assert_eq!(app.selection.input, original_input);
    let published = frame as *const _;
    let signature = app.network_pane.worker.submitted().unwrap().to_owned();
    for _ in 0..8 {
        paint(&mut app, &ctx);
    }
    assert!(std::ptr::eq(app.network_frame().unwrap(), published));
    assert_eq!(
        app.network_pane.worker.submitted(),
        Some(signature.as_str())
    );
}
#[test]
fn fast_network_playback_paints_each_complete_checkpoint_before_advancing() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    let run = app.run.take().unwrap();
    app.transport
        .start_snapshot_playback(&run, &mut app.selection, &[0, 100, 250], 100.)
        .unwrap();
    app.run = Some(run);
    let mut seen = std::collections::BTreeSet::from([0]);
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        std::thread::sleep(Duration::from_millis(20));
        let output = paint(&mut app, &ctx);
        if output
            .shapes
            .iter()
            .any(|s| contains(&s.shape, "Local geometry"))
        {
            seen.insert(app.network_frame().unwrap().snapshot.revision);
        }
        if app.network_ready() && !app.transport.is_playing() {
            break;
        }
        assert!(Instant::now() < end);
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), [0, 100, 250]);
}
#[test]
fn completions_after_the_header_wait_until_the_next_header_publication() {
    let dir = fixtures::fixture();
    let ctx = egui::Context::default();
    let mut app = ready_app(&ctx, dir.path());
    app.select_revision(100).unwrap();
    let run = app.run.take().unwrap();
    let mut observed = None;
    let mut output = ctx.run_ui(Default::default(), |ui| {
        app.network_snapshot_transport(ui, &run).unwrap();
        assert_eq!(app.network_frame().unwrap().snapshot.revision, 0);
        app.network_workspace(ui, &run).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        // Even if a worker finishes between header and workspace/layout paints,
        // the workspace is not a second publication point.
        app.network_workspace(ui, &run).unwrap();
        observed = Some(app.network_frame().unwrap().snapshot.revision);
    });
    output.textures_delta.clear();
    assert_eq!(observed, Some(0));
    app.run = Some(run);
    paint(&mut app, &ctx);
    assert_eq!(app.network_frame().unwrap().snapshot.revision, 100);
}
