use eframe::egui;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::ui::{Laboratory, View};
use std::{collections::BTreeMap, path::Path, time::Duration};

fn writer(path: &Path) -> Recorder {
    Recorder::create(
        path,
        RunManifest::new(
            "provider",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap()
}

fn context(revision: u64) -> RecordContext {
    let mut context = RecordContext::new("provider", revision);
    context.probe_set = Some("fixed".into());
    context
}

fn append_probes(writer: &mut Recorder, revision: u64, offset: f32) {
    writer
        .append(
            &ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "an explicit fixed population",
                vec![
                    Observation::new(vec![offset]),
                    Observation::new(vec![offset + 1.]),
                ],
                None,
                ActionSpace::Discrete { count: 2 },
            )
            .unwrap()
            .record(context(revision))
            .unwrap(),
        )
        .unwrap();
}

fn append_metrics(writer: &mut Recorder, revision: u64, provider: bool) {
    let mut values = BTreeMap::from([("training.loss".into(), 42.)]);
    let mut unavailable = BTreeMap::new();
    let mut arrays = BTreeMap::new();
    if provider {
        values.insert("reference.benchmark_gap".into(), -0.25);
        values.insert("reference.critic_calibration".into(), 2.);
        unavailable.insert(
            "reference.gradient_agreement".into(),
            "Derivative capability missing".into(),
        );
        arrays.insert(
            "reference.critic_calibration".into(),
            Array::f64(vec![2], vec![1.5, 2.5]).unwrap(),
        );
    }
    writer
        .append(&Chunk {
            context: context(revision),
            record: Record::Metrics(MetricRecord {
                values,
                unavailable,
            }),
            arrays,
        })
        .unwrap();
}

fn frame(ctx: &egui::Context, app: &mut Laboratory, events: Vec<egui::Event>) -> egui::FullOutput {
    let time = ctx.input(|input| input.time) + 0.125;
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1440., 960.),
            )),
            time: Some(time),
            events,
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    output.textures_delta.clear();
    output
}

fn paint(ctx: &egui::Context, app: &mut Laboratory) -> String {
    fn collect(shape: &egui::epaint::Shape, text: &mut String) {
        match shape {
            egui::epaint::Shape::Text(shape) => {
                text.push_str(shape.galley.text());
                text.push('\n');
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect(shape, text);
                }
            }
            _ => {}
        }
    }
    let output = frame(ctx, app, Vec::new());
    let mut text = String::new();
    for shape in output.shapes {
        collect(&shape.shape, &mut text);
    }
    text
}

fn text_position(ctx: &egui::Context, app: &mut Laboratory, label: &str) -> egui::Pos2 {
    fn find(shape: &egui::Shape, label: &str) -> Option<egui::Pos2> {
        match shape {
            egui::Shape::Text(text) if text.galley.text() == label => {
                Some(text.galley.rect.translate(text.pos.to_vec2()).center())
            }
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|shape| find(shape, label)),
            _ => None,
        }
    }
    frame(ctx, app, Vec::new())
        .shapes
        .iter()
        .find_map(|shape| find(&shape.shape, label).filter(|&pos| shape.clip_rect.contains(pos)))
        .unwrap_or_else(|| panic!("Missing visible label {label}"))
}

fn open_diagnostics(ctx: &egui::Context, app: &mut Laboratory) -> String {
    settle(ctx, app);
    let pos = text_position(ctx, app, "Recorded diagnostics");
    for pressed in [true, false] {
        frame(
            ctx,
            app,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
    }
    settle(ctx, app)
}

fn settle(ctx: &egui::Context, app: &mut Laboratory) -> String {
    let start = std::time::Instant::now();
    loop {
        let text = paint(ctx, app);
        if app.snapshot_preparation_idle() {
            return paint(ctx, app);
        }
        assert!(start.elapsed() < Duration::from_secs(5), "{text}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn provider_metrics_are_visible_only_at_their_exact_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut writer = writer(&path);
    append_probes(&mut writer, 2, 0.);
    append_metrics(&mut writer, 2, true);
    append_metrics(&mut writer, 3, false);
    writer.finish().unwrap();

    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Value);
    app.select_revision(2).unwrap();
    let text = open_diagnostics(&ctx, &mut app);
    assert!(
        text.contains("Benchmark gap") && text.contains("-0.25"),
        "{text}"
    );
    assert!(text.contains("Anchor row 1.5"), "{text}");
    assert!(!text.contains("Training loss"), "{text}");
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.interaction.tooltip_delay = 0.
    });
    let pos = text_position(&ctx, &mut app, "Gradient agreement (fixed) unavailable");
    frame(&ctx, &mut app, vec![egui::Event::PointerMoved(pos)]);
    let text = settle(&ctx, &mut app);
    assert!(text.contains("Derivative capability missing"), "{text}");

    app.select_revision(3).unwrap();
    let text = settle(&ctx, &mut app);
    assert!(!text.contains("Benchmark gap"), "{text}");
}

#[test]
fn provider_pointwise_metrics_do_not_alias_a_redefined_probe_row() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut writer = writer(&path);
    append_probes(&mut writer, 1, 0.);
    append_probes(&mut writer, 2, 10.);
    append_metrics(&mut writer, 2, true);
    writer.finish().unwrap();

    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Value);
    let text = open_diagnostics(&ctx, &mut app);
    assert!(text.contains("Anchor row 1.5"), "{text}");
    assert!(matches!(
        app.selection().input,
        rl_viewer::data::SelectedInput::ProbeRow { sequence: 1, .. }
    ));

    // The startup hook above used the latest probe. Start again with a
    // policy anchor to the earlier immutable set and seek to the later k.
    let path = directory.path().join("anchored");
    let mut writer = crate::writer(&path);
    append_probes(&mut writer, 1, 0.);
    append_policy(&mut writer, 1, "online");
    append_probes(&mut writer, 2, 10.);
    append_metrics(&mut writer, 2, true);
    writer.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::Value);
    app.select_revision(2).unwrap();
    let text = open_diagnostics(&ctx, &mut app);
    assert!(text.contains("Benchmark gap"), "{text}");
    assert!(!text.contains("Anchor row 1.5"), "{text}");
}

fn append_policy(writer: &mut Recorder, revision: u64, role: &str) {
    let mut context = context(revision);
    context.identity.model_role = role.into();
    writer
        .append(&Chunk {
            context,
            record: Record::Policy(PolicyRecord {
                actions: "a".into(),
                parameters: PolicyParameters::Deterministic,
                entropy: None,
                entropy_method: None,
                masks: None,
                stochastic: false,
            }),
            arrays: BTreeMap::from([("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap())]),
        })
        .unwrap();
}

#[test]
fn a_provider_policy_does_not_become_the_primary_learner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let mut writer = writer(&path);
    append_probes(&mut writer, 2, 0.);
    append_metrics(&mut writer, 2, false);
    append_policy(&mut writer, 4, "reference:benchmark");
    writer.finish().unwrap();

    let ctx = egui::Context::default();
    let app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    assert_eq!(app.selection().model_role, "online");
    assert_eq!(app.selection().revision, 2);
}
