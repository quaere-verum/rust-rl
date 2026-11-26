use eframe::egui;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{
    resolver::{ProbeResolver, ResolveStatus},
    ui::{Laboratory, View},
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};

#[test]
fn every_recorded_panel_renders_headlessly_and_reads_its_selected_probe() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let space = ActionSpace::Discrete { count: 2 };
    let mut writer = Recorder::create(
        &path,
        RunManifest::new("ui", "test", "test", 2, space.clone(), 0.9).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let set = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "fixture",
        vec![
            Observation::new(vec![1., 0.]),
            Observation::new(vec![2., 0.]),
        ],
        None,
        space,
    )
    .unwrap();
    let mut c = RecordContext::new("ui", 2);
    c.probe_set = Some("fixed".into());
    writer.append(&set.record(c.clone()).unwrap()).unwrap();
    writer
        .append(&Chunk {
            context: c.clone(),
            record: Record::Policy(PolicyRecord {
                actions: "a".into(),
                parameters: PolicyParameters::Categorical {
                    probabilities: "p".into(),
                },
                entropy: None,
                entropy_method: None,
                masks: None,
                stochastic: false,
            }),
            arrays: BTreeMap::from([
                ("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
                (
                    "p".into(),
                    Array::f32(vec![2, 2], vec![0.8, 0.2, 0.3, 0.7]).unwrap(),
                ),
            ]),
        })
        .unwrap();
    writer
        .append(&Chunk {
            context: c.clone(),
            record: Record::Activations(ActivationRecord {
                network: "policy".into(),
                layers: BTreeMap::from([("layers.0.post".into(), "a".into())]),
                summaries: BTreeMap::new(),
                ..Default::default()
            }),
            arrays: BTreeMap::from([(
                "a".into(),
                Array::f32(vec![2, 3], vec![0., 1., 2., 3., 2., 1.]).unwrap(),
            )]),
        })
        .unwrap();
    writer
        .append(&Chunk {
            context: RecordContext::new("ui", 37),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::from([("loss".into(), 0.3)]),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    writer
        .append(&Chunk {
            context: RecordContext::new("ui", 37),
            record: Record::Trajectory(TrajectoryRecord {
                render: None,
                id: "unrelated-episode".into(),
                environment: "custom".into(),
                observations: "obs".into(),
                actions: "a".into(),
                rewards: "r".into(),
                terminated: "done".into(),
                truncated: "cut".into(),
                simulation_times: None,
                action_masks: None,
                frames: BTreeMap::new(),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "other-policy".into(),
                    stochastic_policy: false,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "explicit".into(),
                    truncation_handling: "terminal".into(),
                    complete_returns: true,
                },
            }),
            arrays: BTreeMap::from([
                ("obs".into(), Array::f32(vec![2, 2], vec![99.; 4]).unwrap()),
                ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                ("r".into(), Array::f32(vec![1], vec![1.]).unwrap()),
                ("done".into(), Array::u8(vec![1], vec![1]).unwrap()),
                ("cut".into(), Array::u8(vec![1], vec![0]).unwrap()),
            ]),
        })
        .unwrap();
    writer.finish().unwrap();
    // Exercise recorded-position transport through its current keyboard control.
    let run = rl_viewer::data::RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut transport = rl_viewer::timeline_ui::TrainingTransport::default();
    let mut selection = rl_viewer::data::Selection::default();
    selection.select_revision(&run, 37).unwrap();
    selection.select_probe(&run, "fixed", 1);
    let input = selection.input.clone();
    let transport_context = egui::Context::default();
    let mut output = transport_context.run_ui(
        egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Space,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ui| {
            transport
                .show_snapshots(
                    ui,
                    &run,
                    &mut selection,
                    rl_viewer::timeline_ui::SnapshotPresentation {
                        positions: &[2],
                        published: Some(2),
                        ready: true,
                        checking: false,
                        error: None,
                    },
                )
                .unwrap();
        },
    );
    output.textures_delta.clear();
    assert_eq!(selection.revision, 2);
    assert_eq!(selection.input, input);
    assert!(
        !transport.is_playing(),
        "a single recorded position stays paused"
    );
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::PolicyMap);
    app.select_revision(19).unwrap(); // A revision without any record is a valid request.
    assert_eq!(app.selection().revision, 19);
    let requested_input = app.selection().input.clone();
    // Generic resolution accepts intermediate requested time and preserves the
    // actual sparse measurement/source identities. Native recorded navigation
    // then admits only the actual policy checkpoints (this run has just k=2).
    let resolved = ProbeResolver::new(1024 * 1024)
        .resolve_policy(&run, app.selection(), |index| {
            Ok(Some(Arc::new(run.recording().load(index)?)))
        })
        .unwrap();
    assert_eq!(resolved.status, ResolveStatus::Ready);
    assert_eq!(resolved.requested_revision, 19);
    let actual = resolved.record.unwrap();
    assert_eq!(actual.revision(), 2);
    assert_eq!(actual.key.sequence, 1);
    assert_eq!(actual.probe_key.sequence, 0);
    assert_eq!(actual.row, 0);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1440., 960.),
                )),
                ..Default::default()
            },
            |ui| app.show(ui),
        );
        output.textures_delta.clear();
        assert!(app.notice().is_none(), "{:?}", app.notice());
        if app.policy_pane().ready() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            app.policy_pane().status()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let frame = app.policy_frame().unwrap();
    assert_eq!(app.selection().revision, 2);
    assert_eq!(app.selection().input, requested_input);
    assert_eq!(frame.requested_revision, 2);
    assert_eq!(frame.grid.current.revision, 2);
    assert_eq!(frame.grid.current.key.sequence, 1);
    assert_eq!(frame.grid.current.probe_key.as_ref().unwrap().sequence, 0);
    assert_eq!(frame.selected.as_ref().unwrap().chosen_action, Some(0));
    let selected_input = app.selection().input.clone();
    assert!(matches!(
        selected_input,
        rl_viewer::data::SelectedInput::ProbeRow {
            row: 0,
            sequence: 0,
            ..
        }
    ));
    for size in [egui::vec2(1440., 960.), egui::vec2(760., 960.)] {
        for view in [View::PolicyMap, View::Value, View::Network] {
            app.select_view(view);
            let mut text = String::new();
            for _ in 0..12 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
                        ..Default::default()
                    },
                    |ui| app.show(ui),
                );
                assert!(!output.shapes.is_empty());
                text.clear();
                for shape in &output.shapes {
                    collect_text(&shape.shape, &mut text);
                }
                output.textures_delta.clear();
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(app.notice().is_none(), "{view:?}: {:?}", app.notice());
            for destination in ["Policy", "Value", "Network", "Dynamics"] {
                assert!(
                    text.lines().any(|line| line == destination),
                    "missing {destination}: {text}"
                );
            }
            for retired in [
                "Explore",
                "Advanced",
                "Representation",
                "Outcomes",
                "Replay",
                "unrelated-episode",
                "99.0000",
            ] {
                assert!(
                    !text.lines().any(|line| line == retired),
                    "{view:?}: {text}"
                );
            }
            assert_eq!(app.selection().revision, 2);
            assert_eq!(app.selection().input, selected_input);
            if let Some(frame) = app.policy_frame() {
                assert_eq!(frame.requested_revision, 2);
                assert_eq!(
                    frame.grid.current.revision, 2,
                    "never display future policy"
                );
            }
        }
    }
}

fn collect_text(shape: &egui::epaint::Shape, out: &mut String) {
    match shape {
        egui::epaint::Shape::Text(text) => {
            out.push_str(text.galley.text());
            out.push('\n');
        }
        egui::epaint::Shape::Vec(shapes) => {
            for shape in shapes {
                collect_text(shape, out);
            }
        }
        _ => {}
    }
}
