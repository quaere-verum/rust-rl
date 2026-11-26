use super::*;
use crate::dynamics::work::availability_fixture as episode_fixture;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, storage::*},
};

fn fixture() -> (tempfile::TempDir, egui::Context, Laboratory) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    let space = ActionSpace::Discrete { count: 2 };
    let manifest = RunManifest::new("shell", "shell", "ordinary", 1, space.clone(), 0.9).unwrap();
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for revision in [0, 7] {
        let mut context = RecordContext::new("shell", revision);
        context.probe_set = Some("fixed".into());
        recorder
            .append(
                &ProbeSet::new(
                    "fixed",
                    ProbePopulation::FixedReference,
                    "exact inputs",
                    vec![Observation::new(vec![0.])],
                    None,
                    space.clone(),
                )
                .unwrap()
                .record(context.clone())
                .unwrap(),
            )
            .unwrap();
        recorder
            .append(&Chunk {
                context,
                record: Record::Policy(PolicyRecord {
                    actions: "a".into(),
                    parameters: PolicyParameters::Categorical {
                        probabilities: "p".into(),
                    },
                    entropy: None,
                    entropy_method: None,
                    masks: None,
                    stochastic: true,
                }),
                arrays: std::collections::BTreeMap::from([
                    ("a".into(), Array::u64(vec![1], vec![0]).unwrap()),
                    ("p".into(), Array::f32(vec![1, 2], vec![0.8, 0.2]).unwrap()),
                ]),
            })
            .unwrap();
    }
    recorder.finish().unwrap();
    let context = egui::Context::default();
    let mut app = Laboratory::from_context(&context, path.display().to_string()).unwrap();
    app.live = false;
    app.workspace_defaults = None;
    app.select_revision(7).unwrap();
    (directory, context, app)
}

fn key(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::CTRL,
    }
}
fn frame(
    context: &egui::Context,
    app: &mut Laboratory,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut output = context.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 720.),
            )),
            events,
            ..Default::default()
        },
        |ui| app.show(ui),
    );
    output.textures_delta.clear();
    output
}

#[test]
fn selecting_catalogue_run_cancels_an_older_pending_directory_open() {
    let (_first, context, mut app) = fixture();
    let (second, _, _) = fixture();
    let original = app.run.as_ref().unwrap().root().to_owned();
    let (entered, wait_entered) = std::sync::mpsc::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let (finished, wait_finished) = std::sync::mpsc::channel();
    let mut opener =
        recording_opener::RecordingOpener::with_prepare(context.clone(), move |path, _| {
            entered.send(()).unwrap();
            gate.recv().unwrap();
            let result = recording_opener::prepare(path);
            finished.send(()).unwrap();
            result
        })
        .unwrap();
    opener.request(second.path().join("run"));
    opener.poll(&context);
    wait_entered.recv_timeout(Duration::from_secs(2)).unwrap();
    app.recording_opener = Some(opener);
    app.open_run(0);
    release.send(()).unwrap();
    wait_finished.recv_timeout(Duration::from_secs(5)).unwrap();
    for _ in 0..10 {
        frame(&context, &mut app, vec![]);
    }
    assert_eq!(app.run.as_ref().unwrap().root(), &original);
}

#[test]
fn workspace_shortcut_preserves_exact_input_role_and_requested_checkpoint() {
    let (_directory, context, mut app) = fixture();
    frame(&context, &mut app, vec![]);
    let selected = app.selection.clone();
    let generation = app.interactive.retirement_status().0;
    frame(&context, &mut app, vec![key(egui::Key::Num4)]);
    assert_eq!(app.view, View::Network);
    assert_eq!(app.selection.input, selected.input);
    assert_eq!(app.selection.revision, selected.revision);
    assert_eq!(app.selection.model_role, selected.model_role);
    assert!(app.interactive.retirement_status().0 > generation);
}

#[test]
fn workspace_shortcut_does_not_override_a_focused_text_editor() {
    let (_directory, context, mut app) = fixture();
    let mut text = String::from("7");
    let id = egui::Id::unique("focused-number");
    let mut output = context.run_ui(egui::RawInput::default(), |ui| {
        ui.add(egui::TextEdit::singleline(&mut text).id(id))
            .request_focus();
        app.show(ui);
    });
    output.textures_delta.clear();
    let mut output = context.run_ui(
        egui::RawInput {
            events: vec![key(egui::Key::Num2)],
            ..Default::default()
        },
        |ui| {
            ui.add(egui::TextEdit::singleline(&mut text).id(id));
            app.show(ui);
        },
    );
    output.textures_delta.clear();
    assert_eq!(app.view, View::PolicyMap);
}

#[test]
fn requested_checkpoint_remains_visible_without_retained_weights() {
    let (_directory, context, mut app) = fixture();
    let output = frame(&context, &mut app, vec![]);
    fn contains(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(t) => t.galley.text() == "k 7",
            egui::Shape::Vec(v) => v.iter().any(contains),
            _ => false,
        }
    }
    assert!(
        output.shapes.iter().any(|s| contains(&s.shape)),
        "The requested checkpoint must be visible even while navigation or weights are unavailable"
    );
}

#[test]
fn checkpoint_readout_uses_the_revision_selected_during_this_paint() {
    let (_directory, context, mut app) = fixture();
    app.select_revision(0).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        frame(&context, &mut app, vec![]);
        if app.policy_pane.ready() && app.policy_frame().is_some() {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = frame(
        &context,
        &mut app,
        vec![egui::Event::Key {
            key: egui::Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert_eq!(app.selection.revision, 7);
    fn contains(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(t) => t.galley.text() == "k 7",
            egui::Shape::Vec(v) => v.iter().any(contains),
            _ => false,
        }
    }
    assert!(
        output.shapes.iter().any(|s| contains(&s.shape)),
        "The requested readout must update atomically with keyboard/clock navigation, not one paint later"
    );
}

#[test]
fn episode_only_recording_opens_a_useful_exact_decision_without_input_configuration() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("episode-only");
    let space = episode_fixture::bounded(3);
    let mut manifest = episode_fixture::manifest(space.clone());
    manifest.environment_descriptor = None;
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    let mut episode = episode_fixture::chunk(&space);
    episode.context.identity.model_role = "online".into();
    recorder.append(&episode).unwrap();
    recorder.finish().unwrap();

    let context = egui::Context::default();
    let mut app = Laboratory::from_context(&context, path.display().to_string()).unwrap();
    assert_eq!(
        app.view,
        View::Dynamics,
        "Episode-only recordings need a useful initial workspace"
    );
    // Dynamics remains the owner of exact episode admission and selection.
    assert_eq!(app.selection.input, SelectedInput::None);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !app.dynamics_ready() {
        frame(&context, &mut app, vec![]);
        assert!(app.notice().is_none(), "{:?}", app.notice());
        assert!(Instant::now() < deadline, "{:?}", app.selection);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(app.selection.revision, 17);
    assert_eq!(app.selection.model_role, "online");
    assert_eq!(
        app.selection.input,
        SelectedInput::TrajectoryStep {
            sequence: 0,
            step: 0
        }
    );
    fn contains(shape: &egui::Shape, value: &str) -> bool {
        match shape {
            egui::Shape::Text(t) => t.galley.text() == value,
            egui::Shape::Vec(v) => v.iter().any(|shape| contains(shape, value)),
            _ => false,
        }
    }
    let output = frame(&context, &mut app, vec![]);
    assert!(
        output
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "Episode 1 · k17 · step 0"))
    );
    assert!(
        !output
            .shapes
            .iter()
            .any(|shape| contains(&shape.shape, "#0 · k17 · t0"))
    );
    let (key, recorded) = app.dynamics_episode().unwrap();
    assert_eq!(key.sequence, 0);
    assert_eq!(recorded.context().identity.learner_update_revision, 17);
    let (_directory, _context, probe_app) = fixture();
    assert_eq!(
        probe_app.view,
        View::PolicyMap,
        "A valid policy population keeps its ordinary default"
    );
}

#[test]
fn training_is_the_default_for_history_and_keeps_the_selected_input_across_workspaces() {
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("training");
    let action_space = ActionSpace::Discrete { count: 2 };
    let manifest =
        RunManifest::new("train", "Training", "learner", 1, action_space.clone(), 0.9).unwrap();
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for k in [0, 7] {
        let context = RecordContext::new("train", k);
        recorder
            .append(
                &ProbeSet::new(
                    "fixed",
                    ProbePopulation::FixedReference,
                    "fixed input",
                    vec![Observation::new(vec![0.])],
                    None,
                    action_space.clone(),
                )
                .unwrap()
                .record(context.clone())
                .unwrap(),
            )
            .unwrap();
        recorder
            .append(&Chunk {
                context,
                record: Record::Metrics(MetricRecord {
                    values: std::collections::BTreeMap::from([("loss".into(), 7. - k as f64)]),
                    unavailable: Default::default(),
                }),
                arrays: Default::default(),
            })
            .unwrap();
    }
    recorder.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    assert_eq!(app.view, View::Training);
    app.workspace_defaults = None;
    let input = app.selection.input.clone();
    app.select_revision(0).unwrap();
    let limit = Instant::now() + Duration::from_secs(3);
    while app.learning_pane.worker.training_frame().is_none() && Instant::now() < limit {
        frame(&ctx, &mut app, Vec::new());
        std::thread::sleep(Duration::from_millis(5));
    }
    let prepared = app
        .learning_pane
        .worker
        .training_frame()
        .expect("training history");
    assert_eq!(prepared.primary.metrics.len(), 1);
    let mut play = key(egui::Key::Space);
    if let egui::Event::Key { modifiers, .. } = &mut play {
        *modifiers = egui::Modifiers::NONE;
    }
    frame(&ctx, &mut app, vec![play]);
    assert!(app.training_is_playing());
    let limit = Instant::now() + Duration::from_secs(3);
    while app.selection.revision == 0 && Instant::now() < limit {
        frame(&ctx, &mut app, Vec::new());
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        app.selection.revision, 7,
        "Training playback must advance after publication"
    );
    assert_eq!(app.selection.input, input);
    app.select_revision(0).unwrap();

    for view in [
        View::PolicyMap,
        View::Value,
        View::Network,
        View::Dynamics,
        View::Environment,
        View::Training,
    ] {
        app.select_view(view);
        assert_eq!(app.selection.input, input);
        assert_eq!(app.selection.revision, 0);
    }
}

#[test]
fn opening_empty_directory_preserves_active_recording_and_run_catalogue() {
    let (directory, _context, mut app) = fixture();
    let original_root = app.run.as_ref().unwrap().root().clone();
    let original_catalogue: Vec<_> = app
        .catalogue
        .iter()
        .map(|entry| entry.root.clone())
        .collect();
    let original_revision = app.selection.revision;
    let original_path = app.path.clone();
    let empty = directory.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(!app.open_directory(&empty));
    assert_eq!(app.path, original_path);
    assert_eq!(app.run.as_ref().unwrap().root(), &original_root);
    assert_eq!(
        app.catalogue
            .iter()
            .map(|entry| entry.root.clone())
            .collect::<Vec<_>>(),
        original_catalogue
    );
    assert_eq!(app.selection.revision, original_revision);
    assert!(
        app.notice().is_some(),
        "An empty folder must give a useful notice"
    );
}

#[test]
fn empty_start_screen_offers_to_open_a_recording() {
    let directory = tempfile::tempdir().unwrap();
    let context = egui::Context::default();
    let mut app = Laboratory::from_context(
        &context,
        directory.path().join("missing").display().to_string(),
    )
    .unwrap();
    let output = frame(&context, &mut app, vec![]);
    fn contains(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(text) => text.galley.text() == "Open recording",
            egui::Shape::Vec(shapes) => shapes.iter().any(contains),
            _ => false,
        }
    }
    assert!(output.shapes.iter().any(|shape| contains(&shape.shape)));
}

#[test]
fn opened_recording_is_available_on_the_start_screen() {
    let (_directory, context, mut app) = fixture();
    app.clear_recording();
    let output = frame(&context, &mut app, vec![]);
    fn contains(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(text) => text.galley.text() == "Recent recordings",
            egui::Shape::Vec(shapes) => shapes.iter().any(contains),
            _ => false,
        }
    }
    assert!(output.shapes.iter().any(|shape| contains(&shape.shape)));
}

#[allow(dead_code)]
#[path = "../../tests/support/learning_critic.rs"]
mod training_fixture;

#[test]
fn training_shows_evaluation_and_available_secondary_metric_together() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("training-composition");
    let space = ActionSpace::Discrete { count: 2 };
    let manifest = RunManifest::new("learning", "learning", "ordinary", 1, space, 0.9).unwrap();
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for revision in [0, 7] {
        recorder
            .append(&training_fixture::empirical(
                revision,
                vec![1. + revision as f64, 2.],
            ))
            .unwrap();
        recorder
            .append(&Chunk {
                context: RecordContext::new("learning", revision),
                record: Record::Metrics(MetricRecord {
                    values: std::collections::BTreeMap::from([(
                        "critic_loss".into(),
                        1. / (revision + 1) as f64,
                    )]),
                    unavailable: Default::default(),
                }),
                arrays: Default::default(),
            })
            .unwrap();
    }
    recorder.finish().unwrap();
    let context = egui::Context::default();
    let mut app = Laboratory::from_context(&context, path.display().to_string()).unwrap();
    app.live = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.learning_pane.worker.training_frame().is_none() {
        frame(&context, &mut app, vec![]);
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = frame(&context, &mut app, vec![]);
    fn contains(shape: &egui::Shape, expected: &str) -> bool {
        match shape {
            egui::Shape::Text(text) => text.galley.text() == expected,
            egui::Shape::Vec(shapes) => shapes.iter().any(|shape| contains(shape, expected)),
            _ => false,
        }
    }
    for label in ["Evaluation performance", "critic loss", "Secondary plots"] {
        assert!(
            output
                .shapes
                .iter()
                .any(|shape| contains(&shape.shape, label)),
            "missing {label}"
        );
    }
}

#[test]
fn training_does_not_show_local_provider_gap_for_an_external_provider_with_the_same_role() {
    use rl::diagnostics::{
        reference::{
            ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceInformation, ReferenceKind,
            ReferenceLifetime, ReferenceObjective, ReferenceQuality, ReferenceSpec,
            ReferenceTarget, attach_manifest_context,
        },
        value::ValueConventions,
    };
    let directory = tempfile::tempdir().unwrap();
    let space = ActionSpace::Discrete { count: 2 };
    let context = ReferenceContext {
        environment: "custom".into(),
        observation_dim: 1,
        typed_observation_space: None,
        action_space: space.clone(),
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: std::collections::BTreeMap::from([(
            "horizon".into(),
            ConfigValue::Integer(2),
        )]),
        markov_observation: true,
        decision_time_coordinate: Some(0),
        conventions: ValueConventions {
            dynamics: "two decisions".into(),
            observation: "state".into(),
            action: "discrete action".into(),
            reward_units: "reward".into(),
            reward_convention: "reward, larger is better; original environment units".into(),
            discount: 0.9,
            horizon: "two decisions".into(),
            terminal_treatment: "no continuation".into(),
            action_constraints: "two legal actions".into(),
        },
    };
    let spec = ReferenceSpec {
        id: "benchmark".into(),
        context: context.clone(),
        quality: ReferenceQuality {
            kind: ReferenceKind::Heuristic,
            method: "test policy".into(),
            assumptions: vec![],
        },
        target: ReferenceTarget::BenchmarkPolicy,
        lifetime: ReferenceLifetime::Fixed,
        information: ReferenceInformation::LearnerObservation,
        uncertainty: None,
    };
    let spec_bytes = serde_json::to_vec(&spec).unwrap();
    for (name, gap) in [("primary", 1.0), ("external", 9.0)] {
        let mut manifest =
            RunManifest::new("learning", "learning", "ordinary", 1, space.clone(), 0.9).unwrap();
        attach_manifest_context(&mut manifest, &context).unwrap();
        let mut recorder = Recorder::create(
            directory.path().join(name),
            manifest,
            StorageConfig::default(),
        )
        .unwrap();
        recorder
            .append(&training_fixture::empirical(0, vec![1., 2.]))
            .unwrap();
        for role in ["online", "reference:benchmark"] {
            let mut record_context = RecordContext::new("learning", 0);
            record_context.identity.model_role = role.into();
            recorder
                .append(&Chunk {
                    context: record_context,
                    record: Record::Metrics(MetricRecord {
                        values: std::collections::BTreeMap::from([(
                            "reference.benchmark_gap".into(),
                            gap,
                        )]),
                        unavailable: Default::default(),
                    }),
                    arrays: std::collections::BTreeMap::from([(
                        "metadata/reference-provider.v1".into(),
                        Array::u8(vec![spec_bytes.len()], spec_bytes.clone()).unwrap(),
                    )]),
                })
                .unwrap();
        }
        recorder.finish().unwrap();
    }
    let ctx = egui::Context::default();
    let mut app =
        Laboratory::from_context(&ctx, directory.path().join("primary").display().to_string())
            .unwrap();
    app.live = false;
    app.workspace_defaults = None;
    app.select_view(View::Training);
    app.comparison
        .set_reference_role("reference:benchmark")
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let contains = |output: &egui::FullOutput, expected: &str| {
        fn in_shape(shape: &egui::Shape, expected: &str) -> bool {
            match shape {
                egui::Shape::Text(t) => t.galley.text() == expected,
                egui::Shape::Vec(shapes) => shapes.iter().any(|s| in_shape(s, expected)),
                _ => false,
            }
        }
        output
            .shapes
            .iter()
            .any(|shape| in_shape(&shape.shape, expected))
    };
    loop {
        if contains(&frame(&ctx, &mut app, vec![]), "Signed benchmark gap") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "local benchmark gap did not load"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let external =
        RunIndex::open(directory.path().join("external"), ReadLimits::default()).unwrap();
    app.comparison.reset_reference_to(&external).unwrap();
    app.comparison
        .set_reference_role("reference:benchmark")
        .unwrap();
    for _ in 0..3 {
        assert!(!contains(
            &frame(&ctx, &mut app, vec![]),
            "Signed benchmark gap"
        ));
    }
}
