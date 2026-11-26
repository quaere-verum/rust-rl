use super::*;
use rl::{
    core::{
        action::{ActionSpace, FiniteActionKey},
        env::{DiscreteObservationSpace, Observation, ObservationSpace},
    },
    diagnostics::{
        probes::ProbeSet,
        storage::{Recorder, StorageConfig},
        typed::{FiniteCandidate, FinitePredictions},
    },
};

#[test]
fn selected_symbol_never_becomes_a_different_checkpoint_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("typed");
    let symbol = 1_u64 << 55;
    let obs = ObservationSpace::Discrete(DiscreteObservationSpace {
        symbols: symbol + 3,
    });
    let actions = ActionSpace::Discrete { count: 2 };
    let manifest = RunManifest::new_in(
        "typed",
        "test",
        "tabular",
        obs.clone(),
        actions.clone(),
        0.9,
    )
    .unwrap();
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for (revision, state) in [(0, symbol), (1, symbol + 1), (2, symbol)] {
        let probes = ProbeSet::new_in(
            "fixed",
            ProbePopulation::FixedReference,
            "typed selection regression",
            vec![Observation::discrete(state, None).into()],
            None,
            obs.clone(),
            actions.clone(),
        )
        .unwrap();
        let context = RecordContext::new("typed", revision);
        recorder
            .append(&probes.record(context.clone()).unwrap())
            .unwrap();
        let frame = FinitePredictions {
            value_kind: Some(FiniteValueKind::ActionValue),
            rows: vec![FiniteDecision {
                row_id: 0,
                information_state: Some(state.to_string()),
                selected: Some(FiniteActionKey::Discrete(0)),
                candidates: (0..2)
                    .rev()
                    .map(|index| FiniteCandidate {
                        action: FiniteActionKey::Discrete(index),
                        probability: Some(if index == 0 { 1.0 } else { 0.0 }),
                        value: Some(revision as f64 - index as f64),
                        visits: None,
                        updates: None,
                    })
                    .collect(),
            }],
        };
        recorder
            .append(&frame.record(context, &probes, 64 * 1024).unwrap())
            .unwrap();
    }
    recorder.finish().unwrap();
    let ctx = egui::Context::default();
    let mut app = Laboratory::from_context(&ctx, path.display().to_string()).unwrap();
    app.select_view(View::PolicyMap);
    let paint = |app: &mut Laboratory| {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280., 800.),
                )),
                ..Default::default()
            },
            |ui| app.show(ui),
        );
        output.textures_delta.clear();
    };
    let wait = |app: &mut Laboratory, revision| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while app.typed_pane.ready != Some(revision) {
            paint(app);
            assert!(app.notice().is_none(), "{:?}", app.notice());
            assert!(
                Instant::now() < deadline,
                "revision {revision} did not publish"
            );
            std::thread::sleep(Duration::from_millis(3));
        }
    };
    wait(&mut app, 2);
    let original = app.selection().input.clone();
    app.select_revision(1).unwrap();
    for _ in 0..60 {
        paint(&mut app);
        assert_eq!(app.selection().revision, 1);
        assert_eq!(app.selection().input, original);
        assert_eq!(app.typed_pane.ready, None);
        assert!(app.notice().is_none(), "{:?}", app.notice());
        std::thread::sleep(Duration::from_millis(3));
    }
    app.select_view(View::Value);
    app.select_revision(0).unwrap();
    wait(&mut app, 0);
    assert_eq!(app.selection().input, original);
}

#[test]
fn mixed_runs_resolve_each_prediction_to_its_authored_input_kind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed");
    let actions = ActionSpace::Discrete { count: 1 };
    let manifest = RunManifest::new("mixed", "test", "fitted", 1, actions.clone(), 0.9).unwrap();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    let write_policy = |writer: &mut Recorder, context: RecordContext| {
        writer
            .append(&Chunk {
                context,
                record: Record::Policy(PolicyRecord {
                    actions: "actions".into(),
                    parameters: PolicyParameters::Deterministic,
                    entropy: None,
                    entropy_method: None,
                    masks: None,
                    stochastic: false,
                }),
                arrays: std::collections::BTreeMap::from([(
                    "actions".into(),
                    Array::u64(vec![1], vec![0]).unwrap(),
                )]),
            })
            .unwrap();
    };
    for revision in 0..=1 {
        let mut context = RecordContext::new("mixed", revision);
        context.probe_set = Some("probes".into());
        let dense = ProbeSet::new(
            "probes",
            ProbePopulation::FixedReference,
            "dense input",
            vec![Observation::new(vec![0.0])],
            None,
            actions.clone(),
        )
        .unwrap();
        writer
            .append(&dense.record(context.clone()).unwrap())
            .unwrap();
        if revision == 1 {
            let typed = ProbeSet::new_in(
                "probes",
                ProbePopulation::FixedReference,
                "typed input",
                vec![Observation::new(vec![0.0]).into()],
                None,
                ObservationSpace::Dense(1),
                actions.clone(),
            )
            .unwrap();
            writer
                .append(&typed.record(context.clone()).unwrap())
                .unwrap();
            let frame = FinitePredictions {
                value_kind: Some(FiniteValueKind::PreferenceScore),
                rows: vec![FiniteDecision {
                    row_id: 0,
                    information_state: None,
                    selected: Some(FiniteActionKey::Discrete(0)),
                    candidates: vec![FiniteCandidate {
                        action: FiniteActionKey::Discrete(0),
                        value: Some(1.0),
                        probability: Some(1.0),
                        visits: None,
                        updates: None,
                    }],
                }],
            };
            writer
                .append(&frame.record(context.clone(), &typed, 4096).unwrap())
                .unwrap();
        }
        write_policy(&mut writer, context);
    }
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let entries = run.recording().entries();
    assert_eq!(
        entries[run.probe_for_record(4).unwrap()].kind,
        RecordKind::TypedProbes
    );
    assert_eq!(
        entries[run.probe_for_record(5).unwrap()].kind,
        RecordKind::ProbeSet
    );
    let mut selection = crate::presentation::initial_selection(&run);
    assert!(typed_view(&run, View::PolicyMap, &selection));
    selection.select_revision(&run, 0).unwrap();
    assert!(!typed_view(&run, View::PolicyMap, &selection));
}
