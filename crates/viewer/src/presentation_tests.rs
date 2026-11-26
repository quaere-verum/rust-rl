use crate::{
    data::{RunIndex, SelectedInput, Selection},
    timeline_ui::TrainingTransport,
};
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use std::{collections::BTreeMap, time::Duration};
fn fixture(
    algorithm: &str,
    environment: &str,
    include_critic: bool,
) -> (tempfile::TempDir, RunIndex, Chunk) {
    fixture_config(algorithm, environment, include_critic, false)
}

fn fixture_config(
    algorithm: &str,
    environment: &str,
    include_critic: bool,
    redefine: bool,
) -> (tempfile::TempDir, RunIndex, Chunk) {
    fixture_layer(
        algorithm,
        environment,
        include_critic,
        redefine,
        "layers.0.post",
    )
}

fn fixture_layer(
    algorithm: &str,
    environment: &str,
    include_critic: bool,
    redefine: bool,
    layer: &str,
) -> (tempfile::TempDir, RunIndex, Chunk) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("defaults");
    let masked = environment == "masked";
    let space = if masked {
        ActionSpace::Discrete { count: 2 }
    } else {
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        }
    };
    let mut manifest =
        RunManifest::new("defaults", "defaults", algorithm, 2, space.clone(), 0.99).unwrap();
    manifest.environment = environment.into();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    let id = if masked { "fixed-masked" } else { "fixed-grid" };
    let mut context = RecordContext::new("defaults", 3);
    context.probe_set = Some(id.into());
    let observations = vec![
        Observation::dense(vec![-1., 0.], masked.then(|| vec![false, true])),
        Observation::dense(vec![0., 0.5], masked.then(|| vec![true, false])),
        Observation::dense(vec![1., 0.], masked.then(|| vec![true, true])),
    ];
    let probes = ProbeSet::new(
        id,
        ProbePopulation::FixedReference,
        "fixed actual inputs",
        observations,
        None,
        space,
    )
    .unwrap()
    .record(context)
    .unwrap();
    writer.append(&probes).unwrap();
    for (revision, kinds) in [
        (5, vec![RecordKind::Activations]),
        (6, vec![RecordKind::Policy, RecordKind::Critic]),
        (
            7,
            vec![
                RecordKind::Activations,
                RecordKind::Policy,
                RecordKind::Critic,
            ],
        ),
        (60, vec![RecordKind::Policy, RecordKind::Critic]),
        (
            1_000_000_007,
            vec![
                RecordKind::Activations,
                RecordKind::Policy,
                RecordKind::Critic,
            ],
        ),
        (1_000_000_100, vec![RecordKind::Activations]),
    ] {
        if redefine && revision == 60 {
            let mut replacement = probes.clone();
            replacement.context = RecordContext::new("defaults", revision);
            replacement.context.probe_set = Some(id.into());
            writer.append(&replacement).unwrap();
        }
        for kind in kinds {
            if kind == RecordKind::Critic && !include_critic {
                continue;
            }
            let mut context = RecordContext::new("defaults", revision);
            context.probe_set = Some(id.into());
            let (record, arrays) = match kind {
                RecordKind::Activations => (
                    Record::Activations(ActivationRecord {
                        network: "policy".into(),
                        layers: BTreeMap::from([(layer.into(), "h".into())]),
                        ..Default::default()
                    }),
                    BTreeMap::from([(
                        "h".into(),
                        Array::f32(vec![3, 3], vec![0., 0., 0., 1., 2., -1., 2., -1., 3.]).unwrap(),
                    )]),
                ),
                RecordKind::Policy => (
                    Record::Policy(PolicyRecord {
                        actions: "a".into(),
                        parameters: PolicyParameters::Deterministic,
                        entropy: None,
                        entropy_method: None,
                        masks: None,
                        stochastic: false,
                    }),
                    BTreeMap::from([(
                        "a".into(),
                        if masked {
                            Array::u64(vec![3], vec![1, 0, 0]).unwrap()
                        } else {
                            Array::f32(vec![3, 1], vec![0.; 3]).unwrap()
                        },
                    )]),
                ),
                RecordKind::Critic => (
                    Record::Critic(CriticRecord {
                        representation: CriticRepresentation::Scalar { values: "v".into() },
                        action_condition: ActionCondition::StateValue,
                        actions: None,
                        continuation_policy: None,
                        continuation_stochastic: None,
                    }),
                    BTreeMap::from([("v".into(), Array::f32(vec![3], vec![0.; 3]).unwrap())]),
                ),
                _ => unreachable!(),
            };
            writer
                .append(&Chunk {
                    context,
                    record,
                    arrays,
                })
                .unwrap();
        }
    }
    writer.finish().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    (directory, run, probes)
}

fn selection(source: &Chunk, revision: u64) -> Selection {
    Selection {
        revision,
        input: SelectedInput::ProbeRow {
            set: source.context.probe_set.clone().unwrap(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    }
}

#[test]
fn initial_selection_uses_latest_policy_endpoint_and_original_source_identity() {
    let (_directory, run, _source) = fixture("PPO", "lqg", true);
    let context = eframe::egui::Context::default();
    let app =
        crate::ui::Laboratory::from_context(&context, run.root().display().to_string()).unwrap();
    assert_eq!(app.selection().revision, 1_000_000_007);
    assert_eq!(app.selection().model_role, "online");
    assert_eq!(
        app.selection().input,
        SelectedInput::ProbeRow {
            set: "fixed-grid".into(),
            row: 0,
            sequence: 0,
        }
    );
}

#[test]
fn initial_defaults_use_recorded_sources_for_other_algorithms_and_environments() {
    for (algorithm, environment, set) in [
        ("PPO", "lqg", "fixed-grid"),
        ("ordinary", "custom", "fixed-grid"),
        ("DDQN", "masked", "fixed-masked"),
    ] {
        let (_directory, run, _source) = fixture(algorithm, environment, false);
        let context = eframe::egui::Context::default();
        let app = crate::ui::Laboratory::from_context(&context, run.root().display().to_string())
            .unwrap();
        assert_eq!(app.selection().revision, 1_000_000_007);
        assert_eq!(
            app.selection().input,
            SelectedInput::ProbeRow {
                set: set.into(),
                row: 0,
                sequence: 0,
            }
        );
    }
}

#[test]
fn same_name_population_redefinition_selects_its_own_source_without_rebinding_old_evidence() {
    let (_directory, run, source) = fixture_config("PPO", "lqg", true, true);
    let context = eframe::egui::Context::default();
    let app =
        crate::ui::Laboratory::from_context(&context, run.root().display().to_string()).unwrap();
    let replacement = run.probe_set("fixed-grid", 1_000_000_007).unwrap();
    let replacement_sequence = run.recording().entries()[replacement].sequence;
    assert_ne!(replacement_sequence, 0);
    assert_eq!(
        app.selection().input,
        SelectedInput::ProbeRow {
            set: "fixed-grid".into(),
            row: 0,
            sequence: replacement_sequence,
        }
    );
    let original = run.by_sequence(0).unwrap();
    let original_revisions: Vec<_> = run
        .candidates(RecordKind::Policy, "online", Some("fixed-grid"), u64::MAX)
        .filter(|&index| run.probe_for_record(index) == Some(original))
        .map(|index| run.recording().entries()[index].revision)
        .collect();
    assert_eq!(original_revisions, vec![7, 6]);
    let replacement_revisions: Vec<_> = run
        .candidates(RecordKind::Policy, "online", Some("fixed-grid"), u64::MAX)
        .filter(|&index| run.probe_for_record(index) == Some(replacement))
        .map(|index| run.recording().entries()[index].revision)
        .collect();
    assert_eq!(replacement_revisions, vec![1_000_000_007, 60]);
    assert_eq!(
        selection(&source, 1_000_000_007).input,
        SelectedInput::ProbeRow {
            set: "fixed-grid".into(),
            row: 0,
            sequence: 0,
        }
    );
}

#[test]
fn pending_defaults_follow_a_manual_sparse_revision_and_retain_discovered_layer() {
    use crate::{
        data::Loader,
        presentation::{WorkspaceDefaults, initial_selection},
    };
    let (_directory, run, _source) =
        fixture_layer("ordinary", "custom", true, false, "custom.hidden.post");
    let mut selected = initial_selection(&run);
    let mut defaults = WorkspaceDefaults::new(&run, &selected);
    selected.select_revision(&run, 60).unwrap();
    defaults.update_revision(&run, &selected);
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        loader.begin_pass();
        loader.poll();
        let ready = defaults.poll(&run, &mut loader).unwrap();
        loader.finish_pass().unwrap();
        assert_eq!(selected.revision, 60);
        assert_eq!(
            selected.input,
            SelectedInput::ProbeRow {
                set: "fixed-grid".into(),
                row: 0,
                sequence: 0,
            }
        );
        if ready {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "workspace defaults did not settle"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    let plan = defaults.finish().expect("discovered defaults");
    assert_eq!(plan.layer, "custom.hidden.post");
    assert_eq!(plan.input, selected.input);
    assert!(plan.reference_revision <= selected.revision);
}
#[test]
fn transport_keyboard_play_pause_and_endpoint_step_preserve_selected_input() {
    use eframe::egui;
    let (_directory, run, source) = fixture("PPO", "lqg", true);
    let mut selection = selection(&source, 7);
    let mut transport = TrainingTransport::default();
    let original_input = selection.input.clone();
    transport
        .start_snapshot_playback(&run, &mut selection, &[7, 1_000_000_007], 1.)
        .unwrap();
    transport.pause();
    let context = egui::Context::default();
    for (frame, (key, pressed)) in [
        (egui::Key::Space, true),
        (egui::Key::Space, false),
        (egui::Key::Space, true),
        (egui::Key::ArrowLeft, true),
    ]
    .into_iter()
    .enumerate()
    {
        let mut output = context.run_ui(
            egui::RawInput {
                time: Some(frame as f64 / 60.),
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed,
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
                        crate::timeline_ui::SnapshotPresentation {
                            positions: &[7, 1_000_000_007],
                            published: Some(7),
                            ready: true,
                            checking: false,
                            error: None,
                        },
                    )
                    .unwrap();
            },
        );
        output.textures_delta.clear();
        assert_eq!(transport.is_playing(), frame < 2);
        assert_eq!(selection.input, original_input);
    }
    assert_eq!(selection.revision, 7);
    assert!(!transport.is_playing());
}
#[test]
fn bundled_demo_labels_preserve_the_recording_identifier_contract() {
    for (identity, label) in [
        ("rocket-ppo-release-candidate", "Rocket Landing · PPO"),
        (
            "alpha-d4pg-release-candidate",
            "Alpha Execution · Expected Return",
        ),
        ("alpha-cvar-release-candidate", "Alpha Execution · CVaR"),
        ("other-d4pg-run", "Other D4PG Run"),
    ] {
        assert_eq!(super::display_run_identifier(identity), label);
    }
    assert_eq!(
        super::display_identifier("alpha-d4pg-release-candidate"),
        "Alpha D4PG Release Candidate"
    );
}
