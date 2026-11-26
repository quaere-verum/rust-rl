#![cfg(feature = "diagnostics")]
use rl as recording_rl;

#[path = "support/recording_env.rs"]
mod recording_env;
use recording_env::{NAME, RecordingEnv};
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Env,
    },
    diagnostics::{
        evaluation::{EvaluationConfig, evaluate_with_frames},
        schema::{RecordContext, RunManifest},
        storage::*,
        visualization::*,
    },
    train::trainer::RandomActor,
};

fn manifest() -> RunManifest {
    RunManifest::new(
        "semantics",
        "test",
        "ordinary",
        3,
        ActionSpace::Unbounded { dimensions: 1 },
        0.9,
    )
    .unwrap()
}
fn descriptor() -> EnvironmentDescriptor {
    let field = |id: &str, label: &str, range| ObservationField {
        id: id.into(),
        label: label.into(),
        unit: None,
        description: None,
        kind: ObservationType::Numerical,
        range,
        coordinate: true,
    };
    let clock = |label: &str, unit: Option<&str>| QuantityLabel {
        label: label.into(),
        unit: unit.map(str::to_owned),
    };
    let frame = |name: &str, label: &str| FrameChannel {
        name: name.into(),
        label: label.into(),
        width: 1,
        unit: None,
        visibility: FrameVisibility::DiagnosticOnly,
        representation: Some(FrameRepresentation::Scalar),
    };
    EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: vec![
            field("position", "Position", None),
            field("velocity", "Velocity", None),
            field("remaining_fraction", "Remaining fraction", Some([0., 1.])),
        ],
        actions: ActionDescriptor::Continuous {
            components: vec![ActionComponent {
                label: "Velocity command".into(),
                unit: None,
                meaning: None,
            }],
            preferred_component: Some(0),
        },
        clocks: ClockDescriptor {
            learner_revision: clock("Learner revision", None),
            decision_step: clock("Decision step", Some("steps")),
            simulation: Some(clock("Simulation time", Some("steps"))),
        },
        reward: RewardDescriptor {
            label: "Negative distance plus disturbance".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![
            frame("position", "Position"),
            frame("diagnostic_signal", "Private signal"),
        ],
        preferences: VisualizationPreferences {
            coordinates: vec!["position".into(), "remaining_fraction".into()],
            trajectories: vec![
                FrameReference {
                    channel: "position".into(),
                    index: 0,
                },
                FrameReference {
                    channel: "diagnostic_signal".into(),
                    index: 0,
                },
            ],
            groups: vec![],
        },
    }
}

#[test]
fn optional_descriptor_roundtrips_all_supported_manifests_and_unknown_fields() {
    for version in 1..=3 {
        let mut m = manifest();
        m.schema_version = version;
        let mut json = serde_json::to_value(&m).unwrap();
        assert!(json.get("environment_descriptor").is_none());
        json["future_optional"] = serde_json::json!({"ignored": true});
        let old: RunManifest = serde_json::from_value(json).unwrap();
        old.validate().unwrap();
        assert!(old.environment_descriptor.is_none());
        m.environment_descriptor = Some(descriptor());
        let mut json = serde_json::to_value(&m).unwrap();
        json["environment_descriptor"]["future_optional"] = true.into();
        let reopened: RunManifest = serde_json::from_value(json).unwrap();
        reopened.validate().unwrap();
        assert_eq!(reopened.environment_descriptor.unwrap(), descriptor());
    }
    let mut m = manifest();
    let mut d = descriptor();
    d.version = 2;
    m.environment_descriptor = Some(d);
    assert!(
        m.validate()
            .unwrap_err()
            .to_string()
            .contains("unsupported environment descriptor version")
    );
}

#[test]
fn omitted_frame_visibility_remains_unknown_and_storage_limits_still_apply() {
    let mut json = serde_json::to_value(descriptor()).unwrap();
    json["frames"][0]
        .as_object_mut()
        .unwrap()
        .remove("visibility");
    json["preferences"] = serde_json::json!({});
    let d: EnvironmentDescriptor = serde_json::from_value(json).unwrap();
    assert_eq!(d.frames[0].visibility, FrameVisibility::Unknown);
    d.validate(3, &ActionSpace::Unbounded { dimensions: 1 })
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let mut m = manifest();
    m.environment_descriptor = Some(descriptor());
    assert!(
        Recorder::create(
            temp.path().join("small-writer"),
            m.clone(),
            StorageConfig {
                max_metadata_bytes: 128,
                ..Default::default()
            }
        )
        .is_err()
    );
    let path = temp.path().join("normal");
    Recorder::create(&path, m, StorageConfig::default())
        .unwrap()
        .finish()
        .unwrap();
    assert!(
        Recording::open(
            &path,
            ReadLimits {
                max_metadata_bytes: 128,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_dimensions_actions_identifiers_ranges_and_categories() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    assert!(descriptor().validate(2, &space).is_err());
    assert!(
        descriptor()
            .validate(3, &ActionSpace::Discrete { count: 1 })
            .is_err()
    );
    assert!(
        descriptor()
            .validate(3, &ActionSpace::Unbounded { dimensions: 2 })
            .is_err()
    );
    for id in ["", " position", "position\n", "\u{0000}"] {
        let mut d = descriptor();
        d.observations[0].id = id.into();
        assert!(d.validate(3, &space).is_err());
    }
    let mut d = descriptor();
    d.observations[1].id = "position".into();
    assert!(d.validate(3, &space).is_err());
    for range in [[1., 0.], [f32::NAN, 1.], [0., f32::INFINITY]] {
        let mut d = descriptor();
        d.observations[0].range = Some(range);
        assert!(d.validate(3, &space).is_err());
    }
    for codes in [
        vec![0., -0.],
        vec![1., 1.],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![],
    ] {
        let mut d = descriptor();
        d.observations[0].kind = ObservationType::Categorical {
            categories: codes
                .into_iter()
                .map(|code| Category {
                    code,
                    label: "category".into(),
                })
                .collect(),
        };
        assert!(d.validate(3, &space).is_err());
    }
    let mut d = descriptor();
    d.observations[0].kind = ObservationType::Categorical {
        categories: vec![
            Category {
                code: -2.,
                label: "off".into(),
            },
            Category {
                code: 3.,
                label: "on".into(),
            },
        ],
    };
    d.validate(3, &space).unwrap();
    d.actions = ActionDescriptor::Discrete {
        choices: vec![
            DiscreteChoice {
                label: "wait".into(),
                meaning: Some("keep state".into()),
            },
            DiscreteChoice {
                label: "go".into(),
                meaning: None,
            },
        ],
    };
    d.validate(3, &ActionSpace::Discrete { count: 2 }).unwrap();
    assert!(d.validate(3, &ActionSpace::Discrete { count: 1 }).is_err());
}

#[test]
fn rejects_unresolved_preferences_frames_components_and_metadata_budgets() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    type Invalidation = Box<dyn Fn(&mut EnvironmentDescriptor)>;
    let cases: Vec<Invalidation> = vec![
        Box::new(|d| d.preferences.coordinates.push("missing".into())),
        Box::new(|d| d.preferences.coordinates.push("position".into())),
        Box::new(|d| d.observations[0].coordinate = false),
        Box::new(|d| {
            d.preferences.trajectories.push(FrameReference {
                channel: "missing".into(),
                index: 0,
            })
        }),
        Box::new(|d| {
            d.reward.components.push(FrameReference {
                channel: "position".into(),
                index: 1,
            })
        }),
        Box::new(|d| d.frames[0].width = 0),
        Box::new(|d| d.frames[1].name = "position".into()),
        Box::new(|d| {
            d.preferences.groups.push(ObservationGroup {
                id: "group".into(),
                label: "group".into(),
                observations: vec!["missing".into()],
            })
        }),
        Box::new(|d| {
            d.preferences.groups = vec![
                ObservationGroup {
                    id: "g".into(),
                    label: "group".into(),
                    observations: vec![]
                };
                2
            ]
        }),
        Box::new(|d| {
            if let ActionDescriptor::Continuous {
                preferred_component,
                ..
            } = &mut d.actions
            {
                *preferred_component = Some(1);
            }
        }),
        Box::new(|d| d.observations[0].id = "é".repeat(65)),
        Box::new(|d| d.observations[0].unit = Some("x".repeat(129))),
        Box::new(|d| d.observations[0].label = "x".repeat(1025)),
        Box::new(|d| {
            d.frames = (0..129)
                .map(|i| FrameChannel {
                    name: format!("c{i}"),
                    ..d.frames[0].clone()
                })
                .collect()
        }),
        Box::new(|d| {
            d.preferences.groups = (0..65)
                .map(|i| ObservationGroup {
                    id: format!("g{i}"),
                    label: "g".into(),
                    observations: vec![],
                })
                .collect()
        }),
        Box::new(|d| {
            d.observations[0].kind = ObservationType::Categorical {
                categories: (0..257)
                    .map(|i| Category {
                        code: i as f32,
                        label: "c".into(),
                    })
                    .collect(),
            }
        }),
    ];
    for (i, invalidate) in cases.into_iter().enumerate() {
        let mut d = descriptor();
        invalidate(&mut d);
        assert!(d.validate(3, &space).is_err(), "case {i}");
    }
    let mut d = descriptor();
    d.observations = (0..4097)
        .map(|i| ObservationField {
            id: format!("o{i}"),
            ..d.observations[0].clone()
        })
        .collect();
    assert!(d.validate(4097, &space).is_err());
    let mut d = descriptor();
    d.frames = (0..128)
        .map(|i| FrameChannel {
            name: format!("c{i}"),
            label: "x".repeat(1024),
            ..d.frames[0].clone()
        })
        .collect();
    d.preferences.trajectories.clear();
    assert!(d.validate(3, &space).is_err()); // Encoded aggregate, despite valid local sizes.
    let mut d = descriptor();
    d.reward.components.push(FrameReference {
        channel: "position".into(),
        index: 0,
    });
    d.preferences.groups.push(ObservationGroup {
        id: "state".into(),
        label: "State".into(),
        observations: vec!["position".into(), "velocity".into()],
    });
    d.validate(3, &space).unwrap();
}

#[test]
fn fallback_is_lazy_for_extreme_dimensions_and_invents_no_metadata() {
    let mut m = manifest();
    m.observation_dim = usize::MAX;
    m.action_space = ActionSpace::Discrete { count: usize::MAX };
    m.validate().unwrap();
    let s = EnvironmentSemantics::new(&m);
    let o = s.observation(usize::MAX - 1).unwrap();
    assert_eq!(o.id, format!("obs[{}]", usize::MAX - 1));
    assert!(o.kind.is_none() && o.unit.is_none() && o.range.is_none() && !o.coordinate);
    assert!(s.observation(usize::MAX).is_none());
    assert_eq!(s.action(12).unwrap().label, "12");
    assert!(s.action(usize::MAX).is_none());
    assert!(s.clocks().is_none() && s.frame("position").is_none() && s.preferences().is_none());
    assert_eq!(s.action_space(), &m.action_space);
    m.observation_dim = 3;
    m.observation_labels = vec![
        "legacy position".into(),
        "legacy velocity".into(),
        "".into(),
    ];
    m.action_space = ActionSpace::Unbounded {
        dimensions: usize::MAX,
    };
    let s = EnvironmentSemantics::new(&m);
    assert_eq!(s.observation(0).unwrap().label, "legacy position");
    assert_eq!(s.observation(2).unwrap().label, "obs[2]");
    assert_eq!(s.action(12).unwrap().label, "action[12]");
}

#[test]
fn numeric_metadata_and_real_dynamics_survive_recorder_reopen() {
    let mut env = RecordingEnv::new(2, 1);
    assert_eq!(
        env.reset(None).unwrap().observation.dense_values(),
        &[2., 0., 1.]
    );
    let first = env.step(&Action::Continuous(vec![0.5])).unwrap();
    assert_eq!(first.observation.dense_values(), &[2.5, 0.5, 0.5]);
    let final_step = env.step(&Action::Continuous(vec![-0.5])).unwrap();
    assert_eq!(final_step.observation.dense_values(), &[2., -0.5, 0.]);
    assert!(final_step.terminated);
    assert!(env.step(&Action::Continuous(vec![0.])).is_err());
    let mut m = manifest();
    m.environment_descriptor = Some(descriptor());
    m.environment = NAME.into();
    m.action_space = env.action_space().clone();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join(NAME);
    let actor = RandomActor {
        space: env.action_space().clone(),
    };
    let result = evaluate_with_frames(
        |seed| Ok(RecordingEnv::new(2, seed)),
        &actor,
        &EvaluationConfig {
            seeds: vec![1],
            max_steps: 2,
            trajectories: 1,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    let trajectory = result
        .trajectory_record(0, RecordContext::new("semantics", 0), NAME, NAME)
        .unwrap();
    let mut writer = Recorder::create(&path, m, StorageConfig::default()).unwrap();
    writer.append(&trajectory).unwrap();
    writer.finish().unwrap();
    let r = Recording::open(&path, ReadLimits::default()).unwrap();
    let s = EnvironmentSemantics::new(r.manifest());
    assert_eq!(
        s.observation_by_id("remaining_fraction").unwrap().range,
        Some([0., 1.])
    );
    assert_eq!(s.observation(0).unwrap().id, "position");
    assert_eq!(s.observation(1).unwrap().id, "velocity");
    assert_eq!(
        s.frame("position").unwrap().visibility,
        FrameVisibility::DiagnosticOnly
    );
    assert_eq!(
        s.frame("diagnostic_signal").unwrap().visibility,
        FrameVisibility::DiagnosticOnly
    );
    assert_eq!(
        s.clocks()
            .unwrap()
            .simulation
            .as_ref()
            .unwrap()
            .unit
            .as_deref(),
        Some("steps")
    );
    assert!(s.reward().components.is_empty());
    assert!(s.reward().label.contains("Negative distance"));
    let chunk = r.load(0).unwrap();
    assert_eq!(
        chunk.arrays["frames/position"].as_f64().unwrap(),
        &[2., 2., 2.]
    );
    assert_eq!(
        chunk.arrays["simulation_times"].as_f64().unwrap(),
        &[0., 1., 2.]
    );
    assert_eq!(
        s.frame("position").unwrap().width,
        chunk.arrays["frames/position"].shape()[1]
    );
}
