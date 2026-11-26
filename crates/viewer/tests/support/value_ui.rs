use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::SnapshotBuilder, probes::ProbeSet, schema::*, storage::*, value::*},
    nn::{
        critics::{
            quantile::QuantileGrid,
            scalar::{DiscreteQ, QValue},
        },
        matrix,
        network::Mlp,
        policy::Deterministic,
        vector,
    },
};
use std::collections::BTreeMap;
pub fn family(kind: &str, known: bool) -> tempfile::TempDir {
    family_with_heads(kind, known, 16)
}
pub fn family_with_heads(kind: &str, known: bool, heads: usize) -> tempfile::TempDir {
    family_with_actor(kind, known, heads, false)
}
pub fn family_with_actor(
    kind: &str,
    known: bool,
    heads: usize,
    stochastic: bool,
) -> tempfile::TempDir {
    family_with_action_bounds(kind, known, heads, stochastic, false)
}
pub fn bounded_family(kind: &str, known: bool) -> tempfile::TempDir {
    family_with_action_bounds(kind, known, 16, false, true)
}
fn family_with_action_bounds(
    kind: &str,
    known: bool,
    heads: usize,
    stochastic: bool,
    bounded: bool,
) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let space = if kind == "discrete" {
        ActionSpace::Discrete { count: 4 }
    } else if bounded {
        ActionSpace::Box {
            low: vec![-1.; 2],
            high: vec![1.; 2],
        }
    } else {
        ActionSpace::Unbounded { dimensions: 2 }
    };
    let m = RunManifest::new("value-family", "ordinary", kind, 3, space.clone(), 0.9).unwrap();
    let mut w =
        Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
    let mut context = RecordContext::new("value-family", 3);
    context.probe_set = Some("fixed".into());
    let mut o = Observation::new(vec![0.; 3]);
    if kind == "discrete" {
        o.as_dense_mut()
            .expect("dense diagnostic input")
            .action_mask = Some(vec![true, false, true, false]);
    }
    w.append(
        &ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "actual numerical fixture",
            if kind == "discrete" {
                let mut other = Observation::new(vec![0.25, 0.5, 0.75]);
                other
                    .as_dense_mut()
                    .expect("dense diagnostic input")
                    .action_mask = Some(vec![false, true, false, true]);
                vec![o, other]
            } else {
                vec![o]
            },
            None,
            space.clone(),
        )
        .unwrap()
        .record(context)
        .unwrap(),
    )
    .unwrap();
    for rev in [3, 7] {
        let mut rng = ChaCha8Rng::seed_from_u64(rev);
        let mut b = SnapshotBuilder::new(
            "actual",
            RecordContext::new("value-family", rev),
            3,
            space.clone(),
        )
        .unwrap();
        if kind != "discrete" && kind != "state" {
            if stochastic {
                b.policy(
                    "controller",
                    &rl::nn::policy::Policy::new(3, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
            } else {
                let actor = Deterministic::from_network(
                    Mlp::new(3, &[4], 2, &mut rng).unwrap(),
                    space.clone(),
                )
                .unwrap();
                b.deterministic("controller", &actor).unwrap();
            }
        }
        let mut critics = BTreeMap::new();
        let names: Vec<(&str, ValueCriticKind)> = match kind {
            "state" => vec![("baseline", ValueCriticKind::StateValue)],
            "twin" => vec![
                ("omega", ValueCriticKind::ScalarQ),
                ("alpha", ValueCriticKind::ScalarQ),
            ],
            "quantile" => vec![("distribution", ValueCriticKind::QuantileQ)],
            "moments" => vec![("moments", ValueCriticKind::MomentQ)],
            "discrete" => vec![("actions", ValueCriticKind::DiscreteQ)],
            _ => panic!(),
        };
        for (index, (name, k)) in names.iter().enumerate() {
            if kind == "state" {
                b.value(
                    name,
                    &rl::nn::critics::scalar::Value::new(3, &[], &mut rng).unwrap(),
                )
                .unwrap();
            } else if kind == "quantile" {
                b.quantiles(
                    name,
                    &QValue::new(3, 2, heads, &[], &mut rng).unwrap(),
                    &QuantileGrid::uniform(heads).unwrap(),
                )
                .unwrap();
            } else if kind == "moments" {
                let constant = |v: f32, rng: &mut ChaCha8Rng| {
                    let mut q = QValue::new(3, 2, 1, &[], rng).unwrap();
                    q.network.layers[0].weight = q.network.layers[0]
                        .weight
                        .clone()
                        .map(|_| matrix(vec![0.; 5], 5, 1));
                    q.network.layers[0].bias = q.network.layers[0]
                        .bias
                        .take()
                        .map(|p| p.map(|_| vector(&[v])));
                    q
                };
                b.moments(name, &constant(2., &mut rng), &constant(3., &mut rng))
                    .unwrap();
            } else if kind == "discrete" {
                b.discrete(name, &DiscreteQ::new(3, 4, &[], &mut rng).unwrap())
                    .unwrap();
            } else {
                b.scalar(name, &QValue::new(3, 2, 1, &[], &mut rng).unwrap())
                    .unwrap();
            }
            critics.insert(
                (*name).into(),
                CriticDeclaration {
                    kind: *k,
                    learned_baseline: (kind == "state").then_some(true),
                    actor: Some(ContinuationIdentity {
                        identity: "controller".into(),
                        stochastic,
                    }),
                    continuation: Some(ContinuationIdentity {
                        identity: format!("controller-{rev}"),
                        stochastic,
                    }),
                    twin: (kind == "twin").then(|| TwinMembership {
                        group: "declared-pair".into(),
                        index,
                    }),
                    utility: (kind == "moments")
                        .then_some(MomentUtility::MeanVariance { penalty: 2. }),
                },
            );
        }
        let mut chunk = b.finish().unwrap();
        attach_declaration(
            &mut chunk,
            &ValueDeclaration {
                version: 1,
                run_id: m.run_id.clone(),
                revision: rev,
                model_role: "online".into(),
                conventions: known.then(|| ValueConventions {
                    dynamics: "ordinary".into(),
                    observation: "obs".into(),
                    action: "act".into(),
                    reward_units: "reward".into(),
                    reward_convention: m.reward_convention.clone(),
                    discount: 0.9,
                    horizon: "episodic".into(),
                    terminal_treatment: "reward-once".into(),
                    action_constraints: "declared".into(),
                }),
                critics,
            },
        )
        .unwrap();
        w.append(&chunk).unwrap();
    }
    if bounded {
        w.append(&Chunk {
            context: RecordContext::new("value-family", 7),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::from([("reference.fixture".into(), 0.5)]),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        })
        .unwrap();
    }
    if kind != "discrete" && kind != "state" {
        for id in ["episode-a", "episode-b"] {
            let mut context = RecordContext::new("value-family", 3);
            context.trajectory = Some(id.into());
            w.append(&Chunk {
                context,
                record: Record::Trajectory(TrajectoryRecord {
                    render: None,
                    id: id.into(),
                    environment: "ordinary".into(),
                    observations: "o".into(),
                    actions: "a".into(),
                    rewards: "r".into(),
                    terminated: "t".into(),
                    truncated: "x".into(),
                    simulation_times: None,
                    action_masks: None,
                    frames: BTreeMap::new(),
                    evaluation: EvaluationMetadata {
                        conditioning: Conditioning::InitialScenario,
                        initial_action: ActionCondition::CurrentPolicy,
                        continuation_policy: "controller-3".into(),
                        stochastic_policy: false,
                        discount: 0.9,
                        independent_future_streams: true,
                        seed_methodology: "explicit".into(),
                        truncation_handling: "terminal".into(),
                        complete_returns: true,
                    },
                }),
                arrays: BTreeMap::from([
                    (
                        "o".into(),
                        Array::f32(vec![3, 3], vec![0., 0., 0., 0., 0., 0., 999., 999., 999.])
                            .unwrap(),
                    ),
                    ("a".into(), Array::f32(vec![2, 2], vec![0.; 4]).unwrap()),
                    ("r".into(), Array::f32(vec![2], vec![1., 2.]).unwrap()),
                    ("t".into(), Array::u8(vec![2], vec![0, 1]).unwrap()),
                    ("x".into(), Array::u8(vec![2], vec![0, 0]).unwrap()),
                ]),
            })
            .unwrap();
        }
    }
    w.finish().unwrap();
    dir
}

pub fn sparse() -> tempfile::TempDir {
    use rl::core::action::Action;
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = family("quantile", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let bundle = ModelBundle::load(&recording.load(2).unwrap(), ModelLimits::default()).unwrap();
    let result = bundle
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["distribution".into()],
            observations: vec![
                Observation::new(vec![-0.4, 0., 0.]),
                Observation::new(vec![0.4, 0., 0.]),
            ],
            fixed_actions: Some(vec![Action::Continuous(vec![0., 0.]); 2]),
        })
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    w.append(&result.probes).unwrap();
    w.append(&result.critics["distribution"]).unwrap();
    w.finish().unwrap();
    dir
}

/// Typed legacy critics and retained models deliberately share actual masked rows.
pub fn discovery(
    retained: bool,
    declared: bool,
    history: bool,
    samples: bool,
) -> tempfile::TempDir {
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = family("discrete", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let snapshot = recording.load(2).unwrap();
    let model = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
    let mut result = model
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["actions".into()],
            observations: vec![Observation::dense(
                vec![0.; 3],
                Some(vec![true, false, true, false]),
            )],
            fixed_actions: None,
        })
        .unwrap();
    if history {
        if let Record::ProbeSet(p) = &mut result.probes.record {
            p.histories.insert("past".into(), "past".into());
        }
        result.probes.arrays.insert(
            "past".into(),
            Array::f32(vec![1, 2, 3], vec![0.; 6]).unwrap(),
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    w.append(&result.probes).unwrap();
    if retained {
        w.append(&snapshot).unwrap();
    }
    if samples {
        let mut critic = result.critics.remove("actions").unwrap();
        if !declared {
            critic.arrays.retain(|key, _| !key.starts_with("metadata/"));
        }
        // Distinct unnamed records must not silently become one named critic.
        w.append(&critic).unwrap();
        if !declared {
            let Record::Critic(c) = &critic.record else {
                panic!()
            };
            let CriticRepresentation::Discrete { values } = &c.representation else {
                panic!()
            };
            critic.arrays.insert(
                values.clone(),
                Array::f32(vec![1, 4], vec![11., 99., 13., 99.]).unwrap(),
            );
            w.append(&critic).unwrap();
        }
    }
    w.finish().unwrap();
    dir
}

/// Independently sized, genuinely recorded current/reference populations; no model survives.
pub fn sparse_populations(current_rows: usize, reference_rows: usize) -> tempfile::TempDir {
    use rl::core::action::Action;
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    use rl::diagnostics::visualization::*;
    let base = family("quantile", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut manifest = recording.manifest().clone();
    // Both independent populations belong to this declared plotting domain.
    // Without it, automatic bounds use only the current population's extent,
    // and the deliberately disjoint reference rows are correctly out of slice.
    manifest.environment_descriptor = Some(EnvironmentDescriptor {
        version: 1,
        observations: (0..3)
            .map(|i| ObservationField {
                id: format!("obs-{i}"),
                label: format!("obs[{i}]"),
                unit: None,
                description: None,
                kind: ObservationType::Numerical,
                range: Some([-1., 1.]),
                coordinate: true,
            })
            .collect(),
        actions: ActionDescriptor::Continuous {
            components: (0..2)
                .map(|i| ActionComponent {
                    label: format!("action[{i}]"),
                    unit: None,
                    meaning: None,
                })
                .collect(),
            preferred_component: None,
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel {
                label: "k".into(),
                unit: None,
            },
            decision_step: QuantityLabel {
                label: "step".into(),
                unit: None,
            },
            simulation: None,
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: None,
            components: vec![],
        },
        frames: vec![],
        preferences: VisualizationPreferences::default(),
    });
    let mut w =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (index, rows, start) in [(1, reference_rows, -0.8), (2, current_rows, 0.2)] {
        let bundle = ModelBundle::load(
            &recording.load(index).unwrap(),
            ModelLimits {
                max_batch: 8192,
                ..ModelLimits::default()
            },
        )
        .unwrap();
        let result = bundle
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["distribution".into()],
                observations: (0..rows)
                    .map(|i| Observation::new(vec![start + i as f32 * 0.5 / rows as f32, 0., 0.]))
                    .collect(),
                fixed_actions: Some(vec![Action::Continuous(vec![0., 0.]); rows]),
            })
            .unwrap();
        w.append(&result.probes).unwrap();
        w.append(&result.critics["distribution"]).unwrap();
    }
    w.finish().unwrap();
    dir
}

pub fn discrete_populations() -> tempfile::TempDir {
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = family("discrete", true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for (index, states) in [(1, vec![-0.4, 0.5]), (2, vec![0.5])] {
        let model =
            ModelBundle::load(&recording.load(index).unwrap(), ModelLimits::default()).unwrap();
        let result = model
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["actions".into()],
                observations: states
                    .into_iter()
                    .map(|x| {
                        Observation::dense(vec![x, 0., 0.], Some(vec![true, false, true, false]))
                    })
                    .collect(),
                fixed_actions: None,
            })
            .unwrap();
        w.append(&result.probes).unwrap();
        w.append(&result.critics["actions"]).unwrap();
    }
    w.finish().unwrap();
    dir
}

/// Controlled recorded rows for projection/layout tests, not a trained demonstration.
#[allow(dead_code)]
pub fn projection_population(kind: &str, rows: usize, pattern: &str) -> tempfile::TempDir {
    use rl::core::action::Action;
    use rl::diagnostics::models::{CriticInspectionRequest, ModelBundle, ModelLimits};
    let base = family(kind, true);
    let recording = Recording::open(base.path().join("run"), ReadLimits::default()).unwrap();
    let model = ModelBundle::load(
        &recording.load(2).unwrap(),
        ModelLimits {
            max_batch: 8192,
            ..ModelLimits::default()
        },
    )
    .unwrap();
    let discrete = kind == "discrete";
    let name = if discrete { "actions" } else { "distribution" };
    let mut result = model
        .inspect_critics(&CriticInspectionRequest {
            critics: vec![name.into()],
            observations: (0..rows)
                .map(|row| {
                    Observation::dense(
                        vec![
                            if pattern == "overlap" {
                                (row % 2) as f32 * 0.25
                            } else {
                                row as f32 / rows as f32 * 0.5
                            },
                            0.,
                            if pattern == "filtered" && row >= 2 {
                                1.
                            } else {
                                0.
                            },
                        ],
                        discrete.then(|| {
                            if row % 3 == 2 && pattern == "overlap" {
                                vec![false, true, false, true]
                            } else {
                                vec![true, false, true, false]
                            }
                        }),
                    )
                })
                .collect(),
            fixed_actions: (!discrete).then(|| vec![Action::Continuous(vec![0., 0.]); rows]),
        })
        .unwrap();
    if let Record::ProbeSet(p) = &mut result.probes.record {
        p.population = ProbePopulation::ImportedHistory;
        p.histories.insert("past".into(), "past".into());
    }
    result.probes.arrays.insert(
        "past".into(),
        Array::f32(vec![rows, 2, 3], (0..rows * 6).map(|i| i as f32).collect()).unwrap(),
    );
    let critic = result.critics.get_mut(name).unwrap();
    if let Record::Critic(c) = &critic.record
        && let CriticRepresentation::Discrete { values } = &c.representation
    {
        critic.arrays.insert(
            values.clone(),
            Array::f32(
                vec![rows, 4],
                (0..rows * 4).map(|i| i as f32 + 1.).collect(),
            )
            .unwrap(),
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let mut w = Recorder::create(
        dir.path().join("run"),
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    w.append(&result.probes).unwrap();
    w.append(critic).unwrap();
    w.finish().unwrap();
    dir
}
