#![cfg(all(feature = "diagnostics", feature = "neural"))]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        models::*,
        reference::*,
        schema::*,
        storage::*,
        value::{
            self, ContinuationIdentity, CriticDeclaration, ValueConventions, ValueCriticKind,
            ValueDeclaration,
        },
    },
    nn::{
        critics::scalar::{DiscreteQ, Value},
        policy::Policy,
        probes::{Intervention, InterventionValue},
    },
};
use std::collections::BTreeMap;

fn context() -> ReferenceContext {
    ReferenceContext {
        environment: "tiny-mdp".into(),
        observation_dim: 2,
        typed_observation_space: None,
        action_space: ActionSpace::Discrete { count: 2 },
        conventions: ValueConventions {
            dynamics: "tiny-v1".into(),
            observation: "[state, remaining fraction]".into(),
            action: "0 or 1".into(),
            reward_units: "reward".into(),
            reward_convention: "larger better".into(),
            discount: 0.5,
            horizon: "two decisions".into(),
            terminal_treatment: "zero continuation".into(),
            action_constraints: "explicit legal mask".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: BTreeMap::from([
            ("horizon".into(), ConfigValue::Integer(2)),
            ("arrival".into(), ConfigValue::Real(0.25)),
        ]),
        markov_observation: true,
        decision_time_coordinate: Some(1),
    }
}
fn manifest() -> RunManifest {
    let c = context();
    let mut m = RunManifest::new("run", "fixture", "test", 2, c.action_space.clone(), 0.5).unwrap();
    m.environment = c.environment.clone();
    m.reward_convention = c.conventions.reward_convention.clone();
    attach_manifest_context(&mut m, &c).unwrap();
    m
}
fn quality(kind: ReferenceKind) -> ReferenceQuality {
    ReferenceQuality {
        kind,
        method: "independent finite backward induction".into(),
        assumptions: vec!["the declared finite numerical model".into()],
    }
}
fn declared(name: &str, kind: ReferenceKind) -> ReferenceDeclaration {
    ReferenceDeclaration {
        version: 1,
        run_id: "run".into(),
        revision: 7,
        model_role: "online".into(),
        context: context(),
        sources: BTreeMap::from([(name.into(), quality(kind))]),
        finite_table: None,
    }
}
fn policy() -> Chunk {
    Chunk {
        context: {
            let mut c = RecordContext::new("run", 7);
            c.probe_set = Some("inputs".into());
            c
        },
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: Some("m".into()),
            stochastic: false,
        }),
        arrays: BTreeMap::from([
            ("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
            ("m".into(), Array::u8(vec![2, 2], vec![1, 0, 1, 1]).unwrap()),
        ]),
    }
}
fn table() -> (Chunk, Chunk) {
    let mut p = policy();
    let mut d = declared("oracle", ReferenceKind::VerifiedOptimal);
    d.finite_table = Some(FiniteReferenceTable {
        probe_set: "inputs".into(),
        rows: 2,
        coverage: ReferenceCoverage::AllLiveStates,
        kind: ReferenceTableKind::DiscretePolicy,
    });
    attach_declaration(&mut p, &d).unwrap();
    let probes = Chunk {
        context: p.context.clone(),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "inputs".into(),
            population: ProbePopulation::FixedReference,
            methodology: "all live states of tiny fixture".into(),
            observations: "x".into(),
            action_masks: Some("m".into()),
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([
            (
                "x".into(),
                Array::f32(vec![2, 2], vec![0., 1., 1., 0.5]).unwrap(),
            ),
            ("m".into(), Array::u8(vec![2, 2], vec![1, 0, 1, 1]).unwrap()),
            ("ids".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
        ]),
    };
    (p, probes)
}
fn snapshot() -> Chunk {
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let mut b = SnapshotBuilder::new(
        "cp",
        RecordContext::new("run", 7),
        2,
        context().action_space,
    )
    .unwrap();
    b.policy(
        "policy",
        &Policy::new(2, ActionSpace::Discrete { count: 2 }, &[4], &mut rng).unwrap(),
    )
    .unwrap();
    b.value("value", &Value::new(2, &[4], &mut rng).unwrap())
        .unwrap();
    b.discrete("q", &DiscreteQ::new(2, 2, &[4], &mut rng).unwrap())
        .unwrap();
    b.finish().unwrap()
}
#[test]
fn immutable_optional_metadata_and_true_manifest_binding() {
    let mut p = policy();
    assert!(declaration(&p).unwrap().is_none());
    p.validate().unwrap();
    let before = p.arrays.clone();
    let d = declared("rule", ReferenceKind::Heuristic);
    attach_declaration(&mut p, &d).unwrap();
    attach_declaration(&mut p, &d).unwrap();
    assert_eq!(declaration(&p).unwrap(), Some(d.clone()));
    d.validate_manifest(&manifest()).unwrap();
    for (k, v) in before {
        assert_eq!(p.arrays[&k].as_u8().ok(), v.as_u8().ok());
        assert_eq!(p.arrays[&k].as_u64().ok(), v.as_u64().ok());
    }
    let mut other = d.clone();
    other.sources.get_mut("rule").unwrap().kind = ReferenceKind::VerifiedOptimal;
    assert!(attach_declaration(&mut p, &other).is_err());
    for change in [0, 1, 2, 3, 4, 5] {
        let mut m = manifest();
        match change {
            0 => m.discount = 0.9,
            1 => m.environment = "other".into(),
            2 => m.observation_dim = 3,
            3 => m.action_space = ActionSpace::Discrete { count: 3 },
            4 => m.reward_convention = "cost".into(),
            _ => {
                m.metadata.remove(REFERENCE_CONTEXT_METADATA_KEY);
            }
        }
        assert!(d.validate_manifest(&m).is_err());
    }
    let mut m = manifest();
    assert_eq!(manifest_context(&m).unwrap(), Some(context()));
    attach_manifest_context(&mut m, &context()).unwrap();
    let mut c = context();
    c.environment_config
        .insert("arrival".into(), ConfigValue::Real(0.3));
    assert!(attach_manifest_context(&mut m, &c).is_err());
    assert!(d.retained_charge() >= std::mem::size_of::<ReferenceDeclaration>());
}
#[test]
fn malformed_boundaries_and_wrong_subjects_fail_closed() {
    for mutation in 0..10 {
        let mut d = declared("rule", ReferenceKind::Heuristic);
        match mutation {
            0 => d.version = 2,
            1 => d.run_id = "wrong".into(),
            2 => d.revision = 6,
            3 => d.model_role = "other".into(),
            4 => d.sources.clear(),
            5 => d
                .context
                .environment_config
                .insert("arrival".into(), ConfigValue::Real(f64::NAN))
                .map(|_| ())
                .unwrap(),
            6 => d.context.decision_time_coordinate = Some(2),
            7 => d.context.horizon = ReferenceHorizon::Finite { decisions: 3 },
            8 => {
                d.sources.get_mut("rule").unwrap().method = "x".repeat(MAX_REFERENCE_METADATA_BYTES)
            }
            _ => {
                d.context.environment_config.clear();
            }
        }
        assert!(
            attach_declaration(&mut policy(), &d).is_err(),
            "mutation {mutation}"
        );
    }
    for bytes in [
        b"{".to_vec(),
        vec![b' '; MAX_REFERENCE_METADATA_BYTES + 1],
        br#"{"version":999}"#.to_vec(),
    ] {
        let mut p = policy();
        p.arrays.insert(
            REFERENCE_METADATA_KEY.into(),
            Array::u8(vec![bytes.len()], bytes).unwrap(),
        );
        assert!(p.validate().is_err());
    }
    let mut p = policy();
    p.arrays.insert(
        REFERENCE_METADATA_KEY.into(),
        Array::u64(vec![1], vec![1]).unwrap(),
    );
    assert!(p.validate().is_err());
    let mut s = snapshot();
    assert!(attach_declaration(&mut s, &declared("missing", ReferenceKind::Learned)).is_err());
    let mut wrong = declared("policy", ReferenceKind::Learned);
    wrong.context.observation_dim = 3;
    assert!(attach_declaration(&mut s, &wrong).is_err());
    let mut c = context();
    c.environment_config
        .insert("horizon".into(), ConfigValue::Integer(3));
    assert!(compatible_contexts(&context(), &c).is_err());
    for mutate in 0..5 {
        let mut c = context();
        match mutate {
            0 => c.conventions.discount = 0.9,
            1 => c.conventions.terminal_treatment = "bootstrap".into(),
            2 => c.markov_observation = false,
            3 => c.decision_time_coordinate = Some(0),
            _ => {
                c.environment_config
                    .insert("arrival".into(), ConfigValue::Real(0.9));
            }
        }
        assert!(compatible_contexts(&context(), &c).is_err());
    }
}
#[test]
fn finite_table_requires_exact_bound_population_and_masks() {
    let (p, x) = table();
    validate_table(&p, &x).unwrap();
    for mutation in 0..9 {
        let mut bad = x.clone();
        match mutation {
            0 => bad.context.identity.model_role = "wrong".into(),
            1 => bad.context.identity.learner_update_revision = 8,
            2 => bad.context.probe_set = Some("wrong".into()),
            3 => {
                bad.arrays
                    .insert("m".into(), Array::u8(vec![2, 2], vec![1, 1, 1, 1]).unwrap());
            }
            4 => {
                bad.arrays
                    .insert("m".into(), Array::u8(vec![2, 2], vec![0, 0, 1, 1]).unwrap());
            }
            5 => {
                bad.arrays.insert(
                    "x".into(),
                    Array::f64(vec![2, 2], vec![0., 1., 1., 0.5]).unwrap(),
                );
            }
            6 => {
                let Record::ProbeSet(r) = &mut bad.record else {
                    unreachable!()
                };
                r.action_masks = None;
            }
            7 => {
                let Record::ProbeSet(r) = &mut bad.record else {
                    unreachable!()
                };
                r.histories.insert("history".into(), "x".into());
            }
            _ => {
                bad.arrays
                    .insert("x".into(), Array::f32(vec![2, 3], vec![0.; 6]).unwrap());
            }
        }
        assert!(validate_table(&p, &bad).is_err(), "mutation {mutation}");
    }
    let mut d = declaration(&p).unwrap().unwrap();
    d.finite_table.as_mut().unwrap().rows = MAX_REFERENCE_TABLE_ROWS + 1;
    assert!(attach_declaration(&mut policy(), &d).is_err());
    let mut d = declaration(&p).unwrap().unwrap();
    d.sources.get_mut("oracle").unwrap().kind = ReferenceKind::Analytical;
    assert!(attach_declaration(&mut policy(), &d).is_err());
}
#[test]
fn real_model_inspection_projects_subjects_without_changing_outputs() {
    let original = snapshot();
    let obs = vec![
        Observation::new(vec![0., 1.]),
        Observation::new(vec![1., 0.5]),
    ];
    let request = InspectionRequest {
        critics: vec!["value".into()],
        ..InspectionRequest::new("policy", obs.clone())
    };
    let before = ModelBundle::load(&original, ModelLimits::default())
        .unwrap()
        .inspect(&request)
        .unwrap();
    let mut s = original;
    let mut d = declared("policy", ReferenceKind::Learned);
    d.sources
        .insert("value".into(), quality(ReferenceKind::Learned));
    d.sources
        .insert("q".into(), quality(ReferenceKind::Learned));
    attach_declaration(&mut s, &d).unwrap();
    let b = ModelBundle::load(&s, ModelLimits::default()).unwrap();
    assert_eq!(b.reference_declaration(), Some(&d));
    let after = b.inspect(&request).unwrap();
    assert_eq!(before.actions, after.actions);
    assert_eq!(
        before.policy.arrays["probabilities"].as_f32().unwrap(),
        after.policy.arrays["probabilities"].as_f32().unwrap()
    );
    assert_eq!(
        declaration(&after.policy)
            .unwrap()
            .unwrap()
            .sources
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["policy"]
    );
    assert_eq!(
        declaration(&after.critics["value"])
            .unwrap()
            .unwrap()
            .sources
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        vec!["value"]
    );
    let isolated = b
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["value".into()],
            observations: obs.clone(),
            fixed_actions: None,
        })
        .unwrap();
    assert_eq!(
        declaration(&isolated.critics["value"])
            .unwrap()
            .unwrap()
            .sources
            .len(),
        1
    );
    let greedy = b.inspect(&InspectionRequest::new("q", obs)).unwrap();
    assert!(
        declaration(&greedy.policy)
            .unwrap()
            .unwrap()
            .sources
            .contains_key("q")
    );
    let mut intervened = request;
    intervened.interventions.insert(
        "policy".into(),
        vec![Intervention {
            layer: "layers.0.post".into(),
            neuron: 0,
            value: InterventionValue::Set(0.),
        }],
    );
    let result = b.inspect(&intervened).unwrap();
    assert!(declaration(&result.policy).unwrap().is_none());
    assert!(declaration(&result.critics["value"]).unwrap().is_none());
    // Even an empty intervention entry denotes a counterfactual request. This
    // must remain true with missing capture keys and a present empty capture.
    intervened.interventions.insert("policy".into(), vec![]);
    for capture in [false, true] {
        if capture {
            intervened.layers.insert("policy".into(), vec![]);
        }
        let result = b.inspect(&intervened).unwrap();
        assert!(declaration(&result.policy).unwrap().is_none());
        assert!(declaration(&result.critics["value"]).unwrap().is_none());
        assert_eq!(result.actions, before.actions);
    }
    intervened.interventions.clear();
    let restored = b.inspect(&intervened).unwrap();
    assert!(declaration(&restored.policy).unwrap().is_some());
    assert!(declaration(&restored.critics["value"]).unwrap().is_some());
}
#[test]
fn storage_checks_authoritative_manifest_on_append_and_admitted_reads() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("recording");
    let mut p = policy();
    let d = declared("rule", ReferenceKind::Heuristic);
    attach_declaration(&mut p, &d).unwrap();
    let mut recorder = Recorder::create(&root, manifest(), StorageConfig::default()).unwrap();
    recorder.append(&p).unwrap();
    recorder.finish().unwrap();
    let recording = Recording::open(&root, ReadLimits::default()).unwrap();
    assert_eq!(
        declaration(&recording.load(0).unwrap()).unwrap(),
        Some(d.clone())
    );
    assert!(
        recording
            .source(0)
            .unwrap()
            .load_if_within_charge(1024 * 1024)
            .unwrap()
            .is_some()
    );
    let mut changed = manifest();
    let mut c = context();
    c.environment_config
        .insert("arrival".into(), ConfigValue::Real(0.7));
    changed.metadata.insert(
        REFERENCE_CONTEXT_METADATA_KEY.into(),
        serde_json::to_string(&c).unwrap(),
    );
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&changed).unwrap(),
    )
    .unwrap();
    let altered = Recording::open(&root, ReadLimits::default()).unwrap();
    assert!(altered.load(0).is_err());
    assert!(
        altered
            .source(0)
            .unwrap()
            .load_if_within_charge(1024 * 1024)
            .is_err()
    );
    let other = dir.path().join("wrong");
    let mut writer = Recorder::create(other, changed, StorageConfig::default()).unwrap();
    assert!(writer.append(&p).is_err());
}

fn q_table() -> (Chunk, Chunk) {
    let (policy, probes) = table();
    let mut q = Chunk {
        context: policy.context.clone(),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Discrete { values: "q".into() },
            action_condition: ActionCondition::AllDiscreteActions,
            actions: None,
            continuation_policy: Some("optimal".into()),
            continuation_stochastic: Some(false),
        }),
        arrays: BTreeMap::from([(
            "q".into(),
            Array::f64(vec![2, 2], vec![-2., 1000., -5., -3.]).unwrap(),
        )]),
    };
    let v = ValueDeclaration {
        version: 1,
        run_id: "run".into(),
        revision: 7,
        model_role: "online".into(),
        conventions: Some(context().conventions),
        critics: BTreeMap::from([(
            "oracle".into(),
            CriticDeclaration {
                kind: ValueCriticKind::DiscreteQ,
                learned_baseline: None,
                actor: None,
                continuation: Some(ContinuationIdentity {
                    identity: "optimal".into(),
                    stochastic: false,
                }),
                twin: None,
                utility: None,
            },
        )]),
    };
    value::attach_declaration(&mut q, &v).unwrap();
    let mut d = declaration(&policy).unwrap().unwrap();
    d.finite_table.as_mut().unwrap().kind = ReferenceTableKind::OptimalDiscreteActionValues;
    attach_declaration(&mut q, &d).unwrap();
    (q, probes)
}
#[test]
fn optimal_q_table_requires_value_semantics_and_ignores_no_invalid_shape() {
    let (q, probes) = q_table();
    validate_table(&q, &probes).unwrap();
    // The masked finite placeholder 1000 is structurally valid; consumers must
    // exclude it. This layer does not claim to re-prove the Bellman equation.
    for mutation in 0..7 {
        let mut bad = q.clone();
        match mutation {
            0 => {
                bad.arrays.remove(value::VALUE_METADATA_KEY);
            }
            1 => {
                bad.arrays.insert(
                    "q".into(),
                    Array::f64(vec![2, 2], vec![-2., f64::INFINITY, -5., -3.]).unwrap(),
                );
            }
            2 => {
                bad.arrays
                    .insert("q".into(), Array::f64(vec![2, 3], vec![-2.; 6]).unwrap());
            }
            3 => {
                let Record::Critic(r) = &mut bad.record else {
                    unreachable!()
                };
                r.action_condition = ActionCondition::StateValue;
            }
            4 => {
                let Record::Critic(r) = &mut bad.record else {
                    unreachable!()
                };
                r.continuation_policy = Some("other".into());
            }
            5 => {
                bad.arrays.insert(
                    "padding".into(),
                    Array::u8(
                        vec![MAX_REFERENCE_TABLE_ARRAY_BYTES],
                        vec![0; MAX_REFERENCE_TABLE_ARRAY_BYTES],
                    )
                    .unwrap(),
                );
            }
            _ => {
                bad.context.probe_set = None;
            }
        }
        assert!(
            validate_table(&bad, &probes).is_err(),
            "mutation {mutation}"
        );
    }
    let mut raw = q.clone();
    raw.arrays.remove(REFERENCE_METADATA_KEY);
    let original_value = raw.arrays[value::VALUE_METADATA_KEY]
        .as_u8()
        .unwrap()
        .to_vec();
    let mut wrong = declaration(&q).unwrap().unwrap();
    wrong.sources = BTreeMap::from([("other".into(), quality(ReferenceKind::VerifiedOptimal))]);
    assert!(attach_declaration(&mut raw, &wrong).is_err());
    let mut wrong = declaration(&q).unwrap().unwrap();
    wrong.context.conventions.terminal_treatment = "other".into();
    assert!(attach_declaration(&mut raw, &wrong).is_err());
    attach_declaration(&mut raw, &declaration(&q).unwrap().unwrap()).unwrap();
    assert_eq!(
        raw.arrays[value::VALUE_METADATA_KEY].as_u8().unwrap(),
        original_value
    );
}
#[test]
fn metadata_limits_unknown_fields_and_manifest_conflicts_are_not_inferred() {
    let d = declared("rule", ReferenceKind::Heuristic);
    let mut encoded = serde_json::to_value(&d).unwrap();
    encoded["unexpected"] = true.into();
    let mut p = policy();
    let bytes = serde_json::to_vec(&encoded).unwrap();
    p.arrays.insert(
        REFERENCE_METADATA_KEY.into(),
        Array::u8(vec![bytes.len()], bytes).unwrap(),
    );
    assert!(p.validate().is_err());
    for mutation in 0..5 {
        let mut d = d.clone();
        match mutation {
            0 => d.sources.get_mut("rule").unwrap().assumptions = vec!["a".into(); 17],
            1 => {
                d.context.environment_config = (0..513)
                    .map(|i| (format!("k{i}"), ConfigValue::Bool(true)))
                    .collect()
            }
            2 => {
                d.context
                    .environment_config
                    .insert("control".into(), ConfigValue::Text("bad\nvalue".into()));
            }
            3 => {
                d.context.horizon = ReferenceHorizon::Continuing;
            }
            _ => d.context.action_space = ActionSpace::Discrete { count: 65 },
        }
        assert!(d.validate().is_err());
    }
    let mut m = manifest();
    m.metadata.insert(
        REFERENCE_CONTEXT_METADATA_KEY.into(),
        " ".repeat(MAX_REFERENCE_METADATA_BYTES + 1),
    );
    assert!(manifest_context(&m).is_err());
    let mut m = manifest();
    m.metadata
        .insert(REFERENCE_CONTEXT_METADATA_KEY.into(), "{".into());
    assert!(manifest_context(&m).is_err());
    let mut d = d.clone();
    d.finite_table = Some(FiniteReferenceTable {
        probe_set: "inputs".into(),
        rows: 2,
        coverage: ReferenceCoverage::AllLiveStates,
        kind: ReferenceTableKind::DiscretePolicy,
    });
    assert!(d.validate().is_err());
    let mut d = declaration(&table().0).unwrap().unwrap();
    d.context.markov_observation = false;
    assert!(d.validate().is_err());
    let mut d = declaration(&table().0).unwrap().unwrap();
    d.context.decision_time_coordinate = None;
    assert!(d.validate().is_err());
    let mut d = declaration(&table().0).unwrap().unwrap();
    d.context.action_space = ActionSpace::Box {
        low: vec![0.],
        high: vec![1.],
    };
    assert!(d.validate().is_err());
}

fn reference_action_spaces() -> [(ActionSpace, serde_json::Value); 3] {
    [
        (
            ActionSpace::Discrete { count: 2 },
            serde_json::json!({"Discrete": {"count": 2}}),
        ),
        (
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            serde_json::json!({"Box": {"low": [-1.0], "high": [1.0]}}),
        ),
        (
            ActionSpace::Unbounded { dimensions: 1 },
            serde_json::json!({"Unbounded": {"dimensions": 1}}),
        ),
    ]
}

fn reference_action_fixture(space: ActionSpace) -> (ReferenceDeclaration, Chunk, RunManifest) {
    let mut d = declared("rule", ReferenceKind::Heuristic);
    let mut p = policy();
    let mut m = manifest();
    if !matches!(space, ActionSpace::Discrete { .. }) {
        let Record::Policy(record) = &mut p.record else {
            unreachable!()
        };
        record.masks = None;
        p.arrays.remove("m");
        p.arrays
            .insert("a".into(), Array::f32(vec![2, 1], vec![0., 0.5]).unwrap());
    }
    d.context.action_space = space.clone();
    m.action_space = space;
    m.metadata.remove(REFERENCE_CONTEXT_METADATA_KEY);
    attach_manifest_context(&mut m, &d.context).unwrap();
    attach_declaration(&mut p, &d).unwrap();
    (d, p, m)
}

fn add_unknown_action_field(action: &mut serde_json::Value) {
    let variant = action.as_object_mut().unwrap().values_mut().next().unwrap();
    variant["unexpected"] = true.into();
}

#[test]
fn reference_actions_reject_nested_unknown_chunk_fields() {
    let mut results = Vec::new();
    for (space, _) in reference_action_spaces() {
        let (d, mut p, _) = reference_action_fixture(space.clone());
        let mut encoded = serde_json::to_value(&d).unwrap();
        add_unknown_action_field(&mut encoded["context"]["action_space"]);
        let bytes = serde_json::to_vec(&encoded).unwrap();
        p.arrays.insert(
            REFERENCE_METADATA_KEY.into(),
            Array::u8(vec![bytes.len()], bytes).unwrap(),
        );
        results.push((space, declaration(&p).is_err(), p.validate().is_err()));
    }
    assert!(
        results
            .iter()
            .all(|(_, parsed, validated)| *parsed && *validated),
        "{results:?}"
    );
}

#[test]
fn reference_actions_reject_nested_unknown_manifest_fields() {
    let mut results = Vec::new();
    for (space, _) in reference_action_spaces() {
        let (d, _, mut m) = reference_action_fixture(space.clone());
        let mut encoded = serde_json::to_value(&d.context).unwrap();
        add_unknown_action_field(&mut encoded["action_space"]);
        m.metadata.insert(
            REFERENCE_CONTEXT_METADATA_KEY.into(),
            serde_json::to_string(&encoded).unwrap(),
        );
        results.push((
            space,
            manifest_context(&m).is_err(),
            d.validate_manifest(&m).is_err(),
        ));
    }
    assert!(
        results
            .iter()
            .all(|(_, parsed, validated)| *parsed && *validated),
        "{results:?}"
    );
}

#[test]
fn reference_action_wire_roundtrips_keep_legacy_action_spaces_permissive() {
    for (space, wire) in reference_action_spaces() {
        let (d, p, mut m) = reference_action_fixture(space.clone());
        let encoded = serde_json::to_value(&d.context).unwrap();
        assert_eq!(encoded["action_space"], wire);
        assert_eq!(
            serde_json::from_value::<ReferenceContext>(encoded).unwrap(),
            d.context
        );
        assert_eq!(manifest_context(&m).unwrap(), Some(d.context.clone()));
        assert_eq!(declaration(&p).unwrap(), Some(d.clone()));
        d.validate_manifest(&m).unwrap();

        // Reference strictness must not change ordinary/legacy ActionSpace decoding.
        let mut extended = wire;
        add_unknown_action_field(&mut extended);
        assert_eq!(
            serde_json::from_value::<ActionSpace>(extended.clone()).unwrap(),
            space
        );
        m.metadata.remove(REFERENCE_CONTEXT_METADATA_KEY);
        let mut legacy = serde_json::to_value(&m).unwrap();
        legacy["action_space"] = extended;
        let decoded: RunManifest = serde_json::from_value(legacy).unwrap();
        decoded.validate().unwrap();
        assert_eq!(
            serde_json::to_value(&decoded).unwrap(),
            serde_json::to_value(&m).unwrap()
        );
        assert!(manifest_context(&decoded).unwrap().is_none());
    }
}

#[test]
fn maximum_finite_rows_stay_a_bounded_structural_validation() {
    let n = MAX_REFERENCE_TABLE_ROWS;
    let mut p = policy();
    p.arrays
        .insert("a".into(), Array::u64(vec![n], vec![0; n]).unwrap());
    p.arrays
        .insert("m".into(), Array::u8(vec![n, 2], vec![1; n * 2]).unwrap());
    let mut d = declared("oracle", ReferenceKind::VerifiedOptimal);
    d.finite_table = Some(FiniteReferenceTable {
        probe_set: "inputs".into(),
        rows: n,
        coverage: ReferenceCoverage::ListedStates,
        kind: ReferenceTableKind::DiscretePolicy,
    });
    attach_declaration(&mut p, &d).unwrap();
    let (_, mut probes) = table();
    probes.arrays.insert(
        "x".into(),
        Array::f32(vec![n, 2], (0..n).flat_map(|i| [i as f32, 1.]).collect()).unwrap(),
    );
    probes
        .arrays
        .insert("m".into(), Array::u8(vec![n, 2], vec![1; n * 2]).unwrap());
    probes.arrays.insert(
        "ids".into(),
        Array::u64(vec![n], (0..n as u64).collect()).unwrap(),
    );
    validate_table(&p, &probes).unwrap();
    assert!(p.array_bytes() + probes.array_bytes() < MAX_REFERENCE_TABLE_ARRAY_BYTES);
}
#[test]
fn attaching_metadata_is_transactional_at_the_table_payload_cap() {
    let (source, _) = table();
    let d = declaration(&source).unwrap().unwrap();
    let mut raw = policy();
    let before = raw.array_bytes();
    raw.arrays.insert(
        "padding".into(),
        Array::u8(
            vec![MAX_REFERENCE_TABLE_ARRAY_BYTES - before - 1],
            vec![0; MAX_REFERENCE_TABLE_ARRAY_BYTES - before - 1],
        )
        .unwrap(),
    );
    assert!(attach_declaration(&mut raw, &d).is_err());
    assert!(!raw.arrays.contains_key(REFERENCE_METADATA_KEY));
    raw.validate().unwrap();
    let encoded = serde_json::to_vec(&d).unwrap().len();
    let mut exact = policy();
    let padding = MAX_REFERENCE_TABLE_ARRAY_BYTES - exact.array_bytes() - encoded;
    exact.arrays.insert(
        "padding".into(),
        Array::u8(vec![padding], vec![0; padding]).unwrap(),
    );
    attach_declaration(&mut exact, &d).unwrap();
    assert_eq!(exact.array_bytes(), MAX_REFERENCE_TABLE_ARRAY_BYTES);
    exact.validate().unwrap();
}
#[test]
fn labelled_continuous_actions_must_respect_authored_box() {
    let mut d = declared("controller", ReferenceKind::Heuristic);
    d.context.action_space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let make = |a: f32| Chunk {
        context: RecordContext::new("run", 7),
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        arrays: BTreeMap::from([("a".into(), Array::f32(vec![1, 1], vec![a]).unwrap())]),
    };
    attach_declaration(&mut make(1.), &d).unwrap();
    assert!(attach_declaration(&mut make(1.1), &d).is_err());
    assert!(attach_declaration(&mut make(-1.1), &d).is_err());
}
