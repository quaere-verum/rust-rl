#![cfg(all(feature = "diagnostics", feature = "neural"))]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{analysis::QuantileForecast, models::*, schema::*, value::*},
    nn::critics::{
        quantile::QuantileGrid,
        scalar::{DiscreteQ, QValue, Value},
    },
};
use std::collections::BTreeMap;

fn snapshot() -> Chunk {
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let mut builder = SnapshotBuilder::new(
        "checkpoint",
        RecordContext::new("run", 7),
        2,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    )
    .unwrap();
    builder
        .value("v", &Value::new(2, &[4], &mut rng).unwrap())
        .unwrap();
    builder
        .scalar("q1", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
        .unwrap();
    builder
        .scalar("q2", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
        .unwrap();
    builder
        .quantiles(
            "z",
            &QValue::new(2, 1, 3, &[4], &mut rng).unwrap(),
            &QuantileGrid::uniform(3).unwrap(),
        )
        .unwrap();
    builder
        .moments(
            "m",
            &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
            &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        )
        .unwrap();
    builder
        .opaque(
            "custom",
            &Value::new(2, &[4], &mut rng).unwrap(),
            "external",
        )
        .unwrap();
    builder.finish().unwrap()
}
fn named(kind: ValueCriticKind) -> CriticDeclaration {
    CriticDeclaration {
        kind,
        learned_baseline: None,
        actor: None,
        continuation: Some(ContinuationIdentity {
            identity: "controller-a".into(),
            stochastic: false,
        }),
        twin: None,
        utility: None,
    }
}
fn declared() -> ValueDeclaration {
    ValueDeclaration {
        version: 1,
        run_id: "run".into(),
        revision: 7,
        model_role: "online".into(),
        conventions: Some(ValueConventions {
            dynamics: "env-v1".into(),
            observation: "state-f32".into(),
            action: "torque".into(),
            reward_units: "reward".into(),
            reward_convention: "larger better".into(),
            discount: 0.5,
            horizon: "three steps".into(),
            terminal_treatment: "last reward only".into(),
            action_constraints: "box[-1,1]".into(),
        }),
        critics: BTreeMap::from([("v".into(), named(ValueCriticKind::StateValue))]),
    }
}
#[test]
fn legacy_without_declaration_is_unknown() {
    let chunk = snapshot();
    chunk.validate().unwrap();
    assert!(declaration(&chunk).unwrap().is_none());
}

#[test]
fn batched_moment_utility_uses_actual_learner_precision_and_admits_before_allocation() {
    let utility = MomentUtility::MeanVariance { penalty: 2. };
    assert_eq!(
        moment_utilities(&[2., 3.], &[3., 13.], &utility).unwrap(),
        vec![2., -5.]
    );
    assert_eq!(
        moment_utilities(&[2., 3.], &[3., 13.], &MomentUtility::ExpectedReturn).unwrap(),
        vec![2., 3.]
    );
    for (means, seconds) in [
        (vec![], vec![]),
        (vec![1.], vec![]),
        (vec![f64::NAN], vec![1.]),
        (vec![1.], vec![f64::INFINITY]),
        (vec![f64::MAX], vec![1.]),
        (vec![1.; 1025], vec![2.; 1025]),
    ] {
        assert!(moment_utilities(&means, &seconds, &utility).is_err());
    }
    assert!(
        moment_utilities(
            &[1.],
            &[2.],
            &MomentUtility::Unsupported {
                name: "custom".into()
            }
        )
        .is_err()
    );
    assert_eq!(
        moment_utilities(&vec![1.; 1024], &vec![2.; 1024], &utility)
            .unwrap()
            .len(),
        1024
    );
    let actual = moment_utilities(&[0.123456789], &[0.234567891], &utility).unwrap()[0];
    assert_eq!(
        moment_utilities(
            &[0.123456789],
            &[0.234567891],
            &MomentUtility::ExpectedReturn
        )
        .unwrap()[0],
        f64::from(0.123456789_f64 as f32)
    );
    assert_eq!(
        actual,
        ValuePrediction::Moments {
            mean: 0.123456789,
            raw_second: 0.234567891
        }
        .scalar(&ValueOutput::Utility, Some(&utility))
        .unwrap()
    );
}

#[test]
fn direct_head_and_runtime_batch_boundaries() {
    for n in [4096, 4097] {
        let p = ValuePrediction::Quantiles {
            raw: vec![1.; n],
            locations: (0..n).map(|i| (i as f64 + 0.5) / n as f64).collect(),
            masses: vec![1. / n as f64; n],
        };
        assert_eq!(p.ordered_quantiles().is_ok(), n == 4096);
        let ValuePrediction::Quantiles {
            raw,
            locations,
            masses,
        } = p
        else {
            unreachable!()
        };
        let chunk = Chunk {
            context: RecordContext::new("run", 7),
            record: Record::Critic(CriticRecord {
                representation: CriticRepresentation::Quantiles {
                    values: "z".into(),
                    locations: "p".into(),
                    masses: "m".into(),
                },
                action_condition: ActionCondition::Fixed,
                actions: Some("a".into()),
                continuation_policy: None,
                continuation_stochastic: None,
            }),
            arrays: BTreeMap::from([
                ("z".into(), Array::f64(vec![1, n], raw).unwrap()),
                ("p".into(), Array::f64(vec![n], locations).unwrap()),
                ("m".into(), Array::f64(vec![n], masses).unwrap()),
                ("a".into(), Array::f32(vec![1, 1], vec![0.]).unwrap()),
            ]),
        };
        assert_eq!(
            ValuePrediction::from_critic(&chunk, 0, None).is_ok(),
            n == 4096
        );
    }
    let bundle = ModelBundle::load(
        &snapshot(),
        ModelLimits {
            max_batch: 2,
            ..ModelLimits::default()
        },
    )
    .unwrap();
    for n in [0, 2, 3] {
        let r = CriticInspectionRequest {
            critics: vec!["v".into()],
            observations: vec![Observation::new(vec![0., 1.]); n],
            fixed_actions: None,
        };
        assert_eq!(bundle.inspect_critics(&r).is_ok(), n == 2);
    }
}
#[test]
fn declaration_roundtrip_binds_context_and_kind() {
    let mut chunk = snapshot();
    let declaration_value = declared();
    attach_declaration(&mut chunk, &declaration_value).unwrap();
    assert_eq!(declaration(&chunk).unwrap().unwrap(), declaration_value);
    for edit in [0, 1, 2, 3, 4, 5] {
        let mut d = declared();
        match edit {
            0 => d.run_id = "wrong".into(),
            1 => d.revision += 1,
            2 => d.model_role = "target".into(),
            3 => {
                let v = d.critics.remove("v").unwrap();
                d.critics.insert("absent".into(), v);
            }
            4 => d.critics.get_mut("v").unwrap().kind = ValueCriticKind::ScalarQ,
            _ => d.version = 2,
        }
        assert!(attach_declaration(&mut chunk, &d).is_err(), "edit {edit}");
    }
    let result = ModelBundle::load(&chunk, ModelLimits::default())
        .unwrap()
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["v".into()],
            observations: vec![Observation::new(vec![0., 0.])],
            fixed_actions: None,
        })
        .unwrap();
    let mut probe = result.critics["v"].clone();
    attach_declaration(&mut probe, &declared()).unwrap();
    let mut wrong = declared();
    wrong.critics.get_mut("v").unwrap().kind = ValueCriticKind::DiscreteQ;
    assert!(attach_declaration(&mut probe, &wrong).is_err());
}
#[test]
fn declaration_limit_precedes_parse() {
    let mut chunk = snapshot();
    for bytes in [vec![b'x'; 16385], b"{broken".to_vec()] {
        chunk.arrays.insert(
            "metadata/value.v1".into(),
            Array::u8(vec![bytes.len()], bytes).unwrap(),
        );
        assert!(declaration(&chunk).is_err());
    }
    chunk.arrays.insert(
        "metadata/value.v1".into(),
        Array::u8(vec![16385], vec![b'x'; 16385]).unwrap(),
    );
    assert!(
        matches!(declaration(&chunk), Err(rl::Error::Config(_))),
        "admission must reject before JSON decoding"
    );
    chunk.arrays.insert(
        "metadata/value.v1".into(),
        Array::f32(vec![1], vec![1.]).unwrap(),
    );
    assert!(declaration(&chunk).is_err());
    chunk.arrays.insert(
        "metadata/value.v1".into(),
        Array::u8(vec![1, 2], b"{}".to_vec()).unwrap(),
    );
    assert!(declaration(&chunk).is_err());
}
#[test]
fn critic_only_has_no_policy_dependency() {
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    for q in [false, true] {
        let mut b = SnapshotBuilder::new(
            "only",
            RecordContext::new("run", 7),
            2,
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
        )
        .unwrap();
        if q {
            b.scalar("only", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
                .unwrap();
        } else {
            b.value("only", &Value::new(2, &[4], &mut rng).unwrap())
                .unwrap();
        }
        let bundle = ModelBundle::load(&b.finish().unwrap(), ModelLimits::default()).unwrap();
        let mut request = CriticInspectionRequest {
            critics: vec!["only".into()],
            observations: vec![Observation::new(vec![0., 1.])],
            fixed_actions: None,
        };
        assert_eq!(bundle.inspect_critics(&request).is_ok(), !q);
        request.fixed_actions = Some(vec![Action::Continuous(vec![0.5])]);
        let result = bundle.inspect_critics(&request).unwrap();
        assert_eq!(result.critics["only"].context.identity.model_role, "online");
        request.fixed_actions = Some(vec![Action::Continuous(vec![2.])]);
        assert!(bundle.inspect_critics(&request).is_err());
        request.fixed_actions = Some(vec![]);
        assert!(bundle.inspect_critics(&request).is_err());
    }
}
#[test]
fn critic_inspection_preserves_parameters_and_rng() {
    let chunk = snapshot();
    let bytes: Vec<_> = chunk
        .arrays
        .values()
        .map(|a| a.as_u8().unwrap().to_vec())
        .collect();
    let bundle = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
    let mut r = CriticInspectionRequest {
        critics: vec!["v".into(), "q1".into(), "q2".into(), "z".into(), "m".into()],
        observations: vec![
            Observation::new(vec![0.2, 0.4]),
            Observation::new(vec![0.4, 0.8]),
        ],
        fixed_actions: Some(vec![
            Action::Continuous(vec![0.3]),
            Action::Continuous(vec![0.7]),
        ]),
    };
    let first = bundle.inspect_critics(&r).unwrap();
    let second = bundle.inspect_critics(&r).unwrap();
    for name in &r.critics {
        for (key, a) in &first.critics[name].arrays {
            assert_eq!(a.to_f64(), second.critics[name].arrays[key].to_f64());
        }
    }
    for name in ["q1", "q2"] {
        assert_eq!(
            first.critics[name].arrays["actions"].as_f32().unwrap(),
            &[0.3, 0.7]
        );
    }
    assert_eq!(
        bytes,
        chunk
            .arrays
            .values()
            .map(|a| a.as_u8().unwrap().to_vec())
            .collect::<Vec<_>>()
    );
    r.critics = vec!["absent".into()];
    assert!(bundle.inspect_critics(&r).is_err());
    r.critics = vec!["custom".into()];
    assert!(bundle.inspect_critics(&r).is_err());
    r.critics = vec!["q1".into(), "q1".into()];
    assert!(bundle.inspect_critics(&r).is_err());
}
#[test]
fn discrete_requires_legal_selection_before_scalar_output() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut b = SnapshotBuilder::new(
        "discrete",
        RecordContext::new("run", 7),
        2,
        ActionSpace::Discrete { count: 3 },
    )
    .unwrap();
    b.discrete("q", &DiscreteQ::new(2, 3, &[4], &mut rng).unwrap())
        .unwrap();
    let bundle = ModelBundle::load(&b.finish().unwrap(), ModelLimits::default()).unwrap();
    let r = CriticInspectionRequest {
        critics: vec!["q".into()],
        observations: vec![Observation::dense(
            vec![0., 1.],
            Some(vec![true, false, true]),
        )],
        fixed_actions: None,
    };
    assert!(bundle.inspect_critics(&r).is_ok());
    let p = ValuePrediction::Discrete {
        values: vec![1., 99., 3.],
        legal: vec![true, false, true],
    };
    assert!(p.scalar(&ValueOutput::ExpectedReturn, None).is_err());
    assert!(p.select_discrete(1).is_err());
    assert_eq!(
        p.select_discrete(0)
            .unwrap()
            .scalar(&ValueOutput::ExpectedReturn, None)
            .unwrap(),
        1.
    );
    assert_eq!(
        p.greedy_discrete()
            .unwrap()
            .scalar(&ValueOutput::ExpectedReturn, None)
            .unwrap(),
        3.
    );
}
#[test]
fn ordered_quantiles_retain_raw_evidence_and_weighted_atoms() {
    let raw = vec![4., -2., 1.];
    let f =
        QuantileForecast::ordered(raw.clone(), vec![0.1, 0.35, 0.75], vec![0.2, 0.3, 0.5]).unwrap();
    assert_eq!(f.values(), &[-2., 1., 4.]);
    assert_eq!(f.source_crossing_count(), 1);
    assert!((f.mean() - 1.9).abs() < 1e-12);
    for (p, expected) in [(0., -2.), (0.2, -2.), (0.21, 1.), (0.5, 1.), (1., 4.)] {
        assert_eq!(f.quantile(p).unwrap(), expected);
    }
    assert!((f.lower_cvar(0.4).unwrap() + 0.5).abs() < 1e-12);
    assert_eq!(
        f.cdf_points().unwrap(),
        vec![
            [-2., 0.],
            [-2., 0.2],
            [1., 0.2],
            [1., 0.5],
            [4., 0.5],
            [4., 1.]
        ]
    );
    for p in [-0.1, 1.1, f64::NAN] {
        assert!(f.quantile(p).is_err());
    }
    let p = ValuePrediction::Quantiles {
        raw: raw.clone(),
        locations: vec![0.1, 0.35, 0.75],
        masses: vec![0.2, 0.3, 0.5],
    };
    assert_eq!(
        p.scalar(&ValueOutput::Quantile { probability: 0.2 }, None)
            .unwrap(),
        -2.
    );
    assert!(
        (p.scalar(&ValueOutput::Cvar { probability: 0.4 }, None)
            .unwrap()
            + 0.5)
            .abs()
            < 1e-12
    );
    if let ValuePrediction::Quantiles { raw: actual, .. } = p {
        assert_eq!(actual, raw);
    }
    let ties =
        QuantileForecast::ordered(vec![1., 1., 2.], vec![0.1, 0.35, 0.75], vec![0.2, 0.3, 0.5])
            .unwrap();
    assert_eq!(ties.quantile(0.5).unwrap(), 1.);
    let tiny_last_mass =
        QuantileForecast::ordered(vec![1., 2.], vec![0.25, 0.75], vec![1., 1e-20]).unwrap();
    assert_eq!(tiny_last_mass.quantile(1.).unwrap(), 2.);
}
#[test]
fn twins_and_moments_preserve_scientific_evidence() {
    let p = ValuePrediction::Twin {
        q1: Box::new(ValuePrediction::Scalar(3.)),
        q2: Box::new(ValuePrediction::Scalar(-1.)),
    };
    for (output, expected) in [
        (ValueOutput::Q1, 3.),
        (ValueOutput::Q2, -1.),
        (ValueOutput::Minimum, -1.),
        (ValueOutput::Disagreement, 4.),
    ] {
        assert_eq!(p.scalar(&output, None).unwrap(), expected);
    }
    let p = ValuePrediction::Moments {
        mean: 2.,
        raw_second: 3.,
    };
    let summary = p.moment_summary().unwrap();
    assert_eq!(summary.raw_second, 3.);
    assert_eq!(summary.variance, -1.);
    assert_eq!(summary.standard_deviation, None);
    assert_eq!(p.scalar(&ValueOutput::Variance, None).unwrap(), -1.);
    assert_eq!(
        p.scalar(
            &ValueOutput::Utility,
            Some(&MomentUtility::MeanVariance { penalty: 2. })
        )
        .unwrap(),
        2.
    );
    assert!(p.scalar(&ValueOutput::Utility, None).is_err());
    assert!(
        p.scalar(&ValueOutput::Quantile { probability: 0.5 }, None)
            .is_err()
    );
    assert!(
        ValuePrediction::Scalar(f64::NAN)
            .scalar(&ValueOutput::ExpectedReturn, None)
            .is_err()
    );
}
#[test]
fn compatibility_requires_known_conventions_and_statistic_meaning() {
    let current = declared();
    let mut reference = current.clone();
    reference
        .critics
        .get_mut("v")
        .unwrap()
        .continuation
        .as_mut()
        .unwrap()
        .identity = "controller-b".into();
    compatible_values(
        &current,
        "v",
        &reference,
        "v",
        &ValueOutput::ExpectedReturn,
        ComparisonCondition::SameInput,
    )
    .unwrap();
    for field in 0..6 {
        let mut reference = reference.clone();
        match field {
            0 => reference.conventions = None,
            1 => reference.conventions.as_mut().unwrap().discount = 0.9,
            2 => reference.conventions.as_mut().unwrap().horizon = "forever".into(),
            3 => reference.conventions.as_mut().unwrap().reward_units = "dollars".into(),
            4 => reference.conventions.as_mut().unwrap().action_constraints = "unbounded".into(),
            _ => reference.critics.get_mut("v").unwrap().continuation = None,
        }
        assert!(
            compatible_values(
                &current,
                "v",
                &reference,
                "v",
                &ValueOutput::ExpectedReturn,
                ComparisonCondition::SameInput
            )
            .is_err()
        );
    }
    reference.critics.get_mut("v").unwrap().kind = ValueCriticKind::ScalarQ;
    assert!(
        compatible_values(
            &current,
            "v",
            &reference,
            "v",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
    assert!(
        compatible_values(
            &current,
            "v",
            &reference,
            "v",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::ControllerValues
        )
        .is_ok()
    );
    assert!(
        compatible_values(
            &current,
            "v",
            &reference,
            "v",
            &ValueOutput::Variance,
            ComparisonCondition::ControllerValues
        )
        .is_err()
    );
    let mut unknown = current.clone();
    unknown.critics.clear();
    assert!(
        compatible_values(
            &current,
            "v",
            &unknown,
            "v",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
}
fn trajectory() -> Chunk {
    Chunk {
        context: RecordContext::new("run", 7),
        record: Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode".into(),
            environment: "test".into(),
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
                continuation_policy: "controller".into(),
                stochastic_policy: false,
                discount: 0.5,
                independent_future_streams: true,
                seed_methodology: "fixed".into(),
                truncation_handling: "no bootstrap".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "o".into(),
                Array::f64(vec![4, 1], vec![0., 1., 2., 3.]).unwrap(),
            ),
            (
                "a".into(),
                Array::f64(vec![3, 1], vec![0., 0., 0.]).unwrap(),
            ),
            ("r".into(), Array::f64(vec![3], vec![1., 2., 4.]).unwrap()),
            ("t".into(), Array::u8(vec![3], vec![0, 0, 1]).unwrap()),
            ("x".into(), Array::u8(vec![3], vec![0, 0, 0]).unwrap()),
        ]),
    }
}
#[test]
fn complete_return_uses_actual_discount_and_terminal_reward_once() {
    let mut c = trajectory();
    for (i, expected) in [3., 4., 4.].into_iter().enumerate() {
        assert_eq!(trajectory_return_to_go(&c, i).unwrap(), Some(expected));
    }
    if let Record::Trajectory(t) = &mut c.record {
        t.evaluation.discount = 0.;
    }
    assert_eq!(trajectory_return_to_go(&c, 0).unwrap(), Some(1.));
    if let Record::Trajectory(t) = &mut c.record {
        t.evaluation.discount = 1.;
    }
    assert_eq!(trajectory_return_to_go(&c, 0).unwrap(), Some(7.));
    assert!(trajectory_return_to_go(&c, 3).is_err());
    if let Record::Trajectory(t) = &mut c.record {
        t.evaluation.complete_returns = false;
    }
    assert_eq!(trajectory_return_to_go(&c, 0).unwrap(), None);
    c.arrays
        .insert("t".into(), Array::u8(vec![3], vec![0, 0, 0]).unwrap());
    c.arrays
        .insert("x".into(), Array::u8(vec![3], vec![0, 0, 1]).unwrap());
    assert_eq!(trajectory_return_to_go(&c, 0).unwrap(), None);
    c.arrays.insert(
        "r".into(),
        Array::f64(vec![3], vec![1., f64::NAN, 4.]).unwrap(),
    );
    assert!(trajectory_return_to_go(&c, 0).is_err());
}

#[test]
fn declarations_are_immutable_and_disabled_baselines_never_predict() {
    let mut c = snapshot();
    let d = declared();
    attach_declaration(&mut c, &d).unwrap();
    let mut changed = d.clone();
    changed.conventions.as_mut().unwrap().discount = 0.75;
    assert!(attach_declaration(&mut c, &changed).is_err());
    let mut c = snapshot();
    let mut d = declared();
    d.critics.get_mut("v").unwrap().learned_baseline = Some(false);
    attach_declaration(&mut c, &d).unwrap();
    let bundle = ModelBundle::load(&c, ModelLimits::default()).unwrap();
    assert!(
        bundle
            .inspect_critics(&CriticInspectionRequest {
                critics: vec!["v".into()],
                observations: vec![Observation::new(vec![0., 0.])],
                fixed_actions: None
            })
            .is_err()
    );
    let mut policy_only = declared();
    policy_only.critics.clear();
    assert!(
        compatible_values(
            &policy_only,
            "v",
            &d,
            "v",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
}
#[test]
fn selected_critic_compatibility_ignores_unrelated_critics_and_checks_utilities() {
    let current = declared();
    let mut reference = current.clone();
    reference
        .critics
        .insert("unrelated".into(), named(ValueCriticKind::MomentQ));
    compatible_values(
        &current,
        "v",
        &reference,
        "v",
        &ValueOutput::ExpectedReturn,
        ComparisonCondition::SameInput,
    )
    .unwrap();
    assert!(
        compatible_values(
            &current,
            "v",
            &reference,
            "missing",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
    let mut current = declared();
    let mut reference = declared();
    for d in [&mut current, &mut reference] {
        d.critics.get_mut("v").unwrap().kind = ValueCriticKind::MomentQ;
        d.critics.get_mut("v").unwrap().utility = Some(MomentUtility::MeanVariance { penalty: 2. });
    }
    compatible_values(
        &current,
        "v",
        &reference,
        "v",
        &ValueOutput::Utility,
        ComparisonCondition::SameInput,
    )
    .unwrap();
    reference.critics.get_mut("v").unwrap().utility =
        Some(MomentUtility::MeanVariance { penalty: 1. });
    assert!(
        compatible_values(
            &current,
            "v",
            &reference,
            "v",
            &ValueOutput::Utility,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
    reference.critics.get_mut("v").unwrap().utility = Some(MomentUtility::Unsupported {
        name: "custom".into(),
    });
    assert!(
        compatible_values(
            &reference,
            "v",
            &reference,
            "v",
            &ValueOutput::Utility,
            ComparisonCondition::SameInput
        )
        .is_err()
    );
    let mut scalar = declared();
    scalar.critics.get_mut("v").unwrap().kind = ValueCriticKind::ScalarQ;
    for kind in [ValueCriticKind::QuantileQ, ValueCriticKind::MomentQ] {
        let mut other = scalar.clone();
        other.critics.get_mut("v").unwrap().kind = kind;
        compatible_values(
            &scalar,
            "v",
            &other,
            "v",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput,
        )
        .unwrap();
        assert!(
            compatible_values(
                &scalar,
                "v",
                &other,
                "v",
                &ValueOutput::Quantile { probability: 0.5 },
                ComparisonCondition::SameInput
            )
            .is_err()
        );
    }
}

#[test]
fn declarations_validate_manifest_strings_and_ordered_twins() {
    let mut declaration_value = declared();
    let mut manifest = RunManifest::new(
        "run",
        "exp",
        "irrelevant algorithm",
        2,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        0.5,
    )
    .unwrap();
    manifest.reward_convention = "larger better".into();
    declaration_value.validate_manifest(&manifest).unwrap();
    manifest.discount = 0.9;
    assert!(declaration_value.validate_manifest(&manifest).is_err());
    declaration_value.conventions.as_mut().unwrap().dynamics = " ".into();
    assert!(declaration_value.validate().is_err());
    declaration_value.conventions.as_mut().unwrap().dynamics = "x".repeat(1025);
    assert!(declaration_value.validate().is_err());
    let mut d = declared();
    d.critics.clear();
    let mut first = named(ValueCriticKind::ScalarQ);
    first.twin = Some(TwinMembership {
        group: "td3-online".into(),
        index: 0,
    });
    let mut second = first.clone();
    second.twin.as_mut().unwrap().index = 1;
    d.critics.insert("q1".into(), first);
    d.critics.insert("q2".into(), second);
    let mut chunk = snapshot();
    attach_declaration(&mut chunk, &d).unwrap();
    let bundle = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
    let result = bundle
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["q1".into(), "q2".into()],
            observations: vec![Observation::new(vec![0., 1.])],
            fixed_actions: Some(vec![Action::Continuous(vec![0.5])]),
        })
        .unwrap();
    for (index, name) in ["q1", "q2"].into_iter().enumerate() {
        let actual = declaration(&result.critics[name]).unwrap().unwrap();
        assert_eq!(actual.critics[name].twin.as_ref().unwrap().index, index);
        assert_eq!(actual.critics.len(), 1);
        assert_eq!(
            result.critics[name].context.identity,
            chunk.context.identity
        );
    }
    d.critics
        .get_mut("q2")
        .unwrap()
        .twin
        .as_mut()
        .unwrap()
        .index = 0;
    assert!(attach_declaration(&mut snapshot(), &d).is_err());
    d.critics.remove("q2");
    assert!(attach_declaration(&mut snapshot(), &d).is_err());
    let mut d = declared();
    d.critics.get_mut("v").unwrap().kind = ValueCriticKind::MomentQ;
    d.critics.get_mut("v").unwrap().utility = Some(MomentUtility::MeanVariance { penalty: -1. });
    assert!(d.validate().is_err());
}
#[test]
fn prediction_extraction_and_nonfinite_overflow_are_explicit() {
    let mut r = CriticInspectionRequest {
        critics: vec!["v".into(), "z".into(), "m".into()],
        observations: vec![Observation::new(vec![0., 0.])],
        fixed_actions: Some(vec![Action::Continuous(vec![0.])]),
    };
    let bundle = ModelBundle::load(&snapshot(), ModelLimits::default()).unwrap();
    let result = bundle.inspect_critics(&r).unwrap();
    for name in ["v", "z", "m"] {
        assert!(
            ValuePrediction::from_critic(&result.critics[name], 0, None)
                .unwrap()
                .scalar(&ValueOutput::ExpectedReturn, None)
                .unwrap()
                .is_finite()
        );
        assert!(ValuePrediction::from_critic(&result.critics[name], 1, None).is_err());
    }
    assert!(
        ValuePrediction::Moments {
            mean: 1.,
            raw_second: f64::NAN
        }
        .scalar(&ValueOutput::ExpectedReturn, None)
        .is_err()
    );
    assert!(
        ValuePrediction::Moments {
            mean: f64::MAX,
            raw_second: 1.
        }
        .moment_summary()
        .is_err()
    );
    let overflow = ValuePrediction::Twin {
        q1: Box::new(ValuePrediction::Scalar(f64::MAX)),
        q2: Box::new(ValuePrediction::Scalar(-f64::MAX)),
    };
    assert!(overflow.scalar(&ValueOutput::Disagreement, None).is_err());
    r.observations[0].as_dense_mut().unwrap().values = vec![0.];
    assert!(bundle.inspect_critics(&r).is_err());
    let mut c = trajectory();
    c.arrays
        .insert("r".into(), Array::f64(vec![3], vec![f64::MAX; 3]).unwrap());
    assert!(trajectory_return_to_go(&c, 0).is_err());
    c.arrays
        .insert("r".into(), Array::f64(vec![3], vec![1., 2., 4.]).unwrap());
    c.arrays
        .insert("t".into(), Array::u8(vec![3], vec![1, 0, 1]).unwrap());
    assert!(trajectory_return_to_go(&c, 0).is_err());
}

#[test]
fn raw_second_moment_and_standard_deviation_preserve_invalid_variance() {
    let inconsistent = ValuePrediction::Moments {
        mean: 2.,
        raw_second: 3.,
    };
    assert_eq!(
        inconsistent
            .scalar(&ValueOutput::RawSecondMoment, None)
            .unwrap(),
        3.
    );
    assert_eq!(
        inconsistent.scalar(&ValueOutput::Variance, None).unwrap(),
        -1.
    );
    assert!(
        inconsistent
            .scalar(&ValueOutput::StandardDeviation, None)
            .is_err()
    );
    let consistent = ValuePrediction::Moments {
        mean: 2.,
        raw_second: 13.,
    };
    assert_eq!(
        consistent
            .scalar(&ValueOutput::StandardDeviation, None)
            .unwrap(),
        3.
    );
}

#[test]
fn soft_target_metadata_roundtrips_and_excludes_environmental_calibration() {
    let mut historical = snapshot();
    assert_eq!(return_target(&historical).unwrap(), None);
    let soft = ReturnTarget::SoftActionValue {
        alpha: 0.25,
        temperature_source: EntropyTemperatureSource::Snapshot,
        entropy_timing: EntropyTiming::ReplayBootstrap { n_step: 3 },
        log_density: LogDensityConvention::PolicySampleInActionCoordinates,
    };
    let mut environmental = declared();
    environmental.critics = BTreeMap::from([("q1".into(), named(ValueCriticKind::ScalarQ))]);
    attach_declaration(&mut historical, &environmental).unwrap();
    assert!(attach_return_target(&mut historical, soft).is_err());
    let mut historical = snapshot();
    let mut declaration = environmental.clone();
    declaration.conventions = None;
    attach_declaration(&mut historical, &declaration).unwrap();
    attach_return_target(&mut historical, soft).unwrap();
    assert_eq!(return_target(&historical).unwrap(), Some(soft));
    assert!(attach_return_target(&mut historical, ReturnTarget::Environmental).is_err());
    assert!(
        compatible_values(
            &declaration,
            "q1",
            &environmental,
            "q1",
            &ValueOutput::ExpectedReturn,
            ComparisonCondition::SameInput,
        )
        .is_err()
    );
}

#[test]
fn potential_shaped_metadata_cannot_certify_environmental_value_conventions() {
    let shaping = ReturnTarget::PotentialShaped {
        potential_id: [17; 32],
    };
    let mut environmental = declared();
    environmental.critics = BTreeMap::from([("q1".into(), named(ValueCriticKind::ScalarQ))]);
    let mut unshaped = snapshot();
    attach_declaration(&mut unshaped, &environmental).unwrap();
    assert!(attach_return_target(&mut unshaped, shaping).is_err());

    let mut shaped = snapshot();
    attach_return_target(&mut shaped, shaping).unwrap();
    assert_eq!(return_target(&shaped).unwrap(), Some(shaping));
    assert!(attach_declaration(&mut shaped, &environmental).is_err());
    assert!(attach_return_target(&mut shaped, ReturnTarget::Environmental).is_err());
    let mut shaped_value = environmental;
    shaped_value.conventions = None;
    attach_declaration(&mut shaped, &shaped_value).unwrap();
}
