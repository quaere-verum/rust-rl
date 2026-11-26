#![cfg(feature = "diagnostics")]
use rl::{
    core::action::ActionSpace,
    diagnostics::{reference::*, value::*},
};
use std::collections::BTreeMap;

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-10, "{actual} != {expected}");
}

#[test]
fn hand_values_use_the_actual_policy_and_only_legal_optimal_q() {
    let q = [-2., -5.];
    let legal = [true, true];
    for (action, expected_loss) in [(0, 0.), (1, 3.)] {
        let result = decision_loss(&q, &legal, DecisionPolicy::Deterministic(action)).unwrap();
        assert_eq!(result.optimal_value, -2.);
        assert_eq!(result.expected_action_value, q[action]);
        assert_eq!(result.loss, expected_loss);
        let probabilities = if action == 0 { [1., 0.] } else { [0., 1.] };
        assert_eq!(
            result,
            decision_loss(&q, &legal, DecisionPolicy::Categorical(&probabilities)).unwrap()
        );
    }
    let result = decision_loss(&q, &legal, DecisionPolicy::Categorical(&[0.25, 0.75])).unwrap();
    assert_eq!(
        result,
        DecisionLoss {
            optimal_value: -2.,
            expected_action_value: -4.25,
            loss: 2.25
        }
    );
    let masked = decision_loss(
        &[-2., 1000.],
        &[true, false],
        DecisionPolicy::Deterministic(0),
    )
    .unwrap();
    assert_eq!(
        masked,
        DecisionLoss {
            optimal_value: -2.,
            expected_action_value: -2.,
            loss: 0.
        }
    );
    assert!(
        decision_loss(
            &[-2., 1000.],
            &[true, false],
            DecisionPolicy::Deterministic(1)
        )
        .is_err()
    );
    for mass in [f64::MIN_POSITIVE, f64::from_bits(1), 0.25] {
        assert!(
            decision_loss(
                &[-2., 1000.],
                &[true, false],
                DecisionPolicy::Categorical(&[1. - mass, mass])
            )
            .is_err()
        );
    }
    assert!(
        decision_loss(
            &[-2., f64::MAX],
            &[true, false],
            DecisionPolicy::Categorical(&[1., -0.])
        )
        .is_ok()
    );
}

#[test]
fn roundoff_normalization_admits_f32_rows_without_thresholding_positive_loss() {
    for probabilities in [
        [f64::from(0.1_f32), f64::from(0.2_f32), f64::from(0.7_f32)],
        [f64::from(1. / 3_f32); 3],
        [0.2, 0.3, 0.5 + 0.5e-6],
        [0.2, 0.3, 0.5 - 0.5e-6],
    ] {
        let result = decision_loss(
            &[-2.; 3],
            &[true; 3],
            DecisionPolicy::Categorical(&probabilities),
        )
        .unwrap();
        close(result.expected_action_value, -2.);
        close(result.loss, 0.);
        assert!(result.loss >= 0.);
        assert_eq!(
            result.optimal_value - result.expected_action_value,
            result.loss
        );
    }
    for drift in [-1.1e-6, 1.1e-6] {
        assert!(
            decision_loss(
                &[-2.; 3],
                &[true; 3],
                DecisionPolicy::Categorical(&[0.2, 0.3, 0.5 + drift])
            )
            .is_err()
        );
    }
    let result = decision_loss(
        &[1., 1. - f64::EPSILON],
        &[true; 2],
        DecisionPolicy::Deterministic(1),
    )
    .unwrap();
    assert_eq!(result.loss, f64::EPSILON);
    // The arithmetic bound must stay finite at the largest finite scale.
    let result = decision_loss(
        &[f64::MAX; 64],
        &[true; 64],
        DecisionPolicy::Deterministic(63),
    )
    .unwrap();
    assert_eq!(result.loss, 0.);
    let result = decision_loss(
        &[-7.; 64],
        &[true; 64],
        DecisionPolicy::Categorical(&[1. / 64.; 64]),
    )
    .unwrap();
    assert_eq!(result.loss, 0.);
}

#[test]
fn malformed_rows_and_overflowed_loss_are_rejected() {
    for (q, mask) in [
        (vec![], vec![]),
        (vec![0.; 65], vec![true; 65]),
        (vec![0., 1.], vec![true]),
        (vec![0.], vec![false]),
        (vec![f64::NAN], vec![true]),
        (vec![f64::INFINITY], vec![true]),
        (vec![0., f64::NEG_INFINITY], vec![true, false]),
    ] {
        assert!(decision_loss(&q, &mask, DecisionPolicy::Deterministic(0)).is_err());
    }
    assert!(decision_loss(&[0.], &[true], DecisionPolicy::Deterministic(1)).is_err());
    for probabilities in [
        vec![],
        vec![1.],
        vec![0., 0.],
        vec![-0.1, 1.],
        vec![f64::NAN, 0.],
        vec![f64::INFINITY, 0.],
        vec![1. + 0.1e-6, 0.],
        vec![0.5, 0.5, 0.],
    ] {
        assert!(
            decision_loss(
                &[-2., -5.],
                &[true; 2],
                DecisionPolicy::Categorical(&probabilities)
            )
            .is_err(),
            "accepted {probabilities:?}"
        );
    }
    assert!(
        decision_loss(
            &[f64::MAX, -f64::MAX],
            &[true; 2],
            DecisionPolicy::Deterministic(1)
        )
        .is_err()
    );
    assert!(
        decision_loss(
            &[f64::MAX, -f64::MAX],
            &[true; 2],
            DecisionPolicy::Categorical(&[0., 1.])
        )
        .is_err()
    );
}

#[test]
fn negative_roundoff_is_clamped_with_coherent_fields_but_overflow_is_rejected() {
    // Three equal negative values accumulate E slightly above V in binary f64.
    let result = decision_loss(
        &[-7.; 3],
        &[true; 3],
        DecisionPolicy::Categorical(&[1. / 3.; 3]),
    )
    .unwrap();
    assert_eq!(
        result,
        DecisionLoss {
            optimal_value: -7.,
            expected_action_value: -7.,
            loss: 0.
        }
    );
    // A convex expectation can overflow during finite-precision summation even
    // when every input and every individual normalized product is finite.
    for q in [[f64::MAX; 17], [-f64::MAX; 17]] {
        assert!(
            decision_loss(
                &q,
                &[true; 17],
                DecisionPolicy::Categorical(&[1. / 17.; 17])
            )
            .is_err()
        );
    }
}
fn context() -> ReferenceContext {
    ReferenceContext {
        environment: "tiny-mdp".into(),
        observation_dim: 2,
        typed_observation_space: None,
        action_space: ActionSpace::Discrete { count: 2 },
        conventions: ValueConventions {
            dynamics: "tiny-v1".into(),
            observation: "complete [state,time]".into(),
            action: "0 or 1".into(),
            reward_units: "reward".into(),
            reward_convention: "larger better".into(),
            discount: f64::from(0.99_f32),
            horizon: "two decisions".into(),
            terminal_treatment: "zero continuation".into(),
            action_constraints: "explicit mask".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: BTreeMap::from([
            ("horizon".into(), ConfigValue::Integer(2)),
            ("arrival".into(), ConfigValue::Real(0.43)),
            ("service".into(), ConfigValue::Bool(true)),
            ("model".into(), ConfigValue::Text("v1".into())),
        ]),
        markov_observation: true,
        decision_time_coordinate: Some(1),
    }
}
fn oracle() -> ReferenceDeclaration {
    ReferenceDeclaration {
        version: 1,
        run_id: "oracle-run".into(),
        revision: 19,
        model_role: "reference".into(),
        context: context(),
        sources: BTreeMap::from([(
            "q-star".into(),
            ReferenceQuality {
                kind: ReferenceKind::VerifiedOptimal,
                method: "finite backward induction".into(),
                assumptions: vec!["declared numerical MDP".into()],
            },
        )]),
        finite_table: Some(FiniteReferenceTable {
            probe_set: "all".into(),
            rows: 2,
            coverage: ReferenceCoverage::AllLiveStates,
            kind: ReferenceTableKind::OptimalDiscreteActionValues,
        }),
    }
}
#[test]
fn decision_loss_accepts_exact_cross_run_contexts_and_both_coverage_declarations() {
    let current = context();
    for coverage in [
        ReferenceCoverage::AllLiveStates,
        ReferenceCoverage::ListedStates,
    ] {
        let mut oracle = oracle();
        oracle.finite_table.as_mut().unwrap().coverage = coverage;
        compatible_decision_loss(&current, &oracle, "q-star").unwrap();
        oracle.revision += 1;
        oracle.run_id = "another-run".into();
        oracle.model_role = "other-role".into();
        compatible_decision_loss(&current, &oracle, "q-star").unwrap();
    }
}
#[test]
fn decision_loss_refuses_every_context_mismatch_even_when_names_match() {
    let oracle = oracle();
    let mut variants = Vec::new();
    let mut push = |mutate: fn(&mut ReferenceContext)| {
        let mut c = context();
        mutate(&mut c);
        variants.push(c);
    };
    push(|c| c.environment = "other".into());
    push(|c| c.observation_dim = 3);
    push(|c| c.action_space = ActionSpace::Discrete { count: 3 });
    push(|c| c.conventions.dynamics = "other".into());
    push(|c| c.conventions.observation = "other".into());
    push(|c| c.conventions.action = "other".into());
    push(|c| c.conventions.reward_units = "other".into());
    push(|c| c.conventions.reward_convention = "other".into());
    push(|c| c.conventions.discount = 0.99);
    push(|c| c.conventions.horizon = "other".into());
    push(|c| c.conventions.terminal_treatment = "other".into());
    push(|c| c.conventions.action_constraints = "other".into());
    push(|c| {
        c.horizon = ReferenceHorizon::Finite { decisions: 3 };
        c.environment_config
            .insert("horizon".into(), ConfigValue::Integer(3));
    });
    push(|c| {
        c.environment_config
            .insert("arrival".into(), ConfigValue::Real(0.44));
    });
    push(|c| {
        c.environment_config
            .insert("service".into(), ConfigValue::Bool(false));
    });
    push(|c| {
        c.environment_config
            .insert("model".into(), ConfigValue::Text("v2".into()));
    });
    push(|c| {
        c.environment_config
            .insert("extra".into(), ConfigValue::Integer(1));
    });
    push(|c| {
        c.environment_config.remove("arrival");
    });
    push(|c| c.markov_observation = false);
    push(|c| c.decision_time_coordinate = Some(0));
    push(|c| c.decision_time_coordinate = None);
    for current in variants {
        assert!(
            compatible_decision_loss(&current, &oracle, "q-star").is_err(),
            "accepted {current:?}"
        );
    }
}
#[test]
fn decision_loss_requires_an_explicit_verified_finite_markov_q_subject() {
    let current = context();
    assert!(compatible_decision_loss(&current, &oracle(), "missing").is_err());
    let mut variants = Vec::new();
    for kind in [
        ReferenceKind::Learned,
        ReferenceKind::Heuristic,
        ReferenceKind::Analytical,
    ] {
        let mut d = oracle();
        d.sources.get_mut("q-star").unwrap().kind = kind;
        variants.push(d);
    }
    let mut push = |mutate: fn(&mut ReferenceDeclaration)| {
        let mut d = oracle();
        mutate(&mut d);
        variants.push(d);
    };
    push(|d| d.sources.clear());
    push(|d| d.finite_table = None);
    push(|d| d.finite_table.as_mut().unwrap().kind = ReferenceTableKind::DiscretePolicy);
    push(|d| d.context.markov_observation = false);
    push(|d| d.context.decision_time_coordinate = None);
    push(|d| {
        d.context.horizon = ReferenceHorizon::Continuing;
        d.context.environment_config.remove("horizon");
    });
    push(|d| d.context.action_space = ActionSpace::Unbounded { dimensions: 2 });
    push(|d| d.context.action_space = ActionSpace::Discrete { count: 65 });
    push(|d| d.context.environment_config.clear());
    push(|d| d.finite_table.as_mut().unwrap().rows = 0);
    for d in variants {
        assert!(compatible_decision_loss(&current, &d, "q-star").is_err());
        assert!(compatible_decision_loss(&d.context, &d, "q-star").is_err());
    }
}

#[test]
fn ordinary_value_predictions_cannot_masquerade_as_decision_loss() {
    let discrete = ValuePrediction::Discrete {
        values: vec![-2., -5.],
        legal: vec![true; 2],
    };
    let predictions = [
        ValuePrediction::Scalar(-2.),
        discrete.clone(),
        discrete.select_discrete(0).unwrap(),
        ValuePrediction::Quantiles {
            raw: vec![-2.],
            locations: vec![0.5],
            masses: vec![1.],
        },
        ValuePrediction::Moments {
            mean: -2.,
            raw_second: 4.,
        },
        ValuePrediction::Twin {
            q1: Box::new(ValuePrediction::Scalar(-2.)),
            q2: Box::new(ValuePrediction::Scalar(-3.)),
        },
    ];
    for prediction in predictions {
        assert!(prediction.scalar(&ValueOutput::DecisionLoss, None).is_err());
    }
    assert_eq!(
        ValuePrediction::Scalar(-2.)
            .scalar(&ValueOutput::ExpectedReturn, None)
            .unwrap(),
        -2.
    );
    let declared = ValueDeclaration {
        version: 1,
        run_id: "run".into(),
        revision: 0,
        model_role: "online".into(),
        conventions: Some(context().conventions),
        critics: BTreeMap::from([(
            "q".into(),
            CriticDeclaration {
                kind: ValueCriticKind::DiscreteQ,
                learned_baseline: None,
                actor: None,
                continuation: Some(ContinuationIdentity {
                    identity: "policy".into(),
                    stochastic: false,
                }),
                twin: None,
                utility: None,
            },
        )]),
    };
    for condition in [
        ComparisonCondition::SameInput,
        ComparisonCondition::ControllerValues,
    ] {
        assert!(
            compatible_values(
                &declared,
                "q",
                &declared,
                "q",
                &ValueOutput::DecisionLoss,
                condition
            )
            .is_err()
        );
        compatible_values(
            &declared,
            "q",
            &declared,
            "q",
            &ValueOutput::ExpectedReturn,
            condition,
        )
        .unwrap();
    }
}
