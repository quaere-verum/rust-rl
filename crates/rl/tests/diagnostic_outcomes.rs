#![cfg(feature = "diagnostics")]
use rl::{
    core::action::ActionSpace,
    diagnostics::{analysis::*, schema::*},
};
use std::collections::BTreeMap;

#[test]
fn dkw_inversion_preserves_unbounded_tails_and_nearest_rank_atoms() {
    let one = EmpiricalDistribution::new(&[7.]).unwrap();
    assert_eq!(one.quantile_band(0.5, 0.05).unwrap(), [None, None]);
    let atoms = EmpiricalDistribution::new(&[0.; 100]).unwrap();
    assert_eq!(
        atoms.quantile_band(0.5, 0.05).unwrap(),
        [Some(0.), Some(0.)]
    );
    assert_eq!(atoms.quantile_band(0., 0.05).unwrap(), [None, Some(0.)]);
    assert_eq!(atoms.quantile_band(1., 0.05).unwrap(), [Some(0.), None]);
    let empirical =
        EmpiricalDistribution::new(&(1..=100).map(f64::from).collect::<Vec<_>>()).unwrap();
    assert_eq!(
        empirical.quantile_band(0.5, 0.05).unwrap(),
        [Some(37.), Some(64.)]
    );
    for alpha in [0., -0.1, 1., f64::NAN] {
        assert!(one.quantile_band(0.5, alpha).is_err());
    }
    for p in [-0.1, 1.1, f64::NAN] {
        assert!(one.quantile_band(p, 0.05).is_err());
    }
}

fn fixture() -> (Chunk, Chunk, Chunk, Chunk, RunManifest) {
    let mut context = RecordContext::new("rows", 3);
    context.probe_set = Some("fixed".into());
    let definition = |ids| Chunk {
        context: context.clone(),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "fixed".into(),
            population: ProbePopulation::FixedReference,
            methodology: "same exact states".into(),
            observations: "obs".into(),
            action_masks: Some("mask".into()),
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::from([("h".into(), "h".into())]),
        }),
        arrays: BTreeMap::from([
            ("ids".into(), Array::u64(vec![2], ids).unwrap()),
            ("obs".into(), Array::f32(vec![2, 1], vec![2., 2.]).unwrap()),
            (
                "mask".into(),
                Array::u8(vec![2, 3], vec![1, 1, 0, 1, 1, 0]).unwrap(),
            ),
            (
                "h".into(),
                Array::u64(vec![2, 1, 1], vec![u64::MAX, u64::MAX]).unwrap(),
            ),
        ]),
    };
    let probes = definition(vec![u64::MAX - 1, u64::MAX]);
    let sample_probes = definition(vec![u64::MAX, u64::MAX - 1]);
    let critic = Chunk {
        context: context.clone(),
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Discrete { values: "q".into() },
            action_condition: ActionCondition::AllDiscreteActions,
            actions: None,
            continuation_policy: Some("pi3".into()),
            continuation_stochastic: Some(false),
        }),
        arrays: BTreeMap::from([(
            "q".into(),
            Array::f32(vec![2, 3], vec![10., 20., 30., 40., 50., 60.]).unwrap(),
        )]),
    };
    let sample = Chunk {
        context,
        record: Record::Empirical(EmpiricalRecord {
            returns: "returns".into(),
            seeds: "seeds".into(),
            probe_row: Some(1),
            initial_action: Some("action".into()),
            conditioning_observation: Some("obs".into()),
            conditioning_mask: Some("mask".into()),
            simulator_state: None,
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::Observation {
                    methodology: "fully observed".into(),
                },
                initial_action: ActionCondition::Fixed,
                continuation_policy: "pi3".into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "independent".into(),
                truncation_handling: "terminal".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "returns".into(),
                Array::f64(vec![2], vec![19., 21.]).unwrap(),
            ),
            ("seeds".into(), Array::u64(vec![2], vec![1, 2]).unwrap()),
            ("action".into(), Array::u64(vec![1], vec![1]).unwrap()),
            ("obs".into(), Array::f32(vec![1, 1], vec![2.]).unwrap()),
            ("mask".into(), Array::u8(vec![1, 3], vec![1, 1, 0]).unwrap()),
        ]),
    };
    (
        critic,
        probes,
        sample,
        sample_probes,
        RunManifest::new(
            "rows",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 3 },
            0.9,
        )
        .unwrap(),
    )
}

#[test]
fn conditional_environmental_returns_do_not_calibrate_a_soft_critic() {
    use rl::diagnostics::value::{
        EntropyTemperatureSource, EntropyTiming, LogDensityConvention, ReturnTarget,
        attach_return_target,
    };
    let (mut critic, critic_probes, empirical, empirical_probes, manifest) = fixture();
    assert!(
        match_conditional_rows(
            &critic,
            &critic_probes,
            0,
            &empirical,
            &empirical_probes,
            &manifest
        )
        .is_ok()
    );
    attach_return_target(
        &mut critic,
        ReturnTarget::SoftActionValue {
            alpha: 0.25,
            temperature_source: EntropyTemperatureSource::Snapshot,
            entropy_timing: EntropyTiming::TransitionSuccessor,
            log_density: LogDensityConvention::PolicySampleInActionCoordinates,
        },
    )
    .unwrap();
    let error = match_conditional_rows(
        &critic,
        &critic_probes,
        0,
        &empirical,
        &empirical_probes,
        &manifest,
    )
    .unwrap_err();
    assert!(error.to_string().contains("entropy-regularized"));
}

#[test]
fn explicit_rows_bind_stable_typed_ids_and_select_one_discrete_column() {
    let (c, p, e, s, m) = fixture();
    let matched = match_conditional_rows(&c, &p, 0, &e, &s, &m).unwrap();
    assert_eq!(
        (
            matched.critic_row,
            matched.empirical_row,
            matched.discrete_action
        ),
        (0, 1, Some(1))
    );
    assert_eq!(
        c.arrays["q"].as_f32().unwrap()[matched.critic_row * 3 + matched.discrete_action.unwrap()],
        20.
    );
    assert!(
        match_conditional(&c, &s, &e, &m).is_err(),
        "legacy API cannot select a Q column"
    );
    for row in [1, 2, usize::MAX] {
        assert!(match_conditional_rows(&c, &p, row, &e, &s, &m).is_err());
    }
    for reason in 0..15 {
        let (mut c, mut p, mut e, mut s, mut m) = fixture();
        match reason {
            0 => {
                s.arrays.insert(
                    "ids".into(),
                    Array::u64(vec![2], vec![u64::MAX, u64::MAX - 2]).unwrap(),
                );
            }
            1 => {
                s.arrays.insert(
                    "h".into(),
                    Array::u64(vec![2, 1, 1], vec![u64::MAX, 1]).unwrap(),
                );
            }
            2 => {
                e.arrays
                    .insert("action".into(), Array::u64(vec![1], vec![2]).unwrap());
            }
            3 => {
                e.arrays
                    .insert("action".into(), Array::u64(vec![1], vec![3]).unwrap());
            }
            4 => {
                e.arrays.insert(
                    "action".into(),
                    Array::u64(vec![1], vec![u64::MAX]).unwrap(),
                );
            }
            5 => {
                e.arrays
                    .insert("action".into(), Array::f32(vec![1, 1], vec![1.]).unwrap());
            }
            6 => {
                m.action_space = ActionSpace::Unbounded { dimensions: 1 };
            }
            7 => {
                c.arrays.insert(
                    "q".into(),
                    Array::f32(vec![1, 3], vec![10., 20., 30.]).unwrap(),
                );
            }
            8 => {
                if let Record::ProbeSet(p) = &mut p.record {
                    p.methodology = "changed".into();
                }
            }
            9 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.probe_row = None;
                }
            }
            10 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.probe_row = Some(2);
                }
            }
            11 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.initial_action = None;
                }
            }
            12 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.initial_action = ActionCondition::CurrentPolicy;
                }
            }
            13 => {
                p.arrays
                    .insert("obs".into(), Array::f64(vec![2, 1], vec![2., 2.]).unwrap());
            }
            _ => {
                s.context.identity.run_id = "other".into();
            }
        }
        assert!(
            match_conditional_rows(&c, &p, 0, &e, &s, &m).is_err(),
            "mutation {reason}"
        );
    }
}

#[test]
fn scalar_state_value_and_action_value_keep_distinct_first_action_semantics() {
    let (mut c, mut p, mut e, mut s, m) = fixture();
    if let Record::Critic(c) = &mut c.record {
        c.representation = CriticRepresentation::Scalar { values: "v".into() };
        c.action_condition = ActionCondition::StateValue;
    }
    c.arrays
        .insert("v".into(), Array::f32(vec![2], vec![10., 20.]).unwrap());
    assert!(match_conditional_rows(&c, &p, 0, &e, &s, &m).is_err());
    if let Record::Empirical(e) = &mut e.record {
        e.initial_action = None;
        e.evaluation.initial_action = ActionCondition::CurrentPolicy;
    }
    assert!(match_conditional_rows(&c, &p, 0, &e, &s, &m).is_ok());
    assert_eq!(match_conditional(&c, &s, &e, &m).unwrap().probe_row, 1);
    for condition in [ActionCondition::Fixed, ActionCondition::CurrentPolicy] {
        if let Record::Critic(c) = &mut c.record {
            c.action_condition = condition;
            c.actions = Some("a".into());
        }
        c.arrays
            .insert("a".into(), Array::u64(vec![2], vec![1, 1]).unwrap());
        assert!(match_conditional_rows(&c, &p, 0, &e, &s, &m).is_err());
        if let Record::Empirical(e) = &mut e.record {
            e.initial_action = Some("action".into());
            e.evaluation.initial_action = ActionCondition::Fixed;
        }
        assert!(match_conditional_rows(&c, &p, 0, &e, &s, &m).is_ok());
        if let Record::Empirical(e) = &mut e.record {
            e.initial_action = None;
            e.evaluation.initial_action = ActionCondition::CurrentPolicy;
        }
    }
    if let Record::Empirical(e) = &mut e.record {
        e.initial_action = Some("action".into());
        e.evaluation.initial_action = ActionCondition::Fixed;
    }
    for definition in [&mut p, &mut s] {
        if let Record::ProbeSet(p) = &mut definition.record {
            p.fixed_actions = Some("fixed".into());
        }
        definition
            .arrays
            .insert("fixed".into(), Array::u64(vec![2], vec![0, 0]).unwrap());
    }
    assert!(
        match_conditional_rows(&c, &p, 0, &e, &s, &m).is_err(),
        "Q and MC must not override the declared fixed probe action"
    );
}

#[test]
fn discrete_boundary_actions_masks_and_identity_mutations_are_checked_in_core() {
    for action in [0, 2] {
        let (c, mut p, mut e, mut s, m) = fixture();
        for definition in [&mut p, &mut s] {
            definition
                .arrays
                .insert("mask".into(), Array::u8(vec![2, 3], vec![1; 6]).unwrap());
        }
        if let Record::Empirical(e) = &mut e.record {
            e.conditioning_mask = None;
        }
        e.arrays
            .insert("action".into(), Array::u64(vec![1], vec![action]).unwrap());
        if let Record::ProbeSet(s) = &mut s.record {
            s.action_masks = None;
        }
        assert_eq!(
            match_conditional_rows(&c, &p, 0, &e, &s, &m)
                .unwrap()
                .discrete_action,
            Some(action as usize)
        );
    }
    for reason in 0..15 {
        let (mut c, mut p, mut e, mut s, m) = fixture();
        match reason {
            0 => e.context.identity.learner_update_revision += 1,
            1 => e.context.identity.model_role = "target".into(),
            2 => e.context.identity.checkpoint_id = Some("wrong".into()),
            3 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.discount = 0.5;
                }
            }
            4 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.complete_returns = false;
                }
            }
            5 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.independent_future_streams = false;
                }
            }
            6 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.continuation_policy = "wrong".into();
                }
            }
            7 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.stochastic_policy = true;
                }
            }
            8 => {
                if let Record::ProbeSet(p) = &mut p.record {
                    p.population = ProbePopulation::PolicyVisited;
                }
            }
            9 => {
                s.arrays
                    .insert("ids".into(), Array::u64(vec![2], vec![1, 1]).unwrap());
            }
            10 => {
                if let Record::ProbeSet(p) = &mut p.record {
                    p.histories.clear();
                }
            }
            11 => {
                c.arrays
                    .insert("q".into(), Array::f32(vec![2, 2], vec![0.; 4]).unwrap());
            }
            12 => {
                s.arrays
                    .insert("mask".into(), Array::u8(vec![2, 2], vec![1; 4]).unwrap());
            }
            13 => {
                if let Record::Empirical(e) = &mut e.record {
                    e.evaluation.conditioning = Conditioning::InitialScenario;
                }
            }
            _ => {
                if let Record::Critic(c) = &mut c.record {
                    c.representation = CriticRepresentation::Scalar { values: "v".into() };
                }
                c.arrays
                    .insert("v".into(), Array::f32(vec![2], vec![1., 2.]).unwrap());
            }
        }
        assert!(
            match_conditional_rows(&c, &p, 0, &e, &s, &m).is_err(),
            "mutation {reason}"
        );
    }
}
