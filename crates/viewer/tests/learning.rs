//! Independent numerical and lifetime evidence for recorded learning summaries.
#[path = "support/learning_critic.rs"]
mod learning_fixture;
use learning_fixture::{critic_fixture, empirical, probes};
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*},
};
use rl_viewer::learning::{ChangeKind, evaluation_summary, policy_change};
use std::collections::BTreeMap;

fn policy(actions: Array, parameters: PolicyParameters, arrays: BTreeMap<String, Array>) -> Chunk {
    let mut context = RecordContext::new("learning", 1);
    context.probe_set = Some("fixed".into());
    let mut arrays = arrays;
    arrays.insert("actions".into(), actions);
    Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            stochastic: !matches!(parameters, PolicyParameters::Deterministic),
            parameters,
            entropy: None,
            entropy_method: None,
            masks: None,
        }),
        arrays,
    }
}
fn categorical(probabilities: [f64; 4]) -> Chunk {
    policy(
        Array::u64(vec![2], vec![0, 0]).unwrap(),
        PolicyParameters::Categorical {
            probabilities: "probabilities".into(),
        },
        BTreeMap::from([(
            "probabilities".into(),
            Array::f64(vec![2, 2], probabilities.to_vec()).unwrap(),
        )]),
    )
}

#[test]
fn reverse_kl_uses_reference_as_first_distribution_and_preserves_support_infinity() {
    // KL([.5,.5] || [.25,.75]) = .5 ln(4/3), not KL in the reverse direction.
    let space = ActionSpace::Discrete { count: 2 };
    let p = probes(&space, false);
    let reference = categorical([0.5, 0.5, 0.5, 0.5]);
    let current = categorical([0.25, 0.75, 0.25, 0.75]);
    let change = policy_change(&reference, &p, &current, &p, &space).unwrap();
    assert_eq!(change.kind, ChangeKind::ReferenceKl);
    assert!((change.value - 0.143_841_036_225_890_42).abs() < 1e-13);
    assert!(
        policy_change(&reference, &p, &categorical([0., 1., 0., 1.]), &p, &space)
            .unwrap()
            .value
            .is_infinite()
    );
}

#[test]
fn normalized_rms_uses_every_component_and_fixed_box_spans() {
    let space = ActionSpace::Box {
        low: vec![-1., -3.],
        high: vec![1., 3.],
    };
    let p = probes(&space, false);
    let reference = policy(
        Array::f64(vec![2, 2], vec![0., 0., 0., 0.]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    let current = policy(
        Array::f64(vec![2, 2], vec![0., 3., 0., 3.]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    let change = policy_change(&reference, &p, &current, &p, &space).unwrap();
    assert_eq!(change.kind, ChangeKind::NormalizedRms);
    assert_eq!(change.normalization, vec![2., 6.]);
    assert!((change.value - 0.353_553_390_593_273_8).abs() < 1e-14);
}

#[test]
fn policy_population_alignment_uses_ids_and_complete_inputs() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let p = probes(&space, false);
    let reversed = probes(&space, true);
    let reference = policy(
        Array::f64(vec![2, 1], vec![1., 2.]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    let current = policy(
        Array::f64(vec![2, 1], vec![2., 1.]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    assert_eq!(
        policy_change(&reference, &p, &current, &reversed, &space)
            .unwrap()
            .value,
        0.
    );
    assert!(
        policy_change(&reference, &p, &current, &p, &space)
            .unwrap()
            .value
            > 0.
    );
    let mut mismatched = reversed;
    let Record::ProbeSet(m) = &mut mismatched.record else {
        unreachable!()
    };
    m.population = ProbePopulation::PolicyVisited;
    assert!(policy_change(&reference, &p, &current, &mismatched, &space).is_err());
}

#[test]
fn evaluation_statistics_use_genuine_samples_and_do_not_invent_one_sample_uncertainty() {
    let summary = evaluation_summary(&empirical(4, vec![1., 3., 5.])).unwrap();
    assert_eq!(summary.count, 3);
    assert_eq!(summary.mean, 3.);
    assert_eq!(summary.sample_variance, Some(4.));
    assert!((summary.mean_standard_error.unwrap() - 1.154_700_538_379_251_5).abs() < 1e-14);
    let one = evaluation_summary(&empirical(9, vec![7.])).unwrap();
    assert_eq!(one.count, 1);
    assert_eq!(one.mean, 7.);
    assert_eq!(one.mean_standard_error, None);
    let mut prefix = empirical(12, vec![10., 11.]);
    let Record::Empirical(e) = &mut prefix.record else {
        unreachable!()
    };
    e.evaluation.complete_returns = false;
    assert!(evaluation_summary(&prefix).is_err());
}

#[test]
fn gaussian_change_uses_parameters_and_reference_to_current_direction() {
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let p = probes(&space, false);
    let gaussian = |std: f64, sampled: f64| {
        policy(
            Array::f64(vec![2, 1], vec![sampled, sampled]).unwrap(),
            PolicyParameters::Gaussian {
                mean: "mean".into(),
                log_std: "std".into(),
                tanh_bounded: true,
            },
            BTreeMap::from([
                ("mean".into(), Array::f64(vec![2, 1], vec![0., 0.]).unwrap()),
                ("std".into(), Array::f64(vec![1], vec![std.ln()]).unwrap()),
            ]),
        )
    };
    let change = policy_change(&gaussian(1., 0.9), &p, &gaussian(2., -0.9), &p, &space).unwrap();
    assert!((change.value - 0.318_147_180_559_945_3).abs() < 1e-13);
}

#[test]
fn deterministic_discrete_change_is_disagreement_not_distance_between_ids() {
    let space = ActionSpace::Discrete { count: 4 };
    let p = probes(&space, false);
    let a = policy(
        Array::u64(vec![2], vec![0, 0]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    let b = policy(
        Array::u64(vec![2], vec![0, 3]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    let change = policy_change(&a, &p, &b, &p, &space).unwrap();
    assert_eq!(change.kind, ChangeKind::DiscreteDisagreement);
    assert_eq!(change.value, 0.5);
}

#[test]
fn independent_error_requires_declared_observation_conditional_complete_futures() {
    use rl_viewer::learning::{ErrorKind, independent_critic_error};
    let (critic, p, sample, manifest) = critic_fixture();
    let error = independent_critic_error(&critic, &p, 0, &sample, &p, &manifest).unwrap();
    assert_eq!(error.kind, ErrorKind::ScalarAbsolute);
    assert_eq!(error.value, 1.);
    assert_eq!(error.count, 3);
    for change in 0..4 {
        let mut bad = sample.clone();
        if change == 0 {
            bad.arrays
                .remove(rl::diagnostics::empirical::EMPIRICAL_METADATA_KEY);
        }
        let Record::Empirical(e) = &mut bad.record else {
            unreachable!()
        };
        match change {
            1 => e.evaluation.independent_future_streams = false,
            2 => {
                e.evaluation.conditioning = Conditioning::FullSimulatorState {
                    description: "hidden physical state".into(),
                }
            }
            3 => bad.context.identity.learner_update_revision = 4,
            _ => {}
        }
        assert!(
            independent_critic_error(&critic, &p, 0, &bad, &p, &manifest).is_err(),
            "case {change}"
        );
    }
}

#[test]
fn a_single_future_is_not_a_scalar_conditional_mean_reference() {
    let (critic, p, mut sample, manifest) = critic_fixture();
    sample
        .arrays
        .insert("returns".into(), Array::f64(vec![1], vec![3.]).unwrap());
    sample
        .arrays
        .insert("seeds".into(), Array::u64(vec![1], vec![9]).unwrap());
    assert!(
        rl_viewer::learning::independent_critic_error(&critic, &p, 0, &sample, &p, &manifest)
            .is_err()
    );
}

#[test]
fn complete_scientific_populations_are_rejected_rather_than_silently_subsampled() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let mut context = RecordContext::new("learning", 1);
    context.probe_set = Some("large".into());
    let p = ProbeSet::new(
        "large",
        ProbePopulation::FixedReference,
        "all fixed rows",
        (0..1025)
            .map(|i| Observation::new(vec![i as f32]))
            .collect(),
        None,
        space.clone(),
    )
    .unwrap()
    .record(context.clone())
    .unwrap();
    let mut c = policy(
        Array::f64(vec![1025, 1], vec![0.; 1025]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    c.context = context;
    assert!(
        policy_change(&c, &p, &c, &p, &space)
            .unwrap_err()
            .to_string()
            .contains("1024")
    );
    let wide = ActionSpace::Unbounded { dimensions: 65 };
    let p = probes(&wide, false);
    let c = policy(
        Array::f64(vec![2, 65], vec![0.; 130]).unwrap(),
        PolicyParameters::Deterministic,
        BTreeMap::new(),
    );
    assert!(
        policy_change(&c, &p, &c, &p, &wide)
            .unwrap_err()
            .to_string()
            .contains("64")
    );
}

#[test]
fn reference_rms_scales_remain_fixed_for_unbounded_action_comparisons() {
    let space = ActionSpace::Unbounded { dimensions: 1 };
    let p = probes(&space, false);
    let make = |scale: f64| {
        policy(
            Array::f64(vec![2, 1], vec![scale, 2. * scale]).unwrap(),
            PolicyParameters::Deterministic,
            BTreeMap::new(),
        )
    };
    let a = policy_change(&make(1.), &p, &make(2.), &p, &space).unwrap();
    let b = policy_change(&make(1.), &p, &make(3.), &p, &space).unwrap();
    assert_eq!(a.normalization, b.normalization);
    assert!((a.value - 1.).abs() < 1e-14);
    assert!((b.value - 2.).abs() < 1e-14);
    let mut dependent = empirical(3, vec![1., 3.]);
    let Record::Empirical(e) = &mut dependent.record else {
        unreachable!()
    };
    e.evaluation.independent_future_streams = false;
    assert_eq!(
        evaluation_summary(&dependent).unwrap().mean_standard_error,
        None
    );
}

#[test]
fn ordered_quantile_error_scores_the_recorded_atomic_approximation_and_keeps_crossings() {
    use rl::diagnostics::{
        empirical,
        value::{self, ValueCriticKind},
    };
    use rl_viewer::learning::{ErrorKind, independent_critic_error};
    let (mut critic, p, mut sample, manifest) = critic_fixture();
    let mut d = value::declaration(&critic).unwrap().unwrap();
    critic.arrays.remove(value::VALUE_METADATA_KEY);
    d.critics.get_mut("critic").unwrap().kind = ValueCriticKind::QuantileQ;
    d.critics.get_mut("critic").unwrap().learned_baseline = None;
    let Record::Critic(c) = &mut critic.record else {
        unreachable!()
    };
    c.representation = CriticRepresentation::Quantiles {
        values: "q".into(),
        locations: "locations".into(),
        masses: "masses".into(),
    };
    c.action_condition = ActionCondition::Fixed;
    c.actions = Some("a".into());
    critic
        .arrays
        .insert("a".into(), Array::f32(vec![2, 1], vec![0., 0.]).unwrap());
    critic.arrays.insert(
        "q".into(),
        Array::f64(vec![2, 2], vec![2., 0., 2., 0.]).unwrap(),
    );
    critic.arrays.insert(
        "locations".into(),
        Array::f64(vec![2], vec![0.25, 0.75]).unwrap(),
    );
    critic.arrays.insert(
        "masses".into(),
        Array::f64(vec![2], vec![0.5, 0.5]).unwrap(),
    );
    value::attach_declaration(&mut critic, &d).unwrap();
    let declaration = empirical::declaration(&sample).unwrap().unwrap();
    sample.arrays.remove(empirical::EMPIRICAL_METADATA_KEY);
    let Record::Empirical(e) = &mut sample.record else {
        unreachable!()
    };
    e.initial_action = Some("first".into());
    e.evaluation.initial_action = ActionCondition::Fixed;
    sample
        .arrays
        .insert("first".into(), Array::f32(vec![1, 1], vec![0.]).unwrap());
    empirical::attach_declaration(&mut sample, &declaration).unwrap();
    let error = independent_critic_error(&critic, &p, 0, &sample, &p, &manifest).unwrap();
    assert_eq!(error.kind, ErrorKind::OrderedQuantileCrps);
    assert_eq!(error.raw_crossings, 1);
    assert!((error.value - 1.833_333_333_333_333_3).abs() < 1e-14);
}

#[test]
fn full_state_and_history_evidence_stay_diagnostic_even_with_matching_markov_context() {
    use rl::diagnostics::empirical;
    use rl_viewer::learning::independent_critic_error;
    for conditioning in [
        Conditioning::FullSimulatorState {
            description: "complete privileged physical state".into(),
        },
        Conditioning::History {
            methodology: "recorded recurrent state".into(),
        },
    ] {
        let (critic, p, mut sample, manifest) = critic_fixture();
        let mut declaration = empirical::declaration(&sample).unwrap().unwrap();
        sample.arrays.remove(empirical::EMPIRICAL_METADATA_KEY);
        declaration.conditioning = conditioning.clone();
        let Record::Empirical(e) = &mut sample.record else {
            unreachable!()
        };
        e.evaluation.conditioning = conditioning;
        e.simulator_state = Some("physical".into());
        sample
            .arrays
            .insert("physical".into(), Array::f64(vec![1], vec![0.]).unwrap());
        empirical::attach_declaration(&mut sample, &declaration).unwrap();
        assert!(independent_critic_error(&critic, &p, 0, &sample, &p, &manifest).is_err());
    }
    let (critic, mut p, sample, manifest) = critic_fixture();
    let Record::ProbeSet(probe) = &mut p.record else {
        unreachable!()
    };
    probe.histories.insert("memory".into(), "memory".into());
    p.arrays.insert(
        "memory".into(),
        Array::f32(vec![2, 1], vec![0., 0.]).unwrap(),
    );
    assert!(independent_critic_error(&critic, &p, 0, &sample, &p, &manifest).is_err());
}

#[test]
fn policy_record_only_effective_masks_are_part_of_the_matched_input_condition() {
    let space = ActionSpace::Discrete { count: 2 };
    let mut p = probes(&space, false);
    let Record::ProbeSet(probe) = &mut p.record else {
        unreachable!()
    };
    if let Some(key) = probe.action_masks.take() {
        p.arrays.remove(&key);
    }
    for deterministic in [false, true] {
        let mut reference = categorical([1., 0., 1., 0.]);
        let mut current = categorical([0., 1., 0., 1.]);
        current
            .arrays
            .insert("actions".into(), Array::u64(vec![2], vec![1, 1]).unwrap());
        for (chunk, masks) in [
            (&mut reference, vec![1, 0, 1, 0]),
            (&mut current, vec![0, 1, 0, 1]),
        ] {
            let Record::Policy(meta) = &mut chunk.record else {
                unreachable!()
            };
            meta.masks = Some("recorded-mask".into());
            if deterministic {
                meta.parameters = PolicyParameters::Deterministic;
                meta.stochastic = false;
            }
            chunk.arrays.insert(
                "recorded-mask".into(),
                Array::u8(vec![2, 2], masks).unwrap(),
            );
        }
        rl::diagnostics::policy_batch::PolicyBatch::new(&reference, Some(&p), &space).unwrap();
        rl::diagnostics::policy_batch::PolicyBatch::new(&current, Some(&p), &space).unwrap();
        assert!(
            policy_change(&reference, &p, &current, &p, &space).is_err(),
            "conditioning differs for deterministic={deterministic}"
        );
    }
}
