#![cfg(feature = "diagnostics")]
use rl::{
    core::action::ActionSpace,
    diagnostics::{
        analysis::{EmpiricalDistribution, QuantileForecast},
        empirical::*,
        reference::{ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceObjective},
        schema::*,
        value::ValueConventions,
    },
};
use std::collections::BTreeMap;

fn fixture() -> (Chunk, RunManifest, EmpiricalDeclaration) {
    let mut manifest = RunManifest::new(
        "empirical",
        "science",
        "test",
        2,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        0.9,
    )
    .unwrap();
    manifest.environment = "exact-f32".into();
    let context = ReferenceContext {
        environment: manifest.environment.clone(),
        observation_dim: 2,
        typed_observation_space: None,
        action_space: manifest.action_space.clone(),
        conventions: ValueConventions {
            dynamics: "x' = f32(0.5*x + a + independent noise)".into(),
            observation: "exact f32 state and integer clock".into(),
            action: "one f32 control".into(),
            reward_units: "reward".into(),
            reward_convention: manifest.reward_convention.clone(),
            discount: 0.9,
            horizon: "2 decisions, clock observed".into(),
            terminal_treatment: "final reward once, zero terminal tail".into(),
            action_constraints: "[-1,1]".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: BTreeMap::from([("horizon".into(), ConfigValue::Integer(2))]),
        markov_observation: true,
        decision_time_coordinate: Some(1),
    };
    let conditioning = Conditioning::Observation {
        methodology: "Exact f32 observation reconstructs all physical state and integer clock; independently seeded future RNG".into(),
    };
    let mut record_context = RecordContext::new("empirical", 3);
    record_context.identity.checkpoint_id = Some("k3".into());
    record_context.probe_set = Some("fixed".into());
    let chunk = Chunk {
        context: record_context,
        record: Record::Empirical(EmpiricalRecord {
            returns: "returns".into(),
            seeds: "seeds".into(),
            probe_row: Some(0),
            initial_action: Some("action".into()),
            conditioning_observation: Some("observation".into()),
            conditioning_mask: None,
            simulator_state: None,
            evaluation: EvaluationMetadata {
                conditioning: conditioning.clone(),
                initial_action: ActionCondition::Fixed,
                continuation_policy: "frozen-policy-k3".into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "independent explicit seeds".into(),
                truncation_handling: "true terminal, no bootstrap".into(),
                complete_returns: true,
            },
        }),
        arrays: BTreeMap::from([
            (
                "returns".into(),
                Array::f64(vec![3], vec![1., 2., 3.]).unwrap(),
            ),
            (
                "seeds".into(),
                Array::u64(vec![3], vec![11, 12, 13]).unwrap(),
            ),
            (
                "observation".into(),
                Array::f32(vec![1, 2], vec![0.5, 0.]).unwrap(),
            ),
            ("action".into(), Array::f32(vec![1, 1], vec![0.]).unwrap()),
        ]),
    };
    let declaration = EmpiricalDeclaration {
        version: EMPIRICAL_DECLARATION_VERSION,
        run_id: "empirical".into(),
        revision: 3,
        model_role: "online".into(),
        checkpoint_id: Some("k3".into()),
        context,
        conditioning,
    };
    (chunk, manifest, declaration)
}

#[test]
fn mean_standard_error_uses_sample_variance_and_singletons_have_no_estimated_error() {
    let d = EmpiricalDistribution::new(&[1., 2., 3., 4.]).unwrap();
    let s = d.summary().unwrap();
    assert_eq!(s.count, 4);
    assert_eq!(s.mean, 2.5);
    assert!((s.sample_variance.unwrap() - 5. / 3.).abs() < 1e-14);
    assert!((s.mean_standard_error.unwrap() - (5_f64 / 12.).sqrt()).abs() < 1e-14);
    let one = EmpiricalDistribution::new(&[7.])
        .unwrap()
        .summary()
        .unwrap();
    assert_eq!(one.sample_variance, None);
    assert_eq!(one.mean_standard_error, None);
    let constant = EmpiricalDistribution::new(&[7.; 3])
        .unwrap()
        .summary()
        .unwrap();
    assert_eq!(constant.sample_variance, Some(0.));
    assert_eq!(constant.mean_standard_error, Some(0.));
    assert!(
        EmpiricalDistribution::new(&[-1e200, 1e200])
            .unwrap()
            .summary()
            .is_err()
    );
}

#[test]
fn cdf_points_preserve_atomic_jumps_and_coalesce_ties_without_interpolation() {
    let d = EmpiricalDistribution::new(&[2., 0., 2., 0.]).unwrap();
    assert_eq!(
        d.cdf_points().unwrap(),
        vec![[0., 0.], [0., 0.5], [2., 0.5], [2., 1.]]
    );
    assert_eq!(
        EmpiricalDistribution::new(&[7.])
            .unwrap()
            .cdf_points()
            .unwrap(),
        vec![[7., 0.], [7., 1.]]
    );
}

#[test]
fn quantile_mean_crps_integrates_the_weighted_reference_distribution() {
    let forecast =
        QuantileForecast::new(vec![0., 2.], vec![0.125, 0.625], vec![0.25, 0.75]).unwrap();
    let reference = QuantileForecast::new(vec![1., 4.], vec![0.25, 0.75], vec![0.5, 0.5]).unwrap();
    // E|X - Y| is 1.75. Half E|X - X'| is 0.375 for the forecast and 0.75 for
    // the reference, so reversing the forecast and outcome law changes the score.
    assert_eq!(forecast.mean_crps_against(&reference).unwrap(), 1.375);
    assert_eq!(reference.mean_crps_against(&forecast).unwrap(), 1.);
    assert_eq!(forecast.mean_crps_against(&forecast).unwrap(), 0.375);

    let split_atom = QuantileForecast::new(
        vec![0., 0., 2.],
        vec![0.0625, 0.1875, 0.625],
        vec![0.125, 0.125, 0.75],
    )
    .unwrap();
    assert_eq!(split_atom.mean_crps_against(&reference).unwrap(), 1.375);

    let crossed = QuantileForecast::new(vec![4., 1.], vec![0.25, 0.75], vec![0.5, 0.5]).unwrap();
    assert!(forecast.mean_crps_against(&crossed).is_err());
    assert!(crossed.mean_crps_against(&reference).is_err());
}

#[test]
fn quantile_mean_crps_checks_the_weighted_range_for_overflow() {
    let low = QuantileForecast::new(vec![-1e308], vec![0.5], vec![1.]).unwrap();
    let high = QuantileForecast::new(vec![1e308], vec![0.5], vec![1.]).unwrap();
    let mixture =
        QuantileForecast::new(vec![-1e308, 1e308], vec![0.25, 0.75], vec![0.5, 0.5]).unwrap();
    assert_eq!(low.mean_crps_against(&mixture).unwrap(), 1e308);
    assert!(low.mean_crps_against(&high).is_err());
}

#[test]
fn declaration_is_optional_immutable_and_binds_actual_source_and_manifest() {
    let (mut chunk, manifest, d) = fixture();
    assert!(declaration(&chunk).unwrap().is_none());
    assert!(validate_for(&chunk, &manifest).is_err());
    attach_declaration(&mut chunk, &d).unwrap();
    assert_eq!(validate_for(&chunk, &manifest).unwrap(), d);
    compatible_target(&d, &d.context).unwrap();
    attach_declaration(&mut chunk, &d).unwrap();
    let mut changed = d.clone();
    changed.revision += 1;
    assert!(attach_declaration(&mut chunk, &changed).is_err());
    for reason in 0..12 {
        let (mut c, mut m, mut d) = fixture();
        match reason {
            0 => d.run_id = "other".into(),
            1 => d.revision += 1,
            2 => d.model_role = "target".into(),
            3 => d.checkpoint_id = None,
            4 => m.discount = 0.8,
            5 => m.environment = "other".into(),
            6 => {
                if let Record::Empirical(e) = &mut c.record {
                    e.evaluation.complete_returns = false;
                }
            }
            7 => {
                if let Record::Empirical(e) = &mut c.record {
                    e.evaluation.independent_future_streams = false;
                }
            }
            8 => {
                c.arrays.insert(
                    "seeds".into(),
                    Array::u64(vec![3], vec![11, 11, 13]).unwrap(),
                );
            }
            9 => {
                c.arrays.insert(
                    "observation".into(),
                    Array::f32(vec![1, 1], vec![0.5]).unwrap(),
                );
            }
            10 => {
                if let Record::Empirical(e) = &mut c.record {
                    e.evaluation.conditioning = Conditioning::InitialScenario;
                }
            }
            _ => {
                c.arrays
                    .insert("action".into(), Array::f32(vec![1, 1], vec![2.]).unwrap());
            }
        }
        assert!(
            attach_declaration(&mut c, &d)
                .and_then(|()| validate_for(&c, &m))
                .is_err(),
            "mutation {reason}"
        );
    }
    let mut target = d.context.clone();
    target.conventions.terminal_treatment = "different".into();
    assert!(compatible_target(&d, &target).is_err());
    target = d.context.clone();
    target.horizon = ReferenceHorizon::Continuing;
    target.environment_config.remove("horizon");
    assert!(compatible_target(&d, &target).is_err());
}

#[test]
fn full_physical_state_is_descriptive_even_when_observation_is_markov() {
    let (mut c, m, mut d) = fixture();
    d.conditioning = Conditioning::FullSimulatorState {
        description: "physical state".into(),
    };
    if let Record::Empirical(e) = &mut c.record {
        e.evaluation.conditioning = d.conditioning.clone();
    }
    assert!(
        attach_declaration(&mut c, &d).is_err(),
        "physical payload required"
    );
    c.arrays.insert(
        "physical".into(),
        Array::f64(vec![2], vec![0.5, 0.]).unwrap(),
    );
    if let Record::Empirical(e) = &mut c.record {
        e.simulator_state = Some("physical".into());
    }
    attach_declaration(&mut c, &d).unwrap();
    validate_for(&c, &m).unwrap();
    assert!(compatible_target(&d, &d.context).is_err());
    for conditioning in [
        Conditioning::InitialScenario,
        Conditioning::History {
            methodology: "imported history".into(),
        },
    ] {
        let mut d = d.clone();
        d.conditioning = conditioning;
        assert!(compatible_target(&d, &d.context).is_err());
    }
}

#[test]
fn empirical_metadata_and_sample_counts_are_bounded_before_admission() {
    let (mut c, _, mut d) = fixture();
    d.conditioning = Conditioning::Observation {
        methodology: "x".repeat(MAX_EMPIRICAL_METADATA_BYTES),
    };
    if let Record::Empirical(e) = &mut c.record {
        e.evaluation.conditioning = d.conditioning.clone();
    }
    assert!(attach_declaration(&mut c, &d).is_err());
    let (mut c, _, d) = fixture();
    let n = MAX_EMPIRICAL_SAMPLES + 1;
    c.arrays
        .insert("returns".into(), Array::f64(vec![n], vec![0.; n]).unwrap());
    c.arrays.insert(
        "seeds".into(),
        Array::u64(vec![n], (0..n as u64).collect()).unwrap(),
    );
    assert!(attach_declaration(&mut c, &d).is_err());
    let (mut c, _, d) = fixture();
    if let Record::Empirical(e) = &mut c.record {
        e.evaluation.conditioning = Conditioning::Observation {
            methodology: "".into(),
        };
    }
    let mut d = d;
    d.conditioning = Conditioning::Observation {
        methodology: "".into(),
    };
    assert!(attach_declaration(&mut c, &d).is_err());
}

#[test]
fn authoritative_revision_offset_and_optional_observation_width_are_checked() {
    let (mut chunk, mut manifest, d) = fixture();
    attach_declaration(&mut chunk, &d).unwrap();
    manifest.starting_revision = 1;
    assert!(
        validate_for(&chunk, &manifest).is_err(),
        "run revision offset cannot reinterpret a measured revision"
    );
    let (mut chunk, _, mut d) = fixture();
    d.conditioning = Conditioning::InitialScenario;
    if let Record::Empirical(e) = &mut chunk.record {
        e.evaluation.conditioning = d.conditioning.clone();
    }
    chunk.arrays.insert(
        "observation".into(),
        Array::f32(vec![1, 1], vec![0.5]).unwrap(),
    );
    assert!(
        attach_declaration(&mut chunk, &d).is_err(),
        "any supplied observation must match context width"
    );
}

#[test]
fn declared_discrete_first_action_retains_canonical_type_and_effective_legal_mask() {
    let (mut c, mut m, mut d) = fixture();
    let space = ActionSpace::Discrete { count: 3 };
    m.action_space = space.clone();
    d.context.action_space = space;
    c.arrays
        .insert("action".into(), Array::u64(vec![1], vec![1]).unwrap());
    c.arrays
        .insert("mask".into(), Array::u8(vec![1, 3], vec![1, 1, 0]).unwrap());
    if let Record::Empirical(e) = &mut c.record {
        e.conditioning_mask = Some("mask".into());
    }
    attach_declaration(&mut c, &d).unwrap();
    validate_for(&c, &m).unwrap();
    for action in [2, 3, u64::MAX] {
        let mut bad = c.clone();
        bad.arrays
            .insert("action".into(), Array::u64(vec![1], vec![action]).unwrap());
        assert!(validate_for(&bad, &m).is_err());
    }
    let mut bad = c.clone();
    bad.arrays
        .insert("action".into(), Array::f32(vec![1, 1], vec![1.]).unwrap());
    assert!(validate_for(&bad, &m).is_err());
    let mut bad = c.clone();
    bad.arrays
        .insert("mask".into(), Array::u8(vec![1, 2], vec![1, 1]).unwrap());
    assert!(validate_for(&bad, &m).is_err());
}

#[test]
fn declaration_charge_includes_actual_string_capacity_before_cloning() {
    let (_, _, mut d) = fixture();
    let mut reserved = String::with_capacity(256 * 1024);
    reserved.push_str(&d.run_id);
    let capacity = reserved.capacity();
    d.run_id = reserved;
    assert!(
        d.retained_charge() >= capacity,
        "metadata ownership is allocation capacity, not encoded length"
    );
}
