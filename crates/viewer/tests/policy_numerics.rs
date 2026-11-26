use rl::{core::action::ActionSpace, diagnostics::schema::*};
use rl_viewer::{policy::numerics::*, policy_change::PolicyBatch};
use std::collections::BTreeMap;
fn record(
    actions: Array,
    parameters: PolicyParameters,
    arrays: Vec<(&str, Array)>,
    mask: Option<Vec<u8>>,
) -> Chunk {
    let mut arrays: BTreeMap<_, _> = arrays.into_iter().map(|(k, a)| (k.into(), a)).collect();
    arrays.insert("a".into(), actions);
    let masks = mask.map(|m| {
        let n = m.len();
        arrays.insert("mask".into(), Array::u8(vec![1, n], m).unwrap());
        "mask".into()
    });
    Chunk {
        context: RecordContext::new("run", 1),
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters,
            stochastic: false,
            masks,
            entropy: None,
            entropy_method: None,
        }),
        arrays,
    }
}
fn categorical(p: Vec<f64>, mask: Option<Vec<u8>>) -> Chunk {
    record(
        Array::u64(
            vec![1],
            vec![p.iter().position(|&v| v > 0.).unwrap_or(0) as u64],
        )
        .unwrap(),
        PolicyParameters::Categorical {
            probabilities: "p".into(),
        },
        vec![("p", Array::f64(vec![1, p.len()], p).unwrap())],
        mask,
    )
}
#[test]
fn projection_uses_actual_actions_and_full_family_even_in_deterministic_sampling_mode() {
    let space = ActionSpace::Box {
        low: vec![-2., -2.],
        high: vec![2., 2.],
    };
    let c = record(
        Array::f32(vec![1, 2], vec![1.25, -0.75]).unwrap(),
        PolicyParameters::Deterministic,
        vec![],
        None,
    );
    let b = PolicyBatch::new(&c, None, &space).unwrap();
    assert_eq!(project(&b, 0, 1).unwrap(), -0.75);
    assert!(matches!(
        selected_distribution(&b, 0, 1).unwrap(),
        SelectedDistribution::Deterministic
    ));
    assert_eq!(current_reference_kl(&b, 0, &b, 0).unwrap(), None);
    assert!(project(&b, 0, 2).is_err());
    assert!(project(&b, 1, 0).is_err());
    let space = ActionSpace::Discrete { count: 70 };
    let p = vec![1. / 70.; 70];
    let c = categorical(p.clone(), Some(vec![1; 70]));
    let b = PolicyBatch::new(&c, None, &space).unwrap();
    assert_eq!(project(&b, 0, 69).unwrap(), p[69]);
    match selected_distribution(&b, 0, 69).unwrap() {
        SelectedDistribution::Categorical {
            probabilities,
            valid,
        } => {
            assert_eq!(probabilities, p);
            assert_eq!(valid, vec![true; 70]);
        }
        _ => panic!(),
    }
    let c = record(
        Array::u64(vec![1], vec![69]).unwrap(),
        PolicyParameters::Deterministic,
        vec![],
        None,
    );
    let b = PolicyBatch::new(&c, None, &space).unwrap();
    assert_eq!(project(&b, 0, 69).unwrap(), 1.);
    assert_eq!(project(&b, 0, 0).unwrap(), 0.);
}
#[test]
fn categorical_kl_is_current_to_reference_and_preserves_true_support_infinity() {
    let space = ActionSpace::Discrete { count: 2 };
    let a = categorical(vec![0.25, 0.75], None);
    let b = categorical(vec![0.5, 0.5], None);
    let a = PolicyBatch::new(&a, None, &space).unwrap();
    let b = PolicyBatch::new(&b, None, &space).unwrap();
    assert!(
        (current_reference_kl(&a, 0, &b, 0).unwrap().unwrap() - 0.130_812_035_941_136_97).abs()
            < 1e-14
    );
    assert!(
        (current_reference_kl(&b, 0, &a, 0).unwrap().unwrap() - 0.143_841_036_225_890_45).abs()
            < 1e-14
    );
    let zero = categorical(vec![0., 1.], Some(vec![0, 1]));
    let zero = PolicyBatch::new(&zero, None, &space).unwrap();
    assert_eq!(project(&zero, 0, 0).unwrap(), 0.);
    assert!(
        current_reference_kl(&b, 0, &zero, 0)
            .unwrap()
            .unwrap()
            .is_infinite()
    );
    assert!(
        PolicyBatch::new(&categorical(vec![0.1, 0.9], Some(vec![0, 1])), None, &space).is_err()
    );
}
#[test]
fn gaussian_shared_and_per_row_std_and_full_dimension_kl_are_preserved() {
    let gaussian = |per_row: bool, bounded: bool| {
        record(
            Array::f64(vec![2, 2], vec![0.; 4]).unwrap(),
            PolicyParameters::Gaussian {
                mean: "m".into(),
                log_std: "s".into(),
                tanh_bounded: bounded,
            },
            vec![
                ("m", Array::f64(vec![2, 2], vec![0., 0., 0.7, 2.]).unwrap()),
                (
                    "s",
                    Array::f64(
                        if per_row { vec![2, 2] } else { vec![2] },
                        if per_row {
                            vec![0., 0., 0., 5f64.ln()]
                        } else {
                            vec![0., 5f64.ln()]
                        },
                    )
                    .unwrap(),
                ),
            ],
            None,
        )
    };
    let space = ActionSpace::Unbounded { dimensions: 2 };
    let c = gaussian(true, false);
    let b = PolicyBatch::new(&c, None, &space).unwrap();
    assert_eq!(project(&b, 1, 1).unwrap(), 2.);
    assert!(
        (current_reference_kl(&b, 1, &b, 0).unwrap().unwrap() - 12.635_562_087_565_9).abs() < 1e-12
    );
    let bounded = ActionSpace::Box {
        low: vec![-1.; 2],
        high: vec![1.; 2],
    };
    let c = gaussian(false, true);
    let b = PolicyBatch::new(&c, None, &bounded).unwrap();
    let SelectedDistribution::Gaussian(g) = selected_distribution(&b, 1, 1).unwrap() else {
        panic!()
    };
    assert_eq!(g.mean, 2.);
    assert_eq!(g.log_std, 5f64.ln());
    assert_eq!(g.bounds, Some([-1., 1.]));
    assert!(project(&b, 1, 1).unwrap() < 0.4);
    let other = ActionSpace::Box {
        low: vec![-2.; 2],
        high: vec![2.; 2],
    };
    let other = PolicyBatch::new(&c, None, &other).unwrap();
    assert!(current_reference_kl(&b, 1, &other, 1).is_err());
    let d = record(
        Array::f64(vec![1, 2], vec![0., 0.]).unwrap(),
        PolicyParameters::Deterministic,
        vec![],
        None,
    );
    assert!(
        current_reference_kl(&b, 1, &PolicyBatch::new(&d, None, &bounded).unwrap(), 0).is_err()
    );
}
#[test]
fn categorical_kl_uses_actions_beyond_the_preview_limit() {
    let space = ActionSpace::Discrete { count: 70 };
    let mut p = vec![0.; 70];
    p[68] = 0.25;
    p[69] = 0.75;
    let mut q = vec![0.; 70];
    q[68] = 0.5;
    q[69] = 0.5;
    let p = categorical(p, None);
    let q = categorical(q, None);
    let p = PolicyBatch::new(&p, None, &space).unwrap();
    let q = PolicyBatch::new(&q, None, &space).unwrap();
    assert!(
        (current_reference_kl(&p, 0, &q, 0).unwrap().unwrap() - 0.130_812_035_941_136_97).abs()
            < 1e-14
    );
}

#[test]
fn selected_categorical_admits_actual_capacities_just_above_a_power_of_two() {
    let count = 65_537;
    let space = ActionSpace::Discrete { count };
    let chunk = categorical(vec![1. / count as f64; count], Some(vec![1; count]));
    let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
    let SelectedDistribution::Categorical {
        probabilities,
        valid,
    } = selected_distribution(&batch, 0, count - 1).unwrap()
    else {
        panic!()
    };
    assert_eq!(probabilities.len(), count);
    assert_eq!(valid.len(), count);
    assert_eq!(probabilities.capacity(), count);
    assert!(
        probabilities.capacity() * std::mem::size_of::<f64>() + valid.capacity()
            <= MAX_DISTRIBUTION_BYTES
    );
}

#[test]
fn deterministic_kl_does_not_require_a_stochastic_action_width_allocation() {
    let space = ActionSpace::Discrete { count: 131_073 };
    let chunk = record(
        Array::u64(vec![1], vec![131_072]).unwrap(),
        PolicyParameters::Deterministic,
        vec![],
        None,
    );
    let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
    assert_eq!(current_reference_kl(&batch, 0, &batch, 0).unwrap(), None);
    assert!(current_reference_kl(&batch, 1, &batch, 0).is_err());
    let other = ActionSpace::Discrete { count: 131_074 };
    let other = PolicyBatch::new(&chunk, None, &other).unwrap();
    assert!(current_reference_kl(&batch, 0, &other, 0).is_err());
}

#[test]
fn effective_mask_budget_covers_simultaneous_probe_and_policy_capacities() {
    use rl::{core::env::Observation, diagnostics::probes::ProbeSet};
    let space = ActionSpace::Discrete { count: 4 };
    let mut chunk = categorical(vec![0.5, 0., 0.5, 0.], Some(vec![1, 0, 1, 0]));
    chunk.context.probe_set = Some("p".into());
    let probes = ProbeSet::new(
        "p",
        ProbePopulation::FixedReference,
        "fixed",
        vec![Observation::dense(
            vec![0.],
            Some(vec![true, false, true, false]),
        )],
        None,
        space.clone(),
    )
    .unwrap()
    .record(chunk.context.clone())
    .unwrap();
    let batch = PolicyBatch::new(&chunk, Some(&probes), &space).unwrap();
    assert!(batch.effective_mask_with_budget(0, 7).is_err());
    let valid = batch.effective_mask_with_budget(0, 8).unwrap().unwrap();
    assert_eq!(valid, vec![true, false, true, false]);
    assert_eq!(valid.capacity(), 4);
    assert_eq!(batch.effective_mask(0).unwrap().unwrap(), valid);
    assert!(batch.effective_mask_with_budget(1, 8).is_err());
    let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
    assert!(batch.effective_mask_with_budget(0, 3).is_err());
    assert_eq!(
        batch.effective_mask_with_budget(0, 4).unwrap().unwrap(),
        valid
    );
    let chunk = categorical(vec![0.25; 4], None);
    let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
    assert_eq!(batch.effective_mask_with_budget(0, 0).unwrap(), None);
}
