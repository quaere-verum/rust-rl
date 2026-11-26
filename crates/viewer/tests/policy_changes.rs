use rl::{core::action::ActionSpace, diagnostics::schema::*};
use rl_viewer::{
    policy::numerics::{current_reference_kl, project},
    policy_change::PolicyBatch,
};
use std::collections::BTreeMap;
fn record(actions: Vec<f64>, stochastic: bool) -> Chunk {
    let mut context = RecordContext::new("run", 0);
    context.probe_set = Some("p".into());
    Chunk {
        context,
        record: Record::Policy(PolicyRecord {
            actions: "a".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic,
        }),
        arrays: BTreeMap::from([(
            "a".into(),
            Array::f64(vec![1, actions.len()], actions).unwrap(),
        )]),
    }
}
#[test]
fn full_vector_norm_and_sampled_gaussian_parameters_define_policy_change() {
    let a = record(vec![4., 6.], false);
    let b = record(vec![1., 2.], false);
    let space = ActionSpace::Unbounded { dimensions: 2 };
    let a = PolicyBatch::new(&a, None, &space).unwrap();
    let b = PolicyBatch::new(&b, None, &space).unwrap();
    let norm = (0..2).fold(0f64, |norm, component| {
        norm.hypot(project(&a, 0, component).unwrap() - project(&b, 0, component).unwrap())
    });
    assert_eq!(norm, 5.);
    assert_eq!(current_reference_kl(&a, 0, &b, 0).unwrap(), None);
    assert!(
        PolicyBatch::new(
            &record(vec![0.], true),
            None,
            &ActionSpace::Unbounded { dimensions: 1 }
        )
        .is_err()
    );
    let mut a = record(vec![0.9], true);
    let mut b = record(vec![-0.9], true);
    for chunk in [&mut a, &mut b] {
        let Record::Policy(meta) = &mut chunk.record else {
            panic!()
        };
        meta.parameters = PolicyParameters::Gaussian {
            mean: "m".into(),
            log_std: "s".into(),
            tanh_bounded: true,
        };
        chunk
            .arrays
            .insert("m".into(), Array::f64(vec![1, 1], vec![0.]).unwrap());
        chunk
            .arrays
            .insert("s".into(), Array::f64(vec![1], vec![0.]).unwrap());
    }
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    assert_eq!(
        current_reference_kl(
            &PolicyBatch::new(&b, None, &space).unwrap(),
            0,
            &PolicyBatch::new(&a, None, &space).unwrap(),
            0,
        )
        .unwrap()
        .unwrap(),
        0.
    );
    b.arrays
        .insert("m".into(), Array::f64(vec![1, 1], vec![2.]).unwrap());
    assert_eq!(
        current_reference_kl(
            &PolicyBatch::new(&b, None, &space).unwrap(),
            0,
            &PolicyBatch::new(&a, None, &space).unwrap(),
            0,
        )
        .unwrap()
        .unwrap(),
        2.
    );
}

#[test]
fn categorical_change_distinguishes_true_support_infinity_from_tiny_probabilities() {
    let make = |p: Vec<f64>| {
        let mut chunk = record(vec![0.], true);
        chunk
            .arrays
            .insert("a".into(), Array::u64(vec![1], vec![0]).unwrap());
        chunk
            .arrays
            .insert("p".into(), Array::f64(vec![1, 2], p).unwrap());
        let Record::Policy(meta) = &mut chunk.record else {
            panic!()
        };
        meta.parameters = PolicyParameters::Categorical {
            probabilities: "p".into(),
        };
        chunk
    };
    let space = ActionSpace::Discrete { count: 2 };
    let reference = make(vec![0.5, 0.5]);
    let tiny = make(vec![1e-320, 1.]);
    let zero = make(vec![0., 1.]);
    let reference = PolicyBatch::new(&reference, None, &space).unwrap();
    assert!(
        current_reference_kl(
            &reference,
            0,
            &PolicyBatch::new(&tiny, None, &space).unwrap(),
            0
        )
        .unwrap()
        .unwrap()
        .is_finite()
    );
    assert!(
        current_reference_kl(
            &reference,
            0,
            &PolicyBatch::new(&zero, None, &space).unwrap(),
            0
        )
        .unwrap()
        .unwrap()
        .is_infinite()
    );
    assert!(PolicyBatch::new(&make(vec![-0.5, 1.5]), None, &space).is_err());
    assert!(PolicyBatch::new(&make(vec![0.1, 0.1]), None, &space).is_err());
    assert!(
        PolicyBatch::new(
            &make(vec![0.5, 0.5]),
            None,
            &ActionSpace::Discrete { count: 3 }
        )
        .is_err()
    );
}
