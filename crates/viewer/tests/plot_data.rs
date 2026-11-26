use rl::diagnostics::schema::*;
use rl_viewer::plot_data::{matching_input, row, trajectory_points};
use std::collections::BTreeMap;

#[test]
fn matching_uses_actual_inputs_and_masks_not_row_numbers() {
    let probe = Chunk {
        context: RecordContext::new("test", 3),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "fixed".into(),
            population: ProbePopulation::FixedReference,
            methodology: "fixture".into(),
            observations: "x".into(),
            action_masks: Some("m".into()),
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([
            (
                "x".into(),
                Array::f32(vec![2, 2], vec![1., 2., 1., 2.]).unwrap(),
            ),
            ("m".into(), Array::u8(vec![2, 2], vec![1, 0, 0, 1]).unwrap()),
            ("ids".into(), Array::u64(vec![2], vec![80, 90]).unwrap()),
        ]),
    };
    let mut source = probe.clone();
    source
        .arrays
        .insert("m".into(), Array::u8(vec![2, 2], vec![0, 1, 1, 0]).unwrap());
    assert_eq!(matching_input(&source, 0, &probe).unwrap(), Some(1));
    let mut unrestricted = source.clone();
    let Record::ProbeSet(meta) = &mut unrestricted.record else {
        panic!()
    };
    meta.action_masks = None;
    assert_eq!(
        matching_input(&unrestricted, 0, &probe).unwrap(),
        None,
        "an absent mask denotes full support and cannot match restricted support"
    );
    source.arrays.insert(
        "x".into(),
        Array::f32(vec![2, 2], vec![1., 3., 1., 2.]).unwrap(),
    );
    assert_eq!(matching_input(&source, 0, &probe).unwrap(), None);
    assert!(row(&probe, "missing", 0).is_err());
    assert!(row(&probe, "x", 2).is_err());
}

#[test]
fn anchored_inputs_rematch_reordered_rows_and_reject_changed_histories_or_actions() {
    let mut source = Chunk {
        context: RecordContext::new("test", 2),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "reused".into(),
            population: ProbePopulation::PolicyVisited,
            methodology: "fixture".into(),
            observations: "x".into(),
            action_masks: None,
            fixed_actions: Some("a".into()),
            row_ids: "ids".into(),
            histories: BTreeMap::from([("past".into(), "h".into())]),
        }),
        arrays: BTreeMap::from([
            ("x".into(), Array::f32(vec![2, 1], vec![4., 8.]).unwrap()),
            ("a".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
            ("h".into(), Array::f32(vec![2, 1], vec![3., 7.]).unwrap()),
            ("ids".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
        ]),
    };
    let mut later = source.clone();
    later
        .arrays
        .insert("x".into(), Array::f32(vec![2, 1], vec![8., 4.]).unwrap());
    later
        .arrays
        .insert("a".into(), Array::u64(vec![2], vec![1, 0]).unwrap());
    later
        .arrays
        .insert("h".into(), Array::f32(vec![2, 1], vec![7., 3.]).unwrap());
    if let Record::ProbeSet(p) = &mut later.record {
        p.action_masks = Some("m".into());
    }
    later
        .arrays
        .insert("m".into(), Array::u8(vec![2, 2], vec![1; 4]).unwrap());
    assert_eq!(matching_input(&source, 0, &later).unwrap(), Some(1));
    source
        .arrays
        .insert("h".into(), Array::f32(vec![2, 1], vec![0., 7.]).unwrap());
    assert_eq!(matching_input(&source, 0, &later).unwrap(), None);
    source
        .arrays
        .insert("h".into(), Array::f32(vec![2, 1], vec![3., 7.]).unwrap());
    source
        .arrays
        .insert("a".into(), Array::u64(vec![2], vec![1, 1]).unwrap());
    assert_eq!(matching_input(&source, 0, &later).unwrap(), None);
}

#[test]
fn physical_and_decision_axes_are_explicit_and_dimensions_checked() {
    let a = Array::f64(vec![3, 2], vec![1., 4., 2., 5., 3., 6.]).unwrap();
    assert_eq!(
        trajectory_points(&a, Some(&[0., 0.25, 8.]), 1).unwrap(),
        vec![[0., 4.], [0.25, 5.], [8., 6.]]
    );
    assert_eq!(
        trajectory_points(&a, None, 0).unwrap(),
        vec![[0., 1.], [1., 2.], [2., 3.]]
    );
    assert!(trajectory_points(&a, Some(&[0.]), 0).is_err());
    assert!(trajectory_points(&a, None, 2).is_err());
}

#[test]
fn imported_inputs_match_without_narrowing_dtypes_or_flattening_history_shapes() {
    let source = Chunk {
        context: RecordContext::new("test", 2),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "imported".into(),
            population: ProbePopulation::ImportedHistory,
            methodology: "fixture".into(),
            observations: "x".into(),
            action_masks: None,
            fixed_actions: None,
            row_ids: "ids".into(),
            histories: BTreeMap::from([("history".into(), "h".into())]),
        }),
        arrays: BTreeMap::from([
            ("x".into(), Array::f64(vec![1, 1], vec![0.1]).unwrap()),
            (
                "h".into(),
                Array::u64(vec![1, 1, 2], vec![9_007_199_254_740_992, 0]).unwrap(),
            ),
            ("ids".into(), Array::u64(vec![1], vec![0]).unwrap()),
        ]),
    };
    assert_eq!(matching_input(&source, 0, &source).unwrap(), Some(0));
    let mut target = source.clone();
    target
        .arrays
        .insert("x".into(), Array::f32(vec![1, 1], vec![0.1]).unwrap());
    assert_eq!(matching_input(&source, 0, &target).unwrap(), None);
    for history in [
        Array::u64(vec![1, 1, 2], vec![9_007_199_254_740_993, 0]).unwrap(),
        Array::u64(vec![1, 2], vec![9_007_199_254_740_992, 0]).unwrap(),
    ] {
        let mut target = source.clone();
        target.arrays.insert("h".into(), history);
        assert_eq!(matching_input(&source, 0, &target).unwrap(), None);
    }
}

#[test]
fn gaussian_shared_log_standard_deviations_broadcast_to_every_probe_row() {
    let chunk = Chunk {
        context: RecordContext::new("test", 1),
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Gaussian {
                mean: "mean".into(),
                log_std: "std".into(),
                tanh_bounded: false,
            },
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: true,
        }),
        arrays: BTreeMap::from([
            (
                "actions".into(),
                Array::f32(vec![3, 2], vec![0.; 6]).unwrap(),
            ),
            ("mean".into(), Array::f32(vec![3, 2], vec![0.; 6]).unwrap()),
            ("std".into(), Array::f32(vec![2], vec![-1., -2.]).unwrap()),
        ]),
    };
    chunk.validate().unwrap();
    for row in 0..3 {
        assert_eq!(
            rl_viewer::plot_data::gaussian_log_std(&chunk, "std", row).unwrap(),
            [-1., -2.]
        );
    }
}
