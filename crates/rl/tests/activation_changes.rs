#![cfg(feature = "diagnostics")]
use rl::diagnostics::{representations::*, schema::*};

#[test]
fn streamed_drift_matches_hand_computed_norms_and_row_permutation() {
    let reference = Array::f64(vec![2, 2], vec![1., 2., 3., 4.]).unwrap();
    let current = Array::f64(vec![2, 2], vec![3., 6., 2., 0.]).unwrap();
    let diff = ActivationDifference::new(&current, &reference, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(diff.value(1, 0), 1.);
    assert_eq!(diff.value(1, 1), -2.);
    let drift = diff.drift().unwrap();
    assert!((drift.difference_norm - 3.).abs() < 1e-12);
    assert!((drift.reference_norm - 30f64.sqrt()).abs() < 1e-12);
    assert!((drift.relative - 3. / (30f64.sqrt() + 1e-12)).abs() < 1e-12);
    assert_eq!(drift.state_norms, vec![2., 5f64.sqrt()]);
    assert!(!drift.near_zero_reference);
}

#[test]
fn coordinate_drift_and_geometry_are_complementary() {
    let reference = Array::f64(vec![3, 2], vec![1., 0., 0., 1., -1., 0.]).unwrap();
    let current = Array::f64(vec![3, 2], vec![2., 0., 0., 2., -2., 0.]).unwrap();
    let d = ActivationDifference::new(&current, &reference, &[0, 1, 2], &[0, 1])
        .unwrap()
        .drift()
        .unwrap();
    assert!((d.relative - 3f64.sqrt() / (3f64.sqrt() + 1e-12)).abs() < 1e-12);
    assert!(
        (linear_cka(
            reference.as_f64().unwrap(),
            2,
            current.as_f64().unwrap(),
            2,
            3,
            RepresentationLimits::default()
        )
        .unwrap()
            - 1.)
            .abs()
            < 1e-12
    );
    let zero = Array::f64(vec![3, 2], vec![0.; 6]).unwrap();
    assert!(
        ActivationDifference::new(&current, &zero, &[0, 1, 2], &[0, 1])
            .unwrap()
            .drift()
            .unwrap()
            .near_zero_reference
    );
    assert!(
        linear_cka(
            zero.as_f64().unwrap(),
            2,
            zero.as_f64().unwrap(),
            2,
            3,
            RepresentationLimits::default()
        )
        .is_err()
    );
    assert!(ActivationDifference::new(&current, &reference, &[0, 0, 2], &[0, 1]).is_err());
    assert!(ActivationDifference::new(&current, &reference, &[0, 1, 2], &[0]).is_err());
}

#[test]
fn absent_legacy_lineage_is_unknown() {
    let meta: ActivationRecord =
        serde_json::from_str(r#"{"network":"policy","layers":{},"summaries":{}}"#).unwrap();
    assert!(matches!(meta.inputs, ActivationInputs::Unknown));
}

#[cfg(feature = "neural")]
#[test]
fn checked_mlp_input_records_exact_actions_and_preserves_policy_identity() {
    use rand::SeedableRng;
    use rl::{
        core::{
            action::{Action, ActionSpace},
            env::Observation,
        },
        diagnostics::{activation_inputs::*, probes::*},
        nn::{self, network::Mlp},
    };
    let space = ActionSpace::Box {
        low: vec![-2.],
        high: vec![2.],
    };
    let set = ProbeSet::new(
        "grid",
        ProbePopulation::FixedReference,
        "test grid",
        vec![Observation::new(vec![1.]), Observation::new(vec![2.])],
        None,
        space.clone(),
    )
    .unwrap();
    let mut context = RecordContext::new("run", 4);
    context.probe_set = Some("grid".into());
    let source = RecordContext::new("run", 2).identity;
    let model = Mlp::new(2, &[], 1, &mut rand_chacha::ChaCha8Rng::seed_from_u64(9)).unwrap();
    let actions = vec![
        Action::Continuous(vec![0.5]),
        Action::Continuous(vec![-0.25]),
    ];
    let chunk = mlp_activations(
        context.clone(),
        "arbitrary name",
        &model,
        ActivationProbeInput::PolicyActions {
            probes: &set,
            actions: &actions,
            identity: source,
            model: "actor".into(),
            stochastic: false,
        },
        &["layers.0.pre"],
    )
    .unwrap();
    let raw = activations(
        context.clone(),
        "arbitrary name",
        &model,
        nn::matrix(vec![1., 0.5, 2., -0.25], 2, 2),
        &["layers.0.pre"],
    )
    .unwrap();
    assert_eq!(
        chunk.arrays["layers.0.pre"].as_f32().unwrap(),
        raw.arrays["layers.0.pre"].as_f32().unwrap()
    );
    let Record::Activations(meta) = &chunk.record else {
        panic!()
    };
    let ActivationInputs::ObservationsAndActions {
        actions: key,
        source: ActivationActionSource::Policy { identity, .. },
    } = &meta.inputs
    else {
        panic!()
    };
    assert_eq!(identity.learner_update_revision, 2);
    assert_eq!(chunk.arrays[key].as_f32().unwrap(), &[0.5, -0.25]);
    let probes = set.record(context.clone()).unwrap();
    validate_activation_inputs(&chunk, &probes, 1, &space).unwrap();
    let fixed_set = ProbeSet::new(
        "grid",
        ProbePopulation::FixedReference,
        "fixed-action fixture",
        set.observations().to_vec(),
        Some(actions.clone()),
        space.clone(),
    )
    .unwrap();
    let fixed = mlp_activations(
        context.clone(),
        "critic",
        &model,
        ActivationProbeInput::FixedActions(&fixed_set),
        &["layers.0.pre"],
    )
    .unwrap();
    let fixed_probes = fixed_set.record(context.clone()).unwrap();
    validate_activation_inputs(&fixed, &fixed_probes, 1, &space).unwrap();
    assert_eq!(
        fixed.arrays["layers.0.pre"].as_f32().unwrap(),
        raw.arrays["layers.0.pre"].as_f32().unwrap()
    );
    assert_eq!(
        compare_activation_inputs(&fixed, &fixed, &[0, 1]).unwrap(),
        InputEquality::Equal
    );
    let mut wrong_fixed = fixed.clone();
    wrong_fixed.arrays.insert(
        key.clone(),
        Array::f32(vec![2, 1], vec![0.75, -0.25]).unwrap(),
    );
    assert!(validate_activation_inputs(&wrong_fixed, &fixed_probes, 1, &space).is_err());
    {
        use rl::diagnostics::storage::*;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("schema3");
        let manifest = RunManifest::new("run", "test", "test", 1, space.clone(), 0.9).unwrap();
        let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
        writer.append(&probes).unwrap();
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        let recording = Recording::open(&path, ReadLimits::default()).unwrap();
        let decoded = recording.load(1).unwrap();
        assert_eq!(decoded.arrays[key].as_f32().unwrap(), &[0.5, -0.25]);
        assert!(matches!(
            decoded.record,
            Record::Activations(ActivationRecord {
                inputs: ActivationInputs::ObservationsAndActions { .. },
                ..
            })
        ));
    }
    assert_eq!(
        compare_activation_inputs(&chunk, &chunk, &[0, 1]).unwrap(),
        InputEquality::Equal
    );
    assert_eq!(
        compare_activation_inputs(&raw, &chunk, &[0, 1]).unwrap(),
        InputEquality::Unknown
    );
    let mut changed = chunk.clone();
    changed.arrays.insert(
        key.clone(),
        Array::f32(vec![2, 1], vec![0.75, -0.25]).unwrap(),
    );
    assert_eq!(
        compare_activation_inputs(&changed, &chunk, &[0, 1]).unwrap(),
        InputEquality::ChangedPolicy
    );
    assert!(
        mlp_activations(
            context.clone(),
            "policy",
            &model,
            ActivationProbeInput::Observations(&set),
            &["layers.0.pre"]
        )
        .is_err()
    );
    assert!(
        mlp_activations(
            context.clone(),
            "critic",
            &model,
            ActivationProbeInput::FixedActions(&set),
            &["layers.0.pre"]
        )
        .is_err()
    );
    changed.arrays.remove(key);
    assert!(changed.validate().is_err());
    let mut imported = chunk.clone();
    let Record::Activations(meta) = &mut imported.record else {
        panic!()
    };
    let ActivationInputs::ObservationsAndActions { source, .. } = &mut meta.inputs else {
        panic!()
    };
    *source = ActivationActionSource::FixedProbe;
    assert!(validate_activation_inputs(&imported, &probes, 1, &space).is_err());
    assert!(validate_activation_inputs(&chunk, &probes, 2, &space).is_err());
    imported
        .arrays
        .insert(key.clone(), Array::u8(vec![2, 1], vec![0, 0]).unwrap());
    assert!(imported.validate().is_err());
    context.probe_set = Some("wrong".into());
    assert!(
        mlp_activations(
            context,
            "critic",
            &model,
            ActivationProbeInput::Observations(&set),
            &["layers.0.pre"]
        )
        .is_err()
    );
}

#[test]
fn orthogonal_cka_and_constant_drift_are_explicit() {
    let a = [-1., -1., 1., 1.];
    let b = [-1., 1., -1., 1.];
    assert!(
        linear_cka(&a, 1, &b, 1, 4, RepresentationLimits::default())
            .unwrap()
            .abs()
            < 1e-12
    );
    let a = Array::f64(vec![4, 1], a.to_vec()).unwrap();
    let b = Array::f64(vec![4, 1], b.to_vec()).unwrap();
    let d = ActivationDifference::new(&a, &b, &[0, 1, 2, 3], &[0])
        .unwrap()
        .drift()
        .unwrap();
    assert_eq!(d.reference_norm, 2.);
    assert!((d.difference_norm - 8f64.sqrt()).abs() < 1e-12);
    let constant = Array::f64(vec![4, 1], vec![2.; 4]).unwrap();
    assert_eq!(
        ActivationDifference::new(&constant, &constant, &[0, 1, 2, 3], &[0])
            .unwrap()
            .drift()
            .unwrap()
            .relative,
        0.
    );
}
