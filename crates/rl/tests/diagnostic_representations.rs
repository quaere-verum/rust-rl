#![cfg(feature = "diagnostics")]
use rl::diagnostics::{representations::*, schema::*};
use std::collections::BTreeMap;

#[test]
fn pca_has_known_variance_and_reuses_its_center_and_axes() {
    // Population covariance diag(2, 1/2), sample eigenvalues 8/3 and 2/3.
    let x = [-2., 0., 2., 0., 0., -1., 0., 1.];
    let basis = Pca::fit(&x, 4, 2, 2, RepresentationLimits::default()).unwrap();
    assert!((basis.explained_variance()[0] - 8. / 3.).abs() < 1e-12);
    assert!((basis.explained_variance()[1] - 2. / 3.).abs() < 1e-12);
    assert!((basis.explained_fraction()[0] - 0.8).abs() < 1e-12);
    let a = basis.project(&x, 4).unwrap();
    let shifted: Vec<_> = x.chunks(2).flat_map(|r| [r[0] + 10., r[1] - 3.]).collect();
    let b = basis.project(&shifted, 4).unwrap();
    for i in 0..4 {
        assert!((b[i * 2] - a[i * 2] - 10.).abs() < 1e-12);
        assert!((b[i * 2 + 1] - a[i * 2 + 1] + 3.).abs() < 1e-12);
    }
    assert!(Pca::fit(&[1.; 8], 4, 2, 2, RepresentationLimits::default()).is_err());
    assert!(basis.project(&[0.; 6], 4).is_err());
}
#[test]
fn wide_pca_and_linear_cka_match_independent_gram_fixtures() {
    let x = [-1., 0., 0., 0., 1., 0., 1., -1., 0.];
    let p = Pca::fit(&x, 3, 3, 2, RepresentationLimits::default()).unwrap();
    assert!((p.explained_variance().iter().sum::<f64>() - 2.).abs() < 1e-12);
    let x = [-1., 0., 1., 0., 0., -1., 0., 1.];
    let rotated = [5., 2., 5., -2., 7., 0., 3., 0.]; // scaled, rotated, translated
    let score = linear_cka(&x, 2, &rotated, 2, 4, RepresentationLimits::default()).unwrap();
    assert!((score - 1.).abs() < 1e-12);
    let rank_one = [-1., 1., 0., 0.];
    let score = linear_cka(&x, 2, &rank_one, 1, 4, RepresentationLimits::default()).unwrap();
    assert!((score - 1. / 2f64.sqrt()).abs() < 1e-12);
    assert!(linear_cka(&x, 2, &[0.; 4], 1, 4, RepresentationLimits::default()).is_err());
    let limits = RepresentationLimits {
        max_work_elements: 1,
        ..Default::default()
    };
    assert!(Pca::fit(&x, 4, 2, 2, limits).is_err());
}
fn probes(order: &[usize], population: ProbePopulation) -> Chunk {
    Chunk {
        context: RecordContext::new("test", 0),
        record: Record::ProbeSet(ProbeSetRecord {
            id: "fixed".into(),
            population,
            methodology: "grid".into(),
            observations: "x".into(),
            action_masks: Some("m".into()),
            fixed_actions: None,
            row_ids: "id".into(),
            histories: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([
            (
                "x".into(),
                Array::f32(vec![2, 1], order.iter().map(|&i| i as f32).collect()).unwrap(),
            ),
            (
                "m".into(),
                Array::u8(
                    vec![2, 2],
                    order
                        .iter()
                        .flat_map(|&i| [u8::from(i == 0), u8::from(i == 1)])
                        .collect(),
                )
                .unwrap(),
            ),
            (
                "id".into(),
                Array::u64(vec![2], order.iter().map(|&i| i as u64 + 9).collect()).unwrap(),
            ),
        ]),
    }
}
#[test]
fn representation_comparison_matches_reordered_inputs_and_rejects_changed_population() {
    let a = probes(&[0, 1], ProbePopulation::FixedReference);
    let mut b = probes(&[1, 0], ProbePopulation::FixedReference);
    assert_eq!(match_reference_sets(&a, &b).unwrap(), vec![1, 0]);
    b.arrays
        .insert("x".into(), Array::f32(vec![2, 1], vec![5., 0.]).unwrap());
    assert!(match_reference_sets(&a, &b).is_err());
    assert!(match_reference_sets(&a, &probes(&[0, 1], ProbePopulation::PolicyVisited)).is_err());
}
#[test]
fn policy_kl_has_explicit_direction_and_zero_probability_semantics() {
    assert!(
        (categorical_kl(&[0.5, 0.5], &[0.25, 0.75]).unwrap() - 0.5 * (4. / 3f64).ln()).abs()
            < 1e-12
    );
    assert_ne!(
        categorical_kl(&[0.5, 0.5], &[0.25, 0.75]).unwrap(),
        categorical_kl(&[0.25, 0.75], &[0.5, 0.5]).unwrap()
    );
    assert!(categorical_kl(&[1., 0.], &[0., 1.]).unwrap().is_infinite());
    assert_eq!(categorical_kl(&[1., 0.], &[1., 0.]).unwrap(), 0.);
    assert!((diagonal_gaussian_kl(&[0.], &[0.], &[2.], &[0.]).unwrap() - 2.).abs() < 1e-12);
    assert!(categorical_kl(&[1., 1.], &[0.5, 0.5]).is_err());
}
