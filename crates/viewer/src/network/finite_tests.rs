use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::nn::{matrix, network::Mlp, vector};

// Known maps are set on the real MLP, not a product-only numerical constructor.
fn model(relu: bool) -> NumericalMlp {
    let mut m = Mlp::new(
        2,
        if relu { &[2] } else { &[] },
        2,
        &mut ChaCha8Rng::seed_from_u64(1),
    )
    .unwrap();
    for layer in &mut m.layers {
        layer.weight = layer
            .weight
            .clone()
            .map(|_| matrix(vec![1., 0., 0., 1.], 2, 2));
        layer.bias = layer.bias.take().map(|p| p.map(|_| vector(&[0., 0.])));
    }
    NumericalMlp::from_model(&m).unwrap()
}
fn request() -> Request {
    Request {
        network: "fixture".into(),
        anchor: vec![1., 1.],
        endpoint: vec![-1., 1.],
        probes: vec![vec![1., 1.], vec![-1., 1.]],
        axes: vec![0, 1],
        axis_scales: vec![2., 2.],
        ranges: [[-1., 2.], [-1., 2.]],
        direction_angle: 0.,
        layer: 1,
        grid: [2, 2],
        previous_basis: vec![],
        t: 1.,
        radius: 0.75,
        action_mask: None,
    }
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}

// A finite affine displacement must agree exactly; derivative magnitudes alone
// cannot establish this contract.
#[test]
fn affine_and_relu_within_region_have_zero_finite_error() {
    for relu in [false, true] {
        let mut r = request();
        r.endpoint = vec![2., 1.5];
        let f = prepare(&model(relu), &r).unwrap();
        assert_eq!(f.slice_anchor, [1., 1.]);
        assert_eq!(f.slice_position, [2., 1.5]);
        assert_eq!(f.slice_endpoint, [2., 1.5]);
        assert!(f.position_on_slice);
        for layer in f.comparisons {
            close(layer.error, 0.);
            close(layer.linear_norm, 1.25_f64.sqrt());
            close(layer.actual_norm, 1.25_f64.sqrt());
        }
    }
}

#[test]
fn relu_crossing_has_hand_checked_one_third_error() {
    let mut r = request();
    let net = model(true);
    let f = prepare(&net, &r).unwrap();
    close(f.comparisons[0].error, 0.);
    for layer in &f.comparisons[1..] {
        close(layer.linear_norm, 2.);
        close(layer.actual_norm, 1.);
        close(layer.residual_norm, 1.);
        close(layer.error, 1. / 3.);
    }
    r.t = 0.25; // x=(0.5,1), still inside the original active region.
    for layer in prepare(&net, &r).unwrap().comparisons {
        close(layer.error, 0.);
        close(layer.actual_norm, 0.5);
    }
}

#[test]
fn canonical_pair_keeps_orthogonal_residual_and_zero_directions() {
    let pair = finite::compare(
        &Metric::Identity,
        "hidden".into(),
        &[1., 0., 0.],
        &[1., 0., 1.],
    )
    .unwrap();
    close(pair.linear[0], 1.);
    close(pair.actual[0], 1.);
    close(pair.actual[1], 1.);
    close(pair.residual_norm, 1.);
    close(pair.error, 1. / (1. + 2_f64.sqrt()));
    for (linear, actual) in [
        ([0., 0.], [0., 0.]),
        ([0., 0.], [0., 2.]),
        ([2., 0.], [0., 0.]),
    ] {
        let p = finite::compare(&Metric::Identity, "zero".into(), &linear, &actual).unwrap();
        assert!(p.error.is_finite());
        close(p.linear[0].hypot(p.linear[1]), p.linear_norm);
        close(p.actual[0].hypot(p.actual[1]), p.actual_norm);
    }
    let mut r = request();
    r.t = 0.;
    for p in prepare(&model(true), &r).unwrap().comparisons {
        assert_eq!(p.linear, [0., 0.]);
        assert_eq!(p.actual, [0., 0.]);
        close(p.error, 0.);
    }
    r.t = 1.;
    r.endpoint = r.anchor.clone();
    for p in prepare(&model(true), &r).unwrap().comparisons {
        close(p.linear_norm, 0.);
        close(p.actual_norm, 0.);
        close(p.error, 0.);
    }
}

#[test]
fn canonical_pair_uses_the_same_regularized_metric() {
    let rows = [
        vec![0., 0., 0.],
        vec![2., 0., 0.],
        vec![0., 1., 1.],
        vec![2., 1., 1.],
    ];
    let metric = Metric::fit(rows.iter().map(Vec::as_slice), 3, &|| true).unwrap();
    let l = [1., -2., 0.5];
    let d = [-1., 3., 2.];
    let p = finite::compare(&metric, "metric".into(), &l, &d).unwrap();
    let dense = Metric::Dense(covariance_inverse(rows.iter().map(Vec::as_slice), 3).unwrap());
    let q = finite::compare(&dense, "metric".into(), &l, &d).unwrap();
    close(p.linear_norm, q.linear_norm);
    close(p.actual_norm, q.actual_norm);
    close(p.error, q.error);
    close(p.linear[0] * p.actual[0], metric.inner(&l, &d));
    close(p.actual[0].hypot(p.actual[1]), p.actual_norm);
}

#[test]
fn neighborhood_is_bounded_anchored_and_reused_on_layer_and_t_changes() {
    let mut r = request();
    let mut state = Preparation::new(model(true));
    let first = state.prepare(&r).unwrap();
    assert_eq!(first.neighborhood.positions.len(), 256);
    assert_eq!(first.neighborhood.errors.len(), 3);
    assert_eq!(first.neighborhood.bounds, [[-0.5, 2.5], [-0.5, 2.5]]);
    assert!(
        first.neighborhood.errors[0]
            .iter()
            .all(|e| e.is_some_and(|e| e < 1e-12))
    );
    assert!(
        first.neighborhood.errors[1]
            .iter()
            .flatten()
            .any(|&e| e > 0.)
    );
    assert!(
        first
            .neighborhood
            .errors
            .iter()
            .flatten()
            .flatten()
            .all(|e| (0. ..=1.).contains(e))
    );
    r.layer = 2;
    r.t = 0.25;
    let next = state.prepare(&r).unwrap();
    assert!(Arc::ptr_eq(&first.neighborhood, &next.neighborhood));
    r.radius = 0.25;
    let smaller = state.prepare(&r).unwrap();
    assert!(!Arc::ptr_eq(&first.neighborhood, &smaller.neighborhood));
    assert!(
        smaller
            .neighborhood
            .errors
            .iter()
            .flatten()
            .flatten()
            .all(|&e| e < 1e-12)
    );
}

#[test]
fn invalid_neighborhood_cells_remain_missing_and_cancelled_work_is_not_cached() {
    let mut r = request();
    r.anchor = vec![f64::MAX, 1.];
    r.axis_scales = vec![f64::MAX, 2.];
    r.radius = 1.;
    let net = model(false);
    let signal = net
        .trace_directions(&r.anchor, &[vec![1., 0.], vec![0., 1.]])
        .unwrap();
    let metrics: Vec<_> = (0..2)
        .map(|_| Metric::Dense(DMatrix::identity(2, 2) * 1e-310))
        .collect();
    let grid = finite::neighborhood(&net, &r, &signal, &metrics, &|| true).unwrap();
    let original = finite_original::neighborhood(&net, &r, &signal, &metrics, &|| true).unwrap();
    assert_original_neighborhood(&grid, &original);
    assert!(grid.errors[0].iter().any(Option::is_none));
    assert!(grid.errors[0].iter().any(Option::is_some));
    let checks = std::cell::Cell::new(0);
    assert!(
        finite::neighborhood(&net, &request(), &signal, &metrics, &|| {
            checks.set(checks.get() + 1);
            checks.get() < 4
        })
        .is_err()
    );
    assert_eq!(checks.get(), 4);
}

#[test]
fn finite_controls_reject_nonfinite_and_out_of_bounds_values() {
    let net = model(true);
    for t in [f64::NAN, -0.1, 1.1] {
        let mut r = request();
        r.t = t;
        assert!(prepare(&net, &r).is_err());
    }
    for radius in [f64::INFINITY, 0., 1.1] {
        let mut r = request();
        r.radius = radius;
        assert!(prepare(&net, &r).is_err());
    }
}

#[test]
fn cancelled_neighborhood_replacement_is_never_cached() {
    let mut state = Preparation::new(model(true));
    let mut r = request();
    state.prepare(&r).unwrap();
    r.radius = 0.125;
    let checks = std::cell::Cell::new(0);
    assert!(
        state
            .prepare_cancellable(&r, &|| {
                checks.set(checks.get() + 1);
                checks.get() < 20
            })
            .is_err()
    );
    assert!(state.neighborhood.is_none());
    assert_eq!(state.prepare(&r).unwrap().neighborhood.positions.len(), 256);
}

#[test]
fn affine_neighborhood_uses_the_actual_representable_input_displacement() {
    let mut r = request();
    r.anchor = vec![9_007_199_254_740_992., 1.]; // 2^53: positive sub-unit offsets round away.
    r.endpoint = r.anchor.clone();
    r.axis_scales = vec![1., 1.];
    let net = model(false);
    let signal = net
        .trace_directions(&r.anchor, &[vec![1., 0.], vec![0., 1.]])
        .unwrap();
    let metrics = vec![Metric::Identity, Metric::Identity];
    assert_original_neighborhood(
        &finite::neighborhood(&net, &r, &signal, &metrics, &|| true).unwrap(),
        &finite_original::neighborhood(&net, &r, &signal, &metrics, &|| true).unwrap(),
    );
    let frame = prepare(&net, &r).unwrap();
    assert!(
        frame
            .neighborhood
            .errors
            .iter()
            .flatten()
            .all(|e| e.is_some_and(|e| e < 1e-12)),
        "an affine map acquired error from an ideal offset that was lost while constructing x: {:?}",
        frame.neighborhood.errors[0]
    );
}

// These comparisons catch changed reduction order, canonical alignment and
// invalid-cell masks; the oracle is the frozen pre-scratch implementation.
fn comparison_bits(p: &finite::LayerComparison) -> Vec<u64> {
    p.linear
        .iter()
        .chain(&p.actual)
        .copied()
        .chain([p.linear_norm, p.actual_norm, p.residual_norm, p.error])
        .map(f64::to_bits)
        .collect()
}
fn assert_original_neighborhood(
    actual: &finite::Neighborhood,
    expected: &finite_original::Neighborhood,
) {
    assert_eq!(
        actual.bounds.map(|r| r.map(f64::to_bits)),
        expected.bounds.map(|r| r.map(f64::to_bits))
    );
    assert_eq!(actual.positions.len(), expected.positions.len());
    for (a, b) in actual.positions.iter().zip(&expected.positions) {
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    }
    assert_eq!(actual.errors.len(), expected.errors.len());
    for (a, b) in actual.errors.iter().zip(&expected.errors) {
        assert_eq!(
            a.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>(),
            b.iter().map(|x| x.map(f64::to_bits)).collect::<Vec<_>>()
        );
    }
}

#[test]
fn finite_comparisons_match_frozen_original_bits_for_every_metric_and_zero_tangent() {
    let rows: Vec<Vec<_>> = (0..3)
        .map(|i| (0..24).map(|j| ((i * 31 + j * 7) as f64).sin()).collect())
        .collect();
    let metrics = [
        Metric::Identity,
        Metric::Dense(covariance_inverse(rows.iter().map(Vec::as_slice), 24).unwrap()),
        Metric::fit(rows.iter().map(Vec::as_slice), 24, &|| true).unwrap(),
    ];
    assert!(matches!(metrics[2], Metric::LowRank { .. }));
    for metric in &metrics {
        for zero in [false, true] {
            let linear: Vec<_> = (0..24)
                .map(|j| if zero { -0. } else { (j as f64).cos() })
                .collect();
            let actual: Vec<_> = (0..24).map(|j| (j as f64).sin()).collect();
            let a = finite::compare(metric, "pair".into(), &linear, &actual).unwrap();
            let b = finite_original::compare(metric, "pair".into(), &linear, &actual).unwrap();
            assert_eq!(a.name, b.name);
            let expected = b
                .linear
                .iter()
                .chain(&b.actual)
                .copied()
                .chain([b.linear_norm, b.actual_norm, b.residual_norm, b.error])
                .map(f64::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(comparison_bits(&a), expected);
        }
        for (linear, actual) in [
            (vec![], vec![]),
            (vec![0.; 24], vec![0.; 23]),
            (vec![f64::NAN; 24], vec![0.; 24]),
        ] {
            assert_eq!(
                format!(
                    "{:?}",
                    finite::compare(metric, "pair".into(), &linear, &actual).unwrap_err()
                ),
                format!(
                    "{:?}",
                    finite_original::compare(metric, "pair".into(), &linear, &actual).unwrap_err()
                )
            );
        }
    }
}

#[test]
fn complete_neighborhood_matches_frozen_original_bits_and_cancellation_counts() {
    for widths in [&[24, 6][..], &[384, 24][..]] {
        let net = NumericalMlp::from_model(
            &Mlp::new(2, widths, 2, &mut ChaCha8Rng::seed_from_u64(42)).unwrap(),
        )
        .unwrap();
        let mut r = request();
        r.anchor = vec![-0., 0.];
        r.probes = vec![vec![-1., 0.], vec![0., 1.], vec![1., -1.]];
        let anchor = net
            .trace_directions(&r.anchor, &[vec![2., 0.], vec![0., 2.]])
            .unwrap();
        let samples: Vec<_> = r
            .probes
            .iter()
            .map(|p| net.forward_layers(p).unwrap())
            .collect();
        for kind in 0..3 {
            let metrics: Vec<_> = (0..=net.layer_count())
                .map(|l| {
                    let rows = samples.iter().map(|s| s[l].as_slice());
                    match kind {
                        0 => Metric::Identity,
                        1 => Metric::Dense(covariance_inverse(rows, samples[0][l].len()).unwrap()),
                        _ => Metric::fit(rows, samples[0][l].len(), &|| true).unwrap(),
                    }
                })
                .collect();
            if kind == 2 {
                assert!(matches!(metrics[1], Metric::LowRank { .. }));
            }
            for limit in [1, 4, 1279, 1281, usize::MAX] {
                let a_checks = std::cell::Cell::new(0);
                let b_checks = std::cell::Cell::new(0);
                let a = finite::neighborhood(&net, &r, &anchor, &metrics, &|| {
                    a_checks.set(a_checks.get() + 1);
                    a_checks.get() < limit
                });
                let b = finite_original::neighborhood(&net, &r, &anchor, &metrics, &|| {
                    b_checks.set(b_checks.get() + 1);
                    b_checks.get() < limit
                });
                assert_eq!(a_checks.get(), b_checks.get());
                match (a, b) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a_checks.get(), 1281);
                        assert_original_neighborhood(&a, &b);
                    }
                    (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                    (a, b) => panic!("different results: {a:?} vs {b:?}"),
                }
            }
        }
    }
}

#[test]
fn neighborhood_norm_scratch_matches_independent_inner_bits_errors_and_recovers() {
    // Catch changed multiplication/reduction order, lost signed-zero bits,
    // poisoned destinations after overflow, and changed finite-error ordering.
    for population in [3, 5, 9] {
        for kind in 0..3 {
            let metrics: Vec<_> = [24, 384, 24]
                .into_iter()
                .map(|width| {
                    let rows: Vec<Vec<_>> = (0..population)
                        .map(|i| {
                            (0..width)
                                .map(|j| match kind {
                                    0 => 0.25,
                                    1 => 0.25 + (i as f64 - j as f64) * 1e-14,
                                    _ => ((i * 31 + j * 7) as f64).sin(),
                                })
                                .collect()
                        })
                        .collect();
                    Metric::fit(rows.iter().map(Vec::as_slice), width, &|| true).unwrap()
                })
                .collect();
            assert!(metrics.iter().all(|m| matches!(m, Metric::LowRank { .. })));
            let mut scratch = NeighborhoodMetricScratch::new(&metrics);
            assert!(scratch.buffers.is_none(), "initialization must be lazy");
            for (linear, actual) in [
                (vec![], vec![]),
                (vec![0.; 24], vec![f64::NAN; 23]),
                (vec![f64::MAX; 24], vec![f64::NAN; 24]),
                (vec![0.; 24], vec![f64::MAX; 24]),
                (vec![0.; 24], vec![f64::NAN; 24]),
            ] {
                let actual_error = finite::neighborhood_error_into(
                    &metrics[0],
                    &linear,
                    &actual,
                    &mut Vec::new(),
                    &mut scratch,
                )
                .unwrap_err();
                let expected_error =
                    finite_original::compare(&metrics[0], "oracle".into(), &linear, &actual)
                        .unwrap_err();
                assert_eq!(format!("{actual_error:?}"), format!("{expected_error:?}"));
            }
            let mut pointers = None;
            for metric in &metrics {
                let Metric::LowRank { centered, .. } = metric else {
                    unreachable!()
                };
                let width = centered.ncols();
                for case in 0..8 {
                    let values: Vec<_> = (0..width)
                        .map(|j| match case {
                            0 => {
                                if j % 2 == 0 {
                                    -0.
                                } else {
                                    0.
                                }
                            }
                            1 | 7 => (j as f64).cos(),
                            2 => f64::from_bits(1 + j as u64),
                            3 => 1e140 * (j as f64).sin(),
                            4 => {
                                if j == 0 {
                                    f64::MAX
                                } else {
                                    -f64::MAX
                                }
                            }
                            5 => f64::NAN,
                            _ => f64::INFINITY,
                        })
                        .collect();
                    let squared = metric.inner(&values, &values);
                    assert_eq!(
                        scratch.norm_squared(metric, &values).to_bits(),
                        squared.to_bits(),
                        "population={population} kind={kind} width={width} case={case}"
                    );
                    let actual = finite::norm_into(metric, &values, &mut scratch);
                    if values.iter().any(|v| !v.is_finite()) || !squared.is_finite() {
                        assert_eq!(
                            format!("{:?}", actual.unwrap_err()),
                            "Numerical(\"nonfinite finite-displacement metric\")"
                        );
                    } else {
                        assert_eq!(actual.unwrap().to_bits(), squared.max(0.).sqrt().to_bits());
                    }
                    let b = scratch.buffers.as_ref().unwrap();
                    let shape = (b.input_residual.len(), b.projected.len(), b.dual.len());
                    assert_eq!(shape, (width, population, population));
                    let capacities = (
                        b.input_residual.data.as_vec().capacity(),
                        b.projected.data.as_vec().capacity(),
                        b.dual.data.as_vec().capacity(),
                    );
                    assert_eq!(capacities, (384, population, population));
                    let current = (
                        b.input_residual.as_ptr(),
                        b.projected.as_ptr(),
                        b.dual.as_ptr(),
                    );
                    assert_eq!(
                        *pointers.get_or_insert(current),
                        current,
                        "reshape must retain allocations"
                    );
                }
            }
        }
    }
    let metrics = [Metric::Identity, Metric::Dense(DMatrix::identity(2, 2))];
    let mut scratch = NeighborhoodMetricScratch::new(&metrics);
    for metric in &metrics {
        let values = [-0., 1.];
        assert_eq!(
            scratch.norm_squared(metric, &values).to_bits(),
            metric.inner(&values, &values).to_bits()
        );
    }
    assert!(
        scratch.buffers.is_none(),
        "Identity and Dense must not initialize LowRank scratch"
    );
}

#[test]
fn all_failed_complete_forwards_keep_every_layer_missing_and_checkpoint_order() {
    let mut m = Mlp::new(2, &[], 24, &mut ChaCha8Rng::seed_from_u64(42)).unwrap();
    m.layers[0].weight = m.layers[0].weight.clone().map(|_| {
        matrix(
            (0..48)
                .map(|i| if i < 24 { f32::MAX } else { 0. })
                .collect(),
            2,
            24,
        )
    });
    m.layers[0].bias = m.layers[0]
        .bias
        .take()
        .map(|p| p.map(|_| vector(&[0.; 24])));
    let net = NumericalMlp::from_model(&m).unwrap();
    let mut r = request();
    r.anchor = vec![0., 0.];
    r.axis_scales = vec![f64::MAX, 1.];
    r.radius = 1.;
    let anchor = net
        .trace_directions(&r.anchor, &[vec![1., 0.], vec![0., 1.]])
        .unwrap();
    let rows = [vec![0.; 24], vec![0.; 24], vec![0.; 24]];
    let metrics = [
        Metric::Identity,
        Metric::fit(rows.iter().map(Vec::as_slice), 24, &|| true).unwrap(),
    ];
    assert!(matches!(metrics[1], Metric::LowRank { .. }));
    // The even grid never samples x=0, so every real full forward overflows.
    // No layer's norms may run just because its own prefix remains finite.
    for limit in [1, 3, 768, 769, usize::MAX] {
        let a_checks = std::cell::Cell::new(0);
        let b_checks = std::cell::Cell::new(0);
        let a = finite::neighborhood(&net, &r, &anchor, &metrics, &|| {
            a_checks.set(a_checks.get() + 1);
            a_checks.get() < limit
        });
        let b = finite_original::neighborhood(&net, &r, &anchor, &metrics, &|| {
            b_checks.set(b_checks.get() + 1);
            b_checks.get() < limit
        });
        assert_eq!(a_checks.get(), b_checks.get());
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a_checks.get(), 769);
                assert!(a.errors.iter().flatten().all(Option::is_none));
                assert_original_neighborhood(&a, &b);
            }
            (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
            (a, b) => panic!("different results: {a:?} vs {b:?}"),
        }
    }
}
