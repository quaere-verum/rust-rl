#![cfg(feature = "diagnostics")]
use rl::diagnostics::policy::GaussianMarginal;

fn marginal(mean: f64, sigma: f64, bounds: Option<[f64; 2]>) -> GaussianMarginal {
    GaussianMarginal {
        mean,
        log_std: sigma.ln(),
        bounds,
    }
}
fn close(got: f64, want: f64, tolerance: f64) {
    assert!((got - want).abs() < tolerance, "got {got}, want {want}");
}
#[test]
fn selected_spread_rejects_zero_and_nonzero_affine_underflow() {
    let results = [1e-320, 1e-318].map(|scale| {
        let result = marginal(0., 1e-5, Some([-scale, scale])).action_statistics();
        eprintln!("affine scale={scale:e}: {result:?}");
        result
    });
    assert!(
        results.iter().all(Result::is_err),
        "underflowed spread/error thresholds must not satisfy a false 0 <= 0 precision check: {results:?}"
    );
    let normal = marginal(0., 1e-5, Some([-1e-300, 1e-300]))
        .action_statistics()
        .unwrap()
        .standard_deviation;
    assert!(normal.is_normal());
    assert!((normal / 1e-300 / 9.999_999_999e-6 - 1.).abs() <= 1e-8);
}
#[test]
fn selected_spread_resolves_uncertainty_before_cutoff_and_affine_display() {
    // Independent compensated composite 16-point Gauss-Legendre integration
    // on [-12,12], stable saturated residual, two central passes. The review's
    // 768/1536-panel results agree at these digits (sigma = exp(ln(5))).
    let mut failures = Vec::new();
    for (mean, expected) in [
        (35.5, 1.4258558417384744e-6),
        (-35.5, 1.4258558417384744e-6),
        (35.98998472094536, 9.999000169437813e-7),
        (-35.98998472094536, 9.999000169437813e-7),
    ] {
        for scale in [1., 1e6] {
            let result = marginal(mean, 5., Some([-scale, scale])).action_statistics();
            eprintln!(
                "mean={mean} scale={scale} expected_std={} result={result:?}",
                expected * scale
            );
            if let Ok(stats) = result {
                let relative_error = (stats.standard_deviation / (expected * scale) - 1.).abs();
                if expected <= 1e-6 || relative_error > 1e-8 {
                    failures.push((mean, scale, stats.standard_deviation, relative_error));
                }
            } else if expected > 1e-6 {
                failures.push((mean, scale, f64::NAN, f64::NAN));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "unsupported spread/cutoff claims: {failures:?}"
    );
}
#[test]
fn selected_spread_keeps_supported_narrow_and_scaled_precision() {
    for (sigma, scale, expected) in [
        (1., 1e6, 627_928.730_349_106_8),
        // For small symmetric sigma, std(tanh(N)) = sigma *
        // (1 - sigma^2 + 7/3 sigma^4 + O(sigma^6)). The omitted term is
        // far below these assertions, independently of adaptive quadrature.
        (1e-5, 1., 9.999_999_999e-6),
        (0.001, 1e6, 999.999_000_002_333_3),
    ] {
        let stats = marginal(0., sigma, Some([-scale, scale]))
            .action_statistics()
            .unwrap();
        let error = (stats.standard_deviation - expected).abs();
        eprintln!(
            "supported sigma={sigma} scale={scale}: std={} error={error}",
            stats.standard_deviation
        );
        assert!(error <= expected * 1e-8);
        if (1e-4..1e6).contains(&expected) {
            assert!(error <= 1e-7);
        }
    }
}
// Independent composite midpoint integration in standard-normal coordinates.
// A fixed 400,000 cells resolves even the sigma=100 transition; no adaptive
// quadrature, implementation CDF, or implementation density enters this oracle.
fn oracle(mu: f64, sigma: f64) -> f64 {
    let h = 18. / 400_000.;
    (0..400_000)
        .map(|i| {
            let z = -9. + (i as f64 + 0.5) * h;
            (mu + sigma * z).tanh() * (-z * z / 2.).exp() / (2. * std::f64::consts::PI).sqrt() * h
        })
        .sum()
}
#[test]
fn expected_actual_actions_match_independent_oracle_across_variance_and_saturation() {
    for (mu, sigma) in [
        (0.7, 0.001),
        (0.7, 0.8),
        (2., 5.),
        (2., 100.),
        (15., 0.2),
        (-0.7, 3.),
    ] {
        close(
            marginal(mu, sigma, Some([-3., 7.]))
                .expected_action()
                .unwrap(),
            2. + 5. * oracle(mu, sigma),
            5e-8,
        );
    }
    close(
        marginal(0., 100., Some([-3., 7.]))
            .expected_action()
            .unwrap(),
        2.,
        1e-10,
    );
    close(marginal(4., 3., None).expected_action().unwrap(), 4., 1e-12);
    assert!(
        (marginal(2., 5., Some([-1., 1.])).expected_action().unwrap() - 2f64.tanh()).abs() > 0.5
    );
}
#[test]
fn density_and_cdf_have_the_correct_actual_action_jacobian_and_interval_mass() {
    let normal = marginal(0., 1., None);
    close(normal.density(0.).unwrap(), 0.398_942_280_401_432_7, 1e-14);
    close(normal.cdf(1.).unwrap(), 0.841_344_746_068_542_9, 2e-7);
    let bounded = marginal(0., 1., Some([-3., 7.]));
    close(
        bounded.density(2.).unwrap(),
        0.079_788_456_080_286_54,
        1e-14,
    );
    close(
        bounded.cdf(2. + 5. * 1f64.tanh()).unwrap(),
        0.841_344_746_068_542_9,
        2e-7,
    );
    let h = 8. / 100_000.;
    let integrated: f64 = (0..100_000)
        .map(|i| bounded.density(-2. + (i as f64 + 0.5) * h).unwrap() * h)
        .sum();
    close(
        integrated,
        bounded.cdf(6.).unwrap() - bounded.cdf(-2.).unwrap(),
        2e-8,
    );
    // Whole-support normalization from an independent fixed midpoint sum.
    let h = 10. / 100_000.;
    let mass: f64 = (0..100_000)
        .map(|i| bounded.density(-3. + (i as f64 + 0.5) * h).unwrap() * h)
        .sum();
    close(mass, 1., 2e-7);
    for (mu, sigma, want_center_cdf) in [
        (2., 5., 0.344_578_258_389_675_8),
        (2., 100., 0.492_021_686_283_098),
        (15., 0.2, 0.),
        (0.7, 0.001, 0.),
    ] {
        let g = marginal(mu, sigma, Some([-3., 7.]));
        close(g.cdf(2.).unwrap(), want_center_cdf, 2e-7);
        let action = 2. + 5. * mu.tanh();
        if mu < 10. {
            // At saturated tanh values, rounding the actual action loses latent
            // precision; its CDF need not still be exactly the median.
            close(g.cdf(action).unwrap(), 0.5, 2e-7);
            close(
                g.density(action).unwrap(),
                mu.cosh().powi(2) / (5. * sigma * (2. * std::f64::consts::PI).sqrt()),
                1e-8,
            );
        }
    }
    for x in [-4., -3., 7., 8.] {
        assert_eq!(bounded.density(x).unwrap(), 0.);
    }
    assert_eq!(bounded.cdf(-3.).unwrap(), 0.);
    assert_eq!(bounded.cdf(7.).unwrap(), 1.);
    // Close to a boundary, evaluate in log space rather than multiplying a
    // vanishing normal and an overflowing inverse Jacobian.
    assert!(
        marginal(10., 3., Some([-1., 1.]))
            .density(1. - 1e-14)
            .unwrap()
            .is_finite()
    );
}
#[test]
fn invalid_and_unrepresentable_parameters_never_fabricate_an_answer() {
    for g in [
        GaussianMarginal {
            mean: f64::NAN,
            log_std: 0.,
            bounds: None,
        },
        GaussianMarginal {
            mean: 0.,
            log_std: 1000.,
            bounds: None,
        },
        GaussianMarginal {
            mean: 0.,
            log_std: -1000.,
            bounds: None,
        },
        marginal(0., 1., Some([1., 1.])),
        marginal(0., 1., Some([1., -1.])),
        marginal(0., 1., Some([0., f64::INFINITY])),
    ] {
        assert!(g.expected_action().is_err());
        assert!(g.density(0.).is_err());
        assert!(g.cdf(0.).is_err());
    }
    let g = marginal(0., 1., None);
    assert!(g.density(f64::NAN).is_err());
    assert!(g.cdf(f64::NAN).is_err());
    assert_eq!(g.cdf(f64::NEG_INFINITY).unwrap(), 0.);
    assert_eq!(g.cdf(f64::INFINITY).unwrap(), 1.);
    assert!(
        marginal(1., 1e100, Some([-1., 1.]))
            .expected_action()
            .is_err()
    );
}

#[test]
fn standardization_does_not_overflow_before_dividing_by_a_large_sigma() {
    let g = marginal(1e308, 1e308, None);
    close(g.cdf(-1e308).unwrap(), 0.022_750_131_948_179_21, 2e-7);
    assert!(g.density(-1e308).unwrap() > 1e-310);
}
