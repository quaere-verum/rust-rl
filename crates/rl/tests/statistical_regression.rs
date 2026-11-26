use nalgebra::DMatrix;
use rl::approx::regression::{
    DiagonalPenalty, LabelSource, LinearModel, LogisticModel, LogisticTargetKind, Predictor,
    Regressor, WeightedLogistic, WeightedRidge, signed_terminal_returns_to_binary,
};

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn ridge_penalty_is_relative_to_total_weight() {
    let features = DMatrix::from_column_slice(2, 1, &[1.0, 2.0]);
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::Uniform(0.75),
        ..WeightedRidge::default()
    };
    for weights in [[1.0, 3.0], [f64::MAX / 3.0, f64::MAX]] {
        let (model, diagnostics) = ridge.fit(&features, &[5.0, 1.0], &weights).unwrap();
        close(model.coefficients()[0], 11.0 / 16.0, 1e-12);
        close(
            diagnostics.weighted_residual_norm,
            (4869.0_f64 / 1024.0).sqrt(),
            1e-12,
        );
        close(
            diagnostics.solve.relative_residual,
            (327.0_f64 / 448.0).sqrt(),
            1e-12,
        );
        assert_eq!(diagnostics.solve.rank, 1);
        close(diagnostics.solve.condition_number.unwrap(), 1.0, 1e-12);
        close(diagnostics.solve.singular_values[0], 2.0, 1e-12);
    }
}

#[test]
fn ridge_can_leave_an_explicit_intercept_unpenalized() {
    let features = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, 1.0, 1.0]);
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::PerCoefficient(vec![0.0, 2.0]),
        ..WeightedRidge::default()
    };
    let (model, _) = ridge.fit(&features, &[0.0, 4.0], &[1.0, 1.0]).unwrap();
    close(model.coefficients()[0], 2.0, 1e-12);
    close(model.coefficients()[1], 2.0 / 3.0, 1e-12);
}

#[test]
fn prepared_ridge_matches_ordinary_fits_for_independent_target_vectors() {
    let design = DMatrix::from_row_slice(3, 2, &[1.0, -2.0, 1.0, 0.5, 1.0, 1.5]);
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::PerCoefficient(vec![0.0, 0.2]),
        ..WeightedRidge::default()
    };
    let weights = [1.0, 0.0, 3.0];
    let prepared = ridge.prepare(&design, &weights).unwrap();

    let first = [2.0, 5.0, 8.0];
    assert_eq!(
        prepared(&first).unwrap(),
        ridge.fit(&design, &first, &weights).unwrap()
    );
    assert!(matches!(prepared(&[1.0, 2.0]), Err(rl::Error::Shape(_))));
    assert!(matches!(
        prepared(&[1.0, f64::NAN, 3.0]),
        Err(rl::Error::Numerical(_))
    ));

    let second = [-7.0, 4.0, -2.0];
    let fitted = prepared(&second).unwrap();
    assert_eq!(fitted, ridge.fit(&design, &second, &weights).unwrap());
    assert_ne!(fitted.0, prepared(&first).unwrap().0);
}

#[test]
fn default_prepared_fit_defers_logistic_target_validation_to_fit() {
    let design = DMatrix::from_element(2, 1, 1.0);
    let logistic = WeightedLogistic::default();
    let weights = [3.0, 1.0];
    let prepared = logistic.prepare(&design, &weights).unwrap();
    assert!(prepared(&[0.2, 1.0]).is_err());
    let targets = [0.0, 1.0];
    assert_eq!(
        prepared(&targets).unwrap(),
        logistic.fit(&design, &targets, &weights).unwrap()
    );
}

#[test]
fn ridge_returns_minimum_norm_coefficients_in_a_rank_deficient_design() {
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedRidge::default()
    };
    let design = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    let (model, diagnostics) = ridge.fit(&design, &[5.0, 10.0], &[1.0, 1.0]).unwrap();
    close(model.coefficients()[0], 1.0, 1e-12);
    close(model.coefficients()[1], 2.0, 1e-12);
    assert_eq!(diagnostics.solve.rank, 1);
    assert_eq!(diagnostics.solve.condition_number, None);
    close(diagnostics.solve.residual_norm, 0.0, 1e-12);

    let wide_design = DMatrix::from_row_slice(1, 2, &[3.0, 4.0]);
    let (model, diagnostics) = ridge.fit(&wide_design, &[5.0], &[1.0]).unwrap();
    close(model.coefficients()[0], 0.6, 1e-12);
    close(model.coefficients()[1], 0.8, 1e-12);
    assert_eq!(diagnostics.solve.rank, 1);
    assert_eq!(diagnostics.solve.condition_number, None);
}

#[test]
fn zero_weight_rows_do_not_contribute_to_regression() {
    let design = DMatrix::from_column_slice(3, 1, &[1.0, f64::MAX, 1.0]);
    let weights = &[1.0, 0.0, 1.0];
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::Uniform(1.0),
        ..WeightedRidge::default()
    };
    let (model, _) = ridge.fit(&design, &[1.0, f64::MAX, 3.0], weights).unwrap();
    close(model.predict(&[1.0]).unwrap(), 1.0, 1e-12);

    let logistic = WeightedLogistic::default();
    let (model, _) = logistic.fit(&design, &[0.0, 1.0, 1.0], weights).unwrap();
    close(model.predict(&[1.0]).unwrap(), 0.5, 0.0);
}

#[test]
fn fitting_rejects_invalid_weights_and_preserves_the_existing_model() {
    let design = DMatrix::from_column_slice(2, 1, &[1.0, 1.0]);
    let ridge = WeightedRidge::default();
    let logistic = WeightedLogistic::default();
    let (existing, _) = ridge.fit(&design, &[1.0, 3.0], &[1.0, 1.0]).unwrap();
    let original = existing.predict(&[2.0]).unwrap();
    for weights in [
        &[0.0, 0.0][..],
        &[-1.0, 2.0],
        &[f64::NAN, 1.0],
        &[f64::INFINITY, 1.0],
        &[1.0],
    ] {
        assert!(ridge.fit(&design, &[1.0, 3.0], weights).is_err());
        assert!(logistic.fit(&design, &[0.0, 1.0], weights).is_err());
    }
    assert!(ridge.fit(&design, &[f64::NAN, 1.0], &[1.0, 1.0]).is_err());
    let corrupted = DMatrix::from_column_slice(2, 1, &[f64::INFINITY, 1.0]);
    assert!(ridge.fit(&corrupted, &[1.0, 3.0], &[0.0, 1.0]).is_err());
    assert_eq!(existing.predict(&[2.0]).unwrap(), original);
}

#[test]
fn predictor_checks_shape_finiteness_and_typed_restore() {
    assert!(LinearModel::new(vec![]).is_err());
    assert!(LinearModel::new(vec![f64::NAN]).is_err());
    let model = LinearModel::new(vec![1.0, 2.0]).unwrap();
    assert_eq!(model.dimension(), 2);
    assert!(model.predict(&[1.0]).is_err());
    assert!(model.predict(&[1.0, f64::NAN]).is_err());
    assert!(model.predict(&[1.0, f64::MAX]).is_err());
    let overflow = LinearModel::new(vec![f64::MAX, f64::MAX]).unwrap();
    assert!(overflow.predict(&[1.0, 1.0]).is_err());

    let record = serde_json::to_vec(&model).unwrap();
    let restored: LinearModel = serde_json::from_slice(&record).unwrap();
    assert_eq!(restored, model);
    assert!(serde_json::from_str::<LinearModel>(r#"{"coefficients":[]}"#).is_err());
    assert!(serde_json::from_str::<LinearModel>(r#"{"coefficients":[1e999]}"#).is_err());
}

#[test]
fn invalid_configs_are_rejected_on_fit_and_typed_restore() {
    let design = DMatrix::from_element(2, 1, 1.0);
    for ridge in [
        WeightedRidge {
            penalty: DiagonalPenalty::Uniform(-1.0),
            ..WeightedRidge::default()
        },
        WeightedRidge {
            relative_cutoff: 1.0,
            ..WeightedRidge::default()
        },
    ] {
        assert!(ridge.fit(&design, &[0.0, 1.0], &[1.0, 1.0]).is_err());
        let record = serde_json::to_vec(&ridge).unwrap();
        assert!(serde_json::from_slice::<WeightedRidge>(&record).is_err());
    }
    let wrong_dimension = WeightedRidge {
        penalty: DiagonalPenalty::PerCoefficient(vec![0.0, 0.0]),
        ..WeightedRidge::default()
    };
    assert!(
        wrong_dimension
            .fit(&design, &[0.0, 1.0], &[1.0, 1.0])
            .is_err()
    );

    let logistic = WeightedLogistic {
        max_iterations: 0,
        ..WeightedLogistic::default()
    };
    assert!(logistic.fit(&design, &[0.0, 1.0], &[1.0, 1.0]).is_err());
    let record = serde_json::to_vec(&logistic).unwrap();
    assert!(serde_json::from_slice::<WeightedLogistic>(&record).is_err());
}

#[test]
fn logistic_fits_the_weighted_event_rate_and_round_trips_target_metadata() {
    let design = DMatrix::from_element(2, 1, 1.0);
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    let (model, diagnostics) = logistic.fit(&design, &[0.0, 1.0], &[3.0, 1.0]).unwrap();
    close(model.predict(&[1.0]).unwrap(), 0.25, 1e-8);
    close(model.logit(&[1.0]).unwrap(), (1.0_f64 / 3.0).ln(), 1e-7);
    close(diagnostics.objective, 0.562_335_144_618_808_3, 1e-12);
    assert!(diagnostics.objective < diagnostics.initial_objective);
    assert!(diagnostics.gradient_norm <= logistic.gradient_tolerance);
    assert_eq!(model.target_kind(), LogisticTargetKind::FactualBinary);
    let record = serde_json::to_vec(&model).unwrap();
    let restored: LogisticModel = serde_json::from_slice(&record).unwrap();
    assert_eq!(restored, model);
}

#[test]
fn logistic_convergence_is_robust_to_rescaling_an_unpenalized_feature() {
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    for scale in [1e5, 1e-9] {
        let design = DMatrix::from_element(2, 1, scale);
        let (model, _) = logistic.fit(&design, &[0.0, 1.0], &[4.0, 13.0]).unwrap();
        close(model.predict(&[scale]).unwrap(), 13.0 / 17.0, 1e-8);
    }
}

#[test]
fn low_weight_high_leverage_rows_do_not_stop_logistic_convergence_early() {
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    for scale in [1.0, 1e-180, 1e180] {
        let extreme = 1e19 * scale;
        let design =
            DMatrix::from_column_slice(6, 1, &[scale, scale, scale, scale, extreme, extreme]);
        let targets = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let weights = [1.0, 1.0, 1.0, 1.0, 1e-21, 1e-21];
        let (model, _) = logistic.fit(&design, &targets, &weights).unwrap();
        // The last two rows saturate at probability one, so the gradient in
        // the unscaled design is 4 * p - 3 + 1e19 * 1e-21 at the optimum.
        close(model.predict(&[scale]).unwrap(), 0.7475, 1e-8);
    }
}

#[test]
fn logistic_fits_a_feature_whose_unscaled_curvature_underflows() {
    let scale = 1e-200;
    let design = DMatrix::from_element(4, 1, scale);
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    let (model, _) = logistic
        .fit(&design, &[0.0, 1.0, 1.0, 1.0], &[1.0; 4])
        .unwrap();
    close(model.predict(&[scale]).unwrap(), 0.75, 1e-8);
    close(model.coefficients()[0] * scale, 3.0_f64.ln(), 1e-7);
}

#[test]
fn logistic_fits_a_feature_whose_unscaled_curvature_overflows() {
    let scale = 1e200;
    let design = DMatrix::from_element(4, 1, scale);
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    let (model, _) = logistic
        .fit(&design, &[0.0, 1.0, 1.0, 1.0], &[1.0; 4])
        .unwrap();
    close(model.predict(&[scale]).unwrap(), 0.75, 1e-8);
    close(model.coefficients()[0] * scale, 3.0_f64.ln(), 1e-7);
}

#[test]
fn logistic_preserves_the_penalized_objective_under_large_column_rescalings() {
    let logit = 3.0_f64.ln();
    for scale in [2.5e154, 2.5e-154, 1.0] {
        let design = DMatrix::from_row_slice(2, 2, &[1.0, -scale, 1.0, scale]);
        let penalty = (0.25 / logit * scale) * scale;
        let logistic = WeightedLogistic {
            penalty: DiagonalPenalty::PerCoefficient(vec![0.0, penalty]),
            ..WeightedLogistic::default()
        };
        let (model, diagnostics) = logistic.fit(&design, &[0.0, 1.0], &[1.0, 1.0]).unwrap();
        close(model.predict(&[1.0, -scale]).unwrap(), 0.25, 1e-8);
        close(model.predict(&[1.0, scale]).unwrap(), 0.75, 1e-8);
        close(model.coefficients()[1] * scale, logit, 1e-7);
        close(diagnostics.objective, -0.75_f64.ln() + logit / 8.0, 1e-12);
    }
}

#[test]
fn logistic_reports_a_coefficient_that_cannot_fit_in_the_original_units() {
    let design = DMatrix::from_element(4, 1, f64::from_bits(1));
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    assert!(matches!(
        logistic.fit(&design, &[0.0, 1.0, 1.0, 1.0], &[1.0; 4]),
        Err(rl::Error::Numerical(_))
    ));
}

#[test]
fn logistic_regularizes_relative_to_the_weighted_mean_with_a_free_intercept() {
    let design = DMatrix::from_row_slice(2, 2, &[1.0, -1.0, 1.0, 1.0]);
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::PerCoefficient(vec![0.0, 0.25 / 3.0_f64.ln()]),
        ..WeightedLogistic::default()
    };
    for weight in [1.0, f64::MAX] {
        let (model, _) = logistic
            .fit(&design, &[0.0, 1.0], &[weight, weight])
            .unwrap();
        close(model.coefficients()[0], 0.0, 1e-10);
        close(model.coefficients()[1], 3.0_f64.ln(), 1e-7);
        close(model.predict(&[1.0, 1.0]).unwrap(), 0.75, 1e-8);
    }
}

#[test]
fn logistic_requires_declared_soft_labels_or_explicit_signed_conversion() {
    let design = DMatrix::from_element(2, 1, 1.0);
    let binary = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..WeightedLogistic::default()
    };
    assert!(binary.fit(&design, &[0.2, 0.8], &[1.0, 3.0]).is_err());
    assert!(binary.fit(&design, &[-1.0, 1.0], &[1.0, 3.0]).is_err());
    assert!(
        binary
            .validate_label_source(LabelSource::ObservedBinary)
            .is_ok()
    );
    assert!(binary.validate_label_source(LabelSource::Derived).is_err());
    let soft = WeightedLogistic {
        target_kind: LogisticTargetKind::SoftLabels,
        ..binary
    };
    assert!(soft.validate_label_source(LabelSource::Derived).is_ok());
    let (model, _) = soft.fit(&design, &[0.2, 0.8], &[1.0, 3.0]).unwrap();
    assert_eq!(model.target_kind(), LogisticTargetKind::SoftLabels);
    close(model.predict(&[1.0]).unwrap(), 0.65, 1e-8);
    assert!(soft.fit(&design, &[0.2, 1.1], &[1.0, 3.0]).is_err());
    assert_eq!(
        signed_terminal_returns_to_binary(&[-1.0, 1.0]).unwrap(),
        [0.0, 1.0]
    );
    for invalid in [0.0, 2.0, f64::NAN, f64::INFINITY] {
        assert!(signed_terminal_returns_to_binary(&[invalid]).is_err());
    }
}

#[test]
fn logistic_prediction_saturates_without_exponent_overflow() {
    let model = LogisticModel::new(vec![1.0], LogisticTargetKind::FactualBinary).unwrap();
    assert_eq!(model.predict(&[1_000.0]).unwrap(), 1.0);
    assert_eq!(model.predict(&[-1_000.0]).unwrap(), 0.0);
    assert!(model.predict(&[f64::NAN]).is_err());
}

#[test]
fn logistic_reports_nonconvergence_instead_of_installing_an_iterate() {
    let design = DMatrix::from_element(2, 1, 1.0);
    let logistic = WeightedLogistic {
        penalty: DiagonalPenalty::Uniform(0.0),
        max_iterations: 1,
        ..WeightedLogistic::default()
    };
    assert!(matches!(
        logistic.fit(&design, &[0.0, 1.0], &[3.0, 1.0]),
        Err(rl::Error::Numerical(_))
    ));
}
