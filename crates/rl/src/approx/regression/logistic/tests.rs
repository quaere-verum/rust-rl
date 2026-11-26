use super::*;

#[test]
fn objective_gradient_and_curvature_match_finite_differences() {
    let features = DMatrix::from_row_slice(3, 2, &[1.0, -0.8, 1.0, 0.3, 1.0, 1.7]);
    let config = WeightedLogistic {
        penalty: DiagonalPenalty::PerCoefficient(vec![0.0, 0.15]),
        target_kind: LogisticTargetKind::SoftLabels,
        ..WeightedLogistic::default()
    };
    let problem =
        LogisticProblem::new(&config, &features, &[0.3, 1.0, 0.1], &[1.0, 3.0, 2.0]).unwrap();
    let coefficients = DVector::from_vec(vec![0.2, -0.6]);
    let (gradient, hessian) = problem.derivatives(&coefficients).unwrap();
    let step = 1e-5;
    for column in 0..coefficients.len() {
        let mut plus = coefficients.clone();
        let mut minus = coefficients.clone();
        plus[column] += step;
        minus[column] -= step;
        let loss_plus = problem.objective(&plus).unwrap();
        let loss_minus = problem.objective(&minus).unwrap();
        let numerical_gradient = (loss_plus - loss_minus) / (2.0 * step);
        assert!((gradient[column] - numerical_gradient).abs() < 1e-10);
        let grad_plus = problem.derivatives(&plus).unwrap().0;
        let grad_minus = problem.derivatives(&minus).unwrap().0;
        for row in 0..coefficients.len() {
            let numerical_curvature = (grad_plus[row] - grad_minus[row]) / (2.0 * step);
            assert!((hessian[(row, column)] - numerical_curvature).abs() < 1e-10);
        }
    }
}

#[test]
fn saturated_bernoulli_terms_keep_small_losses_and_derivatives() {
    for (logit, target, sign) in [(40.0, 1.0, -1.0), (-40.0, 0.0, 1.0)] {
        let (loss, gradient, curvature) = bernoulli_terms(logit, target);
        let expected = 4.248_354_255_291_589e-18;
        assert!((loss - expected).abs() < 1e-32);
        assert!((gradient - sign * expected).abs() < 1e-32);
        assert!((curvature - expected).abs() < 1e-32);
    }
    assert_eq!(bernoulli_terms(1000.0, 0.0), (1000.0, 1.0, 0.0));
    assert_eq!(bernoulli_terms(-1000.0, 1.0), (1000.0, -1.0, 0.0));
}
