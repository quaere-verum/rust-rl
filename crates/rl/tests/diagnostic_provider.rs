#![cfg(feature = "diagnostics")]

use rl::{
    Result,
    diagnostics::reference::{
        GradientMethod, action_gradient_comparison, continuous_decision_loss, critic_error,
    },
};

#[test]
fn weighted_calibration_distinguishes_bias_from_absolute_and_squared_error() -> Result<()> {
    let estimate = critic_error(&[4., 1., 7.], &[2., 3., 7.], Some(&[1., 3., 0.]))?;
    assert!((estimate.bias + 1.0).abs() < 1e-12);
    assert!((estimate.mae - 2.0).abs() < 1e-12);
    assert!((estimate.rmse - 2.0).abs() < 1e-12);
    assert_eq!(estimate.count, 2);
    assert!(critic_error(&[1.], &[1.], Some(&[0.])).is_err());
    Ok(())
}

#[test]
fn stochastic_decisions_integrate_over_actions_and_report_sampling_error() -> Result<()> {
    let deterministic = continuous_decision_loss(2., &[1.])?;
    assert_eq!(deterministic.loss, 1.);
    assert_eq!(deterministic.standard_error, None);
    let stochastic = continuous_decision_loss(2., &[2., 0., 0., 2.])?;
    assert_eq!(stochastic.loss, 1.);
    assert!((stochastic.standard_error.unwrap() - 1. / 3.0_f64.sqrt()).abs() < 1e-12);
    assert!(continuous_decision_loss(1., &[1., 1.1]).is_err());
    Ok(())
}

#[test]
fn gradient_direction_requires_both_vectors_to_have_nonzero_norm() -> Result<()> {
    let comparison = action_gradient_comparison(
        &[3., 4.],
        GradientMethod::AutomaticDifferentiation,
        &[-3., -4.],
        GradientMethod::Analytic,
    )?;
    assert_eq!(comparison.magnitude_error, 0.);
    assert_eq!(comparison.vector_error, 10.);
    assert_eq!(comparison.cosine, Some(-1.));
    assert_eq!(
        action_gradient_comparison(
            &[0.],
            GradientMethod::Analytic,
            &[1.],
            GradientMethod::Analytic
        )?
        .cosine,
        None,
    );
    Ok(())
}
