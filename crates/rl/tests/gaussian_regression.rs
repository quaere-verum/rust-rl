use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use rl::{Result, approx::regression::GaussianPosterior, core::checkpoint::Snapshot};

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "{actual} differs from {expected}"
    );
}

#[test]
fn gaussian_posterior_matches_independent_mean_covariance_and_information() -> Result<()> {
    let mut posterior = GaussianPosterior::new(2, 2.)?;
    let before = posterior.predict(&[0., 1.], 0.5)?;
    close(before.epistemic_variance, 0.5);
    posterior.observe(&[1., 0.], 1., 0.5)?;
    close(posterior.predict(&[0., 1.], 0.5)?.epistemic_variance, 0.5);
    close(posterior.predict(&[1., 0.], 0.5)?.epistemic_variance, 0.25);
    posterior.observe(&[1., 1.], -0.5, 0.5)?;
    // Precision [[6,2],[2,4]], natural mean [1,-1], inverse
    // [[4,-2],[-2,6]] / 20, posterior mean [0.3,-0.4].
    let first = posterior.predict(&[1., 0.], 0.5)?;
    let second = posterior.predict(&[0., 1.], 0.5)?;
    let joint = posterior.predict(&[1., 1.], 0.5)?;
    close(first.mean, 0.3);
    close(second.mean, -0.4);
    close(first.epistemic_variance, 0.2);
    close(second.epistemic_variance, 0.3);
    close(posterior.epistemic_covariance(&[1., 0.], &[0., 1.])?, -0.1);
    close(joint.mean, -0.1);
    close(joint.epistemic_variance, 0.3);
    close(joint.information_gain, 0.5 * 1.6_f64.ln());
    assert_eq!(posterior.observations(), 2);
    Ok(())
}

#[test]
fn high_scale_likelihood_preserves_orthogonal_uncertainty() -> Result<()> {
    let mut posterior = GaussianPosterior::new(2, 1.)?;
    posterior.observe(&[1e12, 1e12], 0., 1.)?;
    // A likelihood along [1,1] leaves the orthogonal [1,-1] prior unchanged.
    close(posterior.predict(&[1., -1.], 1.)?.epistemic_variance, 2.);
    assert!(posterior.predict(&[1., 1.], 1.)?.epistemic_variance < 2e-24);
    Ok(())
}

#[test]
fn noisy_observations_have_larger_errors_and_lower_information_at_matched_exposure() -> Result<()> {
    let mut posterior = GaussianPosterior::new(2, 1.)?;
    let mut rng = ChaCha8Rng::seed_from_u64(189);
    let mut errors = [0.; 2];
    let mut previous_gain = [f64::INFINITY; 2];
    for _ in 0..128 {
        let informative = posterior.predict(&[1., 0.], 0.25)?;
        let distractor = posterior.predict(&[0., 1.], 64.)?;
        assert!(informative.information_gain > distractor.information_gain);
        for (arm, features, variance, mean, prediction) in [
            (0, [1., 0.], 0.25_f64, 0.7, informative),
            (1, [0., 1.], 64., 0., distractor),
        ] {
            let noise: f64 = StandardNormal.sample(&mut rng);
            let observation = mean + variance.sqrt() * noise;
            errors[arm] += (observation - prediction.mean).powi(2);
            assert!(prediction.information_gain < previous_gain[arm]);
            previous_gain[arm] = prediction.information_gain;
            posterior.observe(&features, observation, variance)?;
        }
    }
    assert!(errors[1] > 20. * errors[0]);
    Ok(())
}

#[test]
fn posterior_restore_is_exact_and_invalid_observations_leave_it_unchanged() -> Result<()> {
    let mut posterior = GaussianPosterior::new(2, 0.4)?;
    posterior.observe(&[2.5, -3.], 0.25, 1.2)?;
    let bytes = serde_json::to_vec(&posterior)?;
    let mut restored: GaussianPosterior = serde_json::from_slice(&bytes)?;
    restored.validate_snapshot()?;
    assert!(posterior.observe(&[f64::MAX, 1.], 1., 1e-300).is_err());
    assert!(posterior.observe(&[1., 2.], f64::NAN, 1.).is_err());
    assert!(posterior.observe(&[1.], 1., 1.).is_err());
    assert!(posterior.observe(&[1., 2.], 1., 0.).is_err());
    assert_eq!(serde_json::to_vec(&posterior)?, bytes);
    for model in [&mut posterior, &mut restored] {
        model.observe(&[0.2, -0.75], 2., 0.5)?;
    }
    assert_eq!(
        serde_json::to_vec(&restored)?,
        serde_json::to_vec(&posterior)?
    );
    let mut corrupted: serde_json::Value = serde_json::from_slice(&bytes)?;
    corrupted["precision_cholesky"][0] = (-1.).into();
    let corrupted: GaussianPosterior = serde_json::from_value(corrupted)?;
    assert!(corrupted.validate_snapshot().is_err());
    Ok(())
}
