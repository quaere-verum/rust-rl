#![cfg(feature = "neural")]

use burn::tensor::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    nn::{distributions::Distribution, matrix, to_vec},
    objectives::{
        hl_gauss::HlGaussConfig,
        reppo::{SoftTrajectory, constraint_dual_loss, kl_gated_loss, soft_td_lambda},
    },
};

fn close(a: f32, b: f32, tolerance: f32) {
    assert!(
        (a - b).abs() <= tolerance,
        "{a} != {b}, tolerance {tolerance}"
    );
}

#[test]
fn kl_gate_switches_per_observation_including_equality() {
    let ordinary = rl::nn::vector(&[3., 4., 5.]).require_grad();
    let kl = rl::nn::vector(&[0.09, 0.1, 0.2]).require_grad();
    let loss = kl_gated_loss(ordinary.clone(), kl.clone(), 2., 0.1).unwrap();
    close(loss.clone().into_scalar(), (3. + 0.2 + 0.4) / 3., 1e-6);
    let grads = loss.backward();
    let ordinary_grad = to_vec(ordinary.grad(&grads).unwrap()).unwrap();
    let kl_grad = to_vec(kl.grad(&grads).unwrap()).unwrap();
    for (a, b) in ordinary_grad.into_iter().zip([1. / 3., 0., 0.]) {
        close(a, b, 1e-6);
    }
    for (a, b) in kl_grad.into_iter().zip([0., 2. / 3., 2. / 3.]) {
        close(a, b, 1e-6);
    }
}

#[test]
fn independent_dual_signs_increase_pressure_on_violations_and_detach_residuals() {
    // Descent on log(multiplier)*exp uses entropy slack H-target, KL slack limit-KL.
    for slack in [-0.3, 0.3] {
        let log = rl::nn::vector(&[0.]).require_grad();
        let residual = rl::nn::vector(&[slack]).require_grad();
        let loss = constraint_dual_loss(log.clone(), residual.clone()).unwrap();
        let grads = loss.backward();
        close(log.grad(&grads).unwrap().into_scalar(), slack, 1e-7);
        assert!(residual.grad(&grads).is_none());
    }
}

#[test]
fn soft_targets_keep_lanes_boundaries_and_next_entropy_separate() {
    // Time-major lanes A/B. A terminates at t=1 and starts again at t=2.
    // B truncates at t=0; its reset episode continues to the fragment cut.
    let rewards = [1., 10., 2., 20., 3., 30.];
    let terminal = [false, false, true, false, false, false];
    let truncated = [false, true, false, false, false, false];
    let bootstrap = [4., 40., 999., 50., 6., 60.];
    let entropy = [0.2, 2., 999., 3., 0.4, 4.];
    for (lambda, expected) in [
        (0., [3.1, 31., 2., 46.5, 6.2, 62.]),
        (0.5, [2.6, 31., 2., 49.5, 6.2, 62.]),
        (1., [2.1, 31., 2., 52.5, 6.2, 62.]),
    ] {
        let targets = soft_td_lambda(
            SoftTrajectory {
                rewards: &rewards,
                terminated: &terminal,
                truncated: &truncated,
                bootstrap: &bootstrap,
                next_entropy: &entropy,
                lanes: 2,
            },
            0.5,
            lambda,
        )
        .unwrap();
        for (actual, expected) in targets.into_iter().zip(expected) {
            close(actual, expected, 1e-5);
        }
    }
}

#[test]
fn soft_targets_cover_discount_extremes_and_reject_invalid_shape_or_nonfinite() {
    let targets = |gamma, lambda, lanes, rewards: &[f32]| {
        soft_td_lambda(
            SoftTrajectory {
                rewards,
                terminated: &[false, true],
                truncated: &[false, false],
                bootstrap: &[5., 0.],
                next_entropy: &[0.5, 0.],
                lanes,
            },
            gamma,
            lambda,
        )
    };
    assert_eq!(targets(0., 1., 1, &[1., 2.]).unwrap(), [1., 2.]);
    assert_eq!(targets(1., 1., 1, &[1., 2.]).unwrap(), [3.5, 2.]);
    assert_eq!(targets(1., 0., 1, &[1., 2.]).unwrap(), [6.5, 2.]);
    for (gamma, lambda, lanes) in [(1.01, 0., 1), (0.9, -0.1, 1), (0.9, 1., 0), (0.9, 1., 3)] {
        assert!(targets(gamma, lambda, lanes, &[1., 2.]).is_err());
    }
    assert!(targets(0.9, 0.5, 1, &[1.]).is_err());
    assert!(targets(0.9, 0.5, 1, &[f32::NAN, 2.]).is_err());
}

fn support() -> HlGaussConfig {
    HlGaussConfig {
        min: -1.,
        max: 1.,
        bins: 5,
        sigma_ratio: 0.75,
    }
}

#[test]
fn hl_gauss_matches_independent_integrated_gaussian_reference() {
    let config = support();
    let values = [0., 0.13, -0.99, 1., -10., 10.];
    let hist = config.targets(&values).unwrap();
    assert_eq!(hist.out_of_support, 2);
    for (&target, row) in values.iter().zip(hist.probabilities.chunks(5)) {
        close(row.iter().sum(), 1., 2e-7);
        let mean = f64::from(target.clamp(-1., 1.));
        // Simpson quadrature of the Gaussian density, independent of CDF subtraction.
        let mass = |left: f64, right: f64| {
            let n = 2000;
            let dx = (right - left) / n as f64;
            let density = |x: f64| (-0.5 * ((x - mean) / 0.375).powi(2)).exp();
            let mut sum = density(left) + density(right);
            for i in 1..n {
                sum += density(left + i as f64 * dx) * if i % 2 == 0 { 2. } else { 4. };
            }
            sum * dx / 3.
        };
        let total = mass(-1.25, 1.25);
        for (i, &p) in row.iter().enumerate() {
            let left = -1.25 + i as f64 * 0.5;
            close(p, (mass(left, left + 0.5) / total) as f32, 2e-7);
        }
    }
    close(hist.probabilities[0], hist.probabilities[4], 1e-7);
    assert!(hist.probabilities[2] > hist.probabilities[1]);
}

#[test]
fn hl_gauss_decodes_centers_and_cross_entropy_has_finite_extreme_gradients() {
    let config = support();
    close(
        config
            .decode(matrix(vec![0.; 5], 1, 5))
            .unwrap()
            .into_scalar(),
        0.,
        1e-7,
    );
    close(
        config
            .decode(matrix(vec![-1000., -1000., -1000., 1000., -1000.], 1, 5))
            .unwrap()
            .into_scalar(),
        0.5,
        1e-7,
    );
    let logits = matrix(vec![1000., -1000., 0., -1000., 1000.], 1, 5).require_grad();
    let hist = config.targets(&[0.13]).unwrap();
    let targets = matrix(hist.probabilities, 1, 5).require_grad();
    let loss = config.loss(logits.clone(), targets.clone()).unwrap();
    assert!(loss.clone().into_scalar::<f32>().is_finite());
    let grads = loss.backward();
    assert!(
        to_vec(logits.grad(&grads).unwrap())
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
    assert!(
        targets.grad(&grads).is_none(),
        "regression targets must be detached"
    );
    assert!(config.decode(matrix(vec![0.; 4], 1, 4)).is_err());
}

#[test]
fn hl_gauss_extreme_valid_widths_stay_normalized_and_invalid_configs_fail() {
    for ratio in [f32::MIN_POSITIVE, 1e-12, 1e12, f32::MAX] {
        let config = HlGaussConfig {
            sigma_ratio: ratio,
            ..support()
        };
        let hist = config.targets(&[0.13]).unwrap();
        assert!(hist.probabilities.iter().all(|x| x.is_finite() && *x >= 0.));
        close(hist.probabilities.iter().sum(), 1., 1e-6);
        if ratio > 1e10 {
            for p in hist.probabilities {
                close(p, 0.2, 1e-6);
            }
        }
    }
    for config in [
        HlGaussConfig {
            bins: 1,
            ..support()
        },
        HlGaussConfig {
            min: 1.,
            ..support()
        },
        HlGaussConfig {
            sigma_ratio: 0.,
            ..support()
        },
        HlGaussConfig {
            sigma_ratio: f32::NAN,
            ..support()
        },
    ] {
        assert!(config.validate().is_err());
    }
    assert!(support().targets(&[f32::INFINITY]).is_err());
}

fn gaussian(mean: Tensor<2>, log_std: Tensor<2>, bounded: bool) -> Distribution {
    Distribution::Gaussian {
        mean,
        log_std,
        bounds: bounded.then(|| (vec![-1.], vec![3.])),
    }
}

#[test]
fn reparameterized_quadratic_q_gradients_match_analytic_and_finite_difference() {
    for bounded in [false, true] {
        let mu = matrix(vec![0.1], 1, 1).require_grad();
        let ls = matrix(vec![0.5_f32.ln()], 1, 1).require_grad();
        let distribution = gaussian(mu.clone(), ls.clone(), bounded);
        let sample = distribution
            .reparameterize_with_noise(matrix(vec![0.4], 1, 1))
            .unwrap();
        let target = if bounded { 1.8 } else { 0.6 };
        let action = if bounded {
            1. + 2. * 0.3_f32.tanh()
        } else {
            0.3
        };
        close(sample.actions.clone().into_scalar(), action, 1e-6);
        let loss = (sample.actions - target).powf_scalar(2.).mean();
        let grads = loss.backward();
        let transform_grad = if bounded {
            2. * (1. - 0.3_f32.tanh().powi(2))
        } else {
            1.
        };
        let analytic = 2. * (action - target) * transform_grad;
        let derivative = mu.grad(&grads).unwrap().into_scalar::<f32>();
        close(derivative, analytic, 2e-6);
        close(ls.grad(&grads).unwrap().into_scalar(), analytic * 0.2, 2e-6);
        let objective = |mean: f64| {
            let z = mean + 0.5 * 0.4;
            let a = if bounded { 1. + 2. * z.tanh() } else { z };
            (a - target as f64).powi(2)
        };
        let delta = 1e-5;
        let fd = (objective(0.1 + delta) - objective(0.1 - delta)) / (2. * delta);
        close(derivative, fd as f32, 3e-6);
    }
}

#[test]
fn pathwise_entropy_gradient_and_stored_likelihood_conventions_agree() {
    for bounded in [false, true] {
        let mu = matrix(vec![0.1], 1, 1).require_grad();
        let ls = matrix(vec![0.5_f32.ln()], 1, 1).require_grad();
        let distribution = gaussian(mu.clone(), ls.clone(), bounded);
        let sample = distribution
            .reparameterize_with_noise(matrix(vec![0.4], 1, 1))
            .unwrap();
        let actions = vec![rl::core::action::Action::Continuous(
            to_vec(sample.actions).unwrap(),
        )];
        let latent = vec![Some(to_vec(sample.latents).unwrap())];
        let stored = distribution.log_prob(&actions, &latent).unwrap();
        close(
            sample.log_probs.clone().into_scalar(),
            stored.into_scalar(),
            1e-6,
        );
        let grads = (sample.log_probs * 0.7).sum().backward();
        let expected_mu = if bounded { 1.4 * 0.3_f32.tanh() } else { 0. };
        close(mu.grad(&grads).unwrap().into_scalar(), expected_mu, 2e-6);
        close(
            ls.grad(&grads).unwrap().into_scalar(),
            -0.7 + expected_mu * 0.2,
            2e-6,
        );
    }
}

#[test]
fn pathwise_samples_use_only_the_owned_rng_and_validate_noise() {
    let distribution = gaussian(matrix(vec![0.; 2], 2, 1), matrix(vec![0.; 2], 2, 1), false);
    let mut a = ChaCha8Rng::seed_from_u64(51);
    let mut b = a.clone();
    assert_eq!(
        to_vec(distribution.reparameterize(&mut a).unwrap().actions).unwrap(),
        to_vec(distribution.reparameterize(&mut b).unwrap().actions).unwrap()
    );
    assert_eq!(a.random::<u64>(), b.random::<u64>());
    assert!(
        distribution
            .reparameterize_with_noise(matrix(vec![0.], 1, 1))
            .is_err()
    );
    let categorical = Distribution::Categorical {
        logits: matrix(vec![0., 0.], 1, 2),
        masks: vec![true; 2],
    };
    assert!(categorical.reparameterize(&mut a).is_err());
}
