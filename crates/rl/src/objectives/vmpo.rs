//! V-MPO's stable selection, temperature weighting and independent dual objective.
use crate::nn::vector;
use burn::tensor::{Tensor, activation::softplus};

pub(crate) fn top_half_indices_into(advantages: &[f32], top: &mut Vec<usize>) {
    top.clear();
    top.extend(0..advantages.len());
    top.sort_by(|&a, &b| advantages[b].total_cmp(&advantages[a]));
    top.truncate(advantages.len().div_ceil(2));
}

pub(crate) fn temperature_weights(advantages: &[f32], eta: f32) -> (Tensor<1>, f32) {
    let maximum = advantages.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = vector(advantages).sub_scalar(maximum).div_scalar(eta).exp();
    let weights = weights.clone() / weights.sum();
    (weights, maximum)
}

pub(crate) fn policy_objective(
    weights: Tensor<1>,
    log_prob: Tensor<1>,
    kl: Tensor<1>,
    alpha: f32,
) -> Tensor<1> {
    (weights * log_prob).sum().neg() + kl * alpha
}

pub(crate) fn dual_objective(
    log_eta: Tensor<1>,
    log_alpha: Tensor<1>,
    advantages: &[f32],
    maximum: f32,
    temperature_constraint: f32,
    kl_constraint: f32,
    kl_number: f32,
) -> Tensor<1> {
    let e = softplus(log_eta, 1.) + 1e-6;
    let a = softplus(log_alpha, 1.) + 1e-6;
    e.clone() * temperature_constraint
        + e.clone() * ((vector(advantages) - maximum) / e).exp().mean().log()
        + maximum
        + a * (kl_constraint - kl_number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::to_vec;

    #[test]
    fn stable_top_half_retains_boundary_ties_and_ceil() {
        let mut selected = Vec::new();
        top_half_indices_into(&[4., 4., 0., 4., 4.], &mut selected);
        assert_eq!(selected, [0, 1, 3]);
        top_half_indices_into(&[2., 9., 9., 3., 0.], &mut selected);
        assert_eq!(selected, [1, 2, 3]);
        top_half_indices_into(&[7.], &mut selected);
        assert_eq!(selected, [0]);
    }

    #[test]
    fn stable_temperature_weights_match_two_to_one_hand_reference() {
        let (weights, maximum) = temperature_weights(&[0., std::f32::consts::LN_2], 1.);
        assert_eq!(maximum, std::f32::consts::LN_2);
        let values = to_vec(weights).unwrap();
        assert!((values[0] - 1. / 3.).abs() < 1e-7);
        assert!((values[1] - 2. / 3.).abs() < 1e-7);
        let (large, _) = temperature_weights(&[1_000_000., 999_999.], 0.5);
        let values = to_vec(large).unwrap();
        let expected = 1. / (1. + (-2f64).exp());
        assert!((values[0] as f64 - expected).abs() < 1e-7);
    }

    #[test]
    fn dual_value_and_both_gradients_match_independent_scalar_calculus() {
        let log_eta = vector(&[0.05]).require_grad();
        let log_alpha = vector(&[0.1]).require_grad();
        let objective = dual_objective(
            log_eta.clone(),
            log_alpha.clone(),
            &[2., 2., 1.],
            2.,
            0.01,
            0.1,
            0.25,
        );
        let actual = to_vec(objective.clone()).unwrap()[0] as f64;
        let eta = (1. + 0.05f64.exp()).ln() + 1e-6;
        let alpha = (1. + 0.1f64.exp()).ln() + 1e-6;
        let low = (-1. / eta).exp();
        let log_mean = ((2. + low) / 3.).ln();
        let expected = eta * 0.01 + eta * log_mean + 2. + alpha * (0.1 - 0.25);
        let eta_gradient = (0.01 + log_mean + low / ((2. + low) * eta)) / (1. + (-0.05f64).exp());
        let alpha_gradient = (0.1 - 0.25) / (1. + (-0.1f64).exp());
        let gradients = objective.backward();
        let actual_eta = to_vec(log_eta.grad(&gradients).unwrap()).unwrap()[0] as f64;
        let actual_alpha = to_vec(log_alpha.grad(&gradients).unwrap()).unwrap()[0] as f64;
        assert!((actual - expected).abs() < 2e-6);
        assert!((actual_eta - eta_gradient).abs() < 2e-6);
        assert!((actual_alpha - alpha_gradient).abs() < 2e-6);
        assert!(
            actual_alpha < 0.,
            "violated KL constraint must raise alpha under gradient descent"
        );
    }
}
