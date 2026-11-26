//! Shared tensor mathematics; learner orchestration and random streams remain local.
use burn::tensor::Tensor;

/// Importance ratio and the existing pre-update PPO KL proxy, in the original order.
pub(crate) fn ppo_ratio_and_kl(logp: Tensor<1>, old_logp: Tensor<1>) -> (Tensor<1>, Tensor<1>) {
    let logratio = logp - old_logp;
    let ratio = logratio.clone().exp();
    let kl = (ratio.clone() - 1. - logratio).mean();
    (ratio, kl)
}

/// Negative clipped policy surrogate averaged over the caller's learning rows.
pub(crate) fn ppo_clipped_surrogate(
    ratio: Tensor<1>,
    advantages: Tensor<1>,
    clip: f32,
) -> Tensor<1> {
    let objective = (ratio.clone() * advantages.clone())
        .min_pair(ratio.clamp(1. - clip, 1. + clip) * advantages);
    objective.mean().neg()
}

#[cfg(test)]
mod tests {
    use super::{ppo_clipped_surrogate, ppo_ratio_and_kl};
    use crate::nn::{to_vec, vector};

    #[test]
    fn ppo_ratio_and_kl_match_literal_values_and_scalar_gradient() {
        let current = vector(&[0.5_f32.ln(), 0., 2_f32.ln(), 4_f32.ln()]).require_grad();
        let (ratio, kl) = ppo_ratio_and_kl(current.clone(), vector(&[0.; 4]));
        let ratios = to_vec(ratio).unwrap();
        for (actual, expected) in ratios.iter().zip([0.5, 1., 2., 4.]) {
            assert!((actual - expected).abs() < 3e-7);
        }
        assert!((kl.clone().into_scalar::<f32>() as f64 - 0.5284264097200273).abs() < 3e-7);
        let gradients = to_vec(current.grad(&kl.backward()).unwrap()).unwrap();
        for (actual, expected) in gradients.iter().zip([-0.125, 0., 0.25, 0.75]) {
            assert!((actual - expected).abs() < 3e-7);
        }
    }

    #[test]
    fn ppo_clamp_uses_both_advantage_signs_and_the_mean_reduction() {
        let ratio = vector(&[0.5, 0.5, 1., 1., 1.5, 1.5]).require_grad();
        let loss = ppo_clipped_surrogate(ratio.clone(), vector(&[2., -2., 2., -2., 2., -2.]), 0.2);
        assert!((loss.clone().into_scalar::<f32>() - 0.2).abs() < 3e-7);
        let gradients = to_vec(ratio.grad(&loss.backward()).unwrap()).unwrap();
        for (actual, expected) in
            gradients
                .iter()
                .zip([-1. / 3., 0., -1. / 3., 1. / 3., 0., 1. / 3.])
        {
            assert!((actual - expected).abs() < 3e-7);
        }
        for (ratio, advantage, expected) in [
            (0.5, 2., -1.),
            (0.5, -2., 1.6),
            (1., 2., -2.),
            (1., -2., 2.),
            (1.5, 2., -2.4),
            (1.5, -2., 3.),
        ] {
            let actual = ppo_clipped_surrogate(vector(&[ratio]), vector(&[advantage]), 0.2)
                .into_scalar::<f32>();
            assert!((actual - expected).abs() < 3e-7);
        }
    }

    fn scalar_loss(current: &[f64], old: &[f64], advantages: &[f64], clip: f64) -> f64 {
        let mut total = 0.;
        for i in 0..current.len() {
            let ratio = (current[i] - old[i]).exp();
            let bounded = if ratio < 1. - clip {
                1. - clip
            } else if ratio > 1. + clip {
                1. + clip
            } else {
                ratio
            };
            let ordinary = ratio * advantages[i];
            let clipped = bounded * advantages[i];
            total -= if ordinary < clipped {
                ordinary
            } else {
                clipped
            };
        }
        total / current.len() as f64
    }

    #[test]
    fn ppo_log_likelihood_gradients_match_independent_f64_finite_differences() {
        let old = [-1.2_f32, -0.8, -2.3, -0.1, -1.7, -0.4];
        let ratios = [0.45_f32, 0.65, 0.9, 1.1, 1.55, 1.8];
        let advantages = [2_f32, -3., 1.5, -0.7, 4., -2.];
        let current: Vec<_> = old
            .iter()
            .zip(ratios)
            .map(|(&old, ratio)| old + ratio.ln())
            .collect();
        for clip in [0_f32, 0.2, 0.4] {
            let logp = vector(&current).require_grad();
            let (ratio, _) = ppo_ratio_and_kl(logp.clone(), vector(&old));
            let loss = ppo_clipped_surrogate(ratio, vector(&advantages), clip);
            let old = old.map(f64::from);
            let advantages = advantages.map(f64::from);
            let current: Vec<_> = current.iter().map(|&x| f64::from(x)).collect();
            let expected = scalar_loss(&current, &old, &advantages, f64::from(clip));
            assert!((f64::from(loss.clone().into_scalar::<f32>()) - expected).abs() < 2e-6);
            let gradient = to_vec(logp.grad(&loss.backward()).unwrap()).unwrap();
            for i in 0..gradient.len() {
                let mut plus = current.clone();
                let mut minus = current.clone();
                plus[i] += 1e-5;
                minus[i] -= 1e-5;
                let expected = (scalar_loss(&plus, &old, &advantages, f64::from(clip))
                    - scalar_loss(&minus, &old, &advantages, f64::from(clip)))
                    / 2e-5;
                assert!(
                    (f64::from(gradient[i]) - expected).abs() < 3e-6,
                    "clip={clip} row={i} autodiff={} finite_difference={expected}",
                    gradient[i]
                );
            }
        }
    }

    #[test]
    fn ppo_zero_clip_and_zero_advantages_keep_their_boundary_semantics() {
        let ratio = vector(&[0.5, 0.5, 1.5, 1.5, 7.]).require_grad();
        let loss = ppo_clipped_surrogate(ratio.clone(), vector(&[2., -2., 2., -2., 0.]), 0.);
        assert!((loss.clone().into_scalar::<f32>() - 0.4).abs() < 1e-7);
        let gradient = to_vec(ratio.grad(&loss.backward()).unwrap()).unwrap();
        assert_eq!(gradient, [-0.4, 0., 0., 0.4, 0.]);
        let current = vector(&[-7., -2., -0.1]).require_grad();
        let (_, kl) = ppo_ratio_and_kl(current.clone(), vector(&[-7., -2., -0.1]));
        assert_eq!(kl.clone().into_scalar::<f32>(), 0.);
        assert_eq!(
            to_vec(current.grad(&kl.backward()).unwrap()).unwrap(),
            [0.; 3]
        );
    }
}
