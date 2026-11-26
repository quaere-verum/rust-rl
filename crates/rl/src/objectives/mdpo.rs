//! Shared unclipped mirror-descent objective; chronological evaluation belongs to callers.
use crate::{
    Error, Result,
    algorithms::recurrent_on_policy::distribution::RecurrentPolicyDistribution,
    core::{action::Action, ensure},
    nn::{finite_loss, to_vec},
};
use burn::tensor::Tensor;

pub(crate) struct MdpoTargets<'a> {
    pub actions: &'a [Action],
    pub latents: &'a [Option<Vec<f32>>],
    pub old_log_probs: &'a Tensor<1>,
    pub weights: &'a Tensor<1>,
    pub advantages: &'a Tensor<1>,
}
pub(crate) struct MdpoObjective {
    pub loss: Tensor<1>,
    pub surrogate: Tensor<1>,
    pub kl: Tensor<1>,
}

/// Preserve the feedforward objective's exact arithmetic and finite checks.
pub(crate) fn mdpo_objective<D: RecurrentPolicyDistribution>(
    current: &D,
    behavior: &D,
    targets: MdpoTargets<'_>,
    coefficient: f32,
) -> Result<MdpoObjective> {
    let ratio =
        (current.log_prob(targets.actions, targets.latents)? - targets.old_log_probs.clone()).exp();
    if !to_vec(ratio.clone())?.iter().all(|x| x.is_finite()) {
        return Err(Error::Numerical(
            "MDPO importance ratio overflow; probability ratios are not clipped".into(),
        ));
    }
    let surrogate = (targets.weights.clone() * ratio * targets.advantages.clone()).sum();
    // The ordering is KL(current || frozen behavior), never the opposite direction.
    let kl = (targets.weights.clone() * current.kl(behavior)?).sum();
    finite_loss(&surrogate)?;
    finite_loss(&kl)?;
    let loss = surrogate.clone().neg() + kl.clone() * coefficient;
    finite_loss(&loss)?;
    Ok(MdpoObjective {
        loss,
        surrogate,
        kl,
    })
}
pub(crate) fn normalized_occupancy_weights(values: &[f32]) -> Result<Vec<f32>> {
    let sum: f64 = values.iter().map(|&x| x as f64).sum();
    ensure(
        sum.is_finite() && sum > 0. && values.iter().all(|x| x.is_finite() && *x >= 0.),
        "MDPO discounted occupancy requires nonnegative weights and positive total mass",
    )?;
    Ok(values.iter().map(|&x| (x as f64 / sum) as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::{MdpoTargets, mdpo_objective, normalized_occupancy_weights};
    use crate::{
        core::action::Action,
        nn::{distributions::Distribution, matrix, to_vec, vector},
    };

    fn categorical(probabilities: &[f32]) -> Distribution {
        Distribution::Categorical {
            logits: matrix(
                probabilities.iter().map(|p| p.ln()).collect(),
                1,
                probabilities.len(),
            ),
            masks: vec![true; probabilities.len()],
        }
    }
    #[test]
    fn asymmetric_current_to_behavior_kl_and_unclipped_ratio_match_scalar_reference() {
        let current = categorical(&[0.6, 0.4]);
        let behavior = categorical(&[0.2, 0.8]);
        let actions = [Action::Discrete(0)];
        let latents = [None];
        let old = vector(&[0.2_f32.ln()]);
        let weights = vector(&[1.]);
        let advantages = vector(&[2.]);
        let result = mdpo_objective(
            &current,
            &behavior,
            MdpoTargets {
                actions: &actions,
                latents: &latents,
                old_log_probs: &old,
                weights: &weights,
                advantages: &advantages,
            },
            0.7,
        )
        .unwrap();
        let forward = 0.6_f64 * 3_f64.ln() + 0.4 * 0.5_f64.ln();
        let reverse = 0.2_f64 * (1_f64 / 3.).ln() + 0.8 * 2_f64.ln();
        assert!((result.surrogate.into_scalar::<f32>() - 6.).abs() < 2e-6);
        assert!((f64::from(result.kl.into_scalar::<f32>()) - forward).abs() < 2e-6);
        assert!((forward - reverse).abs() > 0.04);
        assert!((f64::from(result.loss.into_scalar::<f32>()) - (-6. + 0.7 * forward)).abs() < 2e-6);
    }
    #[test]
    fn occupancy_normalizes_absolute_episode_weights_and_rejects_zero_mass() {
        let weights = normalized_occupancy_weights(&[1., 0.0625, 0.03125]).unwrap();
        for (actual, expected) in weights.iter().zip([32_f32 / 35., 2. / 35., 1. / 35.]) {
            assert!((actual - expected).abs() < 1e-7);
        }
        for invalid in [&[][..], &[0., 0.], &[-1., 2.], &[f32::NAN, 1.]] {
            assert!(normalized_occupancy_weights(invalid).is_err());
        }
    }
    fn reference(logits: &[f64], coefficient: f64) -> f64 {
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|x| (x - maximum).exp()).sum();
        let p: Vec<_> = logits.iter().map(|x| (x - maximum).exp() / total).collect();
        -(p[0] / 0.2) * 1.7 + coefficient * (p[0] * (p[0] / 0.2).ln() + p[1] * (p[1] / 0.8).ln())
    }
    #[test]
    fn unclipped_policy_and_proximity_gradients_match_independent_f64_differences() {
        for coefficient in [0.1_f32, 2.5] {
            let logits = matrix(vec![0.4, -0.7], 1, 2).require_grad();
            let current = Distribution::Categorical {
                logits: logits.clone(),
                masks: vec![true, true],
            };
            let behavior = categorical(&[0.2, 0.8]);
            let actions = [Action::Discrete(0)];
            let latents = [None];
            let old = vector(&[0.2_f32.ln()]);
            let weights = vector(&[1.]);
            let advantages = vector(&[1.7]);
            let objective = mdpo_objective(
                &current,
                &behavior,
                MdpoTargets {
                    actions: &actions,
                    latents: &latents,
                    old_log_probs: &old,
                    weights: &weights,
                    advantages: &advantages,
                },
                coefficient,
            )
            .unwrap();
            let gradient = to_vec(logits.grad(&objective.loss.backward()).unwrap()).unwrap();
            for index in 0..2 {
                let mut plus = [0.4_f32 as f64, -0.7_f32 as f64];
                let mut minus = plus;
                plus[index] += 1e-5;
                minus[index] -= 1e-5;
                let expected = (reference(&plus, f64::from(coefficient))
                    - reference(&minus, f64::from(coefficient)))
                    / 2e-5;
                assert!((f64::from(gradient[index]) - expected).abs() < 3e-6);
            }
        }
    }
}
