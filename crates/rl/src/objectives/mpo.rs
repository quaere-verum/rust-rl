//! Finite-action MPO: a shared reverse-KL E-step dual and weighted M projection.
use crate::{Result, core::ensure};
#[cfg(feature = "neural")]
use burn::tensor::Tensor;
#[cfg(feature = "neural")]
pub use current_policy::sampled_current_policy_continuation;
#[cfg(feature = "neural")]
mod current_policy;

#[derive(Debug, Clone)]
pub struct EStep {
    pub probabilities: Vec<f64>,
    /// None denotes the exact zero-budget identity operator, not an arbitrary temperature.
    pub temperature: Option<f64>,
    pub reverse_kl: f64,
    pub dual: f64,
    pub active: bool,
}

pub fn e_step(
    old: &[f64],
    values: &[f64],
    masks: &[bool],
    actions: usize,
    epsilon: f64,
) -> Result<EStep> {
    ensure(
        actions > 0
            && !old.is_empty()
            && old.len().is_multiple_of(actions)
            && old.len() == values.len()
            && old.len() == masks.len()
            && epsilon.is_finite()
            && epsilon >= 0.,
        "invalid MPO E-step shape or budget",
    )?;
    ensure(
        values.iter().all(|x| x.is_finite())
            && old
                .iter()
                .zip(masks)
                .all(|(&p, &m)| p.is_finite() && p >= 0. && (m || p == 0.))
            && old
                .chunks(actions)
                .all(|r| (r.iter().sum::<f64>() - 1.).abs() < 1e-10),
        "invalid MPO values or masked prior probability mass",
    )?;
    if epsilon == 0. {
        let dual =
            old.iter().zip(values).map(|(p, q)| p * q).sum::<f64>() / (old.len() / actions) as f64;
        return Ok(EStep {
            probabilities: old.to_vec(),
            temperature: None,
            reverse_kl: 0.,
            dual,
            active: false,
        });
    }
    let rows = old.len() / actions;
    let maxima: Vec<_> = old
        .chunks(actions)
        .zip(values.chunks(actions))
        .map(|(p, q)| {
            p.iter()
                .zip(q)
                .filter(|(p, _)| **p > 0.)
                .map(|(_, q)| *q)
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    // Solve in units of the largest supported within-history Q difference.
    // This removes any absolute temperature floor and preserves the operator
    // even when finite advantages are many orders of magnitude below one.
    let range = old
        .chunks(actions)
        .zip(values.chunks(actions))
        .zip(&maxima)
        .flat_map(|((p, q), &m)| {
            p.iter()
                .zip(q)
                .filter(|(p, _)| **p > 0.)
                .map(move |(_, q)| m - q)
        })
        .fold(0., f64::max);
    let scale = if range.is_infinite() {
        values.iter().map(|q| q.abs()).fold(0., f64::max)
    } else if range > 0. {
        range
    } else {
        1.
    };
    let evaluate = |eta: f64| {
        let mut result = Vec::with_capacity(old.len());
        let mut kl = 0.;
        let mut dual = 0.;
        for ((p, q), &max) in old.chunks(actions).zip(values.chunks(actions)).zip(&maxima) {
            let logs: Vec<_> = p
                .iter()
                .zip(q)
                .map(|(&p, &q)| {
                    if p == 0. {
                        f64::NEG_INFINITY
                    } else if eta == 0. {
                        if q == max { p.ln() } else { f64::NEG_INFINITY }
                    } else {
                        let difference = q - max;
                        let centered = if difference.is_finite() {
                            difference / scale
                        } else {
                            q / scale - max / scale
                        };
                        p.ln() + centered / eta
                    }
                })
                .collect();
            let log_max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = logs.iter().map(|l| (l - log_max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            for (&weight, &prior) in weights.iter().zip(p) {
                let posterior = weight / sum;
                if posterior > 0. {
                    kl += posterior * (posterior.ln() - prior.ln());
                }
                result.push(posterior);
            }
            dual += (max + eta * scale * (log_max + sum.ln())) / rows as f64;
        }
        (result, kl / rows as f64, dual + eta * scale * epsilon)
    };
    let greedy = evaluate(0.);
    if greedy.1 <= epsilon {
        return Ok(EStep {
            probabilities: greedy.0,
            temperature: Some(0.),
            reverse_kl: greedy.1,
            dual: greedy.2,
            active: false,
        });
    }
    // Log-sum-exp keeps both rare priors and dominant posteriors representable.
    // eta is shared across histories and only the mean KL is constrained.
    let mut high = 1.;
    let mut low = 0.;
    for _ in 0..1024 {
        if evaluate(high).1 <= epsilon {
            break;
        }
        high *= 2.;
        ensure(high.is_finite(), "MPO temperature bracket overflow")?;
    }
    for _ in 0..80 {
        let mid = (low + high) * 0.5;
        if evaluate(mid).1 > epsilon {
            low = mid;
        } else {
            high = mid;
        }
    }
    let (probabilities, reverse_kl, dual) = evaluate(high);
    ensure(
        dual.is_finite() && (high * scale).is_finite() && reverse_kl.is_finite(),
        "nonfinite MPO dual solution",
    )?;
    Ok(EStep {
        probabilities,
        temperature: Some(high * scale),
        reverse_kl,
        dual,
        active: true,
    })
}

#[cfg(feature = "neural")]
pub fn weighted_projection(log_probabilities: Tensor<2>, q: Tensor<2>) -> Tensor<1> {
    (log_probabilities * q.without_autodiff())
        .sum_dim(1)
        .mean()
        .neg()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "neural")]
    use crate::nn::{matrix, to_vec};
    #[cfg(feature = "neural")]
    use burn::tensor::activation::log_softmax;

    #[test]
    fn dual_solution_matches_independent_two_action_calculus() {
        // For eta=1, pi=(1/2,1/2), Q=(0,ln3), q=(1/4,3/4).
        let budget = 0.25f64 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
        let r = e_step(&[0.5, 0.5], &[0., 3f64.ln()], &[true; 2], 2, budget).unwrap();
        assert!(r.active);
        assert!((r.temperature.unwrap() - 1.).abs() < 1e-9);
        assert!((r.probabilities[0] - 0.25).abs() < 1e-10);
        assert!((r.reverse_kl - budget).abs() < 1e-10);
        let primal = 0.75 * 3f64.ln();
        assert!((r.dual - primal).abs() < 1e-10);
        // Direct independent dual finite differences vanish at the returned eta.
        let g = |eta: f64| eta * budget + eta * (0.5 + 0.5 * (3f64.ln() / eta).exp()).ln();
        assert!(((g(1.00001) - g(0.99999)) / 0.00002).abs() < 1e-9);
    }

    #[test]
    fn one_shared_temperature_controls_mean_not_each_history_kl() {
        let old = [0.5; 4];
        let r = e_step(&old, &[0., 1., 0., 4.], &[true; 4], 2, 0.08).unwrap();
        let kl = |p: &[f64]| {
            p.iter()
                .filter(|p| **p > 0.)
                .map(|p| p * (2. * p).ln())
                .sum::<f64>()
        };
        let first = kl(&r.probabilities[..2]);
        let second = kl(&r.probabilities[2..]);
        assert!(first < 0.08 && second > 0.08);
        assert!(((first + second) / 2. - 0.08).abs() < 1e-10);
        let shifted = e_step(&old, &[1000., 1003., -200., -188.], &[true; 4], 2, 0.08).unwrap();
        for (a, b) in r.probabilities.iter().zip(shifted.probabilities) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!((shifted.temperature.unwrap() / r.temperature.unwrap() - 3.).abs() < 1e-9);
    }

    #[test]
    fn inactive_ties_masks_and_zero_support_preserve_old_relative_mass() {
        let r = e_step(
            &[0.2, 0.3, 0.5, 0., 0.],
            &[7., 7., 0., 99., 1000.],
            &[true, true, true, true, false],
            5,
            2.,
        )
        .unwrap();
        for (actual, expected) in r.probabilities.iter().zip([0.4, 0.6, 0., 0., 0.]) {
            assert!((actual - expected).abs() < 1e-15);
        }
        assert!(!r.active);
        assert_eq!(r.temperature, Some(0.));
        assert!((r.reverse_kl - 2f64.ln()).abs() < 1e-12);
        assert_eq!(r.dual, 7.);
        let identity = e_step(&[0.2, 0.8], &[1., 4.], &[true; 2], 2, 0.).unwrap();
        assert_eq!(identity.probabilities, [0.2, 0.8]);
        assert_eq!(identity.temperature, None);
        assert_eq!(identity.reverse_kl, 0.);
    }

    #[test]
    fn active_dual_never_exponentiates_an_unsupported_large_q_value() {
        let budget = 0.25f64 * 0.5f64.ln() + 0.75 * 1.5f64.ln();
        for legal in [true, false] {
            let r = e_step(
                &[0.5, 0.5, 0.],
                &[0., 3f64.ln(), 1e300],
                &[true, true, legal],
                3,
                budget,
            )
            .unwrap();
            assert!((r.temperature.unwrap() - 1.).abs() < 1e-9);
            assert!((r.probabilities[0] - 0.25).abs() < 1e-10);
            assert_eq!(r.probabilities[2], 0.);
            assert!((r.reverse_kl - budget).abs() < 1e-10);
        }
    }

    #[test]
    fn tiny_prior_has_finite_log_kl_and_attains_the_active_budget() {
        let rare = 1e-320f64;
        let r = e_step(&[1., rare], &[0., 1.], &[true; 2], 2, 0.1).unwrap();
        let q = r.probabilities[1];
        // q=1e-4 is independently feasible, so the maximizer must beat it.
        let kl = |q: f64| (1. - q) * (1. - q).ln() + q * (q.ln() - rare.ln());
        assert!(kl(1e-4) < 0.1);
        assert!(q > 1e-4 && q < 2e-4, "q_rare={q}");
        assert!((kl(q) - 0.1).abs() < 1e-9);
        assert!((r.reverse_kl - 0.1).abs() < 1e-9);
    }

    #[test]
    fn arbitrarily_small_finite_advantages_preserve_the_kl_solution() {
        let base = e_step(&[0.5, 0.5], &[0., 1.], &[true; 2], 2, 0.1).unwrap();
        for scale in [1e-50, 1e-300] {
            let r = e_step(&[0.5, 0.5], &[0., scale], &[true; 2], 2, 0.1).unwrap();
            assert!((r.probabilities[1] - base.probabilities[1]).abs() < 1e-10);
            assert!((r.reverse_kl - 0.1).abs() < 1e-10);
            assert!((r.temperature.unwrap() / scale / base.temperature.unwrap() - 1.).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_shapes_probability_mass_and_nonfinite_inputs_reject() {
        for (old, q, mask, k, budget) in [
            (vec![0.5], vec![1.], vec![true], 2, 0.1),
            (vec![0.2, 0.2], vec![1., 2.], vec![true; 2], 2, 0.1),
            (vec![0.5, 0.5], vec![1., 2.], vec![true, false], 2, 0.1),
            (vec![0.5, 0.5], vec![1., f64::NAN], vec![true; 2], 2, 0.1),
            (vec![0.5, 0.5], vec![1., 2.], vec![true; 2], 2, -0.1),
        ] {
            assert!(e_step(&old, &q, &mask, k, budget).is_err());
        }
    }

    #[test]
    #[cfg(feature = "neural")]
    fn projection_is_weighted_forward_cross_entropy_with_correct_logit_gradient() {
        let logits = matrix(vec![0., 3f32.ln(), 2f32.ln(), 0.], 2, 2).require_grad();
        let q = matrix(vec![0.75, 0.25, 0.25, 0.75], 2, 2);
        let loss = weighted_projection(log_softmax(logits.clone(), 1), q);
        let expected = -(0.75f64 * 0.25f64.ln()
            + 0.25 * 0.75f64.ln()
            + 0.25 * (2f64 / 3.).ln()
            + 0.75 * (1f64 / 3.).ln())
            / 2.;
        assert!((loss.clone().into_scalar::<f32>() as f64 - expected).abs() < 1e-6);
        let gradients = loss.backward();
        let actual = to_vec(logits.grad(&gradients).unwrap()).unwrap();
        for (actual, expected) in actual.iter().zip([-0.25, 0.25, 5. / 24., -5. / 24.]) {
            assert!((*actual as f64 - expected).abs() < 1e-6);
        }
    }
}
