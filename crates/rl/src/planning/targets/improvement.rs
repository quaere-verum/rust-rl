//! Search statistics -> distributions. No neural model or optimizer dependency.
use super::{PolicyTarget, validate_prior, validate_support};
use crate::planning::ActionEstimate;
use crate::{Error, Result, core::ensure, objectives::mpo::e_step};
use std::num::NonZeroU32;

pub(super) fn ordered_estimates<'a, A: PartialEq>(
    feasible: &[A],
    estimates: &'a [ActionEstimate<A>],
) -> Result<Vec<Option<&'a ActionEstimate<A>>>> {
    validate_support(feasible)?;
    let mut ordered = vec![None; feasible.len()];
    for estimate in estimates {
        let index = feasible
            .iter()
            .position(|a| a == &estimate.action)
            .ok_or_else(|| Error::State("search estimate is outside feasible support".into()))?;
        ensure(
            ordered[index].is_none() && estimate.value.is_finite() && estimate.visits != Some(0),
            "duplicate, nonfinite or unvisited search estimate",
        )?;
        ordered[index] = Some(estimate);
    }
    Ok(ordered)
}

impl PolicyTarget {
    /// Complete ordered visit law; unvisited actions retain zero probability.
    /// At zero temperature, use the first maximal count in feasible order.
    pub fn from_visits<A: PartialEq>(
        feasible: &[A],
        estimates: &[ActionEstimate<A>],
        temperature: f64,
    ) -> Result<Self> {
        ensure(
            temperature.is_finite() && temperature >= 0.,
            "invalid visit temperature",
        )?;
        let counts = ordered_estimates(feasible, estimates)?
            .into_iter()
            .map(|estimate| match estimate {
                None => Ok(0),
                Some(estimate) => estimate.visits.filter(|n| *n > 0).ok_or_else(|| {
                    Error::State("visit target requires positive sampled visit counts".into())
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum = counts.iter().copied().max().unwrap_or(0);
        ensure(maximum > 0, "visit target has no completed visits")?;
        let mut probabilities = vec![0.; counts.len()];
        if temperature == 0. {
            probabilities[counts
                .iter()
                .position(|&n| n == maximum)
                .expect("positive maximum")] = 1.;
        } else {
            for (p, &n) in probabilities.iter_mut().zip(&counts) {
                *p = if n == 0 {
                    0.
                } else if temperature == 1. {
                    f64::from(n)
                } else {
                    ((f64::from(n).ln() - f64::from(maximum).ln()) / temperature).exp()
                };
            }
            let total: f64 = probabilities.iter().sum();
            for p in &mut probabilities {
                *p /= total;
            }
        }
        validate_prior(feasible, &probabilities)?;
        Ok(Self::Probabilities(probabilities))
    }

    /// Maximize expected search value subject to KL(q || prior) <= epsilon,
    /// using MPO's existing E-step without changing its temperature convention.
    /// Every legal action must meet the visit threshold, even at zero prior mass.
    pub fn from_action_values<A: PartialEq>(
        feasible: &[A],
        estimates: &[ActionEstimate<A>],
        prior: &[f64],
        epsilon: f64,
        minimum_visits: NonZeroU32,
    ) -> Result<Self> {
        validate_prior(feasible, prior)?;
        let values = ordered_estimates(feasible, estimates)?.into_iter().map(|estimate| {
            estimate.filter(|e| e.visits.is_some_and(|n| n >= minimum_visits.get()))
                .map(|e| e.value).ok_or_else(|| Error::State("search value improvement requires adequately visited complete action support".into()))
        }).collect::<Result<Vec<_>>>()?;
        let improved = e_step(
            prior,
            &values,
            &vec![true; feasible.len()],
            feasible.len(),
            epsilon,
        )?;
        validate_prior(feasible, &improved.probabilities)?;
        Ok(Self::Probabilities(improved.probabilities))
    }
}
