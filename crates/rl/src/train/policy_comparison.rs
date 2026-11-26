//! Descriptive statistics and optional confidence for matched policy returns.

use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};

/// A priori bounds on each complete policy return, in the chosen objective.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReturnBounds {
    pub minimum: f64,
    pub maximum: f64,
}

impl ReturnBounds {
    fn checked_span(self) -> Result<f64> {
        let span = self.maximum - self.minimum;
        ensure(
            span.is_finite() && span >= 0.0,
            "invalid complete return bounds",
        )?;
        Ok(span)
    }

    fn check(self, score: f64) -> Result<()> {
        ensure(
            (self.minimum..=self.maximum).contains(&score),
            "observed complete return is outside its declared bounds",
        )
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairedDescriptives {
    pub pairs: usize,
    pub incumbent_mean: f64,
    pub challenger_mean: f64,
    pub mean_difference: f64,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    /// Absent for one pair or when the sample deviation exceeds f64 range.
    pub difference_standard_deviation: Option<f64>,
    /// Standard error over the supplied independent matched groups.
    pub difference_standard_error: Option<f64>,
}

/// Summarize `(incumbent, challenger)` return pairs. Supply one pair per
/// independent matched group. Average repeated scenarios within each group
/// before passing the returns here so large groups do not dominate the mean.
pub fn compare_paired_returns(returns: &[(f64, f64)]) -> Result<PairedDescriptives> {
    let rows = returns
        .iter()
        .map(|&(incumbent, challenger)| (incumbent, challenger, challenger - incumbent));
    ensure(
        !returns.is_empty()
            && rows.clone().all(|(incumbent, challenger, difference)| {
                incumbent.is_finite() && challenger.is_finite() && difference.is_finite()
            }),
        "paired returns require nonempty finite inputs and finite differences",
    )?;
    describe(rows)
}

/// One-sided Hoeffding lower bound for a fixed comparison. If each policy
/// return lies in an interval of width R, a paired difference lies in [-R, R].
/// The failure probability applies to the specified comparison with independent
/// matched groups. Evaluation design and any publication decision are caller-owned.
pub fn paired_return_lower_bound(
    returns: &[(f64, f64)],
    bounds: ReturnBounds,
    failure_probability: f64,
) -> Result<f64> {
    let span = bounds.checked_span()?;
    ensure(
        failure_probability > 0.0 && failure_probability < 1.0,
        "invalid paired return failure probability",
    )?;
    for &(incumbent, challenger) in returns {
        bounds.check(incumbent)?;
        bounds.check(challenger)?;
    }
    let comparison = compare_paired_returns(returns)?;
    let multiplier = (-2.0 * failure_probability.ln() / comparison.pairs as f64).sqrt();
    let radius = span * multiplier;
    let lower_bound = if radius.is_finite() {
        comparison.mean_difference - radius
    } else {
        // The difference can be representable even if its radius overflows.
        (-span).mul_add(multiplier, comparison.mean_difference)
    };
    ensure(
        lower_bound.is_finite(),
        "nonfinite paired return lower bound",
    )?;
    Ok(lower_bound)
}

fn update_mean(mean: f64, sample: f64, count: f64) -> f64 {
    let difference = sample - mean;
    if difference.is_finite() {
        mean + difference / count
    } else {
        // An opposite-sign sample can overflow the subtraction even though
        // its weighted contribution is representable.
        mean * ((count - 1.0) / count) + sample / count
    }
}

fn describe(rows: impl Iterator<Item = (f64, f64, f64)> + Clone) -> Result<PairedDescriptives> {
    let mut paired = PairedDescriptives::default();
    let scale = rows
        .clone()
        .map(|(_, _, difference)| difference.abs())
        .fold(0.0, f64::max);
    let mut normalized_mean = 0.0;
    let mut normalized_deviation = 0.0_f64;
    for (incumbent, challenger, difference) in rows {
        paired.pairs += 1;
        let n = paired.pairs as f64;
        if difference > 0.0 {
            paired.wins += 1;
        } else if difference < 0.0 {
            paired.losses += 1;
        } else {
            paired.ties += 1;
        }
        paired.incumbent_mean = update_mean(paired.incumbent_mean, incumbent, n);
        paired.challenger_mean = update_mean(paired.challenger_mean, challenger, n);
        paired.mean_difference = update_mean(paired.mean_difference, difference, n);
        // Scale the dispersion computation by the differences' largest
        // magnitude. Keep the mean itself in original return units.
        if scale > 0.0 {
            let delta = difference / scale - normalized_mean;
            normalized_mean += delta / n;
            normalized_deviation = normalized_deviation.hypot(delta * ((n - 1.0) / n).sqrt());
        }
    }
    if paired.pairs > 1 {
        let n = paired.pairs as f64;
        let normalized_sd = normalized_deviation / (n - 1.0).sqrt();
        let sd = normalized_sd * scale;
        paired.difference_standard_deviation = sd.is_finite().then_some(sd);
        paired.difference_standard_error = Some(normalized_sd / n.sqrt() * scale);
    }
    ensure(
        paired.incumbent_mean.is_finite()
            && paired.challenger_mean.is_finite()
            && paired.mean_difference.is_finite(),
        "nonfinite paired return mean",
    )?;
    Ok(paired)
}
