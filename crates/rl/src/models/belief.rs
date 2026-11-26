//! Finite Bayesian beliefs and discrete approximations to Beta priors.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};

const VERSION: u32 = 1;
const NORMALIZED_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// An immutable probability distribution over a caller-ordered finite hypothesis space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "BeliefRecord")]
pub struct FiniteBelief {
    version: u32,
    masses: Vec<f64>,
}

impl FiniteBelief {
    /// Normalize finite nonnegative weights with positive total mass.
    ///
    /// A positive weight whose normalized mass rounds to zero is a numerical error.
    pub fn new(mut weights: Vec<f64>) -> Result<Self> {
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(Error::Numerical(
                "belief weights must be finite and nonnegative".into(),
            ));
        }
        let maximum = weights.iter().copied().fold(0.0, f64::max);
        if maximum == 0.0 {
            return Err(Error::Config(
                "belief requires positive total weight".into(),
            ));
        }
        let scaled_total = compensated_sum(weights.iter().map(|weight| weight / maximum));
        let total = maximum * scaled_total;
        for weight in &mut weights {
            let mass = if total.is_finite() {
                *weight / total
            } else {
                (*weight / scaled_total) / maximum
            };
            if *weight > 0.0 && mass == 0.0 {
                return Err(Error::Numerical(
                    "normalized belief mass underflowed".into(),
                ));
            }
            *weight = mass;
        }
        Ok(Self::from_normalized(weights))
    }

    /// Probabilities in the original hypothesis order.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Return the predictive event probability and the posterior in the same hypothesis order.
    ///
    /// Supply one probability per hypothesis, including hypotheses with zero prior mass.
    /// Exactly impossible events return `(0.0, None)`. Positive evidence or posterior mass
    /// that rounds to zero is a numerical error.
    pub fn conditioned(
        &self,
        likelihoods: impl IntoIterator<Item = f64>,
    ) -> Result<(f64, Option<Self>)> {
        let mut likelihoods = likelihoods.into_iter();
        let mut joint = Vec::with_capacity(self.masses.len());
        let mut rescale = false;
        for &prior in &self.masses {
            let likelihood = likelihoods.next().ok_or_else(likelihood_shape)?;
            if !(0.0..=1.0).contains(&likelihood) {
                return Err(Error::Numerical(
                    "event likelihoods must be finite probabilities".into(),
                ));
            }
            rescale |= prior > 0.0 && likelihood > 0.0 && prior * likelihood < f64::MIN_POSITIVE;
            joint.push(likelihood);
        }
        if likelihoods.next().is_some() {
            return Err(likelihood_shape());
        }
        if rescale {
            return self.conditioned_with_scaling(joint);
        }

        let evidence =
            compensated_sum(joint.iter_mut().zip(&self.masses).map(|(weight, prior)| {
                *weight *= prior;
                *weight
            }));
        if evidence == 0.0 {
            return Ok((0.0, None));
        }
        for weight in &mut joint {
            *weight /= evidence;
        }
        Ok((evidence.min(1.0), Some(Self::from_normalized(joint))))
    }

    fn from_normalized(masses: Vec<f64>) -> Self {
        Self {
            version: VERSION,
            masses,
        }
    }

    fn conditioned_with_scaling(&self, mut likelihoods: Vec<f64>) -> Result<(f64, Option<Self>)> {
        let parts = || {
            self.masses
                .iter()
                .copied()
                .zip(likelihoods.iter().copied())
                .filter(|&(prior, likelihood)| prior > 0.0 && likelihood > 0.0)
                .map(|(prior, likelihood)| product_parts(prior, likelihood))
        };
        // Use a common power of two so the largest scaled product lies in [1, 2).
        let scale = parts().map(|(_, exponent)| exponent - 1).max().unwrap_or(0);
        let scaled_evidence = compensated_sum(
            parts().map(|(fraction, exponent)| libm::scalbn(fraction, exponent - scale)),
        );
        let evidence = libm::scalbn(scaled_evidence, scale);
        if evidence == 0.0 {
            return Err(Error::Numerical(
                "positive event evidence underflowed".into(),
            ));
        }

        for (likelihood, &prior) in likelihoods.iter_mut().zip(&self.masses) {
            if *likelihood == 0.0 || prior == 0.0 {
                *likelihood = 0.0;
                continue;
            }
            let (fraction, exponent) = product_parts(prior, *likelihood);
            // Delay rounding to a subnormal until after dividing out the scaled normalizer.
            *likelihood = libm::scalbn(fraction / scaled_evidence, exponent - scale);
            if *likelihood == 0.0 {
                return Err(Error::Numerical(
                    "positive posterior mass underflowed".into(),
                ));
            }
        }
        Ok((evidence.min(1.0), Some(Self::from_normalized(likelihoods))))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BeliefRecord {
    version: u32,
    masses: Vec<f64>,
}

impl TryFrom<BeliefRecord> for FiniteBelief {
    type Error = Error;

    fn try_from(record: BeliefRecord) -> Result<Self> {
        if record.version != VERSION {
            return Err(Error::State("unsupported finite belief version".into()));
        }
        if record.masses.iter().any(|mass| !(0.0..=1.0).contains(mass))
            || (compensated_sum(record.masses.iter().copied()) - 1.0).abs() > NORMALIZED_TOLERANCE
        {
            return Err(Error::State(
                "restored finite belief masses must be normalized probabilities".into(),
            ));
        }
        Ok(Self::from_normalized(record.masses))
    }
}

fn likelihood_shape() -> Error {
    Error::Shape("finite belief requires one likelihood per hypothesis".into())
}

fn product_parts(left: f64, right: f64) -> (f64, i32) {
    let (left, left_exponent) = libm::frexp(left);
    let (right, right_exponent) = libm::frexp(right);
    let (fraction, exponent) = libm::frexp(left * right);
    (fraction, left_exponent + right_exponent + exponent)
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let sum = total + adjusted;
        correction = (sum - total) - adjusted;
        total = sum;
    }
    total
}

/// Fixed nodes and normalized weights for a discrete approximation of a Beta prior.
#[derive(Clone, Debug, PartialEq)]
pub struct BetaQuadrature {
    nodes: Vec<f64>,
    prior: FiniteBelief,
}

impl BetaQuadrature {
    /// Increasing midpoints in the open unit interval.
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Probability mass assigned to each corresponding node.
    pub fn masses(&self) -> &[f64] {
        self.prior.masses()
    }
}

/// Approximate a Beta prior using equal-width midpoint integration.
///
/// Shapes must be finite and positive. Endpoint singularities for shapes below one can
/// require finer grids. This is a finite approximation whose error depends on the prior,
/// likelihood and requested resolution.
pub fn beta_quadrature(alpha: f64, beta: f64, bins: usize) -> Result<BetaQuadrature> {
    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return Err(Error::Config(
            "Beta shapes must be positive and finite".into(),
        ));
    }
    // Above this size the f64 half-integer midpoint numerator can no longer be exact.
    if bins == 0 || bins as u128 > 1u128 << 52 {
        return Err(Error::Config("invalid Beta quadrature resolution".into()));
    }
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for buffer in [&mut nodes, &mut weights] {
        buffer.try_reserve_exact(bins).map_err(|_| {
            Error::Shape("Beta quadrature resolution exceeds available storage".into())
        })?;
    }
    let mut maximum = f64::NEG_INFINITY;
    for index in 0..bins {
        let node = (index as f64 + 0.5) / bins as f64;
        let weight = (alpha - 1.0) * node.ln() + (beta - 1.0) * (-node).ln_1p();
        if !weight.is_finite() {
            return Err(Error::Numerical(
                "Beta log density is unrepresentable".into(),
            ));
        }
        maximum = maximum.max(weight);
        nodes.push(node);
        weights.push(weight);
    }
    let log_total = compensated_sum(weights.iter().map(|weight| (weight - maximum).exp())).ln();
    for weight in &mut weights {
        *weight = ((*weight - maximum) - log_total).exp();
        if *weight == 0.0 {
            return Err(Error::Numerical(
                "positive Beta quadrature mass underflowed".into(),
            ));
        }
    }
    Ok(BetaQuadrature {
        nodes,
        prior: FiniteBelief::from_normalized(weights),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    #[test]
    fn normalizes_large_and_small_finite_inputs() {
        let least = f64::from_bits(1);
        for input in [f64::MAX, least] {
            let prior = FiniteBelief::new(vec![input, input, 0.]).unwrap();
            assert_eq!(prior.masses(), &[0.5, 0.5, 0.]);
        }
    }

    #[test]
    fn rejects_invalid_weights_and_underflowing_positive_prior_mass() {
        let least = f64::from_bits(1);
        for weights in [
            vec![],
            vec![0.],
            vec![1., -0.1],
            vec![1., f64::INFINITY],
            vec![1., f64::NAN],
            vec![f64::MAX, least],
        ] {
            assert!(FiniteBelief::new(weights).is_err());
        }
    }

    #[test]
    fn sequential_joint_updates_match_independent_enumeration() {
        // Two hidden states crossed with two possible observation laws.
        let prior = FiniteBelief::new(vec![1., 2., 3., 4.]).unwrap();
        let (likelihood, after_signal) = prior.conditioned([0.25, 0.75, 0.75, 0.25]).unwrap();
        close(likelihood, 0.5, 2e-16);
        let after_signal = after_signal.unwrap();
        for (&actual, expected) in after_signal.masses().iter().zip([0.05, 0.3, 0.45, 0.2]) {
            close(actual, expected, 2e-16);
        }
        let (likelihood, after_calibration) =
            after_signal.conditioned([0.9, 0.9, 0.2, 0.2]).unwrap();
        close(likelihood, 89. / 200., 2e-16);
        for (&actual, numerator) in after_calibration
            .unwrap()
            .masses()
            .iter()
            .zip([9., 54., 18., 8.])
        {
            close(actual, numerator / 89., 2e-16);
        }
        assert_eq!(prior.masses(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn distinguishes_invalid_likelihoods_from_a_structurally_impossible_event() {
        let prior = FiniteBelief::new(vec![1., 0.]).unwrap();
        assert_eq!(prior.conditioned([0., 1.]).unwrap(), (0., None));
        for likelihoods in [
            vec![0.],
            vec![0., 0., 0.],
            vec![0., -0.1],
            vec![0., 1.01],
            vec![0., f64::NAN],
            vec![0., f64::INFINITY],
        ] {
            assert!(prior.conditioned(likelihoods).is_err());
        }
        assert_eq!(prior.masses(), &[1., 0.]);
    }

    #[test]
    fn preserves_relative_mass_for_representable_subnormal_evidence() {
        let least = f64::from_bits(1);
        let prior = FiniteBelief::new(vec![1., 3.]).unwrap();
        let (evidence, posterior) = prior.conditioned([least, least]).unwrap();
        assert_eq!(evidence.to_bits(), 1);
        assert_eq!(posterior.unwrap(), prior);

        let prior = FiniteBelief::new(vec![1., 1.]).unwrap();
        let (evidence, posterior) = prior.conditioned([least, least * 3.]).unwrap();
        assert_eq!(evidence.to_bits(), 2);
        assert_eq!(posterior.unwrap().masses(), &[0.25, 0.75]);
    }

    #[test]
    fn reports_unrepresentable_positive_evidence_and_posterior_mass() {
        let least = f64::from_bits(1);
        let uniform = FiniteBelief::new(vec![1., 1.]).unwrap();
        let sparse = FiniteBelief::new(vec![least, 1.]).unwrap();
        assert!(matches!(
            uniform.conditioned([least, 0.]),
            Err(Error::Numerical(_))
        ));
        assert!(matches!(
            sparse.conditioned([least, 1.]),
            Err(Error::Numerical(_))
        ));
    }

    #[test]
    fn checked_snapshot_restores_exact_masses() {
        let prior = FiniteBelief::new(vec![1., 6., 13., 0.]).unwrap();
        let (_, posterior) = prior.conditioned([0.9, 0.34, 0.78, 1.]).unwrap();
        let posterior = posterior.unwrap();
        let restored: FiniteBelief =
            serde_json::from_slice(&serde_json::to_vec(&posterior).unwrap()).unwrap();
        assert_eq!(restored, posterior);
        for malformed in [
            r#"{"version":0,"masses":[0.4,0.6]}"#,
            r#"{"version":1,"masses":[]}"#,
            r#"{"version":1,"masses":[0.4,0.7]}"#,
            r#"{"version":1,"masses":[-0.01,1.01]}"#,
            r#"{"version":1,"masses":[1.0],"foreign":42}"#,
        ] {
            assert!(
                serde_json::from_str::<FiniteBelief>(malformed).is_err(),
                "{malformed}"
            );
        }
    }

    #[test]
    fn beta_quadrature_uses_midpoints_and_normalized_prior() {
        let rule = beta_quadrature(1., 1., 4).unwrap();
        assert_eq!(rule.nodes(), &[0.125, 0.375, 0.625, 0.875]);
        assert_eq!(rule.masses(), &[0.25; 4]);
    }

    #[test]
    fn labeled_updates_approximate_the_analytic_beta_predictive_law() {
        let rule = beta_quadrature(2., 3., 512).unwrap();
        let mut belief = FiniteBelief::new(rule.masses().to_vec()).unwrap();
        for (positive, expected) in [
            (true, 2. / 5.),
            (true, 3. / 6.),
            (false, 3. / 7.),
            (true, 4. / 8.),
            (false, 4. / 9.),
        ] {
            let (predictive, posterior) = belief
                .conditioned(
                    rule.nodes()
                        .iter()
                        .map(|&rate| if positive { rate } else { 1. - rate }),
                )
                .unwrap();
            close(predictive, expected, 5e-6);
            belief = posterior.unwrap();
        }
        let (predictive, _) = belief.conditioned(rule.nodes().iter().copied()).unwrap();
        close(predictive, 5. / 10., 5e-6);
    }

    #[test]
    fn beta_quadrature_rejects_invalid_or_unrepresentable_rules() {
        for (alpha, beta, bins) in [
            (0., 2., 8),
            (2., f64::INFINITY, 4),
            (2., 2., 0),
            (2., 2., usize::MAX),
            (1000., 1., 8),
        ] {
            assert!(beta_quadrature(alpha, beta, bins).is_err());
        }
    }
}
