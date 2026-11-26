//! Exact online Bayesian linear regression with known observation variance.

use crate::{
    Error, Result,
    core::{checkpoint::Snapshot, ensure},
};
use serde::{Deserialize, Serialize};

const VERSION: u32 = 1;
const MAX_FEATURES: usize = 64;

/// A scalar Gaussian prediction conditional on the observed data and input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianPrediction {
    pub mean: f64,
    pub epistemic_variance: f64,
    pub observation_variance: f64,
    /// Expected information about the model parameters, in nats.
    pub information_gain: f64,
}

impl GaussianPrediction {
    pub fn total_variance(self) -> f64 {
        self.epistemic_variance + self.observation_variance
    }
}

/// Exact linear Gaussian inference with a zero-mean isotropic parameter prior.
/// Each new independent outcome contributes one likelihood with known variance.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianPosterior {
    version: u32,
    dimensions: usize,
    prior_precision: f64,
    observations: u64,
    precision_cholesky: Vec<f64>,
    natural_mean: Vec<f64>,
}

impl GaussianPosterior {
    pub fn new(dimensions: usize, prior_precision: f64) -> Result<Self> {
        ensure(
            (1..=MAX_FEATURES).contains(&dimensions) && positive(prior_precision),
            "invalid Gaussian predictor dimension or prior precision",
        )?;
        let mut precision_cholesky = vec![0.; dimensions * dimensions];
        for i in 0..dimensions {
            precision_cholesky[i * dimensions + i] = prior_precision.sqrt();
        }
        Ok(Self {
            version: VERSION,
            dimensions,
            prior_precision,
            observations: 0,
            precision_cholesky,
            natural_mean: vec![0.; dimensions],
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    pub fn prior_precision(&self) -> f64 {
        self.prior_precision
    }
    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn predict(
        &self,
        features: &[f64],
        observation_variance: f64,
    ) -> Result<GaussianPrediction> {
        self.validate_snapshot()?;
        self.validate_features(features)?;
        let projected = self.solve_lower(features)?;
        let natural_mean = self.solve_lower(&self.natural_mean)?;
        let mean = dot(&projected, &natural_mean);
        let epistemic_variance = dot(&projected, &projected);
        let information_gain = gain(epistemic_variance, observation_variance)?;
        let prediction = GaussianPrediction {
            mean,
            epistemic_variance,
            observation_variance,
            information_gain,
        };
        ensure(
            prediction.mean.is_finite() && positive(prediction.total_variance()),
            "nonfinite Gaussian prediction",
        )?;
        Ok(prediction)
    }

    pub fn expected_information(&self, features: &[f64], observation_variance: f64) -> Result<f64> {
        self.validate_snapshot()?;
        self.validate_features(features)?;
        let projected = self.solve_lower(features)?;
        gain(dot(&projected, &projected), observation_variance)
    }

    pub fn epistemic_covariance(&self, first: &[f64], second: &[f64]) -> Result<f64> {
        self.validate_snapshot()?;
        self.validate_features(first)?;
        self.validate_features(second)?;
        let covariance = dot(&self.solve_lower(first)?, &self.solve_lower(second)?);
        ensure(covariance.is_finite(), "nonfinite Gaussian covariance")?;
        Ok(covariance)
    }

    pub fn observe(
        &mut self,
        features: &[f64],
        outcome: f64,
        observation_variance: f64,
    ) -> Result<()> {
        self.validate_snapshot()?;
        self.validate_features(features)?;
        ensure(
            outcome.is_finite() && positive(observation_variance),
            "invalid Gaussian likelihood observation or variance",
        )?;
        let sigma = observation_variance.sqrt();
        let mut row: Vec<_> = features.iter().map(|x| x / sigma).collect();
        let response = outcome / sigma;
        ensure(
            response.is_finite() && row.iter().all(|x| x.is_finite()),
            "Gaussian likelihood scaling overflow",
        )?;
        let mut updated = self.clone();
        updated.observations = self
            .observations
            .checked_add(1)
            .ok_or_else(|| Error::State("Gaussian observation clock overflow".into()))?;
        for (weight, &feature) in updated.natural_mean.iter_mut().zip(&row) {
            *weight += feature * response;
        }
        updated.rank_one(&mut row);
        updated.validate_snapshot()?;
        *self = updated;
        Ok(())
    }

    fn validate_features(&self, features: &[f64]) -> Result<()> {
        ensure(
            features.len() == self.dimensions && features.iter().all(|x| x.is_finite()),
            "invalid Gaussian predictor features",
        )
    }

    fn solve_lower(&self, rhs: &[f64]) -> Result<Vec<f64>> {
        let mut solution = Vec::with_capacity(self.dimensions);
        for (row, (&value, factor)) in rhs
            .iter()
            .zip(self.precision_cholesky.chunks_exact(self.dimensions))
            .enumerate()
        {
            let residual = value - dot(&factor[..row], &solution);
            solution.push(residual / factor[row]);
        }
        ensure(
            solution.iter().all(|value| value.is_finite()),
            "Gaussian precision solve overflow",
        )?;
        Ok(solution)
    }

    fn rank_one(&mut self, row: &mut [f64]) {
        let n = self.dimensions;
        for column in 0..n {
            let diagonal = &mut self.precision_cholesky[column * n + column];
            let updated = diagonal.hypot(row[column]);
            // Orthogonal Givens rotation of the existing factor column and
            // new observation row. Both coefficients stay bounded by one.
            let cosine = *diagonal / updated;
            let sine = row[column] / updated;
            *diagonal = updated;
            for (i, x) in row.iter_mut().enumerate().skip(column + 1) {
                let cell = &mut self.precision_cholesky[i * n + column];
                let old = *cell;
                *cell = cosine * old + sine * *x;
                *x = cosine * *x - sine * old;
            }
        }
    }
}

impl Snapshot for GaussianPosterior {
    fn validate_snapshot(&self) -> Result<()> {
        let n = self.dimensions;
        ensure(
            self.version == VERSION
                && (1..=MAX_FEATURES).contains(&n)
                && positive(self.prior_precision)
                && self.precision_cholesky.len() == n * n
                && self.natural_mean.len() == n
                && self.natural_mean.iter().all(|v| v.is_finite()),
            "invalid Gaussian posterior version, shape or natural mean",
        )?;
        ensure(
            self.precision_cholesky
                .chunks_exact(n)
                .enumerate()
                .all(|(row, factor)| {
                    positive(factor[row])
                        && factor[..row].iter().all(|v| v.is_finite())
                        && factor[row + 1..].iter().all(|&v| v == 0.)
                }),
            "invalid Gaussian precision factor",
        )?;
        if self.observations == 0 {
            let diagonal = self.prior_precision.sqrt();
            ensure(
                self.natural_mean.iter().all(|&v| v == 0.)
                    && self
                        .precision_cholesky
                        .chunks_exact(n)
                        .enumerate()
                        .all(|(row, factor)| {
                            factor[row].to_bits() == diagonal.to_bits()
                                && factor[..row].iter().all(|&v| v == 0.)
                        }),
                "unobserved Gaussian posterior differs from its prior",
            )?;
        }
        Ok(())
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.
}

fn gain(epistemic: f64, observation: f64) -> Result<f64> {
    ensure(
        epistemic.is_finite() && epistemic >= 0. && positive(observation),
        "invalid Gaussian information variances",
    )?;
    let ratio = epistemic / observation;
    let log = if ratio.is_finite() {
        ratio.ln_1p()
    } else {
        epistemic.ln() - observation.ln()
    };
    Ok(0.5 * log)
}
