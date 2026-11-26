//! Gaussian histogram regression for scalar targets; this is not a return distribution.
#[cfg(feature = "neural")]
use crate::nn::matrix_on_device;
use crate::{Result, core::ensure};
#[cfg(feature = "neural")]
use burn::tensor::{
    Tensor,
    activation::{log_softmax, softmax},
};
use serde::{Deserialize, Serialize};

/// Bin centers span `[min,max]`, with outer edges extending half a bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HlGaussConfig {
    pub min: f32,
    pub max: f32,
    pub bins: usize,
    /// Gaussian standard deviation in units of bin width (reference default .75).
    pub sigma_ratio: f32,
}
impl Default for HlGaussConfig {
    fn default() -> Self {
        Self {
            min: -100.,
            max: 100.,
            bins: 101,
            sigma_ratio: 0.75,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HlGaussTargets {
    /// Row-major normalized histograms, each of width `config.bins`.
    pub probabilities: Vec<f32>,
    /// Count before the explicit clipping of scalar targets to `[min,max]`.
    pub out_of_support: usize,
}
impl HlGaussConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.min.is_finite()
                && self.max.is_finite()
                && self.min < self.max
                && self.bins >= 2
                && self.sigma_ratio.is_finite()
                && self.sigma_ratio > 0.,
            "invalid HL-Gauss support, bin count or Gaussian width",
        )
    }
    pub fn centers(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let width = (f64::from(self.max) - f64::from(self.min)) / (self.bins - 1) as f64;
        Ok((0..self.bins)
            .map(|i| (f64::from(self.min) + i as f64 * width) as f32)
            .collect())
    }
    pub fn targets(&self, values: &[f32]) -> Result<HlGaussTargets> {
        self.validate()?;
        ensure(
            !values.is_empty() && values.iter().all(|x| x.is_finite()),
            "invalid HL-Gauss scalar targets",
        )?;
        let count = values
            .len()
            .checked_mul(self.bins)
            .ok_or_else(|| crate::Error::Shape("HL-Gauss histogram size overflow".into()))?;
        let width = (f64::from(self.max) - f64::from(self.min)) / (self.bins - 1) as f64;
        let sigma_root_two = width * f64::from(self.sigma_ratio) * 2_f64.sqrt();
        let edge = f64::from(self.min) - width * 0.5;
        let mut probabilities = Vec::with_capacity(count);
        let mut masses = vec![0.; self.bins];
        let mut out_of_support = 0;
        for &value in values {
            out_of_support += usize::from(value < self.min || value > self.max);
            let mean = f64::from(value.clamp(self.min, self.max));
            for (i, mass) in masses.iter_mut().enumerate() {
                let lower = (edge + i as f64 * width - mean) / sigma_root_two;
                let upper = (edge + (i + 1) as f64 * width - mean) / sigma_root_two;
                // erfc avoids cancellation in the tails; erf avoids it near zero
                // when an extremely broad Gaussian is almost uniform on the support.
                *mass = 0.5
                    * if lower > 1. {
                        libm::erfc(lower) - libm::erfc(upper)
                    } else if upper < -1. {
                        libm::erfc(-upper) - libm::erfc(-lower)
                    } else {
                        libm::erf(upper) - libm::erf(lower)
                    };
            }
            let total: f64 = masses.iter().sum();
            ensure(
                total.is_finite() && total > 0.,
                "unrepresentable HL-Gauss histogram mass",
            )?;
            probabilities.extend(masses.iter().map(|mass| (mass / total) as f32));
        }
        Ok(HlGaussTargets {
            probabilities,
            out_of_support,
        })
    }
    /// Differentiable scalar expectation over bin centers, including action gradients.
    #[cfg(feature = "neural")]
    pub fn decode(&self, logits: Tensor<2>) -> Result<Tensor<1>> {
        self.validate()?;
        let [n, bins] = logits.dims();
        ensure(n > 0 && bins == self.bins, "HL-Gauss logit shape mismatch")?;
        let centers = matrix_on_device(self.centers()?, 1, bins, &logits.device());
        Ok((softmax(logits, 1) * centers).sum_dim(1).squeeze_dim(1))
    }
    /// Stable cross-entropy. Targets are normalized histograms from `targets`.
    #[cfg(feature = "neural")]
    pub fn loss(&self, logits: Tensor<2>, targets: Tensor<2>) -> Result<Tensor<1>> {
        self.validate()?;
        let [n, bins] = logits.dims();
        ensure(
            n > 0 && bins == self.bins && targets.dims() == [n, bins],
            "HL-Gauss loss shape mismatch",
        )?;
        let targets = targets.without_autodiff().to_device(&logits.device());
        Ok((targets * log_softmax(logits, 1)).sum_dim(1).mean().neg())
    }
}
