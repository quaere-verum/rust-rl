use crate::{
    Result,
    core::ensure,
    nn::{critics::quantile::QuantileGrid, matrix},
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
/// Custom actor utilities must return one scalar per batch row, retaining action gradients.
pub trait RiskObjective: Clone + std::fmt::Debug {
    fn quantiles(&self, q: Tensor<2>, grid: &QuantileGrid) -> Result<Tensor<1>>;
    fn moments(&self, mean: Tensor<1>, second: Tensor<1>) -> Result<Tensor<1>>;
    /// Opt in only when actor utility needs no second moment or its side effects.
    fn actor_utility_from_mean(&self, _mean: &Tensor<1>) -> Option<Result<Tensor<1>>> {
        None
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Risk {
    #[default]
    Expected,
    Cvar {
        alpha: f32,
    },
    MeanVariance {
        penalty: f32,
    },
}
impl RiskObjective for Risk {
    fn actor_utility_from_mean(&self, mean: &Tensor<1>) -> Option<Result<Tensor<1>>> {
        matches!(self, Self::Expected).then(|| Ok(mean.clone()))
    }
    fn quantiles(&self, q: Tensor<2>, grid: &QuantileGrid) -> Result<Tensor<1>> {
        grid.validate()?;
        let [b, n] = q.dims();
        ensure(n == grid.len(), "risk grid width mismatch")?;
        let masses = match *self {
            Self::Cvar { alpha } => {
                ensure(
                    alpha.is_finite() && alpha > 0. && alpha <= 1.,
                    "CVaR alpha must be in (0,1]",
                )?;
                grid.edges
                    .windows(2)
                    .map(|e| (e[1].min(alpha) - e[0].min(alpha)) / alpha)
                    .collect()
            }
            _ => grid.masses(),
        };
        let w = matrix(masses, 1, n).repeat_dim(0, b);
        let mean = (q.clone() * w.clone()).sum_dim(1).squeeze_dim(1);
        if let Self::MeanVariance { .. } = self {
            self.moments(mean, (q.powf_scalar(2.) * w).sum_dim(1).squeeze_dim(1))
        } else {
            Ok(mean)
        }
    }
    fn moments(&self, mean: Tensor<1>, second: Tensor<1>) -> Result<Tensor<1>> {
        match *self {
            Self::Expected => Ok(mean),
            Self::MeanVariance { penalty } => {
                ensure(
                    penalty.is_finite() && penalty >= 0.,
                    "variance penalty must be nonnegative",
                )?;
                Ok(mean.clone() - (second - mean.powf_scalar(2.)).clamp_min(0.) * penalty)
            }
            Self::Cvar { .. } => Err(crate::Error::Config(
                "CVaR requires quantiles, not two moments".into(),
            )),
        }
    }
}
