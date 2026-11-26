use crate::{
    Result,
    core::ensure,
    nn::{critics::quantile::QuantileGrid, matrix},
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantileLoss {
    Absolute,
    Huber { kappa: f32 },
    Generalized { scale: f32, ema: f32, minimum: f32 },
}
impl Default for QuantileLoss {
    fn default() -> Self {
        Self::Huber { kappa: 1. }
    }
}
impl QuantileLoss {
    /// Whether this loss updates a detached residual scale before each critic step.
    pub fn requires_adaptation(&self) -> bool {
        matches!(self, Self::Generalized { .. })
    }

    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Absolute => Ok(()),
            Self::Huber { kappa } => ensure(
                kappa.is_finite() && kappa > 0.,
                "Huber kappa must be positive",
            ),
            Self::Generalized {
                scale,
                ema,
                minimum,
            } => ensure(
                scale.is_finite()
                    && minimum.is_finite()
                    && scale >= minimum
                    && minimum > 0.
                    && ema.is_finite()
                    && (0.0..=1.0).contains(&ema),
                "invalid generalized-loss scale state",
            ),
        }
    }
    /// Detached 70th percentile residual scale, updated once per critic update.
    pub fn adapt(&mut self, mut residuals: Vec<f32>) -> Result<()> {
        self.adapt_in_place(&mut residuals)
    }

    /// Uses caller-owned scratch; its order and signs may change during selection.
    pub(crate) fn adapt_in_place(&mut self, residuals: &mut [f32]) -> Result<()> {
        self.validate()?;
        ensure(
            !residuals.is_empty() && residuals.iter().all(|x| x.is_finite()),
            "invalid quantile residuals",
        )?;
        if let Self::Generalized {
            scale,
            ema,
            minimum,
        } = self
        {
            for x in residuals.iter_mut() {
                *x = x.abs();
            }
            let position = (residuals.len() - 1) as f64 * 0.7;
            let lower = position.floor() as usize;
            let upper = position.ceil() as usize;
            let (_, lower_value, upper_partition) =
                residuals.select_nth_unstable_by(lower, f32::total_cmp);
            let upper_value = if upper == lower {
                *lower_value
            } else {
                upper_partition
                    .iter()
                    .copied()
                    .min_by(f32::total_cmp)
                    .expect("interpolated percentile has an upper partition")
            };
            let q = *lower_value + (upper_value - *lower_value) * (position - lower as f64) as f32;
            *scale = (*ema * *scale + (1. - *ema) * q).max(*minimum);
        }
        Ok(())
    }
}
/// Mean over batch, probability-mass weighted over predicted and target quantiles.
/// Targets are detached. Huber is divided by kappa; exact generalized loss tends to |u| as scale -> 0.
pub fn quantile_loss(
    pred: Tensor<2>,
    target: Tensor<2>,
    grid: &QuantileGrid,
    kind: &QuantileLoss,
) -> Result<Tensor<1>> {
    grid.validate()?;
    kind.validate()?;
    let [b, n] = pred.dims();
    ensure(
        target.dims() == [b, n] && n == grid.len(),
        "quantile shapes must match grid",
    )?;
    let u = target.without_autodiff().reshape([b, 1, n]) - pred.reshape([b, n, 1]);
    let abs = u.clone().abs();
    let base = match *kind {
        QuantileLoss::Absolute => abs.clone(),
        QuantileLoss::Huber { kappa } => {
            let quadratic = abs.clone().clamp(0., kappa);
            (quadratic.clone().powf_scalar(2.) * 0.5 + (abs.clone() - quadratic) * kappa) / kappa
        }
        QuantileLoss::Generalized { scale, .. } => {
            abs.clone() * (abs.clone() / (scale * 2.0_f32.sqrt())).erf()
                + ((u.clone().powf_scalar(2.) / (-2. * scale * scale)).exp() - 1.)
                    * (scale * (2. / std::f32::consts::PI).sqrt())
        }
    };
    let tau = matrix(grid.locations(), n, 1).reshape([1, n, 1]);
    let weight = (tau - u.lower_elem(0.).float()).abs();
    let masses = grid.masses();
    let product: Vec<_> = masses
        .iter()
        .flat_map(|a| masses.iter().map(move |b| a * b))
        .collect();
    let mass = matrix(product, n, n).reshape([1, n, n]);
    Ok((base * weight * mass)
        .sum_dim(2)
        .sum_dim(1)
        .reshape([b])
        .mean())
}
