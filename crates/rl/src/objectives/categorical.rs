//! Fixed-support categorical distributions and their Bellman projection.
use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};

/// Strictly increasing finite atoms in the critic's numeric precision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Vec<f32>", into = "Vec<f32>")]
pub struct CategoricalSupport(Vec<f32>);

/// The projected probability mass and the mass clipped to each endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoricalProjection {
    pub probabilities: Vec<f64>,
    pub clipped_below: f64,
    pub clipped_above: f64,
}

impl TryFrom<Vec<f32>> for CategoricalSupport {
    type Error = crate::Error;

    fn try_from(atoms: Vec<f32>) -> Result<Self> {
        Self::new(atoms)
    }
}

impl From<CategoricalSupport> for Vec<f32> {
    fn from(support: CategoricalSupport) -> Self {
        support.0
    }
}

impl CategoricalSupport {
    pub fn new(atoms: Vec<f32>) -> Result<Self> {
        ensure(
            atoms.len() >= 2
                && atoms.iter().all(|atom| atom.is_finite())
                && atoms.windows(2).all(|pair| pair[0] < pair[1]),
            "categorical support requires at least two strictly increasing finite atoms",
        )?;
        Ok(Self(atoms))
    }

    pub fn uniform(min: f32, max: f32, count: usize) -> Result<Self> {
        ensure(
            min.is_finite() && max.is_finite() && min < max && count >= 2,
            "invalid uniform categorical support",
        )?;
        let mut atoms = Vec::new();
        atoms.try_reserve_exact(count).map_err(|error| {
            crate::Error::Config(format!("categorical support allocation failed: {error}"))
        })?;
        let min64 = f64::from(min);
        let span = f64::from(max) - min64;
        atoms.push(min);
        atoms.extend(
            (1..count - 1).map(|i| (min64 + span * (i as f64 / (count - 1) as f64)) as f32),
        );
        atoms.push(max);
        Self::new(atoms)
    }

    pub fn atoms(&self) -> &[f32] {
        &self.0
    }

    /// Project `reward + discount * Z` onto this support by linear interpolation.
    /// The supplied continuation uses this support. At zero discount all mass
    /// comes from the reward, so the continuation can be an empty slice.
    pub fn project(
        &self,
        reward: f64,
        discount: f64,
        future_probabilities: &[f64],
    ) -> Result<CategoricalProjection> {
        ensure(
            reward.is_finite() && discount.is_finite() && (0. ..=1.).contains(&discount),
            "invalid categorical Bellman reward or discount",
        )?;
        let mut projection = CategoricalProjection {
            probabilities: vec![0.; self.0.len()],
            clipped_below: 0.,
            clipped_above: 0.,
        };
        if discount == 0. {
            self.project_atom(reward, 1., &mut projection);
            return Ok(projection);
        }
        let total: f64 = future_probabilities.iter().sum();
        ensure(
            future_probabilities.len() == self.0.len()
                && future_probabilities
                    .iter()
                    .all(|p| p.is_finite() && *p >= 0.)
                && (total - 1.).abs() <= 1e-6,
            "invalid categorical continuation probabilities",
        )?;
        for (&atom, &mass) in self.0.iter().zip(future_probabilities) {
            // Normalize floating-point summation drift without moving mass between atoms.
            self.project_atom(
                reward + discount * f64::from(atom),
                mass / total,
                &mut projection,
            );
        }
        Ok(projection)
    }

    fn project_atom(&self, atom: f64, mass: f64, result: &mut CategoricalProjection) {
        if mass == 0. {
            return;
        }
        let first = f64::from(self.0[0]);
        let last = self.0.len() - 1;
        let final_atom = f64::from(self.0[last]);
        if atom <= first {
            result.probabilities[0] += mass;
            if atom < first {
                result.clipped_below += mass;
            }
        } else if atom >= final_atom {
            result.probabilities[last] += mass;
            if atom > final_atom {
                result.clipped_above += mass;
            }
        } else {
            let hi = self.0.partition_point(|&z| f64::from(z) < atom);
            let upper = f64::from(self.0[hi]);
            let lower = f64::from(self.0[hi - 1]);
            let weight = (atom - lower) / (upper - lower);
            result.probabilities[hi] += mass * weight;
            result.probabilities[hi - 1] += mass * (1. - weight);
        }
    }
}

#[cfg(feature = "neural")]
mod tensor_objectives {
    use super::*;
    use crate::{nn::matrix_on_device, objectives::risk::Risk};
    use burn::tensor::{
        Tensor,
        activation::{log_softmax, softmax},
    };

    pub(crate) fn validate_risk(risk: &Risk) -> Result<()> {
        match *risk {
            Risk::Expected => Ok(()),
            Risk::Cvar { alpha } => ensure(
                alpha.is_finite() && alpha > 0. && alpha <= 1.,
                "CVaR alpha must be in (0,1]",
            ),
            Risk::MeanVariance { penalty } => ensure(
                penalty.is_finite() && penalty >= 0.,
                "variance penalty must be nonnegative",
            ),
        }
    }

    impl CategoricalSupport {
        fn check_logits(&self, logits: &Tensor<2>) -> Result<()> {
            let [rows, width] = logits.dims();
            ensure(
                rows > 0 && width == self.atoms().len(),
                "categorical logit shape mismatch",
            )
        }

        /// Stable softmax across the declared atoms, retaining input gradients.
        pub fn probabilities(&self, logits: Tensor<2>) -> Result<Tensor<2>> {
            self.check_logits(&logits)?;
            Ok(softmax(logits, 1))
        }

        /// Stable mean cross-entropy against normalized, detached projected targets.
        pub(crate) fn cross_entropy(
            &self,
            logits: Tensor<2>,
            targets: Tensor<2>,
        ) -> Result<Tensor<1>> {
            self.check_logits(&logits)?;
            ensure(
                logits.dims() == targets.dims(),
                "categorical target shape mismatch",
            )?;
            let targets = targets.without_autodiff().to_device(&logits.device());
            // A zero-weight atom contributes zero even when its log probability
            // underflows to negative infinity at a representable finite logit.
            let log_probabilities =
                log_softmax(logits, 1).mask_fill(targets.clone().equal_elem(0.), 0.);
            Ok((log_probabilities * targets).sum_dim(1).mean().neg())
        }

        /// One differentiable actor utility per row of categorical logits.
        pub fn actor_utility(&self, logits: Tensor<2>, risk: &Risk) -> Result<Tensor<1>> {
            validate_risk(risk)?;
            let probabilities = self.probabilities(logits)?;
            let [rows, width] = probabilities.dims();
            let device = probabilities.device();
            let atoms = matrix_on_device(self.atoms().to_vec(), 1, width, &device);
            let center =
                ((f64::from(self.atoms()[0]) + f64::from(self.atoms()[width - 1])) * 0.5) as f32;
            let atoms = atoms - center;
            if let Risk::Cvar { alpha } = *risk
                && alpha < 1.
            {
                // Tail weights are increments of the CDF truncated at alpha.
                // Make the last normalized CDF value exactly one despite rounding.
                let cumulative = probabilities.cumsum(1).slice([0..rows, 0..width - 1]);
                let tail_cdf = cumulative.clamp_max(alpha) / alpha;
                let left =
                    Tensor::cat(vec![Tensor::zeros([rows, 1], &device), tail_cdf.clone()], 1);
                let right = Tensor::cat(vec![tail_cdf, Tensor::ones([rows, 1], &device)], 1);
                return Ok(((right - left) * atoms).sum_dim(1).squeeze_dim(1) + center);
            }
            let mean = (probabilities.clone() * atoms.clone()).sum_dim(1);
            let utility = if let Risk::MeanVariance { penalty } = *risk
                && penalty > 0.
            {
                let variance = (probabilities * (atoms - mean.clone()).powf_scalar(2.)).sum_dim(1);
                mean - variance * penalty
            } else {
                mean
            };
            Ok(utility.squeeze_dim(1) + center)
        }
    }
}

#[cfg(feature = "neural")]
pub(crate) use tensor_objectives::validate_risk;

#[cfg(all(test, feature = "neural"))]
mod tests {
    use super::*;
    use crate::nn::{matrix, to_vec};

    #[test]
    fn cross_entropy_ignores_zero_target_mass_and_detaches_targets() {
        let support = CategoricalSupport::uniform(-1., 1., 2).unwrap();
        let logits = matrix(vec![f32::MAX, -f32::MAX], 1, 2).require_grad();
        let targets = matrix(vec![1., 0.], 1, 2).require_grad();
        let loss = support
            .cross_entropy(logits.clone(), targets.clone())
            .unwrap();
        assert_eq!(loss.clone().into_scalar::<f32>(), 0.);
        let grads = loss.backward();
        assert_eq!(to_vec(logits.grad(&grads).unwrap()).unwrap(), [0., 0.]);
        assert!(targets.grad(&grads).is_none());
    }
}
