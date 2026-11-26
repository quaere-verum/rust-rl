//! State-dependent diagonal Gaussian head using the existing network/distribution API.
use super::{
    distributions::Distribution,
    network::{Mlp, Network},
    observations,
    policy::StochasticPolicy,
};
use crate::{
    Result,
    core::{action::ActionSpace, ensure, env::ObservationBatch},
};
use burn::module::Module;
use rand_chacha::ChaCha8Rng;

#[derive(Module, Debug)]
pub struct GaussianPolicy<M: Network = Mlp> {
    pub network: M,
    #[module(skip)]
    space: ActionSpace,
}
impl GaussianPolicy {
    pub fn new(
        width: usize,
        space: ActionSpace,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        validate_gaussian_space(&space)?;
        let outputs = space
            .dimensions()
            .checked_mul(2)
            .ok_or_else(|| crate::Error::Shape("Gaussian head size overflow".into()))?;
        Self::from_network(Mlp::new(width, hidden, outputs, rng)?, space)
    }
}
impl<M: Network> GaussianPolicy<M> {
    /// Network outputs `[means..., log_standard_deviations...]` per row.
    pub fn from_network(network: M, space: ActionSpace) -> Result<Self> {
        validate_gaussian_space(&space)?;
        ensure(
            space.dimensions().checked_mul(2) == Some(network.output_dim()),
            "Gaussian policy requires two outputs per action dimension",
        )?;
        Ok(Self { network, space })
    }
}
impl<M: Network> StochasticPolicy for GaussianPolicy<M> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn distribution(&self, obs: ObservationBatch<'_>) -> Result<Distribution> {
        ensure(!obs.is_empty(), "empty Gaussian observation batch")?;
        let d = self.space.dimensions();
        ensure(
            d.checked_mul(2) == Some(self.network.output_dim()),
            "Gaussian network output width changed",
        )?;
        for o in obs.iter() {
            o.validate(self.network.input_dim(), &self.space, true)?;
        }
        let output = self
            .network
            .forward(observations(obs, self.network.input_dim())?);
        let mean = output.clone().slice([0..obs.len(), 0..d]);
        let log_std = output.slice([0..obs.len(), d..2 * d]).clamp(-20., 2.);
        let bounds = match &self.space {
            ActionSpace::Box { low, high } => Some((low.clone(), high.clone())),
            _ => None,
        };
        Ok(Distribution::Gaussian {
            mean,
            log_std,
            bounds,
        })
    }
}

/// Bounds must support the distribution's actual f32 affine arithmetic.
pub(crate) fn validate_gaussian_space(space: &ActionSpace) -> Result<()> {
    space.validate()?;
    ensure(
        matches!(
            space,
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
        ),
        "Gaussian policy requires continuous actions",
    )?;
    if let ActionSpace::Box { low, high } = space {
        ensure(
            low.iter().zip(high).all(|(&l, &h)| {
                let scale = (h - l) * 0.5;
                let center = (h + l) * 0.5;
                scale.is_finite() && scale > 0. && center.is_finite()
            }),
            "Gaussian affine bounds cannot be represented in f32",
        )?;
    }
    Ok(())
}
