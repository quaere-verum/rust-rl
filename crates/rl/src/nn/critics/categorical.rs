//! Action-conditioned logits and normalized features for REPPO's HL-Gauss regression.
use super::scalar::{ActionValue, QValue};
use crate::{
    Result,
    core::{action::ActionSpace, ensure, env::ObservationBatch},
    nn::{
        device,
        network::{Mlp, Network},
        observations_on_device,
    },
};
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear},
    tensor::{Device, Tensor, activation::silu},
};
use rand_chacha::ChaCha8Rng;

/// ActionValue outputs are categorical regression logits, not quantiles.
/// Custom critics may opt in without auxiliary features when their weight is zero.
pub trait ReppoCritic: ActionValue {
    fn supports_auxiliary(&self) -> bool {
        false
    }
    fn representation(&self, _: ObservationBatch<'_>, _: Tensor<2>) -> Result<Tensor<2>> {
        Err(crate::Error::State(
            "critic has no REPPO auxiliary representation".into(),
        ))
    }
    fn predict_representation(&self, _: ObservationBatch<'_>, _: Tensor<2>) -> Result<Tensor<2>> {
        Err(crate::Error::State(
            "critic has no REPPO auxiliary predictor".into(),
        ))
    }
}
impl<M: Network> ReppoCritic for QValue<M> {}

/// REPPO-specific linear/LayerNorm/SiLU blocks; the general MLP is unchanged.
#[derive(Module, Debug)]
struct NormalizedBlocks {
    layers: Vec<Linear>,
    norms: Vec<LayerNorm>,
}
impl NormalizedBlocks {
    fn new(input: usize, widths: &[usize], rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        ensure(
            !widths.is_empty() && widths.iter().all(|&w| w >= 2),
            "normalized REPPO feature widths must be at least two",
        )?;
        let output = widths[widths.len() - 1];
        let network = Mlp::new_on_device(input, &widths[..widths.len() - 1], output, rng, device)?;
        // Constant ones/zeros only; LayerNorm does not draw global random numbers.
        let norms = widths
            .iter()
            .map(|&w| LayerNormConfig::new(w).init(device))
            .collect();
        Ok(Self {
            layers: network.layers,
            norms,
        })
    }
    fn forward(&self, mut input: Tensor<2>) -> Tensor<2> {
        for (linear, norm) in self.layers.iter().zip(&self.norms) {
            input = silu(norm.forward(linear.forward(input)));
        }
        input
    }
}

/// Recommended HL-Gauss logit critic. Discrete actions are one-hot; continuous
/// action tensors keep their autodiff graph through the normalized encoder.
#[derive(Module, Debug)]
pub struct CategoricalQ {
    encoder: NormalizedBlocks,
    head: Mlp,
    predictor: NormalizedBlocks,
    observation_dim: usize,
    action_dim: usize,
}
impl CategoricalQ {
    pub fn new(
        observation_dim: usize,
        space: ActionSpace,
        bins: usize,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::new_on_device(observation_dim, space, bins, hidden, rng, &device())
    }

    /// Construct the historical model template on a plain inference backend.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn new_inference(
        observation_dim: usize,
        space: ActionSpace,
        bins: usize,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::new_on_device(observation_dim, space, bins, hidden, rng, &Device::flex())
    }

    fn new_on_device(
        observation_dim: usize,
        space: ActionSpace,
        bins: usize,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        space.validate()?;
        ensure(
            bins >= 2 && observation_dim > 0,
            "invalid REPPO critic observation or bin dimensions",
        )?;
        let action_dim = match space {
            ActionSpace::Discrete { count } => count,
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. } => space.dimensions(),
            ActionSpace::Conditional(_) => {
                return Err(crate::Error::Config(
                    "REPPO critic requires a fixed action space".into(),
                ));
            }
        };
        let input = observation_dim
            .checked_add(action_dim)
            .ok_or_else(|| crate::Error::Shape("REPPO critic width overflow".into()))?;
        let encoder = NormalizedBlocks::new(input, hidden, rng, device)?;
        let feature_dim = hidden[hidden.len() - 1];
        Ok(Self {
            encoder,
            head: Mlp::new_on_device(feature_dim, &[], bins, rng, device)?,
            predictor: NormalizedBlocks::new(feature_dim, &[feature_dim], rng, device)?,
            observation_dim,
            action_dim,
        })
    }
    fn features(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        ensure(
            actions.dims() == [obs.len(), self.action_dim],
            "REPPO critic action shape mismatch",
        )?;
        let observations = observations_on_device(obs, self.observation_dim, &actions.device())?;
        Ok(self
            .encoder
            .forward(Tensor::cat(vec![observations, actions], 1)))
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_layout(&self) -> (usize, Vec<usize>) {
        (
            self.observation_dim,
            self.encoder
                .layers
                .iter()
                .map(|layer| layer.weight.val().dims()[1])
                .collect(),
        )
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_device(&self) -> Device {
        self.head.layers[0].weight.val().device()
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_predictor(&self, features: Tensor<2>) -> Tensor<2> {
        self.predictor.forward(features)
    }
}
impl ActionValue for CategoricalQ {
    fn action_dim(&self) -> usize {
        self.action_dim
    }
    fn output_dim(&self) -> usize {
        self.head.output_dim()
    }
    fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        Ok(self.head.forward(self.features(obs, actions)?))
    }
}
impl ReppoCritic for CategoricalQ {
    fn supports_auxiliary(&self) -> bool {
        true
    }
    fn representation(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        self.features(obs, actions)
    }
    fn predict_representation(
        &self,
        obs: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        Ok(self.predictor.forward(self.features(obs, actions)?))
    }
}
