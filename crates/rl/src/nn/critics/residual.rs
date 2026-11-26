//! Action-value functions in explicit frozen-policy residual coordinates.
use burn::{
    module::{Module, ModuleDisplay, Param},
    tensor::{Device, Tensor},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::scalar::{ActionValue, QValue};
use crate::{
    Result,
    core::{action::ActionSpace, ensure, env::ObservationBatch},
    nn::{
        checkpoint::hash_weights, network::Network, policy::Deterministic, target::frozen_copy,
        to_vec,
    },
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResidualScale {
    Scalar(f32),
    Diagonal(Vec<f32>),
}

/// Presents ordinary environment actions to callers and evaluates its inner
/// critic on `(action - anchor(observation)) / scale`.
///
/// The anchor is a frozen deterministic model. Its action and the scale are
/// detached even when the factual action comes from a differentiable actor.
/// A refresh changes the input coordinates and can change predicted Q without
/// any change to the inner critic's learned weights.
#[derive(Module, Debug)]
pub struct ResidualQ<Q: ActionValue + ModuleDisplay = QValue> {
    inner: Q,
    anchor: Deterministic,
    scale: Param<Tensor<1>>,
}

impl<Q: ActionValue + ModuleDisplay> ResidualQ<Q> {
    pub fn new(inner: Q, anchor: &Deterministic, scale: ResidualScale) -> Result<Self> {
        let width = inner.action_dim();
        ensure(
            width > 0
                && matches!(
                    &anchor.space,
                    ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
                )
                && anchor.space.dimensions() == width
                && anchor.network.output_dim() == width,
            "residual critic and anchor action widths differ",
        )?;
        anchor.space.validate()?;
        let scales = match scale {
            ResidualScale::Scalar(value) => vec![value; width],
            ResidualScale::Diagonal(values) => values,
        };
        Self::validate_scales(&scales, width)?;
        Ok(Self {
            inner,
            anchor: frozen_copy(anchor).valid(),
            scale: Param::from_tensor(Tensor::from_data(scales.as_slice(), &Device::flex()))
                .set_require_grad(false),
        })
    }

    fn validate_scales(scales: &[f32], width: usize) -> Result<()> {
        ensure(
            scales.len() == width && scales.iter().all(|scale| scale.is_finite() && *scale > 0.),
            "residual critic requires one positive finite scale per action dimension",
        )
    }

    pub fn inner(&self) -> &Q {
        &self.inner
    }

    pub fn anchor(&self) -> &Deterministic {
        &self.anchor
    }

    pub fn scales(&self) -> Result<Vec<f32>> {
        let scales = to_vec(self.scale.val())?;
        Self::validate_scales(&scales, self.action_dim())?;
        Ok(scales)
    }

    /// Stable fingerprint of the frozen actor parameters and action convention.
    pub fn anchor_identity(&self) -> Result<[u8; 32]> {
        let mut digest = Sha256::new();
        digest.update(b"rust-rl/deterministic-residual-anchor/1");
        digest.update(serde_json::to_vec(&self.anchor.space)?);
        hash_weights(&mut digest, &self.anchor)?;
        Ok(digest.finalize().into())
    }

    /// Change the frozen coordinates at a caller-defined learning boundary.
    /// A learner with target networks should refresh both through its paired API.
    pub fn refresh_anchor(&mut self, anchor: &Deterministic) -> Result<()> {
        self.anchor = self.frozen_replacement(anchor)?;
        Ok(())
    }

    pub(crate) fn frozen_replacement(&self, anchor: &Deterministic) -> Result<Deterministic> {
        ensure(
            anchor.space == self.anchor.space
                && anchor.network.input_dim() == self.anchor.network.input_dim()
                && anchor.network.output_dim() == self.action_dim()
                && anchor.network.layers.len() == self.anchor.network.layers.len()
                && anchor
                    .network
                    .layers
                    .iter()
                    .zip(&self.anchor.network.layers)
                    .all(|(a, b)| {
                        a.weight.val().dims() == b.weight.val().dims()
                            && a.bias.is_some() == b.bias.is_some()
                    }),
            "replacement anchor has incompatible architecture or action coordinates",
        )?;
        Ok(frozen_copy(anchor).valid())
    }

    pub(crate) fn install_anchor(&mut self, anchor: Deterministic) {
        self.anchor = anchor;
    }

    /// Transform actual environment actions while retaining their action gradient.
    pub fn residual_actions(
        &self,
        obs: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        ensure(
            actions.dims() == [obs.len(), self.action_dim()],
            "residual critic action shape differs from its inputs",
        )?;
        let device = actions.device();
        let anchor = self
            .anchor
            .valid()
            .actions_on_device(obs, &Device::flex())?;
        let scale = self
            .scale
            .val()
            .without_autodiff()
            .to_device(&device)
            .unsqueeze_dim::<2>(0);
        Ok((actions - anchor.without_autodiff().to_device(&device)) / scale)
    }
}

impl<Q: ActionValue + ModuleDisplay> ActionValue for ResidualQ<Q> {
    fn action_dim(&self) -> usize {
        self.inner.action_dim()
    }

    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }

    fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        self.inner.values(obs, self.residual_actions(obs, actions)?)
    }

    fn actor_second_forward_skippable(&self) -> bool {
        self.inner.actor_second_forward_skippable()
    }

    fn updated_target(&self, online: &Self, tau: f32) -> Result<Self> {
        // Coordinates change only at the explicit paired refresh boundary.
        // Blending identical anchors/scales can still change their f32 bits.
        let mut target = self.clone();
        target.inner = self.inner.updated_target(&online.inner, tau)?;
        Ok(target)
    }

    fn checkpoint_coordinates(&self) -> Result<Option<serde_json::Value>> {
        Ok(Some(serde_json::json!({
            "transformation": "frozen-policy-residual/1",
            "anchor": self.anchor_identity()?,
            "scale": self.scales()?,
        })))
    }
}
