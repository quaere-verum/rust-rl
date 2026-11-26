//! Independent temporal scalar values; no policy encoder or hidden-state sharing.
use super::{
    network::{Mlp, Network},
    temporal::TemporalEncoder,
    temporal_models::{self, TemporalBackbone, TemporalModelArchitecture},
};
use crate::{
    Result,
    core::{ensure, stateful::HistorySpec},
    data::temporal::SequenceBatch,
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;

pub trait TemporalStateValue: Module {
    fn history_spec(&self) -> &HistorySpec;
    fn values(&self, batch: &SequenceBatch, tokens: &[(usize, usize)]) -> Result<Tensor<1>>;
}
#[derive(Module, Debug)]
pub struct TemporalValue<E: TemporalEncoder = TemporalBackbone, H: Network = Mlp> {
    pub encoder: E,
    pub head: H,
    #[module(skip)]
    spec: HistorySpec,
    #[module(skip)]
    observation_only: bool,
}
impl TemporalValue {
    pub fn new(
        spec: HistorySpec,
        recurrent_hidden: Option<usize>,
        head_hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        spec.validate()?;
        let encoder = TemporalBackbone::for_history(&spec, recurrent_hidden, rng, false)?;
        let head = Mlp::new(encoder.output_dim(), head_hidden, 1, rng)?;
        Self::from_parts(encoder, head, spec, recurrent_hidden.is_none())
    }
    pub fn architecture(&self) -> Result<TemporalModelArchitecture> {
        let architecture = temporal_models::architecture(
            &self.encoder,
            &self.head,
            &self.spec,
            self.observation_only,
        )?;
        ensure(
            architecture.output_dim == 1,
            "temporal state value output must be scalar",
        )?;
        Ok(architecture)
    }
    pub fn from_architecture(
        architecture: &TemporalModelArchitecture,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::from_architecture_with_mode(architecture, rng, false)
    }
    pub(crate) fn from_architecture_with_mode(
        architecture: &TemporalModelArchitecture,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        architecture.validate()?;
        ensure(
            architecture.version == 1
                && architecture.head_input_extra == 0
                && architecture.output_dim == 1,
            "temporal value architecture must be an unconditioned scalar model",
        )?;
        let (encoder, head) = architecture.parts(rng, inference)?;
        Ok(Self {
            encoder,
            head,
            spec: architecture.spec.clone(),
            observation_only: architecture.hidden_dim.is_none(),
        })
    }
}
impl<E: TemporalEncoder, H: Network> TemporalValue<E, H> {
    pub fn from_parts(
        encoder: E,
        head: H,
        spec: HistorySpec,
        observation_only: bool,
    ) -> Result<Self> {
        spec.validate()?;
        ensure(
            encoder.input_dim()
                == if observation_only {
                    spec.observation_dim
                } else {
                    spec.input_dim()
                }
                && encoder.output_dim() > 0
                && head.input_dim() == encoder.output_dim()
                && head.output_dim() == 1,
            "temporal value encoder/head dimensions mismatch",
        )?;
        Ok(Self {
            encoder,
            head,
            spec,
            observation_only,
        })
    }
    pub fn spec(&self) -> &HistorySpec {
        &self.spec
    }
    pub fn observation_only(&self) -> bool {
        self.observation_only
    }
    pub fn device(&self) -> Device {
        temporal_models::module_device(self)
    }
    pub fn values_from_features(&self, features: Tensor<2>) -> Result<Tensor<1>> {
        self.values_from_features_using(features, |x| self.head.forward(x))
    }
    pub(crate) fn values_from_features_using(
        &self,
        features: Tensor<2>,
        forward: impl FnOnce(Tensor<2>) -> Tensor<2>,
    ) -> Result<Tensor<1>> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::ValueEvaluation);

        let [rows, width] = features.dims();
        ensure(
            rows > 0 && width == self.encoder.output_dim(),
            "temporal value feature shape mismatch",
        )?;
        let values = forward(features);
        ensure(
            values.dims() == [rows, 1],
            "temporal value head output must be scalar",
        )?;
        Ok(values.squeeze_dim(1))
    }
}
impl<E: TemporalEncoder, H: Network> TemporalStateValue for TemporalValue<E, H> {
    fn history_spec(&self) -> &HistorySpec {
        &self.spec
    }
    fn values(&self, batch: &SequenceBatch, tokens: &[(usize, usize)]) -> Result<Tensor<1>> {
        let features = temporal_models::selected_features(
            &self.encoder,
            batch,
            &self.spec,
            self.observation_only,
            &self.device(),
            tokens,
        )?;
        self.values_from_features(features)
    }
}
