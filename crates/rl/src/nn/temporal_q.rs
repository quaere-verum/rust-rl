//! Factual temporal history followed by independent candidate-action Q queries.
use super::{
    checkpoint::RecordBurnExt,
    network::{Mlp, Network},
    temporal::TemporalEncoder,
    temporal_models::{self, TemporalBackbone, TemporalModelArchitecture},
};
use crate::{
    Error, Result,
    core::{checkpoint::LearnerRecord, ensure, stateful::HistorySpec},
    data::temporal::SequenceBatch,
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, activation::relu},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Q capability independent of any concrete recurrent representation or head.
pub trait TemporalActionValue: Module {
    fn history_spec(&self) -> &HistorySpec;
    fn action_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn history_features(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>>;
    fn values_from_features(&self, history: Tensor<2>, actions: Tensor<2>) -> Result<Tensor<2>>;
    fn values(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.values_from_features(self.history_features(batch, tokens)?, actions)
    }
}

/// Optional penultimate action-conditioned representation for auxiliary learning.
pub trait TemporalQHead: Network {
    fn representation_dim(&self) -> usize;
    fn representation(&self, input: Tensor<2>) -> Result<Tensor<2>>;
}
impl TemporalQHead for Mlp {
    fn representation_dim(&self) -> usize {
        self.layers
            .last()
            .map_or(self.input_dim(), |last| last.weight.val().dims()[0])
    }
    fn representation(&self, mut input: Tensor<2>) -> Result<Tensor<2>> {
        ensure(
            !self.layers.is_empty() && input.dims()[0] > 0 && input.dims()[1] == self.input_dim(),
            "invalid temporal Q representation input",
        )?;
        for layer in self.layers.iter().take(self.layers.len() - 1) {
            input = relu(layer.forward(input));
        }
        Ok(input)
    }
}

/// Optional REPPO feature capability. Other temporal critics need only the Q trait.
pub trait TemporalReppoCritic: TemporalActionValue {
    fn supports_auxiliary(&self) -> bool {
        false
    }
    fn representation(
        &self,
        _: &SequenceBatch,
        _: &[(usize, usize)],
        _: Tensor<2>,
    ) -> Result<Tensor<2>> {
        Err(Error::State(
            "temporal critic has no auxiliary representation".into(),
        ))
    }
    fn predict_representation(
        &self,
        _: &SequenceBatch,
        _: &[(usize, usize)],
        _: Tensor<2>,
    ) -> Result<Tensor<2>> {
        Err(Error::State(
            "temporal critic has no auxiliary predictor".into(),
        ))
    }
}

#[derive(Module, Debug)]
pub struct TemporalQ<E: TemporalEncoder = TemporalBackbone, H: Network = Mlp, P: Network = Mlp> {
    pub encoder: E,
    pub head: H,
    pub predictor: Option<P>,
    #[module(skip)]
    spec: HistorySpec,
    #[module(skip)]
    observation_only: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalQArchitecture {
    pub version: u32,
    pub model: TemporalModelArchitecture,
    /// None disables auxiliary prediction; Some describes one square affine layer.
    pub predictor_bias: Option<bool>,
}
impl TemporalQArchitecture {
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        ensure(
            self.version == 1
                && self.model.version == 2
                && self.model.head_input_extra == self.model.spec.action_width(),
            "invalid temporal Q conditioned architecture",
        )?;
        self.parameter_count()?;
        Ok(())
    }
    fn representation_dim(&self) -> Result<usize> {
        self.model
            .head_hidden
            .last()
            .copied()
            .map(Ok)
            .unwrap_or_else(|| {
                self.model
                    .hidden_dim
                    .unwrap_or(self.model.spec.observation_dim)
                    .checked_add(self.model.head_input_extra)
                    .ok_or_else(|| Error::Shape("temporal Q representation width overflow".into()))
            })
    }
    pub fn parameter_count(&self) -> Result<usize> {
        let extra = if let Some(bias) = self.predictor_bias {
            let width = self.representation_dim()?;
            width
                .checked_mul(width)
                .and_then(|n| n.checked_add(if bias { width } else { 0 }))
                .ok_or_else(|| Error::Shape("temporal Q predictor size overflow".into()))?
        } else {
            0
        };
        self.model.parameter_count(extra)
    }
}

impl TemporalQ {
    /// One output is scalar Q; multiple outputs are caller-defined regression bins.
    pub fn new(
        spec: HistorySpec,
        recurrent_hidden: Option<usize>,
        head_hidden: &[usize],
        bins: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        spec.validate()?;
        ensure(bins > 0, "temporal Q must have at least one output")?;
        let encoder = TemporalBackbone::for_history(&spec, recurrent_hidden, rng, false)?;
        let conditioned = encoder
            .output_dim()
            .checked_add(spec.action_width())
            .ok_or_else(|| Error::Shape("temporal Q head width overflow".into()))?;
        let head = Mlp::new(conditioned, head_hidden, bins, rng)?;
        let repr = head.representation_dim();
        let predictor = Some(Mlp::new(repr, &[], repr, rng)?);
        Self::from_parts(encoder, head, predictor, spec, recurrent_hidden.is_none())
    }
    pub fn architecture(&self) -> Result<TemporalQArchitecture> {
        let model = temporal_models::architecture_conditioned(
            &self.encoder,
            &self.head,
            &self.spec,
            self.observation_only,
            self.action_dim(),
        )?;
        let predictor_bias = if let Some(predictor) = &self.predictor {
            let width = self.head.representation_dim();
            ensure(
                predictor.layers.len() == 1
                    && predictor.input_dim() == width
                    && predictor.output_dim() == width
                    && predictor.layers[0].weight.val().dims() == [width, width]
                    && predictor.layers[0]
                        .bias
                        .as_ref()
                        .is_none_or(|b| b.val().dims() == [width]),
                "unsupported temporal Q predictor architecture",
            )?;
            Some(predictor.layers[0].bias.is_some())
        } else {
            None
        };
        let architecture = TemporalQArchitecture {
            version: 1,
            model,
            predictor_bias,
        };
        architecture.validate()?;
        Ok(architecture)
    }
    pub fn from_architecture(
        architecture: &TemporalQArchitecture,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::from_architecture_with_mode(architecture, rng, false)
    }
    pub(crate) fn from_architecture_with_mode(
        architecture: &TemporalQArchitecture,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        architecture.validate()?;
        let (encoder, head) = architecture.model.parts(rng, inference)?;
        let predictor = architecture
            .predictor_bias
            .map(|bias| -> Result<Mlp> {
                let width = architecture.representation_dim()?;
                let mut model = if inference {
                    Mlp::new_inference(width, &[], width, rng)?
                } else {
                    Mlp::new(width, &[], width, rng)?
                };
                if !bias {
                    model.layers[0].bias = None;
                }
                Ok(model)
            })
            .transpose()?;
        Self::from_parts(
            encoder,
            head,
            predictor,
            architecture.model.spec.clone(),
            architecture.model.hidden_dim.is_none(),
        )
    }
    pub fn validate_record_model(
        record: &LearnerRecord,
        name: &str,
        architecture: &TemporalQArchitecture,
    ) -> Result<()> {
        architecture.validate()?;
        temporal_models::validate_record_model_count(record, name, architecture.parameter_count()?)
    }
    pub fn restore_from_record(
        record: &LearnerRecord,
        name: &str,
        architecture: &TemporalQArchitecture,
    ) -> Result<Self> {
        Self::restore_from_record_with_mode(record, name, architecture, false)
    }
    pub(crate) fn restore_from_record_with_mode(
        record: &LearnerRecord,
        name: &str,
        architecture: &TemporalQArchitecture,
        inference: bool,
    ) -> Result<Self> {
        Self::validate_record_model(record, name, architecture)?;
        // Raw dimensions, finite tensor contents and allocation bound were checked
        // before constructing this plain-device template from untrusted metadata.
        let template = Self::from_architecture_with_mode(
            architecture,
            &mut ChaCha8Rng::seed_from_u64(0),
            true,
        )?;
        let model = if inference {
            record.load_model_inference(name, &template)?
        } else {
            record.load_model(name, &template)?
        };
        ensure(
            model.architecture()? == *architecture,
            "restored temporal Q architecture differs",
        )?;
        temporal_models::validate_parameters(&model, inference)?;
        Ok(model)
    }
}

impl<E: TemporalEncoder, H: Network, P: Network> TemporalQ<E, H, P> {
    pub fn from_parts(
        encoder: E,
        head: H,
        predictor: Option<P>,
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
                && head.output_dim() > 0
                && encoder.output_dim().checked_add(spec.action_width()) == Some(head.input_dim())
                && predictor
                    .as_ref()
                    .is_none_or(|p| p.input_dim() > 0 && p.output_dim() > 0),
            "temporal Q encoder/action/head dimensions mismatch",
        )?;
        Ok(Self {
            encoder,
            head,
            predictor,
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
    pub fn action_dim(&self) -> usize {
        self.spec.action_width()
    }
    pub fn output_dim(&self) -> usize {
        self.head.output_dim()
    }
    pub fn device(&self) -> Device {
        temporal_models::module_device(self)
    }
    pub fn supports_auxiliary(&self) -> bool {
        self.predictor.is_some()
    }
    pub fn history_features(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>> {
        temporal_models::selected_features(
            &self.encoder,
            batch,
            &self.spec,
            self.observation_only,
            &self.device(),
            tokens,
        )
    }
    fn conditioned(&self, history: Tensor<2>, actions: Tensor<2>) -> Result<Tensor<2>> {
        let n = history.dims()[0];
        ensure(
            n > 0
                && history.dims() == [n, self.encoder.output_dim()]
                && actions.dims() == [n, self.action_dim()],
            "temporal Q history/action query shape mismatch",
        )?;
        // Preserve candidate-action autodiff. Neither query changes or re-assimilates
        // factual history, and the action is never written into a future history token.
        let history = history.to_device(&actions.device());
        Ok(Tensor::cat(vec![history, actions], 1))
    }
    pub fn values_from_features(
        &self,
        history: Tensor<2>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::QEvaluation);
        let n = history.dims()[0];
        let output = self.head.forward(self.conditioned(history, actions)?);
        ensure(
            output.dims() == [n, self.output_dim()],
            "temporal Q output shape mismatch",
        )?;
        Ok(output)
    }
    pub fn values(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.values_from_features(self.history_features(batch, tokens)?, actions)
    }
}
impl<E: TemporalEncoder, H: Network, P: Network> TemporalActionValue for TemporalQ<E, H, P> {
    fn history_spec(&self) -> &HistorySpec {
        &self.spec
    }
    fn action_dim(&self) -> usize {
        self.action_dim()
    }
    fn output_dim(&self) -> usize {
        self.output_dim()
    }
    fn history_features(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>> {
        self.history_features(batch, tokens)
    }
    fn values_from_features(&self, history: Tensor<2>, actions: Tensor<2>) -> Result<Tensor<2>> {
        self.values_from_features(history, actions)
    }
}
impl<E: TemporalEncoder, H: TemporalQHead, P: Network> TemporalQ<E, H, P> {
    pub fn representation_from_features(
        &self,
        history: Tensor<2>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        let predictor = self
            .predictor
            .as_ref()
            .ok_or_else(|| Error::State("temporal Q auxiliary capability is disabled".into()))?;
        let width = self.head.representation_dim();
        let n = history.dims()[0];
        ensure(
            width > 0 && predictor.input_dim() == width && predictor.output_dim() == width,
            "temporal Q auxiliary predictor dimensions mismatch",
        )?;
        let representation = self
            .head
            .representation(self.conditioned(history, actions)?)?;
        ensure(
            representation.dims() == [n, width],
            "temporal Q auxiliary representation shape mismatch",
        )?;
        Ok(representation)
    }
    pub fn predict_representation_from_features(
        &self,
        history: Tensor<2>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        let representation = self.representation_from_features(history, actions)?;
        let dims = representation.dims();
        let result = self
            .predictor
            .as_ref()
            .expect("validated auxiliary capability")
            .forward(representation);
        ensure(
            result.dims() == dims,
            "temporal Q auxiliary prediction shape mismatch",
        )?;
        Ok(result)
    }
    pub fn representation(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.representation_from_features(self.history_features(batch, tokens)?, actions)
    }
    pub fn predict_representation(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.predict_representation_from_features(self.history_features(batch, tokens)?, actions)
    }
}
impl<E: TemporalEncoder, H: TemporalQHead, P: Network> TemporalReppoCritic for TemporalQ<E, H, P> {
    fn supports_auxiliary(&self) -> bool {
        self.supports_auxiliary()
    }
    fn representation(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.representation(batch, tokens, actions)
    }
    fn predict_representation(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.predict_representation(batch, tokens, actions)
    }
}
