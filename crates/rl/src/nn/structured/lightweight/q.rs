use super::super::{
    BuildStructuredActionValue, CandidateValues, InputSchema, PackLimits, StructuredActionValue,
    StructuredSample,
};
use super::{LightweightBatch, LightweightConfig, LightweightEncoder, encoder::candidate_targets};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;
use std::hash::Hash;

/// Shared complete-action Q head. Operation values are maxima of these same
/// values, never separately trained predictions. No replay or targets live here.
#[derive(Module, Debug)]
pub struct LightweightQ {
    pub encoder: LightweightEncoder,
    pub operations: Mlp,
    pub head: Mlp,
    #[module(skip)]
    config: LightweightConfig,
}
impl LightweightQ {
    pub fn with_initialization(
        mut self,
        initialization: crate::nn::critics::CriticInitialization,
    ) -> Result<Self> {
        initialization.apply(&mut self.head)?;
        Ok(self)
    }

    pub fn new(config: LightweightConfig, rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            encoder: LightweightEncoder::new(config.clone(), rng, device)?,
            operations: Mlp::new_on_device(
                config.input.operations.len(),
                &[],
                config.operation_width,
                rng,
                device,
            )?,
            head: Mlp::new_on_device(config.joint_width(), &[config.head_width], 1, rng, device)?,
            config,
        })
    }
    pub fn config(&self) -> &LightweightConfig {
        &self.config
    }
    /// Flat values in packed sample/candidate order; packing is public so callers
    /// may profile preparation separately. Semantic binding uses candidate_values.
    pub fn evaluate(&self, batch: &LightweightBatch) -> Result<Tensor<1>> {
        let input = candidate_features(&self.config, &self.encoder, &self.operations, batch)?;
        Ok(self.head.forward(input).reshape([batch.candidate_count]))
    }
}

pub(super) fn candidate_features(
    config: &LightweightConfig,
    encoder: &LightweightEncoder,
    operations: &Mlp,
    batch: &LightweightBatch,
) -> Result<Tensor<2>> {
    ensure(batch.candidate_count > 0, "no lightweight Q candidates")?;
    config.check_shape(&[
        batch.candidate_count,
        config.joint_width().max(config.head_width),
    ])?;
    let world = encoder.forward(batch)?;
    let [b, n, w] = world.entities.dims();
    config.check_shape(&[b * n + 1, w])?;
    let targets = candidate_targets(world.entities, batch);
    Ok(Tensor::cat(
        vec![
            world.global.select(0, batch.owners.clone()),
            operations.forward(batch.operations.clone()),
            targets,
            batch.arguments.clone(),
        ],
        1,
    ))
}
impl BuildStructuredActionValue for LightweightQ {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new(config, rng, device)
    }
}
impl StructuredActionValue for LightweightQ {
    type Configuration = LightweightConfig;
    fn configuration(&self) -> &LightweightConfig {
        &self.config
    }
    fn validate_configuration(config: &LightweightConfig) -> Result<()> {
        config.validate()
    }
    fn input_schema(&self) -> &InputSchema {
        &self.config.input
    }
    fn model_device(&self) -> Device {
        self.encoder.entities.layers[0].weight.val().device()
    }
    fn candidate_values<'a, K: Clone + Eq + Hash>(
        &self,
        samples: &'a [StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        let batch = LightweightBatch::pack_with_validated_schema(
            &self.config.input,
            samples,
            limits,
            &self.model_device(),
        )?;
        ensure(
            samples
                .iter()
                .all(|s| s.candidates.iter().any(|c| c.enabled)),
            "no enabled lightweight Q candidate",
        )?;
        let values = self.evaluate(&batch)?;
        let mut offset = 0;
        samples
            .iter()
            .map(|s| {
                let start = offset;
                offset += s.candidates.len();
                CandidateValues::from_candidates(
                    &s.candidates,
                    values.clone().narrow(0, start, offset - start),
                    self.config.input.operations.len(),
                )
            })
            .collect()
    }
}
impl Checkpoint for LightweightQ {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut r = LearnerRecord::new("lightweight-q-v1", &self.config)?;
        r.model("q", self)?;
        Ok(r)
    }
    fn restore(&mut self, r: &LearnerRecord) -> Result<()> {
        r.check("lightweight-q-v1")?;
        let config: LightweightConfig = serde_json::from_value(r.metadata.clone())?;
        ensure(
            config == self.config,
            "lightweight Q checkpoint configuration mismatch",
        )?;
        *self = super::super::checkpoint::restore(r, "q", self)?;
        Ok(())
    }
}
