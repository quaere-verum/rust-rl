//! Small neural capability shared by candidate-aware critic fitters.
//! Models choose their representation; factual keys and SMDP targets remain
//! owned by the collector/replay/learner rather than by a neural architecture.
use super::{
    CandidateValues, InputSchema, PackLimits, StructuredBatch, StructuredQ, StructuredSample,
};
use crate::{
    Result,
    core::{
        action::ActionSpace, checkpoint::Checkpoint, ensure, env::ObservationSpace,
        graph::GraphObservation,
    },
};
use burn::{module::Module, tensor::Device};
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, de::DeserializeOwned};
use std::hash::Hash;

pub trait StructuredActionValue: Module + Checkpoint {
    /// Includes semantic schema, architecture family/version and all non-tensor
    /// choices affecting inference. Equal shapes alone are never compatibility.
    type Configuration: Clone + PartialEq + Serialize + DeserializeOwned;
    fn configuration(&self) -> &Self::Configuration;
    fn validate_configuration(config: &Self::Configuration) -> Result<()>;
    /// Must be the schema bound by `configuration()`, not mutable external state.
    fn input_schema(&self) -> &InputSchema;
    /// Bind the model to both semantic features and the complete action grammar.
    fn validate_spaces(
        &self,
        observations: &ObservationSpace,
        actions: &ActionSpace,
    ) -> Result<()> {
        self.input_schema().validate_spaces(observations, actions)
    }
    /// Derive from current module tensors. A valid/forked/restored module must
    /// report its actual device and execution mode, not a cached construction mode.
    fn model_device(&self) -> Device;
    /// Differentiable complete-action values in the supplied sample/candidate
    /// order. Return exactly those semantic keys, including disabled rows.
    /// Callers mask support; models must respect the packing allocation budget.
    fn candidate_values<'a, K: Clone + Eq + Hash>(
        &self,
        samples: &'a [StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>>;

    /// Each graph is the public source of the structured sample at the same index.
    /// Environments with plain structured observations supply an empty slice.
    fn candidate_values_with_graphs<'a, K: Clone + Eq + Hash>(
        &self,
        samples: &'a [StructuredSample<K>],
        graphs: &[&GraphObservation],
        limits: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        ensure(
            graphs.is_empty(),
            "structured model requires plain observations",
        )?;
        self.candidate_values(samples, limits)
    }

    /// The pooled observation embedding used by the candidate heads, where exposed.
    #[cfg(feature = "diagnostics")]
    fn diagnostic_global(
        &self,
        _: &crate::core::env::Observation,
        _: PackLimits,
        _: usize,
    ) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
}

/// Models with a public constructor can be rebuilt before restoring a record.
pub trait BuildStructuredActionValue: StructuredActionValue {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self>;
}

impl BuildStructuredActionValue for StructuredQ {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new(config, rng, device)
    }
}

impl StructuredActionValue for StructuredQ {
    type Configuration = super::StructuredModelConfig;
    fn configuration(&self) -> &Self::Configuration {
        self.config()
    }
    fn validate_configuration(config: &Self::Configuration) -> Result<()> {
        config.validate()
    }
    fn input_schema(&self) -> &InputSchema {
        &self.config().encoder.input
    }
    fn model_device(&self) -> Device {
        self.encoder.conv1.weight.val().device()
    }

    #[cfg(feature = "diagnostics")]
    fn diagnostic_global(
        &self,
        observation: &crate::core::env::Observation,
        limits: PackLimits,
        max_width: usize,
    ) -> Result<Option<Vec<f32>>> {
        if self.config().encoder.fused_width > max_width {
            return Ok(None);
        }
        let crate::core::env::Observation::Structured(frame) = observation else {
            return Err(crate::Error::Shape(
                "structured encoder requires a structured frame".into(),
            ));
        };
        let input = self.input_schema();
        let samples = [StructuredSample::from_observation_with_validated_schema(
            frame, input,
        )?];
        let batch = StructuredBatch::pack_with_validated_schema(
            input,
            &samples,
            limits,
            &self.model_device(),
        )?;
        crate::nn::to_vec(self.encoder.forward(&batch)?.global).map(Some)
    }
    fn candidate_values<'a, K: Clone + Eq + Hash>(
        &self,
        samples: &'a [StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        let batch = StructuredBatch::pack_with_validated_schema(
            self.input_schema(),
            samples,
            limits,
            &self.model_device(),
        )?;
        let rows = self.evaluate(&batch)?;
        // Bind returned keys to the caller's samples, not the temporary batch.
        rows.into_iter()
            .zip(samples)
            .map(|(row, sample)| {
                CandidateValues::from_candidates(
                    &sample.candidates,
                    row.values,
                    self.input_schema().operations.len(),
                )
            })
            .collect()
    }
}

pub(crate) fn evaluate_checked<'a, Q: StructuredActionValue, K: Clone + Eq + Hash>(
    model: &Q,
    samples: &'a [StructuredSample<K>],
    graphs: &[&GraphObservation],
    limits: PackLimits,
) -> Result<Vec<CandidateValues<'a, K>>> {
    ensure(!samples.is_empty(), "empty structured model batch")?;
    for sample in samples {
        sample.validate(model.input_schema())?;
        ensure(
            sample.candidates.iter().any(|candidate| candidate.enabled),
            "live structured state has no enabled candidates",
        )?;
    }
    let rows = model.candidate_values_with_graphs(samples, graphs, limits)?;
    ensure(
        rows.len() == samples.len(),
        "structured model batch length mismatch",
    )?;
    for (row, sample) in rows.iter().zip(samples) {
        ensure(
            row.keys.len() == sample.candidates.len()
                && row.values.dims() == [sample.candidates.len()]
                && row
                    .keys
                    .iter()
                    .zip(&sample.candidates)
                    .all(|(key, c)| **key == c.key),
            "structured model candidate values/keys mismatch",
        )?;
    }
    Ok(rows)
}
