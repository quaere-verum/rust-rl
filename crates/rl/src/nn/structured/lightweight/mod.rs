//! Small shared entity/evidence representation over the canonical public schema.
//! Version 1 validates but ignores map pixels. Public coordinates and candidate
//! costs remain available; this is a lossy representation, not a belief filter.
mod conditional;
mod encoder;
mod feedforward;
mod pack;
mod q;
mod recurrent;
use super::InputSchema;
use crate::{Result, core::ensure};
pub use encoder::{LightweightEncoder, LightweightStage, LightweightWorld};
pub use feedforward::{LightweightPolicy, LightweightValue};
pub use pack::LightweightBatch;
pub use q::LightweightQ;
pub use recurrent::{
    ConditionalPolicyLaw, FrozenRecurrentFeatures, RecurrentCandidatePrediction,
    RecurrentDistributionalQEvaluation, RecurrentLightweightConfig,
    RecurrentLightweightDistributionalQ, RecurrentLightweightMemory, RecurrentLightweightPolicy,
    RecurrentLightweightQ, RecurrentLightweightStage, RecurrentLightweightState,
    RecurrentLightweightValue, RecurrentPolicyEvaluation, StructuredHistoryInput,
    StructuredHistoryValues,
};
use serde::{Deserialize, Serialize};

/// Version 1 fixes linked mean evidence pooling with log(1+count) features,
/// distinct subject/context channels, chronological ordinal, no map tensors,
/// and normalized real-map coordinates for location arguments. A change to any
/// of those meanings requires a new version, even when tensor shapes match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightweightConfig {
    pub version: u32,
    pub input: InputSchema,
    pub entity_width: usize,
    pub scalar_width: usize,
    pub event_width: usize,
    pub operation_width: usize,
    pub fused_width: usize,
    pub head_width: usize,
    /// Ceiling per intermediate tensor; not total autodiff memory.
    pub max_activation_elements: usize,
}
impl LightweightConfig {
    pub fn small(input: InputSchema) -> Self {
        Self {
            version: 1,
            input,
            entity_width: 16,
            scalar_width: 32,
            event_width: 16,
            operation_width: 8,
            fused_width: 32,
            head_width: 32,
            max_activation_elements: 16 * 1024 * 1024,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.input.validate()?;
        ensure(
            self.version == 1
                && self.input.operations.len() <= 1024
                && self.max_activation_elements > 0
                && [
                    self.entity_width,
                    self.scalar_width,
                    self.event_width,
                    self.operation_width,
                    self.fused_width,
                    self.head_width,
                ]
                .iter()
                .all(|&n| n > 0 && n <= 1024),
            "invalid lightweight model configuration",
        )?;
        // All dimensions are bounded above before this arithmetic.
        let parameters = self.encoder_parameter_count()
            + (self.input.operations.len() + 1) * self.operation_width
            + (self.joint_width() + 1) * self.head_width
            + self.head_width
            + 1;
        ensure(
            parameters <= 16 * 1024 * 1024,
            "lightweight parameter budget exceeded",
        )
    }
    fn encoder_parameter_count(&self) -> usize {
        let affine = |input, output| (input + 1) * output;
        affine(self.input.entity_features, self.entity_width)
            + affine(self.entity_width, self.entity_width)
            + affine(self.input.scalar_features, self.scalar_width)
            + affine(self.scalar_width, self.scalar_width)
            + affine(self.event_input_width(), self.event_width)
            + affine(self.event_width, self.event_width)
            + affine(self.fusion_input_width(), self.fused_width)
    }
    fn event_input_width(&self) -> usize {
        self.input.event_features + 1 + 2 * self.entity_width + 2
    }
    fn fusion_input_width(&self) -> usize {
        self.entity_width + self.scalar_width + self.event_width + 2
    }
    pub fn target_width(&self) -> usize {
        self.entity_width + 2 * self.event_width + 2
    }
    fn joint_width(&self) -> usize {
        self.fused_width
            + self.operation_width
            + self.target_width()
            + self.input.candidate_features
            + 5
    }
    fn check_shape(&self, shape: &[usize]) -> Result<()> {
        ensure(
            shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .is_some_and(|n| n <= self.max_activation_elements),
            "lightweight activation budget exceeded",
        )
    }
}
