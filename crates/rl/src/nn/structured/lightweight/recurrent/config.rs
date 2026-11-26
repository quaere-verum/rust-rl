use super::super::LightweightConfig;
use crate::{
    Result,
    core::{ensure, structured::INCREMENTAL_EVENT_CONVENTION},
    nn::structured::InputSchema,
};
use serde::{Deserialize, Serialize};

/// Version 1 binds event/decision ordering, signed-log reward, log duration,
/// feature selection and exact-prefix execution, including allocation limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecurrentLightweightConfig {
    pub version: u32,
    pub model: LightweightConfig,
    pub hidden_width: usize,
    pub suppressed_entity_features: Vec<usize>,
    pub suppressed_scalar_features: Vec<usize>,
    pub max_observations: usize,
    pub max_tokens: usize,
    /// Ceiling on borrowed public-frame and primitive-reward payload per call.
    pub max_history_bytes: usize,
}
impl RecurrentLightweightConfig {
    pub fn small(input: InputSchema) -> Self {
        Self {
            version: 1,
            model: LightweightConfig::small(input),
            hidden_width: 64,
            suppressed_entity_features: vec![],
            suppressed_scalar_features: vec![],
            max_observations: 4096,
            max_tokens: 16384,
            max_history_bytes: 64 * 1024 * 1024,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        let columns =
            |v: &[usize], width| v.iter().all(|&i| i < width) && v.windows(2).all(|w| w[0] < w[1]);
        ensure(
            self.version == 1
                && (1..=1024).contains(&self.hidden_width)
                && self.max_observations > 0
                && self.max_tokens > 0
                && self.max_history_bytes > 0
                && columns(
                    &self.suppressed_entity_features,
                    self.model.input.entity_features,
                )
                && columns(
                    &self.suppressed_scalar_features,
                    self.model.input.scalar_features,
                )
                && self
                    .model
                    .input
                    .semantics
                    .conventions
                    .iter()
                    .any(|c| c == INCREMENTAL_EVENT_CONVENTION),
            "invalid recurrent lightweight configuration or event delivery contract",
        )?;
        let c = &self.model;
        let recurrent_parameters =
            3 * (c.fused_width + 2 + self.hidden_width + 2) * self.hidden_width
                + (self.decision_width() + 1) * c.fused_width
                + (c.event_width + 1) * c.fused_width
                + (self.joint_width() + 1) * c.head_width;
        ensure(
            recurrent_parameters <= 16 * 1024 * 1024,
            "recurrent parameter budget exceeded",
        )
    }
    pub(super) fn decision_width(&self) -> usize {
        let c = &self.model;
        c.fused_width + c.operation_width + c.entity_width + c.input.candidate_features + 8
    }
    pub(super) fn joint_width(&self) -> usize {
        let c = &self.model;
        self.hidden_width + c.operation_width + c.target_width() + c.input.candidate_features + 5
    }

    /// Exact allocation count after validation of the architecture widths.
    pub(super) fn backbone_parameter_count(&self) -> usize {
        let c = &self.model;
        let affine = |input, output| (input + 1) * output;
        affine(c.input.entity_features, c.entity_width)
            + affine(c.entity_width, c.entity_width)
            + affine(c.input.scalar_features, c.scalar_width)
            + affine(c.scalar_width, c.scalar_width)
            + affine(c.event_input_width(), c.event_width)
            + affine(c.event_width, c.event_width)
            + affine(c.fusion_input_width(), c.fused_width)
            + affine(c.input.operations.len(), c.operation_width)
            + affine(c.event_width, c.fused_width)
            + affine(self.decision_width(), c.fused_width)
            + 3 * (c.fused_width + 2 + self.hidden_width + 2) * self.hidden_width
    }
}
