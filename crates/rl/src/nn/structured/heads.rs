//! Shared shape/feature preparation, without policy or Bellman semantics.
use super::{Argument, Candidate, EncodedWorld, EncoderConfig, InputSchema, StructuredBatch};
use crate::{Result, core::ensure};
use burn::tensor::{Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredModelConfig {
    pub version: u32,
    pub encoder: EncoderConfig,
    pub head_width: usize,
}
impl StructuredModelConfig {
    pub fn small(input: InputSchema) -> Self {
        Self {
            version: 1,
            encoder: EncoderConfig::small(input),
            head_width: 32,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.encoder.validate()?;
        ensure(
            self.version == 1
                && self.head_width > 0
                && self.head_width <= 1024
                && self.encoder.input.operations.len() <= 1024,
            "invalid structured head configuration",
        )?;
        let e = self.encoder.entity_width;
        let s = self.encoder.conv_channels[1];
        let c = self.encoder.input.candidate_features;
        let linear = |input: usize, output: usize| (input + 1) * output;
        let actor_parameters = linear(self.joint_width() + 1, self.head_width)
            + linear(self.head_width, 1)
            + linear(self.query_width(), e)
            + linear(e + c, e)
            + linear(self.query_width(), s)
            + linear(s + c, s);
        ensure(
            actor_parameters <= 16 * 1024 * 1024,
            "structured head parameter budget exceeded",
        )
    }
    pub(crate) fn target_width(&self) -> usize {
        self.encoder.entity_width
            + self.encoder.conv_channels[1]
            + 3
            + self.encoder.input.candidate_features
    }
    pub(crate) fn query_width(&self) -> usize {
        self.encoder.fused_width + self.encoder.input.operations.len()
    }
    pub(crate) fn joint_width(&self) -> usize {
        self.query_width() + self.target_width()
    }
    pub(crate) fn check_rows(&self, rows: usize, width: usize) -> Result<()> {
        ensure(
            rows.checked_mul(width.max(self.head_width))
                .is_some_and(|n| n <= self.encoder.max_activation_elements),
            "structured head activation budget exceeded",
        )
    }
}
pub(crate) fn row(values: Vec<f32>, device: &Device) -> Tensor<2> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [1, n]), device)
}
pub(crate) fn query(
    world: &EncodedWorld,
    config: &StructuredModelConfig,
    sample: usize,
    operation: usize,
) -> Tensor<2> {
    let mut indicator = vec![0.; config.encoder.input.operations.len()];
    indicator[operation] = 1.;
    Tensor::cat(
        vec![
            world
                .global
                .clone()
                .slice([sample..sample + 1, 0..config.encoder.fused_width]),
            row(indicator, &world.global.device()),
        ],
        1,
    )
}
pub(crate) fn entity(world: &EncodedWorld, sample: usize, target: usize) -> Tensor<2> {
    let d = world.entities.dims()[2];
    world
        .entities
        .clone()
        .slice([sample..sample + 1, target..target + 1, 0..d])
        .reshape([1, d])
}
pub(crate) fn location(world: &EncodedWorld, sample: usize, x: usize, y: usize) -> Tensor<2> {
    let d = world.spatial.dims()[1];
    world
        .spatial
        .clone()
        .slice([sample..sample + 1, 0..d, y..y + 1, x..x + 1])
        .reshape([1, d])
}
pub(crate) fn target<K>(
    world: &EncodedWorld,
    config: &StructuredModelConfig,
    sample: usize,
    candidate: &Candidate<K>,
) -> Tensor<2> {
    let device = world.global.device();
    let mut kind = vec![0.; 3];
    let (entity, spatial) = match candidate.argument {
        Argument::None => {
            kind[0] = 1.;
            (
                Tensor::zeros([1, config.encoder.entity_width], &device),
                Tensor::zeros([1, config.encoder.conv_channels[1]], &device),
            )
        }
        Argument::Entity(i) => {
            kind[1] = 1.;
            (
                entity(world, sample, i),
                Tensor::zeros([1, config.encoder.conv_channels[1]], &device),
            )
        }
        Argument::Location { x, y } => {
            kind[2] = 1.;
            (
                Tensor::zeros([1, config.encoder.entity_width], &device),
                location(world, sample, x, y),
            )
        }
        Argument::RegionalLocation { .. } => unreachable!("batch resolves regional locations"),
    };
    Tensor::cat(
        vec![
            entity,
            spatial,
            row(kind, &device),
            row(candidate.features.clone(), &device),
        ],
        1,
    )
}
pub(crate) fn check_live<K>(
    batch: &StructuredBatch<K>,
    config: &StructuredModelConfig,
) -> Result<()> {
    for candidates in &batch.candidates {
        ensure(
            candidates.iter().any(|c| c.enabled),
            "structured action model requires a live candidate or explicit fallback",
        )?;
        config.check_rows(candidates.len(), config.joint_width())?;
    }
    Ok(())
}
