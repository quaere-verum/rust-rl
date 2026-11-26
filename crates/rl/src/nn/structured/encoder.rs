use super::{InputSchema, StructuredBatch, mean_rows};
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
    module::{Module, Param},
    nn::{PaddingConfig2d, conv::Conv2d},
    tensor::{Device, Tensor, TensorData, activation::relu},
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderConfig {
    pub version: u32,
    pub input: InputSchema,
    pub conv_channels: [usize; 2],
    pub entity_width: usize,
    pub scalar_width: usize,
    pub event_width: usize,
    pub fused_width: usize,
    /// Ceiling for each intermediate feature tensor, not total autodiff memory.
    pub max_activation_elements: usize,
}
impl EncoderConfig {
    pub fn small(input: InputSchema) -> Self {
        Self {
            version: 1,
            input,
            conv_channels: [16, 32],
            entity_width: 32,
            scalar_width: 16,
            event_width: 32,
            fused_width: 64,
            max_activation_elements: 16 * 1024 * 1024,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.input.validate()?;
        ensure(
            self.version == 1
                && self.input.spatial_channels > 0
                && [
                    self.conv_channels[0],
                    self.conv_channels[1],
                    self.entity_width,
                    self.scalar_width,
                    self.event_width,
                    self.fused_width,
                ]
                .iter()
                .all(|&n| n > 0 && n <= 1024)
                && self.max_activation_elements > 0,
            "invalid structured encoder configuration",
        )?;
        let parameter_elements = self.input.spatial_channels * self.conv_channels[0] * 9
            + self.conv_channels[0] * self.conv_channels[1] * 9
            + self.input.entity_features * self.entity_width
            + self.entity_width * self.entity_width
            + self.event_input_width() * self.event_width
            + self.event_width * self.event_width
            + self.input.scalar_features * self.scalar_width
            + self.scalar_width * self.scalar_width
            + self.fusion_input_width() * self.fused_width;
        ensure(
            parameter_elements <= 16 * 1024 * 1024,
            "structured encoder parameter budget exceeded",
        )
    }
    fn event_input_width(&self) -> usize {
        self.input.event_features + 1 + 2 * self.entity_width + 2
    }
    fn fusion_input_width(&self) -> usize {
        self.conv_channels[1] + self.entity_width + self.scalar_width + self.event_width
    }
}

/// Independent model owner. Instantiate separately for actor and critic.
/// The fused output is a normal tensor suitable for a future temporal core;
/// this feedforward module does not manage recurrent state or collectors.
#[derive(Module, Debug)]
pub struct StructuredEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub entities: Mlp,
    pub scalars: Mlp,
    pub events: Mlp,
    pub fusion: Mlp,
    #[module(skip)]
    config: EncoderConfig,
}
#[derive(Debug, Clone)]
pub struct EncodedWorld {
    /// `[batch, fused_width]`.
    pub global: Tensor<2>,
    /// `[batch, padded_entities, entity_width]`, padded rows exactly zero.
    pub entities: Tensor<3>,
    /// `[batch, final_channels, padded_height, padded_width]`.
    pub spatial: Tensor<4>,
    /// `[batch, padded_events, event_width]`, ordinal-aware chronological rows.
    pub events: Tensor<3>,
}

/// Completed encoder stages exposed to optional caller-owned profiling.
/// Callbacks do not synchronize asynchronous tensor backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderStage {
    Validation,
    Spatial,
    Entities,
    Events,
    ScalarsAndFusion,
}
fn conv(input: usize, output: usize, rng: &mut ChaCha8Rng, device: &Device) -> Conv2d {
    let bound = (6. / ((input + output) * 9) as f32).sqrt();
    let values: Vec<f32> = (0..input * output * 9)
        .map(|_| rng.random_range(-bound..bound))
        .collect();
    Conv2d {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(values, [output, input, 3, 3]),
            device,
        )),
        bias: Some(Param::from_tensor(Tensor::zeros([output], device))),
        stride: [1, 1],
        kernel_size: [3, 3],
        dilation: [1, 1],
        groups: 1,
        padding: PaddingConfig2d::Explicit(1, 1, 1, 1),
    }
}
impl StructuredEncoder {
    pub fn new(config: EncoderConfig, rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            conv1: conv(
                config.input.spatial_channels,
                config.conv_channels[0],
                rng,
                device,
            ),
            conv2: conv(
                config.conv_channels[0],
                config.conv_channels[1],
                rng,
                device,
            ),
            entities: Mlp::new_on_device(
                config.input.entity_features,
                &[config.entity_width],
                config.entity_width,
                rng,
                device,
            )?,
            scalars: Mlp::new_on_device(
                config.input.scalar_features,
                &[config.scalar_width],
                config.scalar_width,
                rng,
                device,
            )?,
            events: Mlp::new_on_device(
                config.event_input_width(),
                &[config.event_width],
                config.event_width,
                rng,
                device,
            )?,
            fusion: Mlp::new_on_device(
                config.fusion_input_width(),
                &[],
                config.fused_width,
                rng,
                device,
            )?,
            config,
        })
    }
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
    pub fn forward<K>(&self, batch: &StructuredBatch<K>) -> Result<EncodedWorld> {
        self.forward_observed(batch, |_| {})
    }
    /// The ordinary forward pass with a callback after each completed stage.
    /// The callback receives no model or tensor access and is not checkpointed.
    /// Use a synchronous backend for stage wall times; GPU enqueue times are
    /// not device execution times. The default forward has a no-op observer.
    pub fn forward_observed<K>(
        &self,
        batch: &StructuredBatch<K>,
        mut observe: impl FnMut(EncoderStage),
    ) -> Result<EncodedWorld> {
        let cfg = &self.config;
        ensure(
            batch.schema == cfg.input,
            "structured encoder input schema mismatch",
        )?;
        let [b, _, h, w] = batch.spatial.dims();
        let [_, n, e] = batch.entities.dims();
        let [_, t, f] = batch.events.dims();
        for dims in [
            [b, h, w, cfg.conv_channels[0]],
            [b, h, w, cfg.conv_channels[1]],
            [b, n, 1, cfg.entity_width],
            [b, t, 1, cfg.event_input_width()],
            [b, t, 1, cfg.event_width],
            [b, 1, 1, cfg.fusion_input_width()],
            [b, 1, 1, cfg.fused_width],
        ] {
            ensure(
                dims.into_iter()
                    .try_fold(1usize, |a, v| a.checked_mul(v))
                    .is_some_and(|n| n <= cfg.max_activation_elements),
                "structured activation exceeds configured limit",
            )?;
        }
        observe(EncoderStage::Validation);
        // Re-mask after *every* convolution. Otherwise bias and neighboring
        // valid pixels create padded activations that leak back on layer two.
        let hidden = relu(self.conv1.forward(batch.spatial.clone())) * batch.cell_mask.clone();
        let spatial = relu(self.conv2.forward(hidden)) * batch.cell_mask.clone();
        let grid_global = spatial
            .clone()
            .sum_dim(2)
            .sum_dim(3)
            .reshape([b, cfg.conv_channels[1]])
            / batch
                .cell_mask
                .clone()
                .sum_dim(2)
                .sum_dim(3)
                .reshape([b, 1]);
        observe(EncoderStage::Spatial);
        let entities = relu(
            self.entities
                .forward(batch.entities.clone().reshape([b * n, e])),
        )
        .reshape([b, n, cfg.entity_width])
            * batch.entity_mask.clone();
        let entity_global = mean_rows(entities.clone(), batch.entity_mask.clone());
        observe(EncoderStage::Entities);
        let targets = batch.event_entities.clone().matmul(entities.clone());
        let context = batch.event_context.clone().matmul(entities.clone());
        let event_input = Tensor::cat(
            vec![
                batch.events.clone(),
                targets,
                context,
                batch.event_entities.clone().sum_dim(2),
                batch.event_context.clone().sum_dim(2),
            ],
            2,
        );
        debug_assert_eq!(event_input.dims()[2], f + 2 * cfg.entity_width + 2);
        let events = relu(
            self.events
                .forward(event_input.reshape([b * t, cfg.event_input_width()])),
        )
        .reshape([b, t, cfg.event_width])
            * batch.event_mask.clone();
        let event_global = mean_rows(events.clone(), batch.event_mask.clone());
        observe(EncoderStage::Events);
        let scalar_global = relu(self.scalars.forward(batch.scalars.clone()));
        let global = self.fusion.forward(Tensor::cat(
            vec![grid_global, entity_global, scalar_global, event_global],
            1,
        ));
        observe(EncoderStage::ScalarsAndFusion);
        Ok(EncodedWorld {
            global,
            entities,
            spatial,
            events,
        })
    }
}
impl Checkpoint for StructuredEncoder {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("structured-encoder-v2", &self.config)?;
        record.model("encoder", self)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("structured-encoder-v2")?;
        let config: EncoderConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "structured encoder checkpoint configuration mismatch",
        )?;
        let restored = super::checkpoint::restore(record, "encoder", self)?;
        *self = restored;
        Ok(())
    }
}
