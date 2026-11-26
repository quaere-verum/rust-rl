use super::{super::mean_rows, LightweightBatch, LightweightConfig};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        canonical,
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, activation::relu},
};
use rand_chacha::ChaCha8Rng;

#[derive(Module, Debug)]
pub struct LightweightEncoder {
    pub entities: Mlp,
    pub scalars: Mlp,
    pub events: Mlp,
    pub fusion: Mlp,
    #[module(skip)]
    config: LightweightConfig,
}
#[derive(Debug, Clone)]
pub struct LightweightWorld {
    /// `[batch, fused_width]`; no recurrent state is owned by this encoder.
    pub global: Tensor<2>,
    /// `[batch, padded_entities, target_width]`: base features, separate mean
    /// subject/context evidence summaries and their log(1+count) features.
    pub entities: Tensor<3>,
    /// `[batch, padded_events, event_width]`; padded rows are exactly zero.
    pub events: Tensor<3>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightweightStage {
    Validation,
    Entities,
    Evidence,
    ScalarsAndFusion,
}

pub(super) fn forward_mlp(model: &Mlp, input: Tensor<2>, lanes: usize) -> Tensor<2> {
    if lanes == 1 {
        model.forward(input)
    } else {
        let rows_per_lane = input.dims()[0] / lanes;
        canonical::grouped_mlp(model, input, rows_per_lane)
    }
}
fn linked_mean(links: Tensor<3>, events: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
    let links = links.swap_dims(1, 2);
    let count = links.clone().sum_dim(2);
    (
        links.matmul(events) / count.clone().clamp_min(1.),
        (count + 1.).log(),
    )
}

pub(super) fn candidate_targets(entities: Tensor<3>, batch: &LightweightBatch) -> Tensor<2> {
    let [b, n, width] = entities.dims();
    let device = entities.device();
    Tensor::cat(
        vec![
            entities.reshape([b * n, width]),
            Tensor::zeros([1, width], &device),
        ],
        0,
    )
    .select(0, batch.targets.clone())
}

impl LightweightEncoder {
    pub fn new(config: LightweightConfig, rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
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
    pub fn config(&self) -> &LightweightConfig {
        &self.config
    }
    pub fn forward(&self, batch: &LightweightBatch) -> Result<LightweightWorld> {
        self.forward_observed(batch, |_| {})
    }
    /// Caller-owned stage timing; synchronous CPU timings are execution timings.
    pub fn forward_observed(
        &self,
        batch: &LightweightBatch,
        observe: impl FnMut(LightweightStage),
    ) -> Result<LightweightWorld> {
        self.forward_with_grouping(batch, false, observe)
    }

    pub(super) fn forward_with_grouping(
        &self,
        batch: &LightweightBatch,
        grouped: bool,
        mut observe: impl FnMut(LightweightStage),
    ) -> Result<LightweightWorld> {
        let c = &self.config;
        ensure(
            batch.schema == c.input,
            "lightweight encoder schema mismatch",
        )?;
        let [b, n, e] = batch.entities.dims();
        let [_, t, _] = batch.events.dims();
        for shape in [
            &[b, n, c.entity_width][..],
            &[b, t, c.event_input_width()],
            &[b, t, c.event_width],
            &[b, n, c.target_width()],
            &[b, c.scalar_width],
            &[b, c.fusion_input_width()],
            &[b, c.fused_width],
        ] {
            c.check_shape(shape)?;
        }
        observe(LightweightStage::Validation);
        let lanes = if grouped { b } else { 1 };
        let entities = relu(forward_mlp(
            &self.entities,
            batch.entities.clone().reshape([b * n, e]),
            lanes,
        ))
        .reshape([b, n, c.entity_width])
            * batch.entity_mask.clone();
        let entity_global = mean_rows(entities.clone(), batch.entity_mask.clone());
        observe(LightweightStage::Entities);
        let context_count = batch.contexts.clone().sum_dim(2);
        let event_input = Tensor::cat(
            vec![
                batch.events.clone(),
                batch.subjects.clone().matmul(entities.clone()),
                batch.contexts.clone().matmul(entities.clone())
                    / context_count.clone().clamp_min(1.),
                batch.subjects.clone().sum_dim(2),
                (context_count + 1.).log(),
            ],
            2,
        );
        let events = relu(forward_mlp(
            &self.events,
            event_input.reshape([b * t, c.event_input_width()]),
            lanes,
        ))
        .reshape([b, t, c.event_width])
            * batch.event_mask.clone();
        let (subject_evidence, subject_count) = linked_mean(batch.subjects.clone(), events.clone());
        let (context_evidence, context_count) = linked_mean(batch.contexts.clone(), events.clone());
        let entities = Tensor::cat(
            vec![
                entities,
                subject_evidence,
                context_evidence,
                subject_count,
                context_count,
            ],
            2,
        ) * batch.entity_mask.clone();
        let event_global = mean_rows(events.clone(), batch.event_mask.clone());
        observe(LightweightStage::Evidence);
        let scalars = relu(forward_mlp(&self.scalars, batch.scalars.clone(), lanes));
        let global = forward_mlp(
            &self.fusion,
            Tensor::cat(
                vec![entity_global, scalars, event_global, batch.counts.clone()],
                1,
            ),
            lanes,
        );
        observe(LightweightStage::ScalarsAndFusion);
        Ok(LightweightWorld {
            global,
            entities,
            events,
        })
    }
}
impl Checkpoint for LightweightEncoder {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut r = LearnerRecord::new("lightweight-encoder-v1", &self.config)?;
        r.model("encoder", self)?;
        Ok(r)
    }
    fn restore(&mut self, r: &LearnerRecord) -> Result<()> {
        r.check("lightweight-encoder-v1")?;
        let config: LightweightConfig = serde_json::from_value(r.metadata.clone())?;
        ensure(
            config == self.config,
            "lightweight encoder checkpoint configuration mismatch",
        )?;
        *self = super::super::checkpoint::restore(r, "encoder", self)?;
        Ok(())
    }
}
