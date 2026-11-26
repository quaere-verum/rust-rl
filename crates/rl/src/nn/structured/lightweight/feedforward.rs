use super::{
    ConditionalPolicyLaw, LightweightBatch, LightweightConfig, LightweightEncoder,
    conditional::{
        ConditionalHeads, LightweightPolicyEvaluation, check_head_budget, head_parameter_count,
        key_width, query_width,
    },
    encoder::candidate_targets,
};
use crate::{
    Result,
    core::{
        action::ConditionalAction,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        structured::StructuredObservation,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
        structured::{
            BuildCandidatePolicy, CandidatePolicy, InputSchema, PackLimits, StructuredSample,
            restore_model,
        },
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;
use std::hash::Hash;

/// Conditional policy over a single public frame and its entire event notebook.
#[derive(Module, Debug)]
pub struct LightweightPolicy {
    pub encoder: LightweightEncoder,
    pub operations: Mlp,
    pub operation_head: Mlp,
    pub entity_query: Mlp,
    pub entity_key: Mlp,
    pub location_query: Mlp,
    pub location_key: Mlp,
    #[module(skip)]
    config: LightweightConfig,
}

impl LightweightPolicy {
    pub fn parameter_count(config: &LightweightConfig) -> Result<usize> {
        config.validate()?;
        let parameters = config.encoder_parameter_count()
            + (config.input.operations.len() + 1) * config.operation_width
            + head_parameter_count(config, config.fused_width);
        ensure(
            parameters <= 16 * 1024 * 1024,
            "lightweight policy parameter budget exceeded",
        )?;
        Ok(parameters)
    }

    pub fn new(config: LightweightConfig, rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        Self::parameter_count(&config)?;
        let query = query_width(&config, config.fused_width);
        let key = key_width(&config);
        Ok(Self {
            encoder: LightweightEncoder::new(config.clone(), rng, device)?,
            operations: Mlp::new_on_device(
                config.input.operations.len(),
                &[],
                config.operation_width,
                rng,
                device,
            )?,
            operation_head: Mlp::new_on_device(
                query + key + 1,
                &[config.head_width],
                1,
                rng,
                device,
            )?,
            entity_query: Mlp::new_on_device(query, &[], config.entity_width, rng, device)?,
            entity_key: Mlp::new_on_device(key, &[], config.entity_width, rng, device)?,
            location_query: Mlp::new_on_device(query, &[], config.entity_width, rng, device)?,
            location_key: Mlp::new_on_device(key, &[], config.entity_width, rng, device)?,
            config,
        })
    }

    pub fn config(&self) -> &LightweightConfig {
        &self.config
    }

    pub fn model_device(&self) -> Device {
        self.encoder.entities.layers[0].weight.val().device()
    }

    pub fn complete_law(
        &self,
        observation: &StructuredObservation,
        limits: PackLimits,
    ) -> Result<ConditionalPolicyLaw> {
        let sample = StructuredSample::from_observation_with_validated_schema(
            observation,
            &self.config.input,
        )?;
        self.evaluate(std::slice::from_ref(&sample), limits)?
            .complete_law(0)
    }

    pub(crate) fn evaluate<'p, 'a, K: Eq + Hash>(
        &'p self,
        samples: &'a [StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<LightweightPolicyEvaluation<'p, 'a, StructuredSample<K>>> {
        ensure(
            samples
                .iter()
                .all(|sample| sample.candidates.iter().any(|candidate| candidate.enabled)),
            "lightweight policy requires live candidate support",
        )?;
        let c = &self.config;
        let batch = LightweightBatch::pack_with_validated_schema(
            &c.input,
            samples,
            limits,
            &self.model_device(),
        )?;
        let head_elements =
            check_head_budget(c, c.fused_width, samples.len(), batch.candidate_count)?;
        ensure(
            batch
                .input_elements()
                .checked_add(head_elements)
                .is_some_and(|total| {
                    total <= limits.max_elements && total <= isize::MAX as usize / size_of::<f32>()
                }),
            "lightweight policy exceeds allocation budget",
        )?;
        let world = self.encoder.forward(&batch)?;
        let [b, n, w] = world.entities.dims();
        c.check_shape(&[b * n + 1, w])?;
        let targets = candidate_targets(world.entities, &batch);
        Ok(ConditionalHeads {
            config: c,
            operations: &self.operations,
            operation_head: &self.operation_head,
            entity_query: &self.entity_query,
            entity_key: &self.entity_key,
            location_query: &self.location_query,
            location_key: &self.location_key,
        }
        .evaluate(
            world.global,
            targets,
            batch.arguments,
            vec![samples.iter().collect()],
        )
        .pop()
        .expect("one lightweight observation group"))
    }
}

impl Checkpoint for LightweightPolicy {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("lightweight-policy-v1", &self.config)?;
        record.model("policy", self)?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("lightweight-policy-v1")?;
        let config: LightweightConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "lightweight policy checkpoint configuration mismatch",
        )?;
        *self = restore_model(record, "policy", self)?;
        Ok(())
    }
}

impl CandidatePolicy for LightweightPolicy {
    type Configuration = LightweightConfig;

    fn configuration(&self) -> &LightweightConfig {
        self.config()
    }

    fn validate_configuration(config: &LightweightConfig) -> Result<()> {
        Self::parameter_count(config).map(|_| ())
    }

    fn input_schema(&self) -> &InputSchema {
        &self.config.input
    }

    fn model_device(&self) -> Device {
        self.model_device()
    }

    fn complete_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
    ) -> Result<Vec<Vec<f64>>> {
        let model = self.valid();
        let evaluation = model.evaluate(samples, limits)?;
        (0..samples.len())
            .map(|row| evaluation.complete_probabilities_f64(row))
            .collect()
    }

    fn complete_log_probabilities<K: Clone + Eq + Hash>(
        &self,
        samples: &[StructuredSample<K>],
        keys: &[Vec<K>],
        limits: PackLimits,
    ) -> Result<Vec<Tensor<1>>> {
        ensure(
            keys.len() == samples.len(),
            "candidate policy key row count differs",
        )?;
        let evaluation = self.evaluate(samples, limits)?;
        keys.iter()
            .enumerate()
            .map(|(row, keys)| evaluation.log_prob_many(row, keys))
            .collect()
    }

    fn greedy_action(
        &self,
        sample: &StructuredSample<ConditionalAction>,
        limits: PackLimits,
    ) -> Result<ConditionalAction> {
        Ok(self
            .valid()
            .evaluate(std::slice::from_ref(sample), limits)?
            .greedy(0)?
            .key)
    }
}

impl BuildCandidatePolicy for LightweightPolicy {
    fn from_configuration(
        config: LightweightConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new(config, rng, device)
    }
}

/// Independent scalar value of the current public frame.
#[derive(Module, Debug)]
pub struct LightweightValue {
    pub encoder: LightweightEncoder,
    pub head: Mlp,
    #[module(skip)]
    config: LightweightConfig,
}

impl LightweightValue {
    pub fn parameter_count(config: &LightweightConfig) -> Result<usize> {
        config.validate()?;
        Ok(config.encoder_parameter_count()
            + (config.fused_width + 1) * config.head_width
            + config.head_width
            + 1)
    }

    pub fn new(config: LightweightConfig, rng: &mut ChaCha8Rng, device: &Device) -> Result<Self> {
        Self::parameter_count(&config)?;
        Ok(Self {
            encoder: LightweightEncoder::new(config.clone(), rng, device)?,
            head: Mlp::new_on_device(config.fused_width, &[config.head_width], 1, rng, device)?,
            config,
        })
    }

    pub fn config(&self) -> &LightweightConfig {
        &self.config
    }

    pub(crate) fn evaluate(
        &self,
        samples: &[StructuredSample<ConditionalAction>],
        limits: PackLimits,
    ) -> Result<Tensor<1>> {
        let c = &self.config;
        c.check_shape(&[samples.len(), c.fused_width.max(c.head_width)])?;
        let device = self.encoder.entities.layers[0].weight.val().device();
        let batch =
            LightweightBatch::pack_with_validated_schema(&c.input, samples, limits, &device)?;
        Ok(self
            .head
            .forward(self.encoder.forward(&batch)?.global)
            .reshape([samples.len()]))
    }
}
