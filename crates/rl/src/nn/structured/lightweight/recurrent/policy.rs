//! Small operation/conditional-target actor with independently owned memory.
use super::super::{
    LightweightEncoder,
    conditional::{
        ConditionalHeads, LightweightPolicyEvaluation, check_head_budget, head_parameter_count,
        key_width, query_width,
    },
};
use super::{
    RecurrentLightweightConfig, RecurrentLightweightState, StructuredHistoryInput,
    history::{EncodedHistories, HistoryBackbone, HistoryEncoder},
};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        stateful::HistoryInput,
        structured::StructuredObservation,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::Mlp,
        structured::{PackLimits, restore_model},
        temporal::GruEncoder,
    },
};
use burn::{module::Module, tensor::Device};
use rand_chacha::ChaCha8Rng;

/// Same public incremental inputs as the recurrent Q model; no shared parameter
/// ownership, hidden-state cache, CNN or numerical object-identity features.
#[derive(Module, Debug)]
pub struct RecurrentLightweightPolicy {
    pub encoder: LightweightEncoder,
    pub operations: Mlp,
    pub event_projection: Mlp,
    pub decision_projection: Mlp,
    pub memory: GruEncoder,
    pub operation_head: Mlp,
    pub entity_query: Mlp,
    pub entity_key: Mlp,
    pub location_query: Mlp,
    pub location_key: Mlp,
    #[module(skip)]
    config: RecurrentLightweightConfig,
}

/// One factual history. States include its real final successor. Distribution
/// rows use the original observation indices; only an unavailable final terminal
/// row is omitted. Asking for a distribution at that row is an error.
pub type RecurrentPolicyEvaluation<'p, 'a> =
    LightweightPolicyEvaluation<'p, 'a, StructuredObservation>;
impl RecurrentLightweightPolicy {
    pub fn parameter_count(config: &RecurrentLightweightConfig) -> Result<usize> {
        config.validate()?;
        let parameters = head_parameter_count(&config.model, config.hidden_width);
        ensure(
            parameters <= 16 * 1024 * 1024,
            "recurrent actor parameter budget exceeded",
        )?;
        Ok(config.backbone_parameter_count() + parameters)
    }

    pub fn new(
        config: RecurrentLightweightConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::parameter_count(&config)?;
        let c = &config.model;
        let query = query_width(c, config.hidden_width);
        let key = key_width(c);
        let HistoryBackbone {
            encoder,
            operations,
            event_projection,
            decision_projection,
            memory,
        } = HistoryBackbone::new(&config, rng, device)?;
        Ok(Self {
            encoder,
            operations,
            event_projection,
            decision_projection,
            memory,
            operation_head: Mlp::new_on_device(query + key + 1, &[c.head_width], 1, rng, device)?,
            entity_query: Mlp::new_on_device(query, &[], c.entity_width, rng, device)?,
            entity_key: Mlp::new_on_device(key, &[], c.entity_width, rng, device)?,
            location_query: Mlp::new_on_device(query, &[], c.entity_width, rng, device)?,
            location_key: Mlp::new_on_device(key, &[], c.entity_width, rng, device)?,
            config,
        })
    }
    pub fn config(&self) -> &RecurrentLightweightConfig {
        &self.config
    }
    pub fn model_device(&self) -> Device {
        self.history_encoder().device()
    }

    fn history_encoder(&self) -> HistoryEncoder<'_> {
        HistoryEncoder {
            config: &self.config,
            encoder: &self.encoder,
            operations: &self.operations,
            event_projection: &self.event_projection,
            decision_projection: &self.decision_projection,
            memory: &self.memory,
        }
    }

    pub fn initial_state(&self) -> Result<RecurrentLightweightState> {
        self.history_encoder().initial_state()
    }

    pub fn assimilate(
        &self,
        prior: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState> {
        self.history_encoder().assimilate(prior, input, limits)
    }

    pub(crate) fn grouped_inference_supported(&self) -> bool {
        self.history_encoder().grouped_inference_supported()
    }

    pub(crate) fn assimilate_grouped(
        &self,
        priors: &[RecurrentLightweightState],
        inputs: &[HistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Vec<RecurrentLightweightState>> {
        self.history_encoder()
            .assimilate_grouped(priors, inputs, limits)
    }

    /// Evaluate the conditional law of an already assimilated observation.
    pub fn evaluate_decision<'p, 'a>(
        &'p self,
        state: &RecurrentLightweightState,
        observation: &'a StructuredObservation,
        limits: PackLimits,
    ) -> Result<RecurrentPolicyEvaluation<'p, 'a>> {
        let encoded =
            self.history_encoder()
                .decision(state, observation, limits, |rows, candidates| {
                    check_head_budget(
                        &self.config.model,
                        self.config.hidden_width,
                        rows,
                        candidates,
                    )
                })?;
        Ok(self
            .evaluate_heads(encoded)
            .pop()
            .expect("one scalar decision"))
    }

    pub(crate) fn evaluate_decisions_grouped<'p, 'a>(
        &'p self,
        states: &[RecurrentLightweightState],
        observations: &[&'a StructuredObservation],
        limits: PackLimits,
    ) -> Result<Vec<RecurrentPolicyEvaluation<'p, 'a>>> {
        ensure(
            states.len() == observations.len(),
            "grouped structured decision count mismatch",
        )?;
        let history = self.history_encoder();
        history.inference_groups(
            states.len(),
            |lane| HistoryEncoder::decision_layout(&states[lane], observations[lane]),
            |lane| self.evaluate_decision(&states[lane], observations[lane], limits),
            |lanes| {
                let states: Vec<_> = lanes.iter().map(|&lane| &states[lane]).collect();
                let observations: Vec<_> = lanes.iter().map(|&lane| observations[lane]).collect();
                Ok(history
                    .decisions_grouped(&states, &observations, limits, |rows, candidates| {
                        check_head_budget(
                            &self.config.model,
                            self.config.hidden_width,
                            rows,
                            candidates,
                        )
                    })?
                    .map(|encoded| self.evaluate_heads(encoded)))
            },
        )
    }

    /// Reconstruct this policy's causal state from each reset. The learning
    /// boundary controls gradient detachment while retaining every live row.
    pub fn evaluate_histories<'p, 'a>(
        &'p self,
        histories: &[StructuredHistoryInput<'a>],
        limits: PackLimits,
    ) -> Result<Vec<RecurrentPolicyEvaluation<'p, 'a>>> {
        let encoded = self.history_encoder().encode(
            histories,
            limits,
            |_| {},
            |rows, candidates| {
                check_head_budget(
                    &self.config.model,
                    self.config.hidden_width,
                    rows,
                    candidates,
                )
            },
        )?;
        Ok(self.evaluate_heads(encoded))
    }

    fn evaluate_heads<'p, 'a>(
        &'p self,
        encoded: EncodedHistories<'a>,
    ) -> Vec<RecurrentPolicyEvaluation<'p, 'a>> {
        ConditionalHeads {
            config: &self.config.model,
            operations: &self.operations,
            operation_head: &self.operation_head,
            entity_query: &self.entity_query,
            entity_key: &self.entity_key,
            location_query: &self.location_query,
            location_key: &self.location_key,
        }
        .evaluate(
            encoded.states,
            encoded.targets,
            encoded.batch.arguments,
            encoded.observations,
        )
    }
}
impl Checkpoint for RecurrentLightweightPolicy {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("recurrent-lightweight-policy-v1", &self.config)?;
        record.model("policy", self)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-lightweight-policy-v1")?;
        let config: RecurrentLightweightConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "recurrent actor checkpoint configuration mismatch",
        )?;
        *self = restore_model(record, "policy", self)?;
        Ok(())
    }
}
