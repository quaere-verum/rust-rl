use super::{
    RecurrentLightweightConfig, RecurrentLightweightQ, RecurrentLightweightStage,
    RecurrentLightweightState, StructuredHistoryInput, StructuredHistoryValues,
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
        network::{Mlp, Network},
        structured::{PackLimits, restore_model},
    },
};
use burn::tensor::{Device, Tensor};
use rand_chacha::ChaCha8Rng;

impl RecurrentLightweightQ {
    pub fn with_initialization(
        mut self,
        initialization: crate::nn::critics::CriticInitialization,
    ) -> Result<Self> {
        initialization.apply(&mut self.head)?;
        Ok(self)
    }

    pub fn new(
        config: RecurrentLightweightConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let c = &config.model;
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
            head: Mlp::new_on_device(config.joint_width(), &[c.head_width], 1, rng, device)?,
            memory,
            config,
        })
    }

    pub fn config(&self) -> &RecurrentLightweightConfig {
        &self.config
    }

    pub fn model_device(&self) -> Device {
        self.history_encoder().device()
    }

    pub(super) fn history_encoder(&self) -> HistoryEncoder<'_> {
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

    /// Evaluate the complete candidates of an already assimilated observation.
    pub fn evaluate_decision(
        &self,
        state: &RecurrentLightweightState,
        observation: &StructuredObservation,
        limits: PackLimits,
    ) -> Result<Option<Tensor<1>>> {
        let encoded =
            self.history_encoder()
                .decision(state, observation, limits, |_, candidates| {
                    self.check_head_budget(candidates)
                })?;
        Ok(self.candidate_values(&encoded))
    }

    /// Reconstruct each lane from its own reset. Return every decision row in
    /// original candidate order, including disabled rows for caller masking.
    pub fn evaluate_histories(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Vec<StructuredHistoryValues>> {
        self.evaluate_histories_observed(histories, limits, |_| {})
    }

    pub fn evaluate_histories_observed(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        limits: PackLimits,
        mut observe: impl FnMut(RecurrentLightweightStage),
    ) -> Result<Vec<StructuredHistoryValues>> {
        let encoded =
            self.history_encoder()
                .encode(histories, limits, &mut observe, |_, candidates| {
                    self.check_head_budget(candidates)
                })?;
        let values = self.candidate_values(&encoded);
        observe(RecurrentLightweightStage::Candidates);
        let (mut row, mut candidate) = (0, 0);
        let outputs = encoded
            .observations
            .iter()
            .map(|observations| {
                let start = row;
                let decisions = observations
                    .iter()
                    .map(|observation| {
                        row += 1;
                        let count = observation.candidates.len();
                        let start = candidate;
                        candidate += count;
                        (count != 0).then(|| {
                            values
                                .as_ref()
                                .expect("packed candidate rows")
                                .clone()
                                .narrow(0, start, count)
                        })
                    })
                    .collect();
                StructuredHistoryValues {
                    states: encoded.states.clone().narrow(0, start, row - start),
                    decisions,
                }
            })
            .collect();
        observe(RecurrentLightweightStage::Binding);
        Ok(outputs)
    }

    fn check_head_budget(&self, candidates: usize) -> Result<usize> {
        let c = &self.config.model;
        c.check_shape(&[
            candidates.max(1),
            self.config.joint_width().max(c.head_width),
        ])?;
        Ok(0)
    }

    fn candidate_values(&self, encoded: &EncodedHistories<'_>) -> Option<Tensor<1>> {
        let batch = &encoded.batch;
        (batch.candidate_count > 0).then(|| {
            self.head
                .forward(candidate_features(encoded, &self.operations))
                .reshape([batch.candidate_count])
        })
    }
}

pub(super) fn candidate_features(encoded: &EncodedHistories<'_>, operations: &Mlp) -> Tensor<2> {
    let batch = &encoded.batch;
    Tensor::cat(
        vec![
            encoded.states.clone().select(0, batch.owners.clone()),
            operations.forward(batch.operations.clone()),
            encoded.targets.clone(),
            batch.arguments.clone(),
        ],
        1,
    )
}

impl Checkpoint for RecurrentLightweightQ {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("recurrent-lightweight-q-v1", &self.config)?;
        record.model("q", self)?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-lightweight-q-v1")?;
        let config: RecurrentLightweightConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "recurrent lightweight checkpoint configuration mismatch",
        )?;
        *self = restore_model(record, "q", self)?;
        Ok(())
    }
}
