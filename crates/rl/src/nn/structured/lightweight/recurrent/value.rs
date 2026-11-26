use super::{
    RecurrentLightweightConfig, RecurrentLightweightState, StructuredHistoryInput,
    history::HistoryBackbone,
};
use crate::{
    Error, Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        stateful::HistoryInput,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
        structured::{PackLimits, restore_model},
    },
};
use burn::{
    module::Module,
    tensor::{Device, Int, Tensor, TensorData},
};
use rand_chacha::ChaCha8Rng;

/// Independent recurrent state value over factual public history.
#[derive(Module, Debug)]
pub struct RecurrentLightweightValue {
    backbone: HistoryBackbone,
    pub head: Mlp,
    #[module(skip)]
    config: RecurrentLightweightConfig,
}

impl RecurrentLightweightValue {
    pub fn parameter_count(config: &RecurrentLightweightConfig) -> Result<usize> {
        config.validate()?;
        let width = config.model.head_width;
        Ok(config.backbone_parameter_count() + (config.hidden_width + 1) * width + width + 1)
    }

    pub fn new(
        config: RecurrentLightweightConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::parameter_count(&config)?;
        Ok(Self {
            backbone: HistoryBackbone::new(&config, rng, device)?,
            head: Mlp::new_on_device(
                config.hidden_width,
                &[config.model.head_width],
                1,
                rng,
                device,
            )?,
            config,
        })
    }

    pub fn config(&self) -> &RecurrentLightweightConfig {
        &self.config
    }

    pub fn model_device(&self) -> Device {
        self.backbone.view(&self.config).device()
    }

    pub fn initial_state(&self) -> Result<RecurrentLightweightState> {
        self.backbone.view(&self.config).initial_state()
    }

    pub fn assimilate(
        &self,
        prior: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState> {
        self.backbone
            .view(&self.config)
            .assimilate(prior, input, limits)
    }

    pub(crate) fn assimilate_grouped(
        &self,
        priors: &[RecurrentLightweightState],
        inputs: &[HistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Vec<RecurrentLightweightState>> {
        self.backbone
            .view(&self.config)
            .assimilate_grouped(priors, inputs, limits)
    }

    /// Scalar head at an already assimilated observation, including a successor
    /// with no enabled candidates.
    pub fn value_from_state(&self, state: &RecurrentLightweightState) -> Result<Tensor<1>> {
        self.backbone.view(&self.config).validate_state(state)?;
        self.values(state.memory.clone())
    }

    /// One chronological value tensor per history, including its final observation.
    pub fn evaluate_histories(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<Vec<Tensor<1>>> {
        let states = self
            .backbone
            .view(&self.config)
            .encode_states(histories, limits)?;
        let values = self.values(states)?;
        let mut offset = 0;
        Ok(histories
            .iter()
            .map(|history| {
                let rows = history.inputs.len();
                let result = values.clone().narrow(0, offset, rows);
                offset += rows;
                result
            })
            .collect())
    }

    /// Head queries in caller order, after each full factual history is reconstructed.
    pub fn evaluate_selected(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        tokens: &[(usize, usize)],
        limits: PackLimits,
    ) -> Result<Tensor<1>> {
        ensure(!tokens.is_empty(), "empty recurrent value selection")?;
        let mut offsets = Vec::with_capacity(histories.len());
        let mut total = 0usize;
        for history in histories {
            offsets.push(total);
            total = total
                .checked_add(history.inputs.len())
                .ok_or_else(|| Error::Shape("recurrent value history size overflow".into()))?;
        }
        let indices = tokens
            .iter()
            .map(|&(lane, time)| {
                ensure(
                    histories
                        .get(lane)
                        .is_some_and(|history| time < history.inputs.len()),
                    "recurrent value position is outside its factual history",
                )?;
                i32::try_from(offsets[lane] + time)
                    .map_err(|_| Error::Shape("recurrent value position overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        self.config.model.check_shape(&[
            tokens.len(),
            self.config.hidden_width.max(self.config.model.head_width),
        ])?;
        let states = self
            .backbone
            .view(&self.config)
            .encode_states(histories, limits)?;
        let indices =
            Tensor::<1, Int>::from_data(TensorData::new(indices, [tokens.len()]), &states.device());
        self.values(states.select(0, indices))
    }

    fn values(&self, states: Tensor<2>) -> Result<Tensor<1>> {
        let [rows, width] = states.dims();
        ensure(
            rows > 0 && width == self.config.hidden_width,
            "invalid recurrent scalar value state shape",
        )?;
        self.config
            .model
            .check_shape(&[rows, self.config.model.head_width])?;
        Ok(self.head.forward(states).squeeze_dim(1))
    }
}

impl Checkpoint for RecurrentLightweightValue {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("recurrent-lightweight-value-v1", &self.config)?;
        record.model("value", self)?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-lightweight-value-v1")?;
        let config: RecurrentLightweightConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "recurrent scalar value checkpoint configuration mismatch",
        )?;
        *self = restore_model(record, "value", self)?;
        Ok(())
    }
}
