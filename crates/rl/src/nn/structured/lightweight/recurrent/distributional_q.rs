use super::{
    super::LightweightBatch,
    RecurrentLightweightConfig, RecurrentLightweightState, StructuredHistoryInput,
    history::{EncodedHistories, HistoryBackbone},
};
use crate::{
    Error, Result,
    core::{
        action::ConditionalAction,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        stateful::HistoryInput,
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
        structured::{PackLimits, restore_model},
        temporal_q::TemporalQHead,
    },
};
use burn::{
    module::Module,
    tensor::{Device, Int, Tensor, TensorData},
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Independently recurrent candidate critic producing HL-Gauss bin logits.
#[derive(Module, Debug)]
pub struct RecurrentLightweightDistributionalQ {
    backbone: HistoryBackbone,
    pub head: Mlp,
    predictor: Option<Mlp>,
    #[module(skip)]
    config: RecurrentLightweightConfig,
}

/// A factual unroll and its public candidate joins. Querying the head leaves
/// the history fixed and never uses a candidate as a previous action.
pub struct RecurrentDistributionalQEvaluation<'p> {
    critic: &'p RecurrentLightweightDistributionalQ,
    states: Tensor<2>,
    batch: LightweightBatch,
    targets: Tensor<2>,
    candidates: Vec<Vec<HashMap<ConditionalAction, i32>>>,
}

pub struct RecurrentCandidatePrediction {
    pub logits: Tensor<2>,
    pub representation: Tensor<2>,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Architecture {
    config: RecurrentLightweightConfig,
    bins: usize,
    predictor: bool,
}

impl RecurrentLightweightDistributionalQ {
    pub fn parameter_count(
        config: &RecurrentLightweightConfig,
        bins: usize,
        predictor: bool,
    ) -> Result<usize> {
        config.validate()?;
        let width = config.model.head_width;
        let fixed =
            (config.joint_width() + 1) * width + usize::from(predictor) * (width + 1) * width;
        let head = bins
            .checked_mul(width + 1)
            .and_then(|n| n.checked_add(fixed))
            .ok_or_else(|| Error::Shape("recurrent histogram Q parameter size overflow".into()))?;
        ensure(
            bins >= 2 && head <= 16 * 1024 * 1024,
            "invalid recurrent histogram Q bin count or parameter budget",
        )?;
        Ok(config.backbone_parameter_count() + head)
    }

    pub fn new(
        config: RecurrentLightweightConfig,
        bins: usize,
        predictor: bool,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::parameter_count(&config, bins, predictor)?;
        let width = config.model.head_width;
        Ok(Self {
            backbone: HistoryBackbone::new(&config, rng, device)?,
            head: Mlp::new_on_device(config.joint_width(), &[width], bins, rng, device)?,
            predictor: predictor
                .then(|| Mlp::new_on_device(width, &[], width, rng, device))
                .transpose()?,
            config,
        })
    }

    pub fn config(&self) -> &RecurrentLightweightConfig {
        &self.config
    }

    pub fn output_dim(&self) -> usize {
        self.head.output_dim()
    }

    pub fn supports_auxiliary(&self) -> bool {
        self.predictor.is_some()
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

    pub fn evaluate_histories(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        limits: PackLimits,
    ) -> Result<RecurrentDistributionalQEvaluation<'_>> {
        let encoded = self.backbone.view(&self.config).encode(
            histories,
            limits,
            |_| {},
            |_, candidates| self.check_candidate_budget(candidates),
        )?;
        self.bind(encoded, None)
    }

    /// Restrict candidate packing to positions that will be queried. Every
    /// observation and introduced event still participates in the causal unroll.
    pub fn evaluate_histories_at(
        &self,
        histories: &[StructuredHistoryInput<'_>],
        positions: &[(usize, usize)],
        limits: PackLimits,
    ) -> Result<RecurrentDistributionalQEvaluation<'_>> {
        ensure(
            !positions.is_empty()
                && positions.iter().all(|&(lane, time)| {
                    histories
                        .get(lane)
                        .is_some_and(|history| time < history.inputs.len())
                }),
            "recurrent histogram Q positions are empty or out of range",
        )?;
        let positions = positions.iter().copied().collect();
        let encoded = self.backbone.view(&self.config).encode_at(
            histories,
            &positions,
            limits,
            |_, candidates| self.check_candidate_budget(candidates),
        )?;
        self.bind(encoded, Some(&positions))
    }

    pub fn predict_representation(&self, representation: Tensor<2>) -> Result<Tensor<2>> {
        let predictor = self.predictor.as_ref().ok_or_else(|| {
            Error::State("recurrent histogram Q has no auxiliary predictor".into())
        })?;
        let [rows, width] = representation.dims();
        ensure(
            rows > 0 && width == predictor.input_dim(),
            "invalid recurrent histogram Q representation shape",
        )?;
        self.config.model.check_shape(&[rows, width])?;
        Ok(predictor.forward(representation))
    }

    pub(crate) fn validate_query_budget(&self, rows: usize) -> Result<()> {
        ensure(rows > 0, "empty recurrent histogram Q query")?;
        self.config.model.check_shape(&[
            rows,
            self.config
                .joint_width()
                .max(self.config.model.head_width)
                .max(self.output_dim()),
        ])
    }

    fn check_candidate_budget(&self, candidates: usize) -> Result<usize> {
        ensure(
            candidates <= i32::MAX as usize,
            "recurrent histogram Q candidate index overflow",
        )?;
        self.config
            .model
            .check_shape(&[candidates.max(1), self.config.model.target_width()])?;
        Ok(0)
    }

    fn bind(
        &self,
        encoded: EncodedHistories<'_>,
        positions: Option<&HashSet<(usize, usize)>>,
    ) -> Result<RecurrentDistributionalQEvaluation<'_>> {
        let mut offset = 0;
        let candidates = encoded
            .observations
            .iter()
            .enumerate()
            .map(|(lane, observations)| {
                observations
                    .iter()
                    .enumerate()
                    .map(|(time, frame)| {
                        if positions.is_some_and(|positions| !positions.contains(&(lane, time))) {
                            return HashMap::new();
                        }
                        let indices = frame
                            .candidates
                            .iter()
                            .enumerate()
                            .filter(|(_, candidate)| candidate.enabled)
                            .map(|(row, candidate)| (candidate.action, (offset + row) as i32))
                            .collect();
                        offset += frame.candidates.len();
                        indices
                    })
                    .collect()
            })
            .collect();
        ensure(
            offset == encoded.batch.candidate_count,
            "recurrent histogram Q candidate packing differs from public support",
        )?;
        Ok(RecurrentDistributionalQEvaluation {
            critic: self,
            states: encoded.states,
            batch: encoded.batch,
            targets: encoded.targets,
            candidates,
        })
    }

    fn architecture(&self) -> Architecture {
        Architecture {
            config: self.config.clone(),
            bins: self.output_dim(),
            predictor: self.supports_auxiliary(),
        }
    }
}

impl RecurrentDistributionalQEvaluation<'_> {
    /// Query enabled complete actions after assimilation, in caller order.
    pub fn query(
        &self,
        requests: &[(usize, usize, ConditionalAction)],
    ) -> Result<RecurrentCandidatePrediction> {
        let critic = self.critic;
        critic.validate_query_budget(requests.len())?;
        let indices = requests
            .iter()
            .map(|&(lane, time, key)| {
                self.candidates
                    .get(lane)
                    .and_then(|history| history.get(time))
                    .and_then(|keys| keys.get(&key))
                    .copied()
                    .ok_or_else(|| {
                        Error::Action(
                            "histogram Q query is outside the selected legal support".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = Tensor::<1, Int>::from_data(
            TensorData::new(indices, [requests.len()]),
            &self.states.device(),
        );
        let owners = self.batch.owners.clone().select(0, indices.clone());
        let operations = self.batch.operations.clone().select(0, indices.clone());
        let features = Tensor::cat(
            vec![
                self.states.clone().select(0, owners),
                critic.backbone.operations.forward(operations),
                self.targets.clone().select(0, indices.clone()),
                self.batch.arguments.clone().select(0, indices),
            ],
            1,
        );
        let representation = critic.head.representation(features)?;
        let logits = critic
            .head
            .layers
            .last()
            .expect("validated Q head")
            .forward(representation.clone());
        Ok(RecurrentCandidatePrediction {
            logits,
            representation,
        })
    }
}

impl Checkpoint for RecurrentLightweightDistributionalQ {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new(
            "recurrent-lightweight-distributional-q-v1",
            self.architecture(),
        )?;
        record.model("q", self)?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-lightweight-distributional-q-v1")?;
        let architecture: Architecture = serde_json::from_value(record.metadata.clone())?;
        ensure(
            architecture == self.architecture(),
            "recurrent histogram Q checkpoint architecture mismatch",
        )?;
        *self = restore_model(record, "q", self)?;
        Ok(())
    }
}
