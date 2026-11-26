pub use crate::nn::structured::RecurrentLightweightMemory as RecurrentCandidateMemory;
use crate::{
    Error, Result,
    algorithms::{
        candidate_ddqn::{CandidateExploration, candidate_frame, epsilon_decision},
        candidate_mpo::CandidateMpoBehavior,
    },
    core::{
        action::{ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        structured::StructuredObservation,
    },
    nn::{
        optimization::parameter_norm,
        structured::{
            PackLimits, RecurrentLightweightConfig, RecurrentLightweightPolicy,
            RecurrentLightweightQ, RecurrentLightweightState,
        },
        to_vec,
    },
};
use burn::{module::Module, tensor::Device};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use std::sync::Arc;

/// Inference and publication contract for recurrent candidate models.
pub trait RecurrentCandidateModel: Module + Checkpoint {
    type Settings: Copy + Serialize + DeserializeOwned;

    fn config(&self) -> &RecurrentLightweightConfig;
    fn device(&self) -> Device;
    fn build(config: RecurrentLightweightConfig, rng: &mut ChaCha8Rng) -> Result<Self>;
    fn validate_settings(settings: Self::Settings) -> Result<()>;
    fn advance(
        &self,
        state: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState>;
    fn select(
        &self,
        state: &RecurrentLightweightState,
        observation: &StructuredObservation,
        settings: Self::Settings,
        limits: PackLimits,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision>;
}

impl RecurrentCandidateModel for RecurrentLightweightQ {
    type Settings = CandidateExploration;

    fn config(&self) -> &RecurrentLightweightConfig {
        self.config()
    }
    fn device(&self) -> Device {
        self.model_device()
    }
    fn build(config: RecurrentLightweightConfig, rng: &mut ChaCha8Rng) -> Result<Self> {
        Self::new(config, rng, &Device::flex())
    }
    fn validate_settings(settings: Self::Settings) -> Result<()> {
        settings.validate()
    }

    fn advance(
        &self,
        state: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState> {
        self.assimilate(state, input, limits)
    }

    fn select(
        &self,
        state: &RecurrentLightweightState,
        observation: &StructuredObservation,
        settings: Self::Settings,
        limits: PackLimits,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let values = self
            .evaluate_decision(state, observation, limits)?
            .ok_or_else(|| {
                Error::Action("recurrent behavior requires a live candidate row".into())
            })?;
        epsilon_decision(
            &observation.candidates,
            &to_vec(values)?,
            settings,
            mode,
            rng,
        )
    }
}

pub type RecurrentCandidateQActor =
    RecurrentCandidateActor<RecurrentLightweightQ, CandidateExploration>;

/// Published immutable parameters. Each collector lane owns its own memory.
#[derive(Clone)]
pub struct RecurrentCandidateActor<M, S> {
    model: Arc<M>,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: S,
    revision: u64,
    limits: PackLimits,
}

impl<M, S> RecurrentCandidateActor<M, S> {
    pub fn model(&self) -> &M {
        &self.model
    }
}

impl RecurrentCandidateQActor {
    /// Read the frozen Q scores after assimilating the actual public history.
    pub fn action_values(
        &self,
        state: &RecurrentCandidateMemory,
        observation: &Observation,
    ) -> Result<Vec<(ConditionalAction, f32)>> {
        self.validate_state(state)?;
        let frame = candidate_frame(observation)?;
        let values = self
            .model
            .evaluate_decision(&state.on_device(&self.model.device()), frame, self.limits)?
            .ok_or_else(|| Error::Action("recurrent Q requires live candidates".into()))?;
        Ok(frame
            .candidates
            .iter()
            .zip(to_vec(values)?)
            .filter(|(candidate, _)| candidate.enabled)
            .map(|(candidate, value)| (candidate.action, value))
            .collect())
    }
}

impl RecurrentCandidateActor<RecurrentLightweightPolicy, CandidateMpoBehavior> {
    /// Complete learned policy before any exploration mixture.
    pub fn action_probabilities(
        &self,
        state: &RecurrentCandidateMemory,
        observation: &Observation,
    ) -> Result<Vec<(ConditionalAction, f64)>> {
        self.validate_state(state)?;
        let frame = candidate_frame(observation)?;
        let probabilities = self
            .model
            .evaluate_decision(&state.on_device(&self.model.device()), frame, self.limits)?
            .complete_probabilities_f64(0)?;
        Ok(frame
            .candidates
            .iter()
            .zip(probabilities)
            .filter(|(candidate, _)| candidate.enabled)
            .map(|(candidate, probability)| (candidate.action, probability))
            .collect())
    }
}

impl<M: RecurrentCandidateModel> RecurrentCandidateActor<M, M::Settings> {
    pub(super) fn new(
        model: M,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        settings: M::Settings,
        revision: u64,
        limits: PackLimits,
    ) -> Result<Self> {
        let actor = Self {
            model: Arc::new(model),
            observation_space,
            action_space,
            settings,
            revision,
            limits,
        };
        actor.validate_contract()?;
        Ok(actor)
    }

    fn validate_contract(&self) -> Result<()> {
        M::validate_settings(self.settings)?;
        ensure(
            self.limits.max_elements > 0,
            "recurrent behavior pack budget is zero",
        )?;
        self.model
            .config()
            .model
            .input
            .validate_spaces(&self.observation_space, &self.action_space)
    }
}

impl<M: RecurrentCandidateModel> StatefulActor for RecurrentCandidateActor<M, M::Settings> {
    type State = RecurrentCandidateMemory;

    fn observation_space(&self) -> ObservationSpace {
        self.observation_space.clone()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }
    fn revision(&self) -> u64 {
        self.revision
    }

    fn initial_state(&self) -> Result<Self::State> {
        Ok(RecurrentCandidateMemory::zero(
            self.model.config().hidden_width,
        ))
    }

    fn validate_state(&self, state: &Self::State) -> Result<()> {
        state.validate(self.model.config().hidden_width)
    }

    fn validate_state_at(
        &self,
        state: &Self::State,
        observation: &Observation,
        preceding_observation: Option<&Observation>,
    ) -> Result<()> {
        state.validate_at(
            self.model.config().hidden_width,
            candidate_frame(observation)?,
            preceding_observation.map(candidate_frame).transpose()?,
        )
    }

    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        let device = self.model.device();
        let next = self
            .model
            .advance(&prior.on_device(&device), input, self.limits)?;
        RecurrentCandidateMemory::from_state(next)
    }

    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        self.model.select(
            &state.on_device(&self.model.device()),
            candidate_frame(observation)?,
            self.settings,
            self.limits,
            mode,
            rng,
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord<S> {
    version: u32,
    config: RecurrentLightweightConfig,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: S,
    revision: u64,
    max_pack_elements: usize,
    model: LearnerRecord,
}

impl<M: RecurrentCandidateModel> Serialize for RecurrentCandidateActor<M, M::Settings> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        ActorRecord {
            version: 1,
            config: self.model.config().clone(),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
            settings: self.settings,
            revision: self.revision,
            max_pack_elements: self.limits.max_elements,
            model: self.model.checkpoint().map_err(serde::ser::Error::custom)?,
        }
        .serialize(serializer)
    }
}

impl<M: RecurrentCandidateModel> RecurrentCandidateActor<M, M::Settings> {
    fn from_record(record: ActorRecord<M::Settings>) -> Result<Self> {
        ensure(
            record.version == 1,
            "unsupported recurrent candidate actor snapshot",
        )?;
        let mut model = M::build(record.config, &mut ChaCha8Rng::seed_from_u64(0))?;
        model.restore(&record.model)?;
        let actor = Self::new(
            model.valid().freeze(),
            record.observation_space,
            record.action_space,
            record.settings,
            record.revision,
            PackLimits {
                max_elements: record.max_pack_elements,
            },
        )?;
        actor.validate_snapshot()?;
        Ok(actor)
    }
}

impl<'de, M: RecurrentCandidateModel> Deserialize<'de> for RecurrentCandidateActor<M, M::Settings> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(ActorRecord::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl<M: RecurrentCandidateModel> Snapshot for RecurrentCandidateActor<M, M::Settings> {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_contract()?;
        parameter_norm(&*self.model)?;
        Ok(())
    }
}
