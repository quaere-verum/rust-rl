//! Recurrent DDQN and MPO over shared exact-prefix replay and complete action keys.
mod actor;
mod critic;
mod mpo;

pub use actor::{
    RecurrentCandidateActor, RecurrentCandidateMemory, RecurrentCandidateModel,
    RecurrentCandidateQActor,
};
pub use mpo::{RecurrentCandidateMpo, RecurrentCandidatePolicyActor, RecurrentMpoGradients};

use super::candidate_ddqn::{CandidateCritic, CandidateDdqnConfig, CandidateExploration};
use crate::{
    Result,
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationSpace,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner, TemporalOffPolicyLearner},
        temporal::ReturnSpec,
    },
    data::temporal::SequenceBatch,
    nn::{structured::RecurrentLightweightQ, target::frozen_copy},
};
use burn::module::Module;
use critic::{RecurrentBackup, RecurrentBatch};
use serde::{Deserialize, Serialize};

/// Both modes use the complete factual prefix in the forward pass.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurrentGradientMode {
    /// Detach before the first learning observation's event or decision token.
    #[default]
    LearningWindow,
    /// Include the prefix in differentiation of losses at the same positions.
    FullHistory,
}

pub struct RecurrentCandidateDdqn {
    critic: CandidateCritic<RecurrentLightweightQ>,
    gradient: RecurrentGradientMode,
}

impl RecurrentCandidateDdqn {
    pub fn new(
        online: RecurrentLightweightQ,
        config: CandidateDdqnConfig,
        gradient: RecurrentGradientMode,
        observations: ObservationSpace,
        actions: ActionSpace,
    ) -> Result<Self> {
        Ok(Self {
            critic: critic::new_critic(online, config, observations, actions)?,
            gradient,
        })
    }

    pub fn online(&self) -> &RecurrentLightweightQ {
        &self.critic.online
    }

    pub fn target(&self) -> &RecurrentLightweightQ {
        &self.critic.target
    }

    pub fn updates(&self) -> u64 {
        self.critic.updates
    }

    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        let replay = RecurrentBatch::new(batch, &self.critic)?;
        let online = self.critic.online_history_values(&replay)?;
        self.critic
            .fit_recurrent(&replay, RecurrentBackup::DoubleQ(&online), self.gradient)
    }
}

impl StatefulLearner for RecurrentCandidateDdqn {
    type Batch = SequenceBatch;

    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        RecurrentCandidateDdqn::update(self, batch)
    }
}

impl TemporalOffPolicyLearner for RecurrentCandidateDdqn {
    fn return_contract(&self) -> ReturnSpec {
        self.critic.config.returns()
    }

    fn replay_spaces(&self) -> Option<(&ObservationSpace, &ActionSpace)> {
        Some((&self.critic.observation_space, &self.critic.action_space))
    }
}

impl BehaviorPublisher for RecurrentCandidateDdqn {
    type Actor = RecurrentCandidateQActor;
    type Settings = CandidateExploration;

    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        ensure(
            !self.critic.poisoned,
            "recurrent candidate DDQN requires checkpoint recovery",
        )?;
        RecurrentCandidateActor::new(
            frozen_copy(&self.critic.online).valid(),
            self.critic.observation_space.clone(),
            self.critic.action_space.clone(),
            *settings,
            revision,
            self.critic.config.pack_limits(),
        )
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    version: u32,
    gradient: RecurrentGradientMode,
    critic: serde_json::Value,
}

impl Checkpoint for RecurrentCandidateDdqn {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.critic.recurrent_record()?;
        record.algorithm = "recurrent-candidate-ddqn-v1".into();
        record.metadata = serde_json::to_value(Meta {
            version: 1,
            gradient: self.gradient,
            critic: record.metadata,
        })?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-candidate-ddqn-v1")?;
        self.critic.validate_recurrent_record(record)?;
        let meta: Meta = serde_json::from_value(record.metadata.clone())?;
        ensure(
            meta.version == 1 && meta.gradient == self.gradient,
            "recurrent DDQN gradient contract differs",
        )?;
        let mut record = record.clone();
        record.algorithm = "structured-ddqn-v1".into();
        record.metadata = meta.critic;
        self.critic = self
            .critic
            .restored_model(&record, self.critic.online.config())?;
        Ok(())
    }
}
