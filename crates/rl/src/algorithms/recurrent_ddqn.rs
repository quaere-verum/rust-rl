//! Recurrent Double DQN facade over the shared recurrent discrete critic.
//! Its public type and legacy record envelope retain their historical identity.
use super::{
    ddqn::DdqnConfig,
    recurrent_critic::{CriticBackup, RecurrentCriticConfig, RecurrentDiscreteCritic},
};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::UpdateReport,
        reconstruction::{
            ReconstructionConfig, ReconstructionDecision, ReplayReconstruction,
            ReplayRepresentations,
        },
        stateful::{BehaviorPublisher, HistorySpec, StatefulLearner, TemporalOffPolicyLearner},
        temporal::{ReturnSpec, TemporalExecutionMode},
    },
    data::temporal::SequenceBatch,
    nn::{reconstruction::GruRepresentation, recurrent_actor::RecurrentActor, temporal::GruQ},
};
use serde::{Deserialize, Serialize};

pub use super::recurrent_critic::{
    ONLINE_REPRESENTATION, RecurrentUpdateMode, TARGET_REPRESENTATION, TargetPoint,
    construct_targets,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentDdqnConfig {
    pub ddqn: DdqnConfig,
    pub gamma: f32,
    pub n_step: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EpsilonGreedySettings {
    pub epsilon: f32,
}
impl Default for RecurrentDdqnConfig {
    fn default() -> Self {
        Self {
            ddqn: DdqnConfig::default(),
            gamma: 0.99,
            n_step: 3,
        }
    }
}
impl RecurrentDdqnConfig {
    pub fn validate(&self) -> Result<()> {
        self.ddqn.optimizer.validate()?;
        ensure(
            self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma)
                && self.n_step > 0
                && self.ddqn.target_interval > 0
                && self.ddqn.target_tau.is_finite()
                && (0.0..=1.0).contains(&self.ddqn.target_tau)
                && self.ddqn.epsilon.is_finite()
                && (0.0..=1.0).contains(&self.ddqn.epsilon),
            "invalid recurrent DDQN configuration",
        )
    }
}

impl From<&RecurrentDdqnConfig> for RecurrentCriticConfig {
    fn from(config: &RecurrentDdqnConfig) -> Self {
        Self {
            optimizer: config.ddqn.optimizer.clone(),
            gamma: config.gamma,
            n_step: config.n_step,
            target_tau: config.ddqn.target_tau,
            target_interval: config.ddqn.target_interval,
        }
    }
}

/// DDQN owns behavior settings; the shared critic owns fitting and target state.
pub struct RecurrentDdqn {
    critic: RecurrentDiscreteCritic,
    config: RecurrentDdqnConfig,
}
impl RecurrentDdqn {
    pub fn new(online: GruQ, spec: HistorySpec, config: RecurrentDdqnConfig) -> Result<Self> {
        config.validate()?;
        let critic = RecurrentDiscreteCritic::new(online, spec, (&config).into())?;
        Ok(Self { critic, config })
    }
    pub fn spec(&self) -> &HistorySpec {
        self.critic.spec()
    }
    pub fn config(&self) -> &RecurrentDdqnConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.critic.updates()
    }
    pub fn update_mode(&self) -> TemporalExecutionMode {
        self.critic.update_mode()
    }
    pub fn set_update_mode(&mut self, mode: TemporalExecutionMode) {
        self.critic.set_update_mode(mode);
    }
    pub fn reconstruction_config(&self) -> ReconstructionConfig {
        self.critic.reconstruction_config()
    }
    pub fn set_reconstruction_config(&mut self, config: ReconstructionConfig) -> Result<()> {
        self.critic.set_reconstruction_config(config)
    }
    pub fn online(&self) -> GruQ {
        self.critic.online()
    }
    pub fn target(&self) -> GruQ {
        self.critic.target()
    }
    pub fn policy(&self, epsilon: f32, revision: u64) -> Result<RecurrentActor> {
        self.critic.q_actor(epsilon, revision)
    }
    pub fn parameter_values(&self) -> Result<Vec<f32>> {
        self.critic.parameter_values()
    }
    pub fn reconstructed_q_values(&self, batch: &SequenceBatch) -> Result<(Vec<f32>, Vec<f32>)> {
        self.critic.reconstructed_q_values(batch)
    }
    pub fn reconstruction_decisions(
        &self,
        batch: &SequenceBatch,
    ) -> Result<Vec<(ReconstructionDecision, ReconstructionDecision)>> {
        self.critic.reconstruction_decisions(batch)
    }
    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        self.critic.fit(batch, CriticBackup::DoubleQOptimality)
    }
}
impl ReplayRepresentations for RecurrentDdqn {
    type Model = GruRepresentation;
    fn replay_representations(
        &self,
    ) -> Result<Vec<(crate::core::anchors::RepresentationId, Self::Model)>> {
        self.critic.replay_representations()
    }
}
impl ReplayReconstruction for RecurrentDdqn {
    fn reconstruction_config(&self) -> ReconstructionConfig {
        self.critic.reconstruction_config()
    }
}
impl StatefulLearner for RecurrentDdqn {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        RecurrentDdqn::update(self, batch)
    }
}
impl BehaviorPublisher for RecurrentDdqn {
    type Actor = RecurrentActor;
    type Settings = EpsilonGreedySettings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<RecurrentActor> {
        self.policy(settings.epsilon, revision)
    }
}
impl TemporalOffPolicyLearner for RecurrentDdqn {
    fn return_contract(&self) -> ReturnSpec {
        self.critic.return_contract()
    }
}

/// Explicit envelope adapter also used by the opt-in PR32 MPO learner import.
/// Model and optimizer payloads, GRU controls and return contracts are unchanged.
pub(crate) fn adapt_legacy_critic_metadata(
    record: &LearnerRecord,
) -> Result<(RecurrentDdqnConfig, serde_json::Value)> {
    record.check("recurrent-ddqn-v1")?;
    decode_legacy_critic_metadata(&record.metadata)
}

pub(super) fn decode_legacy_critic_metadata(
    metadata: &serde_json::Value,
) -> Result<(RecurrentDdqnConfig, serde_json::Value)> {
    #[derive(Deserialize)]
    struct Header {
        version: u32,
        config: RecurrentDdqnConfig,
        #[serde(default)]
        reconstruction: ReconstructionConfig,
    }
    let header: Header = serde_json::from_value(metadata.clone())?;
    header.config.validate()?;
    header.reconstruction.validate()?;
    ensure(
        (header.version == 1 || header.version == 2)
            && ((header.version == 1) == header.reconstruction.is_exact_default()),
        "recurrent checkpoint reconstruction/version mismatch",
    )?;
    let mut metadata = metadata.clone();
    metadata["version"] = serde_json::json!(1);
    metadata["config"] = serde_json::to_value(RecurrentCriticConfig::from(&header.config))?;
    Ok((header.config, metadata))
}
impl Checkpoint for RecurrentDdqn {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.critic.checkpoint()?;
        record.algorithm = "recurrent-ddqn-v1".into();
        record.metadata["version"] =
            serde_json::json!(if self.reconstruction_config().is_exact_default() {
                1
            } else {
                2
            });
        record.metadata["config"] = serde_json::to_value(&self.config)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        let (config, adapted) = adapt_legacy_critic_metadata(record)?;
        self.critic.restore_metadata(record, adapted)?;
        self.config = config;
        Ok(())
    }
    fn validate_anchor_payloads(
        &self,
        record: &LearnerRecord,
        anchors: &[&crate::core::anchors::StateAnchor],
    ) -> Result<()> {
        if anchors.is_empty() {
            return Ok(());
        }
        let (_, adapted) = adapt_legacy_critic_metadata(record)?;
        self.critic
            .validate_anchor_metadata(record, adapted, anchors)
    }
}

#[path = "recurrent_diagnostics.rs"]
mod diagnostics;
pub use diagnostics::RecurrentDiagnostics;

#[cfg(test)]
use super::recurrent_critic::{QLayout, construct_targets_layout, parameter_values};
#[cfg(test)]
use crate::nn::{
    temporal_work::{prepare_exact_window, warm_prefix},
    to_vec,
};
#[cfg(test)]
use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::{Device, Tensor, TensorData},
};
#[cfg(test)]
#[path = "recurrent_ddqn_tests.rs"]
mod tests;
