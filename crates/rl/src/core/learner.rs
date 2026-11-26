use super::actor::Actor;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct UpdateReport {
    pub updates: u64,
    pub metrics: BTreeMap<String, f64>,
    pub early_stopped: bool,
}
impl UpdateReport {
    pub fn metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}
pub trait Learner {
    type Batch;
    fn actor(&self) -> &dyn Actor;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport>;
}

/// Trainer capability independent of a concrete replay representation.
pub trait OffPolicyLearner: Learner {
    fn set_exploration(&mut self, scale: f32) -> Result<()>;
    /// Optional cross-component contract, checked before any replay update.
    fn replay_requirements(&self) -> Option<OffPolicyRequirements> {
        None
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OffPolicyRequirements {
    pub returns: super::temporal::ReturnSpec,
    pub exploration: ExplorationSource,
}
impl PartialEq for OffPolicyRequirements {
    fn eq(&self, other: &Self) -> bool {
        self.returns.gamma.to_bits() == other.returns.gamma.to_bits()
            && self.returns.n_step == other.returns.n_step
            && self.exploration == other.exploration
    }
}
impl Eq for OffPolicyRequirements {}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExplorationSource {
    /// Trainer controls an additional exploration scale.
    External,
    /// The stochastic policy supplies exploration; schedule must be identically zero.
    Policy,
}
pub trait OnPolicyLearner: Learner {
    fn collection_actor(&self) -> super::actor::OnPolicyActorRef<'_>;
    /// Whether the learner normalizes advantages inside its own updates.
    fn normalizes_advantages_in_update(&self) -> bool {
        false
    }
}
