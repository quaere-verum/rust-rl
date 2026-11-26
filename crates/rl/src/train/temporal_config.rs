//! Off-policy temporal scheduling has no intrinsic exploration convention.
use crate::{
    Result,
    core::{ensure, temporal::ReturnSpec},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOffPolicyConfig {
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub warmup_transitions: u64,
    pub n_step: usize,
    pub gamma: f32,
    pub updates_per_transition: f64,
}
impl TemporalOffPolicyConfig {
    pub fn returns(&self) -> ReturnSpec {
        ReturnSpec {
            gamma: self.gamma,
            n_step: self.n_step,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.returns().validate()?;
        ensure(
            super::config::valid_replay_updates(
                self.replay_capacity,
                self.batch_size,
                self.updates_per_transition,
            ),
            "invalid temporal off-policy configuration",
        )
    }
}
impl From<&super::config::OffPolicyConfig> for TemporalOffPolicyConfig {
    fn from(config: &super::config::OffPolicyConfig) -> Self {
        Self {
            replay_capacity: config.replay_capacity,
            batch_size: config.batch_size,
            warmup_transitions: config.warmup_transitions,
            n_step: config.n_step,
            gamma: config.gamma,
            updates_per_transition: config.updates_per_transition,
        }
    }
}
