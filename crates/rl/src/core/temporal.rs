//! Algorithm-independent chronological recording protocol for stateful interaction.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace},
        reward::Reward,
    },
};
use serde::{Deserialize, Serialize};

/// Selects how to execute an exact temporal computation. Historical ancestry
/// and anchor acceptance are controlled separately by reconstruction settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TemporalExecutionMode {
    #[default]
    Reference,
    OptimizedExact,
}

#[cfg(feature = "neural")]
impl TemporalExecutionMode {
    pub(crate) fn is_reference(&self) -> bool {
        *self == Self::Reference
    }

    pub(crate) fn supported(self, optimized_exact: bool) -> bool {
        self.is_reference() || optimized_exact
    }

    pub(crate) fn effective(self, optimized_exact: bool) -> Self {
        if self.supported(optimized_exact) {
            self
        } else {
            Self::Reference
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BehaviorMetadata {
    pub revision: u64,
    pub log_prob: Option<f32>,
}
impl BehaviorMetadata {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.log_prob.is_none_or(f32::is_finite),
            "nonfinite temporal behavior metadata",
        )
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalTransition {
    pub episode_id: u64,
    pub timestep: usize,
    pub observation: Observation,
    pub action: Action,
    pub reward: Reward,
    pub next_observation: Observation,
    pub terminated: bool,
    pub truncated: bool,
    pub behavior: BehaviorMetadata,
}
impl TemporalTransition {
    pub fn ended(&self) -> bool {
        self.terminated || self.truncated
    }
    pub fn validate(&self, width: usize, space: &ActionSpace) -> Result<()> {
        self.validate_in(&ObservationSpace::Dense(width), space)
    }
    pub fn validate_in(&self, observations: &ObservationSpace, space: &ActionSpace) -> Result<()> {
        self.observation.validate_in(observations, space, true)?;
        self.next_observation
            .validate_in(observations, space, !self.terminated)?;
        self.next_observation
            .validate_event_history(Some(&self.observation))?;
        self.observation.check_action(space, &self.action)?;
        self.reward.validate()?;
        self.behavior.validate()
    }
}
pub trait TemporalSink {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()>;
    fn push(&mut self, transition: &TemporalTransition) -> Result<()>;
    fn abandon_episode(&mut self, _id: u64) -> Result<()> {
        Ok(())
    }
}

/// Semantic spaces required by a learner and its saved replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalSpaces {
    pub observations: ObservationSpace,
    pub actions: ActionSpace,
}

impl TemporalSpaces {
    pub fn check(&self, observations: &ObservationSpace, actions: &ActionSpace) -> Result<()> {
        ensure(
            &self.observations == observations && &self.actions == actions,
            "temporal trainer and learner observation/action spaces differ",
        )
    }
}
/// Exact configuration identity for temporal return construction. Equality is
/// bitwise for gamma, including signed zero; there is no numerical tolerance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReturnSpec {
    pub gamma: f32,
    pub n_step: usize,
}
impl ReturnSpec {
    pub fn validate(self) -> crate::Result<()> {
        super::ensure(
            self.gamma.is_finite() && (0.0..=1.0).contains(&self.gamma) && self.n_step > 0,
            "invalid temporal return contract",
        )
    }
    pub fn check(self, other: Self) -> crate::Result<()> {
        self.validate()?;
        other.validate()?;
        super::ensure(
            self.gamma.to_bits() == other.gamma.to_bits() && self.n_step == other.n_step,
            "temporal trainer/learner gamma or n-step mismatch (gamma requires identical f32 bits)",
        )
    }
}
