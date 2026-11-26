use super::{
    action::{Action, ActionSpace},
    env::{Observation, ObservationBatch},
};
use crate::Result;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActMode {
    Sample,
    Deterministic,
}

/// Action-only inference. Collection metadata is a separate capability.
pub trait Actor {
    fn action_space(&self) -> &ActionSpace;
    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()>;
    fn act(
        &self,
        observations: &[Observation],
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<Action>> {
        let mut actions = Vec::new();
        self.act_into(observations.into(), mode, rng, &mut actions)?;
        Ok(actions)
    }
}
#[derive(Debug, Default, Clone)]
pub struct RolloutBehavior {
    pub actions: Vec<Action>,
    pub log_probs: Vec<f32>,
    pub values: Vec<f32>,
    pub latents: Vec<Option<Vec<f32>>>,
}
#[derive(Debug, Default, Clone)]
pub struct EpisodeBehavior {
    pub actions: Vec<Action>,
    pub latents: Vec<Option<Vec<f32>>>,
    pub baseline_values: Option<Vec<f32>>,
}
pub trait RolloutActor: Actor {
    fn sample_rollout_into(
        &self,
        observations: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()>;
    fn values_into(&self, observations: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()>;
}
pub trait EpisodeActor: Actor {
    fn sample_episode_into(
        &self,
        observations: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut EpisodeBehavior,
    ) -> Result<()>;
}
/// Collection metadata for Q-based on-policy learners; no state-value prediction.
#[derive(Debug, Default, Clone)]
pub struct PathwiseBehavior {
    pub actions: Vec<Action>,
    pub log_probs: Vec<f32>,
    pub latents: Vec<Option<Vec<f32>>>,
}
pub trait PathwiseActor: Actor {
    /// Begin a fresh fragment. Stateful learners can invalidate prior collections.
    fn begin_collection(&self) -> Result<()> {
        Ok(())
    }
    /// Changes after optimization or a new collection so old data cannot be reused.
    fn revision(&self) -> u64;
    fn sample_pathwise_into(
        &self,
        observations: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut PathwiseBehavior,
    ) -> Result<()>;
}
pub enum OnPolicyActorRef<'a> {
    Bootstrapped(&'a dyn RolloutActor),
    Episodic(&'a dyn EpisodeActor),
    Pathwise(&'a dyn PathwiseActor),
}
