//! Opt-in recording around the existing stateful interaction clock.
//!
//! A recorder may stage immutable sampling details under an episode and decision
//! key. The collector owns the environment, lane RNG, and candidate actor memory.
use crate::{
    Result,
    core::{
        actor::ActMode,
        ensure,
        env::Observation,
        stateful::{StatefulActor, StatefulDecision},
        temporal::{TemporalSink, TemporalTransition},
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Caller-assigned context; creating or using this value performs no hashing,
/// counter updates, checkpoint I/O or training-loop instrumentation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeIdentity {
    pub run_id: String,
    pub learner_update_revision: u64,
    pub model_role: String,
    pub checkpoint_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordingKey {
    pub episode_id: u64,
    pub timestep: usize,
}

/// Extra recording is optional and independent of the actor's memory type.
pub trait StatefulRecording<A: StatefulActor>: TemporalSink {
    /// The basic strategy needs no temporary key arrays in batched inference.
    const NEEDS_KEYS: bool = true;
    /// Scalar-safe fallback for the versioned canonical execution capability.
    fn decide_canonical_batch(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            keys.len() == states.len()
                && keys.len() == observations.len()
                && keys.len() == rngs.len(),
            "canonical recording count mismatch",
        )?;
        keys.iter()
            .zip(states)
            .zip(observations)
            .zip(rngs)
            .map(|(((&key, state), observation), rng)| {
                self.decide(key, actor, state, observation, mode, rng)
            })
            .collect()
    }
    fn prepare_canonical_successors(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        terminated: &[bool],
    ) -> Result<()> {
        ensure(
            keys.len() == states.len()
                && keys.len() == observations.len()
                && keys.len() == terminated.len(),
            "canonical successor count mismatch",
        )?;
        for i in 0..keys.len() {
            self.prepare_successors(
                &keys[i..i + 1],
                actor,
                &states[i..i + 1],
                &observations[i..i + 1],
                &terminated[i..i + 1],
            )?;
        }
        Ok(())
    }
    fn decide(
        &mut self,
        key: RecordingKey,
        actor: &A,
        state: &A::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision>;

    fn decide_batch(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            keys.len() == states.len()
                && keys.len() == observations.len()
                && keys.len() == rngs.len(),
            "recorded decision group shape mismatch",
        )?;
        keys.iter()
            .zip(states)
            .zip(observations)
            .zip(rngs)
            .map(|(((&key, state), observation), rng)| {
                self.decide(key, actor, state, observation, mode, rng)
            })
            .collect()
    }

    /// All candidate memories are already valid. A detailed recorder can batch
    /// read-only successor value evaluation before publishing any transition.
    /// Never sample another action here; terminal observations may have no actions.
    fn prepare_successors(
        &mut self,
        _keys: &[RecordingKey],
        _actor: &A,
        _states: &[&A::State],
        _observations: &[&Observation],
        _terminated: &[bool],
    ) -> Result<()> {
        Ok(())
    }

    fn push_with_state(
        &mut self,
        transition: &TemporalTransition,
        actor: &A,
        successor_post_state: &A::State,
    ) -> Result<()>;
}
