//! Recording adapters for stateful collection.

use crate::core::recording::{RecordingKey, StatefulRecording};

use crate::{
    Result,
    core::{
        actor::ActMode,
        env::{EpisodeStart, Observation},
        stateful::{StatefulActor, StatefulDecision},
        temporal::{TemporalSink, TemporalTransition},
    },
};
use rand_chacha::ChaCha8Rng;

pub(crate) struct BasicRecording<'a, S>(pub &'a mut S);
impl<S: TemporalSink> TemporalSink for BasicRecording<'_, S> {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()> {
        self.0.begin_episode(id, start)
    }
    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        self.0.push(transition)
    }
    fn abandon_episode(&mut self, id: u64) -> Result<()> {
        self.0.abandon_episode(id)
    }
}
impl<A: StatefulActor, S: TemporalSink> StatefulRecording<A> for BasicRecording<'_, S> {
    const NEEDS_KEYS: bool = false;
    fn decide(
        &mut self,
        _: RecordingKey,
        actor: &A,
        state: &A::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        actor.decide(state, observation, mode, rng)
    }
    fn decide_canonical_batch(
        &mut self,
        _: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        actor.decide_canonical_batch(states, observations, mode, rngs)
    }
    fn decide_batch(
        &mut self,
        _: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        actor.decide_batch(states, observations, mode, rngs)
    }
    fn push_with_state(
        &mut self,
        transition: &TemporalTransition,
        _: &A,
        _: &A::State,
    ) -> Result<()> {
        self.0.push(transition)
    }
}
