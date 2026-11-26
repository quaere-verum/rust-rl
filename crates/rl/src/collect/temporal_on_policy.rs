//! Rich sampling records around the shared stateful lane runtime.
use crate::{
    Result,
    core::{
        actor::ActMode,
        ensure,
        env::{EpisodeStart, Observation},
        recording::{RecordingKey, StatefulRecording},
        stateful::StatefulDecision,
        stateful_on_policy::StatefulOnPolicyActor,
        temporal::{TemporalSink, TemporalTransition},
    },
    data::temporal_on_policy::TemporalRollout,
};
use rand_chacha::ChaCha8Rng;

pub struct OnPolicyRecording<'a> {
    rollout: &'a mut TemporalRollout,
}
impl<'a> OnPolicyRecording<'a> {
    fn decide_batch_impl<A: StatefulOnPolicyActor>(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        execution: (ActMode, bool),
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        let (mode, canonical) = execution;
        ensure(
            keys.len() == states.len()
                && keys.len() == observations.len()
                && keys.len() == rngs.len(),
            "on-policy sampling group shape mismatch",
        )?;
        let samples = if canonical {
            actor.sample_on_policy_canonical_batch(states, observations, mode, rngs)?
        } else {
            actor.sample_on_policy_batch(states, observations, mode, rngs)?
        };
        ensure(
            samples.len() == keys.len(),
            "on-policy sampling output count mismatch",
        )?;
        let mut decisions = Vec::with_capacity(samples.len());
        for (&key, sample) in keys.iter().zip(samples) {
            self.rollout
                .stage_decision(key, actor.revision(), &sample)?;
            decisions.push(sample.decision);
        }
        Ok(decisions)
    }
    fn prepare_successors_impl<A: StatefulOnPolicyActor>(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        terminated: &[bool],
        canonical: bool,
    ) -> Result<()> {
        ensure(
            keys.len() == states.len()
                && keys.len() == observations.len()
                && keys.len() == terminated.len(),
            "on-policy successor group shape mismatch",
        )?;
        // No policy distribution is queried on a final observation. In particular,
        // a true terminal observation may have an empty discrete-action mask.
        let selected: Vec<_> = (0..keys.len()).filter(|&i| !terminated[i]).collect();
        let mut values = vec![None; keys.len()];
        if !selected.is_empty() {
            let live_states: Vec<_> = selected.iter().map(|&i| states[i]).collect();
            let live_obs: Vec<_> = selected.iter().map(|&i| observations[i]).collect();
            let predictions = if canonical {
                actor.state_values_canonical_batch(&live_states, &live_obs)?
            } else {
                actor.state_values_batch(&live_states, &live_obs)?
            };
            ensure(
                predictions.len() == selected.len(),
                "on-policy successor value output count mismatch",
            )?;
            for (&i, prediction) in selected.iter().zip(predictions) {
                values[i] = prediction;
            }
        }
        self.rollout.stage_successors(keys, &values, terminated)
    }
    pub fn new(rollout: &'a mut TemporalRollout) -> Self {
        Self { rollout }
    }
}
impl TemporalSink for OnPolicyRecording<'_> {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()> {
        self.rollout.begin_episode(id, start)
    }
    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        self.rollout.push(transition)
    }
    fn abandon_episode(&mut self, id: u64) -> Result<()> {
        self.rollout.abandon_episode(id)
    }
}
impl<A: StatefulOnPolicyActor> StatefulRecording<A> for OnPolicyRecording<'_> {
    fn decide(
        &mut self,
        key: RecordingKey,
        actor: &A,
        state: &A::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let sample = actor.sample_on_policy(state, observation, mode, rng)?;
        self.rollout
            .stage_decision(key, actor.revision(), &sample)?;
        Ok(sample.decision)
    }
    fn decide_batch(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_batch_impl(keys, actor, states, observations, (mode, false), rngs)
    }
    fn decide_canonical_batch(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_batch_impl(keys, actor, states, observations, (mode, true), rngs)
    }
    fn prepare_successors(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        terminated: &[bool],
    ) -> Result<()> {
        self.prepare_successors_impl(keys, actor, states, observations, terminated, false)
    }
    fn prepare_canonical_successors(
        &mut self,
        keys: &[RecordingKey],
        actor: &A,
        states: &[&A::State],
        observations: &[&Observation],
        terminated: &[bool],
    ) -> Result<()> {
        self.prepare_successors_impl(keys, actor, states, observations, terminated, true)
    }
    fn push_with_state(
        &mut self,
        transition: &TemporalTransition,
        _: &A,
        _: &A::State,
    ) -> Result<()> {
        self.rollout.push(transition)
    }
}
