//! Verified search sidecar -> source-independent factual learning row.
use super::{PolicyTarget, PublicDecision, SearchProvenance, SearchTarget, history};
use crate::{
    Error, Result,
    core::{action::ConditionalAction, ensure},
    data::{
        policy_supervision::PolicySupervision,
        temporal::{EpisodeStore, SequenceSample},
    },
};

impl<H: PartialEq, O: PartialEq> SearchTarget<H, ConditionalAction, O> {
    pub fn join_supervision<'a>(
        &'a self,
        tape: [u8; 32],
        store: &EpisodeStore,
        expected: &SearchProvenance,
        sample: &'a SequenceSample,
        identity: impl FnOnce(&PublicDecision<'_>) -> Result<H>,
    ) -> Result<PolicySupervision<'a>> {
        let actual = self.join_conditional(tape, store, expected, identity)?;
        let Some(PolicyTarget::Probabilities(probabilities)) = &self.policy else {
            return Err(Error::State(
                "policy supervision requires improved probabilities".into(),
            ));
        };
        ensure(
            sample.episode_id == self.binding.episode_id && sample.history_start == 0,
            "search supervision sequence origin differs",
        )?;
        let _ = sample.exact_history_inputs()?;
        let step = self.binding.observation_index;
        let row = PolicySupervision::new(sample, step, &self.feasible, probabilities)?;
        let prefix = PublicDecision {
            observations: &sample.observations[..=step],
            steps: &sample.steps[..step],
        };
        ensure(
            history(sample.episode_id, &prefix) == self.binding.history
                && &sample.steps[step] == actual,
            "search supervision sequence differs from verified factual replay",
        )?;
        Ok(row)
    }
}
