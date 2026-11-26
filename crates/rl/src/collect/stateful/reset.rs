//! Fresh and restored lane initialization, preserving each actor's random stream.
use super::InferenceMode;
use crate::{
    Result,
    collect::driver::ResumeContexts,
    core::{
        ensure,
        env::{Observation, ResumeContext},
        stateful::{HistoryInput, PreviousStep, StatefulActor},
    },
};

pub(super) fn reset_states<A: StatefulActor>(
    actor: &A,
    inference_mode: InferenceMode,
    observations: &[Observation],
    lanes: &[usize],
    contexts: &ResumeContexts,
) -> Result<Vec<A::State>> {
    if contexts.len() == lanes.len() {
        return contexts
            .iter()
            .map(|(_, context)| actor.reconstruct_history(&context.history.inputs()))
            .collect();
    }
    let input = |lane| HistoryInput {
        observation: &observations[lane],
        previous: None,
    };
    let context_at = |lane| {
        contexts
            .binary_search_by_key(&lane, |(index, _)| *index)
            .ok()
            .map(|index| &contexts[index].1)
    };
    let fresh_count = lanes.len() - contexts.len();
    let mut fresh_states = if inference_mode != InferenceMode::Scalar
        && fresh_count > 1
        && actor.batch_compatible(actor)
    {
        let priors = (0..fresh_count)
            .map(|_| actor.initial_state())
            .collect::<Result<Vec<_>>>()?;
        let priors: Vec<_> = priors.iter().collect();
        let inputs: Vec<_> = lanes
            .iter()
            .copied()
            .filter(|&lane| context_at(lane).is_none())
            .map(input)
            .collect();
        let states = if inference_mode == InferenceMode::CanonicalBatched {
            actor.assimilate_canonical_batch(&priors, &inputs)?
        } else {
            actor.assimilate_batch(&priors, &inputs)?
        };
        ensure(
            states.len() == fresh_count,
            "reset stateful batch count mismatch",
        )?;
        if contexts.is_empty() {
            return Ok(states);
        }
        Some(states.into_iter())
    } else {
        None
    };
    lanes
        .iter()
        .map(|&lane| {
            if let Some(context) = context_at(lane) {
                actor.reconstruct_history(&context.history.inputs())
            } else if let Some(states) = &mut fresh_states {
                Ok(states.next().expect("one state per fresh reset"))
            } else {
                actor.assimilate(&actor.initial_state()?, input(lane))
            }
        })
        .collect()
}

pub(super) fn boundary_previous(context: Option<&ResumeContext>) -> Option<PreviousStep> {
    context?.history.previous_input().map(|input| PreviousStep {
        action: input.action.clone(),
        reward: input.reward,
    })
}
