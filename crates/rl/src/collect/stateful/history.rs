//! Retained public-history identity and behavior-adoption provenance.
use crate::{
    Result,
    core::{
        action::Action,
        anchors::RepresentationId,
        ensure,
        env::Observation,
        reconstruction::{ReconstructionConfig, ReconstructionDecision, ReconstructionMode},
        stateful::{HistoryRewards, PreviousInput},
    },
};
use serde::{Deserialize, Serialize};

/// Actual public history beginning at an absolute observation boundary. A suffix
/// carries its real preceding transition; it never invents an episode reset.
#[derive(Debug, Clone, Copy)]
pub struct RetainedHistory<'a> {
    pub base_timestep: usize,
    pub previous: Option<PreviousInput<'a>>,
    pub observations: &'a [Observation],
    pub actions: &'a [Action],
    pub rewards: HistoryRewards<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdoptionSource {
    FullHistory {
        origin_index: usize,
        observations: usize,
    },
    Anchored {
        representation_id: RepresentationId,
        anchor_identity: [u8; 32],
        config: ReconstructionConfig,
        decision: ReconstructionDecision,
        anchor_generation: u64,
        current_generation: u64,
    },
}

/// A behavior change at an existing post-observation clock. Transitions at this
/// index and later use to_revision until another explicit adoption or real reset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdoptionRecord {
    pub version: u32,
    pub episode_id: u64,
    pub observation_index: usize,
    pub from_revision: u64,
    pub to_revision: u64,
    pub history_identity: [u8; 32],
    pub source: AdoptionSource,
}
impl AdoptionRecord {
    pub(super) fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1 && self.to_revision > self.from_revision,
            "invalid adoption record version or revisions",
        )?;
        match &self.source {
            AdoptionSource::FullHistory {
                origin_index,
                observations,
            } => ensure(
                *origin_index == 0 && self.observation_index.checked_add(1) == Some(*observations),
                "full-history adoption must reconstruct from the genuine reset",
            ),
            AdoptionSource::Anchored {
                config,
                decision,
                anchor_generation,
                current_generation,
                ..
            } => {
                config.validate()?;
                ensure(
                    decision.version == 1
                        && decision.observation_index <= self.observation_index
                        && current_generation.checked_sub(*anchor_generation)
                            == Some(decision.generation_lag)
                        && (decision.approximate
                            || decision.source_model_identity == decision.current_model_identity),
                    "invalid anchored adoption provenance",
                )?;
                match config.mode {
                    ReconstructionMode::Exact => ensure(
                        !decision.approximate,
                        "exact adoption contains approximate provenance",
                    ),
                    ReconstructionMode::Anchored { max_generation_lag } => ensure(
                        max_generation_lag.is_none_or(|maximum| decision.generation_lag <= maximum),
                        "adoption exceeds its declared generation lag",
                    ),
                }
            }
        }
    }
}

pub(super) fn previous_input(
    history: RetainedHistory<'_>,
    timestep: usize,
) -> Option<PreviousInput<'_>> {
    timestep
        .checked_sub(1)
        .map(|previous| {
            history.rewards.input(
                previous,
                &history.actions[previous],
                &history.observations[previous],
            )
        })
        .or(history.previous)
}
pub(super) fn actions_equal_bits(left: &Action, right: &Action) -> bool {
    match (left, right) {
        (Action::Conditional(left), Action::Conditional(right)) => left == right,
        (Action::Discrete(left), Action::Discrete(right)) => left == right,
        (Action::Continuous(left), Action::Continuous(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        }
        _ => false,
    }
}
pub(super) fn public_history_identity(episode_id: u64, history: RetainedHistory<'_>) -> [u8; 32] {
    crate::core::stateful::public_history_identity(
        episode_id,
        history.base_timestep,
        history.previous,
        history.observations,
        history.actions.iter().enumerate().map(|(index, action)| {
            history
                .rewards
                .input(index, action, &history.observations[index])
        }),
    )
}

#[cfg(test)]
mod identity_compatibility {
    use super::*;
    use crate::core::reward::Reward;

    #[test]
    fn public_history_digest_preserves_original_collector_wire() {
        let observations = [
            Observation::dense(vec![0.25, -0.0], Some(vec![true, true])),
            Observation::dense(vec![0.5, 1.0], Some(vec![true, false])),
            Observation::dense(vec![0.75, 0.0], Some(vec![false, true])),
        ];
        let actions = [Action::Discrete(0), Action::Discrete(1)];
        let rewards = [
            Reward::One(0.25),
            Reward::from_primitives(vec![0.5, -0.25]).unwrap(),
        ];
        let history = RetainedHistory {
            base_timestep: 7,
            previous: Some(PreviousInput::from_total(&actions[1], -0.5)),
            observations: &observations,
            actions: &actions,
            rewards: HistoryRewards::Traces(&rewards),
        };
        // Captured from the unchanged collector digest at 2dcde61e before
        // extracting the shared iterator-based implementation.
        assert_eq!(
            public_history_identity(13, history),
            [
                181, 234, 203, 14, 62, 23, 91, 152, 120, 64, 232, 19, 211, 129, 195, 233, 54, 58,
                220, 14, 23, 171, 209, 180, 0, 11, 176, 52, 81, 231, 148, 147
            ]
        );
    }
}
