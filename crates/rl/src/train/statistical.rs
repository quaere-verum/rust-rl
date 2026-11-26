//! Complete-run checkpoint for a statistical actor and its factual collector.

use crate::{
    Error, Result,
    approx::{FeatureMap, StatisticalActor, regression::Predictor},
    collect::{stateful::StatefulCollector, vector_env::VectorEnv},
    core::{
        checkpoint::{Checkpoint, LearnerRecord, Snapshot},
        ensure,
        env::PublicHistory,
    },
    data::temporal::EpisodeStore,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Save an ordinary stateful collector alongside its authoritative episode
/// tape through the standard run checkpoint API. In addition to the collector
/// audit, bind each actor's causal state to the lane clock and factual prefix.
/// Saving an active lane requires its genuine reset prefix to be retained.
/// An evicted closed lane resumes at its next genuine reset.
#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(
        serialize = "V: Serialize, F: Serialize, M: Serialize",
        deserialize = "V: DeserializeOwned, F: DeserializeOwned, M: DeserializeOwned"
    )
)]
pub struct StatisticalRunSnapshot<V, F: FeatureMap, M: Predictor> {
    pub collector: StatefulCollector<V, StatisticalActor<F, M>>,
    pub episodes: EpisodeStore,
}

impl<V, F, M> Snapshot for StatisticalRunSnapshot<V, F, M>
where
    V: VectorEnv + Snapshot,
    F: FeatureMap + Clone + Serialize + DeserializeOwned,
    M: Predictor + Serialize + DeserializeOwned,
{
    fn validate_snapshot(&self) -> Result<()> {
        self.collector.validate_snapshot()?;
        self.validate_factual_binding()
    }

    fn validate_learner_record(&self, record: &LearnerRecord) -> Result<()> {
        let latest = self.collector.latest_actor().checkpoint()?;
        ensure(
            serde_json::to_value(&latest)? == serde_json::to_value(record)?,
            "statistical run checkpoint does not match the latest actor publication",
        )
    }
}

impl<V, F, M> StatisticalRunSnapshot<V, F, M>
where
    V: VectorEnv,
    F: FeatureMap + Clone,
    M: Predictor,
{
    fn validate_factual_binding(&self) -> Result<()> {
        self.episodes.validate()?;
        ensure(
            &self.collector.observation_space() == self.episodes.observation_space()
                && self.collector.action_space() == self.episodes.action_space(),
            "statistical run collector and episode tape spaces differ",
        )?;
        let mut active = 0;
        let mut highest_begun = None;
        let mut unbegun = false;
        for lane in 0..self.collector.len() {
            let state = self
                .collector
                .lane_state(lane)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::State("statistical run lane lacks its public history".into())
                })?;
            let clock = self.collector.lane_timestep(lane).expect("live lane");
            let episode_id = self.collector.lane_episode(lane).expect("live lane");
            ensure(
                state.timestep() == clock,
                "statistical run history does not match the lane clock",
            )?;
            let ended = self.collector.lane_needs_reset(lane).expect("live lane");
            if !self.collector.lane_begun(lane).expect("live lane") {
                unbegun = true;
                ensure(
                    !ended
                        && self
                            .episodes
                            .highest_episode_id()
                            .is_none_or(|highest| highest < episode_id),
                    "unbegun statistical lane already has a factual episode identity",
                )?;
                ensure(
                    self.collector
                        .lane_pending_context(lane)
                        .map_or(clock == 0, |resume| same_history(state, &resume.history)),
                    "statistical run differs from its pending public resume context",
                )?;
                continue;
            }
            highest_begun =
                Some(highest_begun.map_or(episode_id, |previous: u64| previous.max(episode_id)));
            active += usize::from(!ended);
            let Some(episode) = self.episodes.episode(episode_id) else {
                ensure(
                    ended
                        && clock > 0
                        && self.episodes.active_timestep(episode_id).is_none()
                        && self
                            .episodes
                            .highest_episode_id()
                            .is_some_and(|highest| episode_id <= highest),
                    "statistical run is missing its lane's factual prefix",
                )?;
                continue;
            };
            let learning_start = self.collector.lane_learning_start(lane).expect("live lane");
            ensure(
                episode.base_timestep == 0
                    && episode.learning_start == learning_start
                    && u64::try_from(episode.steps.len()).ok() == Some(clock)
                    && episode.ended() == ended
                    && self.episodes.active_timestep(episode_id)
                        == (!ended).then_some(episode.steps.len()),
                "statistical run lane differs from its stored factual episode",
            )?;
            for (index, (frame, observation)) in
                state.inputs().iter().zip(&episode.observations).enumerate()
            {
                ensure(
                    frame.observation.matches(observation),
                    "statistical run observation differs from its factual prefix",
                )?;
                if let Some(step) = index.checked_sub(1) {
                    let stored = &episode.steps[step];
                    let previous = frame.previous.expect("causal noninitial frame");
                    ensure(
                        previous.action == &stored.action
                            && previous
                                .factual
                                .is_some_and(|input| input.reward.matches(&stored.reward)),
                        "statistical run action or reward trace differs from its factual prefix",
                    )?;
                    if step >= learning_start {
                        ensure(
                            self.collector.lane_behavior_revision_at(lane, step)
                                == Some(stored.behavior.revision),
                            "statistical run behavior revision differs from its factual prefix",
                        )?;
                    }
                }
            }
        }
        ensure(
            self.episodes.active_count() == active
                && (unbegun || self.episodes.highest_episode_id() == highest_begun),
            "statistical run and factual episode lifecycle differ",
        )
    }
}

fn same_history(left: &PublicHistory, right: &PublicHistory) -> bool {
    left.timestep() == right.timestep()
        && left.inputs().into_iter().zip(right.inputs()).all(|(a, b)| {
            a.observation.matches(b.observation)
                && match (a.previous, b.previous) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        a.action == b.action
                            && a.factual
                                .zip(b.factual)
                                .is_some_and(|(a, b)| a.reward.matches(b.reward))
                    }
                    _ => false,
                }
        })
}
