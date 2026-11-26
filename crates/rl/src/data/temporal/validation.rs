//! Structural audits for factual episodes and the store's derived ledgers.

use super::{
    Episode, EpisodeStore, ResourceUsage, TemporalStorage, VERSION, add_usage, legacy_storage,
};
use crate::{
    Error, Result,
    core::{action::ActionSpace, checkpoint::Snapshot, ensure, env::ObservationSpace},
};

impl Episode {
    pub fn validate(&self, width: usize, space: &ActionSpace) -> Result<()> {
        self.validate_in(&ObservationSpace::Dense(width), space)
    }
    pub fn validate_in(&self, observations: &ObservationSpace, space: &ActionSpace) -> Result<()> {
        ensure(
            self.steps.len().checked_add(1) == Some(self.observations.len()),
            "invalid temporal episode lengths",
        )?;
        ensure(
            self.base_timestep.checked_add(self.steps.len()).is_some()
                && self.context_len() <= self.steps.len()
                && (self.base_timestep == 0) == self.previous.is_none(),
            "invalid retained temporal base or previous input",
        )?;
        if let Some(previous) = &self.previous {
            space.check(&previous.action, None)?;
            ensure(
                previous.reward.is_finite(),
                "nonfinite temporal boundary reward",
            )?;
        }
        let mut last = None;
        for set in &self.anchors {
            set.validate()?;
            let boundary = set.observation_index();
            ensure(
                set.episode_id() == self.id
                    && boundary >= self.base_timestep
                    && boundary <= self.end_timestep()
                    && last.is_none_or(|index| boundary > index),
                "invalid temporal anchor chronology",
            )?;
            last = Some(boundary);
        }
        ensure(
            self.base_timestep == 0
                || self
                    .anchors
                    .first()
                    .is_some_and(|set| set.observation_index() == self.base_timestep),
            "retained temporal suffix has no boundary anchor",
        )?;
        self.observations[0].validate_in(observations, space, true)?;
        if self.base_timestep == 0 {
            self.observations[0].validate_event_history(None)?;
        }
        for (i, step) in self.steps.iter().enumerate() {
            ensure(
                !step.ended() || i + 1 == self.steps.len(),
                "temporal episode continues after its end",
            )?;
            self.observations[i + 1].validate_in(observations, space, !step.terminated)?;
            self.observations[i + 1].validate_event_history(Some(&self.observations[i]))?;
            self.observations[i].check_action(space, &step.action)?;
            step.reward.validate()?;
            step.behavior.validate()?;
            if i < self.context_len() {
                ensure(
                    !step.ended()
                        && step.behavior.revision == 0
                        && step.behavior.log_prob.is_none(),
                    "context-only transition carries an end flag or behavior metadata",
                )?;
            }
        }
        Ok(())
    }
}

impl EpisodeStore {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == 3
                || self.version
                    == if legacy_storage(&self.storage) {
                        VERSION
                    } else {
                        2
                    },
            "unsupported temporal store snapshot version",
        )?;
        if let Some(limits) = self.limits() {
            limits.validate(self.capacity)?;
            ensure(
                limits.contains(self.len, self.usage),
                "temporal store exceeds resource limits",
            )?;
            if let TemporalStorage::Segmented(config) = &self.storage {
                ensure(
                    config.segment_steps > 0
                        && config.segment_steps <= self.capacity
                        && config.minimum_context > 0
                        && config.minimum_context <= self.capacity
                        && !config.required_representations.is_empty()
                        && config.required_representations.len()
                            <= crate::core::anchors::MAX_ANCHOR_KEYS
                        && config
                            .required_representations
                            .windows(2)
                            .all(|keys| keys[0] < keys[1]),
                    "invalid segmented temporal configuration",
                )?;
            }
        }
        ensure(
            self.capacity > 0
                && self.len <= self.capacity
                && self.active.len() <= self.capacity
                && self.episodes.len() <= self.len.saturating_add(self.active.len()),
            "invalid temporal store dimensions",
        )?;
        self.observation_space.validate()?;
        self.space.validate()?;
        let mut count = 0usize;
        let mut previous = None;
        for (&key, episode) in &self.episodes {
            ensure(
                key == episode.id
                    && previous.is_none_or(|id| episode.id > id)
                    && self.highest_episode_id.is_some_and(|id| episode.id <= id),
                "invalid temporal episode identity order",
            )?;
            previous = Some(episode.id);
            episode.validate_in(&self.observation_space, &self.space)?;
            ensure(
                self.version >= 3 || episode.learning_start == 0,
                "temporal store record version does not carry a context boundary",
            )?;
            if legacy_storage(&self.storage)
                || matches!(self.storage, TemporalStorage::FullPrefix(_))
            {
                ensure(
                    episode.base_timestep == 0,
                    "full-prefix replay contains a trimmed suffix",
                )?;
            }
            if legacy_storage(&self.storage) {
                ensure(
                    episode.anchors.is_empty(),
                    "legacy replay contains anchor metadata",
                )?;
            } else {
                for anchors in &episode.anchors {
                    self.check_anchors(anchors)?;
                }
            }
            ensure(
                self.index.weight(episode.id) == Some(episode.learning_len()),
                "temporal episode index weight differs",
            )?;
            count = count
                .checked_add(episode.learning_len())
                .ok_or_else(|| Error::State("temporal transition count overflow".into()))?;
            match (episode.ended(), self.active.get(&episode.id)) {
                (true, None) => (),
                (false, Some(active)) => {
                    ensure(
                        !active.ignored && active.next_timestep == episode.end_timestep(),
                        "temporal active tape metadata differs",
                    )?;
                }
                _ => {
                    return Err(Error::State(
                        "temporal episode activity differs from tape".into(),
                    ));
                }
            }
        }
        ensure(count == self.len, "invalid temporal store transition count")?;
        ensure(
            self.index.validate(self.episodes.len(), self.len),
            "invalid temporal weighted index",
        )?;
        if self.limits().is_some() {
            let audited = self
                .episodes
                .values()
                .try_fold(ResourceUsage::default(), |usage, episode| {
                    add_usage(usage, episode.usage()?)
                })?;
            ensure(
                audited == self.usage && self.active.values().all(|active| !active.ignored),
                "temporal resource accounting or active retention differs",
            )?;
        }
        ensure(
            self.latest_terminal
                .is_none_or(|id| self.episode(id).is_some_and(Episode::ended)),
            "invalid latest temporal terminal identity",
        )?;
        for (&id, active) in &self.active {
            ensure(
                self.highest_episode_id.is_some_and(|highest| id <= highest),
                "invalid temporal active identity metadata",
            )?;
            let is_retained = self.episodes.contains_key(&id);
            ensure(
                active.ignored != is_retained,
                "invalid temporal ignored episode topology",
            )?;
        }
        ensure(
            self.highest_episode_id.is_some()
                || (self.episodes.is_empty() && self.active.is_empty()),
            "missing temporal episode identity high-watermark",
        )
    }
}

impl Snapshot for EpisodeStore {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate()
    }
}
