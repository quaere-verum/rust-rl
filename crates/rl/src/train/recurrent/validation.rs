//! Audits the trainer, factual tape, and reconstruction streams as one run.

use super::{
    BehaviorSchedule, MAX_UPDATES_PER_STEP, ReconstructionConfig, SequenceConfig,
    TemporalOffPolicyConfig, TemporalTrainer, VERSION, temporal_recording,
};
use crate::{
    Error, Result,
    collect::vector_env::VectorEnv,
    core::{
        checkpoint::Snapshot, ensure, reconstruction::ReplayReconstruction, stateful::StatefulActor,
    },
};
use serde::{Serialize, de::DeserializeOwned};

impl<V: VectorEnv, A: StatefulActor + Clone, S: BehaviorSchedule> TemporalTrainer<V, A, S> {
    fn expected_version(&self) -> u32 {
        if self.replay_config.is_some() {
            3
        } else {
            VERSION
        }
    }

    fn audit_recording_structure(&self) -> Result<()> {
        let Some(config) = &self.replay_config else {
            return ensure(
                self.replay_streams.is_empty()
                    && matches!(
                        self.store.storage(),
                        crate::data::temporal::TemporalStorage::ExactEpisodes
                    ),
                "legacy trainer contains hidden reconstruction storage or streams",
            );
        };
        ensure(
            self.collector.inference_mode()
                != crate::collect::stateful::InferenceMode::CanonicalBatched,
            "canonical inference snapshot contains anchors without numerical execution provenance",
        )?;
        config.validate(
            self.config.replay_capacity,
            self.collector.len(),
            &self.sequence_config,
        )?;
        ensure(
            self.store.storage() == &config.storage
                && self.replay_streams.len() <= self.collector.len(),
            "temporal recording storage or stream capacity differs",
        )?;
        if !config.is_segmented() {
            return ensure(
                self.replay_streams.is_empty(),
                "raw full-prefix recording contains unused streams",
            );
        }
        let mut active = 0;
        for lane in 0..self.collector.len() {
            if self.collector.lane_begun(lane) != Some(true)
                || self.collector.lane_needs_reset(lane) == Some(true)
            {
                continue;
            }
            active += 1;
            let id = self.collector.lane_episode(lane).expect("validated lane");
            let clock = self.collector.lane_timestep(lane).expect("validated lane");
            let set = self.replay_streams.get(&id).ok_or_else(|| {
                Error::State("live temporal lane lacks reconstruction stream".into())
            })?;
            temporal_recording::validate_stream_set(config, set)?;
            ensure(
                set.episode_id() == id
                    && u64::try_from(set.observation_index()).ok() == clock.checked_add(1),
                "temporal reconstruction stream clock differs from lane",
            )?;
        }
        ensure(
            active == self.replay_streams.len(),
            "orphaned temporal reconstruction stream",
        )
    }

    pub fn validate_reconstruction<L: ReplayReconstruction>(&self, learner: &L) -> Result<()> {
        let config = self.replay_config.as_ref().ok_or_else(|| {
            Error::State("temporal reconstruction recording is not configured".into())
        })?;
        ensure(
            config.reconstruction == learner.reconstruction_config(),
            "learner and recording reconstruction contracts differ",
        )?;
        self.audit_recording_structure()?;
        if config.is_segmented() {
            let models = learner.replay_representations()?;
            temporal_recording::audit_streams(
                config.reconstruction,
                &models,
                &self.replay_streams,
            )?;
            temporal_recording::validate_source_keys(config, &models)?;
        }
        Ok(())
    }

    /// Private fields and validated restoration establish topology once. Every
    /// tick still checks clocks, dimensions, poison state and current ownership.
    pub fn validate_local(&self) -> Result<()> {
        if !self.audited.get() {
            return self.validate_state();
        }
        ensure(
            self.version == self.expected_version() && !self.poisoned,
            "unsupported or poisoned temporal trainer",
        )?;
        self.collector.validate_local()?;
        ensure(
            self.behavior_revision == self.collector.latest_revision()
                && self.store.capacity() == self.config.replay_capacity,
            "temporal local revision/capacity mismatch",
        )?;
        let lanes = self.collector.len() as u64;
        ensure(
            self.counters.checked_physical_steps().is_some()
                && self.counters.vector_steps.checked_mul(lanes) == Some(self.counters.env_steps)
                && self.counters.episodes <= self.counters.env_steps,
            "temporal local counters disagree",
        )?;
        for lane in 0..self.collector.len() {
            if self.collector.lane_begun(lane) != Some(true) {
                continue;
            }
            let id = self.collector.lane_episode(lane).expect("validated lane");
            let clock = self.collector.lane_timestep(lane).expect("validated lane");
            if let Some(episode) = self.store.episode(id) {
                ensure(
                    episode.end_timestep() as u64 == clock
                        && episode
                            .observations
                            .last()
                            .is_some_and(|last| last.matches(&self.collector.observations()[lane])),
                    "local temporal tape/current observation differs",
                )?;
            } else if let Some(position) = self.store.active_timestep(id) {
                ensure(
                    position as u64 == clock
                        && self.collector.lane_needs_reset(lane) == Some(false),
                    "local ignored temporal clock differs",
                )?;
            } else {
                ensure(
                    self.collector.lane_needs_reset(lane) == Some(true),
                    "local temporal live tape is missing",
                )?;
            }
        }
        self.audit_recording_structure()?;
        Ok(())
    }

    pub fn validate_state(&self) -> Result<()> {
        self.audited.set(false);
        ensure(
            self.version == self.expected_version() && !self.poisoned,
            "unsupported or poisoned temporal trainer snapshot",
        )?;
        self.collector.validate_state()?;
        self.store.validate()?;
        self.sampler.validate()?;
        self.behavior.validate()?;
        self.audit_recording_structure()?;
        validate_config(&self.config, &self.sequence_config, self.collector.len())?;
        ensure(
            self.store.capacity() == self.config.replay_capacity
                && *self.store.observation_space() == self.collector.observation_space()
                && self.store.action_space() == self.collector.action_space()
                && self.behavior_revision == self.collector.latest_revision(),
            "temporal trainer component schemas or revisions differ",
        )?;
        let lanes = u64::try_from(self.collector.len())
            .map_err(|_| Error::State("temporal lane count overflow".into()))?;
        ensure(
            self.counters.checked_physical_steps().is_some()
                && self.counters.vector_steps.checked_mul(lanes) == Some(self.counters.env_steps)
                && self.counters.episodes <= self.counters.env_steps,
            "invalid temporal run counters",
        )?;
        let earned = self
            .counters
            .env_steps
            .saturating_sub(self.config.warmup_transitions) as f64
            * self.config.updates_per_transition;
        ensure(
            self.update_credit.is_finite()
                && self.update_credit >= 0.
                && self.update_credit <= earned + (earned.abs() * 1e-12).max(1e-9)
                && self.last_behavior == self.behavior.settings(&self.counters)?,
            "invalid temporal update credit or exploration state",
        )?;
        let mut active = 0usize;
        let mut highest_begun = None;
        for lane in 0..self.collector.len() {
            let id = self.collector.lane_episode(lane).expect("validated lane");
            let timestep = self.collector.lane_timestep(lane).expect("validated lane");
            let start = self
                .collector
                .lane_learning_start(lane)
                .expect("validated lane");
            let begun = self.collector.lane_begun(lane).expect("validated lane");
            let ended = self
                .collector
                .lane_needs_reset(lane)
                .expect("validated lane");
            ensure(
                timestep.saturating_sub(start as u64) <= self.counters.env_steps,
                "temporal lane clock exceeds run counters",
            )?;
            if !begun {
                ensure(
                    self.counters.vector_steps == 0
                        && timestep == start as u64
                        && !ended
                        && self.store.episode(id).is_none()
                        && !self.store.is_ignored(id),
                    "unbegun temporal lane already has a tape",
                )?;
                continue;
            }
            highest_begun = Some(highest_begun.map_or(id, |previous: u64| previous.max(id)));
            if !ended {
                active += 1;
            }
            if let Some(episode) = self.store.episode(id) {
                let actor = self.collector.lane_actor(lane).expect("validated lane");
                let steps = usize::try_from(timestep)
                    .map_err(|_| Error::State("temporal lane timestep overflow".into()))?;
                ensure(
                    episode.end_timestep() == steps
                        && episode.learning_start == start
                        && episode
                            .observations
                            .last()
                            .is_some_and(|last| last.matches(&self.collector.observations()[lane]))
                        && episode.ended() == ended
                        && episode
                            .steps
                            .iter()
                            .enumerate()
                            .skip(start.saturating_sub(episode.base_timestep))
                            .all(|(index, step)| {
                                Some(step.behavior.revision)
                                    == self.collector.lane_behavior_revision_at(
                                        lane,
                                        episode.base_timestep + index,
                                    )
                            }),
                    "temporal lane and retained episode tape differ",
                )?;
                actor.validate_state_at(
                    self.collector.lane_state(lane).expect("validated lane"),
                    &self.collector.observations()[lane],
                    episode.observations.iter().rev().nth(1),
                )?;
            } else if self.store.is_ignored(id) {
                ensure(!ended, "ended temporal lane is still registered as ignored")?;
                ensure(
                    self.store
                        .active_timestep(id)
                        .and_then(|t| u64::try_from(t).ok())
                        == Some(timestep),
                    "ignored temporal episode clock differs from collector",
                )?;
            } else {
                ensure(
                    ended,
                    "live temporal lane is missing its full episode prefix",
                )?;
            }
        }
        ensure(
            self.store.active_count() == active && self.store.highest_episode_id() == highest_begun,
            "temporal store episode identities differ from collector",
        )?;
        ensure(
            self.last_windows.is_empty() || self.last_windows.len() == self.config.batch_size,
            "invalid temporal sampled-window count",
        )?;
        for &(id, start) in &self.last_windows {
            let episode = self.store.episode(id).ok_or_else(|| {
                Error::State("last sampled temporal prefix is no longer retained".into())
            })?;
            ensure(
                start >= episode.base_timestep
                    && start >= episode.learning_start
                    && start < episode.end_timestep()
                    && (episode.ended()
                        || episode.end_timestep() - start >= self.sequence_config.n_step),
                "last sampled temporal window has incomplete target context",
            )?;
        }
        self.audited.set(true);
        Ok(())
    }
}

pub(super) fn validate_config(
    config: &TemporalOffPolicyConfig,
    sequence: &SequenceConfig,
    lanes: usize,
) -> Result<()> {
    config.validate()?;
    sequence.validate()?;
    ensure(
        config.n_step == sequence.n_step
            && config.replay_capacity >= lanes
            && config.updates_per_transition <= MAX_UPDATES_PER_STEP as f64,
        "temporal n-step, active identity capacity, or update-rate contract differs",
    )?;
    Ok(())
}
impl<V, A, S> Snapshot for TemporalTrainer<V, A, S>
where
    V: VectorEnv + Snapshot,
    A: StatefulActor + Clone + Snapshot,
    A::State: Serialize + DeserializeOwned,
    S: BehaviorSchedule + Serialize + DeserializeOwned,
    S::Settings: Serialize + DeserializeOwned,
{
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_state()?;
        let result = self.collector.validate_snapshot();
        if result.is_err() {
            self.audited.set(false);
        }
        result
    }
    fn validate_learner_record(
        &self,
        record: &crate::core::checkpoint::LearnerRecord,
    ) -> Result<()> {
        let actual = record.temporal_returns.ok_or_else(|| {
            Error::State("temporal checkpoint is missing its typed return contract".into())
        })?;
        self.config.returns().check(actual)?;
        if let Some(spaces) = &record.temporal_spaces {
            spaces.check(
                self.store.observation_space(),
                self.collector.action_space(),
            )?;
        }
        let expected = self
            .replay_config
            .as_ref()
            .map_or_else(ReconstructionConfig::default, |config| {
                config.reconstruction
            });
        ensure(
            record.temporal_reconstruction.unwrap_or_default() == expected,
            "temporal snapshot and learner reconstruction contracts differ",
        )?;
        self.collector
            .latest_actor()
            .validate_learner_checkpoint(record)?;
        for lane in 0..self.collector.len() {
            self.collector
                .lane_actor(lane)
                .expect("validated lane")
                .validate_learner_checkpoint(record)?;
        }
        Ok(())
    }
    fn reconstruction_anchors(&self) -> Vec<&crate::core::anchors::StateAnchor> {
        self.store
            .episodes()
            .flat_map(|episode| episode.anchors.iter())
            .chain(self.replay_streams.values())
            .flat_map(|set| set.anchors.values())
            .collect()
    }
}
