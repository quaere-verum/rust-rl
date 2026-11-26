//! Opt-in coordination of stateful interaction and exact-prefix sequence learning.
use super::{
    behavior::BehaviorSchedule,
    config::Budget,
    counters::RunCounters,
    temporal_config::TemporalOffPolicyConfig,
    temporal_recording::{self, StreamingSink, TemporalReplayConfig},
};
use crate::{
    Error, Result,
    collect::{
        CollectionDelta,
        stateful::{RetainedHistory, StatefulCollector},
        vector_env::VectorEnv,
    },
    core::{
        actor::ActMode,
        anchors::{AnchorSet, RepresentationId},
        ensure,
        learner::UpdateReport,
        reconstruction::{
            ReconstructibleActor, ReconstructionConfig, ReconstructionDecision, ReconstructionMode,
            ReconstructionModel, ReplayReconstruction,
        },
        stateful::{BehaviorPublisher, StatefulActor, StatefulLearner, TemporalOffPolicyLearner},
    },
    data::temporal::{
        AnchorRequirements, EpisodeStore, SampleLimits, SequenceBatch, SequenceConfig,
        SequenceSampler,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::time::Instant;

mod validation;
use validation::validate_config;

const VERSION: u32 = 2;
/// Prevent one collection boundary from becoming an unbounded optimization loop.
const MAX_UPDATES_PER_STEP: usize = 1024;

#[derive(Debug, Default, Clone)]
pub struct TemporalProgress {
    pub collection: CollectionDelta,
    pub reports: Vec<UpdateReport>,
    /// Wall-clock diagnostics for this step; clocks are never part of a snapshot.
    pub collection_seconds: f64,
    pub learning_seconds: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "V: Serialize, A: Serialize, A::State: Serialize, S: Serialize, S::Settings: Serialize",
    deserialize = "V: DeserializeOwned, A: DeserializeOwned, A::State: DeserializeOwned, S: DeserializeOwned, S::Settings: DeserializeOwned"
))]
pub struct TemporalTrainer<V, A: StatefulActor, S: BehaviorSchedule> {
    version: u32,
    collector: StatefulCollector<V, A>,
    store: EpisodeStore,
    sampler: SequenceSampler,
    counters: RunCounters,
    update_credit: f64,
    behavior_revision: u64,
    config: TemporalOffPolicyConfig,
    sequence_config: SequenceConfig,
    behavior: S,
    last_behavior: S::Settings,
    last_windows: Vec<(u64, usize)>,
    poisoned: bool,
    #[serde(skip)]
    audited: Cell<bool>,
    /// Allocation reuse only; no pending sample or algorithm state is persisted.
    #[serde(skip)]
    batch_scratch: Option<SequenceBatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_config: Option<TemporalReplayConfig>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    replay_streams: BTreeMap<u64, AnchorSet>,
}

impl<V: VectorEnv, A: StatefulActor + Clone, S: BehaviorSchedule> TemporalTrainer<V, A, S> {
    /// The supplied actor determines behavior in the initial episodes. Its
    /// exploration should match the schedule at zero counters. Warmup suppresses
    /// optimization credit; it does not replace this generic actor with a hidden
    /// algorithm-specific random policy.
    pub fn new(
        env: V,
        actor: A,
        config: TemporalOffPolicyConfig,
        sequence_config: SequenceConfig,
        behavior: S,
        seed: u64,
    ) -> Result<Self> {
        Self::new_with_inference_mode(
            env,
            actor,
            config,
            sequence_config,
            behavior,
            seed,
            crate::collect::stateful::InferenceMode::Scalar,
        )
    }
    /// Numerical execution is chosen before the first observation is assimilated.
    pub fn new_with_inference_mode(
        env: V,
        actor: A,
        config: TemporalOffPolicyConfig,
        sequence_config: SequenceConfig,
        behavior: S,
        seed: u64,
        inference_mode: crate::collect::stateful::InferenceMode,
    ) -> Result<Self> {
        validate_config(&config, &sequence_config, env.len())?;
        behavior.validate()?;
        let store = EpisodeStore::new(
            config.replay_capacity,
            actor.observation_space(),
            actor.action_space().clone(),
        )?;
        let behavior_revision = actor.revision();
        let last_behavior = behavior.settings(&RunCounters::default())?;
        let trainer = Self {
            version: VERSION,
            collector: StatefulCollector::new_with_inference_mode(
                env,
                actor,
                seed,
                inference_mode,
            )?,
            store,
            sampler: SequenceSampler::new(seed ^ 0x5345_515f_5341_4d50),
            counters: RunCounters::default(),
            update_credit: 0.,
            behavior_revision,
            config,
            sequence_config,
            behavior,
            last_behavior,
            last_windows: Vec::new(),
            poisoned: false,
            audited: Cell::new(false),
            batch_scratch: None,
            replay_config: None,
            replay_streams: BTreeMap::new(),
        };
        trainer.validate_state()?;
        Ok(trainer)
    }
    pub fn counters(&self) -> &RunCounters {
        &self.counters
    }
    pub fn collector(&self) -> &StatefulCollector<V, A> {
        &self.collector
    }
    pub fn store(&self) -> &EpisodeStore {
        &self.store
    }
    pub fn last_windows(&self) -> &[(u64, usize)] {
        &self.last_windows
    }
    /// The actual most recent learner batch of the current step, when still
    /// resident. Scratch batches are cleared by checkpoint restoration.
    pub fn last_update_batch(&self) -> Option<&SequenceBatch> {
        if self.last_windows.is_empty() {
            None
        } else {
            self.batch_scratch.as_ref()
        }
    }
    pub fn config(&self) -> &TemporalOffPolicyConfig {
        &self.config
    }
    pub fn sequence_config(&self) -> &SequenceConfig {
        &self.sequence_config
    }
    pub fn update_credit(&self) -> f64 {
        self.update_credit
    }
    pub fn behavior_revision(&self) -> u64 {
        self.behavior_revision
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    pub fn replay_config(&self) -> Option<&TemporalReplayConfig> {
        self.replay_config.as_ref()
    }
    pub fn replay_streams(&self) -> &BTreeMap<u64, AnchorSet> {
        &self.replay_streams
    }
    pub fn set_replay_config(&mut self, config: TemporalReplayConfig) -> Result<()> {
        ensure(
            self.collector.inference_mode()
                != crate::collect::stateful::InferenceMode::CanonicalBatched,
            "canonical inference requires ordinary full-history replay; anchor recording lacks numerical execution provenance",
        )?;
        self.validate_state()?;
        ensure(
            self.counters.env_steps == 0 && self.store.active_count() == 0 && self.store.is_empty(),
            "temporal replay configuration must be installed before collection",
        )?;
        config.validate(
            self.config.replay_capacity,
            self.collector.len(),
            &self.sequence_config,
        )?;
        let store = EpisodeStore::with_storage(
            self.config.replay_capacity,
            self.collector.observation_space(),
            self.collector.action_space().clone(),
            config.storage.clone(),
        )?;
        self.store = store;
        self.replay_config = Some(config);
        self.version = 3;
        self.audited.set(false);
        self.validate_state()
    }
    pub fn set_inference_mode(
        &mut self,
        mode: crate::collect::stateful::InferenceMode,
    ) -> Result<()> {
        self.validate_local()?;
        self.collector.set_inference_mode(mode)
    }

    pub(super) fn validate_learner(&self, learner: &impl TemporalOffPolicyLearner) -> Result<()> {
        self.config.returns().check(learner.return_contract())?;
        if let Some((observations, actions)) = learner.replay_spaces() {
            ensure(
                self.store.observation_space() == observations
                    && self.collector.action_space() == actions,
                "temporal trainer and learner observation/action spaces differ",
            )?;
        }
        Ok(())
    }

    /// One vector interaction, followed by at most 1024 credited learner calls.
    /// Any failure closes this trainer: restore a previously valid run checkpoint
    /// rather than continuing after potentially partial environment/model mutation.
    pub fn step<L>(&mut self, learner: &mut L) -> Result<TemporalProgress>
    where
        L: StatefulLearner<Batch = SequenceBatch>
            + BehaviorPublisher<Actor = A, Settings = S::Settings>
            + TemporalOffPolicyLearner,
    {
        self.step_with_publication(learner, L::behavior_actor)
    }

    /// Compose a publication from the updated task learner without changing
    /// its fitting, replay or collection schedule. The returned actor becomes
    /// available at the collector's existing episode boundaries.
    pub fn step_with_publication<L>(
        &mut self,
        learner: &mut L,
        publish: impl FnMut(&L, &S::Settings, u64) -> Result<A>,
    ) -> Result<TemporalProgress>
    where
        L: TemporalOffPolicyLearner<Batch = SequenceBatch>,
    {
        // Configuration rejection is not an interaction failure: do not poison or
        // change counters, replay, memories, RNGs or learner state.
        self.validate_learner(learner)?;
        ensure(
            self.replay_config.as_ref().is_none_or(|config| {
                !config.is_segmented() && config.reconstruction == ReconstructionConfig::default()
            }),
            "configured anchored replay requires step_with_reconstruction",
        )?;
        ensure(
            !self.poisoned,
            "temporal trainer requires restoration from a valid run checkpoint",
        )?;
        let result = self.step_coordinated(
            learner,
            |trainer, _| {
                trainer
                    .collector
                    .collect(1, ActMode::Sample, &mut trainer.store)
            },
            |_| Ok(None),
            publish,
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn step_with_reconstruction<L>(&mut self, learner: &mut L) -> Result<TemporalProgress>
    where
        L: StatefulLearner<Batch = SequenceBatch>
            + BehaviorPublisher<Actor = A, Settings = S::Settings>
            + TemporalOffPolicyLearner
            + ReplayReconstruction,
    {
        self.validate_learner(learner)?;
        let config = self.replay_config.clone().ok_or_else(|| {
            Error::State("temporal reconstruction recording is not configured".into())
        })?;
        ensure(
            config.reconstruction == learner.reconstruction_config(),
            "learner and recording reconstruction contracts differ",
        )?;
        ensure(
            !self.poisoned,
            "temporal trainer requires restoration from a valid run checkpoint",
        )?;
        self.validate_local()?;
        // Source/codec/lag rejection is pure: unavailable declared reconstruction
        // does not poison an otherwise consistent run or consume interaction.
        let mut recording_models = if config.is_segmented() {
            let models = learner.replay_representations()?;
            temporal_recording::validate_source_keys(&config, &models)?;
            temporal_recording::audit_streams(
                config.reconstruction,
                &models,
                &self.replay_streams,
            )?;
            Some(models)
        } else {
            None
        };
        let result = self.step_coordinated(
            learner,
            |trainer, _learner| {
                if !config.is_segmented() {
                    return trainer
                        .collector
                        .collect(1, ActMode::Sample, &mut trainer.store);
                }
                let models = recording_models
                    .take()
                    .expect("preflighted reconstruction sources");
                let mut sink = StreamingSink {
                    store: &mut trainer.store,
                    streams: &mut trainer.replay_streams,
                    models: &models,
                    config: &config,
                    max_streams: trainer.collector.len(),
                };
                trainer.collector.collect(1, ActMode::Sample, &mut sink)
            },
            |learner| {
                let models = learner.replay_representations()?;
                temporal_recording::validate_source_keys(&config, &models)?;
                Ok(Some(temporal_recording::requirements(
                    config.reconstruction,
                    &models,
                )?))
            },
            L::behavior_actor,
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn step_coordinated<L, C, R>(
        &mut self,
        learner: &mut L,
        mut collect: C,
        mut request: R,
        publish: impl FnMut(&L, &S::Settings, u64) -> Result<A>,
    ) -> Result<TemporalProgress>
    where
        L: TemporalOffPolicyLearner<Batch = SequenceBatch>,
        C: FnMut(&mut Self, &L) -> Result<CollectionDelta>,
        R: FnMut(&L) -> Result<Option<AnchorRequirements>>,
    {
        self.validate_local()?;
        let lanes = u64::try_from(self.collector.len())
            .map_err(|_| Error::State("temporal lane count overflow".into()))?;
        self.counters
            .env_steps
            .checked_add(lanes)
            .ok_or_else(|| Error::State("temporal environment counter overflow".into()))?;
        self.counters
            .vector_steps
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal vector counter overflow".into()))?;
        self.counters
            .episodes
            .checked_add(lanes)
            .ok_or_else(|| Error::State("temporal episode counter overflow".into()))?;
        self.last_windows.clear();
        let before = self.counters.env_steps;
        let clock = Instant::now();
        let collection = collect(self, learner)?;
        let collection_seconds = clock.elapsed().as_secs_f64();
        ensure(
            self.counters
                .checked_physical_steps()
                .and_then(|total| total.checked_add(collection.physical_steps()))
                .is_some(),
            "temporal physical step counter overflow",
        )?;
        self.counters.add_collection(collection);
        self.update_credit += super::schedule::collection_credit(
            before,
            self.counters.env_steps,
            self.config.warmup_transitions,
            self.config.updates_per_transition,
        );
        ensure(
            self.update_credit.is_finite(),
            "temporal update credit overflow",
        )?;
        let clock = Instant::now();
        let mut reports = Vec::new();
        while self.update_credit >= 1. && reports.len() < MAX_UPDATES_PER_STEP {
            // Parameters may change after every update: never reuse a stale request.
            let requirements = request(learner)?;
            let limits = self
                .replay_config
                .as_ref()
                .map_or_else(SampleLimits::default, |config| config.sample_limits);
            let available = if let Some(requirements) = &requirements {
                self.store
                    .eligible_starts_with(&self.sequence_config, requirements, &limits)?
                    > 0
            } else {
                self.store.eligible_starts(self.sequence_config.n_step)? > 0
            };
            if !available {
                break;
            }
            let batch = self.batch_scratch.get_or_insert_with(|| SequenceBatch {
                samples: Vec::new(),
                config: self.sequence_config,
            });
            if !learner.requires_replay_history()
                && self.sequence_config.burn_in == 0
                && requirements.is_none()
                && limits.has_unbounded_samples()
            {
                self.sampler.sample_stateless_into(
                    &self.store,
                    &self.sequence_config,
                    self.config.batch_size,
                    &limits,
                    batch,
                )?;
            } else if let Some(requirements) = &requirements {
                self.sampler.sample_into_with(
                    &self.store,
                    &self.sequence_config,
                    self.config.batch_size,
                    requirements,
                    &limits,
                    batch,
                )?;
            } else if self.replay_config.is_some() {
                self.sampler.sample_into_with(
                    &self.store,
                    &self.sequence_config,
                    self.config.batch_size,
                    &AnchorRequirements::default(),
                    &limits,
                    batch,
                )?;
            } else {
                self.sampler.sample_into(
                    &self.store,
                    &self.sequence_config,
                    self.config.batch_size,
                    batch,
                )?;
            }
            self.last_windows.clear();
            self.last_windows.extend(
                batch
                    .samples
                    .iter()
                    .map(|sample| (sample.episode_id, sample.absolute_learn_start())),
            );
            let report = learner.update(batch)?;
            ensure(
                report.updates > 0 && report.metrics.values().all(|value| value.is_finite()),
                "temporal learner returned no updates or nonfinite metrics",
            )?;
            self.counters.updates = self
                .counters
                .updates
                .checked_add(report.updates)
                .ok_or_else(|| Error::State("temporal learner counter overflow".into()))?;
            self.update_credit -= 1.;
            reports.push(report);
        }
        let settings = self.behavior.settings(&self.counters)?;
        if !reports.is_empty() || settings != self.last_behavior {
            self.publish_behavior(learner, settings, publish)?;
        }
        Ok(TemporalProgress {
            collection,
            reports,
            collection_seconds,
            learning_seconds: clock.elapsed().as_secs_f64(),
        })
    }

    fn publish_behavior<L>(
        &mut self,
        learner: &L,
        settings: S::Settings,
        publish: impl FnOnce(&L, &S::Settings, u64) -> Result<A>,
    ) -> Result<()> {
        self.republish(|revision| publish(learner, &settings, revision))?;
        self.last_behavior = settings;
        Ok(())
    }

    pub(super) fn republish(&mut self, build: impl FnOnce(u64) -> Result<A>) -> Result<()> {
        let revision = self
            .behavior_revision
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal behavior revision overflow".into()))?;
        let actor = build(revision)?;
        ensure(
            actor.revision() == revision,
            "temporal learner returned a different behavior revision",
        )?;
        self.collector.publish(actor)?;
        self.behavior_revision = revision;
        Ok(())
    }
    /// Budgets are additional to current counters and stop at step boundaries.
    /// An environment/vector bound is required because long discarded partial
    /// episodes may never supply a learnable sequence for an updates-only budget.
    pub fn run<L>(&mut self, learner: &mut L, budget: Budget) -> Result<RunCounters>
    where
        L: StatefulLearner<Batch = SequenceBatch>
            + BehaviorPublisher<Actor = A, Settings = S::Settings>
            + TemporalOffPolicyLearner,
    {
        self.config.returns().check(learner.return_contract())?;
        budget.validate()?;
        ensure(
            budget.env_steps.is_some() || budget.vector_steps.is_some(),
            "temporal runs require an environment-step or vector-step bound",
        )?;
        self.validate_state()?;
        let start = self.counters;
        while !budget.reached(&start, &self.counters) {
            self.step(learner)?;
        }
        Ok(self.counters)
    }
    pub fn run_with_reconstruction<L>(
        &mut self,
        learner: &mut L,
        budget: Budget,
    ) -> Result<RunCounters>
    where
        L: StatefulLearner<Batch = SequenceBatch>
            + BehaviorPublisher<Actor = A, Settings = S::Settings>
            + TemporalOffPolicyLearner
            + ReplayReconstruction,
    {
        self.config.returns().check(learner.return_contract())?;
        budget.validate()?;
        ensure(
            budget.env_steps.is_some() || budget.vector_steps.is_some(),
            "temporal runs require an environment-step or vector-step bound",
        )?;
        self.validate_state()?;
        self.validate_reconstruction(learner)?;
        let start = self.counters;
        while !budget.reached(&start, &self.counters) {
            self.step_with_reconstruction(learner)?;
        }
        Ok(self.counters)
    }
}

impl<V: VectorEnv, A: ReconstructibleActor + Clone, S: BehaviorSchedule> TemporalTrainer<V, A, S> {
    /// Adopt the latest behavior actor at the current observation without stepping
    /// or resetting the environment. Anchor representation selection is explicit.
    pub fn adopt_lane(
        &mut self,
        lane: usize,
        id: RepresentationId,
        config: ReconstructionConfig,
    ) -> Result<Option<ReconstructionDecision>> {
        self.validate_local()?;
        config.validate()?;
        let old_revision = self
            .collector
            .lane_revision(lane)
            .ok_or_else(|| Error::State("adoption lane out of range".into()))?;
        ensure(
            self.collector.latest_revision() > old_revision,
            "adoption requires a newer published behavior actor",
        )?;
        ensure(
            self.collector.lane_needs_reset(lane) == Some(false),
            "cannot adopt an ended temporal lane",
        )?;
        let episode_id = self
            .collector
            .lane_episode(lane)
            .expect("validated adoption lane");
        let episode = self
            .store
            .episode(episode_id)
            .ok_or_else(|| Error::State("adoption history is not retained".into()))?;
        match config.mode {
            ReconstructionMode::Exact => {
                ensure(
                    episode.base_timestep == 0,
                    "exact adoption requires the full actual reset prefix",
                )?;
                let actions = episode
                    .steps
                    .iter()
                    .map(|step| step.action.clone())
                    .collect::<Vec<_>>();
                let rewards = episode
                    .steps
                    .iter()
                    .map(|step| step.reward.clone())
                    .collect::<Vec<_>>();
                self.collector.reconcile_lane(
                    lane,
                    crate::core::stateful::ReconstructionHistory {
                        observations: &episode.observations,
                        actions: &actions,
                        rewards: crate::core::stateful::HistoryRewards::Traces(&rewards),
                    },
                )?;
                self.audited.set(false);
                Ok(None)
            }
            ReconstructionMode::Anchored { max_generation_lag } => {
                let model = self.collector.latest_actor().representation()?;
                let clock = usize::try_from(
                    self.collector
                        .lane_timestep(lane)
                        .expect("validated adoption clock"),
                )
                .map_err(|_| Error::State("adoption clock overflow".into()))?;
                let anchor = episode
                    .anchors
                    .iter()
                    .rev()
                    .filter(|set| set.observation_index() <= clock)
                    .filter_map(|set| set.anchors.get(&id))
                    .find(|anchor| {
                        &anchor.schema == model.schema()
                            && anchor.generation <= model.generation()
                            && max_generation_lag
                                .is_none_or(|lag| model.generation() - anchor.generation <= lag)
                    })
                    .ok_or_else(|| {
                        Error::State(
                            "no compatible retained anchor for explicit behavior adoption".into(),
                        )
                    })?;
                let start = anchor.observation_index - episode.base_timestep;
                let previous = if start == 0 {
                    episode.previous.as_ref().map(|step| {
                        crate::core::stateful::PreviousInput::from_total(&step.action, step.reward)
                    })
                } else {
                    let step = &episode.steps[start - 1];
                    Some(crate::core::stateful::PreviousInput::from_transition(
                        &step.action,
                        &step.reward,
                        &episode.observations[start - 1],
                    ))
                };
                let actions = episode.steps[start..]
                    .iter()
                    .map(|step| step.action.clone())
                    .collect::<Vec<_>>();
                let rewards = episode.steps[start..]
                    .iter()
                    .map(|step| step.reward.clone())
                    .collect::<Vec<_>>();
                // A codec error on the chosen compatible boundary is an error,
                // never a silent retry from zero or an older estimator state.
                let decision = self.collector.reconcile_lane_anchored(
                    lane,
                    id,
                    config,
                    anchor,
                    RetainedHistory {
                        base_timestep: anchor.observation_index,
                        previous,
                        observations: &episode.observations[start..],
                        actions: &actions,
                        rewards: crate::core::stateful::HistoryRewards::Traces(&rewards),
                    },
                )?;
                self.audited.set(false);
                Ok(Some(decision))
            }
        }
    }
}
