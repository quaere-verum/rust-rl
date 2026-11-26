//! Shared exact on-policy collection, update handoff and complete-run snapshots.
use super::{
    config::{Budget, OnPolicyConfig},
    counters::RunCounters,
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::{TemporalLearnerBinding, TemporalOnPolicyLearner};
#[cfg(feature = "neural")]
use crate::nn::temporal_actor::TemporalActor;
use crate::{
    Error, Result,
    collect::{
        CollectionDelta,
        stateful::{InferenceMode, StatefulCollector},
        temporal_on_policy::OnPolicyRecording,
        vector_env::VectorEnv,
    },
    core::{
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        stateful::{
            HistoryInput, HistoryRewards, PreviousInput, ReconstructionHistory, StatefulActor,
            replay_actor_history,
        },
    },
    data::{
        temporal::{SampleLimits, SequenceConfig},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRollout},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{cell::Cell, collections::BTreeMap, time::Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOnPolicyConfig {
    pub collection: OnPolicyConfig,
    pub sequence: SequenceConfig,
    pub capacity: usize,
    pub max_fresh: usize,
    pub max_metadata_bytes: usize,
    /// Bound on retained factual observations, actions, and primitive rewards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rollout_bytes: Option<usize>,
    pub sample_limits: SampleLimits,
    pub inference_mode: InferenceMode,
}
impl Default for TemporalOnPolicyConfig {
    fn default() -> Self {
        Self {
            collection: OnPolicyConfig::default(),
            sequence: SequenceConfig {
                burn_in: 4,
                learn_len: 16,
                n_step: 1,
            },
            capacity: 16384,
            max_fresh: 8192,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_rollout_bytes: None,
            sample_limits: TemporalRollout::default_sample_limits(),
            inference_mode: InferenceMode::Batched,
        }
    }
}
impl TemporalOnPolicyConfig {
    pub fn validate(&self) -> Result<()> {
        self.collection.validate()?;
        self.sequence.validate()?;
        ensure(
            self.sequence.n_step == 1
                && self.capacity > 0
                && self.max_fresh > 0
                && self.max_metadata_bytes > 0
                && self.max_rollout_bytes.is_none_or(|bytes| bytes > 0)
                && self.sample_limits.max_sample_tokens > 0
                && self.sample_limits.max_batch_tokens > 0
                && self.sample_limits.max_sample_payload_bytes > 0
                && self.sample_limits.max_batch_payload_bytes > 0,
            "invalid temporal on-policy limits or sequence horizon",
        )
    }
    fn validate_kind(&self, kind: TemporalBatchKind) -> Result<()> {
        self.validate()?;
        ensure(
            (kind == TemporalBatchKind::Episodic) == self.collection.episode_guard.is_some(),
            "episodic temporal collection requires an explicit episode guard; fragment collection has none",
        )
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TemporalOnPolicyTimings {
    pub collection_seconds: f64,
    pub target_seconds: f64,
    pub learning_seconds: f64,
    pub reconciliation_seconds: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "V: Serialize, A: Serialize, A::State: Serialize",
    deserialize = "V: DeserializeOwned, A: DeserializeOwned, A::State: DeserializeOwned"
))]
pub struct TemporalOnPolicyTrainer<
    V,
    #[cfg(feature = "neural")] A: StatefulActor = TemporalActor,
    #[cfg(not(feature = "neural"))] A: StatefulActor,
> {
    version: u32,
    collector: StatefulCollector<V, A>,
    config: TemporalOnPolicyConfig,
    kind: TemporalBatchKind,
    counters: RunCounters,
    learner_calls: u64,
    collection_ticks: usize,
    rollout: TemporalRollout,
    pending: Option<TemporalOnPolicyBatch<A>>,
    completed: Vec<bool>,
    poisoned: bool,
    timings: TemporalOnPolicyTimings,
    #[serde(skip)]
    audited: Cell<bool>,
}
impl<V: VectorEnv, A: OnPolicyIdentity + Clone> TemporalOnPolicyTrainer<V, A>
where
    A::State: PartialEq,
{
    pub fn new<L: TemporalOnPolicyLearner<A>>(
        environment: V,
        learner: &L,
        config: TemporalOnPolicyConfig,
        seed: u64,
    ) -> Result<Self> {
        Self::initialize(environment, learner, config, seed, None)
    }
    /// Initialize environment streams from explicit per-lane seeds while
    /// keeping action sampling independently seeded. Subsequent resets remain
    /// environment-owned and the configured inference mode is preserved.
    pub fn new_with_reset_seeds<L: TemporalOnPolicyLearner<A>>(
        environment: V,
        learner: &L,
        config: TemporalOnPolicyConfig,
        action_seed: u64,
        reset_seeds: &[u64],
    ) -> Result<Self> {
        Self::initialize(environment, learner, config, action_seed, Some(reset_seeds))
    }
    fn initialize<L: TemporalOnPolicyLearner<A>>(
        environment: V,
        learner: &L,
        config: TemporalOnPolicyConfig,
        seed: u64,
        reset_seeds: Option<&[u64]>,
    ) -> Result<Self> {
        let kind = learner.collection_kind();
        config.validate_kind(kind)?;
        ensure(
            !config.collection.normalize_advantages || !learner.normalizes_advantages_in_update(),
            "disable rollout normalization when the temporal learner normalizes minibatches",
        )?;
        if kind != TemporalBatchKind::Episodic {
            ensure(
                config
                    .collection
                    .rollout_steps
                    .checked_mul(environment.len())
                    .is_some_and(|n| n <= config.max_fresh),
                "temporal rollout exceeds fresh-record limit",
            )?;
        }
        let actor = learner.behavior_actor(&(), learner.revision())?;
        let rollout = TemporalRollout::with_resident_limit(
            config.capacity,
            environment.observation_space(),
            environment.action_space().clone(),
            config.max_fresh,
            config.max_metadata_bytes,
            config.max_rollout_bytes,
        )?;
        let collector = match reset_seeds {
            Some(seeds) => StatefulCollector::new_with_reset_seeds(
                environment,
                actor,
                seed,
                config.inference_mode,
                seeds,
            )?,
            None => StatefulCollector::new_with_inference_mode(
                environment,
                actor,
                seed,
                config.inference_mode,
            )?,
        };
        let lanes = collector.len();
        Ok(Self {
            version: 1,
            collector,
            config,
            kind,
            counters: RunCounters::default(),
            learner_calls: 0,
            collection_ticks: 0,
            rollout,
            pending: None,
            completed: vec![false; lanes],
            poisoned: false,
            timings: Default::default(),
            audited: Cell::new(true),
        })
    }
    pub fn collector(&self) -> &StatefulCollector<V, A> {
        &self.collector
    }
    pub fn rollout(&self) -> &TemporalRollout {
        &self.rollout
    }
    pub fn counters(&self) -> &RunCounters {
        &self.counters
    }
    pub fn config(&self) -> &TemporalOnPolicyConfig {
        &self.config
    }
    pub fn timings(&self) -> &TemporalOnPolicyTimings {
        &self.timings
    }
    pub fn pending_batch(&self) -> Option<&TemporalOnPolicyBatch<A>> {
        self.pending.as_ref()
    }
    pub fn learner_calls(&self) -> u64 {
        self.learner_calls
    }
    fn preflight<L: TemporalOnPolicyLearner<A>>(&self, learner: &L) -> Result<()> {
        ensure(
            !self.poisoned,
            "temporal on-policy trainer requires checkpoint recovery",
        )?;
        if !self.audited.get() {
            self.validate_runtime_state()?;
        }
        self.config.validate_kind(self.kind)?;
        ensure(
            self.config.inference_mode == self.collector.inference_mode(),
            "temporal trainer/collector numerical modes differ",
        )?;
        self.collector.validate_local()?;
        ensure(
            self.kind == learner.collection_kind()
                && self.collector.latest_revision() == learner.revision(),
            "temporal on-policy learner revision or collection kind differs",
        )?;
        ensure(
            !self.config.collection.normalize_advantages
                || !learner.normalizes_advantages_in_update(),
            "duplicate temporal advantage normalization",
        )?;
        let actor = learner.behavior_actor(&(), learner.revision())?;
        ensure(
            actor.behavior_identity() == self.collector.latest_actor().behavior_identity(),
            "temporal on-policy learner parameters differ from collection",
        )
    }
    pub fn collect_tick<L: TemporalOnPolicyLearner<A>>(
        &mut self,
        learner: &L,
    ) -> Result<CollectionDelta> {
        self.preflight(learner)?;
        ensure(
            self.pending.is_none(),
            "cannot collect while a sealed temporal batch is pending",
        )?;
        if let Some(guard) = self.config.collection.episode_guard {
            ensure(
                self.collection_ticks < guard,
                "complete-episode temporal guard exhausted",
            )?;
        }
        let active: Vec<_> = self
            .completed
            .iter()
            .map(|done| self.kind != TemporalBatchKind::Episodic || !*done)
            .collect();
        ensure(
            active.iter().any(|&a| a),
            "complete episodic collection is already ready",
        )?;
        self.counters
            .env_steps
            .checked_add(self.collector.len() as u64)
            .ok_or_else(|| Error::State("temporal transition counter overflow".into()))?;
        self.counters
            .vector_steps
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal vector counter overflow".into()))?;
        self.counters
            .episodes
            .checked_add(self.collector.len() as u64)
            .ok_or_else(|| Error::State("temporal episode counter overflow".into()))?;
        let tick = self
            .collection_ticks
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal fragment clock overflow".into()))?;
        if self.collection_ticks == 0 && self.rollout.fresh_len() == 0 {
            let previous = learner.revision();
            learner.begin_collection()?;
            let handoff = (|| {
                ensure(
                    learner.revision() >= previous,
                    "collection start moved the behavior revision backwards",
                )?;
                if learner.revision() != previous {
                    self.publish_and_reconcile(learner)?;
                }
                self.preflight(learner)
            })();
            if let Err(error) = handoff {
                self.poisoned = true;
                return Err(error);
            }
        }
        let start = Instant::now();
        let result = self.collector.tick_recorded(
            &active,
            ActMode::Sample,
            &mut OnPolicyRecording::new(&mut self.rollout),
        );
        self.timings.collection_seconds += start.elapsed().as_secs_f64();
        let delta = match result {
            Ok(value) => value,
            Err(e) => {
                self.poisoned = true;
                return Err(e);
            }
        };
        if self
            .counters
            .checked_physical_steps()
            .and_then(|ticks| ticks.checked_add(delta.env_steps))
            .and_then(|ticks| ticks.checked_add(delta.additional_physical_steps))
            .is_none()
        {
            self.poisoned = true;
            return Err(Error::State("temporal physical counter overflow".into()));
        }
        self.counters.add_collection(delta);
        self.collection_ticks = tick;
        if self.kind == TemporalBatchKind::Episodic {
            for lane in 0..self.collector.len() {
                if self.collector.lane_needs_reset(lane) == Some(true) {
                    let id = self.collector.lane_episode(lane).expect("existing lane");
                    let episode = self
                        .rollout
                        .store()
                        .episode(id)
                        .expect("recorded final episode");
                    if episode.steps.last().is_some_and(|s| s.truncated) {
                        self.poisoned = true;
                        return Err(Error::State(
                            "Monte Carlo temporal collection rejects truncated episodes".into(),
                        ));
                    }
                    self.completed[lane] = true;
                }
            }
        }
        Ok(delta)
    }
    pub fn seal<L: TemporalOnPolicyLearner<A>>(
        &mut self,
        learner: &L,
    ) -> Result<&TemporalOnPolicyBatch<A>> {
        self.preflight(learner)?;
        if self.pending.is_none() {
            if self.kind == TemporalBatchKind::Episodic {
                ensure(
                    self.completed.iter().all(|&x| x),
                    "Monte Carlo temporal collection has incomplete lanes",
                )?;
            }
            let start = Instant::now();
            let batch = self.rollout.seal_with_limits(
                self.collector.latest_actor().clone(),
                self.kind,
                self.config.collection.gamma,
                if self.kind == TemporalBatchKind::Bootstrapped {
                    self.config.collection.gae_lambda
                } else {
                    None
                },
                self.kind == TemporalBatchKind::Bootstrapped
                    && self.config.collection.normalize_advantages,
                &self.config.sequence,
                &self.config.sample_limits,
            )?;
            self.timings.target_seconds += start.elapsed().as_secs_f64();
            self.pending = Some(batch);
        }
        Ok(self.pending.as_ref().expect("sealed batch"))
    }
    pub fn update<L: TemporalOnPolicyLearner<A>>(
        &mut self,
        learner: &mut L,
    ) -> Result<crate::core::learner::UpdateReport> {
        self.seal(learner)?;
        self.learner_calls
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal learner call counter overflow".into()))?;
        let previous = learner.revision();
        let start = Instant::now();
        let result = learner.update(self.pending.as_ref().expect("sealed batch"));
        self.timings.learning_seconds += start.elapsed().as_secs_f64();
        let report = match result {
            Ok(report) => report,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        let finish = (|| {
            ensure(
                learner.revision() > previous,
                "temporal on-policy learner did not consume its behavior revision",
            )?;
            self.publish_and_reconcile(learner)?;
            self.rollout.acknowledge()?;
            self.counters.updates = self
                .counters
                .updates
                .checked_add(report.updates)
                .ok_or_else(|| Error::State("temporal optimizer counter overflow".into()))?;
            self.learner_calls += 1;
            self.pending = None;
            self.collection_ticks = 0;
            self.completed.fill(false);
            Ok(())
        })();
        if let Err(e) = finish {
            self.poisoned = true;
            return Err(e);
        }
        Ok(report)
    }
    fn publish_and_reconcile<L: TemporalOnPolicyLearner<A>>(&mut self, learner: &L) -> Result<()> {
        let actor = learner.behavior_actor(&(), learner.revision())?;
        let start = Instant::now();
        self.collector.publish(actor)?;
        for lane in 0..self.collector.len() {
            if self.collector.lane_needs_reset(lane) == Some(true)
                || self.collector.reconcile_pending_context(lane)?
            {
                continue;
            }
            let id = self.collector.lane_episode(lane).expect("existing lane");
            let initial;
            let (observations, actions, rewards, behavior) =
                if let Some(episode) = self.rollout.store().episode(id) {
                    (
                        episode.observations.as_slice(),
                        episode
                            .steps
                            .iter()
                            .map(|s| s.action.clone())
                            .collect::<Vec<_>>(),
                        episode
                            .steps
                            .iter()
                            .map(|s| s.reward.clone())
                            .collect::<Vec<_>>(),
                        episode
                            .steps
                            .iter()
                            .map(|s| s.behavior.clone())
                            .collect::<Vec<_>>(),
                    )
                } else {
                    ensure(
                        self.collector.lane_begun(lane) == Some(false)
                            && self.collector.lane_timestep(lane) == Some(0),
                        "exact on-policy history is missing",
                    )?;
                    initial = vec![self.collector.observations()[lane].clone()];
                    (initial.as_slice(), Vec::new(), Vec::new(), Vec::new())
                };
            let history = crate::collect::stateful::RetainedHistory {
                base_timestep: 0,
                previous: None,
                observations,
                actions: &actions,
                rewards: HistoryRewards::Traces(&rewards),
            };
            self.collector
                .validate_lane_revision_prefix(lane, history, &behavior)?;
            self.collector.reconcile_lane_with(
                lane,
                ReconstructionHistory {
                    observations,
                    actions: &actions,
                    rewards: HistoryRewards::Traces(&rewards),
                },
                |actor, history| learner.reconstruct_published_state(actor, history),
            )?;
            self.collector
                .compact_lane_revisions(lane, history, &behavior)?;
        }
        self.timings.reconciliation_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }
    pub fn cycle<L: TemporalOnPolicyLearner<A>>(
        &mut self,
        learner: &mut L,
    ) -> Result<crate::core::learner::UpdateReport> {
        if self.kind == TemporalBatchKind::Episodic {
            while !self.completed.iter().all(|&x| x) {
                self.collect_tick(learner)?;
            }
        } else {
            while self.collection_ticks < self.config.collection.rollout_steps {
                self.collect_tick(learner)?;
            }
        }
        self.update(learner)
    }
    /// Additional bounded work; reports are streamed instead of retained without a limit.
    pub fn train<L: TemporalOnPolicyLearner<A>>(
        &mut self,
        learner: &mut L,
        budget: Budget,
        mut callback: Option<&mut dyn FnMut(&super::trainer::Progress)>,
    ) -> Result<super::trainer::TrainingSummary> {
        budget.validate()?;
        let start = self.counters;
        let clock = Instant::now();
        let before = self.timings.clone();
        let mut last = None;
        let progress = |now: &Self, last: Option<crate::core::learner::UpdateReport>| {
            let elapsed = clock.elapsed().as_secs_f64();
            super::trainer::Progress {
                counters: now.counters,
                collection_seconds: now.timings.collection_seconds - before.collection_seconds,
                learning_seconds: now.timings.learning_seconds - before.learning_seconds,
                elapsed_seconds: elapsed,
                transitions_per_second: (now.counters.env_steps - start.env_steps) as f64
                    / elapsed.max(1e-9),
                last_update: last,
            }
        };
        while !budget.reached(&start, &self.counters) {
            if self.kind == TemporalBatchKind::Episodic {
                last = Some(self.cycle(learner)?);
            } else {
                while self.collection_ticks < self.config.collection.rollout_steps
                    && !budget.reached(&start, &self.counters)
                {
                    self.collect_tick(learner)?;
                }
                if self.rollout.fresh_len() > 0 {
                    last = Some(self.update(learner)?);
                } else {
                    break;
                }
            }
            if let Some(callback) = callback.as_mut() {
                callback(&progress(self, last.clone()));
            }
        }
        Ok(progress(self, last))
    }
}

impl<V, A> TemporalOnPolicyTrainer<V, A>
where
    V: VectorEnv,
    A: OnPolicyIdentity + Clone,
    A::State: PartialEq,
{
    fn validate_runtime_state(&self) -> Result<()> {
        self.audit_runtime_memories(|_, _, saved, native| {
            ensure(
                saved == native,
                "temporal runtime memory differs from its exact public history",
            )
        })
    }
    fn audit_runtime_memories(
        &self,
        mut audit: impl FnMut(usize, &A, &A::State, &A::State) -> Result<()>,
    ) -> Result<()> {
        ensure(
            self.version == 1 && !self.poisoned,
            "invalid or poisoned temporal on-policy snapshot",
        )?;
        self.config.validate_kind(self.kind)?;
        ensure(
            self.config.inference_mode == self.collector.inference_mode(),
            "temporal trainer/collector numerical modes differ",
        )?;
        self.collector.validate_state()?;
        self.rollout.validate_snapshot()?;
        self.rollout.validate_limits(
            self.config.capacity,
            self.config.max_fresh,
            self.config.max_metadata_bytes,
            self.config.max_rollout_bytes,
        )?;
        ensure(
            self.completed.len() == self.collector.len()
                && self.rollout.store().observation_space() == &self.collector.observation_space()
                && self.rollout.store().action_space() == self.collector.action_space()
                && self.counters.checked_physical_steps().is_some(),
            "temporal run lane/observation contract differs",
        )?;
        ensure(
            self.rollout.pending_len() == 0,
            "temporal trainer snapshots cannot resume a partially committed collector tick",
        )?;
        if self.kind != TemporalBatchKind::Episodic {
            ensure(
                self.collection_ticks.checked_mul(self.collector.len())
                    == Some(self.rollout.fresh_len())
                    && self.completed.iter().all(|&done| !done),
                "temporal fragment clock differs from committed transitions",
            )?;
        } else {
            let guard = self
                .config
                .collection
                .episode_guard
                .expect("validated episodic collection guard");
            ensure(
                self.collection_ticks <= guard,
                "episodic collection clock exceeds its declared guard",
            )?;
            if self.rollout.fresh_len() == 0 {
                ensure(
                    self.collection_ticks == 0
                        && self.completed.iter().all(|&done| !done)
                        && self.rollout.store().episodes().len() == 0,
                    "empty episodic collection retains episode history, completion flags or a stale clock",
                )?;
            } else {
                let mut fresh_by_episode = BTreeMap::<u64, usize>::new();
                for recording in self.rollout.fresh_recordings() {
                    let count = fresh_by_episode
                        .entry(recording.key.episode_id)
                        .or_default();
                    *count = count.checked_add(1).ok_or_else(|| {
                        Error::State("episodic fresh transition count overflow".into())
                    })?;
                }
                let mut maximum_clock = 0usize;
                for lane in 0..self.collector.len() {
                    let id = self.collector.lane_episode(lane).expect("existing lane");
                    let clock = usize::try_from(
                        self.collector.lane_timestep(lane).expect("existing clock"),
                    )
                    .map_err(|_| Error::State("episodic lane clock is not representable".into()))?;
                    let episode = self.rollout.store().episode(id).ok_or_else(|| {
                        Error::State("episodic lane has no retained fresh episode".into())
                    })?;
                    let fresh_clock =
                        clock.checked_sub(episode.learning_start).ok_or_else(|| {
                            Error::State("episodic lane precedes its learning boundary".into())
                        })?;
                    ensure(
                        fresh_clock > 0
                            && fresh_by_episode.remove(&id) == Some(fresh_clock)
                            && self.completed[lane] == episode.ended()
                            && episode.steps.iter().all(|step| !step.truncated)
                            && (!episode.ended()
                                || episode.steps.last().is_some_and(|step| step.terminated))
                            && (self.completed[lane] || fresh_clock == self.collection_ticks),
                        "episodic completion or lane clock differs from its committed terminal history",
                    )?;
                    maximum_clock = maximum_clock.max(fresh_clock);
                }
                ensure(
                    fresh_by_episode.is_empty() && maximum_clock == self.collection_ticks,
                    "episodic collection contains reset episodes or an inconsistent vector clock",
                )?;
            }
        }
        for lane in 0..self.collector.len() {
            let id = self.collector.lane_episode(lane).expect("existing lane");
            let clock = self.collector.lane_timestep(lane).expect("existing clock") as usize;
            let actor = self.collector.lane_actor(lane).expect("existing actor");
            if let Some(episode) = self.rollout.store().episode(id) {
                ensure(
                    episode.base_timestep == 0
                        && episode.end_timestep() == clock
                        && self.collector.lane_learning_start(lane) == Some(episode.learning_start),
                    "temporal run is missing exact current history",
                )?;
                ensure(
                    self.collector.lane_begun(lane) == Some(true) && clock > episode.learning_start,
                    "retained temporal episode has no committed begin/step",
                )?;
                ensure(
                    Some(episode.ended()) == self.collector.lane_needs_reset(lane),
                    "temporal lane reset flag differs from its recorded episode boundary",
                )?;
                let actions: Vec<_> = episode.steps.iter().map(|s| s.action.clone()).collect();
                let rewards: Vec<_> = episode.steps.iter().map(|s| s.reward.clone()).collect();
                let behavior: Vec<_> = episode.steps.iter().map(|s| s.behavior.clone()).collect();
                self.collector.validate_lane_revision_prefix(
                    lane,
                    crate::collect::stateful::RetainedHistory {
                        base_timestep: 0,
                        previous: None,
                        observations: &episode.observations,
                        actions: &actions,
                        rewards: HistoryRewards::Traces(&rewards),
                    },
                    &behavior,
                )?;
                let mut state = actor.initial_state()?;
                for (t, observation) in episode.observations.iter().enumerate() {
                    let previous = t.checked_sub(1).map(|i| {
                        PreviousInput::from_transition(
                            &episode.steps[i].action,
                            &episode.steps[i].reward,
                            &episode.observations[i],
                        )
                    });
                    state = actor.assimilate(
                        &state,
                        HistoryInput {
                            observation,
                            previous,
                        },
                    )?;
                }
                audit(
                    lane,
                    actor,
                    self.collector.lane_state(lane).expect("existing memory"),
                    &state,
                )?;
            } else if let Some(context) = self.collector.lane_pending_context(lane) {
                let state = replay_actor_history(actor, context.history.inputs())?;
                audit(
                    lane,
                    actor,
                    self.collector.lane_state(lane).expect("existing memory"),
                    &state,
                )?;
            } else if self.collector.lane_needs_reset(lane) != Some(true) {
                ensure(
                    clock == 0 && self.collector.lane_begun(lane) == Some(false),
                    "continuing temporal lane has no history",
                )?;
                self.collector.validate_lane_revision_prefix(
                    lane,
                    crate::collect::stateful::RetainedHistory {
                        base_timestep: 0,
                        previous: None,
                        observations: std::slice::from_ref(&self.collector.observations()[lane]),
                        actions: &[],
                        rewards: HistoryRewards::Totals(&[]),
                    },
                    &[],
                )?;
                let state = actor.assimilate(
                    &actor.initial_state()?,
                    HistoryInput {
                        observation: &self.collector.observations()[lane],
                        previous: None,
                    },
                )?;
                audit(
                    lane,
                    actor,
                    self.collector.lane_state(lane).expect("existing memory"),
                    &state,
                )?;
            } else {
                ensure(
                    clock > 0 && self.collector.lane_begun(lane) == Some(true),
                    "retired temporal lane was never completed",
                )?;
            }
            if self.collector.lane_needs_reset(lane) != Some(true) {
                ensure(
                    actor.revision() == self.collector.latest_revision()
                        && actor.behavior_identity()
                            == self.collector.latest_actor().behavior_identity(),
                    "continuing on-policy lane is pinned to a stale model",
                )?;
            }
        }
        // A shape-valid host float is not proof of behavior provenance. Rebuild
        // the frozen source's own memories and audit all fresh rows, including
        // episodes that ended earlier in the current fragment.
        let actor = self.collector.latest_actor();
        audit_frozen_recordings(actor, self.rollout.store(), self.rollout.fresh_recordings())?;
        if let Some(batch) = &self.pending {
            batch.validate()?;
            ensure(
                batch.revision == self.collector.latest_revision()
                    && batch.kind == self.kind
                    && batch.behavior.behavior_identity()
                        == self.collector.latest_actor().behavior_identity()
                    && batch.len() == self.rollout.fresh_len(),
                "sealed temporal batch differs from live collection",
            )?;
            let expected = self.rollout.seal_with_limits(
                self.collector.latest_actor().clone(),
                self.kind,
                self.config.collection.gamma,
                if self.kind == TemporalBatchKind::Bootstrapped {
                    self.config.collection.gae_lambda
                } else {
                    None
                },
                self.kind == TemporalBatchKind::Bootstrapped
                    && self.config.collection.normalize_advantages,
                &self.config.sequence,
                &self.config.sample_limits,
            )?;
            ensure(
                batch.gamma.to_bits() == expected.gamma.to_bits()
                    && batch.gae_lambda.map(f32::to_bits) == expected.gae_lambda.map(f32::to_bits)
                    && batch.episode_count == expected.episode_count
                    && batch.rows == expected.rows
                    && batch.sequences == expected.sequences,
                "sealed temporal targets or history differ from recorded collection",
            )?;
        }
        self.audited.set(true);
        Ok(())
    }
}

pub(super) fn audit_frozen_recordings<'a, A: OnPolicyIdentity>(
    actor: &A,
    store: &crate::data::temporal::EpisodeStore,
    recordings: impl Iterator<Item = &'a crate::data::temporal_on_policy::CollectedRecording>,
) -> Result<()> {
    let mut fresh: BTreeMap<_, _> = recordings.map(|row| (row.key, row)).collect();
    for episode in store.episodes() {
        let first = crate::core::recording::RecordingKey {
            episode_id: episode.id,
            timestep: 0,
        };
        if fresh
            .range(first..)
            .next()
            .is_none_or(|(key, _)| key.episode_id != episode.id)
        {
            continue;
        }
        let mut state = actor.assimilate(
            &actor.initial_state()?,
            HistoryInput {
                observation: &episode.observations[0],
                previous: None,
            },
        )?;
        for (time, step) in episode.steps.iter().enumerate() {
            let key = crate::core::recording::RecordingKey {
                episode_id: episode.id,
                timestep: time,
            };
            let recorded = fresh.remove(&key);
            if let Some(recorded) = recorded {
                ensure(
                    recorded.revision == actor.revision(),
                    "fresh recording belongs to a stale behavior",
                )?;
                actor.validate_recording(
                    &state,
                    &episode.observations[time],
                    &recorded.action,
                    recorded.latent.as_deref(),
                    &recorded.distribution,
                    recorded.log_prob,
                    recorded.value,
                )?;
            }
            let next = actor.assimilate(
                &state,
                HistoryInput {
                    observation: &episode.observations[time + 1],
                    previous: Some(PreviousInput::from_transition(
                        &step.action,
                        &step.reward,
                        &episode.observations[time],
                    )),
                },
            )?;
            actor.validate_state(&next)?;
            if let Some(recorded) = recorded {
                let expected = if step.terminated {
                    recorded.value.map(|_| 0.)
                } else {
                    actor.state_value(&next, &episode.observations[time + 1])?
                };
                ensure(
                    match (expected, recorded.next_value) {
                        (None, None) => true,
                        (Some(a), Some(b)) => {
                            a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5 * (1. + a.abs())
                        }
                        _ => false,
                    },
                    "recorded successor value differs from exact behavior memory",
                )?;
            }
            state = next;
        }
    }
    ensure(
        fresh.is_empty(),
        "fresh temporal recording has no exact source history",
    )
}

impl<V, A> Snapshot for TemporalOnPolicyTrainer<V, A>
where
    V: VectorEnv + Snapshot,
    A: OnPolicyIdentity + Clone + Snapshot,
    A::State: Serialize + DeserializeOwned + PartialEq,
{
    fn prepare_portable_restore(
        &mut self,
    ) -> Result<crate::core::checkpoint::PortableRestoreReport> {
        let mut report = crate::core::checkpoint::PortableRestoreReport::default();
        let mut candidates = Vec::new();
        // Audit the UNMODIFIED saved memory and all existing structural/history,
        // fresh-record and pending-target invariants before installing candidates.
        self.audit_runtime_memories(|lane, actor, saved, native| {
            let memories = actor.audit_portable_memory(saved, native)?;
            report
                .lanes
                .push(crate::core::checkpoint::PortableLaneAudit { lane, memories });
            candidates.push((lane, native.clone()));
            Ok(())
        })?;
        for (lane, state) in candidates {
            self.collector.install_imported_memory(lane, state)?;
        }
        self.validate_snapshot()?;
        Ok(report)
    }
    fn validate_snapshot(&self) -> Result<()> {
        let result = (|| {
            self.validate_runtime_state()?;
            self.collector.validate_snapshot()?;
            if let Some(batch) = &self.pending {
                batch.behavior.validate_snapshot()?;
            }
            Ok(())
        })();
        self.audited.set(result.is_ok());
        result
    }
    fn validate_learner_record(&self, record: &LearnerRecord) -> Result<()> {
        let binding = TemporalLearnerBinding::from_record(record)?;
        ensure(
            binding.kind == self.kind,
            "temporal learner and run collection kinds differ",
        )?;
        binding.validate_actor(self.collector.latest_actor())
    }
}
