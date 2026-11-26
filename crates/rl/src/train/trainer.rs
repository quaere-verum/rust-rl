use super::{
    config::{Budget, OffPolicyConfig, OnPolicyConfig},
    counters::RunCounters,
    observer::{NoObserver, UpdateBoundary, UpdateObserver},
};
use crate::{
    Result,
    collect::{OffPolicyCollector, OnPolicyCollector, vector_env::VectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, OnPolicyActorRef},
        ensure,
        env::ObservationBatch,
        learner::{OffPolicyLearner, OnPolicyLearner, UpdateReport},
    },
    data::{
        replay::{NStepAccumulator, ReplayBatch, ReplayBuffer, ReplayWriter},
        rollout::{OnPolicyBatch, RolloutBuffer},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;
#[derive(Debug, Clone, Default)]
pub struct Progress {
    pub counters: RunCounters,
    pub collection_seconds: f64,
    pub learning_seconds: f64,
    pub elapsed_seconds: f64,
    pub transitions_per_second: f64,
    pub last_update: Option<UpdateReport>,
}
pub type TrainingSummary = Progress;
fn progress(
    c: &RunCounters,
    start: &RunCounters,
    collect: f64,
    learn: f64,
    clock: Instant,
    last: Option<UpdateReport>,
) -> Progress {
    let elapsed = clock.elapsed().as_secs_f64();
    Progress {
        counters: *c,
        collection_seconds: collect,
        learning_seconds: learn,
        elapsed_seconds: elapsed,
        transitions_per_second: (c.env_steps - start.env_steps) as f64 / elapsed.max(1e-9),
        last_update: last,
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnPolicyTrainer<V> {
    collector: OnPolicyCollector<V>,
    config: OnPolicyConfig,
    counters: RunCounters,
    #[serde(default)]
    learner_calls: u64,
    #[serde(default)]
    collection_cycles: u64,
    #[serde(skip)]
    rollout: RolloutBuffer,
    #[serde(skip)]
    batch: OnPolicyBatch,
}
impl<V: VectorEnv> OnPolicyTrainer<V> {
    pub fn new(collector: OnPolicyCollector<V>, config: OnPolicyConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            collector,
            config,
            counters: Default::default(),
            learner_calls: 0,
            collection_cycles: 0,
            rollout: Default::default(),
            batch: Default::default(),
        })
    }
    pub fn counters(&self) -> &RunCounters {
        &self.counters
    }
    pub fn collector(&self) -> &OnPolicyCollector<V> {
        &self.collector
    }
    pub fn config(&self) -> &OnPolicyConfig {
        &self.config
    }
    /// Last completed collection's targets, for read-only diagnostics. This is
    /// ephemeral scratch: it is empty before training and after run restoration,
    /// and is replaced by the next collection. Inspect before starting it.
    pub fn last_batch(&self) -> Option<&OnPolicyBatch> {
        (!self.batch.is_empty()).then_some(&self.batch)
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.config.validate()?;
        self.collector.validate_state()
    }
    pub fn train<L: OnPolicyLearner<Batch = OnPolicyBatch> + ?Sized>(
        &mut self,
        learner: &mut L,
        budget: Budget,
        callback: Option<&mut dyn FnMut(&Progress)>,
    ) -> Result<TrainingSummary> {
        self.train_observed(learner, budget, callback, &mut NoObserver)
    }
    pub fn train_observed<
        L: OnPolicyLearner<Batch = OnPolicyBatch> + ?Sized,
        O: UpdateObserver<L>,
    >(
        &mut self,
        learner: &mut L,
        budget: Budget,
        callback: Option<&mut dyn FnMut(&Progress)>,
        observer: &mut O,
    ) -> Result<TrainingSummary> {
        self.train_observed_until(learner, budget, callback, observer, &mut |_| false)
    }
    /// Opt-in environment-defined budget, checked before collection and after
    /// each complete vector step. Partial fragments receive one ordinary update;
    /// counters retain decision-step units, independently of the stop predicate.
    /// The first true result ends this train call after that update.
    pub fn train_bootstrapped_observed_until<
        L: OnPolicyLearner<Batch = OnPolicyBatch> + ?Sized,
        O: UpdateObserver<L>,
        F: FnMut(&V) -> bool,
    >(
        &mut self,
        learner: &mut L,
        budget: Budget,
        observer: &mut O,
        stop: &mut F,
    ) -> Result<TrainingSummary> {
        ensure(
            matches!(
                learner.collection_actor(),
                OnPolicyActorRef::Bootstrapped(_)
            ),
            "environment-defined stopping requires a bootstrapped collection actor",
        )?;
        self.train_observed_until(learner, budget, None, observer, stop)
    }
    fn train_observed_until<
        L: OnPolicyLearner<Batch = OnPolicyBatch> + ?Sized,
        O: UpdateObserver<L>,
        F: FnMut(&V) -> bool,
    >(
        &mut self,
        learner: &mut L,
        budget: Budget,
        mut callback: Option<&mut dyn FnMut(&Progress)>,
        observer: &mut O,
        stop: &mut F,
    ) -> Result<TrainingSummary> {
        budget.validate()?;
        self.config.validate()?;
        ensure(
            !self.config.normalize_advantages || !learner.normalizes_advantages_in_update(),
            "disable rollout advantage normalization when the learner normalizes minibatches",
        )?;
        let start = self.counters;
        let clock = Instant::now();
        let (mut collection, mut learning) = (0., 0.);
        let mut last = None;
        let mut stopped = false;
        while !stopped
            && !budget.reached(&start, &self.counters)
            && !stop(self.collector.environment())
        {
            let remaining = usize::try_from(budget.remaining_vector_steps(
                &start,
                &self.counters,
                self.collector.environment().len(),
            ))
            .unwrap_or(usize::MAX);
            let t = Instant::now();
            let delta = match learner.collection_actor() {
                OnPolicyActorRef::Pathwise(actor) => {
                    ensure(
                        self.config.episode_guard.is_none(),
                        "pathwise learner requires fixed-step collection",
                    )?;
                    self.collector.collect_pathwise_into(
                        actor,
                        self.config.rollout_steps.min(remaining),
                        &mut self.rollout,
                    )?
                }
                OnPolicyActorRef::Bootstrapped(actor) => {
                    ensure(
                        self.config.episode_guard.is_none(),
                        "bootstrapped learner requires fixed-step collection",
                    )?;
                    self.collector.collect_steps_until_into(
                        actor,
                        self.config.rollout_steps.min(remaining),
                        &mut self.rollout,
                        &mut |env| {
                            stopped = stop(env);
                            stopped
                        },
                    )?
                }
                OnPolicyActorRef::Episodic(actor) => {
                    let guard = self.config.episode_guard.ok_or_else(|| {
                        crate::Error::Config("episodic learner requires an episode guard".into())
                    })?;
                    self.collector.collect_episodes_into(
                        actor,
                        guard.min(remaining),
                        &mut self.rollout,
                    )?
                }
            };
            self.counters.add_collection(delta);
            self.collection_cycles += 1;
            self.rollout.to_batch_into(
                self.config.gamma,
                self.config.gae_lambda,
                self.config.normalize_advantages,
                &mut self.batch,
            )?;
            collection += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let report = learner.update(&self.batch)?;
            learning += t.elapsed().as_secs_f64();
            self.counters.updates += report.updates;
            self.learner_calls += 1;
            observer.after_update(
                learner,
                UpdateBoundary {
                    counters: &self.counters,
                    learner_call: self.learner_calls,
                    collection_cycle: self.collection_cycles,
                    report: &report,
                    collection_seconds: collection,
                    learning_seconds: learning,
                },
            )?;
            last = Some(report);
            if let Some(cb) = &mut callback {
                cb(&progress(
                    &self.counters,
                    &start,
                    collection,
                    learning,
                    clock,
                    last.clone(),
                ));
            }
        }
        Ok(progress(
            &self.counters,
            &start,
            collection,
            learning,
            clock,
            last,
        ))
    }
}
#[derive(Debug, Clone)]
pub struct RandomActor {
    pub space: ActionSpace,
}
impl Actor for RandomActor {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        self.space.validate()?;
        out.resize_with(obs.len(), || Action::Discrete(0));
        for (o, slot) in obs.iter().zip(out) {
            o.validate(o.dense_values().len(), &self.space, true)?;
            match &self.space {
                ActionSpace::Discrete { count } => {
                    let mut valid = (0..*count).filter(|&i| o.dense_mask().is_none_or(|m| m[i]));
                    let index = if mode == ActMode::Sample {
                        rng.random_range(0..valid.clone().count())
                    } else {
                        0
                    };
                    *slot = Action::Discrete(valid.nth(index).expect("validated live mask"));
                }
                ActionSpace::Box { low, high } => {
                    if !matches!(slot, Action::Continuous(_)) {
                        *slot = Action::Continuous(vec![]);
                    }
                    let Action::Continuous(v) = slot else {
                        unreachable!()
                    };
                    v.resize(low.len(), 0.);
                    for ((x, &l), &h) in v.iter_mut().zip(low).zip(high) {
                        *x = if mode == ActMode::Sample {
                            rng.random_range(l..h)
                        } else {
                            l + (h - l) * 0.5
                        };
                    }
                }
                ActionSpace::Conditional(_) => {
                    return Err(crate::Error::Action(
                        "dense replay warmup requires numerical actions".into(),
                    ));
                }
                ActionSpace::Unbounded { dimensions } => {
                    if !matches!(slot, Action::Continuous(_)) {
                        *slot = Action::Continuous(vec![]);
                    }
                    let Action::Continuous(v) = slot else {
                        unreachable!()
                    };
                    v.resize(*dimensions, 0.);
                    for x in v {
                        *x = if mode == ActMode::Sample {
                            rng.sample(rand_distr::StandardNormal)
                        } else {
                            0.
                        };
                    }
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffPolicyTrainer<V> {
    collector: OffPolicyCollector<V>,
    config: OffPolicyConfig,
    counters: RunCounters,
    #[serde(default)]
    learner_calls: u64,
    #[serde(default)]
    collection_cycles: u64,
    replay: ReplayBuffer,
    pending: Option<NStepAccumulator>,
    sampling_rng: ChaCha8Rng,
    update_credit: f64,
    #[serde(skip)]
    batch_scratch: ReplayBatch,
}
impl<V: VectorEnv> OffPolicyTrainer<V> {
    pub fn new(
        collector: OffPolicyCollector<V>,
        config: OffPolicyConfig,
        seed: u64,
    ) -> Result<Self> {
        config.validate()?;
        let env = collector.environment();
        let replay = ReplayBuffer::new(
            config.replay_capacity,
            env.observation_space().require_dense_width()?,
            env.action_space().clone(),
        )?;
        let pending = if config.n_step > 1 {
            Some(NStepAccumulator::new(
                config.n_step,
                config.gamma,
                env.len(),
            )?)
        } else {
            None
        };
        Ok(Self {
            collector,
            config,
            counters: Default::default(),
            learner_calls: 0,
            collection_cycles: 0,
            replay,
            pending,
            sampling_rng: ChaCha8Rng::seed_from_u64(seed),
            update_credit: 0.,
            batch_scratch: Default::default(),
        })
    }
    pub fn counters(&self) -> &RunCounters {
        &self.counters
    }
    pub fn collector(&self) -> &OffPolicyCollector<V> {
        &self.collector
    }
    pub fn replay(&self) -> &ReplayBuffer {
        &self.replay
    }
    pub fn config(&self) -> &OffPolicyConfig {
        &self.config
    }
    pub fn update_credit(&self) -> f64 {
        self.update_credit
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.config.validate()?;
        self.collector.validate_state()?;
        let env = self.collector.environment();
        let width = env.observation_space().require_dense_width()?;
        self.replay.validate_state()?;
        ensure(
            self.replay.observation_dim() == width
                && self.replay.action_space() == env.action_space()
                && self.replay.capacity() == self.config.replay_capacity
                && self.update_credit.is_finite()
                && self.update_credit >= 0.
                && self.pending.is_some() == (self.config.n_step > 1),
            "invalid trainer snapshot",
        )?;
        if let Some(p) = &self.pending {
            p.check_config(self.config.n_step, self.config.gamma)?;
            p.validate_state(width, env.action_space(), env.len())?;
        }
        Ok(())
    }
    pub fn train<L: OffPolicyLearner<Batch = ReplayBatch> + ?Sized>(
        &mut self,
        learner: &mut L,
        budget: Budget,
        callback: Option<&mut dyn FnMut(&Progress)>,
    ) -> Result<TrainingSummary> {
        self.train_observed(learner, budget, callback, &mut NoObserver)
    }
    pub fn train_observed<
        L: OffPolicyLearner<Batch = ReplayBatch> + ?Sized,
        O: UpdateObserver<L>,
    >(
        &mut self,
        learner: &mut L,
        budget: Budget,
        mut callback: Option<&mut dyn FnMut(&Progress)>,
        observer: &mut O,
    ) -> Result<TrainingSummary> {
        budget.validate()?;
        self.config.validate()?;
        ensure(
            learner.actor().action_space() == self.collector.environment().action_space(),
            "learner/environment action spaces differ",
        )?;
        let start = self.counters;
        let clock = Instant::now();
        let (mut collection, mut learning) = (0., 0.);
        let mut last = None;
        let random = RandomActor {
            space: self.collector.environment().action_space().clone(),
        };
        self.config
            .validate_requirements(learner.replay_requirements())?;
        while !budget.reached(&start, &self.counters) {
            while self.update_credit >= 1. && !self.replay.is_empty() {
                let t = Instant::now();
                self.replay.sample_into(
                    self.config.batch_size,
                    &mut self.sampling_rng,
                    &mut self.batch_scratch,
                )?;
                let report = learner.update(&self.batch_scratch)?;
                ensure(report.updates > 0, "off-policy learner did not update")?;
                self.counters.updates += report.updates;
                self.update_credit -= 1.;
                self.learner_calls += 1;
                learning += t.elapsed().as_secs_f64();
                observer.after_update(
                    learner,
                    UpdateBoundary {
                        counters: &self.counters,
                        learner_call: self.learner_calls,
                        collection_cycle: self.collection_cycles,
                        report: &report,
                        collection_seconds: collection,
                        learning_seconds: learning,
                    },
                )?;
                last = Some(report);
                if budget.reached(&start, &self.counters) {
                    break;
                }
            }
            if budget.reached(&start, &self.counters) {
                break;
            }
            learner.set_exploration(self.config.exploration.value(&self.counters)? as f32)?;
            let before = self.counters.env_steps;
            let t = Instant::now();
            let warmup = before < self.config.warmup_transitions;
            let mut writer = ReplayWriter::new(
                &mut self.replay,
                self.pending.as_mut(),
                self.config.gamma,
                self.collector.environment().len(),
            )?;
            let delta = self.collector.collect_into(
                if warmup { &random } else { learner.actor() },
                1,
                &mut writer,
            )?;
            self.counters.add_collection(delta);
            self.collection_cycles += 1;
            self.update_credit += super::schedule::collection_credit(
                before,
                self.counters.env_steps,
                self.config.warmup_transitions,
                self.config.updates_per_transition,
            );
            collection += t.elapsed().as_secs_f64();
            if let Some(cb) = &mut callback {
                cb(&progress(
                    &self.counters,
                    &start,
                    collection,
                    learning,
                    clock,
                    last.clone(),
                ));
            }
        }
        Ok(progress(
            &self.counters,
            &start,
            collection,
            learning,
            clock,
            last,
        ))
    }
}
impl<V: VectorEnv + crate::core::checkpoint::Snapshot> crate::core::checkpoint::Snapshot
    for OnPolicyTrainer<V>
{
    fn validate_snapshot(&self) -> Result<()> {
        self.collector.validate_snapshot()?;
        self.validate_state()
    }
}
impl<V: VectorEnv + crate::core::checkpoint::Snapshot> crate::core::checkpoint::Snapshot
    for OffPolicyTrainer<V>
{
    fn validate_snapshot(&self) -> Result<()> {
        self.collector.validate_snapshot()?;
        self.validate_state()
    }
    fn validate_learner_record(
        &self,
        record: &crate::core::checkpoint::LearnerRecord,
    ) -> Result<()> {
        self.config
            .validate_requirements(record.replay_requirements)
    }
}
impl<V: VectorEnv> OnPolicyTrainer<V> {
    /// Recover after a collection failure. Counters retain previously completed collections.
    pub fn reset_environments(&mut self) -> Result<()> {
        self.collector.reset_all()
    }
}
impl<V: VectorEnv> OffPolicyTrainer<V> {
    /// Recover after a collection failure, discarding incomplete n-step trajectories.
    pub fn reset_environments(&mut self) -> Result<()> {
        self.collector.reset_all()?;
        if let Some(p) = &mut self.pending {
            p.clear();
        }
        Ok(())
    }
}
