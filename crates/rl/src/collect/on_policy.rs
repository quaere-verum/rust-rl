use super::{CollectionDelta, driver::LaneDriver, vector_env::VectorEnv};
use crate::{
    Error, Result,
    core::{
        actor::{
            EpisodeActor, EpisodeBehavior, PathwiseActor, PathwiseBehavior, RolloutActor,
            RolloutBehavior,
        },
        ensure,
        env::ObservationBatch,
        transition::TransitionView,
    },
    data::rollout::{RolloutBuffer, RolloutMetadata},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone)]
struct Scratch {
    pathwise: PathwiseBehavior,
    rollout: RolloutBehavior,
    episode: EpisodeBehavior,
    previous: Vec<Option<usize>>,
    boundaries: Vec<usize>,
    values: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnPolicyCollector<V> {
    driver: LaneDriver<V>,
    #[serde(skip)]
    scratch: Scratch,
}
impl<V: VectorEnv> OnPolicyCollector<V> {
    pub fn new(env: V, seed: u64) -> Result<Self> {
        Ok(Self {
            driver: LaneDriver::new(env, seed)?.0,
            scratch: Scratch::default(),
        })
    }
    /// Initialize each environment with its own supplied seed. Unlike `new`,
    /// this consumes no initial reset-seed draws from the action RNG. Later
    /// resets use `reset(None)` and remain owned by the environments.
    pub fn new_with_reset_seeds(env: V, action_seed: u64, reset_seeds: &[u64]) -> Result<Self> {
        Ok(Self {
            driver: LaneDriver::new_with_reset_seeds(env, action_seed, reset_seeds)?.0,
            scratch: Scratch::default(),
        })
    }
    pub fn environment(&self) -> &V {
        &self.driver.env
    }
    pub fn reset_all(&mut self) -> Result<()> {
        self.driver.reset_all().map(|_| ())
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.driver.validate_state()
    }
    pub fn collect_pathwise_into(
        &mut self,
        actor: &dyn PathwiseActor,
        steps: usize,
        out: &mut RolloutBuffer,
    ) -> Result<CollectionDelta> {
        let result = self.collect_pathwise(actor, steps, out);
        if result.is_err() {
            out.invalidate();
            self.driver.fail();
        }
        result
    }
    fn collect_pathwise(
        &mut self,
        actor: &dyn PathwiseActor,
        steps: usize,
        out: &mut RolloutBuffer,
    ) -> Result<CollectionDelta> {
        ensure(
            steps > 0 && actor.action_space() == self.driver.env.action_space(),
            "invalid pathwise budget or action space",
        )?;
        let n = self.driver.len();
        actor.begin_collection()?;
        let revision = actor.revision();
        out.start_pathwise(n, steps, revision)?;
        self.driver.begin(false)?;
        let mut delta = CollectionDelta::default();
        for _ in 0..steps {
            self.driver.prepare()?;
            let observations =
                ObservationBatch::selected(&self.driver.observations, &self.driver.lanes)?;
            actor.sample_pathwise_into(
                observations,
                &mut self.driver.rng,
                &mut self.scratch.pathwise,
            )?;
            let b = &self.scratch.pathwise;
            ensure(
                actor.revision() == revision
                    && [b.actions.len(), b.log_probs.len(), b.latents.len()]
                        .iter()
                        .all(|&len| len == n)
                    && b.log_probs.iter().all(|x| x.is_finite())
                    && b.latents.iter().flatten().flatten().all(|x| x.is_finite()),
                "invalid pathwise actor metadata or changing behavior revision",
            )?;
            self.driver.step(&mut self.scratch.pathwise.actions)?;
            for lane in 0..n {
                let s = self.driver.results[lane]
                    .as_ref()
                    .expect("validated active result");
                let b = &self.scratch.pathwise;
                out.record(
                    Some(TransitionView {
                        observation: &self.driver.observations[lane],
                        action: &b.actions[lane],
                        reward: &s.reward,
                        next_observation: &s.observation,
                        terminated: s.terminated,
                        truncated: s.truncated,
                    }),
                    self.driver.episode_steps[lane],
                    b.latents[lane].as_ref(),
                );
                let RolloutMetadata::Pathwise { log_probs, .. } = &mut out.metadata else {
                    unreachable!()
                };
                log_probs.push(b.log_probs[lane]);
            }
            delta.add(self.driver.commit_step(false));
        }
        Ok(delta)
    }
    pub fn collect_steps_into(
        &mut self,
        actor: &dyn RolloutActor,
        steps: usize,
        out: &mut RolloutBuffer,
    ) -> Result<CollectionDelta> {
        self.collect_steps_until_into(actor, steps, out, &mut |_| false)
    }
    /// Inspect the environment after each committed vector step. A stop keeps
    /// real episode boundaries and bootstraps the partial fragment normally.
    /// At least one vector step is collected; callers check any initial stop.
    pub fn collect_steps_until_into<F: FnMut(&V) -> bool>(
        &mut self,
        actor: &dyn RolloutActor,
        max_steps: usize,
        out: &mut RolloutBuffer,
        stop: &mut F,
    ) -> Result<CollectionDelta> {
        let result = self.collect_steps(actor, max_steps, out, stop);
        if result.is_err() {
            out.invalidate();
            self.driver.fail();
        }
        result
    }
    fn collect_steps<F: FnMut(&V) -> bool>(
        &mut self,
        actor: &dyn RolloutActor,
        steps: usize,
        out: &mut RolloutBuffer,
        stop: &mut F,
    ) -> Result<CollectionDelta> {
        ensure(
            steps > 0 && actor.action_space() == self.driver.env.action_space(),
            "invalid collection budget or action space",
        )?;
        let n = self.driver.len();
        out.start(n, steps, false)?;
        self.driver.begin(false)?;
        self.scratch.previous.resize(n, None);
        self.scratch.previous.fill(None);
        self.scratch.boundaries.clear();
        let mut delta = CollectionDelta::default();
        for _ in 0..steps {
            self.driver.prepare()?;
            let observations =
                ObservationBatch::selected(&self.driver.observations, &self.driver.lanes)?;
            actor.sample_rollout_into(
                observations,
                &mut self.driver.rng,
                &mut self.scratch.rollout,
            )?;
            let behavior = &self.scratch.rollout;
            ensure(
                [
                    behavior.actions.len(),
                    behavior.log_probs.len(),
                    behavior.values.len(),
                    behavior.latents.len(),
                ]
                .iter()
                .all(|&len| len == n)
                    && behavior
                        .log_probs
                        .iter()
                        .chain(&behavior.values)
                        .all(|x| x.is_finite())
                    && behavior
                        .latents
                        .iter()
                        .flatten()
                        .flatten()
                        .all(|x| x.is_finite()),
                "invalid rollout actor metadata",
            )?;
            self.driver.step(&mut self.scratch.rollout.actions)?;
            for lane in 0..n {
                let s = self.driver.results[lane]
                    .as_ref()
                    .expect("validated active result");
                let b = &self.scratch.rollout;
                let index = out.record(
                    Some(TransitionView {
                        observation: &self.driver.observations[lane],
                        action: &b.actions[lane],
                        reward: &s.reward,
                        next_observation: &s.observation,
                        terminated: s.terminated,
                        truncated: s.truncated,
                    }),
                    self.driver.episode_steps[lane],
                    b.latents[lane].as_ref(),
                );
                let RolloutMetadata::Bootstrapped {
                    log_probs,
                    values,
                    next_values,
                } = &mut out.metadata
                else {
                    unreachable!()
                };
                if let Some(previous) = self.scratch.previous[lane].take() {
                    next_values[previous] = b.values[lane];
                }
                log_probs.push(b.log_probs[lane]);
                values.push(b.values[lane]);
                next_values.push(0.);
                if !s.terminated {
                    if s.truncated {
                        self.scratch.boundaries.push(index);
                    } else {
                        self.scratch.previous[lane] = Some(index);
                    }
                }
            }
            delta.add(self.driver.commit_step(false));
            if stop(&self.driver.env) {
                break;
            }
        }
        self.scratch
            .boundaries
            .extend(self.scratch.previous.iter().flatten().copied());
        if !self.scratch.boundaries.is_empty() {
            // All current-state calls are complete; only final observations need an extra forward.
            let refs: Vec<_> = self
                .scratch
                .boundaries
                .iter()
                .map(|&i| &out.rows[i].transition.next_observation)
                .collect();
            actor.values_into(ObservationBatch::from_refs(&refs), &mut self.scratch.values)?;
            ensure(
                self.scratch.values.len() == refs.len()
                    && self.scratch.values.iter().all(|x| x.is_finite()),
                "invalid bootstrap value batch",
            )?;
            let RolloutMetadata::Bootstrapped { next_values, .. } = &mut out.metadata else {
                unreachable!()
            };
            for (&i, &v) in self.scratch.boundaries.iter().zip(&self.scratch.values) {
                next_values[i] = v;
            }
        }
        Ok(delta)
    }
    pub fn collect_episodes_into(
        &mut self,
        actor: &dyn EpisodeActor,
        guard: usize,
        out: &mut RolloutBuffer,
    ) -> Result<CollectionDelta> {
        let result = self.collect_episodes(actor, guard, out);
        if result.is_err() {
            out.invalidate();
            self.driver.fail();
        }
        result
    }
    fn collect_episodes(
        &mut self,
        actor: &dyn EpisodeActor,
        guard: usize,
        out: &mut RolloutBuffer,
    ) -> Result<CollectionDelta> {
        ensure(
            guard > 0 && actor.action_space() == self.driver.env.action_space(),
            "invalid episode guard or action space",
        )?;
        let n = self.driver.len();
        out.start(n, guard, true)?;
        self.driver.begin(true)?;
        let mut delta = CollectionDelta::default();
        let mut with_baseline = None;
        for _ in 0..guard {
            self.driver.prepare()?;
            let observations =
                ObservationBatch::selected(&self.driver.observations, &self.driver.lanes)?;
            actor.sample_episode_into(
                observations,
                &mut self.driver.rng,
                &mut self.scratch.episode,
            )?;
            let b = &self.scratch.episode;
            let active = self.driver.lanes.len();
            ensure(
                b.actions.len() == active
                    && b.latents.len() == active
                    && b.latents.iter().flatten().flatten().all(|x| x.is_finite())
                    && b.baseline_values
                        .as_ref()
                        .is_none_or(|v| v.len() == active && v.iter().all(|x| x.is_finite())),
                "invalid episode actor metadata",
            )?;
            let baseline = b.baseline_values.is_some();
            if let Some(previous) = with_baseline {
                ensure(
                    previous == baseline,
                    "episode baseline capability changed during collection",
                )?;
            } else {
                with_baseline = Some(baseline);
                if let RolloutMetadata::Episodic { baseline_values } = &mut out.metadata {
                    if baseline {
                        baseline_values.get_or_insert_with(Vec::new).clear();
                    } else {
                        *baseline_values = None;
                    }
                }
            }
            self.driver.step(&mut self.scratch.episode.actions)?;
            let mut j = 0;
            let mut truncated = false;
            for lane in 0..n {
                let value = if let Some(s) = &self.driver.results[lane] {
                    let b = &self.scratch.episode;
                    out.record(
                        Some(TransitionView {
                            observation: &self.driver.observations[lane],
                            action: &b.actions[j],
                            reward: &s.reward,
                            next_observation: &s.observation,
                            terminated: s.terminated,
                            truncated: s.truncated,
                        }),
                        self.driver.episode_steps[lane],
                        b.latents[j].as_ref(),
                    );
                    let value = b.baseline_values.as_ref().map_or(0., |v| v[j]);
                    j += 1;
                    truncated |= s.truncated && !s.terminated;
                    value
                } else {
                    out.record(None, 0, None);
                    0.
                };
                if let RolloutMetadata::Episodic {
                    baseline_values: Some(v),
                } = &mut out.metadata
                {
                    v.push(value);
                }
            }
            delta.add(self.driver.commit_step(true));
            if truncated {
                return Err(Error::State(
                    "Monte Carlo collection encountered external truncation".into(),
                ));
            }
            if self.driver.active.iter().all(|x| !*x) {
                return Ok(delta);
            }
        }
        Err(Error::State(
            "complete-episode collection guard exhausted".into(),
        ))
    }
}
impl<V: VectorEnv + crate::core::checkpoint::Snapshot> crate::core::checkpoint::Snapshot
    for OnPolicyCollector<V>
{
    fn validate_snapshot(&self) -> Result<()> {
        self.driver.env.validate_snapshot()?;
        self.validate_state()
    }
}
