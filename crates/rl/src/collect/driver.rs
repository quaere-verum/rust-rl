use super::{CollectionDelta, vector_env::VectorEnv};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace, ResumeContext, Step},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone)]
struct Scratch {
    reset: Vec<bool>,
    vector_actions: Vec<Option<Action>>,
}
/// Only lane mechanics live here. No policy metadata or storage/learning choices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct LaneDriver<V> {
    pub(super) env: V,
    pub(super) observations: Vec<Observation>,
    pub(super) needs_reset: Vec<bool>,
    pub(super) episode_steps: Vec<u64>,
    pub(super) rng: ChaCha8Rng,
    lanes_count: usize,
    #[serde(rename = "width")]
    observation_space: ObservationSpace,
    space: ActionSpace,
    poisoned: bool,
    #[serde(skip)]
    pub(super) active: Vec<bool>,
    #[serde(skip)]
    pub(super) lanes: Vec<usize>,
    #[serde(skip)]
    pub(super) results: Vec<Option<Step>>,
    #[serde(skip)]
    scratch: Scratch,
}
fn contract(operation: &'static str, lane: Option<usize>, reason: impl Into<String>) -> Error {
    Error::VectorEnvContract {
        operation,
        lane,
        reason: reason.into(),
    }
}

pub(super) type ResumeContexts = Vec<(usize, ResumeContext)>;

fn split_starts(starts: Vec<Option<EpisodeStart>>) -> (Vec<Observation>, ResumeContexts) {
    let mut contexts = Vec::new();
    let observations = starts
        .into_iter()
        .enumerate()
        .map(|(lane, start)| {
            let EpisodeStart {
                observation,
                context,
            } = start.expect("validated reset");
            if let Some(context) = context {
                contexts.push((lane, context));
            }
            observation
        })
        .collect();
    (observations, contexts)
}

impl<V: VectorEnv> LaneDriver<V> {
    pub(super) fn new(env: V, seed: u64) -> Result<(Self, ResumeContexts)> {
        Self::initialize(env, seed, None)
    }
    pub(super) fn new_with_reset_seeds(
        env: V,
        seed: u64,
        reset_seeds: &[u64],
    ) -> Result<(Self, ResumeContexts)> {
        Self::initialize(env, seed, Some(reset_seeds))
    }
    fn initialize(
        mut env: V,
        seed: u64,
        reset_seeds: Option<&[u64]>,
    ) -> Result<(Self, ResumeContexts)> {
        ensure(!env.is_empty(), "collector requires nonempty environments")?;
        env.action_space().validate()?;
        let lanes_count = env.len();
        let observation_space = env.observation_space();
        observation_space.validate()?;
        let space = env.action_space().clone();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let generated;
        let seeds = if let Some(seeds) = reset_seeds {
            ensure(
                seeds.len() == lanes_count,
                "initial reset seed count must match lanes",
            )?;
            seeds
        } else {
            // Preserve the default constructor's historical action-RNG offset.
            generated = (0..lanes_count).map(|_| rng.random()).collect::<Vec<u64>>();
            &generated
        };
        let reset = vec![true; lanes_count];
        let result = env.reset(&reset, Some(seeds))?;
        Self::check_reset(&result, &reset, &observation_space, &space)?;
        if env.len() != lanes_count
            || env.observation_space() != observation_space
            || env.action_space() != &space
        {
            return Err(contract(
                "reset",
                None,
                "environment changed its declared shape",
            ));
        }
        let (observations, contexts) = split_starts(result);
        Ok((
            Self {
                env,
                observations,
                needs_reset: vec![false; lanes_count],
                episode_steps: vec![0; lanes_count],
                rng,
                lanes_count,
                observation_space,
                space,
                poisoned: false,
                active: vec![],
                lanes: vec![],
                results: vec![],
                scratch: Scratch::default(),
            },
            contexts,
        ))
    }
    fn check_reset(
        result: &[Option<EpisodeStart>],
        mask: &[bool],
        observation_space: &ObservationSpace,
        space: &ActionSpace,
    ) -> Result<()> {
        if result.len() != mask.len() {
            return Err(contract(
                "reset",
                None,
                "result count differs from lane count",
            ));
        }
        for (i, (obs, &requested)) in result.iter().zip(mask).enumerate() {
            if obs.is_some() != requested {
                return Err(contract(
                    "reset",
                    Some(i),
                    "returned lanes differ from requested mask",
                ));
            }
            if let Some(obs) = obs {
                obs.validate_in(observation_space, space)
                    .map_err(|e| contract("reset", Some(i), e.to_string()))?;
                if usize::try_from(obs.decision_timestep()).is_err() {
                    return Err(contract(
                        "reset",
                        Some(i),
                        "decision timestep exceeds lane index range",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn validate_state(&self) -> Result<()> {
        ensure(
            !self.poisoned,
            "collector requires a full reset after an environment failure",
        )?;
        ensure(
            self.lanes_count > 0
                && self.env.len() == self.lanes_count
                && self.env.observation_space() == self.observation_space
                && self.env.action_space() == &self.space
                && self.observations.len() == self.lanes_count
                && self.needs_reset.len() == self.lanes_count
                && self.episode_steps.len() == self.lanes_count,
            "collector snapshot dimensions or spaces differ",
        )?;
        self.space.validate()?;
        self.observation_space.validate()?;
        for (obs, reset) in self.observations.iter().zip(&self.needs_reset) {
            obs.validate_in(&self.observation_space, &self.space, !reset)?;
        }
        Ok(())
    }
    pub(super) fn begin(&mut self, episodic: bool) -> Result<()> {
        ensure(
            !self.poisoned,
            "collector requires a full reset after an environment failure",
        )?;
        self.active.resize(self.lanes_count, true);
        self.active.fill(true);
        self.scratch.reset.resize(self.lanes_count, false);
        self.scratch
            .vector_actions
            .resize_with(self.lanes_count, || None);
        if episodic {
            self.needs_reset.fill(true);
        }
        Ok(())
    }
    pub(super) fn reset_all(&mut self) -> Result<ResumeContexts> {
        let mask = vec![true; self.lanes_count];
        self.poisoned = true;
        let result = self.env.reset(&mask, None)?;
        Self::check_reset(&result, &mask, &self.observation_space, &self.space)?;
        if self.env.len() != self.lanes_count
            || self.env.observation_space() != self.observation_space
            || self.env.action_space() != &self.space
        {
            return Err(contract(
                "reset",
                None,
                "environment changed its declared shape",
            ));
        }
        let (observations, contexts) = split_starts(result);
        self.observations = observations;
        self.needs_reset.fill(false);
        self.episode_steps.fill(0);
        self.poisoned = false;
        Ok(contexts)
    }
    pub(super) fn prepare(&mut self) -> Result<ResumeContexts> {
        ensure(!self.poisoned, "collector requires full reset")?;
        for i in 0..self.lanes_count {
            self.scratch.reset[i] = self.active[i] && self.needs_reset[i];
        }
        let mut contexts = Vec::new();
        if self.scratch.reset.iter().any(|x| *x) {
            self.poisoned = true;
            let result = self.env.reset(&self.scratch.reset, None)?;
            Self::check_reset(
                &result,
                &self.scratch.reset,
                &self.observation_space,
                &self.space,
            )?;
            if self.env.len() != self.lanes_count
                || self.env.observation_space() != self.observation_space
                || self.env.action_space() != &self.space
            {
                return Err(contract(
                    "reset",
                    None,
                    "environment changed its declared shape",
                ));
            }
            for (i, start) in result.into_iter().enumerate() {
                if let Some(EpisodeStart {
                    observation,
                    context,
                }) = start
                {
                    if let Some(context) = context {
                        contexts.push((i, context));
                    }
                    self.observations[i] = observation;
                    self.needs_reset[i] = false;
                    self.episode_steps[i] = 0;
                }
            }
            self.poisoned = false;
        }
        self.lanes.clear();
        self.lanes
            .extend((0..self.lanes_count).filter(|&i| self.active[i]));
        Ok(contexts)
    }
    pub(super) fn step(&mut self, actions: &mut [Action]) -> Result<()> {
        ensure(
            actions.len() == self.lanes.len(),
            "actor output count differs from active lanes",
        )?;
        for (&i, action) in self.lanes.iter().zip(actions.iter()) {
            self.observations[i].check_action(&self.space, action)?;
        }
        // Move action buffers into the vector call and back; no continuous payload copy.
        for i in 0..self.lanes_count {
            if !self.active[i] {
                self.scratch.vector_actions[i] = None;
            }
        }
        for (j, &i) in self.lanes.iter().enumerate() {
            std::mem::swap(
                &mut actions[j],
                self.scratch.vector_actions[i].get_or_insert(Action::Discrete(0)),
            );
        }
        self.poisoned = true;
        let result = self
            .env
            .step_into(&self.scratch.vector_actions, &mut self.results);
        for (j, &i) in self.lanes.iter().enumerate() {
            std::mem::swap(
                &mut actions[j],
                self.scratch.vector_actions[i]
                    .as_mut()
                    .expect("active action"),
            );
        }
        result?;
        if self.results.len() != self.lanes_count {
            return Err(contract(
                "step",
                None,
                "result count differs from lane count",
            ));
        }
        if self.env.len() != self.lanes_count
            || self.env.observation_space() != self.observation_space
            || self.env.action_space() != &self.space
        {
            return Err(contract(
                "step",
                None,
                "environment changed its declared shape",
            ));
        }
        for (i, result) in self.results.iter().enumerate() {
            if result.is_some() != self.active[i] {
                return Err(contract(
                    "step",
                    Some(i),
                    "returned lanes differ from active mask",
                ));
            }
            if let Some(s) = result {
                s.observation
                    .validate_in(&self.observation_space, &self.space, !s.terminated)
                    .map_err(|e| contract("step", Some(i), e.to_string()))?;
                s.reward
                    .validate()
                    .map_err(|e| contract("step", Some(i), e.to_string()))?;
            }
        }
        // Remain poisoned until the caller persists this complete validated step.
        Ok(())
    }
    pub(super) fn commit_step(&mut self, episodic: bool) -> CollectionDelta {
        let mut delta = CollectionDelta {
            vector_steps: 1,
            ..Default::default()
        };
        for (i, result) in self.results.iter_mut().enumerate() {
            if let Some(s) = result.take() {
                let ended = s.ended();
                self.observations[i] = s.observation;
                self.episode_steps[i] += 1;
                delta.env_steps += 1;
                delta.additional_physical_steps += s.reward.duration() - 1;
                if ended {
                    self.needs_reset[i] = true;
                    delta.episodes += 1;
                    if episodic {
                        self.active[i] = false;
                    }
                }
            }
        }
        self.poisoned = false;
        delta
    }
    pub(super) fn len(&self) -> usize {
        self.lanes_count
    }
    pub(super) fn fail(&mut self) {
        self.poisoned = true;
    }
}
