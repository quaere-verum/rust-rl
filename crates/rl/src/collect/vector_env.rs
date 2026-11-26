use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{Env, EpisodeStart, ObservationSpace, Step},
    },
};
use serde::{Deserialize, Serialize};

pub trait VectorEnv {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The common observation schema for every lane.
    fn observation_space(&self) -> ObservationSpace;
    /// Convenience width for dense observations.
    fn dense_observation_dim(&self) -> Option<usize> {
        self.observation_space().dense_width()
    }
    fn action_space(&self) -> &ActionSpace;
    fn reset(&mut self, mask: &[bool], seeds: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>>;
    fn step(&mut self, actions: &[Option<Action>]) -> Result<Vec<Option<Step>>>;
    /// Override to retain the result vector's allocation. Owned environment observations remain explicit.
    fn step_into(&mut self, actions: &[Option<Action>], out: &mut Vec<Option<Step>>) -> Result<()> {
        *out = self.step(actions)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncVectorEnv<E> {
    envs: Vec<E>,
    ended: Vec<bool>,
    #[serde(rename = "width")]
    observations: ObservationSpace,
    space: ActionSpace,
}
impl<E: Env> SyncVectorEnv<E> {
    pub fn new(envs: Vec<E>) -> Result<Self> {
        ensure(!envs.is_empty(), "vector environment cannot be empty")?;
        let observations = envs[0].observation_space();
        observations.validate()?;
        let space = envs[0].action_space().clone();
        space.validate()?;
        ensure(
            envs.iter()
                .all(|e| e.observation_space() == observations && e.action_space() == &space),
            "vector environment spaces differ",
        )?;
        let ended = vec![true; envs.len()];
        Ok(Self {
            envs,
            ended,
            observations,
            space,
        })
    }
    pub fn environments(&self) -> &[E] {
        &self.envs
    }
}
impl<E: Env> VectorEnv for SyncVectorEnv<E> {
    fn len(&self) -> usize {
        self.envs.len()
    }
    fn observation_space(&self) -> ObservationSpace {
        self.observations.clone()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, mask: &[bool], seeds: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
        ensure(
            mask.len() == self.len() && seeds.is_none_or(|s| s.len() == self.len()),
            "reset mask or seeds length mismatch",
        )?;
        let mut result = Vec::with_capacity(self.len());
        for (i, &reset) in mask.iter().enumerate() {
            if reset {
                let start = self.envs[i].reset(seeds.map(|s| s[i]))?;
                self.validate_lane_schema(i)?;
                start
                    .observation
                    .validate_in(&self.observations, &self.space, true)?;
                self.ended[i] = false;
                result.push(Some(start));
            } else {
                result.push(None);
            }
        }
        Ok(result)
    }
    fn step(&mut self, actions: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
        let mut result = Vec::new();
        self.step_into(actions, &mut result)?;
        Ok(result)
    }
    fn step_into(
        &mut self,
        actions: &[Option<Action>],
        result: &mut Vec<Option<Step>>,
    ) -> Result<()> {
        ensure(actions.len() == self.len(), "action batch size mismatch")?;
        if actions
            .iter()
            .zip(&self.ended)
            .any(|(a, &ended)| a.is_some() && ended)
        {
            return Err(Error::State(
                "reset ended vector lanes before stepping".into(),
            ));
        }
        result.clear();
        result.reserve(self.len());
        for action in actions.iter().flatten() {
            self.space.check(action, None)?;
        }
        for (i, action) in actions.iter().enumerate() {
            if let Some(action) = action {
                let s = self.envs[i].step(action)?;
                self.validate_lane_schema(i)?;
                s.observation
                    .validate_in(&self.observations, &self.space, !s.terminated)?;
                s.reward.validate()?;
                self.ended[i] = s.ended();
                result.push(Some(s));
            } else {
                result.push(None);
            }
        }
        Ok(())
    }
}
impl<E: Env + crate::core::checkpoint::Snapshot> crate::core::checkpoint::Snapshot
    for SyncVectorEnv<E>
{
    fn validate_snapshot(&self) -> Result<()> {
        self.space.validate()?;
        self.observations.validate()?;
        ensure(
            !self.envs.is_empty() && self.ended.len() == self.envs.len(),
            "invalid vector snapshot",
        )?;
        for env in &self.envs {
            env.validate_snapshot()?;
            ensure(
                env.observation_space() == self.observations && env.action_space() == &self.space,
                "vector snapshot space mismatch",
            )?;
        }
        Ok(())
    }
}

impl<E: Env> SyncVectorEnv<E> {
    fn validate_lane_schema(&self, lane: usize) -> Result<()> {
        if self.observations.dense_width().is_none() {
            let env = &self.envs[lane];
            ensure(
                env.observation_space() == self.observations && env.action_space() == &self.space,
                "structured vector lane changed its declared schema or grammar",
            )?;
        }
        Ok(())
    }
}
