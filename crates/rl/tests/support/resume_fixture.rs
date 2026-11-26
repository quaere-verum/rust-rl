//! Fixed caller-owned prefix used to exercise continuation at collector reset.
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, ObservationSpace, PublicHistory, ResumeContext, Step},
    },
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct ResumeFixture<E> {
    env: E,
    capture_at: Option<u64>,
    history: Option<PublicHistory>,
    saved: Option<(E, PublicHistory)>,
    pub resume_on_seed: bool,
}

impl<E> ResumeFixture<E> {
    pub fn new(env: E, capture_at: Option<u64>) -> Self {
        Self {
            env,
            capture_at,
            history: None,
            saved: None,
            resume_on_seed: false,
        }
    }
}

impl<E: Env + Clone> Env for ResumeFixture<E> {
    fn observation_space(&self) -> ObservationSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &ActionSpace {
        self.env.action_space()
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if (seed.is_none() || self.resume_on_seed)
            && let Some((saved, history)) = &self.saved
        {
            self.env = saved.clone();
            self.history = None;
            return Ok(EpisodeStart {
                observation: history.observation().clone(),
                context: Some(ResumeContext {
                    history: history.clone(),
                    physical_timestep: history.physical_timestep(),
                }),
            });
        }
        let start = self.env.reset(seed)?;
        self.history = self
            .capture_at
            .filter(|_| self.saved.is_none())
            .map(|_| PublicHistory::new(start.observation.clone()));
        Ok(start)
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        let step = self.env.step(action)?;
        if let Some(history) = self.history.take().filter(|_| !step.ended()) {
            let next = history.append(
                action.clone(),
                step.reward.clone(),
                step.observation.clone(),
            )?;
            if Some(next.timestep()) == self.capture_at {
                self.saved = Some((self.env.clone(), next));
            } else {
                self.history = Some(next);
            }
        }
        Ok(step)
    }
}

impl<E: Env + Clone + Snapshot> Snapshot for ResumeFixture<E> {
    fn validate_snapshot(&self) -> Result<()> {
        self.env.validate_snapshot()?;
        if let Some(history) = &self.history {
            history.validate_in(&self.observation_space(), self.action_space())?;
        }
        if let Some((saved, history)) = &self.saved {
            saved.validate_snapshot()?;
            history.validate_in(&saved.observation_space(), saved.action_space())?;
        }
        Ok(())
    }
}
