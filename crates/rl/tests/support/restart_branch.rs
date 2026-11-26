//! Fully observed delayed branch for checkpoint restart tests.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::prelude::*;
use serde::{Deserialize, Serialize};

/// Four `1` actions reach a public binary cue. Matching the cue then pays 1;
/// the wrong answer pays 0. Before the cue, action 0 exits for 0.05.
/// All nonterminal rewards are zero. The undiscounted optimal start value is 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    depth: usize,
    cue: bool,
    ended: bool,
    rng: ChaCha8Rng,
    space: ActionSpace,
}
impl Default for Branch {
    fn default() -> Self {
        Self::new()
    }
}
impl Branch {
    pub const DEPTH: usize = 4;
    pub fn new() -> Self {
        Self {
            depth: 0,
            cue: false,
            ended: true,
            rng: ChaCha8Rng::seed_from_u64(0),
            space: ActionSpace::Discrete { count: 2 },
        }
    }
    pub fn observation(&self) -> Observation {
        let mut values = vec![0.; Self::DEPTH + 2];
        values[self.depth] = 1.;
        if self.depth == Self::DEPTH {
            values[Self::DEPTH + 1] = if self.cue { 1. } else { -1. };
        }
        Observation::dense(values, Some(vec![!self.ended; 2]))
    }
}
impl Env for Branch {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(Self::DEPTH + 2)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.depth = 0;
        self.cue = false;
        self.ended = false;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.ended {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        let Action::Discrete(action) = *action else {
            unreachable!()
        };
        let reward = if self.depth == Self::DEPTH {
            self.ended = true;
            f32::from(action == usize::from(self.cue))
        } else if action == 0 {
            self.ended = true;
            0.05
        } else {
            self.depth += 1;
            if self.depth == Self::DEPTH {
                self.cue = self.rng.random();
            }
            0.
        };
        Ok(Step {
            observation: self.observation(),
            reward: reward.into(),
            terminated: self.ended,
            truncated: false,
        })
    }
}
impl Snapshot for Branch {
    fn validate_snapshot(&self) -> Result<()> {
        if self.depth > Self::DEPTH
            || self.space != (ActionSpace::Discrete { count: 2 })
            || (self.depth < Self::DEPTH && self.cue)
        {
            return Err(Error::State("invalid branch snapshot".into()));
        }
        Ok(())
    }
}
