//! Controlled continuous memory task with an initial balanced hidden target.
//!
//! The final reward is `-(action - target)^2`. Earlier actions are ignored and
//! never change observations: the environment supplies no external action memory.
//! A stateful policy may still condition on its own factual previous actions.
//! The optional marker is an ordinary nonterminal observation, not a reset.
//!
//! Over balanced targets, remembering the cue gives reward zero; the best
//! observation-only constant is zero with mean reward -1. An independent
//! Uniform[-2,2] final action has mean reward -7/3. The fully observed control
//! preserves the target cue throughout the episode.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        ensure,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const MAX_DELAY: usize = 128;

fn physical_space() -> ActionSpace {
    ActionSpace::Box {
        low: vec![-2.],
        high: vec![2.],
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenTarget {
    version: u32,
    delay: usize,
    marker: Option<usize>,
    fully_observed: bool,
    tick: usize,
    target: i8,
    ended: bool,
    space: ActionSpace,
    rng: ChaCha8Rng,
}

impl HiddenTarget {
    /// The decision observation arrives after `delay` ordinary transitions. Its
    /// action terminates the episode, giving `delay + 1` actions like DelayedBit.
    pub fn new(
        delay: usize,
        marker: Option<usize>,
        fully_observed: bool,
        seed: u64,
    ) -> Result<Self> {
        ensure(
            (2..=MAX_DELAY).contains(&delay) && marker.is_none_or(|m| m > 0 && m < delay),
            "invalid hidden-target horizon or marker",
        )?;
        Ok(Self {
            version: 1,
            delay,
            marker,
            fully_observed,
            tick: 0,
            target: -1,
            ended: true,
            space: physical_space(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    pub fn delay(&self) -> usize {
        self.delay
    }

    fn observation(&self) -> Observation {
        let cue = if self.tick == 0 || self.fully_observed {
            self.target as f32
        } else {
            0.
        };
        Observation::new(vec![
            cue,
            f32::from(self.tick == 0),
            f32::from(self.tick == self.delay),
            f32::from(self.marker == Some(self.tick)),
            self.tick as f32 / self.delay as f32,
        ])
    }
}

impl Env for HiddenTarget {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(5)
    }

    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        let bit = if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
            (seed % 2) as usize
        } else {
            self.rng.random_range(0..2usize)
        };
        self.target = if bit == 0 { -1 } else { 1 };
        self.tick = 0;
        self.ended = false;
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(!self.ended, "hidden-target reset required")?;
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        let reward = if self.tick == self.delay {
            let error = action[0] - self.target as f32;
            self.ended = true;
            -(error * error)
        } else {
            self.tick += 1;
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

impl Snapshot for HiddenTarget {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(
            self.version == 1
                && (2..=MAX_DELAY).contains(&self.delay)
                && self.marker.is_none_or(|m| m > 0 && m < self.delay)
                && self.tick <= self.delay
                && matches!(self.target, -1 | 1)
                && (!self.ended || self.tick == 0 || self.tick == self.delay)
                && self.space == physical_space(),
            "invalid hidden-target snapshot",
        )
    }
}
