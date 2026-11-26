//! Controlled memory benchmark: only the initial cue identifies the correct decision.
//! Before the decision the only legal action is zero, eliminating action-based
//! external/previous-action memory. Timing, masks and intermediate rewards reveal
//! no hidden information. A public nonterminal marker is an ordinary observation.
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

/// Explicit bounded stress horizon. Existing short tasks keep identical physics,
/// public features, RNG streams, rewards and episode/reset semantics.
pub const MAX_DELAY: usize = 16_384;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayedBit {
    version: u32,
    delay: usize,
    marker: Option<usize>,
    fully_observed: bool,
    tick: usize,
    bit: usize,
    ended: bool,
    rng: ChaCha8Rng,
}
impl DelayedBit {
    pub fn new(
        delay: usize,
        marker: Option<usize>,
        fully_observed: bool,
        seed: u64,
    ) -> Result<Self> {
        ensure(
            (2..=MAX_DELAY).contains(&delay) && marker.is_none_or(|m| m > 0 && m < delay),
            "invalid delayed-information horizon or marker",
        )?;
        Ok(Self {
            version: 1,
            delay,
            marker,
            fully_observed,
            tick: 0,
            bit: 0,
            ended: true,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }
    fn observation(&self) -> Observation {
        let cue = if self.tick == 0 || self.fully_observed {
            if self.bit == 0 { -1. } else { 1. }
        } else {
            0.
        };
        Observation::dense(
            vec![
                cue,
                f32::from(self.tick == 0),
                f32::from(self.tick == self.delay),
                f32::from(self.marker == Some(self.tick)),
                self.tick as f32 / self.delay as f32,
            ],
            Some(vec![true, self.tick == self.delay]),
        )
    }
    pub fn delay(&self) -> usize {
        self.delay
    }
}
impl Env for DelayedBit {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(5)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(s) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(s);
            self.bit = (s % 2) as usize;
        } else {
            self.bit = self.rng.random_range(0..2);
        }
        self.tick = 0;
        self.ended = false;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(!self.ended, "delayed-bit reset required")?;
        self.action_space()
            .check(action, self.observation().dense_mask())?;
        let Action::Discrete(a) = action else {
            unreachable!()
        };
        let reward = if self.tick == self.delay {
            self.ended = true;
            f32::from(*a == self.bit)
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
impl Snapshot for DelayedBit {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(
            self.version == 1
                && (2..=MAX_DELAY).contains(&self.delay)
                && self.marker.is_none_or(|m| m > 0 && m < self.delay)
                && self.tick <= self.delay
                && self.bit < 2,
            "invalid delayed-bit snapshot",
        )
    }
}
