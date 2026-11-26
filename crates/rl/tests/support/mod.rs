//! Shared bounded scientific fixtures; these are test environments, not public APIs.
#![allow(dead_code)]
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
    },
};

pub const SEEDS: [u64; 5] = [7, 19, 31, 43, 59];
pub const HORIZON: usize = 4;
pub const MAX_TRANSITIONS: u64 = 20_000;
pub const MAX_UPDATES: u64 = 5_000;

pub fn pilot() -> bool {
    matches!(
        std::env::var("RL_SCIENCE_PILOT").as_deref(),
        Ok("1" | "full")
    )
}

pub fn full_pilot() -> bool {
    std::env::var("RL_SCIENCE_PILOT").as_deref() == Ok("full")
}

#[derive(Clone)]
pub struct DelayedControl {
    time: usize,
    accumulated: f32,
    space: ActionSpace,
}
impl DelayedControl {
    pub fn new() -> Self {
        Self {
            time: HORIZON,
            accumulated: 0.,
            space: ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
        }
    }
    fn observation(&self) -> Observation {
        Observation::new(vec![self.time as f32 / HORIZON as f32, self.accumulated])
    }
}
impl Env for DelayedControl {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(2)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        self.accumulated = 0.;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.time >= HORIZON {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        self.accumulated += action[0] / HORIZON as f32;
        self.time += 1;
        let terminated = self.time == HORIZON;
        Ok(Step {
            observation: self.observation(),
            reward: (if terminated {
                -(self.accumulated - 0.6).powi(2)
            } else {
                0.
            })
            .into(),
            terminated,
            truncated: false,
        })
    }
}

#[derive(Clone)]
pub struct MaskedChain {
    time: usize,
    correct: usize,
    space: ActionSpace,
}
impl MaskedChain {
    pub fn new() -> Self {
        Self {
            time: HORIZON,
            correct: 0,
            space: ActionSpace::Discrete { count: 3 },
        }
    }
    fn observation(&self) -> Observation {
        let mut values = vec![0.; HORIZON + 1];
        if self.time < HORIZON {
            values[self.time] = 1.;
        }
        values[HORIZON] = self.correct as f32 / HORIZON as f32;
        Observation::dense(
            values,
            Some(if self.time == HORIZON {
                vec![false; 3]
            } else {
                vec![true, true, false]
            }),
        )
    }
}
impl Env for MaskedChain {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(HORIZON + 1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        self.correct = 0;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.time >= HORIZON {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, self.observation().dense_mask())?;
        let Action::Discrete(action) = action else {
            unreachable!()
        };
        self.correct += usize::from(*action == usize::from(self.time.is_multiple_of(2)));
        self.time += 1;
        let terminated = self.time == HORIZON;
        Ok(Step {
            observation: self.observation(),
            reward: (if terminated {
                2. * self.correct as f32 / HORIZON as f32 - 1.
            } else {
                0.
            })
            .into(),
            terminated,
            truncated: false,
        })
    }
}

#[derive(Clone)]
pub struct CommitLottery {
    time: usize,
    committed: f32,
    rng: ChaCha8Rng,
    space: ActionSpace,
}
impl CommitLottery {
    pub fn new(seed: u64) -> Self {
        Self {
            time: HORIZON,
            committed: 0.,
            rng: ChaCha8Rng::seed_from_u64(seed),
            space: ActionSpace::Box {
                low: vec![0.],
                high: vec![1.],
            },
        }
    }
    fn observation(&self) -> Observation {
        Observation::new(vec![self.time as f32 / HORIZON as f32, self.committed])
    }
    pub fn episode(&mut self, action: f32) -> f32 {
        self.reset(None).unwrap();
        let mut total = 0.;
        for _ in 0..HORIZON {
            total += self
                .step(&Action::Continuous(vec![action]))
                .unwrap()
                .reward
                .total();
        }
        total
    }
}
impl Env for CommitLottery {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(2)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.time = 0;
        self.committed = 0.;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.time >= HORIZON {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        if self.time == 0 {
            self.committed = action[0];
        }
        self.time += 1;
        let terminated = self.time == HORIZON;
        let reward = if terminated {
            1. + self.committed
                * if self.rng.random::<f64>() < 0.2 {
                    -4.
                } else {
                    2.
                }
        } else {
            0.
        };
        Ok(Step {
            observation: self.observation(),
            reward: reward.into(),
            terminated,
            truncated: false,
        })
    }
}

pub fn evaluate<E: Env + Clone>(actor: &dyn Actor, prototype: &E, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 10_000);
    let mut env = prototype.clone();
    let mut observation = env.reset(Some(seed + 20_000)).unwrap().observation;
    let mut total = 0.;
    for _ in 0..HORIZON {
        let action = actor
            .act(&[observation], ActMode::Deterministic, &mut rng)
            .unwrap()
            .remove(0);
        let step = env.step(&action).unwrap();
        total += f64::from(step.reward.total());
        if step.ended() {
            return total;
        }
        observation = step.observation;
    }
    panic!("evaluation horizon exceeded")
}

pub fn random_score<E: Env + Clone>(prototype: &E, seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 30_000);
    let mut total = 0.;
    for episode in 0..32 {
        let mut env = prototype.clone();
        let mut observation = env
            .reset(Some(seed + 40_000 + episode))
            .unwrap()
            .observation;
        for _ in 0..HORIZON {
            let action = match env.action_space() {
                ActionSpace::Discrete { count } => {
                    let choices: Vec<_> = (0..*count)
                        .filter(|i| observation.dense_mask().is_none_or(|m| m[*i]))
                        .collect();
                    Action::Discrete(choices[rng.random_range(0..choices.len())])
                }
                ActionSpace::Box { low, high } => Action::Continuous(
                    low.iter()
                        .zip(high)
                        .map(|(low, high)| rng.random_range(*low..*high))
                        .collect(),
                ),
                _ => unreachable!(),
            };
            let step = env.step(&action).unwrap();
            total += f64::from(step.reward.total());
            if step.ended() {
                break;
            }
            observation = step.observation;
        }
    }
    total / 32.
}
