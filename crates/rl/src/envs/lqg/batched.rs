use super::{LqgConfig, serialize_config};
use crate::{
    Error, Result,
    collect::vector_env::VectorEnv,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace, Step},
    },
};
use nalgebra::DVector;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Lane {
    state: DVector<f64>,
    time: usize,
    rng: ChaCha8Rng,
}

#[derive(Debug, Clone)]
struct Scratch {
    action: DVector<f64>,
    action_product: DVector<f64>,
    state_product: DVector<f64>,
    noise: DVector<f64>,
    next: DVector<f64>,
    contribution: DVector<f64>,
}
impl Scratch {
    fn new(config: &LqgConfig) -> Self {
        let n = config.initial.len();
        let d = config.action_space.dimensions();
        Self {
            action: DVector::zeros(d),
            action_product: DVector::zeros(d),
            state_product: DVector::zeros(n),
            noise: DVector::zeros(n),
            next: DVector::zeros(n),
            contribution: DVector::zeros(n),
        }
    }
}

/// Serial LQG lanes with shared immutable parameters and reusable f64 scratch.
///
/// Each lane owns its state, episode time and RNG. Inactive lanes do not advance.
/// Snapshots store parameters once and reconstruct scratch on deserialization;
/// generic vector snapshots do not deduplicate their environments' parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "BatchRecord")]
pub struct BatchedLqg {
    #[serde(serialize_with = "serialize_config")]
    config: Arc<LqgConfig>,
    lanes: Vec<Lane>,
    #[serde(skip)]
    scratch: Scratch,
}

#[derive(Deserialize)]
struct BatchRecord {
    config: LqgConfig,
    lanes: Vec<Lane>,
}
impl TryFrom<BatchRecord> for BatchedLqg {
    type Error = Error;

    fn try_from(record: BatchRecord) -> Result<Self> {
        record.config.validate()?;
        let batch = Self {
            scratch: Scratch::new(&record.config),
            config: Arc::new(record.config),
            lanes: record.lanes,
        };
        batch.validate_state()?;
        Ok(batch)
    }
}

impl BatchedLqg {
    pub fn new(config: LqgConfig, seeds: &[u64]) -> Result<Self> {
        Self::from_shared(Arc::new(config), seeds)
    }

    /// Creates one initially ended lane per seed. Reset lanes before stepping.
    pub fn from_shared(config: Arc<LqgConfig>, seeds: &[u64]) -> Result<Self> {
        config.validate()?;
        ensure(!seeds.is_empty(), "vector environment cannot be empty")?;
        let lanes = seeds
            .iter()
            .map(|seed| Lane {
                state: DVector::from_column_slice(&config.initial),
                time: config.horizon(),
                rng: ChaCha8Rng::seed_from_u64(*seed),
            })
            .collect();
        Ok(Self {
            scratch: Scratch::new(&config),
            config,
            lanes,
        })
    }

    pub fn config(&self) -> &LqgConfig {
        &self.config
    }

    pub(crate) fn validate_state(&self) -> Result<()> {
        self.config.validate()?;
        ensure(
            !self.lanes.is_empty()
                && self.lanes.iter().all(|lane| {
                    lane.state.len() == self.config.initial.len()
                        && lane.state.iter().all(|value| value.is_finite())
                        && lane.time <= self.config.horizon()
                }),
            "invalid batched LQG lane state or time",
        )
    }
}

fn observation(lane: &Lane, horizon: usize) -> Observation {
    let mut values = Vec::with_capacity(lane.state.len() + 1);
    values.extend(lane.state.iter().map(|value| *value as f32));
    values.push(lane.time as f32 / horizon as f32);
    Observation::new(values)
}

impl VectorEnv for BatchedLqg {
    fn len(&self) -> usize {
        self.lanes.len()
    }

    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.config.initial.len() + 1)
    }

    fn action_space(&self) -> &ActionSpace {
        &self.config.action_space
    }

    fn reset(&mut self, mask: &[bool], seeds: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
        ensure(
            mask.len() == self.len() && seeds.is_none_or(|seeds| seeds.len() == self.len()),
            "reset mask or seeds length mismatch",
        )?;
        let mut result = Vec::with_capacity(self.len());
        for (index, (lane, reset)) in self.lanes.iter_mut().zip(mask).enumerate() {
            if *reset {
                if let Some(seeds) = seeds {
                    lane.rng = ChaCha8Rng::seed_from_u64(seeds[index]);
                }
                lane.state
                    .as_mut_slice()
                    .copy_from_slice(&self.config.initial);
                lane.time = 0;
                let obs = observation(lane, self.config.horizon());
                obs.validate(
                    self.config.initial.len() + 1,
                    &self.config.action_space,
                    true,
                )?;
                result.push(Some(EpisodeStart::fresh(obs)));
            } else {
                result.push(None);
            }
        }
        Ok(result)
    }

    fn step(&mut self, actions: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
        let mut result = Vec::with_capacity(self.len());
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
            .zip(&self.lanes)
            .any(|(action, lane)| action.is_some() && lane.time == self.config.horizon())
        {
            return Err(Error::State(
                "reset ended vector lanes before stepping".into(),
            ));
        }
        for action in actions.iter().flatten() {
            self.config.action_space.check(action, None)?;
        }
        result.clear();
        result.reserve(self.len());
        let config = &self.config;
        let scratch = &mut self.scratch;
        for (lane, action) in self.lanes.iter_mut().zip(actions) {
            let Some(Action::Continuous(action)) = action else {
                result.push(None);
                continue;
            };
            for (destination, source) in scratch.action.iter_mut().zip(action) {
                *destination = f64::from(*source);
            }
            let t = lane.time;
            config.q[t].mul_to(&lane.state, &mut scratch.state_product);
            config.r[t].mul_to(&scratch.action, &mut scratch.action_product);
            let mut cost = lane.state.dot(&scratch.state_product)
                + scratch.action.dot(&scratch.action_product);
            for noise in scratch.noise.iter_mut() {
                *noise = lane.rng.sample::<f64, _>(StandardNormal);
            }
            config.a[t].mul_to(&lane.state, &mut scratch.next);
            config.b[t].mul_to(&scratch.action, &mut scratch.contribution);
            scratch.next += &scratch.contribution;
            config.noise[t].mul_to(&scratch.noise, &mut scratch.contribution);
            scratch.next += &scratch.contribution;
            std::mem::swap(&mut lane.state, &mut scratch.next);
            lane.time += 1;
            let terminated = lane.time == config.horizon();
            if terminated {
                config
                    .terminal
                    .mul_to(&lane.state, &mut scratch.state_product);
                cost += lane.state.dot(&scratch.state_product);
            }
            ensure((cost as f32).is_finite(), "LQG cost overflow")?;
            let observation = observation(lane, config.horizon());
            observation.validate(config.initial.len() + 1, &config.action_space, !terminated)?;
            result.push(Some(Step {
                observation,
                reward: (-cost as f32).into(),
                terminated,
                truncated: false,
            }));
        }
        Ok(())
    }
}

impl crate::core::checkpoint::Snapshot for BatchedLqg {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_state()
    }
}
