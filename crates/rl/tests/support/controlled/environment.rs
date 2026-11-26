use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::evaluation::{BranchingEnv, EnvironmentFrame, FrameSource},
    prelude::*,
};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct ContextLottery {
    time: usize,
    committed: f32,
    context: i8,
    rng: ChaCha8Rng,
    space: ActionSpace,
}
impl ContextLottery {
    pub fn new(context: i8, seed: u64) -> Self {
        assert!(matches!(context, -1 | 1));
        Self {
            time: 4,
            committed: 0.,
            context,
            rng: ChaCha8Rng::seed_from_u64(seed),
            space: ActionSpace::Box {
                low: vec![0.],
                high: vec![1.],
            },
        }
    }
    pub fn observation(&self) -> Observation {
        Observation::new(vec![
            self.time as f32 / 4.,
            self.committed,
            self.context as f32,
        ])
    }
}
impl Env for ContextLottery {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(3)
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
        if self.time >= 4 {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        let Action::Continuous(values) = action else {
            unreachable!()
        };
        if self.time == 0 {
            self.committed = values[0];
        }
        self.time += 1;
        let terminated = self.time == 4;
        let reward = if terminated {
            let slope = if self.rng.random::<f64>() < 0.2 {
                -4.
            } else if self.context == 1 {
                2.
            } else {
                0.5
            };
            1. + self.committed * slope + self.rng.random_range(-0.5f32..0.5)
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
impl BranchingEnv for ContextLottery {
    fn branch(&self, future_seed: u64) -> Result<Self> {
        self.branch_observation()?;
        let mut branch = self.clone();
        branch.rng = ChaCha8Rng::seed_from_u64(future_seed);
        Ok(branch)
    }
    fn branch_observation(&self) -> Result<Observation> {
        if self.time >= 4 {
            return Err(Error::State(
                "cannot branch terminal/uninitialized lottery".into(),
            ));
        }
        Ok(self.observation())
    }
    fn branch_state(&self) -> Result<Vec<f64>> {
        Ok(vec![
            self.time as f64,
            self.committed as f64,
            self.context as f64,
        ])
    }
    fn branch_description(&self) -> Result<String> {
        Ok("physical time, committed exposure, context; future lottery and uniform noise independently reseeded".into())
    }
}
impl FrameSource for ContextLottery {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        Ok(EnvironmentFrame {
            environment: "contextual-commit-lottery".into(),
            simulation_time: Some(self.time as f64),
            channels: BTreeMap::from([
                ("committed_exposure".into(), vec![self.committed as f64]),
                ("context".into(), vec![self.context as f64]),
            ]),
        })
    }
}
