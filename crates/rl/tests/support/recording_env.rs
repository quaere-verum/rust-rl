//! Seeded fixture with ordinary public observations and private capture channels.
use super::recording_rl as rl;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
    },
    diagnostics::{
        evaluation::{EnvironmentFrame, FrameSource},
        render::{
            RENDER_VERSION, RenderBounds, RenderItem, RenderPrimitive, RenderScene, RenderSource,
        },
    },
};
use std::collections::BTreeMap;

pub const NAME: &str = "seeded-counter";

pub struct RecordingEnv {
    space: ActionSpace,
    horizon: usize,
    step: usize,
    position: f32,
    velocity: f32,
    diagnostic_signal: f64,
    rng: ChaCha8Rng,
}

impl RecordingEnv {
    pub fn new(horizon: usize, seed: u64) -> Self {
        assert!(horizon > 0);
        Self {
            space: ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            horizon,
            step: horizon,
            position: 0.,
            velocity: 0.,
            diagnostic_signal: 0.,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn observation(&self) -> Observation {
        Observation::new(vec![
            self.position,
            self.velocity,
            (self.horizon - self.step) as f32 / self.horizon as f32,
        ])
    }
}

impl Env for RecordingEnv {
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
        self.position = 2.;
        self.velocity = 0.;
        self.step = 0;
        self.diagnostic_signal = self.rng.random_range(-0.5..0.5);
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.step == self.horizon {
            return Err(Error::State("reset required".into()));
        }
        self.space.check(action, None)?;
        let Action::Continuous(values) = action else {
            unreachable!("action space checked above");
        };
        self.velocity = values[0];
        self.position += self.velocity;
        self.step += 1;
        self.diagnostic_signal = self.rng.random_range(-0.5..0.5);
        Ok(Step {
            observation: self.observation(),
            reward: (-self.position.abs() + self.diagnostic_signal as f32).into(),
            terminated: self.step == self.horizon,
            truncated: false,
        })
    }
}

impl FrameSource for RecordingEnv {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        Ok(EnvironmentFrame {
            environment: NAME.into(),
            simulation_time: Some(self.step as f64),
            channels: BTreeMap::from([
                ("position".into(), vec![f64::from(self.position)]),
                ("diagnostic_signal".into(), vec![self.diagnostic_signal]),
            ]),
        })
    }
}

impl RenderSource for RecordingEnv {
    fn render_scene(&self) -> Result<RenderScene> {
        Ok(RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [0., 0.],
                max: [1., 1.],
            },
            background: [0, 0, 0, 255],
            items: vec![RenderItem {
                layer: 0,
                primitive: RenderPrimitive::Text {
                    position: [0.5, 0.5],
                    text: self.step.to_string(),
                    color: [255; 4],
                    size: 12.,
                },
            }],
        })
    }
}
