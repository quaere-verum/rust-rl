//! Gymnasium-compatible CartPole dynamics; compose with a 500-step `TimeLimit` for v1.
//!
//! Reset draws have the same independent uniform distribution as Gymnasium, but
//! the owned ChaCha8 stream does not reproduce NumPy's same-seed sequence.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        ensure,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step, TimeLimit},
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

const ACTION_SPACE: ActionSpace = ActionSpace::Discrete { count: 2 };
const X_LIMIT: f64 = 2.4;
const THETA_LIMIT: f64 = 12.0 * 2.0 * std::f64::consts::PI / 360.0;

// Decimal strings retain the exact f64 state through serde_json regardless of
// whether serde_json's optional float_roundtrip feature is enabled.
mod exact_state {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(state: &Option<[f64; 4]>, s: S) -> Result<S::Ok, S::Error> {
        state.map(|xs| xs.map(|x| x.to_string())).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f64; 4]>, D::Error> {
        Option::<[String; 4]>::deserialize(d)?
            .map(|xs| {
                let parsed = xs.map(|x| x.parse::<f64>());
                let [a, b, c, d] = parsed;
                Ok([
                    a.map_err(serde::de::Error::custom)?,
                    b.map_err(serde::de::Error::custom)?,
                    c.map_err(serde::de::Error::custom)?,
                    d.map_err(serde::de::Error::custom)?,
                ])
            })
            .transpose()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "CartPoleRecord")]
pub struct CartPole {
    #[serde(with = "exact_state")]
    state: Option<[f64; 4]>,
    ended: bool,
    rng: ChaCha8Rng,
}

#[derive(Deserialize)]
struct CartPoleRecord {
    #[serde(with = "exact_state")]
    state: Option<[f64; 4]>,
    ended: bool,
    rng: ChaCha8Rng,
}

impl TryFrom<CartPoleRecord> for CartPole {
    type Error = crate::Error;

    fn try_from(record: CartPoleRecord) -> Result<Self> {
        let env = Self {
            state: record.state,
            ended: record.ended,
            rng: record.rng,
        };
        env.validate_snapshot()?;
        Ok(env)
    }
}

fn outside(state: &[f64; 4]) -> bool {
    state[0] < -X_LIMIT || state[0] > X_LIMIT || state[2] < -THETA_LIMIT || state[2] > THETA_LIMIT
}

fn finite_observation(state: &[f64; 4]) -> bool {
    state
        .iter()
        .all(|x| x.is_finite() && (*x as f32).is_finite())
}

impl CartPole {
    pub fn new(seed: u64) -> Self {
        Self {
            state: None,
            ended: true,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Start a live episode at an explicit physical state for controlled comparisons.
    /// Unlike an ordinary reset, this does not consume random draws.
    pub fn from_state(seed: u64, state: [f64; 4]) -> Result<Self> {
        ensure(
            finite_observation(&state) && !outside(&state),
            "invalid live CartPole state",
        )?;
        Ok(Self {
            state: Some(state),
            ended: false,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    /// Default v1 time limit with rust-rl's termination-priority convention.
    pub fn v1(seed: u64) -> Result<TimeLimit<Self>> {
        TimeLimit::new(Self::new(seed), 500)
    }

    fn observation(state: &[f64; 4]) -> Observation {
        Observation::new(state.iter().map(|x| *x as f32).collect())
    }
}

impl Snapshot for CartPole {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(
            match self.state {
                None => self.ended,
                Some(state) => finite_observation(&state) && self.ended == outside(&state),
            },
            "invalid CartPole snapshot state",
        )
    }
}

impl Env for CartPole {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(4)
    }
    fn action_space(&self) -> &ActionSpace {
        &ACTION_SPACE
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        let state = std::array::from_fn(|_| self.rng.random_range(-0.05..0.05));
        self.state = Some(state);
        self.ended = false;
        Ok(EpisodeStart::fresh(Self::observation(&state)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(!self.ended && self.state.is_some(), "reset required")?;
        ACTION_SPACE.check(action, None)?;
        let Action::Discrete(action) = action else {
            unreachable!()
        };
        let [x, x_dot, theta, theta_dot] = self.state.expect("live state validated");
        let force = if *action == 1 { 10.0 } else { -10.0 };
        let (sine, cosine) = (theta.sin(), theta.cos());
        let temp = (force + 0.05 * theta_dot.powi(2) * sine) / 1.1;
        let theta_acc =
            (9.8 * sine - cosine * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cosine.powi(2) / 1.1));
        let x_acc = temp - 0.05 * theta_acc * cosine / 1.1;
        let next = [
            x + 0.02 * x_dot,
            x_dot + 0.02 * x_acc,
            theta + 0.02 * theta_dot,
            theta_dot + 0.02 * theta_acc,
        ];
        ensure(finite_observation(&next), "CartPole state overflow")?;
        let terminated = outside(&next);
        self.state = Some(next);
        self.ended = terminated;
        Ok(Step {
            observation: Self::observation(&next),
            reward: (1.).into(),
            terminated,
            truncated: false,
        })
    }
}

#[cfg(feature = "diagnostics")]
impl crate::diagnostics::render::RenderSource for CartPole {
    fn render_scene(&self) -> Result<crate::diagnostics::render::RenderScene> {
        use crate::diagnostics::render::{
            RENDER_VERSION, RenderBounds, RenderItem, RenderPrimitive, RenderScene, RenderStroke,
        };

        let [x, _, theta, _] = self
            .state
            .ok_or_else(|| crate::Error::State("reset required".into()))?;
        let pivot = [x, 0.38];
        // The equation of motion uses a half-length of 0.5 m.
        let tip = [x + theta.sin(), pivot[1] + theta.cos()];
        let line = |layer, from, to, color, width| RenderItem {
            layer,
            primitive: RenderPrimitive::Line {
                from,
                to,
                stroke: RenderStroke { color, width },
            },
        };
        let mut items = vec![
            line(0, [-2.85, 0.], [2.85, 0.], [102, 122, 146, 255], 3.),
            line(
                1,
                [-X_LIMIT, -0.08],
                [-X_LIMIT, 0.12],
                [235, 121, 108, 255],
                2.,
            ),
            line(
                1,
                [X_LIMIT, -0.08],
                [X_LIMIT, 0.12],
                [235, 121, 108, 255],
                2.,
            ),
        ];
        for offset in [-0.18, 0.18] {
            items.push(RenderItem {
                layer: 2,
                primitive: RenderPrimitive::Circle {
                    center: [x + offset, 0.075],
                    radius: 0.075,
                    fill: Some([28, 44, 62, 255]),
                    stroke: Some(RenderStroke {
                        color: [172, 191, 212, 255],
                        width: 1.5,
                    }),
                },
            });
        }
        items.extend([
            RenderItem {
                layer: 3,
                primitive: RenderPrimitive::Rectangle {
                    min: [x - 0.32, 0.12],
                    max: [x + 0.32, pivot[1]],
                    fill: Some([69, 163, 200, 255]),
                    stroke: None,
                },
            },
            line(4, pivot, tip, [247, 187, 92, 255], 5.),
            RenderItem {
                layer: 5,
                primitive: RenderPrimitive::Circle {
                    center: pivot,
                    radius: 0.055,
                    fill: Some([228, 240, 248, 255]),
                    stroke: None,
                },
            },
        ]);
        Ok(RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [-3., -0.25],
                max: [3., 1.65],
            },
            background: [16, 27, 42, 255],
            items,
        })
    }
}
