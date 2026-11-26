//! Default Gymnasium Pendulum dynamics; compose with a 200-step `TimeLimit` for v1.
//!
//! Physical actions must lie in [-2, 2]. Wrap in `ClipAction` to accept and
//! clip unbounded policy samples as SB3 and Gymnasium do. Reset distributions
//! match Gymnasium, but ChaCha8 does not reproduce NumPy's same-seed sequence.
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
use std::f64::consts::PI;

mod exact_state {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(state: &Option<[f64; 2]>, s: S) -> Result<S::Ok, S::Error> {
        state.map(|xs| xs.map(|x| x.to_string())).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<[f64; 2]>, D::Error> {
        Option::<[String; 2]>::deserialize(d)?
            .map(|[a, b]| {
                Ok([
                    a.parse().map_err(serde::de::Error::custom)?,
                    b.parse().map_err(serde::de::Error::custom)?,
                ])
            })
            .transpose()
    }
}

fn physical_space() -> ActionSpace {
    ActionSpace::Box {
        low: vec![-2.],
        high: vec![2.],
    }
}
fn valid_state(state: [f64; 2]) -> bool {
    state[0].is_finite() && state[1].is_finite() && state[1].abs() <= 8.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "PendulumRecord")]
pub struct Pendulum {
    #[serde(with = "exact_state")]
    state: Option<[f64; 2]>,
    space: ActionSpace,
    rng: ChaCha8Rng,
}

#[derive(Deserialize)]
struct PendulumRecord {
    #[serde(with = "exact_state")]
    state: Option<[f64; 2]>,
    space: ActionSpace,
    rng: ChaCha8Rng,
}

impl TryFrom<PendulumRecord> for Pendulum {
    type Error = crate::Error;
    fn try_from(record: PendulumRecord) -> Result<Self> {
        let env = Self {
            state: record.state,
            space: record.space,
            rng: record.rng,
        };
        env.validate_snapshot()?;
        Ok(env)
    }
}

impl Pendulum {
    pub fn new(seed: u64) -> Self {
        Self {
            state: None,
            space: physical_space(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Start at an explicit physical state without consuming random draws.
    pub fn from_state(seed: u64, state: [f64; 2]) -> Result<Self> {
        ensure(valid_state(state), "invalid Pendulum state")?;
        Ok(Self {
            state: Some(state),
            space: physical_space(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    /// Default 200-step v1 limit; add `ClipAction` for unbounded policy samples.
    pub fn v1(seed: u64) -> Result<TimeLimit<Self>> {
        TimeLimit::new(Self::new(seed), 200)
    }

    fn observation([theta, velocity]: [f64; 2]) -> Observation {
        Observation::new(vec![
            theta.cos() as f32,
            theta.sin() as f32,
            velocity as f32,
        ])
    }
}

impl Snapshot for Pendulum {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(
            self.space == physical_space() && self.state.is_none_or(valid_state),
            "invalid Pendulum snapshot",
        )
    }
}

impl Env for Pendulum {
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
        let state = [
            self.rng.random_range(-PI..PI),
            self.rng.random_range(-1.0..1.0),
        ];
        self.state = Some(state);
        Ok(EpisodeStart::fresh(Self::observation(state)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(self.state.is_some(), "reset required")?;
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        let torque = action[0];
        let [theta, velocity] = self.state.expect("live state validated");
        let angle = (theta + PI).rem_euclid(2. * PI) - PI;
        // Gymnasium receives f32 arrays: NumPy 2 retains f32 for the isolated
        // torque products before combining them with the f64 state terms.
        let cost = angle.powi(2) + 0.1 * velocity.powi(2) + (0.001_f32 * torque.powi(2)) as f64;
        let new_velocity =
            (velocity + (15. * theta.sin() + (3.0_f32 * torque) as f64) * 0.05).clamp(-8., 8.);
        let next = [theta + new_velocity * 0.05, new_velocity];
        ensure(
            valid_state(next) && (cost as f32).is_finite(),
            "Pendulum state or reward overflow",
        )?;
        self.state = Some(next);
        Ok(Step {
            observation: Self::observation(next),
            reward: (-(cost as f32)).into(),
            terminated: false,
            truncated: false,
        })
    }
}

#[cfg(feature = "diagnostics")]
impl crate::diagnostics::render::RenderSource for Pendulum {
    fn render_scene(&self) -> Result<crate::diagnostics::render::RenderScene> {
        use crate::diagnostics::render::{
            RENDER_VERSION, RenderBounds, RenderItem, RenderPrimitive, RenderScene, RenderStroke,
        };

        let [theta, _] = self
            .state
            .ok_or_else(|| crate::Error::State("reset required".into()))?;
        let pivot = [0., 0.];
        let bob = [theta.sin(), theta.cos()];
        let circle = |layer, center, radius, fill, stroke| RenderItem {
            layer,
            primitive: RenderPrimitive::Circle {
                center,
                radius,
                fill,
                stroke,
            },
        };
        Ok(RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [-1.3, -1.3],
                max: [1.3, 1.3],
            },
            background: [16, 27, 42, 255],
            items: vec![
                circle(
                    0,
                    pivot,
                    1.,
                    None,
                    Some(RenderStroke {
                        color: [75, 96, 120, 255],
                        width: 1.5,
                    }),
                ),
                circle(
                    1,
                    [0., 1.],
                    0.055,
                    None,
                    Some(RenderStroke {
                        color: [141, 204, 160, 255],
                        width: 2.,
                    }),
                ),
                RenderItem {
                    layer: 2,
                    primitive: RenderPrimitive::Line {
                        from: pivot,
                        to: bob,
                        stroke: RenderStroke {
                            color: [111, 181, 216, 255],
                            width: 5.,
                        },
                    },
                },
                circle(3, bob, 0.11, Some([247, 187, 92, 255]), None),
                circle(4, pivot, 0.07, Some([228, 240, 248, 255]), None),
            ],
        })
    }
}
