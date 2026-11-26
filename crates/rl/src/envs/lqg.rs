use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
    },
};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[path = "lqg/batched.rs"]
mod batched;
pub use batched::BatchedLqg;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LqgConfig {
    pub a: Vec<DMatrix<f64>>,
    pub b: Vec<DMatrix<f64>>,
    pub q: Vec<DMatrix<f64>>,
    pub r: Vec<DMatrix<f64>>,
    pub terminal: DMatrix<f64>,
    pub noise: Vec<DMatrix<f64>>,
    pub initial: Vec<f64>,
    pub action_space: ActionSpace,
}
fn penalty(m: &DMatrix<f64>, positive: bool) -> bool {
    m.nrows() == m.ncols()
        && m.iter().all(|x| x.is_finite())
        && (m - m.transpose()).amax() < 1e-10
        && if positive {
            m.clone().cholesky().is_some()
        } else {
            m.clone().symmetric_eigen().eigenvalues.min() >= -1e-10
        }
}
impl LqgConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn scalar(
        horizon: usize,
        a: f64,
        b: f64,
        q: f64,
        r: f64,
        terminal: f64,
        initial: f64,
        noise_std: f64,
        bound: f32,
    ) -> Result<Self> {
        let mat = |v| DMatrix::from_element(1, 1, v);
        let c = Self {
            a: vec![mat(a); horizon],
            b: vec![mat(b); horizon],
            q: vec![mat(q); horizon],
            r: vec![mat(r); horizon],
            terminal: mat(terminal),
            noise: vec![mat(noise_std); horizon],
            initial: vec![initial],
            action_space: ActionSpace::Box {
                low: vec![-bound],
                high: vec![bound],
            },
        };
        c.validate()?;
        Ok(c)
    }
    pub fn horizon(&self) -> usize {
        self.a.len()
    }
    pub fn validate(&self) -> Result<()> {
        self.action_space.validate()?;
        let h = self.horizon();
        let n = self.initial.len();
        let d = self.action_space.dimensions();
        ensure(
            h > 0
                && n > 0
                && !matches!(self.action_space, ActionSpace::Discrete { .. })
                && self.initial.iter().all(|x| x.is_finite())
                && [self.b.len(), self.q.len(), self.r.len(), self.noise.len()]
                    .iter()
                    .all(|&x| x == h),
            "invalid LQG configuration sizes",
        )?;
        ensure(
            self.a
                .iter()
                .chain(&self.noise)
                .all(|m| m.shape() == (n, n) && m.iter().all(|v| v.is_finite()))
                && self
                    .b
                    .iter()
                    .all(|m| m.shape() == (n, d) && m.iter().all(|v| v.is_finite()))
                && self
                    .q
                    .iter()
                    .all(|m| m.shape() == (n, n) && penalty(m, false))
                && self
                    .r
                    .iter()
                    .all(|m| m.shape() == (d, d) && penalty(m, true))
                && self.terminal.shape() == (n, n)
                && penalty(&self.terminal, false),
            "invalid LQG matrices or cost definiteness",
        )
    }
    pub fn riccati(&self) -> Result<Riccati> {
        self.validate()?;
        let h = self.horizon();
        let mut p = vec![self.terminal.clone(); h + 1];
        let mut gains = vec![DMatrix::zeros(self.action_space.dimensions(), self.initial.len()); h];
        for t in (0..h).rev() {
            let a = &self.a[t];
            let b = &self.b[t];
            let next = &p[t + 1];
            let system = &self.r[t] + b.transpose() * next * b;
            let factor = system.cholesky().ok_or_else(|| {
                crate::Error::Numerical("Riccati solve is not positive definite".into())
            })?;
            let k = factor.solve(&(b.transpose() * next * a));
            let current = &self.q[t] + a.transpose() * next * a - a.transpose() * next * b * &k;
            p[t] = (&current + current.transpose()) * 0.5;
            gains[t] = k;
        }
        Ok(Riccati { values: p, gains })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Riccati {
    pub values: Vec<DMatrix<f64>>,
    pub gains: Vec<DMatrix<f64>>,
}
impl Riccati {
    pub fn action(&self, state: &[f64], time: usize) -> Result<Vec<f64>> {
        ensure(
            time < self.gains.len() && state.len() == self.gains[time].ncols(),
            "invalid Riccati state/time",
        )?;
        Ok((-&self.gains[time] * DVector::from_column_slice(state))
            .as_slice()
            .to_vec())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "LqgRecord")]
pub struct Lqg {
    #[serde(serialize_with = "serialize_config")]
    config: Arc<LqgConfig>,
    state: DVector<f64>,
    time: usize,
    rng: ChaCha8Rng,
}

fn serialize_config<S: serde::Serializer>(
    config: &Arc<LqgConfig>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    config.as_ref().serialize(serializer)
}

#[derive(Deserialize)]
struct LqgRecord {
    config: LqgConfig,
    state: DVector<f64>,
    time: usize,
    rng: ChaCha8Rng,
}

impl TryFrom<LqgRecord> for Lqg {
    type Error = crate::Error;

    fn try_from(record: LqgRecord) -> Result<Self> {
        let env = Self {
            config: Arc::new(record.config),
            state: record.state,
            time: record.time,
            rng: record.rng,
        };
        env.validate_state()?;
        Ok(env)
    }
}

impl Lqg {
    pub fn new(config: LqgConfig, seed: u64) -> Result<Self> {
        Self::from_shared(Arc::new(config), seed)
    }
    /// Uses immutable parameters shared with other environments. State and RNG
    /// remain owned by this environment, including when it is cloned.
    pub fn from_shared(config: Arc<LqgConfig>, seed: u64) -> Result<Self> {
        config.validate()?;
        let time = config.horizon();
        Ok(Self {
            state: DVector::from_column_slice(&config.initial),
            config,
            time,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }
    pub fn config(&self) -> &LqgConfig {
        &self.config
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.config.validate()?;
        ensure(
            self.state.len() == self.config.initial.len()
                && self.state.iter().all(|value| value.is_finite())
                && self.time <= self.config.horizon(),
            "invalid LQG state or time",
        )
    }
    fn observation(&self) -> Observation {
        let mut v: Vec<_> = self.state.iter().map(|x| *x as f32).collect();
        v.push(self.time as f32 / self.config.horizon() as f32);
        Observation::new(v)
    }
}
impl Env for Lqg {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.config.initial.len() + 1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.config.action_space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(s) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(s);
        }
        self.time = 0;
        self.state = DVector::from_column_slice(&self.config.initial);
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(self.time < self.config.horizon(), "reset required")?;
        self.action_space().check(action, None)?;
        let Action::Continuous(u) = action else {
            unreachable!()
        };
        let u = DVector::from_iterator(u.len(), u.iter().map(|x| *x as f64));
        let t = self.time;
        let mut cost =
            self.state.dot(&(&self.config.q[t] * &self.state)) + u.dot(&(&self.config.r[t] * &u));
        let noise = DVector::from_fn(self.state.len(), |_, _| {
            self.rng.sample::<f64, _>(StandardNormal)
        });
        self.state =
            &self.config.a[t] * &self.state + &self.config.b[t] * u + &self.config.noise[t] * noise;
        self.time += 1;
        let terminated = self.time == self.config.horizon();
        if terminated {
            cost += self.state.dot(&(&self.config.terminal * &self.state));
        }
        ensure((cost as f32).is_finite(), "LQG cost overflow")?;
        Ok(Step {
            observation: self.observation(),
            reward: (-cost as f32).into(),
            terminated,
            truncated: false,
        })
    }
}

impl crate::core::checkpoint::Snapshot for Lqg {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_state()
    }
}

#[cfg(feature = "diagnostics")]
impl crate::diagnostics::evaluation::BranchingEnv for Lqg {
    fn branch(&self, future_seed: u64) -> Result<Self> {
        self.validate_state()?;
        ensure(
            self.time < self.config.horizon(),
            "cannot branch terminal LQG state",
        )?;
        let mut branch = self.clone();
        branch.rng = ChaCha8Rng::seed_from_u64(future_seed);
        Ok(branch)
    }
    fn branch_observation(&self) -> Result<Observation> {
        self.validate_state()?;
        ensure(
            self.time < self.config.horizon(),
            "cannot evaluate terminal LQG state",
        )?;
        Ok(self.observation())
    }
    fn branch_state(&self) -> Result<Vec<f64>> {
        self.validate_state()?;
        Ok(std::iter::once(self.time as f64)
            .chain(self.state.iter().copied())
            .collect())
    }
    fn branch_description(&self) -> Result<String> {
        let config = serde_json::to_vec(self.config.as_ref())?;
        Ok(format!(
            "LQG full physical state [decision_step, state...], config CRC32 {:08x}; future RNG replaced independently; observations narrow state to f32",
            crc32fast::hash(&config)
        ))
    }
}
#[cfg(feature = "diagnostics")]
impl crate::diagnostics::evaluation::FrameSource for Lqg {
    fn diagnostic_frame(&self) -> Result<crate::diagnostics::evaluation::EnvironmentFrame> {
        self.validate_state()?;
        Ok(crate::diagnostics::evaluation::EnvironmentFrame {
            environment: "lqg".into(),
            simulation_time: Some(self.time as f64),
            channels: std::collections::BTreeMap::from([
                ("state".into(), self.state.iter().copied().collect()),
                ("decision_step".into(), vec![self.time as f64]),
            ]),
        })
    }
}
