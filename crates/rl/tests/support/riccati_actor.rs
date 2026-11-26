//! Analytic comparators. These retain their mathematical domain and do not
//! relabel clipped or discounted controllers as unconstrained optima.
use nalgebra::DVector;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::ObservationBatch,
    },
    envs::lqg::{LqgConfig, Riccati},
};

pub struct LqgRiccatiActor {
    config: LqgConfig,
    solution: Riccati,
    clip_actions: bool,
}
impl LqgRiccatiActor {
    pub fn new(config: LqgConfig, clip_actions: bool) -> Result<Self> {
        let solution = config.riccati()?;
        Ok(Self {
            config,
            solution,
            clip_actions,
        })
    }
    /// Expected undiscounted reward for the mathematically unconstrained policy,
    /// including Gaussian process noise. Ignores action bounds explicitly.
    /// The terminal state has RL value zero: its terminal cost was already paid.
    pub fn unconstrained_value(&self, state: &[f64], time: usize) -> Result<f64> {
        self.validate_input(state, time)?;
        if time == self.config.horizon() {
            return Ok(0.);
        }
        let state = DVector::from_column_slice(state);
        Ok(-state.dot(&(&self.solution.values[time] * &state)) - self.future_noise_cost(time))
    }
    /// Fixed first action, then the unconstrained optimal continuation (gamma=1).
    /// The supplied action is mathematical and may lie outside environment bounds.
    pub fn unconstrained_action_value(
        &self,
        state: &[f64],
        time: usize,
        action: &[f64],
    ) -> Result<f64> {
        self.validate_input(state, time)?;
        ensure(
            time < self.config.horizon()
                && action.len() == self.config.action_space.dimensions()
                && action.iter().all(|x| x.is_finite()),
            "invalid analytic action/time",
        )?;
        let state = DVector::from_column_slice(state);
        let action = DVector::from_column_slice(action);
        let next = &self.config.a[time] * &state + &self.config.b[time] * &action;
        Ok(-state.dot(&(&self.config.q[time] * &state))
            - action.dot(&(&self.config.r[time] * &action))
            - next.dot(&(&self.solution.values[time + 1] * &next))
            - self.future_noise_cost(time))
    }
    fn future_noise_cost(&self, time: usize) -> f64 {
        (time..self.config.horizon())
            .map(|t| {
                (self.config.noise[t].transpose()
                    * &self.solution.values[t + 1]
                    * &self.config.noise[t])
                    .trace()
            })
            .sum()
    }
    fn validate_input(&self, state: &[f64], time: usize) -> Result<()> {
        ensure(
            state.len() == self.config.initial.len()
                && state.iter().all(|x| x.is_finite())
                && time <= self.config.horizon(),
            "invalid analytic state/time",
        )
    }
}
impl Actor for LqgRiccatiActor {
    fn action_space(&self) -> &ActionSpace {
        &self.config.action_space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        out.clear();
        for observation in obs.iter() {
            observation.validate(self.config.initial.len() + 1, self.action_space(), true)?;
            let n = self.config.initial.len();
            let clock = observation.dense_values()[n] as f64 * self.config.horizon() as f64;
            ensure(
                (clock.round() as f32 / self.config.horizon() as f32)
                    == observation.dense_values()[n]
                    && clock >= 0.
                    && clock < self.config.horizon() as f64,
                "Riccati input clock must identify a live decision step",
            )?;
            let state: Vec<_> = observation.dense_values()[..n]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let mut action: Vec<_> = self
                .solution
                .action(&state, clock.round() as usize)?
                .into_iter()
                .map(|v| v as f32)
                .collect();
            if self.clip_actions
                && let ActionSpace::Box { low, high } = self.action_space()
            {
                for ((a, &lo), &hi) in action.iter_mut().zip(low).zip(high) {
                    *a = a.clamp(lo, hi);
                }
            }
            let action = Action::Continuous(action);
            self.action_space().check(&action, None)?;
            out.push(action);
        }
        Ok(())
    }
}

fn ensure(valid: bool, message: &str) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(rl::Error::Config(message.into()))
    }
}
