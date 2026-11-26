//! Analytical reference for the unconstrained mathematical LQG objective.

use super::{
    ActionGradient, ConfigValue, GradientMethod, OptimalDecisionDomain, ReferenceCheckpoint,
    ReferenceContext, ReferenceHorizon, ReferenceInformation, ReferenceKind, ReferenceLifetime,
    ReferenceObjective, ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, ObservationBatch},
    },
    diagnostics::{policy_cache::hash_json, value::ValueConventions},
    envs::lqg::LqgConfig,
};
use nalgebra::{DMatrix, DVector};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

const MAX_CONFIGURATION_BYTES: usize = 8 * 1024 * 1024;

struct Decision {
    cost: DMatrix<f64>,
    feedback: DMatrix<f64>,
    noise_cost: f64,
}

/// Expected V*, Q* and Q* action derivatives at an observed LQG coordinate.
/// The analytical domain uses the supplied coordinates as real state/action
/// inputs. Evaluating the actor in `Lqg` includes simulator and interface
/// rounding, which is declared in the reference provenance.
pub struct LqgReference {
    specification: ReferenceSpec,
    config: LqgConfig,
    decisions: Vec<Decision>,
}

impl LqgReference {
    pub fn new(config: &LqgConfig, discount: f64) -> Result<Self> {
        ensure(
            matches!(config.action_space, ActionSpace::Unbounded { .. }),
            "exact LQG reference requires an unbounded action space",
        )?;
        let horizon = config.horizon();
        let states = config.initial.len();
        ensure(
            (1..=(1 << 24)).contains(&horizon)
                && (1..super::MAX_DIMENSIONS).contains(&states)
                && (1..=super::MAX_DIMENSIONS).contains(&config.action_space.dimensions()),
            "LQG reference dimensions or clock exceed admission",
        )?;
        let config_hash = hash_json(config, MAX_CONFIGURATION_BYTES)?;
        config.validate()?;
        let specification = ReferenceSpec {
            id: "lqg-riccati".into(),
            context: ReferenceContext {
                environment: "lqg".into(),
                observation_dim: states + 1,
                typed_observation_space: None,
                action_space: config.action_space.clone(),
                conventions: ValueConventions {
                    dynamics: "finite time-varying linear Gaussian dynamics".into(),
                    observation: "f32 state coordinates lifted to f64 and normalized decision clock".into(),
                    action: "continuous control vector".into(),
                    reward_units: "negative quadratic cost".into(),
                    reward_convention: "negative stage cost including terminal cost in the final reward".into(),
                    discount,
                    horizon: format!("{horizon} environment decisions from reset"),
                    terminal_treatment: "zero continuation after environment termination".into(),
                    action_constraints: "unbounded continuous actions".into(),
                },
                objective: ReferenceObjective::ExpectedDiscountedReturn,
                horizon: ReferenceHorizon::Finite { decisions: horizon as u32 },
                environment_config: BTreeMap::from([
                    ("horizon".into(), ConfigValue::Integer(horizon as i64)),
                    ("configuration_sha256".into(), ConfigValue::Text(config_hash)),
                ]),
                // The simulator retains f64 coordinates behind its rounded observations.
                markov_observation: false,
                decision_time_coordinate: Some(states),
            },
            quality: ReferenceQuality {
                kind: ReferenceKind::VerifiedOptimal,
                method: "discounted finite-horizon Riccati solution and Gaussian noise trace".into(),
                assumptions: vec![
                    "Optimal for the unconstrained real-valued LQG model initialized at the supplied observation coordinates".into(),
                    "Independent zero-mean Gaussian process noise and positive-definite action Hessians".into(),
                    "Observation coordinates are treated as exact inputs, with no inferred distribution over underlying simulator states".into(),
                ],
            },
            target: ReferenceTarget::VerifiedOptimal,
            lifetime: ReferenceLifetime::Fixed,
            information: ReferenceInformation::LearnerObservation,
            uncertainty: Some("Analytical expectation with f64 numerical arithmetic. Simulator rollouts and control inputs also have f32 rounding.".into()),
        };
        specification.validate()?;

        let mut config = config.clone();
        // The quadratic forms depend only on the symmetric parts, including for
        // configs admitted with an insignificant numerical asymmetry.
        config.terminal = symmetric(&config.terminal);
        for cost in config.q.iter_mut().chain(&mut config.r) {
            *cost = symmetric(cost);
        }
        let mut decisions = Vec::<Decision>::with_capacity(horizon);
        for time in (0..horizon).rev() {
            let (next_cost, next_noise_cost, weight) = match decisions.last() {
                Some(next) => (&next.cost, next.noise_cost, discount),
                // Lqg charges the terminal quadratic in the final reward.
                None => (&config.terminal, 0., 1.),
            };
            let transition_cost = next_cost * weight;
            let a = &config.a[time];
            let b = &config.b[time];
            let noise = &config.noise[time];
            let cross = b.transpose() * &transition_cost * a;
            let hessian = &config.r[time] + b.transpose() * &transition_cost * b;
            let factor = symmetric(&hessian).cholesky().ok_or_else(|| {
                Error::Numerical("LQG reference action Hessian is not positive definite".into())
            })?;
            let feedback = factor.solve(&cross);
            let cost = symmetric(
                &(&config.q[time] + a.transpose() * &transition_cost * a
                    - cross.transpose() * &feedback),
            );
            let noise_cost =
                (noise.transpose() * transition_cost * noise).trace() + discount * next_noise_cost;
            ensure(
                cost.iter().chain(feedback.iter()).all(|v| v.is_finite()) && noise_cost.is_finite(),
                "nonfinite LQG reference solution",
            )?;
            decisions.push(Decision {
                cost,
                feedback,
                noise_cost,
            });
        }
        decisions.reverse();
        Ok(Self {
            specification,
            config,
            decisions,
        })
    }

    fn decode(&self, observation: &Observation) -> Result<(DVector<f64>, usize)> {
        observation.validate(
            self.specification.context.observation_dim,
            self.action_space(),
            false,
        )?;
        let values = observation.dense_values();
        let states = self.config.initial.len();
        let horizon = self.decisions.len();
        let clock = values[states];
        ensure((0. ..=1.).contains(&clock), "invalid LQG reference clock")?;
        let time = (f64::from(clock) * horizon as f64).round() as usize;
        ensure(
            time as f32 / horizon as f32 == clock,
            "LQG reference clock does not identify a decision step",
        )?;
        Ok((
            DVector::from_iterator(states, values[..states].iter().copied().map(f64::from)),
            time,
        ))
    }

    fn continuation(&self, time: usize) -> (&DMatrix<f64>, f64) {
        self.decisions
            .get(time + 1)
            .map_or((&self.config.terminal, 1.), |next| {
                (&next.cost, self.specification.context.conventions.discount)
            })
    }
}

fn symmetric(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    (matrix + matrix.transpose()) * 0.5
}

fn checked_value(value: f64) -> Result<Option<f64>> {
    ensure(value.is_finite(), "nonfinite LQG reference prediction")?;
    Ok(Some(value))
}

impl Actor for LqgReference {
    fn action_space(&self) -> &ActionSpace {
        &self.specification.context.action_space
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        out.clear();
        for observation in observations.iter() {
            let (state, time) = self.decode(observation)?;
            let decision = self.decisions.get(time).ok_or_else(|| {
                Error::Config("terminal LQG reference state has no action".into())
            })?;
            let action = (-&decision.feedback * state)
                .iter()
                .map(|&x| x as f32)
                .collect();
            let action = Action::Continuous(action);
            self.action_space().check(&action, None)?;
            out.push(action);
        }
        Ok(())
    }
}

impl ReferenceProvider for LqgReference {
    fn optimal_decision_domain(&self) -> Option<OptimalDecisionDomain> {
        Some(OptimalDecisionDomain::ModelCoordinates)
    }
    fn specification(&self) -> &ReferenceSpec {
        &self.specification
    }

    fn actor(&self) -> Option<&dyn Actor> {
        Some(self)
    }

    fn evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn state_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
    ) -> Result<Option<f64>> {
        let (state, time) = self.decode(observation)?;
        let Some(decision) = self.decisions.get(time) else {
            return Ok(Some(0.));
        };
        checked_value(-state.dot(&(&decision.cost * &state)) - decision.noise_cost)
    }

    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        let (state, time) = self.decode(observation)?;
        let Some(decision) = self.decisions.get(time) else {
            return Ok(None);
        };
        let action = continuous_action(self.action_space(), action)?;
        let next = &self.config.a[time] * &state + &self.config.b[time] * &action;
        let (continuation, weight) = self.continuation(time);
        checked_value(
            -state.dot(&(&self.config.q[time] * &state))
                - action.dot(&(&self.config.r[time] * &action))
                - weight * next.dot(&(continuation * &next))
                - decision.noise_cost,
        )
    }

    fn action_gradient(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<ActionGradient>> {
        let (state, time) = self.decode(observation)?;
        if time == self.decisions.len() {
            return Ok(None);
        }
        let action = continuous_action(self.action_space(), action)?;
        let b = &self.config.b[time];
        let next = &self.config.a[time] * state + b * &action;
        let (continuation, weight) = self.continuation(time);
        let gradient =
            -2. * (&self.config.r[time] * action + b.transpose() * continuation * next * weight);
        ensure(
            gradient.iter().all(|v| v.is_finite()),
            "nonfinite LQG reference derivative",
        )?;
        Ok(Some(ActionGradient {
            values: gradient.as_slice().to_vec(),
            method: GradientMethod::Analytic,
        }))
    }
}

fn continuous_action(space: &ActionSpace, action: &Action) -> Result<DVector<f64>> {
    space.check(action, None)?;
    let Action::Continuous(values) = action else {
        return Err(Error::Action(
            "LQG reference requires continuous actions".into(),
        ));
    };
    Ok(DVector::from_iterator(
        values.len(),
        values.iter().copied().map(f64::from),
    ))
}
