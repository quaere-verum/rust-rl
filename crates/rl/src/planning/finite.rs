//! Bellman solvers for a closed, finite set of live public model contexts.
//!
//! Preparation enumerates each state's entire legal action menu once. Solving
//! uses only the public outcome laws. The model already discounts rewards
//! within each action, so the solver discounts only its continuation.

use crate::models::predictive::PlanningReturns;
use std::{
    collections::{HashMap, VecDeque},
    hash::Hash,
    num::{NonZeroU16, NonZeroU32},
};

use crate::{
    Error, Result,
    core::ensure,
    models::predictive::{EnumerableModel, OutcomeDistribution},
};

use super::guidance::{PolicyGuide, validate_prior, validate_support};

/// Reusable action and outcome laws in the caller's public context order.
pub struct FiniteModel<C, A> {
    returns: PlanningReturns,
    states: Vec<State<C, A>>,
    successor_order: Option<Vec<usize>>,
}

struct State<C, A> {
    context: C,
    actions: Vec<A>,
    outcomes: Vec<Vec<Outcome>>,
}

struct Outcome {
    probability: f64,
    reward: f64,
    continuation: Option<(usize, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Converged,
    IterationLimit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionValue<A> {
    pub action: A,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateValue<C, A> {
    pub context: C,
    pub value: f64,
    action_values: Vec<ActionValue<A>>,
    greedy_index: usize,
}

impl<C, A> StateValue<C, A> {
    /// One-step action values at this solution's returned continuation values.
    /// The order is the model's legal action order, including zero-policy-mass actions.
    pub fn action_values(&self) -> &[ActionValue<A>] {
        &self.action_values
    }

    /// Maximum-value action, resolving ties by the model's legal action order.
    pub fn greedy_action(&self) -> &A {
        &self.action_values[self.greedy_index].action
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Solution<C, A> {
    pub states: Vec<StateValue<C, A>>,
    /// Infinity-norm Bellman residual at the returned values.
    pub residual: f64,
    /// Completed synchronous updates, or one successor-first pass for unit discount.
    pub sweeps: u32,
    pub status: Status,
}

/// Backward-induction values in the supplied context order at every live stage.
#[derive(Debug, Clone, PartialEq)]
pub struct HorizonSolution<C, A> {
    stages: Vec<Vec<StateValue<C, A>>>,
}

impl<C, A> HorizonSolution<C, A> {
    /// Get the values for this many remaining decisions. The boundary at zero
    /// decisions has value zero and no decision rows.
    pub fn states(&self, remaining: u16) -> Option<&[StateValue<C, A>]> {
        self.stages
            .get(usize::from(remaining.checked_sub(1)?))
            .map(Vec::as_slice)
    }
}

impl<C: Clone + Eq + Hash, A: Clone + Eq> FiniteModel<C, A> {
    /// Cache a model's complete public outcome laws.
    ///
    /// Supply each canonical live context once. Every live successor must be
    /// present. An outcome with `next: None` is terminal and has zero
    /// continuation. At unit discount, all live transitions must form a DAG.
    pub fn from_model(
        model: &impl EnumerableModel<Context = C, Action = A>,
        contexts: &[C],
        outcomes_per_action: NonZeroU32,
    ) -> Result<Self> {
        let mut prepared = Self::for_horizon(model, contexts, outcomes_per_action)?;
        if prepared.returns.gamma == 1. {
            prepared.successor_order = Some(successor_first_order(&prepared.states)?);
        }
        Ok(prepared)
    }

    /// Cache complete public outcome laws for a finite decision horizon.
    /// Cycles are allowed at any discount. Include a clock in the public
    /// context when legal actions, rewards, or transitions depend on time.
    pub fn for_horizon(
        model: &impl EnumerableModel<Context = C, Action = A>,
        contexts: &[C],
        outcomes_per_action: NonZeroU32,
    ) -> Result<Self> {
        let returns = model.planning_returns();
        returns.validate()?;
        let indexes: HashMap<_, _> = contexts
            .iter()
            .enumerate()
            .map(|(index, context)| (context, index))
            .collect();
        ensure(
            indexes.len() == contexts.len(),
            "duplicate finite model context",
        )?;

        let mut states = Vec::with_capacity(contexts.len());
        for context in contexts {
            let mut actions = Vec::new();
            model.actions(context, &mut actions);
            ensure(
                !actions.is_empty(),
                "live finite context has no legal actions",
            )?;
            validate_support(&actions)?;
            let outcomes = actions
                .iter()
                .map(|action| {
                    let law = model.enumerate(context, action, outcomes_per_action)?;
                    prepare_outcomes(law, &indexes, outcomes_per_action, returns.gamma)
                })
                .collect::<Result<_>>()?;
            states.push(State {
                context: context.clone(),
                actions,
                outcomes,
            });
        }

        Ok(Self {
            returns,
            states,
            successor_order: None,
        })
    }

    pub fn returns(&self) -> PlanningReturns {
        self.returns
    }

    /// Solve the optimal finite-horizon Bellman equations with a zero final value.
    /// Each action consumes one decision and its duration discounts continuation.
    pub fn backward_induction(&self, horizon: NonZeroU16) -> Result<HorizonSolution<C, A>> {
        self.solve_horizon(horizon, |_, _, _| Ok(None))
    }

    /// Evaluate a fixed, possibly nonstationary policy on the same outcome laws.
    /// The callback returns probabilities in the ordered legal action menu for
    /// a context and its positive remaining decision count.
    pub fn evaluate_horizon_policy(
        &self,
        horizon: NonZeroU16,
        mut policy: impl FnMut(u16, &C, &[A]) -> Result<Vec<f64>>,
    ) -> Result<HorizonSolution<C, A>> {
        self.solve_horizon(horizon, |remaining, context, actions| {
            Ok(Some(normalized_prior(
                actions,
                policy(remaining, context, actions)?,
            )?))
        })
    }

    fn solve_horizon(
        &self,
        horizon: NonZeroU16,
        mut prior: impl FnMut(u16, &C, &[A]) -> Result<Option<Vec<f64>>>,
    ) -> Result<HorizonSolution<C, A>> {
        let mut values = vec![0.; self.states.len()];
        let mut stages = Vec::with_capacity(usize::from(horizon.get()));
        for remaining in 1..=horizon.get() {
            let stage = self
                .states
                .iter()
                .map(|state| {
                    let prior = prior(remaining, &state.context, &state.actions)?;
                    evaluate_state(state, &values, prior.as_deref())
                })
                .collect::<Result<Vec<_>>>()?;
            for (value, row) in values.iter_mut().zip(&stage) {
                *value = row.value;
            }
            stages.push(stage);
        }
        Ok(HorizonSolution { stages })
    }

    /// Solve the optimal Bellman equations. For discounted models the
    /// stopping threshold is `value_tolerance * (1 - gamma)`, which bounds
    /// value error by `value_tolerance` in exact Bellman arithmetic.
    /// Unit-discount acyclic models complete in one successor-first pass.
    pub fn value_iteration(
        &self,
        max_sweeps: NonZeroU32,
        value_tolerance: f64,
    ) -> Result<Solution<C, A>> {
        self.solve(None, max_sweeps, value_tolerance)
    }

    /// Solve the Bellman equations for a frozen public policy. Request and
    /// validate one probability row per context before iterating.
    pub fn evaluate_policy(
        &self,
        policy: &(impl PolicyGuide<C, A> + ?Sized),
        max_sweeps: NonZeroU32,
        value_tolerance: f64,
    ) -> Result<Solution<C, A>> {
        let priors = self
            .states
            .iter()
            .map(|state| {
                normalized_prior(
                    &state.actions,
                    policy.prior(&state.context, &state.actions)?,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.solve(Some(&priors), max_sweeps, value_tolerance)
    }

    fn solve(
        &self,
        priors: Option<&[Vec<f64>]>,
        max_sweeps: NonZeroU32,
        value_tolerance: f64,
    ) -> Result<Solution<C, A>> {
        ensure(
            value_tolerance.is_finite() && value_tolerance > 0.,
            "invalid finite planning value tolerance",
        )?;
        let prior = |index: usize| priors.map(|laws| laws[index].as_slice());
        let mut values = vec![0.; self.states.len()];
        let backup = |index: usize, values: &[f64]| {
            bellman(
                self.states[index]
                    .outcomes
                    .iter()
                    .map(|outcomes| action_value(outcomes, values)),
                prior(index),
            )
        };
        let threshold = value_tolerance * (1. - self.returns.gamma);
        let uncached_order = (self.returns.gamma == 1. && self.successor_order.is_none())
            .then(|| successor_first_order(&self.states))
            .transpose()?;
        let sweeps = if let Some(order) = self.successor_order.as_ref().or(uncached_order.as_ref())
        {
            for &index in order {
                values[index] = backup(index, &values)?;
            }
            1
        } else {
            let mut sweeps = 0;
            let mut updated = values.clone();
            loop {
                let mut residual = 0_f64;
                for (index, value) in updated.iter_mut().enumerate() {
                    *value = backup(index, &values)?;
                    residual = residual.max((*value - values[index]).abs());
                }
                if residual <= threshold || sweeps == max_sweeps.get() {
                    break sweeps;
                }
                std::mem::swap(&mut values, &mut updated);
                sweeps += 1;
            }
        };

        let mut states = Vec::with_capacity(self.states.len());
        let mut residual = 0_f64;
        for (index, state) in self.states.iter().enumerate() {
            let mut row = evaluate_state(state, &values, prior(index))?;
            residual = residual.max((row.value - values[index]).abs());
            row.value = values[index];
            states.push(row);
        }
        let residual = finite_value(residual)?;
        let status = if residual <= threshold {
            Status::Converged
        } else {
            Status::IterationLimit
        };
        Ok(Solution {
            states,
            residual,
            sweeps,
            status,
        })
    }
}

fn normalized_prior<A: Eq>(actions: &[A], mut prior: Vec<f64>) -> Result<Vec<f64>> {
    validate_prior(actions, &prior)?;
    let mass = prior.iter().sum::<f64>();
    prior
        .iter_mut()
        .for_each(|probability| *probability /= mass);
    Ok(prior)
}

fn evaluate_state<C: Clone, A: Clone>(
    state: &State<C, A>,
    values: &[f64],
    prior: Option<&[f64]>,
) -> Result<StateValue<C, A>> {
    let action_values = state
        .actions
        .iter()
        .zip(&state.outcomes)
        .map(|(action, outcomes)| {
            Ok(ActionValue {
                action: action.clone(),
                value: action_value(outcomes, values)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let value = bellman(action_values.iter().map(|row| Ok(row.value)), prior)?;
    let mut greedy_index = 0;
    for (index, row) in action_values.iter().enumerate().skip(1) {
        if row.value > action_values[greedy_index].value {
            greedy_index = index;
        }
    }
    Ok(StateValue {
        context: state.context.clone(),
        value,
        action_values,
        greedy_index,
    })
}

fn prepare_outcomes<C: Eq + Hash, O: Eq>(
    law: OutcomeDistribution<C, O>,
    indexes: &HashMap<&C, usize>,
    cap: NonZeroU32,
    gamma: f64,
) -> Result<Vec<Outcome>> {
    ensure(
        law.complete && !law.outcomes.is_empty() && law.outcomes.len() <= cap.get() as usize,
        "finite model requires a complete outcome law within the requested cap",
    )?;
    let mass = law.validated_mass()?;
    law.outcomes
        .into_iter()
        .map(|row| {
            let transition = row.transition;
            let continuation = transition
                .next
                .map(|context| {
                    let index = indexes.get(&context).ok_or_else(|| {
                        Error::Config(
                            "finite model successor is outside the supplied contexts".into(),
                        )
                    })?;
                    Ok::<_, Error>((*index, gamma.powi(i32::from(transition.duration.get()))))
                })
                .transpose()?;
            Ok(Outcome {
                probability: row.probability / mass,
                reward: transition.reward,
                continuation,
            })
        })
        .collect()
}

fn successor_first_order<C, A>(states: &[State<C, A>]) -> Result<Vec<usize>> {
    let mut pending = vec![0; states.len()];
    let mut predecessors = vec![Vec::new(); states.len()];
    for (index, state) in states.iter().enumerate() {
        for (successor, _) in state
            .outcomes
            .iter()
            .flatten()
            .filter_map(|row| row.continuation)
        {
            pending[index] += 1;
            predecessors[successor].push(index);
        }
    }
    let mut ready: VecDeque<_> = (0..states.len())
        .filter(|&index| pending[index] == 0)
        .collect();
    let mut order = Vec::with_capacity(states.len());
    while let Some(index) = ready.pop_front() {
        order.push(index);
        for &before in &predecessors[index] {
            pending[before] -= 1;
            if pending[before] == 0 {
                ready.push_back(before);
            }
        }
    }
    ensure(
        order.len() == states.len(),
        "unit-discount finite model has a live cycle",
    )?;
    Ok(order)
}

fn action_value(outcomes: &[Outcome], values: &[f64]) -> Result<f64> {
    finite_value(
        outcomes
            .iter()
            .map(|row| {
                let continuation = row
                    .continuation
                    .map_or(0., |(index, discount)| discount * values[index]);
                row.probability * (row.reward + continuation)
            })
            .sum(),
    )
}

fn bellman(mut actions: impl Iterator<Item = Result<f64>>, prior: Option<&[f64]>) -> Result<f64> {
    if let Some(prior) = prior {
        let value = actions
            .zip(prior)
            .try_fold(0., |sum, (action, probability)| {
                Ok::<_, Error>(sum + probability * action?)
            })?;
        finite_value(value)
    } else {
        actions.try_fold(f64::NEG_INFINITY, |best, action| Ok(best.max(action?)))
    }
}

fn finite_value(value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::Numerical(
            "nonfinite finite-model Bellman value".into(),
        ))
    }
}
