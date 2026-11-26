//! Bounded conditional expectation for enumerable public outcomes.
use super::{Horizon, RolloutChoice, RolloutPolicy, RolloutReport, begin, policy_law};
use crate::planning::{ActionEstimate, PlanStatus, PlanningBudget, PlanningRequest, PlanningWork};
use crate::{
    Error, Result,
    core::ensure,
    models::predictive::{EnumerableModel, Transition},
    planning::guidance::validate_support,
};
use std::{num::NonZeroU32, time::Instant};

type Estimate<A, O> = (RolloutChoice<A, O>, Vec<ActionEstimate<A>>);
type ActionValue<A, O> = (f64, Vec<(O, A)>);
type Computed<T> = std::result::Result<T, Stop>;

/// The request's expansion budget counts every evaluated live context,
/// including the contexts evaluated under the policy tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactConfig {
    pub horizon: Horizon,
    /// Caps public outcome rows returned by the model, including rows dropped
    /// when a search cannot finish. Internal legality and probability
    /// preparation are measured in the plan's latency.
    pub max_transitions: NonZeroU32,
}

/// Compute bounded model-relative expectations. Return exact point estimates
/// only after completing every root action's planned and baseline continuation.
pub fn exact<M: EnumerableModel>(
    model: &M,
    model_id: [u8; 32],
    request: PlanningRequest<'_, M::Context, M::Action>,
    baseline: RolloutPolicy<'_, M::Context, M::Action>,
    config: ExactConfig,
) -> Result<RolloutReport<M::Action, M::Observation>> {
    let started = Instant::now();
    config.horizon.validate()?;
    let PlanningBudget::Expansions(max_expansions) = request.budget else {
        return Err(Error::Config(
            "exact rollout requires an expansion budget".into(),
        ));
    };
    let mut report = begin(
        model,
        model_id,
        &request,
        &baseline,
        "bounded_exact_rollout",
    )?;
    if report.can_search() {
        let mut evaluator = Evaluator {
            model,
            baseline,
            max_expansions,
            max_transitions: config.max_transitions,
            work: PlanningWork {
                expansions: 1,
                ..Default::default()
            },
            policy_evaluations: report.policy_evaluations,
        };
        let result = evaluator.root(request.context, request.feasible, config.horizon);
        report.decision.work = evaluator.work;
        report.policy_evaluations = evaluator.policy_evaluations;
        match result {
            Ok((selected, estimates)) => {
                report.decision.status = PlanStatus::Exact;
                report.decision.selected = Some(selected);
                report.decision.estimates = estimates;
            }
            Err(Stop::Exhausted) => report.decision.status = PlanStatus::Exhausted,
            Err(Stop::Invalid(error)) => {
                report.decision.status = PlanStatus::Failed;
                report.decision.failure = Some(error.to_string());
            }
        }
    }
    report.latency = started.elapsed();
    Ok(report)
}

enum Stop {
    Exhausted,
    Invalid(Error),
}

impl From<Error> for Stop {
    fn from(error: Error) -> Self {
        Self::Invalid(error)
    }
}

struct Evaluator<'a, M: EnumerableModel> {
    model: &'a M,
    baseline: RolloutPolicy<'a, M::Context, M::Action>,
    max_expansions: NonZeroU32,
    max_transitions: NonZeroU32,
    work: PlanningWork,
    policy_evaluations: u64,
}

impl<M: EnumerableModel> Evaluator<'_, M> {
    fn root(
        &mut self,
        root: &M::Context,
        actions: &[M::Action],
        horizon: Horizon,
    ) -> Computed<Estimate<M::Action, M::Observation>> {
        let mut estimates = Vec::with_capacity(actions.len());
        let mut best = f64::NEG_INFINITY;
        let mut selected = None;
        for action in actions {
            let (value, continuations) = self.action_value(
                root,
                action,
                horizon.depth.layers().get() - 1,
                horizon.termination_bound.get(),
            )?;
            if value > best {
                best = value;
                selected = Some(RolloutChoice {
                    action: action.clone(),
                    continuations,
                });
            }
            estimates.push(ActionEstimate {
                action: action.clone(),
                value,
                upper_bound: Some(value),
                visits: None,
            });
        }
        Ok((
            selected.expect("validated nonempty rollout support"),
            estimates,
        ))
    }

    fn state_value(
        &mut self,
        context: &M::Context,
        layers: u16,
        remaining: u16,
    ) -> Computed<(f64, Option<M::Action>)> {
        if self.work.expansions == self.max_expansions.get() {
            return Err(Stop::Exhausted);
        }
        self.work.expansions += 1;
        let mut actions = Vec::new();
        self.model.actions(context, &mut actions);
        validate_support(&actions)?;
        ensure(
            !actions.is_empty(),
            "enumerated rollout successor is live without actions",
        )?;
        if layers == 0 {
            self.policy_evaluations += 1;
            let prior = policy_law(&self.baseline, context, &actions)?;
            let total: f64 = prior.iter().sum();
            let mut expected = 0.;
            for (action, probability) in actions
                .iter()
                .zip(prior)
                .filter(|(_, probability)| *probability > 0.)
            {
                expected +=
                    (probability / total) * self.action_value(context, action, 0, remaining)?.0;
            }
            ensure(
                expected.is_finite(),
                "nonfinite baseline rollout expectation",
            )?;
            Ok((expected, None))
        } else {
            let mut best = f64::NEG_INFINITY;
            let mut selected = None;
            for action in actions {
                let (value, _) = self.action_value(context, &action, layers - 1, remaining)?;
                if value > best {
                    best = value;
                    selected = Some(action);
                }
            }
            Ok((best, selected))
        }
    }

    fn action_value(
        &mut self,
        context: &M::Context,
        action: &M::Action,
        layers_after: u16,
        remaining: u16,
    ) -> Computed<ActionValue<M::Action, M::Observation>> {
        let cap = NonZeroU32::new(self.max_transitions.get() - self.work.transitions)
            .ok_or(Stop::Exhausted)?;
        let distribution = self.model.enumerate(context, action, cap)?;
        let count = distribution.outcomes.len();
        // Even abandoned branches consume the model budget.
        self.work.transitions += count.min(cap.get() as usize) as u32;
        ensure(
            count <= cap.get() as usize,
            "model exceeded its enumeration cap",
        )?;
        if !distribution.complete {
            return Err(Stop::Exhausted);
        }
        let mass = distribution.validated_mass()?;
        let mut value = 0.;
        let mut continuations = Vec::new();
        for outcome in distribution.outcomes {
            let Transition {
                observation,
                reward,
                duration,
                next,
            } = outcome.transition;
            let mut branch = reward;
            if let Some(context) = next {
                ensure(
                    remaining > 1,
                    "enumerated trajectory violated its termination bound",
                )?;
                let (value, action) = self.state_value(&context, layers_after, remaining - 1)?;
                branch += self.baseline.returns.gamma.powi(i32::from(duration.get())) * value;
                if let Some(action) = action {
                    continuations.push((observation, action));
                }
            }
            value += (outcome.probability / mass) * branch;
        }
        ensure(value.is_finite(), "nonfinite exact rollout expectation")?;
        Ok((value, continuations))
    }
}
