//! One- or two-layer planning with a complete, frozen executable policy tail.
//!
//! The policy guide supplies action probabilities on modeled public states.
//! Its critic is never used as a continuation value. Hypothetical work is
//! bounded separately from the physical interaction and factual replay.
mod exact;

use super::{
    guidance::{PolicyGuide, validate_prior, validate_support},
    pomcp::{self, Controls},
};
use crate::models::predictive::{PlanningModel, PlanningReturns, PredictiveModel};
use crate::planning::{
    Deliberation, PlanDecision, PlanStatus, PlanningBudget, PlanningRequest, PlanningWork,
};
use crate::{Result, core::ensure};
use rand_chacha::ChaCha8Rng;
use std::{
    num::{NonZeroU16, NonZeroU32},
    time::{Duration, Instant},
};

pub use exact::{ExactConfig, exact};

/// Number of decisions selected by lookahead, including the root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    One,
    Two,
}

impl Depth {
    fn layers(self) -> NonZeroU16 {
        match self {
            Self::One => NonZeroU16::MIN,
            Self::Two => NonZeroU16::new(2).unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Horizon {
    pub depth: Depth,
    /// Maximum semantic decisions from the root through the end of the
    /// modeled mission, including those taken under the baseline.
    pub termination_bound: NonZeroU16,
}

impl Horizon {
    fn validate(self) -> Result<()> {
        ensure(
            self.depth.layers() <= self.termination_bound
                && self.termination_bound.get() <= pomcp::MAX_TREE_DEPTH,
            "invalid bounded rollout depth or termination guarantee",
        )
    }
}

/// A caller-owned, immutable policy law and the only continuation used after
/// the planned depth. The law must match how this revision will be executed.
pub struct RolloutPolicy<'a, C, A> {
    pub guide: &'a dyn PolicyGuide<C, A>,
    pub returns: PlanningReturns,
    pub revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Trigger {
    #[default]
    Never,
    /// Deliberate only at genuine choices, with two or more legal actions.
    WhenChoices,
}

impl Trigger {
    pub fn deliberate<A>(self, legal: &[A], budget: PlanningBudget) -> Deliberation {
        if self == Self::WhenChoices && legal.len() > 1 {
            Deliberation::Plan(budget)
        } else {
            Deliberation::Execute
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolloutChoice<A, O> {
    pub action: A,
    /// Second-layer actions keyed by the preceding action's public outcome.
    /// Sampled search lists only outcomes with a visited continuation action.
    /// Use the frozen baseline on an unlisted outcome, including when this list
    /// is empty. Check the physical outcome and action feasibility at execution.
    pub continuations: Vec<(O, A)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolloutReport<A, O> {
    pub decision: PlanDecision<A, RolloutChoice<A, O>>,
    pub returns: PlanningReturns,
    pub baseline_revision: u64,
    /// Actual policy law calls, including fallback preparation and abandoned work.
    pub policy_evaluations: u64,
    pub latency: Duration,
}

/// Root-sampled POMCP budget with an explicit cap on planned history nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampledConfig {
    pub horizon: Horizon,
    pub max_expansions: NonZeroU32,
    pub exploration: f64,
}

/// Bound POMCP's tree to the declared planned layers and sample the baseline's
/// full public policy law through termination on the same hypothetical path.
/// Approximate results require visits for every root action. A second layer can
/// have partial coverage. Execute the baseline at any unlisted public outcome.
/// Partial root coverage or an exhausted cap returns the first legal action
/// with maximal baseline probability, even when the baseline is stochastic.
pub fn sampled<M: PredictiveModel>(
    model: &M,
    model_id: [u8; 32],
    request: PlanningRequest<'_, M::Context, M::Action>,
    baseline: RolloutPolicy<'_, M::Context, M::Action>,
    config: SampledConfig,
    rng: &mut ChaCha8Rng,
) -> Result<RolloutReport<M::Action, M::Observation>> {
    let started = Instant::now();
    config.horizon.validate()?;
    ensure(
        matches!(request.budget, PlanningBudget::Simulations { .. })
            && config.exploration.is_finite()
            && config.exploration >= 0.,
        "sampled rollout requires a simulation budget and valid exploration",
    )?;
    let mut report = begin(model, model_id, &request, &baseline, "bounded_pomcp")?;
    if report.can_search() {
        let mut prior = |context: &M::Context, actions: &[M::Action]| {
            report.policy_evaluations += 1;
            policy_law(&baseline, context, actions)
        };
        let search = pomcp::search_impl(
            model,
            request.context,
            request.feasible,
            pomcp::Config {
                max_depth: config.horizon.termination_bound,
                gamma: baseline.returns.gamma,
                exploration: config.exploration,
            },
            request.budget,
            rng,
            Controls {
                max_expansions: Some(config.max_expansions),
                tree_layers: Some(config.horizon.depth.layers()),
                require_terminal: true,
                rollout_prior: Some(&mut prior),
                ..Default::default()
            },
        )?;
        report.decision.work = search.work;
        report.decision.failure = search.failure;
        report.decision.status = search.status;
        if search.status == PlanStatus::Approximate {
            if let Some(action) = search.action
                && search.estimates.len() == request.feasible.len()
            {
                report.decision.selected = Some(RolloutChoice {
                    action,
                    continuations: search.continuations,
                });
                report.decision.estimates = search.estimates;
            } else {
                report.decision.status = PlanStatus::Exhausted;
            }
        }
    }
    report.latency = started.elapsed();
    Ok(report)
}

fn begin<M: PlanningModel>(
    model: &M,
    model_id: [u8; 32],
    request: &PlanningRequest<'_, M::Context, M::Action>,
    baseline: &RolloutPolicy<'_, M::Context, M::Action>,
    planner: &'static str,
) -> Result<RolloutReport<M::Action, M::Observation>> {
    model.planning_returns().check(&baseline.returns)?;
    let mut actions = Vec::new();
    model.actions(request.context, &mut actions);
    validate_support(&actions)?;
    ensure(
        actions == request.feasible,
        "rollout root support differs from the ordered model actions",
    )?;
    let mut report = RolloutReport {
        decision: PlanDecision {
            selected: None,
            estimates: Vec::new(),
            model: model_id,
            planner,
            work: PlanningWork::default(),
            status: PlanStatus::NoAction,
            failure: None,
        },
        returns: baseline.returns,
        baseline_revision: baseline.revision,
        policy_evaluations: 0,
        latency: Duration::ZERO,
    };
    if !actions.is_empty() {
        report.policy_evaluations = 1;
        let index = match policy_law(baseline, request.context, &actions) {
            Ok(probabilities) => probabilities
                .iter()
                .enumerate()
                .fold(
                    0,
                    |best, (index, &p)| {
                        if p > probabilities[best] { index } else { best }
                    },
                ),
            Err(error) => {
                report.decision.status = PlanStatus::Failed;
                report.decision.failure = Some(error.to_string());
                0
            }
        };
        report.decision.selected = Some(RolloutChoice {
            action: actions.swap_remove(index),
            continuations: Vec::new(),
        });
    }
    Ok(report)
}

impl<A, O> RolloutReport<A, O> {
    fn can_search(&self) -> bool {
        self.decision.selected.is_some() && self.decision.status != PlanStatus::Failed
    }
}

fn policy_law<C, A: Eq>(
    baseline: &RolloutPolicy<'_, C, A>,
    context: &C,
    actions: &[A],
) -> Result<Vec<f64>> {
    let probabilities = baseline.guide.prior(context, actions)?;
    validate_prior(actions, &probabilities)?;
    Ok(probabilities)
}
