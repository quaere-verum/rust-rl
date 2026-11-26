//! Finite contingent Bellman search over an enumerable public model.
//!
//! Decision nodes maximize over actions. Action nodes average over their
//! public outcomes, including physical-time discount. Depth and work cutoffs
//! retain intervals for the uncomputed continuation. Policy guidance only
//! orders expansions. All value certificates are relative to the supplied law
//! and bounds, with the numerical accuracy of f64 Bellman arithmetic.

use super::guidance::{PolicyGuide, validate_prior, validate_support};
use crate::planning::{
    ActionEstimate, PlanDecision, PlanStatus, PlanningBudget, PlanningRequest, PlanningWork,
};
use crate::{Error, Result, core::ensure, models::predictive::EnumerableModel};
use std::{
    num::{NonZeroU16, NonZeroU32},
    time::{Duration, Instant},
};

/// Bounds on complete modeled return, including continuation beyond a cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnBounds {
    pub lower: f64,
    pub upper: f64,
}

impl ReturnBounds {
    fn validate(self) -> Result<()> {
        ensure(
            self.lower.is_finite() && self.upper.is_finite() && self.lower <= self.upper,
            "invalid contingent return interval",
        )
    }

    fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Maximum planned semantic actions along a public branch.
    pub depth: NonZeroU16,
    /// Maximum enumerable outcome rows, including incomplete laws.
    pub max_transitions: NonZeroU32,
    /// A range that covers every complete return from any admitted action
    /// and public state, in the model's declared units.
    pub return_bounds: ReturnBounds,
}

type LeafBounds<'a, C> = &'a dyn Fn(&C) -> Result<ReturnBounds>;

pub struct Options<'a, C, A> {
    /// Must return a normalized law over the supplied public action support.
    pub guide: Option<&'a dyn PolicyGuide<C, A>>,
    /// Optional valid bounds on the optimal value at a live cutoff context.
    pub leaf_bounds: Option<LeafBounds<'a, C>>,
}

impl<C, A> Default for Options<'_, C, A> {
    fn default() -> Self {
        Self {
            guide: None,
            leaf_bounds: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport<A> {
    pub decision: PlanDecision<A, A>,
    /// Attempted calls to the bounded outcome enumeration.
    pub enumerations: u64,
    pub guide_evaluations: u64,
    pub bounds_evaluations: u64,
    pub cutoffs: u64,
    pub depth_reached: u16,
    pub latency: Duration,
}

/// Evaluate every ordered root action and choose the best lower bound, with
/// ties resolved by the original action order. A root consumes one expansion.
/// Split the remaining work among its actions, carrying forward unused work.
/// A complete probability law must have finite positive masses summing to one
/// and distinct public outcome keys. Partial distributions only charge work.
pub fn search<M: EnumerableModel>(
    model: &M,
    model_id: [u8; 32],
    request: PlanningRequest<'_, M::Context, M::Action>,
    config: Config,
    options: Options<'_, M::Context, M::Action>,
) -> Result<SearchReport<M::Action>> {
    let started = Instant::now();
    let returns = model.planning_returns();
    returns.validate()?;
    config.return_bounds.validate()?;
    ensure(config.depth.get() <= 128, "contingent depth exceeds 128")?;
    let PlanningBudget::Expansions(expansions) = request.budget else {
        return Err(Error::Config(
            "contingent search needs an expansion budget".into(),
        ));
    };
    let mut actions = Vec::new();
    model.actions(request.context, &mut actions);
    validate_support(&actions)?;
    ensure(
        actions == request.feasible,
        "contingent root actions differ from the ordered model support",
    )?;

    let mut state = Search {
        model,
        config,
        options,
        gamma: returns.gamma,
        limits: Limits {
            expansions: expansions.get(),
            transitions: config.max_transitions.get(),
        },
        report: SearchReport {
            decision: PlanDecision {
                selected: None,
                estimates: Vec::new(),
                model: model_id,
                planner: "finite_contingent",
                work: PlanningWork::default(),
                status: PlanStatus::NoAction,
                failure: None,
            },
            enumerations: 0,
            guide_evaluations: 0,
            bounds_evaluations: 0,
            cutoffs: 0,
            depth_reached: 0,
            latency: Duration::ZERO,
        },
    };
    if !actions.is_empty() {
        state.report.decision.work.expansions = 1;
        if let Err(error) = state.root(request.context, &actions) {
            state.report.decision.status = PlanStatus::Failed;
            state.report.decision.failure = Some(error.to_string());
        }
    }
    state.report.latency = started.elapsed();
    Ok(state.report)
}

#[derive(Clone, Copy)]
struct Limits {
    expansions: u32,
    transitions: u32,
}

struct Search<'a, 'guide, M: EnumerableModel> {
    model: &'a M,
    config: Config,
    options: Options<'guide, M::Context, M::Action>,
    gamma: f64,
    limits: Limits,
    report: SearchReport<M::Action>,
}

impl<M: EnumerableModel> Search<'_, '_, M> {
    fn root(&mut self, root: &M::Context, actions: &[M::Action]) -> Result<()> {
        let full = self.limits;
        let order = self.order(root, actions)?;
        let mut bounds = vec![self.config.return_bounds; actions.len()];
        let count = u32::try_from(actions.len())
            .map_err(|_| Error::Config("too many contingent root actions".into()))?;
        for (offset, index) in (0..count).zip(order) {
            let work = self.report.decision.work;
            let left = count - offset;
            self.limits = Limits {
                expansions: work.expansions + (full.expansions - work.expansions).div_ceil(left),
                transitions: work.transitions
                    + (full.transitions - work.transitions).div_ceil(left),
            };
            bounds[index] = self.action(root, &actions[index], self.config.depth.get())?;
        }

        let mut best = 0;
        for index in 1..bounds.len() {
            if bounds[index].lower > bounds[best].lower {
                best = index;
            }
        }
        self.report.decision.selected = Some(actions[best].clone());
        self.report.decision.status = if bounds.iter().all(|value| value.lower == value.upper) {
            PlanStatus::Exact
        } else {
            PlanStatus::Bounded
        };
        self.report.decision.estimates = actions
            .iter()
            .cloned()
            .zip(bounds)
            .map(|(action, value)| ActionEstimate {
                action,
                value: value.lower,
                upper_bound: Some(value.upper),
                visits: None,
            })
            .collect();
        Ok(())
    }

    fn order(&mut self, context: &M::Context, actions: &[M::Action]) -> Result<Vec<usize>> {
        let mut order: Vec<_> = (0..actions.len()).collect();
        if let Some(guide) = self.options.guide {
            self.report.guide_evaluations += 1;
            let prior = guide.prior(context, actions)?;
            validate_prior(actions, &prior)?;
            order.sort_unstable_by(|&a, &b| prior[b].total_cmp(&prior[a]).then_with(|| a.cmp(&b)));
        }
        Ok(order)
    }

    fn state(&mut self, context: &M::Context, depth: u16) -> Result<ReturnBounds> {
        let work = &mut self.report.decision.work;
        if depth == 0
            || work.expansions == self.limits.expansions
            || work.transitions == self.limits.transitions
        {
            self.report.cutoffs += 1;
            let range = self.config.return_bounds;
            if let Some(bounds) = self.options.leaf_bounds {
                self.report.bounds_evaluations += 1;
                let value = bounds(context)?;
                value.validate()?;
                ensure(
                    value.lower >= range.lower && value.upper <= range.upper,
                    "contingent leaf interval exceeds the declared return range",
                )?;
                return Ok(value);
            }
            return Ok(range);
        }

        work.expansions += 1;
        let mut actions = Vec::new();
        self.model.actions(context, &mut actions);
        validate_support(&actions)?;
        ensure(
            !actions.is_empty(),
            "live contingent successor has no actions",
        )?;
        let range = self.config.return_bounds;
        let mut value = ReturnBounds::point(range.lower);
        for index in self.order(context, &actions)? {
            let estimate = self.action(context, &actions[index], depth)?;
            value.lower = value.lower.max(estimate.lower);
            value.upper = value.upper.max(estimate.upper);
            if value.lower == range.upper {
                break;
            }
        }
        Ok(value)
    }

    fn action(
        &mut self,
        context: &M::Context,
        action: &M::Action,
        depth: u16,
    ) -> Result<ReturnBounds> {
        let range = self.config.return_bounds;
        let work = &mut self.report.decision.work;
        let Some(cap) = NonZeroU32::new(self.limits.transitions - work.transitions) else {
            self.report.cutoffs += 1;
            return Ok(range);
        };
        self.report.enumerations += 1;
        self.report.depth_reached = self
            .report
            .depth_reached
            .max(self.config.depth.get() + 1 - depth);
        let law = self.model.enumerate(context, action, cap)?;
        let rows = law.outcomes.len();
        work.transitions += rows.min(cap.get() as usize) as u32;
        ensure(
            rows <= cap.get() as usize,
            "contingent model exceeded its outcome cap",
        )?;
        if !law.complete {
            self.report.cutoffs += 1;
            return Ok(range);
        }
        let total = law.validated_mass()?;
        let mut value = ReturnBounds::point(0.);
        for row in law.outcomes {
            let transition = row.transition;
            let discount = self.gamma.powi(i32::from(transition.duration.get()));
            let continuation = match transition.next {
                Some(next) if discount > 0. => self.state(&next, depth - 1)?,
                _ => ReturnBounds::point(0.),
            };
            let p = row.probability / total;
            value.lower += p * (transition.reward + discount * continuation.lower);
            value.upper += p * (transition.reward + discount * continuation.upper);
        }
        value.validate()?;
        let tolerance = 1e-12 * range.lower.abs().max(range.upper.abs()).max(1.);
        ensure(
            value.lower <= range.upper + tolerance && value.upper >= range.lower - tolerance,
            "contingent return is outside the declared range",
        )?;
        Ok(ReturnBounds {
            lower: value.lower.clamp(range.lower, range.upper),
            upper: value.upper.clamp(range.lower, range.upper),
        })
    }
}
