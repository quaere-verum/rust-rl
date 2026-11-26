//! Independently selectable public guidance for the existing POMCP engine.
use super::{PolicyGuide, ValueGuide, validate_prior, validate_support};
use crate::models::predictive::{PlanningReturns, PredictiveModel};
use crate::planning::PlanningBudget;
use crate::{
    Result,
    core::ensure,
    planning::pomcp::{self, Controls, RolloutDecision, RolloutStep},
};
pub use pomcp::{Backup, RootSelection, SearchStrategy, TreeSelection};
use rand_chacha::ChaCha8Rng;
use std::num::{NonZeroU16, NonZeroU32};

/// An explicit computation cutoff before the declared semantic horizon.
pub struct ValueCutoff<'a, C> {
    pub depth: NonZeroU16,
    pub guide: &'a dyn ValueGuide<C>,
}

/// Opt-in search controls. `new` retains UCT and model rollout. The model and
/// guides share a declared objective and physical-time discount; the POMCP
/// config's max_depth is the semantic objective horizon.
pub struct SearchOptions<'a, C, A> {
    pub returns: PlanningReturns,
    pub selection: TreeSelection,
    pub policy: Option<&'a dyn PolicyGuide<C, A>>,
    pub rollout_policy: bool,
    pub value: Option<ValueCutoff<'a, C>>,
    /// Counts root and every inserted history node. At the cap, a needed new
    /// node stops search and discards the incomplete trajectory.
    pub max_expansions: Option<NonZeroU32>,
}
impl<'a, C, A> SearchOptions<'a, C, A> {
    pub fn new(returns: PlanningReturns) -> Self {
        Self {
            returns,
            selection: TreeSelection::Uct,
            policy: None,
            rollout_policy: false,
            value: None,
            max_expansions: None,
        }
    }

    pub fn with_strategy(self, strategy: SearchStrategy) -> StrategicSearchOptions<'a, C, A> {
        StrategicSearchOptions {
            guidance: self,
            strategy,
        }
    }
}

/// Guidance and optional alternative tree-search semantics.
pub struct StrategicSearchOptions<'a, C, A> {
    pub guidance: SearchOptions<'a, C, A>,
    pub strategy: SearchStrategy,
}

pub fn search_with_options<G: PredictiveModel>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    config: pomcp::Config,
    budget: PlanningBudget,
    rng: &mut ChaCha8Rng,
    options: SearchOptions<'_, G::Context, G::Action>,
) -> Result<pomcp::Search<G::Action, G::Observation>> {
    search_with_strategy(
        model,
        root,
        feasible,
        config,
        budget,
        rng,
        options.with_strategy(SearchStrategy::default()),
    )
}

/// Search with independent choices for decision backups and root allocation.
/// Halving's visit counts report sample allocation, including candidates
/// eliminated before the final decision.
pub fn search_with_strategy<G: PredictiveModel>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    config: pomcp::Config,
    budget: PlanningBudget,
    rng: &mut ChaCha8Rng,
    options: StrategicSearchOptions<'_, G::Context, G::Action>,
) -> Result<pomcp::Search<G::Action, G::Observation>> {
    let StrategicSearchOptions {
        guidance: options,
        strategy,
    } = options;
    model.planning_returns().check(&options.returns)?;
    ensure(
        config.gamma == options.returns.gamma,
        "planning guide discount mismatch",
    )?;
    let puct = matches!(options.selection, TreeSelection::Puct { .. });
    if let TreeSelection::Puct { exploration } = options.selection {
        ensure(
            exploration.is_finite() && exploration > 0.,
            "PUCT exploration must be finite and positive",
        )?;
    }
    ensure(
        !(puct || options.rollout_policy) || options.policy.is_some(),
        "guided selection requires a public policy prior",
    )?;
    ensure(
        options
            .value
            .as_ref()
            .is_none_or(|v| v.depth < config.max_depth),
        "value cutoff must precede the semantic horizon",
    )?;
    let mut prior = |context: &G::Context, actions: &[G::Action]| {
        options
            .policy
            .expect("validated policy guide")
            .prior(context, actions)
    };
    let mut rollout = |context: &G::Context, _: RolloutStep| {
        let mut actions = Vec::new();
        model.actions(context, &mut actions);
        validate_support(&actions)?;
        if actions.is_empty() {
            return Ok(RolloutDecision::End);
        }
        let probabilities = options
            .policy
            .expect("validated policy guide")
            .prior(context, &actions)?;
        validate_prior(&actions, &probabilities)?;
        let mut best = 0;
        for index in 1..probabilities.len() {
            if probabilities[index] > probabilities[best] {
                best = index;
            }
        }
        Ok(RolloutDecision::Act(actions[best].clone()))
    };
    let mut leaf = |context: &G::Context, remaining| {
        let mut actions = Vec::new();
        model.actions(context, &mut actions);
        validate_support(&actions)?;
        if actions.is_empty() {
            return Ok(0.);
        }
        options
            .value
            .as_ref()
            .expect("configured cutoff")
            .guide
            .value(context, remaining, &options.returns)
    };
    let mut controls = Controls {
        selection: options.selection,
        strategy,
        max_expansions: options.max_expansions,
        ..Default::default()
    };
    if puct {
        controls.prior = Some(&mut prior);
    }
    if options.rollout_policy {
        controls.rollout = Some(&mut rollout);
    }
    if let Some(value) = &options.value {
        controls.cutoff = Some((value.depth.get(), &mut leaf));
    }
    pomcp::search_impl(model, root, feasible, config, budget, rng, controls)
}
