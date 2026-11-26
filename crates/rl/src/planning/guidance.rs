//! Public policy and value guidance for POMCP.
mod search;
use crate::models::predictive::PlanningReturns;
use crate::{Result, core::ensure};
pub use search::{
    Backup, RootSelection, SearchOptions, SearchStrategy, StrategicSearchOptions, TreeSelection,
    ValueCutoff, search_with_options, search_with_strategy,
};

pub trait PolicyGuide<C, A> {
    /// Normalized probabilities in the exact supplied feasible-action order.
    /// The input is public state; no hypothetical hidden particle is supplied.
    fn prior(&self, context: &C, feasible: &[A]) -> Result<Vec<f64>>;
}

pub trait ValueGuide<C> {
    /// Continuation from this public context with this many semantic decisions
    /// remaining. Rewards and discount follow `returns`. A computation cutoff
    /// is not physical termination; a truly terminal model returns `next=None`.
    fn value(&self, context: &C, remaining: u16, returns: &PlanningReturns) -> Result<f64>;
}

pub(crate) fn validate_support<A: PartialEq>(feasible: &[A]) -> Result<()> {
    ensure(
        feasible
            .iter()
            .enumerate()
            .all(|(index, action)| !feasible[..index].contains(action)),
        "duplicate planning action identity",
    )
}

pub fn validate_prior<A: PartialEq>(feasible: &[A], prior: &[f64]) -> Result<()> {
    validate_support(feasible)?;
    ensure(
        !feasible.is_empty()
            && prior.len() == feasible.len()
            && prior.iter().all(|p| p.is_finite() && *p >= 0.)
            && (prior.iter().sum::<f64>() - 1.).abs() <= 1e-9,
        "planning prior must be normalized over the ordered feasible actions",
    )
}
