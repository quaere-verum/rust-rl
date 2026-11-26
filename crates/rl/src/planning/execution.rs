//! Public-observation continuations for existing actors and collectors.
//!
//! Domains retain primitive execution, duration accounting, interruption during
//! an option and physical feasibility. This helper runs at an option boundary;
//! it neither advances time nor fabricates an environment transition.
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplanReason {
    DifferentAction,
    UnmatchedOutcome,
    Infeasible,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Continuation<A> {
    Continue(A),
    Replan(ReplanReason),
}

/// Resolve a contingent plan against a factual completed action and its public
/// outcome. Branch keys must be unique and identify the complete public outcome
/// relevant to continuation. No hidden particle or imagined successor is used.
///
/// The caller supplies current public feasibility (including resources and the
/// first executable primitive). Consuming branches transfers the selected plan
/// without cloning its route. A mismatch never calls the feasibility callback.
pub fn resolve_continuation<P: PartialEq, O: PartialEq, A>(
    chosen: &P,
    executed: &P,
    outcome: &O,
    branches: impl IntoIterator<Item = (O, A)>,
    feasible: impl FnOnce(&A) -> Result<bool>,
) -> Result<Continuation<A>> {
    if chosen != executed {
        return Ok(Continuation::Replan(ReplanReason::DifferentAction));
    }
    let Some((_, action)) = branches.into_iter().find(|(key, _)| key == outcome) else {
        return Ok(Continuation::Replan(ReplanReason::UnmatchedOutcome));
    };
    if feasible(&action)? {
        Ok(Continuation::Continue(action))
    } else {
        Ok(Continuation::Replan(ReplanReason::Infeasible))
    }
}
