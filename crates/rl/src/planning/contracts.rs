//! Bounded deliberation beside the physical interaction contract.
//!
//! Planning inputs are public. A planner neither steps an environment nor
//! inserts a transition into its history.
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanningBudget {
    Expansions(NonZeroU32),
    Simulations {
        trajectories: NonZeroU32,
        transitions: NonZeroU32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Deliberation {
    Execute,
    Plan(PlanningBudget),
    Interrupt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStatus {
    Exact,
    Bounded,
    Approximate,
    Exhausted,
    NoAction,
    /// The hypothetical model failed during a request. The caller can use
    /// its public fallback and charge the work already attempted.
    Failed,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanningWork {
    /// Expanded search nodes. For memoized DP this is the number of memo
    /// misses expanded below the root. Sampled search counts tree nodes.
    pub expansions: u32,
    pub simulations: u32,
    /// Attempted sampled model transitions or returned exact outcome rows,
    /// including rollout and abandoned search work.
    pub transitions: u32,
    pub cache_hits: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionEstimate<A> {
    pub action: A,
    pub value: f64,
    pub upper_bound: Option<f64>,
    pub visits: Option<u32>,
}

pub struct PlanningRequest<'a, C, A> {
    pub context: &'a C,
    pub feasible: &'a [A],
    pub budget: PlanningBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanDecision<A, C> {
    pub selected: Option<C>,
    pub estimates: Vec<ActionEstimate<A>>,
    pub model: [u8; 32],
    pub planner: &'static str,
    pub work: PlanningWork,
    pub status: PlanStatus,
    pub failure: Option<String>,
}
