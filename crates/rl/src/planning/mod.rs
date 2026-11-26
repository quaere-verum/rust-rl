//! Bounded algorithms over public predictive states.
//!
//! This module owns planning requests, budgets, and work receipts.
//! Planners consume computation without stepping an environment. Existing
//! stateful actors execute their chosen commitments through factual collection.
pub mod contingent;
mod contracts;
pub mod execution;
pub mod finite;
pub mod guidance;
pub mod policy_evaluation;
pub mod pomcp;
pub mod rollout;
pub mod targets;

pub use contracts::{
    ActionEstimate, Deliberation, PlanDecision, PlanStatus, PlanningBudget, PlanningRequest,
    PlanningWork,
};
