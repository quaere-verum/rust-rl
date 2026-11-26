//! Composable reinforcement learning and planning for sequential decisions.
//!
//! Implement [`core::env::Env`] to expose observations and actions, then compose
//! a learner from [`algorithms`] with the execution and replay tools in
//! [`collect`], [`data`] and [`train`]. The [`models`] and [`planning`]
//! modules support explicit and learned transition laws.
//!
//! Default features include neural learners. A build with default features
//! disabled retains the tabular, statistical and planning APIs. Enable
//! `diagnostics` to record experiments for the separate native viewer.
#![forbid(unsafe_code)]

pub mod algorithms;
pub mod approx;
pub mod collect;
pub mod core;
pub mod data;
/// Optional recording, scientific evaluation and historical inspection.
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
pub mod envs;
#[cfg(feature = "neural")]
pub mod imitation;
pub mod models;
#[cfg(feature = "neural")]
pub mod nn;
pub mod objectives;
pub mod planning;
pub mod prelude;
#[cfg(feature = "probes")]
pub mod profiling;
/// Collection and learner state cannot be changed through inspection accessors.
/// ```compile_fail
/// use rl::{prelude::*, envs::lqg::Lqg};
/// fn replace(trainer: &mut OffPolicyTrainer<SyncVectorEnv<Lqg>>) {
///     trainer.config().n_step = 0;
/// }
/// ```
#[cfg_attr(
    feature = "neural",
    doc = " ```compile_fail\nuse rl::prelude::*;\nfn reset_delay(learner: &mut Td3) { learner.inner.updates = 0; }\n```"
)]
pub mod train;
pub use core::{Error, Result};
