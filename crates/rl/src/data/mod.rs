//! Trajectory mathematics and storage, independent of autodiff graphs.
#[cfg(feature = "experimental-acer")]
pub mod categorical_replay;
pub mod gae;
pub mod minibatch;
pub mod pathwise;
pub mod policy_supervision;
pub mod replay;
pub mod returns;
pub mod rollout;
pub mod temporal;
pub mod temporal_on_policy;
pub mod value_supervision;
