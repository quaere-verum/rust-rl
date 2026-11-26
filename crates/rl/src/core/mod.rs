//! Backend-independent environment and learning contracts.
pub mod action;
pub mod actor;
pub mod anchors;
pub mod categorical_behavior;
pub mod env;
pub mod graph;
pub mod learner;
pub mod reconstruction;
pub mod recording;
pub mod reward;
pub mod state_key;
pub mod stateful;
pub mod stateful_on_policy;
pub mod structured;
pub mod temporal;
pub mod transition;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("invalid action: {0}")]
    Action(String),
    #[error("invalid state: {0}")]
    State(String),
    #[error("vector environment contract violated during {operation}, lane {lane:?}: {reason}")]
    VectorEnvContract {
        operation: &'static str,
        lane: Option<usize>,
        reason: String,
    },
    #[error("numerical failure: {0}")]
    Numerical(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Config(message.into()))
    }
}
pub mod checkpoint;
