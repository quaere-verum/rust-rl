//! Recurrent lightweight models over the shared factual public tape.
//! Each frame consumes newly available events in order, then one decision token.
mod config;
mod distributional_q;
mod history;
mod input;
mod law;
mod policy;
mod portable;
mod q;
mod statistical;
mod value;

pub use config::RecurrentLightweightConfig;
pub use distributional_q::{
    RecurrentCandidatePrediction, RecurrentDistributionalQEvaluation,
    RecurrentLightweightDistributionalQ,
};
pub use law::ConditionalPolicyLaw;
pub use policy::{RecurrentLightweightPolicy, RecurrentPolicyEvaluation};
pub use portable::RecurrentLightweightMemory;
pub use statistical::FrozenRecurrentFeatures;
pub use value::RecurrentLightweightValue;

use super::LightweightEncoder;
use crate::{
    core::stateful::HistoryInput,
    nn::{network::Mlp, temporal::GruEncoder},
};
use burn::{module::Module, tensor::Tensor};

/// Shared entity and complete-action parameters with explicit caller-owned memory.
// The checkpoint signature binds this historical module path and field order.
#[derive(Module, Debug)]
pub struct RecurrentLightweightQ {
    pub encoder: LightweightEncoder,
    pub operations: Mlp,
    pub event_projection: Mlp,
    pub decision_projection: Mlp,
    pub memory: GruEncoder,
    pub head: Mlp,
    #[module(skip)]
    config: RecurrentLightweightConfig,
}

/// A factual history starting at a genuine reset, including the final successor.
#[derive(Debug, Clone, Copy)]
pub struct StructuredHistoryInput<'a> {
    pub inputs: &'a [HistoryInput<'a>],
    /// Detach memory before the first token at this row. Zero enables gradients
    /// through the full prefix. This boundary does not affect forward values.
    pub learn_start: usize,
}

pub struct StructuredHistoryValues {
    /// Post-observation memory, `[observations, hidden_width]`.
    pub states: Tensor<2>,
    /// Values in public candidate order. Empty candidate rows are `None`.
    pub decisions: Vec<Option<Tensor<1>>>,
}

/// Caller-owned memory and the start of the last frame's new evidence. Use the
/// same public observation with `evaluate_decision` as with `assimilate`.
#[derive(Debug, Clone)]
pub struct RecurrentLightweightState {
    pub memory: Tensor<2>,
    pub event_start: usize,
}

/// Completed stages for caller-owned profiling on synchronous backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecurrentLightweightStage {
    Validation,
    Packing,
    Encoder(super::LightweightStage),
    PreviousAction,
    Tokens,
    Memory,
    Candidates,
    Binding,
}
