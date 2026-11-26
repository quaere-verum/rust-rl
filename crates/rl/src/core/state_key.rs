//! Public information states and their representations for finite tables.

use super::stateful::HistoryInput;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::hash::Hash;

/// The relationship between a key and the information state it represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateKeyKind {
    /// A lossless identity for the builder's chosen information state.
    Exact,
    /// A key based on explicit bins or thresholds.
    Discretized,
    /// A deliberately lossy grouping of information states.
    Aggregated,
}

/// Updates private actor memory using only public, causal inputs.
///
/// An exact key is only Markov if the builder's memory contains sufficient
/// information for the task. The environment's hidden state stays private.
pub trait StateKeyBuilder {
    type Memory: Clone;
    type Key: Clone + Eq + Hash + Ord;

    fn kind(&self) -> StateKeyKind;
    fn initial_memory(&self) -> Result<Self::Memory>;
    fn observe(&self, prior: &Self::Memory, input: HistoryInput<'_>) -> Result<Self::Memory>;
    fn key(&self, memory: &Self::Memory) -> Result<Self::Key>;

    fn validate_memory(&self, _memory: &Self::Memory) -> Result<()> {
        Ok(())
    }
}

/// Uses the current public observation symbol as the complete key.
///
/// This mapping fits an environment whose symbol itself is sufficient. For
/// aliased observations, use a builder with memory or another information state.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DiscreteSymbolKey;

impl StateKeyBuilder for DiscreteSymbolKey {
    type Memory = Option<u64>;
    type Key = u64;

    fn kind(&self) -> StateKeyKind {
        StateKeyKind::Exact
    }

    fn initial_memory(&self) -> Result<Self::Memory> {
        Ok(None)
    }

    fn observe(&self, _prior: &Self::Memory, input: HistoryInput<'_>) -> Result<Self::Memory> {
        input
            .observation
            .as_discrete()
            .map(|observation| Some(observation.symbol))
            .ok_or_else(|| Error::Shape("the key builder expects a discrete symbol".into()))
    }

    fn key(&self, memory: &Self::Memory) -> Result<Self::Key> {
        memory.ok_or_else(|| Error::State("the key builder has no observation".into()))
    }
}
