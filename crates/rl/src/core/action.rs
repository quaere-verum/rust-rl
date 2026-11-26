use super::structured::{ArgumentKind, EntityId, FeatureIdentity, feature_fields, named};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ActionArgument {
    Entity(EntityId),
    Location {
        x: usize,
        y: usize,
    },
    /// Coordinates in a specific public spatial region, independent of row order.
    RegionalLocation {
        region: u32,
        x: usize,
        y: usize,
    },
}

/// A complete action in its versioned grammar. The argument is a public key,
/// never a row offset in an observation or packed batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ConditionalAction {
    pub operation: usize,
    pub argument: Option<ActionArgument>,
}

impl ConditionalAction {
    pub(crate) fn in_region(mut self, primary_region: u32) -> Self {
        if let Some(ActionArgument::Location { x, y }) = self.argument {
            self.argument = Some(ActionArgument::RegionalLocation {
                region: primary_region,
                x,
                y,
            });
        }
        self
    }
}

/// Identity of a complete finite action within its versioned action space.
///
/// Use `Observation::finite_action_key` to normalize location aliases for a
/// specific frame. Execute the original published action from the legal menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FiniteActionKey {
    Discrete(usize),
    Conditional(ConditionalAction),
}

/// Deterministic selection from a semantic complete-action probability law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalGreedy {
    /// Maximize operation mass, then maximize the argument mass within it.
    Hierarchical,
    /// Maximize the probability of an individual complete action.
    JointMap,
}

impl ConditionalGreedy {
    /// Zero mass denotes an unavailable action. Ties prefer the lowest
    /// operation and then the lowest semantic key. Accumulate operation mass
    /// in key order so results are independent of candidate order.
    pub fn select(
        self,
        probabilities: impl IntoIterator<Item = (ConditionalAction, f64)>,
    ) -> Result<(ConditionalAction, f64)> {
        let mut rows: Vec<_> = probabilities.into_iter().collect();
        super::ensure(
            rows.iter().all(|(_, p)| (0. ..=1.).contains(p)),
            "invalid conditional action probability",
        )?;
        rows.retain(|(_, p)| *p > 0.);
        rows.sort_unstable_by_key(|&(action, _)| action);
        super::ensure(
            !rows.is_empty() && rows.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "empty or duplicate conditional action support",
        )?;
        let candidates = match self {
            Self::JointMap => rows.as_slice(),
            Self::Hierarchical => {
                rows.chunk_by(|a, b| a.0.operation == b.0.operation)
                    .map(|group| (group, group.iter().map(|(_, p)| p).sum::<f64>()))
                    .max_by(|(a, p), (b, q)| {
                        p.total_cmp(q)
                            .then_with(|| b[0].0.operation.cmp(&a[0].0.operation))
                    })
                    .expect("validated live support")
                    .0
            }
        };
        Ok(*candidates
            .iter()
            .max_by(|(a, p), (b, q)| p.total_cmp(q).then_with(|| b.cmp(a)))
            .expect("validated live support"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operation {
    pub name: String,
    pub argument: ArgumentKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionGrammar {
    pub namespace: String,
    pub version: u32,
    pub operations: Vec<Operation>,
    pub candidate_features: Vec<FeatureIdentity>,
}

impl ActionGrammar {
    pub fn validate(&self) -> Result<()> {
        super::ensure(
            self.version > 0
                && !self.namespace.trim().is_empty()
                && self.namespace.len() <= 4096
                && !self.operations.is_empty()
                && self.operations.len() <= 1024
                && named(
                    self.operations
                        .iter()
                        .map(|operation| operation.name.as_str()),
                )
                && feature_fields(&self.candidate_features),
            "invalid conditional action grammar",
        )
    }

    pub fn accepts(&self, action: &ConditionalAction) -> bool {
        let argument = match action.argument {
            None => ArgumentKind::None,
            Some(ActionArgument::Entity(_)) => ArgumentKind::Entity,
            Some(ActionArgument::Location { .. } | ActionArgument::RegionalLocation { .. }) => {
                ArgumentKind::Location
            }
        };
        self.operations
            .get(action.operation)
            .is_some_and(|operation| operation.argument == argument)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Discrete(usize),
    Continuous(Vec<f32>),
    Conditional(Arc<ConditionalAction>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActionSpace {
    Discrete { count: usize },
    Box { low: Vec<f32>, high: Vec<f32> },
    Unbounded { dimensions: usize },
    Conditional(Arc<ActionGrammar>),
}
impl ActionSpace {
    pub fn validate(&self) -> Result<()> {
        let valid = match self {
            Self::Discrete { count } => *count > 0,
            Self::Unbounded { dimensions } => *dimensions > 0,
            Self::Box { low, high } => {
                !low.is_empty()
                    && low.len() == high.len()
                    && low
                        .iter()
                        .zip(high)
                        .all(|(&l, &h)| l.is_finite() && h.is_finite() && l < h)
            }
            Self::Conditional(grammar) => return grammar.validate(),
        };
        crate::core::ensure(valid, "invalid action space")
    }
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Discrete { .. } => 1,
            Self::Box { low, .. } => low.len(),
            Self::Unbounded { dimensions } => *dimensions,
            Self::Conditional(_) => 0,
        }
    }
    pub fn check(&self, action: &Action, mask: Option<&[bool]>) -> Result<()> {
        let valid = match (self, action) {
            (Self::Discrete { count }, Action::Discrete(a)) => {
                *a < *count && mask.is_none_or(|m| m.len() == *count && m[*a])
            }
            (Self::Box { low, high }, Action::Continuous(a)) => {
                a.len() == low.len()
                    && a.iter()
                        .zip(low.iter().zip(high))
                        .all(|(&x, (&l, &h))| x.is_finite() && x >= l && x <= h)
            }
            (Self::Unbounded { dimensions }, Action::Continuous(a)) => {
                a.len() == *dimensions && a.iter().all(|x| x.is_finite())
            }
            (Self::Conditional(grammar), Action::Conditional(a)) => {
                mask.is_none() && grammar.accepts(a)
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(Error::Action("action violates space or mask".into()))
        }
    }
}
