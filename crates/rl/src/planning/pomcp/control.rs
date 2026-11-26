//! Internal optional hooks shared by legacy and guided entry points.
use super::{RolloutDecision, RolloutStep};
use crate::Result;
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum TreeSelection {
    #[default]
    Uct,
    /// After one visit per action, use Q + c P sqrt(N) / (1 + n).
    /// Zero-prior actions receive the initial visit but no exploration bonus.
    Puct { exploration: f64 },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Backup {
    /// Estimate each action's mean return under search's explored continuation.
    #[default]
    MonteCarloMean,
    /// Average chance outcomes using their empirical frequencies, with a
    /// greedy continuation over visited legal actions at expanded decisions.
    /// Frontier outcomes use the rollout or value guide until a child action
    /// is visited. Maximizing sample-based action estimates can be optimistic.
    Bellman,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RootSelection {
    /// Apply the configured tree policy throughout search.
    #[default]
    TreePolicy,
    /// Allocate root simulations evenly within rounds and eliminate the lower
    /// half between rounds. Final deployment is among the surviving candidates.
    SequentialHalving,
}

/// Independent choices for backing up values and allocating work at the root.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SearchStrategy {
    pub backup: Backup,
    pub root_selection: RootSelection,
}

type Prior<'a, C, A> = &'a mut dyn FnMut(&C, &[A]) -> Result<Vec<f64>>;
type Rollout<'a, C, A> = &'a mut dyn FnMut(&C, RolloutStep) -> Result<RolloutDecision<A>>;
type Leaf<'a, C> = &'a mut dyn FnMut(&C, u16) -> Result<f64>;

pub(crate) struct Controls<'a, C, A> {
    pub selection: TreeSelection,
    pub strategy: SearchStrategy,
    pub prior: Option<Prior<'a, C, A>>,
    pub rollout: Option<Rollout<'a, C, A>>,
    pub cutoff: Option<(u16, Leaf<'a, C>)>,
    pub max_expansions: Option<NonZeroU32>,
    pub tree_layers: Option<NonZeroU16>,
    pub require_terminal: bool,
    pub rollout_prior: Option<Prior<'a, C, A>>,
}
impl<C, A> Default for Controls<'_, C, A> {
    fn default() -> Self {
        Self {
            selection: TreeSelection::Uct,
            strategy: SearchStrategy::default(),
            prior: None,
            rollout: None,
            cutoff: None,
            max_expansions: None,
            tree_layers: None,
            require_terminal: false,
            rollout_prior: None,
        }
    }
}
