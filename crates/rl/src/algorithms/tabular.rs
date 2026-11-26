//! Sparse action values for finite, state-dependent action menus.
//!
//! Keys have caller-defined semantics. For partially observed environments,
//! construct keys from a sufficient information state or explicitly choose an
//! approximation. The table keeps only visited pairs. It works independently
//! of an environment, representation, replay buffer, and policy RNG.

mod actor;
pub use actor::TabularActor;

use crate::{
    Error, Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        reward::{PotentialValues, Reward},
    },
};
use rand::Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;

/// Sparse action values with a common initial value for unseen pairs.
#[derive(Clone, Debug)]
pub struct QTable<S, A> {
    initial_value: f64,
    states: BTreeMap<S, BTreeMap<A, Entry>>,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    value: f64,
    visits: u64,
}

/// One-step temporal-difference update parameters.
#[derive(Clone, Copy, Debug)]
pub struct TdConfig {
    /// Discount per primitive environment step, in `[0, 1]`.
    pub gamma: f64,
    /// Constant learning rate, in `(0, 1]`.
    pub step_size: f64,
}

/// A sampled decision and its probability under the executed policy.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    /// Position of the chosen action in the supplied legal menu.
    pub index: usize,
    pub probability: f64,
}

/// One factual decision in a complete episode or time-limited rollout.
#[derive(Clone, Debug)]
pub struct EpisodeStep<S, A> {
    pub state: S,
    pub action: A,
    pub reward: Reward,
}

/// Why a Monte Carlo trajectory ended, and its continuation when truncated.
#[derive(Clone, Copy, Debug)]
pub enum EpisodeEnd {
    Terminated,
    /// Value at the final observed successor. Set to zero to deliberately
    /// estimate the truncated return. Continue assembling a trajectory across
    /// a temporary collection boundary.
    Truncated {
        bootstrap: f64,
    },
}

impl<S: Ord, A: Ord> QTable<S, A> {
    pub fn new(initial_value: f64) -> Result<Self> {
        ensure(initial_value.is_finite(), "nonfinite initial tabular value")?;
        Ok(Self {
            initial_value,
            states: BTreeMap::new(),
        })
    }

    /// Value of a pair under the current table, including unseen initialization.
    pub fn value(&self, state: &S, action: &A) -> f64 {
        self.entry(state, action).value
    }

    /// Number of updates to a pair, excluding pure lookups and action selection.
    pub fn visits(&self, state: &S, action: &A) -> u64 {
        self.entry(state, action).visits
    }

    /// Maximize Q over the legal menu. Tied actions use the smallest key.
    pub fn greedy_index(&self, state: &S, legal: &[A]) -> Result<usize> {
        action_order(legal)?;
        Ok(self.best_index(state, legal))
    }

    /// Mix the greedy action with uniform exploration over distinct legal keys.
    /// The same seed selects the same key under any permutation of the menu.
    /// A singleton menu or zero per-action exploratory mass consumes no RNG.
    pub fn epsilon_greedy(
        &self,
        state: &S,
        legal: &[A],
        epsilon: f64,
        rng: &mut (impl Rng + ?Sized),
    ) -> Result<Choice> {
        ensure((0.0..=1.0).contains(&epsilon), "invalid tabular epsilon")?;
        let order = action_order(legal)?;
        let greedy = self.best_index(state, legal);
        let uniform = epsilon / legal.len() as f64;
        if legal.len() == 1 || uniform == 0.0 {
            return Ok(Choice {
                index: greedy,
                probability: 1.0,
            });
        }
        let index = if epsilon == 1.0 || rng.random_bool(epsilon) {
            let rank = rng.random_range(0..legal.len());
            order.as_ref().map_or(rank, |order| order[rank])
        } else {
            greedy
        };
        let probability = uniform + if index == greedy { 1.0 - epsilon } else { 0.0 };
        Ok(Choice { index, probability })
    }

    fn entry(&self, state: &S, action: &A) -> Entry {
        self.states
            .get(state)
            .and_then(|row| row.get(action))
            .copied()
            .unwrap_or(Entry {
                value: self.initial_value,
                visits: 0,
            })
    }

    fn best_index(&self, state: &S, legal: &[A]) -> usize {
        let mut best = 0;
        let mut best_value = self.value(state, &legal[best]);
        for (index, action) in legal.iter().enumerate().skip(1) {
            let value = self.value(state, action);
            if value > best_value || (value == best_value && action < &legal[best]) {
                best = index;
                best_value = value;
            }
        }
        best
    }
}

impl<S: Clone + Ord, A: Clone + Ord> QTable<S, A> {
    /// Update toward the maximum Q over the successor's legal menu.
    /// Pass `None` at a true terminal and `Some` for a bootstrapped truncation.
    pub fn q_learning(
        &mut self,
        state: &S,
        action: &A,
        reward: &Reward,
        next: Option<(&S, &[A])>,
        config: TdConfig,
    ) -> Result<()> {
        self.q_learning_update(state, action, reward, next, None, config)
    }

    /// Apply a potential to the learning reward while retaining the factual
    /// trace. The next potential must be present exactly when `next` supplies
    /// the Bellman continuation. Pass neither at a true terminal or a modeled
    /// zero-value horizon boundary.
    pub fn q_learning_with_potential(
        &mut self,
        state: &S,
        action: &A,
        reward: &Reward,
        next: Option<(&S, &[A])>,
        potentials: PotentialValues,
        config: TdConfig,
    ) -> Result<()> {
        ensure(
            next.is_some() == potentials.next.is_some(),
            "successor potential must match tabular continuation",
        )?;
        self.q_learning_update(state, action, reward, next, Some(potentials), config)
    }

    fn q_learning_update(
        &mut self,
        state: &S,
        action: &A,
        reward: &Reward,
        next: Option<(&S, &[A])>,
        potentials: Option<PotentialValues>,
        config: TdConfig,
    ) -> Result<()> {
        let continuation = match next {
            Some((state, legal)) => self.value(state, &legal[self.greedy_index(state, legal)?]),
            None => 0.0,
        };
        self.update_td(state, action, reward, continuation, potentials, config)
    }

    /// Update toward the Q of the successor action selected by the policy.
    /// Pass `None` at a true terminal and `Some` for a bootstrapped truncation.
    pub fn sarsa(
        &mut self,
        state: &S,
        action: &A,
        reward: &Reward,
        next: Option<(&S, &A)>,
        config: TdConfig,
    ) -> Result<()> {
        let continuation = next.map_or(0.0, |(state, action)| self.value(state, action));
        self.update_td(state, action, reward, continuation, None, config)
    }

    /// Average the return from each state-action pair's first visit in this
    /// trajectory. Start with a fresh table for sample means over MC episodes.
    /// To run on-policy control, collect each episode with `epsilon_greedy`
    /// using the current table and then apply this update.
    pub fn monte_carlo_first_visit(
        &mut self,
        episode: &[EpisodeStep<S, A>],
        end: EpisodeEnd,
        gamma: f64,
    ) -> Result<()> {
        ensure(!episode.is_empty(), "empty tabular episode")?;
        let mut value = match end {
            EpisodeEnd::Terminated => 0.0,
            EpisodeEnd::Truncated { bootstrap } => bootstrap,
        };
        let mut first = BTreeMap::new();
        for step in episode.iter().rev() {
            value = return_with_bootstrap(&step.reward, gamma, value)?;
            first.insert((&step.state, &step.action), value);
        }
        let mut updates = Vec::with_capacity(first.len());
        for ((state, action), target) in first {
            let entry = self.entry(state, action).updated(target, None)?;
            updates.push((state, action, entry));
        }
        for (state, action, entry) in updates {
            self.install(state, action, entry);
        }
        Ok(())
    }

    fn update_td(
        &mut self,
        state: &S,
        action: &A,
        reward: &Reward,
        continuation: f64,
        potentials: Option<PotentialValues>,
        config: TdConfig,
    ) -> Result<()> {
        ensure(
            config.step_size > 0.0 && config.step_size <= 1.0,
            "invalid tabular step size",
        )?;
        let (reward, discount) = reward.discounted_f64(config.gamma)?;
        let reward = match potentials {
            Some(potentials) => potentials.shape(reward, discount)?,
            None => reward,
        };
        let target = aggregate_with_bootstrap(reward, discount, continuation)?;
        let entry = self
            .entry(state, action)
            .updated(target, Some(config.step_size))?;
        self.install(state, action, entry);
        Ok(())
    }

    fn install(&mut self, state: &S, action: &A, entry: Entry) {
        if let Some(row) = self.states.get_mut(state) {
            if let Some(existing) = row.get_mut(action) {
                *existing = entry;
            } else {
                row.insert(action.clone(), entry);
            }
        } else {
            self.states
                .insert(state.clone(), BTreeMap::from([(action.clone(), entry)]));
        }
    }
}

/// `None` means the menu is already in stable key order.
fn action_order<A: Ord>(legal: &[A]) -> Result<Option<Vec<usize>>> {
    ensure(!legal.is_empty(), "empty tabular action menu")?;
    if legal.windows(2).all(|pair| pair[0] < pair[1]) {
        return Ok(None);
    }
    let mut order: Vec<_> = (0..legal.len()).collect();
    order.sort_unstable_by(|&left, &right| legal[left].cmp(&legal[right]));
    ensure(
        order
            .windows(2)
            .all(|pair| legal[pair[0]] != legal[pair[1]]),
        "duplicate tabular action key",
    )?;
    Ok(Some(order))
}

fn return_with_bootstrap(reward: &Reward, gamma: f64, continuation: f64) -> Result<f64> {
    let (reward, discount) = reward.discounted_f64(gamma)?;
    aggregate_with_bootstrap(reward, discount, continuation)
}

fn aggregate_with_bootstrap(reward: f64, discount: f64, continuation: f64) -> Result<f64> {
    let target = reward + discount * continuation;
    if target.is_finite() {
        Ok(target)
    } else {
        Err(Error::Numerical("nonfinite tabular return".into()))
    }
}

impl Entry {
    fn updated(self, target: f64, step_size: Option<f64>) -> Result<Self> {
        let visits = self
            .visits
            .checked_add(1)
            .ok_or_else(|| Error::State("tabular visit count overflow".into()))?;
        let step_size = step_size.unwrap_or_else(|| 1.0 / visits as f64);
        let value = if step_size == 1.0 {
            target
        } else {
            (1.0 - step_size) * self.value + step_size * target
        };
        if value.is_finite() {
            Ok(Self { value, visits })
        } else {
            Err(Error::Numerical("nonfinite tabular action value".into()))
        }
    }
}

const ALGORITHM: &str = "tabular-q-table";
const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TableRecord<S, A> {
    version: u32,
    initial_value: f64,
    rows: Vec<TableRow<S, A>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TableRow<S, A> {
    state: S,
    action: A,
    value: f64,
    visits: u64,
}

impl<S, A> Checkpoint for QTable<S, A>
where
    S: Ord + Serialize + DeserializeOwned,
    A: Ord + Serialize + DeserializeOwned,
{
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let rows = self.states.iter().flat_map(|(state, row)| {
            row.iter().map(move |(action, entry)| TableRow {
                state,
                action,
                value: entry.value,
                visits: entry.visits,
            })
        });
        LearnerRecord::new(
            ALGORITHM,
            TableRecord {
                version: VERSION,
                initial_value: self.initial_value,
                rows: rows.collect(),
            },
        )
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check(ALGORITHM)?;
        ensure(
            record.models.is_empty()
                && record.optimizers.is_empty()
                && record.temporal_returns.is_none()
                && record.temporal_spaces.is_none()
                && record.temporal_reconstruction.is_none()
                && record.replay_requirements.is_none(),
            "unexpected model or runtime data in tabular checkpoint",
        )?;
        let wire = TableRecord::<S, A>::deserialize(&record.metadata)?;
        ensure(
            wire.version == VERSION,
            "unsupported tabular checkpoint version",
        )?;
        let mut table = Self::new(wire.initial_value)?;
        for row in wire.rows {
            ensure(
                row.value.is_finite() && row.visits > 0,
                "invalid tabular checkpoint row",
            )?;
            let entry = Entry {
                value: row.value,
                visits: row.visits,
            };
            let displaced = table
                .states
                .entry(row.state)
                .or_default()
                .insert(row.action, entry);
            ensure(displaced.is_none(), "duplicate tabular checkpoint key")?;
        }
        *self = table;
        Ok(())
    }
}
