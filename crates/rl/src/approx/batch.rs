//! A read-only fitting view of the existing factual episode tape.
//! Rows retain their episode/decision identity and reference the actual action
//! and behavior metadata. Return projections never modify factual rewards.

use crate::{
    Error, Result,
    approx::{FeatureMap, FeatureSpec, checked_features},
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::PublicHistory,
        temporal::TemporalSpaces,
    },
    data::{
        returns::accumulate_rewards,
        temporal::{Episode, EpisodeId, EpisodeStore, TemporalStep},
    },
};
use serde::{Deserialize, Serialize};
use std::{cell::OnceCell, num::NonZeroUsize, ops::Range};

pub use crate::core::reward::BatchObjective;

mod coverage;
pub use coverage::{ActionCoverage, BatchCoverage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DecisionSpan {
    /// One stored semantic action per step of a target.
    #[default]
    Single,
    /// Collapse intervening unselected decisions up to the next selected
    /// decision or recorded boundary, preserving their rewards and duration.
    NextSelected,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchOptions {
    pub objective: BatchObjective,
    pub span: DecisionSpan,
}

pub struct DecisionPoint<'a> {
    pub episode_id: EpisodeId,
    pub timestep: usize,
    pub history: &'a PublicHistory,
    pub legal: &'a [Action],
    pub step: &'a TemporalStep,
}

pub struct FactualRow<'a> {
    pub episode_id: EpisodeId,
    /// Absolute semantic transition index in the source tape.
    pub timestep: usize,
    pub step: &'a TemporalStep,
    pub weight: f64,
    pub features: Vec<f64>,
    episode_slot: usize,
}

struct BatchEpisode<'a> {
    source: &'a Episode,
    histories: Vec<PublicHistory>,
    rows: Range<usize>,
    successors: Vec<OnceCell<EncodedDecision>>,
    next: Vec<usize>,
    complete_returns: Option<Vec<f64>>,
}

struct EncodedDecision {
    actions: Vec<Action>,
    features: Vec<Vec<f64>>,
    executed_index: Option<usize>,
}

impl EncodedDecision {
    fn new(
        map: &dyn FeatureMap,
        history: &PublicHistory,
        actions: Vec<Action>,
        executed: Option<&FactualRow<'_>>,
    ) -> Result<Self> {
        let (features, executed_index) = if let Some(row) = executed {
            let index = actions
                .iter()
                .position(|action| action == &row.step.action)
                .ok_or_else(|| {
                    Error::State("factual action absent from successor support".into())
                })?;
            let missing: Vec<_> = actions[..index]
                .iter()
                .chain(&actions[index + 1..])
                .cloned()
                .collect();
            let features = if missing.is_empty() {
                Vec::new()
            } else {
                checked_features(map, history, &missing)?
            };
            (features, Some(index))
        } else {
            (checked_features(map, history, &actions)?, None)
        };
        Ok(Self {
            actions,
            features,
            executed_index,
        })
    }

    fn rows<'a>(&'a self, executed: Option<&'a FactualRow<'_>>) -> Vec<&'a [f64]> {
        let mut missing = self.features.iter().map(Vec::as_slice);
        (0..self.actions.len())
            .map(|index| {
                if Some(index) == self.executed_index {
                    executed
                        .expect("cached executed action has a factual row")
                        .features
                        .as_slice()
                } else {
                    missing
                        .next()
                        .expect("cached successor feature for each remaining action")
                }
            })
            .collect()
    }
}

/// Borrowed batch over EpisodeStore. Keeps exact histories shared among rows
/// and precomputes the executed design once. Live successor encodings are
/// retained on demand within this batch and its bound feature revision.
/// Prefixes lost to bounded/anchored retention are rejected explicitly.
pub struct FactualBatch<'data, 'map> {
    features: &'map dyn FeatureMap,
    feature_spec: &'map FeatureSpec,
    spaces: TemporalSpaces,
    options: BatchOptions,
    episodes: Vec<BatchEpisode<'data>>,
    rows: Vec<FactualRow<'data>>,
}

pub struct BatchTarget<'a> {
    pub reward: f64,
    pub discount: f64,
    pub duration: u64,
    pub successor: &'a PublicHistory,
    pub terminated: bool,
    pub truncated: bool,
    pub factual_actions: usize,
}

impl<'data, 'map> FactualBatch<'data, 'map> {
    pub fn from_store(
        store: &'data EpisodeStore,
        features: &'map dyn FeatureMap,
        options: BatchOptions,
    ) -> Result<Self> {
        Self::from_store_with(store, features, options, |_| Ok(Some(1.)))
    }

    /// Select factual row weights. Returning None excludes a decision from
    /// the fit. With NextSelected it also folds that decision into the prior
    /// selected span. Prefix-only checkpoint context is never a training row.
    pub fn from_store_with(
        store: &'data EpisodeStore,
        features: &'map dyn FeatureMap,
        options: BatchOptions,
        select: impl Fn(DecisionPoint<'_>) -> Result<Option<f64>>,
    ) -> Result<Self> {
        let feature_spec = features.spec();
        feature_spec.validate()?;
        options.objective.validate()?;
        let spaces = TemporalSpaces {
            observations: store.observation_space().clone(),
            actions: store.action_space().clone(),
        };
        features.validate_spaces(&spaces)?;
        ensure(
            matches!(
                spaces.actions,
                ActionSpace::Discrete { .. } | ActionSpace::Conditional(_)
            ),
            "batch action values require finite complete actions",
        )?;
        let mut batch = Self {
            features,
            feature_spec,
            spaces,
            options,
            episodes: Vec::new(),
            rows: Vec::new(),
        };
        for episode in store.episodes() {
            batch.add_episode(episode, &select)?;
        }
        batch.validate_feature_revision()?;
        Ok(batch)
    }

    fn add_episode(
        &mut self,
        episode: &'data Episode,
        select: &impl Fn(DecisionPoint<'_>) -> Result<Option<f64>>,
    ) -> Result<()> {
        let count = episode.steps.len();
        if episode.learning_start >= episode.end_timestep() {
            return Ok(());
        }
        ensure(
            episode.base_timestep == 0,
            "statistical batch requires a retained genuine reset prefix",
        )?;
        let mut history = PublicHistory::new(episode.observations[0].clone());
        let mut histories = Vec::with_capacity(count + 1);
        let first_row = self.rows.len();
        let episode_slot = self.episodes.len();
        for (timestep, step) in episode.steps.iter().enumerate() {
            if timestep >= episode.learning_start {
                let legal = history.observation().finite_actions(&self.spaces.actions)?;
                if let Some(weight) = select(DecisionPoint {
                    episode_id: episode.id,
                    timestep,
                    history: &history,
                    legal: &legal,
                    step,
                })? {
                    ensure(
                        weight.is_finite() && weight > 0.,
                        "invalid factual row weight",
                    )?;
                    let mut encoded = checked_features(
                        self.features,
                        &history,
                        std::slice::from_ref(&step.action),
                    )?;
                    self.rows.push(FactualRow {
                        episode_id: episode.id,
                        timestep,
                        step,
                        weight,
                        features: encoded.pop().expect("exactly one checked executed row"),
                        episode_slot,
                    });
                }
            }
            let next = history.append(
                step.action.clone(),
                step.reward.clone(),
                episode.observations[timestep + 1].clone(),
            )?;
            histories.push(history);
            history = next;
        }
        if self.rows.len() == first_row {
            return Ok(());
        }
        histories.push(history);
        let next = match self.options.span {
            DecisionSpan::Single => (1..=count).collect(),
            DecisionSpan::NextSelected => {
                let mut after = count;
                let mut next = vec![count; count];
                let mut selected = self.rows[first_row..].iter().rev().peekable();
                for (timestep, boundary) in next.iter_mut().enumerate().rev() {
                    *boundary = after;
                    if selected.peek().is_some_and(|row| row.timestep == timestep) {
                        selected.next();
                        after = timestep;
                    }
                }
                next
            }
        };
        let complete_returns = factual_returns(episode, self.options.objective)?;
        self.episodes.push(BatchEpisode {
            source: episode,
            successors: std::iter::repeat_with(OnceCell::new)
                .take(histories.len())
                .collect(),
            histories,
            rows: first_row..self.rows.len(),
            next,
            complete_returns,
        });
        Ok(())
    }

    pub fn rows(&self) -> &[FactualRow<'data>] {
        &self.rows
    }

    pub fn objective(&self) -> BatchObjective {
        self.options.objective
    }

    pub fn spaces(&self) -> &TemporalSpaces {
        &self.spaces
    }

    pub fn features(&self) -> &dyn FeatureMap {
        self.features
    }

    /// Feature identity bound when the executed rows were encoded.
    pub fn feature_spec(&self) -> &FeatureSpec {
        self.feature_spec
    }

    pub(crate) fn validate_feature_revision(&self) -> Result<()> {
        let live = self.features.spec();
        ensure(
            std::ptr::eq(live, self.feature_spec) || live == self.feature_spec,
            "feature representation changed after binding factual rows",
        )
    }

    pub(crate) fn successor_features(
        &self,
        source: usize,
        history: &PublicHistory,
    ) -> Result<(&[Action], Vec<&[f64]>)> {
        let (_, episode) = self.source_row(source)?;
        let timestep = usize::try_from(history.timestep()).map_err(|_| {
            Error::State("factual successor timestep exceeds the index range".into())
        })?;
        ensure(
            episode
                .histories
                .get(timestep)
                .is_some_and(|frame| std::ptr::eq(frame, history)),
            "successor feature request is outside the source episode",
        )?;
        let factual = &self.rows[episode.rows.clone()];
        let executed = factual
            .binary_search_by_key(&timestep, |row| row.timestep)
            .ok()
            .map(|index| &factual[index]);
        let slot = &episode.successors[timestep];
        let encoded = if let Some(encoded) = slot.get() {
            encoded
        } else {
            let actions = history.observation().finite_actions(&self.spaces.actions)?;
            let encoded = EncodedDecision::new(self.features, history, actions, executed)?;
            self.validate_feature_revision()?;
            slot.get_or_init(|| encoded)
        };
        Ok((&encoded.actions, encoded.rows(executed)))
    }

    /// Complete behavior-policy return from this factual action. Unfinished
    /// and externally truncated episodes have no complete Monte Carlo target.
    pub fn monte_carlo(&self, row: usize) -> Result<Option<f64>> {
        let (row, episode) = self.source_row(row)?;
        Ok(episode
            .complete_returns
            .as_ref()
            .map(|returns| returns[row.timestep]))
    }

    /// Primitive discounting through the recorded action span.
    /// Ended and incomplete tape boundaries shorten the requested horizon.
    /// At a live boundary the successor is the actual recorded observation.
    pub fn n_step(&self, row: usize, horizon: NonZeroUsize) -> Result<BatchTarget<'_>> {
        let (row, episode) = self.source_row(row)?;
        let start = row.timestep;
        let mut end = start;
        for _ in 0..horizon.get() {
            let Some(&next) = episode.next.get(end) else {
                break;
            };
            end = next;
        }
        let last = &episode.source.steps[end - 1];
        let successor = &episode.histories[end];
        let (reward, discount, duration) = match self.options.objective {
            BatchObjective::Physical { gamma } => accumulate_rewards(
                episode.source.steps[start..end]
                    .iter()
                    .map(|step| &step.reward),
                gamma,
            )?,
            BatchObjective::TerminalSuccess { .. } => (
                if last.terminated {
                    episode
                        .complete_returns
                        .as_ref()
                        .expect("true terminal has factual returns")[end - 1]
                } else {
                    0.
                },
                1.,
                successor.physical_timestep() - episode.histories[start].physical_timestep(),
            ),
        };
        Ok(BatchTarget {
            reward,
            discount: if last.terminated { 0. } else { discount },
            duration,
            successor,
            terminated: last.terminated,
            truncated: last.truncated,
            factual_actions: end - start,
        })
    }

    fn source_row(&self, index: usize) -> Result<(&FactualRow<'data>, &BatchEpisode<'data>)> {
        let row = self
            .rows
            .get(index)
            .ok_or_else(|| Error::Shape("factual row index out of range".into()))?;
        Ok((row, &self.episodes[row.episode_slot]))
    }
}

fn factual_returns(episode: &Episode, objective: BatchObjective) -> Result<Option<Vec<f64>>> {
    let terminal = episode.steps.last().is_some_and(|step| step.terminated);
    if let BatchObjective::TerminalSuccess { success, failure } = objective {
        let mut outcome = 0.;
        for (index, step) in episode.steps.iter().enumerate() {
            let primitives = step.reward.primitives();
            let background = if step.terminated {
                let (last, prefix) = primitives.split_last().expect("validated factual trace");
                ensure(
                    index + 1 == episode.steps.len() && (*last == success || *last == failure),
                    "terminal-success reward label does not match the declared objective",
                )?;
                outcome = if *last == success { 1. } else { 0. };
                prefix
            } else {
                primitives
            };
            ensure(
                background.iter().all(|reward| *reward == 0.),
                "terminal-success objective requires zero nonterminal primitive rewards",
            )?;
        }
        return Ok(terminal.then(|| vec![outcome; episode.steps.len()]));
    }
    if !terminal {
        return Ok(None);
    }
    let mut value = 0.;
    let mut returns = vec![0.; episode.steps.len()];
    for (step, output) in episode.steps.iter().zip(&mut returns).rev() {
        let (reward, discount) = step.reward.discounted_f64(objective.gamma())?;
        value = reward
            + if step.terminated {
                0.
            } else {
                discount * value
            };
        ensure(value.is_finite(), "nonfinite complete factual return")?;
        *output = value;
    }
    Ok(Some(returns))
}
