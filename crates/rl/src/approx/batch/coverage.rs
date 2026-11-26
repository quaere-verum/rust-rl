use super::FactualBatch;
use crate::{
    Error, Result,
    core::{
        action::{Action, ConditionalAction},
        reward::BatchObjective,
    },
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActionCoverage {
    pub action: Action,
    pub legal_rows: usize,
    pub selected_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchCoverage {
    pub objective: BatchObjective,
    pub episodes: usize,
    pub rows: usize,
    pub feature_dimension: usize,
    pub mc_eligible_rows: usize,
    /// Kish effective sample size of the factual row weights. Rows can share
    /// episodes or layouts, so this is a weight-concentration descriptor.
    pub effective_sample_size: Option<f64>,
    pub behavior_revisions: Vec<u64>,
    pub actions: Vec<ActionCoverage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ActionKey {
    Discrete(usize),
    Conditional(ConditionalAction),
}

impl TryFrom<&Action> for ActionKey {
    type Error = Error;

    fn try_from(action: &Action) -> Result<Self> {
        match action {
            Action::Discrete(index) => Ok(Self::Discrete(*index)),
            Action::Conditional(key) => Ok(Self::Conditional(**key)),
            Action::Continuous(_) => Err(Error::Action("expected finite factual action".into())),
        }
    }
}

impl From<ActionKey> for Action {
    fn from(key: ActionKey) -> Self {
        match key {
            ActionKey::Discrete(index) => Self::Discrete(index),
            ActionKey::Conditional(key) => Self::Conditional(key.into()),
        }
    }
}

impl FactualBatch<'_, '_> {
    /// Measure the selected fitting view without encoding features or changing
    /// the factual tape. Counts cover legal and taken complete action keys at
    /// each selected row, with one contribution per row regardless of weight.
    pub fn coverage(&self) -> Result<BatchCoverage> {
        let mut actions = BTreeMap::<ActionKey, (usize, usize)>::new();
        let mut behavior_revisions = BTreeSet::new();
        let mut mc_eligible_rows = 0;
        let mut total = 0.0;
        let mut sum_squares = 0.0;
        let scale = self.rows.iter().map(|row| row.weight).fold(0.0, f64::max);
        for index in 0..self.rows.len() {
            let (row, episode) = self.source_row(index)?;
            let history = &episode.histories[row.timestep];
            for action in history.observation().finite_actions(&self.spaces.actions)? {
                actions.entry(ActionKey::try_from(&action)?).or_default().0 += 1;
            }
            actions
                .entry(ActionKey::try_from(&row.step.action)?)
                .or_default()
                .1 += 1;
            behavior_revisions.insert(row.step.behavior.revision);
            mc_eligible_rows += usize::from(episode.complete_returns.is_some());
            let weight = row.weight / scale;
            total += weight;
            sum_squares += weight * weight;
        }
        Ok(BatchCoverage {
            objective: self.objective(),
            episodes: self.episodes.len(),
            rows: self.rows.len(),
            feature_dimension: self.feature_spec().dimension(),
            mc_eligible_rows,
            effective_sample_size: (!self.rows.is_empty()).then(|| total * total / sum_squares),
            behavior_revisions: behavior_revisions.into_iter().collect(),
            actions: actions
                .into_iter()
                .map(|(key, (legal_rows, selected_rows))| ActionCoverage {
                    action: key.into(),
                    legal_rows,
                    selected_rows,
                })
                .collect(),
        })
    }
}
