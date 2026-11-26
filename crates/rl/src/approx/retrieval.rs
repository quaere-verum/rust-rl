//! Return-weighted nearest-neighbor inference on public features.

use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Exemplar {
    state: Vec<f64>,
    action: Vec<f64>,
    weight: f64,
}

/// Bounded online retrieval of executed actions from positively weighted episodes.
///
/// The caller encodes the public predecision information and semantic actions as
/// two independently scaled vectors with fixed widths. Distances are Euclidean
/// in those coordinates. Use the same transformations for collection and
/// inference. The caller also owns the feature schema and should preserve it
/// alongside checkpoints.
///
/// Capacity is measured in decisions. The bank evicts the oldest decision first
/// and uses insertion order to break equal state distances. A frozen reservoir of
/// successful decisions uses a different retention rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RetrievalRecord")]
pub struct ReturnWeightedRetrieval {
    capacity: usize,
    neighbors: usize,
    examples: VecDeque<Exemplar>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetrievalRecord {
    capacity: usize,
    neighbors: usize,
    examples: VecDeque<Exemplar>,
}

impl TryFrom<RetrievalRecord> for ReturnWeightedRetrieval {
    type Error = crate::Error;

    fn try_from(record: RetrievalRecord) -> Result<Self> {
        let mut retrieval = Self::new(record.capacity, record.neighbors)?;
        ensure(
            record.examples.len() <= record.capacity,
            "retrieval checkpoint exceeds its capacity",
        )?;
        if let Some(first) = record.examples.front() {
            ensure(
                record.examples.iter().all(|example| {
                    valid_features(&example.state, first.state.len())
                        && valid_features(&example.action, first.action.len())
                        && example.weight.is_finite()
                        && example.weight > 0.
                }),
                "invalid retrieval checkpoint example",
            )?;
        }
        retrieval.examples = record.examples;
        Ok(retrieval)
    }
}

impl ReturnWeightedRetrieval {
    /// Construct an empty bank with a positive neighbor count up to its capacity.
    pub fn new(capacity: usize, neighbors: usize) -> Result<Self> {
        ensure(
            neighbors > 0 && neighbors <= capacity,
            "invalid retrieval capacity or neighbor count",
        )?;
        Ok(Self {
            capacity,
            neighbors,
            examples: VecDeque::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn neighbors(&self) -> usize {
        self.neighbors
    }

    /// Widths of the stored state and action coordinates, when populated.
    pub fn feature_dimensions(&self) -> Option<(usize, usize)> {
        self.examples
            .front()
            .map(|first| (first.state.len(), first.action.len()))
    }

    /// Append one completed factual episode in execution order.
    ///
    /// Each pair contains the public state features before a decision and the
    /// semantic feature row of the action actually executed. The nonnegative
    /// episode weight is divided equally among decisions. Zero weight inserts
    /// nothing. An invalid positively weighted episode leaves the bank intact.
    pub fn observe(
        &mut self,
        decisions: &[(Vec<f64>, Vec<f64>)],
        episode_weight: f64,
    ) -> Result<()> {
        ensure(
            episode_weight.is_finite() && episode_weight >= 0.,
            "invalid retrieval episode weight",
        )?;
        if episode_weight == 0. || decisions.is_empty() {
            return Ok(());
        }
        let (state_width, action_width) = self.examples.front().map_or_else(
            || (decisions[0].0.len(), decisions[0].1.len()),
            |first| (first.state.len(), first.action.len()),
        );
        let weight = episode_weight / decisions.len() as f64;
        ensure(
            weight > 0.
                && decisions.iter().all(|(state, action)| {
                    valid_features(state, state_width) && valid_features(action, action_width)
                }),
            "invalid retrieval episode features or underflowed example weight",
        )?;
        for (state, action) in decisions {
            if self.len() == self.capacity {
                self.examples.pop_front();
            }
            self.examples.push_back(Exemplar {
                state: state.clone(),
                action: action.clone(),
                weight,
            });
        }
        Ok(())
    }

    /// Score the supplied legal candidates and return probabilities in their order.
    ///
    /// For each of the nearest public states, an exemplar contributes its episode
    /// weight per decision divided by one plus the squared state distance. This
    /// contribution is multiplied by the exponential of the negated squared
    /// semantic action distance. The accumulated scores are normalized across the
    /// menu. An empty bank yields a uniform law.
    ///
    /// To score a framework environment, use [`crate::core::env::Observation::finite_actions`] and
    /// encode those actions in the resulting order.
    pub fn probabilities(&self, state: &[f64], candidates: &[Vec<f64>]) -> Result<Vec<f64>> {
        let action_width = self.examples.front().map_or_else(
            || candidates.first().map_or(0, Vec::len),
            |first| first.action.len(),
        );
        ensure(
            !candidates.is_empty()
                && candidates
                    .iter()
                    .all(|row| valid_features(row, action_width)),
            "invalid complete legal menu features",
        )?;
        let Some(first) = self.examples.front() else {
            return Ok(vec![1. / candidates.len() as f64; candidates.len()]);
        };
        ensure(
            valid_features(state, first.state.len()),
            "invalid retrieval state features",
        )?;

        let mut nearest = self
            .examples
            .iter()
            .enumerate()
            .map(|(index, example)| Ok((squared_distance(state, &example.state)?, index)))
            .collect::<Result<Vec<_>>>()?;
        let ordered =
            |a: &(f64, usize), b: &(f64, usize)| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1));
        if nearest.len() > self.neighbors {
            nearest.select_nth_unstable_by(self.neighbors, ordered);
            nearest.truncate(self.neighbors);
        }
        nearest.sort_unstable_by(ordered);
        let weighted: Vec<_> = nearest
            .into_iter()
            .map(|(distance, index)| {
                let example = &self.examples[index];
                (example.weight.ln() - distance.ln_1p(), &example.action)
            })
            .collect();

        // Sum the positive kernels in log space before normalizing the menu.
        let mut logits = Vec::with_capacity(candidates.len());
        let mut terms = Vec::with_capacity(weighted.len());
        for candidate in candidates {
            terms.clear();
            for (log_weight, action) in &weighted {
                terms.push(log_weight - squared_distance(candidate, action)?);
            }
            let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = terms.iter().map(|term| (term - max).exp()).sum::<f64>();
            logits.push(max + sum.ln());
        }

        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut probabilities: Vec<_> = logits.iter().map(|logit| (logit - max).exp()).collect();
        let sum = probabilities.iter().sum::<f64>();
        for probability in &mut probabilities {
            *probability /= sum;
        }
        Ok(probabilities)
    }
}

fn valid_features(features: &[f64], width: usize) -> bool {
    width > 0 && features.len() == width && features.iter().all(|value| value.is_finite())
}

fn squared_distance(left: &[f64], right: &[f64]) -> Result<f64> {
    let sum = left
        .iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>();
    ensure(sum.is_finite(), "retrieval feature distance overflow")?;
    Ok(sum)
}
