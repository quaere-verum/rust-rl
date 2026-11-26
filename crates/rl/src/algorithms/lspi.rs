//! Weighted LSTDQ and least-squares policy iteration.
//!
//! See Lagoudakis and Parr (JMLR 2003). A fixed basis and factual batch
//! define a potentially nonsymmetric moment system. SVD supplies a
//! rank-aware solve. The residual concerns the system, not policy value.

use super::batch_q::{Policy, PreparedBatch, Successor, bind};
use crate::{
    Result,
    approx::batch::FactualBatch,
    approx::{
        FittedQ, greedy_index,
        regression::{
            DiagonalPenalty, LinearModel, SolveDiagnostics, normalized_sqrt_weights, solve_system,
        },
    },
    core::ensure,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LspiConfig {
    pub max_iterations: NonZeroUsize,
    /// Added to the diagonal of the normalized moment system.
    pub penalty: DiagonalPenalty,
    /// Relative singular-value cutoff for the regularized system.
    pub relative_cutoff: f64,
}

impl Default for LspiConfig {
    fn default() -> Self {
        Self {
            max_iterations: NonZeroUsize::new(10).expect("nonzero default"),
            penalty: DiagonalPenalty::default(),
            relative_cutoff: 1e-12,
        }
    }
}

pub struct Evaluation {
    pub model: FittedQ<LinearModel>,
    pub solve: SolveDiagnostics,
}

pub struct Iteration {
    pub solve: SolveDiagnostics,
    /// Distinct, positively discounted factual successor frames at which
    /// improvement changed the complete-action law.
    pub changed_actions: usize,
}

pub struct Fit {
    pub model: FittedQ<LinearModel>,
    pub iterations: Vec<Iteration>,
    pub stable: bool,
}

/// Independent policy evaluation. The caller fixes the policy for the entire
/// solve. At a true terminal (or zero discount) it is never consulted.
/// A supplied StatisticalActor uses its deterministic deployment law.
pub fn evaluate(
    batch: &FactualBatch<'_, '_>,
    config: &LspiConfig,
    policy: &Policy<'_>,
) -> Result<Evaluation> {
    let prepared = PreparedBatch::new(batch, NonZeroUsize::MIN, true, false)?;
    let probabilities = prepared
        .successors
        .iter()
        .map(|s| s.probabilities(policy))
        .collect::<Result<Vec<_>>>()?;
    let weight_roots = normalized_sqrt_weights(&prepared.weights)?;
    let penalty = config.penalty.diagonal(prepared.design.ncols())?;
    let (model, solve) = solve(
        &prepared,
        &weight_roots,
        &penalty,
        config.relative_cutoff,
        &probabilities,
    )?;
    Ok(Evaluation {
        model: bind(batch, model)?,
        solve,
    })
}

/// Alternate evaluation and deterministic greedy improvement on one factual
/// batch. A missing initial policy chooses by semantic tie order under zero Q.
/// Stability concerns only the observed live successors. At the iteration cap,
/// the returned Q evaluates the policy used for the last solve, and its greedy
/// actor is the corresponding improvement. Approximation may cycle or degrade.
pub fn fit(
    batch: &FactualBatch<'_, '_>,
    config: &LspiConfig,
    initial: Option<&Policy<'_>>,
) -> Result<Fit> {
    let prepared = PreparedBatch::new(batch, NonZeroUsize::MIN, true, false)?;
    let weight_roots = normalized_sqrt_weights(&prepared.weights)?;
    let penalty = config.penalty.diagonal(prepared.design.ncols())?;
    let mut policy = match initial {
        Some(initial) => prepared
            .successors
            .iter()
            .map(|s| s.probabilities(initial))
            .collect::<Result<Vec<_>>>()?,
        None => greedy_policy(&prepared.successors, None)?,
    };
    let mut fitted = None;
    let mut stable = false;
    let mut iterations = Vec::new();
    for _ in 0..config.max_iterations.get() {
        let (model, solve) = solve(
            &prepared,
            &weight_roots,
            &penalty,
            config.relative_cutoff,
            &policy,
        )?;
        let greedy = greedy_policy(&prepared.successors, Some(&model))?;
        let changed_actions = policy
            .iter()
            .zip(&greedy)
            .filter(|(old, new)| old != new)
            .count();
        iterations.push(Iteration {
            solve,
            changed_actions,
        });
        fitted = Some(model);
        stable = changed_actions == 0;
        if stable {
            break;
        }
        policy = greedy;
    }
    Ok(Fit {
        model: bind(batch, fitted.expect("nonzero LSPI iterations"))?,
        iterations,
        stable,
    })
}

fn greedy_policy(
    successors: &[Successor<'_>],
    model: Option<&LinearModel>,
) -> Result<Vec<Vec<f64>>> {
    successors
        .iter()
        .map(|state| {
            let best = greedy_index(state.actions, &state.values(model)?)?;
            let mut probabilities = vec![0.; state.actions.len()];
            probabilities[best] = 1.;
            Ok(probabilities)
        })
        .collect()
}

fn solve(
    batch: &PreparedBatch<'_>,
    weight_roots: &[f64],
    penalty: &[f64],
    relative_cutoff: f64,
    policy: &[Vec<f64>],
) -> Result<(LinearModel, SolveDiagnostics)> {
    let width = batch.design.ncols();
    let expected: Vec<Vec<f64>> = batch
        .successors
        .iter()
        .zip(policy)
        .map(|(state, probabilities)| {
            let mut mean = vec![0.; width];
            for (features, probability) in state.features.iter().zip(probabilities) {
                for (value, feature) in mean.iter_mut().zip(*features) {
                    *value += probability * feature;
                }
            }
            mean
        })
        .collect();
    let mut matrix = DMatrix::<f64>::zeros(width, width);
    let mut target = DVector::<f64>::zeros(width);
    let mut weighted_features = vec![0.; width];
    for (index, (row, root)) in batch.rows.iter().zip(weight_roots).enumerate() {
        let scaled_reward = root * row.reward;
        for i in 0..width {
            weighted_features[i] = root * batch.design[(index, i)];
            target[i] += weighted_features[i] * scaled_reward;
        }
        for (j, (mut column, &feature)) in
            matrix.column_iter_mut().zip(&weighted_features).enumerate()
        {
            let next = row
                .successor
                .map_or(0., |slot| (root * expected[slot][j]) * row.discount);
            let difference = feature - next;
            for (entry, weighted) in column.iter_mut().zip(&weighted_features) {
                *entry += weighted * difference;
            }
        }
    }
    for (i, penalty) in penalty.iter().enumerate() {
        matrix[(i, i)] += penalty;
    }
    ensure(
        matrix
            .iter()
            .chain(target.iter())
            .all(|value| value.is_finite()),
        "LSTDQ moment system overflow",
    )?;
    let (coefficients, diagnostics) = solve_system(matrix, target, relative_cutoff)?;
    Ok((
        LinearModel::new(coefficients.as_slice().to_vec())?,
        diagnostics,
    ))
}
