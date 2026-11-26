//! Fitted Q-iteration and fixed-policy fitted Q evaluation.
//!
//! See Ernst, Geurts and Wehenkel (JMLR 2005), and Ostrovski et al.
//! (ICML 2017) for the mixed Monte Carlo target. Each sweep fixes the entire
//! target vector before invoking the regression backend.

use super::batch_q::{PreparedBatch, bind, check_initial};
use crate::{
    Result,
    approx::batch::FactualBatch,
    approx::{
        FittedQ, greedy_index,
        regression::{LabelSource, Predictor, Regressor},
    },
    core::{action::Action, ensure, env::PublicHistory, reward::BatchObjective},
    planning::guidance::PolicyGuide,
};
use serde::Serialize;
use std::num::NonZeroUsize;

#[derive(Clone, Copy)]
pub enum TargetMode<'a> {
    Greedy,
    /// A frozen complete-action probability law in legal support order.
    /// A supplied StatisticalActor uses its deterministic deployment law.
    Fixed(&'a dyn PolicyGuide<PublicHistory, Action>),
    MonteCarlo,
    /// The weight on the factual behavior return. Interior mixtures exclude
    /// incomplete rows. They can combine different continuation policies.
    Mixed {
        mc_weight: f64,
    },
}

#[derive(Clone, Copy)]
pub struct FqiConfig<'a> {
    pub sweeps: NonZeroUsize,
    /// Number of factual decision spans. One is ordinary one-step FQI.
    /// Longer spans include the actually executed intervening behavior.
    pub horizon: NonZeroUsize,
    pub target: TargetMode<'a>,
    /// Clip successor predictions to [0, 1] for an explicitly declared
    /// terminal-success objective. Deployed action scores remain raw.
    pub clip_successor: bool,
}

impl Default for FqiConfig<'_> {
    fn default() -> Self {
        Self {
            sweeps: NonZeroUsize::MIN,
            horizon: NonZeroUsize::MIN,
            target: TargetMode::Greedy,
            clip_successor: false,
        }
    }
}

/// The operator used to construct regression targets, excluding the code of a fixed policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Greedy,
    Fixed,
    MonteCarlo,
    Mixed { mc_weight: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TargetDescription {
    pub kind: TargetKind,
    pub horizon: NonZeroUsize,
    pub clip_successor: bool,
}

impl FqiConfig<'_> {
    pub fn target_description(self) -> TargetDescription {
        let kind = match self.target {
            TargetMode::Greedy => TargetKind::Greedy,
            TargetMode::Fixed(_) => TargetKind::Fixed,
            TargetMode::MonteCarlo => TargetKind::MonteCarlo,
            TargetMode::Mixed { mc_weight } => TargetKind::Mixed { mc_weight },
        };
        TargetDescription {
            kind,
            horizon: self.horizon,
            clip_successor: self.clip_successor,
        }
    }
}

pub enum Initial<'a, M> {
    Zero,
    Fitted(&'a FittedQ<M>),
}

pub struct Sweep<D> {
    pub diagnostics: D,
    pub rows: usize,
    pub excluded_incomplete: usize,
    /// Summary of the frozen regression targets for this sweep.
    pub targets: TargetStatistics,
    pub target_description: TargetDescription,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetStatistics {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
}

impl TargetStatistics {
    fn from_targets(targets: &[f64]) -> Self {
        let (minimum, maximum) = targets.iter().copied().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(minimum, maximum), target| (minimum.min(target), maximum.max(target)),
        );
        let scale = minimum.abs().max(maximum.abs());
        let mean = if scale == 0.0 {
            0.0
        } else {
            // Normalize before summing so finite targets cannot overflow the accumulator.
            let scaled_sum = targets.iter().map(|target| target / scale).sum::<f64>();
            (scaled_sum / targets.len() as f64).clamp(-1.0, 1.0) * scale
        };
        Self {
            minimum,
            mean,
            maximum,
        }
    }
}

pub struct Fit<M, D> {
    pub model: FittedQ<M>,
    pub sweeps: Vec<Sweep<D>>,
}

/// Return a fitted model only when every requested sweep succeeds. The
/// initial predictor and the EpisodeStore remain available to their owners.
/// Finite feature coverage and a well-conditioned fit do not imply useful
/// greedy policy improvement under off-policy distribution shift.
pub fn fit<R: Regressor>(
    batch: &FactualBatch<'_, '_>,
    backend: &R,
    config: FqiConfig<'_>,
    initial: Initial<'_, R::Model>,
) -> Result<Fit<R::Model, R::Diagnostics>> {
    let success_objective = matches!(batch.objective(), BatchObjective::TerminalSuccess { .. });
    ensure(
        !config.clip_successor || success_objective,
        "successor clipping requires a terminal-success objective",
    )?;
    let (mc_weight, fixed) = match config.target {
        TargetMode::Greedy => (0., None),
        TargetMode::Fixed(policy) => (0., Some(policy)),
        TargetMode::MonteCarlo => (1., None),
        TargetMode::Mixed { mc_weight } => (mc_weight, None),
    };
    ensure(
        (0.0..=1.0).contains(&mc_weight),
        "invalid Monte Carlo mixture weight",
    )?;
    backend.validate_label_source(if success_objective && mc_weight == 1. {
        LabelSource::ObservedBinary
    } else {
        LabelSource::Derived
    })?;
    let initial = match initial {
        Initial::Zero => None,
        Initial::Fitted(model) => {
            check_initial(batch, model)?;
            Some(model.predictor())
        }
    };
    let prepared = PreparedBatch::new(batch, config.horizon, mc_weight < 1., mc_weight > 0.)?;
    let fit_targets = backend.prepare(&prepared.design, &prepared.weights)?;
    let fixed = fixed
        .map(|policy| {
            prepared
                .successors
                .iter()
                .map(|state| state.probabilities(policy))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    let mut previous = None;
    let mut sweeps = Vec::new();
    let mut targets = Vec::with_capacity(prepared.rows.len());
    for _ in 0..config.sweeps.get() {
        let predictor = previous.as_ref().or(initial);
        let continuation = prepared
            .successors
            .iter()
            .enumerate()
            .map(|(index, state)| {
                let mut values = state.values(predictor)?;
                if config.clip_successor {
                    for value in &mut values {
                        *value = value.clamp(0., 1.);
                    }
                }
                match &fixed {
                    Some(probabilities) => Ok(values
                        .iter()
                        .zip(&probabilities[index])
                        .map(|(q, p)| q * p)
                        .sum()),
                    None => Ok(values[greedy_index(state.actions, &values)?]),
                }
            })
            .collect::<Result<Vec<f64>>>()?;
        targets.clear();
        for row in &prepared.rows {
            let mc = row.monte_carlo.unwrap_or(0.);
            let value = if mc_weight == 1. {
                mc
            } else {
                let td = row.reward
                    + row
                        .successor
                        .map_or(0., |index| row.discount * continuation[index]);
                if mc_weight == 0. {
                    td
                } else {
                    (1. - mc_weight) * td + mc_weight * mc
                }
            };
            ensure(value.is_finite(), "nonfinite fitted Q target")?;
            targets.push(value);
        }
        let (model, diagnostics) = fit_targets(&targets)?;
        model.validate()?;
        ensure(
            model.dimension() == prepared.design.ncols(),
            "regression returned the wrong feature width",
        )?;
        previous = Some(model);
        sweeps.push(Sweep {
            diagnostics,
            rows: targets.len(),
            excluded_incomplete: batch.rows().len() - targets.len(),
            targets: TargetStatistics::from_targets(&targets),
            target_description: config.target_description(),
        });
    }
    let model = bind(batch, previous.expect("nonzero fitted sweeps"))?;
    Ok(Fit { model, sweeps })
}
