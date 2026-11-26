//! Checked full-row adapters. Scalar grid projection allocates no distribution vectors.
use super::policy_batch::PolicyBatch;
use crate::core::ensure;
use crate::{
    Error, Result,
    core::action::ActionSpace,
    diagnostics::{
        policy::GaussianMarginal,
        representations::{categorical_kl, diagonal_gaussian_kl},
        schema::{Array, DType, PolicyParameters},
    },
};

pub const MAX_DISTRIBUTION_BYTES: usize = 1024 * 1024;
const MAX_KL_SCRATCH_BYTES: usize = 4 * 1024 * 1024;
#[derive(Debug, Clone, PartialEq)]
pub enum SelectedDistribution {
    Deterministic,
    Categorical {
        probabilities: Vec<f64>,
        valid: Vec<bool>,
    },
    Gaussian(GaussianMarginal),
}
fn action_count(batch: &PolicyBatch<'_>) -> usize {
    match batch.action_space() {
        ActionSpace::Discrete { count } => *count,
        space => space.dimensions(),
    }
}
fn indices(batch: &PolicyBatch<'_>, row: usize, action: usize) -> Result<()> {
    ensure(
        row < batch.rows() && action < action_count(batch),
        "policy row/action outside checked batch",
    )
}
fn parameter<'a>(batch: &PolicyBatch<'a>, key: &str) -> Result<&'a Array> {
    batch
        .parameter_array(key)
        .ok_or_else(|| Error::State("missing checked policy parameter".into()))
}
fn scalar(array: &Array, index: usize) -> Result<f64> {
    match array.dtype() {
        DType::F32 => array.as_f32()?.get(index).map(|&v| f64::from(v)),
        DType::F64 => array.as_f64()?.get(index).copied(),
        _ => None,
    }
    .ok_or_else(|| Error::Shape("policy scalar outside floating array".into()))
}
fn marginal(
    batch: &PolicyBatch<'_>,
    row: usize,
    action: usize,
    mean: &str,
    log_std: &str,
) -> Result<GaussianMarginal> {
    let width = batch.action_space().dimensions();
    let std = parameter(batch, log_std)?;
    Ok(GaussianMarginal {
        mean: scalar(parameter(batch, mean)?, row * width + action)?,
        log_std: scalar(
            std,
            if std.shape().len() == 1 {
                action
            } else {
                row * width + action
            },
        )?,
        bounds: match batch.action_space() {
            ActionSpace::Box { low, high } => {
                Some([f64::from(low[action]), f64::from(high[action])])
            }
            _ => None,
        },
    })
}
/// Deterministic discrete projections are indicators, even on numerical input grids.
pub fn is_indicator(batch: &PolicyBatch<'_>) -> bool {
    matches!(batch.action_space(), ActionSpace::Discrete { .. })
        && matches!(batch.parameters(), PolicyParameters::Deterministic)
}
pub fn project(batch: &PolicyBatch<'_>, row: usize, action: usize) -> Result<f64> {
    indices(batch, row, action)?;
    match batch.parameters() {
        PolicyParameters::Deterministic => match batch.action_space() {
            ActionSpace::Discrete { .. } => {
                Ok(f64::from(batch.actions().as_u64()?[row] == action as u64))
            }
            space => scalar(batch.actions(), row * space.dimensions() + action),
        },
        PolicyParameters::Categorical { probabilities } => scalar(
            parameter(batch, probabilities)?,
            row * action_count(batch) + action,
        ),
        PolicyParameters::Gaussian { mean, log_std, .. } => {
            marginal(batch, row, action, mean, log_std)?.expected_action()
        }
    }
}
pub fn selected_distribution(
    batch: &PolicyBatch<'_>,
    row: usize,
    action: usize,
) -> Result<SelectedDistribution> {
    indices(batch, row, action)?;
    Ok(match batch.parameters() {
        PolicyParameters::Deterministic => SelectedDistribution::Deterministic,
        PolicyParameters::Categorical { probabilities } => {
            let n = action_count(batch);
            let overhead = std::mem::size_of::<SelectedDistribution>();
            ensure(
                n.checked_mul(9)
                    .and_then(|b| b.checked_add(overhead))
                    .is_some_and(|b| b <= MAX_DISTRIBUTION_BYTES),
                "selected distribution exceeds byte budget",
            )?;
            let a = parameter(batch, probabilities)?;
            // A fallible iterator collect can grow geometrically even though n
            // is known. Reserve once and charge actual capacity before filling.
            let mut probabilities = Vec::with_capacity(n);
            let probability_bytes = probabilities
                .capacity()
                .checked_mul(8)
                .and_then(|b| b.checked_add(overhead))
                .ok_or_else(|| Error::Config("selected distribution capacity overflow".into()))?;
            ensure(
                probability_bytes
                    .checked_add(n)
                    .is_some_and(|b| b <= MAX_DISTRIBUTION_BYTES),
                "selected probability capacity exceeds byte budget",
            )?;
            for i in 0..n {
                probabilities.push(scalar(a, row * n + i)?);
            }
            let valid = batch
                .effective_mask_with_budget(row, MAX_KL_SCRATCH_BYTES - probability_bytes)?
                .unwrap_or_else(|| vec![true; n]);
            ensure(
                probability_bytes
                    .checked_add(valid.capacity())
                    .is_some_and(|b| b <= MAX_DISTRIBUTION_BYTES),
                "selected distribution capacities exceed byte budget",
            )?;
            SelectedDistribution::Categorical {
                probabilities,
                valid,
            }
        }
        PolicyParameters::Gaussian { mean, log_std, .. } => {
            let g = marginal(batch, row, action, mean, log_std)?;
            // Validate positive representable sigma even if only the CDF is used.
            g.cdf(g.mean)?;
            SelectedDistribution::Gaussian(g)
        }
    })
}
/// Full KL(current || reference). The caller must independently certify equal
/// physical inputs/encoding/mask/history conditioning. Independently valid masks
/// do not certify that match; this pure math adapter deliberately retains zeros
/// (including genuine support infinity) and never changes global input matching.
pub fn current_reference_kl(
    current: &PolicyBatch<'_>,
    row: usize,
    reference: &PolicyBatch<'_>,
    reference_row: usize,
) -> Result<Option<f64>> {
    indices(current, row, 0)?;
    indices(reference, reference_row, 0)?;
    ensure(
        current.action_space() == reference.action_space(),
        "policy action spaces/transforms differ",
    )?;
    let width = action_count(current);
    let admit_stochastic = || {
        ensure(
            width
                .checked_mul(32)
                .is_some_and(|b| b <= MAX_KL_SCRATCH_BYTES),
            "full policy KL exceeds scratch byte budget",
        )
    };
    let vector =
        |batch: &PolicyBatch<'_>, key: &str, row: usize, shared: bool| -> Result<Vec<f64>> {
            let a = parameter(batch, key)?;
            let offset = if shared && a.shape().len() == 1 {
                0
            } else {
                row * width
            };
            (0..width).map(|i| scalar(a, offset + i)).collect()
        };
    match (current.parameters(), reference.parameters()) {
        (PolicyParameters::Deterministic, PolicyParameters::Deterministic) => Ok(None),
        (
            PolicyParameters::Categorical { probabilities: p },
            PolicyParameters::Categorical { probabilities: q },
        ) => {
            admit_stochastic()?;
            Ok(Some(categorical_kl(
                &vector(current, p, row, false)?,
                &vector(reference, q, reference_row, false)?,
            )?))
        }
        (
            PolicyParameters::Gaussian {
                mean: pm,
                log_std: ps,
                tanh_bounded: p,
            },
            PolicyParameters::Gaussian {
                mean: qm,
                log_std: qs,
                tanh_bounded: q,
            },
        ) if p == q => {
            admit_stochastic()?;
            Ok(Some(diagonal_gaussian_kl(
                &vector(current, pm, row, false)?,
                &vector(current, ps, row, true)?,
                &vector(reference, qm, reference_row, false)?,
                &vector(reference, qs, reference_row, true)?,
            )?))
        }
        _ => Err(Error::Config(
            "incompatible policy families/transforms".into(),
        )),
    }
}
