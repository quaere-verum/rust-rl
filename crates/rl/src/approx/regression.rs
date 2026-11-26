//! CPU-based, weighted regression over fixed-width `f64` features.

mod gaussian;
mod logistic;
mod ridge;

pub use gaussian::{GaussianPosterior, GaussianPrediction};
pub use logistic::{
    LogisticDiagnostics, LogisticModel, LogisticTargetKind, WeightedLogistic,
    signed_terminal_returns_to_binary,
};
pub use ridge::{RidgeDiagnostics, WeightedRidge};

use crate::{Error, Result, core::ensure};
use nalgebra::{DMatrix, DVector, Dyn, linalg::SVD};
use serde::{Deserialize, Serialize};

const DEFAULT_RELATIVE_CUTOFF: f64 = 1e-12;

/// A fitted scalar predictor over a fixed number of features.
pub trait Predictor: Clone {
    fn dimension(&self) -> usize;
    fn validate(&self) -> Result<()>;
    fn predict(&self, features: &[f64]) -> Result<f64>;
}

/// How a target vector was obtained by a learning algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSource {
    /// Each target is a factual indicator of an observed event.
    ObservedBinary,
    /// Targets were constructed from returns, bootstraps or other estimates.
    Derived,
}

/// A fitted predictor and the diagnostics returned by its regression backend.
pub type FitResult<R> = Result<(<R as Regressor>::Model, <R as Regressor>::Diagnostics)>;

/// Fits a new model from a design matrix with one target and weight per row.
/// Fitting leaves previously returned models available to their owners.
pub trait Regressor {
    type Model: Predictor;
    type Diagnostics;

    /// Checks provenance when the caller can identify the source of its targets.
    fn validate_label_source(&self, _source: LabelSource) -> Result<()> {
        Ok(())
    }

    fn fit(&self, features: &DMatrix<f64>, targets: &[f64], weights: &[f64]) -> FitResult<Self>;

    /// Prepares a fixed design for one or more independent target vectors.
    /// The default implementation calls `fit` for each target vector.
    fn prepare<'a>(
        &'a self,
        features: &'a DMatrix<f64>,
        weights: &'a [f64],
    ) -> Result<impl Fn(&[f64]) -> FitResult<Self> + 'a>
    where
        Self: Sized,
    {
        Ok(move |targets: &[f64]| self.fit(features, targets, weights))
    }
}

/// Nonnegative L2 penalty on each coefficient, relative to the weighted mean loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "DiagonalPenaltyRecord")]
pub enum DiagonalPenalty {
    Uniform(f64),
    /// Supply one penalty per design column. A zero can leave an explicit intercept free.
    PerCoefficient(Vec<f64>),
}

impl Default for DiagonalPenalty {
    fn default() -> Self {
        Self::Uniform(1e-6)
    }
}

impl DiagonalPenalty {
    fn validate(&self) -> Result<()> {
        let values = match self {
            Self::Uniform(value) => std::slice::from_ref(value),
            Self::PerCoefficient(values) => values,
        };
        ensure(
            !values.is_empty()
                && values
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0),
            "diagonal penalties must be nonempty, finite and nonnegative",
        )
    }

    pub(crate) fn diagonal(&self, dimension: usize) -> Result<Vec<f64>> {
        self.validate()?;
        match self {
            Self::Uniform(value) => Ok(vec![*value; dimension]),
            Self::PerCoefficient(values) if values.len() == dimension => Ok(values.clone()),
            Self::PerCoefficient(_) => Err(Error::Shape(
                "per-coefficient penalty requires one entry per feature".into(),
            )),
        }
    }
}

#[derive(Deserialize)]
enum DiagonalPenaltyRecord {
    Uniform(f64),
    PerCoefficient(Vec<f64>),
}

impl TryFrom<DiagonalPenaltyRecord> for DiagonalPenalty {
    type Error = Error;

    fn try_from(record: DiagonalPenaltyRecord) -> Result<Self> {
        let penalty = match record {
            DiagonalPenaltyRecord::Uniform(value) => Self::Uniform(value),
            DiagonalPenaltyRecord::PerCoefficient(values) => Self::PerCoefficient(values),
        };
        penalty.validate()?;
        Ok(penalty)
    }
}

/// Linear prediction with one coefficient per feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "LinearModelRecord")]
pub struct LinearModel {
    coefficients: Vec<f64>,
}

impl LinearModel {
    pub fn new(coefficients: Vec<f64>) -> Result<Self> {
        let model = Self { coefficients };
        model.validate()?;
        Ok(model)
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

impl Predictor for LinearModel {
    fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    fn validate(&self) -> Result<()> {
        if self.coefficients.is_empty() {
            return Err(Error::Shape("linear model requires coefficients".into()));
        }
        if self.coefficients.iter().any(|value| !value.is_finite()) {
            return Err(Error::Numerical(
                "linear coefficients must be finite".into(),
            ));
        }
        Ok(())
    }

    fn predict(&self, features: &[f64]) -> Result<f64> {
        if features.len() != self.dimension() {
            return Err(Error::Shape(format!(
                "linear prediction expects {} features, received {}",
                self.dimension(),
                features.len(),
            )));
        }
        finite_dot(features.iter(), self.coefficients.iter())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LinearModelRecord {
    coefficients: Vec<f64>,
}

impl TryFrom<LinearModelRecord> for LinearModel {
    type Error = Error;

    fn try_from(record: LinearModelRecord) -> Result<Self> {
        Self::new(record.coefficients)
    }
}

/// Diagnostics of the matrix sent to a rank-aware solve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolveDiagnostics {
    pub rows: usize,
    pub columns: usize,
    /// Number of singular values retained by the cutoff.
    pub rank: usize,
    /// Finite spectral condition number when all coefficient directions are retained.
    pub condition_number: Option<f64>,
    /// Singular values in descending order, including directions discarded by the solve.
    pub singular_values: Vec<f64>,
    pub absolute_cutoff: f64,
    /// Euclidean residual norm of the solved system.
    pub residual_norm: f64,
    /// Residual norm divided by the right-hand side norm. A zero right-hand side uses the absolute norm.
    pub relative_residual: f64,
}

/// Minimum-norm least-squares solution, including nonsymmetric and singular systems.
pub(crate) fn solve_system(
    matrix: DMatrix<f64>,
    target: DVector<f64>,
    relative_cutoff: f64,
) -> Result<(DVector<f64>, SolveDiagnostics)> {
    if matrix.nrows() != target.len() {
        return Err(Error::Shape(
            "linear system requires one right-hand-side entry per row".into(),
        ));
    }
    let factor = RankAwareSvd::new(matrix.clone(), relative_cutoff)?;
    let coefficients = factor.solve(&target)?;
    let residual = matrix * &coefficients - &target;
    let diagnostics =
        factor.diagnostics(norm(residual.iter().copied()), norm(target.iter().copied()))?;
    Ok((coefficients, diagnostics))
}

struct RankAwareSvd {
    svd: SVD<f64, Dyn, Dyn>,
    rows: usize,
    columns: usize,
    cutoff: f64,
    rank: usize,
    condition_number: Option<f64>,
    singular_values: Vec<f64>,
}

impl RankAwareSvd {
    fn new(matrix: DMatrix<f64>, relative_cutoff: f64) -> Result<Self> {
        validate_relative_cutoff(relative_cutoff)?;
        let (rows, columns) = matrix.shape();
        if rows == 0 || columns == 0 {
            return Err(Error::Shape(
                "linear solve requires a nonempty matrix".into(),
            ));
        }
        if matrix.iter().any(|entry| !entry.is_finite()) {
            return Err(Error::Numerical(
                "linear system entries must be finite".into(),
            ));
        }
        let svd = SVD::try_new_unordered(matrix, true, true, 5.0 * f64::EPSILON, 10_000)
            .ok_or_else(|| {
                Error::Numerical("singular value decomposition did not converge".into())
            })?;
        if svd.singular_values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Numerical("singular values must be finite".into()));
        }
        let cutoff = relative_cutoff * svd.singular_values.max();
        let rank = svd.rank(cutoff);
        let mut singular_values = svd.singular_values.as_slice().to_vec();
        singular_values.sort_unstable_by(|a, b| b.total_cmp(a));
        let condition_number = (rank == columns)
            .then(|| singular_values[0] / singular_values[rank - 1])
            .filter(|value| value.is_finite());
        Ok(Self {
            svd,
            rows,
            columns,
            cutoff,
            rank,
            condition_number,
            singular_values,
        })
    }

    fn solve(&self, target: &DVector<f64>) -> Result<DVector<f64>> {
        if target.iter().any(|entry| !entry.is_finite()) {
            return Err(Error::Numerical(
                "linear solve targets must be finite".into(),
            ));
        }
        let coefficients = self
            .svd
            .solve(target, self.cutoff)
            .map_err(|reason| Error::Numerical(reason.into()))?;
        if coefficients.iter().any(|entry| !entry.is_finite()) {
            return Err(Error::Numerical(
                "linear solve coefficients must be finite".into(),
            ));
        }
        Ok(coefficients)
    }

    fn diagnostics(&self, residual_norm: f64, target_norm: f64) -> Result<SolveDiagnostics> {
        let relative_residual = if target_norm == 0.0 {
            residual_norm
        } else {
            residual_norm / target_norm
        };
        if !residual_norm.is_finite() || !target_norm.is_finite() || !relative_residual.is_finite()
        {
            return Err(Error::Numerical(
                "linear solve residual norms must be finite".into(),
            ));
        }
        Ok(SolveDiagnostics {
            rows: self.rows,
            columns: self.columns,
            rank: self.rank,
            condition_number: self.condition_number,
            singular_values: self.singular_values.clone(),
            absolute_cutoff: self.cutoff,
            residual_norm,
            relative_residual,
        })
    }
}

fn validate_relative_cutoff(cutoff: f64) -> Result<()> {
    ensure(
        0.0 < cutoff && cutoff < 1.0,
        "relative singular-value cutoff must be in (0, 1)",
    )
}

fn validate_design(features: &DMatrix<f64>, weights: &[f64]) -> Result<()> {
    if features.nrows() == 0 || features.ncols() == 0 || features.nrows() != weights.len() {
        return Err(Error::Shape(
            "regression requires a nonempty design with a weight per row".into(),
        ));
    }
    if features.iter().any(|value| !value.is_finite()) {
        return Err(Error::Numerical(
            "regression features must be finite".into(),
        ));
    }
    Ok(())
}

fn validate_targets(rows: usize, targets: &[f64]) -> Result<()> {
    if rows != targets.len() {
        return Err(Error::Shape("regression requires a target per row".into()));
    }
    if targets.iter().any(|value| !value.is_finite()) {
        return Err(Error::Numerical("regression targets must be finite".into()));
    }
    Ok(())
}

fn weight_scale(weights: &[f64]) -> Result<(f64, f64)> {
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Error::Numerical(
            "sample weights must be finite and nonnegative".into(),
        ));
    }
    let maximum = weights.iter().copied().fold(0.0, f64::max);
    if maximum == 0.0 {
        return Err(Error::Numerical(
            "sample weights must have positive total mass".into(),
        ));
    }
    let scaled_total = weights.iter().map(|weight| weight / maximum).sum();
    Ok((maximum, scaled_total))
}

pub(crate) fn normalized_weights(weights: &[f64]) -> Result<Vec<f64>> {
    let (maximum, scaled_total) = weight_scale(weights)?;
    Ok(weights
        .iter()
        .map(|weight| (weight / maximum) / scaled_total)
        .collect())
}

pub(crate) fn normalized_sqrt_weights(weights: &[f64]) -> Result<Vec<f64>> {
    let (maximum, scaled_total) = weight_scale(weights)?;
    let maximum_root = maximum.sqrt();
    let total_root = scaled_total.sqrt();
    // Taking square roots first preserves contributions from extremely small weight ratios.
    Ok(weights
        .iter()
        .map(|weight| (weight.sqrt() / maximum_root) / total_root)
        .collect())
}

fn finite_dot<'a, 'b>(
    a: impl Iterator<Item = &'a f64>,
    b: impl Iterator<Item = &'b f64>,
) -> Result<f64> {
    let product: f64 = a.zip(b).map(|(a, b)| a * b).sum();
    if !product.is_finite() {
        return Err(Error::Numerical("linear prediction must be finite".into()));
    }
    Ok(product)
}

fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, f64::hypot)
}

fn check_matrix_size(rows: usize, columns: usize) -> Result<()> {
    if rows
        .checked_mul(columns)
        .is_none_or(|size| size > isize::MAX as usize / size_of::<f64>())
    {
        return Err(Error::Shape("regression matrix is too large".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
