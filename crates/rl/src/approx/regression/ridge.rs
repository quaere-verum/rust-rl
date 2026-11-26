use super::{
    DEFAULT_RELATIVE_CUTOFF, DiagonalPenalty, FitResult, LinearModel, RankAwareSvd, Regressor,
    SolveDiagnostics, check_matrix_size, norm, normalized_sqrt_weights, validate_design,
    validate_relative_cutoff, validate_targets,
};
use crate::{Error, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Fits linear ridge on an augmented design via a rank-aware SVD.
/// The objective is the weighted mean squared error plus `sum_j penalty_j * coefficient_j^2`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "WeightedRidgeRecord")]
pub struct WeightedRidge {
    pub penalty: DiagonalPenalty,
    /// Retain singular values above this fraction of the largest one.
    pub relative_cutoff: f64,
}

impl Default for WeightedRidge {
    fn default() -> Self {
        Self {
            penalty: DiagonalPenalty::default(),
            relative_cutoff: DEFAULT_RELATIVE_CUTOFF,
        }
    }
}

impl WeightedRidge {
    pub fn validate(&self) -> Result<()> {
        self.penalty.validate()?;
        validate_relative_cutoff(self.relative_cutoff)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightedRidgeRecord {
    penalty: DiagonalPenalty,
    relative_cutoff: f64,
}

impl TryFrom<WeightedRidgeRecord> for WeightedRidge {
    type Error = Error;

    fn try_from(record: WeightedRidgeRecord) -> Result<Self> {
        let ridge = Self {
            penalty: record.penalty,
            relative_cutoff: record.relative_cutoff,
        };
        ridge.validate()?;
        Ok(ridge)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RidgeDiagnostics {
    /// Root mean squared training error weighted by normalized sample weights.
    pub weighted_residual_norm: f64,
    /// Numerical diagnostics of the augmented design, including penalty rows.
    pub solve: SolveDiagnostics,
}

impl Regressor for WeightedRidge {
    type Model = LinearModel;
    type Diagnostics = RidgeDiagnostics;

    fn fit(
        &self,
        features: &DMatrix<f64>,
        targets: &[f64],
        weights: &[f64],
    ) -> Result<(Self::Model, Self::Diagnostics)> {
        validate_targets(features.nrows(), targets)?;
        let fit_targets = self.prepare(features, weights)?;
        fit_targets(targets)
    }

    fn prepare<'a>(
        &'a self,
        features: &'a DMatrix<f64>,
        weights: &'a [f64],
    ) -> Result<impl Fn(&[f64]) -> FitResult<Self> + 'a> {
        validate_relative_cutoff(self.relative_cutoff)?;
        validate_design(features, weights)?;
        let (rows, columns) = features.shape();
        let penalty_roots: Vec<_> = self
            .penalty
            .diagonal(columns)?
            .into_iter()
            .map(f64::sqrt)
            .collect();
        let penalized_columns: Vec<_> = penalty_roots
            .iter()
            .enumerate()
            .filter_map(|(column, &penalty)| (penalty > 0.0).then_some((column, penalty)))
            .collect();
        let augmented_rows = rows
            .checked_add(penalized_columns.len())
            .ok_or_else(|| Error::Shape("augmented regression design is too large".into()))?;
        check_matrix_size(augmented_rows, columns)?;
        let weight_roots = normalized_sqrt_weights(weights)?;

        let mut augmented = DMatrix::from_fn(augmented_rows, columns, |row, column| {
            if row < rows {
                features[(row, column)] * weight_roots[row]
            } else {
                0.0
            }
        });
        for (row, &(column, penalty)) in penalized_columns.iter().enumerate() {
            augmented[(row + rows, column)] = penalty;
        }
        let prepared = PreparedRidge {
            features,
            weight_roots,
            penalty_roots,
            factor: RankAwareSvd::new(augmented, self.relative_cutoff)?,
        };
        Ok(move |targets: &[f64]| prepared.fit(targets))
    }
}

struct PreparedRidge<'a> {
    features: &'a DMatrix<f64>,
    weight_roots: Vec<f64>,
    penalty_roots: Vec<f64>,
    factor: RankAwareSvd,
}

impl PreparedRidge<'_> {
    fn fit(&self, targets: &[f64]) -> Result<(LinearModel, RidgeDiagnostics)> {
        validate_targets(self.features.nrows(), targets)?;
        let mut target = DVector::zeros(self.factor.rows);
        for (row, (weight, value)) in self.weight_roots.iter().zip(targets).enumerate() {
            target[row] = weight * value;
        }

        let coefficients = self.factor.solve(&target)?;
        let residuals = self
            .features
            .row_iter()
            .zip(&self.weight_roots)
            .zip(target.iter())
            .map(|((row, weight), target)| {
                let prediction: f64 = row
                    .iter()
                    .zip(&coefficients)
                    .map(|(feature, coefficient)| (weight * feature) * coefficient)
                    .sum();
                prediction - target
            });
        let weighted_residual_norm = norm(residuals);
        let penalty_norm = norm(
            self.penalty_roots
                .iter()
                .zip(&coefficients)
                .map(|(penalty, coefficient)| penalty * coefficient),
        );
        let solve = self.factor.diagnostics(
            weighted_residual_norm.hypot(penalty_norm),
            norm(target.iter().copied()),
        )?;
        let model = LinearModel::new(coefficients.as_slice().to_vec())?;
        Ok((
            model,
            RidgeDiagnostics {
                weighted_residual_norm,
                solve,
            },
        ))
    }
}
