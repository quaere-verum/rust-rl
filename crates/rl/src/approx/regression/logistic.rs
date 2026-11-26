use super::{
    DiagonalPenalty, LabelSource, LinearModel, Predictor, Regressor, check_matrix_size, finite_dot,
    normalized_weights, validate_design, validate_targets,
};
use crate::{Error, Result, core::ensure};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Interpretation of logistic targets, retained alongside the fitted coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogisticTargetKind {
    /// Observed binary event indicators, each exactly 0 or 1.
    FactualBinary,
    /// Probability-valued targets in the closed unit interval.
    SoftLabels,
}

/// Converts a declared `-1`/`+1` terminal-return encoding to binary event labels.
pub fn signed_terminal_returns_to_binary(returns: &[f64]) -> Result<Vec<f64>> {
    returns
        .iter()
        .map(|&value| match value {
            -1.0 => Ok(0.0),
            1.0 => Ok(1.0),
            _ => Err(Error::Config(
                "signed terminal returns must be exactly -1 or +1".into(),
            )),
        })
        .collect()
}

/// Logistic predictor whose logit is linear in the supplied features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogisticModel {
    linear: LinearModel,
    target_kind: LogisticTargetKind,
}

impl LogisticModel {
    pub fn new(coefficients: Vec<f64>, target_kind: LogisticTargetKind) -> Result<Self> {
        Ok(Self {
            linear: LinearModel::new(coefficients)?,
            target_kind,
        })
    }

    pub fn coefficients(&self) -> &[f64] {
        self.linear.coefficients()
    }

    pub fn target_kind(&self) -> LogisticTargetKind {
        self.target_kind
    }

    pub fn logit(&self, features: &[f64]) -> Result<f64> {
        self.linear.predict(features)
    }
}

impl Predictor for LogisticModel {
    fn dimension(&self) -> usize {
        self.linear.dimension()
    }

    fn validate(&self) -> Result<()> {
        self.linear.validate()
    }

    /// Predicts the probability associated with a feature vector.
    fn predict(&self, features: &[f64]) -> Result<f64> {
        let logit = self.logit(features)?;
        let exponential = (-logit.abs()).exp();
        if logit >= 0.0 {
            Ok(1.0 / (1.0 + exponential))
        } else {
            Ok(exponential / (1.0 + exponential))
        }
    }
}

/// Fits weighted logistic regression using damped Newton steps with backtracking.
/// The loss is weighted mean Bernoulli cross-entropy plus half the diagonal L2 penalty.
/// Fitting scales each design column and returns a model in the original feature units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "WeightedLogisticRecord")]
pub struct WeightedLogistic {
    pub penalty: DiagonalPenalty,
    pub target_kind: LogisticTargetKind,
    /// Maximum number of accepted Newton steps.
    pub max_iterations: usize,
    /// Convergence tolerance on both scaled infinity norms of the objective gradient.
    pub gradient_tolerance: f64,
    /// Positive diagonal damping after scaling the Newton system.
    pub damping: f64,
    /// Maximum number of trial step sizes per Newton step.
    pub max_backtracking_steps: usize,
}

impl Default for WeightedLogistic {
    fn default() -> Self {
        Self {
            penalty: DiagonalPenalty::default(),
            target_kind: LogisticTargetKind::FactualBinary,
            max_iterations: 100,
            gradient_tolerance: 1e-8,
            damping: 1e-8,
            max_backtracking_steps: 40,
        }
    }
}

impl WeightedLogistic {
    pub fn validate(&self) -> Result<()> {
        self.penalty.validate()?;
        ensure(
            self.max_iterations > 0 && self.max_backtracking_steps > 0,
            "logistic iteration and backtracking limits must be positive",
        )?;
        ensure(
            self.gradient_tolerance.is_finite()
                && self.gradient_tolerance > 0.0
                && self.damping.is_finite()
                && self.damping > 0.0,
            "logistic gradient tolerance and damping must be positive and finite",
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WeightedLogisticRecord {
    penalty: DiagonalPenalty,
    target_kind: LogisticTargetKind,
    max_iterations: usize,
    gradient_tolerance: f64,
    damping: f64,
    max_backtracking_steps: usize,
}

impl TryFrom<WeightedLogisticRecord> for WeightedLogistic {
    type Error = Error;

    fn try_from(record: WeightedLogisticRecord) -> Result<Self> {
        let logistic = Self {
            penalty: record.penalty,
            target_kind: record.target_kind,
            max_iterations: record.max_iterations,
            gradient_tolerance: record.gradient_tolerance,
            damping: record.damping,
            max_backtracking_steps: record.max_backtracking_steps,
        };
        logistic.validate()?;
        Ok(logistic)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogisticDiagnostics {
    pub target_kind: LogisticTargetKind,
    pub initial_objective: f64,
    pub objective: f64,
    /// Infinity norm in the input feature coordinates.
    pub gradient_norm: f64,
    /// Maximum of the gradient infinity norms scaled by diagonal curvature roots
    /// and by weighted mean absolute features plus penalty roots.
    pub scaled_gradient_norm: f64,
    pub iterations: usize,
    /// Number of trial steps rejected by backtracking.
    pub backtracking_steps: usize,
}

impl Regressor for WeightedLogistic {
    type Model = LogisticModel;
    type Diagnostics = LogisticDiagnostics;

    fn validate_label_source(&self, source: LabelSource) -> Result<()> {
        ensure(
            self.target_kind == LogisticTargetKind::SoftLabels
                || source == LabelSource::ObservedBinary,
            "factual binary logistic targets require factual event observations",
        )
    }

    fn fit(
        &self,
        features: &DMatrix<f64>,
        targets: &[f64],
        weights: &[f64],
    ) -> Result<(Self::Model, Self::Diagnostics)> {
        let problem = LogisticProblem::new(self, features, targets, weights)?;
        let mut coefficients = DVector::zeros(features.ncols());
        let initial_objective = problem.objective(&coefficients)?;
        let mut objective = initial_objective;
        let mut backtracking_steps: usize = 0;

        for iterations in 0..=self.max_iterations {
            let (gradient, hessian) = problem.derivatives(&coefficients)?;
            let scales = hessian.diagonal().map(|curvature| {
                if curvature > 0.0 {
                    curvature.sqrt()
                } else {
                    1.0
                }
            });
            let scaled_gradient = gradient.component_div(&scales);
            // The second moment can be dominated by rare high-leverage rows.
            let first_moment_gradient_norm = gradient
                .iter()
                .zip(&problem.first_moment_scales)
                .map(|(gradient, scale)| (gradient / scale).abs())
                .fold(0.0, f64::max);
            let scaled_gradient_norm = scaled_gradient.amax().max(first_moment_gradient_norm);
            if !scaled_gradient_norm.is_finite() {
                return Err(Error::Numerical(
                    "logistic scaled gradient must be finite".into(),
                ));
            }
            if scaled_gradient_norm <= self.gradient_tolerance {
                let input_coefficients = coefficients.component_div(&problem.column_scales);
                if coefficients
                    .iter()
                    .zip(&input_coefficients)
                    .any(|(&scaled, &input)| !input.is_finite() || (input == 0.0 && scaled != 0.0))
                {
                    return Err(Error::Numerical(
                        "fitted logistic coefficients must be representable in the input feature units".into(),
                    ));
                }
                let gradient_norm = gradient.component_mul(&problem.column_scales).amax();
                if !gradient_norm.is_finite() {
                    return Err(Error::Numerical(
                        "logistic gradient in the input units must be finite".into(),
                    ));
                }
                let model =
                    LogisticModel::new(input_coefficients.as_slice().to_vec(), self.target_kind)?;
                return Ok((
                    model,
                    LogisticDiagnostics {
                        target_kind: self.target_kind,
                        initial_objective,
                        objective,
                        gradient_norm,
                        scaled_gradient_norm,
                        iterations,
                        backtracking_steps,
                    },
                ));
            }
            if iterations == self.max_iterations {
                break;
            }

            let (direction, slope) =
                newton_direction(&scaled_gradient, &scales, hessian, self.damping)?;

            let mut step = 1.0;
            let mut accepted = None;
            for rejected in 0..self.max_backtracking_steps {
                let candidate = &coefficients + step * &direction;
                if candidate == coefficients {
                    break;
                }
                if let Ok(candidate_objective) = problem.objective(&candidate)
                    && candidate_objective <= objective + (1e-4 * step) * slope
                {
                    accepted = Some((candidate, candidate_objective, rejected));
                    break;
                }
                step *= 0.5;
            }
            let (candidate, candidate_objective, rejected) = accepted.ok_or_else(|| {
                Error::Numerical("logistic line search could not find a decreasing step".into())
            })?;
            coefficients = candidate;
            objective = candidate_objective;
            backtracking_steps = backtracking_steps.saturating_add(rejected);
        }
        Err(Error::Numerical(format!(
            "logistic regression did not converge within {} iterations",
            self.max_iterations,
        )))
    }
}

struct LogisticProblem<'a> {
    features: DMatrix<f64>,
    targets: &'a [f64],
    weights: Vec<f64>,
    penalty_roots: DVector<f64>,
    column_scales: DVector<f64>,
    first_moment_scales: DVector<f64>,
}

impl<'a> LogisticProblem<'a> {
    fn new(
        config: &WeightedLogistic,
        features: &DMatrix<f64>,
        targets: &'a [f64],
        weights: &[f64],
    ) -> Result<Self> {
        config.validate()?;
        validate_design(features, weights)?;
        validate_targets(features.nrows(), targets)?;
        match config.target_kind {
            LogisticTargetKind::FactualBinary => ensure(
                targets.iter().all(|&target| target == 0.0 || target == 1.0),
                "factual binary logistic targets must be exactly 0 or 1",
            )?,
            LogisticTargetKind::SoftLabels => ensure(
                targets.iter().all(|target| (0.0..=1.0).contains(target)),
                "soft logistic targets must be in [0, 1]",
            )?,
        }
        let (rows, columns) = features.shape();
        check_matrix_size(columns, columns)?;
        let weights = normalized_weights(weights)?;
        let penalty_roots = DVector::from_iterator(
            columns,
            config.penalty.diagonal(columns)?.into_iter().map(f64::sqrt),
        );
        // Bound both the scaled design entries and the transformed penalty roots by one.
        let column_scales = DVector::from_iterator(
            columns,
            features
                .column_iter()
                .zip(&penalty_roots)
                .map(|(column, &root)| {
                    let maximum =
                        column
                            .iter()
                            .zip(&weights)
                            .fold(root, |maximum, (value, &weight)| {
                                if weight > 0.0 {
                                    maximum.max(value.abs())
                                } else {
                                    maximum
                                }
                            });
                    if maximum > 0.0 { maximum } else { 1.0 }
                }),
        );
        let features = DMatrix::from_fn(rows, columns, |row, column| {
            if weights[row] == 0.0 {
                0.0
            } else {
                features[(row, column)] / column_scales[column]
            }
        });
        let penalty_roots = penalty_roots.component_div(&column_scales);
        let first_moment_scales = DVector::from_iterator(
            columns,
            features
                .column_iter()
                .zip(&penalty_roots)
                .map(|(column, root)| {
                    let mean_abs = column
                        .iter()
                        .zip(&weights)
                        .map(|(value, weight)| value.abs() * weight)
                        .sum::<f64>();
                    let scale = mean_abs + root;
                    if scale > 0.0 { scale } else { 1.0 }
                }),
        );
        Ok(Self {
            features,
            targets,
            weights,
            penalty_roots,
            column_scales,
            first_moment_scales,
        })
    }

    fn objective(&self, coefficients: &DVector<f64>) -> Result<f64> {
        let mut objective = 0.0;
        for ((row, &target), &weight) in self
            .features
            .row_iter()
            .zip(self.targets)
            .zip(&self.weights)
        {
            if weight > 0.0 {
                let logit = finite_dot(row.iter(), coefficients.iter())?;
                let (loss, _, _) = bernoulli_terms(logit, target);
                objective += weight * loss;
            }
        }
        for (&coefficient, &root) in coefficients.iter().zip(&self.penalty_roots) {
            let penalized = root * coefficient;
            objective += (0.5 * penalized) * penalized;
        }
        if !objective.is_finite() {
            return Err(Error::Numerical("logistic objective must be finite".into()));
        }
        Ok(objective)
    }

    fn derivatives(&self, coefficients: &DVector<f64>) -> Result<(DVector<f64>, DMatrix<f64>)> {
        let dimension = self.features.ncols();
        let mut gradient = DVector::<f64>::zeros(dimension);
        let mut hessian = DMatrix::<f64>::zeros(dimension, dimension);
        let mut scaled_row = vec![0.0; dimension];
        for ((row, &target), &weight) in self
            .features
            .row_iter()
            .zip(self.targets)
            .zip(&self.weights)
        {
            if weight == 0.0 {
                continue;
            }
            let logit = finite_dot(row.iter(), coefficients.iter())?;
            let (_, residual, curvature) = bernoulli_terms(logit, target);
            let weighted_residual = weight * residual;
            let curvature_root = weight.sqrt() * curvature.sqrt();
            for (column, &value) in row.iter().enumerate() {
                gradient[column] += weighted_residual * value;
                scaled_row[column] = curvature_root * value;
            }
            for column in 0..dimension {
                for row in column..dimension {
                    hessian[(row, column)] += scaled_row[row] * scaled_row[column];
                }
            }
        }
        for (column, &root) in self.penalty_roots.iter().enumerate() {
            gradient[column] += root * (root * coefficients[column]);
            hessian[(column, column)] += root * root;
            for row in 0..column {
                hessian[(row, column)] = hessian[(column, row)];
            }
        }
        if gradient
            .iter()
            .chain(hessian.iter())
            .any(|value| !value.is_finite())
        {
            return Err(Error::Numerical(
                "logistic gradient and curvature must be finite".into(),
            ));
        }
        Ok((gradient, hessian))
    }
}

fn bernoulli_terms(logit: f64, target: f64) -> (f64, f64, f64) {
    let exponential = (-logit.abs()).exp();
    let denominator = 1.0 + exponential;
    let tail = exponential / denominator;
    let (linear, residual) = if logit >= 0.0 {
        ((1.0 - target) * logit, (1.0 - target) - tail)
    } else {
        (-target * logit, tail - target)
    };
    (linear + exponential.ln_1p(), residual, tail / denominator)
}

fn newton_direction(
    scaled_gradient: &DVector<f64>,
    scales: &DVector<f64>,
    mut hessian: DMatrix<f64>,
    damping: f64,
) -> Result<(DVector<f64>, f64)> {
    for column in 0..hessian.ncols() {
        for row in 0..hessian.nrows() {
            hessian[(row, column)] = (hessian[(row, column)] / scales[row]) / scales[column];
        }
        hessian[(column, column)] += damping;
    }
    let factor = hessian.cholesky().ok_or_else(|| {
        Error::Numerical("damped logistic Newton system could not be factored".into())
    })?;
    let scaled_direction = factor.solve(&(-scaled_gradient));
    let slope = scaled_gradient.dot(&scaled_direction);
    let direction = scaled_direction.component_div(scales);
    if !slope.is_finite() || slope >= 0.0 || direction.iter().any(|value| !value.is_finite()) {
        return Err(Error::Numerical(
            "logistic Newton step must be a finite descent direction".into(),
        ));
    }
    Ok((direction, slope))
}

#[cfg(test)]
mod tests;
