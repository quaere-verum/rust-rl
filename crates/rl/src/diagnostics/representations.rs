//! Fixed-input representation comparisons in f64. No neural training occurs here.
//!
//! Linear CKA uses centered Gram matrices (the biased finite-sample estimator),
//! following <https://arxiv.org/abs/1905.00414>. It is not neuron correspondence
//! or causal attribution. PCA projections must share the same fitted basis.
use super::schema::*;
use crate::{Error, Result, core::ensure};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::Serialize;
use std::collections::BTreeMap;

/// Borrowed aligned subtraction; no dense difference matrix is allocated.
/// Maps run from current row/column to the corresponding reference coordinate.
pub struct ActivationDifference<'a> {
    current: &'a Array,
    reference: &'a Array,
    rows: &'a [usize],
    columns: &'a [usize],
}
fn activation_value(array: &Array, index: usize) -> f64 {
    match &array.values {
        Values::F32(v) => v[index] as f64,
        Values::F64(v) => v[index],
        _ => unreachable!("validated floating activation"),
    }
}
#[derive(Debug, Clone)]
pub struct ActivationDrift {
    pub difference_norm: f64,
    pub reference_norm: f64,
    pub relative: f64,
    pub near_zero_reference: bool,
    /// Current recorded row order, over every matched neuron.
    pub state_norms: Vec<f64>,
}
impl<'a> ActivationDifference<'a> {
    pub fn new(
        current: &'a Array,
        reference: &'a Array,
        rows: &'a [usize],
        columns: &'a [usize],
    ) -> Result<Self> {
        ensure(
            current.shape().len() == 2
                && reference.shape().len() == 2
                && current.shape() == reference.shape()
                && current.shape()[0] == rows.len()
                && current.shape()[1] == columns.len()
                && !rows.is_empty()
                && !columns.is_empty()
                && [current, reference]
                    .iter()
                    .all(|a| matches!(a.dtype(), DType::F32 | DType::F64)),
            "activation differences require equal nonempty floating dimensions and complete correspondence",
        )?;
        for map in [rows, columns] {
            let mut seen = vec![false; map.len()];
            for &i in map {
                ensure(
                    i < map.len() && !seen[i],
                    "invalid activation coordinate permutation",
                )?;
                seen[i] = true;
            }
        }
        Ok(Self {
            current,
            reference,
            rows,
            columns,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows.len()
    }
    pub fn columns(&self) -> usize {
        self.columns.len()
    }
    pub fn value(&self, row: usize, column: usize) -> f64 {
        activation_value(self.current, row * self.columns.len() + column)
            - activation_value(
                self.reference,
                self.rows[row] * self.columns.len() + self.columns[column],
            )
    }
    /// Full population norms with O(rows) output and O(1) arithmetic workspace.
    pub fn drift(&self) -> Result<ActivationDrift> {
        self.drift_checked(|| Ok(()))
    }
    /// Check cancellation between bounded batches without retaining dense output.
    pub fn drift_checked(
        &self,
        mut checkpoint: impl FnMut() -> Result<()>,
    ) -> Result<ActivationDrift> {
        let mut reference_norm: f64 = 0.;
        let mut difference_norm: f64 = 0.;
        let mut state_norms = Vec::with_capacity(self.rows());
        for row in 0..self.rows() {
            if row % 128 == 0 {
                checkpoint()?;
            }
            let mut norm: f64 = 0.;
            for column in 0..self.columns() {
                let reference = activation_value(
                    self.reference,
                    self.rows[row] * self.columns() + self.columns[column],
                );
                let difference = self.value(row, column);
                ensure(
                    reference.is_finite() && difference.is_finite(),
                    "nonfinite activation difference",
                )?;
                reference_norm = reference_norm.hypot(reference);
                norm = norm.hypot(difference);
            }
            difference_norm = difference_norm.hypot(norm);
            state_norms.push(norm);
        }
        let relative = difference_norm / (reference_norm + 1e-12);
        ensure(relative.is_finite(), "activation drift overflow")?;
        Ok(ActivationDrift {
            difference_norm,
            reference_norm,
            relative,
            near_zero_reference: reference_norm <= 1e-12,
            state_norms,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RepresentationLimits {
    pub max_samples: usize,
    pub max_features: usize,
    /// Conservative bound for simultaneously allocated scalar workspaces.
    pub max_work_elements: usize,
}
impl Default for RepresentationLimits {
    fn default() -> Self {
        Self {
            max_samples: 1024,
            max_features: 16384,
            max_work_elements: 8 * 1024 * 1024,
        }
    }
}
impl RepresentationLimits {
    fn check(&self, rows: usize, cols: usize, work: Option<usize>) -> Result<()> {
        ensure(
            rows >= 2
                && cols > 0
                && rows <= self.max_samples
                && cols <= self.max_features
                && work.is_some_and(|n| n <= self.max_work_elements),
            "representation analysis exceeds dimensions/workspace limits; record fewer states/features or raise the explicit limit",
        )
    }
}
fn centered(values: &[f64], rows: usize, cols: usize) -> Result<(DMatrix<f64>, Vec<f64>)> {
    ensure(
        rows.checked_mul(cols) == Some(values.len())
            && rows >= 2
            && cols > 0
            && values.iter().all(|v| v.is_finite()),
        "invalid/nonfinite representation matrix",
    )?;
    let mut means = vec![0.; cols];
    for (i, row) in values.chunks_exact(cols).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            means[c] += (v - means[c]) / (i + 1) as f64;
        }
    }
    let x = DMatrix::from_fn(rows, cols, |r, c| values[r * cols + c] - means[c]);
    ensure(
        x.iter().all(|v| v.is_finite()),
        "representation centering overflow",
    )?;
    Ok((x, means))
}
#[derive(Debug, Clone, Serialize)]
pub struct Pca {
    mean: Vec<f64>,
    axes: Vec<Vec<f64>>,
    variance: Vec<f64>,
    total_variance: f64,
}
impl Pca {
    /// Fit once on a reference checkpoint. No per-feature standardization.
    /// For wide matrices, diagonalizes XXᵀ rather than allocating a feature²
    /// covariance. Eigenvector signs use a deterministic largest-loading rule.
    pub fn fit(
        values: &[f64],
        rows: usize,
        cols: usize,
        components: usize,
        limits: RepresentationLimits,
    ) -> Result<Self> {
        let small = rows.min(cols);
        let work = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| {
                small
                    .checked_mul(small)
                    .and_then(|s| s.checked_mul(4))
                    .and_then(|s| n.checked_add(s))
            });
        limits.check(rows, cols, work)?;
        ensure(
            components > 0 && components <= 32,
            "PCA supports 1..32 requested axes",
        )?;
        let (x, mean) = centered(values, rows, cols)?;
        let gram = if cols <= rows {
            x.transpose() * &x
        } else {
            &x * x.transpose()
        };
        ensure(
            gram.iter().all(|v| v.is_finite()),
            "PCA covariance overflow",
        )?;
        let eig = SymmetricEigen::try_new(gram, 1e-12, 100_000).ok_or_else(|| {
            Error::State("PCA eigensolver did not converge within iteration limit".into())
        })?;
        let mut order: Vec<_> = (0..small).collect();
        order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
        let total = eig.eigenvalues.iter().map(|v| v.max(0.)).sum::<f64>();
        ensure(
            total.is_finite() && total > 0.,
            "PCA undefined for a constant representation",
        )?;
        let mut axes = vec![];
        let mut variance = vec![];
        for &i in order.iter().take(components.min(rows - 1).min(cols)) {
            let lambda = eig.eigenvalues[i].max(0.);
            let mut axis = if lambda <= total * 1e-12 {
                vec![0.; cols]
            } else if cols <= rows {
                eig.eigenvectors.column(i).iter().copied().collect()
            } else {
                (x.transpose() * eig.eigenvectors.column(i) / lambda.sqrt())
                    .iter()
                    .copied()
                    .collect()
            };
            let pivot = axis
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .map_or(0, |(i, _)| i);
            if axis[pivot] < 0. {
                for v in &mut axis {
                    *v = -*v;
                }
            }
            axes.push(axis);
            variance.push(lambda / (rows - 1) as f64);
        }
        Ok(Self {
            mean,
            axes,
            variance,
            total_variance: total / (rows - 1) as f64,
        })
    }
    pub fn dimensions(&self) -> usize {
        self.mean.len()
    }
    pub fn components(&self) -> usize {
        self.axes.len()
    }
    pub fn explained_variance(&self) -> &[f64] {
        &self.variance
    }
    pub fn explained_fraction(&self) -> Vec<f64> {
        self.variance
            .iter()
            .map(|v| v / self.total_variance)
            .collect()
    }
    /// Apply the reference mean and axes unchanged, including to other checkpoints.
    pub fn project(&self, values: &[f64], rows: usize) -> Result<Vec<f64>> {
        ensure(
            self.dimensions() > 0
                && self.components() > 0
                && self.axes.iter().all(|a| a.len() == self.dimensions())
                && rows.checked_mul(self.dimensions()) == Some(values.len())
                && values.iter().all(|v| v.is_finite()),
            "invalid PCA projection dimensions or data",
        )?;
        let mut out = Vec::with_capacity(rows * self.components());
        for row in values.chunks_exact(self.dimensions()) {
            for axis in &self.axes {
                out.push(
                    row.iter()
                        .zip(&self.mean)
                        .zip(axis)
                        .map(|((&v, &m), &w)| (v - m) * w)
                        .sum::<f64>(),
                );
            }
        }
        ensure(out.iter().all(|v| v.is_finite()), "PCA projection overflow")?;
        Ok(out)
    }
}
/// Centered linear CKA in `[0,1]`. Both matrices contain the same ordered samples;
/// feature counts may differ. Constant representations return an explicit error.
pub fn linear_cka(
    x: &[f64],
    x_features: usize,
    y: &[f64],
    y_features: usize,
    rows: usize,
    limits: RepresentationLimits,
) -> Result<f64> {
    let work = x_features
        .checked_add(y_features)
        .and_then(|p| rows.checked_mul(p))
        .and_then(|np| np.checked_mul(2))
        .and_then(|np| {
            rows.checked_mul(rows)
                .and_then(|n| n.checked_mul(3))
                .and_then(|n| n.checked_add(np))
        });
    limits.check(rows, x_features.max(y_features), work)?;
    let (mut x, _) = centered(x, rows, x_features)?;
    let (mut y, _) = centered(y, rows, y_features)?;
    let sx = x.amax();
    let sy = y.amax();
    ensure(
        sx > 0. && sy > 0.,
        "CKA undefined for a constant representation",
    )?;
    x /= sx;
    y /= sy;
    let xx = &x * x.transpose();
    let yy = &y * y.transpose();
    let numerator = xx.component_mul(&yy).sum();
    let denominator = xx.norm() * yy.norm();
    ensure(
        denominator > 0. && denominator.is_finite() && numerator.is_finite(),
        "CKA numerical degeneracy",
    )?;
    let score = numerator / denominator;
    ensure(
        (-1e-12..=1. + 1e-12).contains(&score),
        "CKA numerical range error",
    )?;
    Ok(score.clamp(0., 1.))
}
/// Exact typed row equality, including trailing history dimensions. No integer
/// or floating conversion is performed; invalid row indices never match.
pub fn equal_row(a: &Array, b: &Array, left: usize, right: usize) -> bool {
    if a.shape()[1..] != b.shape()[1..]
        || a.dtype() != b.dtype()
        || left >= a.shape()[0]
        || right >= b.shape()[0]
    {
        return false;
    }
    let width = a.len() / a.shape()[0];
    let l = left * width..(left + 1) * width;
    let r = right * width..(right + 1) * width;
    match (&a.values, &b.values) {
        (Values::F32(a), Values::F32(b)) => a.get(l) == b.get(r),
        (Values::F64(a), Values::F64(b)) => a.get(l) == b.get(r),
        (Values::U64(a), Values::U64(b)) => a.get(l) == b.get(r),
        (Values::U8(a), Values::U8(b)) => a.get(l) == b.get(r),
        _ => false,
    }
}
/// Map left rows to right rows using stable IDs, then verify observations,
/// masks, fixed actions and all imported history channels. IDs alone are not
/// evidence of matching inputs. Changing visited populations is rejected.
pub fn match_reference_sets(left: &Chunk, right: &Chunk) -> Result<Vec<usize>> {
    left.validate()?;
    right.validate()?;
    let (Record::ProbeSet(a), Record::ProbeSet(b)) = (&left.record, &right.record) else {
        return Err(Error::Config(
            "comparison requires probe-set records".into(),
        ));
    };
    ensure(
        a.population == b.population
            && a.population != ProbePopulation::PolicyVisited
            && a.methodology == b.methodology,
        "representation comparisons require the same fixed/imported probe methodology",
    )?;
    let aid = left.arrays[&a.row_ids].as_u64()?;
    let bid = right.arrays[&b.row_ids].as_u64()?;
    let map: BTreeMap<_, _> = bid.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    ensure(
        !aid.is_empty()
            && aid.len() == bid.len()
            && map.len() == bid.len()
            && aid.iter().collect::<std::collections::BTreeSet<_>>().len() == aid.len(),
        "missing/duplicate probe row identities",
    )?;
    let mut keys = vec![(&a.observations, &b.observations)];
    for (a, b) in [
        (&a.action_masks, &b.action_masks),
        (&a.fixed_actions, &b.fixed_actions),
    ] {
        match (a, b) {
            (Some(a), Some(b)) => keys.push((a, b)),
            (None, None) => {}
            _ => return Err(Error::Config("probe action/mask conditions differ".into())),
        }
    }
    ensure(
        a.histories.keys().eq(b.histories.keys()),
        "probe history channels differ",
    )?;
    keys.extend(
        a.histories
            .iter()
            .map(|(name, key)| (key, &b.histories[name])),
    );
    ensure(
        keys.iter().all(|(a, b)| {
            left.arrays[*a].shape()[0] == aid.len() && right.arrays[*b].shape()[0] == bid.len()
        }),
        "probe arrays and row counts differ",
    )?;
    aid.iter()
        .enumerate()
        .map(|(i, id)| {
            let j = *map
                .get(id)
                .ok_or_else(|| Error::Config("probe row identity missing in comparison".into()))?;
            ensure(
                keys.iter()
                    .all(|(a, b)| equal_row(&left.arrays[*a], &right.arrays[*b], i, j)),
                "probe input/action/mask/history mismatch",
            )?;
            Ok(j)
        })
        .collect()
}
/// KL(P || Q), including the mathematically infinite case where Q gives zero
/// mass to an event with positive P mass. Invalid probabilities are rejected.
pub fn categorical_kl(p: &[f64], q: &[f64]) -> Result<f64> {
    ensure(
        !p.is_empty()
            && p.len() == q.len()
            && [p, q].iter().all(|v| {
                v.iter().all(|x| x.is_finite() && *x >= 0.)
                    && (v.iter().sum::<f64>() - 1.).abs() < 1e-6
            }),
        "invalid categorical probabilities",
    )?;
    Ok(p.iter()
        .zip(q)
        .map(|(&p, &q)| {
            if p == 0. {
                0.
            } else if q == 0. {
                f64::INFINITY
            } else {
                p * (p.ln() - q.ln())
            }
        })
        .sum::<f64>()
        .max(0.))
}
/// KL of diagonal normals P || Q. This is also the action-distribution KL under
/// the same invertible tanh/affine transform; different bounds must be rejected.
pub fn diagonal_gaussian_kl(pm: &[f64], ps: &[f64], qm: &[f64], qs: &[f64]) -> Result<f64> {
    ensure(
        !pm.is_empty()
            && [ps, qm, qs].iter().all(|x| x.len() == pm.len())
            && [pm, ps, qm, qs]
                .iter()
                .all(|x| x.iter().all(|v| v.is_finite())),
        "invalid Gaussian parameters",
    )?;
    let value = (0..pm.len())
        .map(|i| {
            let delta = pm[i] - qm[i];
            qs[i] - ps[i]
                + 0.5
                    * ((2. * (ps[i] - qs[i])).exp()
                        + if delta == 0. {
                            0.
                        } else {
                            (2. * delta.abs().ln() - 2. * qs[i]).exp()
                        }
                        - 1.)
        })
        .sum::<f64>();
    ensure(
        !value.is_nan(),
        "Gaussian KL indeterminate at numeric range limit",
    )?;
    Ok(value.max(0.))
}
