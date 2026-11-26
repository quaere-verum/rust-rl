//! Geometric preparation for the Network workspace.
//!
//! All numerical work is explicit, bounded and independent of training. The
//! caller supplies the exact historical model and a fixed, labelled reference
//! population. We never fit an auxiliary prediction head.
pub mod finite;
pub mod output;
pub mod representations;
pub mod selection;
pub mod work;
pub use finite::{LayerComparison, Neighborhood};
#[cfg(test)]
#[path = "network/finite_original.rs"]
#[allow(dead_code)] // Frozen oracle retains the original complete implementation.
mod finite_original;
#[cfg(test)]
#[path = "network/finite_tests.rs"]
mod finite_tests;

use nalgebra::{DMatrix, DVector};
use rl::{
    Error, Result,
    diagnostics::{
        network_geometry::NumericalMlp,
        representations::{Pca, RepresentationLimits},
    },
};
use std::sync::Arc;

pub const MAX_PROBES: usize = 256;
pub const MAX_GRID: usize = 256;
pub const MAX_PATH: usize = 64;
const COV_RIDGE: f64 = 1e-3;
// The existing representation workspace allowance also bounds retained reuse.
// Larger valid requests bypass this cache; the bound never changes the metric.
const REFERENCE_CACHE_BYTES: usize = 8 * 1024 * 1024 * 8;

fn current(check: &impl Fn() -> bool) -> Result<()> {
    if check() {
        Ok(())
    } else {
        Err(Error::State("obsolete Network preparation".into()))
    }
}

enum Metric {
    Identity,
    Dense(DMatrix<f64>),
    LowRank {
        centered: DMatrix<f64>,
        inverse: DMatrix<f64>,
        ridge: f64,
    },
}

// Only one neighborhood owns these buffers. All fitted LowRank metrics in a
// request use the same probe count; no backing allocation survives the call.
struct NeighborhoodMetricScratch {
    max_width: usize,
    population: usize,
    buffers: Option<NeighborhoodMetricBuffers>,
}
struct NeighborhoodMetricBuffers {
    input_residual: DVector<f64>,
    projected: DVector<f64>,
    dual: DVector<f64>,
}
impl NeighborhoodMetricScratch {
    fn new(metrics: &[Metric]) -> Self {
        let mut max_width = 0;
        let mut population = 0;
        for metric in metrics {
            if let Metric::LowRank { centered, .. } = metric {
                debug_assert!(population == 0 || population == centered.nrows());
                population = centered.nrows();
                max_width = max_width.max(centered.ncols());
            }
        }
        Self {
            max_width,
            population,
            buffers: None,
        }
    }
    fn norm_squared(&mut self, metric: &Metric, values: &[f64]) -> f64 {
        let Metric::LowRank {
            centered,
            inverse,
            ridge,
        } = metric
        else {
            return metric.inner(values, values);
        };
        let buffers = self.buffers.get_or_insert_with(|| {
            let mut input = Vec::with_capacity(self.max_width);
            input.resize(values.len(), 0.);
            NeighborhoodMetricBuffers {
                input_residual: DVector::from_vec(input),
                projected: DVector::zeros(self.population),
                dual: DVector::zeros(self.population),
            }
        });
        // Consuming storage moves the same Vec through an exact-shape wrapper.
        // Vec::resize keeps capacity, including narrow-wide-narrow transitions.
        if buffers.input_residual.len() != values.len() {
            let old = std::mem::replace(&mut buffers.input_residual, DVector::zeros(0));
            let mut data: Vec<f64> = old.data.into();
            data.resize(values.len(), 0.);
            buffers.input_residual = DVector::from_vec(data);
        }
        buffers
            .input_residual
            .as_mut_slice()
            .copy_from_slice(values);
        centered.mul_to(&buffers.input_residual, &mut buffers.projected);
        inverse.mul_to(&buffers.projected, &mut buffers.dual);
        centered.tr_mul_to(&buffers.dual, &mut buffers.input_residual);
        // The copied input is no longer needed by either multiplication. Keep
        // the original minuends and the original ascending subtraction order.
        for (residual, value) in buffers.input_residual.iter_mut().zip(values) {
            *residual = value - *residual;
        }
        buffers.input_residual.norm_squared() / ridge + buffers.dual.norm_squared()
    }
}
fn low_rank_coordinates(
    centered: &DMatrix<f64>,
    inverse: &DMatrix<f64>,
    values: &[f64],
) -> (DVector<f64>, DVector<f64>) {
    let values = DVector::from_column_slice(values);
    let dual = inverse * (centered * &values);
    let residual = values - centered.tr_mul(&dual);
    (residual, dual)
}
impl Metric {
    fn fit<'a>(
        samples: impl ExactSizeIterator<Item = &'a [f64]> + Clone,
        d: usize,
        check: &impl Fn() -> bool,
    ) -> Result<Self> {
        current(check)?;
        let count = samples.len();
        if count == 0 || samples.clone().any(|row| row.len() != d) {
            return Err(Error::Shape(
                "inconsistent geometry reference population".into(),
            ));
        }
        if count < 3 {
            return Ok(Self::Identity);
        }
        // Keep the dense path when it is the smaller representation; a layer
        // only slightly wider than its population need not pay for two arrays.
        if d.saturating_mul(d) <= count.saturating_mul(d.saturating_add(count)) {
            return covariance_inverse(samples, d).map(Self::Dense);
        }
        let mut mean = vec![0.; d];
        for row in samples.clone() {
            current(check)?;
            for (a, value) in mean.iter_mut().zip(row) {
                *a += *value / count as f64;
            }
        }
        let scale = ((count - 1) as f64).sqrt();
        let mut centered = DMatrix::zeros(count, d);
        let mut trace = 0.;
        for (i, row) in samples.enumerate() {
            current(check)?;
            for j in 0..d {
                let difference = row[j] - mean[j];
                centered[(i, j)] = difference / scale;
                trace += difference * difference / (count - 1) as f64;
            }
        }
        let ridge = COV_RIDGE * (trace / d as f64).max(1e-12);
        let mut small = DMatrix::zeros(count, count);
        for i in 0..count {
            current(check)?;
            for j in 0..=i {
                let mut value = 0.;
                for k in 0..d {
                    value += centered[(i, k)] * centered[(j, k)];
                }
                small[(i, j)] = value;
                small[(j, i)] = value;
            }
            small[(i, i)] += ridge;
        }
        if !ridge.is_finite() || small.iter().any(|v| !v.is_finite()) {
            return Err(Error::Numerical("regularized covariance overflow".into()));
        }
        current(check)?;
        let inverse = small
            .try_inverse()
            .ok_or_else(|| Error::Numerical("regularized covariance inversion failed".into()))?;
        Ok(Self::LowRank {
            centered,
            inverse,
            ridge,
        })
    }
    fn inner(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Self::Identity => a.iter().zip(b).map(|(a, b)| a * b).sum(),
            Self::Dense(inverse) => inner(inverse, a, b),
            Self::LowRank {
                centered,
                inverse,
                ridge,
            } => {
                // Stable equivalent of Woodbury: residual/ridge and dual terms
                // avoid subtracting large nearly equal quadratic forms.
                let (ra, da) = low_rank_coordinates(centered, inverse, a);
                if std::ptr::eq(a, b) {
                    return ra.norm_squared() / ridge + da.norm_squared();
                }
                let (rb, db) = low_rank_coordinates(centered, inverse, b);
                ra.dot(&rb) / ridge + da.dot(&db)
            }
        }
    }
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Dense(m) => m.len() * 8,
            Self::LowRank {
                centered, inverse, ..
            } => (centered.len() + inverse.len()) * 8,
        }
    }
}

struct Reference {
    layers: Vec<Vec<Vec<f64>>>,
    metrics: Vec<Metric>,
}
impl Reference {
    fn build(
        net: &NumericalMlp,
        request: &Request,
        dense: bool,
        check: &impl Fn() -> bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(request.probes.len());
        for row in &request.probes {
            current(check)?;
            layers.push(net.forward_layers(row)?);
        }
        let mut metrics = Vec::with_capacity(net.layer_count() + 1);
        for l in 0..=net.layer_count() {
            current(check)?;
            let rows = layers.iter().map(|layers| layers[l].as_slice());
            metrics.push(if dense {
                Metric::Dense(covariance_inverse(rows, layers[0][l].len())?)
            } else {
                Metric::fit(rows, layers[0][l].len(), check)?
            });
        }
        Ok(Self { layers, metrics })
    }
    fn matches(&self, probes: &[Vec<f64>]) -> bool {
        self.layers.len() == probes.len()
            && self.layers.iter().zip(probes).all(|(layers, row)| {
                layers[0].len() == row.len()
                    && layers[0]
                        .iter()
                        .zip(row)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            })
    }
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<usize>()
            + self.layers.capacity() * std::mem::size_of::<Vec<Vec<f64>>>()
            + self
                .layers
                .iter()
                .map(|layers| {
                    layers.capacity() * std::mem::size_of::<Vec<f64>>()
                        + layers.iter().map(|row| row.capacity() * 8).sum::<usize>()
                })
                .sum::<usize>()
            + self.metrics.capacity() * std::mem::size_of::<Metric>()
            + self
                .metrics
                .iter()
                .map(Metric::retained_bytes)
                .sum::<usize>()
    }
}

/// One immutable numerical model and at most one exact reference population.
/// Fixed PCA bases remain caller-owned and are never replaced by this cache.
///
/// For N probes and layer widths D_l, reference scalar storage is bounded by
/// 8 * (N * sum(D_l) + sum(M_l)) bytes, where M_l is zero for N < 3 and
/// min(D_l², N * D_l + N²) otherwise. Vector/enum headers are
/// included in the actual retention charge. A replacement drops the previous
/// reference before building. Reference and neighborhood cache retention share
/// the 64 MiB cap; a request over that allowance remains supported without reuse.
///
/// Solver scratch is linear in the widest layer plus O(N²), with N <= 256;
/// no wide-layer D_l² matrix is allocated. Geometry additionally owns at most
/// 33 path rows, at most six anchor directions, two grid directions at one location,
/// and the separately bounded PCA workspace. Cache bypass may temporarily own
/// a larger reference to preserve support for previously valid requests.
/// The finite map retains at most 65 × 256 optional scalar errors, 256 input
/// positions and one exact anchor key. Point activation/residual capacities are
/// reused within a neighborhood request and released when that call ends;
/// all layers share each single complete forward.
pub struct Preparation {
    network: Arc<NumericalMlp>,
    reference: Option<Arc<Reference>>,
    neighborhood: Option<Arc<Neighborhood>>,
    cache_budget: usize,
}
impl Preparation {
    pub fn new(network: NumericalMlp) -> Self {
        Self::with_cache_budget(network, REFERENCE_CACHE_BYTES)
    }
    /// A smaller cache budget changes retention only; all valid geometries are
    /// still computed through the same bounded-population low-rank algorithm.
    pub fn with_cache_budget(network: NumericalMlp, bytes: usize) -> Self {
        Self {
            network: Arc::new(network),
            reference: None,
            neighborhood: None,
            cache_budget: bytes.min(REFERENCE_CACHE_BYTES),
        }
    }
    pub fn retained_bytes(&self) -> usize {
        self.network.retained_bytes()
            + self.reference.as_ref().map_or(0, |r| r.retained_bytes())
            + self.neighborhood.as_ref().map_or(0, |n| n.retained_bytes())
    }
    pub fn prepare(&mut self, request: &Request) -> Result<Frame> {
        self.prepare_cancellable(request, &|| true)
    }
    fn prepare_cancellable(
        &mut self,
        request: &Request,
        check: &impl Fn() -> bool,
    ) -> Result<Frame> {
        current(check)?;
        validate(&self.network, request)?;
        if self
            .reference
            .as_ref()
            .is_some_and(|r| !r.matches(&request.probes))
        {
            self.reference = None;
            self.neighborhood = None;
        }
        let reference = match &self.reference {
            Some(reference) => reference.clone(),
            None => {
                self.neighborhood = None;
                let reference = Arc::new(Reference::build(&self.network, request, false, check)?);
                current(check)?;
                if reference.retained_bytes() <= self.cache_budget {
                    self.reference = Some(reference.clone());
                }
                reference
            }
        };
        if self
            .neighborhood
            .as_ref()
            .is_some_and(|n| !n.matches(request))
        {
            self.neighborhood = None;
        }
        let frame = prepare_geometry(
            &self.network,
            request,
            &reference,
            self.neighborhood.as_ref(),
            check,
        )?;
        current(check)?;
        let reference_bytes = self.reference.as_ref().map_or(0, |r| r.retained_bytes());
        if frame.neighborhood.retained_bytes() <= self.cache_budget.saturating_sub(reference_bytes)
        {
            self.neighborhood = Some(frame.neighborhood.clone());
        }
        Ok(frame)
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub network: String,
    pub anchor: Vec<f64>,
    pub endpoint: Vec<f64>,
    pub probes: Vec<Vec<f64>>,
    pub axes: Vec<usize>,
    pub axis_scales: Vec<f64>,
    pub ranges: [[f64; 2]; 2],
    pub direction_angle: f64,
    pub t: f64,
    /// Half-width of the input neighborhood, in fixed full-axis ranges.
    pub radius: f64,
    /// The recorded A mask, held fixed for the actual policy-output comparison.
    pub action_mask: Option<Vec<bool>>,
    pub layer: usize,
    pub grid: [usize; 2],
    pub previous_basis: Vec<Option<Pca>>,
}
#[derive(Clone, Debug)]
pub struct Glyph {
    pub position: [f64; 2],
    /// Normalized-coordinate pullback metric on the selected input slice.
    pub gram: [[f64; 2]; 2],
    pub strength: f64,
}
#[derive(Clone, Debug)]
pub struct Path {
    pub layer: String,
    pub points: Vec<[f64; 2]>,
    pub variance: f64,
    /// Fraction of path-step squared lengths captured by the projection.
    pub projected_energy: f64,
}
#[derive(Clone, Debug)]
pub struct Frame {
    pub grid: Vec<Glyph>,
    pub metric_3d: Option<[[f64; 3]; 3]>,
    pub comparisons: Vec<LayerComparison>,
    pub neighborhood: Arc<Neighborhood>,
    pub t: f64,
    pub radius: f64,
    /// Raw coordinates on the same two selected input axes as Local geometry.
    pub slice_anchor: [f64; 2],
    pub slice_position: [f64; 2],
    pub slice_endpoint: [f64; 2],
    pub position_on_slice: bool,
    pub output_tangent: Vec<f64>,
    pub output_values: Vec<f64>,
    pub finite_output_values: Vec<f64>,
    pub paths: Vec<Path>,
    pub pca_bases: Vec<Option<Pca>>,
    pub reference_count: usize,
    pub kink_count: usize,
}
impl Frame {
    /// Added finite-comparison ownership, separate from legacy geometry/PCA.
    /// The shared map is conservatively charged in full even when also cached.
    pub fn finite_retained_bytes(&self) -> usize {
        self.comparisons.capacity() * std::mem::size_of::<LayerComparison>()
            + 3 * std::mem::size_of::<[f64; 2]>()
            + std::mem::size_of::<bool>()
            + self
                .comparisons
                .iter()
                .map(|c| c.name.capacity())
                .sum::<usize>()
            + self.neighborhood.retained_bytes()
            + self.finite_output_values.capacity() * 8
    }
}
fn covariance_inverse<'a>(
    samples: impl ExactSizeIterator<Item = &'a [f64]> + Clone,
    d: usize,
) -> Result<DMatrix<f64>> {
    let count = samples.len();
    if count == 0 || samples.clone().any(|row| row.len() != d) {
        return Err(Error::Shape(
            "inconsistent geometry reference population".into(),
        ));
    }
    if count < 3 {
        return Ok(DMatrix::<f64>::identity(d, d));
    }
    let mut mean = vec![0.; d];
    for row in samples.clone() {
        for (a, value) in mean.iter_mut().zip(row) {
            *a += *value / count as f64;
        }
    }
    let mut cov = DMatrix::<f64>::zeros(d, d);
    for row in samples {
        for i in 0..d {
            for j in 0..=i {
                cov[(i, j)] += (row[i] - mean[i]) * (row[j] - mean[j]);
            }
        }
    }
    for i in 0..d {
        for j in 0..=i {
            cov[(i, j)] /= (count - 1) as f64;
            cov[(j, i)] = cov[(i, j)];
        }
    }
    let typical = (cov.trace() / d as f64).max(1e-12);
    for i in 0..d {
        cov[(i, i)] += COV_RIDGE * typical;
    }
    cov.try_inverse()
        .ok_or_else(|| Error::Numerical("regularized covariance inversion failed".into()))
}
fn inner(inv: &DMatrix<f64>, a: &[f64], b: &[f64]) -> f64 {
    let mut result = 0.;
    for i in 0..a.len() {
        let mut projected = 0.;
        for j in 0..b.len() {
            projected += inv[(i, j)] * b[j];
        }
        result += a[i] * projected;
    }
    result
}
fn gram(inv: &Metric, columns: &[Vec<f64>]) -> DMatrix<f64> {
    let n = columns.len();
    if let Metric::LowRank {
        centered,
        inverse,
        ridge,
    } = inv
    {
        let coordinates: Vec<_> = columns
            .iter()
            .map(|column| low_rank_coordinates(centered, inverse, column))
            .collect();
        return DMatrix::from_fn(n, n, |i, j| {
            coordinates[i].0.dot(&coordinates[j].0) / ridge
                + coordinates[i].1.dot(&coordinates[j].1)
        });
    }
    DMatrix::from_fn(n, n, |i, j| inv.inner(&columns[i], &columns[j]))
}
fn basis_direction(input_dim: usize, axis: usize, scale: f64) -> Vec<f64> {
    let mut out = vec![0.; input_dim];
    out[axis] = scale;
    out
}
fn metric2(matrix: &DMatrix<f64>) -> [[f64; 2]; 2] {
    [
        [matrix[(0, 0)], matrix[(0, 1)]],
        [matrix[(1, 0)], matrix[(1, 1)]],
    ]
}
pub fn layer_name(layer: usize, last: usize) -> String {
    if layer == 0 {
        "Input".into()
    } else if layer == last {
        "Output".into()
    } else {
        format!("Hidden {layer}")
    }
}
pub fn prepare(net: &NumericalMlp, request: &Request) -> Result<Frame> {
    validate(net, request)?;
    let reference = Reference::build(net, request, true, &|| true)?;
    prepare_geometry(net, request, &reference, None, &|| true)
}
fn validate(net: &NumericalMlp, request: &Request) -> Result<()> {
    let n = net.input_dim();
    if request.anchor.len() != n
        || request.endpoint.len() != n
        || !(2..=3).contains(&request.axes.len())
        || request.axis_scales.len() != request.axes.len()
        || request.axes.iter().any(|&axis| axis >= n)
        || request
            .axes
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != request.axes.len()
        || request
            .axis_scales
            .iter()
            .any(|&v| !v.is_finite() || v <= 0.)
        || request.layer > net.layer_count()
        || request.grid[0] < 2
        || request.grid[1] < 2
        || request.grid[0].saturating_mul(request.grid[1]) > MAX_GRID
        || request.probes.len() > MAX_PROBES
        || request.probes.is_empty()
        || request.probes.iter().any(|row| row.len() != n)
        || request
            .ranges
            .iter()
            .any(|r| r[0] >= r[1] || r.iter().any(|x| !x.is_finite()))
        || request
            .anchor
            .iter()
            .chain(&request.endpoint)
            .chain(request.probes.iter().flatten())
            .any(|v| !v.is_finite())
        || !request.direction_angle.is_finite()
        || !request.t.is_finite()
        || !(0. ..=1.).contains(&request.t)
        || !request.radius.is_finite()
        || !(0.001..=1.).contains(&request.radius)
        || request.axes.iter().take(2).enumerate().any(|(i, &axis)| {
            let extent = request.radius * request.axis_scales[i];
            !(request.anchor[axis] - extent).is_finite()
                || !(request.anchor[axis] + extent).is_finite()
                || !(2. * extent).is_finite()
        })
    {
        return Err(Error::Config("invalid network geometry request".into()));
    }
    Ok(())
}
fn prepare_geometry(
    net: &NumericalMlp,
    request: &Request,
    reference: &Reference,
    cached_neighborhood: Option<&Arc<Neighborhood>>,
    check: &impl Fn() -> bool,
) -> Result<Frame> {
    current(check)?;
    let n = net.input_dim();
    let n_layers = net.layer_count() + 1;
    let inverse = &reference.metrics;
    let reference = &reference.layers;
    let d0 = basis_direction(n, request.axes[0], request.axis_scales[0]);
    let d1 = basis_direction(n, request.axes[1], request.axis_scales[1]);
    let a = request.direction_angle;
    let selected: Vec<f64> = d0
        .iter()
        .zip(&d1)
        .map(|(x, y)| a.cos() * x + a.sin() * y)
        .collect();
    let perpendicular: Vec<f64> = d0
        .iter()
        .zip(&d1)
        .map(|(x, y)| -a.sin() * x + a.cos() * y)
        .collect();
    let selected_columns = request
        .axes
        .iter()
        .zip(&request.axis_scales)
        .map(|(&axis, &scale)| basis_direction(n, axis, scale))
        .collect::<Vec<_>>();
    let mut directions = selected_columns.clone();
    directions.extend([selected, perpendicular]);
    directions.push(
        request
            .endpoint
            .iter()
            .zip(&request.anchor)
            .map(|(b, a)| b - a)
            .collect(),
    );
    let signals = net.trace_directions(&request.anchor, &directions)?;
    let input: Vec<_> = request
        .anchor
        .iter()
        .zip(&request.endpoint)
        .map(|(a, b)| a * (1. - request.t) + b * request.t)
        .collect();
    current(check)?;
    let finite = net.forward_layers(&input)?;
    let comparisons = {
        let mut residual = Vec::new();
        (0..n_layers)
            .map(|l| {
                current(check)?;
                let linear: Vec<_> = signals[l]
                    .tangents
                    .last()
                    .unwrap()
                    .iter()
                    .map(|v| request.t * v)
                    .collect();
                let actual: Vec<_> = finite[l]
                    .iter()
                    .zip(&signals[l].activation)
                    .map(|(x, a)| x - a)
                    .collect();
                finite::compare_into(
                    &inverse[l],
                    layer_name(l, net.layer_count()),
                    &linear,
                    &actual,
                    &mut residual,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };
    let neighborhood = match cached_neighborhood {
        Some(cached) => cached.clone(),
        None => Arc::new(finite::neighborhood(
            net, request, &signals, inverse, check,
        )?),
    };

    let local = gram(
        &inverse[request.layer],
        &signals[request.layer].tangents[..request.axes.len()],
    );
    let metric_3d = if request.axes.len() == 3 {
        Some(std::array::from_fn(|i| {
            std::array::from_fn(|j| local[(i, j)])
        }))
    } else {
        None
    };

    let mut grid = Vec::with_capacity(request.grid[0] * request.grid[1]);
    for iy in 0..request.grid[1] {
        for ix in 0..request.grid[0] {
            current(check)?;
            let sx = request.ranges[0][0]
                + (request.ranges[0][1] - request.ranges[0][0]) * ix as f64
                    / (request.grid[0] - 1) as f64;
            let sy = request.ranges[1][0]
                + (request.ranges[1][1] - request.ranges[1][0]) * iy as f64
                    / (request.grid[1] - 1) as f64;
            let mut x = request.anchor.clone();
            x[request.axes[0]] = sx;
            x[request.axes[1]] = sy;
            let cols = net.tangent_columns(&x, &selected_columns[..2], request.layer)?;
            let g = gram(&inverse[request.layer], &cols);
            grid.push(Glyph {
                position: [sx, sy],
                strength: (g[(0, 0)] + g[(1, 1)]).max(0.).sqrt(),
                gram: metric2(&g),
            });
        }
    }
    let mut pca_bases = request.previous_basis.clone();
    pca_bases.resize(n_layers, None);
    let mut paths = Vec::new();
    let selected_layers = (0..n_layers).take(4).collect::<Vec<_>>();
    let count = MAX_PATH.min(33);
    let path_values = (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            request
                .anchor
                .iter()
                .zip(&request.endpoint)
                .map(|(a, b)| a * (1. - t) + b * t)
                .collect::<Vec<_>>()
        })
        .map(|x| {
            current(check)?;
            net.forward_layers(&x)
        })
        .collect::<Result<Vec<_>>>()?;
    for l in selected_layers {
        current(check)?;
        let width = reference[0][l].len();
        if pca_bases[l]
            .as_ref()
            .is_some_and(|p| p.dimensions() != width)
        {
            pca_bases[l] = None;
        }
        if pca_bases[l].is_none() && request.probes.len() >= 2 {
            let values = reference
                .iter()
                .flat_map(|layers| layers[l].iter().copied())
                .collect::<Vec<_>>();
            pca_bases[l] = Pca::fit(
                &values,
                request.probes.len(),
                width,
                2,
                RepresentationLimits::default(),
            )
            .ok();
        }
        let Some(pca) = &pca_bases[l] else {
            continue;
        };
        let raw = path_values
            .iter()
            .flat_map(|layers| layers[l].iter().copied())
            .collect::<Vec<_>>();
        let projection = pca.project(&raw, count)?;
        let dims = pca.components();
        let mut points = Vec::with_capacity(count);
        for i in 0..count {
            points.push([
                projection[i * dims],
                if dims > 1 {
                    projection[i * dims + 1]
                } else {
                    i as f64 / (count - 1) as f64
                },
            ]);
        }
        let mut actual_energy = 0.;
        let mut projected_energy = 0.;
        for i in 1..count {
            actual_energy += path_values[i][l]
                .iter()
                .zip(&path_values[i - 1][l])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();
            projected_energy += (points[i][0] - points[i - 1][0]).powi(2);
            if dims > 1 {
                projected_energy += (points[i][1] - points[i - 1][1]).powi(2);
            }
        }
        paths.push(Path {
            layer: if l == 0 {
                "Input".into()
            } else if l == net.layer_count() {
                "Output".into()
            } else {
                format!("Hidden {l}")
            },
            points,
            variance: pca.explained_fraction().iter().take(2).sum(),
            projected_energy: if actual_energy > 0. {
                (projected_energy / actual_energy).clamp(0., 1.)
            } else {
                1.
            },
        });
    }

    current(check)?;
    Ok(Frame {
        grid,
        metric_3d,
        comparisons,
        neighborhood,
        t: request.t,
        radius: request.radius,
        slice_anchor: [
            request.anchor[request.axes[0]],
            request.anchor[request.axes[1]],
        ],
        slice_position: [input[request.axes[0]], input[request.axes[1]]],
        slice_endpoint: [
            request.endpoint[request.axes[0]],
            request.endpoint[request.axes[1]],
        ],
        position_on_slice: input
            .iter()
            .zip(&request.anchor)
            .enumerate()
            .all(|(i, (x, a))| request.axes[..2].contains(&i) || x == a),
        output_tangent: signals.last().unwrap().tangents[request.axes.len()].clone(),
        output_values: signals.last().unwrap().activation.clone(),
        finite_output_values: finite.last().unwrap().clone(),
        paths,
        pca_bases,
        reference_count: request.probes.len(),
        kink_count: signals.iter().map(|s| s.relu_kinks).sum(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::nn::network::Mlp;

    fn fixture() -> (NumericalMlp, Request) {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = NumericalMlp::from_model(&Mlp::new(3, &[24, 6], 2, &mut rng).unwrap()).unwrap();
        let probes: Vec<_> = (0..9)
            .map(|i| {
                let t = i as f64 / 8.;
                vec![2. * t - 1., (3. * t).sin(), (7. * t).cos()]
            })
            .collect();
        let request = Request {
            network: "test".into(),
            anchor: probes[2].clone(),
            endpoint: probes[6].clone(),
            probes,
            axes: vec![0, 1, 2],
            axis_scales: vec![1., 2., 3.],
            ranges: [[-1., 1.], [-1., 1.]],
            direction_angle: 0.4,
            t: 1.,
            radius: 0.2,
            action_mask: None,
            layer: 1,
            grid: [3, 3],
            previous_basis: vec![],
        };
        (model, request)
    }

    fn equivalent(a: &Frame, b: &Frame) {
        assert_eq!(a.output_values, b.output_values);
        assert_eq!(a.output_tangent, b.output_tangent);
        assert_eq!(a.reference_count, b.reference_count);
        assert_eq!(a.kink_count, b.kink_count);
        assert_eq!(a.paths.len(), b.paths.len());
        for (a, b) in a.paths.iter().zip(&b.paths) {
            assert_eq!(a.points, b.points);
            assert_eq!(a.variance, b.variance);
            assert_eq!(a.projected_energy, b.projected_energy);
        }
        match (a.metric_3d, b.metric_3d) {
            (Some(a), Some(b)) => {
                for (a, b) in a.iter().flatten().zip(b.iter().flatten()) {
                    assert!((a - b).abs() < 1e-7 * a.abs().max(1.));
                }
            }
            (None, None) => {}
            _ => panic!("3D metric availability differs"),
        }
        for (a, b) in a.grid.iter().zip(&b.grid) {
            for (a, b) in a.gram.iter().flatten().zip(b.gram.iter().flatten()) {
                assert!((a - b).abs() < 1e-7 * a.abs().max(1.), "{a} != {b}");
            }
        }
        for (a, b) in a.comparisons.iter().zip(&b.comparisons) {
            assert!((a.error - b.error).abs() < 1e-7);
            assert!((a.linear_norm - b.linear_norm).abs() < 1e-7 * a.linear_norm.max(1.));
        }
        for (a, b) in a
            .neighborhood
            .errors
            .iter()
            .flatten()
            .zip(b.neighborhood.errors.iter().flatten())
        {
            match (a, b) {
                (Some(a), Some(b)) => assert!((a - b).abs() < 1e-7),
                (None, None) => (),
                _ => panic!("neighborhood validity differs"),
            }
        }
    }

    #[test]
    fn preparation_reuses_reference_for_geometry_and_invalidates_exact_population() {
        let (model, mut request) = fixture();
        let mut state = Preparation::new(model.clone());
        equivalent(
            &prepare(&model, &request).unwrap(),
            &state.prepare(&request).unwrap(),
        );
        let first = std::sync::Arc::downgrade(state.reference.as_ref().unwrap());
        request.direction_angle = 1.2;
        request.endpoint = request.probes[3].clone();
        request.layer = 2;
        equivalent(
            &prepare(&model, &request).unwrap(),
            &state.prepare(&request).unwrap(),
        );
        assert!(std::sync::Weak::ptr_eq(
            &first,
            &std::sync::Arc::downgrade(state.reference.as_ref().unwrap())
        ));
        request.probes[0][0] += 0.2;
        equivalent(
            &prepare(&model, &request).unwrap(),
            &state.prepare(&request).unwrap(),
        );
        assert!(
            first.upgrade().is_none(),
            "replaced reference must be released"
        );
        request.probes[0][0] = -0.;
        state.prepare(&request).unwrap();
        let negative_zero = Arc::downgrade(state.reference.as_ref().unwrap());
        request.probes[0][0] = 0.;
        state.prepare(&request).unwrap();
        assert!(
            negative_zero.upgrade().is_none(),
            "population identity preserves f64 bits"
        );
    }

    #[test]
    fn low_rank_metric_matches_dense_including_constant_and_small_populations() {
        for count in [1, 2, 3, 7] {
            for constant in [false, true] {
                let rows: Vec<Vec<f64>> = (0..count)
                    .map(|i| {
                        (0..48)
                            .map(|j| {
                                if constant {
                                    0.25
                                } else {
                                    ((i * 31 + j * 7) as f64).sin()
                                }
                            })
                            .collect()
                    })
                    .collect();
                let dense = covariance_inverse(rows.iter().map(Vec::as_slice), 48).unwrap();
                let metric = Metric::fit(rows.iter().map(Vec::as_slice), 48, &|| true).unwrap();
                let a: Vec<_> = (0..48).map(|i| (i as f64).cos()).collect();
                let b: Vec<_> = (0..48).map(|i| (i as f64).sin()).collect();
                for (a, b) in [(&a, &a), (&a, &b), (&b, &b)] {
                    let expected = inner(&dense, a, b);
                    let actual = metric.inner(a, b);
                    assert!((actual - expected).abs() < 1e-7 * expected.abs().max(1.));
                }
                assert!(metric.retained_bytes() < 48 * 48 * 8);
            }
        }
    }

    #[test]
    fn cancelled_reference_build_does_not_replace_or_publish_partial_work() {
        let (model, mut request) = fixture();
        let mut state = Preparation::new(model);
        state.prepare(&request).unwrap();
        request.probes[0][0] += 0.2;
        let checks = std::cell::Cell::new(0);
        let result = state.prepare_cancellable(&request, &|| {
            checks.set(checks.get() + 1);
            checks.get() < 5
        });
        assert!(result.is_err());
        assert_eq!(checks.get(), 5, "stop at the first cancelled stage");
        assert!(
            state.reference.is_none(),
            "obsolete reference must be retired before replacement"
        );
    }

    #[test]
    fn cache_budget_bypass_keeps_geometry_supported() {
        let (model, request) = fixture();
        let mut state = Preparation::with_cache_budget(model.clone(), 0);
        equivalent(
            &prepare(&model, &request).unwrap(),
            &state.prepare(&request).unwrap(),
        );
        assert!(state.reference.is_none());
    }

    #[test]
    fn wide_low_rank_metric_preserves_ridge_in_and_out_of_population_span() {
        // Covariance has exactly one nonzero axis, with variance 1. The other
        // 4095 axes must retain the inverse ridge metric, not be discarded.
        let width = 4096;
        let rows: Vec<Vec<f64>> = [-1., 0., 1.]
            .into_iter()
            .map(|x| {
                let mut row = vec![0.; width];
                row[0] = x;
                row
            })
            .collect();
        let metric = Metric::fit(rows.iter().map(Vec::as_slice), width, &|| true).unwrap();
        let mut direction = vec![0.; width];
        direction[0] = 1.;
        let ridge = 0.001 / width as f64;
        assert!((metric.inner(&direction, &direction) - 1. / (1. + ridge)).abs() < 1e-8);
        direction[0] = 0.;
        direction[1] = 1.;
        assert!((metric.inner(&direction, &direction) - 1. / ridge).abs() < 1e-6);
        assert!(
            metric.retained_bytes() < 128 * 1024,
            "wide metric must remain proportional to population × width"
        );
    }

    #[test]
    fn geometry_uses_true_mlps_and_a_fixed_reference_population() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = NumericalMlp::from_model(&Mlp::new(3, &[8], 2, &mut rng).unwrap()).unwrap();
        let probes = (0..21)
            .map(|i| {
                let t = i as f64 / 20.;
                vec![2. * t - 1., (3. * t).sin(), (7. * t).cos()]
            })
            .collect::<Vec<_>>();
        let frame = prepare(
            &model,
            &Request {
                network: "test".into(),
                anchor: probes[7].clone(),
                endpoint: probes[17].clone(),
                probes,
                axes: vec![0, 1],
                axis_scales: vec![1., 1.],
                ranges: [[-1., 1.], [-1., 1.]],
                direction_angle: 0.4,
                t: 1.,
                radius: 0.2,
                action_mask: None,
                layer: 1,
                grid: [8, 6],
                previous_basis: vec![],
            },
        )
        .unwrap();
        assert_eq!(frame.grid.len(), 48);
        assert_eq!(frame.comparisons.len(), 3);
        assert!(frame.grid.iter().all(|p| p.strength.is_finite()));
        assert!(frame.paths.len() >= 2);
        assert!(
            frame
                .paths
                .iter()
                .all(|path| (0. ..=1.).contains(&path.projected_energy))
        );
    }
}
