// Frozen pre-task-5a finite oracle (a5a7a79). Keep independent of scratch helpers.
//! Finite displacement comparisons in the original covariance metric.
use super::{Metric, Request, current};
use rl::{
    Error, Result,
    diagnostics::network_geometry::{LayerDirections, NumericalMlp},
};

pub const SIDE: usize = 16;
pub const ERROR_SCALE: [f64; 2] = [0., 1.];
const EPSILON: f64 = 1e-12;

#[derive(Clone, Debug)]
pub struct LayerComparison {
    pub name: String,
    /// An independent canonical plane for this pair, not neuron/PCA axes.
    pub linear: [f64; 2],
    pub actual: [f64; 2],
    pub linear_norm: f64,
    pub actual_norm: f64,
    pub residual_norm: f64,
    pub error: f64,
}

fn norm(metric: &Metric, vector: &[f64]) -> Result<f64> {
    let squared = metric.inner(vector, vector);
    if vector.iter().any(|v| !v.is_finite()) || !squared.is_finite() {
        return Err(Error::Numerical(
            "nonfinite finite-displacement metric".into(),
        ));
    }
    Ok(squared.max(0.).sqrt())
}

fn error(metric: &Metric, linear: &[f64], actual: &[f64]) -> Result<(f64, f64, f64, f64)> {
    if linear.len() != actual.len() || linear.is_empty() {
        return Err(Error::Shape("inconsistent finite displacement".into()));
    }
    let linear_norm = norm(metric, linear)?;
    let actual_norm = norm(metric, actual)?;
    // Keep the full represented residual, including directions invisible in PCA.
    let residual: Vec<_> = actual.iter().zip(linear).map(|(a, l)| a - l).collect();
    let residual_norm = norm(metric, &residual)?;
    let scale = linear_norm.max(actual_norm).max(EPSILON);
    let relative =
        (residual_norm / scale) / (linear_norm / scale + actual_norm / scale + EPSILON / scale);
    if !relative.is_finite() {
        return Err(Error::Numerical(
            "nonfinite finite-displacement error".into(),
        ));
    }
    Ok((
        linear_norm,
        actual_norm,
        residual_norm,
        relative.clamp(0., 1.),
    ))
}

pub(super) fn compare(
    metric: &Metric,
    name: String,
    linear: &[f64],
    actual: &[f64],
) -> Result<LayerComparison> {
    let (linear_norm, actual_norm, residual_norm, error) = error(metric, linear, actual)?;
    let canonical = if linear_norm > 0. {
        let horizontal = metric.inner(linear, actual) / linear_norm;
        if !horizontal.is_finite() {
            return Err(Error::Numerical("nonfinite canonical alignment".into()));
        }
        let residual: Vec<_> = actual
            .iter()
            .zip(linear)
            .map(|(d, l)| d - (l / linear_norm) * horizontal)
            .collect();
        [horizontal, norm(metric, &residual)?]
    } else {
        // A zero prediction has no distinguished orientation.
        [0., actual_norm]
    };
    Ok(LayerComparison {
        name,
        linear: [linear_norm, 0.],
        actual: canonical,
        linear_norm,
        actual_norm,
        residual_norm,
        error,
    })
}

#[derive(Clone, Debug)]
pub struct Neighborhood {
    pub positions: Vec<[f64; 2]>,
    pub bounds: [[f64; 2]; 2],
    /// Layer-major. None is an invalid numerical evaluation, never zero error.
    pub errors: Vec<Vec<Option<f64>>>,
    anchor: Vec<f64>,
    axes: [usize; 2],
    scales: [f64; 2],
    radius: f64,
}
impl Neighborhood {
    pub(super) fn matches(&self, request: &Request) -> bool {
        self.anchor.len() == request.anchor.len()
            && self
                .anchor
                .iter()
                .zip(&request.anchor)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.axes == request.axes[..2]
            && self
                .scales
                .iter()
                .zip(&request.axis_scales)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.radius.to_bits() == request.radius.to_bits()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.anchor.capacity() * 8
            + self.positions.capacity() * std::mem::size_of::<[f64; 2]>()
            + self.errors.capacity() * std::mem::size_of::<Vec<Option<f64>>>()
            + self
                .errors
                .iter()
                .map(|row| row.capacity() * std::mem::size_of::<Option<f64>>())
                .sum::<usize>()
    }
    pub fn invalid_count(&self, layer: usize) -> usize {
        self.errors.get(layer).map_or(SIDE * SIDE, |row| {
            row.iter().filter(|v| v.is_none()).count()
        })
    }
}

pub(super) fn neighborhood(
    net: &NumericalMlp,
    request: &Request,
    anchor: &[LayerDirections],
    metrics: &[Metric],
    check: &impl Fn() -> bool,
) -> Result<Neighborhood> {
    let bounds = std::array::from_fn(|axis| {
        let center = request.anchor[request.axes[axis]];
        let extent = request.radius * request.axis_scales[axis];
        [center - extent, center + extent]
    });
    let mut result = Neighborhood {
        positions: Vec::with_capacity(SIDE * SIDE),
        bounds,
        errors: (0..metrics.len())
            .map(|_| Vec::with_capacity(SIDE * SIDE))
            .collect(),
        anchor: request.anchor.clone(),
        axes: [request.axes[0], request.axes[1]],
        scales: [request.axis_scales[0], request.axis_scales[1]],
        radius: request.radius,
    };
    for iy in 0..SIDE {
        for ix in 0..SIDE {
            current(check)?;
            let offsets =
                [ix, iy].map(|i| request.radius * (2. * i as f64 / (SIDE - 1) as f64 - 1.));
            let mut input = request.anchor.clone();
            for axis in 0..2 {
                input[request.axes[axis]] += offsets[axis] * request.axis_scales[axis];
            }
            // Construct L from the input actually evaluated. Near floating-point
            // resolution, adding an ideal offset can round back to A; treating
            // that lost offset as a real displacement invents affine error.
            let actual_offsets: [f64; 2] = std::array::from_fn(|axis| {
                (input[request.axes[axis]] - request.anchor[request.axes[axis]])
                    / request.axis_scales[axis]
            });
            result
                .positions
                .push([input[request.axes[0]], input[request.axes[1]]]);
            // A failed full forward conservatively invalidates every layer at
            // this point. No later layer is silently omitted or replaced.
            let values = net.forward_layers(&input).ok();
            for (layer, row) in result.errors.iter_mut().enumerate() {
                current(check)?;
                let value = values.as_ref().and_then(|values| {
                    let delta: Vec<_> = values[layer]
                        .iter()
                        .zip(&anchor[layer].activation)
                        .map(|(x, a)| x - a)
                        .collect();
                    let linear: Vec<_> = anchor[layer].tangents[0]
                        .iter()
                        .zip(&anchor[layer].tangents[1])
                        .map(|(x, y)| actual_offsets[0] * x + actual_offsets[1] * y)
                        .collect();
                    error(&metrics[layer], &linear, &delta).ok().map(|p| p.3)
                });
                row.push(value);
            }
        }
    }
    current(check)?;
    Ok(result)
}
