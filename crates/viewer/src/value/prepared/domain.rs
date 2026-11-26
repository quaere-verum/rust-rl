//! Robust statistics of the displayed slice, never of hidden-coordinate ranges.
use super::*;
const MAX_SAMPLES: usize = 4096;

#[derive(Clone, Copy, Default)]
pub struct ColorRanges {
    pub value: Option<[f64; 2]>,
    pub difference: Option<[f64; 2]>,
}
#[derive(Clone)]
pub struct CalibrationRequest {
    pub serial: u64,
    pub source: ValueSource,
    pub reference: Option<ValueSource>,
}
#[derive(Clone)]
pub struct CalibrationResult {
    pub serial: u64,
    pub source: CacheKey,
    pub ranges: ColorRanges,
    pub error: Option<String>,
}

/// Linear empirical percentiles with 10% padding. Constant data receives a
/// finite magnitude-relative interval. The original predictions are untouched.
pub(crate) fn limits(mut values: Vec<f64>, symmetric: bool) -> Option<[f64; 2]> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    if symmetric {
        for value in &mut values {
            *value = value.abs();
        }
    }
    values.sort_unstable_by(f64::total_cmp);
    let percentile = |p: f64| {
        let at = p * (values.len() - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        values[lo] * (1. - at.fract()) + values[hi] * at.fract()
    };
    let lo = percentile(0.02);
    let hi = percentile(0.98);
    let floor = hi.abs().max(lo.abs()).max(1.) * 1e-6;
    if symmetric {
        let bound = (hi * 1.1).max(floor).min(f64::MAX);
        Some([-bound, bound])
    } else {
        let pad = (hi * 0.1 - lo * 0.1).abs().max(floor);
        Some([(lo - pad).max(-f64::MAX), (hi + pad).min(f64::MAX)])
    }
}

pub(super) fn twin_sample(
    samples: &mut Vec<f64>,
    prediction: &ValuePrediction,
    output: &ValueOutput,
    index: usize,
    count: usize,
) -> Result<()> {
    if matches!(output, ValueOutput::Q1 | ValueOutput::Q2)
        && index.is_multiple_of(count.div_ceil(MAX_SAMPLES / 2).max(1))
        && matches!(prediction, ValuePrediction::Twin { .. })
        && samples.len() + 2 <= MAX_SAMPLES
    {
        samples.push(prediction.scalar(&ValueOutput::Q1, None)?);
        samples.push(prediction.scalar(&ValueOutput::Q2, None)?);
    }
    Ok(())
}

pub(super) fn finish(grid: &mut PreparedValueGrid, twins: Vec<f64>) {
    let stride = grid.values.len().div_ceil(MAX_SAMPLES).max(1);
    let values = if twins.is_empty() {
        (0..grid.values.len())
            .step_by(stride)
            .filter(|&i| grid.valid[i])
            .map(|i| grid.values[i])
            .collect()
    } else {
        twins
    };
    grid.color_domain = limits(values, false);
    grid.reference_color_domain = if grid.empirical_reference() {
        limits(
            grid.reference_series
                .iter()
                .flat_map(|s| s.values.iter().copied())
                .collect(),
            false,
        )
    } else {
        grid.color_domain
    };
    grid.difference_color_domain = limits(
        (0..grid.values.len())
            .step_by(stride)
            .filter_map(|i| grid.difference_at(i))
            .chain(
                grid.empirical_differences
                    .iter()
                    .flat_map(|s| s.values.iter().copied()),
            )
            .collect(),
        true,
    );
}

pub(crate) fn bounded_slice(slice: &ValueSlice) -> Result<ValueSlice> {
    ensure(
        slice.model_eligible(),
        "calibration requires a supported model slice",
    )?;
    let mut slice = slice.clone();
    slice.resolution = if slice.y.is_some() {
        [32, 32]
    } else {
        [256, 1]
    };
    while slice.len()? > 1024 {
        let axis = (0..2)
            .find(|&i| {
                slice.resolution[i] > 2
                    && (if i == 0 { Some(slice.x) } else { slice.y })
                        != Some(super::super::slice::ValueCoordinate::DiscreteAction)
            })
            .ok_or_else(|| Error::Config("calibration slice exceeds sampling budget".into()))?;
        slice.resolution[axis] = slice.resolution[axis].div_ceil(2);
    }
    Ok(slice)
}

pub(super) fn calibrate(
    request: &ValueRequest,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Option<CalibrationResult> {
    let calibration = request.calibration.as_ref()?;
    let result = (|| -> Result<(ColorRanges, Option<String>)> {
        calibration.source.check_available()?;
        ensure(
            !calibration.source.identity().recorded,
            "calibration model weights unavailable",
        )?;
        let mut r = request.clone();
        r.current = calibration.source.clone();
        r.requested_revision = r.current.identity().revision;
        r.reference_value = calibration.reference.clone();
        r.empirical = None; // Evidence is bound to the actual subject, never a calibration predecessor.
        r.reference_empirical = false;
        r.reference_policy = None;
        r.visits.clear();
        r.calibration = None; // The nested grid performs no further calibration.
        r.slice = Arc::new(bounded_slice(&request.slice)?);
        r.pin = ValuePin::State(r.slice.anchor());
        let owner = RequestOwner::new(r, accounting)?;
        let grid = prepare_grid(&owner, accounting, cancelled, with_bundle)?
            .ok_or_else(|| Error::State("calibration cancelled".into()))?;
        ensure(
            grid.color_domain.is_some(),
            "calibration slice has no finite values",
        )?;
        Ok((
            ColorRanges {
                value: grid.color_domain,
                difference: grid.difference_color_domain,
            },
            grid.reference_error.clone(),
        ))
    })();
    Some(match result {
        Ok((ranges, error)) => CalibrationResult {
            serial: calibration.serial,
            source: calibration.source.identity().key.clone(),
            ranges,
            error,
        },
        Err(e) => CalibrationResult {
            serial: calibration.serial,
            source: calibration.source.identity().key.clone(),
            ranges: ColorRanges::default(),
            error: Some(e.to_string()),
        },
    })
}
