//! Checked complete-population learning statistics over recorded evidence.
use super::training::{TrainingAggregatePoint, TrainingClock, TrainingRun};
pub use rl::diagnostics::analysis::EmpiricalSummary;
use rl::{
    Error, Result,
    core::action::ActionSpace,
    diagnostics::{
        analysis::EmpiricalDistribution,
        empirical::MAX_EMPIRICAL_SAMPLES,
        policy_batch::{PolicyBatch, numeric_row},
        policy_numerics::current_reference_kl,
        representations::match_reference_sets,
        schema::*,
    },
};

pub const MAX_FIXED_ROWS: usize = 1024;
pub const MAX_ACTION_COMPONENTS: usize = 64;
pub const MAX_READ_BYTES: usize = 4 * 1024 * 1024;
/// Reference RMS normalizers cannot vanish; this floor is in recorded action units.
pub const NORMALIZATION_FLOOR: f64 = 1e-6;

fn ensure(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Config(reason.into()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChangeKind {
    NormalizedRms,
    ReferenceKl,
    DiscreteDisagreement,
}

#[derive(Clone, Debug)]
pub struct PolicyChange {
    pub kind: ChangeKind,
    pub value: f64,
    pub rows: usize,
    pub normalization: Vec<f64>,
}

/// Complete fixed-population comparison; the reference defines every scale.
pub fn policy_change(
    reference: &Chunk,
    reference_probes: &Chunk,
    current: &Chunk,
    current_probes: &Chunk,
    space: &ActionSpace,
) -> Result<PolicyChange> {
    let width = match space {
        ActionSpace::Discrete { count } => *count,
        _ => space.dimensions(),
    };
    ensure(
        (1..=MAX_ACTION_COMPONENTS).contains(&width),
        "learning action width exceeds 64; full population was not subsampled",
    )?;
    for chunk in [reference, reference_probes, current, current_probes] {
        ensure(
            chunk.retained_charge() <= MAX_READ_BYTES,
            "learning policy evidence exceeds 4 MiB admission",
        )?;
    }
    ensure(
        reference.context.identity.run_id == current.context.identity.run_id
            && reference.context.identity.model_role == current.context.identity.model_role,
        "learning policy run or model role differs from reference",
    )?;
    let (Record::ProbeSet(a), Record::ProbeSet(b)) =
        (&reference_probes.record, &current_probes.record)
    else {
        return Err(Error::Config(
            "learning comparison needs genuine fixed probe populations".into(),
        ));
    };
    ensure(
        a.id == b.id,
        "learning policy population identity differs from the frozen reference",
    )?;
    let row_count = |c: &Chunk, key: &str| {
        c.arrays
            .get(key)
            .and_then(|a| a.shape().first())
            .copied()
            .ok_or_else(|| Error::Shape("learning population observations missing".into()))
    };
    let rows = row_count(reference_probes, &a.observations)?;
    ensure(
        (1..=MAX_FIXED_ROWS).contains(&rows) && row_count(current_probes, &b.observations)? == rows,
        "learning fixed population exceeds 1024 rows or changed size; no scientific rows were subsampled",
    )?;
    let aligned = match_reference_sets(reference_probes, current_probes)?;
    let reference = PolicyBatch::new(reference, Some(reference_probes), space)?;
    let current = PolicyBatch::new(current, Some(current_probes), space)?;
    // A policy may carry its effective legal set without duplicating it in
    // ProbeSet. Independently valid rows still need identical conditioning.
    if matches!(space, ActionSpace::Discrete { .. }) {
        for (i, &j) in aligned.iter().enumerate() {
            let a = reference.effective_mask_with_budget(i, 2 * MAX_ACTION_COMPONENTS)?;
            let b = current.effective_mask_with_budget(j, 2 * MAX_ACTION_COMPONENTS)?;
            ensure(
                (0..width)
                    .all(|c| a.as_ref().is_none_or(|m| m[c]) == b.as_ref().is_none_or(|m| m[c])),
                "learning effective legal action masks differ from the frozen input condition",
            )?;
        }
    }
    let mut normalization = Vec::new();
    let (kind, value) = match (reference.parameters(), current.parameters()) {
        (PolicyParameters::Deterministic, PolicyParameters::Deterministic) => match space {
            ActionSpace::Discrete { .. } => {
                let a = reference.actions().as_u64()?;
                let b = current.actions().as_u64()?;
                let changed = aligned
                    .iter()
                    .enumerate()
                    .filter(|&(i, j)| a[i] != b[*j])
                    .count();
                (
                    ChangeKind::DiscreteDisagreement,
                    changed as f64 / rows as f64,
                )
            }
            _ => {
                normalization = match space {
                    ActionSpace::Box { low, high } => low
                        .iter()
                        .zip(high)
                        .map(|(&l, &h)| f64::from(h) - f64::from(l))
                        .collect(),
                    _ => {
                        let mut scales = vec![0f64; width];
                        for i in 0..rows {
                            for (j, value) in
                                numeric_row(reference.actions(), i)?.iter().enumerate()
                            {
                                scales[j] = scales[j].hypot(*value);
                            }
                        }
                        for scale in &mut scales {
                            *scale = (*scale / (rows as f64).sqrt()).max(NORMALIZATION_FLOOR);
                        }
                        scales
                    }
                };
                ensure(
                    normalization.iter().all(|s| s.is_finite() && *s > 0.),
                    "learning reference normalization is nonfinite",
                )?;
                let mut norm = 0f64;
                for (i, &j) in aligned.iter().enumerate() {
                    let a = numeric_row(reference.actions(), i)?;
                    let b = numeric_row(current.actions(), j)?;
                    for component in 0..width {
                        norm = norm.hypot((a[component] - b[component]) / normalization[component]);
                    }
                }
                let rms = norm / ((rows * width) as f64).sqrt();
                ensure(rms.is_finite(), "learning normalized policy RMS overflow")?;
                (ChangeKind::NormalizedRms, rms)
            }
        },
        _ => {
            let mut mean = 0.;
            for (i, &j) in aligned.iter().enumerate() {
                // This existing helper names its arguments current/reference;
                // reversed here deliberately: our scientific quantity is KL(ref || current).
                let kl = current_reference_kl(&reference, i, &current, j)?
                    .ok_or_else(|| Error::Config("incompatible learning policy families".into()))?;
                if kl.is_infinite() {
                    mean = f64::INFINITY;
                } else if !mean.is_infinite() {
                    mean += (kl - mean) / (i + 1) as f64;
                }
            }
            (ChangeKind::ReferenceKl, mean)
        }
    };
    Ok(PolicyChange {
        kind,
        value,
        rows,
        normalization,
    })
}

pub fn evaluation_summary(chunk: &Chunk) -> Result<EmpiricalSummary> {
    ensure(
        chunk.retained_charge() <= MAX_READ_BYTES,
        "learning evaluation exceeds 4 MiB admission",
    )?;
    let Record::Empirical(e) = &chunk.record else {
        return Err(Error::Config(
            "learning performance requires genuine empirical return samples".into(),
        ));
    };
    ensure(
        e.evaluation.conditioning == Conditioning::InitialScenario
            && e.evaluation.initial_action == ActionCondition::CurrentPolicy
            && e.probe_row.is_none(),
        "learning performance excludes conditional or fixed-first-action returns",
    )?;
    ensure(
        e.evaluation.complete_returns,
        "learning performance excludes incomplete/truncated-prefix returns",
    )?;
    let returns = chunk
        .arrays
        .get(&e.returns)
        .ok_or_else(|| Error::Shape("learning return samples missing".into()))?;
    ensure(
        (1..=MAX_EMPIRICAL_SAMPLES).contains(&returns.len()),
        "learning evaluation exceeds complete sample admission",
    )?;
    chunk.validate()?;
    let mut summary = EmpiricalDistribution::new(&returns.to_f64())?.summary()?;
    if !e.evaluation.independent_future_streams {
        summary.mean_standard_error = None;
    }
    Ok(summary)
}

#[derive(Clone, Copy)]
struct RunMean {
    x: f64,
    mean: Option<f64>,
    starts_segment: bool,
}

/// Align one mean from each independent training run. Duplicate clocks use
/// the latest measurement. A missing point breaks interpolation; no run may
/// contribute outside observed support.
pub(super) fn aggregate_training_runs(
    runs: &[&TrainingRun],
    clock: TrainingClock,
) -> Result<Vec<TrainingAggregatePoint>> {
    let mut curves = Vec::with_capacity(runs.len());
    let mut start = f64::NEG_INFINITY;
    let mut end = f64::INFINITY;
    for run in runs {
        let mut curve: Vec<RunMean> = Vec::with_capacity(run.evaluations.len());
        let mut missing_clock = false;
        for point in &run.evaluations {
            let Some(x) = point.clocks.value(clock).filter(|x| x.is_finite()) else {
                missing_clock = true;
                continue;
            };
            let mean = RunMean {
                x,
                mean: point.summary.map(|summary| summary.mean),
                starts_segment: missing_clock,
            };
            missing_clock = false;
            if let Some(previous) = curve.last_mut() {
                if previous.x > x {
                    return Ok(Vec::new());
                }
                if previous.x == x {
                    *previous = RunMean {
                        starts_segment: previous.starts_segment || mean.starts_segment,
                        ..mean
                    };
                    continue;
                }
            }
            curve.push(mean);
        }
        let mut valid = curve.iter().filter(|point| point.mean.is_some());
        let Some(first) = valid.next() else {
            return Ok(Vec::new());
        };
        start = start.max(first.x);
        end = end.min(valid.next_back().unwrap_or(first).x);
        curves.push(curve);
    }
    if curves.len() < 2 || start > end {
        return Ok(Vec::new());
    }
    let mut times: Vec<_> = curves
        .iter()
        .flatten()
        .filter(|point| point.mean.is_some() && (start..=end).contains(&point.x))
        .map(|point| point.x)
        .collect();
    times.sort_unstable_by(f64::total_cmp);
    times.dedup_by(|a, b| a == b);
    let mut points: Vec<TrainingAggregatePoint> = Vec::with_capacity(times.len());
    let mut means = Vec::with_capacity(curves.len());
    for x in times {
        means.clear();
        for curve in &curves {
            let Ok(index) = curve.binary_search_by(|point| point.x.total_cmp(&x)) else {
                let right = curve.partition_point(|point| point.x < x);
                let Some((lower, upper)) = right
                    .checked_sub(1)
                    .and_then(|left| curve.get(left).zip(curve.get(right)))
                else {
                    break;
                };
                let (Some(a), Some(b)) = (lower.mean, upper.mean) else {
                    break;
                };
                if upper.starts_segment {
                    break;
                }
                let fraction = (x - lower.x) / (upper.x - lower.x);
                means.push((1. - fraction) * a + fraction * b);
                continue;
            };
            let Some(mean) = curve[index].mean else {
                break;
            };
            means.push(mean);
        }
        if means.len() == curves.len() {
            let estimate = EmpiricalDistribution::new(&means)?.summary()?;
            let starts_segment = points.last().is_none_or(|previous| {
                curves.iter().any(|curve| {
                    let start = curve.partition_point(|point| point.x <= previous.x);
                    let end = curve.partition_point(|point| point.x <= x);
                    curve[start..end]
                        .iter()
                        .any(|point| point.mean.is_none() || point.starts_segment)
                })
            });
            points.push(TrainingAggregatePoint {
                x,
                mean: estimate.mean,
                standard_error: estimate.mean_standard_error,
                runs: estimate.count,
                starts_segment,
            });
        }
    }
    Ok(points)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    ScalarAbsolute,
    OrderedQuantileCrps,
}
#[derive(Clone, Debug)]
pub struct CriticError {
    pub kind: ErrorKind,
    pub value: f64,
    pub count: usize,
    pub row_id: u64,
    pub raw_crossings: usize,
}
pub fn independent_critic_error(
    critic: &Chunk,
    critic_probes: &Chunk,
    row: usize,
    sample: &Chunk,
    sample_probes: &Chunk,
    manifest: &RunManifest,
) -> Result<CriticError> {
    use rl::diagnostics::{
        analysis::match_conditional_rows,
        empirical, reference,
        value::{self, ValuePrediction},
    };
    for chunk in [critic, critic_probes, sample, sample_probes] {
        ensure(
            chunk.retained_charge() <= MAX_READ_BYTES,
            "independent critic evidence exceeds 4 MiB admission",
        )?;
    }
    for probes in [critic_probes, sample_probes] {
        let Record::ProbeSet(p) = &probes.record else {
            return Err(Error::Config(
                "independent critic error needs original probe definitions".into(),
            ));
        };
        ensure(
            p.histories.is_empty(),
            "History-conditioned critic inputs remain diagnostic; no certified observation-conditional error is available",
        )?;
    }
    let context = reference::manifest_context(manifest)?
        .ok_or_else(|| Error::Config("critic target scientific context is unknown".into()))?;
    let declared = value::declaration(critic)?
        .ok_or_else(|| Error::Config("critic scientific conventions are unknown".into()))?;
    declared.validate_manifest(manifest)?;
    ensure(
        declared.conventions.as_ref() == Some(&context.conventions),
        "critic conventions differ from the empirical target or are unknown",
    )?;
    ensure(
        declared
            .critics
            .values()
            .all(|c| c.learned_baseline != Some(false)),
        "untrained baseline is not an admitted learned critic",
    )?;
    let empirical_declaration = empirical::validate_for(sample, manifest)?;
    empirical::compatible_target(&empirical_declaration, &context)?;
    let matched =
        match_conditional_rows(critic, critic_probes, row, sample, sample_probes, manifest)?;
    let Record::ProbeSet(p) = &critic_probes.record else {
        unreachable!()
    };
    let ids = critic_probes.arrays[&p.row_ids].as_u64()?;
    ensure(
        ids.len() <= MAX_FIXED_ROWS,
        "independent critic population exceeds 1024 rows",
    )?;
    let Record::Empirical(e) = &sample.record else {
        unreachable!()
    };
    let distribution = EmpiricalDistribution::new(&sample.arrays[&e.returns].to_f64())?;
    let summary = distribution.summary()?;
    let prediction = ValuePrediction::from_critic(critic, matched.critic_row, None)?;
    let (kind,value,raw_crossings)=match prediction {
        ValuePrediction::Scalar(value)=>{
            ensure(summary.count >= 2, "Scalar conditional-mean reference requires at least two independent futures; singleton is a single observation")?;
            (ErrorKind::ScalarAbsolute,(value-summary.mean).abs(),0)
        }
        p@ValuePrediction::Quantiles { .. }=>{
            let q=p.ordered_quantiles()?;
            (ErrorKind::OrderedQuantileCrps,q.mean_crps(&distribution)?,q.source_crossing_count())
        }
        _=>return Err(Error::Config("learning critic error supports scalar absolute error and ordered quantile mean CRPS; this representation is unavailable".into())),
    };
    ensure(
        value.is_finite(),
        "learning independent critic score overflow",
    )?;
    Ok(CriticError {
        kind,
        value,
        count: summary.count,
        row_id: ids[row],
        raw_crossings,
    })
}
