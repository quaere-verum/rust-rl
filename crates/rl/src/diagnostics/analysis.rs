//! Distributional diagnostics in f64. Quantile outputs are never silently sorted.
//!
//! CRPS uses the loss convention E|X-y| - 0.5 E|X-X'| (smaller is better).
//! A finite quantile grid is approximated by atoms with the recorded bin masses;
//! its CRPS/Wasserstein scores are exact for that approximation, not for an
//! unobserved continuous distribution. See Gneiting & Raftery (2007),
//! <https://doi.org/10.1198/016214506000001437>.
use super::schema::*;
use crate::{Result, core::ensure};

#[derive(Debug, Clone)]
pub struct EmpiricalDistribution {
    sorted: Vec<f64>,
    prefix: Vec<f64>,
}
/// Sample uncertainty of the Monte Carlo mean, not uncertainty of a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalSummary {
    pub count: usize,
    pub mean: f64,
    pub sample_variance: Option<f64>,
    pub mean_standard_error: Option<f64>,
}
#[derive(Debug, Clone)]
pub struct Histogram {
    pub edges: Vec<f64>,
    pub counts: Vec<u64>,
}
impl EmpiricalDistribution {
    pub fn new(samples: &[f64]) -> Result<Self> {
        ensure(
            !samples.is_empty() && samples.iter().all(|x| x.is_finite()),
            "empirical returns must be nonempty and finite",
        )?;
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut prefix = Vec::with_capacity(sorted.len() + 1);
        prefix.push(0.);
        for &value in &sorted {
            prefix.push(prefix.last().copied().unwrap() + value);
        }
        ensure(
            prefix.iter().all(|v| v.is_finite()),
            "empirical return sum overflow",
        )?;
        Ok(Self { sorted, prefix })
    }
    pub fn sorted_values(&self) -> &[f64] {
        &self.sorted
    }
    pub fn mean(&self) -> f64 {
        self.prefix[self.sorted.len()] / self.sorted.len() as f64
    }
    /// Sample variance uses n-1. One future has no estimated variance or SE.
    /// Center before squaring to avoid cancellation in `E[G²] - E[G]²`.
    pub fn summary(&self) -> Result<EmpiricalSummary> {
        let count = self.sorted.len();
        let mean = self.mean();
        let (sample_variance, mean_standard_error) = if count > 1 {
            let mut squares = 0.;
            for &value in &self.sorted {
                let difference = value - mean;
                squares += difference * difference;
            }
            let variance = squares / (count - 1) as f64;
            let se = (variance / count as f64).sqrt();
            ensure(
                mean.is_finite() && variance.is_finite() && se.is_finite(),
                "empirical sample variance or standard error overflow",
            )?;
            (Some(variance), Some(se))
        } else {
            (None, None)
        };
        Ok(EmpiricalSummary {
            count,
            mean,
            sample_variance,
            mean_standard_error,
        })
    }
    /// Exact empirical step CDF. Equal returns form one atomic jump; both
    /// endpoints are retained, so consumers must not interpolate a density.
    pub fn cdf_points(&self) -> Result<Vec<[f64; 2]>> {
        let capacity = self
            .sorted
            .len()
            .checked_mul(2)
            .ok_or_else(|| crate::Error::Shape("empirical CDF capacity overflow".into()))?;
        let mut points = Vec::new();
        points.try_reserve_exact(capacity).map_err(|_| {
            crate::Error::Config("empirical CDF allocation exceeds capacity".into())
        })?;
        let mut first = 0;
        while first < self.sorted.len() {
            let value = self.sorted[first];
            let mut end = first + 1;
            while end < self.sorted.len() && self.sorted[end] == value {
                end += 1;
            }
            points.push([value, first as f64 / self.sorted.len() as f64]);
            points.push([value, end as f64 / self.sorted.len() as f64]);
            first = end;
        }
        Ok(points)
    }
    pub fn cdf(&self, value: f64) -> f64 {
        if value.is_nan() {
            return f64::NAN;
        }
        self.sorted.partition_point(|x| *x <= value) as f64 / self.sorted.len() as f64
    }
    /// Inverse empirical CDF (nearest rank). At p=0, returns the sample minimum;
    /// at p=.5 for even sample counts, returns the lower median.
    pub fn quantile(&self, p: f64) -> Result<f64> {
        ensure(
            p.is_finite() && (0. ..=1.).contains(&p),
            "invalid quantile probability",
        )?;
        Ok(self.sorted[((p * self.sorted.len() as f64).ceil() as usize)
            .saturating_sub(1)
            .min(self.sorted.len() - 1)])
    }
    /// Inverted two-sided DKW–Massart CDF band for i.i.d. complete futures.
    /// None denotes an unbounded tail, never the sample minimum/maximum.
    /// Coverage is simultaneous in probability for this one condition/sample.
    pub fn quantile_band(&self, p: f64, alpha: f64) -> Result<[Option<f64>; 2]> {
        ensure(
            p.is_finite() && (0. ..=1.).contains(&p),
            "invalid quantile probability",
        )?;
        ensure(
            alpha.is_finite() && alpha > 0. && alpha < 1.,
            "band alpha must be in (0,1)",
        )?;
        let epsilon = ((2_f64.ln() - alpha.ln()) / (2. * self.sorted.len() as f64)).sqrt();
        Ok([
            if p - epsilon <= 0. {
                None
            } else {
                Some(self.quantile(p - epsilon)?)
            },
            if p + epsilon > 1. {
                None
            } else {
                Some(self.quantile(p + epsilon)?)
            },
        ])
    }
    /// Mean of the worst alpha fraction of rewards, integrating fractional mass
    /// at the boundary. Larger is better; this is not upper-tail cost CVaR.
    pub fn lower_cvar(&self, alpha: f64) -> Result<f64> {
        tail_probability(alpha)?;
        let count = alpha * self.sorted.len() as f64;
        if count <= 1. {
            return Ok(self.sorted[0]);
        }
        let whole = (count.floor() as usize).min(self.sorted.len());
        let remainder = count - whole as f64;
        Ok((self.prefix[whole]
            + if whole < self.sorted.len() {
                remainder * self.sorted[whole]
            } else {
                0.
            })
            / count)
    }
    pub fn histogram(&self, bins: usize) -> Result<Histogram> {
        ensure(
            bins > 0 && bins <= 65_536,
            "histogram bin count outside supported bounds",
        )?;
        let mut low = self.sorted[0];
        let mut high = *self.sorted.last().unwrap();
        if low == high {
            let margin = (low.abs() * 1e-6).max(0.5);
            low -= margin;
            high += margin;
        }
        let span = high - low;
        ensure(span.is_finite() && span > 0., "histogram range overflow")?;
        let edges: Vec<_> = (0..=bins)
            .map(|i| low + span * (i as f64 / bins as f64))
            .collect();
        let mut counts = vec![0; bins];
        for &x in &self.sorted {
            let bin = (((x - low) / span * bins as f64).floor() as usize).min(bins - 1);
            counts[bin] += 1;
        }
        Ok(Histogram { edges, counts })
    }
    fn absolute_error(&self, value: f64) -> f64 {
        let i = self.sorted.partition_point(|&x| x <= value);
        (value * i as f64 - self.prefix[i] + self.prefix[self.sorted.len()]
            - self.prefix[i]
            - value * (self.sorted.len() - i) as f64)
            / self.sorted.len() as f64
    }
}
#[derive(Debug, Clone)]
pub struct QuantileForecast {
    atoms: AtomicForecast,
    locations: Vec<f64>,
    source_crossing_count: usize,
}

/// A discrete probability measure in increasing atom order. Zero masses are
/// admitted so fixed supports retain their declared atom locations.
#[derive(Debug, Clone)]
pub struct AtomicForecast {
    values: Vec<f64>,
    masses: Vec<f64>,
}
/// Validate borrowed heads without constructing, normalizing or ordering a
/// forecast. Return the original mass total for the owning constructor.
pub(super) fn validate_quantiles(values: &[f64], locations: &[f64], masses: &[f64]) -> Result<f64> {
    ensure(
        !values.is_empty()
            && values.len() == locations.len()
            && values.len() == masses.len()
            && values.iter().all(|v| v.is_finite())
            && locations
                .iter()
                .all(|p| p.is_finite() && *p > 0. && *p < 1.)
            && locations.windows(2).all(|p| p[0] < p[1])
            && masses.iter().all(|p| p.is_finite() && *p > 0.),
        "invalid quantile values, locations or masses",
    )?;
    let total: f64 = masses.iter().sum();
    ensure(
        (total - 1.).abs() <= 1e-6,
        "quantile probability masses must sum to one",
    )?;
    Ok(total)
}
impl QuantileForecast {
    pub fn new(values: Vec<f64>, locations: Vec<f64>, mut masses: Vec<f64>) -> Result<Self> {
        let total = validate_quantiles(&values, &locations, &masses)?;
        // Remove only floating-point grid summation drift, never missing mass.
        for mass in &mut masses {
            *mass /= total;
        }
        Ok(Self {
            source_crossing_count: values.windows(2).filter(|v| v[0] > v[1]).count(),
            atoms: AtomicForecast { values, masses },
            locations,
        })
    }
    /// Interpret one complete output row as an ordered quantile function.
    /// Grid locations and masses stay in probability order, independent of heads.
    pub fn ordered(values: Vec<f64>, locations: Vec<f64>, masses: Vec<f64>) -> Result<Self> {
        let mut forecast = Self::new(values, locations, masses)?;
        forecast.atoms.values.sort_unstable_by(f64::total_cmp);
        Ok(forecast)
    }
    /// Adjacent crossings in the original head order, before derived ordering.
    pub fn source_crossing_count(&self) -> usize {
        self.source_crossing_count
    }
    pub fn values(&self) -> &[f64] {
        self.atoms.values()
    }
    pub fn locations(&self) -> &[f64] {
        &self.locations
    }
    pub fn masses(&self) -> &[f64] {
        self.atoms.masses()
    }
    pub fn crossing_count(&self) -> usize {
        self.values().windows(2).filter(|v| v[0] > v[1]).count()
    }
    pub fn into_weighted_atoms(self) -> Result<AtomicForecast> {
        self.atoms.require_monotone()?;
        Ok(self.atoms)
    }
    pub fn mean(&self) -> f64 {
        self.atoms.mean()
    }
    pub fn quantile(&self, probability: f64) -> Result<f64> {
        self.atoms.quantile(probability)
    }
    pub fn lower_cvar(&self, alpha: f64) -> Result<f64> {
        self.atoms.lower_cvar(alpha)
    }
    pub fn cdf_points(&self) -> Result<Vec<[f64; 2]>> {
        self.atoms.cdf_points()
    }
    pub fn crps(&self, outcome: f64) -> Result<f64> {
        self.atoms.crps(outcome)
    }
    pub fn mean_crps(&self, empirical: &EmpiricalDistribution) -> Result<f64> {
        self.atoms.mean_crps(empirical)
    }
    pub fn mean_crps_against(&self, reference: &Self) -> Result<f64> {
        self.atoms.mean_crps_against(&reference.atoms)
    }
    pub fn wasserstein1(&self, empirical: &EmpiricalDistribution) -> Result<f64> {
        self.atoms.wasserstein1(empirical)
    }
}

impl AtomicForecast {
    pub fn new(values: Vec<f64>, mut masses: Vec<f64>) -> Result<Self> {
        let total: f64 = masses.iter().sum();
        ensure(
            !values.is_empty()
                && values.len() == masses.len()
                && values.iter().all(|value| value.is_finite())
                && masses.iter().all(|mass| mass.is_finite() && *mass >= 0.)
                && (total - 1.).abs() <= 1e-6,
            "invalid weighted atomic probability measure",
        )?;
        for mass in &mut masses {
            *mass /= total;
        }
        let forecast = Self { values, masses };
        forecast.require_monotone()?;
        Ok(forecast)
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }
    fn require_monotone(&self) -> Result<()> {
        ensure(
            self.values.windows(2).all(|pair| pair[0] <= pair[1]),
            "crossing quantiles cannot define a CDF or tail distribution; raw outputs are preserved",
        )
    }
    pub fn mean(&self) -> f64 {
        self.values
            .iter()
            .zip(&self.masses)
            .map(|(v, p)| v * p)
            .sum()
    }
    pub fn variance(&self) -> Result<f64> {
        let mean = self.mean();
        finite(
            self.values
                .iter()
                .zip(&self.masses)
                .filter(|(_, mass)| **mass > 0.)
                .map(|(value, mass)| (value - mean).powi(2) * mass)
                .sum(),
        )
    }
    /// Weighted atomic generalized inverse. An exact cumulative-mass boundary
    /// selects its left atom; probabilities zero and one select the extrema
    /// of positive mass.
    pub fn quantile(&self, probability: f64) -> Result<f64> {
        ensure(
            probability.is_finite() && (0. ..=1.).contains(&probability),
            "quantile probability must be in [0,1]",
        )?;
        self.require_monotone()?;
        if probability == 0. || probability == 1. {
            let index = if probability == 0. {
                self.masses.iter().position(|mass| *mass > 0.)
            } else {
                self.masses.iter().rposition(|mass| *mass > 0.)
            };
            return Ok(self.values[index.expect("validated positive mass")]);
        }
        let mut cumulative = 0.;
        for (&value, &mass) in self.values.iter().zip(&self.masses) {
            cumulative += mass;
            if probability <= cumulative {
                return Ok(value);
            }
        }
        self.quantile(1.)
    }
    pub fn lower_cvar(&self, alpha: f64) -> Result<f64> {
        tail_probability(alpha)?;
        self.require_monotone()?;
        if alpha <= self.masses[0] {
            return Ok(self.values[0]);
        }
        let mut remaining = alpha;
        let mut total = 0.;
        for (&value, &mass) in self.values.iter().zip(&self.masses) {
            let take = remaining.min(mass).max(0.);
            total += take * value;
            remaining -= take;
        }
        finite(total / alpha)
    }
    pub fn cdf_points(&self) -> Result<Vec<[f64; 2]>> {
        self.require_monotone()?;
        let mut cumulative = 0.;
        let mut points = Vec::with_capacity(self.values.len() * 2);
        for (&value, &mass) in self.values.iter().zip(&self.masses) {
            if mass == 0. {
                continue;
            }
            points.push([value, cumulative]);
            cumulative += mass;
            points.push([value, cumulative.min(1.)]);
        }
        Ok(points)
    }
    fn half_pairwise_distance(&self) -> f64 {
        let mut mass = 0.;
        let mut weighted = 0.;
        let mut total = 0.;
        for (&v, &p) in self.values.iter().zip(&self.masses) {
            total += p * (v * mass - weighted);
            mass += p;
            weighted += p * v;
        }
        total
    }
    pub fn crps(&self, outcome: f64) -> Result<f64> {
        self.require_monotone()?;
        ensure(outcome.is_finite(), "nonfinite CRPS outcome")?;
        let first: f64 = self
            .values
            .iter()
            .zip(&self.masses)
            .map(|(&v, &p)| p * (v - outcome).abs())
            .sum();
        finite((first - self.half_pairwise_distance()).max(0.))
    }
    /// O(Q log N) after the empirical distribution's O(N log N) preparation.
    pub fn mean_crps(&self, empirical: &EmpiricalDistribution) -> Result<f64> {
        self.require_monotone()?;
        let first: f64 = self
            .values
            .iter()
            .zip(&self.masses)
            .map(|(&v, &p)| p * empirical.absolute_error(v))
            .sum();
        finite((first - self.half_pairwise_distance()).max(0.))
    }
    /// Expected CRPS of this forecast when outcomes follow the reference's
    /// weighted atom distribution. The score is asymmetric and includes the
    /// reference distribution's irreducible uncertainty. O(Q+R) time, O(1)
    /// additional space.
    pub fn mean_crps_against(&self, reference: &Self) -> Result<f64> {
        self.require_monotone()?;
        reference.require_monotone()?;
        let mut previous = self.values[0].min(reference.values[0]);
        let mut forecast_atoms = self
            .values
            .iter()
            .copied()
            .zip(self.masses.iter().copied())
            .peekable();
        let mut reference_atoms = reference
            .values
            .iter()
            .copied()
            .zip(reference.masses.iter().copied())
            .peekable();
        let (mut forecast_cdf, mut reference_cdf) = (0.0_f64, 0.0_f64);
        let mut score = 0.;
        loop {
            let next = match (forecast_atoms.peek(), reference_atoms.peek()) {
                (Some((a, _)), Some((b, _))) => a.min(*b),
                (Some((value, _)), None) | (None, Some((value, _))) => *value,
                (None, None) => break,
            };
            // E[(F - 1{Y <= x})²] = (F-G)² + G(1-G) on each interval.
            let difference = forecast_cdf - reference_cdf;
            let weight = difference * difference + reference_cdf * (1. - reference_cdf);
            if weight > 0. {
                let width = next - previous;
                score += if width.is_finite() {
                    width * weight
                } else {
                    // An opposite-sign span can overflow even when the
                    // weighted interval fits in f64.
                    next * weight - previous * weight
                };
            }
            for (atoms, cdf) in [
                (&mut forecast_atoms, &mut forecast_cdf),
                (&mut reference_atoms, &mut reference_cdf),
            ] {
                while let Some((_, mass)) = atoms.next_if(|(value, _)| *value == next) {
                    *cdf = (*cdf + mass).min(1.);
                }
                if atoms.peek().is_none() {
                    *cdf = 1.;
                }
            }
            previous = next;
        }
        finite(score)
    }
    /// Integral of absolute inverse-CDF difference, exact for the weighted
    /// quantile-atom approximation and empirical distribution. O(Q+N).
    pub fn wasserstein1(&self, empirical: &EmpiricalDistribution) -> Result<f64> {
        self.require_monotone()?;
        let (mut i, mut j) = (0, 0);
        let mut a_end = self.masses[0];
        let mut previous = 0.;
        let mut distance = 0.;
        while i < self.values.len() && j < empirical.sorted.len() {
            let b_end = (j + 1) as f64 / empirical.sorted.len() as f64;
            let end = a_end.min(b_end).min(1.);
            distance += (end - previous).max(0.) * (self.values[i] - empirical.sorted[j]).abs();
            previous = end;
            if a_end <= b_end {
                i += 1;
                if i < self.values.len() {
                    a_end = if i + 1 == self.values.len() {
                        1.
                    } else {
                        a_end + self.masses[i]
                    };
                }
            } else {
                j += 1;
            }
        }
        finite(distance)
    }
}

impl QuantileForecast {
    /// Coverage of each stored predicted quantile (raw or explicitly ordered).
    /// Ties use <=. With atoms, correct quantiles can exceed nominal coverage.
    pub fn coverage(&self, empirical: &EmpiricalDistribution) -> Vec<f64> {
        self.values().iter().map(|&v| empirical.cdf(v)).collect()
    }
    pub fn strict_coverage(&self, empirical: &EmpiricalDistribution) -> Vec<f64> {
        self.values()
            .iter()
            .map(|v| {
                empirical.sorted.partition_point(|x| x < v) as f64 / empirical.sorted.len() as f64
            })
            .collect()
    }
    /// The empirical strict/weak coverage interval for each nominal quantile.
    pub fn coverage_intervals(&self, empirical: &EmpiricalDistribution) -> Vec<[f64; 2]> {
        self.strict_coverage(empirical)
            .into_iter()
            .zip(self.coverage(empirical))
            .map(|(strict, weak)| [strict, weak])
            .collect()
    }
    /// Unweighted mean distance of nominal probabilities to their empirical
    /// strict/weak coverage intervals. Finite-sample sampling error remains.
    pub fn coverage_interval_error(&self, empirical: &EmpiricalDistribution) -> f64 {
        self.coverage_intervals(empirical)
            .iter()
            .zip(&self.locations)
            .map(|(&[low, high], &tau)| (low - tau).max(tau - high).max(0.))
            .sum::<f64>()
            / self.values().len() as f64
    }
}
fn tail_probability(alpha: f64) -> Result<()> {
    ensure(
        alpha.is_finite() && alpha > 0. && alpha <= 1.,
        "tail probability must be in (0,1]",
    )
}
fn finite(value: f64) -> Result<f64> {
    ensure(value.is_finite(), "distributional calculation overflow")?;
    Ok(value)
}
/// Returns a warning, not a fabricated distribution or a silently clipped
/// variance. Tolerance is relative to max(1, |m2|, mean²).
pub fn moment_consistency(mean: f64, raw_second: f64, tolerance: f64) -> Result<Option<String>> {
    ensure(
        mean.is_finite() && raw_second.is_finite() && tolerance.is_finite() && tolerance >= 0.,
        "invalid moment diagnostics",
    )?;
    let square = mean * mean;
    ensure(square.is_finite(), "moment square overflow")?;
    let variance = raw_second - square;
    Ok(
        (variance < -tolerance * raw_second.abs().max(square).max(1.))
            .then(|| format!("raw second moment implies negative variance {variance}")),
    )
}

#[derive(Debug, Clone)]
pub struct ConditionalMatch {
    pub probe_row: usize,
    pub conditioning_label: String,
}
/// Structural preflight before labelling a comparison conditional. It verifies
/// recorded inputs, actions, masks, policy, revision, discount and full returns.
/// It cannot certify the truth of externally imported policy/methodology labels.
/// Full-hidden-state MC is explicitly NOT observation-conditioned calibration.
pub fn match_conditional(
    critic_chunk: &Chunk,
    probe_chunk: &Chunk,
    empirical_chunk: &Chunk,
    manifest: &RunManifest,
) -> Result<ConditionalMatch> {
    // This historical API has no selected action column in its result. Do not
    // let old scalar consumers newly interpret an entire Q vector as a pair.
    ensure(
        !matches!(&critic_chunk.record, Record::Critic(c) if c.action_condition == ActionCondition::AllDiscreteActions),
        "all-action critics require explicit column matching",
    )?;
    let Record::Empirical(empirical) = &empirical_chunk.record else {
        return Err(crate::Error::Config("expected empirical record".into()));
    };
    let row = empirical_row(empirical)?;
    let matched = match_conditional_rows(
        critic_chunk,
        probe_chunk,
        row,
        empirical_chunk,
        probe_chunk,
        manifest,
    )?;
    Ok(ConditionalMatch {
        probe_row: matched.critic_row,
        conditioning_label: matched.conditioning_label,
    })
}

#[derive(Debug, Clone)]
pub struct ConditionalRowMatch {
    pub critic_row: usize,
    pub empirical_row: usize,
    /// Present only for an all-discrete-action Q matrix. Select this one column.
    pub discrete_action: Option<usize>,
    pub conditioning_label: String,
}

fn empirical_row(empirical: &EmpiricalRecord) -> Result<usize> {
    usize::try_from(empirical.probe_row.ok_or_else(|| {
        crate::Error::Config("conditional empirical returns lack a probe row".into())
    })?)
    .map_err(|_| crate::Error::Shape("probe row overflow".into()))
}

/// Validate selected rows of two original immutable definitions. Storage callers
/// must bind each definition via its record position (`probe_for_record`).
pub fn match_conditional_rows(
    critic_chunk: &Chunk,
    critic_probe_chunk: &Chunk,
    critic_row: usize,
    empirical_chunk: &Chunk,
    probe_chunk: &Chunk,
    manifest: &RunManifest,
) -> Result<ConditionalRowMatch> {
    critic_chunk.validate()?;
    critic_probe_chunk.validate()?;
    probe_chunk.validate()?;
    empirical_chunk.validate()?;
    ensure(
        super::value::return_target(critic_chunk)?
            .is_none_or(super::value::ReturnTarget::is_environmental),
        "environmental empirical returns cannot calibrate an entropy-regularized or shaped critic",
    )?;
    let (Record::Critic(critic), Record::ProbeSet(probes), Record::Empirical(empirical)) = (
        &critic_chunk.record,
        &probe_chunk.record,
        &empirical_chunk.record,
    ) else {
        return Err(crate::Error::Config(
            "conditional comparison requires critic, probe set and empirical records".into(),
        ));
    };
    let Record::ProbeSet(critic_probes) = &critic_probe_chunk.record else {
        return Err(crate::Error::Config("expected critic probe set".into()));
    };
    ensure(
        critic_chunk.context.identity.run_id == manifest.run_id
            && empirical_chunk.context.identity.run_id == manifest.run_id
            && probe_chunk.context.identity.run_id == manifest.run_id
            && critic_probe_chunk.context.identity.run_id == manifest.run_id
            && critic_chunk.context.identity.learner_update_revision
                == empirical_chunk.context.identity.learner_update_revision,
        "conditional comparison run/revision mismatch",
    )?;
    ensure(
        critic_chunk.context.probe_set.as_deref() == Some(&critic_probes.id)
            && critic_probes.id == probes.id
            && empirical_chunk.context.probe_set.as_deref() == Some(&probes.id),
        "conditional comparison probe-set mismatch",
    )?;
    ensure(
        critic_chunk.context.identity.model_role == empirical_chunk.context.identity.model_role
            && critic_chunk.context.identity.checkpoint_id
                == empirical_chunk.context.identity.checkpoint_id,
        "conditional comparison model role/checkpoint identity mismatch",
    )?;
    ensure(
        critic.continuation_policy.as_deref() != Some("optimality"),
        "policy-continuation empirical returns cannot calibrate a Bellman optimality critic",
    )?;
    ensure(
        critic.continuation_policy.as_deref() == Some(&empirical.evaluation.continuation_policy),
        "conditional comparison needs the same explicitly identified continuation policy",
    )?;
    ensure(
        critic.continuation_stochastic == Some(empirical.evaluation.stochastic_policy),
        "conditional comparison policy sampling modes differ or are unspecified",
    )?;
    ensure(
        empirical.evaluation.complete_returns && empirical.evaluation.independent_future_streams,
        "conditional comparison requires complete returns with independent future streams",
    )?;
    ensure(
        (empirical.evaluation.discount - manifest.discount).abs() <= 1e-8,
        "conditional comparison discount mismatch",
    )?;
    let row = empirical_row(empirical)?;
    let cids = critic_probe_chunk.arrays[&critic_probes.row_ids].as_u64()?;
    let eids = probe_chunk.arrays[&probes.row_ids].as_u64()?;
    ensure(
        critic_row < cids.len() && row < eids.len(),
        "conditional row out of range",
    )?;
    ensure(
        cids[critic_row] == eids[row]
            && critic_probes.population == probes.population
            && critic_probes.methodology == probes.methodology,
        "conditional probe row identity/population/methodology mismatch",
    )?;
    let equal = super::representations::equal_row;
    ensure(
        equal(
            &critic_probe_chunk.arrays[&critic_probes.observations],
            &probe_chunk.arrays[&probes.observations],
            critic_row,
            row,
        ),
        "conditional probe observation mismatch",
    )?;
    ensure(
        masks_equal(
            critic_probes
                .action_masks
                .as_ref()
                .map(|k| &critic_probe_chunk.arrays[k]),
            critic_row,
            probes.action_masks.as_ref().map(|k| &probe_chunk.arrays[k]),
            row,
        ),
        "conditional probe masks mismatch",
    )?;
    match (&critic_probes.fixed_actions, &probes.fixed_actions) {
        (Some(a), Some(b)) => ensure(
            equal(
                &critic_probe_chunk.arrays[a],
                &probe_chunk.arrays[b],
                critic_row,
                row,
            ),
            "conditional fixed probe actions mismatch",
        )?,
        (None, None) => {}
        _ => {
            return Err(crate::Error::Config(
                "conditional fixed action presence mismatch".into(),
            ));
        }
    }
    ensure(
        critic_probes.histories.keys().eq(probes.histories.keys())
            && critic_probes.histories.iter().all(|(name, k)| {
                equal(
                    &critic_probe_chunk.arrays[k],
                    &probe_chunk.arrays[&probes.histories[name]],
                    critic_row,
                    row,
                )
            }),
        "conditional probe histories mismatch",
    )?;
    let output = match &critic.representation {
        CriticRepresentation::Scalar { values }
        | CriticRepresentation::Discrete { values }
        | CriticRepresentation::Quantiles { values, .. }
        | CriticRepresentation::CategoricalRegression { values, .. } => values,
        CriticRepresentation::CategoricalReturn { probabilities, .. } => probabilities,
        CriticRepresentation::RawMoments { mean, .. } => mean,
    };
    ensure(
        critic_chunk.arrays[output].shape()[0] == cids.len(),
        "critic/probe population mismatch",
    )?;
    let obs = empirical.conditioning_observation.as_ref().ok_or_else(|| {
        crate::Error::Config("conditional returns lack the initial observation".into())
    })?;
    ensure(
        equal_row(
            &probe_chunk.arrays[&probes.observations],
            row,
            &empirical_chunk.arrays[obs],
            0,
        )?,
        "conditional comparison observation mismatch",
    )?;
    match (&probes.action_masks, &empirical.conditioning_mask) {
        (Some(a), Some(b)) => ensure(
            equal_row(&probe_chunk.arrays[a], row, &empirical_chunk.arrays[b], 0)?,
            "conditional comparison mask mismatch",
        )?,
        (None, None) => {}
        (Some(a), None) => ensure(
            row_values(&probe_chunk.arrays[a], row)?
                .iter()
                .all(|&x| x == 1.),
            "conditional comparison mask mismatch",
        )?,
        (None, Some(b)) => ensure(
            row_values(&empirical_chunk.arrays[b], 0)?
                .iter()
                .all(|&x| x == 1.),
            "conditional comparison mask mismatch",
        )?,
    }
    let mut discrete_action = None;
    if critic.action_condition == ActionCondition::AllDiscreteActions {
        let (
            CriticRepresentation::Discrete { values },
            crate::core::action::ActionSpace::Discrete { count },
        ) = (&critic.representation, &manifest.action_space)
        else {
            return Err(crate::Error::Config(
                "all-action pairing requires discrete Q values and action space".into(),
            ));
        };
        ensure(
            critic_chunk.arrays[values].shape()[1] == *count,
            "Q width differs from action count",
        )?;
        let first = empirical
            .initial_action
            .as_ref()
            .ok_or_else(|| crate::Error::Config("discrete comparison lacks first action".into()))?;
        let first = &empirical_chunk.arrays[first];
        ensure(
            empirical.evaluation.initial_action == ActionCondition::Fixed
                && first.dtype() == DType::U64
                && first.shape() == [1],
            "discrete comparison requires fixed canonical u64[1] first action",
        )?;
        let action = usize::try_from(first.as_u64()?[0])
            .map_err(|_| crate::Error::Shape("discrete action overflow".into()))?;
        ensure(action < *count, "discrete first action out of range")?;
        for (chunk, mask, row) in [
            (critic_probe_chunk, &critic_probes.action_masks, critic_row),
            (probe_chunk, &probes.action_masks, row),
            (empirical_chunk, &empirical.conditioning_mask, 0),
        ] {
            if let Some(key) = mask {
                let mask = &chunk.arrays[key];
                ensure(
                    mask.shape().get(1) == Some(count) && mask.as_u8()?[row * count + action] == 1,
                    "discrete first action masked or mask width differs",
                )?;
            }
        }
        if let Some(key) = &critic_probes.fixed_actions {
            ensure(
                equal(&critic_probe_chunk.arrays[key], first, critic_row, 0),
                "discrete first action differs from fixed probe action",
            )?;
        }
        discrete_action = Some(action);
    } else if critic.action_condition == ActionCondition::StateValue {
        ensure(
            empirical.initial_action.is_none()
                && empirical.evaluation.initial_action == ActionCondition::CurrentPolicy,
            "state-value comparison requires sampling the first action from the continuation policy",
        )?;
    } else {
        let action = critic.actions.as_ref().ok_or_else(|| {
            crate::Error::Config("critic comparison lacks evaluated actions".into())
        })?;
        let first = empirical.initial_action.as_ref().ok_or_else(|| {
            crate::Error::Config(
                "action-value comparison requires the same fixed first action".into(),
            )
        })?;
        if let Some(fixed) = &critic_probes.fixed_actions {
            ensure(
                equal(
                    &critic_probe_chunk.arrays[fixed],
                    &empirical_chunk.arrays[first],
                    critic_row,
                    0,
                ),
                "conditional first action differs from fixed probe action",
            )?;
        }
        ensure(
            empirical.evaluation.initial_action == ActionCondition::Fixed
                && equal_row(
                    &critic_chunk.arrays[action],
                    critic_row,
                    &empirical_chunk.arrays[first],
                    0,
                )?,
            "conditional comparison first-action mismatch",
        )?;
    }
    let conditioning_label = match &empirical.evaluation.conditioning {
        Conditioning::InitialScenario => return Err(crate::Error::Config("unconditional scenario returns are not state-conditioned calibration".into())),
        Conditioning::FullSimulatorState { description } => format!("Conditional on full simulator state: {description}. This is not an observation-posterior experiment."),
        Conditioning::Observation { methodology } => format!("Conditional on observation: {methodology}"),
        Conditioning::History { .. } => return Err(crate::Error::Config("automatic history-conditioned calibration needs matched history payloads; imported histories remain viewable".into())),
    };
    Ok(ConditionalRowMatch {
        critic_row,
        empirical_row: row,
        discrete_action,
        conditioning_label,
    })
}
fn row_values(array: &Array, row: usize) -> Result<Vec<f64>> {
    ensure(
        array.shape().len() <= 2 && row < array.shape()[0],
        "diagnostic comparison row out of range",
    )?;
    let width = array.shape().get(1).copied().unwrap_or(1);
    let start = row * width;
    Ok(match &array.values {
        Values::F32(v) => v[start..start + width].iter().map(|&v| v as f64).collect(),
        Values::F64(v) => v[start..start + width].to_vec(),
        Values::U64(v) => {
            ensure(
                v[start..start + width].iter().all(|&v| v <= 1u64 << 53),
                "integer comparison exceeds exact f64 range",
            )?;
            v[start..start + width].iter().map(|&v| v as f64).collect()
        }
        Values::U8(v) => v[start..start + width].iter().map(|&v| v as f64).collect(),
    })
}
fn equal_row(a: &Array, i: usize, b: &Array, j: usize) -> Result<bool> {
    Ok(super::representations::equal_row(a, b, i, j))
}
fn masks_equal(a: Option<&Array>, i: usize, b: Option<&Array>, j: usize) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => super::representations::equal_row(a, b, i, j),
        (None, None) => true,
        (Some(a), None) | (None, Some(a)) => {
            let row = if b.is_none() { i } else { j };
            let width = a.shape()[1];
            a.as_u8()
                .is_ok_and(|v| v[row * width..(row + 1) * width].iter().all(|&v| v == 1))
        }
    }
}
