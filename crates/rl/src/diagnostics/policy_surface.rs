//! Provider-independent policy evaluation. Complete typed arrays retain their
//! original precision; projections are actual-action expectations in f64.
//! No GUI, checkpoint restoration, filesystem, or learner state is involved.
#[cfg(feature = "neural")]
use super::{
    models::{InspectionRequest, ModelBundle},
    policy_slice::GridEligibility,
    visualization::EnvironmentSemantics,
};
use super::{
    policy_batch::{PolicyBatch, array, numeric_row},
    policy_numerics::{self, SelectedDistribution},
    policy_slice::PolicySliceSpec,
    schema::{Chunk, Record},
};
use crate::{
    Error, Result,
    core::{action::ActionSpace, ensure, env::Observation},
};

pub const POLICY_SURFACE_VERSION: u32 = 1;
pub const POLICY_NUMERICAL_CONVENTION: &str = "actual-action-expectation/f64/v1";
pub const EVALUATION_FIXED_SCRATCH: usize = 64 * 1024;

/// Replace this adapter with a projection-only backend without changing callers
/// or the checked numerical representation. Full inspection is internal today.
#[derive(Debug, Clone)]
pub struct EvaluatedPolicyBatch {
    pub probes: Chunk,
    pub policy: Chunk,
}
impl EvaluatedPolicyBatch {
    pub fn checked<'a>(&'a self, space: &'a ActionSpace) -> Result<PolicyBatch<'a>> {
        PolicyBatch::new(&self.policy, Some(&self.probes), space)
    }
}
#[cfg(feature = "neural")]
pub fn evaluate_policy_batch(
    model: &ModelBundle,
    policy: &str,
    observations: &[Observation],
) -> Result<EvaluatedPolicyBatch> {
    let result = model.inspect(&InspectionRequest::new(policy, observations.to_vec()))?;
    Ok(EvaluatedPolicyBatch {
        probes: result.probes,
        policy: result.policy,
    })
}

/// Independent offline limits, never an increase to Interactive admission.
/// Restored parameters and backend intermediates are separately bounded by
/// `ModelLimits`; scratch includes inspection copies, masks and checked readers.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceLimits {
    pub max_batch: usize,
    pub max_surface_bytes: usize,
    pub max_scratch_bytes: usize,
}
impl Default for SurfaceLimits {
    fn default() -> Self {
        Self {
            max_batch: 1024,
            max_surface_bytes: 128 * 1024 * 1024,
            max_scratch_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Conservative simultaneous-owner allowance shared with Interactive. Backend
/// tensor admission remains ModelBundle's responsibility, independent of this.
pub fn batch_capacity(
    observations: usize,
    actions: usize,
    reference: bool,
    scratch_bytes: usize,
    max_batch: usize,
) -> Result<usize> {
    let endpoints = if reference { 2usize } else { 1 };
    let row = observations
        .checked_mul(64)
        .and_then(|n| actions.checked_mul(256).and_then(|a| n.checked_add(a)))
        .and_then(|n| n.checked_add(1024))
        .and_then(|n| n.checked_mul(endpoints))
        .ok_or_else(|| Error::Config("Policy input/action scratch overflow".into()))?;
    let count = scratch_bytes.saturating_sub(EVALUATION_FIXED_SCRATCH) / row;
    ensure(
        count > 0 && max_batch > 0,
        "Policy input/action row exceeds scratch budget or batch limit is zero",
    )?;
    Ok(count.min(max_batch))
}
pub fn action_count(space: &ActionSpace) -> usize {
    match space {
        ActionSpace::Discrete { count } => *count,
        s => s.dimensions(),
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceBatch {
    pub(crate) first_row: usize,
    pub(crate) evaluated: EvaluatedPolicyBatch,
    /// Row-major, all action components. Deterministic discrete values are
    /// explicitly indicators; their policy parameters remain Deterministic.
    pub(crate) projected: Vec<f64>,
    pub(crate) rows: usize,
}
impl SurfaceBatch {
    pub fn from_evaluation(
        first_row: usize,
        evaluated: EvaluatedPolicyBatch,
        space: &ActionSpace,
    ) -> Result<Self> {
        let checked = evaluated.checked(space)?;
        let rows = checked.rows();
        let width = action_count(space);
        let count = rows
            .checked_mul(width)
            .ok_or_else(|| Error::Shape("policy surface dimensions overflow".into()))?;
        let mut projected = Vec::with_capacity(count);
        for row in 0..rows {
            for action in 0..width {
                projected.push(policy_numerics::project(&checked, row, action)?);
            }
        }
        Ok(Self {
            first_row,
            evaluated,
            projected,
            rows,
        })
    }
    pub fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.evaluated.probes.retained_charge())
            .saturating_add(self.evaluated.policy.retained_charge())
            .saturating_add(self.projected.capacity().saturating_mul(8))
    }
}

/// Immutable complete numerical frame, independent of worker leases and render
/// colors. Each batch retains exact evaluated inputs, masks and full policy
/// parameters, including the original context/checkpoint provenance. Persistent
/// exact source/recording identity is supplied by the cache envelope.
#[derive(Debug, Clone)]
pub struct PolicySurface {
    pub(crate) version: u32,
    pub(crate) spec: PolicySliceSpec,
    pub(crate) action_space: ActionSpace,
    pub(crate) batches: Vec<SurfaceBatch>,
    pub(crate) rows: usize,
}
impl PolicySurface {
    pub fn spec(&self) -> &PolicySliceSpec {
        &self.spec
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn action_count(&self) -> usize {
        action_count(&self.action_space)
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.spec.fixed.capacity().saturating_mul(4))
            .saturating_add(self.spec.ranges.capacity().saturating_mul(8))
            .saturating_add(
                self.spec
                    .range_sources
                    .capacity()
                    .saturating_mul(std::mem::size_of::<super::policy_slice::RangeSource>()),
            )
            .saturating_add(self.spec.action_mask.as_ref().map_or(0, Vec::capacity))
            .saturating_add(self.spec.model_role.capacity())
            .saturating_add(self.spec.policy.capacity())
            .saturating_add(match &self.action_space {
                ActionSpace::Box { low, high } => {
                    (low.capacity() + high.capacity()).saturating_mul(4)
                }
                _ => 0,
            })
            .saturating_add(
                self.batches
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SurfaceBatch>()),
            )
            .saturating_add(
                self.batches
                    .iter()
                    .map(SurfaceBatch::owned_bytes)
                    .sum::<usize>(),
            )
    }
    fn locate(&self, row: usize) -> Result<(&SurfaceBatch, usize)> {
        ensure(row < self.rows, "row outside policy surface")?;
        let i = self.batches.partition_point(|b| b.first_row <= row);
        let batch = self
            .batches
            .get(i.saturating_sub(1))
            .ok_or_else(|| Error::State("missing policy surface batch".into()))?;
        let local = row - batch.first_row;
        ensure(local < batch.rows, "row outside policy surface batch")?;
        Ok((batch, local))
    }
    pub fn value(&self, row: usize, action: usize) -> Result<f64> {
        ensure(
            action < self.action_count(),
            "action outside policy surface",
        )?;
        let (batch, row) = self.locate(row)?;
        Ok(batch.projected[row * self.action_count() + action])
    }
    pub fn distribution(&self, row: usize, action: usize) -> Result<SelectedDistribution> {
        let (batch, row) = self.locate(row)?;
        policy_numerics::selected_distribution(
            &batch.evaluated.checked(&self.action_space)?,
            row,
            action,
        )
    }
    pub fn checked_row(&self, row: usize) -> Result<(PolicyBatch<'_>, usize)> {
        let (batch, row) = self.locate(row)?;
        Ok((batch.evaluated.checked(&self.action_space)?, row))
    }
    pub fn input(&self, row: usize) -> Result<Observation> {
        let (batch, row) = self.locate(row)?;
        let probes = &batch.evaluated.probes;
        let Record::ProbeSet(meta) = &probes.record else {
            return Err(Error::State("missing policy surface inputs".into()));
        };
        Ok(Observation::dense(
            numeric_row(array(probes, &meta.observations)?, row)?
                .into_iter()
                .map(|v| v as f32)
                .collect(),
            // Inspection records contain effective all-true support even for
            // unconditioned discrete inputs. The complete slice preserves the
            // original conditioning; do not relabel it as an explicit mask.
            self.spec.action_mask.clone(),
        ))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EvaluationProgress {
    pub completed_rows: usize,
    pub total_rows: usize,
    pub retained_bytes: usize,
}

#[cfg(feature = "neural")]
pub fn evaluate_surface(
    model: &ModelBundle,
    spec: &PolicySliceSpec,
    environment: EnvironmentSemantics<'_>,
    limits: SurfaceLimits,
    cancelled: impl Fn() -> bool,
    mut progress: impl FnMut(EvaluationProgress),
) -> Result<PolicySurface> {
    let config = spec.to_config(environment)?;
    ensure(
        config.grid_eligibility() == GridEligibility::ContinuousFeedforward && !config.is_empty(),
        "offline synthetic slice unavailable for masked, history-bound or samples-only inputs",
    )?;
    let model_config = model.configuration();
    ensure(
        model_config.observation_dim == environment.observation_dim()
            && &model_config.action_space == environment.action_space(),
        "model/slice encoding differs",
    )?;
    ensure(
        model.context().identity.model_role == spec.model_role,
        "model/slice role differs",
    )?;
    let capacity = batch_capacity(
        config.fixed.len(),
        action_count(environment.action_space()),
        false,
        limits.max_scratch_bytes,
        limits.max_batch.min(model.limits().max_batch),
    )?;
    let rows = config.len()?;
    let batches = rows.div_ceil(capacity);
    // Admit a conservative retained-data bound before inference. This covers
    // all typed arrays and projections plus per-record container metadata.
    let estimate = config
        .fixed
        .len()
        .checked_mul(8)
        .and_then(|n| {
            action_count(environment.action_space())
                .checked_mul(64)
                .and_then(|a| n.checked_add(a))
        })
        .and_then(|n| n.checked_add(128))
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| {
            batches
                .checked_mul(16 * 1024)
                .and_then(|b| n.checked_add(b))
        })
        .and_then(|n| {
            n.checked_add(config.owned_bytes())
                .and_then(|n| n.checked_add(4096))
        });
    ensure(
        estimate.is_some_and(|b| b <= limits.max_surface_bytes),
        "policy surface exceeds retained byte budget",
    )?;
    let mut surface = PolicySurface {
        version: POLICY_SURFACE_VERSION,
        spec: spec.clone(),
        action_space: environment.action_space().clone(),
        batches: Vec::with_capacity(batches),
        rows,
    };
    let mut retained_bytes = surface.owned_bytes();
    for first in (0..rows).step_by(capacity) {
        ensure(!cancelled(), "policy surface evaluation cancelled")?;
        let end = (first + capacity).min(rows);
        let observations = (first..end)
            .map(|i| config.observation_at(i))
            .collect::<Result<Vec<_>>>()?;
        let evaluated = evaluate_policy_batch(model, &spec.policy, &observations)?;
        let batch = SurfaceBatch::from_evaluation(first, evaluated, &surface.action_space)?;
        retained_bytes = retained_bytes.saturating_add(batch.owned_bytes());
        ensure(
            retained_bytes <= limits.max_surface_bytes,
            "policy surface retained capacities exceed byte budget",
        )?;
        surface.batches.push(batch);
        ensure(!cancelled(), "policy surface evaluation cancelled")?;
        progress(EvaluationProgress {
            completed_rows: end,
            total_rows: rows,
            retained_bytes,
        });
    }
    Ok(surface)
}
