//! Optional scientific reference provenance, independent of viewer and learner code.
//! Declarations describe an authored model/objective; they do not authenticate a proof.
use super::{
    codec,
    models::{self, ModelLimits},
    schema::*,
    value::{self, ValueConventions, ValueCriticKind},
};
use crate::{
    Error, Result,
    core::{action::ActionSpace, ensure, env::ObservationSpace},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

mod lqg;
mod provider;

pub use lqg::LqgReference;

pub use provider::*;

pub const REFERENCE_METADATA_KEY: &str = "metadata/reference.v1";
pub const REFERENCE_CONTEXT_METADATA_KEY: &str = "reference_context.v1";
pub const MAX_REFERENCE_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_REFERENCE_TABLE_ROWS: usize = 65_536;
pub const MAX_REFERENCE_TABLE_ARRAY_BYTES: usize = 4 * 1024 * 1024;
const MAX_DIMENSIONS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceKind {
    Learned,
    Heuristic,
    Analytical,
    ApproximateNumerical,
    VerifiedOptimal,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceObjective {
    ExpectedDiscountedReturn,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum ReferenceHorizon {
    Finite { decisions: u32 },
    Continuing,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Integer(i64),
    Real(f64),
    Text(String),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceContext {
    pub environment: String,
    pub observation_dim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_observation_space: Option<ObservationSpace>,
    #[serde(with = "ReferenceActionSpace")]
    pub action_space: ActionSpace,
    pub conventions: ValueConventions,
    pub objective: ReferenceObjective,
    pub horizon: ReferenceHorizon,
    pub environment_config: BTreeMap<String, ConfigValue>,
    pub markov_observation: bool,
    pub decision_time_coordinate: Option<usize>,
}
// Keep strict reference parsing local while preserving the public type and wire shape.
#[derive(Serialize, Deserialize)]
#[serde(remote = "ActionSpace", deny_unknown_fields)]
enum ReferenceActionSpace {
    Discrete { count: usize },
    Box { low: Vec<f32>, high: Vec<f32> },
    Unbounded { dimensions: usize },
    Conditional(std::sync::Arc<crate::core::action::ActionGrammar>),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceQuality {
    pub kind: ReferenceKind,
    pub method: String,
    pub assumptions: Vec<String>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceCoverage {
    ListedStates,
    AllLiveStates,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceTableKind {
    DiscretePolicy,
    OptimalDiscreteActionValues,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteReferenceTable {
    pub probe_set: String,
    pub rows: usize,
    pub coverage: ReferenceCoverage,
    pub kind: ReferenceTableKind,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceDeclaration {
    pub version: u32,
    pub run_id: String,
    pub revision: u64,
    pub model_role: String,
    pub context: ReferenceContext,
    pub sources: BTreeMap<String, ReferenceQuality>,
    pub finite_table: Option<FiniteReferenceTable>,
}
fn text(s: &str) -> Result<()> {
    ensure(
        !s.trim().is_empty() && s.len() <= 1024 && !s.chars().any(char::is_control),
        "invalid reference metadata string",
    )
}
fn convention_strings(c: &ValueConventions) -> [&str; 8] {
    [
        &c.dynamics,
        &c.observation,
        &c.action,
        &c.reward_units,
        &c.reward_convention,
        &c.horizon,
        &c.terminal_treatment,
        &c.action_constraints,
    ]
}
impl ReferenceContext {
    pub fn validate(&self) -> Result<()> {
        text(&self.environment)?;
        if let Some(observations) = &self.typed_observation_space {
            observations.validate()?;
            ensure(
                self.observation_dim == 0 && observations.dense_width().is_none(),
                "invalid typed reference observation space",
            )?;
        } else {
            ensure(
                (1..=MAX_DIMENSIONS).contains(&self.observation_dim),
                "reference observation dimension exceeds bounds",
            )?;
        }
        let width = match &self.action_space {
            ActionSpace::Discrete { count } => *count,
            ActionSpace::Box { low, .. } => low.len(),
            ActionSpace::Unbounded { dimensions } => *dimensions,
            ActionSpace::Conditional(grammar) if self.typed_observation_space.is_some() => {
                grammar.operations.len()
            }
            ActionSpace::Conditional(_) => 0,
        };
        ensure(
            (1..=MAX_DIMENSIONS).contains(&width),
            "reference action width exceeds bounds",
        )?;
        self.action_space.validate()?;
        for s in convention_strings(&self.conventions) {
            text(s)?;
        }
        ensure(
            self.conventions.discount.is_finite()
                && (0. ..=1.).contains(&self.conventions.discount),
            "invalid reference discount",
        )?;
        ensure(
            self.decision_time_coordinate
                .is_none_or(|i| i < self.observation_dim),
            "reference decision-time coordinate out of range",
        )?;
        ensure(
            !self.environment_config.is_empty() && self.environment_config.len() <= 512,
            "reference configuration is empty or oversized",
        )?;
        for (key, value) in &self.environment_config {
            text(key)?;
            match value {
                ConfigValue::Real(x) => ensure(x.is_finite(), "nonfinite reference configuration")?,
                ConfigValue::Text(s) => text(s)?,
                _ => {}
            }
        }
        match self.horizon {
            ReferenceHorizon::Finite { decisions } => ensure(
                decisions > 0
                    && self.environment_config.get("horizon")
                        == Some(&ConfigValue::Integer(i64::from(decisions))),
                "reference typed horizon conflicts with configuration",
            )?,
            ReferenceHorizon::Continuing => ensure(
                !self.environment_config.contains_key("horizon"),
                "continuing reference cannot declare finite horizon configuration",
            )?,
        }
        codec::bounded_json(self, MAX_REFERENCE_METADATA_BYTES)?;
        Ok(())
    }
    fn manifest_facts(&self, m: &RunManifest) -> Result<()> {
        self.validate()?;
        m.validate()?;
        ensure(
            self.environment == m.environment
                && self.observation_dim == m.observation_dim
                && self.typed_observation_space == m.typed_observation_space
                && self.action_space == m.action_space
                && self.conventions.discount == m.discount
                && self.conventions.reward_convention == m.reward_convention,
            "reference context conflicts with authoritative manifest",
        )
    }
    fn heap_charge(&self) -> usize {
        // Include a spare root-node allowance even for a one-entry BTreeMap.
        let mut n = self.environment.capacity().saturating_add(1024);
        if self.typed_observation_space.is_some() {
            n = n.saturating_add(2 * MAX_REFERENCE_METADATA_BYTES);
        }
        // Actual owned string capacities, including semantic conventions.
        for s in [
            &self.conventions.dynamics,
            &self.conventions.observation,
            &self.conventions.action,
            &self.conventions.reward_units,
            &self.conventions.reward_convention,
            &self.conventions.horizon,
            &self.conventions.terminal_treatment,
            &self.conventions.action_constraints,
        ] {
            n = n.saturating_add(s.capacity());
        }
        if let ActionSpace::Box { low, high } = &self.action_space {
            n = n.saturating_add(
                low.capacity()
                    .saturating_add(high.capacity())
                    .saturating_mul(4),
            );
        }
        for (k, v) in &self.environment_config {
            n = n
                .saturating_add(128 + std::mem::size_of::<(String, ConfigValue)>())
                .saturating_add(k.capacity());
            if let ConfigValue::Text(s) = v {
                n = n.saturating_add(s.capacity());
            }
        }
        n
    }
}
/// Exact scientific compatibility, independent of run IDs or display captions.
pub fn compatible_contexts(a: &ReferenceContext, b: &ReferenceContext) -> Result<()> {
    a.validate()?;
    b.validate()?;
    ensure(a == b, "incompatible reference contexts")
}
/// The learner's actual policy at one discrete decision, never an oracle policy.
#[derive(Debug, Clone, Copy)]
pub enum DecisionPolicy<'a> {
    Deterministic(usize),
    Categorical(&'a [f64]),
}

/// Value lost by this decision followed by optimal continuation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionLoss {
    pub optimal_value: f64,
    pub expected_action_value: f64,
    pub loss: f64,
}

/// Compute one complete live decision row without sampling or allocating.
///
/// V* is derived from legal Q*. Categorical rows must contain genuine probabilities:
/// only sum roundoff strictly below 1e-6 is normalized. Illegal mass is never repaired.
/// Arithmetic uses f64 in action order; nonfinite intermediates are rejected.
pub fn decision_loss(
    optimal_q: &[f64],
    legal: &[bool],
    policy: DecisionPolicy<'_>,
) -> Result<DecisionLoss> {
    decision_loss_with_limit(optimal_q, legal, policy, 64)
}

pub(super) fn finite_decision_loss(
    optimal_q: &[f64],
    legal: &[bool],
    policy: DecisionPolicy<'_>,
) -> Result<DecisionLoss> {
    decision_loss_with_limit(optimal_q, legal, policy, super::typed::MAX_FINITE_ACTIONS)
}

fn decision_loss_with_limit(
    optimal_q: &[f64],
    legal: &[bool],
    policy: DecisionPolicy<'_>,
    max_actions: usize,
) -> Result<DecisionLoss> {
    ensure(
        (1..=max_actions).contains(&optimal_q.len())
            && optimal_q.len() == legal.len()
            && legal.iter().any(|&allowed| allowed)
            && optimal_q.iter().all(|q| q.is_finite()),
        "decision loss requires a finite complete live discrete Q row",
    )?;
    let finite = |value: f64| -> Result<f64> {
        ensure(
            value.is_finite(),
            "nonfinite/overflowed decision-loss arithmetic",
        )?;
        Ok(value)
    };
    let mut optimal_value = f64::NEG_INFINITY;
    let mut scale = 1_f64;
    for (&q, &allowed) in optimal_q.iter().zip(legal) {
        if allowed {
            optimal_value = optimal_value.max(q);
            scale = scale.max(q.abs());
        }
    }
    let mut expected_action_value = match policy {
        DecisionPolicy::Deterministic(action) => {
            ensure(
                legal.get(action) == Some(&true),
                "decision loss requires an in-range legal policy action",
            )?;
            optimal_q[action]
        }
        DecisionPolicy::Categorical(probabilities) => {
            ensure(
                probabilities.len() == optimal_q.len(),
                "decision-loss categorical width mismatch",
            )?;
            let mut sum = 0.;
            for (&p, &allowed) in probabilities.iter().zip(legal) {
                ensure(
                    p.is_finite() && (0. ..=1.).contains(&p) && (allowed || p == 0.),
                    "invalid decision-loss probability or positive illegal mass",
                )?;
                sum = finite(sum + p)?;
            }
            ensure(
                sum > 0. && (sum - 1.).abs() < 1e-6,
                "decision-loss probabilities must sum to one within roundoff",
            )?;
            let mut expectation = 0.;
            for ((&p, &q), &allowed) in probabilities.iter().zip(optimal_q).zip(legal) {
                if allowed {
                    let normalized = finite(p / sum)?;
                    let product = finite(normalized * q)?;
                    expectation = finite(expectation + product)?;
                }
            }
            expectation
        }
    };
    let mut loss = finite(optimal_value - expected_action_value)?;
    scale = scale.max(expected_action_value.abs());
    // Conservative f64 summation bound proportional to the admitted menu width.
    // Multiply the small factor first so the tolerance cannot overflow at f64::MAX.
    let tolerance = finite((32. * optimal_q.len() as f64 * f64::EPSILON) * scale)?;
    ensure(loss >= -tolerance, "inconsistent negative decision loss")?;
    if loss < 0. {
        expected_action_value = optimal_value;
        loss = 0.;
    }
    Ok(DecisionLoss {
        optimal_value,
        expected_action_value,
        loss,
    })
}

/// Admit explicitly verified finite Q metadata under exact scientific contexts.
///
/// This validates declarations only: callers must separately bind each source to
/// its manifest/checkpoint, match the actual learner policy, validate the payload,
/// and find the exact complete observation, time and effective mask. Coverage does
/// not prove a row exists; unsupported or terminal states must remain unavailable.
pub fn compatible_decision_loss(
    current: &ReferenceContext,
    oracle: &ReferenceDeclaration,
    oracle_name: &str,
) -> Result<()> {
    current.validate()?;
    oracle.validate()?;
    compatible_contexts(current, &oracle.context)?;
    ensure(
        oracle
            .sources
            .get(oracle_name)
            .is_some_and(|source| source.kind == ReferenceKind::VerifiedOptimal)
            && oracle
                .finite_table
                .as_ref()
                .is_some_and(|table| table.kind == ReferenceTableKind::OptimalDiscreteActionValues)
            && matches!(
                oracle.context.action_space,
                ActionSpace::Discrete { count: 1..=64 }
            )
            && matches!(
                oracle.context.horizon,
                ReferenceHorizon::Finite { decisions: 1.. }
            )
            && oracle.context.markov_observation
            && oracle.context.decision_time_coordinate.is_some(),
        "decision loss requires an explicit verified finite Markov optimal-Q subject",
    )
}
/// Attach a separately retained, immutable typed context to the run manifest.
/// The producer must first verify this context against its original environment configuration.
pub fn attach_manifest_context(
    manifest: &mut RunManifest,
    context: &ReferenceContext,
) -> Result<()> {
    context.manifest_facts(manifest)?;
    if let Some(old) = manifest_context(manifest)? {
        return ensure(
            old == *context,
            "immutable manifest reference context already attached",
        );
    }
    let bytes = codec::bounded_json(context, MAX_REFERENCE_METADATA_BYTES)?;
    manifest.metadata.insert(
        REFERENCE_CONTEXT_METADATA_KEY.into(),
        String::from_utf8(bytes).map_err(|e| Error::Config(e.to_string()))?,
    );
    Ok(())
}
pub fn manifest_context(manifest: &RunManifest) -> Result<Option<ReferenceContext>> {
    let Some(encoded) = manifest.metadata.get(REFERENCE_CONTEXT_METADATA_KEY) else {
        return Ok(None);
    };
    ensure(
        encoded.len() <= MAX_REFERENCE_METADATA_BYTES,
        "oversized manifest reference context",
    )?;
    let c: ReferenceContext = serde_json::from_str(encoded)?;
    c.manifest_facts(manifest)?;
    Ok(Some(c))
}
impl ReferenceDeclaration {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1 && !self.sources.is_empty() && self.sources.len() <= 64,
            "unsupported or oversized reference declaration",
        )?;
        text(&self.run_id)?;
        text(&self.model_role)?;
        self.context.validate()?;
        for (name, q) in &self.sources {
            text(name)?;
            ensure(name.trim() == name, "invalid reference subject name")?;
            text(&q.method)?;
            ensure(q.assumptions.len() <= 16, "too many reference assumptions")?;
            for a in &q.assumptions {
                text(a)?;
            }
        }
        if let Some(t) = &self.finite_table {
            text(&t.probe_set)?;
            ensure(
                (1..=MAX_REFERENCE_TABLE_ROWS).contains(&t.rows),
                "finite reference row count exceeds bounds",
            )?;
            ensure(
                self.sources.len() == 1
                    && self
                        .sources
                        .values()
                        .all(|q| q.kind == ReferenceKind::VerifiedOptimal)
                    && self.context.markov_observation
                    && self.context.decision_time_coordinate.is_some()
                    && matches!(self.context.horizon, ReferenceHorizon::Finite { .. })
                    && matches!(self.context.action_space, ActionSpace::Discrete { .. }),
                "finite reference table requires a verified discrete finite Markov subject",
            )?;
        }
        codec::bounded_json(self, MAX_REFERENCE_METADATA_BYTES)?;
        Ok(())
    }
    pub fn validate_manifest(&self, manifest: &RunManifest) -> Result<()> {
        self.validate()?;
        self.context.manifest_facts(manifest)?;
        ensure(
            self.run_id == manifest.run_id,
            "reference declaration belongs to another manifest",
        )?;
        let anchor = manifest_context(manifest)?.ok_or_else(|| {
            Error::Config("reference declaration requires immutable manifest context".into())
        })?;
        compatible_contexts(&self.context, &anchor)
    }
    /// Conservative retained ownership, not an allocator/RSS census. Arc sharing
    /// avoids duplicate charges only when the allocation is actually shared.
    pub fn retained_charge(&self) -> usize {
        let mut n = std::mem::size_of::<Self>()
            .saturating_add(1024)
            .saturating_add(self.run_id.capacity())
            .saturating_add(self.model_role.capacity())
            .saturating_add(self.context.heap_charge());
        for (k, q) in &self.sources {
            n = n
                .saturating_add(128 + std::mem::size_of::<(String, ReferenceQuality)>())
                .saturating_add(k.capacity())
                .saturating_add(q.method.capacity())
                .saturating_add(
                    q.assumptions
                        .capacity()
                        .saturating_mul(std::mem::size_of::<String>()),
                );
            for a in &q.assumptions {
                n = n.saturating_add(a.capacity());
            }
        }
        if let Some(t) = &self.finite_table {
            n = n.saturating_add(t.probe_set.capacity());
        }
        n
    }
}
fn array<'a>(c: &'a Chunk, key: &str) -> Result<&'a Array> {
    c.arrays
        .get(key)
        .ok_or_else(|| Error::Shape("reference payload array missing".into()))
}
fn bind(c: &Chunk, d: &ReferenceDeclaration) -> Result<()> {
    d.validate()?;
    let i = &c.context.identity;
    ensure(
        d.run_id == i.run_id
            && d.revision == i.learner_update_revision
            && d.model_role == i.model_role,
        "reference declaration context mismatch",
    )?;
    let existing = value::declaration(c)?;
    if let Some(v) = &existing {
        ensure(
            v.conventions.as_ref() == Some(&d.context.conventions),
            "reference and value conventions conflict",
        )?;
    }
    match &c.record {
        Record::ModelSnapshot(s) => {
            ensure(
                d.finite_table.is_none(),
                "snapshot cannot declare a finite reference population",
            )?;
            let config = models::snapshot_configuration(s, ModelLimits::default())?;
            ensure(
                config.observation_dim == d.context.observation_dim
                    && config.action_space == d.context.action_space
                    && d.sources
                        .keys()
                        .all(|name| config.models.contains_key(name)),
                "reference snapshot subject or input contract mismatch",
            )?;
        }
        Record::Policy(p) => {
            ensure(
                d.sources.len() == 1,
                "policy reference requires one authored subject",
            )?;
            let actions = array(c, &p.actions)?;
            match &d.context.action_space {
                ActionSpace::Discrete { count } => {
                    ensure(
                        actions.shape().len() == 1
                            && actions.as_u64()?.iter().all(|a| *a < *count as u64),
                        "reference policy discrete action mismatch",
                    )?;
                    if let Some(m) = &p.masks {
                        ensure(
                            array(c, m)?.shape() == [actions.len(), *count],
                            "reference policy mask width mismatch",
                        )?;
                    }
                    if let PolicyParameters::Categorical { probabilities } = &p.parameters {
                        ensure(
                            array(c, probabilities)?.shape() == [actions.len(), *count],
                            "reference policy probability width mismatch",
                        )?;
                    }
                    ensure(
                        !matches!(p.parameters, PolicyParameters::Gaussian { .. }),
                        "discrete reference cannot declare Gaussian policy",
                    )?;
                }
                space => {
                    ensure(
                        actions.shape().len() == 2
                            && actions.shape()[1] == space.dimensions()
                            && p.masks.is_none()
                            && !matches!(p.parameters, PolicyParameters::Categorical { .. }),
                        "reference policy continuous action mismatch",
                    )?;
                    let valid = |i: usize, value: f64| {
                        value.is_finite()
                            && match space {
                                ActionSpace::Box { low, high } => {
                                    let j = i % low.len();
                                    value >= f64::from(low[j]) && value <= f64::from(high[j])
                                }
                                _ => true,
                            }
                    };
                    ensure(
                        match &actions.values {
                            Values::F32(v) => {
                                v.iter().enumerate().all(|(i, x)| valid(i, f64::from(*x)))
                            }
                            Values::F64(v) => v.iter().enumerate().all(|(i, x)| valid(i, *x)),
                            _ => false,
                        },
                        "reference policy action violates declared bounds",
                    )?;
                }
            }
        }
        Record::Critic(p) => {
            ensure(
                d.sources.len() == 1,
                "critic reference requires one authored subject",
            )?;
            if let Some(v) = &existing {
                ensure(
                    v.critics.keys().eq(d.sources.keys()),
                    "reference critic subject conflicts with value declaration",
                )?;
            }
            if let CriticRepresentation::Discrete { values } = &p.representation {
                let ActionSpace::Discrete { count } = d.context.action_space else {
                    return Err(Error::Config(
                        "discrete reference critic requires discrete actions".into(),
                    ));
                };
                ensure(
                    array(c, values)?.shape().len() == 2 && array(c, values)?.shape()[1] == count,
                    "reference critic action width mismatch",
                )?;
            }
        }
        _ => {
            return Err(Error::Config(
                "reference declaration requires snapshot, policy or critic".into(),
            ));
        }
    }
    if let Some(t) = &d.finite_table {
        ensure(
            c.context.probe_set.as_deref() == Some(t.probe_set.as_str())
                && c.array_bytes() <= MAX_REFERENCE_TABLE_ARRAY_BYTES,
            "finite reference population identity or bytes mismatch",
        )?;
        match (&c.record, t.kind) {
            (Record::Policy(p), ReferenceTableKind::DiscretePolicy) => ensure(
                matches!(p.parameters, PolicyParameters::Deterministic)
                    && !p.stochastic
                    && array(c, &p.actions)?.shape() == [t.rows],
                "finite reference policy must contain deterministic discrete rows",
            )?,
            (Record::Critic(p), ReferenceTableKind::OptimalDiscreteActionValues) => {
                let CriticRepresentation::Discrete { values } = &p.representation else {
                    return Err(Error::Config(
                        "finite reference requires all discrete Q heads".into(),
                    ));
                };
                ensure(
                    p.action_condition == ActionCondition::AllDiscreteActions
                        && p.actions.is_none()
                        && array(c, values)?.shape()[0] == t.rows,
                    "finite reference Q conditioning or rows mismatch",
                )?;
                let v = existing.as_ref().ok_or_else(|| {
                    Error::Config("optimal Q table requires value declaration".into())
                })?;
                ensure(
                    v.critics
                        .values()
                        .all(|v| v.kind == ValueCriticKind::DiscreteQ && v.continuation.is_some()),
                    "optimal Q table lacks declared continuation",
                )?;
            }
            _ => {
                return Err(Error::Config(
                    "finite reference table kind conflicts with record".into(),
                ));
            }
        }
    }
    Ok(())
}
pub fn attach_declaration(chunk: &mut Chunk, d: &ReferenceDeclaration) -> Result<()> {
    chunk.validate()?;
    bind(chunk, d)?;
    if let Some(old) = declaration(chunk)? {
        return ensure(
            old == *d,
            "immutable reference declaration already attached",
        );
    }
    let bytes = codec::bounded_json(d, MAX_REFERENCE_METADATA_BYTES)?;
    if d.finite_table.is_some() {
        ensure(
            chunk
                .array_bytes()
                .checked_add(bytes.len())
                .is_some_and(|n| n <= MAX_REFERENCE_TABLE_ARRAY_BYTES),
            "finite reference metadata exceeds table admission",
        )?;
    }
    chunk.arrays.insert(
        REFERENCE_METADATA_KEY.into(),
        Array::u8(vec![bytes.len()], bytes)?,
    );
    Ok(())
}
pub fn declaration(chunk: &Chunk) -> Result<Option<ReferenceDeclaration>> {
    let Some(a) = chunk.arrays.get(REFERENCE_METADATA_KEY) else {
        return Ok(None);
    };
    ensure(
        a.shape().len() == 1 && a.byte_len() <= MAX_REFERENCE_METADATA_BYTES,
        "malformed or oversized reference metadata",
    )?;
    let d: ReferenceDeclaration = serde_json::from_slice(a.as_u8()?)?;
    bind(chunk, &d)?;
    Ok(Some(d))
}
/// Validate a finite table's exact immutable population binding and structural
/// evidence. Domain completeness and Bellman optimality remain producer checks;
/// canonical-key uniqueness and lookup admission belong to the bounded reader.
pub fn validate_table(table: &Chunk, probes: &Chunk) -> Result<()> {
    let d = declaration(table)?
        .ok_or_else(|| Error::Config("finite reference declaration absent".into()))?;
    let t = d
        .finite_table
        .as_ref()
        .ok_or_else(|| Error::Config("finite reference table marker absent".into()))?;
    ensure(
        table
            .array_bytes()
            .checked_add(probes.array_bytes())
            .is_some_and(|n| n <= MAX_REFERENCE_TABLE_ARRAY_BYTES),
        "finite reference arrays exceed admission",
    )?;
    table.validate()?;
    probes.validate()?;
    let Record::ProbeSet(p) = &probes.record else {
        return Err(Error::Config(
            "finite reference requires probe population".into(),
        ));
    };
    ensure(
        probes.context.identity.run_id == d.run_id
            && probes.context.identity.model_role == d.model_role
            && probes.context.identity.learner_update_revision == d.revision
            && probes.context.probe_set.as_deref() == Some(&t.probe_set)
            && p.id == t.probe_set
            && p.histories.is_empty()
            && p.fixed_actions.is_none(),
        "finite reference probe identity or state contract mismatch",
    )?;
    let x = array(probes, &p.observations)?;
    ensure(
        x.shape() == [t.rows, d.context.observation_dim]
            && x.as_f32()?.iter().all(|v| v.is_finite()),
        "finite reference requires complete finite f32 inputs",
    )?;
    let ActionSpace::Discrete { count } = d.context.action_space else {
        unreachable!("validated table")
    };
    let mask = array(
        probes,
        p.action_masks
            .as_ref()
            .ok_or_else(|| Error::Config("finite reference needs explicit legal masks".into()))?,
    )?;
    ensure(
        mask.shape() == [t.rows, count]
            && mask
                .as_u8()?
                .chunks_exact(count)
                .all(|r| r.iter().all(|v| *v <= 1) && r.contains(&1)),
        "finite reference invalid live masks",
    )?;
    match &table.record {
        Record::Policy(p) => {
            if let Some(m) = &p.masks {
                ensure(
                    array(table, m)?.shape() == mask.shape()
                        && array(table, m)?.as_u8()? == mask.as_u8()?,
                    "reference policy and probe masks differ",
                )?;
            }
            ensure(
                array(table, &p.actions)?
                    .as_u64()?
                    .iter()
                    .enumerate()
                    .all(|(i, a)| {
                        *a < count as u64
                            && mask.as_u8().expect("validated")[i * count + *a as usize] == 1
                    }),
                "reference policy action violates population mask",
            )?;
        }
        Record::Critic(p) => {
            let CriticRepresentation::Discrete { values } = &p.representation else {
                unreachable!("validated table")
            };
            let q = array(table, values)?;
            ensure(
                q.shape() == [t.rows, count]
                    && match &q.values {
                        Values::F32(v) => v.iter().all(|x| x.is_finite()),
                        Values::F64(v) => v.iter().all(|x| x.is_finite()),
                        _ => false,
                    },
                "finite reference Q values must be finite and rectangular",
            )?;
        }
        _ => unreachable!("validated table"),
    }
    Ok(())
}
