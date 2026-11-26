//! Optional immutable scientific declarations and bounded value interpretation.
//! Missing metadata remains unknown; algorithm labels never supply conventions.
use super::{
    analysis::{AtomicForecast, QuantileForecast, validate_quantiles},
    codec,
    evaluation::{
        DISCOUNTED_REWARDS_KEY, EvaluationHorizon, PHYSICAL_DURATIONS_KEY, evaluation_horizon,
    },
    models::{self, ModelKind, ModelLimits},
    schema::*,
};
use crate::{Error, Result, core::ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const VALUE_METADATA_KEY: &str = "metadata/value.v1";
pub const RETURN_TARGET_METADATA_KEY: &str = "metadata/value-return-target.v1";
const MAX_RETURN_TARGET_BYTES: usize = 1024;

/// Where entropy enters the action-value target. Both conventions omit
/// entropy of the action on which the critic is conditioned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyTiming {
    /// The factual replay prefix carries environmental rewards. Entropy
    /// first enters at the bootstrap action after the prefix.
    ReplayBootstrap { n_step: usize },
    /// Successor-action entropy is included at each on-policy transition.
    TransitionSuccessor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogDensityConvention {
    /// The policy's mass or transformed density in executed action coordinates.
    PolicySampleInActionCoordinates,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyTemperatureSource {
    /// Multiplier at the immutable model boundary. This describes its prospective
    /// objective and does not assert which multiplier produced prior targets.
    Snapshot,
    /// Fixed multiplier applied to the draws in an independent evaluation.
    IndependentEvaluation,
}

/// Additional return semantics for a critic or snapshot. Legacy declarations
/// may omit the metadata. Environmental calibration requires an environmental target.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReturnTarget {
    Environmental,
    SoftActionValue {
        alpha: f32,
        temperature_source: EntropyTemperatureSource,
        entropy_timing: EntropyTiming,
        log_density: LogDensityConvention,
    },
    /// Return under a caller-declared potential in the current state/time domain.
    /// Use the same identifier only when the potential function and units agree.
    PotentialShaped {
        potential_id: [u8; 32],
    },
}
impl ReturnTarget {
    pub fn is_soft(self) -> bool {
        matches!(self, Self::SoftActionValue { .. })
    }
    pub fn is_environmental(self) -> bool {
        matches!(self, Self::Environmental)
    }
}
pub const MAX_VALUE_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_VALUE_HEADS: usize = 4096;
/// Bounds suffix work even for manually constructed/imported chunks. The
/// existing schema validator's optional f64 clock copy stays about 1 MiB.
pub const MAX_RETURN_STEPS: usize = 131_072;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueConventions {
    pub dynamics: String,
    pub observation: String,
    pub action: String,
    pub reward_units: String,
    pub reward_convention: String,
    pub discount: f64,
    pub horizon: String,
    pub terminal_treatment: String,
    pub action_constraints: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueCriticKind {
    StateValue,
    ScalarQ,
    DiscreteQ,
    QuantileQ,
    MomentQ,
    CategoricalQ,
    CategoricalReturnQ,
}
impl ValueCriticKind {
    pub fn is_return_distribution(self) -> bool {
        matches!(self, Self::QuantileQ | Self::CategoricalReturnQ)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuationIdentity {
    /// Opaque scientific provenance (for example a revision-specific policy ID),
    /// not a key into a snapshot's exported model map. Resolving an actual model
    /// requires an unambiguous adapter or an independent explicit model choice.
    pub identity: String,
    pub stochastic: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwinMembership {
    pub group: String,
    /// Ordered Q1/Q2 identity, zero or one, within this declaration's context.
    pub index: usize,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MomentUtility {
    ExpectedReturn,
    MeanVariance { penalty: f32 },
    Unsupported { name: String },
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CriticDeclaration {
    pub kind: ValueCriticKind,
    /// `Some(false)` explicitly excludes an untrained REINFORCE baseline.
    pub learned_baseline: Option<bool>,
    pub actor: Option<ContinuationIdentity>,
    pub continuation: Option<ContinuationIdentity>,
    pub twin: Option<TwinMembership>,
    pub utility: Option<MomentUtility>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueDeclaration {
    pub version: u32,
    pub run_id: String,
    pub revision: u64,
    pub model_role: String,
    pub conventions: Option<ValueConventions>,
    pub critics: BTreeMap<String, CriticDeclaration>,
}
fn text(value: &str) -> Result<()> {
    ensure(
        !value.trim().is_empty() && value.len() <= 1024 && !value.chars().any(char::is_control),
        "invalid value declaration string",
    )
}
impl ValueDeclaration {
    /// Validate the declaration itself, independently of a particular payload.
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1 && self.critics.len() <= 64,
            "unsupported/oversized value declaration",
        )?;
        text(&self.run_id)?;
        text(&self.model_role)?;
        if let Some(c) = &self.conventions {
            for s in [
                &c.dynamics,
                &c.observation,
                &c.action,
                &c.reward_units,
                &c.reward_convention,
                &c.horizon,
                &c.terminal_treatment,
                &c.action_constraints,
            ] {
                text(s)?;
            }
            ensure(
                c.discount.is_finite() && (0. ..=1.).contains(&c.discount),
                "invalid value discount",
            )?;
        }
        let mut twins = BTreeSet::new();
        for (name, c) in &self.critics {
            text(name)?;
            ensure(name.trim() == name, "invalid critic name")?;
            for id in c.actor.iter().chain(&c.continuation) {
                text(&id.identity)?;
            }
            ensure(
                c.learned_baseline.is_none() || c.kind == ValueCriticKind::StateValue,
                "learned baseline flag requires state value",
            )?;
            if let Some(t) = &c.twin {
                text(&t.group)?;
                ensure(
                    c.kind == ValueCriticKind::ScalarQ
                        && t.index <= 1
                        && twins.insert((&t.group, t.index)),
                    "invalid/duplicate ordered twin membership",
                )?;
            }
            if let Some(u) = &c.utility {
                ensure(
                    c.kind == ValueCriticKind::MomentQ,
                    "moment utility requires raw moments",
                )?;
                validate_utility(u)?;
            }
        }
        // Enforce the same bound when supplied as an owned object.
        codec::bounded_json(self, MAX_VALUE_METADATA_BYTES)?;
        Ok(())
    }
    /// Check manifest facts when a producer has the authoritative run manifest.
    /// Remaining semantic strings are authored scientific declarations.
    pub fn validate_manifest(&self, manifest: &RunManifest) -> Result<()> {
        self.validate()?;
        manifest.validate()?;
        ensure(
            self.run_id == manifest.run_id,
            "value declaration belongs to another manifest",
        )?;
        if let Some(c) = &self.conventions {
            ensure(
                c.discount == manifest.discount
                    && c.reward_convention == manifest.reward_convention,
                "value declaration conflicts with manifest reward/discount",
            )?;
        }
        Ok(())
    }
}
fn validate_utility(utility: &MomentUtility) -> Result<()> {
    match utility {
        MomentUtility::MeanVariance { penalty } => ensure(
            penalty.is_finite() && *penalty >= 0.,
            "invalid variance penalty",
        ),
        MomentUtility::Unsupported { name } => text(name),
        MomentUtility::ExpectedReturn => Ok(()),
    }
}
fn model_kind(kind: &ModelKind) -> Option<ValueCriticKind> {
    kind.critic_kind()
}
fn bind(chunk: &Chunk, declaration: &ValueDeclaration) -> Result<()> {
    declaration.validate()?;
    let identity = &chunk.context.identity;
    ensure(
        declaration.run_id == identity.run_id
            && declaration.revision == identity.learner_update_revision
            && declaration.model_role == identity.model_role,
        "value declaration context mismatch",
    )?;
    match &chunk.record {
        Record::ModelSnapshot(snapshot) => {
            let config = models::snapshot_configuration(snapshot, ModelLimits::default())?;
            let mut groups: BTreeMap<&str, usize> = BTreeMap::new();
            for (name, c) in &declaration.critics {
                ensure(
                    config.models.get(name).and_then(|d| model_kind(&d.kind)) == Some(c.kind),
                    "value declaration model name/kind mismatch",
                )?;
                if let Some(t) = &c.twin {
                    *groups.entry(&t.group).or_default() += 1;
                }
            }
            ensure(
                groups.values().all(|&n| n == 2),
                "snapshot twin group requires both ordered critics",
            )?;
        }
        Record::Critic(probe) => {
            ensure(
                declaration.critics.len() == 1,
                "critic probe needs one authored critic identity",
            )?;
            let c = declaration.critics.values().next().expect("one critic");
            let matches = match (&probe.representation, probe.action_condition) {
                (CriticRepresentation::Scalar { .. }, ActionCondition::StateValue) => {
                    c.kind == ValueCriticKind::StateValue
                }
                (
                    CriticRepresentation::Scalar { .. },
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy,
                ) => c.kind == ValueCriticKind::ScalarQ,
                (CriticRepresentation::Discrete { .. }, ActionCondition::AllDiscreteActions) => {
                    c.kind == ValueCriticKind::DiscreteQ
                }
                (
                    CriticRepresentation::Quantiles { .. },
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy,
                ) => c.kind == ValueCriticKind::QuantileQ,
                (
                    CriticRepresentation::RawMoments { .. },
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy,
                ) => c.kind == ValueCriticKind::MomentQ,
                (
                    CriticRepresentation::CategoricalRegression { .. },
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy,
                ) => c.kind == ValueCriticKind::CategoricalQ,
                (
                    CriticRepresentation::CategoricalReturn { .. },
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy,
                ) => c.kind == ValueCriticKind::CategoricalReturnQ,
                _ => false,
            };
            ensure(
                matches,
                "value declaration probe representation/conditioning mismatch",
            )?;
            if let Some(id) = &probe.continuation_policy {
                ensure(
                    c.continuation.as_ref().is_some_and(|p| {
                        &p.identity == id && probe.continuation_stochastic == Some(p.stochastic)
                    }),
                    "probe continuation conflicts with value declaration",
                )?;
            }
        }
        Record::Typed(record) if record.kind == super::typed::TypedKind::Predictions => {
            use super::typed::{FiniteValueKind, TypedPayload};
            let TypedPayload::Predictions(predictions) = TypedPayload::read(chunk)? else {
                unreachable!("typed payload kind was validated")
            };
            ensure(
                matches!(
                    predictions.value_kind,
                    Some(FiniteValueKind::ActionValue | FiniteValueKind::OptimalActionValue)
                ) && declaration.critics.len() == 1
                    && declaration.critics.values().all(|critic| {
                        critic.kind == ValueCriticKind::DiscreteQ && critic.twin.is_none()
                    }),
                "finite value declaration requires one action-value prediction",
            )?;
        }
        _ => {
            return Err(Error::Config(
                "value declaration requires a snapshot or critic probe".into(),
            ));
        }
    }
    Ok(())
}
pub fn attach_declaration(chunk: &mut Chunk, declaration: &ValueDeclaration) -> Result<()> {
    bind(chunk, declaration)?;
    let target = return_target(chunk)?;
    if target.is_some_and(|target| !target.is_environmental()) {
        ensure(
            declaration.conventions.is_none(),
            "transformed returns cannot certify an environmental return convention",
        )?;
    }
    if target.is_some_and(ReturnTarget::is_soft) {
        ensure(
            declaration
                .critics
                .values()
                .all(|critic| critic.kind != ValueCriticKind::StateValue),
            "soft action-value target cannot bind state-value declarations",
        )?;
    }
    if let Some(existing) = self::declaration(chunk)? {
        ensure(
            existing == *declaration,
            "immutable value declaration is already attached",
        )?;
        return Ok(());
    }
    chunk.validate()?;
    let bytes = codec::bounded_json(declaration, MAX_VALUE_METADATA_BYTES)?;
    chunk.arrays.insert(
        VALUE_METADATA_KEY.into(),
        Array::u8(vec![bytes.len()], bytes)?,
    );
    Ok(())
}
pub fn declaration(chunk: &Chunk) -> Result<Option<ValueDeclaration>> {
    let Some(array) = chunk.arrays.get(VALUE_METADATA_KEY) else {
        return Ok(None);
    };
    ensure(
        array.byte_len() <= MAX_VALUE_METADATA_BYTES && array.shape().len() == 1,
        "oversized/malformed value metadata",
    )?;
    let bytes = array.as_u8()?;
    let declaration: ValueDeclaration = serde_json::from_slice(bytes)?;
    bind(chunk, &declaration)?;
    Ok(Some(declaration))
}

fn bind_return_target(chunk: &Chunk, target: ReturnTarget) -> Result<()> {
    ensure(
        matches!(chunk.record, Record::Critic(_) | Record::ModelSnapshot(_))
            || (matches!(target, ReturnTarget::PotentialShaped { .. })
                && chunk.record.kind() == RecordKind::FinitePredictions),
        "return target requires an eligible value prediction or model snapshot",
    )?;
    if target.is_environmental() {
        return Ok(());
    }
    let declared = declaration(chunk)?;
    ensure(
        declared
            .as_ref()
            .is_none_or(|declaration| declaration.conventions.is_none()),
        "transformed returns cannot certify an environmental return convention",
    )?;
    if let ReturnTarget::SoftActionValue {
        alpha,
        entropy_timing,
        ..
    } = target
    {
        ensure(
            alpha.is_finite()
                && alpha >= 0.
                && !matches!(entropy_timing, EntropyTiming::ReplayBootstrap { n_step: 0 }),
            "invalid soft-return temperature or bootstrap horizon",
        )?;
        ensure(
            !matches!(&chunk.record, Record::Critic(critic) if critic.action_condition == ActionCondition::StateValue),
            "soft action-value target requires an action-conditioned critic",
        )?;
        if let Some(declaration) = declared {
            ensure(
                declaration
                    .critics
                    .values()
                    .all(|critic| critic.kind != ValueCriticKind::StateValue),
                "soft action-value target cannot bind state-value declarations",
            )?;
        }
    }
    Ok(())
}

/// Read typed return semantics where available.
pub fn return_target(chunk: &Chunk) -> Result<Option<ReturnTarget>> {
    let Some(array) = chunk.arrays.get(RETURN_TARGET_METADATA_KEY) else {
        return Ok(None);
    };
    ensure(
        array.shape().len() == 1 && array.byte_len() <= MAX_RETURN_TARGET_BYTES,
        "oversized or malformed return target",
    )?;
    let target = serde_json::from_slice(array.as_u8()?)?;
    bind_return_target(chunk, target)?;
    Ok(Some(target))
}

/// Bind return semantics to the artifact without replacing prior evidence.
pub fn attach_return_target(chunk: &mut Chunk, target: ReturnTarget) -> Result<()> {
    bind_return_target(chunk, target)?;
    if let Some(old) = return_target(chunk)? {
        return ensure(
            old == target,
            "immutable critic return target is already attached",
        );
    }
    let bytes = codec::bounded_json(&target, MAX_RETURN_TARGET_BYTES)?;
    chunk.arrays.insert(
        RETURN_TARGET_METADATA_KEY.into(),
        Array::u8(vec![bytes.len()], bytes)?,
    );
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValueOutput {
    ExpectedReturn,
    /// Derived from the current policy and verified optimal Q, never a critic statistic.
    DecisionLoss,
    Q1,
    Q2,
    Minimum,
    Disagreement,
    Quantile {
        probability: f64,
    },
    Cvar {
        probability: f64,
    },
    RawSecondMoment,
    Variance,
    StandardDeviation,
    Utility,
}
#[derive(Debug, Clone)]
pub enum ValuePrediction {
    Scalar(f64),
    Discrete {
        values: Vec<f64>,
        legal: Vec<bool>,
    },
    Quantiles {
        raw: Vec<f64>,
        locations: Vec<f64>,
        masses: Vec<f64>,
    },
    Moments {
        mean: f64,
        raw_second: f64,
    },
    CategoricalRegression {
        logits: Vec<f64>,
        weights: Vec<f64>,
        centers: Vec<f64>,
        decoded: f64,
    },
    CategoricalReturn {
        atoms: Vec<f64>,
        probabilities: Vec<f64>,
    },
    Twin {
        q1: Box<ValuePrediction>,
        q2: Box<ValuePrediction>,
    },
}
#[derive(Debug, Clone, PartialEq)]
pub struct MomentSummary {
    pub mean: f64,
    pub raw_second: f64,
    pub variance: f64,
    pub standard_deviation: Option<f64>,
}
fn finite(value: f64) -> Result<f64> {
    ensure(value.is_finite(), "nonfinite/overflowed value prediction")?;
    Ok(value)
}
fn validate_quantile_width(raw: &[f64], locations: &[f64], masses: &[f64]) -> Result<()> {
    ensure(
        raw.len() <= MAX_VALUE_HEADS && raw.len() == locations.len() && raw.len() == masses.len(),
        "value quantile width exceeds admission or differs from grid",
    )
}
impl ValuePrediction {
    pub fn select_discrete(&self, action: usize) -> Result<Self> {
        let Self::Discrete { values, legal } = self else {
            return Err(Error::Action("not a discrete value prediction".into()));
        };
        ensure(
            !values.is_empty()
                && values.len() <= MAX_VALUE_HEADS
                && values.len() == legal.len()
                && values.iter().all(|v| v.is_finite())
                && legal.get(action) == Some(&true),
            "illegal/malformed discrete value selection",
        )?;
        Ok(Self::Scalar(values[action]))
    }
    pub fn greedy_discrete(&self) -> Result<Self> {
        let Self::Discrete { values, legal } = self else {
            return Err(Error::Action("not a discrete value prediction".into()));
        };
        ensure(
            !values.is_empty()
                && values.len() <= MAX_VALUE_HEADS
                && values.len() == legal.len()
                && values.iter().all(|v| v.is_finite()),
            "malformed discrete values/mask",
        )?;
        let action = (0..values.len())
            .filter(|&i| legal[i])
            .max_by(|&a, &b| values[a].total_cmp(&values[b]))
            .ok_or_else(|| Error::Action("no legal discrete value action".into()))?;
        self.select_discrete(action)
    }
    pub fn moment_summary(&self) -> Result<MomentSummary> {
        let Self::Moments { mean, raw_second } = *self else {
            return Err(Error::Config("value prediction has no raw moments".into()));
        };
        finite(mean)?;
        finite(raw_second)?;
        let variance = finite(raw_second - mean * mean)?;
        Ok(MomentSummary {
            mean,
            raw_second,
            variance,
            standard_deviation: (variance >= 0.).then(|| variance.sqrt()),
        })
    }
    pub fn ordered_quantiles(&self) -> Result<QuantileForecast> {
        let Self::Quantiles {
            raw,
            locations,
            masses,
        } = self
        else {
            return Err(Error::Config("value prediction has no quantiles".into()));
        };
        validate_quantile_width(raw, locations, masses)?;
        QuantileForecast::ordered(raw.clone(), locations.clone(), masses.clone())
    }
    pub fn weighted_atoms(&self) -> Result<AtomicForecast> {
        match self {
            Self::Quantiles { .. } => self.ordered_quantiles()?.into_weighted_atoms(),
            Self::CategoricalReturn {
                atoms,
                probabilities,
            } => {
                ensure(
                    (2..=MAX_VALUE_HEADS).contains(&atoms.len())
                        && atoms.windows(2).all(|pair| pair[0] < pair[1]),
                    "invalid categorical return support",
                )?;
                AtomicForecast::new(atoms.clone(), probabilities.clone())
            }
            _ => Err(Error::Config(
                "value prediction has no return distribution".into(),
            )),
        }
    }
    pub fn scalar(&self, output: &ValueOutput, utility: Option<&MomentUtility>) -> Result<f64> {
        ensure(
            *output != ValueOutput::DecisionLoss,
            "decision loss requires a current policy and verified optimal Q, not a value prediction",
        )?;
        let value = match (self, output) {
            (Self::Scalar(v), ValueOutput::ExpectedReturn) => *v,
            (Self::CategoricalRegression { decoded, .. }, ValueOutput::ExpectedReturn) => *decoded,
            (Self::Quantiles { .. }, ValueOutput::ExpectedReturn) => self.ordered_quantiles()?.mean(),
            (Self::Quantiles { .. }, ValueOutput::Quantile { probability }) => self.ordered_quantiles()?.quantile(*probability)?,
            (Self::Quantiles { .. }, ValueOutput::Cvar { probability }) => self.ordered_quantiles()?.lower_cvar(*probability)?,
            (Self::CategoricalReturn { .. }, ValueOutput::ExpectedReturn) => self.weighted_atoms()?.mean(),
            (Self::CategoricalReturn { .. }, ValueOutput::Quantile { probability }) => self.weighted_atoms()?.quantile(*probability)?,
            (Self::CategoricalReturn { .. }, ValueOutput::Cvar { probability }) => self.weighted_atoms()?.lower_cvar(*probability)?,
            (Self::CategoricalReturn { .. }, ValueOutput::Variance) => self.weighted_atoms()?.variance()?,
            (Self::CategoricalReturn { .. }, ValueOutput::StandardDeviation) => self.weighted_atoms()?.variance()?.sqrt(),
            (Self::Moments { .. }, ValueOutput::ExpectedReturn) => self.moment_summary()?.mean,
            (Self::Moments { .. }, ValueOutput::RawSecondMoment) => self.moment_summary()?.raw_second,
            (Self::Moments { .. }, ValueOutput::Variance) => self.moment_summary()?.variance,
            (Self::Moments { .. }, ValueOutput::StandardDeviation) => self.moment_summary()?
                .standard_deviation.ok_or_else(|| Error::Numerical("negative implied variance has no real standard deviation".into()))?,
            (Self::Moments { mean, raw_second }, ValueOutput::Utility) => {
                let utility = utility.ok_or_else(|| Error::Config("unknown moment utility".into()))?;
                moment_utilities(&[*mean], &[*raw_second], utility)?[0]
            }
            (Self::Twin { q1, q2 }, output) => {
                // Twin summaries are scalar Q evidence, never synthetic distributions.
                let (Self::Scalar(a), Self::Scalar(b)) = (&**q1, &**q2) else { return Err(Error::Config("twins require scalar Q predictions".into())); };
                finite(*a)?; finite(*b)?;
                match output { ValueOutput::Q1 => *a, ValueOutput::Q2 => *b, ValueOutput::Minimum => a.min(*b), ValueOutput::Disagreement => (*a - *b).abs(), _ => return Err(Error::Config("select an explicit twin statistic".into())) }
            }
            _ => return Err(Error::Config("statistic unavailable for this critic representation or unselected discrete actions".into())),
        };
        finite(value)
    }
    /// Extract a complete admitted row. Discrete callers must supply its actual
    /// legal-action mask (all-true is an explicit unmasked projection).
    pub fn from_critic(chunk: &Chunk, row: usize, legal: Option<&[bool]>) -> Result<Self> {
        let Record::Critic(critic) = &chunk.record else {
            return Err(Error::Config("expected critic probe".into()));
        };
        let get = |key: &str| {
            chunk
                .arrays
                .get(key)
                .ok_or_else(|| Error::Shape("missing critic array".into()))
        };
        let scalar = |key: &str| -> Result<f64> {
            let a = get(key)?;
            ensure(
                a.shape().len() == 1 || a.shape().len() == 2 && a.shape()[1] == 1,
                "invalid scalar critic shape",
            )?;
            number(a, row)
        };
        let vector = |key: &str| -> Result<Vec<f64>> {
            let a = get(key)?;
            ensure(
                a.shape().len() == 2
                    && row < a.shape()[0]
                    && a.shape()[1] > 0
                    && a.shape()[1] <= MAX_VALUE_HEADS,
                "invalid/oversized critic row",
            )?;
            let width = a.shape()[1];
            numbers(a, row * width..(row + 1) * width)
        };
        let prediction = match &critic.representation {
            CriticRepresentation::Scalar { values } => Self::Scalar(scalar(values)?),
            CriticRepresentation::Discrete { values } => {
                let values = vector(values)?;
                let legal = legal.ok_or_else(|| {
                    Error::Action("discrete extraction needs explicit legality".into())
                })?;
                ensure(
                    values.len() == legal.len() && legal.iter().any(|&b| b),
                    "invalid discrete legality",
                )?;
                Self::Discrete {
                    values,
                    legal: legal.to_vec(),
                }
            }
            CriticRepresentation::Quantiles {
                values,
                locations,
                masses,
            } => {
                let raw = vector(values)?;
                let grid = |key: &str| -> Result<Vec<f64>> {
                    let a = get(key)?;
                    ensure(a.shape() == [raw.len()], "invalid critic probability grid")?;
                    numbers(a, 0..raw.len())
                };
                Self::Quantiles {
                    locations: grid(locations)?,
                    masses: grid(masses)?,
                    raw,
                }
            }
            CriticRepresentation::CategoricalRegression {
                logits,
                weights,
                centers,
                values,
            } => {
                let logits = vector(logits)?;
                let weights = vector(weights)?;
                let centers = get(centers)?;
                ensure(
                    weights.len() == logits.len() && centers.shape() == [logits.len()],
                    "invalid categorical regression support",
                )?;
                Self::CategoricalRegression {
                    weights,
                    centers: numbers(centers, 0..logits.len())?,
                    logits,
                    decoded: scalar(values)?,
                }
            }
            CriticRepresentation::CategoricalReturn {
                atoms,
                probabilities,
            } => {
                let probabilities = vector(probabilities)?;
                let atoms = get(atoms)?;
                ensure(
                    atoms.shape() == [probabilities.len()],
                    "invalid categorical return support shape",
                )?;
                let prediction = Self::CategoricalReturn {
                    atoms: numbers(atoms, 0..probabilities.len())?,
                    probabilities,
                };
                prediction.weighted_atoms()?;
                prediction
            }
            CriticRepresentation::RawMoments { mean, raw_second } => Self::Moments {
                mean: scalar(mean)?,
                raw_second: scalar(raw_second)?,
            },
        };
        if let Self::Quantiles {
            raw,
            locations,
            masses,
        } = &prediction
        {
            validate_quantile_width(raw, locations, masses)?;
            validate_quantiles(raw, locations, masses)?;
        }
        Ok(prediction)
    }
}

/// Evaluate the learner's actual moment utility once for an admitted batch.
/// Inputs and arithmetic use the learner's f32 precision.
pub fn moment_utilities(
    means: &[f64],
    raw_seconds: &[f64],
    utility: &MomentUtility,
) -> Result<Vec<f64>> {
    ensure(
        !means.is_empty() && means.len() <= 1024 && means.len() == raw_seconds.len(),
        "invalid moment utility batch",
    )?;
    validate_utility(utility)?;
    for (&mean, &second) in means.iter().zip(raw_seconds) {
        finite(mean)?;
        finite(second)?;
        ensure(
            (mean as f32).is_finite() && (second as f32).is_finite(),
            "moment utility exceeds learner precision",
        )?;
    }
    let penalty = match utility {
        MomentUtility::ExpectedReturn => None,
        MomentUtility::MeanVariance { penalty } => Some(*penalty),
        MomentUtility::Unsupported { .. } => {
            return Err(Error::Config("unsupported custom moment utility".into()));
        }
    };
    means
        .iter()
        .zip(raw_seconds)
        .map(|(&mean, &second)| {
            let mean = mean as f32;
            let value = penalty.map_or(mean, |penalty| {
                mean - ((second as f32) - mean * mean).max(0.) * penalty
            });
            finite(f64::from(value))
        })
        .collect()
}
fn number(array: &Array, index: usize) -> Result<f64> {
    let value = match &array.values {
        Values::F32(v) => v.get(index).map(|&v| v as f64),
        Values::F64(v) => v.get(index).copied(),
        _ => return Err(Error::Shape("value evidence requires floating data".into())),
    }
    .ok_or_else(|| Error::Shape("value evidence row out of bounds".into()))?;
    finite(value)
}
/// Callers admit the range before reserving its returned conversion buffer.
fn numbers(array: &Array, indices: std::ops::Range<usize>) -> Result<Vec<f64>> {
    let mut values = Vec::with_capacity(indices.len());
    for i in indices {
        values.push(number(array, i)?);
    }
    Ok(values)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonCondition {
    SameInput,
    ControllerValues,
}
pub fn compatible_values(
    current: &ValueDeclaration,
    current_critic: &str,
    reference: &ValueDeclaration,
    reference_critic: &str,
    output: &ValueOutput,
    condition: ComparisonCondition,
) -> Result<()> {
    ensure(
        *output != ValueOutput::DecisionLoss,
        "decision loss is not an ordinary critic comparison",
    )?;
    current.validate()?;
    reference.validate()?;
    let a = current
        .critics
        .get(current_critic)
        .ok_or_else(|| Error::Config("current critic is undeclared".into()))?;
    let b = reference
        .critics
        .get(reference_critic)
        .ok_or_else(|| Error::Config("reference critic is undeclared".into()))?;
    ensure(
        current.conventions.is_some() && current.conventions == reference.conventions,
        "unknown/incompatible scientific value conventions",
    )?;
    ensure(
        a.continuation.is_some()
            && b.continuation.is_some()
            && a.learned_baseline != Some(false)
            && b.learned_baseline != Some(false),
        "unknown continuation or disabled learned baseline",
    )?;
    ensure(
        condition != ComparisonCondition::SameInput
            || (a.kind == ValueCriticKind::StateValue) == (b.kind == ValueCriticKind::StateValue),
        "same-input comparison cannot mix V and Q",
    )?;
    ensure(
        condition != ComparisonCondition::ControllerValues
            || *output == ValueOutput::ExpectedReturn,
        "controller comparison supports expected return only",
    )?;
    let supported = |c: &CriticDeclaration| match output {
        ValueOutput::ExpectedReturn => true,
        ValueOutput::DecisionLoss => false,
        ValueOutput::Q1 | ValueOutput::Q2 | ValueOutput::Minimum | ValueOutput::Disagreement => {
            c.twin.is_some()
        }
        ValueOutput::Quantile { probability } => {
            c.kind.is_return_distribution()
                && probability.is_finite()
                && (0. ..=1.).contains(probability)
        }
        ValueOutput::Cvar { probability } => {
            c.kind.is_return_distribution()
                && probability.is_finite()
                && *probability > 0.
                && *probability <= 1.
        }
        ValueOutput::RawSecondMoment => c.kind == ValueCriticKind::MomentQ,
        ValueOutput::Variance | ValueOutput::StandardDeviation => {
            matches!(
                c.kind,
                ValueCriticKind::MomentQ | ValueCriticKind::CategoricalReturnQ
            )
        }
        ValueOutput::Utility => {
            c.kind == ValueCriticKind::MomentQ
                && matches!(
                    c.utility,
                    Some(MomentUtility::ExpectedReturn | MomentUtility::MeanVariance { .. })
                )
        }
    };
    ensure(
        supported(a) && supported(b),
        "incompatible/unknown value statistic",
    )?;
    if *output == ValueOutput::Utility {
        ensure(a.utility == b.utility, "incompatible risk utilities")?;
    }
    Ok(())
}
/// One observed realization, requiring a complete true-terminal episode.
/// Exact episode/observation/action matching is the caller's separate contract.
pub fn trajectory_return_to_go(chunk: &Chunk, step: usize) -> Result<Option<f64>> {
    fold_trajectory_returns(chunk, step, false, |_, _| {})
}

/// Discounted suffixes for a complete recorded trajectory, using its declared
/// objective. Finite-decision suffixes describe that evaluation's playback and
/// are not admitted by [`trajectory_return_to_go`] as episodic critic evidence.
pub fn trajectory_return_suffixes(chunk: &Chunk) -> Result<Option<Vec<f64>>> {
    let mut suffixes = Vec::new();
    let complete = fold_trajectory_returns(chunk, 0, true, |index, value| {
        if suffixes.is_empty() {
            suffixes.resize(index + 1, 0.);
        }
        suffixes[index] = value;
    })?;
    Ok(complete.map(|_| suffixes))
}

fn fold_trajectory_returns(
    chunk: &Chunk,
    step: usize,
    allow_finite: bool,
    mut visit: impl FnMut(usize, f64),
) -> Result<Option<f64>> {
    let Record::Trajectory(trajectory) = &chunk.record else {
        return Err(Error::Config("expected trajectory evidence".into()));
    };
    let rewards = chunk
        .arrays
        .get(&trajectory.rewards)
        .ok_or_else(|| Error::Shape("missing trajectory rewards".into()))?;
    ensure(
        !rewards.is_empty() && rewards.len() <= MAX_RETURN_STEPS && step < rewards.len(),
        "trajectory suffix exceeds admission or step out of range",
    )?;
    chunk.validate()?;
    if !trajectory.evaluation.complete_returns
        || (!allow_finite
            && evaluation_horizon(chunk)? != EvaluationHorizon::EnvironmentTermination)
    {
        return Ok(None);
    }
    let physical = chunk
        .arrays
        .get(PHYSICAL_DURATIONS_KEY)
        .zip(chunk.arrays.get(DISCOUNTED_REWARDS_KEY))
        .map(|(durations, discounted)| -> Result<_> {
            Ok((durations.as_u64()?, discounted.as_f64()?))
        })
        .transpose()?;
    let gamma = trajectory.evaluation.discount;
    let mut total = 0.;
    for index in (step..rewards.len()).rev() {
        let (reward, continuation) = match physical {
            Some((durations, discounted)) => {
                (discounted[index], gamma.powf(durations[index] as f64))
            }
            None => (number(rewards, index)?, gamma),
        };
        total = finite(reward + continuation * total)?;
        visit(index, total);
    }
    Ok(Some(total))
}
