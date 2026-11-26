use super::{ReferenceContext, ReferenceKind, ReferenceQuality, codec, text};
use crate::{
    Result,
    core::{
        action::Action,
        actor::{ActMode, Actor},
        ensure,
        env::Observation,
    },
    train::policy_comparison::{PairedDescriptives, compare_paired_returns},
};
use serde::{Deserialize, Serialize};

use crate::diagnostics::{
    analysis::QuantileForecast,
    evaluation::EvaluationResult,
    schema::{Array, Chunk, Conditioning},
    typed::DiagnosticInput,
};

const PROVIDER_KEY: &str = "metadata/reference-provider.v1";

/// The mathematical object a provider's value predictions describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceTarget {
    BenchmarkPolicy,
    VerifiedOptimal,
    LearnerPolicy,
    CriticContinuation,
}

/// Whether predictions can be reused at an unchanged scientific input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceLifetime {
    Fixed,
    CheckpointDependent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceInformation {
    LearnerObservation,
    PrivilegedSimulatorState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSpec {
    pub id: String,
    pub context: ReferenceContext,
    pub quality: ReferenceQuality,
    pub target: ReferenceTarget,
    pub lifetime: ReferenceLifetime,
    pub information: ReferenceInformation,
    pub uncertainty: Option<String>,
}

impl ReferenceSpec {
    pub fn validate(&self) -> Result<()> {
        text(&self.id)?;
        ensure(
            self.id.len() <= 64
                && self
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')),
            "invalid reference provider identifier",
        )?;
        self.context.validate()?;
        text(&self.quality.method)?;
        ensure(
            self.quality.assumptions.len() <= 16,
            "too many reference assumptions",
        )?;
        for assumption in &self.quality.assumptions {
            text(assumption)?;
        }
        if let Some(description) = &self.uncertainty {
            text(description)?;
        }
        ensure(
            (self.target != ReferenceTarget::VerifiedOptimal
                || self.quality.kind == ReferenceKind::VerifiedOptimal)
                && (!matches!(
                    self.target,
                    ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation
                ) || self.lifetime == ReferenceLifetime::CheckpointDependent),
            "reference quality or lifetime does not support its declared target",
        )?;
        codec::bounded_json(self, super::MAX_REFERENCE_METADATA_BYTES)?;
        Ok(())
    }

    pub fn model_role(&self) -> String {
        format!("reference:{}", self.id)
    }
}

/// The evaluation-only policy is frozen at the requested learner revision.
/// Providers may evaluate it during a call and cannot retain the borrowed actor.
pub struct ReferenceCheckpoint<'a> {
    pub revision: u64,
    pub policy_id: &'a str,
    pub policy: &'a dyn Actor,
    pub mode: ActMode,
}

/// The state domain on which a provider certifies its optimal decision values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimalDecisionDomain {
    MarkovObservation,
    ModelCoordinates,
}

/// An authored diagnostic reference. Optional methods express capability
/// availability for each query, including domain boundaries.
pub trait ReferenceProvider {
    fn specification(&self) -> &ReferenceSpec;

    fn optimal_decision_domain(&self) -> Option<OptimalDecisionDomain> {
        let spec = self.specification();
        (spec.target == ReferenceTarget::VerifiedOptimal && spec.context.markov_observation)
            .then_some(OptimalDecisionDomain::MarkovObservation)
    }

    fn evaluation_mode(&self, requested: ActMode) -> ActMode {
        requested
    }

    fn actor(&self) -> Option<&dyn Actor> {
        None
    }

    fn state_value(&self, _: &ReferenceCheckpoint<'_>, _: &Observation) -> Result<Option<f64>> {
        Ok(None)
    }

    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        _: &Observation,
        _: &Action,
    ) -> Result<Option<f64>> {
        Ok(None)
    }

    /// Typed reference query. History-dependent providers can use the complete
    /// public history or decline inputs outside their declared information set.
    fn action_value_for_input(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        input: &DiagnosticInput,
        action: &Action,
    ) -> Result<Option<f64>> {
        self.action_value(checkpoint, input.observation(), action)
    }

    fn action_gradient(
        &self,
        _: &ReferenceCheckpoint<'_>,
        _: &Observation,
        _: &Action,
    ) -> Result<Option<ActionGradient>> {
        Ok(None)
    }

    fn return_distribution(
        &self,
        _: &ReferenceCheckpoint<'_>,
        _: &Observation,
        _: &Action,
    ) -> Result<Option<QuantileForecast>> {
        Ok(None)
    }
}

/// An action-only benchmark. Its value functions remain unavailable.
pub struct BenchmarkActor<A> {
    specification: ReferenceSpec,
    actor: A,
    mode: Option<ActMode>,
}

impl<A: Actor> BenchmarkActor<A> {
    pub fn new(specification: ReferenceSpec, actor: A) -> Result<Self> {
        specification.validate()?;
        ensure(
            !matches!(
                specification.target,
                ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation
            ) && specification.lifetime == ReferenceLifetime::Fixed
                && actor.action_space() == &specification.context.action_space,
            "benchmark actor conflicts with reference declaration",
        )?;
        Ok(Self {
            specification,
            actor,
            mode: None,
        })
    }

    pub fn with_evaluation_mode(mut self, mode: ActMode) -> Self {
        self.mode = Some(mode);
        self
    }
}

impl<A: Actor> ReferenceProvider for BenchmarkActor<A> {
    fn evaluation_mode(&self, requested: ActMode) -> ActMode {
        self.mode.unwrap_or(requested)
    }
    fn specification(&self) -> &ReferenceSpec {
        &self.specification
    }
    fn actor(&self) -> Option<&dyn Actor> {
        Some(&self.actor)
    }
}

pub(crate) fn attach_provider_spec(chunk: &mut Chunk, spec: &ReferenceSpec) -> Result<()> {
    spec.validate()?;
    let bytes = codec::bounded_json(spec, super::MAX_REFERENCE_METADATA_BYTES)?;
    ensure(
        !chunk.arrays.contains_key(PROVIDER_KEY),
        "reference provenance already recorded",
    )?;
    chunk
        .arrays
        .insert(PROVIDER_KEY.into(), Array::u8(vec![bytes.len()], bytes)?);
    Ok(())
}

/// Read the provider provenance from a portable chunk, including summaries and
/// independent evaluations. Existing recordings can omit this declaration.
pub fn provider_specification(chunk: &Chunk) -> Result<Option<ReferenceSpec>> {
    let Some(array) = chunk.arrays.get(PROVIDER_KEY) else {
        return Ok(None);
    };
    ensure(
        array.shape().len() == 1 && array.byte_len() <= super::MAX_REFERENCE_METADATA_BYTES,
        "invalid reference provider declaration",
    )?;
    let spec: ReferenceSpec = serde_json::from_slice(array.as_u8()?)?;
    spec.validate()?;
    Ok(Some(spec))
}

/// Compare complete independently seeded initial-scenario runs using distinct
/// matched scenarios as the sampling units. The gap is reference minus learner.
pub fn performance_gap(
    learner: &EvaluationResult,
    reference: &EvaluationResult,
) -> Result<PairedDescriptives> {
    let a = &learner.metadata;
    let b = &reference.metadata;
    ensure(
        a.conditioning == Conditioning::InitialScenario
            && b.conditioning == Conditioning::InitialScenario
            && a.complete_returns
            && b.complete_returns
            && a.independent_future_streams
            && b.independent_future_streams
            && a.discount == b.discount
            && learner.horizon == reference.horizon
            && learner.seeds == reference.seeds
            && learner.seeds.len() == learner.returns.len()
            && learner.returns.len() == reference.returns.len(),
        "performance comparison needs complete matched independent scenarios and return conventions",
    )?;
    compare_paired_returns(
        &learner
            .returns
            .iter()
            .copied()
            .zip(reference.returns.iter().copied())
            .collect::<Vec<_>>(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticError {
    pub count: usize,
    pub bias: f64,
    pub mae: f64,
    pub rmse: f64,
}

/// Pointwise prediction minus reference, summarized on the supplied population.
/// Absent weights mean equal weight on each input. Zero-weight inputs are omitted.
pub fn critic_error(
    predicted: &[f64],
    reference: &[f64],
    weights: Option<&[f64]>,
) -> Result<CriticError> {
    ensure(
        !predicted.is_empty()
            && predicted.len() == reference.len()
            && weights.is_none_or(|weights| weights.len() == predicted.len()),
        "critic error needs an aligned nonempty probe population",
    )?;
    let weighted_errors = predicted
        .iter()
        .zip(reference)
        .enumerate()
        .map(|(i, (a, b))| (a - b, weights.map_or(1., |weights| weights[i])));
    let mut error_scale: f64 = 0.;
    let mut weight_scale: f64 = 0.;
    let mut count = 0;
    for (error, weight) in weighted_errors.clone() {
        ensure(
            error.is_finite() && weight.is_finite() && weight >= 0.,
            "invalid critic error or weight",
        )?;
        if weight > 0. {
            count += 1;
            error_scale = error_scale.max(error.abs());
            weight_scale = weight_scale.max(weight);
        }
    }
    ensure(count > 0, "critic error needs positive population weight")?;
    if error_scale == 0. {
        return Ok(CriticError {
            count,
            bias: 0.,
            mae: 0.,
            rmse: 0.,
        });
    }
    let mut totals = [0.; 4];
    for (error, weight) in weighted_errors.filter(|(_, weight)| *weight > 0.) {
        let error = error / error_scale;
        let weight = weight / weight_scale;
        totals[0] += weight;
        totals[1] += weight * error;
        totals[2] += weight * error.abs();
        totals[3] += weight * error * error;
    }
    Ok(CriticError {
        count,
        bias: (totals[1] / totals[0]).clamp(-1., 1.) * error_scale,
        mae: (totals[2] / totals[0]).min(1.) * error_scale,
        rmse: (totals[3] / totals[0]).min(1.).sqrt() * error_scale,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousDecisionLoss {
    pub optimal_value: f64,
    pub expected_action_value: f64,
    pub loss: f64,
    pub standard_error: Option<f64>,
}

/// Use a verified continuous optimum and Q* of the deterministic action or
/// independent draws from the actual stochastic policy.
pub fn continuous_decision_loss(
    optimal_value: f64,
    action_values: &[f64],
) -> Result<ContinuousDecisionLoss> {
    ensure(optimal_value.is_finite(), "nonfinite optimal state value")?;
    let tolerance = optimal_value.abs().max(1.) * (32. * f64::EPSILON);
    ensure(
        action_values
            .iter()
            .all(|&q| q.is_finite() && q - optimal_value <= tolerance),
        "action value exceeds verified optimum",
    )?;
    let pairs = action_values
        .iter()
        .map(|&q| (q, optimal_value))
        .collect::<Vec<_>>();
    let stats = compare_paired_returns(&pairs)?;
    Ok(ContinuousDecisionLoss {
        optimal_value,
        expected_action_value: stats.incumbent_mean,
        loss: stats.mean_difference.max(0.),
        standard_error: stats.difference_standard_error,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GradientMethod {
    Analytic,
    AutomaticDifferentiation,
    FiniteDifference,
}

#[derive(Debug, Clone)]
pub struct ActionGradient {
    pub values: Vec<f64>,
    pub method: GradientMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionGradientComparison {
    pub predicted_method: GradientMethod,
    pub reference_method: GradientMethod,
    pub magnitude_error: f64,
    pub vector_error: f64,
    pub cosine: Option<f64>,
}

pub fn action_gradient_comparison(
    predicted: &[f64],
    predicted_method: GradientMethod,
    reference: &[f64],
    reference_method: GradientMethod,
) -> Result<ActionGradientComparison> {
    ensure(
        !predicted.is_empty()
            && predicted.len() == reference.len()
            && predicted.iter().chain(reference).all(|n| n.is_finite()),
        "action gradient comparison needs aligned finite vectors",
    )?;
    let norm = |values: &[f64]| values.iter().fold(0.0_f64, |norm, &n| norm.hypot(n));
    let a = norm(predicted);
    let b = norm(reference);
    let vector_error = predicted
        .iter()
        .zip(reference)
        .fold(0.0_f64, |norm, (a, b)| norm.hypot(a - b));
    ensure(
        a.is_finite() && b.is_finite() && vector_error.is_finite(),
        "nonfinite action gradient norm",
    )?;
    let cosine = (a > 0. && b > 0.).then(|| {
        predicted
            .iter()
            .zip(reference)
            .map(|(x, y)| (x / a) * (y / b))
            .sum::<f64>()
            .clamp(-1., 1.)
    });
    Ok(ActionGradientComparison {
        predicted_method,
        reference_method,
        magnitude_error: (a - b).abs(),
        vector_error,
        cosine,
    })
}
