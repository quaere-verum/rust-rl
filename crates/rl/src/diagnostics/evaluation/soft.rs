use super::{EvaluationConfig, SoftScoring, StochasticActor, run_observation_conditioned};
use crate::{
    Error, Result,
    core::{
        action::Action,
        actor::ActMode,
        ensure,
        env::{Env, Observation},
    },
    diagnostics::{
        schema::EvaluationMetadata,
        value::{EntropyTemperatureSource, EntropyTiming, LogDensityConvention, ReturnTarget},
    },
    nn::{gaussian_policy::GaussianPolicy, policy::StochasticPolicy, to_vec},
};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone)]
pub struct SoftActionValueConfig {
    pub evaluation: EvaluationConfig,
    /// Hold this snapshot's entropy temperature fixed throughout the rollouts.
    pub alpha: f32,
}

/// Conditional soft action-value samples. They cannot be exported through the
/// environmental empirical/trajectory writer on [`super::EvaluationResult`].
#[derive(Debug, Clone)]
pub struct SoftActionValueSamples {
    pub returns: Vec<f64>,
    pub seeds: Vec<u64>,
    pub metadata: EvaluationMetadata,
    pub target: ReturnTarget,
    pub conditioning_observation: Observation,
    pub initial_action: Action,
    pub elapsed_seconds: f64,
    pub environment_steps: u64,
    pub physical_steps: u64,
}

/// Evaluate the target semantics of one-step SAC replay from a fixed action:
/// keep the first reward environmental, then apply `-alpha * ln π(a|s)` at
/// each subsequent sampled decision. A fresh conditional environment and an
/// independent seeded policy stream are used for every scenario. A terminal
/// transition supplies no successor entropy or bootstrap. The passed policy
/// and temperature remain frozen for the whole estimate.
pub fn evaluate_soft_action_value<E: Env>(
    factory: impl FnMut(u64) -> Result<(E, Observation)>,
    observation: &Observation,
    first_action: &Action,
    policy: &GaussianPolicy,
    config: &SoftActionValueConfig,
    methodology: &str,
) -> Result<SoftActionValueSamples> {
    ensure(
        config.alpha.is_finite()
            && config.alpha >= 0.
            && config.evaluation.mode == ActMode::Sample
            && config.evaluation.trajectories == 0
            && !config.evaluation.allow_truncated_prefix
            && config
                .evaluation
                .seeds
                .len()
                .saturating_mul(config.evaluation.max_steps)
                <= 1_000_000,
        "soft action-value evaluation requires bounded, complete sampled rollouts without retained trajectories",
    )?;
    let actor = StochasticActor::new(policy);
    let sample = |observation: &Observation, rng: &mut ChaCha8Rng| {
        let draw = actor
            .model
            .distribution(std::slice::from_ref(observation).into())?
            .reparameterize_with_sampled_score(rng)?;
        let scores = to_vec(draw.log_probs)?;
        let [score] = scores.as_slice() else {
            return Err(Error::Shape(
                "soft evaluation requires a single action log density".into(),
            ));
        };
        Ok((Action::Continuous(to_vec(draw.actions)?), *score))
    };
    let result = run_observation_conditioned(
        factory,
        observation,
        Some(first_action),
        &actor,
        &config.evaluation,
        methodology,
        Some(SoftScoring {
            alpha: config.alpha,
            sample: &sample,
        }),
    )?;
    let mut metadata = result.metadata;
    metadata.truncation_handling =
        "complete terminal soft returns; scored successor actions; no bootstrap".into();
    Ok(SoftActionValueSamples {
        returns: result.returns,
        seeds: result.seeds,
        metadata,
        target: ReturnTarget::SoftActionValue {
            alpha: config.alpha,
            temperature_source: EntropyTemperatureSource::IndependentEvaluation,
            entropy_timing: EntropyTiming::ReplayBootstrap { n_step: 1 },
            log_density: LogDensityConvention::PolicySampleInActionCoordinates,
        },
        conditioning_observation: observation.clone(),
        initial_action: first_action.clone(),
        elapsed_seconds: result.elapsed_seconds,
        environment_steps: result.environment_steps,
        physical_steps: result.physical_steps,
    })
}
