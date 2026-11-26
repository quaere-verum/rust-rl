//! Independent evaluation with explicit conditioning and owned random streams.

mod clipped;
#[cfg(feature = "neural")]
mod soft;
#[cfg(feature = "neural")]
pub use soft::{SoftActionValueConfig, SoftActionValueSamples, evaluate_soft_action_value};

pub use super::render::RenderCaptureConfig;
use super::{
    render::{RenderFrames, RenderSource},
    schema::*,
    typed::{self, TypedEpisode, TypedEpisodeSource, TypedPayload, TypedStep},
};
use crate::{
    Error, Result,
    core::{
        action::Action,
        actor::{ActMode, Actor},
        ensure,
        env::{Env, Observation},
    },
};
#[cfg(feature = "neural")]
use crate::{
    core::{action::ActionSpace, env::ObservationBatch},
    nn::{
        policy::{DeterministicPolicy, StochasticPolicy},
        target::frozen_copy,
    },
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

const FINITE_HORIZON_KEY: &str = "evaluation/finite_decision_horizon.v1";
const TRAJECTORY_SCENARIO_SEED_KEY: &str = "trajectory/scenario_seed.v1";
pub(super) const PHYSICAL_DURATIONS_KEY: &str = "trajectory/physical_durations.v1";
pub(super) const DISCOUNTED_REWARDS_KEY: &str = "trajectory/discounted_step_rewards.v1";

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub seeds: Vec<u64>,
    pub policy_seed: u64,
    pub mode: ActMode,
    pub discount: f64,
    pub max_steps: usize,
    pub trajectories: usize,
    pub allow_truncated_prefix: bool,
    pub policy_id: String,
}
impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            seeds: (0..32).collect(),
            policy_seed: 0,
            mode: ActMode::Deterministic,
            discount: 0.99,
            max_steps: 10_000,
            trajectories: 1,
            allow_truncated_prefix: false,
            policy_id: "caller-supplied policy; identity unspecified".into(),
        }
    }
}
impl EvaluationConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            !self.seeds.is_empty()
                && self.max_steps > 0
                && self.trajectories <= self.seeds.len()
                && self.discount.is_finite()
                && (0. ..=1.).contains(&self.discount)
                && !self.policy_id.is_empty(),
            "invalid evaluation configuration",
        )?;
        ensure(
            self.seeds.iter().collect::<BTreeSet<_>>().len() == self.seeds.len(),
            "evaluation needs distinct future seeds within a Monte Carlo sample",
        )
    }
}

/// Return objective for evaluation from an initial scenario. `max_steps`
/// remains a collection guard, including when a finite objective is selected.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EvaluationHorizon {
    #[default]
    EnvironmentTermination,
    /// Stop after this many decisions unless the environment terminates first.
    /// Primitive-time reward discounting still applies within each decision.
    FiniteDecisions(usize),
}
impl EvaluationHorizon {
    pub(super) fn completes(self, decisions: usize, terminated: bool) -> bool {
        terminated || self == Self::FiniteDecisions(decisions)
    }

    fn write(self, arrays: &mut BTreeMap<ArrayKey, Array>) -> Result<()> {
        if let Self::FiniteDecisions(decisions) = self {
            arrays.insert(
                FINITE_HORIZON_KEY.into(),
                Array::u64(vec![1], vec![decisions as u64])?,
            );
        }
        Ok(())
    }
}

/// Read the declared horizon of trajectory or empirical evidence. Historical
/// recordings without an explicit horizon use environment termination.
pub fn evaluation_horizon(chunk: &Chunk) -> Result<EvaluationHorizon> {
    if chunk.record.kind() == RecordKind::TypedEpisode {
        let TypedPayload::Episode(episode) = TypedPayload::read(chunk)? else {
            unreachable!("validated typed episode kind")
        };
        return match episode.source {
            TypedEpisodeSource::Evaluation { horizon, .. } => Ok(horizon),
            TypedEpisodeSource::Factual { .. } => Err(Error::State(
                "factual episode has no evaluation horizon".into(),
            )),
        };
    }
    ensure(
        matches!(chunk.record, Record::Trajectory(_) | Record::Empirical(_)),
        "evaluation horizon requires trajectory or empirical evidence",
    )?;
    let Some(array) = chunk.arrays.get(FINITE_HORIZON_KEY) else {
        return Ok(EvaluationHorizon::EnvironmentTermination);
    };
    ensure(
        array.shape() == [1],
        "invalid finite evaluation horizon shape",
    )?;
    let decisions = usize::try_from(array.as_u64()?[0])
        .map_err(|_| Error::Config("finite evaluation horizon exceeds decision count".into()))?;
    ensure(decisions > 0, "finite evaluation horizon must be positive")?;
    Ok(EvaluationHorizon::FiniteDecisions(decisions))
}

/// Read the initial-scenario seed of a recorded evaluation trajectory.
/// Historical trajectories and conditionally sampled branches can omit it.
pub fn trajectory_scenario_seed(chunk: &Chunk) -> Result<Option<u64>> {
    if chunk.record.kind() == RecordKind::TypedEpisode {
        let TypedPayload::Episode(episode) = TypedPayload::read(chunk)? else {
            unreachable!()
        };
        return Ok(match episode.source {
            TypedEpisodeSource::Evaluation { scenario_seed, .. } => Some(scenario_seed),
            TypedEpisodeSource::Factual { .. } => None,
        });
    }
    let Record::Trajectory(trajectory) = &chunk.record else {
        return Err(Error::Config(
            "scenario seed requires a trajectory record".into(),
        ));
    };
    let Some(array) = chunk.arrays.get(TRAJECTORY_SCENARIO_SEED_KEY) else {
        return Ok(None);
    };
    ensure(
        array.shape() == [1] && trajectory.evaluation.conditioning == Conditioning::InitialScenario,
        "invalid trajectory scenario seed metadata",
    )?;
    Ok(Some(array.as_u64()?[0]))
}

/// The observed episode prefix at a state boundary. Rewards are the recorded
/// undiscounted environment rewards of completed actions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryProgress {
    pub observed_reward: f64,
    pub primitive_steps: u64,
    pub simulation_time: Option<f64>,
}

pub fn trajectory_progress(chunk: &Chunk, state: usize) -> Result<TrajectoryProgress> {
    let Record::Trajectory(trajectory) = &chunk.record else {
        return Err(Error::Config(
            "trajectory progress requires an episode".into(),
        ));
    };
    let array = |key: &str| {
        chunk
            .arrays
            .get(key)
            .ok_or_else(|| Error::Shape(format!("missing trajectory array {key}")))
    };
    let rewards = array(&trajectory.rewards)?;
    let transitions = rewards.len();
    ensure(
        state <= transitions,
        "trajectory state exceeds the final observation",
    )?;
    let observed_reward: f64 = match rewards.dtype() {
        DType::F32 => rewards.as_f32()?[..state]
            .iter()
            .copied()
            .map(f64::from)
            .sum(),
        DType::F64 => rewards.as_f64()?[..state].iter().sum(),
        _ => {
            return Err(Error::Shape(
                "trajectory rewards require floating-point values".into(),
            ));
        }
    };
    ensure(
        observed_reward.is_finite(),
        "trajectory cumulative reward overflow",
    )?;
    let primitive_steps = if let Some(durations) = chunk.arrays.get(PHYSICAL_DURATIONS_KEY) {
        ensure(
            durations.shape() == [transitions],
            "invalid trajectory duration shape",
        )?;
        durations.as_u64()?[..state]
            .iter()
            .try_fold(0u64, |elapsed, &duration| {
                elapsed
                    .checked_add(duration)
                    .ok_or_else(|| Error::Shape("trajectory physical clock overflow".into()))
            })?
    } else {
        // Legacy recordings use one primitive step per recorded transition.
        u64::try_from(state)
            .map_err(|_| Error::Shape("trajectory physical clock overflow".into()))?
    };
    let simulation_time = trajectory
        .simulation_times
        .as_ref()
        .map(|key| {
            let times = array(key)?;
            let time = match times.dtype() {
                DType::F32 => times.as_f32()?.get(state).copied().map(f64::from),
                DType::F64 => times.as_f64()?.get(state).copied(),
                _ => None,
            };
            time.filter(|value| value.is_finite())
                .ok_or_else(|| Error::Shape("invalid trajectory simulation clock".into()))
        })
        .transpose()?;
    Ok(TrajectoryProgress {
        observed_reward,
        primitive_steps,
        simulation_time,
    })
}

/// Opt-in physical state branching, independent of `Env` and GUI code. Branching
/// replaces future randomness; cloning/restoring an RNG alone is insufficient.
pub trait BranchingEnv: Env + Sized {
    fn branch(&self, future_seed: u64) -> Result<Self>;
    fn branch_observation(&self) -> Result<Observation>;
    fn branch_state(&self) -> Result<Vec<f64>>;
    fn branch_description(&self) -> Result<String>;
}
/// Numeric presentation channels. Custom environments can provide this through
/// `evaluate_with_frames`; ordinary environments need not implement this trait.
#[derive(Debug, Clone)]
pub struct EnvironmentFrame {
    pub environment: String,
    pub simulation_time: Option<f64>,
    pub channels: BTreeMap<String, Vec<f64>>,
}
pub trait FrameSource {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame>;
}
#[derive(Debug, Clone)]
pub struct EvaluatedTrajectory {
    pub render: Option<RenderFrames>,
    pub seed: u64,
    pub observations: Vec<Observation>,
    pub actions: Vec<Action>,
    pub rewards: Vec<f32>,
    /// Number of primitive ticks represented by each decision.
    pub physical_durations: Vec<u64>,
    /// Within-decision reward discounted from the start of that decision.
    pub discounted_step_rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub frames: Vec<EnvironmentFrame>,
}
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    pub returns: Vec<f64>,
    pub seeds: Vec<u64>,
    pub trajectories: Vec<EvaluatedTrajectory>,
    pub metadata: EvaluationMetadata,
    pub horizon: EvaluationHorizon,
    pub initial_action: Option<Action>,
    pub conditioning_observation: Option<Observation>,
    pub simulator_state: Option<Vec<f64>>,
    pub elapsed_seconds: f64,
    /// Number of environment decisions across all evaluated scenarios.
    pub environment_steps: u64,
    /// Number of primitive environment ticks across those decisions.
    pub physical_steps: u64,
}
#[derive(Debug, Clone, Copy)]
pub struct EvaluationProgress {
    pub completed: usize,
    pub total: usize,
    pub environment_steps: u64,
    pub physical_steps: u64,
    pub elapsed_seconds: f64,
}

/// Owns an isolated inference model, including custom running statistics.
#[cfg(feature = "neural")]
pub struct StochasticActor<P: StochasticPolicy> {
    model: P,
}
#[cfg(feature = "neural")]
impl<P: StochasticPolicy> StochasticActor<P> {
    pub fn new(model: &P) -> Self {
        Self {
            model: frozen_copy(model),
        }
    }
}
#[cfg(feature = "neural")]
impl<P: StochasticPolicy> Actor for StochasticActor<P> {
    fn action_space(&self) -> &ActionSpace {
        self.model.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        self.model
            .distribution(obs)?
            .sample_actions_into(mode, rng, out)
    }
}
#[cfg(feature = "neural")]
pub struct DeterministicActor<P: DeterministicPolicy> {
    model: P,
}
#[cfg(feature = "neural")]
impl<P: DeterministicPolicy> DeterministicActor<P> {
    pub fn new(model: &P) -> Self {
        Self {
            model: frozen_copy(model),
        }
    }
}
#[cfg(feature = "neural")]
impl<P: DeterministicPolicy> Actor for DeterministicActor<P> {
    fn action_space(&self) -> &ActionSpace {
        self.model.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        let actions = super::probes::continuous_actions(self.model.actions(obs)?)?;
        out.clear();
        out.extend(actions);
        Ok(())
    }
}
pub fn evaluate<E: Env>(
    factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
) -> Result<EvaluationResult> {
    evaluate_with_horizon(factory, actor, config, EvaluationHorizon::default())
}

pub fn evaluate_with_horizon<E: Env>(
    mut factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    horizon: EvaluationHorizon,
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let mut env = factory(seed)?;
            let obs = env.reset(Some(seed))?.observation;
            Ok((env, obs))
        },
        actor,
        config,
        Start::initial(horizon),
        None,
        |_| Ok(None),
        |_| {},
    )
}
pub fn evaluate_with_frames<E: Env + FrameSource>(
    factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    evaluate_with_frames_and_horizon(
        factory,
        actor,
        config,
        EvaluationHorizon::default(),
        progress,
    )
}

pub fn evaluate_with_frames_and_horizon<E: Env + FrameSource>(
    mut factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    horizon: EvaluationHorizon,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let mut env = factory(seed)?;
            let obs = env.reset(Some(seed))?.observation;
            Ok((env, obs))
        },
        actor,
        config,
        Start::initial(horizon),
        None,
        |env| env.diagnostic_frame().map(Some),
        progress,
    )
}
/// Capture bounded scenes only for the explicitly selected evaluation episodes.
pub fn evaluate_with_render<E: Env + RenderSource>(
    factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    capture: &RenderCaptureConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    evaluate_with_render_and_horizon(
        factory,
        actor,
        config,
        EvaluationHorizon::default(),
        capture,
        progress,
    )
}

/// Capture selected rendered trajectories with the same objective horizon as
/// the numerical evaluation.
pub fn evaluate_with_render_and_horizon<E: Env + RenderSource>(
    mut factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    horizon: EvaluationHorizon,
    capture: &RenderCaptureConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let mut env = factory(seed)?;
            let obs = env.reset(Some(seed))?.observation;
            Ok((env, obs))
        },
        actor,
        config,
        Start::initial(horizon),
        Some((capture, E::render_scene)),
        |_| Ok(None),
        progress,
    )
}
/// Capture existing numeric frames together with bounded rendered states.
pub fn evaluate_with_frames_and_render<E: Env + FrameSource + RenderSource>(
    factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    capture: &RenderCaptureConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    evaluate_with_frames_and_render_and_horizon(
        factory,
        actor,
        config,
        EvaluationHorizon::default(),
        capture,
        progress,
    )
}

/// Capture numerical frames and rendered scenes under the same declared objective horizon.
pub fn evaluate_with_frames_and_render_and_horizon<E: Env + FrameSource + RenderSource>(
    mut factory: impl FnMut(u64) -> Result<E>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    horizon: EvaluationHorizon,
    capture: &RenderCaptureConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let mut env = factory(seed)?;
            let obs = env.reset(Some(seed))?.observation;
            Ok((env, obs))
        },
        actor,
        config,
        Start::initial(horizon),
        Some((capture, E::render_scene)),
        |env| env.diagnostic_frame().map(Some),
        progress,
    )
}

/// Evaluate an authored observation-conditional experiment. Each seed must
/// initialize a fresh environment from the declared conditional distribution
/// and independently seed its future randomness. The factory returns the
/// environment already initialized and its actual public observation.
pub fn evaluate_observation_conditioned<E: Env>(
    factory: impl FnMut(u64) -> Result<(E, Observation)>,
    observation: &Observation,
    first_action: Option<&Action>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    methodology: &str,
) -> Result<EvaluationResult> {
    run_observation_conditioned(
        factory,
        observation,
        first_action,
        actor,
        config,
        methodology,
        None,
    )
}

fn run_observation_conditioned<E: Env>(
    mut factory: impl FnMut(u64) -> Result<(E, Observation)>,
    observation: &Observation,
    first_action: Option<&Action>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    methodology: &str,
    soft_scoring: Option<SoftScoring<'_>>,
) -> Result<EvaluationResult> {
    ensure(
        !methodology.trim().is_empty()
            && methodology.len() <= 4096
            && !methodology.chars().any(char::is_control),
        "invalid observation-conditional sampling methodology",
    )?;
    run(
        |seed| {
            let (env, actual) = factory(seed)?;
            ensure(
                &actual == observation,
                "conditional sampler observation or legal-action mask differs from its declared input",
            )?;
            Ok((env, actual))
        },
        actor,
        config,
        Start {
            conditioning: Conditioning::Observation {
                methodology: methodology.into(),
            },
            action: first_action,
            observation: Some(observation.clone()),
            state: None,
            horizon: EvaluationHorizon::EnvironmentTermination,
            soft_scoring,
        },
        None,
        |_| Ok(None),
        |_| {},
    )
}

pub fn evaluate_conditional<E: BranchingEnv>(
    source: &E,
    actor: &dyn Actor,
    first_action: Option<&Action>,
    config: &EvaluationConfig,
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let env = source.branch(seed)?;
            let obs = env.branch_observation()?;
            Ok((env, obs))
        },
        actor,
        config,
        Start::conditional(source, first_action)?,
        None,
        |_| Ok(None),
        |_| {},
    )
}
pub fn evaluate_conditional_with_frames<E: BranchingEnv + FrameSource>(
    source: &E,
    actor: &dyn Actor,
    first_action: Option<&Action>,
    config: &EvaluationConfig,
    progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    run(
        |seed| {
            let env = source.branch(seed)?;
            let obs = env.branch_observation()?;
            Ok((env, obs))
        },
        actor,
        config,
        Start::conditional(source, first_action)?,
        None,
        |env| env.diagnostic_frame().map(Some),
        progress,
    )
}
type ScoredAction<'a> = dyn Fn(&Observation, &mut ChaCha8Rng) -> Result<(Action, f32)> + 'a;
struct SoftScoring<'a> {
    alpha: f32,
    sample: &'a ScoredAction<'a>,
}

struct Start<'a> {
    conditioning: Conditioning,
    action: Option<&'a Action>,
    observation: Option<Observation>,
    state: Option<Vec<f64>>,
    horizon: EvaluationHorizon,
    soft_scoring: Option<SoftScoring<'a>>,
}
impl<'a> Start<'a> {
    fn initial(horizon: EvaluationHorizon) -> Self {
        Self {
            conditioning: Conditioning::InitialScenario,
            action: None,
            observation: None,
            state: None,
            horizon,
            soft_scoring: None,
        }
    }
    fn conditional(source: &impl BranchingEnv, action: Option<&'a Action>) -> Result<Self> {
        let observation = source.branch_observation()?;
        if let Some(action) = action {
            source
                .action_space()
                .check(action, observation.dense_mask())?;
        }
        Ok(Self {
            conditioning: Conditioning::FullSimulatorState {
                description: source.branch_description()?,
            },
            action,
            observation: Some(observation),
            state: Some(source.branch_state()?),
            horizon: EvaluationHorizon::EnvironmentTermination,
            soft_scoring: None,
        })
    }
}
type RenderCapture<'a, E> = Option<(
    &'a RenderCaptureConfig,
    fn(&E) -> Result<super::render::RenderScene>,
)>;
fn run<E: Env>(
    mut start_env: impl FnMut(u64) -> Result<(E, Observation)>,
    actor: &dyn Actor,
    config: &EvaluationConfig,
    start: Start<'_>,
    render: RenderCapture<'_, E>,
    mut capture: impl FnMut(&E) -> Result<Option<EnvironmentFrame>>,
    mut progress: impl FnMut(EvaluationProgress),
) -> Result<EvaluationResult> {
    config.validate()?;
    if let EvaluationHorizon::FiniteDecisions(decisions) = start.horizon {
        ensure(
            (1..=config.max_steps).contains(&decisions),
            "finite evaluation objective must fit inside the collection guard",
        )?;
    }
    if let Some((capture, _)) = render {
        capture.limits.validate()?;
        ensure(
            capture.trajectories <= config.trajectories,
            "render trajectories exceed selected evaluation trajectories",
        )?;
    }
    let mut render_bytes = 0usize;
    let clock = Instant::now();
    let mut returns = Vec::with_capacity(config.seeds.len());
    let mut trajectories = Vec::with_capacity(config.trajectories);
    let mut environment_steps = 0u64;
    let mut physical_steps = 0u64;
    let mut any_incomplete = false;
    let mut actions = Vec::with_capacity(1);
    let mut typed_budget = typed::TypedBudget::new(typed::MAX_TYPED_BYTES);
    // Derive a separate key in a reserved domain before assigning scenario
    // streams. Environment constructors use seed_from_u64(seed), stream zero;
    // simply setting the policy stream to the scenario seed collided at zero.
    let mut key_rng = ChaCha8Rng::seed_from_u64(config.policy_seed);
    key_rng.set_stream(0x524c_504f_4c49_4359); // "RLPOLICY"
    let mut policy_key = [0; 32];
    key_rng.fill_bytes(&mut policy_key);
    for (sample, &seed) in config.seeds.iter().enumerate() {
        let (mut env, mut observation) = start_env(seed)?;
        ensure(
            env.action_space() == actor.action_space(),
            "evaluation actor/environment action spaces differ",
        )?;
        let observation_space = env.observation_space();
        let typed_input = observation_space.dense_width().is_none();
        observation.validate_in(&observation_space, env.action_space(), true)?;
        if typed_input && sample < config.trajectories {
            typed_budget.observation(&observation)?;
        }
        // A separate ChaCha stream per scenario gives reproducibility without
        // consuming any actor/collector/learner training RNG.
        let mut rng = ChaCha8Rng::from_seed(policy_key);
        rng.set_stream(seed);
        let mut trajectory = (sample < config.trajectories).then(|| EvaluatedTrajectory {
            seed,
            observations: vec![observation.clone()],
            actions: vec![],
            rewards: vec![],
            physical_durations: vec![],
            discounted_step_rewards: vec![],
            terminated: vec![],
            truncated: vec![],
            frames: vec![],
            render: render
                .filter(|(c, _)| sample < c.trajectories)
                .map(|_| RenderFrames::new()),
        });
        if let Some(trajectory) = &mut trajectory
            && let Some(frame) = capture(&env)?
        {
            trajectory.frames.push(frame);
        }
        if let Some(trajectory) = &mut trajectory
            && let Some(frames) = &mut trajectory.render
            && let Some((capture, source)) = render
        {
            frames.capture(&source(&env)?, &capture.limits, &mut render_bytes)?;
        }
        let mut total = 0.;
        let mut discount = 1.;
        let mut finished = false;
        for t in 0..config.max_steps {
            actions.clear();
            let mut soft_reward = None;
            if t == 0
                && let Some(action) = start.action
            {
                actions.push(action.clone());
            } else if let Some(soft) = &start.soft_scoring {
                let (action, log_density) = (soft.sample)(&observation, &mut rng)?;
                actions.push(action);
                soft_reward = Some(-f64::from(soft.alpha) * f64::from(log_density));
            } else {
                actor.act_into(
                    std::slice::from_ref(&observation).into(),
                    config.mode,
                    &mut rng,
                    &mut actions,
                )?;
            }
            ensure(
                actions.len() == 1,
                "evaluation actor must produce exactly one action",
            )?;
            observation.check_action(env.action_space(), &actions[0])?;
            let step = env.step(&actions[0])?;
            ensure(
                !(step.terminated && step.truncated),
                "contradictory evaluation boundary flags",
            )?;
            step.observation.validate_in(
                &observation_space,
                env.action_space(),
                !step.terminated,
            )?;
            let duration = step.reward.duration();
            let (reward, continuation_discount) = step.reward.discounted_f64(config.discount)?;
            total += discount * reward;
            if let Some(reward) = soft_reward {
                total += discount * reward;
            }
            discount *= continuation_discount;
            environment_steps = environment_steps
                .checked_add(1)
                .ok_or_else(|| Error::State("evaluation decision counter overflow".into()))?;
            physical_steps = physical_steps
                .checked_add(duration)
                .ok_or_else(|| Error::State("evaluation physical clock overflow".into()))?;
            let complete = start.horizon.completes(t + 1, step.terminated);
            if step.truncated && !complete {
                ensure(
                    config.allow_truncated_prefix,
                    "truncated prefix is not a complete Monte Carlo return",
                )?;
                any_incomplete = true;
            }
            if let Some(trajectory) = &mut trajectory {
                if typed_input {
                    typed_budget.observation(&step.observation)?;
                    typed_budget.step(&actions[0])?;
                    ensure(
                        trajectory.actions.len() < typed::MAX_TYPED_ROWS,
                        "retained typed evaluation episode exceeds its decision limit",
                    )?;
                }
                trajectory.actions.push(actions[0].clone());
                trajectory.rewards.push(step.reward.total());
                trajectory.physical_durations.push(duration);
                trajectory.discounted_step_rewards.push(reward);
                trajectory.terminated.push(step.terminated);
                trajectory.truncated.push(step.truncated);
                trajectory.observations.push(step.observation.clone());
                if let Some(frame) = capture(&env)? {
                    trajectory.frames.push(frame);
                }
                if let Some(frames) = &mut trajectory.render
                    && let Some((capture, source)) = render
                {
                    frames.capture(&source(&env)?, &capture.limits, &mut render_bytes)?;
                }
            }
            finished = complete || step.truncated;
            observation = step.observation;
            if finished {
                break;
            }
        }
        ensure(
            finished,
            "evaluation collection guard exhausted; incomplete return rejected",
        )?;
        ensure(total.is_finite(), "evaluation return overflow")?;
        returns.push(total);
        if let Some(trajectory) = trajectory {
            trajectories.push(trajectory);
        }
        progress(EvaluationProgress {
            completed: sample + 1,
            total: config.seeds.len(),
            environment_steps,
            physical_steps,
            elapsed_seconds: clock.elapsed().as_secs_f64(),
        });
    }
    Ok(EvaluationResult {
        returns,
        seeds: config.seeds.clone(),
        trajectories,
        metadata: EvaluationMetadata {
            conditioning: start.conditioning,
            initial_action: if start.action.is_some() {
                ActionCondition::Fixed
            } else {
                ActionCondition::CurrentPolicy
            },
            continuation_policy: config.policy_id.clone(),
            stochastic_policy: config.mode == ActMode::Sample,
            discount: config.discount,
            independent_future_streams: true,
            seed_methodology: format!(
                "distinct explicit environment seeds; policy key=first 32 bytes of ChaCha8(base={}, domain=0x524c504f4c494359), scenario stream=environment seed; cross-policy coupling requires matching simulator noise semantics",
                config.policy_seed
            ),
            truncation_handling: match start.horizon {
                EvaluationHorizon::EnvironmentTermination if config.allow_truncated_prefix => {
                    "environment-terminal objective; external truncation retained as incomplete; no bootstrap".into()
                }
                EvaluationHorizon::EnvironmentTermination => {
                    "complete terminal returns; no bootstrap".into()
                }
                EvaluationHorizon::FiniteDecisions(decisions) => format!(
                    "finite objective of {decisions} decisions or earlier environment termination; no bootstrap"
                ),
            },
            complete_returns: !any_incomplete,
        },
        horizon: start.horizon,
        initial_action: start.action.cloned(),
        conditioning_observation: start.observation,
        simulator_state: start.state,
        elapsed_seconds: clock.elapsed().as_secs_f64(),
        environment_steps,
        physical_steps,
    })
}

impl EvaluationResult {
    pub fn empirical_record(
        &self,
        context: RecordContext,
        probe_row: Option<u64>,
    ) -> Result<Chunk> {
        let n = self.returns.len();
        let mut arrays = BTreeMap::from([
            ("returns".into(), Array::f64(vec![n], self.returns.clone())?),
            ("seeds".into(), Array::u64(vec![n], self.seeds.clone())?),
        ]);
        self.horizon.write(&mut arrays)?;
        let initial_action = if let Some(action) = &self.initial_action {
            arrays.insert(
                "initial_action".into(),
                actions_array(std::slice::from_ref(action))?,
            );
            Some("initial_action".into())
        } else {
            None
        };
        let conditioning_observation = if let Some(observation) = &self.conditioning_observation {
            arrays.insert(
                "conditioning_observation".into(),
                Array::f32(
                    vec![1, observation.dense_values().len()],
                    observation.dense_values().to_vec(),
                )?,
            );
            Some("conditioning_observation".into())
        } else {
            None
        };
        let conditioning_mask = if let Some(mask) = self
            .conditioning_observation
            .as_ref()
            .and_then(|o| o.dense_mask())
        {
            arrays.insert(
                "conditioning_mask".into(),
                Array::u8(
                    vec![1, mask.len()],
                    mask.iter().map(|&b| u8::from(b)).collect(),
                )?,
            );
            Some("conditioning_mask".into())
        } else {
            None
        };
        let simulator_state = if let Some(state) = &self.simulator_state {
            arrays.insert(
                "simulator_state".into(),
                Array::f64(vec![state.len()], state.clone())?,
            );
            Some("simulator_state".into())
        } else {
            None
        };
        Ok(Chunk {
            context,
            record: Record::Empirical(EmpiricalRecord {
                returns: "returns".into(),
                seeds: "seeds".into(),
                evaluation: self.metadata.clone(),
                probe_row,
                initial_action,
                conditioning_observation,
                conditioning_mask,
                simulator_state,
            }),
            arrays,
        })
    }
    pub fn trajectory_record(
        &self,
        index: usize,
        context: RecordContext,
        id: &str,
        environment: &str,
    ) -> Result<Chunk> {
        self.trajectory_record_with_limit(index, context, id, environment, typed::MAX_TYPED_BYTES)
    }

    pub(crate) fn trajectory_record_with_limit(
        &self,
        index: usize,
        mut context: RecordContext,
        id: &str,
        environment: &str,
        max_chunk_bytes: usize,
    ) -> Result<Chunk> {
        let trajectory = self
            .trajectories
            .get(index)
            .ok_or_else(|| Error::State("evaluation trajectory index out of range".into()))?;
        let steps = trajectory.actions.len();
        let initial = trajectory
            .observations
            .first()
            .ok_or_else(|| Error::Shape("evaluation trajectory has no observations".into()))?;
        let width = initial.dense_values().len();
        let evaluation = || EvaluationMetadata {
            complete_returns: self
                .horizon
                .completes(steps, trajectory.terminated.last() == Some(&true)),
            ..self.metadata.clone()
        };
        context.trajectory = Some(id.into());
        if width == 0 {
            ensure(
                trajectory.render.is_none() && trajectory.frames.is_empty(),
                "typed episode scene and frame channels are not supported",
            )?;
            ensure(
                (1..=typed::MAX_TYPED_ROWS).contains(&steps)
                    && trajectory.observations.len() == steps + 1
                    && [
                        trajectory.rewards.len(),
                        trajectory.physical_durations.len(),
                        trajectory.discounted_step_rewards.len(),
                        trajectory.terminated.len(),
                        trajectory.truncated.len(),
                    ]
                    .into_iter()
                    .all(|count| count == steps),
                "typed evaluation trajectory arrays have inconsistent lengths",
            )?;
            let mut budget = typed::TypedBudget::new(max_chunk_bytes);
            budget.evaluation(&self.metadata)?;
            for observation in &trajectory.observations {
                budget.observation(observation)?;
            }
            for action in &trajectory.actions {
                budget.step(action)?;
            }
            return TypedPayload::Episode(TypedEpisode {
                source: TypedEpisodeSource::Evaluation {
                    scenario_seed: trajectory.seed,
                    metadata: evaluation(),
                    horizon: self.horizon,
                },
                observations: trajectory.observations.clone(),
                steps: (0..steps)
                    .map(|i| TypedStep {
                        action: trajectory.actions[i].clone(),
                        reward: trajectory.rewards[i],
                        duration: trajectory.physical_durations[i],
                        discounted_reward: trajectory.discounted_step_rewards[i],
                        terminated: trajectory.terminated[i],
                        truncated: trajectory.truncated[i],
                        factual_reward: None,
                        behavior_revision: None,
                    })
                    .collect(),
            })
            .record(context, max_chunk_bytes);
        }
        let mut arrays = BTreeMap::from([
            (
                "observations".into(),
                Array::f32(
                    vec![steps + 1, width],
                    trajectory
                        .observations
                        .iter()
                        .flat_map(|o| o.dense_values().iter().copied())
                        .collect(),
                )?,
            ),
            ("actions".into(), actions_array(&trajectory.actions)?),
            (
                "rewards".into(),
                Array::f32(vec![steps], trajectory.rewards.clone())?,
            ),
            (
                "terminated".into(),
                Array::u8(
                    vec![steps],
                    trajectory.terminated.iter().map(|&x| u8::from(x)).collect(),
                )?,
            ),
            (
                "truncated".into(),
                Array::u8(
                    vec![steps],
                    trajectory.truncated.iter().map(|&x| u8::from(x)).collect(),
                )?,
            ),
        ]);
        self.horizon.write(&mut arrays)?;
        if self.metadata.conditioning == Conditioning::InitialScenario {
            arrays.insert(
                TRAJECTORY_SCENARIO_SEED_KEY.into(),
                Array::u64(vec![1], vec![trajectory.seed])?,
            );
        }
        if trajectory
            .physical_durations
            .iter()
            .any(|&steps| steps != 1)
        {
            arrays.insert(
                PHYSICAL_DURATIONS_KEY.into(),
                Array::u64(vec![steps], trajectory.physical_durations.clone())?,
            );
            arrays.insert(
                DISCOUNTED_REWARDS_KEY.into(),
                Array::f64(vec![steps], trajectory.discounted_step_rewards.clone())?,
            );
        }
        let action_masks = if trajectory
            .observations
            .iter()
            .any(|o| o.dense_mask().is_some())
        {
            let width = trajectory
                .observations
                .iter()
                .find_map(|o| o.dense_mask().map(<[bool]>::len))
                .unwrap();
            let mut masks = Vec::with_capacity((steps + 1) * width);
            for observation in &trajectory.observations {
                if let Some(mask) = &observation.dense_mask() {
                    ensure(mask.len() == width, "trajectory mask width changed")?;
                    masks.extend(mask.iter().map(|&x| u8::from(x)));
                } else {
                    masks.extend(std::iter::repeat_n(1, width));
                }
            }
            arrays.insert(
                "action_masks".into(),
                Array::u8(vec![steps + 1, width], masks)?,
            );
            Some("action_masks".into())
        } else {
            None
        };
        let mut frames = BTreeMap::new();
        let mut simulation_times = None;
        if !trajectory.frames.is_empty() {
            ensure(
                trajectory.frames.len() == steps + 1,
                "environment frames must include every trajectory state",
            )?;
            let first = &trajectory.frames[0];
            for frame in &trajectory.frames {
                ensure(
                    frame.environment == first.environment
                        && frame.channels.keys().eq(first.channels.keys()),
                    "frame channel schema changed within trajectory",
                )?;
            }
            for (name, values) in &first.channels {
                let width = values.len();
                let mut data = Vec::with_capacity((steps + 1) * width);
                for frame in &trajectory.frames {
                    ensure(
                        frame.channels[name].len() == width,
                        "frame channel width changed",
                    )?;
                    data.extend_from_slice(&frame.channels[name]);
                }
                let key = format!("frames/{name}");
                arrays.insert(key.clone(), Array::f64(vec![steps + 1, width], data)?);
                frames.insert(name.clone(), key);
            }
            if first.simulation_time.is_some() {
                let times = trajectory
                    .frames
                    .iter()
                    .map(|frame| {
                        frame
                            .simulation_time
                            .filter(|t| t.is_finite())
                            .ok_or_else(|| {
                                Error::State("missing or nonfinite frame simulation time".into())
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                arrays.insert(
                    "simulation_times".into(),
                    Array::f64(vec![steps + 1], times)?,
                );
                simulation_times = Some("simulation_times".into());
            } else {
                ensure(
                    trajectory
                        .frames
                        .iter()
                        .all(|frame| frame.simulation_time.is_none()),
                    "mixed physical and absent simulation clocks",
                )?;
            }
        }
        Ok(Chunk {
            context,
            record: Record::Trajectory(TrajectoryRecord {
                render: trajectory
                    .render
                    .as_ref()
                    .map(|r| r.arrays(&mut arrays))
                    .transpose()?,
                id: id.into(),
                environment: environment.into(),
                observations: "observations".into(),
                actions: "actions".into(),
                rewards: "rewards".into(),
                terminated: "terminated".into(),
                truncated: "truncated".into(),
                simulation_times,
                action_masks,
                frames,
                evaluation: evaluation(),
            }),
            arrays,
        })
    }
}
