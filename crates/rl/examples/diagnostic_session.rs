use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        evaluation::EvaluationHorizon,
        probes::ProbeSet,
        reference::{
            BenchmarkActor, ConfigValue, LqgReference, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
        },
        schema::ProbePopulation,
        session::{
            DiagnosticConfig, DiagnosticLearner, DiagnosticSession, DiagnosticSessionBuilder,
            DiagnosticTiming,
        },
        value::ValueConventions,
    },
    envs::{
        cartpole::CartPole,
        lqg::{Lqg, LqgConfig},
    },
    objectives::{hl_gauss::HlGaussConfig, risk::Risk},
    prelude::*,
    train::trainer::Progress,
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

#[path = "support/alpha_execution.rs"]
mod alpha_execution;
#[path = "support/planar_rocket.rs"]
mod planar_rocket;
#[path = "support/session_scenarios.rs"]
mod scenarios;
#[path = "support/trade_execution.rs"]
mod trade_execution;

const DEFAULT_SEED: u64 = 9107;
const SCENARIO_SEED: u64 = DEFAULT_SEED;
const GAMMA: f32 = 0.95;
const HORIZON: usize = 16;
const STEPS: u64 = 120;
const CARTPOLE_HORIZON: usize = 120;

struct Run {
    algorithm: String,
    path: PathBuf,
    profile: String,
    reference: bool,
    live: bool,
    wide_lqg: bool,
    render: bool,
    replay_population: bool,
    seed: u64,
    steps: u64,
    evaluation_episodes: usize,
    evaluation_seed: u64,
    training_preset: String,
    rocket_guarded: bool,
    rich_conditional: bool,
}

impl Run {
    fn stream_seed(&self, offset: u64) -> u64 {
        self.seed.wrapping_add(offset)
    }

    fn config(&self, horizon: usize) -> Result<DiagnosticConfig> {
        let mut config = match self.profile.as_str() {
            "off" | "standard" => DiagnosticConfig::standard(),
            "light" => DiagnosticConfig::light(),
            "detailed" => DiagnosticConfig::detailed(),
            _ => {
                return Err(Error::Config(
                    "profile must be off, light, standard or detailed".into(),
                ));
            }
        };
        config.evaluation_horizon = EvaluationHorizon::FiniteDecisions(horizon);
        config.evaluation.max_steps = horizon;
        config.conditional_evaluation.max_steps = horizon;
        Ok(config)
    }

    fn builder(
        &self,
        env: &impl Env,
        config: DiagnosticConfig,
    ) -> DiagnosticSessionBuilder<'static> {
        let builder = DiagnosticSession::builder(&self.path)
            .experiment("RL Visualization Laboratory")
            .environment(env)
            .discount(f64::from(GAMMA))
            .metadata("training.seed", self.seed.to_string())
            .metadata("training.preset", self.training_preset.clone())
            .config(config);
        if self.replay_population {
            builder.metadata("coverage.replay_probe", "replay-retained")
        } else {
            builder
        }
    }

    fn train<L: DiagnosticLearner>(
        &self,
        mut learner: L,
        builder: DiagnosticSessionBuilder<'static>,
        train: impl FnOnce(&mut L, Option<&mut DiagnosticSession<'static, L>>) -> Result<Progress>,
    ) -> Result<()> {
        if let Some(parent) = self.path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let start = Instant::now();
        let mut session = if self.profile == "off" {
            None
        } else {
            Some(builder.build(&learner)?)
        };
        println!(
            "{} {} starting, learner revision {}",
            self.algorithm,
            self.profile,
            learner.diagnostic_revision()
        );
        let progress = train(&mut learner, session.as_mut())?;
        let (timing, bytes) = if let Some(session) = &mut session {
            session.finish(&learner)?;
            let status = session.status();
            (
                session.timing(),
                status
                    .committed_bytes
                    .saturating_sub(status.reclaimed_bytes),
            )
        } else {
            (DiagnosticTiming::default(), 0)
        };
        let total = start.elapsed().as_secs_f64();
        println!(
            "result,{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            self.algorithm,
            self.profile,
            learner.diagnostic_revision(),
            total,
            progress.collection_seconds,
            progress.learning_seconds,
            timing.total_seconds,
            timing.evaluation_seconds,
            timing.inference_seconds,
            timing.conditional_seconds,
            timing.reference_seconds,
            bytes
        );
        if self.evaluation_episodes > 0
            && (matches!(self.algorithm.as_str(), "planar-rocket" | "planar-ppo")
                || self.algorithm.starts_with("alpha-"))
        {
            let seeds: Vec<_> = (0..self.evaluation_episodes)
                .map(|index| self.evaluation_seed.wrapping_add(index as u64))
                .collect();
            let actor = learner.diagnostic_actor()?;
            if matches!(self.algorithm.as_str(), "planar-rocket" | "planar-ppo") {
                planar_rocket::evaluate_learner(actor.as_ref(), &seeds, self.rocket_guarded)?;
            } else {
                alpha_execution::evaluate_learner(actor.as_ref(), &seeds, self.reference)?;
            }
        }
        Ok(())
    }

    fn train_on_policy<
        E: Env + Clone,
        L: DiagnosticLearner + OnPolicyLearner<Batch = OnPolicyBatch>,
    >(
        &self,
        learner: L,
        builder: DiagnosticSessionBuilder<'static>,
        environment: E,
        config: OnPolicyConfig,
        steps: u64,
    ) -> Result<()> {
        let mut trainer = OnPolicyTrainer::new(
            OnPolicyCollector::new(
                SyncVectorEnv::new(vec![environment; 2])?,
                self.stream_seed(3),
            )?,
            config,
        )?;
        let budget = Budget::vector_steps(steps);
        let mut next_report = 5.;
        let mut progress = |p: &Progress| {
            self.progress(p);
            if !self.live && p.elapsed_seconds >= next_report {
                next_report = p.elapsed_seconds + 5.;
                let remaining = steps.saturating_sub(p.counters.vector_steps);
                let eta =
                    p.elapsed_seconds * remaining as f64 / p.counters.vector_steps.max(1) as f64;
                eprintln!(
                    "training: {}/{} vector steps, {} updates, {:.1}s elapsed, {:.1}s estimated remaining; collection {:.1}s, learning {:.1}s",
                    p.counters.vector_steps,
                    steps,
                    p.counters.updates,
                    p.elapsed_seconds,
                    eta,
                    p.collection_seconds,
                    p.learning_seconds
                );
            }
        };
        self.train(learner, builder, |learner, diagnostics| match diagnostics {
            Some(session) => trainer.train_observed(learner, budget, Some(&mut progress), session),
            None => trainer.train(learner, budget, Some(&mut progress)),
        })
    }

    fn train_off_policy<
        E: Env + Clone,
        L: DiagnosticLearner + OffPolicyLearner<Batch = ReplayBatch>,
    >(
        &self,
        learner: L,
        builder: DiagnosticSessionBuilder<'static>,
        environment: E,
        config: OffPolicyConfig,
    ) -> Result<()> {
        let mut trainer = OffPolicyTrainer::new(
            OffPolicyCollector::new(
                SyncVectorEnv::new(vec![environment; 2])?,
                self.stream_seed(3),
            )?,
            config,
            self.stream_seed(4),
        )?;
        let budget = Budget::vector_steps(self.steps);
        let mut next_report = 5.;
        let mut progress = |p: &Progress| {
            self.progress(p);
            if !self.live && p.elapsed_seconds >= next_report {
                next_report = p.elapsed_seconds + 5.;
                let remaining = self.steps.saturating_sub(p.counters.vector_steps);
                let eta =
                    p.elapsed_seconds * remaining as f64 / p.counters.vector_steps.max(1) as f64;
                eprintln!(
                    "training: {}/{} vector steps, {} updates, {:.1}s elapsed, {:.1}s estimated remaining; collection {:.1}s, learning {:.1}s",
                    p.counters.vector_steps,
                    self.steps,
                    p.counters.updates,
                    p.elapsed_seconds,
                    eta,
                    p.collection_seconds,
                    p.learning_seconds
                );
            }
        };
        self.train(learner, builder, |learner, diagnostics| match diagnostics {
            Some(session) => {
                let progress =
                    trainer.train_observed(learner, budget, Some(&mut progress), session)?;
                let replay = trainer.replay();
                if self.replay_population && !replay.is_empty() {
                    let mut rng = ChaCha8Rng::seed_from_u64(self.stream_seed(9));
                    let sample = replay.sample(replay.len().min(128), &mut rng)?;
                    let (observations, actions) = sample
                        .items
                        .into_iter()
                        .map(|item| (item.transition.observation, item.transition.action))
                        .unzip();
                    let probes = ProbeSet::new(
                        "replay-retained",
                        ProbePopulation::ReplayRetained,
                        "Uniform sample with replacement from the final factual replay buffer",
                        observations,
                        Some(actions),
                        replay.action_space().clone(),
                    )?;
                    session.record_history_probe(learner, progress.counters, &probes)?;
                }
                Ok(progress)
            }
            None => trainer.train(learner, budget, Some(&mut progress)),
        })
    }
    fn progress(&self, progress: &Progress) {
        if self.live {
            println!(
                "training revisions={} transitions={}",
                progress.counters.updates, progress.counters.env_steps
            );
            std::thread::sleep(Duration::from_millis(180));
        }
    }
}

fn lqg(run: &Run) -> Result<()> {
    let episodic = run.algorithm == "reinforce";
    let horizon = if episodic {
        HORIZON.min(run.steps as usize)
    } else {
        HORIZON
    };
    let config = if run.wide_lqg {
        scenarios::wide_lqg_config(horizon)?
    } else {
        let mut config = LqgConfig::scalar(horizon, 0.82, 0.65, 0.8, 0.3, 1., 1., 0.12, 2.)?;
        config.action_space = ActionSpace::Unbounded { dimensions: 1 };
        config
    };
    let env = Lqg::new(config.clone(), run.seed)?;
    let action_space = env.action_space().clone();
    let observations_width = env.observation_space().require_dense_width()?;
    let action_width = action_space.dimensions();
    let diagnostic_config = run.config(horizon)?;
    let detailed = diagnostic_config.schedule.conditional_evaluation.is_some();

    let mut branch = Lqg::new(config.clone(), SCENARIO_SEED + 10)?;
    let observation = branch.reset(Some(SCENARIO_SEED + 11))?.observation;
    let mut observations = vec![observation.clone()];
    for (state_scale, time) in [(-1., 0.), (0.25, 0.25), (0.5, 0.5), (1.5, 0.75)] {
        let mut values = config
            .initial
            .iter()
            .map(|state| state_scale * *state as f32)
            .collect::<Vec<_>>();
        values.push(time);
        observations.push(Observation::new(values));
    }
    let actions = vec![Action::Continuous(vec![0.15; action_width]); observations.len()];
    let probes = ProbeSet::new(
        "lqg-fixed",
        ProbePopulation::FixedReference,
        "Fixed state and decision-time rows, equally weighted",
        observations,
        Some(actions.clone()),
        action_space.clone(),
    )?;
    let factory_config = config.clone();
    let mut builder = run
        .builder(&env, diagnostic_config)
        .metadata(
            "evaluation.scenario_protocol",
            "Configured fixed initial state with independent ChaCha8 process noise per scenario seed",
        )
        .probe_set(probes)
        .evaluation_factory_with_frames(move |seed| Lqg::new(factory_config.clone(), seed));
    if run.wide_lqg {
        let context = scenarios::wide_lqg_context(&config);
        builder = builder.reference_context(context.clone());
        if run.reference {
            builder = builder.reference(scenarios::zero_control_benchmark(context)?);
        }
    } else {
        let reference = LqgReference::new(&config, f64::from(GAMMA))?;
        builder = builder.reference_context(reference.specification().context.clone());
        if run.reference {
            builder = builder.reference(reference);
        }
    }

    let state_value = matches!(
        run.algorithm.as_str(),
        "ppo" | "reinforce" | "vmpo" | "mdpo"
    );
    if detailed {
        let first_action = (!state_value).then(|| actions[0].clone());
        builder = builder.conditional_observation_case(
            "lqg-fixed", 0, observation, first_action,
            "Every reset begins at the configured deterministic state and each future process-noise stream is seeded independently",
            move |seed| {
                let mut environment = Lqg::new(config.clone(), seed)?;
                let observation = environment.reset(Some(seed))?.observation;
                Ok((environment, observation))
            },
        );
    }

    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    if state_value {
        let models = ActorCritic {
            policy: Policy::new(observations_width, action_space, &[12, 12], &mut rng)?,
            value: Value::new(observations_width, &[12, 12], &mut rng)?,
        };
        let mut training = OnPolicyConfig {
            rollout_steps: 8,
            gamma: GAMMA,
            ..Default::default()
        };
        return match run.algorithm.as_str() {
            "ppo" => run.train_on_policy(
                Ppo::new(
                    models,
                    PpoConfig {
                        epochs: 2,
                        minibatch_size: 8,
                        ..Default::default()
                    },
                    run.stream_seed(2),
                )?,
                builder,
                env,
                training,
                run.steps,
            ),
            "vmpo" => run.train_on_policy(
                Vmpo::new(
                    models,
                    VmpoConfig {
                        epochs: 2,
                        minibatch_size: 8,
                        ..Default::default()
                    },
                    run.stream_seed(2),
                )?,
                builder,
                env,
                training,
                run.steps,
            ),
            "mdpo" => run.train_on_policy(
                Mdpo::new(
                    models,
                    MdpoConfig {
                        inner_steps: 2,
                        value_epochs: 2,
                        value_minibatch_size: 8,
                        ..Default::default()
                    },
                    run.stream_seed(2),
                )?,
                builder,
                env,
                training,
                run.steps,
            ),
            "reinforce" => {
                training.episode_guard = Some(horizon);
                training.gae_lambda = None;
                let steps = run.steps / horizon as u64 * horizon as u64;
                if run.steps != steps {
                    println!(
                        "REINFORCE vector steps {steps} within the requested {}-step budget",
                        run.steps
                    );
                }
                run.train_on_policy(
                    Reinforce::new(models, Default::default(), run.stream_seed(2))?,
                    builder,
                    env,
                    training,
                    steps,
                )
            }
            _ => unreachable!(),
        };
    }

    let policy = Deterministic::new(observations_width, action_space, &[12, 12], &mut rng)?;
    let mut critic =
        |heads| QValue::new(observations_width, action_width, heads, &[12, 12], &mut rng);
    let training = off_policy_config();
    match run.algorithm.as_str() {
        "ddpg" => run.train_off_policy(
            Ddpg::new(policy, critic(1)?, Default::default(), run.stream_seed(2))?,
            builder,
            env,
            training,
        ),
        "td3" => run.train_off_policy(
            Td3::new(
                policy,
                critic(1)?,
                critic(1)?,
                Default::default(),
                run.stream_seed(2),
            )?,
            builder,
            env,
            training,
        ),
        "d4pg" => run.train_off_policy(
            D4pg::new(policy, critic(9)?, D4pgConfig::new(9)?, run.stream_seed(2))?,
            builder,
            env,
            training,
        ),
        "md4pg" => run.train_off_policy(
            Md4pg::new(
                policy,
                critic(1)?,
                critic(1)?,
                Md4pgConfig {
                    risk: Risk::MeanVariance { penalty: 0.05 },
                    ..Default::default()
                },
                run.stream_seed(2),
            )?,
            builder,
            env,
            training,
        ),
        _ => Err(Error::Config("unknown continuous demonstration".into())),
    }
}
fn off_policy_config() -> OffPolicyConfig {
    OffPolicyConfig {
        replay_capacity: 512,
        batch_size: 16,
        warmup_transitions: 16,
        gamma: GAMMA,
        updates_per_transition: 0.5,
        ..Default::default()
    }
}

struct CartPoleBenchmark(ActionSpace);

impl Actor for CartPoleBenchmark {
    fn action_space(&self) -> &ActionSpace {
        &self.0
    }
    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            let state = observation.dense_values();
            actions.push(Action::Discrete(usize::from(
                state[2] + 0.25 * state[3] >= 0.,
            )));
        }
        Ok(())
    }
}

fn cartpole(run: &Run) -> Result<()> {
    let mut env = CartPole::new(SCENARIO_SEED);
    let action_space = env.action_space().clone();
    let mut observations = Vec::new();
    for seed in SCENARIO_SEED..SCENARIO_SEED + 8 {
        observations.push(env.reset(Some(seed))?.observation);
    }
    let probe_set = ProbeSet::new(
        "cartpole-resets",
        ProbePopulation::FixedReference,
        "Eight separately seeded reset observations, equally weighted",
        observations,
        None,
        action_space.clone(),
    )?;
    let context = ReferenceContext {
        environment: "cartpole".into(),
        observation_dim: 4,
        typed_observation_space: None,
        action_space: action_space.clone(),
        conventions: ValueConventions {
            dynamics: "rust-rl CartPole Euler dynamics".into(),
            observation: "Cart position, velocity, pole angle, angular velocity".into(),
            action: "Discrete left or right force".into(),
            reward_units: "reward units".into(),
            reward_convention: "One unit per executed decision".into(),
            discount: f64::from(GAMMA),
            horizon: "Up to 120 decisions".into(),
            terminal_treatment: "True failure or the explicit evaluation horizon".into(),
            action_constraints: "Both force directions legal".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite {
            decisions: CARTPOLE_HORIZON as u32,
        },
        environment_config: BTreeMap::from([
            (
                "horizon".into(),
                ConfigValue::Integer(CARTPOLE_HORIZON as i64),
            ),
            ("decision_seconds".into(), ConfigValue::Real(0.02)),
        ]),
        markov_observation: false,
        decision_time_coordinate: None,
    };
    let mut builder = run
        .builder(&env, run.config(CARTPOLE_HORIZON)?)
        .environment_descriptor(scenarios::cartpole_descriptor())
        .reference_context(context.clone())
        .metadata(
            "evaluation.scenario_protocol",
            "Gymnasium CartPole independent uniform reset coordinates with a ChaCha8 scenario seed",
        )
        .metadata("evaluation.decision_seconds", "0.02")
        .probe_set(probe_set);
    let factory = |seed| Ok(CartPole::new(seed));
    builder = if run.render {
        builder
            .evaluation_factory_with_render(factory, scenarios::render_capture(CARTPOLE_HORIZON))
            .reference_trajectories(1)
    } else {
        builder.evaluation_factory(factory)
    };
    if run.reference {
        builder = builder.reference(
            BenchmarkActor::new(
                ReferenceSpec {
                    id: "pole-angle-controller".into(),
                    context,
                    quality: ReferenceQuality {
                        kind: ReferenceKind::Heuristic,
                        method: "Sign of observed pole angle plus one quarter of angular velocity"
                            .into(),
                        assumptions: Vec::new(),
                    },
                    target: ReferenceTarget::BenchmarkPolicy,
                    lifetime: ReferenceLifetime::Fixed,
                    information: ReferenceInformation::LearnerObservation,
                    uncertainty: None,
                },
                CartPoleBenchmark(action_space.clone()),
            )?
            .with_evaluation_mode(ActMode::Deterministic),
        );
    }
    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    if run.algorithm == "reppo" {
        let config = ReppoConfig {
            epochs: 2,
            minibatch_size: 8,
            auxiliary_coefficient: 0.05,
            hl_gauss: HlGaussConfig {
                min: -2.,
                max: 24.,
                bins: 33,
                ..Default::default()
            },
            ..Default::default()
        };
        let learner = Reppo::new(
            Policy::new(4, action_space.clone(), &[12, 12], &mut rng)?,
            CategoricalQ::new(4, action_space, config.hl_gauss.bins, &[12, 12], &mut rng)?,
            config,
            run.stream_seed(2),
        )?;
        return run.train_on_policy(
            learner,
            builder,
            CartPole::new(run.seed),
            OnPolicyConfig {
                rollout_steps: 8,
                gamma: GAMMA,
                ..Default::default()
            },
            run.steps,
        );
    }
    run.train_off_policy(
        Ddqn::new(
            DiscreteQ::new(4, 2, &[12, 12], &mut rng)?,
            DdqnConfig::default(),
        )?,
        builder,
        CartPole::new(run.seed),
        off_policy_config(),
    )
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let algorithm = args.next().ok_or_else(|| Error::Config(
        "usage: diagnostic_session <ppo|reinforce|vmpo|mdpo|reppo|ddqn|ddpg|td3|sac|d4pg|md4pg|trade-d4pg|trade-md4pg|alpha-ddpg|alpha-d4pg|alpha-cvar|alpha-md4pg|planar-rocket|planar-ppo|rocket-benchmark|rocket-history|rocket-history-report|alpha-benchmark|alpha-reference|alpha-reference-check> <recording> [light|standard|detailed|off] [--source=RECORDING] [--reference] [--live] [--steps=N] [--seed=N] [--evaluation-episodes=N] [--evaluation-seed=N] [--preset=baseline|balanced|alpha-quantiles|alpha-wide|alpha-tau] [--rich-conditional] [--guarded-rocket-reward] [--wide-lqg] [--render] [--replay-population]".into(),
    ))?;
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::Config("recording path is required".into()))?,
    );
    let mut run = Run {
        algorithm,
        path,
        profile: "standard".into(),
        reference: false,
        live: false,
        wide_lqg: false,
        render: false,
        replay_population: false,
        seed: DEFAULT_SEED,
        steps: STEPS,
        evaluation_episodes: 0,
        evaluation_seed: 200_000,
        training_preset: "baseline".into(),
        rocket_guarded: false,
        rich_conditional: false,
    };
    let mut source = None;
    for arg in args {
        match arg.as_str() {
            "--reference" => run.reference = true,
            "--live" => run.live = true,
            "--wide-lqg" => run.wide_lqg = true,
            "--render" => run.render = true,
            "--replay-population" => run.replay_population = true,
            "--rich-conditional" => run.rich_conditional = true,
            _ if arg.starts_with("--source=") => {
                let path = &arg["--source=".len()..];
                if path.is_empty() || source.replace(PathBuf::from(path)).is_some() {
                    return Err(Error::Config("one nonempty source path is required".into()));
                }
            }
            _ if arg.starts_with("--seed=") => {
                run.seed = arg["--seed=".len()..]
                    .parse::<u64>()
                    .map_err(|_| Error::Config("seed must be an unsigned 64-bit integer".into()))?;
            }
            _ if arg.starts_with("--evaluation-episodes=") => {
                run.evaluation_episodes = arg["--evaluation-episodes=".len()..]
                    .parse::<usize>()
                    .ok()
                    .filter(|n| (1..=10_000).contains(n))
                    .ok_or_else(|| {
                        Error::Config("evaluation episodes must be between 1 and 10000".into())
                    })?;
            }
            _ if arg.starts_with("--evaluation-seed=") => {
                run.evaluation_seed =
                    arg["--evaluation-seed=".len()..]
                        .parse::<u64>()
                        .map_err(|_| {
                            Error::Config(
                                "evaluation seed must be an unsigned 64-bit integer".into(),
                            )
                        })?;
            }
            _ if arg.starts_with("--preset=") => {
                let preset = &arg["--preset=".len()..];
                if !matches!(
                    preset,
                    "baseline" | "balanced" | "alpha-quantiles" | "alpha-wide" | "alpha-tau"
                ) {
                    return Err(Error::Config("unknown training preset".into()));
                }
                run.training_preset = preset.into();
            }
            "--guarded-rocket-reward" => run.rocket_guarded = true,
            "light" | "standard" | "detailed" | "off" => run.profile = arg,
            _ if arg.starts_with("--steps=") => {
                run.steps = arg["--steps=".len()..]
                    .parse::<u64>()
                    .ok()
                    .filter(|steps| (1..=1_000_000).contains(steps))
                    .ok_or_else(|| {
                        Error::Config("step budget must be between 1 and 1000000".into())
                    })?;
            }
            _ => return Err(Error::Config("unknown diagnostic session option".into())),
        }
    }
    if !matches!(
        run.algorithm.as_str(),
        "rocket-benchmark"
            | "rocket-history"
            | "rocket-history-report"
            | "alpha-benchmark"
            | "alpha-reference"
            | "alpha-reference-check"
    ) {
        println!(
            "format,algorithm,profile,revision,total_s,collection_s,learning_s,diagnostics_s,evaluation_s,inference_s,conditional_s,reference_s,storage_bytes"
        );
    }
    match run.algorithm.as_str() {
        _ if source.is_some() && run.algorithm != "rocket-history" => Err(Error::Config(
            "--source applies only to rocket-history".into(),
        )),
        _ if run.training_preset.starts_with("alpha-") && run.algorithm != "alpha-d4pg" => Err(
            Error::Config("alpha critic study presets apply only to alpha-d4pg".into()),
        ),
        _ if run.rich_conditional
            && !matches!(
                run.algorithm.as_str(),
                "alpha-ddpg" | "alpha-d4pg" | "alpha-cvar" | "alpha-md4pg"
            ) =>
        {
            Err(Error::Config(
                "--rich-conditional applies to alpha execution learners".into(),
            ))
        }
        "ppo" | "reinforce" | "vmpo" | "mdpo" | "reppo" | "planar-ppo" if run.replay_population => {
            Err(Error::Config(
                "--replay-population requires an off-policy learner".into(),
            ))
        }
        "ddqn" | "reppo" | "sac" | "trade-d4pg" | "trade-md4pg" | "alpha-ddpg" | "alpha-d4pg"
        | "alpha-cvar" | "alpha-md4pg" | "planar-rocket" | "planar-ppo"
            if run.wide_lqg =>
        {
            Err(Error::Config(
                "--wide-lqg applies to the LQG demonstrations".into(),
            ))
        }
        _ if run.render
            && !matches!(
                run.algorithm.as_str(),
                "ddqn" | "reppo" | "sac" | "planar-rocket" | "planar-ppo"
            ) =>
        {
            Err(Error::Config(
                "--render applies to the CartPole, Pendulum and planar rocket demonstrations"
                    .into(),
            ))
        }
        "ddqn" | "reppo" => cartpole(&run),
        "sac" => scenarios::pendulum(&run),
        "rocket-benchmark" => planar_rocket::benchmark(&run),
        "rocket-history" => planar_rocket::historical_evaluation(&run, source.as_deref()),
        "rocket-history-report" => planar_rocket::historical_report(&run),
        "alpha-benchmark" => alpha_execution::benchmark(&run),
        "alpha-reference" => alpha_execution::reference_report(),
        "alpha-reference-check" => alpha_execution::reference_check(&run),
        "alpha-ddpg" | "alpha-d4pg" | "alpha-cvar" | "alpha-md4pg" => alpha_execution::run(&run),
        "trade-d4pg" | "trade-md4pg" => trade_execution::run(&run),
        "planar-rocket" | "planar-ppo" => planar_rocket::run(&run),
        "ppo" | "reinforce" | "vmpo" | "mdpo" | "ddpg" | "td3" | "d4pg" | "md4pg" => lqg(&run),
        _ => Err(Error::Config("unknown diagnostic learner".into())),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rl::diagnostics::{
        evaluation::{evaluation_horizon, trajectory_scenario_seed},
        models::{InspectionRequest, ModelBundle, ModelLimits},
        render::{RenderLimits, scene_at},
        schema::{Record, RecordKind},
        storage::{ReadLimits, Recording},
    };

    #[test]
    fn wide_lqg_has_two_bounded_controls_and_seventeen_observation_coordinates() -> Result<()> {
        let mut environment = Lqg::new(scenarios::wide_lqg_config(4)?, DEFAULT_SEED)?;
        assert_eq!(environment.observation_space(), ObservationSpace::Dense(17));
        assert_eq!(
            environment.action_space(),
            &ActionSpace::Box {
                low: vec![-1.5; 2],
                high: vec![1.5; 2]
            }
        );
        let initial = environment.reset(Some(DEFAULT_SEED))?;
        let successor = environment.step(&Action::Continuous(vec![0.25, -0.75]))?;
        assert_eq!(initial.observation.dense_values().len(), 17);
        assert_eq!(successor.observation.dense_values()[16], 0.25);
        assert!(!successor.terminated);
        Ok(())
    }

    #[test]
    fn changing_the_training_seed_changes_the_learned_policy_with_matched_scenarios() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let mut policies = Vec::new();
        let mut scenarios = Vec::new();
        for (index, seed) in [DEFAULT_SEED, DEFAULT_SEED, DEFAULT_SEED + 19]
            .into_iter()
            .enumerate()
        {
            let run = Run {
                algorithm: "ppo".into(),
                path: directory.path().join(format!("run-{index}")),
                profile: "standard".into(),
                reference: false,
                live: false,
                wide_lqg: false,
                render: false,
                replay_population: false,
                seed,
                steps: 16,
                evaluation_episodes: 0,
                evaluation_seed: 200_000,
                training_preset: "baseline".into(),
                rocket_guarded: false,
                rich_conditional: false,
            };
            lqg(&run)?;
            let recording = Recording::open(&run.path, ReadLimits::default())?;
            assert_eq!(
                recording.manifest().metadata["training.seed"],
                seed.to_string()
            );
            let entries = recording.entries();
            let snapshot = entries
                .iter()
                .rposition(|entry| entry.kind == RecordKind::ModelSnapshot)
                .expect("final learner snapshot");
            let bundle = ModelBundle::load(&recording.load(snapshot)?, ModelLimits::default())?;
            let request = InspectionRequest::new("policy", vec![Observation::new(vec![1., 0.])]);
            policies.push(bundle.inspect(&request)?.actions);
            let evaluation = entries
                .iter()
                .rposition(|entry| entry.kind == RecordKind::Empirical)
                .expect("final numerical evaluation");
            scenarios.push(
                recording.load(evaluation)?.arrays["seeds"]
                    .as_u64()?
                    .to_vec(),
            );
        }
        assert_eq!(policies[0], policies[1]);
        assert_ne!(policies[0], policies[2]);
        assert_eq!(scenarios[0], scenarios[1]);
        assert_eq!(scenarios[0], scenarios[2]);
        Ok(())
    }

    #[test]
    fn renders_one_fixed_benchmark_track_with_the_same_initial_scenario() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let run = Run {
            algorithm: "ddqn".into(),
            path: directory.path().join("cartpole"),
            profile: "standard".into(),
            reference: true,
            live: false,
            wide_lqg: false,
            render: true,
            replay_population: true,
            seed: DEFAULT_SEED,
            steps: 16,
            evaluation_episodes: 0,
            evaluation_seed: 200_000,
            training_preset: "baseline".into(),
            rocket_guarded: false,
            rich_conditional: false,
        };
        cartpole(&run)?;

        let recording = Recording::open(&run.path, ReadLimits::default())?;
        let mut online = Vec::new();
        let mut benchmark = Vec::new();
        for (index, entry) in recording.entries().iter().enumerate() {
            if entry.kind != RecordKind::Trajectory {
                continue;
            }
            let chunk = recording.load(index)?;
            let Record::Trajectory(trajectory) = &chunk.record else {
                unreachable!()
            };
            assert!(trajectory.render.is_some());
            let scenario = trajectory_scenario_seed(&chunk)?.expect("portable scenario seed");
            let horizon = evaluation_horizon(&chunk)?;
            let initial = scene_at(&chunk, 0, &RenderLimits::default())?;
            assert!(initial.is_some());
            let rows = if entry.model_role == "online" {
                &mut online
            } else {
                assert_eq!(entry.model_role, "reference:pole-angle-controller");
                &mut benchmark
            };
            rows.push((scenario, horizon, initial));
        }
        assert_eq!(online.len(), 2);
        assert_eq!(benchmark.len(), 1);
        assert_eq!(online[0], benchmark[0]);
        let index = recording
            .entries()
            .iter()
            .position(|entry| {
                entry.kind == RecordKind::ProbeSet
                    && entry.probe_set.as_deref() == Some("replay-retained")
            })
            .expect("factual replay rows");
        let replay = recording.load(index)?;
        let Record::ProbeSet(probes) = &replay.record else {
            unreachable!()
        };
        assert_eq!(probes.population, ProbePopulation::ReplayRetained);
        assert!(replay.context.counters.env_steps > 0);
        assert_eq!(replay.arrays[&probes.observations].shape()[1], 4);
        assert_eq!(
            replay.arrays[probes.fixed_actions.as_ref().unwrap()].shape()[0],
            replay.arrays[&probes.observations].shape()[0]
        );
        Ok(())
    }
}
