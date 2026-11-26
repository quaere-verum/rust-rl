//! Versioned alpha-driven execution demonstration.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{Error, Result};
use rl::{
    diagnostics::{
        evaluation::{BranchingEnv, EnvironmentFrame, FrameSource},
        probes::ProbeSet,
        schema::ProbePopulation,
        visualization::*,
    },
    objectives::risk::Risk,
    prelude::*,
};
use std::collections::BTreeMap;

#[path = "alpha_execution_reference.rs"]
mod reference;

pub(super) fn reference_report() -> Result<()> {
    reference::convergence_report()
}
#[derive(Clone, Copy, Debug)]
struct Config {
    horizon: usize,
    dt: f64,
    eta: f64,
    kappa: f64,
    sigma_alpha: f64,
    sigma_price: f64,
    spread: f64,
    impact: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            horizon: 12,
            dt: 1.,
            eta: 0.4,
            kappa: 0.25,
            sigma_alpha: 0.18,
            sigma_price: 0.35,
            spread: 0.015,
            impact: 0.1,
        }
    }
}

impl Config {
    fn validate(&self) -> Result<()> {
        if self.horizon == 0
            || !self.dt.is_finite()
            || self.dt <= 0.
            || [
                self.eta,
                self.kappa,
                self.sigma_alpha,
                self.sigma_price,
                self.spread,
                self.impact,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(Error::Config("invalid alpha-execution parameters".into()));
        }
        Ok(())
    }

    fn cost(&self, trade: f64) -> f64 {
        self.spread * trade.abs() + self.impact / self.dt * trade * trade
    }

    fn alpha_innovation_scale(&self) -> f64 {
        let variance = if self.kappa == 0. {
            self.dt
        } else {
            -(-2. * self.kappa * self.dt).exp_m1() / (2. * self.kappa)
        };
        self.sigma_alpha * variance.sqrt()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct State {
    inventory: f64,
    alpha: f64,
    period: usize,
    price: f64,
    cumulative_pnl: f64,
}

fn conditional_cases(rich: bool) -> Vec<(State, f32)> {
    if !rich {
        return [-0.3, 0., 0.3]
            .into_iter()
            .zip([-0.5, 0., 0.5])
            .map(|(alpha, action)| {
                (
                    State {
                        alpha,
                        ..State::default()
                    },
                    action,
                )
            })
            .collect();
    }
    [(0, 0., 0.3), (6, 0.5, 0.), (11, -0.5, 0.3)]
        .into_iter()
        .flat_map(|(period, inventory, alpha)| {
            [-0.5, 0., 0.5].into_iter().map(move |action| {
                (
                    State {
                        inventory,
                        alpha,
                        period,
                        ..State::default()
                    },
                    action,
                )
            })
        })
        .collect()
}

fn critic_study_parameters(preset: &str) -> (usize, usize, f32) {
    match preset {
        "alpha-quantiles" => (33, 32, 0.005),
        "alpha-wide" => (17, 64, 0.005),
        "alpha-tau" => (17, 32, 0.02),
        _ => (17, 32, 0.005),
    }
}

#[derive(Clone, Copy, Debug)]
struct Transition {
    next: State,
    trade: f64,
    post_trade_inventory: f64,
    post_trade_alpha: f64,
    execution_cost: f64,
    liquidation_cost: f64,
    price_change: f64,
    reward: f64,
}

fn transition(
    config: &Config,
    state: State,
    target: f64,
    alpha_noise: f64,
    price_noise: f64,
) -> Result<Transition> {
    config.validate()?;
    if state.period >= config.horizon
        || !(-1. ..=1.).contains(&state.inventory)
        || !(-1. ..=1.).contains(&target)
        || [
            state.alpha,
            state.price,
            state.cumulative_pnl,
            alpha_noise,
            price_noise,
        ]
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err(Error::State(
            "invalid or terminal alpha-execution state, action or innovation".into(),
        ));
    }
    let trade = target - state.inventory;
    let post_trade_alpha = state.alpha * (-config.eta * trade.abs()).exp();
    let price_change =
        post_trade_alpha * config.dt + config.sigma_price * config.dt.sqrt() * price_noise;
    let execution_cost = config.cost(trade);
    let terminal = state.period + 1 == config.horizon;
    let liquidation_cost = if terminal { config.cost(-target) } else { 0. };
    let reward = target * price_change - execution_cost - liquidation_cost;
    let next = State {
        inventory: if terminal { 0. } else { target },
        alpha: (-config.kappa * config.dt).exp() * post_trade_alpha
            + config.alpha_innovation_scale() * alpha_noise,
        period: state.period + 1,
        price: state.price + price_change,
        cumulative_pnl: state.cumulative_pnl + reward,
    };
    if [next.alpha, next.price, next.cumulative_pnl, reward]
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err(Error::State("nonfinite alpha-execution transition".into()));
    }
    Ok(Transition {
        next,
        trade,
        post_trade_inventory: target,
        post_trade_alpha,
        execution_cost,
        liquidation_cost,
        price_change,
        reward,
    })
}

#[derive(Clone)]
struct AlphaExecution {
    config: Config,
    state: State,
    space: ActionSpace,
    rng: ChaCha8Rng,
    last: Option<Transition>,
}

impl AlphaExecution {
    fn new(config: Config, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            state: State {
                period: config.horizon,
                ..State::default()
            },
            space: ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            rng: ChaCha8Rng::seed_from_u64(seed),
            last: None,
        })
    }

    fn observation(&self) -> Observation {
        Observation::new(vec![
            self.state.inventory as f32,
            self.state.alpha as f32,
            (self.config.horizon - self.state.period) as f32 / self.config.horizon as f32,
        ])
    }
}

fn descriptor() -> EnvironmentDescriptor {
    EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: [
            ("inventory", "Inventory", Some([-1., 1.])),
            ("alpha", "Available alpha", None),
            ("remaining_fraction", "Remaining horizon fraction", Some([0., 1.])),
        ].into_iter().map(|(id, label, range)| ObservationField {
            id: id.into(), label: label.into(), unit: None, description: None,
            kind: ObservationType::Numerical, range, coordinate: true,
        }).collect(),
        actions: ActionDescriptor::Continuous { components: vec![ActionComponent {
            label: "Target inventory".into(), unit: None,
            meaning: Some("Executed signed trade equals target inventory minus previous inventory".into()),
        }], preferred_component: Some(0) },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel { label: "Learner revision".into(), unit: None },
            decision_step: QuantityLabel { label: "Decision".into(), unit: None },
            simulation: Some(QuantityLabel { label: "Execution time".into(), unit: Some("periods".into()) }),
        },
        reward: RewardDescriptor {
            label: "Net P&L".into(), unit: Some("normalized P&L".into()),
            description: Some("Post-trade inventory earns the next price move, less execution and final liquidation costs".into()), components: Vec::new(),
        },
        frames: [
            ("price", "Midprice"), ("alpha", "Available alpha"), ("post_trade_alpha", "Post-trade alpha (previous decision)"),
            ("inventory", "Inventory"), ("trade", "Trade (previous decision)"), ("terminal_liquidation", "Terminal liquidation"),
            ("incremental_pnl", "P&L (previous decision)"), ("cumulative_pnl", "Cumulative P&L"),
            ("execution_cost", "Costs (previous decision)"), ("price_change", "Price change (previous decision)"),
        ].into_iter().map(|(name, label)| FrameChannel {
            name: name.into(), label: label.into(), width: 1, unit: None,
            visibility: FrameVisibility::DiagnosticOnly, representation: Some(FrameRepresentation::Scalar),
        }).collect(),
        preferences: VisualizationPreferences {
            coordinates: vec!["inventory".into(), "alpha".into()],
            trajectories: ["alpha", "post_trade_alpha", "inventory", "trade", "cumulative_pnl", "execution_cost"].into_iter().map(|channel| FrameReference { channel: channel.into(), index: 0 }).collect(),
            ..Default::default()
        },
    }
}

pub(super) fn run(run: &super::Run) -> Result<()> {
    if run.rich_conditional && run.profile != "detailed" {
        return Err(Error::Config(
            "rich conditional evidence requires the detailed profile".into(),
        ));
    }
    let config = Config::default();
    let environment = AlphaExecution::new(config, run.seed)?;
    let space = environment.action_space().clone();
    let reference_context = reference::context(config, space.clone());
    let mut diagnostic = run.config(config.horizon)?;
    diagnostic.evaluation.mode = ActMode::Deterministic;
    diagnostic.conditional_evaluation.mode = ActMode::Deterministic;
    diagnostic.conditional_evaluation.seeds = (0..64).map(|index| 0xa115_0000 + index).collect();
    if run.rich_conditional {
        diagnostic.conditional_evaluation.seeds =
            (0..512).map(|index| 0xa115_0000 + index).collect();
        diagnostic.max_conditional_cases = 9;
    }
    if run.profile == "detailed" {
        // Keep the entire learning history, including matching conditional futures,
        // within the portable recorder's 64-snapshot-per-role limit.
        let cadence = run.steps.div_ceil(48).max(1);
        let interval = Some(rl::diagnostics::schedule::Interval {
            first: cadence,
            every: cadence,
        });
        diagnostic.schedule.metrics = interval;
        diagnostic.schedule.probes = interval;
        diagnostic.schedule.representations = interval;
        diagnostic.schedule.evaluation = interval;
        diagnostic.schedule.conditional_evaluation = if run.rich_conditional {
            let sparse = cadence * 16;
            Some(rl::diagnostics::schedule::Interval {
                first: sparse,
                every: sparse,
            })
        } else {
            interval
        };
        diagnostic.schedule.reference = interval;
        diagnostic.schedule.snapshots = interval;
        diagnostic.snapshot_retention = 64;
    }
    let periods: &[usize] = if run.profile == "detailed" {
        &[0, 6, 11]
    } else {
        &[0, 11]
    };
    let mut observations = Vec::new();
    for &period in periods {
        for q in [-0.5, 0., 0.5] {
            for alpha in [-0.5, -0.25, 0., 0.25, 0.5] {
                let mut probe = environment.clone();
                probe.state = State {
                    inventory: q,
                    alpha,
                    period,
                    ..State::default()
                };
                observations.push(probe.observation());
            }
        }
    }
    let surfaces = ProbeSet::new(
        "inventory-alpha-slices",
        ProbePopulation::FixedReference,
        "Fixed inventory, alpha and remaining-horizon slices",
        observations,
        None,
        space.clone(),
    )?;
    let cases = conditional_cases(run.rich_conditional);
    let actions: Vec<_> = cases
        .iter()
        .map(|(_, action)| Action::Continuous(vec![*action]))
        .collect();
    let mut branches = Vec::new();
    for (state, _) in cases {
        let mut branch = environment.clone();
        branch.state = state;
        branches.push(branch);
    }
    let conditional = ProbeSet::new(
        "conditional-alpha",
        ProbePopulation::FixedReference,
        "Fixed states and forced initial target inventories for learner-continuation evaluation",
        branches.iter().map(AlphaExecution::observation).collect(),
        Some(actions.clone()),
        space.clone(),
    )?;
    let mut builder = run
        .builder(&environment, diagnostic)
        .discount(1.)
        .environment_descriptor(descriptor())
        .reference_context(reference_context)
        .metadata(
            "alpha.model",
            "alpha_execution_v1; see alpha-execution-design.md",
        )
        .metadata(
            "evaluation.scenario_protocol",
            "Independent uniformly sampled initial alpha and Gaussian innovations",
        )
        .evaluation_factory_with_frames(move |seed| AlphaExecution::new(config, seed))
        .probe_set(surfaces)
        .probe_set(conditional);
    for (row, (branch, action)) in branches.into_iter().zip(actions).enumerate() {
        let observation = branch.observation();
        builder = builder.conditional_observation_case(
            "conditional-alpha",
            row,
            observation,
            Some(action),
            "Initialize inventory, available signal and remaining horizon from the observed Markov state; price and accumulated P&L are translation-invariant bookkeeping. Independently seed future signal and price innovations.",
            move |seed| {
                let mut sampled = branch.clone();
                sampled.rng = ChaCha8Rng::seed_from_u64(seed);
                sampled.last = None;
                let observation = sampled.observation();
                Ok((sampled, observation))
            },
        );
    }
    if run.reference {
        let grid = reference::demonstration_grid();
        builder = builder
            .reference(reference::NumericalProvider::new(config, grid)?)
            .reference_trajectories(1);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    let policy = Deterministic::new(3, space, &[32, 32], &mut rng)?;
    let (quantiles, width, tau) = critic_study_parameters(&run.training_preset);
    let mut critic = |heads| QValue::new(3, 1, heads, &[width, width], &mut rng);
    let training = OffPolicyConfig {
        gamma: 1.,
        ..super::off_policy_config()
    };
    match run.algorithm.as_str() {
        "alpha-ddpg" => run.train_off_policy(
            Ddpg::new(policy, critic(1)?, Default::default(), run.stream_seed(2))?,
            builder,
            environment,
            training,
        ),
        "alpha-d4pg" | "alpha-cvar" => {
            let mut learner = D4pgConfig::new(quantiles)?;
            learner.training.tau = tau;
            if run.algorithm == "alpha-cvar" {
                learner.risk = Risk::Cvar { alpha: 0.2 };
            }
            run.train_off_policy(
                D4pg::new(policy, critic(quantiles)?, learner, run.stream_seed(2))?,
                builder,
                environment,
                training,
            )
        }
        "alpha-md4pg" => run.train_off_policy(
            Md4pg::new(
                policy,
                critic(1)?,
                critic(1)?,
                Md4pgConfig {
                    risk: Risk::MeanVariance { penalty: 2. },
                    ..Default::default()
                },
                run.stream_seed(2),
            )?,
            builder,
            environment,
            training,
        ),
        _ => Err(Error::Config("unknown alpha-execution learner".into())),
    }
}

impl Env for AlphaExecution {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(3)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.state = State {
            alpha: self.rng.random_range(-0.5..0.5),
            ..State::default()
        };
        self.last = None;
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.space.check(action, None)?;
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        if self.state.period >= self.config.horizon {
            return Err(Error::State(
                "reset required after execution horizon".into(),
            ));
        }
        let alpha_noise = self.rng.sample(StandardNormal);
        let price_noise = self.rng.sample(StandardNormal);
        let result = transition(
            &self.config,
            self.state,
            f64::from(action[0]),
            alpha_noise,
            price_noise,
        )?;
        self.state = result.next;
        self.last = Some(result);
        Ok(Step {
            observation: self.observation(),
            reward: (result.reward as f32).into(),
            terminated: self.state.period == self.config.horizon,
            truncated: false,
        })
    }
}

impl BranchingEnv for AlphaExecution {
    fn branch(&self, future_seed: u64) -> Result<Self> {
        self.branch_observation()?;
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(future_seed),
            ..self.clone()
        })
    }
    fn branch_observation(&self) -> Result<Observation> {
        if self.state.period >= self.config.horizon {
            return Err(Error::State(
                "cannot branch a completed execution episode".into(),
            ));
        }
        Ok(self.observation())
    }
    fn branch_state(&self) -> Result<Vec<f64>> {
        Ok(vec![
            self.state.inventory,
            self.state.alpha,
            self.state.period as f64,
            self.state.price,
            self.state.cumulative_pnl,
        ])
    }
    fn branch_description(&self) -> Result<String> {
        Ok("Inventory, alpha, time, price and accumulated P&L are fixed; future independent innovations are reseeded.".into())
    }
}

impl FrameSource for AlphaExecution {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        let last = self.last;
        let terminal_trade = if self.state.period == self.config.horizon {
            last.map_or(0., |step| -step.post_trade_inventory)
        } else {
            0.
        };
        Ok(EnvironmentFrame {
            environment: "alpha_execution_v1".into(),
            simulation_time: Some(self.state.period as f64 * self.config.dt),
            channels: BTreeMap::from([
                ("price".into(), vec![self.state.price]),
                ("alpha".into(), vec![self.state.alpha]),
                (
                    "post_trade_alpha".into(),
                    vec![last.map_or(self.state.alpha, |step| step.post_trade_alpha)],
                ),
                ("inventory".into(), vec![self.state.inventory]),
                ("trade".into(), vec![last.map_or(0., |step| step.trade)]),
                ("terminal_liquidation".into(), vec![terminal_trade]),
                (
                    "incremental_pnl".into(),
                    vec![last.map_or(0., |step| step.reward)],
                ),
                ("cumulative_pnl".into(), vec![self.state.cumulative_pnl]),
                (
                    "execution_cost".into(),
                    vec![last.map_or(0., |step| step.execution_cost + step.liquidation_cost)],
                ),
                (
                    "price_change".into(),
                    vec![last.map_or(0., |step| step.price_change)],
                ),
            ]),
        })
    }
}

#[derive(Default)]
struct EvaluationSummary {
    pnl: Vec<f64>,
    turnover_total: f64,
    cost_total: f64,
    exposure_total: f64,
    depletion_total: f64,
}

fn evaluate_actor(actor: &dyn Actor, config: Config, seeds: &[u64]) -> Result<EvaluationSummary> {
    if seeds.is_empty() {
        return Err(Error::Config("alpha evaluation requires scenarios".into()));
    }
    let mut summary = EvaluationSummary::default();
    for (index, &seed) in seeds.iter().enumerate() {
        let mut env = AlphaExecution::new(config, seed)?;
        let mut observation = env.reset(Some(seed))?.observation;
        let mut actor_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut actions = Vec::new();
        let mut pnl = 0.;
        while env.state.period < config.horizon {
            let previous_alpha = env.state.alpha;
            actor.act_into(
                ObservationBatch::all(std::slice::from_ref(&observation)),
                ActMode::Deterministic,
                &mut actor_rng,
                &mut actions,
            )?;
            let step = env.step(&actions[0])?;
            let evidence = env.last.expect("step records its transition");
            pnl += f64::from(step.reward.total());
            summary.turnover_total += evidence.trade.abs()
                + if step.terminated {
                    evidence.post_trade_inventory.abs()
                } else {
                    0.
                };
            summary.cost_total += evidence.execution_cost + evidence.liquidation_cost;
            summary.exposure_total += evidence.post_trade_inventory.abs() * config.dt;
            summary.depletion_total += previous_alpha.abs() - evidence.post_trade_alpha.abs();
            observation = step.observation;
        }
        summary.pnl.push(pnl);
        if seeds.len() >= 50 && ((index + 1) % 50 == 0 || index + 1 == seeds.len()) {
            eprintln!(
                "alpha evaluation: {}/{} episodes complete",
                index + 1,
                seeds.len()
            );
        }
    }
    Ok(summary)
}

fn print_evaluation(label: &str, summary: &EvaluationSummary) {
    let n = summary.pnl.len();
    let mean = summary.pnl.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        summary.pnl.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.
    };
    let mut sorted = summary.pnl.clone();
    sorted.sort_by(f64::total_cmp);
    let q05 = sorted[((n as f64 * 0.05).ceil() as usize).saturating_sub(1)];
    let tail_weight = 0.1 * n as f64;
    let full = tail_weight.floor() as usize;
    let fractional = tail_weight - full as f64;
    let tail_sum: f64 = sorted.iter().take(full).sum::<f64>()
        + if full < n {
            fractional * sorted[full]
        } else {
            0.
        };
    println!(
        "policy,episodes,mean_net_pnl,pnl_std,mean_standard_error,pnl_q05,pnl_cvar10,mean_turnover,mean_execution_cost,mean_abs_inventory_time,mean_alpha_depletion"
    );
    println!(
        "{label},{n},{mean:.6},{:.6},{:.6},{q05:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
        variance.sqrt(),
        (variance / n as f64).sqrt(),
        tail_sum / tail_weight,
        summary.turnover_total / n as f64,
        summary.cost_total / n as f64,
        summary.exposure_total / n as f64,
        summary.depletion_total / n as f64
    );
}

fn paired_pnl_difference(learner: &[f64], reference: &[f64]) -> Result<(f64, f64)> {
    if learner.is_empty() || learner.len() != reference.len() {
        return Err(Error::Config(
            "paired evaluation requires equally sized, nonempty scenarios".into(),
        ));
    }
    let n = learner.len();
    let mean = learner
        .iter()
        .zip(reference)
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / n as f64;
    let variance = if n > 1 {
        learner
            .iter()
            .zip(reference)
            .map(|(a, b)| (a - b - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64
    } else {
        f64::NAN
    };
    Ok((mean, (variance / n as f64).sqrt()))
}

pub(super) fn benchmark(run: &super::Run) -> Result<()> {
    if run.steps > 10_000 {
        return Err(Error::Config(
            "alpha benchmark supports at most 10000 scenarios".into(),
        ));
    }
    let config = Config::default();
    let grid = reference::demonstration_grid();
    let seeds: Vec<_> = (0..run.steps)
        .map(|index| run.seed.wrapping_add(index))
        .collect();
    let aware = reference::NumericalProvider::new(config, grid)?;
    let unaware = reference::NumericalProvider::new(Config { eta: 0., ..config }, grid)?;
    let aware_summary = evaluate_actor(&aware, config, &seeds)?;
    let unaware_summary = evaluate_actor(&unaware, config, &seeds)?;
    print_evaluation("numerical-grid-benchmark", &aware_summary);
    print_evaluation("depletion-ignorant-grid-policy", &unaware_summary);
    let (difference, standard_error) =
        paired_pnl_difference(&unaware_summary.pnl, &aware_summary.pnl)?;
    println!("paired_comparison,episodes,mean_pnl_difference,standard_error");
    println!(
        "depletion-ignorant-minus-numerical,{},{difference:.6},{standard_error:.6}",
        seeds.len()
    );
    Ok(())
}

pub(super) fn evaluate_learner(
    actor: &dyn Actor,
    seeds: &[u64],
    include_reference: bool,
) -> Result<()> {
    let config = Config::default();
    let learner = evaluate_actor(actor, config, seeds)?;
    print_evaluation("learner", &learner);
    if include_reference {
        let grid = reference::demonstration_grid();
        let reference = reference::NumericalProvider::new(config, grid)?;
        let benchmark = evaluate_actor(&reference, config, seeds)?;
        print_evaluation("numerical-grid-benchmark", &benchmark);
        let (difference, standard_error) = paired_pnl_difference(&learner.pnl, &benchmark.pnl)?;
        println!("paired_comparison,episodes,mean_pnl_difference,standard_error");
        println!(
            "learner-minus-numerical,{},{difference:.6},{standard_error:.6}",
            seeds.len()
        );
        // This policy ignores depletion when planning, but is evaluated in the
        // unchanged depletion environment on exactly the same scenario seeds.
        let unaware = reference::NumericalProvider::new(Config { eta: 0., ..config }, grid)?;
        let unaware_summary = evaluate_actor(&unaware, config, seeds)?;
        print_evaluation("depletion-ignorant-grid-policy", &unaware_summary);
        let (difference, standard_error) =
            paired_pnl_difference(&unaware_summary.pnl, &benchmark.pnl)?;
        println!("paired_comparison,episodes,mean_pnl_difference,standard_error");
        println!(
            "depletion-ignorant-minus-numerical,{},{difference:.6},{standard_error:.6}",
            seeds.len()
        );
    }
    Ok(())
}

fn conditional_returns(
    actor: &dyn Actor,
    config: Config,
    state: State,
    first_action: f32,
    seeds: &[u64],
    progress: bool,
) -> Result<Vec<f64>> {
    if seeds.is_empty() || seeds.len() > 10_000 {
        return Err(Error::Config(
            "conditional evaluation requires 1 to 10000 futures".into(),
        ));
    }
    let start = std::time::Instant::now();
    let mut returns = Vec::with_capacity(seeds.len());
    for (index, &seed) in seeds.iter().enumerate() {
        let mut env = AlphaExecution::new(config, seed)?;
        env.state = state;
        let mut observation = env.observation();
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
        let mut actions = Vec::new();
        let mut total = 0.;
        let mut first = true;
        while env.state.period < config.horizon {
            if first {
                actions.clear();
                actions.push(Action::Continuous(vec![first_action]));
                first = false;
            } else {
                actor.act_into(
                    ObservationBatch::all(std::slice::from_ref(&observation)),
                    ActMode::Deterministic,
                    &mut rng,
                    &mut actions,
                )?;
            }
            let step = env.step(&actions[0])?;
            total += f64::from(step.reward.total());
            observation = step.observation;
        }
        returns.push(total);
        if progress && ((index + 1) % 500 == 0 || index + 1 == seeds.len()) {
            let elapsed = start.elapsed().as_secs_f64();
            let remaining = elapsed * (seeds.len() - index - 1) as f64 / (index + 1) as f64;
            eprintln!(
                "conditional evaluation: {}/{} futures, {:.1}s elapsed, {:.1}s estimated remaining",
                index + 1,
                seeds.len(),
                elapsed,
                remaining
            );
        }
    }
    Ok(returns)
}

pub(super) fn reference_check(run: &super::Run) -> Result<()> {
    use rl::diagnostics::reference::{ReferenceCheckpoint, ReferenceProvider};
    if run.steps < 2 || run.steps > 10_000 {
        return Err(Error::Config(
            "reference check needs 2 to 10000 futures".into(),
        ));
    }
    let config = Config::default();
    let grid = reference::Grid {
        inventory: 41,
        alpha: 161,
        actions: 81,
        quadrature: 9,
        alpha_limit: 2.,
    };
    let provider = reference::NumericalProvider::new(config, grid)?;
    let checkpoint = ReferenceCheckpoint {
        revision: 0,
        policy_id: "numerical-grid-policy",
        policy: &provider,
        mode: ActMode::Deterministic,
    };
    let seeds: Vec<_> = (0..run.steps)
        .map(|offset| run.seed.wrapping_add(offset))
        .collect();
    println!(
        "period,inventory,alpha,forced_target,futures,numerical_q,mc_mean,mc_std,mc_standard_error,mc_q05,mc_cvar10"
    );
    for (period, inventory, alpha, action) in [
        (0, 0., 0.2, 0.5f32),
        (0, 0., 0., 0.),
        (6, 0.5, -0.2, 0.),
        (11, 0.2, 0.4, 0.6),
    ] {
        eprintln!(
            "conditional case: period={period}, inventory={inventory}, alpha={alpha}, target={action}"
        );
        let state = State {
            inventory,
            alpha,
            period,
            ..State::default()
        };
        let mut source = AlphaExecution::new(config, 0)?;
        source.state = state;
        let predicted = provider
            .action_value(
                &checkpoint,
                &source.observation(),
                &Action::Continuous(vec![action]),
            )?
            .ok_or_else(|| Error::State("reference case outside domain".into()))?;
        let values = conditional_returns(&provider, config, state, action, &seeds, true)?;
        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let sd = (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt();
        let mut ordered = values;
        ordered.sort_by(f64::total_cmp);
        let q05 = ordered[((n as f64 * 0.05).ceil() as usize).saturating_sub(1)];
        let tail_n = ((n as f64 * 0.1).ceil() as usize).max(1);
        let cvar = ordered.iter().take(tail_n).sum::<f64>() / tail_n as f64;
        println!(
            "{period},{inventory:.3},{alpha:.3},{action:.3},{n},{predicted:.6},{mean:.6},{sd:.6},{:.6},{q05:.6},{cvar:.6}",
            sd / (n as f64).sqrt()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rich_conditional_cases_compare_multiple_actions_at_each_matched_state() {
        let cases = conditional_cases(true);
        assert_eq!(cases.len(), 9);
        for group in cases.chunks_exact(3) {
            assert!(
                group
                    .iter()
                    .all(|(state, _)| state.inventory == group[0].0.inventory
                        && state.alpha == group[0].0.alpha
                        && state.period == group[0].0.period)
            );
            assert_eq!(
                group.iter().map(|(_, action)| *action).collect::<Vec<_>>(),
                vec![-0.5, 0., 0.5]
            );
        }
        assert_eq!(conditional_cases(false).len(), 3);
    }

    #[test]
    fn targeted_critic_settings_change_one_factor_at_a_time() {
        assert_eq!(critic_study_parameters("baseline"), (17, 32, 0.005));
        assert_eq!(critic_study_parameters("alpha-quantiles"), (33, 32, 0.005));
        assert_eq!(critic_study_parameters("alpha-wide"), (17, 64, 0.005));
        assert_eq!(critic_study_parameters("alpha-tau"), (17, 32, 0.02));
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn price_and_past_pnl_do_not_change_conditional_future_rewards() -> Result<()> {
        let state = State {
            inventory: 0.2,
            alpha: 0.3,
            period: 4,
            price: 0.,
            cumulative_pnl: 0.,
        };
        let changed_bookkeeping = State {
            price: 100.,
            cumulative_pnl: -42.,
            ..state
        };
        let first = transition(&Config::default(), state, 0.4, -0.2, 0.7)?;
        let second = transition(&Config::default(), changed_bookkeeping, 0.4, -0.2, 0.7)?;
        assert_eq!(first.reward, second.reward);
        assert_eq!(first.next.inventory, second.next.inventory);
        assert_eq!(first.next.alpha, second.next.alpha);
        assert_eq!(first.next.period, second.next.period);
        Ok(())
    }

    #[test]
    fn paired_comparison_uses_scenario_differences() -> Result<()> {
        let (mean, standard_error) = paired_pnl_difference(&[2., 5., 8.], &[1., 2., 3.])?;
        close(mean, 3.);
        close(standard_error, (4.0_f64 / 3.0).sqrt());
        assert!(paired_pnl_difference(&[1.], &[]).is_err());
        Ok(())
    }

    #[test]
    fn one_period_accounting_charges_trade_and_terminal_liquidation_once() {
        let config = Config {
            horizon: 1,
            eta: 0.,
            sigma_price: 0.,
            sigma_alpha: 0.,
            spread: 0.1,
            impact: 0.05,
            ..Config::default()
        };
        let state = State {
            inventory: 0.2,
            alpha: 0.4,
            ..State::default()
        };
        let step = transition(&config, state, 0.6, 0., 0.).unwrap();
        close(step.trade, 0.4);
        close(step.execution_cost, 0.048);
        close(step.liquidation_cost, 0.078);
        close(step.price_change, 0.4);
        close(step.reward, 0.114);
        close(step.next.inventory, 0.);
        close(step.next.cumulative_pnl, step.reward);
        assert_eq!(step.next.period, 1);
    }

    #[test]
    fn depletion_precedes_price_move_and_alpha_mean_reversion() {
        let config = Config {
            eta: 0.8,
            kappa: 0.25,
            sigma_alpha: 0.,
            sigma_price: 0.,
            spread: 0.,
            impact: 0.,
            ..Config::default()
        };
        let state = State {
            inventory: 0.,
            alpha: 0.5,
            ..State::default()
        };
        let step = transition(&config, state, 0.5, 0., 0.).unwrap();
        let remaining = 0.5 * (-0.4f64).exp();
        close(step.post_trade_alpha, remaining);
        close(step.price_change, remaining);
        close(step.reward, 0.5 * remaining);
        close(step.next.alpha, remaining * (-0.25f64).exp());
        close(step.next.inventory, 0.5);
    }

    #[test]
    fn zero_reversion_uses_brownian_limit_and_actions_remain_bounded() {
        let config = Config {
            kappa: 0.,
            sigma_alpha: 0.3,
            dt: 4.,
            eta: 0.,
            ..Config::default()
        };
        let state = State {
            alpha: 0.2,
            ..State::default()
        };
        let step = transition(&config, state, -1., 0.5, 0.).unwrap();
        close(step.next.alpha, 0.5);
        assert!(transition(&config, state, 1.01, 0., 0.).is_err());
        assert!(transition(&config, state, f64::NAN, 0., 0.).is_err());
    }

    #[test]
    fn seeded_episodes_and_counterfactual_futures_are_reproducible() -> Result<()> {
        use rl::diagnostics::evaluation::BranchingEnv;
        use rl::prelude::{Action, Env};
        let mut first = AlphaExecution::new(Config::default(), 123)?;
        let mut second = AlphaExecution::new(Config::default(), 123)?;
        assert_eq!(
            first.reset(Some(123))?.observation,
            second.reset(Some(123))?.observation
        );
        let action = Action::Continuous(vec![0.4]);
        assert_eq!(
            first.step(&action)?.observation,
            second.step(&action)?.observation
        );
        let mut branch_a = first.branch(42)?;
        let mut branch_b = first.branch(42)?;
        let mut branch_c = first.branch(43)?;
        let a = branch_a.step(&action)?;
        let b = branch_b.step(&action)?;
        let c = branch_c.step(&action)?;
        assert_eq!(a.observation, b.observation);
        assert_eq!(a.reward.total(), b.reward.total());
        assert_ne!(a.observation, c.observation);
        Ok(())
    }

    #[test]
    fn episode_terminates_with_no_inventory_and_reports_realized_pnl() -> Result<()> {
        use rl::diagnostics::evaluation::FrameSource;
        use rl::prelude::{Action, Env};
        let mut environment = AlphaExecution::new(
            Config {
                horizon: 2,
                ..Config::default()
            },
            5,
        )?;
        environment.reset(Some(5))?;
        let first = environment.step(&Action::Continuous(vec![0.5]))?;
        let second = environment.step(&Action::Continuous(vec![-0.25]))?;
        assert!(!first.ended());
        assert!(second.terminated && !second.truncated);
        let frame = environment.diagnostic_frame()?;
        assert_eq!(frame.channels["inventory"], vec![0.]);
        assert!(
            (frame.channels["cumulative_pnl"][0]
                - f64::from(first.reward.total())
                - f64::from(second.reward.total()))
            .abs()
                < 1e-6
        );
        assert_eq!(frame.channels["terminal_liquidation"], vec![0.25]);
        assert!(environment.step(&Action::Continuous(vec![0.])).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_training_records_alpha_environment_and_numerical_benchmark() -> Result<()> {
        use rl::diagnostics::{
            schema::RecordKind,
            storage::{ReadLimits, Recording},
        };
        let directory = tempfile::tempdir()?;
        let run = super::super::Run {
            algorithm: "alpha-ddpg".into(),
            path: directory.path().join("alpha"),
            profile: "detailed".into(),
            reference: true,
            live: false,
            wide_lqg: false,
            render: false,
            replay_population: false,
            seed: 7,
            steps: 8,
            evaluation_episodes: 0,
            evaluation_seed: 200_000,
            training_preset: "baseline".into(),
            rocket_guarded: false,
            rich_conditional: false,
        };
        super::run(&run)?;
        let recording = Recording::open(&run.path, ReadLimits::default())?;
        assert_eq!(recording.manifest().environment, "alpha_execution_v1");
        assert_eq!(recording.manifest().discount, 1.);
        assert!(
            recording.entries().iter().any(
                |entry| entry.kind == RecordKind::ModelSnapshot && entry.model_role == "online"
            )
        );
        assert!(
            recording
                .entries()
                .iter()
                .any(|entry| entry.kind == RecordKind::Trajectory
                    && entry.model_role == "reference:alpha-grid-dp")
        );
        let index = recording
            .entries()
            .iter()
            .position(|entry| entry.kind == RecordKind::Trajectory && entry.model_role == "online")
            .unwrap();
        let trajectory = recording.load(index)?;
        assert!(trajectory.arrays.contains_key("frames/post_trade_alpha"));
        assert!(trajectory.arrays.contains_key("frames/cumulative_pnl"));
        let declaration = recording
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.kind == RecordKind::Empirical && entry.model_role == "online"
            })
            .find_map(|(index, _)| {
                let chunk = recording.load(index).ok()?;
                rl::diagnostics::empirical::validate_for(&chunk, recording.manifest()).ok()
            })
            .expect("declared conditional samples");
        assert!(matches!(
            declaration.conditioning,
            rl::diagnostics::schema::Conditioning::Observation { .. }
        ));
        Ok(())
    }

    #[test]
    fn independent_evaluation_of_zero_inventory_has_zero_pnl_and_costs() -> Result<()> {
        struct ZeroActor(ActionSpace);
        impl Actor for ZeroActor {
            fn action_space(&self) -> &ActionSpace {
                &self.0
            }
            fn act_into(
                &self,
                observations: ObservationBatch<'_>,
                _: ActMode,
                _: &mut ChaCha8Rng,
                output: &mut Vec<Action>,
            ) -> Result<()> {
                output.clear();
                output.extend((0..observations.len()).map(|_| Action::Continuous(vec![0.])));
                Ok(())
            }
        }
        let environment = AlphaExecution::new(Config::default(), 7)?;
        let actor = ZeroActor(environment.action_space().clone());
        let result = evaluate_actor(&actor, Config::default(), &[1, 2, 3])?;
        assert_eq!(result.pnl, vec![0., 0., 0.]);
        assert_eq!(result.turnover_total, 0.);
        assert_eq!(result.cost_total, 0.);
        assert_eq!(result.depletion_total, 0.);
        Ok(())
    }

    #[test]
    fn conditional_rollouts_match_an_analytic_one_period_return() -> Result<()> {
        struct ZeroActor(ActionSpace);
        impl Actor for ZeroActor {
            fn action_space(&self) -> &ActionSpace {
                &self.0
            }
            fn act_into(
                &self,
                observations: ObservationBatch<'_>,
                _: ActMode,
                _: &mut ChaCha8Rng,
                out: &mut Vec<Action>,
            ) -> Result<()> {
                out.clear();
                out.extend((0..observations.len()).map(|_| Action::Continuous(vec![0.])));
                Ok(())
            }
        }
        let config = Config {
            horizon: 1,
            eta: 0.,
            sigma_alpha: 0.,
            sigma_price: 0.,
            spread: 0.1,
            impact: 0.05,
            ..Config::default()
        };
        let actor = ZeroActor(AlphaExecution::new(config, 0)?.space);
        let state = State {
            inventory: 0.2,
            alpha: 0.4,
            ..State::default()
        };
        let values = conditional_returns(&actor, config, state, 0.6, &[1, 2, 3], false)?;
        for value in values {
            assert!((value - 0.114).abs() < 1e-6);
        }
        Ok(())
    }
}
