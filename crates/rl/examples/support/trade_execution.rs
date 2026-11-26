//! Historical quadratic inventory execution with a stochastic mark-to-market reward.
use super::{GAMMA, Run, off_policy_config};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    diagnostics::{
        analysis::QuantileForecast,
        evaluation::{BranchingEnv, EnvironmentFrame, FrameSource},
        probes::ProbeSet,
        reference::{
            ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
        },
        schema::ProbePopulation,
        value::ValueConventions,
        visualization::*,
    },
    objectives::risk::Risk,
    prelude::*,
};
use std::collections::BTreeMap;

const HORIZON: usize = 8;
const INVENTORY_COST: f64 = 0.03;
const TEMPORARY_COST: f64 = 0.65;
const TERMINAL_COST: f64 = 1.;
const IMPACT: f64 = 0.01;
const VOLATILITY: f64 = 0.30;
const QUANTILES: usize = 17;
const CVAR_ALPHA: f32 = 0.2;
const VARIANCE_PENALTY: f32 = 2.;
const CONDITION_PROBES: &str = "initial-trade-counterfactuals";

#[derive(Clone, Copy)]
struct ExecutionState {
    inventory: f64,
    price: f64,
    period: usize,
}

impl ExecutionState {
    fn initial() -> Self {
        Self {
            inventory: 1.,
            price: 1.,
            period: 0,
        }
    }

    fn observation(self) -> Observation {
        Observation::new(vec![
            self.inventory as f32,
            self.price as f32,
            (HORIZON - self.period) as f32 / HORIZON as f32,
        ])
    }

    fn from_observation(observation: &Observation) -> Result<Self> {
        if let &[inventory, price, remaining] = observation.dense_values()
            && inventory.is_finite()
            && price.is_finite()
            && (0. ..=1.).contains(&remaining)
        {
            Ok(Self {
                inventory: f64::from(inventory),
                price: f64::from(price),
                period: HORIZON - (remaining * HORIZON as f32).round() as usize,
            })
        } else {
            Err(Error::Config("invalid execution observation".into()))
        }
    }

    /// The disturbance is a standard Normal innovation in the quoted price.
    fn transition(self, trade: f64, innovation: f64) -> (Self, f64) {
        let inventory = self.inventory - trade;
        let period = self.period + 1;
        let shock = VOLATILITY * innovation;
        let terminal_cost = if period == HORIZON {
            TERMINAL_COST * inventory.powi(2)
        } else {
            0.
        };
        let reward = inventory * shock
            - INVENTORY_COST * self.inventory.powi(2)
            - TEMPORARY_COST * trade.powi(2)
            - terminal_cost;
        (
            Self {
                inventory,
                price: self.price + shock - IMPACT * trade,
                period,
            },
            reward,
        )
    }
}

#[derive(Clone)]
struct Execution {
    state: ExecutionState,
    space: ActionSpace,
    rng: ChaCha8Rng,
}

impl Execution {
    fn new(seed: u64) -> Self {
        Self {
            state: ExecutionState {
                period: HORIZON,
                ..ExecutionState::initial()
            },
            space: ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Env for Execution {
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
        self.state = ExecutionState::initial();
        Ok(EpisodeStart::fresh(self.state.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.state.period >= HORIZON {
            return Err(Error::State(
                "reset required after execution horizon".into(),
            ));
        }
        let trade = trade_size(&self.space, action)?;
        let (next, reward) = self
            .state
            .transition(trade, self.rng.sample(StandardNormal));
        self.state = next;
        Ok(Step {
            observation: next.observation(),
            reward: (reward as f32).into(),
            terminated: next.period == HORIZON,
            truncated: false,
        })
    }
}

fn trade_size(space: &ActionSpace, action: &Action) -> Result<f64> {
    space.check(action, None)?;
    let Action::Continuous(action) = action else {
        unreachable!()
    };
    Ok(f64::from(action[0]))
}

impl BranchingEnv for Execution {
    fn branch(&self, future_seed: u64) -> Result<Self> {
        self.branch_observation()?;
        Ok(Self {
            rng: ChaCha8Rng::seed_from_u64(future_seed),
            ..self.clone()
        })
    }

    fn branch_observation(&self) -> Result<Observation> {
        if self.state.period >= HORIZON {
            return Err(Error::State(
                "reset required after execution horizon".into(),
            ));
        }
        Ok(self.state.observation())
    }

    fn branch_state(&self) -> Result<Vec<f64>> {
        Ok(vec![
            self.state.inventory,
            self.state.price,
            self.state.period as f64,
        ])
    }

    fn branch_description(&self) -> Result<String> {
        Ok("Inventory, quote and decision period are fixed. Future price innovations are independently reseeded.".into())
    }
}

impl FrameSource for Execution {
    fn diagnostic_frame(&self) -> Result<EnvironmentFrame> {
        Ok(EnvironmentFrame {
            environment: "trade_execution".into(),
            simulation_time: Some(self.state.period as f64),
            channels: BTreeMap::from([
                ("inventory".into(), vec![self.state.inventory]),
                ("price".into(), vec![self.state.price]),
            ]),
        })
    }
}

struct ExecutionReference {
    specification: ReferenceSpec,
    gains: [f64; HORIZON],
    standard_normal_quantiles: [f64; QUANTILES],
}

impl ExecutionReference {
    fn new(space: ActionSpace) -> Result<Self> {
        let mut gains = [0.; HORIZON];
        let mut next_cost = TERMINAL_COST;
        for period in (0..HORIZON).rev() {
            let next_inventory_cost = if period == HORIZON - 1 {
                next_cost
            } else {
                f64::from(GAMMA) * next_cost
            };
            gains[period] = next_inventory_cost / (TEMPORARY_COST + next_inventory_cost);
            next_cost = INVENTORY_COST + TEMPORARY_COST * gains[period];
        }
        let specification = ReferenceSpec {
            id: "quadratic-inventory-policy".into(),
            context: context(space),
            quality: ReferenceQuality {
                kind: ReferenceKind::Analytical,
                method: "Finite-horizon quadratic expected-cost feedback with clipped, f32 trade sizes and a Gaussian return law for its continuation".into(),
                assumptions: vec![
                    "The fixed policy ignores price, so future exposure weights are deterministic.".into(),
                    "Closed-form moments treat per-step rewards as real numbers before f32 reward rounding.".into(),
                    "Quantile values use uniform probability midpoints. Derived distribution summaries use that grid.".into(),
                ],
            },
            target: ReferenceTarget::BenchmarkPolicy,
            lifetime: ReferenceLifetime::Fixed,
            information: ReferenceInformation::LearnerObservation,
            uncertainty: None,
        };
        specification.validate()?;
        Ok(Self {
            specification,
            gains,
            standard_normal_quantiles: std::array::from_fn(|i| {
                let location = (i as f64 + 0.5) / QUANTILES as f64;
                let (mut low, mut high) = (-8., 8.);
                for _ in 0..48 {
                    let mid = (low + high) * 0.5;
                    if 0.5 * (1. + libm::erf(mid / std::f64::consts::SQRT_2)) < location {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                (low + high) * 0.5
            }),
        })
    }

    fn trade(&self, state: ExecutionState) -> f32 {
        (self.gains.get(state.period).copied().unwrap_or(0.) * f64::from(state.inventory as f32))
            .clamp(-1., 1.) as f32
    }

    fn moments(&self, mut state: ExecutionState, first: Option<&Action>) -> Result<(f64, f64)> {
        let mut first = first
            .map(|action| trade_size(self.action_space(), action))
            .transpose()?;
        let (mut mean, mut variance, mut discount) = (0., 0., 1.);
        for _ in state.period..HORIZON {
            let trade = first.take().unwrap_or_else(|| f64::from(self.trade(state)));
            let (next, reward) = state.transition(trade, 0.);
            mean += discount * reward;
            variance += (discount * next.inventory * VOLATILITY).powi(2);
            discount *= f64::from(GAMMA);
            state = next;
        }
        Ok((mean, variance))
    }
}

impl Actor for ExecutionReference {
    fn action_space(&self) -> &ActionSpace {
        &self.specification.context.action_space
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
            let state = ExecutionState::from_observation(observation)?;
            actions.push(Action::Continuous(vec![self.trade(state)]));
        }
        Ok(())
    }
}

impl ReferenceProvider for ExecutionReference {
    fn specification(&self) -> &ReferenceSpec {
        &self.specification
    }

    fn actor(&self) -> Option<&dyn Actor> {
        Some(self)
    }

    fn evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn state_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
    ) -> Result<Option<f64>> {
        let (mean, _) = self.moments(ExecutionState::from_observation(observation)?, None)?;
        Ok(Some(mean))
    }

    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        let (mean, _) =
            self.moments(ExecutionState::from_observation(observation)?, Some(action))?;
        Ok(Some(mean))
    }

    fn return_distribution(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<QuantileForecast>> {
        let (mean, variance) =
            self.moments(ExecutionState::from_observation(observation)?, Some(action))?;
        QuantileForecast::new(
            self.standard_normal_quantiles
                .iter()
                .map(|z| mean + variance.sqrt() * z)
                .collect(),
            (0..QUANTILES)
                .map(|i| (i as f64 + 0.5) / QUANTILES as f64)
                .collect(),
            vec![1. / QUANTILES as f64; QUANTILES],
        )
        .map(Some)
    }
}

fn context(action_space: ActionSpace) -> ReferenceContext {
    ReferenceContext {
        environment: "trade_execution".into(),
        observation_dim: 3,
        typed_observation_space: None,
        action_space,
        conventions: ValueConventions {
            dynamics: "Inventory decreases by signed trade and quote follows independent Gaussian innovations with linear impact".into(),
            observation: "Inventory, quoted price and remaining decision fraction".into(),
            action: "Bounded signed trade where positive values sell inventory".into(),
            reward_units: "normalized P&L".into(),
            reward_convention: "Post-trade inventory times the exogenous quote innovation minus inventory, temporary trading and terminal inventory costs".into(),
            discount: f64::from(GAMMA),
            horizon: format!("{HORIZON} decisions"),
            terminal_treatment: "Finite execution horizon charges residual inventory in the final transition".into(),
            action_constraints: "Signed trade size in [-1, 1]. Short inventory is permitted.".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: HORIZON as u32 },
        environment_config: BTreeMap::from([
            ("horizon".into(), ConfigValue::Integer(HORIZON as i64)),
            ("shock_std".into(), ConfigValue::Real(VOLATILITY)),
            ("linear_impact".into(), ConfigValue::Real(IMPACT)),
            ("inventory_cost".into(), ConfigValue::Real(INVENTORY_COST)),
            ("temporary_cost".into(), ConfigValue::Real(TEMPORARY_COST)),
            ("terminal_cost".into(), ConfigValue::Real(TERMINAL_COST)),
        ]),
        markov_observation: false,
        decision_time_coordinate: Some(2),
    }
}

fn descriptor() -> EnvironmentDescriptor {
    EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations: [
            ("inventory", "Inventory", None),
            ("price", "Quoted price", None),
            ("remaining_fraction", "Remaining decision fraction", Some([0., 1.])),
        ].into_iter().map(|(id, label, range)| ObservationField {
            id: id.into(), label: label.into(), unit: None, description: None,
            kind: ObservationType::Numerical, range, coordinate: true,
        }).collect(),
        actions: ActionDescriptor::Continuous {
            components: vec![ActionComponent {
                label: "Signed trade size".into(), unit: None,
                meaning: Some("Positive sells inventory and negative buys inventory".into()),
            }],
            preferred_component: Some(0),
        },
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel { label: "Learner revision".into(), unit: None },
            decision_step: QuantityLabel { label: "Decision".into(), unit: None },
            simulation: Some(QuantityLabel { label: "Execution period".into(), unit: None }),
        },
        reward: RewardDescriptor {
            label: "Inventory exposure P&L minus costs".into(), unit: Some("normalized P&L".into()),
            description: Some("Post-trade inventory earns the independently sampled exogenous quote innovation. Pre-trade inventory, trade size, and final residual inventory incur quadratic charges.".into()),
            components: Vec::new(),
        },
        frames: [("inventory", "Inventory"), ("price", "Quoted price")].into_iter().map(|(name, label)| FrameChannel {
            name: name.into(), label: label.into(), width: 1, unit: None,
            visibility: FrameVisibility::DiagnosticOnly,
            representation: Some(FrameRepresentation::Scalar),
        }).collect(),
        preferences: VisualizationPreferences {
            coordinates: vec!["inventory".into(), "remaining_fraction".into()],
            trajectories: ["inventory", "price"].into_iter().map(|channel| FrameReference { channel: channel.into(), index: 0 }).collect(),
            groups: Vec::new(),
        },
    }
}

pub(super) fn run(run: &Run) -> Result<()> {
    let mut environment = Execution::new(run.seed);
    let initial = environment.reset(None)?.observation;
    let space = environment.action_space().clone();
    let fixed = [0., 0.5, 1.].map(|trade| Action::Continuous(vec![trade]));
    let probes = ProbeSet::new(
        CONDITION_PROBES,
        ProbePopulation::FixedReference,
        "Same starting inventory and quote with three forced first trades",
        vec![initial; fixed.len()],
        Some(fixed.to_vec()),
        space.clone(),
    )?;
    let observations = [0, 2, 4, 6]
        .into_iter()
        .flat_map(|period| {
            [-0.5, 0., 0.5, 1.].into_iter().map(move |inventory| {
                ExecutionState {
                    inventory,
                    price: 1.,
                    period,
                }
                .observation()
            })
        })
        .collect();
    let slice = ProbeSet::new(
        "inventory-and-period",
        ProbePopulation::UserSupplied,
        "Inventory by remaining execution fraction, at a fixed quote of one",
        observations,
        None,
        space.clone(),
    )?;
    let mut config = run.config(HORIZON)?;
    config.evaluation.mode = ActMode::Deterministic;
    config.conditional_evaluation.mode = ActMode::Deterministic;
    config.conditional_evaluation.seeds = (0..64).map(|offset| 0x8d2d_2200 + offset).collect();
    let mut builder = run
        .builder(&environment, config)
        .environment_descriptor(descriptor())
        .reference_context(context(space.clone()))
        .metadata("execution.initial_inventory", "1")
        .metadata("execution.initial_quote", "1")
        .evaluation_factory_with_frames(|seed| Ok(Execution::new(seed)))
        .probe_set(probes)
        .probe_set(slice);
    for (row, trade) in fixed.into_iter().enumerate() {
        builder =
            builder.conditional_case(CONDITION_PROBES, row, environment.clone(), Some(trade))?;
    }
    if run.reference {
        builder = builder
            .reference(ExecutionReference::new(space.clone())?)
            .reference_trajectories(1);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    let policy = Deterministic::new(3, space, &[16, 16], &mut rng)?;
    let mut critic = |heads| QValue::new(3, 1, heads, &[16, 16], &mut rng);
    let training = off_policy_config();
    if run.algorithm == "trade-md4pg" {
        run.train_off_policy(
            Md4pg::new(
                policy,
                critic(1)?,
                critic(1)?,
                Md4pgConfig {
                    risk: Risk::MeanVariance {
                        penalty: VARIANCE_PENALTY,
                    },
                    ..Default::default()
                },
                run.stream_seed(2),
            )?,
            builder,
            environment,
            training,
        )
    } else {
        let mut config = D4pgConfig::new(QUANTILES)?;
        config.risk = Risk::Cvar { alpha: CVAR_ALPHA };
        run.train_off_policy(
            D4pg::new(policy, critic(QUANTILES)?, config, run.stream_seed(2))?,
            builder,
            environment,
            training,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_period_return_matches_gaussian_exposure_and_terminal_cost() -> Result<()> {
        let mut environment = Execution::new(0);
        let reference = ExecutionReference::new(environment.action_space().clone())?;
        let action = Action::Continuous(vec![0.25]);
        let state = ExecutionState {
            inventory: 1.,
            price: 1.,
            period: HORIZON - 1,
        };
        let (mean, variance) = reference.moments(state, Some(&action))?;
        assert!((mean - (-0.633125)).abs() < 1e-12);
        assert!((variance - 0.050625).abs() < 1e-12);

        let samples = (0..4096)
            .map(|seed| {
                environment.state = state;
                let mut branch = environment.branch(seed)?;
                let step = branch.step(&action)?;
                assert!(step.terminated);
                assert!(!step.truncated);
                assert!(branch.step(&action).is_err());
                Ok(f64::from(step.reward.total()))
            })
            .collect::<Result<Vec<_>>>()?;
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let sample_variance = samples
            .iter()
            .map(|r| (r - sample_mean).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64;
        assert!((sample_mean - mean).abs() < 0.012);
        assert!((sample_variance - variance).abs() < 0.005);

        let (liquidation_mean, liquidation_variance) =
            reference.moments(state, Some(&Action::Continuous(vec![1.])))?;
        assert_eq!(liquidation_variance, 0.);
        assert!(liquidation_mean < mean);
        assert!(liquidation_mean > mean - 2. * variance);
        Ok(())
    }

    #[test]
    fn earlier_innovations_and_terminal_innovations_use_squared_discount_in_variance() -> Result<()>
    {
        let environment = Execution::new(0);
        let reference = ExecutionReference::new(environment.action_space().clone())?;
        let state = ExecutionState {
            inventory: 1.,
            price: 1.,
            period: HORIZON - 2,
        };
        let (mean, variance) = reference.moments(state, Some(&Action::Continuous(vec![0.25])))?;
        // The last-period expected-cost minimizer for remaining inventory 0.75
        // is 0.75 / 1.65, rounded by the actor to a valid f32 action.
        let final_trade = f64::from((0.75_f64 / 1.65) as f32);
        let final_inventory = 0.75 - final_trade;
        let final_cost = 0.016875 + 0.65 * final_trade.powi(2) + final_inventory.powi(2);
        let discount = f64::from(0.95_f32);
        assert!((mean - (-0.070625 - discount * final_cost)).abs() < 1e-12);
        assert!((variance - (0.050625 + (discount * 0.3 * final_inventory).powi(2))).abs() < 1e-12);
        Ok(())
    }
}
