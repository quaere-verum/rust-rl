//! Fixed numerical scenarios for the diagnostic-session example.
use super::*;
use nalgebra::DMatrix;
use rl::{
    diagnostics::{
        render::{RenderCaptureConfig, RenderLimits},
        visualization::{
            ActionComponent, ActionDescriptor, ClockDescriptor, DiscreteChoice,
            ENVIRONMENT_DESCRIPTOR_VERSION, EnvironmentDescriptor, ObservationField,
            ObservationType, QuantityLabel, RewardDescriptor, VisualizationPreferences,
        },
    },
    envs::pendulum::Pendulum,
};

const WIDE_STATES: usize = 16;
const WIDE_ACTIONS: usize = 2;
const PENDULUM_HORIZON: usize = 200;

fn coordinate(
    id: &str,
    label: &str,
    unit: Option<&str>,
    range: Option<[f32; 2]>,
) -> ObservationField {
    ObservationField {
        id: id.into(),
        label: label.into(),
        unit: unit.map(str::to_owned),
        description: None,
        kind: ObservationType::Numerical,
        range,
        coordinate: true,
    }
}

fn episode_descriptor(
    observations: Vec<ObservationField>,
    actions: ActionDescriptor,
    coordinates: [&str; 3],
    reward: &str,
) -> EnvironmentDescriptor {
    EnvironmentDescriptor {
        version: ENVIRONMENT_DESCRIPTOR_VERSION,
        observations,
        actions,
        clocks: ClockDescriptor {
            learner_revision: QuantityLabel {
                label: "Learner revision".into(),
                unit: None,
            },
            decision_step: QuantityLabel {
                label: "Decision step".into(),
                unit: None,
            },
            simulation: Some(QuantityLabel {
                label: "Elapsed".into(),
                unit: Some("s".into()),
            }),
        },
        reward: RewardDescriptor {
            label: "Reward".into(),
            unit: None,
            description: Some(reward.into()),
            components: Vec::new(),
        },
        frames: Vec::new(),
        preferences: VisualizationPreferences {
            coordinates: coordinates.map(str::to_owned).into(),
            ..Default::default()
        },
    }
}

pub(super) fn cartpole_descriptor() -> EnvironmentDescriptor {
    episode_descriptor(
        vec![
            coordinate("position", "Cart position", Some("m"), None),
            coordinate("velocity", "Cart velocity", Some("m/s"), None),
            coordinate("angle", "Pole angle", Some("rad"), None),
            coordinate(
                "angular-velocity",
                "Pole angular velocity",
                Some("rad/s"),
                None,
            ),
        ],
        ActionDescriptor::Discrete {
            choices: ["Push left", "Push right"]
                .map(|label| DiscreteChoice {
                    label: label.into(),
                    meaning: None,
                })
                .into(),
        },
        ["angle", "angular-velocity", "position"],
        "One unit per executed decision.",
    )
}

fn pendulum_descriptor() -> EnvironmentDescriptor {
    episode_descriptor(
        vec![
            coordinate("cosine", "Cosine of angle", None, Some([-1., 1.])),
            coordinate("sine", "Sine of angle", None, Some([-1., 1.])),
            coordinate(
                "angular-velocity",
                "Angular velocity",
                Some("rad/s"),
                Some([-8., 8.]),
            ),
        ],
        ActionDescriptor::Continuous {
            components: vec![ActionComponent {
                label: "Torque".into(),
                unit: None,
                meaning: None,
            }],
            preferred_component: Some(0),
        },
        ["cosine", "sine", "angular-velocity"],
        "Negative angle, angular velocity and torque cost.",
    )
}

pub(super) fn wide_lqg_config(horizon: usize) -> Result<LqgConfig> {
    let input = DMatrix::from_fn(WIDE_STATES, WIDE_ACTIONS, |state, action| {
        if state % WIDE_ACTIONS == action {
            0.6 / (1. + (state / WIDE_ACTIONS) as f64)
        } else {
            0.
        }
    });
    let diagonal = |factor| DMatrix::identity(WIDE_STATES, WIDE_STATES) * factor;
    let config = LqgConfig {
        a: vec![diagonal(0.82); horizon],
        b: vec![input; horizon],
        q: vec![diagonal(0.8); horizon],
        r: vec![DMatrix::identity(WIDE_ACTIONS, WIDE_ACTIONS) * 0.3; horizon],
        terminal: diagonal(1.),
        noise: vec![diagonal(0.08); horizon],
        initial: (0..WIDE_STATES)
            .map(|state| if state % 2 == 0 { 1. } else { -0.5 })
            .collect(),
        action_space: ActionSpace::Box {
            low: vec![-1.5; WIDE_ACTIONS],
            high: vec![1.5; WIDE_ACTIONS],
        },
    };
    config.validate()?;
    Ok(config)
}

pub(super) fn wide_lqg_context(config: &LqgConfig) -> ReferenceContext {
    let horizon = config.horizon();
    ReferenceContext {
        environment: "lqg-wide".into(),
        observation_dim: WIDE_STATES + 1,
        typed_observation_space: None,
        action_space: config.action_space.clone(),
        conventions: ValueConventions {
            dynamics: "Diagonal linear dynamics with alternating control columns and Gaussian process noise".into(),
            observation: "Sixteen physical state coordinates followed by the normalized decision time".into(),
            action: "Two bounded controls".into(),
            reward_units: "negative quadratic cost".into(),
            reward_convention: "Negative state and control costs plus the negative terminal cost at the last decision".into(),
            discount: f64::from(GAMMA),
            horizon: format!("{horizon} decisions"),
            terminal_treatment: "The last decision includes the terminal state cost".into(),
            action_constraints: "Each control lies in [-1.5, 1.5]".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: horizon as u32 },
        environment_config: BTreeMap::from([
            ("horizon".into(), ConfigValue::Integer(horizon as i64)),
            ("state_coordinates".into(), ConfigValue::Integer(WIDE_STATES as i64)),
            ("state_decay".into(), ConfigValue::Real(0.82)),
            ("state_cost".into(), ConfigValue::Real(0.8)),
            ("control_cost".into(), ConfigValue::Real(0.3)),
            ("initial_even".into(), ConfigValue::Real(1.)),
            ("initial_odd".into(), ConfigValue::Real(-0.5)),
            ("process_noise_std".into(), ConfigValue::Real(0.08)),
            ("control_gain".into(), ConfigValue::Text("0.6 / (1 + floor(state index / 2))".into())),
            ("terminal_cost".into(), ConfigValue::Real(1.)),
        ]),
        markov_observation: true,
        decision_time_coordinate: Some(WIDE_STATES),
    }
}

struct ZeroControl(ActionSpace);

impl Actor for ZeroControl {
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
        actions.resize_with(observations.len(), || {
            Action::Continuous(vec![0.; self.0.dimensions()])
        });
        Ok(())
    }
}

pub(super) fn zero_control_benchmark(context: ReferenceContext) -> Result<impl ReferenceProvider> {
    let space = context.action_space.clone();
    space.check(&Action::Continuous(vec![0.; space.dimensions()]), None)?;
    Ok(BenchmarkActor::new(
        ReferenceSpec {
            id: "zero-control".into(),
            context,
            quality: ReferenceQuality {
                kind: ReferenceKind::Heuristic,
                method: "Apply zero control at every decision".into(),
                assumptions: Vec::new(),
            },
            target: ReferenceTarget::BenchmarkPolicy,
            lifetime: ReferenceLifetime::Fixed,
            information: ReferenceInformation::LearnerObservation,
            uncertainty: None,
        },
        ZeroControl(space),
    )?
    .with_evaluation_mode(ActMode::Deterministic))
}

pub(super) fn render_capture(horizon: usize) -> RenderCaptureConfig {
    RenderCaptureConfig {
        trajectories: 1,
        limits: RenderLimits {
            max_frames: horizon + 1,
            max_scene_bytes: 4 * 1024,
            max_primitives: 16,
            max_points: 64,
            max_text_bytes: 0,
            max_trajectory_bytes: 2 * 1024 * 1024,
            max_evaluation_bytes: 2 * 1024 * 1024,
        },
    }
}
pub(super) fn pendulum(run: &Run) -> Result<()> {
    let mut environment = Pendulum::new(SCENARIO_SEED);
    let space = environment.action_space().clone();
    let observations = (SCENARIO_SEED..SCENARIO_SEED + 8)
        .map(|seed| Ok(environment.reset(Some(seed))?.observation))
        .collect::<Result<Vec<_>>>()?;
    let fixed_actions = vec![Action::Continuous(vec![0.]); observations.len()];
    let probes = ProbeSet::new(
        "pendulum-resets",
        ProbePopulation::FixedReference,
        "Eight separately seeded reset observations and zero torque, equally weighted",
        observations,
        Some(fixed_actions),
        space.clone(),
    )?;
    let context = ReferenceContext {
        environment: "pendulum".into(),
        observation_dim: 3,
        typed_observation_space: None,
        action_space: space.clone(),
        conventions: ValueConventions {
            dynamics: "rust-rl Pendulum v1 dynamics".into(),
            observation: "Cosine and sine of the angle and angular velocity".into(),
            action: "Bounded torque".into(),
            reward_units: "negative swing-up cost".into(),
            reward_convention: "Environment reward, larger is better".into(),
            discount: f64::from(GAMMA),
            horizon: format!("{PENDULUM_HORIZON} decisions"),
            terminal_treatment: "Evaluation ends at the explicit decision horizon".into(),
            action_constraints: "Torque lies in [-2, 2]".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite {
            decisions: PENDULUM_HORIZON as u32,
        },
        environment_config: BTreeMap::from([
            (
                "horizon".into(),
                ConfigValue::Integer(PENDULUM_HORIZON as i64),
            ),
            ("decision_seconds".into(), ConfigValue::Real(0.05)),
        ]),
        markov_observation: false,
        decision_time_coordinate: None,
    };
    let mut builder = run
        .builder(&environment, run.config(PENDULUM_HORIZON)?)
        .environment_descriptor(pendulum_descriptor())
        .reference_context(context.clone())
        .metadata(
            "evaluation.scenario_protocol",
            "Gymnasium Pendulum independent uniform angle and angular velocity with a ChaCha8 scenario seed",
        )
        .metadata("evaluation.decision_seconds", "0.05")
        .probe_set(probes);
    let factory = |seed| Ok(Pendulum::new(seed));
    builder = if run.render {
        builder
            .evaluation_factory_with_render(factory, render_capture(PENDULUM_HORIZON))
            .reference_trajectories(1)
    } else {
        builder.evaluation_factory(factory)
    };
    if run.reference {
        builder = builder.reference(zero_control_benchmark(context)?);
    }

    let mut rng = ChaCha8Rng::seed_from_u64(run.stream_seed(1));
    let learner = Sac::new(
        GaussianPolicy::new(3, space, &[12, 12], &mut rng)?,
        QValue::new(3, 1, 1, &[12, 12], &mut rng)?,
        QValue::new(3, 1, 1, &[12, 12], &mut rng)?,
        SacConfig {
            gamma: GAMMA,
            ..Default::default()
        },
        run.stream_seed(2),
    )?;
    run.train_off_policy(
        learner,
        builder,
        Pendulum::v1(run.seed)?,
        OffPolicyConfig {
            exploration: Schedule::Constant(0.),
            ..off_policy_config()
        },
    )
}
