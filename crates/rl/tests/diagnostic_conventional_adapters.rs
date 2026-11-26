#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        models::{InspectionRequest, ModelBundle, ModelLimits},
        probes::ProbeSet,
        reference::GradientMethod,
        schedule::Interval,
        schema::{ProbePopulation, Record, RecordContext, RecordKind},
        session::{DiagnosticConfig, DiagnosticLearner, DiagnosticSession},
        storage::{ReadLimits, Recording},
        value,
    },
    nn::critics::scalar::{ActionValue, StateValue},
    prelude::*,
};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct TestEnv {
    space: ActionSpace,
    state: f32,
    step: usize,
}
impl TestEnv {
    fn new(space: ActionSpace) -> Self {
        Self {
            space,
            state: 0.,
            step: 0,
        }
    }

    fn observation(&self) -> Observation {
        Observation::new(vec![self.state, self.step as f32 / 4.])
    }
}

impl Env for TestEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(2)
    }

    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        self.state = if seed.unwrap_or_default().is_multiple_of(2) {
            -0.5
        } else {
            0.5
        };
        self.step = 0;
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.space.check(action, None)?;
        self.state += match action {
            Action::Discrete(0) => -0.25,
            Action::Discrete(_) => 0.25,
            Action::Continuous(action) => action[0] / 4.,
            Action::Conditional(_) => unreachable!(),
        };
        self.step += 1;
        Ok(Step {
            observation: self.observation(),
            reward: (1. - self.state.powi(2)).into(),
            terminated: self.step == 4,
            truncated: false,
        })
    }
}

fn continuous_space() -> ActionSpace {
    ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    }
}

fn spaces() -> [ActionSpace; 2] {
    [ActionSpace::Discrete { count: 2 }, continuous_space()]
}

fn observations() -> Vec<Observation> {
    vec![
        Observation::new(vec![-0.5, 0.]),
        Observation::new(vec![0.25, 0.5]),
    ]
}

fn models(space: &ActionSpace, rng: &mut ChaCha8Rng) -> Result<ActorCritic<Policy, Value>> {
    Ok(ActorCritic {
        policy: Policy::new(2, space.clone(), &[4], rng)?,
        value: Value::new(2, &[4], rng)?,
    })
}

fn probe_set(space: ActionSpace, fixed_actions: Option<Vec<Action>>) -> Result<ProbeSet> {
    ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "specified test inputs",
        observations(),
        fixed_actions,
        space,
    )
}

fn config() -> DiagnosticConfig {
    let mut config = DiagnosticConfig::detailed();
    let interval = Some(Interval {
        first: 99,
        every: 99,
    });
    config.schedule.probes = interval;
    config.schedule.representations = interval;
    config.schedule.snapshots = interval;
    config.schedule.evaluation = interval;
    config.visited_probe_rows = 0;
    config.evaluation.seeds = vec![11];
    config.evaluation.max_steps = 4;
    config.evaluation.trajectories = 1;
    config
}

fn metric_names(recording: &Recording) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for (index, entry) in recording.entries().iter().enumerate() {
        if entry.kind == RecordKind::Metrics
            && let Record::Metrics(metrics) = recording.load(index)?.record
        {
            names.extend(metrics.values.into_keys());
        }
    }
    Ok(names)
}

fn bundle(recording: &Recording, revision: u64, role: &str) -> Result<ModelBundle> {
    let index = recording
        .entries()
        .iter()
        .position(|entry| {
            entry.kind == RecordKind::ModelSnapshot
                && entry.revision == revision
                && entry.model_role == role
        })
        .expect("snapshot at requested revision");
    ModelBundle::load(&recording.load(index)?, ModelLimits::default())
}

fn assert_recording(
    recording: &Recording,
    final_revision: u64,
    expected_metrics: &[&str],
) -> Result<()> {
    assert!(recording.complete());
    let trajectory_revisions = recording
        .entries()
        .iter()
        .filter(|entry| entry.kind == RecordKind::Trajectory)
        .map(|entry| entry.revision)
        .collect::<Vec<_>>();
    assert_eq!(trajectory_revisions, [0, final_revision]);
    let metrics = metric_names(recording)?;
    for name in expected_metrics {
        assert!(
            metrics.iter().any(|reported| reported == name),
            "missing reported metric {name}"
        );
    }
    Ok(())
}

fn on_policy_session<L>(
    space: ActionSpace,
    build: impl Fn(&ActionSpace, &mut ChaCha8Rng) -> Result<L>,
    baseline: impl Fn(&L) -> Option<Value>,
    config_train: OnPolicyConfig,
    expected_metrics: &[&str],
) -> Result<()>
where
    L: OnPolicyLearner<Batch = OnPolicyBatch> + DiagnosticLearner + Checkpoint,
{
    let env = TestEnv::new(space.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(49);
    let mut observed = build(&space, &mut rng)?;
    let mut plain = build(&space, &mut rng)?;
    plain.restore(&observed.checkpoint()?)?;
    let mut trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()])?, 51)?,
        config_train,
    )?;
    let mut plain_trainer = trainer.clone();
    let path = tempfile::tempdir()?;
    let dir = path.path().join("recording");
    let mut session = DiagnosticSession::builder(&dir)
        .environment(&env)
        .evaluation_factory(|_| Ok(TestEnv::new(space.clone())))
        .probe_set(probe_set(space.clone(), None)?)
        .config(config())
        .build(&observed)?;

    let mut checks = Vec::new();
    for initial in [true, false] {
        if !initial {
            trainer.train_observed(&mut observed, Budget::vector_steps(8), None, &mut session)?;
        }
        let actions =
            observed
                .diagnostic_actor()?
                .act(&observations(), ActMode::Deterministic, &mut rng)?;
        let values = baseline(&observed)
            .map(|value| rl::nn::to_vec(value.values(observations().as_slice().into())?))
            .transpose()?;
        checks.push((observed.diagnostic_revision(), actions, values));
    }
    session.finish(&observed)?;
    plain_trainer.train(&mut plain, Budget::vector_steps(8), None)?;
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&plain.checkpoint()?)
    );
    assert_eq!(
        serde_json::to_value(&trainer)?,
        serde_json::to_value(&plain_trainer)?
    );

    let recording = Recording::open(&dir, ReadLimits::default())?;
    assert_recording(&recording, observed.diagnostic_revision(), expected_metrics)?;
    for (revision, actions, values) in checks {
        let bundle = bundle(&recording, revision, "online")?;
        let mut request = InspectionRequest::new("policy", observations());
        if values.is_some() {
            request.critics.push("value".into());
        }
        let restored = bundle.inspect(&request)?;
        assert_eq!(restored.actions, actions);
        assert_eq!(
            bundle.configuration().models.contains_key("value"),
            values.is_some()
        );
        let critic_entries = recording
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.kind == RecordKind::Critic && e.revision == revision)
            .collect::<Vec<_>>();
        if let Some(values) = values {
            assert_eq!(restored.critics["value"].arrays["values"].as_f32()?, values);
            assert_eq!(critic_entries.len(), 1);
            let chunk = recording.load(critic_entries[0].0)?;
            let declaration = value::declaration(&chunk)?.expect("state-value semantics");
            let critic = &declaration.critics["value"];
            assert_eq!(critic.learned_baseline, Some(true));
            let actor = critic.actor.as_ref().expect("deployed policy");
            let continuation = critic.continuation.as_ref().expect("rollout behavior");
            assert!(actor.stochastic && continuation.stochastic);
            assert_ne!(actor.identity, continuation.identity);
            assert_eq!(chunk.arrays["values"].as_f32()?, values);
        } else {
            assert!(critic_entries.is_empty());
        }
    }
    Ok(())
}

#[test]
fn reinforce_baseline_only_appears_after_training_when_enabled() -> Result<()> {
    for space in spaces() {
        for enabled in [false, true] {
            let metrics: &[&str] = if enabled {
                &["policy_loss", "value_loss"]
            } else {
                &["policy_loss"]
            };
            on_policy_session(
                space.clone(),
                |space, rng| {
                    Reinforce::new(
                        models(space, rng)?,
                        ReinforceConfig {
                            baseline: enabled,
                            ..Default::default()
                        },
                        53,
                    )
                },
                |learner| (enabled && learner.updates() > 0).then(|| learner.value()),
                OnPolicyConfig {
                    episode_guard: Some(4),
                    gae_lambda: None,
                    ..Default::default()
                },
                metrics,
            )?;
        }
    }
    Ok(())
}

#[test]
fn vmpo_records_state_value_and_dual_training_metrics() -> Result<()> {
    for space in spaces() {
        on_policy_session(
            space,
            |space, rng| {
                Vmpo::new(
                    models(space, rng)?,
                    VmpoConfig {
                        epochs: 1,
                        minibatch_size: 4,
                        ..Default::default()
                    },
                    55,
                )
            },
            |learner| Some(learner.value()),
            OnPolicyConfig {
                rollout_steps: 4,
                ..Default::default()
            },
            &["policy_loss", "value_loss", "dual_loss", "kl_new_behavior"],
        )?;
    }
    Ok(())
}

#[test]
fn mdpo_records_state_value_and_mirror_update_metrics() -> Result<()> {
    for space in spaces() {
        on_policy_session(
            space,
            |space, rng| {
                Mdpo::new(
                    models(space, rng)?,
                    MdpoConfig {
                        inner_steps: 1,
                        value_epochs: 1,
                        value_minibatch_size: 4,
                        ..Default::default()
                    },
                    57,
                )
            },
            |learner| Some(learner.value()),
            OnPolicyConfig {
                rollout_steps: 4,
                ..Default::default()
            },
            &[
                "policy_loss",
                "value_loss",
                "mirror_step",
                "actor_gradient_norm",
                "critic_gradient_norm",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn ddpg_records_exact_scalar_targets_and_preserves_training_state() -> Result<()> {
    let space = continuous_space();
    let env = TestEnv::new(space.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(65);
    let build = |rng: &mut ChaCha8Rng| {
        Ddpg::new(
            Deterministic::new(2, space.clone(), &[4], rng)?,
            QValue::new(2, 1, 1, &[], rng)?,
            DdpgConfig {
                target_noise: 10.,
                policy_delay: 4,
                ..Default::default()
            },
            67,
        )
    };
    let mut observed = build(&mut rng)?;
    let mut plain = build(&mut rng)?;
    plain.restore(&observed.checkpoint()?)?;
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()])?, 69)?,
        OffPolicyConfig {
            batch_size: 2,
            replay_capacity: 16,
            warmup_transitions: 0,
            ..Default::default()
        },
        71,
    )?;
    let mut plain_trainer = trainer.clone();
    let path = tempfile::tempdir()?;
    let dir = path.path().join("ddpg");
    let fixed_actions = vec![
        Action::Continuous(vec![0.25]),
        Action::Continuous(vec![-0.5]),
    ];
    let mut session = DiagnosticSession::builder(&dir)
        .environment(&env)
        .evaluation_factory(|_| Ok(env.clone()))
        .probe_set(probe_set(space.clone(), Some(fixed_actions.clone()))?)
        .config(config())
        .build(&observed)?;
    trainer.train_observed(&mut observed, Budget::vector_steps(8), None, &mut session)?;
    session.finish(&observed)?;
    plain_trainer.train(&mut plain, Budget::vector_steps(8), None)?;
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&plain.checkpoint()?)
    );
    assert_eq!(
        serde_json::to_value(&trainer)?,
        serde_json::to_value(&plain_trainer)?
    );

    let revision = observed.updates();
    let context = RecordContext::new("confirmation", revision);
    let (actor, continuation) = observed
        .diagnostic_conditional_actor(&context)?
        .expect("deterministic target");
    assert!(!continuation.stochastic);
    assert_eq!(
        continuation.identity,
        format!("confirmation@{revision}/target-policy")
    );
    let act = |rng: &mut ChaCha8Rng| actor.act(&observations(), ActMode::Sample, rng);
    assert_eq!(
        act(&mut ChaCha8Rng::seed_from_u64(73))?,
        act(&mut ChaCha8Rng::seed_from_u64(75))?
    );
    assert_eq!(
        observed.diagnostic_info()?.configuration["target_noise"],
        0.
    );
    assert_eq!(observed.diagnostic_info()?.configuration["policy_delay"], 1);
    let gradient = observed
        .diagnostic_action_gradient("q", &observations()[0], &fixed_actions[0])?
        .expect("scalar gradient");
    assert!(matches!(gradient.method, GradientMethod::Analytic));
    let weights = rl::nn::to_vec(observed.critic().network.layers[0].weight.val())?;
    assert_eq!(gradient.values, [f64::from(weights[2])]);

    let recording = Recording::open(&dir, ReadLimits::default())?;
    assert_recording(&recording, revision, &["critic_loss", "actor_loss"])?;
    for (role, name, critic) in [
        ("online", "policy", observed.critic()),
        ("target", "target_policy", observed.target_critic()),
    ] {
        let bundle = bundle(&recording, revision, role)?;
        let mut request = InspectionRequest::new(name, observations());
        request.critics.push("q".into());
        request.fixed_actions = Some(fixed_actions.clone());
        let restored = bundle.inspect(&request)?;
        let expected = critic.values(
            observations().as_slice().into(),
            rl::nn::matrix(vec![0.25, -0.5], 2, 1),
        )?;
        assert_eq!(
            restored.critics["q"].arrays["values"].as_f32()?,
            rl::nn::to_vec(expected)?
        );
        let index = recording
            .entries()
            .iter()
            .position(|entry| {
                entry.kind == RecordKind::Critic
                    && entry.revision == revision
                    && entry.model_role == role
            })
            .expect("scalar Q row at final revision");
        let declaration =
            value::declaration(&recording.load(index)?)?.expect("scalar Q provenance");
        let critic = &declaration.critics["q"];
        assert!(critic.twin.is_none());
        let continuation = critic.continuation.as_ref().unwrap();
        assert!(!continuation.stochastic);
        assert!(continuation.identity.ends_with("/target-policy"));
    }
    Ok(())
}

#[test]
fn ddpg_calibrates_scalar_values_and_derivatives_against_its_frozen_target() -> Result<()> {
    use burn::module::Param;
    use rl::diagnostics::reference::{
        ActionGradient, ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
        ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
        ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
    };
    use std::collections::BTreeMap;

    struct ExactZeroContinuation(ReferenceSpec);

    impl ReferenceProvider for ExactZeroContinuation {
        fn specification(&self) -> &ReferenceSpec {
            &self.0
        }

        fn action_value(
            &self,
            checkpoint: &ReferenceCheckpoint<'_>,
            observation: &Observation,
            action: &Action,
        ) -> Result<Option<f64>> {
            assert!(checkpoint.policy_id.ends_with("/target-policy"));
            assert_eq!(checkpoint.mode, ActMode::Deterministic);
            assert_eq!(
                checkpoint.policy.act(
                    std::slice::from_ref(observation),
                    checkpoint.mode,
                    &mut ChaCha8Rng::seed_from_u64(0),
                )?,
                [Action::Continuous(vec![0.])]
            );
            let (remaining, next) = fixture_target(observation, action);
            Ok(Some(remaining * (1. - next.powi(2))))
        }

        fn action_gradient(
            &self,
            _: &ReferenceCheckpoint<'_>,
            observation: &Observation,
            action: &Action,
        ) -> Result<Option<ActionGradient>> {
            let (remaining, next) = fixture_target(observation, action);
            Ok(Some(ActionGradient {
                values: vec![-0.5 * remaining * next],
                method: GradientMethod::Analytic,
            }))
        }
    }

    fn fixture_target(observation: &Observation, action: &Action) -> (f64, f64) {
        let [state, time] = observation.dense_values() else {
            unreachable!()
        };
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        (
            4. * (1. - f64::from(*time)),
            f64::from(*state) + f64::from(action[0]) / 4.,
        )
    }

    let space = continuous_space();
    let env = TestEnv::new(space.clone());
    let context = ReferenceContext {
        environment: "four-decision scalar fixture".into(),
        observation_dim: 2,
        typed_observation_space: None,
        action_space: space.clone(),
        conventions: value::ValueConventions {
            dynamics: "state moves by a quarter of the action".into(),
            observation: "state and fraction of decision horizon elapsed".into(),
            action: "signed scalar control".into(),
            reward_units: "fixture reward".into(),
            reward_convention: "one minus the squared successor state, larger is better".into(),
            discount: 1.,
            horizon: "four decisions".into(),
            terminal_treatment: "zero continuation after four decisions".into(),
            action_constraints: "bounded to [-1, 1]".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 4 },
        environment_config: BTreeMap::from([("horizon".into(), ConfigValue::Integer(4))]),
        markov_observation: true,
        decision_time_coordinate: Some(1),
    };
    let reference = ExactZeroContinuation(ReferenceSpec {
        id: "analytic-zero-continuation".into(),
        context: context.clone(),
        quality: ReferenceQuality {
            kind: ReferenceKind::Analytical,
            method: "sum fixture rewards after the fixed action followed by zero control".into(),
            assumptions: vec!["frozen target always emits zero control".into()],
        },
        target: ReferenceTarget::CriticContinuation,
        lifetime: ReferenceLifetime::CheckpointDependent,
        information: ReferenceInformation::LearnerObservation,
        uncertainty: None,
    });
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let mut policy = Deterministic::new(2, space.clone(), &[], &mut rng)?;
    policy.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![0., 0.], 2, 1));
    let mut critic = QValue::new(2, 1, 1, &[], &mut rng)?;
    critic.network.layers[0].weight = Param::from_tensor(rl::nn::matrix(vec![0., 0., 2.], 3, 1));
    let learner = Ddpg::new(policy, critic, Default::default(), 93)?;
    let tmp = tempfile::tempdir()?;
    let dir = tmp.path().join("calibration");
    let mut session = DiagnosticSession::builder(&dir)
        .environment(&env)
        .reference_context(context)
        .reference(reference)
        .probe_set(probe_set(
            space,
            Some(vec![
                Action::Continuous(vec![0.25]),
                Action::Continuous(vec![-0.5]),
            ]),
        )?)
        .config(config())
        .build(&learner)?;
    session.finish(&learner)?;

    let recording = Recording::open(&dir, ReadLimits::default())?;
    let summaries = recording
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.kind == RecordKind::Metrics)
        .map(|(i, _)| recording.load(i))
        .collect::<Result<Vec<_>>>()?;
    let summary = summaries
        .iter()
        .find(|chunk| {
            chunk
                .arrays
                .contains_key("reference.critic.q.pointwise_error")
        })
        .expect("compatible DDPG reference calibration");
    assert_eq!(
        summary.arrays["reference.critic.q.reference_value"].as_f64()?,
        [3.234375, 1.96875]
    );
    assert_eq!(
        summary.arrays["reference.critic.q.pointwise_error"].as_f64()?,
        [-2.734375, -2.96875]
    );
    assert_eq!(
        summary.arrays["reference.critic.q.action_gradient_vector_error"].as_f64()?,
        [1.125, 2.125]
    );
    Ok(())
}
