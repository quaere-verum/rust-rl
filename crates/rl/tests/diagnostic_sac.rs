#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::sac::{Sac, SacConfig},
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::Checkpoint,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        reward::Reward,
    },
    diagnostics::{
        evaluation::{
            EvaluationConfig, EvaluationHorizon, SoftActionValueConfig, evaluate_soft_action_value,
        },
        models::{
            InspectionRequest, ModelBundle, ModelKind, ModelLimits, SnapshotBuilder,
            snapshot_configuration,
        },
        policy_cache::{CacheLimits, PolicyCache},
        policy_precompute::{PrecomputeOptions, precompute},
        policy_slice::{PolicySliceSpec, SliceConfig},
        probes::ProbeSet,
        schedule::Interval,
        schema::{
            ActionCondition, PolicyParameters, ProbePopulation, Record, RecordContext, RecordKind,
        },
        session::{DiagnosticConfig, DiagnosticSession},
        storage::{ReadLimits, Recording},
        value::{
            self, EntropyTemperatureSource, EntropyTiming, LogDensityConvention, ReturnTarget,
        },
        visualization::EnvironmentSemantics,
    },
    envs::pendulum::Pendulum,
    nn::{
        critics::scalar::{ActionValue, QValue},
        distributions::Distribution,
        gaussian_policy::GaussianPolicy,
        matrix,
        policy::StochasticPolicy,
        to_vec, vector,
    },
    prelude::{
        Budget, OffPolicyCollector, OffPolicyConfig, OffPolicyTrainer, Schedule, SyncVectorEnv,
    },
};
use std::{cell::RefCell, rc::Rc};

#[test]
fn gaussian_snapshot_preserves_state_dependent_and_transformed_policy_inference() -> Result<()> {
    let mut init = ChaCha8Rng::seed_from_u64(18);
    for space in [
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![4.],
        },
        ActionSpace::Unbounded { dimensions: 1 },
    ] {
        let mut policy = GaussianPolicy::new(1, space.clone(), &[], &mut init)?;
        policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.25, 1.], 1, 2));
        let mut builder =
            SnapshotBuilder::new("sac-k3", RecordContext::new("sac", 3), 1, space.clone())?;
        builder.state_gaussian("policy", &policy)?;
        let chunk = builder.finish()?;
        let Record::ModelSnapshot(snapshot) = &chunk.record else {
            panic!("snapshot expected")
        };
        let config = snapshot_configuration(snapshot, ModelLimits::default())?;
        assert!(matches!(
            config.models["policy"].kind,
            ModelKind::StateGaussianPolicy
        ));
        assert!(
            snapshot_configuration(
                snapshot,
                ModelLimits {
                    max_parameters: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
        let bundle = ModelBundle::load(&chunk, ModelLimits::default())?;
        let observations = vec![
            Observation::new(vec![1.]),
            Observation::new(vec![-1.]),
            Observation::new(vec![30.]),
            Observation::new(vec![-30.]),
        ];
        let expected = policy.distribution(observations.as_slice().into())?;
        for mode in [ActMode::Deterministic, ActMode::Sample] {
            let mut request = InspectionRequest::new("policy", observations.clone());
            request.mode = mode;
            request.seed = 3;
            request
                .layers
                .insert("policy".into(), vec!["layers.0.post".into()]);
            let result = bundle.inspect(&request)?;
            assert_eq!(
                result.actions,
                expected.sample(mode, &mut ChaCha8Rng::seed_from_u64(3))?
            );
            let Record::Policy(record) = &result.policy.record else {
                panic!("policy expected")
            };
            let PolicyParameters::Gaussian {
                mean,
                log_std,
                tanh_bounded,
            } = &record.parameters
            else {
                panic!("Gaussian parameters expected")
            };
            assert_eq!(
                result.policy.arrays[mean].as_f32()?,
                [0.25, -0.25, 7.5, -7.5]
            );
            assert_eq!(result.policy.arrays[log_std].as_f32()?, [1., -1., 2., -20.]);
            assert_eq!(*tanh_bounded, matches!(space, ActionSpace::Box { .. }));
            assert_eq!(
                result.activations["policy"].arrays["layers.0.post"].as_f32()?,
                [0.25, 1., -0.25, -1., 7.5, 30., -7.5, -30.]
            );
        }

        let observations = &observations[..2];
        let restored = bundle.policy_distribution("policy", observations)?;
        let Distribution::Gaussian { bounds, .. } = &restored else {
            panic!("Gaussian distribution expected")
        };
        let latent = [0.5f32, -0.25];
        let actions: Vec<_> = latent
            .iter()
            .map(|&z| {
                Action::Continuous(vec![if bounds.is_some() {
                    1. + 3. * z.tanh()
                } else {
                    z
                }])
            })
            .collect();
        let latents = latent.map(|z| bounds.as_ref().map(|_| vec![z]));
        let actual = to_vec(restored.log_prob(&actions, &latents)?)?;
        let original = to_vec(
            policy
                .distribution(observations.into())?
                .log_prob(&actions, &latents)?,
        )?;
        assert_eq!(actual, original);
        for ((&actual, (&z, mean)), log_std) in actual
            .iter()
            .zip(latent.iter().zip([0.25, -0.25]))
            .zip([1f64, -1.])
        {
            let z = f64::from(z);
            let latent_log_density = -0.5 * ((z - mean) / log_std.exp()).powi(2)
                - log_std
                - 0.5 * std::f64::consts::TAU.ln();
            let jacobian = if bounds.is_some() {
                (3. * (1. - z.tanh().powi(2))).ln()
            } else {
                0.
            };
            assert!((f64::from(actual) - (latent_log_density - jacobian)).abs() < 2e-6);
        }
    }
    Ok(())
}

#[test]
fn sac_session_preserves_learning_and_reopens_both_soft_critic_roles() -> Result<()> {
    let env = Pendulum::new(4);
    let action_space = env.action_space().clone();
    let config = SacConfig {
        gamma: 0.5,
        target_interval: 20,
        ..Default::default()
    };
    let learner = || {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        Sac::new(
            GaussianPolicy::new(3, action_space.clone(), &[4], &mut rng)?,
            QValue::new(3, 1, 1, &[4], &mut rng)?,
            QValue::new(3, 1, 1, &[4], &mut rng)?,
            config.clone(),
            8,
        )
    };
    let trainer = || {
        OffPolicyTrainer::new(
            OffPolicyCollector::new(SyncVectorEnv::new(vec![Pendulum::new(4)])?, 5)?,
            OffPolicyConfig {
                replay_capacity: 32,
                batch_size: 2,
                warmup_transitions: 2,
                gamma: config.gamma,
                updates_per_transition: 0.5,
                exploration: Schedule::Constant(0.),
                ..Default::default()
            },
            6,
        )
    };
    let mut observed = learner()?;
    let mut baseline = learner()?;
    baseline.restore(&observed.checkpoint()?)?;
    let initial_alpha = observed.alpha();
    let observations = vec![
        Observation::new(vec![1., 0., 0.]),
        Observation::new(vec![0., 1., 0.5]),
    ];
    let probes = ProbeSet::new(
        "sac-fixed",
        ProbePopulation::FixedReference,
        "two fixed Pendulum policy and critic inputs",
        observations.clone(),
        None,
        action_space,
    )?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("sac-pendulum");
    let mut settings = DiagnosticConfig::standard();
    settings.evaluation_horizon = EvaluationHorizon::FiniteDecisions(4);
    settings.evaluation.max_steps = 4;
    settings.evaluation.seeds = vec![11, 12];
    let every = Some(Interval { first: 1, every: 1 });
    settings.schedule.probes = every;
    settings.schedule.representations = every;
    settings.schedule.snapshots = every;
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .evaluation_factory(|seed| Ok(Pendulum::new(seed)))
        .probe_set(probes)
        .config(settings)
        .build(&observed)?;
    let mut factual_trainer = trainer()?;
    let mut recorded_trainer = factual_trainer.clone();
    let factual = factual_trainer.train(&mut baseline, Budget::vector_steps(8), None)?;
    let recorded = recorded_trainer.train_observed(
        &mut observed,
        Budget::vector_steps(8),
        None,
        &mut session,
    )?;
    session.finish(&observed)?;
    let revision = observed.updates();
    assert!(revision > 0);
    assert_eq!(recorded.counters, factual.counters);
    assert_eq!(
        serde_json::to_value(&recorded_trainer)?,
        serde_json::to_value(&factual_trainer)?
    );
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&baseline.checkpoint()?)
    );

    let run = Recording::open(&path, ReadLimits::default())?;
    assert!(run.complete());
    let target = ReturnTarget::SoftActionValue {
        alpha: observed.alpha(),
        temperature_source: EntropyTemperatureSource::Snapshot,
        entropy_timing: EntropyTiming::ReplayBootstrap { n_step: 1 },
        log_density: LogDensityConvention::PolicySampleInActionCoordinates,
    };
    let policy = observed.policy();
    let expected_actions = policy
        .distribution(observations.as_slice().into())?
        .sample(ActMode::Sample, &mut ChaCha8Rng::seed_from_u64(19))?;
    let mut online_sequence = None;
    for (role, pair) in [
        ("online", [observed.critic(), observed.second_critic()]),
        (
            "target",
            [observed.target_critic(), observed.target_second_critic()],
        ),
    ] {
        let (index, entry) = run
            .entries()
            .iter()
            .enumerate()
            .find(|(_, entry)| {
                entry.revision == revision
                    && entry.model_role == role
                    && entry.kind == RecordKind::ModelSnapshot
            })
            .expect("snapshot of final revision and role");
        if role == "online" {
            online_sequence = Some(entry.sequence);
        }
        let snapshot = run.load(index)?;
        assert_eq!(value::return_target(&snapshot)?, Some(target));
        let declaration = value::declaration(&snapshot)?.expect("twin critic declaration");
        for (i, name) in ["q1", "q2"].into_iter().enumerate() {
            let subject = &declaration.critics[name];
            assert_eq!(subject.twin.as_ref().unwrap().index, i);
            let actor = subject.actor.as_ref().expect("supplied by online policy");
            assert_eq!(actor.identity, format!("sac-pendulum@{revision}"));
            assert!(actor.stochastic);
            assert_eq!(subject.continuation.as_ref(), Some(actor));
        }
        let bundle = ModelBundle::load(&snapshot, ModelLimits::default())?;
        let mut request = InspectionRequest::new("policy", observations.clone());
        request.critics = vec!["q1".into(), "q2".into()];
        request.mode = ActMode::Sample;
        request.seed = 19;
        let result = bundle.inspect(&request)?;
        assert_eq!(result.actions, expected_actions);
        let actions = matrix(
            result
                .actions
                .iter()
                .map(|action| match action {
                    Action::Continuous(action) => action[0],
                    _ => panic!("continuous action expected"),
                })
                .collect(),
            observations.len(),
            1,
        );
        for (name, critic) in ["q1", "q2"].into_iter().zip(pair) {
            assert_eq!(
                result.critics[name].arrays["values"].as_f32()?,
                to_vec(critic.values(observations.as_slice().into(), actions.clone())?)?
            );
        }

        let predicted = run
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.revision == revision && e.model_role == role && e.kind == RecordKind::Critic
            })
            .map(|(index, _)| run.load(index))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(predicted.len(), 2);
        let recorded_actions = |index: usize| -> Result<&[f32]> {
            let chunk = &predicted[index];
            let Record::Critic(record) = &chunk.record else {
                panic!("critic expected")
            };
            assert_eq!(record.action_condition, ActionCondition::CurrentPolicy);
            assert_eq!(value::return_target(chunk)?, Some(target));
            chunk.arrays[record
                .actions
                .as_ref()
                .expect("sampled critic input actions")]
            .as_f32()
        };
        assert_eq!(recorded_actions(0)?, recorded_actions(1)?);
    }

    let alpha_metric = run
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, e)| e.revision == revision && e.kind == RecordKind::Metrics)
        .find_map(|(i, _)| match run.load(i).unwrap().record {
            Record::Metrics(record) => record.values.get("alpha_next").copied(),
            _ => None,
        })
        .expect("training temperature at completed revision");
    assert_eq!(alpha_metric, f64::from(observed.alpha()));
    let manifest_configuration: serde_json::Value =
        serde_json::from_str(&run.manifest().metadata["learner.configuration"])?;
    assert_eq!(
        manifest_configuration["alpha_at_starting_revision"].as_f64(),
        Some(f64::from(initial_alpha))
    );

    let mut slice = SliceConfig::new(
        EnvironmentSemantics::new(run.manifest()),
        &observations[0],
        &[],
    )?;
    slice.resolution = [3, 2];
    let spec = PolicySliceSpec::from_config("online", "policy", &slice);
    let cache =
        PolicyCache::open_or_create(dir.path().join("cache"), &run, CacheLimits::default())?;
    let summary = precompute(
        &run,
        &cache,
        &[spec],
        &PrecomputeOptions {
            sequences: Some(vec![online_sequence.expect("online snapshot sequence")]),
            ..Default::default()
        },
        |_| {},
    )?;
    assert_eq!(
        (summary.computed, summary.failed, summary.evaluated_rows),
        (1, 0, 6)
    );
    Ok(())
}

#[test]
fn soft_rollout_scores_only_successor_actions_in_physical_action_coordinates() -> Result<()> {
    struct TwoActions {
        space: ActionSpace,
        early_end: bool,
        actions: Rc<RefCell<Vec<Action>>>,
        decision: usize,
    }
    impl Env for TwoActions {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }
        fn observation_space(&self) -> ObservationSpace {
            ObservationSpace::Dense(1)
        }
        fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
            self.decision = 0;
            Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
        }
        fn step(&mut self, action: &Action) -> Result<Step> {
            self.actions.borrow_mut().push(action.clone());
            let reward = if self.decision == 0 {
                Reward::from_primitives(vec![2., 4.])?
            } else {
                6.0f32.into()
            };
            self.decision += 1;
            Ok(Step {
                observation: Observation::new(vec![self.decision as f32]),
                reward,
                terminated: self.early_end || self.decision == 2,
                truncated: false,
            })
        }
    }
    let space = ActionSpace::Box {
        low: vec![-2.],
        high: vec![4.],
    };
    let mut model = GaussianPolicy::new(1, space.clone(), &[], &mut ChaCha8Rng::seed_from_u64(2))?;
    model.network.layers[0].weight = Param::from_tensor(matrix(vec![0., 0.], 1, 2));
    model.network.layers[0].bias = Some(Param::from_tensor(vector(&[0., -1.])));
    let settings = SoftActionValueConfig {
        alpha: 0.75,
        evaluation: EvaluationConfig {
            seeds: vec![5, 6],
            policy_seed: 8,
            discount: 0.5,
            max_steps: 2,
            trajectories: 0,
            mode: ActMode::Sample,
            policy_id: "soft-fixture@0".into(),
            ..Default::default()
        },
    };
    let observation = Observation::new(vec![0.]);
    let first = Action::Continuous(vec![2.]);
    for early_end in [false, true] {
        let actions = Rc::new(RefCell::new(Vec::new()));
        let result = evaluate_soft_action_value(
            |seed| {
                let mut env = TwoActions {
                    space: space.clone(),
                    early_end,
                    actions: actions.clone(),
                    decision: 0,
                };
                let initial = env.reset(Some(seed))?.observation;
                Ok((env, initial))
            },
            &observation,
            &first,
            &model,
            &settings,
            "fixed reset with independent future seeds",
        )?;
        assert_eq!(
            result.target,
            ReturnTarget::SoftActionValue {
                alpha: 0.75,
                temperature_source: EntropyTemperatureSource::IndependentEvaluation,
                entropy_timing: EntropyTiming::ReplayBootstrap { n_step: 1 },
                log_density: LogDensityConvention::PolicySampleInActionCoordinates,
            }
        );
        assert!(result.metadata.complete_returns);
        assert_eq!(result.metadata.continuation_policy, "soft-fixture@0");
        assert_eq!(result.initial_action, first);
        let actual = actions.borrow();
        if early_end {
            assert_eq!(*actual, [first.clone(), first.clone()]);
            assert_eq!(result.returns, [4., 4.]);
        } else {
            for (actions, value) in actual.chunks_exact(2).zip(result.returns) {
                assert_eq!(actions[0], first);
                let Action::Continuous(a) = &actions[1] else {
                    panic!("continuous successor action expected")
                };
                let normalized = (f64::from(a[0]) - 1.) / 3.;
                let latent = normalized.atanh();
                let log_density = -0.5 * (latent * std::f64::consts::E).powi(2) + 1.
                    - 0.5 * std::f64::consts::TAU.ln()
                    - (3. * (1. - normalized * normalized)).ln();
                let expected = 4. + 0.25 * (6. - 0.75 * log_density);
                assert!((value - expected).abs() < 2e-6, "{value} != {expected}");
            }
        }
    }
    Ok(())
}
