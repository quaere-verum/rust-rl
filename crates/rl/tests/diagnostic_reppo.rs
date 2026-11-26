#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::reppo::{Reppo, ReppoConfig},
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{
        models::{
            CriticInspectionRequest, InspectionRequest, ModelBundle, ModelLimits, SnapshotBuilder,
        },
        probes::ProbeSet,
        schema::{CriticRepresentation, ProbePopulation, Record, RecordContext},
        session::{CaptureRequest, DiagnosticLearner},
    },
    nn::{
        critics::{
            categorical::{CategoricalQ, ReppoCritic},
            scalar::ActionValue,
        },
        matrix,
        policy::Policy,
        to_vec,
    },
    objectives::hl_gauss::HlGaussConfig,
};

fn near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 3e-6, "actual {a} differs from expected {b}");
    }
}

#[test]
fn historical_categorical_critic_matches_the_frozen_learner_at_exact_legal_actions() {
    let fixtures = [
        (
            ActionSpace::Discrete { count: 3 },
            vec![
                Observation::dense(vec![0.3, -0.7], Some(vec![false, true, true])),
                Observation::dense(vec![0.9, 0.1], Some(vec![true, false, false])),
            ],
            vec![Action::Discrete(2), Action::Discrete(0)],
            vec![0., 0., 1., 1., 0., 0.],
            Action::Discrete(0),
        ),
        (
            ActionSpace::Box {
                low: vec![-1.5, 0.2],
                high: vec![3., 2.0],
            },
            vec![
                Observation::new(vec![0.3, -0.7]),
                Observation::new(vec![0.9, 0.1]),
            ],
            vec![
                Action::Continuous(vec![2.1, 0.3]),
                Action::Continuous(vec![-1.3, 1.9]),
            ],
            vec![2.1, 0.3, -1.3, 1.9],
            Action::Continuous(vec![3.1, 0.3]),
        ),
    ];
    for (space, observations, actions, encoding, illegal) in fixtures {
        let support = HlGaussConfig {
            min: -3.,
            max: 7.,
            bins: 11,
            sigma_ratio: 0.65,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(51);
        let learner = Reppo::new(
            Policy::new(2, space.clone(), &[5], &mut rng).unwrap(),
            CategoricalQ::new(2, space.clone(), support.bins, &[6, 4], &mut rng).unwrap(),
            ReppoConfig {
                hl_gauss: support.clone(),
                auxiliary_coefficient: 0.5,
                ..Default::default()
            },
            99,
        )
        .unwrap();
        let probes = ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "matched queries",
            observations.clone(),
            Some(actions.clone()),
            space.clone(),
        )
        .unwrap();
        let recorded = learner
            .diagnostic_records(&CaptureRequest {
                context: &RecordContext::new("reppo", 0),
                probes: &[probes],
                predictions: true,
                representations: false,
                snapshots: false,
                max_activation_values: 0,
                max_chunk_bytes: 128,
                conventions: None,
                evaluation_mode: rl::core::actor::ActMode::Sample,
                policy_seed: 8,
            })
            .unwrap();
        assert!(
            recorded
                .iter()
                .any(|c| matches!(c.record, Record::Policy(_)))
        );
        assert!(
            !recorded
                .iter()
                .any(|c| matches!(c.record, Record::Critic(_)))
        );
        assert!(
            recorded
                .iter()
                .any(|c| matches!(&c.record, Record::Event(event)
            if event.category == "capture_admission"))
        );

        let policy = learner.policy();
        let critic = learner.critic();
        let encoded = matrix(encoding, observations.len(), critic.action_dim());
        let expected_logits = critic
            .values(observations.as_slice().into(), encoded.clone())
            .unwrap();
        let expected_values = to_vec(support.decode(expected_logits.clone()).unwrap()).unwrap();
        let expected_features = to_vec(
            critic
                .representation(observations.as_slice().into(), encoded.clone())
                .unwrap(),
        )
        .unwrap();
        let expected_predictor = to_vec(
            critic
                .predict_representation(observations.as_slice().into(), encoded)
                .unwrap(),
        )
        .unwrap();
        let mut builder =
            SnapshotBuilder::new("k0", RecordContext::new("reppo", 0), 2, space).unwrap();
        builder.policy("policy", &policy).unwrap();
        builder
            .categorical("soft_q", &critic, &support, true)
            .unwrap();
        let snapshot = builder.finish().unwrap();
        assert!(
            ModelBundle::load(
                &snapshot,
                ModelLimits {
                    max_parameters: 64,
                    ..ModelLimits::default()
                },
            )
            .is_err()
        );
        let bundle = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
        let request = CriticInspectionRequest {
            critics: vec!["soft_q".into()],
            observations: observations.clone(),
            fixed_actions: Some(actions.clone()),
        };
        let result = bundle.inspect_critics(&request).unwrap();
        if observations[0].dense_mask().is_some() {
            let Record::ProbeSet(probes) = &result.probes.record else {
                panic!()
            };
            let masks = &result.probes.arrays[probes.action_masks.as_ref().unwrap()];
            assert_eq!(masks.shape(), [2, 3]);
            assert_eq!(masks.as_u8().unwrap(), &[0, 1, 1, 1, 0, 0]);
        }
        let chunk = &result.critics["soft_q"];
        let Record::Critic(record) = &chunk.record else {
            panic!()
        };
        let CriticRepresentation::CategoricalRegression {
            logits,
            weights,
            centers,
            values,
        } = &record.representation
        else {
            panic!()
        };
        assert_eq!(chunk.arrays[logits].shape(), [2, 11]);
        near(
            chunk.arrays[logits].as_f32().unwrap(),
            &to_vec(expected_logits).unwrap(),
        );
        near(chunk.arrays[values].as_f32().unwrap(), &expected_values);
        let support = chunk.arrays[centers].as_f32().unwrap();
        assert_eq!(support, &[-3., -2., -1., 0., 1., 2., 3., 4., 5., 6., 7.]);
        for (row, expected) in chunk.arrays[weights]
            .as_f32()
            .unwrap()
            .chunks_exact(11)
            .zip(expected_values)
        {
            assert!((row.iter().sum::<f32>() - 1.).abs() < 1e-6);
            let decoded = row
                .iter()
                .zip(support)
                .map(|(w, center)| w * center)
                .sum::<f32>();
            assert!((decoded - expected).abs() < 3e-6);
        }
        let mut illegal_request = request;
        illegal_request.fixed_actions.as_mut().unwrap()[0] = illegal;
        assert!(bundle.inspect_critics(&illegal_request).is_err());

        let mut request = InspectionRequest::new("policy", observations);
        request.critics = vec!["soft_q".into()];
        request.fixed_actions = Some(actions);
        request.layers.insert(
            "soft_q".into(),
            vec!["encoder.features".into(), "predictor.next_features".into()],
        );
        let result = bundle.inspect(&request).unwrap();
        let activations = &result.activations["soft_q"];
        let Record::Activations(meta) = &activations.record else {
            panic!()
        };
        near(
            activations.arrays[&meta.layers["encoder.features"]]
                .as_f32()
                .unwrap(),
            &expected_features,
        );
        near(
            activations.arrays[&meta.layers["predictor.next_features"]]
                .as_f32()
                .unwrap(),
            &expected_predictor,
        );
    }
}

#[test]
fn a_reppo_session_records_training_without_changing_the_learner() {
    use rl::{
        core::{checkpoint::Checkpoint, env::Env},
        diagnostics::{
            evaluation::EvaluationHorizon,
            probes::ProbeSet,
            schedule::Interval,
            schema::{ActionCondition, ProbePopulation, RecordKind},
            session::{DiagnosticConfig, DiagnosticSession},
            storage::{ReadLimits, Recording},
            value::{
                EntropyTemperatureSource, EntropyTiming, LogDensityConvention, ReturnTarget,
                declaration, return_target,
            },
        },
        envs::cartpole::CartPole,
        prelude::{Budget, OnPolicyCollector, OnPolicyConfig, OnPolicyTrainer, SyncVectorEnv},
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reppo");
    let env = CartPole::new(22);
    let space = env.action_space().clone();
    let support = HlGaussConfig {
        min: -3.,
        max: 7.,
        bins: 11,
        sigma_ratio: 0.75,
    };
    let build = || {
        let mut rng = ChaCha8Rng::seed_from_u64(78);
        let learner = Reppo::new(
            Policy::new(4, space.clone(), &[6], &mut rng).unwrap(),
            CategoricalQ::new(4, space.clone(), support.bins, &[6, 4], &mut rng).unwrap(),
            ReppoConfig {
                epochs: 1,
                minibatch_size: 4,
                auxiliary_coefficient: 0.5,
                hl_gauss: support.clone(),
                ..Default::default()
            },
            92,
        )
        .unwrap();
        let trainer = OnPolicyTrainer::new(
            OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()]).unwrap(), 97).unwrap(),
            OnPolicyConfig {
                rollout_steps: 4,
                gamma: 0.9,
                ..Default::default()
            },
        )
        .unwrap();
        (learner, trainer)
    };
    let (mut learner, mut trainer) = build();
    let (mut control, mut control_trainer) = build();
    control.restore(&learner.checkpoint().unwrap()).unwrap();
    let mut config = DiagnosticConfig::standard();
    config.schedule.representations = Some(Interval { first: 1, every: 1 });
    config.evaluation.seeds = vec![401];
    config.evaluation.max_steps = 8;
    config.evaluation_horizon = EvaluationHorizon::FiniteDecisions(8);
    let observations = vec![
        Observation::dense(vec![0.2, -0.1, 0.03, -0.05], Some(vec![false, true])),
        Observation::dense(vec![-0.1, 0.2, -0.04, 0.07], Some(vec![true, false])),
    ];
    let probes = ProbeSet::new(
        "explicit",
        ProbePopulation::FixedReference,
        "two observations with different singleton legal-action sets",
        observations.clone(),
        None,
        space.clone(),
    )
    .unwrap();
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .discount(f64::from(0.9_f32))
        .evaluation_factory(|seed| Ok(CartPole::new(seed)))
        .config(config)
        .probe_set(probes)
        .build(&learner)
        .unwrap();
    trainer
        .train_observed(&mut learner, Budget::vector_steps(8), None, &mut session)
        .unwrap();
    session.finish(&learner).unwrap();
    control_trainer
        .train(&mut control, Budget::vector_steps(8), None)
        .unwrap();
    assert_eq!(
        checkpoint_eq::normalized(&learner.checkpoint().unwrap()),
        checkpoint_eq::normalized(&control.checkpoint().unwrap())
    );
    assert_eq!(learner.updates(), 2);
    let recording = Recording::open(path, ReadLimits::default()).unwrap();
    assert!(recording.complete());
    assert_eq!(recording.manifest().algorithm, "REPPO");
    assert!(
        recording
            .entries()
            .iter()
            .any(|entry| entry.kind == RecordKind::Trajectory)
    );
    let chunk = |kind, probe: Option<&str>, revision| {
        let i = recording
            .entries()
            .iter()
            .rposition(|entry| {
                entry.kind == kind
                    && entry.revision == revision
                    && entry.probe_set.as_deref() == probe
            })
            .unwrap();
        recording.load(i).unwrap()
    };
    let snapshot = chunk(RecordKind::ModelSnapshot, None, learner.updates());
    let declaration = declaration(&snapshot).unwrap().unwrap();
    let critic_declaration = &declaration.critics["soft_q"];
    let actor = critic_declaration.actor.as_ref().unwrap();
    let continuation = critic_declaration.continuation.as_ref().unwrap();
    assert!(actor.stochastic && continuation.stochastic);
    assert_eq!(
        continuation.identity,
        format!("{}/rollout-behavior", actor.identity)
    );
    assert_eq!(
        return_target(&snapshot).unwrap(),
        Some(ReturnTarget::SoftActionValue {
            alpha: learner.multipliers().0,
            temperature_source: EntropyTemperatureSource::Snapshot,
            entropy_timing: EntropyTiming::TransitionSuccessor,
            log_density: LogDensityConvention::PolicySampleInActionCoordinates,
        })
    );
    let recorded = chunk(RecordKind::Critic, Some("explicit"), learner.updates());
    let Record::Critic(meta) = &recorded.record else {
        panic!()
    };
    assert_eq!(meta.action_condition, ActionCondition::CurrentPolicy);
    let actions = recorded.arrays[meta.actions.as_ref().unwrap()]
        .as_u64()
        .unwrap();
    assert_eq!(actions, &[1, 0]);
    let policy = chunk(RecordKind::Policy, Some("explicit"), learner.updates());
    let Record::Policy(policy_meta) = &policy.record else {
        panic!()
    };
    assert_eq!(
        policy.arrays[&policy_meta.actions].as_u64().unwrap(),
        actions
    );
    let bundle = ModelBundle::load(&snapshot, ModelLimits::default()).unwrap();
    let historical = bundle
        .inspect_critics(&CriticInspectionRequest {
            critics: vec!["soft_q".into()],
            observations,
            fixed_actions: Some(vec![Action::Discrete(1), Action::Discrete(0)]),
        })
        .unwrap();
    let historical = &historical.critics["soft_q"];
    assert_eq!(
        return_target(historical).unwrap(),
        return_target(&recorded).unwrap()
    );
    near(
        historical.arrays["values"].as_f32().unwrap(),
        recorded.arrays["values"].as_f32().unwrap(),
    );
    let metrics = chunk(RecordKind::Metrics, None, learner.updates());
    let Record::Metrics(metrics) = &metrics.record else {
        panic!()
    };
    assert!(metrics.values["alpha"].is_finite() && metrics.values["beta"].is_finite());
    let before = chunk(RecordKind::Metrics, None, 1);
    let Record::Metrics(before) = before.record else {
        panic!()
    };
    assert_eq!(metrics.values["target_alpha"], before.values["alpha"]);
    assert_ne!(metrics.values["target_alpha"], metrics.values["alpha"]);
}
