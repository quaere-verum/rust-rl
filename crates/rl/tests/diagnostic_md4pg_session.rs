#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        models::{InspectionRequest, ModelBundle, ModelLimits},
        probes::ProbeSet,
        schedule::Interval,
        schema::{CriticRepresentation, ProbePopulation, Record, RecordKind},
        session::{DiagnosticConfig, DiagnosticSession},
        storage::{ReadLimits, Recording},
        value::{self, MomentUtility},
    },
    envs::lqg::{Lqg, LqgConfig},
    nn::{critics::scalar::ActionValue, matrix, to_vec},
    objectives::risk::Risk,
    prelude::*,
};

#[test]
fn trained_md4pg_session_preserves_moment_roles_and_the_factual_trainer() -> Result<()> {
    const HORIZON: usize = 4;
    const GAMMA: f32 = 0.5;
    let space = ActionSpace::Box {
        low: vec![-2., -0.5],
        high: vec![1., 3.],
    };
    let environment = LqgConfig {
        a: vec![DMatrix::identity(3, 3); HORIZON],
        b: vec![DMatrix::from_row_slice(3, 2, &[0.2, 0., 0., 0.4, 0.1, -0.2]); HORIZON],
        q: vec![DMatrix::identity(3, 3); HORIZON],
        r: vec![DMatrix::identity(2, 2); HORIZON],
        terminal: DMatrix::identity(3, 3),
        noise: vec![DMatrix::identity(3, 3) * 0.05; HORIZON],
        initial: vec![1., -0.5, 0.25],
        action_space: space.clone(),
    };
    let mut settings = Md4pgConfig {
        risk: Risk::MeanVariance { penalty: 0.25 },
        ..Default::default()
    };
    settings.training.tau = 0.5;
    let learner = || {
        let mut rng = ChaCha8Rng::seed_from_u64(21);
        Md4pg::new(
            Deterministic::new(4, space.clone(), &[4], &mut rng)?,
            QValue::new(4, 2, 1, &[4], &mut rng)?,
            QValue::new(4, 2, 1, &[4], &mut rng)?,
            settings.clone(),
            22,
        )
    };
    let mut observed = learner()?;
    let mut plain = learner()?;
    plain.restore(&observed.checkpoint()?)?;

    let env = Lqg::new(environment.clone(), 23)?;
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()])?, 24)?,
        OffPolicyConfig {
            replay_capacity: 32,
            batch_size: 2,
            warmup_transitions: 2,
            updates_per_transition: 0.5,
            gamma: GAMMA,
            ..Default::default()
        },
        25,
    )?;
    let mut control = trainer.clone();
    let observations = vec![
        Observation::new(vec![1., -0.5, 0.25, 0.]),
        Observation::new(vec![0.2, 0.3, -0.5, 0.5]),
    ];
    let actions = vec![
        Action::Continuous(vec![0., 1.]),
        Action::Continuous(vec![-1., 2.]),
    ];
    let probes = ProbeSet::new(
        "two-control",
        ProbePopulation::FixedReference,
        "two bounded controls and two time-dependent inputs",
        observations.clone(),
        Some(actions.clone()),
        space,
    )?;

    let root = tempfile::tempdir()?;
    let path = root.path().join("md4pg-two-control");
    let mut config = DiagnosticConfig::standard();
    let every = Some(Interval { first: 1, every: 1 });
    config.schedule.probes = every;
    config.schedule.snapshots = every;
    config.evaluation.max_steps = HORIZON;
    config.evaluation.seeds = vec![30, 31];
    let mut capture = DiagnosticSession::builder(&path)
        .environment(&env)
        .discount(f64::from(GAMMA))
        .evaluation_factory(move |seed| Lqg::new(environment.clone(), seed))
        .probe_set(probes)
        .config(config)
        .build(&observed)?;
    trainer.train_observed(&mut observed, Budget::vector_steps(8), None, &mut capture)?;
    capture.finish(&observed)?;
    control.train(&mut plain, Budget::vector_steps(8), None)?;
    let revision = observed.updates();
    assert!(revision > 0);
    assert_eq!(
        serde_json::to_value(&trainer)?,
        serde_json::to_value(&control)?
    );
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&plain.checkpoint()?)
    );

    let run = Recording::open(path, ReadLimits::default())?;
    assert!(run.complete());
    assert!(
        run.entries()
            .iter()
            .any(|entry| entry.revision == revision && entry.kind == RecordKind::Empirical)
    );
    let action_tensor = matrix(vec![0., 1., -1., 2.], 2, 2);
    for (role, policy, first, second) in [
        (
            "online",
            "policy",
            observed.critic(),
            observed.second_critic(),
        ),
        (
            "target",
            "target_policy",
            observed.target_critic(),
            observed.target_second_critic(),
        ),
    ] {
        let load = |kind| -> Result<_> {
            let (index, _) = run
                .entries()
                .iter()
                .enumerate()
                .find(|(_, entry)| {
                    entry.revision == revision && entry.model_role == role && entry.kind == kind
                })
                .expect("final revision and model role");
            run.load(index)
        };
        let saved = load(RecordKind::Critic)?;
        let declaration = value::declaration(&saved)?.expect("moment critic declaration");
        let subject = &declaration.critics["moment_q"];
        assert_eq!(
            subject.utility,
            Some(MomentUtility::MeanVariance { penalty: 0.25 })
        );
        assert_eq!(
            subject.continuation.as_ref().unwrap().identity,
            format!("md4pg-two-control@{revision}/target-policy")
        );

        let bundle = ModelBundle::load(&load(RecordKind::ModelSnapshot)?, ModelLimits::default())?;
        let mut request = InspectionRequest::new(policy, observations.clone());
        request.fixed_actions = Some(actions.clone());
        request.critics = vec!["moment_q".into()];
        let historical = bundle.inspect(&request)?;
        let expected_mean =
            to_vec(first.values(observations.as_slice().into(), action_tensor.clone())?)?;
        let expected_second =
            to_vec(second.values(observations.as_slice().into(), action_tensor.clone())?)?;
        for evidence in [&saved, &historical.critics["moment_q"]] {
            let Record::Critic(record) = &evidence.record else {
                panic!("critic expected")
            };
            let CriticRepresentation::RawMoments { mean, raw_second } = &record.representation
            else {
                panic!("two raw moments expected")
            };
            assert_eq!(evidence.arrays[mean].as_f32()?, expected_mean);
            assert_eq!(evidence.arrays[raw_second].as_f32()?, expected_second);
        }
    }
    Ok(())
}
