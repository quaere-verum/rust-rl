#![cfg(all(feature = "diagnostics", feature = "neural"))]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{
        models::{InspectionRequest, ModelBundle, ModelLimits},
        probes::ProbeSet,
        schedule::Interval,
        schema::{Conditioning, ProbePopulation, Record, RecordKind},
        session::{DiagnosticConfig, DiagnosticSession},
        storage::{ReadLimits, Recording},
    },
    envs::lqg::{Lqg, LqgConfig},
    prelude::*,
};
use std::path::Path;

fn environment(seed: u64) -> Result<Lqg> {
    Lqg::new(
        LqgConfig::scalar(4, 1., 0.5, 1., 0.2, 1., 1., 0.1, 2.)?,
        seed,
    )
}

fn probes(env: &Lqg, observation: Observation) -> Result<ProbeSet> {
    ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "two preselected LQG model inputs",
        vec![observation, Observation::new(vec![-0.5, 0.5])],
        None,
        env.action_space().clone(),
    )
}

fn config() -> DiagnosticConfig {
    let mut config = DiagnosticConfig::detailed();
    let each_checkpoint = Some(Interval { first: 1, every: 1 });
    let boundaries_only = Some(Interval {
        first: 64,
        every: 64,
    });
    config.schedule.probes = each_checkpoint;
    config.schedule.representations = each_checkpoint;
    config.schedule.snapshots = boundaries_only;
    config.schedule.evaluation = boundaries_only;
    config.schedule.conditional_evaluation = boundaries_only;
    config.evaluation.seeds = vec![181, 182];
    config.evaluation.max_steps = 4;
    config.evaluation.trajectories = 1;
    config.conditional_evaluation.seeds = vec![191, 192, 193];
    config.conditional_evaluation.max_steps = 4;
    config
}

fn assert_recorded_boundaries(path: &Path, final_revision: u64, stochastic: bool) -> Result<()> {
    let recording = Recording::open(path, ReadLimits::default())?;
    assert!(recording.complete());
    assert_eq!(recording.manifest().starting_revision, 0);

    let request = InspectionRequest::new(
        "policy",
        vec![
            Observation::new(vec![1., 0.]),
            Observation::new(vec![-0.5, 0.5]),
        ],
    );
    let mut snapshots = Vec::new();
    let mut trajectories = Vec::new();
    for (index, entry) in recording.entries().iter().enumerate() {
        match entry.kind {
            RecordKind::ModelSnapshot if entry.model_role == "online" => {
                let snapshot = recording.load(index)?;
                let bundle = ModelBundle::load(&snapshot, ModelLimits::default())?;
                snapshots.push((entry.revision, bundle.inspect(&request)?.actions));
            }
            RecordKind::Trajectory => {
                let chunk = recording.load(index)?;
                let Record::Trajectory(trajectory) = &chunk.record else {
                    unreachable!()
                };
                assert!(trajectory.evaluation.complete_returns);
                assert_eq!(trajectory.evaluation.stochastic_policy, stochastic);
                assert_eq!(
                    trajectory.evaluation.conditioning,
                    Conditioning::InitialScenario
                );
                assert_eq!(chunk.arrays[&trajectory.observations].shape(), [5, 2]);
                assert_eq!(chunk.arrays[&trajectory.terminated].as_u8()?, [0, 0, 0, 1]);
                assert_eq!(chunk.arrays[&trajectory.truncated].as_u8()?, [0; 4]);
                trajectories.push(entry.revision);
            }
            _ => {}
        }
    }
    assert_eq!(
        snapshots
            .iter()
            .map(|(revision, _)| *revision)
            .collect::<Vec<_>>(),
        [0, final_revision]
    );
    assert_eq!(trajectories, [0, final_revision]);
    assert_ne!(snapshots[0].1, snapshots[1].1);
    assert!(
        recording
            .entries()
            .iter()
            .any(|entry| entry.kind == RecordKind::Activations)
    );
    Ok(())
}

#[test]
fn ppo_session_preserves_optimizer_policy_rng_and_collection_state() -> Result<()> {
    let env = environment(101)?;
    let mut rng = ChaCha8Rng::seed_from_u64(102);
    let build_learner = |rng: &mut ChaCha8Rng| {
        Ppo::new(
            ActorCritic {
                policy: Policy::new(2, env.action_space().clone(), &[4], rng)?,
                value: Value::new(2, &[4], rng)?,
            },
            PpoConfig {
                epochs: 2,
                minibatch_size: 2,
                ..Default::default()
            },
            103,
        )
    };
    let mut observed = build_learner(&mut rng)?;
    let mut plain = build_learner(&mut rng)?;
    plain.restore(&observed.checkpoint()?)?;
    let mut observed_trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()])?, 104)?,
        OnPolicyConfig {
            rollout_steps: 4,
            gamma: 1.,
            ..Default::default()
        },
    )?;
    let mut plain_trainer = observed_trainer.clone();

    let temp = tempfile::tempdir()?;
    let path = temp.path().join("ppo");
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .discount(1.)
        .evaluation_factory(environment)
        .probe_set(probes(&env, Observation::new(vec![1., 0.]))?)
        .config(config())
        .build(&observed)?;
    observed_trainer.train_observed(&mut observed, Budget::vector_steps(8), None, &mut session)?;
    session.finish(&observed)?;
    plain_trainer.train(&mut plain, Budget::vector_steps(8), None)?;

    assert_eq!(observed.updates(), 8);
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&plain.checkpoint()?)
    );
    assert_eq!(
        serde_json::to_value(observed_trainer)?,
        serde_json::to_value(plain_trainer)?
    );
    assert_recorded_boundaries(&path, 8, true)
}

#[test]
fn td3_session_preserves_both_optimizer_states_and_replay_rng_with_conditional_evaluation()
-> Result<()> {
    let env = environment(111)?;
    let mut rng = ChaCha8Rng::seed_from_u64(112);
    let build_learner = |rng: &mut ChaCha8Rng| {
        Td3::new(
            Deterministic::new(2, env.action_space().clone(), &[4], rng)?,
            QValue::new(2, 1, 1, &[4], rng)?,
            QValue::new(2, 1, 1, &[4], rng)?,
            Td3Config::default(),
            113,
        )
    };
    let mut observed = build_learner(&mut rng)?;
    let mut plain = build_learner(&mut rng)?;
    plain.restore(&observed.checkpoint()?)?;
    let mut observed_trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()])?, 114)?,
        OffPolicyConfig {
            replay_capacity: 16,
            batch_size: 4,
            warmup_transitions: 0,
            n_step: 3,
            gamma: 1.,
            ..Default::default()
        },
        115,
    )?;
    let mut plain_trainer = observed_trainer.clone();
    let mut conditional = environment(116)?;
    let observation = conditional.reset(Some(116))?.observation;

    let temp = tempfile::tempdir()?;
    let path = temp.path().join("td3");
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .discount(1.)
        .evaluation_factory(environment)
        .probe_set(probes(&env, observation)?)
        .conditional_case("fixed", 0, conditional, None)?
        .config(config())
        .build(&observed)?;
    observed_trainer.train_observed(&mut observed, Budget::vector_steps(7), None, &mut session)?;
    session.finish(&observed)?;
    plain_trainer.train(&mut plain, Budget::vector_steps(7), None)?;

    assert_eq!((observed.updates(), observed.actor_updates()), (6, 3));
    assert_eq!(
        checkpoint_eq::normalized(&observed.checkpoint()?),
        checkpoint_eq::normalized(&plain.checkpoint()?)
    );
    assert_eq!(
        serde_json::to_value(observed_trainer)?,
        serde_json::to_value(plain_trainer)?
    );
    assert_recorded_boundaries(&path, 6, false)?;

    let recording = Recording::open(&path, ReadLimits::default())?;
    let mut conditional_revisions = Vec::new();
    for (index, entry) in recording
        .entries()
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.kind == RecordKind::Empirical)
    {
        let chunk = recording.load(index)?;
        let Record::Empirical(record) = chunk.record else {
            unreachable!()
        };
        if matches!(
            record.evaluation.conditioning,
            Conditioning::FullSimulatorState { .. }
        ) {
            assert!(record.evaluation.stochastic_policy);
            assert!(
                record
                    .evaluation
                    .continuation_policy
                    .contains("Gaussian-smoothing")
            );
            assert!(record.evaluation.complete_returns);
            assert_eq!(chunk.arrays[&record.returns].shape(), [3]);
            conditional_revisions.push(entry.revision);
        }
    }
    assert_eq!(conditional_revisions, [0, 6]);
    Ok(())
}
