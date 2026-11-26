#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    envs::lqg::{Lqg, LqgConfig},
    prelude::*,
    train::checkpoint::{load_run, save_run},
};
fn build() -> (Mdpo, OnPolicyTrainer<SyncVectorEnv<TimeLimit<Lqg>>>) {
    let env = Lqg::new(
        LqgConfig::scalar(7, 0.95, 1., 1., 0.1, 2., 1., 0.05, 2.).unwrap(),
        1,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let models = ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[8], &mut rng).unwrap(),
        value: Value::new(2, &[8], &mut rng).unwrap(),
    };
    let learner = Mdpo::new(
        models,
        MdpoConfig {
            inner_steps: 3,
            value_epochs: 2,
            value_minibatch_size: 5,
            mirror_step: MirrorStepSchedule::Linear {
                start: 1.,
                end: 0.1,
                duration: 10,
            },
            ..Default::default()
        },
        8,
    )
    .unwrap();
    let env = TimeLimit::new(env, 3).unwrap();
    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env; 3]).unwrap(), 9).unwrap();
    let trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 4,
            ..Default::default()
        },
    )
    .unwrap();
    (learner, trainer)
}
#[test]
fn trainer_full_run_checkpoint_reproduces_truncations_rng_and_observers() {
    let (mut original, mut trainer) = build();
    trainer
        .train(&mut original, Budget::vector_steps(8), None)
        .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    save_run(&path, &original, &trainer).unwrap();
    let (mut restored, _) = build();
    let mut resumed: OnPolicyTrainer<SyncVectorEnv<TimeLimit<Lqg>>> =
        load_run(&path, &mut restored).unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    trainer
        .train_observed(
            &mut original,
            Budget::vector_steps(8),
            None,
            &mut |l: &Mdpo, boundary: rl::train::observer::UpdateBoundary<'_>| {
                assert_eq!(
                    boundary.report.metrics["outer_iterations"],
                    l.outer_iterations() as f64
                );
                a.push(boundary.report.metrics.clone());
                Ok(())
            },
        )
        .unwrap();
    resumed
        .train_observed(
            &mut restored,
            Budget::vector_steps(8),
            None,
            &mut |_: &Mdpo, boundary: rl::train::observer::UpdateBoundary<'_>| {
                b.push(boundary.report.metrics.clone());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(
        serde_json::to_value(trainer).unwrap(),
        serde_json::to_value(resumed).unwrap()
    );
    assert_eq!(
        original.checkpoint().unwrap().metadata,
        restored.checkpoint().unwrap().metadata
    );
}
#[test]
fn trainer_supports_mc_and_gae_with_and_without_normalization() {
    for gae_lambda in [None, Some(0.95)] {
        for normalize_advantages in [false, true] {
            let (mut learner, trainer) = build();
            let collector = trainer.collector().clone();
            let mut trainer = OnPolicyTrainer::new(
                collector,
                OnPolicyConfig {
                    rollout_steps: 4,
                    gae_lambda,
                    normalize_advantages,
                    ..Default::default()
                },
            )
            .unwrap();
            let result = trainer
                .train(&mut learner, Budget::vector_steps(8), None)
                .unwrap();
            assert_eq!(result.counters.env_steps, 24);
            assert_eq!(learner.outer_iterations(), 2);
            assert!(
                result
                    .last_update
                    .unwrap()
                    .metrics
                    .values()
                    .all(|x| x.is_finite())
            );
        }
    }
}

#[cfg(feature = "diagnostics")]
#[test]
fn generic_diagnostics_record_mdpo_policy_value_network_and_metrics() {
    use rl::{
        diagnostics::{
            models::SnapshotBuilder,
            observer::DiagnosticObserver,
            probes::{self, ProbeSet},
            schedule::{Due, Interval, RecordingSchedule},
            schema::*,
            storage::*,
        },
        nn::{observations, probes::StateValueProbe},
    };
    let (mut learner, mut trainer) = build();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("recording");
    let space = learner.actor().action_space().clone();
    let probes = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "MDPO LQG states",
        vec![
            Observation::new(vec![1., 0.]),
            Observation::new(vec![-0.5, 0.5]),
        ],
        None,
        space.clone(),
    )
    .unwrap();
    let recorder = Recorder::create(
        &path,
        RunManifest::new("mdpo", "integration", "mdpo", 2, space.clone(), 0.99).unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    let schedule = RecordingSchedule {
        metrics: Some(Interval { first: 1, every: 1 }),
        probes: Some(Interval { first: 1, every: 1 }),
        monte_carlo: None,
        snapshots: Some(Interval { first: 1, every: 1 }),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(990);
    let mut observer = DiagnosticObserver::new(
        recorder,
        schedule,
        |l: &Mdpo, context: &RecordContext, _: Due, writer: &mut Recorder| {
            let policy = l.policy();
            let value = l.value();
            writer.append(&probes::policy(
                context.clone(),
                &policy,
                &probes,
                ActMode::Sample,
                &mut rng,
            )?)?;
            writer.append(&probes::critic(
                context.clone(),
                StateValueProbe(&value).predict(probes.observations().into())?,
                ActionCondition::StateValue,
                None,
            )?)?;
            writer.append(&probes::activations(
                context.clone(),
                "policy",
                &policy.network,
                observations(probes.observations().into(), 2)?,
                &["layers.0.pre", "layers.0.post"],
            )?)?;
            let mut snapshot = SnapshotBuilder::new("mdpo", context.clone(), 2, space.clone())?;
            snapshot.policy("policy", &policy)?;
            snapshot.value("value", &value)?;
            writer.append(&snapshot.finish()?)?;
            Ok(())
        },
    );
    trainer
        .train_observed(&mut learner, Budget::vector_steps(8), None, &mut observer)
        .unwrap();
    observer.finish().unwrap();
    drop(observer);
    let recording = Recording::open(&path, ReadLimits::default()).unwrap();
    for kind in [
        RecordKind::Metrics,
        RecordKind::Policy,
        RecordKind::Critic,
        RecordKind::Activations,
        RecordKind::ModelSnapshot,
    ] {
        assert_eq!(
            recording
                .entries()
                .iter()
                .filter(|e| e.kind == kind)
                .count(),
            2
        );
    }
    for i in 0..recording.entries().len() {
        recording.load(i).unwrap();
    }
}
