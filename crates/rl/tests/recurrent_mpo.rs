#![cfg(feature = "neural")]

//! Ordinary temporal training: publication clocks, frozen lanes and full resume.
#[path = "support/resume_fixture.rs"]
mod resume_fixture;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use resume_fixture::ResumeFixture;
use rl::{
    algorithms::{
        recurrent_critic::{
            RecurrentCriticConfig, RecurrentDiscreteCritic, construct_policy_targets,
            construct_targets,
        },
        recurrent_mpo::{MpoBackup, RecurrentMpo, RecurrentMpoConfig},
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::ActionSpace,
        checkpoint::Checkpoint,
        stateful::{BehaviorPublisher, HistorySpec, StatefulActor},
        temporal::TemporalExecutionMode,
    },
    data::temporal::{SequenceBatch, SequenceConfig, sample_at},
    envs::delayed_bit::DelayedBit,
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
        temporal_categorical_actor::{CategoricalMixtureSettings, TemporalCategoricalActor},
        temporal_policy::TemporalPolicy,
    },
    train::{
        behavior::{BehaviorSchedule, CategoricalMixtureSchedule},
        checkpoint::{load_run, save_run},
        counters::RunCounters,
        recurrent::TemporalTrainer,
        schedule::{Counter, Schedule},
        temporal_config::TemporalOffPolicyConfig,
    },
};

type Trainer = TemporalTrainer<
    SyncVectorEnv<DelayedBit>,
    TemporalCategoricalActor,
    CategoricalMixtureSchedule,
>;

fn learner(backup: MpoBackup, seed: u64) -> RecurrentMpo {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(
        spec.clone(),
        Some(4),
        &[],
        &mut ChaCha8Rng::seed_from_u64(seed),
    )
    .unwrap();
    let mut q_rng = ChaCha8Rng::seed_from_u64(seed ^ 0x4352_4954_4943);
    let critic = RecurrentDiscreteCritic::new(
        RecurrentQ::new(
            GruEncoder::new(spec.input_dim(), 4, &mut q_rng).unwrap(),
            Mlp::new(4, &[], 2, &mut q_rng).unwrap(),
        )
        .unwrap(),
        spec,
        RecurrentCriticConfig {
            gamma: 0.9,
            n_step: 1,
            target_interval: 2,
            ..Default::default()
        },
    )
    .unwrap();
    RecurrentMpo::new(
        policy,
        critic,
        RecurrentMpoConfig {
            backup,
            ..Default::default()
        },
    )
    .unwrap()
}

fn trainer(model: &RecurrentMpo, warmup: u64, schedule: Schedule) -> Trainer {
    let behavior = CategoricalMixtureSchedule(schedule);
    let settings = behavior.settings(&RunCounters::default()).unwrap();
    TemporalTrainer::new(
        SyncVectorEnv::new(vec![
            DelayedBit::new(2, None, false, 101).unwrap(),
            DelayedBit::new(3, None, false, 102).unwrap(),
        ])
        .unwrap(),
        model.behavior_actor(&settings, 0).unwrap(),
        TemporalOffPolicyConfig {
            replay_capacity: 128,
            batch_size: 2,
            warmup_transitions: warmup,
            n_step: 1,
            gamma: 0.9,
            updates_per_transition: 2.,
        },
        SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
        behavior,
        73,
    )
    .unwrap()
}

#[test]
fn checkpointed_mpo_backups_match_fresh_targets_and_resume() {
    type Restart = ResumeFixture<DelayedBit>;
    type RestartTrainer = TemporalTrainer<
        SyncVectorEnv<Restart>,
        TemporalCategoricalActor,
        CategoricalMixtureSchedule,
    >;
    for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
        let make = |mode| {
            let mut model = learner(backup, 11);
            model.set_update_mode(mode);
            let env = ResumeFixture::new(DelayedBit::new(4, None, false, 101).unwrap(), Some(3));
            let schedule = CategoricalMixtureSchedule(Schedule::Constant(0.15));
            let settings = schedule.settings(&RunCounters::default()).unwrap();
            let trainer = TemporalTrainer::new(
                SyncVectorEnv::new(vec![env]).unwrap(),
                model.behavior_actor(&settings, 0).unwrap(),
                TemporalOffPolicyConfig {
                    replay_capacity: 128,
                    batch_size: 8,
                    warmup_transitions: 0,
                    n_step: 1,
                    gamma: 0.9,
                    updates_per_transition: 1.,
                },
                SequenceConfig {
                    burn_in: 1,
                    learn_len: 2,
                    n_step: 1,
                },
                schedule,
                73,
            )
            .unwrap();
            (model, trainer)
        };
        let (mut reference, mut left) = make(TemporalExecutionMode::Reference);
        let (mut optimized, mut right) = make(TemporalExecutionMode::OptimizedExact);
        let mut oracle = learner(backup, 11);
        let mut saw_resume = false;
        for _ in 0..12 {
            oracle.restore(&reference.checkpoint().unwrap()).unwrap();
            let expected = left.step(&mut reference).unwrap();
            let actual = right.step(&mut optimized).unwrap();
            assert_eq!(expected.collection, actual.collection);
            assert_eq!(left.last_windows(), right.last_windows());
            assert_eq!(right.store().len() as u64, right.counters().env_steps);
            assert_eq!(expected.reports.len(), 1);
            assert_eq!(actual.reports.len(), 1);
            for (name, value) in &expected.reports[0].metrics {
                if !name.ends_with("_tokens") && name != "actor_reconstruction_passes" {
                    assert_eq!(
                        value.to_bits(),
                        actual.reports[0].metrics[name].to_bits(),
                        "{name}"
                    );
                }
            }
            if right
                .last_windows()
                .iter()
                .any(|&(id, _)| right.store().episode(id).unwrap().learning_start > 0)
            {
                saw_resume = true;
                let config = *right.sequence_config();
                let batch = SequenceBatch {
                    samples: right
                        .last_windows()
                        .iter()
                        .map(|&(id, start)| {
                            assert!(start >= right.store().episode(id).unwrap().learning_start);
                            sample_at(right.store(), id, start, &config).unwrap()
                        })
                        .collect(),
                    config,
                };
                let (online, target) = oracle.critic().reconstructed_q_values(&batch).unwrap();
                let targets = match backup {
                    MpoBackup::PolicyEvaluation => construct_policy_targets(
                        &batch,
                        &oracle.policy_probabilities(&batch).unwrap(),
                        &target,
                        2,
                        0.9,
                    ),
                    MpoBackup::DoubleQOptimality => {
                        construct_targets(&batch, &online, &target, 2, 0.9)
                    }
                    MpoBackup::MaximumOptimality => {
                        construct_targets(&batch, &target, &target, 2, 0.9)
                    }
                }
                .unwrap();
                let mean =
                    targets.iter().map(|p| f64::from(p.value)).sum::<f64>() / targets.len() as f64;
                assert_eq!(
                    mean.to_bits(),
                    actual.reports[0].metrics["target_mean"].to_bits()
                );
            }
        }
        assert!(saw_resume);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpointed-mpo");
        save_run(&path, &optimized, &right).unwrap();
        let mut restored = learner(backup, 999);
        let mut loaded: RestartTrainer = load_run(&path, &mut restored).unwrap();
        for _ in 0..3 {
            let a = right.step(&mut optimized).unwrap();
            let b = loaded.step(&mut restored).unwrap();
            assert_eq!(
                serde_json::to_value(a.reports).unwrap(),
                serde_json::to_value(b.reports).unwrap()
            );
            assert_eq!(right.last_windows(), loaded.last_windows());
            assert_eq!(
                right.collector().lane_state(0),
                loaded.collector().lane_state(0)
            );
        }
    }
}

#[test]
fn warmup_can_publish_new_settings_without_an_optimizer_step() {
    let mut model = learner(MpoBackup::PolicyEvaluation, 11);
    let mut run = trainer(
        &model,
        80,
        Schedule::Linear {
            counter: Counter::EnvironmentTransitions,
            start: 0.8,
            end: 0.1,
            duration: 40,
        },
    );
    for _ in 0..3 {
        run.step(&mut model).unwrap();
    }
    assert_eq!(model.updates(), 0);
    assert!(run.behavior_revision() > 0);
    assert_eq!(run.collector().lane_revision(1), Some(0));
    let behavior = model
        .behavior_actor(&CategoricalMixtureSettings { epsilon: 0.25 }, 91)
        .unwrap();
    assert_eq!(behavior.revision(), 91);
    assert_eq!(behavior.epsilon(), 0.25);
    assert!(
        model
            .behavior_actor(&CategoricalMixtureSettings { epsilon: f64::NAN }, 0)
            .is_err()
    );
}

#[test]
fn multiple_updates_per_publication_and_pinned_lanes_resume_exactly() {
    for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
        let mut model = learner(backup, 11);
        let mut run = trainer(&model, 0, Schedule::Constant(0.15));
        for _ in 0..5 {
            run.step(&mut model).unwrap();
        }
        assert!(model.updates() > run.behavior_revision());
        assert!(
            (0..2)
                .any(|lane| run.collector().lane_revision(lane).unwrap() < run.behavior_revision())
        );
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mpo-run");
        save_run(&path, &model, &run).unwrap();
        let mut restored = learner(backup, 999);
        let mut resumed: Trainer = load_run(&path, &mut restored).unwrap();
        for _ in 0..6 {
            let expected = run.step(&mut model).unwrap();
            let actual = resumed.step(&mut restored).unwrap();
            assert_eq!(expected.collection, actual.collection);
            assert_eq!(
                serde_json::to_value(expected.reports).unwrap(),
                serde_json::to_value(actual.reports).unwrap()
            );
            assert_eq!(run.last_windows(), resumed.last_windows());
            assert_eq!(
                serde_json::to_value(run.store()).unwrap(),
                serde_json::to_value(resumed.store()).unwrap()
            );
            assert_eq!(
                run.collector().observations(),
                resumed.collector().observations()
            );
            assert_eq!(
                model.critic().parameter_values().unwrap(),
                restored.critic().parameter_values().unwrap()
            );
            let settings = CategoricalMixtureSettings { epsilon: 0.15 };
            assert_eq!(
                model.behavior_actor(&settings, 0).unwrap().model_identity(),
                restored
                    .behavior_actor(&settings, 0)
                    .unwrap()
                    .model_identity()
            );
            for lane in 0..2 {
                assert_eq!(
                    run.collector().lane_state(lane),
                    resumed.collector().lane_state(lane)
                );
                assert_eq!(
                    run.collector().lane_revision(lane),
                    resumed.collector().lane_revision(lane)
                );
            }
        }
    }
}
