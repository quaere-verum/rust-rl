#![cfg(feature = "neural")]

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;
#[allow(dead_code)]
#[path = "support/delayed_candidate.rs"]
mod delayed_candidate;

use burn::tensor::Device;
use delayed_candidate::{DelayedCandidate, model_config};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        candidate_ddqn::{CandidateDdqnConfig, CandidateExploration},
        candidate_mpo::{CandidateMpoBackup, CandidateMpoBehavior, CandidateMpoConfig},
        recurrent_candidates::{
            RecurrentCandidateDdqn, RecurrentCandidateMpo, RecurrentGradientMode,
            RecurrentMpoGradients,
        },
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        checkpoint::{Checkpoint, Snapshot},
        env::{Env, TimeLimit},
        stateful::{BehaviorPublisher, StatefulActor, TemporalOffPolicyLearner},
    },
    data::temporal::{SequenceBatch, SequenceConfig, sample_at},
    nn::structured::{RecurrentLightweightPolicy, RecurrentLightweightQ},
    train::{
        behavior::FixedBehavior,
        checkpoint::{load_run, save_run},
        recurrent::TemporalTrainer,
        temporal_config::TemporalOffPolicyConfig,
    },
};
use serde::{Serialize, de::DeserializeOwned};

const GAMMA: f32 = 0.99;

fn ddqn(seed: u64) -> Result<RecurrentCandidateDdqn> {
    let env = DelayedCandidate::new();
    let model = RecurrentLightweightQ::new(
        model_config(),
        &mut ChaCha8Rng::seed_from_u64(seed),
        &Device::flex().autodiff(),
    )?;
    RecurrentCandidateDdqn::new(
        model,
        CandidateDdqnConfig {
            gamma: GAMMA,
            ..Default::default()
        },
        RecurrentGradientMode::LearningWindow,
        env.observation_space(),
        env.action_space().clone(),
    )
}

fn mpo(seed: u64) -> Result<RecurrentCandidateMpo> {
    let env = DelayedCandidate::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let device = Device::flex().autodiff();
    RecurrentCandidateMpo::new(
        RecurrentLightweightPolicy::new(model_config(), &mut rng, &device)?,
        RecurrentLightweightQ::new(model_config(), &mut rng, &device)?,
        CandidateDdqnConfig {
            gamma: GAMMA,
            ..Default::default()
        },
        CandidateMpoConfig {
            backup: CandidateMpoBackup::DoubleQOptimality,
            ..Default::default()
        },
        RecurrentMpoGradients::default(),
        env.observation_space(),
        env.action_space().clone(),
    )
}

fn native_run_restores<L>(learner: impl Fn(u64) -> Result<L>, settings: L::Settings) -> Result<()>
where
    L: TemporalOffPolicyLearner<Batch = SequenceBatch> + BehaviorPublisher + Checkpoint,
    L::Actor: Snapshot,
    <L::Actor as StatefulActor>::State: Serialize + DeserializeOwned,
    L::Settings: Clone + PartialEq + Serialize + DeserializeOwned,
{
    let mut original = learner(19)?;
    let mut restored = learner(193)?;
    let env = SyncVectorEnv::new(vec![
        TimeLimit::new(DelayedCandidate::new(), 4)?,
        TimeLimit::new(DelayedCandidate::new(), 100)?,
    ])?;
    let sequence = SequenceConfig {
        burn_in: 1,
        learn_len: 3,
        n_step: 1,
    };
    let mut run = TemporalTrainer::new(
        env,
        original.behavior_actor(&settings, 0)?,
        TemporalOffPolicyConfig {
            replay_capacity: 64,
            batch_size: 4,
            warmup_transitions: 4,
            n_step: 1,
            gamma: GAMMA,
            updates_per_transition: 0.5,
        },
        sequence,
        FixedBehavior(settings),
        47,
    )?;
    for _ in 0..7 {
        run.step(&mut original)?;
    }
    assert!(run.counters().updates > 0);
    assert!(run.counters().physical_steps() > run.counters().env_steps);
    assert!(run.collector().latest_revision() > 0);
    assert!(run.store().active_count() > 0);
    assert!(!run.last_windows().is_empty());

    let mut saw_termination = false;
    let mut saw_truncation = false;
    for episode in run.store().episodes() {
        for (observation, step) in episode.observations.iter().zip(&episode.steps) {
            observation.check_action(run.store().action_space(), &step.action)?;
        }
        let Some(step) = episode.steps.last().filter(|step| step.ended()) else {
            continue;
        };
        let last = episode.end_timestep() - 1;
        let sample = sample_at(run.store(), episode.id, last, &sequence)?;
        assert_eq!(sample.history_start, 0);
        assert_eq!(sample.observations[0], episode.observations[0]);
        assert!(sample.loss_mask[..last].iter().all(|&learning| !learning));
        let target = sample.target_at(last, GAMMA, 1)?;
        assert_eq!(target.bootstrap_index, last + 1);
        if step.terminated {
            saw_termination = true;
            assert_eq!(target.discount, 0.);
        } else {
            saw_truncation = true;
            assert!(step.truncated);
            assert_eq!(step.reward.primitives().len(), 2);
            assert!((target.discount - GAMMA * GAMMA).abs() < 1e-7);
        }
    }
    assert!(saw_termination && saw_truncation);

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("run");
    save_run(&path, &original, &run)?;
    let mut resumed: TemporalTrainer<
        SyncVectorEnv<TimeLimit<DelayedCandidate>>,
        L::Actor,
        FixedBehavior<L::Settings>,
    > = load_run(&path, &mut restored)?;
    for _ in 0..4 {
        let expected = run.step(&mut original)?;
        let actual = resumed.step(&mut restored)?;
        assert_eq!(actual.collection, expected.collection);
        assert_eq!(
            serde_json::to_value(&actual.reports)?,
            serde_json::to_value(&expected.reports)?
        );
        assert_eq!(resumed.counters(), run.counters());
        assert_eq!(resumed.last_windows(), run.last_windows());
    }
    assert_eq!(serde_json::to_value(&resumed)?, serde_json::to_value(&run)?);
    assert_eq!(
        checkpoint_eq::normalized(&restored.checkpoint()?),
        checkpoint_eq::normalized(&original.checkpoint()?),
    );
    Ok(())
}

#[test]
fn recurrent_candidate_ddqn_trains_and_restores_through_the_native_runtime() -> Result<()> {
    native_run_restores(
        ddqn,
        CandidateExploration {
            epsilon: 0.2,
            ..Default::default()
        },
    )
}

#[test]
fn recurrent_candidate_mpo_trains_and_restores_through_the_native_runtime() -> Result<()> {
    native_run_restores(mpo, CandidateMpoBehavior { epsilon: 0.2 })
}
