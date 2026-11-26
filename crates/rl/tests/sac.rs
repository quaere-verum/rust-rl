#![cfg(feature = "neural")]

use burn::module::Param;
#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;
use checkpoint_eq::normalized;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::sac::{Sac, SacConfig, Temperature},
    core::{
        action::{Action, ActionSpace},
        checkpoint::Checkpoint,
        env::Observation,
        learner::{Learner, OffPolicyLearner},
        transition::Transition,
    },
    data::replay::{ReplayBatch, ReplayItem},
    nn::{critics::scalar::QValue, gaussian_policy::GaussianPolicy, matrix, vector},
};

fn build(config: SacConfig) -> Sac {
    let mut rng = ChaCha8Rng::seed_from_u64(717);
    let policy = GaussianPolicy::new(
        1,
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![4.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let mut first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let mut second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    for (q, value) in [(&mut first, 3.), (&mut second, 1.)] {
        q.network.layers[0].weight = Param::from_tensor(matrix(vec![0., 0.], 2, 1));
        q.network.layers[0].bias = Some(Param::from_tensor(vector(&[value])));
    }
    Sac::new(policy, first, second, config, 719).unwrap()
}
fn batch() -> ReplayBatch {
    ReplayBatch {
        items: [(true, false), (false, false), (false, true)]
            .into_iter()
            .map(|(terminated, truncated)| ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![1.]),
                    action: Action::Continuous(vec![0.]),
                    reward: 2.0_f32.into(),
                    next_observation: Observation::new(vec![4.]),
                    terminated,
                    truncated,
                },
                discount: if terminated { 0. } else { 0.5 },
            })
            .collect(),
    }
}

#[test]
fn zero_entropy_targets_distinguish_termination_and_time_limit() {
    let mut learner = build(SacConfig {
        gamma: 0.5,
        temperature: Temperature::Fixed(0.),
        ..Default::default()
    });
    let report = learner.update(&batch()).unwrap();
    // Targets [2, 2.5, 2.5]; critics initially 3 and 1.
    assert!((report.metrics["target_mean"] - 7. / 3.).abs() < 1e-6);
    assert!((report.metrics["critic_loss"] - 7. / 3.).abs() < 1e-6);
    assert_eq!(report.metrics["alpha"], 0.);
    assert!(learner.set_exploration(0.1).is_err());
    learner.set_exploration(0.).unwrap();
}

#[test]
fn independent_sampling_and_optimizer_state_continue_exactly() {
    let config = SacConfig {
        gamma: 0.5,
        ..Default::default()
    };
    let mut original = build(config.clone());
    for _ in 0..3 {
        original.update(&batch()).unwrap();
    }
    let record = original.checkpoint().unwrap();
    let mut restored = build(config);
    restored.restore(&record).unwrap();
    for _ in 0..3 {
        assert_eq!(
            original.update(&batch()).unwrap().metrics,
            restored.update(&batch()).unwrap().metrics
        );
    }
    assert_eq!(
        normalized(&original.checkpoint().unwrap()),
        normalized(&restored.checkpoint().unwrap())
    );
}

#[test]
fn target_cadence_temperature_direction_and_invalid_restore_are_explicit() {
    let mut learner = build(SacConfig {
        gamma: 0.5,
        target_interval: 2,
        target_tau: 1.,
        ..Default::default()
    });
    let target = rl::nn::to_vec(
        learner.target_critic().network.layers[0]
            .bias
            .as_ref()
            .unwrap()
            .val(),
    )
    .unwrap();
    let report = learner.update(&batch()).unwrap();
    assert_eq!(report.metrics["target_updated"], 0.);
    assert_eq!(
        rl::nn::to_vec(
            learner.target_critic().network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
        )
        .unwrap(),
        target
    );
    assert_eq!(
        learner.update(&batch()).unwrap().metrics["target_updated"],
        1.
    );
    assert_eq!(
        rl::nn::to_vec(
            learner.target_critic().network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
        )
        .unwrap(),
        rl::nn::to_vec(
            learner.critic().network.layers[0]
                .bias
                .as_ref()
                .unwrap()
                .val()
        )
        .unwrap()
    );
    let record = learner.checkpoint().unwrap();
    for case in 0..5 {
        let mut invalid = record.clone();
        match case {
            0 => invalid.replay_requirements = None,
            1 => invalid.metadata["target_rng"] = invalid.metadata["actor_rng"].clone(),
            2 => {
                invalid
                    .optimizers
                    .insert("first".into(), invalid.optimizers["second"].clone());
            }
            3 => invalid.metadata["config"]["gamma"] = serde_json::json!(0.9),
            _ => invalid.metadata["optimizer_owners"]["actor"] = serde_json::json!([]),
        }
        assert!(learner.restore(&invalid).is_err());
        assert_eq!(
            normalized(&learner.checkpoint().unwrap()),
            normalized(&record)
        );
    }
    for target in [-100., 100.] {
        let mut learner = build(SacConfig {
            gamma: 0.5,
            temperature: Temperature::Learned {
                initial: 0.1,
                target_entropy: Some(target),
                optimizer: Default::default(),
            },
            ..Default::default()
        });
        let before = learner.alpha();
        learner.update(&batch()).unwrap();
        assert_eq!(learner.alpha() > before, target > 0.);
    }
}

fn trainer_pair() -> (
    Sac,
    rl::train::trainer::OffPolicyTrainer<
        rl::collect::vector_env::SyncVectorEnv<rl::envs::lqg::Lqg>,
    >,
) {
    use rl::envs::lqg::{Lqg, LqgConfig};
    use rl::prelude::*;
    let env = Lqg::new(
        LqgConfig::scalar(4, 0.8, 1., 1., 0.1, 1., 1., 0.1, 2.).unwrap(),
        31,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(32);
    let width = env.dense_observation_dim().unwrap();
    let policy = GaussianPolicy::new(width, env.action_space().clone(), &[4], &mut rng).unwrap();
    let q1 = QValue::new(width, 1, 1, &[4], &mut rng).unwrap();
    let q2 = QValue::new(width, 1, 1, &[4], &mut rng).unwrap();
    let learner = Sac::new(policy, q1, q2, Default::default(), 33).unwrap();
    let collector = OffPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 34).unwrap();
    let trainer = OffPolicyTrainer::new(
        collector,
        OffPolicyConfig {
            replay_capacity: 32,
            batch_size: 4,
            warmup_transitions: 2,
            exploration: Schedule::Constant(0.),
            ..Default::default()
        },
        35,
    )
    .unwrap();
    (learner, trainer)
}

#[test]
fn ordinary_trainer_checkpoint_resumes_pending_credit_replay_and_all_rngs() {
    use rl::prelude::*;
    use rl::train::checkpoint::{load_run, save_run};
    let (mut learner, mut trainer) = trainer_pair();
    trainer
        .train(&mut learner, Budget::vector_steps(9), None)
        .unwrap();
    assert!(trainer.update_credit() >= 1.);
    let dir = tempfile::tempdir().unwrap();
    save_run(dir.path().join("run"), &learner, &trainer).unwrap();
    let (mut restored, _) = trainer_pair();
    let mut resumed: OffPolicyTrainer<SyncVectorEnv<rl::envs::lqg::Lqg>> =
        load_run(dir.path().join("run"), &mut restored).unwrap();
    for _ in 0..3 {
        let a = trainer
            .train(&mut learner, Budget::vector_steps(3), None)
            .unwrap();
        let b = resumed
            .train(&mut restored, Budget::vector_steps(3), None)
            .unwrap();
        assert_eq!(
            a.last_update.unwrap().metrics,
            b.last_update.unwrap().metrics
        );
        assert_eq!(
            serde_json::to_value(&trainer).unwrap(),
            serde_json::to_value(&resumed).unwrap()
        );
    }
    assert_eq!(
        normalized(&learner.checkpoint().unwrap()),
        normalized(&restored.checkpoint().unwrap())
    );
}

#[test]
fn trainer_contract_rejects_before_pending_updates_or_filesystem_creation() {
    use rl::prelude::*;
    use rl::train::checkpoint::save_run;
    let (mut learner, mut trainer) = trainer_pair();
    trainer
        .train(&mut learner, Budget::vector_steps(9), None)
        .unwrap();
    let mut snapshot = serde_json::to_value(&trainer).unwrap();
    snapshot["config"]["exploration"] = serde_json::json!({"Constant":1e-300});
    let mut invalid: OffPolicyTrainer<SyncVectorEnv<rl::envs::lqg::Lqg>> =
        serde_json::from_value(snapshot).unwrap();
    let before = learner.checkpoint().unwrap();
    let counters = serde_json::to_value(&invalid).unwrap();
    assert!(
        invalid
            .train(&mut learner, Budget::vector_steps(1), None)
            .is_err()
    );
    assert_eq!(serde_json::to_value(&invalid).unwrap(), counters);
    assert_eq!(
        normalized(&learner.checkpoint().unwrap()),
        normalized(&before)
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("invalid");
    assert!(save_run(&path, &learner, &invalid).is_err());
    assert!(!path.exists());
}
