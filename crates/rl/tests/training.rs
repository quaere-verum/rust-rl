#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::ppo::{Ppo, PpoConfig, PpoUpdateSettings},
    collect::{OffPolicyCollector, OnPolicyCollector, vector_env::SyncVectorEnv},
    core::env::Env,
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        critics::scalar::Value,
        policy::{ActorCritic, Policy},
    },
    train::{
        config::{Budget, OnPolicyConfig},
        schedule::{Counter, Schedule},
        trainer::OnPolicyTrainer,
    },
};
#[test]
fn public_trainer_reports_counts_and_respects_vector_budget() {
    let env = Lqg::new(
        LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        0,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let models = ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[4], &mut rng).unwrap(),
        value: Value::new(2, &[4], &mut rng).unwrap(),
    };
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(), env]).unwrap(), 2).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let mut calls = 0;
    let report = trainer
        .train(
            &mut learner,
            Budget::vector_steps(4),
            Some(&mut |_| calls += 1),
        )
        .unwrap();
    assert_eq!(report.counters.vector_steps, 4);
    assert_eq!(report.counters.env_steps, 8);
    assert_eq!(calls, 2);
    let schedule = Schedule::Linear {
        counter: Counter::EnvironmentTransitions,
        start: 1.,
        end: 0.,
        duration: 8,
    };
    assert_eq!(schedule.value(&report.counters).unwrap(), 0.);
}

#[test]
fn ppo_update_settings_can_follow_rollout_boundary_schedule() {
    use rl::nn::to_vec;
    let env = Lqg::new(
        LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        0,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let models = ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[4], &mut rng).unwrap(),
        value: Value::new(2, &[4], &mut rng).unwrap(),
    };
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(), env]).unwrap(), 2).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let schedule = Schedule::Linear {
        counter: Counter::EnvironmentTransitions,
        start: 0.001,
        end: 0.,
        duration: 8,
    };
    for expected in [4, 8] {
        let mut next = *trainer.counters();
        next.env_steps += 4;
        let rate = schedule.value(&next).unwrap();
        learner
            .set_update_settings(PpoUpdateSettings {
                policy_learning_rate: rate,
                value_learning_rate: rate,
                clip: 0.1,
            })
            .unwrap();
        let before = to_vec(learner.value().network.layers[0].weight.val()).unwrap();
        let report = trainer
            .train(&mut learner, Budget::vector_steps(2), None)
            .unwrap();
        assert_eq!(report.counters.env_steps, expected);
        assert_eq!(learner.updates(), expected / 4);
        let after = to_vec(learner.value().network.layers[0].weight.val()).unwrap();
        assert_eq!(before == after, rate == 0.);
    }
}

#[test]
fn trainer_rejects_double_advantage_normalization_before_collecting() {
    let env = Lqg::new(
        LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        0,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let models = ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[4], &mut rng).unwrap(),
        value: Value::new(2, &[4], &mut rng).unwrap(),
    };
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            ..Default::default()
        },
        1,
    )
    .unwrap()
    .with_minibatch_advantage_normalization(true);
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()]).unwrap(), 2).unwrap();
    let mut trainer = OnPolicyTrainer::new(collector, OnPolicyConfig::default()).unwrap();
    let error = trainer
        .train(&mut learner, Budget::vector_steps(1), None)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("disable rollout advantage normalization")
    );
    assert_eq!(trainer.counters().env_steps, 0);
    assert_eq!(learner.updates(), 0);

    let collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 2).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            normalize_advantages: false,
            rollout_steps: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        trainer
            .train(&mut learner, Budget::vector_steps(1), None)
            .unwrap()
            .counters
            .env_steps,
        1
    );
}

#[test]
fn on_policy_update_failure_retains_successful_collection_counts() {
    use rl::{
        core::{
            actor::{Actor, OnPolicyActorRef},
            learner::{Learner, OnPolicyLearner, UpdateReport},
        },
        data::rollout::OnPolicyBatch,
    };
    struct FailingLearner {
        models: ActorCritic,
        attempts: usize,
    }
    impl Learner for FailingLearner {
        type Batch = OnPolicyBatch;
        fn actor(&self) -> &dyn Actor {
            &self.models
        }
        fn update(&mut self, _: &OnPolicyBatch) -> rl::Result<UpdateReport> {
            self.attempts += 1;
            Err(rl::Error::State("intentional learner failure".into()))
        }
    }
    impl OnPolicyLearner for FailingLearner {
        fn collection_actor(&self) -> OnPolicyActorRef<'_> {
            OnPolicyActorRef::Bootstrapped(&self.models)
        }
    }
    let env = Lqg::new(
        LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        0,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut learner = FailingLearner {
        models: ActorCritic {
            policy: Policy::new(2, env.action_space().clone(), &[4], &mut rng).unwrap(),
            value: Value::new(2, &[4], &mut rng).unwrap(),
        },
        attempts: 0,
    };
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(), env]).unwrap(), 2).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let error = trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap_err();
    assert!(error.to_string().contains("intentional learner failure"));
    assert_eq!(learner.attempts, 1);
    assert_eq!(trainer.counters().env_steps, 4);
    assert_eq!(trainer.counters().vector_steps, 2);
    assert_eq!(trainer.counters().updates, 0);
}

#[test]
fn off_policy_update_failure_retains_collection_and_replay() {
    use rl::{
        core::{
            actor::Actor,
            learner::{Learner, OffPolicyLearner, UpdateReport},
        },
        data::replay::ReplayBatch,
        train::{
            config::OffPolicyConfig,
            trainer::{OffPolicyTrainer, RandomActor},
        },
    };
    struct FailingLearner {
        actor: RandomActor,
        attempts: usize,
    }
    impl Learner for FailingLearner {
        type Batch = ReplayBatch;
        fn actor(&self) -> &dyn Actor {
            &self.actor
        }
        fn update(&mut self, _: &ReplayBatch) -> rl::Result<UpdateReport> {
            self.attempts += 1;
            Err(rl::Error::State("intentional learner failure".into()))
        }
    }
    impl OffPolicyLearner for FailingLearner {
        fn set_exploration(&mut self, _: f32) -> rl::Result<()> {
            Ok(())
        }
    }
    let env = Lqg::new(
        LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0., 2.).unwrap(),
        0,
    )
    .unwrap();
    let mut learner = FailingLearner {
        actor: RandomActor {
            space: env.action_space().clone(),
        },
        attempts: 0,
    };
    let collector =
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(), env]).unwrap(), 2).unwrap();
    let mut trainer = OffPolicyTrainer::new(
        collector,
        OffPolicyConfig {
            warmup_transitions: 0,
            replay_capacity: 8,
            batch_size: 2,
            ..Default::default()
        },
        3,
    )
    .unwrap();
    let error = trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap_err();
    assert!(error.to_string().contains("intentional learner failure"));
    assert_eq!(learner.attempts, 1);
    assert_eq!(trainer.counters().env_steps, 2);
    assert_eq!(trainer.counters().vector_steps, 1);
    assert_eq!(trainer.counters().updates, 0);
    assert_eq!(trainer.replay().len(), 2);
}
