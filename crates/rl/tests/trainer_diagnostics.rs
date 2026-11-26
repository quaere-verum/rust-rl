#![cfg(feature = "neural")]

//! Read-only access to the actual retained collector batch enables independent
//! target diagnostics without a second collection/training implementation.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::ppo::{Ppo, PpoConfig},
    collect::{OnPolicyCollector, vector_env::SyncVectorEnv},
    envs::delayed_bit::DelayedBit,
    nn::{
        critics::scalar::Value,
        policy::{ActorCritic, Policy},
    },
    prelude::*,
    train::{
        config::{Budget, OnPolicyConfig},
        trainer::OnPolicyTrainer,
    },
};

#[test]
fn retained_diagnostic_views_match_independent_gae_without_mutating_runtime() {
    let mut rng = ChaCha8Rng::seed_from_u64(71);
    let models = ActorCritic {
        policy: Policy::new(5, ActionSpace::Discrete { count: 2 }, &[4], &mut rng).unwrap(),
        value: Value::new(5, &[4], &mut rng).unwrap(),
    };
    let frozen = models.clone();
    let mut learner = Ppo::new(
        models,
        PpoConfig {
            epochs: 1,
            minibatch_size: 8,
            ..Default::default()
        },
        72,
    )
    .unwrap();
    let vector = SyncVectorEnv::new(vec![
        DelayedBit::new(2, None, false, 1).unwrap(),
        DelayedBit::new(3, None, false, 2).unwrap(),
    ])
    .unwrap();
    let mut trainer = OnPolicyTrainer::new(
        OnPolicyCollector::new(vector, 73).unwrap(),
        OnPolicyConfig {
            rollout_steps: 4,
            gamma: 0.9,
            gae_lambda: Some(0.95),
            normalize_advantages: false,
            episode_guard: None,
        },
    )
    .unwrap();
    assert!(trainer.last_batch().is_none());
    let reference_vector = SyncVectorEnv::new(vec![
        DelayedBit::new(2, None, false, 1).unwrap(),
        DelayedBit::new(3, None, false, 2).unwrap(),
    ])
    .unwrap();
    let mut reference = OnPolicyCollector::new(reference_vector, 73).unwrap();
    let mut raw = rl::data::rollout::RolloutBuffer::default();
    reference.collect_steps_into(&frozen, 4, &mut raw).unwrap();
    let chronological = (0..raw.len())
        .map(|i| {
            (
                raw.row(i).unwrap().transition.clone(),
                raw.next_value(i).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let expected_batch = raw.to_batch(0.9, Some(0.95), false).unwrap();
    trainer
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let before = serde_json::to_vec(&trainer).unwrap();
    let batch = trainer.last_batch().unwrap();
    assert_eq!(
        serde_json::to_value(batch).unwrap(),
        serde_json::to_value(expected_batch).unwrap()
    );
    assert_eq!(batch.len(), 8);
    let targets = batch.bootstrapped().unwrap();
    for lane in 0..2 {
        let mut future = 0f64;
        for tick in (0..4).rev() {
            let index = tick * 2 + lane;
            let (step, next_value) = &chronological[index];
            let bootstrap = if step.terminated {
                0.
            } else {
                f64::from(*next_value)
            };
            let delta = f64::from(step.reward.total()) + f64::from(0.9f32) * bootstrap
                - f64::from(targets.values[index]);
            let continuation = if step.terminated || step.truncated {
                0.
            } else {
                future
            };
            future = delta + f64::from(0.9f32) * f64::from(0.95f32) * continuation;
            assert!((f64::from(targets.advantages[index]) - future).abs() < 2e-6);
            assert!(
                (f64::from(batch.returns[index]) - future - f64::from(targets.values[index])).abs()
                    < 2e-6
            );
        }
    }
    assert_eq!(serde_json::to_vec(&trainer).unwrap(), before);
}

#[test]
fn stateful_environment_inspection_does_not_change_lanes_rng_or_memory() {
    use rl::{
        collect::stateful::StatefulCollector,
        core::{
            actor::{ActMode, Actor},
            env::ObservationBatch,
            stateful::StatelessActor,
        },
    };
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    struct Fixed {
        space: ActionSpace,
    }
    impl Actor for Fixed {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }
        fn act_into(
            &self,
            obs: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            out: &mut Vec<Action>,
        ) -> rl::Result<()> {
            out.resize(obs.len(), Action::Discrete(0));
            Ok(())
        }
    }
    let env = SyncVectorEnv::new(vec![
        DelayedBit::new(2, None, false, 1).unwrap(),
        DelayedBit::new(3, None, false, 2).unwrap(),
    ])
    .unwrap();
    let actor = StatelessActor {
        actor: Fixed {
            space: ActionSpace::Discrete { count: 2 },
        },
        width: 5,
        revision: 0,
    };
    let collector = StatefulCollector::new(env, actor, 91).unwrap();
    let before = serde_json::to_vec(&collector).unwrap();
    let seen = collector.environment().environments();
    assert_eq!(seen.len(), 2);
    assert_eq!(serde_json::to_value(&seen[0]).unwrap()["delay"], 2);
    assert_eq!(serde_json::to_value(&seen[1]).unwrap()["delay"], 3);
    assert_eq!(serde_json::to_vec(&collector).unwrap(), before);
}
