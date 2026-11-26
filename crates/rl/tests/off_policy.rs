#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        d4pg::{D4pg, D4pgConfig},
        ddpg::{Ddpg, DdpgConfig},
        ddqn::{Ddqn, DdqnConfig},
        md4pg::{Md4pg, Md4pgConfig},
        td3::{Td3, Td3Config},
    },
    core::{
        action::{Action, ActionSpace},
        env::Observation,
        learner::Learner,
        transition::Transition,
    },
    data::replay::{ReplayBatch, ReplayItem},
    nn::{
        critics::scalar::{DiscreteQ, QValue},
        policy::Deterministic,
    },
};
fn terminal(action: Action) -> ReplayBatch {
    ReplayBatch {
        items: vec![ReplayItem {
            transition: Transition {
                observation: Observation::new(vec![1.]),
                action,
                reward: 2.0_f32.into(),
                next_observation: Observation::new(vec![0.]),
                terminated: true,
                truncated: false,
            },
            discount: 0.,
        }],
    }
}
#[test]
fn continuous_families_learn_terminal_rewards_and_preserve_td3_delay() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let p = Deterministic::new(1, space, &[8], &mut rng).unwrap();
    let q = QValue::new(1, 1, 1, &[8], &mut rng).unwrap();
    let q2 = QValue::new(1, 1, 1, &[8], &mut rng).unwrap();
    let b = terminal(Action::Continuous(vec![0.]));
    let mut d = Ddpg::new(p.clone(), q.clone(), DdpgConfig::default(), 5).unwrap();
    assert!(d.update(&b).unwrap().metrics["critic_loss"] > 0.);
    let mut t = Td3::new(p.clone(), q.clone(), q2.clone(), Td3Config::default(), 5).unwrap();
    t.update(&b).unwrap();
    assert_eq!(t.actor_updates(), 0);
    t.update(&b).unwrap();
    assert_eq!(t.actor_updates(), 1);
    t.update(&b).unwrap();
    assert_eq!(t.actor_updates(), 1);
    let quantiles = QValue::new(1, 1, 8, &[8], &mut rng).unwrap();
    let mut z = D4pg::new(p.clone(), quantiles, D4pgConfig::new(8).unwrap(), 5).unwrap();
    assert!(z.update(&b).unwrap().metrics["critic_loss"] > 0.);
    let mut m = Md4pg::new(p, q, q2, Md4pgConfig::default(), 5).unwrap();
    assert!(m.update(&b).unwrap().metrics["critic_loss"] > 0.);
}
#[test]
fn ddqn_does_not_select_an_action_for_terminal_next_states() {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let q = DiscreteQ::new(1, 2, &[4], &mut rng).unwrap();
    let mut d = Ddqn::new(q, DdqnConfig::default()).unwrap();
    let mut b = terminal(Action::Discrete(0));
    b.items[0]
        .transition
        .next_observation
        .as_dense_mut()
        .unwrap()
        .action_mask = Some(vec![false, false]);
    assert_eq!(d.update(&b).unwrap().updates, 1);
    b.items[0].transition.terminated = false;
    b.items[0].discount = 0.9;
    assert!(d.update(&b).is_err());
}

#[test]
fn double_q_selects_online_and_evaluates_target_with_masks() {
    use burn::{module::Param, tensor::Tensor};
    use rl::{core::checkpoint::Checkpoint, nn::checkpoint::RecordBurnExt};
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut q = DiscreteQ::new(1, 3, &[], &mut rng).unwrap();
    q.network.layers[0].weight = Param::from_tensor(Tensor::zeros([1, 3], &rl::nn::device()));
    q.network.layers[0].bias = Some(Param::from_tensor(rl::nn::vector(&[1., 3., 100.])));
    let mut learner = Ddqn::new(q, Default::default()).unwrap();
    let mut target = learner.target();
    target.network.layers[0].bias = Some(Param::from_tensor(rl::nn::vector(&[10., 20., 999.])));
    let mut record = learner.checkpoint().unwrap();
    record.model("target", &target).unwrap();
    learner.restore(&record).unwrap();
    let mut b = terminal(Action::Discrete(0));
    let item = &mut b.items[0];
    item.transition.terminated = false;
    item.discount = 0.5;
    item.transition
        .next_observation
        .as_dense_mut()
        .unwrap()
        .action_mask = Some(vec![true, true, false]);
    // Target = 2 + .5 * 20 = 12; prediction = 1; unit Huber loss = 11 - .5.
    assert!((learner.update(&b).unwrap().metrics["critic_loss"] - 10.5).abs() < 1e-6);
}

#[test]
fn zero_target_smoothing_does_not_advance_the_owned_rng() {
    use rl::train::checkpoint::Checkpoint;
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let critic = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let mut learner = Ddpg::new(policy, critic, Default::default(), 43).unwrap();
    let mut batch = terminal(Action::Continuous(vec![0.]));
    batch.items[0].transition.terminated = false;
    batch.items[0].discount = 0.9;
    let before = learner.checkpoint().unwrap().metadata["rng"].clone();
    learner.update(&batch).unwrap();
    assert_eq!(learner.checkpoint().unwrap().metadata["rng"], before);
}

#[test]
fn quantile_restore_rejects_incompatible_or_incomplete_state_atomically() {
    use rl::{core::checkpoint::Checkpoint, objectives::quantile::QuantileLoss};
    let mut rng = ChaCha8Rng::seed_from_u64(101);
    let policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let critic = QValue::new(1, 1, 4, &[], &mut rng).unwrap();
    let mut config = D4pgConfig::new(4).unwrap();
    config.loss = QuantileLoss::Generalized {
        scale: 1.,
        ema: 0.5,
        minimum: 0.001,
    };
    let mut learner = D4pg::new(policy, critic, config, 103).unwrap();
    let batch = terminal(Action::Continuous(vec![0.]));
    learner.update(&batch).unwrap();
    let record = learner.checkpoint().unwrap();
    let before = serde_json::to_value(&record).unwrap();
    let mut bad_grid = record.clone();
    bad_grid.metadata["critic"]["grid"]["edges"] = serde_json::json!([0., 0.1, 0.5, 0.75, 1.]);
    let mut bad_scale = record.clone();
    bad_scale.metadata["critic"]["loss"]["Generalized"]["scale"] = serde_json::json!(-1.);
    let mut missing_target = record.clone();
    missing_target.models.remove("target_critic");
    let mut missing_optimizer = record.clone();
    missing_optimizer.optimizers.remove("critic");
    let mut wrong_family = record.clone();
    wrong_family.algorithm = "td3".into();
    for corrupted in [
        bad_grid,
        bad_scale,
        missing_target,
        missing_optimizer,
        wrong_family,
    ] {
        assert!(learner.restore(&corrupted).is_err());
        assert_eq!(
            serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
            before
        );
    }
    // Recreated transient scratch does not change the following adaptation or optimizer step.
    let expected = learner.update(&batch).unwrap();
    let expected_record = learner.checkpoint().unwrap();
    learner.restore(&record).unwrap();
    let actual = learner.update(&batch).unwrap();
    assert_eq!(actual.metrics, expected.metrics);
    assert_eq!(
        learner.checkpoint().unwrap().metadata,
        expected_record.metadata
    );
}

#[test]
fn td3_checkpoint_preserves_delayed_actor_update_cadence() {
    use rl::core::checkpoint::Checkpoint;
    let mut rng = ChaCha8Rng::seed_from_u64(111);
    let policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let first = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    let mut learner = Td3::new(policy, first, second, Default::default(), 113).unwrap();
    let mut batch = terminal(Action::Continuous(vec![0.]));
    batch.items[0].transition.terminated = false;
    batch.items[0].discount = 0.9;
    learner.update(&batch).unwrap();
    let record = learner.checkpoint().unwrap();
    let expected = learner.update(&batch).unwrap();
    assert_eq!((learner.updates(), learner.actor_updates()), (2, 1));
    learner.restore(&record).unwrap();
    assert_eq!((learner.updates(), learner.actor_updates()), (1, 0));
    let actual = learner.update(&batch).unwrap();
    assert_eq!(actual.metrics, expected.metrics);
    assert_eq!((learner.updates(), learner.actor_updates()), (2, 1));
}
