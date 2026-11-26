#![cfg(feature = "neural")]

use burn::module::Param;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::continuous::ContinuousConfig,
    core::transition::Transition,
    data::replay::ReplayItem,
    nn::{matrix, vector},
    prelude::*,
};
fn constant(q: &mut QValue, values: &[f32]) {
    let dims = q.network.layers[0].weight.val().dims();
    q.network.layers[0].weight =
        Param::from_tensor(matrix(vec![0.; dims[0] * dims[1]], dims[0], dims[1]));
    q.network.layers[0].bias = Some(Param::from_tensor(vector(values)));
}
#[test]
fn continuing_scalar_twin_quantile_and_moment_targets_match_independent_values() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let p = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let mut q = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    constant(&mut q, &[3.]);
    let b = ReplayBatch {
        items: vec![ReplayItem {
            transition: Transition {
                observation: Observation::new(vec![1.]),
                action: Action::Continuous(vec![0.]),
                reward: 2.0_f32.into(),
                next_observation: Observation::new(vec![1.]),
                terminated: false,
                truncated: false,
            },
            discount: 0.5,
        }],
    };
    let mut d = Ddpg::new(p.clone(), q.clone(), Default::default(), 0).unwrap();
    assert!((d.update(&b).unwrap().metrics["critic_loss"] - 0.25).abs() < 1e-6);
    let mut second = q.clone();
    constant(&mut second, &[1.]);
    let mut t = Td3::new(p.clone(), q.clone(), second, Default::default(), 0).unwrap();
    assert!((t.update(&b).unwrap().metrics["critic_loss"] - 2.5).abs() < 1e-6);
    let mut quantile = QValue::new(1, 1, 4, &[], &mut rng).unwrap();
    constant(&mut quantile, &[3.; 4]);
    let mut z = D4pg::new(p.clone(), quantile, D4pgConfig::new(4).unwrap(), 0).unwrap();
    assert!((z.update(&b).unwrap().metrics["critic_loss"] - 0.0625).abs() < 1e-6);
    let mut second = q.clone();
    constant(&mut second, &[13.]);
    let mut m = Md4pg::new(p, q, second, Default::default(), 0).unwrap();
    let (mean, second) =
        rl::objectives::bellman::moment_targets(&[2.], &[0.5], &[3.], &[13.]).unwrap();
    assert_eq!(mean, [3.5]);
    assert_eq!(second, [13.25]);
    assert!(
        (m.update(&b).unwrap().metrics["critic_loss"] - (0.5_f64.powi(2) + 0.25_f64.powi(2))).abs()
            < 1e-6
    );
}

#[test]
fn mixed_terminal_continuing_and_nstep_targets_match_hand_calculated_losses() {
    let mut rng = ChaCha8Rng::seed_from_u64(73);
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
    let mut mean = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    constant(&mut mean, &[3.]);
    let mut second = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
    constant(&mut second, &[1.]);
    let batch = ReplayBatch {
        items: [(2., 0.), (1., 0.5), (-1., 0.125)]
            .into_iter()
            .map(|(reward, discount)| ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![1.]),
                    action: Action::Continuous(vec![0.]),
                    reward: reward.into(),
                    next_observation: Observation::new(vec![1.]),
                    terminated: discount == 0.,
                    truncated: false,
                },
                discount,
            })
            .collect(),
    };
    let mut scalar = Ddpg::new(policy.clone(), mean.clone(), Default::default(), 0).unwrap();
    assert!((scalar.update(&batch).unwrap().metrics["critic_loss"] - 4.796875).abs() < 1e-6);
    let mut twin = Td3::new(
        policy.clone(),
        mean.clone(),
        second.clone(),
        Default::default(),
        0,
    )
    .unwrap();
    assert!((twin.update(&batch).unwrap().metrics["critic_loss"] - 7.677083333333333).abs() < 2e-6);
    let mut quantiles = QValue::new(1, 1, 4, &[], &mut rng).unwrap();
    constant(&mut quantiles, &[3.; 4]);
    let mut quantile =
        D4pg::new(policy.clone(), quantiles, D4pgConfig::new(4).unwrap(), 0).unwrap();
    assert!((quantile.update(&batch).unwrap().metrics["critic_loss"] - 0.625).abs() < 1e-6);
    constant(&mut second, &[13.]);
    let mut moments = Md4pg::new(policy, mean, second, Default::default(), 0).unwrap();
    // Raw second targets are [4, 7.25, 0.453125], with predictions 13.
    assert!(
        (moments.update(&batch).unwrap().metrics["critic_loss"] - 95.29239908854167).abs() < 1e-5
    );
}

#[derive(burn::module::Module, Debug)]
struct BufferedCritic {
    weight: Param<burn::tensor::Tensor<2>>,
    running: burn::module::RunningState<burn::tensor::Tensor<1>>,
}
impl rl::nn::critics::scalar::ActionValue for BufferedCritic {
    fn action_dim(&self) -> usize {
        1
    }
    fn output_dim(&self) -> usize {
        self.weight.val().dims()[1]
    }
    fn values(
        &self,
        _: ObservationBatch<'_>,
        actions: burn::tensor::Tensor<2>,
    ) -> rl::Result<burn::tensor::Tensor<2>> {
        self.running.update(self.running.value_sync() + 1.);
        Ok(actions.matmul(self.weight.val()))
    }
}
fn buffered(width: usize) -> BufferedCritic {
    BufferedCritic {
        weight: Param::from_tensor(matrix(vec![1.; width], 1, width)),
        running: burn::module::RunningState::new(vector(&[0.])),
    }
}

#[test]
fn every_rule_preserves_actor_gradients_and_isolates_running_state() {
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let mut policy = Deterministic::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.], 1, 1));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.])));
    let observations = [Observation::new(vec![1.])];
    let batch = ReplayBatch {
        items: vec![ReplayItem {
            transition: Transition {
                observation: observations[0].clone(),
                action: Action::Continuous(vec![0.]),
                reward: 2.0_f32.into(),
                next_observation: observations[0].clone(),
                terminated: false,
                truncated: false,
            },
            discount: 0.5,
        }],
    };
    macro_rules! check {
        ($learner:expr) => {{
            let mut learner = $learner;
            let before = learner
                .policy()
                .actions((&observations).into())
                .unwrap()
                .into_scalar::<f32>();
            learner.update(&batch).unwrap();
            let after = learner
                .policy()
                .actions((&observations).into())
                .unwrap()
                .into_scalar::<f32>();
            assert!(
                after > before,
                "actor gradient did not pass through frozen critic"
            );
            assert_eq!(
                learner.critic().running.value_sync().into_scalar::<f32>(),
                1.
            );
            assert_eq!(
                learner
                    .target_critic()
                    .running
                    .value_sync()
                    .into_scalar::<f32>(),
                0.
            );
            learner.critic().running.update(vector(&[99.]));
            assert_eq!(
                learner.critic().running.value_sync().into_scalar::<f32>(),
                1.
            );
            learner
        }};
    }
    let training = ContinuousConfig {
        tau: 0.,
        policy_delay: 1,
        target_noise: 0.,
        ..Default::default()
    };
    check!(Ddpg::new(policy.clone(), buffered(1), training.clone(), 0).unwrap());
    let twin = check!(
        Td3::new(
            policy.clone(),
            buffered(1),
            buffered(1),
            Td3Config {
                training: training.clone()
            },
            0
        )
        .unwrap()
    );
    assert_eq!(
        twin.second_critic()
            .running
            .value_sync()
            .into_scalar::<f32>(),
        1.
    );
    assert_eq!(
        twin.target_second_critic()
            .running
            .value_sync()
            .into_scalar::<f32>(),
        0.
    );
    let mut config = D4pgConfig::new(4).unwrap();
    config.training = training.clone();
    check!(D4pg::new(policy.clone(), buffered(4), config, 0).unwrap());
    let moments = check!(
        Md4pg::new(
            policy,
            buffered(1),
            buffered(1),
            Md4pgConfig {
                training,
                risk: rl::objectives::risk::Risk::MeanVariance { penalty: 0.5 }
            },
            0
        )
        .unwrap()
    );
    assert_eq!(
        moments
            .second_critic()
            .running
            .value_sync()
            .into_scalar::<f32>(),
        1.
    );
    assert_eq!(
        moments
            .target_second_critic()
            .running
            .value_sync()
            .into_scalar::<f32>(),
        0.
    );
}
