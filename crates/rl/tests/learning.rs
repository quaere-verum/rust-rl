#![cfg(feature = "neural")]

//! Bounded multi-seed acceptance suite, explicitly separate from fast component tests.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        OptimizerConfig,
        d4pg::{D4pg, D4pgConfig},
        ddpg::{Ddpg, DdpgConfig},
        ddqn::{Ddqn, DdqnConfig},
        md4pg::{Md4pg, Md4pgConfig},
        ppo::{Ppo, PpoConfig},
        reinforce::{Reinforce, ReinforceConfig},
        td3::{Td3, Td3Config},
        vmpo::{Vmpo, VmpoConfig},
    },
    collect::{OffPolicyCollector, OnPolicyCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        learner::{OffPolicyLearner, OnPolicyLearner},
    },
    data::{replay::ReplayBatch, rollout::OnPolicyBatch},
    nn::{
        critics::scalar::{DiscreteQ, QValue, Value},
        policy::{ActorCritic, Deterministic, Policy},
    },
    train::{
        config::{Budget, OffPolicyConfig, OnPolicyConfig},
        trainer::{OffPolicyTrainer, OnPolicyTrainer},
    },
};
#[derive(Clone)]
struct Bandit {
    space: ActionSpace,
}
impl Bandit {
    fn observation(&self) -> Observation {
        Observation::dense(
            vec![1.],
            if matches!(self.space, ActionSpace::Discrete { .. }) {
                Some(vec![true, true, false])
            } else {
                None
            },
        )
    }
}
impl Env for Bandit {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, a: &Action) -> Result<Step> {
        self.space.check(a, self.observation().dense_mask())?;
        let reward = match a {
            Action::Discrete(a) => {
                if *a == 1 {
                    1.
                } else {
                    -1.
                }
            }
            Action::Continuous(a) => -(a[0] - 0.6).powi(2),
            Action::Conditional(_) => unreachable!("validated fixed-width action"),
        };
        Ok(Step {
            observation: self.observation(),
            reward: reward.into(),
            terminated: true,
            truncated: false,
        })
    }
}
fn score(actor: &dyn Actor, env: &Bandit) -> f32 {
    let a = actor
        .act(
            &[env.observation()],
            ActMode::Deterministic,
            &mut ChaCha8Rng::seed_from_u64(0),
        )
        .unwrap()
        .remove(0);
    env.clone().step(&a).unwrap().reward.total()
}
fn optimizer() -> OptimizerConfig {
    OptimizerConfig {
        learning_rate: 0.003,
        max_grad_norm: Some(1.),
    }
}
#[test]
#[ignore = "bounded multi-seed learning acceptance; run explicitly"]
fn on_policy_families_learn_discrete_and_continuous_tasks() {
    for seed in [7, 19] {
        for discrete in [true, false] {
            for name in ["ppo", "reinforce", "vmpo"] {
                let start = std::time::Instant::now();
                let env = Bandit {
                    space: if discrete {
                        ActionSpace::Discrete { count: 3 }
                    } else {
                        ActionSpace::Box {
                            low: vec![-1.],
                            high: vec![1.],
                        }
                    },
                };
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let models = ActorCritic {
                    policy: Policy::new(1, env.space.clone(), &[8], &mut rng).unwrap(),
                    value: Value::new(1, &[8], &mut rng).unwrap(),
                };
                let mut learner: Box<dyn OnPolicyLearner<Batch = OnPolicyBatch>> = match name {
                    "ppo" => Box::new(
                        Ppo::new(
                            models,
                            PpoConfig {
                                policy: optimizer(),
                                value: optimizer(),
                                epochs: 2,
                                minibatch_size: 32,
                                ..Default::default()
                            },
                            seed,
                        )
                        .unwrap(),
                    ),
                    "reinforce" => Box::new(
                        Reinforce::new(
                            models,
                            ReinforceConfig {
                                policy: optimizer(),
                                value: optimizer(),
                                ..Default::default()
                            },
                            seed,
                        )
                        .unwrap(),
                    ),
                    _ => Box::new(
                        Vmpo::new(
                            models,
                            VmpoConfig {
                                policy: optimizer(),
                                value: optimizer(),
                                epochs: 2,
                                minibatch_size: 32,
                                ..Default::default()
                            },
                            seed,
                        )
                        .unwrap(),
                    ),
                };
                let collector = OnPolicyCollector::new(
                    SyncVectorEnv::new(vec![env.clone(); 32]).unwrap(),
                    seed,
                )
                .unwrap();
                let config = OnPolicyConfig {
                    rollout_steps: 1,
                    episode_guard: if name == "reinforce" { Some(1) } else { None },
                    gae_lambda: None,
                    ..Default::default()
                };
                let mut trainer = OnPolicyTrainer::new(collector, config).unwrap();
                trainer
                    .train(learner.as_mut(), Budget::vector_steps(160), None)
                    .unwrap();
                let score = score(learner.actor(), &env);
                eprintln!(
                    "{name} discrete={discrete} seed={seed}: score={score:.4}, {:.2}s",
                    start.elapsed().as_secs_f64()
                );
                assert!(
                    score > if discrete { 0.9 } else { -0.1 },
                    "{name}, seed {seed}, score {score}"
                );
            }
        }
    }
}
#[test]
#[ignore = "bounded multi-seed learning acceptance; run explicitly"]
fn off_policy_families_learn_terminal_reward_tasks() {
    for seed in [7, 19] {
        for name in ["ddqn", "ddpg", "td3", "d4pg", "md4pg"] {
            let start = std::time::Instant::now();
            let discrete = name == "ddqn";
            let env = Bandit {
                space: if discrete {
                    ActionSpace::Discrete { count: 3 }
                } else {
                    ActionSpace::Box {
                        low: vec![-1.],
                        high: vec![1.],
                    }
                },
            };
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut learner: Box<dyn OffPolicyLearner<Batch = ReplayBatch>> = if discrete {
                Box::new(
                    Ddqn::new(
                        DiscreteQ::new(1, 3, &[16], &mut rng).unwrap(),
                        DdqnConfig {
                            optimizer: optimizer(),
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                )
            } else {
                let p = Deterministic::new(1, env.space.clone(), &[16], &mut rng).unwrap();
                let q = QValue::new(
                    1,
                    1,
                    if name == "d4pg" { 8 } else { 1 },
                    &[16, 16],
                    &mut rng,
                )
                .unwrap();
                let q2 = QValue::new(1, 1, 1, &[16, 16], &mut rng).unwrap();
                let training = DdpgConfig {
                    actor: OptimizerConfig {
                        learning_rate: 0.001,
                        ..Default::default()
                    },
                    critic: optimizer(),
                    ..Default::default()
                };
                match name {
                    "ddpg" => Box::new(Ddpg::new(p, q, training, seed).unwrap()),
                    "td3" => Box::new(
                        Td3::new(
                            p,
                            q,
                            q2,
                            Td3Config {
                                training: DdpgConfig {
                                    policy_delay: 2,
                                    target_noise: 0.2,
                                    ..training
                                },
                            },
                            seed,
                        )
                        .unwrap(),
                    ),
                    "d4pg" => {
                        let mut c = D4pgConfig::new(8).unwrap();
                        c.training = training;
                        Box::new(D4pg::new(p, q, c, seed).unwrap())
                    }
                    _ => Box::new(
                        Md4pg::new(
                            p,
                            q,
                            q2,
                            Md4pgConfig {
                                training,
                                risk: rl::objectives::risk::Risk::Expected,
                            },
                            seed,
                        )
                        .unwrap(),
                    ),
                }
            };
            let collector =
                OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone()]).unwrap(), seed)
                    .unwrap();
            let mut trainer = OffPolicyTrainer::new(
                collector,
                OffPolicyConfig {
                    warmup_transitions: 128,
                    replay_capacity: 1024,
                    batch_size: 64,
                    exploration: rl::train::schedule::Schedule::Constant(0.25),
                    ..Default::default()
                },
                seed,
            )
            .unwrap();
            trainer
                .train(learner.as_mut(), Budget::vector_steps(1000), None)
                .unwrap();
            let score = score(learner.actor(), &env);
            eprintln!(
                "{name} seed={seed}: score={score:.4}, {:.2}s",
                start.elapsed().as_secs_f64()
            );
            assert!(
                score > if discrete { 0.9 } else { -0.1 },
                "{name}, seed {seed}, score {score}"
            );
        }
    }
}
