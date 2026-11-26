#![cfg(feature = "neural")]

//! Fast exact multistep fixtures plus explicitly invoked, bounded neural cases.
mod support;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
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
        action::Action,
        env::{Env, TimeLimit},
        learner::{OffPolicyLearner, OnPolicyLearner},
        transition::Transition,
    },
    data::{
        replay::{NStepAccumulator, ReplayBatch},
        returns::discounted_returns,
        rollout::OnPolicyBatch,
    },
    nn::{
        critics::scalar::{DiscreteQ, QValue, Value},
        policy::{ActorCritic, Deterministic, Policy},
    },
    train::{
        config::{Budget, OffPolicyConfig, OnPolicyConfig},
        schedule::Schedule,
        trainer::{OffPolicyTrainer, OnPolicyTrainer},
    },
};
use std::time::Instant;
use support::{DelayedControl, MAX_TRANSITIONS, MAX_UPDATES, MaskedChain, SEEDS};

#[test]
fn four_step_delayed_reward_has_exact_returns_and_n1_n3_targets() {
    let mut env = DelayedControl::new();
    let mut observation = env.reset(None).unwrap().observation;
    let mut transitions = Vec::new();
    for _ in 0..4 {
        let action = Action::Continuous(vec![1.]);
        let step = env.step(&action).unwrap();
        transitions.push(Transition {
            observation,
            action,
            reward: step.reward.clone(),
            next_observation: step.observation.clone(),
            terminated: step.terminated,
            truncated: step.truncated,
        });
        observation = step.observation;
    }
    let rewards: Vec<_> = transitions.iter().map(|step| step.reward.total()).collect();
    let returns = discounted_returns(
        &rewards,
        &[0.; 4],
        &[false, false, false, true],
        &[false; 4],
        0.5,
    )
    .unwrap();
    for (actual, expected) in returns.iter().zip([-0.02, -0.04, -0.08, -0.16]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (n, expected_rewards, expected_discounts) in [
        (1, [0., 0., 0., -0.16], [0.5, 0.5, 0.5, 0.]),
        (3, [0., -0.04, -0.08, -0.16], [0.125, 0., 0., 0.]),
    ] {
        let mut pending = NStepAccumulator::new(n, 0.5, 1).unwrap();
        let mut output = Vec::new();
        for transition in &transitions {
            output.extend(pending.push(0, transition.clone()).unwrap());
        }
        assert_eq!(output.len(), 4);
        for ((item, reward), discount) in
            output.iter().zip(expected_rewards).zip(expected_discounts)
        {
            assert!((item.transition.reward.total() - reward).abs() < 1e-6);
            assert_eq!(item.discount, discount);
        }
    }
}

#[test]
fn an_interior_truncation_flushes_without_crossing_the_next_longer_episode() {
    let mut short = TimeLimit::new(DelayedControl::new(), 2).unwrap();
    let mut long = DelayedControl::new();
    let mut pending = NStepAccumulator::new(3, 0.5, 1).unwrap();
    let mut outputs = Vec::new();
    for env in [&mut short as &mut dyn Env, &mut long as &mut dyn Env] {
        let mut observation = env.reset(None).unwrap().observation;
        loop {
            let action = Action::Continuous(vec![1.]);
            let step = env.step(&action).unwrap();
            outputs.extend(
                pending
                    .push(
                        0,
                        Transition {
                            observation,
                            action,
                            reward: step.reward.clone(),
                            next_observation: step.observation.clone(),
                            terminated: step.terminated,
                            truncated: step.truncated,
                        },
                    )
                    .unwrap(),
            );
            if step.ended() {
                break;
            }
            observation = step.observation;
        }
    }
    assert_eq!(outputs.len(), 6);
    assert!(
        outputs[..2]
            .iter()
            .all(|item| item.transition.truncated && !item.transition.terminated)
    );
    assert_eq!(outputs[0].discount, 0.25);
    assert_eq!(outputs[1].discount, 0.5);
    assert_eq!(
        outputs[0].transition.next_observation.dense_values(),
        [0.5, 0.5]
    );
    assert_eq!(outputs[2].transition.observation.dense_values(), [0., 0.]);
    assert!((outputs[5].transition.reward.total() + 0.16).abs() < 1e-6);
    let returns = discounted_returns(
        &[0., 0., 0., 0., 0., -0.16],
        &[0., 2., 0., 0., 0., 0.],
        &[false, false, false, false, false, true],
        &[false, true, false, false, false, false],
        0.5,
    )
    .unwrap();
    for (actual, expected) in returns.iter().zip([0.5, 1., -0.02, -0.04, -0.08, -0.16]) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

fn optimizer(rate: f64) -> OptimizerConfig {
    OptimizerConfig {
        learning_rate: rate,
        max_grad_norm: Some(1.),
    }
}

fn off_policy(
    name: &str,
    seed: u64,
    width: usize,
    space: rl::core::action::ActionSpace,
) -> Box<dyn OffPolicyLearner<Batch = ReplayBatch>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    if name == "ddqn" {
        return Box::new(
            Ddqn::new(
                DiscreteQ::new(width, 3, &[16, 16], &mut rng).unwrap(),
                DdqnConfig {
                    optimizer: optimizer(0.003),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
    }
    let policy = Deterministic::new(width, space, &[16], &mut rng).unwrap();
    let critic = QValue::new(
        width,
        1,
        if name == "d4pg" { 8 } else { 1 },
        &[16, 16],
        &mut rng,
    )
    .unwrap();
    let second = QValue::new(width, 1, 1, &[16, 16], &mut rng).unwrap();
    let training = DdpgConfig {
        actor: optimizer(0.001),
        critic: optimizer(0.003),
        ..Default::default()
    };
    match name {
        "ddpg" => Box::new(Ddpg::new(policy, critic, training, seed).unwrap()),
        "td3" => Box::new(
            Td3::new(
                policy,
                critic,
                second,
                Td3Config {
                    training: DdpgConfig {
                        policy_delay: 2,
                        target_noise: 0.1,
                        ..training
                    },
                },
                seed,
            )
            .unwrap(),
        ),
        "d4pg" => {
            let mut config = D4pgConfig::new(8).unwrap();
            config.training = training;
            Box::new(D4pg::new(policy, critic, config, seed).unwrap())
        }
        "md4pg" => Box::new(
            Md4pg::new(
                policy,
                critic,
                second,
                Md4pgConfig {
                    training,
                    ..Default::default()
                },
                seed,
            )
            .unwrap(),
        ),
        _ => unreachable!(),
    }
}

fn run_off_policy<E: Env + Clone>(name: &str, seed: u64, env: E, pilot: bool) -> bool {
    let start = Instant::now();
    let mut learner = off_policy(
        name,
        seed,
        env.dense_observation_dim().unwrap(),
        env.action_space().clone(),
    );
    let before = support::evaluate(learner.actor(), &env, seed);
    let random = support::random_score(&env, seed);
    let collector =
        OffPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(); 8]).unwrap(), seed).unwrap();
    let mut trainer = OffPolicyTrainer::new(
        collector,
        OffPolicyConfig {
            warmup_transitions: 256,
            replay_capacity: 16_000,
            batch_size: 64,
            n_step: 3,
            gamma: 1.,
            updates_per_transition: 0.25,
            exploration: Schedule::Constant(0.25),
        },
        seed,
    )
    .unwrap();
    let mut last = 0;
    let mut progress = |p: &rl::train::trainer::Progress| {
        assert!(
            start.elapsed().as_secs() < if pilot { 60 } else { 180 },
            "{name} seed={seed} wall-clock cap exceeded"
        );
        if p.counters.updates >= last + 250 {
            last = p.counters.updates;
            eprintln!(
                "progress {name} seed={seed} transitions={} updates={} collection_s={:.3} learning_s={:.3} elapsed_s={:.3}",
                p.counters.env_steps,
                p.counters.updates,
                p.collection_seconds,
                p.learning_seconds,
                start.elapsed().as_secs_f64()
            );
        }
    };
    let stats = trainer
        .train(
            learner.as_mut(),
            Budget {
                env_steps: Some(if pilot { 4096 } else { 12_000 }),
                updates: Some(if pilot { 1000 } else { MAX_UPDATES }),
                ..Default::default()
            },
            Some(&mut progress),
        )
        .unwrap();
    assert!(stats.counters.env_steps <= MAX_TRANSITIONS && stats.counters.updates <= MAX_UPDATES);
    let after = support::evaluate(learner.actor(), &env, seed);
    let success = if name == "ddqn" {
        after > 0.5 && after > before + 0.2 && after > random + 0.2
    } else {
        after > -0.1 && after > before + 0.03 && after > random + 0.1
    };
    eprintln!(
        "case {name} seed={seed} pilot={pilot} initial={before:.6} random={random:.6} final={after:.6} pass={success} transitions={} updates={} elapsed_s={:.3}",
        stats.counters.env_steps,
        stats.counters.updates,
        start.elapsed().as_secs_f64()
    );
    success
}

fn run_on_policy(name: &str, seed: u64, pilot: bool) -> bool {
    let start = Instant::now();
    let env = DelayedControl::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let models = ActorCritic {
        policy: Policy::new(2, env.action_space().clone(), &[16], &mut rng).unwrap(),
        value: Value::new(2, &[16], &mut rng).unwrap(),
    };
    let mut learner: Box<dyn OnPolicyLearner<Batch = OnPolicyBatch>> = match name {
        "ppo" => Box::new(
            Ppo::new(
                models,
                PpoConfig {
                    policy: optimizer(0.003),
                    value: optimizer(0.003),
                    epochs: 2,
                    minibatch_size: 64,
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
                    policy: optimizer(0.003),
                    value: optimizer(0.003),
                    ..Default::default()
                },
                seed,
            )
            .unwrap(),
        ),
        "vmpo" => Box::new(
            Vmpo::new(
                models,
                VmpoConfig {
                    policy: optimizer(0.003),
                    value: optimizer(0.003),
                    epochs: 2,
                    minibatch_size: 64,
                    ..Default::default()
                },
                seed,
            )
            .unwrap(),
        ),
        _ => unreachable!(),
    };
    let before = support::evaluate(learner.actor(), &env, seed);
    let random = support::random_score(&env, seed);
    let collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env.clone(); 32]).unwrap(), seed).unwrap();
    let mut trainer = OnPolicyTrainer::new(
        collector,
        OnPolicyConfig {
            rollout_steps: 4,
            episode_guard: (name == "reinforce").then_some(4),
            gamma: 1.,
            gae_lambda: None,
            normalize_advantages: true,
        },
    )
    .unwrap();
    let mut last = 0;
    let mut progress = |p: &rl::train::trainer::Progress| {
        assert!(
            start.elapsed().as_secs() < if pilot { 60 } else { 180 },
            "{name} seed={seed} wall-clock cap exceeded"
        );
        if p.counters.updates >= last + 250 {
            last = p.counters.updates;
            eprintln!(
                "progress {name} seed={seed} transitions={} updates={} elapsed_s={:.3}",
                p.counters.env_steps,
                p.counters.updates,
                start.elapsed().as_secs_f64()
            );
        }
    };
    let stats = trainer
        .train(
            learner.as_mut(),
            Budget {
                env_steps: Some(if pilot { 4096 } else { 15_360 }),
                updates: Some(MAX_UPDATES),
                ..Default::default()
            },
            Some(&mut progress),
        )
        .unwrap();
    assert!(stats.counters.env_steps <= MAX_TRANSITIONS && stats.counters.updates <= MAX_UPDATES);
    let after = support::evaluate(learner.actor(), &env, seed);
    let success = after > -0.1 && after > before + 0.03 && after > random + 0.1;
    eprintln!(
        "case {name} seed={seed} pilot={pilot} initial={before:.6} random={random:.6} final={after:.6} pass={success} transitions={} updates={} elapsed_s={:.3}",
        stats.counters.env_steps,
        stats.counters.updates,
        start.elapsed().as_secs_f64()
    );
    success
}

#[test]
#[ignore = "bounded five-seed multistep acceptance; RL_SCIENCE_PILOT=1 runs a short non-acceptance pilot"]
fn neural_families_improve_delayed_control_and_masked_chain() {
    let start = Instant::now();
    let pilot = support::pilot();
    let seeds = if pilot { &SEEDS[..1] } else { &SEEDS[..] };
    let mut results = Vec::new();
    for name in [
        "ddqn",
        "ddpg",
        "td3",
        "d4pg",
        "md4pg",
        "ppo",
        "reinforce",
        "vmpo",
    ] {
        let mut successes = 0;
        for &seed in seeds {
            assert!(
                start.elapsed().as_secs() < 600,
                "scientific group exceeded ten-minute cap"
            );
            let passed = match name {
                "ddqn" => run_off_policy(name, seed, MaskedChain::new(), pilot),
                "ppo" | "reinforce" | "vmpo" => run_on_policy(name, seed, pilot),
                _ => run_off_policy(name, seed, DelayedControl::new(), pilot),
            };
            successes += usize::from(passed);
        }
        eprintln!(
            "aggregate {name}: {successes}/{} passed; pilot={pilot}",
            seeds.len()
        );
        results.push((name, successes));
    }
    if !pilot {
        assert!(
            results.iter().all(|(_, successes)| *successes >= 4),
            "multistep acceptance results: {results:?}"
        );
    }
}
