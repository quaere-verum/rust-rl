#![cfg(all(feature = "diagnostics", feature = "neural"))]

//! Bounded paired comparison on the existing three-state LQG dynamics.
use std::time::Instant;

use burn::{module::Param, optim::GradientsParams};
use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    algorithms::{
        OptimizerConfig,
        ddpg::{Ddpg, DdpgConfig},
    },
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        env::{Env, Observation},
        learner::Learner,
        transition::Transition,
    },
    data::replay::{ReplayBatch, ReplayItem},
    diagnostics::evaluation::BranchingEnv,
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        critics::{
            residual::{ResidualQ, ResidualScale},
            scalar::{ActionValue, QValue},
        },
        matrix,
        policy::Deterministic,
        target::frozen_copy,
        to_vec, vector,
    },
};

const HORIZON: usize = 8;
const DIM: usize = 3;
const WIDTH: usize = DIM + 1;
const MINIBATCH: usize = 32;
const CRITIC_STEPS: usize = 240;
const CLOSED_DECISIONS: usize = 384;

fn model() -> LqgConfig {
    let diag = |x: &[f64]| DMatrix::from_diagonal(&DVector::from_column_slice(x));
    LqgConfig {
        a: vec![
            DMatrix::from_row_slice(DIM, DIM, &[0.94, 0.12, 0., 0., 0.92, 0.12, 0., 0., 0.9]);
            HORIZON
        ],
        b: vec![diag(&[0.45, 0.6, 0.55]); HORIZON],
        q: vec![diag(&[1., 1.3, 0.7]); HORIZON],
        r: vec![diag(&[0.15, 0.2, 0.12]); HORIZON],
        terminal: diag(&[1., 1.3, 0.7]),
        noise: vec![DMatrix::identity(DIM, DIM) * 0.08; HORIZON],
        initial: vec![0.; DIM],
        action_space: ActionSpace::Unbounded { dimensions: DIM },
    }
}

fn parameters() -> Vec<f32> {
    (0..(WIDTH + 1) * DIM)
        .map(|i| if i / DIM == i % DIM { -0.45 } else { 0. })
        .collect()
}

fn policy(config: &LqgConfig, parameters: &[f32]) -> Deterministic {
    let mut policy = Deterministic::new(
        WIDTH,
        config.action_space.clone(),
        &[],
        &mut ChaCha8Rng::seed_from_u64(71),
    )
    .unwrap();
    policy.network.layers[0].weight =
        Param::from_tensor(matrix(parameters[..WIDTH * DIM].to_vec(), WIDTH, DIM));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&parameters[WIDTH * DIM..])));
    policy
}

fn as_action(parameters: &[f32], observation: &Observation) -> Vec<f32> {
    let mut action = parameters[WIDTH * DIM..].to_vec();
    for (row, &feature) in observation.dense_values().iter().enumerate() {
        for (column, value) in action.iter_mut().enumerate() {
            *value += parameters[row * DIM + column] * feature;
        }
    }
    action
}

fn random_environment(config: &LqgConfig, rng: &mut ChaCha8Rng) -> Lqg {
    let mut config = config.clone();
    config.initial = (0..DIM).map(|_| rng.sample(StandardNormal)).collect();
    Lqg::new(config, rng.random()).unwrap()
}

fn rollout(mut env: Lqg, first: Vec<f32>, fixed: &[f32]) -> f64 {
    let mut action = Action::Continuous(first);
    let mut total = 0.;
    loop {
        let step = env.step(&action).unwrap();
        total += f64::from(step.reward.total());
        if step.terminated {
            return total;
        }
        action = Action::Continuous(as_action(fixed, &step.observation));
    }
}

struct Reference {
    cost: Vec<DMatrix<f64>>,
    noise: Vec<f64>,
}

impl Reference {
    fn new(config: &LqgConfig) -> Self {
        let feedback = DMatrix::identity(DIM, DIM) * f64::from(-0.45_f32);
        let mut cost = vec![config.terminal.clone(); HORIZON + 1];
        let mut noise = vec![0.; HORIZON + 1];
        for t in (0..HORIZON).rev() {
            let transition = &config.a[t] + &config.b[t] * &feedback;
            cost[t] = &config.q[t]
                + feedback.transpose() * &config.r[t] * &feedback
                + transition.transpose() * &cost[t + 1] * transition;
            noise[t] = noise[t + 1]
                + (config.noise[t].transpose() * &cost[t + 1] * &config.noise[t]).trace();
        }
        Self { cost, noise }
    }

    fn value_gradient(
        &self,
        config: &LqgConfig,
        obs: &Observation,
        action: &[f32],
    ) -> (f64, Vec<f64>) {
        let values = obs.dense_values();
        let time = (values[DIM] * HORIZON as f32).round() as usize;
        let x = DVector::from_iterator(DIM, values[..DIM].iter().copied().map(f64::from));
        let u = DVector::from_iterator(DIM, action.iter().copied().map(f64::from));
        let next = &config.a[time] * &x + &config.b[time] * &u;
        let future = &self.cost[time + 1] * &next;
        let current_action = &config.r[time] * &u;
        let value = -x.dot(&(&config.q[time] * &x))
            - u.dot(&current_action)
            - next.dot(&future)
            - self.noise[time];
        let gradient = -2. * (current_action + config.b[time].transpose() * future);
        (value, gradient.as_slice().to_vec())
    }
}

struct Row {
    observation: Observation,
    action: Vec<f32>,
    returns: f32,
}

fn experience(
    config: &LqgConfig,
    fixed: &[f32],
    sigma: f32,
    seed: u64,
    episodes: usize,
    futures: usize,
) -> Vec<Row> {
    let mut factual = ChaCha8Rng::seed_from_u64(seed);
    let mut independent = ChaCha8Rng::seed_from_u64(seed + 1);
    let mut rows = Vec::with_capacity(episodes * HORIZON);
    for _ in 0..episodes {
        let mut env = random_environment(config, &mut factual);
        let mut observation = env.reset(None).unwrap().observation;
        for _ in 0..HORIZON {
            let action: Vec<_> = as_action(fixed, &observation)
                .into_iter()
                .map(|a| a + sigma * factual.sample::<f32, _>(StandardNormal))
                .collect();
            let returns = (0..futures)
                .map(|_| {
                    rollout(
                        env.branch(independent.random()).unwrap(),
                        action.clone(),
                        fixed,
                    )
                })
                .sum::<f64>()
                / futures.max(1) as f64;
            let next = env
                .step(&Action::Continuous(action.clone()))
                .unwrap()
                .observation;
            rows.push(Row {
                observation,
                action,
                returns: returns as f32,
            });
            observation = next;
        }
    }
    rows
}

fn fit<Q: ActionValue>(mut critic: Q, data: &[Row], seed: u64) -> Q {
    let settings = OptimizerConfig {
        learning_rate: 0.004,
        max_grad_norm: Some(10.),
    };
    let mut optimizer = settings.build().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..CRITIC_STEPS {
        let rows: Vec<_> = (0..MINIBATCH)
            .map(|_| &data[rng.random_range(0..data.len())])
            .collect();
        let observations: Vec<_> = rows.iter().map(|row| &row.observation).collect();
        let actions = matrix(
            rows.iter()
                .flat_map(|row| row.action.iter().copied())
                .collect(),
            MINIBATCH,
            DIM,
        );
        let returns = vector(&rows.iter().map(|row| row.returns).collect::<Vec<_>>());
        let prediction = critic
            .values(
                rl::core::env::ObservationBatch::from_refs(&observations),
                actions,
            )
            .unwrap()
            .squeeze_dim::<1>(1);
        let loss = (prediction - returns).powf_scalar(2.).mean();
        let gradients = GradientsParams::from_grads(loss.backward(), &critic);
        critic = optimizer.step(settings.learning_rate, critic, gradients);
    }
    frozen_copy(&critic).valid()
}

fn values<Q: ActionValue>(critic: &Q, observations: &[Observation], actions: &[f32]) -> Vec<f32> {
    to_vec(
        critic
            .values(
                observations.into(),
                matrix(actions.to_vec(), observations.len(), DIM),
            )
            .unwrap(),
    )
    .unwrap()
}

fn gradient<Q: ActionValue>(critic: &Q, observations: &[Observation], actions: &[f32]) -> Vec<f32> {
    let tensor = matrix(actions.to_vec(), observations.len(), DIM).require_grad();
    let result = critic
        .values(observations.into(), tensor.clone())
        .unwrap()
        .sum();
    to_vec(tensor.grad(&result.backward()).unwrap()).unwrap()
}

fn average_return(config: &LqgConfig, fixed: &[f32], seed: u64) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..40)
        .map(|_| {
            let mut env = random_environment(config, &mut rng);
            let observation = env.reset(None).unwrap().observation;
            rollout(env, as_action(fixed, &observation), fixed)
        })
        .sum::<f64>()
        / 40.
}

fn actor_gradient(obs: &[Observation], field: impl Iterator<Item = f64>) -> Vec<f64> {
    let field: Vec<_> = field.collect();
    let mut result = vec![0.; (WIDTH + 1) * DIM];
    for (row, state) in obs.iter().enumerate() {
        for (feature, x) in state
            .dense_values()
            .iter()
            .map(|&x| f64::from(x))
            .chain([1.])
            .enumerate()
        {
            for action in 0..DIM {
                result[feature * DIM + action] += x * field[row * DIM + action] / obs.len() as f64;
            }
        }
    }
    result
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn compare(
    critic: &impl ActionValue,
    config: &LqgConfig,
    reference: &Reference,
    fixed: &[f32],
    evaluation: &[Row],
    seed: u64,
) -> [f64; 5] {
    let observations: Vec<_> = evaluation
        .iter()
        .map(|row| row.observation.clone())
        .collect();
    let actions: Vec<_> = evaluation
        .iter()
        .flat_map(|row| row.action.iter().copied())
        .collect();
    let predictions = values(critic, &observations, &actions);
    let estimate = gradient(critic, &observations, &actions);
    let mut value_error = 0.;
    let mut gradient_error = 0.;
    for (i, row) in evaluation.iter().enumerate() {
        let (truth, derivative) = reference.value_gradient(config, &row.observation, &row.action);
        value_error += (f64::from(predictions[i]) - truth).powi(2);
        gradient_error += derivative
            .iter()
            .enumerate()
            .map(|(j, expected)| (f64::from(estimate[i * DIM + j]) - expected).powi(2))
            .sum::<f64>();
    }
    let bases = observations
        .iter()
        .map(|obs| as_action(fixed, obs))
        .collect::<Vec<_>>();
    let policy_actions: Vec<_> = bases.iter().flatten().copied().collect();
    let mut perturbed = policy_actions.clone();
    for (row, a) in perturbed.chunks_exact_mut(DIM).enumerate() {
        a[row % DIM] += if row % 2 == 0 { 0.1 } else { -0.1 };
    }
    let at_policy = values(critic, &observations, &policy_actions);
    let contrasts = values(critic, &observations, &perturbed);
    let contrast_error = observations
        .iter()
        .enumerate()
        .map(|(i, obs)| {
            let exact = reference
                .value_gradient(config, obs, &perturbed[i * DIM..(i + 1) * DIM])
                .0
                - reference.value_gradient(config, obs, &bases[i]).0;
            (f64::from(contrasts[i] - at_policy[i]) - exact).powi(2)
        })
        .sum::<f64>();
    let field = gradient(critic, &observations, &policy_actions)
        .into_iter()
        .map(f64::from);
    let estimate = actor_gradient(&observations, field);
    let truth = actor_gradient(
        &observations,
        observations
            .iter()
            .zip(&bases)
            .flat_map(|(obs, action)| reference.value_gradient(config, obs, action).1),
    );
    let cosine = estimate.iter().zip(&truth).map(|(x, y)| x * y).sum::<f64>()
        / (norm(&estimate) * norm(&truth));
    let updated: Vec<_> = fixed
        .iter()
        .zip(&estimate)
        .map(|(&p, &g)| p + (0.04 * g / norm(&estimate)) as f32)
        .collect();
    let improvement = average_return(config, &updated, seed) - average_return(config, fixed, seed);
    let n = evaluation.len() as f64;
    [
        (value_error / n).sqrt(),
        (contrast_error / n).sqrt(),
        (gradient_error / (n * DIM as f64)).sqrt(),
        cosine,
        improvement,
    ]
}

fn trained_return<Q: ActionValue>(config: &LqgConfig, fixed: &[f32], critic: Q, seed: u64) -> f64 {
    let mut learner = Ddpg::new(
        policy(config, fixed),
        critic,
        DdpgConfig {
            exploration_std: 0.05,
            actor: OptimizerConfig {
                learning_rate: 0.001,
                ..Default::default()
            },
            critic: OptimizerConfig {
                learning_rate: 0.004,
                ..Default::default()
            },
            ..Default::default()
        },
        seed,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut behavior_rng = ChaCha8Rng::seed_from_u64(seed + 1);
    let mut sample_rng = ChaCha8Rng::seed_from_u64(seed + 2);
    let mut replay = Vec::new();
    for _ in 0..CLOSED_DECISIONS / HORIZON {
        let mut env = random_environment(config, &mut rng);
        let mut observation = env.reset(None).unwrap().observation;
        for _ in 0..HORIZON {
            let action = learner
                .actor()
                .act(
                    std::slice::from_ref(&observation),
                    ActMode::Sample,
                    &mut behavior_rng,
                )
                .unwrap()
                .remove(0);
            let step = env.step(&action).unwrap();
            replay.push(ReplayItem {
                transition: Transition {
                    observation,
                    action,
                    next_observation: step.observation.clone(),
                    reward: step.reward,
                    terminated: step.terminated,
                    truncated: step.truncated,
                },
                discount: if step.terminated { 0. } else { 1. },
            });
            observation = step.observation;
            if replay.len() >= 64 && replay.len().is_multiple_of(4) {
                let items = (0..MINIBATCH)
                    .map(|_| replay[sample_rng.random_range(0..replay.len())].clone())
                    .collect();
                learner.update(&ReplayBatch { items }).unwrap();
            }
        }
    }
    let policy = learner.policy();
    let mut weights = to_vec(policy.network.layers[0].weight.val()).unwrap();
    weights.extend(to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap());
    average_return(config, &weights, seed + 10_000)
}

#[test]
#[ignore = "bounded paired LQG learning study"]
fn matched_lqg_returns_gradients_and_closed_loop_control() {
    let config = model();
    let reference = Reference::new(&config);
    let fixed = parameters();
    let anchor = policy(&config, &fixed);
    println!(
        "fit,seed,sigma,critic,value_rmse,contrast_rmse,action_gradient_rmse,actor_cosine,wrong_sign,controlled_return_improvement,seconds"
    );
    for seed in [17, 31, 47] {
        for sigma in [0.05, 0.2, 0.8] {
            eprintln!("Fitting paired critics for seed {seed}, sigma {sigma}");
            let train = experience(&config, &fixed, sigma, seed, 96, 3);
            let test = experience(&config, &fixed, sigma, seed + 1000, 24, 0);
            let make = || {
                QValue::new(
                    WIDTH,
                    DIM,
                    1,
                    &[24, 24],
                    &mut ChaCha8Rng::seed_from_u64(seed + 2000),
                )
                .unwrap()
            };
            for (label, scores, elapsed) in {
                let mut output = Vec::new();
                let start = Instant::now();
                let q = fit(make(), &train, seed + 3000);
                output.push((
                    "raw",
                    compare(&q, &config, &reference, &fixed, &test, seed + 4000),
                    start.elapsed(),
                ));
                let start = Instant::now();
                let q = fit(
                    ResidualQ::new(make(), &anchor, ResidualScale::Scalar(sigma)).unwrap(),
                    &train,
                    seed + 3000,
                );
                output.push((
                    "residual",
                    compare(&q, &config, &reference, &fixed, &test, seed + 4000),
                    start.elapsed(),
                ));
                output
            } {
                assert!(scores.iter().all(|x| x.is_finite()));
                let [value, contrast, gradient, cosine, improvement] = scores;
                println!(
                    "fit,{seed},{sigma},{label},{value:.6},{contrast:.6},{gradient:.6},{cosine:.6},{},{improvement:.6},{:.3}",
                    cosine < 0.,
                    elapsed.as_secs_f64()
                );
            }
        }
        eprintln!("Running {CLOSED_DECISIONS} closed-loop decisions per critic for seed {seed}");
        let make = || {
            QValue::new(
                WIDTH,
                DIM,
                1,
                &[24, 24],
                &mut ChaCha8Rng::seed_from_u64(seed + 2100),
            )
            .unwrap()
        };
        let baseline = average_return(&config, &fixed, seed + 10_000);
        let start = Instant::now();
        let raw = trained_return(&config, &fixed, make(), seed);
        let raw_time = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let residual = trained_return(
            &config,
            &fixed,
            ResidualQ::new(make(), &anchor, ResidualScale::Scalar(0.05)).unwrap(),
            seed,
        );
        println!(
            "closed,{seed},{CLOSED_DECISIONS},{baseline:.6},{raw:.6},{residual:.6},{raw_time:.3},{:.3}",
            start.elapsed().as_secs_f64()
        );
        assert!(raw.is_finite() && residual.is_finite());
    }
}
