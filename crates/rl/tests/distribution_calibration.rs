#![cfg(feature = "neural")]

//! Calibration of fixed-policy return distributions and oracle-critic risk policies.
//! These tests keep observed returns distinct from predicted moments/quantiles.
mod support;
use burn::{
    module::Module,
    optim::{GradientsParams, ModuleOptimizer},
    tensor::Tensor,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::OptimizerConfig,
    core::{
        action::{Action, ActionSpace},
        env::{Env, Observation},
    },
    nn::{
        critics::{
            quantile::QuantileGrid,
            scalar::{ActionValue, QValue},
        },
        matrix,
        network::{Mlp, Network},
        policy::{Deterministic, DeterministicPolicy},
        to_vec, vector,
    },
    objectives::{
        quantile::{QuantileLoss, quantile_loss},
        risk::{Risk, RiskObjective},
    },
};
use std::time::Instant;
use support::{CommitLottery, MAX_TRANSITIONS, MAX_UPDATES, SEEDS};

#[test]
fn commit_lottery_defers_reward_and_has_the_stated_exact_distribution() {
    for action in [0., 0.5, 1.] {
        let low = 1. - 4. * action;
        let high = 1. + 2. * action;
        let mean: f64 = 0.2 * low + 0.8 * high;
        let second: f64 = 0.2 * low * low + 0.8 * high * high;
        assert!((mean - (1. + 0.8 * action)).abs() < 1e-12);
        assert!((second - (mean * mean + 5.76 * action * action)).abs() < 1e-12);
        let mut env = CommitLottery::new(13);
        env.reset(None).unwrap();
        for time in 0..4 {
            let step = env
                .step(&Action::Continuous(vec![if time == 0 {
                    action as f32
                } else {
                    1. - action as f32
                }]))
                .unwrap();
            if time < 3 {
                assert_eq!(step.reward.total(), 0.);
                assert!(!step.terminated);
            } else {
                assert!(step.reward.total() == low as f32 || step.reward.total() == high as f32);
                assert!(step.terminated);
            }
        }
    }
}

#[test]
fn analytic_expected_cvar_and_mean_variance_preferences_are_distinct() {
    let grid = QuantileGrid::uniform(5).unwrap();
    let quantiles = matrix(vec![1., 1., 1., 1., 1., -3., 3., 3., 3., 3.], 2, 5);
    let expected = to_vec(Risk::Expected.quantiles(quantiles.clone(), &grid).unwrap()).unwrap();
    let cvar = to_vec(
        Risk::Cvar { alpha: 0.2 }
            .quantiles(quantiles, &grid)
            .unwrap(),
    )
    .unwrap();
    assert!((expected[0] - 1.).abs() < 1e-6 && (expected[1] - 1.8).abs() < 1e-6);
    assert_eq!(cvar, [1., -3.]);
    let optimum = 0.8 / 11.52;
    let actions = [0., optimum, 1.];
    let mean: Vec<f32> = actions.iter().map(|a| 1. + 0.8 * a).collect();
    let second: Vec<_> = actions
        .iter()
        .zip(&mean)
        .map(|(a, m)| m * m + 5.76 * a * a)
        .collect();
    let utility = to_vec(
        Risk::MeanVariance { penalty: 1. }
            .moments(vector(&mean), vector(&second))
            .unwrap(),
    )
    .unwrap();
    assert!(utility[1] > utility[0] && utility[1] > utility[2]);
}

fn fit<M: Module>(model: &mut M, optimizer: &mut ModuleOptimizer, rate: f64, loss: Tensor<1>) {
    assert!(
        loss.clone().into_scalar::<f32>().is_finite(),
        "nonfinite calibration loss"
    );
    let gradients = GradientsParams::from_grads(loss.backward(), model);
    *model = optimizer.step(rate, model.clone(), gradients);
}

fn empirical(action: f32, seed: u64, episodes: usize) -> (f64, f64, f64, f64) {
    let mut env = CommitLottery::new(seed + 50_000);
    let mut returns: Vec<_> = (0..episodes)
        .map(|_| f64::from(env.episode(action)))
        .collect();
    let mean = returns.iter().sum::<f64>() / episodes as f64;
    let second = returns.iter().map(|r| r * r).sum::<f64>() / episodes as f64;
    returns.sort_by(f64::total_cmp);
    let lower = ((episodes as f64 * 0.2).ceil() as usize).max(1);
    let tail = returns[..lower].iter().sum::<f64>() / lower as f64;
    (mean, second, second - mean * mean, tail)
}

#[derive(Module, Debug)]
struct QuadraticFeatures {
    linear: Mlp,
}
impl Network for QuadraticFeatures {
    fn input_dim(&self) -> usize {
        3
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.linear
            .forward(Tensor::cat(vec![input.clone(), input.powf_scalar(2.)], 1))
    }
}

fn calibration(seed: u64, pilot: bool) -> bool {
    let start = Instant::now();
    let short_pilot = pilot && !support::full_pilot();
    let mut model_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut sample_rng = ChaCha8Rng::seed_from_u64(seed + 1_000);
    let mut env = CommitLottery::new(seed + 2_000);
    let actions = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
    let episodes = if short_pilot { 256 } else { 4_000 };
    let dataset: Vec<_> = (0..episodes)
        .map(|index| {
            let action = actions[index % actions.len()];
            (action, env.episode(action))
        })
        .collect();
    // The exact quantiles and mean are affine in committed action, so use an
    // identifiable affine model instead of extrapolating hidden ReLU features.
    let mut quantiles = QValue::new(2, 1, 5, &[], &mut model_rng).unwrap();
    let mut first = QValue::new(2, 1, 1, &[], &mut model_rng).unwrap();
    let mut second = QValue {
        network: QuadraticFeatures {
            linear: Mlp::new(6, &[], 1, &mut model_rng).unwrap(),
        },
        observation_dim: 2,
        action_dim: 1,
    };
    let config = OptimizerConfig {
        learning_rate: 0.003,
        max_grad_norm: Some(5.),
    };
    let mut q_opt = config.build().unwrap();
    let mut first_opt = config.build().unwrap();
    let mut second_opt = config.build().unwrap();
    let grid = QuantileGrid::uniform(5).unwrap();
    let observations = vec![Observation::new(vec![0., 0.]); 64];
    let steps = if short_pilot { 400 } else { 1_500 };
    for step in 0..steps {
        assert!(
            start.elapsed().as_secs() < if pilot { 60 } else { 180 },
            "calibration wall cap exceeded"
        );
        let rows: Vec<_> = (0..64)
            .map(|_| dataset[sample_rng.random_range(0..dataset.len())])
            .collect();
        let input_actions = matrix(rows.iter().map(|row| row.0).collect(), 64, 1);
        let returns: Vec<_> = rows.iter().map(|row| row.1).collect();
        let prediction = quantiles
            .values((&observations).into(), input_actions.clone())
            .unwrap();
        let target = matrix(
            returns.iter().flat_map(|value| [*value; 5]).collect(),
            64,
            5,
        );
        let loss = quantile_loss(prediction, target, &grid, &QuantileLoss::Absolute).unwrap();
        fit(&mut quantiles, &mut q_opt, 0.01, loss);
        let loss = (first
            .values((&observations).into(), input_actions.clone())
            .unwrap()
            .squeeze_dim(1)
            - vector(&returns))
        .powf_scalar(2.)
        .mean();
        fit(&mut first, &mut first_opt, config.learning_rate, loss);
        let squared: Vec<_> = returns.iter().map(|value| value * value).collect();
        let loss = (second
            .values((&observations).into(), input_actions)
            .unwrap()
            .squeeze_dim(1)
            - vector(&squared))
        .powf_scalar(2.)
        .mean();
        fit(&mut second, &mut second_opt, 0.01, loss);
        if (step + 1) % 80 == 0 {
            eprintln!(
                "calibration progress seed={seed} calls={} training_transitions={} elapsed_s={:.3}",
                3 * (step + 1),
                episodes * 4,
                start.elapsed().as_secs_f64()
            );
        }
    }
    let evaluation_episodes = if short_pilot { 128 } else { 256 };
    assert!((episodes * 4 + 3 * evaluation_episodes * 4) as u64 <= MAX_TRANSITIONS);
    assert!((steps * 3) as u64 <= MAX_UPDATES);
    let mut success = true;
    for action in [0., 0.5, 1.] {
        let observation = [Observation::new(vec![0., 0.])];
        let q = to_vec(
            quantiles
                .valid()
                .values((&observation).into(), matrix(vec![action], 1, 1))
                .unwrap(),
        )
        .unwrap();
        let predicted_mean = first
            .valid()
            .values((&observation).into(), matrix(vec![action], 1, 1))
            .unwrap()
            .into_scalar::<f32>() as f64;
        let predicted_second = second
            .valid()
            .values((&observation).into(), matrix(vec![action], 1, 1))
            .unwrap()
            .into_scalar::<f32>() as f64;
        let a = f64::from(action);
        let mean = 1. + 0.8 * a;
        let raw_second = mean * mean + 5.76 * a * a;
        let expected = [
            1. - 4. * action,
            1. + 2. * action,
            1. + 2. * action,
            1. + 2. * action,
            1. + 2. * action,
        ];
        let quantile_error = q
            .iter()
            .zip(expected)
            .map(|(q, expected)| (q - expected).abs())
            .fold(0_f32, f32::max);
        let (observed_mean, observed_second, observed_variance, observed_tail) =
            empirical(action, seed + 3_000, evaluation_episodes);
        let passed = quantile_error < 0.25
            && (predicted_mean - mean).abs() < 0.25
            && (predicted_second - raw_second).abs() < 1.0
            && (observed_mean - mean).abs() < 0.6
            && (observed_second - raw_second).abs() < 2.2
            && (observed_variance - 5.76 * a * a).abs() < 2.0;
        success &= passed;
        eprintln!(
            "calibration seed={seed} pilot={pilot} action={action} q={q:?} q_max_error={quantile_error:.5} predicted_mean={predicted_mean:.5} predicted_raw_second={predicted_second:.5} observed_mean={observed_mean:.5} observed_raw_second={observed_second:.5} observed_variance={observed_variance:.5} observed_cvar={observed_tail:.5} pass={passed}"
        );
    }
    eprintln!(
        "calibration case seed={seed} pass={success} calls={} total_transitions={} elapsed_s={:.3}",
        steps * 3,
        episodes * 4 + 3 * evaluation_episodes * 4,
        start.elapsed().as_secs_f64()
    );
    success
}

#[test]
#[ignore = "bounded five-seed fixed-policy calibration; RL_SCIENCE_PILOT=1 is a short non-acceptance pilot"]
fn fixed_policy_quantiles_and_raw_moments_calibrate_on_held_out_actions() {
    let start = Instant::now();
    let pilot = support::pilot();
    let seeds = if pilot { &SEEDS[..1] } else { &SEEDS[..] };
    let mut successes = 0;
    for &seed in seeds {
        assert!(start.elapsed().as_secs() < 600);
        successes += usize::from(calibration(seed, pilot));
    }
    eprintln!(
        "calibration aggregate: {successes}/{}; pilot={pilot}",
        seeds.len()
    );
    if !pilot {
        assert!(
            successes >= 4,
            "fixed-policy calibration passed {successes}/5 seeds"
        );
    }
}

#[test]
#[ignore = "bounded five-seed oracle-critic risk policy test; distinct from end-to-end risk learning"]
fn oracle_critic_policies_choose_expected_cvar_and_variance_optima() {
    let pilot = support::pilot();
    let seeds = if pilot { &SEEDS[..1] } else { &SEEDS[..] };
    let grid = QuantileGrid::uniform(5).unwrap();
    let mut aggregates = Vec::new();
    for (name, risk, optimum) in [
        ("expected", Risk::Expected, 1.),
        ("cvar", Risk::Cvar { alpha: 0.2 }, 0.),
        (
            "mean_variance",
            Risk::MeanVariance { penalty: 1. },
            0.8 / 11.52,
        ),
    ] {
        let mut successes = 0;
        for &seed in seeds {
            let start = Instant::now();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut policy = Deterministic::new(
                1,
                ActionSpace::Box {
                    low: vec![0.],
                    high: vec![1.],
                },
                &[8],
                &mut rng,
            )
            .unwrap();
            let config = OptimizerConfig {
                learning_rate: 0.01,
                max_grad_norm: Some(5.),
            };
            let mut optimizer = config.build().unwrap();
            let observations = [Observation::new(vec![1.])];
            let steps = if pilot { 250 } else { 500 };
            for step in 0..steps {
                assert!(start.elapsed().as_secs() < 60);
                let action = policy.actions((&observations).into()).unwrap();
                let mean = action.clone().squeeze_dim(1) * 0.8 + 1.;
                let second = mean.clone().powf_scalar(2.)
                    + action.clone().squeeze_dim(1).powf_scalar(2.) * 5.76;
                let utility = if matches!(risk, Risk::Cvar { .. }) {
                    let low = action.clone() * -4. + 1.;
                    let high = action * 2. + 1.;
                    risk.quantiles(Tensor::cat(vec![low, high.repeat_dim(1, 4)], 1), &grid)
                        .unwrap()
                } else {
                    risk.moments(mean, second).unwrap()
                };
                fit(
                    &mut policy,
                    &mut optimizer,
                    config.learning_rate,
                    -utility.mean(),
                );
                if (step + 1) % 250 == 0 {
                    eprintln!(
                        "risk progress {name} seed={seed} calls={} elapsed_s={:.3}",
                        step + 1,
                        start.elapsed().as_secs_f64()
                    );
                }
            }
            let action = policy
                .valid()
                .actions((&observations).into())
                .unwrap()
                .into_scalar::<f32>();
            let (mean, second, variance, tail) = empirical(action, seed + 9_000, 256);
            let predicted_mean = 1. + 0.8 * action;
            let predicted_variance = 5.76 * action * action;
            let success = (action - optimum).abs() < 0.04
                && (mean - f64::from(predicted_mean)).abs() < 0.6
                && (variance - f64::from(predicted_variance)).abs() < 2.;
            successes += usize::from(success);
            eprintln!(
                "oracle-risk {name} seed={seed} pilot={pilot} action={action:.6} optimum={optimum:.6} predicted_mean={predicted_mean:.5} predicted_variance={predicted_variance:.5} observed_mean={mean:.5} observed_raw_second={second:.5} observed_variance={variance:.5} observed_cvar={tail:.5} pass={success} elapsed_s={:.3}",
                start.elapsed().as_secs_f64()
            );
        }
        eprintln!(
            "oracle-risk aggregate {name}: {successes}/{}; pilot={pilot}",
            seeds.len()
        );
        aggregates.push((name, successes));
    }
    if !pilot {
        assert!(
            aggregates.iter().all(|(_, successes)| *successes >= 4),
            "oracle risk results: {aggregates:?}"
        );
    }
}
