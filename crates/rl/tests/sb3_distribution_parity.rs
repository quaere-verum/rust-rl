#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::action::Action,
    nn::{distributions::Distribution, matrix, to_vec},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    categorical: Categorical,
    gaussian: Gaussian,
}

#[derive(Deserialize)]
struct Categorical {
    rows: usize,
    columns: usize,
    logits: Vec<f32>,
    other_logits: Vec<f32>,
    actions: Vec<usize>,
    log_prob: Vec<f32>,
    entropy: Vec<f32>,
    kl: Vec<f32>,
    gradients: Vec<f32>,
}

#[derive(Deserialize)]
struct Gaussian {
    rows: usize,
    columns: usize,
    mean: Vec<f32>,
    log_std: Vec<f32>,
    other_mean: Vec<f32>,
    other_log_std: Vec<f32>,
    actions: Vec<f32>,
    log_prob: Vec<f32>,
    entropy: Vec<f32>,
    kl: Vec<f32>,
    mean_gradients: Vec<f32>,
    log_std_gradients: Vec<f32>,
}

fn fixture() -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sb3_distributions.json");
    serde_json::from_slice(&std::fs::read(path).expect("generate SB3 distribution fixture"))
        .unwrap()
}

fn compare(name: &str, actual: Vec<f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name} length");
    let mut max = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        assert!(error <= 5e-5, "{name}[{i}]: {a} versus {e}, error {error}");
        max = max.max(error);
    }
    println!("{name} max absolute error: {max:.9}");
}

#[test]
fn categorical_values_and_gradients_match_sb3() {
    let c = fixture().categorical;
    let logits = matrix(c.logits, c.rows, c.columns).require_grad();
    let dist = Distribution::Categorical {
        logits: logits.clone(),
        masks: vec![true; c.rows * c.columns],
    };
    let other = Distribution::Categorical {
        logits: matrix(c.other_logits, c.rows, c.columns),
        masks: vec![true; c.rows * c.columns],
    };
    let actions: Vec<_> = c.actions.into_iter().map(Action::Discrete).collect();
    let lp = dist.log_prob(&actions, &vec![None; c.rows]).unwrap();
    let entropy = dist.entropy(&mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    compare(
        "categorical log probability",
        to_vec(lp.clone()).unwrap(),
        &c.log_prob,
    );
    compare(
        "categorical entropy",
        to_vec(entropy.clone()).unwrap(),
        &c.entropy,
    );
    compare(
        "categorical KL(p||q)",
        to_vec(dist.kl(&other).unwrap()).unwrap(),
        &c.kl,
    );
    let gradients = (lp.sum() - entropy.sum() * 0.03).backward();
    compare(
        "categorical gradient",
        to_vec(logits.grad(&gradients).unwrap()).unwrap(),
        &c.gradients,
    );
}

#[test]
fn unsquashed_gaussian_values_and_gradients_match_sb3() {
    let c = fixture().gaussian;
    let mean = matrix(c.mean, c.rows, c.columns).require_grad();
    let std = matrix(c.log_std, c.rows, c.columns).require_grad();
    let dist = Distribution::Gaussian {
        mean: mean.clone(),
        log_std: std.clone(),
        bounds: None,
    };
    let other = Distribution::Gaussian {
        mean: matrix(c.other_mean, c.rows, c.columns),
        log_std: matrix(c.other_log_std, c.rows, c.columns),
        bounds: None,
    };
    let actions: Vec<_> = c
        .actions
        .chunks_exact(c.columns)
        .map(|a| Action::Continuous(a.to_vec()))
        .collect();
    let lp = dist.log_prob(&actions, &vec![None; c.rows]).unwrap();
    let entropy = dist.entropy(&mut ChaCha8Rng::seed_from_u64(0)).unwrap();
    compare(
        "gaussian log probability",
        to_vec(lp.clone()).unwrap(),
        &c.log_prob,
    );
    compare(
        "gaussian entropy",
        to_vec(entropy.clone()).unwrap(),
        &c.entropy,
    );
    compare(
        "gaussian KL(p||q)",
        to_vec(dist.kl(&other).unwrap()).unwrap(),
        &c.kl,
    );
    let gradients = (lp.sum() - entropy.sum() * 0.03).backward();
    compare(
        "gaussian mean gradient",
        to_vec(mean.grad(&gradients).unwrap()).unwrap(),
        &c.mean_gradients,
    );
    compare(
        "gaussian log std gradient",
        to_vec(std.grad(&gradients).unwrap()).unwrap(),
        &c.log_std_gradients,
    );
}
