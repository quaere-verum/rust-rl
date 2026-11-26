#![cfg(feature = "neural")]

use burn::module::Param;
use burn_pack::{Bytes, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::ppo::{OptimizerConfig, Ppo, PpoConfig, PpoUpdateSettings},
    core::{
        action::{Action, ActionSpace},
        checkpoint::Checkpoint,
        env::Observation,
        learner::{Learner, OnPolicyLearner},
    },
    data::rollout::{BootstrappedBatch, OnPolicyBatch, OnPolicyMetadata},
    nn::{
        critics::scalar::Value,
        matrix,
        policy::{ActorCritic, Policy},
        to_vec, vector,
    },
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    observations: Vec<Vec<f32>>,
    actions: Vec<usize>,
    advantages: Vec<f32>,
    old_log_probs: Vec<f32>,
    old_values: Vec<f32>,
    returns: Vec<f32>,
    policy_weight: Vec<f32>,
    policy_bias: Vec<f32>,
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
    after_policy_weight: Vec<f32>,
    after_policy_bias: Vec<f32>,
    after_value_weight: Vec<f32>,
    after_value_bias: Vec<f32>,
    policy_loss: f32,
    value_loss: f32,
    approx_kl: f32,
    moments: BTreeMap<String, Vec<f32>>,
}

fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        assert!(
            error <= tolerance,
            "{name}[{i}]: Rust {a}, SB3 {e}, error {error}"
        );
        max = max.max(error);
    }
    println!("{name} max absolute error: {max:.9}");
}

fn compare_second_moment(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let mut max = 0.0_f32;
    let mut max_relative = 0.0_f32;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (a - e).abs();
        let tolerance = 1e-13 + 1e-4 * e.abs();
        assert!(
            error <= tolerance,
            "{name}[{i}]: Rust {a:e}, SB3 {e:e}, error {error:e}, tolerance {tolerance:e}"
        );
        max = max.max(error);
        if e != 0.0 {
            max_relative = max_relative.max(error / e.abs());
        }
    }
    println!(
        "{name} max absolute error: {max:e}; max relative error for nonzero reference: {max_relative:e}"
    );
}

#[test]
#[should_panic(expected = "tiny second moment")]
fn second_moment_comparison_rejects_zero_for_small_reference() {
    compare_second_moment("tiny second moment", &[0.0], &[4.2e-11]);
}

fn load_fixture(name: &str) -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).expect("generate SB3 PPO fixture")).unwrap()
}

fn build(f: &Fixture, max_grad_norm: Option<f32>, joint: bool) -> (Ppo, OnPolicyBatch) {
    build_coefficient(f, max_grad_norm, joint, 1.)
}

fn build_coefficient(
    f: &Fixture,
    max_grad_norm: Option<f32>,
    joint: bool,
    coefficient: f32,
) -> (Ppo, OnPolicyBatch) {
    build_with_size(f, max_grad_norm, joint, coefficient, 4)
}

fn build_with_size(
    f: &Fixture,
    max_grad_norm: Option<f32>,
    joint: bool,
    coefficient: f32,
    minibatch_size: usize,
) -> (Ppo, OnPolicyBatch) {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let mut policy = Policy::new(2, ActionSpace::Discrete { count: 2 }, &[], &mut rng).unwrap();
    let mut value = Value::new(2, &[], &mut rng).unwrap();
    policy.network.layers[0].weight = Param::from_tensor(matrix(f.policy_weight.clone(), 2, 2));
    policy.network.layers[0].bias = Some(Param::from_tensor(vector(&f.policy_bias)));
    value.network.layers[0].weight = Param::from_tensor(matrix(f.value_weight.clone(), 2, 1));
    value.network.layers[0].bias = Some(Param::from_tensor(vector(&f.value_bias)));
    let batch = OnPolicyBatch {
        episode_count: 0,
        observations: f
            .observations
            .iter()
            .cloned()
            .map(Observation::new)
            .collect(),
        actions: f.actions.iter().copied().map(Action::Discrete).collect(),
        latents: vec![None; 4],
        returns: f.returns.clone(),
        discount_weights: vec![1.; 4],
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: f.old_log_probs.clone(),
            values: f.old_values.clone(),
            advantages: f.advantages.clone(),
        }),
    };
    let optimizer = OptimizerConfig {
        learning_rate: 0.0003,
        max_grad_norm: if joint { None } else { max_grad_norm },
    };
    let models = ActorCritic { policy, value };
    let config = PpoConfig {
        policy: optimizer.clone(),
        value: optimizer,
        epochs: 1,
        minibatch_size,
        clip: 0.2,
        entropy: 0.03,
        target_kl: None,
    };
    let learner = if joint {
        Ppo::new_with_joint_gradient_clipping(
            models,
            config,
            123,
            coefficient,
            max_grad_norm.unwrap(),
        )
    } else {
        Ppo::new(models, config, 123)
    }
    .unwrap();
    (learner, batch)
}

fn optimizer_moments(learner: &Ppo) -> BTreeMap<String, Vec<f32>> {
    let mut observed = BTreeMap::new();
    for (name, data) in learner.checkpoint().unwrap().optimizers {
        let reader = Reader::from_bytes(Bytes::from_bytes_vec(data)).unwrap();
        let paths = reader.metadata().clone();
        for tensor in reader.into_tensors().unwrap() {
            let (tensor_name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
            let path = &paths[&id.unwrap().to_string()];
            let param = path.rsplit('.').next().unwrap();
            let moment = tensor_name.rsplit('.').next().unwrap();
            let data = burn::tensor::TensorData::from_bytes(bytes, shape, dtype);
            observed.insert(
                format!("{name}.{param}.{moment}"),
                data.try_to_vec::<f32>().unwrap(),
            );
        }
    }
    observed
}

#[test]
fn one_unclipped_optimizer_step_matches_sb3_ppo_training() {
    let f = load_fixture("sb3_ppo_update.json");
    let (mut learner, batch) = build(&f, None, false);
    let report = learner.update(&batch).unwrap();
    let observed = optimizer_moments(&learner);
    assert_eq!(observed.len(), f.moments.len());
    for (name, expected) in &f.moments {
        compare(name, &observed[name], expected, 1e-6);
    }
    assert_eq!(report.updates, 1);
    compare(
        "policy loss",
        &[report.metrics["policy_loss"] as f32],
        &[f.policy_loss],
        1e-6,
    );
    compare(
        "value loss",
        &[report.metrics["value_loss"] as f32],
        &[f.value_loss],
        1e-6,
    );
    compare(
        "approximate KL",
        &[report.metrics["approx_kl"] as f32],
        &[f.approx_kl],
        1e-6,
    );
    let policy = learner.policy();
    let value = learner.value();
    compare(
        "updated policy weight",
        &to_vec(policy.network.layers[0].weight.val()).unwrap(),
        &f.after_policy_weight,
        1e-6,
    );
    compare(
        "updated policy bias",
        &to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
        &f.after_policy_bias,
        1e-6,
    );
    compare(
        "updated value weight",
        &to_vec(value.network.layers[0].weight.val()).unwrap(),
        &f.after_value_weight,
        1e-6,
    );
    compare(
        "updated value bias",
        &to_vec(value.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
        &f.after_value_bias,
        1e-6,
    );
}

#[test]
fn current_per_tensor_clipping_differs_from_sb3_combined_norm() {
    let f = load_fixture("sb3_ppo_clipped.json");
    let (mut learner, batch) = build(&f, Some(0.1), false);
    learner.update(&batch).unwrap();
    let observed = optimizer_moments(&learner);
    for name in ["policy.weight.moment_1", "value.weight.moment_1"] {
        let max = observed[name]
            .iter()
            .zip(&f.moments[name])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("{name} max absolute difference with clipping: {max:.9}");
        assert!(
            max > 1e-4,
            "this diagnostic requires active clipping in both implementations"
        );
    }
}

#[test]
fn joint_clipping_matches_sb3_moments_and_parameters() {
    let f = load_fixture("sb3_ppo_clipped.json");
    let (mut learner, batch) = build(&f, Some(0.1), true);
    learner.update(&batch).unwrap();
    let observed = optimizer_moments(&learner);
    for (name, expected) in &f.moments {
        compare(name, &observed[name], expected, 1e-6);
    }
    let policy = learner.policy();
    let value = learner.value();
    compare(
        "joint policy weight",
        &to_vec(policy.network.layers[0].weight.val()).unwrap(),
        &f.after_policy_weight,
        1e-6,
    );
    compare(
        "joint policy bias",
        &to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
        &f.after_policy_bias,
        1e-6,
    );
    compare(
        "joint value weight",
        &to_vec(value.network.layers[0].weight.val()).unwrap(),
        &f.after_value_weight,
        1e-6,
    );
    compare(
        "joint value bias",
        &to_vec(value.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
        &f.after_value_bias,
        1e-6,
    );
}

#[test]
fn joint_clipping_matches_sb3_with_half_weighted_value_loss() {
    let f = load_fixture("sb3_ppo_clipped_vf05.json");
    let (mut learner, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let report = learner.update(&batch).unwrap();
    compare(
        "raw value loss",
        &[report.metrics["value_loss"] as f32],
        &[f.value_loss],
        1e-6,
    );
    let observed = optimizer_moments(&learner);
    for (name, expected) in &f.moments {
        compare(name, &observed[name], expected, 1e-6);
    }
    compare(
        "weighted value update",
        &to_vec(learner.value().network.layers[0].weight.val()).unwrap(),
        &f.after_value_weight,
        1e-6,
    );
}

#[test]
fn minibatch_normalization_matches_sb3_ppo() {
    for filename in [
        "sb3_ppo_normalized.json",
        "sb3_ppo_normalized_constant.json",
        "sb3_ppo_normalized_tiny.json",
        "sb3_ppo_normalized_large.json",
        "sb3_ppo_normalized_offset.json",
    ] {
        let f = load_fixture(filename);
        let (mut learner, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
        learner = learner.with_minibatch_advantage_normalization(true);
        let report = learner.update(&batch).unwrap();
        compare(
            filename,
            &[report.metrics["policy_loss"] as f32],
            &[f.policy_loss],
            1e-6,
        );
        let moments = optimizer_moments(&learner);
        for (name, expected) in &f.moments {
            compare(name, &moments[name], expected, 1e-6);
        }
        compare(
            "policy weight after normalization",
            &to_vec(learner.policy().network.layers[0].weight.val()).unwrap(),
            &f.after_policy_weight,
            1e-6,
        );
        compare(
            "value weight after normalization",
            &to_vec(learner.value().network.layers[0].weight.val()).unwrap(),
            &f.after_value_weight,
            1e-6,
        );
    }
}

#[test]
fn singleton_minibatch_keeps_its_unnormalized_advantage() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (a, batch) = build(&f, None, false);
    let (mut b, _) = build(&f, None, false);
    let mut a = a.with_minibatch_advantage_normalization(true);
    let batch = batch.select(&[0]).unwrap();
    assert_eq!(
        a.update(&batch).unwrap().metrics,
        b.update(&batch).unwrap().metrics
    );
    assert_eq!(optimizer_moments(&a), optimizer_moments(&b));
}

#[test]
fn overflowing_f32_mean_keeps_constant_finite_advantages_finite() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (a, mut batch) = build(&f, None, false);
    let (mut b, mut expected) = build(&f, None, false);
    let mut a = a.with_minibatch_advantage_normalization(true);
    let OnPolicyMetadata::Bootstrapped(raw) = &mut batch.metadata else {
        unreachable!()
    };
    raw.advantages.fill(f32::MAX);
    let OnPolicyMetadata::Bootstrapped(normalized) = &mut expected.metadata else {
        unreachable!()
    };
    normalized.advantages.fill(0.);
    assert_eq!(
        a.update(&batch).unwrap().metrics,
        b.update(&expected).unwrap().metrics
    );
    assert_eq!(optimizer_moments(&a), optimizer_moments(&b));
}

#[test]
fn extreme_finite_advantages_produce_the_expected_ppo_update() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (a, mut batch) = build(&f, None, false);
    let (mut b, mut expected) = build(&f, None, false);
    let mut a = a.with_minibatch_advantage_normalization(true);
    let OnPolicyMetadata::Bootstrapped(raw) = &mut batch.metadata else {
        unreachable!()
    };
    raw.advantages = vec![f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX];
    let OnPolicyMetadata::Bootstrapped(normalized) = &mut expected.metadata else {
        unreachable!()
    };
    normalized.advantages = vec![1.5, -0.5, -0.5, -0.5];
    let actual = a.update(&batch).unwrap();
    let reference = b.update(&expected).unwrap();
    assert_eq!(actual.updates, reference.updates);
    for key in ["policy_loss", "value_loss", "approx_kl"] {
        assert!((actual.metrics[key] - reference.metrics[key]).abs() <= 1e-6);
    }
    let actual_moments = optimizer_moments(&a);
    for (name, expected) in optimizer_moments(&b) {
        compare(&name, &actual_moments[&name], &expected, 1e-6);
    }
}

#[test]
fn normalization_uses_each_shuffled_minibatch_separately() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (a, batch) = build_with_size(&f, None, false, 1., 2);
    let (mut b, mut expected_batch) = build_with_size(&f, None, false, 1., 2);
    let mut a = a.with_minibatch_advantage_normalization(true);
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let chunks = rl::data::minibatch::shuffled_batches(4, 2, &mut rng).unwrap();
    let OnPolicyMetadata::Bootstrapped(meta) = &mut expected_batch.metadata else {
        unreachable!()
    };
    for chunk in chunks {
        let first = meta.advantages[chunk[0]];
        let second = meta.advantages[chunk[1]];
        assert!((first - second).abs() > 0.1);
        let sign = (first - second).signum();
        meta.advantages[chunk[0]] = sign * std::f32::consts::FRAC_1_SQRT_2;
        meta.advantages[chunk[1]] = -sign * std::f32::consts::FRAC_1_SQRT_2;
    }
    assert_eq!(a.update(&batch).unwrap().updates, 2);
    assert_eq!(b.update(&expected_batch).unwrap().updates, 2);
    let a_moments = optimizer_moments(&a);
    let b_moments = optimizer_moments(&b);
    for (name, expected) in b_moments {
        compare(&name, &a_moments[&name], &expected, 1e-6);
    }
    compare(
        "two minibatches policy",
        &to_vec(a.policy().network.layers[0].weight.val()).unwrap(),
        &to_vec(b.policy().network.layers[0].weight.val()).unwrap(),
        1e-6,
    );
}

#[test]
fn final_short_minibatch_keeps_singleton_advantage() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (a, full) = build_with_size(&f, None, false, 1., 2);
    let (mut b, _) = build_with_size(&f, None, false, 1., 2);
    let mut a = a.with_minibatch_advantage_normalization(true);
    let batch = full.select(&[0, 1, 2]).unwrap();
    let mut expected_batch = batch.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let chunks = rl::data::minibatch::shuffled_batches(3, 2, &mut rng).unwrap();
    assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1]);
    let OnPolicyMetadata::Bootstrapped(meta) = &mut expected_batch.metadata else {
        unreachable!()
    };
    let first = chunks[0][0];
    let second = chunks[0][1];
    let sign = (meta.advantages[first] - meta.advantages[second]).signum();
    assert_ne!(sign, 0.);
    meta.advantages[first] = sign * std::f32::consts::FRAC_1_SQRT_2;
    meta.advantages[second] = -sign * std::f32::consts::FRAC_1_SQRT_2;
    assert_eq!(a.update(&batch).unwrap().updates, 2);
    assert_eq!(b.update(&expected_batch).unwrap().updates, 2);
    let a_moments = optimizer_moments(&a);
    for (name, expected) in optimizer_moments(&b) {
        compare(&name, &a_moments[&name], &expected, 1e-6);
    }
}

#[test]
fn normalization_checkpoint_restores_mode_and_legacy_default() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (original, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let mut original = original.with_minibatch_advantage_normalization(true);
    original.update(&batch).unwrap();
    let record = original.checkpoint().unwrap();
    let (mut restored, _) = build(&f, None, false);
    restored.restore(&record).unwrap();
    assert!(restored.normalizes_advantages_in_update());
    assert_eq!(
        original.update(&batch).unwrap().metrics,
        restored.update(&batch).unwrap().metrics
    );
    assert_eq!(optimizer_moments(&original), optimizer_moments(&restored));
    let mut invalid = record;
    invalid.metadata["normalize_minibatch_advantages"] = "yes".into();
    let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
    assert!(restored.restore(&invalid).is_err());
    assert_eq!(
        serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
        before
    );
    let (legacy, _) = build(&f, None, false);
    restored.restore(&legacy.checkpoint().unwrap()).unwrap();
    assert!(!restored.normalizes_advantages_in_update());
}

#[test]
fn joint_optimizer_checkpoint_restores_mode_and_continuation() {
    let f = load_fixture("sb3_ppo_clipped.json");
    let (mut original, batch) = build(&f, Some(0.1), true);
    original.update(&batch).unwrap();
    let record = original.checkpoint().unwrap();
    let (mut restored, _) = build(&f, None, false);
    restored.restore(&record).unwrap();
    let a = original.update(&batch).unwrap();
    let b = restored.update(&batch).unwrap();
    assert_eq!(a.metrics, b.metrics);
    assert!(a.metrics.contains_key("joint_gradient_norm"));
    assert_eq!(optimizer_moments(&original), optimizer_moments(&restored));
    assert_eq!(
        to_vec(original.policy().network.layers[0].weight.val()).unwrap(),
        to_vec(restored.policy().network.layers[0].weight.val()).unwrap()
    );

    let mut corrupt = record.clone();
    corrupt.metadata["joint_optimization"]["max_grad_norm"] = (-1.).into();
    let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
    assert!(restored.restore(&corrupt).is_err());
    assert_eq!(
        serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
        before
    );

    let (legacy, _) = build(&f, None, false);
    restored.restore(&legacy.checkpoint().unwrap()).unwrap();
    assert!(
        !restored
            .update(&batch)
            .unwrap()
            .metrics
            .contains_key("joint_gradient_norm")
    );
}

#[test]
fn joint_optimizer_rejects_inconsistent_settings_and_shared_parameters() {
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let make_models = |rng: &mut ChaCha8Rng| ActorCritic {
        policy: Policy::new(2, ActionSpace::Discrete { count: 1 }, &[], rng).unwrap(),
        value: Value::new(2, &[], rng).unwrap(),
    };
    let models = make_models(&mut rng);
    assert!(
        Ppo::new_with_joint_gradient_clipping(models.clone(), PpoConfig::default(), 0, 1., 0.1)
            .is_err()
    );
    let opt = OptimizerConfig {
        max_grad_norm: None,
        ..Default::default()
    };
    let config = PpoConfig {
        policy: opt.clone(),
        value: opt,
        ..Default::default()
    };
    for (coefficient, limit) in [(f32::NAN, 0.1), (-1., 0.1), (1., 0.), (1., f32::INFINITY)] {
        assert!(
            Ppo::new_with_joint_gradient_clipping(
                models.clone(),
                config.clone(),
                0,
                coefficient,
                limit
            )
            .is_err()
        );
    }
    let policy = Policy::new(2, ActionSpace::Discrete { count: 1 }, &[], &mut rng).unwrap();
    let value = Value {
        network: policy.network.clone(),
    };
    assert!(
        Ppo::new_with_joint_gradient_clipping(ActorCritic { policy, value }, config, 0, 1., 0.1)
            .is_err()
    );
}

#[test]
fn joint_checkpoint_checks_restored_parameter_ids_instead_of_template_ids() {
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let opt = OptimizerConfig {
        max_grad_norm: None,
        ..Default::default()
    };
    let config = PpoConfig {
        policy: opt.clone(),
        value: opt,
        ..Default::default()
    };
    let policy = Policy::new(2, ActionSpace::Discrete { count: 1 }, &[], &mut rng).unwrap();
    let tied = ActorCritic {
        value: Value {
            network: policy.network.clone(),
        },
        policy,
    };
    let tied_record = Ppo::new(tied.clone(), config.clone(), 1)
        .unwrap()
        .checkpoint()
        .unwrap();
    let mut bad = tied_record;
    bad.metadata["joint_optimization"] =
        serde_json::json!({"value_loss_coefficient": 1.0, "max_grad_norm": 0.1});
    let independent = || {
        let mut rng = ChaCha8Rng::seed_from_u64(10);
        ActorCritic {
            policy: Policy::new(2, ActionSpace::Discrete { count: 1 }, &[], &mut rng).unwrap(),
            value: Value::new(2, &[], &mut rng).unwrap(),
        }
    };
    let mut target = Ppo::new(independent(), config.clone(), 1).unwrap();
    let before = serde_json::to_value(target.checkpoint().unwrap()).unwrap();
    assert!(target.restore(&bad).is_err());
    assert_eq!(
        serde_json::to_value(target.checkpoint().unwrap()).unwrap(),
        before
    );

    let good = Ppo::new_with_joint_gradient_clipping(independent(), config.clone(), 1, 1., 0.1)
        .unwrap()
        .checkpoint()
        .unwrap();
    let mut tied_template = Ppo::new(tied, config, 1).unwrap();
    tied_template.restore(&good).unwrap();
}

#[derive(Deserialize)]
struct SequenceFixture {
    steps: Vec<SequenceStep>,
}

#[derive(Deserialize)]
struct SequenceStep {
    selection: Vec<usize>,
    policy_loss: f32,
    value_loss: f32,
    approx_kl: f32,
    gradient_norm: f32,
    policy_weight: Vec<f32>,
    policy_bias: Vec<f32>,
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
    moments: BTreeMap<String, Vec<f32>>,
}

#[test]
fn sequential_minibatches_preserve_sb3_adam_and_joint_update_parity() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sb3_ppo_multistep.json");
    let sequence: SequenceFixture = serde_json::from_slice(
        &std::fs::read(path).expect("generate the SB3 multi-step PPO fixture"),
    )
    .unwrap();
    let expected = [
        [0, 1],
        [2, 3],
        [1, 3],
        [2, 0],
        [0, 3],
        [1, 2],
        [2, 3],
        [0, 1],
    ];
    assert_eq!(sequence.steps.len(), expected.len());
    let (learner, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let mut learner = learner.with_minibatch_advantage_normalization(true);
    for (i, (step, selection)) in sequence.steps.iter().zip(expected).enumerate() {
        assert_eq!(step.selection, selection);
        assert!(
            step.gradient_norm > 0.1,
            "clipping must be active at step {i}"
        );
        let report = learner.update(&batch.select(&selection).unwrap()).unwrap();
        assert_eq!(report.updates, 1);
        assert_eq!(learner.updates(), (i + 1) as u64);
        for (key, target) in [
            ("policy_loss", step.policy_loss),
            ("value_loss", step.value_loss),
            ("approx_kl", step.approx_kl),
            ("joint_gradient_norm", step.gradient_norm),
        ] {
            compare(
                &format!("step {i} {key}"),
                &[report.metrics[key] as f32],
                &[target],
                2e-6,
            );
        }
        let moments = optimizer_moments(&learner);
        assert_eq!(moments.len(), step.moments.len());
        for (name, expected) in &step.moments {
            if name.ends_with("moment_2") {
                compare_second_moment(&format!("step {i} {name}"), &moments[name], expected);
            } else {
                compare(&format!("step {i} {name}"), &moments[name], expected, 1e-8);
            }
        }
        let policy = learner.policy();
        let value = learner.value();
        for (name, actual, expected) in [
            (
                "policy weight",
                to_vec(policy.network.layers[0].weight.val()).unwrap(),
                &step.policy_weight,
            ),
            (
                "policy bias",
                to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
                &step.policy_bias,
            ),
            (
                "value weight",
                to_vec(value.network.layers[0].weight.val()).unwrap(),
                &step.value_weight,
            ),
            (
                "value bias",
                to_vec(value.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
                &step.value_bias,
            ),
        ] {
            compare(&format!("step {i} {name}"), &actual, expected, 2e-6);
        }
    }
}

#[test]
fn scheduled_settings_validate_atomically_and_clear_to_config() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (mut learner, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let initial = learner.update_settings();
    assert_eq!(
        initial.policy_learning_rate,
        learner.config().policy.learning_rate
    );
    assert_eq!(
        initial.value_learning_rate,
        learner.config().value.learning_rate
    );
    assert_eq!(initial.clip, learner.config().clip);
    learner.update(&batch).unwrap();
    let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    for bad in [
        PpoUpdateSettings {
            policy_learning_rate: -0.1,
            ..initial
        },
        PpoUpdateSettings {
            policy_learning_rate: f64::NAN,
            ..initial
        },
        PpoUpdateSettings {
            value_learning_rate: f64::INFINITY,
            ..initial
        },
        PpoUpdateSettings {
            clip: -0.1,
            ..initial
        },
        PpoUpdateSettings {
            clip: 1.,
            ..initial
        },
        PpoUpdateSettings {
            clip: f32::NAN,
            ..initial
        },
    ] {
        assert!(learner.set_update_settings(bad).is_err());
        assert_eq!(
            serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
            before
        );
    }
    let changed = PpoUpdateSettings {
        policy_learning_rate: 0.0001,
        value_learning_rate: 0.0007,
        clip: 0.05,
    };
    learner.set_update_settings(changed).unwrap();
    assert_eq!(learner.update_settings(), changed);
    assert_eq!(optimizer_moments(&learner), {
        let (mut restored, _) = build(&f, None, false);
        let original: rl::core::checkpoint::LearnerRecord = serde_json::from_value(before).unwrap();
        restored.restore(&original).unwrap();
        optimizer_moments(&restored)
    });
    learner.clear_update_settings();
    assert_eq!(learner.update_settings(), initial);
    assert!(
        learner
            .checkpoint()
            .unwrap()
            .metadata
            .get("update_settings")
            .is_none()
    );
    let legacy = learner.checkpoint().unwrap();
    learner.set_update_settings(changed).unwrap();
    learner.restore(&legacy).unwrap();
    assert_eq!(learner.update_settings(), initial);
}

#[test]
fn scheduled_zero_rate_preserves_parameters_but_advances_optimizer_and_restores() {
    let f = load_fixture("sb3_ppo_normalized.json");
    for joint in [false, true] {
        let (mut original, batch) = build_coefficient(&f, Some(0.1), joint, 0.5);
        let zero = PpoUpdateSettings {
            policy_learning_rate: 0.,
            value_learning_rate: 0.,
            clip: 0.,
        };
        original.set_update_settings(zero).unwrap();
        let policy_before = to_vec(original.policy().network.layers[0].weight.val()).unwrap();
        let value_before = to_vec(original.value().network.layers[0].weight.val()).unwrap();
        assert_eq!(original.update(&batch).unwrap().updates, 1);
        assert_eq!(
            to_vec(original.policy().network.layers[0].weight.val()).unwrap(),
            policy_before
        );
        assert_eq!(
            to_vec(original.value().network.layers[0].weight.val()).unwrap(),
            value_before
        );
        assert!(
            optimizer_moments(&original)
                .values()
                .flatten()
                .any(|x| *x != 0.)
        );
        let record = original.checkpoint().unwrap();
        let (mut restored, _) = build(&f, None, false);
        restored.restore(&record).unwrap();
        assert_eq!(restored.update_settings(), zero);
        let next = PpoUpdateSettings {
            policy_learning_rate: 0.0002,
            value_learning_rate: 0.0004,
            clip: 0.1,
        };
        original.set_update_settings(next).unwrap();
        restored.set_update_settings(next).unwrap();
        assert_eq!(
            original.update(&batch).unwrap().metrics,
            restored.update(&batch).unwrap().metrics
        );
        assert_eq!(optimizer_moments(&original), optimizer_moments(&restored));
        assert_eq!(
            to_vec(original.policy().network.layers[0].weight.val()).unwrap(),
            to_vec(restored.policy().network.layers[0].weight.val()).unwrap()
        );
        assert_eq!(
            to_vec(original.value().network.layers[0].weight.val()).unwrap(),
            to_vec(restored.value().network.layers[0].weight.val()).unwrap()
        );
        let mut corrupt = record;
        corrupt.metadata["update_settings"]["clip"] = (-0.1).into();
        let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
        assert!(restored.restore(&corrupt).is_err());
        assert_eq!(
            serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
            before
        );
    }
}

#[test]
fn scheduled_clip_changes_ppo_objective_without_changing_stored_config() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let (mut normal, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let (mut clipped, _) = build_coefficient(&f, Some(0.1), true, 0.5);
    let mut settings = clipped.update_settings();
    settings.clip = 0.;
    clipped.set_update_settings(settings).unwrap();
    assert_eq!(clipped.config().clip, 0.2);
    let a = normal.update(&batch).unwrap();
    let b = clipped.update(&batch).unwrap();
    assert!((a.metrics["policy_loss"] - b.metrics["policy_loss"]).abs() > 1e-4);
}

#[derive(Deserialize)]
struct ScheduledFixture {
    steps: Vec<ScheduledStep>,
}

#[derive(Deserialize)]
struct ScheduledStep {
    learning_rate: f64,
    clip: f32,
    #[serde(flatten)]
    reference: SequenceStep,
}

#[test]
fn scheduled_minibatches_match_sb3_with_persistent_adam_state() {
    let f = load_fixture("sb3_ppo_normalized.json");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sb3_ppo_scheduled.json");
    let sequence: ScheduledFixture = serde_json::from_slice(
        &std::fs::read(path).expect("generate the SB3 scheduled PPO fixture"),
    )
    .unwrap();
    let expected = [
        ([0, 1], 0.0003, 0.2),
        ([2, 3], 0.00025, 0.17),
        ([1, 3], 0.0002, 0.14),
        ([2, 0], 0.00015, 0.11),
        ([0, 3], 0.0001, 0.08),
        ([1, 2], 0., 0.),
        ([2, 3], 0.00005, 0.03),
        ([0, 1], 0.00002, 0.01),
    ];
    assert_eq!(sequence.steps.len(), expected.len());
    let (learner, batch) = build_coefficient(&f, Some(0.1), true, 0.5);
    let mut learner = learner.with_minibatch_advantage_normalization(true);
    for (i, (step, (selection, rate, clip))) in sequence.steps.iter().zip(expected).enumerate() {
        assert_eq!(step.reference.selection, selection);
        assert_eq!((step.learning_rate, step.clip), (rate, clip));
        learner
            .set_update_settings(PpoUpdateSettings {
                policy_learning_rate: rate,
                value_learning_rate: rate,
                clip,
            })
            .unwrap();
        let report = learner.update(&batch.select(&selection).unwrap()).unwrap();
        assert_eq!(report.updates, 1);
        assert_eq!(learner.updates(), (i + 1) as u64);
        for (key, value) in [
            ("policy_loss", step.reference.policy_loss),
            ("value_loss", step.reference.value_loss),
            ("approx_kl", step.reference.approx_kl),
            ("joint_gradient_norm", step.reference.gradient_norm),
        ] {
            compare(
                &format!("scheduled {i} {key}"),
                &[report.metrics[key] as f32],
                &[value],
                2e-6,
            );
        }
        let moments = optimizer_moments(&learner);
        assert_eq!(moments.len(), step.reference.moments.len());
        for (name, expected) in &step.reference.moments {
            if name.ends_with("moment_2") {
                compare_second_moment(&format!("scheduled {i} {name}"), &moments[name], expected);
            } else {
                compare(
                    &format!("scheduled {i} {name}"),
                    &moments[name],
                    expected,
                    1e-8,
                );
            }
        }
        let policy = learner.policy();
        let value = learner.value();
        for (name, actual, expected) in [
            (
                "policy weight",
                to_vec(policy.network.layers[0].weight.val()).unwrap(),
                &step.reference.policy_weight,
            ),
            (
                "policy bias",
                to_vec(policy.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
                &step.reference.policy_bias,
            ),
            (
                "value weight",
                to_vec(value.network.layers[0].weight.val()).unwrap(),
                &step.reference.value_weight,
            ),
            (
                "value bias",
                to_vec(value.network.layers[0].bias.as_ref().unwrap().val()).unwrap(),
                &step.reference.value_bias,
            ),
        ] {
            compare(&format!("scheduled {i} {name}"), &actual, expected, 2e-6);
        }
    }
}

#[test]
fn scheduled_separate_rates_affect_only_the_corresponding_model() {
    let f = load_fixture("sb3_ppo_normalized.json");
    for joint in [false, true] {
        for (policy_rate, value_rate) in [(0., 0.0003), (0.0003, 0.)] {
            let (mut learner, batch) = build_coefficient(&f, Some(0.1), joint, 0.5);
            learner
                .set_update_settings(PpoUpdateSettings {
                    policy_learning_rate: policy_rate,
                    value_learning_rate: value_rate,
                    clip: 0.2,
                })
                .unwrap();
            let policy = to_vec(learner.policy().network.layers[0].weight.val()).unwrap();
            let value = to_vec(learner.value().network.layers[0].weight.val()).unwrap();
            assert_eq!(learner.update(&batch).unwrap().updates, 1);
            assert_eq!(
                to_vec(learner.policy().network.layers[0].weight.val()).unwrap() == policy,
                policy_rate == 0.
            );
            assert_eq!(
                to_vec(learner.value().network.layers[0].weight.val()).unwrap() == value,
                value_rate == 0.
            );
            let moments = optimizer_moments(&learner);
            assert!(
                moments
                    .iter()
                    .filter(|(k, _)| k.starts_with("policy."))
                    .any(|(_, v)| v.iter().any(|x| *x != 0.))
            );
            assert!(
                moments
                    .iter()
                    .filter(|(k, _)| k.starts_with("value."))
                    .any(|(_, v)| v.iter().any(|x| *x != 0.))
            );
        }
    }
}
