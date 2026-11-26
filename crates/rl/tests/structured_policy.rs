#![cfg(feature = "neural")]

use burn::tensor::Device;
#[path = "support/structured_semantics.rs"]
mod semantics;

#[test]
fn historical_conditional_likelihoods_gradients_and_rng_order_are_preserved() {
    use rl::core::checkpoint::Checkpoint;
    use serde_json::json;
    let mut cfg = config();
    cfg.encoder.conv_channels = [2, 2];
    cfg.encoder.entity_width = 4;
    cfg.encoder.event_width = 4;
    cfg.encoder.scalar_width = 4;
    cfg.encoder.fused_width = 4;
    cfg.head_width = 4;
    let device = Device::flex().autodiff();
    let mut policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(31), &device).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/structured_policy_v2.json")).unwrap();
    let record = serde_json::from_value(fixture["record"].clone()).unwrap();
    let samples: Vec<StructuredSample<String>> =
        serde_json::from_value(fixture["samples"].clone()).unwrap();
    policy.restore(&record).unwrap();
    let batch = StructuredBatch::pack(&cfg.encoder.input, &samples, PackLimits::default(), &device)
        .unwrap();
    let evaluated = policy.evaluate(&batch).unwrap();
    let operations = to_vec(
        evaluated
            .operations()
            .categorical_terms()
            .unwrap()
            .log_probs,
    )
    .unwrap();
    // Distribution terms use zero placeholders for masked log-probabilities.
    // Keep the masks explicitly so these cannot be mistaken for live log(1).
    let masks = match evaluated.operations() {
        rl::nn::distributions::Distribution::Categorical { masks, .. } => masks.clone(),
        _ => unreachable!(),
    };
    let operations: Vec<_> = operations
        .into_iter()
        .map(|x| x.is_finite().then_some(x))
        .collect();
    let probabilities: Vec<_> = (0..2)
        .map(|i| to_vec(evaluated.complete_probabilities(i).unwrap()).unwrap())
        .collect();
    let keys = ["a", "b", "x", "y", "wait"].map(str::to_owned);
    let logs: Vec<_> = (0..2)
        .map(|i| to_vec(evaluated.log_prob_many(i, &keys).unwrap()).unwrap())
        .collect();
    let singles = to_vec(evaluated.log_prob(&["a".into(), "y".into()]).unwrap()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(789);
    let draws: Vec<_> = (0..12)
        .map(|_| {
            evaluated
                .sample(ActMode::Sample, &mut rng)
                .unwrap()
                .into_iter()
                .map(|d| json!({"key":d.key,"log_prob":d.log_prob}))
                .collect::<Vec<_>>()
        })
        .collect();
    let loss = -evaluated.log_prob_many(0, &keys).unwrap().mean();
    let gradients = loss.backward();
    let leaves = [
        &policy.operation_head.layers[0].weight,
        &policy.entity_query.layers[0].weight,
        &policy.entity_key.layers[0].weight,
        &policy.spatial_query.layers[0].weight,
        &policy.spatial_key.layers[0].weight,
    ];
    let gradients: Vec<_> = leaves
        .iter()
        .map(|p| p.val().grad(&gradients).map(|g| to_vec(g).unwrap()))
        .collect();
    let outputs = json!({"operations":operations,"masks":masks,"probabilities":probabilities,"logs":logs,
        "single_teacher":singles,"draws":draws,"rng":rng,"gradients":gradients});
    assert_eq!(
        outputs, fixture["outputs"],
        "historical policy or RNG drift"
    );
}
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::actor::ActMode,
    nn::{structured::*, to_vec},
};

fn config() -> StructuredModelConfig {
    StructuredModelConfig::small(InputSchema {
        version: 2,
        semantics: semantics::identity(
            [2, 2, 1, 1, 1],
            &["entity", "location", "wait", "other_entity"],
        ),
        spatial_channels: 2,
        entity_features: 2,
        event_features: 1,
        scalar_features: 1,
        candidate_features: 1,
        operations: vec![
            ArgumentKind::Entity,
            ArgumentKind::Location,
            ArgumentKind::None,
            ArgumentKind::Entity,
        ],
    })
}

#[test]
fn many_teacher_keys_match_single_key_likelihoods_without_resampling() {
    let input = sample();
    let cfg = config();
    let device = Device::flex().autodiff();
    let policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(23), &device).unwrap();
    let batch = StructuredBatch::pack(
        &cfg.encoder.input,
        std::slice::from_ref(&input),
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let evaluated = policy.evaluate(&batch).unwrap();
    let keys: Vec<_> = input
        .candidates
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.key.clone())
        .collect();
    let many = evaluated.log_prob_many(0, &keys).unwrap();
    let actual = to_vec(many.clone()).unwrap();
    for (key, value) in keys.iter().zip(actual) {
        assert!(
            (value - to_vec(evaluated.log_prob(std::slice::from_ref(key)).unwrap()).unwrap()[0])
                .abs()
                < 1e-6
        );
    }
    let gradients = many.sum().neg().backward();
    assert!(
        policy.entity_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
    assert!(
        policy.spatial_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
    assert!(evaluated.log_prob_many(0, &["masked".to_owned()]).is_err());
    assert!(evaluated.log_prob_many(1, &keys).is_err());
}
fn sample() -> StructuredSample<String> {
    let candidates = [
        ("a", 0, Argument::Entity(0), true),
        ("b", 0, Argument::Entity(1), true),
        ("masked", 0, Argument::Entity(2), false),
        ("x", 1, Argument::Location { x: 0, y: 0 }, true),
        ("y", 1, Argument::Location { x: 2, y: 1 }, true),
        ("wait", 2, Argument::None, true),
    ]
    .into_iter()
    .map(|(key, operation, argument, enabled)| Candidate {
        key: key.into(),
        operation,
        argument,
        features: vec![0.2],
        enabled,
    })
    .collect();
    StructuredSample {
        primary_region: 0,
        regions: vec![],
        width: 3,
        height: 2,
        spatial: (0..12).map(|i| i as f32 / 12.).collect(),
        entities: vec![vec![0.1, 0.2], vec![0.8, 0.4], vec![0.3, 0.1]],
        scalars: vec![0.5],
        events: vec![Event {
            features: vec![0.9],
            entity: Some(0),
            context: vec![],
        }],
        candidates,
    }
}
#[test]
fn conditional_probabilities_normalize_and_teacher_forcing_matches_sampling() {
    let cfg = config();
    let device = Device::flex();
    let policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(1), &device).unwrap();
    let batch = StructuredBatch::pack(
        &cfg.encoder.input,
        &[sample()],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let evaluation = policy.evaluate(&batch).unwrap();
    let ops = to_vec(
        evaluation
            .operations()
            .categorical_terms()
            .unwrap()
            .probabilities,
    )
    .unwrap();
    assert!((ops.iter().sum::<f32>() - 1.).abs() < 1e-6);
    assert_eq!(ops[3], 0.);
    let joint = to_vec(evaluation.complete_probabilities(0).unwrap()).unwrap();
    assert!((joint.iter().sum::<f32>() - 1.).abs() < 1e-6);
    assert_eq!(joint[2], 0.);
    assert_eq!(joint[5], ops[2]);
    for operation in [0, 1] {
        let conditional = evaluation.conditional(0, operation).unwrap().unwrap();
        let p = to_vec(
            conditional
                .distribution
                .categorical_terms()
                .unwrap()
                .probabilities,
        )
        .unwrap();
        assert!((p.iter().sum::<f32>() - 1.).abs() < 1e-6);
        for (i, &row) in conditional.rows.iter().enumerate() {
            assert!((joint[row] - ops[operation] * p[i]).abs() < 1e-6);
        }
    }
    assert!(evaluation.conditional(0, 2).unwrap().is_none());
    assert!(evaluation.conditional(0, 3).is_err());
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    for _ in 0..50 {
        let decision = evaluation.sample(ActMode::Sample, &mut rng).unwrap();
        let lp = to_vec(evaluation.log_prob(&[decision[0].key.clone()]).unwrap()).unwrap();
        assert!((lp[0] - decision[0].log_prob).abs() < 1e-6);
    }
    assert!(evaluation.log_prob(&["masked".to_string()]).is_err());
}

#[test]
fn semantic_predictions_survive_entity_candidate_and_map_padding_changes() {
    let cfg = config();
    let device = Device::flex();
    let policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(4), &device).unwrap();
    let q = StructuredQ::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(5), &device).unwrap();
    let first = sample();
    let mut changed = first.clone();
    changed.entities.reverse();
    changed.candidates.reverse();
    for c in &mut changed.candidates {
        if let Argument::Entity(i) = &mut c.argument {
            *i = 2 - *i;
        }
    }
    changed.events[0].entity = Some(2);
    let mut large = sample();
    large.width = 5;
    large.height = 4;
    large.spatial = vec![0.5; 40];
    let a = StructuredBatch::pack(&cfg.encoder.input, &[first], PackLimits::default(), &device)
        .unwrap();
    let b = StructuredBatch::pack(
        &cfg.encoder.input,
        &[changed, large],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let pa = to_vec(
        policy
            .evaluate(&a)
            .unwrap()
            .complete_probabilities(0)
            .unwrap(),
    )
    .unwrap();
    let pb = to_vec(
        policy
            .evaluate(&b)
            .unwrap()
            .complete_probabilities(0)
            .unwrap(),
    )
    .unwrap();
    let qa = q.evaluate(&a).unwrap();
    let qb = q.evaluate(&b).unwrap();
    let va = to_vec(qa[0].values.clone()).unwrap();
    let vb = to_vec(qb[0].values.clone()).unwrap();
    for i in 0..6 {
        assert!((pa[i] - pb[5 - i]).abs() < 3e-5);
        assert!((va[i] - vb[5 - i]).abs() < 3e-5);
    }
    let maxima = qa[0].operation_maxima();
    assert_eq!(maxima.len(), cfg.encoder.input.operations.len());
    assert!(
        (maxima[0].as_ref().unwrap().clone().into_scalar::<f32>() - va[0].max(va[1])).abs() < 1e-6
    );
    assert!(maxima[3].is_none());
    let best = qa[0].greedy_key().unwrap();
    assert!(a.candidates(0).iter().any(|c| c.enabled && &c.key == best));
}

#[test]
fn teacher_forcing_differentiates_only_the_supplied_pointer_and_matches_finite_difference() {
    use burn::{
        module::{Module, Param},
        tensor::{Tensor, TensorData},
    };
    let cfg = config();
    let device = Device::flex().autodiff();
    let model =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(31), &device).unwrap();
    let batch = StructuredBatch::pack(
        &cfg.encoder.input,
        &[sample()],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let loss = -model
        .evaluate(&batch)
        .unwrap()
        .log_prob(&["x".into()])
        .unwrap()
        .mean();
    let gradients = loss.backward();
    assert!(
        model.entity_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    assert!(
        model.entity_key.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    for gradient in [
        model.spatial_query.layers[0].weight.val().grad(&gradients),
        model.spatial_key.layers[0].weight.val().grad(&gradients),
        model.operation_head.layers[0].weight.val().grad(&gradients),
    ] {
        assert!(
            to_vec(gradient.unwrap())
                .unwrap()
                .iter()
                .any(|g| g.abs() > 1e-7)
        );
    }
    let analytic = to_vec(
        model.spatial_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .unwrap(),
    )
    .unwrap();
    let index = (0..analytic.len())
        .max_by(|&i, &j| analytic[i].abs().total_cmp(&analytic[j].abs()))
        .unwrap();
    let shape = model.spatial_query.layers[0].weight.val().dims();
    let original = to_vec(model.spatial_query.layers[0].weight.val()).unwrap();
    let inference = StructuredBatch::pack(
        &cfg.encoder.input,
        &[sample()],
        PackLimits::default(),
        &Device::flex(),
    )
    .unwrap();
    let evaluate = |offset| {
        let mut candidate = model.valid();
        let mut values = original.clone();
        values[index] += offset;
        candidate.spatial_query.layers[0].weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, shape),
            &Device::flex(),
        ));
        -candidate
            .evaluate(&inference)
            .unwrap()
            .log_prob(&["x".into()])
            .unwrap()
            .into_scalar::<f32>()
    };
    let numeric = (evaluate(0.001) - evaluate(-0.001)) / 0.002;
    assert!(
        (numeric - analytic[index]).abs() < 0.002,
        "finite difference {numeric}, autodiff {}",
        analytic[index]
    );
}

#[test]
fn policy_and_q_supervised_updates_and_checkpoint_restoration() {
    use burn::{module::Module, optim::AdamConfig};
    use rl::{core::checkpoint::Checkpoint, nn::optimization::optimize_validated};
    let cfg = config();
    let device = Device::flex().autodiff();
    let mut policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(10), &device).unwrap();
    let mut q = StructuredQ::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(11), &device).unwrap();
    let batch = StructuredBatch::pack(
        &cfg.encoder.input,
        &[sample(), sample()],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let policy_loss = |p: &StructuredPolicy| {
        -p.evaluate(&batch)
            .unwrap()
            .log_prob(&["a".into(), "x".into()])
            .unwrap()
            .mean()
    };
    let q_loss = |q: &StructuredQ| {
        (q.evaluate(&batch).unwrap()[0].values.clone() - 0.4)
            .powi_scalar(2)
            .mean()
    };
    let initial = policy_loss(&policy).into_scalar::<f32>();
    let initial_q = q_loss(&q).into_scalar::<f32>();
    let mut opt = AdamConfig::new().init();
    let mut qopt = AdamConfig::new().init();
    for _ in 0..10 {
        let loss = policy_loss(&policy);
        optimize_validated(&mut policy, &mut opt, 0.003, loss).unwrap();
        let loss = q_loss(&q);
        optimize_validated(&mut q, &mut qopt, 0.001, loss).unwrap();
    }
    assert!(policy_loss(&policy).into_scalar::<f32>() < initial * 0.8);
    assert!(q_loss(&q).into_scalar::<f32>() < initial_q * 0.5);
    let grads = policy_loss(&policy).backward();
    assert!(
        to_vec(
            policy.entity_query.layers[0]
                .weight
                .val()
                .grad(&grads)
                .unwrap()
        )
        .unwrap()
        .iter()
        .any(|g| g.abs() > 1e-8)
    );
    let qgrads = q_loss(&q).backward();
    assert!(
        to_vec(q.head.layers[0].weight.val().grad(&qgrads).unwrap())
            .unwrap()
            .iter()
            .any(|g| g.abs() > 1e-8)
    );
    assert!(
        to_vec(q.encoder.conv1.weight.val().grad(&qgrads).unwrap())
            .unwrap()
            .iter()
            .any(|g| g.abs() > 1e-8)
    );
    let p_record =
        serde_json::from_value(serde_json::to_value(policy.checkpoint().unwrap()).unwrap())
            .unwrap();
    let q_record =
        serde_json::from_value(serde_json::to_value(q.checkpoint().unwrap()).unwrap()).unwrap();
    let mut restored =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(99), &device).unwrap();
    restored.restore(&p_record).unwrap();
    let mut restored_q =
        StructuredQ::new(cfg, &mut ChaCha8Rng::seed_from_u64(99), &device).unwrap();
    restored_q.restore(&q_record).unwrap();
    assert_eq!(
        to_vec(
            policy
                .evaluate(&batch)
                .unwrap()
                .complete_probabilities(0)
                .unwrap()
        )
        .unwrap(),
        to_vec(
            restored
                .evaluate(&batch)
                .unwrap()
                .complete_probabilities(0)
                .unwrap()
        )
        .unwrap()
    );
    assert_eq!(
        to_vec(q.evaluate(&batch).unwrap()[0].values.clone()).unwrap(),
        to_vec(restored_q.evaluate(&batch).unwrap()[0].values.clone()).unwrap()
    );
    let p = policy.valid();
    let r = restored.valid();
    let mut rng1 = ChaCha8Rng::seed_from_u64(7);
    let mut rng2 = rng1.clone();
    assert_eq!(
        p.evaluate(&batch)
            .unwrap()
            .sample(ActMode::Sample, &mut rng1)
            .unwrap(),
        r.evaluate(&batch)
            .unwrap()
            .sample(ActMode::Sample, &mut rng2)
            .unwrap()
    );
}

#[test]
fn structured_head_configuration_limits_include_pointer_parameters() {
    let mut cfg = config();
    cfg.head_width = 1;
    cfg.encoder.entity_width = 1024;
    cfg.encoder.input.candidate_features = 65536;
    assert!(cfg.validate().is_err());
}

#[test]
fn inference_restore_keeps_plain_devices_without_trainable_roots() {
    use rl::core::checkpoint::Checkpoint;
    let device = Device::flex();
    let cfg = config();
    let mut p =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(1), &device).unwrap();
    let mut q = StructuredQ::new(cfg, &mut ChaCha8Rng::seed_from_u64(2), &device).unwrap();
    p.restore(&p.checkpoint().unwrap()).unwrap();
    q.restore(&q.checkpoint().unwrap()).unwrap();
    assert!(!p.encoder.conv1.weight.val().is_autodiff());
    assert!(!p.operation_head.layers[0].weight.val().is_require_grad());
    assert!(!q.head.layers[0].weight.val().is_autodiff());
    assert!(!q.head.layers[0].weight.val().is_require_grad());
}

#[test]
fn masks_fallback_target_summaries_and_opaque_renaming() {
    let cfg = config();
    let device = Device::flex().autodiff();
    let mut policy =
        StructuredPolicy::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(10), &device).unwrap();
    let q = StructuredQ::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(11), &device).unwrap();
    let pack = |s| {
        StructuredBatch::pack(&cfg.encoder.input, &[s], PackLimits::default(), &device).unwrap()
    };
    let original = pack(sample());
    let expected = to_vec(
        policy
            .evaluate(&original)
            .unwrap()
            .complete_probabilities(0)
            .unwrap(),
    )
    .unwrap();
    let mut renamed = sample();
    for c in &mut renamed.candidates {
        c.key = format!("unrelated identifier {}", c.key);
    }
    let renamed = pack(renamed);
    assert_eq!(
        expected,
        to_vec(
            policy
                .evaluate(&renamed)
                .unwrap()
                .complete_probabilities(0)
                .unwrap()
        )
        .unwrap()
    );
    let mut removed = sample();
    removed.candidates[1].enabled = false;
    let removed = pack(removed);
    let logits = |b| match policy.evaluate(b).unwrap().operations().clone() {
        rl::nn::distributions::Distribution::Categorical { logits, .. } => to_vec(logits).unwrap(),
        _ => unreachable!(),
    };
    assert_ne!(logits(&original)[0], logits(&removed)[0]);
    for empty in [true, false] {
        let mut s = sample();
        for c in &mut s.candidates {
            c.enabled = false;
        }
        if empty {
            s.candidates.clear();
        }
        let batch = pack(s);
        assert!(policy.evaluate(&batch).is_err());
        assert!(q.evaluate(&batch).is_err());
    }
    // Teacher forcing a no-target action adds no pointer computation, even
    // with other operations available in this same decision.
    let loss = -policy
        .evaluate(&original)
        .unwrap()
        .log_prob(&["wait".into()])
        .unwrap()
        .mean();
    let gradients = loss.backward();
    assert!(
        policy.entity_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    assert!(
        policy.spatial_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    let mut fallback = sample();
    fallback.candidates.retain(|c| c.key == "wait");
    let fallback = pack(fallback);
    assert_eq!(
        to_vec(
            policy
                .evaluate(&fallback)
                .unwrap()
                .complete_probabilities(0)
                .unwrap()
        )
        .unwrap(),
        vec![1.]
    );
    use rl::core::checkpoint::Checkpoint;
    let before = policy.checkpoint().unwrap();
    let mut bad = before.clone();
    bad.metadata["head_width"] = serde_json::json!(99);
    assert!(policy.restore(&bad).is_err());
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(policy.checkpoint().unwrap()).unwrap()
    );
    policy.restore(&policy.checkpoint().unwrap()).unwrap();
    assert!(policy.encoder.conv1.weight.val().is_autodiff());
    assert!(policy.encoder.conv1.weight.val().is_require_grad());
}
