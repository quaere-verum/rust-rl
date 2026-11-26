#![cfg(feature = "neural")]

use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
#[path = "support/structured_semantics.rs"]
mod semantics;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::checkpoint::Checkpoint,
    nn::{structured::*, to_vec},
};

fn config() -> EncoderConfig {
    EncoderConfig::small(InputSchema {
        version: 2,
        semantics: semantics::identity([2, 2, 2, 2, 1], &["entity"]),
        spatial_channels: 2,
        entity_features: 2,
        event_features: 2,
        scalar_features: 2,
        candidate_features: 1,
        operations: vec![ArgumentKind::Entity],
    })
}
fn sample() -> StructuredSample<String> {
    StructuredSample {
        primary_region: 0,
        regions: vec![],
        width: 3,
        height: 2,
        spatial: (0..12).map(|x| (x as f32 - 3.) / 7.).collect(),
        entities: vec![vec![0.2, 0.9], vec![0.8, -0.3]],
        scalars: vec![0.6, 0.1],
        events: vec![
            Event {
                features: vec![0.1, 0.8],
                entity: Some(0),
                context: vec![1],
            },
            Event {
                features: vec![0.4, 0.2],
                entity: Some(1),
                context: vec![0, 1],
            },
        ],
        candidates: vec![],
    }
}
fn close(a: Tensor<2>, b: Tensor<2>) {
    for (x, y) in to_vec(a).unwrap().iter().zip(to_vec(b).unwrap()) {
        assert!((x - y).abs() < 3e-5, "{x} != {y}");
    }
}

#[test]
fn observing_encoder_stages_preserves_outputs_and_gradients() {
    let cfg = config();
    let device = Device::flex().autodiff();
    let model =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(19), &device).unwrap();
    let batch =
        StructuredBatch::pack(&cfg.input, &[sample()], PackLimits::default(), &device).unwrap();
    let plain = model.forward(&batch).unwrap();
    let mut stages = Vec::new();
    let observed = model
        .forward_observed(&batch, |stage| stages.push(stage))
        .unwrap();
    assert_eq!(
        stages,
        [
            EncoderStage::Validation,
            EncoderStage::Spatial,
            EncoderStage::Entities,
            EncoderStage::Events,
            EncoderStage::ScalarsAndFusion
        ]
    );
    assert_eq!(
        to_vec(plain.global.clone()).unwrap(),
        to_vec(observed.global.clone()).unwrap()
    );
    assert_eq!(
        to_vec(plain.entities).unwrap(),
        to_vec(observed.entities).unwrap()
    );
    assert_eq!(
        to_vec(plain.spatial).unwrap(),
        to_vec(observed.spatial).unwrap()
    );
    assert_eq!(
        to_vec(plain.events).unwrap(),
        to_vec(observed.events).unwrap()
    );
    let a = plain.global.powi_scalar(2).mean().backward();
    let b = observed.global.powi_scalar(2).mean().backward();
    for weight in [
        model.entities.layers[0].weight.val(),
        model.events.layers[0].weight.val(),
        model.scalars.layers[0].weight.val(),
        model.fusion.layers[0].weight.val(),
    ] {
        assert_eq!(
            to_vec(weight.grad(&a).unwrap()).unwrap(),
            to_vec(weight.grad(&b).unwrap()).unwrap()
        );
    }
    assert_eq!(
        to_vec(model.conv1.weight.val().grad(&a).unwrap()).unwrap(),
        to_vec(model.conv1.weight.val().grad(&b).unwrap()).unwrap()
    );
}
#[test]
fn encoder_ignores_local_padding_and_entity_row_order() {
    let cfg = config();
    let device = Device::flex();
    let encoder =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(19), &device).unwrap();
    let first = sample();
    let mut other = sample();
    other.width = 7;
    other.height = 5;
    other.spatial = vec![0.3; 70];
    other.entities.extend([vec![1., 2.], vec![3., 4.]]);
    other.events.extend(first.events.clone());
    let solo = StructuredBatch::pack(
        &cfg.input,
        std::slice::from_ref(&first),
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let mixed = StructuredBatch::pack(
        &cfg.input,
        &[first.clone(), other],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let a = encoder.forward(&solo).unwrap();
    let b = encoder.forward(&mixed).unwrap();
    close(a.global.clone(), b.global.slice([0..1, 0..cfg.fused_width]));
    let spatial_a = to_vec(a.spatial).unwrap();
    let spatial_b = to_vec(b.spatial).unwrap();
    for channel in 0..cfg.conv_channels[1] {
        for y in 0..2 {
            for x in 0..3 {
                assert!(
                    (spatial_a[(channel * 2 + y) * 3 + x] - spatial_b[(channel * 5 + y) * 7 + x])
                        .abs()
                        < 3e-5
                );
            }
        }
    }
    let mut permuted = first;
    permuted.entities.reverse();
    for e in &mut permuted.events {
        e.entity = e.entity.map(|i| 1 - i);
        for row in &mut e.context {
            *row = 1 - *row;
        }
    }
    let batch =
        StructuredBatch::pack(&cfg.input, &[permuted], PackLimits::default(), &device).unwrap();
    let p = encoder.forward(&batch).unwrap();
    close(a.global, p.global);
    let ea = to_vec(a.entities).unwrap();
    let ep = to_vec(p.entities).unwrap();
    assert_eq!(&ea[..cfg.entity_width], &ep[cfg.entity_width..]);
}

#[test]
fn convolution_keeps_adjacent_packed_regions_independent() {
    let cfg = config();
    let device = Device::flex();
    let encoder =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(31), &device).unwrap();
    let mut sample = sample();
    let separate = sample.clone();
    sample.regions.push(SpatialSample {
        id: 7,
        width: 2,
        height: 2,
        spatial: vec![100.; 8],
    });
    let features = |s, width, height| {
        let batch =
            StructuredBatch::pack(&cfg.input, &[s], PackLimits::default(), &device).unwrap();
        to_vec(encoder.forward(&batch).unwrap().spatial.slice([
            0..1,
            0..cfg.conv_channels[1],
            0..height,
            0..width,
        ]))
        .unwrap()
    };
    let a = features(separate, 3, 2);
    let b = features(sample, 3, 2);
    assert!(a.iter().zip(b).all(|(a, b)| (*a - b).abs() < 1e-6));
}

#[test]
fn deterministic_encoder_and_versioned_checkpoint_restore() {
    let cfg = config();
    let device = Device::flex();
    let a =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(7), &device).unwrap();
    let b =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(7), &device).unwrap();
    let batch =
        StructuredBatch::pack(&cfg.input, &[sample()], PackLimits::default(), &device).unwrap();
    assert_eq!(
        to_vec(a.forward(&batch).unwrap().global).unwrap(),
        to_vec(b.forward(&batch).unwrap().global).unwrap()
    );
    let record =
        serde_json::from_slice(&serde_json::to_vec(&a.checkpoint().unwrap()).unwrap()).unwrap();
    let mut restored =
        StructuredEncoder::new(cfg, &mut ChaCha8Rng::seed_from_u64(99), &device).unwrap();
    restored.restore(&record).unwrap();
    assert!(!restored.conv1.weight.val().is_autodiff());
    assert!(!restored.conv1.weight.val().is_require_grad());
    close(
        a.forward(&batch).unwrap().global,
        restored.valid().forward(&batch).unwrap().global,
    );
    let mut bad = record;
    bad.algorithm = "old-fixed-width".into();
    assert!(restored.restore(&bad).is_err());
}

#[test]
fn all_encoder_branches_receive_gradients_and_supervised_fit_improves() {
    use burn::optim::AdamConfig;
    use rl::nn::optimization::optimize_validated;
    let cfg = config();
    let device = Device::flex().autodiff();
    let mut model =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(37), &device).unwrap();
    let batch =
        StructuredBatch::pack(&cfg.input, &[sample()], PackLimits::default(), &device).unwrap();
    let loss = |m: &StructuredEncoder| {
        (m.forward(&batch).unwrap().global - 0.7)
            .powi_scalar(2)
            .mean()
    };
    let initial = loss(&model).into_scalar::<f32>();
    let original = model.checkpoint().unwrap();
    let mut opt = AdamConfig::new().init();
    for _ in 0..12 {
        let value = loss(&model);
        optimize_validated(&mut model, &mut opt, 0.003, value).unwrap();
    }
    assert!(loss(&model).into_scalar::<f32>() < initial * 0.5);
    assert_ne!(
        serde_json::to_value(original.models).unwrap(),
        serde_json::to_value(model.checkpoint().unwrap().models).unwrap()
    );
    let grads = loss(&model).backward();
    for gradient in [
        model
            .conv1
            .weight
            .val()
            .grad(&grads)
            .map(|g| to_vec(g).unwrap()),
        model.entities.layers[0]
            .weight
            .val()
            .grad(&grads)
            .map(|g| to_vec(g).unwrap()),
        model.events.layers[0]
            .weight
            .val()
            .grad(&grads)
            .map(|g| to_vec(g).unwrap()),
        model.scalars.layers[0]
            .weight
            .val()
            .grad(&grads)
            .map(|g| to_vec(g).unwrap()),
    ] {
        assert!(gradient.unwrap().iter().any(|g| g.abs() > 1e-9));
    }
}

#[test]
fn empty_streams_single_cell_and_activation_limits_are_explicit() {
    let cfg = config();
    let device = Device::flex();
    let mut first = sample();
    first.width = 1;
    first.height = 1;
    first.spatial = vec![0.1, 0.2];
    first.entities.clear();
    first.events.clear();
    let batch = StructuredBatch::pack(
        &cfg.input,
        std::slice::from_ref(&first),
        PackLimits::default(),
        &device,
    )
    .unwrap();
    let encoder =
        StructuredEncoder::new(cfg.clone(), &mut ChaCha8Rng::seed_from_u64(7), &device).unwrap();
    let solo = encoder.forward(&batch).unwrap();
    assert!(to_vec(solo.entities).unwrap().iter().all(|v| *v == 0.));
    assert!(to_vec(solo.events).unwrap().iter().all(|v| *v == 0.));
    let mixed = StructuredBatch::pack(
        &cfg.input,
        &[first, sample()],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    close(
        solo.global,
        encoder
            .forward(&mixed)
            .unwrap()
            .global
            .slice([0..1, 0..cfg.fused_width]),
    );
    let mut limited = cfg;
    limited.max_activation_elements = 1;
    let encoder =
        StructuredEncoder::new(limited, &mut ChaCha8Rng::seed_from_u64(7), &device).unwrap();
    assert!(encoder.forward(&mixed).is_err());
}
