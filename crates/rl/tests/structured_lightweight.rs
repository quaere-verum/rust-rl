#![cfg(feature = "neural")]

use burn::{module::Module, tensor::Device};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::checkpoint::Checkpoint,
    nn::{structured::*, to_vec},
};
#[path = "support/structured_semantics.rs"]
mod semantics;

fn config() -> LightweightConfig {
    LightweightConfig::small(InputSchema {
        version: 2,
        semantics: semantics::identity([2, 2, 2, 2, 1], &["repair", "return", "move"]),
        spatial_channels: 2,
        entity_features: 2,
        event_features: 2,
        scalar_features: 2,
        candidate_features: 1,
        operations: vec![
            ArgumentKind::Entity,
            ArgumentKind::None,
            ArgumentKind::Location,
        ],
    })
}
fn sample() -> StructuredSample<String> {
    StructuredSample {
        primary_region: 0,
        regions: vec![],
        width: 3,
        height: 2,
        spatial: vec![0.; 12],
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
        candidates: vec![
            Candidate {
                key: "g".into(),
                operation: 0,
                argument: Argument::Entity(0),
                features: vec![0.3],
                enabled: true,
            },
            Candidate {
                key: "r".into(),
                operation: 0,
                argument: Argument::Entity(1),
                features: vec![0.6],
                enabled: true,
            },
            Candidate {
                key: "base".into(),
                operation: 1,
                argument: Argument::None,
                features: vec![0.1],
                enabled: false,
            },
            Candidate {
                key: "destination".into(),
                operation: 2,
                argument: Argument::Location { x: 2, y: 1 },
                features: vec![0.8],
                enabled: true,
            },
        ],
    }
}
fn model(seed: u64, device: &Device) -> LightweightQ {
    LightweightQ::new(config(), &mut ChaCha8Rng::seed_from_u64(seed), device).unwrap()
}
fn values(q: &LightweightQ, samples: &[StructuredSample<String>]) -> Vec<Vec<f32>> {
    q.candidate_values(samples, PackLimits::default())
        .unwrap()
        .into_iter()
        .map(|v| to_vec(v.values).unwrap())
        .collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 3e-5, "{a} != {b}");
    }
}

#[test]
fn lightweight_ragged_permutation_and_identity_contract() {
    let q = model(19, &Device::flex());
    let original = sample();
    let expected = values(&q, std::slice::from_ref(&original)).remove(0);
    let mut permuted = original.clone();
    permuted.entities.reverse();
    for event in &mut permuted.events {
        event.entity = event.entity.map(|i| 1 - i);
        for i in &mut event.context {
            *i = 1 - *i;
        }
        event.context.reverse();
    }
    for c in &mut permuted.candidates {
        if let Argument::Entity(i) = &mut c.argument {
            *i = 1 - *i;
        }
        c.key = format!("opaque-renamed-{}", c.key);
    }
    permuted.candidates.reverse();
    let mut actual = values(&q, &[permuted]).remove(0);
    actual.reverse();
    close(&expected, &actual);
    let mut other = original.clone();
    other.width = 8;
    other.height = 4;
    other.spatial = vec![0.7; 64];
    other.entities.extend([vec![0., 0.], vec![1., 1.]]);
    other.events.extend(original.events.clone());
    let mixed = values(&q, &[original.clone(), other.clone()]);
    close(&expected, &mixed[0]);
    close(&values(&q, &[other])[0], &mixed[1]);
    let rows = q
        .candidate_values(std::slice::from_ref(&original), PackLimits::default())
        .unwrap();
    let maxima = rows[0].operation_maxima();
    assert!(
        (maxima[0].clone().unwrap().into_scalar::<f32>() - expected[0].max(expected[1])).abs()
            < 1e-6
    );
    assert!(maxima[1].is_none());
    assert_ne!(rows[0].greedy_key().unwrap(), "base");
}

#[test]
fn no_map_tensor_but_canonical_validation_and_allocation_limits_remain() {
    let q = model(7, &Device::flex());
    let original = sample();
    let first = LightweightBatch::pack(
        &config().input,
        std::slice::from_ref(&original),
        PackLimits::default(),
        &Device::flex(),
    )
    .unwrap();
    let mut pixels = original.clone();
    pixels.spatial.fill(-8.);
    assert_eq!(
        values(&q, std::slice::from_ref(&original)),
        values(&q, std::slice::from_ref(&pixels))
    );
    pixels.width = 128;
    pixels.height = 128;
    pixels.spatial = vec![0.; 2 * 128 * 128];
    let large = LightweightBatch::pack(
        &config().input,
        std::slice::from_ref(&pixels),
        PackLimits::default(),
        &Device::flex(),
    )
    .unwrap();
    assert_eq!(first.input_elements(), large.input_elements());
    assert!(
        LightweightBatch::pack(
            &config().input,
            std::slice::from_ref(&original),
            PackLimits {
                max_elements: first.input_elements() - 1
            },
            &Device::flex()
        )
        .is_err()
    );
    pixels.spatial[0] = f32::NAN;
    assert!(
        q.candidate_values(&[pixels], PackLimits::default())
            .is_err()
    );
    let mut bad = original.clone();
    bad.events[0].context.push(1);
    assert!(q.candidate_values(&[bad], PackLimits::default()).is_err());
    let mut empty = original;
    empty.entities.clear();
    empty.events.clear();
    empty.width = 1;
    empty.height = 1;
    empty.spatial = vec![0.; 2];
    empty.candidates.retain(|c| c.operation == 1);
    empty.candidates[0].enabled = true;
    assert!(values(&q, &[empty])[0][0].is_finite());
}

#[test]
fn evidence_links_order_and_multiplicity_affect_representation() {
    let q = model(37, &Device::flex());
    let a = sample();
    let original = values(&q, std::slice::from_ref(&a));
    let mut relinked = a.clone();
    relinked.events[0].entity = Some(1);
    let mut reordered = a.clone();
    reordered.events.reverse();
    let mut repeated = a.clone();
    repeated.events.push(a.events[0].clone());
    let mut context = a.clone();
    context.events[0].context.clear();
    for changed in [relinked, reordered, repeated, context] {
        assert!(
            original[0]
                .iter()
                .zip(&values(&q, &[changed])[0])
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }
}

#[test]
fn target_subject_and_context_summaries_match_independent_means() {
    let q = model(37, &Device::flex());
    let mut original = sample();
    // Identical base entity features isolate join placement from event content.
    original.entities[1] = original.entities[0].clone();
    let encode = |s| {
        let batch = LightweightBatch::pack(
            &config().input,
            &[s],
            PackLimits::default(),
            &Device::flex(),
        )
        .unwrap();
        q.encoder.forward(&batch).unwrap()
    };
    let world = encode(original.clone());
    let events = to_vec(world.events).unwrap();
    let targets = to_vec(world.entities).unwrap();
    let cfg = config();
    let e = cfg.entity_width;
    let f = cfg.event_width;
    let w = cfg.target_width();
    close(&targets[e..e + f], &events[..f]);
    close(&targets[w + e..w + e + f], &events[f..2 * f]);
    close(&targets[e + f..e + 2 * f], &events[f..2 * f]);
    let mean: Vec<_> = events[..f]
        .iter()
        .zip(&events[f..2 * f])
        .map(|(a, b)| (a + b) / 2.)
        .collect();
    close(&targets[w + e + f..w + e + 2 * f], &mean);
    close(&targets[w - 2..w], &[2f32.ln(), 2f32.ln()]);
    close(&targets[2 * w - 2..2 * w], &[2f32.ln(), 3f32.ln()]);
    original.events[0].entity = Some(1);
    let relinked = encode(original);
    close(&events, &to_vec(relinked.events).unwrap());
    let relinked = to_vec(relinked.entities).unwrap();
    close(&relinked[e..e + f], &vec![0.; f]);
    close(&relinked[w + e..w + e + f], &mean);
    close(&relinked[e + f..e + 2 * f], &targets[e + f..e + 2 * f]);
    close(
        &relinked[w + e + f..w + e + 2 * f],
        &targets[w + e + f..w + e + 2 * f],
    );
    assert_eq!(relinked[w - 2], 0.);
    assert!((relinked[2 * w - 2] - 3f32.ln()).abs() < 1e-6);
}

#[test]
fn selected_candidate_has_gradients_in_every_branch() {
    let q = model(37, &Device::flex().autodiff());
    let samples = [sample()];
    let rows = q.candidate_values(&samples, PackLimits::default()).unwrap();
    let gradients = (rows[0].values.clone().narrow(0, 0, 1) - 1.)
        .powi_scalar(2)
        .mean()
        .backward();
    for network in [
        &q.encoder.entities,
        &q.encoder.events,
        &q.encoder.scalars,
        &q.encoder.fusion,
        &q.operations,
        &q.head,
    ] {
        let g = to_vec(
            network.layers[0]
                .weight
                .val()
                .grad(&gradients)
                .expect("connected branch"),
        )
        .unwrap();
        assert!(g.iter().all(|v| v.is_finite()));
        assert!(g.iter().any(|v| v.abs() > 1e-9), "zero branch gradient");
    }
}

#[test]
fn lightweight_checkpoint_binds_semantics_and_execution_mode() {
    let q = model(19, &Device::flex().autodiff());
    let record = q.checkpoint().unwrap();
    let mut restored = model(99, &Device::flex());
    restored
        .restore(&serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap())
        .unwrap();
    assert!(
        !restored.encoder.entities.layers[0]
            .weight
            .val()
            .is_require_grad()
    );
    assert_eq!(
        values(&q.valid(), &[sample()]),
        values(&restored, &[sample()])
    );
    let mut wrong = record.clone();
    wrong.metadata["version"] = 99.into();
    assert!(restored.restore(&wrong).is_err());
    let mut changed = config();
    changed.input.semantics.operations.swap(0, 1);
    let mut incompatible =
        LightweightQ::new(changed, &mut ChaCha8Rng::seed_from_u64(19), &Device::flex()).unwrap();
    assert!(incompatible.restore(&record).is_err());
    let mut tiny = config();
    tiny.max_activation_elements = 1;
    let limited =
        LightweightQ::new(tiny, &mut ChaCha8Rng::seed_from_u64(7), &Device::flex()).unwrap();
    assert!(
        limited
            .candidate_values(&[sample()], PackLimits::default())
            .is_err()
    );
}
