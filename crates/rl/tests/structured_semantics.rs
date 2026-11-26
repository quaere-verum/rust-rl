#![cfg(feature = "neural")]

use burn::tensor::Device;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::checkpoint::{Checkpoint, LearnerRecord},
    nn::{structured::*, to_vec},
};
#[path = "support/structured_semantics.rs"]
mod semantics;

fn schema() -> InputSchema {
    let mut semantics =
        semantics::identity([2, 2, 2, 2, 2], &["inspect", "repair", "wait", "craft"]);
    semantics.vocabularies.push(VocabularyIdentity {
        name: "terrain".into(),
        categories: vec!["grass".into(), "rock".into()],
    });
    InputSchema {
        version: 2,
        semantics,
        spatial_channels: 2,
        entity_features: 2,
        event_features: 2,
        scalar_features: 2,
        candidate_features: 2,
        operations: vec![
            ArgumentKind::Entity,
            ArgumentKind::Entity,
            ArgumentKind::None,
            ArgumentKind::Reserved,
        ],
    }
}
fn sample() -> StructuredSample<String> {
    StructuredSample {
        primary_region: 0,
        regions: vec![],
        width: 1,
        height: 1,
        spatial: vec![0.3, 0.4],
        entities: vec![vec![0.1, 0.8]],
        scalars: vec![0.2, 0.7],
        events: vec![],
        candidates: vec![Candidate {
            key: "repair observed thing".into(),
            operation: 1,
            argument: Argument::Entity(0),
            features: vec![1., 0.],
            enabled: true,
        }],
    }
}

#[test]
fn same_shape_different_meanings_reject_before_parameter_restoration() {
    let input = schema();
    let device = Device::flex();
    let source = StructuredQ::new(
        StructuredModelConfig::small(input.clone()),
        &mut ChaCha8Rng::seed_from_u64(1),
        &device,
    )
    .unwrap();
    let record = source.checkpoint().unwrap();
    let mutations: Vec<fn(&mut InputSchema)> = vec![
        |s| s.semantics.operations.swap(0, 1),
        |s| s.semantics.vocabularies[0].categories.swap(0, 1),
        |s| s.semantics.spatial.swap(0, 1),
        |s| s.semantics.entities.swap(0, 1),
        |s| s.semantics.events.swap(0, 1),
        |s| s.semantics.scalars.swap(0, 1),
        |s| s.semantics.candidates.swap(0, 1),
        |s| s.semantics.events[0].transform = "divide_100".into(),
        |s| s.operations[1] = ArgumentKind::Location,
        |s| s.operations[3] = ArgumentKind::None,
        |s| s.semantics.observation_version += 1,
        |s| s.semantics.action_version += 1,
        |s| s.semantics.conventions[0].push_str(";different_order"),
    ];
    for mutate in mutations {
        let mut other = input.clone();
        mutate(&mut other);
        other.validate().unwrap();
        let mut model = StructuredQ::new(
            StructuredModelConfig::small(other),
            &mut ChaCha8Rng::seed_from_u64(2),
            &device,
        )
        .unwrap();
        let before = model.checkpoint().unwrap().metadata;
        assert!(model.restore(&record).is_err());
        assert_eq!(model.checkpoint().unwrap().metadata, before);
    }
    let batch = StructuredBatch::pack(&input, &[sample()], PackLimits::default(), &device).unwrap();
    let expected = to_vec(source.evaluate(&batch).unwrap()[0].values.clone()).unwrap();
    let mut restored = StructuredQ::new(
        StructuredModelConfig::small(input.clone()),
        &mut ChaCha8Rng::seed_from_u64(9),
        &device,
    )
    .unwrap();
    restored.restore(&record).unwrap();
    assert_eq!(
        to_vec(restored.evaluate(&batch).unwrap()[0].values.clone()).unwrap(),
        expected
    );
    let json = serde_json::to_string(&input).unwrap();
    assert_eq!(
        json,
        serde_json::to_string(&serde_json::from_str::<InputSchema>(&json).unwrap()).unwrap()
    );
}

#[test]
fn invalid_versions_and_ambiguous_descriptors_fail_without_execution() {
    let mutations: Vec<fn(&mut InputSchema)> = vec![
        |s| s.version = 1,
        |s| s.semantics.version = 99,
        |s| s.semantics.action_version = 0,
        |s| s.semantics.observation_version = 0,
        |s| s.semantics.spatial[0].name.clear(),
        |s| s.semantics.events[0].transform.clear(),
        |s| s.semantics.operations[1] = s.semantics.operations[0].clone(),
        |s| {
            s.semantics.entities.pop();
        },
    ];
    for mutate in mutations {
        let mut input = schema();
        mutate(&mut input);
        assert!(input.validate().is_err());
        assert!(
            StructuredQ::new(
                StructuredModelConfig::small(input.clone()),
                &mut ChaCha8Rng::seed_from_u64(1),
                &Device::flex()
            )
            .is_err()
        );
        assert!(
            StructuredBatch::pack(&input, &[sample()], PackLimits::default(), &Device::flex())
                .is_err()
        );
    }
    let config = StructuredModelConfig::small(schema());
    let mut model = StructuredQ::new(
        config.clone(),
        &mut ChaCha8Rng::seed_from_u64(1),
        &Device::flex(),
    )
    .unwrap();
    // No tensor payload: identity must reject before parameter loading starts.
    let legacy = LearnerRecord::new("structured-q-v1", &config).unwrap();
    assert!(
        model
            .restore(&legacy)
            .unwrap_err()
            .to_string()
            .contains("algorithm")
    );
    let mut invalid = model.checkpoint().unwrap();
    invalid.metadata["encoder"]["input"]["version"] = 99.into();
    assert!(model.restore(&invalid).is_err());
}

#[test]
fn reserved_operations_never_supply_executable_candidates() {
    let input = schema();
    input.validate().unwrap();
    for enabled in [true, false] {
        let mut s = sample();
        s.candidates[0].operation = 3;
        s.candidates[0].argument = Argument::None;
        s.candidates[0].enabled = enabled;
        assert!(
            StructuredBatch::pack(&input, &[s], PackLimits::default(), &Device::flex()).is_err()
        );
    }
    let mut no_target = sample();
    no_target.candidates[0].operation = 2;
    no_target.candidates[0].argument = Argument::None;
    assert!(
        StructuredBatch::pack(&input, &[no_target], PackLimits::default(), &Device::flex()).is_ok()
    );
}

#[test]
fn semantic_identity_is_checked_when_using_already_packed_tensors() {
    let input = schema();
    let mut other = input.clone();
    other.semantics.operations.swap(0, 1);
    let batch =
        StructuredBatch::pack(&other, &[sample()], PackLimits::default(), &Device::flex()).unwrap();
    let model = StructuredQ::new(
        StructuredModelConfig::small(input),
        &mut ChaCha8Rng::seed_from_u64(1),
        &Device::flex(),
    )
    .unwrap();
    assert!(model.evaluate(&batch).is_err());
}
