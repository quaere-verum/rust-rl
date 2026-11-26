#![cfg(feature = "neural")]

use burn::tensor::Device;
#[path = "support/structured_semantics.rs"]
mod semantics;
use rl::nn::{
    structured::{
        Argument, ArgumentKind, Candidate, Event, InputSchema, PackLimits, StructuredBatch,
        StructuredSample,
    },
    to_vec,
};

fn schema() -> InputSchema {
    InputSchema {
        version: 2,
        semantics: semantics::identity([2, 2, 1, 1, 1], &["entity", "location", "wait"]),
        spatial_channels: 2,
        entity_features: 2,
        event_features: 1,
        scalar_features: 1,
        candidate_features: 1,
        operations: vec![
            ArgumentKind::Entity,
            ArgumentKind::Location,
            ArgumentKind::None,
        ],
    }
}
fn sample() -> StructuredSample<String> {
    StructuredSample {
        primary_region: 0,
        regions: vec![],
        width: 3,
        height: 2,
        spatial: (1..=12).map(|v| v as f32).collect(),
        entities: vec![vec![2., 3.]],
        scalars: vec![4.],
        events: vec![Event {
            features: vec![5.],
            entity: Some(0),
            context: vec![0],
        }],
        candidates: vec![Candidate {
            key: "repair object".into(),
            operation: 0,
            argument: Argument::Entity(0),
            features: vec![1.],
            enabled: true,
        }],
    }
}

#[test]
fn typed_observation_projects_cells_and_stable_entity_references() {
    use rl::core::{
        action::{ActionArgument, ConditionalAction},
        structured::{
            ActionCandidate, EntityId, EntityRow, PublicEvent, SpatialObservation,
            StructuredObservation,
        },
    };
    let first = EntityId { kind: 1, id: 500 };
    let second = EntityId { kind: 2, id: 500 };
    let action = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(second)),
    };
    let mut observation = StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: Some(SpatialObservation {
            width: 2,
            height: 1,
            cells: vec![1., 2., 3., 4.],
        }),
        entities: vec![
            EntityRow {
                id: first,
                features: vec![5., 6.],
            },
            EntityRow {
                id: second,
                features: vec![7., 8.],
            },
        ],
        events: vec![PublicEvent {
            features: [9.].into(),
            entity: Some(second),
            context: [first].into(),
            stamp: None,
        }],
        scalars: vec![10.],
        candidates: vec![ActionCandidate {
            action,
            features: vec![11.],
            enabled: true,
        }],
    };
    let schema = schema();
    let sample = StructuredSample::from_observation(&observation, &schema).unwrap();
    assert_eq!(sample.spatial, [1., 3., 2., 4.]);
    assert_eq!(sample.events[0].entity, Some(1));
    assert_eq!(sample.events[0].context, [0]);
    assert_eq!(sample.candidates[0].key, action);
    assert_eq!(sample.candidates[0].argument, Argument::Entity(1));

    observation.entities.swap(0, 1);
    let sample = StructuredSample::from_observation(&observation, &schema).unwrap();
    assert_eq!(sample.events[0].entity, Some(0));
    assert_eq!(sample.events[0].context, [1]);
    assert_eq!(sample.candidates[0].argument, Argument::Entity(0));
    observation.entities.remove(0);
    assert!(StructuredSample::from_observation(&observation, &schema).is_err());
}

#[test]
fn nonspatial_samples_are_supported_by_the_lightweight_model() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        core::{
            action::ConditionalAction,
            structured::{ActionCandidate, StructuredObservation},
        },
        nn::structured::{EncoderConfig, LightweightConfig, LightweightQ, StructuredActionValue},
    };
    let mut schema = schema();
    schema.spatial_channels = 0;
    schema.semantics.spatial.clear();
    let action = ConditionalAction {
        operation: 2,
        argument: None,
    };
    let observation = StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![],
        events: vec![],
        scalars: vec![2.],
        candidates: vec![ActionCandidate {
            action,
            features: vec![3.],
            enabled: true,
        }],
    };
    let sample = StructuredSample::from_observation(&observation, &schema).unwrap();
    assert!(EncoderConfig::small(schema.clone()).validate().is_err());
    let model = LightweightQ::new(
        LightweightConfig::small(schema),
        &mut ChaCha8Rng::seed_from_u64(7),
        &Device::flex(),
    )
    .unwrap();
    let samples = [sample];
    let values = model
        .candidate_values(&samples, PackLimits::default())
        .unwrap();
    assert_eq!(values[0].keys, [&action]);
    let values = to_vec(values[0].values.clone()).unwrap();
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
}

#[test]
fn local_padding_masks_and_event_joins_keep_exact_rows() {
    let first = sample();
    let mut second = sample();
    second.width = 1;
    second.height = 3;
    second.spatial = vec![20.; 6];
    second.entities.clear();
    second.events.clear();
    second.candidates.clear();
    let batch = StructuredBatch::pack(
        &schema(),
        &[first, second],
        PackLimits::default(),
        &Device::flex(),
    )
    .unwrap();
    assert_eq!(batch.spatial().dims(), [2, 2, 3, 3]);
    assert_eq!(
        to_vec(batch.cell_mask()).unwrap(),
        vec![
            1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.
        ]
    );
    assert_eq!(to_vec(batch.entity_mask()).unwrap(), vec![1., 0.]);
    assert_eq!(to_vec(batch.event_mask()).unwrap(), vec![1., 0.]);
    let grid = to_vec(batch.spatial()).unwrap();
    assert_eq!(&grid[..9], &[1., 2., 3., 4., 5., 6., 0., 0., 0.]);
    assert_eq!(&grid[9..18], &[7., 8., 9., 10., 11., 12., 0., 0., 0.]);
    assert_eq!(batch.candidates(0)[0].key, "repair object");
    assert_eq!(to_vec(batch.event_entities()).unwrap(), vec![1., 0.]);
    assert_eq!(to_vec(batch.event_context()).unwrap(), vec![1., 0.]);
}

#[test]
fn malformed_and_excessive_inputs_fail_before_tensor_allocation() {
    let check = |s| StructuredBatch::pack(&schema(), &[s], PackLimits::default(), &Device::flex());
    let mut s = sample();
    s.width = usize::MAX;
    assert!(check(s).is_err());
    let mut s = sample();
    s.spatial[0] = f32::NAN;
    assert!(check(s).is_err());
    let mut s = sample();
    s.events[0].entity = Some(1);
    assert!(check(s).is_err());
    let mut s = sample();
    s.events[0].context.push(0);
    assert!(check(s).is_err());
    let mut s = sample();
    s.candidates[0].argument = Argument::None;
    assert!(check(s).is_err());
    let mut s = sample();
    s.candidates.push(s.candidates[0].clone());
    assert!(check(s).is_err());
    assert!(
        StructuredBatch::pack(
            &schema(),
            &[sample()],
            PackLimits { max_elements: 1 },
            &Device::flex()
        )
        .is_err()
    );
    assert!(
        StructuredBatch::<String>::pack(&schema(), &[], PackLimits::default(), &Device::flex())
            .is_err()
    );
}

#[test]
fn spatial_candidates_are_checked_against_each_real_map() {
    let mut s = sample();
    s.candidates[0].operation = 1;
    s.candidates[0].argument = Argument::Location { x: 2, y: 1 };
    assert!(
        StructuredBatch::pack(
            &schema(),
            std::slice::from_ref(&s),
            PackLimits::default(),
            &Device::flex()
        )
        .is_ok()
    );
    s.candidates[0].argument = Argument::Location { x: 3, y: 1 };
    assert!(
        StructuredBatch::pack(&schema(), &[s], PackLimits::default(), &Device::flex()).is_err()
    );
}

#[test]
fn regional_targets_keep_local_coordinates_and_mask_cnn_boundaries() {
    use rl::core::{
        action::{ActionArgument, ConditionalAction},
        env::{Observation, ObservationSpace},
        structured::{ActionCandidate, StructuredObservation},
    };
    use std::sync::Arc;

    let schema = schema();
    let mut observation: StructuredObservation = serde_json::from_value(serde_json::json!({
        "spatial": {"width": 2, "height": 1, "cells": [1., 2., 3., 4.] },
        "primary_region": 7,
        "regions": [
            {"id": 8, "map": {"width": 1, "height": 2, "cells": [5., 6., 7., 8.] }},
            {"id": 4, "map": {"width": 1, "height": 1, "cells": [9., 10.] }}
        ],
        "entities": [], "events": [], "scalars": [0.], "candidates": []
    }))
    .unwrap();
    let key = |region, x, y| -> ConditionalAction {
        serde_json::from_value(serde_json::json!({
            "operation": 1, "argument": {"RegionalLocation": {"region": region, "x": x, "y": y}}
        }))
        .unwrap()
    };
    observation.candidates = [(7, 0, 0), (8, 0, 0), (8, 0, 1)]
        .map(|(region, x, y)| ActionCandidate {
            action: key(region, x, y),
            features: vec![0.],
            enabled: true,
        })
        .into();
    let plain = ActionArgument::Location { x: 0, y: 0 };
    assert_ne!(Some(plain), observation.candidates[0].action.argument);

    let space = ObservationSpace::Structured(Arc::new(schema.observation_schema()));
    let grammar = rl::core::action::ActionSpace::Conditional(Arc::new(schema.action_grammar()));
    Observation::Structured(Arc::new(observation.clone()))
        .validate_in(&space, &grammar, true)
        .unwrap();
    let pack = |observation: &StructuredObservation| {
        let sample = StructuredSample::from_observation(observation, &schema).unwrap();
        StructuredBatch::pack(&schema, &[sample], PackLimits::default(), &Device::flex()).unwrap()
    };
    let batch = pack(&observation);
    assert_eq!(batch.spatial().dims(), [1, 2, 6, 2]);
    assert_eq!(
        to_vec(batch.cell_mask()).unwrap(),
        [1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.]
    );
    assert_eq!(
        &to_vec(batch.spatial()).unwrap()[..12],
        &[1., 3., 0., 0., 9., 0., 0., 0., 5., 0., 7., 0.]
    );
    assert_eq!(batch.candidates(0)[2].key, key(8, 0, 1));
    assert_eq!(
        batch.candidates(0)[2].argument,
        Argument::Location { x: 0, y: 5 }
    );
    observation.regions.reverse();
    assert_eq!(
        to_vec(pack(&observation).spatial()).unwrap(),
        to_vec(batch.spatial()).unwrap()
    );

    observation.candidates[2].action = key(8, 1, 0);
    assert!(StructuredSample::from_observation(&observation, &schema).is_err());
    assert!(
        Observation::Structured(Arc::new(observation))
            .validate_in(&space, &grammar, true)
            .is_err()
    );
}
