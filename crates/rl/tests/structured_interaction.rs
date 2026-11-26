use rl::core::{
    action::{Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation},
    env::{Observation, ObservationSpace, PublicHistory},
    reward::Reward,
    structured::{
        ActionCandidate, ArgumentKind, EntityId, EntityRow, EventStamp, FeatureIdentity,
        ObservationSchema, PublicEvent, StructuredObservation,
    },
};
use std::sync::Arc;

#[test]
fn event_suffix_preserves_same_tick_order_and_rejects_rewritten_prefixes() {
    let event = |source, ordinal, value| PublicEvent {
        features: [value].into(),
        entity: None,
        context: [].into(),
        stamp: Some(EventStamp {
            source,
            ordinal,
            tick: 4,
        }),
    };
    let first = StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![],
        events: vec![event(2, 3, 7.), event(1, 5, 8.)],
        scalars: vec![],
        candidates: vec![],
    };
    let mut next = first.clone();
    next.events.push(event(2, 4, 9.));
    assert_eq!(
        first.events[0].features.as_ptr(),
        next.events[0].features.as_ptr()
    );
    assert_eq!(
        next.introduced_events(Some(&first)).unwrap(),
        &next.events[2..]
    );
    assert!(first.introduced_events(Some(&next)).is_err());
    let mut reordered = next.clone();
    reordered.events.swap(0, 1);
    assert!(reordered.introduced_events(Some(&first)).is_err());
    let mut altered = next.clone();
    Arc::make_mut(&mut altered.events[0].features)[0] = 10.;
    assert!(altered.introduced_events(Some(&first)).is_err());
    let tape = PublicHistory::new(Observation::Structured(Arc::new(first.clone())))
        .append(
            Action::Discrete(0),
            0.,
            Observation::Structured(Arc::new(next.clone())),
        )
        .unwrap();
    let mut encoded = serde_json::to_value(&tape).unwrap();
    encoded[1]["observation"]["events"][0]["features"][0] = 10.0.into();
    assert!(serde_json::from_value::<PublicHistory>(encoded).is_err());
    next.events.push(event(1, 5, 11.));
    assert!(next.introduced_events(Some(&first)).is_err());
}

#[test]
fn restored_history_borrows_the_actual_selection_frame_and_primitives() {
    let initial = Observation::new(vec![2.]);
    let next = Observation::new(vec![3.]);
    let rewards = Reward::from_primitives(vec![1., -1., 0.5]).unwrap();
    let history = PublicHistory::new(initial.clone())
        .append(Action::Discrete(1), rewards.clone(), next.clone())
        .unwrap();
    let restored: PublicHistory =
        serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
    let inputs = restored.inputs();
    assert!(inputs[0].previous.is_none());
    assert_eq!(inputs[1].observation, &next);
    let previous = inputs[1].previous.unwrap();
    let factual = previous.factual.unwrap();
    assert_eq!(previous.action, &Action::Discrete(1));
    assert_eq!(previous.reward, 0.5);
    assert_eq!(factual.observation, &initial);
    assert_eq!(factual.reward, &rewards);
    assert_eq!(restored.timestep(), 1);
    assert_eq!(restored.physical_timestep(), 3);
}

#[test]
fn entity_actions_follow_semantic_keys_and_dynamic_legality() {
    let entity = EntityId { kind: 1, id: 17 };
    let action = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity)),
    };
    let actions = ActionSpace::Conditional(Arc::new(ActionGrammar {
        namespace: "test".into(),
        version: 1,
        operations: vec![Operation {
            name: "inspect".into(),
            argument: ArgumentKind::Entity,
        }],
        candidate_features: vec![],
    }));
    let observations = ObservationSpace::Structured(Arc::new(ObservationSchema {
        namespace: "test".into(),
        version: 1,
        spatial: vec![],
        entities: vec![FeatureIdentity::new("energy", "joules")],
        events: vec![],
        scalars: vec![],
        vocabularies: vec![],
        conventions: vec![],
    }));
    let mut frame = StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![
            EntityRow {
                id: entity,
                features: vec![2.],
            },
            EntityRow {
                id: EntityId { kind: 1, id: 12 },
                features: vec![3.],
            },
        ],
        events: vec![],
        scalars: vec![],
        candidates: vec![ActionCandidate {
            action,
            features: vec![],
            enabled: true,
        }],
    };
    let selected = Action::Conditional(Arc::new(action));
    for _ in 0..2 {
        let observation = Observation::Structured(Arc::new(frame.clone()));
        observation
            .validate_in(&observations, &actions, true)
            .unwrap();
        observation.check_action(&actions, &selected).unwrap();
        frame.entities.reverse();
    }
    frame.candidates[0].enabled = false;
    let final_observation = Observation::Structured(Arc::new(frame));
    final_observation
        .validate_in(&observations, &actions, false)
        .unwrap();
    assert!(
        final_observation
            .validate_in(&observations, &actions, true)
            .is_err()
    );
    assert!(final_observation.check_action(&actions, &selected).is_err());
}

#[test]
fn primitive_rewards_retain_order_and_one_step_wire_format() {
    let scalar: Reward = serde_json::from_str("-0.0").unwrap();
    assert_eq!(scalar.total().to_bits(), (-0.0f32).to_bits());
    assert_eq!(scalar.primitives(), &[-0.0]);
    assert_eq!(scalar.duration(), 1);
    assert_eq!(serde_json::to_string(&scalar).unwrap(), "-0.0");
    let rewards = Reward::from_primitives(vec![1., 2., -3.]).unwrap();
    assert_eq!(rewards.duration(), 3);
    assert_eq!(rewards.total(), 0.);
    assert_eq!(rewards.discounted(0.5).unwrap(), (1.25, 0.125));
    assert!(Reward::from_primitives(vec![]).is_err());
    assert!(Reward::from_primitives(vec![f32::NAN]).is_err());
}

#[test]
fn overflowing_discounted_trace_does_not_enqueue_a_transition() {
    use rl::{core::transition::Transition, data::replay::NStepAccumulator};
    let mut pending = NStepAccumulator::new(1, 0.5, 1).unwrap();
    let trace = Reward::from_primitives(vec![f32::MAX, f32::MAX, -f32::MAX, -f32::MAX]).unwrap();
    let mut transition = Transition {
        observation: Observation::new(vec![0.]),
        action: Action::Discrete(0),
        reward: trace,
        next_observation: Observation::new(vec![1.]),
        terminated: false,
        truncated: false,
    };
    assert!(pending.push(0, transition.clone()).is_err());
    transition.reward = 3.0.into();
    let items = pending.push(0, transition).unwrap();
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].transition.reward, 3.0.into());
    assert_eq!(items[0].discount, 0.5);
}

#[test]
fn bounded_replay_charges_canonical_shared_frame_capacity_after_restore() {
    use rl::{
        core::{env::EpisodeStart, temporal::TemporalSink},
        data::temporal::{EpisodeStore, ResourceLimits, TemporalStorage},
    };
    let mut scalars = Vec::with_capacity(16_384);
    scalars.push(3.);
    let frame = Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![],
        events: vec![],
        scalars,
        candidates: vec![],
    });
    let observations = ObservationSpace::Structured(Arc::new(ObservationSchema {
        namespace: "bounded-test".into(),
        version: 1,
        spatial: vec![],
        entities: vec![],
        events: vec![],
        scalars: vec![FeatureIdentity::new("quantity", "identity")],
        vocabularies: vec![],
        conventions: vec![],
    }));
    let mut store = EpisodeStore::with_storage(
        2,
        observations,
        ActionSpace::Discrete { count: 1 },
        TemporalStorage::FullPrefix(ResourceLimits {
            max_transitions: 2,
            max_resident_bytes: 4096,
            max_anchor_bytes: 0,
            max_anchors: 0,
            max_anchor_payload_bytes: 1,
            max_dimension_rank: 1,
        }),
    )
    .unwrap();
    store
        .begin_episode(
            0,
            &EpisodeStart::fresh(Observation::Structured(frame.clone())),
        )
        .unwrap();
    let Observation::Structured(retained) = &store.episode(0).unwrap().observations[0] else {
        panic!("expected a retained structured frame");
    };
    assert_eq!(retained.scalars.capacity(), 1);
    assert_eq!(frame.scalars.capacity(), 16_384);
    let encoded = serde_json::to_string(&store).unwrap();
    let restored: EpisodeStore = serde_json::from_str(&encoded).unwrap();
    assert_eq!(store.resource_usage(), restored.resource_usage());
}
