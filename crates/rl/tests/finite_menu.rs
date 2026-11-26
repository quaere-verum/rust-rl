use rl::core::{
    action::{Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation},
    env::{GraphObservationSpace, Observation, ObservationSpace},
    graph::{GRAPH_SCHEMA_VERSION, GraphNode, GraphNodeId, GraphObservation, GraphSchema},
    structured::{
        ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity, ObservationSchema,
        StructuredObservation,
    },
};
use std::sync::Arc;

#[test]
fn graph_observations_enumerate_enabled_actions_in_semantic_order() -> rl::Result<()> {
    let entity = |id| EntityId { kind: 1, id };
    let choose = |id| ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity(id))),
    };
    let action_space = ActionSpace::Conditional(Arc::new(ActionGrammar {
        namespace: "finite-menu-test".into(),
        version: 1,
        operations: vec![Operation {
            name: "choose".into(),
            argument: ArgumentKind::Entity,
        }],
        candidate_features: vec![],
    }));
    let observation_space = ObservationSpace::Graph(Arc::new(GraphObservationSpace {
        base: ObservationSchema {
            namespace: "finite-menu-test".into(),
            version: 1,
            spatial: vec![],
            entities: vec![FeatureIdentity::new("availability", "identity")],
            events: vec![],
            scalars: vec![],
            vocabularies: vec![],
            conventions: vec![],
        },
        graph: GraphSchema {
            namespace: "finite-menu-test".into(),
            version: GRAPH_SCHEMA_VERSION,
            node_features: vec![],
            edge_features: vec![],
            edge_types: vec!["link".into()],
        },
    }));
    let entities = [3, 8, 2];
    let observation = Observation::Graph(Arc::new(GraphObservation {
        namespace: "finite-menu-test".into(),
        version: GRAPH_SCHEMA_VERSION,
        tick: 0,
        observation: StructuredObservation {
            spatial: None,
            primary_region: 0,
            regions: vec![],
            entities: entities
                .map(|id| EntityRow {
                    id: entity(id),
                    features: vec![1.0],
                })
                .into(),
            events: vec![],
            scalars: vec![],
            candidates: [(8, true), (2, false), (3, true)]
                .map(|(id, enabled)| ActionCandidate {
                    action: choose(id),
                    features: vec![],
                    enabled,
                })
                .into(),
        },
        nodes: entities
            .map(|id| GraphNode {
                id: GraphNodeId::Entity(entity(id)),
                features: vec![],
            })
            .into(),
        edges: vec![],
    }));
    observation.validate_in(&observation_space, &action_space, true)?;

    assert_eq!(
        observation.finite_actions(&action_space)?,
        [
            Action::Conditional(choose(3).into()),
            Action::Conditional(choose(8).into())
        ],
    );
    Ok(())
}

#[test]
fn location_aliases_share_a_table_key_and_execute_in_their_published_form() -> rl::Result<()> {
    use rl::core::structured::SpatialObservation;

    let space = ActionSpace::Conditional(Arc::new(ActionGrammar {
        namespace: "location-test".into(),
        version: 1,
        operations: vec![Operation {
            name: "visit".into(),
            argument: ArgumentKind::Location,
        }],
        candidate_features: vec![],
    }));
    let schema = ObservationSpace::Structured(Arc::new(ObservationSchema {
        namespace: "location-test".into(),
        version: 1,
        spatial: vec![FeatureIdentity::new("surface", "identity")],
        entities: vec![],
        events: vec![],
        scalars: vec![],
        vocabularies: vec![],
        conventions: vec![],
    }));
    let frame = |argument| {
        Observation::Structured(Arc::new(StructuredObservation {
            spatial: Some(SpatialObservation {
                width: 1,
                height: 1,
                cells: vec![1.0],
            }),
            primary_region: 7,
            regions: vec![],
            entities: vec![],
            events: vec![],
            scalars: vec![],
            candidates: vec![ActionCandidate {
                action: ConditionalAction {
                    operation: 0,
                    argument: Some(argument),
                },
                features: vec![],
                enabled: true,
            }],
        }))
    };
    let local = frame(ActionArgument::Location { x: 0, y: 0 });
    let regional = frame(ActionArgument::RegionalLocation {
        region: 7,
        x: 0,
        y: 0,
    });
    local.validate_in(&schema, &space, true)?;
    regional.validate_in(&schema, &space, true)?;
    let local_action = &local.finite_actions(&space)?[0];
    let regional_action = &regional.finite_actions(&space)?[0];
    assert_eq!(
        local.finite_action_key(local_action)?,
        regional.finite_action_key(regional_action)?
    );
    local.check_action(&space, local_action)?;
    regional.check_action(&space, regional_action)?;
    assert!(local.check_action(&space, regional_action).is_err());
    Ok(())
}

#[test]
fn discrete_symbols_keep_their_identity_and_respect_the_public_mask() -> rl::Result<()> {
    use rl::core::env::DiscreteObservationSpace;

    let symbol = (1_u64 << 54) + 1;
    let space = ObservationSpace::Discrete(DiscreteObservationSpace {
        symbols: symbol + 1,
    });
    let actions = ActionSpace::Discrete { count: 3 };
    let observation = Observation::discrete(symbol, Some(vec![false, true, false]));
    space.validate()?;
    observation.validate_in(&space, &actions, true)?;
    assert_eq!(
        observation.finite_actions(&actions)?,
        vec![Action::Discrete(1)]
    );
    assert!(
        observation
            .check_action(&actions, &Action::Discrete(0))
            .is_err()
    );
    assert!(
        Observation::discrete(symbol + 1, None)
            .validate_in(&space, &actions, true)
            .is_err()
    );
    let restored: Observation = serde_json::from_str(&serde_json::to_string(&observation)?)?;
    assert_eq!(restored.as_discrete().unwrap().symbol, symbol);
    assert_eq!(
        restored.action_mask(),
        Some([false, true, false].as_slice())
    );
    assert_eq!(
        serde_json::from_str::<ObservationSpace>(&serde_json::to_string(&space)?)?,
        space
    );
    Ok(())
}
