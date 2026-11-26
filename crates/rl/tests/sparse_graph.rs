#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        graph::{EdgeStatus, GraphEdge, GraphNode, GraphNodeId, GraphObservation, GraphSchema},
        structured::{
            EntityId, EntityRow, FeatureIdentity, SpatialObservation, SpatialRegion,
            StructuredObservation,
        },
    },
    nn::{
        device,
        structured::{PackLimits, SparseGraphBatch, SparseGraphEncoder},
        to_vec,
    },
};

fn entity(id: u64) -> EntityId {
    EntityId { kind: 6, id }
}

fn schema() -> GraphSchema {
    GraphSchema {
        namespace: "sparse-graph-test".into(),
        version: 1,
        node_features: vec![FeatureIdentity::new("signal", "identity")],
        edge_features: vec![FeatureIdentity::new("strength", "identity")],
        edge_types: vec!["control".into(), "crossing".into()],
    }
}

fn frame() -> GraphObservation {
    let location = GraphNodeId::Location {
        region: 9,
        x: 1,
        y: 0,
    };
    let a = GraphNodeId::Entity(entity(4));
    let b = GraphNodeId::Entity(entity(2));
    GraphObservation {
        namespace: "sparse-graph-test".into(),
        version: 1,
        tick: 5,
        observation: StructuredObservation {
            spatial: Some(SpatialObservation {
                width: 1,
                height: 1,
                cells: vec![0.],
            }),
            primary_region: 0,
            regions: vec![SpatialRegion {
                id: 9,
                map: SpatialObservation {
                    width: 2,
                    height: 1,
                    cells: vec![0.; 2],
                },
            }],
            entities: vec![
                EntityRow {
                    id: entity(4),
                    features: vec![1.],
                },
                EntityRow {
                    id: entity(2),
                    features: vec![2.],
                },
            ],
            events: vec![],
            scalars: vec![],
            candidates: vec![],
        },
        nodes: vec![
            GraphNode {
                id: a,
                features: vec![0.5],
            },
            GraphNode {
                id: b,
                features: vec![0.5],
            },
            GraphNode {
                id: location,
                features: vec![0.5],
            },
            GraphNode {
                id: GraphNodeId::Agent,
                features: vec![0.5],
            },
        ],
        edges: vec![
            GraphEdge {
                id: 18,
                from: a,
                to: b,
                kind: 0,
                tick: 2,
                status: EdgeStatus::Remembered,
                conditional: false,
                features: vec![0.75],
            },
            GraphEdge {
                id: 11,
                from: b,
                to: location,
                kind: 1,
                tick: 2,
                status: EdgeStatus::Superseded { tick: 4 },
                conditional: true,
                features: vec![0.5],
            },
        ],
    }
}

#[test]
fn local_endpoints_survive_reordered_nodes_and_entities_and_respect_sample_boundaries() {
    let a = frame();
    let mut moved = a.clone();
    moved.observation.entities.reverse();
    moved.nodes.reverse();
    moved.edges.reverse();
    let mut empty = a.clone();
    empty.edges.clear();
    empty.nodes.clear();
    let device = device();
    let schema = schema();
    let batch = SparseGraphBatch::pack(
        &schema,
        &[&a, &empty, &moved],
        PackLimits::default(),
        &device,
    )
    .unwrap();
    assert_eq!(batch.node_offsets(), &[0, 4, 4, 8]);
    assert_eq!(batch.edge_offsets(), &[0, 2, 2, 4]);
    assert_eq!(
        batch
            .edge_indices()
            .unwrap()
            .0
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap(),
        vec![0, 1, 6, 7]
    );
    assert_eq!(
        batch
            .edge_indices()
            .unwrap()
            .1
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap(),
        vec![1, 2, 5, 6]
    );

    let mut rng = ChaCha8Rng::seed_from_u64(16);
    let encoder = SparseGraphEncoder::new(schema, 8, 16384, &mut rng, &device).unwrap();
    let pooled = to_vec(encoder.forward(&batch).unwrap()).unwrap();
    for i in 0..8 {
        assert!((pooled[i] - pooled[16 + i]).abs() < 1e-6);
        assert_eq!(pooled[8 + i], 0.);
    }
}

#[test]
fn graph_messages_depend_on_wiring_and_receive_gradients() {
    let a = frame();
    let mut different = a.clone();
    // Same nodes and edge features. Redirect the first message into the same
    // destination as the second, changing local aggregation.
    different.edges[0].to = a.nodes[2].id;
    let device = device();
    let schema = schema();
    let batch =
        SparseGraphBatch::pack(&schema, &[&a, &different], PackLimits::default(), &device).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(53);
    let encoder = SparseGraphEncoder::new(schema, 8, 16384, &mut rng, &device).unwrap();
    let output = encoder.forward(&batch).unwrap();
    let pooled = to_vec(output.clone()).unwrap();
    assert!(
        pooled[..8]
            .iter()
            .zip(&pooled[8..])
            .any(|(a, b)| (a - b).abs() > 1e-5)
    );
    let grads = output.powi_scalar(2).mean().backward();
    let gradient = encoder.messages.layers[0]
        .weight
        .val()
        .grad(&grads)
        .unwrap();
    assert!(
        to_vec(gradient)
            .unwrap()
            .iter()
            .any(|value| value.is_finite() && value.abs() > 1e-8)
    );
}

#[test]
fn pack_rejects_unknown_geometry_duplicate_identity_and_unbounded_allocations() {
    let schema = schema();
    let device = device();
    let a = frame();
    let mut bad = a.clone();
    bad.nodes[2].id = GraphNodeId::Location {
        region: 9,
        x: 2,
        y: 0,
    };
    bad.edges[1].to = bad.nodes[2].id;
    let pack = |graph: &GraphObservation| {
        SparseGraphBatch::pack(&schema, &[graph], PackLimits::default(), &device)
    };
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.namespace = "different-semantics".into();
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.observation.spatial = None;
    bad.observation.primary_region = 9;
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.nodes[1].id = bad.nodes[0].id;
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.edges[0].to = GraphNodeId::Region(9);
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.edges[0].id = bad.edges[1].id;
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.edges[0].status = EdgeStatus::Superseded { tick: 1 };
    assert!(pack(&bad).is_err());
    bad = a.clone();
    bad.edges[0].features[0] = f32::NAN;
    assert!(pack(&bad).is_err());
    assert!(
        SparseGraphBatch::pack(&schema, &[&a], PackLimits { max_elements: 4 }, &device).is_err()
    );
}
