//! Deterministic sparse-graph packing and conditional Q inference on Flex CPU.
//! Register as a Cargo benchmark with `harness = false`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    core::{
        action::{ActionArgument, ConditionalAction},
        graph::{EdgeStatus, GraphEdge, GraphNode, GraphNodeId, GraphObservation, GraphSchema},
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity, PublicEvent,
            SpatialObservation, StructuredObservation,
        },
    },
    nn::{
        structured::{
            GraphStructuredBatch, GraphStructuredQ, GraphStructuredQConfig, InputSchema,
            PackLimits, SemanticInput, SparseGraphBatch, StructuredModelConfig,
        },
        to_vec,
    },
};
use std::{hint::black_box, time::Instant};

const SEED: u64 = 481;
const WARMUP: usize = 5;
const TRIALS: usize = 5;
const ITERATIONS: usize = 24;
const NAMESPACE: &str = "rust-rl.synthetic-graph-benchmark";

fn fields(names: &[&str]) -> Vec<FeatureIdentity> {
    names
        .iter()
        .map(|&name| FeatureIdentity::new(name, "identity_f32"))
        .collect()
}

fn schemas() -> (InputSchema, GraphSchema) {
    let input = InputSchema {
        version: 2,
        semantics: SemanticInput {
            version: 1,
            namespace: NAMESPACE.into(),
            observation_version: 1,
            action_version: 1,
            spatial: fields(&["surface"]),
            entities: fields(&["charge", "load"]),
            events: fields(&["reading"]),
            scalars: fields(&["remaining"]),
            candidates: fields(&["effort"]),
            operations: vec!["wait".into(), "inspect".into()],
            vocabularies: vec![],
            conventions: vec!["synthetic fixtures with normalized scalar features".into()],
        },
        spatial_channels: 1,
        entity_features: 2,
        event_features: 1,
        scalar_features: 1,
        candidate_features: 1,
        operations: vec![ArgumentKind::None, ArgumentKind::Entity],
    };
    let graph = GraphSchema {
        namespace: NAMESPACE.into(),
        version: 1,
        node_features: fields(&["level", "capacity"]),
        edge_features: fields(&["strength"]),
        edge_types: vec!["route".into(), "link".into()],
    };
    (input, graph)
}

fn frame(entity_count: usize, rng: &mut ChaCha8Rng) -> GraphObservation {
    let entities: Vec<_> = (0..entity_count)
        .map(|index| EntityRow {
            id: EntityId {
                kind: 0,
                id: index as u64,
            },
            features: vec![rng.random(), rng.random()],
        })
        .collect();
    let mut candidates = vec![ActionCandidate {
        action: ConditionalAction {
            operation: 0,
            argument: None,
        },
        features: vec![0.],
        enabled: true,
    }];
    candidates.extend(entities.iter().take(4).map(|entity| ActionCandidate {
        action: ConditionalAction {
            operation: 1,
            argument: Some(ActionArgument::Entity(entity.id)),
        },
        features: vec![rng.random()],
        enabled: true,
    }));
    let events = entities
        .iter()
        .take(2)
        .map(|entity| PublicEvent {
            features: vec![rng.random()].into(),
            entity: Some(entity.id),
            context: vec![entity.id].into(),
            stamp: None,
        })
        .collect();
    let nodes: Vec<_> = std::iter::once(GraphNodeId::Agent)
        .chain(entities.iter().map(|entity| GraphNodeId::Entity(entity.id)))
        .map(|id| GraphNode {
            id,
            features: vec![rng.random(), rng.random()],
        })
        .collect();
    let mut edges = Vec::with_capacity(nodes.len() * 2);
    for (index, source) in nodes.iter().enumerate() {
        for (kind, offset) in [1, 3].into_iter().enumerate() {
            edges.push(GraphEdge {
                id: edges.len() as u64,
                from: source.id,
                to: nodes[(index + offset) % nodes.len()].id,
                kind,
                tick: 3,
                status: [EdgeStatus::Observed, EdgeStatus::Remembered][kind],
                conditional: kind == 1,
                features: vec![rng.random()],
            });
        }
    }
    GraphObservation {
        namespace: NAMESPACE.into(),
        version: 1,
        tick: 5,
        observation: StructuredObservation {
            spatial: Some(SpatialObservation {
                width: 4,
                height: 4,
                cells: (0..16).map(|_| rng.random()).collect(),
            }),
            primary_region: 0,
            regions: vec![],
            entities,
            events,
            scalars: vec![rng.random()],
            candidates,
        },
        nodes,
        edges,
    }
}

fn measure<T>(size: &str, phase: &str, mut operation: impl FnMut() -> Result<T>) -> Result<()> {
    println!("{size} {phase} warmup={WARMUP}");
    for _ in 0..WARMUP {
        black_box(operation()?);
    }
    let mut timings = [0.; TRIALS];
    for (trial, timing) in timings.iter_mut().enumerate() {
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            black_box(operation()?);
        }
        *timing = start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64;
        println!(
            "{size} {phase} trial={}/{TRIALS} us/iteration={timing:.3}",
            trial + 1
        );
    }
    timings.sort_unstable_by(f64::total_cmp);
    println!("{size} {phase} median_us={:.3}", timings[TRIALS / 2]);
    Ok(())
}

fn main() -> Result<()> {
    let device = burn::tensor::Device::flex();
    let (input, graph) = schemas();
    let limits = PackLimits::default();
    let mut model_rng = ChaCha8Rng::seed_from_u64(SEED);
    let model = GraphStructuredQ::new(
        GraphStructuredQConfig {
            version: 1,
            base: StructuredModelConfig::small(input.clone()),
            graph: graph.clone(),
            graph_width: 16,
        },
        &mut model_rng,
        &device,
    )?;
    println!("seed={SEED} iterations_per_trial={ITERATIONS}");

    let mut input_rng = ChaCha8Rng::seed_from_u64(SEED + 1);
    for (size, counts) in [
        ("small", [4, 8, 12, 16]),
        ("medium", [16, 32, 48, 64]),
        ("large", [64, 128, 192, 256]),
    ] {
        let frames = counts.map(|count| frame(count, &mut input_rng));
        let frames = frames.each_ref();
        let nodes: usize = frames.iter().map(|frame| frame.nodes.len()).sum();
        let edges: usize = frames.iter().map(|frame| frame.edges.len()).sum();
        println!("{size} graphs={} nodes={nodes} edges={edges}", frames.len());

        measure(size, "sparse_pack", || {
            SparseGraphBatch::pack(&graph, black_box(&frames), limits, &device)
        })?;
        let pack =
            || GraphStructuredBatch::pack(&input, &graph, black_box(&frames), limits, &device);
        measure(size, "structured_pack", pack)?;

        let packed = pack()?;
        let forward = || {
            // Materialize every row on the host so queued tensor work is timed.
            model
                .evaluate(black_box(&packed))?
                .into_iter()
                .map(|row| to_vec(row.values))
                .collect::<Result<Vec<_>>>()
        };
        let values = forward()?;
        assert_eq!(values.len(), frames.len());
        assert!(
            values.iter().zip(frames).all(|(row, frame)| {
                row.len() == frame.observation.candidates.len()
                    && row.iter().all(|value| value.is_finite())
            }),
            "inference must produce one finite value per complete legal action"
        );
        let checksum = values.iter().flatten().copied().map(f64::from).sum::<f64>();
        println!("{size} candidate_value_sum={checksum:.9}");
        measure(size, "structured_q_forward", forward)?;
    }
    Ok(())
}
