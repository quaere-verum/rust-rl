//! Canonical public-history bytes shared by actor adoption and factual joins.
//! Callers supply aligned validated observations and preceding factual inputs.
//! The original adoption domain tag and byte ordering are compatibility contracts.
use crate::core::{
    action::{Action, ActionArgument, ConditionalAction},
    env::Observation,
    graph::{EdgeStatus, GraphNodeId, GraphObservation},
    stateful::PreviousInput,
};
use sha2::{Digest, Sha256};

pub(crate) fn public_history_identity<'a>(
    episode_id: u64,
    base_timestep: usize,
    previous: Option<PreviousInput<'a>>,
    observations: &[Observation],
    inputs: impl ExactSizeIterator<Item = PreviousInput<'a>>,
) -> [u8; 32] {
    fn action(hash: &mut Sha256, action: &Action) {
        match action {
            Action::Discrete(value) => {
                hash.update([0]);
                hash.update((*value as u64).to_le_bytes());
            }
            Action::Continuous(values) => {
                hash.update([1]);
                hash.update((values.len() as u64).to_le_bytes());
                for value in values {
                    hash.update(value.to_bits().to_le_bytes());
                }
            }
            Action::Conditional(value) => {
                hash.update([2]);
                conditional(hash, value);
            }
        }
    }
    fn entity(hash: &mut Sha256, id: crate::core::structured::EntityId) {
        hash.update(id.kind.to_le_bytes());
        hash.update(id.id.to_le_bytes());
    }
    fn conditional(hash: &mut Sha256, action: &ConditionalAction) {
        hash.update((action.operation as u64).to_le_bytes());
        match action.argument {
            None => hash.update([0]),
            Some(ActionArgument::Entity(id)) => {
                hash.update([1]);
                entity(hash, id);
            }
            Some(ActionArgument::Location { x, y }) => {
                hash.update([2]);
                hash.update((x as u64).to_le_bytes());
                hash.update((y as u64).to_le_bytes());
            }
            Some(ActionArgument::RegionalLocation { region, x, y }) => {
                hash.update([3]);
                hash.update(region.to_le_bytes());
                hash.update((x as u64).to_le_bytes());
                hash.update((y as u64).to_le_bytes());
            }
        }
    }
    fn values(hash: &mut Sha256, values: &[f32]) {
        hash.update((values.len() as u64).to_le_bytes());
        for value in values {
            hash.update(value.to_bits().to_le_bytes());
        }
    }
    fn node(hash: &mut Sha256, id: GraphNodeId) {
        match id {
            GraphNodeId::Agent => hash.update([0]),
            GraphNodeId::Entity(id) => {
                hash.update([1]);
                entity(hash, id);
            }
            GraphNodeId::Region(id) => {
                hash.update([2]);
                hash.update(id.to_le_bytes());
            }
            GraphNodeId::Location { region, x, y } => {
                hash.update([3]);
                hash.update(region.to_le_bytes());
                hash.update((x as u64).to_le_bytes());
                hash.update((y as u64).to_le_bytes());
            }
        }
    }
    fn graph_fields(hash: &mut Sha256, graph: &GraphObservation) {
        hash.update((graph.namespace.len() as u64).to_le_bytes());
        hash.update(graph.namespace.as_bytes());
        hash.update(graph.version.to_le_bytes());
        hash.update(graph.tick.to_le_bytes());
        hash.update((graph.nodes.len() as u64).to_le_bytes());
        for row in &graph.nodes {
            node(hash, row.id);
            values(hash, &row.features);
        }
        hash.update((graph.edges.len() as u64).to_le_bytes());
        for row in &graph.edges {
            hash.update(row.id.to_le_bytes());
            node(hash, row.from);
            node(hash, row.to);
            hash.update((row.kind as u64).to_le_bytes());
            hash.update(row.tick.to_le_bytes());
            match row.status {
                EdgeStatus::Observed => hash.update([0]),
                EdgeStatus::Remembered => hash.update([1]),
                EdgeStatus::Disclosed => hash.update([2]),
                EdgeStatus::Superseded { tick } => {
                    hash.update([3]);
                    hash.update(tick.to_le_bytes());
                }
            }
            hash.update([u8::from(row.conditional)]);
            values(hash, &row.features);
        }
    }
    fn reward(hash: &mut Sha256, previous: PreviousInput<'_>) {
        hash.update(previous.reward.to_bits().to_le_bytes());
        if let Some(factual) = previous.factual
            && factual.reward.duration() > 1
        {
            hash.update(b"primitive-reward-trace\0");
            values(hash, factual.reward.primitives());
        }
    }
    fn mask(hash: &mut Sha256, mask: Option<&[bool]>) {
        if let Some(mask) = mask {
            hash.update([1]);
            hash.update((mask.len() as u64).to_le_bytes());
            for value in mask {
                hash.update([u8::from(*value)]);
            }
        } else {
            hash.update([0]);
        }
    }
    let mut hash = Sha256::new();
    hash.update(b"rust-rl-adoption-public-history-v1\0");
    hash.update(episode_id.to_le_bytes());
    hash.update((base_timestep as u64).to_le_bytes());
    if let Some(previous) = previous {
        hash.update([1]);
        action(&mut hash, previous.action);
        reward(&mut hash, previous);
    } else {
        hash.update([0]);
    }
    hash.update((observations.len() as u64).to_le_bytes());
    for observation in observations {
        match observation {
            Observation::Dense(dense) => {
                values(&mut hash, &dense.values);
                mask(&mut hash, dense.action_mask.as_deref());
                if let Some(data) = &dense.public_data {
                    hash.update(b"typed-public-data\0");
                    hash.update(data.identity());
                }
            }
            Observation::Discrete(discrete) => {
                hash.update((u64::MAX - 2).to_le_bytes());
                hash.update(discrete.symbol.to_le_bytes());
                mask(&mut hash, discrete.action_mask.as_deref());
            }
            Observation::Structured(_) | Observation::Graph(_) => {
                let frame = observation.as_structured().expect("structured payload");
                // Neither sentinel can be a real dense vector length.
                let graph = observation.as_graph();
                hash.update((u64::MAX - u64::from(graph.is_some())).to_le_bytes());
                if let Some(map) = &frame.spatial {
                    hash.update([1]);
                    hash.update((map.width as u64).to_le_bytes());
                    hash.update((map.height as u64).to_le_bytes());
                    values(&mut hash, &map.cells);
                } else {
                    hash.update([0]);
                }
                if frame.primary_region != 0 || !frame.regions.is_empty() {
                    hash.update(b"public-spatial-regions\0");
                    hash.update(frame.primary_region.to_le_bytes());
                    hash.update((frame.regions.len() as u64).to_le_bytes());
                    for region in &frame.regions {
                        hash.update(region.id.to_le_bytes());
                        hash.update((region.map.width as u64).to_le_bytes());
                        hash.update((region.map.height as u64).to_le_bytes());
                        values(&mut hash, &region.map.cells);
                    }
                }
                values(&mut hash, &frame.scalars);
                hash.update((frame.entities.len() as u64).to_le_bytes());
                for row in &frame.entities {
                    entity(&mut hash, row.id);
                    values(&mut hash, &row.features);
                }
                hash.update((frame.events.len() as u64).to_le_bytes());
                for event in &frame.events {
                    if let Some(id) = event.entity {
                        hash.update([1]);
                        entity(&mut hash, id);
                    } else {
                        hash.update([0]);
                    }
                    hash.update((event.context.len() as u64).to_le_bytes());
                    for &id in event.context.iter() {
                        entity(&mut hash, id);
                    }
                    values(&mut hash, &event.features);
                    if let Some(stamp) = event.stamp {
                        hash.update(b"public-event-stamp\0");
                        hash.update(stamp.source.to_le_bytes());
                        hash.update(stamp.ordinal.to_le_bytes());
                        hash.update(stamp.tick.to_le_bytes());
                    }
                }
                hash.update((frame.candidates.len() as u64).to_le_bytes());
                for candidate in &frame.candidates {
                    conditional(&mut hash, &candidate.action);
                    hash.update([u8::from(candidate.enabled)]);
                    values(&mut hash, &candidate.features);
                }
                if let Some(graph) = graph {
                    graph_fields(&mut hash, graph);
                }
            }
        }
    }
    hash.update((inputs.len() as u64).to_le_bytes());
    for input in inputs {
        action(&mut hash, input.action);
        reward(&mut hash, input);
    }
    hash.finalize().into()
}
