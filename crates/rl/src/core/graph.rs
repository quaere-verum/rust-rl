//! Versioned sparse public graphs layered over structured observations.
use super::{
    Result, ensure,
    structured::{
        EntityId, FeatureIdentity, StructuredObservation, equal_features, feature_fields, named,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const GRAPH_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSchema {
    pub namespace: String,
    pub version: u32,
    pub node_features: Vec<FeatureIdentity>,
    pub edge_features: Vec<FeatureIdentity>,
    /// The index in this vocabulary is a categorical edge type.
    pub edge_types: Vec<String>,
}

impl GraphSchema {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == GRAPH_SCHEMA_VERSION
                && !self.namespace.trim().is_empty()
                && self.namespace.len() <= 4096
                && self.node_features.len() <= 1024
                && self.edge_features.len() <= 1024
                && !self.edge_types.is_empty()
                && self.edge_types.len() <= 1024
                && feature_fields(&self.node_features)
                && feature_fields(&self.edge_features)
                && named(self.edge_types.iter().map(String::as_str)),
            "invalid sparse graph schema",
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphNodeId {
    Agent,
    Entity(EntityId),
    Region(u32),
    Location { region: u32, x: usize, y: usize },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphNode {
    pub id: GraphNodeId,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeStatus {
    Observed,
    Remembered,
    Disclosed,
    Superseded { tick: u64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphEdge {
    pub id: u64,
    pub from: GraphNodeId,
    pub to: GraphNodeId,
    pub kind: usize,
    pub tick: u64,
    pub status: EdgeStatus,
    pub conditional: bool,
    pub features: Vec<f32>,
}

/// The base observation keeps its existing identities and feature semantics.
/// Graph nodes refer to its public entities and geometry. Edges join only
/// explicitly listed nodes. Absence of an edge makes no claim about physics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphObservation {
    pub namespace: String,
    pub version: u32,
    pub tick: u64,
    pub observation: StructuredObservation,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

impl GraphObservation {
    /// Validate the graph and the public references it borrows from the base.
    /// The element ceiling includes numeric attributes and index columns.
    pub fn validate_graph(&self, schema: &GraphSchema, max_elements: usize) -> Result<()> {
        schema.validate()?;
        ensure(
            self.packed_elements(schema)
                .is_some_and(|size| size <= max_elements),
            "sparse graph allocation budget exceeded",
        )?;
        self.resolve_edges(schema, |_, _| {})
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        self.namespace == other.namespace
            && self.version == other.version
            && self.tick == other.tick
            && self.observation.matches(&other.observation)
            && self.nodes.len() == other.nodes.len()
            && self
                .nodes
                .iter()
                .zip(&other.nodes)
                .all(|(a, b)| a.id == b.id && equal_features(&a.features, &b.features))
            && self.edges.len() == other.edges.len()
            && self.edges.iter().zip(&other.edges).all(|(a, b)| {
                a.id == b.id
                    && a.from == b.from
                    && a.to == b.to
                    && a.kind == b.kind
                    && a.tick == b.tick
                    && a.status == b.status
                    && a.conditional == b.conditional
                    && equal_features(&a.features, &b.features)
            })
    }

    pub(crate) fn payload_bytes(&self, allocated: bool) -> usize {
        fn bytes<T>(rows: &Vec<T>, allocated: bool) -> usize {
            let count = if allocated {
                rows.capacity()
            } else {
                rows.len()
            };
            count * size_of::<T>()
        }
        self.observation.payload_bytes(allocated) + size_of::<Self>()
            - size_of::<StructuredObservation>()
            + if allocated {
                self.namespace.capacity()
            } else {
                self.namespace.len()
            }
            + bytes(&self.nodes, allocated)
            + bytes(&self.edges, allocated)
            + self
                .nodes
                .iter()
                .map(|row| bytes(&row.features, allocated))
                .sum::<usize>()
            + self
                .edges
                .iter()
                .map(|row| bytes(&row.features, allocated))
                .sum::<usize>()
    }

    pub(crate) fn canonicalize(&mut self) {
        fn fit<T>(values: &mut Vec<T>) {
            if values.capacity() != values.len() {
                *values = std::mem::take(values).into_boxed_slice().into_vec();
            }
        }
        self.observation.canonicalize();
        if self.namespace.capacity() != self.namespace.len() {
            self.namespace = std::mem::take(&mut self.namespace)
                .into_boxed_str()
                .into_string();
        }
        fit(&mut self.nodes);
        fit(&mut self.edges);
        for node in &mut self.nodes {
            fit(&mut node.features);
        }
        for edge in &mut self.edges {
            fit(&mut edge.features);
        }
    }

    pub(crate) fn packed_elements(&self, schema: &GraphSchema) -> Option<usize> {
        let nodes = self
            .nodes
            .len()
            .checked_mul(schema.node_features.len() + 5)?;
        let edges = self
            .edges
            .len()
            .checked_mul(schema.edge_features.len() + schema.edge_types.len() + 9)?;
        nodes
            .checked_add(edges)?
            .checked_add(3)?
            .checked_add(self.observation.entities.len())?
            .checked_add(self.observation.regions.len())
    }

    // Visit validated edges in their original order. Identity lookup and region
    // geometry are built once, including for edges across distant regions.
    pub(crate) fn resolve_edges(
        &self,
        schema: &GraphSchema,
        mut visit: impl FnMut([usize; 2], &GraphEdge),
    ) -> Result<()> {
        ensure(
            self.version == schema.version && self.namespace == schema.namespace,
            "sparse graph schema identity mismatch",
        )?;
        let public = &self.observation;
        ensure(
            public
                .events
                .iter()
                .all(|row| row.stamp.is_none_or(|stamp| stamp.tick <= self.tick)),
            "graph observation has future evidence",
        )?;
        let entities: HashSet<_> = public.entities.iter().map(|entity| entity.id).collect();
        ensure(
            entities.len() == public.entities.len(),
            "duplicate public entity identity",
        )?;
        let mut regions = HashMap::with_capacity(public.regions.len() + 1);
        if let Some(map) = &public.spatial {
            ensure(map.width > 0 && map.height > 0, "empty primary region")?;
            regions.insert(public.primary_region, (map.width, map.height));
            for region in &public.regions {
                ensure(
                    region.map.width > 0
                        && region.map.height > 0
                        && regions
                            .insert(region.id, (region.map.width, region.map.height))
                            .is_none(),
                    "invalid graph region geometry or identity",
                )?;
            }
        } else {
            ensure(
                public.primary_region == 0 && public.regions.is_empty(),
                "public graph regions require a primary map",
            )?;
        }
        let mut indices = HashMap::with_capacity(self.nodes.len());
        let finite =
            |values: &[f32], width| values.len() == width && values.iter().all(|v| v.is_finite());
        for (index, node) in self.nodes.iter().enumerate() {
            let public_endpoint = match node.id {
                GraphNodeId::Agent => true,
                GraphNodeId::Entity(id) => entities.contains(&id),
                GraphNodeId::Region(id) => regions.contains_key(&id),
                GraphNodeId::Location { region, x, y } => regions
                    .get(&region)
                    .is_some_and(|&(width, height)| x < width && y < height),
            };
            ensure(
                public_endpoint
                    && indices.insert(node.id, index).is_none()
                    && finite(&node.features, schema.node_features.len()),
                "invalid graph node identity, reference or feature row",
            )?;
        }
        let mut ids = HashSet::with_capacity(self.edges.len());
        for edge in &self.edges {
            let valid_status = match edge.status {
                EdgeStatus::Superseded { tick } => tick >= edge.tick && tick <= self.tick,
                _ => true,
            };
            ensure(
                ids.insert(edge.id)
                    && edge.kind < schema.edge_types.len()
                    && edge.tick <= self.tick
                    && valid_status
                    && finite(&edge.features, schema.edge_features.len()),
                "invalid graph edge identity, type, time or feature row",
            )?;
            let (Some(&from), Some(&to)) = (indices.get(&edge.from), indices.get(&edge.to)) else {
                return Err(super::Error::Shape("graph edge endpoint is absent".into()));
            };
            visit([from, to], edge);
        }
        Ok(())
    }
}
