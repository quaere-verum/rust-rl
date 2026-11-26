//! Sparse graph packing and a single trainable directed message-passing layer.
use super::{
    BuildStructuredActionValue, CandidateValues, EncodedWorld, InputSchema, PackLimits,
    StructuredActionValue, StructuredBatch, StructuredModelConfig, StructuredQ, StructuredSample,
};
use crate::{
    Error, Result,
    core::{
        action::{ActionSpace, ConditionalAction},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationSpace,
        graph::{EdgeStatus, GraphNodeId, GraphObservation, GraphSchema},
    },
    nn::{
        checkpoint::RecordBurnExt,
        network::{Mlp, Network},
    },
};
use burn::{
    module::Module,
    tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData, activation::relu},
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::hash::Hash;

fn node_width(schema: &GraphSchema) -> usize {
    schema.node_features.len() + 4
}

fn edge_width(schema: &GraphSchema) -> usize {
    schema.edge_features.len() + schema.edge_types.len() + 7
}

fn floats(values: Vec<f32>, dims: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, dims), device)
}

fn indices(values: Vec<i64>, device: &Device) -> Tensor<1, Int> {
    let count = values.len();
    Tensor::from_data(TensorData::new(values, [count]), device)
}

#[derive(Debug, Clone)]
struct Nodes {
    values: Tensor<2>,
    sample: Tensor<1, Int>,
}

#[derive(Debug, Clone)]
struct Edges {
    values: Tensor<2>,
    from: Tensor<1, Int>,
    to: Tensor<1, Int>,
}

/// Unpadded rows. Each edge's indices are global within this batch, and both
/// endpoints are within the corresponding sample's node range.
#[derive(Debug, Clone)]
pub struct SparseGraphBatch {
    schema: GraphSchema,
    node_offsets: Vec<usize>,
    edge_offsets: Vec<usize>,
    nodes: Option<Nodes>,
    edges: Option<Edges>,
    device: Device,
    elements: usize,
}

impl SparseGraphBatch {
    pub fn pack(
        schema: &GraphSchema,
        frames: &[&GraphObservation],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        schema.validate()?;
        ensure(!frames.is_empty(), "empty sparse graph batch")?;
        let elements = frames
            .iter()
            .try_fold(0usize, |total, frame| {
                let size = frame.packed_elements(schema)?.checked_add(total)?;
                (size <= limits.max_elements && size <= i32::MAX as usize).then_some(size)
            })
            .ok_or_else(|| Error::Shape("sparse graph allocation budget exceeded".into()))?;

        let n = frames.iter().map(|frame| frame.nodes.len()).sum::<usize>();
        let e = frames.iter().map(|frame| frame.edges.len()).sum::<usize>();
        let node_width = node_width(schema);
        let edge_width = edge_width(schema);
        let mut nodes = Vec::with_capacity(n * node_width);
        let mut samples = Vec::with_capacity(n);
        let mut edge_rows = Vec::with_capacity(e * edge_width);
        let mut from = Vec::with_capacity(e);
        let mut to = Vec::with_capacity(e);
        let mut node_offsets = Vec::with_capacity(frames.len() + 1);
        let mut edge_offsets = Vec::with_capacity(frames.len() + 1);
        node_offsets.push(0);
        edge_offsets.push(0);
        for (sample, frame) in frames.iter().enumerate() {
            let offset = samples.len();
            frame.resolve_edges(schema, |[source, target], edge| {
                from.push((offset + source) as i64);
                to.push((offset + target) as i64);
                let start = edge_rows.len();
                edge_rows.resize(start + edge_width, 0.);
                edge_rows[start + edge.kind] = 1.;
                let flags = start + schema.edge_types.len();
                let (status, superseded_age) = match edge.status {
                    EdgeStatus::Observed => (0, 0.),
                    EdgeStatus::Remembered => (1, 0.),
                    EdgeStatus::Disclosed => (2, 0.),
                    EdgeStatus::Superseded { tick } => {
                        (3, ((frame.tick - tick) as f64).ln_1p() as f32)
                    }
                };
                edge_rows[flags + status] = 1.;
                edge_rows[flags + 4] = f32::from(edge.conditional);
                edge_rows[flags + 5] = ((frame.tick - edge.tick) as f64).ln_1p() as f32;
                edge_rows[flags + 6] = superseded_age;
                edge_rows[flags + 7..start + edge_width].copy_from_slice(&edge.features);
            })?;
            for node in &frame.nodes {
                let kind = match node.id {
                    GraphNodeId::Agent => 0,
                    GraphNodeId::Entity(_) => 1,
                    GraphNodeId::Region(_) => 2,
                    GraphNodeId::Location { .. } => 3,
                };
                let mut indicator = [0.; 4];
                indicator[kind] = 1.;
                nodes.extend(indicator);
                nodes.extend_from_slice(&node.features);
                samples.push(sample as i64);
            }
            node_offsets.push(samples.len());
            edge_offsets.push(from.len());
        }
        Ok(Self {
            schema: schema.clone(),
            node_offsets,
            edge_offsets,
            nodes: (n > 0).then(|| Nodes {
                values: floats(nodes, [n, node_width], device),
                sample: indices(samples, device),
            }),
            edges: (e > 0).then(|| Edges {
                values: floats(edge_rows, [e, edge_width], device),
                from: indices(from, device),
                to: indices(to, device),
            }),
            device: device.clone(),
            elements,
        })
    }

    pub fn node_offsets(&self) -> &[usize] {
        &self.node_offsets
    }

    pub fn edge_offsets(&self) -> &[usize] {
        &self.edge_offsets
    }

    pub fn edge_indices(&self) -> Option<(Tensor<1, Int>, Tensor<1, Int>)> {
        self.edges
            .as_ref()
            .map(|edges| (edges.from.clone(), edges.to.clone()))
    }

    pub fn node_features(&self) -> Option<Tensor<2>> {
        self.nodes.as_ref().map(|nodes| nodes.values.clone())
    }

    pub fn edge_features(&self) -> Option<Tensor<2>> {
        self.edges.as_ref().map(|edges| edges.values.clone())
    }

    pub fn input_elements(&self) -> usize {
        self.elements
    }
}

/// Pack both components from the same frame order, so model callers cannot
/// accidentally fuse a graph from one sample with a different observation.
#[derive(Debug, Clone)]
pub struct GraphStructuredBatch {
    base: StructuredBatch<ConditionalAction>,
    graph: SparseGraphBatch,
}

impl GraphStructuredBatch {
    pub fn pack(
        input: &super::InputSchema,
        graph: &GraphSchema,
        frames: &[&GraphObservation],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        input.validate()?;
        let observation_schema = input.observation_schema();
        let action_space = ActionSpace::Conditional(input.action_grammar().into());
        let graphs = SparseGraphBatch::pack(graph, frames, limits, device)?;
        let remaining = limits.max_elements - graphs.input_elements();
        let mut source_elements = 0usize;
        let mut samples = Vec::with_capacity(frames.len());
        for frame in frames {
            let obs = &frame.observation;
            for width in obs
                .spatial
                .iter()
                .map(|map| map.cells.len())
                .chain(obs.regions.iter().map(|region| region.map.cells.len()))
                .chain(obs.entities.iter().map(|row| row.features.len()))
                .chain(
                    obs.events
                        .iter()
                        .flat_map(|row| [row.features.len(), row.context.len()]),
                )
                .chain(obs.candidates.iter().map(|row| row.features.len()))
                .chain([obs.scalars.len()])
            {
                source_elements = source_elements
                    .checked_add(width)
                    .filter(|&n| n <= remaining)
                    .ok_or_else(|| {
                        Error::Shape("structured graph projection budget exceeded".into())
                    })?;
            }
            obs.validate(&observation_schema, &action_space, false)?;
            samples.push(StructuredSample::from_observation_with_validated_schema(
                obs, input,
            )?);
        }
        let base = StructuredBatch::pack_with_validated_schema(
            input,
            &samples,
            PackLimits {
                max_elements: remaining,
            },
            device,
        )?;
        Ok(Self {
            base,
            graph: graphs,
        })
    }

    pub fn graph(&self) -> &SparseGraphBatch {
        &self.graph
    }

    pub fn structured(&self) -> &StructuredBatch<ConditionalAction> {
        &self.base
    }
}

fn add_rows(values: Tensor<2>, index: Tensor<1, Int>, rows: usize, device: &Device) -> Tensor<2> {
    let [_, width] = values.dims();
    Tensor::zeros([rows, width], device).select_assign(0, index, values, IndexingUpdateOp::Add)
}

/// One relation-conditioned directed pass, followed by per-graph mean pooling.
/// The working set scales with the listed nodes and edges.
#[derive(Debug, Module)]
pub struct SparseGraphEncoder {
    pub nodes: Mlp,
    pub messages: Mlp,
    pub update: Mlp,
    #[module(skip)]
    schema: GraphSchema,
    #[module(skip)]
    width: usize,
    #[module(skip)]
    max_activation_elements: usize,
}

impl SparseGraphEncoder {
    pub fn new(
        schema: GraphSchema,
        width: usize,
        max_activation_elements: usize,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        schema.validate()?;
        ensure(
            (1..=1024).contains(&width) && max_activation_elements > 0,
            "invalid sparse graph encoder width or budget",
        )?;
        Ok(Self {
            nodes: Mlp::new_on_device(node_width(&schema), &[], width, rng, device)?,
            messages: Mlp::new_on_device(2 * width + edge_width(&schema), &[], width, rng, device)?,
            update: Mlp::new_on_device(2 * width, &[], width, rng, device)?,
            schema,
            width,
            max_activation_elements,
        })
    }

    pub fn forward(&self, batch: &SparseGraphBatch) -> Result<Tensor<2>> {
        ensure(
            batch.schema == self.schema,
            "sparse graph encoder schema mismatch",
        )?;
        let device = &batch.device;
        let samples = batch.node_offsets.len() - 1;
        let n = *batch.node_offsets.last().unwrap();
        let e = *batch.edge_offsets.last().unwrap();
        let w = self.width;
        for (rows, width) in [
            (samples, w),
            (n, 2 * w),
            (e, 2 * w + edge_width(&self.schema)),
        ] {
            ensure(
                rows.checked_mul(width)
                    .is_some_and(|size| size <= self.max_activation_elements),
                "sparse graph activation budget exceeded",
            )?;
        }
        let Some(nodes) = &batch.nodes else {
            return Ok(Tensor::zeros([samples, w], device));
        };
        let state = relu(self.nodes.forward(nodes.values.clone()));
        let incoming = if let Some(edges) = &batch.edges {
            let messages = Tensor::cat(
                vec![
                    state.clone().select(0, edges.from.clone()),
                    state.clone().select(0, edges.to.clone()),
                    edges.values.clone(),
                ],
                1,
            );
            add_rows(
                relu(self.messages.forward(messages)),
                edges.to.clone(),
                n,
                device,
            )
        } else {
            Tensor::zeros([n, w], device)
        };
        let updated = relu(
            self.update
                .forward(Tensor::cat(vec![state.clone(), incoming], 1)),
        ) + state;
        let count = batch
            .node_offsets
            .windows(2)
            .map(|range| (range[1] - range[0]).max(1) as f32)
            .collect();
        Ok(add_rows(updated, nodes.sample.clone(), samples, device)
            / floats(count, [samples, 1], device))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphStructuredQConfig {
    pub version: u32,
    pub base: StructuredModelConfig,
    pub graph: GraphSchema,
    pub graph_width: usize,
}

impl GraphStructuredQConfig {
    fn validate(&self) -> Result<()> {
        ensure(self.version == 1, "unsupported graph structured Q version")?;
        self.base.validate()?;
        self.graph.validate()?;
        ensure(
            (1..=1024).contains(&self.graph_width),
            "invalid sparse graph encoder width",
        )
    }
}

/// A graph-aware candidate scorer built on the existing structured encoder
/// and heads. Its checkpoint has a distinct kind and explicit graph schema.
#[derive(Debug, Module)]
pub struct GraphStructuredQ {
    pub base: StructuredQ,
    pub graph: SparseGraphEncoder,
    pub fusion: Mlp,
    #[module(skip)]
    config: GraphStructuredQConfig,
}

impl GraphStructuredQ {
    pub fn new(
        config: GraphStructuredQConfig,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        let graph = SparseGraphEncoder::new(
            config.graph.clone(),
            config.graph_width,
            config.base.encoder.max_activation_elements,
            rng,
            device,
        )?;
        let base = StructuredQ::new(config.base.clone(), rng, device)?;
        let fusion = Mlp::new_on_device(
            config.graph_width,
            &[],
            config.base.encoder.fused_width,
            rng,
            device,
        )?;
        Ok(Self {
            base,
            graph,
            fusion,
            config,
        })
    }

    pub fn encode(&self, batch: &GraphStructuredBatch) -> Result<EncodedWorld> {
        self.encode_parts(&batch.base, &batch.graph)
    }

    fn encode_parts<K>(
        &self,
        base: &StructuredBatch<K>,
        graph: &SparseGraphBatch,
    ) -> Result<EncodedWorld> {
        let mut world = self.base.encoder.forward(base)?;
        let graph = self.graph.forward(graph)?;
        world.global = world.global + self.fusion.forward(graph);
        Ok(world)
    }

    pub fn evaluate<'a>(
        &self,
        batch: &'a GraphStructuredBatch,
    ) -> Result<Vec<super::CandidateValues<'a, ConditionalAction>>> {
        self.base.evaluate_with(&batch.base, || self.encode(batch))
    }
}

impl StructuredActionValue for GraphStructuredQ {
    type Configuration = GraphStructuredQConfig;

    fn configuration(&self) -> &Self::Configuration {
        &self.config
    }

    fn validate_configuration(config: &Self::Configuration) -> Result<()> {
        config.validate()
    }

    fn input_schema(&self) -> &InputSchema {
        &self.config.base.encoder.input
    }

    fn validate_spaces(
        &self,
        observations: &ObservationSpace,
        actions: &ActionSpace,
    ) -> Result<()> {
        let input = self.input_schema();
        ensure(
            matches!(observations, ObservationSpace::Graph(schema)
                if schema.base == input.observation_schema() && schema.graph == self.config.graph)
                && matches!(actions, ActionSpace::Conditional(grammar)
                    if **grammar == input.action_grammar()),
            "graph structured model and environment semantics differ",
        )
    }

    fn model_device(&self) -> Device {
        self.base.model_device()
    }

    #[cfg(feature = "diagnostics")]
    fn diagnostic_global(
        &self,
        observation: &crate::core::env::Observation,
        limits: PackLimits,
        max_width: usize,
    ) -> Result<Option<Vec<f32>>> {
        if self.config.base.encoder.fused_width > max_width {
            return Ok(None);
        }
        let frame = observation
            .as_graph()
            .ok_or_else(|| Error::Shape("graph encoder requires a graph frame".into()))?;
        let batch = GraphStructuredBatch::pack(
            self.input_schema(),
            &self.config.graph,
            &[frame],
            limits,
            &self.model_device(),
        )?;
        crate::nn::to_vec(self.encode(&batch)?.global).map(Some)
    }

    fn candidate_values<'a, K: Clone + Eq + Hash>(
        &self,
        _: &'a [StructuredSample<K>],
        _: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        Err(Error::Shape(
            "graph structured Q requires public graph frames".into(),
        ))
    }

    fn candidate_values_with_graphs<'a, K: Clone + Eq + Hash>(
        &self,
        samples: &'a [StructuredSample<K>],
        graphs: &[&GraphObservation],
        limits: PackLimits,
    ) -> Result<Vec<CandidateValues<'a, K>>> {
        ensure(
            !graphs.is_empty() && graphs.len() == samples.len(),
            "graph structured Q requires one graph per structured sample",
        )?;
        let device = self.model_device();
        let graph = SparseGraphBatch::pack(&self.config.graph, graphs, limits, &device)?;
        let base = StructuredBatch::pack_with_validated_schema(
            self.input_schema(),
            samples,
            PackLimits {
                max_elements: limits.max_elements - graph.input_elements(),
            },
            &device,
        )?;
        self.base
            .evaluate_with(&base, || self.encode_parts(&base, &graph))?
            .into_iter()
            .zip(samples)
            .map(|(row, sample)| {
                CandidateValues::from_candidates(
                    &sample.candidates,
                    row.values,
                    self.input_schema().operations.len(),
                )
            })
            .collect()
    }
}

impl BuildStructuredActionValue for GraphStructuredQ {
    fn from_configuration(
        config: Self::Configuration,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new(config, rng, device)
    }
}

impl Checkpoint for GraphStructuredQ {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("graph-structured-q-v1", &self.config)?;
        record.model("q", self)?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("graph-structured-q-v1")?;
        let config: GraphStructuredQConfig = serde_json::from_value(record.metadata.clone())?;
        ensure(
            config == self.config,
            "graph structured Q checkpoint configuration mismatch",
        )?;
        *self = super::checkpoint::restore(record, "q", self)?;
        Ok(())
    }
}
