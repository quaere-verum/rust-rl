use super::{Argument, Candidate, InputSchema, StructuredSample};
use crate::{Error, Result, core::ensure};
use burn::tensor::{Device, Tensor, TensorData};
use std::{collections::HashMap, hash::Hash};

/// Limits numerical input storage (f32 elements, including all masks and joins).
/// This is a caller allocation budget, not a permanent semantic world limit.
#[derive(Debug, Clone, Copy)]
pub struct PackLimits {
    pub max_elements: usize,
}
impl Default for PackLimits {
    fn default() -> Self {
        Self {
            max_elements: 16 * 1024 * 1024,
        }
    }
}

/// Validated minibatch-local padding. Fields stay private so model code can rely
/// on validation. Candidate keys and real sizes preserve row correspondence.
#[derive(Debug, Clone)]
pub struct StructuredBatch<K> {
    pub(crate) schema: InputSchema,
    pub(crate) spatial: Tensor<4>,
    pub(crate) cell_mask: Tensor<4>,
    pub(crate) entities: Tensor<3>,
    pub(crate) entity_mask: Tensor<3>,
    pub(crate) events: Tensor<3>,
    pub(crate) event_mask: Tensor<3>,
    pub(crate) event_entities: Tensor<3>,
    pub(crate) event_context: Tensor<3>,
    pub(crate) scalars: Tensor<2>,
    pub(crate) candidates: Vec<Vec<Candidate<K>>>,
    pub(crate) sizes: Vec<[usize; 2]>,
    elements: usize,
}
impl<K> StructuredBatch<K> {
    pub fn schema(&self) -> &InputSchema {
        &self.schema
    }
    pub fn len(&self) -> usize {
        self.sizes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sizes.is_empty()
    }
    pub fn input_elements(&self) -> usize {
        self.elements
    }
    pub fn size(&self, sample: usize) -> [usize; 2] {
        self.sizes[sample]
    }
    pub fn candidates(&self, sample: usize) -> &[Candidate<K>] {
        &self.candidates[sample]
    }
    pub fn spatial(&self) -> Tensor<4> {
        self.spatial.clone()
    }
    pub fn cell_mask(&self) -> Tensor<4> {
        self.cell_mask.clone()
    }
    pub fn entity_mask(&self) -> Tensor<3> {
        self.entity_mask.clone()
    }
    pub fn event_features(&self) -> Tensor<3> {
        self.events.clone()
    }
    pub fn event_mask(&self) -> Tensor<3> {
        self.event_mask.clone()
    }
    pub fn event_entities(&self) -> Tensor<3> {
        self.event_entities.clone()
    }
    pub fn event_context(&self) -> Tensor<3> {
        self.event_context.clone()
    }
}
fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| Error::Shape("structured batch size overflow".into()))
    })
}
fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(values, shape), device)
}
// A full masked row separates regions. The current CNN uses radius-one
// convolutions and clears the masked cells between layers.
fn packed_map_size<K>(sample: &StructuredSample<K>) -> Result<[usize; 2]> {
    let mut width = sample.width;
    let mut height = sample.height;
    for region in &sample.regions {
        width = width.max(region.width);
        height = height
            .checked_add(1)
            .and_then(|height| height.checked_add(region.height))
            .ok_or_else(|| Error::Shape("packed spatial height overflow".into()))?;
    }
    Ok([width, height])
}

impl<K: Clone + Eq + Hash> StructuredBatch<K> {
    pub fn pack(
        schema: &InputSchema,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        schema.validate()?;
        Self::pack_with_validated_schema(schema, samples, limits, device)
    }

    /// Model constructors and restores validate their immutable input schema.
    /// Every supplied payload and allocation bound remains checked here.
    pub(crate) fn pack_with_validated_schema(
        schema: &InputSchema,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        ensure(!samples.is_empty(), "empty structured minibatch")?;
        ensure(schema.spatial_channels > 0, "structured CNN requires a map")?;
        let b = samples.len();
        let sizes = samples
            .iter()
            .map(packed_map_size)
            .collect::<Result<Vec<_>>>()?;
        let h = sizes.iter().map(|s| s[1]).max().unwrap();
        let w = sizes.iter().map(|s| s[0]).max().unwrap();
        let n = samples
            .iter()
            .map(|s| s.entities.len())
            .max()
            .unwrap()
            .max(1);
        let t = samples.iter().map(|s| s.events.len()).max().unwrap().max(1);
        let c = schema.spatial_channels;
        let e = schema.entity_features;
        let f = schema.event_features + 1;
        let frozen_events = schema.frozen_events();
        let s = schema.scalar_features;
        // Count every allocated numeric buffer before constructing any tensor or
        // cloning metadata. Include candidate features retained on the host.
        let shapes: [&[usize]; 9] = [
            &[b, c, h, w],
            &[b, 1, h, w],
            &[b, n, e],
            &[b, n, 1],
            &[b, t, f],
            &[b, t, 1],
            &[b, t, n],
            &[b, t, n],
            &[b, s],
        ];
        let mut elements = 0usize;
        for dims in shapes {
            elements = elements
                .checked_add(product(dims)?)
                .ok_or_else(|| Error::Shape("structured batch size overflow".into()))?;
        }
        for sample in samples {
            elements = elements
                .checked_add(product(&[
                    sample.candidates.len(),
                    schema.candidate_features,
                ])?)
                .ok_or_else(|| Error::Shape("candidate storage overflow".into()))?;
        }
        ensure(
            elements <= limits.max_elements && elements <= isize::MAX as usize / size_of::<f32>(),
            "structured batch exceeds allocation budget",
        )?;
        for sample in samples {
            sample.validate(schema)?;
        }
        let mut spatial = vec![0.; b * c * h * w];
        let mut cell_mask = vec![0.; b * h * w];
        let mut entities = vec![0.; b * n * e];
        let mut entity_mask = vec![0.; b * n];
        let mut events = vec![0.; b * t * f];
        let mut event_mask = vec![0.; b * t];
        let mut event_entities = vec![0.; b * t * n];
        let mut event_context = vec![0.; b * t * n];
        let mut scalars = Vec::with_capacity(b * s);
        let mut candidates = Vec::with_capacity(b);
        for (i, sample) in samples.iter().enumerate() {
            let mut copy_region = |region_w: usize, region_h, offset, cells: &[f32]| {
                for y in 0..region_h {
                    let mask = (i * h + offset + y) * w;
                    cell_mask[mask..mask + region_w].fill(1.);
                    for channel in 0..c {
                        let dst = ((i * c + channel) * h + offset + y) * w;
                        let src = (channel * region_h + y) * region_w;
                        spatial[dst..dst + region_w].copy_from_slice(&cells[src..src + region_w]);
                    }
                }
            };
            copy_region(sample.width, sample.height, 0, &sample.spatial);
            let mut ordered: Vec<_> = sample.regions.iter().collect();
            ordered.sort_unstable_by_key(|region| region.id);
            let mut offsets = HashMap::with_capacity(ordered.len());
            let mut y = sample.height;
            for region in ordered {
                y += 1;
                offsets.insert(region.id, y);
                copy_region(region.width, region.height, y, &region.spatial);
                y += region.height;
            }
            let mut actions = sample.candidates.clone();
            for candidate in &mut actions {
                if let Argument::RegionalLocation { region, x, y } = candidate.argument {
                    let offset = if region == sample.primary_region {
                        0
                    } else {
                        offsets[&region]
                    };
                    candidate.argument = Argument::Location { x, y: offset + y };
                }
            }
            candidates.push(actions);
            for (j, row) in sample.entities.iter().enumerate() {
                entities[(i * n + j) * e..(i * n + j + 1) * e].copy_from_slice(row);
                entity_mask[i * n + j] = 1.;
            }
            for (j, event) in sample.events.iter().enumerate() {
                let start = (i * t + j) * f;
                events[start..start + f - 1].copy_from_slice(&event.features);
                event_mask[i * t + j] = 1.;
                if frozen_events {
                    events[start + f - 1] = (j as f64).ln_1p() as f32;
                    continue;
                }
                events[start + f - 1] =
                    j as f32 / sample.events.len().saturating_sub(1).max(1) as f32;
                if let Some(row) = event.entity {
                    event_entities[(i * t + j) * n + row] = 1.;
                }
                for &row in &event.context {
                    event_context[(i * t + j) * n + row] = 1. / event.context.len() as f32;
                }
            }
            scalars.extend_from_slice(&sample.scalars);
        }
        Ok(Self {
            schema: schema.clone(),
            spatial: tensor(spatial, [b, c, h, w], device),
            cell_mask: tensor(cell_mask, [b, 1, h, w], device),
            entities: tensor(entities, [b, n, e], device),
            entity_mask: tensor(entity_mask, [b, n, 1], device),
            events: tensor(events, [b, t, f], device),
            event_mask: tensor(event_mask, [b, t, 1], device),
            event_entities: tensor(event_entities, [b, t, n], device),
            event_context: tensor(event_context, [b, t, n], device),
            scalars: tensor(scalars, [b, s], device),
            candidates,
            sizes,
            elements,
        })
    }
}
