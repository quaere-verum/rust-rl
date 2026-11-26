use super::super::{Argument, InputSchema, PackLimits, StructuredSample};
use crate::{Error, Result, core::ensure};
use burn::tensor::{Device, Int, Tensor, TensorData};
use std::hash::Hash;

/// Checked non-spatial tensors, with batch-local padding and row gathers.
/// Numeric identity never enters a feature. Local indices only implement joins.
#[derive(Debug, Clone)]
pub struct LightweightBatch {
    pub(super) schema: InputSchema,
    pub(super) entities: Tensor<3>,
    pub(super) entity_mask: Tensor<3>,
    pub(super) events: Tensor<3>,
    pub(super) event_mask: Tensor<3>,
    pub(super) subjects: Tensor<3>,
    pub(super) contexts: Tensor<3>,
    pub(super) scalars: Tensor<2>,
    pub(super) counts: Tensor<2>,
    pub(super) operations: Tensor<2>,
    pub(super) arguments: Tensor<2>,
    pub(super) owners: Tensor<1, Int>,
    pub(super) targets: Tensor<1, Int>,
    pub(super) candidate_count: usize,
    elements: usize,
}
fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| Error::Shape("lightweight packing size overflow".into()))
    })
}
fn tensor<const D: usize>(v: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(v, shape), device)
}

/// Padded numeric layout, determined before allocating rows or tensors.
#[derive(Clone, Copy)]
pub(super) struct LightweightLayout {
    pub samples: usize,
    pub entities: usize,
    pub events: usize,
    pub candidates: usize,
}

impl LightweightLayout {
    pub fn allocation_elements(self, schema: &InputSchema) -> Result<usize> {
        let Self {
            samples: b,
            entities: n,
            events: t,
            candidates,
        } = self;
        let p = candidates.max(1);
        let shapes: [&[usize]; 12] = [
            &[b, n, schema.entity_features],
            &[b, n, 1],
            &[b, t, schema.event_features + 1],
            &[b, t, 1],
            &[b, t, n],
            &[b, t, n],
            &[b, schema.scalar_features],
            &[b, 2],
            &[p, schema.operations.len()],
            &[p, schema.candidate_features + 5],
            &[p, 2],
            &[p, 2],
        ];
        ensure(
            b > 0 && product(&[b, n])? < i32::MAX as usize,
            "invalid lightweight batch size",
        )?;
        shapes.into_iter().try_fold(0usize, |a, dims| {
            a.checked_add(product(dims)?)
                .ok_or_else(|| Error::Shape("lightweight packing size overflow".into()))
        })
    }
}

#[derive(Clone, Copy)]
pub(super) enum Regions<'a> {
    Public(&'a [crate::core::structured::SpatialRegion]),
    Sample(&'a [crate::nn::structured::SpatialSample]),
}

impl Regions<'_> {
    fn size(self, id: u32) -> [usize; 2] {
        match self {
            Self::Public(regions) => regions
                .iter()
                .find(|region| region.id == id)
                .map(|region| [region.map.width, region.map.height]),
            Self::Sample(regions) => regions
                .iter()
                .find(|region| region.id == id)
                .map(|region| [region.width, region.height]),
        }
        .expect("candidate has a validated region")
    }
}

/// Rows already validated by their source, with real map dimensions for coordinates.
pub(super) struct LightweightRows<'a, K> {
    pub width: usize,
    pub height: usize,
    pub primary_region: u32,
    pub regions: Regions<'a>,
    pub entities: &'a [Vec<f32>],
    pub scalars: &'a [f32],
    pub events: &'a [super::super::Event],
    pub candidates: &'a [super::super::Candidate<K>],
}

impl LightweightBatch {
    /// Allocated numeric input storage in f32-sized words; integer gathers are
    /// conservatively charged two words each. No spatial buffer is allocated.
    pub fn input_elements(&self) -> usize {
        self.elements
    }
    pub fn schema(&self) -> &InputSchema {
        &self.schema
    }
    pub fn pack<K: Eq + Hash>(
        schema: &InputSchema,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        schema.validate()?;
        Self::pack_with_validated_schema(schema, samples, limits, device)
    }

    /// The owning model binds a constructor-validated immutable schema.
    pub(crate) fn pack_with_validated_schema<K: Eq + Hash>(
        schema: &InputSchema,
        samples: &[StructuredSample<K>],
        limits: PackLimits,
        device: &Device,
    ) -> Result<Self> {
        ensure(!samples.is_empty(), "empty lightweight minibatch")?;
        let mut layout = LightweightLayout {
            samples: samples.len(),
            entities: 1,
            events: 1,
            candidates: 0,
        };
        for sample in samples {
            layout.entities = layout.entities.max(sample.entities.len());
            layout.events = layout.events.max(sample.events.len());
            layout.candidates = layout
                .candidates
                .checked_add(sample.candidates.len())
                .ok_or_else(|| Error::Shape("lightweight candidate count overflow".into()))?;
        }
        let elements = layout.allocation_elements(schema)?;
        ensure(
            elements <= limits.max_elements && elements <= isize::MAX as usize / size_of::<f32>(),
            "lightweight batch exceeds allocation budget",
        )?;
        for s in samples {
            s.validate(schema)?;
        }
        Ok(Self::pack_rows(
            schema,
            layout,
            elements,
            samples.iter().map(|sample| LightweightRows {
                width: sample.width,
                height: sample.height,
                primary_region: sample.primary_region,
                regions: Regions::Sample(&sample.regions),
                entities: &sample.entities,
                scalars: &sample.scalars,
                events: &sample.events,
                candidates: &sample.candidates,
            }),
            device,
        ))
    }

    /// Call after checking the allocation budget and validating all row widths and joins.
    pub(super) fn pack_rows<'a, K: 'a>(
        schema: &InputSchema,
        layout: LightweightLayout,
        elements: usize,
        samples: impl Iterator<Item = LightweightRows<'a, K>>,
        device: &Device,
    ) -> Self {
        let LightweightLayout {
            samples: b,
            entities: n,
            events: t,
            candidates: c,
        } = layout;
        let p = c.max(1);
        let e = schema.entity_features;
        let f = schema.event_features + 1;
        let frozen_events = schema.frozen_events();
        let o = schema.operations.len();
        let a = schema.candidate_features + 5;
        let mut entities = vec![0.; b * n * e];
        let mut entity_mask = vec![0.; b * n];
        let mut events = vec![0.; b * t * f];
        let mut event_mask = vec![0.; b * t];
        let mut subjects = vec![0.; b * t * n];
        let mut contexts = vec![0.; b * t * n];
        let mut scalars = Vec::with_capacity(b * schema.scalar_features);
        let mut counts = Vec::with_capacity(b * 2);
        let mut operations = vec![0.; p * o];
        let mut arguments = vec![0.; p * a];
        let mut owners = vec![0i32; p];
        let mut targets = vec![(b * n) as i32; p];
        let mut row = 0;
        for (i, s) in samples.enumerate() {
            for (j, entity) in s.entities.iter().enumerate() {
                entities[(i * n + j) * e..(i * n + j + 1) * e].copy_from_slice(entity);
                entity_mask[i * n + j] = 1.;
            }
            for (j, event) in s.events.iter().enumerate() {
                let start = (i * t + j) * f;
                events[start..start + f - 1].copy_from_slice(&event.features);
                event_mask[i * t + j] = 1.;
                if frozen_events {
                    events[start + f - 1] = (j as f64).ln_1p() as f32;
                    continue;
                }
                events[start + f - 1] = (j + 1) as f32 / s.events.len() as f32;
                if let Some(k) = event.entity {
                    subjects[(i * t + j) * n + k] = 1.;
                }
                for &k in &event.context {
                    contexts[(i * t + j) * n + k] = 1.;
                }
            }
            scalars.extend_from_slice(s.scalars);
            counts.extend([
                (s.entities.len() as f32).ln_1p(),
                (s.events.len() as f32).ln_1p(),
            ]);
            for candidate in s.candidates {
                owners[row] = i as i32;
                operations[row * o + candidate.operation] = 1.;
                let start = row * a;
                arguments[start..start + schema.candidate_features]
                    .copy_from_slice(&candidate.features);
                let flags = start + schema.candidate_features;
                match candidate.argument {
                    Argument::None => arguments[flags] = 1.,
                    Argument::Entity(k) => {
                        arguments[flags + 1] = 1.;
                        targets[row] = (i * n + k) as i32;
                    }
                    argument @ (Argument::Location { .. } | Argument::RegionalLocation { .. }) => {
                        let (x, y, [width, height]) = match argument {
                            Argument::Location { x, y } => (x, y, [s.width, s.height]),
                            Argument::RegionalLocation { region, x, y } => {
                                let size = if region == s.primary_region {
                                    [s.width, s.height]
                                } else {
                                    s.regions.size(region)
                                };
                                (x, y, size)
                            }
                            _ => unreachable!(),
                        };
                        arguments[flags + 2] = 1.;
                        arguments[flags + 3] = x as f32 / width.saturating_sub(1).max(1) as f32;
                        arguments[flags + 4] = y as f32 / height.saturating_sub(1).max(1) as f32;
                    }
                }
                row += 1;
            }
        }
        Self {
            schema: schema.clone(),
            entities: tensor(entities, [b, n, e], device),
            entity_mask: tensor(entity_mask, [b, n, 1], device),
            events: tensor(events, [b, t, f], device),
            event_mask: tensor(event_mask, [b, t, 1], device),
            subjects: tensor(subjects, [b, t, n], device),
            contexts: tensor(contexts, [b, t, n], device),
            scalars: tensor(scalars, [b, schema.scalar_features], device),
            counts: tensor(counts, [b, 2], device),
            operations: tensor(operations, [p, o], device),
            arguments: tensor(arguments, [p, a], device),
            owners: Tensor::from_data(TensorData::new(owners, [p]), device),
            targets: Tensor::from_data(TensorData::new(targets, [p]), device),
            candidate_count: c,
            elements,
        }
    }
}
