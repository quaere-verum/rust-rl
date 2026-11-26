use crate::{
    Error, Result,
    core::{
        action::{ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation},
        ensure,
        env::ObservationSpace,
        structured::{
            ArgumentKind, FROZEN_EVENT_CONTEXT_CONVENTION, INCREMENTAL_EVENT_CONVENTION,
            ObservationSchema, StructuredObservation,
        },
    },
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Argument {
    None,
    /// Batch-local entity row, never an environment identity.
    Entity(usize),
    /// Coordinates in this sample's real map, never in the padded extent.
    Location {
        x: usize,
        y: usize,
    },
    /// A public region and coordinates local to it.
    RegionalLocation {
        region: u32,
        x: usize,
        y: usize,
    },
}
impl Argument {
    pub fn kind(self) -> ArgumentKind {
        match self {
            Self::None => ArgumentKind::None,
            Self::Entity(_) => ArgumentKind::Entity,
            Self::Location { .. } | Self::RegionalLocation { .. } => ArgumentKind::Location,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSchema {
    pub version: u32,
    pub semantics: super::SemanticInput,
    pub spatial_channels: usize,
    pub entity_features: usize,
    pub event_features: usize,
    pub scalar_features: usize,
    pub candidate_features: usize,
    /// Versioned operation vocabulary and its conditional argument grammar.
    pub operations: Vec<ArgumentKind>,
}
impl InputSchema {
    pub fn validate(&self) -> Result<()> {
        self.semantics.validate()?;
        ensure(
            self.version == 2
                && !self.operations.is_empty()
                && self.operations.len() == self.semantics.operations.len()
                && self.spatial_channels == self.semantics.spatial.len()
                && self.entity_features == self.semantics.entities.len()
                && self.event_features == self.semantics.events.len()
                && self.scalar_features == self.semantics.scalars.len()
                && self.candidate_features == self.semantics.candidates.len()
                && [
                    self.entity_features,
                    self.event_features,
                    self.scalar_features,
                    self.candidate_features,
                ]
                .iter()
                .all(|&n| n > 0 && n <= 65536),
            "invalid structured input schema",
        )
    }

    pub(crate) fn frozen_events(&self) -> bool {
        self.semantics
            .conventions
            .iter()
            .any(|convention| convention == FROZEN_EVENT_CONTEXT_CONVENTION)
    }

    pub fn observation_schema(&self) -> ObservationSchema {
        let semantics = &self.semantics;
        let conventions = semantics
            .conventions
            .iter()
            .map(|convention| match convention.as_str() {
                // The historical model descriptor names the packed layout.
                "spatial=CHW;coordinates=row_major_top_left" => {
                    "spatial=cell_major;coordinates=row_major_top_left".into()
                }
                _ => convention.clone(),
            })
            .collect();
        ObservationSchema {
            namespace: semantics.namespace.clone(),
            version: semantics.observation_version,
            spatial: semantics.spatial.clone(),
            entities: semantics.entities.clone(),
            events: semantics.events.clone(),
            scalars: semantics.scalars.clone(),
            vocabularies: semantics.vocabularies.clone(),
            conventions,
        }
    }

    pub fn action_grammar(&self) -> ActionGrammar {
        let semantics = &self.semantics;
        ActionGrammar {
            namespace: semantics.namespace.clone(),
            version: semantics.action_version,
            operations: semantics
                .operations
                .iter()
                .zip(&self.operations)
                .map(|(name, &argument)| Operation {
                    name: name.clone(),
                    argument,
                })
                .collect(),
            candidate_features: semantics.candidates.clone(),
        }
    }

    pub fn validate_spaces(
        &self,
        observation: &ObservationSpace,
        action: &ActionSpace,
    ) -> Result<()> {
        self.validate()?;
        ensure(
            matches!(observation, ObservationSpace::Structured(schema) if **schema == self.observation_schema())
                && matches!(action, ActionSpace::Conditional(grammar) if **grammar == self.action_grammar()),
            "structured model and environment semantics differ",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate<K> {
    /// The complete semantic action (or caller-owned stable key), returned on selection.
    pub key: K,
    pub operation: usize,
    pub argument: Argument,
    /// Public costs/requirements, with explicit known-value flags where needed.
    pub features: Vec<f32>,
    pub enabled: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub features: Vec<f32>,
    pub entity: Option<usize>,
    /// Set of entity rows describing acquisition-time context, not arbitrary IDs.
    pub context: Vec<usize>,
}
/// An additional public map in the same CHW layout as the primary map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialSample {
    pub id: u32,
    pub width: usize,
    pub height: usize,
    pub spatial: Vec<f32>,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn cell_major_to_chw(
    map: &crate::core::structured::SpatialObservation,
    channels: usize,
) -> Vec<f32> {
    let area = map.width * map.height;
    let mut spatial = vec![0.; map.cells.len()];
    for (cell, fields) in map.cells.chunks_exact(channels).enumerate() {
        for (channel, &value) in fields.iter().enumerate() {
            spatial[channel * area + cell] = value;
        }
    }
    spatial
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredSample<K> {
    pub width: usize,
    pub height: usize,
    /// CHW; unknown cells are real cells, represented by caller-provided knowledge features.
    pub spatial: Vec<f32>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub primary_region: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<SpatialSample>,
    pub entities: Vec<Vec<f32>>,
    pub scalars: Vec<f32>,
    /// Chronological public records. The packer appends a normalized ordinal to
    /// each event row, preserving tie order even when timestamps are equal.
    pub events: Vec<Event>,
    pub candidates: Vec<Candidate<K>>,
}

impl StructuredSample<ConditionalAction> {
    /// Project a public decision into model layout. Stable references survive
    /// only as complete action keys; numeric joins use rows of this sample.
    pub fn from_observation(
        observation: &StructuredObservation,
        schema: &InputSchema,
    ) -> Result<Self> {
        schema.validate()?;
        Self::from_observation_with_validated_schema(observation, schema)
    }

    /// The learner owns an immutable schema already checked by its constructor
    /// or restore boundary. Observation values and references remain checked.
    pub(crate) fn from_observation_with_validated_schema(
        observation: &StructuredObservation,
        schema: &InputSchema,
    ) -> Result<Self> {
        Self::project_observation(observation, schema)
    }

    fn project_observation(
        observation: &StructuredObservation,
        schema: &InputSchema,
    ) -> Result<Self> {
        let channels = schema.spatial_channels;
        observation.validate_spatial(channels)?;
        let (width, height, spatial) = observation.spatial.as_ref().map_or_else(
            || (0, 0, Vec::new()),
            |map| (map.width, map.height, cell_major_to_chw(map, channels)),
        );
        let regions = observation
            .regions
            .iter()
            .map(|region| SpatialSample {
                id: region.id,
                width: region.map.width,
                height: region.map.height,
                spatial: cell_major_to_chw(&region.map, channels),
            })
            .collect();
        let rows: HashMap<_, _> = observation
            .entities
            .iter()
            .enumerate()
            .map(|(row, entity)| (entity.id, row))
            .collect();
        ensure(
            rows.len() == observation.entities.len(),
            "duplicate structured entity identity",
        )?;
        let resolve = |id| {
            rows.get(&id)
                .copied()
                .ok_or_else(|| Error::Shape("structured reference to absent entity".into()))
        };
        let acquisition_notebook = schema
            .semantics
            .conventions
            .iter()
            .any(|convention| convention == INCREMENTAL_EVENT_CONVENTION);
        let frozen_context = schema.frozen_events();
        let events = observation
            .events
            .iter()
            .map(|event| {
                // Stamped history keeps its feature rows after an entity departs.
                // Join each historical reference that is present in this frame.
                let (entity, context) = if frozen_context && event.stamp.is_some() {
                    (None, Vec::new())
                } else if acquisition_notebook && event.stamp.is_some() {
                    (
                        event.entity.and_then(|id| rows.get(&id).copied()),
                        event
                            .context
                            .iter()
                            .filter_map(|id| rows.get(id).copied())
                            .collect(),
                    )
                } else {
                    (
                        event.entity.map(resolve).transpose()?,
                        event
                            .context
                            .iter()
                            .copied()
                            .map(resolve)
                            .collect::<Result<_>>()?,
                    )
                };
                Ok(Event {
                    features: event.features.to_vec(),
                    entity,
                    context,
                })
            })
            .collect::<Result<_>>()?;
        let candidates = observation
            .candidates
            .iter()
            .map(|candidate| {
                let key = candidate.action;
                let argument = match key.argument {
                    None => Argument::None,
                    Some(ActionArgument::Entity(id)) => Argument::Entity(resolve(id)?),
                    Some(ActionArgument::Location { x, y }) => Argument::Location { x, y },
                    Some(ActionArgument::RegionalLocation { region, x, y }) => {
                        Argument::RegionalLocation { region, x, y }
                    }
                };
                Ok(Candidate {
                    key,
                    operation: key.operation,
                    argument,
                    features: candidate.features.clone(),
                    enabled: candidate.enabled,
                })
            })
            .collect::<Result<_>>()?;
        let sample = Self {
            width,
            height,
            spatial,
            primary_region: observation.primary_region,
            regions,
            entities: observation
                .entities
                .iter()
                .map(|entity| entity.features.clone())
                .collect(),
            scalars: observation.scalars.clone(),
            events,
            candidates,
        };
        sample.validate(schema)?;
        Ok(sample)
    }
}

fn row(values: &[f32], width: usize) -> Result<()> {
    ensure(
        values.len() == width && values.iter().all(|v| v.is_finite()),
        "invalid structured feature row",
    )
}
impl<K> StructuredSample<K> {
    pub(crate) fn region_size(&self, id: u32) -> Option<[usize; 2]> {
        if id == self.primary_region {
            (self.width > 0 && self.height > 0).then_some([self.width, self.height])
        } else {
            self.regions
                .iter()
                .find(|region| region.id == id)
                .map(|region| [region.width, region.height])
        }
    }
}

impl<K: Eq + Hash> StructuredSample<K> {
    pub(crate) fn validate(&self, schema: &InputSchema) -> Result<()> {
        ensure(
            if schema.spatial_channels == 0 {
                self.width == 0
                    && self.height == 0
                    && self.primary_region == 0
                    && self.regions.is_empty()
            } else {
                self.width > 0 && self.height > 0
            },
            "structured map disagrees with model schema",
        )?;
        let count = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(schema.spatial_channels));
        ensure(
            count == Some(self.spatial.len()),
            "invalid structured spatial dimensions",
        )?;
        row(&self.spatial, self.spatial.len())?;
        let mut ids = HashSet::new();
        for region in &self.regions {
            ensure(
                region.id != self.primary_region
                    && ids.insert(region.id)
                    && region.width > 0
                    && region.height > 0
                    && region
                        .width
                        .checked_mul(region.height)
                        .and_then(|n| n.checked_mul(schema.spatial_channels))
                        == Some(region.spatial.len()),
                "invalid sample region geometry or identity",
            )?;
            row(&region.spatial, region.spatial.len())?;
        }
        self.validate_nonspatial(schema)
    }

    fn validate_nonspatial(&self, schema: &InputSchema) -> Result<()> {
        row(&self.scalars, schema.scalar_features)?;
        for entity in &self.entities {
            row(entity, schema.entity_features)?;
        }
        for event in &self.events {
            row(&event.features, schema.event_features)?;
            ensure(
                event.entity.is_none_or(|i| i < self.entities.len()),
                "event target outside entity table",
            )?;
            let mut seen = HashSet::new();
            ensure(
                event
                    .context
                    .iter()
                    .all(|&i| i < self.entities.len() && seen.insert(i)),
                "invalid event context set",
            )?;
        }
        let mut keys = HashSet::new();
        let mut actions = HashSet::new();
        for c in &self.candidates {
            ensure(
                c.operation < schema.operations.len()
                    && schema.operations[c.operation] == c.argument.kind(),
                "candidate grammar mismatch",
            )?;
            row(&c.features, schema.candidate_features)?;
            ensure(
                keys.insert(&c.key)
                    && actions.insert((
                        c.operation,
                        match c.argument {
                            Argument::Location { x, y } => Argument::RegionalLocation {
                                region: self.primary_region,
                                x,
                                y,
                            },
                            argument => argument,
                        },
                    )),
                "duplicate structured candidate",
            )?;
            ensure(
                match c.argument {
                    Argument::None => true,
                    Argument::Entity(i) => i < self.entities.len(),
                    Argument::Location { x, y } => x < self.width && y < self.height,
                    Argument::RegionalLocation { region, x, y } => self
                        .region_size(region)
                        .is_some_and(|[w, h]| x < w && y < h),
                },
                "candidate target outside real input",
            )?;
        }
        Ok(())
    }
}
