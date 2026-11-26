//! Public semantic features and complete legal candidates. Tensor layouts and
//! model-specific row indices are deliberately absent from these contracts.
use super::{
    Result,
    action::{ActionArgument, ActionGrammar, ActionSpace, ConditionalAction},
    ensure,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureIdentity {
    pub name: String,
    pub transform: String,
}

impl FeatureIdentity {
    pub fn new(name: impl Into<String>, transform: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: transform.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocabularyIdentity {
    pub name: String,
    pub categories: Vec<String>,
}

pub(crate) fn named<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .all(|name| !name.trim().is_empty() && name.len() <= 4096 && seen.insert(name))
}

pub(crate) fn feature_fields(fields: &[FeatureIdentity]) -> bool {
    fields.len() <= 65536
        && named(fields.iter().map(|field| field.name.as_str()))
        && fields
            .iter()
            .all(|field| !field.transform.trim().is_empty() && field.transform.len() <= 4096)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationSchema {
    pub namespace: String,
    pub version: u32,
    pub spatial: Vec<FeatureIdentity>,
    pub entities: Vec<FeatureIdentity>,
    pub events: Vec<FeatureIdentity>,
    pub scalars: Vec<FeatureIdentity>,
    pub vocabularies: Vec<VocabularyIdentity>,
    pub conventions: Vec<String>,
}

impl ObservationSchema {
    pub fn validate(&self) -> Result<()> {
        let groups = [&self.spatial, &self.entities, &self.events, &self.scalars];
        ensure(
            self.version > 0
                && !self.namespace.trim().is_empty()
                && self.namespace.len() <= 4096
                && groups.iter().any(|fields| !fields.is_empty())
                && groups.iter().all(|fields| feature_fields(fields))
                && self.vocabularies.len() <= 1024
                && named(self.vocabularies.iter().map(|v| v.name.as_str()))
                && self.vocabularies.iter().all(|v| {
                    !v.categories.is_empty()
                        && v.categories.len() <= 65536
                        && named(v.categories.iter().map(String::as_str))
                })
                && self.conventions.len() <= 1024
                && named(self.conventions.iter().map(String::as_str)),
            "invalid public observation schema",
        )
    }
}

/// Opaque within an observation namespace. Reordering rows does not change
/// the entity's identity or the meaning of actions and events that reference it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId {
    pub kind: u32,
    pub id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArgumentKind {
    Reserved,
    None,
    Entity,
    Location,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpatialObservation {
    pub width: usize,
    pub height: usize,
    /// Row-major cells, each holding the declared fields in schema order.
    pub cells: Vec<f32>,
}

/// A discovered region using the same channel schema as the primary map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialRegion {
    pub id: u32,
    pub map: SpatialObservation,
}

impl SpatialObservation {
    pub(crate) fn valid(&self, channels: usize) -> bool {
        channels > 0
            && self.width > 0
            && self.height > 0
            && self
                .width
                .checked_mul(self.height)
                .and_then(|cells| cells.checked_mul(channels))
                .is_some_and(|len| feature_row(&self.cells, len))
    }

    fn matches(&self, other: &Self) -> bool {
        self.width == other.width
            && self.height == other.height
            && equal_features(&self.cells, &other.cells)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRow {
    pub id: EntityId,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicEvent {
    /// Immutable numerical payload shared by retained factual frames.
    pub features: Arc<[f32]>,
    pub entity: Option<EntityId>,
    /// Public acquisition-time entity context shared until the event is dropped.
    pub context: Arc<[EntityId]>,
    /// Present for append-only notebooks with acquisition-time feature rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stamp: Option<EventStamp>,
}

pub const INCREMENTAL_EVENT_CONVENTION: &str = "rust_rl.incremental_events.v1:append_only_provenance_suffix;freeze_at_decision_availability;observe_idempotent";

/// Stamped numerical rows include their acquisition context. A model must
/// not join those rows to the current entity feature table or renormalize
/// their row positions as later acquisitions arrive.
pub const FROZEN_EVENT_CONTEXT_CONVENTION: &str =
    "rust_rl.frozen_event_context.v1:stamped_rows_are_self_contained;absolute_log_row_position";

/// Stable public provenance. An ordinal is local to its source within the
/// episode. The enclosing notebook supplies the order between equal ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventStamp {
    pub source: u32,
    pub ordinal: u64,
    pub tick: u64,
}

impl PublicEvent {
    fn matches(&self, other: &Self) -> bool {
        self.stamp == other.stamp
            && self.entity == other.entity
            && (Arc::ptr_eq(&self.context, &other.context) || self.context == other.context)
            && (Arc::ptr_eq(&self.features, &other.features)
                || equal_features(&self.features, &other.features))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionCandidate {
    pub action: ConditionalAction,
    pub features: Vec<f32>,
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredObservation {
    pub spatial: Option<SpatialObservation>,
    /// Identity of the primary map. Single-map observations default to zero.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub primary_region: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<SpatialRegion>,
    pub entities: Vec<EntityRow>,
    /// Chronological, in source order for equal times.
    pub events: Vec<PublicEvent>,
    pub scalars: Vec<f32>,
    pub candidates: Vec<ActionCandidate>,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn feature_row(values: &[f32], width: usize) -> bool {
    values.len() == width && values.iter().all(|value| value.is_finite())
}

pub(crate) fn equal_features(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

impl StructuredObservation {
    pub fn region_map(&self, id: u32) -> Option<&SpatialObservation> {
        if id == self.primary_region {
            self.spatial.as_ref()
        } else {
            self.regions
                .iter()
                .find(|region| region.id == id)
                .map(|region| &region.map)
        }
    }

    pub(crate) fn validate_spatial(&self, channels: usize) -> Result<()> {
        let valid = match &self.spatial {
            None => channels == 0 && self.primary_region == 0 && self.regions.is_empty(),
            Some(map) => {
                let mut ids = HashSet::new();
                map.valid(channels)
                    && self.regions.iter().all(|region| {
                        region.id != self.primary_region
                            && ids.insert(region.id)
                            && region.map.valid(channels)
                    })
            }
        };
        ensure(
            valid,
            "invalid spatial map dimensions, region identities or features",
        )
    }

    pub(crate) fn has_event_stamps(&self) -> bool {
        self.events
            .first()
            .is_some_and(|event| event.stamp.is_some())
    }

    fn validate_event_order(&self) -> Result<()> {
        let mut ordinals = HashMap::new();
        let mut tick = 0;
        for event in &self.events {
            let stamp = event.stamp.ok_or_else(|| {
                super::Error::Shape("incremental event lacks public provenance".into())
            })?;
            ensure(
                stamp.tick >= tick
                    && ordinals
                        .insert(stamp.source, stamp.ordinal)
                        .is_none_or(|old| old < stamp.ordinal),
                "duplicate or out-of-order public event provenance",
            )?;
            tick = stamp.tick;
        }
        Ok(())
    }

    /// Borrow the events first available at this observation. Historical links
    /// may refer to departed entities. Only the introduced rows are joined to
    /// this frame's entity table.
    pub fn introduced_events(&self, previous: Option<&Self>) -> Result<&[PublicEvent]> {
        let prefix = previous.map_or(&[][..], |frame| frame.events.as_slice());
        ensure(
            self.events.len() >= prefix.len()
                && self
                    .events
                    .iter()
                    .zip(prefix)
                    .all(|(new, old)| new.matches(old)),
            "public event prefix changed",
        )?;
        self.validate_event_order()?;
        let introduced = &self.events[prefix.len()..];
        self.validate_event_links(introduced)?;
        Ok(introduced)
    }

    pub(crate) fn validate_event_links(&self, events: &[PublicEvent]) -> Result<()> {
        if !events.is_empty() {
            let entities: HashSet<_> = self.entities.iter().map(|entity| entity.id).collect();
            ensure(
                events.iter().all(|event| {
                    event.entity.is_none_or(|id| entities.contains(&id))
                        && event.context.iter().all(|id| entities.contains(id))
                }),
                "introduced event refers to an absent acquisition entity",
            )?;
        }
        Ok(())
    }

    /// Charge shared payloads per retained reference. Bounded replay canonicalizes
    /// capacities to make this charge stable across snapshot restoration.
    pub(crate) fn payload_bytes(&self, allocated: bool) -> usize {
        fn bytes<T>(values: &Vec<T>, allocated: bool) -> usize {
            (if allocated {
                values.capacity()
            } else {
                values.len()
            }) * size_of::<T>()
        }
        size_of::<Self>()
            + 2 * size_of::<usize>()
            + self
                .spatial
                .as_ref()
                .map_or(0, |map| bytes(&map.cells, allocated))
            + bytes(&self.regions, allocated)
            + self
                .regions
                .iter()
                .map(|region| bytes(&region.map.cells, allocated))
                .sum::<usize>()
            + bytes(&self.scalars, allocated)
            + bytes(&self.entities, allocated)
            + bytes(&self.events, allocated)
            + bytes(&self.candidates, allocated)
            + self
                .entities
                .iter()
                .map(|row| bytes(&row.features, allocated))
                .sum::<usize>()
            + self
                .events
                .iter()
                .map(|row| {
                    4 * size_of::<usize>()
                        + size_of_val(row.features.as_ref())
                        + size_of_val(row.context.as_ref())
                })
                .sum::<usize>()
            + self
                .candidates
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
        if let Some(map) = &mut self.spatial {
            fit(&mut map.cells);
        }
        fit(&mut self.regions);
        for region in &mut self.regions {
            fit(&mut region.map.cells);
        }
        fit(&mut self.scalars);
        fit(&mut self.entities);
        fit(&mut self.events);
        fit(&mut self.candidates);
        for row in &mut self.entities {
            fit(&mut row.features);
        }
        for row in &mut self.candidates {
            fit(&mut row.features);
        }
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        let same_map = match (&self.spatial, &other.spatial) {
            (None, None) => true,
            (Some(a), Some(b)) => a.matches(b),
            _ => false,
        };
        same_map
            && self.primary_region == other.primary_region
            && self.regions.len() == other.regions.len()
            && self
                .regions
                .iter()
                .zip(&other.regions)
                .all(|(a, b)| a.id == b.id && a.map.matches(&b.map))
            && equal_features(&self.scalars, &other.scalars)
            && self.entities.len() == other.entities.len()
            && self
                .entities
                .iter()
                .zip(&other.entities)
                .all(|(a, b)| a.id == b.id && equal_features(&a.features, &b.features))
            && self.events.len() == other.events.len()
            && self
                .events
                .iter()
                .zip(&other.events)
                .all(|(a, b)| a.matches(b))
            && self.candidates.len() == other.candidates.len()
            && self.candidates.iter().zip(&other.candidates).all(|(a, b)| {
                a.action == b.action
                    && a.enabled == b.enabled
                    && equal_features(&a.features, &b.features)
            })
    }

    pub(crate) fn validate(
        &self,
        schema: &ObservationSchema,
        space: &ActionSpace,
        live: bool,
    ) -> Result<()> {
        self.validate_spatial(schema.spatial.len())?;
        ensure(
            feature_row(&self.scalars, schema.scalars.len())
                && (self.entities.is_empty() || !schema.entities.is_empty())
                && (self.events.is_empty() || !schema.events.is_empty()),
            "structured observation disagrees with its schema",
        )?;
        let mut entities = HashSet::with_capacity(self.entities.len());
        ensure(
            self.entities.iter().all(|row| {
                feature_row(&row.features, schema.entities.len()) && entities.insert(row.id)
            }),
            "invalid or duplicate public entity row",
        )?;
        let incremental = !self.events.is_empty()
            && schema
                .conventions
                .iter()
                .any(|name| name == INCREMENTAL_EVENT_CONVENTION);
        ensure(
            self.events.iter().all(|row| {
                feature_row(&row.features, schema.events.len())
                    && row.stamp.is_some() == incremental
                    && (incremental
                        || (row.entity.is_none_or(|id| entities.contains(&id))
                            && row.context.iter().all(|id| entities.contains(id))))
            }),
            "invalid public event or entity reference",
        )?;
        if incremental {
            self.validate_event_order()?;
        }
        if let ActionSpace::Conditional(grammar) = space {
            self.validate_candidates(grammar, &entities, live)
        } else {
            ensure(
                self.candidates.is_empty(),
                "unexpected conditional candidates",
            )
        }
    }

    fn validate_candidates(
        &self,
        grammar: &ActionGrammar,
        entities: &HashSet<EntityId>,
        live: bool,
    ) -> Result<()> {
        let mut keys = HashSet::with_capacity(self.candidates.len());
        ensure(
            self.candidates.iter().all(|candidate| {
                keys.insert(candidate.action.in_region(self.primary_region))
                    && grammar.accepts(&candidate.action)
                    && feature_row(&candidate.features, grammar.candidate_features.len())
                    && match candidate.action.argument {
                        None => true,
                        Some(ActionArgument::Entity(id)) => entities.contains(&id),
                        Some(ActionArgument::Location { x, y }) => self
                            .spatial
                            .as_ref()
                            .is_some_and(|map| x < map.width && y < map.height),
                        Some(ActionArgument::RegionalLocation { region, x, y }) => self
                            .region_map(region)
                            .is_some_and(|map| x < map.width && y < map.height),
                    }
            }) && (!live || self.candidates.iter().any(|candidate| candidate.enabled)),
            "invalid or empty live conditional candidates",
        )
    }

    pub fn is_available(&self, action: &ConditionalAction) -> bool {
        self.candidates
            .iter()
            .any(|candidate| candidate.enabled && candidate.action == *action)
    }
}
