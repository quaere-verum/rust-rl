//! Stable wire records and allocation preflight for a restored episode store.

use super::{
    ActiveEpisode, Episode, EpisodeId, EpisodeIndex, EpisodeStore, MaintenanceStats, ResourceUsage,
    TemporalStep, TemporalStorage, add_usage, canonical_anchor_usage, cloned_action_bytes,
    cloned_observation_bytes, legacy_storage, reward_bytes,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        anchors::AnchorSet,
        ensure,
        env::{Observation, ObservationSpace},
    },
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::BTreeMap, mem::size_of, sync::Arc};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct EpisodeCapacities {
    observations: usize,
    steps: usize,
    anchors: usize,
}
impl EpisodeCapacities {
    fn of(episode: &Episode) -> Self {
        Self {
            observations: episode.observations.capacity(),
            steps: episode.steps.capacity(),
            anchors: episode.anchors.capacity(),
        }
    }
    fn bytes(self) -> Option<usize> {
        self.observations
            .checked_mul(size_of::<Observation>())
            .and_then(|bytes| {
                self.steps
                    .checked_mul(size_of::<TemporalStep>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                self.anchors
                    .checked_mul(size_of::<AnchorSet>())
                    .and_then(|next| bytes.checked_add(next))
            })
    }
}
impl Episode {
    fn canonical_usage(&self, capacities: EpisodeCapacities) -> Result<ResourceUsage> {
        ensure(
            capacities.observations >= self.observations.len()
                && capacities.steps >= self.steps.len()
                && capacities.anchors >= self.anchors.len(),
            "temporal allocation capacity is shorter than its data",
        )?;
        let descriptors = capacities.bytes().ok_or_else(|| {
            Error::Config("temporal allocation capacity exceeds byte address space".into())
        })?;
        let mut usage = ResourceUsage {
            resident_bytes: descriptors
                .checked_add(size_of::<Self>() + 128)
                .ok_or_else(|| {
                    Error::Config("temporal allocation metadata byte overflow".into())
                })?,
            ..Default::default()
        };
        for observation in &self.observations {
            usage = add_usage(
                usage,
                ResourceUsage {
                    resident_bytes: cloned_observation_bytes(observation),
                    ..Default::default()
                },
            )?;
        }
        for step in &self.steps {
            usage = add_usage(
                usage,
                ResourceUsage {
                    resident_bytes: cloned_action_bytes(&step.action) + reward_bytes(&step.reward),
                    ..Default::default()
                },
            )?;
        }
        if let Some(previous) = &self.previous {
            usage = add_usage(
                usage,
                ResourceUsage {
                    resident_bytes: cloned_action_bytes(&previous.action),
                    ..Default::default()
                },
            )?;
        }
        for set in &self.anchors {
            usage = add_usage(usage, canonical_anchor_usage(set))?;
        }
        Ok(usage)
    }
    fn restore_capacities(&mut self, capacities: EpisodeCapacities) -> Result<()> {
        for observation in &mut self.observations {
            match observation {
                Observation::Dense(observation) => {
                    observation.values = std::mem::take(&mut observation.values)
                        .into_boxed_slice()
                        .into_vec();
                    if let Some(mask) = &mut observation.action_mask {
                        *mask = std::mem::take(mask).into_boxed_slice().into_vec();
                    }
                }
                Observation::Discrete(observation) => {
                    if let Some(mask) = &mut observation.action_mask {
                        *mask = std::mem::take(mask).into_boxed_slice().into_vec();
                    }
                }
                Observation::Structured(frame) => Arc::make_mut(frame).canonicalize(),
                Observation::Graph(frame) => Arc::make_mut(frame).canonicalize(),
            }
        }
        fn canonical_action(action: &mut Action) {
            if let Action::Continuous(values) = action {
                *values = std::mem::take(values).into_boxed_slice().into_vec();
            }
        }
        for step in &mut self.steps {
            canonical_action(&mut step.action);
        }
        if let Some(previous) = &mut self.previous {
            canonical_action(&mut previous.action);
        }
        for set in &mut self.anchors {
            set.canonicalize();
        }
        fn restore<T>(values: &mut Vec<T>, capacity: usize) -> Result<()> {
            if values.capacity() == capacity {
                return Ok(());
            }
            let mut next = Vec::new();
            next.try_reserve_exact(capacity)
                .map_err(|e| Error::State(format!("cannot restore temporal capacity: {e}")))?;
            ensure(
                next.capacity() == capacity,
                "allocator cannot restore exact temporal capacity",
            )?;
            next.append(values);
            *values = next;
            Ok(())
        }
        restore(&mut self.observations, capacities.observations)?;
        restore(&mut self.steps, capacities.steps)?;
        restore(&mut self.anchors, capacities.anchors)
    }
}

struct EpisodeList<'a>(&'a BTreeMap<EpisodeId, Episode>);
impl Serialize for EpisodeList<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serialize_episodes(self.0, serializer)
    }
}
struct CapacityList<'a>(&'a BTreeMap<EpisodeId, Episode>);
impl Serialize for CapacityList<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (id, episode) in self.0 {
            map.serialize_entry(id, &EpisodeCapacities::of(episode))?;
        }
        map.end()
    }
}
impl Serialize for EpisodeStore {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Record<'a> {
            version: u32,
            capacity: usize,
            #[serde(rename = "width")]
            observation_space: &'a ObservationSpace,
            space: &'a ActionSpace,
            len: usize,
            highest_episode_id: Option<EpisodeId>,
            episodes: EpisodeList<'a>,
            active: &'a BTreeMap<EpisodeId, ActiveEpisode>,
            #[serde(skip_serializing_if = "legacy_storage")]
            storage: &'a TemporalStorage,
            #[serde(skip_serializing_if = "Option::is_none")]
            latest_terminal: Option<EpisodeId>,
            #[serde(skip_serializing_if = "Option::is_none")]
            allocation_capacities: Option<CapacityList<'a>>,
        }
        Record {
            version: self.version,
            capacity: self.capacity,
            observation_space: &self.observation_space,
            space: &self.space,
            len: self.len,
            highest_episode_id: self.highest_episode_id,
            episodes: EpisodeList(&self.episodes),
            active: &self.active,
            storage: &self.storage,
            latest_terminal: self.latest_terminal,
            allocation_capacities: (!legacy_storage(&self.storage))
                .then_some(CapacityList(&self.episodes)),
        }
        .serialize(serializer)
    }
}

fn serialize_episodes<S: Serializer>(
    episodes: &BTreeMap<EpisodeId, Episode>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_seq(episodes.values())
}

impl<'de> Deserialize<'de> for EpisodeStore {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Record {
            version: u32,
            capacity: usize,
            #[serde(rename = "width")]
            observation_space: ObservationSpace,
            space: ActionSpace,
            len: usize,
            highest_episode_id: Option<EpisodeId>,
            episodes: Vec<Episode>,
            #[serde(default)]
            episode_behaviors: BTreeMap<EpisodeId, serde::de::IgnoredAny>,
            active: BTreeMap<EpisodeId, ActiveEpisode>,
            #[serde(default)]
            storage: TemporalStorage,
            #[serde(default)]
            latest_terminal: Option<EpisodeId>,
            #[serde(default)]
            allocation_capacities: Option<BTreeMap<EpisodeId, EpisodeCapacities>>,
        }
        let record = Record::deserialize(deserializer)?;
        if !record.episode_behaviors.is_empty() {
            return Err(serde::de::Error::custom(
                "delegated episode bindings are unsupported",
            ));
        }
        if legacy_storage(&record.storage) {
            if record.allocation_capacities.is_some() {
                return Err(serde::de::Error::custom(
                    "legacy temporal store contains v2 allocation metadata",
                ));
            }
        } else {
            let capacities = record.allocation_capacities.as_ref().ok_or_else(|| {
                serde::de::Error::custom("v2 temporal store lacks allocation capacities")
            })?;
            if capacities.len() != record.episodes.len() {
                return Err(serde::de::Error::custom(
                    "temporal allocation capacity identities differ",
                ));
            }
            let limits = match &record.storage {
                TemporalStorage::FullPrefix(limits) => limits,
                TemporalStorage::Segmented(config) => &config.limits,
                _ => unreachable!(),
            };
            limits
                .validate(record.capacity)
                .map_err(serde::de::Error::custom)?;
            let mut projected = ResourceUsage::default();
            for episode in &record.episodes {
                episode
                    .validate_in(&record.observation_space, &record.space)
                    .map_err(serde::de::Error::custom)?;
                let capacity = capacities.get(&episode.id).ok_or_else(|| {
                    serde::de::Error::custom("missing temporal allocation identity")
                })?;
                projected = add_usage(
                    projected,
                    episode
                        .canonical_usage(*capacity)
                        .map_err(serde::de::Error::custom)?,
                )
                .map_err(serde::de::Error::custom)?;
            }
            if !limits.contains(record.len, projected) {
                return Err(serde::de::Error::custom(
                    "temporal restoration exceeds resource budget before allocation",
                ));
            }
        }
        let mut episodes = BTreeMap::new();
        let mut index = EpisodeIndex::default();
        let mut previous = None;
        for mut episode in record.episodes {
            if previous.is_some_and(|id| id >= episode.id)
                || index.total().checked_add(episode.learning_len()).is_none()
            {
                return Err(serde::de::Error::custom(
                    "invalid temporal episode identity order or count",
                ));
            }
            previous = Some(episode.id);
            if let Some(capacities) = &record.allocation_capacities {
                episode
                    .restore_capacities(capacities[&episode.id])
                    .map_err(serde::de::Error::custom)?;
            }
            index.set(episode.id, episode.learning_len());
            episodes.insert(episode.id, episode);
        }
        let mut store = Self {
            version: record.version,
            capacity: record.capacity,
            observation_space: record.observation_space,
            space: record.space,
            len: record.len,
            highest_episode_id: record.highest_episode_id,
            episodes,
            active: record.active,
            index,
            storage: record.storage,
            latest_terminal: record.latest_terminal,
            usage: ResourceUsage::default(),
            maintenance: MaintenanceStats::default(),
        };
        store.rebuild_usage().map_err(serde::de::Error::custom)?;
        store.validate().map_err(serde::de::Error::custom)?;
        Ok(store)
    }
}
