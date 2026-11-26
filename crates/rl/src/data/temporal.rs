//! Owned chronological tapes and exact or explicitly anchored sequence sampling.
//!
//! Capacity counts retained transitions. At most `capacity` episode identities may
//! be active simultaneously, bounding empty tapes and discarded-episode bookkeeping
//! as well. The legacy mode evicts complete episodes. Resource-limited modes
//! preserve absolute clocks and return an error rather than ignore live suffixes.
//! IDs must increase at `begin_episode`, including after abandonment or
//! eviction. Evicting an active tape discards its entire prefix; subsequent records
//! for that identity are ignored until its actual end or explicit abandonment.
//!
//! This is factual experience storage: callers submit only environment-produced
//! observations, executed actions and physical rewards. Predictive-model rollouts
//! do not implement a second ingestion path here. Search supervision may be joined
//! to an existing factual decision without presenting imagined transitions as data.

pub use crate::core::temporal::{BehaviorMetadata, TemporalSink, TemporalTransition};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        anchors::{AnchorSchema, AnchorSet, RepresentationId},
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace},
        reward::Reward,
        stateful::PreviousStep,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, mem::size_of, sync::Arc, time::Instant};

mod sampling;
mod sequence;
mod validation;
mod wire;
pub use sampling::{SequenceSampler, sample_at};
pub(crate) use sequence::actions_equal_bits;
pub use sequence::{
    NStepTarget, PaddedSequenceLayout, SequenceBatch, SequenceConfig, SequenceSample,
};

#[path = "temporal_index.rs"]
mod temporal_index;
use temporal_index::EpisodeIndex;

const VERSION: u32 = 1;
pub type EpisodeId = u64;

fn zero(value: &usize) -> bool {
    *value == 0
}
fn reset(origin: &SequenceOrigin) -> bool {
    matches!(origin, SequenceOrigin::Reset)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequenceOrigin {
    #[default]
    Reset,
    Anchored(AnchorSet),
    /// A factual learning window without a state reconstruction prefix.
    Stateless,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_transitions: usize,
    pub max_resident_bytes: usize,
    pub max_anchor_bytes: usize,
    pub max_anchors: usize,
    pub max_anchor_payload_bytes: usize,
    pub max_dimension_rank: usize,
}
impl ResourceLimits {
    fn validate(&self, capacity: usize) -> Result<()> {
        ensure(
            self.max_transitions == capacity
                && self.max_resident_bytes > 0
                && self.max_anchor_payload_bytes > 0
                && self.max_anchor_payload_bytes <= crate::core::anchors::MAX_ANCHOR_PAYLOAD_BYTES
                && self.max_dimension_rank > 0
                && self.max_dimension_rank <= crate::core::anchors::MAX_ANCHOR_DIMENSIONS,
            "invalid temporal resource limits",
        )
    }
    fn contains(&self, len: usize, usage: ResourceUsage) -> bool {
        len <= self.max_transitions
            && usage.resident_bytes <= self.max_resident_bytes
            && usage.anchor_bytes <= self.max_anchor_bytes
            && usage.anchors <= self.max_anchors
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentedConfig {
    pub segment_steps: usize,
    pub minimum_context: usize,
    pub required_representations: Vec<RepresentationId>,
    pub limits: ResourceLimits,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalStorage {
    #[default]
    ExactEpisodes,
    FullPrefix(ResourceLimits),
    Segmented(SegmentedConfig),
}
fn legacy_storage(mode: &TemporalStorage) -> bool {
    matches!(mode, TemporalStorage::ExactEpisodes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorRequirement {
    pub representation_id: RepresentationId,
    pub schema: Option<AnchorSchema>,
    pub model_identity: Option<[u8; 32]>,
    pub require_exact: bool,
    pub min_generation: Option<u64>,
    pub max_generation: Option<u64>,
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnchorRequirements {
    pub required: Vec<AnchorRequirement>,
}
impl AnchorRequirements {
    fn validate(&self) -> Result<()> {
        ensure(
            self.required.len() <= crate::core::anchors::MAX_ANCHOR_KEYS,
            "too many required temporal representations",
        )?;
        for (index, requirement) in self.required.iter().enumerate() {
            ensure(
                self.required[..index]
                    .iter()
                    .all(|old| old.representation_id != requirement.representation_id),
                "duplicate required temporal representation",
            )?;
            if let Some(schema) = &requirement.schema {
                schema.validate()?;
            }
        }
        Ok(())
    }
    fn accepts(&self, set: &AnchorSet) -> bool {
        self.required.iter().all(|requirement| {
            set.anchors
                .get(&requirement.representation_id)
                .is_some_and(|anchor| {
                    requirement
                        .schema
                        .as_ref()
                        .is_none_or(|schema| *schema == anchor.schema)
                        && requirement
                            .model_identity
                            .is_none_or(|identity| identity == anchor.model_identity)
                        && (!requirement.require_exact || anchor.exact_ancestry)
                        && requirement
                            .min_generation
                            .is_none_or(|generation| anchor.generation >= generation)
                        && requirement
                            .max_generation
                            .is_none_or(|generation| anchor.generation <= generation)
                })
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampleLimits {
    pub max_sample_tokens: usize,
    pub max_batch_tokens: usize,
    pub max_sample_payload_bytes: usize,
    pub max_batch_payload_bytes: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResourceUsage {
    /// Owned Vec capacities plus fixed record/index allowances; excludes allocator overhead.
    pub resident_bytes: usize,
    pub anchor_bytes: usize,
    pub anchors: usize,
}
#[derive(Debug, Clone, Copy, Default)]
pub struct MaintenanceStats {
    pub cuts: u64,
    pub trimmed_transitions: u64,
    pub moved_descriptors: u64,
    pub seconds: f64,
}

fn action_bytes(action: &Action) -> usize {
    match action {
        Action::Continuous(values) => values.capacity() * size_of::<f32>(),
        Action::Conditional(_) => {
            size_of::<crate::core::action::ConditionalAction>() + 2 * size_of::<usize>()
        }
        _ => 0,
    }
}
fn observation_bytes(observation: &Observation) -> usize {
    match observation {
        Observation::Dense(dense) => {
            dense.values.capacity() * size_of::<f32>()
                + dense.action_mask.as_ref().map_or(0, Vec::capacity)
                + dense
                    .public_data
                    .as_ref()
                    .map_or(0, |data| data.retained_bytes())
        }
        Observation::Discrete(discrete) => discrete.action_mask.as_ref().map_or(0, Vec::capacity),
        Observation::Structured(frame) => frame.payload_bytes(true),
        Observation::Graph(frame) => frame.payload_bytes(true),
    }
}
pub(super) fn cloned_observation_bytes(observation: &Observation) -> usize {
    match observation {
        Observation::Dense(dense) => {
            dense.values.len() * size_of::<f32>()
                + dense.action_mask.as_ref().map_or(0, Vec::len)
                + dense
                    .public_data
                    .as_ref()
                    .map_or(0, |data| data.retained_bytes())
        }
        Observation::Discrete(discrete) => discrete.action_mask.as_ref().map_or(0, Vec::len),
        Observation::Structured(frame) => frame.payload_bytes(false),
        Observation::Graph(frame) => frame.payload_bytes(false),
    }
}
pub(super) fn reward_bytes(reward: &Reward) -> usize {
    match reward {
        Reward::One(_) => 0,
        Reward::Many(values) => size_of_val(values.as_ref()) + 2 * size_of::<usize>(),
    }
}
pub(super) fn cloned_action_bytes(action: &Action) -> usize {
    match action {
        Action::Continuous(values) => values.len() * size_of::<f32>(),
        Action::Conditional(_) => {
            size_of::<crate::core::action::ConditionalAction>() + 2 * size_of::<usize>()
        }
        _ => 0,
    }
}
fn clone_observation(observation: &Observation) -> Observation {
    match observation {
        Observation::Dense(dense) => Observation::Dense(crate::core::env::DenseObservation {
            values: dense.values.clone().into_boxed_slice().into_vec(),
            action_mask: dense
                .action_mask
                .as_ref()
                .map(|mask| mask.clone().into_boxed_slice().into_vec()),
            public_data: dense.public_data.clone(),
        }),
        Observation::Discrete(discrete) => Observation::discrete(
            discrete.symbol,
            discrete
                .action_mask
                .as_ref()
                .map(|mask| mask.clone().into_boxed_slice().into_vec()),
        ),
        Observation::Structured(frame)
            if frame.payload_bytes(true) != frame.payload_bytes(false) =>
        {
            let mut compact = frame.as_ref().clone();
            compact.canonicalize();
            Observation::Structured(Arc::new(compact))
        }
        Observation::Graph(frame) if frame.payload_bytes(true) != frame.payload_bytes(false) => {
            let mut compact = frame.as_ref().clone();
            compact.canonicalize();
            Observation::Graph(Arc::new(compact))
        }
        Observation::Structured(_) | Observation::Graph(_) => observation.clone(),
    }
}
fn clone_action(action: &Action) -> Action {
    match action {
        Action::Continuous(values) => {
            Action::Continuous(values.clone().into_boxed_slice().into_vec())
        }
        _ => action.clone(),
    }
}
fn anchor_usage(set: &AnchorSet) -> ResourceUsage {
    ResourceUsage {
        resident_bytes: set
            .anchors
            .values()
            .map(|anchor| {
                size_of::<(RepresentationId, crate::core::anchors::StateAnchor)>()
                    + 32
                    + anchor.payload.capacity()
                    + anchor.schema.dimensions.capacity() * size_of::<usize>()
            })
            .sum(),
        anchor_bytes: set
            .anchors
            .values()
            .map(|anchor| anchor.payload.capacity())
            .sum(),
        anchors: set.anchors.len(),
    }
}
fn canonical_anchor_usage(set: &AnchorSet) -> ResourceUsage {
    ResourceUsage {
        resident_bytes: set
            .anchors
            .values()
            .map(|anchor| {
                size_of::<(RepresentationId, crate::core::anchors::StateAnchor)>()
                    + 32
                    + anchor.payload.len()
                    + anchor.schema.dimensions.len() * size_of::<usize>()
            })
            .sum(),
        anchor_bytes: set
            .anchors
            .values()
            .map(|anchor| anchor.payload.len())
            .sum(),
        anchors: set.anchors.len(),
    }
}
fn add_usage(left: ResourceUsage, right: ResourceUsage) -> Result<ResourceUsage> {
    Ok(ResourceUsage {
        resident_bytes: left
            .resident_bytes
            .checked_add(right.resident_bytes)
            .ok_or_else(|| Error::State("temporal resident byte overflow".into()))?,
        anchor_bytes: left
            .anchor_bytes
            .checked_add(right.anchor_bytes)
            .ok_or_else(|| Error::State("temporal anchor byte overflow".into()))?,
        anchors: left
            .anchors
            .checked_add(right.anchors)
            .ok_or_else(|| Error::State("temporal anchor count overflow".into()))?,
    })
}
fn subtract_usage(left: &mut ResourceUsage, right: ResourceUsage) {
    left.resident_bytes -= right.resident_bytes;
    left.anchor_bytes -= right.anchor_bytes;
    left.anchors -= right.anchors;
}
impl SampleLimits {
    pub(crate) fn has_unbounded_samples(&self) -> bool {
        self.max_sample_tokens == usize::MAX && self.max_sample_payload_bytes == usize::MAX
    }
}
impl Default for SampleLimits {
    fn default() -> Self {
        Self {
            max_sample_tokens: usize::MAX,
            max_batch_tokens: usize::MAX,
            max_sample_payload_bytes: usize::MAX,
            max_batch_payload_bytes: usize::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalStep {
    pub action: Action,
    pub reward: Reward,
    pub terminated: bool,
    pub truncated: bool,
    pub behavior: BehaviorMetadata,
}
impl TemporalStep {
    pub fn ended(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// An immutable retained slice. `base_timestep` is absolute; a nonzero base has
/// actual previous input and anchors. Cloning creates an independent owned tape.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub observations: Vec<Observation>,
    pub steps: Vec<TemporalStep>,
    #[serde(default, skip_serializing_if = "zero")]
    pub base_timestep: usize,
    /// Absolute transition at which this trajectory started collecting experience.
    #[serde(default, skip_serializing_if = "zero")]
    pub learning_start: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<PreviousStep>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<AnchorSet>,
}
impl Clone for Episode {
    fn clone(&self) -> Self {
        let mut observations = Vec::with_capacity(self.observations.capacity());
        observations.extend(self.observations.iter().map(clone_observation));
        let mut steps = Vec::with_capacity(self.steps.capacity());
        steps.extend(self.steps.iter().map(|step| TemporalStep {
            action: clone_action(&step.action),
            reward: step.reward.clone(),
            terminated: step.terminated,
            truncated: step.truncated,
            behavior: step.behavior.clone(),
        }));
        let mut anchors = Vec::with_capacity(self.anchors.capacity());
        anchors.extend(self.anchors.iter().map(|set| {
            let mut set = set.clone();
            set.canonicalize();
            set
        }));
        Self {
            id: self.id,
            observations,
            steps,
            anchors,
            base_timestep: self.base_timestep,
            learning_start: self.learning_start,
            previous: self.previous.as_ref().map(|previous| PreviousStep {
                action: clone_action(&previous.action),
                reward: previous.reward,
            }),
        }
    }
}

impl Episode {
    pub fn end_timestep(&self) -> usize {
        self.base_timestep + self.steps.len()
    }
    fn context_len(&self) -> usize {
        self.learning_start.saturating_sub(self.base_timestep)
    }
    fn learning_len(&self) -> usize {
        self.steps.len().saturating_sub(self.context_len())
    }
    fn usage(&self) -> Result<ResourceUsage> {
        let mut usage = ResourceUsage {
            resident_bytes: size_of::<Self>()
                + 128
                + self.observations.capacity() * size_of::<Observation>()
                + self.steps.capacity() * size_of::<TemporalStep>()
                + self.anchors.capacity() * size_of::<AnchorSet>()
                + self
                    .observations
                    .iter()
                    .map(observation_bytes)
                    .sum::<usize>()
                + self
                    .steps
                    .iter()
                    .map(|step| action_bytes(&step.action) + reward_bytes(&step.reward))
                    .sum::<usize>()
                + self
                    .previous
                    .as_ref()
                    .map_or(0, |p| action_bytes(&p.action)),
            ..Default::default()
        };
        for set in &self.anchors {
            usage = add_usage(usage, anchor_usage(set))?;
        }
        Ok(usage)
    }
    pub fn ended(&self) -> bool {
        self.steps.last().is_some_and(TemporalStep::ended)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActiveEpisode {
    next_timestep: usize,
    ignored: bool,
}

#[derive(Debug, Clone)]
pub struct EpisodeStore {
    version: u32,
    capacity: usize,
    observation_space: ObservationSpace,
    space: ActionSpace,
    len: usize,
    highest_episode_id: Option<EpisodeId>,
    episodes: BTreeMap<EpisodeId, Episode>,
    active: BTreeMap<EpisodeId, ActiveEpisode>,
    index: EpisodeIndex,
    storage: TemporalStorage,
    latest_terminal: Option<EpisodeId>,
    usage: ResourceUsage,
    maintenance: MaintenanceStats,
}

impl EpisodeStore {
    fn cut_savings(episode: &Episode, from: usize, to: usize) -> Result<ResourceUsage> {
        let previous_bytes = if from == 0 {
            episode
                .previous
                .as_ref()
                .map_or(0, |p| action_bytes(&p.action))
        } else {
            action_bytes(&episode.steps[from - 1].action)
        };
        let mut savings = ResourceUsage {
            resident_bytes: previous_bytes
                + episode.observations[from..to]
                    .iter()
                    .map(observation_bytes)
                    .sum::<usize>()
                + episode.steps[from..to]
                    .iter()
                    .map(|step| action_bytes(&step.action) + reward_bytes(&step.reward))
                    .sum::<usize>()
                - action_bytes(&episode.steps[to - 1].action),
            ..Default::default()
        };
        for set in &episode.anchors {
            if set.observation_index() >= episode.base_timestep + from
                && set.observation_index() < episode.base_timestep + to
            {
                savings = add_usage(savings, anchor_usage(set))?;
            }
        }
        Ok(savings)
    }

    fn push_bounded(
        &mut self,
        t: &TemporalTransition,
        mut anchors: Option<AnchorSet>,
    ) -> Result<()> {
        t.validate_in(&self.observation_space, &self.space)?;
        let active = self
            .active
            .get(&t.episode_id)
            .ok_or_else(|| Error::State("temporal transition has no active episode".into()))?;
        ensure(
            !active.ignored && active.next_timestep == t.timestep,
            "temporal timestep differs",
        )?;
        let next = t
            .timestep
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal timestep overflow".into()))?;
        let episode = self
            .episode(t.episode_id)
            .expect("bounded active episode is retained");
        ensure(
            episode
                .observations
                .last()
                .is_some_and(|last| last.matches(&t.observation)),
            "temporal successor observation is not contiguous",
        )?;
        if let Some(set) = &anchors {
            self.check_anchors(set)?;
            ensure(
                set.episode_id() == t.episode_id && set.observation_index() == next,
                "append anchor must precede the actual successor observation",
            )?;
            ensure(
                episode
                    .anchors
                    .last()
                    .is_none_or(|old| old.observation_index() < next),
                "duplicate temporal anchor boundary",
            )?;
        }
        let increment = match &self.storage {
            TemporalStorage::Segmented(config) => config.segment_steps,
            _ => 64,
        };
        fn planned_capacity(
            capacity: usize,
            needed: usize,
            increment: usize,
        ) -> Result<Option<usize>> {
            if needed <= capacity {
                return Ok(None);
            }
            let capacity = needed.checked_add(increment - 1).ok_or_else(|| {
                Error::Config("temporal resource limit: descriptor capacity overflow".into())
            })?;
            Ok(Some(capacity))
        }
        let observation_capacity = planned_capacity(
            episode.observations.capacity(),
            episode.observations.len() + 1,
            increment,
        )?;
        let step_capacity =
            planned_capacity(episode.steps.capacity(), episode.steps.len() + 1, increment)?;
        let anchor_capacity = planned_capacity(
            episode.anchors.capacity(),
            episode.anchors.len() + usize::from(anchors.is_some()),
            1,
        )?;
        let descriptor_bytes = observation_capacity
            .unwrap_or(0)
            .checked_mul(size_of::<Observation>())
            .and_then(|bytes| {
                step_capacity
                    .unwrap_or(0)
                    .checked_mul(size_of::<TemporalStep>())
                    .and_then(|next| bytes.checked_add(next))
            })
            .and_then(|bytes| {
                anchor_capacity
                    .unwrap_or(0)
                    .checked_mul(size_of::<AnchorSet>())
                    .and_then(|next| bytes.checked_add(next))
            });
        let mut borrowed_incoming = ResourceUsage {
            resident_bytes: cloned_observation_bytes(&t.next_observation)
                + cloned_action_bytes(&t.action)
                + reward_bytes(&t.reward),
            ..Default::default()
        };
        if let Some(set) = &anchors {
            borrowed_incoming = add_usage(borrowed_incoming, canonical_anchor_usage(set))?;
        }
        let planned_peak = descriptor_bytes
            .and_then(|bytes| self.usage.resident_bytes.checked_add(bytes))
            .and_then(|bytes| bytes.checked_add(borrowed_incoming.resident_bytes));
        let limits = self.limits().expect("bounded append mode");
        // Check requested capacity before any descriptor allocation or nested
        // observation/action clone. Overflow itself is an over-budget request.
        ensure(
            planned_peak.is_some_and(|bytes| bytes <= limits.max_resident_bytes),
            "temporal resource limit: append staging reserve",
        )?;
        if let Some(set) = &mut anchors {
            set.canonicalize();
        }
        fn replacement<T>(capacity: Option<usize>) -> Result<Option<Vec<T>>> {
            let Some(capacity) = capacity else {
                return Ok(None);
            };
            let mut values = Vec::new();
            values
                .try_reserve_exact(capacity)
                .map_err(|e| Error::State(format!("cannot reserve temporal descriptors: {e}")))?;
            Ok(Some(values))
        }
        let mut observation_buffer = replacement::<Observation>(observation_capacity)?;
        let mut step_buffer = replacement::<TemporalStep>(step_capacity)?;
        let mut anchor_buffer = replacement::<AnchorSet>(anchor_capacity)?;
        let next_observation = clone_observation(&t.next_observation);
        let step = TemporalStep {
            action: clone_action(&t.action),
            reward: t.reward.clone(),
            terminated: t.terminated,
            truncated: t.truncated,
            behavior: t.behavior.clone(),
        };
        let mut incoming = ResourceUsage {
            resident_bytes: observation_bytes(&next_observation)
                + action_bytes(&step.action)
                + reward_bytes(&step.reward),
            ..Default::default()
        };
        if let Some(set) = &anchors {
            incoming = add_usage(incoming, anchor_usage(set))?;
        }
        let replacement_bytes = observation_buffer
            .as_ref()
            .map_or(0, |v| v.capacity() * size_of::<Observation>())
            + step_buffer
                .as_ref()
                .map_or(0, |v| v.capacity() * size_of::<TemporalStep>())
            + anchor_buffer
                .as_ref()
                .map_or(0, |v| v.capacity() * size_of::<AnchorSet>());
        let old_replacement_bytes = if observation_buffer.is_some() {
            episode.observations.capacity() * size_of::<Observation>()
        } else {
            0
        } + if step_buffer.is_some() {
            episode.steps.capacity() * size_of::<TemporalStep>()
        } else {
            0
        } + if anchor_buffer.is_some() {
            episode.anchors.capacity() * size_of::<AnchorSet>()
        } else {
            0
        };
        let mut usage = add_usage(self.usage, incoming)?;
        let peak = usage
            .resident_bytes
            .checked_add(replacement_bytes)
            .ok_or_else(|| Error::State("temporal staging byte overflow".into()))?;
        let limits = self.limits().expect("bounded append mode");
        ensure(
            peak <= limits.max_resident_bytes,
            "temporal resource limit: append staging reserve",
        )?;
        usage.resident_bytes = peak - old_replacement_bytes;
        let mut length = self
            .len
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal transition count overflow".into()))?;
        let mut cuts = BTreeMap::<EpisodeId, usize>::new();
        let mut evictions = Vec::new();
        while !limits.contains(length, usage) {
            let mut best = None;
            if let TemporalStorage::Segmented(config) = &self.storage {
                for (&id, tape) in &self.episodes {
                    if evictions.contains(&id) {
                        continue;
                    }
                    let from = cuts.get(&id).copied().unwrap_or(0);
                    let retained_len = tape.steps.len() + usize::from(id == t.episode_id) - from;
                    let end = tape.end_timestep() + usize::from(id == t.episode_id);
                    let boundary = tape.anchors.iter().find(|set| {
                        set.observation_index() > tape.base_timestep + from
                            && end - set.observation_index() >= config.minimum_context
                    });
                    if let Some(set) = boundary
                        && best.is_none_or(|(_, _, size)| retained_len > size)
                    {
                        best = Some((
                            id,
                            set.observation_index() - tape.base_timestep,
                            retained_len,
                        ));
                    }
                }
            }
            if let Some((id, to, _)) = best {
                let from = cuts.get(&id).copied().unwrap_or(0);
                let tape = self.episode(id).expect("planned cut tape");
                subtract_usage(&mut usage, Self::cut_savings(tape, from, to)?);
                let context = tape.context_len();
                length -= to.saturating_sub(context) - from.saturating_sub(context);
                cuts.insert(id, to);
            } else {
                let evictable = |id: &&EpisodeId, tape: &&Episode| {
                    tape.ended() && **id != t.episode_id && !evictions.contains(id)
                };
                // Prefer other completed data. A prior terminal tail is a final
                // fallback when retaining it would prevent all further collection.
                let candidate = self
                    .episodes
                    .iter()
                    .find(|(id, tape)| evictable(id, tape) && Some(**id) != self.latest_terminal)
                    .or_else(|| self.episodes.iter().find(|(id, tape)| evictable(id, tape)));
                let Some((&id, tape)) = candidate else {
                    return Err(Error::State("temporal resource limit: no anchored cut preserving minimum context or evictable completed tape".into()));
                };
                let from = cuts.remove(&id).unwrap_or(0);
                let mut remaining = tape.usage()?;
                if from > 0 {
                    subtract_usage(&mut remaining, Self::cut_savings(tape, 0, from)?);
                }
                subtract_usage(&mut usage, remaining);
                length -= tape.learning_len() - from.saturating_sub(tape.context_len());
                evictions.push(id);
            }
        }
        let clock = Instant::now();
        for id in evictions {
            self.episodes.remove(&id);
            self.index.remove(id);
            if self.latest_terminal == Some(id) {
                self.latest_terminal = None;
            }
        }
        for (id, count) in cuts {
            let tape = self.episodes.get_mut(&id).expect("planned retained tape");
            let learned = count.saturating_sub(tape.context_len());
            let previous = tape
                .steps
                .drain(..count)
                .next_back()
                .expect("nonempty temporal cut");
            tape.previous = Some(PreviousStep {
                action: previous.action,
                reward: previous.reward.total(),
            });
            tape.observations.drain(..count);
            tape.base_timestep += count;
            tape.anchors
                .retain(|set| set.observation_index() >= tape.base_timestep);
            self.index.set(id, tape.learning_len());
            self.maintenance.cuts += 1;
            self.maintenance.trimmed_transitions += learned as u64;
            self.maintenance.moved_descriptors +=
                (tape.steps.len() + tape.observations.len() + tape.anchors.len()) as u64;
        }
        let tape = self
            .episodes
            .get_mut(&t.episode_id)
            .expect("appending tape was preserved");
        if let Some(mut buffer) = observation_buffer.take() {
            buffer.append(&mut tape.observations);
            tape.observations = buffer;
        }
        if let Some(mut buffer) = step_buffer.take() {
            buffer.append(&mut tape.steps);
            tape.steps = buffer;
        }
        if let Some(mut buffer) = anchor_buffer.take() {
            buffer.append(&mut tape.anchors);
            tape.anchors = buffer;
        }
        tape.observations.push(next_observation);
        tape.steps.push(step);
        if let Some(set) = anchors {
            tape.anchors.push(set);
        }
        self.index.set(t.episode_id, tape.learning_len());
        self.len = length;
        self.usage = usage;
        if t.ended() {
            self.active.remove(&t.episode_id);
            self.latest_terminal = Some(t.episode_id);
        } else {
            self.active
                .get_mut(&t.episode_id)
                .expect("validated active tape")
                .next_timestep = next;
        }
        self.maintenance.seconds += clock.elapsed().as_secs_f64();
        Ok(())
    }
    pub fn new(
        capacity: usize,
        observations: impl Into<ObservationSpace>,
        space: ActionSpace,
    ) -> Result<Self> {
        ensure(capacity > 0, "temporal capacity must be positive")?;
        let observation_space = observations.into();
        observation_space.validate()?;
        space.validate()?;
        Ok(Self {
            version: VERSION,
            capacity,
            observation_space,
            space,
            len: 0,
            highest_episode_id: None,
            episodes: BTreeMap::new(),
            active: BTreeMap::new(),
            index: EpisodeIndex::default(),
            storage: TemporalStorage::ExactEpisodes,
            latest_terminal: None,
            usage: ResourceUsage::default(),
            maintenance: MaintenanceStats::default(),
        })
    }
    pub fn with_storage(
        capacity: usize,
        observations: impl Into<ObservationSpace>,
        space: ActionSpace,
        storage: TemporalStorage,
    ) -> Result<Self> {
        let mut store = Self::new(capacity, observations, space)?;
        store.storage = storage;
        store.version = if legacy_storage(&store.storage) { 1 } else { 2 };
        store.validate()?;
        Ok(store)
    }
    pub fn storage(&self) -> &TemporalStorage {
        &self.storage
    }
    pub fn resource_usage(&self) -> ResourceUsage {
        self.usage
    }
    pub fn maintenance_stats(&self) -> MaintenanceStats {
        self.maintenance
    }
    fn limits(&self) -> Option<&ResourceLimits> {
        match &self.storage {
            TemporalStorage::ExactEpisodes => None,
            TemporalStorage::FullPrefix(limits) => Some(limits),
            TemporalStorage::Segmented(config) => Some(&config.limits),
        }
    }
    fn rebuild_usage(&mut self) -> Result<()> {
        self.usage = ResourceUsage::default();
        if self.limits().is_some() {
            for episode in self.episodes.values() {
                self.usage = add_usage(self.usage, episode.usage()?)?;
            }
        }
        Ok(())
    }
    pub fn episode_clock(&self, id: EpisodeId) -> Option<usize> {
        self.episode(id)
            .map(Episode::end_timestep)
            .or_else(|| self.active_timestep(id))
    }
    pub fn push_with_anchors(
        &mut self,
        transition: &TemporalTransition,
        anchors: Option<AnchorSet>,
    ) -> Result<()> {
        if legacy_storage(&self.storage) {
            ensure(
                anchors.is_none(),
                "legacy temporal replay does not accept anchors",
            )?;
            return self.push(transition);
        }
        self.push_bounded(transition, anchors)
    }

    pub fn install_anchors(&mut self, mut anchors: AnchorSet) -> Result<()> {
        self.check_anchors(&anchors)?;
        let id = anchors.episode_id();
        let episode = self
            .episode(id)
            .ok_or_else(|| Error::State("anchor episode is not retained".into()))?;
        let boundary = anchors.observation_index();
        ensure(
            boundary >= episode.base_timestep && boundary <= episode.end_timestep(),
            "anchor boundary is not retained",
        )?;
        let position = episode
            .anchors
            .partition_point(|set| set.observation_index() < boundary);
        ensure(
            episode
                .anchors
                .get(position)
                .is_none_or(|set| set.observation_index() != boundary),
            "duplicate temporal anchor boundary",
        )?;
        let requested_capacity = episode.anchors.len().checked_add(1).ok_or_else(|| {
            Error::Config("temporal resource limit: anchor capacity overflow".into())
        })?;
        let descriptor_bytes = requested_capacity
            .checked_mul(size_of::<AnchorSet>())
            .ok_or_else(|| {
                Error::Config("temporal resource limit: anchor descriptor bytes".into())
            })?;
        let mut usage = add_usage(self.usage, canonical_anchor_usage(&anchors))?;
        let peak = usage
            .resident_bytes
            .checked_add(descriptor_bytes)
            .ok_or_else(|| Error::State("anchor resident overflow".into()))?;
        usage.resident_bytes = usage.resident_bytes + descriptor_bytes
            - episode.anchors.capacity() * size_of::<AnchorSet>();
        let limits = self.limits().expect("checked bounded anchor mode");
        ensure(
            limits.contains(self.len, usage) && peak <= limits.max_resident_bytes,
            "temporal resource limit: anchor installation",
        )?;
        anchors.canonicalize();
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(requested_capacity)
            .map_err(|e| Error::State(format!("cannot reserve temporal anchors: {e}")))?;
        let actual_bytes = replacement.capacity() * size_of::<AnchorSet>();
        ensure(
            peak - descriptor_bytes + actual_bytes <= limits.max_resident_bytes,
            "temporal resource limit: allocated anchor capacity",
        )?;
        usage.resident_bytes = usage.resident_bytes - descriptor_bytes + actual_bytes;
        let episode = self
            .episodes
            .get_mut(&id)
            .expect("validated anchor episode");
        replacement.append(&mut episode.anchors);
        replacement.insert(position, anchors);
        episode.anchors = replacement;
        self.usage = usage;
        Ok(())
    }
    fn check_anchors(&self, anchors: &AnchorSet) -> Result<()> {
        anchors.validate()?;
        let limits = self
            .limits()
            .ok_or_else(|| Error::State("legacy temporal replay does not accept anchors".into()))?;
        for anchor in anchors.anchors.values() {
            ensure(
                anchor.payload.len() <= limits.max_anchor_payload_bytes
                    && anchor.schema.dimensions.len() <= limits.max_dimension_rank,
                "temporal anchor exceeds payload or dimension limit",
            )?;
        }
        if let TemporalStorage::Segmented(config) = &self.storage {
            ensure(
                config
                    .required_representations
                    .iter()
                    .all(|key| anchors.anchors.contains_key(key)),
                "temporal boundary lacks required representation",
            )?;
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn dense_observation_dim(&self) -> Option<usize> {
        self.observation_space.dense_width()
    }
    pub fn observation_space(&self) -> &ObservationSpace {
        &self.observation_space
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    pub fn highest_episode_id(&self) -> Option<EpisodeId> {
        self.highest_episode_id
    }
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
    pub fn ignored_count(&self) -> usize {
        self.active.values().filter(|e| e.ignored).count()
    }
    pub fn is_ignored(&self, id: EpisodeId) -> bool {
        self.active.get(&id).is_some_and(|e| e.ignored)
    }
    pub fn active_timestep(&self, id: EpisodeId) -> Option<usize> {
        self.active.get(&id).map(|e| e.next_timestep)
    }
    pub fn episodes(&self) -> impl ExactSizeIterator<Item = &Episode> {
        self.episodes.values()
    }
    pub fn episode(&self, id: EpisodeId) -> Option<&Episode> {
        self.episodes.get(&id)
    }
    /// Explicitly retire completed tapes after their consumers release them.
    /// Validate every identity before mutation; active prefixes are never retired.
    pub fn retire_completed(&mut self, ids: &[EpisodeId]) -> Result<()> {
        self.validate()?;
        let mut unique = std::collections::BTreeSet::new();
        let mut transitions = 0usize;
        let mut removed_usage = ResourceUsage::default();
        for &id in ids {
            ensure(unique.insert(id), "duplicate completed retirement identity")?;
            let episode = self
                .episodes
                .get(&id)
                .ok_or_else(|| Error::State("completed retirement tape is not retained".into()))?;
            ensure(
                episode.ended() && !self.active.contains_key(&id),
                "completed retirement cannot discard an active prefix",
            )?;
            transitions = transitions
                .checked_add(episode.learning_len())
                .ok_or_else(|| Error::State("completed retirement count overflow".into()))?;
            if self.limits().is_some() {
                removed_usage = add_usage(removed_usage, episode.usage()?)?;
            }
        }
        for &id in ids {
            self.episodes.remove(&id);
            self.index.remove(id);
        }
        self.len -= transitions;
        if self.limits().is_some() {
            subtract_usage(&mut self.usage, removed_usage);
        }
        if self.latest_terminal.is_some_and(|id| unique.contains(&id)) {
            self.latest_terminal = self
                .episodes
                .values()
                .rev()
                .find(|episode| episode.ended())
                .map(|episode| episode.id);
        }
        Ok(())
    }

    /// Number of uniformly sampleable starts. Only live-lane tails need adjustment;
    /// completed episodes are already represented by their indexed raw lengths.
    pub fn eligible_starts(&self, n_step: usize) -> Result<usize> {
        ensure(n_step > 0, "temporal n-step horizon must be positive")?;
        let excluded: usize = self
            .active
            .iter()
            .filter(|(_, active)| !active.ignored)
            .map(|(id, active)| {
                let retained = if self.version == VERSION {
                    active.next_timestep
                } else {
                    self.episodes
                        .get(id)
                        .expect("retained active temporal episode")
                        .learning_len()
                };
                retained.min(n_step - 1)
            })
            .sum();
        Ok(self.len - excluded)
    }
    pub fn eligible_starts_with(
        &self,
        config: &SequenceConfig,
        requirements: &AnchorRequirements,
        limits: &SampleLimits,
    ) -> Result<usize> {
        let ranges = self.sample_ranges(config, requirements, limits)?;
        Ok(ranges.last().map_or(0, |range| range.total))
    }

    // Each entry removes a live tail from raw indexed ranks. Boundaries are in
    // eligible-rank coordinates, so partition_point also skips fully excluded tapes.
    fn population(&self, n_step: usize) -> Result<(usize, Vec<(usize, usize)>)> {
        ensure(n_step > 0, "temporal n-step horizon must be positive")?;
        let mut tails = Vec::new();
        tails
            .try_reserve_exact(self.active.len())
            .map_err(|e| Error::State(format!("cannot reserve temporal live tails: {e}")))?;
        let mut excluded = 0;
        for (&id, active) in &self.active {
            if active.ignored {
                continue;
            }
            let retained = if self.version == VERSION {
                active.next_timestep
            } else {
                self.index
                    .weight(id)
                    .expect("retained active temporal episode")
            };
            let tail = retained.min(n_step - 1);
            if tail > 0 {
                let end = self.index.before(id) + retained - excluded - tail;
                excluded += tail;
                tails.push((end, excluded));
            }
        }
        Ok((self.len - excluded, tails))
    }

    fn evict_to_capacity(&mut self) {
        while self.len > self.capacity {
            let episode = self
                .episodes
                .pop_first()
                .expect("retained transition has an episode")
                .1;
            self.len -= episode.learning_len();
            self.index.remove(episode.id);
            if let Some(active) = self.active.get_mut(&episode.id) {
                active.ignored = true;
            }
        }
    }
}
impl TemporalSink for EpisodeStore {
    fn begin_episode(&mut self, id: EpisodeId, start: &EpisodeStart) -> Result<()> {
        start.validate_in(&self.observation_space, &self.space)?;
        let initial = &start.observation;
        let learning_start = usize::try_from(start.decision_timestep())
            .map_err(|_| Error::Shape("temporal decision timestep is not representable".into()))?;
        ensure(
            self.highest_episode_id.is_none_or(|highest| id > highest),
            "temporal episode IDs must increase and cannot be reused",
        )?;
        ensure(
            self.active.len() < self.capacity,
            "temporal active episode identity capacity exhausted",
        )?;
        let history = start
            .context
            .as_ref()
            .map(|context| context.history.inputs());
        let bounded = self.limits().is_some();
        if let Some(limits) = self.limits() {
            let base = size_of::<Episode>() + 128;
            let bytes = if let Some(inputs) = &history {
                inputs
                    .iter()
                    .try_fold(base, |bytes, input| {
                        let observation =
                            size_of::<Observation>() + cloned_observation_bytes(input.observation);
                        let step = input.previous.map_or(0, |p| {
                            size_of::<TemporalStep>()
                                + cloned_action_bytes(p.action)
                                + reward_bytes(
                                    p.factual.expect("factual public history input").reward,
                                )
                        });
                        bytes
                            .checked_add(observation)
                            .and_then(|bytes| bytes.checked_add(step))
                    })
                    .ok_or_else(|| {
                        Error::Shape("temporal start history byte count overflow".into())
                    })?
            } else {
                base + size_of::<Observation>() + cloned_observation_bytes(initial)
            };
            let anticipated = ResourceUsage {
                resident_bytes: bytes,
                ..Default::default()
            };
            ensure(
                limits.contains(self.len, add_usage(self.usage, anticipated)?),
                "temporal resource limit: beginning episode",
            )?;
        }
        let (observations, steps) = if let Some(inputs) = history {
            let mut observations = Vec::new();
            let mut steps = Vec::new();
            observations.try_reserve_exact(inputs.len()).map_err(|e| {
                Error::State(format!("cannot reserve temporal context observations: {e}"))
            })?;
            steps
                .try_reserve_exact(learning_start)
                .map_err(|e| Error::State(format!("cannot reserve temporal context steps: {e}")))?;
            for input in inputs {
                observations.push(clone_observation(input.observation));
                if let Some(previous) = input.previous {
                    steps.push(TemporalStep {
                        action: clone_action(previous.action),
                        reward: previous
                            .factual
                            .expect("factual public history input")
                            .reward
                            .clone(),
                        terminated: false,
                        truncated: false,
                        behavior: BehaviorMetadata::default(),
                    });
                }
            }
            (observations, steps)
        } else {
            (
                vec![if bounded {
                    clone_observation(initial)
                } else {
                    initial.clone()
                }],
                Vec::new(),
            )
        };
        let episode = Episode {
            id,
            observations,
            steps,
            base_timestep: 0,
            learning_start,
            previous: None,
            anchors: Vec::new(),
        };
        if let Some(limits) = self.limits() {
            let usage = add_usage(self.usage, episode.usage()?)?;
            ensure(
                limits.contains(self.len, usage),
                "temporal resource limit: beginning episode",
            )?;
            self.usage = usage;
        }
        self.episodes.insert(id, episode);
        self.index.set(id, 0);
        self.active.insert(
            id,
            ActiveEpisode {
                next_timestep: learning_start,
                ignored: false,
            },
        );
        self.highest_episode_id = Some(id);
        if learning_start > 0 {
            self.version = 3;
        }
        Ok(())
    }

    fn push(&mut self, t: &TemporalTransition) -> Result<()> {
        if !legacy_storage(&self.storage) {
            return self.push_bounded(t, None);
        }
        t.validate_in(&self.observation_space, &self.space)?;
        let active = self
            .active
            .get(&t.episode_id)
            .ok_or_else(|| Error::State("temporal transition has no active episode".into()))?;
        ensure(
            t.timestep == active.next_timestep,
            "temporal timestep differs",
        )?;
        let next_timestep = active
            .next_timestep
            .checked_add(1)
            .ok_or_else(|| Error::State("temporal timestep overflow".into()))?;
        if !active.ignored {
            let episode = self
                .episodes
                .get_mut(&t.episode_id)
                .expect("active retained episode exists");
            ensure(
                episode
                    .observations
                    .last()
                    .is_some_and(|last| last.matches(&t.observation)),
                "temporal successor observation is not contiguous",
            )?;
            let next_len = self
                .len
                .checked_add(1)
                .ok_or_else(|| Error::State("temporal capacity count overflow".into()))?;
            episode
                .observations
                .try_reserve(1)
                .map_err(|e| Error::State(format!("cannot reserve temporal observation: {e}")))?;
            episode
                .steps
                .try_reserve(1)
                .map_err(|e| Error::State(format!("cannot reserve temporal transition: {e}")))?;
            episode.observations.push(t.next_observation.clone());
            episode.steps.push(TemporalStep {
                action: t.action.clone(),
                reward: t.reward.clone(),
                terminated: t.terminated,
                truncated: t.truncated,
                behavior: t.behavior.clone(),
            });
            self.len = next_len;
            self.index.set(t.episode_id, episode.learning_len());
        }
        if t.ended() {
            self.active.remove(&t.episode_id);
        } else {
            let active = self
                .active
                .get_mut(&t.episode_id)
                .expect("validated active episode");
            active.next_timestep = next_timestep;
        }
        self.evict_to_capacity();
        Ok(())
    }
    fn abandon_episode(&mut self, id: EpisodeId) -> Result<()> {
        if self.active.remove(&id).is_none() {
            return Ok(());
        }
        if let Some(episode) = self.episodes.remove(&id) {
            self.len -= episode.learning_len();
            self.index.remove(id);
            if self.limits().is_some() {
                subtract_usage(&mut self.usage, episode.usage()?);
            }
            if self.latest_terminal == Some(id) {
                self.latest_terminal = None;
            }
        }
        Ok(())
    }
}
