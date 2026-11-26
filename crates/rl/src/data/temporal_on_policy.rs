//! Immutable fresh temporal windows and bounded host-owned collection metadata.

mod contract;
pub use contract::{TemporalLearnerBinding, TemporalOnPolicyLearner};

use super::{
    returns::DiscountedRewards,
    temporal::{
        EpisodeStore, ResourceLimits, SampleLimits, SequenceBatch, SequenceConfig, SequenceOrigin,
        SequenceSample, TemporalStep, TemporalStorage, actions_equal_bits, cloned_action_bytes,
        cloned_observation_bytes, reward_bytes, sample_at,
    },
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        ensure,
        env::{EpisodeStart, Observation, ObservationSpace},
        recording::RecordingKey,
        stateful_on_policy::{DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor},
        temporal::{TemporalSink, TemporalTransition},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    cell::Cell,
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalBatchKind {
    Bootstrapped,
    Episodic,
    Pathwise,
}

/// One learning token. Context remains in `sequences`, including all preceding
/// observations and factual actions, regardless of minibatch/window selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalRow {
    pub key: RecordingKey,
    pub sample: usize,
    pub time: usize,
    pub log_prob: f32,
    pub latent: Option<Vec<f32>>,
    pub distribution: DistributionSnapshot,
    pub value: Option<f32>,
    pub next_value: Option<f32>,
    pub advantage: Option<f32>,
    pub return_: Option<f32>,
    pub discount_weight: f32,
}

/// Low-level owned learning input, like the feedforward `OnPolicyBatch`.
/// Callers may supply their own finite advantages and return estimates; `validate`
/// checks their shape and support, not the estimator's scientific correctness.
/// The standard trainer seals canonical targets and independently audits them
/// against its raw rollout/configuration when restoring a complete run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "A: Serialize", deserialize = "A: DeserializeOwned"))]
pub struct TemporalOnPolicyBatch<A> {
    pub version: u32,
    pub behavior: A,
    pub revision: u64,
    pub kind: TemporalBatchKind,
    pub gamma: f32,
    pub gae_lambda: Option<f32>,
    pub episode_count: usize,
    pub sequences: SequenceBatch,
    pub rows: Vec<TemporalRow>,
}

impl<A: StatefulOnPolicyActor + Clone> TemporalOnPolicyBatch<A> {
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn tokens(&self) -> Vec<(usize, usize)> {
        self.rows.iter().map(|row| (row.sample, row.time)).collect()
    }
    pub fn observations(&self) -> Vec<Observation> {
        self.rows
            .iter()
            .map(|row| self.sequences.samples[row.sample].observations[row.time].clone())
            .collect()
    }
    pub fn actions(&self) -> Vec<Action> {
        self.rows
            .iter()
            .map(|row| {
                self.sequences.samples[row.sample].steps[row.time]
                    .action
                    .clone()
            })
            .collect()
    }
    pub fn old_log_probs(&self) -> Vec<f32> {
        self.rows.iter().map(|row| row.log_prob).collect()
    }
    pub fn latents(&self) -> Vec<Option<Vec<f32>>> {
        self.rows.iter().map(|row| row.latent.clone()).collect()
    }
    pub fn returns(&self) -> Result<Vec<f32>> {
        self.rows
            .iter()
            .map(|row| {
                row.return_
                    .ok_or_else(|| Error::State("temporal batch has no return target".into()))
            })
            .collect()
    }
    pub fn advantages(&self) -> Result<Vec<f32>> {
        self.rows
            .iter()
            .map(|row| {
                row.advantage
                    .ok_or_else(|| Error::State("temporal batch has no advantage target".into()))
            })
            .collect()
    }
    pub fn old_values(&self) -> Result<Vec<f32>> {
        self.rows
            .iter()
            .map(|row| {
                row.value
                    .ok_or_else(|| Error::State("temporal batch has no scalar values".into()))
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::BatchValidation);
        validate_settings(self.kind, self.gamma, self.gae_lambda)?;
        ensure(
            matches!(self.version, 1 | 2)
                && !self.rows.is_empty()
                && self.revision == self.behavior.revision()
                && self.sequences.config.n_step == 1
                && (self.kind != TemporalBatchKind::Episodic || self.episode_count > 0),
            "invalid temporal on-policy batch version, revision, horizon or episode count",
        )?;
        let observations = self.behavior.observation_space();
        let space = self.behavior.action_space();
        observations.validate()?;
        space.validate()?;
        self.sequences.validate_in(&observations, space)?;
        let mut learning = BTreeMap::new();
        let mut prefixes = BTreeMap::<u64, &SequenceSample>::new();
        for (sample_index, sample) in self.sequences.samples.iter().enumerate() {
            ensure(
                (self.version == 2 || sample.trajectory_start == 0)
                    && sample.history_start == 0
                    && sample.previous.is_none()
                    && matches!(sample.origin, SequenceOrigin::Reset),
                "on-policy sequence requires its complete genuine reset prefix",
            )?;
            // Windows own repeated reset prefixes, but all copies of one actual
            // episode must describe the same facts. Keep only its longest borrowed
            // prefix: each comparison costs at most this sample's owned payload.
            match prefixes.entry(sample.episode_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(sample);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    ensure(
                        entry.get().factual_prefixes_equal(sample),
                        "temporal windows contain contradictory factual episode prefixes",
                    )?;
                    if sample.steps.len() > entry.get().steps.len() {
                        entry.insert(sample);
                    }
                }
            }
            if matches!(observations, ObservationSpace::Structured(_)) {
                let mut previous = None;
                for observation in &sample.observations {
                    observation.validate_event_history(previous)?;
                    previous = Some(observation);
                }
            }
            let mut physical_tick = 0;
            for (time, step) in sample.steps.iter().enumerate() {
                if sample.loss_mask[time] {
                    ensure(
                        step.behavior.revision == self.revision
                            && learning
                                .insert((sample_index, time), physical_tick)
                                .is_none(),
                        "fresh temporal learning token has a stale revision",
                    )?;
                }
                physical_tick = advance_physical_tick(physical_tick, step)?;
            }
        }
        let mut previous = None;
        for row in &self.rows {
            let physical_tick = learning.remove(&(row.sample, row.time)).ok_or_else(|| {
                Error::State("temporal row is duplicated or outside learning windows".into())
            })?;
            ensure(
                previous.is_none_or(|key| key < row.key),
                "temporal rows are unordered",
            )?;
            previous = Some(row.key);
            let sample = &self.sequences.samples[row.sample];
            let step = &sample.steps[row.time];
            ensure(
                row.key.episode_id == sample.episode_id
                    && row.key.timestep == row.time
                    && step
                        .behavior
                        .log_prob
                        .is_some_and(|lp| lp.to_bits() == row.log_prob.to_bits())
                    && row.discount_weight.to_bits()
                        == discount_weight(self.gamma, physical_tick).to_bits()
                    && row.advantage.is_none_or(f32::is_finite)
                    && row.return_.is_none_or(f32::is_finite)
                    && row.next_value.is_none_or(f32::is_finite),
                "temporal row provenance, likelihood or target differs",
            )?;
            validate_details(
                space,
                &sample.observations[row.time],
                &step.action,
                row.log_prob,
                row.latent.as_deref(),
                &row.distribution,
                row.value,
            )?;
            match self.kind {
                TemporalBatchKind::Bootstrapped => ensure(
                    row.value.is_some()
                        && row.next_value.is_some()
                        && row.advantage.is_some()
                        && row.return_.is_some(),
                    "bootstrapped temporal row lacks value or targets",
                )?,
                TemporalBatchKind::Episodic => ensure(
                    row.value.is_some() == row.next_value.is_some()
                        && row.advantage.is_some()
                        && row.return_.is_some(),
                    "episodic temporal row lacks return or has inconsistent baseline metadata",
                )?,
                TemporalBatchKind::Pathwise => ensure(
                    row.value.is_none()
                        && row.next_value.is_none()
                        && row.advantage.is_none()
                        && row.return_.is_none(),
                    "pathwise temporal rows cannot invent state-value targets",
                )?,
            }
            ensure(
                !step.terminated || row.next_value.is_none_or(|v| v.to_bits() == 0f32.to_bits()),
                "terminated temporal row has a nonzero bootstrap",
            )?;
        }
        ensure(
            learning.is_empty(),
            "temporal learning windows have missing metadata rows",
        )
    }

    /// Select whole windows without deleting any of their factual context.
    /// Row order remains canonical; token sample indices follow the selected order.
    pub fn select_windows(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        let mut positions = BTreeMap::new();
        ensure(!indices.is_empty(), "empty temporal window selection")?;
        for (new, &old) in indices.iter().enumerate() {
            ensure(
                old < self.sequences.samples.len() && positions.insert(old, new).is_none(),
                "invalid or duplicate temporal window selection",
            )?;
        }
        let rows = self
            .rows
            .iter()
            .filter_map(|row| {
                positions.get(&row.sample).map(|&sample| {
                    let mut row = row.clone();
                    row.sample = sample;
                    row
                })
            })
            .collect();
        let out = Self {
            version: self.version,
            behavior: self.behavior.clone(),
            revision: self.revision,
            kind: self.kind,
            gamma: self.gamma,
            gae_lambda: self.gae_lambda,
            episode_count: self.episode_count,
            sequences: SequenceBatch {
                samples: indices
                    .iter()
                    .map(|&i| self.sequences.samples[i].clone())
                    .collect(),
                config: self.sequences.config,
            },
            rows,
        };
        out.validate()?;
        Ok(out)
    }

    /// Borrow canonical learning rows from an audited batch. The shuffled
    /// selection supplies distinct, in-bounds indices into that batch.
    #[cfg(feature = "neural")]
    pub(crate) fn validated_window_rows(&self, indices: &[usize]) -> Vec<(usize, &TemporalRow)> {
        let mut positions = vec![None; self.sequences.samples.len()];
        for (local, &global) in indices.iter().enumerate() {
            positions[global] = Some(local);
        }
        self.rows
            .iter()
            .filter_map(|row| positions[row.sample].map(|local| (local, row)))
            .collect()
    }
}

impl<A: StatefulOnPolicyActor + Clone + Snapshot> Snapshot for TemporalOnPolicyBatch<A> {
    fn validate_snapshot(&self) -> Result<()> {
        self.behavior.validate_snapshot()?;
        self.validate()
    }
}

/// Host-owned sampling metadata, exposed through a read-only committed view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedRecording {
    pub key: RecordingKey,
    pub revision: u64,
    pub action: Action,
    pub log_prob: f32,
    pub latent: Option<Vec<f32>>,
    pub distribution: DistributionSnapshot,
    pub value: Option<f32>,
    pub next_value: Option<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRow {
    row: CollectedRecording,
    successor: Option<bool>,
}

/// Sidecar budgets count logical host payload bytes and row descriptors, not
/// allocator capacity. Finite count/schema limits additionally bound allocations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalRollout {
    version: u32,
    store: EpisodeStore,
    max_fresh: usize,
    max_metadata_bytes: usize,
    metadata_bytes: usize,
    acknowledged: BTreeMap<u64, usize>,
    pending: Vec<PendingRow>,
    fresh: Vec<CollectedRecording>,
    #[serde(skip)]
    audited: Cell<bool>,
}

impl TemporalRollout {
    pub fn new(
        capacity: usize,
        observations: impl Into<ObservationSpace>,
        space: ActionSpace,
        max_fresh: usize,
        max_metadata_bytes: usize,
    ) -> Result<Self> {
        Self::with_resident_limit(
            capacity,
            observations,
            space,
            max_fresh,
            max_metadata_bytes,
            None,
        )
    }

    pub fn with_resident_limit(
        capacity: usize,
        observations: impl Into<ObservationSpace>,
        space: ActionSpace,
        max_fresh: usize,
        max_metadata_bytes: usize,
        max_resident_bytes: Option<usize>,
    ) -> Result<Self> {
        ensure(
            capacity > 0 && max_fresh > 0 && max_metadata_bytes > 0,
            "invalid temporal rollout resource limits",
        )?;
        let observations = observations.into();
        observations.validate()?;
        space.validate()?;
        let limits = rollout_limits(capacity, &observations, &space, max_resident_bytes)?;
        let store = EpisodeStore::with_storage(
            capacity,
            observations,
            space,
            TemporalStorage::FullPrefix(limits),
        )?;
        let out = Self {
            version: 1,
            store,
            max_fresh,
            max_metadata_bytes,
            metadata_bytes: 0,
            acknowledged: BTreeMap::new(),
            pending: Vec::new(),
            fresh: Vec::new(),
            audited: Cell::new(false),
        };
        out.validate_snapshot()?;
        Ok(out)
    }
    pub(crate) fn validate_limits(
        &self,
        capacity: usize,
        max_fresh: usize,
        max_metadata_bytes: usize,
        max_resident_bytes: Option<usize>,
    ) -> Result<()> {
        let expected = rollout_limits(
            capacity,
            self.store.observation_space(),
            self.store.action_space(),
            max_resident_bytes,
        )?;
        ensure(
            self.max_fresh == max_fresh
                && self.max_metadata_bytes == max_metadata_bytes
                && self.store.storage() == &TemporalStorage::FullPrefix(expected),
            "temporal rollout resource limits differ from trainer config",
        )
    }

    /// Finite bounds for exact prefix duplication and padded neural work.
    /// Call `seal_with_limits` to choose another explicit resource contract.
    pub fn default_sample_limits() -> SampleLimits {
        SampleLimits {
            max_sample_tokens: 100_000,
            max_batch_tokens: 1_000_000,
            max_sample_payload_bytes: 64 * 1024 * 1024,
            max_batch_payload_bytes: 256 * 1024 * 1024,
        }
    }
    pub fn fresh_len(&self) -> usize {
        self.fresh.len()
    }
    /// Borrow committed fresh recordings in collection order. Pending sampling
    /// and successor candidates are excluded; acknowledgement releases this view.
    pub fn fresh_recordings(&self) -> std::slice::Iter<'_, CollectedRecording> {
        self.fresh.iter()
    }
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }
    pub fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }
    pub fn store(&self) -> &EpisodeStore {
        &self.store
    }

    fn validate_local(&self) -> Result<()> {
        if !self.audited.get() {
            self.validate_snapshot()?;
        }
        Ok(())
    }

    pub fn stage_decision(
        &mut self,
        key: RecordingKey,
        revision: u64,
        sample: &OnPolicySample,
    ) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::RolloutStorage);
        self.validate_local()?;
        let episode = self.store().episode(key.episode_id).ok_or_else(|| {
            Error::State("staged temporal decision has no retained reset prefix".into())
        })?;
        ensure(
            !episode.ended()
                && episode.end_timestep() == key.timestep
                && !self.pending.iter().any(|p| p.row.key == key),
            "duplicate or shifted staged temporal decision",
        )?;
        let log_prob = sample.decision.log_prob.ok_or_else(|| {
            Error::State("on-policy decision is missing its sampled likelihood".into())
        })?;
        validate_details(
            self.store().action_space(),
            episode.observations.last().expect("validated prefix"),
            &sample.decision.action,
            log_prob,
            sample.latent.as_deref(),
            &sample.distribution,
            sample.value,
        )?;
        let bytes = detail_bytes(
            &sample.decision.action,
            sample.latent.as_deref(),
            &sample.distribution,
        )?;
        let next_bytes = self
            .metadata_bytes
            .checked_add(bytes)
            .ok_or_else(|| Error::State("temporal metadata byte overflow".into()))?;
        ensure(
            self.fresh
                .len()
                .checked_add(self.pending.len())
                .and_then(|n| n.checked_add(1))
                .is_some_and(|n| n <= self.max_fresh)
                && next_bytes <= self.max_metadata_bytes,
            "temporal rich metadata resource limit",
        )?;
        self.pending
            .try_reserve_exact(1)
            .map_err(|e| Error::State(format!("cannot reserve staged temporal metadata: {e}")))?;
        self.pending.push(PendingRow {
            row: CollectedRecording {
                key,
                revision,
                action: sample.decision.action.clone(),
                log_prob,
                latent: sample.latent.clone(),
                distribution: sample.distribution.clone(),
                value: sample.value,
                next_value: None,
            },
            successor: None,
        });
        self.metadata_bytes = next_bytes;
        Ok(())
    }

    pub fn stage_successors(
        &mut self,
        keys: &[RecordingKey],
        values: &[Option<f32>],
        terminated: &[bool],
    ) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::RolloutStorage);
        self.validate_local()?;
        ensure(
            keys.len() == values.len()
                && keys.len() == terminated.len()
                && keys.len() <= self.pending.len(),
            "temporal successor metadata batch length mismatch",
        )?;
        let mut updates = Vec::new();
        updates
            .try_reserve_exact(keys.len())
            .map_err(|e| Error::State(format!("cannot reserve temporal successors: {e}")))?;
        let mut unique = BTreeSet::new();
        for ((&key, &value), &terminated) in keys.iter().zip(values).zip(terminated) {
            ensure(
                unique.insert(key) && value.is_none_or(f32::is_finite),
                "duplicate or nonfinite temporal successor metadata",
            )?;
            let index = self
                .pending
                .iter()
                .position(|pending| pending.row.key == key)
                .ok_or_else(|| Error::State("temporal successor has no staged decision".into()))?;
            let pending = &self.pending[index];
            ensure(
                pending.successor.is_none(),
                "temporal successor was already staged",
            )?;
            let next = if pending.row.value.is_some() {
                if terminated {
                    Some(0.)
                } else {
                    Some(value.ok_or_else(|| {
                        Error::State("scalar critic successor value is missing".into())
                    })?)
                }
            } else {
                ensure(
                    value.is_none(),
                    "baseline-free temporal decision received a scalar successor",
                )?;
                None
            };
            updates.push((index, next, terminated));
        }
        for (index, next, terminated) in updates {
            self.pending[index].row.next_value = next;
            self.pending[index].successor = Some(terminated);
        }
        Ok(())
    }
    pub fn push_recorded(&mut self, transition: &TemporalTransition) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::RolloutStorage);
        self.validate_local()?;
        transition.validate_in(
            self.store().observation_space(),
            self.store().action_space(),
        )?;
        let key = RecordingKey {
            episode_id: transition.episode_id,
            timestep: transition.timestep,
        };
        let index = self
            .pending
            .iter()
            .position(|p| p.row.key == key)
            .ok_or_else(|| Error::State("temporal transition lacks staged rich metadata".into()))?;
        let pending = &self.pending[index];
        ensure(
            pending.successor == Some(transition.terminated)
                && record_matches_transition(&pending.row, transition),
            "temporal transition differs from sampled decision or successor readiness",
        )?;
        ensure(
            self.store().len() < self.store().capacity(),
            "temporal full-prefix capacity exhausted before acknowledgement",
        )?;
        self.fresh
            .try_reserve_exact(1)
            .map_err(|e| Error::State(format!("cannot reserve fresh temporal metadata: {e}")))?;
        self.store.push(transition)?;
        self.fresh.push(self.pending.swap_remove(index).row);
        Ok(())
    }
    /// Construct immutable targets over complete fresh fragments, then partition
    /// learning windows. Prefixes remain raw factual context under their original
    /// behavior metadata; only freshly collected tokens receive loss masks.
    pub fn seal<A: StatefulOnPolicyActor + Clone>(
        &self,
        behavior: A,
        kind: TemporalBatchKind,
        gamma: f32,
        gae_lambda: Option<f32>,
        normalize_advantages: bool,
        sequence: &SequenceConfig,
    ) -> Result<TemporalOnPolicyBatch<A>> {
        self.seal_with_limits(
            behavior,
            kind,
            gamma,
            gae_lambda,
            normalize_advantages,
            sequence,
            &TemporalRollout::default_sample_limits(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn seal_with_limits<A: StatefulOnPolicyActor + Clone>(
        &self,
        behavior: A,
        kind: TemporalBatchKind,
        gamma: f32,
        gae_lambda: Option<f32>,
        normalize_advantages: bool,
        sequence: &SequenceConfig,
        limits: &SampleLimits,
    ) -> Result<TemporalOnPolicyBatch<A>> {
        self.validate_snapshot()?;
        let store = &self.store;
        validate_settings(kind, gamma, gae_lambda)?;
        sequence.validate()?;
        ensure(
            sequence.n_step == 1 && self.pending.is_empty() && !self.fresh.is_empty(),
            "temporal sealing requires complete fresh metadata and a one-step sequence horizon",
        )?;
        ensure(
            &behavior.observation_space() == store.observation_space()
                && behavior.action_space() == store.action_space(),
            "temporal behavior and rollout schema differ",
        )?;
        let revision = behavior.revision();
        ensure(
            self.fresh.iter().all(|row| row.revision == revision),
            "fresh temporal fragment mixes behavior revisions",
        )?;
        let mut collected: Vec<_> = self.fresh.iter().collect();
        collected.sort_unstable_by_key(|row| row.key);
        preflight_sequences(store, &collected, sequence, limits)?;
        let mut rows: Vec<_> = collected
            .iter()
            .map(|row| TemporalRow {
                key: row.key,
                sample: 0,
                time: row.key.timestep,
                log_prob: row.log_prob,
                latent: row.latent.clone(),
                distribution: row.distribution.clone(),
                value: row.value,
                next_value: row.next_value,
                advantage: None,
                return_: None,
                discount_weight: 0.,
            })
            .collect();
        let mut samples = Vec::new();
        let mut episode_count = 0usize;
        let mut first = 0;
        while first < collected.len() {
            let id = collected[first].key.episode_id;
            let end = first
                + collected[first..]
                    .iter()
                    .take_while(|row| row.key.episode_id == id)
                    .count();
            let episode = store
                .episode(id)
                .expect("audited fresh episode is retained");
            let start = collected[first].key.timestep;
            let steps = &episode.steps[start..];
            let mut physical_tick = episode.steps[..start]
                .iter()
                .try_fold(0, advance_physical_tick)?;
            for (row, step) in rows[first..end].iter_mut().zip(steps) {
                row.discount_weight = discount_weight(gamma, physical_tick);
                physical_tick = advance_physical_tick(physical_tick, step)?;
            }
            episode_count += usize::from(episode.steps.last().is_some_and(|step| step.terminated));
            if kind == TemporalBatchKind::Episodic {
                ensure(
                    start == episode.learning_start
                        && episode.ended()
                        && episode.steps.last().is_some_and(|s| s.terminated)
                        && !episode
                            .steps
                            .iter()
                            .any(|step| step.truncated && !step.terminated),
                    "Monte Carlo temporal collection requires a complete terminal learning suffix",
                )?;
            }
            if kind != TemporalBatchKind::Pathwise {
                let rewards = DiscountedRewards::new(steps.iter().map(|step| &step.reward), gamma)?;
                let terminated: Vec<_> = steps.iter().map(|step| step.terminated).collect();
                let truncated: Vec<_> = steps.iter().map(|step| step.truncated).collect();
                let values: Vec<_> = collected[first..end]
                    .iter()
                    .map(|row| {
                        if kind == TemporalBatchKind::Bootstrapped {
                            row.value.ok_or_else(|| {
                                Error::State(
                                    "bootstrapped collection is missing scalar values".into(),
                                )
                            })
                        } else {
                            Ok(row.value.unwrap_or(0.))
                        }
                    })
                    .collect::<Result<_>>()?;
                let next_values: Vec<_> = collected[first..end]
                    .iter()
                    .map(|row| {
                        if kind == TemporalBatchKind::Bootstrapped {
                            row.next_value.ok_or_else(|| {
                                Error::State(
                                    "bootstrapped collection is missing successor values".into(),
                                )
                            })
                        } else {
                            Ok(0.)
                        }
                    })
                    .collect::<Result<_>>()?;
                let (advantages, returns) = if let Some(lambda) = gae_lambda {
                    super::gae::gae_with(
                        &rewards.rewards,
                        &values,
                        &next_values,
                        &terminated,
                        &truncated,
                        rewards.discounts(),
                        lambda,
                    )?
                } else {
                    let targets = super::returns::discounted_returns_with(
                        &rewards.rewards,
                        &next_values,
                        &terminated,
                        &truncated,
                        rewards.discounts(),
                    )?;
                    let advantages = targets
                        .iter()
                        .zip(&values)
                        .map(|(target, value)| target - value)
                        .collect();
                    (advantages, targets)
                };
                for (row, (advantage, target)) in rows[first..end]
                    .iter_mut()
                    .zip(advantages.into_iter().zip(returns))
                {
                    row.advantage = Some(advantage);
                    row.return_ = Some(target);
                }
            }
            for learn_start in (start..episode.steps.len()).step_by(sequence.learn_len) {
                let sample = sample_at(store, id, learn_start, sequence)?;
                let sample_index = samples.len();
                for row in &mut rows[first..end] {
                    if sample.loss_mask.get(row.time).copied().unwrap_or(false) {
                        row.sample = sample_index;
                    }
                }
                samples.push(sample);
            }
            first = end;
        }
        if normalize_advantages && kind == TemporalBatchKind::Bootstrapped {
            let mean = rows
                .iter()
                .map(|row| row.advantage.expect("prepared target") as f64)
                .sum::<f64>()
                / rows.len() as f64;
            let variance = rows
                .iter()
                .map(|row| (row.advantage.expect("prepared target") as f64 - mean).powi(2))
                .sum::<f64>()
                / rows.len() as f64;
            let scale = variance.sqrt().max(1e-8);
            for row in &mut rows {
                row.advantage =
                    Some(((row.advantage.expect("prepared target") as f64 - mean) / scale) as f32);
            }
        }
        let out = TemporalOnPolicyBatch {
            version: if samples.iter().any(|sample| sample.trajectory_start != 0) {
                2
            } else {
                1
            },
            behavior,
            revision,
            kind,
            gamma,
            gae_lambda,
            episode_count,
            sequences: SequenceBatch {
                samples,
                config: *sequence,
            },
            rows,
        };
        out.validate()?;
        Ok(out)
    }
    /// Release fresh metadata only after its consumer has finished. Complete
    /// episodes can then retire; every ongoing episode retains its reset prefix.
    pub fn acknowledge(&mut self) -> Result<()> {
        self.validate_snapshot()?;
        ensure(
            self.pending.is_empty(),
            "cannot acknowledge pending temporal decisions",
        )?;
        let completed: Vec<_> = self
            .store
            .episodes()
            .filter(|episode| episode.ended())
            .map(|episode| episode.id)
            .collect();
        self.store.retire_completed(&completed)?;
        for id in completed {
            self.acknowledged.remove(&id);
        }
        for episode in self.store.episodes() {
            self.acknowledged.insert(episode.id, episode.end_timestep());
        }
        self.fresh.clear();
        self.metadata_bytes = 0;
        Ok(())
    }
}

impl TemporalSink for TemporalRollout {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::RolloutStorage);
        self.validate_local()?;
        self.store.begin_episode(id, start)?;
        self.acknowledged
            .insert(id, start.decision_timestep() as usize);
        Ok(())
    }
    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        self.push_recorded(transition)
    }
    fn abandon_episode(&mut self, id: u64) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::RolloutStorage);
        self.validate_local()?;
        let remaining = self
            .fresh
            .iter()
            .filter(|r| r.key.episode_id != id)
            .chain(
                self.pending
                    .iter()
                    .map(|p| &p.row)
                    .filter(|r| r.key.episode_id != id),
            )
            .try_fold(0usize, |total, row| {
                total
                    .checked_add(detail_bytes(
                        &row.action,
                        row.latent.as_deref(),
                        &row.distribution,
                    )?)
                    .ok_or_else(|| Error::State("temporal metadata byte overflow".into()))
            })?;
        self.store.abandon_episode(id)?;
        self.pending.retain(|p| p.row.key.episode_id != id);
        self.fresh.retain(|r| r.key.episode_id != id);
        if let Some(episode) = self.store().episode(id) {
            self.acknowledged.insert(id, episode.end_timestep());
        } else {
            self.acknowledged.remove(&id);
        }
        self.metadata_bytes = remaining;
        Ok(())
    }
}

impl Snapshot for TemporalRollout {
    fn record_type() -> &'static str {
        // The owned rollout keeps the version-1 wire format after dropping borrowed stores.
        "rl::data::temporal_on_policy::TemporalRollout<rl::data::temporal::EpisodeStore>"
    }

    fn validate_snapshot(&self) -> Result<()> {
        let store = &self.store;
        self.audited.set(false);
        store.validate_snapshot()?;
        ensure(
            self.version == 1
                && self.max_fresh > 0
                && self.max_metadata_bytes > 0
                && matches!(store.storage(), TemporalStorage::FullPrefix(_))
                && self
                    .pending
                    .len()
                    .checked_add(self.fresh.len())
                    .is_some_and(|n| n <= self.max_fresh)
                && self.acknowledged.len() == store.episodes().len(),
            "invalid temporal rollout snapshot dimensions or storage",
        )?;
        let mut bytes = 0usize;
        let mut keys = BTreeSet::new();
        let mut fresh_keys = BTreeSet::new();
        for (row, successor) in self.fresh.iter().map(|row| (row, None)).chain(
            self.pending
                .iter()
                .map(|pending| (&pending.row, Some(pending.successor))),
        ) {
            ensure(
                keys.insert(row.key) && row.next_value.is_none_or(f32::is_finite),
                "duplicate or nonfinite temporal rich metadata",
            )?;
            ensure(
                self.acknowledged.contains_key(&row.key.episode_id),
                "rich metadata has no acknowledged prefix",
            )?;
            let episode = store
                .episode(row.key.episode_id)
                .ok_or_else(|| Error::State("temporal metadata has no retained prefix".into()))?;
            let observation = episode
                .observations
                .get(row.key.timestep)
                .ok_or_else(|| Error::State("temporal metadata observation is shifted".into()))?;
            validate_details(
                store.action_space(),
                observation,
                &row.action,
                row.log_prob,
                row.latent.as_deref(),
                &row.distribution,
                row.value,
            )?;
            bytes = bytes
                .checked_add(detail_bytes(
                    &row.action,
                    row.latent.as_deref(),
                    &row.distribution,
                )?)
                .ok_or_else(|| Error::State("temporal metadata byte overflow".into()))?;
            if let Some(successor) = successor {
                ensure(
                    !episode.ended() && row.key.timestep == episode.end_timestep(),
                    "pending temporal decision is not at the current live boundary",
                )?;
                ensure(
                    match successor {
                        None => row.next_value.is_none(),
                        Some(terminated) => {
                            row.value.is_some() == row.next_value.is_some()
                                && (!terminated
                                    || row.next_value.is_none_or(|v| v.to_bits() == 0f32.to_bits()))
                        }
                    },
                    "invalid staged temporal successor readiness",
                )?;
            } else {
                let step = episode.steps.get(row.key.timestep).ok_or_else(|| {
                    Error::State("fresh temporal metadata has no transition".into())
                })?;
                ensure(
                    row.revision == step.behavior.revision
                        && actions_equal_bits(&row.action, &step.action)
                        && step
                            .behavior
                            .log_prob
                            .is_some_and(|lp| lp.to_bits() == row.log_prob.to_bits())
                        && row.value.is_some() == row.next_value.is_some()
                        && (!step.terminated
                            || row.next_value.is_none_or(|v| v.to_bits() == 0f32.to_bits())),
                    "fresh temporal metadata differs from recorded behavior",
                )?;
                fresh_keys.insert(row.key);
            }
        }
        ensure(
            bytes == self.metadata_bytes && bytes <= self.max_metadata_bytes,
            "temporal rich metadata byte accounting differs",
        )?;
        let mut count = 0usize;
        for (&id, &start) in &self.acknowledged {
            let episode = store.episode(id).ok_or_else(|| {
                Error::State("temporal prefix has no acknowledgement clock".into())
            })?;
            ensure(
                episode.base_timestep == 0
                    && start >= episode.learning_start
                    && start <= episode.steps.len(),
                "invalid temporal acknowledgement clock or missing reset prefix",
            )?;
            for timestep in start..episode.steps.len() {
                ensure(
                    fresh_keys.remove(&RecordingKey {
                        episode_id: episode.id,
                        timestep,
                    }),
                    "temporal fresh suffix is missing behavior metadata",
                )?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| Error::State("temporal fresh count overflow".into()))?;
            }
        }
        ensure(
            count == self.fresh.len() && fresh_keys.is_empty(),
            "fresh temporal metadata precedes its acknowledged boundary",
        )?;
        self.audited.set(true);
        Ok(())
    }
}

fn rollout_limits(
    capacity: usize,
    observations: &ObservationSpace,
    space: &ActionSpace,
    max_resident_bytes: Option<usize>,
) -> Result<ResourceLimits> {
    let max_resident_bytes = match (max_resident_bytes, observations.dense_width()) {
        (Some(limit), _) => limit,
        (None, Some(width)) => {
            let action_bytes = match space {
                ActionSpace::Discrete { count } => *count,
                _ => space.dimensions().saturating_mul(size_of::<f32>()),
            };
            width
                .checked_mul(size_of::<f32>())
                .and_then(|n| n.checked_add(action_bytes))
                .and_then(|n| n.checked_add(512))
                .and_then(|n| n.checked_mul(512))
                .and_then(|n| n.checked_mul(capacity))
                .and_then(|n| n.checked_add(4096))
                .ok_or_else(|| Error::Config("temporal rollout resident limit overflow".into()))?
        }
        (None, None) => 256 * 1024 * 1024,
    };
    Ok(ResourceLimits {
        max_transitions: capacity,
        max_resident_bytes,
        max_anchor_bytes: 0,
        max_anchors: 0,
        max_anchor_payload_bytes: 1,
        max_dimension_rank: 1,
    })
}

fn validate_settings(kind: TemporalBatchKind, gamma: f32, lambda: Option<f32>) -> Result<()> {
    ensure(
        gamma.is_finite()
            && (0.0..=1.0).contains(&gamma)
            && lambda.is_none_or(|v| v.is_finite() && (0.0..=1.0).contains(&v))
            && (kind == TemporalBatchKind::Bootstrapped || lambda.is_none()),
        "invalid temporal on-policy return settings",
    )
}
fn advance_physical_tick(tick: u64, step: &TemporalStep) -> Result<u64> {
    tick.checked_add(step.reward.duration())
        .ok_or_else(|| Error::State("temporal rollout physical clock overflow".into()))
}

fn discount_weight(gamma: f32, physical_tick: u64) -> f32 {
    (gamma as f64).powf(physical_tick as f64) as f32
}

/// Measure copied logical payload with one source scan per episode. A long
/// prefix is never repeatedly traversed just to discover that copying it into
/// every window would exceed the declared budget.
fn preflight_sequences(
    store: &EpisodeStore,
    rows: &[&CollectedRecording],
    config: &SequenceConfig,
    limits: &SampleLimits,
) -> Result<()> {
    ensure(
        limits.max_sample_tokens > 0
            && limits.max_batch_tokens > 0
            && limits.max_sample_payload_bytes > 0
            && limits.max_batch_payload_bytes > 0,
        "temporal sequence limits must be positive",
    )?;
    let overflow = || Error::State("temporal sealed batch payload or token overflow".into());
    let mut payload = rows
        .iter()
        .try_fold(size_of::<SequenceBatch>(), |total, row| {
            total
                .checked_add(detail_bytes(
                    &row.action,
                    row.latent.as_deref(),
                    &row.distribution,
                )?)
                .ok_or_else(overflow)
        })?;
    ensure(
        payload <= limits.max_batch_payload_bytes,
        "temporal sealed metadata exceeds batch payload limit",
    )?;
    let mut windows = 0usize;
    let mut maximum_tokens = 0usize;
    let mut first = 0;
    while first < rows.len() {
        let id = rows[first].key.episode_id;
        let next = first
            + rows[first..]
                .iter()
                .take_while(|row| row.key.episode_id == id)
                .count();
        let episode = store.episode(id).expect("audited source prefix");
        let mut measured = 0;
        let mut nested = cloned_observation_bytes(&episode.observations[0]);
        for start in (rows[first].key.timestep..episode.steps.len()).step_by(config.learn_len) {
            let end = start
                .checked_add(config.learn_len)
                .ok_or_else(overflow)?
                .min(episode.steps.len());
            let tokens = end.checked_add(1).ok_or_else(overflow)?;
            windows = windows.checked_add(1).ok_or_else(overflow)?;
            maximum_tokens = maximum_tokens.max(tokens);
            ensure(
                tokens <= limits.max_sample_tokens
                    && maximum_tokens
                        .checked_mul(windows)
                        .is_some_and(|n| n <= limits.max_batch_tokens),
                "temporal exact prefixes exceed sample or padded batch token limits",
            )?;
            for t in measured..end {
                let step = &episode.steps[t];
                nested = nested
                    .checked_add(cloned_action_bytes(&step.action))
                    .and_then(|n| n.checked_add(reward_bytes(&step.reward)))
                    .and_then(|n| {
                        n.checked_add(cloned_observation_bytes(&episode.observations[t + 1]))
                    })
                    .ok_or_else(overflow)?;
            }
            measured = end;
            let sample_payload = tokens
                .checked_mul(size_of::<Observation>())
                .and_then(|n| {
                    end.checked_mul(size_of::<TemporalStep>() + size_of::<bool>())
                        .and_then(|m| n.checked_add(m))
                })
                .and_then(|n| n.checked_add(size_of::<SequenceSample>()))
                .and_then(|n| n.checked_add(nested))
                .ok_or_else(overflow)?;
            payload = payload.checked_add(sample_payload).ok_or_else(overflow)?;
            ensure(
                sample_payload <= limits.max_sample_payload_bytes
                    && payload <= limits.max_batch_payload_bytes,
                "temporal exact prefix copies exceed sample or batch payload limits",
            )?;
        }
        first = next;
    }
    Ok(())
}

fn record_matches_transition(row: &CollectedRecording, transition: &TemporalTransition) -> bool {
    row.revision == transition.behavior.revision
        && actions_equal_bits(&row.action, &transition.action)
        && transition
            .behavior
            .log_prob
            .is_some_and(|lp| lp.to_bits() == row.log_prob.to_bits())
}
fn detail_bytes(
    action: &Action,
    latent: Option<&[f32]>,
    distribution: &DistributionSnapshot,
) -> Result<usize> {
    let (floats, bytes) = match distribution {
        DistributionSnapshot::Categorical { logits, masks } => (logits.len(), masks.len()),
        DistributionSnapshot::Gaussian {
            mean,
            log_std,
            bounds,
        } => {
            let bounds = bounds
                .as_ref()
                .map_or(0, |(low, high)| low.len().saturating_add(high.len()));
            (
                mean.len()
                    .saturating_add(log_std.len())
                    .saturating_add(bounds),
                0,
            )
        }
        DistributionSnapshot::Conditional { outcomes } => {
            (0, size_of_val(outcomes.as_ref()) + 2 * size_of::<usize>())
        }
    };
    floats
        .checked_add(latent.map_or(0, <[f32]>::len))
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .and_then(|n| n.checked_add(cloned_action_bytes(action)))
        .and_then(|n| n.checked_add(bytes))
        .and_then(|n| n.checked_add(size_of::<PendingRow>()))
        .ok_or_else(|| Error::State("temporal detail payload size overflow".into()))
}

/// Check host values without constructing neural tensors. The recorded likelihood
/// remains unchanged; tolerance only accounts for scalar/tensor rounding.
fn validate_details(
    space: &ActionSpace,
    observation: &Observation,
    action: &Action,
    log_prob: f32,
    latent: Option<&[f32]>,
    distribution: &DistributionSnapshot,
    value: Option<f32>,
) -> Result<()> {
    distribution.validate_for(space)?;
    observation.check_action(space, action)?;
    ensure(
        log_prob.is_finite()
            && value.is_none_or(f32::is_finite)
            && latent.is_none_or(|v| v.iter().all(|v| v.is_finite())),
        "nonfinite temporal sampling metadata",
    )?;
    let expected = match (distribution, action) {
        (DistributionSnapshot::Categorical { logits, masks }, Action::Discrete(action)) => {
            ensure(
                latent.is_none()
                    && masks.get(*action) == Some(&true)
                    && observation
                        .action_mask()
                        .map_or_else(|| masks.iter().all(|m| *m), |mask| mask == masks),
                "categorical behavior support or latent differs from observation",
            )?;
            let maximum = logits
                .iter()
                .zip(masks)
                .filter(|(_, valid)| **valid)
                .map(|(v, _)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            let normalizer = logits
                .iter()
                .zip(masks)
                .filter(|(_, valid)| **valid)
                .map(|(v, _)| (*v - maximum).exp())
                .sum::<f32>()
                .ln();
            logits[*action] - maximum - normalizer
        }
        (
            DistributionSnapshot::Gaussian {
                mean,
                log_std,
                bounds,
            },
            Action::Continuous(actions),
        ) => {
            ensure(
                observation.action_mask().is_none(),
                "continuous behavior has a categorical mask",
            )?;
            let z = if bounds.is_some() {
                let z = latent.ok_or_else(|| {
                    Error::State("bounded Gaussian behavior is missing its original latent".into())
                })?;
                ensure(
                    z.len() == actions.len(),
                    "Gaussian latent dimensions differ",
                )?;
                z
            } else {
                ensure(
                    latent.is_none(),
                    "unbounded Gaussian behavior has an unexpected transformed latent",
                )?;
                actions.as_slice()
            };
            let mut result = 0.;
            for i in 0..mean.len() {
                let std = log_std[i].exp();
                ensure(
                    std.is_finite() && std > 0.,
                    "Gaussian behavior scale is not representable",
                )?;
                let normalized = (z[i] - mean[i]) / std;
                result +=
                    -0.5 * normalized * normalized - log_std[i] - 0.5 * std::f32::consts::TAU.ln();
                if let Some((low, high)) = bounds {
                    let scale = (high[i] - low[i]) * 0.5;
                    let expected = (low[i] + high[i]) * 0.5 + scale * z[i].tanh();
                    ensure(
                        expected.is_finite()
                            && (expected - actions[i]).abs()
                                <= 8. * f32::EPSILON * expected.abs().max(scale).max(1.),
                        "Gaussian action differs from its original latent transform",
                    )?;
                    let jacobian = 2.
                        * (std::f32::consts::LN_2 - z[i].abs() - (-2. * z[i].abs()).exp().ln_1p());
                    result -= scale.ln() + jacobian;
                }
            }
            result
        }
        (DistributionSnapshot::Conditional { outcomes }, Action::Conditional(action)) => {
            ensure(
                latent.is_none(),
                "conditional behavior has an unexpected latent",
            )?;
            distribution.validate_conditional_support(observation)?;
            outcomes
                .binary_search_by_key(action.as_ref(), |outcome| outcome.action)
                .ok()
                .and_then(|index| outcomes[index].log_prob)
                .ok_or_else(|| {
                    Error::Action("recorded action is outside conditional behavior support".into())
                })?
        }
        _ => {
            return Err(Error::Action(
                "temporal behavior distribution/action family differs".into(),
            ));
        }
    };
    ensure(
        expected.is_finite() && (expected - log_prob).abs() <= 1e-4 * expected.abs().max(1.),
        "temporal sampled likelihood differs from its immutable distribution",
    )
}
