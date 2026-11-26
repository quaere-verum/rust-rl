//! Bounded, opt-in categorical provenance alongside exact chronological replay.
//!
//! Uses the existing stateful collector and sequence sampler. Whole-episode
//! eviction is mirrored; segmented/anchored histories require a separate explicit
//! retention contract and are not accepted here. No ordinary replay schema changes.
use super::temporal::{EpisodeStore, SequenceBatch, SequenceOrigin, TemporalStep};
use crate::{
    Result,
    core::{
        action::ActionSpace,
        actor::ActMode,
        categorical_behavior::{
            BehaviorSource, CategoricalBehavior, CategoricalStatefulActor, validate_probabilities,
        },
        checkpoint::Snapshot,
        ensure,
        env::{EpisodeStart, Observation},
        recording::{RecordingKey, StatefulRecording},
        stateful::{HistorySpec, StatefulDecision},
        temporal::{TemporalSink, TemporalTransition},
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bound on additional dense probability/source columns. Sequence sampling has
/// its own independent token/byte limits.
pub const DEFAULT_BATCH_BYTES: usize = 64 * 1024 * 1024;

fn same_observation(a: &Observation, b: &Observation) -> bool {
    a.matches(b)
}
fn same_step(a: &TemporalStep, b: &TemporalStep) -> bool {
    a.action == b.action
        && a.reward.matches(&b.reward)
        && a.terminated == b.terminated
        && a.truncated == b.truncated
        && a.behavior.revision == b.behavior.revision
        && a.behavior.log_prob.map(f32::to_bits) == b.behavior.log_prob.map(f32::to_bits)
}

/// B,T,A distributions and B,T source identities. Actual steps include all
/// reconstruction prefixes; terminal endpoints and padding have no behavior.
pub struct CategoricalSequenceBatch {
    pub sequences: SequenceBatch,
    pub behavior: Vec<f64>,
    pub sources: Vec<Option<BehaviorSource>>,
}
impl CategoricalSequenceBatch {
    pub fn validate(&self, spec: &HistorySpec) -> Result<()> {
        spec.validate()?;
        self.sequences.validate()?;
        let actions = spec.action_width();
        ensure(
            matches!(spec.action_space, ActionSpace::Discrete { .. }),
            "categorical batch requires discrete actions",
        )?;
        let time = self
            .sequences
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let count = time
            .checked_mul(self.sequences.samples.len())
            .ok_or_else(|| crate::Error::Shape("categorical batch size overflow".into()))?;
        ensure(
            self.sources.len() == count && count.checked_mul(actions) == Some(self.behavior.len()),
            "categorical batch shape differs",
        )?;
        for (r, sample) in self.sequences.samples.iter().enumerate() {
            ensure(
                sample.history_start == 0
                    && sample.origin == SequenceOrigin::Reset
                    && sample.previous.is_none(),
                "categorical replay requires genuine reset ancestry",
            )?;
            sample.observations.last().unwrap().validate(
                spec.observation_dim,
                &spec.action_space,
                !sample.steps.last().unwrap().terminated,
            )?;
            for t in 0..time {
                let i = r * time + t;
                let p = &self.behavior[i * actions..(i + 1) * actions];
                if let Some(step) = sample.steps.get(t) {
                    sample.observations[t].validate(
                        spec.observation_dim,
                        &spec.action_space,
                        true,
                    )?;
                    ensure(self.sources[i].is_some(), "missing behavior source")?;
                    validate_probabilities(
                        &spec.action_space,
                        &sample.observations[t],
                        &step.action,
                        step.behavior.log_prob,
                        p,
                    )?;
                } else {
                    ensure(
                        self.sources[i].is_none() && p.iter().all(|p| *p == 0.),
                        "endpoint or padding invents behavior",
                    )?;
                }
            }
        }
        Ok(())
    }
}

struct Pending {
    observation: Observation,
    decision: StatefulDecision,
    behavior: CategoricalBehavior,
}

pub struct CategoricalReplay {
    version: u32,
    store: EpisodeStore,
    rows: BTreeMap<u64, Vec<CategoricalBehavior>>,
    max_staged: usize,
    staged: BTreeMap<RecordingKey, Pending>,
}
impl<'de> Deserialize<'de> for CategoricalReplay {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Record {
            version: u32,
            store: EpisodeStore,
            rows: BTreeMap<u64, Vec<CategoricalBehavior>>,
            max_staged: usize,
        }
        let r = Record::deserialize(deserializer)?;
        let replay = Self {
            version: r.version,
            store: r.store,
            rows: r.rows,
            max_staged: r.max_staged,
            staged: BTreeMap::new(),
        };
        replay
            .validate_snapshot()
            .map_err(serde::de::Error::custom)?;
        Ok(replay)
    }
}
impl Serialize for CategoricalReplay {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.validate_snapshot()
            .map_err(serde::ser::Error::custom)?;
        #[derive(Serialize)]
        struct Record<'a> {
            version: u32,
            store: &'a EpisodeStore,
            rows: &'a BTreeMap<u64, Vec<CategoricalBehavior>>,
            max_staged: usize,
        }
        Record {
            version: self.version,
            store: &self.store,
            rows: &self.rows,
            max_staged: self.max_staged,
        }
        .serialize(serializer)
    }
}
impl CategoricalReplay {
    pub fn new(
        capacity: usize,
        width: usize,
        space: ActionSpace,
        max_staged: usize,
    ) -> Result<Self> {
        ensure(
            matches!(space, ActionSpace::Discrete { .. })
                && max_staged > 0
                && max_staged <= capacity,
            "invalid categorical recording capacity or action space",
        )?;
        let result = Self {
            version: 1,
            store: EpisodeStore::new(capacity, width, space)?,
            rows: BTreeMap::new(),
            max_staged,
            staged: BTreeMap::new(),
        };
        result.validate_snapshot()?;
        Ok(result)
    }
    pub fn store(&self) -> &EpisodeStore {
        &self.store
    }
    pub fn recorded_rows(&self) -> usize {
        self.rows.values().map(Vec::len).sum()
    }
    pub fn row(&self, key: RecordingKey) -> Option<&CategoricalBehavior> {
        self.rows.get(&key.episode_id)?.get(key.timestep)
    }
    // ExactEpisodes evicts a prefix in increasing episode order. Explicit
    // abandonment below handles interior removal without scanning retained rows.
    fn prune(&mut self) {
        while self
            .rows
            .first_key_value()
            .is_some_and(|(id, _)| self.store.episode(*id).is_none())
        {
            self.rows.pop_first();
        }
    }
    pub fn batch(&self, sequences: SequenceBatch) -> Result<CategoricalSequenceBatch> {
        self.batch_with_limit(sequences, DEFAULT_BATCH_BYTES)
    }
    pub fn batch_with_limit(
        &self,
        sequences: SequenceBatch,
        max_additional_bytes: usize,
    ) -> Result<CategoricalSequenceBatch> {
        ensure(
            self.staged.is_empty(),
            "cannot sample partially committed categorical recording",
        )?;
        sequences.validate()?;
        let time = sequences
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let ActionSpace::Discrete { count: actions } = self.store.action_space() else {
            return Err(crate::Error::Action(
                "categorical replay requires discrete actions".into(),
            ));
        };
        let actions = *actions;
        let count = time
            .checked_mul(sequences.samples.len())
            .ok_or_else(|| crate::Error::Shape("categorical batch size overflow".into()))?;
        let width = count
            .checked_mul(actions)
            .ok_or_else(|| crate::Error::Shape("categorical batch size overflow".into()))?;
        let bytes = width.checked_mul(size_of::<f64>()).and_then(|n| {
            count
                .checked_mul(size_of::<Option<BehaviorSource>>())
                .and_then(|m| n.checked_add(m))
        });
        ensure(
            bytes.is_some_and(|n| n <= max_additional_bytes),
            "categorical batch exceeds byte limit",
        )?;
        let mut behavior = Vec::new();
        let mut sources = Vec::new();
        behavior.try_reserve_exact(width).map_err(|e| {
            crate::Error::State(format!("cannot reserve categorical probabilities: {e}"))
        })?;
        sources
            .try_reserve_exact(count)
            .map_err(|e| crate::Error::State(format!("cannot reserve categorical sources: {e}")))?;
        behavior.resize(width, 0.);
        sources.resize(count, None);
        for (r, sample) in sequences.samples.iter().enumerate() {
            let episode = self.store.episode(sample.episode_id).ok_or_else(|| {
                crate::Error::State("categorical sample episode was evicted".into())
            })?;
            ensure(
                sample.history_start == 0
                    && sample.origin == SequenceOrigin::Reset
                    && sample.previous.is_none()
                    && episode
                        .observations
                        .get(..sample.observations.len())
                        .is_some_and(|rows| {
                            rows.iter()
                                .zip(&sample.observations)
                                .all(|(a, b)| same_observation(a, b))
                        })
                    && episode.steps.get(..sample.steps.len()).is_some_and(|rows| {
                        rows.iter().zip(&sample.steps).all(|(a, b)| same_step(a, b))
                    }),
                "categorical sample differs from its factual retained history",
            )?;
            for t in 0..sample.steps.len() {
                let row = self
                    .row(RecordingKey {
                        episode_id: sample.episode_id,
                        timestep: t,
                    })
                    .ok_or_else(|| {
                        crate::Error::State("missing retained categorical behavior".into())
                    })?;
                ensure(
                    row.revision == sample.steps[t].behavior.revision
                        && row.probabilities.len() == actions,
                    "categorical behavior revision differs",
                )?;
                let i = r * time + t;
                behavior[i * actions..(i + 1) * actions].copy_from_slice(&row.probabilities);
                sources[i] = Some(row.source);
            }
        }
        let batch = CategoricalSequenceBatch {
            sequences,
            behavior,
            sources,
        };
        batch.validate(&HistorySpec::new(
            self.store.observation_space().require_dense_width()?,
            self.store.action_space().clone(),
        )?)?;
        Ok(batch)
    }
}
impl Snapshot for CategoricalReplay {
    fn validate_snapshot(&self) -> Result<()> {
        self.store.validate_snapshot()?;
        ensure(
            self.version == 1
                && matches!(
                    self.store.storage(),
                    super::temporal::TemporalStorage::ExactEpisodes
                )
                && self.store.dense_observation_dim().is_some()
                && matches!(self.store.action_space(), ActionSpace::Discrete { .. })
                && self.max_staged > 0
                && self.max_staged <= self.store.capacity()
                && self.staged.is_empty()
                && self.rows.len() == self.store.episodes().len(),
            "invalid categorical replay snapshot",
        )?;
        for episode in self.store.episodes() {
            let rows = self
                .rows
                .get(&episode.id)
                .ok_or_else(|| crate::Error::State("missing categorical episode".into()))?;
            ensure(
                rows.len() == episode.steps.len(),
                "categorical row count differs from tape",
            )?;
            for (t, (row, step)) in rows.iter().zip(&episode.steps).enumerate() {
                ensure(
                    row.revision == step.behavior.revision,
                    "categorical source revision differs",
                )?;
                row.validate(
                    self.store.action_space(),
                    &episode.observations[t],
                    &StatefulDecision {
                        action: step.action.clone(),
                        log_prob: step.behavior.log_prob,
                    },
                )?;
            }
        }
        Ok(())
    }
}
impl TemporalSink for CategoricalReplay {
    fn begin_episode(&mut self, id: u64, start: &EpisodeStart) -> Result<()> {
        ensure(
            start.context.is_none(),
            "categorical replay requires a fresh episode start",
        )?;
        self.store.begin_episode(id, start)?;
        self.rows.insert(id, Vec::new());
        Ok(())
    }
    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        let key = RecordingKey {
            episode_id: transition.episode_id,
            timestep: transition.timestep,
        };
        let pending = self.staged.get(&key).ok_or_else(|| {
            crate::Error::State("transition lacks same-call categorical sampling record".into())
        })?;
        ensure(
            same_observation(&pending.observation, &transition.observation)
                && pending.decision.action == transition.action
                && pending.decision.log_prob.map(f32::to_bits)
                    == transition.behavior.log_prob.map(f32::to_bits)
                && pending.behavior.revision == transition.behavior.revision,
            "transition differs from its categorical sampling record",
        )?;
        if self.store.episode(key.episode_id).is_some() {
            ensure(
                self.rows
                    .get(&key.episode_id)
                    .is_some_and(|r| r.len() == key.timestep),
                "categorical recording clock differs",
            )?;
            self.rows
                .get_mut(&key.episode_id)
                .unwrap()
                .try_reserve(1)
                .map_err(|e| crate::Error::State(format!("cannot reserve categorical row: {e}")))?;
        }
        self.store.push(transition)?;
        let pending = self.staged.remove(&key).unwrap();
        self.prune();
        if self.store.episode(key.episode_id).is_some() {
            self.rows
                .get_mut(&key.episode_id)
                .unwrap()
                .push(pending.behavior);
        }
        Ok(())
    }
    fn abandon_episode(&mut self, id: u64) -> Result<()> {
        self.store.abandon_episode(id)?;
        // Completed tapes survive reset_all and remain valid replay data.
        if self.store.episode(id).is_none() {
            self.rows.remove(&id);
        }
        self.staged.retain(|key, _| key.episode_id != id);
        Ok(())
    }
}
impl<A: CategoricalStatefulActor> StatefulRecording<A> for CategoricalReplay {
    fn decide(
        &mut self,
        key: RecordingKey,
        actor: &A,
        state: &A::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        ensure(
            mode == ActMode::Sample,
            "categorical provenance records stochastic collection only",
        )?;
        ensure(
            !self.staged.contains_key(&key) && self.staged.len() < self.max_staged,
            "duplicate or excessive staged categorical decision",
        )?;
        ensure(
            &actor.observation_space() == self.store.observation_space()
                && actor.action_space() == self.store.action_space()
                && self.store.active_timestep(key.episode_id) == Some(key.timestep),
            "categorical actor or lane clock differs",
        )?;
        observation.validate_in(
            self.store.observation_space(),
            self.store.action_space(),
            true,
        )?;
        if let Some(episode) = self.store.episode(key.episode_id) {
            ensure(
                episode
                    .observations
                    .last()
                    .is_some_and(|recorded| same_observation(recorded, observation)),
                "categorical sampling observation differs from factual tape",
            )?;
        }
        let sample = actor.sample_with_behavior(state, observation, rng)?;
        sample
            .behavior
            .validate(actor.action_space(), observation, &sample.decision)?;
        ensure(
            sample.behavior.source == actor.behavior_source()
                && sample.behavior.revision == actor.revision(),
            "categorical sample source differs from actor",
        )?;
        let decision = sample.decision.clone();
        self.staged.insert(
            key,
            Pending {
                observation: observation.clone(),
                decision: sample.decision,
                behavior: sample.behavior,
            },
        );
        Ok(decision)
    }
    fn push_with_state(
        &mut self,
        transition: &TemporalTransition,
        actor: &A,
        _: &A::State,
    ) -> Result<()> {
        let key = RecordingKey {
            episode_id: transition.episode_id,
            timestep: transition.timestep,
        };
        ensure(
            self.staged.get(&key).is_some_and(|p| {
                p.behavior.source == actor.behavior_source()
                    && p.behavior.revision == actor.revision()
            }),
            "committed categorical actor differs from sampled actor",
        )?;
        self.push(transition)
    }
}
