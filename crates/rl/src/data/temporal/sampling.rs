//! Borrowed population selection and owned sequence filling over the existing store.
use super::{
    AnchorRequirements, Episode, EpisodeId, EpisodeStore, SampleLimits, SequenceBatch,
    SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep, TemporalStorage, VERSION,
    action_bytes, anchor_usage, cloned_observation_bytes, legacy_storage, reward_bytes,
};
use crate::{
    Error, Result,
    core::{
        action::Action, anchors::AnchorSet, checkpoint::Snapshot, ensure, env::Observation,
        stateful::PreviousStep,
    },
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::mem::size_of;

impl EpisodeStore {
    pub(super) fn sample_ranges<'a>(
        &'a self,
        config: &SequenceConfig,
        requirements: &AnchorRequirements,
        limits: &SampleLimits,
    ) -> Result<Vec<SampleRange<'a>>> {
        config.validate()?;
        requirements.validate()?;
        let mut ranges = Vec::new();
        let mut total = 0usize;
        for episode in self.episodes.values() {
            let mut origins = Vec::new();
            if episode.base_timestep == 0 {
                origins.push(0usize);
            }
            if matches!(self.storage, TemporalStorage::Segmented(_)) {
                for set in &episode.anchors {
                    if set.observation_index() > 0 && requirements.accepts(set) {
                        origins.push(set.observation_index() - episode.base_timestep);
                    }
                }
            }
            let eligible_end = if episode.ended() {
                episode.steps.len()
            } else {
                episode.steps.len().saturating_sub(config.n_step - 1)
            };
            for (index, &history) in origins.iter().enumerate() {
                let absolute_history = episode.base_timestep + history;
                let first = if absolute_history == 0 {
                    0
                } else {
                    history
                        .checked_add(config.burn_in)
                        .ok_or_else(|| Error::Config("temporal burn boundary overflow".into()))?
                }
                .max(episode.context_len());
                let mut end = eligible_end;
                if let Some(&next) = origins.get(index + 1) {
                    end = end.min(next.saturating_add(config.burn_in));
                }
                if let TemporalStorage::Segmented(segmented) = &self.storage {
                    end = end.min(
                        history
                            .saturating_add(config.burn_in)
                            .max(episode.context_len())
                            .saturating_add(segmented.segment_steps),
                    );
                }
                if first >= end {
                    continue;
                }
                let fits = |start: usize| -> Result<bool> {
                    let (_, context_end) = sequence_bounds(episode, start, config)?;
                    Ok(context_end - history < limits.max_sample_tokens
                        && (limits.max_sample_payload_bytes == usize::MAX
                            || window_payload_bytes(episode, history, context_end)
                                <= limits.max_sample_payload_bytes))
                };
                if !fits(first)? {
                    continue;
                }
                let mut low = first + 1;
                let mut high = end;
                while low < high {
                    let middle = low + (high - low) / 2;
                    if fits(middle)? {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
                end = low;
                total = total
                    .checked_add(end - first)
                    .ok_or_else(|| Error::Shape("temporal eligible population overflow".into()))?;
                ranges.push(SampleRange {
                    episode,
                    first,
                    end,
                    history,
                    total,
                });
            }
        }
        Ok(ranges)
    }
}

pub(super) struct SampleRange<'a> {
    episode: &'a Episode,
    first: usize,
    end: usize,
    history: usize,
    pub(super) total: usize,
}

fn window_payload_bytes(episode: &Episode, history: usize, context_end: usize) -> usize {
    let mut bytes = size_of::<SequenceSample>()
        + (context_end - history + 1) * size_of::<Observation>()
        + (context_end - history) * (size_of::<TemporalStep>() + size_of::<bool>());
    for observation in &episode.observations[history..=context_end] {
        bytes += cloned_observation_bytes(observation);
    }
    for step in &episode.steps[history..context_end] {
        bytes += action_bytes(&step.action) + reward_bytes(&step.reward);
    }
    if history > 0 {
        bytes += action_bytes(&episode.steps[history - 1].action);
    } else {
        bytes += episode
            .previous
            .as_ref()
            .map_or(0, |p| action_bytes(&p.action));
    }
    if let Some(set) = episode
        .anchors
        .iter()
        .find(|set| set.observation_index() == episode.base_timestep + history)
    {
        bytes += size_of::<AnchorSet>() + anchor_usage(set).resident_bytes;
    }
    bytes
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSampler {
    version: u32,
    rng: ChaCha8Rng,
}
impl SequenceSampler {
    pub fn sample_into_with(
        &mut self,
        store: &EpisodeStore,
        config: &SequenceConfig,
        batch_size: usize,
        requirements: &AnchorRequirements,
        limits: &SampleLimits,
        out: &mut SequenceBatch,
    ) -> Result<()> {
        if legacy_storage(&store.storage)
            && requirements.required.is_empty()
            && *limits == SampleLimits::default()
        {
            return self.sample_into(store, config, batch_size, out);
        }
        self.validate()?;
        ensure(batch_size > 0, "empty temporal sample batch")?;
        let minimum_payload = size_of::<SequenceSample>()
            + 2 * size_of::<Observation>()
            + size_of::<TemporalStep>()
            + size_of::<bool>()
            + 2 * size_of::<f32>();
        ensure(
            batch_size <= limits.max_batch_tokens
                && batch_size
                    .checked_mul(minimum_payload)
                    .is_some_and(|bytes| bytes <= limits.max_batch_payload_bytes),
            "temporal sample batch cannot fit its minimum token/payload budget",
        )?;
        let ranges = store.sample_ranges(config, requirements, limits)?;
        let total = ranges.last().map_or(0, |range| range.total);
        ensure(
            total > 0,
            "no temporal sequences satisfy anchor, context and resource requirements",
        )?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(batch_size)
            .map_err(|e| Error::State(format!("cannot reserve bounded temporal batch: {e}")))?;
        let mut rng = self.rng.clone();
        let mut maximum_tokens = 0;
        let mut payload_bytes = 0usize;
        for _ in 0..batch_size {
            let draw = rng.random_range(0..total);
            let index = ranges.partition_point(|range| range.total <= draw);
            let preceding = if index == 0 {
                0
            } else {
                ranges[index - 1].total
            };
            let range = &ranges[index];
            let start = range.first + draw - preceding;
            let bounds = sequence_bounds(range.episode, start, config)?;
            maximum_tokens = maximum_tokens.max(bounds.1 - range.history + 1);
            payload_bytes = payload_bytes
                .checked_add(window_payload_bytes(range.episode, range.history, bounds.1))
                .ok_or_else(|| Error::State("temporal batch payload overflow".into()))?;
            selected.push((range.episode, range.history, start, bounds));
        }
        ensure(
            maximum_tokens
                .checked_mul(batch_size)
                .is_some_and(|tokens| tokens <= limits.max_batch_tokens)
                && payload_bytes <= limits.max_batch_payload_bytes,
            "temporal sample batch exceeds resource limits",
        )?;
        out.samples
            .try_reserve(batch_size.saturating_sub(out.samples.len()))
            .map_err(|e| Error::State(format!("cannot reserve temporal samples: {e}")))?;
        out.samples.resize_with(batch_size, empty_sample);
        for (sample, (episode, history, start, bounds)) in out.samples.iter_mut().zip(selected) {
            fill_sample(sample, episode, history, start, config, bounds, false);
        }
        out.config = *config;
        self.rng = rng;
        Ok(())
    }
    pub fn new(seed: u64) -> Self {
        Self {
            version: VERSION,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == VERSION,
            "unsupported temporal sampler snapshot version",
        )
    }
    /// Sample uniformly over eligible transition starts, with replacement.
    /// Empty tapes and live tails lacking the requested target context are excluded.
    pub fn sample(
        &mut self,
        store: &EpisodeStore,
        config: &SequenceConfig,
        batch_size: usize,
    ) -> Result<SequenceBatch> {
        let mut out = SequenceBatch {
            samples: Vec::new(),
            config: *config,
        };
        self.sample_into(store, config, batch_size, &mut out)?;
        Ok(out)
    }

    /// Refill an owned batch, reusing nested history allocations. Replay never
    /// aliases the result. Recoverable errors leave both RNG and output unchanged.
    pub fn sample_into(
        &mut self,
        store: &EpisodeStore,
        config: &SequenceConfig,
        batch_size: usize,
        out: &mut SequenceBatch,
    ) -> Result<()> {
        if !legacy_storage(&store.storage) {
            return self.sample_into_with(
                store,
                config,
                batch_size,
                &AnchorRequirements::default(),
                &SampleLimits::default(),
                out,
            );
        }
        self.sample_indexed_into(store, config, batch_size, out, None)
    }

    /// Draw from the same indexed population while omitting the prefix that
    /// an observation-only learner would never consume. An oversized draw
    /// leaves both RNG and output unchanged.
    pub fn sample_stateless_into(
        &mut self,
        store: &EpisodeStore,
        config: &SequenceConfig,
        batch_size: usize,
        limits: &SampleLimits,
        out: &mut SequenceBatch,
    ) -> Result<()> {
        ensure(
            config.burn_in == 0 && limits.has_unbounded_samples(),
            "indexed stateless replay requires zero burn-in and batch-level limits",
        )?;
        self.sample_indexed_into(store, config, batch_size, out, Some(limits))
    }

    fn sample_indexed_into(
        &mut self,
        store: &EpisodeStore,
        config: &SequenceConfig,
        batch_size: usize,
        out: &mut SequenceBatch,
        stateless_limits: Option<&SampleLimits>,
    ) -> Result<()> {
        self.validate()?;
        config.validate()?;
        ensure(batch_size > 0, "empty temporal sample batch")?;
        let (total, tails) = store.population(config.n_step)?;
        ensure(
            total > 0,
            "no temporal sequences have complete target context",
        )?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(batch_size)
            .map_err(|e| Error::State(format!("cannot reserve temporal sample ranks: {e}")))?;
        let mut rng = self.rng.clone();
        let mut maximum_tokens = 0;
        let mut payload_bytes = 0usize;
        for _ in 0..batch_size {
            let draw = rng.random_range(0..total);
            let tail = tails.partition_point(|&(end, _)| end <= draw);
            let raw = draw + if tail == 0 { 0 } else { tails[tail - 1].1 };
            let (id, rank) = store
                .index
                .select(raw)
                .expect("eligible indexed temporal rank");
            let episode = store.episode(id).expect("indexed temporal episode");
            let start = rank + episode.context_len();
            let bounds = sequence_bounds(episode, start, config)?;
            if stateless_limits.is_some() {
                let tokens = bounds.1 - start + 1;
                let bytes = window_payload_bytes(episode, start, bounds.1);
                maximum_tokens = maximum_tokens.max(tokens);
                payload_bytes = payload_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| Error::State("temporal batch payload overflow".into()))?;
            }
            selected.push((episode, start, bounds));
        }
        if let Some(limits) = stateless_limits {
            ensure(
                maximum_tokens
                    .checked_mul(batch_size)
                    .is_some_and(|tokens| tokens <= limits.max_batch_tokens)
                    && payload_bytes <= limits.max_batch_payload_bytes,
                "stateless batch exceeds token or payload limit",
            )?;
        }
        out.samples
            .try_reserve(batch_size.saturating_sub(out.samples.len()))
            .map_err(|e| Error::State(format!("cannot reserve temporal samples: {e}")))?;
        out.samples.resize_with(batch_size, empty_sample);
        for (sample, (episode, start, bounds)) in out.samples.iter_mut().zip(selected) {
            let stateless = stateless_limits.is_some();
            fill_sample(
                sample,
                episode,
                if stateless { start } else { 0 },
                start,
                config,
                bounds,
                stateless,
            );
        }
        out.config = *config;
        self.rng = rng;
        Ok(())
    }
}
impl Snapshot for SequenceSampler {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate()
    }
}

/// Deterministic counterpart to `SequenceSampler::sample` for a chosen window.
pub fn sample_at(
    store: &EpisodeStore,
    episode_id: EpisodeId,
    learn_start: usize,
    config: &SequenceConfig,
) -> Result<SequenceSample> {
    config.validate()?;
    let episode = store
        .episode(episode_id)
        .ok_or_else(|| Error::State("temporal episode prefix is not retained".into()))?;
    if legacy_storage(&store.storage) {
        return sample_episode(episode, learn_start, config);
    }
    let ranges = store.sample_ranges(
        config,
        &AnchorRequirements::default(),
        &SampleLimits::default(),
    )?;
    let range = ranges
        .iter()
        .find(|range| {
            range.episode.id == episode_id
                && learn_start >= episode.base_timestep + range.first
                && learn_start < episode.base_timestep + range.end
        })
        .ok_or_else(|| {
            Error::State("temporal learning start has no usable retained anchor/context".into())
        })?;
    let local = learn_start - episode.base_timestep;
    let bounds = sequence_bounds(episode, local, config)?;
    let mut sample = empty_sample();
    fill_sample(
        &mut sample,
        episode,
        range.history,
        local,
        config,
        bounds,
        false,
    );
    Ok(sample)
}

fn sample_episode(
    episode: &Episode,
    learn_start: usize,
    config: &SequenceConfig,
) -> Result<SequenceSample> {
    let bounds = sequence_bounds(episode, learn_start, config)?;
    let mut sample = empty_sample();
    fill_sample(&mut sample, episode, 0, learn_start, config, bounds, false);
    Ok(sample)
}

fn sequence_bounds(
    episode: &Episode,
    learn_start: usize,
    config: &SequenceConfig,
) -> Result<(usize, usize)> {
    ensure(
        learn_start >= episode.context_len()
            && learn_start < episode.steps.len()
            && (episode.ended() || episode.steps.len() - learn_start >= config.n_step),
        "temporal learning start has incomplete target context",
    )?;
    let learn_end = learn_start
        .checked_add(config.learn_len)
        .ok_or_else(|| Error::Config("temporal learning boundary overflow".into()))?;
    let context_end = learn_end
        .checked_add(config.n_step - 1)
        .ok_or_else(|| Error::Config("temporal context boundary overflow".into()))?
        .min(episode.steps.len());
    Ok((learn_end, context_end))
}

fn empty_sample() -> SequenceSample {
    SequenceSample {
        episode_id: 0,
        trajectory_start: 0,
        learn_start: 0,
        burn_start: 0,
        observations: Vec::new(),
        steps: Vec::new(),
        loss_mask: Vec::new(),
        history_start: 0,
        previous: None,
        origin: SequenceOrigin::Reset,
    }
}

fn fill_sample(
    sample: &mut SequenceSample,
    episode: &Episode,
    history: usize,
    learn_start: usize,
    config: &SequenceConfig,
    (learn_end, context_end): (usize, usize),
    stateless: bool,
) {
    sample.episode_id = episode.id;
    sample.trajectory_start = episode.learning_start;
    sample.history_start = episode.base_timestep + history;
    sample.learn_start = learn_start - history;
    sample.burn_start = sample.learn_start.saturating_sub(config.burn_in);
    sample.previous = if history == 0 {
        episode.previous.clone()
    } else {
        let previous = &episode.steps[history - 1];
        Some(PreviousStep {
            action: previous.action.clone(),
            reward: previous.reward.total(),
        })
    };
    sample.origin = if stateless {
        SequenceOrigin::Stateless
    } else if sample.history_start == 0 {
        SequenceOrigin::Reset
    } else {
        SequenceOrigin::Anchored(
            episode
                .anchors
                .iter()
                .find(|set| set.observation_index() == sample.history_start)
                .expect("selected temporal anchor")
                .clone(),
        )
    };
    let observations = &episode.observations[history..=context_end];
    sample.observations.truncate(observations.len());
    for (existing, source) in sample.observations.iter_mut().zip(observations) {
        if let (Some(old), Some(new)) = (existing.as_dense_mut(), source.as_dense()) {
            old.values.clone_from(&new.values);
            old.action_mask.clone_from(&new.action_mask);
        } else {
            existing.clone_from(source);
        }
    }
    sample
        .observations
        .extend_from_slice(&observations[sample.observations.len()..]);
    let steps = &episode.steps[history..context_end];
    sample.steps.truncate(steps.len());
    for (existing, source) in sample.steps.iter_mut().zip(steps) {
        if let (Action::Continuous(old), Action::Continuous(new)) =
            (&mut existing.action, &source.action)
        {
            old.clone_from(new);
        } else {
            existing.action.clone_from(&source.action);
        }
        existing.reward.clone_from(&source.reward);
        existing.terminated = source.terminated;
        existing.truncated = source.truncated;
        existing.behavior.clone_from(&source.behavior);
    }
    sample.steps.extend_from_slice(&steps[sample.steps.len()..]);
    sample.loss_mask.clear();
    // Valid stored tapes have no internal end. Target context is therefore a
    // constant-time range check, rather than scanning n_step for each mask entry.
    let ended = context_end == episode.steps.len() && episode.ended();
    sample.loss_mask.extend(
        (history..context_end).map(|t| {
            t >= learn_start && t < learn_end && (ended || context_end - t >= config.n_step)
        }),
    );
}

#[cfg(test)]
mod stateless_tests {
    use super::*;
    use crate::core::{
        action::ActionSpace,
        env::{EpisodeStart, PublicHistory, ResumeContext},
        reward::Reward,
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    };

    #[test]
    fn compact_windows_preserve_draws_returns_and_the_restored_boundary() {
        let observation = |time| Observation::new(vec![time]);
        let action = Action::Discrete(0);
        let mut store = EpisodeStore::new(8, 1, ActionSpace::Discrete { count: 1 }).unwrap();
        let prefix = PublicHistory::new(observation(0.))
            .append(
                action.clone(),
                Reward::from_primitives(vec![1., 2., 3.]).unwrap(),
                observation(1.),
            )
            .unwrap();
        store
            .begin_episode(
                1,
                &EpisodeStart {
                    observation: observation(1.),
                    context: Some(ResumeContext {
                        physical_timestep: 3,
                        history: prefix,
                    }),
                },
            )
            .unwrap();
        for time in 1..=3 {
            store
                .push(&TemporalTransition {
                    episode_id: 1,
                    timestep: time,
                    observation: observation(time as f32),
                    action: action.clone(),
                    reward: Reward::from_primitives(vec![2., 1.]).unwrap(),
                    next_observation: observation((time + 1) as f32),
                    terminated: false,
                    truncated: time == 3,
                    behavior: BehaviorMetadata::default(),
                })
                .unwrap();
        }
        assert_eq!(store.len(), 3);
        assert_eq!(
            store.episode(1).unwrap().steps[0].reward.primitives(),
            &[1., 2., 3.]
        );
        let config = SequenceConfig {
            burn_in: 0,
            learn_len: 1,
            n_step: 2,
        };
        let mut full_sampler = SequenceSampler::new(21);
        let mut compact_sampler = full_sampler.clone();
        let full = full_sampler.sample(&store, &config, 12).unwrap();
        let mut compact = SequenceBatch {
            samples: vec![],
            config,
        };
        compact_sampler
            .sample_stateless_into(&store, &config, 12, &SampleLimits::default(), &mut compact)
            .unwrap();
        for (full, compact) in full.samples.iter().zip(&compact.samples) {
            assert_eq!(compact.origin, SequenceOrigin::Stateless);
            assert!(compact.absolute_learn_start() >= 1);
            assert_eq!(full.absolute_learn_start(), compact.absolute_learn_start());
            let full_target = full.target_at(full.learn_start, 0.5, 2).unwrap();
            let compact_target = compact.target_at(compact.learn_start, 0.5, 2).unwrap();
            assert_eq!(
                full_target.reward.to_bits(),
                compact_target.reward.to_bits()
            );
            assert_eq!(
                full_target.discount.to_bits(),
                compact_target.discount.to_bits()
            );
            assert!(
                full.observations[full_target.bootstrap_index]
                    .matches(&compact.observations[compact_target.bootstrap_index])
            );
        }
        compact.validate().unwrap();
    }
}
