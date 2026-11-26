//! Pack owned episode-prefix samples into the canonical numerical history layout.
use super::temporal::TemporalMasks;
use crate::{
    Error, Result,
    core::stateful::{HistoryInput, HistorySpec, PreviousInput},
    data::temporal::{SequenceBatch, SequenceConfig, SequenceOrigin, SequenceSample},
};
use std::collections::{BTreeMap, btree_map::Entry};

/// Batch-major `[batch, time, input_dim]` history features and token controls.
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedHistory {
    pub batch: usize,
    pub time: usize,
    pub input_dim: usize,
    pub inputs: Vec<f32>,
    pub masks: TemporalMasks,
    pub learning: Vec<bool>,
}

impl EncodedHistory {
    pub(crate) fn zeroed(batch: usize, time: usize, input_dim: usize) -> Result<Self> {
        let tokens = batch
            .checked_mul(time)
            .ok_or_else(|| Error::Shape("history batch token count overflow".into()))?;
        let elements = tokens
            .checked_mul(input_dim)
            .ok_or_else(|| Error::Shape("history batch feature count overflow".into()))?;
        Ok(Self {
            batch,
            time,
            input_dim,
            inputs: filled(elements, 0.)?,
            masks: TemporalMasks {
                batch,
                time,
                valid: filled(tokens, false)?,
                reset: filled(tokens, false)?,
                detach: filled(tokens, false)?,
            },
            learning: filled(tokens, false)?,
        })
    }

    fn fill_controls(
        &mut self,
        row: usize,
        sample: &SequenceSample,
        config: SequenceConfig,
    ) -> Result<()> {
        let learn_end = sample
            .learn_start
            .checked_add(config.learn_len)
            .ok_or_else(|| Error::Shape("history learning boundary overflow".into()))?;
        for t in 0..sample.observations.len() {
            let token = row * self.time + t;
            self.masks.valid[token] = true;
            self.masks.reset[token] = t == 0 && matches!(sample.origin, SequenceOrigin::Reset);
            self.masks.detach[token] = t <= sample.learn_start || t >= learn_end;
            self.learning[token] = sample.loss_mask.get(t).copied().unwrap_or(false);
        }
        Ok(())
    }
}

/// Encode the actual public history, including previous input at a retained
/// boundary. Only a genuine reset origin emits a reset mask; an anchor is a
/// pre-observation prior supplied separately to the network.
///
/// Prefix tokens and the first learning token detach their incoming state. The
/// latter is essential because detach occurs before assimilation: it severs the
/// final prefix operation's ancestry. Target lookahead detaches as well. Padding
/// has zero features and no valid, reset, detach, or learning controls.
pub fn encode_history_batch(batch: &SequenceBatch, spec: &HistorySpec) -> Result<EncodedHistory> {
    spec.validate()?;
    batch.validate()?;
    let batch_size = batch.samples.len();
    let time = batch
        .samples
        .iter()
        .map(|sample| sample.observations.len())
        .max()
        .expect("validated nonempty history batch");
    let input_dim = spec.input_dim();
    let mut out = EncodedHistory::zeroed(batch_size, time, input_dim)?;
    for (row, sample) in batch.samples.iter().enumerate() {
        let start = row * time * input_dim;
        let end = start + sample.observations.len() * input_dim;
        encode_sample(sample, spec, &mut out.inputs[start..end])?;
        out.fill_controls(row, sample, batch.config)?;
    }
    out.masks.validate()?;
    Ok(out)
}

fn encode_sample(sample: &SequenceSample, spec: &HistorySpec, inputs: &mut [f32]) -> Result<()> {
    for (t, (observation, dest)) in sample
        .observations
        .iter()
        .zip(inputs.chunks_exact_mut(spec.input_dim()))
        .enumerate()
    {
        let previous_step = t.checked_sub(1).map(|i| &sample.steps[i]);
        let previous = previous_step
            .map(|step| PreviousInput::from_total(&step.action, step.reward.total()))
            .or_else(|| {
                sample
                    .previous
                    .as_ref()
                    .filter(|_| t == 0)
                    .map(|p| PreviousInput::from_total(&p.action, p.reward))
            });
        observation.validate(
            spec.observation_dim,
            &spec.action_space,
            previous_step.is_none_or(|step| !step.terminated),
        )?;
        if let Some(step) = sample.steps.get(t) {
            spec.action_space
                .check(&step.action, observation.dense_mask())?;
        }
        dest.copy_from_slice(&spec.encode(HistoryInput {
            observation,
            previous,
        })?);
    }
    Ok(())
}

/// Parameter-independent encoding of the longest validated prefix per episode.
/// The borrowed source keeps the sample identities and all controls immutable
/// for the cache's lifetime. Each packed minibatch still has its own row order.
pub(crate) struct EpisodeInputCache<'a> {
    batch: &'a SequenceBatch,
    tapes: BTreeMap<u64, Vec<f32>>,
    input_dim: usize,
}

impl<'a> EpisodeInputCache<'a> {
    /// The on-policy batch audit has checked genuine reset origins and bitwise
    /// agreement of overlapping observations, actions and rewards per episode.
    pub(crate) fn from_validated_on_policy(
        batch: &'a SequenceBatch,
        spec: &HistorySpec,
    ) -> Result<Self> {
        let mut longest = BTreeMap::<u64, &SequenceSample>::new();
        for sample in &batch.samples {
            let tape = longest.entry(sample.episode_id).or_insert(sample);
            if sample.observations.len() > tape.observations.len() {
                *tape = sample;
            }
        }
        Self::from_longest(batch, spec, longest)
    }

    /// Replay sampling may repeat windows and is not an on-policy audit.
    /// Prove equality of the overlap before sharing a genuine reset prefix.
    pub(crate) fn try_from_validated_replay(
        batch: &'a SequenceBatch,
        spec: &HistorySpec,
    ) -> Result<Option<Self>> {
        let mut longest = BTreeMap::<u64, &SequenceSample>::new();
        for sample in &batch.samples {
            if !matches!(sample.origin, SequenceOrigin::Reset)
                || sample.history_start != 0
                || sample.previous.is_some()
            {
                return Ok(None);
            }
            match longest.entry(sample.episode_id) {
                Entry::Vacant(entry) => {
                    entry.insert(sample);
                }
                Entry::Occupied(mut entry) => {
                    let other = entry.get();
                    if !sample.factual_prefixes_equal(other) {
                        return Ok(None);
                    }
                    if sample.observations.len() > other.observations.len() {
                        entry.insert(sample);
                    }
                }
            }
        }
        if longest.len() == batch.samples.len() {
            return Ok(None);
        }
        Self::from_longest(batch, spec, longest).map(Some)
    }

    fn from_longest(
        batch: &'a SequenceBatch,
        spec: &HistorySpec,
        longest: BTreeMap<u64, &SequenceSample>,
    ) -> Result<Self> {
        spec.validate()?;
        let input_dim = spec.input_dim();
        let mut tapes = BTreeMap::new();
        for (id, sample) in longest {
            let size = sample
                .observations
                .len()
                .checked_mul(input_dim)
                .ok_or_else(|| Error::Shape("episode encoding size overflow".into()))?;
            let mut tape = filled(size, 0.)?;
            encode_sample(sample, spec, &mut tape)?;
            tapes.insert(id, tape);
        }
        Ok(Self {
            batch,
            tapes,
            input_dim,
        })
    }

    pub(crate) fn pack(&self, indices: &[usize]) -> Result<EncodedHistory> {
        let time = indices
            .iter()
            .map(|&index| self.batch.samples[index].observations.len())
            .max()
            .expect("nonempty validated window selection");
        let mut out = EncodedHistory::zeroed(indices.len(), time, self.input_dim)?;
        for (row, &index) in indices.iter().enumerate() {
            let sample = &self.batch.samples[index];
            let size = sample.observations.len() * self.input_dim;
            let start = row * time * self.input_dim;
            out.inputs[start..start + size]
                .copy_from_slice(&self.tapes[&sample.episode_id][..size]);
            out.fill_controls(row, sample, self.batch.config)?;
        }
        Ok(out)
    }

    pub(crate) fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub(crate) fn learn_len(&self) -> usize {
        self.batch.config.learn_len
    }

    pub(crate) fn tape(&self, episode_id: u64) -> &[f32] {
        &self.tapes[&episode_id]
    }

    pub(crate) fn samples(&self) -> &[SequenceSample] {
        &self.batch.samples
    }

    pub(crate) fn tapes(&self) -> impl ExactSizeIterator<Item = (u64, &[f32])> {
        self.tapes.iter().map(|(&id, tape)| (id, tape.as_slice()))
    }
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| Error::State(format!("cannot reserve encoded history batch: {e}")))?;
    values.resize(len, value);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{
            action::{Action, ActionSpace},
            env::Observation,
        },
        data::temporal::{BehaviorMetadata, TemporalStep},
    };

    #[test]
    fn cached_reset_tapes_match_exact_encoding_for_reordered_windows() {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let config = SequenceConfig {
            burn_in: 2,
            learn_len: 2,
            n_step: 1,
        };
        let sample = |episode_id, learn_start: usize, end, truncated| SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: SequenceOrigin::Reset,
            episode_id,
            learn_start,
            burn_start: learn_start.saturating_sub(2),
            observations: (0..=end)
                .map(|t| Observation::dense(vec![t as f32], Some(vec![true, t % 2 == 1])))
                .collect(),
            steps: (0..end)
                .map(|t| TemporalStep {
                    action: Action::Discrete(t % 2),
                    reward: (if t == 0 { -0. } else { t as f32 }).into(),
                    terminated: false,
                    truncated: truncated && t + 1 == end,
                    behavior: BehaviorMetadata {
                        revision: 0,
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: (0..end).map(|t| t >= learn_start).collect(),
        };
        let batch = SequenceBatch {
            config,
            samples: vec![
                sample(7, 0, 2, false),
                sample(7, 2, 4, false),
                sample(8, 0, 1, true),
            ],
        };
        batch.validate().unwrap();
        let cache = EpisodeInputCache::from_validated_on_policy(&batch, &spec).unwrap();
        assert_eq!(cache.tapes.len(), 2);
        let indices = [2, 1, 0];
        let packed = cache.pack(&indices).unwrap();
        let reference = encode_history_batch(
            &SequenceBatch {
                config,
                samples: indices.iter().map(|&i| batch.samples[i].clone()).collect(),
            },
            &spec,
        )
        .unwrap();
        assert_eq!(packed, reference);
        assert!(
            packed
                .inputs
                .iter()
                .map(|v| v.to_bits())
                .eq(reference.inputs.iter().map(|v| v.to_bits()))
        );
        assert_eq!(packed.inputs[8].to_bits(), (-0.0f32).to_bits());
    }
}
