use crate::{
    Error, Result,
    core::{
        action::Action,
        env::{Observation, ObservationBatch},
        stateful::HistorySpec,
        stateful_on_policy::{DistributionSnapshot, StatefulOnPolicyActor},
        temporal::TemporalExecutionMode,
    },
    data::{
        temporal::SequenceSample,
        temporal_on_policy::{TemporalOnPolicyBatch, TemporalRow},
    },
    nn::{
        distributions::Distribution,
        temporal_actor::TemporalActor,
        temporal_batch::EpisodeInputCache,
        temporal_models::{PreparedHistory, TemporalBackbone, selected_features},
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_q::TemporalQ,
        temporal_value::{TemporalStateValue, TemporalValue},
        temporal_work::EpisodeWindowPlan,
    },
};
use burn::tensor::{Device, Tensor};
use std::borrow::Cow;

/// Immutable factual histories and ordered learning rows selected for one update.
pub struct Minibatch<'a, A: Clone = TemporalActor>(Selected<'a, A>);

enum Selected<'a, A: Clone> {
    Reference(Cow<'a, TemporalOnPolicyBatch<A>>),
    Cached {
        history: History,
        samples: Vec<&'a SequenceSample>,
        rows: Vec<(usize, &'a TemporalRow)>,
    },
}

enum History {
    Packed(Box<PreparedHistory>),
    Coalesced(Box<EpisodeWindowPlan>),
}

impl History {
    fn features(
        &self,
        encoder: &TemporalBackbone,
        samples: &[&SequenceSample],
        tokens: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>> {
        match self {
            Self::Packed(history) => {
                history.features_on(encoder, samples.iter().copied(), tokens, device)
            }
            Self::Coalesced(plan) => plan.features(
                encoder.gru.as_ref().expect("coalescing requires a GRU"),
                tokens,
                device,
            ),
        }
    }
}

/// A validated rollout and its optional immutable history inputs. Sampling
/// belongs to the caller, which supplies each minibatch's original lane order.
pub struct WindowBatch<'a, A = TemporalActor> {
    batch: &'a TemporalOnPolicyBatch<A>,
    cache: Option<(EpisodeInputCache<'a>, [usize; 2])>,
}

impl<'a, A: StatefulOnPolicyActor + Clone> WindowBatch<'a, A> {
    /// Use the rollout's original factual histories without an encoded cache.
    pub fn reference(batch: &'a TemporalOnPolicyBatch<A>) -> Self {
        Self { batch, cache: None }
    }

    pub(crate) fn with_layout(
        batch: &'a TemporalOnPolicyBatch<A>,
        mode: TemporalExecutionMode,
        spec: &HistorySpec,
        widths: Option<[usize; 2]>,
    ) -> Result<Self> {
        let cache = if let Some(widths) = widths.filter(|_| !mode.is_reference()) {
            #[cfg(feature = "probes")]
            let _profile = crate::profiling::enter(crate::profiling::Phase::HistoryInputCache);
            Some((
                EpisodeInputCache::from_validated_on_policy(&batch.sequences, spec)?,
                widths,
            ))
        } else {
            None
        };
        Ok(Self { batch, cache })
    }

    pub fn window_count(&self) -> usize {
        self.batch.sequences.samples.len()
    }

    pub(crate) fn full(&self) -> Result<Minibatch<'a, A>> {
        if self.cache.is_some() {
            self.select(&(0..self.window_count()).collect::<Vec<_>>())
        } else {
            Ok(Minibatch(Selected::Reference(Cow::Borrowed(self.batch))))
        }
    }

    pub(crate) fn select(&self, indices: &[usize]) -> Result<Minibatch<'a, A>> {
        let batch = self.batch;
        if let Some((cache, widths)) = &self.cache {
            let samples: Vec<_> = indices
                .iter()
                .map(|&i| &batch.sequences.samples[i])
                .collect();
            let history = {
                #[cfg(feature = "probes")]
                let _profile = crate::profiling::enter(crate::profiling::Phase::HistoryPacking);
                let device = crate::nn::device();
                match EpisodeWindowPlan::try_new(cache, &samples, *widths, &device)? {
                    Some(plan) => History::Coalesced(Box::new(plan)),
                    None => History::Packed(Box::new(PreparedHistory::new(
                        cache.pack(indices)?,
                        None,
                        &device,
                    ))),
                }
            };
            Ok(Minibatch(Selected::Cached {
                history,
                samples,
                rows: batch.validated_window_rows(indices),
            }))
        } else {
            Ok(Minibatch(Selected::Reference(Cow::Owned(
                batch.select_windows(indices)?,
            ))))
        }
    }
}

impl<A: StatefulOnPolicyActor + Clone> Minibatch<'_, A> {
    pub(crate) fn len(&self) -> usize {
        match &self.0 {
            Selected::Reference(batch) => batch.len(),
            Selected::Cached { rows, .. } => rows.len(),
        }
    }

    /// Learning rows in update order, paired with their minibatch-local sample index.
    pub fn rows(&self) -> impl Iterator<Item = (usize, &TemporalRow)> {
        let (owned, borrowed) = match &self.0 {
            Selected::Reference(batch) => (batch.rows.as_slice(), &[][..]),
            Selected::Cached { rows, .. } => (&[][..], rows.as_slice()),
        };
        owned
            .iter()
            .map(|row| (row.sample, row))
            .chain(borrowed.iter().copied())
    }

    /// Minibatch-local sample and time for each learning row.
    pub fn tokens(&self) -> Vec<(usize, usize)> {
        self.rows()
            .map(|(sample, row)| (sample, row.time))
            .collect()
    }

    /// Read a factual history by its minibatch-local index.
    ///
    /// Panics when the index is out of bounds.
    pub fn sample(&self, index: usize) -> &SequenceSample {
        match &self.0 {
            Selected::Reference(batch) => &batch.sequences.samples[index],
            Selected::Cached { samples, .. } => samples[index],
        }
    }

    /// Factual histories in minibatch-local order, including their reset prefixes.
    pub fn samples(&self) -> impl ExactSizeIterator<Item = &SequenceSample> {
        let count = match &self.0 {
            Selected::Reference(batch) => batch.sequences.samples.len(),
            Selected::Cached { samples, .. } => samples.len(),
        };
        (0..count).map(|i| self.sample(i))
    }

    pub(crate) fn observations(&self, tokens: &[(usize, usize)]) -> Vec<&Observation> {
        tokens
            .iter()
            .map(|&(sample, time)| &self.sample(sample).observations[time])
            .collect()
    }

    pub(crate) fn actions(&self) -> Vec<Action> {
        match &self.0 {
            Selected::Reference(batch) => batch.actions(),
            Selected::Cached { samples, .. } => self
                .rows()
                .map(|(sample, row)| samples[sample].steps[row.time].action.clone())
                .collect(),
        }
    }

    pub(crate) fn latents(&self) -> Vec<Option<Vec<f32>>> {
        self.rows().map(|(_, row)| row.latent.clone()).collect()
    }

    pub(crate) fn old_log_probs(&self) -> Vec<f32> {
        self.rows().map(|(_, row)| row.log_prob).collect()
    }

    pub(crate) fn behavior_distributions(&self) -> Vec<DistributionSnapshot> {
        self.rows()
            .map(|(_, row)| row.distribution.clone())
            .collect()
    }

    pub(crate) fn advantages(&self) -> Result<Vec<f32>> {
        self.rows()
            .map(|(_, row)| {
                row.advantage
                    .ok_or_else(|| Error::State("temporal batch has no advantage target".into()))
            })
            .collect()
    }

    pub(crate) fn returns(&self) -> Result<Vec<f32>> {
        self.rows()
            .map(|(_, row)| {
                row.return_
                    .ok_or_else(|| Error::State("temporal batch has no return target".into()))
            })
            .collect()
    }

    pub(crate) fn distribution(
        &self,
        policy: &TemporalPolicy,
        tokens: &[(usize, usize)],
    ) -> Result<Distribution> {
        match &self.0 {
            Selected::Reference(batch) => policy.distribution(&batch.sequences, tokens),
            Selected::Cached {
                history, samples, ..
            } => {
                let features =
                    history.features(&policy.encoder, samples, tokens, &policy.device())?;
                let observations = self.observations(tokens);
                policy.distribution_from_features(
                    features,
                    ObservationBatch::from_refs(&observations),
                )
            }
        }
    }

    pub(crate) fn policy_history(
        &self,
        policy: &TemporalPolicy,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>> {
        match &self.0 {
            Selected::Reference(batch) => selected_features(
                &policy.encoder,
                &batch.sequences,
                policy.spec(),
                policy.observation_only(),
                &policy.device(),
                tokens,
            ),
            Selected::Cached {
                history, samples, ..
            } => history.features(&policy.encoder, samples, tokens, &policy.device()),
        }
    }

    pub(crate) fn q_history(
        &self,
        critic: &TemporalQ,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>> {
        match &self.0 {
            Selected::Reference(batch) => critic.history_features(&batch.sequences, tokens),
            Selected::Cached {
                history, samples, ..
            } => history.features(&critic.encoder, samples, tokens, &critic.device()),
        }
    }

    pub(crate) fn values(
        &self,
        value: &TemporalValue,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<1>> {
        match &self.0 {
            Selected::Reference(batch) => value.values(&batch.sequences, tokens),
            Selected::Cached {
                history, samples, ..
            } => {
                let features =
                    history.features(&value.encoder, samples, tokens, &value.device())?;
                value.values_from_features(features)
            }
        }
    }
}
