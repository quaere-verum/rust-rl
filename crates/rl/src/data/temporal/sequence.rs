//! Owned factual sequence records, validation, return targets and padded layouts.
use super::{EpisodeId, SequenceOrigin, TemporalStep, reset, zero};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::Snapshot,
        ensure,
        env::{Observation, ObservationSpace},
        reward::Reward,
        stateful::{HistoryInput, PreviousInput, PreviousStep},
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceConfig {
    pub burn_in: usize,
    pub learn_len: usize,
    pub n_step: usize,
}
impl SequenceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.learn_len > 0 && self.n_step > 0,
            "temporal learning length and n-step horizon must be positive",
        )?;
        self.learn_len
            .checked_add(self.n_step - 1)
            .ok_or_else(|| Error::Config("temporal sequence length overflow".into()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NStepTarget {
    pub reward: f32,
    pub discount: f32,
    pub bootstrap_index: usize,
}

/// Array indices are local to this owned slice; `history_start` is absolute.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceSample {
    pub episode_id: EpisodeId,
    /// Absolute start of this trajectory's newly collected experience.
    #[serde(default, skip_serializing_if = "zero")]
    pub trajectory_start: usize,
    pub learn_start: usize,
    pub burn_start: usize,
    pub observations: Vec<Observation>,
    pub steps: Vec<TemporalStep>,
    pub loss_mask: Vec<bool>,
    #[serde(default, skip_serializing_if = "zero")]
    pub history_start: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<PreviousStep>,
    #[serde(default, skip_serializing_if = "reset")]
    pub origin: SequenceOrigin,
}
pub(crate) fn actions_equal_bits(left: &Action, right: &Action) -> bool {
    match (left, right) {
        (Action::Conditional(a), Action::Conditional(b)) => a == b,
        (Action::Discrete(a), Action::Discrete(b)) => a == b,
        (Action::Continuous(a), Action::Continuous(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        _ => false,
    }
}

impl SequenceSample {
    /// Local eligibility shared by borrowed supervision rows. The source store
    /// owns the full action-space/provenance audit; recurrent callers additionally
    /// require `exact_history_inputs`.
    pub(crate) fn validate_learning_row(&self, step: usize) -> Result<()> {
        ensure(
            self.steps.len().checked_add(1) == Some(self.observations.len())
                && self.loss_mask.len() == self.steps.len()
                && self
                    .history_start
                    .checked_add(self.learn_start)
                    .is_some_and(|start| start >= self.trajectory_start)
                && self.history_start.checked_add(self.steps.len()).is_some()
                && self.burn_start <= self.learn_start
                && step >= self.learn_start
                && self.loss_mask.get(step) == Some(&true)
                && !self.steps[..step].iter().any(TemporalStep::ended),
            "supervision must select a factual learning row with matching sequence shapes",
        )
    }

    /// Borrow the actual prefix from its original reset, including the final
    /// successor. Restored ancestry is present here but excluded by loss_mask.
    pub fn exact_history_inputs(&self) -> Result<impl ExactSizeIterator<Item = HistoryInput<'_>>> {
        ensure(
            matches!(self.origin, SequenceOrigin::Reset)
                && self.history_start == 0
                && self.previous.is_none()
                && self.steps.len().checked_add(1) == Some(self.observations.len()),
            "recurrent input requires the exact reset prefix",
        )?;
        Ok(self
            .observations
            .iter()
            .enumerate()
            .map(|(index, observation)| HistoryInput {
                observation,
                previous: index.checked_sub(1).map(|previous| {
                    let step = &self.steps[previous];
                    PreviousInput::from_transition(
                        &step.action,
                        &step.reward,
                        &self.observations[previous],
                    )
                }),
            }))
    }

    /// Compare the facts of two windows already known to start at the same
    /// origin. Learning ranges and masks can differ over the shared history.
    pub(crate) fn factual_prefixes_equal(&self, other: &Self) -> bool {
        self.trajectory_start == other.trajectory_start
            && self
                .observations
                .iter()
                .zip(&other.observations)
                .all(|(a, b)| a.matches(b))
            && self.steps.iter().zip(&other.steps).all(|(a, b)| {
                actions_equal_bits(&a.action, &b.action)
                    && a.reward.matches(&b.reward)
                    && a.terminated == b.terminated
                    && a.truncated == b.truncated
                    && a.behavior.revision == b.behavior.revision
                    && a.behavior.log_prob.map(f32::to_bits)
                        == b.behavior.log_prob.map(f32::to_bits)
            })
    }

    pub fn absolute_learn_start(&self) -> usize {
        self.history_start + self.learn_start
    }
    fn context_len(&self) -> usize {
        self.trajectory_start.saturating_sub(self.history_start)
    }
    fn first_dense_width(&self) -> Result<usize> {
        self.observations
            .first()
            .and_then(Observation::as_dense)
            .map(|observation| observation.values.len())
            .filter(|&width| width > 0)
            .ok_or_else(|| Error::Shape("expected a nonempty dense temporal observation".into()))
    }

    /// Validate chronology, shapes, and masks. The source store additionally
    /// validates actions against its exact declared bounds and action count.
    pub fn validate(&self, config: &SequenceConfig) -> Result<()> {
        let width = self.first_dense_width()?;
        let space = match &self
            .steps
            .first()
            .ok_or_else(|| Error::Shape("empty temporal sequence".into()))?
            .action
        {
            Action::Discrete(_) => {
                let masked_count = self
                    .observations
                    .iter()
                    .find_map(|o| o.dense_mask().map(<[bool]>::len));
                let count = if let Some(count) = masked_count {
                    count
                } else {
                    self.steps
                        .iter()
                        .filter_map(|s| match s.action {
                            Action::Discrete(a) => Some(a),
                            _ => None,
                        })
                        .chain(
                            self.previous
                                .iter()
                                .filter_map(|previous| match previous.action {
                                    Action::Discrete(action) => Some(action),
                                    _ => None,
                                }),
                        )
                        .max()
                        .and_then(|a| a.checked_add(1))
                        .ok_or_else(|| Error::Action("invalid temporal discrete action".into()))?
                };
                ActionSpace::Discrete { count }
            }
            Action::Continuous(action) => ActionSpace::Unbounded {
                dimensions: action.len(),
            },
            Action::Conditional(_) => {
                return Err(Error::Config(
                    "conditional sequences require a declared observation and action space".into(),
                ));
            }
        };
        space.validate()?;
        self.validate_in(config, &ObservationSpace::Dense(width), &space)
    }
    pub fn validate_in(
        &self,
        config: &SequenceConfig,
        observations: &ObservationSpace,
        space: &ActionSpace,
    ) -> Result<()> {
        config.validate()?;
        ensure(
            !self.steps.is_empty()
                && self.steps.len().checked_add(1) == Some(self.observations.len())
                && self.loss_mask.len() == self.steps.len()
                && self.learn_start >= self.context_len()
                && self.learn_start < self.steps.len()
                && self.burn_start == self.learn_start.saturating_sub(config.burn_in),
            "invalid temporal sequence lengths or burn-in boundary",
        )?;
        ensure(
            self.history_start.checked_add(self.steps.len()).is_some()
                && (self.history_start == 0) == self.previous.is_none(),
            "invalid temporal sample origin or previous input",
        )?;
        match &self.origin {
            SequenceOrigin::Reset => ensure(
                self.history_start == 0,
                "mid-episode temporal sample cannot reset",
            )?,
            SequenceOrigin::Anchored(set) => {
                set.validate()?;
                ensure(
                    self.history_start > 0
                        && set.episode_id() == self.episode_id
                        && set.observation_index() == self.history_start,
                    "temporal sample anchor differs from history origin",
                )?;
            }
            SequenceOrigin::Stateless => ensure(
                config.burn_in == 0 && self.learn_start == 0,
                "stateless samples start at their first learning decision",
            )?,
        }
        if let Some(previous) = &self.previous {
            ensure(
                previous.reward.is_finite(),
                "nonfinite temporal sample boundary reward",
            )?;
        }
        if let Some(previous) = &self.previous {
            space.check(&previous.action, None)?;
        }
        self.observations[0].validate_in(observations, space, true)?;
        for (i, step) in self.steps.iter().enumerate() {
            ensure(
                !step.ended() || i + 1 == self.steps.len(),
                "temporal sequence crosses an episode end",
            )?;
            self.observations[i + 1].validate_in(observations, space, !step.terminated)?;
            self.observations[i].check_action(space, &step.action)?;
            step.behavior.validate()?;
            step.reward.validate()?;
            if i < self.context_len() {
                ensure(
                    !step.ended()
                        && step.behavior.revision == 0
                        && step.behavior.log_prob.is_none(),
                    "sequence context carries an end flag or behavior metadata",
                )?;
            }
        }
        let learn_end = self
            .learn_start
            .checked_add(config.learn_len)
            .ok_or_else(|| Error::Config("temporal learning boundary overflow".into()))?;
        let context_end = learn_end
            .checked_add(config.n_step - 1)
            .ok_or_else(|| Error::Config("temporal context boundary overflow".into()))?;
        ensure(
            self.steps.len() <= context_end,
            "temporal sequence exceeds its configured context",
        )?;
        for (i, &learning) in self.loss_mask.iter().enumerate() {
            let expected = i >= self.learn_start
                && i < learn_end
                && has_target_context(&self.steps, i, config.n_step);
            ensure(
                learning == expected,
                "temporal sequence loss mask differs from its learning context",
            )?;
        }
        ensure(
            self.loss_mask.iter().any(|&learning| learning),
            "temporal sequence has no learning positions",
        )
    }
    /// Aggregate raw rewards without crossing an episode reset. Truncation
    /// bootstraps at the actual final observation; true termination has zero discount.
    pub fn target_at(
        &self,
        local_transition: usize,
        gamma: f32,
        n_step: usize,
    ) -> Result<NStepTarget> {
        ensure(
            gamma.is_finite() && (0.0..=1.0).contains(&gamma) && n_step > 0,
            "invalid temporal n-step settings",
        )?;
        ensure(
            local_transition >= self.context_len()
                && self.steps.len().checked_add(1) == Some(self.observations.len())
                && has_target_context(&self.steps, local_transition, n_step),
            "incomplete temporal n-step context",
        )?;
        let mut reward = 0.;
        let mut discount = 1.;
        let mut bootstrap_index = local_transition;
        for (i, step) in self
            .steps
            .iter()
            .enumerate()
            .skip(local_transition)
            .take(n_step)
        {
            let (accumulated, continuation) = match &step.reward {
                Reward::One(value) => {
                    ensure(value.is_finite(), "nonfinite temporal target reward")?;
                    (*value, gamma)
                }
                trace => trace.discounted(gamma)?,
            };
            reward += discount * accumulated;
            discount *= continuation;
            bootstrap_index = i + 1;
            if step.ended() {
                if step.terminated {
                    discount = 0.;
                }
                break;
            }
        }
        if !reward.is_finite() {
            return Err(Error::Numerical("nonfinite temporal n-step target".into()));
        }
        Ok(NStepTarget {
            reward,
            discount,
            bootstrap_index,
        })
    }
}

fn has_target_context(steps: &[TemporalStep], start: usize, n_step: usize) -> bool {
    if n_step == 0 || start >= steps.len() {
        return false;
    }
    let available = steps.len() - start;
    steps[start..].iter().take(n_step).any(TemporalStep::ended) || available >= n_step
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceBatch {
    pub samples: Vec<SequenceSample>,
    pub config: SequenceConfig,
}
impl SequenceBatch {
    pub fn validate(&self) -> Result<()> {
        self.config.validate()?;
        ensure(!self.samples.is_empty(), "empty temporal sequence batch")?;
        let width = self.samples[0].first_dense_width()?;
        for sample in &self.samples {
            sample.validate(&self.config)?;
            ensure(
                sample.first_dense_width()? == width,
                "temporal batch observation widths differ",
            )?;
        }
        Ok(())
    }
    pub fn validate_in(&self, observations: &ObservationSpace, space: &ActionSpace) -> Result<()> {
        ensure(!self.samples.is_empty(), "empty temporal sequence batch")?;
        for sample in &self.samples {
            sample.validate_in(&self.config, observations, space)?;
        }
        Ok(())
    }
    pub fn padded_layout(&self) -> Result<PaddedSequenceLayout> {
        self.validate()?;
        let batch = self.samples.len();
        let time = self
            .samples
            .iter()
            .map(|sample| sample.observations.len())
            .max()
            .expect("validated nonempty batch");
        let width = self.samples[0].first_dense_width()?;
        let tokens = batch
            .checked_mul(time)
            .ok_or_else(|| Error::Shape("temporal batch token count overflow".into()))?;
        let elements = tokens
            .checked_mul(width)
            .ok_or_else(|| Error::Shape("temporal batch element count overflow".into()))?;
        let mut out = PaddedSequenceLayout {
            batch,
            time,
            width,
            observations: filled(elements, 0.)?,
            valid_mask: filled(tokens, false)?,
            learning_mask: filled(tokens, false)?,
            reset_mask: filled(tokens, false)?,
        };
        for (row, sample) in self.samples.iter().enumerate() {
            out.reset_mask[row * time] = matches!(sample.origin, SequenceOrigin::Reset);
            for (t, observation) in sample.observations.iter().enumerate() {
                let token = row * time + t;
                out.observations[token * width..(token + 1) * width]
                    .copy_from_slice(observation.dense_values());
                out.valid_mask[token] = true;
                out.learning_mask[token] = sample.loss_mask.get(t).copied().unwrap_or(false);
            }
        }
        Ok(out)
    }
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| Error::State(format!("cannot reserve temporal batch: {e}")))?;
    values.resize(len, value);
    Ok(values)
}

impl Snapshot for SequenceBatch {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate()
    }
}

/// Batch-major `[batch, time, width]`; time includes the final observation token.
#[derive(Debug, Clone, PartialEq)]
pub struct PaddedSequenceLayout {
    pub batch: usize,
    pub time: usize,
    pub width: usize,
    pub observations: Vec<f32>,
    pub valid_mask: Vec<bool>,
    pub learning_mask: Vec<bool>,
    pub reset_mask: Vec<bool>,
}
