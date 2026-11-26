//! Complete temporal metadata for pathwise on-policy targets, retained after flattening.
use super::{
    replay::copy_observation,
    rollout::{OnPolicyBatch, OnPolicyMetadata, RolloutBuffer, RolloutMetadata},
};
use crate::{
    Result,
    core::{action::Action, ensure, env::Observation},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PathwiseBatch {
    pub gamma: f32,
    pub lanes: usize,
    pub rollout_len: usize,
    pub revision: u64,
    /// Original time-major index; lane = id % lanes, time = id / lanes.
    pub row_ids: Vec<usize>,
    pub episode_steps: Vec<u64>,
    pub log_probs: Vec<f32>,
    pub next_observations: Vec<Observation>,
    pub rewards: Vec<f32>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    /// Only a complete, unpermuted fragment may construct multistep targets.
    pub complete: bool,
}
impl PathwiseBatch {
    pub fn validate(&self, n: usize) -> Result<()> {
        ensure(
            n > 0
                && self.lanes > 0
                && self.rollout_len > 0
                && self.rollout_len.is_multiple_of(self.lanes)
                && self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma)
                && [
                    self.row_ids.len(),
                    self.episode_steps.len(),
                    self.log_probs.len(),
                    self.next_observations.len(),
                    self.rewards.len(),
                    self.terminated.len(),
                    self.truncated.len(),
                ]
                .iter()
                .all(|&len| len == n)
                && self.row_ids.iter().all(|&id| id < self.rollout_len)
                && self
                    .log_probs
                    .iter()
                    .chain(&self.rewards)
                    .all(|x| x.is_finite())
                && self
                    .next_observations
                    .iter()
                    .all(|o| o.dense_values().iter().all(|x| x.is_finite())),
            "invalid pathwise metadata",
        )?;
        if self.complete {
            ensure(
                n == self.rollout_len && self.row_ids.iter().copied().eq(0..n),
                "complete pathwise fragment must retain time-major order",
            )?;
            for i in 0..n.saturating_sub(self.lanes) {
                let next = i + self.lanes;
                let expected = if self.terminated[i] || self.truncated[i] {
                    Some(0)
                } else {
                    self.episode_steps[i].checked_add(1)
                };
                ensure(
                    expected == Some(self.episode_steps[next]),
                    "inconsistent pathwise episode connectivity",
                )?;
            }
        }
        Ok(())
    }
    pub fn validate_complete(&self) -> Result<()> {
        self.validate(self.row_ids.len())?;
        ensure(
            self.complete,
            "soft targets require a complete pathwise rollout before minibatch selection",
        )
    }
    pub(crate) fn select_into(&self, indices: &[usize], out: &mut Self) {
        fn selected<T: Copy>(source: &[T], ids: &[usize], dest: &mut Vec<T>) {
            dest.clear();
            dest.extend(ids.iter().map(|&i| source[i]));
        }
        out.gamma = self.gamma;
        out.lanes = self.lanes;
        out.rollout_len = self.rollout_len;
        out.revision = self.revision;
        out.complete = self.complete && indices.iter().copied().eq(0..self.rollout_len);
        selected(&self.row_ids, indices, &mut out.row_ids);
        selected(&self.episode_steps, indices, &mut out.episode_steps);
        selected(&self.log_probs, indices, &mut out.log_probs);
        selected(&self.rewards, indices, &mut out.rewards);
        selected(&self.terminated, indices, &mut out.terminated);
        selected(&self.truncated, indices, &mut out.truncated);
        out.next_observations
            .resize_with(indices.len(), || Observation::new(Vec::new()));
        for (dest, &i) in out.next_observations.iter_mut().zip(indices) {
            copy_observation(dest, &self.next_observations[i]);
        }
    }
}

impl RolloutBuffer {
    pub(crate) fn start_pathwise(
        &mut self,
        lanes: usize,
        steps: usize,
        revision: u64,
    ) -> Result<()> {
        let capacity = lanes
            .checked_mul(steps)
            .ok_or_else(|| crate::Error::Shape("pathwise capacity overflow".into()))?;
        self.lanes = lanes;
        self.len = 0;
        if self.rows.len() < capacity {
            self.rows.resize_with(capacity, Default::default);
        }
        self.valid.resize(self.rows.len(), false);
        match &mut self.metadata {
            RolloutMetadata::Pathwise {
                log_probs,
                revision: stored,
            } => {
                log_probs.clear();
                *stored = revision;
            }
            dest => {
                *dest = RolloutMetadata::Pathwise {
                    log_probs: Vec::new(),
                    revision,
                }
            }
        }
        Ok(())
    }
    pub(crate) fn flatten_pathwise_into(
        &mut self,
        gamma: f32,
        out: &mut OnPolicyBatch,
    ) -> Result<()> {
        ensure(
            self.valid[..self.len].iter().all(|x| *x),
            "pathwise fragments require all lanes active",
        )?;
        if !matches!(out.metadata, OnPolicyMetadata::Pathwise(_)) {
            out.metadata = OnPolicyMetadata::Pathwise(PathwiseBatch::default());
        }
        let OnPolicyMetadata::Pathwise(p) = &mut out.metadata else {
            unreachable!()
        };
        let RolloutMetadata::Pathwise {
            log_probs,
            revision,
        } = &self.metadata
        else {
            unreachable!()
        };
        let n = self.len;
        p.gamma = gamma;
        p.lanes = self.lanes;
        p.rollout_len = n;
        p.revision = *revision;
        p.complete = true;
        p.row_ids.clear();
        p.row_ids.extend(0..n);
        p.episode_steps.resize(n, 0);
        p.log_probs.clone_from(log_probs);
        p.rewards.resize(n, 0.);
        p.terminated.resize(n, false);
        p.truncated.resize(n, false);
        p.next_observations
            .resize_with(n, || Observation::new(Vec::new()));
        out.observations
            .resize_with(n, || Observation::new(Vec::new()));
        out.actions.resize(n, Action::Discrete(0));
        out.latents.resize_with(n, || None);
        out.returns.clear();
        out.discount_weights.clear();
        out.episode_count = 0;
        for i in 0..n {
            let row = &mut self.rows[i];
            std::mem::swap(&mut out.observations[i], &mut row.transition.observation);
            std::mem::swap(
                &mut p.next_observations[i],
                &mut row.transition.next_observation,
            );
            std::mem::swap(&mut out.actions[i], &mut row.transition.action);
            std::mem::swap(&mut out.latents[i], &mut row.latent);
            p.episode_steps[i] = row.episode_step;
            p.rewards[i] = row.transition.reward.total();
            p.terminated[i] = row.transition.terminated;
            p.truncated[i] = row.transition.truncated;
            out.episode_count += usize::from(row.transition.ended());
        }
        self.len = 0;
        out.validate()
    }
}
