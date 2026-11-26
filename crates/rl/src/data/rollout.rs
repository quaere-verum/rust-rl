pub use super::pathwise::PathwiseBatch;
use crate::{
    Result,
    core::{
        action::Action,
        ensure,
        env::Observation,
        transition::{Transition, TransitionView},
    },
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BootstrappedBatch {
    pub log_probs: Vec<f32>,
    pub values: Vec<f32>,
    pub advantages: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OnPolicyMetadata {
    Bootstrapped(BootstrappedBatch),
    Episodic { baseline_values: Option<Vec<f32>> },
    Pathwise(PathwiseBatch),
}
impl Default for OnPolicyMetadata {
    fn default() -> Self {
        Self::Bootstrapped(Default::default())
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OnPolicyBatch {
    pub episode_count: usize,
    pub observations: Vec<Observation>,
    pub actions: Vec<Action>,
    pub latents: Vec<Option<Vec<f32>>>,
    /// Ordinary value/Monte Carlo targets; empty for pathwise metadata.
    pub returns: Vec<f32>,
    /// Episodic policy weights; empty for pathwise metadata.
    pub discount_weights: Vec<f32>,
    pub metadata: OnPolicyMetadata,
}
impl OnPolicyBatch {
    pub fn len(&self) -> usize {
        self.observations.len()
    }
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
    pub fn complete_episodes(&self) -> bool {
        matches!(self.metadata, OnPolicyMetadata::Episodic { .. })
    }
    pub fn bootstrapped(&self) -> Result<&BootstrappedBatch> {
        match &self.metadata {
            OnPolicyMetadata::Bootstrapped(b) => Ok(b),
            _ => Err(crate::Error::State(
                "learner requires bootstrapped on-policy metadata".into(),
            )),
        }
    }
    pub fn baseline_values(&self) -> Result<Option<&[f32]>> {
        match &self.metadata {
            OnPolicyMetadata::Episodic { baseline_values } => Ok(baseline_values.as_deref()),
            _ => Err(crate::Error::State(
                "learner requires complete episodic metadata".into(),
            )),
        }
    }
    pub fn pathwise(&self) -> Result<&PathwiseBatch> {
        match &self.metadata {
            OnPolicyMetadata::Pathwise(p) => Ok(p),
            _ => Err(crate::Error::State(
                "learner requires pathwise on-policy metadata".into(),
            )),
        }
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.len();
        ensure(
            n > 0
                && [self.actions.len(), self.latents.len()]
                    .iter()
                    .all(|&l| l == n)
                && if matches!(self.metadata, OnPolicyMetadata::Pathwise(_)) {
                    self.returns.is_empty() && self.discount_weights.is_empty()
                } else {
                    self.returns.len() == n && self.discount_weights.len() == n
                },
            "invalid on-policy batch lengths",
        )?;
        ensure(
            self.returns
                .iter()
                .chain(&self.discount_weights)
                .all(|x| x.is_finite()),
            "nonfinite on-policy targets",
        )?;
        match &self.metadata {
            OnPolicyMetadata::Pathwise(p) => {
                p.validate(n)?;
                let width = self.observations[0].dense_values().len();
                ensure(
                    width > 0
                        && self
                            .observations
                            .iter()
                            .chain(&p.next_observations)
                            .all(|o| {
                                o.dense_values().len() == width
                                    && o.dense_values().iter().all(|x| x.is_finite())
                            }),
                    "invalid pathwise observation shape",
                )?;
                if p.complete {
                    for i in 0..n.saturating_sub(p.lanes) {
                        ensure(
                            p.terminated[i]
                                || p.truncated[i]
                                || p.next_observations[i] == self.observations[i + p.lanes],
                            "pathwise successor observation does not match the next lane step",
                        )?;
                    }
                }
                Ok(())
            }
            OnPolicyMetadata::Bootstrapped(b) => ensure(
                [b.log_probs.len(), b.values.len(), b.advantages.len()]
                    .iter()
                    .all(|&l| l == n)
                    && b.log_probs
                        .iter()
                        .chain(&b.values)
                        .chain(&b.advantages)
                        .all(|x| x.is_finite()),
                "invalid bootstrapped metadata",
            ),
            OnPolicyMetadata::Episodic { baseline_values } => ensure(
                self.episode_count > 0
                    && baseline_values
                        .as_ref()
                        .is_none_or(|v| v.len() == n && v.iter().all(|x| x.is_finite())),
                "invalid episodic metadata",
            ),
        }
    }
    pub fn select(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        ensure(
            !indices.is_empty() && indices.iter().all(|&i| i < self.len()),
            "invalid minibatch indices",
        )?;
        let mut out = Self::default();
        self.select_validated_into(indices, &mut out);
        Ok(out)
    }

    /// Only call after validating the parent and indices once at the learner boundary.
    pub(crate) fn select_validated_into(&self, indices: &[usize], out: &mut Self) {
        debug_assert!(!indices.is_empty() && indices.iter().all(|&i| i < self.len()));
        fn copy_slice<T: Copy>(source: &[T], target: &mut Vec<T>) {
            target.clear();
            // Fixed-width row payloads need no geometric growth on first use.
            target.reserve_exact(source.len());
            target.extend_from_slice(source);
        }
        fn copy_optional<T: Copy>(source: Option<&[T]>, target: &mut Option<Vec<T>>) {
            match source {
                Some(source) => copy_slice(source, target.get_or_insert_with(Vec::new)),
                None => *target = None,
            }
        }
        out.episode_count = self.episode_count;
        out.observations
            .resize_with(indices.len(), || Observation::new(Vec::new()));
        out.actions.resize(indices.len(), Action::Discrete(0));
        out.latents.resize_with(indices.len(), || None);
        for (row, &i) in indices.iter().enumerate() {
            let source = &self.observations[i];
            if let (Some(source), Some(target)) =
                (source.as_dense(), out.observations[row].as_dense_mut())
            {
                copy_slice(&source.values, &mut target.values);
                copy_optional(source.action_mask.as_deref(), &mut target.action_mask);
            } else {
                out.observations[row].clone_from(source);
            }
            match (&self.actions[i], &mut out.actions[row]) {
                (Action::Continuous(source), Action::Continuous(target)) => {
                    copy_slice(source, target);
                }
                (source, target) => target.clone_from(source),
            }
            copy_optional(self.latents[i].as_deref(), &mut out.latents[row]);
        }
        fn copy_selected(source: &[f32], indices: &[usize], out: &mut Vec<f32>) {
            out.clear();
            out.extend(indices.iter().map(|&i| source[i]));
        }
        if matches!(self.metadata, OnPolicyMetadata::Pathwise(_)) {
            out.returns.clear();
            out.discount_weights.clear();
        } else {
            copy_selected(&self.returns, indices, &mut out.returns);
            copy_selected(&self.discount_weights, indices, &mut out.discount_weights);
        }
        match &self.metadata {
            OnPolicyMetadata::Pathwise(source) => {
                if !matches!(out.metadata, OnPolicyMetadata::Pathwise(_)) {
                    out.metadata = OnPolicyMetadata::Pathwise(PathwiseBatch::default());
                }
                let OnPolicyMetadata::Pathwise(target) = &mut out.metadata else {
                    unreachable!()
                };
                source.select_into(indices, target);
            }
            OnPolicyMetadata::Bootstrapped(source) => {
                if !matches!(out.metadata, OnPolicyMetadata::Bootstrapped(_)) {
                    out.metadata = OnPolicyMetadata::Bootstrapped(BootstrappedBatch::default());
                }
                let OnPolicyMetadata::Bootstrapped(target) = &mut out.metadata else {
                    unreachable!()
                };
                copy_selected(&source.log_probs, indices, &mut target.log_probs);
                copy_selected(&source.values, indices, &mut target.values);
                copy_selected(&source.advantages, indices, &mut target.advantages);
            }
            OnPolicyMetadata::Episodic { baseline_values } => {
                if !matches!(out.metadata, OnPolicyMetadata::Episodic { .. }) {
                    out.metadata = OnPolicyMetadata::Episodic {
                        baseline_values: None,
                    };
                }
                let OnPolicyMetadata::Episodic {
                    baseline_values: target,
                } = &mut out.metadata
                else {
                    unreachable!()
                };
                match baseline_values {
                    Some(source) => {
                        copy_selected(source, indices, target.get_or_insert_with(Vec::new))
                    }
                    None => *target = None,
                }
            }
        }
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    fn batch() -> OnPolicyBatch {
        OnPolicyBatch {
            episode_count: 1,
            observations: (0..4).map(|i| Observation::new(vec![i as f32])).collect(),
            actions: (0..4).map(Action::Discrete).collect(),
            latents: vec![None; 4],
            returns: vec![10., 20., 30., 40.],
            discount_weights: vec![1.; 4],
            metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
                log_probs: vec![-1., -2., -3., -4.],
                values: vec![0.; 4],
                advantages: vec![4., 3., 2., 1.],
            }),
        }
    }

    #[test]
    fn validated_selection_reuses_outer_storage_and_matches_public_selection() {
        let source = batch();
        let mut retained = OnPolicyBatch::default();
        source.select_validated_into(&[3, 1, 2], &mut retained);
        assert_eq!(
            serde_json::to_value(&retained).unwrap(),
            serde_json::to_value(source.select(&[3, 1, 2]).unwrap()).unwrap()
        );
        let observations = retained.observations.as_ptr();
        let actions = retained.actions.as_ptr();
        let returns = retained.returns.as_ptr();
        source.select_validated_into(&[2, 0], &mut retained);
        assert_eq!(retained.observations.as_ptr(), observations);
        assert_eq!(retained.actions.as_ptr(), actions);
        assert_eq!(retained.returns.as_ptr(), returns);
        assert_eq!(retained.returns, [30., 10.]);
    }

    #[test]
    fn public_selection_still_rejects_malformed_parent_and_indices() {
        let mut malformed = batch();
        malformed.returns.pop();
        assert!(malformed.select(&[0]).is_err());
        let valid = batch();
        assert!(valid.select(&[]).is_err());
        assert!(valid.select(&[4]).is_err());
    }

    #[test]
    fn repeated_validated_selection_reuses_nested_row_payloads() {
        let mut source = batch();
        for (i, observation) in source.observations.iter_mut().enumerate() {
            let observation = observation.as_dense_mut().unwrap();
            observation.values = vec![i as f32, (i + 1) as f32];
            observation.action_mask = Some(vec![true, false, true]);
            source.actions[i] = Action::Continuous(vec![i as f32, -(i as f32)]);
            source.latents[i] = Some(vec![i as f32, 0.5]);
        }
        let mut selected = OnPolicyBatch::default();
        source.select_validated_into(&[0, 1], &mut selected);
        let values = selected.observations[0].dense_values().as_ptr();
        let mask = selected.observations[0].dense_mask().unwrap().as_ptr();
        let Action::Continuous(action) = &selected.actions[0] else {
            panic!("continuous action")
        };
        let action = action.as_ptr();
        let latent = selected.latents[0].as_ref().unwrap().as_ptr();
        source.select_validated_into(&[2, 3], &mut selected);
        assert_eq!(selected.observations[0].dense_values(), [2., 3.]);
        assert_eq!(selected.actions[0], Action::Continuous(vec![2., -2.]));
        assert_eq!(selected.latents[0], Some(vec![2., 0.5]));
        assert_eq!(selected.observations[0].dense_values().as_ptr(), values);
        assert_eq!(
            selected.observations[0].dense_mask().unwrap().as_ptr(),
            mask
        );
        let Action::Continuous(next_action) = &selected.actions[0] else {
            panic!("continuous action")
        };
        assert_eq!(next_action.as_ptr(), action);
        assert_eq!(selected.latents[0].as_ref().unwrap().as_ptr(), latent);
    }
}
#[derive(Debug, Clone)]
pub struct RolloutStep {
    pub transition: Transition,
    pub episode_step: u64,
    pub latent: Option<Vec<f32>>,
}
impl Default for RolloutStep {
    fn default() -> Self {
        Self {
            transition: Transition {
                observation: Observation::new(vec![]),
                action: Action::Discrete(0),
                reward: 0.0.into(),
                next_observation: Observation::new(vec![]),
                terminated: false,
                truncated: false,
            },
            episode_step: 0,
            latent: None,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum RolloutMetadata {
    Pathwise {
        log_probs: Vec<f32>,
        revision: u64,
    },
    Bootstrapped {
        log_probs: Vec<f32>,
        values: Vec<f32>,
        next_values: Vec<f32>,
    },
    Episodic {
        baseline_values: Option<Vec<f32>>,
    },
}
/// Reusable time-major storage. Inactive slots retain capacity but are never transitions.
#[derive(Debug, Clone)]
pub struct RolloutBuffer {
    pub(crate) lanes: usize,
    pub(crate) len: usize,
    pub(crate) rows: Vec<RolloutStep>,
    pub(crate) valid: Vec<bool>,
    pub(crate) metadata: RolloutMetadata,
    advantages: Vec<f32>,
    returns: Vec<f32>,
}
impl Default for RolloutBuffer {
    fn default() -> Self {
        Self {
            lanes: 0,
            len: 0,
            rows: vec![],
            valid: vec![],
            metadata: RolloutMetadata::Bootstrapped {
                log_probs: vec![],
                values: vec![],
                next_values: vec![],
            },
            advantages: vec![],
            returns: vec![],
        }
    }
}
impl RolloutBuffer {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn lanes(&self) -> usize {
        self.lanes
    }
    pub fn row(&self, index: usize) -> Option<&RolloutStep> {
        if index < self.len && self.valid[index] {
            Some(&self.rows[index])
        } else {
            None
        }
    }
    pub fn next_value(&self, index: usize) -> Option<f32> {
        self.row(index)?;
        match &self.metadata {
            RolloutMetadata::Bootstrapped { next_values, .. } => Some(next_values[index]),
            _ => None,
        }
    }
    pub fn valid_len(&self) -> usize {
        self.valid[..self.len].iter().filter(|x| **x).count()
    }
    pub(crate) fn start(&mut self, lanes: usize, steps: usize, episodic: bool) -> Result<()> {
        let capacity = lanes
            .checked_mul(if episodic { 1 } else { steps })
            .ok_or_else(|| crate::Error::Config("rollout capacity overflow".into()))?;
        self.lanes = lanes;
        self.len = 0;
        if self.rows.len() < capacity {
            self.rows.resize_with(capacity, Default::default);
        }
        self.valid.resize(self.rows.len(), false);

        match (&mut self.metadata, episodic) {
            (
                RolloutMetadata::Bootstrapped {
                    log_probs,
                    values,
                    next_values,
                },
                false,
            ) => {
                log_probs.clear();
                values.clear();
                next_values.clear();
            }
            (RolloutMetadata::Episodic { baseline_values }, true) => {
                if let Some(v) = baseline_values {
                    v.clear();
                }
            }
            (_, true) => {
                self.metadata = RolloutMetadata::Episodic {
                    baseline_values: None,
                }
            }
            (_, false) => {
                self.metadata = RolloutMetadata::Bootstrapped {
                    log_probs: vec![],
                    values: vec![],
                    next_values: vec![],
                }
            }
        }
        Ok(())
    }
    pub(crate) fn record(
        &mut self,
        t: Option<TransitionView<'_>>,
        episode_step: u64,
        latent: Option<&Vec<f32>>,
    ) -> usize {
        if self.len == self.rows.len() {
            // Episodic guards bound execution, not storage. Grow only for observed rows.
            self.rows.push(RolloutStep::default());
            self.valid.push(false);
        }
        let index = self.len;
        self.len += 1;
        self.valid[index] = t.is_some();
        if let Some(t) = t {
            super::replay::copy_transition(&mut self.rows[index].transition, t);
            self.rows[index].episode_step = episode_step;
            match (&mut self.rows[index].latent, latent) {
                (Some(dst), Some(src)) => dst.clone_from(src),
                (dst, src) => *dst = src.cloned(),
            }
        }
        index
    }
    pub(crate) fn invalidate(&mut self) {
        self.len = 0;
    }
    /// Prepare targets and transfer the valid rows into a new batch, emptying this rollout.
    pub fn to_batch(
        &mut self,
        gamma: f32,
        lambda: Option<f32>,
        normalize: bool,
    ) -> Result<OnPolicyBatch> {
        let mut out = OnPolicyBatch::default();
        self.to_batch_into(gamma, lambda, normalize, &mut out)?;
        Ok(out)
    }
    /// Prepare targets before flattening, then transfer observation/action/latent ownership.
    /// The rollout becomes empty; its slots recycle the previous batch's allocations.
    pub fn to_batch_into(
        &mut self,
        gamma: f32,
        lambda: Option<f32>,
        normalize: bool,
        out: &mut OnPolicyBatch,
    ) -> Result<()> {
        ensure(
            self.lanes > 0
                && self.len > 0
                && self.len.is_multiple_of(self.lanes)
                && gamma.is_finite()
                && (0.0..=1.0).contains(&gamma)
                && lambda.is_none_or(|x| x.is_finite() && (0.0..=1.0).contains(&x)),
            "invalid rollout or return settings",
        )?;
        ensure(
            self.rows[..self.len]
                .iter()
                .zip(&self.valid)
                .all(|(row, &valid)| !valid || row.transition.reward.duration() == 1),
            "numerical rollout requires one primitive reward per decision",
        )?;
        let episodic = matches!(self.metadata, RolloutMetadata::Episodic { .. });
        if matches!(self.metadata, RolloutMetadata::Pathwise { .. }) {
            return self.flatten_pathwise_into(gamma, out);
        }
        ensure(
            !episodic || lambda.is_none(),
            "Monte Carlo trajectories do not use GAE",
        )?;
        if !episodic {
            self.advantages.resize(self.len, 0.);
        }
        self.returns.resize(self.len, 0.);
        for lane in 0..self.lanes {
            let mut following = false;
            let mut carry = 0.;
            for index in (lane..self.len).step_by(self.lanes).rev() {
                if !self.valid[index] {
                    continue;
                }
                let t = &self.rows[index].transition;
                let (value, next) = match &self.metadata {
                    RolloutMetadata::Pathwise { .. } => unreachable!("handled before GAE"),
                    RolloutMetadata::Bootstrapped {
                        values,
                        next_values,
                        ..
                    } => (values[index], next_values[index]),
                    RolloutMetadata::Episodic { baseline_values } => {
                        (baseline_values.as_ref().map_or(0., |v| v[index]), 0.)
                    }
                };
                let continuation = following && !t.ended();
                if let Some(lambda) = lambda {
                    let delta =
                        t.reward.total() + if t.terminated { 0. } else { gamma * next } - value;
                    let advantage = delta
                        + if continuation {
                            gamma * lambda * carry
                        } else {
                            0.
                        };
                    self.advantages[index] = advantage;
                    self.returns[index] = advantage + value;
                    carry = advantage;
                } else {
                    let ret = t.reward.total()
                        + if t.terminated {
                            0.
                        } else if continuation {
                            gamma * carry
                        } else {
                            gamma * next
                        };
                    self.returns[index] = ret;
                    if !episodic {
                        self.advantages[index] = ret - value;
                    }
                    carry = ret;
                }
                following = true;
            }
        }
        let n = self.valid_len();
        out.episode_count = (0..self.len)
            .filter(|&i| self.valid[i] && self.rows[i].transition.terminated)
            .count();
        out.observations.resize_with(n, || Observation::new(vec![]));
        out.actions.resize_with(n, || Action::Discrete(0));
        out.latents.resize_with(n, || None);
        out.returns.resize(n, 0.);
        out.discount_weights.resize(n, 0.);
        match (&self.metadata, &mut out.metadata) {
            (RolloutMetadata::Pathwise { .. }, _) => unreachable!("handled before GAE"),
            (RolloutMetadata::Bootstrapped { .. }, OnPolicyMetadata::Bootstrapped(_)) => (),
            (RolloutMetadata::Episodic { .. }, OnPolicyMetadata::Episodic { .. }) => (),
            (RolloutMetadata::Bootstrapped { .. }, dst) => {
                *dst = OnPolicyMetadata::Bootstrapped(Default::default())
            }
            (RolloutMetadata::Episodic { .. }, dst) => {
                *dst = OnPolicyMetadata::Episodic {
                    baseline_values: None,
                }
            }
        }
        match &mut out.metadata {
            OnPolicyMetadata::Pathwise(_) => unreachable!("handled before GAE"),
            OnPolicyMetadata::Bootstrapped(b) => {
                b.log_probs.resize(n, 0.);
                b.values.resize(n, 0.);
                b.advantages.resize(n, 0.);
            }
            OnPolicyMetadata::Episodic { baseline_values } => {
                if matches!(
                    &self.metadata,
                    RolloutMetadata::Episodic {
                        baseline_values: Some(_)
                    }
                ) {
                    baseline_values.get_or_insert_with(Vec::new).resize(n, 0.);
                } else {
                    *baseline_values = None;
                }
            }
        }
        let mut j = 0;
        for i in 0..self.len {
            if !self.valid[i] {
                continue;
            }
            let row = &mut self.rows[i];
            std::mem::swap(&mut out.observations[j], &mut row.transition.observation);
            std::mem::swap(&mut out.actions[j], &mut row.transition.action);
            std::mem::swap(&mut out.latents[j], &mut row.latent);
            out.returns[j] = self.returns[i];
            out.discount_weights[j] = (gamma as f64).powf(row.episode_step as f64) as f32;
            match (&self.metadata, &mut out.metadata) {
                (
                    RolloutMetadata::Bootstrapped {
                        log_probs, values, ..
                    },
                    OnPolicyMetadata::Bootstrapped(b),
                ) => {
                    b.log_probs[j] = log_probs[i];
                    b.values[j] = values[i];
                    b.advantages[j] = self.advantages[i];
                }
                (
                    RolloutMetadata::Episodic {
                        baseline_values: Some(src),
                    },
                    OnPolicyMetadata::Episodic {
                        baseline_values: Some(dst),
                    },
                ) => dst[j] = src[i],
                _ => (),
            }
            j += 1;
        }
        if normalize && let OnPolicyMetadata::Bootstrapped(b) = &mut out.metadata {
            let mean = b.advantages.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
            let variance = b
                .advantages
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            let scale = variance.sqrt().max(1e-8);
            for x in &mut b.advantages {
                *x = ((*x as f64 - mean) / scale) as f32;
            }
        }
        self.len = 0;
        out.validate()
    }
}
