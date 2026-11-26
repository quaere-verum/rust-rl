use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::Observation,
        reward::Reward,
        transition::{Transition, TransitionView},
    },
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayItem {
    pub transition: Transition,
    /// gamma^k for a continuing/truncated k-step target; zero at true termination.
    pub discount: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReplayBatch {
    pub items: Vec<ReplayItem>,
}
impl ReplayBatch {
    pub fn validate(&self) -> Result<()> {
        ensure(!self.items.is_empty(), "empty replay batch")?;
        for item in &self.items {
            ensure(
                matches!(item.transition.reward, Reward::One(value) if value.is_finite())
                    && item.discount.is_finite()
                    && (0.0..=1.0).contains(&item.discount)
                    && (!item.transition.terminated || item.discount == 0.0),
                "invalid replay reward or bootstrap discount",
            )?;
        }
        Ok(())
    }
}

/// Preallocated structure-of-arrays ring. Insertion allocates no per-transition vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayBuffer {
    capacity: usize,
    width: usize,
    space: ActionSpace,
    len: usize,
    cursor: usize,
    observations: Vec<f32>,
    next_observations: Vec<f32>,
    continuous: Vec<f32>,
    discrete: Vec<usize>,
    rewards: Vec<f32>,
    discounts: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    masks: Vec<bool>,
    next_masks: Vec<bool>,
}
impl ReplayBuffer {
    pub fn new(capacity: usize, width: usize, space: ActionSpace) -> Result<Self> {
        ensure(
            capacity > 0 && width > 0 && !matches!(space, ActionSpace::Conditional(_)),
            "replay capacity and observation width must be positive",
        )?;
        space.validate()?;
        let n = capacity
            .checked_mul(width)
            .ok_or_else(|| crate::Error::Config("replay capacity overflow".into()))?;
        let actions = if matches!(space, ActionSpace::Discrete { .. }) {
            0
        } else {
            space.dimensions()
        };
        let masks = if let ActionSpace::Discrete { count } = space {
            count
        } else {
            0
        };
        let a = capacity
            .checked_mul(actions)
            .ok_or_else(|| crate::Error::Config("action capacity overflow".into()))?;
        let m = capacity
            .checked_mul(masks)
            .ok_or_else(|| crate::Error::Config("mask capacity overflow".into()))?;
        Ok(Self {
            capacity,
            width,
            space,
            len: 0,
            cursor: 0,
            observations: vec![0.; n],
            next_observations: vec![0.; n],
            continuous: vec![0.; a],
            discrete: vec![0; capacity],
            rewards: vec![0.; capacity],
            discounts: vec![0.; capacity],
            terminated: vec![false; capacity],
            truncated: vec![false; capacity],
            masks: vec![true; m],
            next_masks: vec![true; m],
        })
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
    pub fn observation_dim(&self) -> usize {
        self.width
    }
    pub fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    pub(crate) fn validate_state(&self) -> Result<()> {
        self.space.validate()?;
        ensure(
            self.capacity > 0
                && self.width > 0
                && !matches!(self.space, ActionSpace::Conditional(_))
                && self.len <= self.capacity
                && self.cursor < self.capacity
                && (self.len == self.capacity || self.cursor == self.len),
            "invalid replay snapshot indices",
        )?;
        let n = self
            .capacity
            .checked_mul(self.width)
            .ok_or_else(|| crate::Error::Config("replay snapshot size overflow".into()))?;
        let continuous = if matches!(self.space, ActionSpace::Discrete { .. }) {
            0
        } else {
            self.capacity
                .checked_mul(self.space.dimensions())
                .ok_or_else(|| crate::Error::Config("replay action size overflow".into()))?
        };
        let masks = if let ActionSpace::Discrete { count } = self.space {
            self.capacity
                .checked_mul(count)
                .ok_or_else(|| crate::Error::Config("replay mask size overflow".into()))?
        } else {
            0
        };
        ensure(
            self.observations.len() == n
                && self.next_observations.len() == n
                && self.continuous.len() == continuous
                && self.masks.len() == masks
                && self.next_masks.len() == masks
                && [
                    self.discrete.len(),
                    self.rewards.len(),
                    self.discounts.len(),
                    self.terminated.len(),
                    self.truncated.len(),
                ]
                .iter()
                .all(|&l| l == self.capacity),
            "invalid replay snapshot field lengths",
        )?;
        for i in 0..self.len {
            let row = i * self.width..(i + 1) * self.width;
            ensure(
                self.observations[row.clone()].iter().all(|x| x.is_finite())
                    && self.next_observations[row].iter().all(|x| x.is_finite()),
                "invalid observation shape or nonfinite observation",
            )?;
            let action_valid = match &self.space {
                ActionSpace::Discrete { count } => {
                    let range = i * *count..(i + 1) * *count;
                    let mask = &self.masks[range.clone()];
                    let next_mask = &self.next_masks[range];
                    ensure(
                        mask.iter().any(|&valid| valid),
                        "invalid or empty live action mask",
                    )?;
                    ensure(
                        self.terminated[i] || next_mask.iter().any(|&valid| valid),
                        "invalid or empty live action mask",
                    )?;
                    self.discrete[i] < *count && mask[self.discrete[i]]
                }
                ActionSpace::Box { low, high } => self.continuous
                    [i * low.len()..(i + 1) * low.len()]
                    .iter()
                    .zip(low.iter().zip(high))
                    .all(|(&x, (&l, &h))| x.is_finite() && x >= l && x <= h),
                ActionSpace::Unbounded { dimensions } => self.continuous
                    [i * *dimensions..(i + 1) * *dimensions]
                    .iter()
                    .all(|x| x.is_finite()),
                ActionSpace::Conditional(_) => false,
            };
            if !action_valid {
                return Err(crate::Error::Action("action violates space or mask".into()));
            }
            ensure(
                self.rewards[i].is_finite()
                    && self.discounts[i].is_finite()
                    && (0.0..=1.0).contains(&self.discounts[i])
                    && (!self.terminated[i] || self.discounts[i] == 0.),
                "invalid replay snapshot item",
            )?;
        }
        Ok(())
    }
    pub fn push(&mut self, item: &ReplayItem) -> Result<()> {
        self.push_view((&item.transition).into(), item.discount)
    }
    pub fn push_view(&mut self, t: TransitionView<'_>, discount: f32) -> Result<()> {
        t.observation.validate(self.width, &self.space, true)?;
        t.next_observation
            .validate(self.width, &self.space, !t.terminated)?;
        self.space.check(t.action, t.observation.dense_mask())?;
        ensure(
            matches!(t.reward, Reward::One(value) if value.is_finite())
                && discount.is_finite()
                && (0.0..=1.0).contains(&discount)
                && (!t.terminated || discount == 0.),
            "invalid replay reward or discount",
        )?;
        let i = self.cursor;
        let range = i * self.width..(i + 1) * self.width;
        self.observations[range.clone()].copy_from_slice(t.observation.dense_values());
        self.next_observations[range].copy_from_slice(t.next_observation.dense_values());
        match &t.action {
            Action::Discrete(a) => self.discrete[i] = *a,
            Action::Continuous(a) => {
                self.continuous[i * a.len()..(i + 1) * a.len()].copy_from_slice(a)
            }
            Action::Conditional(_) => {
                return Err(crate::Error::Action(
                    "dense replay requires numerical actions".into(),
                ));
            }
        }
        if let ActionSpace::Discrete { count } = self.space {
            for (dst, src) in [
                (&mut self.masks, t.observation.dense_mask()),
                (&mut self.next_masks, t.next_observation.dense_mask()),
            ] {
                let row = &mut dst[i * count..(i + 1) * count];
                if let Some(mask) = src {
                    row.copy_from_slice(mask);
                } else {
                    row.fill(true);
                }
            }
        }
        self.rewards[i] = t.reward.total();
        self.discounts[i] = discount;
        self.terminated[i] = t.terminated;
        self.truncated[i] = t.truncated;
        self.cursor = (i + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
        Ok(())
    }
    fn item(&self, i: usize) -> ReplayItem {
        let mask = |buffer: &[bool]| {
            if let ActionSpace::Discrete { count } = self.space {
                Some(buffer[i * count..(i + 1) * count].to_vec())
            } else {
                None
            }
        };
        let action = match self.space {
            ActionSpace::Discrete { .. } => Action::Discrete(self.discrete[i]),
            _ => {
                let d = self.space.dimensions();
                Action::Continuous(self.continuous[i * d..(i + 1) * d].to_vec())
            }
        };
        ReplayItem {
            transition: Transition {
                observation: Observation::dense(
                    self.observations[i * self.width..(i + 1) * self.width].to_vec(),
                    mask(&self.masks),
                ),
                action,
                reward: self.rewards[i].into(),
                next_observation: Observation::dense(
                    self.next_observations[i * self.width..(i + 1) * self.width].to_vec(),
                    mask(&self.next_masks),
                ),
                terminated: self.terminated[i],
                truncated: self.truncated[i],
            },
            discount: self.discounts[i],
        }
    }
    pub fn sample(&self, size: usize, rng: &mut ChaCha8Rng) -> Result<ReplayBatch> {
        ensure(
            self.len > 0 && size > 0,
            "cannot sample empty replay or empty batch",
        )?;
        Ok(ReplayBatch {
            items: (0..size)
                .map(|_| self.item(rng.random_range(0..self.len)))
                .collect(),
        })
    }
    /// Reuses an owned batch. With unchanged dimensions/size this performs no heap allocation
    /// after the first call and consumes exactly the same RNG draws as `sample`.
    pub fn sample_into(
        &self,
        size: usize,
        rng: &mut ChaCha8Rng,
        batch: &mut ReplayBatch,
    ) -> Result<()> {
        ensure(
            self.len > 0 && size > 0,
            "cannot sample empty replay or empty batch",
        )?;
        batch.items.resize_with(size, || self.item(0));
        for slot in &mut batch.items {
            let i = rng.random_range(0..self.len);
            let t = &mut slot.transition;
            let (Some(observation), Some(next)) = (
                t.observation.as_dense_mut(),
                t.next_observation.as_dense_mut(),
            ) else {
                *slot = self.item(i);
                continue;
            };
            let compatible = observation.values.len() == self.width
                && next.values.len() == self.width
                && match (&self.space, &t.action) {
                    (ActionSpace::Discrete { count }, Action::Discrete(_)) => {
                        observation
                            .action_mask
                            .as_ref()
                            .is_some_and(|m| m.len() == *count)
                            && next.action_mask.as_ref().is_some_and(|m| m.len() == *count)
                    }
                    (
                        ActionSpace::Box { .. } | ActionSpace::Unbounded { .. },
                        Action::Continuous(a),
                    ) => {
                        a.len() == self.space.dimensions()
                            && observation.action_mask.is_none()
                            && next.action_mask.is_none()
                    }
                    _ => false,
                };
            if !compatible {
                *slot = self.item(i);
                continue;
            }
            let range = i * self.width..(i + 1) * self.width;
            observation
                .values
                .copy_from_slice(&self.observations[range.clone()]);
            next.values.copy_from_slice(&self.next_observations[range]);
            match &mut t.action {
                Action::Discrete(a) => {
                    *a = self.discrete[i];
                    let ActionSpace::Discrete { count } = self.space else {
                        unreachable!()
                    };
                    observation
                        .action_mask
                        .as_mut()
                        .expect("validated")
                        .copy_from_slice(&self.masks[i * count..(i + 1) * count]);
                    next.action_mask
                        .as_mut()
                        .expect("validated")
                        .copy_from_slice(&self.next_masks[i * count..(i + 1) * count]);
                }
                Action::Continuous(a) => {
                    let d = a.len();
                    a.copy_from_slice(&self.continuous[i * d..(i + 1) * d]);
                }
                Action::Conditional(_) => unreachable!("validated dense replay action"),
            }
            t.reward = self.rewards[i].into();
            t.terminated = self.terminated[i];
            t.truncated = self.truncated[i];
            slot.discount = self.discounts[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn snapshot_validates_live_backing_rows_and_ignores_unused_capacity() {
        let mut replay = ReplayBuffer::new(4, 2, ActionSpace::Discrete { count: 2 }).unwrap();
        replay
            .push(&ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![1., 2.]),
                    action: Action::Discrete(1),
                    reward: 1.0.into(),
                    next_observation: Observation::new(vec![3., 4.]),
                    terminated: false,
                    truncated: true,
                },
                discount: 0.9,
            })
            .unwrap();
        replay.observations[2] = f32::NAN;
        replay.masks[2] = false;
        replay.masks[3] = false;
        replay.validate_state().unwrap();
        replay.next_masks[0] = false;
        replay.next_masks[1] = false;
        assert!(replay.validate_state().is_err());
        replay.next_masks[0] = true;
        replay.observations[0] = f32::NAN;
        assert!(replay.validate_state().is_err());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PendingLane {
    slots: Vec<Transition>,
    head: usize,
    len: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NStepAccumulator {
    n: usize,
    gamma: f32,
    pending: Vec<PendingLane>,
}
impl NStepAccumulator {
    /// Discard incomplete trajectories when collection is explicitly reset after a failure.
    pub fn clear(&mut self) {
        for lane in &mut self.pending {
            lane.head = 0;
            lane.len = 0;
        }
    }
    pub(crate) fn check_config(&self, n: usize, gamma: f32) -> Result<()> {
        ensure(
            self.n == n && self.gamma == gamma,
            "pending n-step configuration differs from trainer",
        )
    }
    pub fn new(n: usize, gamma: f32, lanes: usize) -> Result<Self> {
        ensure(
            n > 0 && lanes > 0 && gamma.is_finite() && (0.0..=1.0).contains(&gamma),
            "invalid n-step configuration",
        )?;
        Ok(Self {
            n,
            gamma,
            pending: vec![PendingLane::default(); lanes],
        })
    }
    pub fn push(&mut self, lane: usize, transition: Transition) -> Result<Vec<ReplayItem>> {
        let mut result = Vec::new();
        self.push_with(lane, (&transition).into(), |view, discount| {
            result.push(ReplayItem {
                transition: view.to_owned(),
                discount,
            });
            Ok(())
        })?;
        Ok(result)
    }
    fn push_with(
        &mut self,
        lane: usize,
        t: TransitionView<'_>,
        mut emit: impl FnMut(TransitionView<'_>, f32) -> Result<()>,
    ) -> Result<()> {
        ensure(lane < self.pending.len(), "invalid n-step lane")?;
        match t.reward {
            Reward::One(_) => t.reward.validate()?,
            trace => {
                trace.discounted(self.gamma)?;
            }
        }
        let queue = &mut self.pending[lane];
        if queue.slots.is_empty() {
            queue.slots.resize_with(self.n, || t.to_owned());
        }
        let index = (queue.head + queue.len) % self.n;
        copy_transition(&mut queue.slots[index], t);
        queue.len += 1;
        while queue.len >= self.n || (t.ended() && queue.len > 0) {
            let mut reward = 0.;
            let mut discount = 1.;
            let (wrapped, contiguous) = queue.slots[..self.n].split_at(queue.head);
            let first = &contiguous[0];
            let mut final_item = first;
            for item in contiguous.iter().chain(wrapped).take(queue.len.min(self.n)) {
                final_item = item;
                let (accumulated, continuation) = match &item.reward {
                    Reward::One(value) => (*value, self.gamma),
                    trace => trace.discounted(self.gamma)?,
                };
                reward += discount * accumulated;
                discount *= continuation;
                if item.ended() {
                    if item.terminated {
                        discount = 0.;
                    }
                    break;
                }
            }
            emit(
                TransitionView {
                    observation: &first.observation,
                    action: &first.action,
                    reward: &Reward::One(reward),
                    next_observation: &final_item.next_observation,
                    terminated: final_item.terminated,
                    truncated: final_item.truncated,
                },
                discount,
            )?;
            queue.head += 1;
            if queue.head == self.n {
                queue.head = 0;
            }
            queue.len -= 1;
        }
        Ok(())
    }
    pub(crate) fn validate_state(
        &self,
        width: usize,
        space: &ActionSpace,
        lanes: usize,
    ) -> Result<()> {
        ensure(
            self.n > 0
                && self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma)
                && self.pending.len() == lanes,
            "invalid n-step snapshot",
        )?;
        for queue in &self.pending {
            ensure(
                queue.len < self.n
                    && queue.head < self.n
                    && (queue.slots.len() == self.n || (queue.len == 0 && queue.slots.is_empty())),
                "invalid pending ring dimensions",
            )?;
            for offset in 0..queue.len {
                let t = &queue.slots[(queue.head + offset) % self.n];
                t.observation.validate(width, space, true)?;
                t.next_observation.validate(width, space, true)?;
                space.check(&t.action, t.observation.dense_mask())?;
                t.reward.discounted(self.gamma)?;
                ensure(!t.ended(), "invalid pending transition")?;
            }
        }
        Ok(())
    }
}

pub(crate) fn copy_transition(dst: &mut Transition, src: TransitionView<'_>) {
    copy_observation(&mut dst.observation, src.observation);
    copy_observation(&mut dst.next_observation, src.next_observation);
    match (&mut dst.action, src.action) {
        (Action::Continuous(a), Action::Continuous(b)) => a.clone_from(b),
        (a, b) => *a = b.clone(),
    }
    dst.reward.clone_from(src.reward);
    dst.terminated = src.terminated;
    dst.truncated = src.truncated;
}
pub(crate) fn copy_observation(dst: &mut Observation, src: &Observation) {
    if let (Some(dst), Some(src)) = (dst.as_dense_mut(), src.as_dense()) {
        dst.values.clone_from(&src.values);
        dst.action_mask.clone_from(&src.action_mask);
    } else {
        dst.clone_from(src);
    }
}

/// Borrowed persistence endpoint; collection itself owns neither replay nor pending trajectories.
pub struct ReplayWriter<'a> {
    replay: &'a mut ReplayBuffer,
    pending: Option<&'a mut NStepAccumulator>,
    gamma: f32,
    lanes: usize,
}
impl<'a> ReplayWriter<'a> {
    pub fn new(
        replay: &'a mut ReplayBuffer,
        pending: Option<&'a mut NStepAccumulator>,
        gamma: f32,
        lanes: usize,
    ) -> Result<Self> {
        ensure(
            gamma.is_finite() && (0.0..=1.0).contains(&gamma) && lanes > 0,
            "invalid replay writer",
        )?;
        if let Some(p) = &pending {
            ensure(
                p.gamma == gamma && p.pending.len() == lanes && p.n > 1,
                "replay writer n-step configuration mismatch",
            )?;
        }
        Ok(Self {
            replay,
            pending,
            gamma,
            lanes,
        })
    }
    pub(crate) fn check_shape(
        &self,
        width: usize,
        space: &ActionSpace,
        lanes: usize,
    ) -> Result<()> {
        ensure(
            self.replay.width == width && &self.replay.space == space && self.lanes == lanes,
            "collector/replay configuration mismatch",
        )
    }
    pub fn push(&mut self, lane: usize, t: TransitionView<'_>) -> Result<()> {
        ensure(lane < self.lanes, "invalid replay lane")?;
        if self.pending.is_none() {
            if let Reward::Many(_) = t.reward {
                let (accumulated, continuation) = t.reward.discounted(self.gamma)?;
                return self.replay.push_view(
                    TransitionView {
                        reward: &Reward::One(accumulated),
                        ..t
                    },
                    if t.terminated { 0. } else { continuation },
                );
            }
            return self
                .replay
                .push_view(t, if t.terminated { 0. } else { self.gamma });
        }
        t.observation
            .validate(self.replay.width, &self.replay.space, true)?;
        t.next_observation
            .validate(self.replay.width, &self.replay.space, !t.terminated)?;
        self.replay
            .space
            .check(t.action, t.observation.dense_mask())?;
        self.pending
            .as_mut()
            .expect("pending checked above")
            .push_with(lane, t, |view, d| self.replay.push_view(view, d))
    }
}
