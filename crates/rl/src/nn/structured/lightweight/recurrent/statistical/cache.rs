//! A private cache for one frozen encoder and immutable public frames.

use super::RecurrentLightweightState;
use crate::{
    Error, Result,
    core::env::{PublicHistory, WeakPublicHistory},
};
use std::{
    collections::{HashMap, VecDeque},
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

#[derive(Debug)]
pub(super) struct Incremental {
    capacity: NonZeroUsize,
    frames: Mutex<Frames>,
}

#[derive(Debug, Default)]
struct Frames {
    states: HashMap<usize, Arc<CachedFrame>>,
    order: VecDeque<usize>,
}

#[derive(Debug)]
pub(super) struct CachedFrame {
    history: WeakPublicHistory,
    pub state: RecurrentLightweightState,
    pub values: Vec<f32>,
}

pub(super) enum Lookup<'a> {
    Complete(Arc<CachedFrame>),
    Ancestor {
        frame: Arc<CachedFrame>,
        suffix: Vec<&'a PublicHistory>,
    },
    Cold,
}

impl Clone for Incremental {
    fn clone(&self) -> Self {
        Self::new(self.capacity)
    }
}

impl Incremental {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            capacity,
            frames: Mutex::default(),
        }
    }

    pub fn capacity(&self) -> NonZeroUsize {
        self.capacity
    }

    pub fn lookup<'a>(&self, history: &'a PublicHistory) -> Result<Lookup<'a>> {
        let frames = self
            .frames
            .lock()
            .map_err(|_| Error::State("recurrent feature cache is poisoned".into()))?;
        if frames.states.is_empty() {
            return Ok(Lookup::Cold);
        }
        let mut suffix = Vec::new();
        let mut current = Some(history);
        while let Some(frame) = current {
            if let Some(cached) = frames.states.get(&frame.frame_identity()) {
                return Ok(if suffix.is_empty() {
                    Lookup::Complete(cached.clone())
                } else {
                    Lookup::Ancestor {
                        frame: cached.clone(),
                        suffix,
                    }
                });
            }
            suffix.push(frame);
            current = frame.parent();
        }
        Ok(Lookup::Cold)
    }

    pub fn remember(
        &self,
        history: &PublicHistory,
        state: RecurrentLightweightState,
        values: Vec<f32>,
    ) -> Result<Arc<CachedFrame>> {
        let mut frames = self
            .frames
            .lock()
            .map_err(|_| Error::State("recurrent feature cache is poisoned".into()))?;
        let key = history.frame_identity();
        if let Some(cached) = frames.states.get(&key) {
            return Ok(cached.clone());
        }
        while let Some(&first) = frames.order.front() {
            if frames.states.len() < self.capacity.get() && frames.states[&first].history.is_alive()
            {
                break;
            }
            frames.order.pop_front();
            frames.states.remove(&first);
        }
        let cached = Arc::new(CachedFrame {
            history: history.downgrade(),
            state,
            values,
        });
        frames.states.insert(key, cached.clone());
        frames.order.push_back(key);
        Ok(cached)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{action::Action, env::Observation};
    use burn::tensor::{Device, Tensor};

    #[test]
    fn cached_frames_keep_bounded_weak_tapes_and_resume_only_exact_ancestors() -> Result<()> {
        let cache = Incremental::new(NonZeroUsize::new(2).unwrap());
        let observation = Observation::dense(vec![1.0], None);
        let first = PublicHistory::new(observation.clone());
        let other = PublicHistory::new(observation.clone());
        let next = first.append(Action::Discrete(0), 0.0, observation)?;
        let state = RecurrentLightweightState {
            memory: Tensor::zeros([1, 1], &Device::default()),
            event_start: 0,
        };
        cache.remember(&first, state.clone(), vec![1.0])?;
        assert!(matches!(cache.lookup(&first)?, Lookup::Complete(_)));
        assert!(matches!(cache.lookup(&other)?, Lookup::Cold));
        let Lookup::Ancestor { suffix, .. } = cache.lookup(&next)? else {
            panic!("cached parent")
        };
        assert_eq!(suffix.len(), 1);
        assert_eq!(suffix[0].frame_identity(), next.frame_identity());
        assert!(matches!(cache.clone().lookup(&first)?, Lookup::Cold));
        cache.remember(&other, state.clone(), vec![2.0])?;
        cache.remember(&next, state, vec![3.0])?;
        assert!(matches!(cache.lookup(&first)?, Lookup::Cold));
        assert_eq!(cache.frames.lock().unwrap().states.len(), 2);
        let weak_next = next.downgrade();
        drop(next);
        assert!(!weak_next.is_alive());
        Ok(())
    }
}
