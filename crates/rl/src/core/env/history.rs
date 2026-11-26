use super::{Action, ActionSpace, Observation, ObservationSpace, Reward};
use crate::{
    Error, Result,
    core::stateful::{HistoryInput, PreviousInput},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "neural")]
use std::sync::Weak;
use std::{fmt, sync::Arc};

/// Shared, immutable public frames from a genuine reset through a live observation.
#[derive(Clone)]
pub struct PublicHistory(Arc<Frame>);

/// Weak ownership pins the frame identity without retaining its public tape.
#[cfg(feature = "neural")]
#[derive(Debug)]
pub(crate) struct WeakPublicHistory(Weak<Frame>);

#[cfg(feature = "neural")]
impl WeakPublicHistory {
    pub(crate) fn is_alive(&self) -> bool {
        self.0.strong_count() > 0
    }
}

struct Frame {
    observation: Observation,
    timestep: u64,
    physical_timestep: u64,
    previous: Option<Link>,
}

struct Link {
    parent: PublicHistory,
    action: Action,
    reward: Reward,
}

impl Drop for Frame {
    fn drop(&mut self) {
        // Drop a uniquely owned suffix iteratively, stopping at a shared parent.
        let mut previous = self.previous.take();
        while let Some(link) = previous {
            let Some(mut parent) = Arc::into_inner(link.parent.0) else {
                break;
            };
            previous = parent.previous.take();
        }
    }
}

impl fmt::Debug for PublicHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicHistory")
            .field("timestep", &self.timestep())
            .finish_non_exhaustive()
    }
}

impl PublicHistory {
    pub fn new(observation: Observation) -> Self {
        Self(Arc::new(Frame {
            observation,
            timestep: 0,
            physical_timestep: 0,
            previous: None,
        }))
    }

    pub fn append(
        &self,
        action: Action,
        reward: impl Into<Reward>,
        observation: Observation,
    ) -> Result<Self> {
        let reward = reward.into();
        reward.validate()?;
        observation.validate_event_history(Some(self.observation()))?;
        let timestep = self
            .timestep()
            .checked_add(1)
            .ok_or_else(|| Error::State("public-history clock exhausted".into()))?;
        let physical_timestep = self
            .physical_timestep()
            .checked_add(reward.duration())
            .ok_or_else(|| Error::State("public-history physical clock exhausted".into()))?;
        Ok(Self(Arc::new(Frame {
            observation,
            timestep,
            physical_timestep,
            previous: Some(Link {
                parent: self.clone(),
                action,
                reward,
            }),
        })))
    }

    pub fn observation(&self) -> &Observation {
        &self.0.observation
    }

    pub fn timestep(&self) -> u64 {
        self.0.timestep
    }

    pub fn physical_timestep(&self) -> u64 {
        self.0.physical_timestep
    }

    #[cfg(feature = "neural")]
    pub(crate) fn frame_identity(&self) -> usize {
        Arc::as_ptr(&self.0).addr()
    }

    #[cfg(feature = "neural")]
    pub(crate) fn downgrade(&self) -> WeakPublicHistory {
        WeakPublicHistory(Arc::downgrade(&self.0))
    }

    #[cfg(feature = "neural")]
    pub(crate) fn parent(&self) -> Option<&Self> {
        self.0.previous.as_ref().map(|link| &link.parent)
    }

    pub(crate) fn previous_input(&self) -> Option<PreviousInput<'_>> {
        self.0.previous.as_ref().map(|link| {
            PreviousInput::from_transition(&link.action, &link.reward, link.parent.observation())
        })
    }

    pub(crate) fn matches(&self, observation: &Observation, timestep: u64) -> bool {
        self.physical_timestep() == timestep && self.observation().matches(observation)
    }

    /// Borrow the chronological model inputs, including the final observation.
    pub fn inputs(&self) -> Vec<HistoryInput<'_>> {
        let mut inputs = Vec::new();
        let mut frame = &*self.0;
        loop {
            inputs.push(HistoryInput {
                observation: &frame.observation,
                previous: frame.previous.as_ref().map(|link| {
                    PreviousInput::from_transition(
                        &link.action,
                        &link.reward,
                        link.parent.observation(),
                    )
                }),
            });
            let Some(link) = &frame.previous else {
                break;
            };
            frame = &link.parent.0;
        }
        inputs.reverse();
        inputs
    }

    pub fn validate(&self, width: usize, space: &ActionSpace) -> Result<()> {
        self.validate_in(&ObservationSpace::Dense(width), space)
    }

    pub fn validate_in(&self, observations: &ObservationSpace, space: &ActionSpace) -> Result<()> {
        let mut frame = &*self.0;
        loop {
            frame.observation.validate_in(observations, space, true)?;
            let Some(link) = &frame.previous else {
                frame.observation.validate_event_history(None)?;
                return Ok(());
            };
            frame
                .observation
                .validate_event_history(Some(link.parent.observation()))?;
            frame = &link.parent.0;
            frame.observation.check_action(space, &link.action)?;
        }
    }
}

impl Serialize for PublicHistory {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut frames = Vec::new();
        let mut cursor = Some(&*self.0);
        while let Some(frame) = cursor {
            frames.push(frame);
            cursor = frame.previous.as_ref().map(|link| &*link.parent.0);
        }
        serializer.collect_seq(frames.into_iter().rev().enumerate().map(|(index, frame)| {
            EncodedFrame {
                observation: &frame.observation,
                previous: frame.previous.as_ref().map(|link| EncodedLink {
                    parent: index - 1,
                    action: &link.action,
                    reward: link.reward.clone(),
                }),
            }
        }))
    }
}

impl<'de> Deserialize<'de> for PublicHistory {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let frames = Vec::<EncodedFrame<Observation, Action>>::deserialize(deserializer)?;
        let mut frames = frames.into_iter();
        let root = frames
            .next()
            .filter(|frame| frame.previous.is_none())
            .ok_or_else(|| {
                serde::de::Error::custom("a public history requires its initial observation")
            })?;
        let mut history = PublicHistory::new(root.observation);
        for (parent, frame) in frames.enumerate() {
            let link = frame
                .previous
                .filter(|link| link.parent == parent)
                .ok_or_else(|| serde::de::Error::custom("expected one complete public history"))?;
            history = history
                .append(link.action, link.reward, frame.observation)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(history)
    }
}

/// Chronological public frames with a link to each preceding frame.
#[derive(Serialize, Deserialize)]
struct EncodedFrame<O, A> {
    observation: O,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous: Option<EncodedLink<A>>,
}

#[derive(Serialize, Deserialize)]
struct EncodedLink<A> {
    parent: usize,
    action: A,
    reward: Reward,
}
