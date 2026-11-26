use super::{action::Action, env::Observation, reward::Reward};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub struct TransitionView<'a> {
    pub observation: &'a super::env::Observation,
    pub action: &'a super::action::Action,
    pub reward: &'a Reward,
    pub next_observation: &'a super::env::Observation,
    pub terminated: bool,
    pub truncated: bool,
}
impl TransitionView<'_> {
    pub fn ended(self) -> bool {
        self.terminated || self.truncated
    }
    pub fn to_owned(self) -> Transition {
        Transition {
            observation: self.observation.clone(),
            action: self.action.clone(),
            reward: self.reward.clone(),
            next_observation: self.next_observation.clone(),
            terminated: self.terminated,
            truncated: self.truncated,
        }
    }
}
impl<'a> From<&'a Transition> for TransitionView<'a> {
    fn from(t: &'a Transition) -> Self {
        Self {
            observation: &t.observation,
            action: &t.action,
            reward: &t.reward,
            next_observation: &t.next_observation,
            terminated: t.terminated,
            truncated: t.truncated,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub observation: Observation,
    pub action: Action,
    pub reward: Reward,
    pub next_observation: Observation,
    pub terminated: bool,
    pub truncated: bool,
}
impl Transition {
    pub fn ended(&self) -> bool {
        self.terminated || self.truncated
    }
}
