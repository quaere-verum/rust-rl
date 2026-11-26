//! Physical rewards and the discount carried across a decision boundary.
use super::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Reward objective for statistical batch targets. Physical rewards are the
/// ordinary contract. TerminalSuccess is an opt-in, undiscounted projection
/// with one explicit success/failure label on the final primitive tick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum BatchObjective {
    Physical { gamma: f64 },
    TerminalSuccess { success: f32, failure: f32 },
}

impl BatchObjective {
    pub fn gamma(self) -> f64 {
        match self {
            Self::Physical { gamma } => gamma,
            Self::TerminalSuccess { .. } => 1.,
        }
    }

    pub fn validate(self) -> Result<()> {
        match self {
            Self::Physical { gamma } => {
                ensure((0.0..=1.0).contains(&gamma), "invalid batch discount")
            }
            Self::TerminalSuccess { success, failure } => ensure(
                success.is_finite() && failure.is_finite() && success != failure,
                "invalid terminal-success reward labels",
            ),
        }
    }
}

/// Potentials at both sides of a decision, optionally computed from public time.
/// Use `None` for a terminal or a modeled zero-value horizon boundary. Use the
/// successor's potential when bootstrapping after a factual truncation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotentialValues {
    pub current: f64,
    pub next: Option<f64>,
}

impl PotentialValues {
    /// Compute the shaped learning reward from an already discounted aggregate.
    /// `continuation_discount` is gamma raised to the action's physical duration.
    /// The factual reward trace remains available to evaluate the original objective.
    pub fn shape(self, reward: f64, continuation_discount: f64) -> Result<f64> {
        let next = self.next.unwrap_or(0.);
        ensure(
            reward.is_finite()
                && self.current.is_finite()
                && next.is_finite()
                && (0. ..=1.).contains(&continuation_discount),
            "invalid reward, discount, or potential for reward shaping",
        )?;
        let shaped = reward + continuation_discount.mul_add(next, -self.current);
        ensure(shaped.is_finite(), "nonfinite shaped learning reward")?;
        Ok(shaped)
    }
}

/// One reward is stored inline. An extended action retains the actual
/// chronological rewards, even when several of them are zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Reward {
    One(f32),
    Many(Arc<[f32]>),
}

impl From<f32> for Reward {
    fn from(reward: f32) -> Self {
        Self::One(reward)
    }
}

impl Reward {
    pub fn from_primitives(rewards: Vec<f32>) -> Result<Self> {
        let reward = match rewards.as_slice() {
            [one] => Self::One(*one),
            _ => Self::Many(rewards.into()),
        };
        reward.validate()?;
        Ok(reward)
    }

    pub fn primitives(&self) -> &[f32] {
        match self {
            Self::One(value) => std::slice::from_ref(value),
            Self::Many(values) => values,
        }
    }

    pub fn duration(&self) -> u64 {
        self.primitives().len() as u64
    }

    pub fn total(&self) -> f32 {
        match self {
            Self::One(value) => *value,
            Self::Many(values) => values.iter().map(|&r| f64::from(r)).sum::<f64>() as f32,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::One(value) => ensure(value.is_finite(), "nonfinite reward"),
            Self::Many(values) => ensure(
                values.len() > 1
                    && values.iter().all(|value| value.is_finite())
                    && self.total().is_finite(),
                "invalid primitive reward trace",
            ),
        }
    }

    /// The discounted accumulated reward and gamma to the power of the actual
    /// duration. The caller gates the continuation on true termination.
    pub fn discounted(&self, gamma: f32) -> Result<(f32, f32)> {
        let (reward, discount) = self.discounted_f64(f64::from(gamma))?;
        let reward = reward as f32;
        ensure(reward.is_finite(), "discounted primitive reward overflow")?;
        Ok((reward, discount as f32))
    }

    /// The same chronological accumulation at planner precision. This avoids
    /// rounding an f64 planning discount through f32 when joining factual returns.
    /// Ordinary learners retain the historical f32 outputs of `discounted`.
    pub fn discounted_f64(&self, gamma: f64) -> Result<(f64, f64)> {
        ensure(
            gamma.is_finite() && (0.0..=1.0).contains(&gamma),
            "invalid physical-time discount",
        )?;
        self.validate()?;
        let Self::Many(values) = self else {
            return Ok((f64::from(self.total()), gamma));
        };
        let mut reward = 0.0_f64;
        let mut discount = 1.0_f64;
        for &value in values.as_ref() {
            reward += discount * f64::from(value);
            discount *= gamma;
        }
        ensure(reward.is_finite(), "discounted primitive reward overflow")?;
        Ok((reward, discount))
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        let a = self.primitives();
        let b = other.primitives();
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
}
