//! Optional rich stochastic sampling without changing the basic stateful runtime.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::ActMode,
        ensure,
        env::Observation,
        stateful::{StatefulActor, StatefulDecision},
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One semantic candidate and its complete-action log probability.
/// Disabled candidates retain their identity and have no probability.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalOutcome {
    pub action: ConditionalAction,
    pub log_prob: Option<f32>,
}

/// One host-owned row of a behavior distribution, including its exact support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionSnapshot {
    Categorical {
        logits: Vec<f32>,
        masks: Vec<bool>,
    },
    Gaussian {
        mean: Vec<f32>,
        log_std: Vec<f32>,
        bounds: Option<(Vec<f32>, Vec<f32>)>,
    },
    Conditional {
        outcomes: Arc<[ConditionalOutcome]>,
    },
}
impl DistributionSnapshot {
    pub fn conditional(mut outcomes: Vec<ConditionalOutcome>) -> Result<Self> {
        outcomes.sort_unstable_by_key(|outcome| outcome.action);
        let result = Self::Conditional {
            outcomes: outcomes.into(),
        };
        result.validate()?;
        Ok(result)
    }

    pub fn conditional_outcomes(&self) -> Result<&[ConditionalOutcome]> {
        match self {
            Self::Conditional { outcomes } => Ok(outcomes),
            _ => Err(crate::Error::Shape(
                "expected a conditional distribution snapshot".into(),
            )),
        }
    }

    pub(crate) fn validate_conditional_support(&self, observation: &Observation) -> Result<()> {
        let outcomes = self.conditional_outcomes()?;
        let Observation::Structured(frame) = observation else {
            return Err(crate::Error::Shape(
                "conditional behavior requires a structured observation".into(),
            ));
        };
        ensure(
            outcomes.len() == frame.candidates.len()
                && frame.candidates.iter().all(|candidate| {
                    outcomes
                        .binary_search_by_key(&candidate.action, |outcome| outcome.action)
                        .is_ok_and(|index| outcomes[index].log_prob.is_some() == candidate.enabled)
                }),
            "conditional behavior support differs from observation",
        )
    }

    pub fn validate(&self) -> Result<()> {
        let valid = match self {
            Self::Categorical { logits, masks } => {
                !logits.is_empty()
                    && logits.len() == masks.len()
                    && logits.iter().all(|x| x.is_finite())
                    && masks.iter().any(|&x| x)
            }
            Self::Gaussian {
                mean,
                log_std,
                bounds,
            } => {
                !mean.is_empty()
                    && mean.len() == log_std.len()
                    && mean.iter().chain(log_std).all(|x| x.is_finite())
                    && bounds.as_ref().is_none_or(|(low, high)| {
                        low.len() == mean.len()
                            && high.len() == mean.len()
                            && low.iter().zip(high).all(|(&l, &h)| {
                                l.is_finite() && h.is_finite() && l < h && (h - l).is_finite()
                            })
                    })
            }
            Self::Conditional { outcomes } => {
                outcomes
                    .windows(2)
                    .all(|pair| pair[0].action < pair[1].action)
                    && outcomes.iter().all(|outcome| {
                        outcome
                            .log_prob
                            .is_none_or(|lp| lp.is_finite() && lp <= 1e-5)
                    })
                    && (outcomes
                        .iter()
                        .filter_map(|outcome| outcome.log_prob)
                        .map(|lp| f64::from(lp).exp())
                        .sum::<f64>()
                        - 1.)
                        .abs()
                        <= 1e-5
            }
        };
        ensure(valid, "invalid on-policy distribution snapshot")
    }
    pub fn validate_for(&self, space: &ActionSpace) -> Result<()> {
        self.validate()?;
        space.validate()?;
        let matches = match (self, space) {
            (Self::Categorical { logits, .. }, ActionSpace::Discrete { count }) => {
                logits.len() == *count
            }
            (
                Self::Gaussian {
                    mean, bounds: None, ..
                },
                ActionSpace::Unbounded { dimensions },
            ) => mean.len() == *dimensions,
            (
                Self::Gaussian {
                    bounds: Some((a, b)),
                    ..
                },
                ActionSpace::Box { low, high },
            ) => a == low && b == high,
            (Self::Conditional { outcomes }, ActionSpace::Conditional(grammar)) => outcomes
                .iter()
                .all(|outcome| grammar.accepts(&outcome.action)),
            _ => false,
        };
        ensure(
            matches,
            "on-policy distribution snapshot action space mismatch",
        )
    }
}

// Host/batched f32 reductions can differ by a few rounding units. The
// reference for these checks comes from the frozen collection model.
#[cfg(feature = "neural")]
pub(crate) fn recording_number_matches(reference: f32, recorded: f32) -> bool {
    reference.is_finite()
        && recorded.is_finite()
        && (reference - recorded).abs() <= 1e-5 * (1. + reference.abs())
}

#[cfg(feature = "neural")]
pub(crate) fn recording_distribution_matches(
    reference: &DistributionSnapshot,
    recorded: &DistributionSnapshot,
) -> bool {
    let close_slice = |a: &[f32], b: &[f32]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(&a, &b)| recording_number_matches(a, b))
    };
    match (reference, recorded) {
        (
            DistributionSnapshot::Categorical {
                logits: a,
                masks: am,
            },
            DistributionSnapshot::Categorical {
                logits: b,
                masks: bm,
            },
        ) => am == bm && close_slice(a, b),
        (
            DistributionSnapshot::Gaussian {
                mean: a,
                log_std: al,
                bounds: ab,
            },
            DistributionSnapshot::Gaussian {
                mean: b,
                log_std: bl,
                bounds: bb,
            },
        ) => ab == bb && close_slice(a, b) && close_slice(al, bl),
        (
            DistributionSnapshot::Conditional { outcomes: a },
            DistributionSnapshot::Conditional { outcomes: b },
        ) => {
            a.len() == b.len()
                && a.iter().zip(b.iter()).all(|(a, b)| {
                    a.action == b.action
                        && match (a.log_prob, b.log_prob) {
                            (Some(a), Some(b)) => recording_number_matches(a, b),
                            (None, None) => true,
                            _ => false,
                        }
                })
        }
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub struct OnPolicySample {
    pub decision: StatefulDecision,
    /// The original sampled pre-transform Gaussian latent, never an inverse tanh.
    pub latent: Option<Vec<f32>>,
    pub distribution: DistributionSnapshot,
    pub value: Option<f32>,
}

/// Sampling and scalar-value queries read already assimilated post-observation state.
pub trait StatefulOnPolicyActor: StatefulActor {
    fn sample_on_policy_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "canonical sampling count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .zip(rngs)
            .map(|((state, observation), rng)| self.sample_on_policy(state, observation, mode, rng))
            .collect()
    }
    fn state_values_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        ensure(
            states.len() == observations.len(),
            "canonical value count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .map(|(state, observation)| self.state_value(state, observation))
            .collect()
    }
    fn sample_on_policy(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample>;
    fn state_value(&self, state: &Self::State, observation: &Observation) -> Result<Option<f32>>;
    fn sample_on_policy_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "on-policy sampling batch count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .zip(rngs)
            .map(|((state, observation), rng)| self.sample_on_policy(state, observation, mode, rng))
            .collect()
    }
    fn state_values_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        ensure(
            states.len() == observations.len(),
            "on-policy value batch count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .map(|(state, observation)| self.state_value(state, observation))
            .collect()
    }
}

/// Identity and numerical audit for frozen on-policy behavior.
pub trait OnPolicyIdentity: StatefulOnPolicyActor {
    fn behavior_identity(&self) -> [u8; 32];
    /// Exact by default for arbitrary structured state. Concrete neural codecs
    /// can opt into a measured import bound, independently for each memory.
    fn audit_portable_memory(
        &self,
        saved: &Self::State,
        native: &Self::State,
    ) -> Result<Vec<crate::core::checkpoint::PortableMemoryAudit>>
    where
        Self::State: PartialEq,
    {
        self.validate_state(saved)?;
        self.validate_state(native)?;
        ensure(
            saved == native,
            "actor does not support foreign-runtime memory drift",
        )?;
        Ok(Vec::new())
    }
    /// Pure numerical audit against an independently reconstructed post-observation
    /// state. Reads heads only: no memory advancement, action sampling, or RNG.
    #[allow(clippy::too_many_arguments)]
    fn validate_recording(
        &self,
        state: &Self::State,
        observation: &Observation,
        action: &Action,
        latent: Option<&[f32]>,
        distribution: &DistributionSnapshot,
        log_prob: f32,
        value: Option<f32>,
    ) -> Result<()>;
}
