use super::schedule::Schedule;
use crate::{
    Result,
    core::{ensure, temporal::ReturnSpec},
    train::counters::RunCounters,
};
use serde::{Deserialize, Serialize};

pub(super) fn valid_replay_updates(capacity: usize, batch: usize, rate: f64) -> bool {
    capacity > 0 && batch > 0 && rate.is_finite() && rate > 0.0
}
/// Per-call additional budgets. Stop at the first reached limit, at a collection/update boundary.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Budget {
    pub env_steps: Option<u64>,
    pub vector_steps: Option<u64>,
    pub episodes: Option<u64>,
    pub updates: Option<u64>,
}
impl Budget {
    pub fn vector_steps(n: u64) -> Self {
        Self {
            vector_steps: Some(n),
            ..Default::default()
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            [
                self.env_steps,
                self.vector_steps,
                self.episodes,
                self.updates,
            ]
            .iter()
            .any(Option::is_some),
            "a finite training budget is required",
        )
    }
    pub(crate) fn reached(&self, start: &RunCounters, now: &RunCounters) -> bool {
        [
            (self.env_steps, now.env_steps - start.env_steps),
            (self.vector_steps, now.vector_steps - start.vector_steps),
            (self.episodes, now.episodes - start.episodes),
            (self.updates, now.updates - start.updates),
        ]
        .iter()
        .any(|(limit, n)| limit.is_some_and(|limit| *n >= limit))
    }
    pub(crate) fn remaining_vector_steps(
        &self,
        start: &RunCounters,
        now: &RunCounters,
        lanes: usize,
    ) -> u64 {
        let v = self.vector_steps.map_or(u64::MAX, |n| {
            n.saturating_sub(now.vector_steps - start.vector_steps)
        });
        let e = self.env_steps.map_or(u64::MAX, |n| {
            n.saturating_sub(now.env_steps - start.env_steps)
                .div_ceil(lanes as u64)
        });
        v.min(e)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnPolicyConfig {
    pub rollout_steps: usize,
    pub episode_guard: Option<usize>,
    pub gamma: f32,
    pub gae_lambda: Option<f32>,
    pub normalize_advantages: bool,
}
impl Default for OnPolicyConfig {
    fn default() -> Self {
        Self {
            rollout_steps: 64,
            episode_guard: None,
            gamma: 0.99,
            gae_lambda: Some(0.95),
            normalize_advantages: true,
        }
    }
}
impl OnPolicyConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.rollout_steps > 0
                && self.episode_guard.is_none_or(|x| x > 0)
                && self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma)
                && self
                    .gae_lambda
                    .is_none_or(|x| x.is_finite() && (0.0..=1.0).contains(&x)),
            "invalid on-policy trainer settings",
        )?;
        ensure(
            self.episode_guard.is_none() || self.gae_lambda.is_none(),
            "complete Monte Carlo collection requires gae_lambda=None",
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffPolicyConfig {
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub warmup_transitions: u64,
    pub n_step: usize,
    pub gamma: f32,
    pub updates_per_transition: f64,
    pub exploration: Schedule,
}
impl Default for OffPolicyConfig {
    fn default() -> Self {
        Self {
            replay_capacity: 100_000,
            batch_size: 64,
            warmup_transitions: 1_000,
            n_step: 1,
            gamma: 0.99,
            updates_per_transition: 1.,
            exploration: Schedule::Constant(0.1),
        }
    }
}
impl OffPolicyConfig {
    pub(crate) fn validate_requirements(
        &self,
        requirements: Option<crate::core::learner::OffPolicyRequirements>,
    ) -> Result<()> {
        let Some(required) = requirements else {
            return Ok(());
        };
        required.returns.validate()?;
        ensure(
            self.gamma.to_bits() == required.returns.gamma.to_bits()
                && self.n_step == required.returns.n_step,
            "replay return contract differs from learner",
        )?;
        if required.exploration == crate::core::learner::ExplorationSource::Policy {
            // Inspect f64 endpoints, not a rounded current f32 value.
            let zero = match self.exploration {
                Schedule::Constant(x) => x == 0.,
                Schedule::Linear { start, end, .. } => start == 0. && end == 0.,
            };
            ensure(
                zero,
                "policy supplies exploration; external schedule must be zero",
            )?;
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            valid_replay_updates(
                self.replay_capacity,
                self.batch_size,
                self.updates_per_transition,
            ) && (ReturnSpec {
                gamma: self.gamma,
                n_step: self.n_step,
            })
            .validate()
            .is_ok(),
            "invalid off-policy trainer settings",
        )?;
        ensure(
            self.exploration.value(&RunCounters::default())? >= 0.,
            "negative exploration",
        )
    }
}

#[cfg(test)]
mod replay_contract_tests {
    use super::*;
    use crate::core::{
        learner::{ExplorationSource, OffPolicyRequirements},
        temporal::ReturnSpec,
    };
    #[test]
    fn policy_exploration_checks_entire_schedule_without_float_underflow() {
        let required = Some(OffPolicyRequirements {
            returns: ReturnSpec {
                gamma: 0.99,
                n_step: 1,
            },
            exploration: ExplorationSource::Policy,
        });
        let mut config = OffPolicyConfig {
            exploration: Schedule::Constant(0.),
            ..Default::default()
        };
        config.validate_requirements(required).unwrap();
        for schedule in [
            Schedule::Constant(1e-300),
            Schedule::Linear {
                counter: crate::train::schedule::Counter::EnvironmentTransitions,
                start: 0.,
                end: 0.1,
                duration: 100,
            },
        ] {
            config.exploration = schedule;
            assert!(config.validate_requirements(required).is_err());
        }
        config.exploration = Schedule::Constant(0.);
        config.n_step = 2;
        assert!(config.validate_requirements(required).is_err());
        config.n_step = 1;
        config.gamma = 1.;
        assert!(config.validate_requirements(required).is_err());
    }
}
