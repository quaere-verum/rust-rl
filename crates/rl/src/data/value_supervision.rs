//! Borrowed scalar supervision from complete actual episode returns.
use super::{returns::accumulate_rewards, temporal::SequenceSample};
use crate::{
    Result,
    core::{ensure, env::Observation, structured::StructuredObservation},
};

/// One selected factual learning row. The constructor accepts physical rewards
/// only; an MCTS value estimate cannot be supplied in place of the actual tail.
#[derive(Clone, Copy)]
pub struct ValueSupervision<'a> {
    sample: &'a SequenceSample,
    step: usize,
    gamma: f64,
    target: f32,
}

impl<'a> ValueSupervision<'a> {
    pub fn monte_carlo(sample: &'a SequenceSample, step: usize, gamma: f64) -> Result<Self> {
        sample.validate_learning_row(step)?;
        ensure(
            matches!(sample.observations[step], Observation::Structured(_)),
            "value supervision needs a structured decision",
        )?;
        ensure(
            sample.steps.last().is_some_and(|last| last.terminated)
                && sample.steps[..sample.steps.len() - 1]
                    .iter()
                    .all(|prior| !prior.ended()),
            "Monte Carlo supervision requires a complete true-terminal factual tail",
        )?;
        // Prefix rewards can enter recurrent inputs even though they do not
        // enter this row's return. Validate them before neural evaluation too.
        for prior in &sample.steps[..step] {
            prior.reward.validate()?;
        }
        let (target, _, _) =
            accumulate_rewards(sample.steps[step..].iter().map(|s| &s.reward), gamma)?;
        let target = target as f32;
        ensure(
            target.is_finite(),
            "Monte Carlo target exceeds neural f32 range",
        )?;
        Ok(Self {
            sample,
            step,
            gamma,
            target,
        })
    }

    pub fn sample(&self) -> &'a SequenceSample {
        self.sample
    }
    pub fn step(&self) -> usize {
        self.step
    }
    pub fn gamma(&self) -> f64 {
        self.gamma
    }
    pub fn target(&self) -> f32 {
        self.target
    }
    pub fn observation(&self) -> &'a StructuredObservation {
        let Observation::Structured(frame) = &self.sample.observations[self.step] else {
            unreachable!("validated structured value row")
        };
        frame
    }
}
