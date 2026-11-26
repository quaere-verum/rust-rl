use crate::{Result, core::ensure, train::counters::RunCounters};
use serde::{Deserialize, Serialize};

/// Credit only the transitions collected after warmup.
pub(super) fn collection_credit(before: u64, after: u64, warmup: u64, rate: f64) -> f64 {
    after.saturating_sub(before.max(warmup)) as f64 * rate
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Counter {
    EnvironmentTransitions,
    PhysicalSteps,
    VectorSteps,
    CompletedEpisodes,
    LearnerUpdates,
}
impl Counter {
    pub fn read(self, c: &RunCounters) -> u64 {
        match self {
            Self::EnvironmentTransitions => c.env_steps,
            Self::PhysicalSteps => c.physical_steps(),
            Self::VectorSteps => c.vector_steps,
            Self::CompletedEpisodes => c.episodes,
            Self::LearnerUpdates => c.updates,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Schedule {
    Constant(f64),
    Linear {
        counter: Counter,
        start: f64,
        end: f64,
        duration: u64,
    },
}
impl Schedule {
    #[cfg(feature = "neural")]
    pub(super) fn probability(&self, counters: &RunCounters, error: &str) -> Result<f64> {
        let value = self.value(counters)?;
        ensure((0.0..=1.0).contains(&value), error)?;
        Ok(value)
    }

    pub fn value(&self, c: &RunCounters) -> Result<f64> {
        match *self {
            Self::Constant(x) => {
                ensure(x.is_finite(), "schedule value must be finite")?;
                Ok(x)
            }
            Self::Linear {
                counter,
                start,
                end,
                duration,
            } => {
                ensure(
                    start.is_finite() && end.is_finite() && duration > 0,
                    "invalid linear schedule",
                )?;
                Ok(start + (end - start) * (counter.read(c) as f64 / duration as f64).min(1.))
            }
        }
    }
}
