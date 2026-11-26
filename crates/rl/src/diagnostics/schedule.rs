//! Recording schedules refer to completed parameter revisions, not minibatches
//! which have not been observed. Crossing several intervals produces one sample.
use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub first: u64,
    pub every: u64,
}
impl Interval {
    pub fn validate(&self) -> Result<()> {
        ensure(self.every > 0, "recording interval must be positive")
    }
    pub fn due(&self, previous: Option<u64>, current: u64) -> Result<bool> {
        self.validate()?;
        ensure(
            previous.is_none_or(|p| p <= current),
            "recording revision moved backwards",
        )?;
        if current < self.first {
            return Ok(false);
        }
        Ok(previous.is_none_or(|p| {
            p < self.first || (p - self.first) / self.every < (current - self.first) / self.every
        }))
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingSchedule {
    pub metrics: Option<Interval>,
    pub probes: Option<Interval>,
    pub monte_carlo: Option<Interval>,
    pub snapshots: Option<Interval>,
}
impl Default for RecordingSchedule {
    fn default() -> Self {
        Self {
            metrics: Some(Interval { first: 1, every: 1 }),
            probes: Some(Interval {
                first: 20,
                every: 20,
            }),
            monte_carlo: Some(Interval {
                first: 100,
                every: 100,
            }),
            snapshots: Some(Interval {
                first: 200,
                every: 200,
            }),
        }
    }
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Due {
    pub metrics: bool,
    pub probes: bool,
    pub monte_carlo: bool,
    pub snapshots: bool,
}
impl RecordingSchedule {
    pub fn due(&self, previous: Option<u64>, current: u64) -> Result<Due> {
        let due = |s: &Option<Interval>| {
            s.map(|s| s.due(previous, current))
                .transpose()
                .map(|v| v.unwrap_or(false))
        };
        Ok(Due {
            metrics: due(&self.metrics)?,
            probes: due(&self.probes)?,
            monte_carlo: due(&self.monte_carlo)?,
            snapshots: due(&self.snapshots)?,
        })
    }
}
