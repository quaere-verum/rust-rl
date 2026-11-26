use crate::collect::CollectionDelta;
use serde::{Deserialize, Serialize};
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunCounters {
    pub env_steps: u64,
    pub vector_steps: u64,
    pub episodes: u64,
    pub updates: u64,
    /// Primitive steps beyond the first in each decision. Old one-step
    /// snapshots omit this field and therefore keep their exact clocks.
    #[serde(default, skip_serializing_if = "crate::collect::zero_steps")]
    pub additional_physical_steps: u64,
}
impl RunCounters {
    pub fn physical_steps(&self) -> u64 {
        self.env_steps
            .saturating_add(self.additional_physical_steps)
    }

    pub(crate) fn checked_physical_steps(&self) -> Option<u64> {
        self.env_steps.checked_add(self.additional_physical_steps)
    }

    pub(crate) fn add_collection(&mut self, d: CollectionDelta) {
        self.env_steps += d.env_steps;
        self.vector_steps += d.vector_steps;
        self.episodes += d.episodes;
        self.additional_physical_steps += d.additional_physical_steps;
    }
}
