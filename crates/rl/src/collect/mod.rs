//! Specialized collectors sharing only private environment lane mechanics.
mod driver;
pub mod off_policy;
pub mod on_policy;
pub mod stateful;
mod stateful_recording;
pub mod temporal_on_policy;
pub mod vector_env;
pub use off_policy::OffPolicyCollector;
pub use on_policy::OnPolicyCollector;
use serde::{Deserialize, Serialize};
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionDelta {
    pub env_steps: u64,
    pub vector_steps: u64,
    pub episodes: u64,
    #[serde(default, skip_serializing_if = "zero_steps")]
    pub additional_physical_steps: u64,
}
impl CollectionDelta {
    pub fn physical_steps(self) -> u64 {
        self.env_steps + self.additional_physical_steps
    }

    pub(crate) fn add(&mut self, other: Self) {
        self.env_steps += other.env_steps;
        self.vector_steps += other.vector_steps;
        self.episodes += other.episodes;
        self.additional_physical_steps += other.additional_physical_steps;
    }
}

pub(crate) fn zero_steps(steps: &u64) -> bool {
    *steps == 0
}
