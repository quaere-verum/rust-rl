//! Bounded recorded learning evolution. No model restoration or training occurs here.
#[path = "learning/policy_space.rs"]
mod policy_space;
pub use policy_space::{PolicySpacePoint, PolicySpaceProjection};
#[path = "learning/math.rs"]
mod math;
pub use math::*;
#[path = "learning/history.rs"]
mod history;
pub use history::*;
#[path = "learning/training.rs"]
mod training;
pub use training::*;
