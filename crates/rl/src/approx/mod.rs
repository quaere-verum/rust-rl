//! Statistical function approximation and finite-action inference.

mod actor;
pub mod batch;
pub mod diagnostics;
mod features;
pub mod regression;
mod retrieval;

pub use actor::{FittedQ, StatisticalActor};
pub use features::{
    FeatureMap, FeatureSpec, FixedLocal, GateAssignment, GateConfig, GateDescriptor, Intercept,
    LocalLayout,
};

pub use retrieval::ReturnWeightedRetrieval;

pub(crate) use actor::greedy_index;
pub(crate) use features::checked_features;
