//! Optional diagnostics independent of presentation and the collection hot path.
//!
//! Numerical payloads use binary arrays; JSON contains metadata and references.
//! Recording is explicit and does not install global hooks or random streams.
pub mod activation_inputs;
#[cfg(feature = "neural")]
mod adapters;
pub mod analysis;
mod codec;
#[cfg(feature = "neural")]
pub mod compact;
pub mod empirical;
pub mod evaluation;
pub mod models;
pub mod network_geometry;
pub mod observer;
pub mod policy;
pub mod policy_batch;
pub mod policy_cache;
pub mod policy_numerics;
#[cfg(feature = "neural")]
pub mod policy_precompute;
pub mod policy_slice;
pub mod policy_surface;
pub mod probes;
pub mod reference;
pub mod render;
pub mod representations;
pub mod schedule;
pub mod schema;
pub mod session;
#[path = "adapters/statistical.rs"]
pub mod statistical;
pub mod storage;
pub mod tabular;
pub mod typed;
mod validation;
pub mod value;
pub mod visualization;
