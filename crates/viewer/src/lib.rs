//! Native laboratory presentation. Evaluation and model logic live in `rl`.
#![forbid(unsafe_code)]
pub mod budgets;
pub mod comparison;
pub mod data;
pub mod dynamics;
pub mod environment;
pub mod interactive;
pub mod landscape;
pub mod learning;
#[cfg(feature = "screenshots")]
pub(crate) mod native_timing;
pub mod network;
#[cfg(test)]
#[path = "../tests/support/network.rs"]
pub(crate) mod network_test_fixture;
#[cfg(feature = "screenshots")]
pub(crate) mod owner_activity;
pub mod pane;
pub mod plot_data;
pub mod policy;
pub mod policy_change;
pub mod presentation;
mod reference;
pub mod resolver;
pub mod snapshot_catalog;
pub mod theme;
pub mod timeline;
pub mod timeline_ui;
pub mod ui;
pub mod value;
pub mod work;
