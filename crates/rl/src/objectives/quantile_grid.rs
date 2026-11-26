//! Quantile locations and probability masses, independent of a predictor.
use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileGrid {
    pub edges: Vec<f32>,
}
impl QuantileGrid {
    pub fn uniform(count: usize) -> Result<Self> {
        ensure(count > 0, "quantile count must be positive")?;
        Ok(Self {
            edges: (0..=count).map(|i| i as f32 / count as f32).collect(),
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.edges.len() >= 2
                && self.edges[0] == 0.
                && *self.edges.last().unwrap() == 1.
                && self
                    .edges
                    .windows(2)
                    .all(|w| w[0].is_finite() && w[0] < w[1]),
            "quantile edges must increase from zero to one",
        )
    }
    pub fn locations(&self) -> Vec<f32> {
        self.edges.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect()
    }
    pub fn masses(&self) -> Vec<f32> {
        self.edges.windows(2).map(|w| w[1] - w[0]).collect()
    }
    pub fn len(&self) -> usize {
        self.edges.len().saturating_sub(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
