pub mod categorical;
pub mod moments;
pub mod quantile;
pub mod residual;
pub mod scalar;

use crate::{Result, nn::network::Mlp};
use serde::{Deserialize, Serialize};

/// Explicit initialization for scalar critics and all-action scalar heads.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CriticInitialization {
    /// Keep the model's seeded constructor initialization.
    #[default]
    Existing,
    /// Zero the final weights and set every output bias to this value.
    /// Choose zero for neutral initialization or a problem-specific optimistic value.
    Constant { value: f32 },
}

impl CriticInitialization {
    pub(crate) fn apply(self, head: &mut Mlp) -> Result<()> {
        match self {
            Self::Existing => Ok(()),
            Self::Constant { value } => head.set_constant_output(value),
        }
    }
}
