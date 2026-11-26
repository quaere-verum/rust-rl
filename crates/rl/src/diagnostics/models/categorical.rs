//! Portable layout for REPPO's linear/LayerNorm/SiLU encoder and predictor.
use crate::{
    Error, Result,
    core::{action::ActionSpace, ensure},
    objectives::hl_gauss::HlGaussConfig,
};
use serde::{Deserialize, Serialize};

pub(crate) const FEATURES: &str = "encoder.features";
pub(crate) const NEXT_FEATURES: &str = "predictor.next_features";

/// The encoder has the declared block widths. Each block has a biased linear
/// layer followed by affine LayerNorm and SiLU. The head is one biased linear
/// layer. The predictor is one normalized block at the final encoder width.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalQLayout {
    pub encoder_widths: Vec<usize>,
    /// Whether the learner trained the auxiliary predictor.
    pub auxiliary: bool,
}

impl CategoricalQLayout {
    fn dimensions(&self, observation_dim: usize, space: &ActionSpace) -> Result<(usize, usize)> {
        ensure(
            observation_dim > 0
                && (1..=64).contains(&self.encoder_widths.len())
                && self.encoder_widths.iter().all(|&width| width >= 2)
                && !matches!(space, ActionSpace::Conditional(_)),
            "invalid normalized categorical critic layout",
        )?;
        let action_width = match space {
            ActionSpace::Discrete { count } => *count,
            _ => space.dimensions(),
        };
        let input = observation_dim
            .checked_add(action_width)
            .ok_or_else(size_overflow)?;
        Ok((input, *self.encoder_widths.last().unwrap()))
    }

    pub(super) fn parameters(
        &self,
        observation_dim: usize,
        space: &ActionSpace,
        support: &HlGaussConfig,
    ) -> Result<usize> {
        support.validate()?;
        let (input, features) = self.dimensions(observation_dim, space)?;
        let block = |input: usize, output: usize| {
            input
                .checked_add(3)
                .and_then(|width| width.checked_mul(output))
                .ok_or_else(size_overflow)
        };
        let mut parameters = block(features, features)?;
        let mut input_width = input;
        for &width in &self.encoder_widths {
            parameters = parameters
                .checked_add(block(input_width, width)?)
                .ok_or_else(size_overflow)?;
            input_width = width;
        }
        features
            .checked_add(1)
            .and_then(|width| width.checked_mul(support.bins))
            .and_then(|head| head.checked_add(parameters))
            .ok_or_else(size_overflow)
    }

    #[cfg(feature = "neural")]
    pub(super) fn work_width(
        &self,
        observation_dim: usize,
        space: &ActionSpace,
        support: &HlGaussConfig,
    ) -> Result<usize> {
        let (input, _) = self.dimensions(observation_dim, space)?;
        Ok(self
            .encoder_widths
            .iter()
            .copied()
            .fold(input.max(support.bins), usize::max))
    }

    /// Actual representation output names and widths. These outputs do not
    /// admit the temporary activation interventions used for MLPs.
    pub fn activation_layers(&self) -> Vec<(&'static str, usize)> {
        let Some(&width) = self.encoder_widths.last() else {
            return Vec::new();
        };
        let mut layers = vec![(FEATURES, width)];
        if self.auxiliary {
            layers.push((NEXT_FEATURES, width));
        }
        layers
    }
}

fn size_overflow() -> Error {
    Error::Shape("categorical critic layout exceeds size admission".into())
}
