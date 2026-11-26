use super::*;
use crate::algorithms::reppo::LogMultiplier;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Temperature {
    Fixed(f32),
    /// Target entropy is in physical action units. None uses -dimension plus
    /// the affine log-Jacobian, equivalent to -dimension in normalized actions.
    Learned {
        initial: f32,
        target_entropy: Option<f32>,
        optimizer: OptimizerConfig,
    },
}
impl Default for Temperature {
    fn default() -> Self {
        Self::Learned {
            initial: 0.1,
            target_entropy: None,
            optimizer: OptimizerConfig {
                max_grad_norm: None,
                ..Default::default()
            },
        }
    }
}
impl Temperature {
    pub(super) fn validate(&self) -> Result<()> {
        match self {
            Self::Fixed(x) => ensure(x.is_finite() && *x >= 0., "invalid fixed SAC temperature"),
            Self::Learned {
                initial,
                target_entropy,
                optimizer,
            } => {
                optimizer.validate()?;
                ensure(
                    initial.is_finite()
                        && *initial > 0.
                        && (-18.0..=18.0).contains(&initial.ln())
                        && target_entropy.is_none_or(f32::is_finite),
                    "invalid learned SAC temperature",
                )
            }
        }
    }
}
pub(super) struct TemperatureState {
    pub model: LogMultiplier,
    pub optimizer: ModuleOptimizer,
    pub target: f32,
    pub learning_rate: f64,
}
impl TemperatureState {
    pub fn new(config: &Temperature, space: &ActionSpace) -> Result<Option<Self>> {
        let Temperature::Learned {
            initial,
            target_entropy,
            optimizer,
        } = config
        else {
            return Ok(None);
        };
        let log_scale: f64 = match space {
            ActionSpace::Box { low, high } => low
                .iter()
                .zip(high)
                .map(|(l, h)| (((h - l) * 0.5) as f64).ln())
                .sum(),
            _ => 0.,
        };
        let target = target_entropy.unwrap_or((-(space.dimensions() as f64) + log_scale) as f32);
        ensure(target.is_finite(), "unrepresentable SAC entropy target")?;
        Ok(Some(Self {
            model: LogMultiplier::new(*initial),
            optimizer: optimizer.build()?,
            target,
            learning_rate: optimizer.learning_rate,
        }))
    }
    pub fn update(&mut self, log_probs: Tensor<1>) -> Result<f64> {
        let loss = objective::temperature_loss(self.model.log_value.val(), log_probs, self.target)?;
        let (value, _) = optimize_validated(
            &mut self.model,
            &mut self.optimizer,
            self.learning_rate,
            loss,
        )?;
        parameter_norm(&self.model)?;
        self.model.project();
        self.model.validate()?;
        Ok(value)
    }
}
