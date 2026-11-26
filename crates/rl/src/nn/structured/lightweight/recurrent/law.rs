use crate::{
    Error, Result,
    core::{action::ConditionalAction, ensure},
};
use burn::tensor::{Int, Tensor, TensorData};
use std::collections::{HashMap, HashSet};

/// Differentiable joint log probabilities for the enabled complete actions.
#[derive(Debug, Clone)]
pub struct ConditionalPolicyLaw {
    keys: Vec<ConditionalAction>,
    log_probs: Tensor<1>,
}

impl ConditionalPolicyLaw {
    /// Binds normalized log probabilities to their semantic keys.
    pub fn new(keys: Vec<ConditionalAction>, log_probs: Tensor<1>) -> Result<Self> {
        ensure(
            !keys.is_empty()
                && keys.len() <= i32::MAX as usize
                && log_probs.dims() == [keys.len()]
                && keys.iter().collect::<HashSet<_>>().len() == keys.len(),
            "invalid conditional policy support or tensor shape",
        )?;
        Ok(Self { keys, log_probs })
    }

    pub fn keys(&self) -> &[ConditionalAction] {
        &self.keys
    }

    pub fn log_probs(&self) -> &Tensor<1> {
        &self.log_probs
    }

    pub fn entropy(&self) -> Tensor<1> {
        (self.log_probs.clone().exp() * self.log_probs.clone())
            .sum()
            .neg()
    }

    /// Computes KL(self || other) on equal semantic support.
    pub fn kl(&self, other: &Self) -> Result<Tensor<1>> {
        ensure(
            self.keys.len() == other.keys.len(),
            "conditional policy KL support differs",
        )?;
        let other_logs = if self.keys == other.keys {
            other.log_probs.clone()
        } else {
            let positions: HashMap<_, _> = other
                .keys
                .iter()
                .enumerate()
                .map(|(row, key)| (key, row as i32))
                .collect();
            let order = self
                .keys
                .iter()
                .map(|key| {
                    positions.get(key).copied().ok_or_else(|| {
                        Error::Action("conditional policy KL support differs".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let indices = Tensor::<1, Int>::from_data(
                TensorData::new(order, [self.keys.len()]),
                &other.log_probs.device(),
            );
            other.log_probs.clone().select(0, indices)
        };
        Ok((self.log_probs.clone().exp() * (self.log_probs.clone() - other_logs)).sum())
    }
}
