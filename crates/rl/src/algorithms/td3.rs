use super::{
    common::OptimizerConfig,
    continuous::{
        ContinuousConfig, ContinuousLearner,
        critic::{
            CriticCheckpoint, CriticModel, CriticRule, PreparedReplayBatch, TargetActionBatch,
            bellman,
        },
        delegate_learner,
    },
};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationBatch,
    },
    nn::{
        critics::{
            residual::ResidualQ,
            scalar::{ActionValue, QValue},
        },
        policy::{Deterministic, DeterministicPolicy},
        target::frozen_copy,
    },
};
use burn::{module::ModuleDisplay, tensor::Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Td3Config {
    pub training: ContinuousConfig,
}
impl Default for Td3Config {
    fn default() -> Self {
        Self {
            training: ContinuousConfig {
                policy_delay: 2,
                target_noise: 0.2,
                ..Default::default()
            },
        }
    }
}
pub struct Td3<P: DeterministicPolicy = Deterministic, Q: ActionValue = QValue> {
    inner: ContinuousLearner<P, TwinCritic<Q>>,
}
impl<P: DeterministicPolicy, Q: ActionValue> Td3<P, Q> {
    pub fn new(policy: P, first: Q, second: Q, config: Td3Config, seed: u64) -> Result<Self> {
        let critic = TwinCritic {
            first: CriticModel::new(first, &config.training.critic)?,
            second: CriticModel::new(second, &config.training.critic)?,
        };
        Ok(Self {
            inner: ContinuousLearner::new(policy, critic, config.training, seed)?,
        })
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn training_config(&self) -> &ContinuousConfig {
        self.inner.training_config()
    }
    pub fn critic(&self) -> Q {
        frozen_copy(&self.inner.critic.first.online)
    }
    pub fn second_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.second.online)
    }
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.first.target)
    }
    pub fn target_second_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.second.target)
    }
}
impl<P: DeterministicPolicy, Q: ActionValue + ModuleDisplay> Td3<P, ResidualQ<Q>> {
    /// Explicitly replace the frozen anchor across both online/target pairs.
    pub fn refresh_residual_anchor(&mut self, anchor: &Deterministic) -> Result<()> {
        let critic = &mut self.inner.critic;
        let first = critic.first.frozen_replacement(anchor)?;
        let second = critic.second.frozen_replacement(anchor)?;
        critic.first.install_anchor(first);
        critic.second.install_anchor(second);
        Ok(())
    }
}
pub(crate) struct TwinCritic<Q: ActionValue> {
    first: CriticModel<Q>,
    second: CriticModel<Q>,
}
impl<Q: ActionValue> CriticRule for TwinCritic<Q> {
    type Targets = Tensor<2>;
    fn action_dim(&self) -> usize {
        self.first.online.action_dim()
    }
    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets> {
        let mut values = self.first.future(batch, next)?;
        let second = self.second.future(batch, next)?;
        for (value, other) in values.iter_mut().zip(second) {
            *value = value.min(other);
        }
        Ok(bellman(batch, values, 1))
    }
    fn update(&mut self, batch: &PreparedReplayBatch<'_>, target: Self::Targets) -> Result<f64> {
        let first = self
            .first
            .online
            .values(batch.observations(), batch.actions.clone())?;
        let loss = self
            .first
            .fit((first - target.clone()).powf_scalar(2.).mean())?;
        let second = self
            .second
            .online
            .values(batch.observations(), batch.actions.clone())?;
        Ok(loss + self.second.fit((second - target).powf_scalar(2.).mean())?)
    }
    fn actor_utility(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<1>> {
        Ok(self
            .first
            .online
            .valid()
            .freeze()
            .values(obs, actions)?
            .squeeze_dim(1))
    }
    fn update_targets(&mut self, tau: f32) -> Result<()> {
        self.first.update_target(tau)?;
        self.second.update_target(tau)
    }
    fn validate(&self) -> Result<()> {
        self.first.validate(1)?;
        self.second.validate(1)?;
        ensure(
            self.first.online.action_dim() == self.second.online.action_dim(),
            "twin critic action dimensions differ",
        )
    }
}
impl<Q: ActionValue> CriticCheckpoint for TwinCritic<Q> {
    type Metadata = ();
    fn metadata(&self) {}
    fn record(&self, record: &mut LearnerRecord) -> Result<()> {
        self.first.record(record, "critic", "target_critic")?;
        self.second.record(record, "second", "target_second")
    }
    fn restored(&self, record: &LearnerRecord, _: (), config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            first: self
                .first
                .restored(record, "critic", "target_critic", config)?,
            second: self
                .second
                .restored(record, "second", "target_second", config)?,
        })
    }
}
impl<P: DeterministicPolicy, Q: ActionValue> Checkpoint for Td3<P, Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.inner.checkpoint("td3")
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.inner.restore(record, "td3")
    }
}
delegate_learner!(Td3);
