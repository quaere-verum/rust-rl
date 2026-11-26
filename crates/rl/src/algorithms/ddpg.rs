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

pub type DdpgConfig = ContinuousConfig;
pub struct Ddpg<P: DeterministicPolicy = Deterministic, Q: ActionValue = QValue> {
    inner: ContinuousLearner<P, ScalarCritic<Q>>,
}
impl<P: DeterministicPolicy, Q: ActionValue> Ddpg<P, Q> {
    pub fn new(policy: P, critic: Q, mut config: DdpgConfig, seed: u64) -> Result<Self> {
        config.policy_delay = 1;
        config.target_noise = 0.;
        let critic = ScalarCritic {
            model: CriticModel::new(critic, &config.critic)?,
        };
        Ok(Self {
            inner: ContinuousLearner::new(policy, critic, config, seed)?,
        })
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn training_config(&self) -> &ContinuousConfig {
        self.inner.training_config()
    }
    pub fn critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.online)
    }
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.target)
    }
}
impl<P: DeterministicPolicy, Q: ActionValue + ModuleDisplay> Ddpg<P, ResidualQ<Q>> {
    /// Explicitly replace the online and target critics' shared frozen anchor.
    pub fn refresh_residual_anchor(&mut self, anchor: &Deterministic) -> Result<()> {
        let model = &mut self.inner.critic.model;
        let anchor = model.frozen_replacement(anchor)?;
        model.install_anchor(anchor);
        Ok(())
    }
}
pub(crate) struct ScalarCritic<Q: ActionValue> {
    pub(super) model: CriticModel<Q>,
}
impl<Q: ActionValue> CriticRule for ScalarCritic<Q> {
    type Targets = Tensor<2>;
    fn action_dim(&self) -> usize {
        self.model.online.action_dim()
    }
    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets> {
        Ok(bellman(batch, self.model.future(batch, next)?, 1))
    }
    fn update(&mut self, batch: &PreparedReplayBatch<'_>, target: Self::Targets) -> Result<f64> {
        let prediction = self
            .model
            .online
            .values(batch.observations(), batch.actions.clone())?;
        self.model.fit((prediction - target).powf_scalar(2.).mean())
    }
    fn actor_utility(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<1>> {
        Ok(self
            .model
            .online
            .valid()
            .freeze()
            .values(obs, actions)?
            .squeeze_dim(1))
    }
    fn update_targets(&mut self, tau: f32) -> Result<()> {
        self.model.update_target(tau)
    }
    fn validate(&self) -> Result<()> {
        self.model.validate(1)
    }
    fn validate_training(&self, config: &ContinuousConfig) -> Result<()> {
        ensure(
            config.policy_delay == 1 && config.target_noise == 0.,
            "DDPG requires every-step actor updates without target smoothing",
        )
    }
}
impl<Q: ActionValue> CriticCheckpoint for ScalarCritic<Q> {
    type Metadata = ();
    fn metadata(&self) {}
    fn record(&self, record: &mut LearnerRecord) -> Result<()> {
        self.model.record(record, "critic", "target_critic")
    }
    fn restored(&self, record: &LearnerRecord, _: (), config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            model: self
                .model
                .restored(record, "critic", "target_critic", config)?,
        })
    }
}
impl<P: DeterministicPolicy, Q: ActionValue> Checkpoint for Ddpg<P, Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.inner.checkpoint("ddpg")
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.inner.restore(record, "ddpg")
    }
}
delegate_learner!(Ddpg);
