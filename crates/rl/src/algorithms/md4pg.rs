use super::{
    common::OptimizerConfig,
    continuous::{
        ContinuousConfig, ContinuousLearner,
        critic::{
            CriticCheckpoint, CriticModel, CriticRule, PreparedReplayBatch, TargetActionBatch,
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
        critics::residual::ResidualQ,
        critics::scalar::{ActionValue, QValue},
        matrix,
        policy::{Deterministic, DeterministicPolicy},
        target::frozen_copy,
        vector,
    },
    objectives::{
        bellman::moment_targets,
        risk::{Risk, RiskObjective},
    },
};
use burn::{module::ModuleDisplay, tensor::Tensor};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Md4pgConfig<R = Risk> {
    pub training: ContinuousConfig,
    pub risk: R,
}
impl<P: DeterministicPolicy, Q: ActionValue + ModuleDisplay, R: RiskObjective>
    Md4pg<P, ResidualQ<Q>, R>
{
    /// Explicitly replace the frozen anchor across the two moment critic pairs.
    pub fn refresh_residual_anchor(&mut self, anchor: &Deterministic) -> Result<()> {
        let critic = &mut self.inner.critic;
        let mean = critic.mean.frozen_replacement(anchor)?;
        let second = critic.second.frozen_replacement(anchor)?;
        critic.mean.install_anchor(mean);
        critic.second.install_anchor(second);
        Ok(())
    }
}
impl Default for Md4pgConfig<Risk> {
    fn default() -> Self {
        Self {
            training: Default::default(),
            risk: Risk::Expected,
        }
    }
}
pub struct Md4pg<
    P: DeterministicPolicy = Deterministic,
    Q: ActionValue = QValue,
    R: RiskObjective = Risk,
> {
    inner: ContinuousLearner<P, MomentCritic<Q, R>>,
}
impl<P: DeterministicPolicy, Q: ActionValue, R: RiskObjective> Md4pg<P, Q, R> {
    pub fn new(policy: P, mean: Q, second: Q, config: Md4pgConfig<R>, seed: u64) -> Result<Self> {
        let critic = MomentCritic {
            mean: CriticModel::new(mean, &config.training.critic)?,
            second: CriticModel::new(second, &config.training.critic)?,
            risk: config.risk,
        };
        Ok(Self {
            inner: ContinuousLearner::new(policy, critic, config.training, seed)?,
        })
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn training_config(&self) -> &ContinuousConfig {
        self.inner.training_config()
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn risk(&self) -> &R {
        &self.inner.critic.risk
    }

    pub fn critic(&self) -> Q {
        frozen_copy(&self.inner.critic.mean.online)
    }
    pub fn second_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.second.online)
    }
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.mean.target)
    }
    pub fn target_second_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.second.target)
    }
}
pub(crate) struct MomentCritic<Q: ActionValue, R: RiskObjective> {
    mean: CriticModel<Q>,
    second: CriticModel<Q>,
    risk: R,
}
impl<Q: ActionValue, R: RiskObjective> CriticRule for MomentCritic<Q, R> {
    type Targets = (Tensor<2>, Tensor<2>);
    fn action_dim(&self) -> usize {
        self.mean.online.action_dim()
    }
    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets> {
        let mean = self.mean.future(batch, next)?;
        let second = self.second.future(batch, next)?;
        let rewards: Vec<_> = batch
            .items
            .iter()
            .map(|item| item.transition.reward.total())
            .collect();
        let discounts: Vec<_> = batch.items.iter().map(|item| item.discount).collect();
        let (mean, second) = moment_targets(&rewards, &discounts, &mean, &second)?;
        Ok((
            matrix(mean, batch.items.len(), 1),
            matrix(second, batch.items.len(), 1),
        ))
    }
    fn update(
        &mut self,
        batch: &PreparedReplayBatch<'_>,
        (mean, second): Self::Targets,
    ) -> Result<f64> {
        let pred = self
            .mean
            .online
            .values(batch.observations(), batch.actions.clone())?;
        let loss = self.mean.fit((pred - mean).powf_scalar(2.).mean())?;
        let pred = self
            .second
            .online
            .values(batch.observations(), batch.actions.clone())?;
        Ok(loss + self.second.fit((pred - second).powf_scalar(2.).mean())?)
    }
    fn actor_utility(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<1>> {
        let mean = self
            .mean
            .online
            .valid()
            .freeze()
            .values(obs, actions.clone())?
            .squeeze_dim(1);
        if self.second.online.actor_second_forward_skippable()
            && let Some(utility) = self.risk.actor_utility_from_mean(&mean)
        {
            return utility;
        }
        let second = self
            .second
            .online
            .valid()
            .freeze()
            .values(obs, actions)?
            .squeeze_dim(1);
        self.risk.moments(mean, second)
    }
    fn update_targets(&mut self, tau: f32) -> Result<()> {
        self.mean.update_target(tau)?;
        self.second.update_target(tau)
    }
    fn validate(&self) -> Result<()> {
        self.mean.validate(1)?;
        self.second.validate(1)?;
        ensure(
            self.mean.online.action_dim() == self.second.online.action_dim(),
            "moment critic action dimensions differ",
        )?;
        let _ = self.risk.moments(vector(&[0.]), vector(&[0.]))?;
        Ok(())
    }
}
impl<Q: ActionValue, R: RiskObjective + Serialize + DeserializeOwned> CriticCheckpoint
    for MomentCritic<Q, R>
{
    type Metadata = R;
    fn metadata(&self) -> R {
        self.risk.clone()
    }
    fn record(&self, record: &mut LearnerRecord) -> Result<()> {
        self.mean.record(record, "critic", "target_critic")?;
        self.second.record(record, "second", "target_second")
    }
    fn restored(&self, record: &LearnerRecord, risk: R, config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            mean: self
                .mean
                .restored(record, "critic", "target_critic", config)?,
            second: self
                .second
                .restored(record, "second", "target_second", config)?,
            risk,
        })
    }
}
impl<P: DeterministicPolicy, Q: ActionValue, R: RiskObjective + Serialize + DeserializeOwned>
    Checkpoint for Md4pg<P, Q, R>
{
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.inner.checkpoint("md4pg")
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.inner.restore(record, "md4pg")
    }
}
delegate_learner!(Md4pg, R);

#[cfg(test)]
mod actor_cost_tests {
    use super::*;
    use burn::module::Module;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[derive(Module, Debug)]
    struct CountedQ {
        #[module(skip)]
        forwards: Arc<AtomicUsize>,
        #[module(skip)]
        safe_to_skip: bool,
        #[module(skip)]
        reject: bool,
    }
    impl ActionValue for CountedQ {
        fn action_dim(&self) -> usize {
            1
        }
        fn output_dim(&self) -> usize {
            1
        }
        fn actor_second_forward_skippable(&self) -> bool {
            self.safe_to_skip
        }
        fn values(&self, _: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
            self.forwards.fetch_add(1, Ordering::Relaxed);
            if self.reject {
                return Err(crate::Error::State("custom second critic called".into()));
            }
            Ok(actions)
        }
    }
    #[derive(Debug, Clone)]
    struct CustomSum;
    impl RiskObjective for CustomSum {
        fn quantiles(
            &self,
            _: Tensor<2>,
            _: &crate::nn::critics::quantile::QuantileGrid,
        ) -> Result<Tensor<1>> {
            unreachable!()
        }
        fn moments(&self, mean: Tensor<1>, second: Tensor<1>) -> Result<Tensor<1>> {
            Ok(mean + second)
        }
    }
    fn critic<R: RiskObjective>(
        risk: R,
        second_skippable: bool,
        second_rejects: bool,
    ) -> (
        MomentCritic<CountedQ, R>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let mean_count = Arc::new(AtomicUsize::new(0));
        let second_count = Arc::new(AtomicUsize::new(0));
        let config = OptimizerConfig::default();
        (
            MomentCritic {
                mean: CriticModel::new(
                    CountedQ {
                        forwards: mean_count.clone(),
                        safe_to_skip: false,
                        reject: false,
                    },
                    &config,
                )
                .unwrap(),
                second: CriticModel::new(
                    CountedQ {
                        forwards: second_count.clone(),
                        safe_to_skip: second_skippable,
                        reject: second_rejects,
                    },
                    &config,
                )
                .unwrap(),
                risk,
            },
            mean_count,
            second_count,
        )
    }
    #[test]
    fn expected_actor_skips_second_forward_and_keeps_action_gradient() {
        let (critic, mean, second) = critic(Risk::Expected, true, false);
        let actions = matrix(vec![2., 3.], 2, 1).require_grad();
        let obs = vec![crate::core::env::Observation::new(vec![0.]); 2];
        let result = critic
            .actor_utility((&obs).into(), actions.clone())
            .unwrap();
        assert_eq!(crate::nn::to_vec(result.clone()).unwrap(), [2., 3.]);
        let gradients = result.sum().backward();
        assert_eq!(
            crate::nn::to_vec(actions.grad(&gradients).unwrap()).unwrap(),
            [1., 1.]
        );
        assert_eq!(mean.load(Ordering::Relaxed), 1);
        assert_eq!(second.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn custom_risk_default_still_forwards_both_critics() {
        let (critic, mean, second) = critic(CustomSum, true, false);
        let obs = vec![crate::core::env::Observation::new(vec![0.]); 1];
        let result = critic
            .actor_utility((&obs).into(), matrix(vec![2.], 1, 1))
            .unwrap();
        assert_eq!(crate::nn::to_vec(result).unwrap(), [4.]);
        assert_eq!(mean.load(Ordering::Relaxed), 1);
        assert_eq!(second.load(Ordering::Relaxed), 1);
    }
    #[test]
    fn expected_risk_preserves_non_opted_custom_second_critic_error() {
        let (critic, mean, second) = critic(Risk::Expected, false, true);
        let obs = vec![crate::core::env::Observation::new(vec![0.]); 1];
        let error = critic
            .actor_utility((&obs).into(), matrix(vec![2.], 1, 1))
            .unwrap_err();
        assert!(error.to_string().contains("custom second critic called"));
        assert_eq!(mean.load(Ordering::Relaxed), 1);
        assert_eq!(second.load(Ordering::Relaxed), 1);
    }
}
