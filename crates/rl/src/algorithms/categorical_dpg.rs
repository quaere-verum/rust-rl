//! Deterministic policy gradients with a fixed-support categorical Bellman critic.
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
        critics::scalar::{ActionValue, QValue},
        matrix,
        policy::{Deterministic, DeterministicPolicy},
        target::frozen_copy,
        to_vec,
    },
    objectives::{
        categorical::{CategoricalSupport, validate_risk},
        risk::Risk,
    },
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalDpgConfig {
    pub training: ContinuousConfig,
    pub support: CategoricalSupport,
    pub risk: Risk,
}

impl CategoricalDpgConfig {
    pub fn new(support: CategoricalSupport) -> Self {
        Self {
            training: Default::default(),
            support,
            risk: Risk::Expected,
        }
    }
}

/// The critic emits one logit per support atom. Its targets are return
/// distributions under the target policy and declared target-action smoothing.
pub struct CategoricalDpg<P: DeterministicPolicy = Deterministic, Q: ActionValue = QValue> {
    inner: ContinuousLearner<P, CategoricalCritic<Q>>,
}

impl<P: DeterministicPolicy, Q: ActionValue> CategoricalDpg<P, Q> {
    pub fn new(policy: P, critic: Q, config: CategoricalDpgConfig, seed: u64) -> Result<Self> {
        let rule = CategoricalCritic {
            model: CriticModel::new(critic, &config.training.critic)?,
            support: config.support,
            risk: config.risk,
        };
        Ok(Self {
            inner: ContinuousLearner::new(policy, rule, config.training, seed)?,
        })
    }

    /// Independent frozen online critic. Its raw outputs are logits.
    pub fn critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.online)
    }

    /// Independent frozen target critic. Its raw outputs are logits.
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.target)
    }

    pub fn support(&self) -> &CategoricalSupport {
        &self.inner.critic.support
    }

    pub fn risk(&self) -> &Risk {
        &self.inner.critic.risk
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn training_config(&self) -> &ContinuousConfig {
        self.inner.training_config()
    }

    /// Predict the online return probabilities for complete environment actions.
    pub fn probabilities(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        self.support()
            .probabilities(self.critic().values(observations, actions)?)
    }

    /// Evaluate the configured actor objective with the critic fixed.
    pub fn action_utility(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<1>> {
        self.inner.critic.actor_utility(observations, actions)
    }
}

struct CategoricalCritic<Q: ActionValue> {
    model: CriticModel<Q>,
    support: CategoricalSupport,
    risk: Risk,
}

impl<Q: ActionValue> CriticRule for CategoricalCritic<Q> {
    type Targets = Tensor<2>;

    fn action_dim(&self) -> usize {
        self.model.online.action_dim()
    }

    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets> {
        let width = self.support.atoms().len();
        let future: Vec<_> = if let Some(actions) = &next.actions {
            let logits = self
                .model
                .target
                .valid()
                .values(next.observations(), actions.clone())?;
            to_vec(self.support.probabilities(logits)?)?
                .into_iter()
                .map(f64::from)
                .collect()
        } else {
            Vec::new()
        };
        ensure(
            future.len() == next.indices.len() * width,
            "misaligned categorical continuation rows",
        )?;
        let mut target = Vec::with_capacity(batch.items.len() * width);
        let mut continuations = future.chunks_exact(width);
        for item in batch.items {
            let probabilities = if item.discount > 0. {
                continuations.next().ok_or_else(|| {
                    crate::Error::Shape("missing categorical continuation row".into())
                })?
            } else {
                &[]
            };
            let projected = self.support.project(
                f64::from(item.transition.reward.total()),
                f64::from(item.discount),
                probabilities,
            )?;
            target.extend(projected.probabilities.into_iter().map(|mass| mass as f32));
        }
        Ok(matrix(target, batch.items.len(), width))
    }

    fn update(&mut self, batch: &PreparedReplayBatch<'_>, targets: Self::Targets) -> Result<f64> {
        let logits = self
            .model
            .online
            .values(batch.observations(), batch.actions.clone())?;
        self.model
            .fit_validated(self.support.cross_entropy(logits, targets)?)
    }

    fn actor_utility(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<1>> {
        let logits = self
            .model
            .online
            .valid()
            .freeze()
            .values(observations, actions)?;
        self.support.actor_utility(logits, &self.risk)
    }

    fn update_targets(&mut self, tau: f32) -> Result<()> {
        self.model.update_target(tau)
    }

    fn validate(&self) -> Result<()> {
        self.model.validate(self.support.atoms().len())?;
        validate_risk(&self.risk)
    }
}

#[derive(Serialize, Deserialize)]
struct CategoricalMetadata {
    support: CategoricalSupport,
    risk: Risk,
}

impl<Q: ActionValue> CriticCheckpoint for CategoricalCritic<Q> {
    type Metadata = CategoricalMetadata;

    fn metadata(&self) -> Self::Metadata {
        CategoricalMetadata {
            support: self.support.clone(),
            risk: self.risk.clone(),
        }
    }

    fn record(&self, record: &mut LearnerRecord) -> Result<()> {
        self.model.record(record, "critic", "target_critic")
    }

    fn restored(
        &self,
        record: &LearnerRecord,
        metadata: Self::Metadata,
        config: &OptimizerConfig,
    ) -> Result<Self> {
        ensure(
            metadata.support == self.support,
            "checkpoint categorical support mismatch",
        )?;
        validate_risk(&metadata.risk)?;
        Ok(Self {
            model: self
                .model
                .restored(record, "critic", "target_critic", config)?,
            support: metadata.support,
            risk: metadata.risk,
        })
    }
}

impl<P: DeterministicPolicy, Q: ActionValue> Checkpoint for CategoricalDpg<P, Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.inner.checkpoint("categorical_dpg")
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.inner.restore(record, "categorical_dpg")
    }
}

delegate_learner!(CategoricalDpg);
