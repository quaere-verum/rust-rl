//! Independent policy/value fitting to complete action laws and factual returns.
//! Collection, target construction, replay and search belong to their existing owners.
mod checkpoint;
mod publication;
pub use publication::PolicyValuePublication;

use super::OptimizerConfig;
use crate::models::predictive::{PlanningReturns, ReturnObjective};
use crate::{
    Result,
    core::{action::ConditionalAction, ensure, learner::UpdateReport, stateful::StatefulLearner},
    data::{
        policy_supervision::PolicySupervision, temporal::SequenceSample,
        value_supervision::ValueSupervision,
    },
    nn::{
        finite_loss,
        optimization::{optimize_validated, parameter_norm, validate_disjoint, validate_trainable},
        structured::{
            LightweightPolicy, LightweightValue, PackLimits, candidate_supervision_loss,
            value_supervision_loss,
        },
    },
};
use burn::{optim::ModuleOptimizer, tensor::Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchGuidedConfig {
    /// Discount per primitive physical tick, including ticks within an action.
    pub gamma: f64,
    pub policy_optimizer: OptimizerConfig,
    pub value_optimizer: OptimizerConfig,
    pub max_pack_elements: usize,
}
impl SearchGuidedConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.gamma.is_finite()
                && (0. ..=1.).contains(&self.gamma)
                && self.max_pack_elements > 0,
            "invalid search-guided learning configuration",
        )?;
        self.policy_optimizer.validate()?;
        self.value_optimizer.validate()
    }
    pub fn limits(&self) -> PackLimits {
        PackLimits {
            max_elements: self.max_pack_elements,
        }
    }
}

/// An ephemeral selected learning row. Target provenance is checked by its
/// source before constructing the batch; no search-node type enters optimization.
#[derive(Clone)]
pub struct PolicyValueRow {
    pub sample: SequenceSample,
    pub step: usize,
    pub actions: Vec<ConditionalAction>,
    pub probabilities: Vec<f64>,
}
#[derive(Clone)]
pub struct PolicyValueBatch {
    pub rows: Vec<PolicyValueRow>,
}

pub struct SearchGuidedPolicyValue {
    policy: LightweightPolicy,
    value: LightweightValue,
    policy_optimizer: ModuleOptimizer,
    value_optimizer: ModuleOptimizer,
    config: SearchGuidedConfig,
    objective: ReturnObjective,
    updates: u64,
    publication_revision: u64,
}
impl SearchGuidedPolicyValue {
    pub fn new(
        policy: LightweightPolicy,
        value: LightweightValue,
        config: SearchGuidedConfig,
    ) -> Result<Self> {
        Self::new_with_objective(policy, value, config, ReturnObjective::Physical)
    }

    pub fn new_with_objective(
        policy: LightweightPolicy,
        value: LightweightValue,
        config: SearchGuidedConfig,
        objective: ReturnObjective,
    ) -> Result<Self> {
        config.validate()?;
        ensure(
            objective.is_physical(),
            "search-guided value fits factual physical rewards",
        )?;
        policy.config().validate()?;
        value.config().validate()?;
        ensure(
            policy.config().input == value.config().input,
            "search-guided policy/value input schemas differ",
        )?;
        validate_disjoint(&policy, &value)?;
        validate_trainable(&policy)?;
        validate_trainable(&value)?;
        parameter_norm(&policy)?;
        parameter_norm(&value)?;
        Ok(Self {
            policy,
            value,
            policy_optimizer: config.policy_optimizer.build()?,
            value_optimizer: config.value_optimizer.build()?,
            config,
            objective,
            updates: 0,
            publication_revision: 0,
        })
    }
    pub fn policy(&self) -> &LightweightPolicy {
        &self.policy
    }
    pub fn value(&self) -> &LightweightValue {
        &self.value
    }
    pub fn config(&self) -> &SearchGuidedConfig {
        &self.config
    }
    pub fn returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: self.objective,
            gamma: self.config.gamma,
        }
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Publish a frozen pair for inference. Updating the learner leaves
    /// previously published probabilities and values unchanged.
    pub fn publish(&mut self) -> Result<PolicyValuePublication> {
        let revision = self
            .publication_revision
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("publication revision exhausted".into()))?;
        let publication = PolicyValuePublication::new(self, revision)?;
        self.publication_revision = revision;
        Ok(publication)
    }

    fn loss_tensors(&self, batch: &PolicyValueBatch) -> Result<(Tensor<1>, Tensor<1>)> {
        ensure(!batch.rows.is_empty(), "empty search-guided learning batch")?;
        let policies = batch
            .rows
            .iter()
            .map(|r| PolicySupervision::new(&r.sample, r.step, &r.actions, &r.probabilities))
            .collect::<Result<Vec<_>>>()?;
        let values = batch
            .rows
            .iter()
            .map(|r| ValueSupervision::monte_carlo(&r.sample, r.step, self.config.gamma))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            candidate_supervision_loss(&self.policy, &policies, self.config.limits())?,
            value_supervision_loss(&self.value, &values, self.config.limits())?,
        ))
    }

    /// Unweighted arithmetic means over selected rows: complete-action CE and
    /// full, genuinely terminated physical-return MSE. No Bellman target or E-step.
    pub fn losses(&self, batch: &PolicyValueBatch) -> Result<(f64, f64)> {
        let (policy, value) = self.loss_tensors(batch)?;
        Ok((finite_loss(&policy)?, finite_loss(&value)?))
    }
}

impl StatefulLearner for SearchGuidedPolicyValue {
    type Batch = PolicyValueBatch;
    fn update(&mut self, batch: &PolicyValueBatch) -> Result<UpdateReport> {
        ensure(
            self.updates + 1 < i32::MAX as u64,
            "search-guided Adam clock exhausted",
        )?;
        let (policy_loss, value_loss) = self.loss_tensors(batch)?;
        // Each optimizer validates before mutation. Staging the pair also guards
        // against the second model failing after the first optimizer succeeded.
        let mut policy = self.policy.clone();
        let mut value = self.value.clone();
        let mut policy_optimizer = self.policy_optimizer.clone();
        let mut value_optimizer = self.value_optimizer.clone();
        let (ce, policy_norm) = optimize_validated(
            &mut policy,
            &mut policy_optimizer,
            self.config.policy_optimizer.learning_rate,
            policy_loss,
        )?;
        let (mse, value_norm) = optimize_validated(
            &mut value,
            &mut value_optimizer,
            self.config.value_optimizer.learning_rate,
            value_loss,
        )?;
        parameter_norm(&policy)?;
        parameter_norm(&value)?;
        self.policy = policy;
        self.value = value;
        self.policy_optimizer = policy_optimizer;
        self.value_optimizer = value_optimizer;
        self.updates += 1;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("policy_loss", ce)
        .metric("value_loss", mse)
        .metric("policy_gradient_norm", policy_norm)
        .metric("value_gradient_norm", value_norm))
    }
}
