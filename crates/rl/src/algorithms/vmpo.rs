use super::common::{OnPolicyRecord, OnPolicyState, OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        actor::{Actor, OnPolicyActorRef},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::{Learner, OnPolicyLearner, UpdateReport},
    },
    data::{minibatch::shuffled_indices_into, rollout::OnPolicyBatch},
    nn::{
        checkpoint::RecordBurnExt,
        critics::scalar::{StateValue, Value},
        policy::{ActorCritic, Policy, StochasticPolicy},
        to_vec, vector,
    },
    objectives::vmpo::{
        dual_objective, policy_objective, temperature_weights, top_half_indices_into,
    },
};
use burn::{
    module::{Module, Param},
    optim::ModuleOptimizer,
    tensor::{Tensor, activation::softplus},
};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmpoConfig {
    pub policy: OptimizerConfig,
    pub value: OptimizerConfig,
    pub dual: OptimizerConfig,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub temperature_constraint: f32,
    pub kl_constraint: f32,
}
impl Default for VmpoConfig {
    fn default() -> Self {
        Self {
            policy: Default::default(),
            value: Default::default(),
            dual: OptimizerConfig {
                learning_rate: 0.02,
                ..Default::default()
            },
            epochs: 4,
            minibatch_size: 64,
            temperature_constraint: 0.01,
            kl_constraint: 0.1,
        }
    }
}
impl VmpoConfig {
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.value.validate()?;
        self.dual.validate()?;
        ensure(
            self.epochs > 0
                && self.minibatch_size > 0
                && self.temperature_constraint.is_finite()
                && self.temperature_constraint > 0.
                && self.kl_constraint.is_finite()
                && self.kl_constraint > 0.,
            "invalid VMPO constraints",
        )
    }
}
#[derive(Module, Debug)]
pub struct Multipliers {
    pub log_eta: Param<Tensor<1>>,
    pub log_alpha: Param<Tensor<1>>,
}

pub struct Vmpo<P: StochasticPolicy = Policy, V: StateValue = Value> {
    pub(crate) state: OnPolicyState<P, V>,
    pub(crate) config: VmpoConfig,
    pub(crate) multipliers: Multipliers,
    pub(crate) dual_optimizer: ModuleOptimizer,
}
impl<P: StochasticPolicy, V: StateValue> Vmpo<P, V> {
    pub fn new(models: ActorCritic<P, V>, config: VmpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: OnPolicyState::new(models, &config.policy, &config.value, seed)?,
            dual_optimizer: config.dual.build()?,
            multipliers: Multipliers {
                log_eta: Param::from_tensor(vector(&[0.05])),
                log_alpha: Param::from_tensor(vector(&[0.1])),
            },
            config,
        })
    }
    pub fn config(&self) -> &VmpoConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.state.updates
    }
    /// Independent inference snapshot, including running-state buffers.
    pub fn policy(&self) -> P {
        crate::nn::target::frozen_copy(&self.state.models.policy)
    }
    pub fn value(&self) -> V {
        crate::nn::target::frozen_copy(&self.state.models.value)
    }
    pub fn multipliers(&self) -> (f32, f32) {
        (
            softplus(self.multipliers.log_eta.val(), 1.)
                .into_scalar::<f32>()
                .max(1e-6),
            softplus(self.multipliers.log_alpha.val(), 1.)
                .into_scalar::<f32>()
                .max(1e-6),
        )
    }
}
impl<P: StochasticPolicy, V: StateValue> OnPolicyLearner for Vmpo<P, V> {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Bootstrapped(&self.state.models)
    }
}
impl<P: StochasticPolicy, V: StateValue> Checkpoint for Vmpo<P, V> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record("vmpo", &self.config)?;
        record.model("multipliers", &self.multipliers)?;
        record.optimizer("dual", &self.dual_optimizer)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("vmpo")?;
        let metadata: OnPolicyRecord<VmpoConfig> = serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let state = self.state.restored(
            record,
            &metadata,
            &metadata.config.policy,
            &metadata.config.value,
        )?;
        let multipliers = record.load_model("multipliers", &self.multipliers)?;
        ensure(
            multipliers.log_eta.val().into_scalar::<f32>().is_finite()
                && multipliers.log_alpha.val().into_scalar::<f32>().is_finite(),
            "invalid VMPO multipliers",
        )?;
        let dual_optimizer = record.load_optimizer("dual", metadata.config.dual.build()?)?;
        *self = Self {
            state,
            config: metadata.config,
            multipliers,
            dual_optimizer,
        };
        Ok(())
    }
}
impl<P: StochasticPolicy, V: StateValue> Learner for Vmpo<P, V> {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        &self.state.models
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        batch.validate()?;
        let metadata = batch.bootstrapped()?;
        let behavior = self.state.models.policy.valid();
        let old = to_vec(
            behavior
                .distribution((&batch.observations).into())?
                .log_prob(&batch.actions, &batch.latents)?,
        )?;
        ensure(
            old.iter()
                .zip(&metadata.log_probs)
                // Host sampling and Burn's reduction use different f32 summation orders.
                // Scale the tolerance like the distribution conformance tests.
                .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5 * (1. + a.abs())),
            "VMPO batch must come from the current frozen behavior policy",
        )?;
        let mut report = UpdateReport::default();
        let mut indices = Vec::new();
        let mut b = OnPolicyBatch::default();
        let mut top = Vec::new();
        let mut selected = OnPolicyBatch::default();
        for _ in 0..self.config.epochs {
            shuffled_indices_into(
                batch.len(),
                self.config.minibatch_size,
                &mut self.state.rng,
                &mut indices,
            )?;
            for selection in indices.chunks(self.config.minibatch_size) {
                batch.select_validated_into(selection, &mut b);
                let metadata = b.bootstrapped()?;
                top_half_indices_into(&metadata.advantages, &mut top);
                b.select_validated_into(&top, &mut selected);
                let selected_metadata = selected.bootstrapped()?;
                let eta = softplus(self.multipliers.log_eta.val(), 1.)
                    .into_scalar::<f32>()
                    .max(1e-6);
                let alpha = softplus(self.multipliers.log_alpha.val(), 1.)
                    .into_scalar::<f32>()
                    .max(1e-6);
                let (weights, maximum) = temperature_weights(&selected_metadata.advantages, eta);
                let lp = self
                    .state
                    .models
                    .policy
                    .distribution((&selected.observations).into())?
                    .log_prob(&selected.actions, &selected.latents)?;
                // KL(new || frozen behavior), matching QRL's stated direction.
                let kl = self
                    .state
                    .models
                    .policy
                    .distribution((&b.observations).into())?
                    .kl(&behavior.distribution((&b.observations).into())?)?
                    .mean();
                let kl_number = kl.clone().into_scalar::<f32>();
                let p = optimize(
                    &mut self.state.models.policy,
                    &mut self.state.policy_optimizer,
                    self.config.policy.learning_rate,
                    policy_objective(weights, lp, kl, alpha),
                )?;
                let dual = dual_objective(
                    self.multipliers.log_eta.val(),
                    self.multipliers.log_alpha.val(),
                    &selected_metadata.advantages,
                    maximum,
                    self.config.temperature_constraint,
                    self.config.kl_constraint,
                    kl_number,
                );
                let dual_loss = optimize(
                    &mut self.multipliers,
                    &mut self.dual_optimizer,
                    self.config.dual.learning_rate,
                    dual,
                )?;
                let v = self.state.fit_value(&b, &self.config.value)?;
                self.state.updates += 1;
                report.updates += 1;
                report = report
                    .metric("policy_loss", p)
                    .metric("value_loss", v)
                    .metric("dual_loss", dual_loss)
                    .metric("kl_new_behavior", kl_number as f64);
            }
        }
        Ok(report)
    }
}

#[cfg(test)]
mod selection_tests {
    use super::top_half_indices_into;

    #[test]
    fn top_half_keeps_stable_ties_and_ceil_for_short_final_minibatches() {
        let mut top = Vec::new();
        top_half_indices_into(&[2., 9., 9., 3., 0.], &mut top);
        assert_eq!(top, [1, 2, 3]);
        let pointer = top.as_ptr();
        top_half_indices_into(&[4.], &mut top);
        assert_eq!(top, [0]);
        assert_eq!(top.as_ptr(), pointer);
    }
}
