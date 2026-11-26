use crate::{
    Result,
    core::{action::ActionSpace, checkpoint::LearnerRecord, ensure},
    data::rollout::OnPolicyBatch,
    nn::{
        checkpoint::RecordBurnExt,
        critics::scalar::StateValue,
        finite_loss,
        policy::{ActorCritic, StochasticPolicy},
        vector,
    },
};
use burn::{
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{AdamConfig, GradientsParams, ModuleOptimizer},
    tensor::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub max_grad_norm: Option<f32>,
}
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            max_grad_norm: Some(1.),
        }
    }
}
impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.
                && self.max_grad_norm.is_none_or(|x| x.is_finite() && x > 0.),
            "invalid optimizer settings",
        )
    }
    pub fn build(&self) -> Result<ModuleOptimizer> {
        self.validate()?;
        Ok(AdamConfig::new()
            .with_grad_clipping(self.max_grad_norm.map(GradientClippingConfig::Norm))
            .init())
    }
}
pub fn optimize<M: Module>(
    model: &mut M,
    opt: &mut ModuleOptimizer,
    lr: f64,
    loss: Tensor<1>,
) -> Result<f64> {
    let number = finite_loss(&loss)?;
    let grads = GradientsParams::from_grads(loss.backward(), model);
    *model = opt.step(lr, model.clone(), grads);
    Ok(number)
}

pub struct OnPolicyState<P: StochasticPolicy, V: StateValue> {
    pub(crate) models: ActorCritic<P, V>,
    pub(crate) policy_optimizer: ModuleOptimizer,
    pub(crate) value_optimizer: ModuleOptimizer,
    pub(crate) rng: ChaCha8Rng,
    pub(crate) updates: u64,
}
#[derive(Serialize, Deserialize)]
pub(crate) struct OnPolicyRecord<C> {
    pub config: C,
    pub rng: ChaCha8Rng,
    pub updates: u64,
    pub space: ActionSpace,
}
impl<P: StochasticPolicy, V: StateValue> OnPolicyState<P, V> {
    pub fn new(
        models: ActorCritic<P, V>,
        policy: &OptimizerConfig,
        value: &OptimizerConfig,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self {
            models,
            policy_optimizer: policy.build()?,
            value_optimizer: value.build()?,
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
        })
    }
    pub fn fit_value(&mut self, b: &OnPolicyBatch, c: &OptimizerConfig) -> Result<f64> {
        let loss = (self.models.value.values((&b.observations).into())? - vector(&b.returns))
            .powf_scalar(2.)
            .mean();
        optimize(
            &mut self.models.value,
            &mut self.value_optimizer,
            c.learning_rate,
            loss,
        )
    }
    pub(crate) fn record(&self, algorithm: &str, config: &impl Serialize) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new(
            algorithm,
            OnPolicyRecord {
                config,
                rng: self.rng.clone(),
                updates: self.updates,
                space: self.models.policy.action_space().clone(),
            },
        )?;
        record.model("policy", &self.models.policy)?;
        record.model("value", &self.models.value)?;
        record.optimizer("policy", &self.policy_optimizer)?;
        record.optimizer("value", &self.value_optimizer)?;
        Ok(record)
    }
    /// Decode into an independent replacement; the owner validates its full config first.
    pub(crate) fn restored<C>(
        &self,
        record: &LearnerRecord,
        metadata: &OnPolicyRecord<C>,
        policy: &OptimizerConfig,
        value: &OptimizerConfig,
    ) -> Result<Self> {
        ensure(
            &metadata.space == self.models.policy.action_space(),
            "checkpoint action space mismatch",
        )?;
        Ok(Self {
            models: ActorCritic {
                policy: record.load_model("policy", &self.models.policy)?,
                value: record.load_model("value", &self.models.value)?,
            },
            policy_optimizer: record.load_optimizer("policy", policy.build()?)?,
            value_optimizer: record.load_optimizer("value", value.build()?)?,
            rng: metadata.rng.clone(),
            updates: metadata.updates,
        })
    }
}
