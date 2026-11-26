//! Small supervised policy pretraining facility; this is not offline RL.
mod checkpoint;
mod dataset;
#[cfg(test)]
mod tests;
use crate::{
    Result,
    algorithms::OptimizerConfig,
    core::{action::Action, ensure, env::ObservationBatch, learner::UpdateReport},
    data::minibatch::shuffled_indices_into,
    nn::{
        finite_loss,
        optimization::{optimize_validated, parameter_norm, validate_trainable},
        policy::{Policy, StochasticPolicy, checked_distribution},
        target::frozen_copy,
    },
};
use burn::{optim::ModuleOptimizer, tensor::Tensor};
pub use dataset::Demonstrations;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloningConfig {
    pub optimizer: OptimizerConfig,
    pub batch_size: usize,
}
impl Default for CloningConfig {
    fn default() -> Self {
        Self {
            optimizer: Default::default(),
            batch_size: 64,
        }
    }
}
impl CloningConfig {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        ensure(self.batch_size > 0, "zero cloning minibatch size")
    }
}
pub struct BehaviorCloning<P: StochasticPolicy = Policy> {
    policy: P,
    config: CloningConfig,
    optimizer: ModuleOptimizer,
    rng: ChaCha8Rng,
    updates: u64,
    epochs: u64,
    poisoned: bool,
    indices: Vec<usize>,
    scratch: BatchScratch,
}
impl<P: StochasticPolicy> BehaviorCloning<P> {
    pub fn new(policy: P, config: CloningConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        policy.action_space().validate()?;
        parameter_norm(&policy.valid())?;
        validate_trainable(&policy)?;
        Ok(Self {
            policy,
            optimizer: config.optimizer.build()?,
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
            epochs: 0,
            poisoned: false,
            indices: Vec::new(),
            scratch: BatchScratch::default(),
        })
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.policy)
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn epochs(&self) -> u64 {
        self.epochs
    }
    fn check_data(&self, data: &Demonstrations) -> Result<()> {
        ensure(
            !self.poisoned,
            "cloning update failed; restore a complete checkpoint",
        )?;
        ensure(
            data.action_space() == self.policy.action_space(),
            "demonstration action space differs from policy",
        )
    }
    /// One independently shuffled epoch without replacement, including a short
    /// final batch. Epoch NLL is weighted by the number of examples, not batches.
    pub fn fit_epoch(&mut self, data: &Demonstrations) -> Result<UpdateReport> {
        self.check_data(data)?;
        let batches = data.len().div_ceil(self.config.batch_size) as u64;
        ensure(
            self.updates
                .checked_add(batches)
                .is_some_and(|n| n < i32::MAX as u64)
                && self.epochs < u64::MAX,
            "cloning update counter overflow",
        )?;
        shuffled_indices_into(
            data.len(),
            self.config.batch_size,
            &mut self.rng,
            &mut self.indices,
        )?;
        self.poisoned = true;
        let mut total = 0.;
        for indices in self.indices.chunks(self.config.batch_size) {
            let mut policy = self.policy.clone().train();
            let loss = self.scratch.loss(&policy, data, indices)?;
            let value = optimize_validated(
                &mut policy,
                &mut self.optimizer,
                self.config.optimizer.learning_rate,
                loss,
            )?
            .0;
            // valid() synchronizes pending running buffers on this private candidate.
            parameter_norm(&policy.valid())?;
            self.policy = policy;
            self.updates += 1;
            total += value * indices.len() as f64;
        }
        self.epochs += 1;
        self.poisoned = false;
        Ok(UpdateReport {
            updates: batches,
            ..Default::default()
        }
        .metric("negative_log_likelihood", total / data.len() as f64)
        .metric("examples", data.len() as f64))
    }
    /// Held-out mean NLL in stable input order. This does not consume training RNG.
    pub fn evaluate(&self, data: &Demonstrations) -> Result<f64> {
        self.check_data(data)?;
        let policy = self.policy.valid();
        let indices: Vec<_> = (0..data.len()).collect();
        let mut scratch = BatchScratch::default();
        let mut total = 0.;
        for selected in indices.chunks(self.config.batch_size) {
            total += finite_loss(&scratch.loss(&policy, data, selected)?)? * selected.len() as f64;
        }
        Ok(total / data.len() as f64)
    }
}
#[derive(Default)]
struct BatchScratch {
    actions: Vec<Action>,
    latents: Vec<Option<Vec<f32>>>,
}
impl BatchScratch {
    fn loss<P: StochasticPolicy>(
        &mut self,
        policy: &P,
        data: &Demonstrations,
        indices: &[usize],
    ) -> Result<Tensor<1>> {
        let observations = ObservationBatch::selected(&data.observations, indices)?;
        self.actions.truncate(indices.len());
        self.latents.truncate(indices.len());
        for (row, &index) in indices.iter().enumerate() {
            if row == self.actions.len() {
                self.actions.push(data.actions[index].clone());
                self.latents.push(data.latents[index].clone());
            } else {
                self.actions[row].clone_from(&data.actions[index]);
                self.latents[row].clone_from(&data.latents[index]);
            }
        }
        Ok(-checked_distribution(policy, observations)?
            .log_prob(&self.actions, &self.latents)?
            .mean())
    }
}
