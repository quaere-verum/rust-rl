//! On-policy mirror descent: an unclipped importance surrogate with KL(new || behavior).
use super::common::{OnPolicyRecord, OnPolicyState};
use crate::{
    Error, Result,
    core::{
        actor::{Actor, OnPolicyActorRef},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::{Learner, OnPolicyLearner, UpdateReport},
    },
    data::{minibatch::shuffled_indices_into, rollout::OnPolicyBatch},
    nn::{
        critics::scalar::{StateValue, Value},
        distributions::Distribution,
        finite_loss,
        optimization::{optimize_validated, parameter_norm},
        policy::{ActorCritic, Policy, StochasticPolicy},
        target::frozen_copy,
        to_vec, vector,
    },
    objectives::mdpo::{MdpoTargets, mdpo_objective, normalized_occupancy_weights},
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Mirror step t_k, distinct from the actor's Adam learning rate. The penalty is 1/t_k.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MirrorStepSchedule {
    Constant(f32),
    /// At outer iteration zero use start; reach end at duration, then hold it.
    Linear {
        start: f32,
        end: f32,
        duration: u64,
    },
}
impl MirrorStepSchedule {
    pub fn value(&self, outer_iteration: u64) -> Result<f32> {
        let positive = |x: f32| x.is_finite() && x > 0. && x.recip().is_finite() && x.recip() > 0.;
        let t = match *self {
            Self::Constant(t) => {
                ensure(
                    positive(t),
                    "MDPO mirror step and reciprocal must be finite and positive",
                )?;
                t
            }
            Self::Linear {
                start,
                end,
                duration,
            } => {
                ensure(
                    positive(start) && positive(end) && duration > 0,
                    "invalid MDPO mirror-step schedule",
                )?;
                if outer_iteration >= duration {
                    end
                } else {
                    // Convex interpolation in f64 avoids cancellation/overflow for valid f32 endpoints.
                    let f = outer_iteration as f64 / duration as f64;
                    ((1. - f) * start as f64 + f * end as f64) as f32
                }
            }
        };
        ensure(positive(t), "invalid evaluated MDPO mirror step")?;
        Ok(t)
    }
}
/// Occupancy convention for both terms of the actor objective. Value fitting stays unweighted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MdpoWeighting {
    #[default]
    Uniform,
    DiscountedOccupancy,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdpoConfig {
    pub policy: super::OptimizerConfig,
    pub value: super::OptimizerConfig,
    /// Individual full-batch actor optimizer steps, never epochs over minibatches.
    pub inner_steps: usize,
    pub value_epochs: usize,
    pub value_minibatch_size: usize,
    pub mirror_step: MirrorStepSchedule,
    pub weighting: MdpoWeighting,
}
impl Default for MdpoConfig {
    fn default() -> Self {
        Self {
            policy: Default::default(),
            value: Default::default(),
            inner_steps: 10,
            value_epochs: 4,
            value_minibatch_size: 64,
            mirror_step: MirrorStepSchedule::Constant(1.),
            weighting: MdpoWeighting::Uniform,
        }
    }
}
impl MdpoConfig {
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.value.validate()?;
        self.mirror_step.value(0)?;
        ensure(
            self.inner_steps > 0 && self.value_epochs > 0 && self.value_minibatch_size > 0,
            "MDPO actor steps, value epochs and minibatch size must be positive",
        )
    }
}

/// Immutable empirical targets and independent behavior distribution for one outer update.
struct MirrorObjective<'a> {
    batch: &'a OnPolicyBatch,
    behavior: Distribution,
    old_log_probs: Tensor<1>,
    weights: Tensor<1>,
    advantages: Tensor<1>,
}
struct Objective {
    loss: Tensor<1>,
    surrogate: Tensor<1>,
    kl: Tensor<1>,
}
impl<'a> MirrorObjective<'a> {
    fn new(
        batch: &'a OnPolicyBatch,
        behavior: Distribution,
        weighting: MdpoWeighting,
    ) -> Result<Self> {
        let metadata = batch.bootstrapped()?;
        let weights = match weighting {
            MdpoWeighting::Uniform => vec![1. / batch.len() as f32; batch.len()],
            MdpoWeighting::DiscountedOccupancy => {
                normalized_occupancy_weights(&batch.discount_weights)?
            }
        };
        let old_log_probs = behavior
            .log_prob(&batch.actions, &batch.latents)?
            .without_autodiff();
        let old = to_vec(old_log_probs.clone())?;
        ensure(
            old.len() == metadata.log_probs.len()
                && old
                    .iter()
                    .zip(&metadata.log_probs)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5 * (1. + a.abs())),
            "MDPO batch must come from the current frozen behavior policy",
        )?;
        Ok(Self {
            batch,
            behavior,
            old_log_probs,
            weights: vector(&weights),
            advantages: vector(&metadata.advantages),
        })
    }
    fn evaluate(&self, current: &Distribution, coefficient: f32) -> Result<Objective> {
        let result = mdpo_objective(
            current,
            &self.behavior,
            MdpoTargets {
                actions: &self.batch.actions,
                latents: &self.batch.latents,
                old_log_probs: &self.old_log_probs,
                weights: &self.weights,
                advantages: &self.advantages,
            },
            coefficient,
        )?;
        Ok(Objective {
            loss: result.loss,
            surrogate: result.surrogate,
            kl: result.kl,
        })
    }
}

/// Full-batch on-policy MDPO, with a separately fitted scalar value critic.
pub struct Mdpo<P: StochasticPolicy = Policy, V: StateValue = Value> {
    state: OnPolicyState<P, V>,
    config: MdpoConfig,
    outer_iterations: u64,
    actor_updates: u64,
    critic_updates: u64,
}
#[derive(Serialize, Deserialize)]
struct MdpoRecord {
    outer_iterations: u64,
    actor_updates: u64,
    critic_updates: u64,
    policy_optimizer_ids: Vec<u64>,
    value_optimizer_ids: Vec<u64>,
}
impl<P: StochasticPolicy, V: StateValue> Mdpo<P, V> {
    pub fn new(models: ActorCritic<P, V>, config: MdpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        parameter_norm(&models.policy)?;
        parameter_norm(&models.value)?;
        Ok(Self {
            state: OnPolicyState::new(models, &config.policy, &config.value, seed)?,
            config,
            outer_iterations: 0,
            actor_updates: 0,
            critic_updates: 0,
        })
    }
    pub fn config(&self) -> &MdpoConfig {
        &self.config
    }
    /// Total individual actor and critic optimizer steps.
    pub fn updates(&self) -> u64 {
        self.state.updates
    }
    pub fn outer_iterations(&self) -> u64 {
        self.outer_iterations
    }
    pub fn actor_updates(&self) -> u64 {
        self.actor_updates
    }
    pub fn critic_updates(&self) -> u64 {
        self.critic_updates
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.state.models.policy)
    }
    pub fn value(&self) -> V {
        frozen_copy(&self.state.models.value)
    }
}
impl<P: StochasticPolicy, V: StateValue> OnPolicyLearner for Mdpo<P, V> {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Bootstrapped(&self.state.models)
    }
}
impl<P: StochasticPolicy, V: StateValue> Checkpoint for Mdpo<P, V> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record("mdpo", &self.config)?;
        record.metadata["outer_iterations"] = self.outer_iterations.into();
        record.metadata["actor_updates"] = self.actor_updates.into();
        record.metadata["critic_updates"] = self.critic_updates.into();
        record.metadata["policy_optimizer_ids"] =
            serde_json::to_value(optimizer_ids(&record, "policy")?)?;
        record.metadata["value_optimizer_ids"] =
            serde_json::to_value(optimizer_ids(&record, "value")?)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("mdpo")?;
        let metadata: MdpoRecord = serde_json::from_value(record.metadata.clone())?;
        let saved: OnPolicyRecord<MdpoConfig> = serde_json::from_value(record.metadata.clone())?;
        saved.config.validate()?;
        ensure(
            metadata.actor_updates.checked_add(metadata.critic_updates) == Some(saved.updates)
                && metadata.outer_iterations <= metadata.actor_updates
                && metadata.outer_iterations <= metadata.critic_updates,
            "inconsistent MDPO checkpoint counters",
        )?;
        validate_optimizer_numbers(
            record,
            "policy",
            &metadata.policy_optimizer_ids,
            metadata.actor_updates,
        )?;
        validate_optimizer_numbers(
            record,
            "value",
            &metadata.value_optimizer_ids,
            metadata.critic_updates,
        )?;
        let config = saved.config.clone();
        let state = self
            .state
            .restored(record, &saved, &config.policy, &config.value)?;
        parameter_norm(&state.models.policy)?;
        parameter_norm(&state.models.value)?;
        *self = Self {
            state,
            config,
            outer_iterations: metadata.outer_iterations,
            actor_updates: metadata.actor_updates,
            critic_updates: metadata.critic_updates,
        };
        Ok(())
    }
}
impl<P: StochasticPolicy, V: StateValue> Learner for Mdpo<P, V> {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        &self.state.models
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        batch.validate()?;
        let t = self.config.mirror_step.value(self.outer_iterations)?;
        // All validation uses an independent inference snapshot, before optimizer/RNG mutation.
        let behavior = self.policy();
        let objective = MirrorObjective::new(
            batch,
            behavior.distribution((&batch.observations).into())?,
            self.config.weighting,
        )?;
        // Validate value shape/targets before policy improvement, without changing live running buffers.
        let prediction = self.value().values((&batch.observations).into())?;
        ensure(
            prediction.dims() == [batch.len()],
            "MDPO state value must return one scalar per observation",
        )?;
        finite_loss(&(prediction - vector(&batch.returns)).powf_scalar(2.).mean())?;
        let critic_steps = self
            .config
            .value_epochs
            .checked_mul(batch.len().div_ceil(self.config.value_minibatch_size))
            .ok_or_else(|| Error::Config("MDPO update count overflow".into()))?;
        let work = self
            .config
            .inner_steps
            .checked_add(critic_steps)
            .and_then(|x| u64::try_from(x).ok())
            .ok_or_else(|| Error::Config("MDPO update count overflow".into()))?;
        ensure(
            self.state.updates.checked_add(work).is_some()
                && self.outer_iterations.checked_add(1).is_some(),
            "MDPO counter overflow",
        )?;
        ensure(
            self.actor_updates
                .checked_add(self.config.inner_steps as u64)
                .is_some_and(|x| x < i32::MAX as u64)
                && self
                    .critic_updates
                    .checked_add(critic_steps as u64)
                    .is_some_and(|x| x < i32::MAX as u64),
            "MDPO Adam step count exceeds supported bias-correction range",
        )?;
        let mut actor_grad = 0.;
        for _ in 0..self.config.inner_steps {
            let current = self
                .state
                .models
                .policy
                .distribution((&batch.observations).into())?;
            let loss = objective.evaluate(&current, t.recip())?.loss;
            let (_, norm) = optimize_validated(
                &mut self.state.models.policy,
                &mut self.state.policy_optimizer,
                self.config.policy.learning_rate,
                loss,
            )?;
            actor_grad = norm;
            self.actor_updates += 1;
            self.state.updates += 1;
        }
        // Final metrics use the same frozen behavior/targets, after all m actor steps.
        let current = self
            .state
            .models
            .policy
            .valid()
            .distribution((&batch.observations).into())?;
        let final_objective = objective.evaluate(&current, t.recip())?;
        let actor_loss = finite_loss(&final_objective.loss)?;
        let surrogate = finite_loss(&final_objective.surrogate)?;
        let kl = finite_loss(&final_objective.kl)?;
        let mut indices = Vec::new();
        let mut selected = OnPolicyBatch::default();
        let mut value_loss = 0.;
        let mut critic_grad = 0.;
        for _ in 0..self.config.value_epochs {
            shuffled_indices_into(
                batch.len(),
                self.config.value_minibatch_size,
                &mut self.state.rng,
                &mut indices,
            )?;
            for selection in indices.chunks(self.config.value_minibatch_size) {
                batch.select_validated_into(selection, &mut selected);
                let loss = (self
                    .state
                    .models
                    .value
                    .values((&selected.observations).into())?
                    - vector(&selected.returns))
                .powf_scalar(2.)
                .mean();
                let (number, norm) = optimize_validated(
                    &mut self.state.models.value,
                    &mut self.state.value_optimizer,
                    self.config.value.learning_rate,
                    loss,
                )?;
                value_loss += number * selection.len() as f64;
                critic_grad = norm;
                self.critic_updates += 1;
                self.state.updates += 1;
            }
        }
        let entropy = finite_loss(
            &(objective.weights.clone() * current.entropy(&mut self.state.rng)?).sum(),
        )?;
        self.outer_iterations += 1;
        Ok(UpdateReport {
            updates: work,
            ..Default::default()
        }
        .metric("policy_loss", actor_loss)
        .metric("advantage_surrogate", surrogate)
        .metric(
            "value_loss",
            value_loss / (batch.len() * self.config.value_epochs) as f64,
        )
        .metric("kl_new_behavior", kl)
        .metric("entropy", entropy)
        .metric("mirror_step", t as f64)
        .metric("kl_coefficient", t.recip() as f64)
        .metric("actor_updates", self.config.inner_steps as f64)
        .metric("critic_updates", critic_steps as f64)
        .metric("outer_iterations", self.outer_iterations as f64)
        .metric(
            "actor_parameter_norm",
            parameter_norm(&self.state.models.policy)?,
        )
        .metric(
            "critic_parameter_norm",
            parameter_norm(&self.state.models.value)?,
        )
        .metric("actor_gradient_norm", actor_grad)
        .metric("critic_gradient_norm", critic_grad))
    }
}

#[cfg(test)]
#[path = "mdpo_math.rs"]
mod math_tests;

// Bind optimizer membership to this checkpoint, including whole parameter groups.
pub(super) fn optimizer_ids(record: &LearnerRecord, name: &str) -> Result<Vec<u64>> {
    use crate::nn::checkpoint::record_error;
    let bytes = record
        .optimizers
        .get(name)
        .ok_or_else(|| Error::State(format!("checkpoint missing optimizer {name}")))?;
    let reader = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.clone()))
        .map_err(record_error)?;
    let ids = reader
        .scalars()
        .keys()
        .filter_map(|key| key.strip_suffix(".__rank"))
        .map(|id| id.parse::<u64>().map_err(record_error))
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    Ok(ids.into_iter().collect())
}

// Shared codecs validate shapes and individual states; MDPO also checks membership,
// model ownership, finite Adam moments and the scalar bias-correction counters.
pub(super) fn validate_optimizer_numbers(
    record: &LearnerRecord,
    name: &str,
    expected_ids: &[u64],
    updates: u64,
) -> Result<()> {
    use crate::nn::checkpoint::record_error;
    ensure(
        updates < i32::MAX as u64,
        "MDPO Adam step count exceeds supported bias-correction range",
    )?;
    let ids = optimizer_ids(record, name)?;
    ensure(
        ids == expected_ids,
        "MDPO optimizer parameter groups are incomplete or foreign",
    )?;
    let model = record
        .models
        .get(name)
        .ok_or_else(|| Error::State(format!("checkpoint missing model {name}")))?;
    let model_reader =
        burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(model.bytes.clone()))
            .map_err(record_error)?;
    let owned: std::collections::BTreeSet<_> = model_reader
        .into_tensors()
        .map_err(record_error)?
        .into_iter()
        .filter_map(|tensor| tensor.param_id)
        .collect();
    ensure(
        ids.iter().all(|id| owned.contains(id)),
        "MDPO optimizer references another model's parameters",
    )?;
    let bytes = record
        .optimizers
        .get(name)
        .ok_or_else(|| Error::State(format!("checkpoint missing optimizer {name}")))?;
    let reader = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.clone()))
        .map_err(record_error)?;
    let mut time_ids = std::collections::BTreeSet::new();
    for (key, value) in reader.scalars() {
        if key.ends_with(".time") {
            let id = key
                .split('.')
                .next()
                .ok_or_else(|| Error::State("invalid MDPO optimizer key".into()))?
                .parse::<u64>()
                .map_err(record_error)?;
            let time = u64::try_from(*value).map_err(record_error)?;
            ensure(
                time > 0 && time <= updates && time < i32::MAX as u64,
                "invalid MDPO Adam step counter",
            )?;
            time_ids.insert(id);
        }
    }
    ensure(
        time_ids.into_iter().collect::<Vec<_>>() == ids,
        "MDPO optimizer step counters are incomplete",
    )?;
    for tensor in reader.into_tensors().map_err(record_error)? {
        let (key, dtype, _, _, bytes) = tensor.into_parts().map_err(record_error)?;
        ensure(
            dtype == burn_pack::DType::F32,
            "MDPO optimizer states must use f32",
        )?;
        for word in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            ensure(
                value.is_finite() && (!key.ends_with(".moment_2") || value >= 0.),
                "invalid MDPO optimizer moment",
            )?;
        }
    }
    Ok(())
}
