//! Replay-based continuous SAC with independent twin critics and Gaussian sampling.
mod checkpoint;
mod temperature;
#[cfg(test)]
mod tests;
pub use temperature::Temperature;

use super::{
    OptimizerConfig,
    continuous::critic::{CriticModel, PreparedReplayBatch, TargetActionBatch, bellman},
};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::ObservationBatch,
        learner::{
            ExplorationSource, Learner, OffPolicyLearner, OffPolicyRequirements, UpdateReport,
        },
        temporal::ReturnSpec,
    },
    data::replay::ReplayBatch,
    nn::{
        critics::scalar::{ActionValue, QValue},
        distributions::Distribution,
        gaussian_policy::{GaussianPolicy, validate_gaussian_space},
        optimization::{optimize_validated, parameter_norm, validate_trainable},
        policy::StochasticPolicy,
        target::frozen_copy,
        to_vec,
    },
    objectives::sac as objective,
};
use burn::{optim::ModuleOptimizer, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use temperature::TemperatureState;
const ACTOR_STREAM: u64 = 0x5341435f41435452;
const TARGET_STREAM: u64 = 0x5341435f54415247;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacConfig {
    pub actor: OptimizerConfig,
    pub critic: OptimizerConfig,
    pub temperature: Temperature,
    pub gamma: f32,
    pub target_tau: f32,
    pub target_interval: u64,
}
impl Default for SacConfig {
    fn default() -> Self {
        let optimizer = OptimizerConfig {
            max_grad_norm: None,
            ..Default::default()
        };
        Self {
            actor: optimizer.clone(),
            critic: optimizer,
            temperature: Temperature::default(),
            gamma: 0.99,
            target_tau: 0.005,
            target_interval: 1,
        }
    }
}
impl SacConfig {
    pub fn validate(&self) -> Result<()> {
        self.actor.validate()?;
        self.critic.validate()?;
        self.temperature.validate()?;
        ensure(
            self.gamma.is_finite()
                && (0.0..=1.0).contains(&self.gamma)
                && self.target_tau.is_finite()
                && (0.0..=1.0).contains(&self.target_tau)
                && self.target_interval > 0,
            "invalid SAC configuration",
        )
    }
    fn requirements(&self) -> OffPolicyRequirements {
        OffPolicyRequirements {
            returns: ReturnSpec {
                gamma: self.gamma,
                n_step: 1,
            },
            exploration: ExplorationSource::Policy,
        }
    }
}

/// One update fits both Q functions, projects the stochastic actor by pathwise
/// differentiation, updates optional temperature, then updates scheduled targets.
/// Direct replay callers must supply one-step transitions (including at gamma=1).
pub struct Sac<P: StochasticPolicy = GaussianPolicy, Q: ActionValue = QValue> {
    policy: P,
    first: CriticModel<Q>,
    second: CriticModel<Q>,
    actor_optimizer: ModuleOptimizer,
    temperature: Option<TemperatureState>,
    config: SacConfig,
    actor_rng: ChaCha8Rng,
    target_rng: ChaCha8Rng,
    updates: u64,
    poisoned: bool,
}
impl<P: StochasticPolicy, Q: ActionValue> Sac<P, Q> {
    pub fn new(policy: P, first: Q, second: Q, config: SacConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        validate_gaussian_space(policy.action_space())?;
        for q in [&first, &second] {
            ensure(
                q.action_dim() == policy.action_space().dimensions() && q.output_dim() == 1,
                "SAC needs scalar critics matching policy actions",
            )?;
            parameter_norm(&q.valid())?;
            validate_trainable(q)?;
            crate::nn::optimization::validate_disjoint(&policy, q)?;
        }
        crate::nn::optimization::validate_disjoint(&first, &second)?;
        parameter_norm(&policy.valid())?;
        validate_trainable(&policy)?;
        let temperature = TemperatureState::new(&config.temperature, policy.action_space())?;
        let mut actor_rng = ChaCha8Rng::seed_from_u64(seed);
        actor_rng.set_stream(ACTOR_STREAM);
        let mut target_rng = ChaCha8Rng::seed_from_u64(seed);
        target_rng.set_stream(TARGET_STREAM);
        Ok(Self {
            policy,
            first: CriticModel::new(first, &config.critic)?,
            second: CriticModel::new(second, &config.critic)?,
            actor_optimizer: config.actor.build()?,
            temperature,
            config,
            actor_rng,
            target_rng,
            updates: 0,
            poisoned: false,
        })
    }
    pub fn config(&self) -> &SacConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.policy)
    }
    pub fn critic(&self) -> Q {
        frozen_copy(&self.first.online)
    }
    pub fn second_critic(&self) -> Q {
        frozen_copy(&self.second.online)
    }
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.first.target)
    }
    pub fn target_second_critic(&self) -> Q {
        frozen_copy(&self.second.target)
    }
    pub fn alpha(&self) -> f32 {
        match (&self.config.temperature, &self.temperature) {
            (Temperature::Fixed(value), None) => *value,
            (_, Some(state)) => state.model.value(),
            _ => unreachable!("validated temperature ownership"),
        }
    }

    fn targets(&self, batch: &PreparedReplayBatch<'_>, rng: &mut ChaCha8Rng) -> Result<Tensor<2>> {
        let indices: Vec<_> = batch
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, item)| (item.discount > 0.).then_some(i))
            .collect();
        let observations: Vec<_> = indices
            .iter()
            .map(|&i| &batch.items[i].transition.next_observation)
            .collect();
        let (actions, log_probs) = if indices.is_empty() {
            (None, Vec::new())
        } else {
            let d = gaussian(
                &self.policy.valid(),
                ObservationBatch::from_refs(&observations),
            )?;
            let sample = d.reparameterize_with_sampled_score(rng)?;
            (Some(sample.actions.detach()), to_vec(sample.log_probs)?)
        };
        let next = TargetActionBatch {
            indices,
            observations,
            actions,
        };
        next.validate(batch.items.len(), self.policy.action_space().dimensions())?;
        let first = self.first.future(batch, &next)?;
        let second = self.second.future(batch, &next)?;
        let mut densities = vec![0.; batch.items.len()];
        for (&i, value) in next.indices.iter().zip(log_probs) {
            densities[i] = value;
        }
        Ok(bellman(
            batch,
            objective::soft_values(&first, &second, &densities, self.alpha())?,
            1,
        ))
    }
}
impl<P: StochasticPolicy, Q: ActionValue> Actor for Sac<P, Q> {
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        ensure(
            !self.poisoned,
            "SAC update failed; restore a complete checkpoint before inference",
        )?;
        crate::nn::policy::sample_checked(&self.policy.valid(), obs, mode, rng, actions)
    }
}
impl<P: StochasticPolicy, Q: ActionValue> OffPolicyLearner for Sac<P, Q> {
    fn set_exploration(&mut self, scale: f32) -> Result<()> {
        ensure(
            scale == 0.,
            "SAC samples its Gaussian policy; external exploration must be zero",
        )
    }
    fn replay_requirements(&self) -> Option<OffPolicyRequirements> {
        Some(self.config.requirements())
    }
}
impl<P: StochasticPolicy, Q: ActionValue> Learner for Sac<P, Q> {
    type Batch = ReplayBatch;
    fn actor(&self) -> &dyn Actor {
        self
    }
    fn update(&mut self, batch: &ReplayBatch) -> Result<UpdateReport> {
        ensure(
            !self.poisoned && self.updates < i32::MAX as u64 - 1,
            "invalid SAC update lifecycle",
        )?;
        let prepared = PreparedReplayBatch::new(batch, self.policy.action_space())?;
        for item in &batch.items {
            let expected = if item.transition.terminated {
                0.
            } else {
                self.config.gamma
            };
            ensure(
                item.discount == expected,
                "SAC requires one-step replay with matching gamma",
            )?;
        }
        let mut target_rng = self.target_rng.clone();
        let target = self.targets(&prepared, &mut target_rng)?;
        let target_mean = crate::nn::finite_loss(&target.clone().mean())?;
        let alpha = self.alpha();
        // Once optimization starts, an error cannot expose an incomplete learner.
        self.poisoned = true;
        self.target_rng = target_rng;
        let first = self
            .first
            .online
            .values(prepared.observations(), prepared.actions.clone())?;
        let first_loss = self
            .first
            .fit_validated((first - target.clone()).powf_scalar(2.).mean())?;
        let second = self
            .second
            .online
            .values(prepared.observations(), prepared.actions.clone())?;
        let second_loss = self
            .second
            .fit_validated((second - target).powf_scalar(2.).mean())?;
        let mut policy = self.policy.valid().train();
        let sample = gaussian(&policy, prepared.observations())?
            .reparameterize_with_sampled_score(&mut self.actor_rng)?;
        let temperature_densities = sample.log_probs.clone().detach();
        let entropy = -crate::nn::finite_loss(&temperature_densities.clone().mean())?;
        let first = self
            .first
            .online
            .valid()
            .freeze()
            .values(prepared.observations(), sample.actions.clone())?
            .squeeze_dim(1);
        let second = self
            .second
            .online
            .valid()
            .freeze()
            .values(prepared.observations(), sample.actions)?
            .squeeze_dim(1);
        let loss = objective::actor_loss(sample.log_probs, first, second, alpha)?;
        let (actor_loss, actor_grad_norm) = optimize_validated(
            &mut policy,
            &mut self.actor_optimizer,
            self.config.actor.learning_rate,
            loss,
        )?;
        parameter_norm(&policy.valid())?;
        self.policy = policy;
        let temperature_loss = match &mut self.temperature {
            Some(state) => state.update(temperature_densities)?,
            None => 0.,
        };
        let updates = self.updates + 1;
        let update_target = updates.is_multiple_of(self.config.target_interval);
        if update_target {
            self.first.update_target(self.config.target_tau)?;
            self.second.update_target(self.config.target_tau)?;
        }
        self.updates = updates;
        self.poisoned = false;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("critic_loss", first_loss + second_loss)
        .metric("actor_loss", actor_loss)
        .metric("actor_grad_norm", actor_grad_norm)
        .metric("target_mean", target_mean)
        .metric("entropy", entropy)
        .metric("alpha", alpha as f64)
        .metric("alpha_next", self.alpha() as f64)
        .metric("temperature_loss", temperature_loss)
        .metric("target_updated", if update_target { 1. } else { 0. }))
    }
}

/// Check custom policy distribution shape and physical action transform, not just its declared API.
fn gaussian<P: StochasticPolicy>(policy: &P, obs: ObservationBatch<'_>) -> Result<Distribution> {
    let d = crate::nn::policy::checked_distribution(policy, obs)?;
    ensure(
        matches!(d, Distribution::Gaussian { .. }),
        "SAC requires a Gaussian distribution matching its action space",
    )?;
    Ok(d)
}
