//! Shared deterministic actor update cadence and target-action preparation.
pub(crate) mod critic;
use self::critic::{CriticCheckpoint, CriticRule, PreparedReplayBatch, TargetActionBatch};
use super::common::{OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        checkpoint::LearnerRecord,
        ensure,
        env::ObservationBatch,
        learner::{Learner, OffPolicyLearner, UpdateReport},
    },
    data::replay::ReplayBatch,
    nn::{
        checkpoint::RecordBurnExt,
        exploration::gaussian,
        matrix,
        policy::DeterministicPolicy,
        target::{frozen_copy, polyak_update},
        to_vec,
    },
};
use burn::optim::ModuleOptimizer;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousConfig {
    pub actor: OptimizerConfig,
    pub critic: OptimizerConfig,
    pub tau: f32,
    pub exploration_std: f32,
    pub policy_delay: u64,
    pub target_noise: f32,
    pub noise_clip: f32,
}
impl Default for ContinuousConfig {
    fn default() -> Self {
        Self {
            actor: Default::default(),
            critic: OptimizerConfig {
                learning_rate: 1e-3,
                ..Default::default()
            },
            tau: 0.005,
            exploration_std: 0.1,
            policy_delay: 1,
            target_noise: 0.,
            noise_clip: 0.5,
        }
    }
}
impl ContinuousConfig {
    pub fn validate(&self) -> Result<()> {
        self.actor.validate()?;
        self.critic.validate()?;
        ensure(
            self.tau.is_finite()
                && (0.0..=1.0).contains(&self.tau)
                && self.policy_delay > 0
                && [self.exploration_std, self.target_noise, self.noise_clip]
                    .iter()
                    .all(|x| x.is_finite() && *x >= 0.),
            "invalid continuous learner configuration",
        )
    }
}

struct ContinuousActor<P: DeterministicPolicy> {
    policy: P,
    noise: f32,
}
impl<P: DeterministicPolicy> Actor for ContinuousActor<P> {
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        let d = self.action_space().dimensions();
        let values = to_vec(self.policy.valid().actions(obs)?)?;
        ensure(
            values.len() == obs.len() * d,
            "policy action batch dimensions differ",
        )?;
        out.truncate(obs.len());
        for (i, row) in values.chunks(d).enumerate() {
            if i == out.len() {
                out.push(Action::Continuous(Vec::with_capacity(d)));
            }
            if !matches!(out[i], Action::Continuous(_)) {
                out[i] = Action::Continuous(Vec::with_capacity(d));
            }
            let Action::Continuous(action) = &mut out[i] else {
                unreachable!()
            };
            action.clear();
            action.extend_from_slice(row);
            if mode == ActMode::Sample {
                gaussian(&mut out[i], self.action_space(), self.noise, rng)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn smooth_target_actions(
    actions: &mut [f32],
    space: &ActionSpace,
    standard_deviation: f32,
    noise_clip: f32,
    rng: &mut ChaCha8Rng,
) {
    for row in actions.chunks_mut(space.dimensions()) {
        for (j, value) in row.iter_mut().enumerate() {
            let noise = rng.sample::<f32, _>(StandardNormal) * standard_deviation;
            *value += noise.clamp(-noise_clip, noise_clip);
            if let ActionSpace::Box { low, high } = space {
                *value = value.clamp(low[j], high[j]);
            }
        }
    }
}

pub(super) struct ContinuousLearner<P: DeterministicPolicy, C: CriticRule> {
    actor: ContinuousActor<P>,
    target_policy: P,
    actor_optimizer: ModuleOptimizer,
    pub(super) critic: C,
    config: ContinuousConfig,
    rng: ChaCha8Rng,
    updates: u64,
    actor_updates: u64,
}
impl<P: DeterministicPolicy, C: CriticRule> ContinuousLearner<P, C> {
    pub(super) fn new(policy: P, critic: C, config: ContinuousConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        critic.validate()?;
        critic.validate_training(&config)?;
        policy.action_space().validate()?;
        ensure(
            matches!(
                policy.action_space(),
                ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
            ) && critic.action_dim() == policy.action_space().dimensions(),
            "policy and critic action dimensions differ",
        )?;
        Ok(Self {
            target_policy: frozen_copy(&policy),
            actor: ContinuousActor {
                policy,
                noise: config.exploration_std,
            },
            actor_optimizer: config.actor.build()?,
            critic,
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
            actor_updates: 0,
        })
    }
    pub(super) fn policy(&self) -> P {
        frozen_copy(&self.actor.policy)
    }
    pub(super) fn target_policy(&self) -> P {
        frozen_copy(&self.target_policy)
    }
    pub(super) fn updates(&self) -> u64 {
        self.updates
    }
    #[cfg(feature = "diagnostics")]
    pub(super) fn training_config(&self) -> &ContinuousConfig {
        &self.config
    }
    pub(super) fn actor_updates(&self) -> u64 {
        self.actor_updates
    }

    fn next_actions<'a>(&mut self, batch: &'a ReplayBatch) -> Result<TargetActionBatch<'a>> {
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
        let actions = if indices.is_empty() {
            None
        } else {
            let actions = self
                .target_policy
                .valid()
                .actions(ObservationBatch::from_refs(&observations))?;
            if self.config.target_noise > 0. {
                let space = self.actor.policy.action_space();
                let mut values = to_vec(actions)?;
                smooth_target_actions(
                    &mut values,
                    space,
                    self.config.target_noise,
                    self.config.noise_clip,
                    &mut self.rng,
                );
                Some(matrix(values, indices.len(), space.dimensions()))
            } else {
                Some(actions)
            }
        };
        let next = TargetActionBatch {
            indices,
            observations,
            actions,
        };
        next.validate(batch.items.len(), self.critic.action_dim())?;
        Ok(next)
    }
}
impl<P: DeterministicPolicy, C: CriticRule> Learner for ContinuousLearner<P, C> {
    type Batch = ReplayBatch;
    fn actor(&self) -> &dyn Actor {
        &self.actor
    }
    fn update(&mut self, replay: &ReplayBatch) -> Result<UpdateReport> {
        let batch = PreparedReplayBatch::new(replay, self.actor.policy.action_space())?;
        let next = self.next_actions(replay)?;
        let targets = self.critic.targets(&batch, &next)?;
        let loss = self.critic.update(&batch, targets)?;
        self.updates += 1;
        let mut report = UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("critic_loss", loss);
        if self.updates.is_multiple_of(self.config.policy_delay) {
            let obs = batch.observations();
            let actions = self.actor.policy.actions(obs)?;
            let utility = self.critic.actor_utility(obs, actions)?;
            let loss = optimize(
                &mut self.actor.policy,
                &mut self.actor_optimizer,
                self.config.actor.learning_rate,
                utility.mean().neg(),
            )?;
            self.actor_updates += 1;
            self.target_policy = polyak_update(
                self.target_policy.clone(),
                &self.actor.policy,
                self.config.tau,
            )?;
            self.critic.update_targets(self.config.tau)?;
            report = report.metric("actor_loss", loss);
        }
        Ok(report)
    }
}
impl<P: DeterministicPolicy, C: CriticRule> OffPolicyLearner for ContinuousLearner<P, C> {
    fn set_exploration(&mut self, scale: f32) -> Result<()> {
        ensure(
            scale.is_finite() && scale >= 0.,
            "invalid exploration standard deviation",
        )?;
        self.actor.noise = scale;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct ContinuousMeta<M> {
    config: ContinuousConfig,
    critic: M,
    rng: ChaCha8Rng,
    updates: u64,
    actor_updates: u64,
    noise: f32,
    space: ActionSpace,
}
impl<P: DeterministicPolicy, C: CriticCheckpoint> ContinuousLearner<P, C> {
    pub(super) fn checkpoint(&self, name: &str) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new(
            name,
            ContinuousMeta {
                config: self.config.clone(),
                critic: self.critic.metadata(),
                rng: self.rng.clone(),
                updates: self.updates,
                actor_updates: self.actor_updates,
                noise: self.actor.noise,
                space: self.actor.policy.action_space().clone(),
            },
        )?;
        record.model("policy", &self.actor.policy)?;
        record.model("target_policy", &self.target_policy)?;
        record.optimizer("actor", &self.actor_optimizer)?;
        self.critic.record(&mut record)?;
        Ok(record)
    }
    pub(super) fn restore(&mut self, record: &LearnerRecord, name: &str) -> Result<()> {
        record.check(name)?;
        let meta: ContinuousMeta<C::Metadata> = serde_json::from_value(record.metadata.clone())?;
        meta.config.validate()?;
        ensure(
            &meta.space == self.actor.policy.action_space(),
            "checkpoint action space mismatch",
        )?;
        ensure(
            meta.noise.is_finite()
                && meta.noise >= 0.
                && meta.actor_updates == meta.updates / meta.config.policy_delay,
            "invalid continuous checkpoint counters or exploration",
        )?;
        let policy = record.load_model("policy", &self.actor.policy)?;
        let target_policy = record.load_model("target_policy", &self.target_policy)?;
        let actor_optimizer = record.load_optimizer("actor", meta.config.actor.build()?)?;
        let critic = self
            .critic
            .restored(record, meta.critic, &meta.config.critic)?;
        critic.validate()?;
        critic.validate_training(&meta.config)?;
        ensure(
            critic.action_dim() == policy.action_space().dimensions(),
            "checkpoint critic dimensions differ",
        )?;
        // Every fallible operation finishes before replacing the live learner.
        *self = Self {
            actor: ContinuousActor {
                policy,
                noise: meta.noise,
            },
            target_policy,
            actor_optimizer,
            critic,
            config: meta.config,
            rng: meta.rng,
            updates: meta.updates,
            actor_updates: meta.actor_updates,
        };
        Ok(())
    }
}

macro_rules! delegate_learner {
    ($name:ident $(, $risk:ident)?) => {
        impl<P: crate::nn::policy::DeterministicPolicy, Q: crate::nn::critics::scalar::ActionValue
            $(, $risk: crate::objectives::risk::RiskObjective)?> crate::core::learner::Learner for $name<P, Q $(, $risk)?> {
            type Batch = crate::data::replay::ReplayBatch;
            fn actor(&self) -> &dyn crate::core::actor::Actor { self.inner.actor() }
            fn update(&mut self, batch: &Self::Batch) -> crate::Result<crate::core::learner::UpdateReport> { self.inner.update(batch) }
        }
        impl<P: crate::nn::policy::DeterministicPolicy, Q: crate::nn::critics::scalar::ActionValue
            $(, $risk: crate::objectives::risk::RiskObjective)?> crate::core::learner::OffPolicyLearner for $name<P, Q $(, $risk)?> {
            fn set_exploration(&mut self, scale: f32) -> crate::Result<()> { self.inner.set_exploration(scale) }
        }
        impl<P: crate::nn::policy::DeterministicPolicy, Q: crate::nn::critics::scalar::ActionValue
            $(, $risk: crate::objectives::risk::RiskObjective)?> $name<P, Q $(, $risk)?> {
            /// Independently owned inference snapshot, including running statistics.
            pub fn policy(&self) -> P { self.inner.policy() }
            pub fn target_policy(&self) -> P { self.inner.target_policy() }
            pub fn updates(&self) -> u64 { self.inner.updates() }
            pub fn actor_updates(&self) -> u64 { self.inner.actor_updates() }
        }
    };
}
pub(crate) use delegate_learner;

#[cfg(test)]
mod smoothing_tests {
    use super::*;
    use crate::{
        algorithms::ddpg::ScalarCritic,
        core::{env::Observation, transition::Transition},
        data::replay::ReplayItem,
        nn::{critics::scalar::QValue, policy::Deterministic, vector},
    };
    use burn::module::Param;

    #[test]
    fn nonzero_target_smoothing_is_clipped_and_respects_action_bounds() {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let mut policy = Deterministic::new(
            1,
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            &[],
            &mut rng,
        )
        .unwrap();
        policy.network.layers[0].weight = Param::from_tensor(matrix(vec![0.], 1, 1));
        policy.network.layers[0].bias = Some(Param::from_tensor(vector(&[0.99_f32.atanh()])));
        let config = ContinuousConfig::default();
        let model =
            critic::CriticModel::new(QValue::new(1, 1, 1, &[], &mut rng).unwrap(), &config.critic)
                .unwrap();
        let mut learner =
            ContinuousLearner::new(policy, ScalarCritic { model }, config, 73).unwrap();
        // Exercise the shared smoothing mechanism independently of scalar rule configuration.
        learner.config.target_noise = 100.;
        learner.config.noise_clip = 0.2;
        let batch = ReplayBatch {
            items: (0..65)
                .map(|i| ReplayItem {
                    transition: Transition {
                        observation: Observation::new(vec![1.]),
                        action: Action::Continuous(vec![0.]),
                        reward: 0.0_f32.into(),
                        next_observation: Observation::new(vec![1.]),
                        terminated: i == 0,
                        truncated: false,
                    },
                    discount: if i == 0 { 0. } else { 0.9 },
                })
                .collect(),
        };
        let before = serde_json::to_string(&learner.rng).unwrap();
        let next = learner.next_actions(&batch).unwrap();
        assert_eq!(next.indices, (1..65).collect::<Vec<_>>());
        let actions = to_vec(next.actions.unwrap()).unwrap();
        assert!(
            actions
                .iter()
                .all(|&x| (-1.0..=1.0).contains(&x) && (x - 0.99).abs() <= 0.200001)
        );
        assert!(actions.contains(&1.));
        assert!(actions.iter().any(|&x| (x - 0.79).abs() < 1e-6));
        assert_ne!(serde_json::to_string(&learner.rng).unwrap(), before);
    }
}
