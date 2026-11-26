use super::common::{OnPolicyRecord, OnPolicyState, OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, EpisodeActor, EpisodeBehavior, OnPolicyActorRef, RolloutActor},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationBatch,
        learner::{Learner, OnPolicyLearner, UpdateReport},
    },
    data::rollout::OnPolicyBatch,
    nn::{
        critics::scalar::{StateValue, Value},
        distributions::Distribution,
        policy::{ActorCritic, Policy, StochasticPolicy},
        vector,
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforceConfig {
    pub policy: OptimizerConfig,
    pub value: OptimizerConfig,
    pub baseline: bool,
    pub entropy: f32,
}
impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            policy: Default::default(),
            value: Default::default(),
            baseline: true,
            entropy: 0.,
        }
    }
}
impl ReinforceConfig {
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.value.validate()?;
        ensure(
            self.entropy.is_finite() && self.entropy >= 0.,
            "invalid entropy coefficient",
        )
    }
}

/// Optional finite-action exploration for REINFORCE.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalExploration {
    pub epsilon: f32,
    /// If supplied, apply the mixture only at this exact dense observation.
    /// For example, an episodic environment can give its root a unique marker.
    pub only_at: Option<Vec<f32>>,
}

impl CategoricalExploration {
    fn validate(&self, space: &ActionSpace) -> Result<()> {
        ensure(
            matches!(space, ActionSpace::Discrete { .. })
                && (0. ..=1.).contains(&self.epsilon)
                && self.only_at.as_ref().is_none_or(|observation| {
                    !observation.is_empty() && observation.iter().all(|v| v.is_finite())
                }),
            "invalid finite-action REINFORCE exploration configuration",
        )
    }

    fn distribution(
        &self,
        obs: ObservationBatch<'_>,
        policy: Distribution,
    ) -> Result<Distribution> {
        let epsilon = if let Some(root) = &self.only_at {
            ensure(
                obs.iter()
                    .all(|state| state.dense_values().len() == root.len()),
                "exploration observation width differs from policy inputs",
            )?;
            obs.iter()
                .map(|state| {
                    if state.dense_values() == root {
                        self.epsilon
                    } else {
                        0.
                    }
                })
                .collect()
        } else {
            vec![self.epsilon; obs.len()]
        };
        policy.with_uniform_mixture(&epsilon)
    }
}

pub struct Reinforce<P: StochasticPolicy = Policy, V: StateValue = Value> {
    pub(crate) state: OnPolicyState<P, V>,
    pub(crate) config: ReinforceConfig,
    categorical_exploration: Option<CategoricalExploration>,
}
impl<P: StochasticPolicy, V: StateValue> Reinforce<P, V> {
    pub fn new(models: ActorCritic<P, V>, config: ReinforceConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: OnPolicyState::new(models, &config.policy, &config.value, seed)?,
            config,
            categorical_exploration: None,
        })
    }
    /// Train the objective under the actual mixture used for episodic collection.
    /// Actor entropy, when enabled, continues to regularize the underlying policy.
    pub fn with_categorical_exploration(
        mut self,
        exploration: CategoricalExploration,
    ) -> Result<Self> {
        exploration.validate(self.action_space())?;
        self.categorical_exploration = Some(exploration);
        Ok(self)
    }
    pub fn categorical_exploration(&self) -> Option<&CategoricalExploration> {
        self.categorical_exploration.as_ref()
    }
    pub fn config(&self) -> &ReinforceConfig {
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
}
impl<P: StochasticPolicy, V: StateValue> Actor for Reinforce<P, V> {
    fn action_space(&self) -> &ActionSpace {
        self.state.models.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        if let Some(exploration) = &self.categorical_exploration
            && mode == ActMode::Sample
        {
            let policy = self.state.models.policy.valid().distribution(obs)?;
            return exploration
                .distribution(obs, policy)?
                .sample_actions_into(mode, rng, actions);
        }
        self.state.models.act_into(obs, mode, rng, actions)
    }
}
impl<P: StochasticPolicy, V: StateValue> EpisodeActor for Reinforce<P, V> {
    fn sample_episode_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut EpisodeBehavior,
    ) -> Result<()> {
        let policy = self.state.models.policy.valid().distribution(obs)?;
        let behavior = if let Some(exploration) = &self.categorical_exploration {
            exploration.distribution(obs, policy)?
        } else {
            policy
        };
        behavior.sample_episode_into(ActMode::Sample, rng, out)?;
        if self.config.baseline {
            self.state
                .models
                .values_into(obs, out.baseline_values.get_or_insert_with(Vec::new))?;
        } else {
            out.baseline_values = None;
        }
        Ok(())
    }
}
impl<P: StochasticPolicy, V: StateValue> OnPolicyLearner for Reinforce<P, V> {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Episodic(self)
    }
}
impl<P: StochasticPolicy, V: StateValue> Checkpoint for Reinforce<P, V> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record("reinforce", &self.config)?;
        if let Some(exploration) = &self.categorical_exploration {
            record.metadata["categorical_exploration"] = serde_json::to_value(exploration)?;
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("reinforce")?;
        let metadata: OnPolicyRecord<ReinforceConfig> =
            serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let categorical_exploration: Option<CategoricalExploration> = record
            .metadata
            .get("categorical_exploration")
            .cloned()
            .map(serde_json::from_value)
            .transpose()?;
        if let Some(exploration) = &categorical_exploration {
            exploration.validate(&metadata.space)?;
        }
        let state = self.state.restored(
            record,
            &metadata,
            &metadata.config.policy,
            &metadata.config.value,
        )?;
        *self = Self {
            state,
            config: metadata.config,
            categorical_exploration,
        };
        Ok(())
    }
}
impl<P: StochasticPolicy, V: StateValue> Learner for Reinforce<P, V> {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        self
    }
    /// Averages sum_t gamma^t G_t log pi(a_t|s_t) over episodes, with detached targets.
    /// No repeated epochs over a stale behavior policy.
    fn update(&mut self, b: &OnPolicyBatch) -> Result<UpdateReport> {
        b.validate()?;
        ensure(
            b.complete_episodes() && b.episode_count > 0,
            "REINFORCE requires complete terminal episodes",
        )?;
        let baseline = b.baseline_values()?;
        ensure(
            !self.config.baseline || baseline.is_some(),
            "REINFORCE baseline values are required",
        )?;
        let obs = (&b.observations).into();
        let dist = self.state.models.policy.distribution(obs)?;
        let log_prob = if let Some(exploration) = &self.categorical_exploration {
            exploration
                .distribution(obs, dist.clone())?
                .log_prob(&b.actions, &b.latents)?
        } else {
            dist.log_prob(&b.actions, &b.latents)?
        };
        let advantage = if self.config.baseline {
            vector(&b.returns) - vector(baseline.expect("validated baseline"))
        } else {
            vector(&b.returns)
        };
        let mut loss = (log_prob * advantage * vector(&b.discount_weights))
            .sum()
            .neg()
            / b.episode_count as f32;
        if self.config.entropy > 0. {
            loss = loss - dist.entropy(&mut self.state.rng)?.mean() * self.config.entropy;
        }
        let p = optimize(
            &mut self.state.models.policy,
            &mut self.state.policy_optimizer,
            self.config.policy.learning_rate,
            loss,
        )?;
        let mut report = UpdateReport::default().metric("policy_loss", p);
        if self.config.baseline {
            report = report.metric("value_loss", self.state.fit_value(b, &self.config.value)?);
        }
        self.state.updates += 1;
        report.updates = 1;
        Ok(report)
    }
}
