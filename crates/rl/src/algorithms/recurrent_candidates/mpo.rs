use super::{
    RecurrentCandidateActor, RecurrentCandidateModel, RecurrentGradientMode,
    critic::{self, RecurrentBackup, RecurrentBatch, decision_values},
};
use crate::nn::optimization::validate_disjoint;
use crate::{
    Result,
    algorithms::{
        candidate_ddqn::{CandidateCritic, CandidateDdqnConfig},
        candidate_mpo::{
            CandidateMpoBackup, CandidateMpoBehavior, CandidateMpoConfig,
            checkpoint::{checkpoint_actor, restore_actor},
            conditional_decision, improvement,
            projection::{ProjectionBatch, project_model},
        },
    },
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationSpace,
        learner::UpdateReport,
        stateful::{
            BehaviorPublisher, HistoryInput, StatefulDecision, StatefulLearner,
            TemporalOffPolicyLearner,
        },
        structured::StructuredObservation,
        temporal::ReturnSpec,
    },
    data::temporal::SequenceBatch,
    nn::{
        optimization::{parameter_norm, validate_trainable},
        structured::{
            PackLimits, RecurrentLightweightConfig, RecurrentLightweightPolicy,
            RecurrentLightweightQ, RecurrentLightweightState,
        },
        target::frozen_copy,
        to_vec,
    },
};
use burn::{module::Module, optim::ModuleOptimizer, tensor::Device};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecurrentMpoGradients {
    pub critic: RecurrentGradientMode,
    pub actor: RecurrentGradientMode,
}

pub struct RecurrentCandidateMpo {
    policy: RecurrentLightweightPolicy,
    critic: CandidateCritic<RecurrentLightweightQ>,
    optimizer: ModuleOptimizer,
    config: CandidateMpoConfig,
    gradients: RecurrentMpoGradients,
    actor_updates: u64,
    adam_updates: u64,
}

impl RecurrentCandidateMpo {
    pub fn new(
        policy: RecurrentLightweightPolicy,
        online: RecurrentLightweightQ,
        critic_config: CandidateDdqnConfig,
        config: CandidateMpoConfig,
        gradients: RecurrentMpoGradients,
        observations: ObservationSpace,
        actions: ActionSpace,
    ) -> Result<Self> {
        config.validate()?;
        policy.config().validate()?;
        policy
            .config()
            .model
            .input
            .validate_spaces(&observations, &actions)?;
        validate_trainable(&policy)?;
        parameter_norm(&policy)?;
        validate_disjoint(&policy, &online)?;
        let critic = critic::new_critic(online, critic_config, observations, actions)?;
        let optimizer = config.optimizer.build()?;
        Ok(Self {
            policy,
            critic,
            optimizer,
            config,
            gradients,
            actor_updates: 0,
            adam_updates: 0,
        })
    }

    pub fn online(&self) -> &RecurrentLightweightQ {
        &self.critic.online
    }
    pub fn policy(&self) -> &RecurrentLightweightPolicy {
        &self.policy
    }
    pub fn updates(&self) -> u64 {
        self.critic.updates
    }
    pub fn actor_updates(&self) -> u64 {
        self.actor_updates
    }

    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        ensure(
            self.actor_updates < i32::MAX as u64 - 1,
            "recurrent MPO actor counter exhausted",
        )?;
        let replay = RecurrentBatch::new(batch, &self.critic)?;
        let limits = self.critic.config.pack_limits();
        let actor_inputs = replay.inputs(self.gradients.actor);
        let current_probabilities = |policy: &RecurrentLightweightPolicy| -> Result<Vec<Vec<f64>>> {
            let frozen = policy.valid();
            let values = frozen.evaluate_histories(&actor_inputs, limits)?;
            replay
                .positions
                .iter()
                .map(|p| values[p.history].complete_probabilities_f64(p.step))
                .collect()
        };
        let online = self.critic.online_history_values(&replay)?;
        let (old, continuation) = {
            let frozen = self.policy.valid();
            let rows = frozen.evaluate_histories(&actor_inputs, limits)?;
            let old = replay
                .positions
                .iter()
                .map(|p| rows[p.history].complete_probabilities_f64(p.step))
                .collect::<Result<Vec<_>>>()?;
            let continuation = if self.config.backup == CandidateMpoBackup::PolicyEvaluation {
                replay
                    .positions
                    .iter()
                    .map(|position| {
                        if position.target.discount == 0. {
                            return Ok(Vec::new());
                        }
                        rows[position.history]
                            .complete_probabilities_f64(position.target.bootstrap_index)
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            (old, continuation)
        };
        let mut values = Vec::with_capacity(replay.positions.len());
        let mut enabled = Vec::with_capacity(replay.positions.len());
        for position in &replay.positions {
            values.push(to_vec(decision_values(
                &online,
                position.history,
                position.step,
                position.candidates.len(),
            )?)?);
            enabled.push(
                position
                    .candidates
                    .iter()
                    .map(|candidate| candidate.enabled)
                    .collect(),
            );
        }
        let (teacher, width) = improvement(
            &enabled,
            &old,
            &values,
            self.config.e_step_kl,
            limits.max_elements,
        )?;
        let positions = replay.positions.len() as f64;
        let expected_q = values
            .iter()
            .zip(&old)
            .map(|(q, p)| {
                q.iter()
                    .zip(p)
                    .map(|(&q, &p)| f64::from(q) * p)
                    .sum::<f64>()
            })
            .sum::<f64>()
            / positions;
        let greedy_q = values
            .iter()
            .zip(&enabled)
            .map(|(q, mask)| {
                q.iter()
                    .zip(mask)
                    .filter_map(|(&q, &enabled)| enabled.then_some(f64::from(q)))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .sum::<f64>()
            / positions;
        let improved: Vec<_> = teacher
            .probabilities
            .chunks_exact(width)
            .zip(&enabled)
            .map(|(row, mask)| &row[..mask.len()])
            .collect();
        let operations: Vec<_> = replay
            .positions
            .iter()
            .map(|position| {
                position
                    .candidates
                    .iter()
                    .map(|candidate| candidate.action.operation)
                    .collect()
            })
            .collect();
        let keys: Vec<Vec<_>> = replay
            .positions
            .iter()
            .map(|position| {
                position
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.enabled)
                    .map(|candidate| candidate.action)
                    .collect()
            })
            .collect();
        let weights: Vec<Vec<_>> = improved
            .iter()
            .zip(&enabled)
            .map(|(row, enabled)| {
                row.iter()
                    .zip(enabled)
                    .filter(|(_, enabled)| **enabled)
                    .map(|(&p, _)| p as f32)
                    .collect()
            })
            .collect();
        let proposal = project_model(
            &self.policy,
            &self.optimizer,
            &self.config,
            ProjectionBatch {
                operations: &operations,
                old: &old,
                teacher: &improved,
            },
            |policy| {
                let policies = policy.evaluate_histories(&actor_inputs, limits)?;
                let likelihoods = replay
                    .positions
                    .iter()
                    .zip(&keys)
                    .map(|(position, keys)| {
                        policies[position.history].log_prob_many(position.step, keys)
                    })
                    .collect::<Result<Vec<_>>>()?;
                crate::objectives::policy_supervision::distribution_loss(likelihoods, &weights)
            },
            current_probabilities,
        )?;
        let backup = match self.config.backup {
            CandidateMpoBackup::DoubleQOptimality => RecurrentBackup::DoubleQ(&online),
            CandidateMpoBackup::PolicyEvaluation => RecurrentBackup::Policy(&continuation),
            CandidateMpoBackup::MaximumOptimality => RecurrentBackup::Maximum,
        };
        let mut report = self
            .critic
            .fit_recurrent(&replay, backup, self.gradients.critic)?;
        if let Some((policy, optimizer, restarted)) = proposal.accepted {
            self.policy = policy;
            self.optimizer = optimizer;
            self.actor_updates += 1;
            self.adam_updates = if restarted { 1 } else { self.adam_updates + 1 };
        }
        report.metrics.extend(proposal.metrics);
        report = report
            .metric("actor_updates", self.actor_updates as f64)
            .metric("actor_positions", positions)
            .metric("actor_expected_q", expected_q)
            .metric("actor_greedy_q", greedy_q)
            .metric("actor_q_gap", greedy_q - expected_q)
            .metric("e_step_reverse_kl", teacher.reverse_kl)
            .metric("e_step_dual", teacher.dual)
            .metric("e_step_active", f64::from(teacher.active));
        if let Some(temperature) = teacher.temperature {
            report = report.metric("e_step_temperature", temperature);
        }
        Ok(report)
    }
}

impl StatefulLearner for RecurrentCandidateMpo {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        RecurrentCandidateMpo::update(self, batch)
    }
}

impl TemporalOffPolicyLearner for RecurrentCandidateMpo {
    fn return_contract(&self) -> ReturnSpec {
        self.critic.config.returns()
    }

    fn replay_spaces(&self) -> Option<(&ObservationSpace, &ActionSpace)> {
        Some((&self.critic.observation_space, &self.critic.action_space))
    }
}

pub type RecurrentCandidatePolicyActor =
    RecurrentCandidateActor<RecurrentLightweightPolicy, CandidateMpoBehavior>;

impl RecurrentCandidateModel for RecurrentLightweightPolicy {
    type Settings = CandidateMpoBehavior;
    fn config(&self) -> &RecurrentLightweightConfig {
        self.config()
    }
    fn device(&self) -> Device {
        self.model_device()
    }
    fn build(config: RecurrentLightweightConfig, rng: &mut ChaCha8Rng) -> Result<Self> {
        Self::new(config, rng, &Device::flex())
    }
    fn validate_settings(settings: Self::Settings) -> Result<()> {
        settings.validate()
    }
    fn advance(
        &self,
        state: &RecurrentLightweightState,
        input: HistoryInput<'_>,
        limits: PackLimits,
    ) -> Result<RecurrentLightweightState> {
        self.assimilate(state, input, limits)
    }
    fn select(
        &self,
        state: &RecurrentLightweightState,
        observation: &StructuredObservation,
        settings: Self::Settings,
        limits: PackLimits,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let evaluation = self.evaluate_decision(state, observation, limits)?;
        if mode == ActMode::Deterministic {
            return Ok(StatefulDecision {
                action: Action::Conditional(Arc::new(evaluation.greedy(0)?.key)),
                log_prob: Some(0.),
            });
        }
        let distribution = evaluation.complete_probabilities_f64(0)?;
        conditional_decision(
            &observation.candidates,
            self.config().model.input.operations.len(),
            distribution,
            settings,
            rng,
        )
    }
}

impl BehaviorPublisher for RecurrentCandidateMpo {
    type Actor = RecurrentCandidatePolicyActor;
    type Settings = CandidateMpoBehavior;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        ensure(
            !self.critic.poisoned,
            "recurrent MPO requires checkpoint recovery",
        )?;
        RecurrentCandidateActor::new(
            frozen_copy(&self.policy).valid(),
            self.critic.observation_space.clone(),
            self.critic.action_space.clone(),
            *settings,
            revision,
            self.critic.config.pack_limits(),
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    version: u32,
    gradients: RecurrentMpoGradients,
    mpo: serde_json::Value,
}

impl Checkpoint for RecurrentCandidateMpo {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = checkpoint_actor(
            self.critic.recurrent_record()?,
            &self.policy,
            self.policy.config(),
            &self.optimizer,
            &self.config,
            self.actor_updates,
            self.adam_updates,
        )?;
        record.algorithm = "recurrent-candidate-mpo-v1".into();
        record.metadata = serde_json::to_value(Meta {
            version: 1,
            gradients: self.gradients,
            mpo: record.metadata,
        })?;
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-candidate-mpo-v1")?;
        self.critic.validate_recurrent_record(record)?;
        let meta: Meta = serde_json::from_value(record.metadata.clone())?;
        ensure(
            meta.version == 1 && meta.gradients == self.gradients,
            "recurrent MPO gradient contract differs",
        )?;
        let mut record = record.clone();
        record.algorithm = "candidate-mpo-v1".into();
        record.metadata = meta.mpo;
        let actor = restore_actor(&record, &self.policy, self.policy.config(), &self.config)?;
        record.algorithm = "structured-ddqn-v1".into();
        record.metadata = actor.critic;
        let critic = self
            .critic
            .restored_model(&record, self.critic.online.config())?;
        ensure(
            actor.updates <= critic.updates,
            "recurrent MPO actor clock exceeds critic clock",
        )?;
        validate_disjoint(&actor.policy, &critic.online)?;
        self.policy = actor.policy;
        self.optimizer = actor.optimizer;
        self.critic = critic;
        self.actor_updates = actor.updates;
        self.adam_updates = actor.adam_updates;
        Ok(())
    }
}
