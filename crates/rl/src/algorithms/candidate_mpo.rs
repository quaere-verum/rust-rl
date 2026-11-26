//! MPO over complete semantic actions, using the shared candidate critic and temporal runtime.
use crate::nn::optimization::validate_disjoint;
mod actor;
pub(super) mod checkpoint;
pub(super) mod projection;

use super::{
    OptimizerConfig,
    candidate_ddqn::{CandidateBackup, CandidateCritic, CandidateDdqnConfig},
};
use crate::{
    Result,
    core::{
        action::ActionSpace,
        ensure,
        env::ObservationSpace,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner, TemporalOffPolicyLearner},
        temporal::ReturnSpec,
    },
    data::temporal::SequenceBatch,
    nn::{
        optimization::{parameter_norm, validate_trainable},
        structured::{
            CandidatePolicy, StructuredActionValue, StructuredPolicy, StructuredQ,
            candidate_probabilities_checked,
        },
        target::frozen_copy,
    },
    objectives::mpo::{EStep, e_step},
};
pub(super) use actor::conditional_decision;
pub use actor::{CandidateMpoActor, CandidateMpoBehavior, CandidateMpoSchedule};
use burn::optim::ModuleOptimizer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateMpoBackup {
    /// Expectation under the current actor before accepting the proposed update.
    PolicyEvaluation,
    /// Select with the online critic and evaluate with the target critic.
    DoubleQOptimality,
    /// Maximize the target critic over the complete enabled candidate menu.
    MaximumOptimality,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionRecovery {
    RollbackOnly,
    RestartAdam,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateMpoConfig {
    pub backup: CandidateMpoBackup,
    pub optimizer: OptimizerConfig,
    pub e_step_kl: f64,
    pub projection_kl: f64,
    /// Additional halvings after the full Adam proposal.
    pub max_backtracks: usize,
    pub recovery: ProjectionRecovery,
}

impl Default for CandidateMpoConfig {
    fn default() -> Self {
        Self {
            backup: CandidateMpoBackup::PolicyEvaluation,
            optimizer: Default::default(),
            e_step_kl: 0.1,
            projection_kl: 0.01,
            max_backtracks: 10,
            recovery: ProjectionRecovery::RollbackOnly,
        }
    }
}

impl CandidateMpoConfig {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        ensure(
            self.e_step_kl.is_finite()
                && self.e_step_kl >= 0.
                && self.projection_kl.is_finite()
                && self.projection_kl >= 0.
                && self.max_backtracks <= 20,
            "invalid candidate MPO configuration",
        )
    }
}

pub struct CandidateMpo<
    Q: StructuredActionValue = StructuredQ,
    P: CandidatePolicy = StructuredPolicy,
> {
    policy: P,
    critic: CandidateCritic<Q>,
    optimizer: ModuleOptimizer,
    config: CandidateMpoConfig,
    actor_updates: u64,
    adam_updates: u64,
}

impl<Q: StructuredActionValue, P: CandidatePolicy> CandidateMpo<Q, P> {
    pub fn new(
        policy: P,
        online: Q,
        critic_config: CandidateDdqnConfig,
        config: CandidateMpoConfig,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    ) -> Result<Self> {
        config.validate()?;
        P::validate_configuration(policy.configuration())?;
        policy
            .input_schema()
            .validate_spaces(&observation_space, &action_space)?;
        validate_trainable(&policy)?;
        parameter_norm(&policy)?;
        validate_disjoint(&policy, &online)?;
        let critic = CandidateCritic::new(online, critic_config, observation_space, action_space)?;
        let optimizer = config.optimizer.build()?;
        Ok(Self {
            policy,
            critic,
            optimizer,
            config,
            actor_updates: 0,
            adam_updates: 0,
        })
    }

    pub fn config(&self) -> &CandidateMpoConfig {
        &self.config
    }

    pub fn updates(&self) -> u64 {
        self.critic.updates
    }

    pub fn actor_updates(&self) -> u64 {
        self.actor_updates
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_critic_config(&self) -> &CandidateDdqnConfig {
        &self.critic.config
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_q_actor(
        &self,
        settings: super::candidate_ddqn::CandidateExploration,
    ) -> Result<super::candidate_ddqn::CandidateQActor<Q>> {
        self.critic.publish(settings, self.updates())
    }

    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        ensure(
            self.actor_updates < i32::MAX as u64 - 1,
            "candidate MPO actor counter exhausted",
        )?;
        let batch = self.critic.prepare(batch)?;
        let limits = self.critic.config.pack_limits();
        let probabilities;
        let backup = match self.config.backup {
            CandidateMpoBackup::PolicyEvaluation => {
                probabilities = if batch.next.is_empty() {
                    Vec::new()
                } else {
                    candidate_probabilities_checked(&self.policy.valid(), &batch.next, limits)?
                };
                CandidateBackup::PolicyEvaluation(&probabilities)
            }
            CandidateMpoBackup::DoubleQOptimality => CandidateBackup::DoubleQOptimality,
            CandidateMpoBackup::MaximumOptimality => CandidateBackup::MaximumOptimality,
        };
        let old = candidate_probabilities_checked(&self.policy, &batch.current, limits)?;
        let values = self
            .critic
            .online_values(&batch.current, &batch.current_graphs)?;
        let enabled: Vec<Vec<_>> = batch
            .current
            .iter()
            .map(|sample| {
                sample
                    .candidates
                    .iter()
                    .map(|candidate| candidate.enabled)
                    .collect()
            })
            .collect();
        let (e, width) = improvement(
            &enabled,
            &old,
            &values,
            self.config.e_step_kl,
            limits.max_elements,
        )?;
        let improved: Vec<_> = e
            .probabilities
            .chunks_exact(width)
            .zip(&batch.current)
            .map(|(row, sample)| &row[..sample.candidates.len()])
            .collect();

        let proposal = projection::project(self, &batch.current, &old, &improved)?;
        let mut report = self.critic.fit(batch, backup)?;
        if let Some((policy, optimizer, restarted)) = proposal.accepted {
            self.policy = policy;
            self.optimizer = optimizer;
            self.actor_updates += 1;
            self.adam_updates = if restarted { 1 } else { self.adam_updates + 1 };
        }
        report.metrics.extend(proposal.metrics);
        let report = report
            .metric("actor_updates", self.actor_updates as f64)
            .metric("e_step_reverse_kl", e.reverse_kl)
            .metric("e_step_dual", e.dual)
            .metric("e_step_active", f64::from(e.active));
        Ok(match e.temperature {
            Some(temperature) => report.metric("e_step_temperature", temperature),
            None => report,
        })
    }
}

pub(super) fn improvement(
    enabled: &[Vec<bool>],
    old: &[Vec<f64>],
    q: &[Vec<f32>],
    epsilon: f64,
    max_elements: usize,
) -> Result<(EStep, usize)> {
    let width = enabled
        .iter()
        .map(Vec::len)
        .max()
        .ok_or_else(|| crate::Error::Shape("empty candidate MPO batch".into()))?;
    ensure(
        enabled.len() == old.len() && enabled.len() == q.len(),
        "candidate MPO teacher row count differs",
    )?;
    let elements = enabled
        .len()
        .checked_mul(width)
        .filter(|&n| n <= max_elements)
        .ok_or_else(|| crate::Error::Config("candidate MPO padding budget exceeded".into()))?;
    let mut padded_prior = vec![0.; elements];
    let mut padded_q = vec![0.; elements];
    let mut mask = vec![false; elements];
    for (i, ((enabled, p), values)) in enabled.iter().zip(old).zip(q).enumerate() {
        ensure(
            enabled.len() == p.len() && enabled.len() == values.len(),
            "candidate MPO teacher candidate count differs",
        )?;
        let range = i * width..i * width + enabled.len();
        padded_prior[range.clone()].copy_from_slice(p);
        for (output, &value) in padded_q[range.clone()].iter_mut().zip(values) {
            *output = f64::from(value);
        }
        mask[range].copy_from_slice(enabled);
    }
    Ok((
        e_step(&padded_prior, &padded_q, &mask, width, epsilon)?,
        width,
    ))
}

impl<Q: StructuredActionValue, P: CandidatePolicy> StatefulLearner for CandidateMpo<Q, P> {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        CandidateMpo::update(self, batch)
    }
}

impl<Q: StructuredActionValue, P: CandidatePolicy> TemporalOffPolicyLearner for CandidateMpo<Q, P> {
    fn return_contract(&self) -> ReturnSpec {
        self.critic.config.returns()
    }
    fn requires_replay_history(&self) -> bool {
        false
    }

    fn replay_spaces(&self) -> Option<(&ObservationSpace, &ActionSpace)> {
        Some((&self.critic.observation_space, &self.critic.action_space))
    }
}

impl<Q: StructuredActionValue, P: CandidatePolicy> BehaviorPublisher for CandidateMpo<Q, P> {
    type Actor = CandidateMpoActor<P>;
    type Settings = CandidateMpoBehavior;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        ensure(
            !self.critic.poisoned,
            "candidate MPO requires checkpoint recovery",
        )?;
        CandidateMpoActor::new(
            frozen_copy(&self.policy).valid(),
            self.critic.observation_space.clone(),
            self.critic.action_space.clone(),
            *settings,
            revision,
            self.critic.config.pack_limits(),
        )
    }
}
