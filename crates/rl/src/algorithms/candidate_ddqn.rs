//! Feedforward Double DQN for typed, variable candidate sets.
mod actor;
mod checkpoint;

#[cfg(test)]
mod tests;

use super::OptimizerConfig;
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        ensure,
        env::{Observation, ObservationSpace},
        graph::GraphObservation,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner, TemporalOffPolicyLearner},
        structured::StructuredObservation,
        temporal::ReturnSpec,
    },
    data::temporal::{NStepTarget, SequenceBatch},
    nn::{
        optimization::{optimize_validated, parameter_norm, validate_trainable},
        structured::{
            InputSchema, PackLimits, StructuredActionValue, StructuredQ, StructuredSample,
            evaluate_checked,
        },
        target::{frozen_copy, polyak_update},
        to_vec,
    },
};
pub(super) use actor::epsilon_decision;
pub use actor::{
    CandidateEpsilonSchedule, CandidateExploration, CandidateQActor, ExplorationDistribution,
};
use burn::{module::Module, optim::ModuleOptimizer, tensor::Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BellmanLoss {
    Squared,
    Huber { delta: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateDdqnConfig {
    pub optimizer: OptimizerConfig,
    pub gamma: f32,
    pub loss: BellmanLoss,
    pub target_tau: f32,
    pub target_interval: u64,
    pub max_pack_elements: usize,
}

impl Default for CandidateDdqnConfig {
    fn default() -> Self {
        Self {
            optimizer: Default::default(),
            gamma: 0.99,
            loss: BellmanLoss::Huber { delta: 1. },
            target_tau: 0.01,
            target_interval: 1,
            max_pack_elements: 16 * 1024 * 1024,
        }
    }
}

impl CandidateDdqnConfig {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        self.returns().validate()?;
        ensure(
            self.target_tau.is_finite()
                && (0. ..=1.).contains(&self.target_tau)
                && self.target_interval > 0
                && self.max_pack_elements > 0,
            "invalid candidate DDQN configuration",
        )?;
        if let BellmanLoss::Huber { delta } = self.loss {
            ensure(delta.is_finite() && delta > 0., "invalid Huber delta")?;
        }
        Ok(())
    }

    pub fn returns(&self) -> ReturnSpec {
        ReturnSpec {
            gamma: self.gamma,
            n_step: 1,
        }
    }

    pub(super) fn pack_limits(&self) -> PackLimits {
        PackLimits {
            max_elements: self.max_pack_elements,
        }
    }
}

pub struct CandidateDdqn<Q: StructuredActionValue = StructuredQ> {
    critic: CandidateCritic<Q>,
}

impl<Q: StructuredActionValue> CandidateDdqn<Q> {
    pub fn new(
        online: Q,
        config: CandidateDdqnConfig,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    ) -> Result<Self> {
        Ok(Self {
            critic: CandidateCritic::new(online, config, observation_space, action_space)?,
        })
    }

    pub fn config(&self) -> &CandidateDdqnConfig {
        &self.critic.config
    }

    pub fn updates(&self) -> u64 {
        self.critic.updates
    }

    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        let batch = self.critic.prepare(batch)?;
        self.critic.fit(batch, CandidateBackup::DoubleQOptimality)
    }
}

pub(super) struct CandidateCritic<Q> {
    pub(super) online: Q,
    pub(super) target: Q,
    optimizer: ModuleOptimizer,
    pub(super) config: CandidateDdqnConfig,
    pub(super) observation_space: ObservationSpace,
    pub(super) action_space: ActionSpace,
    pub(super) updates: u64,
    pub(super) poisoned: bool,
}

pub(super) struct CandidateBatch<'a> {
    pub(super) current: Vec<StructuredSample<ConditionalAction>>,
    pub(super) next: Vec<StructuredSample<ConditionalAction>>,
    pub(super) current_graphs: Vec<&'a GraphObservation>,
    next_graphs: Vec<&'a GraphObservation>,
    actions: Vec<&'a ConditionalAction>,
    returns: Vec<NStepTarget>,
    bootstrap: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(super) enum CandidateBackup<'a> {
    DoubleQOptimality,
    MaximumOptimality,
    PolicyEvaluation(&'a [Vec<f64>]),
}

impl<Q: StructuredActionValue> CandidateCritic<Q> {
    pub(super) fn new(
        online: Q,
        config: CandidateDdqnConfig,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    ) -> Result<Self> {
        Q::validate_configuration(online.configuration())?;
        online.validate_spaces(&observation_space, &action_space)?;
        Self::new_with_checked_spaces(online, config, observation_space, action_space)
    }

    pub(super) fn prepare<'a>(&self, batch: &'a SequenceBatch) -> Result<CandidateBatch<'a>> {
        self.check_update()?;
        batch.validate_in(&self.observation_space, &self.action_space)?;
        ensure(
            batch.config.learn_len == 1 && batch.config.n_step == 1,
            "candidate DDQN requires one learning position and one decision per return",
        )?;

        let mut actions = Vec::with_capacity(batch.samples.len());
        let mut current = Vec::with_capacity(batch.samples.len());
        let mut returns = Vec::with_capacity(batch.samples.len());
        let mut next = Vec::new();
        let mut current_graphs = Vec::new();
        let mut next_graphs = Vec::new();
        let mut bootstrap = Vec::new();
        for sample in &batch.samples {
            let position = sample.learn_start;
            let observation = &sample.observations[position];
            let action = &sample.steps[position].action;
            let Action::Conditional(key) = action else {
                return Err(crate::Error::Action(
                    "candidate DDQN requires a conditional action".into(),
                ));
            };
            actions.push(key.as_ref());
            current.push(model_sample(observation, &self.online)?);
            current_graphs.extend(observation.as_graph());
            let returns_at_position = sample.target_at(position, self.config.gamma, 1)?;
            if returns_at_position.discount > 0. {
                bootstrap.push(returns.len());
                let continuation = &sample.observations[returns_at_position.bootstrap_index];
                next.push(model_sample(continuation, &self.online)?);
                next_graphs.extend(continuation.as_graph());
            }
            returns.push(returns_at_position);
        }

        Ok(CandidateBatch {
            current,
            next,
            current_graphs,
            next_graphs,
            actions,
            returns,
            bootstrap,
        })
    }

    pub(super) fn online_values(
        &self,
        samples: &[StructuredSample<ConditionalAction>],
        graphs: &[&GraphObservation],
    ) -> Result<Vec<Vec<f32>>> {
        evaluate_checked(
            &self.online.valid(),
            samples,
            graphs,
            self.config.pack_limits(),
        )?
        .into_iter()
        .map(|row| {
            let values = to_vec(row.values)?;
            ensure(
                values.iter().all(|value| value.is_finite()),
                "nonfinite candidate Q values",
            )?;
            Ok(values)
        })
        .collect()
    }

    pub(super) fn fit(
        &mut self,
        batch: CandidateBatch<'_>,
        backup: CandidateBackup<'_>,
    ) -> Result<UpdateReport> {
        let CandidateBatch {
            current,
            next,
            current_graphs,
            next_graphs,
            actions,
            returns,
            bootstrap,
        } = batch;
        let limits = self.config.pack_limits();
        let mut targets: Vec<_> = returns.iter().map(|r| r.reward).collect();
        if let CandidateBackup::PolicyEvaluation(probabilities) = &backup {
            ensure(
                probabilities.len() == next.len(),
                "candidate policy backup row count differs",
            )?;
        }
        if !next.is_empty() {
            let online = match backup {
                CandidateBackup::DoubleQOptimality => {
                    Some(self.online_values(&next, &next_graphs)?)
                }
                CandidateBackup::MaximumOptimality | CandidateBackup::PolicyEvaluation(_) => None,
            };
            let target = evaluate_checked(&self.target, &next, &next_graphs, limits)?;
            for (row, ((index, sample), target)) in
                bootstrap.into_iter().zip(&next).zip(target).enumerate()
            {
                let NStepTarget {
                    reward, discount, ..
                } = returns[index];
                // Each evaluator binds its result to the same semantic candidate order.
                let target = to_vec(target.values)?;
                targets[index] = match backup {
                    CandidateBackup::DoubleQOptimality | CandidateBackup::MaximumOptimality => {
                        let enabled: Vec<_> = sample
                            .candidates
                            .iter()
                            .map(|candidate| candidate.enabled)
                            .collect();
                        if let Some(online) = &online {
                            double_q_target(reward, discount, &online[row], &target, &enabled)?
                        } else {
                            maximum_target(reward, discount, &target, &enabled)?
                        }
                    }
                    CandidateBackup::PolicyEvaluation(probabilities) => {
                        let probabilities = &probabilities[row];
                        ensure(
                            probabilities.len() == sample.candidates.len()
                                && sample
                                    .candidates
                                    .iter()
                                    .zip(probabilities)
                                    .all(|(candidate, &p)| candidate.enabled || p == 0.),
                            "candidate policy backup support differs",
                        )?;
                        policy_target(reward, discount, &target, probabilities)?
                    }
                };
            }
        }

        let predictions = evaluate_checked(&self.online, &current, &current_graphs, limits)?
            .into_iter()
            .zip(actions)
            .map(|(row, action)| {
                ensure(
                    to_vec(row.values.clone())?
                        .iter()
                        .all(|value| value.is_finite()),
                    "nonfinite candidate Q value",
                )?;
                let position = row
                    .keys
                    .iter()
                    .position(|key| *key == action)
                    .ok_or_else(|| {
                        crate::Error::Action(
                            "recorded candidate key is missing from current values".into(),
                        )
                    })?;
                Ok(row.values.narrow(0, position, 1))
            })
            .collect::<Result<Vec<_>>>()?;
        let candidates = current.iter().map(|sample| sample.candidates.len()).sum();
        self.fit_predictions(Tensor::cat(predictions, 0), &targets, candidates)
    }
}

impl<Q: Module> CandidateCritic<Q> {
    pub(super) fn new_with_checked_spaces(
        online: Q,
        config: CandidateDdqnConfig,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    ) -> Result<Self> {
        config.validate()?;
        validate_trainable(&online)?;
        parameter_norm(&online)?;
        let target = frozen_copy(&online).valid();
        let optimizer = config.optimizer.build()?;
        Ok(Self {
            online,
            target,
            optimizer,
            config,
            observation_space,
            action_space,
            updates: 0,
            poisoned: false,
        })
    }

    pub(super) fn check_update(&self) -> Result<()> {
        ensure(
            !self.poisoned && self.updates < i32::MAX as u64 - 1,
            "candidate critic requires checkpoint recovery or exhausted its update counter",
        )
    }

    pub(super) fn fit_predictions(
        &mut self,
        predictions: Tensor<1>,
        targets: &[f32],
        candidates: usize,
    ) -> Result<UpdateReport> {
        ensure(
            !targets.is_empty() && predictions.dims()[0] == targets.len(),
            "candidate prediction and target counts differ",
        )?;
        let device = predictions.device();
        let error = predictions - Tensor::from_data(targets, &device);
        let loss = match self.config.loss {
            BellmanLoss::Squared => error.powf_scalar(2.).mean(),
            BellmanLoss::Huber { delta } => {
                let absolute = error.abs();
                let quadratic = absolute.clone().clamp_max(delta);
                (quadratic.clone().powf_scalar(2.) * 0.5 + (absolute - quadratic) * delta).mean()
            }
        };

        self.poisoned = true;
        let (loss, norm) = optimize_validated(
            &mut self.online,
            &mut self.optimizer,
            self.config.optimizer.learning_rate,
            loss,
        )?;
        parameter_norm(&self.online)?;
        self.updates += 1;
        if self.updates.is_multiple_of(self.config.target_interval) {
            self.target =
                polyak_update(self.target.clone(), &self.online, self.config.target_tau)?.valid();
        }
        self.poisoned = false;
        let target_mean =
            targets.iter().map(|&target| f64::from(target)).sum::<f64>() / targets.len() as f64;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("critic_loss", loss)
        .metric("critic_gradient_norm", norm)
        .metric("critic_positions", targets.len() as f64)
        .metric("critic_current_candidates", candidates as f64)
        .metric("target_mean", target_mean))
    }
}

impl<Q: StructuredActionValue> StatefulLearner for CandidateDdqn<Q> {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        CandidateDdqn::update(self, batch)
    }
}

impl<Q: StructuredActionValue> TemporalOffPolicyLearner for CandidateDdqn<Q> {
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

impl<Q: StructuredActionValue> BehaviorPublisher for CandidateDdqn<Q> {
    type Actor = CandidateQActor<Q>;
    type Settings = CandidateExploration;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        self.critic.publish(*settings, revision)
    }
}

impl<Q: StructuredActionValue> CandidateCritic<Q> {
    pub(super) fn publish(
        &self,
        settings: CandidateExploration,
        revision: u64,
    ) -> Result<CandidateQActor<Q>> {
        ensure(
            !self.poisoned,
            "candidate critic requires checkpoint recovery",
        )?;
        CandidateQActor::new(
            frozen_copy(&self.online).valid(),
            self.observation_space.clone(),
            self.action_space.clone(),
            settings,
            revision,
            self.config.pack_limits(),
        )
    }
}

fn model_sample(
    observation: &Observation,
    model: &impl StructuredActionValue,
) -> Result<StructuredSample<ConditionalAction>> {
    StructuredSample::from_observation_with_validated_schema(
        model_frame(observation)?,
        model.input_schema(),
    )
}

fn model_frame(observation: &Observation) -> Result<&StructuredObservation> {
    observation.as_structured().ok_or_else(|| {
        crate::Error::Shape("candidate model requires a structured observation".into())
    })
}

pub(super) fn candidate_sample(
    observation: &Observation,
    schema: &InputSchema,
) -> Result<StructuredSample<ConditionalAction>> {
    StructuredSample::from_observation_with_validated_schema(candidate_frame(observation)?, schema)
}

pub(super) fn candidate_frame(observation: &Observation) -> Result<&StructuredObservation> {
    match observation {
        Observation::Structured(frame) => Ok(frame),
        _ => Err(crate::Error::Shape(
            "candidate model requires a structured observation".into(),
        )),
    }
}

pub(super) fn double_q_target(
    reward: f32,
    discount: f32,
    online: &[f32],
    target: &[f32],
    enabled: &[bool],
) -> Result<f32> {
    if discount == 0. {
        return Ok(reward);
    }
    ensure(
        online.len() == target.len()
            && target.len() == enabled.len()
            && online.iter().chain(target).all(|value| value.is_finite()),
        "invalid candidate bootstrap values",
    )?;
    let maximum = enabled_maximum(online, enabled)?;
    let mut continuation: Vec<_> = online
        .iter()
        .zip(target)
        .zip(enabled)
        .filter_map(|((&q, &target), &enabled)| {
            (enabled && q == maximum).then_some(f64::from(target))
        })
        .collect();
    continuation.sort_by(f64::total_cmp);
    let mean = continuation.iter().sum::<f64>() / continuation.len() as f64;
    let value = (f64::from(reward) + f64::from(discount) * mean) as f32;
    ensure(value.is_finite(), "candidate Bellman target overflow")?;
    Ok(value)
}

pub(super) fn maximum_target(
    reward: f32,
    discount: f32,
    target: &[f32],
    enabled: &[bool],
) -> Result<f32> {
    if discount == 0. {
        return Ok(reward);
    }
    ensure(
        target.len() == enabled.len() && target.iter().all(|value| value.is_finite()),
        "invalid candidate bootstrap values",
    )?;
    let maximum = enabled_maximum(target, enabled)?;
    let value = (f64::from(reward) + f64::from(discount) * f64::from(maximum)) as f32;
    ensure(value.is_finite(), "candidate Bellman target overflow")?;
    Ok(value)
}

fn enabled_maximum(values: &[f32], enabled: &[bool]) -> Result<f32> {
    values
        .iter()
        .zip(enabled)
        .filter_map(|(&q, &enabled)| enabled.then_some(q))
        .max_by(f32::total_cmp)
        .ok_or_else(|| crate::Error::Action("no enabled bootstrap candidate".into()))
}

pub(super) fn policy_target(
    reward: f32,
    discount: f32,
    values: &[f32],
    probabilities: &[f64],
) -> Result<f32> {
    ensure(
        values.len() == probabilities.len()
            && values.iter().all(|value| value.is_finite())
            && probabilities.iter().all(|&p| p.is_finite() && p >= 0.)
            && (probabilities.iter().sum::<f64>() - 1.).abs() <= 1e-10,
        "invalid candidate policy backup values or probability mass",
    )?;
    let continuation = values
        .iter()
        .zip(probabilities)
        .map(|(&q, &p)| f64::from(q) * p)
        .sum::<f64>();
    let value = (f64::from(reward) + f64::from(discount) * continuation) as f32;
    ensure(
        value.is_finite(),
        "candidate policy Bellman target overflow",
    )?;
    Ok(value)
}
