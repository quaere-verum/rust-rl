//! Recurrent discrete MPO with explicitly distinct evaluation/optimality critics.
use super::{
    OptimizerConfig,
    recurrent_critic::{CriticBackup, RecurrentDiscreteCritic},
};
use crate::{
    Result,
    core::{
        ensure,
        env::ObservationBatch,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner, TemporalOffPolicyLearner},
        temporal::TemporalExecutionMode,
    },
    data::temporal::{SequenceBatch, SequenceOrigin},
    nn::{
        distributions::Distribution,
        optimization::optimize_validated,
        target::frozen_copy,
        temporal::TemporalEncoder,
        temporal_batch::{EpisodeInputCache, encode_history_batch},
        temporal_categorical_actor::{CategoricalMixtureSettings, TemporalCategoricalActor},
        temporal_models::{self, PreparedHistory, gather_history_features},
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_work::EpisodeInferencePlan,
        to_vec,
    },
    objectives::mpo::{e_step, weighted_projection},
};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    optim::ModuleOptimizer,
    tensor::{Tensor, TensorData},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[cfg(feature = "probes")]
use crate::profiling::{self, Phase};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MpoBackup {
    /// Expectation backup under the actor frozen before this update.
    PolicyEvaluation,
    /// Hybrid with Double-Q optimality assistance; not ordinary policy evaluation.
    DoubleQOptimality,
    /// Maximize the target critic over legal actions.
    MaximumOptimality,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentMpoConfig {
    pub actor: OptimizerConfig,
    pub e_step_kl: f64,
    pub projection_kl: f64,
    pub backup: MpoBackup,
}
impl Default for RecurrentMpoConfig {
    fn default() -> Self {
        Self {
            actor: OptimizerConfig {
                learning_rate: 0.003,
                max_grad_norm: Some(1.),
            },
            e_step_kl: 0.1,
            projection_kl: 0.01,
            backup: MpoBackup::PolicyEvaluation,
        }
    }
}
impl RecurrentMpoConfig {
    pub fn validate(&self) -> Result<()> {
        self.actor.validate()?;
        ensure(
            self.e_step_kl.is_finite()
                && self.e_step_kl > 0.
                && self.projection_kl.is_finite()
                && self.projection_kl > 0.,
            "invalid MPO KL budgets",
        )
    }
}
pub struct RecurrentMpo {
    policy: TemporalPolicy,
    critic: RecurrentDiscreteCritic,
    config: RecurrentMpoConfig,
    optimizer: ModuleOptimizer,
    actor_updates: u64,
    update_mode: TemporalExecutionMode,
    poisoned: bool,
}
impl RecurrentMpo {
    pub fn new(
        policy: TemporalPolicy,
        critic: RecurrentDiscreteCritic,
        config: RecurrentMpoConfig,
    ) -> Result<Self> {
        config.validate()?;
        policy.architecture()?;
        temporal_models::validate_parameters(&policy, false)?;
        ensure(
            policy.spec() == critic.spec()
                && critic.config().n_step == 1
                && critic.reconstruction_config().is_exact_default()
                && critic.updates() == 0
                && critic.update_mode().is_reference(),
            "MPO requires matching actor/critic histories and a fresh one-step critic",
        )?;
        crate::nn::optimization::validate_disjoint(&policy, &critic.online())?;
        let optimizer = config.actor.build()?;
        Ok(Self {
            policy: policy.valid().fork(&crate::nn::device()).unfreeze().train(),
            critic,
            config,
            optimizer,
            actor_updates: 0,
            update_mode: TemporalExecutionMode::Reference,
            poisoned: false,
        })
    }
    pub fn config(&self) -> &RecurrentMpoConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.actor_updates
    }
    pub fn update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
    }
    /// Change recurrent execution without changing the replay or update schedule.
    pub fn set_update_mode(&mut self, mode: TemporalExecutionMode) {
        self.update_mode = mode;
    }
    pub fn policy(&self) -> TemporalPolicy {
        frozen_copy(&self.policy)
    }
    pub fn critic(&self) -> &RecurrentDiscreteCritic {
        &self.critic
    }
    /// Padded B,T,A probabilities, independently reconstructed with the actor.
    /// True terminal and padding rows are zero; truncation rows bootstrap.
    pub fn policy_probabilities(&self, batch: &SequenceBatch) -> Result<Vec<f32>> {
        self.policy_probabilities_using(batch, |tokens| self.policy().distribution(batch, tokens))
    }

    fn policy_probabilities_using(
        &self,
        batch: &SequenceBatch,
        distribution: impl FnOnce(&[(usize, usize)]) -> Result<Distribution>,
    ) -> Result<Vec<f32>> {
        batch.validate()?;
        let time = batch
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let actions = self.policy.spec().action_width();
        let tokens = batch
            .samples
            .iter()
            .enumerate()
            .flat_map(|(r, s)| {
                (0..s.observations.len())
                    .filter(move |&t| t == 0 || !s.steps[t - 1].terminated)
                    .map(move |t| (r, t))
            })
            .collect::<Vec<_>>();
        let terms = distribution(&tokens)?.categorical_terms()?;
        let p = to_vec(terms.probabilities)?;
        let mut out = vec![0.; batch.samples.len() * time * actions];
        for (token, chunk) in tokens.iter().zip(p.chunks(actions)) {
            let start = (token.0 * time + token.1) * actions;
            out[start..start + actions].copy_from_slice(chunk);
        }
        Ok(out)
    }
    pub fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        #[cfg(feature = "probes")]
        let _update = profiling::enter(Phase::MpoUpdate);
        ensure(
            !self.poisoned,
            "MPO requires checkpoint recovery after a failed update",
        )?;
        self.config.validate()?;
        batch.validate()?;
        ensure(
            batch.config.n_step == 1,
            "MPO requires one-step recurrent batches",
        )?;
        ensure(
            self.actor_updates == self.critic.updates() && self.actor_updates < u64::MAX,
            "MPO counters disagree or overflow",
        )?;
        let tokens = batch
            .samples
            .iter()
            .enumerate()
            .flat_map(|(r, s)| {
                s.loss_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| **m)
                    .map(move |(t, _)| (r, t))
            })
            .collect::<Vec<_>>();
        let actions = self.policy.spec().action_width();
        let time = batch
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let (history, inference) = if self.update_mode == TemporalExecutionMode::OptimizedExact
            && !self.policy.observation_only()
            && batch
                .samples
                .iter()
                .all(|s| matches!(s.origin, SequenceOrigin::Reset))
        {
            #[cfg(feature = "probes")]
            let _encoding = profiling::enter(Phase::HistoryInputCache);
            let spec = self.policy.spec();
            let device = self.policy.device();
            let cache =
                if EpisodeInferencePlan::eligible(batch.samples.len(), spec.input_dim(), &device) {
                    EpisodeInputCache::try_from_validated_replay(batch, spec)?
                } else {
                    None
                };
            let (encoded, plan) = if let Some(cache) = &cache {
                let plan = EpisodeInferencePlan::try_new(
                    cache,
                    &[
                        self.policy.encoder.output_dim(),
                        self.critic.online.hidden_dim(),
                    ],
                    &device,
                )?;
                let indices: Vec<_> = (0..batch.samples.len()).collect();
                (cache.pack(&indices)?, plan)
            } else {
                (encode_history_batch(batch, spec)?, None)
            };
            (Some(PreparedHistory::new(encoded, None, &device)), plan)
        } else {
            (None, None)
        };
        let proposal_history = inference.as_ref().filter(|plan| {
            self.policy
                .encoder
                .gru
                .as_ref()
                .is_some_and(|encoder| plan.supports_hidden(encoder.output_dim()))
        });
        let target_history = inference
            .as_ref()
            .filter(|plan| plan.supports_hidden(self.critic.online.hidden_dim()));
        let (masks, old, log_probs, actor_features) = {
            #[cfg(feature = "probes")]
            let _policy = profiling::enter(Phase::MpoReferencePolicy);
            let features = history
                .as_ref()
                .map(|history| history.unroll_features(&self.policy.encoder, &batch.samples))
                .transpose()?;
            let current = if let Some(features) = &features {
                distribution_from_history(&self.policy, batch, features.clone(), &tokens)?
            } else {
                self.policy().distribution(batch, &tokens)?
            };
            let terms = current.categorical_terms()?;
            let old = normalized_rows(&to_vec(terms.probabilities.without_autodiff())?, actions)?;
            let Distribution::Categorical { masks, .. } = current else {
                return Err(crate::Error::Action(
                    "MPO supports discrete actions only".into(),
                ));
            };
            (masks, old, terms.log_probs, features)
        };
        let (q, critic_forward) = {
            #[cfg(feature = "probes")]
            let _critic = profiling::enter(Phase::MpoEStepQ);
            let forward = history
                .as_ref()
                .map(|history| {
                    let (input, masks) = history.input_and_masks();
                    self.critic
                        .full_q_forward(batch, input, masks, target_history)
                })
                .transpose()?;
            let reference;
            let online = if let Some(forward) = &forward {
                &forward.selection
            } else {
                reference = self.critic.reconstructed_q_values(batch)?.0;
                &reference
            };
            let q = tokens
                .iter()
                .flat_map(|&(r, t)| {
                    online[(r * time + t) * actions..(r * time + t + 1) * actions]
                        .iter()
                        .map(|q| *q as f64)
                })
                .collect::<Vec<_>>();
            (q, forward)
        };
        let e = {
            #[cfg(feature = "probes")]
            let _step = profiling::enter(Phase::MpoEStep);
            e_step(&old, &q, &masks, actions, self.config.e_step_kl)?
        };
        let targets = Tensor::from_data(
            TensorData::new(
                e.probabilities
                    .iter()
                    .map(|p| *p as f32)
                    .collect::<Vec<_>>(),
                [tokens.len(), actions],
            ),
            &self.policy.device(),
        );
        // Form continuation probabilities with the current actor before either update.
        let expected = match (self.config.backup, actor_features) {
            (MpoBackup::PolicyEvaluation, actor_features) => {
                #[cfg(feature = "probes")]
                let _backup = profiling::enter(Phase::MpoPolicyBackup);
                let probabilities = if let Some(features) = actor_features {
                    // Keep the original policy-evaluation token set and head
                    // shape. Only the current-parameter encoder tape is shared.
                    let policy = self.policy();
                    let features = Tensor::from_data(features.into_data(), &policy.device());
                    self.policy_probabilities_using(batch, |tokens| {
                        distribution_from_history(&policy, batch, features, tokens)
                    })?
                } else {
                    self.policy_probabilities(batch)?
                };
                Some(probabilities)
            }
            (MpoBackup::DoubleQOptimality | MpoBackup::MaximumOptimality, _) => None,
        };
        let loss = {
            #[cfg(feature = "probes")]
            let _projection = profiling::enter(Phase::MpoProjectionLoss);
            let log_probs = if history.is_some() {
                log_probs
            } else {
                self.policy
                    .distribution(batch, &tokens)?
                    .categorical_terms()?
                    .log_probs
            };
            weighted_projection(log_probs, targets)
        };
        // Preflight above cannot mutate a model. Failures from this point require
        // full checkpoint recovery; no partially updated model can be published.
        self.poisoned = true;
        let mut report = {
            #[cfg(feature = "probes")]
            let _fit = profiling::enter(Phase::MpoCriticFit);
            let backup = match self.config.backup {
                MpoBackup::PolicyEvaluation => CriticBackup::PolicyEvaluation(
                    expected.as_deref().expect("current policy rows"),
                ),
                MpoBackup::DoubleQOptimality => CriticBackup::DoubleQOptimality,
                MpoBackup::MaximumOptimality => CriticBackup::MaximumOptimality,
            };
            if let Some(forward) = critic_forward {
                self.critic.fit_from_full_forward(batch, backup, forward)?
            } else {
                self.critic.fit(batch, backup)?
            }
        };
        drop(expected);
        // Preserve the validated projection convention: Adam advances once even
        // when backtracking rejects its full displacement (or all displacements).
        let old_model = self.policy.clone();
        // Keep the last accepted actor readable even if proposal evaluation fails.
        // The optimizer still advances exactly once and then requires recovery.
        let mut proposal = self.policy.clone();
        let (actor_loss, norm) = {
            #[cfg(feature = "probes")]
            let _optimizer = profiling::enter(Phase::ActorOptimization);
            optimize_validated(
                &mut proposal,
                &mut self.optimizer,
                self.config.actor.learning_rate,
                loss,
            )?
        };
        self.actor_updates += 1;
        #[cfg(feature = "probes")]
        let backtracking = profiling::enter(Phase::MpoBacktracking);
        let before = -e
            .probabilities
            .iter()
            .zip(&old)
            .filter(|(q, _)| **q > 0.)
            .map(|(q, p)| q * p.ln())
            .sum::<f64>()
            / tokens.len() as f64;
        let mut scale = 1.;
        let mut accepted = None;
        let mut evaluations = 0u64;
        for _ in 0..=10 {
            let candidate = if scale == 1. {
                proposal.clone()
            } else {
                blend_policy(&old_model, &proposal, scale)?
            };
            let frozen = frozen_copy(&candidate);
            let distribution = if let Some(plan) = proposal_history {
                let encoder = frozen.encoder.gru.as_ref().expect("coalesced GRU proposal");
                let features = plan.features(encoder)?.autodiff();
                distribution_from_history(&frozen, batch, features, &tokens)?
            } else if let Some(history) = &history {
                let features = history.unroll_features(&frozen.encoder, &batch.samples)?;
                distribution_from_history(&frozen, batch, features, &tokens)?
            } else {
                frozen.distribution(batch, &tokens)?
            };
            let terms = distribution.categorical_terms()?;
            evaluations += 1;
            let p = normalized_rows(&to_vec(terms.probabilities)?, actions)?;
            let kl = forward_kl(&old, &p, actions);
            let after = -e
                .probabilities
                .iter()
                .zip(&p)
                .filter(|(q, _)| **q > 0.)
                .map(|(q, p)| q * p.ln())
                .sum::<f64>()
                / tokens.len() as f64;
            if kl.is_finite() && kl <= self.config.projection_kl && after <= before + 1e-7 {
                accepted = Some((candidate, kl, after));
                break;
            }
            scale *= 0.5;
        }
        let (model, kl, after) = accepted.unwrap_or_else(|| {
            scale = 0.;
            (old_model, 0., before)
        });
        #[cfg(feature = "probes")]
        drop(backtracking);
        temporal_models::validate_parameters(&model, false)?;
        self.policy = model;
        self.poisoned = false;
        // The optimized path shares one current actor tape across all
        // pre-proposal objectives and one independent pass per Q model.
        let (actor_passes, online_passes, target_passes) = if history.is_some() {
            (1 + evaluations, 1, 1)
        } else {
            (
                2 + u64::from(self.config.backup == MpoBackup::PolicyEvaluation) + evaluations,
                3,
                2,
            )
        };
        let padded = (batch.samples.len() * time) as u64;
        let valid = batch
            .samples
            .iter()
            .map(|s| s.observations.len() as u64)
            .sum::<u64>();
        let (proposal_padded, proposal_valid) =
            proposal_history.map_or((padded, valid), EpisodeInferencePlan::work);
        let (target_padded, target_valid) =
            target_history.map_or((padded, valid), EpisodeInferencePlan::work);
        let actor_padded = padded * (actor_passes - evaluations) + proposal_padded * evaluations;
        let actor_valid = valid * (actor_passes - evaluations) + proposal_valid * evaluations;
        report.metrics.extend(BTreeMap::from([
            ("e_step_reverse_kl".into(), e.reverse_kl),
            ("e_step_temperature".into(), e.temperature.unwrap_or(0.)),
            ("e_step_active".into(), f64::from(e.active)),
            ("e_step_dual".into(), e.dual),
            ("projection_forward_kl".into(), kl),
            ("projection_scale".into(), f64::from(scale)),
            ("projection_before".into(), before),
            ("projection_after".into(), after),
            ("actor_gradient_norm".into(), norm),
            ("actor_loss".into(), actor_loss),
            ("actor_optimizer_calls".into(), 1.),
            ("critic_optimizer_calls".into(), 1.),
            ("projection_evaluations".into(), evaluations as f64),
            ("actor_reconstruction_passes".into(), actor_passes as f64),
            ("actor_forward_tokens".into(), actor_padded as f64),
            (
                "online_critic_forward_tokens".into(),
                (padded * online_passes) as f64,
            ),
            (
                "target_critic_forward_tokens".into(),
                (target_padded * target_passes) as f64,
            ),
            ("actor_valid_history_tokens".into(), actor_valid as f64),
            (
                "online_critic_valid_history_tokens".into(),
                (valid * online_passes) as f64,
            ),
            (
                "target_critic_valid_history_tokens".into(),
                (target_valid * target_passes) as f64,
            ),
        ]));
        Ok(report)
    }
}

fn distribution_from_history(
    policy: &TemporalPolicy,
    batch: &SequenceBatch,
    features: Tensor<3>,
    tokens: &[(usize, usize)],
) -> Result<Distribution> {
    let observations: Vec<_> = tokens
        .iter()
        .map(|&(s, t)| &batch.samples[s].observations[t])
        .collect();
    policy.distribution_from_features(
        gather_history_features(features, tokens),
        ObservationBatch::from_refs(&observations),
    )
}

fn normalized_rows(values: &[f32], actions: usize) -> Result<Vec<f64>> {
    ensure(
        actions > 0 && values.len().is_multiple_of(actions),
        "MPO probability shape mismatch",
    )?;
    let mut p = Vec::with_capacity(values.len());
    for row in values.chunks(actions) {
        let sum = row.iter().map(|p| *p as f64).sum::<f64>();
        ensure(
            row.iter().all(|p| p.is_finite() && *p >= 0.) && (sum - 1.).abs() < 1e-5,
            "invalid MPO actor probability mass",
        )?;
        p.extend(row.iter().map(|p| (*p as f64) / sum));
    }
    Ok(p)
}
fn forward_kl(old: &[f64], new: &[f64], actions: usize) -> f64 {
    old.iter()
        .zip(new)
        .filter(|(p, _)| **p > 0.)
        .map(|(p, q)| p * (p.ln() - q.ln()))
        .sum::<f64>()
        / (old.len() / actions) as f64
}
/// Preserve proposal parameter IDs and optimizer ownership while scaling its displacement.
fn blend_policy(
    old: &TemporalPolicy,
    proposal: &TemporalPolicy,
    scale: f32,
) -> Result<TemporalPolicy> {
    struct Collect(Vec<Tensor<1>>);
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let v = p.val().without_autodiff();
            self.0
                .push(v.clone().reshape([v.dims().iter().product::<usize>()]));
        }
    }
    struct Blend {
        old: std::vec::IntoIter<Tensor<1>>,
        scale: f32,
    }
    impl ModuleMapper for Blend {
        fn map_float<const D: usize>(&mut self, p: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let old = self.old.next().expect("same policy architecture");
            p.map(|new| {
                let shape = new.dims();
                old.reshape(shape) * (1. - self.scale) + new.without_autodiff() * self.scale
            })
        }
    }
    let mut collect = Collect(Vec::new());
    old.visit(&mut collect);
    Ok(proposal
        .clone()
        .map(&mut Blend {
            old: collect.0.into_iter(),
            scale,
        })
        .unfreeze()
        .train())
}
impl StatefulLearner for RecurrentMpo {
    type Batch = SequenceBatch;
    fn update(&mut self, b: &SequenceBatch) -> Result<UpdateReport> {
        RecurrentMpo::update(self, b)
    }
}
impl TemporalOffPolicyLearner for RecurrentMpo {
    fn return_contract(&self) -> crate::core::temporal::ReturnSpec {
        self.critic.return_contract()
    }
}
impl BehaviorPublisher for RecurrentMpo {
    type Actor = TemporalCategoricalActor;
    type Settings = CategoricalMixtureSettings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        ensure(
            !self.poisoned,
            "MPO requires checkpoint recovery after a failed update",
        )?;
        TemporalCategoricalActor::new(self.policy(), settings.epsilon, revision)
    }
}
mod checkpoint;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::checkpoint::Checkpoint;
    use crate::{
        algorithms::recurrent_critic::{RecurrentCriticConfig, construct_policy_targets},
        core::{
            action::{Action, ActionSpace},
            env::Observation,
            stateful::HistorySpec,
            temporal::BehaviorMetadata,
        },
        data::temporal::{SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep},
        nn::{temporal::RecurrentQ, temporal_policy::TemporalStochasticPolicy, to_vec},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn make(backup: MpoBackup) -> RecurrentMpo {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let policy = TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(682),
        )
        .unwrap();
        let q = TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(734),
        )
        .unwrap();
        let critic = RecurrentDiscreteCritic::new(
            RecurrentQ::new(q.encoder.gru.unwrap(), q.head).unwrap(),
            spec,
            RecurrentCriticConfig {
                target_interval: 2,
                gamma: 0.9,
                n_step: 1,
                ..Default::default()
            },
        )
        .unwrap();
        RecurrentMpo::new(
            policy,
            critic,
            RecurrentMpoConfig {
                backup,
                projection_kl: 1e-6,
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn fixture(terminal: bool, burn: usize) -> SequenceBatch {
        let o = |x, m| Observation::dense(vec![x], Some(m));
        SequenceBatch {
            config: SequenceConfig {
                burn_in: burn,
                learn_len: 2 - burn,
                n_step: 1,
            },
            samples: vec![SequenceSample {
                episode_id: 7,
                trajectory_start: 0,
                learn_start: burn,
                burn_start: 0,
                history_start: 0,
                previous: None,
                origin: SequenceOrigin::Reset,
                observations: vec![
                    o(1., vec![true, true]),
                    o(2., vec![true, false]),
                    o(
                        3.,
                        if terminal {
                            vec![false, false]
                        } else {
                            vec![true, true]
                        },
                    ),
                ],
                steps: vec![
                    TemporalStep {
                        action: Action::Discrete(1),
                        reward: 0.25_f32.into(),
                        terminated: false,
                        truncated: false,
                        behavior: BehaviorMetadata::default(),
                    },
                    TemporalStep {
                        action: Action::Discrete(0),
                        reward: 1.0_f32.into(),
                        terminated: terminal,
                        truncated: !terminal,
                        behavior: BehaviorMetadata::default(),
                    },
                ],
                loss_mask: vec![burn == 0, true],
            }],
        }
    }
    fn probabilities(
        model: &TemporalPolicy,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
    ) -> Vec<f32> {
        to_vec(
            model
                .distribution(batch, tokens)
                .unwrap()
                .categorical_terms()
                .unwrap()
                .probabilities,
        )
        .unwrap()
    }

    #[test]
    fn evaluation_uses_the_frozen_actor_at_next_history_including_truncation() {
        for terminal in [true, false] {
            let mut m = make(MpoBackup::PolicyEvaluation);
            let b = fixture(terminal, 0);
            let p = m.policy_probabilities(&b).unwrap();
            assert_eq!(p.len(), 6);
            assert_eq!(p[3], 0.);
            if terminal {
                assert_eq!(&p[4..], [0., 0.]);
            } else {
                assert!((p[4] + p[5] - 1.).abs() < 1e-6);
            }
            let (_, target) = m.critic().reconstructed_q_values(&b).unwrap();
            let expected = construct_policy_targets(&b, &p, &target, 2, 0.9).unwrap();
            assert_eq!(expected[0].value, 0.25 + 0.9 * target[2]);
            if terminal {
                assert_eq!(expected[1].value, 1.);
            } else {
                assert_ne!(expected[1].value, 1.);
            }
            let mean = expected.iter().map(|t| t.value as f64).sum::<f64>() / 2.;
            let r = m.update(&b).unwrap();
            assert!((r.metrics["target_mean"] - mean).abs() < 1e-7);
        }
    }

    #[test]
    fn constant_two_action_fixture_distinguishes_evaluation_from_maximum_backup() {
        use crate::nn::{matrix, vector};
        use burn::module::Param;
        for (mode, expected) in [
            (MpoBackup::PolicyEvaluation, 1.75),
            (MpoBackup::DoubleQOptimality, 2.425),
            (MpoBackup::MaximumOptimality, 2.425),
        ] {
            let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
            let mut p = TemporalPolicy::new(
                spec.clone(),
                Some(4),
                &[],
                &mut ChaCha8Rng::seed_from_u64(63),
            )
            .unwrap();
            p.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 8], 4, 2));
            p.head.layers[0].bias = Some(Param::from_tensor(vector(&[3f32.ln(), 0.])));
            let mut q = TemporalPolicy::new(
                spec.clone(),
                Some(4),
                &[],
                &mut ChaCha8Rng::seed_from_u64(82),
            )
            .unwrap();
            q.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 8], 4, 2));
            q.head.layers[0].bias = Some(Param::from_tensor(vector(&[1., 3.])));
            let critic = RecurrentDiscreteCritic::new(
                RecurrentQ::new(q.encoder.gru.unwrap(), q.head).unwrap(),
                spec,
                RecurrentCriticConfig {
                    gamma: 0.9,
                    n_step: 1,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut m = RecurrentMpo::new(
                p,
                critic,
                RecurrentMpoConfig {
                    backup: mode,
                    ..Default::default()
                },
            )
            .unwrap();
            // y0=.25+.9*1=1.15, y1=1+.9*(.75*1+.25*3)=2.35;
            // the optimality alternative uses y1=1+.9*3=3.7.
            let r = m.update(&fixture(false, 0)).unwrap();
            assert!((r.metrics["target_mean"] - expected).abs() < 2e-7);
        }
    }

    #[test]
    fn projection_respects_forward_kl_and_improves_its_fixed_likelihood() {
        for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
            let mut m = make(backup);
            let b = fixture(true, 0);
            let tokens = [(0, 0), (0, 1)];
            let old = probabilities(&m.policy(), &b, &tokens);
            let report = m.update(&b).unwrap();
            let new = probabilities(&m.policy(), &b, &tokens);
            let kl = old
                .iter()
                .zip(&new)
                .filter(|(p, _)| **p > 0.)
                .map(|(p, q)| (*p as f64) * ((*p as f64) / (*q as f64)).ln())
                .sum::<f64>()
                / 2.;
            assert!(kl <= 1e-6 + 1e-8, "{kl}");
            assert!((report.metrics["projection_forward_kl"] - kl).abs() < 1e-7);
            assert!(
                report.metrics["projection_after"] <= report.metrics["projection_before"] + 1e-7
            );
            assert!(report.metrics["projection_scale"] < 1.);
            assert_eq!(new[3], 0.);
        }
    }

    #[cfg(feature = "probes")]
    #[test]
    fn recurrent_work_accounts_for_every_profiled_actor_reconstruction_retry() {
        use crate::profiling::{self, Phase};
        for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
            for (mode, online, target) in [
                (TemporalExecutionMode::Reference, 3, 2),
                (TemporalExecutionMode::OptimizedExact, 1, 1),
            ] {
                let mut m = make(backup);
                m.set_update_mode(mode);
                let b = fixture(true, 0);
                profiling::reset();
                profiling::set_enabled(true);
                let r = m.update(&b).unwrap();
                profiling::set_enabled(false);
                let phases = profiling::snapshot();
                let calls = |phase| {
                    phases
                        .iter()
                        .find(|s| s.phase == phase)
                        .map_or(0, |s| s.calls)
                };
                let actor = calls(Phase::HistoryUnroll);
                assert_eq!(r.metrics["actor_reconstruction_passes"], actor as f64);
                assert_eq!(r.metrics["actor_forward_tokens"], (actor * 3) as f64);
                assert_eq!(
                    r.metrics["online_critic_forward_tokens"],
                    (online * 3) as f64
                );
                assert_eq!(
                    r.metrics["target_critic_forward_tokens"],
                    (target * 3) as f64
                );
                assert!(r.metrics["projection_evaluations"] > 1.);
                assert_eq!(calls(Phase::MpoUpdate), 1);
                assert_eq!(calls(Phase::MpoEStep), 1);
                assert_eq!(calls(Phase::MpoCriticFit), 1);
                assert_eq!(calls(Phase::MpoBacktracking), 1);
                assert_eq!(calls(Phase::OnlineQHistory), online);
                assert_eq!(calls(Phase::TargetQHistory), target);
                assert_eq!(
                    calls(Phase::MpoPolicyBackup),
                    u64::from(backup == MpoBackup::PolicyEvaluation)
                );
            }
        }
    }

    #[test]
    fn optimized_critic_policy_fit_checks_masks_and_matches_target_refresh() {
        let mut reference = make(MpoBackup::PolicyEvaluation);
        let mut optimized = make(MpoBackup::PolicyEvaluation);
        optimized
            .critic
            .restore(&reference.critic.checkpoint().unwrap())
            .unwrap();
        optimized
            .critic
            .set_update_mode(TemporalExecutionMode::OptimizedExact);
        for terminal in [true, false] {
            let batch = fixture(terminal, 0);
            let probabilities = reference.policy_probabilities(&batch).unwrap();
            let mut invalid = probabilities.clone();
            invalid[2] = 0.5;
            invalid[3] = 0.5;
            let before = serde_json::to_value(optimized.critic.checkpoint().unwrap()).unwrap();
            assert!(
                optimized
                    .critic
                    .fit(&batch, CriticBackup::PolicyEvaluation(&invalid))
                    .is_err()
            );
            assert_eq!(
                serde_json::to_value(optimized.critic.checkpoint().unwrap()).unwrap(),
                before
            );
            let backup = CriticBackup::PolicyEvaluation(&probabilities);
            let a = reference.critic.fit(&batch, backup).unwrap();
            let b = optimized.critic.fit(&batch, backup).unwrap();
            for (name, value) in a.metrics {
                assert_eq!(value.to_bits(), b.metrics[&name].to_bits(), "{name}");
            }
            let readouts = |critic: &RecurrentDiscreteCritic| {
                let (a, b) = critic.reconstructed_q_values(&batch).unwrap();
                a.into_iter().chain(b).map(f32::to_bits).collect::<Vec<_>>()
            };
            assert_eq!(readouts(&reference.critic), readouts(&optimized.critic));
        }
    }

    #[test]
    fn independently_reconstructed_actor_matches_full_prefix_with_burn_in() {
        let m = make(MpoBackup::PolicyEvaluation);
        let b = fixture(true, 1);
        let p = m.policy_probabilities(&b).unwrap();
        let direct = probabilities(&m.policy(), &b, &[(0, 0), (0, 1)]);
        for (a, b) in p[..4].iter().zip(direct) {
            assert_eq!(*a, b);
        }
        let mut changed = b.clone();
        changed.samples[0].observations[0]
            .as_dense_mut()
            .unwrap()
            .values[0] = -3.;
        let modified = m.policy_probabilities(&changed).unwrap();
        // The second mask is singleton; compare recurrent effect at a non-singleton next observation.
        changed.samples[0].observations[1]
            .as_dense_mut()
            .unwrap()
            .action_mask = Some(vec![true, true]);
        let with_history = m.policy_probabilities(&changed).unwrap();
        let mut no_history = changed.clone();
        no_history.samples[0].observations[0]
            .as_dense_mut()
            .unwrap()
            .values[0] = 1.;
        let original = m.policy_probabilities(&no_history).unwrap();
        assert_ne!(&with_history[2..4], &original[2..4]);
        assert_eq!(modified[3], 0.);
    }

    #[test]
    fn failed_projection_keeps_last_accepted_actor_and_requires_recovery() {
        let mut model = make(MpoBackup::PolicyEvaluation);
        let batch = fixture(true, 0);
        let before = probabilities(&model.policy(), &batch, &[(0, 0)]);
        let saved = model.checkpoint().unwrap();
        model.config.actor.learning_rate = 1e300;
        assert!(model.update(&batch).is_err());
        assert!(model.checkpoint().is_err());
        assert!(
            model
                .behavior_actor(&CategoricalMixtureSettings::default(), 0)
                .is_err()
        );
        assert_eq!(probabilities(&model.policy(), &batch, &[(0, 0)]), before);
        model.restore(&saved).unwrap();
        model.update(&batch).unwrap();
    }

    #[test]
    fn missing_return_contract_is_rejected_without_mutating_destination() {
        let mut model = make(MpoBackup::PolicyEvaluation);
        let mut record = model.checkpoint().unwrap();
        record.temporal_returns = None;
        let before = model.critic().parameter_values().unwrap();
        assert!(model.restore(&record).is_err());
        assert_eq!(model.critic().parameter_values().unwrap(), before);
    }

    #[test]
    fn complete_resume_reproduces_next_update_and_invalid_restore_is_atomic() {
        for mode in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
            let mut a = make(mode);
            let b = fixture(true, 0);
            a.update(&b).unwrap();
            let record = a.checkpoint().unwrap();
            let mut resumed = make(mode);
            resumed.restore(&record).unwrap();
            let x = a.update(&b).unwrap();
            let y = resumed.update(&b).unwrap();
            assert_eq!(x.metrics, y.metrics);
            assert_eq!(
                probabilities(&a.policy(), &b, &[(0, 0)]),
                probabilities(&resumed.policy(), &b, &[(0, 0)])
            );
            assert_eq!(
                a.critic().parameter_values().unwrap(),
                resumed.critic().parameter_values().unwrap()
            );
            let before = resumed.critic().parameter_values().unwrap();
            let p = probabilities(&resumed.policy(), &b, &[(0, 0)]);
            let mut corrupt = record.clone();
            corrupt.metadata["actor_updates"] = 999.into();
            assert!(resumed.restore(&corrupt).is_err());
            assert_eq!(before, resumed.critic().parameter_values().unwrap());
            assert_eq!(p, probabilities(&resumed.policy(), &b, &[(0, 0)]));
        }
    }
}
