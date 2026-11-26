//! Experimental discrete recurrent ACER replay estimator.
//! Historical behavior, Retrace, all-action correction and EMA trust projection
//! retain their established numerical conventions. The replay update requires
//! the recorded categorical behavior law for every learning position.
use crate::algorithms::OptimizerConfig;
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        learner::UpdateReport,
    },
    data::temporal::SequenceBatch,
    nn::{temporal_discrete_q::TemporalDiscreteQ, temporal_policy::TemporalPolicy},
};
use crate::{
    core::{
        action::Action,
        ensure,
        stateful::{BehaviorPublisher, StatefulLearner, TemporalOffPolicyLearner},
    },
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer},
        optimization::optimize_validated,
        target::{frozen_copy, polyak_update},
        temporal_categorical_actor::{CategoricalMixtureSettings, TemporalCategoricalActor},
        temporal_models::{self, TemporalModelArchitecture},
        temporal_policy::TemporalStochasticPolicy,
        to_vec, vector,
    },
    objectives::acer::{RetraceStep, actor_gradient, retrace, trust_region},
};
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum AcerBatchKind {
    RecentOnPolicy,
    RecentOffPolicy,
    Replay,
}
/// Validated full-distribution replay input. Public construction deliberately
/// exposes replay updates only; the original recent/Poisson coordinator is not
/// part of this experimental API.
pub struct AcerBatch {
    recorded: crate::data::categorical_replay::CategoricalSequenceBatch,
    kind: AcerBatchKind,
}
impl AcerBatch {
    pub fn from_recorded(
        recorded: crate::data::categorical_replay::CategoricalSequenceBatch,
    ) -> Self {
        Self {
            recorded,
            kind: AcerBatchKind::Replay,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentAcerConfig {
    pub actor: OptimizerConfig,
    pub critic: OptimizerConfig,
    pub gamma: f32,
    pub trace_lambda: f64,
    pub importance_truncation: f64,
    pub trust_delta: f64,
    pub average_decay: f32,
    pub entropy: f32,
}
impl Default for RecurrentAcerConfig {
    fn default() -> Self {
        let optimizer = OptimizerConfig {
            learning_rate: 0.003,
            max_grad_norm: Some(1.),
        };
        Self {
            actor: optimizer.clone(),
            critic: optimizer,
            gamma: 1.,
            trace_lambda: 1.,
            importance_truncation: 10.,
            trust_delta: 1.,
            average_decay: 0.99,
            entropy: 0.001,
        }
    }
}
impl RecurrentAcerConfig {
    pub fn validate(&self) -> Result<()> {
        self.actor.validate()?;
        self.critic.validate()?;
        ensure(
            self.gamma.is_finite()
                && (0. ..=1.).contains(&self.gamma)
                && self.trace_lambda.is_finite()
                && (0. ..=1.).contains(&self.trace_lambda)
                && self.importance_truncation.is_finite()
                && self.importance_truncation >= 0.
                && self.trust_delta.is_finite()
                && self.trust_delta >= 0.
                && self.average_decay.is_finite()
                && (0. ..=1.).contains(&self.average_decay)
                && self.entropy.is_finite()
                && self.entropy >= 0.,
            "invalid recurrent ACER configuration",
        )
    }
}
pub struct AcerPrepared {
    pub tokens: Vec<(usize, usize)>,
    pub targets: Vec<f32>,
    pub actor_gradients: Vec<f32>,
    pub metrics: BTreeMap<String, f64>,
}
pub struct RecurrentAcer {
    policy: TemporalPolicy,
    critic: TemporalDiscreteQ,
    average: TemporalPolicy,
    config: RecurrentAcerConfig,
    policy_optimizer: ModuleOptimizer,
    critic_optimizer: ModuleOptimizer,
    updates: u64,
    actor_steps: u64,
    critic_steps: u64,
    poisoned: bool,
}
impl RecurrentAcer {
    pub fn new(
        policy: TemporalPolicy,
        critic: TemporalDiscreteQ,
        config: RecurrentAcerConfig,
    ) -> Result<Self> {
        config.validate()?;
        policy.architecture()?;
        critic.architecture()?;
        ensure(
            policy.spec() == critic.spec(),
            "ACER actor/critic histories differ",
        )?;
        temporal_models::validate_parameters(&policy, false)?;
        temporal_models::validate_parameters(&critic, false)?;
        crate::nn::optimization::validate_disjoint(&policy, &critic)?;
        let policy = policy.valid().fork(&crate::nn::device()).unfreeze().train();
        let critic = critic.valid().fork(&crate::nn::device()).unfreeze().train();
        let average = frozen_copy(&policy);
        Ok(Self {
            policy,
            critic,
            average,
            policy_optimizer: config.actor.build()?,
            critic_optimizer: config.critic.build()?,
            config,
            updates: 0,
            actor_steps: 0,
            critic_steps: 0,
            poisoned: false,
        })
    }
    pub fn policy(&self) -> TemporalPolicy {
        frozen_copy(&self.policy)
    }
    pub fn critic(&self) -> TemporalDiscreteQ {
        frozen_copy(&self.critic)
    }
    pub fn average(&self) -> TemporalPolicy {
        frozen_copy(&self.average)
    }
    pub fn config(&self) -> &RecurrentAcerConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn optimizer_calls(&self) -> (u64, u64) {
        (self.actor_steps, self.critic_steps)
    }
    /// Detached numerical quantities; no gradient, optimizer or random state changes.
    pub fn prepare(&self, batch: &AcerBatch) -> Result<AcerPrepared> {
        ensure(
            !self.poisoned,
            "ACER requires full recovery after failed mutation",
        )?;
        self.config.validate()?;
        batch.recorded.validate(self.policy.spec())?;
        ensure(
            batch
                .recorded
                .sequences
                .samples
                .iter()
                .all(|sample| sample.steps.iter().all(|step| step.reward.duration() == 1)),
            "ACER requires one primitive reward per decision",
        )?;
        let seq = &batch.recorded.sequences;
        ensure(
            seq.config.n_step == 1,
            "ACER window uses one-step endpoint eligibility, then Retrace within the window",
        )?;
        let time = seq
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let actions = self.policy.spec().action_width();
        let live = seq
            .samples
            .iter()
            .enumerate()
            .flat_map(|(r, s)| {
                (0..s.observations.len())
                    .filter(move |&t| t == 0 || !s.steps[t - 1].terminated)
                    .map(move |t| (r, t))
            })
            .collect::<Vec<_>>();
        let tokens = seq
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
        let policy = self
            .policy()
            .distribution(seq, &live)?
            .categorical_terms()?;
        let probabilities = normalize(&to_vec(policy.probabilities)?, actions)?;
        let mut p = vec![0.; seq.samples.len() * time * actions];
        for (&(r, t), row) in live.iter().zip(probabilities.chunks(actions)) {
            p[(r * time + t) * actions..(r * time + t + 1) * actions].copy_from_slice(row);
        }
        let all = seq
            .samples
            .iter()
            .enumerate()
            .flat_map(|(r, s)| (0..s.observations.len()).map(move |t| (r, t)))
            .collect::<Vec<_>>();
        let q_values = to_vec(self.critic().values(seq, &all)?)?;
        ensure(
            q_values.iter().all(|q| q.is_finite()),
            "nonfinite ACER action value",
        )?;
        let mut q = vec![0.; p.len()];
        for (&(r, t), values) in all.iter().zip(q_values.chunks(actions)) {
            for a in 0..actions {
                q[(r * time + t) * actions + a] = values[a] as f64;
            }
        }
        let avg = normalize(
            &to_vec(
                self.average
                    .distribution(seq, &tokens)?
                    .categorical_terms()?
                    .probabilities,
            )?,
            actions,
        )?;
        let mut returns = BTreeMap::new();
        for (r, s) in seq.samples.iter().enumerate() {
            let row = |table: &[f64], t: usize| {
                table[(r * time + t) * actions..(r * time + t + 1) * actions].to_vec()
            };
            let steps = s
                .steps
                .iter()
                .enumerate()
                .map(|(t, step)| RetraceStep {
                    policy: row(&p, t),
                    behavior: row(&batch.recorded.behavior, t),
                    q: row(&q, t),
                    mask: s.observations[t]
                        .dense_mask()
                        .map_or_else(|| vec![true; actions], <[bool]>::to_vec),
                    action: match step.action {
                        Action::Discrete(a) => a,
                        _ => unreachable!(),
                    },
                    reward: f64::from(step.reward.total()),
                    terminated: step.terminated,
                    truncated: step.truncated,
                })
                .collect::<Vec<_>>();
            let end = s.steps.len();
            let bootstrap = (!s.steps.last().unwrap().terminated).then(|| {
                row(&p, end)
                    .iter()
                    .zip(row(&q, end))
                    .map(|(p, q)| p * q)
                    .sum::<f64>()
            });
            for (t, value) in retrace(
                &steps,
                bootstrap,
                self.config.gamma as f64,
                self.config.trace_lambda,
            )?
            .into_iter()
            .enumerate()
            {
                returns.insert((r, t), value);
            }
        }
        let mut targets = Vec::with_capacity(tokens.len());
        let mut actor_gradients = Vec::with_capacity(tokens.len() * actions);
        let (
            mut truncations,
            mut projections,
            mut correction,
            mut dot_max,
            mut log_max,
            mut recent_error,
        ) = (0., 0., 0., f64::NEG_INFINITY, f64::NEG_INFINITY, 0f64);
        for (i, &(r, t)) in tokens.iter().enumerate() {
            let s = &seq.samples[r];
            let offset = (r * time + t) * actions;
            let pr = &p[offset..offset + actions];
            let mu = &batch.recorded.behavior[offset..offset + actions];
            let qr = &q[offset..offset + actions];
            let mask = s.observations[t]
                .dense_mask()
                .map_or_else(|| vec![true; actions], <[bool]>::to_vec);
            let Action::Discrete(a) = s.steps[t].action else {
                unreachable!()
            };
            let target = returns[&(r, t)];
            for (&p, &mu) in pr.iter().zip(mu) {
                recent_error = recent_error.max((p - mu).abs());
            }
            if batch.kind == AcerBatchKind::RecentOnPolicy {
                ensure(
                    pr.iter().zip(mu).all(|(p, mu)| (p - mu).abs() <= 2e-6),
                    "recent on-policy ACER actions were generated by another policy",
                )?;
            }
            let g = actor_gradient(
                pr,
                mu,
                qr,
                &mask,
                a,
                target,
                self.config.importance_truncation,
            )?;
            let tr = trust_region(
                pr,
                &avg[i * actions..(i + 1) * actions],
                &g.probability_gradient,
                &mask,
                self.config.trust_delta,
            )?;
            truncations += f64::from(g.truncated);
            projections += f64::from(tr.multiplier > 0.);
            correction += g.correction.iter().map(|x| x.abs()).sum::<f64>();
            dot_max = dot_max.max(tr.dot_after);
            log_max = log_max.max(g.log_importance);
            targets.push(target as f32);
            actor_gradients.extend(tr.logit_gradient.iter().map(|g| *g as f32));
        }
        ensure(
            targets
                .iter()
                .chain(&actor_gradients)
                .all(|v| v.is_finite()),
            "ACER numerical estimator exceeds f32 model range",
        )?;
        let n = tokens.len() as f64;
        Ok(AcerPrepared {
            tokens,
            targets,
            actor_gradients,
            metrics: BTreeMap::from([
                ("importance_truncation_fraction".into(), truncations / n),
                ("trust_projection_fraction".into(), projections / n),
                ("bias_correction_l1_mean".into(), correction / n),
                ("trust_linear_dot_max".into(), dot_max),
                ("log_importance_max".into(), log_max),
                ("recent_policy_difference_max".into(), recent_error),
            ]),
        })
    }
    fn actor_objective(
        &self,
        seq: &SequenceBatch,
        tokens: &[(usize, usize)],
        gradients: &[f32],
    ) -> Result<(Tensor<1>, f32)> {
        let n = tokens.len();
        let actions = self.policy.spec().action_width();
        let distribution = self.policy.distribution(seq, tokens)?;
        let crate::nn::distributions::Distribution::Categorical { logits, .. } =
            distribution.clone()
        else {
            unreachable!()
        };
        let terms = distribution.categorical_terms()?;
        let entropy = (terms.probabilities * terms.log_probs)
            .sum_dim(1)
            .mean()
            .neg();
        let entropy_value = entropy.clone().into_scalar::<f32>();
        let coefficients = Tensor::from_data(
            TensorData::new(gradients.to_vec(), [n, actions]),
            &self.policy.device(),
        );
        // Detached probability-space projection, transformed by the exact softmax
        // Jacobian, enters at logits. Entropy is deliberately added afterwards.
        let actor_loss =
            (logits * coefficients).sum().neg() / n as f32 - entropy * self.config.entropy;
        Ok((actor_loss, entropy_value))
    }
    pub fn update(&mut self, batch: &AcerBatch) -> Result<UpdateReport> {
        ensure(
            self.updates < (i32::MAX as u64) - 1,
            "ACER Adam update range exhausted",
        )?;
        let prepared = self.prepare(batch)?;
        let seq = &batch.recorded.sequences;
        let tokens = &prepared.tokens;
        let n = tokens.len();
        let (actor_loss, entropy_value) =
            self.actor_objective(seq, tokens, &prepared.actor_gradients)?;
        let chosen = tokens
            .iter()
            .map(|&(r, t)| match seq.samples[r].steps[t].action {
                Action::Discrete(a) => a as i64,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let index =
            Tensor::<2, Int>::from_data(TensorData::new(chosen, [n, 1]), &self.critic.device());
        let values = self
            .critic
            .values(seq, tokens)?
            .gather(1, index)
            .squeeze_dim(1);
        let loss = (values - vector(&prepared.targets)).powf_scalar(2.).mean() * 0.5;
        ensure(
            loss.clone().into_scalar::<f32>().is_finite(),
            "nonfinite ACER regression loss before mutation",
        )?;
        self.poisoned = true;
        let (actor_loss, actor_norm) = optimize_validated(
            &mut self.policy,
            &mut self.policy_optimizer,
            self.config.actor.learning_rate,
            actor_loss,
        )?;
        self.actor_steps += 1;
        let (critic_loss, critic_norm) = optimize_validated(
            &mut self.critic,
            &mut self.critic_optimizer,
            self.config.critic.learning_rate,
            loss,
        )?;
        self.critic_steps += 1;
        self.average = polyak_update(
            self.average.clone(),
            &self.policy,
            1. - self.config.average_decay,
        )?;
        self.updates += 1;
        temporal_models::validate_parameters(&self.policy, false)?;
        temporal_models::validate_parameters(&self.critic, false)?;
        temporal_models::validate_parameters(&self.average, true)?;
        self.poisoned = false;
        let time = seq
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let padded = (seq.samples.len() * time) as f64;
        let valid = seq
            .samples
            .iter()
            .map(|s| s.observations.len())
            .sum::<usize>() as f64;
        let mut report = UpdateReport {
            updates: 1,
            metrics: prepared.metrics,
            ..Default::default()
        };
        report.metrics.extend(BTreeMap::from([
            ("actor_loss".into(), actor_loss),
            ("critic_loss".into(), critic_loss),
            ("entropy".into(), entropy_value as f64),
            ("actor_gradient_norm".into(), actor_norm),
            ("critic_gradient_norm".into(), critic_norm),
            ("learning_positions".into(), n as f64),
            (
                "target_mean".into(),
                prepared.targets.iter().map(|t| *t as f64).sum::<f64>() / n as f64,
            ),
            ("actor_optimizer_calls".into(), 1.),
            ("critic_optimizer_calls".into(), 1.),
            ("actor_forward_tokens".into(), padded * 2.),
            ("average_forward_tokens".into(), padded),
            ("online_critic_forward_tokens".into(), padded * 2.),
            ("target_critic_forward_tokens".into(), 0.),
            ("actor_valid_history_tokens".into(), valid * 2.),
            ("average_valid_history_tokens".into(), valid),
            ("online_critic_valid_history_tokens".into(), valid * 2.),
            ("target_critic_valid_history_tokens".into(), 0.),
            (
                "recent_on_policy_update".into(),
                f64::from(batch.kind == AcerBatchKind::RecentOnPolicy),
            ),
            (
                "recent_off_policy_update".into(),
                f64::from(batch.kind == AcerBatchKind::RecentOffPolicy),
            ),
            (
                "replay_update".into(),
                f64::from(batch.kind == AcerBatchKind::Replay),
            ),
        ]));
        Ok(report)
    }
}
fn normalize(values: &[f32], actions: usize) -> Result<Vec<f64>> {
    ensure(
        !values.is_empty() && values.len().is_multiple_of(actions),
        "ACER policy table shape mismatch",
    )?;
    let mut p = Vec::with_capacity(values.len());
    for row in values.chunks(actions) {
        let sum = row.iter().map(|p| *p as f64).sum::<f64>();
        ensure(
            row.iter().all(|p| p.is_finite() && *p >= 0.) && (sum - 1.).abs() < 1e-5,
            "invalid recurrent ACER policy mass",
        )?;
        p.extend(row.iter().map(|p| *p as f64 / sum));
    }
    Ok(p)
}
impl StatefulLearner for RecurrentAcer {
    type Batch = AcerBatch;
    fn update(&mut self, b: &AcerBatch) -> Result<UpdateReport> {
        RecurrentAcer::update(self, b)
    }
}
impl TemporalOffPolicyLearner for RecurrentAcer {
    fn return_contract(&self) -> crate::core::temporal::ReturnSpec {
        crate::core::temporal::ReturnSpec {
            gamma: self.config.gamma,
            n_step: 1,
        }
    }
}
impl BehaviorPublisher for RecurrentAcer {
    type Actor = TemporalCategoricalActor;
    type Settings = CategoricalMixtureSettings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        ensure(!self.poisoned, "ACER behavior is poisoned")?;
        TemporalCategoricalActor::new(self.policy(), settings.epsilon, revision)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    version: u32,
    config: RecurrentAcerConfig,
    policy: TemporalModelArchitecture,
    critic: TemporalModelArchitecture,
    updates: u64,
    optimizer_owners: BTreeMap<String, Vec<u64>>,
}
impl Checkpoint for RecurrentAcer {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        ensure(
            !self.poisoned && self.actor_steps == self.updates && self.critic_steps == self.updates,
            "cannot checkpoint an incomplete ACER update",
        )?;
        let mut r = LearnerRecord::new("recurrent-acer-v1", ())?;
        r.model("policy", &self.policy)?;
        r.model("critic", &self.critic)?;
        r.model("average", &self.average)?;
        r.optimizer("policy", &self.policy_optimizer)?;
        r.optimizer("critic", &self.critic_optimizer)?;
        let mut optimizer_owners = BTreeMap::new();
        for name in ["policy", "critic"] {
            optimizer_owners.insert(
                name.into(),
                audit_adam_optimizer(&r, name, name, None, self.updates)?.parameter_ids,
            );
        }
        r.metadata = serde_json::to_value(Metadata {
            version: 1,
            config: self.config.clone(),
            policy: self.policy.architecture()?,
            critic: self.critic.architecture()?,
            updates: self.updates,
            optimizer_owners,
        })?;
        r.temporal_returns = Some(self.return_contract());
        Ok(r)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-acer-v1")?;
        let m: Metadata = serde_json::from_value(record.metadata.clone())?;
        m.config.validate()?;
        ensure(
            m.version == 1
                && m.policy.spec == m.critic.spec
                && m.policy.spec == *self.policy.spec()
                && record
                    .models
                    .keys()
                    .map(String::as_str)
                    .eq(["average", "critic", "policy"])
                && record
                    .optimizers
                    .keys()
                    .map(String::as_str)
                    .eq(["critic", "policy"])
                && m.optimizer_owners.keys().eq(record.optimizers.keys()),
            "invalid ACER checkpoint ownership or schema",
        )?;
        let contract = crate::core::temporal::ReturnSpec {
            gamma: m.config.gamma,
            n_step: 1,
        };
        record
            .temporal_returns
            .ok_or_else(|| crate::Error::State("missing ACER return contract".into()))?
            .check(contract)?;
        ensure(
            record.temporal_reconstruction.is_none(),
            "ACER checkpoint requires full reset ancestry",
        )?;
        for (name, arch, extra) in [
            ("policy", &m.policy, m.policy.spec.action_space.dimensions()),
            (
                "average",
                &m.policy,
                m.policy.spec.action_space.dimensions(),
            ),
            ("critic", &m.critic, 0),
        ] {
            temporal_models::validate_record_model(record, name, arch, extra)?;
        }
        for name in ["policy", "critic"] {
            let audit = audit_adam_optimizer(
                record,
                name,
                name,
                Some(&m.optimizer_owners[name]),
                m.updates,
            )?;
            ensure(
                audit.steps.values().all(|&n| n == m.updates),
                "ACER Adam owner counters differ",
            )?;
        }
        let mut rng = rand::SeedableRng::seed_from_u64(0);
        let policy = record.load_model(
            "policy",
            &TemporalPolicy::from_architecture(&m.policy, &mut rng)?,
        )?;
        let critic = record.load_model(
            "critic",
            &TemporalDiscreteQ::from_architecture(&m.critic, &mut rng)?,
        )?;
        let average = record.load_model_inference(
            "average",
            &TemporalPolicy::from_architecture_with_mode(&m.policy, &mut rng, true)?,
        )?;
        let policy_optimizer = record.load_optimizer_for_model(
            "policy",
            "policy",
            m.config.actor.build()?,
            m.updates > 0,
        )?;
        let critic_optimizer = record.load_optimizer_for_model(
            "critic",
            "critic",
            m.config.critic.build()?,
            m.updates > 0,
        )?;
        crate::nn::optimization::validate_disjoint(&policy, &critic)?;
        temporal_models::validate_parameters(&policy, false)?;
        temporal_models::validate_parameters(&critic, false)?;
        temporal_models::validate_parameters(&average, true)?;
        *self = Self {
            policy,
            critic,
            average,
            policy_optimizer,
            critic_optimizer,
            config: m.config,
            updates: m.updates,
            actor_steps: m.updates,
            critic_steps: m.updates,
            poisoned: false,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{
            action::{Action, ActionSpace},
            env::Observation,
            stateful::HistorySpec,
            temporal::BehaviorMetadata,
        },
        data::temporal::{SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep},
        nn::{matrix, to_vec, vector},
    };
    use burn::{
        module::{Module, ModuleVisitor, Param},
        tensor::Tensor,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    struct Fixture {
        sequences: SequenceBatch,
        behavior: Vec<f64>,
        kind: AcerBatchKind,
    }
    fn test_batch(f: Fixture) -> AcerBatch {
        let time = f
            .sequences
            .samples
            .iter()
            .map(|s| s.observations.len())
            .max()
            .unwrap();
        let sources = f
            .sequences
            .samples
            .iter()
            .flat_map(|s| {
                (0..time).map(move |t| {
                    (t < s.steps.len())
                        .then_some(crate::core::categorical_behavior::BehaviorSource([7; 32]))
                })
            })
            .collect();
        AcerBatch {
            recorded: crate::data::categorical_replay::CategoricalSequenceBatch {
                sequences: f.sequences,
                behavior: f.behavior,
                sources,
            },
            kind: f.kind,
        }
    }
    fn weights(m: &impl Module) -> Vec<f32> {
        struct V(Vec<f32>);
        impl ModuleVisitor for V {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                self.0.extend(to_vec(p.val()).unwrap());
            }
        }
        let mut v = V(Vec::new());
        m.visit(&mut v);
        v.0
    }
    fn make(p: [f32; 2], q: [f32; 2], constant: bool) -> RecurrentAcer {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let mut actor = TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(615),
        )
        .unwrap();
        let mut critic =
            TemporalDiscreteQ::new(spec, Some(4), &[], &mut ChaCha8Rng::seed_from_u64(629))
                .unwrap();
        if constant {
            actor.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 8], 4, 2));
            actor.head.layers[0].bias = Some(Param::from_tensor(vector(&[p[0].ln(), p[1].ln()])));
            critic.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 8], 4, 2));
            critic.head.layers[0].bias = Some(Param::from_tensor(vector(&q)));
        }
        RecurrentAcer::new(
            actor,
            critic,
            RecurrentAcerConfig {
                gamma: 0.9,
                importance_truncation: 2.,
                trust_delta: 1e12,
                entropy: 0.,
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn bandit(action: usize, mu: [f64; 2], reward: f32) -> AcerBatch {
        let o = |v, m| Observation::dense(vec![v], Some(m));
        test_batch(Fixture {
            kind: AcerBatchKind::Replay,
            behavior: vec![mu[0], mu[1], 0., 0.],
            sequences: SequenceBatch {
                config: SequenceConfig {
                    burn_in: 0,
                    learn_len: 1,
                    n_step: 1,
                },
                samples: vec![SequenceSample {
                    episode_id: 11,
                    trajectory_start: 0,
                    learn_start: 0,
                    burn_start: 0,
                    history_start: 0,
                    previous: None,
                    origin: SequenceOrigin::Reset,
                    observations: vec![o(1., vec![true; 2]), o(0., vec![false; 2])],
                    steps: vec![TemporalStep {
                        action: Action::Discrete(action),
                        reward: reward.into(),
                        terminated: true,
                        truncated: false,
                        behavior: BehaviorMetadata {
                            revision: 0,
                            log_prob: Some(mu[action].ln() as f32),
                        },
                    }],
                    loss_mask: vec![true],
                }],
            },
        })
    }
    fn sequence(terminal: bool, burn: usize) -> AcerBatch {
        let o = |x, m| Observation::dense(vec![x], Some(m));
        test_batch(Fixture {
            kind: AcerBatchKind::Replay,
            behavior: vec![0.5, 0.5, 0.5, 0.5, 0., 0.],
            sequences: SequenceBatch {
                config: SequenceConfig {
                    burn_in: burn,
                    learn_len: 2 - burn,
                    n_step: 1,
                },
                samples: vec![SequenceSample {
                    episode_id: 4,
                    trajectory_start: 0,
                    learn_start: burn,
                    burn_start: 0,
                    history_start: 0,
                    previous: None,
                    origin: SequenceOrigin::Reset,
                    observations: vec![
                        o(1., vec![true; 2]),
                        o(-1., vec![true; 2]),
                        o(2., vec![!terminal; 2]),
                    ],
                    steps: vec![
                        TemporalStep {
                            action: Action::Discrete(0),
                            reward: 0.25_f32.into(),
                            terminated: false,
                            truncated: false,
                            behavior: BehaviorMetadata {
                                revision: 0,
                                log_prob: Some(0.5f64.ln() as f32),
                            },
                        },
                        TemporalStep {
                            action: Action::Discrete(1),
                            reward: (-1.0_f32).into(),
                            terminated: terminal,
                            truncated: !terminal,
                            behavior: BehaviorMetadata {
                                revision: 0,
                                log_prob: Some(0.5f64.ln() as f32),
                            },
                        },
                    ],
                    loss_mask: vec![burn == 0, true],
                }],
            },
        })
    }
    #[test]
    fn recurrent_estimator_has_the_independent_literal_bandit_expectation() {
        let m = make([0.2, 0.8], [-0.5, 1.], true);
        let mut expectation = [0.; 2];
        for a in 0..2 {
            let p = m.prepare(&bandit(a, [0.9, 0.1], [-0.5, 1.][a])).unwrap();
            assert_eq!(p.tokens, [(0, 0)]);
            assert_eq!(p.targets, [[-0.5, 1.][a]]);
            for (j, expected) in expectation.iter_mut().enumerate() {
                *expected += [0.9, 0.1][a] * p.actor_gradients[j] as f64;
            }
        }
        assert!((expectation[0] + 0.24).abs() < 2e-7);
        assert!((expectation[1] - 0.24).abs() < 2e-7);
    }
    #[test]
    fn recurrent_retrace_targets_distinguish_terminal_truncation_and_burn_in() {
        let m = make([0.25, 0.75], [2., 4.], true);
        for (terminal, expected) in [(true, [-1.1, -1.]), (false, [1.735, 2.15])] {
            let p = m.prepare(&sequence(terminal, 0)).unwrap();
            for (a, b) in p.targets.iter().zip(expected) {
                assert!((*a as f64 - b).abs() < 1e-6);
            }
            let p = m.prepare(&sequence(terminal, 1)).unwrap();
            assert_eq!(p.tokens, [(0, 1)]);
            assert!((p.targets[0] as f64 - expected[1]).abs() < 1e-6);
        }
    }
    #[test]
    fn malformed_behavior_and_wrong_recent_policy_reject_before_mutation() {
        let mut m = make([0.2, 0.8], [-0.5, 1.], true);
        let p = weights(&m.policy());
        let q = weights(&m.critic());
        let mut b = bandit(0, [0.9, 0.1], -0.5);
        b.recorded.behavior[0] = 0.8;
        b.recorded.behavior[1] = 0.2;
        assert!(m.update(&b).is_err());
        let mut b = bandit(0, [0.9, 0.1], -0.5);
        b.kind = AcerBatchKind::RecentOnPolicy;
        assert!(m.update(&b).is_err());
        assert_eq!(m.updates(), 0);
        assert_eq!(p, weights(&m.policy()));
        assert_eq!(q, weights(&m.critic()));
        assert!(m.checkpoint().is_ok());
    }

    #[test]
    fn invalid_recorded_action_is_an_error_instead_of_a_provenance_index_panic() {
        let mut m = make([0.2, 0.8], [-0.5, 1.], true);
        let mut b = bandit(0, [0.9, 0.1], -0.5);
        b.recorded.sequences.samples[0].steps[0].action = Action::Discrete(99);
        for observation in &mut b.recorded.sequences.samples[0].observations {
            observation.as_dense_mut().unwrap().action_mask = None;
        }
        assert!(m.update(&b).is_err());
        assert_eq!(m.updates(), 0);
        assert!(m.checkpoint().is_ok());
    }
    #[test]
    fn complete_resume_keeps_both_optimizers_ema_and_the_next_update_identical() {
        let mut a = make([0.5, 0.5], [0., 0.], false);
        let b = sequence(true, 0);
        a.update(&b).unwrap();
        a.update(&b).unwrap();
        let old = weights(&a.average());
        let mut r = make([0.5, 0.5], [0., 0.], false);
        let checkpoint = a.checkpoint().unwrap();
        r.restore(&checkpoint).unwrap();
        let left = a.update(&b).unwrap();
        let right = r.update(&b).unwrap();
        assert_eq!(left.metrics, right.metrics);
        assert_eq!(weights(&a.policy()), weights(&r.policy()));
        assert_eq!(weights(&a.critic()), weights(&r.critic()));
        assert_eq!(weights(&a.average()), weights(&r.average()));
        for ((x, y), old) in weights(&a.average())
            .iter()
            .zip(weights(&a.policy()))
            .zip(old)
        {
            assert!((*x - (0.99 * old + 0.01 * y)).abs() < 2e-6);
        }
        let before = weights(&r.policy());
        let mut bad = checkpoint;
        bad.metadata["updates"] = 1000.into();
        assert!(r.restore(&bad).is_err());
        assert_eq!(before, weights(&r.policy()));
    }

    #[test]
    fn nonconstant_current_critic_and_ema_reconstruct_their_own_changed_prefixes() {
        let mut m = make([0.5, 0.5], [0., 0.], false);
        m.update(&sequence(true, 0)).unwrap();
        m.update(&sequence(true, 0)).unwrap();
        m.config.trust_delta = 0.;
        let mut recorded = Vec::new();
        for first in [1., -3.] {
            let mut b = sequence(false, 1);
            b.recorded.sequences.samples[0].observations[0]
                .as_dense_mut()
                .unwrap()
                .values[0] = first;
            let positions = [(0, 1), (0, 2)];
            let row = |policy: &TemporalPolicy, positions: &[(usize, usize)]| {
                let values = to_vec(
                    policy
                        .distribution(&b.recorded.sequences, positions)
                        .unwrap()
                        .categorical_terms()
                        .unwrap()
                        .probabilities,
                )
                .unwrap();
                values
                    .chunks(2)
                    .flat_map(|v| {
                        let total = v.iter().map(|x| *x as f64).sum::<f64>();
                        v.iter().map(move |x| *x as f64 / total)
                    })
                    .collect::<Vec<_>>()
            };
            let p = row(&m.policy(), &positions);
            let avg = row(&m.average(), &[(0, 1)]);
            let q = to_vec(
                m.critic()
                    .values(&b.recorded.sequences, &positions)
                    .unwrap(),
            )
            .unwrap()
            .into_iter()
            .map(|x| x as f64)
            .collect::<Vec<_>>();
            assert_ne!(&p[..2], &avg);
            // The scored step is truncated: its continuation is current pi*Q
            // at the true endpoint after assimilating the SAME factual prefix.
            let target = -1. + (m.config.gamma as f64) * (p[2] * q[2] + p[3] * q[3]);
            let g =
                actor_gradient(&p[..2], &[0.5, 0.5], &q[..2], &[true; 2], 1, target, 2.).unwrap();
            let expected =
                trust_region(&p[..2], &avg, &g.probability_gradient, &[true; 2], 0.).unwrap();
            assert!(expected.multiplier > 0.);
            let prepared = m.prepare(&b).unwrap();
            assert_eq!(prepared.tokens, [(0, 1)]);
            assert!((prepared.targets[0] as f64 - target).abs() < 1e-6);
            for (a, b) in prepared.actor_gradients.iter().zip(expected.logit_gradient) {
                assert!((*a as f64 - b).abs() < 2e-6);
            }
            recorded.push((prepared.targets[0], prepared.actor_gradients));
        }
        assert_ne!(
            recorded[0], recorded[1],
            "dropping the earlier public reading must change this fixture"
        );
    }

    #[test]
    fn active_probability_projection_mask_and_entropy_have_literal_neural_bias_gradients() {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 3 }).unwrap();
        let mut actor = TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(71),
        )
        .unwrap();
        actor.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 12], 4, 3));
        actor.head.layers[0].bias = Some(Param::from_tensor(vector(&[
            0.25f32.ln(),
            0.75f32.ln(),
            100.,
        ])));
        let mut critic =
            TemporalDiscreteQ::new(spec, Some(4), &[], &mut ChaCha8Rng::seed_from_u64(72)).unwrap();
        critic.head.layers[0].weight = Param::from_tensor(matrix(vec![0.; 12], 4, 3));
        critic.head.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0., 999.])));
        let mut m = RecurrentAcer::new(
            actor,
            critic,
            RecurrentAcerConfig {
                importance_truncation: 2.,
                trust_delta: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        m.average.head.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0., 100.])));
        m.average = frozen_copy(&m.average);
        let mut b = bandit(0, [0.5, 0.5], -1.5);
        b.recorded.behavior = vec![0.5, 0.5, 0., 0., 0., 0.];
        b.recorded.sequences.samples[0].observations[0]
            .as_dense_mut()
            .unwrap()
            .action_mask = Some(vec![true, true, false]);
        b.recorded.sequences.samples[0].observations[1]
            .as_dense_mut()
            .unwrap()
            .action_mask = Some(vec![false; 3]);
        let prepared = m.prepare(&b).unwrap();
        assert_eq!(prepared.metrics["trust_projection_fraction"], 1.);
        assert!((prepared.actor_gradients[0] as f64 + 0.230625).abs() < 2e-6);
        let (loss, _) = m
            .actor_objective(
                &b.recorded.sequences,
                &prepared.tokens,
                &prepared.actor_gradients,
            )
            .unwrap();
        let gradients = loss.backward();
        let bias = m.policy.head.layers[0].bias.as_ref().unwrap().val();
        let actual = to_vec(bias.grad(&gradients).unwrap()).unwrap();
        let h = -0.25f64 * 0.25f64.ln() - 0.75f64 * 0.75f64.ln();
        let expected = 0.230625 + 0.001 * 0.25 * (0.25f64.ln() + h);
        assert!(
            (actual[0] as f64 - expected).abs() < 2e-6,
            "actual={} expected={expected}",
            actual[0]
        );
        assert!((actual[1] as f64 + expected).abs() < 2e-6);
        assert_eq!(actual[2], 0.);
        // Omitting entropy differs by >2e-4; putting it before probability-space
        // projection differs by >1e-5. Neither passes the independent tolerance.
    }

    #[test]
    fn failed_critic_gradient_retains_the_already_completed_actor_optimizer_call() {
        let mut m = make([0.5, 0.5], [0., 0.], true);
        let error = m
            .update(&bandit(0, [0.5, 0.5], 1.4e19))
            .expect_err("critic raw gradient must exceed native f32 norm guard");
        assert!(
            error.to_string().contains("gradients exceed safe f32"),
            "{error}"
        );
        assert_eq!(m.updates(), 0);
        assert_eq!(m.optimizer_calls(), (1, 0));
        assert!(m.checkpoint().is_err());
    }
}
