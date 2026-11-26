//! Relative Entropy Pathwise Policy Optimization with fixed soft Q targets.
use super::common::{OnPolicyRecord, OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, OnPolicyActorRef, PathwiseActor, PathwiseBehavior},
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationBatch,
        learner::{Learner, OnPolicyLearner, UpdateReport},
    },
    data::{minibatch::shuffled_indices_into, rollout::OnPolicyBatch},
    nn::{
        checkpoint::RecordBurnExt,
        finite_loss, matrix,
        policy::{Policy, StochasticPolicy},
        target::frozen_copy,
        to_vec, vector,
    },
    objectives::{
        hl_gauss::HlGaussConfig,
        reppo::{
            SoftTrajectory, categorical_objective, constraint_dual_loss, kl_gated_loss,
            self_predictive_loss, soft_td_lambda,
        },
    },
};
use burn::{
    module::{Module, Param},
    optim::ModuleOptimizer,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::cell::Cell;

const MIN_LOG_MULTIPLIER: f32 = -18.;
const MAX_LOG_MULTIPLIER: f32 = 18.;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReppoConfig {
    pub policy: OptimizerConfig,
    pub critic: OptimizerConfig,
    pub entropy_dual: OptimizerConfig,
    pub kl_dual: OptimizerConfig,
    pub epochs: usize,
    pub minibatch_size: usize,
    pub lambda: f32,
    pub entropy_target: f32,
    pub initial_alpha: f32,
    pub initial_beta: f32,
    pub kl_limit: f32,
    pub hl_gauss: HlGaussConfig,
    pub auxiliary_coefficient: f32,
}
impl Default for ReppoConfig {
    fn default() -> Self {
        Self {
            policy: OptimizerConfig::default(),
            critic: OptimizerConfig {
                learning_rate: 1e-3,
                ..Default::default()
            },
            entropy_dual: OptimizerConfig::default(),
            kl_dual: OptimizerConfig::default(),
            epochs: 4,
            minibatch_size: 64,
            lambda: 0.95,
            entropy_target: -1.,
            initial_alpha: 0.1,
            initial_beta: 1.,
            kl_limit: 0.01,
            hl_gauss: HlGaussConfig::default(),
            auxiliary_coefficient: 0.,
        }
    }
}
impl ReppoConfig {
    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.critic.validate()?;
        self.entropy_dual.validate()?;
        self.kl_dual.validate()?;
        self.hl_gauss.validate()?;
        ensure(
            self.epochs > 0
                && self.minibatch_size > 0
                && self.lambda.is_finite()
                && (0.0..=1.0).contains(&self.lambda)
                && self.entropy_target.is_finite()
                && self.kl_limit.is_finite()
                && self.kl_limit > 0.
                && [self.initial_alpha, self.initial_beta].iter().all(|&x| {
                    x.is_finite()
                        && x > 0.
                        && (MIN_LOG_MULTIPLIER..=MAX_LOG_MULTIPLIER).contains(&x.ln())
                })
                && self.auxiliary_coefficient.is_finite()
                && self.auxiliary_coefficient >= 0.,
            "invalid REPPO settings or multiplier initialization",
        )
    }
}

pub use crate::nn::critics::categorical::{CategoricalQ, ReppoCritic};

#[derive(Module, Debug)]
pub(crate) struct LogMultiplier {
    pub(crate) log_value: Param<Tensor<1>>,
}
impl LogMultiplier {
    pub(crate) fn new(value: f32) -> Self {
        Self {
            log_value: Param::from_tensor(vector(&[value.ln()])),
        }
    }
    pub(crate) fn value(&self) -> f32 {
        self.log_value.val().exp().into_scalar()
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let log = self.log_value.val().into_scalar::<f32>();
        ensure(
            log.is_finite() && (MIN_LOG_MULTIPLIER..=MAX_LOG_MULTIPLIER).contains(&log),
            "invalid REPPO log multiplier",
        )
    }
    pub(crate) fn project(&mut self) {
        // Preserve parameter identity and optimizer moments while bounding exp.
        self.log_value = self.log_value.clone().map(|t| {
            t.clamp(MIN_LOG_MULTIPLIER, MAX_LOG_MULTIPLIER)
                .detach()
                .require_grad()
        });
    }
}

pub struct Reppo<P: StochasticPolicy = Policy, C: ReppoCritic = CategoricalQ> {
    policy: P,
    critic: C,
    config: ReppoConfig,
    policy_optimizer: ModuleOptimizer,
    critic_optimizer: ModuleOptimizer,
    alpha: LogMultiplier,
    beta: LogMultiplier,
    entropy_optimizer: ModuleOptimizer,
    kl_optimizer: ModuleOptimizer,
    rng: ChaCha8Rng,
    updates: u64,
    collection_revision: Cell<u64>,
}
struct ActorObjective {
    loss: Tensor<1>,
    entropy: Vec<f32>,
    kl: Vec<f32>,
}
struct AuxiliaryTargets {
    values: Vec<f32>,
    width: usize,
}
struct PreparedTargets {
    returns: Vec<f32>,
    auxiliary: Option<AuxiliaryTargets>,
}
impl<P: StochasticPolicy, C: ReppoCritic> Reppo<P, C> {
    pub fn new(policy: P, critic: C, config: ReppoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        policy.action_space().validate()?;
        ensure(
            !matches!(policy.action_space(), ActionSpace::Conditional(_))
                && critic.action_dim() == encoded_action_dim(policy.action_space())
                && critic.output_dim() == config.hl_gauss.bins,
            "REPPO critic dimensions must match action representation and HL-Gauss bins",
        )?;
        ensure(
            config.auxiliary_coefficient == 0. || critic.supports_auxiliary(),
            "REPPO critic does not provide auxiliary representations",
        )?;
        Ok(Self {
            policy,
            critic,
            policy_optimizer: config.policy.build()?,
            critic_optimizer: config.critic.build()?,
            entropy_optimizer: config.entropy_dual.build()?,
            kl_optimizer: config.kl_dual.build()?,
            alpha: LogMultiplier::new(config.initial_alpha),
            beta: LogMultiplier::new(config.initial_beta),
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
            collection_revision: Cell::new(0),
            config,
        })
    }
    pub fn config(&self) -> &ReppoConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    /// Independent frozen inference snapshot, including running statistics.
    pub fn policy(&self) -> P {
        frozen_copy(&self.policy)
    }
    pub fn critic(&self) -> C {
        frozen_copy(&self.critic)
    }
    /// Entropy alpha and forward-KL beta.
    pub fn multipliers(&self) -> (f32, f32) {
        (self.alpha.value(), self.beta.value())
    }

    fn actor_objective(
        &self,
        obs: ObservationBatch<'_>,
        behavior: &P,
        rng: &mut ChaCha8Rng,
    ) -> Result<ActorObjective> {
        let critic = frozen_copy(&self.critic);
        let current = self.policy.distribution(obs)?;
        // Distribution::kl is KL(self||other); the frozen policy must be first.
        let kl = behavior.distribution(obs)?.kl(&current)?;
        let kl_values = to_vec(kl.clone())?;
        let (alpha, beta) = self.multipliers();
        let (ordinary, entropy) =
            if matches!(self.policy.action_space(), ActionSpace::Discrete { .. }) {
                let terms = current.categorical_terms()?;
                let entropy = to_vec::<1>(
                    (terms.probabilities.clone() * terms.log_probs.clone())
                        .sum_dim(1)
                        .squeeze_dim(1)
                        .neg(),
                )?;
                let q = discrete_q_values(
                    &critic,
                    obs,
                    self.policy.action_space(),
                    &self.config.hl_gauss,
                )?;
                (
                    categorical_objective(terms.probabilities, terms.log_probs, q, alpha)?,
                    entropy,
                )
            } else {
                let sample = current.reparameterize(rng)?;
                let entropy = to_vec(sample.log_probs.clone().neg())?;
                let q = self
                    .config
                    .hl_gauss
                    .decode(critic.values(obs, sample.actions)?)?;
                (sample.log_probs * alpha - q, entropy)
            };
        Ok(ActorObjective {
            loss: kl_gated_loss(ordinary, kl, beta, self.config.kl_limit)?,
            entropy,
            kl: kl_values,
        })
    }

    fn validate_batch(&self, batch: &OnPolicyBatch) -> Result<P> {
        batch.validate()?;
        let p = batch.pathwise()?;
        p.validate_complete()?;
        ensure(
            p.revision == self.collection_revision.get(),
            "REPPO cannot reuse a rollout after a policy update",
        )?;
        let width = batch.observations[0].dense_values().len();
        for i in 0..batch.len() {
            batch.observations[i].validate(width, self.policy.action_space(), true)?;
            self.policy
                .action_space()
                .check(&batch.actions[i], batch.observations[i].dense_mask())?;
            p.next_observations[i].validate(width, self.policy.action_space(), !p.terminated[i])?;
        }
        let old = frozen_copy(&self.policy);
        let log_probs = to_vec(
            old.distribution((&batch.observations).into())?
                .log_prob(&batch.actions, &batch.latents)?,
        )?;
        ensure(
            log_probs
                .iter()
                .zip(&p.log_probs)
                .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-5 * (1. + a.abs())),
            "REPPO batch must come from its frozen current behavior policy",
        )?;
        let steps = self
            .config
            .epochs
            .checked_mul(batch.len().div_ceil(self.config.minibatch_size))
            .and_then(|x| u64::try_from(x).ok())
            .and_then(|x| self.updates.checked_add(x));
        ensure(
            steps.is_some() && self.collection_revision.get().checked_add(1).is_some(),
            "REPPO update count overflow",
        )?;
        Ok(old)
    }
    fn prepare_targets(
        &self,
        batch: &OnPolicyBatch,
        behavior: &P,
        alpha: f32,
        rng: &mut ChaCha8Rng,
    ) -> Result<PreparedTargets> {
        let p = batch.pathwise()?;
        let live: Vec<_> = (0..batch.len()).filter(|&i| !p.terminated[i]).collect();
        let mut bootstrap = vec![0.; batch.len()];
        let mut entropy = vec![0.; batch.len()];
        let mut auxiliary = None;
        if !live.is_empty() {
            let obs = ObservationBatch::selected(&p.next_observations, &live)?;
            let mut next = PathwiseBehavior::default();
            behavior
                .distribution(obs)?
                .sample_pathwise_into(rng, &mut next)?;
            let critic = frozen_copy(&self.critic);
            let q = to_vec(self.config.hl_gauss.decode(critic.values(
                obs,
                action_tensor(&next.actions, self.policy.action_space())?,
            )?)?)?;
            for (j, &i) in live.iter().enumerate() {
                bootstrap[i] = q[j];
                entropy[i] = -alpha * next.log_probs[j];
            }
            if self.config.auxiliary_coefficient > 0. {
                let continuing: Vec<_> = live
                    .iter()
                    .enumerate()
                    .filter(|&(_, &i)| !p.truncated[i])
                    .collect();
                if !continuing.is_empty() {
                    let refs: Vec<_> = continuing
                        .iter()
                        .map(|&(_, &i)| &p.next_observations[i])
                        .collect();
                    let actions: Vec<_> = continuing
                        .iter()
                        .map(|&(j, _)| next.actions[j].clone())
                        .collect();
                    let representation = critic.representation(
                        ObservationBatch::from_refs(&refs),
                        action_tensor(&actions, self.policy.action_space())?,
                    )?;
                    let [rows, width] = representation.dims();
                    ensure(
                        rows == refs.len() && width > 0,
                        "invalid REPPO auxiliary target shape",
                    )?;
                    let representation = to_vec(representation)?;
                    ensure(
                        representation.iter().all(|x| x.is_finite()),
                        "nonfinite REPPO auxiliary targets",
                    )?;
                    let count = batch.len().checked_mul(width).ok_or_else(|| {
                        crate::Error::Shape("REPPO auxiliary size overflow".into())
                    })?;
                    let mut values = vec![0.; count];
                    for (row, &(_, &i)) in continuing.iter().enumerate() {
                        values[i * width..(i + 1) * width]
                            .copy_from_slice(&representation[row * width..(row + 1) * width]);
                    }
                    auxiliary = Some(AuxiliaryTargets { values, width });
                }
            }
        }
        let returns = soft_td_lambda(
            SoftTrajectory {
                rewards: &p.rewards,
                terminated: &p.terminated,
                truncated: &p.truncated,
                bootstrap: &bootstrap,
                next_entropy: &entropy,
                lanes: p.lanes,
            },
            p.gamma,
            self.config.lambda,
        )?;
        Ok(PreparedTargets { returns, auxiliary })
    }
}

pub(crate) fn action_tensor(actions: &[Action], space: &ActionSpace) -> Result<Tensor<2>> {
    action_tensor_on_device(actions, space, &crate::nn::device())
}

pub(crate) fn action_tensor_on_device(
    actions: &[Action],
    space: &ActionSpace,
    device: &Device,
) -> Result<Tensor<2>> {
    let d = encoded_action_dim(space);
    let mut values = Vec::with_capacity(actions.len() * d);
    for action in actions {
        space.check(action, None)?;
        match action {
            Action::Continuous(a) => values.extend_from_slice(a),
            Action::Discrete(a) => {
                let start = values.len();
                values.resize(start + d, 0.);
                values[start + *a] = 1.;
            }
            Action::Conditional(_) => {
                return Err(crate::Error::Action("REPPO requires fixed actions".into()));
            }
        }
    }
    Ok(crate::nn::matrix_on_device(
        values,
        actions.len(),
        d,
        device,
    ))
}
fn encoded_action_dim(space: &ActionSpace) -> usize {
    match space {
        ActionSpace::Discrete { count } => *count,
        _ => space.dimensions(),
    }
}
fn discrete_q_values<C: ReppoCritic>(
    critic: &C,
    obs: ObservationBatch<'_>,
    space: &ActionSpace,
    support: &HlGaussConfig,
) -> Result<Tensor<2>> {
    use burn::tensor::{IndexingUpdateOp, Int, TensorData};
    let k = encoded_action_dim(space);
    let mut refs = Vec::new();
    let mut actions = Vec::new();
    let mut indices = Vec::new();
    for (row, o) in obs.iter().enumerate() {
        for action in 0..k {
            if o.dense_mask().is_none_or(|mask| mask[action]) {
                refs.push(o);
                actions.push(Action::Discrete(action));
                indices.push((row * k + action) as i64);
            }
        }
    }
    let values = support.decode(critic.values(
        ObservationBatch::from_refs(&refs),
        action_tensor(&actions, space)?,
    )?)?;
    let indices =
        Tensor::<1, Int>::from_data(TensorData::new(indices, [refs.len()]), &values.device());
    Ok(Tensor::<1>::zeros([obs.len() * k], &values.device())
        .scatter(0, indices, values, IndexingUpdateOp::Add)
        .reshape([obs.len(), k]))
}

impl<P: StochasticPolicy, C: ReppoCritic> Actor for Reppo<P, C> {
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
        self.policy
            .valid()
            .distribution(obs)?
            .sample_actions_into(mode, rng, actions)
    }
}
impl<P: StochasticPolicy, C: ReppoCritic> PathwiseActor for Reppo<P, C> {
    fn begin_collection(&self) -> Result<()> {
        let next = self
            .collection_revision
            .get()
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("REPPO collection revision overflow".into()))?;
        self.collection_revision.set(next);
        Ok(())
    }
    fn revision(&self) -> u64 {
        self.collection_revision.get()
    }
    fn sample_pathwise_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut PathwiseBehavior,
    ) -> Result<()> {
        self.policy
            .valid()
            .distribution(obs)?
            .sample_pathwise_into(rng, out)
    }
}
impl<P: StochasticPolicy, C: ReppoCritic> OnPolicyLearner for Reppo<P, C> {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Pathwise(self)
    }
}
impl<P: StochasticPolicy, C: ReppoCritic> Learner for Reppo<P, C> {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        self
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        let behavior = self.validate_batch(batch)?;
        // Stage target preparation in a local random stream until all targets are valid.
        let mut target_rng = self.rng.clone();
        let target_alpha = self.alpha.value();
        let PreparedTargets {
            returns: targets,
            auxiliary,
        } = self.prepare_targets(batch, &behavior, target_alpha, &mut target_rng)?;
        let histogram = self.config.hl_gauss.targets(&targets)?;
        self.rng = target_rng;
        let mut report = UpdateReport::default();
        let mut indices = Vec::new();
        let mut b = OnPolicyBatch::default();
        let mut selected_targets = Vec::new();
        let mut totals = [0.; 9];
        let mut observations = 0;
        for _ in 0..self.config.epochs {
            shuffled_indices_into(
                batch.len(),
                self.config.minibatch_size,
                &mut self.rng,
                &mut indices,
            )?;
            for selection in indices.chunks(self.config.minibatch_size) {
                batch.select_validated_into(selection, &mut b);
                let n = b.len();
                let obs = (&b.observations).into();
                selected_targets.clear();
                for &i in selection {
                    let start = i * self.config.hl_gauss.bins;
                    selected_targets.extend_from_slice(
                        &histogram.probabilities[start..start + self.config.hl_gauss.bins],
                    );
                }
                let logits = self
                    .critic
                    .values(obs, action_tensor(&b.actions, self.policy.action_space())?)?;
                let predicted = self
                    .config
                    .hl_gauss
                    .decode(logits.clone())?
                    .mean()
                    .into_scalar::<f32>();
                let critic_loss = self.config.hl_gauss.loss(
                    logits,
                    matrix(selected_targets.clone(), n, self.config.hl_gauss.bins),
                )?;
                let hl_loss = finite_loss(&critic_loss)?;
                let (critic_loss, auxiliary_number) = if let Some(auxiliary) = &auxiliary {
                    let mut selected = Vec::with_capacity(n * auxiliary.width);
                    for &i in selection {
                        selected.extend_from_slice(
                            &auxiliary.values[i * auxiliary.width..(i + 1) * auxiliary.width],
                        );
                    }
                    let metadata = b.pathwise()?;
                    let continuation: Vec<_> = (0..n)
                        .map(|i| {
                            if metadata.terminated[i] || metadata.truncated[i] {
                                0.
                            } else {
                                1.
                            }
                        })
                        .collect();
                    let prediction = self.critic.predict_representation(
                        obs,
                        action_tensor(&b.actions, self.policy.action_space())?,
                    )?;
                    let aux = self_predictive_loss(
                        prediction,
                        matrix(selected, n, auxiliary.width),
                        vector(&continuation),
                    )?;
                    let number = finite_loss(&aux)?;
                    (
                        critic_loss + aux * self.config.auxiliary_coefficient,
                        number,
                    )
                } else {
                    (critic_loss, 0.)
                };
                let critic_loss = optimize(
                    &mut self.critic,
                    &mut self.critic_optimizer,
                    self.config.critic.learning_rate,
                    critic_loss,
                )?;
                let mut actor_rng = self.rng.clone();
                let ActorObjective {
                    loss: actor_loss,
                    entropy,
                    kl: kl_values,
                } = self.actor_objective(obs, &behavior, &mut actor_rng)?;
                self.rng = actor_rng;
                let actor_loss = optimize(
                    &mut self.policy,
                    &mut self.policy_optimizer,
                    self.config.policy.learning_rate,
                    actor_loss,
                )?;
                let alpha_loss = constraint_dual_loss(
                    self.alpha.log_value.val(),
                    vector(
                        &entropy
                            .iter()
                            .zip(&b.observations)
                            .map(|(&h, o)| {
                                let target = match self.policy.action_space() {
                                    ActionSpace::Discrete { count } => {
                                        self.config.entropy_target.min(
                                            (o.dense_mask().map_or(*count, |mask| {
                                                mask.iter().filter(|&&x| x).count()
                                            }) as f32)
                                                .ln(),
                                        )
                                    }
                                    _ => self.config.entropy_target,
                                };
                                h - target
                            })
                            .collect::<Vec<_>>(),
                    ),
                )?;
                optimize(
                    &mut self.alpha,
                    &mut self.entropy_optimizer,
                    self.config.entropy_dual.learning_rate,
                    alpha_loss,
                )?;
                self.alpha.project();
                let beta_loss = constraint_dual_loss(
                    self.beta.log_value.val(),
                    (vector(&kl_values) - self.config.kl_limit).neg(),
                )?;
                optimize(
                    &mut self.beta,
                    &mut self.kl_optimizer,
                    self.config.kl_dual.learning_rate,
                    beta_loss,
                )?;
                self.beta.project();
                let weight = n as f64;
                totals[0] += hl_loss * weight;
                totals[7] += auxiliary_number * weight;
                totals[8] += critic_loss * weight;
                totals[1] += actor_loss * weight;
                totals[2] += f64::from(predicted) * weight;
                totals[3] += entropy.iter().map(|&x| f64::from(x)).sum::<f64>();
                totals[4] += kl_values.iter().map(|&x| f64::from(x)).sum::<f64>();
                totals[5] += kl_values
                    .iter()
                    .filter(|&&x| x >= self.config.kl_limit)
                    .count() as f64;
                totals[6] = totals[6].max(kl_values.iter().copied().fold(0., f32::max) as f64);
                observations += n;
                self.updates += 1;
                report.updates += 1;
            }
        }
        let weight = observations as f64;
        let (alpha, beta) = self.multipliers();
        self.collection_revision
            .set(self.collection_revision.get() + 1);
        Ok(report
            .metric("hl_gauss_loss", totals[0] / weight)
            .metric("auxiliary_loss", totals[7] / weight)
            .metric("critic_loss", totals[8] / weight)
            .metric("policy_loss", totals[1] / weight)
            .metric("predicted_q", totals[2] / weight)
            .metric("entropy", totals[3] / weight)
            .metric("kl_behavior_current", totals[4] / weight)
            .metric("kl_constrained_fraction", totals[5] / weight)
            .metric("kl_max", totals[6])
            .metric("target_alpha", f64::from(target_alpha))
            .metric("alpha", alpha as f64)
            .metric("beta", beta as f64)
            .metric(
                "soft_target_mean",
                targets.iter().map(|&x| f64::from(x)).sum::<f64>() / targets.len() as f64,
            )
            .metric(
                "soft_target_min",
                targets.iter().copied().fold(f32::INFINITY, f32::min) as f64,
            )
            .metric(
                "soft_target_max",
                targets.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64,
            )
            .metric(
                "hl_gauss_out_of_support_fraction",
                histogram.out_of_support as f64 / batch.len() as f64,
            )
            .metric("target_constructions", 1.))
    }
}
impl<P: StochasticPolicy, C: ReppoCritic> Checkpoint for Reppo<P, C> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new(
            "reppo",
            OnPolicyRecord {
                config: &self.config,
                rng: self.rng.clone(),
                updates: self.updates,
                space: self.policy.action_space().clone(),
            },
        )?;
        record.model("policy", &self.policy)?;
        record.metadata["collection_revision"] = serde_json::json!(self.collection_revision.get());
        record.model("critic", &self.critic)?;
        record.model("alpha", &self.alpha)?;
        record.model("beta", &self.beta)?;
        record.optimizer("policy", &self.policy_optimizer)?;
        record.optimizer("critic", &self.critic_optimizer)?;
        record.optimizer("entropy_dual", &self.entropy_optimizer)?;
        record.optimizer("kl_dual", &self.kl_optimizer)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("reppo")?;
        let metadata: OnPolicyRecord<ReppoConfig> =
            serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let collection_revision = record
            .metadata
            .get("collection_revision")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                crate::Error::State("missing or invalid REPPO collection revision".into())
            })?;
        ensure(
            &metadata.space == self.policy.action_space(),
            "REPPO checkpoint action space mismatch",
        )?;
        let policy = record.load_model("policy", &self.policy)?;
        let critic = record.load_model("critic", &self.critic)?;
        let mut restored = Self::new(policy, critic, metadata.config, 0)?;
        restored.alpha = record.load_model("alpha", &self.alpha)?;
        restored.beta = record.load_model("beta", &self.beta)?;
        restored.alpha.validate()?;
        restored.beta.validate()?;
        restored.policy_optimizer = record.load_optimizer_for_model(
            "policy",
            "policy",
            restored.config.policy.build()?,
            false,
        )?;
        restored.critic_optimizer = record.load_optimizer_for_model(
            "critic",
            "critic",
            restored.config.critic.build()?,
            false,
        )?;
        restored.entropy_optimizer = record.load_optimizer_for_model(
            "entropy_dual",
            "alpha",
            restored.config.entropy_dual.build()?,
            metadata.updates > 0,
        )?;
        restored.kl_optimizer = record.load_optimizer_for_model(
            "kl_dual",
            "beta",
            restored.config.kl_dual.build()?,
            metadata.updates > 0,
        )?;
        restored.rng = metadata.rng;
        restored.updates = metadata.updates;
        restored.collection_revision = Cell::new(collection_revision);
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::critics::scalar::QValue;
    #[test]
    fn production_actor_uses_behavior_to_current_kl_for_both_gaussian_transforms() {
        for space in [
            ActionSpace::Unbounded { dimensions: 1 },
            ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            },
        ] {
            let mut rng = ChaCha8Rng::seed_from_u64(4);
            let mut policy = Policy::new(1, space, &[], &mut rng).unwrap();
            policy.log_std = Param::from_tensor(vector(&[2_f32.ln()]));
            let mut old = frozen_copy(&policy);
            old.log_std = Param::from_tensor(vector(&[0.]));
            let old = frozen_copy(&old);
            let config = ReppoConfig::default();
            let q = QValue::new(1, 1, config.hl_gauss.bins, &[4], &mut rng).unwrap();
            let learner = Reppo::new(policy, q, config, 1).unwrap();
            let obs = vec![crate::core::env::Observation::new(vec![1.])];
            let objective = learner
                .actor_objective((&obs).into(), &old, &mut rng)
                .unwrap();
            let expected = 2_f32.ln() + 1. / 8. - 0.5;
            assert!((objective.kl[0] - expected).abs() < 1e-6);
            let reverse = -2_f32.ln() + 2. - 0.5;
            assert!((objective.kl[0] - reverse).abs() > 0.1);
        }
    }
    #[test]
    fn actor_objective_receives_critic_action_gradients_without_critic_parameter_gradients() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let policy = Policy::new(1, space, &[], &mut rng).unwrap();
        let mut critic = QValue::new(1, 1, 3, &[], &mut rng).unwrap();
        critic.network.layers[0].weight =
            Param::from_tensor(matrix(vec![0., 0., 0., -1., 0., 1.], 2, 3));
        let learner = Reppo::new(
            policy,
            critic,
            ReppoConfig {
                initial_alpha: 1e-7,
                hl_gauss: HlGaussConfig {
                    min: -1.,
                    max: 1.,
                    bins: 3,
                    sigma_ratio: 0.75,
                },
                ..Default::default()
            },
            2,
        )
        .unwrap();
        let obs = vec![crate::core::env::Observation::new(vec![1.])];
        let old = frozen_copy(&learner.policy);
        let before = learner.checkpoint().unwrap().models["critic"].clone();
        let objective = learner
            .actor_objective((&obs).into(), &old, &mut rng)
            .unwrap();
        let gradients = objective.loss.backward();
        let policy_grad = learner.policy.network.layers[0]
            .bias
            .as_ref()
            .unwrap()
            .val()
            .grad(&gradients)
            .unwrap()
            .into_scalar::<f32>();
        assert!(
            policy_grad < -1e-4,
            "a larger action has higher critic utility"
        );
        assert!(
            learner.critic.network.layers[0]
                .weight
                .val()
                .grad(&gradients)
                .is_none()
        );
        assert_eq!(
            serde_json::to_value(before).unwrap(),
            serde_json::to_value(&learner.checkpoint().unwrap().models["critic"]).unwrap()
        );
    }
}
