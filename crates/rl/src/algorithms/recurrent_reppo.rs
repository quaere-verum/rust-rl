//! Pathwise temporal policy improvement with fixed, independently reconstructed Q targets.
use super::{
    recurrent_on_policy::{
        checked_update_capacity,
        minibatch::{Minibatch, WindowBatch},
    },
    reppo::{LogMultiplier, ReppoConfig, action_tensor},
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::{TemporalLearnerBinding, TemporalOnPolicyLearner};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::PathwiseBehavior,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::Observation,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner},
        temporal::TemporalExecutionMode,
    },
    data::{
        minibatch::shuffled_indices_into,
        returns::DiscountedRewards,
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    },
    nn::{
        distributions::{CategoricalTerms, Distribution},
        finite_loss, matrix, matrix_on_device,
        optimization::{optimize, optimize_validated},
        target::frozen_copy,
        temporal_models,
        temporal_pathwise_actor::TemporalPathwiseActor,
        temporal_policy::TemporalPolicy,
        temporal_q::TemporalQ,
        to_vec, vector,
    },
    objectives::{
        hl_gauss::{HlGaussConfig, HlGaussTargets},
        reppo::{
            SoftTrajectory, categorical_objective, constraint_dual_loss, kl_gated_loss,
            self_predictive_loss, soft_td_lambda_with,
        },
    },
};
#[cfg(test)]
use crate::{data::temporal::SequenceBatch, nn::temporal_policy::TemporalStochasticPolicy};
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{Int, Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, de::DeserializeOwned};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

mod checkpoint;
mod dense;
mod structured;
mod structured_actor;
pub use structured_actor::{StructuredPathwiseActor, StructuredPathwiseActorState};

type Batch<A = TemporalPathwiseActor> = TemporalOnPolicyBatch<A>;
type ReppoWindow<'a, A = TemporalPathwiseActor> = Minibatch<'a, A>;

struct CheckedHistory<'a, A: Clone> {
    windows: WindowBatch<'a, A>,
    full: ReppoWindow<'a, A>,
    windows_per_batch: usize,
}

/// Model-specific execution for the shared recurrent REPPO schedule.
#[doc(hidden)]
pub trait RecurrentReppoPolicy: Module {
    type Critic: Module;
    type Actor: OnPolicyIdentity + Snapshot + Clone;
    type PolicyArchitecture: PartialEq + Serialize + DeserializeOwned;
    type CriticArchitecture: PartialEq + Serialize + DeserializeOwned;
    type CriticInput;

    const RECORD_VERSION: u32;
    const SPARSE_ADAM: bool = false;
    const REQUIRED_POLICY_ADAM_PATHS: &'static [&'static str] = &[];
    const REQUIRED_CRITIC_ADAM_PATHS: &'static [&'static str] = &[];

    fn architectures(
        &self,
        critic: &Self::Critic,
    ) -> Result<(Self::PolicyArchitecture, Self::CriticArchitecture)>;
    fn validate_pair(&self, critic: &Self::Critic, config: &ReppoConfig) -> Result<()>;
    fn validate_binding(
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
        binding: &TemporalLearnerBinding,
        config: &ReppoConfig,
    ) -> Result<()>;
    fn validate_record_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<()>;
    fn restore_models(
        record: &LearnerRecord,
        policy: &Self::PolicyArchitecture,
        critic: &Self::CriticArchitecture,
    ) -> Result<(Self, Self::Critic)>;
    fn publish(policy: Self, critic: Self::Critic, revision: u64) -> Result<Self::Actor>;
    fn published_models(actor: &Self::Actor) -> (&Self, &Self::Critic);
    fn with_revision(actor: &Self::Actor, revision: u64) -> Self::Actor;
    fn binding(actor: &Self::Actor) -> Result<TemporalLearnerBinding>;
    /// Whether these models can use the optimized exact learning path.
    fn supports_optimized_exact(&self, _critic: &Self::Critic) -> bool {
        false
    }

    fn windows<'a>(
        &self,
        critic: &Self::Critic,
        batch: &'a Batch<Self::Actor>,
        mode: TemporalExecutionMode,
    ) -> Result<WindowBatch<'a, Self::Actor>>;
    fn audit_recordings(
        batch: &Batch<Self::Actor>,
        full: &ReppoWindow<'_, Self::Actor>,
    ) -> Result<()>;
    fn prepare_targets(
        learner: &RecurrentReppo<Self>,
        batch: &Batch<Self::Actor>,
        full: &ReppoWindow<'_, Self::Actor>,
        rng: &mut ChaCha8Rng,
    ) -> Result<PreparedTargets>;
    fn validate_update_queries(
        _learner: &RecurrentReppo<Self>,
        _batch: &Batch<Self::Actor>,
        _windows_per_batch: usize,
        _rng: &ChaCha8Rng,
    ) -> Result<()> {
        Ok(())
    }
    fn critic_input(
        critic: &Self::Critic,
        window: &ReppoWindow<'_, Self::Actor>,
        tokens: &[(usize, usize)],
    ) -> Result<Self::CriticInput>;
    fn critic_logits(critic: &Self::Critic, input: &Self::CriticInput) -> Result<Tensor<2>>;
    fn predict_representation(critic: &Self::Critic, input: Self::CriticInput)
    -> Result<Tensor<2>>;
    fn actor_for_window(
        learner: &RecurrentReppo<Self>,
        window: &ReppoWindow<'_, Self::Actor>,
        tokens: &[(usize, usize)],
        observations: &[&Observation],
        rng: &mut ChaCha8Rng,
    ) -> Result<ActorObjective>;
    fn entropy_support(&self, observation: &Observation) -> Option<usize>;
}

pub struct RecurrentReppo<P: RecurrentReppoPolicy = TemporalPolicy> {
    policy: P,
    critic: P::Critic,
    config: ReppoConfig,
    update_mode: TemporalExecutionMode,
    policy_optimizer: ModuleOptimizer,
    critic_optimizer: ModuleOptimizer,
    alpha: LogMultiplier,
    beta: LogMultiplier,
    entropy_optimizer: ModuleOptimizer,
    kl_optimizer: ModuleOptimizer,
    rng: ChaCha8Rng,
    updates: u64,
    auxiliary_updates: u64,
    revision: Cell<u64>,
    collection_begins: Cell<u64>,
    collection_open: Cell<bool>,
    completed_batches: u64,
    consumed_revision: Option<u64>,
    poisoned: bool,
    // Policy and critic step together; revision-only collection starts reuse
    // frozen weights. This reconstructible cache never enters learner records.
    behavior_cache: RefCell<Option<(u64, P::Actor)>>,
}
#[doc(hidden)]
pub struct ActorObjective {
    loss: Tensor<1>,
    entropy: Vec<f32>,
    kl: Vec<f32>,
}
struct AuxiliaryTargets {
    values: Vec<f32>,
    width: usize,
}
#[doc(hidden)]
pub struct PreparedTargets {
    returns: Vec<f32>,
    auxiliary: Option<AuxiliaryTargets>,
}

impl RecurrentReppo {
    pub fn new(
        policy: TemporalPolicy,
        critic: TemporalQ,
        config: ReppoConfig,
        seed: u64,
    ) -> Result<Self> {
        Self::new_models(policy, critic, config, seed)
    }
}

impl<P: RecurrentReppoPolicy> RecurrentReppo<P> {
    fn new_models(policy: P, critic: P::Critic, config: ReppoConfig, seed: u64) -> Result<Self> {
        validate_models(&policy, &critic, &config)?;
        Ok(Self {
            policy: policy.valid().fork(&crate::nn::device()).unfreeze().train(),
            critic: critic.valid().fork(&crate::nn::device()).unfreeze().train(),
            policy_optimizer: config.policy.build()?,
            critic_optimizer: config.critic.build()?,
            entropy_optimizer: config.entropy_dual.build()?,
            kl_optimizer: config.kl_dual.build()?,
            alpha: LogMultiplier::new(config.initial_alpha),
            beta: LogMultiplier::new(config.initial_beta),
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
            auxiliary_updates: 0,
            revision: Cell::new(0),
            collection_begins: Cell::new(0),
            collection_open: Cell::new(false),
            completed_batches: 0,
            consumed_revision: None,
            poisoned: false,
            behavior_cache: RefCell::new(None),
            config,
            update_mode: TemporalExecutionMode::Reference,
        })
    }
    pub fn config(&self) -> &ReppoConfig {
        &self.config
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.policy)
    }
    pub fn critic(&self) -> P::Critic {
        frozen_copy(&self.critic)
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn revision(&self) -> u64 {
        self.revision.get()
    }
    /// The execution path used by the current model adapter.
    pub fn update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
            .effective(self.policy.supports_optimized_exact(&self.critic))
    }

    /// The saved request, including when it falls back to reference execution.
    pub fn requested_update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
    }
    /// Whether this learner can execute the requested mode.
    pub fn supports_update_mode(&self, mode: TemporalExecutionMode) -> bool {
        mode.supported(self.policy.supports_optimized_exact(&self.critic))
    }

    /// Request a mode. Unsupported requests use reference execution.
    pub fn set_update_mode(&mut self, mode: TemporalExecutionMode) {
        self.update_mode = mode;
    }
    pub fn multipliers(&self) -> (f32, f32) {
        (self.alpha.value(), self.beta.value())
    }
    fn actor(&self) -> Result<P::Actor> {
        ensure(
            !self.poisoned,
            "recurrent REPPO requires checkpoint recovery",
        )?;
        if let Some((updates, actor)) = self.behavior_cache.borrow().as_ref()
            && *updates == self.updates
        {
            return Ok(P::with_revision(actor, self.revision()));
        }
        let actor = P::publish(self.policy.clone(), self.critic.clone(), self.revision())?;
        self.behavior_cache
            .replace(Some((self.updates, actor.clone())));
        Ok(actor)
    }
    fn preflight<'a>(&self, batch: &'a Batch<P::Actor>) -> Result<CheckedHistory<'a, P::Actor>> {
        ensure(
            !self.poisoned,
            "recurrent REPPO requires checkpoint recovery",
        )?;
        self.config.validate()?;
        batch.validate_snapshot()?;
        ensure(
            self.collection_open.get()
                && batch.kind == TemporalBatchKind::Pathwise
                && batch.revision == self.revision()
                && self
                    .consumed_revision
                    .is_none_or(|revision| revision < batch.revision)
                && self.revision().checked_add(1).is_some()
                && self.completed_batches.checked_add(1).is_some(),
            "stale recurrent REPPO batch or exhausted revision",
        )?;
        let actor = self.actor()?;
        let (policy, critic) = P::published_models(&batch.behavior);
        ensure(
            batch.behavior.behavior_identity() == actor.behavior_identity()
                && policy.architectures(critic)? == self.policy.architectures(&self.critic)?,
            "recurrent REPPO behavior differs from collection models",
        )?;
        ensure(
            batch.sequences.config.learn_len <= self.config.minibatch_size,
            "recurrent REPPO learning window exceeds minibatch learning-position limit",
        )?;
        let windows_per_batch =
            (self.config.minibatch_size / batch.sequences.config.learn_len).max(1);
        checked_update_capacity(
            self.updates,
            self.config.epochs,
            batch.sequences.samples.len(),
            windows_per_batch,
        )?;
        let windows = policy.windows(critic, batch, self.update_mode())?;
        let full = windows.full()?;
        P::audit_recordings(batch, &full)?;
        Ok(CheckedHistory {
            windows,
            full,
            windows_per_batch,
        })
    }
    /// Uses precisely the next update's target noise without consuming learner state.
    pub fn target_preview(&self, batch: &Batch<P::Actor>) -> Result<Vec<f32>> {
        let history = self.preflight(batch)?;
        Ok(P::prepare_targets(self, batch, &history.full, &mut self.rng.clone())?.returns)
    }
}

impl RecurrentReppo {
    fn prepare_sampled_targets(
        &self,
        batch: &Batch,
        full: &ReppoWindow<'_>,
        rng: &mut ChaCha8Rng,
    ) -> Result<PreparedTargets> {
        let live: Vec<_> = batch
            .rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !batch.sequences.samples[r.sample].steps[r.time].terminated)
            .map(|(i, _)| i)
            .collect();
        let mut bootstrap = vec![0.; batch.len()];
        let mut entropy = vec![0.; batch.len()];
        let mut auxiliary = None;
        if !live.is_empty() {
            let tokens: Vec<_> = live
                .iter()
                .map(|&i| {
                    let r = &batch.rows[i];
                    (r.sample, r.time + 1)
                })
                .collect();
            let mut next = PathwiseBehavior::default();
            full.distribution(batch.behavior.policy(), &tokens)?
                .sample_pathwise_into(rng, &mut next)?;
            let critic = batch.behavior.critic();
            let history = full.q_history(critic, &tokens)?;
            let actions = action_tensor(&next.actions, &self.policy.spec().action_space)?
                .to_device(&critic.device());
            let q = to_vec(
                self.config
                    .hl_gauss
                    .decode(critic.values_from_features(history.clone(), actions.clone())?)?,
            )?;
            for (j, &i) in live.iter().enumerate() {
                bootstrap[i] = q[j];
                entropy[i] = -self.alpha.value() * next.log_probs[j];
            }
            if self.config.auxiliary_coefficient > 0. {
                let continuing: Vec<_> = live
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| {
                        let r = &batch.rows[**i];
                        !batch.sequences.samples[r.sample].steps[r.time].truncated
                    })
                    .map(|(j, &i)| (j, i))
                    .collect();
                if !continuing.is_empty() {
                    let selection = indices(
                        &continuing.iter().map(|&(j, _)| j).collect::<Vec<_>>(),
                        &critic.device(),
                    );
                    let representation = critic.representation_from_features(
                        history.select(0, selection.clone()),
                        actions.select(0, selection),
                    )?;
                    let [n, width] = representation.dims();
                    ensure(
                        n == continuing.len() && width > 0,
                        "invalid recurrent REPPO auxiliary targets",
                    )?;
                    let representation = to_vec(representation)?;
                    ensure(
                        representation.iter().all(|x| x.is_finite()),
                        "nonfinite recurrent REPPO auxiliary targets",
                    )?;
                    let count = batch.len().checked_mul(width).ok_or_else(|| {
                        Error::Shape("recurrent REPPO auxiliary size overflow".into())
                    })?;
                    let mut values = vec![0.; count];
                    for (j, &(_, i)) in continuing.iter().enumerate() {
                        values[i * width..(i + 1) * width]
                            .copy_from_slice(&representation[j * width..(j + 1) * width]);
                    }
                    auxiliary = Some(AuxiliaryTargets { values, width });
                }
            }
        }
        let returns = recurrent_soft_targets(batch, &bootstrap, &entropy, self.config.lambda)?;
        Ok(PreparedTargets { returns, auxiliary })
    }
    #[cfg(test)]
    fn actor_objective(
        &self,
        batch: &SequenceBatch,
        tokens: &[(usize, usize)],
        behavior: &Distribution,
        rng: &mut ChaCha8Rng,
    ) -> Result<ActorObjective> {
        let current = self.policy.distribution(batch, tokens)?;
        let observations: Vec<_> = tokens
            .iter()
            .map(|&(sample, time)| &batch.samples[sample].observations[time])
            .collect();
        self.actor_objective_from_window(current, &observations, behavior, rng, |critic| {
            critic.history_features(batch, tokens)
        })
    }

    fn actor_objective_from_window(
        &self,
        current: Distribution,
        observations: &[&Observation],
        behavior: &Distribution,
        rng: &mut ChaCha8Rng,
        reconstruct: impl FnOnce(&TemporalQ) -> Result<Tensor<2>>,
    ) -> Result<ActorObjective> {
        let (alpha, beta) = self.multipliers();
        gated_actor_objective(behavior.kl(&current)?, beta, self.config.kl_limit, || {
            // Freeze only parameters so Gaussian action gradients traverse the
            // updated critic head. Its history uses its own encoder.
            let critic = frozen_copy(&self.critic);
            let history = reconstruct(&critic)?;
            if matches!(
                self.policy.spec().action_space,
                ActionSpace::Discrete { .. }
            ) {
                let terms = current.categorical_terms()?;
                let q = discrete_q_values(
                    &critic,
                    history,
                    observations,
                    &self.config.hl_gauss,
                    &self.policy.device(),
                )?;
                let (ordinary, entropy) = categorical_actor_terms(terms, q, alpha)?;
                Ok((ordinary, to_vec(entropy)?))
            } else {
                let sample = current.reparameterize(rng)?;
                let entropy = to_vec(sample.log_probs.clone().neg())?;
                let q = self
                    .config
                    .hl_gauss
                    .decode(critic.values_from_features(history, sample.actions)?)?;
                Ok((sample.log_probs * alpha - q, entropy))
            }
        })
    }
}

impl<P: RecurrentReppoPolicy> RecurrentReppo<P> {
    fn update_inner(
        &mut self,
        batch: &Batch<P::Actor>,
        history: &WindowBatch<'_, P::Actor>,
        windows_per_batch: usize,
        targets: PreparedTargets,
        histogram: HlGaussTargets,
        critic_stepped: &mut bool,
    ) -> Result<UpdateReport> {
        let positions: BTreeMap<_, _> = batch
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| (row.key, i))
            .collect();
        let mut report = UpdateReport::default();
        let mut shuffled = Vec::new();
        let mut totals = [0f64; 9];
        let mut observations = 0usize;
        #[cfg(feature = "probes")]
        let mut gradient_norms = crate::profiling::enabled().then_some([0.; 2]);
        for _ in 0..self.config.epochs {
            shuffled_indices_into(
                history.window_count(),
                windows_per_batch,
                &mut self.rng,
                &mut shuffled,
            )?;
            for windows in shuffled.chunks(windows_per_batch) {
                let selected = history.select(windows)?;
                let tokens = selected.tokens();
                let n = selected.len();
                let selection: Vec<_> = selected.rows().map(|(_, r)| positions[&r.key]).collect();
                let selected_histogram = select_rows(
                    &histogram.probabilities,
                    self.config.hl_gauss.bins,
                    &selection,
                );
                let critic_input = P::critic_input(&self.critic, &selected, &tokens)?;
                let logits = P::critic_logits(&self.critic, &critic_input)?;
                let predicted = self
                    .config
                    .hl_gauss
                    .decode(logits.clone())?
                    .mean()
                    .into_scalar::<f32>();
                ensure(
                    predicted.is_finite(),
                    "nonfinite recurrent REPPO predicted Q",
                )?;
                let critic_loss = self.config.hl_gauss.loss(
                    logits,
                    matrix(selected_histogram, n, self.config.hl_gauss.bins),
                )?;
                let hl_loss = finite_loss(&critic_loss)?;
                let (critic_loss, auxiliary_number) = if let Some(auxiliary) = &targets.auxiliary {
                    let continuation: Vec<_> = selected
                        .rows()
                        .map(|(sample, r)| {
                            let s = &selected.sample(sample).steps[r.time];
                            if s.terminated || s.truncated { 0. } else { 1. }
                        })
                        .collect();
                    let prediction = P::predict_representation(&self.critic, critic_input)?;
                    let aux = self_predictive_loss(
                        prediction,
                        matrix(
                            select_rows(&auxiliary.values, auxiliary.width, &selection),
                            n,
                            auxiliary.width,
                        ),
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
                let (critic_loss, _critic_grad_norm) = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::CriticOptimization);
                    optimize_validated(
                        &mut self.critic,
                        &mut self.critic_optimizer,
                        self.config.critic.learning_rate,
                        critic_loss,
                    )?
                };
                *critic_stepped = true;
                let mut actor_rng = self.rng.clone();
                let observations_selected = selected.observations(&tokens);
                let ActorObjective { loss, entropy, kl } = P::actor_for_window(
                    self,
                    &selected,
                    &tokens,
                    &observations_selected,
                    &mut actor_rng,
                )?;
                self.rng = actor_rng;
                let (actor_loss, _actor_grad_norm) = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::ActorOptimization);
                    optimize_validated(
                        &mut self.policy,
                        &mut self.policy_optimizer,
                        self.config.policy.learning_rate,
                        loss,
                    )?
                };
                let support = observations_selected
                    .iter()
                    .map(|observation| self.policy.entropy_support(observation));
                let alpha_loss = entropy_dual_objective(
                    self.alpha.log_value.val(),
                    entropy.iter().copied().zip(support),
                    self.config.entropy_target,
                )?;
                {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::DualOptimization);
                    optimize(
                        &mut self.alpha,
                        &mut self.entropy_optimizer,
                        self.config.entropy_dual.learning_rate,
                        alpha_loss,
                    )?;
                }
                self.alpha.project();
                let beta_loss =
                    kl_dual_objective(self.beta.log_value.val(), &kl, self.config.kl_limit)?;
                {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::DualOptimization);
                    optimize(
                        &mut self.beta,
                        &mut self.kl_optimizer,
                        self.config.kl_dual.learning_rate,
                        beta_loss,
                    )?;
                }
                self.beta.project();
                let weight = n as f64;
                totals[0] += hl_loss * weight;
                totals[1] += actor_loss * weight;
                totals[2] += predicted as f64 * weight;
                totals[3] += entropy.iter().map(|&x| x as f64).sum::<f64>();
                totals[4] += kl.iter().map(|&x| x as f64).sum::<f64>();
                totals[5] += kl.iter().filter(|&&x| x >= self.config.kl_limit).count() as f64;
                totals[6] = totals[6].max(kl.iter().copied().fold(0., f32::max) as f64);
                totals[7] += auxiliary_number * weight;
                totals[8] += critic_loss * weight;
                #[cfg(feature = "probes")]
                if let Some(norms) = &mut gradient_norms {
                    norms[0] += _critic_grad_norm * weight;
                    norms[1] += _actor_grad_norm * weight;
                }
                observations += n;
                self.updates += 1;
                self.auxiliary_updates += u64::from(targets.auxiliary.is_some());
                report.updates += 1;
            }
        }
        validate_models(&self.policy, &self.critic, &self.config)?;
        self.alpha.validate()?;
        self.beta.validate()?;
        self.completed_batches += 1;
        self.consumed_revision = Some(batch.revision);
        self.collection_open.set(false);
        self.revision.set(self.revision() + 1);
        let weight = observations as f64;
        let (alpha, beta) = self.multipliers();
        #[cfg(feature = "probes")]
        if let Some([critic, actor]) = gradient_norms {
            report = report
                .metric("critic_unclipped_gradient_norm", critic / weight)
                .metric("actor_unclipped_gradient_norm", actor / weight);
        }
        Ok(report
            .metric("hl_gauss_loss", totals[0] / weight)
            .metric("policy_loss", totals[1] / weight)
            .metric("predicted_q", totals[2] / weight)
            .metric("entropy", totals[3] / weight)
            .metric("kl_behavior_current", totals[4] / weight)
            .metric("kl_constrained_fraction", totals[5] / weight)
            .metric("kl_max", totals[6])
            .metric("auxiliary_loss", totals[7] / weight)
            .metric("critic_loss", totals[8] / weight)
            .metric("alpha", alpha as f64)
            .metric("beta", beta as f64)
            .metric(
                "soft_target_mean",
                targets.returns.iter().map(|&x| x as f64).sum::<f64>() / batch.len() as f64,
            )
            .metric(
                "soft_target_min",
                targets
                    .returns
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min) as f64,
            )
            .metric(
                "soft_target_max",
                targets
                    .returns
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max) as f64,
            )
            .metric(
                "hl_gauss_out_of_support_fraction",
                histogram.out_of_support as f64 / batch.len() as f64,
            )
            .metric("target_constructions", 1.))
    }
}

fn recurrent_soft_targets<A>(
    batch: &TemporalOnPolicyBatch<A>,
    bootstrap: &[f32],
    entropy: &[f32],
    lambda: f32,
) -> Result<Vec<f32>> {
    let count = batch.rows.len();
    ensure(
        bootstrap.len() == count && entropy.len() == count,
        "recurrent REPPO successor target count differs from the rollout",
    )?;
    // A chronological fragment can span overlapping windows. Only adjacent
    // fresh decisions in one episode participate in the same recurrence.
    let mut returns = Vec::with_capacity(count);
    let mut start = 0;
    while start < count {
        let mut end = start + 1;
        while end < count
            && batch.rows[end].key.episode_id == batch.rows[end - 1].key.episode_id
            && batch.rows[end].key.timestep == batch.rows[end - 1].key.timestep + 1
        {
            end += 1;
        }
        let steps = batch.rows[start..end]
            .iter()
            .map(|row| &batch.sequences.samples[row.sample].steps[row.time]);
        let rewards = DiscountedRewards::new(steps.clone().map(|step| &step.reward), batch.gamma)?;
        let terminated: Vec<_> = steps.clone().map(|step| step.terminated).collect();
        let truncated: Vec<_> = steps.map(|step| step.truncated).collect();
        returns.extend(soft_td_lambda_with(
            SoftTrajectory {
                rewards: &rewards.rewards,
                terminated: &terminated,
                truncated: &truncated,
                bootstrap: &bootstrap[start..end],
                next_entropy: &entropy[start..end],
                lanes: 1,
            },
            rewards.discounts(),
            lambda,
        )?);
        start = end;
    }
    Ok(returns)
}

fn categorical_actor_terms(
    terms: CategoricalTerms,
    q: Tensor<2>,
    alpha: f32,
) -> Result<(Tensor<1>, Tensor<1>)> {
    let ordinary = categorical_objective(
        terms.probabilities.clone(),
        terms.log_probs.clone(),
        q,
        alpha,
    )?;
    let entropy = (terms.probabilities * terms.log_probs)
        .sum_dim(1)
        .squeeze_dim(1)
        .neg();
    Ok((ordinary, entropy))
}

fn gated_actor_objective(
    kl: Tensor<1>,
    beta: f32,
    limit: f32,
    evaluate: impl FnOnce() -> Result<(Tensor<1>, Vec<f32>)>,
) -> Result<ActorObjective> {
    let kl_values = to_vec(kl.clone())?;
    ensure(
        kl_values.iter().all(|x| x.is_finite()),
        "nonfinite recurrent REPPO KL",
    )?;
    let (ordinary, entropy) = evaluate()?;
    ensure(
        entropy.iter().all(|x| x.is_finite()),
        "nonfinite recurrent REPPO entropy",
    )?;
    Ok(ActorObjective {
        loss: kl_gated_loss(ordinary, kl, beta, limit)?,
        entropy,
        kl: kl_values,
    })
}

fn entropy_dual_objective(
    log_multiplier: Tensor<1>,
    measurements: impl Iterator<Item = (f32, Option<usize>)>,
    target: f32,
) -> Result<Tensor<1>> {
    let slack: Vec<_> = measurements
        .map(|(entropy, support)| {
            let target = support.map_or(target, |count| target.min((count as f32).ln()));
            entropy - target
        })
        .collect();
    constraint_dual_loss(log_multiplier, vector(&slack))
}

fn kl_dual_objective(log_multiplier: Tensor<1>, kl: &[f32], limit: f32) -> Result<Tensor<1>> {
    constraint_dual_loss(log_multiplier, (vector(kl) - limit).neg())
}

fn validate_models<P: RecurrentReppoPolicy>(
    policy: &P,
    critic: &P::Critic,
    config: &ReppoConfig,
) -> Result<()> {
    config.validate()?;
    policy.validate_pair(critic, config)?;
    temporal_models::validate_parameters(policy, false)?;
    temporal_models::validate_parameters(critic, false)?;
    crate::nn::optimization::validate_disjoint(policy, critic)
}
fn indices(values: &[usize], device: &burn::tensor::Device) -> Tensor<1, Int> {
    Tensor::from_data(
        TensorData::new(
            values.iter().map(|&i| i as i64).collect::<Vec<_>>(),
            [values.len()],
        ),
        device,
    )
}
fn select_rows(values: &[f32], width: usize, selection: &[usize]) -> Vec<f32> {
    selection
        .iter()
        .flat_map(|&i| values[i * width..(i + 1) * width].iter().copied())
        .collect()
}
fn discrete_q_values(
    critic: &TemporalQ,
    history: Tensor<2>,
    observations: &[&Observation],
    support: &HlGaussConfig,
    device: &burn::tensor::Device,
) -> Result<Tensor<2>> {
    let k = critic.action_dim();
    let mut owners = Vec::new();
    let mut actions = Vec::new();
    let mut positions = Vec::new();
    for (row, observation) in observations.iter().enumerate() {
        for action in 0..k {
            if observation.dense_mask().is_none_or(|m| m[action]) {
                owners.push(row);
                actions.push(Action::Discrete(action));
                positions.push(row * k + action);
            }
        }
    }
    let features = history.select(0, indices(&owners, &critic.device()));
    let q = to_vec(support.decode(critic.values_from_features(
        features,
        action_tensor(&actions, &critic.spec().action_space)?.to_device(&critic.device()),
    )?)?)?;
    ensure(
        q.iter().all(|x| x.is_finite()),
        "nonfinite recurrent REPPO categorical Q",
    )?;
    let mut values = vec![0.; observations.len() * k];
    for (i, value) in positions.into_iter().zip(q) {
        values[i] = value;
    }
    Ok(matrix_on_device(values, observations.len(), k, device))
}
impl<P: RecurrentReppoPolicy> BehaviorPublisher for RecurrentReppo<P> {
    type Actor = P::Actor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<Self::Actor> {
        ensure(
            revision == self.revision(),
            "requested stale recurrent REPPO behavior",
        )?;
        self.actor()
    }
}
impl<P: RecurrentReppoPolicy> TemporalOnPolicyLearner<P::Actor> for RecurrentReppo<P> {
    fn collection_kind(&self) -> TemporalBatchKind {
        TemporalBatchKind::Pathwise
    }
    fn revision(&self) -> u64 {
        self.revision()
    }
    fn begin_collection(&self) -> Result<()> {
        ensure(
            !self.poisoned,
            "recurrent REPPO requires checkpoint recovery",
        )?;
        let revision = self
            .revision()
            .checked_add(1)
            .ok_or_else(|| Error::State("recurrent REPPO revision overflow".into()))?;
        let begins = self
            .collection_begins
            .get()
            .checked_add(1)
            .ok_or_else(|| Error::State("recurrent REPPO begin count overflow".into()))?;
        self.revision.set(revision);
        self.collection_begins.set(begins);
        self.collection_open.set(true);
        Ok(())
    }
}
impl<P: RecurrentReppoPolicy> StatefulLearner for RecurrentReppo<P> {
    type Batch = Batch<P::Actor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        let CheckedHistory {
            windows,
            full,
            windows_per_batch,
        } = self.preflight(batch)?;
        let mut rng = self.rng.clone();
        let targets = P::prepare_targets(self, batch, &full, &mut rng)?;
        drop(full);
        P::validate_update_queries(self, batch, windows_per_batch, &rng)?;
        let histogram = self.config.hl_gauss.targets(&targets.returns)?;
        let previous_rng = std::mem::replace(&mut self.rng, rng);
        let mut critic_stepped = false;
        let result = self.update_inner(
            batch,
            &windows,
            windows_per_batch,
            targets,
            histogram,
            &mut critic_stepped,
        );
        if result.is_err() {
            if critic_stepped {
                self.poisoned = true;
            } else {
                self.rng = previous_rng;
            }
        }
        result
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::checkpoint::Checkpoint;
    use crate::{
        core::{env::Observation, stateful::HistorySpec},
        data::temporal::{BehaviorMetadata, SequenceConfig, SequenceSample, TemporalStep},
        nn::device,
    };
    use burn::{module::Param, optim::GradientsParams};

    fn recorded_batch(learner: &RecurrentReppo, mut sequences: SequenceBatch) -> Batch {
        use crate::{
            core::recording::RecordingKey, core::stateful_on_policy::DistributionSnapshot,
            data::temporal_on_policy::TemporalRow,
        };
        learner.begin_collection().unwrap();
        let behavior = learner.actor().unwrap();
        let tokens: Vec<_> = (0..sequences.samples[0].steps.len())
            .map(|time| (0, time))
            .collect();
        let actions: Vec<_> = sequences.samples[0]
            .steps
            .iter()
            .map(|step| step.action.clone())
            .collect();
        let latents: Vec<_> = actions
            .iter()
            // These fixtures use an unbounded Gaussian: the physical action
            // already is its latent, and recording requires no transformed latent.
            .map(|_| None)
            .collect();
        let distribution = behavior.policy().distribution(&sequences, &tokens).unwrap();
        let likelihood = to_vec(distribution.log_prob(&actions, &latents).unwrap()).unwrap();
        let gamma = 0.9f32;
        let rows = tokens
            .iter()
            .enumerate()
            .map(|(i, &(sample, time))| {
                sequences.samples[sample].steps[time].behavior = BehaviorMetadata {
                    revision: learner.revision(),
                    log_prob: Some(likelihood[i]),
                };
                TemporalRow {
                    key: RecordingKey {
                        episode_id: 0,
                        timestep: time,
                    },
                    sample,
                    time,
                    log_prob: likelihood[i],
                    latent: latents[i].clone(),
                    distribution: DistributionSnapshot::from_distribution(&distribution, i)
                        .unwrap(),
                    value: None,
                    next_value: None,
                    advantage: None,
                    return_: None,
                    discount_weight: (gamma as f64).powf(time as f64) as f32,
                }
            })
            .collect();
        let batch = Batch {
            version: 1,
            revision: learner.revision(),
            behavior,
            kind: TemporalBatchKind::Pathwise,
            gamma,
            gae_lambda: None,
            episode_count: 1,
            sequences,
            rows,
        };
        batch.validate_snapshot().unwrap();
        batch
    }

    #[test]
    fn soft_targets_discount_primitive_traces_at_each_boundary() {
        use crate::core::reward::Reward;

        let (mut learner, sequences) = fixture(ActionSpace::Discrete { count: 2 });
        let device = learner.policy.device();
        let policy_head = &mut learner.policy.head.layers[0];
        policy_head.weight = Param::from_tensor(Tensor::zeros([3, 2], &device));
        policy_head.bias = Some(Param::from_tensor(vector(&[0., 0.])));
        let q_head = &mut learner.critic.head.layers[0];
        q_head.weight = Param::from_tensor(Tensor::zeros([7, 3], &device));
        q_head.bias = Some(Param::from_tensor(vector(&[0., 0., 2_f32.ln()])));
        learner.alpha = LogMultiplier::new(1.);
        learner.config.lambda = 0.5;
        let mut batch = recorded_batch(&learner, sequences);
        let sample = batch.sequences.samples.pop().unwrap();
        let rows = std::mem::take(&mut batch.rows);
        batch.gamma = 0.5;
        for (index, (rewards, terminated, truncated)) in [
            ([&[-1., 4.][..], &[1., -2.]], false, false),
            ([&[1., 2.][..], &[-2., 6.]], true, false),
            ([&[3., -2.][..], &[4., -2., 8.]], false, true),
        ]
        .into_iter()
        .enumerate()
        {
            let mut sample = sample.clone();
            sample.episode_id = index as u64;
            sample.observations.truncate(3);
            sample.steps.truncate(2);
            sample.loss_mask.truncate(2);
            sample.steps[1].terminated = terminated;
            sample.steps[1].truncated = truncated;
            for (time, reward) in rewards.into_iter().enumerate() {
                sample.steps[time].reward = Reward::from_primitives(reward.to_vec()).unwrap();
                let mut row = rows[time].clone();
                row.key.episode_id = sample.episode_id;
                row.sample = index;
                row.discount_weight = [1., 0.25][time];
                batch.rows.push(row);
            }
            batch.sequences.samples.push(sample);
        }
        batch.episode_count = 3;

        // Bin probabilities are (1/4, 1/4, 1/2) on (-1, 0, 1). Both actions
        // have probability 1/2. The six discounted rewards are 1, 0, 2, 1, 2, 5.
        let entropy = 2_f32.ln();
        let truncation = 5. + 0.125 * (0.25 + entropy);
        let fragment_tail = 0.25 * (0.25 + entropy);
        let expected = [
            1. + 0.25 * (0.125 + 0.5 * fragment_tail + entropy),
            fragment_tail,
            2. + 0.25 * (0.625 + entropy),
            1.,
            2. + 0.25 * (0.125 + 0.5 * truncation + entropy),
            truncation,
        ];
        let targets = learner.target_preview(&batch).unwrap();
        learner.set_update_mode(TemporalExecutionMode::OptimizedExact);
        assert_eq!(learner.target_preview(&batch).unwrap(), targets);
        assert_eq!(targets.len(), expected.len());
        for (target, expected) in targets.into_iter().zip(expected) {
            assert!((target - expected).abs() < 1e-6, "{target} != {expected}");
        }
    }

    #[test]
    fn collection_phase_and_consumption_corruption_are_rejected_atomically() {
        let (mut learner, sequences) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        let batch = recorded_batch(&learner, sequences);
        for trained in [false, true] {
            if trained {
                learner.update(&batch).unwrap();
            }
            let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
            let mut bad = learner.checkpoint().unwrap();
            bad.metadata["collection_open"] = (!learner.collection_open.get()).into();
            assert!(learner.restore(&bad).is_err());
            let mut bad = learner.checkpoint().unwrap();
            bad.metadata["collection_begins"] = u64::MAX.into();
            assert!(learner.restore(&bad).is_err());
            let mut bad = learner.checkpoint().unwrap();
            bad.metadata["consumed_revision"] = learner.revision().into();
            assert!(learner.restore(&bad).is_err());
            assert!(
                serde_json::to_value(learner.checkpoint().unwrap()).unwrap() == before,
                "rejected phase record mutated learner"
            );
        }
    }

    #[test]
    fn recorded_optimizer_groups_are_mandatory_and_wrong_owners_are_atomic() {
        let (mut learner, sequences) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        let batch = recorded_batch(&learner, sequences);
        learner.update(&batch).unwrap();
        let record = learner.checkpoint().unwrap();
        let owners = record
            .metadata
            .get("optimizer_owners")
            .and_then(serde_json::Value::as_object)
            .expect("REPPO must record exact trained optimizer groups");
        assert_eq!(owners.len(), 4);
        let before = serde_json::to_value(&record).unwrap();
        let mut missing = record.clone();
        missing
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("optimizer_owners");
        let mut incomplete = record.clone();
        incomplete.metadata["optimizer_owners"]["policy"] = serde_json::json!([]);
        let mut foreign = record.clone();
        foreign
            .optimizers
            .insert("entropy_dual".into(), record.optimizers["policy"].clone());
        let mut wrong_steps = record.clone();
        wrong_steps.metadata["updates"] = (learner.updates + learner.config.epochs as u64).into();
        for bad in [missing, incomplete, foreign, wrong_steps] {
            assert!(learner.restore(&bad).is_err());
            assert!(
                serde_json::to_value(learner.checkpoint().unwrap()).unwrap() == before,
                "rejected optimizer record mutated learner"
            );
        }
    }

    #[test]
    fn predictor_optimizer_restores_after_batches_without_auxiliary_successors() {
        let (mut learner, sequences) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        learner.config.auxiliary_coefficient = 0.2;
        let batch = recorded_batch(&learner, sequences.clone());
        learner.update(&batch).unwrap();
        let mut ending = sequences;
        ending.samples[0].steps.truncate(1);
        ending.samples[0].steps[0].terminated = true;
        ending.samples[0].observations.truncate(2);
        ending.samples[0].loss_mask.truncate(1);
        let batch = recorded_batch(&learner, ending);
        learner.update(&batch).unwrap();
        assert_eq!(learner.auxiliary_updates, learner.config.epochs as u64);
        assert_eq!(learner.updates, 2 * learner.auxiliary_updates);
        let record = learner.checkpoint().unwrap();
        let (mut restored, _) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        restored.restore(&record).unwrap();
        assert_eq!(restored.updates, learner.updates);
        assert_eq!(restored.auxiliary_updates, learner.auxiliary_updates);
        assert_eq!(restored.multipliers(), learner.multipliers());
    }

    fn fixture(space: ActionSpace) -> (RecurrentReppo, SequenceBatch) {
        let mut rng = ChaCha8Rng::seed_from_u64(601);
        let spec = HistorySpec::new(1, space.clone()).unwrap();
        let policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
        let critic = TemporalQ::new(spec, Some(5), &[], 3, &mut rng).unwrap();
        let learner = RecurrentReppo::new(
            policy,
            critic,
            ReppoConfig {
                hl_gauss: HlGaussConfig {
                    min: -1.,
                    max: 1.,
                    bins: 3,
                    sigma_ratio: 0.75,
                },
                kl_limit: 100.,
                ..Default::default()
            },
            602,
        )
        .unwrap();
        let batch = SequenceBatch {
            config: SequenceConfig {
                burn_in: 0,
                learn_len: 3,
                n_step: 1,
            },
            samples: vec![SequenceSample {
                trajectory_start: 0,
                history_start: 0,
                previous: None,
                origin: Default::default(),
                episode_id: 0,
                learn_start: 0,
                burn_start: 0,
                observations: [0.7, -0.3, 0.2, 0.]
                    .map(|x| Observation::new(vec![x]))
                    .to_vec(),
                steps: (0..3)
                    .map(|i| TemporalStep {
                        action: match space {
                            ActionSpace::Discrete { .. } => Action::Discrete(i % 2),
                            _ => Action::Continuous(vec![[-0.4, 0.1, 0.2][i]]),
                        },
                        reward: [0.6_f32, -0.2, 1.][i].into(),
                        terminated: i == 2,
                        truncated: false,
                        behavior: BehaviorMetadata {
                            revision: 0,
                            log_prob: None,
                        },
                    })
                    .collect(),
                loss_mask: vec![true; 3],
            }],
        };
        (learner, batch)
    }

    #[test]
    fn actor_uses_independent_behavior_gru_and_forward_kl_gate() {
        let (mut learner, batch) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        learner.policy.log_std = Param::from_tensor(vector(&[2_f32.ln()]));
        let mut old = learner.policy();
        old.log_std = Param::from_tensor(vector(&[0.]));
        let gate = &mut old
            .encoder
            .gru
            .as_mut()
            .unwrap()
            .gru
            .new_gate
            .input_transform;
        gate.weight = Param::from_tensor(gate.weight.val() + 0.7);
        let old = frozen_copy(&old);
        let tokens = [(0, 2)];
        let behavior = old.distribution(&batch, &tokens).unwrap();
        let current = learner.policy.distribution(&batch, &tokens).unwrap();
        let (Distribution::Gaussian { mean: om, .. }, Distribution::Gaussian { mean: cm, .. }) =
            (&behavior, &current)
        else {
            panic!()
        };
        let delta = to_vec(om.clone()).unwrap()[0] - to_vec(cm.clone()).unwrap()[0];
        assert!(delta.abs() > 1e-4, "old/current histories must differ");
        let expected = 2_f32.ln() + (1. + delta * delta) / 8. - 0.5;
        learner.config.kl_limit = 0.01;
        let result = learner
            .actor_objective(
                &batch,
                &tokens,
                &behavior,
                &mut ChaCha8Rng::seed_from_u64(603),
            )
            .unwrap();
        assert!((result.kl[0] - expected).abs() < 2e-6);
        assert!((result.loss.into_scalar::<f32>() - learner.beta.value() * expected).abs() < 2e-6);
        let dual = constraint_dual_loss(
            learner.beta.log_value.val(),
            vector(&[learner.config.kl_limit - expected]),
        )
        .unwrap();
        assert!(
            (dual.into_scalar::<f32>()
                - learner.beta.value() * (learner.config.kl_limit - expected))
                .abs()
                < 2e-6
        );
    }

    #[test]
    fn same_noise_actor_gru_finite_difference_flows_through_frozen_q_only() {
        let (mut learner, batch) = fixture(ActionSpace::Unbounded { dimensions: 1 });
        // Make action utility visible and isolate it from entropy's mean gradient.
        learner.critic.head.layers[0].weight = Param::from_tensor(matrix(
            vec![
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 1.,
            ],
            6,
            3,
        ));
        learner.critic.head.layers[0].bias = Some(Param::from_tensor(vector(&[0., 0., 0.])));
        let tokens = [(0, 2)];
        let old = learner.policy().distribution(&batch, &tokens).unwrap();
        let parameter = learner
            .policy
            .encoder
            .gru
            .as_ref()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight
            .val();
        let dims = parameter.dims();
        let original = to_vec(parameter.clone()).unwrap();
        let result = learner
            .actor_objective(&batch, &tokens, &old, &mut ChaCha8Rng::seed_from_u64(604))
            .unwrap();
        let mut gradients = result.loss.backward();
        let analytical = to_vec(parameter.grad(&gradients).unwrap()).unwrap();
        assert!(
            GradientsParams::from_module(&mut gradients, &learner.critic).is_empty(),
            "critic parameters received actor gradients"
        );
        let index = analytical
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!(
            analytical[index].abs() > 1e-5,
            "actor graph lost action utility"
        );
        let mut values = [0.; 2];
        let epsilon = 0.002;
        for (slot, sign) in [-1., 1.].into_iter().enumerate() {
            let mut perturbed = original.clone();
            perturbed[index] += sign * epsilon;
            learner
                .policy
                .encoder
                .gru
                .as_mut()
                .unwrap()
                .gru
                .new_gate
                .input_transform
                .weight = Param::from_tensor(Tensor::from_data(
                TensorData::new(perturbed, dims),
                &device(),
            ));
            values[slot] = learner
                .actor_objective(&batch, &tokens, &old, &mut ChaCha8Rng::seed_from_u64(604))
                .unwrap()
                .loss
                .into_scalar::<f32>();
        }
        let numerical = (values[1] - values[0]) / (2. * epsilon);
        assert!(
            (numerical - analytical[index]).abs() < 1e-4 + 0.01 * analytical[index].abs(),
            "finite difference {numerical}, autodiff {}",
            analytical[index]
        );
        assert_eq!(
            batch.samples[0].steps[0].action,
            Action::Continuous(vec![-0.4])
        );
    }

    #[test]
    fn discrete_actor_matches_manual_masked_expectation_without_rng() {
        let (learner, mut batch) = fixture(ActionSpace::Discrete { count: 2 });
        batch.samples[0].observations[0]
            .as_dense_mut()
            .unwrap()
            .action_mask = Some(vec![true, false]);
        let tokens = [(0, 0), (0, 2)];
        let old = learner.policy().distribution(&batch, &tokens).unwrap();
        let terms = learner
            .policy
            .distribution(&batch, &tokens)
            .unwrap()
            .categorical_terms()
            .unwrap();
        let probabilities = to_vec(terms.probabilities).unwrap();
        let frozen = learner.critic();
        let history = frozen.history_features(&batch, &tokens).unwrap();
        let mut expected = 0.;
        for row in 0..2 {
            for action in 0..2 {
                let p = probabilities[row * 2 + action];
                if p > 0. {
                    let features = history.clone().slice([row..row + 1, 0..5]);
                    let actions = action_tensor(
                        &[Action::Discrete(action)],
                        &learner.policy.spec().action_space,
                    )
                    .unwrap()
                    .to_device(&frozen.device());
                    let q = learner
                        .config
                        .hl_gauss
                        .decode(frozen.values_from_features(features, actions).unwrap())
                        .unwrap()
                        .into_scalar::<f32>();
                    expected += p * (learner.alpha.value() * p.ln() - q) / 2.;
                }
            }
        }
        let mut rng = ChaCha8Rng::seed_from_u64(605);
        let before = rng.clone();
        let result = learner
            .actor_objective(&batch, &tokens, &old, &mut rng)
            .unwrap();
        assert_eq!(rng, before);
        assert!((result.loss.into_scalar::<f32>() - expected).abs() < 2e-6);
        assert!(result.entropy[0].abs() < 1e-7);
    }
}
