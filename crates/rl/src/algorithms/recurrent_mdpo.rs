//! Full-batch recurrent mirror descent with one frozen behavior and independent value fitting.
use super::{
    mdpo::{MdpoConfig, MdpoWeighting, optimizer_ids, validate_optimizer_numbers},
    recurrent_on_policy::{
        RecurrentOnPolicyPolicy, TemporalOnPolicyRecord, TemporalOnPolicyState,
        distribution::RecurrentPolicyDistribution,
        minibatch::{Minibatch, WindowBatch},
    },
};
use crate::data::temporal_on_policy::TemporalOnPolicyLearner;
use crate::{
    Error, Result,
    core::{
        action::Action,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, ReconstructionHistory, StatefulActor, StatefulLearner},
        temporal::TemporalExecutionMode,
    },
    data::{
        minibatch::shuffled_indices_into,
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    },
    nn::{
        finite_loss,
        optimization::{optimize_validated, parameter_norm},
        target::frozen_copy,
        temporal_policy::TemporalPolicy,
        vector,
    },
    objectives::mdpo::{MdpoTargets, mdpo_objective, normalized_occupancy_weights},
};
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Each outer call performs all actor steps before any critic optimization.
pub struct RecurrentMdpo<P: RecurrentOnPolicyPolicy = TemporalPolicy> {
    state: TemporalOnPolicyState<P>,
    config: MdpoConfig,
    outer_iterations: u64,
    update_mode: TemporalExecutionMode,
}
#[derive(Serialize, Deserialize)]
struct MdpoRecord {
    outer_iterations: u64,
    policy_optimizer_ids: Vec<u64>,
    value_optimizer_ids: Vec<u64>,
    #[serde(default)]
    update_mode: TemporalExecutionMode,
}

struct Prepared<'a, P: RecurrentOnPolicyPolicy> {
    full: Minibatch<'a, P::Actor>,
    tokens: Vec<(usize, usize)>,
    behavior: P::Distribution,
    actions: Vec<Action>,
    latents: Vec<Option<Vec<f32>>>,
    old_log_probs: Tensor<1>,
    weights: Tensor<1>,
    advantages: Tensor<1>,
}
impl<'a, P: RecurrentOnPolicyPolicy> Prepared<'a, P> {
    fn new(
        batch: &TemporalOnPolicyBatch<P::Actor>,
        full: Minibatch<'a, P::Actor>,
        weighting: MdpoWeighting,
    ) -> Result<Self> {
        let tokens = full.tokens();
        let behavior = P::published_models(&batch.behavior)
            .0
            .distribution(&full, &tokens, true)?;
        let actions = batch.actions();
        let latents = batch.latents();
        let old_log_probs = behavior.log_prob(&actions, &latents)?.without_autodiff();
        let weights = match weighting {
            MdpoWeighting::Uniform => vec![1. / batch.len() as f32; batch.len()],
            // Batch validation binds these to gamma^absolute_episode_timestep.
            // Never restart discounting at a selected learning window boundary.
            MdpoWeighting::DiscountedOccupancy => normalized_occupancy_weights(
                &batch
                    .rows
                    .iter()
                    .map(|row| row.discount_weight)
                    .collect::<Vec<_>>(),
            )?,
        };
        Ok(Self {
            full,
            tokens,
            behavior,
            actions,
            latents,
            old_log_probs,
            weights: vector(&weights),
            advantages: vector(&batch.advantages()?),
        })
    }
    fn targets(&self) -> MdpoTargets<'_> {
        MdpoTargets {
            actions: &self.actions,
            latents: &self.latents,
            old_log_probs: &self.old_log_probs,
            weights: &self.weights,
            advantages: &self.advantages,
        }
    }
}
impl<P: RecurrentOnPolicyPolicy> RecurrentMdpo<P> {
    pub fn new(policy: P, value: P::Value, config: MdpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        parameter_norm(&policy)?;
        parameter_norm(&value)?;
        let state = TemporalOnPolicyState::new(
            policy,
            Some(value),
            &config.policy,
            Some(&config.value),
            seed,
        )?;
        Ok(Self {
            state,
            config,
            outer_iterations: 0,
            update_mode: TemporalExecutionMode::Reference,
        })
    }
    pub fn config(&self) -> &MdpoConfig {
        &self.config
    }
    /// Individual actor plus critic optimizer steps, not paired minibatches.
    pub fn updates(&self) -> u64 {
        self.state.updates
    }
    pub fn actor_updates(&self) -> u64 {
        self.state.policy_updates
    }
    pub fn critic_updates(&self) -> u64 {
        self.state.value_updates
    }
    pub fn outer_iterations(&self) -> u64 {
        self.outer_iterations
    }
    pub fn revision(&self) -> u64 {
        self.state.revision
    }
    /// The execution path used by the current model adapter.
    pub fn update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
            .effective(self.state.supports_optimized_exact())
    }

    /// The saved request, including when it falls back to reference execution.
    pub fn requested_update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
    }
    /// Whether this learner can execute the requested mode.
    pub fn supports_update_mode(&self, mode: TemporalExecutionMode) -> bool {
        mode.supported(self.state.supports_optimized_exact())
    }

    /// Request a mode. Unsupported requests use reference execution.
    pub fn set_update_mode(&mut self, mode: TemporalExecutionMode) {
        self.update_mode = mode;
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.state.policy)
    }
    pub fn value(&self) -> P::Value {
        frozen_copy(self.state.value.as_ref().expect("MDPO scalar value"))
    }
    fn update_inner(
        &mut self,
        windows: &WindowBatch<'_, P::Actor>,
        prepared: Prepared<'_, P>,
        mirror_step: f32,
        windows_per_batch: usize,
        critic_steps: u64,
        work: u64,
    ) -> Result<UpdateReport> {
        let mut actor_gradient = 0.;
        for _ in 0..self.config.inner_steps {
            // Fresh current-policy reconstruction after every optimizer step;
            // neither behavior memory nor an already-consumed graph is reused.
            let current = self
                .state
                .policy
                .distribution(&prepared.full, &prepared.tokens, true)?;
            let objective = mdpo_objective(
                &current,
                &prepared.behavior,
                prepared.targets(),
                mirror_step.recip(),
            )?;
            let (_, norm) = {
                #[cfg(feature = "probes")]
                let _profile = crate::profiling::enter(crate::profiling::Phase::ActorOptimization);
                optimize_validated(
                    &mut self.state.policy,
                    &mut self.state.policy_optimizer,
                    self.config.policy.learning_rate,
                    objective.loss,
                )?
            };
            actor_gradient = norm;
            self.state.policy_updates += 1;
            self.state.updates += 1;
        }
        let current =
            self.state
                .policy
                .valid()
                .distribution(&prepared.full, &prepared.tokens, true)?;
        let objective = mdpo_objective(
            &current,
            &prepared.behavior,
            prepared.targets(),
            mirror_step.recip(),
        )?;
        let actor_loss = finite_loss(&objective.loss)?;
        let surrogate = finite_loss(&objective.surrogate)?;
        let kl = finite_loss(&objective.kl)?;
        let mut indices = Vec::new();
        let mut value_loss = 0.;
        let mut critic_gradient = 0.;
        for _ in 0..self.config.value_epochs {
            shuffled_indices_into(
                windows.window_count(),
                windows_per_batch,
                &mut self.state.rng,
                &mut indices,
            )?;
            for selection in indices.chunks(windows_per_batch) {
                let selected = windows.select(selection)?;
                let value = self.state.value.as_mut().expect("MDPO scalar value");
                let loss = (P::values(value, &selected, &selected.tokens())?
                    - vector(&selected.returns()?))
                .powf_scalar(2.)
                .mean();
                let (number, norm) = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::CriticOptimization);
                    optimize_validated(
                        value,
                        self.state
                            .value_optimizer
                            .as_mut()
                            .expect("MDPO value optimizer"),
                        self.config.value.learning_rate,
                        loss,
                    )?
                };
                value_loss += number * selected.len() as f64;
                critic_gradient = norm;
                self.state.value_updates += 1;
                self.state.updates += 1;
            }
        }
        // Entropy is a diagnostic, evaluated with the same final policy/occupancy
        // after value shuffling, preserving MDPO's owned-RNG consumption order.
        let entropy =
            finite_loss(&(prepared.weights * current.entropy(&mut self.state.rng)?).sum())?;
        self.state.validate_models()?;
        let actor_norm = parameter_norm(&self.state.policy)?;
        let critic_norm = parameter_norm(self.state.value.as_ref().expect("MDPO scalar value"))?;
        self.outer_iterations += 1;
        self.state.revision += 1;
        Ok(UpdateReport {
            updates: work,
            ..Default::default()
        }
        .metric("policy_loss", actor_loss)
        .metric("advantage_surrogate", surrogate)
        .metric(
            "value_loss",
            value_loss / (prepared.full.len() * self.config.value_epochs) as f64,
        )
        .metric("kl_new_behavior", kl)
        .metric("entropy", entropy)
        .metric("mirror_step", mirror_step as f64)
        .metric("kl_coefficient", mirror_step.recip() as f64)
        .metric("actor_updates", self.config.inner_steps as f64)
        .metric("critic_updates", critic_steps as f64)
        .metric("outer_iterations", self.outer_iterations as f64)
        .metric("actor_parameter_norm", actor_norm)
        .metric("critic_parameter_norm", critic_norm)
        .metric("actor_gradient_norm", actor_gradient)
        .metric("critic_gradient_norm", critic_gradient))
    }
}
impl<P: RecurrentOnPolicyPolicy> BehaviorPublisher for RecurrentMdpo<P> {
    type Actor = P::Actor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<P::Actor> {
        ensure(
            revision == self.state.revision,
            "requested stale recurrent MDPO behavior revision",
        )?;
        self.state.actor()
    }
}
impl<P: RecurrentOnPolicyPolicy> TemporalOnPolicyLearner<P::Actor> for RecurrentMdpo<P> {
    fn collection_kind(&self) -> TemporalBatchKind {
        TemporalBatchKind::Bootstrapped
    }
    fn revision(&self) -> u64 {
        self.state.revision
    }
    fn reconstruct_published_state(
        &self,
        actor: &P::Actor,
        history: ReconstructionHistory<'_>,
    ) -> Result<Option<<P::Actor as StatefulActor>::State>> {
        P::reconstruct_published_state(actor, history, self.update_mode())
    }
}
impl<P: RecurrentOnPolicyPolicy> StatefulLearner for RecurrentMdpo<P> {
    type Batch = TemporalOnPolicyBatch<P::Actor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        self.config.validate()?;
        self.state
            .preflight(batch, TemporalBatchKind::Bootstrapped)?;
        let mirror_step = self.config.mirror_step.value(self.outer_iterations)?;
        ensure(
            self.outer_iterations == self.state.revision,
            "recurrent MDPO outer schedule differs from behavior revision",
        )?;
        ensure(
            batch.sequences.config.learn_len <= self.config.value_minibatch_size,
            "recurrent MDPO learning window exceeds the critic minibatch learning-position limit",
        )?;
        let windows_per_batch =
            (self.config.value_minibatch_size / batch.sequences.config.learn_len).max(1);
        let critic_steps = self
            .config
            .value_epochs
            .checked_mul(batch.sequences.samples.len().div_ceil(windows_per_batch))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| Error::State("recurrent MDPO update count overflow".into()))?;
        let actor_steps = u64::try_from(self.config.inner_steps)
            .map_err(|_| Error::State("recurrent MDPO actor count overflow".into()))?;
        let work = actor_steps
            .checked_add(critic_steps)
            .ok_or_else(|| Error::State("recurrent MDPO update count overflow".into()))?;
        ensure(
            self.state.updates.checked_add(work).is_some()
                && self.outer_iterations.checked_add(1).is_some()
                && batch.len().checked_mul(self.config.value_epochs).is_some(),
            "recurrent MDPO counter overflow",
        )?;
        ensure(
            self.state
                .policy_updates
                .checked_add(actor_steps)
                .is_some_and(|n| n < i32::MAX as u64)
                && self
                    .state
                    .value_updates
                    .checked_add(critic_steps)
                    .is_some_and(|n| n < i32::MAX as u64),
            "MDPO Adam step count exceeds supported bias-correction range",
        )?;
        let windows =
            self.state
                .policy
                .windows(batch, self.update_mode(), self.state.value.as_ref())?;
        let prepared = Prepared::<P>::new(batch, windows.full()?, self.config.weighting)?;
        // Validate immutable objective/value targets before touching live modules,
        // optimizers, random streams, or counters.
        mdpo_objective(
            &prepared.behavior,
            &prepared.behavior,
            prepared.targets(),
            mirror_step.recip(),
        )?;
        let prediction = P::values(&self.value(), &prepared.full, &prepared.tokens)?;
        ensure(
            prediction.dims() == [batch.len()],
            "recurrent MDPO state value must return one scalar per learning token",
        )?;
        finite_loss(
            &(prediction - vector(&batch.returns()?))
                .powf_scalar(2.)
                .mean(),
        )?;
        let result = self.update_inner(
            &windows,
            prepared,
            mirror_step,
            windows_per_batch,
            critic_steps,
            work,
        );
        if result.is_err() {
            self.state.poisoned = true;
        }
        result
    }
}
impl<P: RecurrentOnPolicyPolicy> Checkpoint for RecurrentMdpo<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record(
            "recurrent_mdpo",
            TemporalBatchKind::Bootstrapped,
            &self.config,
        )?;
        record.metadata["outer_iterations"] = self.outer_iterations.into();
        if !self.update_mode.is_reference() {
            record.metadata["update_mode"] = serde_json::to_value(self.update_mode)?;
        }
        record.metadata["policy_optimizer_ids"] =
            serde_json::to_value(optimizer_ids(&record, "policy")?)?;
        record.metadata["value_optimizer_ids"] =
            serde_json::to_value(optimizer_ids(&record, "value")?)?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent_mdpo")?;
        let saved: TemporalOnPolicyRecord<MdpoConfig, P::Architecture> =
            serde_json::from_value(record.metadata.clone())?;
        let metadata: MdpoRecord = serde_json::from_value(record.metadata.clone())?;
        saved.config.validate()?;
        ensure(
            saved.value.is_some()
                && saved.policy_updates.checked_add(saved.value_updates) == Some(saved.updates)
                && metadata.outer_iterations == saved.revision
                && metadata.outer_iterations <= saved.policy_updates
                && metadata.outer_iterations <= saved.value_updates,
            "inconsistent recurrent MDPO checkpoint counters",
        )?;
        saved.config.mirror_step.value(metadata.outer_iterations)?;
        validate_optimizer_numbers(
            record,
            "policy",
            &metadata.policy_optimizer_ids,
            saved.policy_updates,
        )?;
        validate_optimizer_numbers(
            record,
            "value",
            &metadata.value_optimizer_ids,
            saved.value_updates,
        )?;
        let state = TemporalOnPolicyState::restored(
            record,
            &saved,
            TemporalBatchKind::Bootstrapped,
            &saved.config.policy,
            Some(&saved.config.value),
        )?;
        parameter_norm(&state.policy)?;
        parameter_norm(state.value.as_ref().expect("validated MDPO value"))?;
        *self = Self {
            state,
            config: saved.config,
            outer_iterations: metadata.outer_iterations,
            update_mode: metadata.update_mode,
        };
        Ok(())
    }
}
