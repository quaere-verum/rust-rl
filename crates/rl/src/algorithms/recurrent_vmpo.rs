//! Recurrent V-MPO with independent chronological representations and dual variables.
use super::{
    recurrent_on_policy::{
        RecurrentOnPolicyPolicy, TemporalOnPolicyRecord, TemporalOnPolicyState,
        checked_update_capacity, distribution::RecurrentPolicyDistribution, minibatch::WindowBatch,
        validate_required_adam_clocks,
    },
    vmpo::{Multipliers, VmpoConfig},
};
use crate::data::temporal_on_policy::TemporalOnPolicyLearner;
use crate::{
    Error, Result,
    core::{
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
        checkpoint::RecordBurnExt, optimization::optimize, target::frozen_copy, temporal_models,
        temporal_policy::TemporalPolicy, vector,
    },
    objectives::vmpo::{
        dual_objective, policy_objective, temperature_weights, top_half_indices_into,
    },
};
use burn::{
    module::Param,
    optim::ModuleOptimizer,
    tensor::{Int, Tensor, TensorData, activation::softplus},
};

pub struct RecurrentVmpo<P: RecurrentOnPolicyPolicy = TemporalPolicy> {
    state: TemporalOnPolicyState<P>,
    config: VmpoConfig,
    multipliers: Multipliers,
    dual_optimizer: ModuleOptimizer,
    dual_updates: u64,
    update_mode: TemporalExecutionMode,
}

fn initial_multipliers() -> Multipliers {
    Multipliers {
        log_eta: Param::from_tensor(vector(&[0.05])),
        log_alpha: Param::from_tensor(vector(&[0.1])),
    }
}

impl<P: RecurrentOnPolicyPolicy> RecurrentVmpo<P> {
    pub fn new(policy: P, value: P::Value, config: VmpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let state = TemporalOnPolicyState::new(
            policy,
            Some(value),
            &config.policy,
            Some(&config.value),
            seed,
        )?;
        let dual_optimizer = config.dual.build()?;
        Ok(Self {
            state,
            config,
            multipliers: initial_multipliers(),
            dual_optimizer,
            dual_updates: 0,
            update_mode: TemporalExecutionMode::Reference,
        })
    }
    pub fn config(&self) -> &VmpoConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.state.updates
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
        frozen_copy(self.state.value.as_ref().expect("V-MPO value model"))
    }
    pub fn multipliers(&self) -> (f32, f32) {
        (
            softplus(self.multipliers.log_eta.val(), 1.)
                .into_scalar::<f32>()
                .max(1e-6),
            softplus(self.multipliers.log_alpha.val(), 1.)
                .into_scalar::<f32>()
                .max(1e-6),
        )
    }
    fn update_inner(
        &mut self,
        batch: &WindowBatch<'_, P::Actor>,
        windows_per_batch: usize,
    ) -> Result<UpdateReport> {
        let mut report = UpdateReport::default();
        let mut indices = Vec::new();
        let mut top = Vec::new();
        for _ in 0..self.config.epochs {
            {
                #[cfg(feature = "probes")]
                let _profile = crate::profiling::enter(crate::profiling::Phase::MinibatchSelection);
                shuffled_indices_into(
                    batch.window_count(),
                    windows_per_batch,
                    &mut self.state.rng,
                    &mut indices,
                )?;
            }
            for windows in indices.chunks(windows_per_batch) {
                let (selected, tokens) = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::MinibatchPreparation);
                    let selected = batch.select(windows)?;
                    let tokens = selected.tokens();
                    (selected, tokens)
                };
                // Reconstruct every selected window's actual prefix with the current
                // GRU before choosing the high-advantage learning tokens.
                let current = self.state.policy.distribution(&selected, &tokens, true)?;
                let advantages = selected.advantages()?;
                top_half_indices_into(&advantages, &mut top);
                let selected_advantages: Vec<_> = top.iter().map(|&i| advantages[i]).collect();
                let (eta, alpha) = self.multipliers();
                let (weights, maximum) = temperature_weights(&selected_advantages, eta);
                let top_tensor = Tensor::<1, Int>::from_data(
                    TensorData::new(top.iter().map(|&i| i as i64).collect(), [top.len()]),
                    &self.state.policy.policy_device(),
                );
                let log_prob = current
                    .log_prob(&selected.actions(), &selected.latents())?
                    .select(0, top_tensor);
                // Shared preflight independently reconstructs and verifies every
                // immutable behavior row. Its numeric distribution cache therefore
                // remains parameter-compatible without reusing current hidden states.
                let snapshots = selected.behavior_distributions();
                let behavior = P::Distribution::from_snapshots(
                    &snapshots,
                    &self.state.policy.policy_device(),
                )?;
                let kl = current.kl(&behavior)?.mean(); // KL(current || behavior).
                let kl_number = kl.clone().into_scalar::<f32>();
                ensure(kl_number.is_finite(), "nonfinite recurrent V-MPO KL")?;
                let policy_loss = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::ActorOptimization);
                    optimize(
                        &mut self.state.policy,
                        &mut self.state.policy_optimizer,
                        self.config.policy.learning_rate,
                        policy_objective(weights, log_prob, kl, alpha),
                    )?
                };
                // KL is a detached pre-actor-step scalar, preserving the established
                // actor -> temperature/KL duals -> independent critic update order.
                let dual = dual_objective(
                    self.multipliers.log_eta.val(),
                    self.multipliers.log_alpha.val(),
                    &selected_advantages,
                    maximum,
                    self.config.temperature_constraint,
                    self.config.kl_constraint,
                    kl_number,
                );
                let dual_loss = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::DualOptimization);
                    optimize(
                        &mut self.multipliers,
                        &mut self.dual_optimizer,
                        self.config.dual.learning_rate,
                        dual,
                    )?
                };
                let value = self.state.value.as_mut().expect("V-MPO value model");
                let value_loss = (P::values(value, &selected, &tokens)?
                    - vector(&selected.returns()?))
                .powf_scalar(2.)
                .mean();
                let value_loss = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::CriticOptimization);
                    optimize(
                        value,
                        self.state
                            .value_optimizer
                            .as_mut()
                            .expect("V-MPO value optimizer"),
                        self.config.value.learning_rate,
                        value_loss,
                    )?
                };
                self.state.updates += 1;
                self.state.policy_updates += 1;
                self.state.value_updates += 1;
                self.dual_updates += 1;
                report.updates += 1;
                report = report
                    .metric("policy_loss", policy_loss)
                    .metric("value_loss", value_loss)
                    .metric("dual_loss", dual_loss)
                    .metric("kl_new_behavior", kl_number as f64);
            }
        }
        self.state.validate_models()?;
        temporal_models::validate_parameters(&self.multipliers, false)?;
        self.state.revision += 1;
        Ok(report)
    }
}

impl<P: RecurrentOnPolicyPolicy> BehaviorPublisher for RecurrentVmpo<P> {
    type Actor = P::Actor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<P::Actor> {
        ensure(
            revision == self.state.revision,
            "requested stale recurrent V-MPO behavior",
        )?;
        self.state.actor()
    }
}
impl<P: RecurrentOnPolicyPolicy> TemporalOnPolicyLearner<P::Actor> for RecurrentVmpo<P> {
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
impl<P: RecurrentOnPolicyPolicy> StatefulLearner for RecurrentVmpo<P> {
    type Batch = TemporalOnPolicyBatch<P::Actor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        self.config.validate()?;
        self.state
            .preflight(batch, TemporalBatchKind::Bootstrapped)?;
        ensure(
            batch.sequences.config.learn_len <= self.config.minibatch_size,
            "recurrent V-MPO learning window exceeds minibatch learning-position limit",
        )?;
        let windows_per_batch =
            (self.config.minibatch_size / batch.sequences.config.learn_len).max(1);
        checked_update_capacity(
            self.state.updates,
            self.config.epochs,
            batch.sequences.samples.len(),
            windows_per_batch,
        )?;
        let windows =
            self.state
                .policy
                .windows(batch, self.update_mode(), self.state.value.as_ref())?;
        let result = self.update_inner(&windows, windows_per_batch);
        if result.is_err() {
            self.state.poisoned = true;
        }
        result
    }
}
impl<P: RecurrentOnPolicyPolicy> Checkpoint for RecurrentVmpo<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        temporal_models::validate_parameters(&self.multipliers, false)?;
        let mut record = self.state.record(
            "recurrent_vmpo",
            TemporalBatchKind::Bootstrapped,
            &self.config,
        )?;
        record.metadata["dual_updates"] = self.dual_updates.into();
        if !self.update_mode.is_reference() {
            record.metadata["update_mode"] = serde_json::to_value(self.update_mode)?;
        }
        record.model("multipliers", &self.multipliers)?;
        record.optimizer("dual", &self.dual_optimizer)?;
        let dual = crate::nn::checkpoint::audit_adam_optimizer(
            &record,
            "dual",
            "multipliers",
            None,
            self.dual_updates,
        )?;
        record.metadata["optimizer_owners"]["dual"] = serde_json::to_value(dual.parameter_ids)?;
        if P::SPARSE_ADAM {
            validate_required_adam_clocks(
                &record,
                "multipliers",
                &["log_eta", "log_alpha"],
                &dual.steps,
                self.dual_updates,
            )?;
            record.metadata["optimizer_steps"]["dual"] = serde_json::to_value(dual.steps)?;
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent_vmpo")?;
        let mut metadata: TemporalOnPolicyRecord<VmpoConfig, P::Architecture> =
            serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let update_mode = record
            .metadata
            .get("update_mode")
            .map(|mode| serde_json::from_value(mode.clone()))
            .transpose()?
            .unwrap_or_default();
        let dual_updates = record
            .metadata
            .get("dual_updates")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::State("missing recurrent V-MPO dual update count".into()))?;
        ensure(
            P::accepts_record_version(metadata.version)
                && record
                    .models
                    .keys()
                    .map(String::as_str)
                    .eq(["multipliers", "policy", "value"])
                && record
                    .optimizers
                    .keys()
                    .map(String::as_str)
                    .eq(["dual", "policy", "value"])
                && metadata.value.is_some()
                && metadata.updates == metadata.policy_updates
                && metadata.updates == metadata.value_updates
                && metadata.updates == dual_updates,
            "invalid recurrent V-MPO model/optimizer owners or counters",
        )?;
        ensure(
            if metadata.version >= 2 {
                metadata
                    .optimizer_owners
                    .keys()
                    .eq(record.optimizers.keys())
            } else {
                metadata.version == 1 && metadata.optimizer_owners.is_empty()
            },
            "invalid recurrent V-MPO optimizer owner manifest",
        )?;
        ensure(
            if P::SPARSE_ADAM {
                metadata.optimizer_steps.keys().eq(record.optimizers.keys())
            } else {
                metadata.optimizer_steps.is_empty()
            },
            "invalid recurrent V-MPO optimizer clock manifest",
        )?;
        let multiplier_record = &record.models["multipliers"];
        ensure(
            multiplier_record.parameter_shapes.len() == 2
                && multiplier_record
                    .parameter_shapes
                    .values()
                    .all(|shape| shape.as_slice() == [1]),
            "invalid recurrent V-MPO multiplier schema",
        )?;
        // Audit all numerical optimizer payloads before allocating any model
        // template. The shared base repeats its policy/value contract below.
        for (optimizer, model, updates) in [
            ("policy", "policy", metadata.policy_updates),
            ("value", "value", metadata.value_updates),
            ("dual", "multipliers", dual_updates),
        ] {
            let expected = if metadata.version >= 2 {
                metadata.optimizer_owners.get(optimizer).map(Vec::as_slice)
            } else {
                None
            };
            let adam = crate::nn::checkpoint::audit_adam_optimizer(
                record, optimizer, model, expected, updates,
            )?;
            if P::SPARSE_ADAM {
                ensure(
                    metadata.optimizer_steps.get(optimizer) == Some(&adam.steps),
                    "recurrent V-MPO optimizer clock manifest differs",
                )?;
                let required: &[&str] = match optimizer {
                    "value" => P::REQUIRED_VALUE_ADAM_PATHS,
                    "dual" => &["log_eta", "log_alpha"],
                    _ => &[],
                };
                validate_required_adam_clocks(record, model, required, &adam.steps, updates)?;
            }
            ensure(
                (P::SPARSE_ADAM && optimizer != "dual")
                    || adam.steps.values().all(|&time| time == updates),
                "recurrent V-MPO Adam history differs from owner update count",
            )?;
        }
        let multipliers = record.load_model("multipliers", &initial_multipliers())?;
        temporal_models::validate_parameters(&multipliers, false)?;
        let dual_optimizer = record.load_optimizer_for_model(
            "dual",
            "multipliers",
            metadata.config.dual.build()?,
            dual_updates > 0,
        )?;
        // The shared state intentionally accepts only its own exact owner set.
        // Validate the complete V-MPO envelope above, then remove its two known
        // additions in a candidate clone; unknown owners are never ignored.
        let mut base = record.clone();
        base.models.remove("multipliers");
        base.optimizers.remove("dual");
        metadata.optimizer_owners.remove("dual");
        metadata.optimizer_steps.remove("dual");
        if metadata.version >= 2 {
            base.metadata["optimizer_owners"] = serde_json::to_value(&metadata.optimizer_owners)?;
        }
        if P::SPARSE_ADAM {
            base.metadata["optimizer_steps"] = serde_json::to_value(&metadata.optimizer_steps)?;
        }
        let state = TemporalOnPolicyState::restored(
            &base,
            &metadata,
            TemporalBatchKind::Bootstrapped,
            &metadata.config.policy,
            Some(&metadata.config.value),
        )?;
        crate::nn::optimization::validate_disjoint(&state.policy, &multipliers)?;
        crate::nn::optimization::validate_disjoint(
            state.value.as_ref().expect("validated V-MPO value"),
            &multipliers,
        )?;
        *self = Self {
            state,
            config: metadata.config,
            multipliers,
            dual_optimizer,
            dual_updates,
            update_mode,
        };
        Ok(())
    }
}
