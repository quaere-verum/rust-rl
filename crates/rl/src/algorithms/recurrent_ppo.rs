//! PPO with independently reconstructed recurrent actor and scalar critic histories.
use super::{
    ppo::{PpoConfig, PpoUpdateSettings, normalize_advantages},
    ppo_joint,
    recurrent_on_policy::{
        RecurrentOnPolicyPolicy, TemporalOnPolicyRecord, TemporalOnPolicyState,
        checked_update_capacity, distribution::RecurrentPolicyDistribution, minibatch::WindowBatch,
    },
};
use crate::data::temporal_on_policy::TemporalOnPolicyLearner;
use crate::{
    Result,
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
        finite_loss, optimization::optimize, target::frozen_copy, temporal_policy::TemporalPolicy,
        to_vec, vector,
    },
    objectives::on_policy::{ppo_clipped_surrogate, ppo_ratio_and_kl},
};
use burn::optim::GradientsParams;
use serde::{Deserialize, Serialize};

pub struct RecurrentPpo<P: RecurrentOnPolicyPolicy = TemporalPolicy> {
    state: TemporalOnPolicyState<P>,
    config: PpoConfig,
    joint: Option<JointOptimization>,
    normalize_minibatch_advantages: bool,
    update_settings: Option<PpoUpdateSettings>,
    update_mode: TemporalExecutionMode,
    minibatches: Option<Vec<PpoMinibatch>>,
}

/// One minibatch in the most recent PPO update, in actual RNG-selected order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PpoMinibatch {
    /// Indexes of windows in the learner's input sequence batch.
    pub windows: Box<[usize]>,
    /// Both optimizers committed this batch. A KL-rejected batch has evaluated
    /// the selected windows but performs no optimizer step.
    pub optimized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JointOptimization {
    value_loss_coefficient: f32,
    max_grad_norm: f32,
}
impl JointOptimization {
    fn validate(&self, config: &PpoConfig) -> Result<()> {
        ensure(
            ppo_joint::valid_settings(config, self.value_loss_coefficient, self.max_grad_norm),
            "joint recurrent PPO requires finite settings and disabled separate clipping",
        )
    }
}

fn validate_settings(settings: &PpoUpdateSettings) -> Result<()> {
    ensure(
        settings.is_valid(),
        "invalid recurrent PPO effective update settings",
    )
}

impl<P: RecurrentOnPolicyPolicy> RecurrentPpo<P> {
    pub fn new(policy: P, value: P::Value, config: PpoConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            state: TemporalOnPolicyState::new(
                policy,
                Some(value),
                &config.policy,
                Some(&config.value),
                seed,
            )?,
            config,
            joint: None,
            normalize_minibatch_advantages: false,
            update_settings: None,
            update_mode: TemporalExecutionMode::Reference,
            minibatches: None,
        })
    }
    pub fn new_with_joint_gradient_clipping(
        policy: P,
        value: P::Value,
        config: PpoConfig,
        seed: u64,
        value_loss_coefficient: f32,
        max_grad_norm: f32,
    ) -> Result<Self> {
        let joint = JointOptimization {
            value_loss_coefficient,
            max_grad_norm,
        };
        joint.validate(&config)?;
        let mut result = Self::new(policy, value, config, seed)?;
        result.joint = Some(joint);
        Ok(result)
    }
    pub fn config(&self) -> &PpoConfig {
        &self.config
    }
    pub fn policy(&self) -> P {
        frozen_copy(&self.state.policy)
    }
    pub fn value(&self) -> P::Value {
        frozen_copy(self.state.value.as_ref().expect("PPO scalar value"))
    }
    pub fn updates(&self) -> u64 {
        self.state.updates
    }
    pub fn revision(&self) -> u64 {
        self.state.revision
    }
    /// Record only the latest update's minibatches for read-only diagnostics.
    /// The recording is local to this learner and is never checkpointed.
    pub fn set_record_minibatches(&mut self, enabled: bool) {
        self.minibatches = enabled.then(Vec::new);
    }
    pub fn last_minibatches(&self) -> Option<&[PpoMinibatch]> {
        self.minibatches.as_deref()
    }
    fn record_minibatch(&mut self, windows: &[usize], optimized: bool) {
        if let Some(minibatches) = &mut self.minibatches {
            minibatches.push(PpoMinibatch {
                windows: windows.into(),
                optimized,
            });
        }
    }
    pub fn update_settings(&self) -> PpoUpdateSettings {
        self.update_settings.unwrap_or(PpoUpdateSettings {
            policy_learning_rate: self.config.policy.learning_rate,
            value_learning_rate: self.config.value.learning_rate,
            clip: self.config.clip,
        })
    }
    pub fn set_update_settings(&mut self, settings: PpoUpdateSettings) -> Result<()> {
        validate_settings(&settings)?;
        self.update_settings = Some(settings);
        Ok(())
    }
    pub fn clear_update_settings(&mut self) {
        self.update_settings = None;
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
    /// Normalize only the fresh learning rows selected in each window minibatch.
    pub fn with_minibatch_advantage_normalization(mut self, enabled: bool) -> Self {
        self.normalize_minibatch_advantages = enabled;
        self
    }
    fn update_inner(
        &mut self,
        batch: &WindowBatch<'_, P::Actor>,
        windows_per_batch: usize,
    ) -> Result<UpdateReport> {
        let settings = self.update_settings();
        let mut report = UpdateReport::default();
        let mut indices = Vec::new();
        'epochs: for _ in 0..self.config.epochs {
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
            for selection in indices.chunks(windows_per_batch) {
                let (selected, tokens, advantages) = {
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::MinibatchPreparation);
                    let selected = batch.select(selection)?;
                    let tokens = selected.tokens();
                    let mut advantages = selected.advantages()?;
                    if self.normalize_minibatch_advantages {
                        normalize_advantages(&mut advantages);
                    }
                    (selected, tokens, advantages)
                };
                // A new graph is reconstructed after every parameter update. No behavior memory
                // or graph from a previous minibatch can leak into this current-policy unroll.
                let distribution =
                    self.state
                        .policy
                        .distribution(&selected, &tokens, self.config.entropy > 0.)?;
                let (logp, cache) = if self.config.entropy > 0. {
                    distribution.log_prob_with_cache(&selected.actions(), &selected.latents())?
                } else {
                    (
                        distribution.log_prob(&selected.actions(), &selected.latents())?,
                        None,
                    )
                };
                let (ratio, kl) = ppo_ratio_and_kl(logp, vector(&selected.old_log_probs()));
                let kl = to_vec(kl)?[0];
                ensure(kl.is_finite(), "nonfinite recurrent PPO approximate KL")?;
                if self.config.target_kl.is_some_and(|limit| kl > limit) {
                    self.record_minibatch(selection, false);
                    report.early_stopped = true;
                    break 'epochs;
                }
                let mut loss = ppo_clipped_surrogate(ratio, vector(&advantages), settings.clip);
                if self.config.entropy > 0. {
                    loss = loss
                        - distribution
                            .entropy_with_cache(cache, &mut self.state.rng)?
                            .mean()
                            * self.config.entropy;
                }
                let returns = selected.returns()?;
                let (p, v) = if let Some(joint) = &self.joint {
                    let value = self.state.value.as_ref().expect("PPO scalar value");
                    let value_loss = (P::values(value, &selected, &tokens)? - vector(&returns))
                        .powf_scalar(2.)
                        .mean();
                    #[cfg(feature = "probes")]
                    let _profile =
                        crate::profiling::enter(crate::profiling::Phase::JointOptimization);
                    let p = finite_loss(&loss)?;
                    let v = finite_loss(&value_loss)?;
                    let mut gradients =
                        (loss + value_loss * joint.value_loss_coefficient).backward();
                    let mut pg = GradientsParams::from_module(&mut gradients, &self.state.policy);
                    let mut vg = GradientsParams::from_module(&mut gradients, value);
                    let norm = ppo_joint::clip(
                        &self.state.policy,
                        value,
                        &mut pg,
                        &mut vg,
                        joint.max_grad_norm,
                    )?;
                    self.state.policy = self.state.policy_optimizer.step(
                        settings.policy_learning_rate,
                        self.state.policy.clone(),
                        pg,
                    );
                    self.state.value = Some(
                        self.state
                            .value_optimizer
                            .as_mut()
                            .expect("PPO value optimizer")
                            .step(settings.value_learning_rate, value.clone(), vg),
                    );
                    report = report.metric("joint_gradient_norm", norm as f64);
                    (p, v)
                } else {
                    let p = {
                        #[cfg(feature = "probes")]
                        let _profile =
                            crate::profiling::enter(crate::profiling::Phase::ActorOptimization);
                        optimize(
                            &mut self.state.policy,
                            &mut self.state.policy_optimizer,
                            settings.policy_learning_rate,
                            loss,
                        )?
                    };
                    let value = self.state.value.as_mut().expect("PPO scalar value");
                    let value_loss = (P::values(value, &selected, &tokens)? - vector(&returns))
                        .powf_scalar(2.)
                        .mean();
                    let v = {
                        #[cfg(feature = "probes")]
                        let _profile =
                            crate::profiling::enter(crate::profiling::Phase::CriticOptimization);
                        optimize(
                            value,
                            self.state
                                .value_optimizer
                                .as_mut()
                                .expect("PPO value optimizer"),
                            settings.value_learning_rate,
                            value_loss,
                        )?
                    };
                    (p, v)
                };
                self.state.updates += 1;
                self.state.policy_updates += 1;
                self.state.value_updates += 1;
                report.updates += 1;
                self.record_minibatch(selection, true);
                report = report
                    .metric("policy_loss", p)
                    .metric("value_loss", v)
                    .metric("approx_kl", kl as f64);
            }
        }
        self.state.validate_models()?;
        self.state.revision += 1;
        Ok(report)
    }
}

impl<P: RecurrentOnPolicyPolicy> BehaviorPublisher for RecurrentPpo<P> {
    type Actor = P::Actor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<P::Actor> {
        ensure(
            revision == self.state.revision,
            "requested stale recurrent PPO behavior revision",
        )?;
        self.state.actor()
    }
}
impl<P: RecurrentOnPolicyPolicy> TemporalOnPolicyLearner<P::Actor> for RecurrentPpo<P> {
    fn collection_kind(&self) -> TemporalBatchKind {
        TemporalBatchKind::Bootstrapped
    }
    fn revision(&self) -> u64 {
        self.state.revision
    }
    fn normalizes_advantages_in_update(&self) -> bool {
        self.normalize_minibatch_advantages
    }
    fn reconstruct_published_state(
        &self,
        actor: &P::Actor,
        history: ReconstructionHistory<'_>,
    ) -> Result<Option<<P::Actor as StatefulActor>::State>> {
        P::reconstruct_published_state(actor, history, self.update_mode())
    }
}
impl<P: RecurrentOnPolicyPolicy> StatefulLearner for RecurrentPpo<P> {
    type Batch = TemporalOnPolicyBatch<P::Actor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        if let Some(minibatches) = &mut self.minibatches {
            minibatches.clear();
        }
        self.config.validate()?;
        validate_settings(&self.update_settings())?;
        self.state
            .preflight(batch, TemporalBatchKind::Bootstrapped)?;
        ensure(
            batch.sequences.config.learn_len <= self.config.minibatch_size,
            "recurrent PPO learning window exceeds the minibatch learning-position limit",
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
        // All input errors above are pure. A later numerical/optimization error may
        // leave partial optimizer work and requires an explicit checkpoint restore.
        let result = self.update_inner(&windows, windows_per_batch);
        if result.is_err() {
            self.state.poisoned = true;
        }
        result
    }
}
impl<P: RecurrentOnPolicyPolicy> Checkpoint for RecurrentPpo<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record(
            "recurrent_ppo",
            TemporalBatchKind::Bootstrapped,
            &self.config,
        )?;
        record.metadata["joint_optimization"] = serde_json::to_value(&self.joint)?;
        record.metadata["normalize_minibatch_advantages"] =
            self.normalize_minibatch_advantages.into();
        record.metadata["update_settings"] = serde_json::to_value(self.update_settings)?;
        if !self.update_mode.is_reference() {
            record.metadata["update_mode"] = serde_json::to_value(self.update_mode)?;
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent_ppo")?;
        let metadata: TemporalOnPolicyRecord<PpoConfig, P::Architecture> =
            serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        ensure(
            metadata.value.is_some()
                && metadata.updates == metadata.policy_updates
                && metadata.updates == metadata.value_updates,
            "invalid recurrent PPO optimizer counters",
        )?;
        let joint: Option<JointOptimization> = serde_json::from_value(
            record
                .metadata
                .get("joint_optimization")
                .cloned()
                .unwrap_or_default(),
        )?;
        let update_settings: Option<PpoUpdateSettings> = serde_json::from_value(
            record
                .metadata
                .get("update_settings")
                .cloned()
                .unwrap_or_default(),
        )?;
        let normalize_minibatch_advantages: bool = serde_json::from_value(
            record
                .metadata
                .get("normalize_minibatch_advantages")
                .cloned()
                .unwrap_or(false.into()),
        )?;
        let update_mode = record
            .metadata
            .get("update_mode")
            .map(|mode| serde_json::from_value(mode.clone()))
            .transpose()?
            .unwrap_or_default();
        if let Some(joint) = &joint {
            joint.validate(&metadata.config)?;
        }
        if let Some(settings) = &update_settings {
            validate_settings(settings)?;
        }
        let state = TemporalOnPolicyState::restored(
            record,
            &metadata,
            TemporalBatchKind::Bootstrapped,
            &metadata.config.policy,
            Some(&metadata.config.value),
        )?;
        *self = Self {
            state,
            config: metadata.config,
            joint,
            normalize_minibatch_advantages,
            update_settings,
            update_mode,
            minibatches: self.minibatches.as_ref().map(|_| Vec::new()),
        };
        Ok(())
    }
}
