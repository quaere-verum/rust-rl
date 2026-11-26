//! Shared recurrent discrete critic: representation, independent target memory,
//! explicit backup choice, fitting and target updates. No behavior policy is owned.
use super::common::optimize;
use crate::{
    Result,
    core::{
        anchors::RepresentationId,
        ensure,
        learner::UpdateReport,
        reconstruction::{
            ReconstructionConfig, ReconstructionDecision, ReconstructionModel,
            ReplayReconstruction, ReplayRepresentations, decode_anchor,
        },
        stateful::HistorySpec,
        temporal::TemporalExecutionMode,
    },
    data::temporal::{SequenceBatch, SequenceOrigin},
    nn::{
        device,
        reconstruction::GruRepresentation,
        recurrent_actor::RecurrentActor,
        target::{frozen_copy, polyak_update},
        temporal::{GruQ, TemporalMasks},
        temporal_batch::encode_history_batch,
        temporal_work::{prepare_exact_window, warm_prefix},
        to_vec, vector,
    },
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::ModuleOptimizer,
    tensor::{Device, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "probes")]
use crate::profiling::{self, Phase};

/// Network fitting and continuation controls, independent of behavior exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentCriticConfig {
    pub optimizer: super::OptimizerConfig,
    pub gamma: f32,
    pub n_step: usize,
    pub target_tau: f32,
    pub target_interval: u64,
}
impl Default for RecurrentCriticConfig {
    fn default() -> Self {
        Self {
            optimizer: Default::default(),
            gamma: 0.99,
            n_step: 3,
            target_tau: 0.01,
            target_interval: 1,
        }
    }
}
impl RecurrentCriticConfig {
    pub fn validate(&self) -> Result<()> {
        self.optimizer.validate()?;
        self.returns().validate()?;
        ensure(
            self.target_interval > 0
                && self.target_tau.is_finite()
                && (0.0..=1.0).contains(&self.target_tau),
            "invalid recurrent critic target controls",
        )
    }
    pub fn returns(&self) -> crate::core::temporal::ReturnSpec {
        crate::core::temporal::ReturnSpec {
            gamma: self.gamma,
            n_step: self.n_step,
        }
    }
}
/// The continuation operator is chosen explicitly for each fit. Policy rows are
/// frozen, masked probabilities in the full padded [batch, time, action] layout.
/// Expectation backups require one-step data; uncorrected behavior returns do not
/// evaluate the supplied policy over multiple steps.
#[derive(Debug, Clone, Copy)]
pub enum CriticBackup<'a> {
    DoubleQOptimality,
    MaximumOptimality,
    PolicyEvaluation(&'a [f32]),
}

mod targets;
pub(super) use targets::{QLayout, construct_targets_layout};
pub use targets::{TargetPoint, construct_policy_targets, construct_targets};
mod checkpoint;
use checkpoint::validate_gru_settings;
#[cfg(test)]
mod tests;

/// Independent recurrent representations; online states are never target states.
pub const ONLINE_REPRESENTATION: RepresentationId = RepresentationId(0);
pub const TARGET_REPRESENTATION: RepresentationId = RepresentationId(1);

/// Compatibility name for the shared temporal execution mode.
pub use crate::core::temporal::TemporalExecutionMode as RecurrentUpdateMode;

pub(super) struct Objective {
    pub(super) loss: Tensor<1>,
    pub(super) targets: Vec<TargetPoint>,
}

/// Current online tape and independent frozen target values in the original
/// dense layout. A caller can also use the detached online values for selection
/// before passing the same tape to regression.
pub(super) struct FullQForward {
    predictions: Tensor<3>,
    pub(super) selection: Vec<f32>,
    evaluation: Vec<f32>,
}
type InitialStates = (
    Tensor<2>,
    Tensor<2>,
    Vec<(ReconstructionDecision, ReconstructionDecision)>,
);

pub struct RecurrentDiscreteCritic {
    pub(super) online: GruQ,
    pub(super) target: GruQ,
    optimizer: ModuleOptimizer,
    pub(super) spec: HistorySpec,
    config: RecurrentCriticConfig,
    updates: u64,
    pub(super) update_mode: TemporalExecutionMode,
    // Derived immutable inference snapshot; never stores recurrent state. Its
    // omission from checkpoints cannot affect RNG or numerical execution.
    behavior_cache: std::cell::RefCell<Option<RecurrentActor>>,
    target_cache: std::cell::RefCell<Option<RecurrentActor>>,
    reconstruction: ReconstructionConfig,
}
impl ReplayRepresentations for RecurrentDiscreteCritic {
    type Model = GruRepresentation;
    fn replay_representations(&self) -> Result<Vec<(RepresentationId, Self::Model)>> {
        let online = GruRepresentation::new(self.q_actor(0., 0)?, self.updates)?;
        let cached = self.target_cache.borrow().clone();
        let target = if let Some(actor) = cached {
            actor
        } else {
            let actor = RecurrentActor::new(self.target.clone(), self.spec.clone(), 0., 0)?;
            *self.target_cache.borrow_mut() = Some(actor.clone());
            actor
        };
        Ok(vec![
            (ONLINE_REPRESENTATION, online),
            (
                TARGET_REPRESENTATION,
                GruRepresentation::new(target, self.updates)?,
            ),
        ])
    }
}
impl ReplayReconstruction for RecurrentDiscreteCritic {
    fn reconstruction_config(&self) -> ReconstructionConfig {
        self.reconstruction
    }
}
impl RecurrentDiscreteCritic {
    pub fn new(online: GruQ, spec: HistorySpec, config: RecurrentCriticConfig) -> Result<Self> {
        spec.validate()?;
        config.validate()?;
        validate_gru_settings(&online)?;
        #[derive(Default)]
        struct Trainable {
            count: usize,
            valid: bool,
        }
        impl ModuleVisitor for Trainable {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                let t = p.val();
                self.count += 1;
                self.valid &= t.is_autodiff() && t.is_require_grad();
            }
        }
        let mut trainable = Trainable {
            count: 0,
            valid: true,
        };
        online.visit(&mut trainable);
        ensure(
            trainable.count > 0 && trainable.valid,
            "recurrent learner requires trainable autodiff parameters; inference policies cannot be optimized",
        )?;
        parameter_values(&online)?;
        ensure(
            online.input_dim() == spec.input_dim()
                && spec.action_space
                    == (crate::core::action::ActionSpace::Discrete {
                        count: online.action_count(),
                    }),
            "recurrent Q/history specification mismatch",
        )?;
        let target = frozen_copy(&online);
        let optimizer = config.optimizer.build()?;
        Ok(Self {
            online,
            target,
            optimizer,
            spec,
            config,
            updates: 0,
            update_mode: TemporalExecutionMode::Reference,
            behavior_cache: std::cell::RefCell::new(None),
            target_cache: std::cell::RefCell::new(None),
            reconstruction: ReconstructionConfig::default(),
        })
    }
    pub fn spec(&self) -> &HistorySpec {
        &self.spec
    }
    pub fn config(&self) -> &RecurrentCriticConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn update_mode(&self) -> TemporalExecutionMode {
        self.update_mode
    }
    pub fn set_update_mode(&mut self, mode: TemporalExecutionMode) {
        self.update_mode = mode;
    }
    pub fn reconstruction_config(&self) -> ReconstructionConfig {
        self.reconstruction
    }
    pub fn set_reconstruction_config(&mut self, config: ReconstructionConfig) -> Result<()> {
        config.validate()?;
        self.reconstruction = config;
        Ok(())
    }
    pub fn online(&self) -> GruQ {
        frozen_copy(&self.online)
    }
    pub fn target(&self) -> GruQ {
        frozen_copy(&self.target)
    }
    pub fn q_actor(&self, epsilon: f32, revision: u64) -> Result<RecurrentActor> {
        if let Some(actor) = self.behavior_cache.borrow().as_ref() {
            return actor.with_behavior(epsilon, revision);
        }
        let actor = RecurrentActor::new(self.online.clone(), self.spec.clone(), epsilon, revision)?
            .with_parameter_generation(self.updates);
        *self.behavior_cache.borrow_mut() = Some(actor.clone());
        Ok(actor)
    }
    pub fn parameter_values(&self) -> Result<Vec<f32>> {
        parameter_values(&self.online)
    }
    /// Independent online/target reconstruction from genuine reset priors or
    /// explicitly decoded anchors. Both consume the same recorded public history.
    pub fn reconstructed_q_values(&self, batch: &SequenceBatch) -> Result<(Vec<f32>, Vec<f32>)> {
        let encoded = encode_history_batch(batch, &self.spec)?;
        let (online_initial, target_initial, _) = self.initial_states(batch)?;
        let plain = Device::flex();
        let input = Tensor::<3>::from_data(
            TensorData::new(
                encoded.inputs,
                [encoded.batch, encoded.time, encoded.input_dim],
            ),
            &plain,
        );
        let online = self.online.valid();
        let target = self.target.valid();
        let (qo, _) = {
            #[cfg(feature = "probes")]
            let _history = profiling::enter(Phase::OnlineQHistory);
            online.unroll(input.clone(), online_initial, &encoded.masks)?
        };
        let (qt, _) = {
            #[cfg(feature = "probes")]
            let _history = profiling::enter(Phase::TargetQHistory);
            target.unroll(input, target_initial, &encoded.masks)?
        };
        Ok((to_vec(qo)?, to_vec(qt)?))
    }
    /// Decoding is model-specific and graph-free. Each anchor is a prior for the
    /// FIRST retained observation, not a state that already assimilated it.
    fn initial_states(&self, batch: &SequenceBatch) -> Result<InitialStates> {
        self.reconstruction.validate()?;
        if batch
            .samples
            .iter()
            .all(|s| matches!(s.origin, SequenceOrigin::Reset))
        {
            let shape = [batch.samples.len(), self.online.hidden_dim()];
            return Ok((
                Tensor::zeros(shape, &Device::flex()),
                Tensor::zeros(shape, &Device::flex()),
                Vec::new(),
            ));
        }
        let sources = self.replay_representations()?;
        let online = &sources[0].1;
        let target = &sources[1].1;
        let mut ho = Vec::new();
        let mut ht = Vec::new();
        let mut decisions = Vec::new();
        for sample in &batch.samples {
            match &sample.origin {
                SequenceOrigin::Reset => {
                    ho.extend(online.initial_state()?);
                    ht.extend(target.initial_state()?);
                }
                SequenceOrigin::Anchored(set) => {
                    let get = |id| {
                        set.anchors.get(&id).ok_or_else(|| {
                            crate::Error::State(
                                "missing independent online/target reconstruction anchor".into(),
                            )
                        })
                    };
                    let (o, od) = decode_anchor(
                        self.reconstruction,
                        online,
                        ONLINE_REPRESENTATION,
                        sample.episode_id,
                        sample.history_start,
                        get(ONLINE_REPRESENTATION)?,
                    )?;
                    let (t, td) = decode_anchor(
                        self.reconstruction,
                        target,
                        TARGET_REPRESENTATION,
                        sample.episode_id,
                        sample.history_start,
                        get(TARGET_REPRESENTATION)?,
                    )?;
                    ho.extend(o);
                    ht.extend(t);
                    decisions.push((od, td));
                }
                SequenceOrigin::Stateless => {
                    return Err(crate::Error::State(
                        "recurrent critic requires a reconstruction prefix or anchor".into(),
                    ));
                }
            }
        }
        let shape = [batch.samples.len(), self.online.hidden_dim()];
        Ok((
            Tensor::from_data(TensorData::new(ho, shape), &Device::flex()),
            Tensor::from_data(TensorData::new(ht, shape), &Device::flex()),
            decisions,
        ))
    }
    pub fn reconstruction_decisions(
        &self,
        batch: &SequenceBatch,
    ) -> Result<Vec<(ReconstructionDecision, ReconstructionDecision)>> {
        batch.validate()?;
        Ok(self.initial_states(batch)?.2)
    }
    /// Fit the online network and update its independent target on the configured cadence.
    pub fn fit(&mut self, batch: &SequenceBatch, backup: CriticBackup<'_>) -> Result<UpdateReport> {
        match backup {
            CriticBackup::DoubleQOptimality => self.update_optimality(batch),
            CriticBackup::MaximumOptimality => {
                self.config.validate()?;
                batch.validate()?;
                ensure(
                    batch.config.n_step == self.config.n_step,
                    "recurrent batch n-step contract mismatch",
                )?;
                let encoded = encode_history_batch(batch, &self.spec)?;
                let objective = self.two_pass_objective(batch, &encoded, backup)?;
                self.apply_objective(objective)
            }
            CriticBackup::PolicyEvaluation(probabilities) => {
                self.update_for_policy(batch, probabilities)
            }
        }
    }
    pub fn return_contract(&self) -> crate::core::temporal::ReturnSpec {
        self.config.returns()
    }
    fn update_optimality(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        self.config.validate()?;
        batch.validate()?;
        ensure(
            batch.config.n_step == self.config.n_step,
            "recurrent batch n-step contract mismatch",
        )?;
        let objective = match self.update_mode {
            TemporalExecutionMode::Reference => self.reference_objective(batch)?,
            TemporalExecutionMode::OptimizedExact => self.optimized_objective(batch)?,
        };
        self.apply_objective(objective)
    }

    fn apply_objective(&mut self, objective: Objective) -> Result<UpdateReport> {
        let next_update = self
            .updates
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("recurrent update counter overflow".into()))?;
        // Invalidate before any optimizer mutation, including a partially failed
        // update; an unchanged counter must never authorize stale parameters.
        self.behavior_cache.borrow_mut().take();
        let number = {
            #[cfg(feature = "probes")]
            let _optimizer = profiling::enter(Phase::CriticOptimization);
            optimize(
                &mut self.online,
                &mut self.optimizer,
                self.config.optimizer.learning_rate,
                objective.loss,
            )?
        };
        self.updates = next_update;
        if self.updates.is_multiple_of(self.config.target_interval) {
            self.target_cache.borrow_mut().take();
            self.target = polyak_update(self.target.clone(), &self.online, self.config.target_tau)?;
        }
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("critic_loss", number)
        .metric("learning_positions", objective.targets.len() as f64)
        .metric(
            "target_mean",
            objective
                .targets
                .iter()
                .map(|p| f64::from(p.value))
                .sum::<f64>()
                / objective.targets.len() as f64,
        ))
    }

    /// Evaluate an explicitly supplied frozen policy, with the native Q optimizer.
    fn update_for_policy(
        &mut self,
        batch: &SequenceBatch,
        probabilities: &[f32],
    ) -> Result<UpdateReport> {
        self.config.validate()?;
        batch.validate()?;
        ensure(
            self.config.n_step == 1 && batch.config.n_step == 1,
            "policy-evaluation update requires one-step return contracts",
        )?;
        let objective = if self.update_mode.is_reference() {
            self.reference_objective_for_policy(batch, Some(probabilities))?
        } else {
            let encoded = encode_history_batch(batch, &self.spec)?;
            self.two_pass_objective(
                batch,
                &encoded,
                CriticBackup::PolicyEvaluation(probabilities),
            )?
        };
        self.apply_objective(objective)
    }

    pub(super) fn reference_objective(&self, batch: &SequenceBatch) -> Result<Objective> {
        self.reference_objective_for_policy(batch, None)
    }

    fn reference_objective_for_policy(
        &self,
        batch: &SequenceBatch,
        policy: Option<&[f32]>,
    ) -> Result<Objective> {
        let (selection, evaluation) = self.reconstructed_q_values(batch)?;
        let targets = if let Some(probabilities) = policy {
            construct_policy_targets(
                batch,
                probabilities,
                &evaluation,
                self.online.action_count(),
                self.config.gamma,
            )?
        } else {
            construct_targets(
                batch,
                &selection,
                &evaluation,
                self.online.action_count(),
                self.config.gamma,
            )?
        };
        let encoded = encode_history_batch(batch, &self.spec)?;
        let input = Tensor::<3>::from_data(
            TensorData::new(
                encoded.inputs,
                [encoded.batch, encoded.time, encoded.input_dim],
            ),
            &device(),
        );
        let (predictions, _) = {
            #[cfg(feature = "probes")]
            let _history = profiling::enter(Phase::OnlineQHistory);
            self.online.unroll(
                input,
                Tensor::from_data(self.initial_states(batch)?.0.into_data(), &device()),
                &encoded.masks,
            )?
        };
        Ok(Self::regression_objective(predictions, targets, None))
    }

    pub(super) fn optimized_objective(&self, batch: &SequenceBatch) -> Result<Objective> {
        let prepared = prepare_exact_window(batch, &self.spec)?;
        let full = &prepared.prefix;
        let suffix = &prepared.suffix;
        // For heterogeneous short prefixes, warming max(prefix) then unrolling
        // max(suffix) can perform MORE batched recurrent steps than the original
        // full tape. Reuse one differentiable online pass instead. Long prefixes
        // still use encoder-only warmup to bound the autodiff graph.
        if prepared.starts.iter().copied().max().unwrap_or(0) < suffix.time {
            return self.two_pass_objective(batch, full, CriticBackup::DoubleQOptimality);
        }
        let (online_initial, target_initial, _) = self.initial_states(batch)?;
        let plain = Device::flex();
        let input = Tensor::<3>::from_data(
            TensorData::new(full.inputs.clone(), [full.batch, full.time, full.input_dim]),
            &plain,
        );
        let online = self.online.valid();
        let target = self.target.valid();
        // These are independent current-model encoder histories, never a target
        // state borrowed from the online network or an old behavior revision.
        let ho = warm_prefix(
            &online.encoder,
            input.clone(),
            online_initial,
            &prepared.starts,
            &full.masks,
        )?;
        let ht = warm_prefix(
            &target.encoder,
            input,
            target_initial,
            &prepared.starts,
            &full.masks,
        )?;
        let input = Tensor::<3>::from_data(
            TensorData::new(
                suffix.inputs.clone(),
                [suffix.batch, suffix.time, suffix.input_dim],
            ),
            &device(),
        );
        let initial = Tensor::<2>::from_data(ho.into_data(), &device());
        let (prediction, _) = self.online.unroll(input, initial, &suffix.masks)?;
        let input = Tensor::<3>::from_data(
            TensorData::new(
                suffix.inputs.clone(),
                [suffix.batch, suffix.time, suffix.input_dim],
            ),
            &plain,
        );
        let (evaluation, _) = target.unroll(input, ht, &suffix.masks)?;
        // Reuse this differentiable online forward for selection. Numeric values
        // cross a detached host boundary before target construction.
        let selection = to_vec(prediction.clone().without_autodiff())?;
        let evaluation = to_vec(evaluation)?;
        let targets = construct_targets_layout(
            batch,
            &selection,
            &evaluation,
            self.online.action_count(),
            self.config.gamma,
            QLayout {
                time: suffix.time,
                width: self.spec.observation_dim,
                offsets: &prepared.starts,
            },
        )?;
        Ok(Self::regression_objective(
            prediction,
            targets,
            Some(&prepared.starts),
        ))
    }

    fn two_pass_objective(
        &self,
        batch: &SequenceBatch,
        encoded: &crate::nn::temporal_batch::EncodedHistory,
        backup: CriticBackup<'_>,
    ) -> Result<Objective> {
        let input = Tensor::<3>::from_data(
            TensorData::new(
                encoded.inputs.clone(),
                [encoded.batch, encoded.time, encoded.input_dim],
            ),
            &device(),
        );
        let forward = self.full_q_forward(batch, input, &encoded.masks, None)?;
        self.full_objective(batch, backup, forward)
    }

    pub(super) fn full_q_forward(
        &self,
        batch: &SequenceBatch,
        input: Tensor<3>,
        masks: &TemporalMasks,
        target_history: Option<&crate::nn::temporal_work::EpisodeInferencePlan>,
    ) -> Result<FullQForward> {
        let (online_initial, target_initial, _) = self.initial_states(batch)?;
        let evaluation = {
            #[cfg(feature = "probes")]
            let _history = profiling::enter(Phase::TargetQHistory);
            let target = self.target.valid();
            if let Some(history) = target_history {
                target.values_from_history_features(history.features(&target.encoder)?)?
            } else {
                target
                    .unroll(input.clone().without_autodiff(), target_initial, masks)?
                    .0
            }
        };
        let (predictions, _) = {
            #[cfg(feature = "probes")]
            let _history = profiling::enter(Phase::OnlineQHistory);
            self.online.unroll(
                input,
                Tensor::from_data(online_initial.into_data(), &device()),
                masks,
            )?
        };
        Ok(FullQForward {
            selection: to_vec(predictions.clone().without_autodiff())?,
            evaluation: to_vec(evaluation)?,
            predictions,
        })
    }

    pub(super) fn fit_from_full_forward(
        &mut self,
        batch: &SequenceBatch,
        backup: CriticBackup<'_>,
        forward: FullQForward,
    ) -> Result<UpdateReport> {
        let objective = self.full_objective(batch, backup, forward)?;
        self.apply_objective(objective)
    }

    fn full_objective(
        &self,
        batch: &SequenceBatch,
        backup: CriticBackup<'_>,
        forward: FullQForward,
    ) -> Result<Objective> {
        let [lanes, time, actions] = forward.predictions.dims();
        let offsets = vec![0; lanes];
        let targets = match backup {
            CriticBackup::DoubleQOptimality | CriticBackup::MaximumOptimality => {
                construct_targets_layout(
                    batch,
                    if matches!(backup, CriticBackup::MaximumOptimality) {
                        &forward.evaluation
                    } else {
                        &forward.selection
                    },
                    &forward.evaluation,
                    actions,
                    self.config.gamma,
                    QLayout {
                        time,
                        width: self.spec.observation_dim,
                        offsets: &offsets,
                    },
                )?
            }
            CriticBackup::PolicyEvaluation(probabilities) => construct_policy_targets(
                batch,
                probabilities,
                &forward.evaluation,
                actions,
                self.config.gamma,
            )?,
        };
        Ok(Self::regression_objective(
            forward.predictions,
            targets,
            None,
        ))
    }

    fn regression_objective(
        predictions: Tensor<3>,
        targets: Vec<TargetPoint>,
        offsets: Option<&[usize]>,
    ) -> Objective {
        let [lanes, time, action_count] = predictions.dims();
        let rows: Vec<_> = targets
            .iter()
            .map(|p| (p.row * time + p.timestep - offsets.map_or(0, |rows| rows[p.row])) as i64)
            .collect();
        let actions: Vec<_> = targets.iter().map(|p| p.action as i64).collect();
        let values: Vec<_> = targets.iter().map(|p| p.value).collect();
        let rows = Tensor::<1, Int>::from_data(TensorData::new(rows, [targets.len()]), &device());
        let indices =
            Tensor::<2, Int>::from_data(TensorData::new(actions, [targets.len(), 1]), &device());
        let prediction = predictions
            .reshape([lanes * time, action_count])
            .select(0, rows)
            .gather(1, indices)
            .squeeze_dim(1);
        let abs = (prediction - vector(&values)).abs();
        let quadratic = abs.clone().clamp(0., 1.);
        let loss = (quadratic.clone().powf_scalar(2.) * 0.5 + abs - quadratic).mean();
        Objective { loss, targets }
    }
}
pub(super) fn parameter_values(model: &impl Module) -> Result<Vec<f32>> {
    #[derive(Default)]
    struct Collect {
        values: Vec<f32>,
        error: Option<crate::Error>,
    }
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let t = p.val().without_autodiff();
            let n = t.dims().iter().product::<usize>();
            match to_vec(t.reshape([n])) {
                Ok(v) => self.values.extend(v),
                Err(e) => self.error = Some(e),
            }
        }
    }
    let mut c = Collect::default();
    model.visit(&mut c);
    if let Some(e) = c.error {
        return Err(e);
    }
    ensure(
        c.values.iter().all(|v| v.is_finite()),
        "nonfinite recurrent model parameters",
    )?;
    Ok(c.values)
}
