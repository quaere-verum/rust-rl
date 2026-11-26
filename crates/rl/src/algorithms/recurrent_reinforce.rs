//! Complete-episode REINFORCE with independent recurrent policy/value histories.
use super::{
    recurrent_on_policy::{
        RecurrentOnPolicyPolicy, TemporalOnPolicyRecord, TemporalOnPolicyState, checked_adam_steps,
        distribution::RecurrentPolicyDistribution, minibatch::Minibatch,
    },
    reinforce::ReinforceConfig,
};
use crate::data::temporal_on_policy::TemporalOnPolicyLearner;
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, StatefulLearner},
        temporal::TemporalExecutionMode,
    },
    data::temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    nn::{optimization::optimize, target::frozen_copy, temporal_policy::TemporalPolicy, vector},
};

pub struct RecurrentReinforce<P: RecurrentOnPolicyPolicy = TemporalPolicy> {
    state: TemporalOnPolicyState<P>,
    config: ReinforceConfig,
    update_mode: TemporalExecutionMode,
}

impl<P: RecurrentOnPolicyPolicy> RecurrentReinforce<P> {
    pub fn new(
        policy: P,
        value: Option<P::Value>,
        config: ReinforceConfig,
        seed: u64,
    ) -> Result<Self> {
        config.validate()?;
        ensure(
            config.baseline == value.is_some(),
            "recurrent REINFORCE baseline setting and value model differ",
        )?;
        Ok(Self {
            state: TemporalOnPolicyState::new(
                policy,
                value,
                &config.policy,
                config.baseline.then_some(&config.value),
                seed,
            )?,
            config,
            update_mode: TemporalExecutionMode::Reference,
        })
    }

    pub fn config(&self) -> &ReinforceConfig {
        &self.config
    }

    pub fn policy(&self) -> P {
        frozen_copy(&self.state.policy)
    }

    pub fn value(&self) -> Option<P::Value> {
        self.state.value.as_ref().map(frozen_copy)
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

    fn update_inner(
        &mut self,
        batch: &TemporalOnPolicyBatch<P::Actor>,
        history: &Minibatch<'_, P::Actor>,
    ) -> Result<UpdateReport> {
        let tokens = history.tokens();
        let returns = batch.returns()?;
        // Targets and the collected behavior baseline are host values. They do
        // not acquire the current value encoder's graph or any value-loss gradient.
        let advantage = if self.config.baseline {
            let baseline = batch.old_values()?;
            vector(&returns) - vector(&baseline)
        } else {
            vector(&returns)
        };
        let weights: Vec<_> = batch.rows.iter().map(|row| row.discount_weight).collect();
        let distribution =
            self.state
                .policy
                .distribution(history, &tokens, self.config.entropy > 0.)?;
        let mut loss = (distribution.log_prob(&batch.actions(), &batch.latents())?
            * advantage
            * vector(&weights))
        .sum()
        .neg()
            / batch.episode_count as f32;
        if self.config.entropy > 0. {
            // Entropy follows the existing REINFORCE row mean, independent of
            // gamma weighting and episode-count policy-gradient normalization.
            loss = loss - distribution.entropy(&mut self.state.rng)?.mean() * self.config.entropy;
        }
        let policy_loss = {
            #[cfg(feature = "probes")]
            let _profile = crate::profiling::enter(crate::profiling::Phase::ActorOptimization);
            optimize(
                &mut self.state.policy,
                &mut self.state.policy_optimizer,
                self.config.policy.learning_rate,
                loss,
            )?
        };
        let mut report = UpdateReport::default().metric("policy_loss", policy_loss);
        if self.config.baseline {
            let value = self
                .state
                .value
                .as_mut()
                .expect("validated scalar baseline");
            let value_loss = (P::values(value, history, &tokens)? - vector(&returns))
                .powf_scalar(2.)
                .mean();
            let number = {
                #[cfg(feature = "probes")]
                let _profile = crate::profiling::enter(crate::profiling::Phase::CriticOptimization);
                optimize(
                    value,
                    self.state
                        .value_optimizer
                        .as_mut()
                        .expect("validated value optimizer"),
                    self.config.value.learning_rate,
                    value_loss,
                )?
            };
            report = report.metric("value_loss", number);
        }
        self.state.validate_models()?;
        self.state.updates += 1;
        self.state.policy_updates += 1;
        self.state.value_updates += u64::from(self.config.baseline);
        self.state.revision += 1;
        report.updates = 1;
        Ok(report)
    }
}

impl<P: RecurrentOnPolicyPolicy> BehaviorPublisher for RecurrentReinforce<P> {
    type Actor = P::Actor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<P::Actor> {
        ensure(
            revision == self.state.revision,
            "requested stale recurrent REINFORCE behavior revision",
        )?;
        self.state.actor()
    }
}

impl<P: RecurrentOnPolicyPolicy> TemporalOnPolicyLearner<P::Actor> for RecurrentReinforce<P> {
    fn collection_kind(&self) -> TemporalBatchKind {
        TemporalBatchKind::Episodic
    }
    fn revision(&self) -> u64 {
        self.state.revision
    }
}

impl<P: RecurrentOnPolicyPolicy> StatefulLearner for RecurrentReinforce<P> {
    type Batch = TemporalOnPolicyBatch<P::Actor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        self.config.validate()?;
        self.state.preflight(batch, TemporalBatchKind::Episodic)?;
        validate_complete_episodes(batch, self.config.baseline)?;
        ensure(
            self.state.updates.checked_add(1).is_some()
                && self.state.policy_updates.checked_add(1).is_some()
                && (!self.config.baseline || self.state.value_updates.checked_add(1).is_some()),
            "recurrent REINFORCE update counter overflow",
        )?;
        checked_adam_steps(self.state.policy_updates, 1)?;
        if self.config.baseline {
            checked_adam_steps(self.state.value_updates, 1)?;
        }
        let history = self
            .state
            .policy
            .windows(batch, self.update_mode(), self.state.value.as_ref())?
            .full()?;
        let result = self.update_inner(batch, &history);
        if result.is_err() {
            self.state.poisoned = true;
        }
        result
    }
}

impl<P: RecurrentOnPolicyPolicy> Checkpoint for RecurrentReinforce<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = self.state.record(
            "recurrent_reinforce",
            TemporalBatchKind::Episodic,
            &self.config,
        )?;
        if !self.update_mode.is_reference() {
            record.metadata["update_mode"] = serde_json::to_value(self.update_mode)?;
        }
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent_reinforce")?;
        let metadata: TemporalOnPolicyRecord<ReinforceConfig, P::Architecture> =
            serde_json::from_value(record.metadata.clone())?;
        metadata.config.validate()?;
        let update_mode = record
            .metadata
            .get("update_mode")
            .map(|mode| serde_json::from_value(mode.clone()))
            .transpose()?
            .unwrap_or_default();
        ensure(
            metadata.value.is_some() == metadata.config.baseline
                && metadata.updates == metadata.policy_updates
                && metadata.updates == metadata.revision
                && metadata.value_updates
                    == if metadata.config.baseline {
                        metadata.updates
                    } else {
                        0
                    },
            "invalid recurrent REINFORCE model capabilities or optimizer counters",
        )?;
        let state = TemporalOnPolicyState::restored(
            record,
            &metadata,
            TemporalBatchKind::Episodic,
            &metadata.config.policy,
            metadata.config.baseline.then_some(&metadata.config.value),
        )?;
        *self = Self {
            state,
            config: metadata.config,
            update_mode,
        };
        Ok(())
    }
}

/// REINFORCE consumes complete episodes once. A selected subset of recurrent
/// windows remains useful to other objectives, but cannot masquerade as a
/// complete Monte Carlo collection merely by retaining its episode-count field.
fn validate_complete_episodes<A>(batch: &TemporalOnPolicyBatch<A>, baseline: bool) -> Result<()> {
    ensure(
        batch.kind == TemporalBatchKind::Episodic && batch.gae_lambda.is_none(),
        "recurrent REINFORCE requires Monte Carlo episode targets",
    )?;
    let mut first = 0;
    let mut episodes = 0usize;
    while first < batch.rows.len() {
        let id = batch.rows[first].key.episode_id;
        let end = first
            + batch.rows[first..]
                .iter()
                .take_while(|row| row.key.episode_id == id)
                .count();
        let last = &batch.rows[end - 1];
        let reference = &batch.sequences.samples[last.sample];
        let start = reference.trajectory_start;
        ensure(
            reference.steps.len().checked_sub(start) == Some(end - first),
            "REINFORCE episode prefix or learning rows are incomplete",
        )?;
        for (time, row) in batch.rows[first..end].iter().enumerate() {
            let step = &batch.sequences.samples[row.sample].steps[row.time];
            ensure(
                row.key.timestep == start + time
                    && row.value.is_some() == baseline
                    && row.next_value.is_some() == baseline
                    && !step.truncated
                    && step.terminated == (time + 1 == end - first),
                "REINFORCE rejects truncated, incomplete or fragmented episodes and mismatched baselines",
            )?;
        }
        for sample in batch
            .sequences
            .samples
            .iter()
            .filter(|sample| sample.episode_id == id)
        {
            ensure(
                sample.steps.len() <= reference.steps.len()
                    && sample.factual_prefixes_equal(reference),
                "REINFORCE windows disagree about the actual public episode prefix",
            )?;
        }
        episodes = episodes
            .checked_add(1)
            .ok_or_else(|| crate::Error::State("REINFORCE episode count overflow".into()))?;
        first = end;
    }
    ensure(
        episodes > 0 && episodes == batch.episode_count,
        "REINFORCE episode-count normalization differs from complete episode coverage",
    )
}
