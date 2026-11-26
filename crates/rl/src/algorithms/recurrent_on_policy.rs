//! Shared numerical state and staged persistence; learner objectives remain separate.
use super::common::OptimizerConfig;
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Error, Result,
    core::{
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        stateful::{ReconstructionHistory, StatefulActor},
        stateful_on_policy::{recording_distribution_matches, recording_number_matches},
        temporal::TemporalExecutionMode,
    },
    data::temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer, record_error},
        temporal_models::{self, TemporalModelArchitecture},
        temporal_policy::TemporalPolicy,
        to_vec,
    },
};
use burn::{
    module::Module,
    optim::ModuleOptimizer,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

mod dense;
pub(crate) mod distribution;
mod lightweight;
pub(super) mod minibatch;
#[cfg(test)]
mod numerics;
pub(super) mod structured;
pub(super) mod structured_actor;

use distribution::RecurrentPolicyDistribution;
pub use lightweight::LightweightTemporalActor;
use minibatch::{Minibatch, WindowBatch};
pub use structured_actor::{StructuredTemporalActor, StructuredTemporalActorState};

/// Trainable policy and its independent scalar value model for a temporal
/// on-policy learner. The model determines how factual windows are evaluated.
pub trait RecurrentOnPolicyPolicy: Module {
    type Value: Module;
    type Actor: OnPolicyIdentity + Snapshot + Clone;
    type Architecture: Serialize + DeserializeOwned + PartialEq;
    type Distribution: RecurrentPolicyDistribution;

    const RECORD_VERSION: u32;
    const SPARSE_ADAM: bool = false;
    /// Model record paths for parameters used in every owner optimization step.
    const REQUIRED_POLICY_ADAM_PATHS: &'static [&'static str] = &[];
    const REQUIRED_VALUE_ADAM_PATHS: &'static [&'static str] = &[];

    fn policy_device(&self) -> Device;
    fn architectures(
        &self,
        value: Option<&Self::Value>,
    ) -> Result<(Self::Architecture, Option<Self::Architecture>)>;
    fn publish(policy: Self, value: Option<Self::Value>, revision: u64) -> Result<Self::Actor>;
    fn published_models(actor: &Self::Actor) -> (&Self, Option<&Self::Value>);
    fn with_revision(actor: &Self::Actor, revision: u64) -> Self::Actor;
    fn binding(actor: &Self::Actor, kind: TemporalBatchKind) -> Result<TemporalLearnerBinding>;
    fn restore_models(
        record: &LearnerRecord,
        policy: &Self::Architecture,
        value: Option<&Self::Architecture>,
        binding: &TemporalLearnerBinding,
    ) -> Result<(Self, Option<Self::Value>)>;

    /// Whether these models can use the optimized exact learning path.
    fn supports_optimized_exact(&self, _value: Option<&Self::Value>) -> bool {
        false
    }

    fn windows<'a>(
        &self,
        batch: &'a TemporalOnPolicyBatch<Self::Actor>,
        _mode: TemporalExecutionMode,
        _value: Option<&Self::Value>,
    ) -> Result<WindowBatch<'a, Self::Actor>> {
        Ok(WindowBatch::reference(batch))
    }
    /// Evaluate the learning tokens in caller order. Complete support is needed
    /// for KL, entropy, or snapshot verification. A likelihood-only evaluation
    /// may restrict its work to the recorded keys.
    fn distribution(
        &self,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
        complete_support: bool,
    ) -> Result<Self::Distribution>;
    /// Return one scalar state value for each token in caller order.
    fn values(
        value: &Self::Value,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<1>>;
    fn reconstruct_published_state(
        _actor: &Self::Actor,
        _history: ReconstructionHistory<'_>,
        _mode: TemporalExecutionMode,
    ) -> Result<Option<<Self::Actor as StatefulActor>::State>> {
        Ok(None)
    }

    fn accepts_record_version(version: u32) -> bool {
        version == Self::RECORD_VERSION || (Self::RECORD_VERSION == 2 && version == 1)
    }
}

pub(crate) struct TemporalOnPolicyState<P: RecurrentOnPolicyPolicy = TemporalPolicy> {
    pub policy: P,
    pub value: Option<P::Value>,
    pub policy_optimizer: ModuleOptimizer,
    pub value_optimizer: Option<ModuleOptimizer>,
    pub rng: ChaCha8Rng,
    pub updates: u64,
    pub policy_updates: u64,
    pub value_updates: u64,
    pub revision: u64,
    pub poisoned: bool,
    // Reconstructible immutable models only: no lane memory or random stream.
    // A restored state always owns a new cache, even when counters coincide.
    behavior_cache: RefCell<Option<BehaviorSnapshot<P::Actor>>>,
}

struct BehaviorSnapshot<A> {
    updates: (u64, u64),
    actor: A,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TemporalOnPolicyRecord<C, A = TemporalModelArchitecture> {
    pub version: u32,
    pub config: C,
    pub policy: A,
    pub value: Option<A>,
    pub collection: TemporalLearnerBinding,
    pub rng: ChaCha8Rng,
    pub updates: u64,
    pub policy_updates: u64,
    pub value_updates: u64,
    pub revision: u64,
    /// Version one records have no membership manifest. Later records have
    /// an exact sorted entry for each policy/value optimizer, including empty ones.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub optimizer_owners: BTreeMap<String, Vec<u64>>,
    /// Structured heads can skip steps when their conditional branch is absent.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub optimizer_steps: BTreeMap<String, BTreeMap<u64, u64>>,
}

impl<P: RecurrentOnPolicyPolicy> TemporalOnPolicyState<P> {
    pub(super) fn supports_optimized_exact(&self) -> bool {
        self.policy.supports_optimized_exact(self.value.as_ref())
    }

    pub fn new(
        policy: P,
        value: Option<P::Value>,
        policy_config: &OptimizerConfig,
        value_config: Option<&OptimizerConfig>,
        seed: u64,
    ) -> Result<Self> {
        ensure(
            value.is_some() == value_config.is_some(),
            "temporal value optimizer capability mismatch",
        )?;
        policy.architectures(value.as_ref())?;
        temporal_models::validate_parameters(&policy, false)?;
        if let Some(value) = &value {
            temporal_models::validate_parameters(value, false)?;
            crate::nn::optimization::validate_disjoint(&policy, value)?;
        }
        Ok(Self {
            // Frozen behavior getters retain disabled gradient configuration even
            // across valid/fork/train. Explicitly unfreeze independent training
            // copies before restoring autodiff; inference inputs remain frozen.
            policy: policy.valid().fork(&crate::nn::device()).unfreeze().train(),
            value: value.map(|model| model.valid().fork(&crate::nn::device()).unfreeze().train()),
            policy_optimizer: policy_config.build()?,
            value_optimizer: value_config.map(OptimizerConfig::build).transpose()?,
            rng: ChaCha8Rng::seed_from_u64(seed),
            updates: 0,
            policy_updates: 0,
            value_updates: 0,
            revision: 0,
            poisoned: false,
            behavior_cache: RefCell::new(None),
        })
    }
    pub fn actor(&self) -> Result<P::Actor> {
        ensure(
            !self.poisoned,
            "temporal learner requires checkpoint recovery",
        )?;
        let key = (self.policy_updates, self.value_updates);
        if let Some(snapshot) = self.behavior_cache.borrow().as_ref()
            && snapshot.updates == key
        {
            return Ok(P::with_revision(&snapshot.actor, self.revision));
        }
        let actor = P::publish(self.policy.clone(), self.value.clone(), self.revision)?;
        self.behavior_cache.replace(Some(BehaviorSnapshot {
            updates: key,
            actor: actor.clone(),
        }));
        Ok(actor)
    }
    /// Pure validation, before consuming the minibatch stream or changing any optimizer.
    /// Behavior always reconstructs from factual prefixes using its own frozen encoder.
    pub fn preflight(
        &self,
        batch: &TemporalOnPolicyBatch<P::Actor>,
        kind: TemporalBatchKind,
    ) -> Result<()> {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::BehaviorPreflight);
        ensure(
            !self.poisoned,
            "temporal learner requires checkpoint recovery",
        )?;
        batch.validate_snapshot()?;
        ensure(
            batch.kind == kind
                && batch.revision == self.revision
                && self.revision.checked_add(1).is_some(),
            "stale temporal batch or exhausted revision",
        )?;
        let actor = self.actor()?;
        let (policy, value) = P::published_models(&batch.behavior);
        ensure(
            batch.behavior.behavior_identity() == actor.behavior_identity()
                && policy.architectures(value)?
                    == self.policy.architectures(self.value.as_ref())?,
            "temporal batch behavior does not match the current collection model",
        )?;
        let full = WindowBatch::reference(batch).full()?;
        let tokens = batch.tokens();
        let distribution = policy.distribution(&full, &tokens, true)?;
        let logps = to_vec(distribution.log_prob(&batch.actions(), &batch.latents())?)?;
        ensure(
            logps.len() == batch.rows.len(),
            "temporal policy likelihood row count differs",
        )?;
        for (i, (row, log_prob)) in batch.rows.iter().zip(logps).enumerate() {
            ensure(
                recording_number_matches(log_prob, row.log_prob),
                "temporal behavior likelihood reconstruction differs",
            )?;
            let rebuilt =
                distribution.snapshot(i, &full.sample(row.sample).observations[row.time])?;
            ensure(
                recording_distribution_matches(&rebuilt, &row.distribution),
                "temporal behavior distribution reconstruction differs",
            )?;
        }
        if let Some(value) = value {
            let values = to_vec(P::values(value, &full, &tokens)?)?;
            ensure(
                values.len() == batch.rows.len(),
                "temporal value row count differs",
            )?;
            for (row, value) in batch.rows.iter().zip(values) {
                ensure(
                    row.value
                        .is_some_and(|v| recording_number_matches(value, v)),
                    "temporal collected baseline reconstruction differs",
                )?;
            }
            let live: Vec<_> = batch
                .rows
                .iter()
                .enumerate()
                .filter(|(_, row)| !batch.sequences.samples[row.sample].steps[row.time].terminated)
                .collect();
            if !live.is_empty() {
                let next: Vec<_> = live
                    .iter()
                    .map(|(_, row)| (row.sample, row.time + 1))
                    .collect();
                let values = to_vec(P::values(value, &full, &next)?)?;
                ensure(
                    values.len() == live.len(),
                    "temporal successor value row count differs",
                )?;
                for ((_, row), value) in live.into_iter().zip(values) {
                    ensure(
                        row.next_value
                            .is_some_and(|v| recording_number_matches(value, v)),
                        "temporal collected successor reconstruction differs",
                    )?;
                }
            }
        }
        Ok(())
    }
    pub fn validate_models(&self) -> Result<()> {
        temporal_models::validate_parameters(&self.policy, false)?;
        if let Some(value) = &self.value {
            temporal_models::validate_parameters(value, false)?;
        }
        Ok(())
    }
    pub fn record(
        &self,
        algorithm: &str,
        kind: TemporalBatchKind,
        config: &impl Serialize,
    ) -> Result<LearnerRecord> {
        let actor = self.actor()?;
        let (policy, value) = self.policy.architectures(self.value.as_ref())?;
        let mut record = LearnerRecord::new(
            algorithm,
            TemporalOnPolicyRecord {
                version: P::RECORD_VERSION,
                config,
                policy,
                value,
                collection: P::binding(&actor, kind)?,
                rng: self.rng.clone(),
                updates: self.updates,
                policy_updates: self.policy_updates,
                value_updates: self.value_updates,
                revision: self.revision,
                optimizer_owners: BTreeMap::new(),
                optimizer_steps: BTreeMap::new(),
            },
        )?;
        record.model("policy", &self.policy)?;
        record.optimizer("policy", &self.policy_optimizer)?;
        if let (Some(value), Some(optimizer)) = (&self.value, &self.value_optimizer) {
            record.model("value", value)?;
            record.optimizer("value", optimizer)?;
        }
        let mut owners = BTreeMap::new();
        let mut clocks = BTreeMap::new();
        for (name, updates, required_paths) in [
            ("policy", self.policy_updates, P::REQUIRED_POLICY_ADAM_PATHS),
            ("value", self.value_updates, P::REQUIRED_VALUE_ADAM_PATHS),
        ]
        .into_iter()
        .take(1 + usize::from(self.value.is_some()))
        {
            let adam = audit_adam_optimizer(&record, name, name, None, updates)?;
            validate_required_adam_clocks(&record, name, required_paths, &adam.steps, updates)?;
            owners.insert(name.to_owned(), adam.parameter_ids);
            if P::SPARSE_ADAM {
                clocks.insert(name.to_owned(), adam.steps);
            }
        }
        record.metadata["optimizer_owners"] = serde_json::to_value(owners)?;
        if P::SPARSE_ADAM {
            record.metadata["optimizer_steps"] = serde_json::to_value(clocks)?;
        }
        Ok(record)
    }
    pub fn restored<C>(
        record: &LearnerRecord,
        metadata: &TemporalOnPolicyRecord<C, P::Architecture>,
        kind: TemporalBatchKind,
        policy_config: &OptimizerConfig,
        value_config: Option<&OptimizerConfig>,
    ) -> Result<Self> {
        metadata.collection.validate()?;
        ensure(
            P::accepts_record_version(metadata.version)
                && metadata.collection.kind == kind
                && metadata.collection.revision == metadata.revision
                && metadata.value.is_some() == value_config.is_some()
                && metadata.value.is_some() == record.models.contains_key("value")
                && record.models.len() == 1 + usize::from(metadata.value.is_some())
                && record.optimizers.len() == record.models.len()
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none()
                && (metadata.value.is_some() || metadata.value_updates == 0),
            "invalid recurrent learner metadata or model capabilities",
        )?;
        let names: &[&str] = if metadata.value.is_some() {
            &["policy", "value"]
        } else {
            &["policy"]
        };
        ensure(
            record
                .models
                .keys()
                .map(String::as_str)
                .eq(names.iter().copied())
                && record
                    .optimizers
                    .keys()
                    .map(String::as_str)
                    .eq(names.iter().copied())
                && if metadata.version >= 2 {
                    metadata
                        .optimizer_owners
                        .keys()
                        .map(String::as_str)
                        .eq(names.iter().copied())
                } else {
                    metadata.optimizer_owners.is_empty()
                },
            "invalid recurrent optimizer owner manifest or checkpoint owner names",
        )?;
        ensure(
            if P::SPARSE_ADAM {
                metadata
                    .optimizer_steps
                    .keys()
                    .map(String::as_str)
                    .eq(names.iter().copied())
            } else {
                metadata.optimizer_steps.is_empty()
            },
            "invalid recurrent optimizer clock manifest",
        )?;
        let expected_ids = |name: &str| {
            if metadata.version >= 2 {
                metadata.optimizer_owners.get(name).map(Vec::as_slice)
            } else {
                // Legacy records cannot establish complete historical group
                // membership; numeric, layout, and owner checks still apply.
                None
            }
        };
        let policy_adam = audit_adam_optimizer(
            record,
            "policy",
            "policy",
            expected_ids("policy"),
            metadata.policy_updates,
        )?;
        // Every dense group participates in each owner step. A conditional
        // head can be absent from some structured update graphs.
        ensure(
            if P::SPARSE_ADAM {
                metadata.optimizer_steps.get("policy") == Some(&policy_adam.steps)
            } else {
                policy_adam
                    .steps
                    .values()
                    .all(|&time| time == metadata.policy_updates)
            },
            "recurrent policy Adam history differs from owner update count",
        )?;
        validate_required_adam_clocks(
            record,
            "policy",
            P::REQUIRED_POLICY_ADAM_PATHS,
            &policy_adam.steps,
            metadata.policy_updates,
        )?;
        if metadata.value.is_some() {
            let value_adam = audit_adam_optimizer(
                record,
                "value",
                "value",
                expected_ids("value"),
                metadata.value_updates,
            )?;
            ensure(
                if P::SPARSE_ADAM {
                    metadata.optimizer_steps.get("value") == Some(&value_adam.steps)
                } else {
                    value_adam
                        .steps
                        .values()
                        .all(|&time| time == metadata.value_updates)
                },
                "recurrent value Adam history differs from owner update count",
            )?;
            validate_required_adam_clocks(
                record,
                "value",
                P::REQUIRED_VALUE_ADAM_PATHS,
                &value_adam.steps,
                metadata.value_updates,
            )?;
        }
        let (policy, value) = P::restore_models(
            record,
            &metadata.policy,
            metadata.value.as_ref(),
            &metadata.collection,
        )?;
        if let Some(value) = &value {
            crate::nn::optimization::validate_disjoint(&policy, value)?;
        }
        let policy_optimizer = record.load_optimizer_for_model(
            "policy",
            "policy",
            policy_config.build()?,
            metadata.policy_updates > 0,
        )?;
        let value_optimizer = value_config
            .map(|config| {
                record.load_optimizer_for_model(
                    "value",
                    "value",
                    config.build()?,
                    metadata.value_updates > 0,
                )
            })
            .transpose()?;
        let result = Self {
            policy,
            value,
            policy_optimizer,
            value_optimizer,
            rng: metadata.rng.clone(),
            updates: metadata.updates,
            policy_updates: metadata.policy_updates,
            value_updates: metadata.value_updates,
            revision: metadata.revision,
            poisoned: false,
            behavior_cache: RefCell::new(None),
        };
        metadata.collection.validate_actor(&result.actor()?)?;
        Ok(result)
    }
}

pub(super) fn validate_required_adam_clocks(
    record: &LearnerRecord,
    model: &str,
    paths: &[&str],
    clocks: &BTreeMap<u64, u64>,
    updates: u64,
) -> Result<()> {
    if paths.is_empty() || updates == 0 {
        return Ok(());
    }
    let model = record
        .models
        .get(model)
        .ok_or_else(|| Error::State("missing Adam owner model".into()))?;
    let reader =
        burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(model.bytes.clone()))
            .map_err(record_error)?;
    let tensors = reader.into_tensors().map_err(record_error)?;
    let active: Option<BTreeSet<_>> = paths
        .iter()
        .map(|&path| {
            tensors
                .iter()
                .find(|tensor| tensor.name == path)
                .and_then(|tensor| tensor.param_id)
        })
        .collect();
    ensure(
        active.is_some_and(|ids| {
            ids.len() == paths.len() && ids.iter().all(|id| clocks.get(id) == Some(&updates))
        }),
        "required Adam parameter clock differs from owner update count",
    )
}

pub(crate) fn checked_update_capacity(
    current: u64,
    epochs: usize,
    windows: usize,
    windows_per_batch: usize,
) -> Result<()> {
    let batches = windows.div_ceil(windows_per_batch);
    let amount = epochs
        .checked_mul(batches)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| Error::State("temporal update count overflow".into()))?;
    checked_adam_steps(current, amount)
}

/// Bound one optimizer owner's complete pending step budget before RNG/model mutation.
pub(crate) fn checked_adam_steps(current: u64, amount: u64) -> Result<()> {
    ensure(
        current
            .checked_add(amount)
            .is_some_and(|next| next < i32::MAX as u64),
        "temporal Adam step count exceeds supported bias-correction range",
    )
}

#[cfg(test)]
mod adam_capacity_tests {
    use super::checked_update_capacity;

    #[test]
    fn update_capacity_rejects_adam_bias_correction_limit_before_updates() {
        let limit = i32::MAX as u64;
        assert!(checked_update_capacity(limit - 2, 1, 1, 1).is_ok());
        assert!(
            checked_update_capacity(limit - 1, 1, 1, 1).is_err(),
            "u64 capacity alone permits an unsupported Adam step"
        );
        assert!(
            checked_update_capacity(limit - 3, 2, 3, 2).is_err(),
            "every epoch/minibatch must fit the Adam owner step budget"
        );
        assert!(checked_update_capacity(u64::MAX, 1, 1, 1).is_err());
    }
}
