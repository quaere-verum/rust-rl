//! Learner wire metadata and model/optimizer ownership audits.

use super::{RecurrentReppo, RecurrentReppoPolicy, validate_models};
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Error, Result,
    algorithms::{
        recurrent_on_policy::validate_required_adam_clocks,
        reppo::{LogMultiplier, ReppoConfig},
    },
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        temporal::TemporalExecutionMode,
    },
    data::temporal_on_policy::TemporalBatchKind,
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer, record_error},
        temporal_models,
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata<P, Q> {
    version: u32,
    config: ReppoConfig,
    #[serde(default, skip_serializing_if = "TemporalExecutionMode::is_reference")]
    update_mode: TemporalExecutionMode,
    policy: P,
    critic: Q,
    collection: TemporalLearnerBinding,
    rng: ChaCha8Rng,
    updates: u64,
    auxiliary_updates: u64,
    revision: u64,
    collection_begins: u64,
    collection_open: bool,
    completed_batches: u64,
    consumed_revision: Option<u64>,
    optimizer_owners: BTreeMap<String, Vec<u64>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    optimizer_steps: BTreeMap<String, BTreeMap<u64, u64>>,
}

impl<P: RecurrentReppoPolicy> Checkpoint for RecurrentReppo<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let actor = self.actor()?;
        validate_models(&self.policy, &self.critic, &self.config)?;
        self.alpha.validate()?;
        self.beta.validate()?;
        let (policy, critic) = self.policy.architectures(&self.critic)?;
        let metadata = Metadata {
            version: P::RECORD_VERSION,
            config: self.config.clone(),
            update_mode: self.update_mode,
            policy,
            critic,
            collection: P::binding(&actor)?,
            rng: self.rng.clone(),
            updates: self.updates,
            auxiliary_updates: self.auxiliary_updates,
            revision: self.revision(),
            collection_begins: self.collection_begins.get(),
            collection_open: self.collection_open.get(),
            completed_batches: self.completed_batches,
            consumed_revision: self.consumed_revision,
            optimizer_owners: BTreeMap::new(),
            optimizer_steps: BTreeMap::new(),
        };
        validate_metadata::<P>(&metadata)?;
        let mut record = LearnerRecord::new("recurrent_reppo", metadata)?;
        record.model("policy", &self.policy)?;
        record.model("critic", &self.critic)?;
        record.model("alpha", &self.alpha)?;
        record.model("beta", &self.beta)?;
        record.optimizer("policy", &self.policy_optimizer)?;
        record.optimizer("critic", &self.critic_optimizer)?;
        record.optimizer("entropy_dual", &self.entropy_optimizer)?;
        record.optimizer("kl_dual", &self.kl_optimizer)?;
        let (owners, steps) =
            audit_optimizers::<P>(&record, None, self.updates, self.auxiliary_updates)?;
        record.metadata["optimizer_owners"] = serde_json::to_value(owners)?;
        if P::SPARSE_ADAM {
            record.metadata["optimizer_steps"] = serde_json::to_value(steps)?;
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent_reppo")?;
        let metadata: Metadata<P::PolicyArchitecture, P::CriticArchitecture> =
            serde_json::from_value(record.metadata.clone())?;
        validate_metadata::<P>(&metadata)?;
        ensure(
            record
                .models
                .keys()
                .map(String::as_str)
                .eq(["alpha", "beta", "critic", "policy"])
                && record.optimizers.keys().map(String::as_str).eq([
                    "critic",
                    "entropy_dual",
                    "kl_dual",
                    "policy",
                ])
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none(),
            "invalid recurrent REPPO record ownership",
        )?;
        P::validate_record_models(record, &metadata.policy, &metadata.critic)?;
        for name in ["alpha", "beta"] {
            ensure(
                record.models[name].parameter_shapes.len() == 1
                    && record.models[name]
                        .parameter_shapes
                        .values()
                        .all(|shape| shape.as_slice() == [1]),
                "invalid recurrent REPPO multiplier shape",
            )?;
            temporal_models::validate_record_model_count(record, name, 1)?;
        }
        // Check untrusted optimizer bytes and their complete trained-group
        // contracts before allocating or installing any candidate model.
        let (_, steps) = audit_optimizers::<P>(
            record,
            Some(&metadata.optimizer_owners),
            metadata.updates,
            metadata.auxiliary_updates,
        )?;
        ensure(
            if P::SPARSE_ADAM {
                metadata.optimizer_steps == steps
            } else {
                record.metadata.get("optimizer_steps").is_none()
            },
            "recurrent REPPO optimizer step manifest differs",
        )?;
        let (policy, critic) = P::restore_models(record, &metadata.policy, &metadata.critic)?;
        validate_models(&policy, &critic, &metadata.config)?;
        let alpha =
            record.load_model("alpha", &LogMultiplier::new(metadata.config.initial_alpha))?;
        let beta = record.load_model("beta", &LogMultiplier::new(metadata.config.initial_beta))?;
        alpha.validate()?;
        beta.validate()?;
        crate::nn::optimization::validate_disjoint(&policy, &alpha)?;
        crate::nn::optimization::validate_disjoint(&policy, &beta)?;
        crate::nn::optimization::validate_disjoint(&critic, &alpha)?;
        crate::nn::optimization::validate_disjoint(&critic, &beta)?;
        crate::nn::optimization::validate_disjoint(&alpha, &beta)?;
        let policy_optimizer = record.load_optimizer_for_model(
            "policy",
            "policy",
            metadata.config.policy.build()?,
            metadata.updates > 0,
        )?;
        let critic_optimizer = record.load_optimizer_for_model(
            "critic",
            "critic",
            metadata.config.critic.build()?,
            metadata.updates > 0,
        )?;
        let entropy_optimizer = record.load_optimizer_for_model(
            "entropy_dual",
            "alpha",
            metadata.config.entropy_dual.build()?,
            metadata.updates > 0,
        )?;
        let kl_optimizer = record.load_optimizer_for_model(
            "kl_dual",
            "beta",
            metadata.config.kl_dual.build()?,
            metadata.updates > 0,
        )?;
        let result = Self {
            policy,
            critic,
            config: metadata.config,
            update_mode: metadata.update_mode,
            policy_optimizer,
            critic_optimizer,
            alpha,
            beta,
            entropy_optimizer,
            kl_optimizer,
            rng: metadata.rng,
            updates: metadata.updates,
            auxiliary_updates: metadata.auxiliary_updates,
            revision: Cell::new(metadata.revision),
            collection_begins: Cell::new(metadata.collection_begins),
            collection_open: Cell::new(metadata.collection_open),
            completed_batches: metadata.completed_batches,
            consumed_revision: metadata.consumed_revision,
            poisoned: false,
            behavior_cache: RefCell::new(None),
        };
        metadata.collection.validate_actor(&result.actor()?)?;
        *self = result;
        Ok(())
    }
}
fn validate_metadata<P: RecurrentReppoPolicy>(
    m: &Metadata<P::PolicyArchitecture, P::CriticArchitecture>,
) -> Result<()> {
    m.config.validate()?;
    m.collection.validate()?;
    P::validate_binding(&m.policy, &m.critic, &m.collection, &m.config)?;
    ensure(
        m.version == P::RECORD_VERSION
            && m.collection.kind == TemporalBatchKind::Pathwise
            && m.collection.revision == m.revision
            && m.collection_begins.checked_add(m.completed_batches) == Some(m.revision)
            && m.completed_batches <= m.collection_begins
            && m.completed_batches <= m.updates
            && m.auxiliary_updates <= m.updates
            && (m.config.auxiliary_coefficient > 0. || m.auxiliary_updates == 0)
            && (m.updates == 0) == (m.completed_batches == 0)
            && m.consumed_revision.is_some() == (m.completed_batches > 0)
            && m.consumed_revision.is_none_or(|r| {
                r > 0
                    && r.checked_add(1).is_some_and(|n| {
                        if m.collection_open {
                            n < m.revision
                        } else {
                            n == m.revision
                        }
                    })
            })
            && (m.collection_open || m.consumed_revision.is_some() || m.revision == 0)
            && (!m.collection_open || m.revision > 0)
            && m.revision.checked_add(1).is_some(),
        "invalid recurrent REPPO metadata or counters",
    )
}

type OptimizerOwners = BTreeMap<String, Vec<u64>>;
type OptimizerSteps = BTreeMap<String, BTreeMap<u64, u64>>;

fn audit_optimizers<P: RecurrentReppoPolicy>(
    record: &LearnerRecord,
    expected: Option<&OptimizerOwners>,
    updates: u64,
    auxiliary_updates: u64,
) -> Result<(OptimizerOwners, OptimizerSteps)> {
    if let Some(expected) = expected {
        ensure(
            expected
                .keys()
                .map(String::as_str)
                .eq(["critic", "entropy_dual", "kl_dual", "policy"]),
            "invalid recurrent REPPO optimizer owner contracts",
        )?;
    }
    let critic = record
        .models
        .get("critic")
        .ok_or_else(|| Error::State("missing REPPO critic record".into()))?;
    let reader =
        burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(critic.bytes.clone()))
            .map_err(record_error)?;
    let predictor_ids: std::collections::BTreeSet<_> = reader
        .into_tensors()
        .map_err(record_error)?
        .into_iter()
        .filter(|tensor| tensor.name.starts_with("predictor."))
        .filter_map(|tensor| tensor.param_id)
        .collect();
    let mut owners = BTreeMap::new();
    let mut steps = BTreeMap::new();
    for (optimizer, model) in [
        ("policy", "policy"),
        ("critic", "critic"),
        ("entropy_dual", "alpha"),
        ("kl_dual", "beta"),
    ] {
        let summary = audit_adam_optimizer(
            record,
            optimizer,
            model,
            expected.map(|map| map[optimizer].as_slice()),
            updates,
        )?;
        let required_paths = match optimizer {
            "policy" => P::REQUIRED_POLICY_ADAM_PATHS,
            "critic" => P::REQUIRED_CRITIC_ADAM_PATHS,
            _ => &[],
        };
        validate_required_adam_clocks(record, model, required_paths, &summary.steps, updates)?;
        if P::SPARSE_ADAM && optimizer == "critic" && auxiliary_updates > 0 {
            ensure(
                !predictor_ids.is_empty()
                    && predictor_ids
                        .iter()
                        .all(|id| summary.steps.contains_key(id)),
                "structured REPPO predictor Adam membership is incomplete",
            )?;
        }
        // Every policy/base-Q/dual group steps once per minibatch. The optional
        // predictor steps only when this outer batch has continuing aux targets.
        for (&id, &time) in &summary.steps {
            let predictor = optimizer == "critic" && predictor_ids.contains(&id);
            let count = if predictor {
                auxiliary_updates
            } else {
                updates
            };
            ensure(
                time == count
                    || (P::SPARSE_ADAM && !predictor && matches!(optimizer, "critic" | "policy")),
                "recurrent REPPO Adam history differs from owner update count",
            )?;
        }
        owners.insert(optimizer.to_owned(), summary.parameter_ids);
        if P::SPARSE_ADAM {
            steps.insert(optimizer.to_owned(), summary.steps);
        }
    }
    Ok((owners, steps))
}
