use super::*;
use crate::{
    core::checkpoint::{Checkpoint, LearnerRecord},
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer, trainability},
        structured::{LightweightConfig, restore_model},
    },
};
use rand_chacha::ChaCha8Rng;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    version: u32,
    config: SearchGuidedConfig,
    #[serde(default, skip_serializing_if = "ReturnObjective::is_legacy_physical")]
    objective: ReturnObjective,
    policy: LightweightConfig,
    value: LightweightConfig,
    updates: u64,
    publication_revision: u64,
    // Earlier learner records included the retired coordinator shuffle stream.
    #[serde(default, rename = "rng", skip_serializing)]
    _legacy_rng: Option<ChaCha8Rng>,
    policy_trainability: Vec<bool>,
    value_trainability: Vec<bool>,
    policy_optimizer_ids: Vec<u64>,
    value_optimizer_ids: Vec<u64>,
}

fn audit(
    record: &LearnerRecord,
    owner: &str,
    ids: Option<&[u64]>,
    updates: u64,
) -> Result<Vec<u64>> {
    let state = audit_adam_optimizer(record, owner, owner, ids, updates)?;
    ensure(
        state.steps.values().copied().max().unwrap_or(0) == updates,
        "paired optimizer clock differs from learner updates",
    )?;
    Ok(state.parameter_ids)
}

impl Checkpoint for SearchGuidedPolicyValue {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new("search-guided-policy-value-v1", ())?;
        record.model("policy", &self.policy)?;
        record.model("value", &self.value)?;
        record.optimizer("policy", &self.policy_optimizer)?;
        record.optimizer("value", &self.value_optimizer)?;
        record.metadata = serde_json::to_value(Meta {
            version: 1,
            config: self.config.clone(),
            objective: self.objective,
            policy: self.policy.config().clone(),
            value: self.value.config().clone(),
            updates: self.updates,
            publication_revision: self.publication_revision,
            _legacy_rng: None,
            policy_trainability: trainability(&self.policy),
            value_trainability: trainability(&self.value),
            policy_optimizer_ids: audit(&record, "policy", None, self.updates)?,
            value_optimizer_ids: audit(&record, "value", None, self.updates)?,
        })?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("search-guided-policy-value-v1")?;
        let meta: Meta = serde_json::from_value(record.metadata.clone())?;
        meta.config.validate()?;
        ensure(
            meta.version == 1
                && meta.policy == *self.policy.config()
                && meta.value == *self.value.config()
                && serde_json::to_value(&meta.config)? == serde_json::to_value(&self.config)?
                && meta.objective == self.objective
                && meta.policy_trainability == trainability(&self.policy)
                && meta.value_trainability == trainability(&self.value)
                && record.models.len() == 2
                && record.optimizers.len() == 2
                && record.temporal_returns.is_none()
                && record.temporal_spaces.is_none()
                && record.temporal_reconstruction.is_none()
                && record.replay_requirements.is_none(),
            "search-guided checkpoint configuration differs",
        )?;
        audit(
            record,
            "policy",
            Some(&meta.policy_optimizer_ids),
            meta.updates,
        )?;
        audit(
            record,
            "value",
            Some(&meta.value_optimizer_ids),
            meta.updates,
        )?;
        let policy = restore_model(record, "policy", &self.policy)?;
        let value = restore_model(record, "value", &self.value)?;
        validate_disjoint(&policy, &value)?;
        parameter_norm(&policy)?;
        parameter_norm(&value)?;
        let policy_optimizer = record.load_optimizer_for_model(
            "policy",
            "policy",
            self.config.policy_optimizer.build()?,
            meta.updates > 0,
        )?;
        let value_optimizer = record.load_optimizer_for_model(
            "value",
            "value",
            self.config.value_optimizer.build()?,
            meta.updates > 0,
        )?;
        self.policy = policy;
        self.value = value;
        self.policy_optimizer = policy_optimizer;
        self.value_optimizer = value_optimizer;
        self.updates = meta.updates;
        self.publication_revision = meta.publication_revision;
        Ok(())
    }
}
