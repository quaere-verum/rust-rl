//! Explicit MPO record boundary and opt-in import of PR32 learner records.
use super::*;
use crate::{
    core::checkpoint::{Checkpoint, LearnerRecord},
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer},
        temporal_models::TemporalModelArchitecture,
    },
};

#[derive(Serialize, Deserialize)]
struct Metadata {
    version: u32,
    config: RecurrentMpoConfig,
    policy: TemporalModelArchitecture,
    critic_metadata: serde_json::Value,
    updates: u64,
    actor_updates: u64,
    #[serde(default, skip_serializing_if = "TemporalExecutionMode::is_reference")]
    update_mode: TemporalExecutionMode,
    optimizer_owners: BTreeMap<String, Vec<u64>>,
}
impl Checkpoint for RecurrentMpo {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        ensure(
            !self.poisoned && self.actor_updates == self.critic.updates(),
            "cannot checkpoint incomplete MPO update",
        )?;
        let mut r = self.critic.checkpoint()?;
        let critic_metadata = r.metadata.take();
        r.algorithm = "recurrent-mpo-v2".into();
        r.model("actor", &self.policy)?;
        r.optimizer("actor", &self.optimizer)?;
        let mut owners = BTreeMap::new();
        for (opt, model) in [("actor", "actor"), ("critic", "online")] {
            let audit = audit_adam_optimizer(&r, opt, model, None, self.actor_updates)?;
            owners.insert(opt.into(), audit.parameter_ids);
        }
        r.metadata = serde_json::to_value(Metadata {
            version: 2,
            config: self.config.clone(),
            policy: self.policy.architecture()?,
            critic_metadata,
            updates: self.actor_updates,
            actor_updates: self.actor_updates,
            update_mode: self.update_mode,
            optimizer_owners: owners,
        })?;
        Ok(r)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-mpo-v2")?;
        self.restore_metadata(record, serde_json::from_value(record.metadata.clone())?)
    }
}

impl RecurrentMpo {
    fn restore_metadata(&mut self, record: &LearnerRecord, m: Metadata) -> Result<()> {
        m.config.validate()?;
        ensure(
            m.version == 2
                && record.temporal_returns.is_some()
                && m.updates == m.actor_updates
                && m.critic_metadata["updates"].as_u64() == Some(m.actor_updates)
                && m.policy.spec == *self.critic.spec()
                && record
                    .models
                    .keys()
                    .map(String::as_str)
                    .eq(["actor", "online", "target"])
                && record
                    .optimizers
                    .keys()
                    .map(String::as_str)
                    .eq(["actor", "critic"])
                && m.optimizer_owners.keys().eq(record.optimizers.keys()),
            "incompatible MPO checkpoint envelope",
        )?;
        temporal_models::validate_record_model(
            record,
            "actor",
            &m.policy,
            self.policy.spec().action_space.dimensions(),
        )?;
        for (opt, model) in [("actor", "actor"), ("critic", "online")] {
            let a = audit_adam_optimizer(
                record,
                opt,
                model,
                Some(&m.optimizer_owners[opt]),
                m.actor_updates,
            )?;
            ensure(
                a.steps.values().all(|&n| n == m.actor_updates),
                "MPO Adam steps differ from owner counter",
            )?;
        }
        let template =
            TemporalPolicy::from_architecture(&m.policy, &mut rand::SeedableRng::seed_from_u64(0))?;
        let policy = record.load_model("actor", &template)?;
        temporal_models::validate_parameters(&policy, false)?;
        let optimizer = record.load_optimizer_for_model(
            "actor",
            "actor",
            m.config.actor.build()?,
            m.actor_updates > 0,
        )?;
        let mut critic = RecurrentDiscreteCritic::new(
            self.critic.online().unfreeze().train(),
            self.critic.spec().clone(),
            self.critic.config().clone(),
        )?;
        critic.restore_metadata(record, m.critic_metadata)?;
        ensure(
            critic.config().n_step == 1
                && critic.reconstruction_config().is_exact_default()
                && critic.updates() == m.actor_updates
                && critic.update_mode().is_reference(),
            "invalid MPO critic contract",
        )?;
        crate::nn::optimization::validate_disjoint(&policy, &critic.online())?;
        *self = Self {
            policy,
            critic,
            config: m.config,
            optimizer,
            actor_updates: m.actor_updates,
            update_mode: m.update_mode,
            poisoned: false,
        };
        Ok(())
    }
    /// Explicit learner-only PR32 import. Ordinary restore rejects this old envelope.
    /// Full research coordinator snapshots must use their original source.
    pub fn import_pr32_record(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-mpo-v1")?;
        let mut metadata: Metadata = serde_json::from_value(record.metadata.clone())?;
        ensure(
            metadata.version == 1,
            "unsupported PR32 MPO metadata version",
        )?;
        let (_, critic_metadata) =
            super::super::recurrent_ddqn::decode_legacy_critic_metadata(&metadata.critic_metadata)?;
        metadata.critic_metadata = critic_metadata;
        metadata.version = 2;
        self.restore_metadata(record, metadata)
    }
    /// Validate the original filesystem marker and explicitly migrate its learner.
    pub fn import_pr32_checkpoint(&mut self, path: impl AsRef<std::path::Path>) -> Result<()> {
        crate::train::checkpoint::load_learner(path, &mut Pr32Import(self))
    }
}
struct Pr32Import<'a>(&'a mut RecurrentMpo);
impl Checkpoint for Pr32Import<'_> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.0.checkpoint()
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.0.import_pr32_record(record)
    }
}
