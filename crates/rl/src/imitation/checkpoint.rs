use super::*;
use crate::{
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, LearnerRecord},
    },
    nn::checkpoint::{RecordBurnExt, audit_adam_optimizer, trainability},
};
#[derive(Serialize, Deserialize)]
struct Metadata {
    version: u32,
    config: CloningConfig,
    space: ActionSpace,
    rng: ChaCha8Rng,
    updates: u64,
    epochs: u64,
    optimizer_owners: Vec<u64>,
    trainability: Vec<bool>,
}
impl<P: StochasticPolicy> Checkpoint for BehaviorCloning<P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        ensure(!self.poisoned, "cannot checkpoint incomplete cloning epoch")?;
        let mut record = LearnerRecord::new("behavior-cloning-v1", ())?;
        record.model("actor", &self.policy)?;
        record.optimizer("actor", &self.optimizer)?;
        let owners =
            audit_adam_optimizer(&record, "actor", "actor", None, self.updates)?.parameter_ids;
        record.metadata = serde_json::to_value(Metadata {
            version: 1,
            config: self.config.clone(),
            space: self.policy.action_space().clone(),
            rng: self.rng.clone(),
            updates: self.updates,
            epochs: self.epochs,
            optimizer_owners: owners,
            trainability: trainability(&self.policy),
        })?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("behavior-cloning-v1")?;
        let m: Metadata = serde_json::from_value(record.metadata.clone())?;
        m.config.validate()?;
        ensure(
            m.version == 1
                && m.trainability == trainability(&self.policy)
                && &m.space == self.policy.action_space()
                && m.epochs <= m.updates
                && (m.epochs == 0) == (m.updates == 0)
                && record.models.keys().map(String::as_str).eq(["actor"])
                && record.optimizers.keys().map(String::as_str).eq(["actor"])
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none()
                && record.replay_requirements.is_none(),
            "incompatible cloning checkpoint",
        )?;
        audit_adam_optimizer(
            record,
            "actor",
            "actor",
            Some(&m.optimizer_owners),
            m.updates,
        )?;
        let policy = record.load_model("actor", &self.policy)?;
        let mut candidate = Self::new(policy, m.config, 0)?;
        candidate.optimizer = record.load_optimizer_for_model(
            "actor",
            "actor",
            candidate.config.optimizer.build()?,
            m.updates > 0,
        )?;
        candidate.rng = m.rng;
        candidate.updates = m.updates;
        candidate.epochs = m.epochs;
        *self = candidate;
        Ok(())
    }
}
