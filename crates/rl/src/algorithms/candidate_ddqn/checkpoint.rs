use super::{CandidateCritic, CandidateDdqn, CandidateDdqnConfig};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer, trainability},
        optimization::parameter_norm,
        structured::{StructuredActionValue, restore_model},
    },
};
use burn::module::Module;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

// Keep the Phase C learner's metadata and parameter paths, so its learner-only
// checkpoints can initialize the shared temporal runtime without conversion.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta<C> {
    version: u32,
    critic: CriticMeta<C>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CriticMeta<C> {
    version: u32,
    config: CandidateDdqnConfig,
    model: C,
    updates: u64,
    trainability: Vec<bool>,
    optimizer_ids: Vec<u64>,
}

impl<Q: Module> CandidateCritic<Q> {
    pub(in crate::algorithms) fn checkpoint_model<C: Serialize>(
        &self,
        configuration: &C,
    ) -> Result<LearnerRecord> {
        ensure(!self.poisoned, "cannot checkpoint poisoned candidate DDQN")?;
        let mut record = LearnerRecord::new("structured-ddqn-v1", ())?;
        record.model("online", &self.online)?;
        record.model("target", &self.target)?;
        record.optimizer("critic", &self.optimizer)?;
        let audit = audit_adam_optimizer(&record, "critic", "online", None, self.updates)?;
        record.metadata = serde_json::to_value(Meta {
            version: 1,
            critic: CriticMeta {
                version: 1,
                config: self.config.clone(),
                model: configuration,
                updates: self.updates,
                trainability: trainability(&self.online),
                optimizer_ids: audit.parameter_ids,
            },
        })?;
        record.temporal_returns = Some(self.config.returns());
        Ok(record)
    }

    pub(in crate::algorithms) fn restored_model<C: DeserializeOwned + PartialEq>(
        &self,
        record: &LearnerRecord,
        configuration: &C,
    ) -> Result<Self> {
        record.check("structured-ddqn-v1")?;
        if let Some(returns) = record.temporal_returns {
            self.config.returns().check(returns)?;
        }
        ensure(
            record.temporal_reconstruction.is_none() && record.replay_requirements.is_none(),
            "candidate DDQN checkpoint carries incompatible replay contracts",
        )?;
        let metadata: Meta<C> = serde_json::from_value(record.metadata.clone())?;
        let meta = metadata.critic;
        meta.config.validate()?;
        ensure(
            metadata.version == 1
                && meta.version == 1
                && meta.updates < i32::MAX as u64
                && meta.model == *configuration
                && meta.trainability == trainability(&self.online)
                && meta.config.gamma.to_bits() == self.config.gamma.to_bits()
                && serde_json::to_value(&meta.config)? == serde_json::to_value(&self.config)?,
            "candidate DDQN checkpoint configuration differs",
        )?;
        audit_adam_optimizer(
            record,
            "critic",
            "online",
            Some(&meta.optimizer_ids),
            meta.updates,
        )?;
        let online = restore_model(record, "online", &self.online)?;
        let target = restore_model(record, "target", &self.target)?
            .valid()
            .freeze();
        let optimizer = record.load_optimizer_for_model(
            "critic",
            "online",
            self.config.optimizer.build()?,
            meta.updates > 0,
        )?;
        parameter_norm(&online)?;
        parameter_norm(&target)?;
        Ok(Self {
            online,
            target,
            optimizer,
            config: self.config.clone(),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
            updates: meta.updates,
            poisoned: false,
        })
    }
}

impl<Q: StructuredActionValue> Checkpoint for CandidateCritic<Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.checkpoint_model(self.online.configuration())
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        *self = self.restored_model(record, self.online.configuration())?;
        Ok(())
    }
}

impl<Q: StructuredActionValue> Checkpoint for CandidateDdqn<Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.critic.checkpoint()
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.critic.restore(record)
    }
}
