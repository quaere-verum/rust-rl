use super::{CandidateMpo, CandidateMpoConfig, validate_disjoint};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::{
        checkpoint::{RecordBurnExt, audit_adam_optimizer, trainability},
        optimization::parameter_norm,
        structured::{CandidatePolicy, StructuredActionValue, restore_model},
    },
};
use burn::{module::Module, optim::ModuleOptimizer};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta<C> {
    version: u32,
    config: CandidateMpoConfig,
    policy: C,
    actor_updates: u64,
    adam_updates: u64,
    actor_trainability: Vec<bool>,
    actor_optimizer_ids: Vec<u64>,
    critic: serde_json::Value,
}

impl<Q: StructuredActionValue, P: CandidatePolicy> Checkpoint for CandidateMpo<Q, P> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        checkpoint_actor(
            self.critic.checkpoint()?,
            &self.policy,
            self.policy.configuration(),
            &self.optimizer,
            &self.config,
            self.actor_updates,
            self.adam_updates,
        )
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("candidate-mpo-v1")?;
        self.critic
            .config
            .returns()
            .check(record.temporal_returns.ok_or_else(|| {
                crate::Error::State(
                    "candidate MPO checkpoint is missing its return contract".into(),
                )
            })?)?;
        let actor = restore_actor(
            record,
            &self.policy,
            self.policy.configuration(),
            &self.config,
        )?;
        let mut critic_record = record.clone();
        critic_record.algorithm = "structured-ddqn-v1".into();
        critic_record.metadata = actor.critic;
        let critic = self
            .critic
            .restored_model(&critic_record, self.critic.online.configuration())?;
        ensure(
            actor.updates <= critic.updates,
            "MPO actor clock exceeds critic clock",
        )?;
        validate_disjoint(&actor.policy, &critic.online)?;

        self.policy = actor.policy;
        self.optimizer = actor.optimizer;
        self.critic = critic;
        self.actor_updates = actor.updates;
        self.adam_updates = actor.adam_updates;
        Ok(())
    }
}

pub(in crate::algorithms) fn checkpoint_actor<P: Module, C: Serialize>(
    mut record: LearnerRecord,
    policy: &P,
    configuration: &C,
    optimizer: &ModuleOptimizer,
    config: &CandidateMpoConfig,
    actor_updates: u64,
    adam_updates: u64,
) -> Result<LearnerRecord> {
    record.model("policy", policy)?;
    record.optimizer("actor", optimizer)?;
    let groups = audit_adam_optimizer(&record, "actor", "policy", None, adam_updates)?;
    record.algorithm = "candidate-mpo-v1".into();
    record.metadata = serde_json::to_value(Meta {
        version: 1,
        config: config.clone(),
        policy: configuration,
        actor_updates,
        adam_updates,
        actor_trainability: trainability(policy),
        actor_optimizer_ids: groups.parameter_ids,
        critic: record.metadata,
    })?;
    Ok(record)
}

pub(in crate::algorithms) struct RestoredActor<P> {
    pub policy: P,
    pub optimizer: ModuleOptimizer,
    pub updates: u64,
    pub adam_updates: u64,
    pub critic: serde_json::Value,
}

pub(in crate::algorithms) fn restore_actor<P: Module, C: DeserializeOwned + PartialEq>(
    record: &LearnerRecord,
    policy: &P,
    configuration: &C,
    config: &CandidateMpoConfig,
) -> Result<RestoredActor<P>> {
    record.check("candidate-mpo-v1")?;
    let meta: Meta<C> = serde_json::from_value(record.metadata.clone())?;
    meta.config.validate()?;
    ensure(
        meta.version == 1
            && meta.policy == *configuration
            && meta.actor_trainability == trainability(policy)
            && meta.actor_updates < i32::MAX as u64
            && meta.adam_updates <= meta.actor_updates
            && (meta.adam_updates == 0) == (meta.actor_updates == 0)
            && serde_json::to_value(&meta.config)? == serde_json::to_value(config)?,
        "candidate MPO checkpoint configuration or actor clock differs",
    )?;
    audit_adam_optimizer(
        record,
        "actor",
        "policy",
        Some(&meta.actor_optimizer_ids),
        meta.adam_updates,
    )?;
    let policy = restore_model(record, "policy", policy)?;
    let optimizer = record.load_optimizer_for_model(
        "actor",
        "policy",
        config.optimizer.build()?,
        meta.adam_updates > 0,
    )?;
    parameter_norm(&policy)?;
    Ok(RestoredActor {
        policy,
        optimizer,
        updates: meta.actor_updates,
        adam_updates: meta.adam_updates,
        critic: meta.critic,
    })
}
