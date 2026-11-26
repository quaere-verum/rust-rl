//! Versioned, staged restoration of independent models, Adam owners and RNG streams.
use super::*;
use crate::{
    core::checkpoint::{Checkpoint, LearnerRecord},
    nn::checkpoint::{RecordBurnExt, audit_adam_optimizer, trainability},
};
use std::collections::BTreeMap;

#[derive(Serialize, Deserialize)]
struct Metadata {
    version: u32,
    config: SacConfig,
    space: ActionSpace,
    actor_rng: ChaCha8Rng,
    target_rng: ChaCha8Rng,
    updates: u64,
    optimizer_owners: BTreeMap<String, Vec<u64>>,
    trainability: BTreeMap<String, Vec<bool>>,
}
impl<P: StochasticPolicy, Q: ActionValue> Checkpoint for Sac<P, Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        ensure(!self.poisoned, "cannot checkpoint an incomplete SAC update")?;
        let mut record = LearnerRecord::new("sac-v2", ())?;
        record.replay_requirements = Some(self.config.requirements());
        record.model("actor", &self.policy)?;
        record.optimizer("actor", &self.actor_optimizer)?;
        self.first.record(&mut record, "first", "target_first")?;
        self.second.record(&mut record, "second", "target_second")?;
        if let Some(state) = &self.temperature {
            record.model("temperature", &state.model)?;
            record.optimizer("temperature", &state.optimizer)?;
        }
        let mut owners = BTreeMap::new();
        for name in record.optimizers.keys() {
            owners.insert(
                name.clone(),
                audit_adam_optimizer(&record, name, name, None, self.updates)?.parameter_ids,
            );
        }
        record.metadata = serde_json::to_value(Metadata {
            version: 2,
            config: self.config.clone(),
            space: self.policy.action_space().clone(),
            actor_rng: self.actor_rng.clone(),
            target_rng: self.target_rng.clone(),
            updates: self.updates,
            optimizer_owners: owners,
            trainability: self.trainability(),
        })?;
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("sac-v2")?;
        let m: Metadata = serde_json::from_value(record.metadata.clone())?;
        m.config.validate()?;
        let learned = matches!(m.config.temperature, Temperature::Learned { .. });
        let models = [
            "actor",
            "first",
            "second",
            "target_first",
            "target_second",
            "temperature",
        ];
        let optimizers = ["actor", "first", "second", "temperature"];
        ensure(
            m.version == 2
                && m.trainability == self.trainability()
                && m.actor_rng.get_stream() == ACTOR_STREAM
                && m.target_rng.get_stream() == TARGET_STREAM
                && &m.space == self.policy.action_space()
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none()
                && record.replay_requirements == Some(m.config.requirements())
                && record
                    .models
                    .keys()
                    .map(String::as_str)
                    .eq(models[..if learned { 6 } else { 5 }].iter().copied())
                && record
                    .optimizers
                    .keys()
                    .map(String::as_str)
                    .eq(optimizers[..if learned { 4 } else { 3 }].iter().copied())
                && m.optimizer_owners.keys().eq(record.optimizers.keys()),
            "incompatible SAC checkpoint envelope",
        )?;
        for name in record.optimizers.keys() {
            let audit = audit_adam_optimizer(
                record,
                name,
                name,
                Some(&m.optimizer_owners[name]),
                m.updates,
            )?;
            // Conditional custom networks may legitimately omit a parameter on
            // some batches. The audit preserves its own exact Adam step count.
            if name == "temperature" {
                ensure(
                    audit.steps.values().all(|&n| n == m.updates),
                    "SAC temperature step count differs",
                )?;
            }
        }
        let policy = record.load_model("actor", &self.policy)?;
        let first = self
            .first
            .restored(record, "first", "target_first", &m.config.critic)?;
        let second = self
            .second
            .restored(record, "second", "target_second", &m.config.critic)?;
        let mut candidate = Self::new(
            policy,
            first.online.clone(),
            second.online.clone(),
            m.config,
            0,
        )?;
        for q in [&first.online, &first.target, &second.online, &second.target] {
            parameter_norm(q)?;
        }
        candidate.first = first;
        candidate.first.target = candidate.first.target.freeze();
        candidate.second = second;
        candidate.second.target = candidate.second.target.freeze();
        candidate.actor_optimizer = record.load_optimizer_for_model(
            "actor",
            "actor",
            candidate.config.actor.build()?,
            m.updates > 0,
        )?;
        if let Some(state) = &mut candidate.temperature {
            state.model = record.load_model("temperature", &state.model)?;
            state.model.validate()?;
            state.optimizer = record.load_optimizer_for_model(
                "temperature",
                "temperature",
                state.optimizer.clone(),
                m.updates > 0,
            )?;
        }
        candidate.actor_rng = m.actor_rng;
        candidate.target_rng = m.target_rng;
        candidate.updates = m.updates;
        *self = candidate;
        Ok(())
    }
}

impl<P: StochasticPolicy, Q: ActionValue> Sac<P, Q> {
    fn trainability(&self) -> BTreeMap<String, Vec<bool>> {
        [
            ("actor".into(), trainability(&self.policy)),
            ("first".into(), trainability(&self.first.online)),
            ("second".into(), trainability(&self.second.online)),
        ]
        .into_iter()
        .collect()
    }
}
