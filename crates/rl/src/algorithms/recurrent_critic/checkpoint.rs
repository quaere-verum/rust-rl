//! Independent critic record; legacy DDQN envelopes are adapted by its facade.
use super::{RecurrentCriticConfig, RecurrentDiscreteCritic, parameter_values};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        reconstruction::{ReconstructionConfig, ReplayRepresentations, decode_anchor},
        stateful::HistorySpec,
        temporal::TemporalExecutionMode,
    },
    nn::{checkpoint::RecordBurnExt, temporal::GruQ},
};
use burn::module::Module;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Meta {
    version: u32,
    config: RecurrentCriticConfig,
    spec: HistorySpec,
    updates: u64,
    gru: GruSettings,
    #[serde(default, skip_serializing_if = "TemporalExecutionMode::is_reference")]
    update_mode: TemporalExecutionMode,
    #[serde(
        default,
        skip_serializing_if = "ReconstructionConfig::is_exact_default"
    )]
    reconstruction: ReconstructionConfig,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
struct GruSettings {
    reset_after: bool,
    clip: Option<f64>,
}
impl GruSettings {
    fn validate(self) -> Result<()> {
        ensure(
            self.clip.is_none_or(|x| x.is_finite() && x > 0.),
            "invalid recurrent GRU clipping setting",
        )
    }
    fn of(model: &GruQ) -> Self {
        Self {
            reset_after: model.encoder.gru.reset_after,
            clip: model.encoder.gru.clip,
        }
    }
    fn apply(self, model: &mut GruQ) {
        model.encoder.gru.reset_after = self.reset_after;
        model.encoder.gru.clip = self.clip;
    }
}
pub(super) fn validate_gru_settings(model: &GruQ) -> Result<()> {
    use burn::nn::activation::Activation;
    ensure(
        matches!(model.encoder.gru.gate_activation, Activation::Sigmoid(_))
            && matches!(model.encoder.gru.hidden_activation, Activation::Tanh(_))
            && model.encoder.gru.d_hidden == model.hidden_dim(),
        "recurrent learner requires a validated sigmoid/tanh GRU",
    )?;
    GruSettings::of(model).validate()
}
impl Checkpoint for RecurrentDiscreteCritic {
    fn validate_anchor_payloads(
        &self,
        record: &LearnerRecord,
        anchors: &[&crate::core::anchors::StateAnchor],
    ) -> Result<()> {
        if anchors.is_empty() {
            return Ok(());
        }
        record.check("recurrent-discrete-critic-v1")?;
        self.validate_anchor_metadata(record, record.metadata.clone(), anchors)
    }
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut record = LearnerRecord::new(
            "recurrent-discrete-critic-v1",
            Meta {
                version: 1,
                config: self.config.clone(),
                spec: self.spec.clone(),
                updates: self.updates,
                gru: GruSettings::of(&self.online),
                update_mode: self.update_mode,
                reconstruction: self.reconstruction,
            },
        )?;
        record.model("online", &self.online)?;
        record.model("target", &self.target)?;
        record.optimizer("critic", &self.optimizer)?;
        record.temporal_returns = Some(self.return_contract());
        if !self.reconstruction.is_exact_default() {
            record.temporal_reconstruction = Some(self.reconstruction);
        }
        Ok(record)
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        record.check("recurrent-discrete-critic-v1")?;
        self.restore_metadata(record, record.metadata.clone())
    }
}

// Envelopes owned by other algorithms borrow packed payloads. Only their small
// metadata is translated; decoding still validates every replacement before install.
impl RecurrentDiscreteCritic {
    pub(in crate::algorithms) fn validate_anchor_metadata(
        &self,
        record: &LearnerRecord,
        metadata: serde_json::Value,
        anchors: &[&crate::core::anchors::StateAnchor],
    ) -> Result<()> {
        if anchors.is_empty() {
            return Ok(());
        }
        // A private candidate validates complete parameters/optimizer controls,
        // then decodes historical envelopes without imposing current sampling-age
        // eligibility. No destination model, moments or RNG can be modified.
        let mut candidate = Self::new(self.online.clone(), self.spec.clone(), self.config.clone())?;
        candidate.restore_metadata(record, metadata)?;
        let models = candidate.replay_representations()?;
        let allow_stale = ReconstructionConfig {
            version: 1,
            mode: crate::core::reconstruction::ReconstructionMode::Anchored {
                max_generation_lag: None,
            },
        };
        for anchor in anchors {
            let (_, model) = models
                .iter()
                .find(|(id, _)| *id == anchor.representation_id)
                .ok_or_else(|| {
                    crate::Error::State("unknown recurrent checkpoint representation key".into())
                })?;
            decode_anchor(
                allow_stale,
                model,
                anchor.representation_id,
                anchor.episode_id,
                anchor.observation_index,
                anchor,
            )?;
        }
        Ok(())
    }
    pub(in crate::algorithms) fn restore_metadata(
        &mut self,
        record: &LearnerRecord,
        metadata: serde_json::Value,
    ) -> Result<()> {
        let m: Meta = serde_json::from_value(metadata)?;
        ensure(
            m.version == 1 && m.spec == self.spec,
            "incompatible recurrent critic version/history schema",
        )?;
        m.config.validate()?;
        m.reconstruction.validate()?;
        ensure(
            record.temporal_reconstruction.unwrap_or_default() == m.reconstruction,
            "recurrent checkpoint reconstruction contracts disagree",
        )?;
        if let Some(contract) = record.temporal_returns {
            contract.check(crate::core::temporal::ReturnSpec {
                gamma: m.config.gamma,
                n_step: m.config.n_step,
            })?;
        }
        m.gru.validate()?;
        let mut online = record.load_model("online", &self.online)?;
        let mut target = record.load_model("target", &self.target)?.freeze();
        // Burn records tensors; explicit metadata restores nonparameter controls.
        m.gru.apply(&mut online);
        m.gru.apply(&mut target);
        validate_gru_settings(&online)?;
        validate_gru_settings(&target)?;
        let optimizer = record.load_optimizer_for_model(
            "critic",
            "online",
            m.config.optimizer.build()?,
            m.updates > 0,
        )?;
        parameter_values(&online)?;
        parameter_values(&target)?;
        self.online = online;
        self.behavior_cache.borrow_mut().take();
        self.target_cache.borrow_mut().take();
        self.target = target;
        self.optimizer = optimizer;
        self.config = m.config;
        self.updates = m.updates;
        self.update_mode = m.update_mode;
        self.reconstruction = m.reconstruction;
        Ok(())
    }
}
