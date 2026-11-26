//! Backend-independent persistence records and restoration contracts.
use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRecord {
    pub(crate) signature: String,
    pub(crate) bytes: Vec<u8>,
    pub(crate) parameter_shapes: BTreeMap<String, Vec<usize>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerRecord {
    pub algorithm: String,
    pub metadata: serde_json::Value,
    pub models: BTreeMap<String, ModelRecord>,
    pub optimizers: BTreeMap<String, Vec<u8>>,
    /// Optional cross-component contract. Omission preserves feedforward records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_returns: Option<super::temporal::ReturnSpec>,
    /// Semantic binding for learners that carry explicit interaction schemas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_spaces: Option<super::temporal::TemporalSpaces>,
    /// Explicit nondefault recurrent reconstruction semantics; ordinary records omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_reconstruction: Option<super::reconstruction::ReconstructionConfig>,
    /// Opt-in replay contract; legacy records remain byte-schema compatible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay_requirements: Option<super::learner::OffPolicyRequirements>,
}
impl LearnerRecord {
    pub fn new(algorithm: &str, metadata: impl Serialize) -> Result<Self> {
        Ok(Self {
            algorithm: algorithm.into(),
            metadata: serde_json::to_value(metadata)?,
            models: BTreeMap::new(),
            optimizers: BTreeMap::new(),
            temporal_returns: None,
            temporal_spaces: None,
            temporal_reconstruction: None,
            replay_requirements: None,
        })
    }
    pub fn check(&self, algorithm: &str) -> Result<()> {
        ensure(self.algorithm == algorithm, "checkpoint algorithm mismatch")
    }
}
pub trait Checkpoint {
    fn checkpoint(&self) -> Result<LearnerRecord>;
    /// Validate a complete replacement before installing it. Errors leave the destination intact.
    fn restore(&mut self, record: &LearnerRecord) -> Result<()>;
    /// Validate opaque representation payloads with the CANDIDATE model/schema,
    /// before installation. Consumers without codecs reject nonempty anchors.
    fn validate_anchor_payloads(
        &self,
        _record: &LearnerRecord,
        anchors: &[&super::anchors::StateAnchor],
    ) -> Result<()> {
        ensure(
            anchors.is_empty(),
            "learner does not support checkpoint anchor codecs",
        )
    }
}

/// Audit of a single concrete memory representation during an explicit import.
/// Bounds belong to the actor's codec; this is not a global float tolerance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortableMemoryAudit {
    pub role: String,
    pub rule: String,
    pub values: usize,
    pub adjusted_values: usize,
    pub max_absolute_difference: f64,
    pub max_ulp_difference: u32,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortableLaneAudit {
    pub lane: usize,
    pub memories: Vec<PortableMemoryAudit>,
}
/// Native memory reconstruction is a migration, not bitwise foreign-run continuation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortableRestoreReport {
    pub version: u32,
    pub lanes: Vec<PortableLaneAudit>,
}
impl Default for PortableRestoreReport {
    fn default() -> Self {
        Self {
            version: 1,
            lanes: Vec::new(),
        }
    }
}
/// Opt-in complete state persistence, including owned random streams.
pub trait Snapshot: Serialize + DeserializeOwned {
    /// Persisted format identity. Override only when a compatible snapshot's Rust type name changes.
    fn record_type() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn validate_snapshot(&self) -> Result<()>;
    /// Explicit foreign-runtime import preparation on a private candidate.
    /// The default retains exact validation. Implementations may audit and
    /// reconstruct narrowly specified runtime memories; model/optimizer state,
    /// chronology, provenance and random streams must remain unchanged.
    fn prepare_portable_restore(&mut self) -> Result<PortableRestoreReport> {
        self.validate_snapshot()?;
        Ok(PortableRestoreReport::default())
    }
    /// Validate associated learning semantics before filesystem creation or
    /// learner installation. Ordinary snapshots retain the no-op contract.
    fn validate_learner_record(&self, _record: &LearnerRecord) -> Result<()> {
        Ok(())
    }
    fn reconstruction_anchors(&self) -> Vec<&super::anchors::StateAnchor> {
        Vec::new()
    }
}
