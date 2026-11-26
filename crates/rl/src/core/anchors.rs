//! Versioned, representation-neutral state boundaries for chronological replay.
use super::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const ANCHOR_VERSION: u32 = 2;
pub const MAX_ANCHOR_KEYS: usize = 16;
pub const MAX_ANCHOR_DIMENSIONS: usize = 8;
pub const MAX_ANCHOR_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RepresentationId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorPhase {
    BeforeObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorSchema {
    pub codec: u32,
    pub codec_version: u32,
    pub representation_identity: [u8; 32],
    pub dimensions: Vec<usize>,
}
impl AnchorSchema {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.codec_version > 0
                && !self.dimensions.is_empty()
                && self.dimensions.len() <= MAX_ANCHOR_DIMENSIONS,
            "invalid anchor schema version or rank",
        )?;
        let mut elements = 1usize;
        for &dimension in &self.dimensions {
            ensure(dimension > 0, "zero anchor dimension")?;
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| super::Error::Shape("anchor dimensions overflow".into()))?;
        }
        ensure(
            elements <= MAX_ANCHOR_PAYLOAD_BYTES,
            "anchor dimensions exceed resource limit",
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnchorSource {
    ResetReconstruction,
    Continued {
        parent_identity: [u8; 32],
        parent_model_identity: [u8; 32],
        parent_generation: u64,
        parent_exact: bool,
    },
    BehaviorSnapshot {
        behavior_revision: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateAnchor {
    pub version: u32,
    pub episode_id: u64,
    pub observation_index: usize,
    pub phase: AnchorPhase,
    pub representation_id: RepresentationId,
    pub model_identity: [u8; 32],
    pub generation: u64,
    pub schema: AnchorSchema,
    pub payload: Vec<u8>,
    pub exact_ancestry: bool,
    pub source: AnchorSource,
}
impl StateAnchor {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == ANCHOR_VERSION,
            "unsupported state anchor version",
        )?;
        self.schema.validate()?;
        ensure(
            !self.payload.is_empty() && self.payload.len() <= MAX_ANCHOR_PAYLOAD_BYTES,
            "anchor payload exceeds resource limit",
        )?;
        if let AnchorSource::Continued {
            parent_model_identity,
            parent_generation,
            parent_exact,
            ..
        } = &self.source
        {
            ensure(
                *parent_generation <= self.generation,
                "anchor generation precedes source",
            )?;
            ensure(
                !self.exact_ancestry
                    || (*parent_exact && *parent_model_identity == self.model_identity),
                "approximate anchor ancestry cannot become exact",
            )?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchorSet {
    pub version: u32,
    pub anchors: BTreeMap<RepresentationId, StateAnchor>,
}
impl AnchorSet {
    /// Canonical owned buffers make byte accounting independent of caller reserve
    /// history and of serde's allocation strategy. Validate limits before calling.
    pub fn canonicalize(&mut self) {
        for anchor in self.anchors.values_mut() {
            anchor.payload = std::mem::take(&mut anchor.payload)
                .into_boxed_slice()
                .into_vec();
            anchor.schema.dimensions = std::mem::take(&mut anchor.schema.dimensions)
                .into_boxed_slice()
                .into_vec();
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == ANCHOR_VERSION
                && !self.anchors.is_empty()
                && self.anchors.len() <= MAX_ANCHOR_KEYS,
            "invalid anchor set version or key count",
        )?;
        let first = self.anchors.values().next().expect("nonempty anchor set");
        for (key, anchor) in &self.anchors {
            anchor.validate()?;
            ensure(
                *key == anchor.representation_id
                    && anchor.episode_id == first.episode_id
                    && anchor.observation_index == first.observation_index,
                "anchor set keys or chronology differ",
            )?;
        }
        Ok(())
    }
    pub fn episode_id(&self) -> u64 {
        self.anchors.values().next().map_or(0, |a| a.episode_id)
    }
    pub fn observation_index(&self) -> usize {
        self.anchors
            .values()
            .next()
            .map_or(0, |a| a.observation_index)
    }
    pub fn payload_bytes(&self) -> usize {
        self.anchors.values().map(|a| a.payload.len()).sum()
    }
}
