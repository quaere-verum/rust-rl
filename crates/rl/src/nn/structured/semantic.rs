//! Ordered semantic input identity. This is configuration, never a neural input.
use crate::{
    Result,
    core::{
        ensure,
        structured::{FeatureIdentity, VocabularyIdentity, feature_fields, named},
    },
};
use serde::{Deserialize, Serialize};

/// Canonical descriptor: ordered vectors and strings, with no unordered maps.
/// Exact equality, including conventions, binds models and packed inputs.
/// Adapters must change this descriptor whenever feature meanings change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticInput {
    pub version: u32,
    pub namespace: String,
    pub observation_version: u32,
    pub action_version: u32,
    pub spatial: Vec<FeatureIdentity>,
    pub entities: Vec<FeatureIdentity>,
    pub events: Vec<FeatureIdentity>,
    pub scalars: Vec<FeatureIdentity>,
    pub candidates: Vec<FeatureIdentity>,
    /// Same order as InputSchema::operations, including reserved operations.
    pub operations: Vec<String>,
    pub vocabularies: Vec<VocabularyIdentity>,
    /// Cross-column conventions (unknowns, event order/joins, coordinate frame).
    pub conventions: Vec<String>,
}
impl SemanticInput {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1
                && !self.namespace.trim().is_empty()
                && self.namespace.len() <= 4096
                && self.observation_version > 0
                && self.action_version > 0
                && !self.operations.is_empty()
                && self.operations.len() <= 1024
                && named(self.operations.iter().map(String::as_str))
                && self.vocabularies.len() <= 1024
                && named(self.vocabularies.iter().map(|v| v.name.as_str()))
                && self.conventions.len() <= 1024
                && !self.conventions.is_empty()
                && named(self.conventions.iter().map(String::as_str)),
            "invalid structured semantic identity",
        )?;
        ensure(
            [
                &self.spatial,
                &self.entities,
                &self.events,
                &self.scalars,
                &self.candidates,
            ]
            .into_iter()
            .all(|fields| feature_fields(fields)),
            "invalid structured feature identity",
        )?;
        for vocabulary in &self.vocabularies {
            ensure(
                !vocabulary.categories.is_empty()
                    && vocabulary.categories.len() <= 65536
                    && named(vocabulary.categories.iter().map(String::as_str)),
                "invalid structured vocabulary identity",
            )?;
        }
        Ok(())
    }
}
