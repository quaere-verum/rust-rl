//! Typed persistence of the frozen encoder and its execution settings.

use super::{FrozenRecurrentFeatures, Incremental, RecurrentLightweightQ};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
    },
    nn::structured::PackLimits,
};
use burn::tensor::Device;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::num::NonZeroUsize;

const VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenRecord {
    version: u32,
    encoder: LearnerRecord,
    representation: [u8; 32],
    max_elements: usize,
    incremental_capacity: Option<NonZeroUsize>,
}

impl Serialize for FrozenRecurrentFeatures {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        ensure(
            self.model.model_device() == Device::flex(),
            "frozen encoder serialization requires the Flex device",
        )
        .map_err(serde::ser::Error::custom)?;
        FrozenRecord {
            version: VERSION,
            encoder: self.model.checkpoint().map_err(serde::ser::Error::custom)?,
            representation: self
                .spec
                .representation
                .expect("frozen encoder fingerprint"),
            max_elements: self.limits.max_elements,
            incremental_capacity: self.incremental.as_ref().map(Incremental::capacity),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FrozenRecurrentFeatures {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = FrozenRecord::deserialize(deserializer)?;
        Self::restore_frozen(wire).map_err(serde::de::Error::custom)
    }
}

impl FrozenRecurrentFeatures {
    fn restore_frozen(wire: FrozenRecord) -> Result<Self> {
        ensure(
            wire.version == VERSION && wire.max_elements > 0,
            "invalid frozen encoder record",
        )?;
        let config = serde_json::from_value(wire.encoder.metadata.clone())?;
        let mut model =
            RecurrentLightweightQ::new(config, &mut ChaCha8Rng::seed_from_u64(0), &Device::flex())?;
        model.restore(&wire.encoder)?;
        let features = Self::new(
            model,
            PackLimits {
                max_elements: wire.max_elements,
            },
        )?;
        ensure(
            features.spec.representation == Some(wire.representation),
            "frozen encoder fingerprint mismatch",
        )?;
        Ok(match wire.incremental_capacity {
            Some(capacity) => features.with_incremental(capacity),
            None => features,
        })
    }
}
