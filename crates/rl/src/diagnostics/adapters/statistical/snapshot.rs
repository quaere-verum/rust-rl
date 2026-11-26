use crate::{
    Error, Result,
    core::ensure,
    diagnostics::{
        models::ModelLimits,
        schema::{Array, Chunk, EventRecord, ModelArtifact, ModelSnapshotRecord, Record},
        session::CaptureRequest,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const FORMAT: &str = "serde-json/1";
const PAYLOAD: &str = "statistical_artifact";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Reconstruction {
    Portable,
    NeedsFeatures,
}

#[derive(Serialize, Deserialize)]
struct Configuration {
    format: String,
    reconstruction: Reconstruction,
}

pub(super) fn capture(
    request: &CaptureRequest<'_>,
    architecture: &str,
    model: &str,
    reconstruction: Reconstruction,
    encode: impl FnOnce(usize) -> Result<Vec<u8>>,
) -> Result<Chunk> {
    let mut context = request.context.clone();
    let limit = request
        .max_chunk_bytes
        .min(ModelLimits::default().max_model_bytes);
    let bytes = match encode(limit) {
        Ok(bytes) => bytes,
        Err(Error::Json(error))
            if error.is_io()
                && error
                    .to_string()
                    .contains("diagnostic metadata exceeds configured limit") =>
        {
            context.identity.checkpoint_id = None;
            return Ok(Chunk {
                context,
                record: Record::Event(EventRecord {
                    category: "model-snapshot-unavailable".into(),
                    severity: "info".into(),
                    message: "The statistical model exceeds the snapshot byte limit".into(),
                }),
                arrays: BTreeMap::new(),
            });
        }
        Err(error) => return Err(error),
    };
    let id = context
        .identity
        .checkpoint_id
        .clone()
        .unwrap_or_else(|| format!("revision-{}", context.identity.learner_update_revision));
    let signature = format!("sha256:{:x}", Sha256::digest(&bytes));
    let configuration = serde_json::to_value(Configuration {
        format: FORMAT.into(),
        reconstruction,
    })?;
    let capabilities = match reconstruction {
        Reconstruction::Portable => vec!["statistical_model_with_features".into()],
        Reconstruction::NeedsFeatures => vec!["statistical_model_requires_features".into()],
    };
    Ok(Chunk {
        context,
        record: Record::ModelSnapshot(ModelSnapshotRecord {
            id,
            architecture: architecture.into(),
            configuration,
            models: BTreeMap::from([(
                model.into(),
                ModelArtifact {
                    bytes: PAYLOAD.into(),
                    signature,
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
            capabilities,
        }),
        arrays: BTreeMap::from([(PAYLOAD.into(), Array::u8(vec![bytes.len()], bytes)?)]),
    })
}

pub(super) fn load<T: DeserializeOwned>(
    chunk: &Chunk,
    architecture: &str,
    model: &str,
    reconstruction: Reconstruction,
    limits: ModelLimits,
) -> Result<T> {
    let Record::ModelSnapshot(snapshot) = &chunk.record else {
        return Err(Error::Config(
            "expected a statistical model snapshot".into(),
        ));
    };
    ensure(
        snapshot.architecture == architecture
            && snapshot.models.len() == 1
            && limits.max_models > 0,
        "unsupported statistical snapshot architecture",
    )?;
    let config = Configuration::deserialize(&snapshot.configuration)?;
    ensure(
        config.format == FORMAT && config.reconstruction == reconstruction,
        "incompatible statistical model reconstruction mode",
    )?;
    let artifact = snapshot
        .models
        .get(model)
        .ok_or_else(|| Error::Config("missing statistical model artifact".into()))?;
    let array = chunk
        .arrays
        .get(&artifact.bytes)
        .ok_or_else(|| Error::Config("missing statistical model payload".into()))?;
    ensure(
        array.byte_len() <= limits.max_model_bytes && array.shape().len() == 1,
        "statistical model payload exceeds admission limits",
    )?;
    let bytes = array.as_u8()?;
    ensure(
        artifact.signature == format!("sha256:{:x}", Sha256::digest(bytes)),
        "statistical model payload checksum differs",
    )?;
    chunk.validate()?;
    Ok(serde_json::from_slice(bytes)?)
}
