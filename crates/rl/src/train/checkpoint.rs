//! Versioned filesystem encoding; algorithm persistence belongs to each algorithm.
pub use crate::core::checkpoint::{Checkpoint, LearnerRecord, ModelRecord, Snapshot};
#[cfg(feature = "neural")]
use crate::nn::checkpoint::RecordBurnExt;
use crate::{Error, Result, core::ensure};
#[cfg(feature = "neural")]
use burn::module::Module;
use serde::{Deserialize, Serialize};
use serde_json::{ser::Formatter, value::RawValue};
use std::{
    io::{BufWriter, Write},
    path::Path,
};
const VERSION: u32 = 2;
const RUNTIME: &str = "burn-0.22-flex-f32";
#[derive(Deserialize)]
struct Envelope<'a> {
    version: u32,
    kind: String,
    runtime: String,
    learner: Option<LearnerRecord>,
    #[serde(borrow)]
    snapshot: Option<&'a RawValue>,
    snapshot_type: Option<String>,
    #[cfg_attr(not(feature = "neural"), allow(dead_code))]
    policy: Option<ModelRecord>,
    #[cfg_attr(not(feature = "neural"), allow(dead_code))]
    action_space: Option<crate::core::action::ActionSpace>,
}
#[derive(Serialize)]
struct EnvelopeRef<'a, S: Serialize> {
    version: u32,
    kind: &'a str,
    runtime: &'static str,
    learner: Option<&'a LearnerRecord>,
    snapshot: Option<&'a S>,
    snapshot_type: Option<&'a str>,
    policy: Option<&'a ModelRecord>,
    action_space: Option<&'a crate::core::action::ActionSpace>,
}
fn envelope<'a, S: Serialize>(kind: &'a str) -> EnvelopeRef<'a, S> {
    EnvelopeRef {
        version: VERSION,
        kind,
        runtime: RUNTIME,
        learner: None,
        snapshot: None,
        snapshot_type: None,
        policy: None,
        action_space: None,
    }
}
fn checksum(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325, |hash, b| {
        (hash ^ *b as u64).wrapping_mul(0x100000001b3)
    })
}
struct ChecksumWriter<W> {
    inner: W,
    len: usize,
    hash: u64,
}
/// `to_value` historically promoted finite f32 values to f64 before writing JSON.
/// Retain those numerical representations while serializing the snapshot by reference.
struct LegacyFloatFormatter;
impl Formatter for LegacyFloatFormatter {
    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> std::io::Result<()> {
        serde_json::ser::CompactFormatter.write_f64(writer, value as f64)
    }
}
impl<W: Write> Write for ChecksumWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(bytes)?;
        self.len += written;
        self.hash = bytes[..written].iter().fold(self.hash, |hash, byte| {
            (hash ^ *byte as u64).wrapping_mul(0x100000001b3)
        });
        Ok(written)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
fn write<S: Serialize>(path: &Path, data: &EnvelopeRef<'_, S>) -> Result<()> {
    std::fs::create_dir(path)?;
    let file = std::fs::File::create(path.join("checkpoint.json"))?;
    let mut buffer = BufWriter::new(file);
    let (len, hash) = {
        let mut writer = ChecksumWriter {
            inner: &mut buffer,
            len: 0,
            hash: 0xcbf29ce484222325,
        };
        data.serialize(&mut serde_json::Serializer::with_formatter(
            &mut writer,
            LegacyFloatFormatter,
        ))?;
        (writer.len, writer.hash)
    };
    buffer.flush()?;
    buffer.get_ref().sync_all()?;
    let mut marker = std::fs::File::create(path.join("COMPLETE"))?;
    write!(marker, "{len}:{hash:016x}")?;
    marker.sync_all()?;
    Ok(())
}
fn read_verified_checkpoint(path: &Path) -> Result<Vec<u8>> {
    let marker = std::fs::read_to_string(path.join("COMPLETE"))
        .map_err(|e| Error::State(format!("checkpoint incomplete: {e}")))?;
    let bytes = std::fs::read(path.join("checkpoint.json"))?;
    ensure(
        marker == format!("{}:{:016x}", bytes.len(), checksum(&bytes)),
        "checkpoint incomplete or checksum mismatch",
    )?;
    Ok(bytes)
}
fn decode_envelope<'a>(bytes: &'a [u8], kind: &str) -> Result<Envelope<'a>> {
    let data: Envelope<'a> = serde_json::from_slice(bytes)?;
    ensure(
        data.version == VERSION && data.runtime == RUNTIME,
        "unsupported checkpoint version/runtime (expected schema 2; schema 1 is not migrated)",
    )?;
    ensure(
        data.kind == kind,
        "checkpoint kind mismatch: learner restoration and run continuation are distinct",
    )?;
    Ok(data)
}
pub fn save_learner(path: impl AsRef<Path>, learner: &impl Checkpoint) -> Result<()> {
    let record = learner.checkpoint()?;
    let mut e: EnvelopeRef<'_, ()> = envelope("learner");
    e.learner = Some(&record);
    write(path.as_ref(), &e)
}
pub fn load_learner(path: impl AsRef<Path>, learner: &mut impl Checkpoint) -> Result<()> {
    let record = decode_envelope(&read_verified_checkpoint(path.as_ref())?, "learner")?
        .learner
        .ok_or_else(|| Error::State("missing learner".into()))?;
    learner.restore(&record)
}
pub fn save_run<S: Snapshot>(
    path: impl AsRef<Path>,
    learner: &impl Checkpoint,
    state: &S,
) -> Result<()> {
    state.validate_snapshot()?;
    let record = learner.checkpoint()?;
    state.validate_learner_record(&record)?;
    learner.validate_anchor_payloads(&record, &state.reconstruction_anchors())?;
    let mut e: EnvelopeRef<'_, S> = envelope("run");
    e.learner = Some(&record);
    e.snapshot = Some(state);
    e.snapshot_type = Some(S::record_type());
    write(path.as_ref(), &e)
}
pub fn load_run<S: Snapshot>(path: impl AsRef<Path>, learner: &mut impl Checkpoint) -> Result<S> {
    load_run_candidate(path.as_ref(), learner, false).map(|(state, _)| state)
}
/// Explicit cross-runtime import. Concrete codecs may replace *validated*
/// foreign recurrent memory with native full-history reconstruction. Ordinary
/// `load_run` remains strict. All other validation precedes learner installation.
pub fn load_run_portable<S: Snapshot>(
    path: impl AsRef<Path>,
    learner: &mut impl Checkpoint,
) -> Result<(S, crate::core::checkpoint::PortableRestoreReport)> {
    load_run_candidate(path.as_ref(), learner, true)
}
fn load_run_candidate<S: Snapshot>(
    path: &Path,
    learner: &mut impl Checkpoint,
    portable: bool,
) -> Result<(S, crate::core::checkpoint::PortableRestoreReport)> {
    let (mut state, record): (S, _) = decode_run(path)?;
    let report = if portable {
        state.prepare_portable_restore()?
    } else {
        crate::core::checkpoint::PortableRestoreReport::default()
    };
    state.validate_snapshot()?;
    state.validate_learner_record(&record)?;
    learner.validate_anchor_payloads(&record, &state.reconstruction_anchors())?;
    learner.restore(&record)?;
    Ok((state, report))
}
fn decode_run<S: Snapshot>(path: &Path) -> Result<(S, LearnerRecord)> {
    let bytes = read_verified_checkpoint(path)?;
    let e = decode_envelope(&bytes, "run")?;
    ensure(
        e.snapshot_type.as_deref() == Some(S::record_type()),
        "run snapshot type mismatch",
    )?;
    let snapshot = e
        .snapshot
        .ok_or_else(|| Error::State("missing run state".into()))?;
    let record = e
        .learner
        .ok_or_else(|| Error::State("missing learner".into()))?;
    // Decode straight from the verified bytes and release them before auditing the run.
    Ok((serde_json::from_str(snapshot.get())?, record))
}
#[cfg(feature = "neural")]
pub fn export_policy<M: Module>(
    path: impl AsRef<Path>,
    model: &M,
    space: &crate::core::action::ActionSpace,
) -> Result<()> {
    space.validate()?;
    let mut e: EnvelopeRef<'_, ()> = envelope("policy");
    let mut record = LearnerRecord::new("policy", ())?;
    record.model("policy", model)?;
    e.policy = record.models.get("policy");
    e.action_space = Some(space);
    write(path.as_ref(), &e)
}
#[cfg(feature = "neural")]
pub fn load_policy<M: Module>(
    path: impl AsRef<Path>,
    template: &M,
    space: &crate::core::action::ActionSpace,
) -> Result<M> {
    let policy = {
        let bytes = read_verified_checkpoint(path.as_ref())?;
        let e = decode_envelope(&bytes, "policy")?;
        ensure(
            e.action_space.as_ref() == Some(space),
            "policy action space mismatch",
        )?;
        e.policy
            .ok_or_else(|| Error::State("missing policy".into()))?
    };
    let mut record = LearnerRecord::new("policy", ())?;
    record.models.insert("policy".into(), policy);
    Ok(record.load_model("policy", template)?.valid())
}
