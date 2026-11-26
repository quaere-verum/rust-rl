//! Incremental single-writer recording with transactional immutable chunks.
use super::{codec, schema::*};
use crate::{Error, Result, core::ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

mod reuse;
pub use reuse::DecodedChunkCache;

const STATUS_RESERVE: u64 = 8192;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub max_total_bytes: u64,
    /// Bounds uncompressed arrays AND encoded chunk size, including metadata.
    pub max_chunk_bytes: usize,
    pub max_metadata_bytes: usize,
    pub compress: bool,
}
impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_total_bytes: 2 * 1024 * 1024 * 1024,
            max_chunk_bytes: 64 * 1024 * 1024,
            max_metadata_bytes: 1024 * 1024,
            compress: true,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadLimits {
    pub max_chunk_bytes: usize,
    pub max_metadata_bytes: usize,
    pub max_index_entries: usize,
    /// Bounds encoded index metadata, independently of numeric chunks.
    pub max_index_bytes: u64,
}
impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            max_chunk_bytes: 64 * 1024 * 1024,
            max_metadata_bytes: 1024 * 1024,
            max_index_entries: 1_000_000,
            max_index_bytes: 128 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEntry {
    pub sequence: u64,
    pub file: String,
    pub kind: RecordKind,
    pub revision: u64,
    pub model_role: String,
    pub probe_set: Option<String>,
    pub trajectory: Option<String>,
    pub checkpoint_id: Option<String>,
    pub bytes: u64,
    pub checksum: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page: Option<PageAddress>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageAddress {
    pub first_sequence: u64,
    pub offset: u64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BufferConfig {
    pub max_records: usize,
    /// Conservative retained heap charge (payloads, index lines/strings and queue
    /// capacity). The current encoder's scratch is separately bounded by StorageConfig.
    pub max_bytes: usize,
}
impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_records: 32,
            max_bytes: 8 * 1024 * 1024,
        }
    }
}
impl BufferConfig {
    fn validate(&self) -> Result<()> {
        ensure(
            (1..=1024).contains(&self.max_records)
                && (1024..=64 * 1024 * 1024).contains(&self.max_bytes),
            "page buffering supports 1..1024 records and 1 KiB..64 MiB",
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingWindow {
    pub first_sequence: u64,
    pub maximum_records: usize,
    pub maximum_bytes: usize,
}
struct Buffered {
    entry: IndexEntry,
    bytes: Vec<u8>,
    line: Vec<u8>,
}
fn buffer_charge(entry: &IndexEntry, bytes: &Vec<u8>, line: &Vec<u8>) -> usize {
    bytes
        .capacity()
        .saturating_add(line.capacity())
        .saturating_add(entry.file.capacity())
        .saturating_add(entry.model_role.capacity())
        .saturating_add(
            [&entry.probe_set, &entry.trajectory, &entry.checkpoint_id]
                .iter()
                .map(|s| s.as_ref().map_or(0, String::capacity))
                .sum::<usize>(),
        )
        .saturating_add(4 * std::mem::size_of::<Buffered>())
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecordingStatus {
    pub committed_records: u64,
    pub committed_bytes: u64,
    pub skipped_records: u64,
    pub last_skip_reason: Option<String>,
    pub complete: bool,
    #[serde(default)]
    pub reclaimed_bytes: u64,
    #[serde(default)]
    pub pruned_records: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Retention {
    version: u32,
    keep_last_per_role: usize,
    through: BTreeMap<String, u64>,
}
impl Retention {
    fn retained(&self, entry: &IndexEntry) -> bool {
        entry.kind != RecordKind::ModelSnapshot
            || self
                .through
                .get(&entry.model_role)
                .is_none_or(|&s| entry.sequence > s)
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppendOutcome {
    Committed { sequence: u64 },
    Skipped { reason: String },
    Disabled,
    Buffered { sequence: u64 },
}

/// Synchronous backpressure. By default every append is durable; optional bounded
/// page buffering returns `Buffered` until `flush` or `finish` commits the page.
pub struct Recorder {
    root: PathBuf,
    manifest: RunManifest,
    config: StorageConfig,
    index: File,
    status: RecordingStatus,
    poisoned: bool,
    retention: Option<Retention>,
    retained_models: BTreeMap<String, VecDeque<IndexEntry>>,
    buffering: Option<BufferConfig>,
    pending: Vec<Buffered>,
    pending_bytes: usize,
    pending_disk_bytes: u64,
    pending_memory_bytes: usize,
}
impl Recorder {
    pub fn create(
        path: impl AsRef<Path>,
        manifest: RunManifest,
        config: StorageConfig,
    ) -> Result<Self> {
        manifest.validate()?;
        super::reference::manifest_context(&manifest)?;
        ensure(
            manifest.schema_version == SCHEMA_VERSION,
            "new recordings require the current schema version",
        )?;
        ensure(
            config.max_total_bytes > STATUS_RESERVE
                && config.max_chunk_bytes > 0
                && config.max_metadata_bytes > 0
                && config.max_metadata_bytes <= u32::MAX as usize,
            "invalid diagnostic storage limits",
        )?;
        let bytes = codec::bounded_json(&manifest, config.max_metadata_bytes)?;
        ensure(
            bytes.len() as u64 + STATUS_RESERVE < config.max_total_bytes,
            "manifest exceeds recording budget",
        )?;
        let root = path.as_ref().to_owned();
        // Exclusive directory ownership prevents an accidental overwrite or a
        // second writer. Interrupted recordings remain readable, not resumed.
        fs::create_dir(&root)?;
        fs::create_dir(root.join("chunks"))?;
        atomic_write(&root.join("manifest.json"), &bytes)?;
        let index = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("index.jsonl"))?;
        let recorder = Self {
            root,
            manifest,
            config,
            index,
            status: RecordingStatus {
                committed_bytes: bytes.len() as u64,
                ..Default::default()
            },
            poisoned: false,
            retention: None,
            retained_models: BTreeMap::new(),
            buffering: None,
            pending: vec![],
            pending_bytes: 0,
            pending_disk_bytes: 0,
            pending_memory_bytes: 0,
        };
        recorder.write_status()?;
        Ok(recorder)
    }
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }
    /// Configure before the first append. Retains the last N snapshots per model
    /// role; probes and metrics are never pruned. Tombstones are synced before
    /// deletion. A failed replacement may leave N-1 snapshots, with an explicit
    /// retention marker, while all prior non-model data remains readable.
    pub fn with_snapshot_retention(mut self, keep_last_per_role: usize) -> Result<Self> {
        ensure(
            self.status.committed_records == 0
                && self.pending.is_empty()
                && (1..=64).contains(&keep_last_per_role),
            "retention must be set before recording and keep 1..64 models per role",
        )?;
        self.retention = Some(Retention {
            version: 1,
            keep_last_per_role,
            through: BTreeMap::new(),
        });
        self.write_retention()?;
        Ok(self)
    }
    fn write_retention(&self) -> Result<()> {
        atomic_write(
            &self.root.join("retention.json"),
            &codec::bounded_json(self.retention.as_ref().unwrap(), 4096)?,
        )
    }
    pub fn status(&self) -> &RecordingStatus {
        &self.status
    }
    /// Pages become visible atomically before their index entries. A durable
    /// pending-window marker warns of possibly uncommitted samples after a crash.
    /// Models always use separate durable files so retention remains independent.
    pub fn with_buffering(mut self, config: BufferConfig) -> Result<Self> {
        ensure(
            self.status.committed_records == 0 && self.pending.is_empty(),
            "configure buffering before the first record",
        )?;
        config.validate()?;
        fs::create_dir(self.root.join("pages"))?;
        self.buffering = Some(config);
        Ok(self)
    }
    pub fn pending_records(&self) -> usize {
        self.pending.len()
    }
    pub fn pending_memory_bytes(&self) -> usize {
        self.pending_memory_bytes
    }
    pub fn flush(&mut self) -> Result<()> {
        ensure(!self.poisoned, "cannot flush a failed diagnostic writer")?;
        if self.pending.is_empty() {
            return Ok(());
        }
        self.poisoned = true;
        let path = self.root.join(&self.pending[0].entry.file);
        let temporary = path.with_extension("part");
        let mut file = File::create(&temporary)?;
        for p in &self.pending {
            file.write_all(&p.bytes)?;
        }
        file.sync_all()?;
        drop(file);
        fs::rename(temporary, path)?;
        for p in &self.pending {
            self.index.write_all(&p.line)?;
        }
        self.index.sync_all()?;
        self.status.committed_records += self.pending.len() as u64;
        self.status.committed_bytes += self.pending_disk_bytes;
        self.pending.clear();
        self.pending_bytes = 0;
        self.pending_disk_bytes = 0;
        self.pending_memory_bytes = 0;
        self.write_status()?;
        fs::remove_file(self.root.join("pending.json"))?;
        self.poisoned = false;
        Ok(())
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    fn write_status(&self) -> Result<()> {
        atomic_write(
            &self.root.join("status.json"),
            &codec::bounded_json(&self.status, (STATUS_RESERVE / 2) as usize)?,
        )
    }
    fn skip(&mut self, reason: &str) -> Result<AppendOutcome> {
        self.status.skipped_records += 1;
        self.status.last_skip_reason = Some(reason.into());
        self.write_status()?;
        Ok(AppendOutcome::Skipped {
            reason: reason.into(),
        })
    }
    pub fn append(&mut self, chunk: &Chunk) -> Result<AppendOutcome> {
        ensure(
            !self.status.complete && !self.poisoned,
            "diagnostic writer is finished or failed",
        )?;
        chunk.validate()?;
        if let Some(d) = super::reference::declaration(chunk)? {
            d.validate_manifest(&self.manifest)?;
        }
        ensure(
            chunk.context.identity.run_id == self.manifest.run_id,
            "diagnostic chunk belongs to a different run",
        )?;
        ensure(
            self.manifest
                .starting_revision
                .checked_add(chunk.context.counters.updates)
                == Some(chunk.context.identity.learner_update_revision),
            "diagnostic revision and run origin differ",
        )?;
        if chunk.array_bytes() > self.config.max_chunk_bytes {
            return self.skip("uncompressed chunk exceeds memory limit");
        }
        if matches!(chunk.record, Record::Typed(_)) {
            super::typed::validate_in(chunk, &self.manifest)?;
        }
        let bytes = match codec::encode(
            chunk,
            self.config.max_chunk_bytes,
            self.config.max_metadata_bytes,
            self.config.compress,
        ) {
            Ok(bytes) => bytes,
            Err(Error::Config(message)) if message.contains("chunk budget") => {
                return self.skip(&message);
            }
            Err(error) => return Err(error),
        };
        let mut buffer = self.buffering.filter(|b| {
            chunk.record.kind() != RecordKind::ModelSnapshot && bytes.len() <= b.max_bytes
        });
        if !self.pending.is_empty()
            && (buffer.is_none()
                || buffer.is_some_and(|b| {
                    self.pending.len() >= b.max_records
                        || self.pending_bytes.saturating_add(bytes.len()) > b.max_bytes
                }))
        {
            self.flush()?;
        }
        let sequence = self.status.committed_records + self.pending.len() as u64;
        let page = buffer.map(|_| PageAddress {
            first_sequence: self.status.committed_records,
            offset: self.pending_bytes as u64,
        });
        let mut entry = IndexEntry {
            sequence,
            file: page.as_ref().map_or_else(
                || format!("chunks/{sequence:012}.rld"),
                |p| format!("pages/{:012}.rlp", p.first_sequence),
            ),
            kind: chunk.record.kind(),
            revision: chunk.context.identity.learner_update_revision,
            model_role: chunk.context.identity.model_role.clone(),
            probe_set: chunk.context.probe_set.clone(),
            trajectory: chunk.context.trajectory.clone(),
            checkpoint_id: chunk.context.identity.checkpoint_id.clone(),
            bytes: bytes.len() as u64,
            checksum: crc32fast::hash(&bytes),
            page,
        };
        let mut line = codec::bounded_json(&entry, self.config.max_metadata_bytes)?;
        line.push(b'\n');
        let mut charge = buffer_charge(&entry, &bytes, &line);
        if let Some(config) = buffer
            && self.pending_memory_bytes.saturating_add(charge) > config.max_bytes
        {
            self.flush()?;
            entry.page = Some(PageAddress {
                first_sequence: sequence,
                offset: 0,
            });
            entry.file = format!("pages/{sequence:012}.rlp");
            line = codec::bounded_json(&entry, self.config.max_metadata_bytes)?;
            line.push(b'\n');
            charge = buffer_charge(&entry, &bytes, &line);
            if charge > config.max_bytes {
                buffer = None;
                entry.page = None;
                entry.file = format!("chunks/{sequence:012}.rld");
                line = codec::bounded_json(&entry, self.config.max_metadata_bytes)?;
                line.push(b'\n');
            }
        }
        let additional = bytes.len() as u64 + line.len() as u64;
        let retiring = if entry.kind == RecordKind::ModelSnapshot
            && let Some(retention) = &self.retention
        {
            ensure(
                entry.model_role.len() <= 128
                    && (self.retained_models.contains_key(&entry.model_role)
                        || self.retained_models.len() < 16),
                "snapshot retention supports at most 16 roles of 128 bytes",
            )?;
            self.retained_models
                .get(&entry.model_role)
                .filter(|q| q.len() >= retention.keep_last_per_role)
                .and_then(|q| q.front())
                .cloned()
        } else {
            None
        };
        let retained_bytes = self
            .status
            .committed_bytes
            .saturating_sub(self.status.reclaimed_bytes)
            .saturating_sub(retiring.as_ref().map_or(0, |e| e.bytes))
            .saturating_add(self.pending_disk_bytes);
        if additional
            > self
                .config
                .max_total_bytes
                .saturating_sub(retained_bytes + STATUS_RESERVE)
        {
            return self.skip("recording storage budget exhausted");
        }
        if let Some(config) = buffer {
            if self.pending.is_empty() {
                self.poisoned = true;
                atomic_write(
                    &self.root.join("pending.json"),
                    &codec::bounded_json(
                        &PendingWindow {
                            first_sequence: sequence,
                            maximum_records: config.max_records,
                            maximum_bytes: config.max_bytes,
                        },
                        1024,
                    )?,
                )?;
                self.poisoned = false;
            }
            self.pending_bytes += bytes.len();
            self.pending_memory_bytes += charge;
            self.pending_disk_bytes += additional;
            self.pending.push(Buffered { entry, bytes, line });
            if self.pending.len() == config.max_records {
                self.flush()?;
                return Ok(AppendOutcome::Committed { sequence });
            }
            return Ok(AppendOutcome::Buffered { sequence });
        }
        // On any I/O failure, do not reuse the sequence or append beyond a
        // potentially partial index line. Valid older chunks remain available.
        self.poisoned = true;
        if let Some(old) = retiring {
            self.retention
                .as_mut()
                .unwrap()
                .through
                .insert(old.model_role.clone(), old.sequence);
            self.write_retention()?;
            fs::remove_file(self.root.join(&old.file))?;
            self.status.reclaimed_bytes += old.bytes;
            self.status.pruned_records += 1;
            self.retained_models
                .get_mut(&old.model_role)
                .unwrap()
                .pop_front();
        }
        atomic_write(&self.root.join(&entry.file), &bytes)?;
        self.index.write_all(&line)?;
        self.index.sync_all()?;
        self.status.committed_records += 1;
        self.status.committed_bytes += additional;
        if entry.kind == RecordKind::ModelSnapshot && self.retention.is_some() {
            self.retained_models
                .entry(entry.model_role.clone())
                .or_default()
                .push_back(entry);
        }
        self.write_status()?;
        self.poisoned = false;
        Ok(AppendOutcome::Committed { sequence })
    }
    pub fn finish(&mut self) -> Result<()> {
        ensure(!self.poisoned, "cannot complete a failed diagnostic writer")?;
        self.flush()?;
        self.status.complete = true;
        self.write_status()?;
        atomic_write(
            &self.root.join("complete.json"),
            &codec::bounded_json(&self.status, (STATUS_RESERVE / 2) as usize)?,
        )
    }
}
pub struct OptionalRecorder(Option<Recorder>);
impl OptionalRecorder {
    pub const fn disabled() -> Self {
        Self(None)
    }
    pub fn enabled(recorder: Recorder) -> Self {
        Self(Some(recorder))
    }
    pub fn record(&mut self, build: impl FnOnce() -> Result<Chunk>) -> Result<AppendOutcome> {
        match &mut self.0 {
            Some(recorder) => recorder.append(&build()?),
            None => Ok(AppendOutcome::Disabled),
        }
    }
    pub fn finish(&mut self) -> Result<()> {
        if let Some(recorder) = &mut self.0 {
            recorder.finish()?;
        }
        Ok(())
    }
}

/// Loads only the small index. Numeric arrays are checked and loaded on demand.
#[derive(Clone)]
pub struct Recording {
    root: PathBuf,
    manifest: Arc<RunManifest>,
    limits: ReadLimits,
    entries: Arc<Vec<IndexEntry>>,
    offset: u64,
    issues: Vec<String>,
    complete: bool,
    retention: Option<Retention>,
}
/// Small immutable address for loading a single committed record on another
/// thread. It does not copy the run index or retain any numeric payload.
#[derive(Clone)]
pub struct ChunkSource {
    root: PathBuf,
    manifest: Arc<RunManifest>,
    limits: ReadLimits,
    entry: IndexEntry,
}
impl ChunkSource {
    /// Encoded length and recording checksum for this immutable address. This
    /// has the same integrity semantics as ordinary recording reads; it is not
    /// a cryptographic digest.
    pub fn encoded_identity(&self) -> (u64, u32) {
        (self.entry.bytes, self.entry.checksum)
    }
    /// Recheck an immutable address before using an already restored model.
    /// Does not read weights or replace a retired source with a sibling.
    pub fn check_available(&self) -> Result<()> {
        if self.entry.kind == RecordKind::ModelSnapshot {
            ensure(
                read_retention(&self.root)?.is_none_or(|r| r.retained(&self.entry)),
                "model snapshot removed by explicit retention policy",
            )?;
        }
        ensure(
            self.root.join(&self.entry.file).is_file(),
            "selected immutable source file is missing",
        )
    }
    pub fn load(&self) -> Result<Chunk> {
        load_chunk(&self.root, &self.manifest, &self.limits, &self.entry)
    }
    /// Defer materialization when a conservative retained charge exceeds a
    /// caller's budget. `None` is admission refusal, not payload validation.
    /// Encoded length is bounded by `limit`; decoded logical length is admitted
    /// before decompression. Scratch is separate from retained charge: the
    /// existing decoder checks one extra byte and its Vec capacity may grow.
    /// Refusal leaves checksum/schema/numeric validation to ordinary `load`.
    pub fn load_if_within_charge(&self, limit: usize) -> Result<Option<Chunk>> {
        load_chunk_admitted(
            &self.root,
            &self.manifest,
            &self.limits,
            &self.entry,
            Some(limit),
        )
    }
}
pub fn read_manifest(root: impl AsRef<Path>, max_bytes: usize) -> Result<RunManifest> {
    let manifest: RunManifest = serde_json::from_slice(&read_limited(
        &root.as_ref().join("manifest.json"),
        max_bytes,
    )?)?;
    manifest.validate()?;
    super::reference::manifest_context(&manifest)?;
    Ok(manifest)
}
impl Recording {
    pub fn open(path: impl AsRef<Path>, limits: ReadLimits) -> Result<Self> {
        let root = path.as_ref().to_owned();
        let manifest = Arc::new(read_manifest(&root, limits.max_metadata_bytes)?);
        let mut recording = Self {
            root,
            manifest,
            limits,
            entries: Arc::new(vec![]),
            offset: 0,
            issues: vec![],
            complete: false,
            retention: None,
        };
        recording.refresh()?;
        Ok(recording)
    }
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }
    pub fn entries(&self) -> &[IndexEntry] {
        &self.entries
    }
    pub fn is_retained(&self, index: usize) -> bool {
        self.entries
            .get(index)
            .is_some_and(|e| self.retention.as_ref().is_none_or(|r| r.retained(e)))
    }
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
    pub fn complete(&self) -> bool {
        self.complete
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn status(&self) -> Result<RecordingStatus> {
        Ok(serde_json::from_slice(&read_limited(
            &self.root.join("status.json"),
            (STATUS_RESERVE / 2) as usize,
        )?)?)
    }
    /// While a writer is live this is an uncommitted window. After interruption,
    /// up to these declared limits may be absent; do not present it as a full run.
    pub fn pending_window(&self) -> Result<Option<PendingWindow>> {
        let path = self.root.join("pending.json");
        if !path.exists() {
            return Ok(None);
        }
        let window: PendingWindow = serde_json::from_slice(&read_limited(&path, 1024)?)?;
        BufferConfig {
            max_records: window.maximum_records,
            max_bytes: window.maximum_bytes,
        }
        .validate()?;
        Ok(Some(window))
    }
    /// Reads only appended metadata; a partial last line is retried on refresh.
    pub fn refresh(&mut self) -> Result<usize> {
        self.retention = read_retention(&self.root)?;
        self.issues
            .retain(|issue| !issue.starts_with("incomplete index tail"));
        let mut file = File::open(self.root.join("index.jsonl"))?;
        ensure(
            file.metadata()?.len() <= self.limits.max_index_bytes,
            "diagnostic index exceeds metadata byte budget",
        )?;
        ensure(
            file.metadata()?.len() >= self.offset,
            "diagnostic index was truncated",
        )?;
        file.seek(SeekFrom::Start(self.offset))?;
        let mut reader = BufReader::new(file);
        let before = self.entries.len();
        loop {
            let mut line = Vec::new();
            let size = reader
                .by_ref()
                .take(self.limits.max_metadata_bytes as u64 + 1)
                .read_until(b'\n', &mut line)?;
            if size == 0 {
                break;
            }
            ensure(
                self.entries.len().saturating_add(self.issues.len())
                    < self.limits.max_index_entries,
                "diagnostic index and issue count exceed read limit",
            )?;
            ensure(
                size <= self.limits.max_metadata_bytes,
                "diagnostic index entry exceeds read limit",
            )?;
            if !line.ends_with(b"\n") {
                self.issues
                    .push(format!("incomplete index tail at byte {}", self.offset));
                break;
            }
            self.offset += size as u64;
            match serde_json::from_slice::<IndexEntry>(&line) {
                Ok(entry) => {
                    ensure(
                        self.entries.len() < self.limits.max_index_entries,
                        "diagnostic index exceeds entry limit",
                    )?;
                    if !valid_address(&entry) {
                        self.issues
                            .push(format!("invalid chunk path at sequence {}", entry.sequence));
                    } else if self
                        .entries
                        .last()
                        .is_some_and(|previous| previous.sequence >= entry.sequence)
                    {
                        self.issues
                            .push(format!("non-increasing chunk sequence {}", entry.sequence));
                    } else {
                        Arc::make_mut(&mut self.entries).push(entry);
                    }
                }
                Err(error) => self.issues.push(format!(
                    "corrupt index entry ending at {}: {error}",
                    self.offset
                )),
            }
        }
        self.complete = if self.root.join("complete.json").exists() {
            let marker: RecordingStatus = serde_json::from_slice(&read_limited(
                &self.root.join("complete.json"),
                (STATUS_RESERVE / 2) as usize,
            )?)?;
            marker.complete
                && self.pending_window()?.is_none()
                && marker.committed_records == self.entries.len() as u64
                && self.issues.is_empty()
        } else {
            false
        };
        Ok(self.entries.len() - before)
    }
    pub fn load(&self, index: usize) -> Result<Chunk> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| Error::State("diagnostic index out of range".into()))?;
        load_chunk(&self.root, &self.manifest, &self.limits, entry)
    }
    pub fn source(&self, index: usize) -> Result<ChunkSource> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| Error::State("diagnostic index out of range".into()))?;
        Ok(ChunkSource {
            root: self.root.clone(),
            manifest: self.manifest.clone(),
            limits: self.limits.clone(),
            entry: entry.clone(),
        })
    }
}
fn load_chunk(
    root: &Path,
    manifest: &RunManifest,
    limits: &ReadLimits,
    entry: &IndexEntry,
) -> Result<Chunk> {
    Ok(load_chunk_admitted(root, manifest, limits, entry, None)?.expect("unlimited admission"))
}
fn load_chunk_admitted(
    root: &Path,
    manifest: &RunManifest,
    limits: &ReadLimits,
    entry: &IndexEntry,
    charge_limit: Option<usize>,
) -> Result<Option<Chunk>> {
    let Some(bytes) = read_chunk_admitted(root, limits, entry, charge_limit)? else {
        return Ok(None);
    };
    decode_chunk_admitted(&bytes, manifest, limits, entry, charge_limit)
}

fn read_chunk_admitted(
    root: &Path,
    limits: &ReadLimits,
    entry: &IndexEntry,
    charge_limit: Option<usize>,
) -> Result<Option<Vec<u8>>> {
    if entry.kind == RecordKind::ModelSnapshot {
        ensure(
            read_retention(root)?.is_none_or(|r| r.retained(entry)),
            "model snapshot removed by explicit retention policy",
        )?;
    }
    ensure(
        entry.bytes <= limits.max_chunk_bytes as u64,
        "diagnostic chunk exceeds read limit",
    )?;
    ensure(valid_address(entry), "invalid diagnostic chunk address")?;
    if charge_limit.is_some_and(|limit| entry.bytes > limit as u64) {
        return Ok(None);
    }
    let bytes = if let Some(page) = &entry.page {
        let mut file = File::open(root.join(&entry.file))?;
        ensure(
            page.offset
                .checked_add(entry.bytes)
                .is_some_and(|end| end <= file.metadata().map_or(0, |m| m.len())),
            "incomplete diagnostic page",
        )?;
        file.seek(SeekFrom::Start(page.offset))?;
        let mut bytes = vec![0; entry.bytes as usize];
        file.read_exact(&mut bytes)?;
        bytes
    } else {
        read_chunk_limited(
            &root.join(&entry.file),
            charge_limit.map_or(limits.max_chunk_bytes, |limit| {
                limit.min(limits.max_chunk_bytes)
            }),
            entry.bytes as usize,
        )?
    };
    ensure(
        bytes.len() as u64 == entry.bytes && crc32fast::hash(&bytes) == entry.checksum,
        "diagnostic chunk checksum or length mismatch",
    )?;
    Ok(Some(bytes))
}

fn decode_chunk_admitted(
    bytes: &[u8],
    manifest: &RunManifest,
    limits: &ReadLimits,
    entry: &IndexEntry,
    charge_limit: Option<usize>,
) -> Result<Option<Chunk>> {
    let chunk = if let Some(limit) = charge_limit {
        let Some(chunk) = codec::decode_if_within_charge(
            bytes,
            limits.max_chunk_bytes,
            limits.max_metadata_bytes,
            limit,
        )?
        else {
            return Ok(None);
        };
        chunk
    } else {
        codec::decode(bytes, limits.max_chunk_bytes, limits.max_metadata_bytes)?
    };
    ensure(
        chunk.context.identity.run_id == manifest.run_id
            && manifest
                .starting_revision
                .checked_add(chunk.context.counters.updates)
                == Some(entry.revision)
            && chunk.context.identity.learner_update_revision == entry.revision
            && chunk.record.kind() == entry.kind
            && chunk.context.identity.model_role == entry.model_role
            && chunk.context.probe_set == entry.probe_set
            && chunk.context.trajectory == entry.trajectory
            && chunk.context.identity.checkpoint_id == entry.checkpoint_id,
        "diagnostic index and chunk identity differ",
    )?;
    if let Some(d) = super::reference::declaration(&chunk)? {
        d.validate_manifest(manifest)?;
    }
    if matches!(chunk.record, Record::Typed(_)) {
        super::typed::validate_in(&chunk, manifest)?;
    }
    Ok(Some(chunk))
}

fn read_retention(root: &Path) -> Result<Option<Retention>> {
    let path = root.join("retention.json");
    if !path.exists() {
        return Ok(None);
    }
    let r: Retention = serde_json::from_slice(&read_limited(&path, 4096)?)?;
    ensure(
        r.version == 1
            && (1..=64).contains(&r.keep_last_per_role)
            && r.through.len() <= 16
            && r.through.keys().all(|s| s.len() <= 128),
        "invalid/unsupported retention manifest",
    )?;
    Ok(Some(r))
}
fn valid_address(entry: &IndexEntry) -> bool {
    match &entry.page {
        None => entry.file == format!("chunks/{:012}.rld", entry.sequence),
        Some(p) => {
            entry.kind != RecordKind::ModelSnapshot
                && p.first_sequence <= entry.sequence
                && entry.sequence - p.first_sequence < 1024
                && p.offset <= 64 * 1024 * 1024
                && entry.file == format!("pages/{:012}.rlp", p.first_sequence)
        }
    }
}
fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    ensure(
        file.metadata()?.len() <= limit as u64,
        "diagnostic file exceeds read limit",
    )?;
    read_limited_body(file, limit, 0)
}
fn read_chunk_limited(path: &Path, limit: usize, expected: usize) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let observed = file.metadata()?.len();
    ensure(
        observed <= limit as u64,
        "diagnostic file exceeds read limit",
    )?;
    // Metadata is a capacity hint only. Leave length/CRC verification and the
    // limit-plus-one read intact so short, replaced and growing files behave
    // exactly as before. One spare byte permits an EOF probe without growth.
    let capacity = expected.min(observed as usize).saturating_add(1);
    read_limited_body(file, limit, capacity)
}
fn read_limited_body(reader: impl Read, limit: usize, capacity: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(capacity);
    reader.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure(
        bytes.len() <= limit,
        "diagnostic file grew beyond read limit",
    )?;
    Ok(bytes)
}
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let pending = path.with_extension("part");
    let mut file = File::create(&pending)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(pending, path)?;
    Ok(())
}

#[cfg(test)]
mod bounded_load_tests {
    use super::*;
    use crate::core::action::ActionSpace;

    #[test]
    fn standalone_read_limits_checksum_and_admission_keep_their_error_order() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("recording");
        let manifest = RunManifest::new(
            "read-hint",
            "fixture",
            "fixture",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
        let chunk = Chunk {
            context: RecordContext::new("read-hint", 0),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::from([(
                "a".into(),
                Array::u8(vec![16384], vec![42; 16384]).unwrap(),
            )]),
        };
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        let recording = Recording::open(&path, ReadLimits::default()).unwrap();
        let mut source = recording.source(0).unwrap();
        let file = path.join(&source.entry.file);
        let bytes = fs::read(&file).unwrap();
        assert!(
            bytes.len() < chunk.array_bytes(),
            "fixture must be compressed"
        );
        source.limits.max_chunk_bytes = chunk.array_bytes();
        assert!(source.load().is_ok());
        for replacement in [
            vec![],
            bytes[..bytes.len() - 1].to_vec(),
            [bytes.as_slice(), &[0]].concat(),
        ] {
            fs::write(&file, replacement).unwrap();
            assert!(
                source
                    .load()
                    .unwrap_err()
                    .to_string()
                    .contains("chunk checksum or length mismatch")
            );
        }
        let mut corrupt = bytes.clone();
        corrupt[0] ^= 1;
        fs::write(&file, corrupt).unwrap();
        assert!(
            source
                .load()
                .unwrap_err()
                .to_string()
                .contains("chunk checksum or length mismatch")
        );
        fs::write(&file, vec![0; source.limits.max_chunk_bytes + 1]).unwrap();
        assert!(
            source
                .load()
                .unwrap_err()
                .to_string()
                .contains("file exceeds read limit")
        );
        fs::remove_file(&file).unwrap();
        assert!(
            source
                .load_if_within_charge(source.entry.bytes as usize - 1)
                .unwrap()
                .is_none()
        );
        assert!(
            matches!(source.load(), Err(Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound)
        );
        fs::write(&file, &bytes).unwrap();
        // The standalone read helper, independently from decoded-array limits.
        assert_eq!(
            read_chunk_limited(&file, bytes.len(), bytes.len()).unwrap(),
            bytes
        );
        // Correct transport metadata continues to tolerate bytes after Deflate.
        let trailing = [bytes.as_slice(), &[255, 255, 0, 42]].concat();
        fs::write(&file, &trailing).unwrap();
        source.entry.bytes = trailing.len() as u64;
        source.entry.checksum = crc32fast::hash(&trailing);
        assert_eq!(source.load().unwrap().array_bytes(), chunk.array_bytes());
    }

    #[test]
    fn capacity_hint_does_not_bound_reads_or_hide_growth_and_reader_errors() {
        struct Partial<'a> {
            bytes: &'a [u8],
            interrupt: bool,
            error: bool,
        }
        impl Read for Partial<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if std::mem::take(&mut self.interrupt) {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.is_empty() && self.error {
                    return Err(std::io::Error::other("controlled read failure"));
                }
                let len = out.len().min(3);
                self.bytes.read(&mut out[..len])
            }
        }
        let reader = |bytes, error| Partial {
            bytes,
            interrupt: true,
            error,
        };
        // Observed length/capacity may be smaller than the bytes subsequently read.
        assert_eq!(
            read_limited_body(reader(b"abcdef", false), 6, 2).unwrap(),
            b"abcdef"
        );
        assert!(
            read_limited_body(reader(b"abcdefg", false), 6, 2)
                .unwrap_err()
                .to_string()
                .contains("file grew beyond read limit")
        );
        assert!(
            matches!(read_limited_body(reader(b"abc", true), 6, 2), Err(Error::Io(e)) if e.to_string() == "controlled read failure")
        );
        // A source growing beyond the limit stops at limit+one, without reading
        // its later error; an error before that boundary still propagates.
        assert!(
            read_limited_body(reader(b"abcdefghij", true), 6, 2)
                .unwrap_err()
                .to_string()
                .contains("file grew beyond read limit")
        );
    }

    #[test]
    fn bounded_load_admits_small_valid_records_and_defers_compressed_large_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recording");
        let mut writer = Recorder::create(
            &path,
            RunManifest::new(
                "bounded",
                "test",
                "test",
                1,
                ActionSpace::Discrete { count: 2 },
                0.9,
            )
            .unwrap(),
            StorageConfig::default(),
        )
        .unwrap();
        for revision in 0..2 {
            let mut chunk = Chunk {
                context: RecordContext::new("bounded", revision),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), revision as f64)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            };
            if revision == 1 {
                chunk.arrays.insert(
                    "large".into(),
                    Array::u8(vec![2 * 1024 * 1024], vec![0; 2 * 1024 * 1024]).unwrap(),
                );
            }
            writer.append(&chunk).unwrap();
        }
        writer.finish().unwrap();
        let recording = Recording::open(&path, ReadLimits::default()).unwrap();
        assert!(
            recording.entries()[1].bytes < 64 * 1024,
            "fixture must compress below admission size"
        );
        assert!(
            recording
                .source(0)
                .unwrap()
                .load_if_within_charge(64 * 1024)
                .unwrap()
                .is_some()
        );
        assert!(
            recording
                .source(1)
                .unwrap()
                .load_if_within_charge(64 * 1024)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            recording.source(1).unwrap().load().unwrap().array_bytes(),
            2 * 1024 * 1024
        );
        let source = recording.source(0).unwrap();
        let another = recording.source(1).unwrap();
        let cloned = source.clone();
        assert!(Arc::ptr_eq(&recording.manifest, &source.manifest));
        assert!(Arc::ptr_eq(&source.manifest, &another.manifest));
        assert!(Arc::ptr_eq(&source.manifest, &cloned.manifest));
        let file = path.join(&recording.entries()[0].file);
        let mut bytes = fs::read(&file).unwrap();
        bytes[0] ^= 1;
        fs::write(file, bytes).unwrap();
        assert!(
            source
                .load_if_within_charge(64 * 1024)
                .unwrap_err()
                .to_string()
                .contains("checksum")
        );
    }
}
