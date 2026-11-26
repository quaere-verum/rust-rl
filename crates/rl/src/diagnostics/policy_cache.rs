//! Versioned numerical cache, separate from immutable training recordings.
//! Readers authenticate manifest and exact index provenance without weights.
mod payload;

use super::{
    codec,
    policy_slice::PolicySliceSpec,
    policy_surface::{POLICY_NUMERICAL_CONVENTION, POLICY_SURFACE_VERSION, PolicySurface},
    schema::{Chunk, RecordKind, RunManifest, Values},
    storage::{IndexEntry, Recording},
    visualization::EnvironmentSemantics,
};
use crate::{Result, core::ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub const POLICY_CACHE_VERSION: u32 = 1;
#[derive(Debug, Clone, Copy, Default)]
pub struct CacheLoadTiming {
    /// Metadata validation, encoded payload read and SHA-256 integrity.
    pub read_micros: u64,
    /// Decompression, typed array construction and numerical contract validation.
    pub decode_micros: u64,
}
/// Classify a valid sidecar when browsing a parent of recordings. Cache
/// manifests are independent artifacts and must not be reported as corrupt
/// training recordings merely because both formats use manifest.json.
pub fn is_policy_cache(root: impl AsRef<Path>) -> bool {
    let Ok(bytes) = read_limited(&root.as_ref().join("manifest.json"), 1024 * 1024) else {
        return false;
    };
    let Ok(manifest) = serde_json::from_slice::<CacheManifest>(&bytes) else {
        return false;
    };
    manifest.version == POLICY_CACHE_VERSION
        && manifest.surface_version == POLICY_SURFACE_VERSION
        && manifest.numerical_convention == POLICY_NUMERICAL_CONVENTION
        && hash_json(&manifest.recording, 1024 * 1024)
            .is_ok_and(|hash| hash == manifest.recording_sha256)
}
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
pub struct CacheLimits {
    pub max_metadata_bytes: usize,
    /// Conservative decoded catalog allowance, separate from frame data.
    pub max_index_bytes: usize,
    pub max_entries: usize,
    pub max_frame_bytes: usize,
    pub max_chunk_bytes: usize,
    pub max_surface_bytes: usize,
}
impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_metadata_bytes: 1024 * 1024,
            max_index_bytes: 128 * 1024 * 1024,
            max_entries: 1_000_000,
            max_frame_bytes: 256 * 1024 * 1024,
            max_chunk_bytes: 64 * 1024 * 1024,
            max_surface_bytes: 128 * 1024 * 1024,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CacheManifest {
    version: u32,
    surface_version: u32,
    numerical_convention: String,
    recording_sha256: String,
    recording: RunManifest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheEntry {
    /// Hash of this complete metadata object with this field empty.
    pub metadata_sha256: String,
    pub key: String,
    pub version: u32,
    pub surface_version: u32,
    pub numerical_convention: String,
    pub recording_sha256: String,
    pub source_sha256: String,
    /// Content fingerprint of the actual restored snapshot's complete arrays
    /// and metadata, in addition to its immutable index identity.
    pub model_content_sha256: String,
    pub source: IndexEntry,
    pub slice_sha256: String,
    pub slice: PolicySliceSpec,
    pub rows: usize,
    pub batches: usize,
    pub retained_bytes: usize,
    pub payload_bytes: u64,
    pub payload_sha256: String,
}
#[derive(Debug, Default)]
pub struct CacheCatalog {
    pub entries: Vec<CacheEntry>,
    pub issues: Vec<String>,
}
#[derive(Clone)]
pub struct PolicyCache {
    root: PathBuf,
    recording: Recording,
    fingerprint: String,
    limits: CacheLimits,
}
pub(super) fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(super) fn hash_json(value: &impl Serialize, limit: usize) -> Result<String> {
    Ok(hash_bytes(&codec::bounded_json(value, limit)?))
}
fn valid_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
pub(super) fn read_limited(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    ensure(size <= limit as u64, "cache file exceeds read limit")?;
    let mut bytes = vec![0; size as usize];
    file.read_exact(&mut bytes)?;
    ensure(
        file.read(&mut [0u8; 1])? == 0,
        "cache file grew beyond read limit",
    )?;
    Ok(bytes)
}
fn temp_path(root: &Path, prefix: &str) -> PathBuf {
    root.join(format!(
        "{prefix}.partial.{}.{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
/// Hash original bits incrementally; never materialize another full weight pack.
pub fn model_content_hash(chunk: &Chunk, metadata_limit: usize) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"rust-rl/policy-source/1");
    hash.update(codec::bounded_json(
        &(&chunk.context, &chunk.record),
        metadata_limit,
    )?);
    for (name, array) in &chunk.arrays {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(codec::bounded_json(
            &(array.dtype(), array.shape()),
            metadata_limit,
        )?);
        match &array.values {
            Values::F32(v) => {
                for x in v {
                    hash.update(x.to_le_bytes());
                }
            }
            Values::F64(v) => {
                for x in v {
                    hash.update(x.to_le_bytes());
                }
            }
            Values::U64(v) => {
                for x in v {
                    hash.update(x.to_le_bytes());
                }
            }
            Values::U8(v) => hash.update(v),
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
impl PolicyCache {
    fn metadata_hash(&self, entry: &CacheEntry) -> Result<String> {
        let mut unsigned = entry.clone();
        unsigned.metadata_sha256.clear();
        hash_json(&unsigned, self.limits.max_metadata_bytes)
    }
    /// Read-only handle for dry-run planning. An absent cache stays absent.
    pub fn open_or_empty(
        root: impl AsRef<Path>,
        recording: &Recording,
        limits: CacheLimits,
    ) -> Result<Self> {
        if root.as_ref().join("manifest.json").exists() {
            return Self::open(root, recording, limits);
        }
        Ok(Self {
            root: root.as_ref().to_owned(),
            recording: recording.clone(),
            fingerprint: hash_json(recording.manifest(), limits.max_metadata_bytes)?,
            limits,
        })
    }
    pub fn open_or_create(
        root: impl AsRef<Path>,
        recording: &Recording,
        limits: CacheLimits,
    ) -> Result<Self> {
        let root = root.as_ref();
        if root.join("manifest.json").exists() {
            return Self::open(root, recording, limits);
        }
        fs::create_dir_all(root.join("frames"))?;
        let manifest = CacheManifest {
            version: POLICY_CACHE_VERSION,
            surface_version: POLICY_SURFACE_VERSION,
            numerical_convention: POLICY_NUMERICAL_CONVENTION.into(),
            recording_sha256: hash_json(recording.manifest(), limits.max_metadata_bytes)?,
            recording: recording.manifest().clone(),
        };
        let bytes = codec::bounded_json(&manifest, limits.max_metadata_bytes)?;
        let temp = temp_path(root, "manifest");
        write_new(&temp, &bytes)?;
        // Linking a completed, synced temporary file is an atomic no-clobber
        // publication on supported filesystems, including Windows NTFS.
        let published = fs::hard_link(&temp, root.join("manifest.json"));
        fs::remove_file(&temp)?;
        match published {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
        Self::open(root, recording, limits)
    }
    pub fn open(
        root: impl AsRef<Path>,
        recording: &Recording,
        limits: CacheLimits,
    ) -> Result<Self> {
        let root = root.as_ref().to_owned();
        let manifest: CacheManifest = serde_json::from_slice(&read_limited(
            &root.join("manifest.json"),
            limits.max_metadata_bytes,
        )?)?;
        let fingerprint = hash_json(recording.manifest(), limits.max_metadata_bytes)?;
        ensure(
            manifest.version == POLICY_CACHE_VERSION
                && manifest.surface_version == POLICY_SURFACE_VERSION
                && manifest.numerical_convention == POLICY_NUMERICAL_CONVENTION,
            "incompatible policy cache version or numerical convention",
        )?;
        ensure(
            manifest.recording_sha256 == fingerprint
                && hash_json(&manifest.recording, limits.max_metadata_bytes)? == fingerprint,
            "policy cache recording/encoding identity differs",
        )?;
        Ok(Self {
            root,
            recording: recording.clone(),
            fingerprint,
            limits,
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn limits(&self) -> CacheLimits {
        self.limits
    }
    fn source_hash(&self, source: &IndexEntry) -> Result<String> {
        ensure(
            source.kind == RecordKind::ModelSnapshot,
            "cache source is not a checkpoint",
        )?;
        let i = self
            .recording
            .entries()
            .binary_search_by_key(&source.sequence, |e| e.sequence)
            .map_err(|_| {
                crate::Error::State(
                    "cache checkpoint provenance absent from recording index".into(),
                )
            })?;
        let hash = hash_json(source, self.limits.max_metadata_bytes)?;
        ensure(
            hash == hash_json(&self.recording.entries()[i], self.limits.max_metadata_bytes)?,
            "cache checkpoint source identity differs",
        )?;
        Ok(hash)
    }
    pub fn key(&self, source: &IndexEntry, slice: &PolicySliceSpec) -> Result<String> {
        let source_hash = self.source_hash(source)?;
        let slice_hash = hash_json(slice, self.limits.max_metadata_bytes)?;
        hash_json(
            &(
                POLICY_CACHE_VERSION,
                POLICY_SURFACE_VERSION,
                POLICY_NUMERICAL_CONVENTION,
                &self.fingerprint,
                source_hash,
                slice_hash,
            ),
            self.limits.max_metadata_bytes,
        )
    }
    fn validate_entry(&self, entry: &CacheEntry) -> Result<()> {
        ensure(
            entry.metadata_sha256 == self.metadata_hash(entry)?,
            "policy cache metadata checksum mismatch",
        )?;
        ensure(
            entry.version == POLICY_CACHE_VERSION
                && entry.surface_version == POLICY_SURFACE_VERSION
                && entry.numerical_convention == POLICY_NUMERICAL_CONVENTION
                && entry.recording_sha256 == self.fingerprint,
            "incompatible policy cache entry",
        )?;
        ensure(
            entry.source_sha256 == self.source_hash(&entry.source)?
                && entry.slice_sha256 == hash_json(&entry.slice, self.limits.max_metadata_bytes)?
                && entry.key == self.key(&entry.source, &entry.slice)?,
            "policy cache entry identity mismatch",
        )?;
        let config = entry
            .slice
            .to_config(EnvironmentSemantics::new(self.recording.manifest()))?;
        ensure(
            entry.source.model_role == entry.slice.model_role
                && config.len()? == entry.rows
                && entry.batches > 0
                && entry.batches <= entry.rows
                && valid_hash(&entry.model_content_sha256)
                && valid_hash(&entry.payload_sha256),
            "invalid cache slice/source/numerical metadata",
        )?;
        ensure(
            entry.payload_bytes <= self.limits.max_frame_bytes as u64,
            "cache payload exceeds encoded frame limit",
        )?;
        let file = self
            .root
            .join("frames")
            .join(&entry.key)
            .join("payload.bin");
        ensure(
            fs::metadata(file)?.len() == entry.payload_bytes,
            "missing or truncated cache payload",
        )
    }
    fn read_entry(&self, key: &str) -> Result<CacheEntry> {
        ensure(valid_hash(key), "invalid policy cache key")?;
        let entry: CacheEntry = serde_json::from_slice(&read_limited(
            &self.root.join("frames").join(key).join("entry.json"),
            self.limits.max_metadata_bytes,
        )?)?;
        ensure(entry.key == key, "cache directory identity differs")?;
        self.validate_entry(&entry)?;
        Ok(entry)
    }
    pub fn find(&self, source: &IndexEntry, slice: &PolicySliceSpec) -> Result<Option<CacheEntry>> {
        let key = self.key(source, slice)?;
        if !self.root.join("frames").join(&key).exists() {
            return Ok(None);
        }
        self.read_entry(&key).map(Some)
    }
    /// Metadata only: no model restore and no numerical frame materialization.
    /// Payload integrity is certified by `load`/`verify` on a background reader.
    pub fn catalog(&self) -> Result<CacheCatalog> {
        let mut catalog = CacheCatalog::default();
        let mut charge = 0usize;
        for (count, item) in fs::read_dir(self.root.join("frames"))?.enumerate() {
            ensure(
                count < self.limits.max_entries,
                "policy cache catalog entry limit exceeded",
            )?;
            let item = item?;
            let name = item.file_name().to_string_lossy().into_owned();
            if !valid_hash(&name) {
                continue;
            } // Interrupted temporary directories.
            let result = self.read_entry(&name);
            charge = charge
                .saturating_add(
                    fs::metadata(item.path().join("entry.json"))
                        .map_or(1024, |m| m.len() as usize)
                        .saturating_mul(8),
                )
                .saturating_add(1024);
            ensure(
                charge <= self.limits.max_index_bytes,
                "policy cache catalog metadata exceeds byte limit",
            )?;
            match result {
                Ok(entry) => catalog.entries.push(entry),
                Err(error) => catalog.issues.push(format!("{name}: {error}")),
            }
        }
        catalog.entries.sort_by(|a, b| {
            a.slice_sha256
                .cmp(&b.slice_sha256)
                .then(a.source.sequence.cmp(&b.source.sequence))
        });
        Ok(catalog)
    }
    pub fn load(&self, entry: &CacheEntry) -> Result<PolicySurface> {
        self.load_profiled(entry).map(|(surface, _)| surface)
    }
    pub fn load_profiled(&self, entry: &CacheEntry) -> Result<(PolicySurface, CacheLoadTiming)> {
        let started = std::time::Instant::now();
        let current = self.read_entry(&entry.key)?;
        ensure(
            hash_json(&current, self.limits.max_metadata_bytes)?
                == hash_json(entry, self.limits.max_metadata_bytes)?,
            "cache entry changed after indexing",
        )?;
        payload::load(
            &self
                .root
                .join("frames")
                .join(&entry.key)
                .join("payload.bin"),
            entry,
            self.recording.manifest(),
            self.limits,
            started,
        )
    }
    /// Validate existing numerical data without retaining a frame in the caller.
    pub fn verify(&self, entry: &CacheEntry) -> Result<()> {
        self.load(entry).map(drop)
    }
    /// Streaming integrity admission for large catalogs. No decompression,
    /// numerical frame materialization or checkpoint access is performed.
    pub fn verify_payload(&self, entry: &CacheEntry) -> Result<()> {
        let current = self.read_entry(&entry.key)?;
        ensure(
            current.metadata_sha256 == entry.metadata_sha256,
            "cache entry changed after indexing",
        )?;
        let mut file = File::open(
            self.root
                .join("frames")
                .join(&entry.key)
                .join("payload.bin"),
        )?;
        let mut hash = Sha256::new();
        let mut bytes = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            bytes = bytes.saturating_add(n as u64);
            ensure(
                bytes <= entry.payload_bytes,
                "cache payload grew beyond indexed size",
            )?;
            hash.update(&buffer[..n]);
        }
        ensure(
            bytes == entry.payload_bytes
                && format!("{:x}", hash.finalize()) == entry.payload_sha256,
            "policy cache payload checksum mismatch",
        )
    }
    pub fn publish(
        &self,
        source: &IndexEntry,
        model_hash: &str,
        surface: &PolicySurface,
    ) -> Result<CacheEntry> {
        let key = self.key(source, surface.spec())?;
        ensure(valid_hash(model_hash), "invalid model content fingerprint")?;
        if let Some(entry) = self.find(source, surface.spec())? {
            ensure(
                entry.model_content_sha256 == model_hash,
                "existing cache model content provenance disagrees",
            )?;
            self.verify(&entry)?;
            return Ok(entry);
        }
        ensure(valid_hash(model_hash), "invalid model content fingerprint")?;
        payload::validate(
            surface,
            source,
            self.recording.manifest(),
            self.limits.max_surface_bytes,
        )?;
        let temp = temp_path(&self.root.join("frames"), &key);
        fs::create_dir(&temp)?;
        let result = (|| {
            let (bytes, checksum) =
                payload::write(&temp.join("payload.bin"), surface, self.limits)?;
            let mut entry = CacheEntry {
                metadata_sha256: String::new(),
                key: key.clone(),
                version: POLICY_CACHE_VERSION,
                surface_version: POLICY_SURFACE_VERSION,
                numerical_convention: POLICY_NUMERICAL_CONVENTION.into(),
                recording_sha256: self.fingerprint.clone(),
                source_sha256: self.source_hash(source)?,
                model_content_sha256: model_hash.into(),
                source: source.clone(),
                slice_sha256: hash_json(surface.spec(), self.limits.max_metadata_bytes)?,
                slice: surface.spec().clone(),
                rows: surface.rows(),
                batches: surface.batches.len(),
                retained_bytes: surface.owned_bytes(),
                payload_bytes: bytes,
                payload_sha256: checksum,
            };
            entry.metadata_sha256 = self.metadata_hash(&entry)?;
            write_new(
                &temp.join("entry.json"),
                &codec::bounded_json(&entry, self.limits.max_metadata_bytes)?,
            )?;
            match fs::rename(&temp, self.root.join("frames").join(&key)) {
                Ok(()) => Ok(entry),
                Err(e) if self.root.join("frames").join(&key).exists() => {
                    let existing = self.read_entry(&key)?;
                    ensure(
                        existing.payload_sha256 == entry.payload_sha256
                            && existing.model_content_sha256 == entry.model_content_sha256,
                        "concurrent cache publication disagrees",
                    )?;
                    self.verify(&existing)?;
                    let _ = e;
                    Ok(existing)
                }
                Err(e) => Err(e.into()),
            }
        })();
        // Only the exact temporary directory created by this call is removed.
        if temp.exists() {
            fs::remove_dir_all(&temp)?;
        }
        result
    }
}
