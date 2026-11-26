//! Headless index/selection/cache logic. No training or evaluation runs here.
use crate::work::DesiredWork;
use rl::{
    Error, Result,
    diagnostics::{schema::*, storage::*},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{
        Arc,
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
};

#[derive(Clone, Default)]
struct ModalityIndex {
    /// Metadata indices, ordered by (revision, sequence). Keep the unfiltered
    /// population separately so querying all sets requires no merge/allocation.
    all: Vec<usize>,
    by_probe: BTreeMap<String, Vec<usize>>,
}

fn insert_record(records: &mut Vec<usize>, entries: &[IndexEntry], index: usize) {
    let key = (entries[index].revision, entries[index].sequence);
    if records
        .last()
        .is_none_or(|&last| (entries[last].revision, entries[last].sequence) <= key)
    {
        records.push(index);
    } else {
        let position =
            records.partition_point(|&i| (entries[i].revision, entries[i].sequence) < key);
        records.insert(position, index);
    }
}

fn insert_revision(revisions: &mut Vec<u64>, revision: u64) {
    if revisions.last().is_none_or(|&last| last < revision) {
        revisions.push(revision);
    } else if let Err(position) = revisions.binary_search(&revision) {
        revisions.insert(position, revision);
    }
}

#[derive(Clone)]
pub struct RunIndex {
    recording: Recording,
    root: Arc<PathBuf>,
    revisions: Arc<Vec<u64>>,
    probe_sets: Arc<BTreeMap<String, Vec<usize>>>,
    // RecordKind is fieldless; its discriminant is only an internal map key.
    modalities: Arc<BTreeMap<u8, BTreeMap<String, ModalityIndex>>>,
    snapshots: Arc<BTreeMap<String, Vec<u64>>>,
}
impl RunIndex {
    pub fn open(path: impl AsRef<Path>, limits: ReadLimits) -> Result<Self> {
        let root = Arc::new(path.as_ref().canonicalize()?);
        let recording = Recording::open(root.as_ref(), limits)?;
        if matches!(
            recording.manifest().action_space,
            rl::core::action::ActionSpace::Conditional(_)
        ) && recording.manifest().typed_observation_space.is_none()
        {
            return Err(Error::Config(
                "viewer recordings require a numeric action space".into(),
            ));
        }
        let mut index = Self {
            recording,
            root,
            revisions: Arc::new(vec![]),
            probe_sets: Arc::new(BTreeMap::new()),
            modalities: Arc::new(BTreeMap::new()),
            snapshots: Arc::new(BTreeMap::new()),
        };
        index.index_new(0);
        Ok(index)
    }
    fn index_new(&mut self, start: usize) {
        let entries = self.recording.entries();
        for (i, entry) in entries.iter().enumerate().skip(start) {
            insert_revision(Arc::make_mut(&mut self.revisions), entry.revision);
            let modality = Arc::make_mut(&mut self.modalities)
                .entry(entry.kind as u8)
                .or_default()
                .entry(entry.model_role.clone())
                .or_default();
            insert_record(&mut modality.all, entries, i);
            if let Some(id) = &entry.probe_set {
                insert_record(modality.by_probe.entry(id.clone()).or_default(), entries, i);
            }
            if matches!(entry.kind, RecordKind::ProbeSet | RecordKind::TypedProbes)
                && let Some(id) = &entry.probe_set
            {
                insert_record(
                    Arc::make_mut(&mut self.probe_sets)
                        .entry(id.clone())
                        .or_default(),
                    entries,
                    i,
                );
            }
            if entry.kind == RecordKind::ModelSnapshot {
                insert_revision(
                    Arc::make_mut(&mut self.snapshots)
                        .entry(entry.model_role.clone())
                        .or_default(),
                    entry.revision,
                );
            }
        }
    }
    pub fn refresh(&mut self) -> Result<usize> {
        let before = self.recording.entries().len();
        let count = self.recording.refresh()?;
        if count > 0 {
            self.index_new(before);
        }
        Ok(count)
    }
    pub fn recording(&self) -> &Recording {
        &self.recording
    }
    pub fn revisions(&self) -> &[u64] {
        &self.revisions
    }
    pub(crate) fn record_roles(&self, kind: RecordKind) -> impl Iterator<Item = &str> {
        self.modalities
            .get(&(kind as u8))
            .into_iter()
            .flat_map(|roles| roles.keys())
            .map(String::as_str)
    }
    /// Recorded snapshot markers. Retention/file availability is checked separately
    /// by `has_model_for`, so old markers never imply available weights.
    pub fn snapshot_revisions(&self, role: &str) -> &[u64] {
        self.snapshots.get(role).map_or(&[], Vec::as_slice)
    }
    pub(crate) fn snapshot_roles(&self) -> impl Iterator<Item = &str> {
        self.snapshots.keys().map(String::as_str)
    }
    pub fn root(&self) -> &Arc<PathBuf> {
        &self.root
    }
    /// Resolve an exact immutable model source, never a predecessor or sibling.
    pub fn exact_snapshot_index(&self, key: &CacheKey, revision: u64, role: &str) -> Result<usize> {
        if key.root != self.root {
            return Err(Error::State("snapshot belongs to another recording".into()));
        }
        let index = self
            .by_sequence(key.sequence)
            .ok_or_else(|| Error::State("exact snapshot no longer indexed".into()))?;
        let entry = &self.recording.entries()[index];
        if entry.revision != revision
            || entry.model_role != role
            || entry.kind != RecordKind::ModelSnapshot
            || !self.has_model_record(index)
        {
            return Err(Error::State(
                "exact snapshot is unavailable or has a different role/revision".into(),
            ));
        }
        Ok(index)
    }
    /// Newest metadata candidates at or before the requested time, ordered by
    /// (revision, sequence). `None` includes every population, including records
    /// with no probe set. Lookup uses one binary predecessor search and the
    /// returned iterator borrows its index without allocating or loading chunks.
    pub fn candidates(
        &self,
        kind: RecordKind,
        role: &str,
        probe: Option<&str>,
        at_or_before: u64,
    ) -> impl Iterator<Item = usize> {
        let records = self.candidate_indices(kind, role, probe);
        let end =
            records.partition_point(|&i| self.recording.entries()[i].revision <= at_or_before);
        records[..end].iter().rev().copied()
    }
    /// Continue before an immutable record cursor, excluding that record and all
    /// newer candidates. The tuple predecessor search skips a same-revision
    /// prefix directly, so bounded resolver turns do not repeatedly scan it.
    pub fn candidates_before(
        &self,
        kind: RecordKind,
        role: &str,
        probe: Option<&str>,
        before: (u64, u64),
    ) -> impl Iterator<Item = usize> {
        let records = self.candidate_indices(kind, role, probe);
        let end = records.partition_point(|&i| {
            let entry = &self.recording.entries()[i];
            (entry.revision, entry.sequence) < before
        });
        records[..end].iter().rev().copied()
    }
    /// Eligible count and newest immutable tuple for this modality/population.
    /// The count detects late older appends even if the newest tuple is stable;
    /// unrelated records do not invalidate an in-progress predecessor search.
    pub fn candidate_frontier(
        &self,
        kind: RecordKind,
        role: &str,
        probe: Option<&str>,
        at_or_before: u64,
    ) -> (usize, Option<(u64, u64)>) {
        let records = self.candidate_indices(kind, role, probe);
        let end =
            records.partition_point(|&i| self.recording.entries()[i].revision <= at_or_before);
        let newest = records[..end].last().map(|&i| {
            let entry = &self.recording.entries()[i];
            (entry.revision, entry.sequence)
        });
        (end, newest)
    }
    fn candidate_indices(&self, kind: RecordKind, role: &str, probe: Option<&str>) -> &[usize] {
        self.modalities
            .get(&(kind as u8))
            .and_then(|roles| roles.get(role))
            .and_then(|modality| match probe {
                Some(id) => modality.by_probe.get(id),
                None => Some(&modality.all),
            })
            .map_or(&[], Vec::as_slice)
    }
    pub fn probe_set_ids(&self) -> impl Iterator<Item = &str> {
        self.probe_sets.keys().map(String::as_str)
    }
    pub fn probe_set(&self, id: &str, revision: u64) -> Option<usize> {
        let records = self.probe_sets.get(id)?;
        let end = records.partition_point(|&i| self.recording.entries()[i].revision <= revision);
        end.checked_sub(1).map(|last| records[last])
    }
    pub(crate) fn probe_set_of_kind(
        &self,
        id: &str,
        revision: u64,
        kind: RecordKind,
    ) -> Option<usize> {
        let entries = self.recording.entries();
        let records = self.probe_sets.get(id)?;
        let end = records.partition_point(|&index| entries[index].revision <= revision);
        records[..end]
            .iter()
            .rev()
            .copied()
            .find(|&index| entries[index].kind == kind)
    }
    /// Resolve a measurement against the set definition that already existed
    /// when it was recorded. A later append cannot redefine an earlier input,
    /// even when its learner revision is older or equal to the measurement.
    pub fn probe_for_record(&self, record: usize) -> Option<usize> {
        let entries = self.recording.entries();
        let entry = entries.get(record)?;
        let records = self.probe_sets.get(entry.probe_set.as_deref()?)?;
        let end = records.partition_point(|&i| entries[i].revision <= entry.revision);
        let source_kind = match entry.kind {
            RecordKind::FinitePredictions => Some(RecordKind::TypedProbes),
            RecordKind::Policy | RecordKind::Critic => Some(RecordKind::ProbeSet),
            _ => None,
        };
        records[..end].iter().rev().copied().find(|&i| {
            entries[i].sequence <= entry.sequence
                && source_kind.is_none_or(|kind| entries[i].kind == kind)
        })
    }
    /// Exact entry, including retention and artifact existence; siblings cannot substitute.
    pub fn has_model_record(&self, record: usize) -> bool {
        if self
            .recording
            .entries()
            .get(record)
            .is_none_or(|e| e.kind != RecordKind::ModelSnapshot)
        {
            return false;
        }
        self.recording.is_retained(record)
            && self
                .root
                .join(&self.recording.entries()[record].file)
                .is_file()
    }
    pub fn has_model_for(&self, revision: u64, role: &str) -> bool {
        self.candidates(RecordKind::ModelSnapshot, role, None, revision)
            .take_while(|&i| self.recording.entries()[i].revision == revision)
            .any(|i| self.has_model_record(i))
    }
    pub fn key(&self, index: usize) -> Result<CacheKey> {
        let entry = self
            .recording
            .entries()
            .get(index)
            .ok_or_else(|| Error::State("viewer record index out of range".into()))?;
        Ok(CacheKey {
            root: self.root.clone(),
            sequence: entry.sequence,
        })
    }
    pub fn by_sequence(&self, sequence: u64) -> Option<usize> {
        self.recording
            .entries()
            .binary_search_by_key(&sequence, |e| e.sequence)
            .ok()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SelectedInput {
    #[default]
    None,
    ProbeRow {
        set: String,
        row: usize,
        /// Immutable source record; moving training time never reinterprets its row.
        sequence: u64,
    },
    TrajectoryStep {
        sequence: u64,
        step: usize,
    },
}
#[derive(Debug, Clone)]
pub struct Selection {
    pub revision: u64,
    pub input: SelectedInput,
    pub model_role: String,
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            revision: 0,
            input: SelectedInput::None,
            model_role: "online".into(),
        }
    }
}
impl Selection {
    pub fn select_probe(&mut self, index: &RunIndex, set: &str, row: usize) {
        if let Some(i) = index.probe_set(set, self.revision) {
            self.input = SelectedInput::ProbeRow {
                set: set.into(),
                row,
                sequence: index.recording.entries()[i].sequence,
            };
        }
    }
    pub fn select_revision(&mut self, index: &RunIndex, revision: u64) -> Result<()> {
        if !index
            .revisions
            .first()
            .zip(index.revisions.last())
            .is_some_and(|(&first, &last)| (first..=last).contains(&revision))
        {
            return Err(Error::State(
                "revision is outside the recorded run range".into(),
            ));
        }
        self.revision = revision;
        // Environment time is independent. A trajectory from another revision
        // remains labelled by its own provenance, never silently reinterpreted.
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheKey {
    pub root: Arc<PathBuf>,
    pub sequence: u64,
}
struct Cached {
    chunk: Arc<Chunk>,
    charge: usize,
    touched: u64,
}
pub struct ChunkCache {
    entries: BTreeMap<CacheKey, Cached>,
    budget: usize,
    resident: usize,
    clock: u64,
}
impl ChunkCache {
    pub fn new(budget: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget,
            resident: 0,
            clock: 0,
        }
    }
    /// Conservative accounting of array contents, metadata and tree/shape
    /// overhead. This is a cache charge, not an OS resident-memory measurement.
    pub fn charge(chunk: &Chunk) -> usize {
        chunk.retained_charge()
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident
    }
    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<Chunk>> {
        let entry = self.entries.get_mut(key)?;
        self.clock = self.clock.wrapping_add(1);
        entry.touched = self.clock;
        Some(entry.chunk.clone())
    }
    pub fn insert(&mut self, key: CacheKey, chunk: Arc<Chunk>) -> bool {
        let charge = Self::charge(&chunk);
        if charge > self.budget {
            return false;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.resident -= old.charge;
        }
        self.evict_to(self.budget.saturating_sub(charge));
        self.clock = self.clock.wrapping_add(1);
        self.resident += charge;
        self.entries.insert(
            key,
            Cached {
                chunk,
                charge,
                touched: self.clock,
            },
        );
        true
    }
    fn evict_to(&mut self, limit: usize) {
        while self.resident > limit {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            self.resident -= self.entries.remove(&oldest).unwrap().charge;
        }
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.resident = 0;
    }
}
struct LoadRequest {
    ticket: LoadTicket,
    source: ChunkSource,
    #[cfg(feature = "screenshots")]
    timing: crate::native_timing::NativeTiming,
}
#[derive(Clone, PartialEq, Eq)]
struct LoadTicket {
    key: CacheKey,
    epoch: u64,
    generation: u64,
    max_charge: Option<usize>,
    #[cfg(feature = "screenshots")]
    read_id: u64,
}
struct Loaded {
    ticket: LoadTicket,
    result: std::result::Result<Chunk, String>,
}
/// One active read and a bounded replaceable metadata set. A pass collects all
/// independent modality dependencies before dispatch; responses only enter the
/// immutable raw cache, never a display frame.
pub struct Loader {
    pub cache: ChunkCache,
    wake: Arc<dyn Fn() + Send + Sync>,
    requests: SyncSender<LoadRequest>,
    results: Receiver<Loaded>,
    desired: DesiredWork,
    active: Option<LoadTicket>,
    in_pass: bool,
    managed_passes: bool,
    epoch: u64,
    errors: BTreeMap<CacheKey, String>,
    admission_errors: BTreeMap<(CacheKey, usize), String>,
    worker_error: Option<String>,
    pub completed_loads: u64,
    #[cfg(feature = "screenshots")]
    pub(crate) timing: crate::native_timing::NativeTiming,
    #[cfg(feature = "screenshots")]
    next_read_id: u64,
}
impl Loader {
    pub fn new(cache_budget: usize, wake: impl Fn() + Send + 'static) -> Result<Self> {
        Self::with_reader(cache_budget, wake, ChunkSource::load)
    }
    fn with_reader(
        cache_budget: usize,
        wake: impl Fn() + Send + 'static,
        mut read: impl FnMut(&ChunkSource) -> Result<Chunk> + Send + 'static,
    ) -> Result<Self> {
        // Keep the original Send-only callback contract while allowing other
        // bounded recording workers to use the same UI notification channel.
        let callback = std::sync::Mutex::new(wake);
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            if let Ok(callback) = callback.lock() {
                callback();
            }
        });
        let reader_wake = wake.clone();
        let (requests, receiver) = mpsc::sync_channel::<LoadRequest>(1);
        let (sender, results) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("rl-recording-reader".into())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    let ticket = request.ticket;
                    #[cfg(feature = "screenshots")]
                    request
                        .timing
                        .event("read_begin", ticket.read_id, || serde_json::json!({}));
                    let result = match ticket.max_charge {
                        Some(limit) => {
                            request
                                .source
                                .load_if_within_charge(limit)
                                .and_then(|chunk| {
                                    chunk.ok_or_else(|| {
                                        Error::Shape("record exceeds admitted decode budget".into())
                                    })
                                })
                        }
                        None => read(&request.source),
                    }
                    .map_err(|error| error.to_string());
                    #[cfg(feature = "screenshots")]
                    request.timing.event(
                        "read_end",
                        ticket.read_id,
                        || serde_json::json!({"ok":result.is_ok()}),
                    );
                    #[cfg(feature = "screenshots")]
                    let read_id = ticket.read_id;
                    #[cfg(feature = "screenshots")]
                    request
                        .timing
                        .event("publish_begin", read_id, || serde_json::json!({}));
                    if sender.send(Loaded { ticket, result }).is_err() {
                        break;
                    }
                    #[cfg(feature = "screenshots")]
                    request
                        .timing
                        .event("publish_end", read_id, || serde_json::json!({}));
                    reader_wake();
                }
            })?;
        Ok(Self {
            cache: ChunkCache::new(cache_budget),
            wake,
            requests,
            results,
            desired: DesiredWork::default(),
            active: None,
            in_pass: false,
            managed_passes: false,
            epoch: 0,
            errors: BTreeMap::new(),
            admission_errors: BTreeMap::new(),
            worker_error: None,
            completed_loads: 0,
            #[cfg(feature = "screenshots")]
            timing: Default::default(),
            #[cfg(feature = "screenshots")]
            next_read_id: 0,
        })
    }
    pub fn wake_callback(&self) -> Arc<dyn Fn() + Send + Sync> {
        self.wake.clone()
    }
    pub fn poll(&mut self) {
        while let Ok(loaded) = self.results.try_recv() {
            #[cfg(feature = "screenshots")]
            self.timing.event("consume", loaded.ticket.read_id, || serde_json::json!({"stale": loaded.ticket.epoch != self.epoch, "epoch":self.epoch,"queued":self.desired.len(),"completed":self.completed_loads}));
            if self.active.as_ref() != Some(&loaded.ticket) {
                continue;
            }
            self.active = None;
            self.completed_loads += 1;
            if loaded.ticket.epoch != self.epoch {
                continue;
            }
            let error = match loaded.result {
                Ok(chunk) => {
                    if self
                        .cache
                        .insert(loaded.ticket.key.clone(), Arc::new(chunk))
                    {
                        None
                    } else {
                        Some("record exceeds viewer cache budget; increase cache size".into())
                    }
                }
                Err(error) => Some(error),
            };
            if let Some(error) = error {
                if self.errors.len() >= 128 {
                    self.errors.pop_first();
                }
                if let Some(limit) = loaded.ticket.max_charge {
                    if self.admission_errors.len() >= 128 {
                        self.admission_errors.pop_first();
                    }
                    self.admission_errors
                        .insert((loaded.ticket.key, limit), error);
                } else {
                    self.errors.insert(loaded.ticket.key, error);
                }
            }
        }
        // Standalone users have no paint boundary. Managed UI passes
        // dispatch only after current user input and all dependencies are known.
        if !self.managed_passes
            && let Err(error) = self.dispatch()
        {
            self.worker_error = Some(error.to_string());
        }
    }
    pub fn request(&mut self, index: &RunIndex, record: usize) -> Result<Option<Arc<Chunk>>> {
        self.request_inner(index, record, None)
    }
    /// Environment-selected loads use codec admission before inflation. Existing
    /// ordinary consumers retain their original reader/error semantics.
    pub fn request_admitted(
        &mut self,
        index: &RunIndex,
        record: usize,
        max_charge: usize,
    ) -> Result<Option<Arc<Chunk>>> {
        self.request_inner(index, record, Some(max_charge))
    }
    fn request_inner(
        &mut self,
        index: &RunIndex,
        record: usize,
        max_charge: Option<usize>,
    ) -> Result<Option<Arc<Chunk>>> {
        if !index.recording.is_retained(record) {
            return Err(Error::State(
                "record was retired by the snapshot retention policy".into(),
            ));
        }
        let key = index.key(record)?;
        if let Some(limit) = max_charge
            && let Some(error) = self.admission_errors.get(&(key.clone(), limit))
        {
            return Err(Error::State(error.clone()));
        }
        if let Some(chunk) = self.cache.get(&key) {
            if max_charge.is_some_and(|limit| chunk.retained_charge() > limit) {
                return Err(Error::Shape(
                    "cached record exceeds admitted ownership budget".into(),
                ));
            }
            return Ok(Some(chunk));
        }
        if let Some(error) = self.errors.get(&key) {
            return Err(Error::State(error.clone()));
        }
        if let Some(error) = &self.worker_error {
            return Err(Error::State(error.clone()));
        }
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.epoch == self.epoch && active.key == key)
        {
            return Ok(None);
        }
        #[cfg(feature = "screenshots")]
        self.timing.event("desire", self.desired.generation, || serde_json::json!({"sequence":key.sequence,"root":key.root.as_path(),"kind":index.recording.entries()[record].kind,"revision":index.recording.entries()[record].revision}));
        if max_charge.is_some() {
            self.desired
                .offer_admitted(key, || index.recording.source(record), max_charge)?;
        } else {
            self.desired.offer(key, || index.recording.source(record))?;
        }
        if !self.in_pass {
            self.dispatch()?;
        }
        Ok(None)
    }
    pub fn pending(&self) -> usize {
        self.desired.len() + usize::from(self.active.is_some())
    }
    /// Replace the prior pass's queued desires; calls within this pass add to a
    /// shared set rather than replacing one another. The active read continues.
    pub fn begin_pass(&mut self) {
        self.in_pass = true;
        self.managed_passes = true;
        self.desired.begin_pass();
    }
    pub fn finish_pass(&mut self) -> Result<()> {
        self.in_pass = false;
        self.dispatch()
    }
    /// Cached failures are stable until an explicit retry or epoch invalidation.
    pub fn retry_failed(&mut self, key: &CacheKey) -> bool {
        self.errors.remove(key).is_some()
    }
    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.errors.clear();
        self.admission_errors.clear();
        self.desired.clear();
        self.epoch = self.epoch.wrapping_add(1);
    }
    fn dispatch(&mut self) -> Result<()> {
        if self.active.is_some() {
            return Ok(());
        }
        while let Some(read) = self.desired.pop_next() {
            if self.cache.entries.contains_key(&read.key) || self.errors.contains_key(&read.key) {
                continue;
            }
            #[cfg(feature = "screenshots")]
            {
                self.next_read_id += 1;
            }
            let ticket = LoadTicket {
                key: read.key,
                epoch: self.epoch,
                generation: self.desired.generation,
                max_charge: read.max_charge,
                #[cfg(feature = "screenshots")]
                read_id: self.next_read_id,
            };
            #[cfg(feature = "screenshots")]
            self.timing.event("dispatch", ticket.read_id, || serde_json::json!({"sequence":ticket.key.sequence,"root":ticket.key.root.as_path(),"epoch":ticket.epoch,"generation":ticket.generation,"queued":self.desired.len()}));
            match self.requests.try_send(LoadRequest {
                ticket: ticket.clone(),
                source: read.source,
                #[cfg(feature = "screenshots")]
                timing: self.timing.clone(),
            }) {
                Ok(()) => {
                    self.active = Some(ticket);
                    return Ok(());
                }
                Err(TrySendError::Full(request)) => {
                    self.desired.offer_admitted(
                        request.ticket.key,
                        || Ok(request.source),
                        request.ticket.max_charge,
                    )?;
                    return Ok(());
                }
                Err(TrySendError::Disconnected(_)) => {
                    return Err(Error::State("recording reader stopped".into()));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod loader_tests {
    use super::*;
    use rl::core::action::ActionSpace;
    use std::time::{Duration, Instant};

    fn fixture() -> (tempfile::TempDir, RunIndex) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("loader");
        let mut writer = Recorder::create(
            &path,
            RunManifest::new(
                "loader",
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
        for revision in 1..=3 {
            writer
                .append(&Chunk {
                    context: RecordContext::new("loader", revision),
                    record: Record::Metrics(MetricRecord {
                        values: BTreeMap::from([("loss".into(), revision as f64)]),
                        unavailable: BTreeMap::new(),
                    }),
                    arrays: BTreeMap::new(),
                })
                .unwrap();
        }
        writer.finish().unwrap();
        (dir, RunIndex::open(path, ReadLimits::default()).unwrap())
    }
    fn wait_for_read(loader: &mut Loader) {
        let before = loader.completed_loads;
        let deadline = Instant::now() + Duration::from_secs(3);
        while loader.completed_loads == before {
            loader.poll();
            assert!(Instant::now() < deadline, "blocked reader did not finish");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    fn blocked_loader() -> (Loader, Receiver<u64>, SyncSender<()>) {
        let (started, starts) = mpsc::sync_channel(4);
        let (release, released) = mpsc::sync_channel(1);
        let loader = Loader::with_reader(
            1024 * 1024,
            || {},
            move |source| {
                let chunk = source.load()?;
                let revision = chunk.context.identity.learner_update_revision;
                started.send(revision).unwrap();
                if revision == 1 {
                    released.recv().unwrap();
                }
                Ok(chunk)
            },
        )
        .unwrap();
        (loader, starts, release)
    }
    #[test]
    fn blocked_a_then_b_then_c_reads_c_next_and_only_c_can_satisfy_current_demand() {
        let (_dir, run) = fixture();
        let (mut loader, starts, release) = blocked_loader();
        loader.begin_pass();
        loader.request(&run, 0).unwrap();
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 1);
        loader.begin_pass();
        loader.request(&run, 1).unwrap();
        loader.finish_pass().unwrap();
        loader.begin_pass();
        loader.request(&run, 2).unwrap();
        loader.finish_pass().unwrap();
        release.send(()).unwrap();
        wait_for_read(&mut loader);
        assert!(
            loader.cache.get(&run.key(0).unwrap()).is_some(),
            "obsolete immutable results may still enter the raw cache"
        );
        loader.begin_pass();
        assert!(loader.request(&run, 2).unwrap().is_none());
        loader.finish_pass().unwrap();
        assert_eq!(
            starts.recv_timeout(Duration::from_secs(3)).unwrap(),
            3,
            "obsolete queued B must never start"
        );
        wait_for_read(&mut loader);
        assert!(loader.request(&run, 2).unwrap().is_some());
        assert!(loader.cache.get(&run.key(1).unwrap()).is_none());
    }
    #[test]
    fn current_desires_keep_fifo_order_across_reordered_passes() {
        let (_dir, run) = fixture();
        let (mut loader, starts, release) = blocked_loader();
        loader.begin_pass();
        for record in [0, 1, 2, 1] {
            loader.request(&run, record).unwrap();
        }
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 1);
        loader.begin_pass();
        for record in [2, 1, 0, 2] {
            loader.request(&run, record).unwrap();
        }
        loader.finish_pass().unwrap();
        assert_eq!(loader.pending(), 3);
        release.send(()).unwrap();
        wait_for_read(&mut loader);
        loader.begin_pass();
        for record in [2, 1] {
            loader.request(&run, record).unwrap();
        }
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 2);
        wait_for_read(&mut loader);
        assert_eq!(loader.completed_loads, 2);
        assert!(loader.cache.get(&run.key(2).unwrap()).is_none());
    }
    #[test]
    fn explicit_epoch_invalidation_discards_an_old_active_completion() {
        let (_dir, run) = fixture();
        let (mut loader, starts, release) = blocked_loader();
        loader.begin_pass();
        loader.request(&run, 0).unwrap();
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 1);
        loader.invalidate();
        loader.begin_pass();
        loader.request(&run, 2).unwrap();
        loader.finish_pass().unwrap();
        release.send(()).unwrap();
        wait_for_read(&mut loader);
        assert!(
            loader.cache.get(&run.key(0).unwrap()).is_none(),
            "reload/retention invalidation must not resurrect old weights"
        );
        loader.begin_pass();
        loader.request(&run, 2).unwrap();
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 3);
        wait_for_read(&mut loader);
    }
    #[cfg(feature = "screenshots")]
    #[test]
    fn timing_distinguishes_blocked_read_and_stale_ui_delivery_by_physical_id() {
        let (dir, run) = fixture();
        let path = dir.path().join("timing.jsonl");
        let (mut loader, starts, release) = blocked_loader();
        loader.timing = crate::native_timing::NativeTiming::open(&path, 100).unwrap();
        loader.begin_pass();
        loader.request(&run, 0).unwrap();
        loader.finish_pass().unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(3)).unwrap(), 1);
        let read_events = || -> Vec<serde_json::Value> {
            std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let blocked = read_events();
        assert!(
            blocked
                .iter()
                .any(|event| event["event"] == "read_begin" && event["id"] == 1)
        );
        assert!(!blocked.iter().any(|event| event["event"] == "read_end"));
        loader.invalidate();
        release.send(()).unwrap();
        wait_for_read(&mut loader);
        assert!(loader.cache.get(&run.key(0).unwrap()).is_none());
        let completed = read_events();
        let events: Vec<_> = completed.iter().filter(|event| event["id"] == 1).collect();
        let position = |name| {
            events
                .iter()
                .position(|event| event["event"] == name)
                .unwrap()
        };
        assert!(position("dispatch") < position("read_begin"));
        assert!(position("read_begin") < position("read_end"));
        assert!(position("read_end") < position("publish_begin"));
        assert!(position("publish_begin") < position("consume"));
        assert_eq!(events[position("consume")]["details"]["stale"], true);
    }
}
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub root: PathBuf,
    pub manifest: RunManifest,
}
/// Accept a single run or its parent directory. Broken manifests are reported
/// individually and do not prevent browsing healthy neighbors.
pub fn catalog(path: &Path) -> Result<(Vec<CatalogEntry>, Vec<String>)> {
    let roots = if path.join("manifest.json").is_file() {
        vec![path.to_owned()]
    } else {
        std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| path.is_dir() && path.join("manifest.json").is_file())
            .collect()
    };
    let mut entries = vec![];
    let mut errors = vec![];
    for root in roots {
        match read_manifest(&root, 1024 * 1024) {
            Ok(manifest) => entries.push(CatalogEntry { root, manifest }),
            Err(error) => {
                if !rl::diagnostics::policy_cache::is_policy_cache(&root) {
                    errors.push(format!("{}: {error}", root.display()));
                }
            }
        }
    }
    entries.sort_by(|a, b| a.manifest.run_id.cmp(&b.manifest.run_id));
    Ok((entries, errors))
}

/// Bounded, cancellable catalogue for interactive directory opening.
pub(crate) fn interactive_catalog(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<(Vec<CatalogEntry>, Vec<String>)> {
    const MAX_ENTRIES: usize = 4096;
    const MAX_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
    let mut entries = Vec::new();
    let mut errors = Vec::new();
    let mut remaining = MAX_MANIFEST_BYTES;
    fn add_root(
        root: PathBuf,
        entries: &mut Vec<CatalogEntry>,
        errors: &mut Vec<String>,
        remaining: &mut usize,
    ) -> bool {
        let size = match std::fs::metadata(root.join("manifest.json")) {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                errors.push(format!("{}: {error}", root.display()));
                return true;
            }
        };
        if size > 1024 * 1024 {
            errors.push(format!("{}: manifest exceeds 1 MiB", root.display()));
            return true;
        }
        if size > *remaining as u64 {
            errors.push(
                "Catalogue manifest budget reached; open a run directly to view another.".into(),
            );
            return false;
        }
        *remaining -= size as usize;
        match read_manifest(&root, size as usize) {
            Ok(manifest) => entries.push(CatalogEntry { root, manifest }),
            Err(error) if !rl::diagnostics::policy_cache::is_policy_cache(&root) => {
                errors.push(format!("{}: {error}", root.display()))
            }
            Err(_) => (),
        }
        true
    }
    if path.join("manifest.json").is_file() {
        add_root(path.to_owned(), &mut entries, &mut errors, &mut remaining);
    } else {
        for (index, entry) in std::fs::read_dir(path)?.take(MAX_ENTRIES + 1).enumerate() {
            if cancelled() {
                return Err(Error::State("Recording opening cancelled".into()));
            }
            if index == MAX_ENTRIES {
                errors.push(format!("Only the first {MAX_ENTRIES} directory entries were inspected; open a run directly to view another."));
                break;
            }
            if let Ok(entry) = entry {
                let root = entry.path();
                if root.is_dir()
                    && root.join("manifest.json").is_file()
                    && !add_root(root, &mut entries, &mut errors, &mut remaining)
                {
                    break;
                }
            }
        }
    }
    entries.sort_by(|a, b| a.manifest.run_id.cmp(&b.manifest.run_id));
    Ok((entries, errors))
}

#[cfg(test)]
#[test]
fn interactive_catalog_respects_entry_bound_and_cancellation() {
    let directory = tempfile::tempdir().unwrap();
    for index in 0..4097 {
        std::fs::write(directory.path().join(format!("entry-{index}")), []).unwrap();
    }
    let (entries, notices) = interactive_catalog(directory.path(), &|| false).unwrap();
    assert!(entries.is_empty());
    assert!(
        notices
            .iter()
            .any(|notice| notice.contains("4,096") || notice.contains("4096"))
    );
    assert!(interactive_catalog(directory.path(), &|| true).is_err());
}

#[cfg(test)]
#[test]
fn interactive_catalog_bounds_aggregate_manifest_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = RunManifest::new(
        "run",
        "experiment",
        "algorithm",
        1,
        rl::core::action::ActionSpace::Discrete { count: 2 },
        0.9,
    )
    .unwrap();
    manifest
        .metadata
        .insert("large".into(), "x".repeat(900_000));
    let contents = serde_json::to_vec(&manifest).unwrap();
    for index in 0..6 {
        let root = directory.path().join(format!("run-{index}"));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("manifest.json"), &contents).unwrap();
    }
    let (entries, notices) = interactive_catalog(directory.path(), &|| false).unwrap();
    assert!(entries.len() < 6);
    assert!(
        notices
            .iter()
            .any(|notice| notice.contains("manifest budget"))
    );
}

/// Headless integrity and environment-plot validation, including every chunk.
pub fn validate_directory(path: &Path) -> Result<String> {
    use rl::diagnostics::typed::{DiagnosticInput, TypedPayload};
    let (entries, errors) = catalog(path)?;
    if !errors.is_empty() {
        return Err(Error::State(errors.join("\n")));
    }
    if entries.is_empty() {
        return Err(Error::State("no recordings found".into()));
    }
    let mut report = String::new();
    for entry in entries {
        let run = RunIndex::open(&entry.root, ReadLimits::default())?;
        let recording = run.recording();
        if !recording.issues().is_empty() {
            return Err(Error::State(recording.issues().join("\n")));
        }
        let mut bytes = 0;
        let mut typed_probes = None;
        for i in 0..recording.entries().len() {
            if !recording.is_retained(i) {
                continue;
            }
            let chunk = recording.load(i)?;
            bytes += chunk.array_bytes();
            if chunk.record.kind() == RecordKind::FinitePredictions {
                let index = run
                    .probe_for_record(i)
                    .filter(|&index| recording.is_retained(index))
                    .ok_or_else(|| {
                        Error::Shape("finite predictions have no retained typed input".into())
                    })?;
                if typed_probes
                    .as_ref()
                    .is_none_or(|(cached, _)| index != *cached)
                {
                    let TypedPayload::Probes(probes) = TypedPayload::read(&recording.load(index)?)?
                    else {
                        return Err(Error::Shape(
                            "finite predictions have the wrong input kind".into(),
                        ));
                    };
                    typed_probes = Some((index, probes));
                }
                let (_, probes) = typed_probes.as_ref().expect("matched probe loaded");
                let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunk)? else {
                    return Err(Error::Shape("expected finite predictions".into()));
                };
                predictions.validate_support(
                    &probes.row_ids,
                    probes.inputs.iter().map(DiagnosticInput::observation),
                    &recording.manifest().action_space,
                )?;
            }
            if let Record::Trajectory(t) = &chunk.record {
                let a = crate::plot_data::array(&chunk, &t.observations)?;
                let times = t
                    .simulation_times
                    .as_ref()
                    .map(|k| crate::plot_data::array(&chunk, k).map(Array::to_f64))
                    .transpose()?;
                crate::plot_data::trajectory_points(a, times.as_deref(), 0)?;
            }
        }
        use std::fmt::Write;
        let _ = writeln!(
            report,
            "{}: {} indexed chunks, {} retained numeric bytes, complete={}",
            entry.manifest.run_id,
            recording.entries().len(),
            bytes,
            recording.complete()
        );
    }
    Ok(report)
}

#[cfg(test)]
mod admitted_loader_tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::{Duration, Instant},
    };
    #[test]
    fn narrow_admission_bypasses_unbounded_reader_and_does_not_poison_ordinary_reads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run");
        let manifest = RunManifest::new(
            "admission",
            "test",
            "test",
            1,
            rl::core::action::ActionSpace::Discrete { count: 1 },
            0.9,
        )
        .unwrap();
        let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
        writer
            .append(&Chunk {
                context: RecordContext::new("admission", 0),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("value".into(), 1.)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(path, ReadLimits::default()).unwrap();
        let reads = Arc::new(AtomicUsize::new(0));
        let count = reads.clone();
        let mut loader = Loader::with_reader(
            1024 * 1024,
            || {},
            move |source| {
                count.fetch_add(1, Ordering::AcqRel);
                source.load()
            },
        )
        .unwrap();
        assert!(loader.request_admitted(&run, 0, 1).unwrap().is_none());
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            loader.poll();
            if loader.request_admitted(&run, 0, 1).is_err() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(reads.load(Ordering::Acquire), 0);
        loop {
            loader.poll();
            if loader.request(&run, 0).unwrap().is_some() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(reads.load(Ordering::Acquire), 1);
    }
}
