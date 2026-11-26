//! Single-active/latest-desired summaries; all raw reads occur on this worker.
use super::policy_space::{PolicySpaceProjection, PolicySpaceSamples};
use super::training::{TrainingFrame, TrainingKey, TrainingRequest};
use super::{
    ChangeKind, ErrorKind, MAX_READ_BYTES, evaluation_summary, independent_critic_error,
    policy_change,
};
use crate::data::{CacheKey, RunIndex};
use rl::{
    Error, Result,
    diagnostics::{
        representations::{equal_row, match_reference_sets},
        schema::*,
        storage::{ChunkSource, DecodedChunkCache, IndexEntry},
        value::{self, ValueCriticKind},
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

pub const MAX_MEASUREMENT_SOURCES: usize = 512;
pub const MAX_SUMMARY_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_OWNER_BYTES: usize = 64 * 1024 * 1024;
const REQUEST_BYTES: usize = 2 * 1024 * 1024;
// Six simultaneously borrowed decoded chunks (24 MiB), decoder scratch (20
// MiB), temporary statistics/maps and the building summary (4 MiB). Descriptor
// leases and every published/external summary are charged independently.
const WORK_BYTES: usize = 48 * 1024 * 1024;
const MAX_TOTAL_READ_BYTES: usize = 128 * 1024 * 1024;
const MAX_READS: usize = 4096;
const KINDS: [RecordKind; 3] = [
    RecordKind::Empirical,
    RecordKind::Policy,
    RecordKind::Critic,
];

pub(super) fn checked(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Config(reason.into()))
    }
}
fn short(s: &str) -> String {
    s.chars().take(256).collect()
}
fn message(error: impl std::fmt::Display) -> String {
    short(&error.to_string())
}

#[derive(Clone, Debug)]
pub struct LearningPoint {
    pub revision: u64,
    pub sequence: Option<u64>,
    pub value: Option<f64>,
    pub count: usize,
    pub standard_error: Option<f64>,
    pub conditions: usize,
    pub critic_name: Option<String>,
    pub continuation: Option<String>,
    pub condition_counts: Vec<usize>,
    pub label: String,
    pub reason: Option<String>,
}
#[derive(Clone, Debug)]
pub struct PolicyAnchor {
    pub revision: u64,
    pub sequence: u64,
    pub probe_sequence: u64,
    pub rows: usize,
    pub population: String,
    pub kind: ChangeKind,
    pub normalization: Vec<f64>,
}
/// The score always describes this named critic's declared current-revision
/// continuation, on these complete, immutable measured conditions.
#[derive(Clone, Debug)]
pub struct CriticAnchor {
    pub revision: u64,
    pub sequence: u64,
    pub probe_sequence: u64,
    pub critic_name: String,
    pub kind: ValueCriticKind,
    pub score_kind: ErrorKind,
    pub action_condition: ActionCondition,
    pub stochastic: bool,
    pub row_ids: Vec<u64>,
    pub observation_methods: Vec<String>,
    pub population: String,
    pub initial_continuation: String,
}
impl CriticAnchor {
    fn charge(&self) -> usize {
        4096 + self.critic_name.capacity()
            + self.population.capacity()
            + self.initial_continuation.capacity()
            + self.row_ids.capacity() * 8
            + self.observation_methods.capacity() * std::mem::size_of::<String>()
            + self
                .observation_methods
                .iter()
                .map(String::capacity)
                .sum::<usize>()
    }
}
pub struct LearningFrame {
    pub root: Arc<PathBuf>,
    pub role: String,
    pub policy_population: Option<String>,
    pub evaluations: Vec<LearningPoint>,
    pub policy_changes: Vec<LearningPoint>,
    pub policy_space: Option<PolicySpaceProjection>,
    pub policy_space_unavailable: Option<String>,
    pub critic_errors: Vec<LearningPoint>,
    pub anchor: Option<PolicyAnchor>,
    pub critic_anchor: Option<CriticAnchor>,
    pub omitted_sources: usize,
    pub retained_bytes: usize,
    owner: Arc<RequestOwner>,
    anchor_source: Option<Source>,
    critic_anchor_source: Option<Source>,
    unavailable_sources: BTreeSet<u64>,
    _lease: Lease,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct LearningMetrics {
    pub submitted: u64,
    pub completed: u64,
    pub reads: u64,
    /// Physical decoder invocations, distinct from logical read allowances.
    pub decodes: u64,
    pub owned_bytes: usize,
}

#[derive(Clone, PartialEq, Eq)]
struct Key {
    root: Arc<PathBuf>,
    role: String,
    population: Option<String>,
    frontiers: [(usize, Option<(u64, u64)>); 3],
}
impl Key {
    fn new(run: &RunIndex, role: &str, population: Option<&str>) -> Result<Self> {
        checked(
            role.len() <= 256
                && run.root().as_os_str().len() <= 2048
                && population.is_none_or(|set| set.len() <= 1024),
            "Learning recording/role metadata exceeds admission",
        )?;
        Ok(Self {
            root: run.root().clone(),
            role: role.into(),
            population: population.map(str::to_owned),
            frontiers: KINDS.map(|kind| {
                let set = if kind == RecordKind::Policy {
                    population
                } else {
                    None
                };
                run.candidate_frontier(kind, role, set, u64::MAX)
            }),
        })
    }
}
#[derive(Clone)]
struct Source {
    key: CacheKey,
    revision: u64,
    kind: RecordKind,
    probe_id: Option<String>,
    source: ChunkSource,
    probe: Option<ChunkSource>,
    probe_sequence: Option<u64>,
    charge: usize,
}
impl Source {
    fn from_run(run: &RunIndex, index: usize) -> Result<Self> {
        let entry = &run.recording().entries()[index];
        let probe = run.probe_for_record(index);
        for i in [Some(index), probe].into_iter().flatten() {
            let e = &run.recording().entries()[i];
            checked(
                entry_bytes(e) <= 8192,
                "Learning source metadata exceeds bounded descriptor admission",
            )?;
        }
        let charge = 1024
            + 4 * (entry_bytes(entry)
                + probe.map_or(0, |i| entry_bytes(&run.recording().entries()[i]))
                + 2 * run.root().as_os_str().len());
        Ok(Self {
            key: run.key(index)?,
            revision: entry.revision,
            kind: entry.kind,
            probe_id: entry.probe_set.clone(),
            source: run.recording().source(index)?,
            probe: probe.map(|i| run.recording().source(i)).transpose()?,
            probe_sequence: probe.map(|i| run.recording().entries()[i].sequence),
            charge,
        })
    }
    fn charge(&self) -> usize {
        self.charge
    }
    fn check(&self) -> Result<()> {
        self.source.check_available()?;
        if let Some(probe) = &self.probe {
            probe.check_available()?;
        }
        Ok(())
    }
}
pub(super) fn entry_bytes(e: &IndexEntry) -> usize {
    [&e.file, &e.model_role]
        .into_iter()
        .map(|s| s.len())
        .sum::<usize>()
        + [&e.probe_set, &e.trajectory, &e.checkpoint_id]
            .into_iter()
            .flatten()
            .map(|s| s.len())
            .sum::<usize>()
}
struct Counter(usize);
impl Write for Counter {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(b.len());
        if self.0 > 128 * 1024 {
            Err(std::io::Error::other(
                "Learning manifest exceeds metadata admission",
            ))
        } else {
            Ok(b.len())
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
/// Immutable source/frontier and explicit frozen comparison identity. No models.
pub struct LearningRequest {
    key: Key,
    sources: Vec<Source>,
    manifest: RunManifest,
    pinned: Option<Source>,
    pinned_summary: Option<PolicyAnchor>,
    pinned_critic: Option<Source>,
    pinned_critic_summary: Option<CriticAnchor>,
    omitted: usize,
}
impl LearningRequest {
    fn new(run: &RunIndex, key: Key, previous: Option<&LearningFrame>) -> Result<Self> {
        let mut count = Counter(0);
        serde_json::to_writer(&mut count, run.recording().manifest())?;
        let mut candidates = Vec::with_capacity(3 * MAX_MEASUREMENT_SOURCES);
        for kind in KINDS {
            let set = if kind == RecordKind::Policy {
                key.population.as_deref()
            } else {
                None
            };
            candidates.extend(
                run.candidates(kind, &key.role, set, u64::MAX)
                    .take(MAX_MEASUREMENT_SOURCES),
            );
        }
        candidates.sort_unstable_by_key(|&i| {
            let e = &run.recording().entries()[i];
            (e.revision, e.sequence)
        });
        if candidates.len() > MAX_MEASUREMENT_SOURCES {
            candidates.drain(..candidates.len() - MAX_MEASUREMENT_SOURCES);
        }
        let pinned = previous
            .filter(|f| {
                f.root == key.root && f.role == key.role && f.policy_population == key.population
            })
            .and_then(|f| f.anchor_source.clone());
        let pinned_summary = previous
            .filter(|f| {
                f.root == key.root && f.role == key.role && f.policy_population == key.population
            })
            .and_then(|f| f.anchor.clone());
        let pinned_critic = previous
            .filter(|f| f.root == key.root && f.role == key.role)
            .and_then(|f| f.critic_anchor_source.clone());
        let pinned_critic_summary = previous
            .filter(|f| f.root == key.root && f.role == key.role)
            .and_then(|f| f.critic_anchor.clone());
        let mut sources = Vec::with_capacity(MAX_MEASUREMENT_SOURCES.min(candidates.len() + 2));
        let mut charge = manifest_charge(run.recording().manifest())?
            .saturating_add(16384)
            .saturating_add(
                pinned_critic_summary
                    .as_ref()
                    .map_or(0, CriticAnchor::charge),
            )
            .saturating_add(pinned.as_ref().map_or(0, Source::charge))
            .saturating_add(pinned_critic.as_ref().map_or(0, Source::charge));
        for index in candidates {
            let source = Source::from_run(run, index)?;
            charge = charge.saturating_add(source.charge());
            checked(
                charge <= REQUEST_BYTES,
                "Learning source descriptors exceed 2 MiB admission",
            )?;
            sources.push(source);
        }
        for anchor in [pinned.as_ref(), pinned_critic.as_ref()]
            .into_iter()
            .flatten()
        {
            if sources.iter().any(|s| s.key == anchor.key) {
                continue;
            }
            if sources.len() == MAX_MEASUREMENT_SOURCES {
                let i = sources
                    .iter()
                    .position(|s| {
                        ![pinned.as_ref(), pinned_critic.as_ref()]
                            .into_iter()
                            .flatten()
                            .any(|p| p.key == s.key)
                    })
                    .ok_or_else(|| {
                        Error::Config("Learning pinned source admission failed".into())
                    })?;
                let removed = sources.remove(i);
                charge = charge.saturating_sub(removed.charge());
            }
            charge = charge.saturating_add(anchor.charge());
            checked(
                charge <= REQUEST_BYTES,
                "Learning frozen source exceeds descriptor admission",
            )?;
            sources.push(anchor.clone());
        }
        sources.sort_by_key(|s| (s.revision, s.key.sequence));
        let total = key.frontiers.iter().map(|f| f.0).sum::<usize>();
        let omitted = total.saturating_sub(sources.len());
        Ok(Self {
            key,
            sources,
            manifest: run.recording().manifest().clone(),
            pinned,
            pinned_summary,
            pinned_critic,
            pinned_critic_summary,
            omitted,
        })
    }
}
// Include the source handles' original shared manifest and our independent
// immutable copy. Explicit metadata-map node charges prevent tiny strings from
// defeating an estimate based only on JSON length.
pub(super) fn manifest_charge(m: &RunManifest) -> Result<usize> {
    let strings = [
        &m.run_id,
        &m.experiment_id,
        &m.algorithm,
        &m.model_id,
        &m.reward_convention,
        &m.environment,
    ]
    .into_iter()
    .map(|s| s.capacity())
    .sum::<usize>();
    let metadata = m
        .metadata
        .iter()
        .map(|(k, v)| k.capacity() + v.capacity() + 256)
        .sum::<usize>();
    let labels = m.observation_labels.capacity() * std::mem::size_of::<String>()
        + m.observation_labels
            .iter()
            .map(String::capacity)
            .sum::<usize>();
    let actions = match &m.action_space {
        rl::core::action::ActionSpace::Box { low, high } => (low.capacity() + high.capacity()) * 4,
        _ => 0,
    };
    let mut descriptor = Counter(0);
    if let Some(d) = &m.environment_descriptor {
        serde_json::to_writer(&mut descriptor, d)?;
    }
    Ok(2 * (8192 + strings + metadata + labels + actions + 16 * descriptor.0))
}
#[derive(Default)]
struct Memory(AtomicUsize);
impl Memory {
    fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Lease> {
        self.0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes).filter(|&v| v <= MAX_OWNER_BYTES)
            })
            .map_err(|_| {
                Error::Config(
                    "Release old Learning summaries before admitting more work (64 MiB owner cap)"
                        .into(),
                )
            })?;
        Ok(Lease {
            memory: self.clone(),
            bytes,
        })
    }
}
pub(super) struct Lease {
    memory: Arc<Memory>,
    bytes: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.memory.0.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct RequestOwner {
    request: LearningRequest,
    _lease: Lease,
}
struct TrainingOwner {
    request: TrainingRequest,
    _lease: Lease,
}
enum Job {
    Prepare(u64, Arc<RequestOwner>),
    Training(u64, Box<TrainingOwner>),
    Check(u64, Weak<LearningFrame>),
}
enum Answer {
    Prepared(Arc<LearningFrame>),
    Training(Arc<TrainingFrame>),
    Checked,
}
#[derive(Default)]
struct Mailbox {
    desired: Option<Job>,
    answer: Option<(u64, Result<Answer>)>,
    stop: bool,
}
pub(super) struct Shared {
    mailbox: Mutex<Mailbox>,
    changed: Condvar,
    serial: AtomicU64,
    memory: Arc<Memory>,
    reads: AtomicU64,
    decodes: AtomicU64,
    completed: AtomicU64,
}
impl Shared {
    pub(super) fn reserve_training(&self, bytes: usize) -> Result<Lease> {
        checked(
            bytes <= MAX_SUMMARY_BYTES,
            "Training summary exceeds 2 MiB admission",
        )?;
        self.memory.reserve(bytes)
    }
}
pub struct LearningWorker {
    shared: Arc<Shared>,
    key: Option<Key>,
    serial: u64,
    pending: bool,
    frame: Option<Arc<LearningFrame>>,
    training_key: Option<TrainingKey>,
    training_frame: Option<Arc<TrainingFrame>>,
    // Same leased summary Arc; keep its immutable reference identities when a
    // later availability check removes the visible result.
    pinned_frame: Option<Arc<LearningFrame>>,
    error: Option<String>,
    submitted: u64,
}
impl LearningWorker {
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Result<Self> {
        let shared = Arc::new(Shared {
            mailbox: Mutex::default(),
            changed: Condvar::new(),
            serial: AtomicU64::new(0),
            memory: Arc::default(),
            reads: AtomicU64::new(0),
            decodes: AtomicU64::new(0),
            completed: AtomicU64::new(0),
        });
        let worker = shared.clone();
        std::thread::Builder::new()
            .name("recorded-learning".into())
            .spawn(move || {
                loop {
                    let job = {
                        let mut m = worker.mailbox.lock().unwrap();
                        while m.desired.is_none() && !m.stop {
                            m = worker.changed.wait(m).unwrap();
                        }
                        if m.stop {
                            return;
                        }
                        m.desired.take().unwrap()
                    };
                    let serial = match &job {
                        Job::Prepare(s, _) | Job::Training(s, _) | Job::Check(s, _) => *s,
                    };
                    let checkpoint = || {
                        checked(
                            worker.serial.load(Ordering::Acquire) == serial,
                            "Learning request superseded",
                        )
                    };
                    if checkpoint().is_err() {
                        continue;
                    }
                    let answer = match job {
                        Job::Prepare(_, owner) => worker
                            .memory
                            .reserve(WORK_BYTES)
                            .and_then(|_work| prepare(owner, &worker, &checkpoint))
                            .map(Answer::Prepared),
                        Job::Training(_, owner) => worker
                            .memory
                            .reserve(WORK_BYTES)
                            .and_then(|_work| owner.request.prepare(&worker, &checkpoint))
                            .map(|frame| {
                                worker.completed.fetch_add(1, Ordering::AcqRel);
                                Answer::Training(frame)
                            }),
                        Job::Check(_, weak) => weak
                            .upgrade()
                            .ok_or_else(|| Error::State("Learning frame retired".into()))
                            .and_then(|frame| {
                                for s in &frame.owner.request.sources {
                                    checkpoint()?;
                                    if !frame.unavailable_sources.contains(&s.key.sequence) {
                                        s.check()?;
                                    }
                                }
                                Ok(Answer::Checked)
                            }),
                    };
                    if checkpoint().is_err() {
                        continue;
                    }
                    let mut m = worker.mailbox.lock().unwrap();
                    if worker.serial.load(Ordering::Acquire) == serial && !m.stop {
                        m.answer = Some((serial, answer));
                        drop(m);
                        wake();
                    }
                }
            })?;
        Ok(Self {
            shared,
            key: None,
            serial: 0,
            pending: false,
            frame: None,
            training_key: None,
            training_frame: None,
            pinned_frame: None,
            error: None,
            submitted: 0,
        })
    }
    pub fn request(&mut self, run: &RunIndex, role: &str) -> Result<bool> {
        self.request_history(run, role, None)
    }

    pub(crate) fn request_history(
        &mut self,
        run: &RunIndex,
        role: &str,
        population: Option<&str>,
    ) -> Result<bool> {
        if self.training_key.is_some() {
            self.retry();
        }
        let key = Key::new(run, role, population)?;
        if self.key.as_ref() == Some(&key) {
            return Ok(false);
        }
        if self.pinned_frame.as_ref().is_some_and(|f| {
            f.root != key.root || f.role != key.role || f.policy_population != key.population
        }) {
            self.pinned_frame = None;
            self.frame = None;
        }
        self.serial = self
            .shared
            .serial
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        {
            let mut mailbox = self.shared.mailbox.lock().unwrap();
            mailbox.desired = None;
            mailbox.answer = None;
        }
        self.key = Some(key.clone());
        self.pending = false;
        self.error = None;
        let owner = (|| {
            let lease = self.shared.memory.reserve(REQUEST_BYTES)?;
            let request = LearningRequest::new(run, key, self.pinned_frame.as_deref())?;
            Ok::<_, Error>(Arc::new(RequestOwner {
                request,
                _lease: lease,
            }))
        })();
        let owner = match owner {
            Ok(owner) => owner,
            Err(error) => {
                self.frame = None;
                self.error = Some(message(error));
                return Ok(false);
            }
        };
        let mut m = self.shared.mailbox.lock().unwrap();
        m.desired = Some(Job::Prepare(self.serial, owner));
        m.answer = None;
        self.pending = true;
        self.error = None;
        self.submitted += 1;
        self.shared.changed.notify_one();
        Ok(true)
    }
    /// Prepare Training on the same worker, with one latest desired request.
    /// Peer indexes are opened only inside the worker's preparation job.
    pub fn request_training(
        &mut self,
        run: &RunIndex,
        reference: Option<(&RunIndex, &str)>,
        cohort_paths: &[PathBuf],
    ) -> Result<bool> {
        let key = match TrainingKey::new(run, reference, cohort_paths) {
            Ok(key) => key,
            Err(error) => {
                self.retry();
                self.error = Some(message(error));
                return Ok(false);
            }
        };
        let same_selection = self.training_key.as_ref() == Some(&key);
        let previous = if same_selection {
            let Some(frame) = self
                .training_frame
                .as_ref()
                .filter(|frame| !self.pending && frame.begin_peer_recheck())
            else {
                return Ok(false);
            };
            Some(frame.clone())
        } else {
            self.retry();
            self.training_key = Some(key.clone());
            None
        };
        let owner = (|| {
            let lease = self.shared.memory.reserve(REQUEST_BYTES)?;
            let request = TrainingRequest::new(run, reference, key, previous)?;
            Ok::<_, Error>(Box::new(TrainingOwner {
                request,
                _lease: lease,
            }))
        })();
        let owner = match owner {
            Ok(owner) => owner,
            Err(error) => {
                self.error = Some(message(error));
                return Ok(false);
            }
        };
        self.shared.mailbox.lock().unwrap().desired = Some(Job::Training(self.serial, owner));
        self.pending = true;
        self.submitted += 1;
        self.shared.changed.notify_one();
        Ok(true)
    }
    pub fn poll(&mut self) {
        let answer = self.shared.mailbox.lock().unwrap().answer.take();
        if let Some((serial, answer)) = answer
            && serial == self.serial
        {
            self.pending = false;
            match answer {
                Ok(Answer::Prepared(f)) => {
                    self.pinned_frame = Some(f.clone());
                    self.frame = Some(f);
                    self.error = None;
                }
                Ok(Answer::Training(f)) => {
                    self.training_frame = Some(f);
                    self.error = None;
                }
                Ok(Answer::Checked) => {}
                Err(e) => {
                    self.frame = None;
                    self.training_frame = None;
                    self.error = Some(message(e));
                }
            }
        }
    }
    pub fn pending(&self) -> bool {
        self.pending
    }
    pub fn frame(&self) -> Option<&Arc<LearningFrame>> {
        self.frame.as_ref()
    }
    pub fn training_frame(&self) -> Option<&Arc<TrainingFrame>> {
        self.training_frame.as_ref()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn metrics(&self) -> LearningMetrics {
        LearningMetrics {
            submitted: self.submitted,
            completed: self.shared.completed.load(Ordering::Acquire),
            reads: self.shared.reads.load(Ordering::Acquire),
            decodes: self.shared.decodes.load(Ordering::Acquire),
            owned_bytes: self.shared.memory.0.load(Ordering::Acquire),
        }
    }
    pub fn clear(&mut self) {
        self.retry();
        self.pinned_frame = None;
    }
    /// Retry unavailable sources while preserving the population definition.
    pub fn retry(&mut self) {
        self.serial = self
            .shared
            .serial
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut m = self.shared.mailbox.lock().unwrap();
        m.desired = None;
        m.answer = None;
        self.key = None;
        self.training_key = None;
        self.pending = false;
        self.frame = None;
        self.training_frame = None;
        self.error = None;
    }
    /// Idle source availability checks share the worker and retain only a Weak frame.
    pub fn recheck(&mut self) -> bool {
        if self.pending {
            return false;
        }
        let Some(frame) = &self.frame else {
            return false;
        };
        self.shared.mailbox.lock().unwrap().desired =
            Some(Job::Check(self.serial, Arc::downgrade(frame)));
        self.pending = true;
        self.shared.changed.notify_one();
        true
    }
}
impl Drop for LearningWorker {
    fn drop(&mut self) {
        self.clear();
        self.shared.mailbox.lock().unwrap().stop = true;
        self.shared.changed.notify_one();
    }
}

pub(super) struct Reader<'a> {
    shared: &'a Shared,
    checkpoint: &'a dyn Fn() -> Result<()>,
    bytes: usize,
    reads: usize,
    failed: BTreeSet<u64>,
    cache: Option<DecodedChunkCache>,
}
impl<'a> Reader<'a> {
    pub(super) fn cached(shared: &'a Shared, checkpoint: &'a dyn Fn() -> Result<()>) -> Self {
        Self {
            shared,
            checkpoint,
            bytes: 0,
            reads: 0,
            failed: BTreeSet::new(),
            cache: Some(DecodedChunkCache::default()),
        }
    }

    pub(super) fn load(&mut self, source: &ChunkSource) -> Result<Arc<Chunk>> {
        (self.checkpoint)()?;
        checked(
            self.reads < MAX_READS && self.bytes < MAX_TOTAL_READ_BYTES,
            "Learning history read-work allowance exhausted; remaining measurements unavailable",
        )?;
        self.reads += 1;
        self.shared.reads.fetch_add(1, Ordering::AcqRel);
        let limit = MAX_READ_BYTES.min(MAX_TOTAL_READ_BYTES - self.bytes);
        let result = if let Some(cache) = &mut self.cache {
            let before = cache.decodes();
            let result = cache.load_if_within_charge(source, limit);
            self.shared
                .decodes
                .fetch_add(cache.decodes() - before, Ordering::AcqRel);
            result
        } else {
            // Ordinary path for complete-job preservation/measurement checks.
            // Those checks count physical decodes only on valid fixtures.
            self.shared.decodes.fetch_add(1, Ordering::AcqRel);
            source
                .load_if_within_charge(limit)
                .map(|chunk| chunk.map(Arc::new))
        };
        let chunk = result?.ok_or_else(|| {
            Error::Config(
                "Learning source exceeds 4 MiB read or remaining history allowance".into(),
            )
        })?;
        self.bytes = self.bytes.saturating_add(chunk.retained_charge());
        (self.checkpoint)()?;
        Ok(chunk)
    }
    fn measurement(&mut self, s: &Source, run: &str, role: &str) -> Result<Arc<Chunk>> {
        let result = (|| {
            let c = self.load(&s.source)?;
            checked(
                c.context.identity.run_id == run
                    && c.context.identity.model_role == role
                    && c.context.identity.learner_update_revision == s.revision
                    && c.record.kind() == s.kind,
                "Learning immutable source identity mismatch",
            )?;
            Ok(c)
        })();
        if result.is_err() {
            self.failed.insert(s.key.sequence);
        }
        result
    }
    fn probes(&mut self, s: &Source) -> Result<Arc<Chunk>> {
        let result = s
            .probe
            .as_ref()
            .ok_or_else(|| Error::Config("Measurement has no original probe definition".into()))
            .and_then(|p| self.load(p));
        if result.is_err() {
            self.failed.insert(s.key.sequence);
        }
        result
    }
}
fn gap(revision: u64, sequence: Option<u64>, reason: impl std::fmt::Display) -> LearningPoint {
    LearningPoint {
        revision,
        sequence,
        value: None,
        count: 0,
        standard_error: None,
        conditions: 0,
        critic_name: None,
        continuation: None,
        condition_counts: Vec::new(),
        label: String::new(),
        reason: Some(message(reason)),
    }
}
fn estimate(
    source: &Source,
    value: f64,
    count: usize,
    se: Option<f64>,
    conditions: usize,
    label: String,
) -> LearningPoint {
    LearningPoint {
        revision: source.revision,
        sequence: Some(source.key.sequence),
        value: Some(value),
        count,
        standard_error: se,
        conditions,
        critic_name: None,
        continuation: None,
        condition_counts: Vec::new(),
        label: short(&label),
        reason: None,
    }
}
fn add_gaps(points: &mut Vec<LearningPoint>, revisions: &BTreeSet<u64>, reason: &str) {
    for &revision in revisions {
        if !points.iter().any(|p| p.revision == revision) {
            points.push(gap(revision, None, reason));
        }
    }
    points.sort_by_key(|p| (p.revision, p.sequence));
}

fn prepare(
    owner: Arc<RequestOwner>,
    shared: &Shared,
    checkpoint: &dyn Fn() -> Result<()>,
) -> Result<Arc<LearningFrame>> {
    prepare_with_cache(
        owner,
        shared,
        checkpoint,
        Some(DecodedChunkCache::default()),
    )
}
fn prepare_with_cache(
    owner: Arc<RequestOwner>,
    shared: &Shared,
    checkpoint: &dyn Fn() -> Result<()>,
    cache: Option<DecodedChunkCache>,
) -> Result<Arc<LearningFrame>> {
    let request = &owner.request;
    let mut reader = Reader {
        shared,
        checkpoint,
        bytes: 0,
        reads: 0,
        failed: BTreeSet::new(),
        cache,
    };
    let mut reference = None;
    let mut reference_error = None;
    let anchor_candidates: Vec<_> = if let Some(pinned) = &request.pinned {
        vec![pinned]
    } else {
        request
            .sources
            .iter()
            .filter(|s| s.kind == RecordKind::Policy)
            .collect()
    };
    for source in anchor_candidates {
        checkpoint()?;
        let candidate = (|| {
            let c = reader.measurement(source, &request.manifest.run_id, &request.key.role)?;
            let p = reader.probes(source)?;
            let change = policy_change(&c, &p, &c, &p, &request.manifest.action_space)?;
            Ok::<_, Error>((source.clone(), c, p, change))
        })();
        match candidate {
            Ok(value) => {
                reference = Some(value);
                break;
            }
            Err(e) => reference_error = Some(message(e)),
        }
    }
    let anchor = reference
        .as_ref()
        .map(|(source, _, p, change)| {
            let Record::ProbeSet(p) = &p.record else {
                unreachable!()
            };
            PolicyAnchor {
                revision: source.revision,
                sequence: source.key.sequence,
                probe_sequence: source.probe_sequence.unwrap(),
                rows: change.rows,
                population: short(&format!(
                    "{}; {:?}; {}",
                    short(&p.id),
                    p.population,
                    short(&p.methodology)
                )),
                kind: change.kind,
                normalization: change.normalization.clone(),
            }
        })
        .or_else(|| request.pinned_summary.clone());
    let mut evaluations = Vec::with_capacity(request.sources.len());
    let mut changes = Vec::with_capacity(request.sources.len());
    let mut policy_space_samples = PolicySpaceSamples::default();
    let mut policy_space_error = None;
    let mut errors = Vec::with_capacity(request.sources.len());
    let mut critic_reference = request
        .pinned_critic
        .clone()
        .zip(request.pinned_critic_summary.clone());
    let revisions: BTreeSet<_> = request.sources.iter().map(|s| s.revision).collect();
    for source in &request.sources {
        checkpoint()?;
        match source.kind {
            RecordKind::Empirical => {
                let result: Result<Option<LearningPoint>> = (|| {
                    let c =
                        reader.measurement(source, &request.manifest.run_id, &request.key.role)?;
                    let Record::Empirical(e) = &c.record else {
                        unreachable!()
                    };
                    if e.evaluation.conditioning != Conditioning::InitialScenario {
                        return Ok(None);
                    }
                    let s = evaluation_summary(&c)?;
                    Ok(Some(estimate(
                        source,
                        s.mean,
                        s.count,
                        s.mean_standard_error,
                        0,
                        format!(
                            "{}; discount {}; {}; {}; each batch separate",
                            short(&e.evaluation.continuation_policy),
                            e.evaluation.discount,
                            if e.evaluation.stochastic_policy {
                                "sampled policy"
                            } else {
                                "deterministic policy"
                            },
                            if e.evaluation.independent_future_streams {
                                "independent futures"
                            } else {
                                "future independence undeclared; SE unavailable"
                            }
                        ),
                    )))
                })();
                match result {
                    Ok(Some(p)) => evaluations.push(p),
                    Ok(None) => {}
                    Err(e) => evaluations.push(gap(source.revision, Some(source.key.sequence), e)),
                }
            }
            RecordKind::Policy => {
                let result = (|| {
                    let (anchor, c, p, _) = reference.as_ref().ok_or_else(|| {
                        Error::Config(reference_error.clone().unwrap_or_else(|| {
                            "No compatible fixed/imported policy reference".into()
                        }))
                    })?;
                    let current = if source.key == anchor.key {
                        None
                    } else {
                        Some((
                            reader.measurement(
                                source,
                                &request.manifest.run_id,
                                &request.key.role,
                            )?,
                            reader.probes(source)?,
                        ))
                    };
                    let (cur, probes) = current.as_ref().map_or((c, p), |(c, p)| (c, p));
                    let change = policy_change(c, p, cur, probes, &request.manifest.action_space)?;
                    if let Err(error) = policy_space_samples.add(
                        source.revision,
                        cur,
                        probes,
                        p,
                        &request.manifest.action_space,
                        &change.normalization,
                    ) {
                        policy_space_error = Some(message(error));
                    }
                    Ok::<_, Error>(estimate(
                        source,
                        change.value,
                        change.rows,
                        None,
                        change.rows,
                        format!(
                            "{:?}; complete fixed population; reference k={} sequence={}",
                            change.kind, anchor.revision, anchor.key.sequence
                        ),
                    ))
                })();
                changes.push(
                    result.unwrap_or_else(|e| gap(source.revision, Some(source.key.sequence), e)),
                );
            }
            RecordKind::Critic => {
                let result = prepare_error(source, request, &mut reader);
                errors.push(match result {
                    Ok((mut point, candidate)) => {
                        if let Some((reference, anchor)) = &critic_reference {
                            if let Err(error) = match_critic_cohort(
                                reference,
                                anchor,
                                source,
                                &candidate,
                                request,
                                &mut reader,
                            ) {
                                point.value = None;
                                point.reason = Some(message(error));
                            }
                        } else {
                            critic_reference = Some((source.clone(), candidate));
                        }
                        point
                    }
                    Err(e) => gap(source.revision, Some(source.key.sequence), e),
                });
            }
            _ => unreachable!(),
        }
    }
    add_gaps(
        &mut evaluations,
        &revisions,
        "No complete initial-scenario return measurement at this recorded revision",
    );
    add_gaps(
        &mut changes,
        &revisions,
        "No policy measurement on the frozen population at this recorded revision",
    );
    add_gaps(
        &mut errors,
        &revisions,
        "No independently matched certified critic-error measurement at this recorded revision",
    );
    for source in &request.sources {
        checkpoint()?;
        if !reader.failed.contains(&source.key.sequence) {
            source.check()?;
        }
    }
    if anchor.is_some() {
        checkpoint()?;
    }
    let (policy_space, policy_space_unavailable) = match policy_space_samples.fit() {
        Ok(projection) => (Some(projection), None),
        Err(error) => (
            None,
            Some(policy_space_error.unwrap_or_else(|| message(error))),
        ),
    };
    let points = [&evaluations, &changes, &errors];
    let mut bytes =
        8192 + policy_space.as_ref().map_or(0, |p| {
            p.points.capacity() * std::mem::size_of::<super::policy_space::PolicySpacePoint>()
        }) + policy_space_unavailable
            .as_ref()
            .map_or(0, String::capacity)
            + reader.failed.len() * 64
            + points
                .iter()
                .map(|v| {
                    v.capacity() * std::mem::size_of::<LearningPoint>()
                        + v.iter()
                            .map(|p| {
                                p.label.capacity()
                                    + p.reason.as_ref().map_or(0, String::capacity)
                                    + p.critic_name.as_ref().map_or(0, String::capacity)
                                    + p.continuation.as_ref().map_or(0, String::capacity)
                                    + p.condition_counts.capacity() * std::mem::size_of::<usize>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>();
    if let Some(a) = &anchor {
        bytes += a.population.capacity() + a.normalization.capacity() * 8;
    }
    bytes += reference.as_ref().map_or_else(
        || request.pinned.as_ref().map_or(0, Source::charge),
        |r| r.0.charge(),
    );
    if let Some((_, a)) = &critic_reference {
        bytes += a.charge();
    }
    bytes += critic_reference
        .as_ref()
        .map_or(0, |(source, _)| source.charge());
    checked(
        bytes <= MAX_SUMMARY_BYTES,
        "Learning summaries exceed 2 MiB admission",
    )?;
    let lease = shared.memory.reserve(bytes)?;
    checkpoint()?;
    shared.completed.fetch_add(1, Ordering::AcqRel);
    Ok(Arc::new(LearningFrame {
        root: request.key.root.clone(),
        role: request.key.role.clone(),
        policy_population: request.key.population.clone(),
        evaluations,
        policy_changes: changes,
        policy_space,
        policy_space_unavailable,
        critic_errors: errors,
        anchor,
        critic_anchor: critic_reference.as_ref().map(|(_, a)| a.clone()),
        omitted_sources: request.omitted,
        retained_bytes: bytes,
        anchor_source: reference.map(|r| r.0).or_else(|| request.pinned.clone()),
        critic_anchor_source: critic_reference.map(|(s, _)| s),
        unavailable_sources: reader.failed,
        owner,
        _lease: lease,
    }))
}
fn prepare_error(
    source: &Source,
    request: &LearningRequest,
    reader: &mut Reader<'_>,
) -> Result<(LearningPoint, CriticAnchor)> {
    let critic = reader.measurement(source, &request.manifest.run_id, &request.key.role)?;
    let probes = reader.probes(source)?;
    let Record::ProbeSet(p) = &probes.record else {
        return Err(Error::Config(
            "Critic original probes are unavailable".into(),
        ));
    };
    let ids = probes.arrays[&p.row_ids].as_u64()?;
    checked(
        !ids.is_empty()
            && ids.len() <= super::MAX_FIXED_ROWS
            && p.population != ProbePopulation::PolicyVisited,
        "Critic population exceeds complete-row admission",
    )?;
    let declaration = value::declaration(&critic)?
        .ok_or_else(|| Error::Config("Critic scientific declaration unavailable".into()))?;
    let (critic_name, subject) = declaration
        .critics
        .iter()
        .next()
        .ok_or_else(|| Error::Config("Named critic subject unavailable".into()))?;
    let Record::Critic(record) = &critic.record else {
        unreachable!()
    };
    let continuation = subject
        .continuation
        .as_ref()
        .ok_or_else(|| Error::Config("Critic declared continuation unavailable".into()))?;
    let mut seen = BTreeSet::new();
    let mut methods = BTreeMap::new();
    let mut method_bytes = 0usize;
    let mut mean = 0.;
    let mut count = 0usize;
    let mut kind = None;
    let mut crossings = 0;
    let mut rejected = 0;
    let mut last_reason =
        "No same-revision certified observation-conditional future samples".into();
    for sample_source in request.sources.iter().filter(|s| {
        s.kind == RecordKind::Empirical
            && s.revision == source.revision
            && s.probe_id == source.probe_id
    }) {
        let result = (|| {
            let sample =
                reader.measurement(sample_source, &request.manifest.run_id, &request.key.role)?;
            let Record::Empirical(e) = &sample.record else {
                unreachable!()
            };
            let empirical_row = usize::try_from(e.probe_row.ok_or_else(|| {
                Error::Config("Scenario returns cannot calibrate a selected critic state".into())
            })?)
            .map_err(|_| Error::Shape("empirical row overflow".into()))?;
            let sample_probes = reader.probes(sample_source)?;
            let Record::ProbeSet(ep) = &sample_probes.record else {
                return Err(Error::Config(
                    "Empirical original probes unavailable".into(),
                ));
            };
            let row_id = *sample_probes.arrays[&ep.row_ids]
                .as_u64()?
                .get(empirical_row)
                .ok_or_else(|| Error::Shape("empirical row out of bounds".into()))?;
            let row = ids.iter().position(|&id| id == row_id).ok_or_else(|| {
                Error::Config("Conditional row ID missing from critic population".into())
            })?;
            let error = independent_critic_error(
                &critic,
                &probes,
                row,
                &sample,
                &sample_probes,
                &request.manifest,
            )?;
            let Conditioning::Observation { methodology } = &e.evaluation.conditioning else {
                unreachable!()
            };
            Ok::<_, Error>((error, methodology.clone()))
        })();
        match result {
            Ok((error, method)) => {
                checked(
                    seen.insert(error.row_id),
                    "Multiple empirical batches for one critic condition are ambiguous; batches were not pooled",
                )?;
                checked(
                    kind.is_none_or(|k| k == error.kind),
                    "Incompatible independent critic score families",
                )?;
                kind = Some(error.kind);
                method_bytes = method_bytes.saturating_add(method.capacity());
                checked(
                    method_bytes <= 128 * 1024,
                    "Critic cohort methodology exceeds 128 KiB admission",
                )?;
                methods.insert(error.row_id, (method, error.count));
                mean += (error.value - mean) / seen.len() as f64;
                count += error.count;
                crossings += error.raw_crossings;
            }
            Err(e) => {
                rejected += 1;
                last_reason = message(e);
            }
        }
    }
    let kind = kind.ok_or_else(|| Error::Config(last_reason.clone()))?;
    let name = match kind {
        ErrorKind::ScalarAbsolute => "mean |prediction - conditional MC mean|",
        ErrorKind::OrderedQuantileCrps => "mean CRPS of ordered quantile atoms",
    };
    let mut point = estimate(
        source,
        mean,
        count,
        None,
        seen.len(),
        format!(
            "{}; declared current-revision continuation {}; {name}; equal condition weights; {} / {} rows of {}; {crossings} raw crossings",
            critic_name,
            continuation.identity,
            seen.len(),
            ids.len(),
            short(&p.id),
        ),
    );
    // Exact identifiers remain available even if the compact plot caption clips.
    point.critic_name = Some(critic_name.clone());
    point.continuation = Some(continuation.identity.clone());
    point.condition_counts = methods.values().map(|(_, n)| *n).collect();
    if rejected > 0 {
        point.reason = Some(short(&format!(
            "{rejected} unmatched batches excluded: {last_reason}"
        )));
    }
    let anchor = CriticAnchor {
        revision: source.revision,
        sequence: source.key.sequence,
        probe_sequence: source
            .probe_sequence
            .ok_or_else(|| Error::Config("Critic cohort original source missing".into()))?,
        critic_name: critic_name.clone(),
        kind: subject.kind,
        score_kind: kind,
        action_condition: record.action_condition,
        stochastic: continuation.stochastic,
        row_ids: methods.keys().copied().collect(),
        observation_methods: methods.into_values().map(|(method, _)| method).collect(),
        population: short(&format!("{}; {:?}; {}", p.id, p.population, p.methodology)),
        initial_continuation: continuation.identity.clone(),
    };
    Ok((point, anchor))
}

fn match_critic_cohort(
    reference: &Source,
    anchor: &CriticAnchor,
    current: &Source,
    candidate: &CriticAnchor,
    request: &LearningRequest,
    reader: &mut Reader<'_>,
) -> Result<()> {
    checked(
        anchor.critic_name == candidate.critic_name
            && anchor.kind == candidate.kind
            && anchor.score_kind == candidate.score_kind
            && anchor.action_condition == candidate.action_condition
            && anchor.stochastic == candidate.stochastic,
        "Frozen critic subject/score/action-condition/continuation sampling mode differs",
    )?;
    checked(
        anchor.row_ids == candidate.row_ids
            && anchor.observation_methods == candidate.observation_methods,
        "Measured critic condition cohort differs from the frozen reference; no conditions were substituted",
    )?;
    if reference.key == current.key {
        return Ok(());
    }
    let a = reader.measurement(reference, &request.manifest.run_id, &request.key.role)?;
    let ap = reader.probes(reference)?;
    let b = reader.measurement(current, &request.manifest.run_id, &request.key.role)?;
    let bp = reader.probes(current)?;
    let da = value::declaration(&a)?
        .ok_or_else(|| Error::Config("Frozen critic declaration unavailable".into()))?;
    let db = value::declaration(&b)?
        .ok_or_else(|| Error::Config("Current critic declaration unavailable".into()))?;
    let ca = &da.critics[&anchor.critic_name];
    let cb = &db.critics[&candidate.critic_name];
    checked(
        da.conventions == db.conventions
            && ca.learned_baseline == cb.learned_baseline
            && ca.twin == cb.twin
            && ca.utility == cb.utility
            && ca.actor.as_ref().map(|p| p.stochastic) == cb.actor.as_ref().map(|p| p.stochastic),
        "Frozen critic scientific conventions/subject differ",
    )?;
    let (Record::ProbeSet(pa), Record::ProbeSet(pb)) = (&ap.record, &bp.record) else {
        unreachable!()
    };
    checked(
        pa.id == pb.id,
        "Frozen critic cohort population identity differs",
    )?;
    let aligned = match_reference_sets(&ap, &bp)?;
    let ids = ap.arrays[&pa.row_ids].as_u64()?;
    let (Record::Critic(ra), Record::Critic(rb)) = (&a.record, &b.record) else {
        unreachable!()
    };
    match (&ra.actions, &rb.actions) {
        (None, None) => {}
        (Some(ka), Some(kb)) => {
            for id in &anchor.row_ids {
                let i = ids
                    .iter()
                    .position(|row| row == id)
                    .ok_or_else(|| Error::Config("Frozen critic cohort row missing".into()))?;
                checked(
                    equal_row(&a.arrays[ka], &b.arrays[kb], i, aligned[i]),
                    "Complete first-action condition differs from frozen critic cohort",
                )?;
            }
        }
        _ => {
            return Err(Error::Config(
                "Frozen critic cohort first-action availability differs".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "history_reuse_tests.rs"]
mod reuse_tests;
