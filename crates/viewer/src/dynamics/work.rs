//! Single-flight, bounded recorded Dynamics preparation. All reads and scans run
//! here, never in paint. Geometry samples actual adjacent edges; gaps stay gaps.
pub use super::history::DomainSpec;
mod episode;
use super::history::{Coverage, CoveragePoint, FixedGrid, ReferenceExtrema};
use super::source::{ProtocolKey, TrajectorySource};
use super::{ActionAxis, ActionValue, Axes, Episode, Limits, Occupancy, checked, prepare_episode};
use crate::budgets::{DYNAMICS_FRAME, DYNAMICS_OWNER, DYNAMICS_READ, DYNAMICS_SOURCE_SET};
use crate::data::{CacheKey, RunIndex};
pub use episode::{EpisodeFrame, EpisodeRequest};
use rl::{
    Error, Result,
    diagnostics::{
        schema::{Chunk, RecordKind, RunManifest},
        storage::ChunkSource,
    },
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
};

static ID: AtomicU64 = AtomicU64::new(1);
const MIB: usize = 1024 * 1024;
/// Catalog active/desired handles, published metadata, and bounded cursor/error
/// tables share this reservation within the existing owner cap.
pub(super) const CATALOG_RESERVATION: usize = 16 * MIB;
/// Per-owner gate: catalog and preparation share one decoder scratch envelope.
/// Unrelated panes/workers never contend on a process-wide lock.
pub type DecodeGate = Arc<DecodeBudget>;
pub struct DecodeBudget {
    pub(super) lock: Mutex<()>,
    pub(super) read_limit: usize,
    catalog_bytes: AtomicUsize,
}
impl Default for DecodeBudget {
    fn default() -> Self {
        Self {
            lock: Mutex::new(()),
            read_limit: DYNAMICS_READ,
            catalog_bytes: AtomicUsize::new(0),
        }
    }
}
pub(super) struct CatalogReservation {
    budget: DecodeGate,
    bytes: usize,
}
impl DecodeBudget {
    pub(super) fn reserve(self: &Arc<Self>, bytes: usize) -> Result<CatalogReservation> {
        self.catalog_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes)
                    .filter(|total| *total <= CATALOG_RESERVATION)
            })
            .map_err(|_| Error::Shape("Dynamics catalog ownership is still retiring".into()))?;
        Ok(CatalogReservation {
            budget: self.clone(),
            bytes,
        })
    }
}
impl Drop for CatalogReservation {
    fn drop(&mut self) {
        self.budget
            .catalog_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[derive(Clone, Debug)]
pub struct WorkLimits {
    pub max_sources: usize,
    pub max_decisions: usize,
    pub max_source_set_bytes: usize,
    pub max_read_bytes: usize,
    /// Each of the extrema and preparation phases gets this read allowance.
    /// Their aggregate is at most twice this value; retained ownership is unchanged.
    pub max_total_read_bytes: usize,
    pub max_episode_bytes: usize,
    pub max_frame_bytes: usize,
    pub max_grid_cells: usize,
    pub max_displayed_episodes: usize,
    pub max_segments: usize,
    pub max_trace_points: usize,
    pub max_owner_bytes: usize,
}
impl Default for WorkLimits {
    fn default() -> Self {
        Self {
            max_sources: 512,
            max_decisions: 4_000_000,
            max_source_set_bytes: DYNAMICS_SOURCE_SET,
            max_read_bytes: DYNAMICS_READ,
            max_total_read_bytes: 256 * MIB,
            max_episode_bytes: 32 * MIB,
            max_frame_bytes: DYNAMICS_FRAME,
            max_grid_cells: 256 * 256,
            max_displayed_episodes: 32,
            max_segments: 8192,
            max_trace_points: 4096,
            max_owner_bytes: DYNAMICS_OWNER,
        }
    }
}
impl WorkLimits {
    /// Conservative simultaneous ownership: four distinct frames (including
    /// external leases/answer/build), one worker cache, a decoded episode cache
    /// and one streaming episode (each at most max_episode_bytes),
    /// decoder scratch (encoded + logical growth + arrays), four source sets.
    /// Availability queues/results own only Weak references. Its single active
    /// pass shares the worker slot and drops frame/source upgrades before reply;
    /// frame charges already include independent projection leases.
    /// Shared Arcs are deliberately charged more than once. Not process RSS.
    pub fn envelope_bytes(&self) -> Result<usize> {
        let mut bytes = CATALOG_RESERVATION;
        for (n, multiplier) in [
            (self.max_frame_bytes, 5),
            (self.max_episode_bytes, 2),
            (self.max_read_bytes, 5),
            (self.max_source_set_bytes, 4),
        ] {
            bytes = bytes
                .checked_add(n.checked_mul(multiplier).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        checked(
            bytes <= self.max_owner_bytes,
            "Dynamics owner envelope exceeds budget",
        )?;
        checked(
            self.max_sources > 0
                && self.max_sources <= 4096
                && self.max_decisions > 0
                && self.max_grid_cells > 0
                && self.max_displayed_episodes > 0
                && self.max_segments >= 2
                && self.max_trace_points >= 2,
            "invalid Dynamics work limits",
        )?;
        // Preflight every output's count-driven allocation before loops.
        // Two 2D path collections and two 3D collections, plus bounded index
        // scratch. Path metadata and revision evidence also precede allocation.
        let geometry = self.max_segments.checked_mul(224).ok_or_else(overflow)?;
        let path_count = self
            .max_displayed_episodes
            .min(self.max_sources)
            .min(self.max_segments / 2);
        let metadata = path_count
            .checked_mul(1024)
            .and_then(|n| {
                self.max_sources
                    .checked_mul(std::mem::size_of::<CoveragePoint>())
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(overflow)?;
        let traces = self.max_trace_points.checked_mul(64).ok_or_else(overflow)?;
        let grid = self.max_grid_cells.checked_mul(48).ok_or_else(overflow)?;
        checked(
            geometry
                .checked_add(traces)
                .and_then(|n| n.checked_add(grid))
                .and_then(|n| n.checked_add(metadata))
                .ok_or_else(overflow)?
                <= self.max_frame_bytes,
            "Dynamics geometry limits exceed frame budget",
        )?;
        Ok(bytes)
    }
    fn science(&self) -> Limits {
        Limits {
            max_episode_bytes: self.max_episode_bytes,
            max_grid_cells: self.max_grid_cells,
        }
    }
}
fn overflow() -> Error {
    Error::Shape("Dynamics ownership arithmetic overflow".into())
}

#[derive(Debug, PartialEq, Eq)]
struct SourceIdentity {
    session: usize,
    generation: u64,
    sources: Arc<[TrajectorySource]>,
}
pub struct SourceSet {
    identity: Arc<SourceIdentity>,
    generation: u64,
    root: Arc<PathBuf>,
    sources: Arc<[TrajectorySource]>,
    handles: Vec<ChunkSource>,
    manifest: Arc<RunManifest>,
    retained_bytes: usize,
    omitted_sources: usize,
    omitted_decisions: usize,
}
impl SourceSet {
    pub fn new(
        run: &RunIndex,
        entries: &[TrajectorySource],
        generation: u64,
        limits: &WorkLimits,
    ) -> Result<Arc<Self>> {
        limits.envelope_bytes()?;
        checked(
            !entries.is_empty() && entries.len() <= 4096,
            "Dynamics catalog candidate count exceeds budget or is empty",
        )?;
        let first = &entries[0];
        checked(
            first.protocol.unconditional(),
            "Dynamics requires unconditional evaluation evidence",
        )?;
        // Keep the newest whole episodes that fit, in stable (revision,sequence)
        // order. Oversize individual episodes are skipped. This selection never
        // depends on selected episode/action, so occupancy remains comparable.
        let mut candidates: Vec<_> = entries.iter().collect();
        candidates.sort_by_key(|s| (s.revision, s.key.sequence));
        let mut admitted = Vec::with_capacity(limits.max_sources.min(entries.len()));
        let mut remaining = limits.max_decisions;
        let mut omitted_decisions = 0usize;
        for entry in candidates.into_iter().rev() {
            checked(
                Arc::ptr_eq(&entry.key.root, run.root())
                    && entry.protocol == first.protocol
                    && entry.role == first.role,
                "Dynamics sources must share exact run, role and evaluation protocol",
            )?;
            if admitted.len() < limits.max_sources && entry.transitions <= remaining {
                admitted.push(entry);
                remaining -= entry.transitions;
            } else {
                omitted_decisions = omitted_decisions
                    .checked_add(entry.transitions)
                    .ok_or_else(overflow)?;
            }
        }
        checked(
            !admitted.is_empty(),
            "no Dynamics episode fits the decision work budget",
        )?;
        let omitted_sources = entries.len() - admitted.len();
        let mut decisions = 0usize;
        let mut charge = 4096usize;
        for entry in &admitted {
            checked(
                Arc::ptr_eq(&entry.key.root, run.root())
                    && entry.protocol == first.protocol
                    && entry.role == first.role,
                "Dynamics sources must share exact run, role and evaluation protocol",
            )?;
            decisions = decisions
                .checked_add(entry.transitions)
                .ok_or_else(overflow)?;
            checked(
                decisions <= limits.max_decisions,
                "Dynamics source decisions exceed work budget",
            )?;
            // Catalog fields are admitted to 16 KiB individually. Verify here
            // as public callers can also construct TrajectorySource directly.
            let text = entry
                .id
                .len()
                .checked_add(entry.role.len())
                .and_then(|n| n.checked_add(entry.environment.len()))
                .and_then(|n| n.checked_add(entry.protocol.retained_bytes()))
                .ok_or_else(overflow)?;
            charge = charge
                .checked_add(text.checked_mul(4).ok_or_else(overflow)?)
                .and_then(|n| n.checked_add(2048 + run.root().as_os_str().len() * 8))
                .ok_or_else(overflow)?;
            checked(
                charge <= limits.max_source_set_bytes,
                "Dynamics source metadata exceeds budget",
            )?;
            let i = run
                .by_sequence(entry.key.sequence)
                .ok_or_else(|| Error::State("Dynamics exact source no longer indexed".into()))?;
            let index = &run.recording().entries()[i];
            checked(
                index.kind == RecordKind::Trajectory
                    && index.revision == entry.revision
                    && index.model_role == entry.role,
                "Dynamics source index identity mismatch",
            )?;
        }
        let manifest_bytes = serde_json::to_vec(run.recording().manifest())?.len();
        charge = charge
            .checked_add(manifest_bytes.checked_mul(8).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        checked(
            charge <= limits.max_source_set_bytes,
            "Dynamics manifest/source metadata exceeds budget",
        )?;
        let mut sources: Vec<_> = admitted.into_iter().cloned().collect();
        sources.sort_by_key(|s| (s.revision, s.key.sequence));
        checked(
            sources.windows(2).all(|p| p[0].key != p[1].key),
            "duplicate Dynamics source",
        )?;
        let handles = sources
            .iter()
            .map(|s| {
                run.recording()
                    .source(run.by_sequence(s.key.sequence).expect("validated index"))
            })
            .collect::<Result<Vec<_>>>()?;
        let sources: Arc<[TrajectorySource]> = sources.into();
        let identity = Arc::new(SourceIdentity {
            session: Arc::as_ptr(run.root()) as usize,
            generation,
            sources: sources.clone(),
        });
        Ok(Arc::new(Self {
            identity,
            generation,
            root: run.root().clone(),
            sources,
            handles,
            manifest: Arc::new(run.recording().manifest().clone()),
            retained_bytes: charge,
            omitted_sources,
            omitted_decisions,
        }))
    }
    pub fn sources(&self) -> &[TrajectorySource] {
        &self.sources
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn root(&self) -> &Arc<PathBuf> {
        &self.root
    }
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    pub fn omitted_sources(&self) -> usize {
        self.omitted_sources
    }
    pub fn omitted_decisions(&self) -> usize {
        self.omitted_decisions
    }
    /// Bounded metadata-only availability check for explicit application refresh,
    /// never an idle paint scan. No payload read or array conversion occurs.
    pub fn check_available(&self) -> Result<()> {
        for source in &self.handles {
            source.check_available()?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    None,
    State([usize; 3]),
    StateAction,
}
#[derive(Clone)]
pub struct Request {
    pub sources: Arc<SourceSet>,
    pub revision: u64,
    pub axes: Axes,
    pub cells: [usize; 2],
    pub selected: CacheKey,
    pub action: ActionAxis,
    pub projection: Projection,
    pub reference: Option<Arc<Reference>>,
    pub domain: DomainSpec,
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// History identity. Revision-local state is keyed by this identity plus
/// `PreparedState::revision`; selecting another revision preserves this key.
pub struct StateKey {
    source_set: Arc<SourceIdentity>,
    axes: Axes,
    cells: [usize; 2],
    reference: Option<u64>,
    domain: DomainSpec,
}
impl StateKey {
    pub fn sources(&self) -> &[TrajectorySource] {
        &self.source_set.sources
    }
    pub fn generation(&self) -> u64 {
        self.source_set.generation
    }
    pub fn axes(&self) -> Axes {
        self.axes
    }
    pub fn domain(&self) -> DomainSpec {
        self.domain
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpisodeKey {
    pub source: CacheKey,
    pub axes: Axes,
    pub action: ActionAxis,
    pub projection: Projection,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestKey {
    state: StateKey,
    revision: u64,
    episode: EpisodeKey,
}
impl Request {
    fn key(&self) -> RequestKey {
        RequestKey {
            state: StateKey {
                source_set: self.sources.identity.clone(),
                axes: self.axes,
                cells: self.cells,
                reference: self.reference.as_ref().map(|r| r.id),
                domain: self.domain,
            },
            revision: self.revision,
            episode: EpisodeKey {
                source: self.selected.clone(),
                axes: self.axes,
                action: self.action,
                projection: self.projection,
            },
        }
    }
    fn validate(&self, limits: &WorkLimits) -> Result<()> {
        checked(
            self.sources.sources.len() <= limits.max_sources
                && self.sources.retained_bytes <= limits.max_source_set_bytes,
            "Dynamics source set exceeds worker limits",
        )?;
        let decisions = self.sources.sources.iter().try_fold(0usize, |n, s| {
            n.checked_add(s.transitions).ok_or_else(overflow)
        })?;
        checked(
            decisions <= limits.max_decisions,
            "Dynamics decisions exceed worker limits",
        )?;
        checked(
            self.sources
                .sources
                .iter()
                .any(|s| s.key == self.selected && s.revision == self.revision),
            "selected Dynamics episode must belong to requested revision",
        )?;
        checked(
            self.axes.x != self.axes.y,
            "Dynamics state axes must differ",
        )?;
        let cells = self.cells[0]
            .checked_mul(self.cells[1])
            .ok_or_else(overflow)?;
        checked(
            cells > 0 && self.cells.iter().all(|n| *n > 0) && cells <= limits.max_grid_cells,
            "Dynamics grid exceeds worker limit",
        )?;
        if let Some(reference) = &self.reference {
            checked(
                reference.matches(self),
                "Dynamics reference belongs to another projection/protocol/run",
            )?;
            checked(
                reference.retained_bytes <= limits.max_source_set_bytes,
                "Dynamics pinned reference exceeds source ownership budget",
            )?;
        }
        Ok(())
    }
}
pub struct Reference {
    pub grid: FixedGrid,
    pub source_keys: Vec<CacheKey>,
    id: u64,
    root: Arc<PathBuf>,
    protocol: ProtocolKey,
    role: String,
    axes: Axes,
    // Pin every exact immutable address, including sources evicted by append.
    sources: Vec<ChunkSource>,
    domain: DomainSpec,
    extrema: ReferenceExtrema,
    retained_bytes: usize,
}
impl std::fmt::Debug for Reference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reference")
            .field("grid", &self.grid)
            .field("source_keys", &self.source_keys)
            .field("id", &self.id)
            .field("retained_bytes", &self.retained_bytes)
            .finish_non_exhaustive()
    }
}
impl Reference {
    /// Check every exact source behind these fixed bounds during explicit
    /// application refresh, including after it leaves the admitted population.
    /// This is metadata-only and never recomputes or moves the grid.
    pub fn check_available(&self) -> Result<()> {
        self.check_available_with(|| Ok(()), ChunkSource::check_available)
    }
    fn check_available_with(
        &self,
        check: impl Fn() -> Result<()>,
        check_source: impl Fn(&ChunkSource) -> Result<()>,
    ) -> Result<()> {
        for source in &self.sources {
            check()?;
            let available = check_source(source);
            // One OS call cannot be interrupted. Once it returns, cancellation
            // takes precedence over a failure from an obsolete source context.
            check()?;
            available.map_err(|error| {
                Error::State(format!(
                    "Dynamics fixed-reference source unavailable: {error}"
                ))
            })?;
        }
        Ok(())
    }
    /// Conservative ownership for retained handles, their shared manifest,
    /// metadata, source keys and fixed-grid provenance.
    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn domain(&self) -> DomainSpec {
        self.domain
    }
    pub fn episodes(&self) -> usize {
        self.extrema.episodes
    }
    pub fn decisions(&self) -> usize {
        self.extrema.decisions
    }
    /// Exact observed numeric state extrema, including terminal endpoints.
    pub fn state_extrema(&self, coordinate: usize) -> Option<[f64; 2]> {
        self.extrema.state.get(coordinate).copied().flatten()
    }
    /// Declared range when present, otherwise exact observed state extrema.
    /// Manual XY overrides live in `grid`; other coordinates remain unchanged.
    pub fn state_bounds(&self, coordinate: usize) -> Option<[f64; 2]> {
        self.state_extrema(coordinate)
            .map(|observed| self.extrema.declared[coordinate].unwrap_or(observed))
    }
    /// Exact observed continuous actions at decisions only; never terminal.
    pub fn action_bounds(&self, component: usize) -> Option<[f64; 2]> {
        self.extrema.actions.get(component).copied()
    }
    fn matches(&self, req: &Request) -> bool {
        Arc::ptr_eq(&self.root, &req.sources.root)
            && self.protocol == req.sources.sources[0].protocol
            && self.role == req.sources.sources[0].role
            && self.axes == req.axes
            && self.domain == req.domain
            && self.grid.grid.cells() == req.cells
    }
}
#[derive(Clone, Debug)]
pub struct Segment2 {
    pub t: usize,
    pub points: [[f64; 2]; 2],
}
#[derive(Clone, Debug)]
pub struct Segment3 {
    pub t: usize,
    pub points: [[f64; 3]; 2],
}
#[derive(Clone, Debug)]
pub struct Path2 {
    pub key: CacheKey,
    pub segments: Vec<Segment2>,
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub transitions: usize,
}
/// Bounded adjacent edges, with exact endpoints even for a one-decision path.
#[derive(Clone, Debug)]
pub struct Path3 {
    pub key: CacheKey,
    pub segments: Vec<Segment3>,
    pub start: [f64; 3],
    pub end: [f64; 3],
}
#[derive(Debug)]
pub struct PreparedProjection {
    state: StateKey,
    revision: u64,
    projection: Projection,
    action: Option<ActionAxis>,
    pub paths: Vec<Path3>,
    pub bounds: [[f64; 2]; 3],
    pub eligible_episodes: usize,
    pub eligible_segments: usize,
    pub displayed_segments: usize,
    pub retained_bytes: usize,
}
impl PreparedProjection {
    fn matches(&self, state: &PreparedState, req: &Request) -> bool {
        self.state == state.key
            && self.revision == req.revision
            && self.projection == req.projection
            && self.action == (req.projection == Projection::StateAction).then_some(req.action)
    }
}
#[derive(Clone, Debug)]
pub struct ActionSample {
    pub t: usize,
    pub value: ActionValue,
}
#[derive(Debug)]
pub struct PreparedState {
    pub key: StateKey,
    pub revision: u64,
    pub reference: Arc<Reference>,
    pub occupancy: Occupancy,
    pub coverage: Arc<Coverage>,
    pub trajectories: Vec<Path2>,
    pub population_episodes: usize,
    pub population_decisions: usize,
    pub history_episodes: usize,
    pub history_decisions: usize,
    pub displayed_episodes: usize,
    pub displayed_segments: usize,
    pub retained_bytes: usize,
}
#[derive(Debug)]
pub struct SelectedEpisode {
    pub key: CacheKey,
    pub episode: Episode,
    pub reward_trace: Vec<[f64; 2]>,
    pub path: Path2,
    pub retained_bytes: usize,
}
#[derive(Debug)]
pub struct PreparedFrame {
    pub state: Arc<PreparedState>,
    pub selected: Arc<SelectedEpisode>,
    pub background: Arc<PreparedProjection>,
    pub action: ActionAxis,
    pub projection: Projection,
    pub action_trace: Vec<ActionSample>,
    pub geometry3: Vec<Segment3>,
    pub retained_bytes: usize,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub submitted: u64,
    pub source_reads: u64,
    /// Successful reader results measured with the existing conservative chunk
    /// retained-charge metric, not compressed bytes or OS physical disk I/O.
    pub source_read_bytes: u64,
    /// Cache-served loads; selected-frame Arc reuse needs no load at all.
    pub decoded_hits: u64,
    /// Current worker-only cache charge, including keys/tree/path metadata.
    pub decoded_cache_bytes: u64,
    pub state_builds: u64,
    pub selected_builds: u64,
    pub action_builds: u64,
    pub projection_builds: u64,
    pub cancelled: u64,
}
#[derive(Default)]
struct Counts {
    source_reads: AtomicU64,
    source_read_bytes: AtomicU64,
    decoded_hits: AtomicU64,
    decoded_cache_bytes: AtomicU64,
    state_builds: AtomicU64,
    selected_builds: AtomicU64,
    action_builds: AtomicU64,
    projection_builds: AtomicU64,
    cancelled: AtomicU64,
}
struct Job {
    serial: u64,
    request: Request,
}
/// Exact screen transform captured at click time. A resize or axis change
/// invalidates its answer even if the underlying episode is unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitView {
    pub axes: Axes,
    pub bounds: [[f64; 2]; 2],
    pub chart: eframe::egui::Rect,
}
struct HitJob {
    serial: u64,
    hit_serial: u64,
    frame: Weak<PreparedFrame>,
    sources: Arc<SourceSet>,
    view: HitView,
    pointer: eframe::egui::Pos2,
}
enum WorkJob {
    Prepare(Job),
    Episode(Box<episode::EpisodeJob>),
    Hit(HitJob),
    Availability(AvailabilityJob),
}
#[derive(Clone)]
struct AvailabilityJob {
    serial: u64,
    availability_serial: u64,
    frame: Weak<PreparedFrame>,
    sources: Weak<SourceSet>,
}
impl AvailabilityJob {
    fn matches(&self, serial: u64, frame: &Arc<PreparedFrame>, sources: &Arc<SourceSet>) -> bool {
        self.serial == serial
            && self.frame.ptr_eq(&Arc::downgrade(frame))
            && self.sources.ptr_eq(&Arc::downgrade(sources))
    }
    fn current(&self, shared: &Shared) -> bool {
        self.serial == shared.serial.load(Ordering::Acquire)
            && self.availability_serial == shared.availability_serial.load(Ordering::Acquire)
    }
}
struct AvailabilityAnswer {
    job: AvailabilityJob,
    error: Option<String>,
}
struct HitAnswer {
    serial: u64,
    hit_serial: u64,
    frame: Weak<PreparedFrame>,
    view: HitView,
    step: Option<usize>,
}
enum WorkAnswer {
    Prepare(Answer),
    Episode(episode::EpisodeAnswer),
    Hit(HitAnswer),
}
struct Answer {
    serial: u64,
    result: Result<Arc<PreparedFrame>>,
}
#[derive(Default)]
struct Mailbox {
    desired: Option<WorkJob>,
    answer: Option<WorkAnswer>,
    active: bool,
    active_hit: bool,
    availability_desired: Option<AvailabilityJob>,
    availability_answer: Option<AvailabilityAnswer>,
    active_availability: Option<AvailabilityJob>,
    clear: bool,
    stop: bool,
}
struct Shared {
    mailbox: Mutex<Mailbox>,
    changed: Condvar,
    serial: AtomicU64,
    hit_serial: AtomicU64,
    availability_serial: AtomicU64,
    counts: Counts,
    decode_gate: DecodeGate,
}
type Reader = dyn Fn(&ChunkSource, usize) -> Result<Option<Chunk>> + Send + Sync;
pub struct DynamicsWorker {
    shared: Arc<Shared>,
    limits: WorkLimits,
    wanted: Option<RequestKey>,
    frame: Option<Arc<PreparedFrame>>,
    episode_request: Option<EpisodeRequest>,
    episode_frame: Option<Arc<EpisodeFrame>>,
    episode_leases: Vec<Weak<EpisodeFrame>>,
    leases: Vec<Weak<PreparedFrame>>,
    state_leases: Vec<Weak<PreparedState>>,
    history_leases: Vec<Weak<Coverage>>,
    selected_leases: Vec<Weak<SelectedEpisode>>,
    projection_leases: Vec<Weak<PreparedProjection>>,
    error: Option<String>,
    submitted: u64,
    envelope: usize,
}
impl DynamicsWorker {
    pub fn new(limits: WorkLimits, wake: impl Fn() + Send + 'static) -> Result<Self> {
        Self::with_reader(limits, wake, ChunkSource::load_if_within_charge)
    }
    /// Reader injection for deterministic admission/cancellation tests. Reader
    /// must honor the supplied retained-charge cap just like ChunkSource does.
    pub fn with_reader(
        limits: WorkLimits,
        wake: impl Fn() + Send + 'static,
        reader: impl Fn(&ChunkSource, usize) -> Result<Option<Chunk>> + Send + Sync + 'static,
    ) -> Result<Self> {
        Self::with_io(limits, wake, reader, ChunkSource::check_available)
    }
    fn with_io(
        limits: WorkLimits,
        wake: impl Fn() + Send + 'static,
        reader: impl Fn(&ChunkSource, usize) -> Result<Option<Chunk>> + Send + Sync + 'static,
        check_source: impl Fn(&ChunkSource) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        Self::with_source_checks(
            limits,
            wake,
            reader,
            check_source,
            ChunkSource::check_available,
        )
    }
    fn with_source_checks(
        limits: WorkLimits,
        wake: impl Fn() + Send + 'static,
        reader: impl Fn(&ChunkSource, usize) -> Result<Option<Chunk>> + Send + Sync + 'static,
        check_source: impl Fn(&ChunkSource) -> Result<()> + Send + 'static,
        check_reference_source: impl Fn(&ChunkSource) -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        let envelope = limits.envelope_bytes()?;
        let shared = Arc::new(Shared {
            mailbox: Mutex::new(Mailbox::default()),
            changed: Condvar::new(),
            serial: AtomicU64::new(0),
            hit_serial: AtomicU64::new(0),
            availability_serial: AtomicU64::new(0),
            counts: Counts::default(),
            decode_gate: Arc::new(DecodeBudget {
                read_limit: limits.max_read_bytes.min(DYNAMICS_READ),
                ..Default::default()
            }),
        });
        let worker_shared = shared.clone();
        let worker_limits = limits.clone();
        thread::spawn(move || {
            worker_loop(
                worker_shared,
                worker_limits,
                Box::new(reader),
                check_source,
                check_reference_source,
                wake,
            )
        });
        Ok(Self {
            shared,
            limits,
            wanted: None,
            frame: None,
            episode_request: None,
            episode_frame: None,
            episode_leases: Vec::new(),
            leases: Vec::new(),
            state_leases: Vec::new(),
            history_leases: Vec::new(),
            selected_leases: Vec::new(),
            projection_leases: Vec::new(),
            error: None,
            submitted: 0,
            envelope,
        })
    }
    pub fn request(&mut self, request: Request) -> Result<bool> {
        let key = request.key();
        if self.wanted.as_ref() == Some(&key) {
            return Ok(false);
        }
        request.validate(&self.limits)?;
        self.episode_request = None;
        self.episode_frame = None;
        self.episode_leases.retain(|w| w.strong_count() > 0);
        self.leases.retain(|w| w.strong_count() > 0);
        self.state_leases.retain(|w| w.strong_count() > 0);
        self.history_leases.retain(|w| w.strong_count() > 0);
        self.selected_leases.retain(|w| w.strong_count() > 0);
        self.projection_leases.retain(|w| w.strong_count() > 0);
        // At most two old externally held frames plus the next result. Client
        // retaining more receives explicit admission refusal, never hidden RSS.
        let lease_charge = self
            .leases
            .iter()
            .filter_map(Weak::upgrade)
            .map(|f| f.retained_bytes)
            .chain(
                self.state_leases
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|s| s.retained_bytes),
            )
            .chain(
                self.history_leases
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|history| history.retained_bytes),
            )
            .chain(
                self.selected_leases
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|s| s.retained_bytes),
            )
            .chain(
                self.projection_leases
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|p| p.retained_bytes),
            )
            .chain(
                self.episode_leases
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|f| f.retained_bytes),
            )
            .try_fold(0usize, |n, b| n.checked_add(b).ok_or_else(overflow))?;
        checked(
            self.leases.len() < 3
                && self.state_leases.len() < 3
                && self.history_leases.len() < 3
                && self.selected_leases.len() < 3
                && self.projection_leases.len() < 3
                && lease_charge <= self.limits.max_frame_bytes * 2,
            "release old Dynamics frame leases before requesting another frame",
        )?;
        let serial = self
            .shared
            .serial
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        self.shared.hit_serial.fetch_add(1, Ordering::AcqRel);
        cancel_availability(&self.shared, &mut mailbox);
        mailbox.desired = Some(WorkJob::Prepare(Job { serial, request }));
        mailbox.answer = None;
        self.wanted = Some(key);
        self.error = None;
        self.submitted += 1;
        self.shared.changed.notify_one();
        Ok(true)
    }
    pub fn decode_gate(&self) -> DecodeGate {
        self.shared.decode_gate.clone()
    }
    pub fn poll(&mut self) {
        self.poll_episode();
        let answer = {
            let mut mailbox = self.shared.mailbox.lock().unwrap();
            if matches!(mailbox.answer, Some(WorkAnswer::Prepare(_))) {
                match mailbox.answer.take() {
                    Some(WorkAnswer::Prepare(answer)) => Some(answer),
                    _ => unreachable!(),
                }
            } else {
                None
            }
        };
        if let Some(answer) = answer
            && answer.serial == self.shared.serial.load(Ordering::Acquire)
        {
            match answer.result {
                Ok(frame) => {
                    if !self
                        .history_leases
                        .iter()
                        .any(|w| w.ptr_eq(&Arc::downgrade(&frame.state.coverage)))
                    {
                        self.history_leases
                            .push(Arc::downgrade(&frame.state.coverage));
                    }
                    if !self
                        .state_leases
                        .iter()
                        .any(|w| w.ptr_eq(&Arc::downgrade(&frame.state)))
                    {
                        self.state_leases.push(Arc::downgrade(&frame.state));
                    }
                    if !self
                        .selected_leases
                        .iter()
                        .any(|w| w.ptr_eq(&Arc::downgrade(&frame.selected)))
                    {
                        self.selected_leases.push(Arc::downgrade(&frame.selected));
                    }
                    if !self
                        .projection_leases
                        .iter()
                        .any(|w| w.ptr_eq(&Arc::downgrade(&frame.background)))
                    {
                        self.projection_leases
                            .push(Arc::downgrade(&frame.background));
                    }
                    self.leases.push(Arc::downgrade(&frame));
                    self.frame = Some(frame);
                    self.error = None;
                }
                Err(error) => {
                    self.frame = None;
                    self.error = Some(error.to_string().chars().take(512).collect());
                }
            }
        }
    }
    pub fn frame(&self) -> Option<&PreparedFrame> {
        self.frame.as_deref()
    }
    fn availability_eligible(
        &self,
        frame: &Arc<PreparedFrame>,
        sources: &Arc<SourceSet>,
        m: &Mailbox,
    ) -> bool {
        self.frame
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, frame))
            && frame.state.key.source_set == sources.identity
            && (!m.active || m.active_hit || m.active_availability.is_some())
            && !matches!(m.desired, Some(WorkJob::Prepare(_)))
            && !matches!(m.answer, Some(WorkAnswer::Prepare(_)))
            && !m.clear
            && !m.stop
    }
    /// Queue a coalesced metadata check without retaining a frame or source set.
    /// False means ineligible or already outstanding, never source failure.
    pub fn request_availability(
        &mut self,
        frame: &Arc<PreparedFrame>,
        sources: &Arc<SourceSet>,
    ) -> bool {
        let mut m = self.shared.mailbox.lock().unwrap();
        if !self.availability_eligible(frame, sources, &m) {
            return false;
        }
        let serial = self.shared.serial.load(Ordering::Acquire);
        if m.availability_desired
            .iter()
            .chain(m.active_availability.iter())
            .chain(m.availability_answer.iter().map(|answer| &answer.job))
            .any(|job| job.current(&self.shared) && job.matches(serial, frame, sources))
        {
            return false;
        }
        cancel_availability(&self.shared, &mut m);
        m.availability_desired = Some(AvailabilityJob {
            serial,
            availability_serial: self.shared.availability_serial.load(Ordering::Acquire),
            frame: Arc::downgrade(frame),
            sources: Arc::downgrade(sources),
        });
        self.shared.changed.notify_one();
        true
    }
    pub fn poll_availability(
        &mut self,
        frame: &Arc<PreparedFrame>,
        sources: &Arc<SourceSet>,
    ) -> Option<std::result::Result<(), String>> {
        let mut m = self.shared.mailbox.lock().unwrap();
        let answer = m.availability_answer.take()?;
        (self.availability_eligible(frame, sources, &m)
            && answer.job.current(&self.shared)
            && answer
                .job
                .matches(self.shared.serial.load(Ordering::Acquire), frame, sources))
        .then(|| answer.error.map_or(Ok(()), Err))
    }
    /// Enqueue a scan without copying an episode or allocating observation
    /// geometry. Weak frame leases keep obsolete queued clicks from retaining
    /// old frames; the one active scan uses the same worker as preparation.
    pub fn request_hit(
        &mut self,
        frame: &Arc<PreparedFrame>,
        sources: Arc<SourceSet>,
        view: HitView,
        pointer: eframe::egui::Pos2,
    ) -> Result<()> {
        checked(
            self.frame
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, frame))
                && frame.state.key.source_set == sources.identity
                && view.axes == frame.state.key.axes
                && view.bounds == frame.state.reference.grid.grid.bounds(),
            "stale Dynamics click source/view",
        )?;
        checked(
            pointer.is_finite()
                && view.chart.is_finite()
                && view.chart.width() > 0.
                && view.chart.height() > 0.,
            "invalid Dynamics click transform",
        )?;
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        checked(
            (!mailbox.active || mailbox.active_hit || mailbox.active_availability.is_some())
                && !matches!(mailbox.desired, Some(WorkJob::Prepare(_)))
                && !matches!(mailbox.answer, Some(WorkAnswer::Prepare(_))),
            "Dynamics frame is still preparing",
        )?;
        cancel_availability(&self.shared, &mut mailbox);
        let hit_serial = self
            .shared
            .hit_serial
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        mailbox.desired = Some(WorkJob::Hit(HitJob {
            serial: self.shared.serial.load(Ordering::Acquire),
            hit_serial,
            frame: Arc::downgrade(frame),
            sources,
            view,
            pointer,
        }));
        mailbox.answer = None;
        self.shared.changed.notify_one();
        Ok(())
    }
    pub fn cancel_hit(&mut self) {
        self.shared.hit_serial.fetch_add(1, Ordering::AcqRel);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        if matches!(mailbox.desired, Some(WorkJob::Hit(_))) {
            mailbox.desired = None;
        }
        if matches!(mailbox.answer, Some(WorkAnswer::Hit(_))) {
            mailbox.answer = None;
        }
    }
    pub fn poll_hit(&mut self, frame: &Arc<PreparedFrame>, view: HitView) -> Option<usize> {
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        if !matches!(mailbox.answer, Some(WorkAnswer::Hit(_))) {
            return None;
        }
        let Some(WorkAnswer::Hit(answer)) = mailbox.answer.take() else {
            unreachable!()
        };
        (answer.serial == self.shared.serial.load(Ordering::Acquire)
            && answer.hit_serial == self.shared.hit_serial.load(Ordering::Acquire)
            && answer.frame.ptr_eq(&Arc::downgrade(frame))
            && answer.view == view
            && self
                .frame
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(current, frame)))
        .then_some(answer.step)
        .flatten()
    }
    pub fn frame_arc(&self) -> Option<Arc<PreparedFrame>> {
        self.frame.clone()
    }
    pub fn pending(&self) -> bool {
        let m = self.shared.mailbox.lock().unwrap();
        (m.active && m.active_availability.is_none()) || m.desired.is_some() || m.answer.is_some()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn envelope_bytes(&self) -> usize {
        self.envelope
    }
    pub fn counters(&self) -> Counters {
        let c = &self.shared.counts;
        Counters {
            submitted: self.submitted,
            source_reads: c.source_reads.load(Ordering::Relaxed),
            source_read_bytes: c.source_read_bytes.load(Ordering::Relaxed),
            decoded_hits: c.decoded_hits.load(Ordering::Relaxed),
            decoded_cache_bytes: c.decoded_cache_bytes.load(Ordering::Relaxed),
            state_builds: c.state_builds.load(Ordering::Relaxed),
            selected_builds: c.selected_builds.load(Ordering::Relaxed),
            action_builds: c.action_builds.load(Ordering::Relaxed),
            projection_builds: c.projection_builds.load(Ordering::Relaxed),
            cancelled: c.cancelled.load(Ordering::Relaxed),
        }
    }
    /// Failure keys stay stable until retry, a new source generation, or retire.
    pub fn retry(&mut self) {
        self.retire();
    }
    pub fn retire(&mut self) {
        self.shared.serial.fetch_add(1, Ordering::AcqRel);
        self.shared.hit_serial.fetch_add(1, Ordering::AcqRel);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        cancel_availability(&self.shared, &mut mailbox);
        mailbox.desired = None;
        mailbox.answer = None;
        mailbox.clear = true;
        self.frame = None;
        self.episode_frame = None;
        self.episode_request = None;
        self.wanted = None;
        self.error = None;
        self.shared.changed.notify_one();
    }
}
fn cancel_availability(shared: &Shared, mailbox: &mut Mailbox) {
    shared.availability_serial.fetch_add(1, Ordering::AcqRel);
    mailbox.availability_desired = None;
    mailbox.availability_answer = None;
}
impl Drop for DynamicsWorker {
    fn drop(&mut self) {
        self.retire();
        let mut m = self.shared.mailbox.lock().unwrap();
        m.stop = true;
        self.shared.changed.notify_one();
    }
}
#[derive(Default)]
struct Cache {
    state: Option<Arc<PreparedState>>,
    selected: Option<Arc<SelectedEpisode>>,
    projection: Option<Arc<PreparedProjection>>,
    source_set: Option<Arc<SourceIdentity>>,
    decoded: DecodedCache,
}

#[derive(Default)]
struct DecodedCache {
    episodes: BTreeMap<(usize, CacheKey), Episode>,
    bytes: usize,
}
impl DecodedCache {
    fn charge(key: &CacheKey, episode: &Episode) -> Result<usize> {
        // Includes tree nodes, key/path/Arc ownership and allocator overhead.
        episode
            .retained_bytes()
            .checked_add(512)
            .and_then(|n| {
                key.root
                    .as_os_str()
                    .len()
                    .checked_mul(8)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(overflow)
    }
    fn retain(&mut self, sources: &SourceSet) -> Result<()> {
        let session = Arc::as_ptr(&sources.root) as usize;
        self.episodes.retain(|(root, key), _| {
            *root == session && sources.sources.iter().any(|s| s.key == *key)
        });
        self.bytes = self
            .episodes
            .iter()
            .try_fold(0usize, |total, ((_, key), episode)| {
                total
                    .checked_add(Self::charge(key, episode)?)
                    .ok_or_else(overflow)
            })?;
        Ok(())
    }
}
fn worker_loop(
    shared: Arc<Shared>,
    limits: WorkLimits,
    reader: Box<Reader>,
    check_source: impl Fn(&ChunkSource) -> Result<()>,
    check_reference_source: impl Fn(&ChunkSource) -> Result<()>,
    wake: impl Fn(),
) {
    let mut cache = Cache::default();
    loop {
        let job = {
            let mut m = shared.mailbox.lock().unwrap();
            loop {
                if m.stop {
                    return;
                }
                if m.clear {
                    cache = Cache::default();
                    shared
                        .counts
                        .decoded_cache_bytes
                        .store(0, Ordering::Relaxed);
                    m.clear = false;
                }
                if let Some(job) = m.desired.take() {
                    m.active = true;
                    m.active_hit = matches!(job, WorkJob::Hit(_));
                    break job;
                }
                if let Some(job) = m.availability_desired.take() {
                    m.active = true;
                    m.active_availability = Some(job.clone());
                    break WorkJob::Availability(job);
                }
                m = shared.changed.wait(m).unwrap();
            }
        };
        let job = match job {
            WorkJob::Prepare(job) => job,
            WorkJob::Episode(job) => {
                cache = Cache::default();
                shared
                    .counts
                    .decoded_cache_bytes
                    .store(0, Ordering::Relaxed);
                let result = episode::execute(&job, &shared, &limits, reader.as_ref());
                let mut mailbox = shared.mailbox.lock().unwrap();
                mailbox.active = false;
                if shared.serial.load(Ordering::Acquire) == job.serial
                    && !mailbox.clear
                    && !mailbox.stop
                {
                    mailbox.answer = Some(WorkAnswer::Episode(episode::EpisodeAnswer {
                        serial: job.serial,
                        result,
                    }));
                    drop(mailbox);
                    wake();
                } else {
                    shared.counts.cancelled.fetch_add(1, Ordering::Relaxed);
                }
                continue;
            }
            WorkJob::Availability(job) => {
                // The scanner drops both strong upgrades before publication.
                let result = scan_availability(&job, &shared, &check_source);
                let mut mailbox = shared.mailbox.lock().unwrap();
                mailbox.active = false;
                mailbox.active_availability = None;
                if let Some(result) = result
                    && job.current(&shared)
                    && !mailbox.clear
                    && !mailbox.stop
                {
                    mailbox.availability_answer = Some(AvailabilityAnswer {
                        job,
                        error: result.err(),
                    });
                    drop(mailbox);
                    wake();
                }
                continue;
            }
            WorkJob::Hit(job) => {
                let step = exact_hit(&job, &shared, &check_reference_source)
                    .ok()
                    .flatten();
                let mut mailbox = shared.mailbox.lock().unwrap();
                mailbox.active = false;
                mailbox.active_hit = false;
                if shared.serial.load(Ordering::Acquire) == job.serial
                    && shared.hit_serial.load(Ordering::Acquire) == job.hit_serial
                    && !mailbox.stop
                    && !mailbox.clear
                {
                    mailbox.answer = Some(WorkAnswer::Hit(HitAnswer {
                        serial: job.serial,
                        hit_serial: job.hit_serial,
                        frame: job.frame,
                        view: job.view,
                        step,
                    }));
                    drop(mailbox);
                    wake();
                }
                continue;
            }
        };
        let mut context = Context {
            shared: &shared,
            serial: job.serial,
            limits: &limits,
            reader: reader.as_ref(),
            check_reference_source: &check_reference_source,
            read_bytes: 0,
        };
        let result = build(&job.request, &mut cache, &mut context);
        let mut m = shared.mailbox.lock().unwrap();
        m.active = false;
        if shared.serial.load(Ordering::Acquire) == job.serial && !m.clear && !m.stop {
            m.answer = Some(WorkAnswer::Prepare(Answer {
                serial: job.serial,
                result: result.map(Arc::new),
            }));
            drop(m);
            wake();
        } else {
            shared.counts.cancelled.fetch_add(1, Ordering::Relaxed);
        }
    }
}
fn scan_availability(
    job: &AvailabilityJob,
    shared: &Shared,
    mut check_source: impl FnMut(&ChunkSource) -> Result<()>,
) -> Option<std::result::Result<(), String>> {
    if !job.current(shared) {
        return None;
    }
    let frame = job.frame.upgrade()?;
    let sources = job.sources.upgrade()?;
    for (handles, fixed) in [
        (&sources.handles, false),
        (&frame.state.reference.sources, true),
    ] {
        for source in handles {
            if !job.current(shared) {
                return None;
            }
            let result = check_source(source);
            if !job.current(shared) {
                return None;
            }
            if let Err(error) = result {
                let text = if fixed {
                    format!("Dynamics fixed-reference source unavailable: {error}")
                } else {
                    error.to_string()
                };
                return Some(Err(text.chars().take(512).collect()));
            }
        }
    }
    job.current(shared).then_some(Ok(()))
}
fn exact_hit(
    job: &HitJob,
    shared: &Shared,
    check_reference_source: impl Fn(&ChunkSource) -> Result<()>,
) -> Result<Option<usize>> {
    let check = || {
        checked(
            shared.serial.load(Ordering::Acquire) == job.serial
                && shared.hit_serial.load(Ordering::Acquire) == job.hit_serial,
            "Dynamics click superseded",
        )
    };
    check()?;
    let Some(frame) = job.frame.upgrade() else {
        return Ok(None);
    };
    let mut best = (f32::INFINITY, 0usize);
    for t in 0..=frame.selected.episode.transitions() {
        if t % 1024 == 0 {
            check()?;
        }
        let xy = frame.selected.episode.project(t, job.view.axes)?;
        let [x, y] = job.view.bounds;
        let at = eframe::egui::pos2(
            eframe::egui::lerp(
                job.view.chart.left()..=job.view.chart.right(),
                ((xy[0] - x[0]) / (x[1] - x[0])) as f32,
            ),
            eframe::egui::lerp(
                job.view.chart.bottom()..=job.view.chart.top(),
                ((xy[1] - y[0]) / (y[1] - y[0])) as f32,
            ),
        );
        let distance = at.distance_sq(job.pointer);
        if distance < best.0 {
            best = (distance, t);
        }
    }
    // Include final observations, keep earliest equal-distance index, and retain
    // the original strict 15-pixel screen-space threshold exactly.
    for source in &job.sources.handles {
        check()?;
        source.check_available()?;
    }
    frame
        .state
        .reference
        .check_available_with(check, check_reference_source)?;
    check()?;
    Ok((best.0 < 225.).then_some(best.1))
}
struct Context<'a> {
    shared: &'a Shared,
    serial: u64,
    limits: &'a WorkLimits,
    reader: &'a Reader,
    check_reference_source: &'a dyn Fn(&ChunkSource) -> Result<()>,
    read_bytes: usize,
}
impl Context<'_> {
    fn check(&self) -> Result<()> {
        checked(
            self.shared.serial.load(Ordering::Acquire) == self.serial,
            "Dynamics preparation superseded",
        )
    }
    fn load(
        &mut self,
        sources: &SourceSet,
        index: usize,
        decoded: &mut DecodedCache,
    ) -> Result<Episode> {
        self.check()?;
        let handle = &sources.handles[index];
        handle.check_available()?;
        let source = &sources.sources[index];
        let cache_key = (Arc::as_ptr(&sources.root) as usize, source.key.clone());
        if let Some(episode) = decoded.episodes.get(&cache_key) {
            validate_decoded(episode, source)?;
            self.shared
                .counts
                .decoded_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(episode.clone());
        }
        let _decode = self.shared.decode_gate.lock.lock().unwrap();
        self.check()?;
        let remaining = self
            .limits
            .max_total_read_bytes
            .saturating_sub(self.read_bytes);
        checked(remaining > 0, "Dynamics source reads exceed work budget")?;
        self.shared
            .counts
            .source_reads
            .fetch_add(1, Ordering::Relaxed);
        let chunk = (self.reader)(handle, self.limits.max_read_bytes.min(remaining))?
            .ok_or_else(|| Error::Shape("Dynamics source exceeds read budget".into()))?;
        self.shared
            .counts
            .source_read_bytes
            .fetch_add(chunk.retained_charge() as u64, Ordering::Relaxed);
        self.read_bytes = self
            .read_bytes
            .checked_add(chunk.retained_charge())
            .ok_or_else(overflow)?;
        checked(
            self.read_bytes <= self.limits.max_total_read_bytes,
            "Dynamics source reads exceed work budget",
        )?;
        self.check()?;
        let episode = prepare_episode(&chunk, &sources.manifest, &self.limits.science())?;
        validate_decoded(&episode, source)?;
        self.check()?;
        let charge = DecodedCache::charge(&source.key, &episode)?;
        // Stable first-fit retention avoids cyclic eviction between the two
        // ordered passes. Non-fitting episodes remain streaming scratch only.
        if charge <= self.limits.max_episode_bytes.saturating_sub(decoded.bytes) {
            decoded.episodes.insert(cache_key, episode.clone());
            decoded.bytes += charge;
            self.shared
                .counts
                .decoded_cache_bytes
                .store(decoded.bytes as u64, Ordering::Relaxed);
        }
        Ok(episode)
    }
}
fn validate_decoded(episode: &Episode, source: &TrajectorySource) -> Result<()> {
    checked(
        episode.context().identity.learner_update_revision == source.revision
            && episode.context().identity.model_role == source.role
            && episode.record().id == source.id
            && ProtocolKey::for_recorded_trajectory(episode.record(), Some(episode.context()))?
                == source.protocol
            && episode.transitions() == source.transitions,
        "Dynamics decoded source identity differs from catalog",
    )
}

/// Floor-spaced ordered sample indices, including both endpoints. Each sampled
/// path edge uses [s_t,s_(t+1)]; unsampled edges are never joined across a gap.
fn samples(count: usize, cap: usize) -> Vec<usize> {
    let n = count.min(cap);
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    (0..n)
        .map(|i| ((i as u128 * (count - 1) as u128) / (n - 1) as u128) as usize)
        .collect()
}
fn path(episode: &Episode, key: CacheKey, axes: Axes, cap: usize) -> Result<Path2> {
    let mut segments = Vec::with_capacity(episode.transitions().min(cap));
    for t in samples(episode.transitions(), cap) {
        segments.push(Segment2 {
            t,
            points: [episode.project(t, axes)?, episode.project(t + 1, axes)?],
        });
    }
    Ok(Path2 {
        key,
        segments,
        start: episode.project(0, axes)?,
        end: episode.project(episode.transitions(), axes)?,
        transitions: episode.transitions(),
    })
}
fn projection_point(episode: &Episode, req: &Request, t: usize) -> Result<[f64; 3]> {
    match req.projection {
        Projection::State(axes) => episode.project_state3(t, axes),
        Projection::StateAction => episode
            .project_state_action(t, req.axes, req.action)?
            .ok_or_else(|| Error::State("terminal state has no action".into())),
        Projection::None => Err(Error::State("no 3D projection requested".into())),
    }
}
fn projection_edges(projection: Projection, decisions: usize) -> usize {
    match projection {
        Projection::None => 0,
        Projection::State(_) => decisions,
        Projection::StateAction => decisions.saturating_sub(1),
    }
}
fn path3(episode: &Episode, key: CacheKey, req: &Request, cap: usize) -> Result<Path3> {
    let edges = projection_edges(req.projection, episode.transitions());
    let start = projection_point(episode, req, 0)?;
    let end = projection_point(episode, req, edges)?;
    let mut segments = Vec::with_capacity(edges.min(cap));
    for t in samples(edges, cap) {
        segments.push(Segment3 {
            t,
            points: [
                projection_point(episode, req, t)?,
                projection_point(episode, req, t + 1)?,
            ],
        });
    }
    Ok(Path3 {
        key,
        segments,
        start,
        end,
    })
}
fn projection_bounds(reference: &Reference, req: &Request) -> Result<[[f64; 2]; 3]> {
    let xy = reference.grid.grid.bounds();
    let state_bound = |axis| {
        if axis == req.axes.x {
            Some(xy[0])
        } else if axis == req.axes.y {
            Some(xy[1])
        } else {
            reference.state_bounds(axis)
        }
    };
    let bounds = match req.projection {
        Projection::None => return Ok([xy[0], xy[1], [0., 1.]]),
        Projection::State(axes) => axes.map(state_bound),
        Projection::StateAction => [
            Some(xy[0]),
            Some(xy[1]),
            match req.action {
                ActionAxis::Continuous(j) => reference.action_bounds(j),
                _ => None,
            },
        ],
    };
    let mut result = [[0.; 2]; 3];
    for (out, bound) in result.iter_mut().zip(bounds) {
        let [lo, hi] = bound.ok_or_else(|| Error::Shape("missing 3D reference domain".into()))?;
        *out = if lo == hi {
            let pad = (lo.abs() * 0.05).max(0.5);
            [lo - pad, hi + pad]
        } else {
            [lo, hi]
        };
        checked(
            out[0].is_finite()
                && out[1].is_finite()
                && out[0] < out[1]
                && (out[1] - out[0]).is_finite(),
            "invalid 3D reference domain",
        )?;
    }
    Ok(result)
}
fn finish_projection(
    req: &Request,
    state: &PreparedState,
    paths: Vec<Path3>,
    ctx: &Context<'_>,
) -> Result<Arc<PreparedProjection>> {
    let displayed_segments = paths.iter().map(|p| p.segments.len()).sum::<usize>();
    let retained_bytes = paths
        .iter()
        .try_fold(
            paths.capacity().checked_mul(512).ok_or_else(overflow)?,
            |n, p| {
                n.checked_add(
                    p.segments
                        .capacity()
                        .checked_mul(std::mem::size_of::<Segment3>())
                        .ok_or_else(overflow)?,
                )
                .ok_or_else(overflow)
            },
        )?
        .checked_add(req.sources.retained_bytes)
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(overflow)?;
    checked(
        retained_bytes <= ctx.limits.max_frame_bytes,
        "Dynamics projection exceeds frame budget",
    )?;
    let eligible_segments = req
        .sources
        .sources
        .iter()
        .filter(|s| s.revision == req.revision)
        .map(|s| projection_edges(req.projection, s.transitions))
        .sum();
    let result = Arc::new(PreparedProjection {
        state: state.key.clone(),
        revision: req.revision,
        projection: req.projection,
        action: (req.projection == Projection::StateAction).then_some(req.action),
        bounds: projection_bounds(&state.reference, req)?,
        paths,
        eligible_episodes: if req.projection == Projection::None {
            0
        } else {
            state.population_episodes
        },
        eligible_segments,
        displayed_segments,
        retained_bytes,
    });
    ctx.shared
        .counts
        .projection_builds
        .fetch_add(1, Ordering::Relaxed);
    Ok(result)
}
fn build(req: &Request, cache: &mut Cache, ctx: &mut Context<'_>) -> Result<PreparedFrame> {
    ctx.check()?;
    cache.decoded.retain(&req.sources)?;
    ctx.shared
        .counts
        .decoded_cache_bytes
        .store(cache.decoded.bytes as u64, Ordering::Relaxed);
    let mut normalized = req.clone();
    if normalized.reference.is_none() {
        normalized.reference = cache
            .state
            .as_ref()
            .map(|s| s.reference.clone())
            .filter(|r| r.matches(req));
    }
    let req = &normalized;
    if let Some(reference) = &req.reference {
        reference.check_available_with(|| ctx.check(), ctx.check_reference_source)?;
    }
    let key = req.key().state;
    let selected_index = req
        .sources
        .sources
        .iter()
        .position(|s| s.key == req.selected)
        .expect("validated selected source");
    let selected = if let Some(selected) = cache.selected.as_ref().filter(|s| {
        s.key == req.selected
            && cache.source_set.as_ref() == Some(&req.sources.identity)
            && cache
                .state
                .as_ref()
                .is_some_and(|st| st.key.axes == req.axes)
    }) {
        selected.clone()
    } else {
        let episode = ctx.load(&req.sources, selected_index, &mut cache.decoded)?;
        let mut reward_trace =
            Vec::with_capacity(episode.transitions().min(ctx.limits.max_trace_points));
        for t in samples(episode.transitions(), ctx.limits.max_trace_points) {
            reward_trace.push([t as f64, episode.step(t)?.reward.expect("decision reward")]);
        }
        let path = path(
            &episode,
            req.selected.clone(),
            req.axes,
            ctx.limits.max_segments,
        )?;
        let retained_bytes = episode.retained_bytes()
            + reward_trace.len() * 16
            + path.segments.len() * std::mem::size_of::<Segment2>()
            + 4096;
        checked(
            retained_bytes <= ctx.limits.max_frame_bytes,
            "Dynamics selected episode exceeds frame budget",
        )?;
        ctx.shared
            .counts
            .selected_builds
            .fetch_add(1, Ordering::Relaxed);
        Arc::new(SelectedEpisode {
            key: req.selected.clone(),
            episode,
            reward_trace,
            path,
            retained_bytes,
        })
    };
    let mut initial_projection = None;
    let state = if let Some(state) = cache
        .state
        .as_ref()
        .filter(|s| s.key == key && s.revision == req.revision)
    {
        state.clone()
    } else {
        let (state, paths) = build_state(
            req,
            key,
            &selected,
            cache.state.as_deref(),
            &mut cache.decoded,
            ctx,
        )?;
        initial_projection = Some(paths);
        Arc::new(state)
    };
    let background = if let Some(paths) = initial_projection {
        finish_projection(req, &state, paths, ctx)?
    } else if let Some(projection) = cache.projection.as_ref().filter(|p| p.matches(&state, req)) {
        projection.clone()
    } else {
        let mut paths = Vec::with_capacity(if req.projection == Projection::None {
            0
        } else {
            state.trajectories.len()
        });
        if req.projection != Projection::None {
            let cap = ctx.limits.max_segments / state.trajectories.len().max(1);
            for path in &state.trajectories {
                ctx.check()?;
                let i = req
                    .sources
                    .sources
                    .iter()
                    .position(|s| s.key == path.key)
                    .expect("displayed source");
                let episode = if path.key == selected.key {
                    selected.episode.clone()
                } else {
                    ctx.load(&req.sources, i, &mut cache.decoded)?
                };
                paths.push(path3(&episode, path.key.clone(), req, cap)?);
            }
        }
        finish_projection(req, &state, paths, ctx)?
    };
    ctx.check()?;
    selected.episode.action_label(req.action)?;
    let mut action_trace = Vec::with_capacity(
        selected
            .episode
            .transitions()
            .min(ctx.limits.max_trace_points),
    );
    for t in samples(selected.episode.transitions(), ctx.limits.max_trace_points) {
        action_trace.push(ActionSample {
            t,
            value: selected
                .episode
                .action_value(t, req.action)?
                .expect("decision action"),
        });
    }
    let geometry3 = if req.projection == Projection::None {
        Vec::new()
    } else {
        path3(
            &selected.episode,
            selected.key.clone(),
            req,
            ctx.limits.max_segments,
        )?
        .segments
    };
    let retained_bytes = state
        .retained_bytes
        .checked_add(selected.retained_bytes)
        .and_then(|n| n.checked_add(background.retained_bytes))
        .and_then(|n| n.checked_add(action_trace.len() * std::mem::size_of::<ActionSample>()))
        .and_then(|n| n.checked_add(geometry3.len() * std::mem::size_of::<Segment3>()))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(overflow)?;
    checked(
        retained_bytes <= ctx.limits.max_frame_bytes,
        "Dynamics prepared frame exceeds budget",
    )?;
    // Recheck *every* contributing immutable source before publication, including
    // cached layers. This is bounded by source count and never runs in paint.
    for source in &req.sources.handles {
        ctx.check()?;
        source.check_available()?;
    }
    ctx.check()?;
    state
        .reference
        .check_available_with(|| ctx.check(), ctx.check_reference_source)?;
    ctx.shared
        .counts
        .action_builds
        .fetch_add(1, Ordering::Relaxed);
    cache.projection = Some(background.clone());
    cache.state = Some(state.clone());
    cache.selected = Some(selected.clone());
    cache.source_set = Some(req.sources.identity.clone());
    Ok(PreparedFrame {
        background,
        state,
        selected,
        action: req.action,
        projection: req.projection,
        action_trace,
        geometry3,
        retained_bytes,
    })
}

fn build_state(
    req: &Request,
    mut key: StateKey,
    selected: &SelectedEpisode,
    previous: Option<&PreparedState>,
    decoded: &mut DecodedCache,
    ctx: &mut Context<'_>,
) -> Result<(PreparedState, Vec<Path3>)> {
    let reference = if let Some(reference) = &req.reference {
        reference.clone()
    } else {
        // Every already-admitted compatible episode fixes the domain. Decode
        // reuse is bounded independently of this complete frozen population.
        let first = &req.sources.sources[0];
        let provenance_bytes = req
            .sources
            .retained_bytes
            .checked_add(
                req.sources
                    .sources
                    .len()
                    .checked_mul(
                        std::mem::size_of::<CacheKey>() + std::mem::size_of::<ChunkSource>(),
                    )
                    .ok_or_else(overflow)?,
            )
            .and_then(|n| n.checked_add(std::mem::size_of::<Reference>()))
            .ok_or_else(overflow)?;
        let mut extrema = ReferenceExtrema::new(
            &selected.episode,
            ctx.limits
                .max_source_set_bytes
                .saturating_sub(provenance_bytes),
        )?;
        for (i, source) in req.sources.sources.iter().enumerate() {
            ctx.check()?;
            let episode = if source.key == selected.key {
                selected.episode.clone()
            } else {
                ctx.load(&req.sources, i, decoded)?
            };
            episode.validate_axes(req.axes)?;
            extrema.add(&episode, || ctx.check())?;
        }
        let grid = extrema.grid(req.axes, req.cells, req.domain, &ctx.limits.science())?;
        let retained_bytes = provenance_bytes
            .checked_add(extrema.retained_bytes)
            .ok_or_else(overflow)?;
        checked(
            retained_bytes <= ctx.limits.max_source_set_bytes,
            "Dynamics reference exceeds ownership budget",
        )?;
        // Extrema and histogram/projection passes each have the pre-existing
        // allowance. Only actual loads are charged, never cache hits.
        ctx.read_bytes = 0;
        Arc::new(Reference {
            grid,
            source_keys: req
                .sources
                .sources
                .iter()
                .map(|source| source.key.clone())
                .collect(),
            id: ID.fetch_add(1, Ordering::Relaxed),
            root: req.sources.root.clone(),
            protocol: first.protocol.clone(),
            role: first.role.clone(),
            axes: req.axes,
            sources: req.sources.handles.clone(),
            domain: req.domain,
            extrema,
            retained_bytes,
        })
    };
    key.reference = Some(reference.id);
    // History identity excludes the selected revision. Keep only one immutable
    // population/grid history; current occupancy and paths remain revision-local.
    let history = previous.filter(|state| state.key == key);
    let grid = &reference.grid.grid;
    let cells = grid.cell_count();
    let mut counts = vec![0u64; cells];
    let mut included = 0u64;
    let mut excluded = 0u64;
    let mut first_seen = if history.is_none() {
        vec![None; cells]
    } else {
        Vec::new()
    };
    let mut points = Vec::with_capacity(if history.is_none() {
        req.sources.sources.len()
    } else {
        0
    });
    let mut revision_counts = if history.is_none() {
        vec![0u64; cells]
    } else {
        Vec::new()
    };
    let mut cumulative = 0usize;
    let mut revision_evidence = [0u64; 4]; // episodes, decisions, included, excluded
    let mut active_revision = None;
    let mut population_episodes = 0;
    let mut population_decisions = 0;
    let mut history_episodes = history.map_or(0, |state| state.history_episodes);
    let mut history_decisions = history.map_or(0, |state| state.history_decisions);
    let population: Vec<usize> = req
        .sources
        .sources
        .iter()
        .enumerate()
        .filter(|(_, s)| s.revision == req.revision)
        .map(|(i, _)| i)
        .collect();
    let display: Vec<usize> = samples(
        population.len(),
        ctx.limits
            .max_displayed_episodes
            .min(ctx.limits.max_segments / 2),
    )
    .into_iter()
    .map(|i| population[i])
    .collect();
    let per_path = ctx.limits.max_segments / display.len().max(1);
    let mut trajectories = Vec::with_capacity(display.len());
    let mut projection_paths = Vec::with_capacity(if req.projection == Projection::None {
        0
    } else {
        display.len()
    });
    let finish_revision = |revision: u64,
                           revision_counts: &mut [u64],
                           first_seen: &mut [Option<u64>],
                           points: &mut Vec<CoveragePoint>,
                           cumulative: &mut usize,
                           evidence: &mut [u64; 4]| {
        let mut occupied = 0;
        let mut new_cells = 0;
        for (visits, first) in revision_counts.iter_mut().zip(first_seen) {
            if *visits > 0 {
                occupied += 1;
                if first.is_none() {
                    *first = Some(revision);
                    new_cells += 1;
                }
            }
            *visits = 0;
        }
        *cumulative += new_cells;
        points.push(CoveragePoint {
            revision,
            occupied,
            new_cells,
            cumulative: *cumulative,
            episodes: evidence[0] as usize,
            decisions: evidence[1],
            included: evidence[2],
            excluded: evidence[3],
        });
        *evidence = [0; 4];
    };
    for (i, source) in req.sources.sources.iter().enumerate() {
        ctx.check()?;
        if history.is_some() && source.revision != req.revision {
            continue;
        }
        if history.is_none() && active_revision.is_some_and(|r| r != source.revision) {
            finish_revision(
                active_revision.unwrap(),
                &mut revision_counts,
                &mut first_seen,
                &mut points,
                &mut cumulative,
                &mut revision_evidence,
            );
        }
        active_revision = Some(source.revision);
        let episode = if source.key == selected.key {
            selected.episode.clone()
        } else {
            ctx.load(&req.sources, i, decoded)?
        };
        episode.validate_axes(req.axes)?;
        if history.is_none() {
            revision_evidence[0] += 1;
            revision_evidence[1] += episode.transitions() as u64;
            history_episodes += 1;
            history_decisions += episode.transitions();
        }
        if source.revision == req.revision {
            population_episodes += 1;
            population_decisions += episode.transitions();
        }
        for t in 0..episode.transitions() {
            if t % 1024 == 0 {
                ctx.check()?;
            }
            if let Some(bin) = grid.bin(episode.project(t, req.axes)?) {
                if history.is_none() {
                    revision_counts[bin] += 1;
                    revision_evidence[2] += 1;
                }
                if source.revision == req.revision {
                    counts[bin] += 1;
                    included += 1;
                }
            } else {
                if history.is_none() {
                    revision_evidence[3] += 1;
                }
                if source.revision == req.revision {
                    excluded += 1;
                }
            }
        }
        if display.binary_search(&i).is_ok() {
            trajectories.push(path(&episode, source.key.clone(), req.axes, per_path)?);
            if req.projection != Projection::None {
                projection_paths.push(path3(&episode, source.key.clone(), req, per_path)?);
            }
        }
    }
    if history.is_none()
        && let Some(revision) = active_revision
    {
        finish_revision(
            revision,
            &mut revision_counts,
            &mut first_seen,
            &mut points,
            &mut cumulative,
            &mut revision_evidence,
        );
    }
    let mass = counts
        .iter()
        .map(|&n| {
            if included == 0 {
                0.
            } else {
                n as f64 / included as f64
            }
        })
        .collect();
    let displayed_segments = trajectories.iter().map(|p| p.segments.len()).sum::<usize>();
    let coverage = history
        .map(|state| state.coverage.clone())
        .unwrap_or_else(|| {
            Arc::new(Coverage {
                retained_bytes: cells * std::mem::size_of::<Option<u64>>()
                    + points.capacity() * std::mem::size_of::<CoveragePoint>()
                    + 256,
                points,
                first_seen,
                display_grid_cells: cells,
            })
        });
    let retained_bytes = coverage.retained_bytes
        + cells * 16
        + displayed_segments * std::mem::size_of::<Segment2>()
        + trajectories.capacity() * 512
        + req.sources.retained_bytes
        + reference.retained_bytes
        + 4096;
    checked(
        retained_bytes <= ctx.limits.max_frame_bytes,
        "Dynamics state preparation exceeds frame budget",
    )?;
    ctx.shared
        .counts
        .state_builds
        .fetch_add(1, Ordering::Relaxed);
    Ok((
        PreparedState {
            key,
            revision: req.revision,
            reference,
            occupancy: Occupancy {
                counts,
                mass,
                included,
                excluded,
            },
            coverage,
            displayed_episodes: trajectories.len(),
            trajectories,
            population_episodes,
            population_decisions,
            history_episodes,
            history_decisions,
            displayed_segments,
            retained_bytes,
        },
        projection_paths,
    ))
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn catalog_active_pending_and_published_reservations_share_one_owner_cap() {
        let gate = Arc::new(DecodeBudget::default());
        let active = gate.reserve(CATALOG_RESERVATION / 2).unwrap();
        let pending = gate.reserve(CATALOG_RESERVATION / 2).unwrap();
        assert!(gate.reserve(1).is_err());
        drop(active);
        let published = gate.reserve(CATALOG_RESERVATION / 2).unwrap();
        assert!(gate.reserve(1).is_err());
        // Separate app owners do not share a global admission gate.
        let other = Arc::new(DecodeBudget::default());
        let independent = other.reserve(CATALOG_RESERVATION).unwrap();
        drop((pending, published, independent));
        assert!(gate.reserve(CATALOG_RESERVATION).is_ok());
    }
}

#[cfg(test)]
#[path = "../../tests/support/dynamics.rs"]
pub(crate) mod availability_fixture;

#[cfg(test)]
mod availability_tests {
    use super::*;
    use crate::{
        data::Loader,
        dynamics::source::{CatalogLimits, TrajectoryCatalog},
    };
    use rl::diagnostics::storage::ReadLimits;
    use std::{sync::mpsc, time::Duration};

    const TIMEOUT: Duration = Duration::from_secs(10);
    fn fixture() -> (tempfile::TempDir, RunIndex, Request) {
        fixture_with_sources(1)
    }
    fn fixture_with_sources(count: usize) -> (tempfile::TempDir, RunIndex, Request) {
        use rl::diagnostics::storage::{Recorder, StorageConfig};
        let dir = tempfile::tempdir().unwrap();
        let space = availability_fixture::bounded(3);
        let mut recorder = Recorder::create(
            dir.path().join("run"),
            availability_fixture::manifest(space.clone()),
            StorageConfig::default(),
        )
        .unwrap();
        for _ in 0..count {
            recorder
                .append(&availability_fixture::chunk(&space))
                .unwrap();
        }
        recorder.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let (tx, rx) = mpsc::channel();
        let mut loader = Loader::new(1 << 20, move || {
            let _ = tx.send(());
        })
        .unwrap();
        let mut catalog = TrajectoryCatalog::new(&run, "target", CatalogLimits::default()).unwrap();
        while catalog.pending() {
            catalog.poll(&run, &mut loader).unwrap();
            if catalog.pending() {
                rx.recv_timeout(TIMEOUT).unwrap();
            }
        }
        let sources = SourceSet::new(
            &run,
            catalog.entries(),
            catalog.generation(),
            &WorkLimits::default(),
        )
        .unwrap();
        let request = Request {
            selected: sources.sources()[0].key.clone(),
            sources,
            revision: 17,
            axes: Axes { x: 0, y: 1 },
            cells: [2, 2],
            action: ActionAxis::Continuous(0),
            projection: Projection::None,
            reference: None,
            domain: DomainSpec::default(),
        };
        (dir, run, request)
    }

    #[test]
    fn pinned_reference_cancellation_stops_preparation_and_clicks_between_metadata_calls() {
        // Entry validation, prepublication validation, click replacement,
        // retirement and cancellation stop after one already-running OS call.
        for phase in 0..5 {
            let (_dir, _run, request) = fixture_with_sources(2);
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let release_rx = Mutex::new(release_rx);
            let armed = Arc::new(AtomicUsize::new(0));
            let armed_worker = armed.clone();
            let checks = Arc::new(AtomicUsize::new(0));
            let checks_worker = checks.clone();
            let (tx, rx) = mpsc::channel();
            let mut worker = DynamicsWorker::with_source_checks(
                WorkLimits::default(),
                move || {
                    let _ = tx.send(());
                },
                ChunkSource::load_if_within_charge,
                ChunkSource::check_available,
                move |source| {
                    if armed_worker.load(Ordering::Acquire) != 0 {
                        let call = checks_worker.fetch_add(1, Ordering::AcqRel);
                        // Reused-reference preparation validates twice. Gate
                        // either the first entry or first publication check.
                        if call == if phase == 1 { 2 } else { 0 } {
                            entered_tx.send(()).unwrap();
                            release_rx.lock().unwrap().recv_timeout(TIMEOUT).unwrap();
                        }
                    }
                    source.check_available()
                },
            )
            .unwrap();
            worker.request(request.clone()).unwrap();
            rx.recv_timeout(TIMEOUT).unwrap();
            worker.poll();
            let frame = worker.frame_arc().unwrap();
            assert_eq!(frame.state.reference.source_keys.len(), 2);
            armed.store(1, Ordering::Release);
            let view = HitView {
                axes: request.axes,
                bounds: frame.state.reference.grid.grid.bounds(),
                chart: eframe::egui::Rect::from_min_max([0., 0.].into(), [40., 40.].into()),
            };
            if phase < 2 {
                let mut next = request.clone();
                next.action = ActionAxis::Continuous(1);
                worker.request(next).unwrap();
            } else {
                worker
                    .request_hit(&frame, request.sources.clone(), view, [0., 40.].into())
                    .unwrap();
            }
            entered_rx.recv_timeout(TIMEOUT).unwrap();
            if phase == 2 {
                // This advances only the hit token, not the preparation serial.
                worker
                    .request_hit(&frame, request.sources.clone(), view, [40., 0.].into())
                    .unwrap();
            } else if phase == 4 {
                worker.cancel_hit();
                assert!(worker.request_availability(&frame, &request.sources));
            } else {
                if phase == 3 {
                    worker.retire();
                }
                let mut replacement = request.clone();
                replacement.domain = DomainSpec::automatic(1);
                worker.request(replacement).unwrap();
            }
            // Only the already entered call remains counted. A stale scan that
            // reaches another call still sees the armed flag until its job ends.
            release_tx.send(()).unwrap();
            rx.recv_timeout(TIMEOUT).unwrap();
            // New-domain preparation and replacement hits add two calls.
            // Periodic availability uses its independent checker instead.
            assert_eq!(
                checks.load(Ordering::Acquire),
                match phase {
                    1 => 5,
                    4 => 1,
                    _ => 3,
                },
                "phase {phase}: stale job checked another pinned source"
            );
            if phase == 2 {
                assert_eq!(worker.poll_hit(&frame, view), Some(3));
            } else if phase == 4 {
                assert_eq!(worker.poll_hit(&frame, view), None);
                assert_eq!(
                    worker.poll_availability(&frame, &request.sources),
                    Some(Ok(()))
                );
            } else {
                worker.poll();
                assert_eq!(
                    worker.frame().unwrap().state.key.domain(),
                    DomainSpec::automatic(1)
                );
            }
            assert!(rx.try_recv().is_err(), "stale job woke UI in phase {phase}");
        }
    }

    #[test]
    fn availability_preemption_stops_after_inflight_metadata_and_preserves_normal_answers() {
        for prepare in [false, true] {
            let (_dir, _run, request) = fixture();
            let (entered_tx, entered_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel();
            let release_rx = Mutex::new(release_rx);
            let checks = Arc::new(AtomicUsize::new(0));
            let checks_worker = checks.clone();
            let (tx, rx) = mpsc::channel();
            let mut worker = DynamicsWorker::with_io(
                WorkLimits::default(),
                move || {
                    let _ = tx.send(());
                },
                ChunkSource::load_if_within_charge,
                move |source| {
                    if checks_worker.fetch_add(1, Ordering::SeqCst) == 0 {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv_timeout(TIMEOUT).unwrap();
                    }
                    source.check_available()
                },
            )
            .unwrap();
            worker.request(request.clone()).unwrap();
            rx.recv_timeout(TIMEOUT).unwrap();
            worker.poll();
            let frame = worker.frame_arc().unwrap();
            let counters = worker.counters();
            assert!(worker.request_availability(&frame, &request.sources));
            entered_rx.recv_timeout(TIMEOUT).unwrap();
            let token = worker.shared.availability_serial.load(Ordering::Acquire);
            for _ in 0..4 {
                assert!(!worker.request_availability(&frame, &request.sources));
            }
            assert_eq!(
                worker.shared.availability_serial.load(Ordering::Acquire),
                token
            );
            assert!(!worker.pending());
            let view = HitView {
                axes: request.axes,
                bounds: frame.state.reference.grid.grid.bounds(),
                chart: eframe::egui::Rect::from_min_max([0., 0.].into(), [40., 40.].into()),
            };
            if prepare {
                let mut next = request.clone();
                next.domain = DomainSpec::automatic(1);
                worker.request(next).unwrap();
            } else {
                worker
                    .request_hit(&frame, request.sources.clone(), view, [0., 40.].into())
                    .unwrap();
            }
            assert!(worker.pending());
            release_tx.send(()).unwrap();
            rx.recv_timeout(TIMEOUT).unwrap();
            assert_eq!(checks.load(Ordering::SeqCst), 1);
            assert_eq!(worker.poll_availability(&frame, &request.sources), None);
            if prepare {
                worker.poll();
                assert_eq!(
                    worker.frame().unwrap().state.key.domain(),
                    DomainSpec::automatic(1)
                );
            } else {
                assert_eq!(worker.poll_hit(&frame, view), Some(0));
                assert_eq!(worker.counters(), counters);
            }
            assert!(
                rx.try_recv().is_err(),
                "cancelled availability must not wake"
            );
        }
    }

    #[test]
    fn availability_answers_are_private_and_reject_stale_generation() {
        let (_dir, run, request) = fixture();
        let (tx, rx) = mpsc::channel();
        let mut worker = DynamicsWorker::new(WorkLimits::default(), move || {
            let _ = tx.send(());
        })
        .unwrap();
        worker.request(request.clone()).unwrap();
        rx.recv_timeout(TIMEOUT).unwrap();
        worker.poll();
        let frame = worker.frame_arc().unwrap();
        let view = HitView {
            axes: request.axes,
            bounds: frame.state.reference.grid.grid.bounds(),
            chart: eframe::egui::Rect::from_min_max([0., 0.].into(), [40., 40.].into()),
        };
        worker
            .request_hit(&frame, request.sources.clone(), view, [0., 40.].into())
            .unwrap();
        rx.recv_timeout(TIMEOUT).unwrap();
        assert!(worker.request_availability(&frame, &request.sources));
        rx.recv_timeout(TIMEOUT).unwrap();
        assert_eq!(
            worker.poll_availability(&frame, &request.sources),
            Some(Ok(()))
        );
        assert_eq!(worker.poll_hit(&frame, view), Some(0));
        let changed =
            SourceSet::new(&run, request.sources.sources(), 999, &WorkLimits::default()).unwrap();
        for missing in [false, true] {
            if missing {
                std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
            }
            assert!(worker.request_availability(&frame, &request.sources));
            rx.recv_timeout(TIMEOUT).unwrap();
            assert_eq!(worker.poll_availability(&frame, &changed), None);
            assert!(!worker.request_availability(&frame, &changed));
        }
        assert!(worker.request_availability(&frame, &request.sources));
        rx.recv_timeout(TIMEOUT).unwrap();
        assert!(
            worker
                .poll_availability(&frame, &request.sources)
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn availability_retirement_discards_inflight_failure_and_releases_weak_ownership() {
        let (_dir, run, request) = fixture();
        let (_replacement_dir, _replacement_run, replacement) = fixture();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let (tx, rx) = mpsc::channel();
        let mut worker = DynamicsWorker::with_io(
            WorkLimits::default(),
            move || {
                let _ = tx.send(());
            },
            ChunkSource::load_if_within_charge,
            move |source| {
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv_timeout(TIMEOUT).unwrap();
                source.check_available()
            },
        )
        .unwrap();
        worker.request(request.clone()).unwrap();
        rx.recv_timeout(TIMEOUT).unwrap();
        worker.poll();
        let frame = worker.frame_arc().unwrap();
        assert!(worker.request_availability(&frame, &request.sources));
        entered_rx.recv_timeout(TIMEOUT).unwrap();
        let active = worker
            .shared
            .mailbox
            .lock()
            .unwrap()
            .active_availability
            .clone()
            .unwrap();
        let weak_projection = Arc::downgrade(&frame.background);
        let weak_reference = Arc::downgrade(&frame.state.reference);
        std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
        worker.retire();
        drop(frame);
        drop(request);
        assert!(active.frame.strong_count() > 0);
        assert!(active.sources.strong_count() > 0);
        worker.request(replacement.clone()).unwrap();
        release_tx.send(()).unwrap();
        rx.recv_timeout(TIMEOUT).unwrap();
        worker.poll();
        let current = worker.frame_arc().unwrap();
        assert_eq!(current.selected.key, replacement.selected);
        assert_eq!(
            worker.poll_availability(&current, &replacement.sources),
            None
        );
        assert!(
            worker
                .shared
                .mailbox
                .lock()
                .unwrap()
                .active_availability
                .is_none()
        );
        assert_eq!(active.frame.strong_count(), 0);
        assert_eq!(active.sources.strong_count(), 0);
        assert_eq!(weak_projection.strong_count(), 0);
        assert_eq!(weak_reference.strong_count(), 0);
        assert!(rx.try_recv().is_err());
        // An obsolete weak queue is also silent and cannot perform filesystem IO.
        let expired = AvailabilityJob {
            serial: worker.shared.serial.load(Ordering::Acquire),
            availability_serial: worker.shared.availability_serial.load(Ordering::Acquire),
            ..active
        };
        assert_eq!(
            scan_availability(&expired, &worker.shared, |_| panic!(
                "expired weak source checked"
            )),
            None
        );
    }

    #[test]
    fn availability_drop_cancels_without_wake_or_retained_frame() {
        struct Finished(mpsc::Sender<()>);
        impl Drop for Finished {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let (_dir, _run, request) = fixture();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let (finished_tx, finished_rx) = mpsc::channel();
        let finished = Finished(finished_tx);
        let (tx, rx) = mpsc::channel();
        let mut worker = DynamicsWorker::with_io(
            WorkLimits::default(),
            move || {
                let _ = tx.send(());
            },
            ChunkSource::load_if_within_charge,
            move |source| {
                let _keep_until_exit = &finished;
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv_timeout(TIMEOUT).unwrap();
                source.check_available()
            },
        )
        .unwrap();
        worker.request(request.clone()).unwrap();
        rx.recv_timeout(TIMEOUT).unwrap();
        worker.poll();
        let frame = worker.frame_arc().unwrap();
        assert!(worker.request_availability(&frame, &request.sources));
        entered_rx.recv_timeout(TIMEOUT).unwrap();
        let weak_frame = Arc::downgrade(&frame);
        let weak_sources = Arc::downgrade(&request.sources);
        drop(frame);
        drop(request);
        drop(worker);
        release_tx.send(()).unwrap();
        finished_rx.recv_timeout(TIMEOUT).unwrap();
        assert_eq!(weak_frame.strong_count(), 0);
        assert_eq!(weak_sources.strong_count(), 0);
        assert!(rx.try_recv().is_err());
    }
}
