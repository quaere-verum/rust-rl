//! Bounded discovery of actual evaluation trajectory evidence.
//!
//! The recording index knows revision/role, but only decoded chunk metadata
//! declares the evaluation protocol. A bounded completion-driven worker reads
//! one source at a time; paint never decodes arrays or retains raw chunks.
use crate::data::{CacheKey, Loader, RunIndex};
use rl::diagnostics::storage::ChunkSource;
use rl::{
    Error, Result,
    diagnostics::{
        evaluation::{EvaluationHorizon, evaluation_horizon, trajectory_scenario_seed},
        reference::{ReferenceKind, ReferenceLifetime, provider_specification},
        schema::{
            ActionCondition, Chunk, Conditioning, EvaluationMetadata, Record, RecordContext,
            RecordKind, RunManifest, TrajectoryRecord,
        },
    },
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

const MAX_CANDIDATES: usize = 4096;
const MAX_METADATA: usize = 16 * 1024;
const MAX_ERRORS: usize = 8;

/// Constant-size worker evidence. Counts are cumulative for this catalog;
/// timestamps describe the current generation, relative to catalog creation.
/// Read attempts include failed availability/admission checks; decoded sources
/// count successful payload decodes, even if metadata admission later fails.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CatalogMetrics {
    pub generation: u64,
    pub submitted_generations: u64,
    pub started_generations: u64,
    pub completed_generations: u64,
    /// Superseded queued jobs and active jobs that finished cancellation.
    pub cancelled_generations: u64,
    pub read_attempts: u64,
    pub decoded_sources: u64,
    pub submitted_at: Option<Duration>,
    pub started_at: Option<Duration>,
    pub last_progress_at: Option<Duration>,
    pub completed_at: Option<Duration>,
    pub consumed_at: Option<Duration>,
    /// Worker start through publication, including time waiting for decode.
    pub worker_duration: Option<Duration>,
}

#[derive(Clone, Copy, Debug)]
pub struct CatalogLimits {
    /// Most recent immutable records, ordered by (revision, sequence). Omitted
    /// records are disclosed: this window must not imply full-run coverage.
    pub max_sources: usize,
    /// Conservative source/group metadata charge. The bounded cursor table
    /// (at most MAX_CANDIDATES usize values) and 8 short errors are additional.
    pub max_bytes: usize,
}
impl Default for CatalogLimits {
    fn default() -> Self {
        Self {
            max_sources: 512,
            max_bytes: 2 * 1024 * 1024,
        }
    }
}

/// Exact available protocol metadata. A learner revision is not a protocol,
/// but changing arbitrary continuation-policy captions cannot be interpreted
/// as equivalent. No episode pairing is inferred from these strings.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProtocolKey {
    canonical: Arc<str>,
    caption: Arc<str>,
    environment: Arc<str>,
    unconditional: bool,
}
impl ProtocolKey {
    pub fn new(metadata: &EvaluationMetadata) -> Result<Self> {
        Self::with_continuation(metadata, &metadata.continuation_policy, false)
    }

    fn with_continuation(
        metadata: &EvaluationMetadata,
        continuation: &str,
        authored_online: bool,
    ) -> Result<Self> {
        // Bound text before allocating its canonical representation. Generic
        // recording validation already validates the discount and protocol.
        let conditioning = match &metadata.conditioning {
            Conditioning::InitialScenario => 0,
            Conditioning::FullSimulatorState { description } => description.len(),
            Conditioning::Observation { methodology } | Conditioning::History { methodology } => {
                methodology.len()
            }
        };
        let bytes = [
            conditioning,
            metadata.continuation_policy.len(),
            metadata.seed_methodology.len(),
            metadata.truncation_handling.len(),
        ]
        .into_iter()
        .try_fold(0usize, |a, b| a.checked_add(b))
        .ok_or_else(|| Error::Shape("evaluation protocol byte overflow".into()))?;
        if bytes > MAX_METADATA || !metadata.discount.is_finite() {
            return Err(Error::Shape(
                "evaluation protocol exceeds catalog admission".into(),
            ));
        }
        // complete_returns describes batch evidence, not how the evaluation
        // was conducted. Individual episode flags still decide completeness.
        let canonical = serde_json::to_string(&(
            &metadata.conditioning,
            metadata.initial_action,
            authored_online,
            continuation,
            metadata.stochastic_policy,
            metadata.discount,
            metadata.independent_future_streams,
            &metadata.seed_methodology,
            &metadata.truncation_handling,
        ))?;
        if canonical.len() > MAX_METADATA {
            return Err(Error::Shape(
                "evaluation protocol exceeds catalog admission".into(),
            ));
        }
        Ok(Self {
            canonical: canonical.into(),
            caption: continuation.into(),
            environment: "".into(),
            unconditional: metadata.conditioning == Conditioning::InitialScenario
                && metadata.initial_action == ActionCondition::CurrentPolicy,
        })
    }
    pub fn unconditional(&self) -> bool {
        self.unconditional
    }
    pub fn caption(&self) -> &str {
        &self.caption
    }
    pub fn environment(&self) -> &str {
        &self.environment
    }
    /// Conservative text/Arc ownership for downstream request admission.
    pub fn retained_bytes(&self) -> usize {
        256 + 2 * (self.canonical.len() + self.caption.len() + self.environment.len())
    }
    /// Record environment is part of source compatibility even when a run
    /// happens to contain identical evaluation captions for different adapters.
    pub fn for_trajectory(record: &TrajectoryRecord) -> Result<Self> {
        Self::for_recorded_trajectory(record, None)
    }

    pub(crate) fn for_recorded_trajectory(
        record: &TrajectoryRecord,
        context: Option<&RecordContext>,
    ) -> Result<Self> {
        if record.environment.len() > MAX_METADATA {
            return Err(Error::Shape(
                "trajectory environment exceeds catalog admission".into(),
            ));
        }
        // DiagnosticSession includes the snapshot revision in the continuation
        // name. Group those snapshots by the authored protocol. Preserve all
        // other captions, including distinct policies recorded in one run.
        let authored_online = context.is_some_and(|c| {
            c.identity.model_role == "online"
                && record.evaluation.continuation_policy
                    == format!(
                        "{}@{}",
                        c.identity.run_id, c.identity.learner_update_revision
                    )
        });
        let mut key = if authored_online {
            Self::with_continuation(&record.evaluation, "Online learner", true)?
        } else {
            Self::new(&record.evaluation)?
        };
        key.environment = record.environment.as_str().into();
        Ok(key)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectorySource {
    pub key: CacheKey,
    pub revision: u64,
    pub role: String,
    pub id: String,
    pub environment: String,
    pub protocol: ProtocolKey,
    pub transitions: usize,
    pub has_render_scenes: bool,
    pub(crate) scenario: Option<RecordedScenario>,
    pub(crate) benchmark: Option<BenchmarkEpisode>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecordedScenario {
    pub seed: u64,
    horizon: EvaluationHorizon,
    discount_bits: u64,
}
impl RecordedScenario {
    pub(crate) fn for_trajectory(chunk: &Chunk) -> Result<Option<Self>> {
        let Record::Trajectory(record) = &chunk.record else {
            return Ok(None);
        };
        let evaluation = &record.evaluation;
        let seed = trajectory_scenario_seed(chunk)?;
        let horizon = evaluation_horizon(chunk)?;
        Ok(seed
            .filter(|_| {
                evaluation.conditioning == Conditioning::InitialScenario
                    && evaluation.initial_action == ActionCondition::CurrentPolicy
                    && evaluation.independent_future_streams
            })
            .map(|seed| Self {
                seed,
                horizon,
                discount_bits: evaluation.discount.to_bits(),
            }))
    }
}

#[cfg(test)]
mod benchmark_scenario_tests {
    use super::*;
    use rl::diagnostics::schema::*;

    #[test]
    fn benchmark_scenario_uses_typed_seed_horizon_and_discount() {
        let mut path = Chunk {
            context: RecordContext::new("run", 0),
            record: Record::Trajectory(TrajectoryRecord {
                render: None,
                id: "path".into(),
                environment: "custom".into(),
                observations: "states".into(),
                actions: "actions".into(),
                rewards: "rewards".into(),
                terminated: "terminated".into(),
                truncated: "truncated".into(),
                simulation_times: None,
                action_masks: None,
                frames: Default::default(),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "learner".into(),
                    stochastic_policy: true,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "independent policy and environment seeds".into(),
                    truncation_handling: "complete finite objective".into(),
                    complete_returns: true,
                },
            }),
            arrays: Default::default(),
        };
        let seed = "trajectory/scenario_seed.v1";
        let horizon = "evaluation/finite_decision_horizon.v1";
        path.arrays
            .insert(seed.into(), Array::u64(vec![1], vec![8]).unwrap());
        path.arrays
            .insert(horizon.into(), Array::u64(vec![1], vec![12]).unwrap());
        let learner = RecordedScenario::for_trajectory(&path).unwrap().unwrap();
        let Record::Trajectory(record) = &mut path.record else {
            unreachable!()
        };
        record.evaluation.continuation_policy = "benchmark".into();
        record.evaluation.stochastic_policy = false;
        assert_eq!(
            RecordedScenario::for_trajectory(&path).unwrap(),
            Some(learner.clone())
        );
        path.arrays
            .insert(seed.into(), Array::u64(vec![1], vec![9]).unwrap());
        assert_ne!(
            RecordedScenario::for_trajectory(&path).unwrap(),
            Some(learner.clone())
        );
        path.arrays
            .insert(seed.into(), Array::u64(vec![1], vec![8]).unwrap());
        path.arrays
            .insert(horizon.into(), Array::u64(vec![1], vec![13]).unwrap());
        assert_ne!(
            RecordedScenario::for_trajectory(&path).unwrap(),
            Some(learner.clone())
        );
        path.arrays
            .insert(horizon.into(), Array::u64(vec![1], vec![12]).unwrap());
        let Record::Trajectory(record) = &mut path.record else {
            unreachable!()
        };
        record.evaluation.discount = 0.95;
        assert_ne!(
            RecordedScenario::for_trajectory(&path).unwrap(),
            Some(learner)
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BenchmarkEpisode {
    pub quality: ReferenceKind,
    pub lifetime: ReferenceLifetime,
}
#[derive(Clone, Debug)]
pub struct ProtocolGroup {
    pub protocol: ProtocolKey,
    pub positions: Vec<u64>,
    pub episodes: usize,
}

#[derive(Default)]
struct CatalogData {
    _reservation: Option<super::work::CatalogReservation>,
    entries: Vec<TrajectorySource>,
    groups: Vec<ProtocolGroup>,
    errors: Vec<String>,
    rejected: usize,
    retained_bytes: usize,
}
struct CatalogBuilder {
    role: String,
    limits: CatalogLimits,
    data: CatalogData,
}
impl CatalogBuilder {
    fn reject(&mut self, error: &str) {
        self.data.rejected += 1;
        if self.data.errors.len() < MAX_ERRORS {
            self.data.errors.push(error.chars().take(256).collect());
        }
    }
    fn admit(
        &mut self,
        manifest: &RunManifest,
        candidate: &Candidate,
        chunk: &Chunk,
    ) -> Result<()> {
        let Record::Trajectory(record) = &chunk.record else {
            return Err(Error::State(
                "trajectory source has a different record kind".into(),
            ));
        };
        let revision = candidate.revision;
        if chunk.context.identity.run_id != manifest.run_id
            || chunk.context.identity.learner_update_revision != revision
            || chunk.context.identity.model_role != self.role
        {
            return Err(Error::State(
                "trajectory source identity differs from index".into(),
            ));
        }
        if record.id.len() > MAX_METADATA || record.environment.len() > MAX_METADATA {
            return Err(Error::Shape(
                "trajectory labels exceed catalog admission".into(),
            ));
        }
        let protocol = ProtocolKey::for_recorded_trajectory(record, Some(&chunk.context))?;
        let observations = chunk
            .arrays
            .get(&record.observations)
            .ok_or_else(|| Error::Shape("missing trajectory observations".into()))?;
        let actions = chunk
            .arrays
            .get(&record.actions)
            .ok_or_else(|| Error::Shape("missing trajectory actions".into()))?;
        let rewards = chunk
            .arrays
            .get(&record.rewards)
            .ok_or_else(|| Error::Shape("missing trajectory rewards".into()))?;

        let steps = rewards.len();
        let actions_match = match &manifest.action_space {
            rl::core::action::ActionSpace::Discrete { .. } => {
                actions.shape() == [steps] && actions.dtype() == rl::diagnostics::schema::DType::U64
            }
            space => {
                actions.shape() == [steps, space.dimensions()]
                    && matches!(
                        actions.dtype(),
                        rl::diagnostics::schema::DType::F32 | rl::diagnostics::schema::DType::F64
                    )
            }
        };
        if steps == 0
            || steps
                .checked_add(1)
                .is_none_or(|n| observations.shape() != [n, manifest.observation_dim])
            || !actions_match
        {
            return Err(Error::Shape(
                "trajectory shape differs from manifest".into(),
            ));
        }
        // A source's charge includes worst-case Vec spare capacities for both
        // source and group tables, plus a distinct group/position and all text.
        // Arc protocol strings are deliberately charged again for each source.
        let charge = 4
            * (std::mem::size_of::<TrajectorySource>()
                + std::mem::size_of::<ProtocolGroup>()
                + std::mem::size_of::<u64>())
            + 2 * (record.id.len()
                + record.environment.len()
                + self.role.len()
                + protocol.canonical.len()
                + protocol.caption.len())
            + 256;
        let retained = self
            .data
            .retained_bytes
            .checked_add(charge)
            .ok_or_else(|| Error::Shape("trajectory catalog byte overflow".into()))?;
        if retained > self.limits.max_bytes {
            return Err(Error::Shape(
                "trajectory catalog metadata budget exceeded".into(),
            ));
        }
        let scenario = RecordedScenario::for_trajectory(chunk)?;
        let benchmark = provider_specification(chunk)
            .ok()
            .flatten()
            .filter(|spec| spec.model_role() == self.role)
            .map(|spec| BenchmarkEpisode {
                quality: spec.quality.kind,
                lifetime: spec.lifetime,
            });
        let source = TrajectorySource {
            key: candidate.key.clone(),
            revision,
            role: self.role.clone(),
            id: record.id.clone(),
            environment: record.environment.clone(),
            protocol: protocol.clone(),
            transitions: steps,
            has_render_scenes: record.render.is_some(),
            scenario,
            benchmark,
        };
        if let Some(group) = self
            .data
            .groups
            .iter_mut()
            .find(|group| group.protocol == protocol)
        {
            if group.positions.last().copied() != Some(revision) {
                group.positions.push(revision);
            }
            group.episodes += 1;
        } else {
            self.data.groups.push(ProtocolGroup {
                protocol,
                positions: vec![revision],
                episodes: 1,
            });
        }
        self.data.entries.push(source);
        self.data.retained_bytes = retained;
        Ok(())
    }
}
struct Candidate {
    key: CacheKey,
    revision: u64,
    source: ChunkSource,
}
struct CatalogJob {
    generation: u64,
    candidates: Vec<Candidate>,
    manifest: RunManifest,
    builder: CatalogBuilder,
    wake: Arc<dyn Fn() + Send + Sync>,
}
#[derive(Default)]
struct CatalogMailbox {
    desired: Option<CatalogJob>,
    answer: Option<(u64, CatalogData)>,
    progress: (u64, usize),
    metrics: CatalogMetrics,
    stop: bool,
}
struct CatalogShared {
    mailbox: Mutex<CatalogMailbox>,
    changed: Condvar,
    serial: AtomicU64,
    decode_gate: super::work::DecodeGate,
    created: Instant,
}

pub struct TrajectoryCatalog {
    root: Arc<PathBuf>,
    role: String,
    limits: CatalogLimits,
    frontier: (usize, Option<(u64, u64)>),
    candidates: Vec<usize>,
    cursor: usize,
    data: CatalogData,
    omitted: usize,
    generation: u64,
    submitted: bool,
    shared: Arc<CatalogShared>,
}
impl TrajectoryCatalog {
    pub fn new(run: &RunIndex, role: &str, limits: CatalogLimits) -> Result<Self> {
        Self::with_decode_gate(
            run,
            role,
            limits,
            Arc::new(super::work::DecodeBudget::default()),
        )
    }
    pub fn with_decode_gate(
        run: &RunIndex,
        role: &str,
        limits: CatalogLimits,
        decode_gate: super::work::DecodeGate,
    ) -> Result<Self> {
        if role.len() > 128
            || role.is_empty()
            || limits.max_sources == 0
            || limits.max_sources > MAX_CANDIDATES
        {
            return Err(Error::Config(
                "invalid Dynamics catalog limits or role".into(),
            ));
        }
        let shared = Arc::new(CatalogShared {
            mailbox: Mutex::new(CatalogMailbox::default()),
            changed: Condvar::new(),
            serial: AtomicU64::new(0),
            decode_gate,
            created: Instant::now(),
        });
        let worker = shared.clone();
        std::thread::Builder::new()
            .name("rl-dynamics-catalog".into())
            .spawn(move || discover(worker))?;
        let mut catalog = Self {
            root: run.root().clone(),
            role: role.into(),
            limits,
            frontier: (0, None),
            candidates: Vec::new(),
            cursor: 0,
            data: CatalogData::default(),
            omitted: 0,
            generation: 0,
            submitted: false,
            shared,
        };
        catalog.restart(run);
        Ok(catalog)
    }
    pub fn matches(&self, run: &RunIndex, role: &str) -> bool {
        Arc::ptr_eq(&self.root, run.root()) && self.role == role
    }
    fn restart(&mut self, run: &RunIndex) {
        self.frontier = run.candidate_frontier(RecordKind::Trajectory, &self.role, None, u64::MAX);
        self.candidates = run
            .candidates(RecordKind::Trajectory, &self.role, None, u64::MAX)
            .take(self.limits.max_sources)
            .collect();
        self.candidates.reverse();
        self.omitted = self.frontier.0.saturating_sub(self.candidates.len());
        self.cursor = 0;
        self.data = CatalogData::default();
        self.submitted = false;
        self.generation = self.generation.wrapping_add(1);
        self.shared.serial.store(self.generation, Ordering::Release);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        if mailbox.desired.is_some() {
            mailbox.metrics.cancelled_generations += 1;
        }
        mailbox.desired = None;
        mailbox.answer = None;
        mailbox.progress = (self.generation, 0);
        mailbox.metrics = CatalogMetrics {
            generation: self.generation,
            submitted_at: None,
            started_at: None,
            last_progress_at: None,
            completed_at: None,
            consumed_at: None,
            worker_duration: None,
            ..mailbox.metrics
        };
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn metrics(&self) -> CatalogMetrics {
        self.shared.mailbox.lock().unwrap().metrics
    }
    pub fn pending(&self) -> bool {
        self.cursor < self.candidates.len()
    }
    pub fn entries(&self) -> &[TrajectorySource] {
        &self.data.entries
    }
    pub fn groups(&self) -> &[ProtocolGroup] {
        &self.data.groups
    }
    pub fn omitted(&self) -> usize {
        self.omitted
    }
    pub fn rejected(&self) -> usize {
        self.data.rejected
    }
    pub fn retained_bytes(&self) -> usize {
        self.data.retained_bytes
    }
    pub fn errors(&self) -> &[String] {
        &self.data.errors
    }
    pub fn progress(&self) -> (usize, usize) {
        let progress = self.shared.mailbox.lock().unwrap().progress;
        (
            if progress.0 == self.generation {
                progress.1
            } else {
                self.cursor
            },
            self.candidates.len(),
        )
    }
    /// Start one bounded generation; its worker advances all remaining sources
    /// directly on completion. Paint only consumes immutable metadata/progress.
    pub fn poll(&mut self, run: &RunIndex, loader: &mut Loader) -> Result<()> {
        if !self.matches(run, &self.role) {
            return Err(Error::State(
                "Dynamics catalog belongs to another run".into(),
            ));
        }
        let frontier = run.candidate_frontier(RecordKind::Trajectory, &self.role, None, u64::MAX);
        if self.frontier != frontier {
            self.restart(run);
        }
        if self.pending() && !self.submitted {
            // At most two jobs (active + replaceable desired), and one published
            // metadata table. Each job reserves half the catalog envelope for
            // handles, manifest, result and Vec/error/cursor overhead. Preflight
            // variable index strings before cloning any detached handles.
            let manifest = run.recording().manifest();
            let mut charge = serde_json::to_vec(manifest)?
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(self.limits.max_bytes))
                .and_then(|n| n.checked_add(256 * 1024))
                .ok_or_else(|| Error::Shape("catalog ownership overflow".into()))?;
            for &index in &self.candidates {
                let entry = &run.recording().entries()[index];
                charge = charge
                    .checked_add(serde_json::to_vec(entry)?.len().saturating_mul(8))
                    .and_then(|n| n.checked_add(run.root().as_os_str().len().saturating_mul(8)))
                    .and_then(|n| n.checked_add(2048))
                    .ok_or_else(|| Error::Shape("catalog ownership overflow".into()))?;
            }
            if charge > super::work::CATALOG_RESERVATION / 2 {
                return Err(Error::Shape(
                    "Dynamics catalog ownership exceeds budget".into(),
                ));
            }
            let reservation = self.shared.decode_gate.reserve(charge)?;
            let candidates = self
                .candidates
                .iter()
                .map(|&index| {
                    Ok(Candidate {
                        key: run.key(index)?,
                        revision: run.recording().entries()[index].revision,
                        source: run.recording().source(index)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let job = CatalogJob {
                generation: self.generation,
                candidates,
                manifest: manifest.clone(),
                builder: CatalogBuilder {
                    role: self.role.clone(),
                    limits: self.limits,
                    data: CatalogData {
                        _reservation: Some(reservation),
                        ..Default::default()
                    },
                },
                wake: loader.wake_callback(),
            };
            {
                let mut mailbox = self.shared.mailbox.lock().unwrap();
                mailbox.metrics.submitted_generations += 1;
                mailbox.metrics.submitted_at = Some(self.shared.created.elapsed());
                mailbox.desired = Some(job);
            }
            self.submitted = true;
            self.shared.changed.notify_one();
        }
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        if let Some((generation, data)) = mailbox.answer.take()
            && generation == self.generation
        {
            mailbox.metrics.consumed_at = Some(self.shared.created.elapsed());
            self.data = data;
            self.cursor = self.candidates.len();
        }
        Ok(())
    }
    pub fn retry(&mut self, run: &RunIndex, loader: &mut Loader) -> Result<()> {
        if !self.matches(run, &self.role) {
            return Err(Error::State(
                "Dynamics catalog belongs to another run".into(),
            ));
        }
        for &index in &self.candidates {
            loader.retry_failed(&run.key(index)?);
        }
        self.restart(run);
        Ok(())
    }
}
impl Drop for TrajectoryCatalog {
    fn drop(&mut self) {
        self.shared.serial.fetch_add(1, Ordering::AcqRel);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        mailbox.stop = true;
        if mailbox.desired.is_some() {
            mailbox.metrics.cancelled_generations += 1;
        }
        mailbox.desired = None;
        mailbox.answer = None;
        self.shared.changed.notify_one();
    }
}
fn discover(shared: Arc<CatalogShared>) {
    loop {
        let (mut job, started_at) = {
            let mut mailbox = shared.mailbox.lock().unwrap();
            loop {
                if mailbox.stop {
                    return;
                }
                if let Some(job) = mailbox.desired.take() {
                    let started_at = shared.created.elapsed();
                    mailbox.metrics.started_generations += 1;
                    if mailbox.metrics.generation == job.generation {
                        mailbox.metrics.started_at = Some(started_at);
                    }
                    break (job, started_at);
                }
                mailbox = shared.changed.wait(mailbox).unwrap();
            }
        };
        for (position, candidate) in job.candidates.iter().enumerate() {
            if shared.serial.load(Ordering::Acquire) != job.generation {
                break;
            }
            let mut decoded = false;
            {
                let _decode = shared.decode_gate.lock.lock().unwrap();
                if shared.serial.load(Ordering::Acquire) != job.generation {
                    break;
                }
                shared.mailbox.lock().unwrap().metrics.read_attempts += 1;
                let result = candidate
                    .source
                    .check_available()
                    .and_then(|()| {
                        candidate
                            .source
                            .load_if_within_charge(shared.decode_gate.read_limit)
                    })
                    .and_then(|chunk| {
                        chunk.ok_or_else(|| {
                            Error::Shape("Dynamics catalog source exceeds read budget".into())
                        })
                    })
                    .and_then(|chunk| {
                        decoded = true;
                        candidate.source.check_available()?;
                        job.builder.admit(&job.manifest, candidate, &chunk)
                    });
                if let Err(error) = result {
                    job.builder.reject(&error.to_string());
                }
            }
            let mut mailbox = shared.mailbox.lock().unwrap();
            mailbox.metrics.decoded_sources += u64::from(decoded);
            if shared.serial.load(Ordering::Acquire) == job.generation {
                mailbox.progress = (job.generation, position + 1);
                mailbox.metrics.last_progress_at = Some(shared.created.elapsed());
            }
        }
        // Earlier sources may disappear while later sources decode. Recheck
        // exact immutable addresses before publishing any catalog generation.
        let mut removed = false;
        for candidate in &job.candidates {
            if shared.serial.load(Ordering::Acquire) != job.generation {
                break;
            }
            if let Err(error) = candidate.source.check_available()
                && let Some(index) = job
                    .builder
                    .data
                    .entries
                    .iter()
                    .position(|source| source.key == candidate.key)
            {
                job.builder.data.entries.remove(index);
                job.builder.reject(&error.to_string());
                removed = true;
            }
        }
        if removed {
            let data = &mut job.builder.data;
            data.groups.clear();
            for source in &data.entries {
                if let Some(group) = data
                    .groups
                    .iter_mut()
                    .find(|group| group.protocol == source.protocol)
                {
                    if group.positions.last() != Some(&source.revision) {
                        group.positions.push(source.revision);
                    }
                    group.episodes += 1;
                } else {
                    data.groups.push(ProtocolGroup {
                        protocol: source.protocol.clone(),
                        positions: vec![source.revision],
                        episodes: 1,
                    });
                }
            }
        }
        // The reservation travels with the immutable answer. Release decoded
        // job handles first, so a consumer can retire the answer immediately
        // without leaving uncharged handles alive behind a blocked wake hook.
        drop(job.candidates);
        drop(job.manifest);
        let mut mailbox = shared.mailbox.lock().unwrap();
        if shared.serial.load(Ordering::Acquire) == job.generation && !mailbox.stop {
            let completed_at = shared.created.elapsed();
            mailbox.metrics.completed_generations += 1;
            mailbox.metrics.completed_at = Some(completed_at);
            mailbox.metrics.worker_duration = Some(completed_at - started_at);
            mailbox.answer = Some((job.generation, job.builder.data));
            drop(mailbox);
            (job.wake)();
        } else {
            mailbox.metrics.cancelled_generations += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::{
        core::action::ActionSpace,
        diagnostics::{schema::*, storage::*},
    };
    use std::{collections::BTreeMap, time::Instant};

    fn fixture(path: &std::path::Path) -> RunIndex {
        let manifest = RunManifest::new(
            "catalog-test",
            "catalog-test",
            "fixture",
            1,
            ActionSpace::Unbounded { dimensions: 1 },
            0.9,
        )
        .unwrap();
        let mut recorder = Recorder::create(path, manifest, StorageConfig::default()).unwrap();
        for revision in 0..2 {
            let mut context = RecordContext::new("catalog-test", revision);
            context.identity.model_role = "online".into();
            context.trajectory = Some(format!("episode-{revision}"));
            recorder
                .append(&Chunk {
                    context,
                    record: Record::Trajectory(TrajectoryRecord {
                        render: None,
                        id: format!("episode-{revision}"),
                        environment: "fixture".into(),
                        observations: "s".into(),
                        actions: "a".into(),
                        rewards: "r".into(),
                        terminated: "term".into(),
                        truncated: "trunc".into(),
                        simulation_times: None,
                        action_masks: None,
                        frames: BTreeMap::new(),
                        evaluation: EvaluationMetadata {
                            conditioning: Conditioning::InitialScenario,
                            initial_action: ActionCondition::CurrentPolicy,
                            continuation_policy: "policy".into(),
                            stochastic_policy: false,
                            discount: 0.9,
                            independent_future_streams: true,
                            seed_methodology: "explicit seeds".into(),
                            truncation_handling: "reject incomplete".into(),
                            complete_returns: true,
                        },
                    }),
                    arrays: BTreeMap::from([
                        ("s".into(), Array::f32(vec![2, 1], vec![0., 1.]).unwrap()),
                        ("a".into(), Array::f32(vec![1, 1], vec![0.1]).unwrap()),
                        ("r".into(), Array::f32(vec![1], vec![1.]).unwrap()),
                        ("term".into(), Array::u8(vec![1], vec![1]).unwrap()),
                        ("trunc".into(), Array::u8(vec![1], vec![0]).unwrap()),
                    ]),
                })
                .unwrap();
        }
        recorder.finish().unwrap();
        RunIndex::open(path, ReadLimits::default()).unwrap()
    }

    fn wait_for(mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !predicate() {
            assert!(
                Instant::now() < deadline,
                "catalog worker did not reach synchronization point"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn retry_cancels_active_catalog_waiting_for_decode_without_reading_stale_sources() {
        let dir = tempfile::tempdir().unwrap();
        let run = fixture(&dir.path().join("run"));
        let gate = Arc::new(super::super::work::DecodeBudget::default());
        let hold = gate.lock.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut loader = Loader::new(8 << 20, move || {
            let _ = tx.send(());
        })
        .unwrap();
        let mut catalog = TrajectoryCatalog::with_decode_gate(
            &run,
            "online",
            CatalogLimits::default(),
            gate.clone(),
        )
        .unwrap();
        catalog.poll(&run, &mut loader).unwrap();
        wait_for(|| catalog.metrics().started_generations == 1);
        assert_eq!(catalog.metrics().read_attempts, 0);
        catalog.retry(&run, &mut loader).unwrap();
        assert_eq!(catalog.metrics().started_at, None);
        catalog.poll(&run, &mut loader).unwrap();
        drop(hold);
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let metrics = catalog.metrics();
        assert_eq!(metrics.submitted_generations, 2);
        assert_eq!(metrics.started_generations, 2);
        assert_eq!(metrics.cancelled_generations, 1);
        assert_eq!(metrics.completed_generations, 1);
        assert_eq!(metrics.read_attempts, 2);
        assert_eq!(metrics.decoded_sources, 2);
        assert_eq!(metrics.consumed_at, None);
        assert!(metrics.completed_at.is_some());
        assert_eq!(catalog.progress(), (2, 2));
        assert!(catalog.entries().is_empty());
        catalog.poll(&run, &mut loader).unwrap();
        assert_eq!(catalog.entries().len(), 2);
        assert_eq!(catalog.metrics().completed_at, metrics.completed_at);
        assert!(catalog.metrics().consumed_at.unwrap() >= metrics.completed_at.unwrap());
        let consumed = catalog.metrics();
        for _ in 0..20 {
            catalog.poll(&run, &mut loader).unwrap();
        }
        assert_eq!(catalog.metrics(), consumed);
        assert!(
            rx.try_recv().is_err(),
            "obsolete generation must not wake UI"
        );
    }

    #[test]
    fn dropping_active_catalog_does_not_wait_for_decode_or_retain_owner_reservation() {
        let dir = tempfile::tempdir().unwrap();
        let run = fixture(&dir.path().join("run"));
        let gate = Arc::new(super::super::work::DecodeBudget::default());
        let hold = gate.lock.lock().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut loader = Loader::new(8 << 20, move || {
            let _ = tx.send(());
        })
        .unwrap();
        let mut catalog = TrajectoryCatalog::with_decode_gate(
            &run,
            "online",
            CatalogLimits::default(),
            gate.clone(),
        )
        .unwrap();
        let shared = catalog.shared.clone();
        catalog.poll(&run, &mut loader).unwrap();
        wait_for(|| catalog.metrics().started_generations == 1);
        drop(catalog); // Must return while the real decode gate remains held.
        assert!(
            gate.reserve(super::super::work::CATALOG_RESERVATION)
                .is_err()
        );
        drop(hold);
        wait_for(|| Arc::strong_count(&shared) == 1);
        assert!(
            gate.reserve(super::super::work::CATALOG_RESERVATION)
                .is_ok()
        );
        assert!(shared.mailbox.lock().unwrap().answer.is_none());
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn catalog_read_budget_rejects_payload_before_decode() {
        let dir = tempfile::tempdir().unwrap();
        let run = fixture(&dir.path().join("run"));
        let mut gate = super::super::work::DecodeBudget::default();
        gate.read_limit = 1;
        let (tx, rx) = std::sync::mpsc::channel();
        let mut loader = Loader::new(8 << 20, move || {
            let _ = tx.send(());
        })
        .unwrap();
        let mut catalog = TrajectoryCatalog::with_decode_gate(
            &run,
            "online",
            CatalogLimits::default(),
            Arc::new(gate),
        )
        .unwrap();
        catalog.poll(&run, &mut loader).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(catalog.metrics().read_attempts, 2);
        assert_eq!(catalog.metrics().decoded_sources, 0);
        catalog.poll(&run, &mut loader).unwrap();
        assert_eq!(catalog.rejected(), 2);
        assert!(catalog.entries().is_empty());
        assert!(
            catalog
                .errors()
                .iter()
                .all(|error| error.contains("read budget"))
        );
    }
}
