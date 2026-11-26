//! Bounded evaluation, scalar metric and independent-run Training evidence.
use super::history::{Lease, Reader, Shared, entry_bytes, manifest_charge};
use super::{EmpiricalSummary, MAX_READ_BYTES, evaluation_summary};
use crate::data::RunIndex;
use rl::{
    Error, Result,
    diagnostics::{
        evaluation::{EvaluationHorizon, evaluation_horizon},
        reference::{ReferenceContext, ReferenceTarget, manifest_context, provider_specification},
        schema::{
            ActionCondition, Chunk, Conditioning, Record, RecordContext, RecordKind, RunManifest,
        },
        storage::{ChunkSource, ReadLimits},
        value,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

pub const MAX_TRAINING_COHORT_PEERS: usize = 4;
const MAX_EVALUATIONS: usize = 192;
const MAX_METRIC_SOURCES: usize = 384;
const MAX_METRIC_KEYS: usize = 192;
const MAX_METRICS_PER_SOURCE: usize = 1024;
const DESCRIPTOR_BYTES: usize = 2 * 1024 * 1024;
const METRIC_BYTES: usize = 1024 * 1024;
const PEER_RECHECK_INTERVAL: Duration = Duration::from_secs(1);
const SCENARIO_PROTOCOL_KEY: &str = "evaluation.scenario_protocol";
const KINDS: [RecordKind; 2] = [RecordKind::Empirical, RecordKind::Metrics];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrainingClock {
    Revision,
    EnvironmentDecisions,
    PhysicalTicks,
    WallSeconds,
}

impl TrainingClock {
    pub const ALL: [Self; 4] = [
        Self::Revision,
        Self::EnvironmentDecisions,
        Self::PhysicalTicks,
        Self::WallSeconds,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Revision => "Learner updates",
            Self::EnvironmentDecisions => "Environment decisions",
            Self::PhysicalTicks => "Physical interaction ticks",
            Self::WallSeconds => "Session wall time (s)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingClocks {
    pub revision: u64,
    pub environment_decisions: Option<u64>,
    pub physical_ticks: Option<u64>,
    pub wall_seconds: Option<f64>,
}

impl TrainingClocks {
    fn missing(revision: u64) -> Self {
        Self {
            revision,
            environment_decisions: None,
            physical_ticks: None,
            wall_seconds: None,
        }
    }

    fn from_context(context: &RecordContext) -> Self {
        let counters = context.counters;
        // Older hand-authored records may carry a real nonzero counter without
        // the explicit presence flag. Their zero defaults do not establish a clock.
        let environment_decisions = context
            .environment_step
            .or((counters.env_steps != 0).then_some(counters.env_steps));
        let physical_ticks = environment_decisions
            .filter(|&decisions| decisions == counters.env_steps)
            .and_then(|decisions| decisions.checked_add(counters.additional_physical_steps));
        Self {
            revision: context.identity.learner_update_revision,
            environment_decisions,
            physical_ticks,
            wall_seconds: context.wall_elapsed_seconds,
        }
    }

    pub fn value(self, clock: TrainingClock) -> Option<f64> {
        let exact = |value: u64| (value <= (1 << 53)).then_some(value as f64);
        match clock {
            TrainingClock::Revision => exact(self.revision),
            TrainingClock::EnvironmentDecisions => self.environment_decisions.and_then(exact),
            TrainingClock::PhysicalTicks => self.physical_ticks.and_then(exact),
            TrainingClock::WallSeconds => self
                .wall_seconds
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.)
                .map(|seconds| seconds + 0.),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainingEvaluation {
    pub clocks: TrainingClocks,
    pub sequence: u64,
    pub summary: Option<EmpiricalSummary>,
    pub extrema: Option<[f64; 2]>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrainingMetricKey {
    pub name: String,
    pub population: Option<String>,
    /// Authored reference model role, e.g. `reference:benchmark`.
    pub reference: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TrainingMetricPoint {
    pub clocks: TrainingClocks,
    pub sequence: u64,
    pub value: Option<f64>,
    pub reason: Option<String>,
}

pub struct TrainingRun {
    pub root: Arc<PathBuf>,
    pub run_id: String,
    pub algorithm: String,
    pub role: String,
    pub reference_target: Option<ReferenceTarget>,
    pub evaluations: Vec<TrainingEvaluation>,
    pub metrics: BTreeMap<TrainingMetricKey, Vec<TrainingMetricPoint>>,
    pub omitted_sources: usize,
    pub omitted_metrics: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingAggregatePoint {
    pub x: f64,
    pub mean: f64,
    pub standard_error: Option<f64>,
    pub runs: usize,
    /// Plot a new segment here when incomplete data separates this point.
    pub starts_segment: bool,
}

pub struct TrainingCohort {
    pub members: Vec<String>,
    series: BTreeMap<TrainingClock, Vec<TrainingAggregatePoint>>,
}

impl TrainingCohort {
    pub fn points(&self, clock: TrainingClock) -> &[TrainingAggregatePoint] {
        self.series.get(&clock).map_or(&[], Vec::as_slice)
    }
}

pub struct TrainingFrame {
    pub primary: TrainingRun,
    pub reference: Option<TrainingRun>,
    pub reference_unavailable: Option<String>,
    pub cohort: Option<TrainingCohort>,
    pub cohort_unavailable: Option<String>,
    peer_watches: Vec<PeerWatch>,
    peer_checked: Mutex<Instant>,
    _lease: Lease,
}

impl TrainingFrame {
    /// Next idle wake for currently incomplete peer recordings. A finished
    /// cohort has no ongoing disk checks.
    pub fn peer_recheck_after(&self) -> Option<Duration> {
        (!self.peer_watches.is_empty()).then(|| {
            PEER_RECHECK_INTERVAL.saturating_sub(self.peer_checked.lock().unwrap().elapsed())
        })
    }

    pub(super) fn begin_peer_recheck(&self) -> bool {
        if self.peer_watches.is_empty() {
            return false;
        }
        let mut checked = self.peer_checked.lock().unwrap();
        if checked.elapsed() < PEER_RECHECK_INTERVAL {
            return false;
        }
        *checked = Instant::now();
        true
    }
}

#[derive(PartialEq, Eq)]
struct PeerGeneration {
    committed_index_bytes: u64,
    index_modified: Option<SystemTime>,
    complete: bool,
}

impl PeerGeneration {
    fn read(root: &Path) -> Option<Self> {
        let index = fs::metadata(root.join("index.jsonl")).ok()?;
        if !index.is_file() {
            return None;
        }
        Some(Self {
            committed_index_bytes: index.len(),
            index_modified: index.modified().ok(),
            complete: fs::metadata(root.join("complete.json")).is_ok_and(|status| status.is_file()),
        })
    }
}

struct PeerWatch {
    root: PathBuf,
    generation: PeerGeneration,
}

#[derive(Clone, PartialEq, Eq)]
struct RunKey {
    root: Arc<PathBuf>,
    role: String,
    frontier: [(usize, Option<(u64, u64)>); 2],
}

impl RunKey {
    fn new(run: &RunIndex, role: &str) -> Result<Self> {
        require(
            role.len() <= 256 && run.root().as_os_str().len() <= 2048,
            "Training run or role exceeds admission",
        )?;
        Ok(Self {
            root: run.root().clone(),
            role: role.into(),
            frontier: KINDS.map(|kind| run.candidate_frontier(kind, role, None, u64::MAX)),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct TrainingKey {
    primary: RunKey,
    reference: Option<RunKey>,
    peers: Vec<PathBuf>,
}

impl TrainingKey {
    pub(super) fn new(
        run: &RunIndex,
        reference: Option<(&RunIndex, &str)>,
        peers: &[PathBuf],
    ) -> Result<Self> {
        require(
            peers.len() <= MAX_TRAINING_COHORT_PEERS
                && peers
                    .iter()
                    .all(|path| !path.as_os_str().is_empty() && path.as_os_str().len() <= 2048),
            "Training cohort paths exceed admission",
        )?;
        Ok(Self {
            primary: RunKey::new(run, "online")?,
            reference: reference
                .filter(|(other, role)| other.root() != run.root() || *role != "online")
                .map(|(run, role)| RunKey::new(run, role))
                .transpose()?,
            peers: peers.to_vec(),
        })
    }
}

struct TrainingSource {
    revision: u64,
    sequence: u64,
    kind: RecordKind,
    chunk: ChunkSource,
}

struct Sources {
    key: RunKey,
    manifest: RunManifest,
    evaluations: Vec<TrainingSource>,
    metrics: Vec<TrainingSource>,
    omitted: usize,
}

impl Sources {
    fn new(run: &RunIndex, key: RunKey, metrics: bool, capacity: &mut usize) -> Result<Self> {
        let manifest = run.recording().manifest();
        let charge = manifest_charge(manifest)?;
        require(
            charge <= *capacity,
            "Training run metadata exceeds descriptor budget",
        )?;
        *capacity -= charge;
        let mut omitted = 0;
        let mut collect = |kind: RecordKind, limit| -> Result<Vec<TrainingSource>> {
            let mut sources = Vec::new();
            let total = run.candidate_frontier(kind, &key.role, None, u64::MAX).0;
            let mut eligible = 0;
            let mut scanned = 0;
            for index in run
                .candidates(kind, &key.role, None, u64::MAX)
                .take(4 * limit)
            {
                scanned += 1;
                let entry = &run.recording().entries()[index];
                if kind == RecordKind::Empirical && entry.probe_set.is_some() {
                    continue;
                }
                eligible += 1;
                let metadata = entry_bytes(entry);
                let charge = 512usize
                    .saturating_add(4 * metadata)
                    .saturating_add(2 * key.root.as_os_str().len());
                if sources.len() == limit || metadata > 8192 || charge > *capacity {
                    continue;
                }
                *capacity -= charge;
                sources.push(TrainingSource {
                    revision: entry.revision,
                    sequence: entry.sequence,
                    kind,
                    chunk: run.recording().source(index)?,
                });
            }
            omitted += eligible - sources.len() + total.saturating_sub(scanned);
            Ok(sources)
        };
        let evaluations = collect(RecordKind::Empirical, MAX_EVALUATIONS)?;
        let metrics = if metrics {
            collect(RecordKind::Metrics, MAX_METRIC_SOURCES)?
        } else {
            Vec::new()
        };
        Ok(Self {
            key,
            manifest: manifest.clone(),
            evaluations,
            metrics,
            omitted,
        })
    }

    fn load(&self, reader: &mut Reader<'_>, source: &TrainingSource) -> Result<Arc<Chunk>> {
        let chunk = reader.load(&source.chunk)?;
        require(
            chunk.context.identity.run_id == self.manifest.run_id
                && chunk.context.identity.model_role == self.key.role
                && chunk.context.identity.learner_update_revision == source.revision
                && chunk.record.kind() == source.kind,
            "Training source identity differs from its index entry",
        )?;
        Ok(chunk)
    }

    fn evaluations(
        &self,
        reader: &mut Reader<'_>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedRun> {
        let mut run = TrainingRun {
            root: self.key.root.clone(),
            run_id: self.manifest.run_id.clone(),
            algorithm: self.manifest.algorithm.clone(),
            role: self.key.role.clone(),
            reference_target: None,
            evaluations: Vec::with_capacity(self.evaluations.len()),
            metrics: BTreeMap::new(),
            omitted_sources: self.omitted,
            omitted_metrics: 0,
        };
        let mut protocol = None;
        for source in &self.evaluations {
            checkpoint()?;
            let mut point = TrainingEvaluation {
                clocks: TrainingClocks::missing(source.revision),
                sequence: source.sequence,
                summary: None,
                extrema: None,
                reason: None,
            };
            let prepared: Result<bool> = (|| {
                let chunk = self.load(reader, source)?;
                let Record::Empirical(record) = &chunk.record else {
                    unreachable!("loaded empirical entry")
                };
                if record.evaluation.conditioning != Conditioning::InitialScenario
                    || record.evaluation.initial_action != ActionCondition::CurrentPolicy
                {
                    return Ok(false);
                }
                point.clocks = TrainingClocks::from_context(&chunk.context);
                if crate::reference::provider_role(&self.key.role) {
                    let specification = provider_specification(&chunk)?.ok_or_else(|| {
                        Error::Config(
                            "Reference evaluation has no authored provider provenance".into(),
                        )
                    })?;
                    require(
                        specification.model_role() == self.key.role
                            && manifest_context(&self.manifest)?.as_ref()
                                == Some(&specification.context)
                            && run
                                .reference_target
                                .is_none_or(|target| target == specification.target),
                        "Reference evaluation provenance differs from its role or recording",
                    )?;
                    run.reference_target = Some(specification.target);
                }
                require(
                    value::return_target(&chunk)?.is_none_or(value::ReturnTarget::is_environmental),
                    "Training environmental performance requires environmental returns",
                )?;
                let summary = evaluation_summary(&chunk)?;
                require(
                    record.evaluation.discount == self.manifest.discount,
                    "Evaluation discount differs from recording",
                )?;
                let current = EvaluationProtocol {
                    horizon: evaluation_horizon(&chunk)?,
                    discount: record.evaluation.discount,
                    stochastic: record.evaluation.stochastic_policy,
                };
                require(
                    protocol.is_none_or(|expected| current == expected),
                    "Evaluation protocol changed within the retained history",
                )?;
                protocol = Some(current);
                let mut extrema = [f64::INFINITY, f64::NEG_INFINITY];
                for value in chunk.arrays[&record.returns].to_f64() {
                    extrema[0] = extrema[0].min(value);
                    extrema[1] = extrema[1].max(value);
                }
                point.extrema = Some(extrema);
                point.summary = Some(summary);
                Ok(true)
            })();
            match prepared {
                Ok(false) => continue,
                Ok(true) => {}
                Err(error) => point.reason = Some(short(error)),
            }
            run.evaluations.push(point);
        }
        run.evaluations.reverse();
        Ok(PreparedRun { run, protocol })
    }

    fn metrics(
        &self,
        run: &mut TrainingRun,
        reader: &mut Reader<'_>,
        checkpoint: &dyn Fn() -> Result<()>,
        remaining: &mut usize,
    ) -> Result<()> {
        let manifest_anchor = manifest_context(&self.manifest).ok().flatten();
        for source in &self.metrics {
            checkpoint()?;
            let chunk = match self.load(reader, source) {
                Ok(chunk) => chunk,
                Err(_) => {
                    run.omitted_sources += 1;
                    continue;
                }
            };
            let Record::Metrics(record) = &chunk.record else {
                unreachable!("loaded metrics entry")
            };
            let specification = provider_specification(&chunk).ok().flatten();
            let reference = specification.as_ref().map(|s| s.model_role());
            let clocks = TrainingClocks::from_context(&chunk.context);
            let population = chunk.context.probe_set.as_deref();
            let values = record
                .values
                .iter()
                .map(|(key, value)| (key, Some(*value), None));
            let unavailable = record
                .unavailable
                .iter()
                .map(|(key, reason)| (key, None, Some(reason.as_str())));
            run.omitted_metrics += record
                .values
                .len()
                .saturating_add(record.unavailable.len())
                .saturating_sub(MAX_METRICS_PER_SOURCE);
            for (name, value, reason) in values.chain(unavailable).take(MAX_METRICS_PER_SOURCE) {
                if name.len() > 256 || population.is_some_and(|p| p.len() > 256) {
                    run.omitted_metrics += 1;
                    continue;
                }
                if (name == "reference.optimality_gap"
                    || name.starts_with("reference.optimality_gap."))
                    && !specification
                        .as_ref()
                        .is_some_and(|s| s.target == ReferenceTarget::VerifiedOptimal)
                {
                    run.omitted_metrics += 1;
                    continue;
                }
                if name.starts_with("reference.") && reference.is_none() {
                    run.omitted_metrics += 1;
                    continue;
                }
                if name.starts_with("reference.")
                    && value.is_some()
                    && specification
                        .as_ref()
                        .map(|specification| &specification.context)
                        != manifest_anchor.as_ref()
                {
                    run.omitted_metrics += 1;
                    continue;
                }
                let key = TrainingMetricKey {
                    name: name.clone(),
                    population: population.map(str::to_owned),
                    reference: reference.clone(),
                };
                let key_charge = if run.metrics.contains_key(&key) {
                    0
                } else {
                    if run.metrics.len() == MAX_METRIC_KEYS {
                        run.omitted_metrics += 1;
                        continue;
                    }
                    320 + name.len()
                        + population.map_or(0, str::len)
                        + reference.as_ref().map_or(0, String::len)
                };
                let reason = reason.map(short);
                let charge = key_charge
                    + 2 * std::mem::size_of::<TrainingMetricPoint>()
                    + reason.as_ref().map_or(0, String::capacity);
                if charge > *remaining {
                    run.omitted_metrics += 1;
                    continue;
                }
                *remaining -= charge;
                run.metrics
                    .entry(key)
                    .or_default()
                    .push(TrainingMetricPoint {
                        clocks,
                        sequence: source.sequence,
                        value,
                        reason,
                    });
            }
        }
        for points in run.metrics.values_mut() {
            points.reverse();
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
struct EvaluationProtocol {
    horizon: EvaluationHorizon,
    discount: f64,
    stochastic: bool,
}

struct PreparedRun {
    run: TrainingRun,
    protocol: Option<EvaluationProtocol>,
}

pub(super) struct TrainingRequest {
    primary: Sources,
    reference: Option<Sources>,
    reference_unavailable: Option<String>,
    peers: Vec<PathBuf>,
    previous: Option<Arc<TrainingFrame>>,
}

impl TrainingRequest {
    pub(super) fn new(
        run: &RunIndex,
        reference: Option<(&RunIndex, &str)>,
        key: TrainingKey,
        previous: Option<Arc<TrainingFrame>>,
    ) -> Result<Self> {
        let mut budget = if key.reference.is_some() {
            DESCRIPTOR_BYTES / 2
        } else {
            DESCRIPTOR_BYTES
        };
        let primary = Sources::new(run, key.primary, true, &mut budget)?;
        let mut reference_budget = DESCRIPTOR_BYTES / 2 + budget;
        let other = reference
            .zip(key.reference)
            .map(|((run, _), key)| Sources::new(run, key, true, &mut reference_budget))
            .transpose();
        let (reference, reference_unavailable) = match other {
            Ok(sources) => (sources, None),
            Err(error) => (None, Some(short(error))),
        };
        Ok(Self {
            primary,
            reference,
            reference_unavailable,
            peers: key.peers,
            previous,
        })
    }

    pub(super) fn prepare(
        &self,
        shared: &Shared,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<Arc<TrainingFrame>> {
        if let Some(previous) = &self.previous {
            let mut unchanged = true;
            for peer in &previous.peer_watches {
                checkpoint()?;
                if PeerGeneration::read(&peer.root).as_ref() != Some(&peer.generation) {
                    unchanged = false;
                    break;
                }
            }
            if unchanged {
                *previous.peer_checked.lock().unwrap() = Instant::now();
                return Ok(previous.clone());
            }
        }
        // Take the generation before opening indexes. An append concurrent
        // with preparation is either captured now or visible to the next check.
        // Seed checks and comparison failures must not hide future peer appends.
        let mut peer_watches = Vec::new();
        for root in &self.peers {
            checkpoint()?;
            if let Some(generation) =
                PeerGeneration::read(root).filter(|generation| !generation.complete)
            {
                peer_watches.push(PeerWatch {
                    root: root.clone(),
                    generation,
                });
            }
        }
        let mut reader = Reader::cached(shared, checkpoint);
        let mut primary = self.primary.evaluations(&mut reader, checkpoint)?;
        let mut reference = None;
        let mut reference_unavailable = self.reference_unavailable.clone();
        if let Some(sources) = &self.reference {
            let result = (|| {
                let other = sources.evaluations(&mut reader, checkpoint)?;
                let external = self.primary.key.root != sources.key.root;
                compare_protocol(
                    &self.primary.manifest,
                    &primary,
                    &sources.manifest,
                    &other,
                    external,
                    false,
                )?;
                Ok::<_, Error>(other.run)
            })();
            match result {
                Ok(run) => reference = Some(run),
                Err(error) => reference_unavailable = Some(short(error)),
            }
        }
        let (cohort, cohort_unavailable) = if self.peers.is_empty() {
            (None, None)
        } else {
            match self.cohort(&primary, &mut reader, checkpoint) {
                Ok(cohort) => (Some(cohort), None),
                Err(error) => (None, Some(short(error))),
            }
        };
        let mut metric_bytes = METRIC_BYTES;
        self.primary
            .metrics(&mut primary.run, &mut reader, checkpoint, &mut metric_bytes)?;
        if let Some((run, sources)) = reference.as_mut().zip(self.reference.as_ref()) {
            sources.metrics(run, &mut reader, checkpoint, &mut metric_bytes)?;
        }
        checkpoint()?;
        let charge = METRIC_BYTES - metric_bytes
            + published_charge(&primary.run)
            + reference.as_ref().map_or(0, published_charge)
            + cohort.as_ref().map_or(0, |cohort| {
                4096 + cohort
                    .series
                    .values()
                    .map(|points| points.capacity() * std::mem::size_of::<TrainingAggregatePoint>())
                    .sum::<usize>()
            })
            + peer_watches
                .iter()
                .map(|peer| 256 + 2 * peer.root.as_os_str().len())
                .sum::<usize>();
        let lease = shared.reserve_training(charge)?;
        Ok(Arc::new(TrainingFrame {
            primary: primary.run,
            reference,
            reference_unavailable,
            cohort,
            cohort_unavailable,
            peer_watches,
            peer_checked: Mutex::new(Instant::now()),
            _lease: lease,
        }))
    }

    fn cohort(
        &self,
        primary: &PreparedRun,
        reader: &mut Reader<'_>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<TrainingCohort> {
        let manifest = &self.primary.manifest;
        require(
            manifest.starting_revision == 0,
            "Cohort requires independent runs starting at training revision zero",
        )?;
        let mut ids = BTreeSet::from([manifest.run_id.clone()]);
        let mut seeds = BTreeSet::from([training_seed(manifest)?]);
        let mut roots = BTreeSet::from([primary.run.root.clone()]);
        let mut peers = Vec::with_capacity(self.peers.len());
        for path in &self.peers {
            checkpoint()?;
            let index = RunIndex::open(
                path,
                ReadLimits {
                    max_index_entries: 8192,
                    max_index_bytes: 2 * 1024 * 1024,
                    max_metadata_bytes: 64 * 1024,
                    max_chunk_bytes: MAX_READ_BYTES,
                },
            )?;
            let other = index.recording().manifest();
            require(
                roots.insert(index.root().clone()) && ids.insert(other.run_id.clone()),
                "A recording appears more than once in this training cohort",
            )?;
            require(
                seeds.insert(training_seed(other)?),
                "Cohort training seeds must be distinct",
            )?;
            require(
                other.starting_revision == 0
                    && other.algorithm == manifest.algorithm
                    && other.model_id == manifest.model_id
                    && other.experiment_id == manifest.experiment_id
                    && manifest.metadata.get("learner.configuration").is_some_and(
                        |configuration| {
                            other.metadata.get("learner.configuration") == Some(configuration)
                        },
                    ),
                "Cohort runs require the same authored experiment, algorithm and learner configuration",
            )?;
            let mut budget = DESCRIPTOR_BYTES;
            let sources = Sources::new(&index, RunKey::new(&index, "online")?, false, &mut budget)?;
            let run = sources.evaluations(reader, checkpoint)?;
            compare_protocol(manifest, primary, other, &run, true, true)?;
            peers.push(run.run);
        }
        // Keeping only prepared rows lets each peer index retire before opening the next.
        let runs = std::iter::once(&primary.run)
            .chain(&peers)
            .collect::<Vec<_>>();
        let mut series = BTreeMap::new();
        for clock in TrainingClock::ALL {
            checkpoint()?;
            series.insert(clock, super::math::aggregate_training_runs(&runs, clock)?);
        }
        require(
            series.values().any(|points| !points.is_empty()),
            "Cohort runs have no common observed clock support",
        )?;
        Ok(TrainingCohort {
            members: runs.iter().map(|run| run.run_id.clone()).collect(),
            series,
        })
    }
}

fn compare_protocol(
    a: &RunManifest,
    a_run: &PreparedRun,
    b: &RunManifest,
    b_run: &PreparedRun,
    external: bool,
    cohort: bool,
) -> Result<()> {
    let (Some(a_eval), Some(b_eval)) = (a_run.protocol, b_run.protocol) else {
        return Err(Error::Config(
            "A selected run or role has no complete environmental evaluation".into(),
        ));
    };
    require(
        a_eval.discount == b_eval.discount
            && a_eval.horizon == b_eval.horizon
            && (!cohort || a_eval.stochastic == b_eval.stochastic),
        "Evaluation discount, horizon or cohort sampling mode differs",
    )?;
    if external {
        let context = |manifest| -> Result<ReferenceContext> {
            manifest_context(manifest)?.ok_or_else(|| {
                Error::Config(
                    "Cross-run performance requires an authored scientific context".into(),
                )
            })
        };
        require(
            context(a)? == context(b)?,
            "Cross-run scientific environment or return conventions differ",
        )?;
        require(
            a.metadata.get(SCENARIO_PROTOCOL_KEY).is_some_and(|key| {
                !key.trim().is_empty()
                    && key.len() <= 1024
                    && b.metadata.get(SCENARIO_PROTOCOL_KEY) == Some(key)
            }),
            "Cross-run performance requires the same authored initial-scenario protocol",
        )?;
    }
    Ok(())
}

fn training_seed(manifest: &RunManifest) -> Result<u64> {
    manifest
        .metadata
        .get("training.seed")
        .and_then(|seed| seed.parse().ok())
        .ok_or_else(|| {
            Error::Config(
                "Cohort requires an explicitly recorded independent training.seed for each run"
                    .into(),
            )
        })
}

fn published_charge(run: &TrainingRun) -> usize {
    4096 + run.root.as_os_str().len() * 2
        + run.run_id.capacity()
        + run.algorithm.capacity()
        + run.role.capacity()
        + run.evaluations.capacity() * std::mem::size_of::<TrainingEvaluation>()
        + run
            .evaluations
            .iter()
            .filter_map(|point| point.reason.as_ref())
            .map(String::capacity)
            .sum::<usize>()
}

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Config(reason.into()))
    }
}

fn short(value: impl std::fmt::Display) -> String {
    value.to_string().chars().take(256).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_context_has_only_its_revision_unless_a_collection_clock_was_recorded() {
        let mut context = RecordContext::new("legacy", 5);
        let legacy = TrainingClocks::from_context(&context);
        assert_eq!(legacy.value(TrainingClock::Revision), Some(5.));
        assert_eq!(legacy.value(TrainingClock::EnvironmentDecisions), None);
        assert_eq!(legacy.value(TrainingClock::PhysicalTicks), None);
        assert_eq!(legacy.value(TrainingClock::WallSeconds), None);

        context.environment_step = Some(0);
        let initial = TrainingClocks::from_context(&context);
        assert_eq!(initial.value(TrainingClock::EnvironmentDecisions), Some(0.));
        assert_eq!(initial.value(TrainingClock::PhysicalTicks), Some(0.));
        assert_eq!(initial.value(TrainingClock::WallSeconds), None);
    }
}
