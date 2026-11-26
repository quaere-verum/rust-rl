use super::{
    empirical::{self, EMPIRICAL_DECLARATION_VERSION, EmpiricalDeclaration},
    evaluation::{
        self, BranchingEnv, EvaluationConfig, EvaluationHorizon, EvaluationResult, FrameSource,
        RenderCaptureConfig,
    },
    observer::append_metrics,
    probes::ProbeSet,
    reference::{self, ActionGradient, ReferenceContext, ReferenceProvider},
    render::RenderSource,
    schedule::Interval,
    schema::{Chunk, EventRecord, ProbePopulation, Record, RecordContext, RunManifest},
    storage::{BufferConfig, Recorder, RecordingStatus, StorageConfig},
    value::{ContinuationIdentity, ValueConventions},
    visualization::EnvironmentDescriptor,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Env, Observation, ObservationSpace},
        learner::UpdateReport,
        temporal::TemporalTransition,
    },
    train::{
        counters::RunCounters,
        observer::{UpdateBoundary, UpdateObserver},
    },
};
use std::{
    collections::BTreeMap,
    marker::PhantomData,
    path::{Path, PathBuf},
    time::Instant,
};

mod references;
use references::ReferenceCapture;

/// Metadata supplied by a learner adapter. On-policy return settings supplied
/// separately by its trainer should be provided on the session builder.
pub struct LearnerInfo {
    pub algorithm: &'static str,
    pub model: &'static str,
    pub configuration: serde_json::Value,
    pub discount: Option<f64>,
}

/// Optional recording capabilities at an immutable completed revision.
/// Recorded predictions and reconstructable snapshots are independent.
pub trait DiagnosticLearner {
    fn diagnostic_info(&self) -> Result<LearnerInfo>;
    fn diagnostic_revision(&self) -> u64;
    fn diagnostic_evaluation_mode(&self, requested: ActMode) -> ActMode {
        requested
    }
    fn diagnostic_action_gradient(
        &self,
        _critic: &str,
        _observation: &Observation,
        _action: &Action,
    ) -> Result<Option<ActionGradient>> {
        Ok(None)
    }
    fn diagnostic_conditional_actor(
        &self,
        _context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(None)
    }
    /// An evaluation-only actor without mutable factual training state.
    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>>;
    fn diagnostic_metrics(&self, _context: &RecordContext) -> Result<Option<Chunk>> {
        Ok(None)
    }
    fn diagnostic_records(&self, _request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        Ok(Vec::new())
    }
}

/// Bounded, revision-bound inputs to a learner adapter.
pub struct CaptureRequest<'a> {
    pub context: &'a RecordContext,
    pub probes: &'a [ProbeSet],
    pub predictions: bool,
    pub representations: bool,
    pub snapshots: bool,
    pub max_activation_values: usize,
    /// Maximum uncompressed chunk size from the session's storage configuration.
    pub max_chunk_bytes: usize,
    pub conventions: Option<&'a ValueConventions>,
    /// The action law requested for the independently evaluated learner.
    pub evaluation_mode: ActMode,
    pub policy_seed: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagnosticSchedule {
    pub metrics: Option<Interval>,
    pub probes: Option<Interval>,
    pub representations: Option<Interval>,
    pub evaluation: Option<Interval>,
    pub conditional_evaluation: Option<Interval>,
    pub reference: Option<Interval>,
    pub snapshots: Option<Interval>,
    pub initial: bool,
    pub final_checkpoint: bool,
}

#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    pub schedule: DiagnosticSchedule,
    pub evaluation: EvaluationConfig,
    pub evaluation_horizon: EvaluationHorizon,
    pub storage: StorageConfig,
    pub buffering: Option<BufferConfig>,
    pub snapshot_retention: usize,
    pub max_probe_sets: usize,
    pub max_probe_rows: usize,
    pub visited_probe_rows: usize,
    pub max_activation_values: usize,
    pub max_conditional_cases: usize,
    pub reference_action_samples: usize,
    pub conditional_evaluation: EvaluationConfig,
    pub seed: u64,
}

fn every(revisions: u64) -> Option<Interval> {
    Some(Interval {
        first: revisions,
        every: revisions,
    })
}

fn evaluation_config(episodes: u64, max_steps: usize, trajectories: usize) -> EvaluationConfig {
    EvaluationConfig {
        seeds: (0..episodes).map(|n| 0x8d2d_2100 + n).collect(),
        policy_seed: 0xad9a_4251,
        max_steps,
        trajectories,
        mode: ActMode::Sample,
        allow_truncated_prefix: true,
        ..Default::default()
    }
}

impl DiagnosticConfig {
    pub fn light() -> Self {
        Self {
            schedule: DiagnosticSchedule {
                metrics: every(1),
                probes: None,
                representations: None,
                evaluation: every(50),
                conditional_evaluation: None,
                reference: None,
                snapshots: None,
                initial: true,
                final_checkpoint: true,
            },
            evaluation: evaluation_config(4, 256, 1),
            evaluation_horizon: EvaluationHorizon::default(),
            storage: StorageConfig {
                max_total_bytes: 64 * 1024 * 1024,
                ..Default::default()
            },
            buffering: Some(BufferConfig::default()),
            snapshot_retention: 8,
            max_probe_sets: 4,
            max_probe_rows: 32,
            visited_probe_rows: 0,
            max_activation_values: 8192,
            max_conditional_cases: 8,
            reference_action_samples: 16,
            conditional_evaluation: evaluation_config(32, 128, 0),
            seed: 0x549b_5137,
        }
    }

    pub fn standard() -> Self {
        let mut config = Self::light();
        config.schedule.probes = every(20);
        config.schedule.snapshots = every(100);
        config.schedule.reference = every(50);
        config.storage.max_total_bytes = 256 * 1024 * 1024;
        config
    }

    pub fn detailed() -> Self {
        let mut config = Self::standard();
        config.schedule.representations = every(50);
        config.schedule.conditional_evaluation = every(100);
        config.evaluation = evaluation_config(8, 512, 2);
        config.max_probe_rows = 128;
        config.visited_probe_rows = 16;
        config.max_activation_values = 65_536;
        config.storage.max_total_bytes = 512 * 1024 * 1024;
        config
    }

    fn provenance(&self) -> serde_json::Value {
        fn evaluation(config: &EvaluationConfig) -> serde_json::Value {
            serde_json::json!({
                "scenario_seeds": config.seeds,
                "policy_seed": config.policy_seed,
                "requested_mode": format!("{:?}", config.mode),
                "max_decisions": config.max_steps,
                "retained_trajectories": config.trajectories,
                "allow_truncated_prefix": config.allow_truncated_prefix,
            })
        }

        serde_json::json!({
            "schedule": self.schedule,
            "evaluation": evaluation(&self.evaluation),
            "evaluation_horizon": format!("{:?}", self.evaluation_horizon),
            "conditional_evaluation": evaluation(&self.conditional_evaluation),
            "storage": self.storage,
            "buffering": self.buffering,
            "snapshot_retention_per_role": self.snapshot_retention,
            "max_probe_sets": self.max_probe_sets,
            "max_probe_rows": self.max_probe_rows,
            "visited_probe_rows": self.visited_probe_rows,
            "max_activation_values": self.max_activation_values,
            "max_conditional_cases": self.max_conditional_cases,
            "reference_action_samples": self.reference_action_samples,
            "diagnostic_seed": self.seed,
        })
    }

    fn validate(&self) -> Result<()> {
        for interval in self.schedule.intervals().into_iter().flatten() {
            interval.validate()?;
        }
        for config in [&self.evaluation, &self.conditional_evaluation] {
            config.validate()?;
            ensure(
                config.seeds.len() <= empirical::MAX_EMPIRICAL_SAMPLES
                    && config.seeds.len().saturating_mul(config.max_steps) <= 1_000_000
                    && config.trajectories.saturating_mul(config.max_steps) <= 65_536,
                "diagnostic evaluation exceeds the scenario or trajectory budget",
            )?;
        }
        if let EvaluationHorizon::FiniteDecisions(horizon) = self.evaluation_horizon {
            ensure(
                (1..=self.evaluation.max_steps).contains(&horizon),
                "finite evaluation horizon exceeds the independent evaluation guard",
            )?;
        }
        ensure(
            (1..=32).contains(&self.max_probe_sets)
                && (1..=65_536).contains(&self.max_probe_rows)
                && self.visited_probe_rows <= self.max_probe_rows
                && self
                    .reference_action_samples
                    .saturating_mul(self.max_probe_rows)
                    .saturating_mul(self.max_probe_sets)
                    <= 1_000_000
                && (1..=1_048_576).contains(&self.max_activation_values)
                && self.max_conditional_cases <= 128
                && (2..=128).contains(&self.reference_action_samples)
                && self
                    .max_conditional_cases
                    .saturating_mul(self.conditional_evaluation.seeds.len())
                    .saturating_mul(self.conditional_evaluation.max_steps)
                    <= 1_000_000
                && self.snapshot_retention > 0,
            "invalid diagnostic capture budget",
        )
    }
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self::standard()
    }
}

#[derive(Clone, Copy)]
struct Channels([bool; 7]);

impl Channels {
    fn any(self) -> bool {
        self.0.into_iter().any(|enabled| enabled)
    }
}

impl DiagnosticSchedule {
    fn intervals(&self) -> [Option<Interval>; 7] {
        [
            self.metrics,
            self.probes,
            self.representations,
            self.evaluation,
            self.conditional_evaluation,
            self.reference,
            self.snapshots,
        ]
    }

    fn due(&self, previous: Option<u64>, revision: u64, force: bool) -> Result<Channels> {
        let mut channels = [false; 7];
        for (due, interval) in channels.iter_mut().zip(self.intervals()) {
            if let Some(interval) = interval {
                *due = force || interval.due(previous, revision)?;
            }
        }
        Ok(Channels(channels))
    }
}

type EvaluationRunner<'a> = Box<
    dyn FnMut(&dyn Actor, &EvaluationConfig, EvaluationHorizon) -> Result<EvaluationResult> + 'a,
>;

type ConditionalRunner<'a> =
    Box<dyn Fn(&dyn Actor, &EvaluationConfig) -> Result<EvaluationResult> + 'a>;

struct ConditionalCase<'a> {
    probe_set: String,
    row: usize,
    observation: Observation,
    evaluate: ConditionalRunner<'a>,
}

pub struct DiagnosticSessionBuilder<'a> {
    path: PathBuf,
    environment: Option<(ObservationSpace, ActionSpace, String)>,
    environment_name: Option<String>,
    experiment: Option<String>,
    discount: Option<f64>,
    config: DiagnosticConfig,
    metadata: BTreeMap<String, String>,
    descriptor: Option<EnvironmentDescriptor>,
    conventions: Option<ValueConventions>,
    evaluation: Option<EvaluationRunner<'a>>,
    reference: Option<Box<dyn ReferenceProvider + 'a>>,
    reference_trajectories: usize,
    reference_context: Option<ReferenceContext>,
    conditional: Vec<ConditionalCase<'a>>,
    probes: Vec<ProbeSet>,
}

impl<'a> DiagnosticSessionBuilder<'a> {
    /// Read shapes and identity without resetting or stepping the training environment.
    pub fn environment(mut self, environment: &impl Env) -> Self {
        self.environment = Some((
            environment.observation_space(),
            environment.action_space().clone(),
            std::any::type_name_of_val(environment).into(),
        ));
        self
    }

    pub fn environment_name(mut self, name: impl Into<String>) -> Self {
        self.environment_name = Some(name.into());
        self
    }

    pub fn experiment(mut self, name: impl Into<String>) -> Self {
        self.experiment = Some(name.into());
        self
    }

    pub fn discount(mut self, discount: f64) -> Self {
        self.discount = Some(discount);
        self
    }

    pub fn config(mut self, config: DiagnosticConfig) -> Self {
        self.config = config;
        self
    }

    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn environment_descriptor(mut self, descriptor: EnvironmentDescriptor) -> Self {
        self.descriptor = Some(descriptor);
        self
    }

    pub fn value_conventions(mut self, conventions: ValueConventions) -> Self {
        self.conventions = Some(conventions);
        self
    }

    pub fn evaluation_factory<E: Env + 'a>(
        mut self,
        mut factory: impl FnMut(u64) -> Result<E> + 'a,
    ) -> Self {
        self.evaluation = Some(Box::new(move |actor, config, horizon| {
            evaluation::evaluate_with_horizon(&mut factory, actor, config, horizon)
        }));
        self
    }

    pub fn evaluation_factory_with_frames<E: Env + FrameSource + 'a>(
        mut self,
        mut factory: impl FnMut(u64) -> Result<E> + 'a,
    ) -> Self {
        self.evaluation = Some(Box::new(move |actor, config, horizon| {
            evaluation::evaluate_with_frames_and_horizon(
                &mut factory,
                actor,
                config,
                horizon,
                |_| {},
            )
        }));
        self
    }

    pub fn evaluation_factory_with_render<E: Env + RenderSource + 'a>(
        mut self,
        mut factory: impl FnMut(u64) -> Result<E> + 'a,
        capture: RenderCaptureConfig,
    ) -> Self {
        self.evaluation = Some(Box::new(move |actor, config, horizon| {
            let mut capture = capture.clone();
            capture.trajectories = capture.trajectories.min(config.trajectories);
            evaluation::evaluate_with_render_and_horizon(
                &mut factory,
                actor,
                config,
                horizon,
                &capture,
                |_| {},
            )
        }));
        self
    }

    /// Capture numerical telemetry and render scenes from the same evaluation episodes.
    pub fn evaluation_factory_with_frames_and_render<E: Env + FrameSource + RenderSource + 'a>(
        mut self,
        mut factory: impl FnMut(u64) -> Result<E> + 'a,
        capture: RenderCaptureConfig,
    ) -> Self {
        self.evaluation = Some(Box::new(move |actor, config, horizon| {
            let mut capture = capture.clone();
            capture.trajectories = capture.trajectories.min(config.trajectories);
            evaluation::evaluate_with_frames_and_render_and_horizon(
                &mut factory,
                actor,
                config,
                horizon,
                &capture,
                |_| {},
            )
        }));
        self
    }

    /// Declare the scientific context of the actual evaluation environment.
    pub fn reference_context(mut self, context: ReferenceContext) -> Self {
        self.reference_context = Some(context);
        self
    }

    pub fn reference(mut self, provider: impl ReferenceProvider + 'a) -> Self {
        self.reference = Some(Box::new(provider));
        self
    }

    /// Retain up to this many reference episodes from the same initial scenarios
    /// as recorded learner trajectories. Fixed references record episodes once.
    pub fn reference_trajectories(mut self, limit: usize) -> Self {
        self.reference_trajectories = limit;
        self
    }

    /// Branch from a private simulator state at an existing probe row.
    /// This evidence is explicitly conditioned on that simulator state.
    pub fn conditional_case<E: BranchingEnv + 'a>(
        mut self,
        probe_set: impl Into<String>,
        row: usize,
        source: E,
        first_action: Option<Action>,
    ) -> Result<Self> {
        self.conditional.push(ConditionalCase {
            probe_set: probe_set.into(),
            row,
            observation: source.branch_observation()?,
            evaluate: Box::new(move |actor, config| {
                evaluation::evaluate_conditional(&source, actor, first_action.as_ref(), config)
            }),
        });
        Ok(self)
    }

    /// Supply a fresh environment sampled from the declared observation
    /// condition for each independent future seed.
    pub fn conditional_observation_case<E: Env + 'a>(
        mut self,
        probe_set: impl Into<String>,
        row: usize,
        observation: Observation,
        first_action: Option<Action>,
        methodology: impl Into<String>,
        factory: impl Fn(u64) -> Result<(E, Observation)> + 'a,
    ) -> Self {
        let condition = observation.clone();
        let methodology = methodology.into();
        self.conditional.push(ConditionalCase {
            probe_set: probe_set.into(),
            row,
            observation,
            evaluate: Box::new(move |actor, config| {
                evaluation::evaluate_observation_conditioned(
                    &factory,
                    &condition,
                    first_action.as_ref(),
                    actor,
                    config,
                    &methodology,
                )
            }),
        });
        self
    }

    pub fn probe_set(mut self, probes: ProbeSet) -> Self {
        self.probes.push(probes);
        self
    }

    pub fn build<L: DiagnosticLearner + ?Sized>(
        mut self,
        learner: &L,
    ) -> Result<DiagnosticSession<'a, L>> {
        self.config.validate()?;
        let (observations, space, default_environment) = self.environment.ok_or_else(|| {
            Error::Config("diagnostic session needs an observation and action space".into())
        })?;
        let width = observations.dense_width().unwrap_or(0);
        let action_width = match &space {
            ActionSpace::Discrete { count } => *count,
            space => space.dimensions(),
        };
        let bytes_per_row = width.saturating_add(action_width).saturating_mul(8);
        let mut probe_rows = self
            .probes
            .iter()
            .map(|set| set.observations().len())
            .max()
            .unwrap_or(0);
        if self.evaluation.is_some() {
            if self.probes.is_empty()
                && (self.config.schedule.probes.is_some()
                    || self.config.schedule.representations.is_some())
            {
                probe_rows = self.config.max_probe_rows;
            }
            probe_rows = probe_rows.max(self.config.visited_probe_rows);
        }
        let retained_trajectory_rows = if self.evaluation.is_some() {
            self.config
                .evaluation
                .trajectories
                .saturating_mul(self.config.evaluation.max_steps)
        } else {
            0
        };
        ensure(
            probe_rows.saturating_mul(bytes_per_row) <= self.config.storage.max_chunk_bytes
                && retained_trajectory_rows.saturating_mul(bytes_per_row)
                    <= self.config.storage.max_chunk_bytes,
            "diagnostic probe or retained trajectory dimensions exceed the in-memory capture budget",
        )?;
        let id = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| {
                Error::Config("diagnostic session needs a recording directory name".into())
            })?;
        if let Some(context) = &self.reference_context {
            self.environment_name
                .get_or_insert_with(|| context.environment.clone());
            self.discount.get_or_insert(context.conventions.discount);
            self.conventions
                .get_or_insert_with(|| context.conventions.clone());
        }
        let info = learner.diagnostic_info()?;
        let discount = self
            .discount
            .or(info.discount)
            .unwrap_or(self.config.evaluation.discount);
        if let Some(expected) = info.discount {
            ensure(
                expected == discount,
                "session discount differs from learner return discount",
            )?;
        }
        if let Some(conventions) = &self.conventions {
            ensure(
                conventions.discount == discount,
                "session and value convention discounts differ",
            )?;
        }
        let mut manifest = RunManifest::new_in(
            id,
            self.experiment.as_deref().unwrap_or(id),
            info.algorithm,
            observations.clone(),
            space.clone(),
            discount,
        )?;
        manifest.model_id = info.model.into();
        manifest.starting_revision = learner.diagnostic_revision();
        manifest.environment = self.environment_name.unwrap_or(default_environment);
        manifest.environment_descriptor = self.descriptor;
        if let Some(conventions) = &self.conventions {
            manifest
                .reward_convention
                .clone_from(&conventions.reward_convention);
        }
        manifest.metadata = self.metadata;
        manifest.metadata.insert(
            "learner.configuration".into(),
            info.configuration.to_string(),
        );
        manifest.metadata.insert(
            "diagnostics.configuration".into(),
            self.config.provenance().to_string(),
        );
        self.config.evaluation.discount = discount;
        self.config.conditional_evaluation.discount = discount;
        let extra_population = usize::from(self.config.visited_probe_rows > 0);
        ensure(
            self.probes.len() + extra_population <= self.config.max_probe_sets,
            "too many diagnostic populations",
        )?;
        ensure(
            self.config.visited_probe_rows == 0
                || self.probes.iter().all(|p| p.id() != "evaluation-visited"),
            "evaluation-visited is reserved for the sampled policy population",
        )?;
        for (index, probes) in self.probes.iter().enumerate() {
            ensure(
                probes.observations().len() <= self.config.max_probe_rows
                    && probes.observation_space() == &observations
                    && probes.action_space() == &space
                    && !self.probes[..index].iter().any(|p| p.id() == probes.id()),
                "invalid, duplicate or oversized diagnostic population",
            )?;
        }
        if let Some(context) = &self.reference_context {
            ensure(
                self.conventions.as_ref() == Some(&context.conventions),
                "session value conventions differ from the reference context",
            )?;
            reference::attach_manifest_context(&mut manifest, context)?;
        }
        let reference = self
            .reference
            .map(|provider| {
                ReferenceCapture::new(
                    provider,
                    self.reference_context.as_ref(),
                    self.reference_trajectories,
                )
            })
            .transpose()?;
        ensure(
            self.conditional.len() <= self.config.max_conditional_cases,
            "too many conditional diagnostic cases",
        )?;
        ensure(
            self.conditional.is_empty() || observations.dense_width().is_some(),
            "typed conditional returns require an input-aware empirical record",
        )?;
        for case in &self.conditional {
            ensure(
                self.probes
                    .iter()
                    .find(|p| p.id() == case.probe_set)
                    .and_then(|p| p.observations().get(case.row))
                    == Some(&case.observation),
                "conditional simulator observation differs from its diagnostic probe row",
            )?;
        }
        let mut recorder = Recorder::create(&self.path, manifest, self.config.storage.clone())?
            .with_snapshot_retention(self.config.snapshot_retention)?;
        if let Some(buffering) = self.config.buffering {
            recorder = recorder.with_buffering(buffering)?;
        }
        let mut session = DiagnosticSession {
            recorder,
            wall_start: Instant::now(),
            config: self.config,
            conventions: self.conventions,
            evaluation: self.evaluation,
            reference,
            scientific_context: self.reference_context,
            conditional: self.conditional,
            probes: self.probes,
            previous: Some(learner.diagnostic_revision()),
            captured: Channels([false; 7]),
            last: None,
            stats: DiagnosticTiming::default(),
            finished: false,
            failed: false,
            learner: PhantomData,
        };
        if session.config.schedule.initial {
            let info = BoundaryInfo::default();
            session.record(learner, &info, true, "initial")?;
            session.recorder.flush()?;
        }
        Ok(session)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DiagnosticTiming {
    pub total_seconds: f64,
    pub evaluation_seconds: f64,
    pub inference_seconds: f64,
    pub reference_seconds: f64,
    pub conditional_seconds: f64,
}

/// Owns a recording writer for one training stream. Attach it to the trainer's
/// existing immutable update observer and finish it with the final learner.
pub struct DiagnosticSession<'a, L: ?Sized> {
    recorder: Recorder,
    wall_start: Instant,
    config: DiagnosticConfig,
    conventions: Option<ValueConventions>,
    evaluation: Option<EvaluationRunner<'a>>,
    reference: Option<ReferenceCapture<'a>>,
    scientific_context: Option<ReferenceContext>,
    conditional: Vec<ConditionalCase<'a>>,
    probes: Vec<ProbeSet>,
    previous: Option<u64>,
    captured: Channels,
    last: Option<BoundaryInfo>,
    stats: DiagnosticTiming,
    finished: bool,
    failed: bool,
    learner: PhantomData<fn(&L)>,
}

impl<'a> DiagnosticSession<'a, ()> {
    pub fn builder(path: impl AsRef<Path>) -> DiagnosticSessionBuilder<'a> {
        DiagnosticSessionBuilder {
            path: path.as_ref().to_owned(),
            environment: None,
            environment_name: None,
            experiment: None,
            discount: None,
            config: DiagnosticConfig::default(),
            metadata: BTreeMap::new(),
            descriptor: None,
            conventions: None,
            evaluation: None,
            reference: None,
            reference_trajectories: 0,
            reference_context: None,
            conditional: Vec::new(),
            probes: Vec::new(),
        }
    }
}

#[derive(Default)]
struct BoundaryInfo {
    counters: RunCounters,
    learner_call: u64,
    collection_cycle: u64,
    report: UpdateReport,
    collection_seconds: f64,
    learning_seconds: f64,
}

impl BoundaryInfo {
    fn borrowed(&self) -> UpdateBoundary<'_> {
        UpdateBoundary {
            counters: &self.counters,
            learner_call: self.learner_call,
            collection_cycle: self.collection_cycle,
            report: &self.report,
            collection_seconds: self.collection_seconds,
            learning_seconds: self.learning_seconds,
        }
    }
}

impl<L: DiagnosticLearner + ?Sized> DiagnosticSession<'_, L> {
    pub fn manifest(&self) -> &RunManifest {
        self.recorder.manifest()
    }

    pub fn status(&self) -> &RecordingStatus {
        self.recorder.status()
    }

    pub fn timing(&self) -> DiagnosticTiming {
        self.stats
    }

    pub fn flush(&mut self) -> Result<()> {
        self.recorder.flush()
    }

    /// Record an explicit completed revision, including learners fitted outside
    /// minibatch trainers. Counters describe the caller's collection stream.
    pub fn checkpoint(&mut self, learner: &L, counters: RunCounters) -> Result<()> {
        let boundary = BoundaryInfo {
            counters,
            ..Default::default()
        };
        self.record(learner, &boundary, true, "explicit")?;
        self.last = Some(boundary);
        self.flush()
    }

    /// Publish caller-owned factual coverage at its completed learner boundary.
    /// This population is independent of fixed scientific probes and evaluation visits.
    pub fn record_history_probe(
        &mut self,
        learner: &L,
        mut counters: RunCounters,
        probes: &ProbeSet,
    ) -> Result<()> {
        let revision = learner.diagnostic_revision();
        ensure(
            !self.finished && !self.failed && self.previous == Some(revision),
            "history probes require the session's active learner boundary",
        )?;
        ensure(
            matches!(
                probes.population(),
                ProbePopulation::ImportedHistory | ProbePopulation::ReplayRetained
            ) && probes.action_space() == &self.manifest().action_space
                && *probes.observation_space() == self.manifest().observation_space()
                && probes.observations().len() <= self.config.max_probe_rows,
            "history probe population or encoding exceeds session admission",
        )?;
        let mut context = RecordContext::new(&self.manifest().run_id, revision);
        counters.updates = revision - self.manifest().starting_revision;
        context.counters = counters;
        context.environment_step = Some(counters.env_steps);
        context.wall_elapsed_seconds = Some(self.wall_start.elapsed().as_secs_f64());
        self.recorder
            .append(&probes.record_with_limit(context, self.config.storage.max_chunk_bytes)?)?;
        Ok(())
    }

    /// Publish collected transitions from one episode beginning at its factual reset.
    /// A live prefix stays distinct from an independently evaluated return.
    pub fn record_factual_episode(
        &mut self,
        learner: &L,
        mut counters: RunCounters,
        id: &str,
        transitions: &[TemporalTransition],
    ) -> Result<()> {
        let revision = learner.diagnostic_revision();
        ensure(
            !self.finished
                && !self.failed
                && self.previous == Some(revision)
                && self.manifest().typed_observation_space.is_some()
                && !id.trim().is_empty()
                && id.len() <= 256,
            "typed factual episode requires a typed session, active learner boundary and identity",
        )?;
        let episode = super::typed::TypedEpisode::from_factual(
            transitions,
            self.manifest().discount,
            revision,
            self.config.storage.max_chunk_bytes,
        )?;
        let mut context = RecordContext::new(&self.manifest().run_id, revision);
        counters.updates = revision - self.manifest().starting_revision;
        context.counters = counters;
        context.trajectory = Some(id.into());
        context.environment_step = Some(counters.env_steps);
        context.wall_elapsed_seconds = Some(self.wall_start.elapsed().as_secs_f64());
        self.recorder.append(
            &super::typed::TypedPayload::Episode(episode)
                .record(context, self.config.storage.max_chunk_bytes)?,
        )?;
        Ok(())
    }

    pub fn finish(&mut self, learner: &L) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        ensure(!self.failed, "cannot finish a failed diagnostic session")?;
        let boundary = self.last.take().unwrap_or_default();
        let result = if self.config.schedule.final_checkpoint {
            self.record(learner, &boundary, true, "final")
        } else {
            Ok(())
        };
        self.last = Some(boundary);
        result?;
        let start = Instant::now();
        self.recorder.finish()?;
        self.stats.total_seconds += start.elapsed().as_secs_f64();
        self.finished = true;
        Ok(())
    }

    fn record(
        &mut self,
        learner: &L,
        boundary: &BoundaryInfo,
        force: bool,
        kind: &str,
    ) -> Result<()> {
        ensure(
            !self.finished && !self.failed,
            "diagnostic session is complete or failed",
        )?;
        let result = self.record_boundary(learner, boundary, force, kind);
        self.failed = result.is_err();
        result
    }

    fn record_boundary(
        &mut self,
        learner: &L,
        boundary: &BoundaryInfo,
        force: bool,
        kind: &str,
    ) -> Result<()> {
        let revision = learner.diagnostic_revision();
        ensure(
            revision >= self.previous.unwrap_or(self.manifest().starting_revision),
            "diagnostic learner revision moved backwards",
        )?;
        let previous = self.previous;
        if previous != Some(revision) {
            self.captured = Channels([false; 7]);
        }
        let due = self.config.schedule.due(previous, revision, force)?;
        let todo = Channels(std::array::from_fn(|i| due.0[i] && !self.captured.0[i]));
        let [
            metrics,
            predictions,
            representations,
            evaluate,
            conditional,
            reference,
            snapshots,
        ] = todo.0;
        self.previous = Some(revision);
        if !todo.any() && !force {
            return Ok(());
        }
        let start = Instant::now();
        let mut context = RecordContext::new(&self.manifest().run_id, revision);
        context.counters = boundary.counters;
        // The recording wire count is run-local even when a restored trainer
        // continues with cumulative factual counters.
        context.counters.updates = revision - self.manifest().starting_revision;
        context.learner_call = boundary.learner_call;
        context.collection_cycle = boundary.collection_cycle;
        context.environment_step = Some(boundary.counters.env_steps);
        context.wall_elapsed_seconds = Some(start.duration_since(self.wall_start).as_secs_f64());
        if snapshots {
            context.identity.checkpoint_id = Some(format!("revision-{revision}"));
        }

        let evaluate_reference = reference
            && self
                .reference
                .as_ref()
                .is_some_and(ReferenceCapture::can_evaluate);
        let mut evaluation_result = None;
        let mut evaluation_config = self.config.evaluation.clone();
        evaluation_config.mode = learner.diagnostic_evaluation_mode(evaluation_config.mode);
        evaluation_config.policy_id = format!("{}@{revision}", context.identity.run_id);
        if (evaluate || evaluate_reference)
            && let Some(run) = &mut self.evaluation
        {
            let evaluation_start = Instant::now();
            let actor = learner.diagnostic_actor()?;
            let result = run(
                actor.as_ref(),
                &evaluation_config,
                self.config.evaluation_horizon,
            )?;
            self.stats.evaluation_seconds += evaluation_start.elapsed().as_secs_f64();
            self.recorder
                .append(&result.empirical_record(context.clone(), None)?)?;
            for index in 0..result.trajectories.len() {
                self.recorder.append(&result.trajectory_record_with_limit(
                    index,
                    context.clone(),
                    &format!("evaluation-{revision}-{index}"),
                    &self.recorder.manifest().environment,
                    self.config.storage.max_chunk_bytes,
                )?)?;
            }
            if self.probes.iter().all(|p| p.id() == "evaluation-visited")
                && self.config.max_probe_sets > usize::from(self.config.visited_probe_rows > 0)
                && (self.config.schedule.probes.is_some()
                    || self.config.schedule.representations.is_some())
                && let Some(first) = result.trajectories.first()
                && !first.actions.is_empty()
            {
                let end = first.actions.len().min(self.config.max_probe_rows);
                self.probes.push(ProbeSet::from_observations(
                    "evaluation-initial",
                    ProbePopulation::FixedReference,
                    "Fixed input rows from the first retained diagnostic episode",
                    first.observations[..end].iter(),
                    self.recorder.manifest().observation_space(),
                    self.recorder.manifest().action_space.clone(),
                    self.config.storage.max_chunk_bytes,
                )?);
            }
            if self.config.visited_probe_rows > 0
                && let Some(first) = result.trajectories.first()
                && !first.actions.is_empty()
            {
                use rand::SeedableRng;
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(self.config.seed ^ revision);
                let count = self.config.visited_probe_rows.min(first.actions.len());
                self.probes.retain(|p| p.id() != "evaluation-visited");
                self.probes.push(ProbeSet::visited_in(
                    "evaluation-visited",
                    &format!(
                        "Independent evaluation policy visit population at revision {revision}"
                    ),
                    &first.observations[..first.actions.len()],
                    count,
                    (
                        self.recorder.manifest().observation_space(),
                        self.recorder.manifest().action_space.clone(),
                    ),
                    &mut rng,
                    self.config.storage.max_chunk_bytes,
                )?);
            }
            evaluation_result = Some(result);
        }

        let capture_reference = reference && self.reference.is_some();
        let capture_conditional = conditional && !self.conditional.is_empty();
        if predictions || representations || capture_reference || capture_conditional {
            for probes in &self.probes {
                self.recorder.append(
                    &probes
                        .record_with_limit(context.clone(), self.config.storage.max_chunk_bytes)?,
                )?;
            }
        }
        let need_comparison_probes = capture_reference && !self.probes.is_empty();
        let mut learner_chunks = Vec::new();
        if predictions || representations || snapshots || need_comparison_probes {
            let capture_start = Instant::now();
            learner_chunks = learner.diagnostic_records(&CaptureRequest {
                context: &context,
                probes: &self.probes,
                predictions: predictions || need_comparison_probes,
                representations,
                snapshots,
                max_activation_values: self.config.max_activation_values,
                max_chunk_bytes: self.config.storage.max_chunk_bytes,
                conventions: self.conventions.as_ref(),
                evaluation_mode: evaluation_config.mode,
                policy_seed: self.config.seed ^ revision,
            })?;
            self.stats.inference_seconds += capture_start.elapsed().as_secs_f64();
            for chunk in &learner_chunks {
                self.recorder.append(chunk)?;
            }
        }
        if let Some(reference) = self.reference.as_mut().filter(|_| capture_reference) {
            let reference_start = Instant::now();
            let actor = learner.diagnostic_actor()?;
            reference.record(
                learner,
                actor.as_ref(),
                &context,
                &self.probes,
                &learner_chunks,
                evaluation_result.as_ref(),
                &evaluation_config,
                self.config.evaluation_horizon,
                self.evaluation.as_mut(),
                self.config.reference_action_samples,
                &mut self.recorder,
            )?;
            self.stats.reference_seconds += reference_start.elapsed().as_secs_f64();
        }
        if capture_conditional {
            let conditional_start = Instant::now();
            let mut config = self.config.conditional_evaluation.clone();
            let actor = if let Some((actor, continuation)) =
                learner.diagnostic_conditional_actor(&context)?
            {
                config.mode = if continuation.stochastic {
                    ActMode::Sample
                } else {
                    ActMode::Deterministic
                };
                config.policy_id = continuation.identity;
                actor
            } else {
                config.mode = learner.diagnostic_evaluation_mode(config.mode);
                config.policy_id = evaluation_config.policy_id;
                learner.diagnostic_actor()?
            };
            for case in &self.conditional {
                let result = (case.evaluate)(actor.as_ref(), &config)?;
                let mut context = context.clone();
                context.probe_set = Some(case.probe_set.clone());
                let mut chunk = result.empirical_record(context, Some(case.row as u64))?;
                if let Some(scientific_context) = &self.scientific_context
                    && result.metadata.complete_returns
                    && result.metadata.independent_future_streams
                    && result.returns.len() <= empirical::MAX_EMPIRICAL_SAMPLES
                {
                    let identity = &chunk.context.identity;
                    let declaration = EmpiricalDeclaration {
                        version: EMPIRICAL_DECLARATION_VERSION,
                        run_id: identity.run_id.clone(),
                        revision: identity.learner_update_revision,
                        model_role: identity.model_role.clone(),
                        checkpoint_id: identity.checkpoint_id.clone(),
                        context: scientific_context.clone(),
                        conditioning: result.metadata.conditioning.clone(),
                    };
                    empirical::attach_declaration(&mut chunk, &declaration)?;
                }
                self.recorder.append(&chunk)?;
            }
            self.stats.conditional_seconds += conditional_start.elapsed().as_secs_f64();
        }
        if metrics {
            if let Some(chunk) = learner.diagnostic_metrics(&context)? {
                self.recorder.append(&chunk)?;
            }
            append_metrics(
                &mut self.recorder,
                &context,
                boundary.borrowed(),
                self.stats.total_seconds + start.elapsed().as_secs_f64(),
            )?;
        }
        if force {
            self.recorder.append(&Chunk {
                context,
                record: Record::Event(EventRecord {
                    category: "diagnostic-boundary".into(),
                    message: kind.into(),
                    severity: "info".into(),
                }),
                arrays: BTreeMap::new(),
            })?;
        }
        if evaluate || snapshots || capture_reference || capture_conditional {
            self.recorder.flush()?;
        }
        for (captured, due) in self.captured.0.iter_mut().zip(todo.0) {
            *captured |= due;
        }
        self.stats.total_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }
}

impl<L: DiagnosticLearner + ?Sized> UpdateObserver<L> for DiagnosticSession<'_, L> {
    fn after_update(&mut self, learner: &L, boundary: UpdateBoundary<'_>) -> Result<()> {
        let revision = learner.diagnostic_revision();
        let last_revision = self.previous.unwrap_or(self.manifest().starting_revision);
        ensure(
            revision.checked_sub(last_revision) == Some(boundary.report.updates),
            "observed learner update count differs from its completed revision",
        )?;
        let boundary = BoundaryInfo {
            counters: *boundary.counters,
            learner_call: boundary.learner_call,
            collection_cycle: boundary.collection_cycle,
            report: boundary.report.clone(),
            collection_seconds: boundary.collection_seconds,
            learning_seconds: boundary.learning_seconds,
        };
        self.record(learner, &boundary, false, "scheduled")?;
        self.last = Some(boundary);
        Ok(())
    }
}
