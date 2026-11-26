//! Native reinforcement-learning laboratory over portable experiment recordings.
use crate::{
    data::*,
    plot_data::*,
    resolver::{ProbeResolver, ResolveStatus, ResolvedRecord},
};
use eframe::egui::{self, Color32};
use rl::{
    Result,
    diagnostics::{schema::*, storage::ReadLimits},
};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

#[path = "ui/directory_browser.rs"]
mod directory_browser;
#[path = "ui/dynamics_workspace.rs"]
mod dynamics_workspace;
#[path = "ui/environment_workspace.rs"]
mod environment_workspace;
#[path = "ui/learning_evolution.rs"]
mod learning_evolution;
#[path = "ui/learning_workspace.rs"]
mod learning_workspace;
#[path = "ui/network_workspace.rs"]
mod network_workspace;
#[cfg(test)]
#[path = "ui/network_workspace_tests.rs"]
mod network_workspace_tests;
#[cfg(test)]
#[path = "ui/policy_demo.rs"]
mod policy_demo;
#[path = "ui/policy_evolution.rs"]
mod policy_evolution;
#[path = "ui/policy_modes.rs"]
mod policy_modes;
#[path = "ui/policy_workspace.rs"]
mod policy_workspace;
#[path = "ui/shell.rs"]
mod shell;
pub use policy_modes::PolicyMode;
#[cfg(test)]
#[path = "ui/policy_modes_tests.rs"]
mod policy_modes_tests;
#[cfg(test)]
#[path = "ui/policy_workspace_tests.rs"]
mod policy_workspace_tests;
#[path = "ui/recent.rs"]
mod recent;
#[path = "ui/recording_opener.rs"]
mod recording_opener;
#[path = "ui/reference_measurements.rs"]
mod reference_measurements;
#[cfg(test)]
#[path = "ui/resource_lifetime_tests.rs"]
mod resource_lifetime_tests;
#[cfg(test)]
#[path = "ui/shell_tests.rs"]
mod shell_tests;
#[path = "ui/training_workspace.rs"]
mod training_workspace;
#[path = "ui/typed_workspace.rs"]
mod typed_workspace;
#[path = "ui/value_workspace.rs"]
mod value_workspace;

const ACCENT: Color32 = crate::theme::CURRENT;
const WARM: Color32 = crate::theme::WARNING;

fn has_record(run: &RunIndex, role: &str, kind: RecordKind) -> bool {
    run.candidates(kind, role, None, u64::MAX)
        .take(256)
        .any(|index| run.recording().is_retained(index))
}

fn initial_view(run: &RunIndex, selection: &Selection) -> View {
    let role = &selection.model_role;
    if run
        .candidates(RecordKind::Empirical, role, None, u64::MAX)
        .take(512)
        .any(|i| run.recording().entries()[i].probe_set.is_none() && run.recording().is_retained(i))
        || has_record(run, role, RecordKind::Metrics)
    {
        View::Training
    } else if has_record(run, role, RecordKind::Policy)
        || has_record(run, role, RecordKind::FinitePredictions)
    {
        View::PolicyMap
    } else if has_record(run, role, RecordKind::Critic) {
        View::Value
    } else if has_record(run, role, RecordKind::Trajectory)
        || has_record(run, role, RecordKind::TypedEpisode)
    {
        View::Dynamics
    } else if has_record(run, role, RecordKind::Activations) {
        View::Network
    } else {
        View::PolicyMap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    Training,
    PolicyMap,
    Value,
    Network,
    Dynamics,
    Environment,
}
impl View {
    pub fn from_startup_hook(value: &str) -> Self {
        match value {
            "training" => Self::Training,
            "value" => Self::Value,
            "network" => Self::Network,
            "dynamics" => Self::Dynamics,
            "environment" => Self::Environment,
            _ => Self::PolicyMap,
        }
    }
}
#[cfg(feature = "screenshots")]
enum CapturePlayback {
    Disabled,
    AwaitingFrame,
    QueuePlay(u64),
    PlayingFrom(u64),
}

pub struct Laboratory {
    path: String,
    open_dialog: bool,
    open_path: String,
    browser_path: String,
    browser_folders: Vec<std::path::PathBuf>,
    browser_notice: Option<String>,
    browser_worker: Option<directory_browser::DirectoryBrowser>,
    recording_opener: Option<recording_opener::RecordingOpener>,
    recent: recent::RecentRecordings,
    demonstration: Option<std::path::PathBuf>,
    catalogue: Vec<CatalogEntry>,
    run: Option<RunIndex>,
    loader: Loader,
    resolver: ProbeResolver,
    transport: crate::timeline_ui::TrainingTransport,
    snapshot_catalog: Option<crate::snapshot_catalog::SnapshotCatalog>,
    selection: Selection,
    view: View,
    survey_snapshot: bool,
    error: Option<String>,
    last_refresh: Instant,
    live: bool,
    trajectory: Option<u64>,
    comparison: crate::comparison::Comparison,
    interactive: crate::interactive::Interactive,
    policy_pane: crate::policy::view::PolicyPane,
    policy_execution: policy_modes::PolicyExecution,
    policy_evolution: bool,
    // Survives per-recording execution resets; replacement workers are admitted
    // only after this owner acknowledges retirement and releases all leases.
    prepared_retiring: Option<crate::policy::cache::PreparedProvider>,
    value_view: crate::value::view::ValueView,
    network_pane: network_workspace::NetworkPane,
    learning_pane: learning_evolution::LearningPane,
    training_pane: training_workspace::TrainingPane,
    dynamics_pane: dynamics_workspace::DynamicsPane,
    environment_pane: environment_workspace::EnvironmentPane,
    typed_pane: typed_workspace::TypedPane,
    workspace_defaults: Option<(crate::presentation::WorkspaceDefaults, SelectedInput, u64)>,
    reduced_motion: bool,
    #[cfg(feature = "screenshots")]
    capture: Option<std::path::PathBuf>,
    #[cfg(feature = "screenshots")]
    capture_playback: CapturePlayback,
    #[cfg(feature = "screenshots")]
    timing: crate::native_timing::NativeTiming,
    #[cfg(feature = "screenshots")]
    timing_pass: u64,
    #[cfg(feature = "screenshots")]
    timing_readiness: Option<serde_json::Value>,
}
impl Laboratory {
    pub fn new(cc: &eframe::CreationContext<'_>, path: String) -> Result<Self> {
        let mut app = Self::from_context(&cc.egui_ctx, path)?;
        app.recent = recent::RecentRecordings::load();
        app.demonstration = std::env::current_exe()
            .ok()
            .and_then(|executable| {
                executable
                    .parent()
                    .map(|parent| parent.join("demonstrations"))
            })
            .filter(|path| path.is_dir());
        if app.run.is_some() {
            app.recent.remember(Path::new(&app.path));
        }
        Ok(app)
    }
    /// Headless construction also exercises current panel state without a display server.
    pub fn from_context(ctx: &egui::Context, path: String) -> Result<Self> {
        crate::theme::install(ctx);
        #[cfg(feature = "screenshots")]
        if let Ok(value) = std::env::var("RL_VIEWER_ZOOM") {
            let zoom = value
                .parse::<f32>()
                .ok()
                .filter(|v| v.is_finite() && (0.75..=2.).contains(v))
                .ok_or_else(|| {
                    rl::Error::Config("RL_VIEWER_ZOOM must be between0.75 and2".into())
                })?;
            ctx.set_zoom_factor(zoom);
        }
        #[cfg(feature = "screenshots")]
        ctx.data_mut(|data| {
            data.insert_temp(
                egui::Id::unique("native-inspection-default"),
                std::env::var("RL_VIEWER_INSPECT").as_deref() == Ok("true"),
            );
            data.insert_temp(
                egui::Id::unique("native-hide-pointer"),
                std::env::var("RL_VIEWER_POINTER").as_deref() == Ok("false"),
            );
        });
        let context = ctx.clone();
        let inference_context = ctx.clone();
        let mut app = Self {
            open_dialog: false,
            open_path: path.clone(),
            browser_path: String::new(),
            browser_folders: Vec::new(),
            browser_notice: None,
            browser_worker: None,
            recording_opener: None,
            recent: Default::default(),
            demonstration: None,
            path,
            catalogue: vec![],
            run: None,
            loader: Loader::new(crate::budgets::RAW, move || context.request_repaint())?,
            resolver: ProbeResolver::new(crate::budgets::MAIN_RESOLVER),
            transport: Default::default(),
            snapshot_catalog: None,
            selection: Selection::default(),
            view: View::PolicyMap,
            survey_snapshot: false,
            error: None,
            last_refresh: Instant::now(),
            live: true,
            trajectory: None,
            comparison: crate::comparison::Comparison::new(),
            interactive: crate::interactive::Interactive::new(move || {
                inference_context.request_repaint()
            })?,
            policy_pane: crate::policy::view::PolicyPane::new(),
            policy_execution: Default::default(),
            policy_evolution: false,
            prepared_retiring: None,
            value_view: crate::value::view::ValueView::new(),
            network_pane: network_workspace::NetworkPane::new(ctx.clone()),
            learning_pane: learning_evolution::LearningPane::new(ctx.clone())?,
            training_pane: training_workspace::TrainingPane::default(),
            dynamics_pane: dynamics_workspace::DynamicsPane::new(ctx.clone())?,
            environment_pane: environment_workspace::EnvironmentPane::new(ctx.clone())?,
            typed_pane: Default::default(),
            workspace_defaults: None,
            reduced_motion: false,
            #[cfg(feature = "screenshots")]
            capture: std::env::var_os("RL_VIEWER_CAPTURE").map(Into::into),
            #[cfg(feature = "screenshots")]
            capture_playback: if std::env::var("RL_VIEWER_CAPTURE_PLAYBACK").as_deref()
                == Ok("true")
            {
                CapturePlayback::AwaitingFrame
            } else {
                CapturePlayback::Disabled
            },
            #[cfg(feature = "screenshots")]
            timing: crate::native_timing::NativeTiming::from_env()?,
            #[cfg(feature = "screenshots")]
            timing_pass: 0,
            #[cfg(feature = "screenshots")]
            timing_readiness: None,
        };
        #[cfg(feature = "screenshots")]
        {
            app.loader.timing = app.timing.clone();
        }
        if Path::new(&app.path).exists() {
            app.open_catalogue();
        }
        if let Ok(view) = std::env::var("RL_VIEWER_VIEW") {
            app.view = View::from_startup_hook(&view);
            app.survey_snapshot = false;
        }
        #[cfg(feature = "screenshots")]
        if app.view == View::PolicyMap {
            app.policy_evolution =
                std::env::var("RL_VIEWER_POLICY_EVOLUTION").as_deref() == Ok("true");
        }
        #[cfg(feature = "screenshots")]
        if let Ok(value) = std::env::var("RL_VIEWER_LEARNING") {
            app.learning_pane.set_expanded(match value.as_str() {
                "true" => true,
                "false" => false,
                _ => {
                    return Err(rl::Error::Config(
                        "RL_VIEWER_LEARNING must be true or false".into(),
                    ));
                }
            });
        }
        if let Ok(set) = std::env::var("RL_VIEWER_PROBE")
            && let Some(run) = app.run.clone()
            && run.probe_set_ids().any(|id| id == set)
        {
            app.select_input_population(&run, &set, 0);
        }
        #[cfg(feature = "screenshots")]
        if let Ok(row) = std::env::var("RL_VIEWER_ROW") {
            let row = row
                .parse::<usize>()
                .map_err(|_| rl::Error::Config("invalid capture probe row".into()))?;
            if let SelectedInput::ProbeRow { row: selected, .. } = &mut app.selection.input {
                *selected = row;
            }
        }
        #[cfg(feature = "screenshots")]
        if let Some(sequence) = std::env::var_os("RL_VIEWER_TRAJECTORY_SEQUENCE") {
            if std::env::var_os("RL_VIEWER_PROBE").is_some()
                || std::env::var_os("RL_VIEWER_ROW").is_some()
            {
                return Err(rl::Error::Config(
                    "capture trajectory and probe selectors conflict".into(),
                ));
            }
            let sequence = sequence
                .to_str()
                .ok_or_else(|| rl::Error::Config("invalid capture trajectory sequence".into()))?;
            let step = std::env::var_os("RL_VIEWER_TRAJECTORY_STEP")
                .map(|s| {
                    s.into_string()
                        .map_err(|_| rl::Error::Config("invalid capture trajectory step".into()))
                })
                .transpose()?
                .unwrap_or_else(|| "0".into());
            app.select_capture_trajectory(sequence, &step)?;
        } else if std::env::var_os("RL_VIEWER_TRAJECTORY_STEP").is_some() {
            return Err(rl::Error::Config(
                "capture trajectory step requires sequence".into(),
            ));
        }
        #[cfg(feature = "screenshots")]
        if let Ok(value) = std::env::var("RL_VIEWER_REVISION")
            && let Some(run) = &app.run
        {
            let revision = value
                .parse()
                .map_err(|_| rl::Error::Config("invalid capture revision".into()))?;
            app.selection.select_revision(run, revision)?;
        }
        #[cfg(feature = "screenshots")]
        {
            if let Ok(role) = std::env::var("RL_VIEWER_REFERENCE_ROLE") {
                app.comparison
                    .configure_policy_reference(None, None, false, Some(&role))?;
                app.workspace_defaults = None;
            }
            if app.view == View::Training {
                app.configure_training_capture()?;
            }
        }
        // Startup time/input overrides are now final. Trajectory selection has
        // already disabled discovery; explicit input choices keep ownership.
        if matches!(app.view, View::Dynamics | View::Environment) {
            app.workspace_defaults = None;
        }
        if app.workspace_defaults.is_some()
            && let Some(run) = &app.run
        {
            let mut defaults = crate::presentation::WorkspaceDefaults::new(run, &app.selection);
            if std::env::var_os("RL_VIEWER_PROBE").is_some()
                || cfg!(feature = "screenshots") && std::env::var_os("RL_VIEWER_ROW").is_some()
            {
                defaults.restrict_input(run, &app.selection.input);
            }
            app.workspace_defaults = Some((
                defaults,
                app.selection.input.clone(),
                app.comparison.reference_interaction_serial(),
            ));
        }
        Ok(app)
    }
    pub fn select_view(&mut self, view: View) {
        self.survey_snapshot = false;
        if self.view != view {
            let policy_training = !self.policy_evolution
                && self.policy_execution.mode != Some(PolicyMode::Prepared)
                && matches!(
                    (self.view, view),
                    (View::PolicyMap, View::Training) | (View::Training, View::PolicyMap)
                );
            let value_training = !self.value_view.decision_loss_requested()
                && matches!(
                    (self.view, view),
                    (View::Value, View::Training) | (View::Training, View::Value)
                );
            let policy_value = !self.policy_evolution
                && self.policy_execution.mode != Some(PolicyMode::Prepared)
                && !self.value_view.decision_loss_requested()
                && matches!(
                    (self.view, view),
                    (View::PolicyMap, View::Value) | (View::Value, View::PolicyMap)
                );
            self.typed_pane.switch_view();
            self.leave_prepared_policy();
            let parked = policy_value
                && self
                    .snapshot_catalog
                    .as_mut()
                    .is_some_and(|c| c.park_for_retarget());
            if !parked
                && (!(policy_training || value_training)
                    || self.snapshot_catalog.as_ref().is_some_and(|c| c.pending()))
            {
                self.snapshot_catalog = None;
            }
            self.transport.clear_snapshot_state();
            if parked && self.view == View::PolicyMap && view == View::Value {
                self.interactive.suspend_policy_for_value();
            } else if parked && self.view == View::Value && view == View::PolicyMap {
                self.interactive.leave_value_for_policy();
            } else if policy_training {
                self.interactive.suspend_policy();
            } else if value_training {
                self.interactive.suspend_value();
            } else {
                self.interactive.retire_session();
            }
            self.policy_pane.clear();
            if value_training {
                self.value_view.dirty = true;
                self.value_view.ready = false;
            } else {
                self.value_view.clear();
            }
            self.dynamics_pane.clear();
            self.environment_pane.leave();
            if self.view == View::Training || self.view == View::PolicyMap && self.policy_evolution
            {
                self.learning_pane.clear();
            }
            if self.view == View::Network {
                self.network_pane.retire_representations();
            }
        }
        if matches!(view, View::Dynamics | View::Environment) {
            self.workspace_defaults = None;
        }
        self.view = view;
    }
    /// Workspaces share an immutable recorded input and requested learner time.
    pub fn selection(&self) -> &Selection {
        &self.selection
    }
    /// Read-only instrumentation for bounded native/real-data measurements.
    /// Publication readiness is separate: discovery and one model warmup can
    /// continue while the current complete frame is already visible.
    #[doc(hidden)]
    pub fn snapshot_preparation_idle(&self) -> bool {
        self.loader.pending() == 0
            && !self.interactive.pending()
            && self.snapshot_catalog.as_ref().is_none_or(|c| !c.pending())
    }
    /// Scripted native walkthrough entry point; validation runs outside painting.
    /// The ordinary UI continues to publish the same deferred Input intent.
    #[cfg(feature = "screenshots")]
    pub fn select_input(&mut self, input: SelectedInput) -> Result<()> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| rl::Error::State("no recording is open".into()))?;
        if let SelectedInput::ProbeRow { sequence, .. }
        | SelectedInput::TrajectoryStep { sequence, .. } = &input
        {
            let index = run
                .by_sequence(*sequence)
                .ok_or_else(|| rl::Error::State("selected input source missing".into()))?;
            let chunk = run
                .recording()
                .source(index)?
                .load_if_within_charge(crate::budgets::POLICY_SCRATCH)?
                .ok_or_else(|| {
                    rl::Error::Config(
                        "scripted selected input exceeds bounded source admission".into(),
                    )
                })?;
            let valid = match (&input, &chunk.record) {
                (SelectedInput::ProbeRow { set, row, .. }, Record::ProbeSet(p)) => {
                    set == &p.id && *row < chunk.arrays[&p.row_ids].len()
                }
                (SelectedInput::TrajectoryStep { step, .. }, Record::Trajectory(t)) => {
                    *step <= chunk.arrays[&t.rewards].len()
                }
                (SelectedInput::ProbeRow { set, row, .. }, Record::Typed(_)) => {
                    chunk.context.probe_set.as_deref() == Some(set)
                        && matches!(rl::diagnostics::typed::TypedPayload::read(&chunk)?,
                            rl::diagnostics::typed::TypedPayload::Probes(p) if *row < p.inputs.len())
                }
                (SelectedInput::TrajectoryStep { step, .. }, Record::Typed(_)) => {
                    matches!(rl::diagnostics::typed::TypedPayload::read(&chunk)?,
                        rl::diagnostics::typed::TypedPayload::Episode(e) if *step <= e.steps.len())
                }
                _ => false,
            };
            if !valid {
                return Err(rl::Error::State(
                    "selected source kind, population or row differs".into(),
                ));
            }
        }
        crate::pane::SelectionIntent::Input(input).apply(run, &mut self.selection)?;
        self.workspace_defaults = None;
        Ok(())
    }
    pub fn set_learning_expanded(&mut self, expanded: bool) {
        self.learning_pane.set_expanded(expanded);
    }
    pub fn learning_frame(&self) -> Option<&Arc<crate::learning::LearningFrame>> {
        self.learning_pane.frame()
    }
    pub fn learning_metrics(&self) -> crate::learning::LearningMetrics {
        self.learning_pane.metrics()
    }
    #[cfg(feature = "screenshots")]
    pub fn select_capture_trajectory(&mut self, sequence: &str, step: &str) -> Result<()> {
        let sequence = sequence
            .parse::<u64>()
            .map_err(|_| rl::Error::Config("invalid capture trajectory sequence".into()))?;
        let step = step
            .parse::<usize>()
            .map_err(|_| rl::Error::Config("invalid capture trajectory step".into()))?;
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| rl::Error::State("no capture recording".into()))?;
        let index = run
            .by_sequence(sequence)
            .ok_or_else(|| rl::Error::State("capture trajectory sequence missing".into()))?;
        if !matches!(
            run.recording().entries()[index].kind,
            RecordKind::Trajectory | RecordKind::TypedEpisode
        ) {
            return Err(rl::Error::State(
                "capture source is not a trajectory".into(),
            ));
        }
        let chunk = run.recording().load(index)?;
        let steps = match &chunk.record {
            Record::Trajectory(t) => crate::plot_data::array(&chunk, &t.rewards)?.len(),
            Record::Typed(_) => {
                let rl::diagnostics::typed::TypedPayload::Episode(episode) =
                    rl::diagnostics::typed::TypedPayload::read(&chunk)?
                else {
                    unreachable!()
                };
                episode.steps.len()
            }
            _ => unreachable!(),
        };
        if step > steps {
            return Err(rl::Error::State(
                "capture trajectory step outside recording".into(),
            ));
        }
        self.selection.input = SelectedInput::TrajectoryStep { sequence, step };
        self.selection.model_role = chunk.context.identity.model_role.clone();
        self.workspace_defaults = None;
        Ok(())
    }
    pub fn select_revision(&mut self, revision: u64) -> Result<()> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| rl::Error::State("no recording is open".into()))?;
        self.selection.select_revision(run, revision)?;
        Ok(())
    }
    pub fn training_is_playing(&self) -> bool {
        self.transport.is_playing()
    }
    pub fn notice(&self) -> Option<&str> {
        self.error.as_deref()
    }
    fn open_catalogue(&mut self) -> bool {
        let path = self.path.clone();
        self.open_directory(Path::new(&path))
    }
    fn open_directory(&mut self, path: &Path) -> bool {
        self.finish_open(recording_opener::prepare(path))
    }
    fn finish_open(
        &mut self,
        result: std::result::Result<recording_opener::OpenedRecording, String>,
    ) -> bool {
        match result {
            Ok(opened) => {
                self.catalogue = opened.entries;
                self.path = opened.path.to_string_lossy().into_owned();
                self.recent.remember(&opened.path);
                self.error = (!opened.warnings.is_empty()).then(|| opened.warnings.join("\n"));
                self.install_run(opened.run);
                true
            }
            Err(error) => {
                self.error = Some(error);
                false
            }
        }
    }
    fn request_directory(&mut self, path: &Path, ctx: &egui::Context) {
        if self.recording_opener.is_none() {
            match recording_opener::RecordingOpener::new(ctx.clone()) {
                Ok(worker) => self.recording_opener = Some(worker),
                Err(error) => {
                    self.error = Some(error.to_string());
                    return;
                }
            }
        }
        if let Some(worker) = &mut self.recording_opener {
            worker.request(path.to_owned());
            ctx.request_repaint();
        }
    }
    fn open_run(&mut self, i: usize) {
        if let Some(worker) = &mut self.recording_opener {
            worker.cancel();
        }
        #[cfg(feature = "screenshots")]
        let _span = self.timing.span("open_run", i as u64);
        match RunIndex::open(&self.catalogue[i].root, ReadLimits::default()) {
            Ok(run) => self.install_run(run),
            Err(e) => self.error = Some(e.to_string()),
        }
    }
    fn install_run(&mut self, run: RunIndex) {
        self.leave_prepared_policy();
        self.policy_execution = Default::default();
        self.policy_evolution = false;
        self.snapshot_catalog = None;
        self.transport.clear_snapshot_state();
        self.interactive.retire_session();
        self.policy_pane.clear();
        self.value_view.clear();
        self.network_pane.clear();
        self.learning_pane.clear();
        self.dynamics_pane.clear();
        self.environment_pane.clear();
        self.typed_pane.leave();
        let opening_new_run = self.run.as_ref().is_none_or(|old| old.root() != run.root());
        self.selection = crate::presentation::initial_selection(&run);
        if opening_new_run {
            self.view = initial_view(&run, &self.selection);
        }
        self.survey_snapshot = opening_new_run
            && !typed_workspace::typed_run(&run)
            && self.view == View::PolicyMap
            && !has_record(&run, &self.selection.model_role, RecordKind::Policy)
            && !run
                .snapshot_revisions(&self.selection.model_role)
                .is_empty();
        self.training_pane.reset_for_run();
        let comparison = self.comparison.reset_reference_to(&run).and_then(|()| {
            if self.view == View::Training
                && let Some(&revision) = run.revisions().first()
            {
                self.comparison.set_reference_revision(revision)?;
            }
            Ok(())
        });
        if let Err(error) = comparison {
            self.error = Some(error.to_string());
        }
        self.workspace_defaults = Some((
            crate::presentation::WorkspaceDefaults::new(&run, &self.selection),
            self.selection.input.clone(),
            self.comparison.reference_interaction_serial(),
        ));
        self.trajectory = run
            .recording()
            .entries()
            .iter()
            .rev()
            .find(|e| {
                matches!(e.kind, RecordKind::Trajectory | RecordKind::TypedEpisode)
                    && e.model_role == "online"
            })
            .map(|e| e.sequence);
        self.run = Some(run);
        self.loader.invalidate();
        self.resolver.clear();
        self.transport.reset();
    }
    pub fn clear_recording(&mut self) {
        if let Some(worker) = &mut self.recording_opener {
            worker.cancel();
        }
        self.leave_prepared_policy();
        self.policy_execution = Default::default();
        self.select_view(View::PolicyMap);
        self.snapshot_catalog = None;
        self.transport.clear_snapshot_state();
        self.interactive.retire_session();
        self.policy_pane.clear();
        self.value_view.clear();
        self.network_pane.clear();
        self.learning_pane.clear();
        self.dynamics_pane.clear();
        self.environment_pane.clear();
        self.typed_pane.leave();
        self.run = None;
        self.loader.invalidate();
        self.resolver.clear();
        self.comparison.clear_recording();
        self.workspace_defaults = None;
        self.transport.reset();
    }
    fn selected_source(&mut self, run: &RunIndex) -> Result<Option<(Arc<Chunk>, usize)>> {
        let (sequence, row) = match self.selection.input {
            SelectedInput::ProbeRow { sequence, row, .. } => (sequence, row),
            SelectedInput::TrajectoryStep { sequence, step } => (sequence, step),
            SelectedInput::None => return Ok(None),
        };
        let Some(i) = run.by_sequence(sequence) else {
            return Ok(None);
        };
        Ok(self.loader.request(run, i)?.map(|p| (p, row)))
    }
    fn survey_model_route(&mut self, run: &RunIndex) {
        if !self.survey_snapshot {
            return;
        }
        let Some(index) = run
            .candidates(
                RecordKind::ModelSnapshot,
                &self.selection.model_role,
                None,
                self.selection.revision,
            )
            .next()
        else {
            self.survey_snapshot = false;
            return;
        };
        let chunk = match self.loader.request_admitted(run, index, 2 * 1024 * 1024) {
            Ok(None) => return,
            Ok(Some(chunk)) => chunk,
            Err(_) => {
                self.survey_snapshot = false;
                return;
            }
        };
        self.survey_snapshot = false;
        if let Ok(configuration) = crate::value::prepared::ValueSource::configuration(&chunk) {
            use rl::diagnostics::models::ModelKind;
            let mut critic = false;
            for model in configuration.models.values() {
                if model.kind.is_stochastic_policy()
                    || matches!(
                        model.kind,
                        ModelKind::DeterministicPolicy | ModelKind::DiscreteQ
                    )
                {
                    return;
                }
                critic |= !matches!(model.kind, ModelKind::Opaque { .. });
            }
            if critic {
                self.select_view(View::Value);
            }
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let previous_view = self.view;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.view, View::Training, "Training");
            ui.selectable_value(&mut self.view, View::PolicyMap, "Policy");
            ui.selectable_value(&mut self.view, View::Value, "Value");
            ui.add_enabled_ui(
                !typed_workspace::typed_run(run)
                    || has_record(run, &self.selection.model_role, RecordKind::Activations),
                |ui| {
                    ui.selectable_value(&mut self.view, View::Network, "Network");
                },
            );
            ui.selectable_value(&mut self.view, View::Dynamics, "Dynamics");
            ui.add_enabled_ui(
                has_record(run, &self.selection.model_role, RecordKind::Trajectory)
                    || has_record(run, &self.selection.model_role, RecordKind::TypedEpisode),
                |ui| {
                    ui.selectable_value(&mut self.view, View::Environment, "Environment");
                },
            );
        });
        if self.view != previous_view {
            let next = self.view;
            self.view = previous_view;
            self.select_view(next);
            // Retire the old panel before admitting the next panel's transport
            // owners. Start its catalog on the following paint.
            ui.ctx().request_repaint();
            return Ok(());
        }
        Ok(())
    }
    fn input_controls(&mut self, ui: &mut egui::Ui, run: &RunIndex) {
        ui.horizontal_wrapped(|ui| {
            let retained =
                if run.has_model_for(self.selection.revision, &self.selection.model_role) {
                    "Model snapshot retained"
                } else {
                    "No weights at this timestamp"
                };
            ui.weak("Model").on_hover_text(retained);
            if retained != "Model snapshot retained" {
                ui.weak("Recorded inputs").on_hover_text(retained);
            }
            egui::ComboBox::from_id_salt("model-role").width(88.).selected_text(&self.selection.model_role).show_ui(ui, |ui| {
                let roles:std::collections::BTreeSet<_>=run.recording().entries().iter()
                    .map(|e|e.model_role.as_str())
                    .filter(|role| !crate::reference::provider_role(role))
                    .collect();
                for role in roles { if ui.selectable_label(self.selection.model_role == role, role).clicked() { if let Err(error) = crate::pane::SelectionIntent::ModelRole(role.into()).apply(run, &mut self.selection) { self.error = Some(error.to_string()); } self.workspace_defaults = None; } }
            });
            let label = match &self.selection.input {
                SelectedInput::ProbeRow { set, .. } => set.clone(),
                SelectedInput::TrajectoryStep { step, .. } => format!("Episode · step {step}"),
                SelectedInput::None => "Select an input".into(),
            };
            ui.weak("Input");
            egui::ComboBox::from_id_salt("input").selected_text(label).show_ui(ui, |ui| {
                for id in run.probe_set_ids() {
                    if ui.selectable_label(matches!(&self.selection.input, SelectedInput::ProbeRow {set,..} if set==id), id).clicked() {
                        self.select_input_population(run, id, 0);
                        self.workspace_defaults = None;
                    }
                }
                if let Some(sequence) = self.trajectory && ui.button("Use selected trajectory state").clicked() {
                    self.selection.input = SelectedInput::TrajectoryStep { sequence, step: 0 };
                    self.workspace_defaults = None;
                }
            });
            if let SelectedInput::ProbeRow {row,..} = &mut self.selection.input
                && ui.add(egui::DragValue::new(row).range(0..=usize::MAX).prefix("state ")).changed() { self.workspace_defaults = None; }
        });
    }
    pub fn show(&mut self, ui: &mut egui::Ui) {
        if let Some(worker) = &mut self.recording_opener
            && let Some(result) = worker.poll(ui.ctx())
        {
            self.finish_open(result);
        }
        #[cfg(feature = "screenshots")]
        if ui
            .ctx()
            .data(|data| data.get_temp::<bool>(egui::Id::unique("native-hide-pointer")))
            .unwrap_or(false)
        {
            ui.ctx()
                .input_mut(|input| input.pointer = Default::default());
        }
        #[cfg(feature = "screenshots")]
        let _ui_span = {
            self.timing_pass += 1;
            self.timing.event("ui_entry", self.timing_pass, || {
                let egui_pass = ui.ctx().cumulative_pass_nr();
                ui.input(|input| serde_json::json!({"egui_pass":egui_pass,"visible":input.viewport().visible(),"minimized":input.viewport().minimized,"focused":input.viewport().focused,"revision":self.selection.revision,"displayed":self.transport.capture_publication()}))
            });
            self.timing.span("ui_pass", self.timing_pass)
        };
        self.loader.begin_pass();
        self.loader.poll();
        self.reap_prepared_retirement();
        self.interactive.housekeeping();
        if self.live && self.last_refresh.elapsed() > Duration::from_secs(1) {
            if let Some(run) = &mut self.run
                && let Err(e) = run.refresh()
            {
                self.error = Some(e.to_string());
            }
            if self.view == View::Network
                && let Some(run) = &self.run
                && let Some(catalog) = &mut self.snapshot_catalog
            {
                catalog.recheck_availability(run);
            }
            if self.view == View::Environment
                && let Some(run) = &self.run
            {
                self.environment_pane.recheck_source(run);
            }
            if self.view == View::Dynamics {
                self.dynamics_pane.recheck_sources();
            }
            if self.view == View::PolicyMap
                && let Some(request) = &self.policy_pane.request
                && let Err(error) = request.current.check_available()
            {
                self.interactive.fail_policy_context(&error);
                self.policy_pane.request = None;
                self.policy_pane.ready = false;
                self.policy_pane.failed_context = true;
                self.policy_pane.terminal = true;
                self.policy_pane.status = Some(error.to_string());
            }
            self.last_refresh = Instant::now();
        }
        egui::CentralPanel::default().show(ui, |ui| {
            if self
                .recording_opener
                .as_ref()
                .is_some_and(|worker| worker.active())
            {
                ui.label("Opening recording…");
            }
            let previous_view = self.view;
            self.workspace_shortcuts(ui.ctx());
            let mut current_run = None;
            let compact = ui.available_width() < 1200.;
            if self.run.is_some() {
                ui.horizontal(|ui| {
                    ui.set_min_height(28.);
                    self.run_control(ui);
                    ui.add_space(12.);
                    if let Some(run) = self.run.take() {
                        self.survey_model_route(&run);
                        if !typed_workspace::typed_view(&run, self.view, &self.selection)
                            && self.view == View::Environment
                            && let Err(error) = self.prepare_environment_catalog(&run)
                        {
                            self.error = Some(error.to_string());
                        }
                        if !compact && let Err(error) = self.toolbar(ui, &run) {
                            self.error = Some(error.to_string());
                        }
                        current_run = Some(run);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        self.application_controls(ui);
                        if let Some(run) = &current_run {
                            if let Err(error) = self.reference_control(ui, run) {
                                self.error = Some(error.to_string());
                            }
                            let manifest = run.recording().manifest();
                            if ui.available_width() >= 160. {
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!(
                                            "{} · {}",
                                            manifest.algorithm,
                                            crate::presentation::display_identifier(
                                                &manifest.environment
                                            )
                                        ))
                                        .weak(),
                                    )
                                    .truncate(),
                                )
                                .on_hover_text(&manifest.experiment_id);
                            }
                        }
                    });
                });
            }
            if let Some(run) = current_run {
                if compact && let Err(error) = self.toolbar(ui, &run) {
                    self.error = Some(error.to_string());
                }
                let typed = typed_workspace::typed_view(&run, self.view, &self.selection);
                if !typed && self.view == View::PolicyMap && !self.policy_evolution {
                    self.ensure_policy_execution(ui.ctx(), &run);
                }
                if self.view == View::Training
                    || self.view == View::Network && self.network_has_representation_source(&run)
                {
                    self.workspace_defaults = None;
                }
                if !typed
                    && !matches!(self.view, View::Dynamics | View::Environment)
                    && (self.view != View::PolicyMap
                        || self.policy_execution.mode == Some(PolicyMode::Interactive))
                    && let Err(error) = self.prepare_workspace_defaults(&run)
                {
                    self.error = Some(error.to_string());
                }
                self.selection_bar(ui, &run, self.view != previous_view);
                let typed = typed_workspace::typed_view(&run, self.view, &self.selection);
                ui.separator();
                ui.add_space(6.);
                if !typed
                    && self.view == View::PolicyMap
                    && (self.policy_evolution || self.policy_execution.mode.is_none())
                {
                    self.policy_evolution_controls(ui);
                }
                let change = match self.view {
                    View::Training => self.show_training(ui, &run),
                    _ if typed => Ok(None),
                    View::PolicyMap if self.policy_evolution => {
                        self.show_policy_evolution(ui, &run)
                    }
                    _ => self.learning_pane.show(ui, &run, &self.selection),
                };
                match change {
                    Ok(Some(revision)) => {
                        if let Err(error) = crate::pane::SelectionIntent::Revision(revision)
                            .apply(&run, &mut self.selection)
                        {
                            self.error = Some(error.to_string());
                        }
                        self.workspace_defaults = None;
                    }
                    Ok(None) => {}
                    Err(error) => self.error = Some(error.to_string()),
                }
                if !typed
                    && !matches!(self.view, View::Training | View::Value)
                    && self.learning_pane.expanded()
                {
                    reference_measurements::show(
                        ui,
                        &run,
                        &self.selection,
                        &mut self.loader,
                        false,
                    );
                }
                let result = match self.view {
                    View::Training => Ok(()),
                    _ if typed => self.typed_workspace(ui, &run),
                    View::PolicyMap if self.policy_evolution => Ok(()),
                    View::PolicyMap => self.policy_workspace(ui, &run),
                    View::Value => self.value_workspace(ui, &run),
                    View::Network => self.network_workspace(ui, &run),
                    View::Dynamics => self.dynamics_workspace(ui, &run),
                    View::Environment => self.environment_workspace(ui, &run),
                };
                if let Err(error) = result {
                    self.error = Some(error.to_string());
                }
                self.run = Some(run);
            } else {
                ui.vertical_centered(|ui| {
                    ui.add_space((ui.available_height() * 0.24).max(24.));
                    ui.heading("RL Visualization Laboratory");
                    ui.add_space(20.);
                    if ui.button("Open recording").clicked() {
                        self.open_dialog = true;
                    }
                    if let Some(demonstration) = self.demonstration.clone()
                        && ui.button("Open demonstration").clicked()
                    {
                        self.request_directory(&demonstration, ui.ctx());
                    }
                    let paths = self.recent.paths().to_vec();
                    if !paths.is_empty() {
                        ui.add_space(20.);
                        ui.strong("Recent recordings");
                        for path in paths {
                            let label = path
                                .file_name()
                                .unwrap_or(path.as_os_str())
                                .to_string_lossy();
                            if ui
                                .button(crate::presentation::display_identifier(&label))
                                .on_hover_text(path.display().to_string())
                                .clicked()
                            {
                                self.request_directory(&path, ui.ctx());
                            }
                        }
                    }
                });
            }
        });
        self.recording_dialog(ui.ctx());
        if let Err(error) = self.loader.finish_pass() {
            self.error = Some(error.to_string());
        }
        if let Some(error) = self.error.clone() {
            egui::Window::new("Recording notice")
                .collapsible(false)
                .show(ui.ctx(), |ui| {
                    ui.colored_label(WARM, error);
                    if ui.button("Retry recording reads").clicked() {
                        self.snapshot_catalog = None;
                        self.transport.clear_snapshot_state();
                        self.interactive.retire_session();
                        self.policy_pane.clear();
                        self.value_view.clear();
                        self.network_pane.clear();
                        self.learning_pane.retry();
                        self.dynamics_pane.clear();
                        self.environment_pane.clear();
                        self.loader.invalidate();
                        self.resolver.clear();
                        self.comparison.invalidate();
                        self.error = None;
                    }
                    if ui.button("Dismiss").clicked() {
                        self.error = None;
                    }
                });
        }
        ui.ctx()
            .request_repaint_after(if self.transport.is_playing() {
                Duration::from_millis(16)
            } else {
                Duration::from_secs(1)
            });
        #[cfg(feature = "screenshots")]
        self.capture_when_ready(ui);
    }
    #[cfg(feature = "screenshots")]
    fn capture_when_ready(&mut self, ui: &mut egui::Ui) {
        if self.capture.is_none() {
            return;
        }
        if self.timing.enabled() {
            let reasons = serde_json::json!({"playing":self.transport.is_playing(),"loader":self.loader.pending(),"policy_pending":self.interactive.policy_pending(),"policy_ready":self.policy_capture_ready(),"value_pending":self.interactive.value_pending(),"value_ready":self.value_capture_ready(),"value_status":self.value_view.status(),"value_terminal":self.value_view.terminal,"value_view_owned":self.value_view.owned_state_bytes(),"value_view_reserved":self.value_view.state_reservation_bytes(),"policy_status":self.policy_pane.status(),"policy_owners":self.interactive.policy_metrics(),"policy_view_reserved":self.policy_pane.state_reservation_bytes(),"policy_view_owned":self.policy_pane.owned_state_bytes(),"defaults":self.workspace_defaults.is_some(),"dynamics_pending":self.view == View::Dynamics && self.dynamics_pending(),"dynamics_ready":self.view == View::Dynamics && self.dynamics_ready(),"dynamics":(self.view == View::Dynamics).then(||self.dynamics_capture_evidence()),"network":(self.view == View::Network).then(||self.network_capture_evidence()),"learning":self.learning_pane.capture_evidence(),"learning_ready":self.learning_pane.capture_ready(),"environment":(self.view == View::Environment).then(||self.environment_capture_evidence())});
            if self.timing_readiness.as_ref() != Some(&reasons) {
                self.timing
                    .event("readiness", self.timing_pass, || reasons.clone());
                self.timing_readiness = Some(reasons);
            }
        }
        if let Some(error) = self
            .error
            .clone()
            .or_else(|| self.policy_capture_error())
            .or_else(|| self.value_capture_error())
            .or_else(|| self.learning_pane.capture_error())
            .or_else(|| {
                (self.view == View::Training)
                    .then(|| self.training_capture_error())
                    .flatten()
            })
            .or_else(|| {
                (self.view == View::Network)
                    .then(|| self.network_capture_error())
                    .flatten()
            })
            .or_else(|| {
                (self.view == View::Environment)
                    .then(|| self.environment_capture_error())
                    .flatten()
            })
            .or_else(|| {
                (self.view == View::Dynamics)
                    .then(|| self.dynamics_capture_error())
                    .flatten()
            })
        {
            eprintln!("Capture failed: {error}");
            self.timing.event(
                "capture_failed",
                self.timing_pass,
                || serde_json::json!({"error":error}),
            );
            self.capture = None;
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
        ui.ctx().request_repaint_after(Duration::from_millis(50));
        let typed = self
            .run
            .as_ref()
            .is_some_and(|run| typed_workspace::typed_view(run, self.view, &self.selection));
        let ready = if typed && self.view != View::Training {
            self.typed_pane.ready == Some(self.selection.revision)
        } else {
            (match self.view {
                View::Training => self.training_capture_ready(),
                View::PolicyMap if self.policy_evolution => self.policy_evolution_capture_ready(),
                View::PolicyMap => self.policy_capture_ready(),
                View::Value => self.value_capture_ready(),
                View::Network => self.network_ready(),
                View::Dynamics => self.dynamics_ready(),
                View::Environment => self.environment_capture_ready(),
            }) && self.learning_pane.capture_ready()
        };
        if ready && self.view == View::Training {
            match self.capture_playback {
                CapturePlayback::AwaitingFrame => {
                    self.capture_playback = CapturePlayback::QueuePlay(self.selection.revision);
                    return;
                }
                CapturePlayback::QueuePlay(_) => return,
                CapturePlayback::PlayingFrom(start) => {
                    let end = self.selection.revision;
                    if start == end || self.transport.is_playing() {
                        return;
                    }
                    self.timing.event(
                        "capture_playback",
                        self.timing_pass,
                        || serde_json::json!({"start_revision":start,"end_revision":end}),
                    );
                    self.capture_playback = CapturePlayback::Disabled;
                }
                CapturePlayback::Disabled => {}
            }
        }
        if !self.transport.is_playing()
            && self.loader.pending() == 0
            && self.snapshot_catalog.as_ref().is_none_or(|c| !c.pending())
            && ready
        {
            let path = self.capture.take().unwrap();
            let timing = self.timing.clone();
            timing.event("screenshot_request", 1, || serde_json::json!({"path":path,"ui":{"pixels_per_point":ui.ctx().pixels_per_point(),"requested_revision":self.selection.revision,"input":format!("{:?}",self.selection.input),"view":format!("{:?}",self.view),"reduced_motion":self.reduced_motion},"policy":(self.view == View::PolicyMap).then(|| self.policy_capture_evidence()),"value":(self.view == View::Value).then(|| self.value_capture_evidence()),"network":(self.view == View::Network).then(||self.network_capture_evidence()),"learning":self.learning_pane.capture_evidence(),"training":(self.view == View::Training).then(|| self.training_capture_evidence()),"policy_evolution":(self.view == View::PolicyMap && self.policy_evolution).then(|| self.policy_evolution_capture_evidence()),"dynamics":(self.view == View::Dynamics).then(|| self.dynamics_capture_evidence()),"environment":(self.view == View::Environment).then(|| self.environment_capture_evidence())}));
            let ctx = ui.ctx().clone();
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Screenshot(
                    egui::ScreenshotCallback::new(move |image| {
                        timing.event(
                            "screenshot_callback",
                            1,
                            || serde_json::json!({"width":image.size[0],"height":image.size[1]}),
                        );
                        let save = || -> std::result::Result<(), Box<dyn std::error::Error>> {
                            let mut encoder = png::Encoder::new(
                                std::fs::File::create(&path)?,
                                image.size[0] as u32,
                                image.size[1] as u32,
                            );
                            encoder.set_color(png::ColorType::Rgba);
                            encoder.set_depth(png::BitDepth::Eight);
                            let mut writer = encoder.write_header()?;
                            let pixels: Vec<u8> =
                                image.pixels.iter().flat_map(|p| p.to_array()).collect();
                            writer.write_image_data(&pixels)?;
                            Ok(())
                        };
                        let result = save();
                        timing.event(
                            "screenshot_written",
                            1,
                            || serde_json::json!({"ok":result.is_ok()}),
                        );
                        match result {
                            Ok(()) => println!("Captured {}", path.display()),
                            Err(e) => eprintln!("Capture failed: {e}"),
                        }
                        timing.event("close_request", 1, || serde_json::json!({}));
                        timing.summary();
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }),
                ));
        }
    }
}
impl eframe::App for Laboratory {
    #[cfg(feature = "screenshots")]
    fn raw_input_hook(&mut self, ctx: &egui::Context, input: &mut egui::RawInput) {
        if let CapturePlayback::QueuePlay(start) = self.capture_playback {
            input.events.push(egui::Event::Key {
                key: egui::Key::Space,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            });
            self.capture_playback = CapturePlayback::PlayingFrom(start);
        }
        self.value_capture_hover(
            ctx,
            input,
            std::env::var("RL_VIEWER_VALUE_HOVER")
                .ok()
                .map(|value| value == "true"),
        );
        self.policy_capture_hover(
            ctx,
            input,
            std::env::var("RL_VIEWER_POLICY_HOVER")
                .ok()
                .map(|value| value == "true"),
        );
    }
    #[cfg(feature = "screenshots")]
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.timing.event(
            "logic",
            self.timing_pass,
            || serde_json::json!({"egui_pass":ctx.cumulative_pass_nr()}),
        );
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.show(ui);
    }
}
