//! Recorded continuous-state Dynamics workspace. Expensive preparation is owned
//! by `DynamicsWorker`; this module only paints bounded immutable geometry.
use super::*;
#[path = "dynamics_benchmark.rs"]
mod benchmark;
#[path = "dynamics_decisions.rs"]
mod decisions;
#[path = "dynamics_replay.rs"]
mod replay;
use crate::dynamics::{
    ActionAxis, ActionValue, Axes,
    history::{CoverageMode, DomainSpec},
    source::{CatalogLimits, ProtocolKey, TrajectoryCatalog},
    work::{
        DynamicsWorker, HitView, PreparedFrame, Projection, Reference, Request, SourceSet,
        WorkLimits,
    },
};
use egui::{Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, Vec2};
use rl::{
    core::action::ActionSpace,
    diagnostics::visualization::{ActionDescriptor, EnvironmentSemantics, ObservationType},
};

const CYAN: Color32 = crate::theme::CURRENT;
const GOLD: Color32 = crate::theme::EMPIRICAL;
const PLOT: Color32 = crate::theme::INSET;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DisplayMode {
    Occupancy,
    Flow,
    Coverage,
}
impl DisplayMode {
    fn name(self) -> &'static str {
        match self {
            Self::Occupancy => "Occupancy",
            Self::Flow => "Flow",
            Self::Coverage => "Coverage",
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct Desired {
    domain: DomainSpec,
    generation: u64,
    revision: u64,
    sequence: u64,
    axes: Axes,
    action: ActionAxis,
    projection: Projection,
}

pub(super) struct DynamicsPane {
    pub worker: DynamicsWorker,
    catalog: Option<TrajectoryCatalog>,
    sources: Option<Arc<SourceSet>>,
    protocol: Option<ProtocolKey>,
    role: String,
    axes: Option<Axes>,
    z: Option<usize>,
    action: Option<ActionAxis>,
    three_d: bool,
    fit_episode: bool,
    state_action: bool,
    mode: DisplayMode,
    coverage_mode: CoverageMode,
    domain: DomainSpec,
    domain_epoch: u64,
    bounds_edit: [[String; 2]; 2],
    bounds_error: Option<String>,
    yaw: f32,
    pitch: f32,
    selected_sequence: Option<u64>,
    initial_revision: bool,
    requested: Option<Desired>,
    published: Option<Desired>,
    reference: Option<Arc<Reference>>,
    displayed: Option<Arc<PreparedFrame>>,
    occupancy_texture: Option<TextureHandle>,
    coverage_texture: Option<TextureHandle>,
    texture_state: Option<usize>,
    render_builds: u64,
    plot_bounds: Vec<Rect>,
    painted: bool,
    details_open: bool,
    invalidated: Option<String>,
    hit_view: Option<(usize, HitView)>,
    hit_selection: Option<SelectedInput>,
    generic_key: Option<CacheKey>,
    prediction_reason: Option<String>,
    replay: replay::ReplayView,
    benchmark: benchmark::Overlay,
}
impl DynamicsPane {
    pub fn new(ctx: egui::Context) -> Result<Self> {
        let worker = DynamicsWorker::new(WorkLimits::default(), move || ctx.request_repaint())?;
        Ok(Self {
            worker,
            catalog: None,
            sources: None,
            protocol: None,
            role: String::new(),
            axes: None,
            z: None,
            action: None,
            three_d: false,
            fit_episode: true,
            state_action: false,
            mode: DisplayMode::Occupancy,
            coverage_mode: CoverageMode::default(),
            domain: DomainSpec::default(),
            domain_epoch: 0,
            bounds_edit: Default::default(),
            bounds_error: None,
            yaw: 0.55,
            pitch: 0.3,
            selected_sequence: None,
            initial_revision: false,
            requested: None,
            published: None,
            reference: None,
            displayed: None,
            occupancy_texture: None,
            coverage_texture: None,
            texture_state: None,
            render_builds: 0,
            plot_bounds: Vec::with_capacity(3),
            painted: false,
            details_open: false,
            invalidated: None,
            hit_view: None,
            hit_selection: None,
            generic_key: None,
            prediction_reason: None,
            replay: replay::ReplayView::new(),
            benchmark: benchmark::Overlay::default(),
        })
    }
    pub fn clear(&mut self) {
        self.worker.retire();
        self.catalog = None;
        self.sources = None;
        self.protocol = None;
        self.role.clear();
        self.axes = None;
        self.z = None;
        self.action = None;
        self.selected_sequence = None;
        self.initial_revision = false;
        self.requested = None;
        self.published = None;
        self.reference = None;
        self.displayed = None;
        self.occupancy_texture = None;
        self.coverage_texture = None;
        self.texture_state = None;
        self.plot_bounds.clear();
        self.painted = false;
        self.details_open = false;
        self.invalidated = None;
        self.domain = DomainSpec::default();
        self.domain_epoch = 0;
        self.bounds_edit = Default::default();
        self.bounds_error = None;
        self.hit_view = None;
        self.hit_selection = None;
        self.generic_key = None;
        self.prediction_reason = None;
        self.replay.actual = None;
        self.benchmark.clear();
    }
    pub fn recheck_sources(&mut self) {
        if self.generic_key.is_some() {
            self.worker.recheck_episode();
            return;
        }
        if self.availability_current()
            && let (Some(frame), Some(sources)) = (&self.displayed, &self.sources)
        {
            self.worker.request_availability(frame, sources);
        }
    }
    fn availability_current(&self) -> bool {
        self.invalidated.is_none()
            && self.requested.is_some()
            && self.requested == self.published
            && self
                .displayed
                .as_ref()
                .zip(self.reference.as_ref())
                .is_some_and(|(frame, reference)| Arc::ptr_eq(reference, &frame.state.reference))
    }
    fn poll_availability(&mut self) {
        if self.availability_current()
            && let (Some(frame), Some(sources)) = (&self.displayed, &self.sources)
            && let Some(Err(error)) = self.worker.poll_availability(frame, sources)
        {
            self.invalidate_sources(error);
        }
    }
    fn invalidate_sources(&mut self, error: String) {
        self.invalidated = Some(error);
        self.worker.retire();
        self.requested = None;
        self.published = None;
        self.displayed = None;
        self.painted = false;
        self.occupancy_texture = None;
        self.coverage_texture = None;
        self.texture_state = None;
    }
    fn reset_projection(&mut self) {
        self.worker.cancel_hit();
        self.hit_view = None;
        self.reference = None;
        self.requested = None;
        self.painted = false;
        self.domain = DomainSpec::automatic(self.domain_epoch);
        self.bounds_edit = Default::default();
        self.bounds_error = None;
    }
    fn change_domain(&mut self, domain: DomainSpec) {
        if self.domain != domain {
            self.domain = domain;
            self.reference = None;
            self.requested = None;
            self.published = None;
            self.displayed = None;
            self.painted = false;
            self.worker.cancel_hit();
            self.hit_view = None;
        }
        self.bounds_error = None;
    }
    fn frame(&self) -> Option<&PreparedFrame> {
        self.displayed.as_deref()
    }
    fn pending(&self) -> bool {
        self.catalog
            .as_ref()
            .is_some_and(TrajectoryCatalog::pending)
            || self.worker.pending()
            || self.requested != self.published
    }
    fn ready(&self) -> bool {
        self.painted
            && !self.pending()
            && self.invalidated.is_none()
            && self.worker.error().is_none()
            && (self.frame().is_some()
                || self
                    .worker
                    .episode_frame()
                    .is_some_and(|f| self.generic_key.as_ref() == Some(&f.key)))
    }
    fn transport_ready(&self) -> bool {
        self.ready()
            || self.painted
                && self.invalidated.is_none()
                && self.worker.error().is_none()
                && self
                    .catalog
                    .as_ref()
                    .is_none_or(|catalog| !catalog.pending())
                && self.worker.episode_frame().is_some_and(|frame| {
                    self.generic_key.as_ref() == Some(&frame.key)
                        && self.action.unwrap_or(frame.action) == frame.action
                })
    }
    fn choose_defaults(&mut self, run: &RunIndex) {
        let semantics = EnvironmentSemantics::new(run.recording().manifest());
        let numeric: Vec<_> = (0..semantics.observation_dim())
            .filter(|&i| {
                !matches!(
                    semantics.observation(i).and_then(|v| v.kind),
                    Some(ObservationType::Categorical { .. })
                )
            })
            .collect();
        let mut preferred = Vec::new();
        if let Some(d) = semantics.descriptor() {
            for id in &d.preferences.coordinates {
                if let Some(i) = d.observations.iter().position(|f| &f.id == id)
                    && numeric.contains(&i)
                    && !preferred.contains(&i)
                {
                    preferred.push(i);
                }
            }
            for (i, f) in d.observations.iter().enumerate() {
                if f.coordinate && numeric.contains(&i) && !preferred.contains(&i) {
                    preferred.push(i);
                }
            }
        }
        for i in numeric {
            if !preferred.contains(&i) {
                preferred.push(i);
            }
        }
        self.axes = preferred
            .first()
            .zip(preferred.get(1))
            .map(|(&x, &y)| Axes { x, y });
        self.z = preferred.get(2).copied();
        self.action = Some(match &run.recording().manifest().action_space {
            ActionSpace::Discrete { .. } => ActionAxis::Discrete,
            _ => ActionAxis::Continuous(match semantics.descriptor().map(|d| &d.actions) {
                Some(ActionDescriptor::Continuous {
                    preferred_component: Some(j),
                    ..
                }) => *j,
                _ => 0,
            }),
        });
        #[cfg(feature = "screenshots")]
        {
            self.mode = match std::env::var("RL_VIEWER_DYNAMICS_MODE").as_deref() {
                Ok("flow") => DisplayMode::Flow,
                Ok("coverage") => DisplayMode::Coverage,
                _ => DisplayMode::Occupancy,
            };
            self.coverage_mode = match std::env::var("RL_VIEWER_DYNAMICS_COVERAGE").as_deref() {
                Ok("current") => CoverageMode::Current,
                Ok("new") => CoverageMode::New,
                _ => CoverageMode::Cumulative,
            };
            if let Ok(view) = std::env::var("RL_VIEWER_DYNAMICS_3D") {
                self.three_d = true;
                self.state_action = view == "state-action";
            }
            if let Ok(j) = std::env::var("RL_VIEWER_DYNAMICS_ACTION")
                .unwrap_or_default()
                .parse::<usize>()
            {
                self.action = Some(ActionAxis::Continuous(j));
            }
        }
    }
    fn projection(&self) -> Projection {
        if !self.three_d {
            Projection::None
        } else if self.state_action {
            if matches!(self.action, Some(ActionAxis::Continuous(_))) {
                Projection::StateAction
            } else {
                Projection::None
            }
        } else {
            self.axes
                .zip(self.z)
                .map_or(Projection::None, |(a, z)| Projection::State([a.x, a.y, z]))
        }
    }
    fn accept_frame(&mut self) {
        self.worker.poll();
        if let Some(next) = self.worker.frame_arc()
            && self.requested.is_some_and(|d| {
                d.domain == next.state.key.domain()
                    && d.generation == next.state.key.generation()
                    && d.revision == next.state.revision
                    && d.sequence == next.selected.key.sequence
                    && d.axes == next.state.key.axes()
                    && d.action == next.action
                    && d.projection == next.projection
            })
            && self
                .displayed
                .as_ref()
                .is_none_or(|old| !Arc::ptr_eq(old, &next))
        {
            // Replace the old lease only after every requested identity matches.
            if self.bounds_edit[0][0].is_empty() {
                self.bounds_edit = next
                    .state
                    .reference
                    .grid
                    .grid
                    .bounds()
                    .map(|range| range.map(|v| v.to_string()));
            }
            self.reference = Some(next.state.reference.clone());
            self.displayed = Some(next);
            self.published = self.requested;
            self.painted = false;
        }
    }
    fn textures(&mut self, ctx: &egui::Context) {
        let Some(frame) = self.frame() else { return };
        let ptr = Arc::as_ptr(&frame.state) as usize;
        if self.texture_state == Some(ptr) {
            return;
        }
        let grid = &frame.state.reference.grid.grid;
        let [w, h] = grid.cells();
        let mut density = Vec::with_capacity(w * h);
        let mut first = Vec::with_capacity(w * h);
        let revisions = &frame.state.coverage.points;
        let first_revision = revisions.first().map_or(0, |point| point.revision);
        let revision_span = revisions
            .last()
            .map_or(0, |point| point.revision - first_revision);
        for y in (0..h).rev() {
            for x in 0..w {
                let i = y * w + x;
                let p = frame.state.occupancy.mass[i];
                density.push(if p == 0. {
                    PLOT
                } else {
                    crate::landscape::sequential_color(((p.log10() + 4.) / 4.) as f32)
                });
                first.push(
                    frame.state.coverage.first_seen[i]
                        .filter(|&r| r <= frame.state.revision)
                        .map_or(PLOT, |r| {
                            let fraction = if revision_span == 0 {
                                0.5
                            } else {
                                r.saturating_sub(first_revision) as f32 / revision_span as f32
                            };
                            crate::landscape::sequential_color(fraction)
                        }),
                );
            }
        }
        self.occupancy_texture = Some(ctx.load_texture(
            "dynamics-occupancy",
            ColorImage::new([w, h], density),
            TextureOptions::LINEAR,
        ));
        self.coverage_texture = Some(ctx.load_texture(
            "dynamics-first-seen",
            ColorImage::new([w, h], first),
            TextureOptions::NEAREST,
        ));
        self.texture_state = Some(ptr);
        self.render_builds += 1;
    }
}

fn expand([lo, hi]: [f64; 2]) -> [f64; 2] {
    if lo == hi {
        let pad = lo.abs().mul_add(0.05, 0.5);
        [lo - pad, hi + pad]
    } else {
        [lo, hi]
    }
}
fn label(ui: &mut egui::Ui, text: &str, help: &str) {
    ui.label(text).on_hover_text(help);
}
fn coordinate_label(c: &crate::dynamics::Coordinate) -> String {
    c.unit
        .as_ref()
        .map_or_else(|| c.label.clone(), |u| format!("{} [{}]", c.label, u))
}
fn compact_label(label: &str, limit: usize) -> String {
    let mut chars = label.chars();
    let shortened: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        format!("{shortened}…")
    } else {
        shortened
    }
}
fn action_label(e: &crate::dynamics::Episode, a: ActionAxis) -> String {
    let name = e.action_label(a).unwrap_or("Action");
    e.action_unit(a)
        .ok()
        .flatten()
        .map_or_else(|| name.to_owned(), |u| format!("{name} [{u}]"))
}

impl Laboratory {
    pub fn dynamics_frame(&self) -> Option<&PreparedFrame> {
        self.dynamics_pane.frame()
    }
    /// The original selected episode, independent of spatial projection support.
    pub fn dynamics_episode(&self) -> Option<(&CacheKey, &crate::dynamics::Episode)> {
        self.dynamics_frame()
            .map(|f| (&f.selected.key, &f.selected.episode))
            .or_else(|| {
                self.dynamics_pane
                    .worker
                    .episode_frame()
                    .filter(|f| self.dynamics_pane.generic_key.as_ref() == Some(&f.key))
                    .map(|f| (&f.key, &f.episode))
            })
    }
    pub fn dynamics_pending(&self) -> bool {
        self.dynamics_pane.pending() || self.interactive.decision_pending()
    }
    pub fn dynamics_ready(&self) -> bool {
        if self.dynamics_pane.replay.enabled {
            self.dynamics_pane.replay.actual.is_some()
        } else {
            self.dynamics_pane.ready() && !self.interactive.decision_pending()
        }
    }
    pub fn dynamics_counters(&self) -> crate::dynamics::work::Counters {
        self.dynamics_pane.worker.counters()
    }
    pub fn dynamics_render_builds(&self) -> u64 {
        self.dynamics_pane.render_builds
    }
    pub fn dynamics_plot_bounds(&self) -> &[Rect] {
        &self.dynamics_pane.plot_bounds
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn dynamics_capture_error(&self) -> Option<String> {
        self.dynamics_pane
            .invalidated
            .clone()
            .or_else(|| self.dynamics_pane.worker.error().map(str::to_owned))
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn dynamics_capture_evidence(&self) -> serde_json::Value {
        if self.dynamics_pane.replay.enabled {
            return serde_json::json!({"population":"replay retained","source_revision":self.dynamics_pane.replay.actual,
                "requested_revision":self.selection.revision,"input":format!("{:?}",self.selection.input)});
        }
        let Some(frame) = self.dynamics_frame() else {
            return self.dynamics_pane.worker.episode_frame().map_or(serde_json::Value::Null,|frame| {
                let episode=&frame.episode;
                let step=match self.selection.input{SelectedInput::TrajectoryStep{sequence,step}if sequence==frame.key.sequence=>step.min(episode.transitions()),_=>0};
                let row=episode.step(step).ok();
                serde_json::json!({"generic_episode":true,"episode_sequence":frame.key.sequence,"episode_root":frame.key.root.to_string_lossy(),"episode_id":episode.record().id,"evaluation_revision":episode.context().identity.learner_update_revision,
                    "step":step,"transitions":episode.transitions(),"observation":row.map(|r|r.observation),"mask":row.and_then(|r|r.mask),
                    "complete_recorded_action":row.and_then(|r|r.action).map(decisions::action_evidence),"reward":row.and_then(|r|r.reward),"next_observation":row.and_then(|r|r.next_observation),
                    "terminated":row.and_then(|r|r.terminated),"truncated":row.and_then(|r|r.truncated),"return_to_go":row.and_then(|r|r.return_to_go),
                    "source_policy":episode.record().evaluation.continuation_policy,"conditioning":episode.record().evaluation.conditioning,"discount":episode.record().evaluation.discount,
                    "requested_model_revision":self.selection.revision,"prediction":decisions::prediction_evidence(&self.interactive,&frame.key,step,self.selection.revision),
                    "prediction_unavailable":self.dynamics_pane.prediction_reason.as_deref().or_else(||self.interactive.decision_error()),"retained_bytes":frame.retained_bytes,"owner_envelope_bytes":self.dynamics_pane.worker.envelope_bytes(),"painted":self.dynamics_pane.painted})
            });
        };
        let episode = &frame.selected.episode;
        let step = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step }
                if sequence == frame.selected.key.sequence =>
            {
                step.min(episode.transitions())
            }
            _ => 0,
        };
        let current = episode.step(step).ok();
        let action =
            episode
                .action_value(step, frame.action)
                .ok()
                .flatten()
                .map(|value| match value {
                    ActionValue::Continuous(x) => serde_json::json!({"continuous":x}),
                    ActionValue::Discrete(id) => {
                        serde_json::json!({"discrete_id":id,"label":episode.discrete_label(id)})
                    }
                });
        let a = frame.state.key.axes();
        let selected_xy = current.map(|s| [s.observation[a.x], s.observation[a.y]]);
        let selected_z = match frame.projection {
            Projection::State([_, _, z]) => current.map(|s| s.observation[z]),
            _ => None,
        };
        let bounds = frame.state.reference.grid.grid.bounds();
        let counters = self.dynamics_pane.worker.counters();
        let catalog = self.dynamics_pane.catalog.as_ref().map(|c| {
            let m = c.metrics();
            let us = |d: Option<std::time::Duration>| d.map(|v| v.as_micros() as u64);
            serde_json::json!({
                "generation":m.generation,"submitted_generations":m.submitted_generations,
                "started_generations":m.started_generations,"completed_generations":m.completed_generations,
                "cancelled_generations":m.cancelled_generations,"read_attempts":m.read_attempts,"decoded_sources":m.decoded_sources,
                "submitted_at_us":us(m.submitted_at),"started_at_us":us(m.started_at),
                "last_progress_at_us":us(m.last_progress_at),"completed_at_us":us(m.completed_at),
                "consumed_at_us":us(m.consumed_at),"worker_duration_us":us(m.worker_duration),
            })
        });
        let mut projection_bounds = frame.background.bounds;
        if frame.projection == Projection::StateAction
            && let Some(run) = &self.run
            && let (ActionSpace::Box { low, high }, ActionAxis::Continuous(j)) =
                (&run.recording().manifest().action_space, frame.action)
        {
            projection_bounds[2] = expand([low[j] as f64, high[j] as f64]);
        }
        let worker = serde_json::json!({"submitted":counters.submitted,"source_reads":counters.source_reads,"state_builds":counters.state_builds,
                "source_read_bytes":counters.source_read_bytes,"decoded_hits":counters.decoded_hits,
                "decoded_cache_bytes":counters.decoded_cache_bytes,"projection_builds":counters.projection_builds,
                "selected_builds":counters.selected_builds,"action_builds":counters.action_builds,"cancelled":counters.cancelled,
                "envelope_bytes":self.dynamics_pane.worker.envelope_bytes()});
        let mut evidence = serde_json::json!({
            "evaluation_revision":frame.state.revision,"episode_sequence":frame.selected.key.sequence,
            "episode_step":step,"episode_decisions":episode.transitions(),"episode_status":format!("{:?}",episode.status()),
            "observation_dim":episode.coordinates().len(),"selected_xy":selected_xy,"selected_z":selected_z,
            "reward":current.and_then(|s|s.reward),"action_value":action,
            "action_axis":format!("{:?}",frame.action),"action_label":action_label(episode,frame.action),
            "axes":[a.x,a.y],"projection":format!("{:?}",frame.projection),"display_mode":self.dynamics_pane.mode.name(),
            "reference_sources":frame.state.reference.source_keys.iter().map(|k|k.sequence).collect::<Vec<_>>(),
            "reference_bounds":bounds,"reference_empirical":frame.state.reference.grid.empirical,
            "reference_bytes":frame.state.reference.retained_bytes(),
            "population_episodes":frame.state.population_episodes,"population_decisions":frame.state.population_decisions,
            "included_visits":frame.state.occupancy.included,"excluded_visits":frame.state.occupancy.excluded,
            "displayed_episodes":if frame.projection == Projection::None {frame.state.displayed_episodes} else {frame.background.paths.len()},"displayed_segments":if frame.projection == Projection::None {frame.state.displayed_segments} else {frame.background.displayed_segments},
            "history_episodes":frame.state.history_episodes,"history_decisions":frame.state.history_decisions,
            "coverage_revisions":frame.state.coverage.points.len(),"grid_cells":frame.state.coverage.display_grid_cells,
            "source_omitted":self.dynamics_pane.sources.as_ref().map_or(0,|s|s.omitted_sources()),
            "catalog_omitted":self.dynamics_pane.catalog.as_ref().map_or(0,TrajectoryCatalog::omitted),
            "catalog_rejected":self.dynamics_pane.catalog.as_ref().map_or(0,TrajectoryCatalog::rejected),
            "worker":worker,
            "benchmark_path":self.dynamics_pane.benchmark.path.as_ref().map(|path| serde_json::json!({
                "sequence":path.sequence, "revision":path.revision, "scenario_seed":path.seed,
                "points":path.points.len(), "quality":format!("{:?}", path.quality),
            })),
            "render_builds":self.dynamics_pane.render_builds,"painted":self.dynamics_pane.painted,
        });
        evidence.as_object_mut().unwrap().extend(serde_json::json!({
            "complete_observation":current.map(|r|r.observation),"mask":current.and_then(|r|r.mask),"complete_recorded_action":current.and_then(|r|r.action).map(decisions::action_evidence),
            "next_observation":current.and_then(|r|r.next_observation),"terminated":current.and_then(|r|r.terminated),"truncated":current.and_then(|r|r.truncated),"return_to_go":current.and_then(|r|r.return_to_go),
            "prediction":decisions::prediction_evidence(&self.interactive,&frame.selected.key,step,self.selection.revision),"prediction_unavailable":self.dynamics_pane.prediction_reason.as_deref().or_else(||self.interactive.decision_error()),
        }).as_object().unwrap().clone());
        evidence.as_object_mut().unwrap().extend(serde_json::json!({
            "reference_id":frame.state.reference.id(),"domain":format!("{:?}", frame.state.reference.domain()),
            "reference_episodes":frame.state.reference.episodes(),"reference_decisions":frame.state.reference.decisions(),
            "reference_state_extrema":(0..episode.coordinates().len()).map(|i|frame.state.reference.state_extrema(i)).collect::<Vec<_>>(),
            "reference_action_extrema":episode.action_axes().iter().filter_map(|a| if let ActionAxis::Continuous(j)=a {frame.state.reference.action_bounds(*j)} else {None}).collect::<Vec<_>>(),
            "coverage_mode":self.dynamics_pane.coverage_mode.name(),
            "coverage_points":frame.state.coverage.points.iter().map(|p|serde_json::json!({"revision":p.revision,"occupied":p.occupied,"new_cells":p.new_cells,"cumulative":p.cumulative,"episodes":p.episodes,"decisions":p.decisions,"included":p.included,"excluded":p.excluded})).collect::<Vec<_>>(),
            "projection_bounds":projection_bounds,"camera":{"yaw":self.dynamics_pane.yaw,"pitch":self.dynamics_pane.pitch},
            "background":{"displayed_episodes":frame.background.paths.len(),"eligible_episodes":frame.background.eligible_episodes,
                "displayed_segments":frame.background.displayed_segments,"eligible_segments":frame.background.eligible_segments,
                "retained_bytes":frame.background.retained_bytes,
                "sequences":frame.background.paths.iter().map(|p|p.key.sequence).collect::<Vec<_>>()},
            "catalog":catalog,
        }).as_object().unwrap().clone());
        evidence
    }
    pub(super) fn dynamics_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        if self.dynamics_pane.replay.enabled && replay::probe_id(run).is_some() {
            return self.replay_transport(ui, run);
        }
        let pane = &mut self.dynamics_pane;
        let mut positions: Vec<u64> = pane
            .sources
            .as_ref()
            .filter(|s| {
                pane.catalog
                    .as_ref()
                    .is_some_and(|c| !c.pending() && c.generation() == s.generation())
            })
            .map(|sources| {
                sources
                    .sources()
                    .iter()
                    .map(|source| source.revision)
                    .collect()
            })
            .unwrap_or_default();
        positions.dedup();
        if pane.generic_key.is_some() {
            positions = pane
                .catalog
                .as_ref()
                .map(|c| c.entries().iter().map(|s| s.revision).collect())
                .unwrap_or_default();
            positions.sort_unstable();
            positions.dedup();
        }
        let published = pane.frame().map(|f| f.state.revision).or_else(|| {
            pane.worker
                .episode_frame()
                .map(|f| f.episode.context().identity.learner_update_revision)
        });
        // A check-only generic source refresh keeps the painted exact episode.
        // It is not a new checkpoint/component preparation for this transport.
        let ready = pane.transport_ready();
        let checking = pane
            .catalog
            .as_ref()
            .is_some_and(TrajectoryCatalog::pending)
            || (positions.is_empty()
                && pane
                    .catalog
                    .as_ref()
                    .is_some_and(|c| c.groups().iter().any(|g| g.protocol.unconditional()))
                && pane.invalidated.is_none());
        let error = pane.invalidated.as_deref().or_else(|| pane.worker.error());
        self.transport.hold_unavailable_snapshot(
            (!positions.is_empty() && !positions.contains(&self.selection.revision))
                .then_some(self.selection.revision),
        );
        let retry = self.transport.show_evaluations(
            ui,
            run,
            &mut self.selection,
            crate::timeline_ui::SnapshotPresentation {
                positions: &positions,
                published,
                ready,
                checking,
                error,
            },
        )?;
        if retry {
            if let Some(c) = &mut pane.catalog {
                c.retry(run, &mut self.loader)?;
            }
            pane.worker.retry();
            pane.sources = None;
            pane.requested = None;
            pane.published = None;
            pane.displayed = None;
            pane.invalidated = None;
            pane.painted = false;
            pane.occupancy_texture = None;
            pane.coverage_texture = None;
            pane.texture_state = None;
        }
        Ok(())
    }
    pub(super) fn dynamics_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        if replay::probe_id(run).is_some() {
            let previous = self.dynamics_pane.replay.enabled;
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.dynamics_pane.replay.enabled,
                    false,
                    "Evaluation trajectories",
                );
                ui.selectable_value(
                    &mut self.dynamics_pane.replay.enabled,
                    true,
                    "Replay sample",
                );
            });
            if previous != self.dynamics_pane.replay.enabled {
                self.dynamics_pane.clear();
                self.interactive.clear_decision();
                self.transport.clear_snapshot_state();
            }
            if self.dynamics_pane.replay.enabled {
                return self.show_replay(ui, run);
            }
        } else {
            self.dynamics_pane.replay.enabled = false;
        }
        let pane = &mut self.dynamics_pane;
        if pane
            .catalog
            .as_ref()
            .is_none_or(|c| !c.matches(run, &self.selection.model_role))
        {
            pane.clear();
            pane.role = self.selection.model_role.clone();
            pane.choose_defaults(run);
            pane.catalog = Some(TrajectoryCatalog::with_decode_gate(
                run,
                &pane.role,
                CatalogLimits::default(),
                pane.worker.decode_gate(),
            )?);
        }
        pane.plot_bounds.clear();
        // Generic episodes and unpublished spatial requests cannot retain the
        // preceding benchmark overlay.
        if pane.requested != pane.published || pane.axes.is_none() {
            pane.benchmark.clear();
        }
        if let Some(c) = &mut pane.catalog {
            c.poll(run, &mut self.loader)?;
        }
        pane.accept_frame();
        if let SelectedInput::TrajectoryStep { sequence, .. } = self.selection.input
            && pane.catalog.as_ref().is_some_and(|c| {
                !c.pending()
                    && !c
                        .entries()
                        .iter()
                        .any(|source| source.key.sequence == sequence)
            })
        {
            if pane.generic_key.is_some() || pane.displayed.is_some() || pane.requested.is_some() {
                pane.worker.retire();
            }
            pane.generic_key = None;
            pane.displayed = None;
            pane.requested = None;
            pane.published = None;
            pane.painted = false;
            pane.occupancy_texture = None;
            pane.coverage_texture = None;
            pane.texture_state = None;
            self.interactive.clear_decision();
            ui.heading("Dynamics");
            ui.weak(format!("Selected episode unavailable: record #{sequence} is missing, rejected, pruned or outside the admitted catalogue. Its selection is preserved."));
            return Ok(());
        }
        let conditional = pane.catalog.as_ref().is_some_and(|c| {
            let selected = match self.selection.input {
                SelectedInput::TrajectoryStep { sequence, .. } => {
                    c.entries().iter().find(|s| s.key.sequence == sequence)
                }
                _ => None,
            };
            selected.is_some_and(|s| !s.protocol.unconditional())
                || !c.entries().is_empty() && !c.groups().iter().any(|g| g.protocol.unconditional())
        });
        if pane.axes.is_none() || conditional {
            return self.dynamics_generic_episode(ui, run);
        }
        if pane.invalidated.is_none() {
            let catalog = pane.catalog.as_ref().unwrap();
            if !catalog.pending() {
                let group = pane
                    .protocol
                    .as_ref()
                    .and_then(|p| catalog.groups().iter().find(|g| &g.protocol == p))
                    .or_else(|| {
                        // A workspace entry from a probe keeps the shared real
                        // checkpoint when one exact compatible population exists.
                        // Explicit episodes and chosen protocols keep their route.
                        if pane.protocol.is_none()
                            && !pane.initial_revision
                            && !matches!(self.selection.input, SelectedInput::TrajectoryStep { .. })
                        {
                            catalog.groups().iter().find(|group| {
                                group.protocol.unconditional()
                                    && group.positions.contains(&self.selection.revision)
                            })
                        } else {
                            None
                        }
                    })
                    .or_else(|| catalog.groups().iter().find(|g| g.protocol.unconditional()));
                if let Some(group) = group {
                    let changed = pane.protocol.as_ref() != Some(&group.protocol);
                    if changed {
                        pane.protocol = Some(group.protocol.clone());
                        pane.sources = None;
                        pane.reference = None;
                        pane.requested = None;
                        pane.published = None;
                        pane.displayed = None;
                        pane.texture_state = None;
                    }
                    if pane
                        .sources
                        .as_ref()
                        .is_none_or(|s| s.generation() != catalog.generation())
                    {
                        let entries: Vec<_> = catalog
                            .entries()
                            .iter()
                            .filter(|s| s.protocol == group.protocol)
                            .cloned()
                            .collect();
                        pane.sources = Some(SourceSet::new(
                            run,
                            &entries,
                            catalog.generation(),
                            &WorkLimits::default(),
                        )?);
                        pane.requested = None;
                    }
                }
            }
        }
        if let SelectedInput::TrajectoryStep { sequence, .. } = self.selection.input
            && pane.sources.as_ref().is_some_and(|s| {
                !s.sources()
                    .iter()
                    .any(|source| source.key.sequence == sequence)
            })
        {
            return self.dynamics_generic_episode(ui, run);
        }
        if let Some(sources) = pane.sources.as_ref().filter(|_| pane.invalidated.is_none()) {
            if pane.generic_key.take().is_some() {
                pane.worker.retire();
                pane.requested = None;
                pane.published = None;
                pane.displayed = None;
            }
            let entries = sources.sources();
            let revisions: Vec<u64> = entries.iter().map(|s| s.revision).collect();
            if !pane.initial_revision {
                if let SelectedInput::TrajectoryStep { sequence, .. } = self.selection.input
                    && entries.iter().any(|source| source.key.sequence == sequence)
                {
                    pane.selected_sequence = Some(sequence);
                }
                pane.initial_revision = true;
            }
            let revision = if revisions.contains(&self.selection.revision) {
                self.selection.revision
            } else {
                *revisions
                    .iter()
                    .rfind(|&&r| r <= self.selection.revision)
                    .or_else(|| revisions.last())
                    .unwrap()
            };
            let selected = entries
                .iter()
                .find(|s| s.revision == revision && pane.selected_sequence == Some(s.key.sequence))
                .or_else(|| entries.iter().rev().find(|s| s.revision == revision))
                .unwrap();
            if pane.selected_sequence != Some(selected.key.sequence) {
                pane.selected_sequence = Some(selected.key.sequence);
                let step = if cfg!(feature = "screenshots") {
                    match std::env::var("RL_VIEWER_DYNAMICS_STEP") {
                        Ok(value) => value.parse::<usize>().map_err(|_| {
                            rl::Error::Config("invalid Dynamics capture step".into())
                        })?,
                        Err(_) => 0,
                    }
                } else {
                    0
                };
                if step > selected.transitions {
                    return Err(rl::Error::Config(
                        "Dynamics capture step outside selected episode".into(),
                    ));
                }
                if self.selection.input == SelectedInput::None
                    || cfg!(feature = "screenshots")
                        && std::env::var_os("RL_VIEWER_DYNAMICS_STEP").is_some()
                {
                    crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                        sequence: selected.key.sequence,
                        step,
                    })
                    .apply(run, &mut self.selection)?;
                }
            }
            let action = pane.action.unwrap();
            let projection = pane.projection();
            let axes = pane.axes.unwrap();
            let desired = Desired {
                domain: pane.domain,
                generation: sources.generation(),
                revision,
                sequence: selected.key.sequence,
                axes,
                action,
                projection,
            };
            if pane.requested != Some(desired) {
                pane.worker.request(Request {
                    domain: pane.domain,
                    sources: sources.clone(),
                    revision,
                    axes,
                    cells: [64, 64],
                    selected: selected.key.clone(),
                    action,
                    projection,
                    reference: pane.reference.clone(),
                })?;
                pane.requested = Some(desired);
                pane.painted = false;
            }
        }
        // Both frame adoption and catalog/source-generation reconciliation must
        // precede availability consumption, so old errors cannot poison either.
        pane.poll_availability();
        pane.textures(ui.ctx());
        if !pane.three_d
            && pane.requested == pane.published
            && let Some(frame) = &pane.displayed
            && frame.projection == Projection::None
            && let Some(source) = pane.sources.as_ref().and_then(|sources| {
                sources
                    .sources()
                    .iter()
                    .find(|source| source.key == frame.selected.key)
            })
        {
            pane.benchmark.advance(
                run,
                &self.comparison,
                source,
                &frame.selected.episode,
                frame.state.key.axes(),
                &mut self.loader,
            );
        } else {
            pane.benchmark.clear();
        }
        let selected = pane
            .frame()
            .map(|f| (f.selected.key.clone(), f.selected.episode.clone()));
        if let Some((key, episode)) = selected {
            let step = match self.selection.input {
                SelectedInput::TrajectoryStep { sequence, step } if sequence == key.sequence => {
                    step
                }
                _ => 0,
            };
            self.prepare_dynamics_decision(run, key, &episode, step);
        }
        self.dynamics_layout(ui, run)?;
        Ok(())
    }
    fn dynamics_layout(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let available = ui.available_size();
        egui::ScrollArea::vertical()
            .id_salt("dynamics-content")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let content_top = ui.cursor().top();
                ui.horizontal_wrapped(|ui| {
                    for mode in [
                        DisplayMode::Occupancy,
                        DisplayMode::Flow,
                        DisplayMode::Coverage,
                    ] {
                        ui.selectable_value(&mut self.dynamics_pane.mode, mode, mode.name());
                    }
                    ui.selectable_value(&mut self.dynamics_pane.three_d, false, "2D");
                    ui.selectable_value(&mut self.dynamics_pane.three_d, true, "3D");
                    if !self.dynamics_pane.three_d {
                        ui.toggle_value(&mut self.dynamics_pane.fit_episode, "Fit episode")
                            .on_hover_text("Frame the selected displayed trajectory. The visitation grid and colour scale remain fixed; turn this off to see the full domain.");
                    }
                    crate::landscape::popover(ui, "Spatial settings", |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(480.)
                            .show(ui, |ui| {
                                if let Some(f) = self.dynamics_frame() {
                                    ui.small(format!(
                                        "Evaluation k={} · {} episodes",
                                        f.state.revision, f.state.population_episodes
                                    ));
                                }
                                if let Some((label, help)) = self.dynamics_pane.benchmark.status() {
                                    ui.label(label).on_hover_text(help);
                                }
                                self.dynamics_controls(ui, run);
                            });
                    });
                });
                self.dynamics_episode_controls(ui, run);
                let width = available.x;
                let controls_height = ui.cursor().top() - content_top;
                let inspector_height = if self.dynamics_pane.details_open {
                    390.
                } else {
                    198.
                };
                let main_height = (available.y - controls_height - inspector_height).max(220.);
                if width >= 900. {
                    let main_width = (width * 0.63).max(180.);
                    let side_width = (width - main_width - 8.).max(180.);
                    let side_plot_height = ((main_height - 72.) / 2.).max(120.);
                    ui.horizontal_top(|ui| {
                        ui.allocate_ui_with_layout(
                            Vec2::new(main_width, main_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                crate::theme::panel_frame().show(ui, |ui| {
                                    ui.set_min_width((main_width - 24.).max(0.));
                                    ui.label(
                                        egui::RichText::new("Trajectory and visitation")
                                            .size(17.)
                                            .strong(),
                                    );
                                    self.dynamics_main_plot(
                                        ui,
                                        run,
                                        Vec2::new(
                                            (main_width - 24.).max(120.),
                                            (main_height - 46.).max(120.),
                                        ),
                                    );
                                });
                            },
                        );
                        ui.allocate_ui_with_layout(
                            Vec2::new(side_width, main_height),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                crate::theme::panel_frame().show(ui, |ui| {
                                    ui.set_min_width((side_width - 24.).max(0.));
                                    ui.label(
                                        egui::RichText::new("Episode timeline").size(16.).strong(),
                                    );
                                    self.dynamics_timeline_plot(
                                        ui,
                                        run,
                                        Vec2::new(
                                            (side_width - 24.).max(120.),
                                            (side_plot_height - 16.).max(100.),
                                        ),
                                    );
                                });
                                crate::theme::panel_frame().show(ui, |ui| {
                                    ui.set_min_width((side_width - 24.).max(0.));
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(
                                            egui::RichText::new("Evaluation coverage")
                                                .size(16.)
                                                .strong(),
                                        );
                                        for mode in [
                                            CoverageMode::Cumulative,
                                            CoverageMode::Current,
                                            CoverageMode::New,
                                        ] {
                                            ui.selectable_value(
                                                &mut self.dynamics_pane.coverage_mode,
                                                mode,
                                                mode.name(),
                                            );
                                        }
                                    });
                                    self.dynamics_coverage_plot(
                                        ui,
                                        Vec2::new(
                                            (side_width - 24.).max(120.),
                                            (side_plot_height - 16.).max(100.),
                                        ),
                                    );
                                });
                            },
                        );
                    });
                } else {
                    self.dynamics_main_plot(ui, run, Vec2::new(width.max(180.), main_height));
                    ui.horizontal(|ui| {
                        let lower_w = (width - 8.) / 2.;
                        ui.allocate_ui_with_layout(
                            Vec2::new(lower_w, 180.),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.strong("Episode timeline");
                                self.dynamics_timeline_plot(
                                    ui,
                                    run,
                                    Vec2::new(lower_w.max(120.), 156.),
                                );
                            },
                        );
                        ui.allocate_ui_with_layout(
                            Vec2::new(lower_w, 180.),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.strong("Evaluation coverage");
                                    for mode in [
                                        CoverageMode::Cumulative,
                                        CoverageMode::Current,
                                        CoverageMode::New,
                                    ] {
                                        ui.selectable_value(
                                            &mut self.dynamics_pane.coverage_mode,
                                            mode,
                                            mode.name(),
                                        );
                                    }
                                });
                                self.dynamics_coverage_plot(ui, Vec2::new(lower_w.max(120.), 156.));
                            },
                        );
                    });
                }
                if let Some(frame) = self.dynamics_pane.displayed.clone() {
                    let step = match self.selection.input {
                        SelectedInput::TrajectoryStep { sequence, step }
                            if sequence == frame.selected.key.sequence =>
                        {
                            step.min(frame.selected.episode.transitions())
                        }
                        _ => 0,
                    };
                    let details_open = crate::theme::panel_frame()
                        .show(ui, |ui| {
                            ui.set_min_width((width - 24.).max(0.));
                            decisions::decision_strip(
                                ui,
                                &frame.selected.episode,
                                step,
                                frame.action,
                                self.dynamics_pane.axes,
                            );
                            decisions::prediction_summary(
                                ui,
                                &self.interactive,
                                &frame.selected.key,
                                step,
                                self.selection.revision,
                                frame.action,
                                self.dynamics_pane.prediction_reason.as_deref(),
                            );
                            decisions::decision_details(
                                ui,
                                &frame.selected.episode,
                                step,
                                &self.interactive,
                                &frame.selected.key,
                                self.selection.revision,
                                self.dynamics_pane.prediction_reason.as_deref(),
                            )
                        })
                        .inner;
                    if self.dynamics_pane.details_open != details_open {
                        self.dynamics_pane.details_open = details_open;
                        ui.ctx().request_repaint();
                    }
                }
            });
        self.dynamics_pane.painted =
            self.dynamics_pane.frame().is_some() && self.dynamics_pane.plot_bounds.len() == 3;
        Ok(())
    }
}

impl Laboratory {
    fn dynamics_episode_controls(&mut self, ui: &mut egui::Ui, run: &RunIndex) {
        let pane = &mut self.dynamics_pane;
        let revision = pane
            .frame()
            .map_or(self.selection.revision, |frame| frame.state.revision);
        ui.horizontal_wrapped(|ui| {
            ui.small("Episode");
            if let Some(sources) = &pane.sources {
                let selected = sources
                    .sources()
                    .iter()
                    .find(|s| pane.selected_sequence == Some(s.key.sequence));
                let name = selected.map_or("Choose episode".to_owned(), |s| s.id.clone());
                egui::ComboBox::from_id_salt("dynamics-episode")
                    .width(210.)
                    .selected_text(compact_label(&name, 27))
                    .show_ui(ui, |ui| {
                        for source in sources.sources().iter().filter(|s| s.revision == revision) {
                            if ui
                                .selectable_label(
                                    pane.selected_sequence == Some(source.key.sequence),
                                    &source.id,
                                )
                                .clicked()
                            {
                                pane.selected_sequence = Some(source.key.sequence);
                                pane.requested = None;
                                if let Err(error) = crate::pane::SelectionIntent::Input(
                                    SelectedInput::TrajectoryStep {
                                        sequence: source.key.sequence,
                                        step: 0,
                                    },
                                )
                                .apply(run, &mut self.selection)
                                {
                                    self.error = Some(error.to_string());
                                }
                            }
                        }
                    })
                    .response
                    .on_hover_text(name);
            }
            if let Some(frame) = pane.displayed.clone() {
                let episode = &frame.selected.episode;
                let mut action = pane.action.unwrap_or(frame.action);
                ui.small("Action");
                egui::ComboBox::from_id_salt("dynamics-action")
                    .width(125.)
                    .selected_text(compact_label(&action_label(episode, action), 18))
                    .show_ui(ui, |ui| {
                        for &axis in episode.action_axes() {
                            ui.selectable_value(&mut action, axis, action_label(episode, axis));
                        }
                    })
                    .response
                    .on_hover_text(action_label(episode, action));
                if pane.action != Some(action) {
                    pane.action = Some(action);
                    pane.requested = None;
                }
                let total = episode.transitions();
                let step = match self.selection.input {
                    SelectedInput::TrajectoryStep { sequence, step }
                        if sequence == frame.selected.key.sequence =>
                    {
                        step.min(total)
                    }
                    _ => 0,
                };
                let mut next = step;
                ui.small(format!("Decision {step}/{total}"));
                if ui.button("Previous step").clicked() {
                    next = next.saturating_sub(1);
                }
                if ui.button("Next step").clicked() {
                    next = (next + 1).min(total);
                }
                ui.add(egui::Slider::new(&mut next, 0..=total).show_value(false));
                if ui.button("Final state").clicked() {
                    next = total;
                }
                if next != step
                    && let Err(error) =
                        crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                            sequence: frame.selected.key.sequence,
                            step: next,
                        })
                        .apply(run, &mut self.selection)
                {
                    self.error = Some(error.to_string());
                }
            }
        });
    }
    fn dynamics_controls(&mut self, ui: &mut egui::Ui, run: &RunIndex) {
        let pane = &mut self.dynamics_pane;
        ui.set_width(204.);
        if let Some(error) = pane.invalidated.as_deref().or_else(|| pane.worker.error()) {
            ui.colored_label(crate::theme::WARNING, error);
        } else if pane
            .catalog
            .as_ref()
            .is_some_and(TrajectoryCatalog::pending)
        {
            let (done, total) = pane.catalog.as_ref().unwrap().progress();
            ui.weak(format!("Checking evaluations {done}/{total}"));
        } else if pane.sources.is_none() {
            ui.weak("No compatible unconditional evaluation trajectories");
        }
        ui.separator();
        label(
            ui,
            "Source",
            "Occupancy and coverage use only compatible recorded evaluation trajectories. Training transitions are unavailable in this recording.",
        );
        egui::ComboBox::from_id_salt("dynamics-source")
            .width(195.)
            .selected_text("Evaluation")
            .show_ui(ui, |ui| {
                let _ = ui.selectable_label(true, "Evaluation");
                ui.add_enabled_ui(false, |ui| {
                    ui.selectable_label(false, "Training experience")
                        .on_hover_text("Training transition history was not recorded.");
                });
            });
        if let Some(catalog) = &pane.catalog {
            if catalog
                .entries()
                .iter()
                .any(|s| !s.protocol.unconditional())
            {
                egui::ComboBox::from_id_salt("dynamics-conditional-episode")
                    .width(195.)
                    .selected_text("Inspect conditional episode")
                    .show_ui(ui, |ui| {
                        for source in catalog
                            .entries()
                            .iter()
                            .filter(|s| !s.protocol.unconditional())
                        {
                            if ui
                                .selectable_label(
                                    false,
                                    format!(
                                        "{} · k={} · #{}",
                                        source.id, source.revision, source.key.sequence
                                    ),
                                )
                                .clicked()
                                && let Err(error) = crate::pane::SelectionIntent::Input(
                                    SelectedInput::TrajectoryStep {
                                        sequence: source.key.sequence,
                                        step: 0,
                                    },
                                )
                                .apply(run, &mut self.selection)
                            {
                                self.error = Some(error.to_string());
                            }
                        }
                    });
            }
            let groups: Vec<_> = catalog
                .groups()
                .iter()
                .filter(|g| g.protocol.unconditional())
                .collect();
            if groups.len() > 1 {
                let current = pane
                    .protocol
                    .as_ref()
                    .map_or("Choose protocol", ProtocolKey::caption);
                label(
                    ui,
                    "Protocol",
                    "Only identical evaluation protocols and roles are pooled. No paired evaluation claim is implied.",
                );
                egui::ComboBox::from_id_salt("dynamics-protocol")
                    .width(195.)
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (i, group) in groups.iter().enumerate() {
                            let name = format!("{} · {}", group.protocol.caption(), i + 1);
                            if ui
                                .selectable_label(
                                    pane.protocol.as_ref() == Some(&group.protocol),
                                    name,
                                )
                                .clicked()
                            {
                                if let Some(source) = catalog
                                    .entries()
                                    .iter()
                                    .rev()
                                    .find(|source| source.protocol == group.protocol)
                                    && let Err(error) = crate::pane::SelectionIntent::Input(
                                        SelectedInput::TrajectoryStep {
                                            sequence: source.key.sequence,
                                            step: 0,
                                        },
                                    )
                                    .apply(run, &mut self.selection)
                                {
                                    self.error = Some(error.to_string());
                                    continue;
                                }
                                pane.protocol = Some(group.protocol.clone());
                                pane.worker.retire();
                                pane.sources = None;
                                pane.reference = None;
                                pane.requested = None;
                                pane.published = None;
                                pane.displayed = None;
                                pane.selected_sequence = None;
                                pane.initial_revision = false;
                                pane.texture_state = None;
                                pane.occupancy_texture = None;
                                pane.coverage_texture = None;
                                pane.domain = DomainSpec::automatic(pane.domain_epoch);
                                pane.bounds_edit = Default::default();
                                pane.bounds_error = None;
                            }
                        }
                    });
            }
        }
        let frame = pane.displayed.clone();
        if let Some(frame) = frame {
            let coords = frame.selected.episode.coordinates();
            let choices = frame.selected.episode.numeric_coordinates();
            let mut axes = pane.axes.unwrap();
            label(
                ui,
                "Axes",
                "Numerical observation coordinates. This is a marginal projection of observed decision states.",
            );
            for name in ["X", "Y"] {
                let mut axis = if name == "X" { axes.x } else { axes.y };
                let other = if name == "X" { axes.y } else { axes.x };
                ui.horizontal(|ui| {
                    ui.label(name);
                    egui::ComboBox::from_id_salt(format!("dynamics-{name}"))
                        .width(156.)
                        .selected_text(compact_label(&coordinate_label(&coords[axis]), 18))
                        .show_ui(ui, |ui| {
                            for &j in choices {
                                if j != other {
                                    ui.selectable_value(&mut axis, j, coordinate_label(&coords[j]));
                                }
                            }
                        })
                        .response
                        .on_hover_text(coordinate_label(&coords[axis]));
                });
                if name == "X" {
                    axes.x = axis
                } else {
                    axes.y = axis
                };
            }
            if pane.axes != Some(axes) {
                pane.axes = Some(axes);
                pane.reset_projection();
                if pane.z.is_some_and(|z| z == axes.x || z == axes.y) {
                    pane.z = choices
                        .iter()
                        .copied()
                        .find(|&j| j != axes.x && j != axes.y);
                }
            }
            if pane.three_d && !pane.state_action {
                if choices.len() < 3 {
                    ui.weak("Three numerical state coordinates required");
                } else {
                    let mut z = pane.z.unwrap_or(choices[2]);
                    ui.horizontal(|ui| {
                        ui.label("Z");
                        egui::ComboBox::from_id_salt("dynamics-z")
                            .width(156.)
                            .selected_text(compact_label(&coordinate_label(&coords[z]), 18))
                            .show_ui(ui, |ui| {
                                for &j in choices {
                                    if j != axes.x && j != axes.y {
                                        ui.selectable_value(
                                            &mut z,
                                            j,
                                            coordinate_label(&coords[j]),
                                        );
                                    }
                                }
                            })
                            .response
                            .on_hover_text(coordinate_label(&coords[z]));
                    });
                    if pane.z != Some(z) {
                        pane.z = Some(z);
                        pane.requested = None;
                    }
                }
            }
        } else {
            ui.label("Axes");
            ui.weak("Waiting for coordinates…");
        }
        egui::CollapsingHeader::new("Plot bounds").show(ui, |ui| {
            for (axis, name) in ["X", "Y"].into_iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(name);
                    for edge in 0..2 {
                        ui.add(
                            egui::TextEdit::singleline(&mut pane.bounds_edit[axis][edge])
                                .desired_width(68.),
                        );
                    }
                });
            }
            ui.horizontal(|ui| {
                if ui.button("Apply").clicked() {
                    let parsed = (|| -> Result<DomainSpec> {
                        let mut bounds = [[0.; 2]; 2];
                        for (axis, range) in bounds.iter_mut().enumerate() {
                            for (edge, value) in range.iter_mut().enumerate() {
                                *value = pane.bounds_edit[axis][edge].parse().map_err(|_| {
                                    rl::Error::Shape("Enter finite numeric bounds".into())
                                })?;
                            }
                        }
                        DomainSpec::manual(bounds)
                    })();
                    match parsed {
                        Ok(domain) => pane.change_domain(domain),
                        Err(error) => pane.bounds_error = Some(error.to_string()),
                    }
                }
                if ui
                    .button("Recompute")
                    .on_hover_text("Recompute from all admitted evaluations")
                    .clicked()
                {
                    pane.domain_epoch = pane.domain_epoch.wrapping_add(1);
                    pane.change_domain(DomainSpec::automatic(pane.domain_epoch));
                    pane.bounds_edit = Default::default();
                }
            });
            if let Some(error) = &pane.bounds_error {
                ui.colored_label(crate::theme::WARNING, error);
            }
        });
        if pane.three_d {
            ui.label("3D projection");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut pane.state_action, false, "State");
                ui.selectable_value(&mut pane.state_action, true, "State–action");
            });
            if let Some(frame) = pane.frame() {
                if frame.selected.episode.numeric_coordinates().len() < 3 {
                    ui.weak("Three numerical state coordinates required");
                }
                if matches!(pane.action, Some(ActionAxis::Discrete)) {
                    ui.weak("Categorical actions have no continuous 3D axis");
                }
            }
        }
        if let Some(frame) = pane.frame() {
            ui.add_space(5.);
            ui.separator();
            ui.small(format!("Selected 2D edges {} · 3D edges {} · reward {} samples · action {} samples · {} transitions", frame.selected.path.segments.len(), frame.geometry3.len(), frame.selected.reward_trace.len(), frame.action_trace.len(), frame.selected.episode.transitions()));
            ui.small("Display traces retain bounded adjacent recorded samples. Omitted samples remain gaps; the cursor reads exact values. State–action has no terminal-state action edge.");
            ui.weak(format!("{} visits · {} excluded",frame.state.occupancy.included,frame.state.occupancy.excluded))
                .on_hover_text("Decision-state visits in the fixed display grid; states outside plotting bounds are excluded.");
            ui.weak(format!("{} / {} context episodes",if frame.projection == Projection::None {frame.state.displayed_episodes} else {frame.background.paths.len()},frame.state.population_episodes))
                .on_hover_text("Context trajectories are deterministically sampled under a display budget; the selected episode is overlaid separately. Population counts use all admitted compatible episodes.");
            if frame.projection == Projection::None {
                ui.weak(format!(
                    "{} segments / {} cap",
                    frame.state.displayed_segments,
                    WorkLimits::default().max_segments
                ));
            } else {
                ui.weak(format!("{} / {} context edges", frame.background.displayed_segments, frame.background.eligible_segments))
                    .on_hover_text("Bounded adjacent edges from the compatible checkpoint population. Omitted edges remain gaps; the selected episode is overlaid in gold.");
            }
            if frame.state.reference.grid.empirical.iter().any(|&v| v) {
                ui.weak("Empirical plotting bounds").on_hover_text("Bounds are pinned to all admitted reference evaluations for this run, protocol and axis selection; terminal observations are included.");
            }
            if let Some(s) = &pane.sources {
                ui.weak(format!("{} / {} sources admitted",s.sources().len(),s.sources().len()+s.omitted_sources()))
                    .on_hover_text("The bounded source set keeps the newest whole episodes that fit its source and decision budgets.");
                if s.omitted_sources() > 0 {
                    ui.weak(format!(
                        "{} episodes excluded by admission",
                        s.omitted_sources()
                    ));
                }
            }
            if let Some(c) = &pane.catalog
                && c.omitted() + c.rejected() > 0
            {
                ui.weak(format!(
                    "{} catalog sources omitted/rejected",
                    c.omitted() + c.rejected()
                ));
            }
            ui.weak("Evaluation trajectories").on_hover_text("Coverage and occupancy come from recorded evaluation rollouts, not training experience or paired comparisons.");
        }
        let _ = run;
    }
}
fn value_text(e: &crate::dynamics::Episode, value: ActionValue) -> String {
    match value {
        ActionValue::Continuous(x) => format!("{x:.3}"),
        ActionValue::Discrete(id) => e
            .discrete_label(id)
            .map_or_else(|| id.to_string(), str::to_owned),
    }
}

fn text(
    p: &egui::Painter,
    pos: Pos2,
    align: egui::Align2,
    s: impl ToString,
    size: f32,
    color: Color32,
) {
    p.text(pos, align, s, egui::FontId::proportional(size), color);
}
fn axes(
    p: &egui::Painter,
    rect: Rect,
    x: [f64; 2],
    y: [f64; 2],
    xlabel: &str,
    ylabel: &str,
    integer: bool,
) {
    p.rect_filled(rect, 2., PLOT);
    let xticks = if integer {
        integer_ticks(x)
    } else {
        (0..=5)
            .map(|i| x[0] + (x[1] - x[0]) * i as f64 / 5.)
            .collect()
    };
    let yticks = if integer {
        integer_ticks(y)
    } else {
        (0..=5)
            .map(|i| y[0] + (y[1] - y[0]) * i as f64 / 5.)
            .collect()
    };
    for value in xticks {
        let f = ((value - x[0]) / (x[1] - x[0])) as f32;
        let xp = egui::lerp(rect.left()..=rect.right(), f);
        p.line_segment(
            [Pos2::new(xp, rect.top()), Pos2::new(xp, rect.bottom())],
            Stroke::new(0.65, crate::theme::GRID),
        );
        text(
            p,
            Pos2::new(xp, rect.bottom() + 5.),
            egui::Align2::CENTER_TOP,
            if integer {
                format!("{value:.0}")
            } else {
                format!("{value:.2}")
            },
            11.,
            crate::theme::MUTED,
        );
    }
    for value in yticks {
        let f = ((value - y[0]) / (y[1] - y[0])) as f32;
        let yp = egui::lerp(rect.bottom()..=rect.top(), f);
        p.line_segment(
            [Pos2::new(rect.left(), yp), Pos2::new(rect.right(), yp)],
            Stroke::new(0.65, crate::theme::GRID),
        );
        text(
            p,
            Pos2::new(rect.left() - 6., yp),
            egui::Align2::RIGHT_CENTER,
            if integer {
                format!("{value:.0}")
            } else {
                format!("{value:.2}")
            },
            11.,
            crate::theme::MUTED,
        );
    }
    p.rect_stroke(
        rect,
        2.,
        Stroke::new(1., Color32::from_rgb(87, 112, 140)),
        egui::StrokeKind::Inside,
    );
    text(
        p,
        Pos2::new(rect.center().x, rect.bottom() + 27.),
        egui::Align2::CENTER_TOP,
        xlabel,
        12.,
        crate::theme::TEXT,
    );
    let galley = p.layout_no_wrap(
        ylabel.to_owned(),
        egui::FontId::proportional(12.),
        crate::theme::TEXT,
    );
    let mut shape = egui::epaint::TextShape::new(
        Pos2::new(rect.left() - 48., rect.center().y + galley.size().x * 0.5),
        galley,
        crate::theme::TEXT,
    );
    shape.angle = -std::f32::consts::FRAC_PI_2;
    p.add(shape);
}
fn integer_ticks([lo, hi]: [f64; 2]) -> Vec<f64> {
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Vec::new();
    }
    let first = lo.ceil();
    let last = hi.floor();
    if first > last {
        return Vec::new();
    }
    let intervals = (last - first).clamp(0., 5.) as usize;
    if intervals == 0 {
        return vec![first];
    }
    let mut values = Vec::with_capacity(intervals + 1);
    for i in 0..=intervals {
        let value = (first + (last - first) * i as f64 / intervals as f64).round();
        if values.last().is_none_or(|&previous| previous != value) {
            values.push(value);
        }
    }
    values
}
fn pos2(rect: Rect, bounds: [[f64; 2]; 2], point: [f64; 2]) -> Pos2 {
    let fx = ((point[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0])) as f32;
    let fy = ((point[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0])) as f32;
    Pos2::new(
        egui::lerp(rect.left()..=rect.right(), fx),
        egui::lerp(rect.bottom()..=rect.top(), fy),
    )
}

/// Crop the existing fixed grid to the displayed selected path; no cells or
/// visit masses are recomputed. Keep some surrounding context on both axes.
fn episode_view_bounds(full: [[f64; 2]; 2], path: &crate::dynamics::work::Path2) -> [[f64; 2]; 2] {
    let mut observed = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
    for point in std::iter::once(path.start)
        .chain(std::iter::once(path.end))
        .chain(path.segments.iter().flat_map(|segment| segment.points))
        .filter(|point| in_bounds2(full, *point))
    {
        for axis in 0..2 {
            observed[axis][0] = observed[axis][0].min(point[axis]);
            observed[axis][1] = observed[axis][1].max(point[axis]);
        }
    }
    let mut view = full;
    for axis in 0..2 {
        let [lo, hi] = observed[axis];
        let width = full[axis][1] - full[axis][0];
        if !lo.is_finite() || !hi.is_finite() || !width.is_finite() || width <= 0. {
            continue;
        }
        let span = ((hi - lo).max(width * 0.15) * 1.3).min(width);
        let start = ((lo + hi) * 0.5 - span * 0.5).clamp(full[axis][0], full[axis][1] - span);
        view[axis] = [start, start + span];
    }
    view
}

fn grid_view_uv(full: [[f64; 2]; 2], view: [[f64; 2]; 2]) -> Rect {
    let x = |v: f64| ((v - full[0][0]) / (full[0][1] - full[0][0])) as f32;
    let y = |v: f64| ((full[1][1] - v) / (full[1][1] - full[1][0])) as f32;
    Rect::from_min_max(
        Pos2::new(x(view[0][0]), y(view[1][1])),
        Pos2::new(x(view[0][1]), y(view[1][0])),
    )
}
fn in_bounds2(bounds: [[f64; 2]; 2], point: [f64; 2]) -> bool {
    (0..2).all(|i| point[i].is_finite() && point[i] >= bounds[i][0] && point[i] <= bounds[i][1])
}
fn in_bounds3(bounds: [[f64; 2]; 3], point: [f64; 3]) -> bool {
    (0..3).all(|i| point[i].is_finite() && point[i] >= bounds[i][0] && point[i] <= bounds[i][1])
}
fn grid_overlay(p: &egui::Painter, rect: Rect) {
    for i in 0..=5 {
        let f = i as f32 / 5.;
        let x = egui::lerp(rect.left()..=rect.right(), f);
        let y = egui::lerp(rect.bottom()..=rect.top(), f);
        p.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(0.65, Color32::from_rgba_unmultiplied(85, 116, 150, 80)),
        );
        p.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(0.65, Color32::from_rgba_unmultiplied(85, 116, 150, 80)),
        );
    }
}
fn direction_arrow(p: &egui::Painter, a: Pos2, b: Pos2, color: Color32) {
    let v = b - a;
    let len = v.length();
    if len < 14. || !len.is_finite() {
        return;
    }
    let direction = v / len;
    let tip = a + v * 0.68;
    let wing = Vec2::new(-direction.y, direction.x);
    p.line_segment(
        [tip, tip - direction * 7. + wing * 3.],
        Stroke::new(1.2, color),
    );
    p.line_segment(
        [tip, tip - direction * 7. - wing * 3.],
        Stroke::new(1.2, color),
    );
}
fn marker(p: &egui::Painter, point: Pos2, r: f32, color: Color32, label: Option<&str>) {
    p.circle_filled(point, r + 3., PLOT);
    p.circle_filled(point, r, color);
    p.circle_stroke(point, r + 1., Stroke::new(1.2, Color32::WHITE));
    if let Some(label) = label {
        text(
            p,
            point + Vec2::new(11., -9.),
            egui::Align2::LEFT_CENTER,
            label,
            12.,
            crate::theme::TEXT,
        );
    }
}
fn draw_segment(p: &egui::Painter, clip: Rect, a: Pos2, b: Pos2, stroke: Stroke) {
    if clip.expand(5.).intersects(Rect::from_two_pos(a, b)) {
        p.line_segment([a, b], stroke);
    }
}

impl Laboratory {
    fn dynamics_main_plot(&mut self, ui: &mut egui::Ui, run: &RunIndex, size: Vec2) {
        let (outer, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        self.dynamics_pane.plot_bounds.push(outer);
        let chart = Rect::from_min_max(
            outer.min + Vec2::new(54., 8.),
            outer.max - Vec2::new(134., 44.),
        );
        let p = ui.painter_at(outer);
        let Some(frame) = self.dynamics_pane.displayed.clone() else {
            p.rect_filled(chart, 2., PLOT);
            text(
                &p,
                chart.center(),
                egui::Align2::CENTER_CENTER,
                self.dynamics_pane
                    .invalidated
                    .as_deref()
                    .or_else(|| self.dynamics_pane.worker.error())
                    .unwrap_or(
                        if self
                            .dynamics_pane
                            .catalog
                            .as_ref()
                            .is_some_and(|catalog| !catalog.pending())
                            && self.dynamics_pane.sources.is_none()
                        {
                            "No compatible unconditional evaluation trajectories"
                        } else {
                            "Preparing spatial view…"
                        },
                    ),
                14.,
                crate::theme::MUTED,
            );
            return;
        };
        let episode = &frame.selected.episode;
        let full_bounds = frame.state.reference.grid.grid.bounds();
        let coords = episode.coordinates();
        if self.dynamics_pane.requested != self.dynamics_pane.published {
            text(
                &p,
                outer.right_top() - Vec2::new(7., 0.),
                egui::Align2::RIGHT_TOP,
                "Preparing selection…",
                11.,
                crate::theme::WARNING,
            );
        }
        if frame.projection != Projection::None
            || (self.dynamics_pane.three_d
                && self.dynamics_pane.requested == self.dynamics_pane.published)
        {
            self.dynamics_pane.worker.cancel_hit();
            self.dynamics_pane.hit_view = None;
            self.dynamics_3d(&p, chart, &response, &frame, run);
            return;
        }
        let bounds = if self.dynamics_pane.fit_episode {
            episode_view_bounds(full_bounds, &frame.selected.path)
        } else {
            full_bounds
        };
        let uv = grid_view_uv(full_bounds, bounds);
        let shown_axes = frame.state.key.axes();
        let xlabel = coordinate_label(&coords[shown_axes.x]);
        let ylabel = coordinate_label(&coords[shown_axes.y]);
        axes(&p, chart, bounds[0], bounds[1], &xlabel, &ylabel, false);
        match self.dynamics_pane.mode {
            DisplayMode::Occupancy => {
                if let Some(tex) = &self.dynamics_pane.occupancy_texture {
                    p.image(tex.id(), chart, uv, Color32::WHITE);
                }
            }
            DisplayMode::Coverage => {
                if let Some(tex) = &self.dynamics_pane.coverage_texture {
                    p.image(tex.id(), chart, uv, Color32::WHITE);
                }
            }
            DisplayMode::Flow => {}
        }
        grid_overlay(&p, chart);
        let geometry = p.with_clip_rect(chart);
        let faint = if self.dynamics_pane.mode == DisplayMode::Flow {
            Color32::from_rgba_unmultiplied(68, 161, 246, 110)
        } else {
            Color32::from_rgba_unmultiplied(79, 154, 232, 62)
        };
        let arrow_stride = (frame.state.displayed_segments / 48).max(1);
        let mut arrow_index = 0usize;
        for path in &frame.state.trajectories {
            for s in &path.segments {
                if in_bounds2(full_bounds, s.points[0]) && in_bounds2(full_bounds, s.points[1]) {
                    let a = pos2(chart, bounds, s.points[0]);
                    let b = pos2(chart, bounds, s.points[1]);
                    draw_segment(&geometry, chart, a, b, Stroke::new(1., faint));
                    if self.dynamics_pane.mode == DisplayMode::Flow
                        && arrow_index.is_multiple_of(arrow_stride)
                    {
                        direction_arrow(&geometry, a, b, faint);
                    }
                }
                arrow_index += 1;
            }
        }
        if let Some(path) = &self.dynamics_pane.benchmark.path {
            let stroke = Stroke::new(1.8, crate::theme::REFERENCE);
            for segment in path.points.windows(2) {
                if in_bounds2(full_bounds, segment[0]) && in_bounds2(full_bounds, segment[1]) {
                    draw_segment(
                        &geometry,
                        chart,
                        pos2(chart, bounds, segment[0]),
                        pos2(chart, bounds, segment[1]),
                        stroke,
                    );
                }
            }
        }
        let step = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step }
                if sequence == frame.selected.key.sequence =>
            {
                step.min(episode.transitions())
            }
            _ => 0,
        };
        let selected_stride = (frame.selected.path.segments.len() / 14).max(1);
        for (index, s) in frame.selected.path.segments.iter().enumerate() {
            let color = if s.t <= step {
                CYAN
            } else {
                CYAN.gamma_multiply(0.75)
            };
            if in_bounds2(full_bounds, s.points[0]) && in_bounds2(full_bounds, s.points[1]) {
                let a = pos2(chart, bounds, s.points[0]);
                let b = pos2(chart, bounds, s.points[1]);
                draw_segment(&geometry, chart, a, b, Stroke::new(2.4, color));
                if index.is_multiple_of(selected_stride) {
                    direction_arrow(&geometry, a, b, color);
                }
            }
        }
        let current = episode
            .project(step, shown_axes)
            .ok()
            .filter(|&point| in_bounds2(bounds, point))
            .map(|point| pos2(chart, bounds, point));
        if in_bounds2(bounds, frame.selected.path.start) {
            let start = pos2(chart, bounds, frame.selected.path.start);
            marker(
                &p,
                start,
                5.,
                Color32::from_rgb(75, 218, 147),
                current
                    .is_none_or(|point| point.distance(start) > 45.)
                    .then_some("Start"),
            );
        }
        if in_bounds2(bounds, frame.selected.path.end) {
            let end = pos2(chart, bounds, frame.selected.path.end);
            marker(
                &p,
                end,
                4.,
                Color32::from_rgb(219, 226, 236),
                current
                    .is_none_or(|point| point.distance(end) > 45.)
                    .then_some("End"),
            );
        }
        if let Some(current) = current {
            marker(
                &p,
                current,
                6.,
                crate::theme::SELECTED,
                Some(&format!("t = {step}")),
            );
        }
        let legend = match self.dynamics_pane.mode {
            DisplayMode::Occupancy => "Visit mass (log)",
            DisplayMode::Flow => "Observed edges",
            DisplayMode::Coverage => "First seen k",
        };
        text(
            &p,
            Pos2::new(chart.right() + 8., chart.top() + 4.),
            egui::Align2::LEFT_TOP,
            legend,
            11.,
            crate::theme::MUTED,
        );
        if self.dynamics_pane.mode != DisplayMode::Flow {
            for i in 0..40 {
                let t = i as f32 / 40.;
                let r = Rect::from_min_max(
                    Pos2::new(chart.right() + 9., chart.top() + 36. + t * 130.),
                    Pos2::new(
                        chart.right() + 21.,
                        chart.top() + 36. + (t + 1. / 40.) * 130.,
                    ),
                );
                p.rect_filled(r, 0., crate::landscape::sequential_color(1. - t));
            }
            let labels: Vec<String> = match self.dynamics_pane.mode {
                DisplayMode::Occupancy => vec![
                    "1".into(),
                    "1e-1".into(),
                    "1e-2".into(),
                    "1e-3".into(),
                    "<=1e-4".into(),
                ],
                DisplayMode::Coverage => {
                    let points = &frame.state.coverage.points;
                    vec![
                        points.last().map_or("—".into(), |p| p.revision.to_string()),
                        "".into(),
                        "".into(),
                        "".into(),
                        points
                            .first()
                            .map_or("—".into(), |p| p.revision.to_string()),
                    ]
                }
                DisplayMode::Flow => Vec::new(),
            };
            for (i, label) in labels.iter().enumerate() {
                if !label.is_empty() {
                    text(
                        &p,
                        Pos2::new(chart.right() + 27., chart.top() + 36. + i as f32 * 32.5),
                        egui::Align2::LEFT_CENTER,
                        label,
                        10.,
                        crate::theme::TEXT,
                    );
                }
            }
            if self.dynamics_pane.mode == DisplayMode::Occupancy {
                text(
                    &p,
                    Pos2::new(chart.right() + 9., chart.top() + 178.),
                    egui::Align2::LEFT_TOP,
                    "empty = 0",
                    10.,
                    crate::theme::MUTED,
                );
            }
        }
        if frame.state.occupancy.excluded > 0 {
            text(
                &p,
                Pos2::new(chart.right() + 8., chart.top() + 210.),
                egui::Align2::LEFT_TOP,
                format!("{} outside grid", frame.state.occupancy.excluded),
                10.,
                crate::theme::WARNING,
            );
        }
        let view = HitView {
            axes: shown_axes,
            bounds,
            chart,
        };
        let identity = (Arc::as_ptr(&frame) as usize, view);
        let pane = &mut self.dynamics_pane;
        let current = pane.requested == pane.published
            && pane.invalidated.is_none()
            && pane.catalog.as_ref().is_some_and(|catalog| {
                !catalog.pending() && catalog.generation() == frame.state.key.generation()
            });
        if pane.hit_view != Some(identity)
            || pane
                .hit_selection
                .as_ref()
                .is_some_and(|input| input != &self.selection.input)
            || !current
        {
            pane.worker.cancel_hit();
            pane.hit_view = Some(identity);
            pane.hit_selection = None;
        }
        if response.clicked()
            && current
            && let Some(pointer) = response.interact_pointer_pos()
            && let Some(sources) = &pane.sources
        {
            if pane
                .worker
                .request_hit(&frame, sources.clone(), view, pointer)
                .is_ok()
            {
                pane.hit_selection = Some(self.selection.input.clone());
            }
        } else if current
            && let Some(step) = pane.worker.poll_hit(&frame, view)
            && let Err(error) = crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                sequence: frame.selected.key.sequence,
                step,
            })
            .apply(run, &mut self.selection)
        {
            self.error = Some(error.to_string());
        }
    }
    fn dynamics_3d(
        &mut self,
        p: &egui::Painter,
        chart: Rect,
        response: &egui::Response,
        frame: &PreparedFrame,
        run: &RunIndex,
    ) {
        let pane = &mut self.dynamics_pane;
        if response.dragged() {
            let delta = response.drag_delta();
            pane.yaw += delta.x * 0.008;
            pane.pitch = (pane.pitch + delta.y * 0.008).clamp(-1.2, 1.2);
        }
        p.rect_filled(chart, 2., PLOT);
        text(
            p,
            chart.left_top() + Vec2::new(12., 10.),
            egui::Align2::LEFT_TOP,
            "Drag to rotate",
            11.,
            crate::theme::MUTED,
        );
        let projection = frame.projection;
        if projection == Projection::None {
            text(
                p,
                chart.center(),
                egui::Align2::CENTER_CENTER,
                if matches!(frame.action, ActionAxis::Discrete) {
                    "Categorical actions have no continuous 3D axis"
                } else {
                    "Three numerical state coordinates required"
                },
                14.,
                crate::theme::MUTED,
            );
            return;
        }
        let mut bounds = frame.background.bounds;
        if frame.projection == Projection::StateAction
            && let (ActionSpace::Box { low, high }, ActionAxis::Continuous(j)) =
                (&run.recording().manifest().action_space, frame.action)
        {
            bounds[2] = expand([low[j] as f64, high[j] as f64]);
        }
        let coords = frame.selected.episode.coordinates();
        let a = frame.state.key.axes();
        let zlabel = match projection {
            Projection::State([_, _, z]) => coordinate_label(&coords[z]),
            Projection::StateAction => action_label(&frame.selected.episode, frame.action),
            Projection::None => String::new(),
        };
        let names = [
            coordinate_label(&coords[a.x]),
            coordinate_label(&coords[a.y]),
            zlabel.clone(),
        ];
        let geometry = p.with_clip_rect(chart);
        let project = |v: [f64; 3]| -> Pos2 {
            let n = std::array::from_fn::<_, 3, _>(|i| {
                ((v[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) - 0.5) as f32
            });
            let (sy, cy) = pane.yaw.sin_cos();
            let (sp, cp) = pane.pitch.sin_cos();
            let x = n[0] * cy - n[1] * sy;
            let depth = n[0] * sy + n[1] * cy;
            let y = n[2] * cp - depth * sp;
            Pos2::new(
                chart.center().x + x * chart.width() * 0.66,
                chart.center().y - y * chart.height() * 0.7,
            )
        };
        for a in 0..3 {
            let origin = [bounds[0][0], bounds[1][0], bounds[2][0]];
            let mut end = origin;
            end[a] = bounds[a][1];
            geometry.line_segment(
                [project(origin), project(end)],
                Stroke::new(1., crate::theme::MUTED),
            );
            let axis = if a == 2 && matches!(projection, Projection::StateAction) {
                "Action"
            } else {
                ["X", "Y", "Z"][a]
            };
            text(
                p,
                project(end) + Vec2::new(6., -5.),
                egui::Align2::LEFT_BOTTOM,
                format!("{axis}: {}", compact_label(&names[a], 20)),
                11.,
                crate::theme::TEXT,
            );
        }
        let mut excluded = false;
        for path in &frame.background.paths {
            for segment in &path.segments {
                if in_bounds3(bounds, segment.points[0]) && in_bounds3(bounds, segment.points[1]) {
                    draw_segment(
                        &geometry,
                        chart,
                        project(segment.points[0]),
                        project(segment.points[1]),
                        Stroke::new(0.8, CYAN.gamma_multiply(0.28)),
                    );
                } else {
                    excluded = true;
                }
            }
            // One-decision state-action episodes have a point, never a fabricated edge.
            if path.segments.is_empty() && in_bounds3(bounds, path.start) {
                geometry.circle_filled(project(path.start), 1.8, CYAN.gamma_multiply(0.35));
            }
        }
        for segment in &frame.geometry3 {
            if in_bounds3(bounds, segment.points[0]) && in_bounds3(bounds, segment.points[1]) {
                draw_segment(
                    &geometry,
                    chart,
                    project(segment.points[0]),
                    project(segment.points[1]),
                    Stroke::new(2., CYAN),
                );
            } else {
                excluded = true;
            }
        }
        let step = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step }
                if sequence == frame.selected.key.sequence =>
            {
                step.min(frame.selected.episode.transitions())
            }
            _ => 0,
        };
        let at = |t| match projection {
            Projection::State([x, y, z]) => {
                frame.selected.episode.project_state3(t, [x, y, z]).ok()
            }
            Projection::StateAction => frame
                .selected
                .episode
                .project_state_action(t, frame.state.key.axes(), frame.action)
                .ok()
                .flatten(),
            Projection::None => None,
        };
        let current = at(step).filter(|&point| in_bounds3(bounds, point));
        let current_screen = current.map(project);
        if let Some(start) = at(0).filter(|&point| in_bounds3(bounds, point)) {
            let start = project(start);
            marker(
                &geometry,
                start,
                4.5,
                Color32::from_rgb(75, 218, 147),
                current_screen
                    .is_none_or(|point| point.distance(start) > 45.)
                    .then_some("Start"),
            );
        }
        let last = if matches!(projection, Projection::StateAction) {
            frame.selected.episode.transitions().saturating_sub(1)
        } else {
            frame.selected.episode.transitions()
        };
        if let Some(end) = at(last).filter(|&point| in_bounds3(bounds, point)) {
            let end = project(end);
            marker(
                &geometry,
                end,
                4.,
                Color32::from_rgb(219, 226, 236),
                current_screen
                    .is_none_or(|point| point.distance(end) > 45.)
                    .then_some(if matches!(projection, Projection::StateAction) {
                        "Last decision"
                    } else {
                        "End"
                    }),
            );
        }
        if let Some(point) = current {
            marker(
                &geometry,
                project(point),
                6.,
                crate::theme::SELECTED,
                Some(&format!("t = {step}")),
            );
        } else if step == frame.selected.episode.transitions() {
            text(
                p,
                chart.right_bottom() - Vec2::new(8., 8.),
                egui::Align2::RIGHT_BOTTOM,
                "Terminal state · no action coordinate",
                12.,
                crate::theme::MUTED,
            );
        }
        if excluded {
            text(
                p,
                chart.right_top() - Vec2::new(8., -8.),
                egui::Align2::RIGHT_TOP,
                "Out-of-range 3D edges hidden",
                11.,
                crate::theme::WARNING,
            );
        }
        text(
            p,
            chart.left_bottom() + Vec2::new(12., -12.),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "X: {} · Y: {} · {}: {}",
                names[0],
                names[1],
                if matches!(projection, Projection::StateAction) {
                    "Action"
                } else {
                    "Z"
                },
                zlabel
            ),
            12.,
            crate::theme::TEXT,
        );
        if matches!(projection, Projection::StateAction)
            && matches!(
                run.recording().manifest().action_space,
                ActionSpace::Unbounded { .. }
            )
        {
            text(
                p,
                chart.left_top() + Vec2::new(12., 28.),
                egui::Align2::LEFT_TOP,
                "Frozen empirical action range",
                11.,
                crate::theme::MUTED,
            );
        }
        if let Projection::State([_, _, z]) = projection
            && frame.selected.episode.coordinates()[z].range.is_none()
        {
            text(
                p,
                chart.left_top() + Vec2::new(12., 28.),
                egui::Align2::LEFT_TOP,
                "Frozen empirical Z range",
                11.,
                crate::theme::MUTED,
            );
        }
    }
}

impl Laboratory {
    fn dynamics_timeline_plot(&mut self, ui: &mut egui::Ui, run: &RunIndex, size: Vec2) {
        let (outer, response) = ui.allocate_exact_size(size, Sense::click());
        self.dynamics_pane.plot_bounds.push(outer);
        let plot = Rect::from_min_max(
            outer.min + Vec2::new(52., 18.),
            outer.max - Vec2::new(56., 37.),
        );
        let p = ui.painter_at(outer);
        p.rect_filled(plot, 2., PLOT);
        let Some(frame) = self
            .dynamics_pane
            .displayed
            .clone()
            .map(decisions::TimelineFrame::Spatial)
            .or_else(|| {
                self.dynamics_pane
                    .worker
                    .episode_frame()
                    .cloned()
                    .map(decisions::TimelineFrame::Generic)
            })
        else {
            text(
                &p,
                plot.center(),
                egui::Align2::CENTER_CENTER,
                "Preparing episode timeline…",
                12.,
                crate::theme::MUTED,
            );
            return;
        };
        let total = frame.episode().transitions().max(1);
        let step = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step }
                if sequence == frame.key().sequence =>
            {
                step.min(total)
            }
            _ => 0,
        };
        let reward = frame.reward_trace();
        let action = frame.action_trace();
        let terminal = frame.episode().step(total - 1).ok().and_then(|last| {
            (last.terminated == Some(true) || last.truncated == Some(true))
                .then_some(last.reward)
                .flatten()
                .map(|value| (total - 1, value))
        });
        let mut ordinary_range = [f64::INFINITY, f64::NEG_INFINITY];
        let mut ordinary_count = 0;
        for sample in reward {
            if terminal.is_some_and(|(step, _)| sample[0] == step as f64) {
                continue;
            }
            if sample[1].is_finite() {
                ordinary_range = [
                    ordinary_range[0].min(sample[1]),
                    ordinary_range[1].max(sample[1]),
                ];
                ordinary_count += 1;
            }
        }
        let separate_terminal = terminal.filter(|(_, value)| {
            if ordinary_count < 3 || !value.is_finite() {
                return false;
            }
            let span = (ordinary_range[1] - ordinary_range[0]).max(0.01);
            *value < ordinary_range[0] - 8. * span || *value > ordinary_range[1] + 8. * span
        });
        let mut reward_range = reward
            .iter()
            .filter(|sample| !separate_terminal.is_some_and(|(step, _)| sample[0] == step as f64))
            .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], v| {
                [lo.min(v[1]), hi.max(v[1])]
            });
        if let Ok(exact) = frame.episode().step(step)
            && let Some(value) = exact.reward
            && separate_terminal.is_none_or(|(terminal_step, _)| step != terminal_step)
        {
            reward_range = [reward_range[0].min(value), reward_range[1].max(value)];
        }
        let reward_range = expand(reward_range);
        let mut action_range =
            action
                .iter()
                .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], v| {
                    let x = match v.value {
                        ActionValue::Continuous(x) => x,
                        ActionValue::Discrete(id) => id as f64,
                    };
                    [lo.min(x), hi.max(x)]
                });
        if let Ok(Some(value)) = frame.episode().action_value(step, frame.action()) {
            let x = match value {
                ActionValue::Continuous(x) => x,
                ActionValue::Discrete(id) => id as f64,
            };
            action_range = [action_range[0].min(x), action_range[1].max(x)];
        }
        let action_range = expand(action_range);
        let action_color = crate::theme::CURRENT;
        let map = |t: f64, v: f64, range: [f64; 2]| {
            Pos2::new(
                egui::lerp(plot.left()..=plot.right(), (t / total as f64) as f32),
                egui::lerp(
                    plot.bottom()..=plot.top(),
                    ((v - range[0]) / (range[1] - range[0])) as f32,
                ),
            )
        };
        for i in 0..=4 {
            let f = i as f32 / 4.;
            let x = egui::lerp(plot.left()..=plot.right(), f);
            let y = egui::lerp(plot.bottom()..=plot.top(), f);
            p.line_segment(
                [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
                Stroke::new(0.6, crate::theme::GRID),
            );
            p.line_segment(
                [Pos2::new(plot.left(), y), Pos2::new(plot.right(), y)],
                Stroke::new(0.6, crate::theme::GRID),
            );
            text(
                &p,
                Pos2::new(x, plot.bottom() + 5.),
                egui::Align2::CENTER_TOP,
                format!("{}", (total as f32 * f).round() as usize),
                11.,
                crate::theme::MUTED,
            );
            text(
                &p,
                Pos2::new(plot.left() - 5., y),
                egui::Align2::RIGHT_CENTER,
                format!(
                    "{:.2}",
                    reward_range[0] + (reward_range[1] - reward_range[0]) * f as f64
                ),
                11.,
                GOLD,
            );
            text(
                &p,
                Pos2::new(plot.right() + 5., y),
                egui::Align2::LEFT_CENTER,
                format!(
                    "{:.2}",
                    action_range[0] + (action_range[1] - action_range[0]) * f as f64
                ),
                11.,
                action_color,
            );
        }
        p.rect_stroke(
            plot,
            2.,
            Stroke::new(1., crate::theme::GRID),
            egui::StrokeKind::Inside,
        );
        let geometry = p.with_clip_rect(plot);
        for pair in reward.windows(2) {
            if pair[1][0] - pair[0][0] == 1.
                && !separate_terminal.is_some_and(|(step, _)| pair[1][0] == step as f64)
            {
                draw_segment(
                    &geometry,
                    plot,
                    map(pair[0][0], pair[0][1], reward_range),
                    map(pair[1][0], pair[1][1], reward_range),
                    Stroke::new(1.7, GOLD),
                );
            }
        }
        for item in reward {
            if !separate_terminal.is_some_and(|(step, _)| item[0] == step as f64) {
                geometry.circle_filled(map(item[0], item[1], reward_range), 1.7, GOLD);
            }
        }
        if let Some((terminal_step, value)) = separate_terminal {
            let x = map(terminal_step as f64, reward_range[0], reward_range).x;
            let y = if value > reward_range[1] {
                plot.top() + 2.
            } else {
                plot.bottom() - 2.
            };
            geometry.circle_stroke(Pos2::new(x, y), 4., Stroke::new(1.5, GOLD));
        }
        for pair in action.windows(2) {
            if pair[1].t == pair[0].t + 1 {
                let a = match pair[0].value {
                    ActionValue::Continuous(x) => x,
                    ActionValue::Discrete(id) => id as f64,
                };
                let b = match pair[1].value {
                    ActionValue::Continuous(x) => x,
                    ActionValue::Discrete(id) => id as f64,
                };
                if matches!(frame.action(), ActionAxis::Discrete) {
                    let elbow = map(pair[1].t as f64, a, action_range);
                    draw_segment(
                        &geometry,
                        plot,
                        map(pair[0].t as f64, a, action_range),
                        elbow,
                        Stroke::new(1.2, action_color),
                    );
                    draw_segment(
                        &geometry,
                        plot,
                        elbow,
                        map(pair[1].t as f64, b, action_range),
                        Stroke::new(1.2, action_color),
                    );
                } else {
                    draw_segment(
                        &geometry,
                        plot,
                        map(pair[0].t as f64, a, action_range),
                        map(pair[1].t as f64, b, action_range),
                        Stroke::new(1.2, action_color),
                    );
                }
            }
        }
        for item in action {
            let v = match item.value {
                ActionValue::Continuous(x) => x,
                ActionValue::Discrete(id) => id as f64,
            };
            geometry.rect_filled(
                Rect::from_center_size(map(item.t as f64, v, action_range), Vec2::splat(4.)),
                0.,
                action_color,
            );
        }
        let x = map(step as f64, reward_range[0], reward_range).x;
        p.line_segment(
            [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
            Stroke::new(1., Color32::LIGHT_GRAY),
        );
        if let Ok(exact) = frame.episode().step(step)
            && let Some(reward) = exact.reward
            && separate_terminal.is_none_or(|(terminal_step, _)| step != terminal_step)
        {
            marker(
                &geometry,
                map(step as f64, reward, reward_range),
                3.8,
                GOLD,
                None,
            );
        }
        if let Ok(Some(value)) = frame.episode().action_value(step, frame.action()) {
            let v = match value {
                ActionValue::Continuous(x) => x,
                ActionValue::Discrete(id) => id as f64,
            };
            marker(
                &geometry,
                map(step as f64, v, action_range),
                3.8,
                action_color,
                None,
            );
        }
        text(
            &p,
            outer.left_top() + Vec2::new(2., 0.),
            egui::Align2::LEFT_TOP,
            separate_terminal.map_or_else(
                || "Reward · dots".into(),
                |(_, value)| {
                    format!(
                        "Reward · terminal {} outside scale",
                        crate::landscape::statistic_number(value)
                    )
                },
            ),
            11.,
            GOLD,
        );
        text(
            &p,
            outer.right_top() - Vec2::new(2., 0.),
            egui::Align2::RIGHT_TOP,
            format!(
                "{} · squares",
                action_label(frame.episode(), frame.action())
            ),
            11.,
            action_color,
        );
        text(
            &p,
            Pos2::new(plot.center().x, outer.bottom() - 2.),
            egui::Align2::CENTER_BOTTOM,
            "Decision step",
            11.,
            crate::theme::TEXT,
        );
        if response.clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let t = ((pointer.x - plot.left()) / plot.width() * total as f32)
                .round()
                .clamp(0., total as f32) as usize;
            if let Err(error) = crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                sequence: frame.key().sequence,
                step: t,
            })
            .apply(run, &mut self.selection)
            {
                self.error = Some(error.to_string());
            }
        }
        if response.hovered()
            && let Some(pointer) = response.hover_pos()
        {
            let t = ((pointer.x - plot.left()) / plot.width() * total as f32)
                .round()
                .clamp(0., total as f32) as usize;
            let detail = if t == total {
                format!("t={t}: final state; no next action or reward")
            } else {
                let value = frame
                    .episode()
                    .action_value(t, frame.action())
                    .ok()
                    .flatten()
                    .map_or("—".into(), |v| value_text(frame.episode(), v));
                let reward = frame
                    .episode()
                    .step(t)
                    .ok()
                    .and_then(|s| s.reward)
                    .unwrap_or(0.);
                format!(
                    "t={t} · reward {reward:.3} · {} {value}",
                    action_label(frame.episode(), frame.action())
                )
            };
            response.on_hover_text(detail);
        }
    }
    fn dynamics_coverage_plot(&mut self, ui: &mut egui::Ui, size: Vec2) {
        let (outer, response) = ui.allocate_exact_size(size, Sense::hover());
        self.dynamics_pane.plot_bounds.push(outer);
        let plot = Rect::from_min_max(
            outer.min + Vec2::new(48., 18.),
            outer.max - Vec2::new(14., 43.),
        );
        let p = ui.painter_at(outer);
        p.rect_filled(plot, 2., PLOT);
        let Some(frame) = self.dynamics_pane.frame() else {
            text(
                &p,
                plot.center(),
                egui::Align2::CENTER_CENTER,
                "Preparing coverage…",
                12.,
                crate::theme::MUTED,
            );
            return;
        };
        let coverage = &frame.state.coverage;
        let mode = self.dynamics_pane.coverage_mode;
        if coverage.points.is_empty() {
            text(
                &p,
                plot.center(),
                egui::Align2::CENTER_CENTER,
                "No compatible evaluation history",
                12.,
                crate::theme::MUTED,
            );
            return;
        }
        let start = coverage.points.first().unwrap().revision as f64;
        let end = coverage.points.last().unwrap().revision as f64;
        let xr = expand([start, end]);
        let yr = [
            0.,
            coverage
                .points
                .iter()
                .map(|p| mode.value(p))
                .max()
                .unwrap_or(1)
                .max(1) as f64,
        ];
        axes(&p, plot, xr, yr, "Learner revision", "Observed cells", true);
        let b = [xr, yr];
        for [a, b] in mode.edges(&coverage.points) {
            draw_segment(
                &p,
                plot,
                pos2(plot, [xr, yr], a),
                pos2(plot, [xr, yr], b),
                Stroke::new(2., CYAN),
            );
        }
        for point in &coverage.points {
            p.circle_filled(
                pos2(plot, b, [point.revision as f64, mode.value(point) as f64]),
                2.6,
                CYAN,
            );
        }
        let k = frame.state.revision;
        if let Some(point) = coverage.points.iter().find(|point| point.revision == k) {
            let at = pos2(plot, b, [k as f64, mode.value(point) as f64]);
            p.line_segment(
                [Pos2::new(at.x, plot.top()), Pos2::new(at.x, plot.bottom())],
                Stroke::new(1., GOLD),
            );
            marker(&p, at, 4.5, GOLD, None);
            let percent = 100. * mode.value(point) as f64 / coverage.display_grid_cells as f64;
            text(
                &p,
                outer.right_top() - Vec2::new(4., 0.),
                egui::Align2::RIGHT_TOP,
                format!(
                    "{} / {} cells ({percent:.1}% grid)",
                    mode.value(point),
                    coverage.display_grid_cells
                ),
                11.,
                CYAN,
            );
        }
        let unequal = coverage
            .points
            .windows(2)
            .any(|p| (p[0].episodes, p[0].decisions) != (p[1].episodes, p[1].decisions));
        text(
            &p,
            outer.left_top(),
            egui::Align2::LEFT_TOP,
            if unequal {
                "Unequal evaluation budgets"
            } else {
                "Evaluations"
            },
            11.,
            crate::theme::MUTED,
        );
        let nearest = response.hover_pos().and_then(|pointer| {
            coverage.points.iter().min_by(|a, b| {
                let distance = |point: &crate::dynamics::history::CoveragePoint| {
                    (pos2(plot, [xr, yr], [point.revision as f64, 0.]).x - pointer.x).abs()
                };
                distance(a).total_cmp(&distance(b))
            })
        });
        let mut help = "Observed decision-state cells on a fixed display grid; not training experience or reachable-state coverage.".to_owned();
        if let Some(point) = nearest {
            help.push_str(&format!("\nRevision {}: {} episodes, {} decisions; {} included, {} excluded.\nCurrent {} · New {} · Cumulative {} cells", point.revision, point.episodes, point.decisions, point.included, point.excluded, point.occupied, point.new_cells, point.cumulative));
        }
        response.on_hover_text(help);
    }
}
#[cfg(test)]
mod domain_tests {
    use super::*;
    use crate::data::Loader;
    use crate::dynamics::work::availability_fixture as domain_test_fixture;
    use rl::diagnostics::storage::ReadLimits;
    use std::{sync::mpsc, time::Duration};

    #[test]
    fn episode_framing_crops_the_existing_grid_without_changing_coordinates() {
        let full = [[-1., 1.], [0., 2.]];
        let path = crate::dynamics::work::Path2 {
            key: CacheKey {
                root: Arc::new(std::path::PathBuf::from("fixture")),
                sequence: 1,
            },
            start: [0.1, 1.5],
            end: [0.2, 0.2],
            transitions: 1,
            segments: vec![crate::dynamics::work::Segment2 {
                t: 0,
                points: [[0.1, 1.5], [0.2, 0.2]],
            }],
        };
        let view = episode_view_bounds(full, &path);
        assert!(in_bounds2(view, path.start) && in_bounds2(view, path.end));
        assert!(view[0][1] - view[0][0] < full[0][1] - full[0][0]);
        let uv = grid_view_uv(full, view);
        let grid_coordinate = |uv: Pos2| {
            [
                full[0][0] + f64::from(uv.x) * (full[0][1] - full[0][0]),
                full[1][1] - f64::from(uv.y) * (full[1][1] - full[1][0]),
            ]
        };
        let top_left = grid_coordinate(uv.min);
        let bottom_right = grid_coordinate(uv.max);
        assert!((top_left[0] - view[0][0]).abs() < 1e-6 && (top_left[1] - view[1][1]).abs() < 1e-6);
        assert!(
            (bottom_right[0] - view[0][1]).abs() < 1e-6
                && (bottom_right[1] - view[1][0]).abs() < 1e-6
        );
    }

    #[test]
    fn generic_transport_keeps_a_painted_matching_episode_ready_during_source_recheck() {
        let (directory, _, _) = domain_test_fixture::saved(domain_test_fixture::bounded(3));
        let run = RunIndex::open(directory.path().join("run"), ReadLimits::default()).unwrap();
        let (tx, rx) = mpsc::channel();
        let mut pane = DynamicsPane::new(egui::Context::default()).unwrap();
        pane.worker = DynamicsWorker::new(WorkLimits::default(), move || {
            let _ = tx.send(());
        })
        .unwrap();
        pane.worker
            .request_episode(crate::dynamics::work::EpisodeRequest::new(&run, 0).unwrap())
            .unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        pane.worker.poll();
        let frame = pane.worker.episode_frame().unwrap();
        pane.generic_key = Some(frame.key.clone());
        pane.action = Some(frame.action);
        pane.painted = true;
        assert!(pane.ready());
        pane.worker.recheck_episode();
        assert!(pane.worker.pending());
        assert!(
            !pane.ready(),
            "capture/public readiness still waits for source validation"
        );
        assert!(
            pane.transport_ready(),
            "an availability check does not prepare a different displayed checkpoint"
        );
        pane.action = Some(ActionAxis::Continuous(1));
        assert!(
            !pane.transport_ready(),
            "a different requested component is not ready"
        );
        pane.action = Some(ActionAxis::Continuous(0));
        pane.painted = false;
        assert!(
            !pane.transport_ready(),
            "an unpainted replacement is not ready"
        );
    }

    fn prepared_pane() -> (
        tempfile::TempDir,
        RunIndex,
        DynamicsPane,
        Request,
        mpsc::Receiver<()>,
    ) {
        let (dir, _, _) = domain_test_fixture::saved(domain_test_fixture::bounded(3));
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
                rx.recv_timeout(Duration::from_secs(10)).unwrap();
            }
        }
        let sources = SourceSet::new(
            &run,
            catalog.entries(),
            catalog.generation(),
            &WorkLimits::default(),
        )
        .unwrap();
        let mut pane = DynamicsPane::new(egui::Context::default()).unwrap();
        let (tx, rx) = mpsc::channel();
        pane.worker = DynamicsWorker::new(WorkLimits::default(), move || {
            let _ = tx.send(());
        })
        .unwrap();
        let req = Request {
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
        let desired = Desired {
            domain: req.domain,
            generation: req.sources.generation(),
            revision: req.revision,
            sequence: req.selected.sequence,
            axes: req.axes,
            action: req.action,
            projection: req.projection,
        };
        pane.requested = Some(desired);
        pane.sources = Some(req.sources.clone());
        pane.worker.request(req.clone()).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        pane.accept_frame();
        assert!(pane.frame().is_some());
        (dir, run, pane, req, rx)
    }

    #[test]
    fn availability_refresh_keeps_ready_frame_until_current_failure_is_consumed() {
        let (_dir, run, mut pane, req, rx) = prepared_pane();
        pane.painted = true;
        let frame = pane.displayed.clone().unwrap();
        let counters = pane.worker.counters();
        pane.recheck_sources();
        assert!(pane.ready());
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(pane.ready());
        pane.accept_frame();
        pane.poll_availability();
        assert!(pane.ready());
        assert!(Arc::ptr_eq(&frame, pane.displayed.as_ref().unwrap()));
        assert_eq!(pane.worker.counters(), counters);
        std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
        pane.recheck_sources();
        assert!(
            pane.ready(),
            "refresh must only enqueue, never scan/invalidate synchronously"
        );
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        assert!(pane.frame().is_some());
        pane.accept_frame();
        pane.poll_availability();
        assert!(!pane.ready());
        assert!(
            pane.invalidated
                .as_ref()
                .unwrap()
                .contains("source file is missing")
        );
        assert!(pane.displayed.is_none() && pane.requested.is_none() && pane.published.is_none());
        assert!(pane.occupancy_texture.is_none() && pane.coverage_texture.is_none());
        assert_eq!(pane.worker.counters().submitted, counters.submitted);
        assert_eq!(pane.worker.poll_availability(&frame, &req.sources), None);
    }

    #[test]
    fn obsolete_domain_frame_is_not_adopted_while_new_domain_is_requested() {
        let (_dir, _run, mut pane, mut req, rx) = prepared_pane();
        let desired = pane.requested.unwrap();
        pane.recheck_sources();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        pane.change_domain(DomainSpec::automatic(1));
        pane.requested = Some(Desired {
            domain: pane.domain,
            ..desired
        });
        // The worker still holds its complete previous frame. This is the exact
        // adoption path while a replacement domain has not published yet.
        pane.accept_frame();
        pane.poll_availability();
        assert!(pane.frame().is_none());
        assert!(pane.reference.is_none());
        req.domain = pane.domain;
        pane.worker.request(req).unwrap();
        rx.recv_timeout(Duration::from_secs(10)).unwrap();
        pane.accept_frame();
        assert_eq!(
            pane.frame().unwrap().state.key.domain(),
            DomainSpec::automatic(1)
        );
    }
}
