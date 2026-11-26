//! Recorded episode playback with optional portable scenes. The learner
//! trajectory state remains the shared selection across workspaces.
use super::*;
use crate::{
    dynamics::source::{CatalogLimits, TrajectoryCatalog},
    environment::{
        EpisodeEnd, ExecutedAction, PreparedScene, ReadoutSpec, SceneIdentity, SceneTransform,
        SceneWorker,
    },
};
use egui::{FontId, Rect, Sense, Shape, Stroke, StrokeKind};
use rl::diagnostics::render::{RenderPrimitive, RenderStroke};
use rl::diagnostics::{schema::RunManifest, visualization::EnvironmentSemantics};

#[path = "environment_workspace/benchmark.rs"]
mod benchmark;
use benchmark::BenchmarkPlayback;

#[derive(Clone, Debug, PartialEq, Eq)]
enum TelemetryChannel {
    Observation(usize),
    Action(usize),
    Frame(String, usize),
}

pub(super) struct EnvironmentPane {
    catalog: Option<TrajectoryCatalog>,
    decode_gate: crate::dynamics::work::DecodeGate,
    worker: SceneWorker,
    session: u64,
    playing: bool,
    last_tick: Instant,
    painted: bool,
    status: Option<String>,
    selection: Option<(u64, String, SelectedInput)>,
    pending_step: Option<usize>,
    compare_benchmark: bool,
    benchmark: Option<BenchmarkPlayback>,
    telemetry_channels: Vec<TelemetryChannel>,
    telemetry_initialized: bool,
}
impl EnvironmentPane {
    pub fn new(ctx: egui::Context) -> Result<Self> {
        Ok(Self {
            catalog: None,
            decode_gate: Arc::new(crate::dynamics::work::DecodeBudget::default()),
            worker: SceneWorker::new(move || ctx.request_repaint())?,
            session: 0,
            playing: false,
            last_tick: Instant::now(),
            painted: false,
            status: None,
            selection: None,
            pending_step: None,
            compare_benchmark: cfg!(feature = "screenshots")
                && std::env::var("RL_VIEWER_ENVIRONMENT_COMPARE").as_deref() == Ok("true"),
            benchmark: None,
            telemetry_channels: Vec::new(),
            telemetry_initialized: false,
        })
    }
    pub fn leave(&mut self) {
        self.worker.retire();
        self.session = self.session.wrapping_add(1);
        self.playing = false;
        self.painted = false;
        self.status = None;
        self.selection = None;
        self.pending_step = None;
        self.benchmark = None;
    }
    pub fn recheck_source(&mut self, run: &RunIndex) {
        if let Some(frame) = self.worker.frame()
            && let Some(index) = run.by_sequence(frame.identity.key.sequence)
        {
            let path = run.root().join(&run.recording().entries()[index].file);
            if let Err(error) = self.worker.check_path(path) {
                self.status = Some(error.to_string());
                self.painted = false;
            }
        }
        if let Some(benchmark) = &mut self.benchmark {
            benchmark.recheck_source(run);
        }
    }
    pub fn clear(&mut self) {
        self.leave();
        self.catalog = None;
        self.telemetry_channels.clear();
        self.telemetry_initialized = false;
    }
}
impl Laboratory {
    pub fn environment_available(&self) -> bool {
        if let Some(run) = &self.run
            && typed_workspace::typed_view(run, View::Environment, &self.selection)
        {
            return has_record(run, &self.selection.model_role, RecordKind::TypedEpisode);
        }
        self.environment_pane
            .catalog
            .as_ref()
            .filter(|c| {
                self.run
                    .as_ref()
                    .is_some_and(|r| c.matches(r, &self.selection.model_role))
            })
            .is_some_and(|c| !c.entries().is_empty())
    }
    pub fn environment_ready(&self) -> bool {
        if self
            .run
            .as_ref()
            .is_some_and(|run| typed_workspace::typed_view(run, View::Environment, &self.selection))
        {
            return self.view == View::Environment
                && self.typed_pane.episode_publication.is_some_and(
                    |(revision, sequence, step)| {
                        revision == self.selection.revision
                            && self.selection.input
                                == SelectedInput::TrajectoryStep { sequence, step }
                    },
                );
        }
        self.environment_pane.status.is_none()
            && self.environment_pane.painted
            && self.environment_pane.worker.frame().is_some_and(|f| {
                f.identity.revision == self.selection.revision
                    && f.identity.role == self.selection.model_role
                    && self.selection.input
                        == SelectedInput::TrajectoryStep {
                            sequence: f.identity.key.sequence,
                            step: f.identity.step,
                        }
            })
            && !self.environment_pane.worker.pending()
            && self
                .environment_pane
                .benchmark
                .as_ref()
                .is_none_or(|episode| !episode.pending())
    }
    pub fn environment_status(&self) -> Option<&str> {
        self.environment_pane
            .status
            .as_deref()
            .or_else(|| self.environment_pane.worker.error())
    }
    pub fn select_environment_step(&mut self, sequence: u64, step: usize) -> Result<()> {
        if self
            .run
            .as_ref()
            .is_some_and(|run| typed_workspace::typed_view(run, View::Environment, &self.selection))
        {
            return self.select_typed_episode_step(sequence, step);
        }
        let source = self
            .environment_pane
            .catalog
            .as_ref()
            .and_then(|c| {
                c.entries().iter().find(|s| {
                    s.key.sequence == sequence
                        && s.revision == self.selection.revision
                        && s.role == self.selection.model_role
                })
            })
            .ok_or_else(|| rl::Error::State("No recorded episode at this checkpoint".into()))?;
        if step > source.transitions {
            return Err(rl::Error::State(
                "Environment step exceeds final recorded state".into(),
            ));
        }
        self.selection.input = SelectedInput::TrajectoryStep { sequence, step };
        self.workspace_defaults = None;
        self.environment_pane.painted = false;
        Ok(())
    }
    pub(super) fn prepare_environment_catalog(&mut self, run: &RunIndex) -> Result<()> {
        let pane = &mut self.environment_pane;
        if pane
            .catalog
            .as_ref()
            .is_none_or(|c| !c.matches(run, &self.selection.model_role))
        {
            pane.clear();
            pane.catalog = Some(TrajectoryCatalog::with_decode_gate(
                run,
                &self.selection.model_role,
                CatalogLimits::default(),
                pane.decode_gate.clone(),
            )?);
        }
        pane.catalog.as_mut().unwrap().poll(run, &mut self.loader)?;
        pane.worker.poll();
        Ok(())
    }
    pub(super) fn environment_transport(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        let pane = &mut self.environment_pane;
        // Wait for the full catalog so partial discovery cannot prematurely
        // snap the shared selection to a different recorded episode.
        let catalog = pane
            .catalog
            .as_ref()
            .filter(|c| c.matches(run, &self.selection.model_role));
        let checking = catalog.is_none_or(|c| c.pending());
        let positions: Vec<_> = if checking {
            Vec::new()
        } else {
            catalog
                .unwrap()
                .entries()
                .iter()
                .map(|s| s.revision)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        };
        self.transport.hold_unavailable_snapshot(None);
        let ready = pane.painted && !pane.worker.pending();
        let published = ready.then_some(self.selection.revision);
        let retry = self.transport.show_evaluations(
            ui,
            run,
            &mut self.selection,
            crate::timeline_ui::SnapshotPresentation {
                positions: &positions,
                published,
                ready,
                checking,
                error: pane.worker.error(),
            },
        )?;
        if retry {
            pane.leave();
            if let Some(c) = &mut pane.catalog {
                c.retry(run, &mut self.loader)?;
            }
        }
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("environment-role")
                .selected_text(&self.selection.model_role)
                .show_ui(ui, |ui| {
                    for role in run.record_roles(RecordKind::Trajectory) {
                        ui.selectable_value(&mut self.selection.model_role, role.into(), role);
                    }
                });
        });
        Ok(())
    }
    pub(super) fn environment_workspace(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        // Role edits in the toolbar are processed before any canvas publication.
        self.prepare_environment_catalog(run)?;
        let reference_role = self
            .comparison
            .reference_run()
            .filter(|reference| reference.root() == run.root())
            .and_then(|_| benchmark::reference_role(run, self.comparison.reference_role()));
        let pane = &mut self.environment_pane;
        // Publish the selected input together with the prepared scene and its
        // readouts, never while the worker is still decoding the next state.
        if let Some(step) = pane.pending_step
            && pane.selection.as_ref().is_some_and(|old| {
                old.0 == self.selection.revision
                    && old.1 == self.selection.model_role
                    && old.2 == self.selection.input
            })
            && let Some(frame) = pane.worker.frame()
            && frame.identity.step == step
            && frame.identity.revision == self.selection.revision
            && frame.identity.role == self.selection.model_role
            && matches!(self.selection.input, SelectedInput::TrajectoryStep { sequence, .. } if sequence == frame.identity.key.sequence)
        {
            self.selection.input = SelectedInput::TrajectoryStep {
                sequence: frame.identity.key.sequence,
                step,
            };
            pane.selection = Some((
                self.selection.revision,
                self.selection.model_role.clone(),
                self.selection.input.clone(),
            ));
            pane.pending_step = None;
        }
        if pane.worker.error().is_some() {
            pane.pending_step = None;
            pane.playing = false;
        }
        pane.painted = false;
        let catalog = pane.catalog.as_ref().expect("prepared catalog");
        let episodes: Vec<_> = catalog
            .entries()
            .iter()
            .filter(|s| {
                s.revision == self.selection.revision && s.role == self.selection.model_role
            })
            .cloned()
            .collect();
        if catalog.omitted() > 0 || catalog.rejected() > 0 {
            ui.small(format!(
                "Recorded episode window: {} omitted · {} rejected",
                catalog.omitted(),
                catalog.rejected()
            ));
        }
        if episodes.is_empty() {
            pane.worker.retire();
            pane.benchmark = None;
            pane.playing = false;
            pane.status = Some(if catalog.pending() {
                "Checking recorded episodes…".into()
            } else {
                format!(
                    "No recorded episode at checkpoint k = {} for {}.",
                    self.selection.revision, self.selection.model_role
                )
            });
            ui.weak(pane.status.as_deref().unwrap());
            pane.painted = !catalog.pending();
            return Ok(());
        }
        let selected = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, .. } => {
                episodes.iter().position(|s| s.key.sequence == sequence)
            }
            _ => None,
        };
        let mut episode = selected.unwrap_or(0);
        if selected.is_none() {
            self.selection.input = SelectedInput::TrajectoryStep {
                sequence: episodes[episode].key.sequence,
                step: 0,
            };
            self.workspace_defaults = None;
        }
        let old_episode = episode;
        let mut step = match self.selection.input {
            SelectedInput::TrajectoryStep { step, .. } => step,
            _ => 0,
        };
        let displayed_step = step;
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("environment-episode")
                .selected_text(&episodes[episode].id)
                .show_ui(ui, |ui| {
                    for (index, source) in episodes.iter().enumerate() {
                        ui.selectable_value(&mut episode, index, &source.id);
                    }
                });
            if episode != old_episode {
                step = 0;
                pane.playing = false;
            }
            let end = episodes[episode].transitions;
            if ui.small_button("Previous state").clicked() {
                step = step.saturating_sub(1);
                pane.playing = false;
            }
            if ui
                .small_button(if pane.playing {
                    "Pause episode"
                } else {
                    "Play episode"
                })
                .clicked()
            {
                pane.playing = !pane.playing;
                pane.last_tick = Instant::now();
            }
            if ui.small_button("Next state").clicked() {
                step = step.saturating_add(1).min(end);
                pane.playing = false;
            }
            if ui
                .add(egui::Slider::new(&mut step, 0..=end).text("state"))
                .changed()
            {
                pane.playing = false;
            }
            if displayed_step == end {
                ui.weak("Final state");
                pane.playing = false;
            }
        });
        let source = &episodes[episode];
        let identity_before = (
            self.selection.revision,
            self.selection.model_role.clone(),
            self.selection.input.clone(),
        );
        if pane
            .selection
            .as_ref()
            .is_some_and(|old| old != &identity_before)
        {
            pane.playing = false;
            pane.worker.retire();
            pane.pending_step = None;
        }
        if step > source.transitions {
            pane.worker.retire();
            pane.status = Some("Selected state exceeds the recorded episode".into());
            ui.weak(pane.status.as_deref().unwrap());
            return Ok(());
        }
        if pane.playing
            && pane.pending_step.is_none()
            && pane.last_tick.elapsed() >= Duration::from_millis(100)
            && pane
                .worker
                .frame()
                .is_some_and(|f| f.identity.step == step && f.identity.key == source.key)
        {
            step = step.saturating_add(1).min(source.transitions);
            pane.last_tick = Instant::now();
            if step == source.transitions {
                pane.playing = false;
            }
        }
        if episode != old_episode {
            pane.worker.retire();
            pane.pending_step = None;
            self.selection.input = SelectedInput::TrajectoryStep {
                sequence: source.key.sequence,
                step: 0,
            };
        } else if step != displayed_step {
            pane.pending_step = Some(step);
        }
        self.workspace_defaults = None;
        let requested = (
            self.selection.revision,
            self.selection.model_role.clone(),
            self.selection.input.clone(),
        );
        // Local controls and episode playback commit the final identity above.
        // Retire its predecessor even when that request failed without a frame.
        if pane.selection.as_ref().is_some_and(|old| old != &requested) {
            pane.worker.retire();
        }
        pane.selection = Some(requested);
        self.trajectory = Some(source.key.sequence);
        let identity = SceneIdentity {
            session: pane.session,
            key: source.key.clone(),
            revision: self.selection.revision,
            role: self.selection.model_role.clone(),
            step: pane.pending_step.unwrap_or(step),
        };
        if let Some(error) = pane.worker.error() {
            pane.status = Some(error.into());
            ui.weak(pane.status.as_deref().unwrap());
            if ui.button("Retry recorded state").clicked() {
                pane.leave();
                self.loader.invalidate();
            }
            return Ok(());
        }
        let index = run
            .by_sequence(source.key.sequence)
            .ok_or_else(|| rl::Error::State("Recorded trajectory was removed".into()))?;
        let Some(chunk) =
            self.loader
                .request_admitted(run, index, crate::environment::MAX_RAW_BYTES)?
        else {
            pane.status = Some("Loading recorded environment…".into());
            ui.weak(pane.status.as_deref().unwrap());
            return Ok(());
        };
        let mut readouts = Vec::new();
        if let (Record::Trajectory(record), Some(descriptor)) = (
            &chunk.record,
            &run.recording().manifest().environment_descriptor,
        ) {
            for channel in &descriptor.frames {
                if let Some(key) = record.frames.get(&channel.name) {
                    for component in 0..channel.width {
                        if readouts.len() == 6 {
                            break;
                        }
                        let suffix = channel
                            .unit
                            .as_ref()
                            .map(|u| format!(" ({u})"))
                            .unwrap_or_default();
                        let label = if channel.width == 1 {
                            format!("{}{suffix}", channel.label)
                        } else {
                            format!("{}[{component}]{suffix}", channel.label)
                        };
                        readouts.push(ReadoutSpec {
                            key: key.clone(),
                            component,
                            label,
                        });
                    }
                }
                if readouts.len() == 6 {
                    break;
                }
            }
        }
        if let Err(error) = pane.worker.submit(identity.clone(), &chunk, readouts) {
            pane.worker.retire();
            pane.pending_step = None;
            pane.playing = false;
            pane.status = Some(error.to_string());
            ui.weak(pane.status.as_deref().unwrap());
            return Ok(());
        }
        let mut jump = None;
        if let Some(frame) = pane.worker.frame().filter(|f| {
            f.identity.session == identity.session && f.identity.key == identity.key
                && f.identity.revision == identity.revision && f.identity.role == identity.role
                && matches!(self.selection.input, SelectedInput::TrajectoryStep { sequence, step } if sequence == f.identity.key.sequence && step == f.identity.step)
        }).cloned() {
            pane.status = None;
            let matching_role = reference_role.filter(|_| source.scenario.is_some() && !crate::reference::provider_role(&source.role));
            ui.horizontal_wrapped(|ui| {
                if run.record_roles(RecordKind::Policy).next().is_some()
                    && ui.link("Inspect policy prediction").clicked()
                {
                    jump = Some(View::PolicyMap);
                }
                if run.record_roles(RecordKind::Critic).next().is_some()
                    && ui.link("Inspect critic estimate").clicked()
                {
                    jump = Some(View::Value);
                }
                if matching_role.is_some() {
                    ui.checkbox(&mut pane.compare_benchmark, "Compare matched benchmark");
                }
            });
            let result = if let Some(role) = matching_role.filter(|_| pane.compare_benchmark) {
                if pane.benchmark.as_ref().is_none_or(|benchmark| !benchmark.matches(run, role)) {
                    pane.benchmark = Some(BenchmarkPlayback::new(
                        run, role, pane.decode_gate.clone(), &pane.worker, ui.ctx().clone(),
                    )?);
                }
                let benchmark_frame = ui.columns(2, |columns| {
                    columns[0].strong("Learner episode");
                    paint_comparison_readouts(&mut columns[0], &frame, run.recording().manifest());
                    let benchmark = pane.benchmark.as_mut().expect("benchmark playback");
                    match benchmark.show(&mut columns[1], run, &mut self.loader, source, pane.session) {
                        Ok(frame) => frame,
                        Err(error) => {
                            columns[1].weak(error.to_string());
                            None
                        }
                    }
                });
                if let Some(benchmark) = benchmark_frame {
                    let semantics = EnvironmentSemantics::new(run.recording().manifest());
                    let pairs = matched_frame_traces(&frame, &benchmark, semantics);
                    if pairs.is_empty() {
                        ui.columns(2, |columns| {
                            paint_scene_panel(&mut columns[0], &frame)?;
                            paint_scene_panel(&mut columns[1], &benchmark)
                        })
                    } else if ui.available_height() < 360. {
                        egui::ScrollArea::vertical().id_salt(("matched-environment", source.key.sequence)).show(ui, |ui| {
                            ui.columns(2, |columns| {
                                for (column, item) in columns.iter_mut().zip([frame.as_ref(), benchmark.as_ref()]) {
                                    let width = column.available_width();
                                    column.allocate_ui(egui::vec2(width, 180.), |ui| paint_scene_panel(ui, item)).inner?;
                                }
                                Ok::<(), rl::Error>(())
                            })?;
                            ui.add_space(8.);
                            let width = ui.available_width();
                            ui.allocate_ui(egui::vec2(width, 480.), |ui| {
                                crate::theme::panel_frame().show(ui, |ui| {
                                    paint_matched_telemetry(ui, &frame, &benchmark, semantics, &pairs, true);
                                });
                            });
                            Ok(())
                        }).inner
                    } else if ui.available_width() >= 900. && frame.scene.is_some() && benchmark.scene.is_some() {
                        ui.columns(2, |columns| {
                            let height = columns[0].available_height();
                            let scene_height = ((height - 8.) / 2.).max(170.);
                            let width = columns[0].available_width();
                            for (label, item) in [("Learner", frame.as_ref()), ("Benchmark", benchmark.as_ref())] {
                                columns[0].allocate_ui(egui::vec2(width, scene_height), |ui| {
                                    crate::theme::panel_frame().show(ui, |ui| {
                                        ui.strong(label);
                                        paint_scene_panel(ui, item)
                                    }).inner
                                }).inner?;
                                columns[0].add_space(8.);
                            }
                            crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                paint_matched_telemetry(ui, &frame, &benchmark, semantics, &pairs, true);
                            });
                            Ok(())
                        })
                    } else {
                        let scene_height = (ui.available_height() * 0.55).max(170.);
                        let scene_result = ui.columns(2, |columns| {
                            for (column, item) in columns.iter_mut().zip([frame.as_ref(), benchmark.as_ref()]) {
                                let width = column.available_width();
                                column.allocate_ui(egui::vec2(width, scene_height), |ui| paint_scene_panel(ui, item)).inner?;
                            }
                            Ok(())
                        });
                        ui.add_space(8.);
                        paint_matched_telemetry(ui, &frame, &benchmark, semantics, &pairs, false);
                        scene_result
                    }
                } else {
                    paint_scene_panel(ui, &frame)
                }
            } else {
                pane.benchmark = None;
                paint_readouts(ui, &frame, run.recording().manifest());
                let semantics = EnvironmentSemantics::new(run.recording().manifest());
                let observations = frame.telemetry.iter().flat_map(|telemetry| telemetry.observations.iter().enumerate())
                    .filter(|(index, trace)| !trace.is_empty() && !matches!(
                        semantics.observation(*index).and_then(|field| field.kind),
                        Some(rl::diagnostics::visualization::ObservationType::Categorical { .. })
                    ))
                    .map(|(index, trace)| (TelemetryChannel::Observation(index), trace));
                let actions = frame.telemetry.iter().flat_map(|telemetry| telemetry.actions.iter().enumerate()).filter(|(_, trace)| !trace.is_empty())
                    .map(|(index, trace)| (TelemetryChannel::Action(index), trace));
                let frames = frame.telemetry.iter().flat_map(|telemetry| telemetry.frames.iter()).filter(|trace| !trace.points.is_empty())
                    .map(|trace| (TelemetryChannel::Frame(trace.name.clone(), trace.component), &trace.points));
                let traces: Vec<_> = frames.chain(observations).chain(actions).collect();
                let max_channels = if frame.scene.is_some() { 5 } else { 6 };
                if !pane.telemetry_initialized && !traces.is_empty() {
                    if let (Some(telemetry), Some(descriptor)) = (&frame.telemetry, semantics.descriptor()) {
                        for preferred in &descriptor.preferences.trajectories {
                            if telemetry.frames.iter().any(|trace| trace.name == preferred.channel && trace.component == preferred.index) {
                                let channel = TelemetryChannel::Frame(preferred.channel.clone(), preferred.index);
                                if !pane.telemetry_channels.contains(&channel) && pane.telemetry_channels.len() < max_channels { pane.telemetry_channels.push(channel); }
                            }
                        }
                    }
                    if pane.telemetry_channels.is_empty() {
                        pane.telemetry_channels.extend(traces.iter().filter_map(|(channel, _)| matches!(channel, TelemetryChannel::Frame(..)).then_some(channel.clone())).take(max_channels));
                    }
                    let mut observations: Vec<_> = traces.iter().filter_map(|(channel, _)| match channel {
                        TelemetryChannel::Observation(index) => Some((channel.clone(), semantics.observation(*index).map(|field| field.label))),
                        TelemetryChannel::Action(_) | TelemetryChannel::Frame(..) => None,
                    }).collect();
                    observations.sort_by_key(|(_, label)| telemetry_priority(label.as_deref().unwrap_or("")));
                    let has_semantic_priorities = observations.first().is_some_and(|(_, label)| telemetry_priority(label.as_deref().unwrap_or("")) < 4);
                    let count = if has_semantic_priorities { 4 } else { 3 };
                    if pane.telemetry_channels.is_empty() { pane.telemetry_channels = observations.into_iter().take(count).map(|(channel, _)| channel).collect(); }
                    if pane.telemetry_channels.len() < max_channels && traces.iter().any(|(channel, _)| channel == &TelemetryChannel::Action(0)) {
                        pane.telemetry_channels.push(TelemetryChannel::Action(0));
                    }
                    if pane.telemetry_channels.is_empty() {
                        pane.telemetry_channels.push(traces[0].0.clone());
                    }
                    pane.telemetry_initialized = true;
                }
                retain_available_telemetry(&mut pane.telemetry_channels, &traces);
                if !traces.is_empty() {
                    let mut selected_step = None;
                    let numerical_frames = frame.scene.is_none()
                        && frame.telemetry.as_ref().is_some_and(|telemetry| !telemetry.frames.is_empty());
                    let result = if ui.available_width() >= 760. && numerical_frames {
                        ui.columns(2, |columns| {
                            let split = pane.telemetry_channels.len().div_ceil(2);
                            selected_step = crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                ui.set_min_width((ui.available_width() - 24.).max(0.));
                                paint_observation_telemetry(ui, &frame, semantics, &traces, &mut pane.telemetry_channels, TelemetryLayout { visible: 0..split, max_channels, controls: true })
                            }).inner;
                            selected_step = crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                ui.set_min_width((ui.available_width() - 24.).max(0.));
                                paint_observation_telemetry(ui, &frame, semantics, &traces, &mut pane.telemetry_channels, TelemetryLayout { visible: split..usize::MAX, max_channels, controls: false })
                            }).inner.or(selected_step);
                            Ok(())
                        })
                    } else if ui.available_width() >= 760. {
                        let trajectory = frame.scene.as_ref().and_then(|_| frame_trajectory(&frame, semantics));
                        ui.columns(2, |columns| {
                            let result = if let Some(trajectory) = trajectory.as_ref().filter(|_| columns[0].available_height() >= 350.) {
                                let width = columns[0].available_width();
                                let scene_height = (columns[0].available_height() * 0.62).max(170.);
                                let result = columns[0].allocate_ui(egui::vec2(width, scene_height), |ui| {
                                    crate::theme::panel_frame().show(ui, |ui| {
                                        ui.set_min_width((ui.available_width() - 24.).max(0.));
                                        paint_scene_panel(ui, &frame)
                                    }).inner
                                }).inner;
                                columns[0].add_space(8.);
                                selected_step = crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                    ui.set_min_width((ui.available_width() - 24.).max(0.));
                                    paint_frame_trajectory(ui, &frame, trajectory)
                                }).inner;
                                result
                            } else {
                                crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                    ui.set_min_width((ui.available_width() - 24.).max(0.));
                                    paint_scene_panel(ui, &frame)
                                }).inner
                            };
                            selected_step = crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                ui.set_min_width((ui.available_width() - 24.).max(0.));
                                paint_observation_telemetry(ui, &frame, semantics, &traces, &mut pane.telemetry_channels, TelemetryLayout { visible: 0..usize::MAX, max_channels, controls: true })
                            }).inner.or(selected_step);
                            result
                        })
                    } else {
                        let scene_height = (ui.available_height() * 0.4).clamp(170., 300.);
                        let telemetry_height = (pane.telemetry_channels.len().max(1) as f32 * 132.).max(240.);
                        egui::ScrollArea::vertical().id_salt(("environment-stacked", source.key.sequence)).show(ui, |ui| {
                            let width = ui.available_width();
                            let scene = if !numerical_frames { ui.allocate_ui(egui::vec2(width, scene_height), |ui| paint_scene_panel(ui, &frame)).inner } else { Ok(()) };
                            selected_step = ui.allocate_ui(egui::vec2(width, telemetry_height), |ui| paint_observation_telemetry(ui, &frame, semantics, &traces, &mut pane.telemetry_channels, TelemetryLayout { visible: 0..usize::MAX, max_channels, controls: true })).inner;
                            scene
                        }).inner
                    };
                    if let Some(step) = selected_step {
                        pane.pending_step = Some(step.min(source.transitions));
                        pane.playing = false;
                        ui.ctx().request_repaint();
                    }
                    result
                } else {
                    paint_scene_panel(ui, &frame)
                }
            };
            if let Err(error) = result {
                pane.status = Some(error.to_string());
                ui.weak(pane.status.as_deref().unwrap());
            }
            pane.painted = true;
        } else {
            pane.status = Some(
                pane.worker
                    .error()
                    .unwrap_or("Preparing selected episode state…")
                    .into(),
            );
            ui.weak(pane.status.as_deref().unwrap());
        }
        if pane.playing {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        if let Some(view) = jump {
            self.select_view(view);
        }
        Ok(())
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn environment_capture_ready(&self) -> bool {
        self.environment_pane.painted
            && !self.environment_pane.worker.pending()
            && self
                .environment_pane
                .catalog
                .as_ref()
                .is_some_and(|c| !c.pending())
            && self.environment_pane.worker.error().is_none()
            && self
                .environment_pane
                .benchmark
                .as_ref()
                .is_none_or(|episode| !episode.pending())
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn environment_capture_error(&self) -> Option<String> {
        self.environment_pane.worker.error().map(str::to_owned)
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn environment_capture_evidence(&self) -> serde_json::Value {
        let p = &self.environment_pane;
        let f = p.worker.frame();
        serde_json::json!({"requested_revision":self.selection.revision,"source_revision":f.map(|f|f.identity.revision),"sequence":f.map(|f|f.identity.key.sequence),"step":f.map(|f|f.identity.step),"scene_version":f.and_then(|f|f.scene.as_ref()).map(|scene|scene.version),"primitives":f.and_then(|f|f.scene.as_ref()).map(|scene|scene.items.len()),"primitive_steps":f.map(|f|f.playback.progress.primitive_steps),"cumulative_reward":f.map(|f|f.playback.progress.observed_reward),"scenario_seed":f.and_then(|f|f.playback.scenario_seed),"benchmark":p.benchmark.as_ref().and_then(BenchmarkPlayback::evidence),"ready":self.environment_capture_ready(),"pending":p.worker.pending(),"status":self.environment_status(),"ownership":p.worker.metrics(),"catalog":p.catalog.as_ref().map(|c|format!("{:?}",c.metrics()))})
    }
}

fn matched_frame_traces<'a>(
    learner: &'a PreparedScene,
    benchmark: &'a PreparedScene,
    semantics: EnvironmentSemantics<'_>,
) -> Vec<(
    &'a crate::environment::FrameTrace,
    &'a crate::environment::FrameTrace,
)> {
    let (Some(a), Some(b), Some(descriptor)) = (
        &learner.telemetry,
        &benchmark.telemetry,
        semantics.descriptor(),
    ) else {
        return Vec::new();
    };
    if a.physical_time != b.physical_time {
        return Vec::new();
    }
    descriptor
        .preferences
        .trajectories
        .iter()
        .filter_map(|preferred| {
            let first = a.frames.iter().find(|trace| {
                trace.name == preferred.channel
                    && trace.component == preferred.index
                    && !trace.points.is_empty()
            })?;
            let second = b.frames.iter().find(|trace| {
                trace.name == preferred.channel
                    && trace.component == preferred.index
                    && !trace.points.is_empty()
            })?;
            Some((first, second))
        })
        .take(3)
        .collect()
}

fn paint_matched_telemetry(
    ui: &mut egui::Ui,
    learner: &PreparedScene,
    benchmark: &PreparedScene,
    semantics: EnvironmentSemantics<'_>,
    pairs: &[(
        &crate::environment::FrameTrace,
        &crate::environment::FrameTrace,
    )],
    stacked: bool,
) {
    use egui_plot::{Line, LineStyle, Plot, VLine};
    let physical = learner.telemetry.as_ref().is_some_and(|t| t.physical_time);
    let time_label = if physical {
        semantics
            .descriptor()
            .and_then(|d| d.clocks.simulation.as_ref())
            .map_or_else(
                || "Simulation time".to_owned(),
                |clock| crate::landscape::with_unit(clock.label.clone(), clock.unit.as_deref()),
            )
    } else {
        "Decision step".into()
    };
    let cursor = |frame: &PreparedScene| {
        if physical {
            frame
                .playback
                .progress
                .simulation_time
                .unwrap_or(frame.identity.step as f64)
        } else {
            frame.identity.step as f64
        }
    };
    let height = if stacked {
        (ui.available_height() / pairs.len() as f32 - 22.).max(100.)
    } else {
        (ui.available_height() - 30.).max(110.)
    };
    let max_time = pairs
        .iter()
        .flat_map(|(a, b)| [a.points.last(), b.points.last()])
        .flatten()
        .map(|point| point.x)
        .fold(0.0_f64, f64::max);
    let paint = |column: &mut egui::Ui,
                 (a, b): &(
        &crate::environment::FrameTrace,
        &crate::environment::FrameTrace,
    ),
                 last: bool| {
        let label = semantics
            .descriptor()
            .and_then(|d| d.frames.iter().find(|f| f.name == a.name))
            .map_or_else(
                || a.name.clone(),
                |f| {
                    f.unit
                        .as_ref()
                        .map_or_else(|| f.label.clone(), |u| format!("{} [{u}]", f.label))
                },
            );
        column.strong(&label);
        Plot::new(("matched-telemetry", &a.name, a.component))
            .height(height)
            .show_background(false)
            .grid_color(crate::theme::GRID)
            .link_axis(
                egui::Id::unique((
                    "matched-telemetry-time",
                    learner.identity.session,
                    learner.identity.key.sequence,
                    benchmark.identity.key.sequence,
                )),
                egui::Vec2b::new(true, false),
            )
            .y_axis_min_width(40.)
            .include_x(0.)
            .include_x(max_time)
            .x_axis_label(if !stacked || last {
                time_label.as_str()
            } else {
                ""
            })
            .x_axis_formatter(crate::landscape::axis_tick)
            .y_axis_formatter(crate::landscape::axis_tick)
            .legend(egui_plot::Legend::default())
            .show(column, |plot| {
                plot.line(Line::new("Learner", a.points.as_slice()).color(crate::theme::CURRENT));
                plot.line(
                    Line::new("Benchmark", b.points.as_slice())
                        .color(crate::theme::REFERENCE)
                        .style(LineStyle::dashed_dense()),
                );
                plot.vline(VLine::new("", cursor(learner)).color(crate::theme::CURRENT));
                plot.vline(
                    VLine::new("", cursor(benchmark))
                        .color(crate::theme::REFERENCE)
                        .style(LineStyle::dashed_dense()),
                );
            });
    };
    if stacked {
        for (index, pair) in pairs.iter().enumerate() {
            paint(ui, pair, index + 1 == pairs.len());
        }
    } else {
        ui.columns(pairs.len(), |columns| {
            for (column, pair) in columns.iter_mut().zip(pairs) {
                paint(column, pair, true);
            }
        });
    }
}

fn telemetry_priority(label: &str) -> u8 {
    let label = label.to_ascii_lowercase();
    if label.contains("altitude") || label.contains("height") {
        0
    } else if label.contains("vertical velocity") {
        1
    } else if (label.contains("attitude")
        || label.contains("orientation")
        || label.contains("pitch"))
        && !label.contains("rate")
    {
        2
    } else if label.contains("fuel") || label.contains("resource") || label.contains("inventory") {
        3
    } else {
        4
    }
}

fn retain_available_telemetry(
    channels: &mut Vec<TelemetryChannel>,
    traces: &[(TelemetryChannel, &Vec<egui_plot::PlotPoint>)],
) {
    channels.retain(|channel| traces.iter().any(|(available, _)| available == channel));
}

/// Join only observations made at the same recorded instant. Individual
/// telemetry channels may retain different extrema when they are downsampled.
fn paired_frame_trajectory(
    x: &[egui_plot::PlotPoint],
    y: &[egui_plot::PlotPoint],
) -> Vec<[f64; 3]> {
    let valid = |points: &[egui_plot::PlotPoint]| {
        points.iter().all(|p| p.x.is_finite() && p.y.is_finite())
            && points.windows(2).all(|p| p[0].x < p[1].x)
    };
    if !valid(x) || !valid(y) {
        return Vec::new();
    }
    let (mut i, mut j) = (0, 0);
    let mut paired = Vec::with_capacity(x.len().min(y.len()));
    while i < x.len() && j < y.len() {
        if x[i].x < y[j].x {
            i += 1;
        } else if x[i].x > y[j].x {
            j += 1;
        } else {
            paired.push([x[i].y, y[j].y, x[i].x]);
            i += 1;
            j += 1;
        }
    }
    paired
}

struct FrameTrajectory {
    x_label: String,
    y_label: String,
    x_channel: String,
    y_channel: String,
    bounds: [[f64; 2]; 2],
    points: Vec<[f64; 3]>,
}

fn current_frame_pair(readouts: &[(String, usize, f64)], x: &str, y: &str) -> Option<[f64; 2]> {
    if x == y {
        return None;
    }
    let unique = |channel: &str| {
        let mut values = readouts
            .iter()
            .filter(|(name, component, _)| name == channel && *component == 0)
            .map(|(_, _, value)| *value);
        let first = values.next().filter(|value| value.is_finite())?;
        values.next().is_none().then_some(first)
    };
    unique(x).zip(unique(y)).map(|(x, y)| [x, y])
}

fn frame_trajectory(
    frame: &PreparedScene,
    semantics: EnvironmentSemantics<'_>,
) -> Option<FrameTrajectory> {
    use rl::diagnostics::visualization::FrameRepresentation;
    let descriptor = semantics.descriptor()?;
    let telemetry = frame.telemetry.as_ref()?;
    let candidates = descriptor
        .frames
        .iter()
        .filter_map(|field| {
            if field.width != 1 || field.representation != Some(FrameRepresentation::Scalar) {
                return None;
            }
            let unit = field.unit.as_deref().filter(|unit| !unit.is_empty())?;
            let trace = telemetry
                .frames
                .iter()
                .find(|trace| trace.name == field.name && trace.component == 0)?;
            Some((field, unit, trace))
        })
        .collect::<Vec<_>>();
    for (i, (x, unit, x_trace)) in candidates.iter().enumerate() {
        for (y, _, y_trace) in candidates
            .iter()
            .skip(i + 1)
            .filter(|(_, y_unit, _)| y_unit == unit)
        {
            let points = paired_frame_trajectory(&x_trace.points, &y_trace.points);
            if points.len() >= 2 {
                let range = |trace: &crate::environment::FrameTrace| {
                    trace
                        .points
                        .iter()
                        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], p| {
                            [lo.min(p.y), hi.max(p.y)]
                        })
                };
                return Some(FrameTrajectory {
                    x_label: format!("{} [{unit}]", x.label),
                    y_label: format!("{} [{unit}]", y.label),
                    x_channel: x.name.clone(),
                    y_channel: y.name.clone(),
                    bounds: [range(x_trace), range(y_trace)],
                    points,
                });
            }
        }
    }
    None
}

fn paint_frame_trajectory(
    ui: &mut egui::Ui,
    frame: &PreparedScene,
    trajectory: &FrameTrajectory,
) -> Option<usize> {
    use egui_plot::{Line, Plot, Points};
    let title = if ui.available_height() < 160. {
        ui.horizontal(|ui| {
            ui.add_space(30.);
            ui.strong("Recorded trajectory")
        })
        .inner
    } else {
        ui.strong("Recorded trajectory")
    };
    title.on_hover_text("Points use frame values retained at the same recorded time. Lines connect those samples; intermediate values are not inferred.");
    let physical_time = frame
        .telemetry
        .as_ref()
        .is_some_and(|telemetry| telemetry.physical_time);
    let selected_time = if physical_time {
        frame.playback.progress.simulation_time
    } else {
        Some(frame.identity.step as f64)
    };
    let selected = current_frame_pair(
        &frame.frame_readouts,
        &trajectory.x_channel,
        &trajectory.y_channel,
    )
    .or_else(|| {
        selected_time
            .and_then(|time| trajectory.points.iter().find(|point| point[2] == time))
            .map(|point| [point[0], point[1]])
    });
    let points: Vec<_> = trajectory.points.iter().map(|p| [p[0], p[1]]).collect();
    let plot_id = egui::Id::unique((
        "environment-frame-trajectory",
        frame.identity.key.root.as_ref(),
        frame.identity.key.sequence,
    ));
    let result = Plot::new(plot_id)
        .id(plot_id)
        .height((ui.available_height() - 12.).max(90.))
        .width((ui.available_width() - 8.).max(120.))
        .show_background(false)
        .grid_color(crate::theme::GRID)
        .grid_spacing(25.0..=55.0)
        .x_grid_spacer(egui_plot::log_grid_spacer(2))
        .y_grid_spacer(egui_plot::log_grid_spacer(2))
        .include_x(trajectory.bounds[0][0])
        .include_x(trajectory.bounds[0][1])
        .include_y(trajectory.bounds[1][0])
        .include_y(trajectory.bounds[1][1])
        .x_axis_label(&trajectory.x_label)
        .y_axis_label(&trajectory.y_label)
        .x_axis_formatter(crate::landscape::axis_tick)
        .y_axis_formatter(crate::landscape::axis_tick)
        .show(ui, |plot| {
            plot.line(
                Line::new("Recorded trajectory", points)
                    .color(crate::theme::CURRENT)
                    .width(2.),
            );
            if let Some(point) = selected {
                plot.points(
                    Points::new("Selected state", vec![point])
                        .color(crate::theme::SELECTED)
                        .radius(4.),
                );
            }
            plot.pointer_coordinate()
        });
    if !result.response.clicked() {
        return None;
    }
    let position = result.inner?;
    let bounds = trajectory.points.iter().fold(
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        |[xlo, xhi, ylo, yhi], p| [xlo.min(p[0]), xhi.max(p[0]), ylo.min(p[1]), yhi.max(p[1])],
    );
    let x_scale = (bounds[1] - bounds[0]).max(f64::EPSILON);
    let y_scale = (bounds[3] - bounds[2]).max(f64::EPSILON);
    let time = trajectory.points.iter().min_by(|a, b| {
        let distance =
            |p: &[f64; 3]| ((p[0] - position.x) / x_scale).hypot((p[1] - position.y) / y_scale);
        distance(a).total_cmp(&distance(b))
    })?[2];
    if physical_time {
        frame.step_at_time(time)
    } else if time >= 0. && time.fract() == 0. && time < usize::MAX as f64 {
        Some(time as usize)
    } else {
        None
    }
}

#[cfg(test)]
#[test]
fn paired_frame_trajectory_uses_only_identical_recorded_times() {
    use egui_plot::PlotPoint;
    let x = vec![
        PlotPoint::new(0., 4.),
        PlotPoint::new(1., 5.),
        PlotPoint::new(3., 7.),
    ];
    let y = vec![
        PlotPoint::new(0., 9.),
        PlotPoint::new(2., 8.),
        PlotPoint::new(3., 6.),
    ];
    assert_eq!(
        paired_frame_trajectory(&x, &y),
        vec![[4., 9., 0.], [7., 6., 3.]]
    );
    let malformed = vec![PlotPoint::new(1., 0.), PlotPoint::new(1., 2.)];
    assert!(paired_frame_trajectory(&malformed, &y).is_empty());
    let nonfinite = vec![PlotPoint::new(0., f64::NAN)];
    assert!(paired_frame_trajectory(&x, &nonfinite).is_empty());
}

#[cfg(test)]
#[test]
fn selected_frame_pair_rejects_ambiguous_or_nonfinite_readouts() {
    let values = vec![("x".into(), 0, 2.), ("y".into(), 0, 3.)];
    assert_eq!(current_frame_pair(&values, "x", "y"), Some([2., 3.]));
    assert_eq!(current_frame_pair(&values, "x", "x"), None);
    let mut ambiguous = values.clone();
    ambiguous.push(("x".into(), 0, 4.));
    assert_eq!(current_frame_pair(&ambiguous, "x", "y"), None);
    assert_eq!(
        current_frame_pair(&[("x".into(), 0, f64::NAN), ("y".into(), 0, 1.)], "x", "y"),
        None
    );
}

#[cfg(test)]
#[test]
fn omitted_frame_channels_do_not_reassign_an_existing_selection() {
    let alpha = TelemetryChannel::Frame("alpha".into(), 0);
    let beta = TelemetryChannel::Frame("beta".into(), 0);
    let points = vec![egui_plot::PlotPoint::new(0., 1.)];
    let mut selected = vec![alpha, beta.clone()];
    retain_available_telemetry(&mut selected, &[(beta.clone(), &points)]);
    assert_eq!(selected, vec![beta]);
}

struct TelemetryLayout {
    visible: std::ops::Range<usize>,
    max_channels: usize,
    controls: bool,
}

fn paint_observation_telemetry(
    ui: &mut egui::Ui,
    frame: &PreparedScene,
    semantics: EnvironmentSemantics<'_>,
    traces: &[(TelemetryChannel, &Vec<egui_plot::PlotPoint>)],
    channels: &mut Vec<TelemetryChannel>,
    layout: TelemetryLayout,
) -> Option<usize> {
    use egui_plot::{Line, Plot, VLine};
    let TelemetryLayout {
        visible,
        max_channels,
        controls,
    } = layout;
    let label = |channel: &TelemetryChannel| match channel {
        TelemetryChannel::Observation(index) => semantics.observation(*index).map_or_else(
            || format!("obs[{index}]"),
            |field| {
                field.unit.map_or_else(
                    || field.label.to_string(),
                    |unit| format!("{} [{unit}]", field.label),
                )
            },
        ),
        TelemetryChannel::Action(index) => semantics.action(*index).map_or_else(
            || format!("action[{index}]"),
            |field| {
                field.unit.map_or_else(
                    || field.label.to_string(),
                    |unit| format!("{} [{unit}]", field.label),
                )
            },
        ),
        TelemetryChannel::Frame(channel, component) => semantics
            .descriptor()
            .and_then(|descriptor| {
                descriptor
                    .frames
                    .iter()
                    .find(|field| field.name == *channel)
            })
            .map_or_else(
                || {
                    if *component == 0 {
                        channel.clone()
                    } else {
                        format!("{channel}[{component}]")
                    }
                },
                |field| {
                    let name = if field.width == 1 {
                        field.label.clone()
                    } else {
                        format!("{}[{component}]", field.label)
                    };
                    field
                        .unit
                        .as_ref()
                        .map_or(name.clone(), |unit| format!("{name} [{unit}]"))
                },
            ),
    };
    if controls {
        ui.horizontal(|ui| {
        ui.strong("Episode telemetry").on_hover_text("Recorded observations, actions and diagnostic frames retain interval extrema and endpoints; the cursor follows the displayed episode state.");
        crate::landscape::popover(ui, "Channels", |ui| {
            for (channel, _) in traces {
                let mut enabled = channels.contains(channel);
                if ui.add_enabled(enabled || channels.len() < max_channels, egui::Checkbox::new(&mut enabled, label(channel))).changed() {
                    if enabled { channels.push(channel.clone()); } else { channels.retain(|item| item != channel); }
                }
            }
        });
    });
    } else {
        ui.add_space(ui.spacing().interact_size.y + ui.spacing().item_spacing.y);
    }
    let active: Vec<_> = channels
        .iter()
        .skip(visible.start)
        .take(visible.end.saturating_sub(visible.start))
        .filter_map(|selected| traces.iter().find(|(channel, _)| channel == selected))
        .collect();
    if active.is_empty() {
        return None;
    }
    // Plot height excludes the title, tick labels, and x-axis caption. Reserve
    // their space for every row so the last channel remains visible.
    let height =
        ((ui.available_height() - active.len() as f32 * 30.) / active.len() as f32).max(42.);
    let mut selected = None;
    let active_len = active.len();
    let shared_x = traces.iter().flat_map(|(_, points)| points.iter()).fold(
        None,
        |range: Option<[f64; 2]>, point| {
            point
                .x
                .is_finite()
                .then(|| {
                    range.map_or([point.x, point.x], |[lo, hi]| {
                        [lo.min(point.x), hi.max(point.x)]
                    })
                })
                .or(range)
        },
    );
    for (row, (channel, trace)) in active.into_iter().enumerate() {
        let label = label(channel);
        let physical_time = frame
            .telemetry
            .as_ref()
            .is_some_and(|telemetry| telemetry.physical_time);
        let axis_label = if physical_time {
            semantics
                .descriptor()
                .and_then(|descriptor| descriptor.clocks.simulation.as_ref())
                .map_or_else(
                    || "Simulation time".to_owned(),
                    |clock| {
                        clock.unit.as_ref().map_or_else(
                            || clock.label.clone(),
                            |unit| format!("{} [{unit}]", clock.label),
                        )
                    },
                )
        } else {
            "Decision step".into()
        };
        let cursor = if physical_time {
            frame
                .playback
                .progress
                .simulation_time
                .unwrap_or(frame.identity.step as f64)
        } else {
            frame.identity.step as f64
        };
        ui.label(&label);
        let plot_id = match channel {
            TelemetryChannel::Observation(index) => egui::Id::unique((
                "environment-observation",
                frame.identity.key.root.as_ref(),
                frame.identity.key.sequence,
                index,
            )),
            TelemetryChannel::Action(index) => egui::Id::unique((
                "environment-action",
                frame.identity.key.root.as_ref(),
                frame.identity.key.sequence,
                index,
            )),
            TelemetryChannel::Frame(name, component) => egui::Id::unique((
                "environment-frame",
                frame.identity.key.root.as_ref(),
                frame.identity.key.sequence,
                name,
                component,
            )),
        };
        let first_paint = egui_plot::PlotMemory::load(ui.ctx(), plot_id).is_none();
        let mut plot = Plot::new(plot_id)
            .id(plot_id)
            .height(height)
            .width((ui.available_width() - 8.).max(120.))
            .show_background(false)
            .grid_color(crate::theme::GRID)
            .grid_spacing(15.0..=30.0)
            .custom_x_axes(vec![
                egui_plot::AxisHints::new_x().label_spacing(40.0..=60.0),
            ])
            .custom_y_axes(vec![
                egui_plot::AxisHints::new_y().label_spacing(12.0..=16.0),
            ])
            .x_axis_label(if row + 1 == active_len {
                axis_label
            } else {
                String::new()
            })
            .x_grid_spacer(egui_plot::log_grid_spacer(2))
            .y_grid_spacer(egui_plot::log_grid_spacer(2))
            .x_axis_formatter(crate::landscape::axis_tick)
            .y_axis_min_width(40.)
            .y_axis_formatter(crate::landscape::axis_tick);
        if let Some([lo, hi]) = shared_x {
            plot = plot.include_x(lo).include_x(hi);
        }
        let response = plot.show(ui, |plot| {
            plot.line(Line::new(&label, trace.as_slice()).color(crate::theme::CURRENT));
            plot.vline(VLine::new("Selected state", cursor).color(crate::theme::SELECTED));
            plot.pointer_coordinate()
        });
        if first_paint {
            // egui_plot measures axis thickness on its first paint. Give the
            // chart a second pass before an otherwise idle viewer settles.
            ui.ctx().request_repaint();
        }
        if response.response.clicked()
            && let Some(position) = response.inner
            && position.x.is_finite()
        {
            selected = if physical_time {
                frame.step_at_time(position.x)
            } else {
                Some(position.x.max(0.).round() as usize)
            };
        }
    }
    selected
}

fn paint_comparison_readouts(ui: &mut egui::Ui, frame: &PreparedScene, manifest: &RunManifest) {
    let semantics = EnvironmentSemantics::new(manifest);
    ui.horizontal_wrapped(|ui| {
        if let Some(time) = frame.playback.progress.simulation_time {
            let clock = semantics
                .descriptor()
                .and_then(|d| d.clocks.simulation.as_ref());
            let label = clock.map_or("Time", |clock| clock.label.as_str());
            let unit = clock.and_then(|clock| clock.unit.as_deref()).unwrap_or("");
            ui.label(format!("{label} {time:.2} {unit}"));
        }
        ui.label(format!(
            "Cumulative reward {:.2}",
            frame.playback.progress.observed_reward
        ));
        crate::landscape::popover(ui, "Details", |ui| paint_readouts(ui, frame, manifest));
    });
}

fn paint_readouts(ui: &mut egui::Ui, frame: &PreparedScene, manifest: &RunManifest) {
    let state = &frame.playback;
    let semantics = EnvironmentSemantics::new(manifest);
    ui.horizontal_wrapped(|ui| {
        let step = state.progress.primitive_steps;
        ui.label(format!("Primitive steps {step}"));
        if let Some(time) = state.progress.simulation_time {
            let clock = semantics
                .descriptor()
                .and_then(|d| d.clocks.simulation.as_ref());
            let label = clock.map_or("Simulation time", |clock| clock.label.as_str());
            let unit = clock.and_then(|clock| clock.unit.as_deref()).unwrap_or("");
            ui.label(format!("{label} {time:.3} {unit}"));
        } else if let Some(seconds) = manifest
            .metadata
            .get("evaluation.decision_seconds")
            .and_then(|seconds| seconds.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds > 0.)
            .map(|seconds| seconds * frame.identity.step as f64)
        {
            ui.label(format!("Elapsed {seconds:.3} s"));
        }
        ui.label(format!(
            "Cumulative reward {:.3}",
            state.progress.observed_reward
        ));
        if let Some(reward) = state.reward {
            ui.label(format!("Reward {reward:.3}"))
                .on_hover_text("Reward observed after the displayed action");
        }
        if let Some(end) = state.end {
            ui.strong(match end {
                EpisodeEnd::Terminated => "Environment terminated",
                EpisodeEnd::Truncated => "Episode truncated",
                EpisodeEnd::ObjectiveHorizon => "Evaluation horizon reached",
                EpisodeEnd::Incomplete => "End of recorded prefix",
            });
        }
        if let Some(action) = &state.executed_action {
            ui.strong("Executed action");
            match action {
                ExecutedAction::Discrete(index) => {
                    let label = usize::try_from(*index)
                        .ok()
                        .and_then(|i| semantics.action(i));
                    ui.label(
                        label.map_or_else(|| index.to_string(), |label| label.label.into_owned()),
                    );
                }
                ExecutedAction::Continuous(action) => {
                    for (index, value) in action.values.iter().enumerate() {
                        if let Some(label) = semantics.action(index) {
                            ui.label(format!(
                                "{} {value:.3} {}",
                                label.label,
                                label.unit.unwrap_or("")
                            ));
                        }
                    }
                    if action.dimensions > action.values.len() {
                        ui.weak(format!(
                            "{} more components",
                            action.dimensions - action.values.len()
                        ));
                    }
                }
            }
        }
        crate::landscape::popover(ui, "Observations", |ui| {
            paint_observations(ui, frame, semantics);
        });
    });
}

fn paint_observations(
    ui: &mut egui::Ui,
    frame: &PreparedScene,
    semantics: EnvironmentSemantics<'_>,
) {
    let state = &frame.playback;
    ui.strong("Public observations");
    egui::ScrollArea::vertical()
        .id_salt(("episode-observations", frame.identity.key.sequence))
        .max_height(if frame.scene.is_some() {
            270.
        } else {
            (ui.available_height() * 0.35).clamp(78., 158.)
        })
        .show(ui, |ui| {
            egui::Grid::new(("observed-state", frame.identity.key.sequence))
                .striped(true)
                .show(ui, |ui| {
                    for (i, value) in state.observations.values.iter().enumerate() {
                        let field = semantics.observation(i);
                        let label = field
                            .as_ref()
                            .map_or_else(|| format!("obs[{i}]"), |field| field.label.to_string());
                        let unit = field.as_ref().and_then(|field| field.unit).unwrap_or("");
                        ui.label(label);
                        ui.monospace(format!("{value:.5} {unit}"));
                        ui.end_row();
                    }
                });
            if state.observations.dimensions > state.observations.values.len() {
                ui.weak(format!(
                    "Showing {} of {} components",
                    state.observations.values.len(),
                    state.observations.dimensions
                ));
            }
        });
    if !frame.readouts.is_empty() {
        ui.horizontal_wrapped(|ui| {
            for (label, value) in &frame.readouts {
                ui.label(format!("{label} {value:.3}"));
            }
        });
    }
}

fn paint_scene_panel(ui: &mut egui::Ui, frame: &PreparedScene) -> Result<()> {
    if let Some(scene) = &frame.scene {
        let size = egui::vec2(
            ui.available_width().max(1.),
            ui.available_height().max(170.),
        );
        let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
        paint_scene(ui, scene, rect)?;
    } else if let Some(rewards) = &frame.rewards
        && !rewards.is_empty()
    {
        paint_episode_rewards(ui, frame, rewards);
    }
    Ok(())
}

fn paint_episode_rewards(
    ui: &mut egui::Ui,
    frame: &PreparedScene,
    rewards: &[egui_plot::PlotPoint],
) {
    use egui_plot::{Plot, Points, VLine};
    ui.strong("Episode rewards");
    let step = frame.identity.step as f64;
    Plot::new(("episode-rewards", frame.identity.key.sequence))
        .height((ui.available_height() - 4.).max(155.))
        .width((ui.available_width() - 14.).max(180.))
        .x_axis_label("Decision step")
        .y_axis_label("Reward")
        .include_x(rewards.last().map_or(0., |point| point.x + 1.))
        .grid_color(crate::theme::GRID)
        .x_axis_formatter(crate::landscape::axis_tick)
        .y_axis_formatter(crate::landscape::axis_tick)
        .show(ui, |plot| {
            plot.points(
                Points::new("Recorded reward", rewards)
                    .color(crate::theme::EMPIRICAL)
                    .radius(2.5),
            );
            plot.vline(VLine::new("Selected state", step).color(crate::theme::SELECTED));
            if let Some(reward) = frame.playback.reward {
                plot.points(
                    Points::new("Selected reward", vec![[step, reward]])
                        .shape(egui_plot::MarkerShape::Diamond)
                        .radius(5.)
                        .color(crate::theme::SELECTED),
                );
            }
        });
}
fn color(c: [u8; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3])
}
fn stroke(s: &RenderStroke) -> Stroke {
    Stroke::new(s.width, color(s.color))
}
fn scene_text_bounds(anchor: egui::Pos2, size: egui::Vec2, previous: &[Rect], clip: Rect) -> Rect {
    let original = Rect::from_center_size(anchor, size);
    let mut candidate = original;
    // Keep this visual adjustment bounded even for scenes with many labels.
    for _ in 0..32 {
        let Some(overlap) = previous.iter().find(|other| candidate.intersects(**other)) else {
            return candidate;
        };
        candidate = candidate.translate(egui::vec2(0., overlap.bottom() + 2. - candidate.top()));
        if candidate.bottom() > clip.bottom() {
            return original;
        }
    }
    original
}

#[cfg(test)]
#[test]
fn scene_text_avoids_overlap_without_moving_unrelated_labels() {
    let clip = Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(300., 200.));
    let first = scene_text_bounds(egui::pos2(100., 20.), egui::vec2(100., 14.), &[], clip);
    let second = scene_text_bounds(egui::pos2(100., 26.), egui::vec2(100., 12.), &[first], clip);
    assert!(second.top() > first.bottom());
    let separate = scene_text_bounds(egui::pos2(250., 26.), egui::vec2(30., 12.), &[first], clip);
    assert_eq!(separate.center(), egui::pos2(250., 26.));
}

fn paint_scene(
    ui: &egui::Ui,
    scene: &rl::diagnostics::render::RenderScene,
    rect: Rect,
) -> Result<()> {
    let painter = ui.painter().with_clip_rect(rect);
    painter.rect_filled(rect, 0., color(scene.background));
    let t = SceneTransform::for_scene(scene, rect.shrink(8.))?;
    let mut text_bounds = Vec::new();
    for item in &scene.items {
        match &item.primitive {
            RenderPrimitive::Line {
                from,
                to,
                stroke: s,
            } => {
                painter.line_segment([t.point(*from), t.point(*to)], stroke(s));
            }
            RenderPrimitive::Polyline { points, stroke: s } => {
                painter.add(Shape::line(
                    points.iter().map(|p| t.point(*p)).collect(),
                    stroke(s),
                ));
            }
            RenderPrimitive::Polygon {
                points,
                fill,
                stroke: s,
            } => {
                painter.add(Shape::convex_polygon(
                    points.iter().map(|p| t.point(*p)).collect(),
                    fill.map(color).unwrap_or(Color32::TRANSPARENT),
                    s.as_ref().map(stroke).unwrap_or(Stroke::NONE),
                ));
            }
            RenderPrimitive::Circle {
                center,
                radius,
                fill,
                stroke: s,
            } => {
                painter.circle(
                    t.point(*center),
                    t.radius(*radius),
                    fill.map(color).unwrap_or(Color32::TRANSPARENT),
                    s.as_ref().map(stroke).unwrap_or(Stroke::NONE),
                );
            }
            RenderPrimitive::Rectangle {
                min,
                max,
                fill,
                stroke: s,
            } => {
                let r = Rect::from_two_pos(t.point(*min), t.point(*max));
                painter.rect(
                    r,
                    0.,
                    fill.map(color).unwrap_or(Color32::TRANSPARENT),
                    s.as_ref().map(stroke).unwrap_or(Stroke::NONE),
                    StrokeKind::Middle,
                );
            }
            RenderPrimitive::Text {
                position,
                text,
                color: c,
                size,
            } => {
                let galley =
                    painter.layout_no_wrap(text.clone(), FontId::proportional(*size), color(*c));
                let bounds =
                    scene_text_bounds(t.point(*position), galley.size(), &text_bounds, rect);
                if text_bounds.len() < 64 {
                    text_bounds.push(bounds);
                }
                painter.galley(bounds.min, galley, color(*c));
            }
        }
    }
    Ok(())
}
