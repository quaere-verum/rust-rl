//! Generic exact decision inspection shares the Dynamics episode and global input.
use super::*;
use crate::{
    dynamics::{DecisionAction, Episode, work::EpisodeRequest},
    interactive::{Interactive, decision::Request as DecisionRequest},
};
use rl::diagnostics::schema::RecordKind;

pub(super) enum TimelineFrame {
    Spatial(Arc<PreparedFrame>),
    Generic(Arc<crate::dynamics::work::EpisodeFrame>),
}
impl TimelineFrame {
    pub(super) fn episode(&self) -> &Episode {
        match self {
            Self::Spatial(f) => &f.selected.episode,
            Self::Generic(f) => &f.episode,
        }
    }
    pub(super) fn key(&self) -> &CacheKey {
        match self {
            Self::Spatial(f) => &f.selected.key,
            Self::Generic(f) => &f.key,
        }
    }
    pub(super) fn reward_trace(&self) -> &[[f64; 2]] {
        match self {
            Self::Spatial(f) => &f.selected.reward_trace,
            Self::Generic(f) => &f.reward_trace,
        }
    }
    pub(super) fn action_trace(&self) -> &[crate::dynamics::work::ActionSample] {
        match self {
            Self::Spatial(f) => &f.action_trace,
            Self::Generic(f) => &f.action_trace,
        }
    }
    pub(super) fn action(&self) -> ActionAxis {
        match self {
            Self::Spatial(f) => f.action,
            Self::Generic(f) => f.action,
        }
    }
}

impl Laboratory {
    pub(super) fn prepare_dynamics_decision(
        &mut self,
        run: &RunIndex,
        key: CacheKey,
        episode: &Episode,
        step: usize,
    ) {
        let request = (|| -> Result<_> {
            if let Some(reason) = run
                .recording()
                .manifest()
                .metadata
                .get("inference_history")
                .filter(|s| s.as_str() != "stateless")
            {
                return Err(rl::Error::Config(format!(
                    "Recurrent memory/history unavailable: {reason}"
                )));
            }
            let snapshot = run
                .candidates(
                    RecordKind::ModelSnapshot,
                    &self.selection.model_role,
                    None,
                    self.selection.revision,
                )
                .take_while(|&i| run.recording().entries()[i].revision == self.selection.revision)
                .find(|&i| run.has_model_record(i))
                .ok_or_else(|| {
                    rl::Error::Config(format!(
                        "No retained model at requested k={} for role {}",
                        self.selection.revision, self.selection.model_role
                    ))
                })?;
            let snapshot_key = run.key(snapshot)?;
            if self
                .interactive
                .decision_requested_for(&key, step, &snapshot_key)
            {
                return Ok(None);
            }
            DecisionRequest::new(run, snapshot, key, episode, step).map(Some)
        })();
        match request {
            Ok(Some(request)) => {
                self.dynamics_pane.prediction_reason = None;
                if let Err(error) = self
                    .interactive
                    .submit_decision(request)
                    .and_then(|()| self.interactive.poll_decision())
                {
                    self.dynamics_pane.prediction_reason = Some(error.to_string());
                }
            }
            Ok(None) => {
                self.dynamics_pane.prediction_reason = None;
                if let Err(error) = self.interactive.poll_decision() {
                    self.dynamics_pane.prediction_reason = Some(error.to_string());
                }
            }
            Err(error) => {
                self.interactive.clear_decision();
                self.dynamics_pane.prediction_reason = Some(error.to_string());
            }
        }
    }
    pub(super) fn dynamics_generic_episode(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        egui::ScrollArea::vertical()
            .id_salt("generic-dynamics-content")
            .auto_shrink([false, false])
            .show(ui, |ui| self.dynamics_generic_content(ui, run))
            .inner
    }
    fn dynamics_generic_content(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let pane = &mut self.dynamics_pane;
        let Some(catalog) = &pane.catalog else {
            ui.weak("Loading episodes…");
            return Ok(());
        };
        if catalog.pending() {
            ui.weak("Loading episodes…");
            return Ok(());
        }
        if catalog.omitted() > 0 || catalog.rejected() > 0 {
            ui.small(format!(
                "Episode catalogue: {} available · {} omitted · {} rejected",
                catalog.entries().len(),
                catalog.omitted(),
                catalog.rejected()
            ));
        }
        let explicit = matches!(self.selection.input, SelectedInput::TrajectoryStep { .. });
        let selected = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, .. } => catalog
                .entries()
                .iter()
                .find(|s| s.key.sequence == sequence),
            _ => None,
        }
        .or_else(|| {
            (!explicit)
                .then(|| {
                    catalog
                        .entries()
                        .iter()
                        .rev()
                        .find(|s| s.revision <= self.selection.revision)
                })
                .flatten()
        })
        .or_else(|| (!explicit).then(|| catalog.entries().last()).flatten())
        .cloned();
        let Some(source) = selected else {
            ui.weak(if explicit {
                "Selected episode unavailable; its exact selection is preserved"
            } else {
                "No episodes available"
            });
            return Ok(());
        };
        let mut sequence = source.key.sequence;
        egui::ComboBox::from_id_salt("generic-episode")
            .selected_text(format!(
                "{} · episode k={} · #{}",
                source.id, source.revision, sequence
            ))
            .show_ui(ui, |ui| {
                for entry in catalog.entries() {
                    ui.selectable_value(
                        &mut sequence,
                        entry.key.sequence,
                        format!(
                            "{} · k={} · #{}",
                            entry.id, entry.revision, entry.key.sequence
                        ),
                    );
                }
            });
        if sequence != source.key.sequence {
            crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                sequence,
                step: 0,
            })
            .apply(run, &mut self.selection)?;
            return Ok(());
        }
        if pane.generic_key.as_ref() != Some(&source.key) {
            #[cfg(feature = "screenshots")]
            if let Ok(value) = std::env::var("RL_VIEWER_DYNAMICS_STEP") {
                let step = value
                    .parse::<usize>()
                    .map_err(|_| rl::Error::Config("invalid Dynamics capture step".into()))?;
                if step > source.transitions {
                    return Err(rl::Error::Config(
                        "Dynamics capture step outside selected episode".into(),
                    ));
                }
                crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                    sequence: source.key.sequence,
                    step,
                })
                .apply(run, &mut self.selection)?;
            }
            let index = run
                .by_sequence(source.key.sequence)
                .ok_or_else(|| rl::Error::State("selected episode missing".into()))?;
            pane.worker.request_episode(
                EpisodeRequest::new(run, index)?
                    .with_action(pane.action.unwrap_or(ActionAxis::Continuous(0)))?,
            )?;
            pane.generic_key = Some(source.key.clone());
            pane.requested = None;
            pane.published = None;
            pane.displayed = None;
            pane.reference = None;
            pane.occupancy_texture = None;
            pane.coverage_texture = None;
            pane.texture_state = None;
        }
        pane.worker.poll();
        if let Some(error) = pane.worker.error() {
            ui.weak(format!("Episode unavailable: {error}"));
            return Ok(());
        }
        let frame = pane
            .worker
            .episode_frame()
            .filter(|f| f.key == source.key)
            .cloned();
        let Some(frame) = frame else {
            ui.weak("Preparing exact episode…");
            return Ok(());
        };
        let episode = &frame.episode;
        let total = episode.transitions();
        let step = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step } if sequence == frame.key.sequence => {
                step.min(total)
            }
            _ => 0,
        };
        let mut next = step;
        let mut component = pane.action.unwrap_or(frame.action);
        ui.horizontal_wrapped(|ui| {
            ui.label("Action");
            egui::ComboBox::from_id_salt("generic-episode-action")
                .selected_text(action_label(episode, component))
                .show_ui(ui, |ui| {
                    for &axis in episode.action_axes() {
                        ui.selectable_value(&mut component, axis, action_label(episode, axis));
                    }
                });
            ui.label(format!("Decision {step}/{total}"));
            if ui.button("Previous step").clicked() {
                next = step.saturating_sub(1);
            }
            if ui.button("Next step").clicked() {
                next = (step + 1).min(total);
            }
            if ui.button("Final state").clicked() {
                next = total;
            }
            ui.add(egui::Slider::new(&mut next, 0..=total).show_value(false));
        });
        if component != frame.action {
            pane.action = Some(component);
            pane.painted = false;
            let index = run
                .by_sequence(frame.key.sequence)
                .ok_or_else(|| rl::Error::State("selected episode missing".into()))?;
            pane.worker
                .request_episode(EpisodeRequest::new(run, index)?.with_action(component)?)?;
            ui.weak("Preparing action component…");
            return Ok(());
        }
        if next != step
            || !matches!(self.selection.input,SelectedInput::TrajectoryStep{sequence,..}if sequence==frame.key.sequence)
        {
            crate::pane::SelectionIntent::Input(SelectedInput::TrajectoryStep {
                sequence: frame.key.sequence,
                step: next,
            })
            .apply(run, &mut self.selection)?;
        }
        let step = next;
        self.prepare_dynamics_decision(run, frame.key.clone(), episode, step);
        crate::theme::panel_frame().show(ui, |ui| {
            decision_strip(ui, episode, step, component, None);
            prediction_summary(
                ui,
                &self.interactive,
                &frame.key,
                step,
                self.selection.revision,
                component,
                self.dynamics_pane.prediction_reason.as_deref(),
            );
        });
        ui.add_space(8.);
        ui.strong("Episode timeline");
        self.dynamics_timeline_plot(ui, run, Vec2::new(ui.available_width().max(180.), 200.));
        decision_details(
            ui,
            episode,
            step,
            &self.interactive,
            &frame.key,
            self.selection.revision,
            self.dynamics_pane.prediction_reason.as_deref(),
        );
        egui::CollapsingHeader::new("Spatial availability").show(ui, |ui| {
            ui.small("Spatial views require two numeric observation coordinates and compatible evaluation episodes.");
        });
        self.dynamics_pane.painted = true;
        Ok(())
    }
}
#[cfg(feature = "screenshots")]
pub(super) fn action_evidence(action: DecisionAction<'_>) -> serde_json::Value {
    match action {
        DecisionAction::Discrete(id) => serde_json::json!({"discrete_id":id}),
        DecisionAction::Continuous(values) => serde_json::json!({"continuous":values}),
    }
}
#[cfg(feature = "screenshots")]
pub(super) fn prediction_evidence(
    interactive: &Interactive,
    key: &CacheKey,
    step: usize,
    revision: u64,
) -> serde_json::Value {
    interactive.decision_frame().filter(|f|&f.episode_key==key&&f.step==step&&f.model_revision==revision).map_or(serde_json::Value::Null,|f|serde_json::json!({
        "episode_sequence":f.episode_key.sequence,"episode_revision":f.episode_revision,"step":f.step,"model_sequence":f.snapshot_key.sequence,"model_revision":f.model_revision,
        "recorded_action":f.recorded_action,"inferred_action":f.inferred_action,"policy":f.policy_name,"stochastic_representative":f.representative_stochastic,
        "critics":f.critic_summaries.iter().map(|(name,s)|serde_json::json!({"name":name,"expected_return":s.expected_return,"details":s.details})).collect::<Vec<_>>(),"critic_unavailable":f.critic_note}))
}

fn observation_rows(ui: &mut egui::Ui, episode: &Episode, values: &[f64], axes: Option<Axes>) {
    let primary = axes.map_or([0, usize::from(values.len() > 1)], |a| [a.x, a.y]);
    let mut indices = Vec::with_capacity(4);
    for index in primary.into_iter().chain(0..values.len()) {
        if index < values.len() && !indices.contains(&index) {
            indices.push(index);
        }
        if indices.len() == 4 {
            break;
        }
    }
    ui.scope(|ui| {
        ui.spacing_mut().interact_size.y = 16.;
        ui.spacing_mut().item_spacing.y = 0.;
        for index in indices {
            if let (Some(value), Some(coordinate)) =
                (values.get(index), episode.coordinates().get(index))
            {
                let label = coordinate.unit.as_ref().map_or_else(
                    || coordinate.label.clone(),
                    |unit| format!("{} [{unit}]", coordinate.label),
                );
                ui.horizontal(|ui| {
                    let label_width = (ui.available_width() - 72.).max(20.);
                    ui.allocate_ui_with_layout(
                        egui::vec2(label_width, 16.),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.add(
                                egui::Label::new(egui::RichText::new(&label).small()).truncate(),
                            )
                            .on_hover_text(label);
                        },
                    );
                    ui.monospace(crate::landscape::statistic_number(*value))
                        .on_hover_text(format!("{value}"));
                });
            }
        }
    });
}

pub(super) fn decision_strip(
    ui: &mut egui::Ui,
    episode: &Episode,
    step: usize,
    action: ActionAxis,
    axes: Option<Axes>,
) {
    let Ok(row) = episode.step(step) else {
        return;
    };
    ui.horizontal_wrapped(|ui| {
        ui.strong("Selected transition");
        if row.action.is_none() {
            ui.small("Final observation · no transition");
        }
        ui.small(format!(
            "t={step}/{} · episode k={}",
            episode.transitions(),
            episode.context().identity.learner_update_revision
        ));
        ui.small(match episode.status() {
            crate::dynamics::EpisodeStatus::Terminated => "completed episode",
            crate::dynamics::EpisodeStatus::FiniteHorizon => "completed finite-horizon evaluation",
            crate::dynamics::EpisodeStatus::Truncated => "truncated prefix",
            crate::dynamics::EpisodeStatus::Prefix => "episode prefix",
        });
        if let Some(time) = row.simulation_time {
            ui.small(format!("Simulation time {time:.3}"))
                .on_hover_text(format!("Recorded time: {time}"));
        }
        if let Some(value) = row.return_to_go {
            ui.small(format!("Return-to-go {}", crate::landscape::statistic_number(value)))
                .on_hover_text("One realized complete discounted episode suffix, not a conditional estimate, bootstrap target or critic error.");
        }
        if row.terminated == Some(true) { ui.small("Terminated"); }
        if row.truncated == Some(true) { ui.small("Truncated"); }
        if let Some(mask) = row.mask { ui.small(format!("Legal mask · {} actions", mask.len())); }
    });
    let reward = row
        .reward
        .map_or_else(|| "— no reward".into(), crate::landscape::statistic_number);
    ui.columns(4, |columns| {
        columns[0].weak(egui::RichText::new("Observation").small());
        observation_rows(&mut columns[0], episode, row.observation, axes);
        columns[1].weak(egui::RichText::new("Action taken").small());
        if row.action.is_none() {
            columns[1].small("— no action");
        } else {
            let mut shown = Vec::with_capacity(3);
            for component in std::iter::once(action).chain(episode.action_axes().iter().copied()) {
                if shown.contains(&component) {
                    continue;
                }
                shown.push(component);
                if let Ok(Some(value)) = episode.action_value(step, component) {
                    let label = format!(
                        "{} {}",
                        action_label(episode, component),
                        value_text(episode, value)
                    );
                    columns[1]
                        .label(egui::RichText::new(compact_label(&label, 72)).small())
                        .on_hover_text(label);
                }
                if shown.len() == 3 {
                    break;
                }
            }
        }
        columns[2].weak(egui::RichText::new("Reward").small());
        columns[2].label(egui::RichText::new(reward).small());
        columns[3].weak(egui::RichText::new("Next observation").small());
        if let Some(next) = row.next_observation {
            observation_rows(&mut columns[3], episode, next, axes);
        } else {
            columns[3].small("— no next input");
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prediction_summary(
    ui: &mut egui::Ui,
    interactive: &Interactive,
    key: &CacheKey,
    step: usize,
    revision: u64,
    component: ActionAxis,
    reason: Option<&str>,
) {
    let frame = interactive
        .decision_frame()
        .filter(|f| &f.episode_key == key && f.step == step && f.model_revision == revision);
    ui.horizontal_wrapped(|ui| {
        ui.small(egui::RichText::new("Model prediction").color(crate::theme::CURRENT));
        if let Some(reason) = reason.or_else(|| interactive.decision_error()) {
            ui.small(
                egui::RichText::new(if reason.to_lowercase().contains("recurrent") {
                    "Unavailable · recurrent state missing"
                } else {
                    "Prediction unavailable · see Details"
                })
                .color(crate::theme::WARNING),
            )
            .on_hover_text(reason);
        } else if let Some(frame) = frame {
            if let Some(action) = &frame.inferred_action {
                let value = match action {
                    rl::core::action::Action::Discrete(id) => id.to_string(),
                    rl::core::action::Action::Continuous(values) => match component {
                        ActionAxis::Continuous(i) => values.get(i).map_or_else(
                            || "unavailable".into(),
                            |v| {
                                format!(
                                    "a[{i}] {}",
                                    crate::landscape::statistic_number(f64::from(*v))
                                )
                            },
                        ),
                        ActionAxis::Discrete => "unavailable".into(),
                    },
                    rl::core::action::Action::Conditional(_) => "unavailable".into(),
                };
                ui.label(
                    egui::RichText::new(format!(
                        "{} action {value}",
                        if frame.representative_stochastic {
                            "Representative"
                        } else {
                            "Inferred"
                        }
                    ))
                    .color(crate::theme::CURRENT),
                );
            }
            for (name, summary) in &frame.critic_summaries {
                ui.label(
                    egui::RichText::new(summary.expected_return.map_or_else(
                        || format!("Critic {name} —"),
                        |v| format!("Critic {name} {}", crate::landscape::statistic_number(v)),
                    ))
                    .color(crate::theme::CURRENT),
                );
            }
            if frame.critic_note.is_some() {
                ui.small("Critic unavailable · see Details");
            }
        } else {
            ui.small(if interactive.decision_pending() {
                "Preparing prediction…"
            } else {
                "Prediction unavailable · see Details"
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decision_details(
    ui: &mut egui::Ui,
    episode: &Episode,
    step: usize,
    interactive: &Interactive,
    key: &CacheKey,
    revision: u64,
    reason: Option<&str>,
) -> bool {
    egui::CollapsingHeader::new("Decision details")
        .default_open(crate::theme::inspection_default(ui))
        .show(ui, |ui| {
            let height = ui.available_height().clamp(180., 360.);
            ui.columns(2, |columns| {
                egui::ScrollArea::vertical()
                    .id_salt("complete-recorded-decision")
                    .max_height(height)
                    .show(&mut columns[0], |ui| recorded(ui, episode, step));
                egui::ScrollArea::vertical()
                    .id_salt("complete-model-decision")
                    .max_height(height)
                    .show(&mut columns[1], |ui| {
                        predictions(ui, interactive, key, step, revision, reason)
                    });
            });
        })
        .body_response
        .is_some()
}

pub(super) fn recorded(ui: &mut egui::Ui, episode: &Episode, step: usize) {
    ui.separator();
    ui.strong("Decision");
    ui.small(format!(
        "Episode {} · actual k={} · step {} · role {}",
        episode.record().id,
        episode.context().identity.learner_update_revision,
        step,
        episode.context().identity.model_role
    ));
    ui.small(format!(
        "Episode status: {:?} · complete returns: {}",
        episode.status(),
        episode.record().evaluation.complete_returns
    ));
    ui.small(format!(
        "{}: {:.6}",
        match episode.status() {
            crate::dynamics::EpisodeStatus::Terminated => "Undiscounted episode reward sum",
            crate::dynamics::EpisodeStatus::FiniteHorizon => "Finite-horizon reward sum",
            crate::dynamics::EpisodeStatus::Truncated | crate::dynamics::EpisodeStatus::Prefix => {
                "Undiscounted prefix reward sum"
            }
        },
        episode.realized_return()
    ));
    let conditioning = match episode.record().evaluation.conditioning {
        rl::diagnostics::schema::Conditioning::InitialScenario => "initial scenario",
        rl::diagnostics::schema::Conditioning::Observation { .. } => "observation",
        rl::diagnostics::schema::Conditioning::FullSimulatorState { .. } => "full simulator state",
        rl::diagnostics::schema::Conditioning::History { .. } => "history",
    };
    ui.small(format!(
        "Conditioning: {conditioning} · γ={}",
        episode.record().evaluation.discount
    ));
    let Ok(row) = episode.step(step) else {
        return;
    };
    if let Some(reward) = row.reward {
        ui.label(format!("Immediate reward: {reward}"));
    }
    if let Some(value) = row.return_to_go {
        ui.label(format!("Return-to-go: {value:.6}")).on_hover_text("One realized complete discounted episode suffix, not a conditional estimate, bootstrap target or critic error.");
    }
    if let Some(action) = row.action {
        ui.strong("Action taken");
        match action {
            DecisionAction::Discrete(id) => {
                ui.monospace(format!(
                    "{id} · {}",
                    episode.discrete_label(id).unwrap_or("unlabelled category")
                ));
            }
            DecisionAction::Continuous(values) => {
                egui::ScrollArea::vertical()
                    .id_salt("decision-action")
                    .max_height(100.)
                    .show_rows(ui, 20., values.len(), |ui, rows| {
                        for i in rows {
                            ui.monospace(format!(
                                "{} = {} {}",
                                episode
                                    .action_label(ActionAxis::Continuous(i))
                                    .unwrap_or("action"),
                                values[i],
                                episode
                                    .action_unit(ActionAxis::Continuous(i))
                                    .ok()
                                    .flatten()
                                    .unwrap_or("")
                            ));
                        }
                    });
            }
        }
    } else {
        ui.strong("Final observation");
        ui.weak("No action, reward, next observation or transition flags");
    }
    ui.strong("Complete observation");
    egui::ScrollArea::vertical()
        .id_salt("decision-observation")
        .max_height(120.)
        .show_rows(ui, 20., row.observation.len(), |ui, rows| {
            for i in rows {
                let c = &episode.coordinates()[i];
                ui.monospace(format!(
                    "{} = {} {}",
                    c.label,
                    row.observation[i],
                    c.unit.as_deref().unwrap_or("")
                ));
            }
        });
    if let Some(mask) = row.mask {
        ui.strong("Legal mask");
        egui::ScrollArea::vertical()
            .id_salt("decision-mask")
            .max_height(80.)
            .show_rows(ui, 20., mask.len(), |ui, rows| {
                for i in rows {
                    ui.monospace(format!("action {i}: {}", mask[i] != 0));
                }
            });
    }
    if let Some(terminated) = row.terminated {
        ui.label(format!(
            "Terminated: {terminated} · truncated: {}",
            row.truncated.unwrap_or(false)
        ));
    }
    if let Some(time) = row.simulation_time {
        ui.label(format!("Simulation time: {time:.3}"))
            .on_hover_text(format!("Recorded time: {time}"));
    }
    if let Some(next) = row.next_observation {
        ui.strong("Complete next observation");
        egui::ScrollArea::vertical()
            .id_salt("decision-next")
            .max_height(120.)
            .show_rows(ui, 20., next.len(), |ui, rows| {
                for i in rows {
                    ui.monospace(format!("{} = {}", episode.coordinates()[i].label, next[i]));
                }
            });
    }
    egui::CollapsingHeader::new("Episode provenance").show(ui, |ui| {
        ui.small(format!(
            "Source policy: {}",
            episode.record().evaluation.continuation_policy
        ));
        ui.small(format!(
            "Conditioning: {:?}",
            episode.record().evaluation.conditioning
        ));
    });
}
pub(super) fn predictions(
    ui: &mut egui::Ui,
    interactive: &Interactive,
    key: &CacheKey,
    step: usize,
    revision: u64,
    reason: Option<&str>,
) {
    ui.separator();
    ui.strong("Model prediction");
    if let Some(reason) = reason.or_else(|| interactive.decision_error()) {
        ui.weak(format!("Historical prediction unavailable: {reason}"));
        return;
    }
    let Some(frame) = interactive
        .decision_frame()
        .filter(|f| &f.episode_key == key && f.step == step && f.model_revision == revision)
    else {
        ui.weak(if interactive.decision_pending() {
            "Preparing selected prediction…"
        } else {
            "Historical prediction unavailable"
        });
        return;
    };
    ui.small(format!(
        "Episode k={} · actual model k={} · snapshot #{}",
        frame.episode_revision, frame.model_revision, frame.snapshot_key.sequence
    ));
    if let Some(action) = &frame.inferred_action {
        ui.label(format!("{} {}: {action:?}",if frame.representative_stochastic{"Representative policy action"}else{"Inferred policy action"},frame.policy_name.as_deref().unwrap_or(""))).on_hover_text("Historical stateless inference at this exact input. Recorded behavior action remains separate; stochastic policies use their deterministic representative action here.");
    }
    for (name, summary) in &frame.critic_summaries {
        match summary.expected_return {
            Some(value) => {
                ui.label(format!("{name}: {value:.6}"));
            }
            None => {
                ui.weak(format!("{name}: statistic unavailable"));
            }
        }
        if let Some(details) = &summary.details {
            ui.small(details);
        }
    }
    if let Some(note) = &frame.critic_note {
        ui.weak(format!("Critic prediction unavailable: {note}"));
    }
    ui.small("Q critics evaluate the action taken. One return-to-go does not establish prediction error.");
}
