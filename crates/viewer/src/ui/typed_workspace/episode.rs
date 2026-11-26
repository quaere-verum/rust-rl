use super::*;
use crate::theme::EMPIRICAL;
use egui_plot::VLine;
use rl::diagnostics::typed::TypedEpisodeSource;

impl Laboratory {
    pub(super) fn typed_episode(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let entries = run.recording().entries();
        let revision = self.selection.revision;
        let role = &self.selection.model_role;
        let eligible = |i: usize| {
            entries[i].kind == RecordKind::TypedEpisode
                && entries[i].model_role == *role
                && entries[i].revision == revision
        };
        let requested = match self.selection.input {
            SelectedInput::TrajectoryStep { sequence, step } => run
                .by_sequence(sequence)
                .filter(|&i| eligible(i))
                .map(|i| (i, step)),
            _ => None,
        };
        let Some((index, step)) = requested
            .or_else(|| exact(run, RecordKind::TypedEpisode, role, None, revision).map(|i| (i, 0)))
        else {
            ui.weak("No episode at this checkpoint");
            return Ok(());
        };
        let sequence = entries[index].sequence;
        self.trajectory = Some(sequence);
        let Some(loaded) = self.typed_pane.episode.read(run, index, &mut self.loader)? else {
            ui.weak("Loading episode…");
            return Ok(());
        };
        let TypedPayload::Episode(episode) = loaded.as_ref() else {
            return Err(Error::Shape("expected a typed episode".into()));
        };
        if self.typed_pane.episode_sequence != Some(sequence) {
            self.typed_pane.episode_tick = None;
            self.typed_pane.episode_sequence = Some(sequence);
        }
        let mut step = step.min(episode.steps.len());
        let mut changed_episode = None;
        ui.horizontal(|ui| {
            ui.heading(if self.view == View::Dynamics {
                "Episode visitation"
            } else {
                "Recorded episode"
            });
            egui::ComboBox::from_id_salt("typed-episode")
                .selected_text(format!("#{sequence}"))
                .show_ui(ui, |ui| {
                    for i in run
                        .candidates(RecordKind::TypedEpisode, role, None, revision)
                        .take_while(|&i| eligible(i))
                        .take(128)
                    {
                        let other = entries[i].sequence;
                        if ui
                            .selectable_label(sequence == other, format!("#{other}"))
                            .clicked()
                        {
                            changed_episode = Some(other);
                        }
                    }
                });
            ui.weak(match &episode.source {
                TypedEpisodeSource::Evaluation { scenario_seed, .. } => {
                    format!("Evaluation · seed {scenario_seed}")
                }
                TypedEpisodeSource::Factual { episode_id, .. } => {
                    format!("Training · episode {episode_id}")
                }
            });
        });
        if let Some(sequence) = changed_episode
            && sequence != entries[index].sequence
        {
            self.selection.input = SelectedInput::TrajectoryStep { sequence, step: 0 };
            ui.ctx().request_repaint();
            return Ok(());
        }
        if self.view == View::Environment {
            let playing = self.typed_pane.episode_tick.is_some();
            ui.horizontal(|ui| {
                if ui
                    .button(if playing {
                        "Pause episode"
                    } else {
                        "Play episode"
                    })
                    .clicked()
                {
                    self.typed_pane.episode_tick = if playing {
                        None
                    } else {
                        if step == episode.steps.len() {
                            step = 0
                        }
                        Some(Instant::now())
                    };
                }
                if ui
                    .add(egui::Slider::new(&mut step, 0..=episode.steps.len()).text("Decision"))
                    .changed()
                {
                    self.typed_pane.episode_tick = None;
                }
            });
            if let Some(last) = self.typed_pane.episode_tick {
                if last.elapsed() >= Duration::from_millis(100) {
                    step = (step + 1).min(episode.steps.len());
                    self.typed_pane.episode_tick = (step < episode.steps.len()).then(Instant::now);
                }
                ui.ctx().request_repaint_after(Duration::from_millis(16));
            }
        }
        if self.view == View::Dynamics {
            let mut counts = BTreeMap::<u64, usize>::new();
            for observation in &episode.observations[..episode.steps.len()] {
                if let Some(discrete) = observation.as_discrete() {
                    *counts.entry(discrete.symbol).or_default() += 1;
                }
            }
            if counts.is_empty() {
                ui.label(format!("{} recorded decisions", episode.steps.len()));
            } else {
                egui::Grid::new("typed-visits")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Observation symbol");
                        ui.strong("Decision visits");
                        ui.end_row();
                        for (symbol, count) in counts.into_iter().take(ROW_LIMIT) {
                            ui.monospace(symbol.to_string());
                            ui.monospace(count.to_string());
                            ui.end_row();
                        }
                    });
            }
        } else {
            let observation = &episode.observations[step];
            draw_observation(ui, observation);
            if let Some(state) = episode.steps.get(step) {
                ui.label(action_label(
                    &state.action,
                    observation,
                    &run.recording().manifest().action_space,
                ));
                ui.label(format!("Reward  {:.4}", state.reward));
            } else if let Some(last) = episode.steps.last() {
                ui.weak(if last.terminated {
                    "Terminated"
                } else if last.truncated {
                    "Truncated"
                } else {
                    "Final decision"
                });
            }
            ui.strong("Episode rewards");
            Plot::new("typed-episode-rewards")
                .height(160.)
                .x_axis_label("Decision")
                .y_axis_label("Reward")
                .show(ui, |plot| {
                    plot.line(
                        Line::new(
                            "Reward",
                            episode
                                .steps
                                .iter()
                                .enumerate()
                                .map(|(i, step)| [i as f64, f64::from(step.reward)])
                                .collect::<Vec<_>>(),
                        )
                        .color(EMPIRICAL),
                    );
                    plot.vline(VLine::new("Selected decision", step as f64).color(CURRENT));
                });
        }
        self.selection.input = SelectedInput::TrajectoryStep { sequence, step };
        self.typed_pane.ready = Some(revision);
        self.typed_pane.episode_publication = Some((revision, sequence, step));
        Ok(())
    }

    pub(in crate::ui) fn select_typed_episode_step(
        &mut self,
        sequence: u64,
        step: usize,
    ) -> Result<()> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| Error::State("No recording is open".into()))?;
        let index = run
            .by_sequence(sequence)
            .filter(|&i| {
                let entry = &run.recording().entries()[i];
                entry.kind == RecordKind::TypedEpisode
                    && entry.revision == self.selection.revision
                    && entry.model_role == self.selection.model_role
            })
            .ok_or_else(|| Error::State("No recorded episode at this checkpoint".into()))?;
        let Some(chunk) = run
            .recording()
            .source(index)?
            .load_if_within_charge(PAYLOAD_ADMISSION)?
        else {
            return Err(Error::Shape(
                "Typed episode exceeds the capture budget".into(),
            ));
        };
        let TypedPayload::Episode(episode) = TypedPayload::read(&chunk)? else {
            return Err(Error::Shape("Expected a typed episode".into()));
        };
        if step > episode.steps.len() {
            return Err(Error::State(
                "Environment step exceeds final recorded state".into(),
            ));
        }
        self.selection.input = SelectedInput::TrajectoryStep { sequence, step };
        self.typed_pane.switch_view();
        Ok(())
    }
}
