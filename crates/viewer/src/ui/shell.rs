//! Shared presentation controls. Selection and transport remain their existing owners.
use super::*;

impl Laboratory {
    pub(super) fn workspace_shortcuts(&mut self, ctx: &egui::Context) {
        if self.run.is_none() || ctx.egui_wants_keyboard_input() {
            return;
        }
        for (key, view) in [
            (egui::Key::Num1, View::Training),
            (egui::Key::Num2, View::PolicyMap),
            (egui::Key::Num3, View::Value),
            (egui::Key::Num4, View::Network),
            (egui::Key::Num5, View::Dynamics),
            (egui::Key::Num6, View::Environment),
        ] {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, key)) {
                if view == View::Network
                    && self.run.as_ref().is_some_and(|run| {
                        typed_workspace::typed_run(run)
                            && !has_record(run, &self.selection.model_role, RecordKind::Activations)
                    })
                {
                    continue;
                }
                self.select_view(view);
                ctx.request_repaint();
            }
        }
    }

    pub(super) fn application_controls(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("···", |ui| {
            if ui.button("Open recording…").clicked() {
                self.open_dialog = true;
                ui.close();
            }
            ui.separator();
            if ui
                .checkbox(&mut self.reduced_motion, "Reduce motion")
                .changed()
            {
                ui.ctx().style_mut_of(egui::Theme::Dark, |s| {
                    s.animation_time = if self.reduced_motion { 0. } else { 0.12 }
                });
            }
            ui.checkbox(&mut self.live, "Watch recording");
            ui.separator();
            ui.label("Space · Play / pause");
            ui.label("Left / Right · Checkpoint");
            ui.label("Ctrl + 1–6 · Workspace");
        })
        .response
        .on_hover_text("Application settings and keyboard shortcuts");
    }

    pub(super) fn recording_dialog(&mut self, ctx: &egui::Context) {
        if !self.open_dialog {
            if let Some(worker) = &mut self.browser_worker {
                worker.cancel();
            }
            return;
        }
        if self.open_path.is_empty() {
            self.open_path = std::env::current_dir()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
        }
        if self.browser_path != self.open_path {
            self.browser_path.clone_from(&self.open_path);
            self.browser_folders.clear();
            self.browser_notice = None;
        }
        if self.browser_worker.is_none() {
            match directory_browser::DirectoryBrowser::new(ctx.clone()) {
                Ok(worker) => self.browser_worker = Some(worker),
                Err(error) => self.browser_notice = Some(error.to_string()),
            }
        }
        if let Some(worker) = &mut self.browser_worker
            && let Some(listing) = worker.poll(&self.browser_path, ctx)
        {
            self.browser_folders = listing.folders;
            self.browser_notice = listing.notice;
        }
        let mut visible = self.open_dialog;
        let mut chosen = None;
        egui::Window::new("Open recording")
            .open(&mut visible)
            .collapsible(false)
            .default_width(540.)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Up").clicked()
                        && let Some(parent) = Path::new(&self.open_path).parent()
                    {
                        self.open_path = parent.to_string_lossy().into_owned();
                    }
                    ui.add(
                        egui::TextEdit::singleline(&mut self.open_path)
                            .desired_width(f32::INFINITY),
                    );
                });
                egui::ScrollArea::vertical()
                    .max_height(280.)
                    .show(ui, |ui| {
                        for folder in &self.browser_folders {
                            if ui
                                .selectable_label(
                                    false,
                                    folder
                                        .file_name()
                                        .unwrap_or(folder.as_os_str())
                                        .to_string_lossy(),
                                )
                                .clicked()
                            {
                                self.open_path = folder.to_string_lossy().into_owned();
                            }
                        }
                    });
                if let Some(notice) = &self.browser_notice {
                    ui.weak(notice);
                }
                ui.separator();
                if ui.button("Open directory").clicked() {
                    chosen = Some(self.open_path.clone());
                }
            });
        self.open_dialog = visible;
        if let Some(path) = chosen {
            self.request_directory(Path::new(&path), ctx);
            self.open_dialog = false;
        }
    }

    pub(super) fn reference_control(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let recorded = self.view == View::Value
            || self.view == View::PolicyMap
                && self
                    .policy_pane
                    .request
                    .as_ref()
                    .and_then(|request| request.reference.as_ref())
                    .is_some_and(|source| source.identity().recorded);
        crate::landscape::popover(ui, "Reference source", |ui| {
            ui.set_max_width(520.);
            let typed = typed_workspace::typed_run(run);
            let follow_current = typed
                && crate::reference::provider_role(self.comparison.reference_role())
                && self
                    .comparison
                    .reference_run()
                    .is_some_and(|reference| reference.root() == run.root());
            self.comparison
                .reference_controls(ui, &self.catalogue, recorded, typed, follow_current)
        })
        .transpose()?;
        if let Some(reference) = self.comparison.reference_run() {
            let role = self.comparison.reference_role();
            let revision = self.comparison.reference_revision();
            let label = if crate::reference::provider_role(role) {
                short(
                    &crate::presentation::display_identifier(
                        role.strip_prefix("reference:").unwrap_or(role),
                    ),
                    23,
                )
            } else if reference.root() == run.root() {
                format!("k {revision}")
            } else {
                short(
                    &crate::presentation::display_identifier(
                        &reference.recording().manifest().run_id,
                    ),
                    23,
                )
            };
            ui.weak(label).on_hover_text(format!(
                "{} · {} · k {revision}",
                reference.recording().manifest().run_id,
                role
            ));
        }
        Ok(())
    }

    pub(super) fn selection_bar(&mut self, ui: &mut egui::Ui, run: &RunIndex, switched: bool) {
        let previous_role = self.selection.model_role.clone();
        let compact = ui.available_width() < 680.;
        ui.horizontal(|ui| {
            ui.set_min_height(28.);
            let digits = run.revisions().last().map_or(1, |k|k.to_string().len());
            let (checkpoint_rect, _) = ui.allocate_exact_size(
                egui::vec2(((digits+5) as f32*8.).clamp(88.,200.),26.),egui::Sense::hover());
            let input_width = if compact { 0. } else { 230. };
            let width = (ui.available_width() - input_width).clamp(1., 950.);
            ui.allocate_ui_with_layout(egui::vec2(width,26.),egui::Layout::left_to_right(egui::Align::Center),|ui| {
                let result = if switched {
                    self.transport.show_snapshots(ui,run,&mut self.selection,
                        crate::timeline_ui::SnapshotPresentation { positions:&[],published:None,ready:false,checking:true,error:None }).map(|_| ())
                } else {
                    self.snapshot_transport(ui,run)
                };
                if let Err(error) = result { self.error = Some(error.to_string()); }
            });
            if !compact {
                self.selected_input_control(ui, run);
            }
            // Render after transport updates, so the requested revision cannot
            // lag the timeline by one frame during playback.
            ui.scope_builder(egui::UiBuilder::new().max_rect(checkpoint_rect), |ui| {
                let mut requested = self.selection.revision;
                egui::ComboBox::from_id_salt("direct-checkpoint")
                    .width((checkpoint_rect.width() - 24.).max(48.))
                    .selected_text(format!("k {requested}"))
                    .show_ui(ui, |ui| {
                        egui::ScrollArea::vertical().max_height(280.).show_rows(
                            ui, 24., run.revisions().len(), |ui, range| {
                                for &revision in &run.revisions()[range] {
                                    ui.selectable_value(&mut requested, revision, format!("k {revision}"));
                                }
                            },
                        );
                    }).response.on_hover_text("Requested learner checkpoint; the displayed source remains identified while a new frame is prepared.");
                if requested != self.selection.revision
                    && let Err(error) = self.selection.select_revision(run, requested)
                {
                    self.error = Some(error.to_string());
                }
            });
        });
        if compact {
            ui.horizontal(|ui| self.selected_input_control(ui, run));
        }
        if previous_role != self.selection.model_role {
            self.workspace_defaults = None;
            self.learning_pane.clear();
        }
    }

    fn selected_input_control(&mut self, ui: &mut egui::Ui, run: &RunIndex) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let source = match &self.selection.input {
                SelectedInput::ProbeRow { set, row, .. } => format!("{} · row {row}", short(&crate::presentation::display_identifier(set), 20)),
                SelectedInput::TrajectoryStep { sequence, step } => {
                    run.by_sequence(*sequence).map(|i| {
                        let entry = &run.recording().entries()[i];
                        format!("{} · k{} · step {step}", short(&crate::presentation::display_identifier(entry.trajectory.as_deref().unwrap_or("Episode")), 20), entry.revision)
                    }).unwrap_or_else(|| format!("Episode unavailable · step {step}"))
                }
                SelectedInput::None => "No input".into(),
            };
            let summary = if matches!(self.view, View::PolicyMap | View::Value) {
                format!("Anchor · {source}")
            } else {
                source
            };
            crate::landscape::popover(ui, "Input", |ui| {
                ui.set_max_width(460.);
                self.input_controls(ui, run);
                if self.view == View::PolicyMap && !typed_workspace::typed_run(run) {
                    ui.separator();
                    self.policy_mode_controls(ui, run);
                }
            });
            ui.add(egui::Label::new(egui::RichText::new(summary).small().color(crate::theme::MUTED)).truncate())
                .on_hover_text(format!("Recorded anchor: {:?}\n{}\nContinuous plot interventions stay local; their complete actual inputs are in Selected > Details.", self.selection.input, run.root().display()));
        });
    }
}

fn short(value: &str, length: usize) -> String {
    if value.chars().count() <= length {
        value.into()
    } else {
        format!("{}…", value.chars().take(length - 1).collect::<String>())
    }
}
