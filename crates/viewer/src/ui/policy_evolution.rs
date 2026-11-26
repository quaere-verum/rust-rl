//! Fixed-population policy change over recorded learner revisions.
use super::*;
use crate::learning::{ChangeKind, LearningPoint};
use egui_plot::{Line, MarkerShape, Plot, Points};

fn change_label(kind: ChangeKind) -> &'static str {
    match kind {
        ChangeKind::NormalizedRms => "Normalized action RMS",
        ChangeKind::ReferenceKl => "KL(reference || current)",
        ChangeKind::DiscreteDisagreement => "Action disagreement",
    }
}

fn nearest_revision(points: &[[f64; 2]], revisions: &[u64], x: f64, y: f64) -> Option<u64> {
    points
        .iter()
        .zip(revisions)
        .min_by(|(a, _), (b, _)| {
            let distance = |p: &[f64; 2]| (p[0] - x).hypot(p[1] - y);
            distance(a).total_cmp(&distance(b))
        })
        .map(|(_, &revision)| revision)
}

fn change_chart(
    ui: &mut egui::Ui,
    points: &[LearningPoint],
    kind: ChangeKind,
    selected: u64,
    height: f32,
) -> Option<u64> {
    let caption = change_label(kind);
    let quantity = match kind {
        ChangeKind::NormalizedRms => "Normalized RMS",
        ChangeKind::ReferenceKl => "KL",
        ChangeKind::DiscreteDisagreement => "Fraction",
    };
    let response = Plot::new("policy-change-history")
        .height(height)
        .x_axis_label("Learner revision")
        .y_axis_label(quantity)
        .grid_color(crate::theme::GRID)
        .x_axis_formatter(crate::landscape::axis_tick)
        .y_axis_formatter(crate::landscape::axis_tick)
        .show(ui, |plot| {
            for segment in points
                .split(|p| p.value.is_none_or(|v| !v.is_finite()))
                .filter(|s| !s.is_empty())
            {
                let values = segment
                    .iter()
                    .map(|p| [p.revision as f64, p.value.unwrap()])
                    .collect::<Vec<_>>();
                if values.len() == 1 {
                    plot.points(Points::new(caption, values).color(ACCENT).radius(3.5));
                } else {
                    plot.line(
                        Line::new(caption, values)
                            .allow_band(false)
                            .color(ACCENT)
                            .width(1.7),
                    );
                }
            }
            plot.vline(
                egui_plot::VLine::new("Selected checkpoint", selected as f64)
                    .color(crate::theme::SELECTED),
            );
            plot.pointer_coordinate()
        });
    if response.response.clicked()
        && let Some(pos) = response.inner
    {
        return points
            .iter()
            .filter(|p| p.value.is_some_and(f64::is_finite))
            .min_by(|a, b| {
                (a.revision as f64 - pos.x)
                    .abs()
                    .total_cmp(&(b.revision as f64 - pos.x).abs())
            })
            .map(|p| p.revision);
    }
    None
}

impl Laboratory {
    fn policy_population(&self) -> Option<&str> {
        match &self.selection.input {
            SelectedInput::ProbeRow { set, .. } => Some(set),
            _ => None,
        }
    }

    pub(super) fn policy_evolution_controls(&mut self, ui: &mut egui::Ui) {
        let previous = self.policy_evolution;
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.policy_evolution, false, "Mapping");
            ui.selectable_value(&mut self.policy_evolution, true, "Evolution");
        });
        if previous != self.policy_evolution {
            self.transport.clear_snapshot_state();
            if self.policy_evolution {
                self.leave_prepared_policy();
                self.interactive.retire_session();
                self.policy_pane.clear();
            } else {
                self.learning_pane.clear();
            }
        }
    }

    pub(super) fn policy_evolution_transport(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        let worker = &self.learning_pane.worker;
        let frame = worker.frame().filter(|f| {
            f.root == *run.root()
                && f.role == self.selection.model_role
                && f.policy_population.as_deref() == self.policy_population()
        });
        let mut positions = frame.map_or_else(Vec::new, |f| {
            f.policy_changes
                .iter()
                .filter(|p| p.value.is_some())
                .map(|p| p.revision)
                .collect::<Vec<_>>()
        });
        positions.dedup();
        let exact = positions.contains(&self.selection.revision);
        let published = (frame.is_some() && exact).then_some(self.selection.revision);
        self.transport.hold_unavailable_snapshot(
            (!positions.is_empty() && !exact).then_some(self.selection.revision),
        );
        self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            crate::timeline_ui::SnapshotPresentation {
                positions: &positions,
                published,
                ready: frame.is_some(),
                checking: worker.pending() || frame.is_none() && worker.error().is_none(),
                error: worker.error(),
            },
        )?;
        Ok(())
    }

    pub(super) fn show_policy_evolution(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<Option<u64>> {
        let population = self.policy_population().map(str::to_owned);
        let worker = &mut self.learning_pane.worker;
        worker.poll();
        worker.request_history(run, &self.selection.model_role, population.as_deref())?;
        worker.poll();
        let Some(frame) = worker.frame().filter(|f| {
            f.root == *run.root()
                && f.role == self.selection.model_role
                && f.policy_population == population
        }) else {
            ui.weak(
                worker
                    .error()
                    .unwrap_or("Preparing the fixed policy population…"),
            );
            return Ok(None);
        };
        let revision = self.selection.revision;
        let mut clicked = None;
        if let Some(pca) = &frame.policy_space {
            ui.horizontal_wrapped(|ui| {
                ui.strong("Policy function PCA");
                ui.weak(format!("{} fixed inputs", pca.population_rows))
                    .on_hover_text(pca.representation);
                if pca.omitted_checkpoints > 0 {
                    ui.weak(format!(
                        "{} checkpoints omitted by the PCA budget",
                        pca.omitted_checkpoints
                    ));
                }
            });
            let values = pca.points.iter().map(|p| p.coordinates).collect::<Vec<_>>();
            let revisions = pca.points.iter().map(|p| p.revision).collect::<Vec<_>>();
            let response = Plot::new("policy-function-pca")
                .height((ui.available_height() - 191.).max(160.))
                .x_axis_label(format!("PC1 ({:.0}%)", 100. * pca.explained[0]))
                .y_axis_label(format!("PC2 ({:.0}%)", 100. * pca.explained[1]))
                .grid_color(crate::theme::GRID)
                .x_axis_formatter(crate::landscape::axis_tick)
                .y_axis_formatter(crate::landscape::axis_tick)
                .show(ui, |plot| {
                    plot.line(
                        Line::new("Learning trajectory", values.clone())
                            .allow_band(false)
                            .color(crate::theme::MUTED)
                            .width(1.3),
                    );
                    plot.points(
                        Points::new("Recorded checkpoint", values.clone())
                            .color(ACCENT)
                            .radius(3.5),
                    );
                    if let Some(point) = pca.points.iter().find(|p| p.revision == revision) {
                        plot.points(
                            Points::new("Selected checkpoint", vec![point.coordinates])
                                .shape(MarkerShape::Diamond)
                                .color(crate::theme::SELECTED)
                                .radius(6.),
                        );
                    }
                    plot.pointer_coordinate()
                });
            if response.response.clicked()
                && let Some(point) = response.inner
            {
                clicked = nearest_revision(&values, &revisions, point.x, point.y);
            }
        } else if let Some(reason) = &frame.policy_space_unavailable {
            ui.weak("Policy PCA unavailable").on_hover_text(reason);
        }
        if let Some(anchor) = &frame.anchor {
            let label = change_label(anchor.kind);
            ui.horizontal_wrapped(|ui| {
                ui.strong(label);
                ui.weak(format!("Reference k {}", anchor.revision))
                    .on_hover_text(&anchor.population);
                let point = frame
                    .policy_changes
                    .iter()
                    .rfind(|p| p.revision == revision);
                if let Some(p) = point {
                    if p.value.is_some_and(f64::is_infinite) {
                        ui.weak("∞ on this checkpoint");
                    } else if let Some(reason) = &p.reason {
                        ui.weak("Comparison unavailable").on_hover_text(reason);
                    }
                }
            });
            let change_clicked = change_chart(
                ui,
                &frame.policy_changes,
                anchor.kind,
                revision,
                if frame.policy_space.is_some() {
                    152.
                } else {
                    (ui.available_height() - 24.).max(155.)
                },
            );
            clicked = clicked.or(change_clicked);
        } else {
            ui.weak("No comparable fixed policy population");
        }
        Ok(clicked)
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn policy_evolution_capture_ready(&self) -> bool {
        !self.learning_pane.worker.pending() && self.learning_pane.worker.frame().is_some()
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn policy_evolution_capture_evidence(&self) -> serde_json::Value {
        serde_json::json!(self.learning_pane.worker.frame().and_then(|f| f.policy_space.as_ref()).map(|pca| {
            serde_json::json!({"population_rows":pca.population_rows,"representation":pca.representation,
                "explained":pca.explained,"checkpoints":pca.points.len(),"omitted":pca.omitted_checkpoints})
        }))
    }
}
