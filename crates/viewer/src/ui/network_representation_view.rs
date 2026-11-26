//! One representation plot and its exact recorded-input inspector.
use super::{RepresentationControls, RepresentationFrame, SelectedInput};
use crate::theme;
use eframe::egui::{self, RichText};
use egui_plot::{Legend, MarkerShape, Plot, PlotPoint, Points};

pub(super) fn show(
    ui: &mut egui::Ui,
    pane: &mut RepresentationControls,
    selection: &SelectedInput,
    model_architecture: &str,
) -> Option<SelectedInput> {
    pane.painted = false;
    let frame = pane.worker.frame.as_ref();
    let mut query_changed = false;
    ui.horizontal_wrapped(|ui| {
        if let Some(frame) = frame {
            egui::ComboBox::from_id_salt("representation-network")
                .selected_text(&frame.network)
                .show_ui(ui, |ui| {
                    for network in frame.choices.keys() {
                        if ui
                            .selectable_label(network == &frame.network, network)
                            .clicked()
                            && network != &frame.network
                        {
                            pane.network = Some(network.clone());
                            pane.layer = None;
                            pane.historical = false;
                            query_changed = true;
                        }
                    }
                })
                .response
                .on_hover_text(model_architecture);
            egui::ComboBox::from_id_salt("representation-layer")
                .selected_text(&frame.layer)
                .show_ui(ui, |ui| {
                    for layer in frame.choices.get(&frame.network).into_iter().flatten() {
                        if ui.selectable_label(layer == &frame.layer, layer).clicked()
                            && layer != &frame.layer
                        {
                            pane.layer = Some(layer.clone());
                            query_changed = true;
                        }
                    }
                });
        }
        if pane.historical || frame.is_some_and(|frame| frame.historical_available) {
            query_changed |= ui
                .checkbox(&mut pane.historical, "Restore checkpoint")
                .on_hover_text(
                    "Evaluate the retained model on the complete recorded inputs and actions.",
                )
                .changed();
        }
        if frame.is_some() {
            ui.separator();
            ui.selectable_value(&mut pane.heatmap, false, "Representation PCA");
            ui.selectable_value(&mut pane.heatmap, true, "Activation heatmap");
        }
    });
    if query_changed {
        ui.ctx().request_repaint();
        return None;
    }
    let Some(frame) = frame else {
        if let Some(error) = &pane.worker.error {
            ui.label(RichText::new(error).color(theme::WARNING));
            if pane.network.is_some() && ui.button("Use default model").clicked() {
                pane.network = None;
                pane.layer = None;
                ui.ctx().request_repaint();
            }
        } else {
            ui.spinner();
            ui.weak("Preparing representation…");
        }
        return None;
    };
    pane.feature = pane.feature.min(frame.width - 1);
    let heatmap = pane.heatmap || frame.points.is_empty();
    let count = if frame.width == frame.source_width {
        format!("{} features", frame.width)
    } else {
        format!("{} of {} features", frame.width, frame.source_width)
    };
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(&frame.population).strong());
        ui.weak(format!(
            "{} inputs · {count} · k {}",
            frame.row_ids.len(),
            frame.revision
        ));
        if frame.historical {
            if frame.historical_fixed_reference {
                ui.weak("Restored · reference actions held fixed");
            } else {
                ui.weak("Restored · recorded actions replayed");
            }
        }
        ui.weak(frame.input_lineage).on_hover_text(&frame.methodology);
        if !heatmap {
            ui.checkbox(&mut pane.colour, "Colour by observation 0");
            if let Some(revision) = frame.basis_revision {
                let basis = if frame.reference_points.is_some() { "Fixed PCA basis" } else { "Local PCA basis" };
                ui.weak(format!("{basis} · k {revision}"))
                    .on_hover_text(frame.basis_recording.as_deref().unwrap_or_default());
            }
        }
        if let Some(cka) = frame.cka {
            ui.label(format!("Linear CKA {cka:.3}")).on_hover_text(
                "Centered kernel alignment of the full feature matrices using identical ordered state or state–action inputs.",
            );
        } else if let Some(note) = &frame.comparison_note {
            ui.weak("Fixed-input comparison unavailable").on_hover_text(note);
        }
        if let Some(note) = &frame.projection_note {
            ui.weak("PCA unavailable").on_hover_text(note);
        }
    });
    let selected = frame.row_for_input(selection);
    let height = (ui.available_height() - 41.).max(190.);
    let clicked = if heatmap {
        show_heatmap(ui, frame, selected, &mut pane.feature, height)
    } else if ui.available_width() >= 900. && frame.row_ids.len() <= 32 && frame.width <= 256 {
        ui.columns(2, |columns| {
            columns[0].label(RichText::new("Representation PCA").size(16.).strong());
            let pca = show_pca(
                &mut columns[0],
                frame,
                selected,
                pane.colour,
                (height - 24.).max(160.),
            );
            columns[1].label(RichText::new("Activation heatmap").size(16.).strong());
            let heatmap = show_heatmap(
                &mut columns[1],
                frame,
                pca.or(selected),
                &mut pane.feature,
                (height - 24.).max(160.),
            );
            heatmap.or(pca)
        })
    } else {
        show_pca(ui, frame, selected, pane.colour, height)
    };
    let active = clicked.or(selected);
    if let Some(row) = active {
        let values = &frame.values[row * frame.width..(row + 1) * frame.width];
        let feature = pane.feature.min(frame.width - 1);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(format!("Selected input {}", frame.row_ids[row])).strong());
            if let Some(action) = frame.actions.get(row) {
                ui.weak(action);
            }
            ui.label(format!(
                "Feature {}  {:.4}",
                frame.feature_ids[feature], values[feature]
            ));
            let rms = (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64)
                .sqrt();
            ui.weak(format!("Activation RMS  {rms:.4}"));
            if frame.width > 1 {
                ui.add(
                    egui::DragValue::new(&mut pane.feature)
                        .range(0..=frame.width - 1)
                        .prefix("Column "),
                );
            }
        });
    } else {
        ui.weak(match selection {
            SelectedInput::TrajectoryStep { .. } => {
                "The selected trajectory step is outside this projection."
            }
            SelectedInput::ProbeRow { .. } => {
                "The selected recorded input is outside this projection."
            }
            SelectedInput::None => "No input selected",
        });
    }
    pane.painted = true;
    clicked.map(|row| frame.input(row))
}

fn show_pca(
    ui: &mut egui::Ui,
    frame: &RepresentationFrame,
    selected: Option<usize>,
    colour: bool,
    height: f32,
) -> Option<usize> {
    let axis = |index: usize| {
        frame.explained.get(index).map_or_else(
            || format!("PC {}", index + 1),
            |variance| format!("PC {} ({:.1}%)", index + 1, variance * 100.),
        )
    };
    let mut plot = Plot::new((
        "representation-pca",
        &frame.network,
        &frame.layer,
        &frame.population,
    ))
    .x_axis_label(axis(0))
    .y_axis_label(axis(1))
    .grid_color(theme::GRID)
    .grid_spacing(64.0..=152.0)
    .grid_fade(0.82)
    .x_grid_spacer(egui_plot::log_grid_spacer(2))
    .y_grid_spacer(egui_plot::log_grid_spacer(2))
    .x_axis_formatter(crate::landscape::axis_tick)
    .y_axis_formatter(crate::landscape::axis_tick)
    .height(height)
    .sense(egui::Sense::click_and_drag());
    if !colour {
        plot = plot.legend(Legend::default().text_style(egui::TextStyle::Small));
    }
    let response = plot.show(ui, |plot| {
        if let Some(reference) = &frame.reference_points {
            plot.points(
                Points::new("Reference inputs", reference.clone())
                    .radius(4.)
                    .shape(MarkerShape::Diamond)
                    .filled(false)
                    .color(theme::REFERENCE),
            );
        }
        if colour {
            let lo = frame
                .observation_coordinate
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let hi = frame
                .observation_coordinate
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut bins = [const { Vec::new() }; 9];
            for (&point, &value) in frame.points.iter().zip(&frame.observation_coordinate) {
                let fraction = if hi > lo {
                    (value - lo) / (hi - lo)
                } else {
                    0.5
                };
                bins[(fraction * 8.).round() as usize].push(point);
            }
            for (bin, points) in bins
                .into_iter()
                .enumerate()
                .filter(|(_, points)| !points.is_empty())
            {
                let fraction = bin as f32 / 8.;
                let label = format!(
                    "Observation 0 ≈ {:.3}",
                    lo + (hi - lo) * f64::from(fraction)
                );
                plot.points(
                    Points::new(label, points)
                        .radius(4.)
                        .color(theme::REFERENCE.lerp_to_gamma(theme::CURRENT, fraction)),
                );
            }
        } else {
            plot.points(
                Points::new("Current inputs", frame.points.clone())
                    .radius(4.)
                    .color(theme::CURRENT),
            );
        }
        if let Some(row) = selected {
            plot.points(
                Points::new("Selected input", vec![frame.points[row]])
                    .shape(MarkerShape::Circle)
                    .radius(7.)
                    .filled(false)
                    .color(theme::SELECTED),
            );
        }
    });
    if !response.response.clicked() {
        return None;
    }
    let mouse = response.response.interact_pointer_pos()?;
    frame
        .points
        .iter()
        .enumerate()
        .chain(
            frame
                .reference_points
                .iter()
                .flat_map(|points| points.iter().enumerate()),
        )
        .map(|(row, &[x, y])| {
            let point = response
                .transform
                .position_from_point(&PlotPoint::new(x, y));
            (row, point.distance(mouse))
        })
        .filter(|(_, distance)| *distance <= 14.)
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(row, _)| row)
}

fn show_heatmap(
    ui: &mut egui::Ui,
    frame: &RepresentationFrame,
    selected: Option<usize>,
    feature: &mut usize,
    height: f32,
) -> Option<usize> {
    const COLS: usize = 40;
    let rows = frame.row_ids.len();
    let feature_start = (*feature / COLS) * COLS;
    let columns = (frame.width - feature_start).min(COLS);
    let scale = crate::landscape::LandscapeScale::new(
        [-frame.absolute_max, frame.absolute_max],
        crate::landscape::Palette::Difference,
    );
    let (bounds, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::hover(),
    );
    let rect = egui::Rect::from_min_max(
        bounds.min + egui::vec2(60., 8.),
        bounds.max - egui::vec2(92., 34.),
    );
    let response = ui.interact(
        rect,
        ui.make_persistent_id("activation-grid"),
        egui::Sense::click(),
    );
    let painter = ui.painter_at(bounds);
    let cell = egui::vec2(rect.width() / columns as f32, rect.height() / rows as f32);
    let hovered = response.hover_pos().map(|position| {
        let relative = position - rect.min;
        let row = (relative.y / cell.y) as usize;
        let col = (relative.x / cell.x) as usize;
        (row.min(rows - 1), col.min(columns - 1) + feature_start)
    });
    let mut mesh = egui::Mesh::default();
    for row in 0..rows {
        for col in 0..columns {
            let value = frame.values[row * frame.width + feature_start + col];
            let position = rect.min + egui::vec2(col as f32 * cell.x, row as f32 * cell.y);
            mesh.add_colored_rect(
                egui::Rect::from_min_size(position, cell),
                scale.color(value),
            );
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    let x_tick_step = (45. / cell.x).ceil() as usize;
    for col in (0..columns).step_by(x_tick_step.max(1)) {
        painter.text(
            egui::pos2(
                rect.left() + (col as f32 + 0.5) * cell.x,
                rect.bottom() + 3.,
            ),
            egui::Align2::CENTER_TOP,
            frame.feature_ids[feature_start + col],
            egui::FontId::proportional(11.),
            theme::MUTED,
        );
    }
    let y_tick_step = (24. / cell.y).ceil() as usize;
    for row in (0..rows).step_by(y_tick_step.max(1)) {
        painter.text(
            egui::pos2(rect.left() - 5., rect.top() + (row as f32 + 0.5) * cell.y),
            egui::Align2::RIGHT_CENTER,
            frame.row_ids[row],
            egui::FontId::proportional(11.),
            theme::MUTED,
        );
    }
    painter.text(
        egui::pos2(rect.center().x, bounds.bottom()),
        egui::Align2::CENTER_BOTTOM,
        "Feature ID",
        egui::FontId::proportional(13.),
        theme::TEXT,
    );
    let y_title = painter.layout_no_wrap(
        "Input ID".into(),
        egui::FontId::proportional(13.),
        theme::TEXT,
    );
    let mut title = egui::epaint::TextShape::new(
        egui::pos2(bounds.left(), rect.center().y + y_title.size().x * 0.5),
        y_title,
        theme::TEXT,
    );
    title.angle = -std::f32::consts::FRAC_PI_2;
    painter.add(title);
    crate::landscape::color_bar(ui, rect, scale, "Activation");
    if let Some(row) = selected {
        let top = rect.min.y + row as f32 * cell.y;
        painter.rect_stroke(
            egui::Rect::from_min_size(
                egui::pos2(rect.min.x, top),
                egui::vec2(rect.width(), cell.y),
            ),
            0.,
            egui::Stroke::new(1., theme::SELECTED),
            egui::StrokeKind::Inside,
        );
    }
    let clicked = response.clicked();
    if let Some((row, band)) = hovered {
        response.on_hover_text(format!(
            "Input {} · feature {} · {:.4}",
            frame.row_ids[row],
            frame.feature_ids[band],
            frame.values[row * frame.width + band]
        ));
        if clicked {
            *feature = band;
            return Some(row);
        }
    }
    None
}
