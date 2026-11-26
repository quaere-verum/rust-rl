use super::*;
use eframe::egui::Color32;
use egui_plot::{Bar, BarChart, Line, Plot, VLine};
fn profile_key(ui: &mut egui::Ui, label: &str, color: Color32, dashed: bool, width: f32) {
    let galley = ui.painter().layout_no_wrap(
        label.into(),
        egui::TextStyle::Small.resolve(ui.style()),
        color,
    );
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(24. + galley.size().x, galley.size().y.max(16.)),
        egui::Sense::hover(),
    );
    let start = rect.left_center();
    let stroke = egui::Stroke::new(width, color);
    if dashed {
        for offset in [0., 7., 14.] {
            ui.painter().line_segment(
                [
                    start + egui::vec2(offset, 0.),
                    start + egui::vec2(offset + 4., 0.),
                ],
                stroke,
            );
        }
    } else {
        ui.painter()
            .line_segment([start, start + egui::vec2(18., 0.)], stroke);
    }
    ui.painter().galley(
        egui::pos2(rect.left() + 24., rect.center().y - galley.size().y * 0.5),
        galley,
        color,
    );
}
fn card(ui: &mut egui::Ui, label: &str, value: f64) -> egui::Response {
    ui.horizontal(|ui| {
        let title = ui.weak(label);
        let number = ui
            .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(format!("{value:.5}")).size(17.).color(
                    if label.contains("empirical") || label == "Return-to-go" {
                        crate::theme::EMPIRICAL
                    } else if label == "Reference" {
                        crate::theme::REFERENCE
                    } else if label == "Difference" {
                        crate::theme::SELECTED
                    } else {
                        crate::theme::CURRENT
                    },
                ))
            })
            .inner;
        title.union(number)
    })
    .inner
}
impl ValueView {
    fn return_profile_plot(
        &self,
        ui: &mut egui::Ui,
        point: &PreparedValuePoint,
        cdf: &[[f64; 2]],
        height: f32,
    ) {
        let quantiles = self.return_profile == ReturnProfileMode::Quantile;
        let coordinates = |cdf: &[[f64; 2]]| -> Vec<[f64; 2]> {
            if quantiles {
                cdf.iter()
                    .map(|&[value, probability]| [probability, value])
                    .collect()
            } else {
                cdf.to_vec()
            }
        };
        ui.horizontal_wrapped(|ui| {
            profile_key(ui, "Predicted", crate::theme::CURRENT, false, 1.);
            if point
                .reference_profile
                .as_ref()
                .and_then(PreparedValueProfile::cdf)
                .is_some()
            {
                profile_key(ui, "Reference", crate::theme::REFERENCE, true, 1.);
            }
            if point.empirical.is_some() && self.empirical_cdf_admitted {
                profile_key(ui, "Empirical", crate::theme::EMPIRICAL, false, 2.);
            }
        });
        let plot = Plot::new(("value-return-profile", quantiles))
            .height(height)
            .x_grid_spacer(egui_plot::log_grid_spacer(10))
            .y_grid_spacer(egui_plot::log_grid_spacer(2))
            .x_axis_formatter(landscape::axis_tick)
            .y_axis_formatter(landscape::axis_tick)
            .show_background(false)
            .grid_color(crate::theme::GRID);
        let plot = if quantiles {
            plot.include_x(0.)
                .include_x(1.)
                .x_axis_label("Probability")
                .y_axis_label("Return quantile")
        } else {
            plot.include_y(0.)
                .include_y(1.)
                .x_axis_label("Return")
                .y_axis_label(if height < 140. {
                    "Probability"
                } else {
                    "Cumulative probability"
                })
        };
        plot.show(ui, |p| {
            p.line(
                Line::new("Predicted", coordinates(cdf))
                    .allow_band(false)
                    .color(crate::theme::CURRENT),
            );
            if let Some(cdf) = point
                .reference_profile
                .as_ref()
                .and_then(PreparedValueProfile::cdf)
            {
                p.add(dashes::DashedLine::curve(
                    Line::new("Reference", coordinates(cdf))
                        .allow_band(false)
                        .color(crate::theme::REFERENCE),
                    1.,
                    self.dash_admission.clone(),
                ));
            }
            if !quantiles && let Some(value) = point.observed_return {
                p.vline(VLine::new("Observed return", value).color(crate::theme::EMPIRICAL));
            }
            if self.empirical_cdf_admitted
                && let Some(evidence) = &point.empirical
            {
                p.line(
                    Line::new("Empirical", coordinates(&evidence.cdf))
                        .allow_band(false)
                        .width(2.)
                        .color(crate::theme::EMPIRICAL),
                );
            }
        });
    }

    pub(super) fn selected(
        &mut self,
        ui: &mut egui::Ui,
        frame: Option<&PreparedValueFrame>,
        compact: bool,
    ) {
        let Some(point) = frame.and_then(|f| f.selected.as_ref()) else {
            ui.weak(if self.terminal {
                "Unavailable"
            } else {
                "Loading…"
            });
            return;
        };
        if !self.ready && self.pin.as_ref().is_some_and(|pin| !pin.same(point.pin())) {
            ui.colored_label(crate::theme::WARNING, "Previous input")
                .on_hover_text(presentation::selected_input(frame.unwrap(), point));
        }
        if !self.focus_distribution && !compact {
            ui.strong("Selected");
            if point.final_observation {
                ui.small(
                "Final recorded observation · critic prediction only; no recorded next action or return-to-go",
            );
            }
            if let Some(loss) = point.decision_loss {
                card(ui, "Decision loss", loss.loss);
                ui.label(format!(
                    "V* = {:.5}   E Q* = {:.5}",
                    loss.optimal_value, loss.expected_action_value
                ));
                ui.small("Loss from this decision assuming optimal continuation; not the full-policy performance gap.");
                if let Some(policy) = point.current_policy() {
                    ui.label(format!(
                        "Current policy {} · actual k={}",
                        policy.decision_subject(),
                        policy.identity().revision
                    ));
                }
                if let Some(reference) = point.reference() {
                    ui.label(format!(
                        "Reference {} · {} · k={}",
                        reference.identity().name,
                        reference.model_role(),
                        reference.identity().revision
                    ));
                    reference.reference_quality_ui(ui);
                }
                if let Some(row) = point.recorded_row {
                    ui.small(format!(
                        "Selected recorded row {row} · input k={}",
                        point.recorded_revision.unwrap_or(0)
                    ));
                }
                egui::CollapsingHeader::new("Details").default_open(crate::theme::inspection_default(ui)).show(ui, |ui| {
                ui.label(format!("Complete state {:?}\nLegal mask {:?}", point.input.dense_values(), point.input.dense_mask()));
                if let Some(policy) = point.current_policy() { ui.label(format!("Current policy source {:?}", policy.identity())); }
                if let Some(reference) = point.reference() { ui.label(format!("Reference source {:?}", reference.identity())); }
                ui.small("Loss assumes optimal continuation after this decision. It is not the full-policy performance gap.");
            });
                return;
            }
            if let Some(reference) = point.reference_value
                && point.value.is_finite()
            {
                let label =
                    source_output_label(Some(point.current()), frame.unwrap().grid.output());
                card(ui, label, point.value)
                    .on_hover_text(presentation::selected_input(frame.unwrap(), point));
                card(
                    ui,
                    if frame.unwrap().grid.empirical_reference() {
                        "Reference · empirical"
                    } else {
                        "Reference"
                    },
                    reference,
                )
                .on_hover_text(presentation::selected_input(frame.unwrap(), point));
                card(ui, "Difference", point.value - reference)
                    .on_hover_text(presentation::selected_input(frame.unwrap(), point));
            } else if !matches!(
                point.prediction,
                ValuePrediction::Twin { .. } | ValuePrediction::Moments { .. }
            ) {
                let label =
                    source_output_label(Some(point.current()), frame.unwrap().grid.output());
                card(ui, label, point.value)
                    .on_hover_text(presentation::selected_input(frame.unwrap(), point));
            }
            if let ValuePrediction::Twin { q1, q2 } = &point.prediction
                && let (ValuePrediction::Scalar(a), ValuePrediction::Scalar(b)) = (&**q1, &**q2)
            {
                match (point.reference_value, frame.unwrap().grid.output()) {
                    (Some(_), ValueOutput::Q1) => {
                        card(ui, "Q2", *b);
                    }
                    (Some(_), ValueOutput::Q2) => {
                        card(ui, "Q1", *a);
                    }
                    _ => {
                        card(ui, "Q1", *a);
                        card(ui, "Q2", *b);
                    }
                }
                ui.small(format!("Disagreement {:.5}", (a - b).abs())).on_hover_text("Absolute difference between the two current critics; not a calibrated uncertainty estimate.");
            }
            if let Some(ReturnTarget::SoftActionValue { alpha, .. }) =
                &point.current().return_target
            {
                ui.small(format!("Snapshot entropy temperature α = {alpha:.5}"))
                .on_hover_text("Multiplier at this model revision. Training metrics identify the multiplier used to construct the preceding targets.");
            }
            ui.small(format!(
                "{} · actual k={}",
                point.current().identity().name,
                point.current().identity().revision
            ));
            if let Some(source) = frame
                .and_then(|f| f.grid.reference())
                .filter(|_| point.reference_value.is_some())
            {
                ui.small(format!(
                    "Reference · actual k={}",
                    source.identity().revision
                ));
            }
            if let Some(g) = frame.map(|f| &f.grid).filter(|g| g.visits.is_some()) {
                ui.label(format!("Visits {}",point.neighborhood_count)).on_hover_text(format!("Exact selected input neighborhood; fixed tolerance {}; {} compatible genuine decisions",g.slice.state.tolerance,g.visits.as_ref().map_or(0,|v|v.filtered_count)));
            }
            if let Some(value) = point.observed_return {
                card(ui, "Return-to-go", value).on_hover_text(format!(
                    "Actual observed return · recorded k={} · episode {} · step {}",
                    point.recorded_revision.unwrap_or(0),
                    point.recorded_episode().unwrap_or("?"),
                    point.recorded_row.unwrap_or(0)
                ));
                ui.small("Single recorded realization");
                if let Some(e) = point.recorded_evaluation() {
                    ui.small(format!(
                        "Episode {} · step {}",
                        point.recorded_episode().unwrap_or("?"),
                        point.recorded_row.unwrap_or(0)
                    ));
                    ui.small(format!("recorded k={} · γ={}",point.recorded_revision.unwrap_or(0),e.discount)).on_hover_text(format!("{}\nOne complete observed return from this exact immutable state/action. It is a single realization, not an estimation error or calibration statistic.", e.continuation_policy));
                }
            }
            if let Some(evidence) = &point.empirical {
                let condition = &evidence.condition;
                let summary = condition.summary;
                if !frame.is_some_and(|f| f.grid.empirical_reference())
                    || point.reference_value.is_none()
                {
                    card(
                        ui,
                        if summary.count > 1 {
                            "Conditional empirical mean"
                        } else {
                            "Single empirical observation"
                        },
                        summary.mean,
                    );
                }
                ui.small(if summary.count > 1 {
                    "Conditional estimate"
                } else {
                    "Single empirical observation"
                });
                ui.small(format!(
                    "n={}{} · actual k={}",
                    summary.count,
                    summary
                        .mean_standard_error
                        .map_or_else(String::new, |se| format!(" · mean SE {se:.5}")),
                    condition.revision
                ));
                ui.small(format!("{} continuation", if condition.stochastic {"stochastic"} else {"deterministic"}))
                .on_hover_text(format!("{}\nOriginal empirical record #{} · probe #{} · stable row {}\n{}\nFull observation {:?}\nMask {:?}\nFirst action {:?}\nSE is sample SD / sqrt(n), assuming independent futures. It measures uncertainty of the conditional mean, not return dispersion or critic uncertainty.",
                    condition.source_key.root.display(),condition.source_key.sequence,condition.probe_key.sequence,condition.row_id,condition.methodology,condition.input.dense_values(),condition.input.dense_mask(),condition.action));
            } else if let Some(reason) = &point.empirical_error {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(format!("Conditional evidence unavailable: {reason}"))
                            .small()
                            .color(crate::theme::MUTED),
                    )
                    .truncate(),
                )
                .on_hover_text(reason);
            }
            if let Some(grid) = frame
                .map(|f| &f.grid)
                .filter(|g| !g.empirical.conditions.is_empty())
            {
                egui::CollapsingHeader::new("Recorded empirical conditions").show(ui, |ui| {
                ui.small("Adopt an original measured input and complete first action. Other inputs remain unmeasured.");
                for condition in &grid.empirical.conditions {
                    if ui.button(format!("row {} · n={} · mean {:.4}", condition.row_id, condition.summary.count, condition.summary.mean)).clicked() {
                        let _ = self.adopt_empirical(condition);
                    }
                }
                if grid.empirical.omitted > 0 { ui.small(format!("{} sources outside bounded admission",grid.empirical.omitted)); }
            });
            }
            if point
                .profile
                .as_ref()
                .and_then(PreparedValueProfile::cdf)
                .is_none()
                && let Some(evidence) = &point.empirical
            {
                egui::CollapsingHeader::new("Empirical return distribution").show(ui, |ui| {
                if !self.empirical_cdf_admitted {
                    ui.small("Empirical CDF exceeds view admission; its sample count, mean and SE remain available.");
                    return;
                }
                ui.small("Observed conditional returns. Scalar or moment outputs do not specify a predicted CDF.");
                Plot::new("value-scalar-empirical-cdf").height(180.).x_axis_label("Return").y_axis_label("Empirical CDF")
                    .include_y(0.).include_y(1.).show(ui, |p| {
                        p.line(Line::new("Empirical CDF",evidence.cdf.clone()).allow_band(false).color(crate::theme::EMPIRICAL));
                        if frame.is_some_and(|f|matches!(f.grid.output(),ValueOutput::ExpectedReturn|ValueOutput::Q1|ValueOutput::Q2)) {
                            p.vline(VLine::new("Critic expected return",point.value).color(crate::theme::CURRENT));
                        }
                    });
            });
            }
        } else if self.focus_distribution {
            ui.weak(format!(
                "{} · actual k={}",
                point.current().identity().name,
                point.current().identity().revision
            ))
            .on_hover_text(presentation::selected_input(frame.unwrap(), point));
            if point.final_observation {
                ui.weak("Final recorded observation");
            }
        }
        match &point.profile {
            Some(
                PreparedValueProfile::Quantiles {
                    cdf, statistics, ..
                }
                | PreparedValueProfile::Categorical { cdf, statistics },
            ) => {
                let crossings = match &point.profile {
                    Some(PreparedValueProfile::Quantiles {
                        source_crossings, ..
                    }) => Some(*source_crossings),
                    _ => None,
                };
                if !compact {
                    ui.add_space(12.);
                }
                ui.horizontal(|ui| {
                    let title = if self.focus_distribution || compact { "Conditional return distribution" } else { "Return profile" };
                    let response = if self.focus_distribution || compact { ui.heading(title) } else { ui.strong(title) };
                    if let Some(source_crossings) = crossings {
                        response.on_hover_text(format!("Full weighted atomic ordered CDF; {} raw adjacent head crossings. Raw heads remain unchanged; no density is inferred.",source_crossings));
                    } else {
                        response.on_hover_text("Return distribution over the recorded fixed atoms and probabilities.");
                    }
                    if !self.focus_distribution
                        && (compact || ui.ctx().content_rect().width() >= 850.)
                        && ui.small_button(if compact { "Expand" } else { "Expand distribution" }).clicked()
                    {
                        self.focus_distribution = true;
                    }
                    if compact && let Some(source_crossings) = crossings.filter(|count| *count > 0) {
                        ui.colored_label(crate::theme::WARNING, "Crossing")
                            .on_hover_text(format!("{source_crossings} adjacent crossings in raw probability order. Derived CDF and statistics use QuantileForecast::ordered."));
                    }
                });
                if !compact && let Some(source_crossings) = crossings.filter(|count| *count > 0) {
                    ui.colored_label(crate::theme::WARNING,"Crossing raw quantiles").on_hover_text(format!("{source_crossings} adjacent crossings in raw probability order. Derived CDF and statistics use QuantileForecast::ordered."));
                }
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.return_profile, ReturnProfileMode::Cdf, "CDF");
                    ui.selectable_value(
                        &mut self.return_profile,
                        ReturnProfileMode::Quantile,
                        "Quantiles",
                    );
                    ui.add_enabled_ui(
                        point
                            .empirical
                            .as_ref()
                            .is_some_and(|e| e.histogram.is_some()),
                        |ui| {
                            ui.selectable_value(
                                &mut self.return_profile,
                                ReturnProfileMode::Histogram,
                                "Histogram",
                            );
                        },
                    );
                });
                if self.return_profile == ReturnProfileMode::Histogram
                    && point
                        .empirical
                        .as_ref()
                        .is_none_or(|e| e.histogram.is_none())
                {
                    self.return_profile = ReturnProfileMode::Cdf;
                }
                let height = if self.focus_distribution {
                    (ui.available_height() * 0.68).clamp(300., 700.)
                } else if compact {
                    let minimum = if ui.ctx().content_rect().height() < 600. {
                        60.
                    } else if ui.ctx().content_rect().height() < 700. {
                        80.
                    } else {
                        200.
                    };
                    // A scroll area's available height is not its visible
                    // height. Reserve room for the profile key inside its
                    // actual clip rectangle at compact sizes.
                    let visible = (ui.clip_rect().bottom() - ui.cursor().top() - 55.).max(minimum);
                    (ui.available_height() - 150.)
                        .min(visible)
                        .clamp(minimum, 480.)
                } else if frame.is_some_and(|f| f.grid.promote_selected()) {
                    360.
                } else {
                    220.
                };
                if let Some(histogram) = point.empirical.as_ref().and_then(|e| e.histogram.as_ref())
                    && self.return_profile == ReturnProfileMode::Histogram
                {
                    let bars = histogram
                        .edges
                        .windows(2)
                        .zip(&histogram.counts)
                        .map(|(edge, &count)| {
                            Bar::new(edge[0] + (edge[1] - edge[0]) / 2., count as f64)
                                .width(edge[1] - edge[0])
                        })
                        .collect::<Vec<_>>();
                    Plot::new("value-empirical-histogram")
                        .height(height)
                        .x_axis_label("Return")
                        .y_axis_label("Episodes")
                        .include_y(0.)
                        .show(ui, |p| {
                            p.bar_chart(
                                BarChart::new("Empirical observations", bars)
                                    .color(crate::theme::EMPIRICAL),
                            );
                        });
                } else {
                    self.return_profile_plot(ui, point, cdf, height);
                }
                ui.horizontal_wrapped(|ui| {
                    ui.small(format!("Mean {:.3}", statistics.mean));
                    if let Some(sd) = statistics.standard_deviation {
                        ui.small(format!("SD {sd:.3}"));
                    }
                    if !compact {
                        ui.small(format!("q₁₀ {:.4}", statistics.lower_quantile));
                    }
                    ui.small(format!("CVaR₁₀ {:.3}", statistics.lower_cvar));
                });
                if let Some(evidence) = &point.empirical {
                    ui.horizontal_wrapped(|ui| {
                        if let Some(w) = evidence.wasserstein {
                            ui.small(format!("Wasserstein-1 {w:.4}"));
                        }
                        if let Some(crps) = evidence.mean_crps {
                            ui.small(format!("Mean CRPS {crps:.4}"));
                        }
                    });
                }
                if point.empirical.is_some() && !self.empirical_cdf_admitted {
                    ui.small("Empirical CDF overlay exceeds view admission; the predicted CDF and empirical mean are retained.");
                }
                if compact {
                    return;
                }
                match &point.prediction {
                    ValuePrediction::Quantiles {
                        raw,
                        locations,
                        masses,
                    } => {
                        egui::CollapsingHeader::new("Raw quantile details")
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical().max_height(150.).show_rows(
                                    ui,
                                    20.,
                                    raw.len(),
                                    |ui, rows| {
                                        for i in rows {
                                            ui.monospace(format!(
                                                "head {i} · τ={:.5} · mass={:.5} · raw={:.6}",
                                                locations[i], masses[i], raw[i]
                                            ));
                                        }
                                    },
                                );
                            })
                            .header_response
                    }
                    ValuePrediction::CategoricalReturn {
                        atoms,
                        probabilities,
                    } => {
                        egui::CollapsingHeader::new("Return atom details")
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical().max_height(150.).show_rows(
                                    ui,
                                    20.,
                                    atoms.len(),
                                    |ui, rows| {
                                        for i in rows {
                                            ui.monospace(format!(
                                                "Return {:.6} · probability {:.6}",
                                                atoms[i], probabilities[i]
                                            ));
                                        }
                                    },
                                );
                            })
                            .header_response
                    }
                    _ => unreachable!("distribution profile has a distribution prediction"),
                };
            }
            Some(PreparedValueProfile::Moments { summary, utility }) => {
                card(ui, "Mean", summary.mean);
                card(ui, "Signed variance", summary.variance).on_hover_text(format!("Raw second moment {:.9}; variance = raw second moment − mean². No return distribution is inferred.", summary.raw_second));
                if let Some(sd) = summary.standard_deviation {
                    card(ui, "Standard deviation", sd);
                }
                if let Some(value) = utility {
                    card(ui, "Declared utility", *value).on_hover_text(format!("{:?}. Actual declared learner Risk operation, including its variance clamp and f32 precision; separate from the selected plot statistic.", point.current().utility()));
                }
                if summary.standard_deviation.is_none() {
                    ui.colored_label(crate::theme::WARNING,"Negative variance").on_hover_text("The raw second moment is below the squared mean. Standard deviation is unavailable; the signed variance is preserved.");
                }
            }
            None => {}
        }
        if let ValuePrediction::CategoricalRegression {
            logits,
            weights,
            centers,
            ..
        } = &point.prediction
        {
            egui::CollapsingHeader::new("HL-Gauss regression bins").show(ui, |ui| {
                ui.small("Softmax weights over regression bins. Soft Q is their weighted center.");
                egui::ScrollArea::vertical().max_height(150.).show_rows(
                    ui,
                    20.,
                    centers.len(),
                    |ui, rows| {
                        for i in rows {
                            ui.monospace(format!(
                                "center {:.5} · weight {:.5} · logit {:.5}",
                                centers[i], weights[i], logits[i]
                            ));
                        }
                    },
                );
            });
        }
        if let ValuePrediction::Discrete { values, legal } = &point.prediction {
            let greedy = values
                .iter()
                .enumerate()
                .filter(|(i, _)| legal[*i])
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i);
            if let Some(greedy) = greedy {
                let e = point.current().semantics();
                let label = e
                    .action(greedy)
                    .map_or_else(|| format!("action[{greedy}]"), |a| a.label.into_owned());
                ui.small(format!("Legal greedy action {label}"));
            }
            let duplicate = frame.is_some_and(|f| {
                self.main_rect.is_some()
                    && !f.grid.promote_selected()
                    && (f.grid.slice.x == ValueCoordinate::DiscreteAction
                        || f.grid.slice.y == Some(ValueCoordinate::DiscreteAction))
            });
            if !duplicate {
                ui.strong("Action values");
                let id = egui::Id::unique("value-selected-actions");
                let initialize = egui_plot::PlotMemory::load(ui.ctx(), id).is_none();
                let response = Plot::new("value-selected-actions")
                    .id(id)
                    .height(300.)
                    .allow_scroll([true, false])
                    .allow_zoom([true, false])
                    .x_axis_label("Action")
                    .x_axis_formatter(|mark, _| {
                        if mark.value >= 0.
                            && mark.value < values.len() as f64
                            && mark.value.fract() == 0.
                        {
                            point
                                .current()
                                .semantics()
                                .action(mark.value as usize)
                                .map_or_else(
                                    || format!("{:.0}", mark.value),
                                    |a| a.label.into_owned(),
                                )
                        } else {
                            String::new()
                        }
                    })
                    .y_axis_label("Q")
                    .show(ui, |p| {
                        let (reference_values, reference_legal) = match &point.reference_prediction
                        {
                            Some(ValuePrediction::Discrete { values, legal }) => {
                                (values.as_slice(), legal.as_slice())
                            }
                            _ => (&[][..], &[][..]),
                        };
                        let (lo, hi) = values
                            .iter()
                            .chain(
                                reference_values
                                    .iter()
                                    .zip(reference_legal)
                                    .filter_map(|(value, legal)| legal.then_some(value)),
                            )
                            .copied()
                            .fold((0_f64, 0_f64), |(lo, hi), value| {
                                (lo.min(value), hi.max(value))
                            });
                        let pad = ((hi - lo) * 0.08).max(1e-6);
                        // Only horizontal navigation is interactive. The vertical
                        // range follows current gray/valid bars and only legal
                        // reference bars; hidden masked outliers do not count.
                        // Refresh on publication of a row, action or revision.
                        p.set_plot_bounds_y((lo - pad)..=(hi + pad));
                        if initialize {
                            p.set_plot_bounds_x(-0.7..=(values.len().min(16) as f64 - 0.3));
                        }
                        let bars = values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| {
                                *i as f64 >= p.plot_bounds().min()[0] - 1.
                                    && *i as f64 <= p.plot_bounds().max()[0] + 1.
                            })
                            .map(|(i, &v)| {
                                let label = point.current().semantics().action(i).map_or_else(
                                    || format!("action[{i}]"),
                                    |a| a.label.into_owned(),
                                );
                                Bar::new(i as f64, v)
                                    .width(0.75)
                                    .name(format!(
                                        "{label} · Q={v:.8}{}",
                                        if legal[i] { "" } else { " · masked" }
                                    ))
                                    .fill(if !legal[i] {
                                        Color32::from_gray(70)
                                    } else {
                                        crate::theme::CURRENT
                                    })
                                    .stroke(egui::Stroke::new(
                                        if point.action == Some(Action::Discrete(i)) {
                                            2.5
                                        } else {
                                            0.5
                                        },
                                        Color32::WHITE,
                                    ))
                            })
                            .collect();
                        p.bar_chart(BarChart::new("Action values", bars));
                        if let Some(ValuePrediction::Discrete { values, legal }) =
                            &point.reference_prediction
                        {
                            let bars = values
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| {
                                    legal[*i]
                                        && *i as f64 >= p.plot_bounds().min()[0] - 1.
                                        && *i as f64 <= p.plot_bounds().max()[0] + 1.
                                })
                                .map(|(i, &v)| {
                                    Bar::new(i as f64, v)
                                        .width(0.32)
                                        .fill(Color32::TRANSPARENT)
                                        .stroke(egui::Stroke::new(1.5, crate::theme::REFERENCE))
                                })
                                .collect();
                            p.bar_chart(BarChart::new("Reference Q", bars));
                        }
                        p.pointer_coordinate()
                    });
                if response.response.clicked()
                    && let Some(cursor) = response.inner
                {
                    let index = cursor.x.round();
                    if index >= 0. && index < values.len() as f64 && legal[index as usize] {
                        let action = Some(Action::Discrete(index as usize));
                        self.pin = Some(match point.pin() {
                            ValuePin::Probe { source, row, .. } => ValuePin::Probe {
                                source: source.clone(),
                                row: *row,
                                action,
                            },
                            _ => ValuePin::Input {
                                observation: point.input.clone(),
                                action,
                            },
                        });
                        self.dirty = true;
                        self.ready = false;
                    }
                }
            }
        }
        if let Some(error) = &point.reference_error {
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("Reference unavailable: {error}"))
                        .small()
                        .color(crate::theme::MUTED),
                )
                .truncate(),
            )
            .on_hover_text(error);
        }
        if let Some(g) = frame.map(|f| &f.grid) {
            if let Some(error) = &g.reference_policy_error {
                ui.weak("Controller overlay unavailable")
                    .on_hover_text(error);
            }
            if let Some(visits) = &g.visits {
                egui::CollapsingHeader::new("Genuine visits").show(ui, |ui| {
                    for (i, v) in visits.markers.iter().enumerate() {
                        let label = match &v.source.chunk.record {
                            rl::diagnostics::schema::Record::Trajectory(t) => format!(
                                "{} · step {} · k={}",
                                t.id,
                                v.row,
                                v.source.chunk.context.identity.learner_update_revision
                            ),
                            _ => format!(
                                "Probe row {} · k={}",
                                v.row, v.source.chunk.context.identity.learner_update_revision
                            ),
                        };
                        if ui.selectable_label(false, label).clicked() {
                            self.pending_visit = Some(i);
                        }
                    }
                });
            }
        }
        egui::CollapsingHeader::new("Details").default_open(crate::theme::inspection_default(ui)).show(ui, |ui| {
            ui.label(format!(
                "Complete observation {:?}\nMask {:?}",
                point.input.dense_values(), point.input.dense_mask()
            ));
            ui.label(presentation::value_source(
                "Current",
                point.current(),
                frame.map(|f| f.requested_revision),
                point.action.as_ref(),
            ));
            ui.label(format!("Complete current prediction {:?}", point.prediction));
            if let Some(prediction) = &point.reference_prediction {
                ui.label(format!("Complete reference prediction {prediction:?}"));
            }
            if let Some(f) = frame {
                ui.label(presentation::comparison(f.grid.comparison_condition()));
                if point.reference_value.is_some()
                    && let Some(source) = f.grid.reference()
                {
                    ui.label(presentation::value_source(
                        "Reference",
                        source,
                        f.reference_requested_revision,
                        point.reference_action.as_ref(),
                    ));
                }
            }
            if let Some(evidence) = &point.empirical {
                let c = &evidence.condition;
                ui.label(format!("Empirical run {} · record #{} · probe #{} · row {}\n{}\nContinuation {} · {}\nComplete observation {:?}\nMask {:?}\nFirst action {:?}", c.source_key.root.display(), c.source_key.sequence, c.probe_key.sequence, c.row_id, c.methodology, c.continuation, if c.stochastic { "stochastic" } else { "deterministic" }, c.input.dense_values(), c.input.dense_mask(), c.action));
                ui.small("Mean SE = sample SD / sqrt(n), assuming independent futures. It describes uncertainty of the conditional mean, not return dispersion or critic uncertainty.");
            }
            if let Some(reason) = &point.empirical_error { ui.label(format!("Conditional evidence unavailable: {reason}")); }
            if let Some(reason) = &point.reference_error { ui.label(format!("Reference unavailable: {reason}")); }
            if let Some(e) = point.recorded_evaluation() { ui.label(format!("Recorded continuation {} · γ={}", e.continuation_policy, e.discount)); }
        });
    }
}
