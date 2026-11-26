//! Model-aware selected policy presentation; reads worker-prepared statistics.
use super::*;
impl PolicyPane {
    fn selected_action_value(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: &PreparedPolicyFrame,
        point: &PreparedPoint,
        curve: &PreparedPolicyActionValue,
    ) {
        let source = &frame.grid.current;
        let current = crate::landscape::numeric_bounds(
            curve
                .points
                .iter()
                .map(|p| p[1])
                .chain(std::iter::once(curve.actor_value)),
            false,
        );
        let same = self.action_value_range.as_ref().is_some_and(|r| {
            r.root == source.key.root
                && r.actor == source.name
                && r.critic == curve.critic
                && r.role == curve.role
                && r.units == curve.reward_units
                && r.component == curve.component
        });
        if !same {
            self.action_value_range = Some(ActionValueRange {
                root: source.key.root.clone(),
                actor: source.name.clone(),
                critic: curve.critic.clone(),
                role: curve.role.clone(),
                units: curve.reward_units.clone(),
                component: curve.component,
                bounds: current,
            });
        } else if let Some(range) = &mut self.action_value_range {
            range.bounds[0] = range.bounds[0].min(current[0]);
            range.bounds[1] = range.bounds[1].max(current[1]);
        }
        let bounds = self
            .action_value_range
            .as_ref()
            .expect("range initialized")
            .bounds;
        let compact = ui.ctx().content_rect().height() < 720.;
        ui.horizontal(|ui| {
            ui.heading("Action-value · selected input").on_hover_text(format!(
                "Same-checkpoint critic: {}. Expected return in {} follows the critic's declared continuation. Other action components remain at the actor's selected action; the reference marker is an action, not a critic value.",
                curve.critic, curve.reward_units
            ));
            if compact {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(&curve.reward_units).small().color(crate::theme::MUTED)).truncate())
                        .on_hover_text(&curve.reward_units);
                });
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.;
            let (line, _) = ui.allocate_exact_size(egui::vec2(14., 14.), egui::Sense::hover());
            ui.painter().line_segment(
                [line.left_center(), line.right_center()],
                egui::Stroke::new(2., CURRENT),
            );
            ui.colored_label(CURRENT, "Critic");
            ui.add_space(4.);
            let (marker, _) = ui.allocate_exact_size(egui::vec2(12., 14.), egui::Sense::hover());
            ui.painter()
                .circle_filled(marker.center(), 3.5, crate::theme::SELECTED);
            ui.colored_label(
                crate::theme::SELECTED,
                format!(
                    "Actor {}",
                    crate::landscape::statistic_number(curve.actor_action)
                ),
            );
            if let Some(reference) = point.reference_value.filter(|v| v.is_finite()) {
                ui.add_space(4.);
                let (line, _) = ui.allocate_exact_size(egui::vec2(14., 14.), egui::Sense::hover());
                let center = line.center();
                ui.painter().line_segment(
                    [line.left_center(), center - egui::vec2(2., 0.)],
                    egui::Stroke::new(2., REFERENCE),
                );
                ui.painter().line_segment(
                    [center + egui::vec2(2., 0.), line.right_center()],
                    egui::Stroke::new(2., REFERENCE),
                );
                ui.colored_label(
                    REFERENCE,
                    format!(
                        "Reference {}",
                        crate::landscape::statistic_number(reference)
                    ),
                );
            }
        });
        let height = if compact {
            (ui.available_height() * 0.22).clamp(95., 120.)
        } else {
            (ui.available_height() * 0.28).clamp(145., 185.)
        };
        Plot::new("policy-selected-action-value")
            .height(height)
            .show_background(false)
            .grid_color(crate::theme::GRID)
            .grid_spacing(if compact { 12.0..=18.0 } else { 8.0..=300.0 })
            .y_grid_spacer(egui_plot::log_grid_spacer(if compact { 2 } else { 10 }))
            .custom_y_axes(vec![
                egui_plot::AxisHints::new_y().label_spacing(if compact {
                    12.0..=16.0
                } else {
                    20.0..=30.0
                }),
            ])
            .x_axis_formatter(axis_tick)
            .y_axis_formatter(axis_tick)
            .x_axis_label(Self::action(e, curve.component))
            .include_y(bounds[0])
            .include_y(bounds[1])
            .y_axis_label(if compact {
                "Q".into()
            } else if curve.reward_units.chars().count() <= 24 {
                format!("Q [{}]", curve.reward_units)
            } else {
                "Expected return".into()
            })
            .show(ui, |plot| {
                plot.line(
                    Line::new("Critic", curve.points.clone())
                        .color(CURRENT)
                        .width(2.),
                );
                plot.points(
                    Points::new(
                        "Actor action",
                        vec![[curve.actor_action, curve.actor_value]],
                    )
                    .color(crate::theme::SELECTED)
                    .radius(4.5),
                );
                if let Some(reference) = point.reference_value.filter(|v| v.is_finite()) {
                    plot.vline(
                        egui_plot::VLine::new("Reference action", reference)
                            .color(REFERENCE)
                            .style(LineStyle::dashed_dense()),
                    );
                }
            });
    }
    fn selected_action_comparison(
        &self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: &PreparedPolicyFrame,
        point: &PreparedPoint,
        reference: f64,
    ) {
        ui.strong("Action comparison");
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(CURRENT, format!("Learner {:.3}", point.value));
            ui.colored_label(REFERENCE, format!("Reference {reference:.3}"));
        });
        let action = frame.grid.config.action;
        let height = (ui.available_height() * 0.18).clamp(68., 96.);
        let mut plot = Plot::new("policy-selected-action-comparison")
            .height(height)
            .show_background(false)
            .show_axes([true, false])
            .show_grid([true, false])
            .allow_drag(false)
            .allow_zoom(false)
            .include_y(-0.8)
            .include_y(0.8)
            .grid_color(crate::theme::GRID)
            .x_axis_formatter(axis_tick)
            .x_axis_label(Self::action(e, action));
        if let ActionSpace::Box { low, high } = e.action_space()
            && let (Some(low), Some(high)) = (low.get(action), high.get(action))
        {
            plot = plot.include_x(f64::from(*low)).include_x(f64::from(*high));
        } else if let Some([low, high]) = self.colour_range {
            plot = plot.include_x(low).include_x(high);
        }
        plot.show(ui, |plot| {
            plot.points(
                Points::new("Learner", vec![[point.value, 0.3]])
                    .color(CURRENT)
                    .radius(5.),
            );
            plot.points(
                Points::new("Reference", vec![[reference, -0.3]])
                    .shape(egui_plot::MarkerShape::Diamond)
                    .color(REFERENCE)
                    .radius(5.),
            );
        });
        if let Some(difference) = point.difference {
            ui.weak(format!("Difference {difference:+.3}"));
        }
    }

    pub(super) fn selected(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedPolicyFrame>,
    ) {
        egui::ScrollArea::vertical()
            .id_salt("policy-selected")
            .auto_shrink([false, false])
            .max_height(ui.available_height())
            .show(ui, |ui| {
                crate::theme::panel_frame().show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    if frame
                        .and_then(|f| f.selected.as_ref())
                        .and_then(|p| p.action_value_curve())
                        .is_none()
                    {
                        ui.heading("Selected");
                    }
                    self.selected_contents(ui, e, frame);
                });
                self.policy_cross_section(ui, e, frame);
            });
    }

    /// Prefer the worker-prepared exact selected-input slice. Prepared-cache
    /// views without one retain the explicitly identified evaluated column.
    fn policy_cross_section(
        &self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedPolicyFrame>,
    ) {
        let Some(frame) = frame else { return };
        let Some(point) = frame.selected.as_ref() else {
            return;
        };
        let grid = &frame.grid;
        let c = &grid.config;
        if !grid.regular()
            || c.mode != SliceMode::TwoDimensional
            || !matches!(point.distribution, SelectedDistribution::Deterministic)
            || point.chosen_action.is_some()
        {
            return;
        }
        let Some(y_axis) = c.y else { return };
        let [nx, ny] = c.resolution;
        if !(2..=256).contains(&nx)
            || !(2..=256).contains(&ny)
            || nx.checked_mul(ny) != Some(grid.values.len())
        {
            return;
        }
        let [xlo, xhi] = c.ranges[c.x];
        let selected_x = point.input.dense_values()[c.x];
        if !selected_x.is_finite() || selected_x < xlo || selected_x > xhi {
            return;
        }
        let [ylo, yhi] = c.ranges[y_axis];
        let (fixed_x, current, reference, exact, omitted, scan_limited) = if let Some(cross) =
            point.cross_section()
        {
            (
                cross.fixed_value,
                cross.points.clone(),
                cross.reference_samples.clone(),
                true,
                cross.omitted_reference_samples,
                cross.reference_scan_limited,
            )
        } else {
            let x = (((f64::from(selected_x) - f64::from(xlo)) / (f64::from(xhi) - f64::from(xlo)))
                * (nx - 1) as f64)
                .round() as usize;
            let fixed_x = (f64::from(xlo)
                + (f64::from(xhi) - f64::from(xlo)) * x as f64 / (nx - 1) as f64)
                as f32;
            let mut current = Vec::with_capacity(ny);
            let mut reference = Vec::with_capacity(ny);
            for y in 0..ny {
                let coordinate = (f64::from(ylo)
                    + (f64::from(yhi) - f64::from(ylo)) * y as f64 / (ny - 1) as f64)
                    as f32;
                let index = y * nx + x;
                let value = grid.values[index];
                if !value.is_finite() {
                    return;
                }
                current.push([f64::from(coordinate), value]);
                if let Some(value) = grid.reference_at(index) {
                    reference.push([f64::from(coordinate), value]);
                }
            }
            (fixed_x, current, reference, false, 0, false)
        };
        ui.add_space(8.);
        let height = (ui.available_height() - 72.).clamp(100., 600.);
        crate::theme::panel_frame().show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.heading("Policy cross-section");
            let mut context = format!(
                "{} = {:.3}{}",
                Self::coordinate(e, c.x),
                fixed_x,
                if exact { "" } else { " · grid column" }
            );
            let differences: Vec<_> = if exact {
                point
                    .input
                    .dense_values()
                    .iter()
                    .zip(&c.fixed)
                    .enumerate()
                    .filter(|(index, (selected, fixed))| {
                        *index != c.x && *index != y_axis && selected.to_bits() != fixed.to_bits()
                    })
                    .map(|(index, (selected, _))| {
                        format!(
                            "{} = {}",
                            Self::coordinate(e, index),
                            crate::landscape::statistic_number(f64::from(*selected))
                        )
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for difference in differences.iter().take(2) {
                context.push_str(" · ");
                context.push_str(difference);
            }
            if differences.len() > 2 {
                context.push_str(&format!(" · +{}", differences.len() - 2));
            }
            let short = height < 150.;
            ui.weak(if short {
                format!("{context} · {}", Self::action(e, c.action))
            } else {
                context
            })
            .on_hover_text(format!(
                "{}{}{}{}",
                "Reference markers are recorded samples at the same complete fixed input.",
                if differences.is_empty() {
                    String::new()
                } else {
                    format!(
                        "\nSelected input differs from the map slice: {}",
                        differences.join(", ")
                    )
                },
                if omitted == 0 {
                    String::new()
                } else {
                    format!(
                        "\n{omitted} matching reference rows omitted by the bounded display limit"
                    )
                },
                if scan_limited {
                    "\nReference table exceeds the cross-section scan limit."
                } else {
                    ""
                },
            ));
            let mut plot = Plot::new("policy-cross-section")
                .height(height)
                .show_background(false)
                .grid_color(crate::theme::GRID)
                .grid_spacing(if short { 30.0..=60.0 } else { 65.0..=120.0 })
                .legend(egui_plot::Legend::default())
                .x_axis_formatter(axis_tick)
                .y_axis_formatter(axis_tick)
                .x_axis_label(crate::landscape::observation_axis_label(e, y_axis));
            if let ActionSpace::Box { low, high } = e.action_space()
                && let (Some(low), Some(high)) = (low.get(c.action), high.get(c.action))
            {
                plot = plot.include_y(f64::from(*low)).include_y(f64::from(*high));
            }
            let plot = if short {
                plot
            } else {
                plot.y_axis_label(Self::action(e, c.action))
            };
            let has_reference_samples = !reference.is_empty();
            plot.show(ui, |plot| {
                plot.line(Line::new("Learner", current).color(CURRENT).width(2.));
                if !exact && reference.len() == ny {
                    plot.line(
                        Line::new("Reference", reference)
                            .color(REFERENCE)
                            .style(LineStyle::dashed_dense()),
                    );
                } else if !reference.is_empty() {
                    plot.points(
                        Points::new("Reference samples", reference)
                            .shape(egui_plot::MarkerShape::Diamond)
                            .color(REFERENCE)
                            .radius(4.),
                    );
                }
                if exact {
                    let selected_y = point.input.dense_values()[y_axis];
                    if selected_y.is_finite() && (ylo..=yhi).contains(&selected_y) {
                        plot.points(
                            Points::new("", vec![[f64::from(selected_y), point.value]])
                                .color(Color32::WHITE)
                                .radius(4.5),
                        );
                        if let Some(value) = point.reference_value {
                            plot.points(
                                Points::new(
                                    if has_reference_samples {
                                        ""
                                    } else {
                                        "Reference"
                                    },
                                    vec![[f64::from(selected_y), value]],
                                )
                                .shape(egui_plot::MarkerShape::Diamond)
                                .color(REFERENCE)
                                .radius(5.),
                            );
                        }
                    }
                }
            });
        });
    }
    pub(super) fn selected_contents(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedPolicyFrame>,
    ) {
        let point = frame.and_then(|f| f.selected.as_ref());
        let Some(point) = point else {
            ui.weak(if self.terminal {
                "Unavailable"
            } else {
                "Loading…"
            });
            return;
        };
        if !self.ready
            && self
                .pin_input
                .as_ref()
                .is_some_and(|input| input != &point.input)
        {
            ui.colored_label(crate::theme::WARNING, "Previous input")
                .on_hover_text(format!(
                    "Displayed complete observation {:?}\nLegal mask {:?}",
                    point.input.dense_values(),
                    point.input.dense_mask()
                ));
        }
        if let Some(row) = point.cached_row {
            ui.weak(format!("Cached grid point {}", row + 1));
        }
        let card = |ui: &mut egui::Ui, label: &str, value: f64, color: Color32| {
            ui.horizontal(|ui| {
                ui.weak(label);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(if value.is_infinite() {
                            "∞".into()
                        } else {
                            crate::landscape::statistic_number(value)
                        })
                        .size(17.)
                        .color(color),
                    );
                });
            });
        };
        let deterministic_discrete =
            matches!(point.distribution, SelectedDistribution::Deterministic)
                && point.chosen_action.is_some();
        if let Some(curve) = point.action_value_curve() {
            self.selected_action_value(ui, e, frame.unwrap(), point, curve);
        } else if deterministic_discrete {
            ui.strong(format!(
                "Chosen action · {}",
                Self::action(e, point.chosen_action.unwrap())
            ));
            if let Some(reference) = point.reference_chosen_action {
                ui.label(format!("Reference · {}", Self::action(e, reference)));
                ui.label(if Some(reference) == point.chosen_action {
                    "Actions agree"
                } else {
                    "Actions differ"
                });
            }
        } else if let Some(reference) = point.reference_value {
            if matches!(point.distribution, SelectedDistribution::Deterministic)
                && point.chosen_action.is_none()
                && point.value.is_finite()
                && reference.is_finite()
            {
                self.selected_action_comparison(ui, e, frame.unwrap(), point, reference);
            } else {
                let label = Self::action(e, frame.unwrap().grid.config.action);
                card(ui, &label, point.value, CURRENT);
                card(ui, "Reference", reference, REFERENCE);
                if let Some(v) = point.difference {
                    card(ui, "Difference", v, crate::theme::SELECTED);
                }
            }
        } else {
            let label = Self::action(e, frame.unwrap().grid.config.action);
            card(ui, &label, point.value, CURRENT);
        }
        if let Some(v) = point.visits.filter(|_| {
            frame
                .and_then(|f| f.grid.visits.as_ref())
                .is_some_and(|v| v.filtered_count > 0)
        }) {
            ui.label(format!("Visits  {v}")).on_hover_text(frame.and_then(|f|f.grid.visits.as_ref()).map_or(String::new(),|v|format!("{:?}, k={}; local uniform plot bin; fixed tolerance {}; {} compatible observations; bounded deterministic markers",v.source,v.revision,v.tolerance,v.filtered_count)));
        }
        if !matches!(point.distribution, SelectedDistribution::Deterministic) {
            ui.add_space(8.);
            ui.strong("Action distribution").on_hover_text("Actual action marginal. Initial unbounded display extent: union of current/reference mean ±5 standard deviations; retained through playback. The distribution is never truncated or renormalized. Density uses the actual action transform and Jacobian; CDF replaces numerically unresolved density.");
            let categories = match &point.distribution {
                SelectedDistribution::Categorical { probabilities, .. } => {
                    Some(probabilities.len())
                }
                _ => None,
            };
            let current_cdf = point.plot.as_ref().is_some_and(|p| p.is_cdf);
            let reference_cdf = point.reference_plot.as_ref().is_some_and(|p| p.is_cdf);
            ui.horizontal(|ui| {
                ui.colored_label(
                    CURRENT,
                    if categories.is_some() {
                        "Current"
                    } else if current_cdf {
                        "Current CDF"
                    } else {
                        "Current density"
                    },
                );
                if point.reference_distribution.is_some() {
                    ui.colored_label(
                        REFERENCE,
                        if categories.is_some() {
                            "Reference"
                        } else if reference_cdf {
                            "Reference CDF"
                        } else {
                            "Reference density"
                        },
                    );
                }
            });
            let plot = Plot::new("policy-distribution")
                .grid_spacing(30.0..=80.0)
                .x_grid_spacer(egui_plot::log_grid_spacer(2))
                .y_grid_spacer(egui_plot::log_grid_spacer(2))
                .custom_x_axes(vec![
                    egui_plot::AxisHints::new_x()
                        .label_spacing(if categories.is_some() {
                            20.0..=35.0
                        } else {
                            60.0..=80.0
                        })
                        .formatter(move |mark, range| match categories {
                            Some(count)
                                if mark.value >= 0.
                                    && mark.value < count as f64
                                    && mark.value.fract() == 0. =>
                            {
                                if e.descriptor().is_some() {
                                    Self::action(e, mark.value as usize)
                                } else {
                                    format!("{:.0}", mark.value)
                                }
                            }
                            Some(_) => String::new(),
                            None => axis_tick(mark, range),
                        }),
                ])
                .x_axis_label(if categories.is_some() {
                    "Action".into()
                } else {
                    Self::action(e, frame.unwrap().grid.config.action)
                })
                .y_axis_label(if categories.is_some() {
                    "Probability"
                } else if current_cdf && (point.reference_plot.is_none() || reference_cdf) {
                    "CDF"
                } else if current_cdf || reference_cdf {
                    "Density / CDF"
                } else {
                    "Density"
                })
                .y_axis_formatter(axis_tick)
                .allow_scroll([true, false])
                .allow_zoom([true, false])
                .id(Self::distribution_plot_id())
                .height(if frame.is_some_and(|f| f.grid.promote_selected()) {
                    (ui.available_height() - 110.).clamp(220., 480.)
                } else {
                    280.
                })
                .allow_double_click_reset(false)
                .show_background(false)
                .grid_color(crate::theme::GRID);
            let initialize = !self.distribution_initialized;
            let distribution_view = match &point.distribution {
                SelectedDistribution::Gaussian(_) => DistributionView::Gaussian {
                    cdf: point.plot.as_ref().is_some_and(|p| p.is_cdf),
                    reference: frame
                        .and_then(|f| f.grid.reference.as_ref())
                        .zip(point.reference_plot.as_ref())
                        .map(|(source, plot)| (source.key.clone(), plot.is_cdf)),
                },
                _ => DistributionView::Categorical,
            };
            if !initialize
                && let Some(automatic) = self.distribution_auto_y
                && let Some(memory) =
                    egui_plot::PlotMemory::load(ui.ctx(), Self::distribution_plot_id())
                && [memory.bounds().min()[1], memory.bounds().max()[1]] != automatic
            {
                self.distribution_auto_y = None;
            }
            let ymax = distribution_ymax(point);
            let fit_y = initialize
                || self.distribution_auto_y.is_some_and(|automatic| {
                    self.distribution_view.as_ref() != Some(&distribution_view)
                        || ymax > automatic[1]
                });
            if fit_y {
                self.distribution_auto_y = Some([0., ymax]);
            }
            let response = plot.show(ui, |p| {
                match &point.distribution {
                    SelectedDistribution::Categorical {
                        probabilities,
                        valid,
                    } => {
                        if initialize {
                            p.set_plot_bounds(PlotBounds::from_min_max(
                                [-0.7, 0.],
                                [(probabilities.len().min(16)) as f64 - 0.3, 1.],
                            ));
                        }
                        let bars = probabilities
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| {
                                *i as f64 >= p.plot_bounds().min()[0] - 1.
                                    && *i as f64 <= p.plot_bounds().max()[0] + 1.
                            })
                            .map(|(i, &v)| {
                                Bar::new(i as f64, v)
                                    .width(0.75)
                                    .name(format!(
                                        "{} · p={v:.8}{}",
                                        Self::action(e, i),
                                        if valid[i] { "" } else { " · invalid" }
                                    ))
                                    .stroke(egui::Stroke::new(
                                        if i == frame.unwrap().grid.config.action {
                                            2.5
                                        } else {
                                            0.5
                                        },
                                        if i == frame.unwrap().grid.config.action {
                                            Color32::WHITE
                                        } else {
                                            CURRENT
                                        },
                                    ))
                                    .fill(if valid[i] {
                                        CURRENT
                                    } else {
                                        Color32::from_gray(80)
                                    })
                            })
                            .collect();
                        p.bar_chart(BarChart::new("Current", bars));
                        let mut zero = Vec::new();
                        let mut invalid = Vec::new();
                        for (i, &is_valid) in valid.iter().enumerate().filter(|(i, _)| {
                            *i as f64 >= p.plot_bounds().min()[0] - 1.
                                && *i as f64 <= p.plot_bounds().max()[0] + 1.
                        }) {
                            if !is_valid || probabilities[i] == 0. {
                                if is_valid {
                                    zero.push([i as f64, 0.]);
                                } else {
                                    invalid.push([i as f64, 0.]);
                                }
                            }
                        }
                        p.points(Points::new("Valid zero", zero).radius(3.).color(CURRENT));
                        p.points(
                            Points::new("Invalid", invalid)
                                .radius(3.)
                                .shape(egui_plot::MarkerShape::Cross)
                                .color(REFERENCE),
                        );
                        if let Some((i, &probability)) = probabilities
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| valid[*i])
                            .max_by(|a, b| a.1.total_cmp(b.1))
                        {
                            p.points(
                                Points::new("Greedy action", vec![[i as f64, probability]])
                                    .radius(5.)
                                    .shape(egui_plot::MarkerShape::Diamond)
                                    .color(Color32::WHITE),
                            );
                        }
                        if let Some(SelectedDistribution::Categorical { probabilities, .. }) =
                            &point.reference_distribution
                        {
                            p.bar_chart(BarChart::new(
                                "Reference",
                                probabilities
                                    .iter()
                                    .enumerate()
                                    .filter(|(i, _)| {
                                        *i as f64 >= p.plot_bounds().min()[0] - 1.
                                            && *i as f64 <= p.plot_bounds().max()[0] + 1.
                                    })
                                    .map(|(i, &v)| {
                                        Bar::new(i as f64, v)
                                            .width(0.35)
                                            .fill(REFERENCE.linear_multiply(0.6))
                                    })
                                    .collect(),
                            ));
                        }
                    }
                    SelectedDistribution::Gaussian(_) => {
                        for (name, mean, color) in [
                            (
                                "Current mean",
                                Some(point.action_statistics.map_or(point.value, |s| s.mean)),
                                CURRENT,
                            ),
                            (
                                "Reference mean",
                                point
                                    .reference_statistics
                                    .map(|s| s.mean)
                                    .or(point.reference_value),
                                REFERENCE,
                            ),
                        ] {
                            if let Some(mean) = mean {
                                p.vline(
                                    egui_plot::VLine::new(name, mean)
                                        .color(color)
                                        .style(LineStyle::dashed_dense()),
                                );
                            }
                        }
                        if let Some(plot) = &point.plot {
                            if initialize {
                                p.set_plot_bounds_x(plot.domain[0]..=plot.domain[1]);
                            }
                            if fit_y {
                                p.set_plot_bounds_y(0. ..=ymax);
                            }
                            p.line(
                                Line::new(
                                    if plot.is_cdf {
                                        "Current CDF"
                                    } else {
                                        "Current density"
                                    },
                                    plot.samples.clone(),
                                )
                                .color(CURRENT),
                            );
                        }
                        if let Some(plot) = &point.reference_plot {
                            p.line(
                                Line::new(
                                    if plot.is_cdf {
                                        "Reference CDF"
                                    } else {
                                        "Reference density"
                                    },
                                    plot.samples.clone(),
                                )
                                .color(REFERENCE)
                                .style(LineStyle::dashed_dense()),
                            );
                        }
                    }
                    SelectedDistribution::Deterministic => {}
                }
                p.pointer_coordinate()
            });
            if response.response.contains_pointer()
                && let Some(cursor) = response.inner
                && let SelectedDistribution::Categorical {
                    probabilities,
                    valid,
                } = &point.distribution
            {
                let index = cursor.x.round();
                if index >= 0. && index < probabilities.len() as f64 {
                    let index = index as usize;
                    if response.response.clicked()
                        && let Some(c) = &mut self.config
                    {
                        c.action = index;
                        self.dirty = true;
                        self.ready = false;
                    }
                    egui::Tooltip::always_open(
                        ui.ctx().clone(),
                        response.response.layer_id,
                        response.response.id.with("policy-action-tooltip"),
                        egui::PopupAnchor::Pointer,
                    )
                    .show(|ui| {
                        ui.label(format!(
                            "{} · p={:.8} · {}",
                            Self::action(e, index),
                            probabilities[index],
                            if valid[index] { "valid" } else { "invalid" }
                        ));
                    });
                }
            }
            self.distribution_initialized = true;
            self.distribution_view = Some(distribution_view);
        }
        match &point.distribution {
            SelectedDistribution::Categorical {
                probabilities,
                valid,
            } => {
                ui.label(format!(
                    "Selected action · {}",
                    Self::action(e, frame.unwrap().grid.config.action)
                ));
                let greedy = probabilities
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| valid[*i])
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i);
                if let Some(i) = greedy {
                    ui.label(format!("Greedy action · {}", Self::action(e, i)));
                }
                let entropy = -probabilities
                    .iter()
                    .filter(|&&p| p > 0.)
                    .map(|p| p * p.ln())
                    .sum::<f64>();
                ui.label(format!("Entropy {entropy:.5} nats"));
                if let Some(SelectedDistribution::Categorical {
                    probabilities,
                    valid,
                }) = &point.reference_distribution
                    && let Some((i, _)) = probabilities
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| valid[*i])
                        .max_by(|a, b| a.1.total_cmp(b.1))
                {
                    ui.label(format!("Reference greedy · {}", Self::action(e, i)));
                }
            }
            SelectedDistribution::Gaussian(_) => {
                if let Some(s) = point.action_statistics {
                    ui.label(format!(
                        "Action mean {} · std {}",
                        crate::landscape::statistic_number(s.mean),
                        crate::landscape::statistic_number(s.standard_deviation)
                    ));
                } else {
                    ui.label(format!(
                        "Action mean {}",
                        crate::landscape::statistic_number(point.value)
                    ));
                    ui.weak("Action spread precision unavailable").on_hover_text("Selected central integration cannot resolve spread below one millionth of the action half-range, or exhausted its numerical budget. A continuous Gaussian is never reported as deterministic zero.");
                }
                if let Some(s) = point.reference_statistics {
                    ui.label(format!(
                        "Reference mean {} · std {}",
                        crate::landscape::statistic_number(s.mean),
                        crate::landscape::statistic_number(s.standard_deviation)
                    ));
                }
            }
            _ => {}
        }
        if frame.is_some_and(|f| f.grid.reference.is_some()) && point.reference_value.is_none() {
            ui.weak("Policy reference unavailable at this exact input");
        }
        if let Some(kl) = point.kl {
            ui.separator();
            ui.horizontal(|ui|{ui.label("KL divergence").on_hover_text("Full distribution D_KL(current || reference), at the same complete input. Categorical sum over all actions; full diagonal Gaussian KL under identical invertible action transforms.");ui.strong(if kl.is_infinite(){"∞".into()}else{format!("{kl:.6}")});});
        }
        if let Some(frame) = frame {
            if self.terminal || frame.grid.current.revision != frame.requested_revision {
                ui.colored_label(
                    crate::theme::WARNING,
                    format!("Showing checkpoint {}", frame.grid.current.revision),
                );
            }
            egui::CollapsingHeader::new("Details").default_open(crate::theme::inspection_default(ui)).show(ui, |ui| {
                ui.label(format!("Complete observation {:?}\nLegal mask {:?}", point.input.dense_values(), point.input.dense_mask()));
                ui.label(format!("Current source {:?}", frame.grid.current));
                if let Some(reference) = &frame.grid.reference { ui.label(format!("Reference source {reference:?}")); }
                ui.label(format!("Recorded input {:?}\nReference input {:?}", point.recorded_input, point.reference_recorded_input));
                ui.label(format!("Current distribution {:?}", point.distribution));
                if let Some(reference) = &point.reference_distribution { ui.label(format!("Reference distribution {reference:?}")); }
                ui.small("Action marginals use the prepared transform and Jacobian. Complete categorical probabilities, masks and Gaussian parameters are retained above; KL uses the full distribution at the same complete input.");
            });
        }
    }
}
