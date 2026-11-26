use super::*;
use eframe::egui::{Color32, ColorImage, TextureOptions};
use egui_plot::{
    Bar, BarChart, Line, Plot, PlotBounds, PlotImage, PlotPoint, Points, Polygon, VLine,
};
const CYAN: Color32 = crate::theme::CURRENT;
const REFERENCE: Color32 = crate::theme::REFERENCE;
const EMPIRICAL: Color32 = crate::theme::EMPIRICAL;

enum ActionLegendMark {
    Line,
    Selected,
    Maximum,
    Empirical,
    Reference,
}

fn action_legend_key(ui: &mut egui::Ui, label: &str, color: Color32, mark: ActionLegendMark) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(24., 16.), egui::Sense::hover());
        let center = rect.center();
        let stroke = egui::Stroke::new(1.7, color);
        match mark {
            ActionLegendMark::Line => ui
                .painter()
                .line_segment([rect.left_center(), rect.right_center()], stroke),
            ActionLegendMark::Selected => {
                ui.painter().line_segment(
                    [center - egui::vec2(0., 7.), center + egui::vec2(0., 7.)],
                    stroke,
                );
                ui.painter().circle_filled(center, 3., color)
            }
            ActionLegendMark::Maximum => ui.painter().add(egui::Shape::convex_polygon(
                vec![
                    center - egui::vec2(0., 5.),
                    center + egui::vec2(5., 0.),
                    center + egui::vec2(0., 5.),
                    center - egui::vec2(5., 0.),
                ],
                color,
                egui::Stroke::NONE,
            )),
            ActionLegendMark::Reference => ui.painter().rect_filled(
                egui::Rect::from_center_size(center, egui::vec2(8., 8.)),
                0.,
                color,
            ),
            ActionLegendMark::Empirical => {
                ui.painter().line_segment(
                    [center - egui::vec2(0., 7.), center + egui::vec2(0., 7.)],
                    stroke,
                );
                ui.painter().line_segment(
                    [center + egui::vec2(-4., -7.), center + egui::vec2(4., -7.)],
                    stroke,
                );
                ui.painter().line_segment(
                    [center + egui::vec2(-4., 7.), center + egui::vec2(4., 7.)],
                    stroke,
                );
                ui.painter().circle_filled(center, 3., color)
            }
        };
        ui.label(egui::RichText::new(label).size(12.));
    });
}

#[derive(Clone, Copy)]
struct SampledMaximum {
    point: [f64; 2],
    samples: usize,
    tied: bool,
}

fn sampled_maximum(
    positions: &[[f32; 2]],
    values: impl Iterator<Item = Option<f64>>,
) -> Option<SampledMaximum> {
    let mut maximum = None::<SampledMaximum>;
    let mut samples = 0;
    for (&[action, _], value) in positions.iter().zip(values) {
        let Some(value) = value.filter(|v| v.is_finite() && action.is_finite()) else {
            continue;
        };
        samples += 1;
        match &mut maximum {
            Some(best) if value == best.point[1] => best.tied = true,
            Some(best) if value < best.point[1] => {}
            _ => {
                maximum = Some(SampledMaximum {
                    point: [f64::from(action), value],
                    samples: 0,
                    tied: false,
                });
            }
        }
    }
    maximum.map(|mut maximum| {
        maximum.samples = samples;
        maximum
    })
}

fn maximum_displacement(
    current: Option<SampledMaximum>,
    reference: Option<SampledMaximum>,
    grid_points: usize,
) -> Option<f64> {
    let (current, reference) = (current?, reference?);
    (grid_points > 1
        && current.samples == grid_points
        && reference.samples == grid_points
        && !current.tied
        && !reference.tied)
        .then_some((current.point[0] - reference.point[0]).abs())
}

struct ActionSliceMaxima {
    current: Option<SampledMaximum>,
    reference: Option<SampledMaximum>,
    grid_points: usize,
    discrete: bool,
}

impl ActionSliceMaxima {
    fn from_grid(g: &PreparedValueGrid, reference_enabled: bool) -> Option<Self> {
        let slice = &g.slice;
        if slice.mode != SliceMode::OneDimensional
            || !matches!(
                slice.x,
                ValueCoordinate::Action(_) | ValueCoordinate::DiscreteAction
            )
            || !g.regular()
            || !matches!(
                g.output(),
                ValueOutput::ExpectedReturn
                    | ValueOutput::Q1
                    | ValueOutput::Q2
                    | ValueOutput::Minimum
                    | ValueOutput::Quantile { .. }
                    | ValueOutput::Cvar { .. }
                    | ValueOutput::Utility
            )
        {
            return None;
        }
        let current = sampled_maximum(&g.positions, g.values.iter().copied().map(Some));
        let reference =
            if reference_enabled && !g.empirical_reference() && g.reference_error.is_none() {
                // Only grid-aligned values have passed the exact state-action
                // match. The sparse series can include a tolerance neighborhood.
                sampled_maximum(&g.positions, g.reference_values.iter().copied())
            } else {
                None
            };
        (current.is_some() || reference.is_some()).then_some(Self {
            current,
            reference,
            grid_points: g.positions.len(),
            discrete: slice.x == ValueCoordinate::DiscreteAction,
        })
    }

    fn show(&self, ui: &mut egui::Ui) {
        let label = if self.discrete {
            "action maximum"
        } else {
            "grid maximum"
        };
        ui.horizontal_wrapped(|ui| {
            for (side, maximum, color) in [
                ("Current", self.current, CYAN),
                ("Reference", self.reference, REFERENCE),
            ] {
                let Some(maximum) = maximum else {
                    continue;
                };
                let [action, value] = maximum.point;
                let action = if self.discrete { format!("{action:.0}") } else { format!("{action:.5}") };
                let coverage = if maximum.samples < self.grid_points {
                    format!(" · {}/{} samples", maximum.samples, self.grid_points)
                } else {
                    String::new()
                };
                ui.colored_label(color, format!("{side} {label} {value:.5} at {action}{coverage}"))
                    .on_hover_text(if self.discrete {
                        "Highest available value among the legal actions shown at this state. A tie displays the first action."
                    } else {
                        "Highest sampled value along this action coordinate at the fixed state. The estimate is limited to the plotting grid. A tie displays the first sample."
                    });
            }
            if !self.discrete
                && let Some(displacement) = maximum_displacement(self.current, self.reference, self.grid_points)
            {
                ui.small(format!("Action displacement {displacement:.5}"))
                    .on_hover_text("Absolute coordinate difference between the unique maxima on the same sampled action grid.");
            }
        });
    }
}

impl ValueView {
    pub(super) fn action_curve(
        &self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: &PreparedValueFrame,
        height: f32,
    ) {
        let Some(point) = frame.selected.as_ref() else {
            return;
        };
        let Some(curve) = &point.action_curve else {
            return;
        };
        let plot_height = (height - 36.).max(120.);
        let short = plot_height < 150.;
        let quantity = source_output_label(Some(frame.grid.current()), frame.grid.output());
        let title = if short {
            format!("Action-value function · selected input · {quantity}")
        } else {
            "Action-value function · selected input".into()
        };
        ui.label(egui::RichText::new(title).size(17.).strong()).on_hover_text("The displayed action component varies over its configured bounds; all other action components remain at the selected action. The marked maximum is a sampled grid maximum.");
        let selected = match &point.action {
            Some(Action::Continuous(values)) => values.get(curve.component).copied().map(f64::from),
            _ => None,
        };
        let reference_point = selected.zip(point.reference_value).filter(|(_, value)| {
            value.is_finite()
                && frame.grid.reference().is_some()
                && frame.grid.comparison_condition() == ComparisonCondition::SameInput
        });
        let maximum = curve
            .points
            .iter()
            .copied()
            .filter(|p| p[1].is_finite())
            .max_by(|a, b| a[1].total_cmp(&b[1]));
        let empirical: Vec<_> = if matches!(
            frame.grid.output(),
            ValueOutput::ExpectedReturn | ValueOutput::Q1 | ValueOutput::Q2
        ) {
            frame
                .grid
                .empirical
                .conditions
                .iter()
                .filter_map(|condition| {
                    let (Some(Action::Continuous(actual)), Some(Action::Continuous(base))) =
                        (&condition.action, &point.action)
                    else {
                        return None;
                    };
                    if condition.input != point.input
                        || actual.len() != base.len()
                        || actual
                            .iter()
                            .zip(base)
                            .enumerate()
                            .any(|(i, (a, b))| i != curve.component && a != b)
                    {
                        return None;
                    }
                    Some((
                        [
                            f64::from(*actual.get(curve.component)?),
                            condition.summary.mean,
                        ],
                        condition.summary.mean_standard_error,
                    ))
                })
                .collect()
        } else {
            Vec::new()
        };
        let plot_width = (ui.available_width() - 236.).max(100.);
        ui.horizontal_top(|ui| {
            let plot = Plot::new("value-selected-action-curve")
                .id(egui::Id::unique("value-selected-action-curve"))
                .width(plot_width)
                .height(plot_height)
                .show_background(false)
                .grid_color(crate::theme::GRID)
                .grid_spacing(if short { 15.0..=35.0 } else { 25.0..=50.0 })
                .y_grid_spacer(egui_plot::log_grid_spacer(2))
                .x_axis_formatter(landscape::axis_tick)
                .y_axis_formatter(landscape::axis_tick)
                .x_axis_label(coordinate_axis_label(
                    e,
                    ValueCoordinate::Action(curve.component),
                ));
            let plot = if short {
                plot
            } else {
                plot.y_axis_label(quantity)
            };
            plot.show(ui, |plot| {
                plot.line(
                    Line::new("Learned critic", curve.points.clone())
                        .color(CYAN)
                        .width(2.),
                );
                if let Some(x) = selected {
                    plot.vline(VLine::new("Selected action", x).color(crate::theme::SELECTED));
                    if point.value.is_finite() {
                        plot.points(
                            Points::new("", vec![[x, point.value]])
                                .color(crate::theme::SELECTED)
                                .radius(4.),
                        );
                    }
                }
                if let Some(maximum) = maximum {
                    plot.points(
                        Points::new("Grid maximum (sampled)", vec![maximum])
                            .color(REFERENCE)
                            .shape(egui_plot::MarkerShape::Diamond)
                            .radius(4.),
                    );
                }
                if let Some((x, value)) = reference_point {
                    plot.points(
                        Points::new("Reference value", vec![[x, value]])
                            .color(REFERENCE)
                            .shape(egui_plot::MarkerShape::Square)
                            .radius(4.5),
                    );
                }
                if !empirical.is_empty() {
                    plot.points(
                        Points::new(
                            "Independent futures · ±1.96 SE",
                            empirical
                                .iter()
                                .map(|(point, _)| *point)
                                .collect::<Vec<_>>(),
                        )
                        .color(EMPIRICAL)
                        .radius(4.),
                    );
                    for &(point, se) in &empirical {
                        if let Some(se) = se {
                            plot.line(
                                Line::new(
                                    "",
                                    vec![
                                        [point[0], point[1] - 1.96 * se],
                                        [point[0], point[1] + 1.96 * se],
                                    ],
                                )
                                .color(EMPIRICAL),
                            );
                        }
                    }
                }
            });
            egui::Frame::new()
                .inner_margin(8)
                .fill(crate::theme::BG)
                .stroke(egui::Stroke::new(1., crate::theme::GRID))
                .corner_radius(5)
                .show(ui, |ui| {
                    ui.set_min_width(200.);
                    ui.vertical(|ui| {
                        if short {
                            ui.spacing_mut().item_spacing.y = 1.;
                            ui.spacing_mut().interact_size.y = 16.;
                        }
                        action_legend_key(ui, "Learned critic", CYAN, ActionLegendMark::Line);
                        if reference_point.is_some() {
                            action_legend_key(
                                ui,
                                "Reference value",
                                REFERENCE,
                                ActionLegendMark::Reference,
                            );
                        }
                        if !empirical.is_empty() {
                            action_legend_key(
                                ui,
                                "Independent futures · ±1.96 SE",
                                EMPIRICAL,
                                ActionLegendMark::Empirical,
                            );
                        }
                        if selected.is_some() {
                            action_legend_key(
                                ui,
                                "Selected action",
                                crate::theme::SELECTED,
                                ActionLegendMark::Selected,
                            );
                        }
                        if maximum.is_some() {
                            action_legend_key(
                                ui,
                                "Grid maximum (sampled)",
                                REFERENCE,
                                ActionLegendMark::Maximum,
                            );
                        }
                    });
                });
        });
    }
    fn position(&self, g: &PreparedValueGrid, i: usize, two: bool) -> Option<[f64; 2]> {
        let v = self.project(g, i)?;
        let p = g.positions.get(i)?;
        Some([p[0] as f64, if two { p[1] as f64 } else { v }])
    }
    pub(super) fn main(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedValueFrame>,
        height: f32,
    ) -> Option<SelectionIntent> {
        let Some(frame) = frame else {
            ui.allocate_ui(egui::vec2(ui.available_width(), height), |ui| {
                ui.centered_and_justified(|ui| {
                    ui.weak(self.status.as_deref().unwrap_or("Loading…"));
                });
            });
            return None;
        };
        let g = &frame.grid;
        let c = &g.slice;
        let two = c.mode == SliceMode::TwoDimensional;
        let kind = if two {
            "Value map"
        } else if matches!(
            c.x,
            ValueCoordinate::Action(_) | ValueCoordinate::DiscreteAction
        ) {
            "Action-value slice"
        } else {
            "Value slice"
        };
        let comparison = match self.display {
            Display::Current => "",
            Display::Reference => " · reference",
            Display::Difference => " · difference",
        };
        let conditioning = if two
            && c.state_only()
            && g.current().kind() != ValueCriticKind::StateValue
            && *g.output() != ValueOutput::DecisionLoss
        {
            match c.conditioning {
                ActionConditioning::Policy => " · policy action",
                ActionConditioning::Fixed(_) => " · fixed action",
            }
        } else {
            ""
        };
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "{kind} · {}{conditioning}{comparison}",
                    source_output_label(Some(g.current()), g.output())
                ))
                .size(17.)
                .strong(),
            );
            if self.ready
                && let Some(status) = self.status.as_ref()
            {
                let label = if status.starts_with("Value reference unavailable") {
                    "Reference unavailable"
                } else {
                    "Details"
                };
                ui.weak(label).on_hover_text(status);
            }
        });
        let height = (height - 24.).max(120.);
        let action_maxima = ActionSliceMaxima::from_grid(g, self.reference_enabled);
        let readout_height = if action_maxima.is_some() {
            ui.spacing().interact_size.y + ui.spacing().item_spacing.y
        } else {
            0.
        };
        let empirical_sparse = g.empirical_reference() && self.display != Display::Current;
        let range = self.scale(g);
        let scale = landscape::LandscapeScale::new(
            range,
            if self.display == Display::Difference {
                landscape::Palette::Difference
            } else {
                landscape::Palette::Value
            },
        );
        let nx = c.axis_len(c.x, 0).unwrap_or(1);
        let ny = c.y.and_then(|a| c.axis_len(a, 1).ok()).unwrap_or(1);
        let categorical =
            c.x == ValueCoordinate::DiscreteAction || c.y == Some(ValueCoordinate::DiscreteAction);
        if two && g.regular() && !categorical && !empirical_sparse {
            let key = (
                Arc::as_ptr(g) as usize,
                self.display,
                range.map(f64::to_bits),
                scale.palette,
            );
            if self.texture_key != Some(key) || self.texture_grid.upgrade().is_none() {
                self.color_clipped = [0; 2];
                for i in 0..g.values.len() {
                    if let Some(value) = self.project(g, i) {
                        self.color_clipped[0] += usize::from(value < range[0]);
                        self.color_clipped[1] += usize::from(value > range[1]);
                    }
                }
                let pixels = (0..ny)
                    .rev()
                    .flat_map(|y| (0..nx).map(move |x| y * nx + x))
                    .map(|i| scale.color(self.project(g, i).unwrap_or(f64::NAN)))
                    .collect();
                let image = ColorImage::new([nx, ny], pixels);
                if let Some(t) = &mut self.texture {
                    t.set(image, TextureOptions::LINEAR);
                } else {
                    self.texture = Some(ui.ctx().load_texture(
                        "Value",
                        image,
                        TextureOptions::LINEAR,
                    ));
                }
                self.texture_key = Some(key);
                self.texture_grid = Arc::downgrade(g);
                self.texture_uploads += 1;
            }
        }
        let bounds = |a: ValueCoordinate, n: usize| -> [f64; 2] {
            if a == ValueCoordinate::DiscreteAction {
                let values = (0..n)
                    .filter_map(|i| c.coordinate_at(a, i, n).ok())
                    .map(f64::from);
                let b = landscape::numeric_bounds(values, false);
                [b[0] - 0.5, b[1] + 0.5]
            } else {
                match a {
                    ValueCoordinate::Observation(i) => c.state.ranges[i].map(f64::from),
                    ValueCoordinate::Action(i) => c.action_ranges[i].map(f64::from),
                    _ => unreachable!(),
                }
            }
        };
        let xr = bounds(c.x, nx);
        let yr = if !two && c.x == ValueCoordinate::DiscreteAction {
            // A Q bar has a meaningful zero baseline. Fit every rendered value,
            // including a genuine reference overlay, with room for pin markers.
            // This extent is applied only on initialization/quantity change below.
            let (lo, hi) = (0..g.values.len())
                .filter_map(|i| self.project(g, i))
                .chain(
                    g.reference_values
                        .iter()
                        .copied()
                        .flatten()
                        .filter(|_| self.reference_enabled && self.display == Display::Current),
                )
                .filter(|v| v.is_finite())
                .fold((0.0_f64, 0.0_f64), |(lo, hi), v| (lo.min(v), hi.max(v)));
            let padding = ((hi - lo) * 0.05).max(1e-6);
            [lo - padding, hi + padding]
        } else {
            c.y.map(|a| bounds(a, ny)).unwrap_or(range)
        };
        let plot = Plot::new("value-main")
            .width((ui.available_width() - if two { 86. } else { 0. }).max(100.))
            .show_background(false)
            .grid_color(crate::theme::GRID)
            .show_grid(!(two && g.regular() && !categorical))
            .id(Self::main_plot_id())
            .height((height - readout_height).max(120.))
            .grid_spacing(40.0..=100.0)
            .x_grid_spacer(egui_plot::log_grid_spacer(2))
            .y_grid_spacer(egui_plot::log_grid_spacer(2))
            .custom_x_axes(vec![
                egui_plot::AxisHints::new(egui_plot::Axis::X).label_spacing(90.0..=120.0),
            ])
            .x_axis_formatter(landscape::axis_tick)
            .y_axis_formatter(landscape::axis_tick)
            .x_axis_label(coordinate_axis_label(e, c.x))
            .y_axis_label(
                c.y.map(|a| coordinate_axis_label(e, a))
                    .unwrap_or_else(|| source_output_label(Some(g.current()), g.output()).into()),
            )
            .allow_double_click_reset(false);
        let initialize = !self.initialized;
        let quantity = (
            self.display,
            g.output().clone(),
            g.empirical_reference() && self.reference_enabled,
        );
        let quantity_changed = self.main_quantity.as_ref() != Some(&quantity);
        let response = plot.show(ui, |p| {
            if initialize {
                let mut visible_y = yr;
                if two && empirical_sparse {
                    let pad = (yr[1] - yr[0]) * 0.04;
                    visible_y = [yr[0] - pad, yr[1] + pad];
                }
                p.set_plot_bounds(PlotBounds::from_min_max(
                    [xr[0], visible_y[0]],
                    [xr[1], visible_y[1]],
                ));
            }
            if quantity_changed && !initialize && !two {
                let bounds = p.plot_bounds();
                p.set_plot_bounds(PlotBounds::from_min_max(
                    [bounds.min()[0], yr[0]],
                    [bounds.max()[0], yr[1]],
                ));
            }
            let categorical = c.x == ValueCoordinate::DiscreteAction
                || c.y == Some(ValueCoordinate::DiscreteAction);
            if empirical_sparse {
                if self.display == Display::Difference
                    && let Some(series) = &g.empirical_differences
                {
                    for (&position, &value) in series.positions.iter().zip(&series.values) {
                        p.points(
                            Points::new(
                                "Critic − conditional mean (measured input)",
                                vec![[
                                    position[0] as f64,
                                    if two { position[1] as f64 } else { value },
                                ]],
                            )
                            .color(if two {
                                scale.color(value)
                            } else {
                                crate::theme::CURRENT
                            })
                            .radius(5.),
                        );
                    }
                }
            } else if two && g.regular() && !categorical {
                if let Some(t) = &self.texture {
                    let expand = |r: [f64; 2], n: usize| {
                        let step = (r[1] - r[0]) / (n.saturating_sub(1).max(1)) as f64;
                        [r[0] - step * 0.5, r[1] + step * 0.5]
                    };
                    let x = if c.x == ValueCoordinate::DiscreteAction {
                        xr
                    } else {
                        expand(xr, nx)
                    };
                    let y = if c.y == Some(ValueCoordinate::DiscreteAction) {
                        yr
                    } else {
                        expand(yr, ny)
                    };
                    p.image(PlotImage::new(
                        "Value",
                        t.id(),
                        PlotPoint::new((x[0] + x[1]) / 2., (y[0] + y[1]) / 2.),
                        egui::vec2((x[1] - x[0]) as f32, (y[1] - y[0]) as f32),
                    ));
                }
            } else if two && categorical {
                for i in 0..g.values.len() {
                    if let Some(v) = self.project(g, i) {
                        let pos = g.positions[i];
                        let dx = if c.x == ValueCoordinate::DiscreteAction {
                            0.4
                        } else {
                            (xr[1] - xr[0]) / (nx.saturating_sub(1).max(1)) as f64 * 0.5
                        };
                        let dy = if c.y == Some(ValueCoordinate::DiscreteAction) {
                            0.4
                        } else {
                            (yr[1] - yr[0]) / (ny.saturating_sub(1).max(1)) as f64 * 0.5
                        };
                        let x = pos[0] as f64;
                        let y = pos[1] as f64;
                        p.polygon(
                            Polygon::new(
                                "Legal value cell",
                                vec![
                                    [x - dx, y - dy],
                                    [x + dx, y - dy],
                                    [x + dx, y + dy],
                                    [x - dx, y + dy],
                                ],
                            )
                            .fill_color(scale.color(v))
                            .stroke(egui::Stroke::NONE),
                        );
                    }
                }
            } else if !two && c.x == ValueCoordinate::DiscreteAction {
                let greedy = frame.selected.as_ref().and_then(|p| {
                    if let ValuePrediction::Discrete { values, legal } = &p.prediction {
                        values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| legal[*i])
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(i, _)| i)
                    } else {
                        None
                    }
                });
                let bars = (0..g.values.len())
                    .filter_map(|i| {
                        self.position(g, i, false).map(|v| {
                            Bar::new(v[0], v[1]).width(0.75).fill(
                                if greedy == Some(v[0] as usize) {
                                    CYAN
                                } else {
                                    crate::theme::CURRENT
                                },
                            )
                        })
                    })
                    .collect();
                p.bar_chart(
                    BarChart::new("Legal action values", bars).color(crate::theme::CURRENT),
                );
            } else if !two
                && g.regular()
                && (self.display == Display::Current
                    || g.reference().is_some_and(|s| !s.identity().recorded)
                        && g.reference_values.iter().all(Option::is_some))
            {
                let line = Line::new(
                    if self.display == Display::Reference {
                        "Reference"
                    } else {
                        "Current"
                    },
                    (0..g.values.len())
                        .filter_map(|i| self.position(g, i, false))
                        .collect::<Vec<_>>(),
                );
                if self.display == Display::Reference {
                    p.add(dashes::DashedLine::curve(
                        line.color(REFERENCE),
                        1.,
                        self.dash_admission.clone(),
                    ));
                } else {
                    p.line(line.color(crate::theme::CURRENT));
                }
            } else if !two {
                p.points(
                    Points::new(
                        "Value",
                        (0..g.values.len())
                            .filter_map(|i| self.position(g, i, false))
                            .collect::<Vec<_>>(),
                    )
                    .color(if self.display == Display::Reference {
                        REFERENCE
                    } else {
                        crate::theme::CURRENT
                    })
                    .radius(4.),
                );
            } else {
                // Preserve every sample and its exact existing color. Admission has already
                // charged the worst case (one color batch per row), including map/item storage.
                let mut batches = std::collections::BTreeMap::<[u8; 4], Vec<[f64; 2]>>::new();
                for i in 0..g.values.len() {
                    if let Some(pos) = self.position(g, i, two) {
                        let color = scale.color(self.project(g, i).unwrap());
                        batches.entry(color.to_array()).or_default().push(pos);
                    }
                }
                for (color, points) in batches {
                    p.points(
                        Points::new("Value", points)
                            .color(Color32::from_rgba_premultiplied(
                                color[0], color[1], color[2], color[3],
                            ))
                            .radius(4.),
                    );
                }
            }
            // Recorded evidence is always sparse points. No line spans a missing input.
            if self.reference_enabled && self.display != Display::Difference {
                if let Some(s) = &g.reference_series {
                    let points = s
                        .positions
                        .iter()
                        .zip(&s.values)
                        .map(|(pos, &v)| [pos[0] as f64, if two { pos[1] as f64 } else { v }])
                        .collect::<Vec<_>>();
                    p.points(
                        Points::new(presentation::reference_series_label(g), points)
                            .color(if g.empirical_reference() {
                                EMPIRICAL
                            } else {
                                REFERENCE
                            })
                            .shape(if g.empirical_reference() {
                                egui_plot::MarkerShape::Circle
                            } else {
                                egui_plot::MarkerShape::Diamond
                            })
                            .radius(4.),
                    );
                } else if !two && self.display == Display::Current {
                    let points = (0..g.values.len())
                        .filter_map(|i| {
                            g.reference_values
                                .get(i)
                                .copied()
                                .flatten()
                                .map(|v| [g.positions[i][0] as f64, v])
                        })
                        .collect::<Vec<_>>();
                    if g.reference().is_some_and(|s| !s.identity().recorded)
                        && g.reference_values.iter().all(Option::is_some)
                        && c.x != ValueCoordinate::DiscreteAction
                    {
                        p.add(dashes::DashedLine::curve(
                            Line::new("Reference", points).color(REFERENCE),
                            1.,
                            self.dash_admission.clone(),
                        ));
                    } else {
                        p.points(Points::new("Reference", points).color(REFERENCE).radius(3.));
                    }
                }
            }
            for (label, overlay, color, dashed) in [
                ("Current actor", &g.policy_overlay, CYAN, false),
                (
                    "Reference controller",
                    &g.reference_policy_overlay,
                    REFERENCE,
                    true,
                ),
            ] {
                let mut points = Vec::new();
                for (i, a) in overlay.iter().enumerate() {
                    if let Some(a) = a
                        && let Some(pos) = overlay_position(c, g.positions[i], a)
                    {
                        let keep = overlay_kept(c, i, nx);
                        if keep {
                            points.push(pos);
                        }
                    }
                }
                if c.y.is_none() && matches!(c.x, ValueCoordinate::Action(_)) {
                    for pos in points {
                        if dashed {
                            p.add(dashes::DashedLine::vertical(
                                label,
                                pos[0],
                                color,
                                2.,
                                self.dash_admission.clone(),
                            ));
                        } else {
                            p.vline(VLine::new(label, pos[0]).color(color).width(2.));
                        }
                    }
                } else if !points.is_empty() {
                    let mixed = matches!(
                        (c.x, c.y),
                        (
                            ValueCoordinate::Observation(_),
                            Some(ValueCoordinate::Action(_))
                        ) | (
                            ValueCoordinate::Action(_),
                            Some(ValueCoordinate::Observation(_))
                        )
                    );
                    if mixed
                        && g.regular()
                        && (!dashed || g.reference_policy().is_some_and(|s| !s.identity().recorded))
                    {
                        let line = Line::new(label, points).color(color).width(2.);
                        if dashed {
                            p.add(dashes::DashedLine::curve(
                                line,
                                2.,
                                self.dash_admission.clone(),
                            ));
                        } else {
                            p.line(line);
                        }
                    } else {
                        p.points(Points::new(label, points).color(color).radius(3.));
                    }
                }
            }
            if let Some(s) = &g.reference_policy_series {
                let points = s
                    .positions
                    .iter()
                    .zip(&s.actions)
                    .filter_map(|(&pos, a)| overlay_position(c, pos, a))
                    .collect::<Vec<_>>();
                p.points(
                    Points::new(
                        if s.stochastic {
                            "Reference sampled actions"
                        } else {
                            "Reference controller samples"
                        },
                        points,
                    )
                    .color(REFERENCE)
                    .radius(4.),
                );
            }
            if self.visits_enabled
                && let Some(visits) = &g.visits
            {
                p.points(
                    Points::new(
                        "Visited states",
                        visits
                            .markers
                            .iter()
                            .map(|v| {
                                [
                                    v.position[0] as f64,
                                    if two { v.position[1] as f64 } else { range[0] },
                                ]
                            })
                            .collect::<Vec<_>>(),
                    )
                    .radius(2.5)
                    .color(Color32::from_white_alpha(85)),
                );
            }
            if let Some(maxima) = &action_maxima
                && self.display != Display::Difference
            {
                for (maximum, label, color, shape) in [
                    (
                        maxima.current.filter(|_| self.display == Display::Current),
                        "Current sampled maximum",
                        CYAN,
                        egui_plot::MarkerShape::Square,
                    ),
                    (
                        maxima.reference,
                        "Reference sampled maximum",
                        REFERENCE,
                        egui_plot::MarkerShape::Diamond,
                    ),
                ] {
                    if let Some(maximum) = maximum {
                        p.points(
                            Points::new(label, vec![maximum.point])
                                .color(color)
                                .shape(shape)
                                .radius(6.),
                        );
                    }
                }
            }
            if let Some(point) = &frame.selected
                && let Some(pos) = point_position(c, point, two, self.display)
            {
                p.points(
                    Points::new("Selected", vec![pos])
                        .color(crate::theme::SELECTED)
                        .radius(7.),
                );
                p.points(
                    Points::new("Selected", vec![pos])
                        .color(crate::theme::CURRENT)
                        .radius(4.5),
                );
            }
            (p.pointer_coordinate(), *p.transform())
        });
        self.initialized = true;
        self.main_quantity = Some(quantity);
        if let Some(maxima) = &action_maxima {
            maxima.show(ui);
        }
        if two {
            let bar = landscape::color_bar(
                ui,
                *response.inner.1.frame(),
                scale,
                source_output_label(Some(g.current()), g.output()),
            );
            if g.regular() && !categorical && !empirical_sparse {
                for (end, count) in self.color_clipped.iter().copied().enumerate() {
                    // Normal percentile trimming is expected; flag only >5% of cells.
                    if count.saturating_mul(20) <= g.values.len() {
                        continue;
                    }
                    let center = if end == 0 {
                        bar.center_bottom() + egui::vec2(0., 7.)
                    } else {
                        bar.center_top() - egui::vec2(0., 7.)
                    };
                    ui.painter().text(
                        center,
                        egui::Align2::CENTER_CENTER,
                        if end == 0 { "▼" } else { "▲" },
                        egui::FontId::proportional(10.),
                        crate::theme::MUTED,
                    );
                    ui.interact(egui::Rect::from_center_size(center, egui::vec2(16., 12.)), ui.scope_id().with(("value-clipped", end)), egui::Sense::hover())
                        .on_hover_text(format!("{count} cells {} the colour limit; colours saturate but numerical values are unchanged. Inspect exact predictions on hover or change Color scale in Slice settings.", if end == 0 { "below" } else { "above" }));
                }
            }
        }
        // Retained output keeps its plot geometry while its new request is pending
        // or unavailable. Show any persistent explanation inside the plot.
        if *g.output() == ValueOutput::StandardDeviation {
            let invalid = g.valid.iter().filter(|&&valid| !valid).count();
            if invalid > 0 {
                ui.small(format!("Standard deviation unavailable at {invalid} inputs with negative implied variance"));
            }
        }
        if !self.ready
            && let Some(status) = landscape::complete_frame_status(self.status.as_deref(), false)
            && let Some(rect) = self.main_rect
        {
            let painter = ui.painter().with_clip_rect(rect);
            painter.text(
                rect.left_top() + egui::vec2(70., 30.),
                egui::Align2::LEFT_TOP,
                status,
                egui::FontId::proportional(11.),
                crate::theme::MUTED,
            );
        }
        let cursor = response
            .inner
            .0
            .filter(|_| response.response.contains_pointer())?;
        let transform = &response.inner.1;
        let visit_hit = if self.visits_enabled {
            g.visits.as_ref().and_then(|v| {
                landscape::nearest_point(
                    v.markers.iter().enumerate().map(|(i, v)| {
                        (
                            i,
                            [
                                v.position[0] as f64,
                                if two { v.position[1] as f64 } else { range[0] },
                            ],
                        )
                    }),
                    cursor,
                    transform,
                    8.,
                )
            })
        } else {
            None
        };
        let nearest = if empirical_sparse {
            None
        } else if g.regular()
            && c.x != ValueCoordinate::DiscreteAction
            && c.y != Some(ValueCoordinate::DiscreteAction)
        {
            landscape::cell_index(
                [cursor.x, cursor.y],
                [xr.map(|v| v as f32), yr.map(|v| v as f32)],
                [nx, if two { ny } else { 1 }],
            )
            .filter(|&i| self.project(g, i).is_some())
        } else {
            landscape::nearest_point(
                (0..g.values.len()).filter_map(|i| {
                    self.position(g, i, two)
                        .filter(|p| category_hit(c, *p, cursor))
                        .map(|p| (i, p))
                }),
                cursor,
                transform,
                10.,
            )
        };
        let reference_hit = if self.reference_enabled && self.display != Display::Difference {
            g.reference_series.as_ref().and_then(|s| {
                landscape::nearest_point(
                    s.positions.iter().zip(&s.values).enumerate().filter_map(
                        |(i, (pos, &value))| {
                            let p = [pos[0] as f64, if two { pos[1] as f64 } else { value }];
                            category_hit(c, p, cursor).then_some((i, p))
                        },
                    ),
                    cursor,
                    transform,
                    10.,
                )
            })
        } else {
            None
        };
        let empirical_series = if self.display == Display::Difference {
            g.empirical_differences.as_ref()
        } else {
            g.reference_series.as_ref()
        };
        let empirical_hit = if g.empirical_reference() {
            empirical_series.and_then(|s| {
                landscape::nearest_point(
                    s.positions
                        .iter()
                        .zip(&s.values)
                        .enumerate()
                        .map(|(i, (pos, &v))| {
                            (i, [pos[0] as f64, if two { pos[1] as f64 } else { v }])
                        }),
                    cursor,
                    transform,
                    10.,
                )
            })
        } else {
            None
        };
        let overlay_hit = [&g.policy_overlay, &g.reference_policy_overlay]
            .into_iter()
            .enumerate()
            .find_map(|(which, overlay)| {
                landscape::nearest_point(
                    overlay.iter().enumerate().filter_map(|(i, a)| {
                        if !overlay_kept(c, i, nx) {
                            return None;
                        }
                        let mut pos = overlay_position(c, g.positions[i], a.as_ref()?)?;
                        if c.y.is_none() {
                            pos[1] = cursor.y;
                        }
                        Some((i, pos))
                    }),
                    cursor,
                    transform,
                    8.,
                )
                .map(|i| (which, i))
            });
        if let Some(i) = empirical_hit {
            if let Some(condition) = empirical_series
                .and_then(|s| s.rows.get(i))
                .and_then(|&i| g.empirical.conditions.get(i))
            {
                response.response.clone().on_hover_text(format!("Conditional empirical mean {:.8} · n={} · mean SE {:?}\nActual k={} · record #{} · stable row {}\nComplete observation {:?}\nMask {:?}\nFirst action {:?}\n{}\nUnsupported locations are empty. Click to adopt this exact measured input/action.",condition.summary.mean,condition.summary.count,condition.summary.mean_standard_error,condition.revision,condition.source_key.sequence,condition.row_id,condition.input.dense_values(),condition.input.dense_mask(),condition.action,condition.methodology));
            }
        } else if let Some(i) = visit_hit {
            let v = &g.visits.as_ref()?.markers[i];
            response.response.clone().on_hover_text(match &v.source.chunk.record {rl::diagnostics::schema::Record::Trajectory(t)=>format!("Episode {} · step {} · recorded k={}\nClick for the exact original state/action and observed return. Repeated clicks cycle coincident decisions.",t.id,v.row,v.source.chunk.context.identity.learner_update_revision),_=>format!("Genuine visited probe row {}",v.row)});
        } else if let Some((which, i)) = overlay_hit {
            let (label, actions, stochastic) = if which == 0 {
                (
                    "Current actor",
                    &g.policy_overlay,
                    g.current().actor_stochastic(),
                )
            } else {
                (
                    "Reference controller",
                    &g.reference_policy_overlay,
                    g.reference_policy_identity
                        .as_ref()
                        .is_some_and(|p| p.stochastic),
                )
            };
            if let Some(action) = &actions[i] {
                let qualifiers =
                    presentation::overlay_qualifiers(c, action, g.actions[i].as_ref(), stochastic);
                let provenance = if which == 0 {
                    format!(
                        "Actor {}\n{}",
                        g.current()
                            .actor_name()
                            .unwrap_or("legal greedy controller"),
                        presentation::value_source(
                            "Current critic",
                            g.current(),
                            Some(frame.requested_revision),
                            Some(action)
                        )
                    )
                } else if let (Some(source), Some(meta)) =
                    (g.reference_policy(), &g.reference_policy_identity)
                {
                    format!(
                        "{} · {} · actual k={} / requested k={}\nComplete first action: {}",
                        meta.name,
                        meta.role,
                        source.identity().revision,
                        frame
                            .reference_requested_revision
                            .map_or_else(|| "unknown".into(), |k| k.to_string()),
                        presentation::action(e, Some(action))
                    )
                } else {
                    format!(
                        "Complete first action: {}",
                        presentation::action(e, Some(action))
                    )
                };
                response
                    .response
                    .clone()
                    .on_hover_text(format!("{label}{qualifiers}\n{provenance}"));
            }
        } else if let Some(i) = reference_hit.filter(|_| nearest.is_none()) {
            if let (Some(series), Some(source)) = (&g.reference_series, g.reference())
                && let Ok((input, _)) = source.recorded_input(series.rows[i])
            {
                response.response.clone().on_hover_text(format!("Reference sample {:.8}\n{}\nRecorded row {}\nComplete state {:?}\nMask {:?}\n{}\nCurrent value unavailable at this input; excluded from paired differences",series.values[i],presentation::reference_series_label(g),series.rows[i],input.dense_values(),input.dense_mask(),presentation::value_source("Reference", source, frame.reference_requested_revision, series.actions[i].as_ref())));
            }
        } else if let Some(i) = nearest
            && let Ok(input) = g.input(i)
        {
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                response.response.layer_id,
                response.response.id.with("value-input"),
                egui::PopupAnchor::Pointer,
            )
            .show(|ui| {
                ui.set_max_width(440.);
                egui::ScrollArea::vertical()
                    .max_height(450.)
                    .show(ui, |ui| {
                        for (j, v) in input.dense_values().iter().enumerate() {
                            ui.label(format!("{} = {v:?}", landscape::observation_label(e, j)));
                        }
                        if let Some(mask) = &input.dense_mask() {
                            ui.label(format!("Mask {mask:?}"));
                        }
                        ui.label(format!(
                            "{} {:.8}",
                            source_output_label(Some(g.current()), g.output()),
                            g.values[i]
                        ));
                        if let Some(v) = g.reference_values.get(i).copied().flatten() {
                            ui.label(format!(
                                "Reference {v:.8} · difference {:.8}",
                                g.values[i] - v
                            ));
                        }
                        ui.label(presentation::comparison(g.comparison_condition()));
                        ui.label(presentation::value_source(
                            "Current",
                            g.current(),
                            Some(frame.requested_revision),
                            g.actions.get(i).and_then(|a| a.as_ref()),
                        ));
                        if g.reference_values.get(i).is_some_and(Option::is_some)
                            && let Some(source) = g.reference()
                        {
                            ui.label(presentation::value_source(
                                "Reference",
                                source,
                                frame.reference_requested_revision,
                                g.reference_actions.get(i).and_then(|a| a.as_ref()),
                            ));
                        }
                    });
            });
        }
        if response.response.clicked() {
            if let Some(condition) = empirical_hit
                .and_then(|i| empirical_series.and_then(|s| s.rows.get(i)))
                .and_then(|&i| g.empirical.conditions.get(i))
            {
                let _ = self.adopt_empirical(condition);
                return self
                    .pending_empirical_input
                    .take()
                    .map(SelectionIntent::Input);
            }
            if let Some(i) = visit_hit {
                // Repeated clicks cycle genuine coincident decisions; the list also selects each identity.
                let visits = g.visits.as_ref()?;
                let position = visits.markers[i].position;
                let coincident: Vec<_> = visits
                    .markers
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.position == position)
                    .map(|(i, _)| i)
                    .collect();
                let selected = coincident[self.visit_cycle % coincident.len()];
                self.visit_cycle = self.visit_cycle.wrapping_add(1);
                return self.select_visit(g, selected).ok();
            }
            if let Some(i) = nearest
                && self.select_cell(g, i).is_ok()
                && g.regular()
                && c.x != ValueCoordinate::DiscreteAction
                && c.y != Some(ValueCoordinate::DiscreteAction)
            {
                let (mut input, mut action) = match &self.pin {
                    Some(ValuePin::State(o)) => (o.clone(), None),
                    Some(ValuePin::Input {
                        observation,
                        action,
                    }) => (observation.clone(), action.clone()),
                    _ => (g.input(i).ok()?, g.actions[i].clone()),
                };
                for (axis, value) in std::iter::once(c.x).chain(c.y).zip([cursor.x, cursor.y]) {
                    match axis {
                        ValueCoordinate::Observation(j) => {
                            input.as_dense_mut().expect("dense diagnostic input").values[j] =
                                value as f32
                        }
                        ValueCoordinate::Action(j) => {
                            if let Some(Action::Continuous(a)) = &mut action {
                                a[j] = value as f32;
                            }
                        }
                        _ => {}
                    }
                }
                let _ = self.pin_input(input, action);
            }
        }
        None
    }
}
fn action_value(a: &Action, i: usize) -> Option<f64> {
    match a {
        Action::Continuous(v) => v.get(i).map(|&v| v as f64),
        Action::Discrete(i) => Some(*i as f64),
        Action::Conditional(_) => None,
    }
}
pub(super) fn overlay_position(c: &ValueSlice, pos: [f32; 2], a: &Action) -> Option<[f64; 2]> {
    match (c.x, c.y) {
        (ValueCoordinate::Observation(_), Some(ValueCoordinate::Action(i))) => {
            Some([pos[0] as f64, action_value(a, i)?])
        }
        (ValueCoordinate::Action(i), Some(ValueCoordinate::Observation(_))) => {
            Some([action_value(a, i)?, pos[1] as f64])
        }
        (ValueCoordinate::Action(i), Some(ValueCoordinate::Action(j))) => {
            Some([action_value(a, i)?, action_value(a, j)?])
        }
        (ValueCoordinate::Action(i), None) => Some([action_value(a, i)?, 0.]),
        (ValueCoordinate::DiscreteAction, None) => Some([action_value(a, 0)?, 0.]),
        _ => None,
    }
}
pub(super) fn overlay_kept(c: &ValueSlice, i: usize, nx: usize) -> bool {
    match (c.x, c.y) {
        (ValueCoordinate::Observation(_), Some(ValueCoordinate::Action(_))) => i < nx,
        (ValueCoordinate::Action(_), Some(ValueCoordinate::Observation(_))) => i.is_multiple_of(nx),
        _ => i == 0,
    }
}
fn point_position(
    c: &ValueSlice,
    p: &PreparedValuePoint,
    two: bool,
    display: Display,
) -> Option<[f64; 2]> {
    let coord = |a| match a {
        ValueCoordinate::Observation(i) => p.input.dense_values().get(i).map(|&v| v as f64),
        ValueCoordinate::Action(i) => p.action.as_ref().and_then(|a| action_value(a, i)),
        ValueCoordinate::DiscreteAction => p.action.as_ref().and_then(|a| action_value(a, 0)),
    };
    Some([
        coord(c.x)?,
        if two {
            coord(c.y?)?
        } else {
            match display {
                Display::Current => p.value,
                Display::Reference => p.reference_value?,
                Display::Difference => p.value - p.reference_value?,
            }
        },
    ])
}
fn category_hit(c: &ValueSlice, point: [f64; 2], cursor: PlotPoint) -> bool {
    (c.x != ValueCoordinate::DiscreteAction || (point[0] - cursor.x).abs() <= 0.4)
        && (c.y != Some(ValueCoordinate::DiscreteAction) || (point[1] - cursor.y).abs() <= 0.4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_grid_maxima_measure_displacement_only_on_complete_unique_support() {
        let positions = [[-2., 0.], [0., 0.], [2., 0.]];
        let current = sampled_maximum(&positions, [Some(-5.), Some(-3.), Some(-6.)].into_iter());
        let reference = sampled_maximum(&positions, [Some(-6.), Some(-4.), Some(-2.)].into_iter());
        assert_eq!(current.unwrap().point, [0., -3.]);
        assert_eq!(reference.unwrap().point, [2., -2.]);
        assert_eq!(
            maximum_displacement(current, reference, positions.len()),
            Some(2.)
        );

        let partial = sampled_maximum(&positions, [Some(-6.), None, Some(-2.)].into_iter());
        assert_eq!(partial.unwrap().point, [2., -2.]);
        assert_eq!(partial.unwrap().samples, 2);
        assert_eq!(
            maximum_displacement(current, partial, positions.len()),
            None
        );

        let tied = sampled_maximum(&positions, [Some(-3.), Some(-3.), Some(-8.)].into_iter());
        assert!(tied.unwrap().tied);
        assert_eq!(tied.unwrap().point, [-2., -3.]);
        assert_eq!(maximum_displacement(tied, reference, positions.len()), None);
        assert!(
            sampled_maximum(
                &positions,
                [None, Some(f64::NAN), Some(f64::INFINITY)].into_iter()
            )
            .is_none()
        );
    }
}
