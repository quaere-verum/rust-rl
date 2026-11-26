use super::{ROW_LIMIT, ReferenceRow, View, egui};
use crate::theme::{CURRENT, EMPIRICAL, REFERENCE};
use eframe::egui::Color32;
use egui_plot::{Bar, BarChart, Legend, MarkerShape, Plot, Points};
use rl::{
    core::{
        action::{Action, ActionArgument, ActionSpace, FiniteActionKey},
        env::Observation,
    },
    diagnostics::typed::{DiagnosticInput, FiniteCandidate, FiniteDecision, FiniteValueKind},
};
use std::collections::BTreeMap;

pub(super) fn draw_population(
    ui: &mut egui::Ui,
    inputs: &[DiagnosticInput],
    rows: &[FiniteDecision],
    space: &ActionSpace,
    selected: Option<usize>,
) -> Option<usize> {
    let DiagnosticInput::Projected { projection, .. } = inputs.first()? else {
        return None;
    };
    let [x, y] = projection.labels.as_slice() else {
        return None;
    };
    let coordinate = |input: &DiagnosticInput| match input {
        DiagnosticInput::Projected { projection, .. }
            if projection.labels.as_slice() == [x.as_str(), y.as_str()] =>
        {
            Some([projection.coordinates[0], projection.coordinates[1]])
        }
        _ => None,
    };
    let coordinates = inputs.iter().map(coordinate).collect::<Option<Vec<_>>>()?;
    let mut actions = BTreeMap::<FiniteActionKey, Vec<[f64; 2]>>::new();
    for (&point, row) in coordinates.iter().zip(rows) {
        if let Some(action) = row.selected {
            actions.entry(action).or_default().push(point);
        }
    }
    if actions.is_empty() || actions.len() > 8 {
        return None;
    }
    ui.strong("Selected action by state");
    // Keep a useful projection as the principal view while leaving room for
    // the selected action and input below it.
    let height = (ui.available_height() * 0.58).clamp(160., 440.);
    let bounds =
        |axis| crate::landscape::numeric_bounds(coordinates.iter().map(|p| p[axis]), false);
    let [x_min, x_max] = bounds(0);
    let [y_min, y_max] = bounds(1);
    let x_margin = ((x_max - x_min) * 0.12).max(0.25);
    let y_margin = ((y_max - y_min) * 0.12).max(0.25);
    let plot_width = ((x_max - x_min + 2. * x_margin) / (y_max - y_min + 2. * y_margin))
        .clamp(1.0, 2.3) as f32
        * height
        + 68.;
    let response = Plot::new("typed-policy-projection")
        .width(plot_width.min(ui.available_width()))
        .height(height)
        .data_aspect(1.0)
        .include_x(x_min - x_margin)
        .include_x(x_max + x_margin)
        .include_y(y_min - y_margin)
        .include_y(y_max + y_margin)
        .x_axis_label(x)
        .y_axis_label(y)
        .legend(Legend::default())
        .show(ui, |plot| {
            for (i, (action, points)) in actions.into_iter().enumerate() {
                let color = [
                    CURRENT,
                    REFERENCE,
                    EMPIRICAL,
                    Color32::from_rgb(243, 150, 182),
                ][i % 4];
                let shape = [
                    MarkerShape::Square,
                    MarkerShape::Diamond,
                    MarkerShape::Up,
                    MarkerShape::Down,
                ][i % 4];
                plot.points(
                    Points::new(key_label(action, space), points)
                        .shape(shape)
                        .color(color)
                        .radius(7.),
                );
            }
            if let Some(selected) = selected {
                plot.points(
                    Points::new("Selected", vec![coordinates[selected]])
                        .shape(MarkerShape::Cross)
                        .color(crate::theme::SELECTED)
                        .radius(8.),
                );
            }
            plot.pointer_coordinate()
        });
    if response.response.clicked()
        && let Some(pointer) = response.inner
    {
        return coordinates
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let distance =
                    |point: &[f64; 2]| (point[0] - pointer.x).hypot(point[1] - pointer.y);
                distance(a).total_cmp(&distance(b))
            })
            .map(|(row, _)| row);
    }
    None
}

pub(super) fn input_label(input: &DiagnosticInput) -> String {
    let obs = input.observation();
    if let Some(discrete) = obs.as_discrete() {
        format!("State {}", discrete.symbol)
    } else if let Some(graph) = obs.as_graph() {
        format!("Graph · tick {}", graph.tick)
    } else if let Some(history) = input.public_history() {
        format!("History · t {}", history.timestep())
    } else {
        "Observation".into()
    }
}

pub(super) fn draw_input(ui: &mut egui::Ui, input: &DiagnosticInput) {
    draw_observation(ui, input.observation());
    if let DiagnosticInput::Projected { projection, .. } = input {
        for (label, coordinate) in projection.labels.iter().zip(&projection.coordinates) {
            ui.label(format!("{label}  {coordinate:.4}"));
        }
    }
}

pub(super) fn draw_observation(ui: &mut egui::Ui, obs: &Observation) {
    if let Some(discrete) = obs.as_discrete() {
        ui.label(format!("Observation symbol  {}", discrete.symbol));
    }
    if let Some(dense) = obs.as_dense() {
        ui.monospace(numbers(&dense.values));
    }
    if let Some(frame) = obs.as_structured() {
        if !frame.scalars.is_empty() {
            ui.label(format!("Scalars  {}", numbers(&frame.scalars)));
        }
        if let Some(spatial) = &frame.spatial {
            ui.label(format!(
                "Region {} · {} × {}",
                frame.primary_region, spatial.width, spatial.height
            ));
        }
        ui.collapsing(format!("Entities · {}", frame.entities.len()), |ui| {
            for row in frame.entities.iter().take(ROW_LIMIT) {
                ui.monospace(format!("{:?}  {}", row.id, numbers(&row.features)));
            }
        });
        ui.collapsing(format!("Public events · {}", frame.events.len()), |ui| {
            for row in frame.events.iter().rev().take(ROW_LIMIT) {
                ui.monospace(format!("{:?} · {}", row.stamp, numbers(&row.features)));
            }
        });
    }
    if let Some(graph) = obs.as_graph() {
        ui.collapsing(format!("Nodes · {}", graph.nodes.len()), |ui| {
            for node in graph.nodes.iter().take(ROW_LIMIT) {
                ui.monospace(format!("{:?}  {}", node.id, numbers(&node.features)));
            }
        });
        ui.collapsing(format!("Edges · {}", graph.edges.len()), |ui| {
            for edge in graph.edges.iter().take(ROW_LIMIT) {
                ui.monospace(format!(
                    "#{} · {:?} → {:?} · kind {} · {:?}",
                    edge.id, edge.from, edge.to, edge.kind, edge.status
                ))
                .on_hover_text(numbers(&edge.features));
            }
        });
    }
}

fn numbers(values: &[f32]) -> String {
    values
        .iter()
        .take(16)
        .map(|x| format!("{x:.3}"))
        .collect::<Vec<_>>()
        .join("  ")
}

pub(super) fn action_label(
    action: &Action,
    observation: &Observation,
    space: &ActionSpace,
) -> String {
    if let Ok(key) = observation.finite_action_key(action) {
        key_label(key, space)
    } else if let Action::Continuous(values) = action {
        numbers(values)
    } else {
        "Action unavailable".into()
    }
}

fn key_label(key: FiniteActionKey, space: &ActionSpace) -> String {
    match (key, space) {
        (FiniteActionKey::Discrete(index), _) => format!("Action {index}"),
        (FiniteActionKey::Conditional(action), ActionSpace::Conditional(grammar)) => {
            let label = grammar
                .operations
                .get(action.operation)
                .map_or("Operation", |op| &op.name);
            match action.argument {
                None => label.into(),
                Some(ActionArgument::Entity(id)) => format!("{label} · {}:{}", id.kind, id.id),
                Some(ActionArgument::Location { x, y }) => format!("{label} · ({x}, {y})"),
                Some(ActionArgument::RegionalLocation { region, x, y }) => {
                    format!("{label} · {region} ({x}, {y})")
                }
            }
        }
        _ => "Unknown action".into(),
    }
}

fn compatible_value_reference(
    reference: Option<&ReferenceRow>,
    value_kind: Option<FiniteValueKind>,
) -> Option<&ReferenceRow> {
    reference.filter(|r| {
        (r.value_kind == Some(FiniteValueKind::PreferenceScore))
            == (value_kind == Some(FiniteValueKind::PreferenceScore))
    })
}

pub(super) fn draw_values(
    ui: &mut egui::Ui,
    decision: &FiniteDecision,
    reference: Option<&ReferenceRow>,
    value_kind: Option<FiniteValueKind>,
    space: &ActionSpace,
) {
    if decision.candidates.len() > 12 || decision.candidates.iter().all(|row| row.value.is_none()) {
        return;
    }
    let reference = compatible_value_reference(reference, value_kind);
    let order: BTreeMap<_, _> = decision
        .candidates
        .iter()
        .map(|row| (row.action, row))
        .collect();
    let comparison = reference.map(|reference| {
        reference
            .decision
            .candidates
            .iter()
            .map(|row| (row.action, row.value))
            .collect::<BTreeMap<_, _>>()
    });
    let chart = |peer| {
        order
            .iter()
            .enumerate()
            .filter_map(|(index, (&key, row))| {
                let offset = if comparison.is_none() {
                    0.
                } else if peer {
                    0.18
                } else {
                    -0.18
                };
                let y = if peer {
                    comparison.as_ref()?.get(&key).copied().flatten()?
                } else {
                    row.value?
                };
                let color = if peer { REFERENCE } else { CURRENT };
                Some(
                    Bar::new(index as f64 + offset, y)
                        .width(0.34)
                        .fill(color)
                        .name(key_label(key, space)),
                )
            })
            .collect::<Vec<_>>()
    };
    let labels: Vec<_> = order.keys().map(|&key| key_label(key, space)).collect();
    let quantity = if value_kind == Some(FiniteValueKind::PreferenceScore) {
        "Preference score"
    } else {
        "Q"
    };
    let plot_height = (ui.available_height() * 0.53).clamp(175., 372.);
    Plot::new("typed-action-values")
        .height(plot_height)
        .x_axis_label("Legal action")
        .y_axis_label(quantity)
        .include_y(0.)
        .x_axis_formatter(move |mark, _| {
            let label = mark.value.round();
            if label >= 0. && (label - mark.value).abs() < 1e-8 {
                labels.get(label as usize).cloned().unwrap_or_default()
            } else {
                String::new()
            }
        })
        .y_axis_formatter(crate::landscape::axis_tick)
        .legend(Legend::default())
        .show(ui, |plot| {
            plot.bar_chart(BarChart::new("Learner", chart(false)).color(CURRENT));
            if let Some(reference) = reference {
                let bars = chart(true);
                if !bars.is_empty() {
                    let name = if reference.value_kind == Some(FiniteValueKind::OptimalActionValue)
                    {
                        format!("Q* · {}", reference.label)
                    } else {
                        reference.label.clone()
                    };
                    plot.bar_chart(BarChart::new(name, bars).color(REFERENCE));
                }
            }
        });
}

pub(super) fn draw_actions(
    ui: &mut egui::Ui,
    decision: &FiniteDecision,
    reference: Option<&ReferenceRow>,
    value_kind: Option<FiniteValueKind>,
    space: &ActionSpace,
    view: View,
) {
    let value = |c: &FiniteCandidate| {
        if view == View::PolicyMap {
            c.probability
        } else {
            c.value
        }
    };
    let mut rows: Vec<_> = decision.candidates.iter().collect();
    rows.sort_by(|a, b| {
        value(b)
            .partial_cmp(&value(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.action.cmp(&b.action))
    });
    let reference = if view == View::PolicyMap {
        reference
    } else {
        compatible_value_reference(reference, value_kind)
    };
    let references = reference.map(|r| {
        r.decision
            .candidates
            .iter()
            .map(|c| (c.action, c))
            .collect::<BTreeMap<_, _>>()
    });
    let label = if view == View::PolicyMap {
        "Probability"
    } else {
        "Prediction"
    };
    egui::ScrollArea::vertical()
        .id_salt("finite-action-menu")
        .show(ui, |ui| {
            egui::Grid::new("finite-action-values")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Legal action");
                    ui.colored_label(CURRENT, label);
                    if let Some(reference) = reference {
                        let caption = if view == View::Value
                            && reference.value_kind == Some(FiniteValueKind::OptimalActionValue)
                        {
                            format!("Q* · {}", reference.label)
                        } else {
                            reference.label.clone()
                        };
                        ui.colored_label(REFERENCE, caption);
                    }
                    let updates = rows.iter().any(|row| row.updates.is_some());
                    if updates {
                        ui.colored_label(EMPIRICAL, "Updates");
                    }
                    ui.end_row();
                    for row in rows.into_iter().take(ROW_LIMIT) {
                        let mut action = egui::RichText::new(key_label(row.action, space));
                        if decision.selected == Some(row.action) {
                            action = action.strong().color(CURRENT);
                        }
                        ui.label(action);
                        ui.monospace(value(row).map_or("—".into(), |x| format!("{x:.4}")));
                        if let Some(references) = &references {
                            let prediction = references.get(&row.action).and_then(|r| value(r));
                            let selected =
                                reference.is_some_and(|r| r.decision.selected == Some(row.action));
                            let text = prediction.map_or(
                                if selected {
                                    "Selected".into()
                                } else {
                                    "—".into()
                                },
                                |x| format!("{x:.4}"),
                            );
                            if selected {
                                ui.colored_label(
                                    REFERENCE,
                                    egui::RichText::new(text).monospace().strong(),
                                );
                            } else {
                                ui.monospace(text);
                            }
                        }
                        if updates {
                            ui.monospace(row.updates.map_or("—".into(), |x| x.to_string()));
                        }
                        ui.end_row();
                    }
                });
        });
}
