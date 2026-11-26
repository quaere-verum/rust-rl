//! The Training workspace uses the shared history worker and learner selection.
use super::*;
use crate::learning::{
    TrainingClock, TrainingEvaluation, TrainingFrame, TrainingMetricKey, TrainingMetricPoint,
};
use egui_plot::{Legend, Line, LineStyle, Plot, PlotUi, Points, VLine};
use std::{path::PathBuf, time::Duration};

const MAX_SEED_PEERS: usize = 4;

fn secondary_metric_priority(name: &str) -> u8 {
    if name.ends_with(".standard_error") {
        4
    } else if name.contains("success_rate") || name.contains("landing_rate") || name.contains("gap")
    {
        0
    } else if name.contains("regret")
        || name.contains("calibration")
        || name.contains("success")
        || name.contains("fuel")
        || name.contains("duration")
        || name.contains("pnl")
    {
        1
    } else if name.starts_with("reference.") || name.starts_with("storage.") {
        3
    } else {
        2
    }
}

#[test]
fn meaningful_evaluation_metrics_precede_optimizer_diagnostics() {
    assert!(
        secondary_metric_priority("evaluation.landing_success_rate")
            < secondary_metric_priority("approx_kl")
    );
    assert!(
        secondary_metric_priority("evaluation.landing_success_rate")
            < secondary_metric_priority("evaluation.success_abs_x")
    );
    assert!(
        secondary_metric_priority("reference.benchmark_gap")
            < secondary_metric_priority("policy_loss")
    );
    assert!(
        secondary_metric_priority("reference.benchmark_gap.standard_error")
            > secondary_metric_priority("reference.benchmark_gap")
    );
}

pub(super) struct TrainingPane {
    clock: TrainingClock,
    metric: Option<TrainingMetricKey>,
    secondary: [Option<TrainingMetricKey>; 2],
    secondary_explicit: [bool; 2],
    episode_range: bool,
    cohort_paths: Vec<PathBuf>,
    #[cfg(feature = "screenshots")]
    capture_metric: Option<String>,
}

impl Default for TrainingPane {
    fn default() -> Self {
        Self {
            clock: TrainingClock::EnvironmentDecisions,
            metric: None,
            secondary: [None, None],
            secondary_explicit: [false, false],
            episode_range: false,
            cohort_paths: Vec::new(),
            #[cfg(feature = "screenshots")]
            capture_metric: None,
        }
    }
}

impl TrainingPane {
    pub(super) fn reset_for_run(&mut self) {
        self.metric = None;
        self.secondary = [None, None];
        self.secondary_explicit = [false, false];
        self.cohort_paths.clear();
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: Option<f64>,
    value: Option<f64>,
    revision: u64,
    se: Option<f64>,
    range: Option<[f64; 2]>,
}

fn shared_clock_extent(values: impl IntoIterator<Item = Option<f64>>) -> Option<[f64; 2]> {
    values
        .into_iter()
        .flatten()
        .filter(|x| x.is_finite())
        .fold(None, |range, x| {
            Some(range.map_or([x, x], |[min, max]: [f64; 2]| [min.min(x), max.max(x)]))
        })
}

fn link_training_clock(plot: Plot<'_>, group: egui::Id) -> Plot<'_> {
    plot.link_axis(group, egui::Vec2b::new(true, false))
}

fn success_grid_steps(base: f64) -> [f64; 3] {
    if !base.is_finite() || base <= 0. {
        return [0.25, 0.5, 1.];
    }
    if !(1e-100..=1e100).contains(&base) {
        return [base, base, base];
    }
    let scale = 10f64.powf(base.log10().floor());
    let step = [1., 2., 5., 10.]
        .into_iter()
        .map(|factor| factor * scale)
        .find(|candidate| *candidate >= base)
        .unwrap_or(10. * scale);
    [step, 2. * step, 5. * step]
}

#[test]
fn shared_training_extent_ignores_missing_and_nonfinite_clock_values() {
    assert_eq!(shared_clock_extent([None, Some(f64::NAN)]), None);
    assert_eq!(
        shared_clock_extent([Some(20.), Some(0.), None, Some(12.), Some(f64::INFINITY)]),
        Some([0., 20.])
    );
}

#[test]
fn success_axis_uses_compact_decimal_steps_at_different_zoom_levels() {
    let steps = success_grid_steps(0.13);
    assert!((steps[0] - 0.2).abs() < 1e-12);
    assert!((steps[1] - 0.4).abs() < 1e-12);
    let zoomed = success_grid_steps(0.003);
    assert!((zoomed[0] - 0.005).abs() < 1e-12);
    assert!((zoomed[1] - 0.01).abs() < 1e-12);
}

#[test]
fn training_plots_share_horizontal_zoom_without_linking_vertical_ranges() {
    let context = egui::Context::default();
    let group = egui::Id::unique("test-training-clock-link");
    let mut final_bounds = None;
    for frame in 0..3 {
        let mut bounds = None;
        let mut output = context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800., 600.),
                )),
                ..Default::default()
            },
            |ui| {
                let first = link_training_clock(
                    Plot::new("test-training-main")
                        .width(600.)
                        .height(200.)
                        .include_x(0.)
                        .include_x(100.),
                    group,
                )
                .show(ui, |plot| {
                    if frame == 1 {
                        plot.set_plot_bounds_x(10. ..=20.);
                        plot.set_plot_bounds_y(-5. ..=5.);
                    }
                });
                let second = link_training_clock(
                    Plot::new("test-training-secondary")
                        .width(300.)
                        .height(200.)
                        .include_x(0.)
                        .include_x(100.),
                    group,
                )
                .show(ui, |plot| {
                    if frame == 1 {
                        plot.set_plot_bounds_y(40. ..=60.);
                    }
                });
                bounds = Some((*first.transform.bounds(), *second.transform.bounds()));
            },
        );
        output.textures_delta.clear();
        final_bounds = bounds;
    }
    let (main, secondary) = final_bounds.unwrap();
    assert_eq!(main.range_x(), 10. ..=20.);
    assert_eq!(secondary.range_x(), 10. ..=20.);
    assert_eq!(main.range_y(), -5. ..=5.);
    assert_eq!(secondary.range_y(), 40. ..=60.);
}

fn evaluation_points(evaluations: &[TrainingEvaluation], clock: TrainingClock) -> Vec<Point> {
    evaluations
        .iter()
        .map(|p| Point {
            x: p.clocks.value(clock),
            value: p.summary.as_ref().map(|s| s.mean),
            revision: p.clocks.revision,
            se: p.summary.as_ref().and_then(|s| s.mean_standard_error),
            range: p.extrema,
        })
        .collect()
}

fn metric_points(metrics: &[TrainingMetricPoint], clock: TrainingClock) -> Vec<Point> {
    metrics
        .iter()
        .map(|p| Point {
            x: p.clocks.value(clock),
            value: p.value,
            revision: p.clocks.revision,
            se: None,
            range: None,
        })
        .collect()
}

fn consistent_checkpoint_x<'a>(
    main: &[Point],
    metrics: impl Iterator<Item = &'a TrainingMetricPoint>,
    revision: u64,
    clock: TrainingClock,
) -> Option<f64> {
    if clock == TrainingClock::Revision {
        return Some(revision as f64);
    }
    let main_positions = main.iter().filter_map(|point| {
        (point.revision == revision && point.value.is_some())
            .then_some(point.x)
            .flatten()
    });
    let metric_positions = metrics.filter_map(|point| {
        (point.clocks.revision == revision && point.value.is_some())
            .then(|| point.clocks.value(clock))
            .flatten()
    });
    let mut positions = main_positions
        .chain(metric_positions)
        .filter(|x| x.is_finite());
    let first = positions.next()?;
    positions.all(|position| position == first).then_some(first)
}

fn metric_reference_matches(key: &TrainingMetricKey, role: &str, external: bool) -> bool {
    key.reference
        .as_deref()
        .is_none_or(|source| !external && source == role)
}

fn connected(plot: &mut PlotUi<'_>, points: &[Point], name: &str, color: Color32, dashed: bool) {
    let mut segments = points
        .split(|p| p.x.is_none() || p.value.is_none())
        .filter(|segment| !segment.is_empty());
    let mut labeled = false;
    for segment in &mut segments {
        let values = segment
            .iter()
            .map(|p| [p.x.unwrap(), p.value.unwrap()])
            .collect::<Vec<_>>();
        let label = if labeled { "" } else { name };
        if values.len() <= 2 {
            plot.points(Points::new(label, values.clone()).color(color).radius(3.5));
        }
        if values.len() > 1 {
            plot.line(
                Line::new(label, values)
                    .color(color)
                    .style(if dashed {
                        LineStyle::dashed_dense()
                    } else {
                        LineStyle::Solid
                    })
                    .width(2.)
                    .allow_band(false),
            );
        }
        labeled = true;
    }
}

fn selected_point(plot: &mut PlotUi<'_>, point: Point, selected_x: Option<f64>, color: Color32) {
    let (Some(x), Some(y)) = (point.x, point.value) else {
        return;
    };
    if selected_x != Some(x) || !x.is_finite() || !y.is_finite() {
        return;
    }
    plot.points(
        Points::new("", vec![[x, y]])
            .id("selected-outline")
            .color(crate::theme::TEXT)
            .radius(6.)
            .allow_hover(false),
    );
    plot.points(
        Points::new("", vec![[x, y]])
            .id("selected-center")
            .color(color)
            .radius(4.)
            .allow_hover(false),
    );
}

fn uncertainty(plot: &mut PlotUi<'_>, points: &[Point], range: bool, color: Color32) {
    let mut bars = Vec::new();
    for p in points {
        let (Some(x), Some(y)) = (p.x, p.value) else {
            continue;
        };
        let bounds = if range {
            p.range
        } else {
            p.se.map(|se| [y - se, y + se])
        };
        if let Some([low, high]) = bounds {
            bars.push([[x, low], [x, high]]);
        }
    }
    for bounds in bars {
        plot.line(
            Line::new("", bounds.to_vec())
                .color(color.gamma_multiply(if range { 0.29 } else { 0.67 }))
                .allow_hover(false)
                .allow_band(false),
        );
    }
}

fn metric_label(key: &TrainingMetricKey) -> String {
    let label = match key.name.as_str() {
        "evaluation.episodes" => "Evaluated episodes",
        "evaluation.landing_success_rate" => "Landing success rate",
        "evaluation.flight_duration_seconds" => "Flight duration [s]",
        "evaluation.terminal_abs_x" => "Terminal |x| [m]",
        "evaluation.terminal_abs_vx" => "Terminal |vₓ| [m/s]",
        "evaluation.terminal_abs_vy" => "Terminal |vᵧ| [m/s]",
        "evaluation.success_fuel_used" => "Successful landing fuel used [kg]",
        "evaluation.success_duration_seconds" => "Successful landing duration [s]",
        "evaluation.success_abs_x" => "Successful landing |x| [m]",
        "evaluation.success_abs_vx" => "Successful landing |vₓ| [m/s]",
        "evaluation.success_abs_vy" => "Successful landing |vᵧ| [m/s]",
        "reference.benchmark_mean_return" => "Reference mean return",
        "reference.benchmark_gap" => "Signed benchmark gap",
        "reference.benchmark_gap.standard_error" => "Benchmark gap standard error",
        "reference.optimality_gap" => "Verified optimality gap",
        "reference.local_decision_regret" => "Local decision regret",
        "reference.model_coordinate_decision_regret" => "Model-coordinate decision regret",
        _ => key.name.as_str(),
    };
    let mut result = label.replace('_', " ");
    if let Some(population) = &key.population {
        result.push_str(" · ");
        result.push_str(population);
    }
    result
}

fn axis_label(title: &str) -> String {
    const LIMIT: usize = 24;
    if title.chars().count() <= LIMIT {
        return title.to_owned();
    }
    let mut compact: String = title.chars().take(LIMIT - 1).collect();
    compact.push('…');
    compact
}

#[cfg(test)]
#[test]
fn long_secondary_axis_labels_are_compact_and_keep_the_full_title() {
    let key = TrainingMetricKey {
        name: "reference.benchmark_gap.standard_error".into(),
        population: None,
        reference: None,
    };
    let title = metric_label(&key);
    assert_eq!(title, "Benchmark gap standard error");
    let axis = axis_label(&title);
    assert!(axis.chars().count() <= 24);
    assert!(axis.ends_with('…'));
    assert_eq!(axis_label("Critic loss"), "Critic loss");
    let success = TrainingMetricKey {
        name: "evaluation.landing_success_rate".into(),
        population: None,
        reference: None,
    };
    assert_eq!(metric_label(&success), "Landing success rate");
}

#[cfg(test)]
#[test]
fn secondary_metrics_do_not_confuse_an_external_provider_with_a_local_one() {
    let mut key = TrainingMetricKey {
        name: "reference.benchmark_gap".into(),
        population: None,
        reference: Some("reference:benchmark".into()),
    };
    assert!(metric_reference_matches(&key, "reference:benchmark", false));
    assert!(!metric_reference_matches(&key, "reference:benchmark", true));
    assert!(!metric_reference_matches(&key, "reference:other", false));
    key.reference = None;
    assert!(metric_reference_matches(&key, "reference:benchmark", true));
}

#[cfg(test)]
#[test]
fn secondary_cursor_uses_the_recorded_clock_when_main_evaluation_is_missing() {
    let points = [Point {
        x: Some(80.),
        value: Some(0.25),
        revision: 7,
        se: None,
        range: None,
    }];
    let metric = |position| TrainingMetricPoint {
        clocks: crate::learning::TrainingClocks {
            revision: 7,
            environment_decisions: Some(position),
            physical_ticks: None,
            wall_seconds: None,
        },
        sequence: 0,
        value: Some(0.25),
        reason: None,
    };
    let agreeing = [metric(80), metric(80)];
    let conflicting = [metric(80), metric(81)];
    assert_eq!(
        consistent_checkpoint_x(
            &points,
            agreeing.iter(),
            7,
            TrainingClock::EnvironmentDecisions
        ),
        Some(80.)
    );
    assert_eq!(
        consistent_checkpoint_x(
            &[],
            conflicting.iter(),
            7,
            TrainingClock::EnvironmentDecisions
        ),
        None
    );
    assert_eq!(
        consistent_checkpoint_x(
            &points,
            [metric(81)].iter(),
            7,
            TrainingClock::EnvironmentDecisions
        ),
        None
    );
    assert_eq!(
        consistent_checkpoint_x(&[], agreeing.iter(), 7, TrainingClock::EnvironmentDecisions),
        Some(80.)
    );
    assert_eq!(
        consistent_checkpoint_x(&points, conflicting.iter(), 7, TrainingClock::Revision),
        Some(7.)
    );
}

fn reference_metrics<'a>(
    frame: &'a TrainingFrame,
    role: &str,
    name: &str,
) -> Option<&'a [TrainingMetricPoint]> {
    frame.primary.metrics.iter().find_map(|(key, rows)| {
        (key.name == name && key.reference.as_deref() == Some(role)).then_some(rows.as_slice())
    })
}

fn clock_available(frame: &TrainingFrame, clock: TrainingClock) -> bool {
    frame
        .primary
        .evaluations
        .iter()
        .any(|p| p.clocks.value(clock).is_some())
        || frame
            .primary
            .metrics
            .values()
            .flatten()
            .any(|p| p.clocks.value(clock).is_some())
}

impl Laboratory {
    pub(super) fn training_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        use crate::timeline_ui::SnapshotPresentation;
        let worker = &self.learning_pane.worker;
        let frame = worker
            .training_frame()
            .filter(|frame| frame.primary.root == *run.root());
        let mut positions = Vec::new();
        if let Some(frame) = frame {
            positions.extend(
                frame
                    .primary
                    .evaluations
                    .iter()
                    .filter(|p| p.summary.is_some())
                    .map(|p| p.clocks.revision),
            );
        }
        if positions.is_empty() {
            positions.extend(run.revisions().iter().rev().take(512).copied());
        }
        positions.sort_unstable();
        positions.dedup();
        let exact = positions.contains(&self.selection.revision);
        let published = (frame.is_some() && exact).then_some(self.selection.revision);
        self.transport.hold_unavailable_snapshot(
            (!positions.is_empty() && !exact).then_some(self.selection.revision),
        );
        self.transport.show_evaluations(
            ui,
            run,
            &mut self.selection,
            SnapshotPresentation {
                positions: &positions,
                published,
                ready: frame.is_some(),
                checking: worker.pending(),
                error: None,
            },
        )?;
        Ok(())
    }

    pub(super) fn show_training(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<Option<u64>> {
        let worker = &mut self.learning_pane.worker;
        worker.poll();
        let reference_role = self.comparison.reference_role();
        let reference = self.comparison.reference_run().map(|r| (r, reference_role));
        let external = reference.is_some_and(|(r, _)| r.root() != run.root());
        worker.request_training(run, reference, &self.training_pane.cohort_paths)?;
        worker.poll();
        let worker_pending = worker.pending();
        let frame = worker.training_frame().cloned();
        if worker_pending {
            ui.ctx().request_repaint_after(Duration::from_millis(40));
        }
        let Some(frame) = frame.filter(|f| f.primary.root == *run.root()) else {
            if let Some(error) = worker.error() {
                ui.colored_label(WARM, error);
                if ui.button("Retry history").clicked() {
                    worker.retry();
                }
            } else {
                ui.weak("Loading training history…");
            }
            return Ok(None);
        };
        if let Some(after) = frame.peer_recheck_after() {
            ui.ctx()
                .request_repaint_after(after.max(Duration::from_millis(40)));
        }
        let pane = &mut self.training_pane;
        if !clock_available(&frame, pane.clock) {
            pane.clock = TrainingClock::Revision;
        }
        if pane.metric.as_ref().is_some_and(|k| {
            !frame.primary.metrics.contains_key(k)
                || !metric_reference_matches(k, reference_role, external)
        }) {
            pane.metric = None;
        }
        #[cfg(feature = "screenshots")]
        if let Some(name) = &pane.capture_metric {
            pane.metric = frame
                .primary
                .metrics
                .iter()
                .find(|(key, points)| {
                    key.name == *name
                        && metric_reference_matches(key, reference_role, external)
                        && points.iter().any(|point| point.value.is_some())
                })
                .map(|(key, _)| key.clone());
        }
        if pane.metric.is_none()
            && frame
                .primary
                .evaluations
                .iter()
                .all(|p| p.summary.is_none())
        {
            pane.metric = frame
                .primary
                .metrics
                .iter()
                .find(|(key, points)| {
                    key.reference.is_none() && points.iter().any(|p| p.value.is_some())
                })
                .map(|(key, _)| key.clone());
        }
        let mut available_secondary: Vec<_> = frame
            .primary
            .metrics
            .iter()
            .filter(|(key, points)| {
                Some(*key) != pane.metric.as_ref()
                    && key.name != "reference.benchmark_mean_return"
                    && metric_reference_matches(key, reference_role, external)
                    && points.iter().any(|point| {
                        point.value.is_some_and(f64::is_finite)
                            && point.clocks.value(pane.clock).is_some()
                    })
            })
            .collect();
        available_secondary.sort_by_key(|(key, _)| secondary_metric_priority(&key.name));
        for index in 0..2 {
            let valid = pane.secondary[index].as_ref().is_some_and(|selected| {
                available_secondary.iter().any(|(key, _)| *key == selected)
                    && (index == 0 || pane.secondary[0].as_ref() != Some(selected))
            });
            if !valid || !pane.secondary_explicit[index] {
                if !valid {
                    pane.secondary_explicit[index] = false;
                }
                pane.secondary[index] = available_secondary
                    .iter()
                    .map(|(key, _)| *key)
                    .find(|key| {
                        (index == 0 || pane.secondary[0].as_ref() != Some(*key))
                            && (index != 0
                                || !pane.secondary_explicit[1]
                                || pane.secondary[1].as_ref() != Some(*key))
                    })
                    .cloned();
            }
        }
        ui.horizontal_wrapped(|ui| {
            egui::ComboBox::from_id_salt("training-quantity")
                .selected_text(
                    pane.metric
                        .as_ref()
                        .map_or_else(|| "Evaluation return".into(), metric_label),
                )
                .width(186.)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut pane.metric, None, "Evaluation return");
                    for key in frame.primary.metrics.keys() {
                        if metric_reference_matches(key, reference_role, external) {
                            ui.selectable_value(
                                &mut pane.metric,
                                Some(key.clone()),
                                metric_label(key),
                            );
                        }
                    }
                });
            egui::ComboBox::from_id_salt("training-clock")
                .selected_text(pane.clock.label())
                .width(130.)
                .show_ui(ui, |ui| {
                    for clock in TrainingClock::ALL {
                        ui.add_enabled_ui(clock_available(&frame, clock), |ui| {
                            ui.selectable_value(&mut pane.clock, clock, clock.label());
                        });
                    }
                });
            if pane.metric.is_none() {
                ui.checkbox(&mut pane.episode_range, "Episode range");
                if self.catalogue.len() >= 2 {
                    crate::landscape::popover(ui, "Training seeds", |ui| {
                        ui.set_max_width(380.);
                        for entry in self.catalogue.iter().filter(|e| e.root != **run.root()) {
                            let mut enrolled = pane.cohort_paths.contains(&entry.root);
                            if ui
                                .add_enabled(
                                    enrolled || pane.cohort_paths.len() < MAX_SEED_PEERS,
                                    egui::Checkbox::new(&mut enrolled, &entry.manifest.run_id),
                                )
                                .changed()
                            {
                                if enrolled {
                                    pane.cohort_paths.push(entry.root.clone());
                                } else {
                                    pane.cohort_paths.retain(|p| p != &entry.root);
                                }
                            }
                        }
                    });
                }
            }
            if !available_secondary.is_empty() {
                crate::landscape::popover(ui, "Secondary plots", |ui| {
                    for index in 0..2 {
                        let selected = pane.secondary[index]
                            .as_ref()
                            .map_or_else(|| "Unavailable".into(), metric_label);
                        egui::ComboBox::from_id_salt(("training-secondary-choice", index))
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                for (key, _) in &available_secondary {
                                    if ui
                                        .selectable_label(
                                            pane.secondary[index].as_ref() == Some(*key),
                                            metric_label(key),
                                        )
                                        .clicked()
                                    {
                                        pane.secondary[index] = Some((*key).clone());
                                        pane.secondary_explicit[index] = true;
                                        if pane.secondary[1 - index].as_ref() == Some(*key) {
                                            pane.secondary[1 - index] = None;
                                            pane.secondary_explicit[1 - index] = false;
                                        }
                                    }
                                }
                            });
                    }
                });
            }
            if frame.primary.omitted_sources > 0 {
                ui.weak(format!(
                    "{} source records omitted by the history budget",
                    frame.primary.omitted_sources
                ));
            }
        });
        let clock = pane.clock;
        let quantity = pane
            .metric
            .as_ref()
            .map_or_else(|| "Discounted environmental return".into(), metric_label);
        let main = if let Some(metric) = &pane.metric {
            metric_points(&frame.primary.metrics[metric], clock)
        } else {
            evaluation_points(&frame.primary.evaluations, clock)
        };
        let reference_label = if external {
            frame
                .reference
                .as_ref()
                .map(|r| format!("{} · {}", r.run_id, r.algorithm))
        } else if crate::reference::provider_role(reference_role) {
            Some(
                if reference_metrics(&frame, reference_role, "reference.optimality_gap").is_some() {
                    "Verified optimum".into()
                } else {
                    "Benchmark".into()
                },
            )
        } else {
            Some(format!(
                "Checkpoint k {}",
                self.comparison.reference_revision()
            ))
        };
        let mut other = Vec::new();
        let mut checkpoint_reference = None;
        let mut cohort = Vec::new();
        if pane.metric.is_none() {
            if external {
                if let Some(reference) = &frame.reference {
                    other = evaluation_points(&reference.evaluations, clock);
                }
            } else if crate::reference::provider_role(reference_role) {
                if let Some(metrics) =
                    reference_metrics(&frame, reference_role, "reference.benchmark_mean_return")
                {
                    other = metric_points(metrics, clock);
                } else if let Some(reference) = &frame.reference {
                    other = evaluation_points(&reference.evaluations, clock);
                }
            } else {
                let k = self.comparison.reference_revision();
                let reference_evaluations = if frame.primary.role == reference_role {
                    Some(frame.primary.evaluations.as_slice())
                } else {
                    frame
                        .reference
                        .as_ref()
                        .filter(|r| r.role == reference_role)
                        .map(|r| r.evaluations.as_slice())
                };
                checkpoint_reference = reference_evaluations
                    .and_then(|points| points.iter().rfind(|p| p.clocks.revision == k))
                    .and_then(|p| Some([p.clocks.value(clock)?, p.summary.as_ref()?.mean]));
            }
            if let Some(aggregate) = &frame.cohort {
                for point in aggregate.points(clock) {
                    if point.starts_segment && !cohort.is_empty() {
                        cohort.push(Point {
                            x: None,
                            value: None,
                            revision: 0,
                            se: None,
                            range: None,
                        });
                    }
                    cohort.push(Point {
                        x: Some(point.x),
                        value: Some(point.mean),
                        revision: 0,
                        se: point.standard_error,
                        range: None,
                    });
                }
            }
        }
        let selected_revision = self.selection.revision;
        let actual = main
            .iter()
            .rfind(|p| p.revision == selected_revision && p.value.is_some())
            .copied();
        let mut secondary: Vec<_> =
            pane.secondary
                .iter()
                .filter_map(|choice| {
                    let key = choice.as_ref()?;
                    available_secondary.iter().copied().find(|(candidate, _)| {
                        *candidate == key && Some(key) != pane.metric.as_ref()
                    })
                })
                .collect();
        if ui.available_height() < 400. {
            secondary.clear();
        }
        let selected_x = consistent_checkpoint_x(
            &main,
            secondary.iter().flat_map(|(_, rows)| rows.iter()),
            selected_revision,
            clock,
        );
        let clock_extent = shared_clock_extent(
            main.iter()
                .chain(&other)
                .chain(&cohort)
                .map(|point| point.x)
                .chain(
                    secondary
                        .iter()
                        .flat_map(|(_, rows)| rows.iter().map(|point| point.clocks.value(clock))),
                )
                .chain(std::iter::once(checkpoint_reference.map(|point| point[0]))),
        );
        let clock_axis_group =
            ui.make_persistent_id(("training-shared-clock", run.root(), clock.label()));
        let mut clicked = None;
        if main.iter().any(|p| p.value.is_some() && p.x.is_some()) {
            let height = if secondary.is_empty() {
                (ui.available_height() - 64.).max(164.)
            } else {
                (ui.available_height() * 0.55 - 16.).max(164.)
            };
            crate::theme::panel_frame().inner_margin(8).show(ui, |ui| {
                ui.label(
                    egui::RichText::new(if pane.metric.is_none() {
                        "Evaluation performance".to_owned()
                    } else {
                        quantity.clone()
                    })
                    .size(17.)
                    .strong(),
                );
                let mut plot = Plot::new(("training-history", pane.metric.as_ref(), clock.label()))
                    .height(height)
                    .width(ui.available_width().max(180.))
                    .show_background(false)
                    .grid_color(crate::theme::GRID)
                    .allow_drag(true)
                    .legend(Legend::default().position(egui_plot::Corner::RightBottom))
                    .x_axis_formatter(crate::landscape::axis_tick)
                    .y_axis_formatter(crate::landscape::axis_tick)
                    .x_axis_label(clock.label())
                    .y_axis_label(quantity);
                plot = link_training_clock(plot, clock_axis_group);
                if let Some([min, max]) = clock_extent {
                    plot = plot.include_x(min).include_x(max);
                }
                let response = plot.show(ui, |plot| {
                    if pane.episode_range {
                        uncertainty(plot, &main, true, crate::theme::EMPIRICAL);
                    }
                    uncertainty(plot, &main, false, crate::theme::CURRENT);
                    if !other.is_empty() {
                        connected(
                            plot,
                            &other,
                            reference_label.as_deref().unwrap_or("Reference"),
                            crate::theme::REFERENCE,
                            true,
                        );
                    }
                    if let Some(point) = checkpoint_reference {
                        plot.points(
                            Points::new(
                                reference_label
                                    .as_deref()
                                    .unwrap_or("Historical checkpoint"),
                                vec![point],
                            )
                            .shape(egui_plot::MarkerShape::Diamond)
                            .color(crate::theme::REFERENCE)
                            .radius(5.),
                        );
                    }
                    if !cohort.is_empty() {
                        uncertainty(plot, &cohort, false, crate::theme::EMPIRICAL);
                        let n = frame
                            .cohort
                            .as_ref()
                            .unwrap()
                            .points(clock)
                            .first()
                            .map_or(0, |p| p.runs);
                        connected(
                            plot,
                            &cohort,
                            &format!("Seed mean ± SE · n={n}"),
                            crate::theme::EMPIRICAL,
                            true,
                        );
                    }
                    connected(plot, &main, "Current run", crate::theme::CURRENT, false);
                    if let Some(x) = selected_x {
                        plot.vline(
                            VLine::new("Selected checkpoint", x)
                                .color(crate::theme::SELECTED.gamma_multiply(0.7))
                                .style(LineStyle::dashed_dense()),
                        );
                    }
                    if let Some(point) = actual {
                        selected_point(plot, point, selected_x, crate::theme::CURRENT);
                    }
                    plot.pointer_coordinate()
                });
                if response.response.clicked()
                    && let Some(position) = response.inner
                {
                    clicked = main
                        .iter()
                        .filter_map(|p| {
                            p.x.zip(p.value)
                                .map(|(x, _)| (p.revision, (x - position.x).abs()))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(revision, _)| revision);
                }
            });
        } else {
            ui.weak("No complete measurements for this quantity on the selected clock.");
        }

        if !secondary.is_empty() {
            let height = (ui.available_height() - 86.).max(120.);
            ui.columns(secondary.len(), |columns| {
                for (column, (key, rows)) in columns.iter_mut().zip(&secondary) {
                    crate::theme::panel_frame()
                        .inner_margin(8)
                        .show(column, |column| {
                            let points = metric_points(rows, clock);
                            let title = metric_label(key);
                            column
                                .label(egui::RichText::new(&title).size(16.).strong())
                                .on_hover_text(&key.name);
                            let mut plot = Plot::new(("training-secondary", *key, clock.label()))
                                .height(height)
                                .width((column.available_width() - 12.).max(120.))
                                .show_background(false)
                                .grid_color(crate::theme::GRID)
                                .x_axis_formatter(crate::landscape::axis_tick)
                                .y_axis_formatter(crate::landscape::axis_tick)
                                .x_axis_label(clock.label())
                                .y_axis_label(axis_label(&title));
                            plot = link_training_clock(plot, clock_axis_group);
                            if let Some([min, max]) = clock_extent {
                                plot = plot.include_x(min).include_x(max);
                            }
                            if key.name == "evaluation.landing_success_rate" {
                                plot = plot
                                    .include_y(0.)
                                    .include_y(1.)
                                    .grid_spacing(18.0..=60.0)
                                    .y_grid_spacer(egui_plot::uniform_grid_spacer(|input| {
                                        success_grid_steps(input.base_step_size)
                                    }))
                                    .custom_y_axes(vec![
                                        egui_plot::AxisHints::new_y()
                                            .label(axis_label(&title))
                                            .label_spacing(18.0..=35.0)
                                            .formatter(crate::landscape::axis_tick),
                                    ]);
                            }
                            let response = plot.show(column, |plot| {
                                let color = if key.name.contains("gap") {
                                    crate::theme::EMPIRICAL
                                } else if key.name.contains("calibration") {
                                    crate::theme::REFERENCE
                                } else {
                                    crate::theme::CURRENT
                                };
                                connected(plot, &points, &title, color, false);
                                if let Some(x) = selected_x {
                                    plot.vline(
                                        VLine::new("Selected checkpoint", x)
                                            .color(crate::theme::SELECTED.gamma_multiply(0.7))
                                            .style(LineStyle::dashed_dense()),
                                    );
                                }
                                if let Some(point) = points.iter().rfind(|point| {
                                    point.revision == selected_revision && point.value.is_some()
                                }) {
                                    selected_point(plot, *point, selected_x, color);
                                }
                                plot.pointer_coordinate()
                            });
                            if response.response.clicked()
                                && let Some(position) = response.inner
                            {
                                clicked = points
                                    .iter()
                                    .filter_map(|point| {
                                        point
                                            .x
                                            .zip(point.value)
                                            .map(|(x, _)| (point.revision, (x - position.x).abs()))
                                    })
                                    .min_by(|a, b| a.1.total_cmp(&b.1))
                                    .map(|(revision, _)| revision);
                            }
                        });
                }
            });
        }

        ui.horizontal_wrapped(|ui| {
            if let Some(p) = actual
                && let Some(value) = p.value
            {
                ui.strong(format!("k {}   {:.4}", p.revision, value));
                if let Some(se) = p.se {
                    ui.weak(format!("± {se:.4} episode SE"));
                }
            } else {
                ui.weak(format!("No measurement · k {selected_revision}"));
            }
            if external && clock == TrainingClock::Revision {
                ui.weak("Revision clock")
                    .on_hover_text("Learner updates are specific to each algorithm.");
            }
            if pane.metric.is_none() {
                if let Some(reason) = &frame.reference_unavailable {
                    ui.weak("Reference unavailable").on_hover_text(reason);
                } else if !external
                    && !crate::reference::provider_role(reference_role)
                    && checkpoint_reference.is_none()
                {
                    ui.weak("Reference unavailable").on_hover_text(format!(
                        "No complete evaluation for {reference_role} at reference k {}",
                        self.comparison.reference_revision()
                    ));
                }
            }
            if let Some(reason) = &frame.cohort_unavailable {
                ui.weak("Seed comparison unavailable").on_hover_text(reason);
            }
        });
        Ok(clicked)
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn configure_training_capture(&mut self) -> Result<()> {
        self.training_pane.capture_metric = std::env::var("RL_VIEWER_TRAINING_METRIC").ok();
        if let Some(paths) = std::env::var_os("RL_VIEWER_TRAINING_COHORT") {
            let peers = std::env::split_paths(&paths).collect::<Vec<_>>();
            if peers.len() > MAX_SEED_PEERS {
                return Err(rl::Error::Config("Too many Training cohort paths".into()));
            }
            self.training_pane.cohort_paths = peers;
        }
        let reference = std::env::var_os("RL_VIEWER_TRAINING_REFERENCE_RUN").map(PathBuf::from);
        let role = std::env::var("RL_VIEWER_TRAINING_REFERENCE_ROLE").ok();
        if reference.is_some() || role.is_some() {
            self.comparison
                .configure_policy_reference(reference, None, false, role.as_deref())?;
        }
        Ok(())
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn training_capture_ready(&self) -> bool {
        !self.learning_pane.worker.pending()
            && self.learning_pane.worker.training_frame().is_some()
            && self
                .training_pane
                .capture_metric
                .as_ref()
                .is_none_or(|name| {
                    self.training_pane
                        .metric
                        .as_ref()
                        .is_some_and(|key| key.name == *name)
                })
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn training_capture_error(&self) -> Option<String> {
        let worker = &self.learning_pane.worker;
        worker.error().map(str::to_owned).or_else(|| {
            if !worker.pending()
                && worker.training_frame().is_some()
                && !self.training_capture_ready()
            {
                self.training_pane
                    .capture_metric
                    .as_ref()
                    .map(|name| format!("Recorded training metric unavailable: {name}"))
            } else {
                None
            }
        })
    }

    #[cfg(feature = "screenshots")]
    pub(super) fn training_capture_evidence(&self) -> serde_json::Value {
        let frame = self.learning_pane.worker.training_frame();
        serde_json::json!({
            "clock": self.training_pane.clock.label(),
            "evaluations": frame.map(|f| f.primary.evaluations.len()),
            "metrics": frame.map(|f| f.primary.metrics.len()),
            "selected_metric": self.training_pane.metric.as_ref().map(|metric| &metric.name),
            "reference": frame.and_then(|f| f.reference.as_ref().map(|r| &r.run_id)),
            "cohort_points": frame.and_then(|f| f.cohort.as_ref()).map(|c| c.points(self.training_pane.clock).len()),
        })
    }
}
