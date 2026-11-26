//! Optional secondary learning history over the global requested learner revision.
use crate::{
    data::{RunIndex, Selection},
    learning::{ChangeKind, LearningFrame, LearningMetrics, LearningPoint, LearningWorker},
};
use eframe::egui;
use rl::Result;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) struct LearningPane {
    pub(super) worker: LearningWorker,
    expanded: bool,
    was_expanded: bool,
    last_check: Instant,
}
impl LearningPane {
    pub(super) fn new(ctx: egui::Context) -> Result<Self> {
        Ok(Self {
            worker: LearningWorker::new(move || ctx.request_repaint())?,
            expanded: false,
            was_expanded: false,
            last_check: Instant::now(),
        })
    }
    pub(super) fn clear(&mut self) {
        self.worker.clear();
        self.last_check = Instant::now();
    }
    pub(super) fn retry(&mut self) {
        self.worker.retry();
        self.last_check = Instant::now();
    }
    pub(super) fn set_expanded(&mut self, value: bool) {
        self.expanded = value;
    }
    pub(super) fn expanded(&self) -> bool {
        self.expanded
    }
    pub(super) fn frame(&self) -> Option<&Arc<LearningFrame>> {
        self.worker.frame()
    }
    pub(super) fn metrics(&self) -> LearningMetrics {
        self.worker.metrics()
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn capture_ready(&self) -> bool {
        !self.expanded
            || (!self.worker.pending()
                && self.worker.frame().is_some()
                && self.worker.error().is_none())
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn capture_error(&self) -> Option<String> {
        self.expanded
            .then(|| self.worker.error().map(str::to_owned))
            .flatten()
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn capture_evidence(&self) -> serde_json::Value {
        let metrics = self.worker.metrics();
        let points = |points: &[LearningPoint]| {
            points.iter().map(|p|serde_json::json!({"actual_revision":p.revision,"sequence":p.sequence,"value":p.value.filter(|v|v.is_finite()),"infinite":p.value.is_some_and(|v|v.is_infinite()),"n":p.count,"mean_standard_error":p.standard_error,"conditions":p.conditions,"critic_name":p.critic_name,"continuation":p.continuation,"condition_counts":p.condition_counts,"definition":p.label,"unavailable":p.reason})).collect::<Vec<_>>()
        };
        let frame = self.worker.frame().map(|f| {
            let policy = f.anchor.as_ref().map(|a|serde_json::json!({"actual_revision":a.revision,"sequence":a.sequence,"probe_sequence":a.probe_sequence,"rows":a.rows,"population":a.population,"kind":format!("{:?}",a.kind),"normalization":a.normalization}));
            let critic = f.critic_anchor.as_ref().map(|a|serde_json::json!({"actual_revision":a.revision,"sequence":a.sequence,"probe_sequence":a.probe_sequence,"critic_name":a.critic_name,"kind":format!("{:?}",a.kind),"score_kind":format!("{:?}",a.score_kind),"action_condition":format!("{:?}",a.action_condition),"stochastic":a.stochastic,"row_ids":a.row_ids,"observation_methods":a.observation_methods,"population":a.population,"initial_continuation":a.initial_continuation}));
            serde_json::json!({"root":f.root.as_ref(),"role":f.role,"omitted_sources":f.omitted_sources,"summary_bytes":f.retained_bytes,"anchor":policy,"critic_anchor":critic,"evaluation":points(&f.evaluations),"policy_change":points(&f.policy_changes),"independent_critic_error":points(&f.critic_errors)})
        });
        serde_json::json!({"expanded":self.expanded,"pending":self.worker.pending(),"submitted":metrics.submitted,"completed":metrics.completed,"reads":metrics.reads,"owned_bytes":metrics.owned_bytes,"error":self.worker.error(),"frame":frame})
    }
    pub(super) fn show(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
        selection: &Selection,
    ) -> Result<Option<u64>> {
        self.worker.poll();
        if !self.expanded {
            if self.was_expanded {
                self.clear();
            }
            self.was_expanded = false;
            return Ok(None);
        }
        self.was_expanded = true;
        self.worker.request(run, &selection.model_role)?;
        self.worker.poll();
        if self.last_check.elapsed() >= Duration::from_secs(1) && self.worker.recheck() {
            self.last_check = Instant::now();
        }
        ui.ctx().request_repaint_after(Duration::from_secs(1));
        if let Some(error) = self.worker.error().map(str::to_owned) {
            ui.small(egui::RichText::new("Learning unavailable").color(crate::theme::WARNING));
            details_popover(ui, |ui| {
                ui.small(error);
                if ui.small_button("Retry history").clicked() {
                    self.retry();
                }
            });
            return Ok(None);
        }
        let Some(frame) = self
            .worker
            .frame()
            .filter(|f| f.root == *run.root() && f.role == selection.model_role)
            .cloned()
        else {
            ui.weak("Checking recorded evaluation, fixed-policy and independent critic evidence…");
            return Ok(None);
        };
        let revision_bounds = frame
            .evaluations
            .iter()
            .chain(&frame.policy_changes)
            .chain(&frame.critic_errors)
            .fold(None, |bounds, p| {
                Some(
                    bounds.map_or([p.revision, p.revision], |[lo, hi]: [u64; 2]| {
                        [lo.min(p.revision), hi.max(p.revision)]
                    }),
                )
            });
        let mut selected = None;
        ui.columns(3, |columns| {
            for (column, (name, points, color)) in columns.iter_mut().zip([
                ("Evaluation return", &frame.evaluations, crate::theme::EMPIRICAL),
                ("Policy change", &frame.policy_changes, crate::theme::REFERENCE),
                ("Independent critic score", &frame.critic_errors, crate::theme::CURRENT),
            ]) {
                column.strong(name).on_hover_text(match name {
                    "Evaluation return"=>"Complete initial-scenario return means at actual measurement revisions. Bars show ±1 Monte Carlo standard error (sample divisor n−1), available for n≥2 with declared independent futures. Each recorded batch remains separate.",
                    "Policy change"=>"All admitted rows of one frozen exact population; reference is the earliest compatible policy admitted when this history opened. Every action component participates. KL direction is reference || current.",
                    _=>"Scalar absolute error against certified conditional means, or mean CRPS of ordered quantile atoms. Measured conditions receive equal weight; point details identify sample counts and population coverage. Unsupported conditioning stays unavailable.",
                });
                if let Some(revision) = chart(column, points, revision_bounds, selection.revision, color) {
                    selected = Some(revision);
                }
                let point = selected_point(points, selection.revision);
                column.label(egui::RichText::new(point.map_or_else(|| "No measurement at requested k".into(), point_summary)).small())
                    .on_hover_text(point.map_or_else(|| "No measurement at requested revision".into(), point_text));
            }
        });
        details_popover(ui, |ui| {
            ui.small("Actual learner k · independent y scales · click a measurement to navigate the shared learner revision.");
            if self.worker.pending() {
                ui.small("Updating…");
            }
            if ui.small_button("Retry history").clicked() {
                self.retry();
            }
            if let Some(a) = &frame.anchor {
                let definition = match a.kind {
                    ChangeKind::ReferenceKl => "mean KL(reference || current)",
                    ChangeKind::NormalizedRms => "all-component normalized RMS",
                    ChangeKind::DiscreteDisagreement => "deterministic action disagreement",
                };
                ui.small(format!(
                    "{definition} · frozen earliest reference k={} / source {} · {} rows",
                    a.revision, a.sequence, a.rows
                ));
                ui.small(format!("Original probe source {}: {}. Box controls use fixed action spans; unbounded controls use reference RMS with floor 1e-6 action units. Scales: {:?}",a.probe_sequence,a.population,a.normalization));
            } else {
                ui.weak(
                "Policy change unavailable: no admitted complete fixed/imported policy population.",
            );
            }
            if let Some(a) = &frame.critic_anchor {
                ui.small(format!("Critic {} · frozen measured cohort k={} / source {} · {} conditions, equal weights",a.critic_name,a.revision,a.sequence,a.row_ids.len()));
                ui.small(format!("{:?}; {:?}; {}. Row IDs {:?}; original probe source {}. Each point targets this named critic's explicitly declared current-revision continuation (actual ID below). Complete observations, masks and first actions stay fixed. Finite Monte Carlo target error is noisy, not an unbiased true-error estimate.",a.kind,a.score_kind,a.population,a.row_ids,a.probe_sequence));
            } else {
                ui.weak("Independent critic unavailable: no admitted compatible measured cohort.");
            }
            if frame.omitted_sources > 0 {
                ui.colored_label(crate::theme::WARNING,format!("{} older measurement sources omitted by the bounded history window; reference remains pinned.",frame.omitted_sources));
            }
            for (name, points) in [
                ("Evaluation return", &frame.evaluations),
                ("Policy change", &frame.policy_changes),
                ("Independent critic score", &frame.critic_errors),
            ] {
                ui.strong(name);
                if let Some(point) = selected_point(points, selection.revision) {
                    ui.small(point_text(point));
                } else {
                    ui.weak("No measurement at requested revision");
                }
            }
            ui.small("Markers are recorded measurements; × marks unavailable or infinite values. No interpolation joins missing or incompatible measurements.");
        });
        Ok(selected)
    }
}

fn details_popover(ui: &mut egui::Ui, body: impl FnOnce(&mut egui::Ui)) {
    let viewport = ui.ctx().content_rect();
    let height = (viewport.height() - 80.).clamp(80., 300.);
    crate::landscape::popover(ui, "Learning details", |ui| {
        ui.set_width((viewport.width() - 32.).clamp(120., 640.));
        egui::ScrollArea::vertical()
            .id_salt("learning-details-content")
            .max_height(height)
            .show(ui, body);
    });
}

fn selected_point(points: &[LearningPoint], revision: u64) -> Option<&LearningPoint> {
    points
        .iter()
        .find(|p| p.revision == revision && p.value.is_some())
        .or_else(|| points.iter().find(|p| p.revision == revision))
}

fn point_summary(point: &LearningPoint) -> String {
    let value = point.value.map_or_else(
        || "Unavailable".into(),
        |v| {
            if v.is_infinite() {
                "∞".into()
            } else {
                format!("{v:.6}")
            }
        },
    );
    let se = point
        .standard_error
        .map_or_else(|| "SE —".into(), |v| format!("SE {v:.6}"));
    format!("k={} · {value} · n={} · {se}", point.revision, point.count)
}

fn point_text(p: &LearningPoint) -> String {
    let value = p.value.map_or_else(
        || "Unavailable".into(),
        |v| {
            if v.is_infinite() {
                "∞ (support mismatch)".into()
            } else {
                format!("{v:.6}")
            }
        },
    );
    let uncertainty = p
        .standard_error
        .map_or_else(|| "MC SE unavailable".into(), |se| format!("MC SE {se:.6}"));
    let subject = p.critic_name.as_ref().map_or(String::new(), |name| {
        format!(
            "\nCritic {name}; declared continuation {}; futures per frozen condition {:?}",
            p.continuation.as_deref().unwrap_or("unavailable"),
            p.condition_counts,
        )
    });
    format!(
        "Actual k={} · source {:?}\n{value} · n={} · {uncertainty} · measured conditions {}\n{}{subject}{}",
        p.revision,
        p.sequence,
        p.count,
        p.conditions,
        p.label,
        p.reason
            .as_ref()
            .map_or(String::new(), |r| format!("\n{r}"))
    )
}
fn y_tick(value: f64, range: [f64; 2]) -> String {
    // Half-span is the actual separation between the three fixed history ticks.
    // Derive precision from that span, independently of requested learner time.
    let step = (range[1] / 2. - range[0] / 2.).abs();
    if value != 0. && !(1e-4..1e6).contains(&value.abs()) && step > 0. && step.is_finite() {
        // The shared formatter's shortest scientific representation can retain
        // every binary digit; compact plots need only the step's precision.
        let decimals = (value.abs().log10().floor() - step.log10().floor()).clamp(0., 16.) as usize;
        format!("{value:.decimals$e}")
    } else {
        crate::landscape::axis_tick(
            egui_plot::GridMark {
                value,
                step_size: step,
            },
            &(range[0]..=range[1]),
        )
    }
}
fn chart(
    ui: &mut egui::Ui,
    points: &[LearningPoint],
    revision_bounds: Option<[u64; 2]>,
    requested: u64,
    color: egui::Color32,
) -> Option<u64> {
    let (outer, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(60.), 112.),
        egui::Sense::click(),
    );
    let plot = egui::Rect::from_min_max(
        outer.min + egui::vec2(58., 12.),
        outer.max - egui::vec2(8., 23.),
    );
    let painter = ui.painter_at(outer);
    painter.rect_filled(plot, 2., crate::theme::INSET);
    let Some([first, last]) = revision_bounds else {
        painter.text(
            plot.center(),
            egui::Align2::CENTER_CENTER,
            "No measurements",
            egui::FontId::proportional(11.),
            crate::theme::MUTED,
        );
        return None;
    };
    if points.is_empty() {
        painter.text(
            plot.center(),
            egui::Align2::CENTER_CENTER,
            "Unavailable",
            egui::FontId::proportional(11.),
            crate::theme::MUTED,
        );
        return None;
    }
    let mut range = points
        .iter()
        .filter_map(|p| {
            p.value
                .filter(|v| v.is_finite())
                .map(|v| (v, p.standard_error))
        })
        .fold(None, |range, (v, se)| {
            let lo = se.map(|s| v - s).filter(|x| x.is_finite()).unwrap_or(v);
            let hi = se.map(|s| v + s).filter(|x| x.is_finite()).unwrap_or(v);
            Some(range.map_or([lo, hi], |[l, h]: [f64; 2]| [l.min(lo), h.max(hi)]))
        })
        .unwrap_or([0., 1.]);
    if range[0] == range[1] {
        let padding = range[0].abs().max(1.) * 0.1;
        let lo = range[0] - padding;
        let hi = range[1] + padding;
        if lo.is_finite() {
            range[0] = lo;
        }
        if hi.is_finite() {
            range[1] = hi;
        }
    }
    let map_x = |revision: u64| {
        egui::lerp(
            plot.left()..=plot.right(),
            (revision.saturating_sub(first) as f64 / (last - first).max(1) as f64) as f32,
        )
    };
    let map_y = |value: f64| {
        egui::lerp(
            plot.bottom()..=plot.top(),
            ((value / 2. - range[0] / 2.) / (range[1] / 2. - range[0] / 2.)) as f32,
        )
    };
    for i in 0..=2 {
        let f = i as f32 / 2.;
        let y = egui::lerp(plot.bottom()..=plot.top(), f);
        painter.line_segment(
            [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
            egui::Stroke::new(0.5, crate::theme::GRID),
        );
        painter.text(
            egui::pos2(plot.left() - 4., y),
            egui::Align2::RIGHT_CENTER,
            y_tick(range[0] * (1. - f as f64) + range[1] * f as f64, range),
            egui::FontId::proportional(10.),
            crate::theme::MUTED,
        );
    }
    for (revision, align) in [
        (first, egui::Align2::LEFT_TOP),
        (last, egui::Align2::RIGHT_TOP),
    ] {
        painter.text(
            egui::pos2(map_x(revision), plot.bottom() + 5.),
            align,
            revision.to_string(),
            egui::FontId::proportional(10.),
            crate::theme::MUTED,
        );
    }
    let geometry = painter.with_clip_rect(plot);
    if (first..=last).contains(&requested) {
        let x = map_x(requested);
        geometry.line_segment(
            [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
            egui::Stroke::new(1., crate::theme::TEXT),
        );
    }
    let position = |point: &LearningPoint| {
        egui::pos2(
            map_x(point.revision),
            point
                .value
                .filter(|v| v.is_finite())
                .map_or(plot.bottom() - 3., map_y),
        )
    };
    for point in points {
        let pos = position(point);
        if let Some(value) = point.value.filter(|v| v.is_finite()) {
            if let Some(se) = point.standard_error {
                let lo = value - se;
                let hi = value + se;
                if lo.is_finite() && hi.is_finite() {
                    geometry.line_segment(
                        [egui::pos2(pos.x, map_y(lo)), egui::pos2(pos.x, map_y(hi))],
                        egui::Stroke::new(1., color),
                    );
                }
            }
            geometry.circle_filled(pos, 3., color);
            if point.revision == requested {
                geometry.circle_stroke(pos, 4.5, egui::Stroke::new(1., crate::theme::SELECTED));
            }
        } else {
            let pos = if point.value.is_some() {
                egui::pos2(pos.x, plot.top() + 5.)
            } else {
                pos
            };
            for sign in [-1., 1.] {
                geometry.line_segment(
                    [
                        pos + egui::vec2(-3., sign * 3.),
                        pos + egui::vec2(3., -sign * 3.),
                    ],
                    egui::Stroke::new(1., crate::theme::WARNING),
                );
            }
        }
    }
    let hovered = response.hover_pos().and_then(|pointer| {
        points.iter().min_by(|a, b| {
            position(a)
                .distance_sq(pointer)
                .total_cmp(&position(b).distance_sq(pointer))
        })
    });
    let selected = if response.clicked() {
        hovered.map(|p| p.revision)
    } else {
        None
    };
    if let Some(point) = hovered {
        response.on_hover_text(point_text(point));
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::core::action::ActionSpace;
    use rl::diagnostics::{schema::*, storage::*};
    use std::{
        collections::BTreeMap,
        time::{Duration, Instant},
    };
    fn y_labels(values: [f64; 3], requested: u64) -> Vec<String> {
        let points: Vec<_> = values
            .into_iter()
            .enumerate()
            .map(|(i, value)| LearningPoint {
                revision: (i * 8) as u64,
                sequence: Some(i as u64),
                value: Some(value),
                count: 64,
                standard_error: None,
                conditions: 3,
                critic_name: None,
                continuation: None,
                condition_counts: Vec::new(),
                label: String::new(),
                reason: None,
            })
            .collect();
        let ctx = egui::Context::default();
        let mut chart_left = 0.;
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(500., 200.),
                )),
                ..Default::default()
            },
            |ui| {
                chart_left = ui.cursor().left();
                chart(ui, &points, Some([0, 16]), requested, crate::theme::CURRENT);
            },
        );
        fn labels(shape: &egui::Shape, out: &mut Vec<String>, chart_left: f32) {
            match shape {
                egui::Shape::Text(t) => {
                    if out.len() < 3 {
                        assert!(
                            t.pos.x >= chart_left,
                            "Y tick {:?} clips at {} before chart left {chart_left}",
                            t.galley.job.text,
                            t.pos.x
                        );
                    }
                    out.push(t.galley.job.text.clone());
                }
                egui::Shape::Vec(v) => {
                    for shape in v {
                        labels(shape, out, chart_left);
                    }
                }
                _ => {}
            }
        }
        let mut text = Vec::new();
        output.textures_delta.clear();
        for shape in &output.shapes {
            labels(&shape.shape, &mut text, chart_left);
        }
        assert!(
            text.len() >= 5,
            "Missing actual chart axis labels: {text:?}"
        );
        text.truncate(3); // The painter emits the three Y ticks before learner-time ticks.
        text
    }
    fn assert_distinct_ticks(values: [f64; 3]) {
        let labels = y_labels(values, 0);
        assert!(
            labels.windows(2).all(|pair| pair[0] != pair[1]),
            "Collapsed adjacent ticks: {labels:?}"
        );
        let step = (values[2] - values[0]) / 2.;
        for (label, expected) in labels.iter().zip(values) {
            let actual: f64 = label.parse().unwrap();
            assert!(
                (actual - expected).abs() < step / 4.,
                "Tick {label} misrepresents {expected}"
            );
            assert!(
                label.len() <= 11,
                "Axis label is too long for the compact chart: {label}"
            );
        }
        assert_eq!(
            labels,
            y_labels(values, 16),
            "Scrubbing must not change axis precision"
        );
    }
    #[test]
    fn y_ticks_distinguish_narrow_critic_scores_and_stay_fixed_while_scrubbing() {
        assert_distinct_ticks([1.80, 1.82, 1.84]);
    }
    #[test]
    fn y_ticks_distinguish_tiny_policy_kl_over_a_narrow_scaled_span() {
        assert_distinct_ticks([1.80e-6, 1.82e-6, 1.84e-6]);
    }
    #[test]
    fn y_ticks_fit_the_compact_plot_gutter() {
        assert_distinct_ticks([0., 0.000128, 0.000256]);
    }
    fn fixture() -> (tempfile::TempDir, RunIndex) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("run");
        let m = RunManifest::new(
            "ui-learning",
            "test",
            "test",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let mut w = Recorder::create(&p, m, StorageConfig::default()).unwrap();
        w.append(&Chunk {
            context: RecordContext::new("ui-learning", 3),
            record: Record::Empirical(EmpiricalRecord {
                returns: "r".into(),
                seeds: "s".into(),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "learner".into(),
                    stochastic_policy: false,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "independent explicit futures".into(),
                    truncation_handling: "true terminal".into(),
                    complete_returns: true,
                },
                probe_row: None,
                initial_action: None,
                conditioning_observation: None,
                conditioning_mask: None,
                simulator_state: None,
            }),
            arrays: BTreeMap::from([
                ("r".into(), Array::f64(vec![2], vec![1., 3.]).unwrap()),
                ("s".into(), Array::u64(vec![2], vec![1, 2]).unwrap()),
            ]),
        })
        .unwrap();
        w.finish().unwrap();
        let run = RunIndex::open(p, ReadLimits::default()).unwrap();
        (d, run)
    }
    fn paint(pane: &mut LearningPane, ctx: &egui::Context, run: &RunIndex) {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            pane.show(ui, run, &Selection::default()).unwrap();
        });
        output.textures_delta.clear();
    }
    #[test]
    fn secondary_section_starts_only_on_expansion_and_releases_on_collapse() {
        let (_d, run) = fixture();
        let ctx = egui::Context::default();
        let mut pane = LearningPane::new(ctx.clone()).unwrap();
        paint(&mut pane, &ctx, &run);
        assert_eq!(pane.worker.metrics().submitted, 0);
        assert_eq!(pane.worker.metrics().reads, 0);
        pane.set_expanded(true);
        paint(&mut pane, &ctx, &run);
        assert_eq!(pane.worker.metrics().submitted, 1);
        let deadline = Instant::now() + Duration::from_secs(10);
        while pane.worker.pending() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
            paint(&mut pane, &ctx, &run);
        }
        let reads = pane.worker.metrics().reads;
        paint(&mut pane, &ctx, &run);
        assert_eq!(pane.worker.metrics().reads, reads);
        pane.set_expanded(false);
        paint(&mut pane, &ctx, &run);
        assert!(pane.worker.frame().is_none());
        assert_eq!(pane.worker.metrics().owned_bytes, 0);
    }

    #[test]
    fn compact_history_keeps_exact_summary_and_opens_full_details_without_hover() {
        let (_directory, run) = fixture();
        let ctx = egui::Context::default();
        let mut pane = LearningPane::new(ctx.clone()).unwrap();
        pane.set_expanded(true);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            paint(&mut pane, &ctx, &run);
            if pane.frame().is_some() {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        let selection = Selection {
            revision: 3,
            ..Selection::default()
        };
        let mut render = |events| {
            let mut height = 0.;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1440., 900.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    pane.show(ui, &run, &selection).unwrap();
                    height = ui.min_rect().height();
                },
            );
            fn collect(shape: &egui::Shape, result: &mut Vec<(String, egui::Rect)>) {
                match shape {
                    egui::Shape::Text(text) => result.push((
                        text.galley.job.text.clone(),
                        egui::Rect::from_min_size(text.pos, text.galley.size()),
                    )),
                    egui::Shape::Vec(shapes) => {
                        for shape in shapes {
                            collect(shape, result);
                        }
                    }
                    _ => {}
                }
            }
            let mut labels = Vec::new();
            for shape in &output.shapes {
                collect(&shape.shape, &mut labels);
            }
            output.textures_delta.clear();
            (height, labels)
        };
        let (height, labels) = render(vec![]);
        assert!(height <= 190., "Expanded compact history used {height}px");
        assert!(
            labels
                .iter()
                .any(|(text, _)| text == "k=3 · 2.000000 · n=2 · SE 1.000000")
        );
        assert!(
            !labels
                .iter()
                .any(|(text, _)| text.contains("Actual k=3 · source"))
        );
        let details = labels
            .iter()
            .find(|(text, _)| text == "Learning details")
            .unwrap()
            .1
            .center();
        let event = |pressed| egui::Event::PointerButton {
            pos: details,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        render(vec![egui::Event::PointerMoved(details), event(true)]);
        render(vec![event(false)]);
        let (expanded_height, labels) = render(vec![]);
        assert!(
            (expanded_height - height).abs() < 1.,
            "Opening metadata must preserve the primary workspace allocation: closed {height}px, open {expanded_height}px"
        );
        assert!(
            labels
                .iter()
                .any(|(text, _)| text.contains("Actual k=3 · source")),
            "Full selected provenance is accessible by click: {labels:?}"
        );
        assert!(labels.iter().any(|(text, _)| {
            text.contains("no admitted complete fixed/imported policy population")
        }));
    }

    #[test]
    fn compact_chart_click_selects_a_real_revision_including_unavailable_measurements() {
        let points = [0, 7, 19].map(|revision| LearningPoint {
            revision,
            sequence: Some(revision),
            value: (revision != 7).then_some(revision as f64),
            count: 1,
            standard_error: None,
            conditions: 1,
            critic_name: None,
            continuation: None,
            condition_counts: Vec::new(),
            label: String::new(),
            reason: (revision == 7).then(|| "Missing compatible evidence".into()),
        });
        let ctx = egui::Context::default();
        let mut cursor = egui::Pos2::ZERO;
        let mut render = |events| {
            let mut selected = None;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(500., 200.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let origin = ui.cursor().min;
                    let width = ui.available_width();
                    cursor = origin + egui::vec2(58. + (width - 66.) * 7. / 19., 112. - 23. - 3.);
                    selected = chart(ui, &points, Some([0, 19]), 11, crate::theme::CURRENT);
                },
            );
            output.textures_delta.clear();
            (selected, cursor)
        };
        let (_, position) = render(vec![]);
        let event = |pressed| egui::Event::PointerButton {
            pos: position,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        assert_eq!(
            render(vec![egui::Event::PointerMoved(position), event(true)]).0,
            None
        );
        assert_eq!(
            render(vec![event(false)]).0,
            Some(7),
            "Shared navigation selects the recorded missing point, never interpolated k=11"
        );
    }
}
