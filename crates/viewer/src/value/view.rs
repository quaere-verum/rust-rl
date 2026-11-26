//! Value presentation. All critic evaluation and distribution preparation belongs to the worker.
pub(crate) mod colors;
mod controls;
mod dashes;
mod drawing;
mod plot;
mod presentation;
mod selected;
use super::{prepared::*, slice::*};
use crate::{
    landscape,
    pane::SelectionIntent,
    policy::{
        prepared::{ModelSearch, PolicyViewLease, VisitSource},
        slice::SliceMode,
        view::{Display, PolicyContextKey},
    },
};
use eframe::egui::{self, TextureHandle};
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{value::*, visualization::EnvironmentSemantics},
};
use std::sync::Arc;
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CriticChoice {
    pub name: String,
    pub twin: Option<String>,
    pub kind: ValueCriticKind,
}
pub(crate) struct DecisionControls {
    x: ValueCoordinate,
    y: Option<ValueCoordinate>,
    conditioning: ActionConditioning,
    display: Display,
    comparison: ComparisonCondition,
    colors: colors::ColorMode,
    mode: crate::policy::slice::SliceMode,
    resolution: [usize; 2],
}
#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum ReturnProfileMode {
    #[default]
    Cdf,
    Quantile,
    Histogram,
}

#[derive(Default)]
pub struct ValueView {
    pub(crate) reservation: Option<PolicyViewLease>,
    pub(crate) config: Option<ValueSlice>,
    pub(crate) pin: Option<ValuePin>,
    pub(crate) request: Option<ValueRequest>,
    pub(crate) current_search: Option<ModelSearch>,
    pub(crate) decision_population_search: Option<ModelSearch>,
    pub(crate) value_search: Option<ModelSearch>,
    pub(crate) policy_search: Option<ModelSearch>,
    pub(crate) context_key: Option<PolicyContextKey>,
    pub(crate) source_eligibility: Option<(bool, Option<crate::data::CacheKey>)>,
    pub(crate) choices: Vec<CriticChoice>,
    pub(crate) choice: Option<CriticChoice>,
    pub(crate) metadata_key: Option<crate::data::CacheKey>,
    pub(crate) metadata_models: Option<bool>,
    pub(crate) probe_metadata_key: Option<crate::data::CacheKey>,
    pub(crate) probe_metadata_cursor: usize,
    pub(crate) reference_metadata_key: Option<crate::data::CacheKey>,
    pub(crate) reference_choices: Vec<CriticChoice>,
    pub(crate) reference_policies: Vec<String>,
    pub(crate) anchor_key: Option<(crate::data::CacheKey, usize)>,
    pub(crate) visits: Vec<VisitSource>,
    pub(crate) visits_key: Option<(Arc<std::path::PathBuf>, String, u64, usize)>,
    pub(crate) range_seed_key: Option<(Arc<std::path::PathBuf>, String, u64, usize)>,
    pub(crate) visits_enabled: bool,
    pub(crate) reference_enabled: bool,
    pub(crate) empirical_frontier:
        Option<(crate::data::CacheKey, Arc<std::path::PathBuf>, usize, usize)>,
    pub(crate) comparison_available: bool,
    pub(crate) difference_available: bool,
    pub(crate) dirty: bool,
    pub(crate) status: Option<String>,
    pub(crate) ready: bool,
    pub(crate) terminal: bool,
    pub(crate) output: Option<ValueOutput>,
    pub(crate) decision_controls: Option<DecisionControls>,
    pub(crate) unavailable_policy: Option<(crate::data::CacheKey, Option<String>)>,
    pub(crate) condition: Option<ComparisonCondition>,
    pub display: Display,
    pub main_rect: Option<egui::Rect>,
    pub inspector_rect: Option<egui::Rect>,
    pub texture_uploads: usize,
    texture: Option<TextureHandle>,
    texture_grid: std::sync::Weak<PreparedValueGrid>,
    texture_key: Option<(usize, Display, [u64; 2], landscape::Palette)>,
    color_clipped: [usize; 2],
    dash_admission: std::rc::Rc<dashes::DashAdmission>,
    range: Option<[f64; 2]>,
    pub(crate) colors: colors::Colors,
    initialized: bool,
    main_quantity: Option<(Display, ValueOutput, bool)>,
    narrow_open: bool,
    pub(crate) resolution_explicit: [bool; 2],
    resolutions: [[usize; 2]; 2],
    search: String,
    visit_cycle: usize,
    pending_visit: Option<usize>,
    pending_empirical_input: Option<crate::data::SelectedInput>,
    empirical_cdf_admitted: bool,
    return_profile: ReturnProfileMode,
    focus_distribution: bool,
    pub(crate) empirical_condition: Option<crate::data::CacheKey>,
    #[cfg(feature = "screenshots")]
    pub(crate) capture_configured: bool,
    #[cfg(feature = "screenshots")]
    pub(crate) capture_decision_loss: bool,
}
impl ValueView {
    fn adopt_empirical(
        &mut self,
        condition: &super::empirical::EmpiricalCondition,
    ) -> rl::Result<()> {
        let current = &self
            .request
            .as_ref()
            .ok_or_else(|| rl::Error::State("Value source unavailable".into()))?
            .current;
        let c = self
            .config
            .as_mut()
            .ok_or_else(|| rl::Error::State("Value slice unavailable".into()))?;
        let mut state =
            crate::policy::slice::SliceConfig::new(current.semantics(), &condition.input, &[])?;
        state.ranges.clone_from(&c.state.ranges);
        state.range_sources.clone_from(&c.state.range_sources);
        state.tolerance = c.state.tolerance;
        c.state = state;
        self.pin_input(condition.input.clone(), condition.action.clone())?;
        self.empirical_condition = Some(condition.source_key.clone());
        self.anchor_key = Some((condition.probe_key.clone(), condition.row));
        self.pending_empirical_input = Some(crate::data::SelectedInput::ProbeRow {
            set: condition.probe_set.clone(),
            row: condition.row,
            sequence: condition.probe_key.sequence,
        });
        Ok(())
    }
    pub fn new() -> Self {
        Self {
            resolutions: [[256, 1], [64, 64]],
            #[cfg(feature = "screenshots")]
            capture_decision_loss: std::env::var("RL_VIEWER_VALUE_OUTPUT").as_deref()
                == Ok("decision_loss"),
            ..Self::default()
        }
    }
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    pub(crate) fn decision_loss_requested(&self) -> bool {
        let active = self.output == Some(ValueOutput::DecisionLoss);
        #[cfg(feature = "screenshots")]
        {
            active || self.capture_decision_loss
        }
        #[cfg(not(feature = "screenshots"))]
        {
            active
        }
    }
    pub fn slice(&self) -> Option<&ValueSlice> {
        self.config.as_ref()
    }
    pub(crate) fn automatic_resolution(&self) -> bool {
        !self.resolution_explicit[usize::from(
            self.config
                .as_ref()
                .is_some_and(|c| c.mode == SliceMode::TwoDimensional),
        )]
    }
    pub fn ready(&self) -> bool {
        self.ready
    }
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }
    pub fn color_limits(&self) -> Option<[f64; 2]> {
        self.range
    }
    pub fn texture_bytes(&self) -> usize {
        self.texture
            .as_ref()
            .map_or(0, |t| t.size()[0] * t.size()[1] * 4)
    }
    pub fn state_reservation_bytes(&self) -> usize {
        self.reservation.as_ref().map_or(0, |r| r.bytes)
    }
    pub fn owned_state_bytes(&self) -> usize {
        let mut charges = super::source::ViewSourceCharges::default();
        std::mem::size_of::<Self>()
            + std::mem::size_of::<dashes::DashAdmission>()
            + self.texture_bytes()
            + self.colors.owned_bytes(&mut charges)
            + self.pin.as_ref().map_or(0, ValuePin::bytes)
            + self.empirical_condition.as_ref().map_or(0, |_| 512)
            + self.visits.capacity() * std::mem::size_of::<VisitSource>()
            + self.config.as_ref().map_or(0, ValueSlice::owned_bytes)
            + self.unavailable_policy.as_ref().map_or(0, |(key, actor)| {
                key.root.as_os_str().len() * 2 + actor.as_ref().map_or(0, String::capacity)
            })
            + self
                .decision_controls
                .as_ref()
                .map_or(0, |c| match &c.conditioning {
                    ActionConditioning::Fixed(Action::Continuous(a)) => a.capacity() * 4,
                    _ => 0,
                })
            + self
                .request
                .as_ref()
                .map_or(0, |r| r.view_owned_bytes(&mut charges))
            + [
                &self.current_search,
                &self.decision_population_search,
                &self.value_search,
                &self.policy_search,
            ]
            .into_iter()
            .flatten()
            .map(|s| s.view_owned_bytes_with(&mut charges))
            .sum::<usize>()
            + self
                .choices
                .iter()
                .chain(&self.reference_choices)
                .map(|c| {
                    c.name.capacity()
                        + c.twin.as_ref().map_or(0, String::capacity)
                        + std::mem::size_of::<CriticChoice>()
                })
                .sum::<usize>()
            + self
                .reference_policies
                .iter()
                .map(String::capacity)
                .sum::<usize>()
            + self.search.capacity()
            + self.status.as_ref().map_or(0, String::capacity)
            + 4096
    }
    pub fn main_plot_id() -> egui::Id {
        egui::Id::unique("value-main")
    }
    pub fn set_color_limits(&mut self, range: [f64; 2]) -> rl::Result<()> {
        if !range.iter().all(|x| x.is_finite()) || range[0] >= range[1] {
            return Err(rl::Error::Config("Invalid Value scale".into()));
        }
        self.range = Some(if self.display == Display::Difference {
            let m = range[0].abs().max(range[1].abs());
            [-m, m]
        } else {
            range
        });
        self.colors.mode = colors::ColorMode::Manual;
        if self.display == Display::Difference {
            self.colors.manual.difference = self.range;
        } else {
            self.colors.manual.value = self.range;
        }
        Ok(())
    }
    /// Pin complete inputs; an absent action preserves policy conditioning only for a state slice.
    pub fn pin_input(
        &mut self,
        observation: Observation,
        action: Option<Action>,
    ) -> rl::Result<()> {
        self.empirical_condition = None;
        let c = self
            .config
            .as_mut()
            .ok_or_else(|| rl::Error::State("Value inputs unavailable".into()))?;
        observation.validate(c.state.fixed.len(), c.action_space(), false)?;
        if let Some(a) = &action {
            c.action_space().check(a, observation.dense_mask())?;
            c.conditioning = ActionConditioning::Fixed(a.clone());
        }
        let action = action.or_else(|| match &c.conditioning {
            ActionConditioning::Fixed(a) => Some(a.clone()),
            _ => None,
        });
        self.pin = Some(
            if c.state_only()
                && action.is_none()
                && matches!(c.conditioning, ActionConditioning::Policy)
            {
                ValuePin::State(observation)
            } else {
                ValuePin::Input {
                    observation,
                    action,
                }
            },
        );
        self.dirty = true;
        self.ready = false;
        Ok(())
    }
    pub fn set_axes(&mut self, x: ValueCoordinate, y: Option<ValueCoordinate>) -> rl::Result<()> {
        let request = self
            .request
            .as_ref()
            .ok_or_else(|| rl::Error::State("Value source unavailable".into()))?;
        let mut c = self
            .config
            .clone()
            .ok_or_else(|| rl::Error::State("Value inputs unavailable".into()))?;
        self.resolutions[usize::from(c.mode == SliceMode::TwoDimensional)] = c.resolution;
        c.x = x;
        c.y = y;
        c.mode = if y.is_some() {
            SliceMode::TwoDimensional
        } else {
            SliceMode::OneDimensional
        };
        c.resolution = self.resolutions[usize::from(y.is_some())];
        c.validate(request.current.semantics())?;
        if request.current.kind() == ValueCriticKind::StateValue && !c.state_only() {
            return Err(rl::Error::Config("V requires observation axes".into()));
        }
        self.config = Some(c);
        self.initialized = false;
        self.dirty = true;
        self.ready = false;
        Ok(())
    }
    pub fn select_cell(&mut self, g: &PreparedValueGrid, index: usize) -> rl::Result<()> {
        let (pin, slice) = g.pin_at(index)?;
        self.pin = Some(pin);
        self.config = Some(slice);
        self.dirty = true;
        self.ready = false;
        Ok(())
    }
    pub fn select_visit(
        &mut self,
        g: &PreparedValueGrid,
        index: usize,
    ) -> rl::Result<SelectionIntent> {
        let (pin, slice, intent) = g.select_visit(index)?;
        self.pin = Some(pin);
        self.config = Some(slice);
        self.dirty = true;
        self.ready = false;
        Ok(intent)
    }
    pub(crate) fn quantity(&self) -> ValueOutput {
        self.output.clone().unwrap_or(ValueOutput::ExpectedReturn)
    }
    fn comparison(&self) -> ComparisonCondition {
        self.condition.unwrap_or(ComparisonCondition::SameInput)
    }
    fn project(&self, g: &PreparedValueGrid, i: usize) -> Option<f64> {
        if self.display != Display::Reference && !g.valid.get(i).copied().unwrap_or(false) {
            return None;
        }
        match self.display {
            Display::Current => g.values.get(i).copied(),
            Display::Reference => g.reference_values.get(i).copied().flatten(),
            Display::Difference => g.difference_at(i),
        }
        .filter(|v| v.is_finite())
    }
    fn scale(&mut self, g: &PreparedValueGrid) -> [f64; 2] {
        let mut range = if g.empirical_reference() && self.display == Display::Difference {
            // Conditional evidence does not follow the calibration model's revision.
            g.difference_color_domain.unwrap_or([-1., 1.])
        } else if g.empirical_reference() && self.display == Display::Reference {
            g.reference_color_domain.unwrap_or([0., 1.])
        } else {
            self.colors.scale(g, self.display == Display::Difference)
        };
        if g.empirical_reference()
            && self.reference_enabled
            && self.display == Display::Current
            && g.slice.y.is_none()
            && let Some(series) = &g.reference_series
        {
            for &value in &series.values {
                range[0] = range[0].min(value);
                range[1] = range[1].max(value);
            }
            let pad = ((range[1] - range[0]) * 0.04).max(1e-6);
            range = [range[0] - pad, range[1] + pad];
        }
        self.range = Some(range);
        range
    }
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedValueFrame>,
    ) -> Option<SelectionIntent> {
        self.dash_admission.reset(0);
        let frame = frame.filter(|f| {
            self.quantity() != ValueOutput::DecisionLoss
                || self.ready && f.grid.output() == &ValueOutput::DecisionLoss
        });
        let frame = frame.filter(|f| {
            self.empirical_cdf_admitted = false;
            let base = self.drawing_bytes(f);
            let available = self.state_reservation_bytes().saturating_sub(self.owned_state_bytes()).saturating_sub(base);
            self.empirical_cdf_admitted = f.selected.as_ref().and_then(|p|p.empirical.as_ref())
                .is_some_and(|e| e.cdf.capacity().saturating_mul(128).saturating_add(1024) <= available);
            let drawing = self.drawing_bytes(f);
            if self.owned_state_bytes().saturating_add(drawing) > self.state_reservation_bytes() {
                self.status =
                    Some("Value drawing exceeds shared view admission; reduce resolution or disable overlays".into());
                self.terminal = true;
                self.ready = false;
                false
            } else {
                self.dash_admission.reset(self.state_reservation_bytes()
                    .saturating_sub(self.owned_state_bytes()).saturating_sub(drawing));
                true
            }
        });
        let mut intent = None;
        let has_distribution =
            frame
                .and_then(|frame| frame.selected.as_ref())
                .is_some_and(|point| {
                    point.decision_loss.is_none()
                        && point
                            .profile
                            .as_ref()
                            .and_then(PreparedValueProfile::cdf)
                            .is_some()
                });
        if has_distribution && (self.focus_distribution || ui.available_width() < 850.) {
            if ui
                .button(if self.focus_distribution {
                    "Restore layout"
                } else {
                    "Expand distribution"
                })
                .clicked()
            {
                self.focus_distribution = !self.focus_distribution;
            }
        } else if !has_distribution {
            self.focus_distribution = false;
        }
        if self.focus_distribution {
            self.main_rect = None;
            let response = egui::ScrollArea::vertical()
                .id_salt("value-distribution-expanded")
                .auto_shrink([false, false])
                .show(ui, |ui| self.selected(ui, frame, false));
            self.inspector_rect = Some(response.inner_rect);
            if let Some(status) =
                landscape::complete_frame_status(self.status.as_deref(), self.ready)
            {
                ui.painter().with_clip_rect(response.inner_rect).text(
                    response.inner_rect.right_top() + egui::vec2(-8., 8.),
                    egui::Align2::RIGHT_TOP,
                    status,
                    egui::TextStyle::Small.resolve(ui.style()),
                    crate::theme::MUTED,
                );
            }
            return self
                .pending_empirical_input
                .take()
                .map(SelectionIntent::Input);
        }
        if let Some(f) = frame.filter(|f| f.grid.population_count > 0) {
            if let Some(p) = f.grid.recorded_population() {
                ui.weak(format!(
                    "{:?} · {} · k={}",
                    p.population,
                    p.id,
                    f.grid.current().identity().revision
                ))
                .on_hover_text(&p.methodology);
            }
            intent = landscape::sample_navigator(
                ui,
                f.grid.population_count,
                f.selected
                    .as_ref()
                    .and_then(|p| match p.pin() {
                        ValuePin::Probe { row, .. } => Some(*row),
                        _ => None,
                    })
                    .unwrap_or(0),
            )
            .and_then(|row| f.grid.population_selection(row));
            if f.grid.output() != &ValueOutput::DecisionLoss && f.grid.promote_selected() {
                if f.grid.slice.mode != SliceMode::SamplesOnly
                    && f.grid.slice.x != ValueCoordinate::DiscreteAction
                    && f.grid.slice.y != Some(ValueCoordinate::DiscreteAction)
                {
                    let response = ui.scope(|ui| self.main(ui, e, frame, 120.));
                    intent = response.inner.or(intent);
                    self.main_rect = Some(response.response.rect);
                } else {
                    self.main_rect = None;
                }
                ui.strong("Selected");
                let response = egui::ScrollArea::vertical()
                    .id_salt("value-promoted")
                    .show(ui, |ui| self.selected(ui, frame, false));
                self.inspector_rect = Some(response.inner_rect);
                if let Some(status) =
                    landscape::complete_frame_status(self.status.as_deref(), self.ready)
                {
                    let rect = self.main_rect.unwrap_or(response.inner_rect);
                    ui.painter().with_clip_rect(rect).text(
                        rect.right_top() + egui::vec2(-8., 8.),
                        egui::Align2::RIGHT_TOP,
                        status,
                        egui::TextStyle::Small.resolve(ui.style()),
                        crate::theme::MUTED,
                    );
                }
                return self
                    .pending_empirical_input
                    .take()
                    .map(SelectionIntent::Input)
                    .or(intent);
            }
        }
        let mut height = ui.available_height().max(160.);
        let three_panels = ui.available_width() >= 850.
            && height >= 360.
            && frame
                .and_then(|f| f.selected.as_ref())
                .is_some_and(|p| p.action_curve.is_some());
        if three_panels {
            let upper = (height * 0.39).clamp(160., 350.);
            crate::theme::panel_frame().show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                self.action_curve(ui, e, frame.unwrap(), upper - 20.);
            });
            ui.add_space(8.);
            height = ui.available_height().max(160.);
        }
        let selected = frame.is_some_and(|f| f.selected.is_some());
        let mut l = landscape::layout(
            ui.available_width(),
            height,
            selected && (ui.available_width() >= 850. || self.narrow_open),
        );
        if !l.stacked && selected {
            l.selected_width = if three_panels {
                (ui.available_width() * 0.46).max(340.)
            } else if has_distribution {
                (ui.available_width() * 0.36).clamp(340., 520.)
            } else {
                304.
            };
            l.main_width = (ui.available_width() - l.selected_width - 16.).max(180.);
        }
        self.inspector_rect = None;
        if !l.stacked {
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(l.main_width, height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        if three_panels {
                            let panel = crate::theme::panel_frame().show(ui, |ui| {
                                ui.set_min_size(egui::vec2(
                                    (l.main_width - 24.).max(0.),
                                    (height - 24.).max(0.),
                                ));
                                self.main(ui, e, frame, (height - 24.).max(140.))
                            });
                            intent = panel.inner.or(intent.take());
                            self.main_rect = Some(panel.response.rect);
                        } else {
                            ui.set_min_width(l.main_width);
                            intent = self.main(ui, e, frame, height).or(intent.take());
                            self.main_rect = Some(ui.min_rect());
                        }
                    },
                );
                if !selected {
                    return;
                }
                ui.add_space(8.);
                ui.allocate_ui_with_layout(
                    egui::vec2(l.selected_width, height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let draw = |this: &mut Self, ui: &mut egui::Ui| {
                            egui::ScrollArea::vertical()
                                .id_salt("value-selected")
                                .auto_shrink([false, false])
                                .max_height((height - 35.).max(100.))
                                .show(ui, |ui| {
                                    this.selected(ui, frame, three_panels && has_distribution);
                                    if three_panels && has_distribution {
                                        egui::CollapsingHeader::new("Input and evidence").show(
                                            ui,
                                            |ui| {
                                                ui.push_id("value-details", |ui| {
                                                    this.selected(ui, frame, false)
                                                });
                                            },
                                        );
                                    }
                                });
                        };
                        if three_panels {
                            let panel = crate::theme::panel_frame().show(ui, |ui| {
                                ui.set_min_size(egui::vec2(
                                    (l.selected_width - 24.).max(0.),
                                    (height - 24.).max(0.),
                                ));
                                draw(self, ui);
                            });
                            self.inspector_rect = Some(panel.response.rect);
                        } else {
                            draw(self, ui);
                            self.inspector_rect = Some(ui.min_rect());
                        }
                    },
                );
            });
        } else {
            let r = ui.allocate_ui_with_layout(
                egui::vec2(l.main_width, l.plot_height.max(140.)),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.main(ui, e, frame, l.plot_height.max(140.)),
            );
            intent = r.inner.or(intent);
            self.main_rect = Some(r.response.rect);
            self.inspector_rect = None;
            let r = egui::CollapsingHeader::new("Selected")
                .default_open({
                    #[cfg(feature = "screenshots")]
                    {
                        std::env::var("RL_VIEWER_SELECTED_OPEN").as_deref() == Ok("true")
                    }
                    #[cfg(not(feature = "screenshots"))]
                    {
                        false
                    }
                })
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("value-selected-narrow")
                        .auto_shrink([false, false])
                        .max_height((height * 0.5 - 36.).max(100.))
                        .show(ui, |ui| self.selected(ui, frame, false));
                    self.inspector_rect = Some(ui.min_rect());
                });
            self.narrow_open = r.body_response.is_some();
        }
        if self.dash_admission.rejected.get() {
            self.status = Some(
                "Value dashed drawing exceeds shared view admission; zoom out or disable reference"
                    .into(),
            );
            self.ready = false;
            self.terminal = true;
            if let Some(rect) = self.main_rect {
                ui.painter().text(
                    rect.left_top() + egui::vec2(65., 28.),
                    egui::Align2::LEFT_TOP,
                    self.status.as_deref().unwrap(),
                    egui::FontId::proportional(12.),
                    crate::theme::WARNING,
                );
            }
        }
        if let (Some(i), Some(f)) = (self.pending_visit.take(), frame) {
            intent = self.select_visit(&f.grid, i).ok();
        }
        self.pending_empirical_input
            .take()
            .map(SelectionIntent::Input)
            .or(intent)
    }
}
pub(crate) fn coordinate_label(e: EnvironmentSemantics<'_>, a: ValueCoordinate) -> String {
    match a {
        ValueCoordinate::Observation(i) => landscape::observation_label(e, i),
        ValueCoordinate::Action(i) => e
            .action(i)
            .map_or_else(|| format!("action[{i}]"), |a| a.label.into_owned()),
        ValueCoordinate::DiscreteAction => "Action category".into(),
    }
}
fn source_output_label(source: Option<&ValueSource>, output: &ValueOutput) -> &'static str {
    output_label_for_target(
        source.is_some_and(|s| s.return_target.is_some_and(ReturnTarget::is_soft)),
        output,
    )
}

fn output_label_for_target(soft: bool, output: &ValueOutput) -> &'static str {
    if soft {
        match output {
            ValueOutput::ExpectedReturn => return "Soft Q",
            ValueOutput::Q1 => return "Soft Q1",
            ValueOutput::Q2 => return "Soft Q2",
            ValueOutput::Minimum => return "Min(soft Q1, soft Q2)",
            _ => {}
        }
    }
    output_label(output)
}

pub fn output_label(output: &ValueOutput) -> &'static str {
    match output {
        ValueOutput::ExpectedReturn => "Expected return",
        ValueOutput::DecisionLoss => "Decision loss",
        ValueOutput::Q1 => "Q1",
        ValueOutput::Q2 => "Q2",
        ValueOutput::Minimum => "Min(Q1, Q2)",
        ValueOutput::Disagreement => "Disagreement",
        ValueOutput::Quantile { .. } => "Quantile",
        ValueOutput::Cvar { .. } => "CVaR",
        ValueOutput::RawSecondMoment => "Raw second moment",
        ValueOutput::Variance => "Implied variance",
        ValueOutput::StandardDeviation => "Standard deviation",
        ValueOutput::Utility => "Utility",
    }
}

fn coordinate_axis_label(e: EnvironmentSemantics<'_>, a: ValueCoordinate) -> String {
    match a {
        ValueCoordinate::Observation(i) => landscape::observation_axis_label(e, i),
        ValueCoordinate::Action(i) => {
            landscape::with_unit(coordinate_label(e, a), e.action(i).and_then(|a| a.unit))
        }
        ValueCoordinate::DiscreteAction => coordinate_label(e, a),
    }
}
