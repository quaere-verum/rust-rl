//! Presentation of worker-prepared policy slices. Scientific evaluation stays on the worker.
mod selected;
use super::{numerics::SelectedDistribution, prepared::*, slice::*};
use crate::pane::SelectionIntent;
use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use egui_plot::{Bar, BarChart, Line, LineStyle, Plot, PlotBounds, PlotImage, PlotPoint, Points};
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::visualization::{EnvironmentSemantics, ObservationType},
};
use std::sync::Arc;

pub const CURRENT: Color32 = crate::theme::CURRENT;
const REFERENCE: Color32 = crate::theme::REFERENCE;
use crate::landscape::axis_tick;
struct ActionValueRange {
    root: Arc<std::path::PathBuf>,
    actor: String,
    critic: String,
    role: String,
    units: String,
    component: usize,
    bounds: [f64; 2],
}
#[derive(PartialEq, Eq)]
pub(crate) struct PolicyContextKey {
    pub root: Arc<std::path::PathBuf>,
    pub revision: u64,
    pub role: String,
    pub input: crate::data::SelectedInput,
    pub reference_serial: u64,
    pub reference_root: Option<Arc<std::path::PathBuf>>,
    pub reference_revision: u64,
    pub reference_enabled: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Display {
    #[default]
    Current,
    Reference,
    Difference,
}
impl Display {
    fn project(self, current: f64, reference: Option<f64>) -> Option<f64> {
        match self {
            Self::Current => Some(current),
            Self::Reference => reference,
            Self::Difference => reference.map(|value| current - value),
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Current => "Current",
            Self::Reference => "Reference",
            Self::Difference => "Difference",
        }
    }
}
#[derive(PartialEq, Eq)]
enum DistributionView {
    Categorical,
    Gaussian {
        cdf: bool,
        reference: Option<(crate::data::CacheKey, bool)>,
    },
}
fn distribution_ymax(point: &PreparedPoint) -> f64 {
    let mut ymax = 1e-6f64;
    for plot in point.plot.iter().chain(&point.reference_plot) {
        ymax = ymax.max(plot.ymax);
    }
    ymax * 1.08
}
#[derive(Default)]
pub struct PolicyPane {
    pub(crate) reservation: Option<PolicyViewLease>,
    pub(crate) config: Option<SliceConfig>,
    pub(crate) pin: Option<PolicyPin>,
    pub(crate) pin_input: Option<Observation>,
    pub(crate) cached_row: Option<usize>,
    pub(crate) export_requested: bool,
    pub(crate) prepared_reservation: usize,
    pub(crate) request: Option<PolicyRequest>,
    pub(crate) input_population: Option<PolicySource>,
    pub(crate) current_search: Option<ModelSearch>,
    pub(crate) reference_search: Option<ModelSearch>,
    pub(crate) context_key: Option<PolicyContextKey>,
    pub(crate) anchor_key: Option<(crate::data::CacheKey, crate::data::CacheKey, usize)>,
    pub(crate) reference_enabled: bool,
    pub(crate) reference_initialized: bool,
    pub(crate) reference_automatic: bool,
    pub(crate) comparison_available: bool,
    pub(crate) visits_enabled: bool,
    pub(crate) visits: Option<VisitSource>,
    pub(crate) visit_cursor: Option<(u64, u64)>,
    pub(crate) visit_loading: Option<usize>,
    pub(crate) visit_checked: usize,
    pub(crate) visit_search_done: bool,
    pub(crate) visit_search_context: Option<(Arc<std::path::PathBuf>, String, u64, usize)>,
    pub(crate) domain: Option<[f64; 2]>,
    pub(crate) dirty: bool,
    pub(crate) edited: bool,
    pub(crate) status: Option<String>,
    pub(crate) ready: bool,
    pub(crate) terminal: bool,
    pub(crate) failed_context: bool,
    #[cfg(feature = "screenshots")]
    pub(crate) capture_configured: bool,
    pub display: Display,
    texture: Option<TextureHandle>,
    texture_grid: std::sync::Weak<PreparedGrid>,
    texture_key: Option<(usize, Display, [u64; 2], crate::landscape::Palette)>,
    pub texture_uploads: usize,
    colour_range: Option<[f64; 2]>,
    difference_range: Option<[f64; 2]>,
    fit: bool,
    initialized: bool,
    main_quantity: Option<(Display, usize)>,
    distribution_initialized: bool,
    distribution_view: Option<DistributionView>,
    // The last Y extent established by us. A different retained extent means
    // deliberate navigation, which automatic peak expansion must preserve.
    distribution_auto_y: Option<[f64; 2]>,
    action_value_range: Option<ActionValueRange>,
    dragging_pin: bool,
    search: String,
    resolutions: [[usize; 2]; 2],
    pub(crate) resolution_explicit: [bool; 2],
    last_y: Option<usize>,
    narrow_inspector_open: bool,
    pub main_rect: Option<egui::Rect>,
    pub inspector_rect: Option<egui::Rect>,
}
impl PolicyPane {
    pub fn new() -> Self {
        Self {
            resolutions: [[256, 1], [64, 64]],
            main_rect: None,
            inspector_rect: None,
            ..Default::default()
        }
    }
    pub fn clear(&mut self) {
        *self = Self::new();
    }
    pub fn texture_bytes(&self) -> usize {
        self.texture
            .as_ref()
            .map_or(0, |t| t.size()[0] * t.size()[1] * 4)
    }
    pub fn config(&self) -> Option<&SliceConfig> {
        self.config.as_ref()
    }
    pub fn pinned_input(&self) -> Option<&Observation> {
        self.pin_input.as_ref()
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
    pub fn state_reservation_bytes(&self) -> usize {
        self.reservation
            .as_ref()
            .map_or(0, |r| r.bytes)
            .saturating_add(self.prepared_reservation)
    }
    pub(crate) fn cached_quantity_changed(&mut self, discrete: bool) {
        self.colour_range = None;
        self.difference_range = None;
        self.domain = None;
        if !discrete {
            self.distribution_initialized = false;
        }
    }
    pub(crate) fn fit_prepared(&mut self) {
        self.fit = true;
        self.colour_range = None;
        self.difference_range = None;
        self.domain = None;
        self.action_value_range = None;
        self.initialized = false;
        self.distribution_initialized = false;
    }
    /// Conservative retained inventory. Shared raw arrays keep their existing
    /// resolver charges; shared source metadata may be counted twice here.
    pub fn owned_state_bytes(&self) -> usize {
        let input = |o: &Observation| {
            std::mem::size_of::<Observation>()
                + o.as_dense()
                    .expect("dense diagnostic input")
                    .values
                    .capacity()
                    * 4
                + o.as_dense()
                    .and_then(|dense| dense.action_mask.as_ref())
                    .map_or(0, Vec::capacity)
        };
        std::mem::size_of::<Self>()
            + self.config.as_ref().map_or(0, SliceConfig::owned_bytes)
            + self.pin_input.as_ref().map_or(0, input)
            + self.pin.as_ref().map_or(0, |p| match p {
                PolicyPin::Synthetic(o) => input(o),
                _ => 0,
            })
            + self
                .request
                .as_ref()
                .map_or(0, PolicyRequest::metadata_bytes)
            + self
                .current_search
                .as_ref()
                .map_or(0, ModelSearch::view_owned_bytes)
            + self
                .reference_search
                .as_ref()
                .map_or(0, ModelSearch::view_owned_bytes)
            + self.context_key.as_ref().map_or(0, |k| {
                k.role.capacity()
                    + match &k.input {
                        crate::data::SelectedInput::ProbeRow { set, .. } => set.capacity(),
                        _ => 0,
                    }
            })
            + self.search.capacity()
            + self.action_value_range.as_ref().map_or(0, |r| {
                r.actor.capacity() + r.critic.capacity() + r.role.capacity() + r.units.capacity()
            })
            + self
                .visit_search_context
                .as_ref()
                .map_or(0, |(_, role, _, _)| role.capacity())
            + self.status.as_ref().map_or(0, String::capacity)
    }
    /// Six simultaneous config buffers cover initialization, transactional
    /// controls and old/new requests; eight full inputs cover independent pins
    /// and replacements. Four 256KiB metadata bounds cover old/new current and
    /// reference search/request handles. This reservation precedes allocation.
    pub(crate) fn reservation_bound(e: EnvironmentSemantics<'_>) -> rl::Result<usize> {
        let dim = e.observation_dim();
        let mask = match e.action_space() {
            ActionSpace::Discrete { count } => *count,
            _ => 0,
        };
        let mut codes = 0usize;
        let mut categorical = 0usize;
        if let Some(d) = e.descriptor() {
            for field in &d.observations {
                if let ObservationType::Categorical { categories } = &field.kind {
                    codes = codes.saturating_add(categories.len());
                    categorical += 1;
                }
            }
        }
        let config = dim
            .saturating_mul(16)
            .saturating_add(mask)
            .saturating_add(codes.saturating_mul(8))
            .saturating_add(categorical.saturating_mul(64))
            .saturating_add(512);
        let input = dim
            .saturating_mul(4)
            .saturating_add(mask)
            .saturating_add(96);
        let bytes = (4 * 256 * 1024usize)
            .saturating_add(64 * 1024)
            .saturating_add(config.saturating_mul(6))
            .saturating_add(input.saturating_mul(8));
        if config > MAX_SLICE_STATE_BYTES || bytes > crate::budgets::POLICY_PREPARED {
            return Err(rl::Error::Config(
                "Policy editable inputs exceed prepared byte budget".into(),
            ));
        }
        Ok(bytes)
    }
    pub fn set_color_limits(&mut self, range: [f64; 2], probability: bool) -> rl::Result<()> {
        if !range[0].is_finite()
            || !range[1].is_finite()
            || range[0] >= range[1]
            || !(range[1] - range[0]).is_finite()
            || (probability && range != [0., 1.])
        {
            return Err(rl::Error::Config("Invalid color limits".into()));
        }
        self.colour_range = Some(if self.display == Display::Difference {
            let limit = range[0].abs().max(range[1].abs());
            [-limit, limit]
        } else {
            range
        });
        Ok(())
    }
    pub fn set_display(&mut self, display: Display) {
        if (display == Display::Difference) != (self.display == Display::Difference) {
            std::mem::swap(&mut self.colour_range, &mut self.difference_range);
        }
        self.display = display;
    }
    pub fn color_limits(&self) -> Option<[f64; 2]> {
        self.colour_range
    }
    pub fn main_plot_id() -> egui::Id {
        egui::Id::unique("policy-main")
    }
    pub fn distribution_plot_id() -> egui::Id {
        egui::Id::unique("policy-distribution")
    }
    pub(crate) fn coordinate(e: EnvironmentSemantics<'_>, i: usize) -> String {
        e.observation(i)
            .map_or_else(|| format!("obs[{i}]"), |o| o.label.into_owned())
    }
    pub(crate) fn action(e: EnvironmentSemantics<'_>, i: usize) -> String {
        e.action(i)
            .map_or_else(|| format!("action[{i}]"), |a| a.label.into_owned())
    }
    fn fixed_editor(&mut self, ui: &mut egui::Ui, e: EnvironmentSemantics<'_>, large: bool) {
        let Some(c) = &mut self.config else { return };
        if c.mode == SliceMode::SamplesOnly {
            ui.horizontal(|ui| {
                ui.weak("Recorded samples · no numerical observation axes");
                let before = c.action;
                let count = match e.action_space() {
                    ActionSpace::Discrete { count } => *count,
                    s => s.dimensions(),
                };
                egui::ComboBox::from_id_salt("policy-action")
                    .selected_text(Self::action(e, c.action))
                    .show_ui(ui, |ui| {
                        egui::ScrollArea::vertical().max_height(220.).show_rows(
                            ui,
                            22.,
                            count,
                            |ui, rows| {
                                for i in rows {
                                    ui.selectable_value(&mut c.action, i, Self::action(e, i));
                                }
                            },
                        );
                    });
                if before != c.action {
                    self.dirty = true;
                    self.ready = false;
                }
                if ui
                    .checkbox(&mut self.reference_enabled, "Reference")
                    .changed()
                {
                    self.reference_automatic = false;
                }
                crate::landscape::popover(ui, "Sample settings", |ui| {
                    if ui.button("Export slice…").clicked() {
                        self.export_requested = true;
                        ui.close();
                    }
                });
            });
            return;
        }
        for i in 0..c.fixed.len() {
            if i == c.x || (c.mode == SliceMode::TwoDimensional && Some(i) == c.y) {
                continue;
            }
            let label = Self::coordinate(e, i);
            if large && !label.to_lowercase().contains(&self.search.to_lowercase()) {
                continue;
            }
            self.dirty |=
                crate::landscape::fixed_input(ui, e, i, &mut c.fixed[i], "policy-fixed-category");
        }
    }
    pub(crate) fn controls(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        reference_available: bool,
    ) {
        if self
            .config
            .as_ref()
            .is_some_and(|c| c.mode == SliceMode::SamplesOnly)
        {
            self.fixed_editor(ui, e, false);
            return;
        }
        let colour_before = self.colour_range;
        let domain_before = self.domain;
        ui.horizontal_wrapped(|ui| {
            let Some(c) = &mut self.config else { return };
            let before = c.clone();
            let numeric: Vec<_> = (0..c.fixed.len())
                .filter(|&i| {
                    !matches!(
                        e.observation(i).and_then(|o| o.kind),
                        Some(ObservationType::Categorical { .. })
                    )
                })
                .collect();
            let mut two = c.mode == SliceMode::TwoDimensional;
            ui.add_enabled_ui(numeric.len() > 1, |ui| {
                crate::landscape::segment(ui, &mut two, true, "Map");
            });
            crate::landscape::segment(ui, &mut two, false, "Slice");
            if two != (c.mode == SliceMode::TwoDimensional) {
                if !two {
                    self.last_y = c.y;
                }
                let old = if c.mode == SliceMode::OneDimensional {
                    0
                } else {
                    1
                };
                self.resolutions[old] = c.resolution;
                let y = two.then(|| {
                    self.last_y
                        .or(c.y)
                        .filter(|&i| i != c.x)
                        .unwrap_or_else(|| *numeric.iter().find(|&&i| i != c.x).unwrap())
                });
                if c.set_axes(e, c.x, y).is_ok() {
                    c.resolution = self.resolutions[usize::from(two)];
                    self.initialized = false;
                }
            }
            ui.label("Action");
            let count = match e.action_space() {
                ActionSpace::Discrete { count } => *count,
                s => s.dimensions(),
            };
            egui::ComboBox::from_id_salt("policy-action")
                .width(130.)
                .selected_text(Self::action(e, c.action))
                .show_ui(ui, |ui| {
                    for i in 0..count {
                        ui.selectable_value(&mut c.action, i, Self::action(e, i));
                    }
                });
            crate::landscape::popover(ui, "Slice settings", |ui| {
                let height = (ui.ctx().content_rect().height() - 120.).max(180.);
                egui::ScrollArea::vertical()
                    .max_height(height)
                    .show(ui, |ui| {
                        ui.set_width(300.);
                        ui.strong("Axes and domain");
                        ui.label("X axis");
                        let mut x = c.x;
                        egui::ComboBox::from_id_salt("policy-x")
                            .width(130.)
                            .selected_text(Self::coordinate(e, x))
                            .show_ui(ui, |ui| {
                                for &i in &numeric {
                                    ui.selectable_value(&mut x, i, Self::coordinate(e, i));
                                }
                            });
                        if x != c.x {
                            let y = if c.y == Some(x) { Some(c.x) } else { c.y };
                            let _ = c.set_axes(e, x, y);
                            self.initialized = false;
                        }
                        if c.mode == SliceMode::TwoDimensional {
                            ui.label("Y axis");
                            let mut y = c.y.unwrap();
                            egui::ComboBox::from_id_salt("policy-y")
                                .width(130.)
                                .selected_text(Self::coordinate(e, y))
                                .show_ui(ui, |ui| {
                                    for &i in &numeric {
                                        ui.selectable_value(&mut y, i, Self::coordinate(e, i));
                                    }
                                });
                            if Some(y) != c.y {
                                let x = if y == c.x { c.y.unwrap() } else { c.x };
                                let _ = c.set_axes(e, x, Some(y));
                                self.initialized = false;
                            }
                        }
                        ui.small(format!(
                            "Evaluated grid: {} x {} · {}",
                            c.resolution[0],
                            c.resolution[1],
                            if self.resolution_explicit
                                [usize::from(c.mode == SliceMode::TwoDimensional)]
                            {
                                "explicit"
                            } else {
                                "automatic admission"
                            }
                        ));
                        if ui.button("Automatic resolution").clicked() {
                            self.resolution_explicit
                                [usize::from(c.mode == SliceMode::TwoDimensional)] = false;
                            c.resolution = if c.mode == SliceMode::TwoDimensional {
                                [128, 128]
                            } else {
                                [256, 1]
                            };
                            self.dirty = true;
                        }
                        for (name, axis) in [
                            ("X range", Some(c.x)),
                            (
                                "Y range",
                                c.y.filter(|_| c.mode == SliceMode::TwoDimensional),
                            ),
                        ] {
                            if let Some(i) = axis {
                                ui.horizontal(|ui| {
                                    ui.label(name);
                                    ui.add(egui::DragValue::new(&mut c.ranges[i][0]).speed(0.01));
                                    ui.add(egui::DragValue::new(&mut c.ranges[i][1]).speed(0.01));
                                });
                            }
                        }
                        let resolution_before = c.resolution;
                        ui.horizontal(|ui| {
                            ui.label("Resolution");
                            ui.add(egui::DragValue::new(&mut c.resolution[0]).range(
                                2..=if c.mode == SliceMode::TwoDimensional {
                                    256
                                } else {
                                    4096
                                },
                            ));
                            if c.mode == SliceMode::TwoDimensional {
                                ui.add(egui::DragValue::new(&mut c.resolution[1]).range(2..=256));
                            }
                        });
                        if c.resolution != resolution_before {
                            self.resolution_explicit
                                [usize::from(c.mode == SliceMode::TwoDimensional)] = true;
                        }
                        ui.horizontal(|ui| {
                            ui.label("Tolerance");
                            ui.add(
                                egui::DragValue::new(&mut c.tolerance)
                                    .range(0.0..=f32::MAX)
                                    .speed(0.001),
                            );
                        });
                        if let Some(range) = &mut self.colour_range {
                            ui.add_enabled_ui(
                                !matches!(e.action_space(), ActionSpace::Discrete { .. })
                                    || self.display == Display::Difference,
                                |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Color limits");
                                        ui.add(egui::DragValue::new(&mut range[0]).speed(0.01));
                                        ui.add(egui::DragValue::new(&mut range[1]).speed(0.01));
                                    });
                                },
                            );
                        }
                        if let Some(domain) = &mut self.domain {
                            ui.horizontal(|ui| {
                                ui.label("Action range");
                                ui.add(egui::DragValue::new(&mut domain[0]).speed(0.01));
                                ui.add(egui::DragValue::new(&mut domain[1]).speed(0.01));
                            });
                        }
                        if ui.button("Fit").clicked() {
                            self.fit = true;
                            self.colour_range = None;
                            self.difference_range = None;
                            self.domain = None;
                            self.action_value_range = None;
                            self.dirty = true;
                            self.initialized = false;
                            self.distribution_initialized = false;
                            ui.close();
                        }
                        if ui.button("Export slice…").clicked() {
                            self.export_requested = true;
                            ui.close();
                        }
                    });
            });
            if (before.x != c.x || before.y != c.y || before.mode != c.mode)
                && let Some(pin) = &self.pin_input
            {
                c.fixed.clear();
                c.fixed.extend_from_slice(pin.dense_values());
            }
            if c.validate(e).is_err() {
                *c = before.clone();
                self.status = Some("Invalid plot settings".into());
            }
            if before.ranges != c.ranges {
                self.initialized = false;
            }
            if before.action != c.action {
                self.colour_range = None;
                self.difference_range = None;
                self.domain = None;
                if !matches!(e.action_space(), ActionSpace::Discrete { .. }) {
                    self.distribution_initialized = false;
                }
            }
            self.dirty |= before != *c;
            self.edited |= before != *c;
            let fixed = c.fixed.len()
                - if c.mode == SliceMode::TwoDimensional {
                    2
                } else {
                    1
                };
            if let Some(range) = self.colour_range
                && Some(range) != colour_before
                && self
                    .set_color_limits(
                        range,
                        matches!(e.action_space(), ActionSpace::Discrete { .. })
                            && self.display != Display::Difference,
                    )
                    .is_err()
            {
                self.colour_range = colour_before;
                self.status = Some("Invalid color limits".into());
            }
            if self.domain != domain_before {
                if self.domain.is_some_and(|[lo, hi]| {
                    !lo.is_finite() || !hi.is_finite() || lo >= hi || !(hi - lo).is_finite()
                }) {
                    self.domain = domain_before;
                    self.status = Some("Invalid action range".into());
                } else {
                    self.dirty = true;
                    self.distribution_initialized = false;
                }
            }
            if self.display == Display::Reference
                && self
                    .config
                    .as_ref()
                    .is_some_and(|c| c.mode == SliceMode::OneDimensional)
            {
                self.set_display(Display::Current);
            }
            if fixed > 0 {
                crate::landscape::popover(ui, "Fixed inputs", |ui| {
                    ui.set_width(300.);
                    if fixed > 6 {
                        ui.add(
                            egui::TextEdit::singleline(&mut self.search)
                                .char_limit(128)
                                .hint_text("Search inputs"),
                        );
                    }
                    egui::ScrollArea::vertical()
                        .max_height(280.)
                        .show(ui, |ui| self.fixed_editor(ui, e, fixed > 6));
                });
            }
            if self.visits.is_some() {
                let changed = ui
                    .checkbox(&mut self.visits_enabled, "Visited states")
                    .changed();
                self.dirty |= changed;
                self.edited |= changed;
            }
            if reference_available {
                let changed = ui
                    .checkbox(&mut self.reference_enabled, "Reference")
                    .changed();
                if changed {
                    self.reference_automatic = false;
                }
                self.dirty |= changed;
                self.edited |= changed;
                if changed && !self.reference_enabled {
                    self.set_display(Display::Current);
                }
            }
            if self.reference_enabled && reference_available && self.comparison_available {
                let previous = self.display;
                let two_dimensional = self
                    .config
                    .as_ref()
                    .is_some_and(|c| c.mode == SliceMode::TwoDimensional);
                if ui.ctx().content_rect().width() < 1100. {
                    let current = match self.display {
                        Display::Current => "Current",
                        Display::Reference => "Reference",
                        Display::Difference => "Difference",
                    };
                    egui::ComboBox::from_id_salt("policy-display-mode")
                        .width(85.)
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.display, Display::Current, "Current");
                            if two_dimensional {
                                ui.selectable_value(
                                    &mut self.display,
                                    Display::Reference,
                                    "Reference",
                                );
                            }
                            ui.selectable_value(
                                &mut self.display,
                                Display::Difference,
                                "Difference",
                            );
                        })
                        .response
                        .on_hover_text("Display");
                } else {
                    ui.selectable_value(&mut self.display, Display::Current, "Current");
                    if two_dimensional {
                        ui.selectable_value(&mut self.display, Display::Reference, "Reference");
                    }
                    ui.selectable_value(&mut self.display, Display::Difference, "Difference");
                }
                if previous != self.display {
                    let display = self.display;
                    self.display = previous;
                    self.set_display(display);
                }
            }
        });
        if self.dirty
            && let (Some(c), Some(input)) = (&self.config, &mut self.pin_input)
        {
            let mut changed = false;
            for i in 0..input.dense_values().len() {
                if i != c.x && !(c.mode == SliceMode::TwoDimensional && Some(i) == c.y) {
                    changed |= input.dense_values()[i] != c.fixed[i];
                    input.as_dense_mut().expect("dense diagnostic input").values[i] = c.fixed[i];
                }
            }
            if changed || c.grid_eligibility() == GridEligibility::ContinuousFeedforward {
                self.pin = Some(PolicyPin::Synthetic(input.clone()));
            }
        }
    }
    fn scale(&mut self, g: &PreparedGrid, e: EnvironmentSemantics<'_>) -> [f64; 2] {
        *self.colour_range.get_or_insert_with(|| {
            if self.display != Display::Difference
                && matches!(e.action_space(), ActionSpace::Discrete { .. })
            {
                return [0., 1.];
            }
            if self.display != Display::Difference
                && let ActionSpace::Box { low, high } = e.action_space()
            {
                let lo = f64::from(low[g.config.action]);
                let hi = f64::from(high[g.config.action]);
                return if lo < 0. && hi > 0. {
                    let limit = lo.abs().max(hi.abs());
                    [-limit, limit]
                } else {
                    [lo, hi]
                };
            }
            let [lo, hi] = crate::landscape::numeric_bounds(
                g.values
                    .iter()
                    .enumerate()
                    .flat_map(|(i, &v)| {
                        if self.display == Display::Difference {
                            [g.difference_at(i), None]
                        } else {
                            [Some(v), g.reference_at(i)]
                        }
                    })
                    .flatten(),
                false,
            );
            if self.display == Display::Difference
                || lo < 0.
                || matches!(e.action_space(), ActionSpace::Unbounded { .. })
            {
                let m = lo.abs().max(hi.abs()).max(1e-6);
                [-m, m]
            } else {
                [0., hi.max(1e-6)]
            }
        })
    }
    fn value(&self, g: &PreparedGrid, i: usize) -> Option<f64> {
        self.display.project(*g.values.get(i)?, g.reference_at(i))
    }
    /// Same physical coordinates for rendering and sparse hit testing.
    fn sample_position(
        g: &PreparedGrid,
        i: usize,
        two: bool,
        display: Display,
    ) -> Option<[f64; 2]> {
        let value = display.project(*g.values.get(i)?, g.reference_at(i))?;
        if g.regular() {
            let c = &g.config;
            let [lo, hi] = c.ranges[c.x].map(f64::from);
            let x = (lo + (hi - lo) * i as f64 / (c.resolution[0] - 1) as f64) as f32;
            Some([f64::from(x), value])
        } else {
            let position = g.positions.get(i)?;
            Some([
                f64::from(position[0]),
                if two { f64::from(position[1]) } else { value },
            ])
        }
    }
    fn palette(&self, e: EnvironmentSemantics<'_>, action: usize) -> crate::landscape::Palette {
        use crate::landscape::Palette;
        if self.display == Display::Difference {
            return Palette::Difference;
        }
        match e.action_space() {
            ActionSpace::Discrete { .. } => Palette::Probability,
            ActionSpace::Box { low, high } if low[action] < 0. && high[action] > 0. => {
                Palette::SignedAction
            }
            ActionSpace::Unbounded { .. } => Palette::SignedAction,
            _ => Palette::Sequential,
        }
    }
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedPolicyFrame>,
    ) -> Option<SelectionIntent> {
        let mut intent = None;
        if let Some(f) = frame.filter(|f| f.grid.population_count > 0) {
            let row = f
                .selected
                .as_ref()
                .and_then(|p| match &p.recorded_input {
                    Some(crate::data::SelectedInput::ProbeRow { row, .. }) => Some(*row),
                    _ => None,
                })
                .unwrap_or(0);
            intent = crate::landscape::sample_navigator(ui, f.grid.population_count, row)
                .and_then(|row| f.grid.population_selection(row));
            if let Some(probes) = f.grid.recorded_probes()
                && let rl::diagnostics::schema::Record::ProbeSet(p) = &probes.record
            {
                let label = match p.population {
                    rl::diagnostics::schema::ProbePopulation::FixedReference => {
                        "Fixed probe inputs"
                    }
                    rl::diagnostics::schema::ProbePopulation::UserSupplied => {
                        "User-supplied inputs"
                    }
                    rl::diagnostics::schema::ProbePopulation::PolicyVisited => "Evaluation visits",
                    rl::diagnostics::schema::ProbePopulation::ReplayRetained => {
                        "Factual replay inputs"
                    }
                    rl::diagnostics::schema::ProbePopulation::ImportedHistory => {
                        "Recorded history inputs"
                    }
                };
                ui.weak(label).on_hover_text(format!(
                    "{} · inputs k={}",
                    p.id, probes.context.identity.learner_update_revision
                ));
            }
            if f.grid.promote_selected() {
                if f.grid.config.mode != SliceMode::SamplesOnly {
                    let response = ui.scope(|ui| self.main(ui, e, frame, 120.));
                    intent = response.inner.or(intent);
                    self.main_rect = Some(response.response.rect);
                } else {
                    self.main_rect = None;
                }
                let response = ui.scope(|ui| self.selected(ui, e, frame));
                self.inspector_rect = Some(response.response.rect);
                if let Some(status) =
                    crate::landscape::complete_frame_status(self.status.as_deref(), self.ready)
                {
                    let rect = self.main_rect.unwrap_or(response.response.rect);
                    ui.painter().with_clip_rect(rect).text(
                        rect.right_top() + egui::vec2(-8., 8.),
                        egui::Align2::RIGHT_TOP,
                        status,
                        egui::TextStyle::Small.resolve(ui.style()),
                        crate::theme::MUTED,
                    );
                }
                return intent;
            }
        }
        let height = ui.available_height().max(160.);
        let selected = frame.is_some_and(|f| f.selected.is_some());
        let mut layout = crate::landscape::layout(
            ui.available_width(),
            height,
            selected && (ui.available_width() >= 850. || self.narrow_inspector_open),
        );
        if !layout.stacked && selected {
            layout.selected_width = (ui.available_width() * 0.36).clamp(340., 700.);
            layout.main_width = (ui.available_width() - layout.selected_width - 16.).max(180.);
        }
        self.inspector_rect = None;
        if !layout.stacked {
            let width = ui.available_width();
            let main = layout.main_width;
            ui.horizontal_top(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(main, height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        let panel = crate::theme::panel_frame().show(ui, |ui| {
                            ui.set_min_size(egui::vec2(
                                (main - 24.).max(0.),
                                (height - 24.).max(0.),
                            ));
                            self.main(ui, e, frame, (height - 24.).max(140.))
                        });
                        intent = panel.inner.or(intent.take());
                        self.main_rect = Some(panel.response.rect);
                    },
                );
                if !selected {
                    return;
                }
                ui.add_space(8.);
                ui.allocate_ui_with_layout(
                    egui::vec2((width - main - 16.).max(180.), height),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        self.selected(ui, e, frame);
                        self.inspector_rect = Some(ui.min_rect());
                    },
                );
            });
        } else {
            let plot_height = if self.narrow_inspector_open {
                height * 0.5
            } else {
                height - 36.
            };
            let main = ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), plot_height.max(140.)),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.main(ui, e, frame, plot_height.max(140.)),
            );
            intent = main.inner.or(intent);
            self.main_rect = Some(main.response.rect);
            self.inspector_rect = None;
            let selected = egui::CollapsingHeader::new("Selected")
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
                        .id_salt("policy-selected-narrow")
                        .max_height((height * 0.5 - 36.).max(100.))
                        .show(ui, |ui| self.selected_contents(ui, e, frame));
                    self.inspector_rect = Some(ui.min_rect());
                });
            self.narrow_inspector_open = selected.body_response.is_some();
        }
        // Keep persistent feedback in the existing chart, below foreground tooltips.
        if frame.is_some()
            && let (Some(status), Some(main)) = (
                crate::landscape::complete_frame_status(self.status.as_deref(), self.ready),
                self.main_rect,
            )
        {
            let galley = ui.painter().layout_no_wrap(
                status.clone(),
                egui::TextStyle::Small.resolve(ui.style()),
                crate::theme::MUTED,
            );
            let position = egui::pos2(
                (main.right() - galley.size().x - 12.).max(main.left() + 8.),
                main.top() + 8.,
            );
            let rectangle = egui::Rect::from_min_size(position, galley.size()).expand(3.);
            let painter = ui.painter().with_clip_rect(main.shrink(4.));
            painter.rect_filled(rectangle, 3., crate::theme::SURFACE);
            painter.galley(position, galley, crate::theme::MUTED);
            ui.interact(
                rectangle.intersect(main),
                ui.scope_id().with("policy-status"),
                egui::Sense::hover(),
            )
            .on_hover_text(status);
        }
        intent
    }
    fn main(
        &mut self,
        ui: &mut egui::Ui,
        e: EnvironmentSemantics<'_>,
        frame: Option<&PreparedPolicyFrame>,
        height: f32,
    ) -> Option<SelectionIntent> {
        let Some(frame) = frame else {
            ui.centered_and_justified(|ui| {
                ui.weak(self.status.as_deref().unwrap_or("Loading…"));
            });
            return None;
        };
        let g = &frame.grid;
        let c = &g.config;
        let two = c.mode == SliceMode::TwoDimensional;
        let subject = match (self.display, two) {
            (Display::Current, true) => "Policy map",
            (Display::Current, false) => "Policy slice",
            (Display::Reference, true) => "Reference policy map",
            (Display::Reference, false) => "Reference policy slice",
            (Display::Difference, _) => "Policy difference",
        };
        ui.label(
            egui::RichText::new(format!("{subject} · {}", Self::action(e, c.action)))
                .size(17.)
                .strong(),
        );
        let height = (height - 24.).max(120.);
        let range = self.scale(g, e);
        let indicator = match self.display {
            Display::Current => g.current_indicator,
            Display::Reference => g.reference_indicator,
            Display::Difference => g.current_indicator || g.reference_indicator,
        };
        let palette = if self.display != Display::Difference && indicator {
            crate::landscape::Palette::Categorical
        } else {
            self.palette(e, c.action)
        };
        let scale = crate::landscape::LandscapeScale::new(range, palette);
        let range = scale.range;
        self.colour_range = Some(range);
        if two && g.regular() {
            let key = (
                Arc::as_ptr(g) as usize,
                self.display,
                [range[0].to_bits(), range[1].to_bits()],
                scale.palette,
            );
            if self.texture_key != Some(key) || self.texture_grid.upgrade().is_none() {
                let [nx, ny] = c.resolution;
                let pixels = (0..ny)
                    .rev()
                    .flat_map(|y| (0..nx).map(move |x| y * nx + x))
                    .map(|i| scale.color(self.value(g, i).unwrap_or(f64::NAN)))
                    .collect();
                let image = ColorImage::new([nx, ny], pixels);
                // A diverging difference palette does not make an indicator
                // continuous; mixed indicator/probability differences also jump.
                let options = if indicator {
                    TextureOptions::NEAREST
                } else {
                    TextureOptions::LINEAR
                };
                if let Some(t) = &mut self.texture {
                    t.set(image, options);
                } else {
                    self.texture = Some(ui.ctx().load_texture("Policy", image, options));
                }
                self.texture_key = Some(key);
                self.texture_grid = Arc::downgrade(g);
                self.texture_uploads += 1;
            }
        }
        let plot = Plot::new("policy-main")
            .width((ui.available_width() - if two { 86. } else { 0. }).max(100.))
            .show_background(false)
            .grid_color(crate::theme::GRID)
            .show_grid(!(two && g.regular()))
            .grid_spacing(40.0..=100.0)
            // Decimal powers can jump from 0.1 straight to 1 on a short unit
            // interval. Binary subdivisions preserve interior numerical ticks.
            .x_grid_spacer(egui_plot::log_grid_spacer(2))
            .y_grid_spacer(egui_plot::log_grid_spacer(2))
            .x_axis_formatter(axis_tick)
            .y_axis_formatter(axis_tick)
            .id(Self::main_plot_id())
            .height(height.max(120.))
            .x_axis_label(crate::landscape::observation_axis_label(e, c.x))
            .y_axis_label(if two {
                crate::landscape::observation_axis_label(e, c.y.unwrap())
            } else if self.display == Display::Difference {
                format!("Difference · {}", Self::action(e, c.action))
            } else {
                Self::action(e, c.action)
            })
            .allow_double_click_reset(false)
            .allow_drag(!self.dragging_pin);
        let initialize = !self.initialized || self.fit;
        let quantity_changed = self.main_quantity != Some((self.display, c.action));
        let pin = frame.selected.as_ref().and_then(|point| {
            Some([
                f64::from(point.input.dense_values()[c.x]),
                if two {
                    f64::from(point.input.dense_values()[c.y.unwrap()])
                } else {
                    self.display.project(point.value, point.reference_value)?
                },
            ])
        });
        let response = plot.show_background(false).show(ui, |p| {
            if initialize {
                let xr = c.ranges[c.x];
                let yr = if two {
                    c.ranges[c.y.unwrap()].map(f64::from)
                } else {
                    range
                };
                p.set_plot_bounds(PlotBounds::from_min_max(
                    [f64::from(xr[0]), yr[0]],
                    [f64::from(xr[1]), yr[1]],
                ));
            } else if !two && quantity_changed {
                p.set_plot_bounds_y(range[0]..=range[1]);
            }
            if two && g.regular() {
                let xr = c.ranges[c.x];
                let yr = c.ranges[c.y.unwrap()];
                let dx = (f64::from(xr[1]) - f64::from(xr[0])) / (c.resolution[0] - 1) as f64;
                let dy = (f64::from(yr[1]) - f64::from(yr[0])) / (c.resolution[1] - 1) as f64;
                p.image(PlotImage::new(
                    "Policy",
                    self.texture.as_ref().unwrap().id(),
                    PlotPoint::new(
                        (f64::from(xr[0]) + f64::from(xr[1])) / 2.,
                        (f64::from(yr[0]) + f64::from(yr[1])) / 2.,
                    ),
                    egui::vec2(
                        (f64::from(xr[1]) - f64::from(xr[0]) + dx) as f32,
                        (f64::from(yr[1]) - f64::from(yr[0]) + dy) as f32,
                    ),
                ));
            } else if !two && g.regular() {
                let recorded_reference = g.reference.as_ref().is_some_and(|r| r.recorded);
                let points: Vec<_> = (0..g.values.len())
                    .filter_map(|i| Self::sample_position(g, i, false, self.display))
                    .collect();
                if self.display != Display::Current && recorded_reference {
                    p.points(
                        Points::new(self.display.label(), points)
                            .color(if self.display == Display::Reference {
                                REFERENCE
                            } else {
                                CURRENT
                            })
                            .radius(4.),
                    );
                } else {
                    let line = Line::new(self.display.label(), points);
                    p.line(if self.display == Display::Reference {
                        line.color(REFERENCE).style(LineStyle::dashed_dense())
                    } else {
                        line.color(CURRENT)
                    });
                }
                if self.reference_enabled && self.display == Display::Current {
                    let points: Vec<_> = (0..g.values.len())
                        .filter_map(|i| Self::sample_position(g, i, false, Display::Reference))
                        .collect();
                    if recorded_reference {
                        p.points(Points::new("Reference", points).color(REFERENCE).radius(4.));
                    } else {
                        p.line(
                            Line::new("Reference", points)
                                .color(REFERENCE)
                                .style(LineStyle::dashed_dense()),
                        );
                    }
                }
            } else {
                let mut batches: [Vec<[f64; 2]>; 32] = std::array::from_fn(|_| Vec::new());
                let mut references = Vec::new();
                for i in 0..g.positions.len() {
                    if let Some(position) = Self::sample_position(g, i, two, self.display) {
                        let bin = if two {
                            (((self.value(g, i).unwrap() - range[0]) / (range[1] - range[0]))
                                .clamp(0., 1.)
                                * 31.)
                                .round() as usize
                        } else {
                            0
                        };
                        batches[bin].push(position);
                    }
                    if !two
                        && self.reference_enabled
                        && self.display == Display::Current
                        && let Some(position) =
                            Self::sample_position(g, i, false, Display::Reference)
                    {
                        references.push(position);
                    }
                }
                for (bin, points) in batches
                    .into_iter()
                    .enumerate()
                    .filter(|(_, v)| !v.is_empty())
                {
                    p.points(
                        Points::new(self.display.label(), points)
                            .color(if two {
                                scale.color(range[0] + (range[1] - range[0]) * bin as f64 / 31.)
                            } else {
                                if self.display == Display::Reference {
                                    REFERENCE
                                } else {
                                    CURRENT
                                }
                            })
                            .radius(4.),
                    );
                }
                if !references.is_empty() {
                    p.points(
                        Points::new("Reference", references)
                            .color(REFERENCE)
                            .radius(4.),
                    );
                }
            }
            if self.visits_enabled
                && let Some(visits) = &g.visits
            {
                let points: Vec<_> = visits
                    .markers
                    .iter()
                    .map(|pos| {
                        [
                            f64::from(pos[0]),
                            if two { f64::from(pos[1]) } else { range[0] },
                        ]
                    })
                    .collect();
                p.points(
                    Points::new("Visited states", points)
                        .color(Color32::from_white_alpha(75))
                        .radius(1.8),
                );
            }
            if let Some(pin) = pin {
                p.points(
                    Points::new("Selected", vec![pin])
                        .color(Color32::WHITE)
                        .radius(6.),
                );
                p.points(Points::new("Selected", vec![pin]).color(CURRENT).radius(4.));
            }
            (
                p.ctx()
                    .input(|i| i.pointer.latest_pos())
                    .map(|pos| p.plot_from_screen(pos)),
                *p.transform(),
            )
        });
        self.initialized = true;
        self.main_quantity = Some((self.display, c.action));
        self.fit = false;
        if two && !(indicator && g.promote_selected()) {
            crate::landscape::color_bar(
                ui,
                *response.inner.1.frame(),
                scale,
                &crate::landscape::with_unit(
                    Self::action(e, c.action),
                    e.action(c.action).and_then(|a| a.unit),
                ),
            );
        }
        let mut intent = None;
        if let Some(cursor) = response.inner.0 {
            let nearest = if g.regular() {
                // Keep Policy's endpoint-clamped hover semantics while sharing the physical mapping.
                let x = cursor
                    .x
                    .clamp(c.ranges[c.x][0] as f64, c.ranges[c.x][1] as f64);
                let yr = if two {
                    c.ranges[c.y.unwrap()]
                } else {
                    [0., 1.]
                };
                let y = if two {
                    cursor.y.clamp(yr[0] as f64, yr[1] as f64)
                } else {
                    0.
                };
                crate::landscape::cell_index(
                    [x, y],
                    [c.ranges[c.x], yr],
                    [c.resolution[0], if two { c.resolution[1] } else { 1 }],
                )
            } else {
                let points = (0..g.positions.len()).flat_map(|i| {
                    let primary = Self::sample_position(g, i, two, self.display);
                    let reference =
                        (!two && self.reference_enabled && self.display == Display::Current)
                            .then(|| Self::sample_position(g, i, false, Display::Reference))
                            .flatten();
                    [primary, reference]
                        .into_iter()
                        .flatten()
                        .map(move |point| (i, point))
                });
                crate::landscape::nearest_point(points, cursor, &response.inner.1, 10.)
            };
            if response.response.contains_pointer()
                && let Some(i) = nearest
                && let Ok(input) = g.input(i)
            {
                egui::Tooltip::always_open(
                    ui.ctx().clone(),
                    response.response.layer_id,
                    response.response.id.with("policy-input-tooltip"),
                    egui::PopupAnchor::Pointer,
                )
                .show(|ui| {
                    ui.set_max_width(350.);
                    egui::ScrollArea::vertical()
                        .max_height(240.)
                        .show(ui, |ui| {
                            for (j, v) in input.dense_values().iter().enumerate() {
                                ui.label(format!("{} = {v:?}", Self::coordinate(e, j)));
                            }
                            if let Some(m) = input.dense_mask() {
                                ui.label(format!("Mask {m:?}"));
                            }
                            ui.label(format!("Output {:.8}", g.values[i]));
                            if let Some(v) = g.reference_at(i) {
                                ui.label(format!(
                                    "Reference {v:.8}\nDifference {:.8}",
                                    g.values[i] - v
                                ));
                            }
                            ui.weak(format!("{} · k={}", g.current.name, g.current.revision));
                            if let Some(r) = &g.reference {
                                ui.weak(format!("{} · k={}", r.name, r.revision));
                            }
                        });
                });
            }
            if ui.input(|i| i.pointer.primary_pressed())
                && let Some(pin) = pin
            {
                self.dragging_pin = response
                    .inner
                    .1
                    .position_from_point(&PlotPoint::from(pin))
                    .distance(response.inner.1.position_from_point(&cursor))
                    < 12.;
            }
            let drag = self.dragging_pin && ui.input(|i| i.pointer.primary_down());
            if response.response.clicked() || drag {
                if g.is_cached() {
                    if let Some(row) = nearest {
                        self.cached_row = Some(row);
                        self.dirty = true;
                        self.edited = true;
                        self.ready = false;
                    }
                } else if g.regular() {
                    let mut input = self
                        .pin_input
                        .clone()
                        .unwrap_or_else(|| Observation::new(c.fixed.clone()));
                    input.as_dense_mut().expect("dense diagnostic input").values[c.x] =
                        cursor.x as f32;
                    if two {
                        input.as_dense_mut().expect("dense diagnostic input").values
                            [c.y.unwrap()] = cursor.y as f32;
                    }
                    if input.dense_values().iter().all(|v| v.is_finite()) {
                        self.pin = Some(PolicyPin::Synthetic(input.clone()));
                        self.pin_input = Some(input);
                        self.dirty = true;
                        self.edited = true;
                        self.ready = false;
                    }
                } else if let Some(i) = nearest {
                    self.pin = Some(PolicyPin::Recorded {
                        key: g.current.key.clone(),
                        probe_key: g.current.probe_key.clone().unwrap(),
                        row: g.rows[i],
                    });
                    self.pin_input = g.input(i).ok();
                    self.dirty = true;
                    self.edited = true;
                    self.ready = false;
                    intent = g.selection_intent(i);
                }
            }
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            self.dragging_pin = false;
        }
        intent
    }
}

#[cfg(test)]
#[path = "view/texture_tests.rs"]
mod texture_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn color_limits_reject_invalid_edits_and_keep_probability_and_difference_scales_honest() {
        let mut pane = PolicyPane::new();
        pane.set_color_limits([-2., 2.], false).unwrap();
        let before = pane.colour_range;
        for range in [
            [1., 1.],
            [2., 1.],
            [f64::NAN, 1.],
            [0., f64::INFINITY],
            [-f64::MAX, f64::MAX],
        ] {
            assert!(pane.set_color_limits(range, false).is_err());
            assert_eq!(pane.colour_range, before);
        }
        assert!(pane.set_color_limits([0., 2.], true).is_err());
        pane.set_color_limits([0., 1.], true).unwrap();
        assert_eq!(pane.colour_range, Some([0., 1.]));
        pane.set_display(Display::Reference);
        assert_eq!(pane.colour_range, Some([0., 1.]));
        pane.set_display(Display::Difference);
        pane.set_color_limits([-1., 3.], false).unwrap();
        assert_eq!(pane.colour_range, Some([-3., 3.]));
        pane.set_display(Display::Current);
        assert_eq!(pane.colour_range, Some([0., 1.]));
        pane.set_display(Display::Reference);
        assert_eq!(pane.colour_range, Some([0., 1.]));
    }
}
