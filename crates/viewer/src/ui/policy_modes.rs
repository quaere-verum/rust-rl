//! Execution mode routing; both lanes feed the existing PolicyPane renderer.
use super::*;
use crate::policy::{
    cache::{PreparedLimits, PreparedMetrics, PreparedProvider},
    prepared::{PolicyPin, SourceIdentity},
    slice::SliceConfig,
    view::{Display, PolicyPane},
};
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        policy_cache::CacheEntry, policy_slice::PolicySliceSpec,
        visualization::EnvironmentSemantics,
    },
};
use std::{collections::BTreeMap, io::Write, path::PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyMode {
    Prepared,
    Interactive,
}
struct SavedSlice {
    config: SliceConfig,
    pin: Option<PolicyPin>,
    input: Option<Observation>,
    anchor: Option<(CacheKey, CacheKey, usize)>,
    domain: Option<[f64; 2]>,
    explicit: [bool; 2],
    reference: bool,
}
pub(super) struct PreparedSlice {
    id: String,
    spec: PolicySliceSpec,
    entries: Arc<Vec<Arc<CacheEntry>>>,
    positions: Vec<u64>,
    checkpoints: usize,
}
#[derive(Default)]
pub(super) struct PolicyExecution {
    pub mode: Option<PolicyMode>,
    pub provider: Option<PreparedProvider>,
    groups: Vec<PreparedSlice>,
    catalog_token: usize,
    indexed_records: usize,
    slice: Option<String>,
    pub sequence: Option<u64>,
    missing_selection: bool,
    reference_key: Option<String>,
    reference_options: Arc<Vec<Arc<CacheEntry>>>,
    reference_slice: Option<String>,
    requested_slice: Option<String>,
    painted_quantity: Option<(String, usize)>,
    fit_pending: bool,
    reference_unavailable: bool,
    saved: Option<SavedSlice>,
    preferred: Option<PolicySliceSpec>,
    reference_identity: Option<SourceIdentity>,
    pub interactive_source: Option<SourceIdentity>,
    pub export_open: bool,
    exported: Vec<PolicySliceSpec>,
    pub export_text: String,
    pub export_path: String,
    export_status: Option<String>,
}
impl PolicyExecution {
    fn clear_catalog(&mut self) {
        self.groups.clear();
        self.reference_options = Default::default();
        self.reference_slice = None;
        self.catalog_token = 0;
    }
    fn group(&self) -> Option<&PreparedSlice> {
        let id = self.slice.as_ref()?;
        self.groups.iter().find(|g| &g.id == id)
    }
}
impl Laboratory {
    pub(super) fn leave_prepared_policy(&mut self) {
        if let Some(mut provider) = self.policy_execution.provider.take() {
            provider.retire();
            debug_assert!(self.prepared_retiring.is_none());
            self.prepared_retiring = Some(provider);
        }
        self.policy_execution.clear_catalog();
    }
    pub(super) fn reap_prepared_retirement(&mut self) {
        if self
            .prepared_retiring
            .as_ref()
            .is_some_and(|p| p.retired() && p.metrics().retained_bytes == 0)
        {
            self.prepared_retiring = None;
        }
    }
    pub fn policy_mode(&self) -> Option<PolicyMode> {
        self.policy_execution.mode
    }
    pub fn prepared_policy_metrics(&self) -> Option<PreparedMetrics> {
        self.policy_execution
            .provider
            .as_ref()
            .or(self.prepared_retiring.as_ref())
            .map(PreparedProvider::metrics)
    }
    pub fn set_policy_mode(&mut self, mode: PolicyMode) {
        let run = self.run.clone();
        self.change_policy_mode(mode, run.as_ref());
    }
    fn requested_source(&self, run: &RunIndex) -> (Option<u64>, bool) {
        if let Some(frame) = self.policy_frame().filter(|f| {
            f.grid.current.revision == self.selection.revision
                && f.grid.current.key.root == *run.root()
        }) {
            return (Some(frame.grid.current.key.sequence), false);
        }
        let entries = run.recording().entries();
        let exact = entries.iter().rev().find(|e| {
            e.kind == RecordKind::ModelSnapshot
                && e.revision == self.selection.revision
                && e.model_role == self.selection.model_role
        });
        if let Some(entry) = exact {
            return (Some(entry.sequence), false);
        }
        (
            entries
                .iter()
                .find(|e| e.revision == self.selection.revision)
                .or_else(|| entries.first())
                .map(|e| e.sequence),
            true,
        )
    }
    fn change_policy_mode(&mut self, mode: PolicyMode, run: Option<&RunIndex>) {
        if self.policy_execution.mode == Some(mode) {
            if mode == PolicyMode::Prepared
                && self
                    .policy_execution
                    .provider
                    .as_ref()
                    .is_some_and(|p| p.retired() && p.error().is_some())
            {
                self.leave_prepared_policy();
            }
            return;
        }
        if mode == PolicyMode::Prepared {
            if self.policy_execution.provider.as_ref().is_some_and(|p| {
                !p.catalog_pending() && p.catalog().is_none() && p.error().is_some()
            }) {
                self.leave_prepared_policy();
            }
            if let Some(run) = run {
                let (sequence, missing) = self.requested_source(run);
                self.policy_execution.sequence = sequence;
                self.policy_execution.missing_selection = missing;
                self.policy_execution.preferred = self.export_slice_for(run).ok();
            }
            self.policy_execution.reference_identity = self
                .interactive
                .policy_frame()
                .and_then(|f| f.grid.reference.clone());
            self.policy_execution.reference_unavailable = self.policy_pane.reference_enabled;
            if !self.policy_pane.reference_enabled
                || self.policy_execution.reference_identity.is_some()
            {
                self.policy_execution.reference_key = None;
            }
            self.policy_execution.reference_slice = None;
            self.policy_execution.saved =
                self.policy_pane.config.clone().map(|config| SavedSlice {
                    config,
                    pin: self.policy_pane.pin.clone(),
                    input: self.policy_pane.pin_input.clone(),
                    anchor: self.policy_pane.anchor_key.clone(),
                    domain: self.policy_pane.domain,
                    explicit: self.policy_pane.resolution_explicit,
                    reference: self.policy_pane.reference_enabled,
                });
        } else {
            self.policy_execution.interactive_source = self
                .policy_execution
                .group()
                .and_then(|g| {
                    self.policy_execution
                        .sequence
                        .and_then(|s| g.entries.iter().find(|e| e.source.sequence == s))
                })
                .map(|entry| SourceIdentity {
                    key: CacheKey {
                        root: run.map(|r| r.root().clone()).unwrap_or_default(),
                        sequence: entry.source.sequence,
                    },
                    probe_key: None,
                    revision: entry.source.revision,
                    name: entry.slice.policy.clone(),
                    recorded: false,
                    explicit_controller: false,
                });
            if self.policy_execution.saved.is_none() {
                self.policy_execution.saved =
                    self.policy_pane.config.clone().map(|config| SavedSlice {
                        config,
                        pin: self.policy_pane.pin_input.clone().map(PolicyPin::Synthetic),
                        input: self.policy_pane.pin_input.clone(),
                        anchor: self.policy_pane.anchor_key.clone(),
                        domain: self.policy_pane.domain,
                        explicit: [true, true],
                        reference: self.policy_pane.reference_enabled,
                    });
            }
            if let Some(provider) = &mut self.policy_execution.provider {
                provider.cancel();
            }
        }
        self.interactive.retire_session();
        self.snapshot_catalog = None;
        self.transport.clear_snapshot_state();
        self.policy_pane.clear();
        self.workspace_defaults = None;
        if mode == PolicyMode::Interactive
            && let Some(saved) = self.policy_execution.saved.take()
        {
            self.policy_pane.config = Some(saved.config);
            self.policy_pane.pin = saved.pin;
            self.policy_pane.pin_input = saved.input;
            self.policy_pane.anchor_key = saved.anchor;
            self.policy_pane.domain = saved.domain;
            self.policy_pane.resolution_explicit = saved.explicit;
            self.policy_pane.reference_enabled = saved.reference;
        }
        self.policy_execution.mode = Some(mode);
    }
    pub(super) fn ensure_policy_execution(&mut self, ctx: &egui::Context, run: &RunIndex) {
        self.reap_prepared_retirement();
        if self.prepared_retiring.is_some()
            && self.policy_execution.mode != Some(PolicyMode::Interactive)
        {
            self.policy_pane.ready = false;
            self.policy_pane.terminal = false;
            self.policy_pane.status = Some("Waiting for previous prepared work to release…".into());
            ctx.request_repaint_after(Duration::from_millis(16));
            return;
        }
        if self.policy_execution.mode.is_none() && self.interactive.policy_frame().is_some() {
            self.policy_execution.mode = Some(PolicyMode::Interactive);
        }
        if self.policy_execution.provider.is_none()
            && self.policy_execution.mode != Some(PolicyMode::Interactive)
        {
            let path = std::env::var_os("RL_VIEWER_POLICY_CACHE")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    let mut name = run.root().file_name().unwrap_or_default().to_os_string();
                    name.push("-policy-cache");
                    run.root().with_file_name(name)
                });
            let wake = ctx.clone();
            match PreparedProvider::new(run.clone(), path, PreparedLimits::default(), move || {
                wake.request_repaint()
            }) {
                Ok(provider) => self.policy_execution.provider = Some(provider),
                Err(error) => {
                    self.policy_pane.status = Some(error.to_string());
                    self.policy_pane.terminal = true;
                }
            }
        }
        if let Some(provider) = &mut self.policy_execution.provider {
            provider.poll();
        }
        let token = self
            .policy_execution
            .provider
            .as_ref()
            .and_then(|p| p.catalog())
            .map_or(0, |c| c as *const _ as usize);
        if token != 0 && token != self.policy_execution.catalog_token {
            let mut groups: BTreeMap<String, PreparedSlice> = BTreeMap::new();
            for entry in &self
                .policy_execution
                .provider
                .as_ref()
                .unwrap()
                .catalog()
                .unwrap()
                .entries
            {
                let group =
                    groups
                        .entry(entry.slice_sha256.clone())
                        .or_insert_with(|| PreparedSlice {
                            id: entry.slice_sha256.clone(),
                            spec: entry.slice.clone(),
                            entries: Arc::new(vec![]),
                            positions: vec![],
                            checkpoints: 0,
                        });
                Arc::get_mut(&mut group.entries)
                    .expect("unpublished group")
                    .push(entry.clone());
                group.positions.push(entry.source.sequence);
            }
            self.policy_execution.groups = groups.into_values().collect();
            self.policy_execution.catalog_token = token;
            self.policy_execution.indexed_records = 0;
            let preferred = self.policy_execution.preferred.as_ref();
            if let Some(group) = self
                .policy_execution
                .groups
                .iter()
                .find(|g| {
                    preferred.is_some_and(|p| {
                        p.policy == g.spec.policy
                            && p.model_role == g.spec.model_role
                            && p.same_inputs(&g.spec)
                    })
                })
                .or_else(|| {
                    self.policy_execution
                        .groups
                        .iter()
                        .find(|g| Some(&g.id) == self.policy_execution.slice.as_ref())
                })
                .or_else(|| {
                    self.policy_execution
                        .groups
                        .iter()
                        .find(|g| g.spec.model_role == self.selection.model_role)
                })
                .or_else(|| self.policy_execution.groups.first())
            {
                self.policy_execution.slice = Some(group.id.clone());
            }
        }
        if self.policy_execution.indexed_records != run.recording().entries().len() {
            let mut counts = BTreeMap::new();
            for entry in run
                .recording()
                .entries()
                .iter()
                .filter(|e| e.kind == RecordKind::ModelSnapshot)
            {
                *counts.entry(entry.model_role.as_str()).or_insert(0usize) += 1;
            }
            for group in &mut self.policy_execution.groups {
                group.checkpoints = counts
                    .get(group.spec.model_role.as_str())
                    .copied()
                    .unwrap_or(0);
            }
            self.policy_execution.indexed_records = run.recording().entries().len();
        }
        if self.policy_execution.mode.is_none()
            && self
                .policy_execution
                .provider
                .as_ref()
                .is_some_and(|p| !p.catalog_pending())
        {
            if self.policy_execution.groups.is_empty() {
                self.policy_execution.mode = Some(PolicyMode::Interactive);
            } else {
                let (sequence, missing) = self.requested_source(run);
                self.policy_execution.sequence = sequence;
                self.policy_execution.missing_selection = missing;
                self.policy_execution.mode = Some(PolicyMode::Prepared);
                self.workspace_defaults = None;
                self.interactive.retire_session();
                self.snapshot_catalog = None;
            }
        }
        if self.policy_execution.mode == Some(PolicyMode::Prepared)
            && self
                .policy_execution
                .group()
                .and_then(|g| {
                    self.policy_execution
                        .sequence
                        .and_then(|s| g.entries.iter().find(|e| e.source.sequence == s))
                })
                .is_some_and(|e| e.source.revision != self.selection.revision)
        {
            let (sequence, missing) = self.requested_source(run);
            self.policy_execution.sequence = sequence;
            self.policy_execution.missing_selection = missing;
        }
    }
    pub(super) fn policy_mode_controls(&mut self, ui: &mut egui::Ui, run: &RunIndex) {
        self.ensure_policy_execution(ui.ctx(), run);
        let mut mode = self.policy_execution.mode;
        ui.horizontal(|ui| {
            crate::landscape::segment(ui, &mut mode, Some(PolicyMode::Prepared), "Prepared");
            crate::landscape::segment(ui, &mut mode, Some(PolicyMode::Interactive), "Interactive");
            if mode == Some(PolicyMode::Interactive) { ui.weak("ⓘ").on_hover_text("Interactive mode evaluates arbitrary slices from saved models. Computation and checkpoint playback may be slow."); }
        });
        if let Some(mode) = mode
            && self.policy_execution.mode != Some(mode)
        {
            self.change_policy_mode(mode, Some(run));
            self.ensure_policy_execution(ui.ctx(), run);
        }
    }
    pub fn select_prepared_source(&mut self, sequence: u64) -> Result<()> {
        let entry = self
            .policy_execution
            .group()
            .and_then(|g| g.entries.iter().find(|e| e.source.sequence == sequence))
            .cloned()
            .ok_or_else(|| {
                rl::Error::State("prepared source is unavailable for this slice".into())
            })?;
        self.policy_execution.sequence = Some(sequence);
        self.policy_execution.missing_selection = false;
        self.selection.revision = entry.source.revision;
        self.selection.model_role = entry.source.model_role.clone();
        Ok(())
    }
    pub fn select_prepared_slice(&mut self, slice: &str) -> Result<()> {
        if !self.policy_execution.groups.iter().any(|g| g.id == slice) {
            return Err(rl::Error::State("prepared slice is unavailable".into()));
        }
        self.policy_execution.slice = Some(slice.into());
        Ok(())
    }
    pub(super) fn prepared_policy_transport(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        self.ensure_policy_execution(ui.ctx(), run);
        if self.policy_execution.mode == Some(PolicyMode::Interactive) {
            return Ok(());
        }
        let checking = self
            .policy_execution
            .provider
            .as_ref()
            .is_none_or(|p| p.catalog_pending());
        let positions = self
            .policy_execution
            .group()
            .map_or(&[][..], |g| g.positions.as_slice());
        let published = self
            .policy_execution
            .provider
            .as_ref()
            .and_then(|p| p.frame())
            .map(|f| f.grid.current.key.sequence);
        let ready = self
            .policy_execution
            .provider
            .as_ref()
            .is_some_and(|p| p.ready())
            && !self.policy_pane.dirty;
        let error = self
            .policy_execution
            .provider
            .as_ref()
            .and_then(|p| p.error());
        let mut sequence = self
            .policy_execution
            .sequence
            .unwrap_or_else(|| run.recording().entries().first().map_or(0, |e| e.sequence));
        let before = sequence;
        let retry = self.transport.show_prepared(
            ui,
            run,
            &mut sequence,
            crate::timeline_ui::SnapshotPresentation {
                positions,
                published,
                ready,
                checking,
                error,
            },
        )?;
        if sequence != before {
            self.select_prepared_source(sequence)?;
        }
        if retry {
            self.leave_prepared_policy();
        }
        Ok(())
    }
    pub(super) fn prepared_policy_workspace(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        let e = EnvironmentSemantics::new(run.recording().manifest());
        if self
            .policy_execution
            .provider
            .as_ref()
            .is_none_or(|p| p.catalog_pending())
        {
            self.policy_evolution_controls(ui);
            if self.policy_evolution {
                return Ok(());
            }
            self.policy_pane.ready = false;
            self.policy_pane.terminal = false;
            ui.weak(if self.prepared_retiring.is_some() {
                "Waiting for previous prepared work to release…"
            } else {
                "Checking prepared Policy data…"
            });
            ui.ctx().request_repaint_after(Duration::from_millis(16));
            return Ok(());
        }
        ui.horizontal_wrapped(|ui| {
            self.policy_evolution_controls(ui);
            if self.policy_evolution {
                return;
            }
            ui.label("Slice");
            egui::ComboBox::from_id_salt("prepared-policy-slice")
                .selected_text(self.policy_execution.group().map_or_else(
                    || "No prepared slices".into(),
                    |g| {
                        format!(
                            "{} · x{}{} · {}×{}",
                            g.spec.policy,
                            g.spec.x,
                            g.spec.y.map_or(String::new(), |y| format!(" / x{y}")),
                            g.spec.resolution[0],
                            g.spec.resolution[1]
                        )
                    },
                ))
                .show_ui(ui, |ui| {
                    for (i, group) in self.policy_execution.groups.iter().enumerate() {
                        ui.selectable_value(
                            &mut self.policy_execution.slice,
                            Some(group.id.clone()),
                            format!(
                                "Slice {} · {} · {}",
                                i + 1,
                                group.spec.model_role,
                                group.spec.policy
                            ),
                        );
                    }
                });
            if ui
                .small_button("Reload")
                .on_hover_text("Reload prepared data")
                .clicked()
            {
                self.leave_prepared_policy();
            }
            if ui.small_button("Fit").clicked() {
                self.policy_execution.fit_pending = true;
                self.policy_pane.domain = None;
            }
        });
        if self.policy_evolution {
            return Ok(());
        }
        let Some(group) = self.policy_execution.group() else {
            self.policy_pane.ready = false;
            self.policy_pane.terminal = true;
            self.policy_pane.status = Some(
                self.policy_execution
                    .provider
                    .as_ref()
                    .and_then(|p| p.error())
                    .unwrap_or(
                        "No compatible prepared Policy data. Run rl-precompute policy separately.",
                    )
                    .into(),
            );
            ui.weak(self.policy_pane.status.as_ref().unwrap());
            return Ok(());
        };
        let spec = group.spec.clone();
        let checkpoints = group.checkpoints;
        let entries = group.entries.clone();
        if self.policy_execution.requested_slice != self.policy_execution.slice {
            self.policy_execution.requested_slice = self.policy_execution.slice.clone();
            self.policy_pane.cached_row = None;
            self.policy_pane.domain = None;
        }
        if self.policy_execution.reference_slice != self.policy_execution.slice {
            self.policy_execution.reference_options = Arc::new(
                self.policy_execution
                    .groups
                    .iter()
                    .filter(|g| g.spec.same_inputs(&spec))
                    .flat_map(|g| g.entries.iter().cloned())
                    .collect(),
            );
            self.policy_execution.reference_slice = self.policy_execution.slice.clone();
            if self.policy_execution.reference_key.is_none()
                && let Some(identity) = &self.policy_execution.reference_identity
                && identity.key.root == *run.root()
            {
                self.policy_execution.reference_key = self
                    .policy_execution
                    .reference_options
                    .iter()
                    .find(|entry| {
                        entry.source.sequence == identity.key.sequence
                            && entry.slice.policy == identity.name
                    })
                    .map(|entry| entry.key.clone());
            }
        }
        let references = self.policy_execution.reference_options.clone();
        if let Some(key) = &self.policy_execution.reference_key {
            self.policy_execution.reference_unavailable = !references.iter().any(|e| &e.key == key);
        }
        let desired = self
            .policy_execution
            .sequence
            .and_then(|s| entries.iter().find(|e| e.source.sequence == s))
            .cloned()
            .filter(|_| !self.policy_execution.missing_selection);
        let mut config = spec.to_config(e)?;
        let action = self
            .policy_pane
            .config
            .as_ref()
            .map_or(spec.action, |c| c.action);
        config.action = action.min(
            match e.action_space() {
                ActionSpace::Discrete { count } => *count,
                s => s.dimensions(),
            } - 1,
        );
        self.policy_pane.config = Some(config);
        let before_action = self.policy_pane.config.as_ref().unwrap().action;
        let before_display = self.policy_pane.display;
        ui.horizontal_wrapped(|ui| {
            crate::landscape::popover(ui, "Cache details", |ui| {
            ui.strong("Exact cached source");
            egui::ComboBox::from_id_salt("prepared-checkpoint")
                .selected_text(desired.as_ref().map_or_else(
                    || format!("Unavailable at k={}", self.selection.revision),
                    |e| format!("k={} · source {}", e.source.revision, e.source.sequence),
                ))
                .show_ui(ui, |ui| {
                    for entry in entries.iter() {
                        if ui
                            .selectable_label(
                                desired.as_ref().is_some_and(|e| e.key == entry.key),
                                format!(
                                    "k={} · source {}",
                                    entry.source.revision, entry.source.sequence
                                ),
                            )
                            .clicked()
                        {
                            let _ = self.select_prepared_source(entry.source.sequence);
                        }
                    }
                });
            ui.small(format!("{} / {checkpoints} cached", entries.len())).on_hover_text("Valid cached predictions for this slice / recorded model checkpoints in this role. No missing predictions are inferred automatically.");
            });
            ui.label("Action");
            let count = match e.action_space() {
                ActionSpace::Discrete { count } => *count,
                s => s.dimensions(),
            };
            egui::ComboBox::from_id_salt("prepared-action")
                .selected_text(PolicyPane::action(
                    e,
                    self.policy_pane.config.as_ref().unwrap().action,
                ))
                .show_ui(ui, |ui| {
                    for action in 0..count {
                        ui.selectable_value(
                            &mut self.policy_pane.config.as_mut().unwrap().action,
                            action,
                            PolicyPane::action(e, action),
                        );
                    }
                });
            ui.label("Reference");
            egui::ComboBox::from_id_salt("prepared-reference")
                .selected_text(self.policy_execution.reference_key.as_ref().map_or_else(
                    || {
                        if self.policy_execution.reference_unavailable {
                            "Unavailable".into()
                        } else {
                            "Off".into()
                        }
                    },
                    |key| {
                        references
                            .iter()
                            .find(|e| &e.key == key)
                            .map_or("Unavailable".into(), |e| {
                                format!("{} · source {}", e.slice.policy, e.source.sequence)
                            })
                    },
                ))
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(&mut self.policy_execution.reference_key, None, "Off")
                        .clicked()
                    {
                        self.policy_execution.reference_unavailable = false;
                        self.policy_pane.set_display(Display::Current);
                    }
                    for entry in references.iter() {
                        if ui
                            .selectable_value(
                                &mut self.policy_execution.reference_key,
                                Some(entry.key.clone()),
                                format!(
                                    "{} / {} · k={} · source {}",
                                    entry.slice.policy,
                                    entry.source.model_role,
                                    entry.source.revision,
                                    entry.source.sequence
                                ),
                            )
                            .clicked()
                        {
                            self.policy_execution.reference_unavailable = false;
                        }
                    }
                });
            let reference = self
                .policy_execution
                .reference_key
                .as_ref()
                .and_then(|key| references.iter().find(|e| &e.key == key));
            ui.selectable_value(&mut self.policy_pane.display, Display::Current, "Current");
            ui.add_enabled_ui(reference.is_some(), |ui| {
                ui.selectable_value(
                    &mut self.policy_pane.display,
                    Display::Reference,
                    "Reference",
                );
                ui.selectable_value(
                    &mut self.policy_pane.display,
                    Display::Difference,
                    "Difference",
                );
            });
        });
        if self.policy_pane.config.as_ref().unwrap().action != before_action {
            self.policy_pane.domain = None;
        }
        if self.policy_pane.display != before_display {
            let next = self.policy_pane.display;
            self.policy_pane.display = before_display;
            self.policy_pane.set_display(next);
        }
        if self.policy_execution.reference_unavailable {
            ui.weak("Reference unavailable in this cache").on_hover_text("Choose a cached reference explicitly. No missing predictions are inferred automatically.");
        }
        let desired = self
            .policy_execution
            .sequence
            .and_then(|s| entries.iter().find(|e| e.source.sequence == s))
            .cloned()
            .filter(|_| !self.policy_execution.missing_selection);
        let Some(entry) = desired else {
            if let Some(provider) = &mut self.policy_execution.provider {
                provider.cancel();
            }
            self.policy_pane.ready = false;
            self.policy_pane.terminal = true;
            self.policy_pane.status = Some("Prepared predictions unavailable for the requested checkpoint. Choose a cached checkpoint or run precomputation.".into());
            self.policy_pane.show(
                ui,
                e,
                self.policy_execution
                    .provider
                    .as_ref()
                    .and_then(|p| p.frame()),
            );
            return Ok(());
        };
        let bound = PolicyPane::reservation_bound(e)?;
        if bound > crate::budgets::PREPARED_POLICY_VIEW {
            return Err(rl::Error::Config(
                "Prepared Policy view state exceeds its separate 4 MiB budget".into(),
            ));
        }
        self.policy_pane.prepared_reservation = crate::budgets::PREPARED_POLICY_VIEW;
        let reference = self
            .policy_execution
            .reference_key
            .as_ref()
            .and_then(|key| references.iter().find(|e| &e.key == key));
        self.policy_pane.reference_enabled =
            reference.is_some() || self.policy_execution.reference_unavailable;
        let row = self.policy_pane.cached_row.unwrap_or(0).min(entry.rows - 1);
        let next = entries.iter().position(|e| e.key == entry.key).unwrap();
        let prefetch = entries
            .iter()
            .skip(next + 1)
            .take(2)
            .map(|e| e.key.clone())
            .collect::<Vec<_>>();
        let provider = self.policy_execution.provider.as_mut().unwrap();
        if self.policy_pane.domain.is_none()
            && !self.policy_execution.fit_pending
            && provider.ready()
            && let Some(frame) = provider.frame().filter(|f| {
                f.grid.cache_entry().is_some_and(|e| e.key == entry.key)
                    && f.grid.config.action == self.policy_pane.config.as_ref().unwrap().action
            })
            && let Some(domain) = frame
                .selected
                .as_ref()
                .and_then(|p| p.plot.as_ref())
                .map(|p| p.domain)
        {
            self.policy_pane.domain = Some(domain);
            provider.retain_domain(domain);
        }
        provider.request(
            &entry.key,
            self.policy_pane.config.as_ref().unwrap().action,
            row,
            reference.map(|e| e.key.as_str()),
            self.policy_pane.domain,
            &prefetch,
        )?;
        self.policy_pane.ready = provider.ready();
        self.policy_pane.dirty = false;
        self.policy_pane.terminal = provider.error().is_some();
        self.policy_pane.status = provider
            .error()
            .map(str::to_owned)
            .or_else(|| (!provider.ready()).then(|| "Loading prepared frame…".into()));
        if provider.ready()
            && let Some(frame) = provider.frame()
        {
            let quantity = (entry.slice_sha256.clone(), frame.grid.config.action);
            if self.policy_execution.painted_quantity.as_ref() != Some(&quantity) {
                self.policy_pane.cached_quantity_changed(matches!(
                    e.action_space(),
                    ActionSpace::Discrete { .. }
                ));
                if self
                    .policy_execution
                    .painted_quantity
                    .as_ref()
                    .is_none_or(|old| old.0 != quantity.0)
                {
                    self.policy_pane.fit_prepared();
                }
                self.policy_execution.painted_quantity = Some(quantity);
            }
            if self.policy_execution.fit_pending {
                self.policy_pane.fit_prepared();
                self.policy_execution.fit_pending = false;
            }
            self.policy_pane.pin_input = frame.selected.as_ref().map(|p| p.input.clone());
            if self.policy_pane.domain.is_none()
                && let Some(domain) = frame
                    .selected
                    .as_ref()
                    .and_then(|p| p.plot.as_ref())
                    .map(|p| p.domain)
            {
                self.policy_pane.domain = Some(domain);
                provider.retain_domain(domain);
            }
        }
        self.policy_pane.show(ui, e, provider.frame());
        if provider.pending() {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        Ok(())
    }
    fn export_slice_for(&self, run: &RunIndex) -> Result<PolicySliceSpec> {
        let request = self.policy_pane.request.as_ref().ok_or_else(|| {
            rl::Error::State("Choose a saved model policy before exporting a slice".into())
        })?;
        let source = request.current.identity();
        if source.recorded || source.key.root != *run.root() {
            return Err(rl::Error::Config(
                "Slice export requires a saved model source".into(),
            ));
        }
        let index = run
            .by_sequence(source.key.sequence)
            .ok_or_else(|| rl::Error::State("Slice model source is unavailable".into()))?;
        let name = if source.name.is_empty() {
            self.interactive
                .policy_frame()
                .filter(|f| f.grid.current.key == source.key)
                .map(|f| f.grid.current.name.as_str())
                .ok_or_else(|| {
                    rl::Error::State("Wait for the named model policy before exporting".into())
                })?
        } else {
            &source.name
        };
        let config = self
            .policy_pane
            .config
            .as_ref()
            .ok_or_else(|| rl::Error::State("Policy slice is unavailable".into()))?;
        let spec = PolicySliceSpec::from_config(
            &run.recording().entries()[index].model_role,
            name,
            config,
        );
        spec.to_config(EnvironmentSemantics::new(run.recording().manifest()))?;
        Ok(spec)
    }
    pub fn export_policy_slice(&self) -> Result<PolicySliceSpec> {
        self.export_slice_for(
            self.run
                .as_ref()
                .ok_or_else(|| rl::Error::State("No recording is open".into()))?,
        )
    }
    pub(super) fn policy_export_action(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        if std::mem::take(&mut self.policy_pane.export_requested) {
            match self.export_slice_for(run) {
                Ok(spec) => {
                    if !self.policy_execution.exported.contains(&spec) {
                        self.policy_execution.exported.push(spec);
                    }
                    let text = serde_json::to_string_pretty(&self.policy_execution.exported)?;
                    if text.len() > 1024 * 1024 || self.policy_execution.exported.len() > 256 {
                        self.policy_execution.exported.pop();
                        return Err(rl::Error::Config(
                            "Export exceeds the CLI slice-file limit".into(),
                        ));
                    }
                    self.policy_execution.export_text = text;
                    self.policy_execution.export_open = true;
                    if self.policy_execution.export_path.is_empty() {
                        self.policy_execution.export_path = run
                            .root()
                            .with_file_name("policy-slices.json")
                            .display()
                            .to_string();
                    }
                }
                Err(e) => self.policy_pane.status = Some(e.to_string()),
            }
        }
        let mut open = self.policy_execution.export_open;
        egui::Window::new("Export slices")
            .open(&mut open)
            .resizable(true)
            .show(ui.ctx(), |ui| {
                ui.label(format!(
                    "{} slice specifications · ready for rl-precompute",
                    self.policy_execution.exported.len()
                ));
                ui.add(
                    egui::TextEdit::multiline(&mut self.policy_execution.export_text)
                        .code_editor()
                        .desired_rows(12)
                        .interactive(false),
                );
                if ui.button("Copy JSON").clicked() {
                    ui.ctx()
                        .copy_text(self.policy_execution.export_text.clone());
                }
                ui.add(
                    egui::TextEdit::singleline(&mut self.policy_execution.export_path)
                        .char_limit(4096),
                );
                if ui.button("Save file").clicked() {
                    let result = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&self.policy_execution.export_path)
                        .and_then(|mut file| {
                            file.write_all(self.policy_execution.export_text.as_bytes())?;
                            file.sync_all()
                        });
                    self.policy_execution.export_status = Some(match result {
                        Ok(()) => "Saved".into(),
                        Err(e) => format!("{e}; choose a new filename or copy the JSON"),
                    });
                }
                if let Some(status) = &self.policy_execution.export_status {
                    ui.weak(status);
                }
            });
        self.policy_execution.export_open = open;
        Ok(())
    }
}
