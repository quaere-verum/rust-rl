//! Ordinary Policy route: existing selection/reference owners feed one historical worker.
use super::*;
use crate::policy::{
    prepared::{self, ModelSearch, PolicyPin, PolicyRequest, PolicySource, VisitSource},
    slice::{GridEligibility, SliceConfig},
    view::{PolicyContextKey, PolicyPane},
};
use rl::{core::action::ActionSpace, diagnostics::visualization::EnvironmentSemantics};

impl Laboratory {
    #[cfg(feature = "screenshots")]
    pub(super) fn policy_capture_evidence(&self) -> serde_json::Value {
        let Some(frame) = self.policy_frame() else {
            return serde_json::Value::Null;
        };
        serde_json::json!({
            "execution_mode": format!("{:?}", self.policy_mode()),
            "prepared": self.prepared_policy_metrics(),
            "displayed":self.transport.capture_publication(),
            "resolution":frame.grid.config.resolution,
            "resolution_explicit":self.policy_pane.resolution_explicit,
            "input_identity":format!("{:?}",self.selection.input),
            "owners":self.interactive.owner_snapshot(),
            "requested_revision": frame.requested_revision,
            "snapshot_catalog": self.snapshot_catalog.as_ref().map(|c| serde_json::json!({"checking":c.pending(),"limited":c.limited(),"positions":c.positions()})),
            "current": format!("{:?}", frame.grid.current),
            "reference": format!("{:?}", frame.grid.reference),
            "config": format!("{:?}", frame.grid.config),
            "regular": frame.grid.regular(),
            "rows": frame.grid.values.len(),
            "population_rows": frame.grid.population_count,
            "distinct_positions": frame.grid.distinct_positions,
            "input_population": self.policy_pane.request.as_ref().and_then(|r|r.input_population.as_ref()).map(|s|format!("{:?}",s.identity())),
            "scale": self.policy_pane.color_limits(),
            "metrics": self.policy_metrics(),
            "texture_bytes": self.policy_pane.texture_bytes(),
            "texture_uploads": self.policy_pane.texture_uploads,
            "selected": frame.selected.as_ref().map(|point| serde_json::json!({
                "input": point.input.dense_values(), "mask": point.input.dense_mask(),
                "recorded_input": format!("{:?}", point.recorded_input),
                "reference_recorded_input": format!("{:?}", point.reference_recorded_input),
                "output": point.value, "reference": point.reference_value,
                "difference": point.difference, "visits": point.visits,
                "kl": point.kl, "distribution": format!("{:?}", point.distribution),
                "reference_distribution": format!("{:?}", point.reference_distribution),
                "action_domain": point.plot.as_ref().map(|plot| plot.domain),
            })),
            "visitation": frame.grid.visits.as_ref().map(|visits| serde_json::json!({
                "source": format!("{:?}", visits.source), "revision": visits.revision,
                "tolerance": visits.tolerance, "filtered_count": visits.filtered_count,
                "markers": visits.markers.len(),
            })),
        })
    }
    pub fn policy_pane(&self) -> &PolicyPane {
        &self.policy_pane
    }
    pub fn policy_frame(&self) -> Option<&prepared::PreparedPolicyFrame> {
        if self.policy_execution.mode == Some(PolicyMode::Prepared) {
            return self
                .policy_execution
                .provider
                .as_ref()
                .and_then(|p| p.frame());
        }
        self.interactive.policy_frame()
    }
    pub fn policy_pending(&self) -> bool {
        if self.policy_execution.mode == Some(PolicyMode::Prepared) {
            return self
                .policy_execution
                .provider
                .as_ref()
                .is_some_and(|p| p.pending() || p.catalog_pending());
        }
        self.interactive.policy_pending()
    }
    pub fn policy_metrics(&self) -> prepared::PolicyMetrics {
        self.interactive.policy_metrics()
    }
    pub(super) fn run_control(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let manifest = self.run.as_ref().map(|r| r.recording().manifest());
            let name = manifest.map_or("Choose run", |m| m.run_id.as_str());
            let display = manifest.map_or_else(
                || "Choose run".to_owned(),
                |m| crate::presentation::display_run_identifier(&m.run_id),
            );
            let mut chosen = None;
            ui.weak("Run");
            let response = egui::ComboBox::from_id_salt("laboratory-run")
                .selected_text(display)
                .width(170.)
                .truncate()
                .show_ui(ui, |ui| {
                    for (i, c) in self.catalogue.iter().enumerate() {
                        if ui
                            .selectable_label(
                                name == c.manifest.run_id,
                                crate::presentation::display_run_identifier(&c.manifest.run_id),
                            )
                            .on_hover_text(&c.manifest.run_id)
                            .clicked()
                        {
                            chosen = Some(i);
                        }
                    }
                })
                .response;
            if let Some(manifest) = manifest {
                response.on_hover_text(format!(
                    "{}\n{} · {}\n{}",
                    manifest.experiment_id,
                    manifest.algorithm,
                    manifest.environment,
                    manifest.run_id
                ));
            }
            if let Some(i) = chosen {
                self.open_run(i);
            }
        });
    }
    pub(super) fn policy_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        self.ensure_policy_execution(ui.ctx(), run);
        if self.policy_execution.mode == Some(PolicyMode::Prepared) {
            return self.prepared_policy_workspace(ui, run);
        }
        if self.policy_execution.mode.is_none() {
            ui.weak("Checking prepared Policy data…");
            return Ok(());
        }
        ui.visuals_mut().selection.bg_fill = crate::policy::view::CURRENT.linear_multiply(0.45);
        ui.visuals_mut().selection.stroke.color = crate::policy::view::CURRENT;
        let e = EnvironmentSemantics::new(run.recording().manifest());
        let identity_bytes = match &self.selection.input {
            SelectedInput::ProbeRow { set, .. } => set.capacity(),
            _ => 0,
        }
        .saturating_add(self.selection.model_role.capacity())
        .saturating_mul(4);
        let reservation_bound =
            PolicyPane::reservation_bound(e).map(|bytes| bytes.saturating_add(identity_bytes));
        if self.policy_pane.reservation.is_none()
            || reservation_bound
                .as_ref()
                .is_ok_and(|bytes| *bytes > self.policy_pane.state_reservation_bytes())
        {
            let reservation =
                reservation_bound.and_then(|bytes| self.interactive.reserve_policy_view(bytes));
            match reservation {
                Ok(lease) => self.policy_pane.reservation = Some(lease),
                Err(error) => {
                    self.policy_pane.status = Some(error.to_string());
                    self.policy_pane.terminal = true;
                    self.policy_pane.ready = false;
                    self.policy_pane.show(ui, e, None);
                    return Ok(());
                }
            }
        }
        let selected = self.selection.clone();
        let mut view = self
            .resolver
            .resolve_policy(run, &selected, |i| self.loader.request(run, i))?;
        // The original input population is independent of the current model.
        // Snapshot stepping must not substitute newly recorded outputs/masks.
        if let Some(source) = &self.policy_pane.input_population {
            if let (Some(frame), SelectedInput::ProbeRow { sequence, row, .. }) =
                (source.recorded_frame(), &selected.input)
                && frame.probe_key.root == *run.root()
                && frame.probe_key.sequence == *sequence
            {
                let mut frame = frame.clone();
                frame.row = *row;
                view.record = Some(frame);
                view.status = ResolveStatus::Ready;
            } else {
                self.policy_pane.input_population = None;
            }
        }
        let initialization_key = view
            .record
            .as_ref()
            .map(|f| (f.key.clone(), f.probe_key.clone(), f.row));
        if self.policy_pane.anchor_key != initialization_key
            && let Some(frame) = &view.record
        {
            let checked = crate::pane::PolicySource::check_frame(
                frame,
                e.action_space(),
                crate::budgets::POLICY_SCRATCH,
            )?;
            let mut input = prepared::recorded_input(&checked.batch, frame.row)?;
            let mut config = SliceConfig::new(e, &input, &[])?;
            if let Record::ProbeSet(p) = &frame.probes.record
                && !p.histories.is_empty()
            {
                config = config.with_history();
            }
            if let Some(old) = &self.policy_pane.config {
                if old.mode != crate::policy::slice::SliceMode::SamplesOnly {
                    config.set_axes(e, old.x, old.y)?;
                }
                let same_row = self
                    .policy_pane
                    .anchor_key
                    .as_ref()
                    .is_some_and(|(_, probe, row)| probe == &frame.probe_key && *row == frame.row);
                if same_row {
                    config.fixed.clone_from(&old.fixed);
                }
                config.ranges.clone_from(&old.ranges);
                config.range_sources.clone_from(&old.range_sources);
                config.resolution = old.resolution;
                config.action = old.action;
                config.tolerance = old.tolerance;
                if same_row && let Some(pin) = &self.policy_pane.pin_input {
                    let dense = input.as_dense_mut().expect("dense diagnostic input");
                    dense.values.clear();
                    dense.values.extend_from_slice(pin.dense_values());
                }
                if !same_row
                    && let Some(mask) = input.dense_mask()
                    && !mask[config.action]
                {
                    config.action = mask.iter().position(|&v| v).unwrap_or(0);
                }
            } else if let ActionSpace::Discrete { .. } = e.action_space() {
                config.action = checked.batch.actions().as_u64()?[frame.row] as usize;
            }
            self.policy_pane.pin_input = Some(input);
            self.policy_pane.pin = Some(PolicyPin::Recorded {
                key: frame.key.clone(),
                probe_key: frame.probe_key.clone(),
                row: frame.row,
            });
            if matches!(
                config.grid_eligibility(),
                GridEligibility::Masked | GridEligibility::UnsupportedDiscrete
            ) {
                self.policy_pane.input_population =
                    Some(PolicySource::recorded(run, frame, &selected.model_role)?);
            }
            self.policy_pane.config = Some(config);
            self.policy_pane.anchor_key = initialization_key.clone();
            self.policy_pane.dirty = true;
        }
        // No recorded Policy is required for a continuous model when a real selected
        // probe supplies the complete anchor; no synthetic row identity is invented.
        if self.policy_pane.config.is_none()
            && view.status != ResolveStatus::Loading
            && let Some((chunk, row)) = self.selected_source(run)?
        {
            let input = source_observation(&chunk, row)?;
            let mut config = SliceConfig::new(e, &input, &[])?;
            if let Record::ProbeSet(p) = &chunk.record
                && !p.histories.is_empty()
            {
                config = config.with_history();
            }
            self.policy_pane.pin_input = Some(input.clone());
            self.policy_pane.pin = Some(PolicyPin::Synthetic(input));
            self.policy_pane.config = Some(config);
            self.policy_pane.dirty = true;
        }
        if self.policy_pane.config.is_some() && !self.policy_pane.reference_initialized {
            if !self.policy_pane.reference_enabled
                && crate::reference::provider_role(self.comparison.reference_role())
            {
                self.policy_pane.reference_enabled = true;
                self.policy_pane.reference_automatic = true;
            }
            self.policy_pane.reference_initialized = true;
        }
        #[cfg(feature = "screenshots")]
        if self.policy_pane.config.is_some() && !self.policy_pane.capture_configured {
            self.configure_policy_capture(e)?;
        }
        let reference_available = self.comparison.reference_run().is_some();
        ui.horizontal_wrapped(|ui| {
            self.policy_evolution_controls(ui);
            if !self.policy_evolution {
                self.policy_pane.controls(ui, e, reference_available);
            }
        });
        if self.policy_evolution {
            return Ok(());
        }
        self.policy_export_action(ui, run)?;
        if self.policy_pane.reference_enabled {
            ui.horizontal_wrapped(|ui| -> Result<()> {
                if let Some(source) = self
                    .policy_pane
                    .request
                    .as_ref()
                    .and_then(|r| r.reference.as_ref())
                    .filter(|s| {
                        self.comparison.quality_source_is_current(
                            self.policy_pane
                                .context_key
                                .as_ref()
                                .map(|k| k.reference_serial),
                            s.identity(),
                        )
                    })
                {
                    source.reference_quality_ui(ui, true);
                } else {
                    ui.small(crate::reference::quality_label(None));
                }
                Ok(())
            })
            .inner?;
        }
        let key = Some(PolicyContextKey {
            root: run.root().clone(),
            revision: selected.revision,
            role: selected.model_role.clone(),
            input: selected.input.clone(),
            reference_serial: self.comparison.reference_interaction_serial(),
            reference_root: self.comparison.reference_run().map(|r| r.root().clone()),
            reference_revision: self.comparison.reference_revision(),
            reference_enabled: self.policy_pane.reference_enabled,
        });
        if self.policy_pane.context_key != key {
            if !self.interactive.snapshot_validation_pending() {
                self.interactive.invalidate_policy_context();
            }
            self.policy_pane.context_key = key;
            self.policy_pane.current_search = Some(ModelSearch::new(
                &selected.model_role,
                selected.revision,
                None,
            )?);
            self.policy_pane.reference_search = if self.policy_pane.reference_enabled {
                Some(ModelSearch::new(
                    self.comparison.reference_role(),
                    self.comparison.reference_revision(),
                    None,
                )?)
            } else {
                None
            };
            self.policy_pane.request = None;
            self.policy_pane.dirty = true;
            self.policy_pane.ready = false;
            self.policy_pane.terminal = false;
            self.policy_pane.comparison_available = false;
            self.policy_pane.failed_context = false;
        }
        if self.policy_pane.failed_context && !self.policy_pane.dirty {
            self.policy_pane
                .show(ui, e, self.interactive.policy_frame());
            return Ok(());
        }
        self.policy_pane.failed_context = false;
        if self.policy_pane.edited {
            self.workspace_defaults = None;
        }
        let mut pending = view.status == ResolveStatus::Loading
            || self.workspace_defaults.is_some()
            || (self.policy_pane.config.is_none() && self.loader.pending() > 0);
        let exact = self.snapshot_catalog.is_some();
        let mut current = if let Some(catalog) = &self.snapshot_catalog {
            match catalog.policy_source_reusing(
                run,
                selected.revision,
                self.policy_pane.request.as_ref().map(|r| &r.current),
            ) {
                Ok(source) => source,
                Err(error) => {
                    self.interactive.fail_policy_context(&error);
                    None
                }
            }
        } else {
            self.policy_pane.request.as_ref().map(|r| r.current.clone())
        };
        if exact && let Some(source) = current.take() {
            let identity = source.identity();
            let index =
                run.exact_snapshot_index(&identity.key, identity.revision, &selected.model_role)?;
            if let Some(candidate) =
                source.bind_metadata(run, || self.loader.request(run, index))?
            {
                current = Some(
                    self.policy_pane
                        .request
                        .as_ref()
                        .map(|r| &r.current)
                        .filter(|old| old.same(&candidate))
                        .cloned()
                        .unwrap_or(candidate),
                );
            } else {
                pending = true;
            }
        }
        pending |= self.interactive.snapshot_validation_pending()
            || (exact
                && current.is_none()
                && self.snapshot_catalog.as_ref().is_some_and(|c| c.pending()));
        let mut reference = self
            .policy_pane
            .request
            .as_ref()
            .and_then(|r| r.reference.clone());
        if let Some(config) = &self.policy_pane.config {
            if !exact
                && current.is_none()
                && config.grid_eligibility() == GridEligibility::ContinuousFeedforward
                && let Some(search) = &mut self.policy_pane.current_search
            {
                match search.advance(run, &mut self.loader) {
                    Ok(source) => current = source,
                    Err(error) => {
                        self.policy_pane.status = Some(error.to_string());
                        self.policy_pane.terminal = true;
                    }
                }
                pending |= search.pending();
            }
            if !exact
                && current.is_none()
                && !pending
                && let Some(frame) = &view.record
            {
                current = Some(PolicySource::recorded(run, frame, &selected.model_role)?);
            }
            if self.policy_pane.reference_enabled {
                if reference.is_none()
                    && !self.comparison.policy_uses_controller()
                    && config.grid_eligibility() != GridEligibility::HistoryBound
                    && let (Some(search), Some(reference_run)) = (
                        &mut self.policy_pane.reference_search,
                        self.comparison.reference_run(),
                    )
                {
                    reference = search.advance(reference_run, &mut self.loader)?;
                    pending |= search.pending();
                }
                if reference.is_none() && !pending {
                    let controller = self
                        .comparison
                        .policy_controller(&mut self.loader, Some(&selected.input))?;
                    pending |= controller.status == ResolveStatus::Loading;
                    if let (Some(frame), Some(reference_run)) =
                        (controller.record.as_ref(), self.comparison.reference_run())
                    {
                        reference = Some(PolicySource::recorded_controller(
                            reference_run,
                            frame,
                            self.comparison.reference_role(),
                        )?);
                    }
                }
            }
            if let Some(candidate) = &mut reference
                && let Some(retained) = self
                    .interactive
                    .policy_frame()
                    .and_then(|f| f.grid.reference_source())
                    .filter(|s| s.same(candidate))
            {
                *candidate = retained.clone();
            }
            self.prepare_policy_visits(run, &selected, view.record.as_ref())?;
            pending |= self.policy_pane.visits_enabled
                && self.policy_pane.visits.is_none()
                && !self.policy_pane.visit_search_done;
            let config = self.policy_pane.config.as_ref().unwrap();
            if !pending && let Some(current) = current {
                // Catalog discovery can replace recorded predictions with an
                // actual model at the same learner revision. Revision equality
                // alone must not keep the old prediction source published.
                if self
                    .policy_pane
                    .request
                    .as_ref()
                    .is_some_and(|request| !request.current.same(&current))
                {
                    self.policy_pane.dirty = true;
                    self.policy_pane.ready = false;
                }
                if self.policy_pane.reference_enabled && reference.is_none() {
                    self.policy_pane.status = Some("Reference unavailable".into());
                    self.policy_pane.terminal = true;
                }
                if self.policy_pane.dirty || self.policy_pane.request.is_none() {
                    let pin = if current.identity().recorded
                        || self.policy_pane.input_population.is_some()
                    {
                        // A recorded click retains its genuine row. A new source at a
                        // checkpoint follows global immutable input matching, never row index.
                        if let Some(frame) = &view.record {
                            match &self.policy_pane.pin {
                                Some(PolicyPin::Synthetic(input)) => {
                                    PolicyPin::Synthetic(input.clone())
                                }
                                Some(PolicyPin::Recorded { key, .. }) if *key == frame.key => {
                                    self.policy_pane.pin.clone().unwrap()
                                }
                                _ => PolicyPin::Recorded {
                                    key: frame.key.clone(),
                                    probe_key: frame.probe_key.clone(),
                                    row: frame.row,
                                },
                            }
                        } else {
                            self.policy_pane.pin.clone().unwrap()
                        }
                    } else {
                        PolicyPin::Synthetic(self.policy_pane.pin_input.clone().unwrap())
                    };
                    let request = PolicyRequest::new(
                        current,
                        reference,
                        selected.revision,
                        config.clone(),
                        pin.clone(),
                        self.policy_pane.domain,
                    );
                    let mut request = match request {
                        Ok(request) => request,
                        Err(error) => {
                            self.interactive.retire_session();
                            self.policy_pane.request = None;
                            self.policy_pane.dirty = false;
                            self.policy_pane.status = Some(error.to_string());
                            self.policy_pane.terminal = true;
                            self.policy_pane.ready = false;
                            self.policy_pane.failed_context = true;
                            self.policy_pane.show(ui, e, None);
                            return Ok(());
                        }
                    };
                    if !request.current.identity().recorded {
                        request.input_population = self.policy_pane.input_population.clone();
                    }
                    if self.policy_pane.visits_enabled {
                        request.visits = self.policy_pane.visits.clone();
                    }
                    let admission = if self.policy_pane.automatic_resolution() {
                        self.interactive.automatic_policy_resolution(&mut request)
                    } else {
                        Ok(())
                    }
                    .and_then(|()| {
                        if self.policy_pane.reference_automatic
                            && !prepared::reference_match_fits(&request)?
                        {
                            // Keep the learner visible if a default recorded
                            // reference would exceed the bounded match work.
                            // Explicit reference choices still report errors.
                            request.reference = None;
                            self.policy_pane.reference_enabled = false;
                            self.policy_pane.reference_automatic = false;
                            self.policy_pane
                                .set_display(crate::policy::view::Display::Current);
                            self.policy_pane.reference_search = None;
                            if let Some(key) = &mut self.policy_pane.context_key {
                                key.reference_enabled = false;
                            }
                        }
                        Ok(())
                    });
                    if let Err(error) = admission {
                        self.interactive.fail_policy_context(&error);
                        self.policy_pane.failed_context = true;
                    } else {
                        self.policy_pane.config = Some((*request.slice).clone());
                        self.policy_pane.pin = Some(pin);
                        self.interactive.submit_policy(request.clone())?;
                    }
                    self.policy_pane.request = Some(request);
                    self.policy_pane.dirty = false;
                }
            }
        }
        if let Some(error) = self.interactive.policy_error() {
            self.policy_pane.status = Some(error.into());
            self.policy_pane.terminal = true;
            self.policy_pane.ready = false;
        } else if pending || self.interactive.policy_pending() {
            self.policy_pane.status = Some("Loading…".into());
            self.policy_pane.ready = false;
            self.policy_pane.terminal = false;
        } else if let Some(frame) = self.interactive.policy_frame() {
            if let Some(point) = &frame.selected {
                let previous = self.policy_pane.comparison_available;
                self.policy_pane.comparison_available = point.reference_value.is_some();
                if previous != self.policy_pane.comparison_available {
                    // Compact controls precede worker publication in this paint.
                    ui.ctx().request_repaint();
                }
            }
            self.policy_pane.ready = frame
                .selected
                .as_ref()
                .is_some_and(|p| self.policy_pane.pin_input.as_ref() == Some(&p.input))
                && (!self.policy_pane.reference_enabled
                    || frame
                        .selected
                        .as_ref()
                        .is_some_and(|p| p.reference_value.is_some()));
            if self.policy_pane.domain.is_none() {
                self.policy_pane.domain = frame
                    .selected
                    .as_ref()
                    .and_then(|p| p.plot.as_ref())
                    .map(|p| p.domain);
            }
            self.policy_pane.status = if self.policy_pane.reference_enabled
                && frame
                    .selected
                    .as_ref()
                    .is_some_and(|p| p.reference_value.is_none())
            {
                self.policy_pane.terminal = true;
                Some("Reference unavailable".into())
            } else if frame.grid.regular() {
                None
            } else {
                Some("Recorded samples · surface unavailable".into())
            };
        } else if !pending {
            if !self.policy_pane.terminal {
                self.policy_pane.status = Some(
                    if matches!(selected.input, SelectedInput::None) {
                        "Select a recorded input"
                    } else {
                        "Policy unavailable"
                    }
                    .into(),
                );
            }
            self.policy_pane.terminal = true;
        }
        let intent = self
            .policy_pane
            .show(ui, e, self.interactive.policy_frame());
        if let Some(intent) = intent {
            intent.apply(run, &mut self.selection)?;
        }
        if self.policy_pane.dirty {
            self.policy_pane.ready = false;
            // Clear the old exact point in this same event pass; the map survives.
            if let (Some(mut request), Some(pin)) = (
                self.policy_pane.request.clone(),
                self.policy_pane.pin.clone(),
            ) {
                request.pin = pin;
                request.distribution_domain = self.policy_pane.domain;
                self.interactive.submit_policy(request.clone())?;
                self.policy_pane.request = Some(request);
                self.policy_pane.dirty = false;
            }
        }
        if pending || self.interactive.policy_pending() || self.policy_pane.dirty {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        Ok(())
    }
}

#[cfg(feature = "screenshots")]
impl Laboratory {
    pub(super) fn policy_capture_hover(
        &self,
        ctx: &egui::Context,
        input: &mut egui::RawInput,
        enabled: Option<bool>,
    ) {
        if self.view != View::PolicyMap || !self.policy_pane.ready() {
            return;
        }
        match enabled {
            None => return,
            Some(false) => {
                input.events.push(egui::Event::PointerGone);
                return;
            }
            Some(true) => {}
        }
        if let (Some(config), Some(pin), Some(memory)) = (
            self.policy_pane.config(),
            self.policy_pane.pinned_input(),
            egui_plot::PlotMemory::load(ctx, PolicyPane::main_plot_id()),
        ) && let Some(y) = config.y
        {
            let position = memory
                .transform()
                .position_from_point(&egui_plot::PlotPoint::new(
                    pin.dense_values()[config.x] as f64,
                    pin.dense_values()[y] as f64,
                ));
            input.events.push(egui::Event::PointerMoved(position));
        }
    }
    pub(super) fn policy_capture_ready(&self) -> bool {
        self.policy_capture_ready_for(
            std::env::var("RL_VIEWER_POLICY_EXPECT_UNAVAILABLE")
                .ok()
                .as_deref(),
        )
    }
    fn policy_capture_idle(&self) -> bool {
        !self.interactive.pending()
            && !self.interactive.policy_pending()
            && self.loader.pending() == 0
            && self.workspace_defaults.is_none()
    }
    pub(super) fn policy_capture_ready_for(&self, expected_unavailable: Option<&str>) -> bool {
        if self.policy_execution.mode == Some(PolicyMode::Prepared) {
            return self.policy_pane.ready
                && self
                    .policy_execution
                    .provider
                    .as_ref()
                    .is_some_and(|p| p.ready() && !p.pending())
                && self.policy_frame().is_some_and(|f| {
                    Some(f.grid.current.key.sequence) == self.policy_execution.sequence
                });
        }
        if let Some(expected) = expected_unavailable {
            return self.view == View::PolicyMap
                && !expected.trim().is_empty()
                && self.policy_pane.terminal
                && !self.policy_pane.ready
                && self.policy_capture_idle()
                && self.policy_pane.status.as_deref() == Some(expected);
        }
        if !self.policy_pane.ready || self.interactive.policy_pending() {
            return false;
        }
        let (Some(request), Some(frame)) =
            (&self.policy_pane.request, self.interactive.policy_frame())
        else {
            return false;
        };
        request.requested_revision == self.selection.revision
            && frame.requested_revision == self.selection.revision
            && frame.grid.matches_request(request)
            && frame.selected.as_ref().is_some_and(|point| {
                self.policy_pane.pin_input.as_ref() == Some(&point.input)
                    && (!self.policy_pane.reference_enabled
                        || (request.reference.is_some() && point.reference_value.is_some()))
            })
    }
    fn configure_policy_capture(&mut self, e: EnvironmentSemantics<'_>) -> Result<()> {
        use crate::policy::{slice::SliceMode, view::Display};
        let invalid = || rl::Error::Config("Invalid Policy capture configuration".into());
        let index = |name: &str| -> Result<Option<usize>> {
            std::env::var(name)
                .ok()
                .map(|s| s.parse().map_err(|_| invalid()))
                .transpose()
        };
        let c = self.policy_pane.config.as_mut().unwrap();
        let x = index("RL_VIEWER_POLICY_X")?.unwrap_or(c.x);
        let mode = std::env::var("RL_VIEWER_POLICY_MODE").ok();
        let two = match mode.as_deref() {
            Some("1d") => false,
            Some("2d") => true,
            None => c.mode == SliceMode::TwoDimensional,
            _ => return Err(invalid()),
        };
        let y = if two {
            Some(index("RL_VIEWER_POLICY_Y")?.or(c.y).ok_or_else(invalid)?)
        } else {
            None
        };
        if c.mode != SliceMode::SamplesOnly
            || mode.is_some()
            || index("RL_VIEWER_POLICY_X")?.is_some()
            || index("RL_VIEWER_POLICY_Y")?.is_some()
        {
            c.set_axes(e, x, y)?;
            if !two {
                c.resolution = [256, 1];
            }
        }
        if let Some(action) = index("RL_VIEWER_POLICY_ACTION")? {
            c.action = action;
        }
        if let Ok(fixed) = std::env::var("RL_VIEWER_POLICY_FIXED") {
            for field in fixed.split(',') {
                let (i, value) = field.split_once('=').ok_or_else(invalid)?;
                let i: usize = i.parse().map_err(|_| invalid())?;
                *c.fixed.get_mut(i).ok_or_else(invalid)? = value.parse().map_err(|_| invalid())?;
            }
        }
        for (name, axis) in [
            ("RL_VIEWER_POLICY_X_RANGE", Some(x)),
            ("RL_VIEWER_POLICY_Y_RANGE", y),
        ] {
            if let (Ok(range), Some(axis)) = (std::env::var(name), axis) {
                let (lo, hi) = range.split_once(',').ok_or_else(invalid)?;
                c.ranges[axis] = [
                    lo.parse().map_err(|_| invalid())?,
                    hi.parse().map_err(|_| invalid())?,
                ];
            }
        }
        if let Ok(resolution) = std::env::var("RL_VIEWER_POLICY_RESOLUTION") {
            let values: Vec<usize> = resolution
                .split(',')
                .map(|v| v.parse().map_err(|_| invalid()))
                .collect::<Result<_>>()?;
            self.policy_pane.resolution_explicit[usize::from(two)] = true;
            c.resolution = [
                *values.first().ok_or_else(invalid)?,
                if two {
                    *values.get(1).ok_or_else(invalid)?
                } else {
                    1
                },
            ];
        }
        if let Ok(tolerance) = std::env::var("RL_VIEWER_POLICY_TOLERANCE") {
            c.tolerance = tolerance.parse().map_err(|_| invalid())?;
        }
        c.validate(e)?;
        if let Some(input) = &mut self.policy_pane.pin_input {
            input
                .as_dense_mut()
                .expect("dense diagnostic input")
                .values
                .clone_from(&c.fixed);
        }
        if let Ok(reference) = std::env::var("RL_VIEWER_POLICY_REFERENCE") {
            self.policy_pane.reference_enabled = reference == "true";
            self.policy_pane.reference_automatic = false;
        }
        self.policy_pane.visits_enabled =
            std::env::var("RL_VIEWER_POLICY_VISITS").is_ok_and(|v| v == "true");
        self.policy_pane.display = match std::env::var("RL_VIEWER_POLICY_DISPLAY").as_deref() {
            Ok("reference") => Display::Reference,
            Ok("difference") => Display::Difference,
            Ok("current") | Err(_) => Display::Current,
            _ => return Err(invalid()),
        };
        let revision = std::env::var("RL_VIEWER_POLICY_REFERENCE_K")
            .ok()
            .map(|v| v.parse().map_err(|_| invalid()))
            .transpose()?;
        let path = std::env::var_os("RL_VIEWER_POLICY_REFERENCE_RUN").map(std::path::PathBuf::from);
        let controller =
            std::env::var("RL_VIEWER_POLICY_REFERENCE_SOURCE").is_ok_and(|v| v == "controller");
        let role = std::env::var("RL_VIEWER_POLICY_REFERENCE_ROLE").ok();
        if self.policy_pane.reference_enabled
            && (path.is_some() || revision.is_some() || controller || role.is_some())
        {
            self.comparison.configure_policy_reference(
                path,
                revision,
                controller,
                role.as_deref(),
            )?;
        }
        self.policy_pane.capture_configured = true;
        self.policy_pane.dirty = true;
        self.policy_pane.edited = std::env::vars_os()
            .any(|(key, _)| key.to_string_lossy().starts_with("RL_VIEWER_POLICY_"));
        Ok(())
    }
    pub(super) fn policy_capture_error(&self) -> Option<String> {
        self.policy_capture_problem(
            std::env::var("RL_VIEWER_POLICY_EXPECT").ok().as_deref(),
            std::env::var("RL_VIEWER_POLICY_EXPECT_UNAVAILABLE")
                .ok()
                .as_deref(),
        )
    }
    pub(super) fn policy_capture_problem(
        &self,
        expected: Option<&str>,
        expected_unavailable: Option<&str>,
    ) -> Option<String> {
        if self.view != View::PolicyMap {
            return None;
        }
        if expected_unavailable.is_some_and(|s| s.trim().is_empty()) {
            return Some("Expected unavailable Policy status must be nonempty".into());
        }
        if self.policy_pane.terminal && !self.policy_pane.ready && self.policy_capture_idle() {
            if self.policy_capture_ready_for(expected_unavailable) {
                return None;
            }
            return Some(
                self.policy_pane
                    .status
                    .clone()
                    .unwrap_or_else(|| "Policy unavailable".into()),
            );
        }
        if expected_unavailable.is_some() {
            return (self.policy_pane.ready && self.policy_capture_idle())
                .then(|| "Expected unavailable Policy status was not observed".into());
        }
        if !self.policy_pane.ready {
            return None;
        }
        let Some(frame) = self.policy_frame() else {
            return Some("No matching Policy frame".into());
        };
        match expected {
            Some("grid") if !frame.grid.regular() => {
                Some("Required Policy grid is unavailable; only recorded samples exist".into())
            }
            Some("samples") if frame.grid.regular() => {
                Some("Expected recorded Policy samples".into())
            }
            Some("distribution")
                if frame.selected.as_ref().is_none_or(|p| {
                    matches!(
                        p.distribution,
                        crate::policy::numerics::SelectedDistribution::Deterministic
                    )
                }) =>
            {
                Some("Required stochastic Policy distribution is unavailable".into())
            }
            Some("grid" | "samples" | "distribution") | None => None,
            _ => Some("Invalid Policy capture expectation".into()),
        }
    }
}

impl Laboratory {
    fn prepare_policy_visits(
        &mut self,
        run: &RunIndex,
        selected: &Selection,
        frame: Option<&ResolvedRecord>,
    ) -> Result<()> {
        // A retained population has stable ownership across checkpoint navigation.
        // Only unfinished/exhausted discovery follows the current source context.
        if self.policy_pane.visits.is_some() {
            return Ok(());
        }
        let unchanged = self.policy_pane.visit_search_context.as_ref().is_some_and(
            |(root, role, revision, entries)| {
                root == run.root()
                    && role == &selected.model_role
                    && *revision == selected.revision
                    && *entries == run.recording().entries().len()
            },
        );
        if !unchanged {
            self.policy_pane.visit_search_context = Some((
                run.root().clone(),
                selected.model_role.clone(),
                selected.revision,
                run.recording().entries().len(),
            ));
            self.policy_pane.visit_cursor = None;
            self.policy_pane.visit_loading = None;
            self.policy_pane.visit_checked = 0;
            self.policy_pane.visit_search_done = false;
        }
        if self.policy_pane.visit_search_done {
            return Ok(());
        }
        if let Some(frame) = frame
            && matches!(&frame.probes.record,Record::ProbeSet(p) if p.population==ProbePopulation::PolicyVisited)
        {
            self.policy_pane.visits = Some(VisitSource {
                key: frame.probe_key.clone(),
                chunk: frame.probes.clone(),
            });
            return Ok(());
        }
        if let Some(index) = run
            .candidates(
                RecordKind::Trajectory,
                &selected.model_role,
                None,
                selected.revision,
            )
            .next()
        {
            if let Some(chunk) = self.loader.request(run, index)? {
                let key = run.key(index)?;
                self.policy_pane.visits = Some(VisitSource {
                    chunk: self.resolver.lease_visitation(run, &key, chunk)?,
                    key,
                });
            }
            return Ok(());
        }
        for _ in 0..8 {
            let index = if let Some(index) = self.policy_pane.visit_loading {
                index
            } else {
                if self.policy_pane.visit_checked >= 256 {
                    self.policy_pane.visit_search_done = true;
                    return Ok(());
                }
                let next = if let Some(before) = self.policy_pane.visit_cursor {
                    run.candidates_before(RecordKind::ProbeSet, &selected.model_role, None, before)
                        .next()
                } else {
                    run.candidates(
                        RecordKind::ProbeSet,
                        &selected.model_role,
                        None,
                        selected.revision,
                    )
                    .next()
                };
                let Some(index) = next else {
                    self.policy_pane.visit_search_done = true;
                    return Ok(());
                };
                self.policy_pane.visit_checked += 1;
                self.policy_pane.visit_loading = Some(index);
                index
            };
            let Some(chunk) = self.loader.request(run, index)? else {
                return Ok(());
            };
            self.policy_pane.visit_loading = None;
            let entry = &run.recording().entries()[index];
            self.policy_pane.visit_cursor = Some((entry.revision, entry.sequence));
            if matches!(&chunk.record,Record::ProbeSet(p) if p.population==ProbePopulation::PolicyVisited)
            {
                let key = run.key(index)?;
                self.policy_pane.visits = Some(VisitSource {
                    chunk: self.resolver.lease_visitation(run, &key, chunk)?,
                    key,
                });
                return Ok(());
            }
        }
        Ok(())
    }
}

impl Laboratory {
    /// The only frame commit in a native paint occurs before its transport header.
    pub(super) fn snapshot_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        if self.view == View::Training {
            return self.training_transport(ui, run);
        }
        if typed_workspace::typed_view(run, self.view, &self.selection) {
            return self.typed_transport(ui, run);
        }
        if self.view == View::PolicyMap && self.policy_evolution {
            return self.policy_evolution_transport(ui, run);
        }
        if self.view == View::PolicyMap {
            self.ensure_policy_execution(ui.ctx(), run);
            if self.policy_execution.mode != Some(PolicyMode::Interactive) {
                return self.prepared_policy_transport(ui, run);
            }
        }
        if self.view == View::Environment {
            return self.environment_transport(ui, run);
        }
        if self.view == View::Dynamics {
            return self.dynamics_transport(ui, run);
        }
        if self.view == View::Network {
            return self.network_snapshot_transport(ui, run);
        }
        use crate::snapshot_catalog::{SnapshotCatalog, SnapshotPanel};
        // The workspace discovers its anchor and slice after the toolbar. Until
        // then neither exact-model nor recorded-only navigation is established.
        if (self.view == View::PolicyMap && self.policy_pane.config.is_none())
            || (self.view == View::Value && self.value_view.config.is_none())
        {
            self.transport.show_snapshots(
                ui,
                run,
                &mut self.selection,
                crate::timeline_ui::SnapshotPresentation {
                    positions: &[],
                    published: None,
                    ready: false,
                    checking: true,
                    error: None,
                },
            )?;
            return Ok(());
        }
        let decision_loss = self.view == View::Value && self.value_view.decision_loss_requested();
        let panel = match self.view {
            View::Value if decision_loss => Some(SnapshotPanel::Policy),
            View::PolicyMap
                if self.policy_pane.config.as_ref().is_some_and(|c| {
                    c.grid_eligibility() == GridEligibility::ContinuousFeedforward
                        || (matches!(
                            c.grid_eligibility(),
                            GridEligibility::Masked | GridEligibility::UnsupportedDiscrete
                        ) && self.policy_pane.input_population.is_some()
                            && !run
                                .snapshot_revisions(&self.selection.model_role)
                                .is_empty())
                }) =>
            {
                Some(SnapshotPanel::Policy)
            }
            View::Value
                if self
                    .value_view
                    .config
                    .as_ref()
                    .is_some_and(|c| c.model_eligible()) =>
            {
                Some(SnapshotPanel::Value {
                    name: self.value_view.choice.as_ref().map(|c| c.name.clone()),
                    twin: self.value_view.choice.as_ref().and_then(|c| c.twin.clone()),
                    output: self
                        .value_view
                        .output
                        .clone()
                        .unwrap_or(rl::diagnostics::value::ValueOutput::ExpectedReturn),
                    actor_required: self.value_view.config.as_ref().is_some_and(|c| {
                        matches!(
                            c.conditioning,
                            crate::value::slice::ActionConditioning::Policy
                        )
                    }),
                })
            }
            _ => None,
        };
        let Some(panel) = panel.filter(|_| {
            !run.snapshot_revisions(&self.selection.model_role)
                .is_empty()
        }) else {
            self.interactive.cancel_snapshot_validation();
            self.snapshot_catalog = None;
            if self.view == View::PolicyMap {
                self.interactive.poll_policy()?;
            }
            if self.view == View::Value {
                self.interactive.poll_value()?;
            }
            let kind = if self.view == View::PolicyMap {
                RecordKind::Policy
            } else {
                RecordKind::Critic
            };
            let mut positions: Vec<_> = run
                .candidates(kind, &self.selection.model_role, None, u64::MAX)
                .take(256)
                .map(|i| run.recording().entries()[i].revision)
                .collect();
            positions.sort_unstable();
            positions.dedup();
            let published = if self.view == View::PolicyMap {
                self.policy_frame().map(|f| f.grid.current.revision)
            } else {
                self.value_frame()
                    .map(|f| f.grid.current().identity().revision)
            };
            let ready = if self.view == View::PolicyMap {
                !self.policy_pending()
            } else {
                !self.value_pending()
            };
            self.transport.show_snapshots(
                ui,
                run,
                &mut self.selection,
                crate::timeline_ui::SnapshotPresentation {
                    positions: &positions,
                    published,
                    ready,
                    checking: false,
                    error: None,
                },
            )?;
            return Ok(());
        };
        if self
            .snapshot_catalog
            .as_ref()
            .is_none_or(|c| !c.matches(run, &self.selection.model_role, &panel))
        {
            self.interactive.cancel_snapshot_validation();
            let reused = self.snapshot_catalog.as_mut().is_some_and(|catalog| {
                catalog.retarget(run, &self.selection.model_role, panel.clone())
            });
            if !reused {
                // Release the prior lease before admitting a new catalogue.
                self.snapshot_catalog = None;
                self.snapshot_catalog = Some(SnapshotCatalog::new(
                    run,
                    &self.selection.model_role,
                    panel,
                    &self.interactive,
                )?);
            }
        }
        if self.view == View::PolicyMap {
            self.interactive.poll_policy()?;
        } else {
            self.interactive.poll_value()?;
        }
        let foreground = if self.view == View::PolicyMap {
            self.policy_frame()
                .is_some_and(|f| f.grid.current.revision != self.selection.revision)
                || self.policy_pane.dirty
        } else {
            self.value_frame().is_some_and(|f| {
                if decision_loss {
                    f.selected
                        .as_ref()
                        .and_then(|p| p.current_policy())
                        .is_none_or(|p| p.identity().revision != self.selection.revision)
                } else {
                    f.grid.current().identity().revision != self.selection.revision
                }
            }) || self.value_view.dirty
        };
        let foreground_ready = !foreground
            && if self.view == View::PolicyMap {
                self.policy_pane.ready()
                    && self
                        .policy_frame()
                        .is_some_and(|f| f.requested_revision == self.selection.revision)
            } else {
                self.value_view.ready()
                    && self
                        .value_frame()
                        .is_some_and(|f| f.requested_revision == self.selection.revision)
            };
        let catalog = self.snapshot_catalog.as_mut().unwrap();
        if self.view == View::PolicyMap
            && let Some(source) = &self.policy_execution.interactive_source
        {
            catalog.prefer_policy_source(run, source)?;
        }
        let priority = run
            .snapshot_revisions(&self.selection.model_role)
            .iter()
            .rev()
            .find(|&&k| k <= self.selection.revision)
            .copied()
            .unwrap_or(self.selection.revision);
        catalog.prioritize_revision(priority);
        if foreground && catalog.selected(self.selection.revision).is_some() {
            catalog.yield_to_foreground(&mut self.interactive);
        } else {
            catalog.advance_with_followup(
                run,
                &mut self.loader,
                &mut self.interactive,
                foreground_ready,
            )?;
        }
        let catalog = self.snapshot_catalog.as_ref().unwrap();
        let positions = catalog.positions();
        if catalog.limited() {
            ui.weak("Snapshot catalog limited to 256 candidates; remaining snapshots unchecked");
        }
        let (published, ready, error) = if self.view == View::PolicyMap {
            (
                self.interactive
                    .policy_frame()
                    .filter(|f| {
                        catalog
                            .selected(f.grid.current.revision)
                            .is_some_and(|e| e.key == f.grid.current.key)
                    })
                    .map(|f| f.grid.current.revision),
                !self.interactive.policy_pending(),
                self.interactive.policy_error(),
            )
        } else if decision_loss {
            (
                self.interactive
                    .value_frame()
                    .and_then(|f| f.selected.as_ref())
                    .and_then(|p| p.current_policy())
                    .filter(|p| {
                        catalog
                            .selected(p.identity().revision)
                            .is_some_and(|e| e.key == p.identity().key)
                    })
                    .map(|p| p.identity().revision),
                !self.interactive.value_pending(),
                self.interactive.value_error(),
            )
        } else {
            (
                self.interactive
                    .value_frame()
                    .filter(|f| {
                        catalog
                            .selected(f.grid.current().identity().revision)
                            .is_some_and(|e| e.key == f.grid.current().identity().key)
                    })
                    .map(|f| f.grid.current().identity().revision),
                !self.interactive.value_pending(),
                self.interactive.value_error(),
            )
        };
        let retry = self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            crate::timeline_ui::SnapshotPresentation {
                positions: &positions,
                published,
                ready: ready && !self.interactive.snapshot_validation_pending(),
                checking: catalog.pending(),
                error,
            },
        )?;
        if retry {
            self.snapshot_catalog.as_mut().unwrap().retry_rejected();
            if self.view == View::PolicyMap {
                self.interactive.invalidate_policy_context();
                self.policy_pane.request = None;
                self.policy_pane.dirty = true;
            } else {
                self.interactive.invalidate_value_context();
                self.value_view.request = None;
                self.value_view.dirty = true;
            }
        }
        // Model-only speculation starts after all discovery and foreground work.
        // It neither advances this transport nor changes a complete publication.
        let catalog = self.snapshot_catalog.as_ref().unwrap();
        if !retry
            && !foreground
            && !catalog.pending()
            && ready
            && {
                #[cfg(feature = "screenshots")]
                {
                    std::env::var("RL_VIEWER_MODEL_WARMUP").as_deref() != Ok("false")
                }
                #[cfg(not(feature = "screenshots"))]
                {
                    true
                }
            }
            && published == Some(self.selection.revision)
            && let Some(next) = self.transport.next_snapshot()
            && let Some(entry) = catalog.selected(next)
        {
            let mut protected = Vec::with_capacity(3);
            if self.view == View::PolicyMap {
                if let Some(frame) = self.interactive.policy_frame() {
                    protected.push(frame.grid.current.key.clone());
                    if let Some(reference) = &frame.grid.reference
                        && !reference.recorded
                    {
                        protected.push(reference.key.clone());
                    }
                }
            } else if let Some(frame) = self.interactive.value_frame() {
                if decision_loss {
                    if let Some(policy) = frame.selected.as_ref().and_then(|p| p.current_policy()) {
                        protected.push(policy.identity().key.clone());
                    }
                } else {
                    protected.push(frame.grid.current().identity().key.clone());
                }
                if let Some(reference) = frame.grid.reference()
                    && !reference.identity().recorded
                {
                    protected.push(reference.identity().key.clone());
                }
                if let Some(reference) = frame.grid.reference_policy()
                    && !reference.identity().recorded
                {
                    protected.push(reference.identity().key.clone());
                }
            }
            if let Ok(index) = run.exact_snapshot_index(&entry.key, entry.revision, &entry.role) {
                if self.view == View::PolicyMap
                    && let Some(base) = self.policy_pane.request.as_ref()
                    && base.visits.is_none()
                    && base.input_population.is_none()
                    && matches!(base.pin, PolicyPin::Synthetic(_))
                    && let Some(source) = catalog.policy_source(run, next).ok().flatten()
                    && let Some(source) = source
                        .bind_metadata(run, || self.loader.request(run, index))
                        .ok()
                        .flatten()
                {
                    let mut request = base.clone();
                    request.current = source;
                    request.requested_revision = next;
                    if self.interactive.policy_prefetch_ready(&request)
                        || self.interactive.prefetch_policy(request).unwrap_or(false)
                    {
                        return Ok(());
                    }
                }
                self.interactive.warm_snapshot(run, index, protected)?;
            }
        }
        Ok(())
    }
}
