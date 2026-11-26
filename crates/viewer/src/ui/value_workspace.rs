//! Value route orchestration through existing immutable selection, Loader, resolver and worker.
use super::*;
use crate::{
    policy::{
        prepared::{ModelSearch, PolicySource, VisitSource},
        slice::SliceMode,
        view::{PolicyContextKey, PolicyPane},
    },
    value::{
        prepared::*,
        slice::*,
        view::{CriticChoice, ValueView},
    },
};
use rl::{
    core::action::ActionSpace,
    diagnostics::{models::ModelKind, value::*, visualization::EnvironmentSemantics},
};
fn choices(chunk: &Chunk) -> Result<(Vec<CriticChoice>, Vec<String>)> {
    let config = ValueSource::configuration(chunk)?;
    let declaration = declaration(chunk)?;
    let mut choices = Vec::new();
    let mut policies = Vec::new();
    for (name, m) in &config.models {
        let Some(kind) = m.kind.critic_kind() else {
            if matches!(
                m.kind,
                ModelKind::DeterministicPolicy
                    | ModelKind::StochasticPolicy
                    | ModelKind::StateGaussianPolicy
            ) {
                policies.push(name.clone());
            }
            continue;
        };
        let twin = declaration
            .as_ref()
            .and_then(|d| d.critics.get(name))
            .and_then(|c| c.twin.as_ref());
        if let Some(t) = twin {
            if t.index == 1 {
                continue;
            }
            let complete = declaration.as_ref().is_some_and(|d| {
                d.critics.iter().any(|(other, c)| {
                    c.twin
                        .as_ref()
                        .is_some_and(|u| u.group == t.group && u.index == 1)
                        && config
                            .models
                            .get(other)
                            .is_some_and(|m| m.kind.critic_kind() == Some(ValueCriticKind::ScalarQ))
                })
            });
            if complete {
                choices.push(CriticChoice {
                    name: name.clone(),
                    twin: Some(t.group.clone()),
                    kind,
                });
                continue;
            }
        }
        choices.push(CriticChoice {
            name: name.clone(),
            twin: None,
            kind,
        });
    }
    Ok((choices, policies))
}
impl Laboratory {
    pub fn value_view(&self) -> &ValueView {
        &self.value_view
    }
    pub fn value_view_mut(&mut self) -> &mut ValueView {
        &mut self.value_view
    }
    pub fn value_frame(&self) -> Option<&PreparedValueFrame> {
        self.interactive.value_frame()
    }
    pub fn value_pending(&self) -> bool {
        self.interactive.value_pending()
    }
    pub fn value_metrics(&self) -> ValueMetrics {
        self.interactive.value_metrics()
    }
    pub(super) fn value_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let e = EnvironmentSemantics::new(run.recording().manifest());
        if self.value_view.reservation.is_none() {
            let bound = PolicyPane::reservation_bound(e)?.saturating_add(768 * 1024);
            match self.interactive.reserve_policy_view(bound) {
                Ok(lease) => self.value_view.reservation = Some(lease),
                Err(err) => {
                    self.value_view.status = Some(err.to_string());
                    self.value_view.terminal = true;
                    ui.heading("Value");
                    self.value_view.show(ui, e, None);
                    return Ok(());
                }
            }
        }
        let selected = self.selection.clone();
        let source = self.selected_source(run)?;
        if let Some((chunk, row)) = &source {
            let key = match &selected.input {
                SelectedInput::ProbeRow { sequence, .. }
                | SelectedInput::TrajectoryStep { sequence, .. } => Some((
                    CacheKey {
                        root: run.root().clone(),
                        sequence: *sequence,
                    },
                    *row,
                )),
                _ => None,
            };
            if self.value_view.anchor_key != key {
                let input = source_observation(chunk, *row)?;
                if self.value_view.config.is_none() {
                    let mut c = ValueSlice::new(e, &input, &[])?;
                    let rows = crate::value::visitation::input_rows(chunk)?;
                    c.adopt_recorded_ranges((0..rows).map(|row| source_observation(chunk, row)))?;
                    if matches!(&chunk.record,Record::ProbeSet(p)if !p.histories.is_empty()) {
                        c.state = c.state.with_history();
                    }
                    if c.mode != SliceMode::SamplesOnly
                        && matches!(e.action_space(), ActionSpace::Discrete { .. })
                    {
                        c.mode = SliceMode::OneDimensional;
                        c.x = ValueCoordinate::DiscreteAction;
                        c.y = None;
                        c.resolution = [256, 1];
                    }
                    self.value_view.config = Some(c);
                } else if let Some(c) = &mut self.value_view.config {
                    let mut state = crate::policy::slice::SliceConfig::new(e, &input, &[])?;
                    if matches!(&chunk.record,Record::ProbeSet(p)if !p.histories.is_empty()) {
                        state = state.with_history();
                    }
                    state.ranges.clone_from(&c.state.ranges);
                    state.range_sources.clone_from(&c.state.range_sources);
                    state.tolerance = c.state.tolerance;
                    c.state = state;
                    if let ActionConditioning::Fixed(a) = &c.conditioning
                        && e.action_space().check(a, input.dense_mask()).is_err()
                    {
                        c.conditioning = ActionConditioning::Policy;
                    }
                }
                // A global input event chooses its exact immutable row. Checkpoint-only changes preserve pins.
                let key = key.unwrap();
                let visit = VisitSource {
                    key: key.0.clone(),
                    chunk: self.resolver.lease_record(run, &key.0, chunk.clone())?,
                };
                self.value_view.pin = Some(if matches!(chunk.record, Record::Trajectory(_)) {
                    ValuePin::Recorded {
                        source: visit,
                        row: *row,
                    }
                } else {
                    let action = match &chunk.record {
                        Record::ProbeSet(probes) => probes
                            .fixed_actions
                            .as_ref()
                            .map(|actions| {
                                crate::value::visitation::action_row(
                                    chunk,
                                    actions,
                                    *row,
                                    e.action_space(),
                                )
                            })
                            .transpose()?,
                        _ => None,
                    };
                    ValuePin::Probe {
                        source: visit,
                        row: *row,
                        action,
                    }
                });
                self.value_view.empirical_condition = None;
                self.value_view.anchor_key = Some(key);
                self.value_view.dirty = true;
            }
        }
        let choice_before_metadata = self.value_view.choice.clone();
        let decision_loss = self.value_view.decision_loss_requested();
        let metadata_models = !decision_loss
            && self
                .value_view
                .config
                .as_ref()
                .is_none_or(|c| c.model_eligible());
        if self.value_view.metadata_models != Some(metadata_models) {
            self.value_view.metadata_key = None;
            self.value_view.probe_metadata_key = None;
            self.value_view.probe_metadata_cursor = 0;
            self.value_view.metadata_models = Some(metadata_models);
        }
        let mut metadata_pending = false;
        if let Some(index) = metadata_models
            .then(|| {
                if let Some(catalog) = &self.snapshot_catalog {
                    catalog
                        .selected(selected.revision)
                        .and_then(|e| run.by_sequence(e.key.sequence))
                        .or_else(|| {
                            // Initial critic names are metadata choices, never
                            // eligible playback positions. A multi-critic model
                            // cannot be validated for the selected critic until
                            // those choices have been discovered.
                            self.value_view
                                .metadata_key
                                .is_none()
                                .then(|| {
                                    run.candidates(
                                        RecordKind::ModelSnapshot,
                                        &selected.model_role,
                                        None,
                                        selected.revision,
                                    )
                                    .next()
                                })
                                .flatten()
                        })
                } else {
                    run.candidates(
                        RecordKind::ModelSnapshot,
                        &selected.model_role,
                        None,
                        selected.revision,
                    )
                    .next()
                }
            })
            .flatten()
        {
            let key = run.key(index)?;
            if self.value_view.metadata_key.as_ref() != Some(&key) {
                if let Some(chunk) = self.loader.request(run, index)? {
                    let (c, _) = choices(&chunk)?;
                    let previous = self.value_view.choice.clone();
                    self.value_view.choices = c;
                    self.value_view.choice = previous
                        .and_then(|p| {
                            self.value_view
                                .choices
                                .iter()
                                .find(|c| c.name == p.name && c.twin == p.twin)
                                .cloned()
                        })
                        .or_else(|| {
                            (self.value_view.choices.len() == 1)
                                .then(|| self.value_view.choices[0].clone())
                        });
                    if self.value_view.output.is_none() {
                        self.value_view.output = Some(
                            if self
                                .value_view
                                .choice
                                .as_ref()
                                .is_some_and(|c| c.twin.is_some())
                            {
                                ValueOutput::Q1
                            } else {
                                ValueOutput::ExpectedReturn
                            },
                        );
                    }
                    self.value_view.metadata_key = Some(key);
                } else {
                    metadata_pending = true;
                }
            }
        } else if metadata_models && self.snapshot_catalog.is_some() {
            // An unchecked exact source is not permission to select a recorded
            // critic. Preserve the requested model choice while validation runs.
            metadata_pending = self.snapshot_catalog.as_ref().is_some_and(|c| c.pending());
        } else {
            // Bound legacy discovery just as model metadata discovery is bounded.
            // Each unnamed record remains an explicit immutable choice.
            let indices = run
                .candidates(
                    RecordKind::Critic,
                    &selected.model_role,
                    None,
                    selected.revision,
                )
                .take(crate::policy::prepared::MAX_SOURCE_CANDIDATES)
                .collect::<Vec<_>>();
            let newest = indices.first().map(|&i| run.key(i)).transpose()?;
            if self.value_view.metadata_key != newest {
                if self.value_view.probe_metadata_key != newest {
                    self.value_view.probe_metadata_key = newest.clone();
                    self.value_view.probe_metadata_cursor = 0;
                    self.value_view.choices.clear();
                }
                // Retain only small labels as each candidate finishes. A population
                // larger than Loader's cache must not cause perpetual reloads.
                while let Some(&index) = indices.get(self.value_view.probe_metadata_cursor) {
                    if run.probe_for_record(index).is_none() {
                        self.value_view.probe_metadata_cursor += 1;
                        continue;
                    }
                    if let Some(chunk) = self.loader.request(run, index)? {
                        if let Record::Critic(c) = &chunk.record {
                            let name =
                                ValueSource::recorded_name(&chunk, run.key(index)?.sequence)?;
                            if !self.value_view.choices.iter().any(|c| c.name == name) {
                                self.value_view.choices.push(CriticChoice {
                                    name,
                                    twin: None,
                                    kind: ValueSource::recorded_kind(c),
                                });
                            }
                        }
                        self.value_view.probe_metadata_cursor += 1;
                    } else {
                        metadata_pending = true;
                        break;
                    }
                }
                if !metadata_pending {
                    self.value_view.choice = self
                        .value_view
                        .choice
                        .as_ref()
                        .and_then(|old| {
                            self.value_view
                                .choices
                                .iter()
                                .find(|c| c.name == old.name)
                                .cloned()
                        })
                        .or_else(|| self.value_view.choices.first().cloned());
                    self.value_view.metadata_key = newest;
                }
            }
        }
        #[cfg(feature = "screenshots")]
        self.configure_value_capture(e)?;
        let before = (
            self.value_view.reference_enabled,
            self.value_view.visits_enabled,
            self.value_view.choice.clone(),
        );
        let mut reference_sources = None;
        let mut reference_policy_names = Vec::new();
        if self.value_view.reference_enabled && !self.comparison.value_empirical() {
            let candidate = self.comparison.reference_run().and_then(|r| {
                r.candidates(
                    RecordKind::ModelSnapshot,
                    self.comparison.reference_role(),
                    None,
                    self.comparison.reference_revision(),
                )
                .next()
                .map(|i| (r, i))
            });
            let candidate_key = candidate.map(|(r, i)| r.key(i)).transpose()?;
            if candidate_key != self.value_view.reference_metadata_key {
                self.value_view.reference_choices.clear();
                self.value_view.reference_policies.clear();
                self.value_view.reference_metadata_key = None;
            }
            if let Some((r, i)) = candidate {
                let key = r.key(i)?;
                if let Some(chunk) = self.loader.request(r, i)? {
                    if self.value_view.reference_metadata_key.as_ref() != Some(&key) {
                        let (critics, policies) = choices(&chunk)?;
                        self.value_view.reference_choices = critics;
                        self.value_view.reference_policies = policies;
                        self.value_view.reference_metadata_key = Some(key);
                    }
                    reference_policy_names.clone_from(&self.value_view.reference_policies);
                } else {
                    metadata_pending = true;
                }
            }
            let names = self
                .value_view
                .reference_choices
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>();
            reference_sources = Some(names);
        }
        if self.value_view.reference_enabled && self.comparison.value_empirical() {
            reference_sources = Some(Vec::new());
        }
        let reference_generation = self.comparison.reference_interaction_serial();
        let quality = self.value_view.request.as_ref().map(|request| {
            (
                request.reference_empirical,
                request
                    .reference_value
                    .as_ref()
                    .map(|s| (s.reference_info.clone(), s.identity().clone())),
                request
                    .reference_policy
                    .as_ref()
                    .map(|s| (s.reference_info.clone(), s.identity().clone())),
            )
        });
        let quality_generation = self
            .value_view
            .context_key
            .as_ref()
            .map(|k| k.reference_serial);
        let comparison = &mut self.comparison;
        let selection = &self.selection;
        let loader = &mut self.loader;
        self.value_view.controls(ui, e, |ui| {
            if let Some(names) = &reference_sources {
                crate::landscape::popover(ui, "Compare value", |ui| {
                    comparison.value_source_controls(ui, names, &reference_policy_names);
                    super::reference_measurements::show(ui, run, selection, loader, true);
                    if let Some((empirical, value, policy)) = &quality {
                        if *empirical {
                            ui.small("Conditional empirical reference");
                        } else if let Some((info, identity)) =
                            value.as_ref().filter(|(_, identity)| {
                                comparison.quality_source_is_current(quality_generation, identity)
                            })
                        {
                            info.ui(ui, &identity.name, identity.recorded, true);
                        } else {
                            ui.small(crate::reference::quality_label(None));
                        }
                        if let Some((info, identity)) = policy.as_ref().filter(|(_, identity)| {
                            comparison.quality_source_is_current(quality_generation, identity)
                        }) {
                            ui.horizontal(|ui| {
                                ui.small("Policy:");
                                info.ui(ui, &identity.name, identity.recorded, false);
                            });
                        }
                    } else {
                        ui.small(crate::reference::quality_label(None));
                    }
                });
            } else if !before.0 {
                super::reference_measurements::show(ui, run, selection, loader, true);
            }
        });
        if decision_loss != self.value_view.decision_loss_requested() {
            // The transport header on the next paint must select the matching
            // snapshot catalog before this new output resolves its policy.
            ui.ctx().request_repaint();
            return Ok(());
        }
        if !before.0 && self.value_view.reference_enabled {
            // Reference metadata is discovered before the toolbar. Do not
            // publish a learner-only frame as a completed reference request.
            metadata_pending = true;
            ui.ctx().request_repaint();
        }
        if reference_generation != self.comparison.reference_interaction_serial() {
            // Controls run after metadata discovery. Defer source search until
            // the next paint has discovered the newly selected reference.
            self.value_view.reference_choices.clear();
            self.value_view.reference_policies.clear();
            self.value_view.reference_metadata_key = None;
            metadata_pending = true;
        }
        let key = PolicyContextKey {
            root: run.root().clone(),
            revision: selected.revision,
            role: selected.model_role.clone(),
            input: selected.input.clone(),
            reference_serial: self.comparison.reference_interaction_serial(),
            reference_root: self.comparison.reference_run().map(|r| r.root().clone()),
            reference_revision: self.comparison.reference_revision(),
            reference_enabled: self.value_view.reference_enabled,
        };
        let eligibility = self.value_view.config.as_ref().map(|c| {
            let history =
                c.state.grid_eligibility() == crate::policy::slice::GridEligibility::HistoryBound;
            (
                !decision_loss && c.model_eligible(),
                history
                    .then(|| {
                        self.value_view
                            .anchor_key
                            .as_ref()
                            .map(|(key, _)| key.clone())
                    })
                    .flatten(),
            )
        });
        if self.value_view.context_key.as_ref() != Some(&key)
            || self.value_view.source_eligibility != eligibility
            || before.0 != self.value_view.reference_enabled
            || before.2 != self.value_view.choice
            || choice_before_metadata != self.value_view.choice
        {
            if !self.interactive.snapshot_validation_pending() {
                self.interactive.invalidate_value_context();
            }
            let same_reference = self.value_view.context_key.as_ref().is_some_and(|old| {
                old.root == key.root
                    && old.reference_serial == key.reference_serial
                    && old.reference_root == key.reference_root
                    && old.reference_revision == key.reference_revision
                    && old.reference_enabled == key.reference_enabled
            }) && self.value_view.source_eligibility == eligibility
                && before.2 == self.value_view.choice
                && choice_before_metadata == self.value_view.choice;
            self.value_view.request = None;
            self.value_view.current_search = None;
            self.value_view.decision_population_search = None;
            if !same_reference {
                self.value_view.value_search = None;
                self.value_view.policy_search = None;
            }
            self.value_view.context_key = Some(key);
            self.value_view.source_eligibility = eligibility.clone();
            self.value_view.dirty = true;
            self.value_view.ready = false;
            self.value_view.terminal = false;
            self.value_view.difference_available = false;
            self.value_view.comparison_available = false;
        }
        if before.1 != self.value_view.visits_enabled {
            self.value_view.dirty = true;
        }
        let mut pending = metadata_pending
            || self.workspace_defaults.is_some()
            || source.is_none() && self.loader.pending() > 0;
        let exact = !decision_loss && self.snapshot_catalog.is_some();
        let mut capability_only = !decision_loss
            && self
                .value_view
                .request
                .as_ref()
                .is_some_and(|r| r.capability_only);
        let mut current = if capability_only {
            self.value_view.request.as_ref().map(|r| r.current.clone())
        } else if !decision_loss && let Some(catalog) = &self.snapshot_catalog {
            match catalog.value_source_reusing(
                run,
                selected.revision,
                &mut self.loader,
                self.value_view.request.as_ref().map(|r| &r.current),
            ) {
                Ok(source) => source,
                Err(error) => {
                    self.interactive.fail_value_context(&error);
                    None
                }
            }
        } else {
            self.value_view.request.as_ref().map(|r| r.current.clone())
        };
        pending |= self.interactive.snapshot_validation_pending()
            || (exact
                && current.is_none()
                && self.snapshot_catalog.as_ref().is_some_and(|c| c.pending()));
        let mut reference = self
            .value_view
            .request
            .as_ref()
            .and_then(|r| r.reference_value.clone());
        let mut policy = self
            .value_view
            .request
            .as_ref()
            .and_then(|r| r.reference_policy.clone());
        if !exact && !metadata_pending && current.is_none() {
            if self.value_view.current_search.is_none() {
                self.value_view.current_search = Some(ModelSearch::new(
                    &selected.model_role,
                    selected.revision,
                    self.value_view.choice.as_ref().map(|c| c.name.as_str()),
                )?);
                if let Some((false, probe)) = &eligibility {
                    self.value_view
                        .current_search
                        .as_mut()
                        .unwrap()
                        .value_samples_only(probe.clone());
                }
            }
            let s = self.value_view.current_search.as_mut().unwrap();
            current = s.advance_value_for_slice(
                run,
                &mut self.loader,
                &mut self.resolver,
                self.value_view.config.as_ref().filter(|_| !decision_loss),
            )?;
            pending |= s.pending();
            if let Some(group) = self
                .value_view
                .choice
                .as_ref()
                .and_then(|c| c.twin.as_deref())
            {
                current = current.map(|s| s.with_twin(group)).transpose()?;
            }
        }
        // Preserve ordinary matching-source selection. After it settles with no
        // prediction source, retain a bounded recorded population solely for
        // independent capability discovery, never as an ordinary value fallback.
        if current.is_none() && !pending && !decision_loss && self.value_view.reference_enabled {
            if self.value_view.decision_population_search.is_none() {
                let mut search = ModelSearch::new(
                    &selected.model_role,
                    selected.revision,
                    self.value_view.choice.as_ref().map(|c| c.name.as_str()),
                )?;
                search
                    .value_samples_only(eligibility.as_ref().and_then(|(_, probe)| probe.clone()));
                self.value_view.decision_population_search = Some(search);
            }
            let search = self.value_view.decision_population_search.as_mut().unwrap();
            current =
                search.advance_value_for_slice(run, &mut self.loader, &mut self.resolver, None)?;
            pending |= search.pending();
            capability_only = current.is_some();
        }
        let empirical_reference = self.value_view.reference_enabled
            && self.comparison.value_empirical()
            && !decision_loss;
        if self.value_view.reference_enabled && !metadata_pending && !empirical_reference {
            if let Some(r) = self.comparison.reference_run() {
                let selected_choice = self
                    .comparison
                    .value_critic_override()
                    .and_then(|n| {
                        self.value_view
                            .reference_choices
                            .iter()
                            .find(|c| c.name == n)
                    })
                    .cloned()
                    .or_else(|| {
                        self.value_view.choice.as_ref().and_then(|p| {
                            self.value_view
                                .reference_choices
                                .iter()
                                .find(|c| c.name == p.name)
                                .cloned()
                        })
                    })
                    .or_else(|| {
                        (self.value_view.reference_choices.len() == 1)
                            .then(|| self.value_view.reference_choices[0].clone())
                    });
                if reference.is_none() {
                    if self.value_view.value_search.is_none() {
                        self.value_view.value_search = Some(ModelSearch::new(
                            self.comparison.reference_role(),
                            self.comparison.reference_revision(),
                            selected_choice.as_ref().map(|c| c.name.as_str()),
                        )?);
                        if crate::reference::provider_role(self.comparison.reference_role())
                            || matches!(eligibility, Some((false, _)))
                        {
                            self.value_view
                                .value_search
                                .as_mut()
                                .unwrap()
                                .value_samples_only(None);
                        }
                    }
                    let s = self.value_view.value_search.as_mut().unwrap();
                    reference = s.advance_value_for_slice(
                        r,
                        &mut self.loader,
                        &mut self.resolver,
                        self.value_view.config.as_ref(),
                    )?;
                    // An unnamed recorded reference can contain both V and Q
                    // records. Keep searching when the first candidate cannot
                    // participate in the requested same-input comparison.
                    // Explicit critic choices retain their normal error path.
                    if !decision_loss
                        && selected_choice.is_none()
                        && self.comparison.value_critic_override().is_none()
                        && self
                            .value_view
                            .condition
                            .unwrap_or(ComparisonCondition::SameInput)
                            == ComparisonCondition::SameInput
                        && current
                            .as_ref()
                            .zip(reference.as_ref())
                            .is_some_and(|(a, b)| {
                                (a.kind() == ValueCriticKind::StateValue)
                                    != (b.kind() == ValueCriticKind::StateValue)
                            })
                    {
                        s.reject_value_candidate(
                            "Same-input reference has incompatible V/Q conditioning",
                        );
                        reference = None;
                        ui.ctx().request_repaint();
                    }
                    pending |= s.pending();
                    if let Some(group) = selected_choice.as_ref().and_then(|c| c.twin.as_deref()) {
                        reference = reference.map(|s| s.with_twin(group)).transpose()?;
                    }
                }
                if policy.is_none() && !self.comparison.policy_uses_controller() {
                    if self.value_view.policy_search.is_none() {
                        self.value_view.policy_search = Some(ModelSearch::new(
                            self.comparison.reference_role(),
                            self.comparison.reference_revision(),
                            self.comparison.value_policy_override(),
                        )?);
                    }
                    let s = self.value_view.policy_search.as_mut().unwrap();
                    policy = s.advance(r, &mut self.loader)?;
                    pending |= s.pending();
                }
            }
            if policy.is_none() {
                let controller = self
                    .comparison
                    .policy_controller(&mut self.loader, Some(&selected.input))?;
                pending |= controller.status == ResolveStatus::Loading;
                if let (Some(frame), Some(r)) =
                    (controller.record.as_ref(), self.comparison.reference_run())
                {
                    policy = Some(PolicySource::recorded_controller(
                        r,
                        frame,
                        self.comparison.reference_role(),
                    )?);
                }
            }
        }
        if empirical_reference {
            reference = None;
            policy = None;
        }
        // Reuse only actual retained Arc owners after all immutable model,
        // critic, actor and declaration identities match. Equal reconstructed
        // allocations are never deduplicated by the accounting layer.
        if let Some(frame) = self.interactive.value_frame() {
            for candidate in current.iter_mut().chain(reference.iter_mut()) {
                if let Some(retained) = std::iter::once(frame.grid.current())
                    .chain(frame.grid.reference())
                    .find(|s| s.same(candidate) && (!decision_loss || s.same_payload(candidate)))
                {
                    *candidate = retained.clone();
                }
            }
            if let Some(candidate) = &mut policy
                && let Some(retained) = frame.grid.reference_policy().filter(|s| s.same(candidate))
            {
                *candidate = retained.clone();
            }
        }
        if let (Some(current), Some(reference)) = (&current, &mut reference)
            && current.same(reference)
        {
            *reference = current.clone();
        }
        let needs_ranges = self.value_view.config.as_ref().is_some_and(|c| {
            c.state
                .range_sources
                .contains(&crate::policy::slice::RangeSource::ModelInputDefault)
        });
        if self.value_view.visits_enabled || needs_ranges {
            pending |= self.prepare_value_visits(run, &selected)?;
            if !pending
                && needs_ranges
                && self.value_view.range_seed_key != self.value_view.visits_key
                && let Some(c) = &mut self.value_view.config
            {
                let counts = self
                    .value_view
                    .visits
                    .iter()
                    .map(|v| crate::value::visitation::input_rows(&v.chunk))
                    .collect::<Result<Vec<_>>>()?;
                let observations =
                    self.value_view
                        .visits
                        .iter()
                        .zip(counts)
                        .flat_map(|(v, rows)| {
                            (0..rows).map(move |row| source_observation(&v.chunk, row))
                        });
                if c.adopt_recorded_ranges(observations)? {
                    self.value_view.dirty = true;
                }
                self.value_view.range_seed_key = self.value_view.visits_key.clone();
            }
        }
        let (current_policy, policy_pending) = if let Some(current) = current
            .as_ref()
            .filter(|_| self.value_view.reference_enabled)
        {
            match self.resolve_value_current_policy(run, &selected, current) {
                Ok(resolved) => resolved,
                Err(_) if !decision_loss => (None, false),
                Err(error) => return Err(error),
            }
        } else {
            (None, false)
        };
        pending |= policy_pending;
        let current_available = current.is_some();
        if let Some(frame) = self.interactive.value_frame() {
            self.value_view.colors.accept(&frame.grid);
        }
        if let (Some(current), Some(mut c), Some(pin)) = (
            current,
            self.value_view.config.clone(),
            self.value_view.pin.clone(),
        ) {
            if !current.identity().recorded
                && current.return_target.is_some_and(ReturnTarget::is_soft)
                && matches!(c.conditioning, ActionConditioning::Policy)
                && !c.has_discrete_axis()
                && let Some(action) = c.default_first_action()
            {
                c.conditioning = ActionConditioning::Fixed(action);
                self.value_view.config = Some(c.clone());
                self.value_view.dirty = true;
            }
            let evidence_run = if empirical_reference {
                self.comparison.reference_run().unwrap_or(run)
            } else {
                run
            };
            let evidence_frontier = (
                current.identity().key.clone(),
                evidence_run.root().clone(),
                evidence_run
                    .candidate_frontier(
                        RecordKind::Empirical,
                        current.model_role(),
                        None,
                        current.identity().revision,
                    )
                    .0,
                evidence_run
                    .candidate_frontier(
                        RecordKind::Critic,
                        current.model_role(),
                        None,
                        current.identity().revision,
                    )
                    .0,
            );
            let evidence_refresh = self.value_view.request.is_none()
                || self.value_view.empirical_frontier.as_ref() != Some(&evidence_frontier);
            if evidence_refresh {
                self.value_view.dirty = true;
            }
            let color_request = if !pending && !decision_loss && !capability_only {
                self.value_view.colors.prepare(
                    crate::value::view::colors::ScaleInput {
                        current: &current,
                        slice: &c,
                        output: self
                            .value_view
                            .output
                            .as_ref()
                            .unwrap_or(&ValueOutput::ExpectedReturn),
                        reference: reference.as_ref(),
                        condition: self
                            .value_view
                            .condition
                            .unwrap_or(ComparisonCondition::SameInput),
                    },
                    run,
                    &mut self.loader,
                    &mut self.resolver,
                )?
            } else {
                None
            };
            if color_request.is_some() && self.value_view.colors.unsubmitted() {
                self.value_view.dirty = true;
            }
            if !pending && (self.value_view.dirty || self.value_view.request.is_none()) {
                let pin = if !current.identity().recorded
                    && matches!(pin, ValuePin::Probe { action: None, .. })
                    && (current.kind() == ValueCriticKind::StateValue
                        || !matches!(e.action_space(), ActionSpace::Discrete { .. }))
                {
                    ValuePin::State(c.anchor())
                } else {
                    pin
                };
                let pin = match pin {
                    ValuePin::Probe {
                        source,
                        row,
                        action: None,
                    } if current.identity().recorded
                        && current.identity().probe_key.as_ref() == Some(&source.key) =>
                    {
                        let (_, action) = current.recorded_input(row)?;
                        ValuePin::Probe {
                            source,
                            row,
                            action,
                        }
                    }
                    pin => pin,
                };
                self.value_view.pin = Some(pin.clone());
                // A capture may request derived preparation before the output has
                // been admitted to the UI. It must never bootstrap through a
                // learned critic prediction at the selected state.
                let mut c = c;
                let output = if decision_loss {
                    ValueView::decision_slice(&mut c);
                    ValueOutput::DecisionLoss
                } else {
                    self.value_view.quantity()
                };
                let (empirical, empirical_unavailable) = if decision_loss {
                    (None, None)
                } else if !evidence_refresh {
                    self.value_view.request.as_ref().map_or((None, None), |r| {
                        (r.empirical.clone(), r.empirical_unavailable.clone())
                    })
                } else {
                    match crate::value::empirical::EmpiricalSource::discover(evidence_run, &current)
                    {
                        Ok(source) => (source.map(std::sync::Arc::new), None),
                        Err(error) => (None, Some(crate::value::empirical::error_text(error))),
                    }
                };
                let mut request = ValueRequest::new(current, output, c, pin)?;
                request.empirical = empirical;
                request.reference_empirical = empirical_reference;
                request.empirical_condition = self.value_view.empirical_condition.clone();
                request.empirical_unavailable = empirical_unavailable;
                request.requested_revision = selected.revision;
                request.capability_only = capability_only;
                request.calibration = color_request;
                request.reference_requested_revision =
                    self.value_view.reference_enabled.then(|| {
                        if empirical_reference {
                            selected.revision
                        } else {
                            self.comparison.reference_revision()
                        }
                    });
                request.reference_value = reference;
                request.reference_policy = policy;
                request.current_policy = current_policy;
                request.condition = if decision_loss {
                    ComparisonCondition::SameInput
                } else {
                    self.value_view
                        .condition
                        .unwrap_or(ComparisonCondition::SameInput)
                };
                if self.value_view.visits_enabled {
                    request.visits = self.value_view.visits.clone();
                }
                if request.calibration.is_none()
                    && let Some(previous) = &self.value_view.request
                    && previous.same_grid(&request)
                    && previous.pin.same(&request.pin)
                    && previous.empirical_condition == request.empirical_condition
                {
                    request.calibration = self.value_view.colors.completed_calibration(previous);
                }
                if request.empirical.is_some() {
                    request.coalesce_calibration_sources();
                }
                let pending_catalog_bytes = if self.snapshot_catalog.is_none()
                    && !run.snapshot_revisions(&selected.model_role).is_empty()
                {
                    crate::snapshot_catalog::CATALOG_BYTES
                } else {
                    0
                };
                let mut admission = if self.value_view.automatic_resolution() {
                    self.interactive
                        .automatic_value_resolution(&mut request, pending_catalog_bytes)
                } else {
                    Ok(())
                };
                if admission.is_err()
                    && self.value_view.automatic_resolution()
                    && self.interactive.value_frame().is_some()
                {
                    self.interactive.release_value_frames();
                    admission = self
                        .interactive
                        .automatic_value_resolution(&mut request, pending_catalog_bytes);
                }
                if admission.is_err() && request.empirical.is_some() {
                    request.withdraw_empirical("Conditional evidence exceeds current Value replacement admission; ordinary critic prediction retained");
                    admission = if self.value_view.automatic_resolution() {
                        self.interactive
                            .automatic_value_resolution(&mut request, pending_catalog_bytes)
                    } else {
                        Ok(())
                    };
                }
                request.withdraw_unadmitted_empirical();
                match admission.and_then(|()| {
                    if !decision_loss || self.value_view.output == Some(ValueOutput::DecisionLoss) {
                        self.value_view.config = Some((*request.slice).clone());
                    }
                    self.interactive.submit_value(request.clone())
                }) {
                    Ok(()) => {
                        self.value_view.terminal = false;
                        self.value_view.colors.submitted(&request);
                        self.value_view.request = Some(request);
                        self.value_view.empirical_frontier = Some(evidence_frontier.clone());
                        self.value_view.dirty = false;
                    }
                    Err(err) => {
                        self.interactive.fail_value_context(&err);
                        self.value_view.request = Some(request);
                        self.value_view.status = Some(err.to_string());
                        self.value_view.terminal = true;
                        self.value_view.dirty = false;
                    }
                }
            }
        } else if !pending {
            self.value_view.status = Some(
                self.value_view
                    .current_search
                    .as_ref()
                    .and_then(|s| s.last_rejection.as_deref())
                    .unwrap_or(if eligibility.as_ref().is_some_and(|(model, _)| !model) {
                        "No compatible recorded critic samples available"
                    } else {
                        "No value model available"
                    })
                    .into(),
            );
            self.value_view.terminal = true;
        }
        if let Some(error) = self.interactive.value_error() {
            self.value_view.status = Some(error.into());
            self.value_view.ready = false;
            self.value_view.terminal = true;
        } else if pending || self.interactive.value_pending() {
            self.value_view.status = Some("Loading…".into());
            self.value_view.ready = false;
        } else if !current_available {
            // A held complete publication cannot certify a newly unavailable
            // model/sample selection or clear its explanatory status.
            self.value_view.ready = false;
        } else if let Some(frame) = self.interactive.value_frame() {
            self.value_view.ready = frame.selected.is_some();
            self.value_view.difference_available =
                frame.grid.reference_values.iter().any(Option::is_some)
                    || frame.grid.empirical_differences.is_some();
            self.value_view.comparison_available =
                frame.grid.reference_values.iter().any(Option::is_some)
                    || frame.grid.reference_series.is_some();
            if !self.value_view.comparison_available
                && frame.grid.output() == &self.value_view.quantity()
            {
                self.value_view.display = crate::policy::view::Display::Current;
            }
            self.value_view.status = if let Some(error) = &frame.grid.reference_error {
                Some(format!("Value reference unavailable: {error}"))
            } else if !frame.grid.regular() {
                Some("Recorded samples · surface unavailable".into())
            } else {
                None
            };
        }
        #[cfg(feature = "screenshots")]
        if self.value_view.capture_decision_loss
            && self.value_view.output != Some(ValueOutput::DecisionLoss)
            && self
                .value_view
                .request
                .as_ref()
                .is_some_and(ValueRequest::decision_loss_available)
        {
            self.value_view.output = Some(ValueOutput::DecisionLoss);
            if let Some(mut c) = self.value_view.config.take() {
                self.value_view
                    .sync_decision_controls(&mut c, &ValueOutput::DecisionLoss);
                self.value_view.config = Some(c);
            }
            self.value_view.dirty = true;
            self.value_view.ready = false;
        }
        #[cfg(feature = "screenshots")]
        if self.value_view.capture_decision_loss
            && !pending
            && !self.interactive.value_pending()
            && self.value_view.output != Some(ValueOutput::DecisionLoss)
            && self
                .value_view
                .request
                .as_ref()
                .is_some_and(|r| !r.decision_loss_available())
        {
            self.value_view.status = Some("Decision loss unavailable: current policy or compatible verified optimal Q reference unavailable".into());
            self.value_view.terminal = true;
            self.value_view.ready = false;
        }
        let shown = self.interactive.value_frame();
        #[cfg(feature = "screenshots")]
        let shown =
            shown.filter(|_| !(self.value_view.terminal && self.value_view.capture_decision_loss));
        let intent = self.value_view.show(ui, e, shown);
        if let Some(intent) = intent {
            intent.apply(run, &mut self.selection)?;
        }
        if self.value_view.dirty {
            self.value_view.ready = false;
            ui.ctx().request_repaint();
        }
        if pending || self.interactive.value_pending() {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        Ok(())
    }
    fn resolve_value_current_policy(
        &mut self,
        run: &RunIndex,
        selected: &Selection,
        current: &ValueSource,
    ) -> Result<(Option<PolicySource>, bool)> {
        if self.value_view.decision_loss_requested() {
            let Some(catalog) = &self.snapshot_catalog else {
                return Ok((None, true));
            };
            let previous = self
                .value_view
                .request
                .as_ref()
                .and_then(|r| r.current_policy.as_ref());
            let Some(policy) = catalog.policy_source_reusing(run, selected.revision, previous)?
            else {
                return Ok((None, catalog.pending()));
            };
            let index = run
                .by_sequence(policy.identity().key.sequence)
                .ok_or_else(|| rl::Error::State("Current policy snapshot missing".into()))?;
            let policy = policy.bind_metadata(run, || self.loader.request(run, index))?;
            let pending = policy.is_none();
            return Ok((
                policy.filter(|p| {
                    p.decision_supported()
                        && current
                            .actor_name()
                            .is_none_or(|actor| actor == p.decision_subject())
                }),
                pending,
            ));
        }
        if let Some(policy) = self
            .value_view
            .request
            .as_ref()
            .and_then(|r| r.current_policy.as_ref())
            && policy.identity().revision == selected.revision
            && policy.identity().key.root == *run.root()
        {
            if let Err(error) = policy.check_available() {
                self.value_view.unavailable_policy = Some((
                    policy.identity().key.clone(),
                    current.actor_name().map(str::to_owned),
                ));
                return Err(error);
            }
            return Ok((Some(policy.clone()), false));
        }
        let Some(index) = run
            .candidates(
                RecordKind::ModelSnapshot,
                &selected.model_role,
                None,
                selected.revision,
            )
            .find(|&i| run.recording().entries()[i].revision == selected.revision)
        else {
            return Ok((None, false));
        };
        let key = run.key(index)?;
        if self
            .value_view
            .unavailable_policy
            .as_ref()
            .is_some_and(|(cached, actor)| {
                *cached == key && actor.as_deref() == current.actor_name()
            })
        {
            return Ok((None, false));
        }
        let resolved = (|| -> Result<(Option<PolicySource>, bool)> {
            let Some(chunk) = self.loader.request(run, index)? else {
                return Ok((None, true));
            };
            let config = ValueSource::configuration(&chunk)?;
            // Match the shared Policy catalog's conservative, unambiguous subject rule.
            // An authored actor binding cannot silently select a different adapter.
            let name = match crate::policy::prepared::selected_name(&config, None) {
                Ok(name) => name,
                Err(_) => return Ok((None, false)),
            };
            if current.actor_name().is_some_and(|actor| actor != name) {
                return Ok((None, false));
            }
            let policy =
                PolicySource::model(run, index, Some(&name))?.with_metadata(run, &chunk)?;
            Ok((policy.decision_supported().then_some(policy), false))
        })();
        // Settled absence is distinct from pending. Immutable source/actor changes
        // miss this bounded cache; explicit Retry clears the complete Value view.
        if matches!(&resolved, Ok((None, false)) | Err(_)) {
            self.value_view.unavailable_policy =
                Some((key, current.actor_name().map(str::to_owned)));
        }
        resolved
    }
    fn prepare_value_visits(&mut self, run: &RunIndex, selected: &Selection) -> Result<bool> {
        let key = (
            run.root().clone(),
            selected.model_role.clone(),
            selected.revision,
            run.recording().entries().len(),
        );
        if self.value_view.visits_key.as_ref() == Some(&key) {
            return Ok(false);
        }
        let mut visits = Vec::new();
        let mut pending = false;
        // One bounded genuine population keeps identical decisions individually addressable.
        for i in run
            .candidates(
                RecordKind::Trajectory,
                &selected.model_role,
                None,
                selected.revision,
            )
            .take(8)
        {
            if let Some(chunk) = self.loader.request(run, i)? {
                let key = run.key(i)?;
                visits.push(VisitSource {
                    chunk: self.resolver.lease_visitation(run, &key, chunk)?,
                    key,
                });
            } else {
                pending = true;
            }
        }
        if visits.is_empty() {
            for i in run
                .candidates(
                    RecordKind::ProbeSet,
                    &selected.model_role,
                    None,
                    selected.revision,
                )
                .take(16)
            {
                if let Some(chunk) = self.loader.request(run, i)? {
                    if matches!(&chunk.record,Record::ProbeSet(p)if p.population==ProbePopulation::PolicyVisited)
                    {
                        let key = run.key(i)?;
                        visits.push(VisitSource {
                            chunk: self.resolver.lease_visitation(run, &key, chunk)?,
                            key,
                        });
                        break;
                    }
                } else {
                    pending = true;
                }
            }
        }
        if !self
            .value_view
            .visits
            .iter()
            .map(|v| &v.key)
            .eq(visits.iter().map(|v| &v.key))
        {
            self.value_view.dirty = true;
        }
        self.value_view.visits = visits;
        if !pending {
            self.value_view.visits_key = Some(key);
        }
        Ok(pending)
    }
}
#[cfg(feature = "screenshots")]
impl Laboratory {
    pub(super) fn value_capture_hover(
        &self,
        ctx: &egui::Context,
        input: &mut egui::RawInput,
        enabled: Option<bool>,
    ) {
        if self.view != View::Value {
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
        let Some(frame) = self.value_frame() else {
            return;
        };
        let Some(point) = &frame.selected else {
            return;
        };
        if std::env::var("RL_VIEWER_VALUE_HOVER_TARGET").as_deref() == Ok("selected") {
            if let Some(rect) = self.value_view.inspector_rect {
                input
                    .events
                    .push(egui::Event::PointerMoved(rect.min + egui::vec2(30., 48.)));
            }
            return;
        }
        let override_position = std::env::var("RL_VIEWER_VALUE_HOVER_POSITION")
            .ok()
            .and_then(|v| {
                let (x, y) = v.split_once(',')?;
                Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?))
            })
            .or_else(|| {
                if std::env::var("RL_VIEWER_VALUE_HOVER_TARGET").as_deref() != Ok("actor") {
                    return None;
                }
                let g = &frame.grid;
                let i = g.slice.axis_len(g.slice.x, 0).ok()? / 2;
                let rl::core::action::Action::Continuous(a) = g.policy_overlay.get(i)?.as_ref()?
                else {
                    return None;
                };
                let ValueCoordinate::Observation(_) = g.slice.x else {
                    return None;
                };
                let Some(ValueCoordinate::Action(j)) = g.slice.y else {
                    return None;
                };
                Some((g.positions[i][0] as f64, a[j] as f64))
            });
        if let Some((x, y)) = override_position
            && let Some(memory) = egui_plot::PlotMemory::load(ctx, ValueView::main_plot_id())
        {
            input.events.push(egui::Event::PointerMoved(
                memory
                    .transform()
                    .position_from_point(&egui_plot::PlotPoint::new(x, y)),
            ));
            return;
        }
        let coordinate = |axis| match axis {
            ValueCoordinate::Observation(i) => {
                point.input.dense_values().get(i).copied().map(f64::from)
            }
            ValueCoordinate::Action(i) => match &point.action {
                Some(rl::core::action::Action::Continuous(a)) => a.get(i).copied().map(f64::from),
                _ => None,
            },
            ValueCoordinate::DiscreteAction => match &point.action {
                Some(rl::core::action::Action::Discrete(a)) => Some(*a as f64),
                _ => None,
            },
        };
        if let (Some(x), Some(y), Some(memory)) = (
            coordinate(frame.grid.slice.x),
            frame
                .grid
                .slice
                .y
                .and_then(coordinate)
                .or(Some(point.value)),
            egui_plot::PlotMemory::load(ctx, ValueView::main_plot_id()),
        ) {
            input.events.push(egui::Event::PointerMoved(
                memory
                    .transform()
                    .position_from_point(&egui_plot::PlotPoint::new(x, y)),
            ));
        }
    }
    fn configure_value_capture(&mut self, e: EnvironmentSemantics<'_>) -> Result<()> {
        if self.value_view.capture_configured || self.value_view.config.is_none() {
            return Ok(());
        }
        let invalid = || rl::Error::Config("Invalid Value capture configuration".into());
        if let Ok(condition) = std::env::var("RL_VIEWER_VALUE_CONDITION") {
            self.value_view.condition = Some(capture_condition(&condition)?);
        }
        let coordinate = |s: &str| -> Result<ValueCoordinate> {
            if s == "category" {
                return Ok(ValueCoordinate::DiscreteAction);
            }
            let (kind, index) = s.split_once(':').ok_or_else(invalid)?;
            let i = index.parse().map_err(|_| invalid())?;
            match kind {
                "obs" => Ok(ValueCoordinate::Observation(i)),
                "action" => Ok(ValueCoordinate::Action(i)),
                _ => Err(invalid()),
            }
        };
        let c = self.value_view.config.as_mut().unwrap();
        if let Ok(mode) = std::env::var("RL_VIEWER_VALUE_MODE") {
            match mode.as_str() {
                "1d" => {
                    c.mode = SliceMode::OneDimensional;
                    c.y = None;
                    c.resolution = [256, 1];
                }
                "2d" => {
                    c.mode = SliceMode::TwoDimensional;
                    c.resolution = [64, 64];
                }
                _ => return Err(invalid()),
            }
        }
        if let Ok(x) = std::env::var("RL_VIEWER_VALUE_X") {
            c.x = coordinate(&x)?;
        }
        if let Ok(y) = std::env::var("RL_VIEWER_VALUE_Y") {
            c.y = Some(coordinate(&y)?);
        }
        if let Ok(fixed) = std::env::var("RL_VIEWER_VALUE_FIXED") {
            for field in fixed.split(',') {
                let (i, v) = field.split_once('=').ok_or_else(invalid)?;
                let i: usize = i.parse().map_err(|_| invalid())?;
                *c.state.fixed.get_mut(i).ok_or_else(invalid)? =
                    v.parse().map_err(|_| invalid())?;
            }
        }
        if let Ok(r) = std::env::var("RL_VIEWER_VALUE_RESOLUTION") {
            self.value_view.resolution_explicit[usize::from(c.mode == SliceMode::TwoDimensional)] =
                true;
            let mut fields = r.split(',');
            c.resolution[0] = fields
                .next()
                .ok_or_else(invalid)?
                .parse()
                .map_err(|_| invalid())?;
            c.resolution[1] = if c.y.is_some() {
                fields
                    .next()
                    .ok_or_else(invalid)?
                    .parse()
                    .map_err(|_| invalid())?
            } else {
                1
            };
        }
        c.validate(e)?;
        if std::env::var_os("RL_VIEWER_VALUE_FIXED").is_some() || self.value_view.pin.is_none() {
            self.value_view.pin = Some(ValuePin::State(c.anchor()));
        }
        if let Ok(action) = std::env::var("RL_VIEWER_VALUE_ACTION") {
            let action = rl::core::action::Action::Continuous(
                action
                    .split(',')
                    .map(|v| v.parse().map_err(|_| invalid()))
                    .collect::<Result<Vec<f32>>>()?,
            );
            e.action_space().check(&action, c.state.anchor_mask())?;
            c.conditioning = ActionConditioning::Fixed(action.clone());
            self.value_view.pin = Some(ValuePin::Input {
                observation: c.anchor(),
                action: Some(action),
            });
        }
        self.value_view.reference_enabled =
            std::env::var("RL_VIEWER_VALUE_REFERENCE").is_ok_and(|v| v == "true");
        self.comparison.set_value_empirical(
            std::env::var("RL_VIEWER_VALUE_REFERENCE_SOURCE").as_deref() == Ok("empirical"),
        );
        self.value_view.visits_enabled =
            std::env::var("RL_VIEWER_VALUE_VISITS").is_ok_and(|v| v == "true");
        if self.value_view.reference_enabled {
            let revision = std::env::var("RL_VIEWER_VALUE_REFERENCE_K")
                .ok()
                .map(|v| v.parse().map_err(|_| invalid()))
                .transpose()?;
            let path =
                std::env::var_os("RL_VIEWER_VALUE_REFERENCE_RUN").map(std::path::PathBuf::from);
            let controller =
                std::env::var("RL_VIEWER_VALUE_REFERENCE_SOURCE").is_ok_and(|s| s == "controller");
            let role = std::env::var("RL_VIEWER_VALUE_REFERENCE_ROLE").ok();
            self.comparison.configure_policy_reference(
                path,
                revision,
                controller,
                role.as_deref(),
            )?;
        }
        self.value_view.display = match std::env::var("RL_VIEWER_VALUE_DISPLAY").as_deref() {
            Ok("reference") => crate::policy::view::Display::Reference,
            Ok("difference") => crate::policy::view::Display::Difference,
            Ok("current") | Err(_) => crate::policy::view::Display::Current,
            _ => return Err(invalid()),
        };
        self.value_view.capture_configured = true;
        Ok(())
    }
    pub(super) fn value_capture_ready(&self) -> bool {
        self.value_capture_ready_for(
            std::env::var("RL_VIEWER_VALUE_EXPECT_UNAVAILABLE")
                .ok()
                .as_deref(),
        )
    }
    fn value_capture_idle(&self) -> bool {
        !self.interactive.pending()
            && !self.interactive.value_pending()
            && self.loader.pending() == 0
            && self.workspace_defaults.is_none()
    }
    fn value_capture_ready_for(&self, expected: Option<&str>) -> bool {
        if let Some(expected) = expected {
            return self.view == View::Value
                && !expected.trim().is_empty()
                && self.value_view.terminal
                && !self.value_view.ready
                && self.value_capture_idle()
                && self.value_view.status.as_deref() == Some(expected);
        }
        self.value_view.ready
            && !self.interactive.value_pending()
            && self.value_frame().is_some_and(|f| {
                f.requested_revision == self.selection.revision
                    && f.reference_requested_revision
                        == self
                            .value_view
                            .request
                            .as_ref()
                            .and_then(|r| r.reference_requested_revision)
                    && self
                        .value_view
                        .request
                        .as_ref()
                        .is_some_and(|r| f.grid.matches_request(r))
                    && (!self.value_view.capture_decision_loss
                        || f.grid.output() == &ValueOutput::DecisionLoss
                            && f.selected
                                .as_ref()
                                .is_some_and(|p| p.decision_loss.is_some()))
            })
    }
    pub(super) fn value_capture_error(&self) -> Option<String> {
        self.value_capture_problem(
            std::env::var("RL_VIEWER_VALUE_EXPECT_UNAVAILABLE")
                .ok()
                .as_deref(),
        )
    }
    fn value_capture_problem(&self, expected: Option<&str>) -> Option<String> {
        if self.view != View::Value {
            return None;
        }
        if expected.is_some_and(|s| s.trim().is_empty()) {
            return Some("Expected unavailable Value status must be nonempty".into());
        }
        if self.value_view.terminal && !self.value_view.ready && self.value_capture_idle() {
            return if self.value_capture_ready_for(expected) {
                None
            } else {
                Some(
                    self.value_view
                        .status
                        .clone()
                        .unwrap_or_else(|| "Value rendering unavailable".into()),
                )
            };
        }
        (expected.is_some() && self.value_view.ready && self.value_capture_idle())
            .then(|| "Expected unavailable Value status was not observed".into())
    }
    pub(super) fn value_capture_evidence(&self) -> serde_json::Value {
        self.value_frame().map_or(serde_json::Value::Null,|f| {
            let conditions = f.grid.empirical.conditions.iter().map(|c|serde_json::json!({"revision":c.revision,"record":c.source_key.sequence,"probe":c.probe_key.sequence,"row_id":c.row_id,"input":c.input.dense_values(),"action":format!("{:?}",c.action),"n":c.summary.count,"mean":c.summary.mean,"mean_standard_error":c.summary.mean_standard_error})).collect::<Vec<_>>();
            let selected = f.selected.as_ref().map(|p| {
                let empirical = p.empirical.as_ref().map(|e|serde_json::json!({"record":e.condition.source_key.sequence,"n":e.condition.summary.count,"mean":e.condition.summary.mean,"mean_standard_error":e.condition.summary.mean_standard_error,"cdf_points":e.cdf.len(),"conditioning":e.condition.methodology,"continuation":e.condition.continuation}));
                serde_json::json!({
                    "input":p.input.dense_values(),"mask":p.input.dense_mask(),"action":format!("{:?}",p.action),
                    "value":p.value,"reference":p.reference_value,"reference_action":format!("{:?}",p.reference_action),
                    "reference_error":p.reference_error,"policy_conditioned":p.policy_conditioned,
                    "recorded_revision":p.recorded_revision,"recorded_key":format!("{:?}",p.recorded_key),"recorded_row":p.recorded_row,
                    "observed_return":p.observed_return,"empirical":empirical,"empirical_error":p.empirical_error,
                    "current_policy":p.current_policy().map(|s|format!("{:?}",s.identity())),
                    "decision_loss":p.decision_loss.map(|d|serde_json::json!({"optimal_value":d.optimal_value,"expected_action_value":d.expected_action_value,"loss":d.loss}))
                })
            });
            serde_json::json!({
            "snapshot_catalog":self.snapshot_catalog.as_ref().map(|c|serde_json::json!({"checking":c.pending(),"limited":c.limited(),"positions":c.positions()})),
            "requested_revision":f.requested_revision,"displayed":self.transport.capture_publication(),
            "current":format!("{:?}",f.grid.current().identity()),
            "output":format!("{:?}",f.grid.output()),
            "critic_reference":f.grid.reference().map(|r|format!("{:?}",r.identity())),
            "actor_reference":f.grid.reference_policy().map(|r|format!("{:?}",r.identity())),
            "reference_requested_revision":f.reference_requested_revision,
            "comparison_condition":format!("{:?}",f.grid.comparison_condition()),
            "empirical_reference":f.grid.empirical_reference(),
            "empirical_conditions":conditions,
            "empirical_rejections":f.grid.empirical.rejections,"empirical_error":f.grid.empirical_error,
            "empirical_difference_count":f.grid.empirical_differences.as_ref().map_or(0,|s|s.values.len()),
            "input_identity":format!("{:?}",self.selection.input),
            "resolution":f.grid.slice.resolution,"resolution_explicit":self.value_view.resolution_explicit,
            "slice":format!("{:?}",f.grid.slice),"rows":f.grid.values.len(),"regular":f.grid.regular(),
            "scale":self.value_view.color_limits(),"metrics":self.value_metrics(),"owners":self.interactive.owner_snapshot(),
            "texture_bytes":self.value_view.texture_bytes(),"texture_uploads":self.value_view.texture_uploads,
            "selected":selected
        })})
    }
}

#[cfg(feature = "screenshots")]
fn capture_condition(value: &str) -> Result<ComparisonCondition> {
    match value {
        "same_input" => Ok(ComparisonCondition::SameInput),
        "controller_values" => Ok(ComparisonCondition::ControllerValues),
        _ => Err(rl::Error::Config(
            "Invalid Value comparison capture condition".into(),
        )),
    }
}

#[cfg(all(test, feature = "screenshots"))]
mod capture_tests {
    use super::*;
    #[test]
    fn empirical_value_large_optional_cdf_preserves_ordinary_prediction_and_mean() {
        let (_dir, run, current) = crate::value::empirical::tests::fixture_samples(7000);
        let source = crate::value::empirical::EmpiricalSource::discover(&run, &current)
            .unwrap()
            .unwrap();
        let batch = source
            .prepare(&current, &ValueOutput::ExpectedReturn, &|| false)
            .unwrap();
        assert_eq!(
            batch.conditions.len(),
            2,
            "Large analytical samples must fit source admission"
        );
        let context = egui::Context::default();
        let mut app = Laboratory::from_context(&context, run.root().display().to_string()).unwrap();
        app.select_view(View::Value);
        app.value_view.capture_configured = true;
        let until = Instant::now() + Duration::from_secs(20);
        loop {
            let mut output = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600., 1000.),
                    )),
                    ..Default::default()
                },
                |ui| app.show(ui),
            );
            output.textures_delta.clear();
            assert!(!app.value_view.terminal, "{:?}", app.value_view.status());
            if app.value_view.ready {
                break;
            }
            assert!(Instant::now() < until, "{:?}", app.value_view.status());
            std::thread::sleep(Duration::from_millis(2));
        }
        let point = app.value_frame().unwrap().selected.as_ref().unwrap();
        assert!(point.value.is_finite());
        assert_eq!(
            point.empirical.as_ref().unwrap().condition.summary.count,
            7000
        );
        assert_eq!(point.empirical.as_ref().unwrap().cdf.len(), 14000);
    }
    #[test]
    fn empirical_value_ui_keeps_original_selection_and_shows_conditional_evidence() {
        let (_dir, run, _) = crate::value::empirical::tests::fixture();
        let context = egui::Context::default();
        let mut app = Laboratory::from_context(&context, run.root().display().to_string()).unwrap();
        app.select_view(View::Value);
        app.value_view.capture_configured = true; // Exercise manual controls, not startup screenshot overrides.
        app.comparison.set_value_empirical(true);
        app.value_view.reference_enabled = true;
        let until = Instant::now() + Duration::from_secs(20);
        let mut configured = false;
        loop {
            let mut output = context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1600., 1000.),
                    )),
                    ..Default::default()
                },
                |ui| app.show(ui),
            );
            output.textures_delta.clear();
            if !configured && app.value_view.ready {
                app.value_view
                    .set_axes(ValueCoordinate::Observation(0), None)
                    .unwrap();
                configured = true;
                continue;
            }
            if configured && app.value_view.ready {
                break;
            }
            assert!(Instant::now() < until, "{:?}", app.value_view.status());
            std::thread::sleep(Duration::from_millis(2));
        }
        let frame = app.value_frame().unwrap();
        assert!(frame.grid.empirical_reference());
        assert_eq!(frame.grid.empirical.conditions.len(), 2);
        assert!(frame.selected.as_ref().unwrap().empirical.is_some());
        let limits = app.value_view.color_limits().unwrap();
        for condition in &frame.grid.empirical.conditions {
            assert!(
                limits[0] <= condition.summary.mean && condition.summary.mean <= limits[1],
                "Empirical overlay {:.4} is outside the visible Value range {:?}",
                condition.summary.mean,
                limits
            );
        }
        assert!(matches!(
            app.selection.input,
            SelectedInput::ProbeRow { .. }
        ));
    }
    #[test]
    fn capture_condition_uses_existing_semantics_and_rejects_unknown_values() {
        assert_eq!(
            capture_condition("same_input").unwrap(),
            ComparisonCondition::SameInput
        );
        assert_eq!(
            capture_condition("controller_values").unwrap(),
            ComparisonCondition::ControllerValues
        );
        assert!(capture_condition("decision_loss").is_err());
    }
    #[test]
    fn decision_loss_unavailable_capture_requires_exact_status_and_idle() {
        let dir = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = Laboratory::from_context(&context, dir.path().display().to_string()).unwrap();
        app.select_view(View::Value);
        let until = Instant::now() + Duration::from_secs(10);
        while !app.value_capture_idle() {
            app.interactive.housekeeping();
            app.loader.poll();
            assert!(Instant::now() < until);
            std::thread::sleep(Duration::from_millis(1));
        }
        let expected = "Decision loss unavailable: current policy or compatible verified optimal Q reference unavailable";
        app.value_view.terminal = true;
        app.value_view.status = Some(expected.into());
        assert!(app.value_capture_ready_for(Some(expected)));
        assert_eq!(app.value_capture_problem(Some(expected)), None);
        assert!(!app.value_capture_ready_for(Some("wrong")));
        assert!(app.value_capture_problem(Some("wrong")).is_some());
        assert!(app.value_capture_problem(None).is_some());
        assert!(!app.value_capture_ready_for(Some("")));
        app.value_view.terminal = false;
        assert!(!app.value_capture_ready_for(Some(expected)));
    }
    #[test]
    fn unavailable_value_capture_settles_and_pointer_can_be_removed() {
        let dir = tempfile::tempdir().unwrap();
        let context = egui::Context::default();
        let mut app = Laboratory::from_context(&context, dir.path().display().to_string()).unwrap();
        app.select_view(View::Value);
        let until = Instant::now() + Duration::from_secs(10);
        while !app.value_capture_idle() {
            app.interactive.housekeeping();
            app.loader.poll();
            assert!(Instant::now() < until);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!app.value_capture_ready());
        app.value_view.terminal = true;
        app.value_view.status = Some("No value model available".into());
        assert!(!app.value_capture_ready());
        assert!(app.value_capture_problem(None).is_some());
        let mut input = egui::RawInput::default();
        app.value_capture_hover(&context, &mut input, Some(false));
        assert!(matches!(
            input.events.as_slice(),
            [egui::Event::PointerGone]
        ));
    }
}
