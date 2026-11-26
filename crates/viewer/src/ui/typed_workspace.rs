//! Typed inputs presented through the shared viewer workspaces.
use super::*;
use crate::{theme::CURRENT, timeline_ui::SnapshotPresentation};
use egui_plot::{Line, Plot};
use rl::{
    Error,
    diagnostics::{
        reference::{
            ReferenceInformation, ReferenceLifetime, ReferenceTarget, compatible_contexts,
            manifest_context, provider_specification,
        },
        tabular::{restore, tabular_decision},
        typed::{DiagnosticInput, FiniteDecision, FiniteValueKind, MAX_TYPED_BYTES, TypedPayload},
    },
};
use std::collections::BTreeMap;

#[path = "typed_workspace/presentation.rs"]
mod presentation;
use presentation::{
    action_label, draw_actions, draw_input, draw_observation, draw_population, draw_values,
    input_label,
};

#[path = "typed_workspace/episode.rs"]
mod episode;

const PAYLOAD_ADMISSION: usize = MAX_TYPED_BYTES + 96 * 1024;
const ROW_LIMIT: usize = 128;

#[cfg(test)]
#[path = "typed_workspace/tests.rs"]
mod tests;

#[derive(Default)]
struct PayloadSlot(Option<(CacheKey, Arc<TypedPayload>)>);

struct ReferenceRow {
    label: String,
    decision: FiniteDecision,
    value_kind: Option<FiniteValueKind>,
}

#[derive(PartialEq, Eq)]
struct PositionKey {
    run: Arc<std::path::PathBuf>,
    entries: usize,
    role: String,
    probe: Option<String>,
    kind: RecordKind,
}

impl PayloadSlot {
    fn read(
        &mut self,
        run: &RunIndex,
        index: usize,
        loader: &mut Loader,
    ) -> Result<Option<Arc<TypedPayload>>> {
        let key = run.key(index)?;
        if let Some((cached, payload)) = &self.0
            && cached == &key
        {
            return Ok(Some(payload.clone()));
        }
        self.0 = None;
        let Some(chunk) = loader.request_admitted(run, index, PAYLOAD_ADMISSION)? else {
            return Ok(None);
        };
        let payload = Arc::new(TypedPayload::read(&chunk)?);
        self.0 = Some((key, payload.clone()));
        Ok(Some(payload))
    }
}

#[derive(Default)]
pub(super) struct TypedPane {
    inputs: PayloadSlot,
    anchor: PayloadSlot,
    predictions: PayloadSlot,
    reference_inputs: PayloadSlot,
    reference: PayloadSlot,
    episode: PayloadSlot,
    positions: Vec<u64>,
    position_key: Option<PositionKey>,
    episode_tick: Option<Instant>,
    episode_sequence: Option<u64>,
    verified_predictions: Option<(CacheKey, CacheKey)>,
    verified_reference: Option<(CacheKey, CacheKey)>,
    probe_anchor: Option<SelectedInput>,
    pub ready: Option<u64>,
    pub episode_publication: Option<(u64, u64, usize)>,
}

impl TypedPane {
    pub(super) fn leave(&mut self) {
        *self = Self::default();
    }

    pub(super) fn switch_view(&mut self) {
        self.ready = None;
        self.episode_publication = None;
        self.episode_tick = None;
    }
}

pub(super) fn typed_run(run: &RunIndex) -> bool {
    run.recording().manifest().typed_observation_space.is_some()
        || run
            .record_roles(RecordKind::FinitePredictions)
            .next()
            .is_some()
}

pub(super) fn typed_view(run: &RunIndex, view: View, selection: &Selection) -> bool {
    if run.recording().manifest().typed_observation_space.is_some() {
        return view != View::Training;
    }
    let dense_kind = match view {
        View::PolicyMap => RecordKind::Policy,
        View::Value => RecordKind::Critic,
        View::Network => RecordKind::Activations,
        _ => return false,
    };
    let set = match &selection.input {
        SelectedInput::ProbeRow { set, .. } => Some(set.as_str()),
        _ => None,
    };
    let revision = |kind| {
        run.candidates(kind, &selection.model_role, set, selection.revision)
            .find(|&index| run.recording().is_retained(index))
            .map(|index| run.recording().entries()[index].revision)
    };
    if view == View::Network {
        return run
            .candidates(dense_kind, &selection.model_role, set, selection.revision)
            .find(|&index| run.recording().is_retained(index))
            .and_then(|index| run.probe_for_record(index))
            .is_some_and(|probe| run.recording().entries()[probe].kind == RecordKind::TypedProbes);
    }
    match (
        revision(RecordKind::FinitePredictions),
        revision(dense_kind),
    ) {
        (Some(typed), Some(dense)) => typed >= dense,
        (Some(_), None) => true,
        _ => false,
    }
}

impl Laboratory {
    pub(super) fn select_input_population(&mut self, run: &RunIndex, set: &str, row: usize) {
        let mut selection = self.selection.clone();
        selection.select_probe(run, set, row);
        let typed = typed_view(run, self.view, &selection);
        if let Some(index) = run.probe_set_of_kind(
            set,
            selection.revision,
            if typed {
                RecordKind::TypedProbes
            } else {
                RecordKind::ProbeSet
            },
        ) {
            selection.input = SelectedInput::ProbeRow {
                set: set.into(),
                row,
                sequence: run.recording().entries()[index].sequence,
            };
        }
        self.selection = selection;
    }
}

fn exact(
    run: &RunIndex,
    kind: RecordKind,
    role: &str,
    probe: Option<&str>,
    revision: u64,
) -> Option<usize> {
    run.candidates(kind, role, probe, revision)
        .take_while(|&i| run.recording().entries()[i].revision == revision)
        .find(|&i| run.recording().is_retained(i))
}

impl Laboratory {
    pub(super) fn typed_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let kind = match self.view {
            View::PolicyMap | View::Value => RecordKind::FinitePredictions,
            View::Network => RecordKind::Activations,
            _ => RecordKind::TypedEpisode,
        };
        let probe = match (&self.selection.input, kind) {
            (
                SelectedInput::ProbeRow { set, .. },
                RecordKind::FinitePredictions | RecordKind::Activations,
            ) => Some(set.clone()),
            _ => None,
        };
        let key = PositionKey {
            run: run.root().clone(),
            entries: run.recording().entries().len(),
            role: self.selection.model_role.clone(),
            probe,
            kind,
        };
        if self.typed_pane.position_key.as_ref() != Some(&key) {
            self.typed_pane.ready = None;
            let entries = run.recording().entries();
            self.typed_pane.positions = run
                .candidates(kind, &key.role, key.probe.as_deref(), u64::MAX)
                .filter(|&i| run.recording().is_retained(i))
                .map(|i| entries[i].revision)
                .collect();
            self.typed_pane.positions.reverse();
            self.typed_pane.positions.dedup();
            self.typed_pane.position_key = Some(key);
        }
        let requested = self.selection.revision;
        let ready = self.typed_pane.ready == Some(requested);
        self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            SnapshotPresentation {
                positions: &self.typed_pane.positions,
                published: self.typed_pane.ready,
                ready,
                checking: false,
                error: None,
            },
        )?;
        Ok(())
    }

    pub(super) fn typed_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        self.typed_pane.ready = None;
        self.typed_pane.episode_publication = None;
        if matches!(self.view, View::Dynamics | View::Environment) {
            if matches!(self.selection.input, SelectedInput::ProbeRow { .. }) {
                self.typed_pane.probe_anchor = Some(self.selection.input.clone());
            }
            return self.typed_episode(ui, run);
        }
        if matches!(self.selection.input, SelectedInput::TrajectoryStep { .. })
            && let Some(anchor) = &self.typed_pane.probe_anchor
        {
            self.selection.input = anchor.clone();
        }
        if self.view == View::Network {
            return self.typed_network(ui, run);
        }
        self.typed_decisions(ui, run)
    }

    fn typed_decisions(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let entries = run.recording().entries();
        let requested = self.selection.revision;
        let role = self.selection.model_role.as_str();
        let anchor = match &self.selection.input {
            SelectedInput::ProbeRow { set, sequence, row } => Some((set.as_str(), *sequence, *row)),
            _ => None,
        };
        let Some(record) = exact(
            run,
            RecordKind::FinitePredictions,
            role,
            anchor.map(|a| a.0),
            requested,
        ) else {
            ui.weak(format!("No finite predictions at k {requested}"));
            return Ok(());
        };
        let Some(input_index) = run.probe_for_record(record) else {
            return Err(Error::State(
                "finite predictions have no recorded inputs".into(),
            ));
        };
        let Some(inputs) = self
            .typed_pane
            .inputs
            .read(run, input_index, &mut self.loader)?
        else {
            ui.weak("Loading inputs…");
            return Ok(());
        };
        let TypedPayload::Probes(inputs) = inputs.as_ref() else {
            return Err(Error::Shape(
                "finite prediction inputs have the wrong record type".into(),
            ));
        };
        let Some(frame) = self
            .typed_pane
            .predictions
            .read(run, record, &mut self.loader)?
        else {
            ui.weak("Loading predictions…");
            return Ok(());
        };
        let TypedPayload::Predictions(frame) = frame.as_ref() else {
            return Err(Error::Shape("expected finite predictions".into()));
        };
        let manifest = run.recording().manifest();
        let validation = (run.key(record)?, run.key(input_index)?);
        if self.typed_pane.verified_predictions.as_ref() != Some(&validation) {
            frame.validate_support(
                &inputs.row_ids,
                inputs.inputs.iter().map(DiagnosticInput::observation),
                &manifest.action_space,
            )?;
            self.typed_pane.verified_predictions = Some(validation);
        }
        let selected = if let Some((_, sequence, row)) = anchor {
            if sequence == entries[input_index].sequence {
                (row < inputs.inputs.len()).then_some(row)
            } else if let Some(anchor_index) = run
                .by_sequence(sequence)
                .filter(|&i| entries[i].kind == RecordKind::TypedProbes)
            {
                let Some(previous) =
                    self.typed_pane
                        .anchor
                        .read(run, anchor_index, &mut self.loader)?
                else {
                    ui.weak("Loading selected input…");
                    return Ok(());
                };
                if let TypedPayload::Probes(previous) = previous.as_ref() {
                    previous
                        .row_ids
                        .get(row)
                        .zip(previous.inputs.get(row))
                        .and_then(|(id, input)| {
                            inputs.row_ids.iter().zip(&inputs.inputs).position(
                                |(candidate, other)| id == candidate && input.same_input_as(other),
                            )
                        })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            Some(0)
        };
        let mut next_row = selected;
        let title = match self.view {
            View::PolicyMap => "Legal action policy",
            _ => match frame.value_kind {
                Some(FiniteValueKind::PreferenceScore) => "Action preference scores",
                Some(FiniteValueKind::OptimalActionValue) => "Optimal action values",
                _ => "Action values",
            },
        };
        ui.horizontal(|ui| {
            ui.heading(title);
            ui.weak(format!("k {requested}"));
            egui::ComboBox::from_id_salt("typed-row")
                .width(148.)
                .selected_text(
                    selected
                        .map(|row| input_label(&inputs.inputs[row]))
                        .unwrap_or_else(|| "Select input".into()),
                )
                .show_ui(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(280.).show_rows(
                        ui,
                        24.,
                        inputs.inputs.len(),
                        |ui, range| {
                            for row in range {
                                if ui
                                    .selectable_label(
                                        selected == Some(row),
                                        input_label(&inputs.inputs[row]),
                                    )
                                    .clicked()
                                {
                                    next_row = Some(row);
                                }
                            }
                        },
                    );
                });
        });
        if self.view == View::PolicyMap
            && let Some(clicked) = draw_population(
                ui,
                &inputs.inputs,
                &frame.rows,
                &manifest.action_space,
                next_row,
            )
        {
            next_row = Some(clicked);
        }
        if next_row != selected || anchor.is_none() && next_row.is_some() {
            self.selection.input = SelectedInput::ProbeRow {
                set: entries[input_index].probe_set.clone().unwrap_or_default(),
                row: next_row.unwrap_or(0),
                sequence: entries[input_index].sequence,
            };
        }
        let Some(row) = next_row else {
            ui.weak("Selected input is absent at this checkpoint");
            return Ok(());
        };
        let input = &inputs.inputs[row];
        let reference = self.typed_reference(
            run,
            input,
            frame.rows[row].row_id,
            entries[input_index].probe_set.as_deref(),
        )?;
        let decision = &frame.rows[row];
        if self.view == View::Value {
            draw_values(
                ui,
                decision,
                reference.as_ref(),
                frame.value_kind,
                &manifest.action_space,
            );
        }
        ui.columns(2, |columns| {
            draw_actions(
                &mut columns[0],
                decision,
                reference.as_ref(),
                frame.value_kind,
                &manifest.action_space,
                self.view,
            );
            columns[1].heading("Selected input");
            draw_input(&mut columns[1], input);
            if let Some(state) = &decision.information_state {
                columns[1].label(format!("Information state  {state}"));
            }
            if let Some(history) = input.public_history() {
                columns[1].label(format!("History decisions  {}", history.timestep()));
            }
        });
        self.typed_pane.ready = Some(requested);
        Ok(())
    }

    fn typed_reference(
        &mut self,
        current: &RunIndex,
        input: &DiagnosticInput,
        row_id: u64,
        set: Option<&str>,
    ) -> Result<Option<ReferenceRow>> {
        let Some(run) = self.comparison.reference_run() else {
            return Ok(None);
        };
        let role = self.comparison.reference_role();
        let provider = crate::reference::provider_role(role);
        let same_run = run.root() == current.root();
        let revision = if provider && same_run {
            self.selection.revision
        } else {
            self.comparison.reference_revision()
        };
        if same_run && role == self.selection.model_role && revision == self.selection.revision {
            return Ok(None);
        }
        let a = current.recording().manifest();
        let b = run.recording().manifest();
        if a.observation_space() != b.observation_space() || a.action_space != b.action_space {
            return Ok(None);
        }
        if !same_run
            && manifest_context(a)?
                .zip(manifest_context(b)?)
                .is_none_or(|(a, b)| compatible_contexts(&a, &b).is_err())
        {
            return Ok(None);
        }
        let label = if provider {
            role.trim_start_matches("reference:").into()
        } else {
            format!("k {revision}")
        };
        if let Some(record) = exact(run, RecordKind::FinitePredictions, role, set, revision) {
            let Some(probe) = run.probe_for_record(record) else {
                return Ok(None);
            };
            let Some(loaded) =
                self.typed_pane
                    .reference_inputs
                    .read(run, probe, &mut self.loader)?
            else {
                return Ok(None);
            };
            let TypedPayload::Probes(probes) = loaded.as_ref() else {
                return Ok(None);
            };
            let Some(row) = probes
                .row_ids
                .iter()
                .zip(&probes.inputs)
                .position(|(&id, other)| id == row_id && input.same_input_as(other))
            else {
                return Ok(None);
            };
            let verified = if provider {
                let Some(chunk) = self
                    .loader
                    .request_admitted(run, record, PAYLOAD_ADMISSION)?
                else {
                    return Ok(None);
                };
                let Some(spec) = provider_specification(&chunk)? else {
                    return Ok(None);
                };
                if spec.model_role() != role
                    || spec.information == ReferenceInformation::PrivilegedSimulatorState
                    || !same_run && spec.lifetime == ReferenceLifetime::CheckpointDependent
                    || manifest_context(a)?
                        .is_none_or(|context| compatible_contexts(&context, &spec.context).is_err())
                {
                    return Ok(None);
                }
                spec.target == ReferenceTarget::VerifiedOptimal && spec.context.markov_observation
            } else {
                false
            };
            let Some(ready) = self
                .typed_pane
                .reference
                .read(run, record, &mut self.loader)?
            else {
                return Ok(None);
            };
            let TypedPayload::Predictions(ready) = ready.as_ref() else {
                return Ok(None);
            };
            if ready.value_kind == Some(FiniteValueKind::OptimalActionValue) && !verified {
                return Ok(None);
            }
            let verification = (run.key(record)?, run.key(probe)?);
            if self.typed_pane.verified_reference.as_ref() != Some(&verification) {
                ready.validate_support(
                    &probes.row_ids,
                    probes.inputs.iter().map(DiagnosticInput::observation),
                    &b.action_space,
                )?;
                self.typed_pane.verified_reference = Some(verification);
            }
            return Ok(Some(ReferenceRow {
                label,
                decision: ready.rows[row].clone(),
                value_kind: ready.value_kind,
            }));
        }
        if !provider
            && b.model_id == "sparse-tabular.v1"
            && let Some(record) = exact(run, RecordKind::ModelSnapshot, role, None, revision)
            && let Some(chunk) = self
                .loader
                .request_admitted(run, record, PAYLOAD_ADMISSION)?
            && let Ok(actor) = restore::<rl::core::state_key::DiscreteSymbolKey>(&chunk)
        {
            return Ok(Some(ReferenceRow {
                label: format!("{label} · snapshot"),
                decision: tabular_decision(&actor, row_id, input)?,
                value_kind: Some(FiniteValueKind::ActionValue),
            }));
        }
        Ok(None)
    }

    fn typed_network(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let revision = self.selection.revision;
        let role = &self.selection.model_role;
        let (set, anchor) = match &self.selection.input {
            SelectedInput::ProbeRow { set, sequence, row } => {
                (Some(set.as_str()), Some((*sequence, *row)))
            }
            _ => (None, None),
        };
        let Some(index) = exact(run, RecordKind::Activations, role, set, revision) else {
            ui.weak("No recorded representation at this checkpoint");
            return Ok(());
        };
        let Some(probe) = run.probe_for_record(index) else {
            return Ok(());
        };
        let Some(inputs) = self.typed_pane.inputs.read(run, probe, &mut self.loader)? else {
            return Ok(());
        };
        let TypedPayload::Probes(inputs) = inputs.as_ref() else {
            return Ok(());
        };
        let Some(chunk) = self
            .loader
            .request_admitted(run, index, PAYLOAD_ADMISSION)?
        else {
            return Ok(());
        };
        let Record::Activations(record) = &chunk.record else {
            return Err(Error::Shape("expected a recorded representation".into()));
        };
        let sequence = run.recording().entries()[probe].sequence;
        let mut selected = match anchor {
            None => Some(0),
            Some((source, row)) if source == sequence => Some(row),
            Some((source, row)) => {
                let Some(index) = run.by_sequence(source) else {
                    return Ok(());
                };
                let Some(previous) = self.typed_pane.anchor.read(run, index, &mut self.loader)?
                else {
                    return Ok(());
                };
                let TypedPayload::Probes(previous) = previous.as_ref() else {
                    return Ok(());
                };
                previous
                    .row_ids
                    .get(row)
                    .zip(previous.inputs.get(row))
                    .and_then(|(key, input)| {
                        inputs
                            .row_ids
                            .iter()
                            .zip(&inputs.inputs)
                            .position(|(id, other)| key == id && input.same_input_as(other))
                    })
            }
        }
        .filter(|&row| row < inputs.inputs.len());
        ui.horizontal(|ui| {
            ui.heading("Recorded representation");
            ui.weak(&record.network);
            let mut row = selected.unwrap_or(0);
            if ui
                .add(
                    egui::DragValue::new(&mut row)
                        .range(0..=inputs.inputs.len() - 1)
                        .prefix("Input row "),
                )
                .changed()
            {
                selected = Some(row);
                self.selection.input = SelectedInput::ProbeRow {
                    set: chunk.context.probe_set.clone().unwrap_or_default(),
                    row,
                    sequence,
                };
            }
        });
        let Some(row) = selected else {
            ui.weak("Selected input is absent at this checkpoint");
            return Ok(());
        };
        for (name, key) in record.layers.iter().take(3) {
            let array = crate::plot_data::array(&chunk, key)?;
            let [rows, width] = array.shape() else {
                return Err(Error::Shape("expected a representation matrix".into()));
            };
            if *rows != inputs.inputs.len() || *width == 0 {
                return Err(Error::Shape(
                    "representation rows differ from recorded inputs".into(),
                ));
            }
            let width = *width;
            let values = array.as_f32()?;
            let end = width.min(1024);
            ui.label(name);
            Plot::new(("typed-representation", name))
                .height(216.)
                .x_axis_label("Feature")
                .y_axis_label("Activation")
                .show(ui, |plot| {
                    plot.line(
                        Line::new(
                            name,
                            (0..end)
                                .map(|i| [i as f64, f64::from(values[row * width + i])])
                                .collect::<Vec<_>>(),
                        )
                        .color(CURRENT),
                    );
                });
        }
        draw_input(ui, &inputs.inputs[row]);
        self.typed_pane.ready = Some(revision);
        Ok(())
    }
}
