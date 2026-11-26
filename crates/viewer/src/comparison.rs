//! Reference recording state and source controls shared by Policy and Value.
use crate::{
    data::*,
    reference::provider_role,
    resolver::{ProbeResolver, ResolvedView},
};
use eframe::egui;
use rl::{
    Error, Result,
    diagnostics::{schema::*, storage::ReadLimits},
};
use std::{collections::BTreeSet, path::PathBuf};

pub struct Comparison {
    reference: Option<RunIndex>,
    reference_path: PathBuf,
    revision: u64,
    role: String,
    probe: String,
    reference_records: ProbeResolver,
    reference_anchor: Option<(String, CacheKey)>,
    reference_roles: BTreeSet<String>,
    reference_interactions: u64,
    policy_controller_selected: bool,
    value_critic_override: Option<String>,
    value_policy_override: Option<String>,
    value_empirical: bool,
}

impl Default for Comparison {
    fn default() -> Self {
        Self::new()
    }
}

impl Comparison {
    pub(crate) fn value_empirical(&self) -> bool {
        self.value_empirical
    }
    #[cfg(feature = "screenshots")]
    pub(crate) fn set_value_empirical(&mut self, value: bool) {
        if self.value_empirical != value {
            self.value_empirical = value;
            self.reference_interactions = self.reference_interactions.wrapping_add(1);
        }
    }
    pub(crate) fn value_critic_override(&self) -> Option<&str> {
        self.value_critic_override.as_deref()
    }
    pub(crate) fn value_policy_override(&self) -> Option<&str> {
        self.value_policy_override.as_deref()
    }
    pub(crate) fn value_source_controls(
        &mut self,
        ui: &mut egui::Ui,
        critics: &[String],
        policies: &[String],
    ) {
        let before = (
            self.value_critic_override.clone(),
            self.value_policy_override.clone(),
            self.value_empirical,
        );
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.value_empirical,
                false,
                "Model / recorded reference",
            );
            ui.selectable_value(&mut self.value_empirical, true, "Empirical reference");
        });
        if self.value_empirical {
            ui.small("Independent conditional return evidence at the current critic's actual revision. Only exact inputs and complete first actions are compared; unsupported locations stay empty.");
        }
        egui::CollapsingHeader::new("Reference sources").show(ui, |ui| {
            for (label, names, selected) in [
                ("Value critic", critics, &mut self.value_critic_override),
                ("Policy model", policies, &mut self.value_policy_override),
            ] {
                ui.label(label);
                egui::ComboBox::from_id_salt(("value-reference-source", label))
                    .selected_text(selected.as_deref().unwrap_or("Automatic"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(selected, None, "Automatic");
                        for name in names {
                            ui.selectable_value(selected, Some(name.clone()), name);
                        }
                    });
            }
        });
        if before
            != (
                self.value_critic_override.clone(),
                self.value_policy_override.clone(),
                self.value_empirical,
            )
        {
            self.reference_interactions = self.reference_interactions.wrapping_add(1);
        }
    }
    pub fn reference_role(&self) -> &str {
        &self.role
    }
    pub fn reference_revision(&self) -> u64 {
        self.revision
    }
    pub fn reference_run(&self) -> Option<&RunIndex> {
        self.reference.as_ref()
    }
    /// Changes for explicit reference run/time/role/population edits. Startup
    /// discovery records this generation before it begins any asynchronous read.
    pub(crate) fn reference_interaction_serial(&self) -> u64 {
        self.reference_interactions
    }
    /// A held frame may belong to the previously selected reference. Provenance
    /// captions follow the active resolved request, never that stale publication.
    pub(crate) fn quality_source_is_current(
        &self,
        generation: Option<u64>,
        source: &crate::policy::prepared::SourceIdentity,
    ) -> bool {
        generation == Some(self.reference_interactions)
            && self.reference.as_ref().is_some_and(|run| {
                source.key.root == *run.root()
                    && run.by_sequence(source.key.sequence).is_some_and(|i| {
                        let entry = &run.recording().entries()[i];
                        entry.model_role == self.role
                            && entry.revision == source.revision
                            && source.revision <= self.revision
                    })
            })
    }
    /// Install a discovered local population once, preserving all later manual choices.
    pub(crate) fn use_local_defaults(
        &mut self,
        primary: &RunIndex,
        role: &str,
        input: &SelectedInput,
        revision: u64,
    ) -> Result<()> {
        if self
            .reference
            .as_ref()
            .is_none_or(|reference| reference.root() != primary.root())
        {
            return Err(Error::State(
                "local defaults cannot change an external reference recording".into(),
            ));
        }
        self.role = role.into();
        if let SelectedInput::ProbeRow { set, sequence, .. } = input {
            self.probe.clone_from(set);
            self.reference_anchor = self.reference.as_ref().map(|run| {
                (
                    set.clone(),
                    CacheKey {
                        root: run.root().clone(),
                        sequence: *sequence,
                    },
                )
            });
        }
        self.set_reference_revision(revision)?;
        self.prefer_unique_provider()?;
        Ok(())
    }
    fn prefer_unique_provider(&mut self) -> Result<()> {
        let mut providers = self
            .reference_roles
            .iter()
            .filter(|role| provider_role(role));
        let provider = providers.next().cloned();
        if providers.next().is_none()
            && let Some(provider) = provider
        {
            self.set_reference_role(&provider)?;
        }
        Ok(())
    }
    pub fn new() -> Self {
        Self {
            reference: None,
            reference_path: PathBuf::new(),
            revision: 0,
            role: "online".into(),
            probe: String::new(),
            reference_records: ProbeResolver::new(crate::budgets::REFERENCE_RESOLVER),
            reference_anchor: None,
            reference_roles: BTreeSet::new(),
            reference_interactions: 0,
            policy_controller_selected: false,
            value_critic_override: None,
            value_policy_override: None,
            value_empirical: false,
        }
    }
    /// Retry reference reads without changing the user's source selection.
    pub fn invalidate(&mut self) {
        self.reference_records.clear();
    }
    /// Release records retained by reference-controller resolution.
    pub fn clear_recording(&mut self) {
        self.invalidate();
    }
    /// Retained charges for the reference-controller resolver.
    pub fn resident_bytes(&self) -> usize {
        self.reference_records.resident_bytes()
    }
    pub(crate) fn open(&mut self, path: PathBuf) -> Result<()> {
        let index = RunIndex::open(&path, ReadLimits::default())?;
        self.install(index, path);
        Ok(())
    }
    fn install(&mut self, index: RunIndex, path: PathBuf) {
        self.reference_interactions = self.reference_interactions.wrapping_add(1);
        self.revision = index
            .recording()
            .entries()
            .iter()
            .filter(|e| matches!(e.kind, RecordKind::Policy | RecordKind::FinitePredictions))
            .map(|e| e.revision)
            .min()
            .or_else(|| {
                index
                    .recording()
                    .entries()
                    .iter()
                    .filter(|e| e.kind == RecordKind::Activations)
                    .map(|e| e.revision)
                    .min()
            })
            .or_else(|| index.revisions().first().copied())
            .unwrap_or(0);
        self.value_critic_override = None;
        self.value_policy_override = None;
        self.reference = Some(index);
        self.reference_path = path;
        self.reference_anchor = None;
        self.reference_records.clear();
        self.reference_roles = self
            .reference
            .as_ref()
            .unwrap()
            .recording()
            .entries()
            .iter()
            .map(|entry| entry.model_role.clone())
            .collect();
        if !self.reference_roles.contains(&self.role) {
            self.role = self
                .reference_roles
                .get("online")
                .or_else(|| self.reference_roles.first())
                .cloned()
                .unwrap_or_else(|| "online".into());
        }
        if provider_role(&self.role) {
            self.probe.clear();
        }
    }
    /// Opening a different main run starts a local reference. Changing expert
    /// tabs or learner revision does not call this and preserves that reference.
    pub fn reset_reference_to(&mut self, run: &RunIndex) -> Result<()> {
        self.install(run.clone(), run.root().as_ref().clone());
        self.role = run
            .recording()
            .entries()
            .iter()
            .find(|e| e.model_role == "online")
            .or_else(|| {
                run.recording()
                    .entries()
                    .iter()
                    .find(|e| matches!(e.kind, RecordKind::Policy | RecordKind::Activations))
            })
            .map_or_else(|| "online".into(), |e| e.model_role.clone());
        if let Some(revision) = run
            .recording()
            .entries()
            .iter()
            .filter(|e| e.kind == RecordKind::Activations && e.model_role == self.role)
            .map(|e| e.revision)
            .min()
        {
            self.revision = revision;
        }
        self.probe.clear();
        // Initial local defaults are applied only when opening a different run.
        self.prefer_unique_provider()?;
        Ok(())
    }
    /// Select any learner revision within the opened reference run. The selected
    /// reference input stays anchored while its modalities resolve independently.
    pub fn set_reference_revision(&mut self, revision: u64) -> Result<()> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| Error::State("no reference recording is open".into()))?;
        let mut selection = Selection::default();
        selection.select_revision(reference, revision)?;
        self.revision = revision;
        self.reference_interactions = self.reference_interactions.wrapping_add(1);
        Ok(())
    }
    pub(crate) fn set_reference_role(&mut self, role: &str) -> Result<()> {
        if !self.reference_roles.contains(role) {
            return Err(Error::Config("Unknown reference role".into()));
        }
        if self.role == role {
            return Ok(());
        }
        self.role = role.into();
        self.reference_anchor = None;
        self.reference_records.clear();
        self.value_critic_override = None;
        self.value_policy_override = None;
        let reference = self.reference.as_ref().unwrap();
        let entries = reference.recording().entries();
        self.probe = if provider_role(role) {
            if let Some(first) = entries
                .iter()
                .filter(|e| {
                    e.model_role == role
                        && matches!(
                            e.kind,
                            RecordKind::Policy | RecordKind::Critic | RecordKind::FinitePredictions
                        )
                })
                .map(|e| e.revision)
                .min()
            {
                self.revision = self.revision.max(first);
            }
            None
        } else {
            entries
                .iter()
                .find(|e| {
                    e.model_role == role
                        && matches!(e.kind, RecordKind::ProbeSet | RecordKind::TypedProbes)
                })
                .and_then(|e| e.probe_set.clone())
        }
        .unwrap_or_default();
        self.reference_interactions = self.reference_interactions.wrapping_add(1);
        Ok(())
    }
    pub(crate) fn policy_uses_controller(&self) -> bool {
        self.policy_controller_selected || provider_role(&self.role)
    }
    #[cfg(feature = "screenshots")]
    pub(crate) fn configure_policy_reference(
        &mut self,
        path: Option<std::path::PathBuf>,
        revision: Option<u64>,
        controller: bool,
        role: Option<&str>,
    ) -> Result<()> {
        if let Some(path) = path {
            self.open(path)?;
        }
        if let Some(role) = role {
            self.set_reference_role(role)?;
        }
        if let Some(revision) = revision {
            self.set_reference_revision(revision)?;
        }
        self.policy_controller_selected = controller;
        self.reference_interactions = self.reference_interactions.wrapping_add(1);
        Ok(())
    }
    pub(crate) fn reference_controls(
        &mut self,
        ui: &mut egui::Ui,
        catalogue: &[CatalogEntry],
        recorded: bool,
        typed: bool,
        reference_follows_current: bool,
    ) -> Result<()> {
        if self.reference.is_none() {
            return Ok(());
        }
        let mut chosen = None;
        let mut revision = self.revision;
        ui.horizontal_wrapped(|ui| {
            ui.label("Run");
            egui::ComboBox::from_id_salt("comparison-run")
                .selected_text(crate::presentation::display_identifier(
                    &self
                        .reference
                        .as_ref()
                        .unwrap()
                        .recording()
                        .manifest()
                        .run_id,
                ))
                .show_ui(ui, |ui| {
                    for c in catalogue {
                        if ui
                            .selectable_label(
                                self.reference_path == c.root,
                                crate::presentation::display_identifier(&c.manifest.run_id),
                            )
                            .on_hover_text(&c.manifest.run_id)
                            .clicked()
                        {
                            chosen = Some(c.root.clone());
                        }
                    }
                });
            if !reference_follows_current {
                ui.label("k");
                let revisions = self.reference.as_ref().unwrap().revisions();
                if let Some((&first, &last)) = revisions.first().zip(revisions.last()) {
                    ui.add(egui::Slider::new(&mut revision, first..=last));
                }
            }
        });
        if let Some(path) = chosen {
            self.open(path)?;
            revision = self.revision;
        }
        // Keep requested time continuous; modalities resolve their own actual
        // predecessors. Changing time preserves the immutable reference input.
        if revision != self.revision {
            self.set_reference_revision(revision)?;
        }
        let mut chosen_role = None;
        ui.horizontal_wrapped(|ui| {
            if self.reference_roles.len() > 1 {
                egui::ComboBox::from_id_salt("reference-role")
                    .selected_text(self.role.strip_prefix("reference:").map_or_else(
                        || self.role.clone(),
                        crate::presentation::display_identifier,
                    ))
                    .show_ui(ui, |ui| {
                        for role in &self.reference_roles {
                            let label = role.strip_prefix("reference:").map_or_else(
                                || role.clone(),
                                crate::presentation::display_identifier,
                            );
                            if ui
                                .selectable_label(&self.role == role, label)
                                .on_hover_text(role)
                                .clicked()
                            {
                                chosen_role = Some(role.clone());
                            }
                        }
                    });
            }
            if !typed {
                let provider = provider_role(&self.role);
                egui::ComboBox::from_id_salt("reference-probes")
                    .selected_text(if provider && self.probe.is_empty() {
                        "Automatic"
                    } else {
                        &self.probe
                    })
                    .show_ui(ui, |ui| {
                        if provider
                            && ui
                                .selectable_value(&mut self.probe, String::new(), "Automatic")
                                .changed()
                        {
                            self.reference_interactions =
                                self.reference_interactions.wrapping_add(1);
                        }
                        for id in self.reference.as_ref().unwrap().probe_set_ids() {
                            if ui
                                .selectable_value(&mut self.probe, id.into(), id)
                                .clicked()
                            {
                                self.reference_interactions =
                                    self.reference_interactions.wrapping_add(1);
                            }
                        }
                    });
            }
            if !typed && !provider_role(&self.role) {
                egui::ComboBox::from_id_salt("policy-reference-source")
                    .selected_text(if self.policy_controller_selected {
                        "Recorded controller"
                    } else if recorded {
                        "Recorded policy"
                    } else {
                        "Historical model"
                    })
                    .show_ui(ui, |ui| {
                        for (value, label) in
                            [(false, "Historical model"), (true, "Recorded controller")]
                        {
                            if ui
                                .selectable_value(
                                    &mut self.policy_controller_selected,
                                    value,
                                    label,
                                )
                                .changed()
                            {
                                self.reference_interactions =
                                    self.reference_interactions.wrapping_add(1);
                            }
                        }
                    });
            }
        });
        if let Some(role) = chosen_role {
            self.set_reference_role(&role)?;
        }
        Ok(())
    }
    /// Resolve the explicitly selected reference population as a controller.
    /// Physical input/conditioning matching belongs to the Policy worker, which
    /// preserves this population's independent row and source identity.
    pub(crate) fn policy_controller(
        &mut self,
        loader: &mut Loader,
        current_input: Option<&SelectedInput>,
    ) -> Result<ResolvedView> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| Error::State("No reference recording".into()))?;
        let probe = self
            .reference_anchor
            .as_ref()
            .filter(|(id, _)| id == &self.probe)
            .map(|(id, key)| (id, key.sequence))
            .or_else(|| {
                let index = if provider_role(&self.role) {
                    let find =
                        |set| provider_policy_probe(reference, &self.role, set, self.revision);
                    if self.probe.is_empty() {
                        current_input
                            .and_then(|input| match input {
                                SelectedInput::ProbeRow { set, .. } => find(Some(set.as_str())),
                                _ => None,
                            })
                            .or_else(|| find(None))
                    } else {
                        find(Some(&self.probe))
                    }
                } else {
                    reference
                        .candidates(
                            RecordKind::ProbeSet,
                            &self.role,
                            Some(&self.probe),
                            self.revision,
                        )
                        .next()
                };
                let entry = &reference.recording().entries()[index?];
                entry.probe_set.as_ref().map(|set| (set, entry.sequence))
            });
        let input = probe.map_or(SelectedInput::None, |(set, sequence)| {
            SelectedInput::ProbeRow {
                set: set.clone(),
                row: 0,
                sequence,
            }
        });
        let selection = Selection {
            revision: self.revision,
            model_role: self.role.clone(),
            input,
        };
        self.reference_records
            .resolve_policy(reference, &selection, |i| loader.request(reference, i))
    }
}

fn provider_policy_probe(
    run: &RunIndex,
    role: &str,
    set: Option<&str>,
    revision: u64,
) -> Option<usize> {
    run.candidates(RecordKind::Policy, role, set, revision)
        .take(crate::policy::prepared::MAX_SOURCE_CANDIDATES)
        .filter(|&index| run.recording().is_retained(index))
        .filter_map(|index| run.probe_for_record(index))
        .find(|&index| run.recording().is_retained(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolver::ResolveStatus;
    use rl::{
        core::{action::ActionSpace, env::Observation},
        diagnostics::{probes::ProbeSet, storage::*},
    };
    use std::{collections::BTreeMap, sync::Arc};

    fn fixture() -> (tempfile::TempDir, RunIndex, Vec<Arc<Chunk>>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reference");
        let mut writer = Recorder::create(
            &path,
            RunManifest::new(
                "reference",
                "test",
                "ppo",
                1,
                ActionSpace::Discrete { count: 2 },
                0.9,
            )
            .unwrap(),
            StorageConfig::default(),
        )
        .unwrap();
        let mut chunks = Vec::new();
        for (revision, reversed) in [(1, false), (7, true), (11, false)] {
            let mut context = RecordContext::new("reference", revision);
            context.probe_set = Some("fixed".into());
            let mut probes = ProbeSet::new(
                "fixed",
                ProbePopulation::FixedReference,
                "stable grid",
                if revision == 11 {
                    [999., 888.]
                } else if reversed {
                    [2., 1.]
                } else {
                    [1., 2.]
                }
                .into_iter()
                .map(|value| Observation::new(vec![value]))
                .collect(),
                None,
                ActionSpace::Discrete { count: 2 },
            )
            .unwrap()
            .record(context)
            .unwrap();
            if reversed {
                let Record::ProbeSet(meta) = &probes.record else {
                    unreachable!()
                };
                probes.arrays.insert(
                    meta.row_ids.clone(),
                    Array::u64(vec![2], vec![1, 0]).unwrap(),
                );
            }
            chunks.push(Arc::new(probes));
            if revision != 11 {
                let mut context = RecordContext::new("reference", revision + 2);
                context.probe_set = Some("fixed".into());
                chunks.push(Arc::new(Chunk {
                    context,
                    record: Record::Policy(PolicyRecord {
                        actions: "actions".into(),
                        parameters: PolicyParameters::Deterministic,
                        entropy: None,
                        entropy_method: None,
                        masks: None,
                        stochastic: false,
                    }),
                    arrays: BTreeMap::from([(
                        "actions".into(),
                        Array::u64(vec![2], vec![0, 1]).unwrap(),
                    )]),
                }));
            }
        }
        chunks.push(Arc::new(Chunk {
            context: RecordContext::new("reference", 13),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        }));
        for chunk in &chunks {
            writer.append(chunk).unwrap();
        }
        writer.finish().unwrap();
        (
            dir,
            RunIndex::open(path, ReadLimits::default()).unwrap(),
            chunks,
        )
    }

    fn anchored_reference(run: &RunIndex) -> Comparison {
        let mut comparison = Comparison::new();
        comparison.reset_reference_to(run).unwrap();
        comparison
            .use_local_defaults(
                run,
                "online",
                &SelectedInput::ProbeRow {
                    set: "fixed".into(),
                    sequence: 0,
                    row: 1,
                },
                3,
            )
            .unwrap();
        comparison
    }

    #[test]
    fn provider_controller_uses_its_linked_probe_even_when_the_learner_authored_it() {
        let (_directory, fixture, chunks) = fixture();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("provider");
        let mut writer = Recorder::create(
            &path,
            fixture.recording().manifest().clone(),
            StorageConfig::default(),
        )
        .unwrap();
        writer.append(&chunks[0]).unwrap();
        let mut provider = chunks[1].as_ref().clone();
        provider.context.identity.model_role = "reference:benchmark".into();
        writer.append(&provider).unwrap();
        writer.finish().unwrap();

        let run = RunIndex::open(path, ReadLimits::default()).unwrap();
        let mut comparison = Comparison::new();
        comparison.reset_reference_to(&run).unwrap();
        assert_eq!(comparison.reference_role(), "reference:benchmark");
        comparison
            .use_local_defaults(&run, "online", &SelectedInput::None, 3)
            .unwrap();
        assert_eq!(comparison.reference_role(), "reference:benchmark");
        comparison
            .set_reference_role("reference:benchmark")
            .unwrap();
        let mut loader = Loader::new(crate::budgets::REFERENCE_RESOLVER, || {}).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            loader.poll();
            match comparison.policy_controller(&mut loader, None).unwrap() {
                ResolvedView {
                    status: ResolveStatus::Ready,
                    record: Some(record),
                    ..
                } => {
                    assert_eq!(record.probe_key.sequence, 0);
                    assert_eq!(record.revision(), 3);
                    assert_eq!(
                        record.chunk.context.identity.model_role,
                        "reference:benchmark"
                    );
                    break;
                }
                ResolvedView {
                    status: ResolveStatus::Loading,
                    ..
                } => {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                other => panic!("{:?}: {:?}", other.status, other.warnings),
            }
        }
    }

    #[test]
    fn reference_time_is_continuous_and_invalid_requests_leave_state_unchanged() {
        let (_dir, run, _) = fixture();
        let mut comparison = Comparison::new();
        assert!(comparison.reference_run().is_none());
        assert!(comparison.set_reference_revision(3).is_err());
        assert_eq!(comparison.reference_interaction_serial(), 0);
        comparison.reset_reference_to(&run).unwrap();
        assert_eq!(comparison.reference_revision(), 3);
        assert_eq!(comparison.reference_role(), "online");
        assert_eq!(comparison.reference_run().unwrap().root(), run.root());
        let serial = comparison.reference_interaction_serial();
        for invalid in [0, 14] {
            assert!(comparison.set_reference_revision(invalid).is_err());
            assert_eq!(comparison.reference_revision(), 3);
            assert_eq!(comparison.reference_interaction_serial(), serial);
        }
        comparison.set_reference_revision(5).unwrap();
        assert_eq!(comparison.reference_revision(), 5);
        assert_eq!(comparison.reference_interaction_serial(), serial + 1);
        comparison.set_reference_revision(5).unwrap();
        assert_eq!(comparison.reference_interaction_serial(), serial + 2);
    }

    #[test]
    fn controller_preserves_anchor_and_independent_actual_source_across_sparse_time() {
        let (_dir, run, chunks) = fixture();
        let mut comparison = anchored_reference(&run);
        let anchor = comparison.reference_anchor.clone();
        let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
        for (index, chunk) in chunks.iter().enumerate() {
            loader.cache.insert(run.key(index).unwrap(), chunk.clone());
        }
        for (requested, actual, row, definition) in [
            (3, 3, 0, 0),
            (5, 3, 0, 0),
            (9, 9, 1, 2),
            (12, 9, 1, 2),
            (3, 3, 0, 0),
        ] {
            comparison.set_reference_revision(requested).unwrap();
            let view = comparison.policy_controller(&mut loader, None).unwrap();
            assert_eq!(view.status, ResolveStatus::Ready);
            assert_eq!(view.requested_revision, requested);
            let record = view.record.unwrap();
            assert_eq!(
                (record.revision(), record.row, record.probe_key.sequence),
                (actual, row, definition)
            );
            assert_eq!(record.key.root, *run.root());
            assert_eq!(comparison.reference_anchor, anchor);
            assert_eq!(comparison.reference_revision(), requested);
        }
    }

    #[test]
    fn retry_preserves_reference_choices_and_external_frame_charges() {
        let (_dir, run, chunks) = fixture();
        let mut comparison = anchored_reference(&run);
        comparison.policy_controller_selected = true;
        comparison.value_critic_override = Some("critic-b".into());
        comparison.value_policy_override = Some("policy-b".into());
        comparison.set_reference_revision(12).unwrap();
        let serial = comparison.reference_interaction_serial();
        let anchor = comparison.reference_anchor.clone();
        let mut loader = Loader::new(1024 * 1024, || {}).unwrap();
        for (index, chunk) in chunks.iter().enumerate() {
            loader.cache.insert(run.key(index).unwrap(), chunk.clone());
        }
        let view = comparison.policy_controller(&mut loader, None).unwrap();
        assert_eq!(view.status, ResolveStatus::Ready);
        assert!(comparison.resident_bytes() > 0);
        comparison.invalidate();
        assert!(
            comparison.resident_bytes() > 0,
            "caller-held controller frames remain charged"
        );
        drop(view);
        assert_eq!(comparison.resident_bytes(), 0);
        assert_eq!(comparison.reference_run().unwrap().root(), run.root());
        assert_eq!(comparison.reference_revision(), 12);
        assert_eq!(comparison.reference_role(), "online");
        assert_eq!(comparison.reference_anchor, anchor);
        assert_eq!(comparison.reference_interaction_serial(), serial);
        assert!(comparison.policy_uses_controller());
        assert_eq!(comparison.value_critic_override(), Some("critic-b"));
        assert_eq!(comparison.value_policy_override(), Some("policy-b"));
        let retried = comparison.policy_controller(&mut loader, None).unwrap();
        assert_eq!(retried.status, ResolveStatus::Ready);
        assert_eq!(retried.record.as_ref().unwrap().revision(), 9);
        drop(retried);
        comparison.clear_recording();
        assert_eq!(comparison.resident_bytes(), 0);
        assert_eq!(comparison.reference_interaction_serial(), serial);
    }

    #[test]
    fn defaults_cannot_overwrite_external_reference_and_reset_replaces_its_anchor() {
        let (_dir, run, _) = fixture();
        let (_other_dir, other, _) = fixture();
        let mut comparison = anchored_reference(&run);
        comparison.value_critic_override = Some("critic-b".into());
        comparison.value_policy_override = Some("policy-b".into());
        comparison.policy_controller_selected = true;
        let anchor = comparison.reference_anchor.clone();
        let serial = comparison.reference_interaction_serial();
        assert!(
            comparison
                .use_local_defaults(&other, "wrong-role", &SelectedInput::None, 7)
                .is_err()
        );
        assert_eq!(comparison.reference_run().unwrap().root(), run.root());
        assert_eq!(comparison.reference_role(), "online");
        assert_eq!(comparison.reference_revision(), 3);
        assert_eq!(comparison.reference_anchor, anchor);
        assert_eq!(comparison.reference_interaction_serial(), serial);
        comparison.reset_reference_to(&other).unwrap();
        assert_eq!(comparison.reference_run().unwrap().root(), other.root());
        assert_eq!(comparison.reference_revision(), 3);
        assert!(comparison.reference_anchor.is_none());
        assert!(comparison.probe.is_empty());
        assert_eq!(comparison.value_critic_override(), None);
        assert_eq!(comparison.value_policy_override(), None);
        assert!(comparison.policy_uses_controller());
        assert_eq!(comparison.reference_interaction_serial(), serial + 1);
    }
}
