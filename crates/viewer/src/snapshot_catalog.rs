//! Bounded exact snapshot discovery. Only restored artifacts enter playback.
use crate::{
    data::{CacheKey, Loader, RunIndex},
    interactive::Interactive,
    network::selection::{NetworkChoice, choices},
    policy::prepared::{PolicySource, PolicyViewLease, SourceIdentity, selected_name},
    value::prepared::ValueSource,
};
use rl::{
    Error, Result,
    diagnostics::{
        models::ModelKind,
        schema::{Chunk, RecordKind},
        value::{ValueCriticKind, ValueOutput},
    },
};
use std::sync::{Arc, Mutex, Weak};

pub const MAX_SNAPSHOTS: usize = 256;
#[cfg(test)]
use crate::network_test_fixture as reuse_fixture;
#[cfg(test)]
#[path = "snapshot_catalog_reuse_tests.rs"]
mod reuse_tests;
pub(crate) const CATALOG_BYTES: usize = 256 * 1024;
#[derive(Clone, Debug, PartialEq)]
pub enum SnapshotPanel {
    Policy,
    Network {
        name: Option<String>,
    },
    Value {
        name: Option<String>,
        twin: Option<String>,
        actor_required: bool,
        output: ValueOutput,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SnapshotStatus {
    Pending,
    Eligible,
    Rejected(String),
}
#[derive(Clone, Debug)]
pub struct SnapshotEntry {
    pub key: CacheKey,
    pub revision: u64,
    pub role: String,
    pub name: String,
    pub status: SnapshotStatus,
}
pub struct SnapshotCatalog {
    panel: SnapshotPanel,
    root: std::sync::Arc<std::path::PathBuf>,
    limited: bool,
    role: String,
    entries: Vec<SnapshotEntry>,
    cursor: usize,
    priority_revision: Option<u64>,
    preferred_source: Option<CacheKey>,
    indexed_until: usize,
    known_entries: usize,
    validating: Option<usize>,
    networks: Vec<NetworkChoice>,
    // Only a small identity witness is retained, within CATALOG_BYTES. Raw
    // chunks and source metadata remain owned/charged by their existing owners.
    value_reuse: Mutex<Option<ValueReuse>>,
    // Weak witnesses never retain raw snapshots or models. They prove that the
    // same cached decoded allocation accompanied a successful whole-bundle
    // restoration, and are bounded by the existing catalogue entry limit.
    restored: Vec<RestoredWitness>,
    pending_witness: Option<RestoredWitness>,
    parked: bool,
    _lease: PolicyViewLease,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct SnapshotStamp {
    bytes: u64,
    checksum: u32,
    page: Option<(u64, u64)>,
    file_bytes: u64,
    modified: std::time::SystemTime,
}
impl SnapshotStamp {
    fn read(run: &RunIndex, index: usize) -> Option<Self> {
        let entry = run.recording().entries().get(index)?;
        if !run.has_model_record(index) {
            return None;
        }
        let metadata = std::fs::metadata(run.root().join(&entry.file)).ok()?;
        Some(Self {
            bytes: entry.bytes,
            checksum: entry.checksum,
            page: entry.page.as_ref().map(|p| (p.first_sequence, p.offset)),
            file_bytes: metadata.len(),
            modified: metadata.modified().ok()?,
        })
    }
}
struct RestoredWitness {
    key: CacheKey,
    chunk: Weak<Chunk>,
    stamp: SnapshotStamp,
}
struct SnapshotCandidate {
    name: String,
    chunk: Arc<Chunk>,
    stamp: Option<SnapshotStamp>,
}
impl RestoredWitness {
    fn matches(&self, key: &CacheKey, chunk: &Arc<Chunk>, stamp: &SnapshotStamp) -> bool {
        self.key == *key
            && self.stamp == *stamp
            && self
                .chunk
                .upgrade()
                .is_some_and(|old| Arc::ptr_eq(&old, chunk))
    }
}
struct ValueReuse {
    chunk: Weak<Chunk>,
    panel: SnapshotPanel,
    identity: SourceIdentity,
    names: Vec<String>,
    actor: Option<String>,
    actor_stochastic: bool,
    kind: ValueCriticKind,
    heads: usize,
    role: String,
}
impl ValueReuse {
    fn new(chunk: &Arc<Chunk>, panel: &SnapshotPanel, source: &ValueSource) -> Self {
        Self {
            chunk: Arc::downgrade(chunk),
            panel: panel.clone(),
            identity: source.identity.clone(),
            names: source.names.clone(),
            actor: source.actor.clone(),
            actor_stochastic: source.actor_stochastic,
            kind: source.kind,
            heads: source.heads,
            role: source.role.clone(),
        }
    }
    fn matches(&self, chunk: &Arc<Chunk>, panel: &SnapshotPanel, source: &ValueSource) -> bool {
        self.chunk
            .upgrade()
            .is_some_and(|old| Arc::ptr_eq(&old, chunk))
            && self.panel == *panel
            && self.identity == source.identity
            && self.names == source.names
            && self.actor == source.actor
            && self.actor_stochastic == source.actor_stochastic
            && self.kind == source.kind
            && self.heads == source.heads
            && self.role == source.role
    }
}
impl SnapshotCatalog {
    pub fn new(
        run: &RunIndex,
        role: &str,
        panel: SnapshotPanel,
        interactive: &Interactive,
    ) -> Result<Self> {
        if role.len() > 128 {
            return Err(Error::Config("snapshot role exceeds admission".into()));
        }
        if let SnapshotPanel::Network { name: Some(name) } = &panel
            && name.len() > 128
        {
            return Err(Error::Config(
                "snapshot network name exceeds admission".into(),
            ));
        }
        if let SnapshotPanel::Value { name, twin, .. } = &panel
            && name.iter().chain(twin).any(|s| s.len() > 128)
        {
            return Err(Error::Config(
                "snapshot model selection exceeds admission".into(),
            ));
        }
        let lease = interactive.reserve_policy_view(CATALOG_BYTES)?;
        let mut entries = Vec::new();
        let mut limited = false;
        // Earliest bounded window; newest sequence first within each revision.
        'revisions: for &revision in run.snapshot_revisions(role) {
            for index in run
                .candidates(RecordKind::ModelSnapshot, role, None, revision)
                .take_while(|&i| run.recording().entries()[i].revision == revision)
            {
                if entries.len() == MAX_SNAPSHOTS {
                    limited = true;
                    break 'revisions;
                }
                entries.push(SnapshotEntry {
                    key: run.key(index)?,
                    revision,
                    role: role.into(),
                    name: String::new(),
                    status: SnapshotStatus::Pending,
                });
            }
        }
        Ok(Self {
            panel,
            root: run.root().clone(),
            limited,
            role: role.into(),
            entries,
            cursor: 0,
            priority_revision: None,
            preferred_source: None,
            indexed_until: run.recording().entries().len(),
            known_entries: run.recording().entries().len(),
            validating: None,
            networks: Vec::new(),
            value_reuse: Mutex::new(None),
            restored: Vec::new(),
            pending_witness: None,
            parked: false,
            _lease: lease,
        })
    }
    pub fn matches(&self, run: &RunIndex, role: &str, panel: &SnapshotPanel) -> bool {
        self.role == role && &self.panel == panel && self.root == *run.root()
    }
    /// Hide prior panel eligibility immediately while the next workspace
    /// discovers its exact model choice. No old position may be relabelled.
    pub(crate) fn park_for_retarget(&mut self) -> bool {
        if self.pending()
            || !matches!(
                self.panel,
                SnapshotPanel::Policy | SnapshotPanel::Value { .. }
            )
        {
            return false;
        }
        for entry in &mut self.entries {
            entry.name.clear();
            entry.status = SnapshotStatus::Pending;
        }
        self.cursor = 0;
        self.priority_revision = None;
        self.preferred_source = None;
        self.parked = true;
        *self
            .value_reuse
            .get_mut()
            .unwrap_or_else(|e| e.into_inner()) = None;
        true
    }
    /// Retarget a settled or parked catalogue without keeping another
    /// catalogue or any model alive. New capability checks are repeated.
    pub(crate) fn retarget(&mut self, run: &RunIndex, role: &str, panel: SnapshotPanel) -> bool {
        let compatible = matches!(
            (&self.panel, &panel),
            (SnapshotPanel::Policy, SnapshotPanel::Value { .. })
                | (SnapshotPanel::Value { .. }, SnapshotPanel::Policy)
        );
        if !compatible
            || (!self.parked && self.pending())
            || self.root != *run.root()
            || self.role != role
        {
            return false;
        }
        self.panel = panel;
        for entry in &mut self.entries {
            entry.name.clear();
            entry.status = SnapshotStatus::Pending;
        }
        self.cursor = 0;
        self.priority_revision = None;
        self.preferred_source = None;
        self.networks.clear();
        self.parked = false;
        *self
            .value_reuse
            .get_mut()
            .unwrap_or_else(|e| e.into_inner()) = None;
        true
    }
    pub fn limited(&self) -> bool {
        self.limited
    }
    pub fn entries(&self) -> &[SnapshotEntry] {
        &self.entries
    }
    pub fn networks(&self) -> &[NetworkChoice] {
        &self.networks
    }
    /// Bounded retention refresh, called at the live-recording refresh cadence.
    /// A pruned identity never silently becomes a different sibling.
    pub fn recheck_availability(&mut self, run: &RunIndex) {
        for entry in &mut self.entries {
            if entry.status == SnapshotStatus::Eligible
                && run
                    .exact_snapshot_index(&entry.key, entry.revision, &entry.role)
                    .is_err()
            {
                entry.status = SnapshotStatus::Rejected("snapshot missing or retained away".into());
            }
        }
    }
    pub fn pending(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.status == SnapshotStatus::Pending)
            || self.validating.is_some()
            || self.indexed_until < self.known_entries
    }
    pub fn positions(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|e| e.status == SnapshotStatus::Eligible)
            .map(|e| e.revision)
            .collect()
    }
    pub fn selected(&self, revision: u64) -> Option<&SnapshotEntry> {
        if let Some(entry) = self
            .entries
            .iter()
            .find(|e| Some(&e.key) == self.preferred_source.as_ref() && e.revision == revision)
        {
            return (entry.status == SnapshotStatus::Eligible).then_some(entry);
        }
        self.entries
            .iter()
            .find(|e| e.revision == revision && e.status == SnapshotStatus::Eligible)
    }
    /// Preserve an explicitly selected Prepared source on entry to Interactive,
    /// while retaining the historical 256-entry admission and validation path.
    pub(crate) fn prefer_policy_source(
        &mut self,
        run: &RunIndex,
        source: &SourceIdentity,
    ) -> Result<()> {
        if !matches!(self.panel, SnapshotPanel::Policy) || self.root != source.key.root {
            return Ok(());
        }
        let Some(index) = run.by_sequence(source.key.sequence) else {
            return Ok(());
        };
        let entry = &run.recording().entries()[index];
        if entry.kind != RecordKind::ModelSnapshot
            || entry.model_role != self.role
            || entry.revision != source.revision
        {
            return Ok(());
        }
        if self.preferred_source.as_ref() == Some(&source.key) {
            return Ok(());
        }
        if self.validating.is_some() {
            return Err(Error::State(
                "cannot change exact policy source during snapshot validation".into(),
            ));
        }
        if !self.entries.iter().any(|e| e.key == source.key) {
            if self.entries.len() == MAX_SNAPSHOTS {
                self.entries.pop();
                self.limited = true;
            }
            self.entries.push(SnapshotEntry {
                key: source.key.clone(),
                revision: source.revision,
                role: entry.model_role.clone(),
                name: source.name.clone(),
                status: SnapshotStatus::Pending,
            });
        }
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == source.key) {
            entry.name = source.name.clone();
        }
        self.preferred_source = Some(source.key.clone());
        Ok(())
    }
    pub fn policy_source(&self, run: &RunIndex, revision: u64) -> Result<Option<PolicySource>> {
        self.policy_source_reusing(run, revision, None)
    }
    pub(crate) fn policy_source_reusing(
        &self,
        run: &RunIndex,
        revision: u64,
        previous: Option<&PolicySource>,
    ) -> Result<Option<PolicySource>> {
        self.selected(revision)
            .map(|e| {
                let index = run.exact_snapshot_index(&e.key, e.revision, &e.role)?;
                if let Some(source) = previous.filter(|source| {
                    let identity = source.identity();
                    identity.key == e.key
                        && identity.revision == e.revision
                        && identity.name == e.name
                        && !identity.recorded
                        && !identity.explicit_controller
                }) {
                    return Ok(source.clone());
                }
                PolicySource::model(run, index, Some(&e.name))
            })
            .transpose()
    }
    pub fn value_source(
        &self,
        run: &RunIndex,
        revision: u64,
        loader: &mut Loader,
    ) -> Result<Option<ValueSource>> {
        self.value_source_reusing(run, revision, loader, None)
    }
    pub(crate) fn value_source_reusing(
        &self,
        run: &RunIndex,
        revision: u64,
        loader: &mut Loader,
        previous: Option<&ValueSource>,
    ) -> Result<Option<ValueSource>> {
        let Some(e) = self.selected(revision) else {
            return Ok(None);
        };
        let index = run.exact_snapshot_index(&e.key, e.revision, &e.role)?;
        let Some(chunk) = loader.request(run, index)? else {
            return Ok(None);
        };
        if let Some(source) = previous.filter(|source| {
            source.identity.key == e.key
                && source.identity.name == e.name
                && source.identity.revision == e.revision
                && !source.identity.recorded
                && !source.identity.explicit_controller
        }) {
            let witness = self
                .value_reuse
                .lock()
                .map_err(|_| Error::State("Value metadata reuse unavailable".into()))?;
            if witness
                .as_ref()
                .is_some_and(|w| w.matches(&chunk, &self.panel, source))
            {
                return Ok(Some(source.clone()));
            }
        }
        let actor_required = matches!(
            self.panel,
            SnapshotPanel::Value {
                actor_required: true,
                ..
            }
        );
        let mut source =
            ValueSource::model_for_conditioning(run, index, &e.name, &chunk, actor_required)?;
        if let SnapshotPanel::Value {
            twin: Some(group), ..
        } = &self.panel
        {
            source = source.with_twin(group)?;
        }
        *self
            .value_reuse
            .lock()
            .map_err(|_| Error::State("Value metadata reuse unavailable".into()))? =
            Some(ValueReuse::new(&chunk, &self.panel, &source));
        Ok(Some(source))
    }
    pub(crate) fn prioritize_revision(&mut self, revision: u64) {
        self.priority_revision = Some(revision);
    }
    pub(crate) fn retry_rejected(&mut self) {
        *self
            .value_reuse
            .get_mut()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.restored.retain(|witness| {
            !self.entries.iter().any(|entry| {
                entry.key == witness.key && matches!(entry.status, SnapshotStatus::Rejected(_))
            })
        });
        for e in &mut self.entries {
            if matches!(e.status, SnapshotStatus::Rejected(_)) {
                e.status = SnapshotStatus::Pending;
            }
        }
        self.cursor = self
            .entries
            .iter()
            .position(|e| e.status == SnapshotStatus::Pending)
            .unwrap_or(self.entries.len());
    }
    pub(crate) fn yield_to_foreground(&mut self, interactive: &mut Interactive) {
        if self.validating.take().is_some() {
            self.pending_witness = None;
            interactive.cancel_snapshot_validation();
        }
    }
    /// At most one completion and one metadata candidate per paint; model I/O
    /// stays on the worker, and the requested source yields to foreground work.
    pub fn advance(
        &mut self,
        run: &RunIndex,
        loader: &mut Loader,
        interactive: &mut Interactive,
    ) -> Result<()> {
        self.advance_with_followup(run, loader, interactive, false)
    }
    pub(crate) fn advance_with_followup(
        &mut self,
        run: &RunIndex,
        loader: &mut Loader,
        interactive: &mut Interactive,
        allow_followup: bool,
    ) -> Result<()> {
        self.known_entries = run.recording().entries().len();
        if self.validating.is_none() && self.indexed_until < self.known_entries {
            let end = self
                .indexed_until
                .saturating_add(16)
                .min(self.known_entries);
            for e in &run.recording().entries()[self.indexed_until..end] {
                if e.kind != RecordKind::ModelSnapshot || e.model_role != self.role {
                    continue;
                }
                if self.entries.len() == MAX_SNAPSHOTS {
                    self.limited = true;
                    continue;
                }
                self.entries.push(SnapshotEntry {
                    key: CacheKey {
                        root: run.root().clone(),
                        sequence: e.sequence,
                    },
                    revision: e.revision,
                    role: self.role.clone(),
                    name: String::new(),
                    status: SnapshotStatus::Pending,
                });
            }
            self.indexed_until = end;
            self.entries
                .sort_by_key(|e| (e.revision, std::cmp::Reverse(e.key.sequence)));
            self.cursor = self
                .entries
                .iter()
                .position(|e| e.status == SnapshotStatus::Pending)
                .unwrap_or(self.entries.len());
        }
        if let Some(index) = self.validating {
            if !interactive.snapshot_validation_pending() {
                self.validating = None;
                self.pending_witness = None;
                return Ok(());
            }
            if let Some(result) = interactive.poll_snapshot_validation()? {
                self.entries[index].status = match result {
                    Ok(key) if key == self.entries[index].key => {
                        if let Some(witness) = self.pending_witness.take()
                            && witness.key == key
                            && witness.chunk.upgrade().is_some()
                            && run
                                .by_sequence(key.sequence)
                                .and_then(|i| SnapshotStamp::read(run, i))
                                .as_ref()
                                == Some(&witness.stamp)
                        {
                            self.restored.retain(|old| old.key != key);
                            if self.restored.len() < MAX_SNAPSHOTS {
                                self.restored.push(witness);
                            }
                        }
                        SnapshotStatus::Eligible
                    }
                    Ok(_) => SnapshotStatus::Rejected("stale source identity".into()),
                    Err(e) => SnapshotStatus::Rejected(e.to_string().chars().take(128).collect()),
                };
                self.pending_witness = None;
                self.validating = None;
                self.cursor += 1;
                // The newly eligible requested source belongs to foreground
                // preparation. Yield before scheduling any speculative work.
                if !allow_followup
                    || self.entries[index].status == SnapshotStatus::Eligible
                        && (Some(self.entries[index].revision) == self.priority_revision
                            || Some(&self.entries[index].key) == self.preferred_source.as_ref())
                {
                    return Ok(());
                }
            } else {
                return Ok(());
            }
        }
        self.cursor = self
            .entries
            .iter()
            .position(|e| {
                Some(&e.key) == self.preferred_source.as_ref()
                    && e.status == SnapshotStatus::Pending
            })
            .or_else(|| {
                self.entries.iter().position(|e| {
                    e.status == SnapshotStatus::Pending
                        && Some(e.revision) == self.priority_revision
                })
            })
            .or_else(|| {
                self.entries
                    .iter()
                    .position(|e| e.status == SnapshotStatus::Pending)
            })
            .unwrap_or(self.entries.len());
        if interactive.pending() || self.cursor >= self.entries.len() {
            return Ok(());
        }
        let _candidate_lease = match interactive.reserve_policy_view(512 * 1024) {
            Ok(lease) => lease,
            Err(_) => return Ok(()), // Held frames have priority over validation scratch.
        };
        let e = &self.entries[self.cursor];
        let duplicate_revision = self
            .entries
            .iter()
            .any(|old| old.revision == e.revision && old.status == SnapshotStatus::Eligible);
        if duplicate_revision
            && !matches!(self.panel, SnapshotPanel::Network { .. })
            && self.preferred_source.as_ref() != Some(&e.key)
        {
            self.entries[self.cursor].status =
                SnapshotStatus::Rejected("newer validated sibling selected".into());
            self.cursor += 1;
            return Ok(());
        }
        let index = run
            .by_sequence(e.key.sequence)
            .ok_or_else(|| Error::State("snapshot index unavailable".into()))?;
        let result = (|| -> Result<Option<SnapshotCandidate>> {
            if !run.has_model_record(index) {
                return Err(Error::State("snapshot missing or retained away".into()));
            }
            let Some(chunk) = loader.request(run, index)? else {
                return Ok(None);
            };
            let config = ValueSource::configuration(&chunk)?;
            if config.observation_dim != run.recording().manifest().observation_dim
                || config.action_space != run.recording().manifest().action_space
            {
                return Err(Error::Config(
                    "snapshot encoding differs from recording".into(),
                ));
            }
            let name = match &self.panel {
                SnapshotPanel::Network { name } => {
                    let available = choices(&config);
                    for choice in &available {
                        if self.networks.len() < MAX_SNAPSHOTS
                            && !self.networks.iter().any(|old| old.name == choice.name)
                        {
                            self.networks.push(choice.clone());
                        }
                    }
                    if duplicate_revision {
                        return Err(Error::State("newer validated sibling selected".into()));
                    }
                    available
                        .iter()
                        .find(|choice| name.as_ref().is_none_or(|name| *name == choice.name))
                        .ok_or_else(|| Error::Config("selected inspectable network absent".into()))?
                        .name
                        .clone()
                }
                SnapshotPanel::Policy => {
                    let name =
                        selected_name(&config, (!e.name.is_empty()).then_some(e.name.as_str()))?;
                    PolicySource::model(run, index, Some(&name))?;
                    name
                }
                SnapshotPanel::Value {
                    name,
                    twin,
                    actor_required,
                    output,
                } => {
                    let names: Vec<_> = config
                        .models
                        .iter()
                        .filter(|(_, m)| {
                            m.kind
                                .critic_kind()
                                .is_some_and(|kind| kind != ValueCriticKind::CategoricalQ)
                        })
                        .map(|(n, _)| n.clone())
                        .collect();
                    let name = name
                        .clone()
                        .or_else(|| (names.len() == 1).then(|| names[0].clone()))
                        .ok_or_else(|| Error::Config("choose an unambiguous critic".into()))?;
                    let mut source = ValueSource::model_for_conditioning(
                        run,
                        index,
                        &name,
                        &chunk,
                        *actor_required,
                    )?;
                    if let Some(group) = twin {
                        source = source.with_twin(group)?;
                    }
                    let supported = match output {
                        ValueOutput::ExpectedReturn | ValueOutput::DecisionLoss => true,
                        ValueOutput::Q1
                        | ValueOutput::Q2
                        | ValueOutput::Minimum
                        | ValueOutput::Disagreement => source.critic_names().len() == 2,
                        ValueOutput::Quantile { .. } | ValueOutput::Cvar { .. } => {
                            source.kind().is_return_distribution()
                        }
                        ValueOutput::RawSecondMoment => source.kind() == ValueCriticKind::MomentQ,
                        ValueOutput::Variance | ValueOutput::StandardDeviation => matches!(
                            source.kind(),
                            ValueCriticKind::MomentQ | ValueCriticKind::CategoricalReturnQ
                        ),
                        ValueOutput::Utility => {
                            source.kind() == ValueCriticKind::MomentQ
                                && source
                                    .declaration()
                                    .and_then(|d| d.critics.get(&name))
                                    .is_some_and(|c| c.utility.is_some())
                        }
                    };
                    if !supported {
                        return Err(Error::Config(
                            "critic does not support selected output".into(),
                        ));
                    }
                    if source
                        .critic_names()
                        .iter()
                        .any(|n| !config.models.contains_key(n))
                    {
                        return Err(Error::Config("required named twin critic absent".into()));
                    }
                    if let Some(actor) = source.actor_name()
                        && !config.models.get(actor).is_some_and(|m| {
                            matches!(
                                m.kind,
                                ModelKind::DeterministicPolicy
                                    | ModelKind::StochasticPolicy
                                    | ModelKind::StateGaussianPolicy
                            )
                        })
                    {
                        return Err(Error::Config("required named actor absent".into()));
                    }
                    if *actor_required
                        && !matches!(
                            source.kind(),
                            ValueCriticKind::StateValue | ValueCriticKind::DiscreteQ
                        )
                        && source.actor_name().is_none()
                    {
                        return Err(Error::Config("required associated actor absent".into()));
                    }
                    name
                }
            };
            if name.len() > 128 {
                return Err(Error::Config(
                    "snapshot model name exceeds admission".into(),
                ));
            }
            Ok(Some(SnapshotCandidate {
                name,
                chunk,
                stamp: SnapshotStamp::read(run, index),
            }))
        })();
        match result {
            Ok(Some(SnapshotCandidate { name, chunk, stamp })) => {
                self.entries[self.cursor].name = name;
                let reused = !matches!(self.panel, SnapshotPanel::Network { .. })
                    && stamp.as_ref().is_some_and(|stamp| {
                        self.restored.iter().any(|witness| {
                            witness.matches(&self.entries[self.cursor].key, &chunk, stamp)
                        })
                    });
                let validation = if reused {
                    interactive.verify_snapshot(run, index, &chunk)
                } else if matches!(self.panel, SnapshotPanel::Network { .. }) {
                    interactive.validate_network_snapshot(
                        run,
                        index,
                        self.entries[self.cursor].name.clone(),
                    )
                } else {
                    interactive.validate_snapshot(run, index)
                };
                match validation {
                    Ok(()) => {
                        self.validating = Some(self.cursor);
                        self.pending_witness =
                            if matches!(self.panel, SnapshotPanel::Network { .. }) {
                                None
                            } else {
                                stamp.map(|stamp| RestoredWitness {
                                    key: self.entries[self.cursor].key.clone(),
                                    chunk: Arc::downgrade(&chunk),
                                    stamp,
                                })
                            };
                    }
                    Err(error) => {
                        self.entries[self.cursor].status =
                            SnapshotStatus::Rejected(error.to_string().chars().take(128).collect());
                        self.cursor += 1;
                    }
                }
            }
            Ok(None) => {}
            Err(error) => {
                self.entries[self.cursor].status =
                    SnapshotStatus::Rejected(error.to_string().chars().take(128).collect());
                self.cursor += 1;
            }
        }
        Ok(())
    }
}
