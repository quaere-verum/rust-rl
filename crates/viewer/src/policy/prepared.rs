//! Bounded preparation on Interactive's historical worker. No renderer evaluates
//! models, scans populations, or reconstructs stochastic charts.
use super::{
    ensure,
    numerics::{self, SelectedDistribution},
    slice::{GridEligibility, MAX_POPULATION_BYTES, MAX_POPULATION_ROWS, SliceConfig, SliceMode},
    visitation::{MAX_VISIT_MARKERS, PreparedVisitation, VisitationSource},
};
use crate::{
    budgets::{POLICY_FRAME, POLICY_PREPARED, POLICY_SCRATCH},
    data::{CacheKey, RunIndex, SelectedInput},
    pane::SelectionIntent,
    plot_data::matching_input_where,
    policy_change::{PolicyBatch, admission::checked_batch},
    resolver::{LeasedChunk, ResolvedRecord},
};
use rl::{
    Error, Result,
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        models::{ModelKind, ModelLimits, SnapshotConfiguration, snapshot_configuration},
        policy_surface::EvaluatedPolicyBatch as InspectionResult,
        schema::{Chunk, ProbePopulation, Record, RecordKind, RunManifest},
        storage::ChunkSource,
        value::ValueCriticKind,
        visualization::EnvironmentSemantics,
    },
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};
#[path = "prepared_cached.rs"]
mod cached;
pub(crate) use cached::cached_frame;

pub const MAX_SOURCE_CANDIDATES: usize = 16;
pub const MAX_SOURCE_SEARCH: usize = 256;
pub const MAX_RECORDED_MATCH_CHECKS: usize = 1024 * 1024;
pub const DISTRIBUTION_SAMPLES: usize = 257;
const MAX_CROSS_REFERENCE_SAMPLES: usize = 256;
const FIXED_SCRATCH: usize = 64 * 1024;

/// Observation copies inside inspection, returned arrays, full KL and selected
/// distributions are admitted together, for BOTH endpoints. Backend tensors
/// retain the separate ModelLimits runtime admission.
pub fn batch_capacity(observations: usize, actions: usize, reference: bool) -> Result<usize> {
    rl::diagnostics::policy_surface::batch_capacity(
        observations,
        actions,
        reference,
        POLICY_SCRATCH,
        1024,
    )
}
pub fn compatible_encoding(
    same_run: bool,
    a: EnvironmentSemantics<'_>,
    b: EnvironmentSemantics<'_>,
) -> bool {
    a.observation_dim() == b.observation_dim()
        && a.action_space() == b.action_space()
        && (same_run
            || matches!((a.descriptor(), b.descriptor()), (Some(a), Some(b)) if a.observations == b.observations && a.actions == b.actions))
}
pub fn policy_names(configuration: &SnapshotConfiguration) -> impl Iterator<Item = &str> {
    configuration
        .models
        .iter()
        .filter(|(_, model)| {
            matches!(
                model.kind,
                ModelKind::StochasticPolicy
                    | ModelKind::StateGaussianPolicy
                    | ModelKind::DeterministicPolicy
                    | ModelKind::DiscreteQ
            )
        })
        .map(|(name, _)| name.as_str())
}
pub(crate) fn selected_name(
    configuration: &SnapshotConfiguration,
    selected: Option<&str>,
) -> Result<String> {
    if let Some(name) = selected {
        ensure(
            policy_names(configuration).any(|n| n == name),
            "Selected model has no supported policy adapter",
        )?;
        return Ok(name.into());
    }
    let mut names = policy_names(configuration);
    let first = names
        .next()
        .ok_or_else(|| Error::Config("No supported policy adapter in snapshot".into()))?;
    ensure(
        names.next().is_none(),
        "Multiple named policies; choose a policy explicitly",
    )?;
    Ok(first.into())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceIdentity {
    pub key: CacheKey,
    pub probe_key: Option<CacheKey>,
    pub revision: u64,
    /// Actual named model or the explicit recorded controller caption.
    pub name: String,
    pub recorded: bool,
    /// Explicit controller comparison matches physical conditioning, while
    /// population names and row labels remain independent provenance.
    pub explicit_controller: bool,
}
#[derive(Clone)]
enum SourceData {
    Model(ChunkSource),
    Recorded(ResolvedRecord),
}
#[derive(Clone)]
pub struct PolicySource {
    pub(crate) identity: SourceIdentity,
    data: SourceData,
    environment: Arc<RunManifest>,
    metadata_bytes: usize,
    explicit_name: bool,
    metadata_bound: bool,
    decision_supported: bool,
    action_value_critic: Option<Arc<PolicyCriticSpec>>,
    pub(crate) reference_info: Arc<crate::reference::ReferenceInfo>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PolicyCriticSpec {
    pub name: String,
    pub heads: usize,
    pub reward_units: String,
    pub role: String,
}
impl PolicySource {
    /// Shared retained metadata is charged by its actual Arc allocation, using
    /// the same conservative bound as Policy requests.
    pub(crate) fn shared_metadata(&self) -> (usize, usize) {
        (Arc::as_ptr(&self.environment) as usize, self.metadata_bytes)
    }
    fn environment(run: &RunIndex) -> Result<(Arc<RunManifest>, usize)> {
        let original = run.recording().manifest();
        let descriptor_bytes = original
            .environment_descriptor
            .as_ref()
            .map(|d| serde_json::to_vec(d).map(|b| b.len()))
            .transpose()?
            .unwrap_or(0);
        let bytes = descriptor_bytes
            .saturating_mul(8)
            .saturating_add(original.action_space.dimensions().saturating_mul(16))
            .saturating_add(4096);
        ensure(
            bytes <= 256 * 1024,
            "Policy encoding metadata exceeds byte budget",
        )?;
        let mut manifest = RunManifest::new(
            "policy-input",
            "encoding",
            "uninterpreted",
            original.observation_dim,
            original.action_space.clone(),
            0.9,
        )?;
        manifest.environment_descriptor = original.environment_descriptor.clone();
        Ok((Arc::new(manifest), bytes))
    }
    /// Select exactly this immutable index. This advertises no unread weight
    /// capability; adapter/configuration validation occurs on the worker.
    pub fn model(run: &RunIndex, index: usize, name: Option<&str>) -> Result<Self> {
        let entry = run
            .recording()
            .entries()
            .get(index)
            .ok_or_else(|| Error::State("Missing snapshot index".into()))?;
        ensure(
            entry.kind == RecordKind::ModelSnapshot && run.has_model_record(index),
            "Selected snapshot missing or retired",
        )?;
        ensure(
            name.is_none_or(|n| !n.is_empty() && n.len() <= 128),
            "Invalid policy name",
        )?;
        let (environment, mut metadata_bytes) = Self::environment(run)?;
        let reference_info =
            crate::reference::ReferenceInfo::new(run.recording().manifest(), None)?;
        metadata_bytes += reference_info.bytes();
        Ok(Self {
            identity: SourceIdentity {
                key: run.key(index)?,
                probe_key: None,
                revision: entry.revision,
                name: name.unwrap_or_default().into(),
                recorded: false,
                explicit_controller: false,
            },
            data: SourceData::Model(run.recording().source(index)?),
            environment,
            metadata_bytes,
            explicit_name: name.is_some(),
            metadata_bound: false,
            decision_supported: false,
            action_value_critic: None,
            reference_info,
        })
    }
    /// Retains existing resolver leases; numeric validation remains on the worker.
    pub fn recorded(run: &RunIndex, frame: &ResolvedRecord, controller: &str) -> Result<Self> {
        ensure(
            frame.key.root == *run.root() && frame.probe_key.root == *run.root(),
            "Recorded policy belongs to another run",
        )?;
        ensure(
            !controller.is_empty() && controller.len() <= 128,
            "Invalid recorded controller identity",
        )?;
        let (environment, mut metadata_bytes) = Self::environment(run)?;
        let reference_info =
            crate::reference::ReferenceInfo::new(run.recording().manifest(), Some(&frame.chunk))?;
        metadata_bytes += reference_info.bytes();
        Ok(Self {
            identity: SourceIdentity {
                key: frame.key.clone(),
                probe_key: Some(frame.probe_key.clone()),
                revision: frame.revision(),
                name: controller.into(),
                recorded: true,
                explicit_controller: false,
            },
            data: SourceData::Recorded(frame.clone()),
            environment,
            metadata_bytes,
            explicit_name: true,
            metadata_bound: true,
            decision_supported: matches!(
                run.recording().manifest().action_space,
                ActionSpace::Discrete { count: 1..=64 }
            ) && matches!(&frame.probes.record, Record::ProbeSet(p) if p.histories.is_empty())
                && matches!(&frame.chunk.record, Record::Policy(p) if matches!(p.parameters, rl::diagnostics::schema::PolicyParameters::Categorical { .. }) || matches!(p.parameters, rl::diagnostics::schema::PolicyParameters::Deterministic) && !p.stochastic),
            action_value_critic: None,
            reference_info,
        })
    }
    pub fn recorded_controller(
        run: &RunIndex,
        frame: &ResolvedRecord,
        controller: &str,
    ) -> Result<Self> {
        let mut source = Self::recorded(run, frame, controller)?;
        source.identity.explicit_controller = true;
        Ok(source)
    }
    pub(crate) fn with_metadata(mut self, run: &RunIndex, chunk: &Chunk) -> Result<Self> {
        let info = crate::reference::ReferenceInfo::new(run.recording().manifest(), Some(chunk))?;
        self.metadata_bytes = self.metadata_bytes - self.reference_info.bytes() + info.bytes();
        // The shared accounting key represents this complete metadata owner.
        // A newly decoded declaration must not alias an older owner's key.
        self.environment = Arc::new((*self.environment).clone());
        self.reference_info = info;
        self.decision_supported = if let Record::ModelSnapshot(snapshot) = &chunk.record {
            let config = snapshot_configuration(snapshot, ModelLimits::default())?;
            self.action_value_critic = policy_critic_spec(run, chunk, &config, self.name());
            self.metadata_bytes = self.metadata_bytes.saturating_add(
                self.action_value_critic.as_ref().map_or(0, |s| {
                    s.name.capacity() + s.reward_units.capacity() + s.role.capacity() + 256
                }),
            );
            matches!(config.action_space, ActionSpace::Discrete { count: 1..=64 })
                && config.models.get(&self.identity.name).is_some_and(|m| {
                    matches!(
                        m.kind,
                        ModelKind::DiscreteQ
                            | ModelKind::DeterministicPolicy
                            | ModelKind::StochasticPolicy
                            | ModelKind::StateGaussianPolicy
                    )
                })
        } else {
            false
        };
        self.metadata_bound = true;
        Ok(self)
    }
    /// Catalog reuse returns the same immutable source. Metadata absence is
    /// also a validated result; idle paints must not schedule reads or decode it.
    pub(crate) fn bind_metadata(
        self,
        run: &RunIndex,
        load: impl FnOnce() -> Result<Option<Arc<Chunk>>>,
    ) -> Result<Option<Self>> {
        if self.metadata_bound {
            return Ok(Some(self));
        }
        load()?
            .map(|chunk| self.with_metadata(run, &chunk))
            .transpose()
    }
    pub(crate) fn decision_subject(&self) -> &str {
        if self.identity.recorded {
            self.reference_info
                .declaration()
                .and_then(|d| d.sources.keys().next())
                .map_or(self.identity.name.as_str(), String::as_str)
        } else {
            &self.identity.name
        }
    }
    pub(crate) fn same_decision_payload(&self, other: &Self) -> bool {
        match (self.recorded_frame(), other.recorded_frame()) {
            (Some(a), Some(b)) => {
                std::ptr::eq(a.chunk.as_ref(), b.chunk.as_ref())
                    && std::ptr::eq(a.probes.as_ref(), b.probes.as_ref())
            }
            (None, None) => self.identity == other.identity,
            _ => false,
        }
    }
    pub(crate) fn decision_supported(&self) -> bool {
        self.metadata_bound
            && self.decision_supported
            && self.reference_info.declaration().is_some_and(|d| {
                if self.identity.recorded {
                    d.sources.len() == 1
                } else {
                    d.sources.contains_key(&self.identity.name)
                }
            })
    }
    pub(crate) fn finite_reference(&self) -> bool {
        self.reference_info.finite()
    }
    pub(crate) fn prepare_reference(
        &self,
        accounting: &Arc<Accounting>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        if let Some(frame) = self.recorded_frame() {
            self.reference_info
                .prepare(&frame.chunk, &frame.probes, accounting, cancelled)?;
        }
        Ok(())
    }
    pub(crate) fn reference_quality_ui(&self, ui: &mut eframe::egui::Ui, capture: bool) {
        self.reference_info
            .ui(ui, &self.identity.name, self.identity.recorded, capture);
    }
    fn matching_row(
        &self,
        probes: &Chunk,
        row: usize,
        other: &Chunk,
        batch: &PolicyBatch<'_>,
        mask: Option<&[bool]>,
    ) -> Result<Option<usize>> {
        if self.finite_reference() {
            let mut input = exact_observation(probes, row)?;
            input
                .as_dense_mut()
                .expect("dense diagnostic input")
                .action_mask = mask.map(<[bool]>::to_vec);
            return self.reference_info.lookup(other, &input);
        }
        matching_input_where(
            probes,
            row,
            other,
            !self.identity.explicit_controller,
            |candidate| same_effective_mask(batch, candidate, mask),
        )
    }
    pub fn identity(&self) -> &SourceIdentity {
        &self.identity
    }
    pub fn semantics(&self) -> EnvironmentSemantics<'_> {
        EnvironmentSemantics::new(&self.environment)
    }
    pub(crate) fn model_source(&self) -> Option<&ChunkSource> {
        match &self.data {
            SourceData::Model(source) => Some(source),
            _ => None,
        }
    }
    pub(crate) fn name(&self) -> Option<&str> {
        self.explicit_name.then_some(self.identity.name.as_str())
    }
    pub(crate) fn action_value_critic(&self) -> Option<&PolicyCriticSpec> {
        self.action_value_critic.as_deref()
    }
    pub(crate) fn check_available(&self) -> Result<()> {
        if let Some(source) = self.model_source() {
            source.check_available()?;
        }
        Ok(())
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        self.identity == other.identity
            && self.metadata_bound == other.metadata_bound
            && self.decision_supported == other.decision_supported
            && self.action_value_critic == other.action_value_critic
            && self.reference_info.same(&other.reference_info)
            && self.environment.observation_dim == other.environment.observation_dim
            && self.environment.action_space == other.environment.action_space
            && self.environment.environment_descriptor == other.environment.environment_descriptor
    }
    pub(crate) fn recorded_frame(&self) -> Option<&ResolvedRecord> {
        match &self.data {
            SourceData::Recorded(frame) => Some(frame),
            _ => None,
        }
    }
}

/// Only an unambiguous declared action-conditioned critic from this exact
/// snapshot can supply the optional Policy diagnostic.
fn policy_critic_spec(
    run: &RunIndex,
    chunk: &Chunk,
    config: &SnapshotConfiguration,
    policy_name: Option<&str>,
) -> Option<Arc<PolicyCriticSpec>> {
    if !matches!(config.action_space, ActionSpace::Box { .. }) {
        return None;
    }
    let actor = selected_name(config, policy_name).ok()?;
    if !matches!(
        config.models.get(&actor)?.kind,
        ModelKind::DeterministicPolicy
    ) {
        return None;
    }
    let declaration = rl::diagnostics::value::declaration(chunk).ok()??;
    declaration
        .validate_manifest(run.recording().manifest())
        .ok()?;
    let reward_units = declaration.conventions.as_ref()?.reward_units.clone();
    let role = declaration.model_role.clone();
    let mut candidates = declaration.critics.iter().filter(|(_, critic)| {
        critic.learned_baseline != Some(false)
            && matches!(
                critic.kind,
                ValueCriticKind::ScalarQ
                    | ValueCriticKind::QuantileQ
                    | ValueCriticKind::MomentQ
                    | ValueCriticKind::CategoricalQ
                    | ValueCriticKind::CategoricalReturnQ
                    | ValueCriticKind::DiscreteQ
            )
    });
    let (name, critic) = candidates.next()?;
    if candidates.next().is_some()
        || critic.continuation.is_none()
        || critic.twin.is_some()
        || name.len() > 128
    {
        return None;
    }
    let model = config.models.get(name)?;
    if !matches!(
        critic.kind,
        ValueCriticKind::ScalarQ | ValueCriticKind::QuantileQ | ValueCriticKind::CategoricalReturnQ
    ) || model.kind.critic_kind() != Some(critic.kind)
    {
        return None;
    }
    let heads = *model.network.as_ref()?.widths.last()?;
    (heads > 0 && heads <= 4096).then(|| {
        Arc::new(PolicyCriticSpec {
            name: name.clone(),
            heads,
            reward_units,
            role,
        })
    })
}

/// An on-demand predecessor search using the existing Loader. UI availability
/// markers alone never invoke this or certify unread model weights. One loaded
/// metadata candidate is validated per call, with at most 16 skipped addresses
/// and 256 total candidates. Once selected, identity is sticky until a NEW search.
pub struct ModelSearch {
    role: String,
    revision: u64,
    name: Option<String>,
    root: Option<Arc<std::path::PathBuf>>,
    before: Option<(u64, u64)>,
    loading: Option<usize>,
    selected: Option<PolicySource>,
    selected_value: Option<crate::value::prepared::ValueSource>,
    value_probe_phase: bool,
    value_models: bool,
    value_probe_key: Option<CacheKey>,
    indexed_len: usize,
    checked: usize,
    done: bool,
    pub last_rejection: Option<String>,
}
impl ModelSearch {
    pub(crate) fn view_owned_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.role.capacity()
            + self.name.as_ref().map_or(0, String::capacity)
            + self.last_rejection.as_ref().map_or(0, String::capacity)
            + self
                .selected
                .as_ref()
                .map_or(0, |s| s.metadata_bytes + s.identity.name.capacity())
            + self.selected_value.as_ref().map_or(0, |s| s.metadata_bytes)
    }
    pub(crate) fn view_owned_bytes_with(
        &self,
        charges: &mut crate::value::ViewSourceCharges,
    ) -> usize {
        self.view_owned_bytes() - self.selected_value.as_ref().map_or(0, |s| s.metadata_bytes)
            + self
                .selected_value
                .as_ref()
                .map_or(0, |s| charges.charge(s))
    }
    pub fn new(role: &str, revision: u64, name: Option<&str>) -> Result<Self> {
        ensure(
            !role.is_empty()
                && role.len() <= 128
                && name.is_none_or(|n| !n.is_empty() && n.len() <= 128),
            "Invalid Policy source selection",
        )?;
        Ok(Self {
            role: role.into(),
            revision,
            name: name.map(str::to_owned),
            root: None,
            before: None,
            loading: None,
            selected: None,
            selected_value: None,
            value_probe_phase: false,
            value_models: true,
            value_probe_key: None,
            indexed_len: 0,
            checked: 0,
            done: false,
            last_rejection: None,
        })
    }
    pub fn pending(&self) -> bool {
        !self.done
    }
    pub(crate) fn value_samples_only(&mut self, probe: Option<CacheKey>) {
        self.value_models = false;
        self.value_probe_phase = true;
        self.value_probe_key = probe;
        self.last_rejection = Some("No compatible recorded critic samples available".into());
    }
    pub fn checked_candidates(&self) -> usize {
        self.checked
    }
    /// Continue the bounded shared search after a Value-specific incompatibility.
    pub(crate) fn reject_value_candidate(&mut self, reason: &str) {
        self.selected_value = None;
        self.done = false;
        self.last_rejection = Some(reason.into());
    }
    /// Explicitly changing selection invalidates positive and negative searches.
    pub fn retarget(&mut self, role: &str, revision: u64, name: Option<&str>) -> Result<()> {
        *self = Self::new(role, revision, name)?;
        Ok(())
    }
    fn next_loaded(
        &mut self,
        run: &RunIndex,
        loader: &mut crate::data::Loader,
        kind: RecordKind,
    ) -> Result<Option<(usize, Arc<Chunk>)>> {
        for _ in 0..MAX_SOURCE_CANDIDATES {
            if self.loading.is_none() && self.checked >= MAX_SOURCE_SEARCH {
                self.done = true;
                self.last_rejection = Some("Snapshot candidate limit reached".into());
                return Ok(None);
            }
            let index = if let Some(index) = self.loading {
                index
            } else {
                let next = if let Some(before) = self.before {
                    run.candidates_before(kind, &self.role, None, before).next()
                } else {
                    run.candidates(kind, &self.role, None, self.revision).next()
                };
                let Some(index) = next else {
                    self.done = true;
                    return Ok(None);
                };
                let entry = &run.recording().entries()[index];
                self.before = Some((entry.revision, entry.sequence));
                self.checked += 1;
                if kind == RecordKind::ModelSnapshot && !run.has_model_record(index) {
                    continue;
                }
                self.loading = Some(index);
                index
            };
            match loader.request(run, index) {
                Err(e) => {
                    self.last_rejection = Some(e.to_string());
                    self.loading = None;
                }
                Ok(None) => return Ok(None),
                Ok(Some(chunk)) => {
                    self.loading = None;
                    return Ok(Some((index, chunk)));
                }
            }
        }
        Ok(None)
    }
    /// Critic capability search through the same bounded candidate cursor and
    /// Loader. Probe-only declarations use existing resolver leases. A negative
    /// result is invalidated by appended records or a changed recording root.
    pub fn advance_value(
        &mut self,
        run: &RunIndex,
        loader: &mut crate::data::Loader,
        resolver: &mut crate::resolver::ProbeResolver,
    ) -> Result<Option<crate::value::prepared::ValueSource>> {
        self.advance_value_for_slice(run, loader, resolver, None)
    }
    pub(crate) fn advance_value_for_slice(
        &mut self,
        run: &RunIndex,
        loader: &mut crate::data::Loader,
        resolver: &mut crate::resolver::ProbeResolver,
        slice: Option<&crate::value::slice::ValueSlice>,
    ) -> Result<Option<crate::value::prepared::ValueSource>> {
        use crate::value::prepared::ValueSource;
        let length = run.recording().entries().len();
        if self.root.as_ref().is_some_and(|r| r != run.root())
            || self.selected_value.is_none() && self.indexed_len != length
        {
            self.root = Some(run.root().clone());
            self.before = None;
            self.loading = None;
            self.selected = None;
            self.selected_value = None;
            self.checked = 0;
            self.done = false;
            self.value_probe_phase = !self.value_models;
            self.last_rejection = None;
        }
        self.root = Some(run.root().clone());
        self.indexed_len = length;
        if let Some(s) = &self.selected_value {
            s.check_available()?;
            return Ok(Some(s.clone()));
        }
        if self.done {
            return Ok(None);
        }
        let kind = if self.value_probe_phase {
            RecordKind::Critic
        } else {
            RecordKind::ModelSnapshot
        };
        let Some((index, chunk)) = self.next_loaded(run, loader, kind)? else {
            if self.done && !self.value_probe_phase && self.checked < MAX_SOURCE_SEARCH {
                self.done = false;
                self.value_probe_phase = true;
                self.before = None;
            }
            return Ok(None);
        };
        let result = (|| -> Result<Option<ValueSource>> {
            if kind == RecordKind::ModelSnapshot {
                let configuration = ValueSource::configuration(&chunk)?;
                let mut names = configuration
                    .models
                    .iter()
                    .filter(|(_, m)| {
                        m.kind
                            .critic_kind()
                            .is_some_and(|kind| kind != ValueCriticKind::CategoricalQ)
                    })
                    .map(|(name, _)| name.as_str());
                let name = if let Some(name) = &self.name {
                    name.as_str()
                } else {
                    let first = names
                        .next()
                        .ok_or_else(|| Error::Config("snapshot has no supported critic".into()))?;
                    ensure(
                        names.next().is_none(),
                        "multiple critics; select an explicit name",
                    )?;
                    first
                };
                return ValueSource::model(run, index, name, &chunk).map(Some);
            }
            let name = if let (Some(name), Some(d)) =
                (&self.name, rl::diagnostics::value::declaration(&chunk)?)
            {
                ensure(
                    d.critics.contains_key(name),
                    "named critic absent from probe",
                )?;
                name.clone()
            } else {
                ValueSource::recorded_name(&chunk, run.key(index)?.sequence)?
            };
            ensure(
                self.name.as_ref().is_none_or(|selected| selected == &name),
                "named critic absent from probe",
            )?;
            let set = chunk
                .context
                .probe_set
                .as_deref()
                .ok_or_else(|| Error::Config("critic missing genuine probe identity".into()))?;
            let pi = run
                .probe_for_record(index)
                .ok_or_else(|| Error::Config("critic probe source unavailable".into()))?;
            let probe_key = run.key(pi)?;
            ensure(
                self.value_probe_key
                    .as_ref()
                    .is_none_or(|key| key == &probe_key),
                "history requires the original immutable probe source",
            )?;
            let Some(probes) = loader.request(run, pi)? else {
                self.loading = Some(index);
                return Ok(None);
            };
            ensure(
                probes.context.identity.run_id == chunk.context.identity.run_id
                    && probes.context.probe_set.as_deref() == Some(set),
                "probe/critic context differs",
            )?;
            let key = run.key(index)?;
            let probe_key = run.key(pi)?;
            let frame = ResolvedRecord {
                chunk: resolver.lease_record(run, &key, chunk.clone())?,
                probes: resolver.lease_record(run, &probe_key, probes)?,
                key,
                probe_key,
                row: 0,
            };
            let source = ValueSource::recorded(run, &frame, &name)?;
            if let Some(slice) = slice {
                ensure(
                    source.finite_reference() || source.has_recorded_sample(slice)?,
                    "No compatible recorded critic samples in this slice",
                )?;
            }
            Ok(Some(source))
        })();
        match result {
            Ok(Some(source)) => {
                self.selected_value = Some(source.clone());
                self.done = true;
                Ok(Some(source))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                self.last_rejection = Some(e.to_string());
                Ok(None)
            }
        }
    }
    pub fn advance(
        &mut self,
        run: &RunIndex,
        loader: &mut crate::data::Loader,
    ) -> Result<Option<PolicySource>> {
        if let Some(root) = &self.root {
            ensure(
                root == run.root(),
                "Policy source search belongs to another run; start a new search",
            )?;
        } else {
            self.root = Some(run.root().clone());
        }
        if let Some(source) = &self.selected {
            source.check_available()?;
            return Ok(Some(source.clone()));
        }
        if self.done {
            return Ok(None);
        }
        {
            let Some((index, chunk)) = self.next_loaded(run, loader, RecordKind::ModelSnapshot)?
            else {
                return Ok(None);
            };
            let configuration = match &chunk.record {
                Record::ModelSnapshot(snapshot) => {
                    snapshot_configuration(snapshot, ModelLimits::default())
                }
                _ => Err(Error::State("Selected index is not a snapshot".into())),
            };
            let configuration = match configuration {
                Ok(c) => c,
                Err(error) => {
                    self.last_rejection = Some(error.to_string());
                    return Ok(None);
                }
            };
            if configuration.observation_dim != run.recording().manifest().observation_dim
                || configuration.action_space != run.recording().manifest().action_space
            {
                self.last_rejection =
                    Some("Snapshot input/action encoding differs from recording".into());
                return Ok(None);
            }
            let candidates = policy_names(&configuration).count();
            if self.name.is_none() && candidates > 1 {
                self.done = true;
                return Err(Error::Config(
                    "Multiple named policies; choose a policy explicitly".into(),
                ));
            }
            let name = match selected_name(&configuration, self.name.as_deref()) {
                Ok(name) => name,
                Err(error) => {
                    self.last_rejection = Some(error.to_string());
                    return Ok(None);
                }
            };
            let source =
                PolicySource::model(run, index, Some(&name))?.with_metadata(run, &chunk)?;
            self.selected = Some(source.clone());
            self.done = true;
            Ok(Some(source))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PolicyPin {
    /// Exact complete feedforward model input. Never a global recorded row.
    Synthetic(Observation),
    /// Real row in the current recorded Policy source; preserves conditioning.
    Recorded {
        key: CacheKey,
        probe_key: CacheKey,
        row: usize,
    },
}
impl PolicyPin {
    pub fn synthetic(input: Observation) -> Self {
        Self::Synthetic(input)
    }
}
#[derive(Clone)]
pub struct VisitSource {
    pub key: CacheKey,
    pub chunk: LeasedChunk,
}
#[derive(Clone)]
pub struct PolicyRequest {
    pub current: PolicySource,
    /// Genuine immutable input rows for a model evaluated on a sampled population.
    /// Stored predictions are never used as the current model output.
    pub input_population: Option<PolicySource>,
    pub reference: Option<PolicySource>,
    pub requested_revision: u64,
    pub slice: Arc<SliceConfig>,
    pub pin: PolicyPin,
    /// Reuse the first suitable domain across checkpoints; required for
    /// unbounded Gaussian plots, optional for declared bounded actions.
    pub distribution_domain: Option<[f64; 2]>,
    pub visits: Option<VisitSource>,
}
impl PolicyRequest {
    pub(crate) fn input_frame(&self) -> Option<&ResolvedRecord> {
        self.input_population
            .as_ref()
            .unwrap_or(&self.current)
            .recorded_frame()
    }
    pub(crate) fn check_inputs(&self) -> Result<()> {
        if let Some(source) = &self.input_population {
            source.check_available()?;
        }
        Ok(())
    }
    pub fn new(
        current: PolicySource,
        reference: Option<PolicySource>,
        requested_revision: u64,
        slice: SliceConfig,
        pin: PolicyPin,
        distribution_domain: Option<[f64; 2]>,
    ) -> Result<Self> {
        let value = Self {
            current,
            input_population: None,
            reference,
            requested_revision,
            slice: Arc::new(slice),
            pin,
            distribution_domain,
            visits: None,
        };
        value.validate()?;
        Ok(value)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        self.slice.validate(self.current.semantics())?;
        if let Some(source) = &self.input_population {
            ensure(
                !self.current.identity.recorded,
                "Model input population requires a retained current model",
            )?;
            let frame = source.recorded_frame().ok_or_else(|| {
                Error::Config("Model input population must be genuine recorded rows".into())
            })?;
            ensure_recorded_feedforward(&frame.probes)?;
            ensure(
                compatible_encoding(
                    self.current.identity.key.root == source.identity.key.root,
                    self.current.semantics(),
                    source.semantics(),
                ),
                "Model/input population encodings differ",
            )?;
            let PolicyPin::Recorded {
                key,
                probe_key,
                row,
            } = &self.pin
            else {
                return Err(Error::Config(
                    "Model population requires an exact recorded pin".into(),
                ));
            };
            ensure(
                key == &frame.key && probe_key == &frame.probe_key,
                "Recorded pin source identity differs",
            )?;
            let Record::ProbeSet(probes) = &frame.probes.record else {
                return Err(Error::Config("Missing input probes".into()));
            };
            ensure(
                *row < frame.probes.arrays[&probes.observations].shape()[0],
                "Recorded pin row is outside its input population",
            )?;
        }
        if let Some(reference) = &self.reference {
            self.current
                .reference_info
                .compatible(&reference.reference_info)?;
            ensure(
                compatible_encoding(
                    self.current.identity.key.root == reference.identity.key.root,
                    self.current.semantics(),
                    reference.semantics(),
                ),
                "Reference input encoding/action descriptors are incompatible or unknown",
            )?;
        }
        if let Some([lo, hi]) = self.distribution_domain {
            ensure(
                lo.is_finite() && hi.is_finite() && lo < hi && (hi - lo).is_finite(),
                "Invalid stable action domain",
            )?;
        }
        if let PolicyPin::Synthetic(input) = &self.pin {
            input.validate(
                self.slice.fixed.len(),
                self.current.semantics().action_space(),
                true,
            )?;
            let dense = input.as_dense().expect("validated dense input");
            ensure(
                dense
                    .values
                    .capacity()
                    .saturating_mul(4)
                    .saturating_add(dense.action_mask.as_ref().map_or(0, Vec::capacity))
                    <= 256 * 1024,
                "Policy pin exceeds byte budget",
            )?;
        }
        batch_capacity(
            self.slice.fixed.len(),
            action_width(self.current.semantics().action_space()),
            self.reference.is_some(),
        )?;
        Ok(())
    }
    pub(crate) fn same_grid(&self, other: &Self) -> bool {
        self.current.same(&other.current)
            && match (&self.input_population, &other.input_population) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same(b),
                _ => false,
            }
            && match (&self.reference, &other.reference) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same(b),
                _ => false,
            }
            && self.slice == other.slice
            && self.slice.fixed.iter().map(|v| v.to_bits()).eq(other
                .slice
                .fixed
                .iter()
                .map(|v| v.to_bits()))
            && self
                .slice
                .ranges
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .eq(other.slice.ranges.iter().flatten().map(|v| v.to_bits()))
            && self.visits.as_ref().map(|v| &v.key) == other.visits.as_ref().map(|v| &v.key)
    }
    pub(crate) fn same_pin(&self, other: &Self) -> bool {
        self.pin == other.pin
            && self.distribution_domain == other.distribution_domain
            && match (&self.pin, &other.pin) {
                (PolicyPin::Synthetic(a), PolicyPin::Synthetic(b)) => a
                    .dense_values()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(b.dense_values().iter().map(|v| v.to_bits())),
                _ => true,
            }
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.slice.owned_bytes()
            + self.current.metadata_bytes
            + self.reference.as_ref().map_or(0, |s| s.metadata_bytes)
            + self
                .input_population
                .as_ref()
                .map_or(0, |s| s.metadata_bytes)
            + match &self.pin {
                PolicyPin::Synthetic(input) => input.as_dense().map_or(0, |dense| {
                    dense.values.capacity() * 4
                        + dense.action_mask.as_ref().map_or(0, Vec::capacity)
                }),
                _ => 256,
            }
            + 4096
    }
}
fn action_width(space: &ActionSpace) -> usize {
    match space {
        ActionSpace::Discrete { count } => *count,
        _ => space.dimensions(),
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PolicyMetrics {
    pub prepared_bytes: usize,
    pub prepared_peak: usize,
    pub scratch_peak: usize,
    pub restores: u64,
    pub snapshot_verifications: u64,
    pub grid_batches: u64,
    pub point_batches: u64,
    pub cancellations: u64,
    pub reused_grids: u64,
    pub source_rows: u64,
    pub restore_micros: u64,
    pub inference_micros: u64,
    pub preparation_micros: u64,
}
#[derive(Default)]
pub(crate) struct Accounting {
    bytes: AtomicUsize,
    peak: AtomicUsize,
    scratch: AtomicUsize,
    shared: Mutex<BTreeMap<usize, (usize, Lease)>>,
    pub restores: AtomicU64,
    pub snapshot_verifications: AtomicU64,
    pub grid_batches: AtomicU64,
    pub point_batches: AtomicU64,
    pub cancellations: AtomicU64,
    pub reused_grids: AtomicU64,
    pub source_rows: AtomicU64,
    pub restore_micros: AtomicU64,
    pub inference_micros: AtomicU64,
    pub preparation_micros: AtomicU64,
}
impl Accounting {
    pub fn snapshot(&self) -> PolicyMetrics {
        PolicyMetrics {
            prepared_bytes: self.bytes.load(Ordering::Acquire),
            prepared_peak: self.peak.load(Ordering::Acquire),
            scratch_peak: self.scratch.load(Ordering::Acquire),
            restores: self.restores.load(Ordering::Acquire),
            snapshot_verifications: self.snapshot_verifications.load(Ordering::Acquire),
            grid_batches: self.grid_batches.load(Ordering::Acquire),
            point_batches: self.point_batches.load(Ordering::Acquire),
            cancellations: self.cancellations.load(Ordering::Acquire),
            reused_grids: self.reused_grids.load(Ordering::Acquire),
            source_rows: self.source_rows.load(Ordering::Acquire),
            restore_micros: self.restore_micros.load(Ordering::Acquire),
            inference_micros: self.inference_micros.load(Ordering::Acquire),
            preparation_micros: self.preparation_micros.load(Ordering::Acquire),
        }
    }
    pub(crate) fn scratch(&self, bytes: usize) -> Result<()> {
        ensure(bytes <= POLICY_SCRATCH, "Policy scratch admission exceeded")?;
        self.scratch.fetch_max(bytes, Ordering::AcqRel);
        Ok(())
    }
}
/// Pre-admitted active-pane state and replacement temporaries. Recorded arrays
/// remain shared under their existing resolver leases.
pub(crate) struct PolicyViewLease {
    _lease: Lease,
    pub bytes: usize,
}
impl PolicyViewLease {
    pub(crate) fn new(accounting: &Arc<Accounting>, bytes: usize) -> Result<Self> {
        Ok(Self {
            _lease: Lease::acquire(accounting, bytes)?,
            bytes,
        })
    }
}
pub(crate) struct Lease {
    accounting: Arc<Accounting>,
    pub(crate) bytes: usize,
}
impl Lease {
    pub(crate) fn acquire(accounting: &Arc<Accounting>, bytes: usize) -> Result<Self> {
        let old = accounting
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                old.checked_add(bytes).filter(|n| *n <= POLICY_PREPARED)
            })
            .map_err(|retained| {
                Error::Config(format!("Policy prepared owners exceed 4 MiB; retained {retained} bytes, requested {bytes} bytes; release retired frames"))
            })?;
        accounting.peak.fetch_max(old + bytes, Ordering::AcqRel);
        Ok(Self {
            accounting: accounting.clone(),
            bytes,
        })
    }
    /// Release an allocation only after its backing storage has been dropped.
    fn release(&mut self, bytes: usize) {
        assert!(
            bytes <= self.bytes,
            "cannot release more than the admitted lease"
        );
        self.bytes -= bytes;
        self.accounting.bytes.fetch_sub(bytes, Ordering::AcqRel);
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.accounting
            .bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub(crate) struct SharedLease {
    accounting: Arc<Accounting>,
    pub(crate) key: usize,
    pub(crate) bytes: usize,
}
impl SharedLease {
    /// Allocation keys, rather than source identities, determine actual sharing.
    pub(crate) fn additional_charge(
        accounting: &Accounting,
        allocations: &[(usize, usize)],
    ) -> Result<usize> {
        Self::additional_charge_after(accounting, allocations, &[])
    }
    /// Calculate a future charge after the specified owners have retired.
    pub(crate) fn additional_charge_after(
        accounting: &Accounting,
        allocations: &[(usize, usize)],
        retiring: &[SharedLease],
    ) -> Result<usize> {
        let shared = accounting
            .shared
            .lock()
            .map_err(|_| Error::State("Policy accounting lock poisoned".into()))?;
        Ok(allocations
            .iter()
            .enumerate()
            .filter(|(i, (key, _))| {
                !allocations[..*i].iter().any(|(old, _)| old == key)
                    && shared.get(key).is_none_or(|(count, _)| {
                        *count == retiring.iter().filter(|lease| lease.key == *key).count()
                    })
            })
            .map(|(_, (_, bytes))| *bytes)
            .sum())
    }
    /// Only allocations whose last registered owner will retire may be removed.
    pub(crate) fn exclusive_charge(leases: &[&Self]) -> usize {
        let Some(first) = leases.first() else {
            return 0;
        };
        let Ok(shared) = first.accounting.shared.lock() else {
            return 0;
        };
        leases
            .iter()
            .enumerate()
            .filter(|(i, lease)| {
                !leases[..*i].iter().any(|old| old.key == lease.key)
                    && shared.get(&lease.key).is_some_and(|(count, _)| {
                        *count == leases.iter().filter(|s| s.key == lease.key).count()
                    })
            })
            .map(|(_, lease)| lease.bytes)
            .sum()
    }
    pub(crate) fn acquire(accounting: &Arc<Accounting>, key: usize, bytes: usize) -> Result<Self> {
        let mut shared = accounting
            .shared
            .lock()
            .map_err(|_| Error::State("Policy accounting lock poisoned".into()))?;
        if let Some((count, _)) = shared.get_mut(&key) {
            *count += 1;
        } else {
            shared.insert(key, (1, Lease::acquire(accounting, bytes)?));
        }
        Ok(Self {
            accounting: accounting.clone(),
            key,
            bytes,
        })
    }
}
impl Drop for SharedLease {
    fn drop(&mut self) {
        if let Ok(mut shared) = self.accounting.shared.lock()
            && let Some((count, _)) = shared.get_mut(&self.key)
        {
            *count -= 1;
            if *count == 0 {
                shared.remove(&self.key);
            }
        }
    }
}
pub(crate) struct RequestOwner {
    pub request: PolicyRequest,
    _lease: Lease,
    _shared: Vec<SharedLease>,
}
impl RequestOwner {
    pub(crate) fn shared_leases(&self) -> &[SharedLease] {
        &self._shared
    }
    fn metadata_layout(request: &PolicyRequest) -> (usize, Vec<(usize, usize)>) {
        let mut allocations = vec![(
            Arc::as_ptr(&request.slice) as usize,
            request.slice.owned_bytes(),
        )];
        for source in std::iter::once(&request.current)
            .chain(request.reference.as_ref())
            .chain(request.input_population.as_ref())
        {
            allocations.push(source.shared_metadata());
        }
        let private =
            request.metadata_bytes() - allocations.iter().map(|(_, bytes)| bytes).sum::<usize>();
        (private, allocations)
    }
    pub(crate) fn metadata_charge(
        request: &PolicyRequest,
        accounting: &Accounting,
    ) -> Result<usize> {
        let (private, allocations) = Self::metadata_layout(request);
        Ok(private + SharedLease::additional_charge(accounting, &allocations)?)
    }

    fn additional_bytes(&self, retained: &Self) -> usize {
        self._lease.bytes
            + self
                ._shared
                .iter()
                .filter(|lease| !retained._shared.iter().any(|old| old.key == lease.key))
                .map(|lease| lease.bytes)
                .sum::<usize>()
    }
    pub fn new(request: PolicyRequest, accounting: &Arc<Accounting>) -> Result<Arc<Self>> {
        request.validate()?;
        let (bytes, allocations) = Self::metadata_layout(&request);
        let shared = allocations
            .into_iter()
            .map(|(key, bytes)| SharedLease::acquire(accounting, key, bytes))
            .collect::<Result<Vec<_>>>()?;
        let lease = Lease::acquire(accounting, bytes)?;
        Ok(Arc::new(Self {
            request,
            _lease: lease,
            _shared: shared,
        }))
    }
}

enum FrameLease {
    Historical(Lease),
    Cached(super::cache::CacheLease),
}
impl FrameLease {
    fn bytes(&self) -> usize {
        match self {
            Self::Historical(lease) => lease.bytes,
            Self::Cached(lease) => lease.bytes(),
        }
    }
}
enum GridSource {
    Historical(Arc<RequestOwner>),
    Cached {
        current: Arc<super::cache::LoadedSurface>,
        reference: Option<Arc<super::cache::LoadedSurface>>,
    },
}
impl GridSource {
    fn historical(&self) -> Option<&Arc<RequestOwner>> {
        match self {
            Self::Historical(owner) => Some(owner),
            Self::Cached { .. } => None,
        }
    }
    fn request(&self) -> Option<&PolicyRequest> {
        self.historical().map(|o| &o.request)
    }
}
pub struct PreparedGrid {
    /// Genuine source rows, independent of spatial filtering; indices remain original 0..N.
    pub population_count: usize,
    /// Distinct filtered input coordinates, capped at five for presentation.
    /// Computed by the worker; original population rows remain untouched.
    pub distinct_positions: usize,
    pub config: Arc<SliceConfig>,
    pub current: SourceIdentity,
    pub reference: Option<SourceIdentity>,
    /// Checked projection semantics from each source, independent of Selected.
    pub(crate) current_indicator: bool,
    pub(crate) reference_indicator: bool,
    pub values: Vec<f64>,
    /// Missing reference samples are NaN, never interpolated. Use reference_at.
    reference_values: Vec<f64>,
    /// Empty for a regular grid, otherwise genuine recorded row indices.
    pub rows: Vec<usize>,
    pub positions: Vec<[f32; 2]>,
    pub visits: Option<PreparedVisitation>,
    source: GridSource,
    _lease: FrameLease,
}
impl PreparedGrid {
    pub fn cache_entry(&self) -> Option<&rl::diagnostics::policy_cache::CacheEntry> {
        match &self.source {
            GridSource::Cached { current, .. } => Some(&current.entry),
            _ => None,
        }
    }
    pub fn reference_cache_entry(&self) -> Option<&rl::diagnostics::policy_cache::CacheEntry> {
        match &self.source {
            GridSource::Cached { reference, .. } => reference.as_ref().map(|r| r.entry.as_ref()),
            _ => None,
        }
    }
    /// Actual owned component-vector capacities, independent of provider.
    pub fn projection_bytes(&self) -> usize {
        self.values
            .capacity()
            .saturating_add(self.reference_values.capacity())
            .saturating_mul(8)
    }
    pub fn is_cached(&self) -> bool {
        matches!(self.source, GridSource::Cached { .. })
    }
    pub fn promote_selected(&self) -> bool {
        self.population_count > 0
            && (self.config.mode == SliceMode::SamplesOnly || self.distinct_positions <= 4)
    }
    pub fn population_selection(&self, row: usize) -> Option<SelectionIntent> {
        if row >= self.population_count {
            return None;
        }
        let Record::ProbeSet(p) = &self.recorded_probes()?.record else {
            return None;
        };
        Some(SelectionIntent::Input(SelectedInput::ProbeRow {
            set: p.id.clone(),
            row,
            sequence: self.source.request()?.input_frame()?.probe_key.sequence,
        }))
    }
    pub(crate) fn matches_request(&self, request: &PolicyRequest) -> bool {
        self.source.request().is_some_and(|r| r.same_grid(request))
    }
    pub(crate) fn reference_source(&self) -> Option<&PolicySource> {
        self.source.request()?.reference.as_ref()
    }
    pub fn regular(&self) -> bool {
        self.config.mode != SliceMode::SamplesOnly
            && self.population_count == 0
            && !self.current.recorded
    }
    pub fn reference_at(&self, index: usize) -> Option<f64> {
        self.reference_values
            .get(index)
            .copied()
            .filter(|v| !v.is_nan())
    }
    pub fn difference_at(&self, index: usize) -> Option<f64> {
        self.values
            .get(index)
            .zip(self.reference_at(index))
            .map(|(a, b)| a - b)
    }
    /// Borrow the genuine complete conditioning arrays, including histories.
    pub fn recorded_probes(&self) -> Option<&Chunk> {
        self.source
            .request()?
            .input_frame()
            .map(|f| f.probes.as_ref())
    }
    /// Describes this evaluated cell, never arbitrary cursor coordinates.
    pub fn input(&self, index: usize) -> Result<Observation> {
        ensure(
            index < self.values.len(),
            "Policy cell outside prepared frame",
        )?;
        if let GridSource::Cached { current, .. } = &self.source {
            // Both endpoints are retained by this presentation owner. Exact
            // cached inputs, never arbitrary cursor coordinates, are exposed.
            return current.surface.input(index);
        }
        if let Some(frame) = self.source.request().and_then(PolicyRequest::input_frame) {
            let mut input = exact_observation(&frame.probes, self.rows[index])?;
            // Worker filtering certified every retained row against this exact
            // effective anchor mask. Hover needs no policy scan or validation.
            input
                .as_dense_mut()
                .expect("dense diagnostic input")
                .action_mask = self.config.anchor_mask().map(<[bool]>::to_vec);
            Ok(input)
        } else {
            self.config.observation_at(index)
        }
    }
    pub fn selection_intent(&self, index: usize) -> Option<SelectionIntent> {
        let source = self.source.request()?.input_frame()?;
        let Record::ProbeSet(p) = &source.probes.record else {
            return None;
        };
        Some(SelectionIntent::Input(SelectedInput::ProbeRow {
            set: p.id.clone(),
            row: *self.rows.get(index)?,
            sequence: source.probe_key.sequence,
        }))
    }
}
#[derive(Clone)]
pub struct PreparedPolicyFrame {
    pub grid: Arc<PreparedGrid>,
    pub selected: Option<Arc<PreparedPoint>>,
    pub requested_revision: u64,
}
impl PreparedPolicyFrame {
    pub(crate) fn retiring_owner<'a>(
        &'a self,
        desired: Option<&Arc<RequestOwner>>,
    ) -> Option<&'a RequestOwner> {
        let owner = self.grid.source.historical()?;
        let grid_retires = Arc::strong_count(&self.grid) == 1;
        let known =
            usize::from(grid_retires) + usize::from(desired.is_some_and(|d| Arc::ptr_eq(d, owner)));
        (Arc::strong_count(owner) == known).then_some(owner)
    }
    pub(crate) fn retirement_bytes(&self, desired: Option<&Arc<RequestOwner>>) -> usize {
        let Some(owner) = self.grid.source.historical() else {
            return 0;
        };
        let grid_retires = Arc::strong_count(&self.grid) == 1;
        let mut bytes = if grid_retires {
            self.grid._lease.bytes()
        } else {
            0
        };
        bytes += self
            .selected
            .as_ref()
            .filter(|p| Arc::strong_count(p) == 1)
            .map_or(0, |p| {
                p._lease.bytes()
                    + p.action_value
                        .as_ref()
                        .map_or(0, |curve| curve._lease.bytes)
            });
        // The grid and optional desired request are its only internal owners;
        // texture caches are Weak. Any external Arc makes metadata non-retiring.
        if self.retiring_owner(desired).is_some() {
            bytes += owner._lease.bytes
                + SharedLease::exclusive_charge(&owner._shared.iter().collect::<Vec<_>>());
        }
        bytes
    }
}
pub struct PreparedPoint {
    /// Exact numerical-cache grid row, independent of arbitrary cursor input.
    pub cached_row: Option<usize>,
    pub chosen_action: Option<usize>,
    pub reference_chosen_action: Option<usize>,
    pub action_statistics: Option<rl::diagnostics::policy::GaussianActionStatistics>,
    pub reference_statistics: Option<rl::diagnostics::policy::GaussianActionStatistics>,
    pub reference_recorded_input: Option<SelectedInput>,
    pub input: Observation,
    pub recorded_input: Option<SelectedInput>,
    pub value: f64,
    pub reference_value: Option<f64>,
    pub difference: Option<f64>,
    pub kl: Option<f64>,
    pub distribution: SelectedDistribution,
    pub reference_distribution: Option<SelectedDistribution>,
    pub plot: Option<DistributionPlot>,
    pub reference_plot: Option<DistributionPlot>,
    pub visits: Option<u32>,
    cross_section: Option<Arc<PreparedCrossSection>>,
    action_value: Option<PreparedPolicyActionValue>,
    _lease: FrameLease,
}
/// Same-snapshot critic values at the selected observation and fixed remaining
/// action components. The actor marker is evaluated at its exact action.
pub struct PreparedPolicyActionValue {
    pub critic: String,
    pub component: usize,
    pub reward_units: String,
    pub role: String,
    pub points: Vec<[f64; 2]>,
    pub actor_action: f64,
    pub actor_value: f64,
    _lease: Lease,
}
/// A selected-input slice through a deterministic model and any matching
/// recorded reference samples. Coordinates other than `varying_axis` are held
/// at the exact selected observation, not at a nearby grid column.
pub struct PreparedCrossSection {
    pub fixed_value: f32,
    pub varying_axis: usize,
    pub points: Vec<[f64; 2]>,
    pub reference_samples: Vec<[f64; 2]>,
    pub omitted_reference_samples: usize,
    pub reference_scan_limited: bool,
}
impl PreparedPoint {
    pub fn cross_section(&self) -> Option<&PreparedCrossSection> {
        self.cross_section.as_deref()
    }
    pub fn action_value_curve(&self) -> Option<&PreparedPolicyActionValue> {
        self.action_value.as_ref()
    }
}

/// Prepare an optional same-snapshot critic slice on the existing historical
/// worker. Unsupported, ambiguous or over-budget cases leave Policy usable.
pub(crate) fn attach_action_value_curve(
    point: &mut Arc<PreparedPoint>,
    owner: &Arc<RequestOwner>,
    grid: &PreparedGrid,
    model: &rl::diagnostics::models::ModelBundle,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
) -> Result<()> {
    use rl::{
        core::action::Action,
        diagnostics::{
            models::{CriticInspectionRequest, InspectionRequest},
            value::{ValueOutput, ValuePrediction},
        },
    };
    let request = &owner.request;
    let Some(spec) = request.current.action_value_critic() else {
        return Ok(());
    };
    if !matches!(point.distribution, SelectedDistribution::Deterministic) || cancelled() {
        return Ok(());
    }
    let ActionSpace::Box { low, high } = request.current.semantics().action_space() else {
        return Ok(());
    };
    let component = request.slice.action;
    let (Some(&lo), Some(&hi)) = (low.get(component), high.get(component)) else {
        return Ok(());
    };
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Ok(());
    }
    let row = request
        .current
        .semantics()
        .observation_dim()
        .checked_mul(64)
        .and_then(|n| low.len().checked_mul(96).and_then(|a| n.checked_add(a)))
        .and_then(|n| spec.heads.checked_mul(96).and_then(|h| n.checked_add(h)))
        .and_then(|n| n.checked_add(2048))
        .ok_or_else(|| Error::Config("Policy critic batch overflow".into()))?;
    let count = ((POLICY_SCRATCH - 256 * 1024) / row)
        .saturating_sub(1)
        .min(65);
    if count < 2 {
        return Ok(());
    }
    accounting.scratch(POLICY_SCRATCH)?;
    let actor = selected_name(model.configuration(), request.current.name())?;
    let started = Instant::now();
    let inspected = model.inspect(&InspectionRequest::new(actor, vec![point.input.clone()]));
    accounting
        .inference_micros
        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
    accounting.point_batches.fetch_add(1, Ordering::AcqRel);
    let inspected = inspected?;
    let Some(Action::Continuous(base)) = inspected.actions.into_iter().next() else {
        return Ok(());
    };
    if base.len() != low.len() || f64::from(base[component]).to_bits() != point.value.to_bits() {
        return Ok(());
    }
    let mut actions = Vec::with_capacity(count + 1);
    for index in 0..count {
        let mut values = base.clone();
        values[component] = (f64::from(lo)
            + (f64::from(hi) - f64::from(lo)) * index as f64 / (count - 1) as f64)
            as f32;
        let action = Action::Continuous(values);
        request
            .current
            .semantics()
            .action_space()
            .check(&action, point.input.dense_mask())?;
        actions.push(action);
    }
    actions.push(Action::Continuous(base.clone()));
    if cancelled() {
        return Ok(());
    }
    let started = Instant::now();
    let result = model.inspect_critics(&CriticInspectionRequest {
        critics: vec![spec.name.clone()],
        observations: vec![point.input.clone(); actions.len()],
        fixed_actions: Some(actions.clone()),
    });
    accounting
        .inference_micros
        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
    accounting.point_batches.fetch_add(1, Ordering::AcqRel);
    let result = result?;
    let critic = result
        .critics
        .get(&spec.name)
        .ok_or_else(|| Error::State("Critic result missing".into()))?;
    let mut values = Vec::with_capacity(count);
    for (index, action) in actions.iter().take(count).enumerate() {
        let Action::Continuous(action) = action else {
            unreachable!()
        };
        let value = ValuePrediction::from_critic(critic, index, None)?
            .scalar(&ValueOutput::ExpectedReturn, None)?;
        values.push([f64::from(action[component]), value]);
    }
    let actor_value = ValuePrediction::from_critic(critic, count, None)?
        .scalar(&ValueOutput::ExpectedReturn, None)?;
    request.current.check_available()?;
    if cancelled() {
        return Ok(());
    }
    let name = spec.name.clone();
    let reward_units = spec.reward_units.clone();
    let role = spec.role.clone();
    let bytes = values
        .capacity()
        .saturating_mul(std::mem::size_of::<[f64; 2]>())
        .saturating_add(name.capacity())
        .saturating_add(reward_units.capacity())
        .saturating_add(role.capacity())
        .saturating_add(std::mem::size_of::<PreparedPolicyActionValue>());
    let Some(grid_owner) = grid.source.historical() else {
        return Ok(());
    };
    let additional = if Arc::ptr_eq(owner, grid_owner) {
        0
    } else {
        owner.additional_bytes(grid_owner)
    };
    ensure(
        grid._lease
            .bytes()
            .saturating_add(grid_owner.request.metadata_bytes())
            .saturating_add(additional)
            .saturating_add(point._lease.bytes())
            .saturating_add(bytes)
            <= POLICY_FRAME,
        "Policy action-value curve exceeds frame admission",
    )?;
    ensure(
        accounting
            .snapshot()
            .prepared_bytes
            .saturating_add(bytes)
            .saturating_add(512 * 1024)
            <= POLICY_PREPARED,
        "Policy action-value curve would consume validation headroom",
    )?;
    let lease = Lease::acquire(accounting, bytes)?;
    let Some(point) = Arc::get_mut(point) else {
        return Ok(());
    };
    point.action_value = Some(PreparedPolicyActionValue {
        critic: name,
        component,
        reward_units,
        role,
        points: values,
        actor_action: f64::from(base[component]),
        actor_value,
        _lease: lease,
    });
    Ok(())
}
#[derive(Debug, Clone)]
pub struct DistributionPlot {
    pub domain: [f64; 2],
    pub is_cdf: bool,
    /// Prepared on the worker, so automatic viewport expansion never scans samples on paint.
    pub ymax: f64,
    pub samples: Vec<[f64; 2]>,
}
pub fn distribution_plot(
    distribution: &SelectedDistribution,
    domain: Option<[f64; 2]>,
) -> Result<Option<DistributionPlot>> {
    let SelectedDistribution::Gaussian(g) = distribution else {
        return Ok(None);
    };
    let domain = domain.or(g.bounds).ok_or_else(|| {
        Error::Config("Unbounded Gaussian chart requires a stable explicit action domain".into())
    })?;
    let [lo, hi] = domain;
    ensure(
        lo.is_finite() && hi.is_finite() && lo < hi && (hi - lo).is_finite(),
        "Invalid stable action domain",
    )?;
    let step = (hi - lo) / (DISTRIBUTION_SAMPLES - 1) as f64;
    let mut samples = Vec::with_capacity(DISTRIBUTION_SAMPLES);
    let mut mass = 0.;
    let mut finite = true;
    for i in 0..DISTRIBUTION_SAMPLES {
        let x = if i + 1 == DISTRIBUTION_SAMPLES {
            hi
        } else {
            lo + i as f64 * step
        };
        let y = g.density(x).unwrap_or(f64::NAN);
        finite &= y.is_finite();
        mass += y
            * step
            * if i == 0 || i + 1 == DISTRIBUTION_SAMPLES {
                0.5
            } else {
                1.
            };
        samples.push([x, y]);
    }
    let covered = g.cdf(hi)? - g.cdf(lo)?;
    let is_cdf = !finite || (mass - covered).abs() > 0.02 || (covered > 0.01 && mass <= 0.);
    if is_cdf {
        for sample in &mut samples {
            sample[1] = g.cdf(sample[0])?;
        }
    }
    Ok(Some(DistributionPlot {
        domain,
        is_cdf,
        ymax: if is_cdf {
            1.
        } else {
            samples.iter().map(|p| p[1]).fold(0., f64::max)
        },
        samples,
    }))
}
pub fn distribution_plots(
    current: &SelectedDistribution,
    reference: Option<&SelectedDistribution>,
    domain: Option<[f64; 2]>,
) -> Result<(Option<DistributionPlot>, Option<DistributionPlot>)> {
    // Initialize a display extent once on the historical worker. This never
    // truncates or renormalizes the distribution; the UI retains the returned
    // common domain until a deliberate action/Fit edit.
    let domain = if domain.is_some() {
        domain
    } else {
        let mut extent: Option<[f64; 2]> = None;
        for distribution in std::iter::once(current).chain(reference) {
            if let SelectedDistribution::Gaussian(g) = distribution {
                let bounds = g.bounds.unwrap_or_else(|| {
                    let radius = 5. * g.log_std.exp();
                    [g.mean - radius, g.mean + radius]
                });
                ensure(
                    bounds[0].is_finite()
                        && bounds[1].is_finite()
                        && bounds[0] < bounds[1]
                        && (bounds[1] - bounds[0]).is_finite(),
                    "Cannot initialize finite Gaussian display extent",
                )?;
                extent = Some(
                    extent.map_or(bounds, |old| [old[0].min(bounds[0]), old[1].max(bounds[1])]),
                );
            }
        }
        extent
    };
    let mut current_plot = distribution_plot(current, domain)?;
    let domain = current_plot.as_ref().map(|p| p.domain).or(domain);
    let mut reference_plot = reference
        .map(|d| distribution_plot(d, domain))
        .transpose()?
        .flatten();
    if current_plot.as_ref().is_some_and(|p| p.is_cdf)
        || reference_plot.as_ref().is_some_and(|p| p.is_cdf)
    {
        for (distribution, plot) in std::iter::once((current, &mut current_plot))
            .chain(reference.zip(Some(&mut reference_plot)))
        {
            if let (SelectedDistribution::Gaussian(g), Some(plot)) = (distribution, plot) {
                plot.is_cdf = true;
                plot.ymax = 1.;
                for sample in &mut plot.samples {
                    sample[1] = g.cdf(sample[0])?;
                }
            }
        }
    }
    Ok((current_plot, reference_plot))
}

pub(crate) enum Work {
    Grid(Arc<RequestOwner>),
    Point {
        request: Arc<RequestOwner>,
        grid: Arc<PreparedGrid>,
        previous: Option<Arc<PreparedPoint>>,
    },
}
pub(crate) enum Completion {
    ReferenceUnavailable,
    Grid(Arc<PreparedGrid>),
    Point(Arc<PreparedPoint>),
}

/// `inspect` is supplied by the existing worker/cache, not a second executor.
pub(crate) fn execute(
    work: &Work,
    accounting: &Arc<Accounting>,
    cancelled: impl Fn() -> bool + Sync,
    mut inspect: impl FnMut(&PolicySource, &[Observation]) -> Result<(InspectionResult, String)>,
) -> Result<Option<Completion>> {
    let request = match work {
        Work::Grid(owner) => &owner.request,
        Work::Point { request, .. } => &request.request,
    };
    request.current.check_available()?;
    request.check_inputs()?;
    if request
        .reference
        .as_ref()
        .is_some_and(|r| r.check_available().is_err())
    {
        return Ok(Some(Completion::ReferenceUnavailable));
    }
    let mut reference_failed = false;
    let mut inspect_optional = |source: &PolicySource, observations: &[Observation]| {
        let result = inspect(source, observations);
        reference_failed |= result.is_err()
            && request
                .reference
                .as_ref()
                .is_some_and(|r| std::ptr::eq(r, source));
        result
    };
    let start = Instant::now();
    let inference_before = accounting.inference_micros.load(Ordering::Acquire);
    let restore_before = accounting.restore_micros.load(Ordering::Acquire);
    let result = match work {
        Work::Grid(owner) => prepare_grid(owner, accounting, &cancelled, &mut inspect_optional)
            .map(|g| g.map(Completion::Grid)),
        Work::Point {
            request,
            grid,
            previous,
        } => prepare_point(
            request,
            grid,
            previous.as_deref(),
            accounting,
            &cancelled,
            &mut inspect_optional,
        )
        .map(|p| p.map(Completion::Point)),
    };
    let inference = accounting.inference_micros.load(Ordering::Acquire) - inference_before;
    let restore = accounting.restore_micros.load(Ordering::Acquire) - restore_before;
    accounting.preparation_micros.fetch_add(
        (start.elapsed().as_micros() as u64).saturating_sub(inference + restore),
        Ordering::AcqRel,
    );
    // Current-source failures take precedence over optional comparison loss.
    // Only successful work or a classified reference-inspection failure can
    // retry; allocation/input failures must retain their original hard error.
    request.current.check_available()?;
    request.check_inputs()?;
    let optional_lost = result.as_ref().is_ok_and(|answer| answer.is_some())
        && request
            .reference
            .as_ref()
            .is_some_and(|r| r.check_available().is_err());
    if (result.is_err() && reference_failed) || optional_lost {
        Ok(Some(Completion::ReferenceUnavailable))
    } else {
        result
    }
}

type Inspect<'a> =
    dyn FnMut(&PolicySource, &[Observation]) -> Result<(InspectionResult, String)> + 'a;
fn stop(cancelled: &impl Fn() -> bool, accounting: &Accounting) -> bool {
    if cancelled() {
        accounting.cancellations.fetch_add(1, Ordering::AcqRel);
        true
    } else {
        false
    }
}

fn project_rows(
    output: &mut [f64],
    project: impl Fn(usize) -> Result<f64> + Sync,
    cancelled: impl Fn() -> bool + Sync,
) -> Result<bool> {
    let parallelism = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(8);
    project_rows_configured(output, project, cancelled, parallelism, |_| true)
}

fn project_rows_configured(
    output: &mut [f64],
    project: impl Fn(usize) -> Result<f64> + Sync,
    cancelled: impl Fn() -> bool + Sync,
    parallelism: usize,
    may_spawn: impl Fn(usize) -> bool,
) -> Result<bool> {
    if output.len() < 128 || parallelism == 1 {
        for (row, value) in output.iter_mut().enumerate() {
            if cancelled() {
                return Ok(false);
            }
            *value = project(row)?;
        }
        return Ok(true);
    }
    let width = output.len().div_ceil(parallelism);
    let result = std::thread::scope(|scope| -> Result<Option<bool>> {
        let mut workers = Vec::with_capacity(parallelism);
        let project = &project;
        let cancelled = &cancelled;
        let mut spawn_failed = false;
        for (part, values) in output.chunks_mut(width).enumerate() {
            if !may_spawn(part) {
                spawn_failed = true;
                break;
            }
            match std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn_scoped(scope, move || -> Result<bool> {
                    for (offset, value) in values.iter_mut().enumerate() {
                        if cancelled() {
                            return Ok(false);
                        }
                        *value = project(part * width + offset)?;
                    }
                    Ok(true)
                }) {
                Ok(worker) => workers.push(worker),
                Err(_) => {
                    spawn_failed = true;
                    break;
                }
            }
        }
        let mut complete = true;
        for worker in workers {
            complete &= worker
                .join()
                .map_err(|_| Error::State("policy projection worker panicked".into()))??;
        }
        Ok(if !complete {
            Some(false)
        } else if spawn_failed {
            None
        } else {
            Some(true)
        })
    })?;
    if let Some(complete) = result {
        return Ok(complete);
    }
    for (row, value) in output.iter_mut().enumerate() {
        if cancelled() {
            return Ok(false);
        }
        *value = project(row)?;
    }
    Ok(true)
}

#[cfg(test)]
#[test]
fn projection_falls_back_when_a_worker_cannot_start() {
    let mut output = vec![0.; 256];
    let complete = project_rows_configured(
        &mut output,
        |row| Ok(row as f64 * 0.5),
        || false,
        4,
        |part| part == 0,
    )
    .unwrap();
    assert!(complete);
    for (row, value) in output.iter().enumerate() {
        assert_eq!(*value, row as f64 * 0.5);
    }
}

#[cfg(test)]
#[test]
fn independent_projection_rows_keep_exact_order_and_use_multiple_workers() {
    use rl::diagnostics::policy::GaussianMarginal;
    use std::{collections::HashSet, sync::Mutex};
    let gaussian = |row| GaussianMarginal {
        mean: (row as f64 - 128.) / 64.,
        log_std: -0.7,
        bounds: Some([-1., 1.]),
    };
    let workers = Mutex::new(HashSet::new());
    let mut actual = vec![0.; 256];
    let completed = project_rows(
        &mut actual,
        |row| {
            workers.lock().unwrap().insert(std::thread::current().id());
            gaussian(row).expected_action()
        },
        || false,
    )
    .unwrap();
    assert!(completed);
    for (row, value) in actual.iter().enumerate() {
        assert_eq!(
            value.to_bits(),
            gaussian(row).expected_action().unwrap().to_bits()
        );
    }
    let count = workers.lock().unwrap().len();
    if std::thread::available_parallelism().is_ok_and(|n| n.get() > 1) {
        assert!(count > 1, "large projections must use more than one worker");
    } else {
        assert_eq!(count, 1);
    }
}

#[cfg(test)]
#[test]
fn parallel_projection_stops_on_cancellation_and_reports_errors() {
    let mut output = vec![0.; 256];
    assert!(!project_rows(&mut output, |_| Ok(1.), || true).unwrap());
    let failure = project_rows(
        &mut output,
        |row| {
            if row == 130 {
                Err(Error::Config("bad projection".into()))
            } else {
                Ok(1.)
            }
        },
        || false,
    )
    .unwrap_err();
    assert!(failure.to_string().contains("bad projection"));
}
/// The same charge used by allocation admission, without allocating a grid.
fn grid_charge(request: &PolicyRequest, count: usize, recorded: bool) -> Result<usize> {
    let bytes = count
        .checked_mul(if recorded { 32 } else { 16 })
        .and_then(|n| {
            n.checked_add(
                request
                    .slice
                    .len()
                    .ok()?
                    .checked_mul(if request.visits.is_some() { 4 } else { 0 })?,
            )
        })
        .and_then(|n| n.checked_add(32 * 1024))
        .ok_or_else(|| Error::Config("Policy frame size overflow".into()))?;
    ensure(
        bytes + request.metadata_bytes() <= POLICY_FRAME,
        "Policy grid exceeds 2 MiB frame allowance",
    )?;
    Ok(bytes)
}
pub(crate) fn publication_charge(request: &PolicyRequest) -> Result<usize> {
    let count = if let Some(frame) = request.input_frame() {
        let Record::ProbeSet(p) = &frame.probes.record else {
            return Err(Error::Config("Missing input probes".into()));
        };
        frame.probes.arrays[&p.observations].shape()[0]
    } else {
        request.slice.len()?
    };
    let grid = grid_charge(request, count, request.input_frame().is_some())?;
    let point = action_width(request.current.semantics().action_space()) * 18
        + request.slice.fixed.capacity() * 4
        + request.slice.anchor_mask().map_or(0, <[bool]>::len)
        + 32 * 1024;
    let bytes = grid
        .saturating_add(point)
        .saturating_add(request.metadata_bytes());
    ensure(bytes <= POLICY_FRAME, "Policy complete frame exceeds 2 MiB")?;
    Ok(bytes)
}

fn recorded_match_fits(count: usize, rows: usize) -> bool {
    count
        .checked_mul(rows)
        .is_some_and(|n| n <= MAX_RECORDED_MATCH_CHECKS)
}

/// Check the same bounded matching rule used during preparation before an
/// automatically selected reference can prevent the learner map appearing.
pub(crate) fn reference_match_fits(request: &PolicyRequest) -> Result<bool> {
    let Some(source) = &request.reference else {
        return Ok(true);
    };
    let Some(frame) = source
        .recorded_frame()
        .filter(|_| !source.finite_reference())
    else {
        return Ok(true);
    };
    let rows = admit_reference(source, frame, request.current.semantics().action_space())?.rows();
    let count = match request.input_frame() {
        Some(frame) => admit_recorded(frame, request.current.semantics().action_space())?.rows(),
        None => request.slice.len()?,
    };
    Ok(recorded_match_fits(count, rows))
}
fn prepare_grid(
    owner: &Arc<RequestOwner>,
    accounting: &Arc<Accounting>,
    cancelled: &(impl Fn() -> bool + Sync),
    inspect: &mut Inspect<'_>,
) -> Result<Option<Arc<PreparedGrid>>> {
    let request = &owner.request;
    request.current.check_available()?;
    let recorded = request.input_frame();
    let recorded_batch = recorded
        .map(|frame| admit_recorded(frame, request.current.semantics().action_space()))
        .transpose()?;
    let count = if let Some(batch) = &recorded_batch {
        batch.rows()
    } else {
        ensure(
            request.slice.grid_eligibility() == GridEligibility::ContinuousFeedforward,
            "Synthetic grid unavailable for masked or history-bound inputs; use recorded samples",
        )?;
        request.slice.len()?
    };
    if let Some(reference) = &request.reference {
        reference.prepare_reference(accounting, cancelled)?;
    }
    let recorded_reference = request
        .reference
        .as_ref()
        .and_then(|source| source.recorded_frame().map(|frame| (source, frame)))
        .map(|(source, frame)| {
            admit_reference(source, frame, request.current.semantics().action_space())
        })
        .transpose()?;
    if let Some(reference) = &recorded_reference
        && !request.reference.as_ref().unwrap().finite_reference()
    {
        let rows = reference.rows();
        ensure(
            recorded_match_fits(count, rows),
            "Recorded reference matching exceeds bounded row-check budget; reduce the slice/population",
        )?;
    }
    let bytes = grid_charge(request, count, recorded.is_some())?;
    let lease = Lease::acquire(accounting, bytes)?;
    let spatial_count = if request.slice.mode == SliceMode::SamplesOnly {
        0
    } else {
        count
    };
    let mut grid = PreparedGrid {
        population_count: recorded_batch.as_ref().map_or(0, PolicyBatch::rows),
        distinct_positions: 0,
        config: request.slice.clone(),
        current: request.current.identity.clone(),
        reference: request.reference.as_ref().map(|r| r.identity.clone()),
        current_indicator: recorded_batch.as_ref().is_some_and(numerics::is_indicator),
        reference_indicator: recorded_reference
            .as_ref()
            .is_some_and(numerics::is_indicator),
        values: Vec::with_capacity(spatial_count),
        reference_values: Vec::with_capacity(spatial_count),
        rows: Vec::with_capacity(if recorded.is_some() { spatial_count } else { 0 }),
        positions: Vec::with_capacity(if recorded.is_some() { spatial_count } else { 0 }),
        visits: None,
        source: GridSource::Historical(owner.clone()),
        _lease: FrameLease::Historical(lease),
    };
    if let Some(frame) = recorded.filter(|_| request.input_population.is_none()) {
        let mut distinct = [[0_f32; 2]; 5];
        let batch = recorded_batch.as_ref().unwrap();
        let reference = &recorded_reference;
        accounting.scratch(
            FIXED_SCRATCH
                + (request.slice.fixed.len() * 64
                    + action_width(request.current.semantics().action_space()) * 256
                    + 1024)
                    * if request.reference.is_some() { 2 } else { 1 },
        )?;
        for row in 0..count {
            if stop(cancelled, accounting) {
                return Ok(None);
            }
            accounting.source_rows.fetch_add(1, Ordering::AcqRel);
            if request.slice.mode == SliceMode::SamplesOnly {
                continue;
            }
            let input = recorded_input(batch, row)?;
            if request.slice.visit_bin(&input)?.is_none() {
                continue;
            }
            grid.values
                .push(numerics::project(batch, row, request.slice.action)?);
            grid.rows.push(row);
            grid.positions.push([
                input.dense_values()[request.slice.x],
                request.slice.y.map_or(0., |y| input.dense_values()[y]),
            ]);
            let position = *grid.positions.last().unwrap();
            if grid.distinct_positions < distinct.len()
                && !distinct[..grid.distinct_positions].contains(&position)
            {
                distinct[grid.distinct_positions] = position;
                grid.distinct_positions += 1;
            }
            let value = if let (Some(source), Some(reference)) = (&request.reference, &reference) {
                let other = source.recorded_frame().unwrap();
                source
                    .matching_row(
                        &frame.probes,
                        row,
                        &other.probes,
                        reference,
                        input.dense_mask(),
                    )?
                    .map(|r| numerics::project(reference, r, request.slice.action))
                    .transpose()?
            } else if let Some(source) = &request.reference {
                ensure_recorded_feedforward(&frame.probes)?;
                let (result, name) = inspect(source, std::slice::from_ref(&input))?;
                grid.reference.as_mut().unwrap().name = name;
                let batch = checked_batch(
                    &result.policy,
                    &result.probes,
                    request.current.semantics().action_space(),
                    POLICY_SCRATCH,
                )?;
                grid.reference_indicator = numerics::is_indicator(&batch);
                Some(numerics::project(&batch, 0, request.slice.action)?)
            } else {
                None
            };
            grid.reference_values.push(value.unwrap_or(f64::NAN));
        }
        ensure(
            request.slice.mode == SliceMode::SamplesOnly || !grid.values.is_empty(),
            "No recorded samples match the fixed input slice",
        )?;
    } else {
        let width = action_width(request.current.semantics().action_space());
        let capacity = batch_capacity(
            request.slice.fixed.len(),
            width,
            request.reference.is_some(),
        )?;
        for first in (0..count).step_by(capacity) {
            if stop(cancelled, accounting) {
                return Ok(None);
            }
            let n = (count - first).min(capacity);
            accounting.scratch(
                FIXED_SCRATCH
                    + n * (request.slice.fixed.len() * 64 + width * 256 + 1024)
                        * if request.reference.is_some() { 2 } else { 1 },
            )?;
            if request.slice.mode == SliceMode::SamplesOnly {
                continue;
            }
            let mut original_rows = Vec::with_capacity(if recorded.is_some() { n } else { 0 });
            let mut inputs = Vec::with_capacity(n);
            for i in first..first + n {
                let input = if let Some(batch) = &recorded_batch {
                    recorded_input(batch, i)?
                } else {
                    request.slice.observation_at(i)?
                };
                if recorded.is_some() && request.slice.visit_bin(&input)?.is_none() {
                    continue;
                }
                if recorded.is_some() {
                    original_rows.push(i);
                }
                inputs.push(input);
            }
            if inputs.is_empty() {
                continue;
            }
            let (result, name) = inspect(&request.current, &inputs)?;
            grid.current.name = name;
            accounting.grid_batches.fetch_add(1, Ordering::AcqRel);
            let batch = checked_batch(
                &result.policy,
                &result.probes,
                request.current.semantics().action_space(),
                POLICY_SCRATCH,
            )?;
            grid.current_indicator = numerics::is_indicator(&batch);
            let reference = if let Some(source) = &request.reference {
                if source.model_source().is_some() {
                    let (result, name) = inspect(source, &inputs)?;
                    grid.reference.as_mut().unwrap().name = name;
                    Some(result)
                } else {
                    None
                }
            } else {
                None
            };
            let reference_batch = reference
                .as_ref()
                .map(|r| {
                    checked_batch(
                        &r.policy,
                        &r.probes,
                        request.current.semantics().action_space(),
                        POLICY_SCRATCH,
                    )
                })
                .transpose()?;
            if let Some(reference) = &reference_batch {
                grid.reference_indicator = numerics::is_indicator(reference);
            }
            let parallel = matches!(
                batch.parameters(),
                rl::diagnostics::schema::PolicyParameters::Gaussian { .. }
            ) && matches!(batch.action_space(), ActionSpace::Box { .. })
                && inputs.len() >= 128;
            if parallel {
                let base = grid.values.len();
                grid.values.resize(base + inputs.len(), 0.);
                if !project_rows(
                    &mut grid.values[base..],
                    |row| numerics::project(&batch, row, request.slice.action),
                    cancelled,
                )? {
                    accounting.cancellations.fetch_add(1, Ordering::AcqRel);
                    return Ok(None);
                }
            }
            let parallel_reference = reference_batch.as_ref().is_some_and(|reference| {
                matches!(
                    reference.parameters(),
                    rl::diagnostics::schema::PolicyParameters::Gaussian { .. }
                ) && matches!(reference.action_space(), ActionSpace::Box { .. })
                    && inputs.len() >= 128
            });
            if parallel_reference {
                let reference = reference_batch.as_ref().unwrap();
                let base = grid.reference_values.len();
                grid.reference_values.resize(base + inputs.len(), 0.);
                if !project_rows(
                    &mut grid.reference_values[base..],
                    |row| numerics::project(reference, row, request.slice.action),
                    cancelled,
                )? {
                    accounting.cancellations.fetch_add(1, Ordering::AcqRel);
                    return Ok(None);
                }
            }
            for (row, input) in inputs.iter().enumerate() {
                if stop(cancelled, accounting) {
                    return Ok(None);
                }
                if !parallel {
                    grid.values
                        .push(numerics::project(&batch, row, request.slice.action)?);
                }
                if recorded.is_some() {
                    grid.rows.push(original_rows[row]);
                    grid.positions.push([
                        input.dense_values()[request.slice.x],
                        request.slice.y.map_or(0., |y| input.dense_values()[y]),
                    ]);
                }
                if parallel_reference {
                    continue;
                }
                let other = if let Some(reference) = &reference_batch {
                    Some(numerics::project(reference, row, request.slice.action)?)
                } else if let Some(batch) = recorded_reference.as_ref() {
                    (if let Some(frame) = recorded {
                        request.reference.as_ref().unwrap().matching_row(
                            &frame.probes,
                            original_rows[row],
                            &request
                                .reference
                                .as_ref()
                                .unwrap()
                                .recorded_frame()
                                .unwrap()
                                .probes,
                            batch,
                            input.dense_mask(),
                        )?
                    } else if request.reference.as_ref().unwrap().finite_reference() {
                        request.reference.as_ref().unwrap().reference_info.lookup(
                            &request
                                .reference
                                .as_ref()
                                .unwrap()
                                .recorded_frame()
                                .unwrap()
                                .probes,
                            input,
                        )?
                    } else {
                        exact_policy_row(
                            batch,
                            input,
                            request
                                .reference
                                .as_ref()
                                .unwrap()
                                .identity
                                .explicit_controller,
                        )?
                    })
                    .map(|r| numerics::project(batch, r, request.slice.action))
                    .transpose()?
                } else {
                    None
                };
                grid.reference_values.push(other.unwrap_or(f64::NAN));
            }
        }
    }
    if request.input_population.is_some() {
        ensure(
            request.slice.mode == SliceMode::SamplesOnly || !grid.values.is_empty(),
            "No recorded inputs match the fixed slice",
        )?;
        let mut distinct = [[0_f32; 2]; 5];
        for &p in &grid.positions {
            if grid.distinct_positions < 5 && !distinct[..grid.distinct_positions].contains(&p) {
                distinct[grid.distinct_positions] = p;
                grid.distinct_positions += 1;
            }
        }
    }
    grid.visits = prepare_visits(request, accounting, cancelled)?;
    request.current.check_available()?;
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    let actual = grid.values.capacity() * 8
        + grid.reference_values.capacity() * 8
        + grid.rows.capacity() * size_of::<usize>()
        + grid.positions.capacity() * 8
        + grid
            .visits
            .as_ref()
            .map_or(0, |v| v.counts.capacity() * 4 + v.markers.capacity() * 8)
        + 4096;
    ensure(
        actual <= bytes,
        "Policy grid capacities exceed admitted reservation",
    )?;
    // Missing reference samples are represented as NaNs during preparation.
    // An entirely missing reference has the same public lookup semantics with
    // no vector at all, so do not retain or charge a full empty projection.
    if grid.reference_values.iter().all(|value| value.is_nan()) {
        let released = grid.reference_values.capacity() * size_of::<f64>();
        drop(std::mem::take(&mut grid.reference_values));
        if let FrameLease::Historical(lease) = &mut grid._lease {
            lease.release(released);
        }
    }
    Ok(Some(Arc::new(grid)))
}
fn admit_recorded<'a>(
    frame: &'a ResolvedRecord,
    space: &'a ActionSpace,
) -> Result<PolicyBatch<'a>> {
    admit_inputs(&frame.probes)?;
    let batch = checked_batch(
        &frame.chunk,
        &frame.probes,
        space,
        POLICY_SCRATCH - FIXED_SCRATCH,
    )?;
    ensure(
        batch.rows() <= MAX_POPULATION_ROWS,
        "Policy recorded population exceeds 8192-row bound",
    )?;
    Ok(batch)
}
fn admit_reference<'a>(
    source: &PolicySource,
    frame: &'a ResolvedRecord,
    space: &'a ActionSpace,
) -> Result<PolicyBatch<'a>> {
    if source.finite_reference() {
        checked_batch(
            &frame.chunk,
            &frame.probes,
            space,
            POLICY_SCRATCH - FIXED_SCRATCH,
        )
    } else {
        admit_recorded(frame, space)
    }
}
fn admit_inputs(chunk: &Chunk) -> Result<()> {
    let mut bytes = 0usize;
    let mut add = |key: &str| -> Result<()> {
        bytes = bytes.saturating_add(crate::plot_data::array(chunk, key)?.byte_len());
        Ok(())
    };
    let observations = match &chunk.record {
        Record::ProbeSet(p) => {
            add(&p.observations)?;
            for key in p
                .action_masks
                .iter()
                .chain(p.fixed_actions.iter())
                .chain(p.histories.values())
            {
                add(key)?;
            }
            &p.observations
        }
        Record::Trajectory(t) => {
            add(&t.observations)?;
            if let Some(key) = &t.action_masks {
                add(key)?;
            }
            &t.observations
        }
        _ => return Err(Error::State("Expected recorded input population".into())),
    };
    let rows = crate::plot_data::array(chunk, observations)?.shape()[0];
    ensure(
        rows <= MAX_POPULATION_ROWS,
        "Policy input population exceeds 8192-row bound",
    )?;
    ensure(
        bytes.saturating_add(rows.saturating_mul(std::mem::size_of::<Observation>()))
            <= MAX_POPULATION_BYTES,
        "Policy input population exceeds 4 MiB byte scan budget",
    )
}
fn ensure_recorded_feedforward(probes: &Chunk) -> Result<()> {
    let Record::ProbeSet(p) = &probes.record else {
        return Err(Error::State("Expected genuine probe input".into()));
    };
    ensure(
        p.histories.is_empty() && p.fixed_actions.is_none(),
        "Recorded history/fixed-action conditioning has no historical inference adapter",
    )
}
/// Imported f64 arrays are only representable by the existing inference input
/// type when conversion is exact. Never attach a rounded coordinate to a result.
pub(crate) fn exact_observation(chunk: &Chunk, row: usize) -> Result<Observation> {
    let (key, mask) = match &chunk.record {
        Record::ProbeSet(p) => (&p.observations, &p.action_masks),
        Record::Trajectory(t) => (&t.observations, &t.action_masks),
        _ => return Err(Error::State("Expected recorded input".into())),
    };
    let array = crate::plot_data::array(chunk, key)?;
    ensure(
        array.shape().len() == 2 && row < array.shape()[0],
        "Recorded observation row outside source",
    )?;
    let width = array.shape()[1];
    let range = row * width..(row + 1) * width;
    let values = match array.dtype() {
        rl::diagnostics::schema::DType::F32 => array.as_f32()?[range].to_vec(),
        rl::diagnostics::schema::DType::F64 => array.as_f64()?[range]
            .iter()
            .map(|&v| {
                let f = v as f32;
                ensure(
                    f.is_finite() && f64::from(f) == v,
                    "Recorded observation cannot be represented exactly as an f32 model input",
                )?;
                Ok(f)
            })
            .collect::<Result<Vec<_>>>()?,
        _ => {
            return Err(Error::Shape(
                "Recorded observations must be floating point".into(),
            ));
        }
    };
    let action_mask = mask
        .as_ref()
        .map(|key| -> Result<Vec<bool>> {
            let array = crate::plot_data::array(chunk, key)?;
            ensure(
                array.shape().len() == 2 && row < array.shape()[0],
                "Recorded mask row outside source",
            )?;
            let width = array.shape()[1];
            Ok(array.as_u8()?[row * width..(row + 1) * width]
                .iter()
                .map(|&v| v != 0)
                .collect())
        })
        .transpose()?;
    Ok(Observation::dense(values, action_mask))
}
/// Complete input from a checked policy/probe pair, including policy-only
/// masks. Use once on keyed selection to initialize SliceConfig/Recorded pin;
/// worker preparation uses the same path. Never rescan a population in paint.
pub fn recorded_input(batch: &PolicyBatch<'_>, row: usize) -> Result<Observation> {
    let probes = batch
        .recorded_probes()
        .ok_or_else(|| Error::State("Policy input requires checked probes".into()))?;
    let Record::ProbeSet(p) = &probes.record else {
        return Err(Error::State("Expected probe input".into()));
    };
    let dimension = crate::plot_data::array(probes, &p.observations)?.shape()[1];
    batch_capacity(dimension, action_width(batch.action_space()), false)?;
    let mut input = exact_observation(probes, row)?;
    input
        .as_dense_mut()
        .expect("dense diagnostic input")
        .action_mask = batch
        .effective_mask_with_budget(row, action_width(batch.action_space()).saturating_mul(2))?;
    Ok(input)
}
fn same_effective_mask(batch: &PolicyBatch<'_>, row: usize, mask: Option<&[bool]>) -> Result<bool> {
    Ok(batch
        .effective_mask_with_budget(row, action_width(batch.action_space()).saturating_mul(2))?
        .as_deref()
        == mask)
}
fn exact_policy_row(
    batch: &PolicyBatch<'_>,
    input: &Observation,
    explicit_controller: bool,
) -> Result<Option<usize>> {
    let probes = batch
        .recorded_probes()
        .ok_or_else(|| Error::State("Policy input requires checked probes".into()))?;
    let Record::ProbeSet(p) = &probes.record else {
        return Err(Error::State("Expected probe input".into()));
    };
    // A declared controller maps observations to actions. Fixed actions shared
    // with critic records do not condition its observation-to-action map.
    if !p.histories.is_empty() || !explicit_controller && p.fixed_actions.is_some() {
        return Ok(None);
    }
    let rows = probes.arrays[&p.observations].shape()[0];
    ensure(
        rows <= MAX_POPULATION_ROWS,
        "Reference population exceeds row bound",
    )?;
    for row in 0..rows {
        if recorded_input(batch, row)? == *input {
            return Ok(Some(row));
        }
    }
    Ok(None)
}

fn same_cross_section_input(a: &Observation, b: &Observation, varying_axis: usize) -> bool {
    let (a_values, b_values) = (a.dense_values(), b.dense_values());
    a_values.len() == b_values.len()
        && varying_axis < a_values.len()
        && a.dense_mask() == b.dense_mask()
        && a_values
            .iter()
            .zip(b_values)
            .enumerate()
            .all(|(index, (a, b))| index == varying_axis || a.to_bits() == b.to_bits())
}

#[derive(Default)]
struct CrossReferenceSamples {
    points: Vec<[f64; 2]>,
    omitted: usize,
    scan_limited: bool,
}

fn cross_section_reference_samples(
    source: &PolicySource,
    batch: &PolicyBatch<'_>,
    input: &Observation,
    varying_axis: usize,
    range: [f32; 2],
    action: usize,
    cancelled: &impl Fn() -> bool,
) -> Result<CrossReferenceSamples> {
    // Finite reference tables may exceed the ordinary population limit. Do not
    // turn a selected-point update into an unbounded scan of the entire table.
    if batch.rows() > MAX_POPULATION_ROWS {
        return Ok(CrossReferenceSamples {
            scan_limited: true,
            ..Default::default()
        });
    }
    let Some(probes) = batch.recorded_probes() else {
        return Ok(CrossReferenceSamples::default());
    };
    let Record::ProbeSet(probes) = &probes.record else {
        return Ok(CrossReferenceSamples::default());
    };
    if !probes.histories.is_empty()
        || (!source.identity.explicit_controller && probes.fixed_actions.is_some())
    {
        return Ok(CrossReferenceSamples::default());
    }
    let mut samples = Vec::new();
    let mut omitted = 0usize;
    for row in 0..batch.rows() {
        if cancelled() {
            return Err(Error::State("Policy cross-section superseded".into()));
        }
        let Ok(candidate) = recorded_input(batch, row) else {
            continue;
        };
        if !same_cross_section_input(input, &candidate, varying_axis) {
            continue;
        }
        let coordinate = candidate.dense_values()[varying_axis];
        if !coordinate.is_finite() || coordinate < range[0] || coordinate > range[1] {
            continue;
        }
        if samples.len() == MAX_CROSS_REFERENCE_SAMPLES {
            omitted += 1;
            continue;
        }
        let value = numerics::project(batch, row, action)?;
        if value.is_finite() {
            samples.push([f64::from(coordinate), value]);
        }
    }
    Ok(CrossReferenceSamples {
        points: samples,
        omitted,
        scan_limited: false,
    })
}

fn prepare_cross_section(
    request: &PolicyRequest,
    grid: &PreparedGrid,
    input: &Observation,
    reference: CrossReferenceSamples,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    inspect: &mut Inspect<'_>,
) -> Result<Option<Arc<PreparedCrossSection>>> {
    if !grid.regular() || request.slice.mode != SliceMode::TwoDimensional {
        return Ok(None);
    }
    let Some(y_axis) = request.slice.y else {
        return Ok(None);
    };
    let [nx, ny] = request.slice.resolution;
    if !(2..=256).contains(&nx)
        || !(2..=256).contains(&ny)
        || nx.checked_mul(ny) != Some(grid.values.len())
    {
        return Ok(None);
    }
    let x_axis = request.slice.x;
    let fixed_value = input.dense_values()[x_axis];
    let [xlo, xhi] = request.slice.ranges[x_axis];
    if !fixed_value.is_finite() || fixed_value < xlo || fixed_value > xhi {
        return Ok(None);
    }
    let [lo, hi] = request.slice.ranges[y_axis];
    let nearest = (((f64::from(fixed_value) - f64::from(xlo)) / (f64::from(xhi) - f64::from(xlo)))
        * (nx - 1) as f64)
        .round() as usize;
    let grid_input = request.slice.observation_at(nearest)?;
    let exact_grid_column = same_cross_section_input(input, &grid_input, y_axis);
    let mut points = Vec::with_capacity(ny);
    if exact_grid_column {
        for row in 0..ny {
            let value = grid.values[row * nx + nearest];
            let coordinate = (f64::from(lo)
                + (f64::from(hi) - f64::from(lo)) * row as f64 / (ny - 1) as f64)
                as f32;
            points.push([f64::from(coordinate), value]);
        }
    } else {
        let width = action_width(request.current.semantics().action_space());
        let capacity = batch_capacity(request.slice.fixed.len(), width, false)?.min(ny);
        for first in (0..ny).step_by(capacity) {
            if cancelled() {
                return Ok(None);
            }
            let count = (ny - first).min(capacity);
            accounting.scratch(
                FIXED_SCRATCH + count * (request.slice.fixed.len() * 64 + width * 256 + 1024),
            )?;
            let mut observations = Vec::with_capacity(count);
            for index in first..first + count {
                let mut observation = input.clone();
                observation
                    .as_dense_mut()
                    .expect("dense diagnostic input")
                    .values[y_axis] = (f64::from(lo)
                    + (f64::from(hi) - f64::from(lo)) * index as f64 / (ny - 1) as f64)
                    as f32;
                observations.push(observation);
            }
            let (result, _) = inspect(&request.current, &observations)?;
            accounting.point_batches.fetch_add(1, Ordering::AcqRel);
            let batch = checked_batch(
                &result.policy,
                &result.probes,
                request.current.semantics().action_space(),
                POLICY_SCRATCH,
            )?;
            for (row, observation) in observations.iter().enumerate() {
                if cancelled() {
                    return Ok(None);
                }
                points.push([
                    f64::from(observation.dense_values()[y_axis]),
                    numerics::project(&batch, row, request.slice.action)?,
                ]);
            }
        }
    }
    if points.iter().any(|p| !p[1].is_finite()) {
        return Ok(None);
    }
    Ok(Some(Arc::new(PreparedCrossSection {
        fixed_value,
        varying_axis: y_axis,
        points,
        reference_samples: reference.points,
        omitted_reference_samples: reference.omitted,
        reference_scan_limited: reference.scan_limited,
    })))
}

#[cfg(test)]
#[test]
fn explicit_policy_controller_matches_probes_shared_with_a_fixed_action_critic() {
    use rl::{
        core::action::Action,
        diagnostics::{
            probes::ProbeSet,
            schema::{Array, PolicyParameters, PolicyRecord, RecordContext},
        },
    };

    let space = ActionSpace::Unbounded { dimensions: 1 };
    let input = Observation::new(vec![0.5]);
    let probes = ProbeSet::new(
        "shared",
        ProbePopulation::FixedReference,
        "An observation and its separate critic query",
        vec![input.clone()],
        Some(vec![Action::Continuous(vec![0.75])]),
        space.clone(),
    )
    .unwrap()
    .record(RecordContext::new("policy-test", 0))
    .unwrap();
    let policy = Chunk {
        context: probes.context.clone(),
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            stochastic: false,
            masks: None,
            entropy: None,
            entropy_method: None,
        }),
        arrays: BTreeMap::from([(
            "actions".into(),
            Array::f32(vec![1, 1], vec![0.25]).unwrap(),
        )]),
    };
    let batch = checked_batch(&policy, &probes, &space, POLICY_SCRATCH).unwrap();
    assert_eq!(exact_policy_row(&batch, &input, false).unwrap(), None);
    assert_eq!(exact_policy_row(&batch, &input, true).unwrap(), Some(0));
    assert_eq!(
        exact_policy_row(&batch, &Observation::new(vec![0.75]), true).unwrap(),
        None
    );
}

fn prepare_point(
    owner: &Arc<RequestOwner>,
    grid: &PreparedGrid,
    previous: Option<&PreparedPoint>,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    inspect: &mut Inspect<'_>,
) -> Result<Option<Arc<PreparedPoint>>> {
    let request = &owner.request;
    request.current.check_available()?;
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    let recorded = request.input_frame();
    let recorded_batch = recorded
        .map(|frame| admit_recorded(frame, request.current.semantics().action_space()))
        .transpose()?;
    let (input, row, recorded_input) = match &request.pin {
        PolicyPin::Synthetic(input) => {
            ensure(
                request.slice.grid_eligibility() == GridEligibility::ContinuousFeedforward,
                "Arbitrary masked/history/discrete input is unsupported; select a real recorded row",
            )?;
            let row = recorded_batch
                .as_ref()
                .map(|batch| exact_policy_row(batch, input, false))
                .transpose()?
                .flatten();
            if recorded.is_some() {
                ensure(row.is_some(), "Exact pinned input not recorded")?;
            }
            (input.clone(), row, None)
        }
        PolicyPin::Recorded {
            key,
            probe_key,
            row,
        } => {
            let frame = recorded.ok_or_else(|| {
                Error::Config("Recorded pin requires its genuine current Policy source".into())
            })?;
            ensure(
                key == &frame.key && probe_key == &frame.probe_key,
                "Recorded pin source identity differs",
            )?;
            let Record::ProbeSet(p) = &frame.probes.record else {
                return Err(Error::State("Expected probe input".into()));
            };
            (
                recorded_input(recorded_batch.as_ref().unwrap(), *row)?,
                Some(*row),
                Some(SelectedInput::ProbeRow {
                    set: p.id.clone(),
                    row: *row,
                    sequence: probe_key.sequence,
                }),
            )
        }
    };
    let width = action_width(request.current.semantics().action_space());
    let bytes = width
        .checked_mul(18)
        .and_then(|b| {
            b.checked_add(
                input
                    .as_dense()
                    .expect("dense diagnostic input")
                    .values
                    .capacity()
                    * 4
                    + input
                        .as_dense()
                        .and_then(|dense| dense.action_mask.as_ref())
                        .map_or(0, Vec::capacity)
                    + 32 * 1024,
            )
        })
        .ok_or_else(|| Error::Config("Policy point size overflow".into()))?;
    let grid_source = grid
        .source
        .historical()
        .ok_or_else(|| Error::State("historical point request received a cached grid".into()))?;
    let latest_metadata = if Arc::ptr_eq(owner, grid_source) {
        0
    } else {
        owner.additional_bytes(grid_source)
    };
    ensure(
        bytes + grid._lease.bytes() + grid_source.request.metadata_bytes() + latest_metadata
            <= POLICY_FRAME,
        "Selected distributions plus grid exceed 2 MiB frame allowance",
    )?;
    let lease = Lease::acquire(accounting, bytes)?;
    accounting.scratch(
        FIXED_SCRATCH
            + (request.slice.fixed.len() * 64 + width * 256 + 1024)
                * if request.reference.is_some() { 2 } else { 1 },
    )?;
    let inferred = if !request.current.identity.recorded {
        Some(inspect(&request.current, std::slice::from_ref(&input))?.0)
    } else {
        None
    };
    let batch = if let Some(result) = &inferred {
        checked_batch(
            &result.policy,
            &result.probes,
            request.current.semantics().action_space(),
            POLICY_SCRATCH,
        )?
    } else {
        recorded_batch.unwrap()
    };
    let selected_row = if inferred.is_some() {
        0
    } else {
        row.unwrap_or(0)
    };
    let value = numerics::project(&batch, selected_row, request.slice.action)?;
    let distribution = numerics::selected_distribution(&batch, selected_row, request.slice.action)?;
    let chosen = |batch: &PolicyBatch<'_>, row: usize| -> Result<Option<usize>> {
        if matches!(batch.action_space(), ActionSpace::Discrete { .. }) {
            Ok(Some(batch.actions().as_u64()?[row] as usize))
        } else {
            Ok(None)
        }
    };
    let chosen_action = chosen(&batch, selected_row)?;
    let cross_axis = if grid.regular()
        && request.slice.mode == SliceMode::TwoDimensional
        && matches!(distribution, SelectedDistribution::Deterministic)
        && chosen_action.is_none()
    {
        request.slice.y.filter(|_| {
            let value = input.dense_values()[request.slice.x];
            let [lo, hi] = request.slice.ranges[request.slice.x];
            value.is_finite() && (lo..=hi).contains(&value)
        })
    } else {
        None
    };
    let reused_cross = cross_axis.and_then(|axis| {
        previous
            .filter(|old| same_cross_section_input(&old.input, &input, axis))
            .and_then(|old| old.cross_section.as_ref())
            .filter(|cross| cross.varying_axis == axis)
            .cloned()
    });
    let mut reference_samples = CrossReferenceSamples::default();
    let mut reference_chosen_action = None;
    let mut reference_value = None;
    let mut reference_distribution = None;
    let mut reference_recorded_input = None;
    let mut kl = None;
    if let Some(source) = &request.reference {
        source.prepare_reference(accounting, cancelled)?;
        let other_inferred = if source.model_source().is_some() {
            if let Some(frame) = recorded {
                ensure_recorded_feedforward(&frame.probes)?;
            }
            Some(inspect(source, std::slice::from_ref(&input))?.0)
        } else {
            None
        };
        let other_batch = if let Some(result) = &other_inferred {
            checked_batch(
                &result.policy,
                &result.probes,
                request.current.semantics().action_space(),
                POLICY_SCRATCH,
            )?
        } else {
            admit_reference(
                source,
                source.recorded_frame().unwrap(),
                request.current.semantics().action_space(),
            )?
        };
        let other_row = if let Some(other) = source.recorded_frame() {
            if let Some(frame) = recorded {
                source.matching_row(
                    &frame.probes,
                    row.unwrap_or(selected_row),
                    &other.probes,
                    &other_batch,
                    input.dense_mask(),
                )?
            } else if source.finite_reference() {
                source.reference_info.lookup(&other.probes, &input)?
            } else {
                exact_policy_row(&other_batch, &input, source.identity.explicit_controller)?
            }
        } else {
            Some(0)
        };
        if let Some(axis) = cross_axis
            && reused_cross.is_none()
            && source.recorded_frame().is_some()
        {
            reference_samples = cross_section_reference_samples(
                source,
                &other_batch,
                &input,
                axis,
                request.slice.ranges[axis],
                request.slice.action,
                cancelled,
            )
            .unwrap_or_default();
        }
        if let Some(other_row) = other_row {
            if let Some(frame) = source.recorded_frame()
                && let Record::ProbeSet(probes) = &frame.probes.record
            {
                reference_recorded_input = Some(SelectedInput::ProbeRow {
                    set: probes.id.clone(),
                    row: other_row,
                    sequence: frame.probe_key.sequence,
                });
            }
            let other = other_batch;
            reference_chosen_action = chosen(&other, other_row)?;
            reference_value = Some(numerics::project(&other, other_row, request.slice.action)?);
            reference_distribution = Some(numerics::selected_distribution(
                &other,
                other_row,
                request.slice.action,
            )?);
            // A different family can compare physical scalar action values,
            // but must not acquire a fabricated distribution divergence.
            if std::mem::discriminant(batch.parameters())
                == std::mem::discriminant(other.parameters())
            {
                kl = numerics::current_reference_kl(&batch, selected_row, &other, other_row)?;
            }
        }
    }
    accounting.point_batches.fetch_add(1, Ordering::AcqRel);
    let cross_section = if cross_axis.is_some() {
        if let Some(reused) = reused_cross {
            Some(reused)
        } else {
            prepare_cross_section(
                request,
                grid,
                &input,
                reference_samples,
                accounting,
                cancelled,
                inspect,
            )
            .ok()
            .flatten()
        }
    } else {
        None
    };
    let (plot, reference_plot) = distribution_plots(
        &distribution,
        reference_distribution.as_ref(),
        request.distribution_domain,
    )?;
    let statistics = |d: &SelectedDistribution| match d {
        SelectedDistribution::Gaussian(g) => g.action_statistics().ok(),
        _ => None,
    };
    let action_statistics = statistics(&distribution);
    let reference_statistics = reference_distribution.as_ref().and_then(statistics);
    let visits = grid.visits.as_ref().and_then(|v| {
        grid.config
            .visit_bin(&input)
            .ok()
            .flatten()
            .and_then(|b| v.counts.get(b).copied())
    });
    let distribution_bytes = |d: &SelectedDistribution| match d {
        SelectedDistribution::Categorical {
            probabilities,
            valid,
        } => probabilities.capacity() * 8 + valid.capacity(),
        _ => 0,
    };
    let actual = input
        .as_dense()
        .expect("dense diagnostic input")
        .values
        .capacity()
        * 4
        + input
            .as_dense()
            .and_then(|dense| dense.action_mask.as_ref())
            .map_or(0, Vec::capacity)
        + distribution_bytes(&distribution)
        + reference_distribution
            .as_ref()
            .map_or(0, distribution_bytes)
        + plot.as_ref().map_or(0, |p| p.samples.capacity() * 16)
        + reference_plot
            .as_ref()
            .map_or(0, |p| p.samples.capacity() * 16)
        + cross_section.as_ref().map_or(0, |s| {
            s.points.capacity() * 16 + s.reference_samples.capacity() * 16
        })
        + 4096;
    ensure(
        actual <= bytes,
        "Policy selected-point capacities exceed admitted reservation",
    )?;
    request.current.check_available()?;
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    Ok(Some(Arc::new(PreparedPoint {
        cached_row: None,
        chosen_action,
        reference_chosen_action,
        action_statistics,
        reference_statistics,
        input,
        recorded_input,
        reference_recorded_input,
        value,
        reference_value,
        difference: reference_value.map(|r| value - r),
        kl,
        distribution,
        reference_distribution,
        plot,
        reference_plot,
        visits,
        cross_section,
        action_value: None,
        _lease: FrameLease::Historical(lease),
    })))
}
fn prepare_visits(
    request: &PolicyRequest,
    accounting: &Accounting,
    cancelled: &impl Fn() -> bool,
) -> Result<Option<PreparedVisitation>> {
    let Some(evidence) = &request.visits else {
        return Ok(None);
    };
    admit_inputs(&evidence.chunk)?;
    let (source, rows) = match &evidence.chunk.record {
        Record::ProbeSet(p) if p.population == ProbePopulation::PolicyVisited => (
            VisitationSource::PolicyVisited,
            crate::plot_data::array(&evidence.chunk, &p.observations)?.shape()[0],
        ),
        Record::Trajectory(t) => {
            // Counts describe policy decisions: T real inputs with actions,
            // excluding the final observation after termination or truncation.
            let decisions = crate::plot_data::array(&evidence.chunk, &t.actions)?.shape()[0];
            ensure(
                crate::plot_data::array(&evidence.chunk, &t.observations)?.shape()[0]
                    == decisions + 1,
                "Trajectory visitation needs T actions and T+1 observations",
            )?;
            (VisitationSource::EvaluationTrajectory, decisions)
        }
        _ => {
            return Err(Error::Config(
                "Visits require actual evaluation-trajectory or policy-visited evidence".into(),
            ));
        }
    };
    ensure(
        evidence.key.root == request.current.identity.key.root,
        "Visitation belongs to another run",
    )?;
    ensure(
        rows <= MAX_POPULATION_ROWS,
        "Visitation population exceeds 8192-row bound",
    )?;
    let mut counts = vec![0; request.slice.len()?];
    let mut matched = Vec::with_capacity(rows);
    for row in 0..rows {
        if cancelled() {
            return Ok(None);
        }
        let input = exact_observation(&evidence.chunk, row)?;
        input.validate(
            request.slice.fixed.len(),
            request.current.semantics().action_space(),
            true,
        )?;
        if let Some(bin) = request.slice.visit_bin(&input)? {
            counts[bin] += 1;
            matched.push(row);
        }
    }
    accounting
        .source_rows
        .fetch_add(rows as u64, Ordering::AcqRel);
    let n = matched.len().min(MAX_VISIT_MARKERS);
    let mut markers = Vec::with_capacity(n);
    for i in 0..n {
        let input = exact_observation(&evidence.chunk, matched[i * matched.len() / n])?;
        markers.push([
            input.dense_values()[request.slice.x],
            request.slice.y.map_or(0., |y| input.dense_values()[y]),
        ]);
    }
    Ok(Some(PreparedVisitation {
        source,
        revision: evidence.chunk.context.identity.learner_update_revision,
        tolerance: request.slice.tolerance,
        filtered_count: matched.len(),
        counts,
        markers,
    }))
}

#[cfg(test)]
mod view_reservation_tests {
    use super::*;
    #[test]
    fn recorded_reference_work_limit_handles_boundary_and_overflow() {
        assert!(recorded_match_fits(1024, 1024));
        assert!(!recorded_match_fits(1025, 1024));
        assert!(!recorded_match_fits(usize::MAX, 2));
    }

    #[test]
    fn view_admission_counts_old_and_new_owners_and_repeated_edits_drop_to_zero() {
        let accounting = Arc::new(Accounting::default());
        let old = PolicyViewLease::new(&accounting, POLICY_PREPARED - 1024).unwrap();
        assert!(PolicyViewLease::new(&accounting, 2048).is_err());
        assert_eq!(accounting.snapshot().prepared_bytes, POLICY_PREPARED - 1024);
        let replacement = PolicyViewLease::new(&accounting, 1024).unwrap();
        assert_eq!(accounting.snapshot().prepared_bytes, POLICY_PREPARED);
        drop(old);
        for _ in 0..100 {
            let edit = PolicyViewLease::new(&accounting, 512).unwrap();
            assert!(accounting.snapshot().prepared_bytes <= POLICY_PREPARED);
            drop(edit);
        }
        drop(replacement);
        assert_eq!(accounting.snapshot().prepared_bytes, 0);
        assert_eq!(accounting.snapshot().prepared_peak, POLICY_PREPARED);
    }
}
