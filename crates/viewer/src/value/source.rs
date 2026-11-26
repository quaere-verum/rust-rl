use super::ensure;
use crate::{data::RunIndex, policy::prepared::SourceIdentity, resolver::ResolvedRecord};
use rl::{
    Error, Result,
    diagnostics::{
        models::{ModelKind, ModelLimits, SnapshotConfiguration, snapshot_configuration},
        schema::*,
        storage::ChunkSource,
        value::*,
        visualization::EnvironmentSemantics,
    },
};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum Data {
    Model(Arc<ChunkSource>),
    Recorded(Vec<ResolvedRecord>),
}
#[derive(Clone)]
pub struct ValueSource {
    pub(crate) identity: SourceIdentity,
    pub(crate) data: Data,
    pub(crate) manifest: Arc<RunManifest>,
    pub(crate) declaration: Option<Arc<ValueDeclaration>>,
    pub(crate) return_target: Option<ReturnTarget>,
    pub(crate) reference_info: Arc<crate::reference::ReferenceInfo>,
    pub(crate) names: Vec<String>,
    pub(crate) kind: ValueCriticKind,
    pub(crate) heads: usize,
    pub(crate) actor: Option<String>,
    pub(crate) actor_stochastic: bool,
    pub(crate) role: String,
    pub(crate) metadata_bytes: usize,
    pub(crate) checkpoint_id: Option<String>,
}

/// Allocation-local view estimate. Recorded sources share their decoded
/// manifest/declarations across search, request and color handles. The handles'
/// strings and frame vectors remain private. Distinct or partially shared Arc
/// bundles retain the full conservative charge; saturation never undercounts.
#[derive(Default)]
pub(crate) struct ViewSourceCharges {
    seen: [[usize; 3]; 32],
    len: usize,
}
impl ViewSourceCharges {
    pub(crate) fn charge(&mut self, source: &ValueSource) -> usize {
        let Data::Recorded(frames) = &source.data else {
            return source.metadata_bytes;
        };
        let private = std::mem::size_of::<ValueSource>()
            + source.identity.name.capacity()
            + source.names.capacity() * std::mem::size_of::<String>()
            + source.names.iter().map(String::capacity).sum::<usize>()
            + source.actor.as_ref().map_or(0, String::capacity)
            + source.role.capacity()
            + source.checkpoint_id.as_ref().map_or(0, String::capacity)
            + frames.capacity() * std::mem::size_of::<ResolvedRecord>();
        let key = [
            Arc::as_ptr(&source.manifest) as usize,
            source
                .declaration
                .as_ref()
                .map_or(0, |d| Arc::as_ptr(d) as usize),
            Arc::as_ptr(&source.reference_info) as usize,
        ];
        if self.seen[..self.len].contains(&key) {
            return private;
        }
        if self.len < self.seen.len() {
            self.seen[self.len] = key;
            self.len += 1;
        }
        source.metadata_bytes.max(private)
    }
}
impl ValueSource {
    /// Per-handle allocations remain charged even when the immutable source
    /// metadata and model address are shared by cloned handles.
    pub(crate) fn private_metadata_bytes(&self) -> usize {
        std::mem::size_of::<ValueSource>()
            + self.identity.name.capacity()
            + self.names.capacity() * std::mem::size_of::<String>()
            + self.names.iter().map(String::capacity).sum::<usize>()
            + self.actor.as_ref().map_or(0, String::capacity)
            + self.role.capacity()
            + self.checkpoint_id.as_ref().map_or(0, String::capacity)
            + match &self.data {
                Data::Model(_) => 0,
                Data::Recorded(frames) => frames.capacity() * std::mem::size_of::<ResolvedRecord>(),
            }
    }
    /// Undeclared records have no trustworthy model name. Keep each immutable
    /// observation distinct rather than inventing continuity between critics.
    pub(crate) fn recorded_name(chunk: &Chunk, sequence: u64) -> Result<String> {
        if let Some(d) = declaration(chunk)? {
            ensure(
                d.critics.len() == 1,
                "choose an unambiguous recorded critic declaration",
            )?;
            Ok(d.critics.keys().next().unwrap().clone())
        } else {
            Ok(format!("Recorded critic #{sequence}"))
        }
    }
    pub(crate) fn recorded_kind(c: &CriticRecord) -> ValueCriticKind {
        match &c.representation {
            CriticRepresentation::Scalar { .. }
                if c.action_condition == ActionCondition::StateValue =>
            {
                ValueCriticKind::StateValue
            }
            CriticRepresentation::Scalar { .. } => ValueCriticKind::ScalarQ,
            CriticRepresentation::Discrete { .. } => ValueCriticKind::DiscreteQ,
            CriticRepresentation::Quantiles { .. } => ValueCriticKind::QuantileQ,
            CriticRepresentation::RawMoments { .. } => ValueCriticKind::MomentQ,
            CriticRepresentation::CategoricalRegression { .. } => ValueCriticKind::CategoricalQ,
            CriticRepresentation::CategoricalReturn { .. } => ValueCriticKind::CategoricalReturnQ,
        }
    }
    pub(crate) fn configuration(chunk: &Chunk) -> Result<SnapshotConfiguration> {
        let Record::ModelSnapshot(s) = &chunk.record else {
            return Err(Error::Config("expected Value snapshot".into()));
        };
        snapshot_configuration(
            s,
            ModelLimits {
                max_model_bytes: 2 * 1024 * 1024,
                ..ModelLimits::default()
            },
        )
    }
    /// Actor declarations carry opaque provenance, not exported model-map keys.
    /// Automatic conditioning requires a sole actual actor; explicit named
    /// selection remains available through `model_with_actor`.
    pub(crate) fn model_for_conditioning(
        run: &RunIndex,
        index: usize,
        name: &str,
        chunk: &Chunk,
        actor_required: bool,
    ) -> Result<Self> {
        let source = Self::model(run, index, name, chunk)?;
        if actor_required
            && !matches!(
                source.kind,
                ValueCriticKind::StateValue | ValueCriticKind::DiscreteQ
            )
        {
            ensure(source.actor.is_some(), "required unambiguous actor absent")?;
        }
        Ok(source)
    }
    pub fn model_with_actor(
        run: &RunIndex,
        index: usize,
        name: &str,
        actor: &str,
        chunk: &Chunk,
    ) -> Result<Self> {
        let mut source = Self::model(run, index, name, chunk)?;
        ensure(actor.len() <= 1024, "actor name exceeds admission")?;
        let config = Self::configuration(chunk)?;
        let model = config
            .models
            .get(actor)
            .ok_or_else(|| Error::Config("named actor absent".into()))?;
        ensure(
            matches!(
                model.kind,
                ModelKind::DeterministicPolicy
                    | ModelKind::StochasticPolicy
                    | ModelKind::StateGaussianPolicy
            ),
            "selected actor is not a policy model",
        )?;
        source.actor = Some(actor.into());
        source.actor_stochastic = model.kind.is_stochastic_policy();
        Ok(source)
    }
    /// Metadata comes from the existing asynchronous Loader. Weight bytes are
    /// never retained here; the historical worker restores through ChunkSource.
    pub fn model(run: &RunIndex, index: usize, name: &str, chunk: &Chunk) -> Result<Self> {
        let entry = run
            .recording()
            .entries()
            .get(index)
            .ok_or_else(|| Error::State("missing Value source".into()))?;
        ensure(
            entry.kind == RecordKind::ModelSnapshot && run.has_model_record(index),
            "Value snapshot unavailable",
        )?;
        ensure(
            chunk.context.identity.run_id == run.recording().manifest().run_id
                && chunk.context.identity.learner_update_revision == entry.revision
                && chunk.context.identity.model_role == entry.model_role
                && chunk.context.identity.checkpoint_id == entry.checkpoint_id,
            "Value source context differs from immutable index",
        )?;
        let config = Self::configuration(chunk)?;
        ensure(
            config.observation_dim == run.recording().manifest().observation_dim
                && config.action_space == run.recording().manifest().action_space,
            "Value snapshot encoding differs from actual manifest",
        )?;
        let model = config
            .models
            .get(name)
            .ok_or_else(|| Error::Config("named critic absent".into()))?;
        let kind = kind(&model.kind)
            .ok_or_else(|| Error::Config("model is not a supported critic".into()))?;
        let heads = match &model.kind {
            ModelKind::CategoricalQ { support, .. } => support.bins,
            _ => model
                .network
                .as_ref()
                .and_then(|n| n.widths.last())
                .copied()
                .unwrap_or(1),
        };
        ensure(
            heads <= MAX_VALUE_HEADS,
            "critic exceeds complete head admission",
        )?;
        let actors: Vec<_> = config
            .models
            .iter()
            .filter(|(_, m)| {
                matches!(
                    m.kind,
                    ModelKind::DeterministicPolicy
                        | ModelKind::StochasticPolicy
                        | ModelKind::StateGaussianPolicy
                )
            })
            .collect();
        let (actor, actor_stochastic) = if actors.len() == 1 {
            ensure(actors[0].0.len() <= 1024, "actor name exceeds admission")?;
            (
                Some(actors[0].0.clone()),
                actors[0].1.kind.is_stochastic_policy(),
            )
        } else {
            (None, false)
        };
        Self::build(
            run,
            SourceIdentity {
                key: run.key(index)?,
                probe_key: None,
                revision: entry.revision,
                name: name.into(),
                recorded: false,
                explicit_controller: false,
            },
            Data::Model(Arc::new(run.recording().source(index)?)),
            chunk,
            kind,
            heads,
            actor,
            actor_stochastic,
        )
    }
    pub fn recorded(run: &RunIndex, frame: &ResolvedRecord, name: &str) -> Result<Self> {
        ensure(
            frame.key.root == *run.root() && frame.probe_key.root == *run.root(),
            "recorded Value belongs to another run",
        )?;
        let index = run
            .by_sequence(frame.key.sequence)
            .ok_or_else(|| Error::Config("missing critic identity".into()))?;
        ensure(
            run.probe_for_record(index)
                .map(|i| run.key(i))
                .transpose()?
                .as_ref()
                == Some(&frame.probe_key),
            "critic must retain its original immutable probe definition",
        )?;
        let Record::Critic(c) = &frame.chunk.record else {
            return Err(Error::Config("expected recorded critic".into()));
        };
        let Record::ProbeSet(p) = &frame.probes.record else {
            return Err(Error::Config("critic has no genuine probe inputs".into()));
        };
        let a = crate::plot_data::array(&frame.probes, &p.observations)?;
        ensure(
            a.shape().len() == 2 && a.shape()[1] == run.recording().manifest().observation_dim,
            "probe encoding differs from actual manifest",
        )?;
        let kind = Self::recorded_kind(c);
        let heads = match &c.representation {
            CriticRepresentation::Quantiles { values, .. }
            | CriticRepresentation::Discrete { values }
            | CriticRepresentation::CategoricalReturn {
                probabilities: values,
                ..
            }
            | CriticRepresentation::CategoricalRegression { logits: values, .. } => {
                crate::plot_data::array(&frame.chunk, values)?
                    .shape()
                    .get(1)
                    .copied()
                    .unwrap_or(0)
            }
            _ => 1,
        };
        ensure(
            heads > 0 && heads <= MAX_VALUE_HEADS,
            "recorded critic head admission",
        )?;
        Self::build(
            run,
            SourceIdentity {
                key: frame.key.clone(),
                probe_key: Some(frame.probe_key.clone()),
                revision: frame.revision(),
                name: name.into(),
                recorded: true,
                explicit_controller: false,
            },
            Data::Recorded(vec![frame.clone()]),
            &frame.chunk,
            kind,
            heads,
            None,
            false,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn build(
        run: &RunIndex,
        identity: SourceIdentity,
        data: Data,
        chunk: &Chunk,
        kind: ValueCriticKind,
        heads: usize,
        actor: Option<String>,
        actor_stochastic: bool,
    ) -> Result<Self> {
        ensure(
            !identity.name.is_empty()
                && identity.name.len() <= 1024
                && chunk.context.identity.model_role.len() <= 1024
                && chunk
                    .context
                    .identity
                    .checkpoint_id
                    .as_ref()
                    .is_none_or(|id| id.len() <= 1024),
            "invalid critic name",
        )?;
        let declaration = declaration(chunk)?;
        if let Some(d) = &declaration {
            d.validate_manifest(run.recording().manifest())?;
            ensure(
                d.critics
                    .get(&identity.name)
                    .is_none_or(|c| c.kind == kind && c.learned_baseline != Some(false))
                    && (!matches!(data, Data::Recorded(_))
                        || d.critics.contains_key(&identity.name)),
                "critic declaration missing, disabled or mismatched",
            )?;
        }
        // The source manifest is already schema-admitted by Recording. Retained
        // strings/descriptor/declaration are conservatively charged before clone.
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(b.len());
                if self.0 > 48 * 1024 {
                    return Err(std::io::Error::other("Value manifest metadata admission"));
                }
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut counter = Count(0);
        serde_json::to_writer(&mut counter, run.recording().manifest())?;
        let encoded = counter.0;
        let reference_info =
            crate::reference::ReferenceInfo::new(run.recording().manifest(), Some(chunk))?;
        let metadata_bytes = encoded
            .saturating_mul(8)
            .saturating_add(128 * 1024)
            .saturating_add(4096)
            .saturating_add(reference_info.bytes());
        ensure(
            metadata_bytes <= 512 * 1024,
            "Value source metadata exceeds admission",
        )?;
        Ok(Self {
            names: vec![identity.name.clone()],
            identity,
            data,
            manifest: Arc::new(run.recording().manifest().clone()),
            declaration: declaration.map(Arc::new),
            return_target: return_target(chunk)?,
            reference_info,
            kind,
            heads,
            actor,
            actor_stochastic,
            role: chunk.context.identity.model_role.clone(),
            metadata_bytes,
            checkpoint_id: chunk.context.identity.checkpoint_id.clone(),
        })
    }
    pub fn with_twin(mut self, group: &str) -> Result<Self> {
        let d = self
            .declaration
            .as_ref()
            .ok_or_else(|| Error::Config("unknown twin declaration".into()))?;
        ensure(
            matches!(self.data, Data::Model(_)),
            "recorded twins require explicit paired sources",
        )?;
        self.names = twin_names(d, group)?;
        ensure(
            self.names.contains(&self.identity.name),
            "selected critic is outside twin group",
        )?;
        Ok(self)
    }
    pub fn recorded_twin(mut self, other: Self, group: &str) -> Result<Self> {
        let (Data::Recorded(a), Data::Recorded(b)) = (&mut self.data, other.data) else {
            return Err(Error::Config("recorded twin needs probe sources".into()));
        };
        ensure(
            a.len() == 1
                && b.len() == 1
                && a[0].probe_key == b[0].probe_key
                && a[0].chunk.context.identity == b[0].chunk.context.identity,
            "twins require same immutable probe/context",
        )?;
        let da = self
            .declaration
            .as_ref()
            .ok_or_else(|| Error::Config("unknown recorded twin".into()))?;
        let db = other
            .declaration
            .as_ref()
            .ok_or_else(|| Error::Config("unknown recorded twin".into()))?;
        let ta = da.critics[&self.identity.name]
            .twin
            .as_ref()
            .ok_or_else(|| Error::Config("not twin".into()))?;
        let tb = db.critics[&other.identity.name]
            .twin
            .as_ref()
            .ok_or_else(|| Error::Config("not twin".into()))?;
        ensure(
            ta.group == group
                && tb.group == group
                && ta.index != tb.index
                && da.conventions == db.conventions,
            "incompatible recorded twins",
        )?;
        self.names.push(other.identity.name);
        a.extend(b);
        if ta.index == 1 {
            a.swap(0, 1);
            self.names.swap(0, 1);
        }
        let mut combined = (**da).clone();
        combined.critics.extend(db.critics.clone());
        combined.validate()?;
        self.declaration = Some(Arc::new(combined));
        self.metadata_bytes += other.metadata_bytes;
        // The combined metadata has a distinct shared-ownership charge.
        self.manifest = Arc::new((*self.manifest).clone());
        Ok(self)
    }
    pub fn identity(&self) -> &SourceIdentity {
        &self.identity
    }
    pub(crate) fn finite_reference(&self) -> bool {
        self.reference_info.finite()
    }
    pub(crate) fn prepare_reference(
        &self,
        accounting: &Arc<crate::policy::prepared::Accounting>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        if let Data::Recorded(frames) = &self.data {
            self.reference_info.prepare(
                &frames[0].chunk,
                &frames[0].probes,
                accounting,
                cancelled,
            )?;
        }
        Ok(())
    }
    pub(crate) fn reference_quality_ui(&self, ui: &mut eframe::egui::Ui) {
        self.reference_info
            .ui(ui, &self.identity.name, self.identity.recorded, true);
    }
    pub fn manifest(&self) -> &RunManifest {
        &self.manifest
    }
    pub fn semantics(&self) -> EnvironmentSemantics<'_> {
        EnvironmentSemantics::new(&self.manifest)
    }
    pub fn kind(&self) -> ValueCriticKind {
        self.kind
    }
    pub fn critic_names(&self) -> &[String] {
        &self.names
    }
    pub fn model_role(&self) -> &str {
        &self.role
    }
    pub fn actor_name(&self) -> Option<&str> {
        self.actor.as_deref()
    }
    pub fn actor_stochastic(&self) -> bool {
        self.actor_stochastic
    }
    pub fn declaration(&self) -> Option<&ValueDeclaration> {
        self.declaration.as_deref()
    }
    /// Exact immutable evidence for a sparse-series row; models have no recorded rows.
    pub fn recorded_input(
        &self,
        row: usize,
    ) -> Result<(rl::core::env::Observation, Option<rl::core::action::Action>)> {
        let Data::Recorded(frames) = &self.data else {
            return Err(Error::State("model source has no recorded input".into()));
        };
        Ok((
            crate::policy::prepared::exact_observation(&frames[0].probes, row)?,
            super::visitation::critic_action(&frames[0].chunk, row, &self.manifest.action_space)?,
        ))
    }
    /// Only admission/matching here: no critic evaluation or reconstructed rows.
    pub(crate) fn has_recorded_sample(&self, slice: &super::slice::ValueSlice) -> Result<bool> {
        use super::slice::{ActionConditioning, ValueCoordinate};
        use rl::core::action::Action;
        let Data::Recorded(frames) = &self.data else {
            return Ok(false);
        };
        let rows = super::visitation::input_rows(&frames[0].probes)?;
        ensure(
            rows <= 8192 && rows.saturating_mul(self.manifest.observation_dim) <= 1024 * 1024,
            "recorded sample matching exceeds admission",
        )?;
        if slice.mode == crate::policy::slice::SliceMode::SamplesOnly {
            return Ok(rows > 0);
        }
        for row in 0..rows {
            let (input, mut action) = self.recorded_input(row)?;
            if self.kind == ValueCriticKind::DiscreteQ {
                let category_axis = std::iter::once(slice.x)
                    .chain(slice.y)
                    .any(|a| a == ValueCoordinate::DiscreteAction);
                let index = match &slice.conditioning {
                    ActionConditioning::Fixed(Action::Discrete(i)) if !category_axis => Some(*i),
                    _ => (0..self.heads).find(|&i| input.dense_mask().is_none_or(|m| m[i])),
                };
                action = index.map(Action::Discrete);
            }
            if super::visitation::position_in_slice(slice, &input, action.as_ref(), None)?.is_some()
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(crate) fn check_available(&self) -> Result<()> {
        if let Data::Model(s) = &self.data {
            s.check_available()?;
        }
        Ok(())
    }
    pub(crate) fn same_payload(&self, other: &Self) -> bool {
        match (&self.data, &other.data) {
            (Data::Model(_), Data::Model(_)) => self.identity == other.identity,
            (Data::Recorded(a), Data::Recorded(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(a, b)| {
                        std::ptr::eq(a.chunk.as_ref(), b.chunk.as_ref())
                            && std::ptr::eq(a.probes.as_ref(), b.probes.as_ref())
                    })
            }
            _ => false,
        }
    }
    pub(crate) fn same(&self, b: &Self) -> bool {
        self.identity == b.identity
            && self.reference_info.same(&b.reference_info)
            && match (&self.data, &b.data) {
                (Data::Model(_), Data::Model(_)) => true,
                (Data::Recorded(a), Data::Recorded(b)) => {
                    a.len() == b.len()
                        && a.iter()
                            .zip(b)
                            .all(|(a, b)| a.key == b.key && a.probe_key == b.probe_key)
                }
                _ => false,
            }
            && self.names == b.names
            && self.actor == b.actor
            && self.declaration == b.declaration
            && self.return_target == b.return_target
            && self.manifest.discount == b.manifest.discount
            && self.manifest.reward_convention == b.manifest.reward_convention
            && self.manifest.environment_descriptor == b.manifest.environment_descriptor
    }
    pub(crate) fn utility(&self) -> Option<&MomentUtility> {
        self.declaration
            .as_ref()?
            .critics
            .get(&self.names[0])?
            .utility
            .as_ref()
    }
}
pub(crate) fn kind(kind: &ModelKind) -> Option<ValueCriticKind> {
    kind.critic_kind()
}
fn twin_names(d: &ValueDeclaration, group: &str) -> Result<Vec<String>> {
    let mut names = [None, None];
    for (name, c) in &d.critics {
        if let Some(t) = &c.twin
            && t.group == group
        {
            ensure(
                t.index < 2 && names[t.index].is_none(),
                "invalid twin members",
            )?;
            names[t.index] = Some(name.clone());
        }
    }
    names
        .into_iter()
        .map(|n| n.ok_or_else(|| Error::Config("incomplete twin group".into())))
        .collect()
}
