//! Versioned, backend-independent diagnostic records.
pub use crate::core::recording::ProbeIdentity;
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        ensure,
        env::ObservationSpace,
    },
    train::counters::RunCounters,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 5;
pub type ArrayKey = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub experiment_id: String,
    pub algorithm: String,
    pub model_id: String,
    /// Learner revision before this run's first update (e.g. learner-only restore).
    #[serde(default)]
    pub starting_revision: u64,
    pub observation_dim: usize,
    /// The actual model input space for runs without a dense observation width.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_observation_space: Option<ObservationSpace>,
    pub observation_labels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment_descriptor: Option<super::visualization::EnvironmentDescriptor>,
    pub action_space: ActionSpace,
    pub reward_convention: String,
    pub discount: f64,
    pub environment: String,
    pub metadata: BTreeMap<String, String>,
}
impl RunManifest {
    pub fn new(
        run: &str,
        experiment: &str,
        algorithm: &str,
        observation_dim: usize,
        action_space: ActionSpace,
        discount: f64,
    ) -> Result<Self> {
        Self::new_in(
            run,
            experiment,
            algorithm,
            ObservationSpace::Dense(observation_dim),
            action_space,
            discount,
        )
    }

    pub fn new_in(
        run: &str,
        experiment: &str,
        algorithm: &str,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        discount: f64,
    ) -> Result<Self> {
        let width = observation_space.dense_width();
        let value = Self {
            schema_version: SCHEMA_VERSION,
            run_id: run.into(),
            experiment_id: experiment.into(),
            algorithm: algorithm.into(),
            model_id: algorithm.into(),
            starting_revision: 0,
            observation_dim: width.unwrap_or(0),
            typed_observation_space: width.is_none().then_some(observation_space),
            observation_labels: vec![],
            environment_descriptor: None,
            action_space,
            reward_convention: "reward, larger is better; original environment units".into(),
            discount,
            environment: "custom".into(),
            metadata: BTreeMap::new(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn observation_space(&self) -> ObservationSpace {
        self.typed_observation_space
            .clone()
            .unwrap_or(ObservationSpace::Dense(self.observation_dim))
    }

    pub fn validate(&self) -> Result<()> {
        ensure(
            (1..=SCHEMA_VERSION).contains(&self.schema_version),
            "unsupported diagnostic schema version",
        )?;
        ensure(
            !self.run_id.is_empty()
                && !self.experiment_id.is_empty()
                && !self.algorithm.is_empty()
                && !self.model_id.is_empty(),
            "invalid diagnostic run identity",
        )?;
        if let Some(space) = &self.typed_observation_space {
            ensure(
                self.schema_version >= 5
                    && self.observation_dim == 0
                    && space.dense_width().is_none(),
                "invalid typed diagnostic observation space",
            )?;
        }
        self.observation_space().validate()?;
        ensure(
            self.discount.is_finite() && (0. ..=1.).contains(&self.discount),
            "invalid diagnostic discount",
        )?;
        ensure(
            self.observation_labels.is_empty()
                || self.observation_labels.len() == self.observation_dim,
            "observation labels do not match dimensions",
        )?;
        self.action_space.validate()?;
        if let Some(descriptor) = &self.environment_descriptor {
            descriptor.validate(self.observation_dim, &self.action_space)?;
        }
        Ok(())
    }
}

/// All counters refer to the same completed learner boundary. A probe-only
/// timestamp need not have preserved model weights (`checkpoint_id = None`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordContext {
    pub identity: ProbeIdentity,
    pub counters: RunCounters,
    pub learner_call: u64,
    pub collection_cycle: u64,
    pub probe_set: Option<String>,
    pub trajectory: Option<String>,
    pub environment_step: Option<u64>,
    pub simulation_time: Option<f64>,
    /// Monotonic seconds since the recording session began, measured at capture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_elapsed_seconds: Option<f64>,
}
impl RecordContext {
    pub fn new(run: &str, revision: u64) -> Self {
        Self {
            identity: ProbeIdentity {
                run_id: run.into(),
                learner_update_revision: revision,
                model_role: "online".into(),
                checkpoint_id: None,
            },
            counters: RunCounters {
                updates: revision,
                ..Default::default()
            },
            learner_call: 0,
            collection_cycle: 0,
            probe_set: None,
            trajectory: None,
            environment_step: None,
            simulation_time: None,
            wall_elapsed_seconds: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    Metrics,
    Event,
    ProbeSet,
    Policy,
    Critic,
    Activations,
    Empirical,
    Trajectory,
    ModelSnapshot,
    TypedProbes,
    FinitePredictions,
    TypedEpisode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRecord {
    /// Missing metrics are explicit and never replaced with estimates.
    pub values: BTreeMap<String, f64>,
    pub unavailable: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub category: String,
    pub message: String,
    pub severity: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbePopulation {
    FixedReference,
    UserSupplied,
    PolicyVisited,
    ImportedHistory,
    ReplayRetained,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeSetRecord {
    pub id: String,
    pub population: ProbePopulation,
    pub methodology: String,
    pub observations: ArrayKey,
    pub action_masks: Option<ArrayKey>,
    pub fixed_actions: Option<ArrayKey>,
    /// Stable row identifiers, encoded as u64. Never match different runs by row
    /// index alone without checking inputs and methodology.
    pub row_ids: ArrayKey,
    pub histories: BTreeMap<String, ArrayKey>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionCondition {
    Fixed,
    CurrentPolicy,
    StateValue,
    AllDiscreteActions,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyParameters {
    Categorical {
        probabilities: ArrayKey,
    },
    Gaussian {
        mean: ArrayKey,
        log_std: ArrayKey,
        tanh_bounded: bool,
    },
    Deterministic,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRecord {
    pub actions: ArrayKey,
    pub parameters: PolicyParameters,
    pub entropy: Option<ArrayKey>,
    /// Exact categorical/Gaussian entropy or the bounded-policy MC estimator.
    #[serde(default)]
    pub entropy_method: Option<String>,
    pub masks: Option<ArrayKey>,
    pub stochastic: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CriticRepresentation {
    Scalar {
        values: ArrayKey,
    },
    Discrete {
        values: ArrayKey,
    },
    Quantiles {
        values: ArrayKey,
        locations: ArrayKey,
        masses: ArrayKey,
    },
    RawMoments {
        mean: ArrayKey,
        raw_second: ArrayKey,
    },
    CategoricalRegression {
        logits: ArrayKey,
        weights: ArrayKey,
        centers: ArrayKey,
        values: ArrayKey,
    },
    CategoricalReturn {
        atoms: ArrayKey,
        probabilities: ArrayKey,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticRecord {
    pub representation: CriticRepresentation,
    pub action_condition: ActionCondition,
    pub actions: Option<ArrayKey>,
    /// Explicit intended continuation policy, required for conditional scoring.
    #[serde(default)]
    pub continuation_policy: Option<String>,
    #[serde(default)]
    pub continuation_stochastic: Option<bool>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationSummary {
    pub count: u64,
    pub finite_count: u64,
    pub mean: Option<f64>,
    pub standard_deviation: Option<f64>,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub zero_fraction: f64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActivationRecord {
    /// Complete model-input contract. Absent legacy/custom metadata is unknown.
    #[serde(default)]
    pub inputs: ActivationInputs,
    pub network: String,
    /// Stable pre/post layer names. Values have shape [states, neurons].
    pub layers: BTreeMap<String, ArrayKey>,
    pub summaries: BTreeMap<String, ActivationSummary>,
    /// Optional u64 neuron indices for compact selected-neuron matrices.
    #[serde(default)]
    pub selected_neurons: BTreeMap<String, ArrayKey>,
    #[serde(default)]
    pub source_widths: BTreeMap<String, usize>,
}
/// Provenance declaration, not authentication of an imported computation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ActivationInputs {
    #[default]
    Unknown,
    Observations,
    ObservationsAndActions {
        actions: ArrayKey,
        source: ActivationActionSource,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationActionSource {
    FixedProbe,
    Policy {
        identity: ProbeIdentity,
        model: String,
        stochastic: bool,
    },
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Conditioning {
    InitialScenario,
    FullSimulatorState { description: String },
    Observation { methodology: String },
    History { methodology: String },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetadata {
    pub conditioning: Conditioning,
    pub initial_action: ActionCondition,
    pub continuation_policy: String,
    pub stochastic_policy: bool,
    pub discount: f64,
    pub independent_future_streams: bool,
    pub seed_methodology: String,
    pub truncation_handling: String,
    #[serde(default)]
    pub complete_returns: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalRecord {
    pub returns: ArrayKey,
    pub seeds: ArrayKey,
    pub evaluation: EvaluationMetadata,
    pub probe_row: Option<u64>,
    pub initial_action: Option<ArrayKey>,
    #[serde(default)]
    pub conditioning_observation: Option<ArrayKey>,
    #[serde(default)]
    pub conditioning_mask: Option<ArrayKey>,
    /// Full physical simulator coordinates; future randomness is separately seeded.
    #[serde(default)]
    pub simulator_state: Option<ArrayKey>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryRecord {
    /// Optional evaluation-only T+1 rendered states.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<super::render::RenderTrack>,
    pub id: String,
    pub environment: String,
    /// Observations have T+1 rows. Actions, rewards and flags have T rows.
    pub observations: ArrayKey,
    pub actions: ArrayKey,
    pub rewards: ArrayKey,
    pub terminated: ArrayKey,
    pub truncated: ArrayKey,
    pub simulation_times: Option<ArrayKey>,
    pub action_masks: Option<ArrayKey>,
    /// Environment-specific numeric channels, with their meaning documented
    /// by the environment adapter. Unknown channels remain inspectable.
    pub frames: BTreeMap<String, ArrayKey>,
    pub evaluation: EvaluationMetadata,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub bytes: ArrayKey,
    pub signature: String,
    pub parameter_shapes: BTreeMap<String, Vec<usize>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshotRecord {
    pub id: String,
    pub architecture: String,
    pub configuration: serde_json::Value,
    pub models: BTreeMap<String, ModelArtifact>,
    pub capabilities: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum Record {
    Metrics(MetricRecord),
    Event(EventRecord),
    ProbeSet(ProbeSetRecord),
    Policy(PolicyRecord),
    Critic(CriticRecord),
    Activations(ActivationRecord),
    Empirical(EmpiricalRecord),
    Trajectory(TrajectoryRecord),
    ModelSnapshot(ModelSnapshotRecord),
    Typed(super::typed::TypedRecord),
}
impl Record {
    pub fn kind(&self) -> RecordKind {
        match self {
            Self::Metrics(_) => RecordKind::Metrics,
            Self::Event(_) => RecordKind::Event,
            Self::ProbeSet(_) => RecordKind::ProbeSet,
            Self::Policy(_) => RecordKind::Policy,
            Self::Critic(_) => RecordKind::Critic,
            Self::Activations(_) => RecordKind::Activations,
            Self::Empirical(_) => RecordKind::Empirical,
            Self::Trajectory(_) => RecordKind::Trajectory,
            Self::ModelSnapshot(_) => RecordKind::ModelSnapshot,
            Self::Typed(r) => r.kind.record_kind(),
        }
    }
    pub(crate) fn references(&self) -> Vec<&str> {
        let mut refs: Vec<&str> = vec![];
        match self {
            Self::Metrics(_) | Self::Event(_) => {}
            Self::ProbeSet(r) => {
                refs.extend([r.observations.as_str(), &r.row_ids]);
                refs.extend(r.action_masks.as_deref());
                refs.extend(r.fixed_actions.as_deref());
                refs.extend(r.histories.values().map(String::as_str));
            }
            Self::Policy(r) => {
                refs.push(&r.actions);
                refs.extend(r.entropy.as_deref());
                refs.extend(r.masks.as_deref());
                match &r.parameters {
                    PolicyParameters::Categorical { probabilities } => refs.push(probabilities),
                    PolicyParameters::Gaussian { mean, log_std, .. } => {
                        refs.extend([mean.as_str(), log_std])
                    }
                    PolicyParameters::Deterministic => {}
                }
            }
            Self::Critic(r) => {
                refs.extend(r.actions.as_deref());
                match &r.representation {
                    CriticRepresentation::Scalar { values }
                    | CriticRepresentation::Discrete { values } => refs.push(values),
                    CriticRepresentation::Quantiles {
                        values,
                        locations,
                        masses,
                    } => refs.extend([values.as_str(), locations, masses]),
                    CriticRepresentation::RawMoments { mean, raw_second } => {
                        refs.extend([mean.as_str(), raw_second])
                    }
                    CriticRepresentation::CategoricalRegression {
                        logits,
                        weights,
                        centers,
                        values,
                    } => refs.extend([logits.as_str(), weights, centers, values]),
                    CriticRepresentation::CategoricalReturn {
                        atoms,
                        probabilities,
                    } => {
                        refs.extend([atoms.as_str(), probabilities]);
                    }
                }
            }
            Self::Activations(r) => {
                if let ActivationInputs::ObservationsAndActions { actions, .. } = &r.inputs {
                    refs.push(actions.as_str());
                }
                refs.extend(r.layers.values().map(String::as_str));
                refs.extend(r.selected_neurons.values().map(String::as_str));
            }
            Self::Empirical(r) => {
                refs.extend([r.returns.as_str(), &r.seeds]);
                refs.extend(r.initial_action.as_deref());
                refs.extend(r.conditioning_observation.as_deref());
                refs.extend(r.conditioning_mask.as_deref());
                refs.extend(r.simulator_state.as_deref());
            }
            Self::Trajectory(r) => {
                refs.extend([
                    r.observations.as_str(),
                    &r.actions,
                    &r.rewards,
                    &r.terminated,
                    &r.truncated,
                ]);
                refs.extend(r.simulation_times.as_deref());
                refs.extend(r.action_masks.as_deref());
                refs.extend(r.frames.values().map(String::as_str));
                if let Some(render) = &r.render {
                    refs.extend([render.offsets.as_str(), render.data.as_str()]);
                }
            }
            Self::ModelSnapshot(r) => refs.extend(r.models.values().map(|v| v.bytes.as_str())),
            Self::Typed(r) => refs.push(&r.payload),
        }
        refs
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    U64,
    U8,
}
impl DType {
    pub fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 | Self::U64 => 8,
            Self::U8 => 1,
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum Values {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U64(Vec<u64>),
    U8(Vec<u8>),
}
#[derive(Debug, Clone)]
pub struct Array {
    shape: Vec<usize>,
    pub(crate) values: Values,
}
impl Array {
    pub fn f32(shape: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        Self::checked(shape, Values::F32(values))
    }
    pub fn f64(shape: Vec<usize>, values: Vec<f64>) -> Result<Self> {
        Self::checked(shape, Values::F64(values))
    }
    pub fn u64(shape: Vec<usize>, values: Vec<u64>) -> Result<Self> {
        Self::checked(shape, Values::U64(values))
    }
    pub fn u8(shape: Vec<usize>, values: Vec<u8>) -> Result<Self> {
        Self::checked(shape, Values::U8(values))
    }
    pub(crate) fn checked(shape: Vec<usize>, values: Values) -> Result<Self> {
        let array = Self { shape, values };
        ensure(
            elements(&array.shape)? == array.len(),
            "diagnostic array shape mismatch",
        )?;
        Ok(array)
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn len(&self) -> usize {
        match &self.values {
            Values::F32(v) => v.len(),
            Values::F64(v) => v.len(),
            Values::U64(v) => v.len(),
            Values::U8(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dtype(&self) -> DType {
        match self.values {
            Values::F32(_) => DType::F32,
            Values::F64(_) => DType::F64,
            Values::U64(_) => DType::U64,
            Values::U8(_) => DType::U8,
        }
    }
    pub fn byte_len(&self) -> usize {
        self.len() * self.dtype().bytes()
    }
    pub fn as_f32(&self) -> Result<&[f32]> {
        if let Values::F32(v) = &self.values {
            Ok(v)
        } else {
            Err(type_error())
        }
    }
    pub fn as_f64(&self) -> Result<&[f64]> {
        if let Values::F64(v) = &self.values {
            Ok(v)
        } else {
            Err(type_error())
        }
    }
    pub fn as_u64(&self) -> Result<&[u64]> {
        if let Values::U64(v) = &self.values {
            Ok(v)
        } else {
            Err(type_error())
        }
    }
    pub fn as_u8(&self) -> Result<&[u8]> {
        if let Values::U8(v) = &self.values {
            Ok(v)
        } else {
            Err(type_error())
        }
    }
    pub fn to_f64(&self) -> Vec<f64> {
        match &self.values {
            Values::F32(v) => v.iter().map(|&v| v as f64).collect(),
            Values::F64(v) => v.clone(),
            Values::U64(v) => v.iter().map(|&v| v as f64).collect(),
            Values::U8(v) => v.iter().map(|&v| v as f64).collect(),
        }
    }
}
fn type_error() -> Error {
    Error::Shape("diagnostic array dtype mismatch".into())
}
pub(crate) fn elements(shape: &[usize]) -> Result<usize> {
    ensure(
        !shape.is_empty() && shape.len() <= 8,
        "invalid diagnostic array rank",
    )?;
    shape.iter().try_fold(1usize, |size, &n| {
        size.checked_mul(n)
            .ok_or_else(|| Error::Shape("diagnostic array size overflow".into()))
    })
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub context: RecordContext,
    pub record: Record,
    pub arrays: BTreeMap<String, Array>,
}
impl Chunk {
    /// Conservative retained charge for numeric contents, metadata and tree/shape
    /// overhead. This is ownership accounting, not an allocator or RSS census.
    pub fn retained_charge(&self) -> usize {
        retained_chunk_charge(
            &self.context,
            &self.record,
            self.array_bytes(),
            self.arrays.len(),
        )
    }
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.context.counters.updates <= self.context.identity.learner_update_revision,
            "diagnostic revision precedes run update count",
        )?;
        ensure(
            self.context.simulation_time.is_none_or(f64::is_finite),
            "invalid simulation time",
        )?;
        ensure(
            self.context
                .wall_elapsed_seconds
                .is_none_or(|seconds| seconds.is_finite() && seconds >= 0.),
            "invalid recording-session elapsed time",
        )?;
        for key in self.record.references() {
            ensure(
                self.arrays.contains_key(key),
                "diagnostic record references a missing array",
            )?;
        }
        if let Record::Metrics(r) = &self.record {
            ensure(
                r.values.values().all(|v| v.is_finite()),
                "nonfinite metric: record a numerical event instead",
            )?;
        }
        super::validation::validate(self)?;
        Ok(())
    }
    pub fn array_bytes(&self) -> usize {
        self.arrays.values().map(Array::byte_len).sum()
    }
}
pub(super) fn retained_chunk_charge(
    context: &RecordContext,
    record: &Record,
    array_bytes: usize,
    array_count: usize,
) -> usize {
    let metadata = serde_json::to_vec(&(context, record)).map_or(0, |v| v.len());
    array_bytes
        .saturating_add(metadata.saturating_mul(2))
        .saturating_add(array_count.saturating_mul(256))
        .saturating_add(std::mem::size_of::<Chunk>())
}

/// Encode one homogeneous batch of diagnostic actions without changing its precision.
pub fn actions_array(actions: &[Action]) -> Result<Array> {
    ensure(!actions.is_empty(), "empty diagnostic actions")?;
    match &actions[0] {
        Action::Discrete(_) => Array::u64(
            vec![actions.len()],
            actions
                .iter()
                .map(|a| match a {
                    Action::Discrete(i) => Ok(*i as u64),
                    _ => Err(crate::Error::Action("mixed diagnostic actions".into())),
                })
                .collect::<Result<_>>()?,
        ),
        Action::Continuous(first) => {
            let width = first.len();
            ensure(width > 0, "empty continuous action")?;
            let mut values = Vec::with_capacity(actions.len() * width);
            for action in actions {
                let Action::Continuous(v) = action else {
                    return Err(crate::Error::Action("mixed diagnostic actions".into()));
                };
                ensure(
                    v.len() == width && v.iter().all(|v| v.is_finite()),
                    "invalid diagnostic action vector",
                )?;
                values.extend_from_slice(v);
            }
            Array::f32(vec![actions.len(), width], values)
        }
        Action::Conditional(_) => Err(crate::Error::Action(
            "numerical diagnostics require discrete or continuous actions".into(),
        )),
    }
}

/// Decode one recorded action batch against its declared action space.
pub(crate) fn decode_actions(array: &Array, space: &ActionSpace) -> Result<Vec<Action>> {
    let actions = match space {
        ActionSpace::Discrete { .. } if array.shape().len() == 1 => array
            .as_u64()?
            .iter()
            .map(|&v| {
                usize::try_from(v)
                    .map(Action::Discrete)
                    .map_err(|_| Error::Action("discrete action overflow".into()))
            })
            .collect::<Result<Vec<_>>>()?,
        ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
            if array.shape().get(1) == Some(&space.dimensions()) && array.shape().len() == 2 =>
        {
            match array.dtype() {
                DType::F32 => array
                    .as_f32()?
                    .chunks_exact(space.dimensions())
                    .map(|row| Action::Continuous(row.to_vec()))
                    .collect(),
                DType::F64 => array
                    .as_f64()?
                    .chunks_exact(space.dimensions())
                    .map(|row| Action::Continuous(row.iter().map(|&x| x as f32).collect()))
                    .collect(),
                _ => return Err(Error::Shape("invalid continuous action array dtype".into())),
            }
        }
        _ => {
            return Err(Error::Shape(
                "recorded action array differs from its action space".into(),
            ));
        }
    };
    for action in &actions {
        space.check(action, None)?;
    }
    Ok(actions)
}
