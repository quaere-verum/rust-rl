use super::{ActionAxis, ActionValue, Axes, Coordinate, EpisodeStatus, Limits, checked};
use rl::{
    Error, Result,
    core::action::ActionSpace,
    diagnostics::{
        evaluation::{EvaluationHorizon, evaluation_horizon},
        schema::*,
        visualization::{ActionDescriptor, EnvironmentSemantics, ObservationType},
    },
};
use std::{borrow::Cow, sync::Arc};

/// Cheap immutable clone; numeric vectors and source evidence share ownership.
#[derive(Clone, Debug)]
pub struct Episode(Arc<Prepared>);

#[derive(Debug)]
struct Prepared {
    context: RecordContext,
    record: TrajectoryRecord,
    observations: Vec<f64>,
    actions: Actions,
    rewards: Vec<f64>,
    times: Option<Vec<f64>>,
    terminated: Vec<u8>,
    truncated: Vec<u8>,
    masks: Option<(usize, Vec<u8>)>,
    suffix_returns: Option<Vec<f64>>,
    coordinates: Vec<Coordinate>,
    numeric_coordinates: Vec<usize>,
    axes: Option<Axes>,
    action_axes: Vec<ActionAxis>,
    action_labels: Vec<String>,
    action_units: Vec<Option<String>>,
    default_action_axis: ActionAxis,
    status: EpisodeStatus,
    realized_return: f64,
    retained_bytes: usize,
}
#[derive(Debug)]
enum Actions {
    Discrete(Vec<u64>),
    Continuous { dimensions: usize, values: Vec<f64> },
}
#[derive(Clone, Copy, Debug)]
pub struct Step<'a> {
    pub observation: &'a [f64],
    pub mask: Option<&'a [u8]>,
    pub action: Option<DecisionAction<'a>>,
    pub next_observation: Option<&'a [f64]>,
    pub terminated: Option<bool>,
    pub truncated: Option<bool>,
    /// One realized complete discounted suffix, never a conditional estimate.
    pub return_to_go: Option<f64>,
    /// At decision index t this is recorded r_(t+1). None at t=T.
    pub reward: Option<f64>,
    pub simulation_time: Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecisionAction<'a> {
    Discrete(u64),
    Continuous(&'a [f64]),
}

impl Episode {
    pub fn context(&self) -> &RecordContext {
        &self.0.context
    }
    pub fn record(&self) -> &TrajectoryRecord {
        &self.0.record
    }
    pub fn terminated(&self) -> &[u8] {
        &self.0.terminated
    }
    pub fn truncated(&self) -> &[u8] {
        &self.0.truncated
    }
    pub fn transitions(&self) -> usize {
        self.0.rewards.len()
    }
    pub fn coordinates(&self) -> &[Coordinate] {
        &self.0.coordinates
    }
    pub fn numeric_coordinates(&self) -> &[usize] {
        &self.0.numeric_coordinates
    }
    pub fn default_axes(&self) -> Option<Axes> {
        self.0.axes
    }
    pub fn action_axes(&self) -> &[ActionAxis] {
        &self.0.action_axes
    }
    pub fn default_action_axis(&self) -> ActionAxis {
        self.0.default_action_axis
    }
    pub fn status(&self) -> EpisodeStatus {
        self.0.status
    }
    pub fn realized_return(&self) -> f64 {
        self.0.realized_return
    }
    pub fn retained_bytes(&self) -> usize {
        self.0.retained_bytes
    }
    pub fn step(&self, t: usize) -> Result<Step<'_>> {
        checked(
            t <= self.transitions(),
            "dynamics cursor exceeds final state",
        )?;
        let d = self.0.coordinates.len();
        Ok(Step {
            observation: &self.0.observations[t * d..(t + 1) * d],
            mask: self
                .0
                .masks
                .as_ref()
                .map(|(width, values)| &values[t * width..(t + 1) * width]),
            action: if t == self.transitions() {
                None
            } else {
                Some(match &self.0.actions {
                    Actions::Discrete(values) => DecisionAction::Discrete(values[t]),
                    Actions::Continuous { dimensions, values } => {
                        DecisionAction::Continuous(&values[t * dimensions..(t + 1) * dimensions])
                    }
                })
            },
            next_observation: (t < self.transitions())
                .then(|| &self.0.observations[(t + 1) * d..(t + 2) * d]),
            terminated: self.0.terminated.get(t).map(|&v| v != 0),
            truncated: self.0.truncated.get(t).map(|&v| v != 0),
            return_to_go: self
                .0
                .suffix_returns
                .as_ref()
                .and_then(|v| v.get(t))
                .copied(),
            reward: self.0.rewards.get(t).copied(),
            simulation_time: self.0.times.as_ref().map(|v| v[t]),
        })
    }
    pub(super) fn validate_axes(&self, axes: Axes) -> Result<()> {
        checked(axes.x != axes.y, "dynamics axes must differ")?;
        self.validate_coordinate(axes.x)?;
        self.validate_coordinate(axes.y)
    }
    fn validate_coordinate(&self, axis: usize) -> Result<()> {
        checked(
            self.0.coordinates.get(axis).is_some_and(|c| c.numeric),
            "dynamics projection requires numerical coordinates",
        )
    }
    pub fn project(&self, t: usize, axes: Axes) -> Result<[f64; 2]> {
        self.validate_axes(axes)?;
        let s = self.step(t)?.observation;
        Ok([s[axes.x], s[axes.y]])
    }
    pub fn project_state3(&self, t: usize, axes: [usize; 3]) -> Result<[f64; 3]> {
        checked(
            axes[0] != axes[1] && axes[0] != axes[2] && axes[1] != axes[2],
            "dynamics 3D axes must differ",
        )?;
        for axis in axes {
            self.validate_coordinate(axis)?;
        }
        let s = self.step(t)?.observation;
        Ok([s[axes[0]], s[axes[1]], s[axes[2]]])
    }
    fn action_index(&self, axis: ActionAxis) -> Result<usize> {
        match (&self.0.actions, axis) {
            (Actions::Discrete(_), ActionAxis::Discrete) => Ok(0),
            (Actions::Continuous { dimensions, .. }, ActionAxis::Continuous(j))
                if j < *dimensions =>
            {
                Ok(j)
            }
            _ => Err(Error::Shape("invalid dynamics action axis".into())),
        }
    }
    pub fn action_label(&self, axis: ActionAxis) -> Result<&str> {
        let j = self.action_index(axis)?;
        if axis == ActionAxis::Discrete {
            return Ok("Action ID");
        }
        Ok(&self.0.action_labels[j])
    }
    pub fn action_unit(&self, axis: ActionAxis) -> Result<Option<&str>> {
        let j = self.action_index(axis)?;
        Ok(self.0.action_units[j].as_deref())
    }
    pub fn discrete_label(&self, id: u64) -> Option<&str> {
        if !matches!(self.0.actions, Actions::Discrete(_)) {
            return None;
        }
        self.0
            .action_labels
            .get(usize::try_from(id).ok()?)
            .map(String::as_str)
    }
    pub fn action_value(&self, t: usize, axis: ActionAxis) -> Result<Option<ActionValue>> {
        let j = self.action_index(axis)?;
        checked(
            t <= self.transitions(),
            "dynamics cursor exceeds final state",
        )?;
        if t == self.transitions() {
            return Ok(None);
        }
        Ok(Some(match &self.0.actions {
            Actions::Discrete(v) => ActionValue::Discrete(v[t]),
            Actions::Continuous { dimensions, values } => {
                ActionValue::Continuous(values[t * dimensions + j])
            }
        }))
    }
    pub fn project_state_action(
        &self,
        t: usize,
        axes: Axes,
        action: ActionAxis,
    ) -> Result<Option<[f64; 3]>> {
        checked(
            matches!(action, ActionAxis::Continuous(_)),
            "categorical actions are not continuous 3D coordinates",
        )?;
        let [x, y] = self.project(t, axes)?;
        Ok(match self.action_value(t, action)? {
            Some(ActionValue::Continuous(z)) => Some([x, y, z]),
            _ => None,
        })
    }
}

fn array<'a>(chunk: &'a Chunk, key: &str) -> Result<&'a Array> {
    chunk
        .arrays
        .get(key)
        .ok_or_else(|| Error::Shape("missing dynamics array".into()))
}
fn floating(array: &Array) -> Result<()> {
    checked(
        matches!(array.dtype(), DType::F32 | DType::F64),
        "dynamics requires floating values",
    )
}
fn decode(array: &Array) -> Result<Vec<f64>> {
    floating(array)?;
    let values = array.to_f64();
    checked(
        values.iter().all(|x| x.is_finite()),
        "nonfinite dynamics values",
    )?;
    Ok(values)
}
fn add(total: &mut usize, bytes: usize) -> Result<()> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| Error::Shape("dynamics byte accounting overflow".into()))?;
    Ok(())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Shape("dynamics byte accounting overflow".into()))
}

fn observation_label(manifest: &RunManifest, index: usize) -> Cow<'_, str> {
    manifest
        .environment_descriptor
        .as_ref()
        .and_then(|d| d.observations.get(index))
        .map(|field| field.label.as_str())
        .or_else(|| {
            manifest
                .observation_labels
                .get(index)
                .map(String::as_str)
                .filter(|label| !label.trim().is_empty())
        })
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(format!("s[{index}]")))
}

fn action_label(manifest: &RunManifest, index: usize) -> Cow<'_, str> {
    let discrete = matches!(manifest.action_space, ActionSpace::Discrete { .. });
    let declared = manifest
        .environment_descriptor
        .as_ref()
        .and_then(|d| match &d.actions {
            ActionDescriptor::Discrete { choices } => {
                choices.get(index).map(|choice| choice.label.as_str())
            }
            ActionDescriptor::Continuous { components, .. } => components
                .get(index)
                .map(|component| component.label.as_str()),
        });
    declared.map(Cow::Borrowed).unwrap_or_else(|| {
        Cow::Owned(if discrete {
            index.to_string()
        } else {
            format!("a[{index}]")
        })
    })
}

const BUDGET_ERROR: &str = "dynamics episode exceeds byte budget; exact samples cannot be admitted";

fn add_charge(bytes: &mut usize, extra: usize, cap: usize) -> Result<()> {
    add(bytes, extra)?;
    checked(*bytes <= cap, BUDGET_ERROR)
}

fn string_charge(bytes: &mut usize, text: &str, cap: usize) -> Result<()> {
    // Owned copies use their length; this margin also covers format! capacity
    // for generated labels and small allocator size classes.
    add_charge(bytes, mul(text.len(), 2)?, cap)?;
    add_charge(bytes, 32, cap)
}

/// Upper bound on the final owned allocations, before any full-array decoding.
/// Numeric vectors are allocated at exact source lengths. Serialized source
/// context/record metadata is doubled to cover string and tree storage. Original
/// raw numeric arrays and ancillary frames/masks are not cloned. Coordinate/action
/// collections use typed allocation sizes plus copied label/unit string charges.
/// Unretained manifest metadata is excluded. This is ownership accounting, not RSS.
fn charge(
    chunk: &Chunk,
    manifest: &RunManifest,
    decoded_values: usize,
    action_labels: usize,
    transitions: usize,
    cap: usize,
) -> Result<usize> {
    let mut bytes = std::mem::size_of::<Prepared>() + std::mem::size_of::<Episode>() + 64;
    checked(bytes <= cap, BUDGET_ERROR)?;
    add_charge(&mut bytes, mul(decoded_values, 8)?, cap)?;
    add_charge(&mut bytes, mul(transitions, 2)?, cap)?;
    // Complete discounted suffixes and T+1 effective legal masks are prepared
    // once off paint; terminal observations retain their recorded mask too.
    if transitions <= rl::diagnostics::value::MAX_RETURN_STEPS {
        add_charge(&mut bytes, mul(transitions, 8)?, cap)?;
    }
    if let Record::Trajectory(record) = &chunk.record
        && let Some(key) = &record.action_masks
    {
        add_charge(&mut bytes, array(chunk, key)?.len(), cap)?;
    }
    add_charge(
        &mut bytes,
        mul(
            manifest.observation_dim,
            std::mem::size_of::<Coordinate>() + std::mem::size_of::<usize>(),
        )?,
        cap,
    )?;
    add_charge(
        &mut bytes,
        mul(
            action_labels,
            std::mem::size_of::<String>() + std::mem::size_of::<Option<String>>(),
        )?,
        cap,
    )?;
    let axes_count = if matches!(manifest.action_space, ActionSpace::Discrete { .. }) {
        1
    } else {
        action_labels
    };
    add_charge(
        &mut bytes,
        mul(axes_count, std::mem::size_of::<ActionAxis>())?,
        cap,
    )?;
    // Admit count-driven lower bounds before enumerating or formatting any
    // labels. A tiny legacy discrete record may declare billions of choices.
    // Frame-reference nodes and fixed collection costs are lower bounds too.
    if let Record::Trajectory(r) = &chunk.record {
        add_charge(&mut bytes, mul(r.frames.len(), 256)?, cap)?;
    }
    add_charge(&mut bytes, 8 * 256, cap)?;
    let metadata = serde_json::to_vec(&(&chunk.context, &chunk.record))?;
    add_charge(&mut bytes, mul(metadata.len(), 2)?, cap)?;
    let semantics = EnvironmentSemantics::new(manifest);
    for i in 0..manifest.observation_dim {
        string_charge(&mut bytes, &observation_label(manifest, i), cap)?;
        if let Some(unit) = semantics.observation(i).and_then(|c| c.unit) {
            string_charge(&mut bytes, unit, cap)?;
        }
    }
    for j in 0..action_labels {
        string_charge(&mut bytes, &action_label(manifest, j), cap)?;
        if let Some(unit) = semantics.action(j).and_then(|a| a.unit) {
            string_charge(&mut bytes, unit, cap)?;
        }
    }
    // References retained by the TrajectoryRecord include map key/node costs.
    if let Record::Trajectory(r) = &chunk.record {
        for key in r.frames.keys() {
            add_charge(&mut bytes, key.len(), cap)?;
        }
    }
    Ok(bytes)
}

/// Worker-callable full-episode admission and conversion. Source context and
/// trajectory metadata are retained alongside exact decoded values.
/// Frame/mask array references remain in record metadata; their payload stays in
/// the source recording. A caller must apply its own population/cache budget
/// across episodes; this limit bounds one admitted episode.
pub fn prepare_episode(chunk: &Chunk, manifest: &RunManifest, limits: &Limits) -> Result<Episode> {
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::Shape("dynamics requires a trajectory record".into()));
    };
    let observations = array(chunk, &record.observations)?;
    let actions = array(chunk, &record.actions)?;
    let rewards = array(chunk, &record.rewards)?;
    let t = rewards.shape()[0];
    checked(t > 0, "dynamics trajectory has no transitions")?;
    let states = t
        .checked_add(1)
        .ok_or_else(|| Error::Shape("dynamics transition count overflow".into()))?;
    checked(
        observations.shape() == [states, manifest.observation_dim],
        "dynamics observation dimension or T+1 alignment mismatch",
    )?;
    checked(
        rewards.shape() == [t] || rewards.shape() == [t, 1],
        "dynamics rewards must be scalar",
    )?;
    let labels = match &manifest.action_space {
        ActionSpace::Discrete { count } => *count,
        s => s.dimensions(),
    };
    let mut decoded_values = 0;
    for a in [observations, actions, rewards] {
        add(&mut decoded_values, a.len())?;
    }
    if let Some(key) = &record.simulation_times {
        add(&mut decoded_values, array(chunk, key)?.len())?;
    }
    let retained_bytes = charge(
        chunk,
        manifest,
        decoded_values,
        labels,
        t,
        limits.max_episode_bytes,
    )?;
    manifest.validate()?;
    checked(
        chunk.context.identity.run_id == manifest.run_id,
        "dynamics trajectory belongs to another run",
    )?;
    checked(
        !chunk.context.identity.model_role.is_empty()
            && chunk.context.counters.updates <= chunk.context.identity.learner_update_revision
            && chunk.context.simulation_time.is_none_or(f64::is_finite),
        "invalid dynamics record context",
    )?;
    checked(
        chunk
            .context
            .trajectory
            .as_ref()
            .is_none_or(|id| id == &record.id),
        "dynamics trajectory identity mismatch",
    )?;
    let evaluation = &record.evaluation;
    checked(
        evaluation.discount.is_finite()
            && (0. ..=1.).contains(&evaluation.discount)
            && !evaluation.continuation_policy.is_empty()
            && !evaluation.seed_methodology.is_empty()
            && !evaluation.truncation_handling.is_empty(),
        "invalid dynamics evaluation metadata",
    )?;
    let terminated = array(chunk, &record.terminated)?;
    let truncated = array(chunk, &record.truncated)?;
    checked(
        terminated.shape() == [t] && truncated.shape() == [t],
        "dynamics flag shape mismatch",
    )?;
    let terminated = terminated.as_u8()?;
    let truncated = truncated.as_u8()?;
    checked(
        terminated
            .iter()
            .zip(truncated)
            .enumerate()
            .all(|(i, (&a, &b))| a <= 1 && b <= 1 && a + b <= 1 && (i + 1 == t || a + b == 0)),
        "dynamics trajectory crosses an episode boundary or has invalid flags",
    )?;
    let reached_horizon = evaluation_horizon(chunk)? == EvaluationHorizon::FiniteDecisions(t);
    checked(
        !evaluation.complete_returns || terminated[t - 1] == 1 || reached_horizon,
        "complete dynamics return requires the declared objective boundary",
    )?;
    for key in record.frames.values() {
        checked(
            array(chunk, key)?.shape()[0] == states,
            "dynamics frame T+1 alignment mismatch",
        )?;
    }
    let observations = decode(observations)?;
    let rewards = decode(rewards)?;
    let actions = match &manifest.action_space {
        ActionSpace::Discrete { count } => {
            checked(
                actions.shape() == [t] && actions.dtype() == DType::U64,
                "dynamics discrete action shape mismatch",
            )?;
            let v = actions.as_u64()?;
            checked(
                v.iter().all(|&a| a < *count as u64),
                "dynamics discrete action ID exceeds space",
            )?;
            Actions::Discrete(v.to_vec())
        }
        ActionSpace::Conditional(_) => {
            return Err(rl::Error::Config(
                "dynamics requires a numeric action space".into(),
            ));
        }
        space => {
            let dimensions = space.dimensions();
            checked(
                actions.shape() == [t, dimensions],
                "dynamics continuous action dimensions mismatch",
            )?;
            let values = decode(actions)?;
            if let ActionSpace::Box { low, high } = space {
                checked(
                    values.chunks_exact(dimensions).all(|a| {
                        a.iter()
                            .zip(low.iter().zip(high))
                            .all(|(&x, (&lo, &hi))| x >= lo as f64 && x <= hi as f64)
                    }),
                    "dynamics bounded action violates space",
                )?;
            }
            Actions::Continuous { dimensions, values }
        }
    };
    let times = record
        .simulation_times
        .as_ref()
        .map(|key| -> Result<Vec<f64>> {
            let a = array(chunk, key)?;
            checked(
                a.shape() == [states] || a.shape() == [states, 1],
                "dynamics clock T+1 alignment mismatch",
            )?;
            let values = decode(a)?;
            checked(
                values.windows(2).all(|w| w[0] <= w[1]),
                "nonmonotone dynamics simulation clock",
            )?;
            Ok(values)
        })
        .transpose()?;
    let masks = if let Some(key) = &record.action_masks {
        let mask = array(chunk, key)?;
        let ActionSpace::Discrete { count } = manifest.action_space else {
            return Err(Error::Shape(
                "continuous dynamics actions cannot use discrete masks".into(),
            ));
        };
        checked(
            mask.shape() == [states, count],
            "dynamics action mask dimensions mismatch",
        )?;
        let values = mask.as_u8()?;
        checked(
            values.iter().all(|&b| b <= 1),
            "invalid dynamics action mask",
        )?;
        if let Actions::Discrete(ids) = &actions {
            checked(
                ids.iter()
                    .enumerate()
                    .all(|(t, &id)| values[t * count + id as usize] == 1),
                "dynamics action violates mask",
            )?;
        }
        Some((count, values.to_vec()))
    } else {
        None
    };
    let suffix_returns = if t <= rl::diagnostics::value::MAX_RETURN_STEPS {
        rl::diagnostics::value::trajectory_return_suffixes(chunk)?
    } else {
        None
    };
    let semantics = EnvironmentSemantics::new(manifest);
    let coordinates: Vec<_> = (0..manifest.observation_dim)
        .map(|i| {
            let view = semantics
                .observation(i)
                .expect("checked observation dimension");
            Coordinate {
                label: observation_label(manifest, i).into_owned(),
                unit: view.unit.map(str::to_owned),
                range: view.range.map(|r| r.map(f64::from)),
                numeric: !matches!(view.kind, Some(ObservationType::Categorical { .. })),
            }
        })
        .collect();
    let mut numeric_coordinates = Vec::with_capacity(coordinates.len());
    for (i, c) in coordinates.iter().enumerate() {
        if c.numeric {
            numeric_coordinates.push(i);
        }
    }
    let mut preferred = Vec::with_capacity(coordinates.len());
    let mut seen = vec![false; coordinates.len()];
    if let Some(descriptor) = semantics.descriptor() {
        for id in &descriptor.preferences.coordinates {
            if let Some(i) = descriptor.observations.iter().position(|f| &f.id == id)
                && coordinates[i].numeric
                && !seen[i]
            {
                preferred.push(i);
                seen[i] = true;
            }
        }
        for (i, field) in descriptor.observations.iter().enumerate() {
            if field.coordinate && coordinates[i].numeric && !seen[i] {
                preferred.push(i);
                seen[i] = true;
            }
        }
    }
    for &i in &numeric_coordinates {
        if !seen[i] {
            preferred.push(i);
            seen[i] = true;
        }
    }
    let axes = (preferred.len() >= 2).then(|| Axes {
        x: preferred[0],
        y: preferred[1],
    });
    let discrete = matches!(manifest.action_space, ActionSpace::Discrete { .. });
    let action_axes = if discrete {
        vec![ActionAxis::Discrete]
    } else {
        (0..labels).map(ActionAxis::Continuous).collect()
    };
    let default_action_axis = if discrete {
        ActionAxis::Discrete
    } else {
        let index = match semantics.descriptor().map(|d| &d.actions) {
            Some(ActionDescriptor::Continuous {
                preferred_component: Some(j),
                ..
            }) => *j,
            _ => 0,
        };
        ActionAxis::Continuous(index)
    };
    let action_labels = (0..labels)
        .map(|j| action_label(manifest, j).into_owned())
        .collect();
    let action_units = (0..labels)
        .map(|j| semantics.action(j).and_then(|a| a.unit.map(str::to_owned)))
        .collect();
    let status = if terminated[t - 1] != 0 {
        EpisodeStatus::Terminated
    } else if reached_horizon && evaluation.complete_returns {
        EpisodeStatus::FiniteHorizon
    } else if truncated[t - 1] != 0 {
        EpisodeStatus::Truncated
    } else {
        EpisodeStatus::Prefix
    };
    let realized_return: f64 = rewards.iter().sum();
    checked(
        realized_return.is_finite(),
        "nonfinite dynamics realized return",
    )?;
    Ok(Episode(Arc::new(Prepared {
        context: chunk.context.clone(),
        record: record.clone(),
        observations,
        actions,
        rewards,
        times,
        terminated: terminated.to_vec(),
        truncated: truncated.to_vec(),
        masks,
        suffix_returns,
        coordinates,
        numeric_coordinates,
        axes,
        action_axes,
        action_labels,
        action_units,
        default_action_axis,
        status,
        realized_return,
        retained_bytes,
    })))
}
