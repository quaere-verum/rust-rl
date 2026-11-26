use super::action::{Action, ActionSpace};
use super::graph::{GraphObservation, GraphSchema};
use super::reward::Reward;
use super::structured::{ObservationSchema, StructuredObservation};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod finite_actions;
pub(crate) mod history;
mod public_data;
pub use history::PublicHistory;
#[cfg(feature = "neural")]
pub(crate) use history::WeakPublicHistory;
pub use public_data::PublicData;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseObservation {
    pub values: Vec<f32>,
    pub action_mask: Option<Vec<bool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_data: Option<PublicData>,
}

/// An environment's public observation symbol, which may be partially observable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscreteObservation {
    pub symbol: u64,
    pub action_mask: Option<Vec<bool>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Observation {
    Dense(DenseObservation),
    Discrete(DiscreteObservation),
    Structured(Arc<StructuredObservation>),
    Graph(Arc<GraphObservation>),
}

/// Public symbols have IDs from zero up to the exclusive `symbols` bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscreteObservationSpace {
    pub symbols: u64,
}

/// Both the structured and sparse graph vocabularies bound to an environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphObservationSpace {
    pub base: ObservationSchema,
    pub graph: GraphSchema,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ObservationSpace {
    Dense(usize),
    Discrete(DiscreteObservationSpace),
    Structured(Arc<ObservationSchema>),
    Graph(Arc<GraphObservationSpace>),
}

impl From<usize> for ObservationSpace {
    fn from(width: usize) -> Self {
        Self::Dense(width)
    }
}

impl ObservationSpace {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Dense(width) => super::ensure(*width > 0, "empty dense observation space"),
            Self::Discrete(space) => {
                super::ensure(space.symbols > 0, "empty discrete observation space")
            }
            Self::Structured(schema) => schema.validate(),
            Self::Graph(schema) => {
                schema.base.validate()?;
                schema.graph.validate()
            }
        }
    }

    /// Returns the declared width of a dense space.
    #[inline]
    pub fn dense_width(&self) -> Option<usize> {
        match self {
            Self::Dense(width) => Some(*width),
            Self::Discrete(_) | Self::Structured(_) | Self::Graph(_) => None,
        }
    }

    /// Width required by an operation over a nonempty dense observation.
    pub fn require_dense_width(&self) -> Result<usize> {
        self.dense_width()
            .filter(|&width| width > 0)
            .ok_or_else(|| {
                crate::Error::Config("expected a nonempty dense observation space".into())
            })
    }
}

impl Observation {
    pub fn new(values: Vec<f32>) -> Self {
        Self::dense(values, None)
    }

    pub fn dense(values: Vec<f32>, action_mask: Option<Vec<bool>>) -> Self {
        Self::Dense(DenseObservation {
            values,
            action_mask,
            public_data: None,
        })
    }

    pub fn discrete(symbol: u64, action_mask: Option<Vec<bool>>) -> Self {
        Self::Discrete(DiscreteObservation {
            symbol,
            action_mask,
        })
    }

    pub fn dense_with_public_data(
        values: Vec<f32>,
        action_mask: Option<Vec<bool>>,
        public_data: PublicData,
    ) -> Self {
        Self::Dense(DenseObservation {
            values,
            action_mask,
            public_data: Some(public_data),
        })
    }

    pub fn as_dense(&self) -> Option<&DenseObservation> {
        match self {
            Self::Dense(dense) => Some(dense),
            _ => None,
        }
    }

    pub fn as_dense_mut(&mut self) -> Option<&mut DenseObservation> {
        match self {
            Self::Dense(dense) => Some(dense),
            _ => None,
        }
    }

    pub fn as_discrete(&self) -> Option<&DiscreteObservation> {
        match self {
            Self::Discrete(discrete) => Some(discrete),
            _ => None,
        }
    }

    /// View the public structured payload, including the base of a graph.
    pub fn as_structured(&self) -> Option<&StructuredObservation> {
        match self {
            Self::Dense(_) | Self::Discrete(_) => None,
            Self::Structured(frame) => Some(frame),
            Self::Graph(frame) => Some(&frame.observation),
        }
    }

    pub fn as_graph(&self) -> Option<&GraphObservation> {
        match self {
            Self::Graph(frame) => Some(frame),
            _ => None,
        }
    }

    /// Dense-kernel view. Structured observations have no dense input width.
    pub fn dense_values(&self) -> &[f32] {
        self.as_dense().map_or(&[], |dense| &dense.values)
    }

    pub fn dense_mask(&self) -> Option<&[bool]> {
        self.as_dense()
            .and_then(|dense| dense.action_mask.as_deref())
    }

    /// The public discrete-action mask for an observation that carries one.
    pub fn action_mask(&self) -> Option<&[bool]> {
        match self {
            Self::Dense(dense) => dense.action_mask.as_deref(),
            Self::Discrete(discrete) => discrete.action_mask.as_deref(),
            Self::Structured(_) | Self::Graph(_) => None,
        }
    }

    pub fn check_action(&self, space: &ActionSpace, action: &Action) -> Result<()> {
        space.check(action, self.action_mask())?;
        if let Action::Conditional(action) = action
            && !self
                .as_structured()
                .is_some_and(|frame| frame.is_available(action))
        {
            return Err(crate::Error::Action(
                "conditional action is unavailable".into(),
            ));
        }
        Ok(())
    }

    pub fn validate(&self, width: usize, space: &ActionSpace, live: bool) -> Result<()> {
        self.validate_in(&ObservationSpace::Dense(width), space, live)
    }

    pub fn validate_in(
        &self,
        observations: &ObservationSpace,
        space: &ActionSpace,
        live: bool,
    ) -> Result<()> {
        let mask = match (self, observations) {
            (Self::Dense(dense), ObservationSpace::Dense(width)) => {
                super::ensure(
                    !matches!(space, ActionSpace::Conditional(_))
                        && dense.values.len() == *width
                        && dense.values.iter().all(|x| x.is_finite()),
                    "invalid observation shape or nonfinite observation",
                )?;
                dense.action_mask.as_deref()
            }
            (Self::Discrete(discrete), ObservationSpace::Discrete(schema)) => {
                super::ensure(
                    !matches!(space, ActionSpace::Conditional(_))
                        && discrete.symbol < schema.symbols,
                    "invalid discrete observation symbol or action space",
                )?;
                discrete.action_mask.as_deref()
            }
            (Self::Structured(frame), ObservationSpace::Structured(schema)) => {
                return frame.validate(schema, space, live);
            }
            (Self::Graph(frame), ObservationSpace::Graph(schema)) => {
                frame.observation.validate(&schema.base, space, live)?;
                return frame.validate_graph(&schema.graph, usize::MAX);
            }
            _ => {
                return Err(crate::Error::Shape(
                    "observation kind disagrees with its space".into(),
                ));
            }
        };
        if let Some(mask) = mask {
            match space {
                ActionSpace::Discrete { count } => super::ensure(
                    mask.len() == *count && (!live || mask.iter().any(|x| *x)),
                    "invalid or empty live action mask",
                )?,
                _ => {
                    return Err(crate::Error::Config(
                        "continuous observations cannot have action masks".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dense(a), Self::Dense(b)) => {
                a.action_mask == b.action_mask
                    && a.public_data == b.public_data
                    && a.values.len() == b.values.len()
                    && a.values
                        .iter()
                        .zip(&b.values)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (Self::Discrete(a), Self::Discrete(b)) => a == b,
            (Self::Structured(a), Self::Structured(b)) => Arc::ptr_eq(a, b) || a.matches(b),
            (Self::Graph(a), Self::Graph(b)) => Arc::ptr_eq(a, b) || a.matches(b),
            _ => false,
        }
    }

    pub(crate) fn validate_event_history(&self, previous: Option<&Self>) -> Result<()> {
        if let Some(frame) = self.as_structured() {
            let previous = previous.and_then(Self::as_structured);
            if frame.has_event_stamps()
                || previous.is_some_and(StructuredObservation::has_event_stamps)
            {
                frame.introduced_events(previous)?;
            }
        }
        Ok(())
    }
}

/// The observation and optional factual ancestry of a new collected trajectory.
#[derive(Debug, Clone)]
pub struct EpisodeStart {
    pub observation: Observation,
    pub context: Option<ResumeContext>,
}

impl EpisodeStart {
    pub fn fresh(observation: Observation) -> Self {
        Self {
            observation,
            context: None,
        }
    }

    pub fn physical_timestep(&self) -> u64 {
        self.context
            .as_ref()
            .map_or(0, |context| context.physical_timestep)
    }

    pub fn decision_timestep(&self) -> u64 {
        self.context
            .as_ref()
            .map_or(0, |context| context.history.timestep())
    }

    pub fn validate(&self, width: usize, space: &ActionSpace) -> Result<()> {
        self.validate_in(&ObservationSpace::Dense(width), space)
    }

    pub fn validate_in(&self, observations: &ObservationSpace, space: &ActionSpace) -> Result<()> {
        self.observation.validate_in(observations, space, true)?;
        if let Some(context) = &self.context {
            super::ensure(
                context
                    .history
                    .matches(&self.observation, context.physical_timestep),
                "episode start disagrees with its public history",
            )?;
            context.history.validate_in(observations, space)?;
        } else {
            self.observation.validate_event_history(None)?;
        }
        Ok(())
    }
}

/// Public observations, actions and rewards ending at the restored observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResumeContext {
    pub history: PublicHistory,
    pub physical_timestep: u64,
}

/// Borrowed observations, optionally selecting active vector lanes without copying payloads.
#[derive(Debug, Clone, Copy)]
pub struct ObservationBatch<'a> {
    observations: &'a [Observation],
    references: Option<&'a [&'a Observation]>,
    lanes: Option<&'a [usize]>,
}
impl<'a> ObservationBatch<'a> {
    pub fn all(observations: &'a [Observation]) -> Self {
        Self {
            observations,
            references: None,
            lanes: None,
        }
    }
    pub fn from_refs(references: &'a [&'a Observation]) -> Self {
        Self {
            observations: &[],
            references: Some(references),
            lanes: None,
        }
    }
    pub fn selected(observations: &'a [Observation], lanes: &'a [usize]) -> Result<Self> {
        super::ensure(
            lanes.iter().all(|&i| i < observations.len()),
            "observation lane out of bounds",
        )?;
        Ok(Self {
            observations,
            references: None,
            lanes: Some(lanes),
        })
    }
    pub fn len(self) -> usize {
        self.lanes.map_or_else(
            || {
                self.references
                    .map_or(self.observations.len(), <[&Observation]>::len)
            },
            <[usize]>::len,
        )
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn get(self, index: usize) -> Option<&'a Observation> {
        let lane = if let Some(lanes) = self.lanes {
            *lanes.get(index)?
        } else {
            index
        };
        if let Some(references) = self.references {
            references.get(lane).copied()
        } else {
            self.observations.get(lane)
        }
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = &'a Observation> {
        (0..self.len()).map(move |i| self.get(i).expect("validated observation view"))
    }
}
impl<'a> From<&'a [Observation]> for ObservationBatch<'a> {
    fn from(value: &'a [Observation]) -> Self {
        Self::all(value)
    }
}
impl<'a> From<&'a Vec<Observation>> for ObservationBatch<'a> {
    fn from(value: &'a Vec<Observation>) -> Self {
        Self::all(value)
    }
}
impl<'a, const N: usize> From<&'a [Observation; N]> for ObservationBatch<'a> {
    fn from(value: &'a [Observation; N]) -> Self {
        Self::all(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    pub observation: Observation,
    pub reward: Reward,
    pub terminated: bool,
    pub truncated: bool,
}
impl Step {
    pub fn ended(&self) -> bool {
        self.terminated || self.truncated
    }
}

pub trait Env {
    /// The observation schema, stable across resets and steps.
    fn observation_space(&self) -> ObservationSpace;
    /// Convenience width for dense observations.
    fn dense_observation_dim(&self) -> Option<usize> {
        self.observation_space().dense_width()
    }
    fn action_space(&self) -> &ActionSpace;
    /// Reseeds when supplied; `None` continues this environment's owned RNG stream.
    /// The declared observation schema and action grammar are stable across resets and steps.
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart>;
    /// Enforces the current observation's action availability. Returns the
    /// actual final observation before reset. An ended episode requires reset.
    fn step(&mut self, action: &Action) -> Result<Step>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeLimit<E> {
    pub inner: E,
    limit: usize,
    elapsed: usize,
    ended: bool,
}
impl<E: Env> TimeLimit<E> {
    pub fn new(inner: E, limit: usize) -> Result<Self> {
        super::ensure(limit > 0, "time limit must be positive")?;
        Ok(Self {
            inner,
            limit,
            elapsed: 0,
            ended: true,
        })
    }
}
impl<E: Env> Env for TimeLimit<E> {
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.inner.action_space()
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        let start = self.inner.reset(seed)?;
        self.ended = true;
        let elapsed = usize::try_from(start.physical_timestep())
            .map_err(|_| crate::Error::State("restored time exceeds the episode clock".into()))?;
        super::ensure(
            elapsed < self.limit,
            "restored state reaches the time limit",
        )?;
        self.elapsed = elapsed;
        self.ended = false;
        Ok(start)
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        if self.ended {
            return Err(crate::Error::State("reset required".into()));
        }
        let mut step = self.inner.step(action)?;
        self.elapsed = self
            .elapsed
            .checked_add(step.reward.duration() as usize)
            .ok_or_else(|| crate::Error::State("episode clock exhausted".into()))?;
        if self.elapsed >= self.limit && !step.terminated {
            step.truncated = true;
        }
        self.ended = step.ended();
        Ok(step)
    }
}
impl<E: Env + super::checkpoint::Snapshot> super::checkpoint::Snapshot for TimeLimit<E> {
    fn validate_snapshot(&self) -> Result<()> {
        super::ensure(
            self.limit > 0 && (self.elapsed < self.limit || self.ended),
            "invalid time-limit snapshot",
        )?;
        self.inner.validate_snapshot()
    }
}

/// Presents an unbounded policy action space while clipping actions sent to a
/// finite Box environment. Collectors retain the raw policy action, so its
/// Gaussian likelihood remains the likelihood of the sampled action.
/// When numeric frames are recorded, `clip_action.low` and `clip_action.high`
/// contain the physical bounds at every state. Recorded trajectory actions are
/// raw policy samples; clamp them to those bounds to recover physical actions.
/// Evaluations without numeric frames do not record the physical bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipAction<E> {
    inner: E,
    physical_space: ActionSpace,
    policy_space: ActionSpace,
}

impl<E: Env> ClipAction<E> {
    pub fn new(inner: E) -> Result<Self> {
        let physical_space = inner.action_space().clone();
        physical_space.validate()?;
        let ActionSpace::Box { low, .. } = &physical_space else {
            return Err(crate::Error::Config(
                "action clipping requires a finite Box space".into(),
            ));
        };
        let policy_space = ActionSpace::Unbounded {
            dimensions: low.len(),
        };
        Ok(Self {
            inner,
            physical_space,
            policy_space,
        })
    }

    pub fn inner(&self) -> &E {
        &self.inner
    }
    pub fn into_inner(self) -> E {
        self.inner
    }
    pub fn physical_action_space(&self) -> &ActionSpace {
        &self.physical_space
    }

    pub(crate) fn validate(&self) -> Result<()> {
        self.physical_space.validate()?;
        let ActionSpace::Box { low, .. } = &self.physical_space else {
            return Err(crate::Error::Config(
                "action clipping requires a finite Box space".into(),
            ));
        };
        super::ensure(
            self.inner.action_space() == &self.physical_space
                && self.policy_space
                    == (ActionSpace::Unbounded {
                        dimensions: low.len(),
                    }),
            "action clipping space changed or is invalid",
        )
    }

    /// Compute the physical action without stepping or changing the raw sample.
    pub fn clipped_action(&self, action: &Action) -> Result<Action> {
        self.validate()?;
        self.policy_space.check(action, None)?;
        let Action::Continuous(values) = action else {
            unreachable!()
        };
        let ActionSpace::Box { low, high } = &self.physical_space else {
            unreachable!()
        };
        Ok(Action::Continuous(
            values
                .iter()
                .zip(low.iter().zip(high))
                .map(|(&x, (&l, &h))| x.clamp(l, h))
                .collect(),
        ))
    }
}

impl<E: Env> Env for ClipAction<E> {
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.policy_space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        self.validate()?;
        let start = self.inner.reset(seed)?;
        self.validate()?;
        super::ensure(
            start.context.is_none(),
            "place ClipAction inside the checkpoint wrapper so public history records raw actions",
        )?;
        Ok(start)
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        let clipped = self.clipped_action(action)?;
        let step = self.inner.step(&clipped)?;
        self.validate()?;
        Ok(step)
    }
}

impl<E: Env + super::checkpoint::Snapshot> super::checkpoint::Snapshot for ClipAction<E> {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate()?;
        self.inner.validate_snapshot()
    }
}
