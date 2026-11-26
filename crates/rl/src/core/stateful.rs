//! Optional typed internal-state inference. Memory is post-observation h_t.
use super::{
    action::{Action, ActionSpace},
    actor::{ActMode, Actor},
    checkpoint::LearnerRecord,
    ensure,
    env::{Observation, ObservationBatch, ObservationSpace},
    reward::Reward,
};
use crate::Result;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[path = "stateful_identity.rs"]
mod identity;
pub(crate) use identity::public_history_identity;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreviousStep {
    pub action: Action,
    pub reward: f32,
}
#[derive(Debug, Clone, Copy)]
pub struct PreviousInput<'a> {
    pub action: &'a Action,
    pub reward: f32,
    pub factual: Option<FactualInput<'a>>,
}

/// Borrowed action-selection data. The observation is the actual predecessor
/// even if a candidate or its target has disappeared at the successor.
#[derive(Debug, Clone, Copy)]
pub struct FactualInput<'a> {
    pub observation: &'a Observation,
    pub reward: &'a Reward,
}

impl<'a> PreviousInput<'a> {
    pub fn from_total(action: &'a Action, reward: f32) -> Self {
        Self {
            action,
            reward,
            factual: None,
        }
    }

    pub fn from_transition(
        action: &'a Action,
        reward: &'a Reward,
        observation: &'a Observation,
    ) -> Self {
        Self {
            action,
            reward: reward.total(),
            factual: Some(FactualInput {
                observation,
                reward,
            }),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct HistoryInput<'a> {
    pub observation: &'a Observation,
    pub previous: Option<PreviousInput<'a>>,
}
#[derive(Debug, Clone)]
pub struct StatefulDecision {
    pub action: Action,
    pub log_prob: Option<f32>,
}

/// Complete public history for explicit mid-episode state reconstruction. The
/// caller supplies the actual episode tape; observations include the current one.
pub struct ReconstructionHistory<'a> {
    pub observations: &'a [Observation],
    pub actions: &'a [Action],
    pub rewards: HistoryRewards<'a>,
}

/// Dense histories may borrow their existing scalar tape. Structured histories
/// retain primitive traces so model inputs can recover the actual duration.
#[derive(Debug, Clone, Copy)]
pub enum HistoryRewards<'a> {
    Totals(&'a [f32]),
    Traces(&'a [Reward]),
}

impl<'a> HistoryRewards<'a> {
    pub fn len(self) -> usize {
        match self {
            Self::Totals(values) => values.len(),
            Self::Traces(values) => values.len(),
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub(crate) fn input(
        self,
        index: usize,
        action: &'a Action,
        observation: &'a Observation,
    ) -> PreviousInput<'a> {
        match self {
            Self::Totals(values) => PreviousInput::from_total(action, values[index]),
            Self::Traces(values) => {
                PreviousInput::from_transition(action, &values[index], observation)
            }
        }
    }

    pub(crate) fn slice(self, range: std::ops::Range<usize>) -> Self {
        match self {
            Self::Totals(values) => Self::Totals(&values[range]),
            Self::Traces(values) => Self::Traces(&values[range]),
        }
    }
}

/// Learning remains independent of the interaction/runtime storage choice. This
/// companion capability avoids pretending a recurrent actor is observation-only.
pub trait StatefulLearner {
    type Batch;
    fn update(&mut self, batch: &Self::Batch) -> Result<super::learner::UpdateReport>;
}

/// Optional, independently usable behavior construction. Settings belong to the
/// algorithm/policy, not the collector or the generic learning contract.
pub trait BehaviorPublisher {
    type Actor: StatefulActor + Clone;
    type Settings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor>;
}

/// Off-policy continuation semantics are a separate capability from learning.
pub trait TemporalOffPolicyLearner: StatefulLearner {
    fn return_contract(&self) -> super::temporal::ReturnSpec;
    /// Learners with an explicit schema can bind it before collection starts.
    fn replay_spaces(&self) -> Option<(&ObservationSpace, &ActionSpace)> {
        None
    }
    /// Whether a replay sample must carry its exact reconstruction prefix.
    fn requires_replay_history(&self) -> bool {
        true
    }
}

/// `assimilate` is deterministic and returns a candidate without modifying prior.
/// `decide` reads an already assimilated observation; it never advances memory.
/// Serialization is an optional capability imposed by snapshot users, not inference.
pub trait StatefulActor {
    type State: Clone;
    /// Optional validation of model-owned invariants across the run checkpoint.
    fn validate_learner_checkpoint(&self, _record: &LearnerRecord) -> Result<()> {
        Ok(())
    }

    /// The observation schema accepted during assimilation and decisions.
    fn observation_space(&self) -> ObservationSpace;
    /// Convenience width for dense observations.
    fn dense_observation_dim(&self) -> Option<usize> {
        self.observation_space().dense_width()
    }
    fn action_space(&self) -> &ActionSpace;
    fn revision(&self) -> u64;
    fn initial_state(&self) -> Result<Self::State>;
    fn validate_state(&self, state: &Self::State) -> Result<()>;
    /// Snapshot audit for state components tied to a particular public frame.
    fn validate_state_at(
        &self,
        state: &Self::State,
        _observation: &Observation,
        _preceding_observation: Option<&Observation>,
    ) -> Result<()> {
        self.validate_state(state)
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State>;
    /// Rebuild post-observation memory under this actor from a genuine reset
    /// and its chronological public inputs. Implementations may use an exact
    /// encoded path when it preserves the scalar assimilation arithmetic.
    fn reconstruct_history(&self, history: &[HistoryInput<'_>]) -> Result<Self::State> {
        validate_history_inputs(history)?;
        replay_actor_history(self, history.iter().copied())
    }
    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision>;

    /// Explicit opt-in to evaluating this actor's immutable model for another
    /// actor. Compatibility must include parameters, state/input schema, revision
    /// and behavior settings; a revision number alone is insufficient. Returning
    /// false retains the scalar path, including for arbitrary structured states.
    fn batch_compatible(&self, _other: &Self) -> bool {
        false
    }

    /// Independent candidates in input order. This default is a scalar fallback;
    /// implementations may combine neural work without sharing lane memories.
    fn assimilate_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        ensure(
            priors.len() == inputs.len(),
            "stateful batch input count mismatch",
        )?;
        priors
            .iter()
            .zip(inputs)
            .map(|(state, input)| self.assimilate(state, *input))
            .collect()
    }

    /// RNGs are caller-owned candidates, one stream per lane. Consumption for
    /// every lane must match scalar `decide`, independently of grouping/order.
    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "stateful decision batch count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .zip(rngs)
            .map(|((state, observation), rng)| self.decide(state, observation, mode, rng))
            .collect()
    }

    /// Explicit scalar-row numerical contract. Arbitrary actors retain a scalar
    /// fallback; neural implementations may share tensor work without changing bits.
    fn assimilate_canonical_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        ensure(
            priors.len() == inputs.len(),
            "canonical assimilation count mismatch",
        )?;
        priors
            .iter()
            .zip(inputs)
            .map(|(state, input)| self.assimilate(state, *input))
            .collect()
    }
    fn decide_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "canonical decision count mismatch",
        )?;
        states
            .iter()
            .zip(observations)
            .zip(rngs)
            .map(|((state, observation), rng)| self.decide(state, observation, mode, rng))
            .collect()
    }
}

pub(crate) fn validate_history_inputs(history: &[HistoryInput<'_>]) -> Result<()> {
    ensure(
        history
            .first()
            .is_some_and(|input| input.previous.is_none())
            && history[1..].iter().all(|input| input.previous.is_some()),
        "history must start at a genuine reset and include every previous input",
    )
}

pub(crate) fn replay_actor_history<'a, A: StatefulActor + ?Sized>(
    actor: &A,
    inputs: impl IntoIterator<Item = HistoryInput<'a>>,
) -> Result<A::State> {
    let mut state = actor.initial_state()?;
    for input in inputs {
        state = actor.assimilate(&state, input)?;
        actor.validate_state(&state)?;
    }
    Ok(state)
}

/// Existing feedforward actors can opt into temporal recording with unit memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatelessActor<A> {
    pub actor: A,
    pub width: usize,
    pub revision: u64,
}
impl<A: Actor> StatefulActor for StatelessActor<A> {
    type State = ();
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.width)
    }
    fn action_space(&self) -> &ActionSpace {
        self.actor.action_space()
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<()> {
        Ok(())
    }
    fn validate_state(&self, _: &()) -> Result<()> {
        Ok(())
    }
    fn assimilate(&self, _: &(), input: HistoryInput<'_>) -> Result<()> {
        input
            .observation
            .validate(self.width, self.action_space(), false)
    }
    fn decide(
        &self,
        _: &(),
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        observation.validate(self.width, self.action_space(), true)?;
        let refs = [observation];
        let mut actions = Vec::new();
        self.actor
            .act_into(ObservationBatch::from_refs(&refs), mode, rng, &mut actions)?;
        ensure(
            actions.len() == 1,
            "stateless adapter action count mismatch",
        )?;
        Ok(StatefulDecision {
            action: actions.remove(0),
            log_prob: None,
        })
    }
}
impl<A: Actor + super::checkpoint::Snapshot> super::checkpoint::Snapshot for StatelessActor<A> {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(self.width > 0, "stateless actor width must be positive")?;
        self.actor.action_space().validate()?;
        self.actor.validate_snapshot()
    }
}

/// One canonical neural history encoding, independent of environment or algorithm.
/// Ordering: observation, previous action (one-hot or continuous), previous reward,
/// previous-present mask. The first observation has all history channels zero.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistorySpec {
    pub observation_dim: usize,
    pub action_space: ActionSpace,
}
impl HistorySpec {
    pub fn new(observation_dim: usize, action_space: ActionSpace) -> Result<Self> {
        let s = Self {
            observation_dim,
            action_space,
        };
        s.validate()?;
        Ok(s)
    }
    pub fn validate(&self) -> Result<()> {
        self.action_space.validate()?;
        ensure(
            self.observation_dim > 0 && !matches!(self.action_space, ActionSpace::Conditional(_)),
            "numerical history requires a positive width and numerical actions",
        )?;
        self.observation_dim
            .checked_add(self.action_width())
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| crate::Error::Config("history width overflow".into()))?;
        Ok(())
    }
    pub fn action_width(&self) -> usize {
        match &self.action_space {
            ActionSpace::Discrete { count } => *count,
            s => s.dimensions(),
        }
    }
    pub fn input_dim(&self) -> usize {
        self.observation_dim + self.action_width() + 2
    }
    pub fn encode(&self, input: HistoryInput<'_>) -> Result<Vec<f32>> {
        self.validate()?;
        input
            .observation
            .validate(self.observation_dim, &self.action_space, false)?;
        let mut x = Vec::with_capacity(self.input_dim());
        x.extend_from_slice(input.observation.dense_values());
        x.resize(self.input_dim(), 0.);
        if let Some(previous) = input.previous {
            self.action_space.check(previous.action, None)?;
            ensure(previous.reward.is_finite(), "nonfinite previous reward")?;
            match previous.action {
                Action::Discrete(a) => x[self.observation_dim + *a] = 1.,
                Action::Continuous(a) => {
                    x[self.observation_dim..self.observation_dim + a.len()].copy_from_slice(a)
                }
                Action::Conditional(_) => unreachable!("validated numerical history action"),
            }
            let n = x.len();
            x[n - 2] = previous.reward;
            x[n - 1] = 1.;
        }
        Ok(x)
    }
}
