//! Finite-action statistical inference and typed policy checkpoints.

use super::{
    FeatureMap, FeatureSpec, checked_features,
    regression::{LogisticModel, Predictor},
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace, PublicHistory},
        reward::BatchObjective,
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        temporal::TemporalSpaces,
    },
    planning::guidance::{PolicyGuide, validate_prior},
};
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned};

fn action_order(a: &Action, b: &Action) -> std::cmp::Ordering {
    match (a, b) {
        (Action::Discrete(a), Action::Discrete(b)) => a.cmp(b),
        (Action::Conditional(a), Action::Conditional(b)) => a.cmp(b),
        // A checked action list belongs to exactly one finite space.
        _ => unreachable!("mixed finite action spaces"),
    }
}

pub(crate) fn greedy_index(actions: &[Action], values: &[f64]) -> Result<usize> {
    ensure(
        !actions.is_empty()
            && actions.len() == values.len()
            && values.iter().all(|q| q.is_finite()),
        "invalid statistical action values",
    )?;
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best]
            || (values[i] == values[best] && action_order(&actions[i], &actions[best]).is_lt())
        {
            best = i;
        }
    }
    Ok(best)
}

/// A predictor bound to feature order, observation/action schemas and the
/// fitted reward objective. All bindings must match when reusing coefficients.
#[derive(Debug, Clone, Serialize)]
pub struct FittedQ<M> {
    feature_spec: FeatureSpec,
    spaces: TemporalSpaces,
    objective: BatchObjective,
    predictor: M,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FittedWire<M> {
    feature_spec: FeatureSpec,
    spaces: TemporalSpaces,
    objective: BatchObjective,
    predictor: M,
}

impl<'de, M: Predictor + Deserialize<'de>> Deserialize<'de> for FittedQ<M> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = FittedWire::deserialize(deserializer)?;
        Self::new(
            wire.feature_spec,
            wire.spaces,
            wire.objective,
            wire.predictor,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<M: Predictor> FittedQ<M> {
    pub fn new(
        feature_spec: FeatureSpec,
        spaces: TemporalSpaces,
        objective: BatchObjective,
        predictor: M,
    ) -> Result<Self> {
        let fitted = Self {
            feature_spec,
            spaces,
            objective,
            predictor,
        };
        fitted.validate()?;
        Ok(fitted)
    }

    pub fn feature_spec(&self) -> &FeatureSpec {
        &self.feature_spec
    }

    pub fn spaces(&self) -> &TemporalSpaces {
        &self.spaces
    }

    pub fn objective(&self) -> BatchObjective {
        self.objective
    }

    pub fn predictor(&self) -> &M {
        &self.predictor
    }

    pub fn validate(&self) -> Result<()> {
        self.feature_spec.validate()?;
        self.spaces.observations.validate()?;
        self.spaces.actions.validate()?;
        self.objective.validate()?;
        self.predictor.validate()?;
        ensure(
            self.feature_spec.dimension() == self.predictor.dimension()
                && matches!(
                    self.spaces.actions,
                    ActionSpace::Discrete { .. } | ActionSpace::Conditional(_)
                )
                && (!matches!(self.spaces.actions, ActionSpace::Conditional(_))
                    || matches!(
                        self.spaces.observations,
                        ObservationSpace::Structured(_) | ObservationSpace::Graph(_)
                    )),
            "fitted predictor width or action space mismatch",
        )
    }
}

impl FittedQ<LogisticModel> {
    /// Convert a declared undiscounted terminal-success prediction into
    /// signed terminal-return units.
    pub fn signed_terminal_return(&self, features: &[f64]) -> Result<f64> {
        ensure(
            matches!(self.objective, BatchObjective::TerminalSuccess { .. }),
            "signed probability conversion requires an undiscounted terminal-success objective",
        )?;
        Ok(2. * self.predictor.predict(features)? - 1.)
    }
}

/// Q-based actor over a causal public tape. Deployment chooses a single
/// stable semantic argmax. Sample mode shares exploitation mass among all
/// exact maximizers and exploration mass among all legal complete actions.
#[derive(Debug, Clone, Serialize)]
pub struct StatisticalActor<F, M> {
    features: F,
    fitted: FittedQ<M>,
    revision: u64,
    epsilon: f64,
}

#[derive(Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "F: Deserialize<'de>, M: Predictor + Deserialize<'de>")
)]
struct ActorWire<F, M> {
    features: F,
    fitted: FittedQ<M>,
    revision: u64,
    epsilon: f64,
}

impl<'de, F, M> Deserialize<'de> for StatisticalActor<F, M>
where
    F: FeatureMap + Deserialize<'de>,
    M: Predictor + Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let wire = ActorWire::deserialize(deserializer)?;
        Self::new(wire.features, wire.fitted, wire.revision, wire.epsilon)
            .map_err(serde::de::Error::custom)
    }
}

impl<F: FeatureMap, M: Predictor> StatisticalActor<F, M> {
    pub fn new(features: F, fitted: FittedQ<M>, revision: u64, epsilon: f64) -> Result<Self> {
        let actor = Self {
            features,
            fitted,
            revision,
            epsilon,
        };
        actor.validate()?;
        Ok(actor)
    }

    pub fn model(&self) -> &FittedQ<M> {
        &self.fitted
    }

    /// The frozen feature representation used by this actor.
    pub fn feature_map(&self) -> &F {
        &self.features
    }

    /// Exploration mass in the sample-mode action distribution.
    pub fn sampling_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn action_values(&self, history: &PublicHistory) -> Result<Vec<(Action, f64)>> {
        let actions = self.support(history.observation())?;
        let values = self.values_on(history, &actions)?;
        Ok(actions.into_iter().zip(values).collect())
    }

    pub fn greedy(&self, history: &PublicHistory) -> Result<Action> {
        let actions = self.support(history.observation())?;
        let values = self.values_on(history, &actions)?;
        Ok(actions[greedy_index(&actions, &values)?].clone())
    }

    pub fn epsilon_greedy(&self, history: &PublicHistory) -> Result<Vec<(Action, f64)>> {
        let actions = self.support(history.observation())?;
        let values = self.values_on(history, &actions)?;
        let probabilities = self.probabilities(&actions, &values)?;
        Ok(actions.into_iter().zip(probabilities).collect())
    }

    fn support(&self, observation: &Observation) -> Result<Vec<Action>> {
        observation.validate_in(
            &self.fitted.spaces.observations,
            &self.fitted.spaces.actions,
            true,
        )?;
        observation.finite_actions(&self.fitted.spaces.actions)
    }

    fn values_on(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<f64>> {
        checked_features(&self.features, history, actions)?
            .into_iter()
            .map(|row| {
                let value = self.fitted.predictor.predict(&row)?;
                ensure(value.is_finite(), "nonfinite statistical action value")?;
                Ok(value)
            })
            .collect()
    }

    pub(crate) fn probabilities(&self, actions: &[Action], values: &[f64]) -> Result<Vec<f64>> {
        let maximum = values[greedy_index(actions, values)?];
        let maximizers = values.iter().filter(|&&q| q == maximum).count();
        let exploration = self.epsilon / actions.len() as f64;
        let exploitation = (1. - self.epsilon) / maximizers as f64;
        Ok(values
            .iter()
            .map(|&q| exploration + if q == maximum { exploitation } else { 0. })
            .collect())
    }

    fn validate(&self) -> Result<()> {
        self.fitted.validate()?;
        self.features.validate_spaces(&self.fitted.spaces)?;
        ensure(
            self.features.spec() == &self.fitted.feature_spec
                && (0.0..=1.0).contains(&self.epsilon),
            "statistical actor feature identity or epsilon mismatch",
        )
    }
}

impl<F: FeatureMap, M: Predictor> StatefulActor for StatisticalActor<F, M> {
    type State = Option<PublicHistory>;

    fn observation_space(&self) -> ObservationSpace {
        self.fitted.spaces.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.fitted.spaces.actions
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn initial_state(&self) -> Result<Self::State> {
        Ok(None)
    }

    fn validate_state(&self, state: &Self::State) -> Result<()> {
        if let Some(history) = state {
            history.observation().validate_in(
                &self.fitted.spaces.observations,
                self.action_space(),
                false,
            )?;
        }
        Ok(())
    }

    fn validate_state_at(
        &self,
        state: &Self::State,
        observation: &Observation,
        preceding_observation: Option<&Observation>,
    ) -> Result<()> {
        let history = state
            .as_ref()
            .ok_or_else(|| Error::State("statistical state is uninitialized".into()))?;
        ensure(
            history.observation().matches(observation)
                && preceding_observation.is_none_or(|preceding| {
                    history
                        .previous_input()
                        .and_then(|input| input.factual)
                        .is_some_and(|input| input.observation.matches(preceding))
                }),
            "statistical checkpoint history differs from its public boundary",
        )?;
        let inputs = history.inputs();
        for (index, frame) in inputs.iter().enumerate() {
            frame.observation.validate_in(
                &self.fitted.spaces.observations,
                self.action_space(),
                index + 1 < inputs.len(),
            )?;
            if let Some(previous) = frame.previous {
                inputs[index - 1]
                    .observation
                    .check_action(self.action_space(), previous.action)?;
            }
        }
        Ok(())
    }

    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        input.observation.validate_in(
            &self.fitted.spaces.observations,
            self.action_space(),
            input.previous.is_none(),
        )?;
        let history = match (prior, input.previous) {
            (None, None) => {
                input.observation.validate_event_history(None)?;
                PublicHistory::new(input.observation.clone())
            }
            (Some(history), Some(previous)) => {
                history
                    .observation()
                    .check_action(self.action_space(), previous.action)?;
                let factual = previous.factual.ok_or_else(|| {
                    Error::State("statistical history requires the factual reward trace".into())
                })?;
                ensure(
                    history.observation().matches(factual.observation)
                        && previous.reward.to_bits() == factual.reward.total().to_bits(),
                    "statistical history predecessor or reward differs",
                )?;
                history.append(
                    previous.action.clone(),
                    factual.reward.clone(),
                    input.observation.clone(),
                )?
            }
            _ => {
                return Err(Error::State(
                    "statistical history requires a genuine reset and causal inputs".into(),
                ));
            }
        };
        Ok(Some(history))
    }

    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let history = state
            .as_ref()
            .ok_or_else(|| Error::State("statistical state is uninitialized".into()))?;
        ensure(
            history.observation().matches(observation),
            "statistical decision history differs from observation",
        )?;
        let actions = self.support(observation)?;
        let values = self.values_on(history, &actions)?;
        let (choice, log_prob) = match mode {
            ActMode::Deterministic => (greedy_index(&actions, &values)?, None),
            ActMode::Sample => {
                let probabilities = self.probabilities(&actions, &values)?;
                let choice = WeightedIndex::new(&probabilities)
                    .map_err(|e| Error::Numerical(e.to_string()))?
                    .sample(rng);
                (choice, Some(probabilities[choice].ln() as f32))
            }
        };
        Ok(StatefulDecision {
            action: actions[choice].clone(),
            log_prob,
        })
    }
}

/// Deterministic executable continuation over the planner's supplied feasible
/// actions. The predictive context must expose the corresponding public tape.
impl<F: FeatureMap, M: Predictor> PolicyGuide<PublicHistory, Action> for StatisticalActor<F, M> {
    fn prior(&self, history: &PublicHistory, feasible: &[Action]) -> Result<Vec<f64>> {
        let observation = history.observation();
        observation.validate_in(&self.fitted.spaces.observations, self.action_space(), true)?;
        for action in feasible {
            observation.check_action(self.action_space(), action)?;
        }
        let values = self.values_on(history, feasible)?;
        let mut prior = vec![0.; feasible.len()];
        prior[greedy_index(feasible, &values)?] = 1.;
        validate_prior(feasible, &prior)?;
        Ok(prior)
    }
}

impl<F: FeatureMap, M: Predictor> PolicyGuide<PublicHistory, ConditionalAction>
    for StatisticalActor<F, M>
{
    fn prior(&self, history: &PublicHistory, feasible: &[ConditionalAction]) -> Result<Vec<f64>> {
        let actions = feasible
            .iter()
            .map(|key| Action::Conditional((*key).into()))
            .collect::<Vec<_>>();
        self.prior(history, &actions)
    }
}

impl<F: FeatureMap, M: Predictor> PolicyGuide<PublicHistory, usize> for StatisticalActor<F, M> {
    fn prior(&self, history: &PublicHistory, feasible: &[usize]) -> Result<Vec<f64>> {
        let actions = feasible
            .iter()
            .copied()
            .map(Action::Discrete)
            .collect::<Vec<_>>();
        self.prior(history, &actions)
    }
}

const CHECKPOINT: &str = "statistical-action-value-v1";

#[derive(Serialize, Deserialize)]
#[serde(
    deny_unknown_fields,
    bound(deserialize = "M: Predictor + Deserialize<'de>")
)]
struct ActorRecord<M> {
    fitted: FittedQ<M>,
    epsilon: f64,
    revision: u64,
}

fn read_record<M: Predictor + DeserializeOwned>(record: &LearnerRecord) -> Result<ActorRecord<M>> {
    record.check(CHECKPOINT)?;
    ensure(
        record.models.is_empty()
            && record.optimizers.is_empty()
            && record.temporal_returns.is_none()
            && record.temporal_reconstruction.is_none()
            && record.replay_requirements.is_none(),
        "unexpected statistical checkpoint attachments",
    )?;
    let state: ActorRecord<M> = serde_json::from_value(record.metadata.clone())?;
    ensure(
        record.temporal_spaces.as_ref() == Some(&state.fitted.spaces),
        "statistical checkpoint schema mismatch",
    )?;
    Ok(state)
}

impl<F: FeatureMap, M: Predictor + DeserializeOwned> StatisticalActor<F, M> {
    /// Restore against a caller-supplied immutable map or frozen encoder.
    pub fn restore_with_features(features: F, record: &LearnerRecord) -> Result<Self> {
        let state = read_record(record)?;
        Self::new(features, state.fitted, state.revision, state.epsilon)
    }
}

impl<F: FeatureMap, M: Predictor + Serialize + DeserializeOwned> Checkpoint
    for StatisticalActor<F, M>
{
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.validate()?;
        let mut record = LearnerRecord::new(
            CHECKPOINT,
            ActorRecord {
                fitted: self.fitted.clone(),
                epsilon: self.epsilon,
                revision: self.revision,
            },
        )?;
        record.temporal_spaces = Some(self.fitted.spaces.clone());
        Ok(record)
    }

    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        let state: ActorRecord<M> = read_record(record)?;
        state.fitted.validate()?;
        ensure(
            self.features.spec() == &state.fitted.feature_spec
                && self.fitted.objective == state.fitted.objective
                && self.fitted.spaces == state.fitted.spaces
                && (0.0..=1.0).contains(&state.epsilon),
            "statistical checkpoint feature, objective, or schema mismatch",
        )?;
        self.fitted = state.fitted;
        self.epsilon = state.epsilon;
        self.revision = state.revision;
        Ok(())
    }
}

impl<F, M> Snapshot for StatisticalActor<F, M>
where
    F: FeatureMap + Serialize + DeserializeOwned,
    M: Predictor + Serialize + DeserializeOwned,
{
    fn validate_snapshot(&self) -> Result<()> {
        self.validate()
    }
}
