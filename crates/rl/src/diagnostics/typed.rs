//! Portable diagnostics for inputs without a dense numerical representation.

use super::{
    codec,
    evaluation::EvaluationHorizon,
    probes::ProbeSet,
    schema::{
        ActionCondition, Array, Chunk, Conditioning, EvaluationMetadata, ProbePopulation, Record,
        RecordContext, RecordKind, RunManifest,
    },
    validation,
    value::{
        self, ContinuationIdentity, CriticDeclaration, ValueConventions, ValueCriticKind,
        ValueDeclaration,
    },
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction, FiniteActionKey},
        ensure,
        env::{Observation, ObservationSpace, PublicHistory},
        reward::Reward,
        temporal::TemporalTransition,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MAX_TYPED_BYTES: usize = 4 * 1024 * 1024;
const MAX_OBSERVATION_BYTES: usize = 256 * 1024;
pub const MAX_TYPED_ROWS: usize = 4096;
pub const MAX_FINITE_ACTIONS: usize = 4096;
pub(crate) const MAX_PREDICTIONS: usize = 32_768;
const MAX_HISTORY_STEPS: u64 = 255;
const PAYLOAD: &str = "typed/payload";

/// The model input, recorded independently of any visualization coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum DiagnosticInput {
    Observation(Observation),
    History(PublicHistory),
    Projected {
        observation: Observation,
        projection: DisplayProjection,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisplayProjection {
    pub labels: Vec<String>,
    pub coordinates: Vec<f64>,
}

impl DiagnosticInput {
    pub fn observation(&self) -> &Observation {
        match self {
            Self::Observation(obs)
            | Self::Projected {
                observation: obs, ..
            } => obs,
            Self::History(history) => history.observation(),
        }
    }

    pub fn public_history(&self) -> Option<&PublicHistory> {
        match self {
            Self::History(history) => Some(history),
            _ => None,
        }
    }

    /// Exact model-input equality, independent of the display projection.
    pub fn same_input_as(&self, other: &Self) -> bool {
        match (self.public_history(), other.public_history()) {
            (None, None) => self.observation().matches(other.observation()),
            (Some(left), Some(right)) => {
                left.timestep() == right.timestep()
                    && left.inputs().into_iter().zip(right.inputs()).all(|(a, b)| {
                        a.observation.matches(b.observation)
                            && match (a.previous, b.previous) {
                                (None, None) => true,
                                (Some(a), Some(b)) => {
                                    a.action == b.action
                                        && a.factual
                                            .zip(b.factual)
                                            .is_some_and(|(a, b)| a.reward == b.reward)
                                }
                                _ => false,
                            }
                    })
            }
            _ => false,
        }
    }

    pub(crate) fn validate(&self, space: &ObservationSpace, actions: &ActionSpace) -> Result<()> {
        if let Self::Projected { projection, .. } = self {
            ensure(
                (1..=32).contains(&projection.labels.len())
                    && projection.labels.len() == projection.coordinates.len()
                    && projection.labels.iter().all(|label| valid_text(label, 128))
                    && projection.coordinates.iter().all(|x| x.is_finite()),
                "invalid diagnostic display coordinates",
            )?;
        }
        if let Self::History(history) = self {
            ensure(
                history.timestep() <= MAX_HISTORY_STEPS,
                "diagnostic public history exceeds the step limit",
            )?;
            for input in history.inputs() {
                check_observation_size(input.observation)?;
            }
            history.validate_in(space, actions)
        } else {
            check_observation_size(self.observation())?;
            self.observation().validate_in(space, actions, true)
        }
    }
}

impl From<Observation> for DiagnosticInput {
    fn from(observation: Observation) -> Self {
        Self::Observation(observation)
    }
}

pub(crate) fn check_observation_size(obs: &Observation) -> Result<usize> {
    let bytes = match obs {
        Observation::Dense(dense) => dense
            .values
            .len()
            .saturating_mul(size_of::<f32>())
            .saturating_add(dense.action_mask.as_ref().map_or(0, Vec::len)),
        Observation::Discrete(discrete) => discrete.action_mask.as_ref().map_or(0, Vec::len),
        Observation::Structured(frame) => frame.payload_bytes(false),
        Observation::Graph(frame) => frame.payload_bytes(false),
    };
    let bytes = bytes.saturating_add(size_of::<Observation>());
    ensure(
        bytes <= MAX_OBSERVATION_BYTES,
        "diagnostic typed observation exceeds the byte limit",
    )?;
    Ok(bytes)
}

/// Bound native materialization independently of the encoded JSON limit.
pub(super) struct TypedBudget(usize);

impl TypedBudget {
    pub(super) fn new(limit: usize) -> Self {
        Self(limit.min(MAX_TYPED_BYTES))
    }

    pub(super) fn charge(&mut self, bytes: usize) -> Result<()> {
        self.0 = self
            .0
            .checked_sub(bytes)
            .ok_or_else(|| Error::Config("diagnostic typed payload exceeds chunk budget".into()))?;
        Ok(())
    }

    pub(super) fn observation(&mut self, observation: &Observation) -> Result<()> {
        self.charge(check_observation_size(observation)?)
    }

    pub(super) fn action(&mut self, action: &Action) -> Result<()> {
        let bytes = match action {
            Action::Discrete(_) => 0,
            Action::Continuous(values) => size_of_val(values.as_slice()),
            Action::Conditional(_) => size_of::<ConditionalAction>(),
        };
        self.charge(bytes.saturating_add(size_of::<Action>()))
    }

    fn reward(&mut self, reward: &Reward) -> Result<()> {
        self.charge(size_of_val(reward.primitives()))
    }

    pub(super) fn input(&mut self, input: &DiagnosticInput) -> Result<()> {
        match input {
            DiagnosticInput::History(history) => {
                ensure(
                    history.timestep() <= MAX_HISTORY_STEPS,
                    "diagnostic public history exceeds the step limit",
                )?;
                for input in history.inputs() {
                    self.observation(input.observation)?;
                    if let Some(previous) = input.previous {
                        self.action(previous.action)?;
                        if let Some(factual) = previous.factual {
                            self.reward(factual.reward)?;
                        }
                    }
                }
            }
            DiagnosticInput::Projected {
                observation,
                projection,
            } => {
                self.observation(observation)?;
                self.charge(size_of_val(projection.coordinates.as_slice()))?;
                for label in &projection.labels {
                    self.charge(size_of::<String>().saturating_add(label.len()))?;
                }
            }
            DiagnosticInput::Observation(observation) => self.observation(observation)?,
        }
        self.charge(size_of::<DiagnosticInput>())
    }

    pub(super) fn step(&mut self, action: &Action) -> Result<()> {
        self.charge(size_of::<TypedStep>())?;
        self.action(action)
    }

    pub(super) fn evaluation(&mut self, metadata: &EvaluationMetadata) -> Result<()> {
        let description = match &metadata.conditioning {
            Conditioning::InitialScenario => 0,
            Conditioning::FullSimulatorState { description }
            | Conditioning::Observation {
                methodology: description,
            }
            | Conditioning::History {
                methodology: description,
            } => description.len(),
        };
        for bytes in [
            size_of::<EvaluationMetadata>(),
            description,
            metadata.continuation_policy.len(),
            metadata.seed_methodology.len(),
            metadata.truncation_handling.len(),
        ] {
            self.charge(bytes)?;
        }
        Ok(())
    }
}

fn valid_text(s: &str, max: usize) -> bool {
    !s.trim().is_empty() && s.len() <= max && !s.chars().any(char::is_control)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedProbes {
    pub population: ProbePopulation,
    pub methodology: String,
    pub row_ids: Vec<u64>,
    pub inputs: Vec<DiagnosticInput>,
    pub fixed_actions: Option<Vec<Action>>,
}

impl TypedProbes {
    pub(crate) fn validate_in(
        &self,
        space: &ObservationSpace,
        actions: &ActionSpace,
    ) -> Result<()> {
        ensure(
            (1..=MAX_TYPED_ROWS).contains(&self.inputs.len())
                && self.row_ids.len() == self.inputs.len()
                && self.row_ids.iter().collect::<BTreeSet<_>>().len() == self.row_ids.len()
                && valid_text(&self.methodology, 4096),
            "invalid typed probe values or identities",
        )?;
        let mut budget = TypedBudget::new(MAX_TYPED_BYTES);
        budget.charge(
            self.methodology
                .len()
                .saturating_add(size_of_val(self.row_ids.as_slice())),
        )?;
        for input in &self.inputs {
            budget.input(input)?;
            input.validate(space, actions)?;
        }
        if let Some(fixed) = &self.fixed_actions {
            ensure(
                fixed.len() == self.inputs.len(),
                "typed probe action count mismatch",
            )?;
            for (input, action) in self.inputs.iter().zip(fixed) {
                budget.action(action)?;
                input.observation().check_action(actions, action)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FiniteValueKind {
    ActionValue,
    PreferenceScore,
    OptimalActionValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteCandidate {
    pub action: FiniteActionKey,
    pub probability: Option<f64>,
    pub value: Option<f64>,
    pub visits: Option<u64>,
    pub updates: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FiniteDecision {
    pub row_id: u64,
    pub information_state: Option<String>,
    pub selected: Option<FiniteActionKey>,
    pub candidates: Vec<FiniteCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinitePredictions {
    pub value_kind: Option<FiniteValueKind>,
    pub rows: Vec<FiniteDecision>,
}

impl FinitePredictions {
    fn validate(&self) -> Result<()> {
        ensure(
            (1..=MAX_TYPED_ROWS).contains(&self.rows.len())
                && self
                    .rows
                    .iter()
                    .map(|row| row.candidates.len())
                    .sum::<usize>()
                    <= MAX_PREDICTIONS
                && self
                    .rows
                    .iter()
                    .map(|r| r.row_id)
                    .collect::<BTreeSet<_>>()
                    .len()
                    == self.rows.len(),
            "invalid finite prediction row identities",
        )?;
        for row in &self.rows {
            let keys: BTreeSet<_> = row.candidates.iter().map(|row| row.action).collect();
            ensure(
                (1..=MAX_FINITE_ACTIONS).contains(&row.candidates.len())
                    && keys.len() == row.candidates.len()
                    && row.selected.is_none_or(|key| keys.contains(&key))
                    && row
                        .information_state
                        .as_ref()
                        .is_none_or(|s| valid_text(s, 4096)),
                "invalid finite action identities or selected action",
            )?;
            let probabilities = row.candidates.iter().filter_map(|c| c.probability);
            let count = probabilities.clone().count();
            ensure(
                (count == 0
                    || count == row.candidates.len()
                        && probabilities.clone().all(|p| (0.0..=1.0).contains(&p))
                        && (probabilities.sum::<f64>() - 1.0).abs() <= 1e-5)
                    && row.candidates.iter().all(|c| {
                        c.value
                            .is_none_or(|value| self.value_kind.is_some() && value.is_finite())
                    }),
                "invalid finite prediction probabilities or values",
            )?;
        }
        Ok(())
    }

    /// Ensure each prediction row describes the same legal semantic actions as its input.
    pub fn validate_support<'a>(
        &self,
        row_ids: &[u64],
        observations: impl IntoIterator<Item = &'a Observation>,
        actions: &ActionSpace,
    ) -> Result<()> {
        self.validate()?;
        ensure(
            self.rows.len() == row_ids.len(),
            "finite prediction row count mismatch",
        )?;
        let mut seen = 0;
        for ((row, id), observation) in self.rows.iter().zip(row_ids).zip(observations) {
            let legal = legal_keys(observation, actions)?;
            ensure(
                row.row_id == *id
                    && legal.len() == row.candidates.len()
                    && row
                        .candidates
                        .iter()
                        .all(|candidate| legal.contains(&candidate.action)),
                "finite prediction support differs from the typed probe",
            )?;
            seen += 1;
        }
        ensure(seen == row_ids.len(), "missing typed probe observations")
    }

    pub fn record(
        &self,
        mut context: RecordContext,
        set: &ProbeSet,
        max_chunk_bytes: usize,
    ) -> Result<Chunk> {
        self.validate_support(set.row_ids(), set.observations(), set.action_space())?;
        context.probe_set = Some(set.id().into());
        TypedPayload::Predictions(self.clone()).record(context, max_chunk_bytes)
    }
}

pub(crate) fn legal_keys(
    obs: &Observation,
    actions: &ActionSpace,
) -> Result<BTreeSet<FiniteActionKey>> {
    let candidate_count = match actions {
        ActionSpace::Discrete { count } => *count,
        ActionSpace::Conditional(_) => obs
            .as_structured()
            .map_or(0, |frame| frame.candidates.len()),
        _ => 0,
    };
    ensure(
        (1..=MAX_FINITE_ACTIONS).contains(&candidate_count),
        "finite diagnostic action menu exceeds its limit",
    )?;
    let actions = obs.finite_actions(actions)?;
    let keys: BTreeSet<_> = actions
        .iter()
        .map(|action| obs.finite_action_key(action))
        .collect::<Result<_>>()?;
    ensure(
        keys.len() == actions.len(),
        "duplicate semantic action aliases",
    )?;
    Ok(keys)
}

pub(crate) fn declare_action_values(
    chunk: &mut Chunk,
    conventions: Option<&ValueConventions>,
    actor: Option<ContinuationIdentity>,
    continuation: Option<ContinuationIdentity>,
) -> Result<()> {
    let identity = &chunk.context.identity;
    value::attach_declaration(
        chunk,
        &ValueDeclaration {
            version: 1,
            run_id: identity.run_id.clone(),
            revision: identity.learner_update_revision,
            model_role: identity.model_role.clone(),
            conventions: conventions.cloned(),
            critics: BTreeMap::from([(
                "action_value".into(),
                CriticDeclaration {
                    kind: ValueCriticKind::DiscreteQ,
                    actor,
                    continuation,
                    twin: None,
                    utility: None,
                    learned_baseline: None,
                },
            )]),
        },
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEpisode {
    pub source: TypedEpisodeSource,
    pub observations: Vec<Observation>,
    pub steps: Vec<TypedStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TypedEpisodeSource {
    Evaluation {
        scenario_seed: u64,
        metadata: EvaluationMetadata,
        horizon: EvaluationHorizon,
    },
    Factual {
        episode_id: u64,
        discount: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedStep {
    pub action: Action,
    pub reward: f32,
    pub duration: u64,
    pub discounted_reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factual_reward: Option<Reward>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_revision: Option<u64>,
}

impl TypedEpisode {
    pub(crate) fn from_factual(
        transitions: &[TemporalTransition],
        discount: f64,
        recorded_revision: u64,
        limit: usize,
    ) -> Result<Self> {
        ensure(
            (1..=MAX_TYPED_ROWS).contains(&transitions.len()),
            "invalid factual diagnostic episode length",
        )?;
        let first = &transitions[0];
        let mut budget = TypedBudget::new(limit);
        budget.observation(&first.observation)?;
        for (index, transition) in transitions.iter().enumerate() {
            ensure(
                transition.episode_id == first.episode_id
                    && transition.timestep == index
                    && transition.behavior.revision <= recorded_revision
                    && (index == 0
                        || transition
                            .observation
                            .matches(&transitions[index - 1].next_observation)),
                "factual recording requires one contiguous episode from its reset",
            )?;
            budget.step(&transition.action)?;
            budget.reward(&transition.reward)?;
            budget.observation(&transition.next_observation)?;
        }
        validation::physical_durations(
            transitions
                .iter()
                .map(|transition| transition.reward.duration()),
        )?;
        let mut observations = Vec::with_capacity(transitions.len() + 1);
        observations.push(first.observation.clone());
        let mut steps = Vec::with_capacity(transitions.len());
        for transition in transitions {
            let (discounted_reward, _) = transition.reward.discounted_f64(discount)?;
            steps.push(TypedStep {
                action: transition.action.clone(),
                reward: transition.reward.total(),
                duration: transition.reward.duration(),
                discounted_reward,
                terminated: transition.terminated,
                truncated: transition.truncated,
                factual_reward: Some(transition.reward.clone()),
                behavior_revision: Some(transition.behavior.revision),
            });
            observations.push(transition.next_observation.clone());
        }
        Ok(Self {
            source: TypedEpisodeSource::Factual {
                episode_id: first.episode_id,
                discount,
            },
            observations,
            steps,
        })
    }

    fn validate_in(
        &self,
        space: &ObservationSpace,
        manifest: &RunManifest,
        revision: u64,
    ) -> Result<()> {
        let mut budget = TypedBudget::new(MAX_TYPED_BYTES);
        let (discount, factual) = match &self.source {
            TypedEpisodeSource::Evaluation {
                metadata, horizon, ..
            } => {
                validation::evaluation(metadata)?;
                budget.evaluation(metadata)?;
                ensure(
                    matches!(
                        metadata.initial_action,
                        ActionCondition::CurrentPolicy | ActionCondition::Fixed
                    ),
                    "invalid typed trajectory initial action condition",
                )?;
                let conditioning_valid = match &metadata.conditioning {
                    Conditioning::InitialScenario => true,
                    Conditioning::FullSimulatorState {
                        description: methodology,
                    }
                    | Conditioning::Observation { methodology }
                    | Conditioning::History { methodology } => !methodology.trim().is_empty(),
                };
                ensure(
                    conditioning_valid,
                    "empty typed trajectory conditioning description",
                )?;
                let decisions = self.steps.len();
                if let EvaluationHorizon::FiniteDecisions(limit) = horizon {
                    validation::finite_initial_evaluation(metadata)?;
                    ensure(
                        *limit > 0 && decisions <= *limit,
                        "typed trajectory exceeds its finite decision objective",
                    )?;
                }
                ensure(
                    metadata.complete_returns
                        == horizon.completes(
                            decisions,
                            self.steps.last().is_some_and(|step| step.terminated),
                        ),
                    "typed trajectory completeness differs from its declared objective",
                )?;
                (metadata.discount, false)
            }
            TypedEpisodeSource::Factual { discount, .. } => (*discount, true),
        };
        ensure(
            (1..=MAX_TYPED_ROWS).contains(&self.steps.len())
                && self.observations.len() == self.steps.len() + 1
                && discount == manifest.discount,
            "invalid typed episode shape or discount",
        )?;
        validation::physical_durations(self.steps.iter().map(|step| step.duration))?;
        for (index, step) in self.steps.iter().enumerate() {
            let final_step = index + 1 == self.steps.len();
            ensure(
                step.reward.is_finite()
                    && step.discounted_reward.is_finite()
                    && !(step.terminated && step.truncated)
                    && (final_step || !(step.terminated || step.truncated)),
                "invalid typed episode step",
            )?;
            budget.step(&step.action)?;
            match (&step.factual_reward, step.behavior_revision, factual) {
                (Some(reward), Some(behavior_revision), true) => {
                    budget.reward(reward)?;
                    let (discounted, _) = reward.discounted_f64(discount)?;
                    ensure(
                        behavior_revision <= revision
                            && step.reward == reward.total()
                            && step.duration == reward.duration()
                            && step.discounted_reward == discounted,
                        "factual episode reward provenance differs",
                    )?;
                }
                (None, None, false) => {}
                _ => {
                    return Err(Error::Shape(
                        "typed episode decision provenance differs".into(),
                    ));
                }
            }
            let obs = &self.observations[index];
            budget.observation(obs)?;
            obs.validate_in(space, &manifest.action_space, true)?;
            obs.validate_event_history(
                index.checked_sub(1).map(|prior| &self.observations[prior]),
            )?;
            obs.check_action(&manifest.action_space, &step.action)?;
        }
        let last = self.observations.last().expect("validated episode shape");
        budget.observation(last)?;
        last.validate_event_history(self.observations.get(self.steps.len() - 1))?;
        last.validate_in(
            space,
            &manifest.action_space,
            !self.steps.last().is_some_and(|step| step.terminated),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedKind {
    Probes,
    Predictions,
    Episode,
}

impl TypedKind {
    pub(super) fn record_kind(self) -> RecordKind {
        match self {
            Self::Probes => RecordKind::TypedProbes,
            Self::Predictions => RecordKind::FinitePredictions,
            Self::Episode => RecordKind::TypedEpisode,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedRecord {
    pub kind: TypedKind,
    pub payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum TypedPayload {
    Probes(TypedProbes),
    Predictions(FinitePredictions),
    Episode(TypedEpisode),
}

impl TypedPayload {
    fn kind(&self) -> TypedKind {
        match self {
            Self::Probes(_) => TypedKind::Probes,
            Self::Predictions(_) => TypedKind::Predictions,
            Self::Episode(_) => TypedKind::Episode,
        }
    }

    pub(crate) fn record(&self, context: RecordContext, limit: usize) -> Result<Chunk> {
        let bytes = codec::bounded_json(self, limit.min(MAX_TYPED_BYTES))?;
        Ok(Chunk {
            record: Record::Typed(TypedRecord {
                kind: self.kind(),
                payload: PAYLOAD.into(),
            }),
            context,
            arrays: BTreeMap::from([(PAYLOAD.into(), Array::u8(vec![bytes.len()], bytes)?)]),
        })
    }

    pub fn read(chunk: &Chunk) -> Result<Self> {
        let Record::Typed(record) = &chunk.record else {
            return Err(Error::Shape("expected typed diagnostics".into()));
        };
        let array = chunk
            .arrays
            .get(&record.payload)
            .ok_or_else(|| Error::Shape("missing typed diagnostics payload".into()))?;
        let bytes = array.as_u8()?;
        ensure(
            !bytes.is_empty() && bytes.len() <= MAX_TYPED_BYTES && array.shape() == [bytes.len()],
            "typed payload exceeds the diagnostic byte limit",
        )?;
        let payload: Self = serde_json::from_slice(bytes)?;
        ensure(
            payload.kind() == record.kind,
            "typed payload identity mismatch",
        )?;
        Ok(payload)
    }
}

pub(super) fn validate_in(chunk: &Chunk, manifest: &RunManifest) -> Result<()> {
    if !matches!(chunk.record, Record::Typed(_)) {
        return Ok(());
    }
    match TypedPayload::read(chunk)? {
        TypedPayload::Probes(probes) => {
            probes.validate_in(&manifest.observation_space(), &manifest.action_space)
        }
        TypedPayload::Episode(episode) => episode.validate_in(
            &manifest.observation_space(),
            manifest,
            chunk.context.identity.learner_update_revision,
        ),
        TypedPayload::Predictions(predictions) => {
            predictions.validate()?;
            ensure(
                predictions
                    .rows
                    .iter()
                    .flat_map(|row| &row.candidates)
                    .all(
                        |candidate| match (&manifest.action_space, candidate.action) {
                            (ActionSpace::Discrete { count }, FiniteActionKey::Discrete(index)) => {
                                index < *count
                            }
                            (
                                ActionSpace::Conditional(grammar),
                                FiniteActionKey::Conditional(key),
                            ) => {
                                grammar.accepts(&key)
                                    && !matches!(
                                        key.argument,
                                        Some(ActionArgument::Location { .. })
                                    )
                            }
                            _ => false,
                        },
                    ),
                "finite diagnostic action key differs from the manifest grammar",
            )
        }
    }
}
