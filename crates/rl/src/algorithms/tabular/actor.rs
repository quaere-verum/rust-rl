use std::sync::Arc;

use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};

use super::QTable;
use crate::{
    Result,
    core::{
        action::{ActionArgument, ActionSpace, FiniteActionKey},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        state_key::StateKeyBuilder,
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        temporal::TemporalSpaces,
    },
};

/// An executable policy over an immutable table and a public key builder.
///
/// When binding a standalone table, supply its original key mapping and
/// observation and action schemas. A run snapshot carries this binding when
/// the builder and keys support Serde. The collector owns exploration RNG state.
#[derive(Clone)]
pub struct TabularActor<B: StateKeyBuilder> {
    pub(crate) keys: B,
    pub(crate) table: Arc<QTable<B::Key, FiniteActionKey>>,
    spaces: TemporalSpaces,
    revision: u64,
    pub(crate) epsilon: f64,
}

impl<B: StateKeyBuilder> TabularActor<B> {
    pub fn new(
        keys: B,
        table: QTable<B::Key, FiniteActionKey>,
        spaces: TemporalSpaces,
        revision: u64,
        epsilon: f64,
    ) -> Result<Self> {
        let actor = Self {
            keys,
            table: Arc::new(table),
            spaces,
            revision,
            epsilon,
        };
        actor.validate_binding()?;
        Ok(actor)
    }

    fn validate_binding(&self) -> Result<()> {
        self.spaces.observations.validate()?;
        self.spaces.actions.validate()?;
        ensure(
            matches!(
                self.spaces.actions,
                ActionSpace::Discrete { .. } | ActionSpace::Conditional(_)
            ),
            "tabular actors need a finite action space",
        )?;
        ensure(
            (0.0..=1.0).contains(&self.epsilon),
            "invalid tabular exploration",
        )?;
        let matches_space = |key: &FiniteActionKey| match (&self.spaces.actions, key) {
            (ActionSpace::Discrete { count }, FiniteActionKey::Discrete(action)) => action < count,
            (ActionSpace::Conditional(grammar), FiniteActionKey::Conditional(action)) => {
                grammar.accepts(action)
                    && !matches!(action.argument, Some(ActionArgument::Location { .. }))
            }
            _ => false,
        };
        ensure(
            self.table
                .states
                .values()
                .all(|row| row.keys().all(matches_space)),
            "tabular action key disagrees with the bound action space",
        )
    }

    fn validate_initial_memory(&self) -> Result<()> {
        self.keys.validate_memory(&self.keys.initial_memory()?)
    }
}

const ACTOR_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord<B, S> {
    version: u32,
    keys: B,
    table: LearnerRecord,
    spaces: S,
    revision: u64,
    epsilon: f64,
}

impl<B> Serialize for TabularActor<B>
where
    B: StateKeyBuilder + Serialize,
    B::Key: Serialize + DeserializeOwned,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        ActorRecord {
            version: ACTOR_VERSION,
            keys: &self.keys,
            table: self.table.checkpoint().map_err(serde::ser::Error::custom)?,
            spaces: &self.spaces,
            revision: self.revision,
            epsilon: self.epsilon,
        }
        .serialize(serializer)
    }
}

impl<'de, B> Deserialize<'de> for TabularActor<B>
where
    B: StateKeyBuilder + Deserialize<'de>,
    B::Key: Serialize + DeserializeOwned,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let record = ActorRecord::<B, TemporalSpaces>::deserialize(deserializer)?;
        if record.version != ACTOR_VERSION {
            return Err(serde::de::Error::custom(
                "unsupported tabular actor snapshot version",
            ));
        }
        let mut table = QTable::new(0.0).map_err(serde::de::Error::custom)?;
        table
            .restore(&record.table)
            .map_err(serde::de::Error::custom)?;
        let actor = Self::new(
            record.keys,
            table,
            record.spaces,
            record.revision,
            record.epsilon,
        )
        .map_err(serde::de::Error::custom)?;
        actor
            .validate_initial_memory()
            .map_err(serde::de::Error::custom)?;
        Ok(actor)
    }
}

impl<B> Snapshot for TabularActor<B>
where
    B: StateKeyBuilder + Serialize + DeserializeOwned,
    B::Key: Serialize + DeserializeOwned,
{
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_binding()?;
        self.validate_initial_memory()
    }
}

impl<B: StateKeyBuilder> StatefulActor for TabularActor<B> {
    type State = B::Memory;

    fn observation_space(&self) -> ObservationSpace {
        self.spaces.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.spaces.actions
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn initial_state(&self) -> Result<Self::State> {
        self.keys.initial_memory()
    }

    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.keys.validate_memory(state)
    }

    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        input.observation.validate_in(
            &self.spaces.observations,
            &self.spaces.actions,
            input.previous.is_none(),
        )?;
        self.keys.observe(prior, input)
    }

    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        observation.validate_in(&self.spaces.observations, &self.spaces.actions, true)?;
        let state_key = self.keys.key(state)?;
        let actions = observation.finite_actions(&self.spaces.actions)?;
        let action_keys = actions
            .iter()
            .map(|action| observation.finite_action_key(action))
            .collect::<Result<Vec<_>>>()?;
        let (index, log_prob) = match mode {
            ActMode::Deterministic => (self.table.greedy_index(&state_key, &action_keys)?, None),
            ActMode::Sample => {
                let choice =
                    self.table
                        .epsilon_greedy(&state_key, &action_keys, self.epsilon, rng)?;
                (choice.index, Some(choice.probability.ln() as f32))
            }
        };
        Ok(StatefulDecision {
            action: actions[index].clone(),
            log_prob,
        })
    }
}
