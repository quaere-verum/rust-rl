//! Explicit sparse tabular checkpoints on the shared diagnostic timeline.

use super::{
    codec,
    schema::{Array, Chunk, ModelArtifact, ModelSnapshotRecord, Record, RecordContext},
    session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
    typed::{
        DiagnosticInput, FiniteCandidate, FiniteDecision, FinitePredictions, FiniteValueKind,
        MAX_PREDICTIONS, declare_action_values, legal_keys,
    },
    value::ContinuationIdentity,
};
use crate::{
    Error, Result,
    algorithms::tabular::TabularActor,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        checkpoint::Snapshot,
        ensure,
        env::{Observation, ObservationBatch},
        state_key::{DiscreteSymbolKey, StateKeyBuilder},
        stateful::{HistoryInput, StatefulActor},
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;

const ARCHITECTURE: &str = "sparse-tabular.v1";

#[derive(Debug, Clone, Copy)]
pub enum TabularMethod {
    QLearning,
    Sarsa,
    MonteCarlo,
}

pub struct TabularDiagnostics<'a, B: StateKeyBuilder> {
    actor: &'a TabularActor<B>,
    method: TabularMethod,
    discount: f64,
    observation_sufficient: bool,
}

impl<'a, B: StateKeyBuilder> TabularDiagnostics<'a, B> {
    /// Every prediction requires a factual history for reconstruction.
    pub fn history(actor: &'a TabularActor<B>, method: TabularMethod, discount: f64) -> Self {
        Self {
            actor,
            method,
            discount,
            observation_sufficient: false,
        }
    }
}

impl<'a> TabularDiagnostics<'a, DiscreteSymbolKey> {
    pub fn discrete(
        actor: &'a TabularActor<DiscreteSymbolKey>,
        method: TabularMethod,
        discount: f64,
    ) -> Self {
        Self {
            actor,
            method,
            discount,
            observation_sufficient: true,
        }
    }
}

impl<B> DiagnosticLearner for TabularDiagnostics<'_, B>
where
    B: StateKeyBuilder + Serialize + DeserializeOwned,
    B::Key: Serialize + DeserializeOwned,
{
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: match self.method {
                TabularMethod::QLearning => "Tabular Q-learning",
                TabularMethod::Sarsa => "Tabular SARSA",
                TabularMethod::MonteCarlo => "Tabular Monte Carlo",
            },
            model: ARCHITECTURE,
            configuration: serde_json::json!({
                "key_builder": std::any::type_name::<B>(),
                "key_kind": format!("{:?}", self.actor.keys.kind()),
                "observation_sufficient": self.observation_sufficient,
                "epsilon": self.actor.epsilon,
            }),
            discount: Some(self.discount),
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.actor.revision()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        ensure(
            self.observation_sufficient,
            "history-dependent tabular evaluation requires a stateful runner",
        )?;
        Ok(Box::new(ObservationPolicy(self.actor)))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        if request.predictions {
            for set in request.probes {
                let Some(inputs) = set.typed_inputs() else {
                    continue;
                };
                let mut count = 0;
                for input in inputs {
                    count += legal_keys(input.observation(), self.actor.action_space())?.len();
                    ensure(
                        count <= MAX_PREDICTIONS,
                        "tabular diagnostic prediction count exceeds its limit",
                    )?;
                }
                let rows = set
                    .row_ids()
                    .iter()
                    .zip(inputs)
                    .map(|(&id, input)| {
                        ensure(
                            self.observation_sufficient || input.public_history().is_some(),
                            "tabular input is missing its required factual history",
                        )?;
                        decision(self.actor, id, input, request.evaluation_mode)
                    })
                    .collect::<Result<_>>()?;
                let mut chunk = FinitePredictions {
                    value_kind: Some(FiniteValueKind::ActionValue),
                    rows,
                }
                .record(request.context.clone(), set, request.max_chunk_bytes)?;
                let continuation =
                    matches!(self.method, TabularMethod::QLearning).then(|| ContinuationIdentity {
                        identity: "optimality".into(),
                        stochastic: false,
                    });
                declare_action_values(&mut chunk, request.conventions, None, continuation)?;
                chunks.push(chunk);
            }
        }
        if request.snapshots {
            chunks.push(snapshot(
                self.actor,
                request.context.clone(),
                request.max_chunk_bytes,
            )?);
        }
        Ok(chunks)
    }
}

fn current_state<B: StateKeyBuilder>(
    actor: &TabularActor<B>,
    observation: &Observation,
) -> Result<B::Memory> {
    actor.assimilate(
        &actor.initial_state()?,
        HistoryInput {
            observation,
            previous: None,
        },
    )
}

struct ObservationPolicy<'a, B: StateKeyBuilder>(&'a TabularActor<B>);
impl<B: StateKeyBuilder> Actor for ObservationPolicy<'_, B> {
    fn action_space(&self) -> &ActionSpace {
        self.0.action_space()
    }
    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            let state = current_state(self.0, observation)?;
            actions.push(self.0.decide(&state, observation, mode, rng)?.action);
        }
        Ok(())
    }
}

/// Read a portable sparse snapshot with the same application-owned key builder.
pub fn restore<B>(chunk: &Chunk) -> Result<TabularActor<B>>
where
    B: StateKeyBuilder + Serialize + DeserializeOwned,
    B::Key: Serialize + DeserializeOwned,
{
    let Record::ModelSnapshot(model) = &chunk.record else {
        return Err(Error::State("expected a tabular model snapshot".into()));
    };
    ensure(
        model.architecture == ARCHITECTURE,
        "wrong snapshot architecture",
    )?;
    let artifact = model
        .models
        .get("table")
        .ok_or_else(|| Error::State("tabular model snapshot has no table".into()))?;
    ensure(
        artifact.signature == TabularActor::<B>::record_type(),
        "tabular state-key mapping differs",
    )?;
    let bytes = chunk
        .arrays
        .get(&artifact.bytes)
        .ok_or_else(|| Error::State("tabular model snapshot is missing its payload".into()))?
        .as_u8()?;
    ensure(
        bytes.len() <= super::typed::MAX_TYPED_BYTES,
        "tabular snapshot exceeds its byte limit",
    )?;
    let actor: TabularActor<B> = serde_json::from_slice(bytes)?;
    actor.validate_snapshot()?;
    ensure(
        actor.revision() == chunk.context.identity.learner_update_revision,
        "tabular model revision differs",
    )?;
    Ok(actor)
}

pub fn tabular_decision<B>(
    actor: &TabularActor<B>,
    row_id: u64,
    input: &DiagnosticInput,
) -> Result<FiniteDecision>
where
    B: StateKeyBuilder,
    B::Key: Serialize,
{
    decision(actor, row_id, input, ActMode::Sample)
}

fn decision<B>(
    actor: &TabularActor<B>,
    row_id: u64,
    input: &DiagnosticInput,
    mode: ActMode,
) -> Result<FiniteDecision>
where
    B: StateKeyBuilder,
    B::Key: Serialize,
{
    input.validate(&actor.observation_space(), actor.action_space())?;
    let observation = input.observation();
    let state = if let Some(history) = input.public_history() {
        history
            .inputs()
            .into_iter()
            .try_fold(actor.initial_state()?, |state, input| {
                actor.assimilate(&state, input)
            })?
    } else {
        current_state(actor, observation)?
    };
    let key = actor.keys.key(&state)?;
    let legal: Vec<_> = super::typed::legal_keys(observation, actor.action_space())?
        .into_iter()
        .collect();
    let greedy = actor.table.greedy_index(&key, &legal)?;
    let epsilon = if mode == ActMode::Sample {
        actor.epsilon
    } else {
        0.0
    };
    let uniform = epsilon / legal.len() as f64;
    let information_state =
        String::from_utf8(codec::bounded_json(&key, 4096)?).expect("JSON is UTF-8");
    Ok(FiniteDecision {
        row_id,
        information_state: Some(information_state),
        selected: Some(legal[greedy]),
        candidates: legal
            .iter()
            .enumerate()
            .map(|(index, &action)| FiniteCandidate {
                action,
                probability: Some(uniform + if index == greedy { 1.0 - epsilon } else { 0.0 }),
                value: Some(actor.table.value(&key, &action)),
                visits: None,
                updates: Some(actor.table.visits(&key, &action)),
            })
            .collect(),
    })
}

fn snapshot<B>(actor: &TabularActor<B>, mut context: RecordContext, limit: usize) -> Result<Chunk>
where
    B: StateKeyBuilder + Serialize + DeserializeOwned,
    B::Key: Serialize + DeserializeOwned,
{
    let array_key = "models/table";
    let bytes = codec::bounded_json(actor, limit.min(super::typed::MAX_TYPED_BYTES))?;
    let id = format!("revision-{}", actor.revision());
    context.identity.checkpoint_id = Some(id.clone());
    Ok(Chunk {
        context,
        record: Record::ModelSnapshot(ModelSnapshotRecord {
            id,
            architecture: ARCHITECTURE.into(),
            configuration: serde_json::Value::Null,
            models: BTreeMap::from([(
                "table".into(),
                ModelArtifact {
                    bytes: array_key.into(),
                    signature: TabularActor::<B>::record_type().into(),
                    parameter_shapes: BTreeMap::new(),
                },
            )]),
            capabilities: vec!["sparse-state-action-values".into()],
        }),
        arrays: BTreeMap::from([(array_key.into(), Array::u8(vec![bytes.len()], bytes)?)]),
    })
}
