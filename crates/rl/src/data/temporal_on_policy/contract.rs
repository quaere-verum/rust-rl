//! Temporal learner update and collection identity contracts.

use super::{TemporalBatchKind, TemporalOnPolicyBatch};
use crate::{
    Error, Result,
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationSpace,
        stateful::{BehaviorPublisher, HistorySpec, ReconstructionHistory, StatefulLearner},
        stateful_on_policy::OnPolicyIdentity,
        temporal::TemporalSpaces,
    },
};
use serde::{Deserialize, Serialize};

pub trait TemporalOnPolicyLearner<A: OnPolicyIdentity + Clone>:
    StatefulLearner<Batch = TemporalOnPolicyBatch<A>>
    + BehaviorPublisher<Actor = A, Settings = ()>
    + Checkpoint
{
    fn collection_kind(&self) -> TemporalBatchKind;
    fn revision(&self) -> u64;
    /// Start a fresh frozen collection transaction. Most learners need no extra
    /// bookkeeping. A learner may advance its behavior nonce without changing
    /// parameters; errors must leave its state unchanged. The trainer reconciles
    /// any newer revision before the first environment action.
    fn begin_collection(&self) -> Result<()> {
        Ok(())
    }
    fn normalizes_advantages_in_update(&self) -> bool {
        false
    }
    /// Optional exact reconstruction under the supplied newly published actor.
    /// The collector validates the public tape before calling this hook. A
    /// learner can return `None` to use sequential scalar assimilation.
    fn reconstruct_published_state(
        &self,
        _actor: &A,
        _history: ReconstructionHistory<'_>,
    ) -> Result<Option<A::State>> {
        Ok(None)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalLearnerBinding {
    pub version: u32,
    pub kind: TemporalBatchKind,
    pub revision: u64,
    pub identity: [u8; 32],
    spec: BindingSpec,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum BindingSpec {
    Dense(HistorySpec),
    Structured(TemporalSpaces),
}

impl TemporalLearnerBinding {
    pub fn new(
        kind: TemporalBatchKind,
        revision: u64,
        identity: [u8; 32],
        spec: HistorySpec,
    ) -> Result<Self> {
        let result = Self {
            version: 1,
            kind,
            revision,
            identity,
            spec: BindingSpec::Dense(spec),
        };
        result.validate()?;
        Ok(result)
    }
    pub fn new_structured(
        kind: TemporalBatchKind,
        revision: u64,
        identity: [u8; 32],
        spaces: TemporalSpaces,
    ) -> Result<Self> {
        let binding = Self {
            version: 2,
            kind,
            revision,
            identity,
            spec: BindingSpec::Structured(spaces),
        };
        binding.validate()?;
        Ok(binding)
    }

    pub fn dense_spec(&self) -> Option<&HistorySpec> {
        match &self.spec {
            BindingSpec::Dense(spec) => Some(spec),
            BindingSpec::Structured(_) => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match (&self.spec, self.version) {
            (BindingSpec::Dense(spec), 1) => spec.validate(),
            (BindingSpec::Structured(spaces), 2) => {
                ensure(
                    matches!(spaces.observations, ObservationSpace::Structured(_))
                        && matches!(spaces.actions, ActionSpace::Conditional(_)),
                    "structured temporal learner binding requires a schema and conditional grammar",
                )?;
                spaces.observations.validate()?;
                spaces.actions.validate()
            }
            _ => Err(Error::State(
                "unsupported temporal learner collection version".into(),
            )),
        }
    }
    pub fn validate_actor<A: OnPolicyIdentity>(&self, actor: &A) -> Result<()> {
        self.validate()?;
        let observations = actor.observation_space();
        let matches_spaces = match &self.spec {
            BindingSpec::Dense(spec) => {
                observations == ObservationSpace::Dense(spec.observation_dim)
                    && &spec.action_space == actor.action_space()
            }
            BindingSpec::Structured(spaces) => {
                spaces.observations == observations && &spaces.actions == actor.action_space()
            }
        };
        ensure(
            self.revision == actor.revision()
                && self.identity == actor.behavior_identity()
                && matches_spaces,
            "temporal learner collection binding differs from behavior actor",
        )
    }
    pub fn from_record(record: &LearnerRecord) -> Result<Self> {
        let binding: Self =
            serde_json::from_value(record.metadata.get("collection").cloned().ok_or_else(
                || Error::State("missing temporal learner collection binding".into()),
            )?)?;
        binding.validate()?;
        Ok(binding)
    }
}
