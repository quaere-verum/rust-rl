use super::{
    Minibatch, RecurrentOnPolicyPolicy, distribution::RecurrentPolicyDistribution,
    structured_actor::StructuredTemporalActor,
};
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        checkpoint::LearnerRecord,
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor},
        stateful_on_policy::{ConditionalOutcome, DistributionSnapshot, StatefulOnPolicyActor},
        structured::StructuredObservation,
        temporal::TemporalSpaces,
    },
    data::temporal_on_policy::TemporalBatchKind,
    nn::{
        checkpoint::RecordBurnExt,
        structured::{
            ConditionalPolicyLaw, InputSchema, PackLimits, RecurrentLightweightConfig,
            RecurrentLightweightPolicy, RecurrentLightweightValue, RecurrentPolicyEvaluation,
            StructuredHistoryInput,
        },
        temporal_models::validate_record_model_count,
        to_vec,
    },
};
use burn::tensor::{Device, Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::{collections::HashMap, sync::Arc};

pub(crate) const OPERATION_HEAD_ADAM_PATHS: &[&str] = &[
    "operation_head.layers.1.weight",
    "operation_head.layers.1.bias",
];

/// A batch of differentiable complete-action conditional policy laws.
pub struct ConditionalBatch(ConditionalRows);

enum ConditionalRows {
    Selected {
        keys: Vec<ConditionalAction>,
        log_probs: Box<Tensor<1>>,
    },
    Complete(Vec<ConditionalPolicyLaw>),
}

impl ConditionalBatch {
    /// Stack nonempty laws in the same order as the caller's learning rows.
    pub fn from_laws(rows: Vec<ConditionalPolicyLaw>) -> Result<Self> {
        ensure(!rows.is_empty(), "empty conditional policy law batch")?;
        Ok(Self(ConditionalRows::Complete(rows)))
    }

    pub(super) fn from_selection(keys: Vec<ConditionalAction>, log_probs: Tensor<1>) -> Self {
        Self(ConditionalRows::Selected {
            keys,
            log_probs: Box::new(log_probs),
        })
    }

    pub(crate) fn complete(&self) -> Result<&[ConditionalPolicyLaw]> {
        match &self.0 {
            ConditionalRows::Complete(rows) => Ok(rows),
            ConditionalRows::Selected { .. } => {
                Err(Error::State("complete conditional law required".into()))
            }
        }
    }
}

pub(crate) fn conditional_key(action: &Action) -> Result<ConditionalAction> {
    match action {
        Action::Conditional(key) => Ok(**key),
        _ => Err(Error::Action(
            "expected a complete conditional action".into(),
        )),
    }
}

pub(crate) fn conditional_likelihood(
    law: &ConditionalPolicyLaw,
    key: ConditionalAction,
) -> Result<Tensor<1>> {
    let row = law
        .keys()
        .iter()
        .position(|&candidate| candidate == key)
        .ok_or_else(|| Error::Action("conditional action is absent from policy support".into()))?;
    Ok(law.log_probs().clone().narrow(0, row, 1))
}

pub(crate) fn conditional_snapshot(
    law: &ConditionalPolicyLaw,
    frame: &StructuredObservation,
) -> Result<DistributionSnapshot> {
    let mut enabled: HashMap<_, _> = law
        .keys()
        .iter()
        .copied()
        .zip(to_vec(law.log_probs().clone())?)
        .collect();
    let outcomes = frame
        .candidates
        .iter()
        .map(|candidate| {
            let log_prob = enabled.remove(&candidate.action);
            ensure(
                log_prob.is_some() == candidate.enabled,
                "conditional policy support differs from observation",
            )?;
            Ok(ConditionalOutcome {
                action: candidate.action,
                log_prob,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure(
        enabled.is_empty(),
        "conditional policy reported foreign candidates",
    )?;
    DistributionSnapshot::conditional(outcomes)
}

impl RecurrentPolicyDistribution for ConditionalBatch {
    type EntropyCache = ();

    fn log_prob(&self, actions: &[Action], latents: &[Option<Vec<f32>>]) -> Result<Tensor<1>> {
        ensure(
            !actions.is_empty()
                && actions.len() == latents.len()
                && latents.iter().all(Option::is_none),
            "conditional likelihood requires one action per row and no latents",
        )?;
        let actions = actions
            .iter()
            .map(conditional_key)
            .collect::<Result<Vec<_>>>()?;
        match &self.0 {
            ConditionalRows::Selected { keys, log_probs } => {
                ensure(
                    &actions == keys,
                    "conditional likelihood actions differ from evaluated keys",
                )?;
                Ok(log_probs.as_ref().clone())
            }
            ConditionalRows::Complete(rows) => {
                ensure(
                    actions.len() == rows.len(),
                    "conditional likelihood batch mismatch",
                )?;
                let selected = rows
                    .iter()
                    .zip(actions)
                    .map(|(row, key)| conditional_likelihood(row, key))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Tensor::cat(selected, 0))
            }
        }
    }

    fn entropy(&self, _: &mut ChaCha8Rng) -> Result<Tensor<1>> {
        Ok(Tensor::cat(
            self.complete()?
                .iter()
                .map(ConditionalPolicyLaw::entropy)
                .collect(),
            0,
        ))
    }

    fn kl(&self, behavior: &Self) -> Result<Tensor<1>> {
        let current = self.complete()?;
        let behavior = behavior.complete()?;
        ensure(
            current.len() == behavior.len(),
            "conditional KL batch mismatch",
        )?;
        let rows = current
            .iter()
            .zip(behavior)
            .map(|(current, behavior)| current.kl(behavior))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(rows, 0))
    }

    fn snapshot(&self, row: usize, observation: &Observation) -> Result<DistributionSnapshot> {
        let row = self
            .complete()?
            .get(row)
            .ok_or_else(|| Error::Shape("conditional snapshot row is out of range".into()))?;
        let Observation::Structured(frame) = observation else {
            return Err(Error::Shape(
                "conditional snapshot requires a structured observation".into(),
            ));
        };
        conditional_snapshot(row, frame)
    }

    fn from_snapshots(snapshots: &[DistributionSnapshot], device: &Device) -> Result<Self> {
        ensure(
            !snapshots.is_empty(),
            "empty conditional distribution snapshot batch",
        )?;
        let rows = snapshots
            .iter()
            .map(|snapshot| {
                snapshot.validate()?;
                let (keys, logs): (Vec<_>, Vec<_>) = snapshot
                    .conditional_outcomes()?
                    .iter()
                    .filter_map(|outcome| Some((outcome.action, outcome.log_prob?)))
                    .unzip();
                ConditionalPolicyLaw::new(keys, Tensor::from_data(logs.as_slice(), device))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_laws(rows)
    }
}

pub(crate) fn model_spaces(config: &RecurrentLightweightConfig) -> TemporalSpaces {
    model_spaces_for(&config.model.input)
}

pub(super) fn model_spaces_for(input: &InputSchema) -> TemporalSpaces {
    TemporalSpaces {
        observations: ObservationSpace::Structured(Arc::new(input.observation_schema())),
        actions: ActionSpace::Conditional(Arc::new(input.action_grammar())),
    }
}

pub(super) fn validate_architectures(
    policy: &RecurrentLightweightConfig,
    value: Option<&RecurrentLightweightConfig>,
) -> Result<()> {
    RecurrentLightweightPolicy::parameter_count(policy)?;
    if let Some(value) = value {
        RecurrentLightweightValue::parameter_count(value)?;
        ensure(
            policy.model.input == value.model.input,
            "structured actor and value input schemas differ",
        )?;
    }
    Ok(())
}

pub(crate) fn load_structured_models(
    record: &LearnerRecord,
    policy: &RecurrentLightweightConfig,
    value: Option<&RecurrentLightweightConfig>,
    device: &Device,
) -> Result<(
    RecurrentLightweightPolicy,
    Option<RecurrentLightweightValue>,
)> {
    validate_architectures(policy, value)?;
    validate_record_model_count(
        record,
        "policy",
        RecurrentLightweightPolicy::parameter_count(policy)?,
    )?;
    if let Some(value) = value {
        validate_record_model_count(
            record,
            "value",
            RecurrentLightweightValue::parameter_count(value)?,
        )?;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let policy = RecurrentLightweightPolicy::new(policy.clone(), &mut rng, device)?;
    let policy = if device.is_autodiff() {
        record.load_model("policy", &policy)?
    } else {
        record.load_model_inference("policy", &policy)?
    };
    let value = value
        .map(|value| {
            let value = RecurrentLightweightValue::new(value.clone(), &mut rng, device)?;
            if device.is_autodiff() {
                record.load_model("value", &value)
            } else {
                record.load_model_inference("value", &value)
            }
        })
        .transpose()?;
    Ok((policy, value))
}

pub(crate) fn exact_histories<'a, A: StatefulOnPolicyActor + Clone>(
    batch: &'a Minibatch<'_, A>,
) -> Result<Vec<(usize, Vec<HistoryInput<'a>>)>> {
    batch
        .samples()
        .map(|sample| Ok((sample.learn_start, sample.exact_history_inputs()?.collect())))
        .collect()
}

pub(crate) fn history_inputs<'a>(
    histories: &'a [(usize, Vec<HistoryInput<'a>>)],
) -> Vec<StructuredHistoryInput<'a>> {
    histories
        .iter()
        .map(|(learn_start, inputs)| StructuredHistoryInput {
            inputs,
            learn_start: *learn_start,
        })
        .collect()
}

pub(crate) fn evaluate_complete_laws(
    evaluations: &[RecurrentPolicyEvaluation<'_, '_>],
    tokens: &[(usize, usize)],
) -> Result<Vec<ConditionalPolicyLaw>> {
    let mut laws = Vec::with_capacity(tokens.len());
    for window in tokens.chunk_by(|a, b| a.0 == b.0) {
        let evaluation = evaluations.get(window[0].0).ok_or_else(|| {
            Error::Shape("conditional law window is outside the evaluated histories".into())
        })?;
        let times: Vec<_> = window.iter().map(|&(_, time)| time).collect();
        laws.extend(evaluation.complete_laws(&times)?);
    }
    Ok(laws)
}

impl RecurrentOnPolicyPolicy for RecurrentLightweightPolicy {
    type Value = RecurrentLightweightValue;
    type Actor = StructuredTemporalActor;
    type Architecture = RecurrentLightweightConfig;
    type Distribution = ConditionalBatch;

    const RECORD_VERSION: u32 = 3;
    const SPARSE_ADAM: bool = true;
    const REQUIRED_POLICY_ADAM_PATHS: &'static [&'static str] = OPERATION_HEAD_ADAM_PATHS;
    const REQUIRED_VALUE_ADAM_PATHS: &'static [&'static str] =
        &["head.layers.1.weight", "head.layers.1.bias"];

    fn policy_device(&self) -> Device {
        self.model_device()
    }

    fn architectures(
        &self,
        value: Option<&Self::Value>,
    ) -> Result<(Self::Architecture, Option<Self::Architecture>)> {
        let value = value.map(RecurrentLightweightValue::config);
        validate_architectures(self.config(), value)?;
        Ok((self.config().clone(), value.cloned()))
    }

    fn publish(policy: Self, value: Option<Self::Value>, revision: u64) -> Result<Self::Actor> {
        StructuredTemporalActor::new(policy, value, revision)
    }

    fn published_models(actor: &Self::Actor) -> (&Self, Option<&Self::Value>) {
        (actor.policy(), actor.value())
    }

    fn with_revision(actor: &Self::Actor, revision: u64) -> Self::Actor {
        actor.with_revision(revision)
    }

    fn binding(actor: &Self::Actor, kind: TemporalBatchKind) -> Result<TemporalLearnerBinding> {
        TemporalLearnerBinding::new_structured(
            kind,
            actor.revision(),
            actor.model_identity(),
            TemporalSpaces {
                observations: actor.observation_space(),
                actions: actor.action_space().clone(),
            },
        )
    }

    fn restore_models(
        record: &LearnerRecord,
        policy: &Self::Architecture,
        value: Option<&Self::Architecture>,
        binding: &TemporalLearnerBinding,
    ) -> Result<(Self, Option<Self::Value>)> {
        ensure(
            binding.dense_spec().is_none(),
            "structured learner requires a structured collection binding",
        )?;
        load_structured_models(record, policy, value, &crate::nn::device())
    }

    fn distribution(
        &self,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
        complete_support: bool,
    ) -> Result<ConditionalBatch> {
        ensure(!tokens.is_empty(), "empty structured policy selection")?;
        let exact = exact_histories(batch)?;
        let histories = history_inputs(&exact);
        let evaluations = self.evaluate_histories(&histories, PackLimits::default())?;
        if complete_support {
            return ConditionalBatch::from_laws(evaluate_complete_laws(&evaluations, tokens)?);
        }
        let keys = tokens
            .iter()
            .map(|&(sample, time)| conditional_key(&batch.sample(sample).steps[time].action))
            .collect::<Result<Vec<_>>>()?;
        let positions: Vec<_> = tokens
            .iter()
            .zip(&keys)
            .map(|(&(sample, time), &key)| (sample, time, key))
            .collect();
        let mut rows = Vec::new();
        for group in positions.chunk_by(|a, b| a.0 == b.0) {
            let selected: Vec<_> = group.iter().map(|&(_, time, key)| (time, key)).collect();
            rows.push(evaluations[group[0].0].selected_log_probs(&selected)?);
        }
        Ok(ConditionalBatch::from_selection(keys, Tensor::cat(rows, 0)))
    }

    fn values(
        value: &Self::Value,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<1>> {
        let exact = exact_histories(batch)?;
        value.evaluate_selected(&history_inputs(&exact), tokens, PackLimits::default())
    }
}
