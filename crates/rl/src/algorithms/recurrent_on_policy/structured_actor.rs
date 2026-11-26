//! Frozen policy and optional scalar value with independently owned lane memories.
use super::structured::{
    conditional_likelihood, conditional_snapshot, load_structured_models, model_spaces,
    validate_architectures,
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        stateful_on_policy::{
            DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor,
            recording_distribution_matches, recording_number_matches,
        },
        structured::StructuredObservation,
        temporal::TemporalSpaces,
    },
    nn::{
        checkpoint::{RecordBurnExt, hash_weights},
        distributions::Distribution,
        structured::{
            ConditionalPolicyLaw, PackLimits, RecurrentLightweightConfig,
            RecurrentLightweightPolicy, RecurrentLightweightState, RecurrentLightweightValue,
            RecurrentPolicyEvaluation,
        },
        temporal_actor::validate_disjoint,
        temporal_models::validate_parameters,
        to_vec,
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::{rc::Rc, sync::Arc};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Memory {
    hidden: Vec<f32>,
    pub(crate) event_start: usize,
}

impl Memory {
    pub(crate) fn from_state(state: RecurrentLightweightState) -> Result<Self> {
        let hidden = to_vec(state.memory)?;
        ensure(
            hidden.iter().all(|v| v.is_finite()),
            "nonfinite structured actor memory",
        )?;
        Ok(Self {
            hidden,
            event_start: state.event_start,
        })
    }

    pub(crate) fn on_device(&self, device: &Device) -> RecurrentLightweightState {
        RecurrentLightweightState {
            memory: Tensor::<1>::from_data(self.hidden.as_slice(), device)
                .reshape([1, self.hidden.len()]),
            event_start: self.event_start,
        }
    }

    pub(crate) fn valid(&self, width: usize) -> bool {
        self.hidden.len() == width && self.hidden.iter().all(|v| v.is_finite())
    }
}

/// Host-owned post-observation memories. Each component records the current
/// frame's event suffix so the decision head reads the same evidence as before.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredTemporalActorState {
    policy: Memory,
    value: Option<Memory>,
}

impl StructuredTemporalActorState {
    pub(crate) fn event_start(&self) -> usize {
        self.policy.event_start
    }
}

#[derive(Debug)]
struct Models {
    policy: RecurrentLightweightPolicy,
    value: Option<RecurrentLightweightValue>,
    spaces: TemporalSpaces,
}

/// Immutable conditional behavior for the temporal on-policy runtime.
#[derive(Debug, Clone)]
pub struct StructuredTemporalActor {
    models: Rc<Models>,
    identity: [u8; 32],
    revision: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord {
    version: u32,
    revision: u64,
    policy: RecurrentLightweightConfig,
    value: Option<RecurrentLightweightConfig>,
}

impl StructuredTemporalActor {
    pub fn new(
        policy: RecurrentLightweightPolicy,
        value: Option<RecurrentLightweightValue>,
        revision: u64,
    ) -> Result<Self> {
        validate_disjoint_if_present(&policy, value.as_ref())?;
        let device = Device::flex();
        Self::from_frozen(
            policy.valid().fork(&device).freeze(),
            value.map(|model| model.valid().fork(&device).freeze()),
            revision,
        )
    }

    fn from_frozen(
        policy: RecurrentLightweightPolicy,
        value: Option<RecurrentLightweightValue>,
        revision: u64,
    ) -> Result<Self> {
        let spaces = model_spaces(policy.config());
        let mut actor = Self {
            models: Rc::new(Models {
                policy,
                value,
                spaces,
            }),
            identity: [0; 32],
            revision,
        };
        actor.validate_snapshot()?;
        actor.identity = actor.content_identity()?;
        Ok(actor)
    }

    pub fn policy(&self) -> &RecurrentLightweightPolicy {
        &self.models.policy
    }

    pub fn value(&self) -> Option<&RecurrentLightweightValue> {
        self.models.value.as_ref()
    }

    pub fn limits(&self) -> PackLimits {
        PackLimits::default()
    }

    pub fn model_identity(&self) -> [u8; 32] {
        self.identity
    }

    pub fn with_revision(&self, revision: u64) -> Self {
        Self {
            revision,
            ..self.clone()
        }
    }

    fn metadata(&self) -> ActorRecord {
        ActorRecord {
            version: 1,
            revision: self.revision,
            policy: self.policy().config().clone(),
            value: self.value().map(|value| value.config().clone()),
        }
    }

    fn content_identity(&self) -> Result<[u8; 32]> {
        let mut hash = Sha256::new();
        hash.update(b"rust-rl/structured-temporal-on-policy-actor-v1\0");
        let mut metadata = self.metadata();
        metadata.revision = 0;
        hash.update(serde_json::to_vec(&metadata)?);
        hash_weights(&mut hash, self.policy())?;
        if let Some(value) = self.value() {
            hash_weights(&mut hash, value)?;
        }
        Ok(hash.finalize().into())
    }

    fn record(&self) -> Result<LearnerRecord> {
        self.validate_snapshot()?;
        let mut record = LearnerRecord::new("structured_temporal_actor", self.metadata())?;
        record.model("policy", self.policy())?;
        if let Some(value) = self.value() {
            record.model("value", value)?;
        }
        Ok(record)
    }

    fn from_record(record: LearnerRecord) -> Result<Self> {
        record.check("structured_temporal_actor")?;
        let metadata: ActorRecord = serde_json::from_value(record.metadata.clone())?;
        ensure(
            metadata.version == 1
                && record.models.contains_key("policy")
                && record.models.len() == 1 + usize::from(metadata.value.is_some())
                && record.models.contains_key("value") == metadata.value.is_some()
                && record.optimizers.is_empty()
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none(),
            "invalid structured temporal actor record",
        )?;
        let (policy, value) = load_structured_models(
            &record,
            &metadata.policy,
            metadata.value.as_ref(),
            &Device::flex(),
        )?;
        Self::from_frozen(policy, value, metadata.revision)
    }

    fn frame<'a>(
        &self,
        observation: &'a Observation,
        live: bool,
    ) -> Result<&'a StructuredObservation> {
        observation.validate_in(&self.models.spaces.observations, self.action_space(), live)?;
        match observation {
            Observation::Structured(frame) => Ok(frame),
            _ => Err(Error::Shape(
                "structured behavior requires a structured observation".into(),
            )),
        }
    }

    fn evaluation<'p, 'a>(
        &'p self,
        state: &StructuredTemporalActorState,
        frame: &'a StructuredObservation,
    ) -> Result<RecurrentPolicyEvaluation<'p, 'a>> {
        self.validate_state(state)?;
        let memory = state.policy.on_device(&self.policy().model_device());
        self.policy()
            .evaluate_decision(&memory, frame, self.limits())
    }

    fn evaluations<'p, 'a>(
        &'p self,
        states: &[&StructuredTemporalActorState],
        observations: &[&'a Observation],
    ) -> Result<Vec<(&'a StructuredObservation, RecurrentPolicyEvaluation<'p, 'a>)>> {
        let mut frames = Vec::with_capacity(observations.len());
        let mut memories = Vec::with_capacity(states.len());
        let device = self.policy().model_device();
        for (state, observation) in states.iter().zip(observations) {
            frames.push(self.frame(observation, true)?);
            self.validate_state(state)?;
            memories.push(state.policy.on_device(&device));
        }
        let evaluations =
            self.policy()
                .evaluate_decisions_grouped(&memories, &frames, self.limits())?;
        Ok(frames.into_iter().zip(evaluations).collect())
    }

    fn value_at(&self, state: &StructuredTemporalActorState) -> Result<Option<f32>> {
        self.value()
            .zip(state.value.as_ref())
            .map(|(model, state)| {
                let number =
                    to_vec(model.value_from_state(&state.on_device(&model.model_device()))?)?[0];
                ensure(number.is_finite(), "nonfinite structured state value")?;
                Ok(number)
            })
            .transpose()
    }

    fn sample_evaluation(
        &self,
        state: &StructuredTemporalActorState,
        frame: &StructuredObservation,
        evaluation: &RecurrentPolicyEvaluation<'_, '_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        let law = evaluation.complete_law(0)?;
        let distribution = conditional_snapshot(&law, frame)?;
        let key = match mode {
            ActMode::Sample => law.keys()[select_complete(&law, rng)?],
            ActMode::Deterministic => evaluation.greedy(0)?.key,
        };
        let outcomes = distribution.conditional_outcomes()?;
        let row = outcomes
            .binary_search_by_key(&key, |outcome| outcome.action)
            .expect("sampled action belongs to the recorded law");
        Ok(OnPolicySample {
            decision: StatefulDecision {
                action: Action::Conditional(Arc::new(key)),
                log_prob: outcomes[row].log_prob,
            },
            latent: None,
            distribution,
            value: self.value_at(state)?,
        })
    }
}

fn validate_disjoint_if_present(
    policy: &RecurrentLightweightPolicy,
    value: Option<&RecurrentLightweightValue>,
) -> Result<()> {
    if let Some(value) = value {
        validate_disjoint(policy, value)?;
    }
    Ok(())
}

fn select_complete(law: &ConditionalPolicyLaw, rng: &mut ChaCha8Rng) -> Result<usize> {
    let width = law.keys().len();
    let categorical = Distribution::Categorical {
        logits: law.log_probs().clone().reshape([1, width]),
        masks: vec![true; width],
    };
    let Action::Discrete(selected) = categorical.sample(ActMode::Sample, rng)?[0] else {
        unreachable!("categorical sampler returns a discrete row")
    };
    Ok(selected)
}

pub(super) fn sample_complete(
    law: &ConditionalPolicyLaw,
    rng: &mut ChaCha8Rng,
) -> Result<StatefulDecision> {
    let selected = select_complete(law, rng)?;
    Ok(StatefulDecision {
        action: Action::Conditional(Arc::new(law.keys()[selected])),
        log_prob: Some(law.log_probs().clone().narrow(0, selected, 1).into_scalar()),
    })
}

fn decide_evaluation(
    evaluation: &RecurrentPolicyEvaluation<'_, '_>,
    mode: ActMode,
    rng: &mut ChaCha8Rng,
) -> Result<StatefulDecision> {
    let sample = evaluation.sample_row(0, mode, rng)?;
    Ok(StatefulDecision {
        action: Action::Conditional(Arc::new(sample.key)),
        log_prob: Some(sample.log_prob),
    })
}

impl StatefulActor for StructuredTemporalActor {
    type State = StructuredTemporalActorState;

    fn observation_space(&self) -> ObservationSpace {
        self.models.spaces.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.models.spaces.actions
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn initial_state(&self) -> Result<Self::State> {
        Ok(StructuredTemporalActorState {
            policy: Memory::from_state(self.policy().initial_state()?)?,
            value: self
                .value()
                .map(|value| Memory::from_state(value.initial_state()?))
                .transpose()?,
        })
    }

    fn validate_state(&self, state: &Self::State) -> Result<()> {
        ensure(
            state.policy.valid(self.policy().config().hidden_width)
                && match (self.value(), &state.value) {
                    (Some(model), Some(value)) => {
                        value.valid(model.config().hidden_width)
                            && value.event_start == state.policy.event_start
                    }
                    (None, None) => true,
                    _ => false,
                },
            "invalid structured temporal actor compound state",
        )
    }

    fn validate_state_at(
        &self,
        state: &Self::State,
        observation: &Observation,
        preceding_observation: Option<&Observation>,
    ) -> Result<()> {
        self.validate_state(state)?;
        let frame = self.frame(observation, false)?;
        if let Some(preceding) = preceding_observation {
            ensure(
                self.frame(preceding, true)?.events.len() == state.policy.event_start,
                "structured temporal memory disagrees with the factual event boundary",
            )?;
        }
        let events = frame
            .events
            .get(state.policy.event_start..)
            .ok_or_else(|| {
                Error::State("structured temporal event boundary exceeds observation".into())
            })?;
        frame.validate_event_links(events)
    }

    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        self.validate_state(prior)?;
        let policy = self.policy();
        let state = StructuredTemporalActorState {
            policy: Memory::from_state(policy.assimilate(
                &prior.policy.on_device(&policy.model_device()),
                input,
                self.limits(),
            )?)?,
            value: self
                .value()
                .zip(prior.value.as_ref())
                .map(|(model, prior)| {
                    Memory::from_state(model.assimilate(
                        &prior.on_device(&model.model_device()),
                        input,
                        self.limits(),
                    )?)
                })
                .transpose()?,
        };
        self.validate_state(&state)?;
        Ok(state)
    }

    fn batch_compatible(&self, other: &Self) -> bool {
        self.revision == other.revision
            && self.identity == other.identity
            && self.policy().grouped_inference_supported()
    }

    fn assimilate_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        self.assimilate_canonical_batch(priors, inputs)
    }

    fn assimilate_canonical_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        ensure(
            priors.len() == inputs.len(),
            "structured actor assimilation count mismatch",
        )?;
        if inputs.len() < 2 || !self.policy().grouped_inference_supported() {
            return priors
                .iter()
                .zip(inputs)
                .map(|(prior, &input)| self.assimilate(prior, input))
                .collect();
        }
        for prior in priors {
            self.validate_state(prior)?;
        }
        let policy = self.policy();
        let device = policy.model_device();
        let policy_priors: Vec<_> = priors
            .iter()
            .map(|prior| prior.policy.on_device(&device))
            .collect();
        let policies = policy.assimilate_grouped(&policy_priors, inputs, self.limits())?;
        let values = self
            .value()
            .map(|value| {
                let device = value.model_device();
                let priors: Vec<_> = priors
                    .iter()
                    .map(|prior| {
                        prior
                            .value
                            .as_ref()
                            .expect("validated value memory")
                            .on_device(&device)
                    })
                    .collect();
                value.assimilate_grouped(&priors, inputs, self.limits())
            })
            .transpose()?;
        let mut values = values.into_iter().flatten();
        policies
            .into_iter()
            .map(|policy| {
                let state = StructuredTemporalActorState {
                    policy: Memory::from_state(policy)?,
                    value: values.next().map(Memory::from_state).transpose()?,
                };
                self.validate_state(&state)?;
                Ok(state)
            })
            .collect()
    }

    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let frame = self.frame(observation, true)?;
        let evaluation = self.evaluation(state, frame)?;
        decide_evaluation(&evaluation, mode, rng)
    }

    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_canonical_batch(states, observations, mode, rngs)
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
            "structured actor decision count mismatch",
        )?;
        self.evaluations(states, observations)?
            .iter()
            .zip(rngs)
            .map(|((_, evaluation), rng)| decide_evaluation(evaluation, mode, rng))
            .collect()
    }
}

impl StatefulOnPolicyActor for StructuredTemporalActor {
    fn sample_on_policy(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        let frame = self.frame(observation, true)?;
        let evaluation = self.evaluation(state, frame)?;
        self.sample_evaluation(state, frame, &evaluation, mode, rng)
    }

    fn sample_on_policy_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        self.sample_on_policy_canonical_batch(states, observations, mode, rngs)
    }

    fn sample_on_policy_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "structured actor sampling count mismatch",
        )?;
        self.evaluations(states, observations)?
            .iter()
            .zip(states)
            .zip(rngs)
            .map(|(((frame, evaluation), state), rng)| {
                self.sample_evaluation(state, frame, evaluation, mode, rng)
            })
            .collect()
    }

    fn state_value(&self, state: &Self::State, observation: &Observation) -> Result<Option<f32>> {
        self.validate_state_at(state, observation, None)?;
        self.value_at(state)
    }
}

impl OnPolicyIdentity for StructuredTemporalActor {
    fn behavior_identity(&self) -> [u8; 32] {
        self.model_identity()
    }

    fn validate_recording(
        &self,
        state: &Self::State,
        observation: &Observation,
        action: &Action,
        latent: Option<&[f32]>,
        distribution: &DistributionSnapshot,
        log_prob: f32,
        value: Option<f32>,
    ) -> Result<()> {
        let frame = self.frame(observation, true)?;
        observation.check_action(self.action_space(), action)?;
        let Action::Conditional(key) = action else {
            return Err(Error::Action(
                "structured behavior requires a complete action".into(),
            ));
        };
        ensure(
            latent.is_none(),
            "conditional behavior cannot record a latent",
        )?;
        distribution.validate_for(self.action_space())?;
        let law = self.evaluation(state, frame)?.complete_law(0)?;
        let expected = conditional_snapshot(&law, frame)?;
        ensure(
            recording_distribution_matches(&expected, distribution),
            "conditional recording distribution differs from reconstructed behavior",
        )?;
        let expected_log_prob = to_vec(conditional_likelihood(&law, **key)?)?[0];
        ensure(
            recording_number_matches(expected_log_prob, log_prob),
            "conditional recording likelihood differs from reconstructed behavior",
        )?;
        ensure(
            match (self.value_at(state)?, value) {
                (Some(expected), Some(recorded)) => recording_number_matches(expected, recorded),
                (None, None) => true,
                _ => false,
            },
            "structured recording baseline differs from reconstructed behavior",
        )
    }
}

impl Snapshot for StructuredTemporalActor {
    fn validate_snapshot(&self) -> Result<()> {
        validate_architectures(
            self.policy().config(),
            self.value().map(RecurrentLightweightValue::config),
        )?;
        self.policy()
            .config()
            .model
            .input
            .validate_spaces(&self.models.spaces.observations, self.action_space())?;
        validate_disjoint_if_present(self.policy(), self.value())?;
        validate_parameters(self.policy(), true)?;
        if let Some(value) = self.value() {
            validate_parameters(value, true)?;
        }
        Ok(())
    }
}

impl Serialize for StructuredTemporalActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.record()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StructuredTemporalActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(LearnerRecord::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
