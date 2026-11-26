//! Stateless conditional policy and scalar value in the shared temporal lifecycle.
use super::{
    Minibatch, RecurrentOnPolicyPolicy,
    structured::{
        ConditionalBatch, OPERATION_HEAD_ADAM_PATHS, conditional_key, conditional_likelihood,
        conditional_snapshot, model_spaces_for,
    },
    structured_actor::sample_complete,
};
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::data::temporal_on_policy::TemporalLearnerBinding;
use crate::{
    Result,
    algorithms::candidate_ddqn::{candidate_frame, candidate_sample},
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, ReconstructionHistory, StatefulActor, StatefulDecision},
        stateful_on_policy::{
            DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor,
            recording_distribution_matches, recording_number_matches,
        },
        temporal::{TemporalExecutionMode, TemporalSpaces},
    },
    data::temporal_on_policy::TemporalBatchKind,
    nn::{
        checkpoint::{RecordBurnExt, hash_weights},
        structured::{
            ConditionalPolicyLaw, InputSchema, LightweightConfig, LightweightPolicy,
            LightweightValue, PackLimits, StructuredSample,
        },
        temporal_actor::validate_disjoint,
        temporal_models::{validate_parameters, validate_record_model_count},
        to_vec,
    },
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::{rc::Rc, sync::Arc};

type Sample = StructuredSample<ConditionalAction>;
const ACTOR_RECORD: &str = "lightweight_temporal_actor";

#[derive(Debug)]
struct Models {
    policy: LightweightPolicy,
    value: Option<LightweightValue>,
    spaces: TemporalSpaces,
}

/// Frozen conditional behavior whose entire input is the current public frame.
#[derive(Debug, Clone)]
pub struct LightweightTemporalActor {
    models: Rc<Models>,
    identity: [u8; 32],
    revision: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord {
    version: u32,
    revision: u64,
    policy: LightweightConfig,
    value: Option<LightweightConfig>,
}

impl LightweightTemporalActor {
    pub fn new(
        policy: LightweightPolicy,
        value: Option<LightweightValue>,
        revision: u64,
    ) -> Result<Self> {
        if let Some(value) = &value {
            validate_disjoint(&policy, value)?;
        }
        let device = Device::flex();
        Self::from_frozen(
            policy.valid().fork(&device).freeze(),
            value.map(|model| model.valid().fork(&device).freeze()),
            revision,
        )
    }

    fn from_frozen(
        policy: LightweightPolicy,
        value: Option<LightweightValue>,
        revision: u64,
    ) -> Result<Self> {
        let mut actor = Self {
            models: Rc::new(Models {
                spaces: model_spaces_for(&policy.config().input),
                policy,
                value,
            }),
            identity: [0; 32],
            revision,
        };
        actor.validate_snapshot()?;
        let mut hash = Sha256::new();
        hash.update(b"rust-rl/lightweight-temporal-on-policy-actor-v1\0");
        hash.update(serde_json::to_vec(&actor.metadata(0))?);
        hash_weights(&mut hash, actor.policy())?;
        if let Some(value) = actor.value() {
            hash_weights(&mut hash, value)?;
        }
        actor.identity = hash.finalize().into();
        Ok(actor)
    }

    pub fn policy(&self) -> &LightweightPolicy {
        &self.models.policy
    }

    pub fn value(&self) -> Option<&LightweightValue> {
        self.models.value.as_ref()
    }

    fn metadata(&self, revision: u64) -> ActorRecord {
        ActorRecord {
            version: 1,
            revision,
            policy: self.policy().config().clone(),
            value: self.value().map(|model| model.config().clone()),
        }
    }

    fn record(&self) -> Result<LearnerRecord> {
        self.validate_snapshot()?;
        let mut record = LearnerRecord::new(ACTOR_RECORD, self.metadata(self.revision))?;
        record.model("policy", self.policy())?;
        if let Some(value) = self.value() {
            record.model("value", value)?;
        }
        Ok(record)
    }

    fn from_record(record: LearnerRecord) -> Result<Self> {
        record.check(ACTOR_RECORD)?;
        let metadata: ActorRecord = serde_json::from_value(record.metadata.clone())?;
        ensure(
            metadata.version == 1
                && record.models.contains_key("policy")
                && record.models.len() == 1 + usize::from(metadata.value.is_some())
                && record.models.contains_key("value") == metadata.value.is_some()
                && record.optimizers.is_empty()
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none(),
            "invalid lightweight temporal actor record",
        )?;
        let (policy, value) = load_models(
            &record,
            &metadata.policy,
            metadata.value.as_ref(),
            &Device::flex(),
        )?;
        Self::from_frozen(policy, value, metadata.revision)
    }

    fn validate_observation(&self, observation: &Observation, live: bool) -> Result<()> {
        observation.validate_in(&self.models.spaces.observations, self.action_space(), live)
    }

    fn sample(&self, observation: &Observation, live: bool) -> Result<Sample> {
        self.validate_observation(observation, live)?;
        candidate_sample(observation, &self.policy().config().input)
    }

    fn law(&self, sample: &Sample) -> Result<ConditionalPolicyLaw> {
        self.policy()
            .evaluate(std::slice::from_ref(sample), PackLimits::default())?
            .complete_law(0)
    }

    fn value_at(&self, sample: &Sample) -> Result<Option<f32>> {
        self.value()
            .map(|model| {
                let value =
                    to_vec(model.evaluate(std::slice::from_ref(sample), PackLimits::default())?)?
                        [0];
                ensure(value.is_finite(), "nonfinite lightweight state value")?;
                Ok(value)
            })
            .transpose()
    }
}

impl StatefulActor for LightweightTemporalActor {
    type State = ();

    fn observation_space(&self) -> ObservationSpace {
        self.models.spaces.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.models.spaces.actions
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

    fn validate_state_at(
        &self,
        _: &(),
        observation: &Observation,
        _: Option<&Observation>,
    ) -> Result<()> {
        self.validate_observation(observation, false)
    }

    fn assimilate(&self, _: &(), input: HistoryInput<'_>) -> Result<()> {
        self.validate_observation(input.observation, false)
    }

    fn decide(
        &self,
        _: &(),
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let sample = self.sample(observation, true)?;
        let evaluation = self
            .policy()
            .evaluate(std::slice::from_ref(&sample), PackLimits::default())?;
        match mode {
            ActMode::Sample => sample_complete(&evaluation.complete_law(0)?, rng),
            ActMode::Deterministic => {
                let choice = evaluation.greedy(0)?;
                Ok(StatefulDecision {
                    action: Action::Conditional(Arc::new(choice.key)),
                    log_prob: Some(choice.log_prob),
                })
            }
        }
    }
}

impl StatefulOnPolicyActor for LightweightTemporalActor {
    fn sample_on_policy(
        &self,
        _: &(),
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        let sample = self.sample(observation, true)?;
        let evaluation = self
            .policy()
            .evaluate(std::slice::from_ref(&sample), PackLimits::default())?;
        let law = evaluation.complete_law(0)?;
        let distribution = conditional_snapshot(&law, candidate_frame(observation)?)?;
        let decision = match mode {
            ActMode::Sample => sample_complete(&law, rng)?,
            ActMode::Deterministic => {
                let key = evaluation.greedy(0)?.key;
                StatefulDecision {
                    action: Action::Conditional(Arc::new(key)),
                    log_prob: Some(conditional_likelihood(&law, key)?.into_scalar()),
                }
            }
        };
        Ok(OnPolicySample {
            decision,
            latent: None,
            distribution,
            value: self.value_at(&sample)?,
        })
    }

    fn state_value(&self, _: &(), observation: &Observation) -> Result<Option<f32>> {
        self.value_at(&self.sample(observation, false)?)
    }
}

impl OnPolicyIdentity for LightweightTemporalActor {
    fn behavior_identity(&self) -> [u8; 32] {
        self.identity
    }

    fn validate_recording(
        &self,
        _: &(),
        observation: &Observation,
        action: &Action,
        latent: Option<&[f32]>,
        distribution: &DistributionSnapshot,
        log_prob: f32,
        value: Option<f32>,
    ) -> Result<()> {
        let sample = self.sample(observation, true)?;
        observation.check_action(self.action_space(), action)?;
        ensure(
            latent.is_none(),
            "conditional behavior cannot record a latent",
        )?;
        distribution.validate_for(self.action_space())?;
        let law = self.law(&sample)?;
        let expected = conditional_snapshot(&law, candidate_frame(observation)?)?;
        ensure(
            recording_distribution_matches(&expected, distribution),
            "conditional recording distribution differs from reconstructed behavior",
        )?;
        let expected_log_prob = to_vec(conditional_likelihood(&law, conditional_key(action)?)?)?[0];
        ensure(
            recording_number_matches(expected_log_prob, log_prob),
            "conditional recording likelihood differs from reconstructed behavior",
        )?;
        ensure(
            match (self.value_at(&sample)?, value) {
                (Some(expected), Some(recorded)) => recording_number_matches(expected, recorded),
                (None, None) => true,
                _ => false,
            },
            "lightweight recording baseline differs from reconstructed behavior",
        )
    }
}

impl Snapshot for LightweightTemporalActor {
    fn validate_snapshot(&self) -> Result<()> {
        validate_architectures(
            self.policy().config(),
            self.value().map(LightweightValue::config),
        )?;
        self.policy()
            .config()
            .input
            .validate_spaces(&self.models.spaces.observations, self.action_space())?;
        validate_parameters(self.policy(), true)?;
        if let Some(value) = self.value() {
            validate_disjoint(self.policy(), value)?;
            validate_parameters(value, true)?;
        }
        Ok(())
    }
}

impl Serialize for LightweightTemporalActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.record()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LightweightTemporalActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(LearnerRecord::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}

fn validate_architectures(
    policy: &LightweightConfig,
    value: Option<&LightweightConfig>,
) -> Result<()> {
    LightweightPolicy::parameter_count(policy)?;
    if let Some(value) = value {
        LightweightValue::parameter_count(value)?;
        ensure(
            policy.input == value.input,
            "lightweight policy and value input schemas differ",
        )?;
    }
    Ok(())
}

fn load_models(
    record: &LearnerRecord,
    policy: &LightweightConfig,
    value: Option<&LightweightConfig>,
    device: &Device,
) -> Result<(LightweightPolicy, Option<LightweightValue>)> {
    validate_architectures(policy, value)?;
    validate_record_model_count(
        record,
        "policy",
        LightweightPolicy::parameter_count(policy)?,
    )?;
    if let Some(value) = value {
        validate_record_model_count(record, "value", LightweightValue::parameter_count(value)?)?;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let policy = LightweightPolicy::new(policy.clone(), &mut rng, device)?;
    let policy = load_model(record, "policy", &policy, device)?;
    let value = value
        .map(|config| {
            let value = LightweightValue::new(config.clone(), &mut rng, device)?;
            load_model(record, "value", &value, device)
        })
        .transpose()?;
    Ok((policy, value))
}

fn load_model<M: Module>(
    record: &LearnerRecord,
    name: &str,
    model: &M,
    device: &Device,
) -> Result<M> {
    if device.is_autodiff() {
        record.load_model(name, model)
    } else {
        record.load_model_inference(name, model)
    }
}

fn samples(
    batch: &Minibatch<'_, LightweightTemporalActor>,
    tokens: &[(usize, usize)],
    schema: &InputSchema,
) -> Result<Vec<Sample>> {
    batch
        .observations(tokens)
        .into_iter()
        .map(|observation| candidate_sample(observation, schema))
        .collect()
}

impl RecurrentOnPolicyPolicy for LightweightPolicy {
    type Value = LightweightValue;
    type Actor = LightweightTemporalActor;
    type Architecture = LightweightConfig;
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
        let value = value.map(LightweightValue::config);
        validate_architectures(self.config(), value)?;
        Ok((self.config().clone(), value.cloned()))
    }

    fn publish(policy: Self, value: Option<Self::Value>, revision: u64) -> Result<Self::Actor> {
        LightweightTemporalActor::new(policy, value, revision)
    }

    fn published_models(actor: &Self::Actor) -> (&Self, Option<&Self::Value>) {
        (actor.policy(), actor.value())
    }

    fn with_revision(actor: &Self::Actor, revision: u64) -> Self::Actor {
        LightweightTemporalActor {
            revision,
            ..actor.clone()
        }
    }

    fn binding(actor: &Self::Actor, kind: TemporalBatchKind) -> Result<TemporalLearnerBinding> {
        TemporalLearnerBinding::new_structured(
            kind,
            actor.revision,
            actor.identity,
            actor.models.spaces.clone(),
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
            "lightweight learner requires a structured collection binding",
        )?;
        load_models(record, policy, value, &crate::nn::device())
    }

    fn distribution(
        &self,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
        complete_support: bool,
    ) -> Result<ConditionalBatch> {
        let samples = samples(batch, tokens, &self.config().input)?;
        let evaluation = self.evaluate(&samples, PackLimits::default())?;
        if complete_support {
            let rows: Vec<_> = (0..tokens.len()).collect();
            return ConditionalBatch::from_laws(evaluation.complete_laws(&rows)?);
        }
        let keys = tokens
            .iter()
            .map(|&(sample, time)| conditional_key(&batch.sample(sample).steps[time].action))
            .collect::<Result<Vec<_>>>()?;
        let selected: Vec<_> = keys.iter().copied().enumerate().collect();
        Ok(ConditionalBatch::from_selection(
            keys,
            evaluation.selected_log_probs(&selected)?,
        ))
    }

    fn values(
        value: &Self::Value,
        batch: &Minibatch<'_, Self::Actor>,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<1>> {
        let samples = samples(batch, tokens, &value.config().input)?;
        value.evaluate(&samples, PackLimits::default())
    }

    fn reconstruct_published_state(
        _: &Self::Actor,
        _: ReconstructionHistory<'_>,
        _: TemporalExecutionMode,
    ) -> Result<Option<()>> {
        Ok(Some(()))
    }
}
