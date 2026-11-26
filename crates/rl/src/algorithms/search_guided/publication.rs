use super::*;
use crate::{
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
    },
    nn::{
        checkpoint::{RecordBurnExt, hash_weights, trainability},
        structured::{LightweightConfig, StructuredSample, candidate_probabilities_checked},
        target::frozen_copy,
    },
};
use burn::{module::Module, tensor::Device};
use rand::{
    SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserializer, Serializer};
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Immutable inference pair. Identity covers both networks, their schemas and
/// physical-return semantics; revision identifies the publication event.
#[derive(Clone)]
pub struct PolicyValuePublication {
    policy: Arc<LightweightPolicy>,
    value: Arc<LightweightValue>,
    returns: PlanningReturns,
    limits: PackLimits,
    revision: u64,
    source_updates: u64,
    identity: [u8; 32],
    action_space: ActionSpace,
}
impl PolicyValuePublication {
    pub(super) fn new(learner: &SearchGuidedPolicyValue, revision: u64) -> Result<Self> {
        let mut publication = Self {
            policy: Arc::new(frozen_copy(&learner.policy)),
            value: Arc::new(frozen_copy(&learner.value)),
            returns: learner.returns(),
            limits: learner.config.limits(),
            revision,
            source_updates: learner.updates,
            identity: [0; 32],
            action_space: ActionSpace::Conditional(Arc::new(
                learner.policy.config().input.action_grammar(),
            )),
        };
        publication.identity = publication.digest()?;
        publication.validate_snapshot()?;
        Ok(publication)
    }
    pub fn policy(&self) -> &LightweightPolicy {
        &self.policy
    }
    pub fn value(&self) -> &LightweightValue {
        &self.value
    }
    pub fn returns(&self) -> PlanningReturns {
        self.returns
    }
    pub fn limits(&self) -> PackLimits {
        self.limits
    }
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn source_updates(&self) -> u64 {
        self.source_updates
    }

    fn digest(&self) -> Result<[u8; 32]> {
        let mut hash = Sha256::new();
        hash.update(b"rust_rl.policy_value_publication.v1");
        hash.update(serde_json::to_vec(&(
            self.policy.config(),
            self.value.config(),
            self.returns,
            self.limits.max_elements,
        ))?);
        hash_weights(&mut hash, &*self.policy)?;
        hash_weights(&mut hash, &*self.value)?;
        Ok(hash.finalize().into())
    }
    fn sample(&self, observation: &Observation) -> Result<StructuredSample<ConditionalAction>> {
        let Observation::Structured(frame) = observation else {
            return Err(crate::Error::Shape(
                "policy/value publication needs a structured observation".into(),
            ));
        };
        StructuredSample::from_observation_with_validated_schema(frame, &self.policy.config().input)
    }
    pub fn action_probabilities(
        &self,
        observation: &Observation,
    ) -> Result<Vec<(ConditionalAction, f64)>> {
        let sample = self.sample(observation)?;
        let law = candidate_probabilities_checked(&*self.policy, &[sample], self.limits)?.remove(0);
        let Observation::Structured(frame) = observation else {
            unreachable!("validated structured input")
        };
        Ok(frame
            .candidates
            .iter()
            .zip(law)
            .filter(|(c, _)| c.enabled)
            .map(|(c, p)| (c.action, p))
            .collect())
    }
    pub fn value_prediction(&self, observation: &Observation) -> Result<f64> {
        let sample = self.sample(observation)?;
        finite_loss(&self.value.evaluate(&[sample], self.limits)?)
    }
}

impl StatefulActor for PolicyValuePublication {
    type State = ();
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Structured(Arc::new(self.policy.config().input.observation_schema()))
    }
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
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
            .validate_in(&self.observation_space(), &self.action_space, false)
    }
    fn decide(
        &self,
        _: &(),
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let law = self.action_probabilities(observation)?;
        let index = if mode == ActMode::Deterministic {
            // Stable complete-action maximum, including across operation heads.
            (0..law.len()).fold(0, |best, i| if law[i].1 > law[best].1 { i } else { best })
        } else {
            WeightedIndex::new(law.iter().map(|(_, p)| p))
                .map_err(|e| crate::Error::Numerical(e.to_string()))?
                .sample(rng)
        };
        Ok(StatefulDecision {
            action: Action::Conditional(Arc::new(law[index].0)),
            log_prob: Some(if mode == ActMode::Deterministic {
                0.
            } else {
                law[index].1.ln().min(0.) as f32
            }),
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Meta {
    version: u32,
    policy: LightweightConfig,
    value: LightweightConfig,
    returns: PlanningReturns,
    max_pack_elements: usize,
    revision: u64,
    source_updates: u64,
    identity: [u8; 32],
}
impl Serialize for PolicyValuePublication {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let record = (|| -> Result<LearnerRecord> {
            let mut record = LearnerRecord::new(
                "policy-value-publication-v1",
                Meta {
                    version: 1,
                    policy: self.policy.config().clone(),
                    value: self.value.config().clone(),
                    returns: self.returns,
                    max_pack_elements: self.limits.max_elements,
                    revision: self.revision,
                    source_updates: self.source_updates,
                    identity: self.identity,
                },
            )?;
            record.model("policy", &*self.policy)?;
            record.model("value", &*self.value)?;
            Ok(record)
        })()
        .map_err(serde::ser::Error::custom)?;
        record.serialize(serializer)
    }
}
impl PolicyValuePublication {
    fn from_record(record: LearnerRecord) -> Result<Self> {
        record.check("policy-value-publication-v1")?;
        let meta: Meta = serde_json::from_value(record.metadata.clone())?;
        ensure(
            meta.version == 1
                && record.models.len() == 2
                && record.optimizers.is_empty()
                && record.temporal_returns.is_none()
                && record.temporal_spaces.is_none()
                && record.temporal_reconstruction.is_none()
                && record.replay_requirements.is_none(),
            "invalid policy/value publication record",
        )?;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let policy = LightweightPolicy::new(meta.policy, &mut rng, &Device::flex())?;
        let value = LightweightValue::new(meta.value, &mut rng, &Device::flex())?;
        let policy = record.load_model_inference("policy", &policy)?.freeze();
        let value = record.load_model_inference("value", &value)?.freeze();
        let action_space =
            ActionSpace::Conditional(Arc::new(policy.config().input.action_grammar()));
        let publication = Self {
            policy: Arc::new(policy),
            value: Arc::new(value),
            returns: meta.returns,
            limits: PackLimits {
                max_elements: meta.max_pack_elements,
            },
            revision: meta.revision,
            source_updates: meta.source_updates,
            identity: meta.identity,
            action_space,
        };
        publication.validate_snapshot()?;
        Ok(publication)
    }
}
impl<'de> Deserialize<'de> for PolicyValuePublication {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(LearnerRecord::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
impl Snapshot for PolicyValuePublication {
    fn validate_snapshot(&self) -> Result<()> {
        self.returns.validate()?;
        self.policy.config().validate()?;
        self.value.config().validate()?;
        ensure(
            self.returns.objective.is_physical()
                && self.policy.config().input == self.value.config().input
                && self.revision > 0
                && self.source_updates < i32::MAX as u64
                && self.limits.max_elements > 0
                && !trainability(&*self.policy).contains(&true)
                && !trainability(&*self.value).contains(&true),
            "invalid frozen policy/value publication",
        )?;
        parameter_norm(&*self.policy)?;
        parameter_norm(&*self.value)?;
        validate_disjoint(&*self.policy, &*self.value)?;
        ensure(
            self.digest()? == self.identity,
            "policy/value publication identity differs from weights or semantics",
        )
    }
}
