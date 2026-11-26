//! Immutable policy sampling with independently reconstructed action-value history.
use super::{
    checkpoint::RecordBurnExt,
    target::frozen_copy,
    temporal::TemporalEncoder,
    temporal_actor::{TemporalActor, TemporalActorState, assimilate_model, validate_disjoint},
    temporal_models,
    temporal_policy::TemporalPolicy,
    temporal_q::{TemporalQ, TemporalQArchitecture},
    to_vec,
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
        stateful::{HistoryInput, HistorySpec, StatefulActor, StatefulDecision},
        stateful_on_policy::{DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor},
    },
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::Tensor,
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::rc::Rc;

/// Both memories are post-observation; candidate actions never advance either one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPathwiseActorState {
    pub policy: TemporalActorState,
    pub critic: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TemporalPathwiseActor {
    policy_actor: TemporalActor,
    critic: Rc<TemporalQ>,
    identity: [u8; 32],
}
impl TemporalPathwiseActor {
    pub fn new(policy: TemporalPolicy, critic: TemporalQ, revision: u64) -> Result<Self> {
        ensure(
            policy.spec() == critic.spec(),
            "pathwise policy and critic history specs differ",
        )?;
        critic.architecture()?;
        temporal_models::validate_parameters(&critic, false)?;
        validate_disjoint(&policy, &critic)?;
        Self::from_frozen(
            TemporalActor::new(policy, None, revision)?,
            frozen_copy(&critic),
        )
    }
    fn from_frozen(policy_actor: TemporalActor, critic: TemporalQ) -> Result<Self> {
        let mut actor = Self {
            policy_actor,
            critic: Rc::new(critic),
            identity: [0; 32],
        };
        actor.validate_snapshot()?;
        actor.identity = actor.content_identity()?;
        Ok(actor)
    }
    pub fn policy(&self) -> &TemporalPolicy {
        self.policy_actor.policy()
    }
    pub fn critic(&self) -> &TemporalQ {
        &self.critic
    }
    pub fn spec(&self) -> &HistorySpec {
        self.policy_actor.spec()
    }
    pub fn revision(&self) -> u64 {
        self.policy_actor.revision()
    }
    pub fn model_identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn with_revision(&self, revision: u64) -> Self {
        Self {
            policy_actor: self.policy_actor.with_revision(revision),
            critic: Rc::clone(&self.critic),
            identity: self.identity,
        }
    }
    fn content_identity(&self) -> Result<[u8; 32]> {
        struct Content {
            digest: Sha256,
            error: Option<Error>,
        }
        impl ModuleVisitor for Content {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                self.digest.update((D as u64).to_le_bytes());
                let tensor = param.val().without_autodiff();
                for width in tensor.dims() {
                    self.digest.update((width as u64).to_le_bytes());
                }
                match to_vec(tensor) {
                    Ok(values) => {
                        for value in values {
                            self.digest.update(value.to_bits().to_le_bytes());
                        }
                    }
                    Err(error) => self.error = Some(error),
                }
            }
        }
        let mut content = Content {
            digest: Sha256::new(),
            error: None,
        };
        content
            .digest
            .update(b"rust-rl/temporal-pathwise-actor-v1\0");
        content.digest.update(self.policy_actor.model_identity());
        content
            .digest
            .update(serde_json::to_vec(&self.critic().architecture()?)?);
        self.critic().visit(&mut content);
        if let Some(error) = content.error {
            return Err(error);
        }
        Ok(content.digest.finalize().into())
    }
    fn policy_states<'a>(
        &self,
        states: &[&'a TemporalPathwiseActorState],
    ) -> Result<Vec<&'a TemporalActorState>> {
        states
            .iter()
            .map(|state| {
                self.validate_state(state)?;
                Ok(&state.policy)
            })
            .collect()
    }
    fn assimilate_impl(
        &self,
        priors: &[&TemporalPathwiseActorState],
        inputs: &[HistoryInput<'_>],
        canonical: bool,
    ) -> Result<Vec<TemporalPathwiseActorState>> {
        ensure(
            priors.len() == inputs.len(),
            "pathwise assimilation batch count mismatch",
        )?;
        let policy_priors = self.policy_states(priors)?;
        if priors.is_empty() {
            return Ok(Vec::new());
        }
        let policy = if canonical {
            self.policy_actor
                .assimilate_canonical_batch(&policy_priors, inputs)?
        } else {
            self.policy_actor.assimilate_batch(&policy_priors, inputs)?
        };
        let critic = assimilate_model(
            &self.critic().encoder,
            self.spec(),
            self.critic().observation_only(),
            &self.critic().device(),
            &priors
                .iter()
                .map(|state| state.critic.as_slice())
                .collect::<Vec<_>>(),
            inputs,
            canonical,
        )?;
        Ok(policy
            .into_iter()
            .zip(critic)
            .map(|(policy, critic)| TemporalPathwiseActorState { policy, critic })
            .collect())
    }
    fn wire(&self) -> Result<PathwiseWire> {
        self.validate_snapshot()?;
        let mut critic = LearnerRecord::new(
            "temporal_pathwise_critic",
            CriticMetadata {
                version: 1,
                architecture: self.critic().architecture()?,
            },
        )?;
        critic.model("critic", self.critic())?;
        Ok(PathwiseWire {
            version: 1,
            policy: self.policy_actor.clone(),
            critic,
        })
    }
    fn from_wire(wire: PathwiseWire) -> Result<Self> {
        ensure(
            wire.version == 1,
            "unsupported temporal pathwise actor snapshot version",
        )?;
        wire.critic.check("temporal_pathwise_critic")?;
        ensure(
            wire.critic.models.len() == 1
                && wire.critic.models.contains_key("critic")
                && wire.critic.optimizers.is_empty()
                && wire.critic.temporal_returns.is_none()
                && wire.critic.temporal_reconstruction.is_none(),
            "invalid pathwise critic record ownership",
        )?;
        let metadata: CriticMetadata = serde_json::from_value(wire.critic.metadata.clone())?;
        ensure(
            metadata.version == 1,
            "unsupported pathwise critic metadata version",
        )?;
        ensure(
            wire.policy.value().is_none()
                && wire.policy.spec() == &metadata.architecture.model.spec,
            "pathwise actor snapshot history schema mismatch",
        )?;
        // Q restoration checks raw tensor dimensions, contents and bounded allocation
        // against the descriptor before constructing an inference template.
        let critic = TemporalQ::restore_from_record_with_mode(
            &wire.critic,
            "critic",
            &metadata.architecture,
            true,
        )?;
        Self::from_frozen(wire.policy, critic)
    }
}

impl StatefulActor for TemporalPathwiseActor {
    type State = TemporalPathwiseActorState;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.spec().observation_dim)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.spec().action_space
    }
    fn revision(&self) -> u64 {
        self.revision()
    }
    fn initial_state(&self) -> Result<Self::State> {
        Ok(TemporalPathwiseActorState {
            policy: self.policy_actor.initial_state()?,
            critic: vec![0.; self.critic().encoder.output_dim()],
        })
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.policy_actor.validate_state(&state.policy)?;
        ensure(
            state.critic.len() == self.critic().encoder.output_dim()
                && state.critic.iter().all(|value| value.is_finite()),
            "invalid pathwise critic memory",
        )
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        Ok(self.assimilate_batch(&[prior], &[input])?.remove(0))
    }
    fn assimilate_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        self.assimilate_impl(priors, inputs, false)
    }
    fn assimilate_canonical_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        self.assimilate_impl(priors, inputs, true)
    }
    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        self.validate_state(state)?;
        self.policy_actor
            .decide(&state.policy, observation, mode, rng)
    }
    fn decide_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.policy_actor.decide_canonical_batch(
            &self.policy_states(states)?,
            observations,
            mode,
            rngs,
        )
    }
    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.policy_actor
            .decide_batch(&self.policy_states(states)?, observations, mode, rngs)
    }
    fn batch_compatible(&self, other: &Self) -> bool {
        self.revision() == other.revision() && self.identity == other.identity
    }
}
impl StatefulOnPolicyActor for TemporalPathwiseActor {
    fn sample_on_policy(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        self.validate_state(state)?;
        self.policy_actor
            .sample_on_policy(&state.policy, observation, mode, rng)
    }
    fn sample_on_policy_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        self.policy_actor.sample_on_policy_canonical_batch(
            &self.policy_states(states)?,
            observations,
            mode,
            rngs,
        )
    }
    fn state_values_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        self.policy_actor
            .state_values_canonical_batch(&self.policy_states(states)?, observations)
    }
    fn sample_on_policy_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        self.policy_actor.sample_on_policy_batch(
            &self.policy_states(states)?,
            observations,
            mode,
            rngs,
        )
    }
    fn state_value(&self, state: &Self::State, observation: &Observation) -> Result<Option<f32>> {
        self.validate_state(state)?;
        self.policy_actor.state_value(&state.policy, observation)
    }
    fn state_values_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        self.policy_actor
            .state_values_batch(&self.policy_states(states)?, observations)
    }
}
impl OnPolicyIdentity for TemporalPathwiseActor {
    fn behavior_identity(&self) -> [u8; 32] {
        self.model_identity()
    }
    fn audit_portable_memory(
        &self,
        saved: &Self::State,
        native: &Self::State,
    ) -> Result<Vec<crate::core::checkpoint::PortableMemoryAudit>> {
        self.validate_state(saved)?;
        self.validate_state(native)?;
        let mut audits = self
            .policy_actor
            .audit_portable_memory(&saved.policy, &native.policy)?;
        audits.push(super::temporal_actor::audit_portable_component(
            "q",
            &saved.critic,
            &native.critic,
            self.critic().encoder.gru.is_some(),
        )?);
        Ok(audits)
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
        self.validate_state(state)?;
        self.policy_actor.validate_recording(
            &state.policy,
            observation,
            action,
            latent,
            distribution,
            log_prob,
            value,
        )
    }
}
impl Snapshot for TemporalPathwiseActor {
    fn validate_snapshot(&self) -> Result<()> {
        self.policy_actor.validate_snapshot()?;
        self.critic().architecture()?;
        ensure(
            self.policy_actor.value().is_none() && self.spec() == self.critic().spec(),
            "invalid pathwise actor history schema",
        )?;
        temporal_models::validate_parameters(self.critic(), true)?;
        validate_disjoint(self.policy(), self.critic())
    }
}
impl Serialize for TemporalPathwiseActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.wire()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for TemporalPathwiseActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_wire(PathwiseWire::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PathwiseWire {
    version: u32,
    policy: TemporalActor,
    critic: LearnerRecord,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CriticMetadata {
    version: u32,
    architecture: TemporalQArchitecture,
}
