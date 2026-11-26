use super::structured::StructuredReppoQArchitecture;
use crate::core::stateful_on_policy::OnPolicyIdentity;
use crate::{
    Result,
    algorithms::recurrent_on_policy::{
        StructuredTemporalActor, StructuredTemporalActorState, structured_actor::Memory,
    },
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        stateful_on_policy::{DistributionSnapshot, OnPolicySample, StatefulOnPolicyActor},
    },
    nn::{
        checkpoint::{RecordBurnExt, hash_weights},
        structured::{RecurrentLightweightDistributionalQ, RecurrentLightweightPolicy},
        target::frozen_copy,
        temporal_actor::validate_disjoint,
        temporal_models::validate_parameters,
    },
};
use burn::tensor::Device;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::rc::Rc;

type Critic = RecurrentLightweightDistributionalQ;

/// Frozen structured policy and histogram Q with independent lane memories.
#[derive(Debug, Clone)]
pub struct StructuredPathwiseActor {
    policy: StructuredTemporalActor,
    critic: Rc<Critic>,
    identity: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredPathwiseActorState {
    policy: StructuredTemporalActorState,
    critic: Memory,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    version: u32,
    policy: StructuredTemporalActor,
    critic: LearnerRecord,
}

impl StructuredPathwiseActor {
    pub(super) fn new(
        policy: RecurrentLightweightPolicy,
        critic: Critic,
        revision: u64,
    ) -> Result<Self> {
        validate_disjoint(&policy, &critic)?;
        Self::from_frozen(
            StructuredTemporalActor::new(policy, None, revision)?,
            frozen_copy(&critic),
        )
    }

    fn from_frozen(policy: StructuredTemporalActor, critic: Critic) -> Result<Self> {
        let mut actor = Self {
            policy,
            critic: Rc::new(critic),
            identity: [0; 32],
        };
        actor.validate_snapshot()?;
        let mut hash = Sha256::new();
        hash.update(b"rust-rl/structured-pathwise-actor-v1\0");
        hash.update(actor.policy.model_identity());
        hash.update(serde_json::to_vec(&StructuredReppoQArchitecture::new(
            actor.critic(),
        ))?);
        hash_weights(&mut hash, actor.critic())?;
        actor.identity = hash.finalize().into();
        Ok(actor)
    }

    pub fn policy(&self) -> &RecurrentLightweightPolicy {
        self.policy.policy()
    }

    pub fn critic(&self) -> &Critic {
        &self.critic
    }

    pub(super) fn with_revision(&self, revision: u64) -> Self {
        Self {
            policy: self.policy.with_revision(revision),
            ..self.clone()
        }
    }

    fn policy_states<'a>(
        &self,
        states: &[&'a StructuredPathwiseActorState],
    ) -> Result<Vec<&'a StructuredTemporalActorState>> {
        states
            .iter()
            .map(|state| {
                self.validate_state(state)?;
                Ok(&state.policy)
            })
            .collect()
    }

    fn wire(&self) -> Result<Wire> {
        self.validate_snapshot()?;
        let mut critic = LearnerRecord::new(
            "structured_pathwise_critic",
            StructuredReppoQArchitecture::new(self.critic()),
        )?;
        critic.model("critic", self.critic())?;
        Ok(Wire {
            version: 1,
            policy: self.policy.clone(),
            critic,
        })
    }

    fn from_wire(wire: Wire) -> Result<Self> {
        wire.critic.check("structured_pathwise_critic")?;
        ensure(
            wire.version == 1
                && wire.critic.models.keys().map(String::as_str).eq(["critic"])
                && wire.critic.optimizers.is_empty()
                && wire.critic.temporal_returns.is_none()
                && wire.critic.temporal_reconstruction.is_none()
                && wire.critic.temporal_spaces.is_none(),
            "invalid structured pathwise actor record ownership",
        )?;
        let architecture: StructuredReppoQArchitecture =
            serde_json::from_value(wire.critic.metadata.clone())?;
        architecture.validate_record(&wire.critic, "critic")?;
        let critic = architecture.load(&wire.critic, "critic", &Device::flex())?;
        Self::from_frozen(wire.policy, critic)
    }
}

impl StatefulActor for StructuredPathwiseActor {
    type State = StructuredPathwiseActorState;

    fn observation_space(&self) -> ObservationSpace {
        self.policy.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }
    fn revision(&self) -> u64 {
        self.policy.revision()
    }

    fn initial_state(&self) -> Result<Self::State> {
        Ok(StructuredPathwiseActorState {
            policy: self.policy.initial_state()?,
            critic: Memory::from_state(self.critic().initial_state()?)?,
        })
    }

    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.policy.validate_state(&state.policy)?;
        ensure(
            state.critic.valid(self.critic().config().hidden_width)
                && state.critic.event_start == state.policy.event_start(),
            "invalid structured pathwise Q memory",
        )
    }

    fn validate_state_at(
        &self,
        state: &Self::State,
        observation: &Observation,
        preceding: Option<&Observation>,
    ) -> Result<()> {
        self.validate_state(state)?;
        self.policy
            .validate_state_at(&state.policy, observation, preceding)
    }

    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        self.validate_state(prior)?;
        let critic = self.critic();
        let state = StructuredPathwiseActorState {
            policy: self.policy.assimilate(&prior.policy, input)?,
            critic: Memory::from_state(critic.assimilate(
                &prior.critic.on_device(&critic.model_device()),
                input,
                self.policy.limits(),
            )?)?,
        };
        self.validate_state(&state)?;
        Ok(state)
    }

    fn batch_compatible(&self, other: &Self) -> bool {
        self.identity == other.identity && self.policy.batch_compatible(&other.policy)
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
            "structured pathwise assimilation count mismatch",
        )?;
        if inputs.len() < 2 || !self.batch_compatible(self) {
            return priors
                .iter()
                .zip(inputs)
                .map(|(prior, &input)| self.assimilate(prior, input))
                .collect();
        }
        let policies = self
            .policy
            .assimilate_canonical_batch(&self.policy_states(priors)?, inputs)?;
        let critic = self.critic();
        let device = critic.model_device();
        let critic_priors: Vec<_> = priors
            .iter()
            .map(|prior| prior.critic.on_device(&device))
            .collect();
        let critics = critic.assimilate_grouped(&critic_priors, inputs, self.policy.limits())?;
        policies
            .into_iter()
            .zip(critics)
            .map(|(policy, critic)| {
                let state = StructuredPathwiseActorState {
                    policy,
                    critic: Memory::from_state(critic)?,
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
        self.validate_state(state)?;
        self.policy.decide(&state.policy, observation, mode, rng)
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
        self.policy
            .decide_canonical_batch(&self.policy_states(states)?, observations, mode, rngs)
    }
}

impl StatefulOnPolicyActor for StructuredPathwiseActor {
    fn sample_on_policy(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        self.validate_state(state)?;
        self.policy
            .sample_on_policy(&state.policy, observation, mode, rng)
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
        self.policy.sample_on_policy_canonical_batch(
            &self.policy_states(states)?,
            observations,
            mode,
            rngs,
        )
    }

    fn state_value(&self, state: &Self::State, observation: &Observation) -> Result<Option<f32>> {
        self.validate_state_at(state, observation, None)?;
        Ok(None)
    }
}

impl OnPolicyIdentity for StructuredPathwiseActor {
    fn behavior_identity(&self) -> [u8; 32] {
        self.identity
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
        self.policy.validate_recording(
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

impl Snapshot for StructuredPathwiseActor {
    fn validate_snapshot(&self) -> Result<()> {
        self.policy.validate_snapshot()?;
        ensure(
            self.policy.value().is_none()
                && self.policy().config().model.input == self.critic().config().model.input,
            "invalid structured pathwise policy and Q schema",
        )?;
        validate_parameters(self.critic(), true)?;
        validate_disjoint(self.policy(), self.critic())
    }
}

impl Serialize for StructuredPathwiseActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.wire()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StructuredPathwiseActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_wire(Wire::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}
