//! Immutable stochastic behavior with independent actor/value post-observation memories.
use super::{
    checkpoint::RecordBurnExt,
    matrix_on_device,
    network::Network,
    target::frozen_copy,
    temporal::TemporalEncoder,
    temporal_models::{self, TemporalModelArchitecture},
    temporal_policy::TemporalPolicy,
    temporal_value::TemporalValue,
    temporal_work::reconstruct_scalar_chunk,
    to_vec,
};
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, RolloutBehavior},
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationBatch, ObservationSpace},
        stateful::{
            HistoryInput, HistorySpec, ReconstructionHistory, StatefulActor, StatefulDecision,
            replay_actor_history, validate_history_inputs,
        },
        stateful_on_policy::{
            DistributionSnapshot, OnPolicyIdentity, OnPolicySample, StatefulOnPolicyActor,
            recording_distribution_matches, recording_number_matches,
        },
    },
};
use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    tensor::{Device, Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, rc::Rc};

/// Host-owned memory. A memoryless model retains only its current observation features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalActorState {
    pub policy: Vec<f32>,
    pub value: Option<Vec<f32>>,
}

/// Version-one GRU import bound, calibrated against the retained Windows/Linux
/// fixture (maximum 2 ULP / 2^-25 endpoint discrepancy). Never used by inference,
/// optimization, ordinary restore, or observation-only state.
pub(crate) fn audit_portable_component(
    role: &str,
    saved: &[f32],
    native: &[f32],
    recurrent: bool,
) -> Result<crate::core::checkpoint::PortableMemoryAudit> {
    ensure(saved.len() == native.len(), "portable memory width differs")?;
    let mut audit = crate::core::checkpoint::PortableMemoryAudit {
        role: role.into(),
        rule: if recurrent {
            "gru-f32-4ulp-v1"
        } else {
            "exact-observation-v1"
        }
        .into(),
        values: saved.len(),
        adjusted_values: 0,
        max_absolute_difference: 0.,
        max_ulp_difference: 0,
    };
    let ordered = |bits: u32| {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    };
    for (&saved, &native) in saved.iter().zip(native) {
        ensure(
            saved.is_finite() && native.is_finite(),
            "nonfinite portable memory",
        )?;
        let absolute = (f64::from(saved) - f64::from(native)).abs();
        let ulps = ordered(saved.to_bits()).abs_diff(ordered(native.to_bits()));
        ensure(
            saved.to_bits() == native.to_bits()
                || (recurrent && ulps <= 4 && absolute <= 2.384_185_791_015_625e-7),
            "portable GRU memory exceeds measured reconstruction bound",
        )?;
        audit.adjusted_values += usize::from(saved.to_bits() != native.to_bits());
        audit.max_absolute_difference = audit.max_absolute_difference.max(absolute);
        audit.max_ulp_difference = audit.max_ulp_difference.max(ulps);
    }
    Ok(audit)
}
#[derive(Debug)]
struct Models {
    policy: TemporalPolicy,
    value: Option<TemporalValue>,
}

/// Portable concrete GRU/MLP behavior; generic learning models remain separate.
#[derive(Debug, Clone)]
pub struct TemporalActor {
    models: Rc<Models>,
    identity: [u8; 32],
    revision: u64,
}
impl TemporalActor {
    pub fn new(
        policy: TemporalPolicy,
        value: Option<TemporalValue>,
        revision: u64,
    ) -> Result<Self> {
        policy.architecture()?;
        temporal_models::validate_parameters(&policy, false)?;
        if let Some(value) = &value {
            value.architecture()?;
            ensure(
                value.spec() == policy.spec(),
                "temporal actor and value history specs differ",
            )?;
            temporal_models::validate_parameters(value, false)?;
            validate_disjoint(&policy, value)?;
        }
        let policy = frozen_copy(&policy);
        let value = value.as_ref().map(frozen_copy);
        Self::from_frozen(policy, value, revision)
    }
    fn from_frozen(
        policy: TemporalPolicy,
        value: Option<TemporalValue>,
        revision: u64,
    ) -> Result<Self> {
        let mut actor = Self {
            models: Rc::new(Models { policy, value }),
            identity: [0; 32],
            revision,
        };
        actor.validate_snapshot()?;
        actor.identity = actor.content_identity()?;
        Ok(actor)
    }
    pub fn policy(&self) -> &TemporalPolicy {
        &self.models.policy
    }
    pub fn value(&self) -> Option<&TemporalValue> {
        self.models.value.as_ref()
    }
    pub fn spec(&self) -> &HistorySpec {
        self.policy().spec()
    }
    pub(crate) fn reconstruct_encoded_history(
        &self,
        history: ReconstructionHistory<'_>,
    ) -> Result<Option<TemporalActorState>> {
        ensure(
            history.actions.len().checked_add(1) == Some(history.observations.len())
                && history.rewards.len() == history.actions.len(),
            "invalid encoded reconstruction history lengths",
        )?;
        self.reconstruct_encoded_inputs(history.observations.iter().enumerate().map(
            |(time, observation)| HistoryInput {
                observation,
                previous: time.checked_sub(1).map(|t| {
                    history
                        .rewards
                        .input(t, &history.actions[t], &history.observations[t])
                }),
            },
        ))
    }

    fn reconstruct_encoded_inputs<'a>(
        &self,
        mut history: impl ExactSizeIterator<Item = HistoryInput<'a>>,
    ) -> Result<Option<TemporalActorState>> {
        let Some(policy) = &self.policy().encoder.gru else {
            return Ok(None);
        };
        let Some(value) = self.value().and_then(|v| v.encoder.gru.as_ref()) else {
            return Ok(None);
        };
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::EncodedLaneReconstruction);
        let device = self.policy().device();
        let width = self.spec().input_dim();
        let max_width = width.max(policy.output_dim()).max(value.output_dim());
        let rows_per_chunk = (8192 / max_width).clamp(1, 128);
        let mut state = self.initial_state()?;
        while history.len() > 0 {
            let rows = history.len().min(rows_per_chunk);
            let mut features = Vec::new();
            features.try_reserve_exact(rows * width).map_err(|error| {
                Error::State(format!("cannot reserve reconstruction chunk: {error}"))
            })?;
            for input in history.by_ref().take(rows) {
                features.extend(self.spec().encode(input)?);
            }
            let inputs = Tensor::from_data(TensorData::new(features, [1, rows, width]), &device);
            state.policy = reconstruct_scalar_chunk(policy, inputs.clone(), &state.policy)?;
            let value_state = state.value.as_mut().expect("actor has a recurrent value");
            *value_state = reconstruct_scalar_chunk(value, inputs, value_state)?;
        }
        Ok(Some(state))
    }
    pub fn model_identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn with_revision(&self, revision: u64) -> Self {
        Self {
            models: Rc::clone(&self.models),
            identity: self.identity,
            revision,
        }
    }
    fn metadata(&self) -> Result<ActorMetadata> {
        Ok(ActorMetadata {
            version: 1,
            revision: self.revision,
            policy: self.policy().architecture()?,
            value: self.value().map(TemporalValue::architecture).transpose()?,
        })
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
                for d in tensor.dims() {
                    self.digest.update((d as u64).to_le_bytes());
                }
                match to_vec(tensor) {
                    Ok(values) => {
                        for value in values {
                            self.digest.update(value.to_bits().to_le_bytes());
                        }
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
        let mut content = Content {
            digest: Sha256::new(),
            error: None,
        };
        content
            .digest
            .update(b"rust-rl/temporal-on-policy-actor-v1\0");
        let mut metadata = self.metadata()?;
        metadata.revision = 0;
        content.digest.update(serde_json::to_vec(&metadata)?);
        self.policy().visit(&mut content);
        if let Some(value) = self.value() {
            value.visit(&mut content);
        }
        if let Some(e) = content.error {
            return Err(e);
        }
        Ok(content.digest.finalize().into())
    }
    fn record(&self) -> Result<LearnerRecord> {
        self.validate_snapshot()?;
        let mut record = LearnerRecord::new("temporal_actor", self.metadata()?)?;
        record.model("policy", self.policy())?;
        if let Some(value) = self.value() {
            record.model("value", value)?;
        }
        Ok(record)
    }
    fn from_record(record: LearnerRecord) -> Result<Self> {
        record.check("temporal_actor")?;
        let metadata: ActorMetadata = serde_json::from_value(record.metadata.clone())?;
        metadata.validate()?;
        ensure(
            record.models.len() == 1 + usize::from(metadata.value.is_some())
                && record.optimizers.is_empty()
                && record.temporal_returns.is_none()
                && record.temporal_reconstruction.is_none(),
            "invalid temporal actor model record",
        )?;
        temporal_models::validate_record_model(
            &record,
            "policy",
            &metadata.policy,
            metadata.policy.spec.action_space.dimensions(),
        )?;
        if let Some(value) = &metadata.value {
            temporal_models::validate_record_model(&record, "value", value, 0)?;
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let policy_template =
            TemporalPolicy::from_architecture_with_mode(&metadata.policy, &mut rng, true)?;
        let policy = record.load_model_inference("policy", &policy_template)?;
        let value = metadata
            .value
            .as_ref()
            .map(|architecture| {
                let template =
                    TemporalValue::from_architecture_with_mode(architecture, &mut rng, true)?;
                record.load_model_inference("value", &template)
            })
            .transpose()?;
        if let Some(value) = &value {
            validate_disjoint(&policy, value)?;
        }
        Self::from_frozen(policy, value, metadata.revision)
    }
    fn assimilate_impl(
        &self,
        priors: &[&TemporalActorState],
        inputs: &[HistoryInput<'_>],
        canonical: bool,
    ) -> Result<Vec<TemporalActorState>> {
        ensure(
            priors.len() == inputs.len(),
            "temporal assimilation batch count mismatch",
        )?;
        if priors.is_empty() {
            return Ok(Vec::new());
        }
        for prior in priors {
            self.validate_state(prior)?;
        }
        let policy = assimilate_model(
            &self.policy().encoder,
            self.spec(),
            self.policy().observation_only(),
            &self.policy().device(),
            &priors
                .iter()
                .map(|s| s.policy.as_slice())
                .collect::<Vec<_>>(),
            inputs,
            canonical,
        )?;
        let value = if let Some(model) = self.value() {
            Some(assimilate_model(
                &model.encoder,
                model.spec(),
                model.observation_only(),
                &model.device(),
                &priors
                    .iter()
                    .map(|s| s.value.as_deref().expect("validated critic state"))
                    .collect::<Vec<_>>(),
                inputs,
                canonical,
            )?)
        } else {
            None
        };
        let states: Vec<_> = policy
            .into_iter()
            .enumerate()
            .map(|(i, policy)| TemporalActorState {
                policy,
                value: value.as_ref().map(|v| v[i].clone()),
            })
            .collect();
        for state in &states {
            self.validate_state(state)?;
        }
        Ok(states)
    }
    fn state_values_impl(
        &self,
        states: &[&TemporalActorState],
        observations: &[&Observation],
        canonical: bool,
    ) -> Result<Vec<Option<f32>>> {
        ensure(
            states.len() == observations.len(),
            "temporal value batch count mismatch",
        )?;
        for (state, observation) in states.iter().zip(observations) {
            self.validate_state(state)?;
            observation.validate(self.spec().observation_dim, self.action_space(), false)?;
        }
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let Some(model) = self.value() else {
            return Ok(vec![None; states.len()]);
        };
        let features: Vec<_> = states
            .iter()
            .flat_map(|s| {
                s.value
                    .as_ref()
                    .expect("validated critic state")
                    .iter()
                    .copied()
            })
            .collect();
        let features = Tensor::from_data(
            TensorData::new(features, [states.len(), model.encoder.output_dim()]),
            &model.device(),
        );
        let values = to_vec(model.values_from_features_using(features, |x| {
            if canonical {
                super::canonical::mlp(&model.head, x)
            } else {
                model.head.forward(x)
            }
        })?)?;
        ensure(
            values.iter().all(|x| x.is_finite()),
            "nonfinite temporal actor state value",
        )?;
        Ok(values.into_iter().map(Some).collect())
    }
    fn policy_distribution(
        &self,
        states: &[&TemporalActorState],
        observations: &[&Observation],
    ) -> Result<super::distributions::Distribution> {
        self.policy_distribution_impl(states, observations, false)
    }
    fn sample_impl(
        &self,
        states: &[&TemporalActorState],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
        canonical: bool,
    ) -> Result<Vec<OnPolicySample>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "temporal on-policy sampling batch count mismatch",
        )?;
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let distribution = self.policy_distribution_impl(states, observations, canonical)?;
        let values = self.state_values_impl(states, observations, canonical)?;
        let mut samples = Vec::with_capacity(states.len());
        for (row, rng) in rngs.iter_mut().enumerate() {
            let snapshot = DistributionSnapshot::from_distribution(&distribution, row)?;
            let mut behavior = RolloutBehavior::default();
            snapshot
                .to_distribution(&self.policy().device())?
                .sample_behavior_into(mode, rng, &mut behavior)?;
            samples.push(OnPolicySample {
                decision: StatefulDecision {
                    action: behavior.actions.remove(0),
                    log_prob: Some(behavior.log_probs[0]),
                },
                latent: behavior.latents.remove(0),
                distribution: snapshot,
                value: values[row],
            });
        }
        Ok(samples)
    }
    fn decide_impl(
        &self,
        states: &[&TemporalActorState],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
        canonical: bool,
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "temporal decision batch count mismatch",
        )?;
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let distribution = self.policy_distribution_impl(states, observations, canonical)?;
        let mut decisions = Vec::with_capacity(states.len());
        for (row, rng) in rngs.iter_mut().enumerate() {
            let snapshot = DistributionSnapshot::from_distribution(&distribution, row)?;
            let mut sample = RolloutBehavior::default();
            snapshot
                .to_distribution(&self.policy().device())?
                .sample_behavior_into(mode, rng, &mut sample)?;
            decisions.push(StatefulDecision {
                action: sample.actions.remove(0),
                log_prob: Some(sample.log_probs[0]),
            });
        }
        Ok(decisions)
    }
    fn policy_distribution_impl(
        &self,
        states: &[&TemporalActorState],
        observations: &[&Observation],
        canonical: bool,
    ) -> Result<super::distributions::Distribution> {
        ensure(
            !states.is_empty() && states.len() == observations.len(),
            "temporal policy batch count mismatch",
        )?;
        for state in states {
            self.validate_state(state)?;
        }
        let width = self.policy().encoder.output_dim();
        let values = states
            .iter()
            .flat_map(|s| s.policy.iter().copied())
            .collect::<Vec<_>>();
        let features = Tensor::from_data(
            TensorData::new(values, [states.len(), width]),
            &self.policy().device(),
        );
        self.policy().distribution_from_features_using(
            features,
            ObservationBatch::from_refs(observations),
            |x| {
                if canonical {
                    super::canonical::mlp(&self.policy().head, x)
                } else {
                    self.policy().head.forward(x)
                }
            },
        )
    }
}

impl StatefulActor for TemporalActor {
    type State = TemporalActorState;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.spec().observation_dim)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.spec().action_space
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<Self::State> {
        Ok(TemporalActorState {
            policy: vec![0.; self.policy().encoder.output_dim()],
            value: self.value().map(|v| vec![0.; v.encoder.output_dim()]),
        })
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        ensure(
            state.policy.len() == self.policy().encoder.output_dim()
                && state.policy.iter().all(|x| x.is_finite())
                && match (self.value(), &state.value) {
                    (Some(model), Some(value)) => {
                        value.len() == model.encoder.output_dim()
                            && value.iter().all(|x| x.is_finite())
                    }
                    (None, None) => true,
                    _ => false,
                },
            "invalid temporal actor compound state",
        )
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        let mut states = self.assimilate_batch(&[prior], &[input])?;
        Ok(states.remove(0))
    }
    fn reconstruct_history(&self, history: &[HistoryInput<'_>]) -> Result<Self::State> {
        validate_history_inputs(history)?;
        if let Some(state) = self.reconstruct_encoded_inputs(history.iter().copied())? {
            return Ok(state);
        }
        replay_actor_history(self, history.iter().copied())
    }
    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let distribution = self.policy_distribution(&[state], &[observation])?;
        let mut behavior = RolloutBehavior::default();
        distribution.sample_behavior_into(mode, rng, &mut behavior)?;
        Ok(StatefulDecision {
            action: behavior.actions.remove(0),
            log_prob: Some(behavior.log_probs[0]),
        })
    }
    fn batch_compatible(&self, other: &Self) -> bool {
        self.revision == other.revision
            && (Rc::ptr_eq(&self.models, &other.models) || self.identity == other.identity)
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
    fn decide_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_impl(states, observations, mode, rngs, true)
    }
    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_impl(states, observations, mode, rngs, false)
    }
}
impl StatefulOnPolicyActor for TemporalActor {
    fn sample_on_policy(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<OnPolicySample> {
        let distribution = self.policy_distribution(&[state], &[observation])?;
        let snapshot = DistributionSnapshot::from_distribution(&distribution, 0)?;
        let mut behavior = RolloutBehavior::default();
        distribution.sample_behavior_into(mode, rng, &mut behavior)?;
        Ok(OnPolicySample {
            decision: StatefulDecision {
                action: behavior.actions.remove(0),
                log_prob: Some(behavior.log_probs[0]),
            },
            latent: behavior.latents.remove(0),
            distribution: snapshot,
            value: self.state_value(state, observation)?,
        })
    }
    fn state_value(&self, state: &Self::State, observation: &Observation) -> Result<Option<f32>> {
        Ok(self.state_values_batch(&[state], &[observation])?.remove(0))
    }
    fn sample_on_policy_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        self.sample_impl(states, observations, mode, rngs, false)
    }
    fn sample_on_policy_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<OnPolicySample>> {
        self.sample_impl(states, observations, mode, rngs, true)
    }
    fn state_values_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        self.state_values_impl(states, observations, false)
    }
    fn state_values_canonical_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
    ) -> Result<Vec<Option<f32>>> {
        self.state_values_impl(states, observations, true)
    }
}
pub(crate) fn assimilate_model<E: super::canonical::InferenceEncoder>(
    encoder: &E,
    spec: &HistorySpec,
    observation_only: bool,
    device: &Device,
    priors: &[&[f32]],
    inputs: &[HistoryInput<'_>],
    canonical: bool,
) -> Result<Vec<Vec<f32>>> {
    let width = encoder.input_dim();
    let hidden = encoder.output_dim();
    let batch = priors.len();
    ensure(
        batch == inputs.len(),
        "temporal history batch count mismatch",
    )?;
    let count = |width: usize| {
        batch
            .checked_mul(width)
            .ok_or_else(|| Error::Shape("temporal history batch overflow".into()))
    };
    let mut features = Vec::with_capacity(count(width)?);
    let mut states = Vec::with_capacity(count(hidden)?);
    for (prior, input) in priors.iter().zip(inputs) {
        let row = if observation_only {
            input
                .observation
                .validate(spec.observation_dim, &spec.action_space, false)?;
            input.observation.dense_values().to_vec()
        } else {
            spec.encode(*input)?
        };
        features.extend(row);
        states.extend_from_slice(prior);
    }
    let input = Tensor::from_data(TensorData::new(features, [batch, width]), device);
    let state = Tensor::from_data(TensorData::new(states, [batch, hidden]), device);
    let (_, next) = encoder.infer_step(input, state, canonical)?;
    let next = to_vec(next.without_autodiff())?;
    ensure(
        next.len() == batch * hidden && next.iter().all(|x| x.is_finite()),
        "invalid temporal behavior encoder output",
    )?;
    Ok(next.chunks_exact(hidden).map(<[f32]>::to_vec).collect())
}
pub(crate) fn validate_disjoint<P: Module, V: Module>(policy: &P, value: &V) -> Result<()> {
    struct Ids(HashSet<ParamId>);
    impl ModuleVisitor for Ids {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0.insert(p.id);
        }
    }
    let mut p = Ids(HashSet::new());
    let mut v = Ids(HashSet::new());
    policy.visit(&mut p);
    value.visit(&mut v);
    ensure(
        p.0.is_disjoint(&v.0),
        "temporal actor and critic parameters must be independent",
    )
}
impl Serialize for TemporalActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.record()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for TemporalActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(LearnerRecord::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
impl Snapshot for TemporalActor {
    fn validate_snapshot(&self) -> Result<()> {
        self.metadata()?.validate()?;
        temporal_models::validate_parameters(self.policy(), true)?;
        if let Some(value) = self.value() {
            temporal_models::validate_parameters(value, true)?;
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorMetadata {
    version: u32,
    revision: u64,
    policy: TemporalModelArchitecture,
    value: Option<TemporalModelArchitecture>,
}
impl ActorMetadata {
    fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1,
            "unsupported temporal actor snapshot version",
        )?;
        self.policy.validate()?;
        let outputs = match &self.policy.spec.action_space {
            ActionSpace::Discrete { count } => *count,
            space => space.dimensions(),
        };
        ensure(
            self.policy.output_dim == outputs,
            "temporal actor policy output mismatch",
        )?;
        if let Some(value) = &self.value {
            value.validate()?;
            ensure(
                value.spec == self.policy.spec && value.output_dim == 1,
                "temporal actor value architecture mismatch",
            )?;
        }
        Ok(())
    }
}

impl OnPolicyIdentity for TemporalActor {
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
        let mut audits = vec![audit_portable_component(
            "policy",
            &saved.policy,
            &native.policy,
            self.policy().encoder.gru.is_some(),
        )?];
        if let (Some(model), Some(saved), Some(native)) =
            (self.value(), &saved.value, &native.value)
        {
            audits.push(audit_portable_component(
                "value",
                saved,
                native,
                model.encoder.gru.is_some(),
            )?);
        }
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
        observation.validate(self.spec().observation_dim, self.action_space(), true)?;
        self.action_space()
            .check(action, observation.dense_mask())?;
        distribution.validate_for(self.action_space())?;
        let policy = self.policy();
        let features = matrix_on_device(
            state.policy.clone(),
            1,
            state.policy.len(),
            &policy.device(),
        );
        let current = policy.distribution_from_features(
            features,
            ObservationBatch::all(std::slice::from_ref(observation)),
        )?;
        let expected = DistributionSnapshot::from_distribution(&current, 0)?;
        ensure(
            recording_distribution_matches(&expected, distribution),
            "temporal recording distribution differs from reconstructed behavior",
        )?;
        let likelihood = to_vec(
            current.log_prob(std::slice::from_ref(action), &[latent.map(<[f32]>::to_vec)])?,
        )?[0];
        ensure(
            recording_number_matches(likelihood, log_prob),
            "temporal recording likelihood differs from reconstructed behavior",
        )?;
        let expected_value = self.state_value(state, observation)?;
        ensure(
            match (expected_value, value) {
                (Some(expected), Some(recorded)) => recording_number_matches(expected, recorded),
                (None, None) => true,
                _ => false,
            },
            "temporal recording baseline differs from reconstructed behavior",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{action::Action, stateful::PreviousInput};

    #[test]
    fn encoded_reconstruction_matches_every_scalar_chunk_boundary() {
        for space in [
            ActionSpace::Discrete { count: 2 },
            ActionSpace::Box {
                low: vec![-1., -2.],
                high: vec![1., 2.],
            },
        ] {
            let spec = HistorySpec::new(2, space.clone()).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(91);
            let mut policy = TemporalPolicy::new(spec.clone(), Some(17), &[3], &mut rng).unwrap();
            policy.encoder.gru.as_mut().unwrap().gru.clip = Some(0.2);
            policy.encoder.gru.as_mut().unwrap().gru.reset_after = false;
            let value = TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap();
            let actor = TemporalActor::new(policy, Some(value), 7).unwrap();
            let observations: Vec<_> = (0..260)
                .map(|t| {
                    Observation::dense(
                        vec![(t % 17) as f32 * 0.02 - 0.3, -(t as f32) * 0.01],
                        match space {
                            ActionSpace::Discrete { .. } => Some(vec![true, t % 2 == 0]),
                            _ => None,
                        },
                    )
                })
                .collect();
            let actions: Vec<_> = (0..259)
                .map(|t| match space {
                    ActionSpace::Discrete { .. } => Action::Discrete(usize::from(t % 2 == 0)),
                    _ => Action::Continuous(vec![-0., t as f32 * 0.005]),
                })
                .collect();
            let rewards: Vec<_> = (0..259).map(|t| (t % 11) as f32 * -0.02).collect();
            let mut expected = actor.initial_state().unwrap();
            for (time, observation) in observations.iter().enumerate() {
                expected = actor
                    .assimilate(
                        &expected,
                        HistoryInput {
                            observation,
                            previous: time.checked_sub(1).map(|t| PreviousInput {
                                factual: None,
                                action: &actions[t],
                                reward: rewards[t],
                            }),
                        },
                    )
                    .unwrap();
                if [0, 1, 127, 128, 129, 255, 259].contains(&time) {
                    let actual = actor
                        .reconstruct_encoded_history(ReconstructionHistory {
                            observations: &observations[..=time],
                            actions: &actions[..time],
                            rewards: crate::core::stateful::HistoryRewards::Totals(
                                &rewards[..time],
                            ),
                        })
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        serde_json::to_vec(&expected).unwrap(),
                        serde_json::to_vec(&actual).unwrap(),
                        "{space:?} at observation {time}",
                    );
                }
            }
            let inputs: Vec<_> = observations
                .iter()
                .enumerate()
                .map(|(time, observation)| HistoryInput {
                    observation,
                    previous: time.checked_sub(1).map(|t| PreviousInput {
                        factual: None,
                        action: &actions[t],
                        reward: rewards[t],
                    }),
                })
                .collect();
            #[cfg(feature = "probes")]
            {
                crate::profiling::reset();
                crate::profiling::set_enabled(true);
            }
            let actual = actor.reconstruct_history(&inputs).unwrap();
            #[cfg(feature = "probes")]
            {
                crate::profiling::set_enabled(false);
                assert!(crate::profiling::snapshot().iter().any(|sample| {
                    sample.phase == crate::profiling::Phase::EncodedLaneReconstruction
                }));
            }
            assert_eq!(
                serde_json::to_vec(&expected).unwrap(),
                serde_json::to_vec(&actual).unwrap()
            );
        }
    }
}
