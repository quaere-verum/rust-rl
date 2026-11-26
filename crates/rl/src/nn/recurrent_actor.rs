//! Frozen GRU behavior policies with explicit post-observation host memory.
use super::{
    checkpoint::RecordBurnExt,
    network::{Mlp, Network},
    reconstruction::GruRepresentation,
    target::frozen_copy,
    temporal::{GruEncoder, GruQ, RecurrentQ, TemporalEncoder},
    to_vec,
};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        reconstruction::ReconstructibleActor,
        stateful::{HistoryInput, HistorySpec, StatefulActor, StatefulDecision},
    },
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    nn::activation::Activation,
    tensor::{Tensor, TensorData},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::rc::Rc;

/// An immutable model/exploration revision. Runtime memory is a separate value.
/// Standard sigmoid/tanh GRUs and arbitrary MLP head widths are supported.
#[derive(Debug)]
pub struct RecurrentActor {
    // The synchronous runtime never mutates a behavior model. Rc avoids atomic
    // ownership work and does not impose Send/Sync on Burn's local tensor engine.
    // Each lane still owns its state and RNG; training owns a separate model.
    model: Rc<GruQ>,
    model_identity: [u8; 32],
    spec: HistorySpec,
    epsilon: f32,
    revision: u64,
    parameter_generation: u64,
}

impl RecurrentActor {
    pub fn new(model: GruQ, spec: HistorySpec, epsilon: f32, revision: u64) -> Result<Self> {
        let mut actor = Self {
            model: Rc::new(frozen_copy(&model)),
            model_identity: [0; 32],
            spec,
            epsilon,
            revision,
            parameter_generation: 0,
        };
        actor.validate_snapshot()?;
        actor.model_identity = content_identity(actor.model(), &actor.spec)?;
        Ok(actor)
    }

    fn assimilate_impl(
        &self,
        priors: &[&Vec<f32>],
        inputs: &[HistoryInput<'_>],
        canonical: bool,
    ) -> Result<Vec<Vec<f32>>> {
        ensure(
            priors.len() == inputs.len(),
            "recurrent assimilation batch count mismatch",
        )?;
        if priors.is_empty() {
            return Ok(Vec::new());
        }
        for state in priors {
            self.validate_state(state)?;
        }
        super::temporal_actor::assimilate_model(
            &self.model.encoder,
            &self.spec,
            false,
            &self
                .model
                .encoder
                .gru
                .update_gate
                .input_transform
                .weight
                .val()
                .device(),
            &priors
                .iter()
                .map(|state| state.as_slice())
                .collect::<Vec<_>>(),
            inputs,
            canonical,
        )
    }
    fn decide_impl(
        &self,
        states: &[&Vec<f32>],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
        canonical: bool,
    ) -> Result<Vec<StatefulDecision>> {
        ensure(
            states.len() == observations.len() && states.len() == rngs.len(),
            "recurrent decision batch count mismatch",
        )?;
        if states.is_empty() {
            return Ok(Vec::new());
        }
        let batch = states.len();
        let hidden = self.model.hidden_dim();
        let mut values = Vec::with_capacity(
            batch
                .checked_mul(hidden)
                .ok_or_else(|| crate::Error::Shape("recurrent decision batch overflow".into()))?,
        );
        for (state, observation) in states.iter().zip(observations) {
            self.validate_state(state)?;
            observation.validate(self.spec.observation_dim, self.action_space(), true)?;
            values.extend_from_slice(state);
        }
        let device = self.model.head.layers[0].weight.val().device();
        let states = Tensor::from_data(TensorData::new(values, [batch, hidden]), &device);
        let values = to_vec(
            (if canonical {
                super::canonical::mlp(&self.model.head, states)
            } else {
                self.model.head.forward(states)
            })
            .without_autodiff(),
        )?;
        let actions = self.model.action_count();
        ensure(
            values.len() == batch * actions && values.iter().all(|x| x.is_finite()),
            "nonfinite or invalid recurrent batch Q values",
        )?;
        Ok(values
            .chunks_exact(actions)
            .zip(observations)
            .zip(rngs)
            .map(|((values, observation), rng)| self.choose(values, observation, mode, rng))
            .collect())
    }
    pub fn model(&self) -> &GruQ {
        &self.model
    }
    pub fn spec(&self) -> &HistorySpec {
        &self.spec
    }
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Stable, derived identity of the complete frozen representation, including
    /// the history schema and GRU controls. Computed once on construction/load;
    /// checkpoint-supplied identifiers and noncanonical packed bytes are not used.
    pub fn model_identity(&self) -> [u8; 32] {
        self.model_identity
    }

    pub fn parameter_generation(&self) -> u64 {
        self.parameter_generation
    }

    /// Parameter age is independent of publication/exploration revision. All u64
    /// generations are representable; reconstruction validates their chronology.
    pub fn with_parameter_generation(&self, generation: u64) -> Self {
        let mut actor = self.clone();
        actor.parameter_generation = generation;
        actor
    }

    /// Publish different behavior metadata for this already frozen model. The
    /// collector enforces increasing publication revisions; no model copy, tensor
    /// audit or identity hashing is needed for a settings-only change.
    pub fn with_behavior(&self, epsilon: f32, revision: u64) -> Result<Self> {
        ensure(
            epsilon.is_finite() && (0.0..=1.0).contains(&epsilon),
            "invalid recurrent actor exploration setting",
        )?;
        Ok(Self {
            model: Rc::clone(&self.model),
            model_identity: self.model_identity,
            spec: self.spec.clone(),
            epsilon,
            revision,
            parameter_generation: self.parameter_generation,
        })
    }

    fn metadata(&self) -> Result<ActorMetadata> {
        self.spec.validate()?;
        let ActionSpace::Discrete { count } = self.spec.action_space else {
            return Err(crate::Error::Config(
                "recurrent Q actors require discrete actions".into(),
            ));
        };
        let hidden = self.model.hidden_dim();
        ensure(
            hidden > 0
                && self.model.input_dim() == self.spec.input_dim()
                && self.model.action_count() == count
                && self.model.head.input_dim() == hidden
                && self.model.encoder.gru.d_hidden == hidden,
            "recurrent actor model/history dimensions mismatch",
        )?;
        let gru = &self.model.encoder.gru;
        ensure(
            matches!(&gru.gate_activation, Activation::Sigmoid(_))
                && matches!(&gru.hidden_activation, Activation::Tanh(_)),
            "recurrent actor requires sigmoid gates and tanh candidates",
        )?;
        let mut gate_bias = [[false; 2]; 3];
        for (index, gate) in [&gru.update_gate, &gru.reset_gate, &gru.new_gate]
            .into_iter()
            .enumerate()
        {
            ensure(
                gate.input_transform.weight.val().dims() == [self.spec.input_dim(), hidden]
                    && gate.hidden_transform.weight.val().dims() == [hidden, hidden],
                "invalid recurrent actor gate weights",
            )?;
            for (part, transform) in [&gate.input_transform, &gate.hidden_transform]
                .into_iter()
                .enumerate()
            {
                if let Some(bias) = &transform.bias {
                    ensure(
                        bias.val().dims() == [hidden],
                        "invalid recurrent actor gate bias",
                    )?;
                    gate_bias[index][part] = true;
                }
            }
        }
        ensure(
            !self.model.head.layers.is_empty(),
            "recurrent actor Q head cannot be empty",
        )?;
        let mut previous = hidden;
        let mut widths = Vec::new();
        let mut head_bias = Vec::new();
        for layer in &self.model.head.layers {
            let [input, output] = layer.weight.val().dims();
            ensure(
                input == previous && output > 0,
                "invalid recurrent actor head weights",
            )?;
            if let Some(bias) = &layer.bias {
                ensure(
                    bias.val().dims() == [output],
                    "invalid recurrent actor head bias",
                )?;
            }
            previous = output;
            widths.push(output);
            head_bias.push(layer.bias.is_some());
        }
        ensure(
            previous == count,
            "recurrent actor head action width mismatch",
        )?;
        widths.pop();
        let metadata = ActorMetadata {
            version: if self.parameter_generation == 0 { 1 } else { 2 },
            spec: self.spec.clone(),
            epsilon: self.epsilon,
            revision: self.revision,
            parameter_generation: self.parameter_generation,
            hidden_dim: hidden,
            head_hidden: widths,
            gate_bias,
            head_bias,
            reset_after: gru.reset_after,
            clip: gru.clip,
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn record(&self) -> Result<LearnerRecord> {
        self.validate_snapshot()?;
        let mut record = LearnerRecord::new("recurrent_actor", self.metadata()?)?;
        record.model("q", self.model.as_ref())?;
        Ok(record)
    }

    fn from_record(record: LearnerRecord) -> Result<Self> {
        record.check("recurrent_actor")?;
        ensure(
            record.models.len() == 1 && record.optimizers.is_empty(),
            "invalid recurrent actor model record",
        )?;
        let metadata: ActorMetadata = serde_json::from_value(record.metadata.clone())?;
        metadata.validate()?;
        let stored = record
            .models
            .get("q")
            .ok_or_else(|| crate::Error::State("recurrent actor is missing its Q model".into()))?;
        // Reject corrupted widths before allocating a template. Its raw parameter
        // payload cannot be larger than the complete stored container.
        ensure(
            metadata.parameter_count()? <= stored.bytes.len() / size_of::<f32>(),
            "recurrent actor architecture exceeds stored model size",
        )?;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut encoder =
            GruEncoder::new_inference(metadata.spec.input_dim(), metadata.hidden_dim, &mut rng)?;
        encoder.gru.reset_after = metadata.reset_after;
        encoder.gru.clip = metadata.clip;
        for (gate, biases) in [
            &mut encoder.gru.update_gate,
            &mut encoder.gru.reset_gate,
            &mut encoder.gru.new_gate,
        ]
        .into_iter()
        .zip(metadata.gate_bias)
        {
            if !biases[0] {
                gate.input_transform.bias = None;
            }
            if !biases[1] {
                gate.hidden_transform.bias = None;
            }
        }
        let ActionSpace::Discrete { count } = metadata.spec.action_space else {
            unreachable!("validated discrete space")
        };
        let mut head =
            Mlp::new_inference(metadata.hidden_dim, &metadata.head_hidden, count, &mut rng)?;
        for (layer, &bias) in head.layers.iter_mut().zip(&metadata.head_bias) {
            if !bias {
                layer.bias = None;
            }
        }
        let template = RecurrentQ::new(encoder, head)?;
        let model = record.load_model_inference("q", &template)?;
        let mut actor = Self {
            model: Rc::new(model),
            model_identity: [0; 32],
            spec: metadata.spec,
            epsilon: metadata.epsilon,
            revision: metadata.revision,
            parameter_generation: metadata.parameter_generation,
        };
        actor.validate_snapshot()?;
        actor.model_identity = content_identity(actor.model(), &actor.spec)?;
        Ok(actor)
    }
}

impl Clone for RecurrentActor {
    fn clone(&self) -> Self {
        Self {
            model: Rc::clone(&self.model),
            model_identity: self.model_identity,
            spec: self.spec.clone(),
            epsilon: self.epsilon,
            revision: self.revision,
            parameter_generation: self.parameter_generation,
        }
    }
}

impl StatefulActor for RecurrentActor {
    type State = Vec<f32>;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(self.spec.observation_dim)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.spec.action_space
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<Self::State> {
        Ok(vec![0.; self.model.hidden_dim()])
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        ensure(
            state.len() == self.model.hidden_dim() && state.iter().all(|v| v.is_finite()),
            "invalid recurrent actor hidden state",
        )
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        self.validate_state(prior)?;
        let values = self.spec.encode(input)?;
        let device = self
            .model
            .encoder
            .gru
            .update_gate
            .input_transform
            .weight
            .val()
            .device();
        let input = Tensor::from_data(TensorData::new(values, [1, self.spec.input_dim()]), &device);
        let hidden = Tensor::from_data(TensorData::new(prior.clone(), [1, prior.len()]), &device);
        let (_, next) = self.model.encoder.step(input, hidden)?;
        let next = to_vec(next.without_autodiff())?;
        self.validate_state(&next)?;
        Ok(next)
    }
    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        self.validate_state(state)?;
        observation.validate(self.spec.observation_dim, self.action_space(), true)?;
        let device = self.model.head.layers[0].weight.val().device();
        let hidden = Tensor::from_data(TensorData::new(state.clone(), [1, state.len()]), &device);
        let values = to_vec(self.model.head.forward(hidden).without_autodiff())?;
        ensure(
            values.len() == self.model.action_count() && values.iter().all(|v| v.is_finite()),
            "nonfinite or invalid recurrent actor Q values",
        )?;
        Ok(self.choose(&values, observation, mode, rng))
    }

    fn batch_compatible(&self, other: &Self) -> bool {
        self.revision == other.revision
            && self.parameter_generation == other.parameter_generation
            && self.epsilon.to_bits() == other.epsilon.to_bits()
            && self.spec == other.spec
            && (Rc::ptr_eq(&self.model, &other.model)
                || self.model_identity == other.model_identity)
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
    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<StatefulDecision>> {
        self.decide_impl(states, observations, mode, rngs, false)
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
}

impl ReconstructibleActor for RecurrentActor {
    type Model = GruRepresentation;
    fn representation(&self) -> Result<Self::Model> {
        GruRepresentation::new(self.clone(), self.parameter_generation)
    }
}

impl RecurrentActor {
    fn choose(
        &self,
        values: &[f32],
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> StatefulDecision {
        let valid =
            (0..values.len()).filter(|&a| observation.dense_mask().is_none_or(|mask| mask[a]));
        let action =
            if mode == ActMode::Sample && self.epsilon > 0. && rng.random::<f32>() < self.epsilon {
                let count = valid.clone().count();
                valid
                    .clone()
                    .nth(rng.random_range(0..count))
                    .expect("validated live action mask")
            } else {
                valid
                    .max_by(|&a, &b| values[a].total_cmp(&values[b]))
                    .expect("validated live action mask")
            };
        StatefulDecision {
            action: Action::Discrete(action),
            log_prob: None,
        }
    }
}

fn content_identity(model: &GruQ, spec: &HistorySpec) -> Result<[u8; 32]> {
    struct Content {
        digest: Sha256,
        error: Option<crate::Error>,
    }
    impl ModuleVisitor for Content {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let tensor = param.val().without_autodiff();
            self.digest.update((D as u64).to_le_bytes());
            for dimension in tensor.dims() {
                self.digest.update((dimension as u64).to_le_bytes());
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
        .update(b"rust-rl/recurrent-gru-q-content-v1\0");
    let schema = serde_json::to_vec(spec)?;
    content.digest.update((schema.len() as u64).to_le_bytes());
    content.digest.update(schema);
    for width in [
        model.input_dim(),
        model.hidden_dim(),
        model.action_count(),
        model.head.layers.len(),
    ] {
        content.digest.update((width as u64).to_le_bytes());
    }
    content
        .digest
        .update([u8::from(model.encoder.gru.reset_after)]);
    match model.encoder.gru.clip {
        Some(clip) => {
            content.digest.update([1]);
            content.digest.update(clip.to_bits().to_le_bytes());
        }
        None => content.digest.update([0]),
    }
    // Architecture/bias presence distinguishes otherwise equal traversal shapes.
    for gate in [
        &model.encoder.gru.update_gate,
        &model.encoder.gru.reset_gate,
        &model.encoder.gru.new_gate,
    ] {
        content.digest.update([
            u8::from(gate.input_transform.bias.is_some()),
            u8::from(gate.hidden_transform.bias.is_some()),
        ]);
    }
    for layer in &model.head.layers {
        content.digest.update([u8::from(layer.bias.is_some())]);
    }
    model.visit(&mut content);
    if let Some(error) = content.error {
        return Err(error);
    }
    Ok(content.digest.finalize().into())
}

impl Serialize for RecurrentActor {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.record()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RecurrentActor {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(LearnerRecord::deserialize(deserializer)?)
            .map_err(serde::de::Error::custom)
    }
}
impl Snapshot for RecurrentActor {
    fn validate_snapshot(&self) -> Result<()> {
        self.metadata()?;
        struct Values {
            valid: bool,
        }
        impl ModuleVisitor for Values {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                let value = param.val();
                self.valid &= !value.is_autodiff()
                    && !value.is_require_grad()
                    && to_vec(value).is_ok_and(|values| values.iter().all(|v| v.is_finite()));
            }
        }
        let mut values = Values { valid: true };
        self.model.visit(&mut values);
        ensure(
            values.valid,
            "recurrent actor snapshot has tracked or nonfinite parameters",
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorMetadata {
    version: u32,
    spec: HistorySpec,
    epsilon: f32,
    revision: u64,
    #[serde(default, skip_serializing_if = "zero_generation")]
    parameter_generation: u64,
    hidden_dim: usize,
    head_hidden: Vec<usize>,
    gate_bias: [[bool; 2]; 3],
    head_bias: Vec<bool>,
    reset_after: bool,
    clip: Option<f64>,
}
impl ActorMetadata {
    fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        ensure(
            self.version == if self.parameter_generation == 0 { 1 } else { 2 },
            "incompatible recurrent actor snapshot version",
        )?;
        ensure(
            matches!(self.spec.action_space, ActionSpace::Discrete { .. })
                && self.epsilon.is_finite()
                && (0.0..=1.0).contains(&self.epsilon)
                && self.hidden_dim > 0
                && self.head_hidden.iter().all(|&w| w > 0)
                && self.head_bias.len() == self.head_hidden.len() + 1
                && self.clip.is_none_or(|v| v.is_finite() && v > 0.),
            "invalid recurrent actor snapshot metadata",
        )?;
        self.parameter_count()?;
        Ok(())
    }
    fn parameter_count(&self) -> Result<usize> {
        let mut total = Some(0usize);
        let mut affine = |input: usize, output: usize, bias: bool| {
            total = total.and_then(|sum| {
                input
                    .checked_mul(output)
                    .and_then(|n| n.checked_add(if bias { output } else { 0 }))
                    .and_then(|n| sum.checked_add(n))
            });
        };
        for [input_bias, hidden_bias] in self.gate_bias {
            affine(self.spec.input_dim(), self.hidden_dim, input_bias);
            affine(self.hidden_dim, self.hidden_dim, hidden_bias);
        }
        let mut previous = self.hidden_dim;
        let ActionSpace::Discrete { count } = self.spec.action_space else {
            return Err(crate::Error::Config(
                "recurrent actor requires discrete actions".into(),
            ));
        };
        for (output, &bias) in self
            .head_hidden
            .iter()
            .copied()
            .chain(std::iter::once(count))
            .zip(&self.head_bias)
        {
            affine(previous, output, bias);
            previous = output;
        }
        total
            .filter(|&count| count <= isize::MAX as usize / size_of::<f32>())
            .ok_or_else(|| crate::Error::Shape("recurrent actor architecture size overflow".into()))
    }
}

fn zero_generation(generation: &u64) -> bool {
    *generation == 0
}
