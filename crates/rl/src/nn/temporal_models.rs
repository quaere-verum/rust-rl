//! Shared temporal representation, selection, and bounded portable architecture codecs.
use super::{
    network::{Mlp, Network},
    temporal::{GruEncoder, TemporalEncoder, TemporalMasks, row_mask},
    temporal_batch::{EncodedHistory, encode_history_batch},
    to_vec,
};
use crate::{
    Error, Result,
    core::{checkpoint::LearnerRecord, ensure, stateful::HistorySpec},
    data::temporal::{SequenceBatch, SequenceOrigin, SequenceSample},
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    nn::activation::Activation,
    tensor::{Device, Int, Tensor, TensorData},
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// GRU memory or a parameter-free observation identity for memoryless controls.
#[derive(Module, Debug)]
pub struct TemporalBackbone {
    pub gru: Option<GruEncoder>,
    #[module(skip)]
    input: usize,
    #[module(skip)]
    output: usize,
}
impl TemporalBackbone {
    pub fn new(input: usize, hidden: Option<usize>, rng: &mut ChaCha8Rng) -> Result<Self> {
        Self::with_mode(input, hidden, rng, false)
    }
    pub(super) fn for_history(
        spec: &HistorySpec,
        hidden: Option<usize>,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        let input = if hidden.is_some() {
            spec.input_dim()
        } else {
            spec.observation_dim
        };
        Self::with_mode(input, hidden, rng, inference)
    }

    fn with_mode(
        input: usize,
        hidden: Option<usize>,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        ensure(
            input > 0 && hidden.is_none_or(|h| h > 0),
            "invalid temporal backbone dimensions",
        )?;
        let gru = hidden
            .map(|h| {
                if inference {
                    GruEncoder::new_inference(input, h, rng)
                } else {
                    GruEncoder::new(input, h, rng)
                }
            })
            .transpose()?;
        Ok(Self {
            gru,
            input,
            output: hidden.unwrap_or(input),
        })
    }
}
impl TemporalEncoder for TemporalBackbone {
    type State = Tensor<2>;
    fn input_dim(&self) -> usize {
        self.input
    }
    fn output_dim(&self) -> usize {
        self.output
    }
    fn initial_state(&self, batch: usize) -> Result<Self::State> {
        if let Some(gru) = &self.gru {
            return gru.initial_state(batch);
        }
        ensure(
            batch > 0
                && batch
                    .checked_mul(self.output)
                    .is_some_and(|n| n <= isize::MAX as usize),
            "invalid memoryless temporal state size",
        )?;
        Ok(Tensor::zeros([batch, self.output], &super::device()))
    }
    fn step(&self, input: Tensor<2>, state: Self::State) -> Result<(Tensor<2>, Self::State)> {
        if let Some(gru) = &self.gru {
            return gru.step(input, state);
        }
        ensure(
            input.dims()[0] > 0
                && input.dims()[1] == self.input
                && state.dims() == [input.dims()[0], self.output],
            "memoryless temporal step shape mismatch",
        )?;
        Ok((input.clone(), input))
    }
    fn unroll(
        &self,
        input: Tensor<3>,
        state: Self::State,
        masks: &TemporalMasks,
    ) -> Result<(Tensor<3>, Self::State)> {
        if let Some(gru) = &self.gru {
            return gru.unroll(input, state, masks);
        }
        masks.validate()?;
        let [batch, time, width] = input.dims();
        ensure(
            batch == masks.batch
                && time == masks.time
                && width == self.input
                && state.dims() == [batch, self.output],
            "memoryless temporal sequence shape mismatch",
        )?;
        let mut state = state.to_device(&input.device());
        let mut outputs = Vec::with_capacity(time);
        for t in 0..time {
            let valid: Vec<_> = (0..batch).map(|b| masks.valid[b * time + t]).collect();
            let row = input
                .clone()
                .slice([0..batch, t..t + 1, 0..width])
                .squeeze_dim(1);
            if valid.iter().all(|&x| x) {
                state = row;
            } else if valid.iter().any(|&x| x) {
                let mask = row_mask(valid, width, &input.device());
                state = state.mask_where(mask, row);
            }
            outputs.push(state.clone().unsqueeze_dim(1));
        }
        Ok((Tensor::cat(outputs, 1), state))
    }
}

/// A complete architecture, independent of parameter values or optimizer state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalModelArchitecture {
    pub version: u32,
    pub spec: HistorySpec,
    pub hidden_dim: Option<usize>,
    pub head_hidden: Vec<usize>,
    pub output_dim: usize,
    pub gate_bias: Option<[[bool; 2]; 3]>,
    pub head_bias: Vec<bool>,
    pub reset_after: bool,
    pub clip: Option<f64>,
    /// Version two permits candidate-action channels after history assimilation.
    /// Omission preserves every existing policy/value descriptor and identity hash.
    #[serde(default, skip_serializing_if = "zero")]
    pub head_input_extra: usize,
}
fn zero(value: &usize) -> bool {
    *value == 0
}
impl TemporalModelArchitecture {
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        ensure(
            ((self.version == 1 && self.head_input_extra == 0)
                || (self.version == 2 && self.head_input_extra > 0))
                && self.hidden_dim.is_none_or(|h| h > 0)
                && self.output_dim > 0
                && self.head_hidden.iter().all(|&h| h > 0)
                && self.head_bias.len() == self.head_hidden.len() + 1
                && self.hidden_dim.is_some() == self.gate_bias.is_some()
                && self.clip.is_none_or(|c| c.is_finite() && c > 0.)
                && (self.hidden_dim.is_some() || (self.clip.is_none() && self.reset_after)),
            "invalid temporal model architecture",
        )?;
        self.parameter_count(0)?;
        Ok(())
    }
    pub fn parameter_count(&self, extra: usize) -> Result<usize> {
        let mut count = Some(extra);
        let mut affine = |input: usize, output: usize, bias: bool| {
            count = count.and_then(|n| {
                input
                    .checked_mul(output)
                    .and_then(|w| w.checked_add(if bias { output } else { 0 }))
                    .and_then(|w| n.checked_add(w))
            });
        };
        if let Some(hidden) = self.hidden_dim {
            for [input_bias, hidden_bias] in self.gate_bias.unwrap_or([[false; 2]; 3]) {
                affine(self.spec.input_dim(), hidden, input_bias);
                affine(hidden, hidden, hidden_bias);
            }
        }
        let mut input = self
            .hidden_dim
            .unwrap_or(self.spec.observation_dim)
            .checked_add(self.head_input_extra)
            .ok_or_else(|| Error::Shape("temporal conditioned head width overflow".into()))?;
        for (output, bias) in self
            .head_hidden
            .iter()
            .copied()
            .chain(std::iter::once(self.output_dim))
            .zip(self.head_bias.iter().copied())
        {
            affine(input, output, bias);
            input = output;
        }
        count
            .filter(|&n| n <= isize::MAX as usize / size_of::<f32>())
            .ok_or_else(|| Error::Shape("temporal architecture size overflow".into()))
    }
    pub(crate) fn parts(
        &self,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<(TemporalBackbone, Mlp)> {
        self.validate()?;
        let mut encoder =
            TemporalBackbone::for_history(&self.spec, self.hidden_dim, rng, inference)?;
        if let Some(gru) = &mut encoder.gru {
            gru.gru.reset_after = self.reset_after;
            gru.gru.clip = self.clip;
            for (gate, bias) in [
                &mut gru.gru.update_gate,
                &mut gru.gru.reset_gate,
                &mut gru.gru.new_gate,
            ]
            .into_iter()
            .zip(self.gate_bias.expect("validated GRU controls"))
            {
                if !bias[0] {
                    gate.input_transform.bias = None;
                }
                if !bias[1] {
                    gate.hidden_transform.bias = None;
                }
            }
        }
        let mut head = if inference {
            Mlp::new_inference(
                encoder.output_dim() + self.head_input_extra,
                &self.head_hidden,
                self.output_dim,
                rng,
            )?
        } else {
            Mlp::new(
                encoder.output_dim() + self.head_input_extra,
                &self.head_hidden,
                self.output_dim,
                rng,
            )?
        };
        for (layer, &bias) in head.layers.iter_mut().zip(&self.head_bias) {
            if !bias {
                layer.bias = None;
            }
        }
        Ok((encoder, head))
    }
}

pub(crate) fn architecture(
    encoder: &TemporalBackbone,
    head: &Mlp,
    spec: &HistorySpec,
    observation_only: bool,
) -> Result<TemporalModelArchitecture> {
    architecture_conditioned(encoder, head, spec, observation_only, 0)
}

pub(crate) fn architecture_conditioned(
    encoder: &TemporalBackbone,
    head: &Mlp,
    spec: &HistorySpec,
    observation_only: bool,
    head_input_extra: usize,
) -> Result<TemporalModelArchitecture> {
    spec.validate()?;
    ensure(
        observation_only == encoder.gru.is_none()
            && encoder.input_dim()
                == if observation_only {
                    spec.observation_dim
                } else {
                    spec.input_dim()
                }
            && Some(head.input_dim()) == encoder.output_dim().checked_add(head_input_extra)
            && !head.layers.is_empty(),
        "temporal model dimensions mismatch",
    )?;
    let mut gate_bias = None;
    let mut reset_after = true;
    let mut clip = None;
    let hidden_dim = if let Some(encoder) = &encoder.gru {
        let hidden = encoder.output_dim();
        let gru = &encoder.gru;
        ensure(
            encoder.input_dim() == spec.input_dim()
                && hidden.checked_add(head_input_extra) == Some(head.input_dim())
                && gru.d_hidden == hidden
                && matches!(&gru.gate_activation, Activation::Sigmoid(_))
                && matches!(&gru.hidden_activation, Activation::Tanh(_)),
            "unsupported temporal GRU controls",
        )?;
        let mut bias = [[false; 2]; 3];
        for (i, gate) in [&gru.update_gate, &gru.reset_gate, &gru.new_gate]
            .into_iter()
            .enumerate()
        {
            for (j, (transform, input)) in [
                (&gate.input_transform, spec.input_dim()),
                (&gate.hidden_transform, hidden),
            ]
            .into_iter()
            .enumerate()
            {
                ensure(
                    transform.weight.val().dims() == [input, hidden]
                        && transform
                            .bias
                            .as_ref()
                            .is_none_or(|b| b.val().dims() == [hidden]),
                    "invalid temporal GRU weight shape",
                )?;
                bias[i][j] = transform.bias.is_some();
            }
        }
        gate_bias = Some(bias);
        reset_after = gru.reset_after;
        clip = gru.clip;
        Some(hidden)
    } else {
        None
    };
    let mut widths = Vec::new();
    let mut bias = Vec::new();
    let mut input = head.input_dim();
    for layer in &head.layers {
        let [i, o] = layer.weight.val().dims();
        ensure(
            i == input && o > 0 && layer.bias.as_ref().is_none_or(|b| b.val().dims() == [o]),
            "invalid temporal head shape",
        )?;
        widths.push(o);
        bias.push(layer.bias.is_some());
        input = o;
    }
    ensure(
        input == head.output_dim(),
        "temporal head declared output mismatch",
    )?;
    let output_dim = widths.pop().expect("nonempty head");
    let result = TemporalModelArchitecture {
        version: if head_input_extra == 0 { 1 } else { 2 },
        spec: spec.clone(),
        hidden_dim,
        head_hidden: widths,
        output_dim,
        gate_bias,
        head_bias: bias,
        reset_after,
        clip,
        head_input_extra,
    };
    result.validate()?;
    Ok(result)
}

pub(crate) fn module_device<M: Module>(model: &M) -> Device {
    struct First(Option<Device>);
    impl ModuleVisitor for First {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            if self.0.is_none() {
                self.0 = Some(param.val().device());
            }
        }
    }
    let mut first = First(None);
    model.visit(&mut first);
    first.0.unwrap_or_else(super::device)
}

pub(crate) fn validate_parameters<M: Module>(model: &M, frozen: bool) -> Result<()> {
    struct Audit {
        frozen: bool,
        valid: bool,
    }
    impl ModuleVisitor for Audit {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let t = p.val();
            self.valid &= (!self.frozen || (!t.is_autodiff() && !t.is_require_grad()))
                && to_vec(t).is_ok_and(|v| v.iter().all(|x| x.is_finite()));
        }
    }
    let mut audit = Audit {
        frozen,
        valid: true,
    };
    model.visit(&mut audit);
    ensure(
        audit.valid,
        "temporal model contains nonfinite or tracked frozen parameters",
    )
}

/// Reconstruct factual prefixes before gathering tokens; never unroll a selected subsequence.
pub(crate) fn selected_features<E: TemporalEncoder>(
    encoder: &E,
    batch: &SequenceBatch,
    spec: &HistorySpec,
    observation_only: bool,
    device: &Device,
    tokens: &[(usize, usize)],
) -> Result<Tensor<2>> {
    #[cfg(feature = "probes")]
    let _profile = crate::profiling::enter(crate::profiling::Phase::HistoryReconstruction);
    ensure(!tokens.is_empty(), "empty temporal model token selection")?;
    let encoded = {
        #[cfg(feature = "probes")]
        let _profile = crate::profiling::enter(crate::profiling::Phase::HistoryEncoding);
        encode_history_batch(batch, spec)?
    };
    ensure(
        batch
            .samples
            .iter()
            .all(|s| matches!(s.origin, SequenceOrigin::Reset)),
        "temporal model requires exact reset-origin prefixes",
    )?;
    ensure(
        tokens.iter().all(|&(s, t)| {
            batch
                .samples
                .get(s)
                .is_some_and(|sample| t < sample.observations.len())
        }),
        "invalid temporal model token selection",
    )?;
    PreparedHistory::new(
        encoded,
        observation_only.then_some(spec.observation_dim),
        device,
    )
    .features(encoder, &batch.samples, tokens)
}

/// Immutable inputs and masks can be shared by distinct encoders on the same
/// device. Each call still builds its own parameter-dependent recurrent graph.
pub(crate) struct PreparedHistory {
    inputs: Tensor<3>,
    masks: TemporalMasks,
    #[cfg(feature = "probes")]
    learning: Option<Vec<bool>>,
}

impl PreparedHistory {
    pub(crate) fn new(
        encoded: EncodedHistory,
        observation_width: Option<usize>,
        device: &Device,
    ) -> Self {
        let (width, values) = if let Some(width) = observation_width {
            let values = encoded
                .inputs
                .chunks_exact(encoded.input_dim)
                .flat_map(|row| row[..width].iter().copied())
                .collect();
            (width, values)
        } else {
            (encoded.input_dim, encoded.inputs)
        };
        Self {
            inputs: Tensor::from_data(
                TensorData::new(values, [encoded.batch, encoded.time, width]),
                device,
            ),
            masks: encoded.masks,
            #[cfg(feature = "probes")]
            learning: observation_width.is_none().then_some(encoded.learning),
        }
    }

    pub(crate) fn features<'a, E: TemporalEncoder>(
        &self,
        encoder: &E,
        samples: impl IntoIterator<Item = &'a SequenceSample>,
        tokens: &[(usize, usize)],
    ) -> Result<Tensor<2>> {
        self.features_on(encoder, samples, tokens, &self.inputs.device())
    }

    pub(crate) fn features_on<'a, E: TemporalEncoder>(
        &self,
        encoder: &E,
        samples: impl IntoIterator<Item = &'a SequenceSample>,
        tokens: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>> {
        self.features_from(
            encoder,
            encoder.initial_state(self.masks.batch)?,
            samples.into_iter().map(|sample| sample.learn_start),
            tokens,
            device,
        )
    }

    pub(crate) fn input_and_masks(&self) -> (Tensor<3>, &TemporalMasks) {
        (self.inputs.clone(), &self.masks)
    }

    pub(crate) fn unroll_features<'a, E: TemporalEncoder>(
        &self,
        encoder: &E,
        samples: impl IntoIterator<Item = &'a SequenceSample>,
    ) -> Result<Tensor<3>> {
        self.unroll_from(
            encoder,
            self.inputs.clone(),
            encoder.initial_state(self.masks.batch)?,
            samples.into_iter().map(|sample| sample.learn_start),
        )
    }

    pub(crate) fn features_from<E: TemporalEncoder>(
        &self,
        encoder: &E,
        initial: E::State,
        prefix_ends: impl IntoIterator<Item = usize>,
        tokens: &[(usize, usize)],
        device: &Device,
    ) -> Result<Tensor<2>> {
        let inputs = self.inputs.clone();
        let inputs = if device.is_autodiff() {
            inputs.autodiff()
        } else {
            inputs.without_autodiff()
        };
        Ok(gather_history_features(
            self.unroll_from(encoder, inputs, initial, prefix_ends)?,
            tokens,
        ))
    }

    fn unroll_from<E: TemporalEncoder>(
        &self,
        encoder: &E,
        inputs: Tensor<3>,
        initial: E::State,
        prefix_ends: impl IntoIterator<Item = usize>,
    ) -> Result<Tensor<3>> {
        let [batch, time, width] = inputs.dims();
        ensure(
            encoder.input_dim() == width,
            "temporal encoder history width mismatch",
        )?;
        let unroll = || {
            #[cfg(feature = "probes")]
            let _profile = crate::profiling::enter(crate::profiling::Phase::HistoryUnroll);
            encoder.unroll(inputs, initial, &self.masks)
        };
        #[cfg(feature = "probes")]
        let (features, _) = if let Some(learning) = &self.learning {
            crate::profiling::with_history_steps(
                self.masks.time,
                &self.masks.valid,
                learning,
                prefix_ends,
                unroll,
            )?
        } else {
            unroll()?
        };
        #[cfg(not(feature = "probes"))]
        let (features, _) = {
            let _ = prefix_ends;
            unroll()?
        };
        ensure(
            features.dims() == [batch, time, encoder.output_dim()],
            "temporal encoder output shape mismatch",
        )?;
        Ok(features)
    }
}

/// Read tokens in caller order from one current-model feature tape.
pub(crate) fn gather_history_features(features: Tensor<3>, tokens: &[(usize, usize)]) -> Tensor<2> {
    #[cfg(feature = "probes")]
    let _gather = crate::profiling::enter(crate::profiling::Phase::HistoryGather);
    let [batch, time, width] = features.dims();
    let indices: Vec<_> = tokens.iter().map(|&(s, t)| (s * time + t) as i64).collect();
    let indices =
        Tensor::<1, Int>::from_data(TensorData::new(indices, [tokens.len()]), &features.device());
    features.reshape([batch * time, width]).select(0, indices)
}

/// Audit raw tensors and bound declared allocations before constructing a restoration template.
pub(crate) fn validate_record_model(
    record: &LearnerRecord,
    name: &str,
    arch: &TemporalModelArchitecture,
    extra: usize,
) -> Result<()> {
    arch.validate()?;
    validate_record_model_count(record, name, arch.parameter_count(extra)?)
}

/// Common bounded raw-record audit for temporal models with additional heads.
pub(crate) fn validate_record_model_count(
    record: &LearnerRecord,
    name: &str,
    parameter_count: usize,
) -> Result<()> {
    use super::checkpoint::record_error;
    let model = record
        .models
        .get(name)
        .ok_or_else(|| Error::State(format!("missing temporal model {name}")))?;
    ensure(
        parameter_count <= model.bytes.len() / size_of::<f32>(),
        "temporal architecture exceeds stored model size",
    )?;
    let reader =
        burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(model.bytes.clone()))
            .map_err(record_error)?;
    for tensor in reader.into_tensors().map_err(record_error)? {
        let (_, dtype, shape, _, bytes) = tensor.into_parts().map_err(record_error)?;
        let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        ensure(
            dtype == burn_pack::DType::F32
                && elements.and_then(|n| n.checked_mul(4)) == Some(bytes.len()),
            "invalid temporal model tensor shape or dtype",
        )?;
        ensure(
            bytes
                .chunks_exact(4)
                .all(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]).is_finite()),
            "nonfinite temporal model checkpoint weight",
        )?;
    }
    Ok(())
}
