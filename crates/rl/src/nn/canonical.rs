//! Canonical scalar-row CPU inference, using existing trainable parameters.
//!
//! The extra singleton dimension prevents Burn0.22's public matmul folding
//! independent rows into one GEMM. Flex broadcasts the unchanged weight buffer
//! and executes the scalar GEMV shape for each row. Training keeps its existing
//! differentiable sequence operations; this is an explicit inference capability.
use super::{
    network::Mlp,
    temporal::{GruEncoder, TemporalEncoder},
};
use crate::{Result, core::ensure};
use burn::{
    nn::Linear,
    tensor::{Device, Tensor, activation::relu},
};

pub(crate) trait InferenceEncoder: TemporalEncoder<State = Tensor<2>> {
    fn infer_step(
        &self,
        input: Tensor<2>,
        state: Tensor<2>,
        canonical: bool,
    ) -> Result<(Tensor<2>, Tensor<2>)>;
}
impl InferenceEncoder for GruEncoder {
    fn infer_step(
        &self,
        input: Tensor<2>,
        state: Tensor<2>,
        canonical: bool,
    ) -> Result<(Tensor<2>, Tensor<2>)> {
        if canonical {
            gru_step(self, input, state)
        } else {
            self.step(input, state)
        }
    }
}
impl InferenceEncoder for super::temporal_models::TemporalBackbone {
    fn infer_step(
        &self,
        input: Tensor<2>,
        state: Tensor<2>,
        canonical: bool,
    ) -> Result<(Tensor<2>, Tensor<2>)> {
        if let Some(gru) = &self.gru {
            gru.infer_step(input, state, canonical)
        } else {
            self.step(input, state)
        }
    }
}

/// A bulk tensor call with the scalar row's reduction order, not a host lane loop.
pub fn affine(input: Tensor<2>, linear: &Linear) -> Tensor<2> {
    grouped_affine(input, linear, 1)
}

fn grouped_affine(input: Tensor<2>, linear: &Linear, rows_per_lane: usize) -> Tensor<2> {
    let [rows, width] = input.dims();
    let output = linear.weight.val().dims()[1];
    let value = input
        .reshape([rows / rows_per_lane, 1, rows_per_lane, width])
        .matmul(linear.weight.val().reshape([1, 1, width, output]))
        .reshape([rows, output]);
    match &linear.bias {
        Some(bias) => value + bias.val().unsqueeze(),
        None => value,
    }
}

pub fn mlp(model: &Mlp, input: Tensor<2>) -> Tensor<2> {
    grouped_mlp(model, input, 1)
}

/// Keep each lane's original matrix row count, including entity and event MLPs
/// whose scalar actor already evaluates multiple rows together.
pub(crate) fn grouped_mlp(model: &Mlp, mut input: Tensor<2>, rows_per_lane: usize) -> Tensor<2> {
    for (index, layer) in model.layers.iter().enumerate() {
        input = grouped_affine(input, layer, rows_per_lane);
        if index + 1 < model.layers.len() {
            input = relu(input);
        }
    }
    input
}

pub(crate) fn grouped_flex_supported(device: &Device) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        *device == Device::flex() && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = device;
        false
    }
}

/// Same operation order and gate/clip options as the pinned Burn GRU equations.
pub fn gru_step(
    model: &GruEncoder,
    input: Tensor<2>,
    state: Tensor<2>,
) -> Result<(Tensor<2>, Tensor<2>)> {
    let [batch, width] = input.dims();
    ensure(
        batch > 0 && width == model.input_dim(),
        "canonical GRU input dimensions mismatch",
    )?;
    model.validate_state(&state, batch, &input.device())?;
    let gru = &model.gru;
    let update = gru.gate_activation.forward(
        affine(input.clone(), &gru.update_gate.input_transform)
            + affine(state.clone(), &gru.update_gate.hidden_transform),
    );
    let reset = gru.gate_activation.forward(
        affine(input.clone(), &gru.reset_gate.input_transform)
            + affine(state.clone(), &gru.reset_gate.hidden_transform),
    );
    let candidate = if gru.reset_after {
        affine(input, &gru.new_gate.input_transform)
            + reset * affine(state.clone(), &gru.new_gate.hidden_transform)
    } else {
        affine(input, &gru.new_gate.input_transform)
            + affine(state.clone() * reset, &gru.new_gate.hidden_transform)
    };
    let candidate = gru.hidden_activation.forward(candidate);
    let mut next = candidate * update.clone().neg().add_scalar(1.0) + update * state;
    if let Some(clip) = gru.clip {
        next = next.clamp(-clip, clip);
    }
    Ok((next.clone(), next))
}
