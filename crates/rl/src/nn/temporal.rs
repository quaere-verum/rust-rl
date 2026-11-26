//! Explicit neural memory and differentiable `[batch, time, feature]` sequences.
//!
//! State belongs to the caller. Reset and detach masks apply before assimilating
//! the corresponding input; padding preserves the previous state entirely.
use super::network::{Mlp, Network};
use crate::{Result, core::ensure};
use burn::{
    module::{Module, ModuleDisplay},
    nn::{GateController, Gru, activation::ActivationConfig},
    tensor::{Bool, Device, Tensor, TensorData},
};
use rand_chacha::ChaCha8Rng;

/// Row-major `[batch, time]` masks. Controls on padded entries are ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalMasks {
    pub batch: usize,
    pub time: usize,
    pub valid: Vec<bool>,
    pub reset: Vec<bool>,
    pub detach: Vec<bool>,
}

impl TemporalMasks {
    pub fn all_valid(batch: usize, time: usize) -> Result<Self> {
        let count = mask_count(batch, time)?;
        Ok(Self {
            batch,
            time,
            valid: vec![true; count],
            reset: vec![false; count],
            detach: vec![false; count],
        })
    }

    pub fn validate(&self) -> Result<()> {
        let count = mask_count(self.batch, self.time)?;
        ensure(
            self.valid.len() == count && self.reset.len() == count && self.detach.len() == count,
            "temporal masks must have batch * time entries",
        )
    }
}

fn mask_count(batch: usize, time: usize) -> Result<usize> {
    ensure(
        batch > 0 && time > 0,
        "temporal batch and time must be positive",
    )?;
    batch
        .checked_mul(time)
        .filter(|&count| count <= isize::MAX as usize)
        .ok_or_else(|| crate::Error::Shape("temporal mask size overflow".into()))
}

/// Reusable encoder with caller-owned memory and explicit sequence boundaries.
pub trait TemporalEncoder: Module + ModuleDisplay {
    type State: Clone;

    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn initial_state(&self, batch: usize) -> Result<Self::State>;
    /// Assimilate one input: `[batch, input] -> [batch, output]`.
    fn step(&self, input: Tensor<2>, state: Self::State) -> Result<(Tensor<2>, Self::State)>;
    /// Padding must preserve memory. Reset/detach occur before a valid step.
    fn unroll(
        &self,
        input: Tensor<3>,
        state: Self::State,
        masks: &TemporalMasks,
    ) -> Result<(Tensor<3>, Self::State)>;
}

/// Single-layer, unidirectional reset-after GRU with `[batch, hidden]` state.
#[derive(Module, Debug)]
pub struct GruEncoder {
    pub gru: Gru,
    input: usize,
    hidden: usize,
}

impl GruEncoder {
    /// Xavier-uniform weights from the caller's RNG, with zero gate biases.
    pub fn new(input: usize, hidden: usize, rng: &mut ChaCha8Rng) -> Result<Self> {
        Self::new_with_mode(input, hidden, rng, false)
    }

    /// Plain-device restoration templates never register autodiff roots.
    pub(crate) fn new_inference(input: usize, hidden: usize, rng: &mut ChaCha8Rng) -> Result<Self> {
        Self::new_with_mode(input, hidden, rng, true)
    }

    fn new_with_mode(
        input: usize,
        hidden: usize,
        rng: &mut ChaCha8Rng,
        inference: bool,
    ) -> Result<Self> {
        ensure(input > 0 && hidden > 0, "GRU widths must be positive")?;
        ensure(
            input.checked_mul(hidden).is_some()
                && hidden.checked_mul(hidden).is_some()
                && input.checked_add(hidden).is_some()
                && hidden.checked_add(hidden).is_some(),
            "GRU parameter size overflow",
        )?;
        let mut transform = |width| -> Result<burn::nn::Linear> {
            let mut network = if inference {
                Mlp::new_inference(width, &[], hidden, rng)?
            } else {
                Mlp::new(width, &[], hidden, rng)?
            };
            Ok(network.layers.remove(0))
        };
        let mut gate = || -> Result<GateController> {
            Ok(GateController {
                input_transform: transform(input)?,
                hidden_transform: transform(hidden)?,
            })
        };
        let device = if inference {
            Device::flex()
        } else {
            super::device()
        };
        let gru = Gru {
            update_gate: gate()?,
            reset_gate: gate()?,
            new_gate: gate()?,
            d_hidden: hidden,
            reset_after: true,
            gate_activation: ActivationConfig::Sigmoid.init(&device),
            hidden_activation: ActivationConfig::Tanh.init(&device),
            clip: None,
        };
        Ok(Self { gru, input, hidden })
    }

    fn device(&self) -> Device {
        self.gru.update_gate.input_transform.weight.val().device()
    }

    pub(crate) fn validate_state(
        &self,
        state: &Tensor<2>,
        batch: usize,
        device: &Device,
    ) -> Result<()> {
        ensure(
            batch > 0 && state.dims() == [batch, self.hidden],
            "GRU state must have shape [batch, hidden]",
        )?;
        ensure(
            state.device() == *device && self.device() == *device,
            "GRU input, state and parameters must use the same device",
        )
    }
}

impl TemporalEncoder for GruEncoder {
    type State = Tensor<2>;

    fn input_dim(&self) -> usize {
        self.input
    }

    fn output_dim(&self) -> usize {
        self.hidden
    }

    fn initial_state(&self, batch: usize) -> Result<Self::State> {
        ensure(
            batch > 0
                && batch
                    .checked_mul(self.hidden)
                    .is_some_and(|n| n <= isize::MAX as usize),
            "GRU initial state size must be positive and representable",
        )?;
        Ok(Tensor::zeros([batch, self.hidden], &self.device()))
    }

    fn step(&self, input: Tensor<2>, state: Self::State) -> Result<(Tensor<2>, Self::State)> {
        let [batch, width] = input.dims();
        ensure(width == self.input, "GRU input width mismatch")?;
        self.validate_state(&state, batch, &input.device())?;
        let next = self
            .gru
            .forward(input.unsqueeze_dim(1), Some(state))
            .squeeze_dim(1);
        Ok((next.clone(), next))
    }

    fn unroll(
        &self,
        input: Tensor<3>,
        mut state: Self::State,
        masks: &TemporalMasks,
    ) -> Result<(Tensor<3>, Self::State)> {
        masks.validate()?;
        let [batch, time, width] = input.dims();
        ensure(
            batch == masks.batch && time == masks.time && width == self.input,
            "GRU sequence and temporal mask dimensions mismatch",
        )?;
        self.validate_state(&state, batch, &input.device())?;
        let mut outputs = Vec::with_capacity(time);
        for t in 0..time {
            let valid: Vec<_> = (0..batch).map(|b| masks.valid[b * time + t]).collect();
            if valid.iter().all(|v| !v) {
                outputs.push(state.clone().unsqueeze_dim(1));
                continue;
            }
            #[cfg(feature = "probes")]
            let _step_profile = crate::profiling::enter_history_step(t);
            let reset: Vec<_> = (0..batch)
                .map(|b| valid[b] && masks.reset[b * time + t])
                .collect();
            let detach: Vec<_> = (0..batch)
                .map(|b| valid[b] && masks.detach[b * time + t])
                .collect();
            let mut active_state = detach_rows(state.clone(), &detach);
            if reset.iter().any(|&reset| reset) {
                active_state = if reset.iter().all(|&reset| reset) {
                    Tensor::zeros([batch, self.hidden], &state.device())
                } else {
                    active_state.mask_fill(row_mask(reset, self.hidden, &state.device()), 0.)
                };
            }
            let mut input_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..width])
                .squeeze_dim(1);
            let all_valid = valid.iter().all(|&v| v);
            if !all_valid {
                // Mask before matmul: zero * NaN in a backward pass is still NaN.
                let invalid: Vec<_> = valid.iter().map(|v| !v).collect();
                input_t = input_t.mask_fill(row_mask(invalid.clone(), width, &input.device()), 0.);
                active_state =
                    active_state.mask_fill(row_mask(invalid, self.hidden, &state.device()), 0.);
            }
            let (_, next) = self.step(input_t, active_state)?;
            state = if all_valid {
                next
            } else {
                state.mask_where(row_mask(valid, self.hidden, &next.device()), next)
            };
            outputs.push(state.clone().unsqueeze_dim(1));
        }
        Ok((Tensor::cat(outputs, 1), state))
    }
}

pub(super) fn row_mask(rows: Vec<bool>, width: usize, device: &Device) -> Tensor<2, Bool> {
    let batch = rows.len();
    Tensor::from_data(TensorData::new(rows, [batch, 1]), device).repeat_dim(1, width)
}

fn detach_rows(state: Tensor<2>, detach: &[bool]) -> Tensor<2> {
    if detach.iter().all(|&detach| detach) {
        // An actual lineage break also permits a new backward after a previous
        // chunk's tape has been consumed. A masked all-true branch would not.
        state.without_autodiff()
    } else if detach.iter().any(|&detach| detach) {
        let [_, hidden] = state.dims();
        let constant = state.clone().without_autodiff();
        let mask = row_mask(detach.to_vec(), hidden, &state.device());
        state.mask_where(mask, constant)
    } else {
        state
    }
}

/// A temporal feature encoder followed by any existing feedforward Q head.
#[derive(Module, Debug)]
pub struct RecurrentQ<E: TemporalEncoder, H: Network = Mlp> {
    pub encoder: E,
    pub head: H,
}

pub type GruQ = RecurrentQ<GruEncoder, Mlp>;

impl<E: TemporalEncoder> RecurrentQ<E, Mlp> {
    pub fn with_initialization(
        mut self,
        initialization: super::critics::CriticInitialization,
    ) -> Result<Self> {
        initialization.apply(&mut self.head)?;
        Ok(self)
    }
}

impl<E: TemporalEncoder, H: Network> RecurrentQ<E, H> {
    pub fn new(encoder: E, head: H) -> Result<Self> {
        ensure(
            encoder.input_dim() > 0
                && encoder.output_dim() > 0
                && head.input_dim() == encoder.output_dim()
                && head.output_dim() > 0,
            "recurrent Q encoder and head dimensions mismatch",
        )?;
        Ok(Self { encoder, head })
    }

    pub fn input_dim(&self) -> usize {
        self.encoder.input_dim()
    }

    pub fn hidden_dim(&self) -> usize {
        self.encoder.output_dim()
    }

    pub fn action_count(&self) -> usize {
        self.head.output_dim()
    }

    pub fn initial_state(&self, batch: usize) -> Result<E::State> {
        self.encoder.initial_state(batch)
    }

    pub fn step(&self, input: Tensor<2>, state: E::State) -> Result<(Tensor<2>, E::State)> {
        let [batch, width] = input.dims();
        ensure(
            batch > 0 && width == self.input_dim(),
            "recurrent Q input shape mismatch",
        )?;
        let (encoded, next) = self.encoder.step(input, state)?;
        ensure(
            encoded.dims() == [batch, self.head.input_dim()],
            "recurrent Q encoded shape mismatch",
        )?;
        let values = self.head.forward(encoded);
        ensure(
            values.dims() == [batch, self.action_count()],
            "recurrent Q output shape mismatch",
        )?;
        Ok((values, next))
    }

    pub fn unroll(
        &self,
        input: Tensor<3>,
        state: E::State,
        masks: &TemporalMasks,
    ) -> Result<(Tensor<3>, E::State)> {
        masks.validate()?;
        let [batch, time, width] = input.dims();
        ensure(
            batch == masks.batch && time == masks.time && width == self.input_dim(),
            "recurrent Q sequence shape mismatch",
        )?;
        let (encoded, next) = self.encoder.unroll(input, state, masks)?;
        ensure(
            encoded.dims() == [batch, time, self.head.input_dim()],
            "recurrent Q encoded sequence shape mismatch",
        )?;
        Ok((self.values_from_history_features(encoded)?, next))
    }

    pub(crate) fn values_from_history_features(&self, encoded: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, time, hidden] = encoded.dims();
        ensure(
            hidden == self.head.input_dim(),
            "recurrent Q feature width mismatch",
        )?;
        let values = self
            .head
            .forward(encoded.reshape([batch * time, self.hidden_dim()]));
        ensure(
            values.dims() == [batch * time, self.action_count()],
            "recurrent Q output sequence shape mismatch",
        )?;
        Ok(values.reshape([batch, time, self.action_count()]))
    }
}
