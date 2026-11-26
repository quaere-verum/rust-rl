use super::device;
use crate::{Result, core::ensure};
use burn::{
    module::{Module, Param},
    nn::Linear,
    tensor::{Device, Tensor, TensorData, activation::relu},
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Custom feedforward networks implement this interface and Burn's `Module`.
pub trait Network: Module + burn::module::ModuleDisplay {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    /// `[batch, input_dim] -> [batch, output_dim]`.
    fn forward(&self, input: Tensor<2>) -> Tensor<2>;
    /// Explicitly allow skipping an unused second actor inference through this network.
    fn actor_second_forward_skippable(&self) -> bool {
        false
    }
}

#[derive(Module, Debug)]
pub struct Mlp {
    pub layers: Vec<Linear>,
    input: usize,
    output: usize,
}
impl Mlp {
    pub(crate) fn set_constant_output(&mut self, value: f32) -> Result<()> {
        ensure(value.is_finite(), "critic initial value must be finite")?;
        let layer = self
            .layers
            .last_mut()
            .ok_or_else(|| crate::Error::Config("missing critic output head".into()))?;
        let bias = layer.bias.as_mut().ok_or_else(|| {
            crate::Error::Config("constant critic initialization requires an output bias".into())
        })?;
        *bias = bias.clone().map(|bias| {
            Tensor::full(bias.dims(), value, &bias.device())
                .set_require_grad(bias.is_require_grad())
        });
        layer.weight = layer.weight.clone().map(|weight| {
            Tensor::zeros(weight.dims(), &weight.device())
                .set_require_grad(weight.is_require_grad())
        });
        Ok(())
    }

    /// Xavier-uniform weights generated from the caller's owned RNG; zero biases.
    pub fn new(
        input: usize,
        hidden: &[usize],
        output: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::new_on_device(input, hidden, output, rng, &device())
    }
    /// Historical templates must never register trainable autodiff roots.
    pub(crate) fn new_inference(
        input: usize,
        hidden: &[usize],
        output: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Self::new_on_device(input, hidden, output, rng, &Device::flex())
    }
    pub(super) fn new_on_device(
        input: usize,
        hidden: &[usize],
        output: usize,
        rng: &mut ChaCha8Rng,
        device: &Device,
    ) -> Result<Self> {
        Self::new_initialized(input, hidden, output, rng, device, |rng, i, o| {
            let bound = (6.0 / (i + o) as f32).sqrt();
            (0..i * o)
                .map(|_| rng.random_range(-bound..bound))
                .collect()
        })
    }

    fn new_initialized(
        input: usize,
        hidden: &[usize],
        output: usize,
        rng: &mut ChaCha8Rng,
        device: &Device,
        mut weights: impl FnMut(&mut ChaCha8Rng, usize, usize) -> Vec<f32>,
    ) -> Result<Self> {
        ensure(
            input > 0 && output > 0 && hidden.iter().all(|&n| n > 0),
            "network widths must be positive",
        )?;
        let widths: Vec<_> = std::iter::once(input)
            .chain(hidden.iter().copied())
            .chain(std::iter::once(output))
            .collect();
        let layers = widths
            .windows(2)
            .map(|pair| {
                let (i, o) = (pair[0], pair[1]);
                Linear {
                    weight: Param::from_tensor(Tensor::from_data(
                        TensorData::new(weights(rng, i, o), [i, o]),
                        device,
                    )),
                    bias: Some(Param::from_tensor(Tensor::zeros([o], device))),
                }
            })
            .collect();
        Ok(Self {
            layers,
            input,
            output,
        })
    }
}
impl Network for Mlp {
    fn actor_second_forward_skippable(&self) -> bool {
        true
    }
    fn input_dim(&self) -> usize {
        self.input
    }
    fn output_dim(&self) -> usize {
        self.output
    }
    fn forward(&self, mut x: Tensor<2>) -> Tensor<2> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x);
            if i + 1 < self.layers.len() {
                x = relu(x);
            }
        }
        x
    }
}
