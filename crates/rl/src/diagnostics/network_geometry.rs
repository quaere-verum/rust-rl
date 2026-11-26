//! Optional numerical inspection of an immutable built-in MLP.
//!
//! This module does not attach hooks to training or modify model parameters.
//! Directional derivatives are propagated explicitly through the stored affine
//! maps and ReLUs. A zero derivative is used at a ReLU kink; callers receive
//! a kink count so it cannot be mistaken for a classical derivative there.
#[cfg(feature = "neural")]
use crate::nn::network::{Mlp, Network};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct NumericalLayer {
    input: usize,
    output: usize,
    weights: Vec<f64>,
    biases: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct NumericalMlp {
    layers: Vec<NumericalLayer>,
    input: usize,
}
#[derive(Clone, Debug)]
pub struct LayerSignal {
    pub name: String,
    pub activation: Vec<f64>,
    pub tangent: Vec<f64>,
    pub relu_kinks: usize,
}
/// Activations shared by a bounded collection of directional derivatives.
#[derive(Clone, Debug)]
pub struct LayerDirections {
    pub activation: Vec<f64>,
    pub tangents: Vec<Vec<f64>>,
    pub relu_kinks: usize,
}
impl NumericalMlp {
    /// Read a detached numerical representation of the built-in network.
    #[cfg(feature = "neural")]
    pub fn from_model(model: &Mlp) -> Result<Self> {
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut expected = model.input_dim();
        for layer in &model.layers {
            let [input, output] = layer.weight.val().dims();
            if input != expected || output == 0 {
                return Err(Error::Shape("invalid numerical MLP layout".into()));
            }
            let weights = layer
                .weight
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| Error::Numerical(format!("MLP weight extraction: {e:?}")))?
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>();
            let biases = match &layer.bias {
                Some(bias) => bias
                    .val()
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| Error::Numerical(format!("MLP bias extraction: {e:?}")))?
                    .into_iter()
                    .map(f64::from)
                    .collect(),
                None => vec![0.; output],
            };
            if weights.len() != input * output
                || biases.len() != output
                || weights.iter().chain(&biases).any(|v| !v.is_finite())
            {
                return Err(Error::Numerical("invalid numerical MLP parameters".into()));
            }
            layers.push(NumericalLayer {
                input,
                output,
                weights,
                biases,
            });
            expected = output;
        }
        if expected != model.output_dim() || layers.is_empty() {
            return Err(Error::Shape("invalid numerical MLP output".into()));
        }
        Ok(Self {
            layers,
            input: model.input_dim(),
        })
    }
    pub fn input_dim(&self) -> usize {
        self.input
    }
    pub fn layer_widths(&self) -> Vec<usize> {
        std::iter::once(self.input)
            .chain(self.layers.iter().map(|layer| layer.output))
            .collect()
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    /// Owned numerical parameter storage, excluding allocator bookkeeping.
    pub fn retained_bytes(&self) -> usize {
        self.layers.capacity() * std::mem::size_of::<NumericalLayer>()
            + self
                .layers
                .iter()
                .map(|l| (l.weights.capacity() + l.biases.capacity()) * 8)
                .sum::<usize>()
    }
    pub fn forward_layers(&self, input: &[f64]) -> Result<Vec<Vec<f64>>> {
        let mut output = Vec::new();
        self.forward_layers_into(input, &mut output)?;
        Ok(output)
    }
    /// Evaluate every layer, reusing the supplied activation rows and capacities.
    ///
    /// On error, output is unspecified and may contain partial or previous
    /// activations. Callers must consume it only after a successful evaluation.
    pub fn forward_layers_into(&self, input: &[f64], output: &mut Vec<Vec<f64>>) -> Result<()> {
        if input.len() != self.input || input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Shape("invalid numerical input".into()));
        }
        // Fresh owned outputs retain the exact original capacities: reference
        // cache admission charges these rows and their outer vector directly.
        output.reserve_exact((self.layers.len() + 1).saturating_sub(output.len()));
        output.resize_with(self.layers.len() + 1, Vec::new);
        output[0].clear();
        output[0].reserve_exact(input.len());
        output[0].extend_from_slice(input);
        for (i, layer) in self.layers.iter().enumerate() {
            let (previous, next) = output.split_at_mut(i + 1);
            let x = &previous[i];
            let y = &mut next[0];
            y.clear();
            y.reserve_exact(layer.biases.len());
            y.extend_from_slice(&layer.biases);
            for (j, &value) in x.iter().enumerate() {
                for (k, target) in y.iter_mut().enumerate() {
                    *target += value * layer.weights[j * layer.output + k];
                }
            }
            if i + 1 < self.layers.len() {
                for value in y.iter_mut() {
                    *value = value.max(0.);
                }
            }
            if y.iter().any(|v| !v.is_finite()) {
                return Err(Error::Numerical("nonfinite MLP activation".into()));
            }
        }
        Ok(())
    }
    /// Trace one input tangent through every layer.
    ///
    /// The zero derivative convention at ReLU(0) is explicitly reported by
    /// `relu_kinks`; geometric conclusions there require one-sided inspection.
    pub fn trace(&self, input: &[f64], direction: &[f64]) -> Result<Vec<LayerSignal>> {
        if input.len() != self.input
            || direction.len() != self.input
            || input.iter().chain(direction).any(|v| !v.is_finite())
        {
            return Err(Error::Shape("invalid MLP tangent input".into()));
        }
        let mut values = input.to_vec();
        let mut tangent = direction.to_vec();
        let mut result = vec![LayerSignal {
            name: "Input".into(),
            activation: values.clone(),
            tangent: tangent.clone(),
            relu_kinks: 0,
        }];
        for (i, layer) in self.layers.iter().enumerate() {
            debug_assert_eq!(layer.input, values.len());
            let mut next = layer.biases.clone();
            let mut derivative = vec![0.; layer.output];
            for j in 0..layer.input {
                for k in 0..layer.output {
                    let weight = layer.weights[j * layer.output + k];
                    next[k] += values[j] * weight;
                    derivative[k] += tangent[j] * weight;
                }
            }
            let mut relu_kinks = 0;
            if i + 1 < self.layers.len() {
                for k in 0..layer.output {
                    if next[k] == 0. {
                        relu_kinks += 1;
                    }
                    if next[k] <= 0. {
                        next[k] = 0.;
                        derivative[k] = 0.;
                    }
                }
            }
            if next.iter().chain(&derivative).any(|v| !v.is_finite()) {
                return Err(Error::Numerical("nonfinite MLP tangent".into()));
            }
            values = next;
            tangent = derivative;
            result.push(LayerSignal {
                name: if i + 1 == self.layers.len() {
                    "Output".into()
                } else {
                    format!("Hidden {}", i + 1)
                },
                activation: values.clone(),
                tangent: tangent.clone(),
                relu_kinks,
            });
        }
        Ok(result)
    }
    /// The represented tangent columns at a selected post-activation layer.
    ///
    /// Layer zero denotes the input, and `layer_count()` denotes the output.
    pub fn tangent_columns(
        &self,
        input: &[f64],
        directions: &[Vec<f64>],
        layer: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if layer > self.layer_count() || directions.len() > 64 {
            return Err(Error::Shape("invalid layer or direction count".into()));
        }
        // Preserve the empty-column behavior of the independent-trace API.
        if directions.is_empty() {
            return Ok(Vec::new());
        }
        // Traverse every layer even when an early layer is requested: malformed
        // downstream activations/derivatives must still be rejected.
        let mut trace = self.trace_directions(input, directions)?;
        Ok(std::mem::take(&mut trace[layer].tangents))
    }
    /// Propagate all directions with one affine activation/ReLU traversal.
    /// Each derivative keeps the scalar trace's accumulation order and zero
    /// derivative convention. All output activations and tangents are checked.
    pub fn trace_directions(
        &self,
        input: &[f64],
        directions: &[Vec<f64>],
    ) -> Result<Vec<LayerDirections>> {
        if input.len() != self.input
            || directions.len() > 64
            || directions.iter().any(|d| d.len() != self.input)
            || input
                .iter()
                .chain(directions.iter().flatten())
                .any(|v| !v.is_finite())
        {
            return Err(Error::Shape("invalid MLP tangent input".into()));
        }
        let mut result = vec![LayerDirections {
            activation: input.to_vec(),
            tangents: directions.to_vec(),
            relu_kinks: 0,
        }];
        for (i, layer) in self.layers.iter().enumerate() {
            let previous = result.last().unwrap();
            let mut next = layer.biases.clone();
            let mut derivatives = vec![vec![0.; layer.output]; directions.len()];
            for j in 0..layer.input {
                for (k, target) in next.iter_mut().enumerate() {
                    let weight = layer.weights[j * layer.output + k];
                    *target += previous.activation[j] * weight;
                    for (derivative, tangent) in derivatives.iter_mut().zip(&previous.tangents) {
                        derivative[k] += tangent[j] * weight;
                    }
                }
            }
            let mut relu_kinks = 0;
            if i + 1 < self.layers.len() {
                for (k, value) in next.iter_mut().enumerate() {
                    if *value == 0. {
                        relu_kinks += 1;
                    }
                    if *value <= 0. {
                        *value = 0.;
                        for derivative in &mut derivatives {
                            derivative[k] = 0.;
                        }
                    }
                }
            }
            if next
                .iter()
                .chain(derivatives.iter().flatten())
                .any(|v| !v.is_finite())
            {
                return Err(Error::Numerical("nonfinite MLP tangent".into()));
            }
            result.push(LayerDirections {
                activation: next,
                tangents: derivatives,
                relu_kinks,
            });
        }
        Ok(result)
    }
}
#[cfg(all(test, feature = "neural"))]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn original_forward(net: &NumericalMlp, input: &[f64]) -> Result<Vec<Vec<f64>>> {
        if input.len() != net.input || input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Shape("invalid numerical input".into()));
        }
        let mut result = Vec::with_capacity(net.layers.len() + 1);
        result.push(input.to_vec());
        let mut x = input.to_vec();
        for (i, layer) in net.layers.iter().enumerate() {
            let mut y = layer.biases.clone();
            for (j, &value) in x.iter().enumerate() {
                for (k, target) in y.iter_mut().enumerate() {
                    *target += value * layer.weights[j * layer.output + k];
                }
            }
            if i + 1 < net.layers.len() {
                for value in &mut y {
                    *value = value.max(0.);
                }
            }
            if y.iter().any(|v| !v.is_finite()) {
                return Err(Error::Numerical("nonfinite MLP activation".into()));
            }
            x = y;
            result.push(x.clone());
        }
        Ok(result)
    }
    fn same_forward_bits(net: &NumericalMlp, input: &[f64], scratch: &mut Vec<Vec<f64>>) {
        let into = net
            .forward_layers_into(input, scratch)
            .map(|()| scratch.clone());
        for actual in [net.forward_layers(input), into] {
            match (actual, original_forward(net, input)) {
                (Ok(actual), Ok(expected)) => {
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert_eq!(
                            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(format!("{actual:?}"), format!("{expected:?}"))
                }
                (actual, expected) => {
                    panic!("different forward results: {actual:?} vs {expected:?}")
                }
            }
        }
    }

    #[test]
    fn allocating_forward_preserves_original_owned_capacities() {
        let mut rng = ChaCha8Rng::seed_from_u64(614);
        for width in [1, 2, 3, 5] {
            for hidden in [&[][..], &[3][..], &[3, 2][..], &[3, 2, 1][..]] {
                let net =
                    NumericalMlp::from_model(&Mlp::new(width, hidden, width, &mut rng).unwrap())
                        .unwrap();
                let input = vec![0.25; width];
                let actual = net.forward_layers(&input).unwrap();
                let original = original_forward(&net, &input).unwrap();
                assert_eq!(
                    actual.capacity(),
                    original.capacity(),
                    "outer activation storage affects reference-cache admission"
                );
                assert_eq!(
                    actual.iter().map(Vec::capacity).collect::<Vec<_>>(),
                    original.iter().map(Vec::capacity).collect::<Vec<_>>(),
                    "activation capacities affect reference-cache admission"
                );
                let charge = |rows: &Vec<Vec<f64>>| {
                    rows.capacity() * std::mem::size_of::<Vec<f64>>()
                        + rows.iter().map(|row| row.capacity() * 8).sum::<usize>()
                };
                assert_eq!(charge(&actual), charge(&original));
            }
        }
    }

    #[test]
    fn complete_forward_matches_frozen_original_bits_and_errors() {
        let mut rng = ChaCha8Rng::seed_from_u64(613);
        let mut scratch = vec![vec![123.; 100]; 8];
        for hidden in [&[8, 3, 8][..], &[3, 3][..], &[][..]] {
            let net = NumericalMlp::from_model(&Mlp::new(3, hidden, 2, &mut rng).unwrap()).unwrap();
            for input in [
                &[0.13, -0.28, 0.47][..],
                &[-0., 0., -0.][..],
                &[f64::NAN, 1., 2.][..],
                &[1.][..],
                &[0.13, -0.28, 0.47][..],
            ] {
                same_forward_bits(&net, input, &mut scratch);
            }
        }
        let net = NumericalMlp {
            input: 2,
            layers: vec![
                NumericalLayer {
                    input: 2,
                    output: 2,
                    weights: vec![1., 0., 0., 1.],
                    biases: vec![-0., -0.],
                },
                NumericalLayer {
                    input: 2,
                    output: 1,
                    weights: vec![f64::MAX, f64::MAX],
                    biases: vec![-0.],
                },
            ],
        };
        for input in [[-0., -0.], [0., 0.], [2., 2.], [-0., -0.]] {
            same_forward_bits(&net, &input, &mut scratch);
        }
        assert!(
            net.forward_layers(&[2., 2.]).is_err(),
            "a successful early row must not hide suffix overflow"
        );
    }

    #[test]
    fn shared_directions_match_independent_traces_at_every_layer() {
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let net = NumericalMlp::from_model(&Mlp::new(3, &[8, 6], 2, &mut rng).unwrap()).unwrap();
        let input = [0.13, -0.28, 0.47];
        let directions = vec![vec![1., 0., 0.], vec![0., -2., 0.], vec![0.7, -0.3, 0.2]];
        let shared = net.trace_directions(&input, &directions).unwrap();
        for (column, direction) in directions.iter().enumerate() {
            let oracle = net.trace(&input, direction).unwrap();
            let epsilon = 1e-5;
            let plus: Vec<_> = input
                .iter()
                .zip(direction)
                .map(|(x, d)| x + epsilon * d)
                .collect();
            let minus: Vec<_> = input
                .iter()
                .zip(direction)
                .map(|(x, d)| x - epsilon * d)
                .collect();
            let plus = net.forward_layers(&plus).unwrap();
            let minus = net.forward_layers(&minus).unwrap();
            for (layer, signal) in oracle.iter().enumerate() {
                assert_eq!(shared[layer].activation, signal.activation);
                assert_eq!(shared[layer].tangents[column], signal.tangent);
                assert_eq!(shared[layer].relu_kinks, signal.relu_kinks);
                for (j, tangent) in shared[layer].tangents[column].iter().enumerate() {
                    let finite = (plus[layer][j] - minus[layer][j]) / (2. * epsilon);
                    assert!((finite - tangent).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn shared_directions_keep_relu_zero_and_downstream_finite_checks() {
        let net = NumericalMlp {
            input: 2,
            layers: vec![
                NumericalLayer {
                    input: 2,
                    output: 2,
                    weights: vec![1., 0., 0., 1.],
                    biases: vec![0.; 2],
                },
                NumericalLayer {
                    input: 2,
                    output: 1,
                    weights: vec![f64::MAX, f64::MAX],
                    biases: vec![0.],
                },
            ],
        };
        let shared = net
            .trace_directions(&[0., -1.], &[vec![1., 1.], vec![-1., 2.]])
            .unwrap();
        assert_eq!(shared[1].relu_kinks, 1);
        assert_eq!(shared[1].tangents, vec![vec![0., 0.], vec![0., 0.]]);
        assert!(net.tangent_columns(&[2., 2.], &[vec![0., 0.]], 0).is_err());
        assert!(
            net.tangent_columns(&[0., 0.], &[vec![f64::NAN, 0.]], 0)
                .is_err()
        );
        assert!(net.tangent_columns(&[0., 0.], &[], 0).unwrap().is_empty());
    }

    #[test]
    fn tangent_matches_finite_difference_away_from_kinks() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let network = Mlp::new(3, &[8, 6], 2, &mut rng).unwrap();
        let numerical = NumericalMlp::from_model(&network).unwrap();
        let x = [0.13, -0.28, 0.47];
        let direction = [0.7, -0.3, 0.2];
        let epsilon = 1e-5;
        let plus = std::array::from_fn::<_, 3, _>(|i| x[i] + epsilon * direction[i]);
        let minus = std::array::from_fn::<_, 3, _>(|i| x[i] - epsilon * direction[i]);
        let a = numerical.forward_layers(&plus).unwrap();
        let b = numerical.forward_layers(&minus).unwrap();
        let exact = numerical.trace(&x, &direction).unwrap();
        for (index, state) in exact.iter().enumerate() {
            if state.relu_kinks > 0 {
                continue;
            }
            for j in 0..state.tangent.len() {
                let finite = (a[index][j] - b[index][j]) / (2. * epsilon);
                assert!((finite - state.tangent[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn identity_input_and_zero_direction() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let m = NumericalMlp::from_model(&Mlp::new(2, &[4], 1, &mut rng).unwrap()).unwrap();
        let result = m.trace(&[0.4, -0.2], &[0., 0.]).unwrap();
        assert_eq!(result[0].activation, vec![0.4, -0.2]);
        assert!(
            result
                .iter()
                .all(|layer| layer.tangent.iter().all(|v| *v == 0.))
        );
        assert_eq!(m.layer_widths(), vec![2, 4, 1]);
    }
}
