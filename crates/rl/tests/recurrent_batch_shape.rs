#![cfg(feature = "neural")]

use burn::tensor::{Device, Tensor, TensorData};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::nn::{
    temporal::{GruEncoder, TemporalEncoder},
    to_vec,
};

#[test]
fn wide_recurrence_keeps_scalar_lane_bits_when_grouped() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let encoder = rl::nn::target::frozen_copy(&GruEncoder::new(2520, 4, &mut rng).unwrap());
    let input: Vec<f32> = (0..5040).map(|_| rng.random_range(-1.0..1.0)).collect();
    let hidden: Vec<f32> = (0..8).map(|_| rng.random_range(-0.7..0.7)).collect();
    let device = Device::flex();
    let (_, scalar) = encoder
        .step(
            Tensor::from_data(TensorData::new(input[..2520].to_vec(), [1, 2520]), &device),
            Tensor::from_data(TensorData::new(hidden[..4].to_vec(), [1, 4]), &device),
        )
        .unwrap();
    let (_, batch) = rl::nn::canonical::gru_step(
        &encoder,
        Tensor::from_data(TensorData::new(input, [2, 2520]), &device),
        Tensor::from_data(TensorData::new(hidden, [2, 4]), &device),
    )
    .unwrap();
    let scalar = to_vec(scalar).unwrap();
    let batch = to_vec(batch).unwrap();
    assert_eq!(
        scalar.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        batch[..4].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn canonical_equations_preserve_nonzero_optional_biases_reset_order_and_clip() {
    use burn::module::Param;
    for reset_after in [false, true] {
        for clip in [None, Some(0.2)] {
            let mut rng = ChaCha8Rng::seed_from_u64(19);
            let mut encoder = GruEncoder::new(513, 16, &mut rng).unwrap();
            encoder.gru.reset_after = reset_after;
            encoder.gru.clip = clip;
            encoder.gru.update_gate.input_transform.bias = None;
            for (index, gate) in [
                &mut encoder.gru.update_gate,
                &mut encoder.gru.reset_gate,
                &mut encoder.gru.new_gate,
            ]
            .into_iter()
            .enumerate()
            {
                gate.hidden_transform.bias = Some(Param::from_tensor(Tensor::from_data(
                    vec![(index as f32 + 1.) * 0.17; 16].as_slice(),
                    &rl::nn::device(),
                )));
            }
            let encoder = rl::nn::target::frozen_copy(&encoder);
            let device = Device::flex();
            let input: Vec<f32> = (0..8 * 513).map(|_| rng.random_range(-1.0..1.0)).collect();
            let prior: Vec<f32> = (0..8 * 16).map(|_| rng.random_range(-0.7..0.7)).collect();
            let (_, actual) = rl::nn::canonical::gru_step(
                &encoder,
                Tensor::from_data(TensorData::new(input.clone(), [8, 513]), &device),
                Tensor::from_data(TensorData::new(prior.clone(), [8, 16]), &device),
            )
            .unwrap();
            let actual = to_vec(actual).unwrap();
            for row in 0..8 {
                let (_, expected) = encoder
                    .step(
                        Tensor::from_data(
                            TensorData::new(input[row * 513..(row + 1) * 513].to_vec(), [1, 513]),
                            &device,
                        ),
                        Tensor::from_data(
                            TensorData::new(prior[row * 16..(row + 1) * 16].to_vec(), [1, 16]),
                            &device,
                        ),
                    )
                    .unwrap();
                let expected = to_vec(expected).unwrap();
                assert_eq!(
                    actual[row * 16..(row + 1) * 16]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}
