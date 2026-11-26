#![cfg(feature = "neural")]

use burn::{
    module::{Module, Param},
    nn::{GateController, Linear},
    tensor::{Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::nn::{
    device,
    network::Mlp,
    temporal::{GruEncoder, RecurrentQ, TemporalEncoder, TemporalMasks},
    to_vec,
};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-6,
            "element {i}: {actual} != {expected}"
        );
    }
}

fn scalar_encoder() -> GruEncoder {
    fn affine(weight: f32, bias: f32) -> Linear {
        Linear {
            weight: Param::from_tensor(Tensor::from_data([[weight]], &device())),
            bias: Some(Param::from_tensor(Tensor::from_data([bias], &device()))),
        }
    }
    let mut encoder = GruEncoder::new(1, 1, &mut ChaCha8Rng::seed_from_u64(13)).unwrap();
    encoder.gru.update_gate = GateController {
        input_transform: affine(0.4, 0.2),
        hidden_transform: affine(0.3, -0.1),
    };
    encoder.gru.reset_gate = GateController {
        input_transform: affine(-0.2, 0.05),
        hidden_transform: affine(0.1, 0.15),
    };
    encoder.gru.new_gate = GateController {
        input_transform: affine(0.7, -0.1),
        hidden_transform: affine(-0.5, 0.25),
    };
    encoder
}

// Independent scalar reference for PyTorch's reset-after GRU convention.
fn reference_step(input: f64, hidden: f64) -> f64 {
    let sigmoid = |x: f64| 1. / (1. + (-x).exp());
    let update = sigmoid(0.4 * input + 0.3 * hidden + 0.2 - 0.1);
    let reset = sigmoid(-0.2 * input + 0.1 * hidden + 0.05 + 0.15);
    let candidate = (0.7 * input - 0.1 + reset * (-0.5 * hidden + 0.25)).tanh();
    (1. - update) * candidate + update * hidden
}

#[test]
fn scalar_gru_matches_independent_reset_after_equations() {
    let encoder = scalar_encoder();
    let input = Tensor::from_data([[[0.8], [-0.3], [0.2]]], &device());
    let state = Tensor::from_data([[0.35]], &device());
    let (output, final_state) = encoder
        .unroll(input, state, &TemporalMasks::all_valid(1, 3).unwrap())
        .unwrap();
    let mut hidden = 0.35;
    let expected: Vec<_> = [0.8, -0.3, 0.2]
        .into_iter()
        .map(|x| {
            hidden = reference_step(x, hidden);
            hidden as f32
        })
        .collect();
    close(&to_vec(output).unwrap(), &expected);
    close(&to_vec(final_state).unwrap(), &[expected[2]]);
}

#[test]
fn repeated_step_equals_unroll_for_batches_lengths_and_nonzero_states() {
    for (batch, time) in [(1, 1), (1, 4), (3, 1), (3, 4)] {
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let encoder = GruEncoder::new(2, 3, &mut rng).unwrap();
        let input = Tensor::from_data(
            TensorData::new(
                (0..batch * time * 2)
                    .map(|i| (i as f32 - 3.) * 0.07)
                    .collect::<Vec<_>>(),
                [batch, time, 2],
            ),
            &device(),
        );
        let initial = Tensor::from_data(
            TensorData::new(
                (0..batch * 3).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
                [batch, 3],
            ),
            &device(),
        );
        let (unrolled, unrolled_state) = encoder
            .unroll(
                input.clone(),
                initial.clone(),
                &TemporalMasks::all_valid(batch, time).unwrap(),
            )
            .unwrap();
        let mut state = initial;
        let mut steps: Vec<Tensor<3>> = Vec::new();
        for t in 0..time {
            let (output, next) = encoder
                .step(
                    input
                        .clone()
                        .slice([0..batch, t..t + 1, 0..2])
                        .squeeze_dim(1),
                    state,
                )
                .unwrap();
            steps.push(output.unsqueeze_dim(1));
            state = next;
        }
        close(
            &to_vec(unrolled).unwrap(),
            &to_vec(Tensor::cat(steps, 1)).unwrap(),
        );
        close(&to_vec(unrolled_state).unwrap(), &to_vec(state).unwrap());
    }
}

#[test]
fn padding_preserves_state_and_resets_only_the_selected_live_lane() {
    let encoder = scalar_encoder();
    let values = [0.8, -0.3, f32::NAN, f32::INFINITY, -0.2, 0.4, 0.6, -0.1];
    let masks = TemporalMasks {
        batch: 2,
        time: 4,
        valid: vec![true, true, false, false, true, true, true, true],
        reset: vec![false, true, true, false, false, false, false, true],
        detach: vec![false; 8],
    };
    let input = Tensor::from_data(TensorData::new(values.to_vec(), [2, 4, 1]), &device());
    let initial = Tensor::from_data([[0.35], [-0.2]], &device());
    let (output, final_state) = encoder.unroll(input, initial, &masks).unwrap();
    let mut expected = Vec::new();
    let mut final_expected = Vec::new();
    for (lane, mut hidden) in [0.35, -0.2].into_iter().enumerate() {
        for t in 0..4 {
            let index = lane * 4 + t;
            if masks.valid[index] {
                if masks.reset[index] {
                    hidden = 0.;
                }
                hidden = reference_step(values[index] as f64, hidden);
            }
            expected.push(hidden as f32);
        }
        final_expected.push(hidden as f32);
    }
    close(&to_vec(output.clone()).unwrap(), &expected);
    close(&to_vec(final_state).unwrap(), &final_expected);
    let weight = encoder.gru.new_gate.input_transform.weight.val();
    let grads = output.sum().backward();
    assert!(to_vec(weight.grad(&grads).unwrap()).unwrap()[0].is_finite());
}

#[test]
fn last_timestep_loss_reaches_earlier_inputs_and_recurrent_weights() {
    let encoder = scalar_encoder();
    let input = Tensor::from_data([[[0.3], [0.7], [-0.4]]], &device()).require_grad();
    let recurrent_weight = encoder.gru.new_gate.hidden_transform.weight.val();
    let (output, _) = encoder
        .unroll(
            input.clone(),
            encoder.initial_state(1).unwrap(),
            &TemporalMasks::all_valid(1, 3).unwrap(),
        )
        .unwrap();
    let loss = output.slice([0..1, 2..3, 0..1]).square().mean();
    let grads = loss.backward();
    let input_grad = to_vec(input.grad(&grads).unwrap()).unwrap();
    assert!(input_grad.iter().all(|g| g.is_finite() && g.abs() > 1e-7));
    let weight_grad = to_vec(recurrent_weight.grad(&grads).unwrap()).unwrap();
    assert!(weight_grad.iter().all(|g| g.is_finite() && g.abs() > 1e-7));
}

#[test]
fn detach_blocks_only_the_selected_lanes_prefix_gradient() {
    let encoder = scalar_encoder();
    let input = Tensor::from_data([[[0.3], [0.7], [-0.4]], [[-0.2], [0.4], [0.6]]], &device())
        .require_grad();
    let mut masks = TemporalMasks::all_valid(2, 3).unwrap();
    masks.detach[1] = true;
    let (output, _) = encoder
        .unroll(input.clone(), encoder.initial_state(2).unwrap(), &masks)
        .unwrap();
    let (without_detach, _) = encoder
        .unroll(
            input.clone(),
            encoder.initial_state(2).unwrap(),
            &TemporalMasks::all_valid(2, 3).unwrap(),
        )
        .unwrap();
    close(
        &to_vec(output.clone()).unwrap(),
        &to_vec(without_detach).unwrap(),
    );
    let grads = output.slice([0..2, 2..3, 0..1]).sum().backward();
    let input_grad = to_vec(input.grad(&grads).unwrap()).unwrap();
    assert_eq!(input_grad[0], 0.);
    assert!(
        input_grad[1..]
            .iter()
            .all(|g| g.is_finite() && g.abs() > 1e-7)
    );
}

#[test]
fn detached_chunk_can_follow_a_consumed_backward_tape() {
    let encoder = scalar_encoder();
    let masks = TemporalMasks::all_valid(2, 1).unwrap();
    let (first, carried) = encoder
        .unroll(
            Tensor::from_data([[[0.3]], [[-0.2]]], &device()),
            encoder.initial_state(2).unwrap(),
            &masks,
        )
        .unwrap();
    let _ = first.sum().backward();
    let mut detached = masks;
    detached.detach.fill(true);
    let (second, _) = encoder
        .unroll(
            Tensor::from_data([[[0.7]], [[0.4]]], &device()),
            carried,
            &detached,
        )
        .unwrap();
    let _ = second.sum().backward();
}

#[test]
fn reset_blocks_the_selected_lanes_old_history_gradient() {
    let encoder = scalar_encoder();
    let input = Tensor::from_data([[[0.3], [0.7]], [[-0.2], [0.4]]], &device()).require_grad();
    let mut masks = TemporalMasks::all_valid(2, 2).unwrap();
    masks.reset[1] = true;
    let (output, _) = encoder
        .unroll(input.clone(), encoder.initial_state(2).unwrap(), &masks)
        .unwrap();
    let grads = output.slice([0..2, 1..2, 0..1]).sum().backward();
    let input_grad = to_vec(input.grad(&grads).unwrap()).unwrap();
    assert_eq!(input_grad[0], 0.);
    assert!(
        input_grad[1..]
            .iter()
            .all(|g| g.is_finite() && g.abs() > 1e-7)
    );
}

#[test]
fn caller_rng_initialization_and_plain_inference_are_reproducible() {
    let mut rng_a = ChaCha8Rng::seed_from_u64(29);
    let mut rng_b = ChaCha8Rng::seed_from_u64(29);
    let first = GruEncoder::new(2, 3, &mut rng_a).unwrap();
    let second = GruEncoder::new(2, 3, &mut rng_b).unwrap();
    let input = Tensor::from_data([[[0.3, 0.1], [0.7, -0.4]]], &device());
    let masks = TemporalMasks::all_valid(1, 2).unwrap();
    let (training, _) = first
        .unroll(input.clone(), first.initial_state(1).unwrap(), &masks)
        .unwrap();
    let plain = second.valid();
    let (inference, state) = plain
        .unroll(
            input.without_autodiff(),
            plain.initial_state(1).unwrap(),
            &masks,
        )
        .unwrap();
    assert!(!inference.is_autodiff());
    assert!(!state.is_autodiff());
    close(&to_vec(training).unwrap(), &to_vec(inference).unwrap());
    use rand::Rng;
    assert_eq!(rng_a.random::<u64>(), rng_b.random::<u64>());
}

#[test]
fn compositional_q_head_has_expected_shapes_and_receives_encoder_gradients() {
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let q = RecurrentQ::new(
        GruEncoder::new(2, 3, &mut rng).unwrap(),
        Mlp::new(3, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    assert_eq!(q.input_dim(), 2);
    assert_eq!(q.action_count(), 2);
    assert_eq!(q.hidden_dim(), 3);
    let input = Tensor::from_data([[[0.3, 0.1], [0.7, -0.4]]], &device());
    let (values, state) = q
        .unroll(
            input.clone(),
            q.initial_state(1).unwrap(),
            &TemporalMasks::all_valid(1, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(values.dims(), [1, 2, 2]);
    assert_eq!(state.dims(), [1, 3]);
    let (step_values, _) = q
        .step(
            input.slice([0..1, 0..1, 0..2]).squeeze_dim(1),
            q.initial_state(1).unwrap(),
        )
        .unwrap();
    assert_eq!(step_values.dims(), [1, 2]);
    let encoder_weight = q.encoder.gru.new_gate.input_transform.weight.val();
    let head_weight = q.head.layers[1].weight.val();
    let grads = values.square().sum().backward();
    for gradient in [
        encoder_weight.grad(&grads).unwrap(),
        head_weight.grad(&grads).unwrap(),
    ] {
        assert!(
            to_vec(gradient)
                .unwrap()
                .iter()
                .any(|g| g.is_finite() && g.abs() > 1e-7)
        );
    }
}

#[test]
fn recurrent_model_and_optimizer_checkpoint_continue_the_same_update() {
    use burn::optim::{AdamConfig, GradientsParams, ModuleOptimizer};
    use rl::{
        core::checkpoint::LearnerRecord,
        nn::{checkpoint::RecordBurnExt, temporal::GruQ},
    };
    fn new_q(seed: u64) -> GruQ {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        RecurrentQ::new(
            GruEncoder::new(1, 2, &mut rng).unwrap(),
            Mlp::new(2, &[], 2, &mut rng).unwrap(),
        )
        .unwrap()
    }
    fn values(q: &GruQ) -> Tensor<3> {
        q.unroll(
            Tensor::from_data([[[0.3], [0.7], [-0.4]]], &device()),
            q.initial_state(1).unwrap(),
            &TemporalMasks::all_valid(1, 3).unwrap(),
        )
        .unwrap()
        .0
    }
    fn update(q: &mut GruQ, optimizer: &mut ModuleOptimizer) {
        let loss = values(q).square().mean();
        let gradients = GradientsParams::from_grads(loss.backward(), q);
        *q = optimizer.step(0.001, q.clone(), gradients);
    }
    let mut q = new_q(43);
    let before = to_vec(q.encoder.gru.new_gate.input_transform.weight.val()).unwrap();
    let mut optimizer = AdamConfig::new().init();
    update(&mut q, &mut optimizer);
    assert_ne!(
        to_vec(q.encoder.gru.new_gate.input_transform.weight.val()).unwrap(),
        before
    );
    let mut record = LearnerRecord::new("temporal_nn", ()).unwrap();
    record.model("q", &q).unwrap();
    record.optimizer("optimizer", &optimizer).unwrap();
    let mut restored = record.load_model("q", &new_q(59)).unwrap();
    let mut restored_optimizer = record
        .load_optimizer_for_model("optimizer", "q", AdamConfig::new().init(), true)
        .unwrap();
    close(
        &to_vec(values(&q)).unwrap(),
        &to_vec(values(&restored)).unwrap(),
    );
    update(&mut q, &mut optimizer);
    update(&mut restored, &mut restored_optimizer);
    close(
        &to_vec(values(&q)).unwrap(),
        &to_vec(values(&restored)).unwrap(),
    );
}

#[test]
fn invalid_masks_states_features_and_head_widths_return_errors() {
    assert!(TemporalMasks::all_valid(0, 2).is_err());
    assert!(TemporalMasks::all_valid(1, 0).is_err());
    let mut overflow = TemporalMasks {
        batch: usize::MAX,
        time: 2,
        valid: vec![],
        reset: vec![],
        detach: vec![],
    };
    assert!(overflow.validate().is_err());
    overflow.batch = 1;
    overflow.time = 1;
    assert!(overflow.validate().is_err());
    let encoder = scalar_encoder();
    assert!(encoder.initial_state(0).is_err());
    assert!(
        encoder
            .step(
                Tensor::zeros([2, 2], &device()),
                encoder.initial_state(2).unwrap()
            )
            .is_err()
    );
    assert!(
        encoder
            .step(
                Tensor::zeros([2, 1], &device()),
                encoder.initial_state(1).unwrap()
            )
            .is_err()
    );
    assert!(
        encoder
            .step(
                Tensor::zeros([2, 1], &device()),
                Tensor::zeros([2, 2], &device())
            )
            .is_err()
    );
    assert!(
        encoder
            .unroll(
                Tensor::zeros([1, 2, 2], &device()),
                encoder.initial_state(1).unwrap(),
                &TemporalMasks::all_valid(1, 2).unwrap()
            )
            .is_err()
    );
    assert!(
        encoder
            .unroll(
                Tensor::zeros([1, 2, 1], &device()),
                encoder.initial_state(1).unwrap(),
                &TemporalMasks::all_valid(1, 1).unwrap()
            )
            .is_err()
    );
    let mut malformed = TemporalMasks::all_valid(1, 2).unwrap();
    malformed.reset.pop();
    assert!(
        encoder
            .unroll(
                Tensor::zeros([1, 2, 1], &device()),
                encoder.initial_state(1).unwrap(),
                &malformed
            )
            .is_err()
    );
    let mut rng = ChaCha8Rng::seed_from_u64(37);
    assert!(GruEncoder::new(0, 1, &mut rng).is_err());
    assert!(GruEncoder::new(1, 0, &mut rng).is_err());
    assert!(RecurrentQ::new(encoder, Mlp::new(2, &[], 2, &mut rng).unwrap()).is_err());
}
