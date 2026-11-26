use super::*;
use crate::{
    core::{
        action::{Action, ActionSpace},
        env::{Observation, ObservationBatch},
    },
    data::temporal::{BehaviorMetadata, SequenceConfig, SequenceSample, TemporalStep},
    nn::{
        device,
        distributions::Distribution,
        temporal::GruEncoder,
        temporal_models::selected_features,
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec,
    },
};
use burn::{
    grad_clipping::GradientClipping,
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn window(id: u64, start: usize, len: usize, ended: bool) -> SequenceSample {
    let end = start + len;
    SequenceSample {
        episode_id: id,
        trajectory_start: 0,
        history_start: 0,
        origin: Default::default(),
        previous: None,
        burn_start: start.saturating_sub(3),
        learn_start: start,
        observations: (0..=end)
            .map(|t| {
                Observation::dense(
                    vec![((17 * t + id as usize) % 47) as f32 * 0.03 - 0.7],
                    Some(vec![true, t % 3 == 0]),
                )
            })
            .collect(),
        steps: (0..end)
            .map(|t| TemporalStep {
                action: Action::Discrete(usize::from(t % 3 == 0)),
                reward: (((7 * t + id as usize) % 19) as f32 * 0.02 - 0.2).into(),
                terminated: false,
                truncated: ended && t + 1 == end,
                behavior: BehaviorMetadata {
                    revision: 2,
                    log_prob: Some(-0.3),
                },
            })
            .collect(),
        loss_mask: (0..end).map(|t| t >= start).collect(),
    }
}

fn gradients(encoder: &GruEncoder, features: Tensor<2>) -> (Vec<f32>, Vec<f32>) {
    let weights = (0..features.dims()[0])
        .map(|row| (row as f32 - 4.) * 0.125)
        .collect::<Vec<_>>();
    let weighted = features.square() * crate::nn::vector(&weights).unsqueeze_dim(1);
    let gradients = GradientsParams::from_grads(weighted.sum().backward(), encoder);
    struct Values<'a> {
        gradients: &'a GradientsParams,
        raw: Vec<f32>,
        clipped: Vec<f32>,
    }
    impl ModuleVisitor for Values<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
            let grad = self.gradients.get::<D>(param.id).unwrap();
            self.raw.extend(to_vec(grad.clone()).unwrap());
            self.clipped
                .extend(to_vec(GradientClipping::Norm(0.05).clip_gradient(grad)).unwrap());
        }
    }
    let mut values = Values {
        gradients: &gradients,
        raw: Vec::new(),
        clipped: Vec::new(),
    };
    encoder.visit(&mut values);
    (values.raw, values.clipped)
}

fn assert_bits(stage: &str, hidden: usize, reference: &[f32], optimized: &[f32]) {
    assert_eq!(reference.len(), optimized.len());
    let mut max_abs = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut max_ulp = 0;
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if value.is_sign_negative() {
            !bits
        } else {
            bits | 0x8000_0000
        }
    };
    for (&left, &right) in reference.iter().zip(optimized) {
        assert!(left.is_finite() && right.is_finite());
        let abs = (left as f64 - right as f64).abs();
        max_abs = max_abs.max(abs);
        max_relative =
            max_relative.max(abs / (left.abs().max(right.abs()) as f64).max(f64::MIN_POSITIVE));
        max_ulp = max_ulp.max(ordered(left).abs_diff(ordered(right)));
    }
    assert_eq!(
        max_ulp, 0,
        "{stage} hidden={hidden} max_abs={max_abs} max_relative={max_relative} max_ulp={max_ulp}",
    );
}

fn comparison_batch() -> SequenceBatch {
    SequenceBatch {
        config: SequenceConfig {
            learn_len: 4,
            burn_in: 3,
            n_step: 1,
        },
        samples: vec![
            window(7, 69, 2, true),
            window(8, 52, 4, false),
            window(7, 58, 4, false),
            window(7, 65, 4, false),
        ],
    }
}

#[test]
fn replay_coalescence_keeps_duplicate_rows_padding_and_bitwise_features() {
    if !compact_flex_supported() {
        return;
    }
    let mut batch = comparison_batch();
    batch.samples.push(batch.samples[2].clone());
    batch.samples.push(window(8, 25, 4, false));
    batch.validate().unwrap();
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let reference = encode_history_batch(&batch, &spec).unwrap();
    let cache = EpisodeInputCache::try_from_validated_replay(&batch, &spec)
        .unwrap()
        .unwrap();
    let indices: Vec<_> = (0..batch.samples.len()).collect();
    assert_eq!(reference, cache.pack(&indices).unwrap());
    let plan = EpisodeInferencePlan::try_new(&cache, &[31], &device())
        .unwrap()
        .unwrap();
    assert!(!plan.supports_hidden(64));
    assert_eq!(plan.work(), (2 * 72, 72 + 57));
    let history = PreparedHistory::new(reference, None, &device());
    let (input, masks) = history.input_and_masks();
    for hidden in [1, 5, 17, 31] {
        let encoder = GruEncoder::new(
            spec.input_dim(),
            hidden,
            &mut ChaCha8Rng::seed_from_u64(199),
        )
        .unwrap();
        let plain = encoder.valid();
        let (reference, _) = plain
            .unroll(
                input.clone().without_autodiff(),
                plain.initial_state(masks.batch).unwrap(),
                masks,
            )
            .unwrap();
        let optimized = plan.features(&encoder).unwrap();
        assert_eq!(reference.dims(), optimized.dims());
        assert_bits(
            "replay feature tape",
            hidden,
            &to_vec(reference).unwrap(),
            &to_vec(optimized).unwrap(),
        );
    }
    // Identical episode IDs do not permit sharing contradictory prefix facts.
    let reward = &mut batch.samples[2].steps[0].reward;
    *reward = (reward.total() + 1.).into();
    batch.validate().unwrap();
    assert!(
        EpisodeInputCache::try_from_validated_replay(&batch, &spec)
            .unwrap()
            .is_none()
    );
}

#[test]
fn coalesced_windows_preserve_incoming_states_forward_and_gradients() {
    if !compact_flex_supported() {
        return;
    }
    let batch = comparison_batch();
    batch.validate().unwrap();
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let cache = EpisodeInputCache::from_validated_on_policy(&batch, &spec).unwrap();
    let samples: Vec<_> = batch.samples.iter().collect();
    let plan = EpisodeWindowPlan::try_new(&cache, &samples, [64; 2], &device())
        .unwrap()
        .unwrap();
    let boundaries: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(row, _)| (row, 51))
        .collect();
    let tokens: Vec<_> = [2, 3, 0, 1]
        .into_iter()
        .flat_map(|row| (samples[row].learn_start..samples[row].steps.len()).map(move |t| (row, t)))
        .collect();
    for hidden in [1, 3, 5, 8, 16, 17, 31, 64] {
        let encoder = GruEncoder::new(
            spec.input_dim(),
            hidden,
            &mut ChaCha8Rng::seed_from_u64(107),
        )
        .unwrap();
        let prior =
            selected_features(&encoder, &batch, &spec, false, &device(), &boundaries).unwrap();
        assert_bits(
            "incoming",
            hidden,
            &to_vec(prior).unwrap(),
            &to_vec(plan.warm(&encoder).unwrap()).unwrap(),
        );
        let reference =
            selected_features(&encoder, &batch, &spec, false, &device(), &tokens).unwrap();
        let optimized = plan.features(&encoder, &tokens, &device()).unwrap();
        assert_bits(
            "features",
            hidden,
            &to_vec(reference.clone()).unwrap(),
            &to_vec(optimized.clone()).unwrap(),
        );
        let (reference_raw, reference_clipped) = gradients(&encoder, reference);
        let (optimized_raw, optimized_clipped) = gradients(&encoder, optimized);
        assert_ne!(reference_raw, reference_clipped);
        assert_bits("gradient", hidden, &reference_raw, &optimized_raw);
        assert_bits(
            "clipped gradient",
            hidden,
            &reference_clipped,
            &optimized_clipped,
        );
    }
}

#[test]
fn coalesced_windows_preserve_masked_policy_and_value_outputs() {
    if !compact_flex_supported() {
        return;
    }
    let batch = comparison_batch();
    batch.validate().unwrap();
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let cache = EpisodeInputCache::from_validated_on_policy(&batch, &spec).unwrap();
    let samples: Vec<_> = batch.samples.iter().collect();
    let plan = EpisodeWindowPlan::try_new(&cache, &samples, [8, 7], &device())
        .unwrap()
        .unwrap();
    let tokens = [(2, 58), (2, 59), (0, 69), (0, 70), (1, 53)];
    let observations: Vec<_> = tokens
        .iter()
        .map(|&(row, time)| &samples[row].observations[time])
        .collect();
    let actions: Vec<_> = tokens
        .iter()
        .map(|&(row, time)| samples[row].steps[time].action.clone())
        .collect();
    let mut rng = ChaCha8Rng::seed_from_u64(103);
    let policy = TemporalPolicy::new(spec.clone(), Some(8), &[5], &mut rng).unwrap();
    let value = TemporalValue::new(spec, Some(7), &[6], &mut rng).unwrap();
    let expected = policy.distribution(&batch, &tokens).unwrap();
    let actual = policy
        .distribution_from_features(
            plan.features(policy.encoder.gru.as_ref().unwrap(), &tokens, &device())
                .unwrap(),
            ObservationBatch::from_refs(&observations),
        )
        .unwrap();
    let logits = |distribution: &Distribution| match distribution {
        Distribution::Categorical { logits, masks } => {
            (to_vec(logits.clone()).unwrap(), masks.clone())
        }
        _ => panic!("categorical fixture"),
    };
    let (expected_logits, expected_masks) = logits(&expected);
    let (actual_logits, actual_masks) = logits(&actual);
    assert_eq!(expected_masks, actual_masks);
    assert_bits("masked logits", 8, &expected_logits, &actual_logits);
    let latents = vec![None; tokens.len()];
    let expected_log_prob = expected.log_prob(&actions, &latents).unwrap();
    let actual_log_prob = actual.log_prob(&actions, &latents).unwrap();
    assert_bits(
        "log probability",
        8,
        &to_vec(expected_log_prob).unwrap(),
        &to_vec(actual_log_prob).unwrap(),
    );
    let expected = value.values(&batch, &tokens).unwrap();
    let actual = value
        .values_from_features(
            plan.features(value.encoder.gru.as_ref().unwrap(), &tokens, &device())
                .unwrap(),
        )
        .unwrap();
    assert_bits(
        "values",
        7,
        &to_vec(expected).unwrap(),
        &to_vec(actual).unwrap(),
    );
}

#[test]
fn compact_warmup_matches_full_width_across_flex_matrix_tails() {
    for (lanes, episodes, observation_width, hidden) in [
        (3, 1, 1, 1),
        (5, 3, 4, 8),
        (8, 3, 13, 7),
        (17, 4, 13, 15),
        (33, 9, 60, 31),
        (64, 9, 60, 64),
    ] {
        let batch = SequenceBatch {
            config: SequenceConfig {
                learn_len: 4,
                burn_in: 3,
                n_step: 1,
            },
            samples: (0..lanes)
                .map(|row| {
                    let mut sample =
                        window((row % episodes) as u64, 12 + 4 * (row / episodes), 4, false);
                    for observation in &mut sample.observations {
                        let dense = observation.as_dense_mut().unwrap();
                        let first = dense.values[0];
                        dense.values.extend(
                            (1..observation_width)
                                .map(|channel| first * (channel as f32 * 0.002 - 0.1)),
                        );
                    }
                    sample
                })
                .collect(),
        };
        batch.validate().unwrap();
        let spec = HistorySpec::new(observation_width, ActionSpace::Discrete { count: 2 }).unwrap();
        let cache = EpisodeInputCache::from_validated_on_policy(&batch, &spec).unwrap();
        let samples: Vec<_> = batch.samples.iter().collect();
        let plan = EpisodeWindowPlan::try_new(&cache, &samples, [hidden; 2], &device()).unwrap();
        if !compact_flex_supported() {
            assert!(plan.is_none());
            continue;
        }
        let plan = plan.unwrap();
        assert!(
            EpisodeWindowPlan::try_new(&cache, &samples, [65, hidden], &device())
                .unwrap()
                .is_none()
        );
        if lanes == 17 {
            assert!(
                EpisodeWindowPlan::try_new(&cache, &samples, [hidden, 17], &device())
                    .unwrap()
                    .is_none()
            );
        }
        let encoder = GruEncoder::new(
            spec.input_dim(),
            hidden,
            &mut ChaCha8Rng::seed_from_u64(109),
        )
        .unwrap();
        let full = cache.pack(&(0..lanes).collect::<Vec<_>>()).unwrap();
        let plain = encoder.valid();
        let input = Tensor::from_data(
            TensorData::new(full.inputs, [full.batch, full.time, full.input_dim]),
            &Device::flex(),
        );
        let expected = warm_prefix(
            &plain,
            input,
            plain.initial_state(lanes).unwrap(),
            &vec![12; lanes],
            &full.masks,
        )
        .unwrap();
        assert_bits(
            "compact Flex matrix",
            hidden,
            &to_vec(expected).unwrap(),
            &to_vec(plan.warm(&encoder).unwrap()).unwrap(),
        );
    }
}
