#![cfg(feature = "neural")]

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    core::actor::{EpisodeBehavior, RolloutBehavior},
    nn::{distributions::Distribution, matrix, to_vec},
    prelude::*,
};
#[test]
fn custom_categorical_distribution_applies_its_own_mask() {
    let d = Distribution::Categorical {
        logits: matrix(vec![100., 0.], 1, 2),
        masks: vec![false, true],
    };
    let p = d
        .log_prob(&[Action::Discrete(1)], &[None])
        .unwrap()
        .into_scalar::<f32>();
    assert_eq!(p, 0.);
}

fn assert_near(actual: f32, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual as f64 - expected).abs() <= 1e-5 * (1. + expected.abs()),
        "actual {actual}, expected {expected}"
    );
}

#[test]
fn masked_host_likelihood_matches_scalar_and_training_normalization() {
    let logits = [100., 100., -100., 99., -100., -99., 100., -98.];
    let masks = [true, true, false, true, true, true, false, true];
    let dist = Distribution::Categorical {
        logits: matrix(logits.to_vec(), 2, 4),
        masks: masks.to_vec(),
    };
    for mode in [ActMode::Sample, ActMode::Deterministic] {
        let mut out = RolloutBehavior::default();
        dist.sample_behavior_into(mode, &mut ChaCha8Rng::seed_from_u64(17), &mut out)
            .unwrap();
        let training = to_vec(dist.log_prob(&out.actions, &out.latents).unwrap()).unwrap();
        for (row, &training_lp) in training.iter().enumerate() {
            let Action::Discrete(chosen) = out.actions[row] else {
                panic!("discrete")
            };
            assert!(masks[row * 4 + chosen]);
            let max = (0..4)
                .filter(|&j| masks[row * 4 + j])
                .map(|j| logits[row * 4 + j] as f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = (0..4)
                .filter(|&j| masks[row * 4 + j])
                .map(|j| (logits[row * 4 + j] as f64 - max).exp())
                .sum();
            let expected = logits[row * 4 + chosen] as f64 - max - sum.ln();
            assert_near(out.log_probs[row], expected);
            assert_near(training_lp, expected);
            assert!(out.latents[row].is_none());
        }
    }
}

#[test]
fn gaussian_host_likelihood_matches_training_at_saturated_latents() {
    let means = [20., -20., 0.3, -0.7];
    let log_stds = [-1., -2., 0.2, -0.4];
    for bounds in [None, Some((vec![-3., 2.], vec![7., 5.]))] {
        let dist = Distribution::Gaussian {
            mean: matrix(means.to_vec(), 2, 2),
            log_std: matrix(log_stds.to_vec(), 2, 2),
            bounds: bounds.clone(),
        };
        for mode in [ActMode::Sample, ActMode::Deterministic] {
            let mut out = RolloutBehavior::default();
            dist.sample_behavior_into(mode, &mut ChaCha8Rng::seed_from_u64(53), &mut out)
                .unwrap();
            let training = to_vec(dist.log_prob(&out.actions, &out.latents).unwrap()).unwrap();
            for (row, &training_lp) in training.iter().enumerate() {
                let Action::Continuous(action) = &out.actions[row] else {
                    panic!("continuous")
                };
                assert_eq!(out.latents[row].is_some(), bounds.is_some());
                let latent = out.latents[row].as_deref().unwrap_or(action);
                let expected: f64 = (0..2)
                    .map(|j| {
                        let k = row * 2 + j;
                        let z = latent[j] as f64;
                        let scale_z = (z - means[k] as f64) / (log_stds[k] as f64).exp();
                        let mut lp = -0.5 * scale_z * scale_z
                            - log_stds[k] as f64
                            - 0.5 * std::f64::consts::TAU.ln();
                        if let Some((lo, hi)) = &bounds {
                            // Independent absolute-value identity for log(sech(z)^2).
                            let jacobian = 2.
                                * (std::f64::consts::LN_2
                                    - z.abs()
                                    - (-2_f64 * z.abs()).exp().ln_1p());
                            lp -= ((hi[j] as f64 - lo[j] as f64) * 0.5).ln() + jacobian;
                        }
                        lp
                    })
                    .sum();
                assert_near(out.log_probs[row], expected);
                assert_near(training_lp, expected);
            }
        }
    }
}

#[test]
fn sampling_modes_preserve_owned_rng_and_normal_draw_order() {
    let categorical = Distribution::Categorical {
        logits: matrix(vec![0., 0., 0., 100., 100., 100.], 2, 3),
        masks: vec![true, false, true, true, true, false],
    };
    let gaussian = Distribution::Gaussian {
        mean: matrix(vec![0.; 6], 2, 3),
        log_std: matrix(vec![0.; 6], 2, 3),
        bounds: None,
    };
    let bounded = Distribution::Gaussian {
        mean: matrix(vec![0.; 6], 2, 3),
        log_std: matrix(vec![0.; 6], 2, 3),
        bounds: Some((vec![-1.; 3], vec![1.; 3])),
    };
    for dist in [&categorical, &gaussian, &bounded] {
        for mode in [ActMode::Sample, ActMode::Deterministic] {
            let mut action_rng = ChaCha8Rng::seed_from_u64(77);
            let mut behavior_rng = action_rng.clone();
            let mut episode_rng = action_rng.clone();
            let mut expected_rng = action_rng.clone();
            let mut actions = Vec::new();
            let mut behavior = RolloutBehavior::default();
            let mut episode = EpisodeBehavior::default();
            dist.sample_actions_into(mode, &mut action_rng, &mut actions)
                .unwrap();
            dist.sample_behavior_into(mode, &mut behavior_rng, &mut behavior)
                .unwrap();
            dist.sample_episode_into(mode, &mut episode_rng, &mut episode)
                .unwrap();
            assert_eq!(actions, behavior.actions);
            assert_eq!(actions, episode.actions);
            assert_eq!(behavior.latents, episode.latents);
            match dist {
                Distribution::Categorical { .. } if mode == ActMode::Sample => {
                    for _ in 0..2 {
                        let _ = expected_rng.random::<f32>();
                    }
                }
                Distribution::Categorical { .. } => {
                    assert_eq!(actions, [Action::Discrete(2), Action::Discrete(1)]);
                }
                Distribution::Gaussian { bounds, .. } if mode == ActMode::Sample => {
                    for action in &actions {
                        let Action::Continuous(action) = action else {
                            panic!("continuous")
                        };
                        for &z in action {
                            let expected = expected_rng.sample::<f32, _>(StandardNormal);
                            assert_eq!(
                                z,
                                if bounds.is_some() {
                                    expected.tanh()
                                } else {
                                    expected
                                }
                            );
                        }
                    }
                }
                _ => {}
            }
            let next = expected_rng.next_u64();
            assert_eq!(action_rng.next_u64(), next);
            assert_eq!(behavior_rng.next_u64(), next);
            assert_eq!(episode_rng.next_u64(), next);
        }
    }
}

#[test]
fn a_single_valid_categorical_action_still_consumes_one_uniform() {
    let dist = Distribution::Categorical {
        logits: matrix(vec![100., -100., 0.], 1, 3),
        masks: vec![false, true, false],
    };
    let mut rng = ChaCha8Rng::seed_from_u64(29);
    let mut expected = rng.clone();
    let actions = dist.sample(ActMode::Sample, &mut rng).unwrap();
    assert_eq!(actions, [Action::Discrete(1)]);
    let _ = expected.random::<f32>();
    assert_eq!(rng.next_u64(), expected.next_u64());
}

#[test]
fn sampling_reuses_continuous_action_and_bounded_latent_allocations() {
    let dist = Distribution::Gaussian {
        mean: matrix(vec![0.; 8], 4, 2),
        log_std: matrix(vec![0.; 8], 4, 2),
        bounds: Some((vec![-1.; 2], vec![1.; 2])),
    };
    let mut out = RolloutBehavior {
        values: vec![123.],
        ..Default::default()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    dist.sample_behavior_into(ActMode::Sample, &mut rng, &mut out)
        .unwrap();
    let pointers: Vec<_> = out
        .actions
        .iter()
        .zip(&out.latents)
        .map(|(a, z)| {
            let Action::Continuous(a) = a else {
                panic!("continuous")
            };
            (a.as_ptr(), z.as_ref().unwrap().as_ptr())
        })
        .collect();
    dist.sample_behavior_into(ActMode::Sample, &mut rng, &mut out)
        .unwrap();
    for ((a, z), (a_ptr, z_ptr)) in out.actions.iter().zip(&out.latents).zip(pointers) {
        let Action::Continuous(a) = a else {
            panic!("continuous")
        };
        assert_eq!(a.as_ptr(), a_ptr);
        assert_eq!(z.as_ref().unwrap().as_ptr(), z_ptr);
    }
    assert_eq!(out.values, [123.]);
}
#[test]
fn distribution_rejects_malformed_shapes_and_mismatched_latent() {
    let d = Distribution::Categorical {
        logits: matrix(vec![0., 0.], 1, 2),
        masks: vec![],
    };
    assert!(
        d.sample(ActMode::Sample, &mut ChaCha8Rng::seed_from_u64(0))
            .is_err()
    );
    let d = Distribution::Gaussian {
        mean: matrix(vec![0.], 1, 1),
        log_std: matrix(vec![0.], 1, 1),
        bounds: Some((vec![-1.], vec![1.])),
    };
    assert!(
        d.log_prob(&[Action::Continuous(vec![0.5])], &[Some(vec![0.])])
            .is_err()
    );
}
#[test]
fn gaussian_kl_direction_is_explicit() {
    let a = Distribution::Gaussian {
        mean: matrix(vec![0.], 1, 1),
        log_std: matrix(vec![0.], 1, 1),
        bounds: None,
    };
    let b = Distribution::Gaussian {
        mean: matrix(vec![1.], 1, 1),
        log_std: matrix(vec![2_f32.ln()], 1, 1),
        bounds: None,
    };
    assert!((a.kl(&b).unwrap().into_scalar::<f32>() - (2_f32.ln() - 0.25)).abs() < 1e-6);
    assert!((b.kl(&a).unwrap().into_scalar::<f32>() - (2. - 2_f32.ln())).abs() < 1e-6);
}
