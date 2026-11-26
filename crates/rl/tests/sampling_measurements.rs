#![cfg(feature = "neural")]

//! Bounded, opt-in sampling measurements. Run alone with --ignored --nocapture.
use burn::tensor::{Bool, Tensor, TensorData, activation::log_softmax};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use rl::{
    core::{
        action::Action,
        actor::{ActMode, EpisodeBehavior, RolloutBehavior},
    },
    nn::{device, distributions::Distribution, matrix, to_vec},
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

struct Counting;
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
// SAFETY: All operations delegate the original pointer/layout unchanged to System;
// the counters only observe successful allocations and do not access their memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, size) };
        if !ptr.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        ptr
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Baseline sampling operations from the pre-refinement distribution sampler.
/// Its output carries the same action/latent/likelihood/value payload as ActorOutput did.
type PreviousSample = (Action, Option<Vec<f32>>, Option<f32>, f32);
fn previous_sample(dist: &Distribution, rng: &mut ChaCha8Rng) -> Vec<PreviousSample> {
    let (actions, latents) = match dist {
        Distribution::Categorical { logits, masks } => {
            let [n, k] = logits.dims();
            let invalid = Tensor::<2, Bool>::from_data(
                TensorData::new(masks.iter().map(|x| !x).collect::<Vec<_>>(), [n, k]),
                &device(),
            );
            let p =
                to_vec(log_softmax(logits.clone().mask_fill(invalid, f32::NEG_INFINITY), 1).exp())
                    .unwrap();
            let actions = (0..n)
                .map(|i| {
                    let valid: Vec<_> = (0..k).filter(|&j| masks[i * k + j]).collect();
                    let u = rng.random::<f32>();
                    let mut sum = 0.;
                    let mut chosen = *valid.last().unwrap();
                    for j in valid {
                        sum += p[i * k + j];
                        if u < sum {
                            chosen = j;
                            break;
                        }
                    }
                    Action::Discrete(chosen)
                })
                .collect::<Vec<_>>();
            (actions, vec![None; n])
        }
        Distribution::Gaussian {
            mean,
            log_std,
            bounds,
        } => {
            let [n, d] = mean.dims();
            let means = to_vec(mean.clone()).unwrap();
            let stds = to_vec(log_std.clone().exp()).unwrap();
            let mut actions = Vec::with_capacity(n);
            let mut latents = Vec::with_capacity(n);
            for i in 0..n {
                let z: Vec<_> = (0..d)
                    .map(|j| {
                        means[i * d + j] + stds[i * d + j] * rng.sample::<f32, _>(StandardNormal)
                    })
                    .collect();
                let action = if let Some((lo, hi)) = bounds {
                    z.iter()
                        .enumerate()
                        .map(|(j, z)| (lo[j] + hi[j]) * 0.5 + (hi[j] - lo[j]) * 0.5 * z.tanh())
                        .collect()
                } else {
                    z.clone()
                };
                actions.push(Action::Continuous(action));
                latents.push(Some(z));
            }
            (actions, latents)
        }
    };
    let log_probs = to_vec(dist.log_prob(&actions, &latents).unwrap()).unwrap();
    actions
        .into_iter()
        .zip(latents)
        .zip(log_probs)
        .map(|((a, z), lp)| (a, z, Some(lp), 0.))
        .collect()
}

fn measure(label: &str, batch: usize, mut work: impl FnMut()) {
    const REPEATS: u64 = 100;
    for _ in 0..3 {
        work();
    }
    let allocations = CALLS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let start = Instant::now();
    for _ in 0..REPEATS {
        work();
    }
    let elapsed = start.elapsed();
    let allocations = CALLS.load(Ordering::Relaxed) - allocations;
    let bytes = BYTES.load(Ordering::Relaxed) - bytes;
    eprintln!(
        "sampling,{label},{batch},{:.3},{:.2},{:.2}",
        elapsed.as_secs_f64() * 1e6 / REPEATS as f64,
        allocations as f64 / REPEATS as f64,
        bytes as f64 / REPEATS as f64
    );
}

#[test]
#[ignore = "bounded sampling time/allocation experiment; run separately in release mode"]
fn sampling_paths_at_representative_batch_sizes() {
    eprintln!(
        "kind,path,batch,microseconds_per_call,allocations_per_call,requested_bytes_per_call"
    );
    for n in [1, 32, 64, 256] {
        let cases = [
            (
                "categorical",
                Distribution::Categorical {
                    logits: matrix([0.5, -0.5, 1., 2.].repeat(n), n, 4),
                    masks: [true, false, true, true].repeat(n),
                },
            ),
            (
                "bounded",
                Distribution::Gaussian {
                    mean: matrix(vec![0.; n * 2], n, 2),
                    log_std: matrix(vec![-0.5; n * 2], n, 2),
                    bounds: Some((vec![-2., -1.], vec![1., 3.])),
                },
            ),
            (
                "unbounded",
                Distribution::Gaussian {
                    mean: matrix(vec![0.; n * 2], n, 2),
                    log_std: matrix(vec![-0.5; n * 2], n, 2),
                    bounds: None,
                },
            ),
        ];
        for (name, dist) in cases {
            let mut rng = ChaCha8Rng::seed_from_u64(11);
            let mut actions = Vec::new();
            let mut behavior = RolloutBehavior::default();
            let mut episode = EpisodeBehavior::default();
            measure(&format!("{name}/previous"), n, || {
                black_box(previous_sample(&dist, &mut rng));
            });
            measure(&format!("{name}/actions"), n, || {
                dist.sample_actions_into(ActMode::Sample, &mut rng, &mut actions)
                    .unwrap();
                black_box(&actions);
            });
            measure(&format!("{name}/episode"), n, || {
                dist.sample_episode_into(ActMode::Sample, &mut rng, &mut episode)
                    .unwrap();
                black_box(&episode);
            });
            measure(&format!("{name}/behavior"), n, || {
                dist.sample_behavior_into(ActMode::Sample, &mut rng, &mut behavior)
                    .unwrap();
                black_box(&behavior);
            });
            measure(&format!("{name}/tensor_extraction"), n, || match &dist {
                Distribution::Categorical { logits, .. } => {
                    black_box(to_vec(logits.clone()).unwrap());
                }
                Distribution::Gaussian { mean, log_std, .. } => {
                    black_box(to_vec(mean.clone()).unwrap());
                    black_box(to_vec(log_std.clone()).unwrap());
                }
            });
        }
    }
}
