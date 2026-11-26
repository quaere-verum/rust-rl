//! Bounded complete-update measurements. Run with QUANTILE_BENCH_LABEL=before/after.
//! Counts requested allocation bytes, not live bytes or process resident memory.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::objectives::quantile::QuantileLoss;
use rl::{core::transition::Transition, data::replay::ReplayItem, prelude::*};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

struct Counting;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn record(size: usize) {
    if ACTIVE.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size as u64, Ordering::Relaxed);
    }
}

// SAFETY: Every operation preserves System's pointer and layout contracts.
// The counters only observe successful allocation sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, size) };
        if !out.is_null() {
            record(size);
        }
        out
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure(label: &str, case: &str, mut run: impl FnMut()) {
    const WARMUP: usize = 2;
    const SAMPLES: usize = 5;
    for _ in 0..WARMUP {
        run();
    }
    let mut micros = [0.; SAMPLES];
    CALLS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    for elapsed in &mut micros {
        ACTIVE.store(true, Ordering::Relaxed);
        let start = Instant::now();
        run();
        *elapsed = start.elapsed().as_secs_f64() * 1e6;
        ACTIVE.store(false, Ordering::Relaxed);
    }
    micros.sort_by(f64::total_cmp);
    println!(
        "{label},{case},{SAMPLES},{:.3},{:.3},{:.3},{:.1},{:.1}",
        micros[SAMPLES / 2],
        micros[0],
        micros[SAMPLES - 1],
        CALLS.load(Ordering::Relaxed) as f64 / SAMPLES as f64,
        BYTES.load(Ordering::Relaxed) as f64 / SAMPLES as f64,
    );
}

fn main() {
    let label = std::env::var("QUANTILE_BENCH_LABEL").unwrap_or_else(|_| "unlabelled".into());
    println!("label,case,samples,median_us,min_us,max_us,allocations,requested_bytes");
    // Same small MLPs, seeded initialization, Adam optimizers, and update cadence in every case.
    for (batch, quantiles) in [(32, 8), (32, 32), (256, 32), (256, 128)] {
        for terminal in [true, false] {
            for (name, loss) in [
                ("huber", QuantileLoss::Huber { kappa: 1. }),
                ("absolute", QuantileLoss::Absolute),
                (
                    "generalized",
                    QuantileLoss::Generalized {
                        scale: 1.,
                        ema: 0.9,
                        minimum: 0.001,
                    },
                ),
            ] {
                let mut rng = ChaCha8Rng::seed_from_u64(17);
                let policy = Deterministic::new(
                    2,
                    ActionSpace::Box {
                        low: vec![-1.],
                        high: vec![1.],
                    },
                    &[32],
                    &mut rng,
                )
                .unwrap();
                let critic = QValue::new(2, 1, quantiles, &[32, 32], &mut rng).unwrap();
                let mut config = D4pgConfig::new(quantiles).unwrap();
                config.loss = loss;
                let mut learner = D4pg::new(policy, critic, config, 19).unwrap();
                let data = ReplayBatch {
                    items: (0..batch)
                        .map(|i| ReplayItem {
                            transition: Transition {
                                observation: Observation::new(vec![i as f32 / batch as f32, 1.]),
                                action: Action::Continuous(vec![0.]),
                                reward: (1. + (i % 3) as f32 * 0.1).into(),
                                next_observation: Observation::new(vec![
                                    1.,
                                    i as f32 / batch as f32,
                                ]),
                                terminated: terminal,
                                truncated: false,
                            },
                            discount: if terminal { 0. } else { 0.99 },
                        })
                        .collect(),
                };
                let target = if terminal { "terminal" } else { "continuing" };
                let case = format!("update/{name}/{target}/b{batch}/k{quantiles}");
                eprintln!("running {label}/{case}");
                measure(&label, &case, || {
                    black_box(learner.update(&data).unwrap());
                });
            }
        }
        // Owned convenience API isolates percentile adaptation. The input clone is included
        // identically in both revisions; this is separate from full-update scratch reuse.
        let mut rng = ChaCha8Rng::seed_from_u64(23);
        let residuals: Vec<f32> = (0..batch * quantiles * quantiles)
            .map(|_| rng.random_range(-10.0..10.0))
            .collect();
        let mut loss = QuantileLoss::Generalized {
            scale: 1.,
            ema: 0.9,
            minimum: 0.001,
        };
        let case = format!("adapt_owned/generalized/b{batch}/k{quantiles}");
        eprintln!("running {label}/{case}");
        measure(&label, &case, || {
            loss.adapt(black_box(residuals.clone())).unwrap();
        });
    }
    quantile_constant_cases(&label);
}

// Attribution only: mirror the public loss's f32, row-major constant setup.
// This is not a learner cache and does not include validation or loss arithmetic.
fn quantile_constants(
    grid: &rl::nn::critics::quantile::QuantileGrid,
) -> (burn::tensor::Tensor<3>, burn::tensor::Tensor<3>) {
    let n = grid.len();
    let tau = rl::nn::matrix(grid.locations(), n, 1).reshape([1, n, 1]);
    let masses = grid.masses();
    let product: Vec<_> = masses
        .iter()
        .flat_map(|a| masses.iter().map(move |b| a * b))
        .collect();
    let mass = rl::nn::matrix(product, n, n).reshape([1, n, n]);
    (tau, mass)
}

fn quantile_constant_cases(label: &str) {
    const OPERATIONS: usize = 1000;
    for n in [8, 32, 128] {
        let grid = rl::nn::critics::quantile::QuantileGrid::uniform(n).unwrap();
        // Setup and complete numeric verification happen outside every measure call.
        let constants = quantile_constants(&grid);
        assert_eq!(constants.0.dims(), [1, n, 1]);
        assert_eq!(constants.1.dims(), [1, n, n]);
        let locations = rl::nn::to_vec(constants.0.clone()).unwrap();
        let product = rl::nn::to_vec(constants.1.clone()).unwrap();
        assert_eq!(locations.len(), n);
        assert_eq!(product.len(), n * n);
        // Derive references from edges directly, independently of the setup helper's
        // grid.locations()/masses() and iterator-based outer product.
        for i in 0..n {
            let expected = (grid.edges[i] + grid.edges[i + 1]) * 0.5_f32;
            assert_eq!(locations[i].to_bits(), expected.to_bits());
            let first_mass = grid.edges[i + 1] - grid.edges[i];
            for j in 0..n {
                let second_mass = grid.edges[j + 1] - grid.edges[j];
                assert_eq!(
                    product[i * n + j].to_bits(),
                    (first_mass * second_mass).to_bits(),
                );
            }
        }

        let case = format!("constant_setup_1000/k{n}");
        eprintln!("running {label}/{case}");
        measure(label, &case, || {
            for _ in 0..OPERATIONS {
                drop(black_box(quantile_constants(black_box(&grid))));
            }
        });

        // Optimistic handle-clone lower bound: omits cache keys, lookups and misses.
        let case = format!("constant_handles_1000/k{n}");
        eprintln!("running {label}/{case}");
        measure(label, &case, || {
            for _ in 0..OPERATIONS {
                drop(black_box((constants.0.clone(), constants.1.clone())));
            }
        });
    }
}
