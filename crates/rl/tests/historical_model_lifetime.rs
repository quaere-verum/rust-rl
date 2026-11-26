#![cfg(all(feature = "diagnostics", feature = "neural"))]
//! Isolated executable: allocation observations must not include concurrent tests.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{models::*, schema::*},
    nn::{
        critics::{
            quantile::QuantileGrid,
            scalar::{DiscreteQ, QValue, Value},
        },
        policy::{Deterministic, Policy},
        probes::{Intervention, InterventionValue},
    },
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicIsize, Ordering},
};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
// SAFETY: Delegates identical pointer/layout contracts to System. The counter
// observes successful allocation sizes and never reads allocated memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(ptr, layout, size) };
        if !out.is_null() {
            LIVE.fetch_add(size as isize - layout.size() as isize, Ordering::Relaxed);
        }
        out
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn fixture(discrete: bool, bounded: bool) -> (Chunk, Vec<InspectionRequest>) {
    let mut rng = ChaCha8Rng::seed_from_u64(2026);
    let space = if discrete {
        ActionSpace::Discrete { count: 3 }
    } else if bounded {
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        }
    } else {
        ActionSpace::Unbounded { dimensions: 1 }
    };
    let mut builder =
        SnapshotBuilder::new("lifetime", RecordContext::new("test", 7), 2, space.clone()).unwrap();
    builder
        .policy(
            "policy",
            &Policy::new(2, space.clone(), &[4], &mut rng).unwrap(),
        )
        .unwrap();
    builder
        .value("value", &Value::new(2, &[4], &mut rng).unwrap())
        .unwrap();
    let mut request = InspectionRequest::new("policy", vec![Observation::new(vec![0.2, 0.5])]);
    request.critics.push("value".into());
    if discrete {
        builder
            .discrete("q", &DiscreteQ::new(2, 3, &[4], &mut rng).unwrap())
            .unwrap();
        request.critics.push("q".into());
    } else {
        builder
            .deterministic(
                "deterministic",
                &Deterministic::new(2, space, &[4], &mut rng).unwrap(),
            )
            .unwrap();
        builder
            .scalar("scalar", &QValue::new(2, 1, 1, &[4], &mut rng).unwrap())
            .unwrap();
        builder
            .quantiles(
                "quantiles",
                &QValue::new(2, 1, 3, &[4], &mut rng).unwrap(),
                &QuantileGrid::uniform(3).unwrap(),
            )
            .unwrap();
        builder
            .moments(
                "moments",
                &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
                &QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
            )
            .unwrap();
        request
            .critics
            .extend(["scalar".into(), "quantiles".into(), "moments".into()]);
    }
    let chunk = builder.finish().unwrap();
    let mut requests = vec![request.clone()];
    request.policy = if discrete { "q" } else { "deterministic" }.into();
    requests.push(request);
    // Every exercised network, including both moments, captures and intervenes
    // on real saved weights. Inputs change without modifying the snapshot.
    let bundle = ModelBundle::load(&chunk, ModelLimits::default()).unwrap();
    for request in &mut requests {
        for name in bundle.activation_networks().keys() {
            if name == &request.policy
                || request
                    .critics
                    .iter()
                    .any(|critic| name == critic || name.starts_with(&format!("{critic}.")))
            {
                request
                    .layers
                    .insert(name.clone(), vec!["layers.0.post".into()]);
            }
        }
    }
    (chunk, requests)
}

fn run_cycle(fixtures: &[(Chunk, Vec<InspectionRequest>)]) {
    for (chunk, requests) in fixtures {
        let bundle = ModelBundle::load(chunk, ModelLimits::default()).unwrap();
        for request in requests {
            let baseline = bundle.inspect(request).unwrap();
            let mut edited = request.clone();
            edited.observations[0].as_dense_mut().unwrap().values[0] = -0.7;
            for name in request.layers.keys() {
                edited.interventions.insert(
                    name.clone(),
                    vec![Intervention {
                        layer: "layers.0.post".into(),
                        neuron: 0,
                        value: InterventionValue::Set(0.75),
                    }],
                );
            }
            let changed = bundle.inspect(&edited).unwrap();
            for activation in changed.activations.values() {
                assert_eq!(
                    activation.arrays["layers.0.post"].as_f32().unwrap()[0],
                    0.75
                );
            }
            let again = bundle.inspect(request).unwrap();
            assert_eq!(baseline.actions, again.actions);
            for name in baseline.critics.keys() {
                for (array, values) in &baseline.critics[name].arrays {
                    assert_eq!(values.to_f64(), again.critics[name].arrays[array].to_f64());
                }
            }
            for name in baseline.activations.keys() {
                for (array, values) in &baseline.activations[name].arrays {
                    assert_eq!(
                        values.to_f64(),
                        again.activations[name].arrays[array].to_f64()
                    );
                }
            }
        }
    }
}

#[test]
fn repeated_snapshot_load_inspect_drop_releases_live_allocations() {
    let fixtures = [
        fixture(false, false),
        fixture(false, true),
        fixture(true, false),
    ];
    let load_before = LIVE.load(Ordering::Relaxed);
    for _ in 0..8 {
        for (chunk, _) in &fixtures {
            drop(ModelBundle::load(chunk, ModelLimits::default()).unwrap());
        }
    }
    let load_after = LIVE.load(Ordering::Relaxed);
    eprintln!(
        "load-only diagnostic: {} retained bytes",
        load_after - load_before
    );
    assert_eq!(
        load_after, load_before,
        "record restoration retained allocation bytes"
    );
    let saved: Vec<_> = fixtures
        .iter()
        .map(|(chunk, _)| {
            chunk
                .arrays
                .iter()
                .map(|(name, array)| (name.clone(), array.as_u8().unwrap().to_vec()))
                .collect::<Vec<_>>()
        })
        .collect();
    for _ in 0..4 {
        run_cycle(&fixtures);
    }
    // Exact live requested bytes, not process working set, allocator capacity or
    // a guessed graph count. Fixed batches expose transient registered roots
    // that survive dropping otherwise-frozen models.
    let before = LIVE.load(Ordering::Relaxed);
    for _ in 0..8 {
        run_cycle(&fixtures);
    }
    let middle = LIVE.load(Ordering::Relaxed);
    for _ in 0..8 {
        run_cycle(&fixtures);
    }
    let after = LIVE.load(Ordering::Relaxed);
    eprintln!(
        "historical lifetime: warmup=4 batches=8+8 live bytes={before},{middle},{after}; deltas={},{}",
        middle - before,
        after - middle
    );
    for ((chunk, _), original) in fixtures.iter().zip(saved) {
        for (name, bytes) in original {
            assert_eq!(chunk.arrays[&name].as_u8().unwrap(), bytes);
        }
    }
    assert_eq!(
        middle, before,
        "first bounded batch retained allocation bytes"
    );
    assert_eq!(
        after, middle,
        "second bounded batch retained allocation bytes"
    );
}
