//! Bounded same-process LQG comparison. Run with `cargo bench --bench lqg_batch`.
//! Heap counts include the vector adapter, state, RNGs and parameters; snapshots
//! stream into a counting writer so their size does not distort live heap use.
use nalgebra::DMatrix;
use rl::{
    collect::vector_env::{SyncVectorEnv, VectorEnv},
    core::action::{Action, ActionSpace},
    envs::lqg::{BatchedLqg, Lqg, LqgConfig},
};
use serde::Serialize;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

struct Counting;
static CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
// SAFETY: The allocator delegates every pointer/layout operation unchanged to
// System. Relaxed atomics count allocations without accessing their contents.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(size as u64, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[derive(Default)]
struct ByteCount(u64);
impl io::Write for ByteCount {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn config(dimensions: usize) -> LqgConfig {
    let matrix = |scale| DMatrix::identity(dimensions, dimensions) * scale;
    LqgConfig {
        a: vec![matrix(0.9); 128],
        b: vec![matrix(0.5); 128],
        q: vec![matrix(1.); 128],
        r: vec![matrix(0.2); 128],
        terminal: matrix(2.),
        noise: vec![matrix(0.01); 128],
        initial: vec![0.5; dimensions],
        action_space: ActionSpace::Box {
            low: vec![-2.; dimensions],
            high: vec![2.; dimensions],
        },
    }
}

fn measure<V: VectorEnv + Serialize>(
    label: &str,
    dimensions: usize,
    lanes: usize,
    repeat: usize,
    build: impl FnOnce() -> V,
) {
    let initial_live = LIVE.load(Ordering::Relaxed);
    let start = Instant::now();
    let mut env = build();
    let initialization = start.elapsed();
    let retained = LIVE.load(Ordering::Relaxed) - initial_live;
    let mask = vec![true; lanes];
    let seeds: Vec<_> = (0..lanes as u64).map(|lane| lane + 100).collect();
    let actions = vec![Some(Action::Continuous(vec![0.125; dimensions])); lanes];
    env.reset(&mask, Some(&seeds)).unwrap();
    let calls = CALLS.load(Ordering::Relaxed);
    let start = Instant::now();
    // Exactly ten full episodes: no unbounded adaptive benchmark iteration.
    for episode in 0..10 {
        if episode > 0 {
            black_box(env.reset(&mask, None).unwrap());
        }
        for _ in 0..128 {
            black_box(env.step(&actions).unwrap());
        }
    }
    let elapsed = start.elapsed();
    let allocations = CALLS.load(Ordering::Relaxed) - calls;
    let mut snapshot = ByteCount::default();
    let start = Instant::now();
    serde_json::to_writer(&mut snapshot, &env).unwrap();
    let snapshot_time = start.elapsed();
    println!(
        "{label},{dimensions},{lanes},{repeat},{:.3},{retained},{:.3},{:.1},{allocations},{},{:.3}",
        initialization.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e3,
        (lanes * 1280) as f64 / elapsed.as_secs_f64(),
        snapshot.0,
        snapshot_time.as_secs_f64() * 1e3,
    );
}

fn main() {
    println!(
        "adapter,dimensions,lanes,repeat,init_ms,retained_heap_bytes,step_ms,lane_steps_per_second,step_allocations,snapshot_bytes,snapshot_ms"
    );
    for dimensions in [1, 8] {
        let config = Arc::new(config(dimensions));
        let prototype = Lqg::from_shared(Arc::clone(&config), 100).unwrap();
        for lanes in [1, 32, 256, 1024] {
            let seeds: Vec<_> = (100..100 + lanes as u64).collect();
            for repeat in 0..3 {
                measure("sync", dimensions, lanes, repeat, || {
                    SyncVectorEnv::new(vec![prototype.clone(); lanes]).unwrap()
                });
                measure("batched", dimensions, lanes, repeat, || {
                    BatchedLqg::from_shared(Arc::clone(&config), &seeds).unwrap()
                });
            }
        }
    }
}
