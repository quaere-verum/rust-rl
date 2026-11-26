//! System-allocator instrumentation for standalone measurement executables only.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicU64, Ordering},
};
pub struct Counting;
static CALLS: AtomicU64 = AtomicU64::new(0);
static REQUESTED: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
fn add(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    REQUESTED.fetch_add(size as u64, Ordering::Relaxed);
    let now = LIVE.fetch_add(size as u64, Ordering::Relaxed) + size as u64;
    PEAK.fetch_max(now, Ordering::Relaxed);
}
// SAFETY: Pointer/layout operations delegate unchanged to System. Counters never
// access allocation contents; they report requested bytes, not OS working set.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let out = unsafe { System.realloc(p, layout, size) };
        if !out.is_null() {
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
            add(size);
        }
        out
    }
}
#[derive(Clone, Copy)]
pub struct Measurement {
    pub calls: u64,
    pub requested: u64,
    pub live: u64,
    pub peak: u64,
}
pub fn start() -> Measurement {
    let live = LIVE.load(Ordering::Relaxed);
    PEAK.store(live, Ordering::Relaxed);
    Measurement {
        calls: CALLS.load(Ordering::Relaxed),
        requested: REQUESTED.load(Ordering::Relaxed),
        live,
        peak: live,
    }
}
pub fn finish(before: Measurement) -> Measurement {
    Measurement {
        calls: CALLS.load(Ordering::Relaxed) - before.calls,
        requested: REQUESTED.load(Ordering::Relaxed) - before.requested,
        live: LIVE.load(Ordering::Relaxed).saturating_sub(before.live),
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(before.live),
    }
}
