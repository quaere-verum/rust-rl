use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicU64, Ordering},
};

struct Counting;
static CALLS: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);

// SAFETY: Every allocation operation delegates unchanged pointer/layout contracts to System.
// Atomics only observe successful operations and never access the allocated memory.
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
        let resized = unsafe { System.realloc(pointer, layout, size) };
        if !resized.is_null() {
            CALLS.fetch_add(1, Ordering::Relaxed);
            LIVE.fetch_add(size as u64, Ordering::Relaxed);
            LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        }
        resized
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

pub(super) fn snapshot() -> (u64, u64) {
    (CALLS.load(Ordering::Relaxed), LIVE.load(Ordering::Relaxed))
}
