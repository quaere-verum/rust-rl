//! The release workload with a benchmark-only Rust allocator counter.
#[path = "support/counting_allocator.rs"]
mod counting_allocator;
#[path = "support/recurrent_mpo_workload.rs"]
mod workload;

fn main() -> rl::Result<()> {
    let (calls_before, bytes_before) = counting_allocator::snapshot();
    workload::main()?;
    let (calls_after, bytes_after) = counting_allocator::snapshot();
    let live_delta = i128::from(bytes_after) - i128::from(bytes_before);
    eprintln!(
        "allocation_calls={} live_heap_delta_bytes={live_delta}",
        calls_after - calls_before,
    );
    Ok(())
}
