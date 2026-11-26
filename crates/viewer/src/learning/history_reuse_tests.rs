use super::*;
use rl::diagnostics::storage::{ReadLimits, Recorder, StorageConfig};
use std::cell::Cell;

#[allow(dead_code)]
mod fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/learning_critic.rs"
    ));
}

fn recording() -> (tempfile::TempDir, RunIndex) {
    recording_with_critics(2)
}
fn recording_with_critics(critics: usize) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let (critic, probes, sample, manifest) = fixture::critic_fixture();
    let mut writer = Recorder::create(&root, manifest, StorageConfig::default()).unwrap();
    for chunk in [&probes, &critic, &sample] {
        writer.append(chunk).unwrap();
    }
    for _ in 1..critics {
        writer.append(&critic).unwrap();
    }
    writer.finish().unwrap();
    (dir, RunIndex::open(root, ReadLimits::default()).unwrap())
}

#[test]
fn task3a2_recency_keeps_recurrent_sources_through_complete_history() {
    let (_dir, run) = recording_with_critics(32);
    let (cold, cold_reads, cold_decodes, cold_checks, cold_memory) = run_case(&run, false, || {});
    let (warm, warm_reads, warm_decodes, warm_checks, warm_memory) = run_case(&run, true, || {});
    let cold = cold.unwrap();
    let warm = warm.unwrap();
    assert_eq!(snapshot(&cold), snapshot(&warm));
    assert_eq!((&cold.root, &cold.role), (&warm.root, &warm.role));
    assert_eq!((cold_reads, cold_decodes, cold_checks), (253, 253, 573));
    assert_eq!((warm_reads, warm_checks), (cold_reads, cold_checks));
    assert_eq!(
        warm.critic_errors
            .iter()
            .filter(|point| point.value.is_some())
            .count(),
        32
    );
    eprintln!(
        "task3a2 complete history: reads={warm_reads}, decodes={warm_decodes}, checkpoints={warm_checks}, points=32"
    );
    assert_eq!(
        warm_decodes, 34,
        "each of the 34 immutable sources should decode only once"
    );
    assert_eq!(
        cold_memory.0.load(Ordering::Acquire),
        warm_memory.0.load(Ordering::Acquire)
    );
    drop(cold);
    drop(warm);
    assert_eq!(cold_memory.0.load(Ordering::Acquire), 0);
    assert_eq!(warm_memory.0.load(Ordering::Acquire), 0);
}

#[test]
#[ignore = "uncontended complete Learning prepare timing; run directly with --ignored --nocapture --test-threads=1"]
fn measure_complete_learning_prepare_reuse() {
    let (_dir, run) = recording_with_critics(32);
    let cold = run_case(&run, false, || {}).0.unwrap();
    let warm = run_case(&run, true, || {}).0.unwrap();
    assert_eq!(snapshot(&cold), snapshot(&warm));
    drop(cold);
    drop(warm);
    let headers: Vec<_> = run
        .recording()
        .entries()
        .iter()
        .map(|entry| {
            let bytes = std::fs::read(run.recording().root().join(&entry.file)).unwrap();
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize
        })
        .collect();
    eprintln!(
        "complete_learning_fixture entries={} header_min={} header_max={} eligible_headers={} frozen_critic_points=32",
        headers.len(),
        headers.iter().min().unwrap(),
        headers.iter().max().unwrap(),
        headers.iter().filter(|&&n| n <= 2048).count()
    );
    println!("reuse,repeat,iterations,elapsed_ns,logical_reads,physical_decodes,points");
    for repeat in 0..5 {
        // Alternate order, with exact snapshots checked outside each interval.
        for reuse in if repeat % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let shared = shared();
            let owner = owner(&run, &shared);
            let work = shared.memory.reserve(WORK_BYTES).unwrap();
            let start = std::time::Instant::now();
            let mut frame = None;
            for _ in 0..10 {
                drop(frame.take());
                frame = Some(
                    prepare_with_cache(
                        owner.clone(),
                        &shared,
                        &|| Ok(()),
                        reuse.then(DecodedChunkCache::default),
                    )
                    .unwrap(),
                );
            }
            let elapsed = start.elapsed().as_nanos();
            let frame = frame.unwrap();
            assert_eq!(
                snapshot(&frame),
                snapshot(&run_case(&run, false, || {}).0.unwrap())
            );
            println!(
                "{reuse},{repeat},10,{elapsed},{},{},{}",
                shared.reads.load(Ordering::Acquire),
                shared.decodes.load(Ordering::Acquire),
                frame.critic_errors.len()
            );
            drop(frame);
            drop(owner);
            drop(work);
            assert_eq!(shared.memory.0.load(Ordering::Acquire), 0);
        }
    }
}
fn shared() -> Shared {
    Shared {
        mailbox: Mutex::default(),
        changed: Condvar::new(),
        serial: AtomicU64::new(0),
        memory: Arc::default(),
        reads: AtomicU64::new(0),
        decodes: AtomicU64::new(0),
        completed: AtomicU64::new(0),
    }
}
fn owner(run: &RunIndex, shared: &Shared) -> Arc<RequestOwner> {
    Arc::new(RequestOwner {
        request: LearningRequest::new(run, Key::new(run, "online", None).unwrap(), None).unwrap(),
        _lease: shared.memory.reserve(REQUEST_BYTES).unwrap(),
    })
}
fn snapshot(frame: &LearningFrame) -> String {
    // Complete externally observable summaries and private failed-sequence/
    // anchor identities; Debug preserves float digits and signed zero.
    format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}|{:?}|{:?}|{:?}",
        frame.evaluations,
        frame.policy_changes,
        frame.critic_errors,
        frame.anchor,
        frame.critic_anchor,
        frame.omitted_sources,
        frame.retained_bytes,
        frame.unavailable_sources,
        frame.anchor_source.as_ref().map(|s| &s.key),
        frame.critic_anchor_source.as_ref().map(|s| &s.key)
    )
}
fn run_case(
    run: &RunIndex,
    reuse: bool,
    mutate: impl FnOnce(),
) -> (Result<Arc<LearningFrame>>, u64, u64, usize, Arc<Memory>) {
    let shared = shared();
    let work = shared.memory.reserve(WORK_BYTES).unwrap();
    let owner = owner(run, &shared);
    let mutation = std::cell::RefCell::new(Some(mutate));
    let checks = Cell::new(0);
    let checkpoint = || {
        checks.set(checks.get() + 1);
        // After the first sample load, before its formerly repeated probe.
        if shared.reads.load(Ordering::Acquire) == 3
            && let Some(mutate) = mutation.borrow_mut().take()
        {
            mutate();
        }
        Ok(())
    };
    let result = prepare_with_cache(
        owner,
        &shared,
        &checkpoint,
        reuse.then(DecodedChunkCache::default),
    );
    drop(work);
    (
        result,
        shared.reads.load(Ordering::Acquire),
        shared.decodes.load(Ordering::Acquire),
        checks.get(),
        shared.memory,
    )
}
#[test]
fn shared_probe_and_cohort_reuse_preserves_every_summary_and_logical_read() {
    let (_dir, run) = recording();
    let (cold, cold_reads, cold_decodes, cold_checks, cold_memory) = run_case(&run, false, || {});
    let (warm, warm_reads, warm_decodes, warm_checks, warm_memory) = run_case(&run, true, || {});
    let cold = cold.unwrap();
    let warm = warm.unwrap();
    assert_eq!(snapshot(&cold), snapshot(&warm));
    assert_eq!((cold_reads, cold_decodes), (13, 13));
    assert_eq!(warm_reads, cold_reads);
    assert_eq!(warm_checks, cold_checks);
    assert_eq!(warm_decodes, 4);
    assert_eq!(
        warm.critic_errors
            .iter()
            .filter(|p| p.value.is_some())
            .count(),
        2
    );
    assert_eq!(
        cold_memory.0.load(Ordering::Acquire),
        warm_memory.0.load(Ordering::Acquire)
    );
    eprintln!(
        "cohort history: logical reads={warm_reads}, physical decodes {cold_decodes}->{warm_decodes}, checkpoint calls={warm_checks}, points={}",
        warm.critic_errors.len()
    );
    drop(cold);
    drop(warm);
    assert_eq!(cold_memory.0.load(Ordering::Acquire), 0);
    assert_eq!(warm_memory.0.load(Ordering::Acquire), 0);
}
#[test]
fn formerly_repeated_probe_reads_detect_mutations_with_identical_failures() {
    let (_dir, run) = recording();
    let path = run
        .recording()
        .root()
        .join(&run.recording().entries()[0].file);
    let original = std::fs::read(&path).unwrap();
    for case in 0..4 {
        let mutate = || match case {
            0 => std::fs::remove_file(&path).unwrap(),
            1 => std::fs::write(&path, &original[..original.len() / 2]).unwrap(),
            2 => {
                let mut bytes = original.clone();
                bytes[0] ^= 1;
                std::fs::write(&path, bytes).unwrap();
            }
            _ => std::fs::write(&path, vec![0; original.len()]).unwrap(),
        };
        std::fs::write(&path, &original).unwrap();
        let (cold, cold_reads, _, cold_checks, cold_memory) = run_case(&run, false, mutate);
        std::fs::write(&path, &original).unwrap();
        let (warm, warm_reads, _, warm_checks, warm_memory) = run_case(&run, true, mutate);
        let summarize = |result: &Result<Arc<LearningFrame>>| match result {
            Ok(frame) => snapshot(frame),
            Err(error) => error.to_string(),
        };
        assert_eq!(summarize(&cold), summarize(&warm), "case {case}");
        assert_eq!((cold_reads, cold_checks), (warm_reads, warm_checks));
        drop(cold);
        drop(warm);
        assert_eq!(cold_memory.0.load(Ordering::Acquire), 0);
        assert_eq!(warm_memory.0.load(Ordering::Acquire), 0);
    }
}
#[test]
fn cached_reader_preserves_read_and_byte_thresholds_and_cancellation() {
    let (_dir, run) = recording();
    let source = run.recording().source(0).unwrap();
    let charge = source.load().unwrap().retained_charge();
    for reusable in [false, true] {
        let shared = shared();
        let checkpoints = Cell::new(0);
        let checkpoint = || {
            checkpoints.set(checkpoints.get() + 1);
            Ok(())
        };
        let mut reader = Reader {
            shared: &shared,
            checkpoint: &checkpoint,
            reads: 0,
            bytes: 0,
            failed: BTreeSet::new(),
            cache: reusable.then(DecodedChunkCache::default),
        };
        drop(reader.load(&source).unwrap());
        reader.bytes = MAX_TOTAL_READ_BYTES - charge;
        reader.reads = MAX_READS - 1;
        drop(reader.load(&source).unwrap());
        assert_eq!(
            (reader.reads, reader.bytes),
            (MAX_READS, MAX_TOTAL_READ_BYTES)
        );
        assert!(
            reader
                .load(&source)
                .unwrap_err()
                .to_string()
                .contains("allowance exhausted")
        );
        assert_eq!(checkpoints.get(), 5);
        reader.reads = 0;
        reader.bytes = MAX_TOTAL_READ_BYTES - charge + 1;
        assert!(
            reader
                .load(&source)
                .unwrap_err()
                .to_string()
                .contains("remaining history allowance")
        );
        assert_eq!(reader.reads, 1);
    }
    for cancel_at in 1..18 {
        let shared = shared();
        let work = shared.memory.reserve(WORK_BYTES).unwrap();
        let owner = owner(&run, &shared);
        let calls = Cell::new(0);
        let checkpoint = || {
            calls.set(calls.get() + 1);
            checked(calls.get() < cancel_at, "cancelled")
        };
        assert!(prepare(owner, &shared, &checkpoint).is_err());
        drop(work);
        assert_eq!(shared.memory.0.load(Ordering::Acquire), 0);
    }
}
