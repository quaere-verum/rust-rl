#![cfg(feature = "diagnostics")]
//! One test keeps the whole-storage allocation comparison uncontended.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;

use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use std::{collections::BTreeMap, sync::Arc};

type Cache = DecodedChunkCache;
fn cached(source: &ChunkSource, limit: usize, cache: &mut Cache) -> rl::Result<Option<Arc<Chunk>>> {
    cache.load_if_within_charge(source, limit)
}

// Solve the 32-bit CRC correction in the final four unused numeric bytes.
// Equal CRC is deliberately insufficient evidence that a decode can be reused.
fn collide(bytes: &mut [u8], wanted: u32) {
    let tail = bytes.len() - 4;
    bytes[tail..].fill(0);
    let base = crc32fast::hash(bytes);
    let mut basis = [(0u32, 0u32); 32];
    for bit in 0..32 {
        bytes[tail + bit / 8] ^= 1 << (bit % 8);
        let mut value = crc32fast::hash(bytes) ^ base;
        bytes[tail + bit / 8] ^= 1 << (bit % 8);
        let mut mask = 1u32 << bit;
        while value != 0 {
            let pivot = value.ilog2() as usize;
            if basis[pivot].0 == 0 {
                basis[pivot] = (value, mask);
                break;
            }
            value ^= basis[pivot].0;
            mask ^= basis[pivot].1;
        }
    }
    let mut residual = wanted ^ base;
    let mut correction = 0;
    while residual != 0 {
        let (value, mask) = basis[residual.ilog2() as usize];
        assert_ne!(value, 0);
        residual ^= value;
        correction ^= mask;
    }
    bytes[tail..].copy_from_slice(&correction.to_le_bytes());
    assert_eq!(crc32fast::hash(bytes), wanted);
}

#[test]
fn exact_repeated_storage_load_reuses_decoding_but_still_observes_replacements() {
    const LIMIT: usize = 4 * 1024 * 1024;
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("recording");
    let manifest = RunManifest::new(
        "reuse",
        "fixture",
        "fixture",
        1,
        ActionSpace::Discrete { count: 2 },
        0.9,
    )
    .unwrap();
    let chunk = Chunk {
        context: RecordContext::new("reuse", 0),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::new(),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([(
            "bytes".into(),
            Array::u8(vec![128 * 1024], vec![42; 128 * 1024]).unwrap(),
        )]),
    };
    let mut writer = Recorder::create(
        &root,
        manifest,
        StorageConfig {
            compress: false,
            ..Default::default()
        },
    )
    .unwrap();
    writer.append(&chunk).unwrap();
    let mut oversized_header = Chunk {
        context: chunk.context.clone(),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::new(),
            unavailable: BTreeMap::from([("large".into(), "x".repeat(64 * 1024))]),
        }),
        arrays: BTreeMap::new(),
    };
    writer.append(&oversized_header).unwrap();
    oversized_header.record = chunk.record.clone();
    oversized_header.arrays.insert(
        "large".into(),
        Array::u8(vec![2 * 1024 * 1024], vec![0; 2 * 1024 * 1024]).unwrap(),
    );
    writer.append(&oversized_header).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let recording = Recording::open(&root, ReadLimits::default()).unwrap();
    let source = recording.source(0).unwrap();
    let path = root.join(&recording.entries()[0].file);
    let original = std::fs::read(&path).unwrap();
    let charge = chunk.retained_charge();
    let mut cache = Cache::default();
    let first = cached(&source, LIMIT, &mut cache).unwrap().unwrap();
    let before = allocation::start();
    let plain = source.load_if_within_charge(LIMIT).unwrap().unwrap();
    let ordinary = allocation::finish(before);
    drop(plain);
    let before = allocation::start();
    let second = cached(&source, LIMIT, &mut cache).unwrap().unwrap();
    let reused = allocation::finish(before);
    assert_eq!(
        first.arrays["bytes"].as_u8().unwrap(),
        second.arrays["bytes"].as_u8().unwrap()
    );
    eprintln!(
        "ordinary calls={} requested={} peak={}; reuse calls={} requested={} peak={}",
        ordinary.calls,
        ordinary.requested,
        ordinary.peak,
        reused.calls,
        reused.requested,
        reused.peak
    );
    assert!(
        reused.requested + 128 * 1024 <= ordinary.requested,
        "warm repeated loads must eliminate a complete numeric allocation"
    );
    assert!(Arc::ptr_eq(&first, &second));
    drop(second);
    assert!(cached(&source, charge - 1, &mut cache).unwrap().is_none());
    assert!(cached(&source, charge, &mut cache).unwrap().is_some());

    for replacement in [vec![], original[..original.len() - 1].to_vec(), {
        let mut bytes = original.clone();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 1;
        bytes
    }] {
        std::fs::write(&path, replacement).unwrap();
        for limit in [LIMIT, charge - 1] {
            let plain = source.load_if_within_charge(limit).unwrap_err().to_string();
            let reuse = cached(&source, limit, &mut cache).unwrap_err().to_string();
            assert_eq!(plain, reuse);
        }
    }
    std::fs::remove_file(&path).unwrap();
    assert!(
        cached(&source, original.len() - 1, &mut cache)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        source.load_if_within_charge(LIMIT).unwrap_err().to_string(),
        cached(&source, LIMIT, &mut cache).unwrap_err().to_string()
    );
    std::fs::write(&path, &original).unwrap();
    assert!(Arc::ptr_eq(
        &first,
        &cached(&source, LIMIT, &mut cache).unwrap().unwrap()
    ));
    let mut collision = original.clone();
    let header = u32::from_le_bytes(collision[8..12].try_into().unwrap()) as usize;
    collision[12 + header] = 99;
    collide(&mut collision, recording.entries()[0].checksum);
    std::fs::write(&path, &collision).unwrap();
    let replacement = cached(&source, LIMIT, &mut cache).unwrap().unwrap();
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert_eq!(replacement.arrays["bytes"].as_u8().unwrap()[0], 99);
    assert_eq!(
        replacement.arrays["bytes"].as_u8().unwrap(),
        source.load().unwrap().arrays["bytes"].as_u8().unwrap()
    );
    std::fs::write(&path, &original).unwrap();
    let reopened = Recording::open(&root, ReadLimits::default())
        .unwrap()
        .source(0)
        .unwrap();
    let fresh = cached(&reopened, LIMIT, &mut cache).unwrap().unwrap();
    assert!(
        !Arc::ptr_eq(&first, &fresh),
        "an independent manifest origin must miss"
    );
    drop(first);
    drop(cache);

    let mut cache = Cache::default();
    let old = cached(&source, LIMIT, &mut cache).unwrap().unwrap();
    let old_weak = Arc::downgrade(&old);
    drop(old);
    let large_header = recording.source(1).unwrap();
    let before = allocation::start();
    let fallback = cached(&large_header, LIMIT, &mut cache).unwrap().unwrap();
    let fallback_memory = allocation::finish(before);
    assert!(old_weak.upgrade().is_none());
    assert!(
        fallback_memory.peak <= recording.entries()[1].bytes + 8192,
        "oversized header must release old cached ownership before allocating its header tree"
    );
    assert_eq!(
        format!("{:?}", fallback.record),
        format!("{:?}", large_header.load().unwrap().record)
    );
    let fallback_weak = Arc::downgrade(&fallback);
    drop(fallback);
    assert!(
        fallback_weak.upgrade().is_none(),
        "oversized headers are not retained"
    );
    let large_numeric = recording.source(2).unwrap();
    let first = cached(&large_numeric, LIMIT, &mut cache).unwrap().unwrap();
    let second = cached(&large_numeric, LIMIT, &mut cache).unwrap().unwrap();
    assert!(
        !Arc::ptr_eq(&first, &second),
        "sources above cache capacity must bypass without refusal"
    );
    assert_eq!(first.array_bytes(), 2 * 1024 * 1024);
    drop(first);
    drop(second);
    drop(cache);

    // A shortened page must fail at the original extent check, even when the
    // cache has already validated this exact page slice.
    let page_root = temporary.path().join("page");
    let mut writer = Recorder::create(
        &page_root,
        recording.manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap()
    .with_buffering(BufferConfig::default())
    .unwrap();
    writer.append(&chunk).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let page = Recording::open(&page_root, ReadLimits::default()).unwrap();
    let page_source = page.source(0).unwrap();
    let mut cache = Cache::default();
    let held = cached(&page_source, LIMIT, &mut cache).unwrap().unwrap();
    let page_path = page_root.join(&page.entries()[0].file);
    let page_bytes = std::fs::read(&page_path).unwrap();
    std::fs::write(&page_path, &page_bytes[..page_bytes.len() - 1]).unwrap();
    assert_eq!(
        page_source
            .load_if_within_charge(LIMIT)
            .unwrap_err()
            .to_string(),
        cached(&page_source, LIMIT, &mut cache)
            .unwrap_err()
            .to_string()
    );
    std::fs::write(&page_path, page_bytes).unwrap();
    assert!(Arc::ptr_eq(
        &held,
        &cached(&page_source, LIMIT, &mut cache).unwrap().unwrap()
    ));
    let weak = Arc::downgrade(&held);
    drop(held);
    drop(cache);
    assert!(weak.upgrade().is_none());
    audit_header_layouts(temporary.path());
}

fn audit_header_layouts(root: &std::path::Path) {
    eprintln!(
        "x64 layouts Chunk={} Record={} RecordContext={} Array={} Value={} ModelArtifact={} cache={}",
        std::mem::size_of::<Chunk>(),
        std::mem::size_of::<Record>(),
        std::mem::size_of::<RecordContext>(),
        std::mem::size_of::<Array>(),
        std::mem::size_of::<serde_json::Value>(),
        std::mem::size_of::<ModelArtifact>(),
        std::mem::size_of::<Cache>()
    );
    for branches in [20, 120] {
        let mut chain = serde_json::Value::Null;
        for _ in 0..10 {
            chain = serde_json::json!({"": chain});
        }
        let chunk = Chunk {
            context: RecordContext::new("layout", 0),
            record: Record::ModelSnapshot(ModelSnapshotRecord {
                id: "snapshot".into(),
                architecture: "fixture".into(),
                configuration: serde_json::Value::Array(vec![chain; branches]),
                models: BTreeMap::from([(
                    "model".into(),
                    ModelArtifact {
                        bytes: "model".into(),
                        signature: "fixture".into(),
                        parameter_shapes: BTreeMap::new(),
                    },
                )]),
                capabilities: vec![],
            }),
            arrays: BTreeMap::from([("model".into(), Array::u8(vec![1], vec![0]).unwrap())]),
        };
        let path = root.join(format!("header-{branches}"));
        let manifest = RunManifest::new(
            "layout",
            "fixture",
            "fixture",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let mut writer = Recorder::create(
            &path,
            manifest,
            StorageConfig {
                compress: false,
                ..Default::default()
            },
        )
        .unwrap();
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        drop(writer);
        drop(chunk);
        let recording = Recording::open(&path, ReadLimits::default()).unwrap();
        let encoded = std::fs::read(path.join(&recording.entries()[0].file)).unwrap();
        let header = u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;
        let source = recording.source(0).unwrap();
        let before = allocation::start();
        let loaded = source
            .load_if_within_charge(4 * 1024 * 1024)
            .unwrap()
            .unwrap();
        let measured = allocation::finish(before);
        let bound = 256 * header + 64 * 1024;
        eprintln!(
            "nested JSON header={header}, retained={}, peak={}, former_header_bound={}, guarded_header_bound={bound}",
            measured.live,
            measured.peak,
            64 * header + 64 * 1024
        );
        assert!(measured.peak <= encoded.len() as u64 + bound as u64);
        if branches == 20 {
            assert!(header <= 2048);
        }
        drop(loaded);
    }
}

#[test]
#[ignore = "uncontended whole storage-load comparison; run directly with --ignored --nocapture --test-threads=1"]
fn measure_complete_storage_load_reuse() {
    fn measure<T>(name: &str, repeat: usize, mut load: impl FnMut() -> T) {
        drop(load());
        let before = allocation::start();
        let start = std::time::Instant::now();
        let mut retained = None;
        for _ in 0..100 {
            drop(retained.take());
            retained = Some(std::hint::black_box(load()));
        }
        let elapsed = start.elapsed().as_nanos();
        let measured = allocation::finish(before);
        println!(
            "{name},{repeat},100,{elapsed},{},{},{},{}",
            measured.calls, measured.requested, measured.live, measured.peak
        );
        drop(retained);
    }
    println!(
        "case,repeat,iterations,elapsed_ns,allocations,requested_bytes,retained_bytes,peak_bytes"
    );
    for compress in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("run");
        let manifest = RunManifest::new(
            "measure",
            "fixture",
            "fixture",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let chunk = Chunk {
            context: RecordContext::new("measure", 0),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::from([(
                "bytes".into(),
                Array::u8(
                    vec![128 * 1024],
                    (0..128 * 1024).map(|i| (i % 251) as u8).collect(),
                )
                .unwrap(),
            )]),
        };
        let mut writer = Recorder::create(
            &root,
            manifest,
            StorageConfig {
                compress,
                ..Default::default()
            },
        )
        .unwrap();
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let recording = Recording::open(&root, ReadLimits::default()).unwrap();
        let source = recording.source(0).unwrap();
        let mut cache = Cache::default();
        let before = allocation::start();
        let warm = cached(&source, 4 * 1024 * 1024, &mut cache)
            .unwrap()
            .unwrap();
        let ownership = allocation::finish(before);
        assert_eq!(format!("{chunk:?}"), format!("{warm:?}"));
        drop(warm);
        eprintln!(
            "storage fixture compressed={compress}, encoded={}, checksum={}, warm_cache_requested_live={}",
            recording.entries()[0].bytes,
            recording.entries()[0].checksum,
            ownership.live
        );
        for repeat in 0..5 {
            if repeat % 2 == 0 {
                measure(&format!("ordinary_{compress}"), repeat, || {
                    source
                        .load_if_within_charge(4 * 1024 * 1024)
                        .unwrap()
                        .unwrap()
                });
                measure(&format!("exact_reuse_{compress}"), repeat, || {
                    cached(&source, 4 * 1024 * 1024, &mut cache)
                        .unwrap()
                        .unwrap()
                });
            } else {
                measure(&format!("exact_reuse_{compress}"), repeat, || {
                    cached(&source, 4 * 1024 * 1024, &mut cache)
                        .unwrap()
                        .unwrap()
                });
                measure(&format!("ordinary_{compress}"), repeat, || {
                    source
                        .load_if_within_charge(4 * 1024 * 1024)
                        .unwrap()
                        .unwrap()
                });
            }
        }
        assert_eq!(cache.decodes(), 1);
    }
}
