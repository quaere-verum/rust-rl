#![cfg(feature = "diagnostics")]
//! Isolated executable: peak observations must not include concurrent tests.
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;

use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, storage::*},
};
use std::collections::BTreeMap;

#[test]
fn mixed_dtype_storage_load_has_only_encoded_and_final_array_payloads_live() {
    let chunk = Chunk {
        context: RecordContext::new("decode-memory", 0),
        record: Record::Metrics(MetricRecord {
            values: BTreeMap::new(),
            unavailable: BTreeMap::new(),
        }),
        arrays: BTreeMap::from([
            (
                "f32".into(),
                Array::f32(
                    vec![262144],
                    (0..262144).map(|i| (i % 257) as f32 - 128.).collect(),
                )
                .unwrap(),
            ),
            (
                "f64".into(),
                Array::f64(
                    vec![131072],
                    (0..131072).map(|i| (i % 509) as f64 - 256.).collect(),
                )
                .unwrap(),
            ),
            (
                "u64".into(),
                Array::u64(vec![1025], (0..1025).map(|i| u64::MAX - i).collect()).unwrap(),
            ),
            ("u8".into(), Array::u8(vec![8193], vec![42; 8193]).unwrap()),
        ]),
    };
    let mut observations = vec![];
    for compress in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("recording");
        let manifest = RunManifest::new(
            "decode-memory",
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
                compress,
                ..Default::default()
            },
        )
        .unwrap();
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        drop(writer);
        let recording = Recording::open(&path, ReadLimits::default()).unwrap();
        let source = recording.source(0).unwrap();
        // Warm file and lazy library state before recording requested live bytes.
        drop(source.load().unwrap());
        let before = allocation::start();
        let loaded = source.load().unwrap();
        let measured = allocation::finish(before);
        for (key, expected) in &chunk.arrays {
            let actual = &loaded.arrays[key];
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(actual.shape(), expected.shape());
            match actual.dtype() {
                DType::F32 => assert!(
                    actual
                        .as_f32()
                        .unwrap()
                        .iter()
                        .zip(expected.as_f32().unwrap())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                ),
                DType::F64 => assert!(
                    actual
                        .as_f64()
                        .unwrap()
                        .iter()
                        .zip(expected.as_f64().unwrap())
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                ),
                DType::U64 => assert_eq!(actual.as_u64().unwrap(), expected.as_u64().unwrap()),
                DType::U8 => assert_eq!(actual.as_u8().unwrap(), expected.as_u8().unwrap()),
            }
        }
        let encoded = recording.entries()[0].bytes;
        let bound = encoded + chunk.array_bytes() as u64 + 128 * 1024;
        eprintln!(
            "compress={compress}: encoded={encoded}, arrays={}, live={}, peak={}, bound={bound}, allocations={}, requested={}",
            chunk.array_bytes(),
            measured.live,
            measured.peak,
            measured.calls,
            measured.requested
        );
        observations.push((compress, measured.peak, bound));
    }
    for (compress, peak, bound) in observations {
        assert!(
            peak <= bound,
            "compress={compress}: peak {peak} > bounded encoded + final arrays + 128 KiB scratch {bound}"
        );
    }
}
