//! Bounded recording file/index/cache measurements (no GPU).
#[path = "../../../benches/support/allocation.rs"]
mod allocation;
#[global_allocator]
static ALLOCATOR: allocation::Counting = allocation::Counting;

use rl::{
    Result,
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::data::{ChunkCache, RunIndex};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc, time::Instant};
fn main() -> Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .skip(1)
            .find(|argument| argument != "--bench")
            .unwrap_or_else(|| "runs/large-synthetic".into()),
    );
    let path = if path.is_relative() {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    } else {
        path
    };
    if !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let mut writer = Recorder::create(
            &path,
            RunManifest::new(
                "large-synthetic",
                "performance",
                "synthetic activations",
                2,
                ActionSpace::Discrete { count: 2 },
                0.99,
            )?,
            StorageConfig {
                compress: false,
                ..Default::default()
            },
        )?
        .with_buffering(BufferConfig::default())?;
        let set = ProbeSet::new(
            "fixed",
            ProbePopulation::FixedReference,
            "256 deterministic synthetic inputs; no learning claim",
            (0..256)
                .map(|i| Observation::new(vec![i as f32, 0.]))
                .collect(),
            None,
            ActionSpace::Discrete { count: 2 },
        )?;
        writer.append(&set.record(RecordContext::new("large-synthetic", 0))?)?;
        let start = Instant::now();
        for k in 0..128 {
            let mut c = RecordContext::new("large-synthetic", k);
            c.probe_set = Some("fixed".into());
            let values = (0..256 * 4096)
                .map(|i| ((i % 509) as f32 / 509. - 0.5) * 2. + k as f32 * 0.001)
                .collect();
            writer.append(&Chunk {
                context: c,
                record: Record::Activations(ActivationRecord {
                    network: "policy".into(),
                    layers: BTreeMap::from([("layers.0.post".into(), "a".into())]),
                    ..Default::default()
                }),
                arrays: BTreeMap::from([("a".into(), Array::f32(vec![256, 4096], values)?)]),
            })?;
            if k % 16 == 15 {
                eprintln!(
                    "generated {}/128 layers; elapsed {:.2}s",
                    k + 1,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        writer.finish()?;
    }
    println!(
        "case,seconds,allocations,requested_bytes,live_delta_bytes,peak_extra_bytes,cache_bytes"
    );
    let before = allocation::start();
    let time = Instant::now();
    let run = RunIndex::open(&path, ReadLimits::default())?;
    report("open_512MiB_index", time, before, 0);
    let mut cache = ChunkCache::new(32 * 1024 * 1024);
    let before = allocation::start();
    let time = Instant::now();
    for i in 0..32 {
        let index = 1 + (i * 37 % 128);
        let key = run.key(index)?;
        if cache.get(&key).is_none() {
            cache.insert(key, Arc::new(run.recording().load(index)?));
        }
    }
    report("load_cache_32_layers", time, before, cache.resident_bytes());
    assert!(cache.resident_bytes() <= cache.budget());
    Ok(())
}
fn report(label: &str, start: Instant, before: allocation::Measurement, cache: usize) {
    let elapsed = start.elapsed().as_secs_f64();
    let after = allocation::finish(before);
    println!(
        "{label},{elapsed:.6},{},{},{},{},{cache}",
        after.calls, after.requested, after.live, after.peak
    );
}
