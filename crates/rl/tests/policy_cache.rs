#![cfg(all(feature = "diagnostics", feature = "neural"))]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{
        models::{ModelBundle, ModelLimits, SnapshotBuilder},
        policy_cache::{CacheLimits, PolicyCache},
        policy_precompute::{PrecomputeOptions, precompute},
        policy_slice::{PolicySliceSpec, SliceConfig},
        policy_surface::{SurfaceLimits, evaluate_surface},
        schema::{RecordContext, RunManifest},
        storage::{ReadLimits, Recorder, Recording, StorageConfig},
        visualization::EnvironmentSemantics,
    },
    nn::policy::{Deterministic, Policy},
};

fn fixture(
    stochastic: bool,
    space: ActionSpace,
) -> (tempfile::TempDir, Recording, PolicySliceSpec) {
    let dir = tempfile::tempdir().unwrap();
    let manifest = RunManifest::new("cached", "test", "ordinary", 2, space.clone(), 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for seed in 0..3 {
        let mut builder =
            SnapshotBuilder::new("saved", RecordContext::new("cached", 7), 2, space.clone())
                .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        if stochastic {
            builder
                .policy(
                    "actor",
                    &Policy::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        } else {
            builder
                .deterministic(
                    "actor",
                    &Deterministic::new(2, space.clone(), &[4], &mut rng).unwrap(),
                )
                .unwrap();
        }
        writer.append(&builder.finish().unwrap()).unwrap();
    }
    writer.finish().unwrap();
    let run = Recording::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut config = SliceConfig::new(
        EnvironmentSemantics::new(run.manifest()),
        &Observation::new(vec![0., 0.]),
        &[],
    )
    .unwrap();
    config.resolution = [3, 2];
    let spec = PolicySliceSpec::from_config("online", "actor", &config);
    (dir, run, spec)
}

#[test]
fn cache_is_incremental_checkpoint_first_and_preserves_all_policy_families() {
    for (stochastic, space) in [
        (false, ActionSpace::Unbounded { dimensions: 2 }),
        (true, ActionSpace::Discrete { count: 3 }),
        (
            true,
            ActionSpace::Box {
                low: vec![-3., -1.],
                high: vec![7., 2.],
            },
        ),
    ] {
        let (dir, run, spec) = fixture(stochastic, space);
        let cache =
            PolicyCache::open_or_create(dir.path().join("cache"), &run, CacheLimits::default())
                .unwrap();
        let options = PrecomputeOptions::default();
        let summary =
            precompute(&run, &cache, std::slice::from_ref(&spec), &options, |_| {}).unwrap();
        assert_eq!(
            (
                summary.computed,
                summary.skipped,
                summary.failed,
                summary.restores
            ),
            (3, 0, 0, 3)
        );
        assert_eq!(summary.inference_batches, 3);
        let entries = cache.catalog().unwrap();
        assert_eq!(entries.entries.len(), 3);
        assert!(entries.issues.is_empty());
        for (index, source) in run.entries().iter().enumerate() {
            let entry = cache.find(source, &spec).unwrap().unwrap();
            let loaded = cache.load(&entry).unwrap();
            let model =
                ModelBundle::load(&run.load(index).unwrap(), ModelLimits::default()).unwrap();
            let direct = evaluate_surface(
                &model,
                &spec,
                EnvironmentSemantics::new(run.manifest()),
                SurfaceLimits::default(),
                || false,
                |_| {},
            )
            .unwrap();
            for row in 0..loaded.rows() {
                assert_eq!(loaded.input(row).unwrap(), direct.input(row).unwrap());
                for action in 0..loaded.action_count() {
                    assert_eq!(
                        loaded.value(row, action).unwrap(),
                        direct.value(row, action).unwrap()
                    );
                    assert_eq!(
                        loaded.distribution(row, action).unwrap(),
                        direct.distribution(row, action).unwrap()
                    );
                }
            }
        }
        let again =
            precompute(&run, &cache, std::slice::from_ref(&spec), &options, |_| {}).unwrap();
        assert_eq!(
            (
                again.computed,
                again.skipped,
                again.restores,
                again.inference_batches
            ),
            (0, 3, 0, 0)
        );
        let mut second = spec.clone();
        second.fixed[0] = 0.75;
        second.ranges[0] = [-2., 2.];
        let added = precompute(&run, &cache, &[spec.clone(), second], &options, |_| {}).unwrap();
        assert_eq!((added.computed, added.skipped, added.restores), (3, 3, 3));
        assert_eq!(cache.catalog().unwrap().entries.len(), 6);
        let keys: std::collections::BTreeSet<_> = entries.entries.iter().map(|e| &e.key).collect();
        assert_eq!(
            keys.len(),
            3,
            "same learner revision never aliases distinct sources"
        );
    }
}

#[test]
fn prepared_data_survives_pruning_but_missing_weights_never_evaluate_new_slices() {
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let cache = PolicyCache::open_or_create(dir.path().join("cache"), &run, CacheLimits::default())
        .unwrap();
    precompute(
        &run,
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    for source in run.entries() {
        std::fs::remove_file(run.root().join(&source.file)).unwrap();
    }
    let reopened = Recording::open(run.root(), ReadLimits::default()).unwrap();
    let cache =
        PolicyCache::open(dir.path().join("cache"), &reopened, CacheLimits::default()).unwrap();
    let entry = cache.find(&reopened.entries()[0], &spec).unwrap().unwrap();
    assert_eq!(cache.load(&entry).unwrap().rows(), 6);
    let again = precompute(
        &reopened,
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!((again.skipped, again.restores), (3, 0));
    let mut missing = spec;
    missing.ranges[0] = [-5., 5.];
    let failed = precompute(
        &reopened,
        &cache,
        &[missing],
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (failed.computed, failed.unavailable, failed.restores),
        (0, 3, 0)
    );
}

#[test]
fn interrupted_and_corrupt_entries_are_explicit_and_completed_entries_are_immutable() {
    use std::io::{Seek, SeekFrom, Write};
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let path = dir.path().join("cache");
    let cache = PolicyCache::open_or_create(&path, &run, CacheLimits::default()).unwrap();
    std::fs::create_dir(path.join("frames").join("abandoned.partial")).unwrap();
    std::fs::write(
        path.join("frames/abandoned.partial/payload.bin"),
        b"partial",
    )
    .unwrap();
    precompute(
        &run,
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(cache.catalog().unwrap().entries.len(), 3);
    let entry = cache.find(&run.entries()[0], &spec).unwrap().unwrap();
    let payload = path.join("frames").join(&entry.key).join("payload.bin");
    let original = std::fs::read(&payload).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&payload)
        .unwrap();
    file.seek(SeekFrom::Start(15)).unwrap();
    file.write_all(b"corrupt").unwrap();
    drop(file);
    assert!(cache.load(&entry).is_err());
    let summary = precompute(
        &run,
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (summary.failed, summary.skipped, summary.computed),
        (1, 2, 0)
    );
    assert_ne!(
        std::fs::read(&payload).unwrap(),
        original,
        "corrupt completed entries are not silently replaced"
    );
    std::fs::write(&payload, original).unwrap();
    assert!(cache.load(&entry).is_ok());
    let limited = PolicyCache::open(
        &path,
        &run,
        CacheLimits {
            max_surface_bytes: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(limited.load(&entry).is_err());

    let bounded_path = dir.path().join("bounded-cache");
    let bounded = PolicyCache::open_or_create(
        &bounded_path,
        &run,
        CacheLimits {
            max_frame_bytes: 64,
            ..Default::default()
        },
    )
    .unwrap();
    let failed = precompute(
        &run,
        &bounded,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!((failed.computed, failed.failed), (0, 3));
    assert!(bounded.catalog().unwrap().entries.is_empty());
    assert_eq!(
        std::fs::read_dir(bounded_path.join("frames"))
            .unwrap()
            .count(),
        0,
        "failed publication releases its own partial entry"
    );
    let retry = PolicyCache::open(&bounded_path, &run, CacheLimits::default()).unwrap();
    let resumed = precompute(&run, &retry, &[spec], &PrecomputeOptions::default(), |_| {}).unwrap();
    assert_eq!(resumed.computed, 3);
}

#[test]
fn dry_run_is_read_only_and_estimates_work_without_restoring_models() {
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let cache = PolicyCache::open_or_create(dir.path().join("cache"), &run, CacheLimits::default())
        .unwrap();
    let options = PrecomputeOptions {
        dry_run: true,
        ..Default::default()
    };
    let summary = precompute(&run, &cache, &[spec], &options, |_| {}).unwrap();
    assert_eq!(
        (
            summary.planned,
            summary.estimated_rows,
            summary.computed,
            summary.restores
        ),
        (3, 18, 0, 0)
    );
    assert!(summary.estimated_output_bytes > 0);
    assert_eq!(cache.catalog().unwrap().entries.len(), 0);
}

#[test]
fn incompatible_policy_is_isolated_and_unknown_roles_fail_explicitly() {
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let cache = PolicyCache::open_or_create(dir.path().join("cache"), &run, CacheLimits::default())
        .unwrap();
    let mut unknown = spec.clone();
    unknown.policy = "not-in-checkpoint".into();
    let planned = precompute(
        &run,
        &cache,
        &[unknown.clone(), spec.clone()],
        &PrecomputeOptions {
            dry_run: true,
            ..Default::default()
        },
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (planned.planned, planned.failed, planned.restores),
        (3, 3, 0),
        "dry-run must validate named adapters without restoration"
    );
    let summary = precompute(
        &run,
        &cache,
        &[unknown, spec.clone()],
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (summary.computed, summary.failed, summary.restores),
        (3, 3, 3)
    );
    let mut role = spec;
    role.model_role = "absent-role".into();
    assert!(precompute(&run, &cache, &[role], &PrecomputeOptions::default(), |_| {}).is_err());
}

#[test]
fn cli_accepts_exported_multi_slice_input_and_dry_run_creates_no_destination() {
    use std::process::Command;
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let slices = dir.path().join("slices.json");
    let mut second = spec.clone();
    second.ranges[0] = [-2., 2.];
    std::fs::write(&slices, serde_json::to_vec(&vec![spec, second]).unwrap()).unwrap();
    let cache = dir.path().join("cache");
    let output = Command::new(env!("CARGO_BIN_EXE_rl-precompute"))
        .arg("policy")
        .arg("--run")
        .arg(run.root())
        .arg("--slices")
        .arg(&slices)
        .arg("--cache")
        .arg(&cache)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!cache.exists());
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["planned"], 6);
    assert_eq!(summary["restores"], 0);
    let output = Command::new(env!("CARGO_BIN_EXE_rl-precompute"))
        .arg("policy")
        .arg("--run")
        .arg(run.root())
        .arg("--slices")
        .arg(&slices)
        .arg("--cache")
        .arg(&cache)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["computed"], 6);
    assert_eq!(
        summary["restores"], 3,
        "all slices share each restored checkpoint"
    );
    let cache = PolicyCache::open(&cache, &run, CacheLimits::default()).unwrap();
    assert_eq!(cache.catalog().unwrap().entries.len(), 6);
}

#[test]
fn cache_rejects_changed_source_manifest_and_entry_metadata() {
    let (dir, run, spec) = fixture(false, ActionSpace::Unbounded { dimensions: 2 });
    let path = dir.path().join("cache");
    let cache = PolicyCache::open_or_create(&path, &run, CacheLimits::default()).unwrap();
    precompute(
        &run,
        &cache,
        std::slice::from_ref(&spec),
        &PrecomputeOptions::default(),
        |_| {},
    )
    .unwrap();
    let entry = cache.find(&run.entries()[0], &spec).unwrap().unwrap();
    let surface = cache.load(&entry).unwrap();
    assert!(
        cache
            .publish(&run.entries()[0], &"a".repeat(64), &surface)
            .is_err(),
        "an existing entry must reject conflicting model provenance"
    );
    let header = path.join("frames").join(&entry.key).join("entry.json");
    let original = std::fs::read(&header).unwrap();
    let mut corrupted: serde_json::Value = serde_json::from_slice(&original).unwrap();
    corrupted["model_content_sha256"] = "a".repeat(64).into();
    std::fs::write(&header, serde_json::to_vec(&corrupted).unwrap()).unwrap();
    assert!(
        cache.find(&run.entries()[0], &spec).is_err(),
        "metadata corruption must invalidate model provenance"
    );
    std::fs::write(&header, original).unwrap();
    let mut altered = run.entries()[0].clone();
    altered.checksum ^= 1;
    assert!(cache.find(&altered, &spec).is_err());
    let mut entry = cache.find(&run.entries()[0], &spec).unwrap().unwrap();
    entry.slice.resolution[0] += 1;
    assert!(cache.load(&entry).is_err());
    let manifest = run.root().join("manifest.json");
    let mut data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    data["model_id"] = "different-experiment".into();
    std::fs::write(&manifest, serde_json::to_vec(&data).unwrap()).unwrap();
    let changed = Recording::open(run.root(), ReadLimits::default()).unwrap();
    assert!(PolicyCache::open(&path, &changed, CacheLimits::default()).is_err());
}
