//! Standalone numerical precomputation; this binary never links the viewer.
use rl::diagnostics::{
    policy_cache::{CacheLimits, PolicyCache},
    policy_precompute::{PrecomputeEvent, PrecomputeOptions, precompute, read_slice_specs},
    storage::{ReadLimits, Recording},
};
use std::{collections::BTreeMap, path::PathBuf};

const HELP: &str = "rl-precompute policy --run RUN --slices FILE [--checkpoints all|SEQUENCE,...]
  --cache DIR                  Default: RUN-policy-cache (sibling directory)
  --dry-run                    Estimate missing work; zero model restoration
  --backend cpu                Historical Burn Flex CPU backend (only supported backend)
  --max-parameters N           Default: 16777216
  --max-model-bytes N          Default: 67108864; backend/model admission
  --max-models N               Default: 64 named models per snapshot
  --snapshot-bytes N           Default: 67108864; encoded/decoded recording chunk
  --batch-rows N               Default: 1024; inference batch upper bound
  --scratch-bytes N            Default: 16777216; inspection scratch
  --surface-bytes N            Default: 134217728; retained numerical surface
  --cache-frame-bytes N        Default: 268435456; encoded cache frame
  --cache-chunk-bytes N        Default: 67108864; encoded/decoded per-batch chunk
  --cache-index-bytes N        Default: 134217728; catalog metadata allowance
Slice input accepts one PolicySliceSpec object or an array of up to 256 specs.
Checkpoint selectors use immutable recording sequence numbers, not learner revisions.
Numerical caches preserve precision and never modify the recording.";

fn main() {
    if let Err(error) = run() {
        eprintln!("rl-precompute: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    if command
        .as_deref()
        .is_none_or(|a| a == "--help" || a == "-h")
    {
        println!("{HELP}");
        return Ok(());
    }
    if command.as_deref() != Some("policy") {
        return Err("expected policy subcommand; use --help".into());
    }
    let mut values = BTreeMap::new();
    let mut dry_run = false;
    while let Some(key) = args.next() {
        if key == "--help" || key == "-h" {
            println!("{HELP}");
            return Ok(());
        }
        if key == "--dry-run" {
            dry_run = true;
            continue;
        }
        if ![
            "--run",
            "--slices",
            "--cache",
            "--checkpoints",
            "--backend",
            "--max-parameters",
            "--max-model-bytes",
            "--max-models",
            "--snapshot-bytes",
            "--batch-rows",
            "--scratch-bytes",
            "--surface-bytes",
            "--cache-frame-bytes",
            "--cache-chunk-bytes",
            "--cache-index-bytes",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}; use --help").into());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option {key}").into());
        }
    }
    let required = |key: &str| {
        values
            .get(key)
            .ok_or_else(|| format!("required option {key}"))
    };
    let run_path = PathBuf::from(required("--run")?);
    let slices_path = PathBuf::from(required("--slices")?);
    if values.get("--backend").is_some_and(|v| v != "cpu") {
        return Err("only the existing historical Burn Flex CPU backend is supported; GPU inference is not available in this adapter".into());
    }
    let number = |key: &str, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
        let value = values
            .get(key)
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(default);
        if value == 0 {
            return Err(format!("{key} must be positive").into());
        }
        Ok(value)
    };
    let mut options = PrecomputeOptions {
        dry_run,
        ..Default::default()
    };
    options.models.max_parameters = number("--max-parameters", options.models.max_parameters)?;
    options.models.max_model_bytes = number("--max-model-bytes", options.models.max_model_bytes)?;
    options.models.max_models = number("--max-models", options.models.max_models)?;
    options.models.max_batch = number("--batch-rows", options.models.max_batch)?;
    options.surfaces.max_batch = options.models.max_batch;
    options.surfaces.max_scratch_bytes =
        number("--scratch-bytes", options.surfaces.max_scratch_bytes)?;
    options.surfaces.max_surface_bytes =
        number("--surface-bytes", options.surfaces.max_surface_bytes)?;
    if let Some(selection) = values.get("--checkpoints").filter(|s| s.as_str() != "all") {
        options.sequences = Some(
            selection
                .split(',')
                .map(str::parse)
                .collect::<Result<Vec<u64>, _>>()?,
        );
    }
    let read_limits = ReadLimits {
        max_chunk_bytes: number("--snapshot-bytes", ReadLimits::default().max_chunk_bytes)?,
        ..Default::default()
    };
    let recording = Recording::open(&run_path, read_limits)?;
    if !recording.issues().is_empty() {
        return Err(format!(
            "recording index has issues: {}",
            recording.issues().join("; ")
        )
        .into());
    }
    let slices = read_slice_specs(&slices_path)?;
    let mut cache_limits = CacheLimits {
        max_surface_bytes: options.surfaces.max_surface_bytes,
        ..Default::default()
    };
    cache_limits.max_frame_bytes = number("--cache-frame-bytes", cache_limits.max_frame_bytes)?;
    cache_limits.max_chunk_bytes = number("--cache-chunk-bytes", cache_limits.max_chunk_bytes)?;
    cache_limits.max_index_bytes = number("--cache-index-bytes", cache_limits.max_index_bytes)?;
    let cache_path = values.get("--cache").map(PathBuf::from).unwrap_or_else(|| {
        let mut name = run_path.file_name().unwrap_or_default().to_os_string();
        name.push("-policy-cache");
        run_path.with_file_name(name)
    });
    let cache = if dry_run {
        PolicyCache::open_or_empty(&cache_path, &recording, cache_limits)?
    } else {
        PolicyCache::open_or_create(&cache_path, &recording, cache_limits)?
    };
    let result = precompute(&recording, &cache, &slices, &options, |event| match event {
        PrecomputeEvent::Progress {
            sequence,
            slice,
            completed_rows,
            total_rows,
            retained_bytes,
        } => eprintln!(
            "checkpoint {sequence}, slice {slice}: {completed_rows}/{total_rows} rows, retained {retained_bytes} bytes"
        ),
        event => eprintln!("{event:?}"),
    });
    let summary = result?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    if summary.failed > 0 || summary.unavailable > 0 {
        return Err(
            "some requested computations failed or required weights were unavailable".into(),
        );
    }
    Ok(())
}
