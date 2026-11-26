//! Sequential offline orchestration. One restored checkpoint serves all missing
//! slices, and one numerical surface is published and released at a time.
use super::{
    models::{ModelBundle, ModelKind, ModelLimits, snapshot_configuration},
    policy_cache::{PolicyCache, model_content_hash},
    policy_slice::{GridEligibility, PolicySliceSpec},
    policy_surface::{SurfaceLimits, action_count, batch_capacity, evaluate_surface},
    schema::{Record, RecordKind},
    storage::Recording,
    visualization::EnvironmentSemantics,
};
use crate::{Result, core::ensure};
use serde::Serialize;
use std::time::Instant;

pub fn read_slice_specs(path: impl AsRef<std::path::Path>) -> Result<Vec<PolicySliceSpec>> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Input {
        One(PolicySliceSpec),
        Many(Vec<PolicySliceSpec>),
    }
    let bytes = super::policy_cache::read_limited(path.as_ref(), 1024 * 1024)?;
    Ok(match serde_json::from_slice::<Input>(&bytes)? {
        Input::One(s) => vec![s],
        Input::Many(s) => s,
    })
}

#[derive(Debug, Clone, Default)]
pub struct PrecomputeOptions {
    pub models: ModelLimits,
    pub surfaces: SurfaceLimits,
    /// Immutable recording sequence numbers, never ambiguous learner revisions.
    pub sequences: Option<Vec<u64>>,
    pub dry_run: bool,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrecomputeSummary {
    pub computed: usize,
    pub skipped: usize,
    pub failed: usize,
    pub unavailable: usize,
    pub planned: usize,
    pub estimated_rows: usize,
    pub estimated_output_bytes: u64,
    pub restores: usize,
    pub inference_batches: usize,
    pub evaluated_rows: usize,
    pub snapshot_load_micros: u64,
    pub source_hash_micros: u64,
    pub restore_micros: u64,
    pub evaluation_micros: u64,
    pub publication_micros: u64,
    pub output_bytes: u64,
    pub peak_surface_bytes: usize,
}
#[derive(Debug)]
pub enum PrecomputeEvent {
    Restoring {
        sequence: u64,
    },
    Progress {
        sequence: u64,
        slice: usize,
        completed_rows: usize,
        total_rows: usize,
        retained_bytes: usize,
    },
    Completed {
        sequence: u64,
        slice: usize,
        key: String,
    },
    Skipped {
        sequence: u64,
        slice: usize,
    },
    Unavailable {
        sequence: u64,
        slice: usize,
        reason: String,
    },
    Failed {
        sequence: u64,
        slice: usize,
        reason: String,
    },
}
pub fn precompute(
    run: &Recording,
    cache: &PolicyCache,
    slices: &[PolicySliceSpec],
    options: &PrecomputeOptions,
    mut event: impl FnMut(PrecomputeEvent),
) -> Result<PrecomputeSummary> {
    ensure(
        !slices.is_empty() && slices.len() <= 256,
        "precomputation requires 1..256 slices",
    )?;
    let semantics = EnvironmentSemantics::new(run.manifest());
    for spec in slices {
        let config = spec.to_config(semantics)?;
        ensure(
            config.grid_eligibility() == GridEligibility::ContinuousFeedforward
                && !config.is_empty(),
            "precomputation does not synthesize masked, history-bound or samples-only inputs",
        )?;
        batch_capacity(
            config.fixed.len(),
            action_count(semantics.action_space()),
            false,
            options.surfaces.max_scratch_bytes,
            options.surfaces.max_batch.min(options.models.max_batch),
        )?;
    }
    if let Some(sequences) = &options.sequences {
        ensure(
            !sequences.is_empty()
                && sequences.iter().all(|s| {
                    run.entries()
                        .iter()
                        .any(|e| e.sequence == *s && e.kind == RecordKind::ModelSnapshot)
                }),
            "requested checkpoint sequence is absent or is not a model snapshot",
        )?;
    }
    let selected_roles = run
        .entries()
        .iter()
        .filter(|e| {
            e.kind == RecordKind::ModelSnapshot
                && options
                    .sequences
                    .as_ref()
                    .is_none_or(|s| s.contains(&e.sequence))
        })
        .map(|e| e.model_role.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    ensure(
        slices
            .iter()
            .all(|s| selected_roles.contains(s.model_role.as_str())),
        "no selected checkpoints exist for a requested model role",
    )?;
    let unique = slices
        .iter()
        .map(|s| super::policy_cache::hash_json(s, cache.limits().max_metadata_bytes))
        .collect::<Result<std::collections::BTreeSet<_>>>()?;
    ensure(
        unique.len() == slices.len(),
        "duplicate slice specifications; include each slice once",
    )?;
    let mut summary = PrecomputeSummary::default();
    for (index, source) in run.entries().iter().enumerate().filter(|(_, e)| {
        e.kind == RecordKind::ModelSnapshot
            && options
                .sequences
                .as_ref()
                .is_none_or(|s| s.contains(&e.sequence))
    }) {
        let mut missing = Vec::new();
        for (slice, spec) in slices
            .iter()
            .enumerate()
            .filter(|(_, s)| s.model_role == source.model_role)
        {
            match cache
                .find(source, spec)
                .and_then(|e| e.map(|e| cache.verify(&e).map(|()| e)).transpose())
            {
                Ok(Some(_)) => {
                    summary.skipped += 1;
                    event(PrecomputeEvent::Skipped {
                        sequence: source.sequence,
                        slice,
                    });
                }
                Ok(None) => missing.push(slice),
                Err(e) => {
                    summary.failed += 1;
                    event(PrecomputeEvent::Failed {
                        sequence: source.sequence,
                        slice,
                        reason: e.to_string(),
                    });
                }
            }
        }
        if missing.is_empty() {
            continue;
        }
        if let Err(error) = run.source(index)?.check_available() {
            for slice in missing {
                summary.unavailable += 1;
                event(PrecomputeEvent::Unavailable {
                    sequence: source.sequence,
                    slice,
                    reason: error.to_string(),
                });
            }
            continue;
        }
        let start = Instant::now();
        let prepared = (|| {
            let chunk = run.load(index)?;
            let Record::ModelSnapshot(snapshot) = &chunk.record else {
                return Err(crate::Error::Config(
                    "source is not a model snapshot".into(),
                ));
            };
            let configuration = snapshot_configuration(snapshot, options.models)?;
            ensure(
                configuration.observation_dim == semantics.observation_dim()
                    && &configuration.action_space == semantics.action_space(),
                "snapshot and recording encoding differ",
            )?;
            missing.retain(|&slice| {
                let supported = configuration
                    .models
                    .get(&slices[slice].policy)
                    .is_some_and(|m| {
                        m.kind.is_stochastic_policy()
                            || matches!(
                                m.kind,
                                ModelKind::DeterministicPolicy | ModelKind::DiscreteQ
                            )
                    });
                if !supported {
                    summary.failed += 1;
                    event(PrecomputeEvent::Failed {
                        sequence: source.sequence,
                        slice,
                        reason: "selected snapshot has no requested supported policy adapter"
                            .into(),
                    });
                }
                supported
            });
            Ok(chunk)
        })();
        summary.snapshot_load_micros = summary
            .snapshot_load_micros
            .saturating_add(start.elapsed().as_micros() as u64);
        let chunk = match prepared {
            Ok(chunk) => chunk,
            Err(error) => {
                for slice in missing {
                    summary.failed += 1;
                    event(PrecomputeEvent::Failed {
                        sequence: source.sequence,
                        slice,
                        reason: error.to_string(),
                    });
                }
                continue;
            }
        };
        if missing.is_empty() {
            continue;
        }
        for &slice in &missing {
            let config = slices[slice].to_config(semantics)?;
            summary.planned += 1;
            summary.estimated_rows = summary.estimated_rows.saturating_add(config.len()?);
            // Uncompressed numerical estimate: original inputs, actions,
            // distribution parameters, effective masks and f64 projections.
            let capacity = batch_capacity(
                config.fixed.len(),
                action_count(semantics.action_space()),
                false,
                options.surfaces.max_scratch_bytes,
                options.surfaces.max_batch.min(options.models.max_batch),
            )?;
            summary.estimated_output_bytes = summary.estimated_output_bytes.saturating_add(
                (config.len()? as u64)
                    .saturating_mul(
                        (config.fixed.len() as u64)
                            .saturating_mul(4)
                            .saturating_add(
                                (action_count(semantics.action_space()) as u64).saturating_mul(40),
                            )
                            .saturating_add(8),
                    )
                    .saturating_add(
                        (config.len()?.div_ceil(capacity) as u64).saturating_mul(16 * 1024),
                    ),
            );
        }
        if options.dry_run {
            continue;
        }
        event(PrecomputeEvent::Restoring {
            sequence: source.sequence,
        });
        let start = Instant::now();
        let hash = model_content_hash(&chunk, cache.limits().max_metadata_bytes)?;
        summary.source_hash_micros = summary
            .source_hash_micros
            .saturating_add(start.elapsed().as_micros() as u64);
        let start = Instant::now();
        let restore = ModelBundle::load(&chunk, options.models);
        summary.restore_micros = summary
            .restore_micros
            .saturating_add(start.elapsed().as_micros() as u64);
        let model = match restore {
            Ok(model) => model,
            Err(error) => {
                for slice in missing {
                    summary.failed += 1;
                    event(PrecomputeEvent::Failed {
                        sequence: source.sequence,
                        slice,
                        reason: error.to_string(),
                    });
                }
                continue;
            }
        };
        drop(chunk);
        summary.restores += 1;
        for slice in missing {
            let start = Instant::now();
            let surface = evaluate_surface(
                &model,
                &slices[slice],
                semantics,
                options.surfaces,
                || false,
                |p| {
                    summary.inference_batches += 1;
                    summary.peak_surface_bytes = summary.peak_surface_bytes.max(p.retained_bytes);
                    event(PrecomputeEvent::Progress {
                        sequence: source.sequence,
                        slice,
                        completed_rows: p.completed_rows,
                        total_rows: p.total_rows,
                        retained_bytes: p.retained_bytes,
                    });
                },
            );
            summary.evaluation_micros = summary
                .evaluation_micros
                .saturating_add(start.elapsed().as_micros() as u64);
            let result = surface.and_then(|surface| {
                summary.evaluated_rows += surface.rows();
                let start = Instant::now();
                let published = cache.publish(source, &hash, &surface);
                summary.publication_micros = summary
                    .publication_micros
                    .saturating_add(start.elapsed().as_micros() as u64);
                published
            });
            match result {
                Ok(entry) => {
                    summary.computed += 1;
                    summary.output_bytes = summary.output_bytes.saturating_add(entry.payload_bytes);
                    event(PrecomputeEvent::Completed {
                        sequence: source.sequence,
                        slice,
                        key: entry.key,
                    });
                }
                Err(e) => {
                    summary.failed += 1;
                    event(PrecomputeEvent::Failed {
                        sequence: source.sequence,
                        slice,
                        reason: e.to_string(),
                    });
                }
            }
        }
    }
    Ok(summary)
}
