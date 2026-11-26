use rl::{
    core::{action::ActionSpace, env::Observation},
    diagnostics::{probes::ProbeSet, schema::*, storage::*},
};
use rl_viewer::{
    data::{ChunkCache, RunIndex, SelectedInput, Selection},
    resolver::{ProbeResolver, ResolveStatus},
};
use std::{collections::BTreeMap, sync::Arc};

fn context(k: u64) -> RecordContext {
    let mut c = RecordContext::new("resolve", k);
    c.probe_set = Some("fixed".into());
    c
}
fn probes(k: u64, reversed: bool, changed: bool) -> Chunk {
    let obs = if reversed { vec![2., 1.] } else { vec![1., 2.] };
    let mut chunk = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        if changed {
            "different population methodology"
        } else {
            "stable grid"
        },
        obs.into_iter().map(|v| Observation::new(vec![v])).collect(),
        None,
        ActionSpace::Discrete { count: 2 },
    )
    .unwrap()
    .record(context(k))
    .unwrap();
    if reversed {
        let Record::ProbeSet(p) = &chunk.record else {
            unreachable!()
        };
        chunk
            .arrays
            .insert(p.row_ids.clone(), Array::u64(vec![2], vec![1, 0]).unwrap());
    }
    chunk
}
fn policy(k: u64) -> Chunk {
    Chunk {
        context: context(k),
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
            masks: None,
            stochastic: false,
        }),
        arrays: BTreeMap::from([
            ("actions".into(), Array::u64(vec![2], vec![0, 1]).unwrap()),
            (
                "h".into(),
                Array::f32(vec![2, 1], vec![k as f32, -(k as f32)]).unwrap(),
            ),
        ]),
    }
}

fn fixture() -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "resolve",
            "test",
            "ppo",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for k in [1, 21, 41] {
        writer.append(&probes(k, k == 21, false)).unwrap();
        writer.append(&policy(k)).unwrap();
    }
    writer.finish().unwrap();
    (dir, RunIndex::open(path, ReadLimits::default()).unwrap())
}
fn selection(k: u64) -> Selection {
    Selection {
        revision: k,
        input: SelectedInput::ProbeRow {
            set: "fixed".into(),
            row: 0,
            sequence: 0,
        },
        ..Default::default()
    }
}
#[test]
fn sparse_resolution_uses_recorded_probe_rows_and_never_a_future_revision() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(view.status, ResolveStatus::Ready);
    let frame = view.record.unwrap();
    assert_eq!(frame.revision(), 21);
    assert_eq!(frame.row, 1);
    assert_eq!(frame.probe_key.sequence, 2);
    assert_eq!(frame.chunk.arrays["h"].as_f32().unwrap()[frame.row], -21.);
    let empty = resolver
        .resolve_policy(&run, &selection(0), |_| {
            panic!("future data must not be read")
        })
        .unwrap();
    assert_eq!(empty.status, ResolveStatus::Unavailable);
    assert!(empty.record.is_none());
}
#[test]
fn replacement_loading_retains_only_a_compatible_earlier_frame() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    resolver
        .resolve_policy(&run, &selection(20), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    let held = resolver
        .resolve_policy(&run, &selection(37), |i| {
            if i == 3 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            }
        })
        .unwrap();
    assert_eq!(held.status, ResolveStatus::Loading);
    assert_eq!(held.record.unwrap().revision(), 1);
    let mut changed = selection(37);
    changed.model_role = "target".into();
    let other = resolver
        .resolve_policy(&run, &changed, |_| {
            panic!("unavailable role has no inputs to load")
        })
        .unwrap();
    assert!(other.record.is_none());
    assert_eq!(other.status, ResolveStatus::Unavailable);
}
#[test]
fn corrupt_latest_record_falls_back_with_a_warning_and_bounded_retention() {
    let (_dir, run) = fixture();
    std::fs::remove_file(run.root().join(&run.recording().entries()[3].file)).unwrap();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(view.record.unwrap().revision(), 1);
    assert!(!view.warnings.is_empty());
    assert!(resolver.resident_bytes() <= 1024 * 1024);
}
#[test]
fn changed_methodology_and_identity_do_not_match_even_when_observations_match() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(if i == 2 {
                probes(21, true, true)
            } else {
                run.recording().load(i)?
            })))
        })
        .unwrap();
    assert_eq!(view.record.unwrap().revision(), 1);
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            let mut chunk = run.recording().load(i)?;
            if i == 2 {
                let Record::ProbeSet(p) = &chunk.record else {
                    unreachable!()
                };
                chunk.arrays.insert(
                    p.row_ids.clone(),
                    Array::u64(vec![2], vec![99, 98]).unwrap(),
                );
            }
            Ok(Some(Arc::new(chunk)))
        })
        .unwrap();
    assert_eq!(view.record.unwrap().revision(), 1);
}

#[test]
fn mismatched_measurement_population_is_rejected_even_for_in_bounds_row() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            let mut chunk = run.recording().load(i)?;
            if i == 3 {
                chunk
                    .arrays
                    .insert("actions".into(), Array::u64(vec![1], vec![1]).unwrap());
            }
            Ok(Some(Arc::new(chunk)))
        })
        .unwrap();
    assert_eq!(view.record.unwrap().revision(), 1);
    assert!(!view.warnings.is_empty());
}

#[test]
fn insufficient_budget_never_pins_an_oversized_frame() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1);
    let view = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(view.status, ResolveStatus::Unavailable);
    assert!(view.record.is_none());
    assert_eq!(resolver.resident_bytes(), 0);
    assert!(!view.warnings.is_empty());
}

#[test]
fn pending_dependencies_survive_a_raw_cache_that_holds_only_one_chunk() {
    let (_dir, run) = fixture();
    let mut cache = ChunkCache::new(
        (0..6)
            .map(|i| ChunkCache::charge(&run.recording().load(i).unwrap()))
            .max()
            .unwrap(),
    );
    let budget = 1024 * 1024;
    let mut resolver = ProbeResolver::new(budget);
    let mut view = None;
    for _ in 0..16 {
        let next = resolver
            .resolve_policy(&run, &selection(37), |i| {
                let key = run.key(i)?;
                if let Some(chunk) = cache.get(&key) {
                    return Ok(Some(chunk));
                }
                assert!(cache.insert(key, Arc::new(run.recording().load(i)?)));
                Ok(None)
            })
            .unwrap();
        assert!(resolver.resident_bytes() <= budget);
        if next.status != ResolveStatus::Loading {
            view = Some(next);
            break;
        }
    }
    let frame = view
        .expect("bounded cache churn must not leave a request loading forever")
        .record
        .unwrap();
    assert_eq!(frame.revision(), 21);
    assert_eq!(
        resolver.resident_bytes(),
        ChunkCache::charge(&frame.chunk) + ChunkCache::charge(&frame.probes),
        "completed searches must release input and candidate pins"
    );
}

#[test]
fn pending_dependencies_are_charged_and_released_when_the_requested_time_changes() {
    let (_dir, run) = fixture();
    let source_charge = ChunkCache::charge(&run.recording().load(0).unwrap());
    let candidate_charge = ChunkCache::charge(&run.recording().load(3).unwrap());
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let pending = resolver
        .resolve_policy(&run, &selection(37), |i| {
            if i == 2 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            }
        })
        .unwrap();
    assert_eq!(pending.status, ResolveStatus::Loading);
    assert_eq!(resolver.resident_bytes(), source_charge + candidate_charge);
    let absent = resolver
        .resolve_policy(&run, &selection(0), |_| panic!("no eligible candidates"))
        .unwrap();
    assert_eq!(absent.status, ResolveStatus::Unavailable);
    assert_eq!(resolver.resident_bytes(), 0);
}

#[test]
fn semantic_selection_changes_release_obsolete_pending_inputs() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    resolver
        .resolve_policy(&run, &selection(37), |i| {
            if i == 2 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            }
        })
        .unwrap();
    let mut different = selection(37);
    if let SelectedInput::ProbeRow { row, .. } = &mut different.input {
        *row = 1;
    }
    let pending = resolver
        .resolve_policy(&run, &different, |i| {
            if i == 0 {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            } else {
                Ok(None)
            }
        })
        .unwrap();
    assert_eq!(pending.status, ResolveStatus::Loading);
    assert!(pending.record.is_none());
    assert_eq!(
        resolver.resident_bytes(),
        ChunkCache::charge(&run.recording().load(0).unwrap())
    );
}

#[test]
fn backward_pending_selection_cannot_hold_a_future_frame() {
    let (_dir, run) = fixture();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    resolver
        .resolve_policy(&run, &selection(41), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    let view = resolver
        .resolve_policy(&run, &selection(37), |_| Ok(None))
        .unwrap();
    assert_eq!(view.status, ResolveStatus::Loading);
    assert!(view.record.is_none());
    resolver.clear();
    assert_eq!(resolver.resident_bytes(), 0);
}

#[test]
fn invalid_selected_source_row_is_unavailable_with_a_reason() {
    let (_dir, run) = fixture();
    let mut invalid = selection(37);
    if let SelectedInput::ProbeRow { row, .. } = &mut invalid.input {
        *row = usize::MAX;
    }
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let view = resolver
        .resolve_policy(&run, &invalid, |i| {
            let mut chunk = run.recording().load(i)?;
            if i == 0 {
                let Record::ProbeSet(p) = &mut chunk.record else {
                    unreachable!()
                };
                p.action_masks = None;
            }
            Ok(Some(Arc::new(chunk)))
        })
        .unwrap();
    assert_eq!(view.status, ResolveStatus::Unavailable);
    assert!(view.record.is_none());
    assert!(!view.warnings.is_empty());
    assert_eq!(resolver.resident_bytes(), 0);
}

#[test]
fn held_frames_and_pending_dependencies_share_the_same_budget() {
    let (_dir, run) = fixture();
    let source = ChunkCache::charge(&run.recording().load(0).unwrap());
    let old_frame = source + ChunkCache::charge(&run.recording().load(1).unwrap());
    let candidate = ChunkCache::charge(&run.recording().load(3).unwrap());
    let budget = old_frame + source + candidate - 1;
    let mut resolver = ProbeResolver::new(budget);
    resolver
        .resolve_policy(&run, &selection(20), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    let pending = resolver
        .resolve_policy(&run, &selection(37), |i| {
            if i == 2 {
                Ok(None)
            } else {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            }
        })
        .unwrap();
    assert_eq!(pending.status, ResolveStatus::Loading);
    assert!(
        pending.record.is_none(),
        "held frame must not escape the pin budget"
    );
    assert!(
        !pending.warnings.is_empty(),
        "releasing a held frame must be visible"
    );
    assert_eq!(resolver.resident_bytes(), source + candidate);
    assert!(resolver.resident_bytes() <= budget);
}

#[test]
fn a_working_set_over_budget_falls_back_instead_of_reloading_forever() {
    let (_dir, run) = fixture();
    let source = ChunkCache::charge(&run.recording().load(0).unwrap());
    let newest = ChunkCache::charge(&run.recording().load(3).unwrap());
    let budget = source + newest - 1;
    let mut cache = ChunkCache::new(
        (0..6)
            .map(|i| ChunkCache::charge(&run.recording().load(i).unwrap()))
            .max()
            .unwrap(),
    );
    let mut resolver = ProbeResolver::new(budget);
    let mut result = None;
    for _ in 0..16 {
        let view = resolver
            .resolve_policy(&run, &selection(37), |i| {
                let key = run.key(i)?;
                if let Some(chunk) = cache.get(&key) {
                    return Ok(Some(chunk));
                }
                assert!(cache.insert(key, Arc::new(run.recording().load(i)?)));
                Ok(None)
            })
            .unwrap();
        assert!(resolver.resident_bytes() <= budget);
        if view.status != ResolveStatus::Loading {
            result = Some(view);
            break;
        }
    }
    let result = result.expect("a memory limit must produce an explicit outcome");
    assert_eq!(result.record.unwrap().revision(), 1);
    assert!(!result.warnings.is_empty());
}

#[test]
fn live_append_invalidates_a_finished_resolution_at_the_same_requested_time() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "resolve",
            "test",
            "ppo",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    writer.append(&probes(1, false, false)).unwrap();
    writer.append(&policy(1)).unwrap();
    writer.flush().unwrap();
    let mut run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let first = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(first.record.unwrap().revision(), 1);
    writer.append(&probes(21, true, false)).unwrap();
    writer.append(&policy(21)).unwrap();
    writer.flush().unwrap();
    assert_eq!(run.refresh().unwrap(), 2);
    let appended = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(appended.record.as_ref().unwrap().revision(), 21);
    assert_eq!(appended.record.unwrap().row, 1);
}

#[test]
fn live_metrics_do_not_restart_unchanged_pending_dependencies() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "resolve",
            "test",
            "ppo",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    for k in [1, 21] {
        writer.append(&probes(k, k == 21, false)).unwrap();
        writer.append(&policy(k)).unwrap();
    }
    writer.flush().unwrap();
    let mut run = RunIndex::open(path, ReadLimits::default()).unwrap();
    let mut cache = ChunkCache::new(
        (0..4)
            .map(|i| ChunkCache::charge(&run.recording().load(i).unwrap()))
            .max()
            .unwrap(),
    );
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let mut result = None;
    for i in 0..16 {
        writer
            .append(&Chunk {
                context: context(100 + i),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), 1.0)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
        writer.flush().unwrap();
        run.refresh().unwrap();
        let next = resolver
            .resolve_policy(&run, &selection(37), |i| {
                let key = run.key(i)?;
                if let Some(chunk) = cache.get(&key) {
                    return Ok(Some(chunk));
                }
                assert!(cache.insert(key, Arc::new(run.recording().load(i)?)));
                Ok(None)
            })
            .unwrap();
        if next.status != ResolveStatus::Loading {
            result = Some(next);
            break;
        }
    }
    assert_eq!(
        result
            .expect("unrelated appended metrics must not starve pending inputs")
            .record
            .unwrap()
            .revision(),
        21
    );
}

#[test]
fn playback_between_measurements_preserves_pending_dependencies() {
    let (_dir, run) = fixture();
    let mut cache = ChunkCache::new(
        (0..6)
            .map(|i| ChunkCache::charge(&run.recording().load(i).unwrap()))
            .max()
            .unwrap(),
    );
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let mut result = None;
    for revision in 22..=37 {
        let next = resolver
            .resolve_policy(&run, &selection(revision), |i| {
                let key = run.key(i)?;
                if let Some(chunk) = cache.get(&key) {
                    return Ok(Some(chunk));
                }
                assert!(cache.insert(key, Arc::new(run.recording().load(i)?)));
                Ok(None)
            })
            .unwrap();
        if next.status == ResolveStatus::Ready {
            result = Some(next);
            break;
        }
    }
    assert_eq!(result.expect("advancing requested time between measurements must not restart the same dependency loads").record.unwrap().revision(), 21);
}

fn crowded_run() -> (tempfile::TempDir, Recorder, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run");
    let mut writer = Recorder::create(
        &path,
        RunManifest::new(
            "resolve",
            "test",
            "ppo",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap();
    writer.append(&probes(1, false, false)).unwrap();
    writer.append(&policy(1)).unwrap();
    writer.append(&probes(21, false, true)).unwrap();
    for _ in 0..130 {
        let chunk = policy(21);
        writer.append(&chunk).unwrap();
    }
    writer.flush().unwrap();
    let run = RunIndex::open(path, ReadLimits::default()).unwrap();
    (dir, writer, run)
}

#[test]
fn playback_preserves_cursor_past_a_large_incompatible_prefix() {
    let (_dir, _writer, run) = crowded_run();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let mut result = None;
    for k in 37..41 {
        let view = resolver
            .resolve_policy(&run, &selection(k), |i| {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            })
            .unwrap();
        if view.status == ResolveStatus::Ready {
            result = view.record;
            break;
        }
    }
    assert_eq!(
        result
            .expect("playback must advance beyond the rejected prefix")
            .revision(),
        1
    );
}

#[test]
fn unrelated_appends_preserve_cursor_past_a_large_incompatible_prefix() {
    let (_dir, mut writer, mut run) = crowded_run();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let mut result = None;
    for k in 100..104 {
        writer
            .append(&Chunk {
                context: context(k),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::from([("loss".into(), 1.0)]),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
        writer.flush().unwrap();
        run.refresh().unwrap();
        let view = resolver
            .resolve_policy(&run, &selection(37), |i| {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            })
            .unwrap();
        if view.status == ResolveStatus::Ready {
            result = view.record;
            break;
        }
    }
    assert_eq!(
        result
            .expect("metrics must not restart the rejected prefix")
            .revision(),
        1
    );
}

#[test]
fn late_older_candidate_invalidates_resolution_even_when_the_newest_key_is_unchanged() {
    let (_dir, mut writer, mut run) = crowded_run();
    let mut resolver = ProbeResolver::new(1024 * 1024);
    let mut first = None;
    for _ in 0..4 {
        let view = resolver
            .resolve_policy(&run, &selection(37), |i| {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            })
            .unwrap();
        if view.status == ResolveStatus::Ready {
            first = view.record;
            break;
        }
    }
    assert_eq!(first.unwrap().revision(), 1);
    writer.append(&probes(10, false, false)).unwrap();
    writer.append(&policy(10)).unwrap();
    writer.flush().unwrap();
    run.refresh().unwrap();
    let mut latest = None;
    for _ in 0..4 {
        let view = resolver
            .resolve_policy(&run, &selection(37), |i| {
                Ok(Some(Arc::new(run.recording().load(i)?)))
            })
            .unwrap();
        if view.status == ResolveStatus::Ready {
            latest = view.record;
            break;
        }
    }
    assert_eq!(latest.unwrap().revision(), 10);
}

#[test]
fn returned_frames_and_chunk_clones_remain_charged_after_eviction() {
    let (_dir, run) = fixture();
    let frame_charge = ChunkCache::charge(&run.recording().load(3).unwrap())
        + ChunkCache::charge(&run.recording().load(2).unwrap());
    let budget = 2 * frame_charge;
    let mut resolver = ProbeResolver::new(budget);
    let first_selection = selection(37);
    let mut second_selection = first_selection.clone();
    if let SelectedInput::ProbeRow { row, .. } = &mut second_selection.input {
        *row = 1;
    }
    let first = resolver
        .resolve_policy(&run, &first_selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let second = resolver
        .resolve_policy(&run, &second_selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let waiting = resolver
        .resolve_policy(&run, &selection(41), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert!(
        waiting.record.is_none(),
        "externally held frames leave no room for another frame"
    );
    assert_eq!(
        resolver.resident_bytes(),
        budget,
        "eviction must not forget returned owners"
    );
    assert!(!waiting.warnings.is_empty());
    let retained_chunk = first.chunk.clone();
    drop(first);
    resolver.clear();
    assert_eq!(resolver.resident_bytes(), budget);
    drop(retained_chunk);
    assert_eq!(resolver.resident_bytes(), frame_charge);
    drop(second);
    assert_eq!(resolver.resident_bytes(), 0);
    let resumed = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(resumed.status, ResolveStatus::Ready);
}

#[test]
fn releasing_a_returned_frame_unblocks_a_waiting_query_without_refresh() {
    let (_dir, run) = fixture();
    let charge = ChunkCache::charge(&run.recording().load(3).unwrap())
        + ChunkCache::charge(&run.recording().load(2).unwrap());
    let mut resolver = ProbeResolver::new(charge);
    let frame = resolver
        .resolve_policy(&run, &selection(37), |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap()
        .record
        .unwrap();
    let mut other_selection = selection(37);
    if let SelectedInput::ProbeRow { row, .. } = &mut other_selection.input {
        *row = 1;
    }
    let waiting = resolver
        .resolve_policy(&run, &other_selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(waiting.status, ResolveStatus::Loading);
    assert!(waiting.record.is_none());
    assert_eq!(resolver.resident_bytes(), charge);
    drop(frame);
    let resumed = resolver
        .resolve_policy(&run, &other_selection, |i| {
            Ok(Some(Arc::new(run.recording().load(i)?)))
        })
        .unwrap();
    assert_eq!(resumed.status, ResolveStatus::Ready);
    assert_eq!(resumed.record.unwrap().revision(), 21);
    assert_eq!(resolver.resident_bytes(), charge);
}
