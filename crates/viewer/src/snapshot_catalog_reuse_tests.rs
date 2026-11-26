use super::*;
use rl::diagnostics::{storage::ReadLimits, value::ValueOutput};
use std::sync::Arc;

fn finish_catalog(
    catalog: &mut SnapshotCatalog,
    run: &RunIndex,
    loader: &mut Loader,
    interactive: &mut Interactive,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while catalog.pending() {
        loader.poll();
        catalog.advance(run, loader, interactive).unwrap();
        assert!(
            std::time::Instant::now() < deadline,
            "catalogue did not settle"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn retire_worker(interactive: &mut Interactive) {
    interactive.retire_session();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(
            std::time::Instant::now() < deadline,
            "worker did not retire"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn restoration_witnesses_fit_the_existing_catalogue_envelope() {
    // Policy/Value catalogues never retain Network choices. Each entry admits
    // at most 128 bytes for role/name and 128 UTF-8 error characters.
    let entry = std::mem::size_of::<SnapshotEntry>() + 128 + 128 + 512;
    let witnesses = MAX_SNAPSHOTS * std::mem::size_of::<RestoredWitness>();
    let conservative =
        std::mem::size_of::<SnapshotCatalog>() + MAX_SNAPSHOTS * entry + witnesses + 8192;
    assert!(
        conservative <= CATALOG_BYTES,
        "bounded catalogue metadata {conservative} exceeds {CATALOG_BYTES}"
    );
}

#[test]
fn completed_background_validation_starts_the_next_candidate_but_yields_at_priority() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    let mut catalog =
        SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &interactive).unwrap();
    for entry in catalog.entries() {
        let index = run.by_sequence(entry.key.sequence).unwrap();
        loader.cache.insert(
            entry.key.clone(),
            Arc::new(run.recording().load(index).unwrap()),
        );
    }
    catalog.prioritize_revision(250);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while catalog.selected(250).is_none() {
        loader.poll();
        catalog
            .advance_with_followup(&run, &mut loader, &mut interactive, true)
            .unwrap();
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        catalog.validating.is_none(),
        "the requested revision must yield to foreground work"
    );
    catalog
        .advance_with_followup(&run, &mut loader, &mut interactive, true)
        .unwrap();
    while catalog.selected(0).is_none() {
        loader.poll();
        catalog
            .advance_with_followup(&run, &mut loader, &mut interactive, true)
            .unwrap();
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        catalog.validating.is_some(),
        "a completed background check should dispatch the next cached candidate without another paint"
    );
}

#[test]
fn validation_does_not_chain_while_the_foreground_frame_is_unready() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    let mut catalog =
        SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &interactive).unwrap();
    for entry in catalog.entries() {
        let index = run.by_sequence(entry.key.sequence).unwrap();
        loader.cache.insert(
            entry.key.clone(),
            Arc::new(run.recording().load(index).unwrap()),
        );
    }
    catalog.prioritize_revision(250);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while catalog.selected(250).is_none() {
        loader.poll();
        catalog
            .advance_with_followup(&run, &mut loader, &mut interactive, false)
            .unwrap();
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    catalog
        .advance_with_followup(&run, &mut loader, &mut interactive, false)
        .unwrap();
    while catalog.selected(0).is_none() {
        loader.poll();
        catalog
            .advance_with_followup(&run, &mut loader, &mut interactive, false)
            .unwrap();
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert!(
        catalog.validating.is_none(),
        "the next candidate must wait until foreground publication"
    );
}

#[test]
fn panel_retarget_reuses_only_exact_restored_snapshots_and_rechecks_capabilities() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    let mut catalog =
        SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &interactive).unwrap();
    assert!(
        !catalog.retarget(&run, "online", value_panel(true)),
        "pending validation cannot be retargeted"
    );
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    let positions = catalog.positions();
    assert_eq!(positions, vec![0, 100, 250]);
    let before = interactive.policy_metrics().restores;
    assert!(before >= 3);
    let other_dir = reuse_fixture::fixture();
    let other_run = RunIndex::open(other_dir.path().join("run"), ReadLimits::default()).unwrap();
    assert!(
        !catalog.retarget(&other_run, "online", value_panel(true)),
        "another recording cannot inherit validation"
    );
    assert!(
        !catalog.retarget(&run, "other", value_panel(true)),
        "another role cannot inherit validation"
    );
    assert!(
        !catalog.retarget(&run, "online", SnapshotPanel::Network { name: None }),
        "network-specific validation is distinct"
    );
    assert!(catalog.park_for_retarget());
    assert!(
        catalog.positions().is_empty(),
        "parked positions must not appear as the next panel"
    );
    assert!(catalog.retarget(&run, "online", value_panel(true)));
    retire_worker(&mut interactive);
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert_eq!(catalog.positions(), positions);
    assert_eq!(
        interactive.policy_metrics().restores,
        before,
        "matching snapshots must not be restored twice"
    );
    assert_eq!(
        interactive.policy_metrics().snapshot_verifications,
        3,
        "each reused snapshot must still be read and checksummed"
    );

    let missing = SnapshotPanel::Value {
        name: Some("absent".into()),
        twin: None,
        actor_required: false,
        output: ValueOutput::ExpectedReturn,
    };
    assert!(catalog.retarget(&run, "online", SnapshotPanel::Policy));
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert!(catalog.retarget(&run, "online", missing));
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert!(
        catalog.positions().is_empty(),
        "prior restoration must not bypass the new panel's critic checks"
    );

    assert!(catalog.retarget(&run, "online", SnapshotPanel::Policy));
    loader.invalidate();
    retire_worker(&mut interactive);
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert_eq!(catalog.positions(), positions);
    assert!(
        interactive.policy_metrics().restores > before,
        "an evicted source cannot reuse its proof"
    );
}

#[test]
fn panel_retarget_detects_a_changed_snapshot_before_reusing_restoration() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    let mut catalog =
        SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &interactive).unwrap();
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    let entry = catalog.selected(100).unwrap().clone();
    let index = run.by_sequence(entry.key.sequence).unwrap();
    let path = run.root().join(&run.recording().entries()[index].file);
    let stamp = SnapshotStamp::read(&run, index).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(stamp.modified))
        .unwrap();
    assert_eq!(
        SnapshotStamp::read(&run, index).unwrap(),
        stamp,
        "test must preserve file size and modification time"
    );
    assert!(catalog.retarget(&run, "online", value_panel(true)));
    retire_worker(&mut interactive);
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert!(
        catalog.selected(100).is_none(),
        "changed bytes must not inherit prior validation"
    );
    assert!(matches!(
        catalog
            .entries()
            .iter()
            .find(|item| item.key == entry.key)
            .unwrap()
            .status,
        SnapshotStatus::Rejected(_)
    ));
}

#[test]
fn panel_retarget_rejects_a_cached_snapshot_with_different_metadata() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    let mut catalog =
        SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &interactive).unwrap();
    let entry = catalog
        .entries()
        .iter()
        .find(|entry| entry.revision == 100)
        .unwrap()
        .clone();
    let index = run.by_sequence(entry.key.sequence).unwrap();
    let mut replacement = run.recording().load(index).unwrap();
    replacement.context.wall_elapsed_seconds = Some(1.);
    loader
        .cache
        .insert(entry.key.clone(), Arc::new(replacement));
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert!(catalog.selected(100).is_some());
    assert!(catalog.retarget(&run, "online", value_panel(true)));
    retire_worker(&mut interactive);
    finish_catalog(&mut catalog, &run, &mut loader, &mut interactive);
    assert!(
        catalog.selected(100).is_none(),
        "a cached allocation with different source metadata must not inherit restoration"
    );
    assert!(matches!(
        catalog
            .entries()
            .iter()
            .find(|item| item.key == entry.key)
            .unwrap()
            .status,
        SnapshotStatus::Rejected(_)
    ));
}

fn admitted(run: &RunIndex, panel: SnapshotPanel, name: &str) -> SnapshotCatalog {
    let interactive = Interactive::new(|| {}).unwrap();
    let mut catalog = SnapshotCatalog::new(run, "online", panel, &interactive).unwrap();
    // These fixtures contain real valid exports. Source construction is tested
    // here independently of the separately covered asynchronous validator.
    for entry in &mut catalog.entries {
        entry.name = name.into();
        entry.status = SnapshotStatus::Eligible;
    }
    catalog
}

fn value_panel(actor_required: bool) -> SnapshotPanel {
    SnapshotPanel::Value {
        name: Some("q1".into()),
        twin: None,
        actor_required,
        output: ValueOutput::ExpectedReturn,
    }
}

#[test]
fn policy_reuses_owned_metadata_and_rechecks_exact_source() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let catalog = admitted(&run, SnapshotPanel::Policy, "actor");
    let first = catalog.policy_source(&run, 0).unwrap().unwrap();
    let again = catalog
        .policy_source_reusing(&run, 0, Some(&first))
        .unwrap()
        .unwrap();
    assert_eq!(first.shared_metadata(), again.shared_metadata());
    let other = catalog
        .policy_source_reusing(&run, 100, Some(&first))
        .unwrap()
        .unwrap();
    assert_ne!(first.identity(), other.identity());
    let index = run.by_sequence(first.identity().key.sequence).unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    assert!(
        catalog
            .policy_source_reusing(&run, 0, Some(&first))
            .is_err()
    );
}

#[test]
fn value_reuses_metadata_but_revalidates_raw_replacement_and_retry() {
    let dir = reuse_fixture::fixture();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut catalog = admitted(&run, value_panel(true), "q1");
    let entry = catalog.selected(0).unwrap().clone();
    let index = run.by_sequence(entry.key.sequence).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    loader.cache.insert(
        entry.key.clone(),
        Arc::new(run.recording().load(index).unwrap()),
    );
    let first = catalog.value_source(&run, 0, &mut loader).unwrap().unwrap();
    let again = catalog
        .value_source_reusing(&run, 0, &mut loader, Some(&first))
        .unwrap()
        .unwrap();
    assert!(std::ptr::eq(first.manifest(), again.manifest()));
    assert_eq!(again.actor_name(), Some("actor"));

    // Same immutable address with a replacement raw allocation must be checked
    // again, not accepted using a stale validated-source witness.
    let mut replacement = run.recording().load(index).unwrap();
    if let rl::diagnostics::schema::Record::ModelSnapshot(model) = &mut replacement.record {
        model.configuration["observation_dim"] = 2.into();
    }
    loader
        .cache
        .insert(entry.key.clone(), Arc::new(replacement));
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&first))
            .is_err()
    );
    loader.cache.insert(
        entry.key.clone(),
        Arc::new(run.recording().load(index).unwrap()),
    );
    let rebuilt = catalog
        .value_source_reusing(&run, 0, &mut loader, Some(&first))
        .unwrap()
        .unwrap();
    assert!(!std::ptr::eq(first.manifest(), rebuilt.manifest()));
    catalog.retry_rejected();
    let retried = catalog
        .value_source_reusing(&run, 0, &mut loader, Some(&rebuilt))
        .unwrap()
        .unwrap();
    assert!(!std::ptr::eq(rebuilt.manifest(), retried.manifest()));
    std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&retried))
            .is_err()
    );
}

#[test]
fn value_reuse_preserves_conditioning_name_twin_and_role_checks() {
    let (dir, mut writer) = reuse_fixture::recorder();
    writer
        .append(&reuse_fixture::snapshot(0, "online", &["q1"]))
        .unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut catalog = admitted(&run, value_panel(false), "q1");
    let entry = catalog.selected(0).unwrap().clone();
    let index = run.by_sequence(entry.key.sequence).unwrap();
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    loader.cache.insert(
        entry.key.clone(),
        Arc::new(run.recording().load(index).unwrap()),
    );
    let first = catalog.value_source(&run, 0, &mut loader).unwrap().unwrap();
    assert_eq!(first.actor_name(), None);

    catalog.panel = value_panel(true);
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&first))
            .is_err()
    );
    catalog.panel = SnapshotPanel::Value {
        name: Some("q1".into()),
        twin: Some("missing-twin".into()),
        actor_required: false,
        output: ValueOutput::ExpectedReturn,
    };
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&first))
            .is_err()
    );
    catalog.panel = value_panel(false);
    catalog.entries[0].name = "missing-critic".into();
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&first))
            .is_err()
    );
    catalog.entries[0].name = "q1".into();
    catalog.entries[0].role = "target".into();
    assert!(
        catalog
            .value_source_reusing(&run, 0, &mut loader, Some(&first))
            .is_err()
    );
}

#[test]
fn value_reuse_cannot_substitute_same_revision_siblings() {
    let (dir, mut writer) = reuse_fixture::recorder();
    for _ in 0..2 {
        writer
            .append(&reuse_fixture::snapshot(0, "online", &["actor", "q1"]))
            .unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut catalog = admitted(&run, value_panel(true), "q1");
    let mut loader = Loader::new(crate::budgets::RAW, || {}).unwrap();
    for entry in &catalog.entries {
        let index = run.by_sequence(entry.key.sequence).unwrap();
        loader.cache.insert(
            entry.key.clone(),
            Arc::new(run.recording().load(index).unwrap()),
        );
    }
    let first = catalog.value_source(&run, 0, &mut loader).unwrap().unwrap();
    catalog.entries[0].status = SnapshotStatus::Rejected("test sibling selection".into());
    let other = catalog
        .value_source_reusing(&run, 0, &mut loader, Some(&first))
        .unwrap()
        .unwrap();
    assert_ne!(first.identity.key, other.identity.key);
    assert!(!std::ptr::eq(first.manifest(), other.manifest()));
}
