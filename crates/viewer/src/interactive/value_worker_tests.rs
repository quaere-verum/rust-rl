use super::*;
use crate::{
    policy::{prepared::PolicyPin, slice::SliceConfig},
    value::{
        prepared::{ValuePin, ValueRequest, ValueSource},
        slice::ValueSlice,
    },
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{schema::*, storage::*, value::*},
    nn::{critics::scalar::QValue, matrix, network::Mlp, policy::Deterministic, vector},
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

pub(crate) fn fixture() -> (tempfile::TempDir, RunIndex) {
    fixture_with_space(ActionSpace::Unbounded { dimensions: 1 })
}
fn fixture_with_space(space: ActionSpace) -> (tempfile::TempDir, RunIndex) {
    let dir = tempfile::tempdir().unwrap();
    let m = RunManifest::new("worker-value", "test", "test", 1, space.clone(), 0.95).unwrap();
    let mut w =
        Recorder::create(dir.path().join("run"), m.clone(), StorageConfig::default()).unwrap();
    for revision in 0..3 {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut net = Mlp::new(1, &[], 1, &mut rng).unwrap();
        net.layers[0].weight = net.layers[0].weight.clone().map(|_| matrix(vec![0.], 1, 1));
        net.layers[0].bias = net.layers[0]
            .bias
            .take()
            .map(|b| b.map(|_| vector(&[revision as f32])));
        let actor = Deterministic::from_network(net, space.clone()).unwrap();
        let mut q = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        q.network.layers[0].weight = q.network.layers[0]
            .weight
            .clone()
            .map(|_| matrix(vec![1., 2.], 2, 1));
        q.network.layers[0].bias = q.network.layers[0]
            .bias
            .take()
            .map(|b| b.map(|_| vector(&[0.])));
        let mut b = SnapshotBuilder::new(
            "source",
            RecordContext::new("worker-value", revision),
            1,
            space.clone(),
        )
        .unwrap();
        b.deterministic("actor", &actor).unwrap();
        b.scalar("q", &q).unwrap();
        let mut c = b.finish().unwrap();
        attach_declaration(
            &mut c,
            &ValueDeclaration {
                version: 1,
                run_id: m.run_id.clone(),
                revision,
                model_role: "online".into(),
                conventions: Some(ValueConventions {
                    dynamics: "test".into(),
                    observation: "test".into(),
                    action: "test".into(),
                    reward_units: "test".into(),
                    reward_convention: m.reward_convention.clone(),
                    discount: m.discount,
                    horizon: "test".into(),
                    terminal_treatment: "test".into(),
                    action_constraints: "test".into(),
                }),
                critics: BTreeMap::from([(
                    "q".into(),
                    CriticDeclaration {
                        kind: ValueCriticKind::ScalarQ,
                        learned_baseline: None,
                        actor: Some(ContinuationIdentity {
                            identity: "actor".into(),
                            stochastic: false,
                        }),
                        continuation: Some(ContinuationIdentity {
                            identity: format!("actor-{revision}"),
                            stochastic: false,
                        }),
                        twin: None,
                        utility: None,
                    },
                )]),
            },
        )
        .unwrap();
        w.append(&c).unwrap();
    }
    w.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    (dir, run)
}

#[test]
fn cached_bundle_does_not_reuse_a_changed_encoded_source_at_the_same_key() {
    let (_dir, run) = fixture();
    let key = run.key(0).unwrap();
    let first = run.recording().source(0).unwrap();
    let replacement = run.recording().source(1).unwrap();
    assert_ne!(first.encoded_identity(), replacement.encoded_identity());
    let accounting = Arc::new(Accounting::default());
    let mut bundles = Vec::new();
    #[cfg(feature = "screenshots")]
    let observed = ModelOwners::default();
    cached_bundle(
        &mut bundles,
        &key,
        &first,
        Some(&accounting),
        None,
        #[cfg(feature = "screenshots")]
        &observed,
    )
    .unwrap();
    cached_bundle(
        &mut bundles,
        &key,
        &replacement,
        Some(&accounting),
        None,
        #[cfg(feature = "screenshots")]
        &observed,
    )
    .unwrap();
    assert_eq!(accounting.snapshot().restores, 2);
    assert_eq!(bundles.len(), 1);
    assert_eq!(bundles[0].encoded_identity, replacement.encoded_identity());
}

#[test]
fn catalogue_validation_rechecks_bytes_even_when_the_model_is_cached() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let wait = |interactive: &mut Interactive| {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(result) = interactive.poll_snapshot_validation().unwrap() {
                break result;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    interactive.validate_snapshot(&run, 0).unwrap();
    assert_eq!(wait(&mut interactive).unwrap(), run.key(0).unwrap());
    let path = run.root().join(&run.recording().entries()[0].file);
    let stamp = std::fs::metadata(&path).unwrap().modified().unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, bytes).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(stamp))
        .unwrap();
    interactive.validate_snapshot(&run, 0).unwrap();
    assert!(
        wait(&mut interactive).is_err(),
        "a cached model must not hide corrupted recorded bytes"
    );
}
fn request(run: &RunIndex, index: usize) -> ValueRequest {
    let s = ValueSource::model(run, index, "q", &run.recording().load(index).unwrap()).unwrap();
    let mut slice = ValueSlice::new(s.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
    slice.resolution = [256, 64];
    ValueRequest::new(
        s,
        ValueOutput::ExpectedReturn,
        slice,
        ValuePin::State(Observation::new(vec![0.25])),
    )
    .unwrap()
}
fn wait(i: &mut Interactive) {
    let end = Instant::now() + Duration::from_secs(10);
    while i.value_pending() {
        i.poll_value().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(i.value_error().is_none(), "{:?}", i.value_error());
}
fn retired(i: &mut Interactive) {
    let end = Instant::now() + Duration::from_secs(10);
    i.retire_session();
    while i.retirement_status().0 != i.retirement_status().1 {
        i.housekeeping();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(i.value_metrics().prepared_bytes, 0);
}

#[test]
fn state_map_prepares_an_action_curve_at_the_exact_selected_state() {
    let dir = tempfile::tempdir().unwrap();
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let manifest = RunManifest::new("curve", "test", "test", 2, space.clone(), 0.95).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let actor = Deterministic::new(2, space.clone(), &[], &mut rng).unwrap();
    let mut q = QValue::new(2, 1, 1, &[], &mut rng).unwrap();
    q.network.layers[0].weight = q.network.layers[0]
        .weight
        .clone()
        .map(|_| matrix(vec![1., 2., 3.], 3, 1));
    q.network.layers[0].bias = q.network.layers[0]
        .bias
        .take()
        .map(|b| b.map(|_| vector(&[0.])));
    let mut builder =
        SnapshotBuilder::new("source", RecordContext::new("curve", 0), 2, space).unwrap();
    builder.deterministic("actor", &actor).unwrap();
    builder.scalar("q", &q).unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let source = ValueSource::model(&run, 0, "q", &run.recording().load(0).unwrap()).unwrap();
    let observation = Observation::new(vec![0.25, 0.5]);
    let mut slice = ValueSlice::new(source.semantics(), &observation, &[]).unwrap();
    slice.resolution = [8, 8];
    let request = ValueRequest::new(
        source,
        ValueOutput::ExpectedReturn,
        slice,
        ValuePin::Input {
            observation,
            action: Some(Action::Continuous(vec![0.1])),
        },
    )
    .unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_value(request).unwrap();
    wait(&mut interactive);
    let frame = interactive.value_frame().unwrap();
    let curve = frame
        .selected
        .as_ref()
        .unwrap()
        .action_curve
        .as_ref()
        .unwrap();
    assert_eq!(curve.component, 0);
    assert_eq!(curve.points.first().unwrap()[0], -1.);
    assert_eq!(curve.points.last().unwrap()[0], 1.);
    assert!((curve.points.first().unwrap()[1] - (-1.75)).abs() < 1e-5);
    assert!((curve.points.last().unwrap()[1] - 4.25).abs() < 1e-5);
    retired(&mut interactive);
}

#[test]
fn retained_replacement_also_leaves_snapshot_validation_headroom() {
    let interactive = Interactive::new(|| {}).unwrap();
    let _held = interactive.reserve_policy_view(3 * 1024 * 1024).unwrap();
    // A frame counted as eventually retired can remain in the prepared cache.
    // Its coexistence with the staged replacement must not starve validation.
    assert!(!interactive.grid_fits(768 * 1024, 768 * 1024, 0, 512 * 1024));
}

#[test]
fn single_value_frame_admission_keeps_validation_headroom_and_counts_external_owners() {
    let interactive = Interactive::new(|| {}).unwrap();
    let view = interactive.reserve_policy_view(2 * 1024 * 1024).unwrap();
    let frame = 5 * 256 * 1024;
    assert!(interactive.single_value_grid_fits(frame, frame, 0));
    assert!(!interactive.grid_fits(frame, frame, frame, 0));
    let external = interactive.reserve_policy_view(512 * 1024).unwrap();
    assert!(!interactive.single_value_grid_fits(frame, frame, 0));
    drop(external);
    drop(view);
}

#[test]
fn identical_calibration_sources_share_real_owners_without_merging_other_revisions() {
    let (_dir, run) = fixture();
    let mut r = request(&run, 0);
    let independently_loaded = request(&run, 0).current;
    assert!(r.current.same(&independently_loaded));
    assert!(!Arc::ptr_eq(
        &r.current.manifest,
        &independently_loaded.manifest
    ));
    r.reference_value = Some(request(&run, 1).current);
    let independent_reference = request(&run, 1).current;
    assert!(
        r.reference_value
            .as_ref()
            .unwrap()
            .same(&independent_reference)
    );
    assert!(!Arc::ptr_eq(
        &r.reference_value.as_ref().unwrap().manifest,
        &independent_reference.manifest
    ));
    r.calibration = Some(value_prepared::CalibrationRequest {
        serial: 7,
        source: independently_loaded,
        reference: Some(independent_reference),
    });
    let before = value_prepared::RequestOwner::standalone_metadata_charge(&r);
    r.coalesce_calibration_sources();
    assert!(Arc::ptr_eq(
        &r.current.manifest,
        &r.calibration.as_ref().unwrap().source.manifest
    ));
    let (
        crate::value::Data::Model(current_address),
        crate::value::Data::Model(calibration_address),
    ) = (
        &r.current.data,
        &r.calibration.as_ref().unwrap().source.data,
    )
    else {
        panic!("historical model sources expected")
    };
    assert!(Arc::ptr_eq(current_address, calibration_address));
    assert!(Arc::ptr_eq(
        &r.reference_value.as_ref().unwrap().manifest,
        &r.calibration
            .as_ref()
            .unwrap()
            .reference
            .as_ref()
            .unwrap()
            .manifest
    ));
    let after = value_prepared::RequestOwner::standalone_metadata_charge(&r);
    assert_eq!(
        before - after,
        r.current.metadata_bytes
            - r.calibration
                .as_ref()
                .unwrap()
                .source
                .private_metadata_bytes()
            + r.reference_value.as_ref().unwrap().metadata_bytes
            - r.calibration
                .as_ref()
                .unwrap()
                .reference
                .as_ref()
                .unwrap()
                .private_metadata_bytes()
    );
    let accounting = Arc::new(prepared::Accounting::default());
    let owner = value_prepared::RequestOwner::new(r.clone(), &accounting).unwrap();
    assert_eq!(accounting.snapshot().prepared_bytes, after);
    drop(owner);
    assert_eq!(accounting.snapshot().prepared_bytes, 0);
    r.calibration.as_mut().unwrap().source = request(&run, 2).current;
    r.coalesce_calibration_sources();
    assert!(!Arc::ptr_eq(
        &r.current.manifest,
        &r.calibration.as_ref().unwrap().source.manifest
    ));
    assert_eq!(
        r.calibration.as_ref().unwrap().source.identity().revision,
        2
    );
}

#[test]
fn automatic_value_resolution_can_admit_one_non_empirical_frame_with_headroom() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    r.current.metadata_bytes += 700 * 1024;
    let slice = Arc::make_mut(&mut r.slice);
    slice.mode = crate::policy::slice::SliceMode::TwoDimensional;
    slice.y = Some(crate::value::slice::ValueCoordinate::Action(0));
    slice.resolution = [32, 32];
    let frame = crate::value::prepared::publication_charge(&r).unwrap();
    assert!(frame > 512 * 1024);
    let reservation = crate::budgets::POLICY_PREPARED - 512 * 1024 - frame - 1024;
    let view = interactive.reserve_policy_view(reservation).unwrap();
    assert!(!interactive.grid_fits(frame, frame, frame, 0));
    assert!(interactive.single_value_grid_fits(frame, frame, 0));
    assert!(r.empirical.is_none());
    interactive.automatic_value_resolution(&mut r, 0).unwrap();
    assert_eq!(r.slice.resolution, [32, 32]);
    drop(view);
}

#[test]
fn value_admission_evicts_the_suspended_policy_when_headroom_is_needed() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive
        .submit_policy(policy_reference_request(&run))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.policy_frame().is_some());
    interactive.suspend_policy_for_value();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(interactive.policy_recent.len(), 1);
    let cache_bytes = interactive.policy_recent[0]
        .frame
        .retirement_bytes(Some(&interactive.policy_recent[0].owner));
    assert!(cache_bytes > 0);
    let mut value = request(&run, 0);
    let complete = value_prepared::publication_charge(&value).unwrap();
    let incremental = complete - value.metadata_bytes()
        + value_prepared::RequestOwner::metadata_charge(
            &value,
            &interactive.worker.policy_accounting,
        )
        .unwrap();
    let standalone = complete - value.metadata_bytes()
        + value_prepared::RequestOwner::standalone_metadata_charge(&value);
    let current = interactive.policy_metrics().prepared_bytes;
    let target = crate::budgets::POLICY_PREPARED - 512 * 1024 - incremental.max(standalone)
        + cache_bytes / 2;
    let filler = interactive
        .reserve_policy_view(
            target
                .checked_sub(current)
                .expect("fixture must leave room for pressure"),
        )
        .unwrap();
    assert!(!interactive.single_value_grid_fits(incremental, standalone, 0));
    interactive
        .automatic_value_resolution(&mut value, 0)
        .unwrap();
    assert!(
        interactive.policy_recent.is_empty(),
        "foreground Value admission must evict speculative Policy reuse"
    );
    assert!(interactive.single_value_grid_fits(incremental, standalone, 0));
    drop(filler);
}

#[test]
fn compatible_policy_value_roundtrip_reuses_both_prepared_frames() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let policy = policy_reference_request(&run);
    let value = request(&run, 0);
    let wait_policy = |interactive: &mut Interactive| {
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            interactive.policy_error().is_none(),
            "{:?}",
            interactive.policy_error()
        );
        assert!(interactive.policy_frame().is_some());
    };
    interactive.submit_policy(policy.clone()).unwrap();
    wait_policy(&mut interactive);
    interactive.suspend_policy_for_value();
    interactive.submit_value(value.clone()).unwrap();
    wait(&mut interactive);
    assert!(interactive.value_frame().is_some());
    interactive.leave_value_for_policy();
    interactive.submit_policy(policy.clone()).unwrap();
    wait_policy(&mut interactive);
    let mut same_policy = policy.clone();
    interactive
        .automatic_policy_resolution(&mut same_policy)
        .unwrap();
    assert_eq!(
        interactive.value_recent.len(),
        1,
        "an unchanged displayed Policy must keep its compatible Value frame"
    );
    interactive.suspend_policy_for_value();
    let before = interactive.value_metrics();
    let mut returned_value = value;
    let prior_resolution = returned_value.slice.resolution;
    Arc::make_mut(&mut returned_value.slice).resolution = [64, 1];
    interactive
        .automatic_value_resolution(&mut returned_value, 0)
        .unwrap();
    assert_eq!(
        returned_value.slice.resolution, prior_resolution,
        "automatic reopening should retain the exact compatible cached grid"
    );
    interactive.submit_value(returned_value).unwrap();
    wait(&mut interactive);
    let after = interactive.value_metrics();
    assert!(interactive.value_frame().is_some());
    assert_eq!(
        (after.restores, after.grid_batches, after.point_batches),
        (before.restores, before.grid_batches, before.point_batches),
        "a compatible Value revisit should reuse its complete charged frame"
    );
    assert!(after.prepared_peak <= crate::budgets::POLICY_PREPARED);
    retired(&mut interactive);
}

#[test]
fn cross_workspace_value_cache_yields_to_changed_policy_and_validation_headroom() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let policy = policy_reference_request(&run);
    let wait_policy = |interactive: &mut Interactive| {
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(interactive.policy_error().is_none());
    };
    interactive.submit_policy(policy.clone()).unwrap();
    wait_policy(&mut interactive);
    interactive.suspend_policy_for_value();
    interactive.submit_value(request(&run, 0)).unwrap();
    wait(&mut interactive);
    interactive.leave_value_for_policy();
    assert_eq!(interactive.value_recent.len(), 1);
    interactive.submit_policy(policy.clone()).unwrap();
    wait_policy(&mut interactive);
    let mut changed = policy;
    changed.pin = PolicyPin::Synthetic(Observation::new(vec![0.75]));
    interactive
        .automatic_policy_resolution(&mut changed)
        .unwrap();
    assert!(
        interactive.value_recent.is_empty(),
        "changed foreground input must retire the optional Value cache"
    );
    interactive.submit_policy(changed).unwrap();
    wait_policy(&mut interactive);
    assert_eq!(
        interactive
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .input
            .dense_values(),
        &[0.75]
    );
    let validation = interactive.reserve_policy_view(512 * 1024).unwrap();
    drop(validation);
    retired(&mut interactive);
}

#[test]
fn cross_workspace_value_cache_is_not_retained_without_validation_headroom() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive
        .submit_policy(policy_reference_request(&run))
        .unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    interactive.suspend_policy_for_value();
    interactive.submit_value(request(&run, 0)).unwrap();
    wait(&mut interactive);
    let used = interactive.policy_metrics().prepared_bytes;
    let target = crate::budgets::POLICY_PREPARED - 512 * 1024 + 1;
    let filler = interactive
        .reserve_policy_view(target.checked_sub(used).unwrap())
        .unwrap();
    interactive.leave_value_for_policy();
    assert!(
        interactive.value_recent.is_empty(),
        "optional Value frame must yield to validation headroom"
    );
    drop(filler);
    let validation = interactive.reserve_policy_view(512 * 1024).unwrap();
    drop(validation);
    retired(&mut interactive);
}

#[test]
fn cross_workspace_value_revisit_rejects_a_missing_source() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let policy = policy_reference_request(&run);
    let value = request(&run, 0);
    let wait_policy = |interactive: &mut Interactive| {
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(interactive.policy_error().is_none());
    };
    interactive.submit_policy(policy.clone()).unwrap();
    wait_policy(&mut interactive);
    interactive.suspend_policy_for_value();
    interactive.submit_value(value.clone()).unwrap();
    wait(&mut interactive);
    interactive.leave_value_for_policy();
    interactive.submit_policy(policy).unwrap();
    wait_policy(&mut interactive);
    interactive.suspend_policy_for_value();
    assert_eq!(interactive.value_recent.len(), 1);
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    assert!(
        interactive.submit_value(value).is_err(),
        "a cached frame must recheck its source"
    );
    assert!(
        interactive.value_frame().is_none(),
        "the retired source must not be published"
    );
    retired(&mut interactive);
}

#[test]
fn empirical_point_admission_keeps_or_evicts_policy_using_verified_sample_count() {
    use crate::value::empirical::{EmpiricalSource, tests::fixture_samples};
    for samples in [512, 7000] {
        let (_policy_dir, policy_run) = fixture();
        let (_value_dir, value_run, current) = fixture_samples(samples);
        let mut interactive = Interactive::new(|| {}).unwrap();
        let mut policy = policy_reference_request(&policy_run);
        policy.reference = None;
        // Exercise eviction with an actually owned, charged allocation.
        Arc::make_mut(&mut policy.slice).fixed.reserve(60_000);
        interactive.submit_policy(policy).unwrap();
        let deadline = Instant::now() + Duration::from_secs(20);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        interactive.suspend_policy_for_value();
        while interactive.retirement_status().0 != interactive.retirement_status().1 {
            interactive.housekeeping();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(interactive.policy_recent.len(), 1);
        let input = Observation::dense(vec![0., 0., 0., 0., 0.375], Some(vec![true, true]));
        let mut slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
        slice.mode = crate::policy::slice::SliceMode::OneDimensional;
        slice.x = crate::value::slice::ValueCoordinate::Observation(0);
        slice.y = None;
        slice.resolution = [3, 1];
        let mut value = ValueRequest::new(
            current,
            ValueOutput::ExpectedReturn,
            slice,
            ValuePin::State(input),
        )
        .unwrap();
        value.empirical = Some(Arc::new(
            EmpiricalSource::discover(&value_run, &value.current)
                .unwrap()
                .unwrap(),
        ));
        value.reference_empirical = true;
        let deferred = value_prepared::deferred_publication_charge(&value).unwrap();
        let target = crate::budgets::POLICY_PREPARED - 512 * 1024 - deferred - 100 * 1024;
        let used = interactive.policy_metrics().prepared_bytes;
        let filler = interactive
            .reserve_policy_view(target.checked_sub(used).unwrap())
            .unwrap();
        interactive.submit_value(value).unwrap();
        wait(&mut interactive);
        let point = interactive
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap();
        let empirical = point.empirical.as_ref().unwrap();
        assert_eq!(empirical.condition.summary.count, samples);
        assert_eq!(empirical.cdf.len(), samples * 2);
        assert_eq!(interactive.policy_recent.is_empty(), samples == 7000);
        assert!(interactive.policy_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
        let validation = interactive.reserve_policy_view(512 * 1024).unwrap();
        drop((validation, filler));
        retired(&mut interactive);
    }
}

#[test]
fn suspended_policy_does_not_reduce_automatic_value_resolution() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive
        .submit_policy(policy_reference_request(&run))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    interactive.suspend_policy_for_value();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let mut value = request(&run, 0);
    value.current.metadata_bytes += 700 * 1024;
    let slice = Arc::make_mut(&mut value.slice);
    slice.mode = crate::policy::slice::SliceMode::TwoDimensional;
    slice.y = Some(crate::value::slice::ValueCoordinate::Action(0));
    let mut baseline = value.clone();
    interactive
        .automatic_value_resolution(&mut value, 0)
        .unwrap();
    interactive.policy_recent.clear();
    interactive
        .automatic_value_resolution_inner(&mut baseline, 0)
        .unwrap();
    assert_eq!(value.slice.resolution, baseline.slice.resolution);
}

#[test]
fn future_catalogue_lease_cannot_consume_validation_headroom() {
    let interactive = Interactive::new(|| {}).unwrap();
    let _view = interactive.reserve_policy_view(1_904_768).unwrap();
    let frame = 1_641_487;
    assert!(interactive.single_value_grid_fits(frame, frame, 0));
    let with_catalog = frame + crate::snapshot_catalog::CATALOG_BYTES;
    assert!(!interactive.single_value_grid_fits(with_catalog, with_catalog, 0));
}

#[test]
fn releasing_a_value_frame_keeps_external_owners_charged() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_value(request(&run, 0)).unwrap();
    wait(&mut interactive);
    let held = interactive.value_frame().unwrap().clone();
    interactive.release_value_frames();
    assert!(interactive.value_frame().is_none());
    assert!(interactive.value_metrics().prepared_bytes > 0);
    drop(held);
    interactive.housekeeping();
    assert_eq!(interactive.value_metrics().prepared_bytes, 0);
    interactive.submit_value(request(&run, 2)).unwrap();
    wait(&mut interactive);
    assert_eq!(
        interactive
            .value_frame()
            .unwrap()
            .grid
            .current()
            .identity()
            .revision,
        2
    );
    retired(&mut interactive);
}
#[test]
fn automatic_value_resolution_preserves_eligible_cached_revisits() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let view = interactive.reserve_policy_view(1740 * 1024).unwrap();
    let make = |index| {
        let mut r = request(&run, index);
        let slice = Arc::make_mut(&mut r.slice);
        slice.mode = crate::policy::slice::SliceMode::TwoDimensional;
        slice.y = Some(crate::value::slice::ValueCoordinate::Action(0));
        slice.resolution = [64, 64];
        r
    };
    let mut resolution = [0, 0];
    for index in [0, 2] {
        let mut r = make(index);
        interactive.automatic_value_resolution(&mut r, 0).unwrap();
        resolution = r.slice.resolution;
        interactive.submit_value(r).unwrap();
        wait(&mut interactive);
    }
    let before = interactive.value_metrics();
    let mut r = make(0);
    Arc::make_mut(&mut r.slice).resolution = resolution;
    interactive.automatic_value_resolution(&mut r, 0).unwrap();
    interactive.submit_value(r).unwrap();
    wait(&mut interactive);
    assert_eq!(
        interactive
            .value_frame()
            .unwrap()
            .grid
            .current()
            .identity()
            .revision,
        0
    );
    assert_eq!(
        interactive.value_metrics().grid_batches,
        before.grid_batches
    );
    assert_eq!(interactive.value_metrics().restores, before.restores);
    drop(view);
    retired(&mut interactive);
}

#[test]
fn cached_value_revisit_preserves_the_exact_selected_input() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    for (revision, pin) in [(0, 0.25), (0, 0.75), (2, 0.25), (0, 0.25)] {
        let mut r = request(&run, revision);
        r.pin = ValuePin::State(Observation::new(vec![pin]));
        interactive.submit_value(r).unwrap();
        wait(&mut interactive);
        assert_eq!(
            interactive
                .value_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .input
                .dense_values(),
            &[pin]
        );
    }
    retired(&mut interactive);
}

#[test]
fn suspended_value_frame_rechecks_its_source_before_reuse() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let original = request(&run, 0);
    interactive.submit_value(original.clone()).unwrap();
    wait(&mut interactive);
    interactive.suspend_value();
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    assert!(interactive.submit_value(original).is_err());
    assert!(interactive.value_frame().is_none());
    retired(&mut interactive);
}

#[test]
fn value_revisits_a_prepared_checkpoint_without_restoring_or_recomputing() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    for index in [0, 2] {
        interactive.submit_value(request(&run, index)).unwrap();
        wait(&mut interactive);
    }
    let before = interactive.value_metrics();
    interactive.submit_value(request(&run, 0)).unwrap();
    wait(&mut interactive);
    let after = interactive.value_metrics();
    assert_eq!(
        interactive
            .value_frame()
            .unwrap()
            .grid
            .current()
            .identity()
            .revision,
        0
    );
    assert_eq!(
        interactive
            .value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .value,
        0.25
    );
    assert_eq!(after.restores, before.restores);
    assert_eq!(after.grid_batches, before.grid_batches);
    retired(&mut interactive);
}

#[test]
fn value_pending_grid_uses_latest_pin_then_source_generation_rejects_stale_answers() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    i.submit_value(r.clone()).unwrap();
    r.pin = ValuePin::State(Observation::new(vec![0.625]));
    i.submit_value(r).unwrap();
    wait(&mut i);
    assert_eq!(
        i.value_frame().unwrap().selected.as_ref().unwrap().value,
        0.625
    );
    assert_eq!(i.value_metrics().grid_batches, 1);
    for n in 0..12 {
        i.submit_value(request(&run, n % 3)).unwrap();
    }
    wait(&mut i);
    assert_eq!(
        i.value_frame().unwrap().grid.current().identity().revision,
        2
    );
    assert_eq!(
        i.value_frame().unwrap().selected.as_ref().unwrap().value,
        4.25
    );
    retired(&mut i);
}
#[test]
fn three_sources_are_grouped_and_policy_value_share_one_owner_and_two_models() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    r.reference_value = Some(request(&run, 1).current);
    r.reference_policy = Some(prepared::PolicySource::model(&run, 2, Some("actor")).unwrap());
    i.submit_value(r.clone()).unwrap();
    wait(&mut i);
    let frame = i.value_frame().unwrap();
    assert_eq!(frame.grid.reference_policy_overlay.len(), 256);
    assert_eq!(
        frame.grid.reference_policy_overlay[0],
        Some(Action::Continuous(vec![2.]))
    );
    assert_eq!(frame.selected.as_ref().unwrap().reference_value, Some(0.25));
    assert_eq!(
        i.value_metrics().restores,
        5,
        "three grid source groups then two selected source groups"
    );
    r.pin = ValuePin::State(Observation::new(vec![0.75]));
    i.submit_value(r).unwrap();
    wait(&mut i);
    assert_eq!(i.value_metrics().restores, 5);
    let source = prepared::PolicySource::model(&run, 0, Some("actor")).unwrap();
    let mut slice = SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
    slice.resolution = [2, 2];
    let r = PolicyRequest::new(
        source,
        None,
        0,
        slice,
        PolicyPin::Synthetic(Observation::new(vec![0.])),
        None,
    )
    .unwrap();
    i.submit_policy(r).unwrap();
    assert!(i.value_frame().is_none());
    let end = Instant::now() + Duration::from_secs(10);
    while i.policy_pending() {
        i.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(i.policy_error().is_none());
    i.submit_value(request(&run, 1)).unwrap();
    assert!(i.policy_frame().is_none());
    wait(&mut i);
    #[cfg(feature = "screenshots")]
    assert!(i.owner_snapshot().models <= 2);
    retired(&mut i);
    #[cfg(feature = "screenshots")]
    assert_eq!(i.owner_snapshot().models, 0);
}

fn await_value_completion(i: &mut Interactive) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while i.worker.answer.is_none() {
        i.worker.poll().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}
fn optional_retention(missing_policy: bool, phase: usize) {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    r.reference_value = Some(request(&run, 1).current);
    r.reference_policy = Some(prepared::PolicySource::model(&run, 2, Some("actor")).unwrap());
    let missing = if missing_policy { 2 } else { 1 };
    let path = run.root().join(&run.recording().entries()[missing].file);
    let mut retained = None;
    if phase == 0 {
        std::fs::remove_file(&path).unwrap();
    }
    i.submit_value(r).unwrap();
    if phase > 0 {
        await_value_completion(&mut i);
        if phase == 2 {
            i.poll_value().unwrap(); // Publish grid and submit selected point.
            retained = Some(i.value_staged.as_ref().unwrap().grid.clone());
            await_value_completion(&mut i);
        }
        std::fs::remove_file(&path).unwrap();
        i.poll_value().unwrap();
    }
    wait(&mut i);
    let frame = i.value_frame().unwrap();
    assert_eq!(frame.selected.as_ref().unwrap().value, 0.25);
    if missing_policy {
        assert!(frame.grid.reference_policy_error.is_some());
        assert!(frame.grid.reference_policy_overlay.is_empty());
        assert!(frame.grid.reference_error.is_none());
        assert_eq!(frame.selected.as_ref().unwrap().reference_value, Some(0.25));
        assert!(frame.grid.reference_values.iter().all(Option::is_some));
    } else {
        assert!(frame.grid.reference_error.is_some());
        assert!(frame.grid.reference_values.is_empty());
        assert!(frame.grid.reference_actions.is_empty());
        assert!(frame.grid.reference_series.is_none());
        assert!(frame.selected.as_ref().unwrap().reference_error.is_some());
        assert!(
            frame
                .selected
                .as_ref()
                .unwrap()
                .reference_prediction
                .is_none()
        );
        assert!(frame.selected.as_ref().unwrap().reference_value.is_none());
        assert!(frame.grid.reference_policy_error.is_none());
        assert!(
            frame
                .grid
                .reference_policy_overlay
                .iter()
                .all(|a| *a == Some(Action::Continuous(vec![2.])))
        );
    }
    if let Some(old) = &retained {
        assert!(!Arc::ptr_eq(old, &frame.grid));
        assert!(old.reference_error.is_none());
        assert!(old.reference_policy_error.is_none());
        assert!(!old.reference_values.is_empty());
        assert!(!old.reference_policy_overlay.is_empty());
        let charged = i.value_metrics().prepared_bytes;
        drop(retained.take());
        assert!(
            i.value_metrics().prepared_bytes < charged,
            "retained immutable grid clone has its own charged storage"
        );
    }
    assert!(i.value_metrics().prepared_peak <= 4 * 1024 * 1024);
    retired(&mut i);
}
#[test]
fn value_optional_policy_retention_before_preparation() {
    optional_retention(true, 0);
}
#[test]
fn value_optional_value_retention_before_preparation() {
    optional_retention(false, 0);
}
#[test]
fn value_optional_policy_retention_at_grid_completion() {
    optional_retention(true, 1);
}
#[test]
fn value_optional_value_retention_at_grid_completion() {
    optional_retention(false, 1);
}
#[test]
fn value_optional_policy_retention_at_point_completion_with_retained_grid() {
    optional_retention(true, 2);
}
#[test]
fn value_optional_value_retention_at_point_completion_with_retained_grid() {
    optional_retention(false, 2);
}
#[test]
fn value_retention_copy_is_admitted_before_allocation_and_budget_failure_retires_ui() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    r.reference_value = Some(request(&run, 1).current);
    i.submit_value(r).unwrap();
    await_value_completion(&mut i);
    i.poll_value().unwrap();
    let retained = i.value_staged.as_ref().unwrap().grid.clone();
    await_value_completion(&mut i);
    let reserve = crate::budgets::POLICY_PREPARED - i.value_metrics().prepared_bytes - 1;
    let view = i.reserve_policy_view(reserve).unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    assert!(i.poll_value().is_err());
    assert!(i.value_error().is_some());
    assert!(i.value_frame().is_none());
    assert!(retained.reference_error.is_none());
    assert!(!retained.reference_values.is_empty());
    assert!(i.value_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    drop(view);
    drop(retained);
    retired(&mut i);
}

#[test]
fn snapshot_frame_is_atomic_and_held_until_grid_and_point_complete() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    i.submit_value(request(&run, 0)).unwrap();
    await_value_completion(&mut i);
    i.poll_value().unwrap();
    assert!(i.value_frame().is_none(), "grid alone must not publish");
    wait(&mut i);
    i.submit_value(request(&run, 1)).unwrap();
    assert_eq!(
        i.value_frame().unwrap().grid.current().identity().revision,
        0
    );
    await_value_completion(&mut i);
    i.poll_value().unwrap();
    assert_eq!(
        i.value_frame().unwrap().selected.as_ref().unwrap().value,
        0.25
    );
    wait(&mut i);
    assert_eq!(
        i.value_frame().unwrap().grid.current().identity().revision,
        1
    );
    assert_eq!(
        i.value_frame().unwrap().selected.as_ref().unwrap().value,
        2.25
    );
    let grids = i.value_metrics().grid_batches;
    let mut r = request(&run, 1);
    r.requested_revision = 100;
    i.submit_value(r).unwrap();
    wait(&mut i);
    assert_eq!(
        i.value_metrics().grid_batches,
        grids,
        "immutable model reused"
    );
    assert_eq!(i.value_frame().unwrap().requested_revision, 100);
    retired(&mut i);
}

#[test]
fn policy_prefetch_keeps_the_exact_reference_source() {
    let (_dir, run) = fixture();
    let make = |index| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let reference = prepared::PolicySource::model(&run, 2, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [16, 16];
        PolicyRequest::new(
            source,
            Some(reference),
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_policy(make(0)).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.prefetch_policy(make(1)).unwrap());
    while interactive.warming {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    let before = interactive.policy_metrics();
    interactive.submit_policy(make(1)).unwrap();
    assert!(!interactive.policy_pending());
    let frame = interactive.policy_frame().unwrap();
    assert_eq!(frame.grid.current.revision, 1);
    assert_eq!(frame.grid.reference.as_ref().unwrap().revision, 2);
    assert_eq!(frame.selected.as_ref().unwrap().reference_value, Some(2.));
    assert_eq!(
        interactive.policy_metrics().grid_batches,
        before.grid_batches
    );
    assert_eq!(interactive.policy_metrics().restores, before.restores);
    retired(&mut interactive);
}

#[test]
fn cached_policy_revisit_rejects_a_retired_recording_source() {
    let (_dir, run) = fixture();
    let make = |index| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [16, 16];
        PolicyRequest::new(
            source,
            None,
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    for index in [0, 2] {
        interactive.submit_policy(make(index)).unwrap();
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let old_request = make(0);
    std::fs::remove_file(run.root().join(&run.recording().entries()[0].file)).unwrap();
    assert!(interactive.submit_policy(old_request).is_err());
    assert_eq!(interactive.policy_frame().unwrap().grid.current.revision, 2);
    retired(&mut interactive);
}

#[test]
fn foreground_checkpoint_supersedes_policy_prefetch_without_stale_publication() {
    let (_dir, run) = fixture();
    let make = |index| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [128, 128];
        PolicyRequest::new(
            source,
            None,
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_policy(make(0)).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.prefetch_policy(make(1)).unwrap());
    interactive.submit_policy(make(2)).unwrap();
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        interactive.policy_error().is_none(),
        "{:?}",
        interactive.policy_error()
    );
    assert_eq!(interactive.policy_frame().unwrap().grid.current.revision, 2);
    assert_eq!(
        interactive
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .value,
        2.
    );
    assert!(interactive.policy_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    retired(&mut interactive);
}

#[test]
fn policy_prefetch_prepares_next_checkpoint_without_changing_current_publication() {
    let (_dir, run) = fixture();
    let make = |index| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [16, 16];
        PolicyRequest::new(
            source,
            None,
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive.submit_policy(make(0)).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.prefetch_policy(make(1)).unwrap());
    while interactive.warming {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(interactive.policy_frame().unwrap().grid.current.revision, 0);
    let before = interactive.policy_metrics();
    interactive.submit_policy(make(1)).unwrap();
    assert!(!interactive.policy_pending());
    assert_eq!(interactive.policy_frame().unwrap().grid.current.revision, 1);
    assert_eq!(
        interactive.policy_metrics().grid_batches,
        before.grid_batches
    );
    assert_eq!(interactive.policy_metrics().restores, before.restores);
    retired(&mut interactive);
}

#[test]
fn cached_policy_revisit_never_relabels_a_different_selected_input() {
    let (_dir, run) = fixture();
    let make = |index, pin| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [16, 16];
        PolicyRequest::new(
            source,
            None,
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![pin])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    for (revision, pin) in [(0, 0.25), (0, 0.75), (2, 0.25), (0, 0.25)] {
        interactive.submit_policy(make(revision, pin)).unwrap();
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(interactive.policy_error().is_none());
        assert_eq!(
            interactive
                .policy_frame()
                .unwrap()
                .selected
                .as_ref()
                .unwrap()
                .input
                .dense_values(),
            &[pin]
        );
    }
    retired(&mut interactive);
}

#[test]
fn policy_revisits_a_prepared_checkpoint_without_restoring_or_recomputing() {
    let (_dir, run) = fixture();
    let make = |index| {
        let source = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut slice =
            SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        slice.resolution = [16, 16];
        PolicyRequest::new(
            source,
            None,
            index as u64,
            slice,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut interactive = Interactive::new(|| {}).unwrap();
    for index in [0, 2] {
        interactive.submit_policy(make(index)).unwrap();
        let end = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(interactive.policy_error().is_none());
    }
    let before = interactive.policy_metrics();
    interactive.submit_policy(make(0)).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    let after = interactive.policy_metrics();
    assert_eq!(interactive.policy_frame().unwrap().grid.current.revision, 0);
    assert_eq!(
        interactive
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .value,
        0.
    );
    assert_eq!(after.restores, before.restores);
    assert_eq!(after.grid_batches, before.grid_batches);
    retired(&mut interactive);
}

#[test]
fn policy_atomic_frame_keeps_old_selected_point_and_reuses_exact_model() {
    let (_dir, run) = fixture();
    let make = |index| {
        let s = prepared::PolicySource::model(&run, index, Some("actor")).unwrap();
        let mut c = SliceConfig::new(s.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
        c.resolution = [16, 16];
        PolicyRequest::new(
            s,
            None,
            index as u64,
            c,
            PolicyPin::Synthetic(Observation::new(vec![0.25])),
            None,
        )
        .unwrap()
    };
    let mut i = Interactive::new(|| {}).unwrap();
    i.submit_policy(make(0)).unwrap();
    await_value_completion(&mut i);
    i.poll_policy().unwrap();
    assert!(i.policy_frame().is_none());
    await_value_completion(&mut i);
    i.poll_policy().unwrap();
    i.submit_policy(make(2)).unwrap();
    assert_eq!(
        i.policy_frame().unwrap().selected.as_ref().unwrap().value,
        0.
    );
    await_value_completion(&mut i);
    i.poll_policy().unwrap();
    assert_eq!(i.policy_frame().unwrap().grid.current.revision, 0);
    await_value_completion(&mut i);
    i.poll_policy().unwrap();
    assert_eq!(
        i.policy_frame().unwrap().selected.as_ref().unwrap().value,
        2.
    );
    let batches = i.policy_metrics().grid_batches;
    let mut r = make(2);
    r.requested_revision = 100;
    i.submit_policy(r).unwrap();
    assert!(!i.policy_pending());
    assert_eq!(i.policy_metrics().grid_batches, batches);
    assert_eq!(i.policy_frame().unwrap().requested_revision, 100);
    retired(&mut i);
}

#[test]
fn snapshot_catalog_requires_readable_exact_weights_and_selects_newest_valid_sibling() {
    use crate::snapshot_catalog::{SnapshotCatalog, SnapshotPanel, SnapshotStatus};
    let (_dir, run) = fixture();
    std::fs::write(
        run.root().join(&run.recording().entries()[1].file),
        b"corrupt",
    )
    .unwrap();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    let mut catalog = SnapshotCatalog::new(
        &run,
        "online",
        SnapshotPanel::Value {
            name: Some("q".into()),
            twin: None,
            actor_required: true,
            output: ValueOutput::ExpectedReturn,
        },
        &i,
    )
    .unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while catalog.pending() {
        loader.poll();
        catalog.advance(&run, &mut loader, &mut i).unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(catalog.positions(), vec![0, 2]);
    assert!(matches!(
        catalog.entries()[1].status,
        SnapshotStatus::Rejected(_)
    ));
    assert_eq!(catalog.selected(2).unwrap().key, run.key(2).unwrap());
    drop(catalog);
    retired(&mut i);
}

#[test]
fn policy_optional_reference_failure_publishes_complete_current_and_unavailable_reference() {
    let (_dir, run) = fixture();
    let source = prepared::PolicySource::model(&run, 0, Some("actor")).unwrap();
    let reference = prepared::PolicySource::model(&run, 1, Some("actor")).unwrap();
    let mut c = SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
    c.resolution = [4, 4];
    let r = PolicyRequest::new(
        source,
        Some(reference),
        0,
        c,
        PolicyPin::Synthetic(Observation::new(vec![0.25])),
        None,
    )
    .unwrap();
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    let mut i = Interactive::new(|| {}).unwrap();
    i.submit_policy(r).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while i.policy_pending() {
        i.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(i.policy_error().is_none(), "{:?}", i.policy_error());
    let f = i.policy_frame().unwrap();
    assert_eq!(f.selected.as_ref().unwrap().value, 0.);
    assert!(f.grid.reference.is_none());
    assert!(f.selected.as_ref().unwrap().reference_value.is_none());
    retired(&mut i);
}

#[test]
fn pin_edit_during_next_grid_never_uses_held_old_grid() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    i.submit_value(request(&run, 0)).unwrap();
    wait(&mut i);
    let mut next = request(&run, 1);
    i.submit_value(next.clone()).unwrap();
    next.pin = ValuePin::State(Observation::new(vec![0.75]));
    i.submit_value(next).unwrap();
    wait(&mut i);
    let f = i.value_frame().unwrap();
    assert_eq!(f.grid.current().identity().revision, 1);
    assert_eq!(f.selected.as_ref().unwrap().value, 2.75);
    retired(&mut i);
}

#[test]
fn irregular_catalogs_differ_by_capability_reject_corrupt_weights_and_freeze_exact_sibling() {
    use crate::snapshot_catalog::{SnapshotCatalog, SnapshotPanel};
    let (_original, original) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        original.recording().manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    for (revision, remove, broken_weights, role) in [
        (0, None, false, "online"),
        (120, Some("q"), false, "online"),
        (240, None, false, "online"),
        (240, None, true, "online"),
        (360, Some("actor"), false, "online"),
        (420, None, false, "online"),
        (460, None, false, "online"),
        (480, None, false, "target"),
    ] {
        let mut chunk = original.recording().load(0).unwrap();
        let mut d = declaration(&chunk).unwrap().unwrap();
        chunk.context.identity.learner_update_revision = revision;
        chunk.context.counters.updates = revision;
        chunk.context.identity.model_role = role.into();
        d.revision = revision;
        d.model_role = role.into();
        if revision == 420 {
            d.critics
                .get_mut("q")
                .unwrap()
                .actor
                .as_mut()
                .unwrap()
                .identity = "opaque-training-policy-k420".into();
        }
        chunk.arrays.remove(VALUE_METADATA_KEY);
        attach_declaration(&mut chunk, &d).unwrap();
        if let Record::ModelSnapshot(s) = &mut chunk.record {
            if revision == 460 {
                // Two real exported adapters: opaque provenance is not a key
                // that authorizes choosing one of these models automatically.
                s.models
                    .insert("another-actor".into(), s.models["actor"].clone());
                let actor = s.configuration["models"]["actor"].clone();
                s.configuration["models"]["another-actor"] = actor;
            }
            if let Some(name) = remove {
                s.configuration["models"]
                    .as_object_mut()
                    .unwrap()
                    .remove(name);
                s.models.remove(name);
            }
            if broken_weights {
                let key = s.models["actor"].bytes.clone();
                chunk
                    .arrays
                    .insert(key, Array::u8(vec![3], vec![1, 2, 3]).unwrap());
            }
        }
        if remove == Some("q") {
            chunk.arrays.remove(VALUE_METADATA_KEY);
        }
        writer.append(&chunk).unwrap();
        if revision == 0 {
            writer
                .append(&Chunk {
                    context: RecordContext::new("worker-value", 37),
                    record: Record::Metrics(MetricRecord {
                        values: BTreeMap::from([("loss".into(), 1.)]),
                        unavailable: BTreeMap::new(),
                    }),
                    arrays: BTreeMap::new(),
                })
                .unwrap();
        }
    }
    writer.finish().unwrap();
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    for (panel, expected) in [
        (SnapshotPanel::Policy, vec![0, 120, 240, 420]),
        (
            SnapshotPanel::Value {
                name: Some("q".into()),
                twin: None,
                actor_required: true,
                output: ValueOutput::ExpectedReturn,
            },
            vec![0, 240, 420],
        ),
    ] {
        let mut catalog = SnapshotCatalog::new(&run, "online", panel.clone(), &i).unwrap();
        let end = Instant::now() + Duration::from_secs(10);
        while catalog.pending() {
            loader.poll();
            catalog.advance(&run, &mut loader, &mut i).unwrap();
            assert!(Instant::now() < end);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(catalog.positions(), expected, "{:?}", catalog.entries());
        assert_eq!(
            catalog.selected(240).unwrap().key,
            run.key(3).unwrap(),
            "newer corrupt sibling never substitutes for selected readable identity"
        );
        assert!(catalog.selected(37).is_none());
        assert!(!catalog.matches(&run, "target", &SnapshotPanel::Policy));
        if matches!(panel, SnapshotPanel::Value { .. }) {
            let source = loop {
                loader.poll();
                if let Some(source) = catalog.value_source(&run, 420, &mut loader).unwrap() {
                    break source;
                }
                assert!(Instant::now() < end);
                std::thread::sleep(Duration::from_millis(1));
            };
            assert_eq!(source.actor_name(), Some("actor"));
            assert_eq!(
                source.declaration().unwrap().critics["q"]
                    .actor
                    .as_ref()
                    .unwrap()
                    .identity,
                "opaque-training-policy-k420"
            );
            let index = run.by_sequence(source.identity().key.sequence).unwrap();
            let chunk = run.recording().load(index).unwrap();
            assert!(
                ValueSource::model_with_actor(
                    &run,
                    index,
                    "q",
                    "opaque-training-policy-k420",
                    &chunk
                )
                .is_err()
            );
            assert!(ValueSource::model_with_actor(&run, index, "q", "q", &chunk).is_err());
            let mut request = request(&run, index);
            request.current = source;
            i.submit_value(request).unwrap();
            wait(&mut i);
            let frame = i.value_frame().unwrap();
            assert_eq!(frame.grid.current().identity().revision, 420);
            assert_eq!(
                frame.selected.as_ref().unwrap().input.dense_values(),
                vec![0.25]
            );
            assert_eq!(frame.selected.as_ref().unwrap().value, 0.25);
            assert!(
                frame
                    .grid
                    .policy_overlay
                    .iter()
                    .all(|a| matches!(a,Some(Action::Continuous(v))if v==&vec![0.]))
            );
        }
    }
    retired(&mut i);
}

#[test]
fn live_catalog_discovers_appends_without_replacing_published_same_revision_identity() {
    use crate::snapshot_catalog::{SnapshotCatalog, SnapshotPanel};
    let (_original, original) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Recorder::create(
        dir.path().join("run"),
        original.recording().manifest().clone(),
        StorageConfig::default(),
    )
    .unwrap();
    writer
        .append(&original.recording().load(0).unwrap())
        .unwrap();
    let mut run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut loader = Loader::new(4 * 1024 * 1024, || {}).unwrap();
    let mut catalog = SnapshotCatalog::new(&run, "online", SnapshotPanel::Policy, &i).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while catalog.pending() {
        loader.poll();
        catalog.advance(&run, &mut loader, &mut i).unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    writer
        .append(&original.recording().load(0).unwrap())
        .unwrap();
    writer
        .append(&original.recording().load(1).unwrap())
        .unwrap();
    run.refresh().unwrap();
    for _ in 0..100 {
        loader.poll();
        catalog.advance(&run, &mut loader, &mut i).unwrap();
        if catalog.positions() == [0, 1] {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(catalog.positions(), vec![0, 1]);
    assert_eq!(catalog.selected(0).unwrap().key.sequence, 0);
    drop(catalog);
    retired(&mut i);
}

#[test]
fn held_value_frame_and_staged_64_reference_fit_existing_shared_budget() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut r = request(&run, 0);
    let slice = Arc::make_mut(&mut r.slice);
    slice.mode = crate::policy::slice::SliceMode::TwoDimensional;
    slice.y = Some(crate::value::slice::ValueCoordinate::Action(0));
    slice.resolution = [64, 64];
    let view = i
        .reserve_policy_view(
            crate::policy::view::PolicyPane::reservation_bound(r.current.semantics()).unwrap()
                + 768 * 1024,
        )
        .unwrap();
    let catalog = i.reserve_policy_view(256 * 1024).unwrap();
    i.submit_value(r.clone()).unwrap();
    wait(&mut i);
    let old = i.value_frame().unwrap().selected.as_ref().unwrap().clone();
    r.reference_value = Some(request(&run, 1).current);
    r.reference_policy = Some(prepared::PolicySource::model(&run, 1, Some("actor")).unwrap());
    i.submit_value(r).unwrap();
    assert!(Arc::ptr_eq(
        i.value_frame().unwrap().selected.as_ref().unwrap(),
        &old
    ));
    wait(&mut i);
    assert!(
        i.value_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .reference_value
            .is_some()
    );
    let validation = i.reserve_policy_view(512 * 1024).unwrap();
    drop(validation);
    let mut next = i.value_desired.as_ref().unwrap().request.clone();
    next.current = next.reference_value.clone().unwrap();
    next.requested_revision = 1;
    i.submit_value(next).unwrap();
    wait(&mut i);
    assert_eq!(
        i.value_frame().unwrap().grid.current().identity().revision,
        1
    );
    let validation = i.reserve_policy_view(512 * 1024).unwrap();
    drop(validation);
    eprintln!(
        "held/staged64 view={} catalog={} metrics={:?}",
        view.bytes,
        catalog.bytes,
        i.value_metrics()
    );
    assert!(i.value_metrics().prepared_peak <= 4 * 1024 * 1024);
    drop((old, view, catalog));
    retired(&mut i);
}

#[test]
fn review_fix_policy_optional_loss_after_grid_answer_before_poll() {
    policy_optional_answer_loss(false);
}
#[test]
fn review_fix_policy_optional_loss_after_point_answer_before_poll() {
    policy_optional_answer_loss(true);
}
fn policy_optional_answer_loss(point: bool) {
    let (_dir, run) = fixture();
    let r = policy_reference_request(&run);
    let mut i = Interactive::new(|| {}).unwrap();
    i.submit_policy(r).unwrap();
    await_value_completion(&mut i); // Poll the shared worker only, not Policy publication.
    if point {
        i.poll_policy().unwrap();
        await_value_completion(&mut i);
    }
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    i.poll_policy().unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while i.policy_pending() {
        i.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(i.policy_error().is_none(), "{:?}", i.policy_error());
    let f = i.policy_frame().unwrap();
    assert_eq!(f.selected.as_ref().unwrap().value, 0.);
    assert!(
        f.grid.reference.is_none(),
        "unavailable reference cannot publish"
    );
    assert!(f.selected.as_ref().unwrap().reference_value.is_none());
    retired(&mut i);
}
fn policy_reference_request(run: &RunIndex) -> PolicyRequest {
    let source = prepared::PolicySource::model(run, 0, Some("actor")).unwrap();
    let reference = prepared::PolicySource::model(run, 1, Some("actor")).unwrap();
    let mut c = SliceConfig::new(source.semantics(), &Observation::new(vec![0.]), &[]).unwrap();
    c.resolution = [4, 4];
    PolicyRequest::new(
        source,
        Some(reference),
        0,
        c,
        PolicyPin::Synthetic(Observation::new(vec![0.25])),
        None,
    )
    .unwrap()
}

#[test]
fn optional_policy_critic_curve_yields_to_validation_headroom() {
    let (_dir, run) = fixture_with_space(ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    });
    let mut request = policy_reference_request(&run);
    request.reference = None;
    request.current = request
        .current
        .with_metadata(&run, &run.recording().load(0).unwrap())
        .unwrap();
    let wait_policy = |interactive: &mut Interactive| {
        let deadline = Instant::now() + Duration::from_secs(10);
        while interactive.policy_pending() {
            interactive.poll_policy().unwrap();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            interactive.policy_error().is_none(),
            "{:?}",
            interactive.policy_error()
        );
    };
    let mut baseline = Interactive::new(|| {}).unwrap();
    baseline.submit_policy(request.clone()).unwrap();
    wait_policy(&mut baseline);
    let curve = baseline
        .policy_frame()
        .unwrap()
        .selected
        .as_ref()
        .unwrap()
        .action_value_curve()
        .unwrap();
    let curve_bytes = curve.points.capacity() * std::mem::size_of::<[f64; 2]>()
        + curve.critic.capacity()
        + curve.reward_units.capacity()
        + curve.role.capacity()
        + std::mem::size_of::<prepared::PreparedPolicyActionValue>();
    let base = baseline.policy_metrics().prepared_bytes - curve_bytes;
    retired(&mut baseline);

    let mut constrained = Interactive::new(|| {}).unwrap();
    let filler = constrained
        .reserve_policy_view(crate::budgets::POLICY_PREPARED - 512 * 1024 - base - curve_bytes / 2)
        .unwrap();
    constrained.submit_policy(request).unwrap();
    wait_policy(&mut constrained);
    assert!(
        constrained
            .policy_frame()
            .unwrap()
            .selected
            .as_ref()
            .unwrap()
            .action_value_curve()
            .is_none()
    );
    let validation = constrained.reserve_policy_view(512 * 1024).unwrap();
    assert!(constrained.policy_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    drop((validation, filler));
    retired(&mut constrained);
}

#[test]
fn cached_policy_grid_recomputes_an_evicted_point_before_publication() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let request = policy_reference_request(&run);
    interactive.submit_policy(request.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    interactive.suspend_policy_for_value();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let grid = Arc::downgrade(&interactive.policy_recent[0].frame.grid);
    interactive.policy_recent[0].frame.selected = None;
    let before = interactive.policy_metrics();
    interactive.submit_policy(request).unwrap();
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let frame = interactive.policy_frame().unwrap();
    assert!(Arc::ptr_eq(&grid.upgrade().unwrap(), &frame.grid));
    assert_eq!(
        frame.selected.as_ref().unwrap().input,
        Observation::new(vec![0.25])
    );
    let after = interactive.policy_metrics();
    assert_eq!(after.grid_batches, before.grid_batches);
    assert!(after.point_batches > before.point_batches);
    retired(&mut interactive);
}

#[test]
fn value_pressure_discards_policy_point_before_the_grid() {
    let (_dir, run) = fixture();
    let mut interactive = Interactive::new(|| {}).unwrap();
    let policy = policy_reference_request(&run);
    interactive.submit_policy(policy.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    interactive.suspend_policy_for_value();
    while interactive.retirement_status().0 != interactive.retirement_status().1 {
        interactive.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let grid = Arc::downgrade(&interactive.policy_recent[0].frame.grid);
    let value = request(&run, 0);
    let complete = value_prepared::deferred_publication_charge(&value).unwrap();
    let base = complete - value.metadata_bytes();
    let incremental = base
        + value_prepared::RequestOwner::metadata_charge(
            &value,
            &interactive.worker.policy_accounting,
        )
        .unwrap();
    let standalone = base + value_prepared::RequestOwner::standalone_metadata_charge(&value);
    let target = crate::budgets::POLICY_PREPARED - 512 * 1024 - incremental.max(standalone) + 8192;
    let filler = interactive
        .reserve_policy_view(
            target
                .checked_sub(interactive.policy_metrics().prepared_bytes)
                .unwrap(),
        )
        .unwrap();
    interactive.submit_value(value).unwrap();
    wait(&mut interactive);
    assert_eq!(interactive.policy_recent.len(), 1);
    assert!(interactive.policy_recent[0].frame.selected.is_none());
    assert!(Arc::ptr_eq(
        &grid.upgrade().unwrap(),
        &interactive.policy_recent[0].frame.grid
    ));
    let validation = interactive.reserve_policy_view(512 * 1024).unwrap();
    drop((validation, filler));
    let before = interactive.policy_metrics();
    interactive.submit_policy(policy).unwrap();
    while interactive.policy_pending() {
        interactive.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        interactive.policy_metrics().grid_batches,
        before.grid_batches
    );
    assert!(interactive.policy_frame().unwrap().selected.is_some());
    retired(&mut interactive);
}

#[test]
fn empty_policy_reference_releases_its_projection_without_losing_matches() {
    let (_dir, run) = fixture();
    for has_reference in [false, true] {
        let accounting = Arc::new(prepared::Accounting::default());
        let mut request = policy_reference_request(&run);
        if !has_reference {
            request.reference = None;
        }
        Arc::make_mut(&mut request.slice).resolution = [128, 128];
        let owner = RequestOwner::new(request, &accounting).unwrap();
        let before = accounting.snapshot().prepared_bytes;
        let completed = prepared::execute(
            &Work::Grid(owner.clone()),
            &accounting,
            || false,
            |source, inputs| {
                let index = run.by_sequence(source.identity().key.sequence).unwrap();
                let model = ModelBundle::load(
                    &run.recording().load(index).unwrap(),
                    ModelLimits::default(),
                )
                .unwrap();
                rl::diagnostics::policy_surface::evaluate_policy_batch(&model, "actor", inputs)
                    .map(|r| (r, "actor".into()))
            },
        )
        .unwrap()
        .unwrap();
        let Completion::Grid(grid) = completed else {
            panic!("grid expected")
        };
        let n = grid.values.len();
        assert_eq!(
            grid.projection_bytes(),
            n * 8 * if has_reference { 2 } else { 1 }
        );
        assert_eq!(
            accounting.snapshot().prepared_bytes - before,
            n * 8 * if has_reference { 2 } else { 1 } + 32 * 1024
        );
        for i in 0..n {
            assert_eq!(grid.reference_at(i).is_some(), has_reference);
        }
        drop(grid);
        assert_eq!(accounting.snapshot().prepared_bytes, before);
    }
}
#[test]
fn review_fix_policy_optional_loss_after_grid_inference_is_typed() {
    policy_optional_inference_loss(false);
}
#[test]
fn review_fix_policy_optional_loss_after_point_inference_is_typed() {
    policy_optional_inference_loss(true);
}
fn policy_optional_inference_loss(point: bool) {
    let (_dir, run) = fixture();
    let accounting = Arc::new(prepared::Accounting::default());
    let owner = RequestOwner::new(policy_reference_request(&run), &accounting).unwrap();
    let inspect = |source: &prepared::PolicySource, inputs: &[Observation]| {
        let index = run.by_sequence(source.identity().key.sequence).unwrap();
        let model = ModelBundle::load(
            &run.recording().load(index).unwrap(),
            ModelLimits::default(),
        )
        .unwrap();
        rl::diagnostics::policy_surface::evaluate_policy_batch(&model, "actor", inputs)
            .map(|r| (r, "actor".into()))
    };
    let work = if point {
        let completed =
            prepared::execute(&Work::Grid(owner.clone()), &accounting, || false, inspect)
                .unwrap()
                .unwrap();
        let Completion::Grid(grid) = completed else {
            panic!("grid expected");
        };
        Work::Point {
            request: owner.clone(),
            grid,
            previous: None,
        }
    } else {
        Work::Grid(owner.clone())
    };
    let result = prepared::execute(
        &work,
        &accounting,
        || false,
        |source, inputs| {
            let result = inspect(source, inputs);
            if source.identity().key.sequence == 1 {
                std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
            }
            result
        },
    );
    assert!(
        matches!(result, Ok(Some(Completion::ReferenceUnavailable))),
        "successful reference inference followed by retention must yield optional unavailability"
    );
}

#[test]
fn review_fix_policy_current_loss_before_poll_is_not_optional_retry() {
    let (_dir, run) = fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    let mut held_request = policy_reference_request(&run);
    held_request.current = prepared::PolicySource::model(&run, 2, Some("actor")).unwrap();
    held_request.reference = None;
    i.submit_policy(held_request).unwrap();
    let end = Instant::now() + Duration::from_secs(10);
    while i.policy_pending() {
        i.poll_policy().unwrap();
        assert!(Instant::now() < end);
        std::thread::sleep(Duration::from_millis(1));
    }
    let old = i.policy_frame().unwrap().clone();
    i.submit_policy(policy_reference_request(&run)).unwrap();
    await_value_completion(&mut i);
    i.poll_policy().unwrap();
    await_value_completion(&mut i);
    let batches = i.policy_metrics().grid_batches;
    for index in [0, 1] {
        std::fs::remove_file(run.root().join(&run.recording().entries()[index].file)).unwrap();
    }
    i.poll_policy().unwrap();
    assert!(i.policy_error().is_some());
    assert!(!i.policy_pending());
    assert_eq!(i.policy_metrics().grid_batches, batches);
    let held = i.policy_frame().unwrap();
    assert!(Arc::ptr_eq(&old.grid, &held.grid));
    assert!(Arc::ptr_eq(
        old.selected.as_ref().unwrap(),
        held.selected.as_ref().unwrap()
    ));
    drop(old);
    retired(&mut i);
}
