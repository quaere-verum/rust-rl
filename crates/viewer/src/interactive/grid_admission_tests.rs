//! Bounded, real serialized-model preparation evidence; these fixtures are not trained runs.
use super::*;
#[allow(dead_code)]
#[path = "../../tests/support/value_ui.rs"]
mod fixture;
use crate::{
    policy::{prepared::VisitSource, slice::SliceConfig, view::PolicyPane},
    value::{
        prepared::{ValuePin, ValueRequest, ValueSource},
        slice::ValueSlice,
    },
};
use rl::{core::env::Observation, diagnostics::storage::ReadLimits};
use std::time::{Duration, Instant};

fn value_request(run: &RunIndex, index: usize, n: usize) -> ValueRequest {
    let current = ValueSource::model(run, index, "omega", &run.recording().load(index).unwrap())
        .unwrap()
        .with_twin("declared-pair")
        .unwrap();
    let input = Observation::new(vec![0.; 3]);
    let mut slice = ValueSlice::new(current.semantics(), &input, &[]).unwrap();
    slice.resolution = [n, n];
    let mut r = ValueRequest::new(
        current,
        rl::diagnostics::value::ValueOutput::Q1,
        slice,
        ValuePin::State(input),
    )
    .unwrap();
    r.reference_value = Some(
        ValueSource::model(run, 1, "omega", &run.recording().load(1).unwrap())
            .unwrap()
            .with_twin("declared-pair")
            .unwrap(),
    );
    r.reference_policy = Some(prepared::PolicySource::model(run, 1, Some("controller")).unwrap());
    let mut resolver = crate::resolver::ProbeResolver::new(1024 * 1024);
    let key = run.key(3).unwrap();
    let chunk = resolver
        .lease_visitation(run, &key, Arc::new(run.recording().load(3).unwrap()))
        .unwrap();
    r.visits.push(VisitSource { key, chunk });
    r
}
fn wait(i: &mut Interactive, value: bool) {
    let deadline = Instant::now() + Duration::from_secs(15);
    while if value {
        i.value_pending()
    } else {
        i.policy_pending()
    } {
        if value {
            i.poll_value().unwrap();
        } else {
            i.poll_policy().unwrap();
        }
        assert!(Instant::now() < deadline, "bounded preparation deadline");
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn measured_regular_grids_64_96_128_with_retained_view_catalog_and_reference() {
    let (_dir, run) = fixture_run();
    for value in [false, true] {
        for n in [64, 96, 128] {
            let mut i = Interactive::new(|| {}).unwrap();
            let bound = PolicyPane::reservation_bound(
                rl::diagnostics::visualization::EnvironmentSemantics::new(
                    run.recording().manifest(),
                ),
            )
            .unwrap()
                + if value { 768 * 1024 } else { 0 };
            let _view = i.reserve_policy_view(bound).unwrap();
            let _catalog = i.reserve_policy_view(256 * 1024).unwrap();
            let start = Instant::now();
            if value {
                i.submit_value(value_request(&run, 2, n)).unwrap();
            } else {
                let current = prepared::PolicySource::model(&run, 2, Some("controller")).unwrap();
                let input = Observation::new(vec![0.; 3]);
                let mut slice = SliceConfig::new(current.semantics(), &input, &[]).unwrap();
                slice.resolution = [n, n];
                let r = PolicyRequest::new(
                    current,
                    Some(prepared::PolicySource::model(&run, 1, Some("controller")).unwrap()),
                    7,
                    slice,
                    prepared::PolicyPin::Synthetic(input),
                    None,
                )
                .unwrap();
                i.submit_policy(r).unwrap();
            }
            wait(&mut i, value);
            let elapsed = start.elapsed();
            let error = if value {
                i.value_error()
            } else {
                i.policy_error()
            };
            eprintln!(
                "GRID family={} requested={n}x{n} latency_us={} metrics={:?} error={error:?}",
                if value {
                    "twin-value+references+visits"
                } else {
                    "policy+reference"
                },
                elapsed.as_micros(),
                i.policy_metrics()
            );
            if value && n == 128 {
                assert!(
                    error.is_some(),
                    "128 Value exceeds the unchanged frame envelope"
                );
            }
            if !value || n == 64 {
                assert!(error.is_none(), "{error:?}");
            }
        }
    }
}
fn fixture_run() -> (tempfile::TempDir, RunIndex) {
    let dir = fixture::family("twin", true);
    let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    (dir, run)
}
#[test]
fn automatic_value_plans_the_reference_successor_and_validation_headroom() {
    let (_dir, run) = fixture_run();
    let mut i = Interactive::new(|| {}).unwrap();
    let bound = PolicyPane::reservation_bound(
        rl::diagnostics::visualization::EnvironmentSemantics::new(run.recording().manifest()),
    )
    .unwrap()
        + 768 * 1024;
    let _view = i.reserve_policy_view(bound).unwrap();
    let _catalog = i.reserve_policy_view(256 * 1024).unwrap();
    let reference = value_request(&run, 2, 64);
    for index in [2, 1, 2] {
        let mut r = value_request(&run, index, 64);
        // Reference owners are deliberately held stable across current snapshots.
        r.reference_value = reference.reference_value.clone();
        r.reference_policy = reference.reference_policy.clone();
        let planning = Instant::now();
        i.automatic_value_resolution(&mut r, 0).unwrap();
        eprintln!("PREFLIGHT elapsed_us={}", planning.elapsed().as_micros());
        assert_eq!(r.slice.resolution, [48, 48]);
        i.submit_value(r).unwrap();
        wait(&mut i, true);
        assert!(i.value_error().is_none(), "{:?}", i.value_error());
        assert_eq!(
            i.value_frame().unwrap().grid.current().identity().key,
            run.key(index).unwrap()
        );
        let validation = i
            .reserve_policy_view(512 * 1024)
            .expect("catalog discovery can continue after publication");
        drop(validation);
        eprintln!(
            "AUTO successor_index={index} selected=48x48 metrics={:?}",
            i.value_metrics()
        );
        assert!(i.value_metrics().prepared_peak <= crate::budgets::POLICY_PREPARED);
    }
    i.retire_session();
    drop(_view);
    drop(_catalog);
    let deadline = Instant::now() + Duration::from_secs(5);
    while i.value_metrics().prepared_bytes != 0 {
        i.housekeeping();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    eprintln!("AUTO retirement prepared_bytes=0");
}
#[test]
fn preflight_metadata_matches_actual_leases_and_keeps_independent_equal_sources() {
    let (_dir, run) = fixture_run();
    let accounting = Arc::new(Accounting::default());
    let request = value_request(&run, 2, 64);
    let owned = request.clone();
    let predicted = value_prepared::RequestOwner::metadata_charge(&owned, &accounting).unwrap();
    let first = value_prepared::RequestOwner::new(owned, &accounting).unwrap();
    assert_eq!(accounting.snapshot().prepared_bytes, predicted);
    let owned = request.clone();
    let shared = value_prepared::RequestOwner::metadata_charge(&owned, &accounting).unwrap();
    let second = value_prepared::RequestOwner::new(owned, &accounting).unwrap();
    assert_eq!(accounting.snapshot().prepared_bytes, predicted + shared);
    let independent = value_request(&run, 2, 64);
    assert_eq!(independent.current.identity(), request.current.identity());
    let additional =
        value_prepared::RequestOwner::metadata_charge(&independent, &accounting).unwrap();
    assert!(
        additional > shared,
        "independently allocated equal-identity metadata must remain charged"
    );
    let third = value_prepared::RequestOwner::new(independent, &accounting).unwrap();
    assert_eq!(
        accounting.snapshot().prepared_bytes,
        predicted + shared + additional
    );
    drop((first, second, third));
    assert_eq!(accounting.snapshot().prepared_bytes, 0);
}

#[test]
fn hypothetical_retirement_counts_only_allocations_whose_last_lease_leaves() {
    use crate::policy::prepared::SharedLease;
    let accounting = Arc::new(Accounting::default());
    let retiring = vec![
        SharedLease::acquire(&accounting, 1, 100).unwrap(),
        SharedLease::acquire(&accounting, 2, 200).unwrap(),
    ];
    let survivor = SharedLease::acquire(&accounting, 2, 200).unwrap();
    let allocations = [(1, 100), (1, 100), (2, 200), (3, 300)];
    assert_eq!(
        SharedLease::additional_charge(&accounting, &allocations).unwrap(),
        300
    );
    assert_eq!(
        SharedLease::additional_charge_after(&accounting, &allocations, &retiring).unwrap(),
        400
    );
    drop(retiring);
    assert_eq!(
        SharedLease::additional_charge(&accounting, &allocations).unwrap(),
        400
    );
    drop(survivor);
    assert_eq!(accounting.snapshot().prepared_bytes, 0);
}
