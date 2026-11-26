use super::*;
use std::time::{Duration, Instant};

fn settle(i: &mut Interactive) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while i.pending() {
        i.poll_policy().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn warmup_restores_exact_successor_once_without_preparing_or_publishing() {
    let (_dir, run) = value_worker_tests::fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    assert!(i.warm_snapshot(&run, 1, vec![run.key(0).unwrap()]).unwrap());
    settle(&mut i);
    assert_eq!(i.policy_metrics().restores, 1);
    assert_eq!(i.policy_metrics().grid_batches, 0);
    assert_eq!(i.policy_metrics().point_batches, 0);
    assert!(i.policy_frame().is_none());
    assert!(i.value_frame().is_none());
    for _ in 0..10 {
        assert!(!i.warm_snapshot(&run, 1, vec![run.key(0).unwrap()]).unwrap());
        i.poll_policy().unwrap();
    }
    assert_eq!(i.policy_metrics().restores, 1);
}

fn validate(i: &mut Interactive, run: &RunIndex, index: usize) {
    i.validate_snapshot(run, index).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = i.poll_snapshot_validation().unwrap() {
            assert_eq!(result.unwrap(), run.key(index).unwrap());
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn warmup_protects_current_evicts_stale_and_refuses_independent_reference() {
    let (_dir, run) = value_worker_tests::fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    validate(&mut i, &run, 0);
    validate(&mut i, &run, 2);
    let current = run.key(0).unwrap();
    assert!(
        i.warm_snapshot(&run, 1, vec![current.clone(), current])
            .unwrap()
    );
    settle(&mut i);
    assert_eq!(i.policy_metrics().restores, 3);
    validate(&mut i, &run, 0);
    validate(&mut i, &run, 1);
    assert_eq!(
        i.policy_metrics().restores,
        3,
        "both required bundles retained"
    );
    assert!(
        !i.warm_snapshot(&run, 2, vec![run.key(0).unwrap(), run.key(1).unwrap()])
            .unwrap()
    );
    settle(&mut i);
    validate(&mut i, &run, 0);
    validate(&mut i, &run, 1);
    assert_eq!(i.policy_metrics().restores, 3);
}

#[test]
fn lost_successor_is_not_retried_or_reported_as_a_foreground_error() {
    let (_dir, run) = value_worker_tests::fixture();
    let mut i = Interactive::new(|| {}).unwrap();
    validate(&mut i, &run, 1); // Even a cache hit must recheck availability.
    std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
    let _ = i.warm_snapshot(&run, 1, vec![run.key(0).unwrap()]).unwrap();
    settle(&mut i);
    assert!(i.policy_error().is_none());
    assert!(i.value_error().is_none());
    assert!(!i.warm_snapshot(&run, 1, vec![run.key(0).unwrap()]).unwrap());
    assert_eq!(i.policy_metrics().restores, 1);
}

#[test]
fn foreground_cancels_blocked_warmup_and_retires_its_charge() {
    let (_dir, run) = value_worker_tests::fixture();
    let (started, starts) = mpsc::channel();
    let (release, blocked) = mpsc::channel();
    let mut i = Interactive::new(|| {}).unwrap();
    i.worker = Worker::spawn_tasks(
        || {},
        move |job, _, _| match job {
            Some(Task::Warm(..)) => {
                started.send(()).unwrap();
                blocked.recv().unwrap();
                Ok(Some(TaskAnswer::Warmed))
            }
            Some(Task::Validate(key, _, _)) => Ok(Some(TaskAnswer::Validated(key.clone()))),
            _ => Ok(None),
        },
    )
    .unwrap();
    assert!(i.warm_snapshot(&run, 1, vec![run.key(0).unwrap()]).unwrap());
    starts.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(!i.policy_pending(), "warmup cannot gate readiness");
    i.validate_snapshot(&run, 2).unwrap();
    assert!(!i.warming);
    release.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = i.poll_snapshot_validation().unwrap() {
            assert_eq!(result.unwrap(), run.key(2).unwrap());
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(i.policy_metrics().prepared_bytes, 0);
    assert!(i.policy_frame().is_none());
}
