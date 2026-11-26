#[path = "support/dynamics.rs"]
mod support;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    diagnostics::{models::SnapshotBuilder, schema::*, storage::*},
    prelude::*,
};
use rl_viewer::{
    data::RunIndex,
    dynamics::{Limits, prepare_episode},
    interactive::{Interactive, decision::Request},
};
use std::time::{Duration, Instant};

#[test]
fn one_selected_task_keeps_recorded_vector_q_input_separate_from_inferred_action() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let space = support::bounded(3);
    let manifest = support::manifest(space.clone());
    let trajectory = support::chunk(&space);
    let episode = prepare_episode(&trajectory, &manifest, &Limits::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut context = RecordContext::new("dynamics", 17);
    context.identity.model_role = "target".into();
    let mut builder = SnapshotBuilder::new("exact-17", context, 3, space.clone()).unwrap();
    builder
        .deterministic(
            "policy",
            &Deterministic::new(3, space, &[4], &mut rng).unwrap(),
        )
        .unwrap();
    builder
        .scalar("q", &QValue::new(3, 3, 1, &[4], &mut rng).unwrap())
        .unwrap();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.append(&trajectory).unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut different = trajectory.clone();
    different.context.identity.learner_update_revision = 18;
    different.context.counters.updates = 18;
    let different =
        prepare_episode(&different, run.recording().manifest(), &Limits::default()).unwrap();
    assert!(
        Request::new(&run, 1, run.key(0).unwrap(), &different, 2).is_err(),
        "episode payload must belong to its immutable key"
    );
    let mut interactive = Interactive::new(|| {}).unwrap();
    for step in [0, 1, 2] {
        interactive
            .submit_decision(Request::new(&run, 1, run.key(0).unwrap(), &episode, step).unwrap())
            .unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.decision_pending() {
        interactive.poll_decision().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let frame = interactive.decision_frame().unwrap();
    assert_eq!(frame.step, 2);
    assert_eq!(frame.episode_revision, 17);
    assert_eq!(frame.model_revision, 17);
    assert_eq!(
        frame.recorded_action,
        Some(Action::Continuous(vec![2., 12., 22.]))
    );
    assert!(matches!(&frame.inferred_action,Some(Action::Continuous(v))if v.len()==3));
    assert!(frame.critics.contains_key("q"));
    assert!(interactive.decision_error().is_none());
    assert_eq!(
        interactive.value_metrics().grid_batches,
        0,
        "selected prediction never prepares a grid"
    );
    assert!(
        Request::new(&run, 1, run.key(0).unwrap(), &episode, 4).is_err(),
        "final state is not a live decision"
    );
}

#[test]
fn selected_prediction_rejects_oversized_labels_before_cloning_answer_maps() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let space = support::bounded(3);
    let manifest = support::manifest(space.clone());
    let trajectory = support::chunk(&space);
    let episode = prepare_episode(&trajectory, &manifest, &Limits::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let mut context = RecordContext::new("dynamics", 17);
    context.identity.model_role = "target".into();
    let mut builder = SnapshotBuilder::new("long-label-17", context, 3, space).unwrap();
    builder
        .scalar(
            &"q".repeat(1025),
            &QValue::new(3, 3, 1, &[4], &mut rng).unwrap(),
        )
        .unwrap();
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.append(&trajectory).unwrap();
    writer.append(&builder.finish().unwrap()).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut interactive = Interactive::new(|| {}).unwrap();
    interactive
        .submit_decision(Request::new(&run, 1, run.key(0).unwrap(), &episode, 1).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while interactive.decision_pending() {
        interactive.poll_decision().unwrap();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(interactive.decision_frame().is_none());
    assert!(
        interactive
            .decision_error()
            .is_some_and(|reason| reason.contains("model name") && reason.contains("admission")),
        "{:?}",
        interactive.decision_error()
    );
}
