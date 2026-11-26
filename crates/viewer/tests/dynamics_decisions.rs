#[path = "support/dynamics.rs"]
mod support;
#[path = "support/policy_ui.rs"]
mod ui_support;
use rl::{
    core::action::ActionSpace,
    diagnostics::{schema::*, value::trajectory_return_to_go},
};
use rl_viewer::dynamics::{ActionAxis, ActionValue, DecisionAction, Limits, prepare_episode};
use rl_viewer::{
    data::RunIndex,
    dynamics::work::{DynamicsWorker, EpisodeRequest, WorkLimits},
};
use std::time::{Duration, Instant};

#[test]
fn exact_decision_exposes_complete_vector_next_input_flags_and_prepared_single_suffix() {
    let space = support::bounded(3);
    let manifest = support::manifest(space.clone());
    let chunk = support::chunk(&space);
    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    let step = episode.step(2).unwrap();
    assert_eq!(
        step.action,
        Some(DecisionAction::Continuous(&[2., 12., 22.]))
    );
    assert_eq!(step.next_observation, Some([2., 2., 13.].as_slice()));
    assert_eq!(step.terminated, Some(false));
    assert_eq!(step.truncated, Some(false));
    assert_eq!(
        step.return_to_go,
        trajectory_return_to_go(&chunk, 2).unwrap()
    );
    assert_eq!(step.return_to_go, Some(3. + 0.95 * 4.));
    let final_step = episode.step(4).unwrap();
    assert_eq!(final_step.action, None);
    assert_eq!(final_step.next_observation, None);
    assert_eq!(final_step.terminated, None);
    assert_eq!(final_step.truncated, None);
    assert_eq!(final_step.return_to_go, None);
}

#[test]
fn masks_survive_preparation_and_truncated_suffixes_remain_unavailable() {
    let space = ActionSpace::Discrete { count: 3 };
    let manifest = support::manifest(space.clone());
    let mut chunk = support::chunk(&space);
    if let Record::Trajectory(t) = &mut chunk.record {
        t.action_masks = Some("mask".into());
    }
    chunk.arrays.insert(
        "mask".into(),
        Array::u8(
            vec![5, 3],
            vec![1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        )
        .unwrap(),
    );
    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    assert_eq!(
        episode.step(1).unwrap().action,
        Some(DecisionAction::Discrete(2))
    );
    assert_eq!(episode.step(1).unwrap().mask, Some([0, 0, 1].as_slice()));
    assert_eq!(episode.step(4).unwrap().mask, Some([0, 0, 0].as_slice()));
    assert!(episode.retained_bytes() >= 15);
    if let Record::Trajectory(t) = &mut chunk.record {
        t.evaluation.complete_returns = false;
    }
    chunk
        .arrays
        .insert("terminated".into(), Array::u8(vec![4], vec![0; 4]).unwrap());
    chunk.arrays.insert(
        "truncated".into(),
        Array::u8(vec![4], vec![0, 0, 0, 1]).unwrap(),
    );
    let episode = prepare_episode(&chunk, &manifest, &Limits::default()).unwrap();
    assert_eq!(episode.step(3).unwrap().truncated, Some(true));
    assert_eq!(episode.step(1).unwrap().return_to_go, None);
}

#[test]
fn generic_worker_admits_one_axis_conditional_episode_without_spatial_population() {
    use rl::diagnostics::storage::*;
    let temp = tempfile::tempdir().unwrap();
    let space = support::bounded(3);
    let mut manifest = support::manifest(space.clone());
    manifest.observation_dim = 1;
    manifest.environment_descriptor = None;
    let mut chunk = support::chunk(&space);
    chunk.arrays.insert(
        "obs".into(),
        Array::f32(vec![5, 1], vec![0., 1., 1., 2., 0.]).unwrap(),
    );
    if let Record::Trajectory(t) = &mut chunk.record {
        t.evaluation.conditioning = Conditioning::FullSimulatorState {
            description: "descriptive conditional episode".into(),
        };
    }
    let path = temp.path().join("run");
    let mut recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    recorder.append(&chunk).unwrap();
    recorder.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut worker = DynamicsWorker::new(WorkLimits::default(), || {}).unwrap();
    assert!(
        worker
            .request_episode(EpisodeRequest::new(&run, 0).unwrap())
            .unwrap()
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    let frame = worker.episode_frame().unwrap();
    assert_eq!(frame.key, run.key(0).unwrap());
    assert_eq!(frame.episode.numeric_coordinates(), [0]);
    assert_eq!(
        frame.episode.step(2).unwrap().action,
        Some(DecisionAction::Continuous(&[2., 12., 22.]))
    );
    assert!(worker.frame().is_none());
    assert_eq!(
        frame.reward_trace,
        vec![[0., 1.], [1., -2.], [2., 3.], [3., 4.]]
    );
    assert_eq!(frame.action_trace[2].value, ActionValue::Continuous(2.));
    assert!(
        !worker
            .request_episode(EpisodeRequest::new(&run, 0).unwrap())
            .unwrap(),
        "unchanged episode is idle"
    );
    worker
        .request_episode(
            EpisodeRequest::new(&run, 0)
                .unwrap()
                .with_action(ActionAxis::Continuous(1))
                .unwrap(),
        )
        .unwrap();
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        worker.episode_frame().unwrap().action,
        ActionAxis::Continuous(1)
    );
    assert_eq!(
        worker.episode_frame().unwrap().action_trace[2].value,
        ActionValue::Continuous(12.)
    );
    assert_eq!(
        worker.counters().source_reads,
        1,
        "component changes reuse the exact prepared episode"
    );
}

#[test]
fn zero_and_one_numerical_axis_keep_exact_episode_controls_and_shared_step() {
    use rl::diagnostics::{storage::*, visualization::*};
    use rl_viewer::{data::SelectedInput, ui::View};
    for numeric in [0, 1] {
        let temp = tempfile::tempdir().unwrap();
        let space = support::bounded(3);
        let mut manifest = support::manifest(space.clone());
        if numeric == 0 {
            manifest.metadata.insert(
                "inference_history".into(),
                "recorded episode has no selected recurrent memory".into(),
            );
        }
        manifest
            .environment_descriptor
            .as_mut()
            .unwrap()
            .preferences
            .coordinates
            .clear();
        for (i, field) in manifest
            .environment_descriptor
            .as_mut()
            .unwrap()
            .observations
            .iter_mut()
            .enumerate()
        {
            if i >= numeric {
                field.coordinate = false;
                field.range = None;
                field.kind = ObservationType::Categorical {
                    categories: (0..25)
                        .map(|n| Category {
                            code: n as f32,
                            label: n.to_string(),
                        })
                        .collect(),
                };
            }
        }
        let mut chunk = support::chunk(&space);
        chunk.context.identity.model_role = "online".into();
        if numeric == 0
            && let Record::Trajectory(t) = &mut chunk.record
        {
            t.evaluation.conditioning = Conditioning::FullSimulatorState {
                description: "conditional evidence stays inspectable".into(),
            };
        }
        let mut writer =
            Recorder::create(temp.path().join("run"), manifest, StorageConfig::default()).unwrap();
        writer.append(&chunk).unwrap();
        writer.finish().unwrap();
        let mut h = ui_support::Harness::new(&temp, eframe::egui::vec2(1440., 900.));
        h.app.select_view(View::Dynamics);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            h.frame(vec![]);
            if h.has("Action taken") {
                break;
            }
            assert!(Instant::now() < deadline, "{:?}", h.text);
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(h.has("Observation") && h.has("Action taken") && h.has("Return-to-go"));
        assert!(h.has("Selected transition"));
        assert!(h.has("Episode timeline") && h.has("Action"));
        assert!(
            !h.has("Complete observation"),
            "complete vectors start disclosed"
        );
        if numeric == 0 {
            assert!(h.has("Unavailable · recurrent state missing"));
        }
        let selection = h.app.selection().input.clone();
        let counters = h.app.dynamics_counters();
        h.click("Decision details");
        h.frame(vec![]);
        assert!(h.has("Complete observation") && h.has("Return-to-go"));
        if numeric == 0 {
            assert!(h.has("Recurrent memory/history unavailable"));
        }
        assert_eq!(
            h.app.selection().input,
            selection,
            "Details is presentation only"
        );
        assert_eq!(
            h.app.dynamics_counters(),
            counters,
            "Details must not prepare or decode an episode"
        );
        h.click("Decision details");
        h.click("Next step");
        h.frame(vec![]);
        assert!(matches!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep { step: 1, .. }
        ));
        h.click("Final state");
        h.frame(vec![]);
        assert!(h.has("Final observation · no transition"));
        assert!(h.has("— no action") && h.has("— no reward") && h.has("— no next input"));
        assert!(!h.has("Return-to-go"));
    }
}

#[test]
fn generic_worker_keeps_latest_episode_and_invalidates_a_pruned_publication() {
    use rl::diagnostics::storage::*;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let space = support::bounded(1);
    let manifest = support::manifest(space.clone());
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    writer.append(&support::chunk(&space)).unwrap();
    writer.append(&support::chunk(&space)).unwrap();
    writer.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release = Arc::new(Mutex::new(release_rx));
    let first = AtomicBool::new(true);
    let mut worker = DynamicsWorker::with_reader(
        WorkLimits::default(),
        || {},
        move |source, limit| {
            if first.swap(false, Ordering::SeqCst) {
                entered_tx.send(()).unwrap();
                release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            source.load_if_within_charge(limit)
        },
    )
    .unwrap();
    worker
        .request_episode(EpisodeRequest::new(&run, 0).unwrap())
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    worker
        .request_episode(EpisodeRequest::new(&run, 1).unwrap())
        .unwrap();
    release_tx.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(worker.episode_frame().unwrap().key, run.key(1).unwrap());
    assert!(worker.counters().cancelled >= 1);
    let old = worker.episode_frame().unwrap().clone();
    std::fs::remove_file(path.join(&run.recording().entries()[1].file)).unwrap();
    worker.recheck_episode();
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(worker.episode_frame().is_none() && worker.error().is_some());
    assert_eq!(
        old.key,
        run.key(1).unwrap(),
        "external immutable old ownership remains identified"
    );
}

#[cfg(feature = "screenshots")]
#[test]
fn two_axis_episode_outside_spatial_protocol_keeps_its_generic_frame_across_paints() {
    use rl::diagnostics::{storage::*, visualization::*};
    use rl_viewer::{data::SelectedInput, ui::View};
    let temp = tempfile::tempdir().unwrap();
    let space = support::bounded(3);
    let mut manifest = support::manifest(space.clone());
    let third = &mut manifest
        .environment_descriptor
        .as_mut()
        .unwrap()
        .observations[2];
    third.coordinate = false;
    third.range = None;
    third.kind = ObservationType::Categorical {
        categories: (10..=14)
            .map(|n| Category {
                code: n as f32,
                label: n.to_string(),
            })
            .collect(),
    };
    let mut writer =
        Recorder::create(temp.path().join("run"), manifest, StorageConfig::default()).unwrap();
    for (revision, policy) in [(17, "first policy"), (63, "second policy")] {
        let mut chunk = support::chunk(&space);
        chunk.context.identity.model_role = "online".into();
        chunk.context.identity.learner_update_revision = revision;
        chunk.context.counters.updates = revision;
        let id = format!("episode-{revision}");
        chunk.context.trajectory = Some(id.clone());
        if let Record::Trajectory(record) = &mut chunk.record {
            record.id = id;
            record.evaluation.continuation_policy = policy.into();
        }
        writer.append(&chunk).unwrap();
    }
    writer.finish().unwrap();
    let mut h = ui_support::Harness::new(&temp, eframe::egui::vec2(1440., 900.));
    h.app.set_learning_expanded(false);
    h.app.select_revision(63).unwrap();
    h.app
        .select_input(SelectedInput::TrajectoryStep {
            sequence: 1,
            step: 1,
        })
        .unwrap();
    h.app.select_view(View::Dynamics);
    let deadline = Instant::now() + Duration::from_secs(10);
    while !h.app.dynamics_ready() {
        h.frame(vec![]);
        assert!(
            Instant::now() < deadline,
            "explicit generic fallback must settle: {:?}",
            h.text
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let (key, episode) = h.app.dynamics_episode().unwrap();
    assert_eq!(key.sequence, 1);
    assert_eq!(episode.numeric_coordinates(), [0, 1]);
    assert_eq!(episode.context().identity.learner_update_revision, 63);
    assert_eq!(episode.step(1).unwrap().observation, [1., 0., 11.]);
    let episode_owner = episode as *const _ as usize;
    let counters = h.app.dynamics_counters();
    for _ in 0..8 {
        h.frame(vec![]);
        assert!(h.app.dynamics_ready(), "idle repaint must retain readiness");
        assert!(
            !h.has("Loading"),
            "Ready exact generic episodes must not show false transport work: {:?}",
            h.text
        );
        assert_eq!(h.app.dynamics_counters(), counters);
        let (key, episode) = h.app.dynamics_episode().unwrap();
        assert_eq!(key.sequence, 1);
        assert_eq!(episode as *const _ as usize, episode_owner);
        assert_eq!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep {
                sequence: 1,
                step: 1,
            }
        );
    }
}

#[cfg(feature = "screenshots")]
#[test]
fn explicit_missing_episode_is_preserved_instead_of_replaced_by_another_catalogue_entry() {
    use rl::diagnostics::storage::*;
    use rl_viewer::{data::SelectedInput, ui::View};
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    let space = support::bounded(1);
    let mut manifest = support::manifest(space.clone());
    manifest.observation_dim = 1;
    manifest.environment_descriptor = None;
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    for id in ["original-A", "unrelated-B"] {
        let mut c = support::chunk(&space);
        c.context.identity.model_role = "online".into();
        c.context.trajectory = Some(id.into());
        c.arrays.insert(
            "obs".into(),
            Array::f32(vec![5, 1], vec![0., 1., 1., 2., 0.]).unwrap(),
        );
        if let Record::Trajectory(t) = &mut c.record {
            t.id = id.into();
        }
        writer.append(&c).unwrap();
    }
    writer.finish().unwrap();
    let run = RunIndex::open(&path, ReadLimits::default()).unwrap();
    let mut h = ui_support::Harness::new(&temp, eframe::egui::vec2(1440., 900.));
    h.app.select_view(View::Dynamics);
    h.app.select_capture_trajectory("0", "2").unwrap();
    std::fs::remove_file(path.join(&run.recording().entries()[0].file)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        h.frame(vec![]);
        if !h.app.dynamics_pending() {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        matches!(
            h.app.selection().input,
            SelectedInput::TrajectoryStep {
                sequence: 0,
                step: 2
            }
        ),
        "an explicit missing A must not become B"
    );
    assert!(h.has("Selected episode unavailable"));
    assert_eq!(
        h.app.dynamics_counters().source_reads,
        0,
        "no unrelated B episode preparation"
    );
}
