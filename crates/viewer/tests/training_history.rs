use rl::{
    Result,
    core::action::ActionSpace,
    diagnostics::{
        reference::{
            ConfigValue, ReferenceContext, ReferenceHorizon, ReferenceObjective,
            attach_manifest_context,
        },
        schema::{
            ActionCondition, Array, Chunk, Conditioning, EmpiricalRecord, EvaluationMetadata,
            MetricRecord, Record, RecordContext, RunManifest,
        },
        storage::{ReadLimits, Recorder, StorageConfig},
        value::ValueConventions,
    },
};
use rl_viewer::{
    data::RunIndex,
    learning::{LearningWorker, TrainingClock, TrainingFrame},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

fn recording(
    root: &Path,
    id: &str,
    seed: u64,
    protocol: &str,
    points: &[(u64, f64, bool)],
) -> Result<RunIndex> {
    let (path, mut writer) = open_writer(root, id, seed, protocol)?;
    append_points(&mut writer, id, points)?;
    writer.finish()?;
    RunIndex::open(path, ReadLimits::default())
}

fn open_writer(root: &Path, id: &str, seed: u64, protocol: &str) -> Result<(PathBuf, Recorder)> {
    let space = ActionSpace::Discrete { count: 2 };
    let mut manifest = RunManifest::new(id, "training-fixture", "example", 1, space.clone(), 0.5)?;
    manifest.environment = "bounded-test".into();
    let scientific = ReferenceContext {
        environment: manifest.environment.clone(),
        observation_dim: 1,
        typed_observation_space: None,
        action_space: space,
        conventions: ValueConventions {
            dynamics: "test dynamics".into(),
            observation: "test observation".into(),
            action: "discrete".into(),
            reward_units: "test reward".into(),
            reward_convention: manifest.reward_convention.clone(),
            discount: 0.5,
            horizon: "environment terminal".into(),
            terminal_treatment: "zero tail".into(),
            action_constraints: "all actions legal".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Continuing,
        environment_config: BTreeMap::from([("fixture".into(), ConfigValue::Integer(1))]),
        markov_observation: true,
        decision_time_coordinate: None,
    };
    attach_manifest_context(&mut manifest, &scientific)?;
    manifest
        .metadata
        .insert("training.seed".into(), seed.to_string());
    manifest
        .metadata
        .insert("learner.configuration".into(), "{}".into());
    manifest
        .metadata
        .insert("evaluation.scenario_protocol".into(), protocol.into());
    let path = root.join(id);
    let writer = Recorder::create(&path, manifest, StorageConfig::default())?;
    Ok((path, writer))
}

fn append_points(writer: &mut Recorder, id: &str, points: &[(u64, f64, bool)]) -> Result<()> {
    for &(x, mean, complete) in points {
        let mut context = RecordContext::new(id, x);
        context.environment_step = Some(x);
        context.counters.env_steps = x;
        context.counters.additional_physical_steps = x;
        context.wall_elapsed_seconds = Some(x as f64 / 2.);
        writer.append(&Chunk {
            context: context.clone(),
            record: Record::Empirical(EmpiricalRecord {
                returns: "returns".into(),
                seeds: "seeds".into(),
                probe_row: None,
                initial_action: None,
                conditioning_observation: None,
                conditioning_mask: None,
                simulator_state: None,
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: format!("{id}@{x}"),
                    stochastic_policy: true,
                    discount: 0.5,
                    independent_future_streams: true,
                    seed_methodology: "fixture".into(),
                    truncation_handling: "complete true terminal".into(),
                    complete_returns: complete,
                },
            }),
            arrays: BTreeMap::from([
                (
                    "returns".into(),
                    Array::f64(vec![2], vec![mean - 1., mean + 1.])?,
                ),
                ("seeds".into(), Array::u64(vec![2], vec![33, 44])?),
            ]),
        })?;
        writer.append(&Chunk {
            context,
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::from([("custom.loss".into(), mean)]),
                unavailable: BTreeMap::from([("custom.sparse".into(), "not measured".into())]),
            }),
            arrays: BTreeMap::new(),
        })?;
    }
    Ok(())
}

fn frame(worker: &mut LearningWorker) -> Arc<TrainingFrame> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        worker.poll();
        if let Some(frame) = worker.training_frame() {
            return frame.clone();
        }
        assert!(worker.error().is_none(), "{:?}", worker.error());
        assert!(Instant::now() < deadline, "Training worker timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn training_recovers_recorded_clocks_metrics_and_environmental_dispersion() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let run = recording(
        directory.path(),
        "first",
        1,
        "fixture reset",
        &[(4, 5., true), (8, 20., false)],
    )?;
    let mut worker = LearningWorker::new(|| {})?;
    assert!(worker.request_training(&run, None, &[])?);
    let actual = frame(&mut worker);
    let [first, incomplete] = actual.primary.evaluations.as_slice() else {
        panic!("expected both observed checkpoints")
    };
    assert_eq!(first.clocks.value(TrainingClock::Revision), Some(4.));
    assert_eq!(
        first.clocks.value(TrainingClock::EnvironmentDecisions),
        Some(4.)
    );
    assert_eq!(first.clocks.value(TrainingClock::PhysicalTicks), Some(8.));
    assert_eq!(first.clocks.value(TrainingClock::WallSeconds), Some(2.));
    let summary = first.summary.unwrap();
    assert_eq!(
        (
            summary.count,
            summary.mean,
            summary.sample_variance,
            summary.mean_standard_error
        ),
        (2, 5., Some(2.), Some(1.))
    );
    assert_eq!(first.extrema, Some([4., 6.]));
    assert!(
        incomplete.summary.is_none()
            && incomplete.reason.as_deref().unwrap().contains("incomplete")
    );
    let metrics = &actual.primary.metrics;
    let loss = metrics
        .iter()
        .find(|(key, _)| key.name == "custom.loss")
        .unwrap()
        .1;
    assert_eq!(
        loss.iter().map(|p| p.value).collect::<Vec<_>>(),
        [Some(5.), Some(20.)]
    );
    let sparse = metrics
        .iter()
        .find(|(key, _)| key.name == "custom.sparse")
        .unwrap()
        .1;
    assert!(
        sparse
            .iter()
            .all(|p| p.value.is_none() && p.reason.as_deref() == Some("not measured"))
    );
    assert!(!worker.request_training(&run, None, &[])?);
    assert!(worker.frame().is_none());
    Ok(())
}

#[test]
fn independent_cohort_aligns_only_common_support_and_reports_between_run_se() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let a = recording(
        directory.path(),
        "a",
        1,
        "fixture reset",
        &[(0, 10., true), (10, 20., true), (20, 30., true)],
    )?;
    let b = recording(
        directory.path(),
        "b",
        2,
        "fixture reset",
        &[(5, 100., true), (15, 200., true)],
    )?;
    let mut worker = LearningWorker::new(|| {})?;
    assert!(worker.request_training(&a, Some((&b, "online")), &[b.root().as_ref().clone()])?);
    let trained = frame(&mut worker);
    assert_eq!(trained.reference.as_ref().unwrap().run_id, "b");
    let cohort = trained.cohort.as_ref().unwrap();
    assert_eq!(cohort.members, ["a", "b"]);
    let points = cohort.points(TrainingClock::EnvironmentDecisions);
    assert_eq!(
        points
            .iter()
            .map(|p| (p.x, p.mean, p.standard_error, p.runs))
            .collect::<Vec<_>>(),
        [
            (5., 57.5, Some(42.5), 2),
            (10., 85., Some(65.), 2),
            (15., 112.5, Some(87.5), 2),
        ]
    );
    assert_eq!(
        trained.primary.evaluations[0]
            .summary
            .unwrap()
            .mean_standard_error,
        Some(1.)
    );

    let duplicate = recording(
        directory.path(),
        "same-seed",
        1,
        "fixture reset",
        &[(5, 0., true)],
    )?;
    assert!(worker.request_training(&a, None, &[duplicate.root().as_ref().clone()])?);
    assert!(
        worker.training_frame().is_none(),
        "previous comparison must retire immediately"
    );
    let duplicate = frame(&mut worker);
    assert!(duplicate.cohort.is_none());
    assert!(
        duplicate
            .cohort_unavailable
            .as_deref()
            .unwrap()
            .contains("distinct")
    );
    assert_eq!(duplicate.primary.run_id, "a");
    Ok(())
}

#[test]
fn training_does_not_interpolate_across_incomplete_evaluation_or_mix_reset_protocols() -> Result<()>
{
    let directory = tempfile::tempdir()?;
    let a = recording(
        directory.path(),
        "gaps",
        1,
        "fixture reset",
        &[(0, 0., true), (10, 7., false), (20, 20., true)],
    )?;
    let b = recording(
        directory.path(),
        "middle",
        2,
        "fixture reset",
        &[
            (0, 50., true),
            (5, 100., true),
            (15, 200., true),
            (20, 250., true),
        ],
    )?;
    let other = recording(
        directory.path(),
        "different-law",
        3,
        "different reset law",
        &[(5, 0., true)],
    )?;
    let mut worker = LearningWorker::new(|| {})?;
    assert!(worker.request_training(&a, Some((&other, "online")), &[b.root().as_ref().clone()])?);
    let actual = frame(&mut worker);
    assert!(actual.reference.is_none());
    assert!(
        actual
            .reference_unavailable
            .as_deref()
            .unwrap()
            .contains("initial-scenario protocol")
    );
    let points = actual
        .cohort
        .as_ref()
        .unwrap()
        .points(TrainingClock::EnvironmentDecisions);
    assert_eq!(
        points
            .iter()
            .map(|point| (point.x, point.starts_segment))
            .collect::<Vec<_>>(),
        [(0., true), (20., true)]
    );
    Ok(())
}

#[test]
fn live_peer_commit_refreshes_an_unchanged_primary_without_rereading_idle_payloads() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let primary = recording(
        directory.path(),
        "static-primary",
        1,
        "fixture reset",
        &[(0, 0., true), (10, 10., true), (20, 20., true)],
    )?;
    let (peer, mut writer) = open_writer(directory.path(), "live-peer", 2, "fixture reset")?;
    append_points(&mut writer, "live-peer", &[(0, 10., true)])?;
    let peers = [peer];
    let mut worker = LearningWorker::new(|| {})?;
    assert!(worker.request_training(&primary, None, &peers)?);
    let initial = frame(&mut worker);
    assert_eq!(
        initial
            .cohort
            .as_ref()
            .unwrap()
            .points(TrainingClock::Revision)
            .len(),
        1
    );

    // A due check on the unchanged live index has no payload work and keeps
    // the same already published frame.
    let reads = worker.metrics().reads;
    std::thread::sleep(initial.peer_recheck_after().unwrap() + Duration::from_millis(5));
    assert!(worker.request_training(&primary, None, &peers)?);
    let deadline = Instant::now() + Duration::from_secs(5);
    while worker.pending() {
        worker.poll();
        assert!(Instant::now() < deadline, "idle peer check timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(Arc::ptr_eq(worker.training_frame().unwrap(), &initial));
    assert_eq!(worker.metrics().reads, reads);

    append_points(&mut writer, "live-peer", &[(15, 25., true)])?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        worker.poll();
        worker.request_training(&primary, None, &peers)?;
        if let Some(updated) = worker
            .training_frame()
            .filter(|updated| !Arc::ptr_eq(updated, &initial))
        {
            let points = updated
                .cohort
                .as_ref()
                .unwrap()
                .points(TrainingClock::Revision);
            assert_eq!(
                points
                    .iter()
                    .map(|point| (point.x, point.mean))
                    .collect::<Vec<_>>(),
                [(0., 5.), (10., 15.), (15., 20.)]
            );
            assert!(updated.peer_recheck_after().is_some());
            break;
        }
        assert!(worker.error().is_none(), "{:?}", worker.error());
        assert!(
            Instant::now() < deadline,
            "peer extension did not refresh the static primary cohort"
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    writer.finish()?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        worker.poll();
        worker.request_training(&primary, None, &peers)?;
        if worker
            .training_frame()
            .is_some_and(|frame| frame.peer_recheck_after().is_none())
        {
            break;
        }
        assert!(worker.error().is_none(), "{:?}", worker.error());
        assert!(Instant::now() < deadline, "completed peer stayed watched");
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(())
}
