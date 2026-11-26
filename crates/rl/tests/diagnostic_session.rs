#![cfg(feature = "diagnostics")]

use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::{Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step},
        learner::UpdateReport,
    },
    diagnostics::{
        probes::ProbeSet,
        schedule::Interval,
        schema::{Chunk, ProbePopulation, Record, RecordKind},
        session::{
            CaptureRequest, DiagnosticConfig, DiagnosticLearner, DiagnosticSession, LearnerInfo,
        },
        storage::{ReadLimits, Recording},
    },
    train::{
        counters::RunCounters,
        observer::{UpdateBoundary, UpdateObserver},
    },
};
use std::cell::Cell;

struct TestEnv(ActionSpace);

struct DiscreteEnv(ActionSpace);

impl Env for DiscreteEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Discrete(rl::core::env::DiscreteObservationSpace {
            symbols: (1 << 55) + 2,
        })
    }
    fn action_space(&self) -> &ActionSpace {
        &self.0
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        Ok(EpisodeStart::fresh(Observation::discrete(1 << 55, None)))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        assert_eq!(action, &Action::Discrete(0));
        Ok(Step {
            observation: Observation::discrete((1 << 55) + 1, Some(vec![false])),
            reward: 1.0.into(),
            terminated: true,
            truncated: false,
        })
    }
}

#[test]
fn discrete_session_retains_lossless_evaluation_episode() {
    use rl::diagnostics::typed::TypedPayload;

    let env = DiscreteEnv(ActionSpace::Discrete { count: 1 });
    let learner = TestLearner {
        revision: 0,
        space: env.0.clone(),
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("discrete");
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .evaluation_factory(|_| Ok(DiscreteEnv(ActionSpace::Discrete { count: 1 })))
        .config(DiagnosticConfig::light())
        .build(&learner)
        .unwrap();
    session.finish(&learner).unwrap();

    let recording = Recording::open(path, ReadLimits::default()).unwrap();
    let index = recording
        .entries()
        .iter()
        .position(|entry| entry.kind == RecordKind::TypedEpisode)
        .unwrap();
    let TypedPayload::Episode(episode) =
        TypedPayload::read(&recording.load(index).unwrap()).unwrap()
    else {
        panic!("expected a typed episode");
    };
    assert_eq!(
        episode.observations[0].as_discrete().unwrap().symbol,
        1 << 55
    );
    assert_eq!(
        episode.observations[1].as_discrete().unwrap().symbol,
        (1 << 55) + 1
    );
    assert!(episode.steps[0].terminated);
}

impl Env for TestEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.0
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, _: &Action) -> Result<Step> {
        Ok(Step {
            observation: Observation::new(vec![1.]),
            reward: 1.0.into(),
            terminated: true,
            truncated: false,
        })
    }
}

struct TestLearner {
    revision: u64,
    space: ActionSpace,
}

impl Actor for TestLearner {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        actions.resize(obs.len(), Action::Discrete(0));
        Ok(())
    }
}

impl DiagnosticLearner for TestLearner {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "test",
            model: "test actor",
            configuration: serde_json::json!({}),
            discount: Some(0.95),
        })
    }
    fn diagnostic_revision(&self) -> u64 {
        self.revision
    }
    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        struct Greedy(ActionSpace);
        impl Actor for Greedy {
            fn action_space(&self) -> &ActionSpace {
                &self.0
            }
            fn act_into(
                &self,
                obs: ObservationBatch<'_>,
                _: ActMode,
                _: &mut ChaCha8Rng,
                out: &mut Vec<Action>,
            ) -> Result<()> {
                out.clear();
                out.resize(obs.len(), Action::Discrete(0));
                Ok(())
            }
        }
        Ok(Box::new(Greedy(self.space.clone())))
    }
    fn diagnostic_records(&self, _: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        Ok(Vec::new())
    }
}

#[test]
fn uses_frozen_revision_with_restored_or_fresh_trainer_counters() -> Result<()> {
    for trainer_origin in [0, 11] {
        let tmp = tempfile::tempdir()?;
        let dir = tmp.path().join("diagnostics");
        let space = ActionSpace::Discrete { count: 2 };
        let env = TestEnv(space.clone());
        let mut learner = TestLearner {
            revision: 11,
            space,
        };
        let mut config = DiagnosticConfig::light();
        config.schedule.metrics = Some(Interval {
            first: 12,
            every: 4,
        });
        config.schedule.evaluation = Some(Interval {
            first: 12,
            every: 4,
        });
        let mut session = DiagnosticSession::builder(&dir)
            .environment(&env)
            .evaluation_factory(|_| Ok(TestEnv(ActionSpace::Discrete { count: 2 })))
            .config(config)
            .build(&learner)?;

        for (revision, updates, n) in [(14, 3, 3), (15, 4, 1), (16, 5, 1), (17, 6, 1)] {
            learner.revision = revision;
            session.after_update(
                &learner,
                UpdateBoundary {
                    counters: &RunCounters {
                        updates: trainer_origin + updates,
                        ..Default::default()
                    },
                    learner_call: updates,
                    collection_cycle: updates,
                    report: &UpdateReport {
                        updates: n,
                        ..Default::default()
                    },
                    collection_seconds: 0.0,
                    learning_seconds: 0.0,
                },
            )?;
        }
        session.finish(&learner)?;

        let recording = Recording::open(&dir, ReadLimits::default())?;
        assert!(recording.complete());
        assert_eq!(recording.manifest().starting_revision, 11);
        let eval_revisions = recording
            .entries()
            .iter()
            .filter(|entry| entry.kind == RecordKind::Empirical)
            .map(|entry| entry.revision)
            .collect::<Vec<_>>();
        assert_eq!(eval_revisions, [11, 14, 16, 17]);
        let elapsed = recording
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind == RecordKind::Empirical)
            .map(|(index, _)| {
                Ok(recording
                    .load(index)?
                    .context
                    .wall_elapsed_seconds
                    .expect("session records capture time"))
            })
            .collect::<Result<Vec<_>>>()?;
        assert!(
            elapsed
                .iter()
                .all(|seconds| seconds.is_finite() && *seconds >= 0.)
        );
        assert!(elapsed.windows(2).all(|times| times[0] <= times[1]));
        let metrics = recording
            .entries()
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind == RecordKind::Metrics)
            .map(|(i, _)| recording.load(i))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            metrics
                .iter()
                .map(|c| c.context.identity.learner_update_revision)
                .collect::<Vec<_>>(),
            [11, 14, 16, 17]
        );
        let Record::Metrics(final_metrics) = &metrics.last().unwrap().record else {
            unreachable!()
        };
        assert_eq!(final_metrics.values["updates.in_learner_call"], 1.0);
    }
    Ok(())
}

#[test]
fn resumed_session_waits_until_the_next_interval_when_initial_capture_is_disabled() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("resumed");
    let space = ActionSpace::Discrete { count: 2 };
    let env = TestEnv(space.clone());
    let mut learner = TestLearner {
        revision: 101,
        space: space.clone(),
    };
    let mut config = DiagnosticConfig::light();
    let interval = Some(Interval {
        first: 100,
        every: 100,
    });
    config.schedule.initial = false;
    config.schedule.final_checkpoint = false;
    config.schedule.metrics = interval;
    config.schedule.evaluation = interval;
    config.evaluation.seeds = vec![21];
    config.evaluation.trajectories = 0;
    let evaluations = Cell::new(0);
    let mut session = DiagnosticSession::builder(&path)
        .environment(&env)
        .evaluation_factory(|_| {
            evaluations.set(evaluations.get() + 1);
            Ok(TestEnv(space.clone()))
        })
        .config(config)
        .build(&learner)?;

    assert_eq!(evaluations.get(), 0);
    for (call, revision, trainer_updates, call_updates, expected_evaluations) in
        [(1, 102, 1, 1, 0), (2, 201, 100, 99, 1)]
    {
        learner.revision = revision;
        session.after_update(
            &learner,
            UpdateBoundary {
                counters: &RunCounters {
                    updates: trainer_updates,
                    ..Default::default()
                },
                learner_call: call,
                collection_cycle: call,
                report: &UpdateReport {
                    updates: call_updates,
                    ..Default::default()
                },
                collection_seconds: 0.,
                learning_seconds: 0.,
            },
        )?;
        assert_eq!(evaluations.get(), expected_evaluations);
        if expected_evaluations == 0 {
            session.flush()?;
            assert!(
                Recording::open(&path, ReadLimits::default())?
                    .entries()
                    .is_empty()
            );
        }
    }
    session.finish(&learner)?;

    let recording = Recording::open(&path, ReadLimits::default())?;
    assert!(recording.complete());
    assert_eq!(recording.manifest().starting_revision, 101);
    for kind in [RecordKind::Empirical, RecordKind::Metrics] {
        let revisions = recording
            .entries()
            .iter()
            .filter(|entry| entry.kind == kind)
            .map(|entry| entry.revision)
            .collect::<Vec<_>>();
        assert_eq!(revisions, [201]);
    }
    Ok(())
}

#[test]
fn discrete_mask_capture_budget_is_checked_before_creating_the_recording() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("oversized");
    let space = ActionSpace::Discrete { count: 100_000_000 };
    let env = TestEnv(space.clone());
    let learner = TestLearner {
        revision: 0,
        space: space.clone(),
    };
    let probes = ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "one unmasked live observation",
        vec![Observation::new(vec![0.])],
        None,
        space,
    )?;
    let mut config = DiagnosticConfig::standard();
    config.storage.max_chunk_bytes = 1024 * 1024;
    let result = DiagnosticSession::builder(&path)
        .environment(&env)
        .probe_set(probes)
        .config(config)
        .build(&learner);

    assert!(
        matches!(result, Err(rl::Error::Config(message)) if message.contains("capture budget"))
    );
    assert!(!path.exists());
    Ok(())
}
