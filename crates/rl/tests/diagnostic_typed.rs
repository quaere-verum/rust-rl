#![cfg(feature = "diagnostics")]

use rl::{
    core::{
        action::{Action, ActionSpace, FiniteActionKey},
        env::{DiscreteObservationSpace, Observation, ObservationSpace, PublicHistory},
    },
    diagnostics::{
        probes::ProbeSet,
        schema::{ProbePopulation, RecordContext, RunManifest},
        storage::{ReadLimits, Recorder, Recording, StorageConfig},
        typed::{
            DiagnosticInput, FiniteCandidate, FiniteDecision, FinitePredictions, FiniteValueKind,
            TypedPayload,
        },
    },
};

const SYMBOL: u64 = 1 << 55;

fn spaces() -> (ObservationSpace, ActionSpace) {
    (
        ObservationSpace::Discrete(DiscreteObservationSpace {
            symbols: SYMBOL + 2,
        }),
        ActionSpace::Discrete { count: 3 },
    )
}

fn fixture() -> ProbeSet {
    let (observations, actions) = spaces();
    let history = PublicHistory::new(Observation::discrete(SYMBOL, None))
        .append(
            Action::Discrete(1),
            4.0,
            Observation::discrete(SYMBOL + 1, Some(vec![true, false, true])),
        )
        .unwrap();
    ProbeSet::new_in(
        "history",
        ProbePopulation::ImportedHistory,
        "A fixed factual history",
        vec![DiagnosticInput::History(history)],
        None,
        observations,
        actions,
    )
    .unwrap()
    .with_row_ids(vec![SYMBOL + 9])
    .unwrap()
}

fn predictions() -> FinitePredictions {
    let candidate = |index, probability, value, updates| FiniteCandidate {
        action: FiniteActionKey::Discrete(index),
        probability,
        value,
        visits: None,
        updates,
    };
    FinitePredictions {
        value_kind: Some(FiniteValueKind::ActionValue),
        rows: vec![FiniteDecision {
            row_id: SYMBOL + 9,
            information_state: Some("history-key-1".into()),
            selected: Some(FiniteActionKey::Discrete(2)),
            candidates: vec![
                candidate(2, Some(1.), Some(3.), Some(1)),
                candidate(0, Some(0.), Some(3.), Some(0)),
            ],
        }],
    }
}

#[test]
fn portable_history_preserves_symbol_action_reward_and_sparse_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (observations, actions) = spaces();
    let manifest =
        RunManifest::new_in("run", "test", "tabular", observations, actions, 0.9).unwrap();
    let mut writer =
        Recorder::create(dir.path().join("run"), manifest, StorageConfig::default()).unwrap();
    let probes = fixture();
    let context = RecordContext::new("run", 7);
    writer
        .append(&probes.record(context.clone()).unwrap())
        .unwrap();
    writer
        .append(&predictions().record(context, &probes, 1024 * 1024).unwrap())
        .unwrap();
    writer.finish().unwrap();

    let run = Recording::open(dir.path().join("run"), ReadLimits::default()).unwrap();
    assert_eq!(run.manifest().observation_dim, 0);
    assert_eq!(run.manifest().observation_space(), spaces().0);
    let TypedPayload::Probes(probes) = TypedPayload::read(&run.load(0).unwrap()).unwrap() else {
        panic!("expected typed probes");
    };
    assert_eq!(probes.row_ids, [SYMBOL + 9]);
    let history = probes.inputs[0].public_history().unwrap();
    assert_eq!(history.timestep(), 1);
    assert_eq!(
        history.observation().as_discrete().unwrap().symbol,
        SYMBOL + 1
    );
    let inputs = history.inputs();
    assert_eq!(inputs[0].observation.as_discrete().unwrap().symbol, SYMBOL);
    let previous = inputs[1].previous.as_ref().unwrap();
    assert_eq!(previous.action, &Action::Discrete(1));
    assert_eq!(previous.factual.unwrap().reward.total(), 4.0);
    let TypedPayload::Predictions(frame) = TypedPayload::read(&run.load(1).unwrap()).unwrap()
    else {
        panic!("expected finite predictions");
    };
    assert_eq!(frame.rows, predictions().rows);
}

#[test]
fn finite_candidates_require_exact_legal_support_and_a_byte_budget() {
    let context = RecordContext::new("run", 7);
    let probes = fixture();
    assert!(predictions().record(context.clone(), &probes, 20).is_err());
    let mut invalid = predictions();
    invalid.rows[0].candidates[1].action = FiniteActionKey::Discrete(1);
    assert!(
        invalid
            .record(context.clone(), &probes, 1024 * 1024)
            .is_err()
    );
    invalid.rows[0].candidates[1].action = FiniteActionKey::Discrete(2);
    assert!(invalid.record(context, &probes, 1024 * 1024).is_err());
}

#[test]
fn typed_episode_admission_rejects_inconsistent_clock_and_provenance() -> rl::Result<()> {
    use rl::diagnostics::{
        evaluation::EvaluationHorizon,
        schema::{ActionCondition, Array, Chunk, Conditioning, EvaluationMetadata, Record},
        typed::{TypedEpisode, TypedEpisodeSource, TypedKind, TypedRecord, TypedStep},
    };
    use std::collections::BTreeMap;

    let directory = tempfile::tempdir()?;
    let (observations, actions) = spaces();
    let manifest = RunManifest::new_in("run", "test", "tabular", observations, actions, 0.9)?;
    let mut writer = Recorder::create(
        directory.path().join("run"),
        manifest,
        StorageConfig::default(),
    )?;
    let step = |terminated| TypedStep {
        action: Action::Discrete(0),
        reward: 1.,
        duration: 1,
        discounted_reward: 1.,
        terminated,
        truncated: false,
        factual_reward: None,
        behavior_revision: None,
    };
    let episode = TypedEpisode {
        source: TypedEpisodeSource::Evaluation {
            scenario_seed: 17,
            metadata: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "policy-4".into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "explicit scenarios".into(),
                truncation_handling: "complete terminal returns".into(),
                complete_returns: true,
            },
            horizon: EvaluationHorizon::FiniteDecisions(2),
        },
        observations: vec![Observation::discrete(SYMBOL, None); 3],
        steps: vec![step(false), step(true)],
    };
    let chunk = |episode| -> rl::Result<Chunk> {
        let bytes = serde_json::to_vec(&TypedPayload::Episode(episode))?;
        let mut context = RecordContext::new("run", 4);
        context.trajectory = Some("episode".into());
        Ok(Chunk {
            context,
            record: Record::Typed(TypedRecord {
                kind: TypedKind::Episode,
                payload: "episode".into(),
            }),
            arrays: BTreeMap::from([("episode".into(), Array::u8(vec![bytes.len()], bytes)?)]),
        })
    };
    writer.append(&chunk(episode.clone())?)?;

    let mut invalid = episode.clone();
    invalid.steps[0].duration = u64::MAX;
    assert!(writer.append(&chunk(invalid)?).is_err());

    for condition in 0..5 {
        let mut invalid = episode.clone();
        let TypedEpisodeSource::Evaluation {
            metadata, horizon, ..
        } = &mut invalid.source
        else {
            unreachable!()
        };
        match condition {
            0 => metadata.continuation_policy.clear(),
            1 => *horizon = EvaluationHorizon::FiniteDecisions(0),
            2 => *horizon = EvaluationHorizon::FiniteDecisions(1),
            3 => metadata.complete_returns = false,
            _ => metadata.discount = 0.8,
        }
        assert!(
            writer.append(&chunk(invalid)?).is_err(),
            "invalid case {condition}"
        );
    }

    let mut factual = episode;
    factual.source = TypedEpisodeSource::Factual {
        episode_id: 17,
        discount: 0.9,
    };
    for step in &mut factual.steps {
        step.factual_reward = Some(1_f32.into());
        step.behavior_revision = Some(4);
    }
    writer.append(&chunk(factual.clone())?)?;
    factual.steps[1].behavior_revision = Some(5);
    assert!(writer.append(&chunk(factual)?).is_err());
    Ok(())
}

#[test]
fn typed_probe_admission_caps_the_complete_history_payload() {
    let observation = Observation::new(vec![0.; 32_768]);
    let mut history = PublicHistory::new(observation.clone());
    for _ in 0..33 {
        history = history
            .append(Action::Discrete(0), 0., observation.clone())
            .unwrap();
    }
    assert!(
        ProbeSet::new_in(
            "bounded",
            ProbePopulation::ImportedHistory,
            "A large factual history",
            vec![DiagnosticInput::History(history)],
            None,
            ObservationSpace::Dense(32_768),
            ActionSpace::Discrete { count: 2 },
        )
        .is_err()
    );
}

#[test]
fn portable_prediction_keys_must_fit_the_manifest_action_space() -> rl::Result<()> {
    let directory = tempfile::tempdir()?;
    let (observations, _) = spaces();
    let manifest = RunManifest::new_in(
        "run",
        "test",
        "tabular",
        observations,
        ActionSpace::Discrete { count: 2 },
        0.9,
    )?;
    let mut writer = Recorder::create(
        directory.path().join("run"),
        manifest,
        StorageConfig::default(),
    )?;
    let chunk = predictions().record(RecordContext::new("run", 7), &fixture(), 1024 * 1024)?;
    assert!(writer.append(&chunk).is_err());
    Ok(())
}

#[test]
fn each_typed_trajectory_reports_its_own_completion() -> rl::Result<()> {
    use rl::diagnostics::{
        evaluation::{EvaluatedTrajectory, EvaluationHorizon, EvaluationResult},
        schema::{ActionCondition, Conditioning, EvaluationMetadata},
        typed::TypedEpisodeSource,
    };
    let result = EvaluationResult {
        returns: vec![1., 1.],
        seeds: vec![0, 1],
        trajectories: [true, false]
            .into_iter()
            .enumerate()
            .map(|(seed, terminated)| EvaluatedTrajectory {
                render: None,
                seed: seed as u64,
                observations: vec![Observation::discrete(SYMBOL, None); 2],
                actions: vec![Action::Discrete(0)],
                rewards: vec![1.],
                physical_durations: vec![1],
                discounted_step_rewards: vec![1.],
                terminated: vec![terminated],
                truncated: vec![!terminated],
                frames: Vec::new(),
            })
            .collect(),
        metadata: EvaluationMetadata {
            conditioning: Conditioning::InitialScenario,
            initial_action: ActionCondition::CurrentPolicy,
            continuation_policy: "policy-4".into(),
            stochastic_policy: false,
            discount: 0.9,
            independent_future_streams: true,
            seed_methodology: "explicit scenarios".into(),
            truncation_handling: "external truncation retained as incomplete".into(),
            complete_returns: false,
        },
        horizon: EvaluationHorizon::EnvironmentTermination,
        initial_action: None,
        conditioning_observation: None,
        simulator_state: None,
        elapsed_seconds: 0.,
        environment_steps: 2,
        physical_steps: 2,
    };
    for (index, complete) in [true, false].into_iter().enumerate() {
        let chunk =
            result.trajectory_record(index, RecordContext::new("run", 4), "typed", "test")?;
        let TypedPayload::Episode(episode) = TypedPayload::read(&chunk)? else {
            unreachable!()
        };
        let TypedEpisodeSource::Evaluation { metadata, .. } = episode.source else {
            unreachable!()
        };
        assert_eq!(metadata.complete_returns, complete);
    }
    assert!(!result.metadata.complete_returns);
    Ok(())
}

#[test]
fn oversized_typed_attachments_are_skipped_before_commit() -> rl::Result<()> {
    use rl::diagnostics::{schema::Array, storage::AppendOutcome, typed::MAX_TYPED_BYTES};

    let directory = tempfile::tempdir()?;
    let path = directory.path().join("run");
    let (observations, actions) = spaces();
    let manifest = RunManifest::new_in("run", "test", "tabular", observations, actions, 0.9)?;
    let mut writer = Recorder::create(&path, manifest, StorageConfig::default())?;
    let mut chunk = fixture().record(RecordContext::new("run", 7))?;
    chunk.arrays.insert(
        "unused".into(),
        Array::u8(vec![MAX_TYPED_BYTES], vec![0; MAX_TYPED_BYTES])?,
    );
    assert!(matches!(
        writer.append(&chunk)?,
        AppendOutcome::Skipped { .. }
    ));
    writer.finish()?;

    let recording = Recording::open(path, ReadLimits::default())?;
    assert!(recording.complete());
    assert!(recording.entries().is_empty());
    assert!(recording.issues().is_empty());
    Ok(())
}
