#![cfg(feature = "diagnostics")]

use rl::{
    Result,
    algorithms::{fqi, lspi},
    approx::{
        FeatureMap, FeatureSpec, FittedQ, ReturnWeightedRetrieval, StatisticalActor,
        batch::{BatchObjective, BatchOptions, DecisionSpan, FactualBatch},
        regression::{DiagonalPenalty, LinearModel, WeightedRidge},
    },
    core::{
        action::{Action, ActionSpace, FiniteActionKey},
        actor::ActMode,
        env::{EpisodeStart, Observation, PublicHistory},
        structured::FeatureIdentity,
        temporal::{BehaviorMetadata, TemporalSink, TemporalSpaces, TemporalTransition},
    },
    data::temporal::EpisodeStore,
    diagnostics::{
        models::ModelLimits,
        probes::ProbeSet,
        schema::{ProbePopulation, Record, RecordContext},
        session::{CaptureRequest, DiagnosticLearner},
        statistical::{
            FittedQDiagnostics, RetrievalDiagnostics, RetrievalFeatureMap, restore_fitted_q,
            restore_fitted_q_with_features, restore_retrieval,
        },
        typed::{DiagnosticInput, FiniteValueKind, TypedPayload},
        value,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct ActionFeatures(FeatureSpec);

impl FeatureMap for ActionFeatures {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, _: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        Ok(actions
            .iter()
            .map(|action| match action {
                Action::Discrete(0) => vec![1.0, 0.0],
                Action::Discrete(1) => vec![0.0, 1.0],
                _ => unreachable!("fixture has two actions"),
            })
            .collect())
    }
}

fn spec(name: &str, terms: &[&str]) -> FeatureSpec {
    FeatureSpec {
        namespace: name.into(),
        version: 1,
        terms: terms
            .iter()
            .map(|term| FeatureIdentity::new(*term, *term))
            .collect(),
        configuration: Vec::new(),
        representation: None,
    }
}

fn action_space() -> ActionSpace {
    ActionSpace::Discrete { count: 2 }
}

fn store() -> Result<EpisodeStore> {
    let mut store = EpisodeStore::new(8, 1, action_space())?;
    for (episode_id, action, reward) in [
        (0, Action::Discrete(0), -1.0),
        (1, Action::Discrete(1), 3.0),
    ] {
        let observation = Observation::new(vec![0.0]);
        store.begin_episode(episode_id, &EpisodeStart::fresh(observation.clone()))?;
        store.push(&TemporalTransition {
            episode_id,
            timestep: 0,
            observation,
            action,
            reward: reward.into(),
            next_observation: Observation::new(vec![1.0]),
            terminated: true,
            truncated: false,
            behavior: BehaviorMetadata {
                revision: 0,
                log_prob: None,
            },
        })?;
    }
    Ok(store)
}

fn batch_options() -> BatchOptions {
    BatchOptions {
        objective: BatchObjective::Physical { gamma: 0.9 },
        span: DecisionSpan::Single,
    }
}

fn probes(observation: Observation) -> Result<Vec<ProbeSet>> {
    Ok(vec![ProbeSet::new(
        "fixture",
        ProbePopulation::FixedReference,
        "Fitted statistical adapter fixture",
        vec![observation],
        None,
        action_space(),
    )?])
}

fn capture<'a>(context: &'a RecordContext, probes: &'a [ProbeSet]) -> CaptureRequest<'a> {
    CaptureRequest {
        context,
        probes,
        predictions: true,
        representations: false,
        snapshots: true,
        max_activation_values: 0,
        max_chunk_bytes: 64 * 1024,
        conventions: None,
        policy_seed: 1,
        evaluation_mode: ActMode::Deterministic,
    }
}

#[test]
fn fqi_records_batch_fit_evidence_and_round_trips_the_frozen_feature_map() -> Result<()> {
    let store = store()?;
    let features = ActionFeatures(spec("fixture", &["left", "right"]));
    let batch = FactualBatch::from_store(&store, &features, batch_options())?;
    let config = fqi::FqiConfig::default();
    let fit = fqi::fit(
        &batch,
        &WeightedRidge {
            penalty: DiagonalPenalty::Uniform(0.0),
            ..Default::default()
        },
        config,
        fqi::Initial::Zero,
    )?;
    let actor = StatisticalActor::new(features.clone(), fit.model, 7, 0.2)?;
    let coverage = batch.coverage()?;
    let diagnostic = FittedQDiagnostics::fqi(&actor, &fit.sweeps, &coverage)?
        .with_fqi_config(config)
        .with_portable_model();
    let mut context = RecordContext::new("fixture", 7);
    context.identity.checkpoint_id = Some("revision-7".into());
    let chunks =
        diagnostic.diagnostic_records(&capture(&context, &probes(Observation::new(vec![0.0]))?))?;
    let critic = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::Critic(_)))
        .unwrap();
    let values = critic.arrays["values"].as_f64()?;
    assert!((values[0] + 1.0).abs() < 1e-12 && (values[1] - 3.0).abs() < 1e-12);
    let declaration = value::declaration(critic)?.unwrap();
    assert_eq!(
        declaration.critics["action_value"]
            .continuation
            .as_ref()
            .unwrap()
            .identity,
        "optimality"
    );
    let metric = diagnostic.diagnostic_metrics(&context)?.unwrap();
    let Record::Metrics(metrics) = &metric.record else {
        panic!("missing metrics")
    };
    assert_eq!(metrics.values["fit.fqi.targets.mean"], 1.0);
    assert_eq!(metrics.values["fit.fqi.sweeps"], 1.0);
    assert!(metrics.values["fit.fqi.regression.weighted_residual_norm"] < 1e-12);
    assert_eq!(metrics.values["dataset.rows"], 2.0);
    let snapshot = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::ModelSnapshot(_)))
        .unwrap();
    let frozen = restore_fitted_q::<ActionFeatures, LinearModel>(snapshot, ModelLimits::default())?;
    assert_eq!(
        frozen.greedy(&PublicHistory::new(Observation::new(vec![0.0])))?,
        Action::Discrete(1)
    );

    let fitted_only = FittedQDiagnostics::fqi(&actor, &fit.sweeps, &coverage)?.with_fitted_model();
    let chunks = fitted_only.diagnostic_records(&capture(&context, &[]))?;
    let snapshot = &chunks[0];
    let limits = ModelLimits::default();
    assert!(restore_fitted_q::<ActionFeatures, LinearModel>(snapshot, limits).is_err());
    let different_map = ActionFeatures(spec("different", &["left", "right"]));
    assert!(
        restore_fitted_q_with_features::<_, LinearModel>(snapshot, different_map, limits).is_err()
    );
    let frozen = restore_fitted_q_with_features::<_, LinearModel>(snapshot, features, limits)?;
    assert_eq!(
        frozen.greedy(&PublicHistory::new(Observation::new(vec![0.0])))?,
        Action::Discrete(1)
    );
    Ok(())
}

struct HistoryFeature(FeatureSpec);

impl FeatureMap for HistoryFeature {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let previous_reward = history
            .inputs()
            .last()
            .and_then(|input| input.previous)
            .map_or(0.0, |step| f64::from(step.reward));
        Ok(vec![vec![previous_reward]; actions.len()])
    }
}

#[test]
fn typed_fitted_probes_predict_from_the_full_history_on_the_legal_support() -> Result<()> {
    let store = store()?;
    let feature = HistoryFeature(spec("previous reward", &["previous reward"]));
    let batch = FactualBatch::from_store(&store, &feature, batch_options())?;
    let coverage = batch.coverage()?;
    let spaces = TemporalSpaces {
        observations: 1.into(),
        actions: action_space(),
    };
    let fitted = FittedQ::new(
        feature.0.clone(),
        spaces,
        coverage.objective,
        LinearModel::new(vec![2.0])?,
    )?;
    let actor = StatisticalActor::new(feature, fitted, 8, 0.0)?;
    let learner = FittedQDiagnostics::fqi(&actor, &[] as &[fqi::Sweep<()>], &coverage)?;
    let initial = PublicHistory::new(Observation::new(vec![0.0]));
    let observation = Observation::dense(vec![1.0], Some(vec![false, true]));
    let set = ProbeSet::new_in(
        "histories",
        ProbePopulation::ImportedHistory,
        "One terminal feature input follows distinct factual rewards",
        vec![
            DiagnosticInput::History(initial.append(
                Action::Discrete(0),
                1.0,
                observation.clone(),
            )?),
            DiagnosticInput::History(initial.append(Action::Discrete(1), 3.0, observation)?),
        ],
        None,
        1.into(),
        action_space(),
    )?;
    let context = RecordContext::new("fixture", 8);
    let chunks = learner.diagnostic_records(&capture(&context, &[set]))?;
    let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunks[0])? else {
        panic!("missing predictions");
    };
    for (row, value) in predictions.rows.iter().zip([2.0, 6.0]) {
        assert_eq!(row.selected, Some(FiniteActionKey::Discrete(1)));
        assert_eq!(row.candidates.len(), 1);
        assert_eq!(row.candidates[0].action, FiniteActionKey::Discrete(1));
        assert_eq!(row.candidates[0].value, Some(value));
    }
    Ok(())
}

#[test]
fn lspi_preserves_numerical_diagnostics_without_claiming_a_global_continuation() -> Result<()> {
    let store = store()?;
    let features = ActionFeatures(spec("fixture", &["left", "right"]));
    let batch = FactualBatch::from_store(&store, &features, batch_options())?;
    let fit = lspi::fit(&batch, &lspi::LspiConfig::default(), None)?;
    let actor = StatisticalActor::new(features.clone(), fit.model, 2, 0.0)?;
    let coverage = batch.coverage()?;
    let diagnostic = FittedQDiagnostics::lspi(&actor, &fit.iterations, &coverage, fit.stable)?;
    let context = RecordContext::new("fixture", 2);
    let chunks =
        diagnostic.diagnostic_records(&capture(&context, &probes(Observation::new(vec![0.0]))?))?;
    let critic = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::Critic(_)))
        .unwrap();
    assert!(
        value::declaration(critic)?.unwrap().critics["action_value"]
            .continuation
            .is_none()
    );
    let metric = diagnostic.diagnostic_metrics(&context)?.unwrap();
    let Record::Metrics(metrics) = &metric.record else {
        panic!("missing metrics")
    };
    assert!(metrics.values["fit.lspi.moment_system.residual_norm"] < 1e-12);
    assert_eq!(metrics.values["fit.lspi.stable_on_sampled_successors"], 1.0);
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct RetrievalFeatures {
    state: FeatureSpec,
    action: FeatureSpec,
    state_scale: f64,
    action_scale: f64,
}

impl RetrievalFeatureMap for RetrievalFeatures {
    fn state_spec(&self) -> &FeatureSpec {
        &self.state
    }
    fn action_spec(&self) -> &FeatureSpec {
        &self.action
    }
    fn encode_state(&self, history: &PublicHistory) -> Result<Vec<f64>> {
        Ok(vec![
            f64::from(history.observation().dense_values()[0]) * self.state_scale,
        ])
    }
    fn encode_actions(&self, _: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        Ok(actions
            .iter()
            .map(|action| match action {
                Action::Discrete(index) => vec![*index as f64 * self.action_scale],
                _ => unreachable!("fixture has two actions"),
            })
            .collect())
    }
}

#[test]
fn retrieval_round_trip_reuses_recorded_population_and_feature_scaling() -> Result<()> {
    let mut bank = ReturnWeightedRetrieval::new(2, 1)?;
    bank.observe(&[(vec![0.0], vec![0.0]), (vec![10.0], vec![2.0])], 2.0)?;
    let features = RetrievalFeatures {
        state: spec("state", &["scaled position"]),
        action: spec("action", &["scaled choice"]),
        state_scale: 10.0,
        action_scale: 2.0,
    };
    let spaces = TemporalSpaces {
        observations: 1.into(),
        actions: action_space(),
    };
    let diagnostic = RetrievalDiagnostics::new(&bank, &features, spaces, 3)?.with_portable_model();
    let mut context = RecordContext::new("fixture", 3);
    context.identity.checkpoint_id = Some("revision-3".into());
    let observation = Observation::new(vec![0.9]);
    let chunks =
        diagnostic.diagnostic_records(&capture(&context, &probes(observation.clone())?))?;
    assert!(
        !chunks
            .iter()
            .any(|chunk| matches!(chunk.record, Record::Critic(_)))
    );
    let policy = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::Policy(_)))
        .unwrap();
    let expected = 1.0 / (1.0 + (-4.0_f64).exp());
    assert!((policy.arrays["probabilities"].as_f64()?[1] - expected).abs() < 1e-12);
    let snapshot = chunks
        .iter()
        .find(|chunk| matches!(chunk.record, Record::ModelSnapshot(_)))
        .unwrap();
    let restored = restore_retrieval::<RetrievalFeatures>(snapshot, ModelLimits::default())?;
    let probability = restored.probabilities(&PublicHistory::new(observation.clone()))?[1].1;
    assert!((probability - expected).abs() < 1e-12);

    let set = ProbeSet::new_in(
        "typed-retrieval",
        ProbePopulation::FixedReference,
        "A scaled retrieval coordinate",
        vec![observation.into()],
        None,
        1.into(),
        action_space(),
    )?;
    let sets = [set];
    let deterministic = capture(&context, &sets);
    let chunks = diagnostic.diagnostic_records(&deterministic)?;
    let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunks[0])? else {
        panic!("missing typed retrieval predictions");
    };
    assert_eq!(
        predictions.value_kind,
        Some(FiniteValueKind::PreferenceScore)
    );
    let chosen = &predictions.rows[0].candidates[1];
    assert_eq!(chosen.probability, Some(1.0));
    assert!((chosen.value.unwrap() - expected).abs() < 1e-12);

    let sampled = CaptureRequest {
        evaluation_mode: ActMode::Sample,
        ..deterministic
    };
    let chunks = diagnostic.diagnostic_records(&sampled)?;
    let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunks[0])? else {
        panic!("missing typed retrieval predictions");
    };
    assert!((predictions.rows[0].candidates[1].probability.unwrap() - expected).abs() < 1e-12);
    Ok(())
}
