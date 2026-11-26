#![cfg(feature = "neural")]

use burn::tensor::Device;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        candidate_ddqn::CandidateDdqnConfig,
        candidate_mpo::{
            CandidateMpo, CandidateMpoActor, CandidateMpoBackup, CandidateMpoBehavior,
            CandidateMpoConfig,
        },
    },
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        env::{Observation, ObservationSpace},
        reward::Reward,
        stateful::{BehaviorPublisher, StatefulActor},
        structured::{
            ActionCandidate, EntityId, EntityRow, SpatialObservation, StructuredObservation,
        },
        temporal::BehaviorMetadata,
    },
    data::temporal::{SequenceBatch, SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep},
    nn::structured::{
        Argument, StructuredModelConfig, StructuredPolicy, StructuredQ, StructuredSample,
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

#[path = "support/candidate_policy.rs"]
mod candidate_policy;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    record: LearnerRecord,
    batch: Vec<LegacyTransition>,
    expected_report: Value,
    expected_record: LearnerRecord,
}

#[derive(Deserialize)]
struct LegacyTransition {
    episode_id: u64,
    observation: StructuredSample<usize>,
    step: LegacyStep,
}

#[derive(Deserialize)]
struct LegacyStep {
    action: usize,
    rewards: Vec<f32>,
    next: StructuredSample<usize>,
    terminated: bool,
    truncated: bool,
}

fn fixtures() -> Fixture {
    serde_json::from_str(include_str!("fixtures/structured_mpo_projection_v1.json")).unwrap()
}

#[cfg(feature = "diagnostics")]
#[test]
fn diagnostics_match_sampled_policy_law_and_deterministic_publication() -> rl::Result<()> {
    use rl::diagnostics::{
        probes::ProbeSet,
        schema::{ProbePopulation, RecordContext},
        session::{CaptureRequest, DiagnosticLearner},
        typed::TypedPayload,
    };

    let case = fixtures().cases.into_iter().next().unwrap();
    let record = adapt_record(case.record, false);
    let learner = make(&record, 42);
    let actor = learner.behavior_actor(&CandidateMpoBehavior::default(), learner.updates())?;
    let observation = observation(&case.batch[0].observation);
    let probes = vec![ProbeSet::new_in(
        "candidate-menu",
        ProbePopulation::FixedReference,
        "A factual conditional decision",
        vec![observation.clone().into()],
        None,
        actor.observation_space(),
        actor.action_space().clone(),
    )?];
    let context = RecordContext::new("candidate-mpo", learner.updates());
    for mode in [ActMode::Sample, ActMode::Deterministic] {
        let chunks = learner.diagnostic_records(&CaptureRequest {
            context: &context,
            probes: &probes,
            predictions: true,
            representations: false,
            snapshots: false,
            max_activation_values: 0,
            max_chunk_bytes: 1024 * 1024,
            conventions: None,
            evaluation_mode: mode,
            policy_seed: 37,
        })?;
        let [chunk] = chunks.as_slice() else {
            panic!("expected exactly one prediction chunk")
        };
        let TypedPayload::Predictions(predictions) = TypedPayload::read(chunk)? else {
            unreachable!()
        };
        let row = &predictions.rows[0];
        let expected_action = actor
            .decide(&(), &observation, mode, &mut ChaCha8Rng::seed_from_u64(37))?
            .action;
        let expected_key = observation.finite_action_key(&expected_action)?;
        assert_eq!(row.selected, Some(expected_key));
        let expected = actor.action_probabilities(&observation)?;
        assert_eq!(row.candidates.len(), expected.len());
        for (action, probability) in expected {
            let key = observation.finite_action_key(&Action::Conditional(action.into()))?;
            let candidate = row
                .candidates
                .iter()
                .find(|candidate| candidate.action == key)
                .unwrap();
            let probability = if mode == ActMode::Sample {
                probability
            } else {
                f64::from(u8::from(key == expected_key))
            };
            assert_eq!(candidate.probability, Some(probability));
            assert!(candidate.value.is_some_and(f64::is_finite));
        }
    }
    Ok(())
}

fn adapt_record(mut legacy: LearnerRecord, adam_reset: bool) -> LearnerRecord {
    let actor_updates = legacy.metadata["actor_updates"].as_u64().unwrap();
    let critic: CandidateDdqnConfig =
        serde_json::from_value(legacy.metadata["critic"]["config"].clone()).unwrap();
    let metadata = &legacy.metadata;
    legacy.algorithm = "candidate-mpo-v1".into();
    legacy.temporal_returns = Some(critic.returns());
    legacy.metadata = json!({
        "version": 1,
        "config": metadata["config"],
        "policy": metadata["actor"],
        "actor_updates": actor_updates,
        "adam_updates": if adam_reset { 1 } else { actor_updates },
        "actor_trainability": metadata["actor_trainability"],
        "actor_optimizer_ids": metadata["actor_optimizer_ids"],
        "critic": {"version": 1, "critic": metadata["critic"]},
    });
    legacy
}

fn entity(row: usize) -> EntityId {
    EntityId {
        kind: 1,
        id: row as u64,
    }
}

fn observation(sample: &StructuredSample<usize>) -> Observation {
    // The frozen transaction fixture has a single spatial cell and no events.
    assert_eq!([sample.width, sample.height], [1, 1]);
    assert!(sample.events.is_empty());
    Observation::Structured(Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: Some(SpatialObservation {
            width: 1,
            height: 1,
            cells: sample.spatial.clone(),
        }),
        entities: sample
            .entities
            .iter()
            .enumerate()
            .map(|(row, features)| EntityRow {
                id: entity(row),
                features: features.clone(),
            })
            .collect(),
        events: Vec::new(),
        scalars: sample.scalars.clone(),
        candidates: sample
            .candidates
            .iter()
            .map(|candidate| {
                let argument = match candidate.argument {
                    Argument::None => None,
                    Argument::Entity(row) => Some(ActionArgument::Entity(entity(row))),
                    Argument::Location { x, y } => Some(ActionArgument::Location { x, y }),
                    Argument::RegionalLocation { region, x, y } => {
                        Some(ActionArgument::RegionalLocation { region, x, y })
                    }
                };
                ActionCandidate {
                    action: ConditionalAction {
                        operation: candidate.operation,
                        argument,
                    },
                    features: candidate.features.clone(),
                    enabled: candidate.enabled,
                }
            })
            .collect(),
    }))
}

fn sequence(transitions: &[LegacyTransition]) -> SequenceBatch {
    SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 1,
            n_step: 1,
        },
        samples: transitions
            .iter()
            .map(|transition| {
                let observation = observation(&transition.observation);
                let Observation::Structured(frame) = &observation else {
                    unreachable!()
                };
                let index = transition
                    .observation
                    .candidates
                    .iter()
                    .position(|c| c.key == transition.step.action)
                    .unwrap();
                let action = Action::Conditional(Arc::new(frame.candidates[index].action));
                SequenceSample {
                    episode_id: transition.episode_id,
                    trajectory_start: 0,
                    learn_start: 0,
                    burn_start: 0,
                    observations: vec![observation, self::observation(&transition.step.next)],
                    steps: vec![TemporalStep {
                        action,
                        reward: Reward::from_primitives(transition.step.rewards.clone()).unwrap(),
                        terminated: transition.step.terminated,
                        truncated: transition.step.truncated,
                        behavior: BehaviorMetadata::default(),
                    }],
                    loss_mask: vec![true],
                    history_start: 0,
                    previous: None,
                    origin: SequenceOrigin::Reset,
                }
            })
            .collect(),
    }
}

fn make(record: &LearnerRecord, seed: u64) -> CandidateMpo {
    let device = Device::flex().autodiff();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let policy_cfg: StructuredModelConfig =
        serde_json::from_value(record.metadata["policy"].clone()).unwrap();
    let q_cfg: StructuredModelConfig =
        serde_json::from_value(record.metadata["critic"]["critic"]["model"].clone()).unwrap();
    let critic =
        serde_json::from_value(record.metadata["critic"]["critic"]["config"].clone()).unwrap();
    let config: CandidateMpoConfig =
        serde_json::from_value(record.metadata["config"].clone()).unwrap();
    let schema = &q_cfg.encoder.input;
    let observations = ObservationSpace::Structured(Arc::new(schema.observation_schema()));
    let actions = ActionSpace::Conditional(Arc::new(schema.action_grammar()));
    let policy = StructuredPolicy::new(policy_cfg, &mut rng, &device).unwrap();
    let q = StructuredQ::new(q_cfg, &mut rng, &device).unwrap();
    CandidateMpo::new(policy, q, critic, config, observations, actions).unwrap()
}

#[test]
fn historical_candidate_mpo_projection_preserves_acceptance_and_rollback() {
    for case in fixtures().cases {
        let before = adapt_record(case.record, false);
        let mut learner = make(&before, 99);
        learner.restore(&before).unwrap();
        let result = learner.update(&sequence(&case.batch)).unwrap();
        let expected = adapt_record(case.expected_record, case.name == "accepted_restart");
        assert_eq!(
            serde_json::to_value(result).unwrap(),
            case.expected_report,
            "{}",
            case.name
        );
        assert_eq!(
            checkpoint_eq::normalized(&learner.checkpoint().unwrap()),
            checkpoint_eq::normalized(&expected),
            "{}",
            case.name,
        );
    }
}

#[test]
fn truncated_policy_evaluation_differs_from_optimality_and_actor_resumes() {
    let case = fixtures().cases.into_iter().next().unwrap();
    let optimality = adapt_record(case.record, false);
    let mut evaluation = optimality.clone();
    evaluation.metadata["config"]["backup"] = json!(CandidateMpoBackup::PolicyEvaluation);
    let mut optimality_learner = make(&optimality, 80);
    let mut evaluation_learner = make(&evaluation, 80);
    optimality_learner.restore(&optimality).unwrap();
    evaluation_learner.restore(&evaluation).unwrap();
    let mut batch = sequence(&case.batch);
    for sample in &mut batch.samples {
        sample.steps[0].terminated = false;
        sample.steps[0].truncated = true;
        sample.observations[1] = sample.observations[0].clone();
    }
    let a = optimality_learner.update(&batch).unwrap();
    let b = evaluation_learner.update(&batch).unwrap();
    assert!((a.metrics["target_mean"] - b.metrics["target_mean"]).abs() > 1e-3);

    let actor = evaluation_learner
        .behavior_actor(&CandidateMpoBehavior { epsilon: 1. }, 11)
        .unwrap();
    let restored: CandidateMpoActor =
        serde_json::from_slice(&serde_json::to_vec(&actor).unwrap()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(92);
    let mut restored_rng = rng.clone();
    for _ in 0..12 {
        let observation = &batch.samples[0].observations[0];
        let before = actor
            .decide(&(), observation, ActMode::Sample, &mut rng)
            .unwrap();
        let after = restored
            .decide(&(), observation, ActMode::Sample, &mut restored_rng)
            .unwrap();
        assert_eq!(before.action, after.action);
        assert_eq!(before.log_prob, after.log_prob);
        assert_eq!(after.log_prob, Some((1.0_f64 / 3.).ln() as f32));
    }
}

/// The same harness can run against the reviewed base without changing its code.
/// Timings exclude restoration and verify each measured update against the frozen
/// transaction. Run alone in release mode; no timing assertions belong in CI.
#[test]
#[ignore = "matched release performance probe; run alone with --ignored --nocapture"]
fn candidate_mpo_inference_and_projection_profile() {
    use std::{hint::black_box, time::Instant};
    for case in fixtures().cases {
        let before = adapt_record(case.record, false);
        let mut learner = make(&before, 99);
        let batch = sequence(&case.batch);
        let mut updates = Vec::new();
        for iteration in 0..40 {
            learner.restore(&before).unwrap();
            let start = Instant::now();
            let report = learner.update(&batch).unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            assert_eq!(serde_json::to_value(report).unwrap(), case.expected_report);
            if iteration >= 8 {
                updates.push(elapsed);
            }
        }
        let actor = learner
            .behavior_actor(&CandidateMpoBehavior { epsilon: 0. }, 11)
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(92);
        let mut inference = Vec::new();
        let observation = &batch.samples[0].observations[0];
        let expected = actor
            .decide(&(), observation, ActMode::Deterministic, &mut rng)
            .unwrap();
        for iteration in 0..40 {
            let start = Instant::now();
            for _ in 0..64 {
                black_box(
                    actor
                        .decide(&(), observation, ActMode::Deterministic, &mut rng)
                        .unwrap(),
                );
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000. / 64.;
            if iteration >= 8 {
                inference.push(elapsed);
            }
        }
        let after = actor
            .decide(&(), observation, ActMode::Deterministic, &mut rng)
            .unwrap();
        assert_eq!(expected.action, after.action);
        assert_eq!(expected.log_prob, after.log_prob);
        updates.sort_by(f64::total_cmp);
        inference.sort_by(f64::total_cmp);
        println!(
            "{}",
            json!({"case": case.name, "samples": 32,
            "update_median_ms": (updates[15] + updates[16]) / 2.,
            "inference_median_ms": (inference[15] + inference[16]) / 2.,
            "report": case.expected_report, "action": after.action, "log_prob": after.log_prob})
        );
    }
}
