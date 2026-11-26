use super::*;
use crate::{
    core::{
        action::{ActionArgument, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        reward::Reward,
        stateful::StatefulActor,
        structured::{
            ActionCandidate, EntityId, EntityRow, SpatialObservation, StructuredObservation,
        },
    },
    data::temporal::{
        BehaviorMetadata, SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep,
    },
    nn::structured::{InputSchema, StructuredModelConfig},
};
use burn::{
    module::{Module, Param},
    tensor::Device,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[test]
fn double_q_integrates_only_enabled_online_maximizers() {
    let online = [2., 100., 2., 1.];
    let target = [12., 200., -4., 100.];
    let enabled = [true, false, true, true];
    assert_eq!(
        double_q_target(-2., 0.25, &online, &target, &enabled).unwrap(),
        -1.
    );
    assert_eq!(double_q_target(-2., 0., &[], &[], &[]).unwrap(), -2.);
}

#[test]
fn maximum_uses_target_values_and_ignores_disabled_candidates() {
    let online = [3., 999., 1.];
    let target = [-2., 500., 8.];
    let enabled = [true, false, true];
    assert_eq!(maximum_target(1., 0.5, &target, &enabled).unwrap(), 5.);
    assert_eq!(
        double_q_target(1., 0.5, &online, &target, &enabled).unwrap(),
        0.
    );
    assert_eq!(
        policy_target(1., 0.5, &target, &[0.5, 0., 0.5]).unwrap(),
        2.5
    );
    assert_eq!(maximum_target(1., 0., &[], &[]).unwrap(), 1.);
    assert!(maximum_target(1., 0.5, &target, &[false; 3]).is_err());
}

fn historical_learner() -> (CandidateDdqn, LearnerRecord) {
    let bytes = include_bytes!("../../../tests/fixtures/structured-ddqn-phase-c/learner.json");
    let source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/structured-ddqn-phase-c/source.json"
    ))
    .unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        source["learner_sha256"].as_str().unwrap()
    );
    let record: LearnerRecord = serde_json::from_slice(bytes).unwrap();
    let config: StructuredModelConfig =
        serde_json::from_value(record.metadata["critic"]["model"].clone()).unwrap();
    let critic = serde_json::from_value(record.metadata["critic"]["config"].clone()).unwrap();
    let schema = &config.encoder.input;
    let observation_space = ObservationSpace::Structured(Arc::new(schema.observation_schema()));
    let action_space = ActionSpace::Conditional(Arc::new(schema.action_grammar()));
    let model = StructuredQ::new(
        config,
        &mut ChaCha8Rng::seed_from_u64(91001),
        &Device::flex().autodiff(),
    )
    .unwrap();
    let mut learner = CandidateDdqn::new(model, critic, observation_space, action_space).unwrap();
    learner.restore(&record).unwrap();
    (learner, record)
}

fn entity(id: u64) -> EntityId {
    EntityId { kind: 1, id }
}

fn candidate(
    schema: &InputSchema,
    operation: usize,
    argument: Option<ActionArgument>,
    enabled: bool,
) -> ActionCandidate {
    ActionCandidate {
        action: ConditionalAction {
            operation,
            argument,
        },
        features: vec![0.; schema.candidate_features],
        enabled,
    }
}

fn frame(schema: &InputSchema) -> StructuredObservation {
    StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: Some(SpatialObservation {
            width: 1,
            height: 1,
            cells: vec![0.; schema.spatial_channels],
        }),
        entities: vec![EntityRow {
            id: entity(7),
            features: vec![0.; schema.entity_features],
        }],
        scalars: vec![0.; schema.scalar_features],
        events: vec![],
        candidates: vec![candidate(
            schema,
            0,
            Some(ActionArgument::Entity(entity(7))),
            true,
        )],
    }
}

#[test]
fn original_cnn_learner_record_preserves_predictions_and_adam_metadata() {
    let (learner, legacy) = historical_learner();
    let observation =
        Observation::Structured(Arc::new(frame(learner.critic.online.input_schema())));
    let samples = [model_sample(&observation, &learner.critic.online).unwrap()];
    let online = evaluate_checked(
        &learner.critic.online.valid(),
        &samples,
        &[],
        learner.critic.config.pack_limits(),
    )
    .unwrap();
    let target = evaluate_checked(
        &learner.critic.target,
        &samples,
        &[],
        learner.critic.config.pack_limits(),
    )
    .unwrap();
    assert!((to_vec(online[0].values.clone()).unwrap()[0] - 0.06264479).abs() < 1e-7);
    assert!((to_vec(target[0].values.clone()).unwrap()[0] - 0.028453114).abs() < 1e-7);
    let saved = learner.checkpoint().unwrap();
    assert_eq!(saved.metadata, legacy.metadata);
    saved
        .temporal_returns
        .unwrap()
        .check(learner.critic.config.returns())
        .unwrap();
}

#[test]
fn actor_snapshot_retains_complete_key_likelihoods_for_operation_uniform_noise() {
    let (learner, _) = historical_learner();
    let schema = learner.critic.online.input_schema();
    let mut frame = frame(schema);
    frame.entities.push(EntityRow {
        id: entity(12),
        features: vec![0.; schema.entity_features],
    });
    frame.candidates.extend([
        candidate(schema, 0, Some(ActionArgument::Entity(entity(12))), true),
        candidate(schema, 8, None, true),
        candidate(schema, 1, Some(ActionArgument::Entity(entity(12))), false),
    ]);
    let observation = Observation::Structured(Arc::new(frame));
    let actor = learner
        .behavior_actor(
            &CandidateExploration {
                epsilon: 1.,
                distribution: ExplorationDistribution::Operations,
            },
            13,
        )
        .unwrap();
    let restored: CandidateQActor =
        serde_json::from_slice(&serde_json::to_vec(&actor).unwrap()).unwrap();
    assert_eq!(restored.revision(), 13);
    let mut original_rng = ChaCha8Rng::seed_from_u64(9);
    let mut restored_rng = original_rng.clone();
    let mut seen = [false; 2];
    for _ in 0..12 {
        let original = actor
            .decide(&(), &observation, ActMode::Sample, &mut original_rng)
            .unwrap();
        let chosen = restored
            .decide(&(), &observation, ActMode::Sample, &mut restored_rng)
            .unwrap();
        assert_eq!(chosen.action, original.action);
        assert_eq!(chosen.log_prob, original.log_prob);
        let Action::Conditional(action) = chosen.action else {
            panic!("expected a conditional action")
        };
        let mass = match action.operation {
            0 => {
                seen[0] = true;
                0.25_f64
            }
            8 => {
                seen[1] = true;
                0.50_f64
            }
            _ => panic!("selected a disabled candidate"),
        };
        assert_eq!(chosen.log_prob, Some(mass.ln() as f32));
    }
    assert_eq!(seen, [true, true]);
}

#[test]
fn truncation_bootstraps_after_every_primitive_reward_and_termination_does_not() {
    let (historical, _) = historical_learner();
    let mut config = historical.critic.config.clone();
    config.gamma = 0.5;
    let model_config = historical.critic.online.config().clone();
    let schema = &model_config.encoder.input;
    let current = frame(schema);
    let selected = current.candidates[0].action;
    let mut terminal = current.clone();
    terminal.candidates.clear();
    let current = Observation::Structured(Arc::new(current));
    let terminal = Observation::Structured(Arc::new(terminal));
    for (terminated, successor, expected) in [(true, terminal, 3.), (false, current.clone(), 3.5)] {
        let device = Device::flex().autodiff();
        let mut model = StructuredQ::new(
            model_config.clone(),
            &mut ChaCha8Rng::seed_from_u64(15),
            &device,
        )
        .unwrap();
        let output = model.head.layers.last_mut().unwrap();
        output.weight = Param::from_tensor(Tensor::zeros(output.weight.val().dims(), &device));
        output.bias = Some(Param::from_tensor(Tensor::full([1], 4., &device)));
        let mut learner = CandidateDdqn::new(
            model,
            config.clone(),
            historical.critic.observation_space.clone(),
            historical.critic.action_space.clone(),
        )
        .unwrap();
        let batch = SequenceBatch {
            config: SequenceConfig {
                burn_in: 0,
                learn_len: 1,
                n_step: 1,
            },
            samples: vec![SequenceSample {
                episode_id: 0,
                trajectory_start: 0,
                learn_start: 0,
                burn_start: 0,
                observations: vec![current.clone(), successor],
                steps: vec![TemporalStep {
                    action: Action::Conditional(Arc::new(selected)),
                    reward: Reward::from_primitives(vec![1., 2., 4.]).unwrap(),
                    terminated,
                    truncated: !terminated,
                    behavior: BehaviorMetadata::default(),
                }],
                loss_mask: vec![true],
                history_start: 0,
                previous: None,
                origin: SequenceOrigin::Reset,
            }],
        };
        let report = learner.update(&batch).unwrap();
        assert_eq!(report.metrics["target_mean"], expected);
    }
}

#[test]
fn evaluation_uses_the_frozen_complete_action_distribution() {
    assert_eq!(
        policy_target(-2., 0.25, &[12., -4., 100.], &[0.75, 0.25, 0.]).unwrap(),
        0.
    );
}
