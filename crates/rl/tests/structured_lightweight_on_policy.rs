#![cfg(feature = "neural")]

use burn::{
    module::Param,
    tensor::{Device, Tensor, TensorData},
};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::core::stateful_on_policy::OnPolicyIdentity;
use rl::{
    Result,
    algorithms::{
        LightweightTemporalActor, StructuredTemporalActor, ppo::PpoConfig,
        recurrent_ppo::RecurrentPpo,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction, ConditionalGreedy},
        actor::ActMode,
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::{BehaviorPublisher, HistoryInput, PreviousInput, StatefulActor},
        stateful_on_policy::StatefulOnPolicyActor,
        structured::{
            ActionCandidate, EntityId, EntityRow, EventStamp, INCREMENTAL_EVENT_CONVENTION,
            PublicEvent, StructuredObservation,
        },
    },
    data::temporal::SequenceConfig,
    nn::{
        network::Mlp,
        structured::{
            ArgumentKind, CandidatePolicy, InputSchema, LightweightBatch, LightweightConfig,
            LightweightPolicy, LightweightValue, PackLimits, RecurrentLightweightConfig,
            RecurrentLightweightPolicy, StructuredSample,
        },
        to_vec,
    },
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::Arc;

#[path = "support/structured_semantics.rs"]
mod structured_semantics;

const CURRENT: EntityId = EntityId { kind: 0, id: 1 };
const DEPARTED: EntityId = EntityId { kind: 0, id: 2 };
const MASKED: EntityId = EntityId { kind: 0, id: 3 };
const WAIT: ConditionalAction = ConditionalAction {
    operation: 1,
    argument: None,
};

fn config() -> LightweightConfig {
    let mut semantics = structured_semantics::identity([0, 1, 1, 1, 1], &["inspect", "wait"]);
    semantics
        .conventions
        .push(INCREMENTAL_EVENT_CONVENTION.into());
    LightweightConfig::small(InputSchema {
        version: 2,
        spatial_channels: 0,
        entity_features: 1,
        event_features: 1,
        scalar_features: 1,
        candidate_features: 1,
        operations: vec![ArgumentKind::Entity, ArgumentKind::None],
        semantics,
    })
}

fn frame() -> StructuredObservation {
    StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        entities: vec![EntityRow {
            id: CURRENT,
            features: vec![1.],
        }],
        events: vec![PublicEvent {
            features: [4.].into(),
            entity: Some(DEPARTED),
            context: [CURRENT, DEPARTED].into(),
            stamp: Some(EventStamp {
                source: 0,
                ordinal: 0,
                tick: 0,
            }),
        }],
        scalars: vec![0.],
        candidates: vec![ActionCandidate {
            action: WAIT,
            features: vec![0.],
            enabled: true,
        }],
    }
}

fn live_frame() -> StructuredObservation {
    let mut frame = frame();
    frame
        .entities
        .extend([DEPARTED, MASKED].map(|id| EntityRow {
            id,
            features: vec![1.],
        }));
    frame
        .candidates
        .extend([CURRENT, DEPARTED, MASKED].map(|id| ActionCandidate {
            action: ConditionalAction {
                operation: 0,
                argument: Some(ActionArgument::Entity(id)),
            },
            features: vec![0.],
            enabled: id != MASKED,
        }));
    frame
}

fn models(seed: u64, device: &Device) -> (LightweightPolicy, LightweightValue) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (
        LightweightPolicy::new(config(), &mut rng, device).unwrap(),
        LightweightValue::new(config(), &mut rng, device).unwrap(),
    )
}

#[test]
fn frame_projection_keeps_stamped_evidence_of_departed_entities() {
    let sample = StructuredSample::from_observation(&frame(), &config().input).unwrap();
    assert_eq!(sample.events.len(), 1);
    assert_eq!(sample.events[0].features, [4.]);
    assert_eq!(sample.events[0].entity, None);
    assert_eq!(sample.events[0].context, [0]);
}

#[test]
fn identical_current_frame_yields_identical_behavior_after_a_prefix() {
    let (policy, value) = models(17, &Device::flex());
    let actor = LightweightTemporalActor::new(policy, Some(value), 0).unwrap();
    let mut early = live_frame();
    early.scalars[0] = 9.;
    let early = Observation::Structured(Arc::new(early));
    let current = Observation::Structured(Arc::new(live_frame()));
    actor.initial_state().unwrap();
    actor
        .assimilate(
            &(),
            HistoryInput {
                observation: &current,
                previous: None,
            },
        )
        .unwrap();
    actor
        .reconstruct_history(&[
            HistoryInput {
                observation: &early,
                previous: None,
            },
            HistoryInput {
                observation: &current,
                previous: Some(PreviousInput::from_transition(
                    &Action::Conditional(Arc::new(WAIT)),
                    &(-3f32).into(),
                    &early,
                )),
            },
        ])
        .unwrap();
    let mut fresh_rng = ChaCha8Rng::seed_from_u64(53);
    let mut prefix_rng = fresh_rng.clone();
    let fresh = actor
        .sample_on_policy(&(), &current, ActMode::Sample, &mut fresh_rng)
        .unwrap();
    let after_prefix = actor
        .sample_on_policy(&(), &current, ActMode::Sample, &mut prefix_rng)
        .unwrap();
    assert_eq!(fresh.distribution, after_prefix.distribution);
    assert_eq!(fresh.value, after_prefix.value);
    assert_eq!(fresh.decision.action, after_prefix.decision.action);
    assert_eq!(fresh.decision.log_prob, after_prefix.decision.log_prob);
}

#[test]
fn old_events_in_the_current_notebook_change_the_complete_law() {
    let (policy, _) = models(17, &Device::flex());
    let with_event = live_frame();
    let mut without_event = with_event.clone();
    without_event.events.clear();
    let limits = PackLimits::default();
    let with_event = policy.complete_law(&with_event, limits).unwrap();
    let without_event = policy.complete_law(&without_event, limits).unwrap();
    assert_eq!(with_event.keys(), without_event.keys());
    let with = to_vec(with_event.log_probs().clone()).unwrap();
    let without = to_vec(without_event.log_probs().clone()).unwrap();
    assert!(with.iter().zip(without).any(|(a, b)| (a - b).abs() > 1e-5));
}

#[test]
fn feedforward_policy_budget_includes_conditional_head_inputs() {
    let device = Device::flex();
    let (policy, _) = models(17, &device);
    let observation = live_frame();
    let schema = &policy.config().input;
    let sample = StructuredSample::from_observation(&observation, schema).unwrap();
    let packed = LightweightBatch::pack(schema, &[sample], PackLimits::default(), &device)
        .unwrap()
        .input_elements();
    let ops = schema.operations.len();
    let max_elements = packed + ops * ops + 2 * ops + 2;
    assert!(
        policy
            .complete_law(&observation, PackLimits { max_elements })
            .is_ok()
    );
    assert!(
        policy
            .complete_law(
                &observation,
                PackLimits {
                    max_elements: max_elements - 1,
                },
            )
            .is_err()
    );
}

fn equal_operations() -> LightweightPolicy {
    let device = Device::flex();
    let (mut policy, _) = models(17, &device);
    let final_layer = policy.operation_head.layers.last_mut().unwrap();
    final_layer.weight =
        Param::from_tensor(Tensor::zeros(final_layer.weight.val().dims(), &device));
    final_layer.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
    policy
}

fn prefer_operation_with_more_arguments(head: &mut Mlp, query: &mut Mlp, device: &Device) {
    let [input, width] = head.layers[0].weight.val().dims();
    let mut first = vec![0.; input * width];
    first[(input - 1) * width] = 1.;
    let mut last = vec![0.; width];
    last[0] = 1.;
    // The final input is ln(1 + enabled arguments), including the no-argument row.
    head.layers[0].weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(first, [input, width]),
        device,
    ));
    head.layers[1].weight =
        Param::from_tensor(Tensor::from_data(TensorData::new(last, [width, 1]), device));
    query.layers[0].weight =
        Param::from_tensor(Tensor::zeros(query.layers[0].weight.val().dims(), device));
}

fn check_hierarchical_deployment<A: StatefulOnPolicyActor + Serialize + DeserializeOwned>(
    actor: A,
) {
    let restored: A = serde_json::from_str(&serde_json::to_string(&actor).unwrap()).unwrap();
    let expected = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(CURRENT)),
    };
    for actor in [&actor, &restored] {
        let mut frame = live_frame();
        for permute in [false, true] {
            if permute {
                frame.candidates.reverse();
                frame.entities.reverse();
            }
            let observation = Observation::Structured(Arc::new(frame.clone()));
            let state = actor
                .reconstruct_history(&[HistoryInput {
                    observation: &observation,
                    previous: None,
                }])
                .unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(53);
            let mut untouched = rng.clone();
            let decision = actor
                .decide(&state, &observation, ActMode::Deterministic, &mut rng)
                .unwrap();
            let sample = actor
                .sample_on_policy(&state, &observation, ActMode::Deterministic, &mut rng)
                .unwrap();
            // Inspect has mass 0.6 split equally between two legal arguments.
            // Wait alone has mass 0.4 and would be the joint mode.
            assert_eq!(decision.action, Action::Conditional(Arc::new(expected)));
            assert_eq!(sample.decision.action, decision.action);
            assert!((decision.log_prob.unwrap().exp() - 0.3).abs() < 1e-6);
            assert!((sample.decision.log_prob.unwrap().exp() - 0.3).abs() < 1e-6);
            assert_eq!(rng.next_u64(), untouched.next_u64());
        }
        frame
            .candidates
            .iter_mut()
            .for_each(|row| row.enabled = row.action == WAIT);
        let observation = Observation::Structured(Arc::new(frame));
        let state = actor
            .reconstruct_history(&[HistoryInput {
                observation: &observation,
                previous: None,
            }])
            .unwrap();
        assert_eq!(
            actor
                .decide(
                    &state,
                    &observation,
                    ActMode::Deterministic,
                    &mut ChaCha8Rng::seed_from_u64(1)
                )
                .unwrap()
                .action,
            Action::Conditional(Arc::new(WAIT)),
        );
    }
}

#[test]
fn feedforward_actor_deploys_the_most_probable_operation() {
    let device = Device::flex();
    let (mut policy, _) = models(17, &device);
    prefer_operation_with_more_arguments(
        &mut policy.operation_head,
        &mut policy.entity_query,
        &device,
    );
    check_hierarchical_deployment(LightweightTemporalActor::new(policy, None, 0).unwrap());
}

#[test]
fn recurrent_actor_deploys_the_most_probable_operation() {
    let device = Device::flex();
    let mut config = RecurrentLightweightConfig::small(config().input);
    config.hidden_width = 8;
    let mut policy =
        RecurrentLightweightPolicy::new(config, &mut ChaCha8Rng::seed_from_u64(17), &device)
            .unwrap();
    prefer_operation_with_more_arguments(
        &mut policy.operation_head,
        &mut policy.entity_query,
        &device,
    );
    check_hierarchical_deployment(StructuredTemporalActor::new(policy, None, 0).unwrap());
}

#[test]
fn complete_law_greedy_modes_have_distinct_semantic_orderings() {
    let choose = |entity| ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity)),
    };
    let probabilities = [
        (WAIT, 0.375),
        (choose(DEPARTED), 0.3125),
        (choose(CURRENT), 0.3125),
        (choose(MASKED), 0.),
    ];
    for rows in [
        probabilities.to_vec(),
        probabilities.into_iter().rev().collect(),
    ] {
        assert_eq!(
            ConditionalGreedy::Hierarchical
                .select(rows.clone())
                .unwrap(),
            (choose(CURRENT), 0.3125)
        );
        assert_eq!(
            ConditionalGreedy::JointMap.select(rows).unwrap(),
            (WAIT, 0.375)
        );
    }
    assert_eq!(
        ConditionalGreedy::Hierarchical
            .select([(WAIT, 0.5), (choose(CURRENT), 0.5)])
            .unwrap()
            .0,
        choose(CURRENT),
    );
    assert_eq!(
        ConditionalGreedy::Hierarchical
            .select([(WAIT, 1.), (choose(CURRENT), 0.)])
            .unwrap()
            .0,
        WAIT,
    );
}

#[test]
fn candidate_policy_preserves_masked_rows_and_requested_key_order() {
    let policy = equal_operations();
    let mut frame = live_frame();
    frame.events[0].entity = None;
    let sample = StructuredSample::from_observation(&frame, &config().input).unwrap();
    let selected = vec![sample.candidates[2].key, WAIT, sample.candidates[1].key];
    let masked = sample.candidates[3].key;
    let limits = PackLimits::default();
    let probabilities = policy
        .complete_probabilities(std::slice::from_ref(&sample), limits)
        .unwrap();
    assert_eq!(probabilities, [vec![0.5, 0.25, 0.25, 0.]]);
    let logs = policy
        .complete_log_probabilities(std::slice::from_ref(&sample), &[selected], limits)
        .unwrap();
    let logs = to_vec(logs.into_iter().next().unwrap()).unwrap();
    for (log, p) in logs.into_iter().zip([0.25f32, 0.5, 0.25]) {
        assert!((log - p.ln()).abs() < 1e-6);
    }
    assert!(
        policy
            .complete_log_probabilities(&[sample], &[vec![masked]], limits)
            .is_err()
    );
}

#[derive(Serialize, Deserialize)]
struct NotebookEnv {
    tick: usize,
    actions: ActionSpace,
}

impl Env for NotebookEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Structured(Arc::new(config().input.observation_schema()))
    }
    fn action_space(&self) -> &ActionSpace {
        &self.actions
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.tick = 0;
        Ok(EpisodeStart::fresh(self.observation()))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.observation().check_action(&self.actions, action)?;
        self.tick += 1;
        Ok(Step {
            observation: self.observation(),
            reward: (self.tick as f32).into(),
            terminated: self.tick == 2,
            truncated: false,
        })
    }
}

impl NotebookEnv {
    fn observation(&self) -> Observation {
        let mut frame = if self.tick == 0 {
            live_frame()
        } else {
            frame()
        };
        frame.scalars[0] = self.tick as f32;
        if self.tick == 2 {
            frame.candidates.clear();
        }
        Observation::Structured(Arc::new(frame))
    }
}

impl Snapshot for NotebookEnv {
    fn validate_snapshot(&self) -> Result<()> {
        assert!(self.tick <= 2);
        Ok(())
    }
}

#[test]
fn lightweight_ppo_updates_and_resumes_a_saved_run() {
    for entropy in [0., 0.01] {
        check_ppo_resume(entropy);
    }
}

fn check_ppo_resume(entropy: f32) {
    let build = |seed| {
        let (policy, value) = models(seed, &Device::flex().autodiff());
        RecurrentPpo::new(
            policy,
            value,
            PpoConfig {
                epochs: 2,
                minibatch_size: 2,
                entropy,
                ..Default::default()
            },
            seed,
        )
        .unwrap()
    };
    let mut learner = build(17);
    let mut trainer = TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(vec![NotebookEnv {
            tick: 0,
            actions: ActionSpace::Conditional(Arc::new(config().input.action_grammar())),
        }])
        .unwrap(),
        &learner,
        TemporalOnPolicyConfig {
            collection: OnPolicyConfig {
                rollout_steps: 3,
                ..Default::default()
            },
            sequence: SequenceConfig {
                burn_in: 0,
                learn_len: 2,
                n_step: 1,
            },
            ..Default::default()
        },
        61,
    )
    .unwrap();
    let updated = trainer.cycle(&mut learner).unwrap();
    assert!(updated.updates > 0);
    assert!(!updated.metrics.is_empty());
    assert!(updated.metrics.values().all(|loss| loss.is_finite()));

    let path = tempfile::tempdir().unwrap();
    let path = path.path().join("lightweight-ppo");
    save_run(&path, &learner, &trainer).unwrap();
    let mut resumed = build(91);
    let mut resumed_trainer: TemporalOnPolicyTrainer<
        SyncVectorEnv<NotebookEnv>,
        LightweightTemporalActor,
    > = load_run(&path, &mut resumed).unwrap();
    assert_eq!(resumed.revision(), learner.revision());
    assert_eq!(resumed.updates(), learner.updates());
    assert_eq!(
        resumed
            .behavior_actor(&(), resumed.revision())
            .unwrap()
            .behavior_identity(),
        learner
            .behavior_actor(&(), learner.revision())
            .unwrap()
            .behavior_identity()
    );
    let next = trainer.cycle(&mut learner).unwrap();
    let resumed_next = resumed_trainer.cycle(&mut resumed).unwrap();
    assert_eq!(next.updates, resumed_next.updates);
    assert_eq!(next.metrics, resumed_next.metrics);
}
