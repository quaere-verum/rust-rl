#![cfg(feature = "neural")]

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::{Device, Gradients, Tensor, TensorData},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionArgument, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        env::{Observation, PublicHistory},
        reward::Reward,
        stateful::{HistoryInput, PreviousInput},
        structured::{
            ActionCandidate, EntityId, EntityRow, EventStamp, PublicEvent, SpatialObservation,
            StructuredObservation,
        },
    },
    nn::{
        distributions::Distribution,
        structured::{
            Argument, ConditionalPolicyLaw, PackLimits, RecurrentLightweightConfig,
            RecurrentLightweightDistributionalQ, RecurrentLightweightPolicy, RecurrentLightweightQ,
            RecurrentLightweightValue, StructuredHistoryInput, StructuredSample,
        },
        to_vec,
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Deserialize)]
struct LegacyStep {
    action: String,
    rewards: Vec<f32>,
    next: StructuredSample<String>,
}

#[derive(Deserialize)]
struct LegacyHistory {
    initial: StructuredSample<String>,
    steps: Vec<LegacyStep>,
}

#[derive(Deserialize)]
struct Fixture {
    record: LearnerRecord,
    histories: Vec<LegacyHistory>,
    outputs: Value,
}

fn entity(row: usize) -> EntityId {
    EntityId {
        kind: 0,
        id: row as u64,
    }
}

fn public_action(candidate: &rl::nn::structured::Candidate<String>) -> ConditionalAction {
    ConditionalAction {
        operation: candidate.operation,
        argument: match candidate.argument {
            Argument::None => None,
            Argument::Entity(row) => Some(ActionArgument::Entity(entity(row))),
            Argument::Location { x, y } => Some(ActionArgument::Location { x, y }),
            Argument::RegionalLocation { region, x, y } => {
                Some(ActionArgument::RegionalLocation { region, x, y })
            }
        },
    }
}

fn observation(
    sample: &StructuredSample<String>,
    events: &mut Vec<PublicEvent>,
    tick: u64,
) -> Observation {
    for event in &sample.events {
        events.push(PublicEvent {
            features: event.features.as_slice().into(),
            entity: event.entity.map(entity),
            context: event
                .context
                .iter()
                .copied()
                .map(entity)
                .collect::<Vec<_>>()
                .into(),
            stamp: Some(EventStamp {
                source: 0,
                ordinal: events.len() as u64,
                tick,
            }),
        });
    }
    Observation::Structured(Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: Some(SpatialObservation {
            width: sample.width,
            height: sample.height,
            // The frozen fixture has exactly one spatial channel.
            cells: sample.spatial.clone(),
        }),
        entities: sample
            .entities
            .iter()
            .enumerate()
            .map(|(i, features)| EntityRow {
                id: entity(i),
                features: features.clone(),
            })
            .collect(),
        events: events.clone(),
        scalars: sample.scalars.clone(),
        candidates: sample
            .candidates
            .iter()
            .map(|candidate| ActionCandidate {
                action: public_action(candidate),
                features: candidate.features.clone(),
                enabled: candidate.enabled,
            })
            .collect(),
    }))
}

fn public_history(history: &LegacyHistory) -> PublicHistory {
    let mut events = Vec::new();
    let mut previous = &history.initial;
    let mut tape = PublicHistory::new(observation(previous, &mut events, 0));
    for step in &history.steps {
        let candidate = previous
            .candidates
            .iter()
            .find(|c| c.key == step.action)
            .unwrap();
        let action = Action::Conditional(Arc::new(public_action(candidate)));
        let reward = Reward::from_primitives(step.rewards.clone()).unwrap();
        let tick = tape.physical_timestep() + reward.duration();
        tape = tape
            .append(action, reward, observation(&step.next, &mut events, tick))
            .unwrap();
        previous = &step.next;
    }
    tape
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/structured_recurrent_q_v1.json")).unwrap()
}

fn model(record: &LearnerRecord, device: &Device) -> RecurrentLightweightQ {
    let config: RecurrentLightweightConfig =
        serde_json::from_value(record.metadata.clone()).unwrap();
    let mut q =
        RecurrentLightweightQ::new(config, &mut ChaCha8Rng::seed_from_u64(41), device).unwrap();
    q.restore(record).unwrap();
    q
}

#[test]
fn historical_record_preserves_forward_and_gradient_arithmetic_in_both_modes() {
    let Fixture {
        record,
        histories,
        outputs: expected,
    } = fixture();
    let tapes: Vec<_> = histories.iter().map(public_history).collect();
    let tapes: Vec<_> = tapes.iter().map(PublicHistory::inputs).collect();
    let mut outputs = Vec::new();
    for detach in [false, true] {
        let q = model(&record, &Device::flex().autodiff());
        let inputs: Vec<_> = tapes
            .iter()
            .map(|inputs| StructuredHistoryInput {
                inputs,
                learn_start: if detach { inputs.len() - 1 } else { 0 },
            })
            .collect();
        let histories = q
            .evaluate_histories(&inputs, PackLimits::default())
            .unwrap();
        let values: Vec<_> = histories.iter().map(|h| json!({
            "states":to_vec(h.states.clone()).unwrap(),
            "q":h.decisions.iter().map(|d| d.as_ref().map(|d| to_vec(d.clone()).unwrap())).collect::<Vec<_>>(),
        })).collect();
        let losses: Vec<_> = histories
            .iter()
            .map(|h| {
                h.decisions
                    .last()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .clone()
                    .powi_scalar(2)
                    .sum()
            })
            .collect();
        let gradients = Tensor::cat(losses, 0).sum().backward();
        let leaves = [
            &q.encoder.events.layers[0].weight,
            &q.event_projection.layers[0].weight,
            &q.decision_projection.layers[0].weight,
            &q.head.layers[0].weight,
            &q.memory.gru.new_gate.input_transform.weight,
            &q.memory.gru.new_gate.hidden_transform.weight,
        ];
        let gradients: Vec<_> = leaves
            .iter()
            .map(|p| p.val().grad(&gradients).map(|g| to_vec(g).unwrap()))
            .collect();
        outputs.push(json!({"detached":detach,"values":values,"gradients":gradients}));
    }
    assert_eq!(json!(outputs), expected);
}

#[test]
fn scalar_assimilation_matches_exact_history_with_cumulative_events() {
    let Fixture {
        record, histories, ..
    } = fixture();
    let tape = public_history(&histories[0]);
    let inputs = tape.inputs();
    let q = model(&record, &Device::flex());
    let limits = PackLimits::default();
    let exact = q
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: 0,
            }],
            limits,
        )
        .unwrap();
    let mut state = q.initial_state().unwrap();
    for (row, &input) in inputs.iter().enumerate() {
        state = q.assimilate(&state, input, limits).unwrap();
        let Observation::Structured(observation) = input.observation else {
            unreachable!()
        };
        let decisions = q
            .evaluate_decision(&state, observation, limits)
            .unwrap()
            .unwrap();
        let expected = exact[0].decisions[row].as_ref().unwrap().clone();
        assert_eq!(decisions.dims(), expected.dims());
        for (actual, expected) in to_vec(decisions)
            .unwrap()
            .iter()
            .zip(to_vec(expected).unwrap())
        {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        let expected = to_vec(exact[0].states.clone().narrow(0, row, 1)).unwrap();
        for (actual, expected) in to_vec(state.memory.clone()).unwrap().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(state.event_start, usize::from(row != 0));
    }
}

#[test]
fn assimilation_uses_selection_rows_and_primitive_duration_after_entity_removal() {
    let Fixture {
        record, histories, ..
    } = fixture();
    let history = &histories[0];
    let mut selection = history.initial.clone();
    selection.candidates = history.steps.last().unwrap().next.candidates.clone();
    let action = Action::Conditional(Arc::new(public_action(&selection.candidates[0])));
    let mut notebook = Vec::new();
    let selected = observation(&selection, &mut notebook, 0);
    let mut successor = history.steps[0].next.clone();
    successor.entities.clear();
    let successor = observation(&successor, &mut notebook, 1);
    let mut changed_selection = selected.clone();
    let Observation::Structured(frame) = &mut changed_selection else {
        unreachable!()
    };
    Arc::make_mut(frame).entities[0].features[0] = -3.;

    let q = model(&record, &Device::flex());
    let prior = q.initial_state().unwrap();
    let limits = PackLimits::default();
    let state = |selection, reward| {
        let input = HistoryInput {
            observation: &successor,
            previous: Some(PreviousInput::from_transition(&action, reward, selection)),
        };
        let state = q.assimilate(&prior, input, limits).unwrap();
        assert_eq!(state.event_start, 1);
        to_vec(state.memory).unwrap()
    };
    let single = Reward::from(1.);
    let multi = Reward::from_primitives(vec![0.25, 0.75]).unwrap();
    let reference = state(&selected, &single);
    assert_ne!(reference, state(&changed_selection, &single));
    assert_ne!(reference, state(&selected, &multi));
    assert!(
        q.assimilate(
            &prior,
            HistoryInput {
                observation: &successor,
                previous: Some(PreviousInput::from_total(&action, 1.)),
            },
            limits
        )
        .is_err()
    );
}

#[test]
fn recurrent_location_uses_the_selection_regions_dimensions() {
    let Fixture {
        record, histories, ..
    } = fixture();
    let mut config: RecurrentLightweightConfig = serde_json::from_value(record.metadata).unwrap();
    config
        .model
        .input
        .operations
        .push(rl::nn::structured::ArgumentKind::Location);
    config
        .model
        .input
        .semantics
        .operations
        .push("regional location".into());
    let q = RecurrentLightweightQ::new(config, &mut ChaCha8Rng::seed_from_u64(7), &Device::flex())
        .unwrap();

    let mut notebook = Vec::new();
    let selection = observation(&histories[0].initial, &mut notebook, 0);
    let Observation::Structured(selection) = selection else {
        unreachable!()
    };
    let mut regional = selection.as_ref().clone();
    let map = SpatialObservation {
        width: 3,
        height: 2,
        cells: vec![0.; 6],
    };
    regional.regions.push(rl::core::structured::SpatialRegion {
        id: 9,
        map: map.clone(),
    });
    let mut ordinary = selection.as_ref().clone();
    ordinary.spatial = Some(map);

    let action = |argument| ConditionalAction {
        operation: 2,
        argument: Some(argument),
    };
    let regional_action = action(ActionArgument::RegionalLocation {
        region: 9,
        x: 2,
        y: 1,
    });
    let ordinary_action = action(ActionArgument::Location { x: 2, y: 1 });
    for (frame, action) in [
        (&mut regional, regional_action),
        (&mut ordinary, ordinary_action),
    ] {
        frame.candidates = vec![ActionCandidate {
            action,
            features: vec![0.],
            enabled: true,
        }];
    }
    let successor = observation(&histories[0].steps[0].next, &mut notebook, 1);
    let reward = Reward::from(0.);
    let prior = q.initial_state().unwrap();
    let limits = PackLimits::default();

    let assimilate = |selection, action| {
        let selection = Observation::Structured(Arc::new(selection));
        let action = Action::Conditional(Arc::new(action));
        q.assimilate(
            &prior,
            HistoryInput {
                observation: &successor,
                previous: Some(PreviousInput::from_transition(&action, &reward, &selection)),
            },
            limits,
        )
    };
    let expected = assimilate(ordinary, ordinary_action).unwrap();
    let actual = assimilate(regional.clone(), regional_action).unwrap();
    assert_eq!(
        to_vec(actual.memory).unwrap(),
        to_vec(expected.memory).unwrap()
    );
    regional.regions[0].map.width = 1;
    regional.regions[0].map.cells.truncate(2);
    assert!(assimilate(regional, regional_action).is_err());
}

#[test]
fn ragged_token_and_input_budgets_reject_before_packing() {
    let Fixture {
        record, histories, ..
    } = fixture();
    let tapes: Vec<_> = histories.iter().map(public_history).collect();
    let inputs: Vec<_> = tapes.iter().map(PublicHistory::inputs).collect();
    let rows: Vec<_> = inputs
        .iter()
        .map(|inputs| StructuredHistoryInput {
            inputs,
            learn_start: 0,
        })
        .collect();
    let mut config: RecurrentLightweightConfig = serde_json::from_value(record.metadata).unwrap();
    let build = |config| {
        RecurrentLightweightQ::new(config, &mut ChaCha8Rng::seed_from_u64(7), &Device::flex())
            .unwrap()
    };
    // Histories have four and six actual tokens, and share twelve padded slots.
    config.max_tokens = 11;
    let q = build(config.clone());
    let mut stages = Vec::new();
    assert!(
        q.evaluate_histories_observed(&rows, PackLimits::default(), |stage| stages.push(stage))
            .is_err()
    );
    assert!(stages.is_empty());
    config.max_tokens = 12;
    let q = build(config.clone());
    assert!(q.evaluate_histories(&rows, PackLimits::default()).is_ok());
    assert!(
        q.evaluate_histories_observed(&rows, PackLimits { max_elements: 32 }, |stage| stages
            .push(stage))
            .is_err()
    );
    assert!(stages.is_empty());
    config.max_history_bytes = 1;
    let q = build(config);
    assert!(
        q.evaluate_histories_observed(&rows, PackLimits::default(), |stage| stages.push(stage))
            .is_err()
    );
    assert!(stages.is_empty());
}

fn policy_fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/structured_recurrent_policy_v1.json")).unwrap()
}

fn policy_model(record: &LearnerRecord, device: &Device) -> RecurrentLightweightPolicy {
    let config = serde_json::from_value(record.metadata.clone()).unwrap();
    let mut policy =
        RecurrentLightweightPolicy::new(config, &mut ChaCha8Rng::seed_from_u64(23), device)
            .unwrap();
    policy.restore(record).unwrap();
    policy
}

/// Captured before adapting the public history input.
#[test]
fn historical_policy_record_preserves_logits_likelihoods_and_gradients() {
    let Fixture {
        record,
        histories,
        outputs: expected,
    } = policy_fixture();
    let tapes: Vec<_> = histories.iter().map(public_history).collect();
    let inputs: Vec<_> = tapes.iter().map(PublicHistory::inputs).collect();
    let device = Device::flex().autodiff();
    let mut outputs = Vec::new();
    for detached in [false, true] {
        let policy = policy_model(&record, &device);
        let rows: Vec<_> = inputs
            .iter()
            .map(|inputs| StructuredHistoryInput {
                inputs,
                learn_start: if detached { inputs.len() - 1 } else { 0 },
            })
            .collect();
        let evaluated = policy
            .evaluate_histories(&rows, PackLimits::default())
            .unwrap();
        let mut values = Vec::new();
        let mut losses = Vec::new();
        for (history, tape) in evaluated.iter().zip(&tapes) {
            let row = tape.timestep() as usize;
            let Distribution::Categorical { logits, masks } = history.operations() else {
                unreachable!()
            };
            let logits = to_vec(logits.clone()).unwrap();
            let branch = |operation| {
                let Distribution::Categorical { logits, .. } = history
                    .conditional(row, operation)
                    .unwrap()
                    .unwrap()
                    .distribution
                else {
                    unreachable!()
                };
                to_vec(logits).unwrap()
            };
            let Observation::Structured(observation) = tape.observation() else {
                unreachable!()
            };
            let keys: Vec<_> = observation
                .candidates
                .iter()
                .filter(|c| c.enabled)
                .map(|c| c.action)
                .collect();
            let logs = history.log_prob_many(row, &keys).unwrap();
            values.push(json!({
                "states":to_vec(history.states.clone()).unwrap(),
                "operations":logits,
                "masks":masks,
                "entity":branch(0),
                "location":branch(2),
                "logs":to_vec(logs.clone()).unwrap(),
            }));
            let weights = Tensor::from_data(
                TensorData::new(vec![0.07f32, 0.11, 0.23, 0.29, 0.30], [5]),
                &device,
            );
            losses.push((logs * weights).sum());
        }
        let gradients = Tensor::cat(losses, 0).sum().backward();
        let leaves = [
            &policy.encoder.events.layers[0].weight,
            &policy.event_projection.layers[0].weight,
            &policy.decision_projection.layers[0].weight,
            &policy.memory.gru.new_gate.input_transform.weight,
            &policy.operation_head.layers[0].weight,
            &policy.entity_query.layers[0].weight,
            &policy.entity_key.layers[0].weight,
            &policy.location_query.layers[0].weight,
            &policy.location_key.layers[0].weight,
        ];
        let gradients: Vec<_> = leaves
            .iter()
            .map(|p| p.val().grad(&gradients).map(|g| to_vec(g).unwrap()))
            .collect();
        outputs.push(json!({"detached":detached,"values":values,"gradients":gradients}));
    }
    assert_eq!(json!(outputs), expected);
}

#[test]
fn scalar_policy_matches_history_law_and_draws_with_a_new_event_suffix() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let inputs = tape.inputs();
    let policy = policy_model(&record, &Device::flex());
    let limits = PackLimits::default();
    let exact = policy
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: 0,
            }],
            limits,
        )
        .unwrap();
    let mut state = policy.initial_state().unwrap();
    for (row, &input) in inputs.iter().enumerate() {
        state = policy.assimilate(&state, input, limits).unwrap();
        let Observation::Structured(observation) = input.observation else {
            unreachable!()
        };
        let scalar = policy
            .evaluate_decision(&state, observation, limits)
            .unwrap();
        let actual = scalar.complete_probabilities_f64(0).unwrap();
        let expected = exact[0].complete_probabilities_f64(row).unwrap();
        assert!((actual.iter().sum::<f64>() - 1.).abs() < 1e-12);
        assert_eq!(actual[2], 0., "masked complete candidate");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        let mut scalar_rng = ChaCha8Rng::seed_from_u64(row as u64);
        let mut exact_rng = scalar_rng.clone();
        let actual = scalar
            .sample_row(0, ActMode::Sample, &mut scalar_rng)
            .unwrap();
        let expected = exact[0]
            .sample_row(row, ActMode::Sample, &mut exact_rng)
            .unwrap();
        assert_eq!(actual.key, expected.key);
        assert!((actual.log_prob - expected.log_prob).abs() < 1e-6);
        assert_eq!(scalar_rng.random::<u64>(), exact_rng.random::<u64>());
    }
}

#[test]
fn terminal_only_history_retains_memory_without_policy_support() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let mut sample = histories[0].initial.clone();
    sample.candidates.clear();
    let terminal = observation(&sample, &mut Vec::new(), 0);
    let inputs = [HistoryInput {
        observation: &terminal,
        previous: None,
    }];
    let policy = policy_model(&record, &Device::flex());
    let evaluated = policy
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: 0,
            }],
            PackLimits::default(),
        )
        .unwrap();
    let h = &evaluated[0];
    assert_eq!(h.states.dims(), [1, policy.config().hidden_width]);
    let Distribution::Categorical { logits, masks } = h.operations() else {
        unreachable!()
    };
    assert_eq!(
        logits.dims(),
        [0, policy.config().model.input.operations.len()]
    );
    assert!(masks.is_empty());
    assert!(h.complete_probabilities_f64(0).is_err());
}

#[test]
fn learning_boundary_precedes_the_first_new_event() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let inputs = tape.inputs();
    let device = Device::flex().autodiff();
    let policy = policy_model(&record, &device);
    // The two events added on row two belong to the learning window.
    let evaluation = policy
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: 2,
            }],
            PackLimits::default(),
        )
        .unwrap();
    let Observation::Structured(final_frame) = tape.observation() else {
        unreachable!()
    };
    let key = final_frame.candidates[0].action;
    let gradients = evaluation[0]
        .log_prob_many(inputs.len() - 1, &[key])
        .unwrap()
        .backward();
    let leaf = policy.encoder.events.layers[0].weight.val();
    assert!(
        to_vec(leaf.grad(&gradients).unwrap())
            .unwrap()
            .iter()
            .any(|value| value.abs() > 1e-7)
    );
}

#[test]
fn complete_policy_law_matches_teacher_forcing_and_retains_gradients() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let inputs = tape.inputs();
    let device = Device::flex().autodiff();
    let policy = policy_model(&record, &device);
    assert_eq!(
        RecurrentLightweightPolicy::parameter_count(policy.config()).unwrap(),
        policy.num_params()
    );
    let evaluations = policy
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: inputs.len() - 1,
            }],
            PackLimits::default(),
        )
        .unwrap();
    let evaluation = &evaluations[0];
    let row = inputs.len() - 1;
    let Observation::Structured(observation) = inputs[row].observation else {
        unreachable!()
    };
    assert!(observation.candidates.iter().any(|c| !c.enabled));
    let keys = observation
        .candidates
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.action)
        .collect::<Vec<_>>();
    let law = evaluation.complete_law(row).unwrap();
    assert_eq!(law.keys(), keys);
    let log_probs = to_vec(law.log_probs().clone()).unwrap();
    let teacher = to_vec(evaluation.log_prob_many(row, &keys).unwrap()).unwrap();
    assert_eq!(log_probs, teacher);
    assert!((log_probs.iter().map(|p| p.exp()).sum::<f32>() - 1.).abs() < 1e-6);
    let gradients = law.entropy().backward();
    for head in [&policy.entity_query, &policy.location_query] {
        let gradient = head.layers[0].weight.val().grad(&gradients).unwrap();
        assert!(to_vec(gradient).unwrap().iter().any(|g| g.abs() > 1e-7));
    }
}

#[test]
fn selected_likelihood_only_evaluates_the_requested_target_branch() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let inputs = tape.inputs();
    let policy = policy_model(&record, &Device::flex().autodiff());
    let evaluations = policy
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: inputs.len() - 1,
            }],
            PackLimits::default(),
        )
        .unwrap();
    let row = inputs.len() - 1;
    let Observation::Structured(observation) = inputs[row].observation else {
        unreachable!()
    };
    let selected = observation
        .candidates
        .iter()
        .find(|c| c.enabled && matches!(c.action.argument, Some(ActionArgument::Entity(_))))
        .unwrap()
        .action;
    let likelihood = evaluations[0]
        .selected_log_probs(&[(row, selected)])
        .unwrap();
    assert_eq!(
        to_vec(likelihood.clone()).unwrap(),
        to_vec(evaluations[0].log_prob_many(row, &[selected]).unwrap()).unwrap()
    );
    let gradients = likelihood.backward();
    assert!(
        policy.entity_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
    assert!(
        policy.location_query.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
}

#[test]
fn conditional_kl_matches_semantic_keys_instead_of_candidate_positions() {
    let a = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity(0))),
    };
    let b = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity(1))),
    };
    let device = Device::flex().autodiff();
    let log_p = Tensor::from_data([0.25_f32.ln(), 0.75_f32.ln()], &device).set_require_grad(true);
    let p = ConditionalPolicyLaw::new(vec![a, b], log_p.clone()).unwrap();
    let q = ConditionalPolicyLaw::new(
        vec![b, a],
        Tensor::from_data([0.6_f32.ln(), 0.4_f32.ln()], &device),
    )
    .unwrap();
    let kl = p.kl(&q).unwrap();
    let expected = 0.25 * (0.25_f32 / 0.4).ln() + 0.75 * (0.75_f32 / 0.6).ln();
    assert!((to_vec(kl.clone()).unwrap()[0] - expected).abs() < 1e-6);
    let reverse = 0.4 * (0.4_f32 / 0.25).ln() + 0.6 * (0.6_f32 / 0.75).ln();
    assert!((to_vec(q.kl(&p).unwrap()).unwrap()[0] - reverse).abs() < 1e-6);
    let gradient = to_vec(log_p.grad(&kl.backward()).unwrap()).unwrap();
    for (actual, (p, q)) in gradient.into_iter().zip([(0.25, 0.4_f32), (0.75, 0.6)]) {
        assert!((actual - p * ((p / q).ln() + 1.)).abs() < 1e-6);
    }
}

#[test]
fn scalar_value_uses_factual_memory_without_successor_candidates() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let mut inputs = tape.inputs();
    let final_row = inputs.len() - 1;
    let mut terminal = inputs[final_row].observation.clone();
    let Observation::Structured(frame) = &mut terminal else {
        unreachable!()
    };
    Arc::make_mut(frame).candidates.clear();
    let config = serde_json::from_value(record.metadata).unwrap();
    let value =
        RecurrentLightweightValue::new(config, &mut ChaCha8Rng::seed_from_u64(29), &Device::flex())
            .unwrap();
    assert_eq!(
        RecurrentLightweightValue::parameter_count(value.config()).unwrap(),
        value.num_params()
    );
    let limits = PackLimits::default();
    let original = value
        .evaluate_histories(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: final_row,
            }],
            limits,
        )
        .unwrap();
    inputs[final_row].observation = &terminal;
    let selected = value
        .evaluate_selected(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: final_row,
            }],
            &[(0, final_row)],
            limits,
        )
        .unwrap();
    let mut state = value.initial_state().unwrap();
    for &input in &inputs {
        state = value.assimilate(&state, input, limits).unwrap();
    }
    let scalar = to_vec(value.value_from_state(&state).unwrap()).unwrap()[0];
    let selected = to_vec(selected).unwrap()[0];
    assert!((to_vec(original[0].clone()).unwrap()[final_row] - selected).abs() < 1e-6);
    assert!((scalar - selected).abs() < 1e-6);
}

#[test]
fn histogram_q_joins_requested_candidates_by_key_in_query_order() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let mut inputs = tape.inputs();
    let row = inputs.len() - 1;
    let mut reordered = inputs[row].observation.clone();
    let Observation::Structured(frame) = &mut reordered else {
        unreachable!()
    };
    let keys = frame
        .candidates
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.action)
        .collect::<Vec<_>>();
    let disabled = frame.candidates.iter().find(|c| !c.enabled).unwrap().action;
    Arc::make_mut(frame).candidates.reverse();
    let config: RecurrentLightweightConfig = serde_json::from_value(record.metadata).unwrap();
    let width = config.model.head_width;
    let q = RecurrentLightweightDistributionalQ::new(
        config,
        7,
        true,
        &mut ChaCha8Rng::seed_from_u64(29),
        &Device::flex().autodiff(),
    )
    .unwrap();
    assert_eq!(
        RecurrentLightweightDistributionalQ::parameter_count(q.config(), 7, true).unwrap(),
        q.num_params()
    );
    let limits = PackLimits::default();
    let histories = [StructuredHistoryInput {
        inputs: &inputs,
        learn_start: row,
    }];
    let query = [(0, row, keys[1]), (0, row, keys[0]), (0, row, keys[1])];
    let evaluated = q
        .evaluate_histories_at(&histories, &[(0, row)], limits)
        .unwrap();
    let prediction = evaluated.query(&query).unwrap();
    assert!(evaluated.query(&[(0, row, disabled)]).is_err());
    assert!(evaluated.query(&[(0, 0, keys[0])]).is_err());
    assert_eq!(prediction.logits.dims(), [3, 7]);
    assert_eq!(prediction.representation.dims(), [3, width]);
    let predicted = q.predict_representation(prediction.representation).unwrap();
    assert_eq!(predicted.dims(), [3, width]);
    let output = to_vec(prediction.logits.clone()).unwrap();
    assert_eq!(&output[..7], &output[14..]);
    assert_ne!(&output[..7], &output[7..14]);
    let gradients = prediction.logits.sum().backward();
    assert!(q.head.layers[0].weight.val().grad(&gradients).is_some());
    drop(evaluated);
    inputs[row].observation = &reordered;
    let reordered = q
        .evaluate_histories_at(
            &[StructuredHistoryInput {
                inputs: &inputs,
                learn_start: row,
            }],
            &[(0, row)],
            limits,
        )
        .unwrap()
        .query(&query)
        .unwrap();
    assert_eq!(to_vec(reordered.logits).unwrap(), output);
}

fn tensor_bits<const D: usize>(tensor: Tensor<D>) -> Vec<u32> {
    to_vec(tensor)
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn gradient_rows(model: &impl Module, gradients: &Gradients) -> Vec<Option<Vec<u32>>> {
    struct Rows<'a>(&'a Gradients, Vec<Option<Vec<u32>>>);
    impl ModuleVisitor for Rows<'_> {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            self.1.push(parameter.val().grad(self.0).map(tensor_bits));
        }
    }
    let mut visitor = Rows(gradients, Vec::new());
    model.visit(&mut visitor);
    visitor.1
}

#[test]
fn batched_complete_laws_match_rowwise_bits_and_gradients() {
    let Fixture {
        record, histories, ..
    } = policy_fixture();
    let tape = public_history(&histories[1]);
    let inputs = tape.inputs();
    let policy = policy_model(&record, &Device::flex().autodiff());
    let rows = [inputs.len() - 1, 0, 1];
    let run = |batched| {
        let evaluations = policy
            .evaluate_histories(
                &[StructuredHistoryInput {
                    inputs: &inputs,
                    learn_start: 0,
                }],
                PackLimits::default(),
            )
            .unwrap();
        let evaluation = &evaluations[0];
        let laws = if batched {
            evaluation.complete_laws(&rows).unwrap()
        } else {
            rows.iter()
                .map(|&row| evaluation.complete_law(row).unwrap())
                .collect()
        };
        let output: Vec<_> = laws
            .iter()
            .map(|law| (law.keys().to_vec(), tensor_bits(law.log_probs().clone())))
            .collect();
        let terms: Vec<_> = laws
            .iter()
            .map(|law| law.entropy() + law.log_probs().clone().sum() * 0.03125)
            .collect();
        let gradients = Tensor::cat(terms, 0).sum().backward();
        (output, gradient_rows(&policy, &gradients))
    };
    assert_eq!(run(false), run(true));
}
