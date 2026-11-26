#![cfg(feature = "neural")]

use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor},
};
use burn_pack::{Bytes, Reader, Scalar, Writer};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::core::stateful_on_policy::OnPolicyIdentity;
use rl::{
    Result,
    algorithms::{
        OptimizerConfig, StructuredTemporalActor, recurrent_reinforce::RecurrentReinforce,
        reinforce::ReinforceConfig,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::{BehaviorPublisher, HistoryInput, StatefulActor},
        stateful_on_policy::StatefulOnPolicyActor,
        structured::{
            ActionCandidate, EntityId, EntityRow, SpatialObservation, StructuredObservation,
        },
    },
    nn::{
        checkpoint::audit_adam_optimizer,
        distributions::Distribution,
        structured::{
            PackLimits, RecurrentLightweightConfig, RecurrentLightweightPolicy,
            RecurrentLightweightValue, StructuredHistoryInput,
        },
        to_vec,
    },
    train::{
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::sync::Arc;

#[path = "support/checkpoint_eq.rs"]
mod checkpoint_eq;

struct SingleDecision {
    observations: ObservationSpace,
    actions: ActionSpace,
    episodes: usize,
}

fn frame(entities: bool, terminal: bool) -> Observation {
    let ids = [EntityId { kind: 0, id: 1 }, EntityId { kind: 0, id: 2 }];
    let mut candidates = Vec::new();
    if !terminal && entities {
        candidates.extend(ids.iter().enumerate().map(|(row, &id)| ActionCandidate {
            action: ConditionalAction {
                operation: 0,
                argument: Some(ActionArgument::Entity(id)),
            },
            features: vec![row as f32],
            enabled: true,
        }));
    } else if !terminal {
        candidates.push(ActionCandidate {
            action: ConditionalAction {
                operation: 1,
                argument: None,
            },
            features: vec![0.],
            enabled: true,
        });
    }
    Observation::Structured(Arc::new(StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: Some(SpatialObservation {
            width: 1,
            height: 1,
            cells: vec![0.],
        }),
        entities: ids
            .into_iter()
            .enumerate()
            .map(|(row, id)| EntityRow {
                id,
                features: vec![row as f32],
            })
            .collect(),
        events: vec![],
        scalars: vec![0.],
        candidates,
    }))
}

impl Env for SingleDecision {
    fn observation_space(&self) -> ObservationSpace {
        self.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.actions
    }

    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.episodes += 1;
        Ok(EpisodeStart::fresh(frame(self.episodes == 1, false)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        frame(self.episodes == 1, false).check_action(&self.actions, action)?;
        Ok(Step {
            observation: frame(false, true),
            reward: 1f32.into(),
            terminated: true,
            truncated: false,
        })
    }
}

fn architecture() -> RecurrentLightweightConfig {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/structured_recurrent_policy_v1.json")).unwrap();
    let record: LearnerRecord = serde_json::from_value(fixture["record"].clone()).unwrap();
    serde_json::from_value(record.metadata).unwrap()
}

fn environment(architecture: &RecurrentLightweightConfig) -> SyncVectorEnv<SingleDecision> {
    let schema = &architecture.model.input;
    SyncVectorEnv::new(vec![SingleDecision {
        observations: ObservationSpace::Structured(Arc::new(schema.observation_schema())),
        actions: ActionSpace::Conditional(Arc::new(schema.action_grammar())),
        episodes: 0,
    }])
    .unwrap()
}

fn episodic() -> TemporalOnPolicyConfig {
    TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            episode_guard: Some(2),
            gae_lambda: None,
            normalize_advantages: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn baseline_free_structured_reinforce_restores_sparse_adam_history() {
    let architecture = architecture();
    let env = environment(&architecture);
    let build = |seed| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let policy = RecurrentLightweightPolicy::new(
            architecture.clone(),
            &mut rng,
            &Device::flex().autodiff(),
        )
        .unwrap()
        .freeze();
        RecurrentReinforce::new(
            policy,
            None,
            ReinforceConfig {
                baseline: false,
                ..Default::default()
            },
            seed,
        )
        .unwrap()
    };
    let mut learner = build(21);
    let mut restored = build(23);
    restored.restore(&learner.checkpoint().unwrap()).unwrap();
    assert_eq!(restored.updates(), 0);
    let mut trainer = TemporalOnPolicyTrainer::new(env, &learner, episodic(), 22).unwrap();
    for _ in 0..2 {
        assert_eq!(trainer.cycle(&mut learner).unwrap().updates, 1);
    }
    let saved = learner.checkpoint().unwrap();
    let adam = audit_adam_optimizer(&saved, "policy", "policy", None, 2).unwrap();
    assert!(adam.steps.values().any(|&step| step == 1));
    assert!(adam.steps.values().any(|&step| step == 2));

    // Each fixture step enables one operation. Its zero operation-logit
    // gradient still contributes an Adam step for the final head.
    let policy = learner.policy();
    let final_layer = policy.operation_head.layers.last().unwrap();
    let weight = final_layer.weight.id.val();
    let bias = final_layer.bias.as_ref().unwrap().id.val();
    assert_eq!(adam.steps.get(&weight), Some(&2));
    assert_eq!(adam.steps.get(&bias), Some(&2));

    restored.restore(&saved).unwrap();
    assert_eq!(restored.updates(), 2);
    assert_eq!(
        restored.behavior_actor(&(), 2).unwrap().behavior_identity(),
        learner.behavior_actor(&(), 2).unwrap().behavior_identity()
    );
    let mut lower_clock = saved.clone();
    let id = *adam.steps.iter().find(|&(_, &time)| time == 2).unwrap().0;
    change_adam_clock(&mut lower_clock, "policy", id, Some(1));
    let before = checkpoint_eq::normalized(&restored.checkpoint().unwrap());
    assert!(restored.restore(&lower_clock).is_err());
    assert!(checkpoint_eq::normalized(&restored.checkpoint().unwrap()) == before);
    for (id, time) in [(weight, Some(1)), (bias, None)] {
        let mut tampered = saved.clone();
        change_consistent_adam_clock(&mut tampered, "policy", id, time);
        assert!(
            restored.restore(&tampered).is_err(),
            "tampered final operation-head clock {time:?}"
        );
        assert!(checkpoint_eq::normalized(&restored.checkpoint().unwrap()) == before);
    }
}

fn change_adam_clock(record: &mut LearnerRecord, owner: &str, id: u64, time: Option<usize>) {
    let reader =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers[owner].clone())).unwrap();
    let scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    if time.is_none() {
        tensors.retain(|tensor| tensor.param_id != Some(id));
    }
    let mut writer = Writer::new(tensors);
    let prefix = format!("{id}.");
    let clock_key = format!("{id}.momentum.time");
    for (key, value) in scalars {
        if time.is_none() && key.starts_with(&prefix) {
            continue;
        }
        let value = if key == clock_key {
            Scalar::from(time.unwrap())
        } else {
            value
        };
        writer = writer.with_scalar(&key, value);
    }
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    record
        .optimizers
        .insert(owner.into(), writer.into_bytes().unwrap().to_vec());
}

fn change_consistent_adam_clock(
    record: &mut LearnerRecord,
    owner: &str,
    id: u64,
    time: Option<usize>,
) {
    change_adam_clock(record, owner, id, time);
    let clocks = record.metadata["optimizer_steps"][owner]
        .as_object_mut()
        .unwrap();
    if let Some(time) = time {
        clocks.insert(id.to_string(), time.into());
    } else {
        clocks.remove(&id.to_string());
        record.metadata["optimizer_owners"][owner]
            .as_array_mut()
            .unwrap()
            .retain(|entry| entry.as_u64() != Some(id));
    }
}

#[test]
fn structured_value_restore_requires_final_head_adam_state_after_every_step() {
    let architecture = architecture();
    let device = Device::flex().autodiff();
    let mut rng = ChaCha8Rng::seed_from_u64(54);
    let policy = RecurrentLightweightPolicy::new(architecture.clone(), &mut rng, &device).unwrap();
    let value = RecurrentLightweightValue::new(architecture.clone(), &mut rng, &device).unwrap();
    let mut learner =
        RecurrentReinforce::new(policy, Some(value), ReinforceConfig::default(), 56).unwrap();
    let mut trainer =
        TemporalOnPolicyTrainer::new(environment(&architecture), &learner, episodic(), 57).unwrap();
    for _ in 0..2 {
        trainer.cycle(&mut learner).unwrap();
    }
    let saved = learner.checkpoint().unwrap();
    let before = checkpoint_eq::normalized(&saved);
    let value = learner.value().unwrap();
    let final_layer = value.head.layers.last().unwrap();
    let cases = [
        (final_layer.weight.id.val(), Some(1)),
        (final_layer.bias.as_ref().unwrap().id.val(), None),
    ];
    for (id, time) in cases {
        let mut tampered = saved.clone();
        change_consistent_adam_clock(&mut tampered, "value", id, time);
        assert!(
            learner.restore(&tampered).is_err(),
            "tampered final value-head clock {time:?}"
        );
        assert!(checkpoint_eq::normalized(&learner.checkpoint().unwrap()) == before);
    }
}

#[test]
fn structured_reinforce_score_and_entropy_match_the_scalar_target_and_adam_gradient() {
    let architecture = architecture();
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let device = Device::flex().autodiff();
    let mut policy =
        RecurrentLightweightPolicy::new(architecture.clone(), &mut rng, &device).unwrap();
    // Route the public candidate feature through one query coordinate so the
    // finite difference has a reliable signal independent of random features.
    let [key_width, width] = policy.entity_key.layers[0].weight.val().dims();
    let mut key_weights = vec![0.; key_width * width];
    key_weights[architecture.model.target_width() * width] = 1.;
    policy.entity_key.layers[0].weight = Param::from_tensor(
        Tensor::<1>::from_data(key_weights.as_slice(), &device).reshape([key_width, width]),
    );
    let query_shape = policy.entity_query.layers[0].weight.val().dims();
    policy.entity_query.layers[0].weight = Param::from_tensor(Tensor::zeros(query_shape, &device));
    let mut query_bias = vec![0.; width];
    query_bias[0] = 0.7;
    policy.entity_query.layers[0].bias = Some(Param::from_tensor(Tensor::from_data(
        query_bias.as_slice(),
        &device,
    )));
    let value = RecurrentLightweightValue::new(architecture.clone(), &mut rng, &device).unwrap();
    let entropy = 0.17;
    let mut learner = RecurrentReinforce::new(
        policy,
        Some(value),
        ReinforceConfig {
            entropy,
            policy: OptimizerConfig {
                max_grad_norm: None,
                ..Default::default()
            },
            ..Default::default()
        },
        21,
    )
    .unwrap();
    let mut trainer =
        TemporalOnPolicyTrainer::new(environment(&architecture), &learner, episodic(), 22).unwrap();
    trainer.collect_tick(&learner).unwrap();
    let batch = trainer.seal(&learner).unwrap();
    let row = &batch.rows[0];
    let sample = &batch.sequences.samples[row.sample];
    let Action::Conditional(key) = &sample.steps[row.time].action else {
        panic!()
    };
    assert_eq!(batch.len(), 1);
    assert_eq!(row.return_, Some(1.));
    assert_eq!(key.operation, 0); // All available complete actions share this operation.
    let advantage = 1. - f64::from(row.value.unwrap());
    let inputs: Vec<_> = sample.exact_history_inputs().unwrap().collect();
    let histories = [StructuredHistoryInput {
        inputs: &inputs,
        learn_start: 0,
    }];
    let reference = |policy: &RecurrentLightweightPolicy| {
        let evaluation = policy
            .evaluate_histories(&histories, PackLimits::default())
            .unwrap();
        let branch = evaluation[0]
            .conditional(row.time, key.operation)
            .unwrap()
            .unwrap();
        let chosen = branch
            .keys
            .iter()
            .position(|candidate| **candidate == **key)
            .unwrap();
        let Distribution::Categorical { logits, masks } = branch.distribution else {
            panic!()
        };
        assert_eq!(masks, [true, true]);
        let logits: Vec<_> = to_vec(logits).unwrap().into_iter().map(f64::from).collect();
        let log_z = logits.iter().map(|x| x.exp()).sum::<f64>().ln();
        let entropy_term = logits
            .iter()
            .map(|l| {
                let logp = l - log_z;
                -logp.exp() * logp
            })
            .sum::<f64>();
        -advantage * (logits[chosen] - log_z) - f64::from(entropy) * entropy_term
    };
    let policy = learner.policy();
    let bias = policy.entity_query.layers[0].bias.as_ref().unwrap();
    let param_id = bias.id.val();
    let base = to_vec(bias.val()).unwrap();
    let expected_loss = reference(&policy);
    let epsilon = 0.002;
    let evaluate = |index, delta: f32| {
        let mut changed = policy.clone();
        let mut weights = base.clone();
        weights[index] += delta;
        changed.entity_query.layers[0].bias = Some(Param::from_tensor(Tensor::from_data(
            weights.as_slice(),
            &policy.model_device(),
        )));
        reference(&changed)
    };
    let expected_gradient: Vec<_> = (0..base.len())
        .map(|i| (evaluate(i, epsilon) - evaluate(i, -epsilon)) / (2. * f64::from(epsilon)))
        .collect();
    assert!(expected_gradient.iter().any(|g| g.abs() > 1e-3));

    let report = trainer.update(&mut learner).unwrap();
    assert!((report.metrics["policy_loss"] - expected_loss).abs() < 3e-6);
    assert!((report.metrics["value_loss"] - advantage.powi(2)).abs() < 3e-6);
    let saved = learner.checkpoint().unwrap();
    let moment_name = format!("{param_id}.momentum.moment_1");
    let moments = Reader::from_bytes(Bytes::from_bytes_vec(saved.optimizers["policy"].clone()))
        .unwrap()
        .into_tensors()
        .unwrap();
    let first_moment = moments
        .into_iter()
        .find(|tensor| tensor.name == moment_name)
        .unwrap();
    let (_, _, _, _, bytes) = first_moment.into_parts().unwrap();
    let actual_gradient: Vec<_> = bytes
        .chunks_exact(4)
        .map(|raw| {
            // This is the first unclipped update with Adam's default beta1=0.9.
            f64::from(f32::from_le_bytes(raw.try_into().unwrap())) / (1. - f64::from(0.9_f32))
        })
        .collect();
    assert_eq!(actual_gradient.len(), expected_gradient.len());
    for (actual, expected) in actual_gradient.into_iter().zip(expected_gradient) {
        assert!(
            (actual - expected).abs() < 6e-5,
            "actual={actual} expected={expected}"
        );
    }
}

#[test]
fn rich_structured_behavior_draws_and_scores_the_recorded_complete_law() {
    let architecture = architecture();
    let device = Device::flex();
    let mut policy =
        RecurrentLightweightPolicy::new(architecture, &mut ChaCha8Rng::seed_from_u64(47), &device)
            .unwrap();
    // The operation logits are proportional to log(1 + candidate count).
    // The entity operation is likelier than wait but each of its targets is less likely.
    let [inputs, hidden] = policy.operation_head.layers[0].weight.val().dims();
    let mut count_weights = vec![0.; inputs * hidden];
    count_weights[(inputs - 1) * hidden] = 1.;
    policy.operation_head.layers[0].weight = Param::from_tensor(
        Tensor::<1>::from_data(count_weights.as_slice(), &device).reshape([inputs, hidden]),
    );
    let mut read_count = vec![0.; hidden];
    read_count[0] = 0.5;
    policy.operation_head.layers[1].weight = Param::from_tensor(
        Tensor::<1>::from_data(read_count.as_slice(), &device).reshape([hidden, 1]),
    );
    let query_shape = policy.entity_query.layers[0].weight.val().dims();
    policy.entity_query.layers[0].weight = Param::from_tensor(Tensor::zeros(query_shape, &device));

    let wait = ConditionalAction {
        operation: 1,
        argument: None,
    };
    let unavailable = ConditionalAction {
        operation: 2,
        argument: Some(ActionArgument::Location { x: 0, y: 0 }),
    };
    let mut observation = frame(true, false);
    let Observation::Structured(frame) = &mut observation else {
        panic!()
    };
    Arc::get_mut(frame).unwrap().candidates.extend([
        ActionCandidate {
            action: wait,
            features: vec![0.],
            enabled: true,
        },
        ActionCandidate {
            action: unavailable,
            features: vec![0.],
            enabled: false,
        },
    ]);

    let actor = StructuredTemporalActor::new(policy, None, 0).unwrap();
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &observation,
                previous: None,
            },
        )
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(48);
    let simple = actor
        .decide(&state, &observation, ActMode::Deterministic, &mut rng)
        .unwrap();
    let rich = actor
        .sample_on_policy(&state, &observation, ActMode::Deterministic, &mut rng)
        .unwrap();
    let expected = ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(EntityId { kind: 0, id: 1 })),
    };
    assert_eq!(
        rich.decision.action,
        Action::Conditional(Arc::new(expected))
    );
    assert_eq!(simple.action, rich.decision.action);
    assert_eq!(simple.log_prob, rich.decision.log_prob);
    let law = rich.distribution.conditional_outcomes().unwrap();
    assert_eq!(law.len(), 4);
    assert_eq!(
        law.iter()
            .find(|row| row.action == unavailable)
            .unwrap()
            .log_prob,
        None
    );
    let recorded = law
        .iter()
        .find(|row| row.action == expected)
        .unwrap()
        .log_prob
        .unwrap();
    assert_eq!(
        rich.decision.log_prob.unwrap().to_bits(),
        recorded.to_bits()
    );
    assert!(
        (f64::from(recorded).exp() - 3f64.sqrt() / (2. * (2f64.sqrt() + 3f64.sqrt()))).abs() < 1e-6
    );
    for _ in 0..16 {
        let sample = actor
            .sample_on_policy(&state, &observation, ActMode::Sample, &mut rng)
            .unwrap();
        let Action::Conditional(key) = sample.decision.action else {
            panic!()
        };
        let outcomes = sample.distribution.conditional_outcomes().unwrap();
        let expected = outcomes
            .iter()
            .find(|row| row.action == *key)
            .unwrap()
            .log_prob
            .unwrap();
        assert_eq!(
            sample.decision.log_prob.unwrap().to_bits(),
            expected.to_bits()
        );
    }
}
