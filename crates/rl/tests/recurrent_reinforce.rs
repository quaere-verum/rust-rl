#![cfg(feature = "neural")]

use burn::{
    module::Param,
    tensor::{Tensor, TensorData},
};
use burn_pack::{Bytes, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::nn::temporal_actor::TemporalActor;
use rl::{
    algorithms::{recurrent_reinforce::RecurrentReinforce, reinforce::ReinforceConfig},
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord},
        env::Observation,
        recording::RecordingKey,
        stateful::{BehaviorPublisher, HistorySpec, StatefulActor, StatefulLearner},
        stateful_on_policy::DistributionSnapshot,
        temporal::TemporalExecutionMode,
    },
    data::{
        temporal::{BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRow},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec, vector,
    },
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::collections::BTreeMap;

fn learner(seed: u64, baseline: bool, space: ActionSpace) -> RecurrentReinforce {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(1, space).unwrap();
    let policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
    let value = baseline.then(|| TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap());
    RecurrentReinforce::new(
        policy,
        value,
        ReinforceConfig {
            baseline,
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}
fn discrete(seed: u64, baseline: bool) -> RecurrentReinforce {
    learner(seed, baseline, ActionSpace::Discrete { count: 2 })
}
fn record(learner: &RecurrentReinforce) -> serde_json::Value {
    serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
}

// Optimizer packs traverse a HashMap; restoring can change entry offsets/order.
// Cross-instance comparisons still require every named tensor/scalar byte exactly.
fn checkpoint_parts(record: &LearnerRecord) -> (serde_json::Value, BTreeMap<String, Vec<u8>>) {
    let mut envelope = serde_json::to_value(record).unwrap();
    let mut packs = BTreeMap::new();
    for (name, model) in envelope["models"].as_object_mut().unwrap() {
        packs.insert(
            format!("model {name}"),
            serde_json::from_value(model["bytes"].take()).unwrap(),
        );
    }
    for (name, optimizer) in envelope["optimizers"].as_object_mut().unwrap() {
        packs.insert(
            format!("optimizer {name}"),
            serde_json::from_value(optimizer.take()).unwrap(),
        );
    }
    (envelope, packs)
}

fn same_pack(left: &[u8], right: &[u8], context: &str) {
    let left = Reader::from_bytes(Bytes::from_bytes_vec(left.to_vec())).unwrap();
    let right = Reader::from_bytes(Bytes::from_bytes_vec(right.to_vec())).unwrap();
    assert!(
        left.scalars() == right.scalars(),
        "{context}: scalar mismatch"
    );
    assert!(
        left.metadata() == right.metadata(),
        "{context}: parameter paths mismatch"
    );
    let tensors = |reader: Reader| {
        reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                (name, (dtype, shape.to_vec(), id, bytes.to_vec()))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let left = tensors(left);
    let right = tensors(right);
    assert!(
        left.keys().eq(right.keys()),
        "{context}: named tensor mismatch"
    );
    for (name, (dtype, shape, id, bytes)) in left {
        let (other_dtype, other_shape, other_id, other_bytes) = &right[&name];
        assert_eq!(dtype, *other_dtype, "{context} {name}: dtype mismatch");
        assert_eq!(shape, *other_shape, "{context} {name}: shape mismatch");
        assert_eq!(id, *other_id, "{context} {name}: parameter ID mismatch");
        assert!(
            bytes == *other_bytes,
            "{context} {name}: exact tensor bytes differ; first difference {:?}",
            bytes.iter().zip(other_bytes).position(|(a, b)| a != b)
        );
    }
}

fn same_record_contents(left: &LearnerRecord, right: &LearnerRecord) {
    let (left_envelope, left_packs) = checkpoint_parts(left);
    let (right_envelope, right_packs) = checkpoint_parts(right);
    assert!(
        left_envelope == right_envelope,
        "checkpoint metadata or architecture differs"
    );
    assert!(
        left_packs.keys().eq(right_packs.keys()),
        "checkpoint pack names differ"
    );
    for (name, bytes) in left_packs {
        same_pack(&bytes, &right_packs[&name], &name);
    }
}

#[test]
fn checkpoint_restore_preserves_exact_named_state() {
    for baseline in [false, true] {
        let mut original = discrete(120, baseline);
        original.update(&complete_batch(&original)).unwrap();
        let saved = original.checkpoint().unwrap();
        let mut restored = discrete(121, !baseline);
        restored.restore(&saved).unwrap();
        let loaded = restored.checkpoint().unwrap();
        same_record_contents(&saved, &loaded);
    }
}

#[test]
fn exact_reinforce_preserves_complete_episode_updates_with_optional_baseline() {
    for baseline in [false, true] {
        let mut reference = learner(
            220,
            baseline,
            ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            },
        );
        let mut record = reference.checkpoint().unwrap();
        record.metadata["config"]["entropy"] = serde_json::json!(0.04);
        reference.restore(&record).unwrap();
        let mut optimized = discrete(221, !baseline);
        optimized.restore(&record).unwrap();
        optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
        for _ in 0..3 {
            let batch = complete_batch(&reference);
            let a = reference.update(&batch).unwrap();
            let b = optimized.update(&batch).unwrap();
            assert_eq!(a.updates, b.updates);
            assert!(a.metrics.keys().eq(b.metrics.keys()));
            for (name, value) in a.metrics {
                assert_eq!(value.to_bits(), b.metrics[&name].to_bits(), "{name}");
            }
            let mut actual = optimized.checkpoint().unwrap();
            actual
                .metadata
                .as_object_mut()
                .unwrap()
                .remove("update_mode");
            same_record_contents(&reference.checkpoint().unwrap(), &actual);
        }
    }
}

fn complete_batch(learner: &RecurrentReinforce) -> TemporalOnPolicyBatch<TemporalActor> {
    let behavior = learner.behavior_actor(&(), learner.revision()).unwrap();
    let mut samples = Vec::new();
    for (episode_id, rewards, observations) in [
        (3, vec![1., 2., 3.], vec![0.7, -0.4, 0.2, 0.]),
        (7, vec![-2., 4.], vec![-0.5, 0.9, 0.]),
    ] {
        let len = rewards.len();
        samples.push(SequenceSample {
            episode_id,
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            learn_start: 0,
            burn_start: 0,
            observations: observations
                .into_iter()
                .map(|x| Observation::new(vec![x]))
                .collect(),
            steps: rewards
                .into_iter()
                .enumerate()
                .map(|(i, reward)| {
                    let latent = [-0.6f32, 0.1, 0.5][i];
                    let action = match behavior.policy().action_space() {
                        ActionSpace::Discrete { .. } => Action::Discrete(i % 2),
                        ActionSpace::Box { .. } => Action::Continuous(vec![2. * latent.tanh()]),
                        _ => Action::Continuous(vec![latent]),
                    };
                    TemporalStep {
                        action,
                        reward: reward.into(),
                        terminated: i + 1 == len,
                        truncated: false,
                        behavior: BehaviorMetadata {
                            revision: learner.revision(),
                            log_prob: None,
                        },
                    }
                })
                .collect(),
            loss_mask: vec![true; len],
        });
    }
    let mut sequences = SequenceBatch {
        samples,
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 3,
            n_step: 1,
        },
    };
    let tokens: Vec<_> = sequences
        .samples
        .iter()
        .enumerate()
        .flat_map(|(sample, s)| (0..s.steps.len()).map(move |time| (sample, time)))
        .collect();
    let actions: Vec<_> = tokens
        .iter()
        .map(|&(s, t)| sequences.samples[s].steps[t].action.clone())
        .collect();
    let latents: Vec<_> = tokens
        .iter()
        .map(|&(_, t)| {
            matches!(behavior.policy().action_space(), ActionSpace::Box { .. })
                .then(|| vec![[-0.6, 0.1, 0.5][t]])
        })
        .collect();
    let distribution = behavior.policy().distribution(&sequences, &tokens).unwrap();
    let logps = to_vec(distribution.log_prob(&actions, &latents).unwrap()).unwrap();
    let values = behavior
        .value()
        .map(|value| to_vec(value.values(&sequences, &tokens).unwrap()).unwrap());
    // Independent MC arithmetic for gamma=1/2: [1+.5*2+.25*3, 2+.5*3, 3], [-2+.5*4,4].
    let returns = [2.75, 3.5, 3., 0., 4.];
    let rows = tokens
        .iter()
        .enumerate()
        .map(|(i, &(sample, time))| {
            let step = &mut sequences.samples[sample].steps[time];
            step.behavior.log_prob = Some(logps[i]);
            let value = values.as_ref().map(|values| values[i]);
            let next_value = values
                .as_ref()
                .map(|values| if step.terminated { 0. } else { values[i + 1] });
            TemporalRow {
                key: RecordingKey {
                    episode_id: sequences.samples[sample].episode_id,
                    timestep: time,
                },
                sample,
                time,
                log_prob: logps[i],
                latent: latents[i].clone(),
                distribution: DistributionSnapshot::from_distribution(&distribution, i).unwrap(),
                value,
                next_value,
                advantage: Some(returns[i] - value.unwrap_or(0.)),
                return_: Some(returns[i]),
                discount_weight: (0.5f32 as f64).powf(time as f64) as f32,
            }
        })
        .collect();
    let batch = TemporalOnPolicyBatch {
        version: 1,
        behavior,
        revision: learner.revision(),
        kind: TemporalBatchKind::Episodic,
        gamma: 0.5,
        gae_lambda: None,
        episode_count: 2,
        sequences,
        rows,
    };
    batch.validate().unwrap();
    batch
}

fn categorical_reference(
    batch: &TemporalOnPolicyBatch<TemporalActor>,
    entropy: f64,
    baseline: bool,
) -> f64 {
    let actions = batch.actions();
    let mut policy_loss = 0.;
    let mut entropy_sum = 0.;
    for (row, action) in batch.rows.iter().zip(actions) {
        let DistributionSnapshot::Categorical { logits, masks } = &row.distribution else {
            panic!()
        };
        let Action::Discrete(action) = action else {
            panic!()
        };
        let maximum = logits
            .iter()
            .zip(masks)
            .filter(|(_, valid)| **valid)
            .map(|(&value, _)| value as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let normalizer = logits
            .iter()
            .zip(masks)
            .filter(|(_, valid)| **valid)
            .map(|(&value, _)| (value as f64 - maximum).exp())
            .sum::<f64>();
        let logp = logits[action] as f64 - maximum - normalizer.ln();
        let advantage = row.return_.unwrap() as f64
            - if baseline {
                row.value.unwrap() as f64
            } else {
                0.
            };
        policy_loss -= logp * advantage * (batch.gamma as f64).powf(row.key.timestep as f64)
            / batch.episode_count as f64;
        for (&logit, &valid) in logits.iter().zip(masks) {
            if valid {
                let logp = logit as f64 - maximum - normalizer.ln();
                entropy_sum -= logp.exp() * logp;
            }
        }
    }
    policy_loss - entropy * entropy_sum / batch.rows.len() as f64
}

fn policy_weights(learner: &RecurrentReinforce) -> Vec<f32> {
    to_vec(
        learner
            .policy()
            .encoder
            .gru
            .as_ref()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight
            .val(),
    )
    .unwrap()
}
fn value_weights(learner: &RecurrentReinforce) -> Vec<f32> {
    to_vec(
        learner
            .value()
            .unwrap()
            .encoder
            .gru
            .as_ref()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight
            .val(),
    )
    .unwrap()
}

#[test]
fn complete_mc_discount_episode_reduction_and_unweighted_entropy_match_scalar_math() {
    let base = discrete(101, false);
    let mut config = base.config().clone();
    config.entropy = 0.17;
    let mut learner = RecurrentReinforce::new(base.policy(), None, config, 102).unwrap();
    let input = complete_batch(&learner);
    assert_eq!(input.returns().unwrap(), vec![2.75, 3.5, 3., 0., 4.]);
    let expected = categorical_reference(&input, 0.17, false);
    let report = learner.update(&input).unwrap();
    assert!((report.metrics["policy_loss"] - expected).abs() < 3e-6);
    assert_eq!(report.updates, 1);
    assert!(!report.metrics.contains_key("value_loss"));
    assert_eq!(learner.updates(), 1);
    assert_eq!(learner.revision(), 1);
    assert!(learner.value().is_none());
    let actor = learner.behavior_actor(&(), 1).unwrap();
    assert!(actor.initial_state().unwrap().value.is_none());
    let saved = learner.checkpoint().unwrap();
    assert_eq!(saved.models.len(), 1);
    assert_eq!(saved.optimizers.len(), 1);
    assert!(saved.metadata["value"].is_null());
    assert_eq!(saved.metadata["value_updates"], 0);
}

#[test]
fn collected_baseline_is_detached_and_value_fit_follows_the_one_actor_step() {
    let original = discrete(103, true);
    let mut slow = original.config().clone();
    slow.value.learning_rate = 1e-4;
    let mut fast = slow.clone();
    fast.value.learning_rate = 1e-2;
    let mut a = RecurrentReinforce::new(original.policy(), original.value(), slow, 104).unwrap();
    let mut b = RecurrentReinforce::new(original.policy(), original.value(), fast, 104).unwrap();
    let batch_a = complete_batch(&a);
    let batch_b = complete_batch(&b);
    let expected_policy = categorical_reference(&batch_a, 0., true);
    let expected_value = batch_a
        .rows
        .iter()
        .map(|row| (row.value.unwrap() as f64 - row.return_.unwrap() as f64).powi(2))
        .sum::<f64>()
        / batch_a.len() as f64;
    let report_a = a.update(&batch_a).unwrap();
    b.update(&batch_b).unwrap();
    assert!((report_a.metrics["policy_loss"] - expected_policy).abs() < 3e-6);
    assert!((report_a.metrics["value_loss"] - expected_value).abs() < 3e-6);
    assert_eq!(policy_weights(&a), policy_weights(&b));
    assert_ne!(value_weights(&a), value_weights(&b));
    assert_eq!(report_a.updates, 1);
    let saved = a.checkpoint().unwrap();
    assert_eq!(saved.metadata["updates"], 1);
    assert_eq!(saved.metadata["policy_updates"], 1);
    assert_eq!(saved.metadata["value_updates"], 1);
}

#[test]
fn independently_recurrent_encoders_train_for_discrete_and_bounded_gaussian_actions() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        },
    ] {
        let mut learner = learner(105, true, space);
        let batch = complete_batch(&learner);
        let frozen = serde_json::to_value(&batch.behavior).unwrap();
        let policy_before = policy_weights(&learner);
        let value_before = value_weights(&learner);
        assert_eq!(learner.update(&batch).unwrap().updates, 1);
        assert_ne!(policy_before, policy_weights(&learner));
        assert_ne!(value_before, value_weights(&learner));
        assert_eq!(serde_json::to_value(&batch.behavior).unwrap(), frozen);
    }
}

#[test]
fn recurrent_policy_gradient_matches_a_fixed_target_finite_difference() {
    let mut rng = ChaCha8Rng::seed_from_u64(106);
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(spec, Some(3), &[], &mut rng).unwrap();
    let mut learner = RecurrentReinforce::new(
        policy.clone(),
        None,
        ReinforceConfig {
            baseline: false,
            ..Default::default()
        },
        107,
    )
    .unwrap();
    let batch = complete_batch(&learner);
    let weights: Vec<_> = batch
        .rows
        .iter()
        .map(|row| row.return_.unwrap() * row.discount_weight)
        .collect();
    let likelihood = policy
        .distribution(&batch.sequences, &batch.tokens())
        .unwrap()
        .log_prob(&batch.actions(), &batch.latents())
        .unwrap();
    let loss = (likelihood * vector(&weights)).sum().neg() / batch.episode_count as f32;
    let parameter = policy
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .input_transform
        .weight
        .val();
    let gradient = to_vec(parameter.grad(&loss.backward()).unwrap()).unwrap();
    let index = gradient
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        .unwrap()
        .0;
    assert!(gradient[index].abs() > 1e-5);
    let evaluate = |delta: f32| {
        let mut changed = rl::nn::target::frozen_copy(&policy);
        let original = changed
            .encoder
            .gru
            .as_ref()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight
            .val();
        let mut values = to_vec(original.clone()).unwrap();
        values[index] += delta;
        changed
            .encoder
            .gru
            .as_mut()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(values, original.dims()),
            &original.device(),
        ));
        let logp = to_vec(
            changed
                .distribution(&batch.sequences, &batch.tokens())
                .unwrap()
                .log_prob(&batch.actions(), &batch.latents())
                .unwrap(),
        )
        .unwrap();
        -logp
            .iter()
            .zip(&weights)
            .map(|(&lp, &weight)| lp as f64 * weight as f64)
            .sum::<f64>()
            / batch.episode_count as f64
    };
    let epsilon = 0.002;
    let finite_difference = (evaluate(epsilon) - evaluate(-epsilon)) / (2. * epsilon as f64);
    assert!(
        (gradient[index] as f64 - finite_difference).abs() < 3e-4,
        "autodiff={} finite_difference={finite_difference}",
        gradient[index]
    );
    let before = policy_weights(&learner)[index];
    learner.update(&batch).unwrap();
    assert!((policy_weights(&learner)[index] - before) as f64 * finite_difference < 0.);
}

#[test]
fn incomplete_truncated_fragmented_foreign_and_wrong_episode_count_batches_are_pure_errors() {
    let mut learner = discrete(107, false);
    let batch = complete_batch(&learner);
    let before = record(&learner);
    for corruption in 0..5 {
        let mut bad = batch.clone();
        match corruption {
            0 => bad.episode_count += 1,
            1 => bad.sequences.samples[0].steps[2].terminated = false,
            2 => bad.sequences.samples[0].steps[2].truncated = true,
            3 => {
                bad.sequences.samples[0].learn_start = 1;
                bad.sequences.samples[0].burn_start = 1;
                bad.sequences.samples[0].loss_mask[0] = false;
                bad.rows.remove(0);
            }
            _ => bad.kind = TemporalBatchKind::Pathwise,
        }
        assert!(learner.update(&bad).is_err());
        assert_eq!(before, record(&learner));
    }
    let foreign = complete_batch(&discrete(108, false));
    assert!(learner.update(&foreign).is_err());
    assert_eq!(before, record(&learner));
    learner.update(&batch).unwrap();
    let after = record(&learner);
    assert!(learner.update(&batch).is_err());
    assert_eq!(after, record(&learner));
    assert!(learner.behavior_actor(&(), 0).is_err());
    learner.update(&complete_batch(&learner)).unwrap();
    assert_eq!(learner.revision(), 2);
}

#[test]
fn checkpoint_stages_optional_value_optimizer_rng_and_counter_invariants_atomically() {
    for baseline in [false, true] {
        let base = learner(
            109,
            baseline,
            ActionSpace::Box {
                low: vec![-2.],
                high: vec![2.],
            },
        );
        let mut config = base.config().clone();
        config.entropy = 0.07;
        let mut original =
            RecurrentReinforce::new(base.policy(), base.value(), config, 110).unwrap();
        original.update(&complete_batch(&original)).unwrap();
        let saved = original.checkpoint().unwrap();
        let mut restored = discrete(111, !baseline);
        restored.restore(&saved).unwrap();
        same_record_contents(
            &original.checkpoint().unwrap(),
            &restored.checkpoint().unwrap(),
        );
        let a = original.update(&complete_batch(&original)).unwrap();
        let b = restored.update(&complete_batch(&restored)).unwrap();
        assert_eq!(a.metrics, b.metrics);
        same_record_contents(
            &original.checkpoint().unwrap(),
            &restored.checkpoint().unwrap(),
        );
        for corruption in 0..5 {
            let mut invalid = original.checkpoint().unwrap();
            match corruption {
                0 => invalid.metadata["config"]["baseline"] = (!baseline).into(),
                1 => invalid.metadata["revision"] = 0.into(),
                2 => invalid.metadata["policy_updates"] = 99.into(),
                3 => invalid.metadata["value_updates"] = 99.into(),
                _ => {
                    invalid.optimizers.remove("policy");
                }
            }
            let before = record(&restored);
            assert!(restored.restore(&invalid).is_err());
            assert_eq!(before, record(&restored));
        }
    }
}

#[test]
fn baseline_configuration_requires_exact_optional_model_capability() {
    let no_value = discrete(113, false);
    let with_value = discrete(113, true);
    assert!(
        RecurrentReinforce::new(no_value.policy(), None, with_value.config().clone(), 114).is_err()
    );
    assert!(
        RecurrentReinforce::new(
            with_value.policy(),
            with_value.value(),
            no_value.config().clone(),
            114
        )
        .is_err()
    );
}

type Trainer = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>;
fn runtime_record(trainer: &Trainer) -> serde_json::Value {
    let mut value = serde_json::to_value(trainer).unwrap();
    value.as_object_mut().unwrap().remove("timings");
    value
}

#[test]
fn complete_episode_trainer_restores_partial_collection_into_a_new_learner() {
    for mode in [
        TemporalExecutionMode::Reference,
        TemporalExecutionMode::OptimizedExact,
    ] {
        complete_episode_trainer_restore(mode);
    }
}

fn complete_episode_trainer_restore(mode: TemporalExecutionMode) {
    let mut rng = ChaCha8Rng::seed_from_u64(115);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut original = RecurrentReinforce::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        Some(TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap()),
        ReinforceConfig::default(),
        116,
    )
    .unwrap();
    original.set_update_mode(mode);
    let config = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 2,
            episode_guard: Some(16),
            gamma: 0.9,
            gae_lambda: None,
            normalize_advantages: false,
        },
        sequence: SequenceConfig {
            burn_in: 0,
            learn_len: 8,
            n_step: 1,
        },
        capacity: 128,
        max_fresh: 64,
        ..Default::default()
    };
    let environment = SyncVectorEnv::new(vec![
        DelayedBit::new(3, Some(1), false, 1).unwrap(),
        DelayedBit::new(5, Some(2), false, 2).unwrap(),
    ])
    .unwrap();
    let mut a = Trainer::new(environment, &original, config, 117).unwrap();
    a.collect_tick(&original).unwrap();
    a.collect_tick(&original).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("partial-reinforce.json");
    save_run(&path, &original, &a).unwrap();
    let mut restored = discrete(118, false);
    let mut b: Trainer = load_run(&path, &mut restored).unwrap();
    assert_eq!(restored.update_mode(), mode);
    same_record_contents(
        &original.checkpoint().unwrap(),
        &restored.checkpoint().unwrap(),
    );
    assert_eq!(runtime_record(&a), runtime_record(&b));
    for _ in 0..2 {
        let report_a = a.cycle(&mut original).unwrap();
        let report_b = b.cycle(&mut restored).unwrap();
        assert_eq!(report_a.metrics, report_b.metrics);
        same_record_contents(
            &original.checkpoint().unwrap(),
            &restored.checkpoint().unwrap(),
        );
        assert_eq!(runtime_record(&a), runtime_record(&b));
    }
    assert_eq!(a.counters().episodes, 4);
    assert_eq!(a.counters().updates, 2);
}
