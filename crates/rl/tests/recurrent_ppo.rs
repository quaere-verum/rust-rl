#![cfg(feature = "neural")]

use burn_pack::{Bytes, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::core::stateful_on_policy::OnPolicyIdentity;
use rl::{
    algorithms::{
        ppo::{PpoConfig, PpoUpdateSettings},
        recurrent_critic::RecurrentUpdateMode,
        recurrent_ppo::RecurrentPpo,
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::{Checkpoint, LearnerRecord},
        env::Observation,
        recording::RecordingKey,
        stateful::{BehaviorPublisher, HistoryInput, HistorySpec, StatefulActor, StatefulLearner},
        stateful_on_policy::{DistributionSnapshot, StatefulOnPolicyActor},
    },
    data::{
        temporal::{BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep},
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch, TemporalRow},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_actor::TemporalActor,
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec,
    },
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::collections::BTreeMap;

fn learner(seed: u64, epochs: usize) -> RecurrentPpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
    let value = TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap();
    let config = PpoConfig {
        epochs,
        minibatch_size: 3,
        target_kl: None,
        ..Default::default()
    };
    RecurrentPpo::new(policy, value, config, seed + 1).unwrap()
}

fn batch(learner: &RecurrentPpo) -> TemporalOnPolicyBatch<TemporalActor> {
    let behavior = learner.behavior_actor(&(), learner.revision()).unwrap();
    let actions: Vec<_> = (0..3)
        .map(|i| match behavior.policy().action_space() {
            ActionSpace::Discrete { .. } => Action::Discrete(i % 2),
            ActionSpace::Box { .. } => Action::Continuous(vec![[0.2f32, -0.4, 0.6][i].tanh()]),
            _ => Action::Continuous(vec![[0.2, -0.4, 0.6][i]]),
        })
        .collect();
    let latents: Vec<_> = (0..3)
        .map(|i| match behavior.policy().action_space() {
            ActionSpace::Box { .. } => Some(vec![[0.2, -0.4, 0.6][i]]),
            _ => None,
        })
        .collect();
    let mut sequences = SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 3,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            episode_id: 1,
            learn_start: 0,
            burn_start: 0,
            observations: [0.8, -0.3, 0.4, 0.]
                .map(|x| Observation::new(vec![x]))
                .to_vec(),
            steps: (0..3)
                .map(|i| TemporalStep {
                    action: actions[i].clone(),
                    reward: [0.2_f32, -0.3, 1.][i].into(),
                    terminated: i == 2,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision: learner.revision(),
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 3],
        }],
    };
    let tokens = vec![(0, 0), (0, 1), (0, 2)];
    let distribution = behavior.policy().distribution(&sequences, &tokens).unwrap();
    let logps = to_vec(distribution.log_prob(&actions, &latents).unwrap()).unwrap();
    let values = to_vec(
        behavior
            .value()
            .unwrap()
            .values(&sequences, &tokens)
            .unwrap(),
    )
    .unwrap();
    let rows = (0..3)
        .map(|i| {
            sequences.samples[0].steps[i].behavior.log_prob = Some(logps[i]);
            TemporalRow {
                key: RecordingKey {
                    episode_id: 1,
                    timestep: i,
                },
                sample: 0,
                time: i,
                log_prob: logps[i],
                latent: latents[i].clone(),
                distribution: DistributionSnapshot::from_distribution(&distribution, i).unwrap(),
                value: Some(values[i]),
                next_value: Some(if i == 2 { 0. } else { values[i + 1] }),
                advantage: Some([0.5, -0.7, 1.2][i]),
                return_: Some([0.8, -0.2, 1.][i]),
                discount_weight: (0.9f32 as f64).powf(i as f64) as f32,
            }
        })
        .collect();
    let result = TemporalOnPolicyBatch {
        version: 1,
        behavior,
        revision: learner.revision(),
        kind: TemporalBatchKind::Bootstrapped,
        gamma: 0.9,
        gae_lambda: Some(0.95),
        episode_count: 1,
        sequences,
        rows,
    };
    result.validate().unwrap();
    result
}

fn numerical_record(learner: &RecurrentPpo) -> serde_json::Value {
    serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
}

// Burn's optimizer serialization traverses a HashMap. After restore its packed
// offsets/order can change while every named tensor and scalar remains identical.
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
        "{context}: tensor names mismatch"
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
        "checkpoint metadata or architecture changed"
    );
    assert!(
        left_packs.keys().eq(right_packs.keys()),
        "checkpoint pack names changed"
    );
    for (name, bytes) in left_packs {
        same_pack(&bytes, &right_packs[&name], &name);
    }
}

#[test]
fn exact_ppo_matches_reference_updates_and_continuing_lane_memory() {
    matching_trace(TraceCase {
        seed: 201,
        delays: [17, 12],
        rollout: 8,
        window: 3,
        cycles: 4,
        resume: false,
    });
}

#[test]
#[ignore = "release multi-seed training study"]
fn long_memory_ppo_modes_have_identical_training_traces() {
    for horizon in [256, 1024] {
        for seed in [11, 29, 47] {
            let (episodes, successes) = matching_trace(TraceCase {
                seed,
                delays: [horizon - 1; 2],
                rollout: 64,
                window: 16,
                cycles: horizon / 32,
                resume: true,
            });
            assert_eq!(episodes, 4);
            eprintln!("matched horizon={horizon} seed={seed} successes={successes}/{episodes}");
        }
    }
}

struct TraceCase {
    seed: u64,
    delays: [usize; 2],
    rollout: usize,
    window: usize,
    cycles: usize,
    resume: bool,
}

fn matching_trace(case: TraceCase) -> (usize, usize) {
    let make_learner = || {
        let mut rng = ChaCha8Rng::seed_from_u64(case.seed);
        let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
        let policy = TemporalPolicy::new(spec.clone(), Some(3), &[4], &mut rng).unwrap();
        let value = TemporalValue::new(spec, Some(5), &[4], &mut rng).unwrap();
        let mut config = PpoConfig {
            epochs: 2,
            minibatch_size: 3 * case.window,
            target_kl: None,
            ..Default::default()
        };
        if case.resume {
            config.policy.max_grad_norm = None;
            config.value.max_grad_norm = None;
            RecurrentPpo::new_with_joint_gradient_clipping(
                policy,
                value,
                config,
                case.seed + 3,
                0.5,
                0.05,
            )
        } else {
            RecurrentPpo::new(policy, value, config, case.seed + 3)
        }
        .unwrap()
        .with_minibatch_advantage_normalization(true)
    };
    let mut reference = make_learner();
    let mut optimized = make_learner();
    optimized.restore(&reference.checkpoint().unwrap()).unwrap();
    optimized.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    let make_trainer = |learner: &RecurrentPpo| {
        let envs = vec![
            DelayedBit::new(case.delays[0], None, false, case.seed + 1).unwrap(),
            DelayedBit::new(case.delays[1], None, false, case.seed + 2).unwrap(),
        ];
        TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(envs).unwrap(),
            learner,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: case.rollout,
                    normalize_advantages: false,
                    ..Default::default()
                },
                sequence: SequenceConfig {
                    learn_len: case.window,
                    burn_in: 2,
                    n_step: 1,
                },
                ..Default::default()
            },
            case.seed + 4,
        )
        .unwrap()
    };
    let mut left = make_trainer(&reference);
    let mut right = make_trainer(&optimized);
    let collected = |run: &TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>| {
        let rows: Vec<_> = run
            .rollout()
            .fresh_recordings()
            .map(|row| {
                let tape = run.rollout().store().episode(row.key.episode_id).unwrap();
                (row, &tape.steps[row.key.timestep])
            })
            .collect();
        let episodes = rows.iter().filter(|(_, step)| step.terminated).count();
        let successes = rows
            .iter()
            .filter(|(_, step)| step.terminated && step.reward.total() == 1.)
            .count();
        (serde_json::to_vec(&rows).unwrap(), episodes, successes)
    };
    let mut episodes = 0;
    let mut successes = 0;
    for cycle in 1..=case.cycles {
        for _ in 0..case.rollout {
            left.collect_tick(&reference).unwrap();
            right.collect_tick(&optimized).unwrap();
        }
        let trace = collected(&left);
        assert!(
            trace == collected(&right),
            "rollout differs at cycle {cycle}"
        );
        episodes += trace.1;
        successes += trace.2;
        let expected = left.update(&mut reference).unwrap();
        let actual = right.update(&mut optimized).unwrap();
        assert_eq!(expected.updates, actual.updates);
        assert_eq!(expected.early_stopped, actual.early_stopped);
        for (key, value) in expected.metrics {
            assert_eq!(actual.metrics[&key].to_bits(), value.to_bits(), "{key}");
        }
        assert_eq!(left.counters(), right.counters());
        assert_eq!(
            left.collector().adoption_count(),
            right.collector().adoption_count()
        );
        assert_eq!(
            left.collector().compaction_count(),
            right.collector().compaction_count()
        );
        for lane in 0..2 {
            assert_eq!(
                left.collector().lane_adoptions(lane),
                right.collector().lane_adoptions(lane)
            );
            assert_eq!(
                left.collector().lane_episode(lane),
                right.collector().lane_episode(lane)
            );
            assert_eq!(
                left.collector().lane_timestep(lane),
                right.collector().lane_timestep(lane)
            );
            assert_eq!(
                left.collector().lane_revision(lane),
                right.collector().lane_revision(lane)
            );
            assert_eq!(
                serde_json::to_vec(&left.collector().lane_state(lane)).unwrap(),
                serde_json::to_vec(&right.collector().lane_state(lane)).unwrap()
            );
        }
        let mut actual_record = optimized.checkpoint().unwrap();
        actual_record
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("update_mode");
        same_record_contents(&reference.checkpoint().unwrap(), &actual_record);
        if case.resume {
            eprintln!(
                "trace delay={} seed={} cycle={cycle}/{} transitions={}",
                case.delays[0],
                case.seed,
                case.cycles,
                left.counters().env_steps
            );
            if cycle == case.cycles / 2 + 1 {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("trace-resume");
                save_run(&path, &optimized, &right).unwrap();
                optimized = make_learner();
                right = load_run(&path, &mut optimized).unwrap();
                assert_eq!(optimized.update_mode(), RecurrentUpdateMode::OptimizedExact);
            }
        }
    }
    let mut restored = make_learner();
    restored.restore(&optimized.checkpoint().unwrap()).unwrap();
    assert_eq!(restored.update_mode(), RecurrentUpdateMode::OptimizedExact);
    (episodes, successes)
}

#[test]
fn ppo_report_matches_stationary_scalar_surrogate_kl_and_entropy_reference() {
    let base = learner(70, 1);
    let mut config = base.config().clone();
    config.entropy = 0.13;
    let mut learner = RecurrentPpo::new(base.policy(), base.value(), config, 71).unwrap();
    learner
        .set_update_settings(PpoUpdateSettings {
            policy_learning_rate: 0.,
            value_learning_rate: 0.,
            clip: 0.2,
        })
        .unwrap();
    let input = batch(&learner);
    let mut entropy = 0f64;
    for row in &input.rows {
        let DistributionSnapshot::Categorical { logits, .. } = &row.distribution else {
            panic!()
        };
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let total: f64 = logits.iter().map(|&v| (v as f64 - maximum).exp()).sum();
        for &value in logits {
            let p = (value as f64 - maximum).exp() / total;
            entropy -= p * p.ln() / 3.;
        }
    }
    let report = learner.update(&input).unwrap();
    let expected = -(0.5 - 0.7 + 1.2) / 3. - 0.13 * entropy;
    assert!((report.metrics["policy_loss"] - expected).abs() < 1e-6);
    assert!(report.metrics["approx_kl"].abs() < 1e-7);
}

#[test]
fn recurrent_actor_and_critic_encoders_train_and_behavior_stays_frozen() {
    let mut learner = learner(71, 3);
    let batch = batch(&learner);
    let old_policy = to_vec(
        batch
            .behavior
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
    .unwrap();
    let old_value = to_vec(
        batch
            .behavior
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
    .unwrap();
    let frozen = serde_json::to_value(&batch.behavior).unwrap();
    let report = learner.update(&batch).unwrap();
    assert_eq!(report.updates, 3);
    assert_eq!(learner.revision(), 1);
    assert_eq!(learner.updates(), 3);
    assert_ne!(
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
                .val()
        )
        .unwrap(),
        old_policy
    );
    assert_ne!(
        to_vec(
            learner
                .value()
                .encoder
                .gru
                .as_ref()
                .unwrap()
                .gru
                .new_gate
                .input_transform
                .weight
                .val()
        )
        .unwrap(),
        old_value
    );
    assert!(
        serde_json::to_value(&batch.behavior).unwrap() == frozen,
        "frozen behavior changed"
    );
    let current = to_vec(
        learner
            .policy()
            .distribution(&batch.sequences, &batch.tokens())
            .unwrap()
            .log_prob(&batch.actions(), &batch.latents())
            .unwrap(),
    )
    .unwrap();
    assert_ne!(current, batch.old_log_probs());
}

#[test]
fn stale_foreign_and_malformed_batches_are_atomic_and_do_not_poison() {
    let mut learner = learner(72, 1);
    let fresh = batch(&learner);
    let before = numerical_record(&learner);
    let foreign = batch(&self::learner(99, 1));
    assert!(learner.update(&foreign).is_err());
    assert!(
        numerical_record(&learner) == before,
        "foreign batch mutated learner"
    );
    let mut invalid = fresh.clone();
    invalid.rows[0].advantage = Some(f32::NAN);
    assert!(learner.update(&invalid).is_err());
    assert!(
        numerical_record(&learner) == before,
        "malformed batch mutated learner"
    );
    learner.update(&fresh).unwrap();
    let after = numerical_record(&learner);
    assert!(learner.update(&fresh).is_err());
    assert!(
        numerical_record(&learner) == after,
        "stale batch mutated learner"
    );
    assert!(learner.behavior_actor(&(), 0).is_err());
    learner.update(&batch(&learner)).unwrap();
}

#[test]
fn checkpoint_restores_optimizer_rng_schedule_and_future_updates() {
    let mut first = learner(73, 2).with_minibatch_advantage_normalization(true);
    first.update(&batch(&first)).unwrap();
    first
        .set_update_settings(PpoUpdateSettings {
            policy_learning_rate: 0.001,
            value_learning_rate: 0.002,
            clip: 0.1,
        })
        .unwrap();
    let saved = first.checkpoint().unwrap();
    let mut restored = learner(91, 1);
    restored.restore(&saved).unwrap();
    assert_eq!(first.update_settings(), restored.update_settings());
    assert_eq!(first.revision(), restored.revision());
    let next = batch(&first);
    let a = first.update(&next).unwrap();
    let b = restored.update(&next).unwrap();
    assert_eq!(a.updates, b.updates);
    assert_eq!(a.metrics, b.metrics);
    same_record_contents(
        &first.checkpoint().unwrap(),
        &restored.checkpoint().unwrap(),
    );
    let good = numerical_record(&restored);
    let mut malformed = saved;
    malformed.metadata["collection"]["revision"] = 999.into();
    assert!(restored.restore(&malformed).is_err());
    assert!(
        numerical_record(&restored) == good,
        "invalid restore mutated learner"
    );
}

#[test]
fn zero_effective_rates_advance_moments_and_revision_without_changing_models() {
    let mut learner = learner(74, 2);
    let before = learner.checkpoint().unwrap();
    learner
        .set_update_settings(PpoUpdateSettings {
            policy_learning_rate: 0.,
            value_learning_rate: 0.,
            clip: 0.,
        })
        .unwrap();
    learner.update(&batch(&learner)).unwrap();
    let after = learner.checkpoint().unwrap();
    assert!(
        serde_json::to_value(&before.models).unwrap()
            == serde_json::to_value(&after.models).unwrap(),
        "zero LR changed model parameters"
    );
    assert!(
        before.optimizers != after.optimizers,
        "zero LR did not advance optimizer moments"
    );
    assert_eq!(learner.updates(), 2);
    assert_eq!(learner.revision(), 1);
}

#[test]
fn joint_clipping_is_reported_and_restored() {
    let base = learner(75, 1);
    let mut config = base.config().clone();
    config.policy.max_grad_norm = None;
    config.value.max_grad_norm = None;
    let mut joint = RecurrentPpo::new_with_joint_gradient_clipping(
        base.policy(),
        base.value(),
        config,
        42,
        0.5,
        0.1,
    )
    .unwrap();
    let report = joint.update(&batch(&joint)).unwrap();
    assert!(report.metrics["joint_gradient_norm"].is_finite());
    let saved = joint.checkpoint().unwrap();
    let mut restored = learner(76, 1);
    restored.restore(&saved).unwrap();
    same_record_contents(
        &joint.checkpoint().unwrap(),
        &restored.checkpoint().unwrap(),
    );
}

#[test]
fn early_stop_precedes_entropy_rng_value_work_and_consumes_revision_once() {
    let mut rng = ChaCha8Rng::seed_from_u64(84);
    let spec = HistorySpec::new(
        1,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    )
    .unwrap();
    let policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
    let value = TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap();
    let mut config = PpoConfig {
        epochs: 1,
        minibatch_size: 3,
        entropy: 0.1,
        target_kl: None,
        ..Default::default()
    };
    config.policy.learning_rate = 0.1;
    let mut one = RecurrentPpo::new(policy, value, config, 85).unwrap();
    let mut record = one.checkpoint().unwrap();
    record.metadata["config"]["epochs"] = 4.into();
    record.metadata["config"]["target_kl"] = 1e-8.into();
    let mut stopped = learner(86, 1);
    stopped.restore(&record).unwrap();
    let input = batch(&one);
    one.update(&input).unwrap();
    let report = stopped.update(&input).unwrap();
    assert!(report.early_stopped);
    assert_eq!(report.updates, 1);
    assert_eq!(stopped.revision(), 1);
    let a = one.checkpoint().unwrap();
    let b = stopped.checkpoint().unwrap();
    assert!(
        serde_json::to_value(a.models).unwrap() == serde_json::to_value(b.models).unwrap(),
        "KL stop performed additional model updates"
    );
    assert!(
        a.optimizers.keys().eq(b.optimizers.keys()),
        "KL stop changed optimizer owners"
    );
    for (name, bytes) in &a.optimizers {
        same_pack(bytes, &b.optimizers[name], name);
    }
    assert_eq!(a.metadata["rng"], b.metadata["rng"]);
}

#[test]
fn prefix_tokens_are_context_and_only_fresh_window_rows_count() {
    let mut learner = learner(87, 2);
    let mut input = batch(&learner);
    input.sequences.config.learn_len = 2;
    input.sequences.samples[0].learn_start = 1;
    input.sequences.samples[0].burn_start = 1;
    input.sequences.samples[0].loss_mask[0] = false;
    input.rows.remove(0);
    input.validate().unwrap();
    let report = learner.update(&input).unwrap();
    assert_eq!(report.updates, 2);
    assert!(report.metrics.values().all(|v| v.is_finite()));
}

#[test]
fn short_padded_window_and_legal_action_mask_have_no_extra_loss_rows() {
    let base = learner(88, 1);
    let mut config = base.config().clone();
    config.minibatch_size = 6;
    let mut learner = RecurrentPpo::new(base.policy(), base.value(), config, 89).unwrap();
    learner
        .set_update_settings(PpoUpdateSettings {
            policy_learning_rate: 0.,
            value_learning_rate: 0.,
            clip: 0.2,
        })
        .unwrap();
    let mut input = batch(&learner);
    let mut short = input.sequences.samples[0].clone();
    short.episode_id = 2;
    short.observations.truncate(2);
    short.observations[0].as_dense_mut().unwrap().action_mask = Some(vec![true, false]);
    short.steps.truncate(1);
    short.steps[0].terminated = true;
    short.steps[0].behavior.log_prob = Some(0.);
    short.loss_mask.truncate(1);
    input.sequences.samples.push(short);
    let mut row = input.rows[0].clone();
    row.sample = 1;
    row.key.episode_id = 2;
    row.log_prob = 0.;
    row.next_value = Some(0.);
    let dist = input
        .behavior
        .policy()
        .distribution(&input.sequences, &[(1, 0)])
        .unwrap();
    row.distribution = DistributionSnapshot::from_distribution(&dist, 0).unwrap();
    input.rows.push(row);
    input.episode_count = 2;
    input.validate().unwrap();
    let report = learner.update(&input).unwrap();
    assert_eq!(report.updates, 1);
    assert!((report.metrics["policy_loss"] + 0.375).abs() < 1e-6);
}

#[test]
fn mismatched_recorded_distribution_is_rejected_before_training() {
    let mut learner = learner(90, 1);
    let mut input = batch(&learner);
    let DistributionSnapshot::Categorical { logits, .. } = &mut input.rows[0].distribution else {
        panic!()
    };
    // A constant logit shift leaves the sampled likelihood unchanged, so the
    // host-level likelihood validator accepts it. Frozen reconstruction must not.
    for logit in logits {
        *logit += 2.;
    }
    input.validate().unwrap();
    let before = numerical_record(&learner);
    assert!(learner.update(&input).is_err());
    assert!(
        numerical_record(&learner) == before,
        "forged distribution mutated learner"
    );
    learner.update(&batch(&learner)).unwrap();
}

#[test]
fn numerical_failure_requires_explicit_checkpoint_recovery() {
    let mut learner = learner(93, 1);
    let saved = learner.checkpoint().unwrap();
    let valid = batch(&learner);
    let mut overflow = valid.clone();
    overflow.rows[0].return_ = Some(f32::MAX);
    overflow.validate().unwrap();
    assert!(learner.update(&overflow).is_err());
    assert!(learner.checkpoint().is_err());
    assert!(learner.behavior_actor(&(), learner.revision()).is_err());
    assert!(learner.update(&valid).is_err());
    learner.restore(&saved).unwrap();
    learner.update(&valid).unwrap();
    assert_eq!(learner.revision(), 1);
}

#[test]
fn recording_audit_rejects_finite_forgery_without_sampling_or_advancing_memory() {
    let learner = learner(94, 1);
    let actor = learner.behavior_actor(&(), 0).unwrap();
    let observation = Observation::new(vec![0.7]);
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &observation,
                previous: None,
            },
        )
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(95);
    let sampled = actor
        .sample_on_policy(&state, &observation, ActMode::Sample, &mut rng)
        .unwrap();
    let prior_rng = rng.clone();
    let prior_state = state.clone();
    let prior_actor = serde_json::to_value(&actor).unwrap();
    let audit = |snapshot: &DistributionSnapshot, log_prob, value| {
        actor.validate_recording(
            &state,
            &observation,
            &sampled.decision.action,
            sampled.latent.as_deref(),
            snapshot,
            log_prob,
            value,
        )
    };
    audit(
        &sampled.distribution,
        sampled.decision.log_prob.unwrap(),
        sampled.value,
    )
    .unwrap();
    let mut forged = sampled.distribution.clone();
    let DistributionSnapshot::Categorical { logits, .. } = &mut forged else {
        panic!()
    };
    for value in logits {
        *value += 2.;
    }
    assert!(audit(&forged, sampled.decision.log_prob.unwrap(), sampled.value).is_err());
    assert!(
        audit(
            &sampled.distribution,
            sampled.decision.log_prob.unwrap(),
            sampled.value.map(|v| v + 1.)
        )
        .is_err()
    );
    assert!(
        audit(
            &sampled.distribution,
            sampled.decision.log_prob.unwrap() + 1.,
            sampled.value
        )
        .is_err()
    );
    assert!(
        audit(
            &sampled.distribution,
            sampled.decision.log_prob.unwrap(),
            None
        )
        .is_err()
    );
    assert_eq!(state, prior_state);
    assert_eq!(rng, prior_rng);
    assert!(
        serde_json::to_value(&actor).unwrap() == prior_actor,
        "recording audit changed frozen actor"
    );
}

#[test]
fn recording_audit_supports_baseline_free_behavior() {
    let learner = learner(96, 1);
    let actor = TemporalActor::new(learner.policy(), None, 0).unwrap();
    let observation = Observation::new(vec![0.2]);
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &observation,
                previous: None,
            },
        )
        .unwrap();
    let sampled = actor
        .sample_on_policy(
            &state,
            &observation,
            ActMode::Sample,
            &mut ChaCha8Rng::seed_from_u64(97),
        )
        .unwrap();
    assert!(sampled.value.is_none());
    actor
        .validate_recording(
            &state,
            &observation,
            &sampled.decision.action,
            sampled.latent.as_deref(),
            &sampled.distribution,
            sampled.decision.log_prob.unwrap(),
            None,
        )
        .unwrap();
    assert!(
        actor
            .validate_recording(
                &state,
                &observation,
                &sampled.decision.action,
                sampled.latent.as_deref(),
                &sampled.distribution,
                sampled.decision.log_prob.unwrap(),
                Some(0.)
            )
            .is_err()
    );
}

#[test]
fn frozen_getter_models_become_trainable_without_thawing_inference_copies() {
    let source = learner(98, 1);
    let frozen_policy = source.policy();
    let frozen_value = source.value();
    let policy_weight = frozen_policy
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .input_transform
        .weight
        .val();
    let value_weight = frozen_value
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .input_transform
        .weight
        .val();
    assert!(!policy_weight.is_autodiff() && !policy_weight.is_require_grad());
    assert!(!value_weight.is_autodiff() && !value_weight.is_require_grad());
    let policy_before = to_vec(policy_weight).unwrap();
    let value_before = to_vec(value_weight).unwrap();
    let mut trainable = RecurrentPpo::new(
        frozen_policy.clone(),
        frozen_value.clone(),
        source.config().clone(),
        99,
    )
    .unwrap();
    // Executes both actor and critic backward passes. Merely setting train mode
    // while retaining Burn's persistent freeze configuration fails here.
    let report = trainable.update(&batch(&trainable)).unwrap();
    assert_eq!(report.updates, 1);
    let policy_after = to_vec(
        trainable
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
    .unwrap();
    let value_after = to_vec(
        trainable
            .value()
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
    .unwrap();
    assert_ne!(
        policy_before, policy_after,
        "actor recurrent parameters did not receive gradients"
    );
    assert_ne!(
        value_before, value_after,
        "critic recurrent parameters did not receive gradients"
    );
    let policy_still_frozen = frozen_policy
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .input_transform
        .weight
        .val();
    let value_still_frozen = frozen_value
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .input_transform
        .weight
        .val();
    assert!(!policy_still_frozen.is_autodiff() && !policy_still_frozen.is_require_grad());
    assert!(!value_still_frozen.is_autodiff() && !value_still_frozen.is_require_grad());
    assert_eq!(to_vec(policy_still_frozen).unwrap(), policy_before);
    assert_eq!(to_vec(value_still_frozen).unwrap(), value_before);
}
