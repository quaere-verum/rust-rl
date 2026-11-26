#![cfg(feature = "neural")]

use burn_pack::{Bytes, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::nn::temporal_actor::TemporalActor;
use rl::{
    algorithms::{recurrent_vmpo::RecurrentVmpo, vmpo::VmpoConfig},
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        checkpoint::{Checkpoint, LearnerRecord},
        env::Observation,
        recording::RecordingKey,
        stateful::{BehaviorPublisher, HistorySpec, StatefulLearner},
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
        to_vec,
    },
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};
use std::collections::BTreeMap;

fn learner(seed: u64, epochs: usize, space: ActionSpace, width: usize) -> RecurrentVmpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(width, space).unwrap();
    RecurrentVmpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
        VmpoConfig {
            epochs,
            minibatch_size: 5,
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}
fn discrete(seed: u64, epochs: usize) -> RecurrentVmpo {
    learner(seed, epochs, ActionSpace::Discrete { count: 2 }, 1)
}
fn batch(learner: &RecurrentVmpo) -> TemporalOnPolicyBatch<TemporalActor> {
    let behavior = learner.behavior_actor(&(), learner.revision()).unwrap();
    let bounded = matches!(behavior.policy().action_space(), ActionSpace::Box { .. });
    let latents: Vec<_> = (0..5)
        .map(|i| bounded.then(|| vec![[0.2, -0.4, 0.6, -0.1, 0.5][i]]))
        .collect();
    let actions: Vec<_> = (0..5)
        .map(|i| {
            if bounded {
                Action::Continuous(vec![[0.2f32, -0.4, 0.6, -0.1, 0.5][i].tanh()])
            } else {
                Action::Discrete(i % 2)
            }
        })
        .collect();
    let mut sequences = SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 5,
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
            observations: [0.8, -0.3, 0.4, -0.7, 0.5, 0.]
                .map(|x| Observation::new(vec![x]))
                .to_vec(),
            steps: (0..5)
                .map(|i| TemporalStep {
                    action: actions[i].clone(),
                    reward: (0.2 * i as f32).into(),
                    terminated: i == 4,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision: learner.revision(),
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 5],
        }],
    };
    let tokens: Vec<_> = (0..5).map(|t| (0, t)).collect();
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
    let rows = (0..5)
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
                next_value: Some(if i == 4 { 0. } else { values[i + 1] }),
                advantage: Some([1., 2., 2., -1., 0.][i]),
                return_: Some(values[i] + 0.3 + i as f32 * 0.2),
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
fn raw(learner: &RecurrentVmpo) -> serde_json::Value {
    serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
}
fn same_record(left: &LearnerRecord, right: &LearnerRecord) {
    let parts = |record: &LearnerRecord| {
        let mut metadata = serde_json::to_value(record).unwrap();
        let mut packs: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for (name, model) in metadata["models"].as_object_mut().unwrap() {
            packs.insert(
                format!("model {name}"),
                serde_json::from_value(model["bytes"].take()).unwrap(),
            );
        }
        for (name, optimizer) in metadata["optimizers"].as_object_mut().unwrap() {
            packs.insert(
                format!("optimizer {name}"),
                serde_json::from_value(optimizer.take()).unwrap(),
            );
        }
        (metadata, packs)
    };
    let (a, ap) = parts(left);
    let (b, bp) = parts(right);
    assert!(a == b, "V-MPO checkpoint metadata mismatch");
    assert!(ap.keys().eq(bp.keys()), "V-MPO checkpoint owners mismatch");
    for (name, bytes) in ap {
        let a = Reader::from_bytes(Bytes::from_bytes_vec(bytes)).unwrap();
        let b = Reader::from_bytes(Bytes::from_bytes_vec(bp[&name].clone())).unwrap();
        assert!(a.scalars() == b.scalars(), "{name}: scalar mismatch");
        assert!(a.metadata() == b.metadata(), "{name}: path mismatch");
        let tensors = |reader: Reader| {
            reader
                .into_tensors()
                .unwrap()
                .into_iter()
                .map(|tensor| {
                    let (name, dtype, shape, id, data) = tensor.into_parts().unwrap();
                    (name, (dtype, shape.to_vec(), id, data.to_vec()))
                })
                .collect::<BTreeMap<_, _>>()
        };
        assert!(
            tensors(a) == tensors(b),
            "{name}: exact named tensor payload mismatch"
        );
    }
}

#[test]
fn selected_noncontiguous_tokens_match_hand_temperature_and_dual_objectives() {
    let mut learner = discrete(301, 1);
    let input = batch(&learner);
    let (eta, alpha) = learner.multipliers();
    let selected = [1, 2, 0]; // ceil(5/2), stable descending [2, 2, 1].
    let unnormalized = [1f64, 1., (-1. / eta as f64).exp()];
    let sum: f64 = unnormalized.iter().sum();
    let expected_policy = selected
        .iter()
        .zip(unnormalized)
        .map(|(&i, w)| -w / sum * input.rows[i].log_prob as f64)
        .sum::<f64>();
    let e = eta as f64 + 1e-6;
    let a = alpha as f64 + 1e-6;
    let expected_dual = e * 0.01 + e * ((2. + (-1. / e).exp()) / 3.).ln() + 2. + a * 0.1;
    let report = learner.update(&input).unwrap();
    assert!((report.metrics["policy_loss"] - expected_policy).abs() < 2e-6);
    assert!((report.metrics["dual_loss"] - expected_dual).abs() < 2e-6);
    assert!(report.metrics["kl_new_behavior"].abs() < 1e-7);
    let (new_eta, new_alpha) = learner.multipliers();
    assert!(
        new_eta > eta,
        "temperature should rise under concentrated weights"
    );
    assert!(
        new_alpha < alpha,
        "KL multiplier should fall with positive KL slack"
    );
    assert_eq!(learner.updates(), 1);
    assert_eq!(learner.revision(), 1);
}

#[test]
fn top_half_boundary_ties_choose_earliest_fresh_rows() {
    let mut learner = discrete(302, 1);
    let mut input = batch(&learner);
    for (row, advantage) in input.rows.iter_mut().zip([4., 4., 0., 4., 4.]) {
        row.advantage = Some(advantage);
    }
    let expected =
        -(input.rows[0].log_prob + input.rows[1].log_prob + input.rows[3].log_prob) as f64 / 3.;
    let report = learner.update(&input).unwrap();
    assert!((report.metrics["policy_loss"] - expected).abs() < 2e-6);
}

#[test]
fn both_recurrent_encoders_train_for_discrete_and_bounded_gaussian_policies() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    ] {
        let mut learner = learner(303, 3, space, 1);
        let input = batch(&learner);
        let before_p = to_vec(
            learner
                .policy()
                .encoder
                .gru
                .unwrap()
                .gru
                .new_gate
                .input_transform
                .weight
                .val(),
        )
        .unwrap();
        let before_v = to_vec(
            learner
                .value()
                .encoder
                .gru
                .unwrap()
                .gru
                .new_gate
                .input_transform
                .weight
                .val(),
        )
        .unwrap();
        let report = learner.update(&input).unwrap();
        assert_eq!(report.updates, 3);
        assert!(report.metrics.values().all(|x| x.is_finite()));
        assert_ne!(
            before_p,
            to_vec(
                learner
                    .policy()
                    .encoder
                    .gru
                    .unwrap()
                    .gru
                    .new_gate
                    .input_transform
                    .weight
                    .val()
            )
            .unwrap()
        );
        assert_ne!(
            before_v,
            to_vec(
                learner
                    .value()
                    .encoder
                    .gru
                    .unwrap()
                    .gru
                    .new_gate
                    .input_transform
                    .weight
                    .val()
            )
            .unwrap()
        );
    }
}

fn probabilities(distribution: &DistributionSnapshot) -> Vec<f64> {
    let DistributionSnapshot::Categorical { logits, .. } = distribution else {
        panic!()
    };
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let weights: Vec<_> = logits.iter().map(|&v| (v as f64 - maximum).exp()).collect();
    let sum: f64 = weights.iter().sum();
    weights.into_iter().map(|v| v / sum).collect()
}
#[test]
fn second_epoch_uses_current_to_behavior_kl_with_renewed_hidden_states() {
    let mut one = discrete(304, 1);
    let mut initial = one.checkpoint().unwrap();
    initial.metadata["config"]["policy"]["learning_rate"] = 0.15.into();
    one.restore(&initial).unwrap();
    initial.metadata["config"]["epochs"] = 2.into();
    let mut two = discrete(999, 1);
    two.restore(&initial).unwrap();
    let input = batch(&one);
    one.update(&input).unwrap();
    let now = one
        .policy()
        .distribution(&input.sequences, &input.tokens())
        .unwrap();
    let mut forward = 0.;
    let mut reverse = 0.;
    for (i, row) in input.rows.iter().enumerate() {
        let p = probabilities(&DistributionSnapshot::from_distribution(&now, i).unwrap());
        let q = probabilities(&row.distribution);
        for (&p, &q) in p.iter().zip(&q) {
            forward += p * (p / q).ln() / 5.;
            reverse += q * (q / p).ln() / 5.;
        }
    }
    assert!(
        (forward - reverse).abs() > 1e-6,
        "fixture KL must distinguish directions"
    );
    let report = two.update(&input).unwrap();
    assert!(
        (report.metrics["kl_new_behavior"] - forward).abs() < 2e-6,
        "KL direction/history mismatch: actual={}, forward={forward}, reverse={reverse}",
        report.metrics["kl_new_behavior"]
    );
}

#[test]
fn fresh_prefix_masks_and_stale_rejection_leave_inputs_atomic() {
    let mut learner = discrete(305, 1);
    let mut input = batch(&learner);
    input.sequences.samples[0].learn_start = 2;
    input.sequences.samples[0].burn_start = 2;
    input.sequences.samples[0].loss_mask = vec![false, false, true, true, true];
    input.rows.drain(0..2);
    input.validate().unwrap();
    let mut invalid = input.clone();
    invalid.rows[0].return_ = Some(f32::NAN);
    let before = raw(&learner);
    assert!(learner.update(&invalid).is_err());
    assert!(raw(&learner) == before, "invalid input mutated learner");
    learner.update(&input).unwrap();
    let after = raw(&learner);
    assert!(learner.update(&input).is_err());
    assert!(raw(&learner) == after, "stale input mutated learner");
}

#[test]
fn dual_schema_ownership_and_counter_corruption_restore_atomically() {
    let mut learner = discrete(306, 1);
    learner.update(&batch(&learner)).unwrap();
    let saved = learner.checkpoint().unwrap();
    let mut destination = discrete(307, 1);
    let original = raw(&destination);
    let mut swapped = saved.clone();
    swapped
        .optimizers
        .insert("dual".into(), saved.optimizers["policy"].clone());
    let mut counter = saved.clone();
    counter.metadata["dual_updates"] = 99.into();
    let mut schema = saved.clone();
    schema.algorithm = "vmpo".into();
    let mut unknown = saved.clone();
    unknown
        .models
        .insert("unexpected".into(), saved.models["multipliers"].clone());
    for corrupted in [swapped, counter, schema, unknown] {
        assert!(destination.restore(&corrupted).is_err());
        assert!(
            raw(&destination) == original,
            "invalid restore mutated learner"
        );
    }
    destination.restore(&saved).unwrap();
    same_record(&saved, &destination.checkpoint().unwrap());
    let next = batch(&learner);
    let a = learner.update(&next).unwrap();
    let b = destination.update(&next).unwrap();
    assert_eq!(a.metrics, b.metrics);
    same_record(
        &learner.checkpoint().unwrap(),
        &destination.checkpoint().unwrap(),
    );
}

#[test]
fn full_run_restore_preserves_future_rows_duals_and_optimizer_updates_exactly() {
    let mut learner = learner(308, 2, ActionSpace::Discrete { count: 2 }, 5);
    let environments = SyncVectorEnv::new(vec![
        DelayedBit::new(7, Some(3), false, 31).unwrap(),
        DelayedBit::new(9, Some(4), false, 32).unwrap(),
    ])
    .unwrap();
    let config = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 2,
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 1,
            learn_len: 2,
            n_step: 1,
        },
        ..Default::default()
    };
    let mut trainer = TemporalOnPolicyTrainer::new(environments, &learner, config, 309).unwrap();
    for _ in 0..2 {
        trainer.collect_tick(&learner).unwrap();
    }
    trainer.update(&mut learner).unwrap();
    trainer.collect_tick(&learner).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("vmpo-run");
    save_run(&path, &learner, &trainer).unwrap();
    let mut restored = self::learner(555, 1, ActionSpace::Discrete { count: 2 }, 5);
    let mut other: TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>> =
        load_run(&path, &mut restored).unwrap();
    for cycle in 0..3 {
        for _ in 0..if cycle == 0 { 1 } else { 2 } {
            trainer.collect_tick(&learner).unwrap();
            other.collect_tick(&restored).unwrap();
        }
        let a = trainer.seal(&learner).unwrap().clone();
        let b = other.seal(&restored).unwrap();
        assert!(
            a.rows == b.rows && a.sequences == b.sequences,
            "future V-MPO chronology differs"
        );
        let a = trainer.update(&mut learner).unwrap();
        let b = other.update(&mut restored).unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(learner.multipliers(), restored.multipliers());
        same_record(
            &learner.checkpoint().unwrap(),
            &restored.checkpoint().unwrap(),
        );
        for lane in 0..2 {
            assert_eq!(
                trainer.collector().lane_state(lane),
                other.collector().lane_state(lane)
            );
        }
    }
}

#[test]
fn exact_vmpo_matches_duals_optimizer_and_resumed_training() {
    #[cfg(feature = "probes")]
    use rl::profiling;

    let make_learner = |seed| learner(seed, 2, ActionSpace::Discrete { count: 2 }, 5);
    let mut reference = make_learner(311);
    let mut optimized = make_learner(312);
    optimized.restore(&reference.checkpoint().unwrap()).unwrap();
    optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
    let make_trainer = |learner: &RecurrentVmpo| {
        let environments = SyncVectorEnv::new(vec![
            DelayedBit::new(17, Some(7), false, 33).unwrap(),
            DelayedBit::new(25, Some(9), false, 34).unwrap(),
        ])
        .unwrap();
        TemporalOnPolicyTrainer::new(
            environments,
            learner,
            TemporalOnPolicyConfig {
                collection: OnPolicyConfig {
                    rollout_steps: 4,
                    normalize_advantages: false,
                    ..Default::default()
                },
                sequence: SequenceConfig {
                    burn_in: 2,
                    learn_len: 1,
                    n_step: 1,
                },
                ..Default::default()
            },
            313,
        )
        .unwrap()
    };
    let mut left = make_trainer(&reference);
    let mut right = make_trainer(&optimized);

    #[cfg(feature = "probes")]
    {
        profiling::reset();
        profiling::set_enabled(true);
    }
    for cycle in 0..7 {
        for tick in 0..4 {
            left.collect_tick(&reference).unwrap();
            right.collect_tick(&optimized).unwrap();
            if cycle == 3 && tick == 1 {
                let directory = tempfile::tempdir().unwrap();
                let path = directory.path().join("vmpo-exact-run");
                save_run(&path, &optimized, &right).unwrap();
                optimized = make_learner(314);
                right = load_run(&path, &mut optimized).unwrap();
                assert_eq!(
                    optimized.update_mode(),
                    TemporalExecutionMode::OptimizedExact
                );
            }
        }
        let a = left.seal(&reference).unwrap();
        let b = right.seal(&optimized).unwrap();
        assert!(
            a.rows == b.rows && a.sequences == b.sequences,
            "cycle {cycle}"
        );
        let a = left.update(&mut reference).unwrap();
        let b = right.update(&mut optimized).unwrap();
        assert_eq!(a.updates, b.updates);
        for (name, expected) in a.metrics {
            assert_eq!(b.metrics[&name].to_bits(), expected.to_bits(), "{name}");
        }
        assert_eq!(reference.multipliers(), optimized.multipliers());
        assert_eq!(left.counters(), right.counters());
        let mut actual = optimized.checkpoint().unwrap();
        assert_eq!(actual.metadata["update_mode"], "OptimizedExact");
        actual
            .metadata
            .as_object_mut()
            .unwrap()
            .remove("update_mode");
        same_record(&reference.checkpoint().unwrap(), &actual);
        for lane in 0..2 {
            assert_eq!(
                serde_json::to_vec(&left.collector().lane_state(lane)).unwrap(),
                serde_json::to_vec(&right.collector().lane_state(lane)).unwrap()
            );
            assert_eq!(
                left.collector().lane_adoptions(lane),
                right.collector().lane_adoptions(lane)
            );
        }
    }
    #[cfg(feature = "probes")]
    {
        profiling::set_enabled(false);
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("fma") {
            assert!(profiling::snapshot().iter().any(|sample| {
                sample.phase == profiling::Phase::DetachedPrefix && sample.calls > 0
            }));
        }
    }
}

#[test]
fn numerical_failure_requires_checkpoint_recovery() {
    let mut learner = discrete(310, 1);
    let saved = learner.checkpoint().unwrap();
    let valid = batch(&learner);
    let mut overflow = valid.clone();
    overflow.rows[0].return_ = Some(f32::MAX);
    assert!(learner.update(&overflow).is_err());
    assert!(learner.checkpoint().is_err());
    assert!(learner.behavior_actor(&(), learner.revision()).is_err());
    learner.restore(&saved).unwrap();
    learner.update(&valid).unwrap();
}
