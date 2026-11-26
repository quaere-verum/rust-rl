#![cfg(feature = "neural")]

//! Frozen numerical traces and explicit learner-only migration.
use burn_pack::{Bytes, DType, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        OptimizerConfig,
        recurrent_critic::{
            RecurrentCriticConfig, RecurrentDiscreteCritic, construct_policy_targets,
            construct_targets,
        },
        recurrent_mpo::{MpoBackup, RecurrentMpo, RecurrentMpoConfig},
    },
    collect::vector_env::SyncVectorEnv,
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, LearnerRecord},
        env::TimeLimit,
        learner::UpdateReport,
        stateful::{BehaviorPublisher, HistorySpec},
        temporal::TemporalExecutionMode,
    },
    data::temporal::{SequenceBatch, SequenceConfig},
    envs::delayed_bit::DelayedBit,
    nn::{
        checkpoint::audit_adam_optimizer, temporal::RecurrentQ,
        temporal_categorical_actor::CategoricalMixtureSettings, temporal_policy::TemporalPolicy,
    },
    train::{
        behavior::FixedBehavior,
        checkpoint::{load_learner, load_run, save_learner, save_run},
        recurrent::TemporalTrainer,
        temporal_config::TemporalOffPolicyConfig,
    },
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recurrent-mpo-pr32")
}
fn fixtures() -> (Value, SequenceBatch, SequenceBatch) {
    let expected =
        serde_json::from_slice(&std::fs::read(root().join("expected.json")).unwrap()).unwrap();
    let inputs: Value =
        serde_json::from_slice(&std::fs::read(root().join("inputs.json")).unwrap()).unwrap();
    (
        expected,
        serde_json::from_value(inputs["terminal"].clone()).unwrap(),
        serde_json::from_value(inputs["truncated"].clone()).unwrap(),
    )
}
fn learner(backup: MpoBackup) -> RecurrentMpo {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(
        spec.clone(),
        Some(4),
        &[],
        &mut ChaCha8Rng::seed_from_u64(682),
    )
    .unwrap();
    let q = TemporalPolicy::new(
        spec.clone(),
        Some(4),
        &[],
        &mut ChaCha8Rng::seed_from_u64(734),
    )
    .unwrap();
    let critic = RecurrentDiscreteCritic::new(
        RecurrentQ::new(q.encoder.gru.unwrap(), q.head).unwrap(),
        spec,
        RecurrentCriticConfig {
            target_interval: 2,
            gamma: 0.9,
            n_step: 1,
            ..Default::default()
        },
    )
    .unwrap();
    RecurrentMpo::new(
        policy,
        critic,
        RecurrentMpoConfig {
            backup,
            projection_kl: 1e-6,
            ..Default::default()
        },
    )
    .unwrap()
}

fn tensors(bytes: Vec<u8>) -> BTreeMap<String, (Option<u64>, Value)> {
    let reader = Reader::from_bytes(Bytes::from_bytes_vec(bytes)).unwrap();
    reader
        .into_tensors()
        .unwrap()
        .into_iter()
        .map(|tensor| {
            let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
            assert_eq!(dtype, DType::F32, "unexpected dtype for {name}");
            let words = bytes.chunks_exact(4);
            assert!(words.remainder().is_empty());
            let bits = words
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(bits.len(), shape.iter().product::<usize>());
            (name, (id, json!({"shape": shape, "f32_bits": bits})))
        })
        .collect()
}

// Burn-pack order and process-local parameter IDs are excluded from numerical
// comparison. Model paths are stable, and Adam moments resolve through owner IDs.
fn normalized(record: &LearnerRecord, updates: u64) -> Value {
    let mut envelope = serde_json::to_value(record).unwrap();
    let mut models = BTreeMap::new();
    let mut owners = BTreeMap::new();
    for (model, stored) in envelope["models"].as_object_mut().unwrap() {
        let mut names = BTreeMap::new();
        let mut values = BTreeMap::new();
        for (name, (id, value)) in tensors(serde_json::from_value(stored["bytes"].take()).unwrap())
        {
            assert!(names.insert(id.unwrap(), name.clone()).is_none());
            values.insert(name, value);
        }
        owners.insert(model.clone(), names);
        models.insert(model.clone(), values);
    }
    let mut adam = BTreeMap::new();
    for (optimizer, model) in [("actor", "actor"), ("critic", "online")] {
        let state = audit_adam_optimizer(record, optimizer, model, None, updates).unwrap();
        let mut groups = state
            .steps
            .iter()
            .map(|(id, step)| (owners[model][id].clone(), json!({"step": step})))
            .collect::<BTreeMap<_, _>>();
        for (key, (id, moment)) in tensors(record.optimizers[optimizer].clone()) {
            let (prefix, name) = key.split_once('.').unwrap();
            let id = id.unwrap();
            assert_eq!(prefix.parse::<u64>().unwrap(), id);
            groups.get_mut(&owners[model][&id]).unwrap()[name] = moment;
        }
        adam.insert(optimizer, groups);
    }
    json!({"models": models, "adam": adam})
}

fn readout(learner: &RecurrentMpo, batch: &SequenceBatch) -> Value {
    let probabilities = learner.policy_probabilities(batch).unwrap();
    let (online, target) = learner.critic().reconstructed_q_values(batch).unwrap();
    let actions = learner.critic().spec().action_width();
    let gamma = learner.critic().config().gamma;
    let targets = match learner.config().backup {
        MpoBackup::PolicyEvaluation => {
            construct_policy_targets(batch, &probabilities, &target, actions, gamma)
        }
        MpoBackup::DoubleQOptimality => construct_targets(batch, &online, &target, actions, gamma),
        MpoBackup::MaximumOptimality => construct_targets(batch, &target, &target, actions, gamma),
    }
    .unwrap()
    .into_iter()
    .map(|point| {
        json!({
            "row": point.row,
            "timestep": point.timestep,
            "action": point.action,
            "value": point.value,
            "bootstrap_index": point.bootstrap_index,
            "discount": point.discount,
        })
    })
    .collect::<Vec<_>>();
    json!({
        "actor_probabilities": probabilities,
        "online_q": online,
        "target_q": target,
        "td_targets": targets,
    })
}

fn readouts(learner: &RecurrentMpo, terminal: &SequenceBatch, truncated: &SequenceBatch) -> Value {
    json!({
        "terminal": readout(learner, terminal),
        "truncated": readout(learner, truncated),
    })
}

fn snapshot(learner: &RecurrentMpo, terminal: &SequenceBatch, truncated: &SequenceBatch) -> Value {
    json!({
        "updates": learner.updates(),
        "readouts": readouts(learner, terminal, truncated),
        "state": normalized(&learner.checkpoint().unwrap(), learner.updates()),
    })
}

fn assert_same_mpo_learning(expected: &UpdateReport, actual: &UpdateReport) {
    assert_eq!(expected.updates, actual.updates);
    assert!(expected.metrics.keys().eq(actual.metrics.keys()));
    for (name, value) in &expected.metrics {
        let measured = actual.metrics[name];
        if name.ends_with("_forward_tokens")
            || name.ends_with("_valid_history_tokens")
            || name == "actor_reconstruction_passes"
        {
            assert!(measured < *value, "work did not decrease for {name}");
        } else {
            assert_eq!(measured.to_bits(), value.to_bits(), "{name}");
        }
    }
}

#[test]
fn reused_mpo_passes_preserve_reference_update_and_restore_bits() {
    let (_, terminal, truncated) = fixtures();
    let mut mixed = terminal.clone();
    mixed.samples.push(terminal.samples[0].clone());
    mixed.samples.extend(truncated.samples.iter().cloned());
    mixed.validate().unwrap();
    for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
        let mut reference = learner(backup);
        let mut optimized = learner(backup);
        optimized.restore(&reference.checkpoint().unwrap()).unwrap();
        optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
        for (update, batch) in [&terminal, &truncated, &mixed].into_iter().enumerate() {
            let expected = reference.update(batch).unwrap();
            let actual = optimized.update(batch).unwrap();
            assert_same_mpo_learning(&expected, &actual);
            assert_eq!(
                snapshot(&reference, &terminal, &truncated),
                snapshot(&optimized, &terminal, &truncated)
            );
            if update == 1 {
                let saved = optimized.checkpoint().unwrap();
                optimized = learner(backup);
                optimized.restore(&saved).unwrap();
                assert_eq!(
                    optimized.update_mode(),
                    TemporalExecutionMode::OptimizedExact
                );
            }
        }
    }
}

#[test]
fn reused_mpo_passes_keep_adam_advance_when_all_proposals_reject() {
    let (_, terminal, _) = fixtures();
    let backup = MpoBackup::PolicyEvaluation;
    let mut reference = learner(backup);
    let mut optimized = learner(backup);
    let mut initial = reference.checkpoint().unwrap();
    initial.metadata["config"]["projection_kl"] = json!(1e-30);
    initial.metadata["config"]["actor"]["learning_rate"] = json!(0.1);
    let before = normalized(&initial, 0);
    reference.restore(&initial).unwrap();
    optimized.restore(&initial).unwrap();
    optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);

    let expected = reference.update(&terminal).unwrap();
    let actual = optimized.update(&terminal).unwrap();
    assert_same_mpo_learning(&expected, &actual);
    assert_eq!(actual.metrics["projection_evaluations"], 11.);
    assert_eq!(actual.metrics["projection_scale"], 0.);
    let after = normalized(&optimized.checkpoint().unwrap(), 1);
    assert_eq!(after, normalized(&reference.checkpoint().unwrap(), 1));
    assert_eq!(after["models"]["actor"], before["models"]["actor"]);
    let adam = after["adam"]["actor"].as_object().unwrap();
    assert!(!adam.is_empty());
    assert!(adam.values().all(|group| group["step"] == 1));
}

fn paired_mpo_replay(horizon: usize, seed: u64) {
    const UPDATES: usize = 16;
    for backup in [MpoBackup::PolicyEvaluation, MpoBackup::DoubleQOptimality] {
        let make_learner = || {
            let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let policy = TemporalPolicy::new(spec.clone(), Some(5), &[4], &mut rng).unwrap();
            let q = TemporalPolicy::new(spec.clone(), Some(7), &[5], &mut rng).unwrap();
            let critic = RecurrentDiscreteCritic::new(
                RecurrentQ::new(q.encoder.gru.unwrap(), q.head).unwrap(),
                spec,
                RecurrentCriticConfig {
                    gamma: 0.9,
                    n_step: 1,
                    target_interval: 2,
                    ..Default::default()
                },
            )
            .unwrap();
            RecurrentMpo::new(
                policy,
                critic,
                RecurrentMpoConfig {
                    actor: OptimizerConfig {
                        learning_rate: 0.003,
                        max_grad_norm: Some(0.005),
                    },
                    backup,
                    projection_kl: 1e-7,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let make_trainer = |model: &RecurrentMpo| {
            let envs = SyncVectorEnv::new(vec![
                TimeLimit::new(
                    DelayedBit::new(horizon - 1, None, false, seed + 1).unwrap(),
                    horizon + 1,
                )
                .unwrap(),
                TimeLimit::new(
                    DelayedBit::new(horizon + 3, None, false, seed + 2).unwrap(),
                    horizon - 3,
                )
                .unwrap(),
            ])
            .unwrap();
            let settings = CategoricalMixtureSettings { epsilon: 0.15 };
            TemporalTrainer::new(
                envs,
                model.behavior_actor(&settings, 0).unwrap(),
                TemporalOffPolicyConfig {
                    replay_capacity: (horizon + UPDATES) * 2,
                    batch_size: if horizon < 64 { 8 } else { 16 },
                    warmup_transitions: (horizon * 2) as u64,
                    n_step: 1,
                    gamma: 0.9,
                    updates_per_transition: 0.5,
                },
                SequenceConfig {
                    burn_in: 2,
                    learn_len: (horizon / 3).clamp(4, 16),
                    n_step: 1,
                },
                FixedBehavior(settings),
                seed + 3,
            )
            .unwrap()
        };
        let mut reference = make_learner();
        let mut optimized = make_learner();
        optimized.restore(&reference.checkpoint().unwrap()).unwrap();
        optimized.set_update_mode(TemporalExecutionMode::OptimizedExact);
        let mut left = make_trainer(&reference);
        let mut right = make_trainer(&optimized);
        for _ in 0..horizon {
            let a = left.step(&mut reference).unwrap();
            let b = right.step(&mut optimized).unwrap();
            assert_eq!(a.collection, b.collection);
            assert!(a.reports.is_empty() && b.reports.is_empty());
        }
        let mut duplicate_windows = 0;
        #[cfg(target_arch = "x86_64")]
        let mut coalesced = (false, false);
        for update in 1..=UPDATES {
            let a = left.step(&mut reference).unwrap();
            let b = right.step(&mut optimized).unwrap();
            assert_eq!(a.collection, b.collection);
            assert_eq!(a.reports.len(), 1);
            assert_eq!(b.reports.len(), 1);
            assert_same_mpo_learning(&a.reports[0], &b.reports[0]);
            #[cfg(target_arch = "x86_64")]
            {
                let dense = a.reports[0].metrics["online_critic_forward_tokens"] / 3.;
                let work = &b.reports[0].metrics;
                coalesced.0 |=
                    work["actor_forward_tokens"] < dense * work["actor_reconstruction_passes"];
                coalesced.1 |= work["target_critic_forward_tokens"] < dense;
            }
            assert_eq!(left.last_windows(), right.last_windows());
            let unique = left.last_windows().iter().collect::<BTreeSet<_>>();
            duplicate_windows += left.last_windows().len() - unique.len();
            assert_eq!(left.counters(), right.counters());
            assert_eq!(left.behavior_revision(), right.behavior_revision());
            assert_eq!(
                serde_json::to_value(left.store()).unwrap(),
                serde_json::to_value(right.store()).unwrap()
            );
            for lane in 0..2 {
                assert_eq!(
                    serde_json::to_vec(&left.collector().lane_state(lane)).unwrap(),
                    serde_json::to_vec(&right.collector().lane_state(lane)).unwrap(),
                );
                assert_eq!(
                    left.collector().lane_revision(lane),
                    right.collector().lane_revision(lane),
                );
            }
            assert_eq!(
                normalized(&reference.checkpoint().unwrap(), update as u64),
                normalized(&optimized.checkpoint().unwrap(), update as u64),
                "{backup:?} update {update}"
            );
            if update == UPDATES / 2 {
                let directory = tempfile::tempdir().unwrap();
                let path = directory.path().join("mpo-replay-parity");
                save_run(&path, &optimized, &right).unwrap();
                optimized = make_learner();
                right = load_run(&path, &mut optimized).unwrap();
                assert_eq!(
                    optimized.update_mode(),
                    TemporalExecutionMode::OptimizedExact
                );
            }
            eprintln!("MPO {backup:?} horizon={horizon} seed={seed} update={update}/{UPDATES}");
        }
        if horizon < 64 {
            assert!(duplicate_windows > 0);
        }
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("fma") {
            assert_eq!(
                coalesced,
                (true, true),
                "fixture must exercise replay coalescence"
            );
        }
        assert!(
            left.store()
                .episodes()
                .any(|e| e.steps.iter().any(|s| s.terminated))
        );
        assert!(
            left.store()
                .episodes()
                .any(|e| e.steps.iter().any(|s| s.truncated))
        );
    }
}

#[test]
fn optimized_mpo_matches_sampled_training_and_full_run_restore() {
    paired_mpo_replay(23, 11);
}

#[test]
#[ignore = "release long-memory replay study"]
fn optimized_mpo_matches_long_memory_training_across_seeds() {
    for horizon in [256, 1024] {
        for seed in [11, 29] {
            paired_mpo_replay(horizon, seed);
        }
    }
}

#[test]
fn explicit_import_preserves_original_parameters_moments_and_native_resume() {
    let (expected, terminal, truncated) = fixtures();
    for case in expected["scenarios"].as_array().unwrap() {
        let backup = serde_json::from_value(case["backup"].clone()).unwrap();
        let path = root().join(case["original_checkpoint_after_first"].as_str().unwrap());
        let mut model = learner(backup);
        let before = normalized(&model.checkpoint().unwrap(), 0);
        assert!(
            load_learner(&path, &mut model).is_err(),
            "ordinary restore silently accepted the old schema"
        );
        assert_eq!(normalized(&model.checkpoint().unwrap(), 0), before);
        model.import_pr32_checkpoint(&path).unwrap();
        assert_eq!(model.config().backup, backup);
        assert_eq!(model.checkpoint().unwrap().algorithm, "recurrent-mpo-v2");
        // Stored tensor and Adam bits transfer exactly, without host arithmetic.
        assert_eq!(
            normalized(&model.checkpoint().unwrap(), 1),
            case["first"]["after"]["state"]
        );
        let directory = tempfile::tempdir().unwrap();
        let current = directory.path().join("current");
        save_learner(&current, &model).unwrap();
        let mut resumed = learner(backup);
        load_learner(&current, &mut resumed).unwrap();
        let report = model.update(&truncated).unwrap();
        let resumed_report = resumed.update(&truncated).unwrap();
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::to_value(resumed_report).unwrap()
        );
        assert_eq!(
            snapshot(&model, &terminal, &truncated),
            snapshot(&resumed, &terminal, &truncated)
        );
    }
}
#[test]
#[cfg(all(target_os = "windows", target_arch = "x86_64", target_env = "msvc"))]
fn original_e_step_projection_backtracking_targets_and_adam_match_bit_for_bit() {
    let (expected, terminal, truncated) = fixtures();
    assert_eq!(
        expected["source_commit"],
        "7c59ee43c4d8d9c09d502c1cb30a96ff70860b41"
    );
    for case in expected["scenarios"].as_array().unwrap() {
        let mut model = learner(serde_json::from_value(case["backup"].clone()).unwrap());
        assert_eq!(
            readouts(&model, &terminal, &truncated),
            case["initial_readouts"]
        );
        for (stage, batch) in [("first", &terminal), ("second", &truncated)] {
            let report = model.update(batch).unwrap();
            assert_eq!(serde_json::to_value(report).unwrap(), case[stage]["report"]);
            assert_eq!(
                snapshot(&model, &terminal, &truncated),
                case[stage]["after"]
            );
        }
    }
}
