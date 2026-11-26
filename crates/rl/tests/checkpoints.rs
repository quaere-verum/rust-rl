#[cfg(feature = "neural")]
use rand::SeedableRng;
#[cfg(feature = "neural")]
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "neural")]
use rl::{
    algorithms::td3::{Td3, Td3Config},
    collect::{OffPolicyCollector, vector_env::SyncVectorEnv},
    core::{actor::ActMode, env::Env, learner::Learner},
    envs::lqg::{Lqg, LqgConfig},
    nn::{critics::scalar::QValue, policy::Deterministic},
    train::{
        checkpoint::{load_learner, load_run, save_learner, save_run},
        config::{Budget, OffPolicyConfig},
        trainer::OffPolicyTrainer,
    },
};
#[cfg(feature = "neural")]
fn build() -> (Td3, OffPolicyTrainer<SyncVectorEnv<Lqg>>) {
    let env = Lqg::new(
        LqgConfig::scalar(4, 1., 1., 1., 1., 1., 1., 0.1, 2.).unwrap(),
        1,
    )
    .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let p = Deterministic::new(2, env.action_space().clone(), &[4], &mut rng).unwrap();
    let q = QValue::new(2, 1, 1, &[4], &mut rng).unwrap();
    let q2 = QValue::new(2, 1, 1, &[4], &mut rng).unwrap();
    let l = Td3::new(p, q, q2, Td3Config::default(), 4).unwrap();
    let c = OffPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 3).unwrap();
    let t = OffPolicyTrainer::new(
        c,
        OffPolicyConfig {
            replay_capacity: 16,
            warmup_transitions: 0,
            n_step: 3,
            batch_size: 4,
            ..Default::default()
        },
        2,
    )
    .unwrap();
    (l, t)
}
#[test]
#[cfg(feature = "neural")]
fn full_run_restoration_reproduces_replay_rng_updates_and_delay() {
    let (mut learner, mut trainer) = build();
    trainer
        .train(&mut learner, Budget::vector_steps(7), None)
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    save_run(&path, &learner, &trainer).unwrap();
    let (mut restored, _) = build();
    let mut resumed: OffPolicyTrainer<SyncVectorEnv<Lqg>> = load_run(&path, &mut restored).unwrap();
    trainer
        .train(&mut learner, Budget::vector_steps(5), None)
        .unwrap();
    resumed
        .train(&mut restored, Budget::vector_steps(5), None)
        .unwrap();
    assert_eq!(learner.updates(), restored.updates());
    assert_eq!(learner.actor_updates(), restored.actor_updates());
    assert_eq!(
        serde_json::to_string(&trainer).unwrap(),
        serde_json::to_string(&resumed).unwrap()
    );
    let obs = vec![rl::core::env::Observation::new(vec![1., 0.])];
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    assert_eq!(
        learner
            .actor()
            .act(&obs, ActMode::Deterministic, &mut rng)
            .unwrap()[0],
        restored
            .actor()
            .act(&obs, ActMode::Deterministic, &mut rng)
            .unwrap()[0]
    );
}
#[test]
#[cfg(feature = "neural")]
fn incomplete_or_incompatible_checkpoint_is_rejected() {
    let (l, _) = build();
    let temp = tempfile::tempdir().unwrap();
    let p = temp.path().join("learner");
    save_learner(&p, &l).unwrap();
    std::fs::remove_file(p.join("COMPLETE")).unwrap();
    let (mut restored, _) = build();
    assert!(load_learner(&p, &mut restored).is_err());
}

#[test]
#[cfg(feature = "neural")]
fn streamed_run_keeps_schema_and_marker_checksum_and_rejects_corruption() {
    use rl::core::checkpoint::Checkpoint;
    let (mut learner, mut trainer) = build();
    trainer
        .train(&mut learner, Budget::vector_steps(3), None)
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    save_run(&path, &learner, &trainer).unwrap();
    let bytes = std::fs::read(path.join("checkpoint.json")).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["version"], 2);
    assert_eq!(value["kind"], "run");
    assert_eq!(value["runtime"], "burn-0.22-flex-f32");
    assert_eq!(
        value["learner"],
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
    let legacy_snapshot = serde_json::to_value(&trainer).unwrap();
    let legacy_snapshot: serde_json::Value =
        serde_json::from_slice(&serde_json::to_vec(&legacy_snapshot).unwrap()).unwrap();
    fn first_difference<'a>(
        path: &str,
        left: &'a serde_json::Value,
        right: &'a serde_json::Value,
    ) -> Option<String> {
        if left == right {
            return None;
        }
        if let (Some(a), Some(b)) = (left.as_object(), right.as_object()) {
            for (key, value) in a {
                if let Some(other) = b.get(key) {
                    if let Some(found) = first_difference(&format!("{path}.{key}"), value, other) {
                        return Some(found);
                    }
                } else {
                    return Some(format!("{path}.{key}: missing"));
                }
            }
        } else if let (Some(a), Some(b)) = (left.as_array(), right.as_array()) {
            for (index, (value, other)) in a.iter().zip(b).enumerate() {
                if let Some(found) = first_difference(&format!("{path}[{index}]"), value, other) {
                    return Some(found);
                }
            }
        }
        Some(format!("{path}: {left} != {right}"))
    }
    assert_eq!(
        first_difference("snapshot", &value["snapshot"], &legacy_snapshot),
        None
    );
    assert_eq!(
        value["snapshot_type"],
        std::any::type_name::<OffPolicyTrainer<SyncVectorEnv<Lqg>>>()
    );
    assert!(value["policy"].is_null());
    assert!(value["action_space"].is_null());
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ *byte as u64).wrapping_mul(0x100000001b3)
    });
    assert_eq!(
        std::fs::read_to_string(path.join("COMPLETE")).unwrap(),
        format!("{}:{hash:016x}", bytes.len())
    );
    let before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    let mut corrupted = bytes;
    corrupted[0] ^= 1;
    std::fs::write(path.join("checkpoint.json"), corrupted).unwrap();
    assert!(load_run::<OffPolicyTrainer<SyncVectorEnv<Lqg>>>(&path, &mut learner).is_err());
    assert_eq!(
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap(),
        before
    );
}

#[test]
#[cfg(feature = "neural")]
fn on_policy_records_round_trip_and_quantile_scale_is_persistent() {
    use rl::{objectives::quantile::QuantileLoss, prelude::*, train::checkpoint::Checkpoint};
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let obs = vec![Observation::new(vec![1.])];
    let actor = ActorCritic {
        policy: Policy::new(1, space.clone(), &[4], &mut rng).unwrap(),
        value: Value::new(1, &[4], &mut rng).unwrap(),
    };
    let mut o = RolloutBehavior::default();
    actor
        .sample_rollout_into((&obs).into(), &mut rng, &mut o)
        .unwrap();
    let mut batch = OnPolicyBatch {
        episode_count: 1,
        observations: obs,
        actions: o.actions,
        latents: o.latents,
        metadata: OnPolicyMetadata::Bootstrapped(BootstrappedBatch {
            log_probs: o.log_probs,
            values: o.values,
            advantages: vec![1.],
        }),
        returns: vec![1.],
        discount_weights: vec![1.],
    };
    macro_rules! check {
        ($l:expr) => {{
            let mut learner = $l;
            learner.update(&batch).unwrap();
            let record = learner.checkpoint().unwrap();
            learner.restore(&record).unwrap();
            assert_eq!(learner.checkpoint().unwrap().metadata, record.metadata);
        }};
    }
    check!(Ppo::new(actor.clone(), Default::default(), 1).unwrap());

    check!(Vmpo::new(actor.clone(), Default::default(), 1).unwrap());
    batch.metadata = OnPolicyMetadata::Episodic {
        baseline_values: Some(vec![0.]),
    };
    check!(Reinforce::new(actor, Default::default(), 1).unwrap());
    let p = Deterministic::new(1, space, &[4], &mut rng).unwrap();
    let q = QValue::new(1, 1, 4, &[4], &mut rng).unwrap();
    let mut c = D4pgConfig::new(4).unwrap();
    c.loss = QuantileLoss::Generalized {
        scale: 1.,
        ema: 0.5,
        minimum: 0.001,
    };
    let mut learner = D4pg::new(p, q, c, 1).unwrap();
    let item = rl::data::replay::ReplayItem {
        transition: rl::core::transition::Transition {
            observation: Observation::new(vec![1.]),
            action: Action::Continuous(vec![0.]),
            reward: 2.0_f32.into(),
            next_observation: Observation::new(vec![0.]),
            terminated: true,
            truncated: false,
        },
        discount: 0.,
    };
    learner.update(&ReplayBatch { items: vec![item] }).unwrap();
    let record = learner.checkpoint().unwrap();
    learner.restore(&record).unwrap();
    assert_eq!(learner.checkpoint().unwrap().metadata, record.metadata);
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("policy");
    rl::train::checkpoint::export_policy(&path, &learner.policy(), learner.actor().action_space())
        .unwrap();
    let restored = rl::train::checkpoint::load_policy(
        &path,
        &learner.policy(),
        learner.actor().action_space(),
    )
    .unwrap();
    let a = rl::nn::to_vec(
        restored
            .actions((&[Observation::new(vec![1.])]).into())
            .unwrap(),
    )
    .unwrap();
    let b = rl::nn::to_vec(
        learner
            .policy()
            .actions((&[Observation::new(vec![1.])]).into())
            .unwrap(),
    )
    .unwrap();
    assert_eq!(a, b);
}

#[cfg(feature = "neural")]
fn rewrite_envelope(path: &std::path::Path, change: impl FnOnce(&mut serde_json::Value)) {
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("checkpoint.json")).unwrap()).unwrap();
    change(&mut value);
    let bytes = serde_json::to_vec(&value).unwrap();
    write_checkpoint_with_checksum(path, &bytes);
}

#[cfg(feature = "neural")]
fn write_checkpoint_with_checksum(path: &std::path::Path, bytes: &[u8]) {
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ *b as u64).wrapping_mul(0x100000001b3)
    });
    std::fs::write(path.join("checkpoint.json"), bytes).unwrap();
    std::fs::write(
        path.join("COMPLETE"),
        format!("{}:{hash:016x}", bytes.len()),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "neural")]
fn run_with_duplicate_snapshot_field_is_rejected_before_learner_installation() {
    use rl::{Error, prelude::Checkpoint};
    let (mut learner, mut trainer) = build();
    trainer
        .train(&mut learner, Budget::vector_steps(5), None)
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("run");
    save_run(&path, &learner, &trainer).unwrap();
    let json = std::fs::read_to_string(path.join("checkpoint.json")).unwrap();
    let duplicate = json.replacen("\"snapshot\":{", "\"snapshot\":{\"update_credit\":0,", 1);
    assert_ne!(json, duplicate);
    write_checkpoint_with_checksum(&path, duplicate.as_bytes());

    let (mut restored, _) = build();
    let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
    let result = load_run::<OffPolicyTrainer<SyncVectorEnv<Lqg>>>(&path, &mut restored);
    assert!(matches!(result, Err(Error::Json(_))));
    assert_eq!(
        serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
        before
    );
}

#[test]
#[cfg(feature = "neural")]
fn invalid_snapshots_and_schema_one_are_rejected_before_learner_changes() {
    use rl::prelude::Checkpoint;
    for case in 0..9 {
        let (mut l, mut t) = build();
        t.train(&mut l, Budget::vector_steps(5), None).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("run");
        save_run(&path, &l, &t).unwrap();
        rewrite_envelope(&path, |e| match case {
            0 => e["version"] = 1.into(),
            1 => e["snapshot"]["collector"]["driver"]["observations"] = serde_json::json!([]),
            2 => e["snapshot"]["pending"]["n"] = 9.into(),
            3 => e["snapshot"]["pending"]["gamma"] = 0.1.into(),
            4 => e["snapshot"]["replay"]["rewards"] = serde_json::json!([]),
            5 => e["snapshot"]["update_credit"] = (-1).into(),
            6 => e["runtime"] = "wrong-runtime".into(),
            7 => e["kind"] = "learner".into(),
            _ => e["snapshot_type"] = "wrong-snapshot-type".into(),
        });
        let (mut restored, _) = build();
        let before = serde_json::to_value(restored.checkpoint().unwrap()).unwrap();
        let result = load_run::<OffPolicyTrainer<SyncVectorEnv<Lqg>>>(&path, &mut restored);
        assert!(result.is_err(), "case {case}");
        assert_eq!(
            serde_json::to_value(restored.checkpoint().unwrap()).unwrap(),
            before
        );
    }
}

#[test]
fn validated_lqg_snapshot_rejects_invalid_physical_clock() {
    use rl::envs::lqg::{Lqg, LqgConfig};

    let config = LqgConfig::scalar(3, 1., 1., 1., 1., 1., 1., 0.1, 2.).unwrap();
    let env = Lqg::new(config, 0).unwrap();
    let mut json = serde_json::to_value(&env).unwrap();
    assert!(serde_json::from_value::<Lqg>(json.clone()).is_ok());
    json["time"] = 100.into();
    assert!(serde_json::from_value::<Lqg>(json).is_err());
}
