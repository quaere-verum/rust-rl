#![cfg(all(feature = "diagnostics", feature = "neural"))]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{
        d4pg::{D4pg, D4pgConfig},
        ppo::{Ppo, PpoConfig},
        td3::{Td3, Td3Config},
    },
    collect::{OffPolicyCollector, OnPolicyCollector, vector_env::SyncVectorEnv},
    core::{
        actor::ActMode,
        checkpoint::Checkpoint,
        env::{Env, Observation},
        learner::Learner,
    },
    diagnostics::{
        observer::DiagnosticObserver,
        probes::{self, ProbeSet},
        schedule::*,
        schema::*,
        storage::*,
    },
    envs::lqg::{Lqg, LqgConfig},
    nn::{
        critics::scalar::{QValue, Value},
        observations,
        policy::{ActorCritic, Deterministic, DeterministicPolicy, Policy},
        probes::{CriticProbe, QuantileProbe, ScalarProbe, StateValueProbe},
    },
    train::{
        config::{Budget, OffPolicyConfig, OnPolicyConfig},
        trainer::{OffPolicyTrainer, OnPolicyTrainer},
    },
};

fn environment() -> Lqg {
    Lqg::new(
        LqgConfig::scalar(3, 1., 0.5, 1., 0.2, 1., 1., 0.1, 2.).unwrap(),
        10,
    )
    .unwrap()
}
fn ppo() -> Ppo {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    Ppo::new(
        ActorCritic {
            policy: Policy::new(2, environment().action_space().clone(), &[4], &mut rng).unwrap(),
            value: Value::new(2, &[4], &mut rng).unwrap(),
        },
        PpoConfig {
            epochs: 2,
            minibatch_size: 2,
            ..Default::default()
        },
        12,
    )
    .unwrap()
}
fn on_trainer() -> OnPolicyTrainer<SyncVectorEnv<Lqg>> {
    OnPolicyTrainer::new(
        OnPolicyCollector::new(SyncVectorEnv::new(vec![environment()]).unwrap(), 13).unwrap(),
        OnPolicyConfig {
            rollout_steps: 4,
            ..Default::default()
        },
    )
    .unwrap()
}
fn probe_set() -> ProbeSet {
    ProbeSet::new(
        "fixed",
        ProbePopulation::FixedReference,
        "specified LQG states",
        vec![
            Observation::new(vec![1., 0.]),
            Observation::new(vec![-0.5, 0.5]),
        ],
        None,
        environment().action_space().clone(),
    )
    .unwrap()
}
fn writer(path: &std::path::Path, algorithm: &str) -> Recorder {
    Recorder::create(
        path,
        RunManifest::new(
            "training",
            "acceptance",
            algorithm,
            2,
            environment().action_space().clone(),
            0.99,
        )
        .unwrap(),
        StorageConfig::default(),
    )
    .unwrap()
}
fn schedule() -> RecordingSchedule {
    RecordingSchedule {
        metrics: Some(Interval { first: 1, every: 1 }),
        probes: Some(Interval { first: 1, every: 1 }),
        monte_carlo: None,
        snapshots: None,
    }
}

#[test]
fn ppo_records_real_revisions_and_probing_preserves_the_entire_learner_state() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("ppo");
    let mut learner = ppo();
    let mut trainer = on_trainer();
    let probes = probe_set();
    let mut recorder = writer(&path, "ppo");
    recorder
        .append(&probes.record(RecordContext::new("training", 0)).unwrap())
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let mut revisions = vec![];
    let mut observer = DiagnosticObserver::new(
        recorder,
        schedule(),
        |learner: &Ppo, context: &RecordContext, due: Due, writer: &mut Recorder| {
            let before = serde_json::to_value(learner.checkpoint()?).unwrap();
            assert_eq!(context.identity.learner_update_revision, learner.updates());
            revisions.push(context.identity.learner_update_revision);
            if due.probes {
                let mut context = context.clone();
                context.probe_set = Some(probes.id().into());
                let policy = learner.policy();
                let value = learner.value();
                writer.append(&probes::policy(
                    context.clone(),
                    &policy,
                    &probes,
                    ActMode::Sample,
                    &mut rng,
                )?)?;
                writer.append(&probes::critic(
                    context.clone(),
                    StateValueProbe(&value).predict(probes.observations().into())?,
                    ActionCondition::StateValue,
                    None,
                )?)?;
                writer.append(&probes::activations(
                    context,
                    "policy",
                    &policy.network,
                    observations(probes.observations().into(), 2)?,
                    &["layers.0.pre", "layers.0.post"],
                )?)?;
            }
            assert_eq!(before, serde_json::to_value(learner.checkpoint()?).unwrap());
            Ok(())
        },
    );
    trainer
        .train_observed(&mut learner, Budget::vector_steps(8), None, &mut observer)
        .unwrap();
    observer.finish().unwrap();
    drop(observer);
    assert_eq!(revisions, [4, 8]);
    let reader = Recording::open(path, ReadLimits::default()).unwrap();
    for kind in [
        RecordKind::Metrics,
        RecordKind::Policy,
        RecordKind::Critic,
        RecordKind::Activations,
    ] {
        assert_eq!(
            reader.entries().iter().filter(|e| e.kind == kind).count(),
            2
        );
    }
    for index in 0..reader.entries().len() {
        reader.load(index).unwrap();
    }
}

#[test]
fn observed_and_unobserved_training_have_identical_future_losses_and_actions() {
    let mut learner_a = ppo();
    let mut learner_b = ppo();
    learner_b.restore(&learner_a.checkpoint().unwrap()).unwrap();
    let mut trainer_a = on_trainer();
    let mut trainer_b = trainer_a.clone();
    let set = probe_set();
    let mut rng = ChaCha8Rng::seed_from_u64(44);
    let mut observe = |learner: &Ppo, boundary: rl::train::observer::UpdateBoundary<'_>| {
        let context = RecordContext::new("training", boundary.counters.updates);
        probes::policy(context, &learner.policy(), &set, ActMode::Sample, &mut rng)?;
        Ok(())
    };
    trainer_a
        .train_observed(&mut learner_a, Budget::vector_steps(8), None, &mut observe)
        .unwrap();
    trainer_b
        .train(&mut learner_b, Budget::vector_steps(8), None)
        .unwrap();
    let a = trainer_a
        .train(&mut learner_a, Budget::vector_steps(4), None)
        .unwrap();
    let b = trainer_b
        .train(&mut learner_b, Budget::vector_steps(4), None)
        .unwrap();
    assert_eq!(
        a.last_update.unwrap().metrics,
        b.last_update.unwrap().metrics
    );
    assert_eq!(
        serde_json::to_value(trainer_a).unwrap(),
        serde_json::to_value(trainer_b).unwrap()
    );
    assert_semantic_checkpoint_equal(&learner_a, &learner_b);
}

fn assert_semantic_checkpoint_equal(left: &impl Checkpoint, right: &impl Checkpoint) {
    let left = serde_json::to_value(left.checkpoint().unwrap()).unwrap();
    let right = serde_json::to_value(right.checkpoint().unwrap()).unwrap();
    assert_eq!(left["metadata"], right["metadata"]);
    assert_eq!(left["algorithm"], right["algorithm"]);
    for category in ["models", "optimizers"] {
        assert_eq!(
            left[category]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            right[category]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>()
        );
        for (name, value) in left[category].as_object().unwrap() {
            let other = &right[category][name];
            let (value, other) = if category == "models" {
                assert_eq!(value["signature"], other["signature"]);
                assert_eq!(value["parameter_shapes"], other["parameter_shapes"]);
                (&value["bytes"], &other["bytes"])
            } else {
                (value, other)
            };
            let read = |value: &serde_json::Value| {
                let bytes = value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u8)
                    .collect();
                burn_pack::Reader::from_bytes(burn::tensor::Bytes::from_bytes_vec(bytes)).unwrap()
            };
            let a = read(value);
            let b = read(other);
            assert!(
                a.scalars() == b.scalars(),
                "{category}/{name}: scalar state differs"
            );
            assert!(
                a.metadata() == b.metadata(),
                "{category}/{name}: metadata differs"
            );
            let tensors = |reader: burn_pack::Reader| {
                reader
                    .into_tensors()
                    .unwrap()
                    .into_iter()
                    .map(|tensor| {
                        let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                        (name, (dtype, shape.to_vec(), id, bytes.to_vec()))
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
            let a = tensors(a);
            let b = tensors(b);
            assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
            for (key, value) in a {
                assert!(
                    Some(&value) == b.get(&key),
                    "{category}/{name}/{key}: tensor state differs"
                );
            }
        }
    }
}

#[test]
fn td3_observation_preserves_future_training_state() {
    let build = || {
        let env = Lqg::new(
            LqgConfig::scalar(4, 1., 1., 1., 1., 1., 1., 0.1, 2.).unwrap(),
            1,
        )
        .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let learner = Td3::new(
            Deterministic::new(2, env.action_space().clone(), &[4], &mut rng).unwrap(),
            QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
            QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
            Td3Config::default(),
            4,
        )
        .unwrap();
        let trainer = OffPolicyTrainer::new(
            OffPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 3).unwrap(),
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
        (learner, trainer)
    };
    let (mut learner_a, mut trainer_a) = build();
    let (mut learner_b, _) = build();
    learner_b.restore(&learner_a.checkpoint().unwrap()).unwrap();
    let mut trainer_b = trainer_a.clone();
    assert_semantic_checkpoint_equal(&learner_a, &learner_b);
    assert_eq!(
        serde_json::to_value(&trainer_a).unwrap(),
        serde_json::to_value(&trainer_b).unwrap()
    );
    let set = probe_set();
    let mut diagnostic_rng = ChaCha8Rng::seed_from_u64(44);
    let rng_before = serde_json::to_value(&diagnostic_rng).unwrap();
    let mut observation_count = 0;
    let mut observe = |learner: &Td3, boundary: rl::train::observer::UpdateBoundary<'_>| {
        assert_eq!(boundary.counters.updates, learner.updates());
        let mut context = RecordContext::new("training", learner.updates());
        context.probe_set = Some(set.id().into());
        let policy = learner.policy();
        let actions = policy.actions(set.observations().into())?;
        let policy_chunk = probes::deterministic_policy(context.clone(), &policy, &set)?;
        assert!(matches!(policy_chunk.record, Record::Policy(_)));
        assert_eq!(policy_chunk.arrays["actions"].shape(), [2, 1]);
        // Sampled actor inspection owns its randomness, independently of collection
        // and TD3's target-smoothing RNG. Deterministic policy probes need no RNG.
        let sampled =
            learner
                .actor()
                .act(set.observations(), ActMode::Sample, &mut diagnostic_rng)?;
        assert_eq!(sampled.len(), 2);
        for (role, critic) in [
            ("online-first", learner.critic()),
            ("online-second", learner.second_critic()),
            ("target-first", learner.target_critic()),
            ("target-second", learner.target_second_critic()),
        ] {
            context.identity.model_role = role.into();
            let chunk = probes::critic(
                context.clone(),
                ScalarProbe::new(&critic)?.predict(set.observations().into(), actions.clone())?,
                ActionCondition::CurrentPolicy,
                Some(&probes::continuous_actions(actions.clone())?),
            )?;
            assert!(matches!(chunk.record, Record::Critic(_)));
            assert!(
                chunk
                    .arrays
                    .values()
                    .all(|a| !a.is_empty() && a.to_f64().iter().all(|v| v.is_finite()))
            );
        }
        context.identity.model_role = "online".into();
        let activations = probes::activations(
            context,
            "policy",
            &policy.network,
            observations(set.observations().into(), 2)?,
            &["layers.0.pre", "layers.0.post"],
        )?;
        assert!(matches!(activations.record, Record::Activations(_)));
        assert_eq!(activations.arrays.len(), 2);
        for array in activations.arrays.values() {
            assert_eq!(array.shape(), [2, 4]);
            assert!(array.as_f32()?.iter().all(|v| v.is_finite()));
        }
        observation_count += 1;
        Ok(())
    };
    trainer_a
        .train_observed(&mut learner_a, Budget::vector_steps(7), None, &mut observe)
        .unwrap();
    trainer_b
        .train(&mut learner_b, Budget::vector_steps(7), None)
        .unwrap();
    assert_eq!(observation_count, 6);
    assert_ne!(rng_before, serde_json::to_value(&diagnostic_rng).unwrap());
    assert_eq!((learner_a.updates(), learner_a.actor_updates()), (6, 3));
    assert!(trainer_a.replay().len() >= 4);
    let split = serde_json::to_value(&trainer_a).unwrap();
    assert!(split["pending"]["pending"][0]["len"].as_u64().unwrap() > 0);

    // Neither continuation has a diagnostic observer. One-step calls expose every
    // future update report while retaining pending n-step and delayed-update state.
    for _ in 0..5 {
        let a = trainer_a
            .train(&mut learner_a, Budget::vector_steps(1), None)
            .unwrap();
        let b = trainer_b
            .train(&mut learner_b, Budget::vector_steps(1), None)
            .unwrap();
        assert_eq!(a.counters, b.counters);
        assert!(a.last_update.is_some());
        assert_eq!(
            serde_json::to_value(a.last_update).unwrap(),
            serde_json::to_value(b.last_update).unwrap(),
            "future update report differs"
        );
        assert_eq!(
            serde_json::to_value(&trainer_a).unwrap(),
            serde_json::to_value(&trainer_b).unwrap(),
            "future replay, pending, RNG or counters differ"
        );
        assert_semantic_checkpoint_equal(&learner_a, &learner_b);
    }
    assert_eq!((learner_a.updates(), learner_a.actor_updates()), (11, 5));
    let mut action_rng_a = ChaCha8Rng::seed_from_u64(91);
    let mut action_rng_b = action_rng_a.clone();
    assert_eq!(
        learner_a
            .actor()
            .act(
                set.observations(),
                ActMode::Deterministic,
                &mut action_rng_a
            )
            .unwrap(),
        learner_b
            .actor()
            .act(
                set.observations(),
                ActMode::Deterministic,
                &mut action_rng_b
            )
            .unwrap()
    );
}

#[test]
fn d4pg_records_online_and_target_quantiles_at_current_policy_actions() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("d4pg");
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut learner = D4pg::new(
        Deterministic::new(2, environment().action_space().clone(), &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 3, &[4], &mut rng).unwrap(),
        D4pgConfig::new(3).unwrap(),
        8,
    )
    .unwrap();
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![environment()]).unwrap(), 9).unwrap(),
        OffPolicyConfig {
            replay_capacity: 32,
            batch_size: 2,
            warmup_transitions: 0,
            ..Default::default()
        },
        10,
    )
    .unwrap();
    let set = probe_set();
    let mut recorder = writer(&path, "d4pg");
    recorder
        .append(&set.record(RecordContext::new("training", 0)).unwrap())
        .unwrap();
    let mut observer = DiagnosticObserver::new(
        recorder,
        schedule(),
        |learner: &D4pg, context: &RecordContext, _: Due, writer: &mut Recorder| {
            let before = serde_json::to_value(learner.checkpoint()?).unwrap();
            let policy = learner.policy();
            let actions = policy.actions(set.observations().into())?;
            let mut context = context.clone();
            context.probe_set = Some(set.id().into());
            writer.append(&probes::deterministic_policy(
                context.clone(),
                &policy,
                &set,
            )?)?;
            for (role, critic) in [
                ("online", learner.critic()),
                ("target", learner.target_critic()),
            ] {
                context.identity.model_role = role.into();
                writer.append(&probes::critic(
                    context.clone(),
                    QuantileProbe::new(&critic, learner.grid())?
                        .predict(set.observations().into(), actions.clone())?,
                    ActionCondition::CurrentPolicy,
                    Some(&probes::continuous_actions(actions.clone())?),
                )?)?;
            }
            context.identity.model_role = "online".into();
            writer.append(&probes::activations(
                context,
                "policy",
                &policy.network,
                observations(set.observations().into(), 2)?,
                &["layers.0.post"],
            )?)?;
            assert_eq!(before, serde_json::to_value(learner.checkpoint()?).unwrap());
            Ok(())
        },
    );
    trainer
        .train_observed(
            &mut learner,
            Budget {
                updates: Some(3),
                ..Default::default()
            },
            None,
            &mut observer,
        )
        .unwrap();
    observer.finish().unwrap();
    let reader = Recording::open(path, ReadLimits::default()).unwrap();
    assert_eq!(
        reader
            .entries()
            .iter()
            .filter(|e| e.kind == RecordKind::Critic)
            .count(),
        6
    );
    for index in 0..reader.entries().len() {
        reader.load(index).unwrap();
    }
}

#[test]
fn md4pg_records_both_raw_moments_and_restores_model_only_inference_without_altering_training() {
    use rl::{
        algorithms::md4pg::{Md4pg, Md4pgConfig},
        diagnostics::models::*,
        nn::probes::MomentsProbe,
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("md4pg");
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let mut learner = Md4pg::new(
        Deterministic::new(2, environment().action_space().clone(), &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        QValue::new(2, 1, 1, &[4], &mut rng).unwrap(),
        Md4pgConfig::default(),
        32,
    )
    .unwrap();
    let mut trainer = OffPolicyTrainer::new(
        OffPolicyCollector::new(SyncVectorEnv::new(vec![environment()]).unwrap(), 33).unwrap(),
        OffPolicyConfig {
            replay_capacity: 32,
            batch_size: 2,
            warmup_transitions: 0,
            ..Default::default()
        },
        34,
    )
    .unwrap();
    let set = probe_set();
    let mut recorder = writer(&path, "md4pg");
    recorder
        .append(&set.record(RecordContext::new("training", 0)).unwrap())
        .unwrap();
    let mut observer = DiagnosticObserver::new(
        recorder,
        schedule(),
        |learner: &Md4pg, c: &RecordContext, _: Due, recorder: &mut Recorder| {
            let before = serde_json::to_value(learner.checkpoint()?)?;
            let mut c = c.clone();
            c.probe_set = Some(set.id().into());
            let policy = learner.policy();
            let actions = policy.actions(set.observations().into())?;
            recorder.append(&probes::deterministic_policy(c.clone(), &policy, &set)?)?;
            for (role, first, second) in [
                ("online", learner.critic(), learner.second_critic()),
                (
                    "target",
                    learner.target_critic(),
                    learner.target_second_critic(),
                ),
            ] {
                c.identity.model_role = role.into();
                let chunk = probes::critic(
                    c.clone(),
                    MomentsProbe::new(&rl::nn::critics::moments::Moments {
                        first: first.clone(),
                        second: second.clone(),
                    })?
                    .predict(set.observations().into(), actions.clone())?,
                    ActionCondition::CurrentPolicy,
                    Some(&probes::continuous_actions(actions.clone())?),
                )?;
                recorder.append(&chunk)?;
                let mut snapshot = SnapshotBuilder::new(
                    &format!("{role}-{}", learner.updates()),
                    c.clone(),
                    2,
                    environment().action_space().clone(),
                )?;
                snapshot.deterministic("policy", &policy)?;
                snapshot.moments("moments", &first, &second)?;
                let model = snapshot.finish()?;
                let loaded = ModelBundle::load(&model, ModelLimits::default())?;
                let mut request = InspectionRequest::new("policy", set.observations().to_vec());
                request.critics.push("moments".into());
                let restored = loaded.inspect(&request)?;
                let Record::Critic(CriticRecord {
                    representation:
                        CriticRepresentation::RawMoments {
                            mean,
                            raw_second: second,
                        },
                    ..
                }) = &chunk.record
                else {
                    panic!()
                };
                let restored = &restored.critics["moments"];
                let Record::Critic(CriticRecord {
                    representation:
                        CriticRepresentation::RawMoments {
                            mean: rm,
                            raw_second: rs,
                        },
                    ..
                }) = &restored.record
                else {
                    panic!()
                };
                assert_eq!(chunk.arrays[mean].to_f64(), restored.arrays[rm].to_f64());
                assert_eq!(chunk.arrays[second].to_f64(), restored.arrays[rs].to_f64());
                recorder.append(&model)?;
            }
            assert_eq!(before, serde_json::to_value(learner.checkpoint()?)?);
            Ok(())
        },
    );
    trainer
        .train_observed(
            &mut learner,
            Budget {
                updates: Some(3),
                ..Default::default()
            },
            None,
            &mut observer,
        )
        .unwrap();
    observer.finish().unwrap();
    let reader = Recording::open(&path, ReadLimits::default()).unwrap();
    assert_eq!(
        reader
            .entries()
            .iter()
            .filter(|e| e.kind == RecordKind::Critic)
            .count(),
        6
    );
    for i in 0..reader.entries().len() {
        reader.load(i).unwrap();
    }
}

#[test]
fn visited_probe_sampling_is_reproducible_and_invalid_live_inputs_are_rejected() {
    use rl::core::action::{Action, ActionSpace};
    let pool: Vec<_> = (0..20)
        .map(|i| Observation::new(vec![i as f32, 0.]))
        .collect();
    let mut a = ChaCha8Rng::seed_from_u64(5);
    let mut b = a.clone();
    let x = ProbeSet::visited(
        "visited",
        "uniform evaluation trajectory rows",
        &pool,
        5,
        environment().action_space().clone(),
        &mut a,
    )
    .unwrap();
    let y = ProbeSet::visited(
        "visited",
        "uniform evaluation trajectory rows",
        &pool,
        5,
        environment().action_space().clone(),
        &mut b,
    )
    .unwrap();
    assert_eq!(x.observations(), y.observations());
    assert_eq!(x.observations().len(), 5);
    assert!(
        ProbeSet::new(
            "bad",
            ProbePopulation::FixedReference,
            "fixture",
            vec![Observation::dense(vec![0.], Some(vec![false, false]))],
            None,
            ActionSpace::Discrete { count: 2 }
        )
        .is_err()
    );
    assert!(
        ProbeSet::new(
            "bad",
            ProbePopulation::FixedReference,
            "fixture",
            vec![Observation::new(vec![0., 0.])],
            Some(vec![Action::Continuous(vec![3.])]),
            environment().action_space().clone()
        )
        .is_err()
    );
    let history = Array::f32(vec![5, 3, 2], vec![0.; 30]).unwrap();
    let record = x
        .with_history("external_hidden", history)
        .unwrap()
        .record(RecordContext::new("training", 1))
        .unwrap();
    assert_eq!(record.arrays["history/external_hidden"].shape(), &[5, 3, 2]);
}

#[test]
fn categorical_probabilities_respect_masks_and_activation_statistics_have_independent_values() {
    use rl::core::action::ActionSpace;
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let set = ProbeSet::new(
        "masked",
        ProbePopulation::FixedReference,
        "single allowed action",
        vec![Observation::dense(vec![1.], Some(vec![false, true, false]))],
        None,
        ActionSpace::Discrete { count: 3 },
    )
    .unwrap();
    let policy = Policy::new(1, set.action_space().clone(), &[2], &mut rng).unwrap();
    let recorded = probes::policy(
        RecordContext::new("training", 0),
        &policy,
        &set,
        ActMode::Sample,
        &mut rng,
    )
    .unwrap();
    assert_eq!(
        recorded.arrays["probabilities"].as_f32().unwrap(),
        &[0., 1., 0.]
    );
    assert_eq!(recorded.arrays["actions"].as_u64().unwrap(), &[1]);
    assert_eq!(recorded.arrays["entropy"].as_f32().unwrap(), &[0.]);
    let summary = probes::activation_summary(&[-2., 0., 2., f32::NAN]);
    assert_eq!(summary.finite_count, 3);
    assert_eq!(summary.mean, Some(0.));
    assert!((summary.standard_deviation.unwrap() - (8.0f64 / 3.).sqrt()).abs() < 1e-12);
    assert_eq!(summary.zero_fraction, 0.25);
}

#[test]
fn observer_failure_commits_update_counts_and_restored_trainer_continues_call_identity() {
    use rl::train::observer::UpdateBoundary;
    let mut learner = ppo();
    let mut trainer = on_trainer();
    let mut fail =
        |_: &Ppo, _: UpdateBoundary<'_>| Err(rl::Error::State("observer failure".into()));
    assert!(
        trainer
            .train_observed(&mut learner, Budget::vector_steps(4), None, &mut fail)
            .is_err()
    );
    assert_eq!(trainer.counters().updates, 4);
    assert_eq!(trainer.counters().vector_steps, 4);
    let mut trainer: OnPolicyTrainer<SyncVectorEnv<Lqg>> =
        serde_json::from_value(serde_json::to_value(trainer).unwrap()).unwrap();
    let mut ids = vec![];
    trainer
        .train_observed(
            &mut learner,
            Budget::vector_steps(4),
            None,
            &mut |_: &Ppo, boundary: UpdateBoundary<'_>| {
                ids.push((
                    boundary.learner_call,
                    boundary.collection_cycle,
                    boundary.counters.updates,
                ));
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(ids, [(2, 2, 8)]);
}

#[test]
fn a_new_run_after_learner_only_restore_retains_absolute_model_revision() {
    let mut learner = ppo();
    on_trainer()
        .train(&mut learner, Budget::vector_steps(4), None)
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("continued");
    let mut manifest = RunManifest::new(
        "continued",
        "acceptance",
        "ppo",
        2,
        environment().action_space().clone(),
        0.99,
    )
    .unwrap();
    manifest.starting_revision = learner.updates();
    let recorder = Recorder::create(&path, manifest, StorageConfig::default()).unwrap();
    let mut observer = DiagnosticObserver::new(
        recorder,
        RecordingSchedule {
            metrics: Some(Interval { first: 1, every: 1 }),
            probes: None,
            monte_carlo: None,
            snapshots: None,
        },
        |_: &Ppo, _: &RecordContext, _: Due, _: &mut Recorder| -> rl::Result<()> {
            panic!("metrics-only callback evaluated models");
        },
    );
    on_trainer()
        .train_observed(&mut learner, Budget::vector_steps(4), None, &mut observer)
        .unwrap();
    observer.finish().unwrap();
    let data = Recording::open(path, ReadLimits::default())
        .unwrap()
        .load(0)
        .unwrap();
    assert_eq!(data.context.counters.updates, 4);
    assert_eq!(data.context.identity.learner_update_revision, 8);
    assert_eq!(learner.updates(), 8);
}
