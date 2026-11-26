#![cfg(feature = "neural")]

#[path = "support/resume_fixture.rs"]
mod resume_fixture;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use resume_fixture::ResumeFixture;
use rl::{
    algorithms::recurrent_ddqn::{ONLINE_REPRESENTATION, TARGET_REPRESENTATION},
    collect::stateful::InferenceMode,
    core::reconstruction::*,
    data::temporal::{ResourceLimits, SampleLimits, SegmentedConfig, TemporalStorage},
    envs::delayed_bit::DelayedBit,
    prelude::*,
    train::{
        checkpoint::{load_run, save_run},
        temporal_recording::TemporalReplayConfig,
    },
};

type Trainer = TemporalTrainer<SyncVectorEnv<DelayedBit>, RecurrentActor, EpsilonGreedySchedule>;
fn learner(seed: u64) -> RecurrentDdqn {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    let mut config = RecurrentDdqnConfig {
        gamma: 0.9,
        n_step: 2,
        ..Default::default()
    };
    config.ddqn.target_interval = 2;
    config.ddqn.target_tau = 0.4;
    let mut l = RecurrentDdqn::new(q, spec, config).unwrap();
    l.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    l
}
fn mode(anchored: bool) -> ReconstructionConfig {
    ReconstructionConfig {
        version: 1,
        mode: if anchored {
            ReconstructionMode::Anchored {
                max_generation_lag: None,
            }
        } else {
            ReconstructionMode::Exact
        },
    }
}
fn build(anchored: bool) -> (RecurrentDdqn, Trainer) {
    let mut l = learner(11);
    l.set_reconstruction_config(mode(anchored)).unwrap();
    let env = SyncVectorEnv::new(
        [1000, 97]
            .into_iter()
            .map(|h| DelayedBit::new(h, Some(40), false, 9).unwrap())
            .collect(),
    )
    .unwrap();
    let config = OffPolicyConfig {
        replay_capacity: if anchored { 128 } else { 4096 },
        batch_size: 2,
        warmup_transitions: 0,
        n_step: 2,
        gamma: 0.9,
        updates_per_transition: 0.125,
        exploration: Schedule::Constant(0.35),
    };
    let seq = SequenceConfig {
        burn_in: 8,
        learn_len: 8,
        n_step: 2,
    };
    let mut t = ddqn_trainer(env, &l, config.clone(), seq, 73).unwrap();
    t.set_inference_mode(InferenceMode::Batched).unwrap();
    let limits = ResourceLimits {
        max_transitions: config.replay_capacity,
        max_resident_bytes: 4 * 1024 * 1024,
        max_anchor_bytes: 128 * 1024,
        max_anchors: 4096,
        max_anchor_payload_bytes: 4096,
        max_dimension_rank: 2,
    };
    t.set_replay_config(TemporalReplayConfig {
        version: 1,
        storage: if anchored {
            TemporalStorage::Segmented(SegmentedConfig {
                segment_steps: 8,
                minimum_context: 32,
                required_representations: vec![ONLINE_REPRESENTATION, TARGET_REPRESENTATION],
                limits,
            })
        } else {
            TemporalStorage::FullPrefix(limits)
        },
        reconstruction: mode(anchored),
        sample_limits: SampleLimits {
            max_sample_tokens: if anchored { 40 } else { 1010 },
            max_batch_tokens: if anchored { 80 } else { 2020 },
            max_sample_payload_bytes: 1024 * 1024,
            max_batch_payload_bytes: 2 * 1024 * 1024,
        },
        anchor_interval: 8,
    })
    .unwrap();
    (l, t)
}
fn step(
    l: &mut RecurrentDdqn,
    t: &mut Trainer,
    anchored: bool,
) -> rl::train::recurrent::TemporalProgress {
    if anchored {
        t.step_with_reconstruction(l).unwrap()
    } else {
        t.step(l).unwrap()
    }
}

#[test]
fn checkpointed_ddqn_resumes_exact_and_anchored_replay() {
    type Restart = ResumeFixture<DelayedBit>;
    type RestartTrainer =
        TemporalTrainer<SyncVectorEnv<Restart>, RecurrentActor, EpsilonGreedySchedule>;
    for anchored in [false, true] {
        let (mut model, template) = build(anchored);
        let env = SyncVectorEnv::new(vec![ResumeFixture::new(
            DelayedBit::new(4, None, false, 19).unwrap(),
            Some(3),
        )])
        .unwrap();
        let mut config = template.config().clone();
        config.updates_per_transition = 1.;
        let mut trainer = TemporalTrainer::new(
            env,
            model.policy(0.35, 0).unwrap(),
            config,
            SequenceConfig {
                burn_in: 1,
                learn_len: 2,
                n_step: 2,
            },
            EpsilonGreedySchedule(Schedule::Constant(0.35)),
            73,
        )
        .unwrap();
        let mut replay = template.replay_config().unwrap().clone();
        replay.anchor_interval = 2;
        trainer.set_replay_config(replay).unwrap();
        let mut sampled_resume = false;
        for _ in 0..20 {
            if anchored {
                trainer.step_with_reconstruction(&mut model).unwrap();
            } else {
                trainer.step(&mut model).unwrap();
            }
            assert_eq!(trainer.store().len() as u64, trainer.counters().env_steps);
            for &(id, start) in trainer.last_windows() {
                let episode = trainer.store().episode(id).unwrap();
                assert!(start >= episode.learning_start);
                sampled_resume |= episode.learning_start > 0;
            }
        }
        assert!(sampled_resume);
        if anchored {
            assert!(trainer.store().episodes().any(|ep| {
                ep.learning_start > 0
                    && ep.anchors.iter().any(|set| {
                        set.anchors.len() == 2 && set.observation_index() > ep.learning_start
                    })
            }));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpointed-ddqn");
        save_run(&path, &model, &trainer).unwrap();
        let mut restored = learner(79);
        let mut resumed: RestartTrainer = load_run(&path, &mut restored).unwrap();
        for _ in 0..4 {
            let (expected, actual) = if anchored {
                (
                    trainer.step_with_reconstruction(&mut model),
                    resumed.step_with_reconstruction(&mut restored),
                )
            } else {
                (trainer.step(&mut model), resumed.step(&mut restored))
            };
            assert_eq!(
                serde_json::to_value(expected.unwrap().reports).unwrap(),
                serde_json::to_value(actual.unwrap().reports).unwrap()
            );
            assert_eq!(trainer.last_windows(), resumed.last_windows());
            assert_eq!(
                trainer.collector().lane_state(0),
                resumed.collector().lane_state(0)
            );
        }
    }
}

#[test]
fn canonical_trainer_rejects_anchor_recording_configuration_and_forged_snapshot() {
    let (learner, old) = build(true);
    let replay = old.replay_config().unwrap().clone();
    let env = SyncVectorEnv::new(
        [1000, 97]
            .into_iter()
            .map(|h| DelayedBit::new(h, Some(40), false, 9).unwrap())
            .collect(),
    )
    .unwrap();
    let mut canonical = TemporalTrainer::new_with_inference_mode(
        env,
        learner.policy(0.35, 0).unwrap(),
        old.config().clone(),
        *old.sequence_config(),
        EpsilonGreedySchedule(Schedule::Constant(0.35)),
        73,
        InferenceMode::CanonicalBatched,
    )
    .unwrap();
    let before = serde_json::to_value(&canonical).unwrap();
    assert!(
        canonical.set_replay_config(replay).is_err(),
        "v1 anchors do not identify their numerical execution"
    );
    assert_eq!(serde_json::to_value(&canonical).unwrap(), before);
    let mut forged = serde_json::to_value(&old).unwrap();
    forged["collector"]["version"] = serde_json::json!(5);
    forged["collector"]["inference_mode"] = serde_json::json!("canonical_batched_v1");
    let restored: Trainer = serde_json::from_value(forged).unwrap();
    assert!(restored.validate_snapshot().is_err());
}
#[test]
fn full_run_exact_and_anchored_restore_subsequent_actions_windows_targets_and_parameters() {
    for anchored in [false, true] {
        let (mut l, mut t) = build(anchored);
        for i in 0..75 {
            step(&mut l, &mut t, anchored);
            if i == 49 {
                for lane in 0..2 {
                    t.adopt_lane(lane, ONLINE_REPRESENTATION, mode(anchored))
                        .unwrap();
                }
            }
        }
        assert!(l.updates() > 0);
        assert_eq!(t.collector().lane_timestep(0), Some(75));
        if anchored {
            assert!(t.store().episode(0).unwrap().base_timestep > 0);
            assert!(t.store().len() <= 128);
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run");
        save_run(&path, &l, &t).unwrap();
        let mut restored = learner(999);
        let mut rt: Trainer = load_run(&path, &mut restored).unwrap();
        assert_eq!(
            serde_json::to_value(&t).unwrap(),
            serde_json::to_value(&rt).unwrap()
        );
        for i in 0..48 {
            let a = step(&mut l, &mut t, anchored);
            let b = step(&mut restored, &mut rt, anchored);
            assert_eq!(
                a.reports.iter().map(|r| &r.metrics).collect::<Vec<_>>(),
                b.reports.iter().map(|r| &r.metrics).collect::<Vec<_>>()
            );
            assert_eq!(t.last_windows(), rt.last_windows());
            assert_eq!(
                l.parameter_values().unwrap(),
                restored.parameter_values().unwrap()
            );
            for lane in 0..2 {
                assert_eq!(
                    t.collector().lane_state(lane),
                    rt.collector().lane_state(lane)
                );
                assert_eq!(
                    t.collector().lane_revision(lane),
                    rt.collector().lane_revision(lane)
                );
            }
            if i % 8 == 0 {
                assert_eq!(
                    serde_json::to_value(&t).unwrap(),
                    serde_json::to_value(&rt).unwrap()
                );
            }
        }
        assert!(t.counters().episodes > 0);
        t.validate_snapshot().unwrap();
        rt.validate_snapshot().unwrap();
    }
}
fn rewrite(path: &std::path::Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let file = path.join("checkpoint.json");
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    mutate(&mut v);
    let bytes = serde_json::to_vec(&v).unwrap();
    let h = bytes.iter().fold(0xcbf29ce484222325_u64, |a, b| {
        (a ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    std::fs::write(file, &bytes).unwrap();
    std::fs::write(path.join("COMPLETE"), format!("{}:{h:016x}", bytes.len())).unwrap();
}
#[test]
fn checksummed_corrupt_anchor_payloads_fail_before_installing_model_or_optimizer() {
    let (mut source, mut t) = build(true);
    for _ in 0..75 {
        step(&mut source, &mut t, true);
    }
    for corruption in 0..3 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("run");
        save_run(&path, &source, &t).unwrap();
        rewrite(&path, |v| {
            let a = &mut v["snapshot"]["store"]["episodes"][0]["anchors"][0]["anchors"]["0"];
            match corruption {
                0 => a["schema"]["codec"] = 99.into(),
                1 => {
                    for (i, byte) in [0, 0, 192, 127].into_iter().enumerate() {
                        a["payload"][i] = byte.into();
                    }
                }
                _ => a["generation"] = 999999.into(),
            }
        });
        let mut destination = learner(999);
        let before = serde_json::to_vec(&destination.checkpoint().unwrap()).unwrap();
        assert!(
            load_run::<Trainer>(&path, &mut destination).is_err(),
            "corruption {corruption}"
        );
        assert_eq!(
            before,
            serde_json::to_vec(&destination.checkpoint().unwrap()).unwrap()
        );
    }
}
