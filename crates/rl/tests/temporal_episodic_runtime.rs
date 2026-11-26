#![cfg(feature = "neural")]

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::Tensor,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{recurrent_reinforce::RecurrentReinforce, reinforce::ReinforceConfig},
    collect::{stateful::InferenceMode, vector_env::SyncVectorEnv},
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, Snapshot},
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
    },
    data::{temporal::SequenceConfig, temporal_on_policy::TemporalRollout},
    envs::delayed_bit::DelayedBit,
    nn::{temporal_policy::TemporalPolicy, temporal_value::TemporalValue},
    train::{
        checkpoint::{load_run, save_run},
        config::OnPolicyConfig,
        temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
    },
};

type Trainer = TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>;

fn learner(seed: u64, baseline: bool) -> RecurrentReinforce {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
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
fn build(delays: &[usize], baseline: bool, guard: usize) -> (RecurrentReinforce, Trainer) {
    let learner = learner(140, baseline);
    let env = SyncVectorEnv::new(
        delays
            .iter()
            .enumerate()
            .map(|(i, &delay)| {
                DelayedBit::new(delay, Some(delay / 2), false, 100 + i as u64).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let config = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 2,
            episode_guard: Some(guard),
            gamma: 0.9,
            gae_lambda: None,
            normalize_advantages: false,
        },
        sequence: SequenceConfig {
            burn_in: 1,
            learn_len: 8,
            n_step: 1,
        },
        capacity: 2048,
        max_fresh: 1024,
        inference_mode: InferenceMode::Batched,
        ..Default::default()
    };
    let trainer = Trainer::new(env, &learner, config, 47).unwrap();
    (learner, trainer)
}
fn runtime(trainer: &Trainer) -> serde_json::Value {
    let mut value = serde_json::to_value(trainer).unwrap();
    value.as_object_mut().unwrap().remove("timings");
    value
}
fn model_bits(model: &impl Module) -> Vec<u32> {
    struct Values(Vec<u32>);
    impl ModuleVisitor for Values {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            self.0.extend(
                parameter
                    .val()
                    .to_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits),
            );
        }
    }
    let mut values = Values(Vec::new());
    model.visit(&mut values);
    values.0
}

#[test]
fn unequal_episodes_pause_finished_lanes_and_keep_terminal_observations_for_1_4_16_lanes() {
    for lanes in [1, 4, 16] {
        let delays: Vec<_> = (0..lanes).map(|lane| 2 + lane % 4).collect();
        let (mut learner, mut trainer) = build(&delays, lanes == 4, 16);
        trainer.validate_snapshot().unwrap();
        let rounds = delays.iter().max().unwrap() + 1;
        for tick in 1..=rounds {
            let before = runtime(&trainer);
            let delta = trainer.collect_tick(&learner).unwrap();
            let after = runtime(&trainer);
            let active = delays.iter().filter(|&&delay| tick <= delay + 1).count();
            assert_eq!(delta.env_steps, active as u64);
            for (lane, &delay) in delays.iter().enumerate() {
                assert_eq!(trainer.collector().lane_episode(lane), Some(lane as u64));
                assert_eq!(
                    trainer.collector().lane_timestep(lane),
                    Some(tick.min(delay + 1) as u64)
                );
                assert_eq!(after["completed"][lane], tick > delay);
                if tick > delay + 1 {
                    assert_eq!(
                        before["collector"]["agents"][lane],
                        after["collector"]["agents"][lane]
                    );
                    assert_eq!(
                        before["collector"]["driver"]["env"]["envs"][lane],
                        after["collector"]["driver"]["env"]["envs"][lane]
                    );
                }
                let episode = trainer.rollout().store().episode(lane as u64).unwrap();
                assert_eq!(episode.observations.len(), episode.steps.len() + 1);
                if tick > delay {
                    assert!(episode.steps.last().unwrap().terminated);
                    assert!(!episode.steps.last().unwrap().truncated);
                    assert_eq!(
                        episode.observations.last().unwrap(),
                        &trainer.collector().observations()[lane]
                    );
                    assert_eq!(episode.observations.last().unwrap().dense_values()[2], 1.);
                }
            }
            trainer.validate_snapshot().unwrap();
        }
        let expected: usize = delays.iter().map(|delay| delay + 1).sum();
        assert_eq!(trainer.rollout().fresh_len(), expected);
        let batch = trainer.seal(&learner).unwrap();
        assert_eq!(batch.len(), expected);
        assert_eq!(batch.episode_count, lanes);
        assert!(batch.rows.iter().all(|row| row.return_.is_some()));
        assert_eq!(trainer.update(&mut learner).unwrap().updates, 1);
        assert_eq!(trainer.rollout().fresh_len(), 0);
        assert_eq!(runtime(&trainer)["collection_ticks"], 0);
        assert!(
            runtime(&trainer)["completed"]
                .as_array()
                .unwrap()
                .iter()
                .all(|value| value == false)
        );
        // The acknowledged/retired ended lanes legitimately retain old clocks.
        trainer.validate_snapshot().unwrap();
    }
}

#[test]
fn ordinary_marker_and_partial_collection_keep_memory_without_monte_carlo_bootstrap() {
    let (learner, mut trainer) = build(&[6], true, 16);
    let episode_id = trainer.collector().lane_episode(0);
    for _ in 0..3 {
        trainer.collect_tick(&learner).unwrap();
    }
    let observation = &trainer.collector().observations()[0];
    assert_eq!(observation.dense_values()[1], 0.);
    assert_eq!(observation.dense_values()[3], 1.);
    assert_eq!(trainer.collector().lane_episode(0), episode_id);
    assert_eq!(trainer.collector().lane_needs_reset(0), Some(false));
    let episode = trainer
        .rollout()
        .store()
        .episode(episode_id.unwrap())
        .unwrap();
    assert!(
        episode
            .steps
            .iter()
            .all(|step| !step.ended() && step.reward.total() == 0.)
    );
    let actor = trainer.collector().lane_actor(0).unwrap();
    let mut state = actor.initial_state().unwrap();
    for (time, observation) in episode.observations.iter().enumerate() {
        state = actor
            .assimilate(
                &state,
                HistoryInput {
                    observation,
                    previous: time.checked_sub(1).map(|previous| PreviousInput {
                        factual: None,
                        action: &episode.steps[previous].action,
                        reward: episode.steps[previous].reward.total(),
                    }),
                },
            )
            .unwrap();
    }
    assert_eq!(trainer.collector().lane_state(0), Some(&state));
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(trainer.seal(&learner).is_err());
    assert!(trainer.pending_batch().is_none());
    assert_eq!(before, serde_json::to_value(&trainer).unwrap());
}

fn corrupt_file(
    path: &std::path::Path,
    learner: &RecurrentReinforce,
    trainer: &Trainer,
    snapshot: serde_json::Value,
) {
    save_run(path, learner, trainer).unwrap();
    let file = path.join("checkpoint.json");
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    envelope["snapshot"] = snapshot;
    let bytes = serde_json::to_vec(&envelope).unwrap();
    let checksum = bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ *byte as u64).wrapping_mul(0x100000001b3)
    });
    std::fs::write(&file, &bytes).unwrap();
    std::fs::write(
        path.join("COMPLETE"),
        format!("{}:{checksum:016x}", bytes.len()),
    )
    .unwrap();
}

fn rejects_without_mutation(
    source: &Trainer,
    learner: &RecurrentReinforce,
    invalid: serde_json::Value,
) {
    let mut candidate: Trainer = serde_json::from_value(invalid.clone()).unwrap();
    let before = serde_json::to_value(&candidate).unwrap();
    assert!(
        candidate.validate_snapshot().is_err(),
        "corrupt episodic snapshot was accepted"
    );
    assert!(
        candidate.collect_tick(learner).is_err(),
        "corrupt episodic state reached interaction"
    );
    assert_eq!(before, serde_json::to_value(&candidate).unwrap());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corrupt-episodic");
    corrupt_file(&path, learner, source, invalid);
    let mut destination = self::learner(999, false);
    let model_before = serde_json::to_value(destination.checkpoint().unwrap()).unwrap();
    let source_before = serde_json::to_value(source).unwrap();
    assert!(
        load_run::<Trainer>(&path, &mut destination).is_err(),
        "typed load installed corrupt episodic state"
    );
    assert!(
        model_before == serde_json::to_value(destination.checkpoint().unwrap()).unwrap(),
        "typed load mutated destination learner before rejecting snapshot"
    );
    assert_eq!(source_before, serde_json::to_value(source).unwrap());
}

#[test]
fn premature_and_missing_completion_flags_reject_before_live_or_typed_load_mutation() {
    let (learner, mut trainer) = build(&[2, 4, 3, 5], false, 16);
    trainer.collect_tick(&learner).unwrap();
    let mut premature = serde_json::to_value(&trainer).unwrap();
    premature["completed"][0] = true.into();
    rejects_without_mutation(&trainer, &learner, premature);
    trainer.collect_tick(&learner).unwrap();
    trainer.collect_tick(&learner).unwrap();
    assert_eq!(trainer.collector().lane_needs_reset(0), Some(true));
    let mut missing = serde_json::to_value(&trainer).unwrap();
    missing["completed"][0] = false.into();
    rejects_without_mutation(&trainer, &learner, missing);
}

#[test]
fn episodic_collection_clock_and_guard_are_bound_to_committed_history() {
    let (learner, mut trainer) = build(&[5, 7, 6, 8], false, 16);
    for _ in 0..3 {
        trainer.collect_tick(&learner).unwrap();
    }
    let snapshot = serde_json::to_value(&trainer).unwrap();
    for corruption in 0..4 {
        let mut invalid = snapshot.clone();
        match corruption {
            0 => invalid["collection_ticks"] = 2.into(),
            1 => invalid["collection_ticks"] = 4.into(),
            2 => invalid["collection_ticks"] = 0.into(),
            _ => invalid["config"]["collection"]["episode_guard"] = 2.into(),
        }
        rejects_without_mutation(&trainer, &learner, invalid);
    }
}

#[test]
fn acknowledged_empty_collection_accepts_retired_lanes_but_no_flags_or_ticks() {
    let (mut learner, mut trainer) = build(&[2, 4], true, 16);
    trainer.cycle(&mut learner).unwrap();
    trainer.validate_snapshot().unwrap();
    assert_eq!(trainer.rollout().fresh_len(), 0);
    assert!(trainer.collector().lane_timestep(0).unwrap() > 0);
    let snapshot = serde_json::to_value(&trainer).unwrap();
    for corruption in 0..2 {
        let mut invalid = snapshot.clone();
        if corruption == 0 {
            invalid["completed"][0] = true.into();
        } else {
            invalid["collection_ticks"] = 1.into();
        }
        rejects_without_mutation(&trainer, &learner, invalid);
    }
}

#[test]
fn forged_empty_acknowledgement_cannot_discard_a_live_monte_carlo_prefix() {
    let (learner, mut trainer) = build(&[6], false, 16);
    for _ in 0..3 {
        trainer.collect_tick(&learner).unwrap();
    }
    let id = trainer.collector().lane_episode(0).unwrap().to_string();
    let mut invalid = serde_json::to_value(&trainer).unwrap();
    invalid["rollout"]["acknowledged"][&id] = 3.into();
    invalid["rollout"]["fresh"] = serde_json::json!([]);
    invalid["rollout"]["metadata_bytes"] = 0.into();
    invalid["collection_ticks"] = 0.into();
    // This is a valid generic rollout acknowledgement, but an impossible MC
    // trainer phase: the only current episode has neither ended nor been reset.
    let rollout: TemporalRollout = serde_json::from_value(invalid["rollout"].clone()).unwrap();
    rollout.validate_snapshot().unwrap();
    rejects_without_mutation(&trainer, &learner, invalid);
}

#[test]
fn sealed_episode_count_is_bound_to_two_complete_collected_lanes() {
    let (learner, mut trainer) = build(&[2, 4], false, 16);
    for _ in 0..5 {
        trainer.collect_tick(&learner).unwrap();
    }
    let batch = trainer.seal(&learner).unwrap();
    assert_eq!(batch.episode_count, 2);
    trainer.validate_snapshot().unwrap();

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("two-complete-episodes");
    save_run(&path, &learner, &trainer).unwrap();
    let mut destination = self::learner(997, true);
    let restored: Trainer = load_run(&path, &mut destination).unwrap();
    assert_eq!(runtime(&trainer), runtime(&restored));
    assert_eq!(
        learner.checkpoint().unwrap().metadata,
        destination.checkpoint().unwrap().metadata
    );

    let mut invalid = serde_json::to_value(&trainer).unwrap();
    invalid["pending"]["episode_count"] = 1.into();
    // A positive scalar is structurally valid for an arbitrary supplied batch,
    // but this sealed run must retain its two actual terminal episodes.
    let forged: Trainer = serde_json::from_value(invalid.clone()).unwrap();
    forged.pending_batch().unwrap().validate().unwrap();
    rejects_without_mutation(&trainer, &learner, invalid);
}

#[test]
fn exhausted_episode_guard_rejects_next_tick_and_partial_seal_without_mutation() {
    let (learner, mut trainer) = build(&[8], true, 2);
    trainer.collect_tick(&learner).unwrap();
    trainer.collect_tick(&learner).unwrap();
    trainer.validate_snapshot().unwrap();
    let before = serde_json::to_value(&trainer).unwrap();
    let model_before = serde_json::to_value(learner.checkpoint().unwrap()).unwrap();
    assert!(trainer.collect_tick(&learner).is_err());
    assert!(trainer.seal(&learner).is_err());
    assert_eq!(before, serde_json::to_value(&trainer).unwrap());
    assert_eq!(
        model_before,
        serde_json::to_value(learner.checkpoint().unwrap()).unwrap()
    );
}

#[test]
fn partial_finished_and_sealed_episode_snapshots_preserve_exact_future() {
    for ticks in [2, 3, 6] {
        let (mut a_learner, mut a) = build(&[2, 3, 4, 5], false, 16);
        for _ in 0..ticks {
            a.collect_tick(&a_learner).unwrap();
        }
        if ticks == 6 {
            a.seal(&a_learner).unwrap();
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("exact-episodic");
        save_run(&path, &a_learner, &a).unwrap();
        let mut b_learner = learner(888, true);
        let mut b: Trainer = load_run(&path, &mut b_learner).unwrap();
        assert_eq!(runtime(&a), runtime(&b));
        for _ in 0..2 {
            let left = a.cycle(&mut a_learner).unwrap();
            let right = b.cycle(&mut b_learner).unwrap();
            assert_eq!(left.metrics, right.metrics);
            assert_eq!(
                model_bits(&a_learner.policy()),
                model_bits(&b_learner.policy())
            );
            assert_eq!(
                a_learner.checkpoint().unwrap().metadata,
                b_learner.checkpoint().unwrap().metadata
            );
            assert_eq!(runtime(&a), runtime(&b));
            a.validate_snapshot().unwrap();
            b.validate_snapshot().unwrap();
        }
    }
}
