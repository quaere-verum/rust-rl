#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::recurrent_ddqn::{RecurrentDdqn, RecurrentDdqnConfig},
    collect::vector_env::SyncVectorEnv,
    core::{
        action::ActionSpace,
        checkpoint::{Checkpoint, Snapshot},
        learner::UpdateReport,
        stateful::{HistorySpec, StatefulLearner},
    },
    data::temporal::{SequenceBatch, SequenceConfig},
    envs::delayed_bit::DelayedBit,
    nn::{
        network::Mlp,
        recurrent_actor::RecurrentActor,
        temporal::{GruEncoder, RecurrentQ},
    },
    train::{
        checkpoint::{load_run, save_run},
        config::{Budget, OffPolicyConfig},
        recurrent::TemporalTrainer,
        schedule::{Counter, Schedule},
    },
};

type Trainer = TemporalTrainer<
    SyncVectorEnv<DelayedBit>,
    RecurrentActor,
    rl::train::behavior::EpsilonGreedySchedule,
>;

fn learner(seed: u64, n_step: usize) -> RecurrentDdqn {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    let mut config = RecurrentDdqnConfig {
        gamma: 0.9,
        n_step,
        ..Default::default()
    };
    config.ddqn.target_interval = 2;
    config.ddqn.target_tau = 0.4;
    config.ddqn.optimizer.learning_rate = 0.002;
    RecurrentDdqn::new(model, spec, config).unwrap()
}

fn environment(delays: &[usize]) -> SyncVectorEnv<DelayedBit> {
    SyncVectorEnv::new(
        delays
            .iter()
            .enumerate()
            .map(|(i, &delay)| DelayedBit::new(delay, Some(2), false, 100 + i as u64).unwrap())
            .collect(),
    )
    .unwrap()
}

fn config() -> OffPolicyConfig {
    OffPolicyConfig {
        replay_capacity: 128,
        batch_size: 3,
        warmup_transitions: 0,
        n_step: 2,
        gamma: 0.9,
        updates_per_transition: 0.5,
        exploration: Schedule::Linear {
            counter: Counter::EnvironmentTransitions,
            start: 1.,
            end: 0.1,
            duration: 40,
        },
    }
}

fn sequences(n_step: usize) -> SequenceConfig {
    SequenceConfig {
        burn_in: 2,
        learn_len: 3,
        n_step,
    }
}

fn build() -> (RecurrentDdqn, Trainer) {
    let learner = learner(11, 2);
    let actor = learner.policy(1., 0).unwrap();
    let trainer = new_trainer(environment(&[6, 4]), actor, config(), sequences(2), 73).unwrap();
    (learner, trainer)
}

fn new_trainer(
    env: SyncVectorEnv<DelayedBit>,
    actor: RecurrentActor,
    config: OffPolicyConfig,
    sequence: SequenceConfig,
    seed: u64,
) -> Result<Trainer> {
    let behavior = rl::train::behavior::EpsilonGreedySchedule(config.exploration.clone());
    TemporalTrainer::new(env, actor, (&config).into(), sequence, behavior, seed)
}

#[test]
fn discount_mismatch_is_rejected_before_any_training_state_or_rng_mutation() {
    for gamma in [0.99, f32::from_bits(0.9_f32.to_bits() + 1)] {
        let mut model = learner(11, 2);
        let mut cfg = config();
        cfg.gamma = gamma;
        let mut trainer = new_trainer(
            environment(&[6, 4]),
            model.policy(1., 0).unwrap(),
            cfg,
            sequences(2),
            73,
        )
        .unwrap();
        let state = serde_json::to_value(&trainer).unwrap();
        let parameters = model.parameter_values().unwrap();
        let optimizer = model.checkpoint().unwrap().optimizers;
        assert!(
            trainer.step(&mut model).is_err(),
            "discount mismatch was silently accepted"
        );
        assert_eq!(serde_json::to_value(&trainer).unwrap(), state);
        assert_eq!(model.parameter_values().unwrap(), parameters);
        assert_eq!(model.checkpoint().unwrap().optimizers, optimizer);
        assert_eq!(model.updates(), 0);
    }
}

#[test]
fn saving_incompatible_temporal_return_contract_creates_no_checkpoint() {
    let (model, trainer) = build();
    let mut record = model.checkpoint().unwrap();
    record.metadata["config"]["gamma"] = serde_json::json!(0.99_f32);
    record.temporal_returns.as_mut().unwrap().gamma = 0.99;
    let mut other = learner(999, 2);
    other.restore(&record).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("incompatible");
    let parameters = other.parameter_values().unwrap();
    assert!(save_run(&path, &other, &trainer).is_err());
    assert!(!path.exists());
    assert_eq!(other.parameter_values().unwrap(), parameters);
}

#[test]
fn restoration_rejects_a_cross_component_discount_mismatch_before_installation() {
    let (mut source, mut trainer) = build();
    for _ in 0..4 {
        trainer.step(&mut source).unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    save_run(&path, &source, &trainer).unwrap();
    let file = path.join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    value["snapshot"]["config"]["gamma"] = serde_json::json!(0.99_f32);
    let bytes = serde_json::to_vec(&value).unwrap();
    let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    std::fs::write(file, &bytes).unwrap();
    std::fs::write(
        path.join("COMPLETE"),
        format!("{}:{hash:016x}", bytes.len()),
    )
    .unwrap();
    let mut destination = learner(999, 2);
    let parameters = destination.parameter_values().unwrap();
    let optimizer = destination.checkpoint().unwrap().optimizers;
    assert!(load_run::<Trainer>(&path, &mut destination).is_err());
    assert_eq!(destination.parameter_values().unwrap(), parameters);
    assert_eq!(destination.checkpoint().unwrap().optimizers, optimizer);
    assert_eq!(destination.updates(), 0);
}

#[test]
fn ddqn_convenience_constructor_checks_returns_before_initial_environment_reset() {
    use rl::core::{
        action::Action,
        env::{Env, EpisodeStart, ObservationSpace, Step},
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct ResetProbe {
        inner: DelayedBit,
        resets: Arc<AtomicUsize>,
    }
    impl Env for ResetProbe {
        fn observation_space(&self) -> ObservationSpace {
            self.inner.observation_space()
        }
        fn action_space(&self) -> &ActionSpace {
            self.inner.action_space()
        }
        fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
            self.resets.fetch_add(1, Ordering::SeqCst);
            self.inner.reset(seed)
        }
        fn step(&mut self, action: &Action) -> Result<Step> {
            self.inner.step(action)
        }
    }
    let count = Arc::new(AtomicUsize::new(0));
    let env = SyncVectorEnv::new(vec![ResetProbe {
        inner: DelayedBit::new(6, None, false, 11).unwrap(),
        resets: count.clone(),
    }])
    .unwrap();
    let model = learner(11, 2);
    let mut cfg = config();
    cfg.gamma = 0.99;
    assert!(rl::train::behavior::ddqn_trainer(env, &model, cfg, sequences(2), 73).is_err());
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

fn check_full_run_continuation(optimized: bool) {
    check_full_run_continuation_mode(optimized, false);
}

#[test]
fn canonical_recurrent_ddqn_full_run_preserves_windows_rng_and_optimizer_updates() {
    check_full_run_continuation_mode(true, true);
}

fn check_full_run_continuation_mode(optimized: bool, canonical: bool) {
    let (mut uninterrupted, mut trainer) = if canonical {
        let learner = learner(11, 2);
        let cfg = config();
        let behavior = rl::train::behavior::EpsilonGreedySchedule(cfg.exploration.clone());
        let trainer = TemporalTrainer::new_with_inference_mode(
            environment(&[6, 4]),
            learner.policy(1., 0).unwrap(),
            (&cfg).into(),
            sequences(2),
            behavior,
            73,
            rl::collect::stateful::InferenceMode::CanonicalBatched,
        )
        .unwrap();
        (learner, trainer)
    } else {
        build()
    };
    if optimized {
        uninterrupted
            .set_update_mode(rl::algorithms::recurrent_ddqn::RecurrentUpdateMode::OptimizedExact);
        if !canonical {
            trainer
                .set_inference_mode(rl::collect::stateful::InferenceMode::Batched)
                .unwrap();
        }
    }
    for _ in 0..4 {
        trainer.step(&mut uninterrupted).unwrap();
    }
    assert!(uninterrupted.updates() >= 3);
    assert_eq!(trainer.collector().lane_timestep(0), Some(4));
    assert_eq!(trainer.collector().lane_revision(0), Some(0));
    assert!(trainer.collector().latest_revision() > 0);
    assert!(!trainer.last_windows().is_empty());
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    save_run(&path, &uninterrupted, &trainer).unwrap();
    let mut restored = learner(999, 2);
    let mut resumed: Trainer = load_run(&path, &mut restored).unwrap();
    assert_eq!(
        uninterrupted.parameter_values().unwrap(),
        restored.parameter_values().unwrap()
    );
    assert_eq!(
        serde_json::to_value(&trainer).unwrap(),
        serde_json::to_value(&resumed).unwrap()
    );
    for _ in 0..12 {
        let a = trainer.step(&mut uninterrupted).unwrap();
        let b = resumed.step(&mut restored).unwrap();
        assert_eq!(a.collection, b.collection);
        assert_eq!(
            serde_json::to_value(&a.reports).unwrap(),
            serde_json::to_value(&b.reports).unwrap()
        );
        assert!(a.collection_seconds.is_finite() && a.collection_seconds >= 0.);
        assert!(a.learning_seconds.is_finite() && a.learning_seconds >= 0.);
        assert_eq!(trainer.counters(), resumed.counters());
        assert_eq!(trainer.last_windows(), resumed.last_windows());
        assert_eq!(
            serde_json::to_value(trainer.store()).unwrap(),
            serde_json::to_value(resumed.store()).unwrap()
        );
        assert_eq!(
            serde_json::to_value(trainer.collector()).unwrap(),
            serde_json::to_value(resumed.collector()).unwrap()
        );
        assert_eq!(
            uninterrupted.parameter_values().unwrap(),
            restored.parameter_values().unwrap()
        );
        assert_eq!(uninterrupted.updates(), restored.updates());
    }
    assert!(trainer.counters().episodes > 2);
    assert!(trainer.collector().lane_revision(0).unwrap() > 0);
    trainer.validate_snapshot().unwrap();
    resumed.validate_snapshot().unwrap();
}

#[test]
fn run_checkpoint_continues_mid_episode_across_updates_and_pinned_behavior_revisions() {
    check_full_run_continuation(false);
}

#[test]
fn optimized_batched_full_run_continuation_preserves_actions_memories_windows_and_parameters() {
    check_full_run_continuation(true);
}

#[test]
fn fractional_update_credit_excludes_all_warmup_transitions() {
    let mut learner = learner(11, 1);
    let mut config = config();
    config.n_step = 1;
    config.warmup_transitions = 3;
    config.exploration = Schedule::Constant(1.);
    let mut trainer = new_trainer(
        environment(&[6]),
        learner.policy(1., 0).unwrap(),
        config,
        sequences(1),
        73,
    )
    .unwrap();
    let mut reports = Vec::new();
    for _ in 0..7 {
        reports.push(trainer.step(&mut learner).unwrap().reports.len());
    }
    assert_eq!(reports, [0, 0, 0, 0, 1, 0, 1]);
    assert_eq!(trainer.counters().env_steps, 7);
    assert_eq!(trainer.counters().updates, 2);
    assert_eq!(learner.updates(), 2);
    trainer.validate_snapshot().unwrap();
}

#[test]
fn snapshots_allow_explicitly_ignored_live_tapes_and_evicted_completed_lanes() {
    let mut learner = learner(11, 1);
    let mut config = config();
    config.replay_capacity = 2;
    config.n_step = 1;
    config.updates_per_transition = 0.25;
    config.exploration = Schedule::Constant(1.);
    let mut trainer = new_trainer(
        environment(&[6]),
        learner.policy(1., 0).unwrap(),
        config,
        sequences(1),
        73,
    )
    .unwrap();
    for _ in 0..3 {
        trainer.step(&mut learner).unwrap();
    }
    assert!(trainer.store().is_ignored(0));
    trainer.validate_snapshot().unwrap();
    for _ in 0..4 {
        trainer.step(&mut learner).unwrap();
    }
    assert!(!trainer.store().is_ignored(0));
    assert!(trainer.store().episode(0).is_none());
    assert_eq!(trainer.collector().lane_needs_reset(0), Some(true));
    trainer.validate_snapshot().unwrap();
    trainer.step(&mut learner).unwrap();
    assert_eq!(trainer.collector().lane_episode(0), Some(1));
    assert_eq!(learner.updates(), 2);
    trainer.validate_snapshot().unwrap();
}

struct FailAfterUpdate {
    inner: RecurrentDdqn,
}
impl StatefulLearner for FailAfterUpdate {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        self.inner.update(batch)?;
        Err(rl::Error::State(
            "injected failure after a real parameter update".into(),
        ))
    }
}
impl rl::core::stateful::BehaviorPublisher for FailAfterUpdate {
    type Actor = RecurrentActor;
    type Settings = rl::algorithms::recurrent_ddqn::EpsilonGreedySettings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<Self::Actor> {
        self.inner.policy(settings.epsilon, revision)
    }
}
impl rl::core::stateful::TemporalOffPolicyLearner for FailAfterUpdate {
    fn return_contract(&self) -> rl::core::temporal::ReturnSpec {
        rl::core::stateful::TemporalOffPolicyLearner::return_contract(&self.inner)
    }
}

#[test]
fn an_update_failure_after_parameter_mutation_poison_closes_the_trainer_until_restore() {
    let inner = learner(11, 1);
    let actor = inner.policy(1., 0).unwrap();
    let mut config = config();
    config.n_step = 1;
    config.updates_per_transition = 1.;
    config.exploration = Schedule::Constant(1.);
    let mut trainer = new_trainer(environment(&[6]), actor, config, sequences(1), 73).unwrap();
    let mut failing = FailAfterUpdate { inner };
    assert!(trainer.step(&mut failing).is_err());
    assert_eq!(failing.inner.updates(), 1);
    let frozen = serde_json::to_value(&trainer).unwrap();
    assert!(trainer.step(&mut failing).is_err());
    assert_eq!(serde_json::to_value(&trainer).unwrap(), frozen);
    assert_eq!(failing.inner.updates(), 1);
    assert!(trainer.validate_snapshot().is_err());
    let directory = tempfile::tempdir().unwrap();
    assert!(save_run(directory.path().join("poisoned"), &failing.inner, &trainer).is_err());
}

#[test]
fn bounded_run_cannot_wait_forever_for_updates_when_every_partial_prefix_is_evicted() {
    let mut learner = learner(11, 3);
    let mut config = config();
    config.replay_capacity = 2;
    config.n_step = 3;
    config.exploration = Schedule::Constant(1.);
    let mut trainer = new_trainer(
        environment(&[6]),
        learner.policy(1., 0).unwrap(),
        config,
        sequences(3),
        73,
    )
    .unwrap();
    let update_only = Budget {
        updates: Some(1),
        ..Default::default()
    };
    assert!(trainer.run(&mut learner, update_only).is_err());
    assert_eq!(trainer.counters().env_steps, 0);
    let result = trainer.run(&mut learner, Budget::vector_steps(10)).unwrap();
    assert_eq!(result.vector_steps, 10);
    assert_eq!(result.updates, 0);
    trainer.validate_snapshot().unwrap();
}

#[test]
fn trainer_rejects_mismatched_or_unbounded_configuration_before_collection() {
    let learner = learner(11, 2);
    for bad in 0..3 {
        let mut config = config();
        match bad {
            0 => config.n_step = 3,
            1 => config.updates_per_transition = 1024.1,
            _ => config.replay_capacity = 1,
        }
        assert!(
            new_trainer(
                environment(&[6, 4]),
                learner.policy(1., 0).unwrap(),
                config,
                sequences(2),
                73
            )
            .is_err()
        );
    }
}

#[test]
fn malformed_snapshot_topology_is_rejected_without_installing_a_learner() {
    let (mut learner, mut trainer) = build();
    for _ in 0..3 {
        trainer.step(&mut learner).unwrap();
    }
    let original = serde_json::to_value(&trainer).unwrap();
    for bad in 0..7 {
        let mut value = original.clone();
        match bad {
            0 => value["version"] = 99.into(),
            1 => value["counters"]["env_steps"] = 0.into(),
            2 => value["behavior_revision"] = 999.into(),
            3 => value["update_credit"] = (-1.).into(),
            4 => value["store"]["episodes"][0]["observations"][3]["values"][0] = 999.into(),
            5 => value["store"]["active"]["0"]["next_timestep"] = 2.into(),
            _ => value["store"]["highest_episode_id"] = 999.into(),
        }
        if let Ok(invalid) = serde_json::from_value::<Trainer>(value) {
            assert!(invalid.validate_snapshot().is_err(), "corruption {bad}");
        }
    }
}

#[test]
fn existing_run_loader_rejects_a_trainer_version_before_changing_the_destination_learner() {
    let (mut original, mut trainer) = build();
    for _ in 0..3 {
        trainer.step(&mut original).unwrap();
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("run");
    save_run(&path, &original, &trainer).unwrap();
    let file = path.join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
    value["snapshot"]["version"] = 99.into();
    let bytes = serde_json::to_vec(&value).unwrap();
    let checksum = bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    std::fs::write(file, &bytes).unwrap();
    std::fs::write(
        path.join("COMPLETE"),
        format!("{}:{checksum:016x}", bytes.len()),
    )
    .unwrap();
    let mut destination = learner(999, 2);
    let before = destination.parameter_values().unwrap();
    assert!(load_run::<Trainer>(&path, &mut destination).is_err());
    assert_eq!(destination.parameter_values().unwrap(), before);
}

#[test]
fn snapshot_rejects_a_corrupted_ignored_episode_clock_before_environment_advances() {
    let mut l = learner(11, 3);
    let mut c = config();
    c.replay_capacity = 2;
    c.n_step = 3;
    c.exploration = Schedule::Constant(1.);
    let mut t = new_trainer(
        environment(&[6]),
        l.policy(1., 0).unwrap(),
        c,
        sequences(3),
        73,
    )
    .unwrap();
    for _ in 0..3 {
        t.step(&mut l).unwrap();
    }
    assert!(t.store().is_ignored(0));
    let mut value = serde_json::to_value(&t).unwrap();
    value["store"]["active"]["0"]["next_timestep"] = 999.into();
    let corrupt: Trainer = serde_json::from_value(value).unwrap();
    assert!(corrupt.validate_snapshot().is_err());
}
