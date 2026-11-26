use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    collect::vector_env::SyncVectorEnv,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        anchors::{AnchorSchema, RepresentationId},
        checkpoint::{LearnerRecord, Snapshot},
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        learner::UpdateReport,
        reconstruction::{
            ReconstructibleActor, ReconstructionConfig, ReconstructionMode, ReconstructionModel,
            ReplayReconstruction, ReplayRepresentations, decode_anchor,
        },
        stateful::{
            BehaviorPublisher, HistoryInput, StatefulActor, StatefulDecision, StatefulLearner,
            TemporalOffPolicyLearner,
        },
        temporal::ReturnSpec,
    },
    data::temporal::{
        ResourceLimits, SampleLimits, SegmentedConfig, SequenceBatch, SequenceConfig,
        SequenceOrigin, TemporalStorage,
    },
    train::{
        behavior::FixedBehavior, recurrent::TemporalTrainer,
        temporal_config::TemporalOffPolicyConfig, temporal_recording::TemporalReplayConfig,
    },
};
use serde::{Deserialize, Serialize};

fn observation(tick: usize) -> Observation {
    Observation::dense(
        vec![f32::from(tick.is_multiple_of(11))],
        Some(vec![true, true]),
    )
}
#[derive(Clone, Serialize, Deserialize)]
struct Continuing {
    tick: usize,
    horizon: Option<usize>,
}
impl Env for Continuing {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.tick = 0;
        Ok(EpisodeStart::fresh(observation(0)))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.action_space().check(action, None)?;
        self.tick += 1;
        Ok(Step {
            observation: observation(self.tick),
            reward: ((self.tick % 5) as f32).into(),
            terminated: false,
            truncated: self.horizon == Some(self.tick),
        })
    }
}
impl Snapshot for Continuing {
    fn validate_snapshot(&self) -> Result<()> {
        if self.horizon.is_some_and(|end| self.tick > end) {
            return Err(Error::State("invalid test environment clock".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct CounterActor {
    revision: u64,
}
impl StatefulActor for CounterActor {
    type State = u64;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<u64> {
        Ok(0)
    }
    fn validate_state(&self, _: &u64) -> Result<()> {
        Ok(())
    }
    fn assimilate(&self, prior: &u64, _: HistoryInput<'_>) -> Result<u64> {
        Ok(prior + 1)
    }
    fn decide(
        &self,
        _: &u64,
        _: &Observation,
        _: ActMode,
        _: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        Ok(StatefulDecision {
            action: Action::Discrete(0),
            log_prob: None,
        })
    }
}
impl Snapshot for CounterActor {
    fn validate_snapshot(&self) -> Result<()> {
        Ok(())
    }
}
impl ReconstructibleActor for CounterActor {
    type Model = CounterModel;
    fn representation(&self) -> Result<CounterModel> {
        Ok(CounterModel::new(0, self.revision))
    }
}

#[derive(Clone)]
struct CounterModel {
    key: u32,
    generation: u64,
    schema: AnchorSchema,
}
impl CounterModel {
    fn new(key: u32, generation: u64) -> Self {
        Self {
            key,
            generation,
            schema: AnchorSchema {
                codec: 101,
                codec_version: 1,
                representation_identity: [7; 32],
                dimensions: vec![1],
            },
        }
    }
}
impl ReconstructionModel for CounterModel {
    type State = u64;
    fn model_identity(&self) -> [u8; 32] {
        let mut identity = [0; 32];
        identity[0] = self.key as u8;
        let generation = if self.key == 0 {
            self.generation
        } else {
            self.generation / 4
        };
        identity[1..9].copy_from_slice(&generation.to_le_bytes());
        identity
    }
    fn generation(&self) -> u64 {
        self.generation
    }
    fn schema(&self) -> &AnchorSchema {
        &self.schema
    }
    fn initial_state(&self) -> Result<u64> {
        Ok(0)
    }
    fn validate_state(&self, state: &u64) -> Result<()> {
        if *state > 1_000_000 {
            return Err(Error::State("invalid counter codec payload".into()));
        }
        Ok(())
    }
    fn assimilate(&self, prior: &u64, _: HistoryInput<'_>) -> Result<u64> {
        Ok(prior + 1 + 2 * u64::from(self.key))
    }
    fn encode_state(&self, state: &u64) -> Result<Vec<u8>> {
        self.validate_state(state)?;
        Ok(state.to_le_bytes().to_vec())
    }
    fn decode_state(&self, payload: &[u8]) -> Result<u64> {
        let bytes: [u8; 8] = payload
            .try_into()
            .map_err(|_| Error::State("invalid counter codec size".into()))?;
        let state = u64::from_le_bytes(bytes);
        self.validate_state(&state)?;
        Ok(state)
    }
}
#[derive(Clone)]
struct Learner {
    updates: u64,
    anchored_updates: u64,
    config: ReconstructionConfig,
}
impl Learner {
    fn models(&self) -> Vec<(RepresentationId, CounterModel)> {
        (0..=1)
            .map(|key| (RepresentationId(key), CounterModel::new(key, self.updates)))
            .collect()
    }
}
impl StatefulLearner for Learner {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        batch.validate()?;
        for sample in &batch.samples {
            if let SequenceOrigin::Anchored(set) = &sample.origin {
                let mut states = Vec::new();
                for (id, model) in self.models() {
                    states.push(
                        decode_anchor(
                            self.config,
                            &model,
                            id,
                            sample.episode_id,
                            sample.history_start,
                            &set.anchors[&id],
                        )?
                        .0,
                    );
                }
                assert_eq!(
                    states[1],
                    3 * states[0],
                    "target stream reused behavior/online state"
                );
                self.anchored_updates += 1;
            }
        }
        self.updates += 1;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        })
    }
}
impl TemporalOffPolicyLearner for Learner {
    fn return_contract(&self) -> ReturnSpec {
        ReturnSpec {
            gamma: 0.9,
            n_step: 1,
        }
    }
}
impl BehaviorPublisher for Learner {
    type Actor = CounterActor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<CounterActor> {
        Ok(CounterActor { revision })
    }
}
impl ReplayRepresentations for Learner {
    type Model = CounterModel;
    fn replay_representations(&self) -> Result<Vec<(RepresentationId, CounterModel)>> {
        Ok(self.models())
    }
}
impl ReplayReconstruction for Learner {
    fn reconstruction_config(&self) -> ReconstructionConfig {
        self.config
    }
}

type Trainer = TemporalTrainer<SyncVectorEnv<Continuing>, CounterActor, FixedBehavior<()>>;
fn replay(mode: ReconstructionMode) -> TemporalReplayConfig {
    TemporalReplayConfig {
        version: 1,
        storage: TemporalStorage::Segmented(SegmentedConfig {
            segment_steps: 8,
            minimum_context: 16,
            required_representations: vec![RepresentationId(0), RepresentationId(1)],
            limits: ResourceLimits {
                max_transitions: 128,
                max_resident_bytes: 2_000_000,
                max_anchor_bytes: 100_000,
                max_anchors: 256,
                max_anchor_payload_bytes: 64,
                max_dimension_rank: 4,
            },
        }),
        reconstruction: ReconstructionConfig { version: 1, mode },
        sample_limits: SampleLimits {
            max_sample_tokens: 24,
            max_batch_tokens: 48,
            max_sample_payload_bytes: 20_000,
            max_batch_payload_bytes: 40_000,
        },
        anchor_interval: 8,
    }
}
fn build(lanes: usize, horizon: Option<usize>, warmup: u64, rate: f64) -> (Learner, Trainer) {
    let cfg = replay(ReconstructionMode::Anchored {
        max_generation_lag: None,
    });
    let learner = Learner {
        updates: 0,
        anchored_updates: 0,
        config: cfg.reconstruction,
    };
    let env = SyncVectorEnv::new(
        (0..lanes)
            .map(|_| Continuing { tick: 0, horizon })
            .collect(),
    )
    .unwrap();
    let mut trainer = TemporalTrainer::new(
        env,
        CounterActor { revision: 0 },
        TemporalOffPolicyConfig {
            replay_capacity: 128,
            batch_size: 2,
            warmup_transitions: warmup,
            n_step: 1,
            gamma: 0.9,
            updates_per_transition: rate,
        },
        SequenceConfig {
            burn_in: 4,
            learn_len: 8,
            n_step: 1,
        },
        FixedBehavior(()),
        27,
    )
    .unwrap();
    trainer.set_replay_config(cfg).unwrap();
    (learner, trainer)
}

#[test]
fn independent_streams_keep_learning_through_thousands_of_nonterminal_markers() {
    let (mut learner, mut trainer) = build(2, None, 0, 0.25);
    for _ in 0..2_000 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    assert_eq!(trainer.counters().env_steps, 4_000);
    assert!(learner.anchored_updates > 500);
    assert!(trainer.store().len() <= 128);
    assert_eq!(trainer.replay_streams().len(), 2);
    for lane in 0..2 {
        assert_eq!(trainer.collector().lane_timestep(lane), Some(2_000));
        let id = trainer.collector().lane_episode(lane).unwrap();
        let stream = &trainer.replay_streams()[&id];
        assert_eq!(stream.observation_index(), 2_001);
        assert_eq!(
            u64::from_le_bytes(
                stream.anchors[&RepresentationId(0)]
                    .payload
                    .as_slice()
                    .try_into()
                    .unwrap()
            ),
            2_001
        );
        assert_eq!(
            u64::from_le_bytes(
                stream.anchors[&RepresentationId(1)]
                    .payload
                    .as_slice()
                    .try_into()
                    .unwrap()
            ),
            6_003
        );
        assert!(!stream.anchors[&RepresentationId(0)].exact_ancestry);
    }
    trainer.validate_snapshot().unwrap();
    trainer.validate_reconstruction(&learner).unwrap();
    let state = serde_json::to_value(&trainer).unwrap();
    assert_eq!(state["version"], 3);
    assert!(state.get("replay_config").is_some());
}

#[test]
fn reconstruction_configuration_mismatch_and_plain_step_reject_before_mutation() {
    let (mut learner, mut trainer) = build(1, None, 0, 0.25);
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(trainer.step(&mut learner).is_err());
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    learner.config = ReconstructionConfig::default();
    assert!(trainer.step_with_reconstruction(&mut learner).is_err());
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    assert_eq!(learner.updates, 0);
    assert!(!trainer.is_poisoned());
}

#[test]
fn current_model_requirements_refresh_after_every_optimizer_call() {
    let (mut learner, mut trainer) = build(1, None, 140, 2.);
    let config = replay(ReconstructionMode::Anchored {
        max_generation_lag: Some(0),
    });
    learner.config = config.reconstruction;
    trainer.set_replay_config(config).unwrap();
    for _ in 0..140 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    let progress = trainer.step_with_reconstruction(&mut learner).unwrap();
    assert_eq!(
        progress.reports.len(),
        1,
        "stale requirements allowed a second optimizer call"
    );
    assert_eq!(learner.updates, 1);
    assert_eq!(trainer.update_credit(), 1.);
    assert!(!trainer.is_poisoned());
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(trainer.step_with_reconstruction(&mut learner).is_err());
    assert!(!trainer.is_poisoned());
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
}

#[test]
fn v3_recording_restores_rng_streams_and_real_resets_without_orphans() {
    let (mut learner, mut trainer) = build(2, Some(87), 0, 0.25);
    for _ in 0..150 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    let value = serde_json::to_value(&trainer).unwrap();
    let mut restored: Trainer = serde_json::from_value(value.clone()).unwrap();
    restored.validate_snapshot().unwrap();
    let mut resumed = learner.clone();
    for _ in 0..80 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
        restored.step_with_reconstruction(&mut resumed).unwrap();
        assert_eq!(trainer.last_windows(), restored.last_windows());
    }
    assert_eq!(
        serde_json::to_value(&trainer).unwrap(),
        serde_json::to_value(&restored).unwrap()
    );
    for bad in 0..5 {
        let mut corrupted = value.clone();
        if bad == 4 {
            corrupted.as_object_mut().unwrap().remove("replay_config");
            corrupted.as_object_mut().unwrap().remove("replay_streams");
            corrupted["version"] = 2.into();
            let invalid: Trainer = serde_json::from_value(corrupted).unwrap();
            assert!(
                invalid.validate_snapshot().is_err(),
                "legacy trainer must not reinterpret a segmented store"
            );
            continue;
        }
        if bad == 3 {
            corrupted["sequence_config"]["n_step"] = 0.into();
            let invalid: Trainer = serde_json::from_value(corrupted).unwrap();
            assert!(invalid.validate_snapshot().is_err());
            continue;
        }
        let streams = corrupted["replay_streams"].as_object_mut().unwrap();
        let first = streams.values_mut().next().unwrap();
        match bad {
            0 => first["anchors"]["0"]["observation_index"] = 0.into(),
            1 => {
                first["anchors"].as_object_mut().unwrap().remove("1");
            }
            _ => first["anchors"]["0"]["payload"] = serde_json::json!(vec![0; 65]),
        }
        let invalid: Trainer = serde_json::from_value(corrupted).unwrap();
        assert!(invalid.validate_snapshot().is_err(), "corruption {bad}");
    }
}

#[test]
fn reconstruction_snapshot_contract_and_pure_source_codec_errors_preserve_state() {
    let (mut learner, mut trainer) = build(1, None, 0, 0.25);
    let mut record = LearnerRecord::new("counter", ()).unwrap();
    record.temporal_returns = Some(learner.return_contract());
    assert!(trainer.validate_learner_record(&record).is_err());
    record.temporal_reconstruction = Some(learner.config);
    trainer.validate_learner_record(&record).unwrap();
    trainer.step_with_reconstruction(&mut learner).unwrap();
    let mut value = serde_json::to_value(&trainer).unwrap();
    value["replay_streams"]["0"]["anchors"]["0"]["payload"] = serde_json::json!(vec![255; 8]);
    let mut corrupt: Trainer = serde_json::from_value(value).unwrap();
    corrupt.validate_snapshot().unwrap(); // Opaque codec belongs to source, not generic metadata.
    let before = serde_json::to_value(&corrupt).unwrap();
    assert!(corrupt.step_with_reconstruction(&mut learner).is_err());
    assert_eq!(corrupt.counters().env_steps, 1);
    assert_eq!(corrupt.collector().lane_timestep(0), Some(1));
    assert!(!corrupt.is_poisoned());
    assert_eq!(serde_json::to_value(&corrupt).unwrap(), before);
}

#[test]
fn trainer_adoption_uses_explicit_actual_history_and_audits_revision_boundaries() {
    let (mut learner, mut trainer) = build(1, None, 0, 0.5);
    for _ in 0..200 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(
        trainer
            .adopt_lane(0, RepresentationId(0), ReconstructionConfig::default())
            .is_err()
    );
    assert!(
        trainer
            .adopt_lane(0, RepresentationId(99), learner.config)
            .is_err()
    );
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    let clock = trainer.collector().lane_timestep(0);
    let state = *trainer.collector().lane_state(0).unwrap();
    let decision = trainer
        .adopt_lane(0, RepresentationId(0), learner.config)
        .unwrap()
        .unwrap();
    assert!(decision.approximate);
    assert_eq!(trainer.collector().lane_timestep(0), clock);
    assert_eq!(*trainer.collector().lane_state(0).unwrap(), state);
    assert_eq!(trainer.collector().lane_adoptions(0).unwrap().len(), 1);
    for _ in 0..32 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    trainer
        .adopt_lane(0, RepresentationId(0), learner.config)
        .unwrap();
    trainer.validate_snapshot().unwrap();
    let mut corrupt = serde_json::to_value(&trainer).unwrap();
    corrupt["store"]["episodes"][0]["steps"][0]["behavior"]["revision"] = 777.into();
    let invalid: Trainer = serde_json::from_value(corrupt).unwrap();
    assert!(invalid.validate_snapshot().is_err());
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(
        trainer
            .adopt_lane(0, RepresentationId(0), learner.config)
            .is_err()
    );
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
}

#[test]
fn exact_adoption_requires_reset_origin_and_ended_lanes_never_adopt() {
    let (mut learner, mut trainer) = build(1, Some(8), 0, 0.5);
    for _ in 0..6 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    assert!(
        trainer
            .adopt_lane(0, RepresentationId(0), ReconstructionConfig::default())
            .unwrap()
            .is_none()
    );
    for _ in 0..2 {
        trainer.step_with_reconstruction(&mut learner).unwrap();
    }
    trainer.validate_snapshot().unwrap();
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(
        trainer
            .adopt_lane(0, RepresentationId(0), learner.config)
            .is_err()
    );
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
}
