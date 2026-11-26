#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{
        stateful::{AdoptionSource, RetainedHistory, StatefulCollector},
        vector_env::SyncVectorEnv,
    },
    core::{
        anchors::{RepresentationId, StateAnchor},
        checkpoint::Snapshot,
        reconstruction::{
            ReconstructibleActor, ReconstructionConfig, ReconstructionMode, ReconstructionModel,
            advance_anchor, reset_anchor,
        },
        stateful::{
            HistoryInput, HistorySpec, PreviousInput, ReconstructionHistory, StatefulActor,
        },
        temporal::{TemporalSink, TemporalTransition},
    },
    nn::{
        network::Mlp,
        recurrent_actor::RecurrentActor,
        temporal::{GruEncoder, RecurrentQ},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

const REPRESENTATION: RepresentationId = RepresentationId(1);
#[derive(Clone, Serialize, Deserialize)]
struct Clock {
    tick: usize,
    limit: usize,
    resets: usize,
}
impl Env for Clock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.tick = 0;
        self.resets += 1;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.tick += 1;
        let Action::Discrete(a) = action else {
            unreachable!()
        };
        Ok(Step {
            observation: Observation::new(vec![self.tick as f32 * 0.1]),
            reward: (self.tick as f32 * 0.1 + *a as f32 * 0.25).into(),
            terminated: self.tick == self.limit,
            truncated: false,
        })
    }
}
impl Snapshot for Clock {
    fn validate_snapshot(&self) -> Result<()> {
        if self.limit > 0 && self.tick <= self.limit {
            Ok(())
        } else {
            Err(Error::State("invalid clock".into()))
        }
    }
}
#[derive(Clone, Default)]
struct Tape(Vec<TemporalTransition>);
impl TemporalSink for Tape {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> Result<()> {
        Ok(())
    }
    fn push(&mut self, t: &TemporalTransition) -> Result<()> {
        self.0.push(t.clone());
        Ok(())
    }
}
type Runtime = StatefulCollector<SyncVectorEnv<Clock>, RecurrentActor>;
fn actor(seed: u64, revision: u64, generation: u64) -> RecurrentActor {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    RecurrentActor::new(q, spec, 0.35, revision)
        .unwrap()
        .with_parameter_generation(generation)
}
fn runtime(limit: usize) -> Runtime {
    StatefulCollector::new(
        SyncVectorEnv::new(vec![
            Clock {
                tick: 0,
                limit,
                resets: 0,
            },
            Clock {
                tick: 0,
                limit: limit + 2,
                resets: 0,
            },
        ])
        .unwrap(),
        actor(17, 0, 1),
        73,
    )
    .unwrap()
}
struct History {
    observations: Vec<Observation>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
}
impl History {
    fn from(runtime: &Runtime, tape: &Tape, lane: usize) -> Self {
        let id = runtime.lane_episode(lane).unwrap();
        let steps = tape
            .0
            .iter()
            .filter(|t| t.episode_id == id)
            .collect::<Vec<_>>();
        let mut observations = vec![steps.first().map_or_else(
            || runtime.observations()[lane].clone(),
            |t| t.observation.clone(),
        )];
        observations.extend(steps.iter().map(|t| t.next_observation.clone()));
        Self {
            observations,
            actions: steps.iter().map(|t| t.action.clone()).collect(),
            rewards: steps.iter().map(|t| t.reward.total()).collect(),
        }
    }
    fn full(&self) -> ReconstructionHistory<'_> {
        ReconstructionHistory {
            observations: &self.observations,
            actions: &self.actions,
            rewards: rl::core::stateful::HistoryRewards::Totals(&self.rewards),
        }
    }
    fn retained(&self, base: usize) -> RetainedHistory<'_> {
        RetainedHistory {
            base_timestep: base,
            previous: base.checked_sub(1).map(|i| PreviousInput {
                factual: None,
                action: &self.actions[i],
                reward: self.rewards[i],
            }),
            observations: &self.observations[base..],
            actions: &self.actions[base..],
            rewards: rl::core::stateful::HistoryRewards::Totals(&self.rewards[base..]),
        }
    }
    fn state(&self, actor: &RecurrentActor) -> Vec<f32> {
        let mut state = actor.initial_state().unwrap();
        for (i, observation) in self.observations.iter().enumerate() {
            state = actor
                .assimilate(
                    &state,
                    HistoryInput {
                        observation,
                        previous: i.checked_sub(1).map(|j| PreviousInput {
                            factual: None,
                            action: &self.actions[j],
                            reward: self.rewards[j],
                        }),
                    },
                )
                .unwrap();
        }
        state
    }
    fn anchor(&self, actor: &RecurrentActor, episode: u64, base: usize) -> StateAnchor {
        let model = actor.representation().unwrap();
        let mut anchor = reset_anchor(&model, REPRESENTATION, episode).unwrap();
        for i in 0..base {
            anchor = advance_anchor(
                ReconstructionConfig::default(),
                &model,
                &anchor,
                HistoryInput {
                    observation: &self.observations[i],
                    previous: i.checked_sub(1).map(|j| PreviousInput {
                        factual: None,
                        action: &self.actions[j],
                        reward: self.rewards[j],
                    }),
                },
            )
            .unwrap()
            .0;
        }
        anchor
    }
}
fn approximate() -> ReconstructionConfig {
    ReconstructionConfig {
        version: 1,
        mode: ReconstructionMode::Anchored {
            max_generation_lag: Some(2),
        },
    }
}
fn unchanged_driver_and_streams(before: &serde_json::Value, after: &serde_json::Value) {
    assert_eq!(before["driver"], after["driver"]);
    for lane in 0..2 {
        for key in ["rng", "episode_id", "previous", "begun"] {
            assert_eq!(before["agents"][lane][key], after["agents"][lane][key]);
        }
    }
    assert_eq!(before["agents"][1], after["agents"][1]);
}

#[test]
fn actor_parameter_generation_is_separate_preserved_and_versioned() {
    let original = actor(17, 3, 0);
    let legacy = serde_json::to_value(&original).unwrap();
    assert_eq!(legacy["metadata"]["version"], 1);
    assert!(legacy["metadata"].get("parameter_generation").is_none());
    let aged = original.with_parameter_generation(12);
    let changed = aged.with_behavior(0.8, 99).unwrap();
    assert_eq!(changed.parameter_generation(), 12);
    assert_eq!(changed.revision(), 99);
    assert_eq!(changed.model_identity(), original.model_identity());
    assert!(std::ptr::eq(changed.model(), original.model()));
    assert_eq!(changed.representation().unwrap().generation(), 12);
    let saved = serde_json::to_value(&changed).unwrap();
    assert_eq!(saved["metadata"]["version"], 2);
    let restored: RecurrentActor = serde_json::from_value(saved.clone()).unwrap();
    assert_eq!(restored.parameter_generation(), 12);
    assert_eq!(restored.revision(), 99);
    assert!(restored.batch_compatible(&changed));
    let mut invalid = saved;
    invalid["metadata"]["version"] = 1.into();
    assert!(serde_json::from_value::<RecurrentActor>(invalid).is_err());
    let mut invalid = legacy;
    invalid["metadata"]["version"] = 2.into();
    assert!(serde_json::from_value::<RecurrentActor>(invalid).is_err());
}

#[test]
fn full_history_adoption_preserves_physical_lane_and_records_revision_boundary() {
    let mut runtime = runtime(12);
    let mut tape = Tape::default();
    runtime.collect(4, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    let next = actor(19, 1, 3);
    let expected = history.state(&next);
    runtime.publish(next).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    runtime.reconcile_lane(0, history.full()).unwrap();
    let after = serde_json::to_value(&runtime).unwrap();
    unchanged_driver_and_streams(&before, &after);
    assert_eq!(runtime.lane_state(0).unwrap(), &expected);
    assert_eq!(runtime.lane_actor(0).unwrap().parameter_generation(), 3);
    assert_eq!(after["version"], 3);
    assert_eq!(runtime.adoption_count(), 1);
    let records = runtime.lane_adoptions(0).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].observation_index, 4);
    assert_eq!(records[0].from_revision, 0);
    assert_eq!(records[0].to_revision, 1);
    assert!(matches!(
        &records[0].source,
        AdoptionSource::FullHistory { .. }
    ));
    assert_eq!(runtime.lane_behavior_revision_at(0, 3), Some(0));
    assert_eq!(runtime.lane_behavior_revision_at(0, 4), Some(1));
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    assert_eq!(
        tape.0
            .iter()
            .find(|t| t.episode_id == 0 && t.timestep == 4)
            .unwrap()
            .behavior
            .revision,
        1
    );
}

#[test]
fn compatible_exact_boundary_reproduces_new_model_full_history_without_reset() {
    let mut runtime = runtime(12);
    let mut tape = Tape::default();
    runtime.collect(5, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    let next = actor(19, 1, 3);
    let anchor = history.anchor(&next, 0, 2);
    let expected = history.state(&next);
    runtime.publish(next).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    let decision = runtime
        .reconcile_lane_anchored(
            0,
            REPRESENTATION,
            ReconstructionConfig::default(),
            &anchor,
            history.retained(2),
        )
        .unwrap();
    assert!(!decision.approximate);
    assert_eq!(decision.observation_index, 2);
    assert_eq!(runtime.lane_state(0).unwrap(), &expected);
    unchanged_driver_and_streams(&before, &serde_json::to_value(&runtime).unwrap());
    runtime.validate_snapshot().unwrap();
}

#[test]
fn canonical_collector_rejects_anchors_without_numerical_provenance_atomically() {
    let mut runtime = StatefulCollector::new_with_inference_mode(
        SyncVectorEnv::new(vec![
            Clock {
                tick: 0,
                limit: 12,
                resets: 0,
            },
            Clock {
                tick: 0,
                limit: 14,
                resets: 0,
            },
        ])
        .unwrap(),
        actor(17, 0, 1),
        73,
        rl::collect::stateful::InferenceMode::CanonicalBatched,
    )
    .unwrap();
    let mut tape = Tape::default();
    runtime.collect(5, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    let next = actor(19, 1, 3);
    let anchor = history.anchor(&next, 0, 2);
    runtime.publish(next).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    let error = runtime
        .reconcile_lane_anchored(
            0,
            REPRESENTATION,
            ReconstructionConfig::default(),
            &anchor,
            history.retained(2),
        )
        .expect_err("unversioned numerical anchor must not enter canonical execution");
    assert!(error.to_string().contains("canonical"));
    assert_eq!(serde_json::to_value(&runtime).unwrap(), before);
}

#[test]
fn stale_boundary_adoption_is_explicit_and_checkpoint_future_is_exact() {
    let mut uninterrupted = runtime(20);
    let mut tape = Tape::default();
    uninterrupted
        .collect(5, ActMode::Sample, &mut tape)
        .unwrap();
    let history = History::from(&uninterrupted, &tape, 0);
    let anchor = history.anchor(uninterrupted.lane_actor(0).unwrap(), 0, 2);
    uninterrupted.publish(actor(19, 1, 3)).unwrap();
    let before = serde_json::to_value(&uninterrupted).unwrap();
    assert!(
        uninterrupted
            .reconcile_lane_anchored(
                0,
                REPRESENTATION,
                ReconstructionConfig::default(),
                &anchor,
                history.retained(2)
            )
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&uninterrupted).unwrap());
    let decision = uninterrupted
        .reconcile_lane_anchored(
            0,
            REPRESENTATION,
            approximate(),
            &anchor,
            history.retained(2),
        )
        .unwrap();
    assert!(decision.approximate);
    assert_eq!(decision.generation_lag, 2);
    let snapshot = serde_json::to_value(&uninterrupted).unwrap();
    let mut restored: Runtime = serde_json::from_value(snapshot.clone()).unwrap();
    let mut other = tape.clone();
    assert_eq!(snapshot["version"], 3);
    assert_eq!(uninterrupted.lane_adoptions(0), restored.lane_adoptions(0));
    for _ in 0..8 {
        uninterrupted
            .collect(1, ActMode::Sample, &mut tape)
            .unwrap();
        restored.collect(1, ActMode::Sample, &mut other).unwrap();
        assert_eq!(tape.0, other.0);
        assert_eq!(
            serde_json::to_value(&uninterrupted).unwrap(),
            serde_json::to_value(&restored).unwrap()
        );
    }
}

#[test]
fn bad_anchor_tape_and_preceding_inputs_fail_before_any_runtime_mutation() {
    let mut runtime = runtime(12);
    let mut tape = Tape::default();
    runtime.collect(4, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    let next = actor(19, 1, 3);
    let anchor = history.anchor(&next, 0, 4);
    runtime.publish(next).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    for corruption in 0..5 {
        let mut bad = anchor.clone();
        let mut retained = history.retained(4);
        match corruption {
            0 => bad.episode_id = 99,
            1 => bad.representation_id = RepresentationId(99),
            2 => bad.generation = 4,
            3 => retained.previous = None,
            _ => {
                retained.previous = Some(PreviousInput {
                    factual: None,
                    action: &history.actions[3],
                    reward: 99.,
                })
            }
        }
        assert!(
            runtime
                .reconcile_lane_anchored(
                    0,
                    REPRESENTATION,
                    ReconstructionConfig::default(),
                    &bad,
                    retained
                )
                .is_err()
        );
        assert_eq!(before, serde_json::to_value(&runtime).unwrap());
        runtime.validate_local().unwrap();
    }
    let mut short = history.retained(2);
    short.observations = &history.observations[2..4];
    assert!(
        runtime
            .reconcile_lane_anchored(
                0,
                REPRESENTATION,
                ReconstructionConfig::default(),
                &anchor,
                short
            )
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());
}

#[test]
fn ended_lanes_and_ledger_exhaustion_require_explicit_handling_without_fake_resets() {
    let mut runtime = runtime(4);
    let mut tape = Tape::default();
    runtime.collect(3, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    for revision in 1..=64 {
        runtime.publish(actor(17, revision, revision)).unwrap();
        runtime.reconcile_lane(0, history.full()).unwrap();
    }
    assert_eq!(runtime.lane_adoptions(0).unwrap().len(), 64);
    assert_eq!(runtime.adoption_count(), 64);
    runtime.publish(actor(17, 65, 65)).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    assert!(runtime.reconcile_lane(0, history.full()).is_err());
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    let ended = History::from(&runtime, &tape, 0);
    let before = serde_json::to_value(&runtime).unwrap();
    assert!(runtime.reconcile_lane(0, ended.full()).is_err());
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    assert!(runtime.lane_adoptions(0).unwrap().is_empty());
    assert_eq!(runtime.adoption_count(), 64);
    assert_eq!(serde_json::to_value(&runtime).unwrap()["version"], 3);
    runtime.validate_snapshot().unwrap();
}

#[test]
fn corrupted_adoption_ledger_is_rejected_before_interaction() {
    let mut runtime = runtime(12);
    let mut tape = Tape::default();
    runtime.collect(4, ActMode::Sample, &mut tape).unwrap();
    let history = History::from(&runtime, &tape, 0);
    runtime.publish(actor(19, 1, 3)).unwrap();
    runtime.reconcile_lane(0, history.full()).unwrap();
    let snapshot = serde_json::to_value(&runtime).unwrap();
    for corruption in 0..6 {
        let mut value = snapshot.clone();
        match corruption {
            0 => value["version"] = 1.into(),
            1 => value["adoption_count"] = 0.into(),
            2 => value["agents"][0]["adoptions"][0]["episode_id"] = 99.into(),
            3 => value["agents"][0]["adoptions"][0]["observation_index"] = 99.into(),
            4 => value["agents"][0]["adoptions"][0]["to_revision"] = 0.into(),
            _ => {
                value["agents"][0]["adoptions"][0]["source"]["FullHistory"]["origin_index"] =
                    1.into()
            }
        }
        let mut loaded: Runtime = serde_json::from_value(value).unwrap();
        let before = serde_json::to_value(&loaded).unwrap();
        let mut sink = Tape::default();
        assert!(loaded.validate_local().is_err());
        assert!(loaded.collect(1, ActMode::Sample, &mut sink).is_err());
        assert_eq!(before, serde_json::to_value(&loaded).unwrap());
        assert!(sink.0.is_empty());
    }
}

#[test]
fn revision_lookup_on_an_unaudited_short_clock_vector_never_panics() {
    let mut value = serde_json::to_value(runtime(12)).unwrap();
    value["driver"]["episode_steps"] = serde_json::json!([]);
    let loaded: Runtime = serde_json::from_value(value).unwrap();
    assert_eq!(loaded.lane_behavior_revision_at(0, 0), None);
    assert!(loaded.validate_snapshot().is_err());
}
