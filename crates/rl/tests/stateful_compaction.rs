use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{
        stateful::{RetainedHistory, StatefulCollector},
        vector_env::SyncVectorEnv,
    },
    core::{
        checkpoint::Snapshot,
        stateful::{
            HistoryInput, PreviousInput, ReconstructionHistory, StatefulActor, StatefulDecision,
        },
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

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
        let Action::Discrete(action) = action else {
            unreachable!()
        };
        Ok(Step {
            observation: Observation::new(vec![self.tick as f32]),
            reward: (self.tick as f32 + *action as f32 * 0.25).into(),
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
    fn assimilate(&self, state: &u64, input: HistoryInput<'_>) -> Result<u64> {
        let mut next = state
            .wrapping_mul(17)
            .wrapping_add(input.observation.dense_values()[0].to_bits() as u64)
            .wrapping_add(self.revision);
        if let Some(previous) = input.previous {
            let Action::Discrete(action) = previous.action else {
                unreachable!()
            };
            next = next
                .wrapping_add(*action as u64)
                .wrapping_add(previous.reward.to_bits() as u64);
        }
        Ok(next)
    }
    fn decide(
        &self,
        _: &u64,
        _: &Observation,
        _: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        Ok(StatefulDecision {
            action: Action::Discrete(usize::from(rng.random::<bool>())),
            log_prob: Some(-0.),
        })
    }
}
impl Snapshot for CounterActor {
    fn validate_snapshot(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Default)]
struct Tape(Vec<TemporalTransition>);
impl TemporalSink for Tape {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> Result<()> {
        Ok(())
    }
    fn push(&mut self, transition: &TemporalTransition) -> Result<()> {
        self.0.push(transition.clone());
        Ok(())
    }
}

type Runtime = StatefulCollector<SyncVectorEnv<Clock>, CounterActor>;
fn runtime(limit: usize) -> Runtime {
    StatefulCollector::new(
        SyncVectorEnv::new(vec![Clock {
            tick: 0,
            limit,
            resets: 0,
        }])
        .unwrap(),
        CounterActor { revision: 0 },
        ChaCha8Rng::seed_from_u64(19).random(),
    )
    .unwrap()
}

#[derive(Clone)]
struct History {
    observations: Vec<Observation>,
    actions: Vec<Action>,
    rewards: Vec<f32>,
    behavior: Vec<BehaviorMetadata>,
}
impl History {
    fn from(runtime: &Runtime, tape: &Tape) -> Self {
        let id = runtime.lane_episode(0).unwrap();
        let steps: Vec<_> = tape.0.iter().filter(|t| t.episode_id == id).collect();
        let mut observations = vec![steps.first().map_or_else(
            || runtime.observations()[0].clone(),
            |t| t.observation.clone(),
        )];
        observations.extend(steps.iter().map(|t| t.next_observation.clone()));
        Self {
            observations,
            actions: steps.iter().map(|t| t.action.clone()).collect(),
            rewards: steps.iter().map(|t| t.reward.total()).collect(),
            behavior: steps.iter().map(|t| t.behavior.clone()).collect(),
        }
    }
    fn retained(&self) -> RetainedHistory<'_> {
        RetainedHistory {
            base_timestep: 0,
            previous: None,
            observations: &self.observations,
            actions: &self.actions,
            rewards: rl::core::stateful::HistoryRewards::Totals(&self.rewards),
        }
    }
    fn full(&self) -> ReconstructionHistory<'_> {
        ReconstructionHistory {
            observations: &self.observations,
            actions: &self.actions,
            rewards: rl::core::stateful::HistoryRewards::Totals(&self.rewards),
        }
    }
}

fn handoff(runtime: &mut Runtime, tape: &Tape, revision: u64) -> History {
    let history = History::from(runtime, tape);
    runtime.publish(CounterActor { revision }).unwrap();
    runtime.reconcile_lane(0, history.full()).unwrap();
    history
}

#[test]
fn compaction_is_opt_in_and_preserves_the_existing_ledger_limit() {
    let mut runtime = runtime(200);
    let initial = serde_json::to_value(&runtime).unwrap();
    assert_eq!(initial["version"], 1);
    assert!(initial.get("compaction_count").is_none());
    assert!(initial["agents"][0].get("adoption_base").is_none());
    let mut tape = Tape::default();
    runtime.collect(3, ActMode::Sample, &mut tape).unwrap();
    for revision in 1..=64 {
        handoff(&mut runtime, &tape, revision);
    }
    let history = History::from(&runtime, &tape);
    runtime.publish(CounterActor { revision: 65 }).unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    assert_eq!(before["version"], 3);
    assert!(before.get("compaction_count").is_none());
    assert!(runtime.reconcile_lane(0, history.full()).is_err());
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());

    runtime
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    let base = runtime.lane_adoption_base(0).unwrap();
    assert_eq!(base.version, 1);
    assert_eq!(base.episode_id, 0);
    assert_eq!(base.observation_index, 3);
    assert_eq!(base.revision, 64);
    assert_eq!(base.retired_count, 64);
    assert_eq!(runtime.compaction_count(), 1);
    assert!(runtime.lane_adoptions(0).unwrap().is_empty());
    assert_eq!(runtime.lane_behavior_revision_at(0, 2), None);
    assert_eq!(runtime.lane_behavior_revision_at(0, 3), Some(64));
    assert_eq!(serde_json::to_value(&runtime).unwrap()["version"], 4);
    runtime.reconcile_lane(0, history.full()).unwrap();
    assert_eq!(runtime.adoption_count(), 65);
    assert_eq!(serde_json::to_value(&runtime).unwrap()["version"], 4);
    runtime.validate_snapshot().unwrap();
}

#[test]
fn more_than_128_handoffs_preserve_one_physical_episode_and_immutable_tape() {
    let mut runtime = runtime(200);
    let mut tape = Tape::default();
    for revision in 1..=137 {
        runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
        let history = handoff(&mut runtime, &tape, revision);
        let before = serde_json::to_value(&runtime).unwrap();
        let recorded = tape.0.clone();
        runtime
            .compact_lane_revisions(0, history.retained(), &history.behavior)
            .unwrap();
        let after = serde_json::to_value(&runtime).unwrap();
        assert_eq!(before["driver"], after["driver"]);
        for key in ["actor", "state", "rng", "episode_id", "previous", "begun"] {
            assert_eq!(before["agents"][0][key], after["agents"][0][key]);
        }
        assert_eq!(recorded, tape.0);
        assert_eq!(runtime.lane_episode(0), Some(0));
        assert_eq!(runtime.lane_timestep(0), Some(revision));
        assert_eq!(runtime.compaction_count(), revision);
        assert_eq!(runtime.adoption_count(), revision);
        assert_eq!(
            runtime.lane_adoption_base(0).unwrap().retired_count,
            revision
        );
        runtime
            .validate_lane_revision_prefix(0, history.retained(), &history.behavior)
            .unwrap();
        runtime.validate_snapshot().unwrap();
    }
    assert_eq!(
        serde_json::to_value(&runtime).unwrap()["driver"]["env"]["envs"][0]["resets"],
        1
    );
}

#[test]
fn corrupted_public_history_and_behavior_fail_without_mutation_or_poisoning() {
    let mut runtime = runtime(40);
    let mut tape = Tape::default();
    runtime.collect(3, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut runtime, &tape, 1);
    runtime
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    runtime.collect(2, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut runtime, &tape, 2);
    let before = serde_json::to_value(&runtime).unwrap();
    for corruption in 0..9 {
        let mut bad = history.clone();
        match corruption {
            0 => bad.observations[0].as_dense_mut().unwrap().values[0] = f32::from_bits(1),
            1 => {
                bad.actions[0] = Action::Discrete(
                    1 - match bad.actions[0] {
                        Action::Discrete(a) => a,
                        _ => unreachable!(),
                    },
                )
            }
            2 => bad.rewards[0] += 0.5,
            3 => bad.behavior[0].revision = 99,
            4 => bad.behavior[0].log_prob = Some(0.),
            5 => bad.behavior[3].revision = 0,
            6 => {
                bad.behavior.pop();
            }
            7 => bad.behavior[0].log_prob = Some(f32::NAN),
            _ => {
                bad.observations
                    .last_mut()
                    .unwrap()
                    .as_dense_mut()
                    .unwrap()
                    .values[0] += 1.
            }
        }
        assert!(
            runtime
                .validate_lane_revision_prefix(0, bad.retained(), &bad.behavior)
                .is_err()
        );
        assert!(
            runtime
                .compact_lane_revisions(0, bad.retained(), &bad.behavior)
                .is_err()
        );
        assert_eq!(before, serde_json::to_value(&runtime).unwrap());
        runtime.validate_local().unwrap();
    }
    let suffix = RetainedHistory {
        base_timestep: 1,
        previous: Some(PreviousInput {
            factual: None,
            action: &history.actions[0],
            reward: history.rewards[0],
        }),
        observations: &history.observations[1..],
        actions: &history.actions[1..],
        rewards: rl::core::stateful::HistoryRewards::Totals(&history.rewards[1..]),
    };
    assert!(
        runtime
            .compact_lane_revisions(0, suffix, &history.behavior[1..])
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());
    runtime
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
}

#[test]
fn malformed_compaction_base_and_revision_suffix_are_rejected_before_interaction() {
    let mut runtime = runtime(40);
    let mut tape = Tape::default();
    runtime.collect(3, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut runtime, &tape, 1);
    runtime
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut runtime, &tape, 2);
    let snapshot = serde_json::to_value(&runtime).unwrap();
    for corruption in 0..10 {
        let mut value = snapshot.clone();
        match corruption {
            0 => value["version"] = 3.into(),
            1 => value["compaction_count"] = 0.into(),
            2 => value["agents"][0]["adoption_base"]["version"] = 2.into(),
            3 => value["agents"][0]["adoption_base"]["episode_id"] = 99.into(),
            4 => value["agents"][0]["adoption_base"]["observation_index"] = 99.into(),
            5 => value["agents"][0]["adoption_base"]["revision"] = 0.into(),
            6 => value["agents"][0]["adoption_base"]["retired_count"] = 99.into(),
            7 => {
                value["agents"][0]["adoption_base"]["history_identity"][0] = ((value["agents"][0]
                    ["adoption_base"]["history_identity"][0]
                    .as_u64()
                    .unwrap()
                    + 1)
                    % 256)
                    .into()
            }
            8 => value["agents"][0]["adoptions"][0]["from_revision"] = 0.into(),
            _ => value["adoption_count"] = 1.into(),
        }
        let mut loaded: Runtime = serde_json::from_value(value).unwrap();
        let before = serde_json::to_value(&loaded).unwrap();
        assert!(
            loaded
                .validate_lane_revision_prefix(0, history.retained(), &history.behavior)
                .is_err()
        );
        assert!(
            loaded
                .compact_lane_revisions(0, history.retained(), &history.behavior)
                .is_err()
        );
        assert_eq!(before, serde_json::to_value(&loaded).unwrap());
        if corruption != 7 {
            assert!(loaded.validate_local().is_err());
            let mut out = Tape::default();
            assert!(loaded.collect(1, ActMode::Sample, &mut out).is_err());
            assert!(out.0.is_empty());
            assert_eq!(before, serde_json::to_value(&loaded).unwrap());
        }
    }
}

#[test]
fn compaction_restore_preserves_suffix_audit_and_future_rng_stream() {
    let mut original = runtime(40);
    let mut tape = Tape::default();
    original.collect(3, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut original, &tape, 1);
    original
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    let before = serde_json::to_value(&original).unwrap();
    assert!(
        original
            .compact_lane_revisions(0, history.retained(), &history.behavior)
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&original).unwrap());
    original.collect(2, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut original, &tape, 2);
    let mut restored: Runtime =
        serde_json::from_value(serde_json::to_value(&original).unwrap()).unwrap();
    restored
        .validate_lane_revision_prefix(0, history.retained(), &history.behavior)
        .unwrap();
    let mut restored_tape = tape.clone();
    for runtime in [&mut original, &mut restored] {
        runtime
            .compact_lane_revisions(0, history.retained(), &history.behavior)
            .unwrap();
    }
    original.collect(4, ActMode::Sample, &mut tape).unwrap();
    restored
        .collect(4, ActMode::Sample, &mut restored_tape)
        .unwrap();
    assert_eq!(tape.0, restored_tape.0);
    assert_eq!(
        serde_json::to_value(&original).unwrap(),
        serde_json::to_value(&restored).unwrap()
    );
    original.collect(1, ActMode::Sample, &mut tape).unwrap();
    restored
        .collect(1, ActMode::Sample, &mut restored_tape)
        .unwrap();
    assert_eq!(tape.0.last(), restored_tape.0.last());
}

#[test]
fn genuine_reset_clears_lane_base_but_preserves_monotonic_compaction_count() {
    let mut runtime = runtime(4);
    let mut tape = Tape::default();
    runtime.collect(2, ActMode::Sample, &mut tape).unwrap();
    let history = handoff(&mut runtime, &tape, 1);
    runtime
        .compact_lane_revisions(0, history.retained(), &history.behavior)
        .unwrap();
    runtime.collect(2, ActMode::Sample, &mut tape).unwrap();
    let ended = History::from(&runtime, &tape);
    runtime
        .validate_lane_revision_prefix(0, ended.retained(), &ended.behavior)
        .unwrap();
    let before = serde_json::to_value(&runtime).unwrap();
    assert!(
        runtime
            .compact_lane_revisions(0, ended.retained(), &ended.behavior)
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&runtime).unwrap());
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    assert_eq!(runtime.lane_episode(0), Some(1));
    assert!(runtime.lane_adoption_base(0).is_none());
    assert!(runtime.lane_adoptions(0).unwrap().is_empty());
    assert_eq!(runtime.compaction_count(), 1);
    assert_eq!(runtime.adoption_count(), 1);
    assert_eq!(runtime.lane_behavior_revision_at(0, 0), Some(1));
    assert_eq!(serde_json::to_value(&runtime).unwrap()["version"], 4);
    runtime.validate_snapshot().unwrap();
}
