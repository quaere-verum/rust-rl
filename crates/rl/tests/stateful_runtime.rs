use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::stateful::*,
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Serialize, Deserialize)]
struct RecordingSink {
    transitions: Vec<rl::core::temporal::TemporalTransition>,
}
impl rl::core::temporal::TemporalSink for RecordingSink {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> rl::Result<()> {
        Ok(())
    }
    fn push(&mut self, t: &rl::core::temporal::TemporalTransition) -> rl::Result<()> {
        self.transitions.push(t.clone());
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CounterActor {
    revision: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Memory {
    observations: Vec<f32>,
    rewards: Vec<f32>,
    actions: Vec<Action>,
}
impl StatefulActor for CounterActor {
    type State = Memory;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> rl::Result<Memory> {
        Ok(Memory {
            observations: vec![],
            rewards: vec![],
            actions: vec![],
        })
    }
    fn validate_state(&self, s: &Memory) -> rl::Result<()> {
        if s.observations.len() == s.rewards.len() + 1 || s.observations.is_empty() {
            Ok(())
        } else {
            Err(rl::Error::State("bad memory".into()))
        }
    }
    fn assimilate(&self, s: &Memory, input: HistoryInput<'_>) -> rl::Result<Memory> {
        let mut s = s.clone();
        s.observations.push(input.observation.dense_values()[0]);
        if let Some(p) = input.previous {
            s.rewards.push(p.reward);
            s.actions.push(p.action.clone());
        }
        Ok(s)
    }
    fn decide(
        &self,
        s: &Memory,
        _: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> rl::Result<StatefulDecision> {
        let a = if mode == ActMode::Sample {
            rng.random_range(0..2)
        } else {
            s.observations.len() % 2
        };
        Ok(StatefulDecision {
            action: Action::Discrete(a),
            log_prob: None,
        })
    }
}
impl Snapshot for CounterActor {
    fn validate_snapshot(&self) -> rl::Result<()> {
        Ok(())
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Clock {
    t: usize,
    limit: usize,
    ended: bool,
}
impl Env for Clock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> rl::Result<EpisodeStart> {
        self.t = 0;
        self.ended = false;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, _: &Action) -> rl::Result<Step> {
        assert!(!self.ended);
        self.t += 1;
        self.ended = self.t == self.limit;
        Ok(Step {
            observation: Observation::new(vec![self.t as f32]),
            reward: (self.t as f32).into(),
            terminated: self.ended,
            truncated: false,
        })
    }
}
impl Snapshot for Clock {
    fn validate_snapshot(&self) -> rl::Result<()> {
        Ok(())
    }
}
fn runtime() -> StatefulCollector<SyncVectorEnv<Clock>, CounterActor> {
    StatefulCollector::new(
        SyncVectorEnv::new(vec![
            Clock {
                t: 0,
                limit: 4,
                ended: true,
            },
            Clock {
                t: 0,
                limit: 2,
                ended: true,
            },
        ])
        .unwrap(),
        CounterActor { revision: 0 },
        47,
    )
    .unwrap()
}

#[test]
fn post_observation_clock_persists_across_fragments_and_nonterminal_events() {
    let mut r = runtime();
    let mut sink = RecordingSink::default();
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0.]);
    r.collect(1, ActMode::Deterministic, &mut sink).unwrap();
    r.collect(1, ActMode::Deterministic, &mut sink).unwrap();
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0., 1., 2.]);
    assert_eq!(r.lane_state(0).unwrap().rewards, vec![1., 2.]);
    assert_eq!(
        r.lane_state(0).unwrap().actions,
        vec![Action::Discrete(1), Action::Discrete(0)]
    );
    r.publish(CounterActor { revision: 1 }).unwrap();
    r.collect(1, ActMode::Deterministic, &mut sink).unwrap();
    assert_eq!(r.lane_revision(0).unwrap(), 0);
    assert_eq!(r.lane_revision(1).unwrap(), 1);
    assert_eq!(r.lane_state(1).unwrap().observations, vec![0., 1.]);
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0., 1., 2., 3.]);
}

#[test]
fn selected_lanes_do_not_consume_other_rng_or_state_and_restore_exactly() {
    let mut r = runtime();
    let mut sink = RecordingSink::default();
    r.collect(1, ActMode::Sample, &mut sink).unwrap();
    let before = r.lane_state(1).unwrap().clone();
    r.tick(&[true, false], ActMode::Sample, &mut sink).unwrap();
    assert_eq!(r.lane_state(1).unwrap(), &before);
    r.validate_snapshot().unwrap();
    let mut record = serde_json::to_value(&r).unwrap();
    assert!(record.get("selection_rngs").is_none());
    record["selection_rngs"] = serde_json::Value::Null;
    let mut restored: StatefulCollector<SyncVectorEnv<Clock>, CounterActor> =
        serde_json::from_value(record.clone()).unwrap();
    let mut delegated = record.clone();
    delegated["selection_rngs"] = serde_json::json!([record["agents"][0]["rng"]]);
    assert!(
        serde_json::from_value::<StatefulCollector<SyncVectorEnv<Clock>, CounterActor>>(delegated)
            .is_err()
    );
    record["version"] = 11.into();
    let invalid: StatefulCollector<SyncVectorEnv<Clock>, CounterActor> =
        serde_json::from_value(record).unwrap();
    assert!(invalid.validate_snapshot().is_err());
    let mut other = sink.clone();
    for _ in 0..6 {
        r.collect(1, ActMode::Sample, &mut sink).unwrap();
        restored.collect(1, ActMode::Sample, &mut other).unwrap();
    }
    assert_eq!(
        serde_json::to_value(&r).unwrap(),
        serde_json::to_value(restored).unwrap()
    );
    assert_eq!(
        serde_json::to_value(&sink).unwrap(),
        serde_json::to_value(other).unwrap()
    );
}

#[test]
fn shared_history_features_never_shift_previous_action_reward() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let o = Observation::new(vec![3.]);
    assert_eq!(
        spec.encode(HistoryInput {
            observation: &o,
            previous: None
        })
        .unwrap(),
        vec![3., 0., 0., 0., 0.]
    );
    assert_eq!(
        spec.encode(HistoryInput {
            observation: &o,
            previous: Some(PreviousInput {
                factual: None,
                action: &Action::Discrete(1),
                reward: 7.
            })
        })
        .unwrap(),
        vec![3., 0., 1., 7., 1.]
    );
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    assert!(rng.random::<f32>().is_finite());
}

#[test]
fn sink_failure_poison_requires_recovery_without_committing_memory() {
    struct Reject;
    impl rl::core::temporal::TemporalSink for Reject {
        fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> rl::Result<()> {
            Ok(())
        }
        fn push(&mut self, _: &rl::core::temporal::TemporalTransition) -> rl::Result<()> {
            Err(rl::Error::State("injected sink failure".into()))
        }
    }
    let mut r = runtime();
    let before = r.lane_state(0).unwrap().clone();
    assert!(r.collect(1, ActMode::Sample, &mut Reject).is_err());
    assert_eq!(r.lane_state(0).unwrap(), &before);
    assert!(r.validate_snapshot().is_err());
    let mut sink = RecordingSink::default();
    assert!(r.collect(1, ActMode::Sample, &mut sink).is_err());
    r.reset_all(&mut sink).unwrap();
    r.collect(1, ActMode::Sample, &mut sink).unwrap();
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0., 1.]);
    assert!(r.lane_episode(0).unwrap() >= 2);
}

#[test]
fn other_lane_random_stream_is_independent_of_selected_steps() {
    let mut a = runtime();
    let mut b = runtime();
    let mut sa = RecordingSink::default();
    let mut sb = RecordingSink::default();
    a.tick(&[true, false], ActMode::Sample, &mut sa).unwrap();
    a.tick(&[false, true], ActMode::Sample, &mut sa).unwrap();
    b.tick(&[false, true], ActMode::Sample, &mut sb).unwrap();
    assert_eq!(
        sa.transitions.last().unwrap().action,
        sb.transitions.last().unwrap().action
    );
    assert_eq!(a.lane_state(1), b.lane_state(1));
    let mut json = serde_json::to_value(&a).unwrap();
    json["version"] = 99.into();
    let incompatible: StatefulCollector<SyncVectorEnv<Clock>, CounterActor> =
        serde_json::from_value(json).unwrap();
    assert!(incompatible.validate_snapshot().is_err());
}

#[test]
fn explicit_history_reconciliation_changes_revision_without_false_reset() {
    let mut r = runtime();
    let mut sink = RecordingSink::default();
    r.collect(2, ActMode::Deterministic, &mut sink).unwrap();
    let tape: Vec<_> = sink
        .transitions
        .iter()
        .filter(|t| t.episode_id == 0)
        .collect();
    let mut observations = vec![tape[0].observation.clone()];
    observations.extend(tape.iter().map(|t| t.next_observation.clone()));
    let actions: Vec<_> = tape.iter().map(|t| t.action.clone()).collect();
    let rewards: Vec<_> = tape.iter().map(|t| t.reward.total()).collect();
    let before = r.lane_state(0).unwrap().clone();
    r.publish(CounterActor { revision: 1 }).unwrap();
    assert!(
        r.reconcile_lane(
            0,
            ReconstructionHistory {
                observations: &observations[..2],
                actions: &actions,
                rewards: rl::core::stateful::HistoryRewards::Totals(&rewards)
            }
        )
        .is_err()
    );
    assert_eq!(r.lane_revision(0), Some(0));
    r.reconcile_lane(
        0,
        ReconstructionHistory {
            observations: &observations,
            actions: &actions,
            rewards: rl::core::stateful::HistoryRewards::Totals(&rewards),
        },
    )
    .unwrap();
    assert_eq!(r.lane_revision(0), Some(1));
    assert_eq!(r.lane_state(0), Some(&before));
    r.collect(1, ActMode::Deterministic, &mut sink).unwrap();
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0., 1., 2., 3.]);
    assert_eq!(r.lane_episode(0), Some(0));
}

#[test]
fn recovery_abandons_a_tape_when_begin_commits_then_returns_an_error() {
    use rl::core::temporal::{TemporalSink, TemporalTransition};
    struct CommitThenFail {
        store: rl::data::temporal::EpisodeStore,
        fail: bool,
    }
    impl TemporalSink for CommitThenFail {
        fn begin_episode(&mut self, id: u64, obs: &EpisodeStart) -> rl::Result<()> {
            self.store.begin_episode(id, obs)?;
            if self.fail {
                self.fail = false;
                Err(rl::Error::State("post-commit begin failure".into()))
            } else {
                Ok(())
            }
        }
        fn push(&mut self, t: &TemporalTransition) -> rl::Result<()> {
            self.store.push(t)
        }
        fn abandon_episode(&mut self, id: u64) -> rl::Result<()> {
            self.store.abandon_episode(id)
        }
    }
    let mut r = StatefulCollector::new(
        SyncVectorEnv::new(vec![Clock {
            t: 0,
            limit: 4,
            ended: true,
        }])
        .unwrap(),
        CounterActor { revision: 0 },
        47,
    )
    .unwrap();
    let mut sink = CommitThenFail {
        store: rl::data::temporal::EpisodeStore::new(1, 1, ActionSpace::Discrete { count: 2 })
            .unwrap(),
        fail: true,
    };
    assert!(r.collect(1, ActMode::Sample, &mut sink).is_err());
    assert_eq!(sink.store.active_count(), 1);
    r.reset_all(&mut sink).unwrap();
    assert_eq!(sink.store.active_count(), 0);
    r.collect(1, ActMode::Sample, &mut sink).unwrap();
    assert_eq!(r.lane_state(0).unwrap().observations, vec![0., 1.]);
}

#[test]
fn stateless_adapter_preserves_actor_outputs_and_is_optionally_checkpointable() {
    #[derive(Clone, Serialize, Deserialize)]
    struct Fixed;
    impl Actor for Fixed {
        fn action_space(&self) -> &ActionSpace {
            &ActionSpace::Discrete { count: 2 }
        }
        fn act_into(
            &self,
            obs: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            out: &mut Vec<Action>,
        ) -> rl::Result<()> {
            out.clear();
            out.extend(
                obs.iter()
                    .map(|o| Action::Discrete(usize::from(o.dense_values()[0] > 0.))),
            );
            Ok(())
        }
    }
    impl Snapshot for Fixed {
        fn validate_snapshot(&self) -> rl::Result<()> {
            Ok(())
        }
    }
    let a = StatelessActor {
        actor: Fixed,
        width: 1,
        revision: 0,
    };
    a.validate_snapshot().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let obs = Observation::new(vec![2.]);
    assert_eq!(
        a.decide(&(), &obs, ActMode::Deterministic, &mut rng)
            .unwrap()
            .action,
        a.actor
            .act(&[obs], ActMode::Deterministic, &mut rng)
            .unwrap()[0]
    );
    let r = StatefulCollector::new(
        SyncVectorEnv::new(vec![Clock {
            t: 0,
            limit: 4,
            ended: true,
        }])
        .unwrap(),
        a,
        47,
    )
    .unwrap();
    r.validate_snapshot().unwrap();
}
