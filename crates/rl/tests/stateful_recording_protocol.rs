use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{
        stateful::{InferenceMode, StatefulCollector},
        vector_env::SyncVectorEnv,
    },
    core::{
        recording::{RecordingKey, StatefulRecording},
        stateful::*,
        temporal::{TemporalSink, TemporalTransition},
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
struct Clock {
    time: usize,
    limit: usize,
    truncate: bool,
}
impl Env for Clock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> rl::Result<EpisodeStart> {
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> rl::Result<Step> {
        if self.time >= self.limit {
            return Err(Error::State("ended clock".into()));
        }
        self.action_space().check(action, None)?;
        self.time += 1;
        let ended = self.time == self.limit;
        let mut observation = Observation::new(vec![self.time as f32]);
        if ended && !self.truncate {
            observation.as_dense_mut().unwrap().action_mask = Some(vec![false; 2]);
        }
        Ok(Step {
            observation,
            reward: (self.time as f32).into(),
            terminated: ended && !self.truncate,
            truncated: ended && self.truncate,
        })
    }
}
impl Snapshot for Clock {
    fn validate_snapshot(&self) -> rl::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct Counter;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Memory {
    last: f32,
    count: usize,
    reward_sum: f32,
}
impl StatefulActor for Counter {
    type State = Memory;
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        0
    }
    fn initial_state(&self) -> rl::Result<Memory> {
        Ok(Memory {
            last: -1.,
            count: 0,
            reward_sum: 0.,
        })
    }
    fn validate_state(&self, s: &Memory) -> rl::Result<()> {
        if s.last.is_finite() && s.reward_sum.is_finite() {
            Ok(())
        } else {
            Err(Error::State("bad memory".into()))
        }
    }
    fn assimilate(&self, s: &Memory, input: HistoryInput<'_>) -> rl::Result<Memory> {
        Ok(Memory {
            last: input.observation.dense_values()[0],
            count: s.count + 1,
            reward_sum: s.reward_sum + input.previous.map_or(0., |p| p.reward),
        })
    }
    fn decide(
        &self,
        _: &Memory,
        _: &Observation,
        _: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> rl::Result<StatefulDecision> {
        Ok(StatefulDecision {
            action: Action::Discrete(rng.random_range(0..2)),
            log_prob: Some(-std::f32::consts::LN_2),
        })
    }
    fn batch_compatible(&self, _: &Self) -> bool {
        true
    }
}
impl Snapshot for Counter {
    fn validate_snapshot(&self) -> rl::Result<()> {
        Ok(())
    }
}

#[derive(Default, Clone)]
struct Trace {
    transitions: Vec<TemporalTransition>,
    pending: BTreeMap<(u64, usize), Action>,
    candidates: Vec<(bool, bool, Memory)>,
    fail_once: bool,
    abandoned: usize,
}
impl TemporalSink for Trace {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> rl::Result<()> {
        Ok(())
    }
    fn push(&mut self, t: &TemporalTransition) -> rl::Result<()> {
        self.transitions.push(t.clone());
        Ok(())
    }
    fn abandon_episode(&mut self, id: u64) -> rl::Result<()> {
        self.abandoned += 1;
        self.transitions.retain(|t| t.episode_id != id);
        self.pending.retain(|(episode, _), _| *episode != id);
        Ok(())
    }
}
impl StatefulRecording<Counter> for Trace {
    fn decide(
        &mut self,
        key: RecordingKey,
        actor: &Counter,
        state: &Memory,
        obs: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> rl::Result<StatefulDecision> {
        let out = actor.decide(state, obs, mode, rng)?;
        assert!(
            self.pending
                .insert((key.episode_id, key.timestep), out.action.clone())
                .is_none()
        );
        Ok(out)
    }
    fn push_with_state(
        &mut self,
        t: &TemporalTransition,
        _: &Counter,
        state: &Memory,
    ) -> rl::Result<()> {
        assert_eq!(
            self.pending.remove(&(t.episode_id, t.timestep)),
            Some(t.action.clone())
        );
        assert_eq!(state.last, t.next_observation.dense_values()[0]);
        assert_eq!(state.count, t.timestep + 2);
        self.candidates
            .push((t.terminated, t.truncated, state.clone()));
        self.push(t)?;
        if self.fail_once {
            self.fail_once = false;
            return Err(Error::State("persisted sink failure".into()));
        }
        Ok(())
    }
}
fn runtime(lanes: usize, truncate: bool) -> StatefulCollector<SyncVectorEnv<Clock>, Counter> {
    StatefulCollector::new_with_inference_mode(
        SyncVectorEnv::new(
            (0..lanes)
                .map(|i| Clock {
                    time: 0,
                    limit: 2 + i % 3,
                    truncate,
                })
                .collect(),
        )
        .unwrap(),
        Counter,
        43,
        InferenceMode::Batched,
    )
    .unwrap()
}

#[test]
fn keyed_recording_preserves_basic_actions_rngs_and_lane_chronology() {
    for lanes in [1, 4, 16] {
        let mut basic = runtime(lanes, false);
        let mut detailed = runtime(lanes, false);
        let mut left = Trace::default();
        let mut right = Trace::default();
        for tick in 0..9 {
            let active: Vec<_> = (0..lanes).map(|i| (i + tick) % 4 != 3).collect();
            basic.tick(&active, ActMode::Sample, &mut left).unwrap();
            detailed
                .tick_recorded(&active, ActMode::Sample, &mut right)
                .unwrap();
            assert_eq!(left.transitions, right.transitions);
            for i in 0..lanes {
                assert_eq!(basic.lane_state(i), detailed.lane_state(i));
            }
        }
        assert_eq!(
            serde_json::to_value(&basic).unwrap(),
            serde_json::to_value(&detailed).unwrap()
        );
        assert!(right.pending.is_empty());
    }
}

#[test]
fn final_candidate_is_recorded_before_terminal_or_truncated_reset() {
    for truncate in [false, true] {
        let mut runtime = runtime(1, truncate);
        let mut trace = Trace::default();
        for _ in 0..3 {
            runtime
                .tick_recorded(&[true], ActMode::Sample, &mut trace)
                .unwrap();
        }
        assert_eq!(
            trace.candidates[1],
            (
                !truncate,
                truncate,
                Memory {
                    last: 2.,
                    count: 3,
                    reward_sum: 3.
                }
            )
        );
        assert_eq!(
            trace.candidates[2].2,
            Memory {
                last: 1.,
                count: 2,
                reward_sum: 1.
            }
        );
        assert_ne!(
            trace.transitions[1].episode_id,
            trace.transitions[2].episode_id
        );
        if !truncate {
            assert_eq!(
                trace.transitions[1].next_observation.dense_mask(),
                Some([false; 2].as_slice())
            );
        }
    }
}

#[test]
fn failed_rich_recording_cannot_advance_until_abandoned_recovery() {
    let mut runtime = runtime(4, false);
    let mut trace = Trace {
        fail_once: true,
        ..Default::default()
    };
    assert!(
        runtime
            .tick_recorded(&[true; 4], ActMode::Sample, &mut trace)
            .is_err()
    );
    assert!(
        runtime
            .tick_recorded(&[true; 4], ActMode::Sample, &mut trace)
            .is_err()
    );
    runtime.reset_all(&mut trace).unwrap();
    assert_eq!(trace.abandoned, 4);
    runtime
        .tick_recorded(&[true; 4], ActMode::Sample, &mut trace)
        .unwrap();
    assert!(trace.pending.is_empty());
    assert_eq!(trace.transitions.len(), 4);
}
