use rl::{
    Error, Result,
    algorithms::tabular::{QTable, TabularActor, TdConfig},
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace, FiniteActionKey},
        actor::ActMode,
        checkpoint::{Checkpoint, Snapshot},
        env::{DiscreteObservationSpace, Env, EpisodeStart, Observation, ObservationSpace, Step},
        state_key::{StateKeyBuilder, StateKeyKind},
        stateful::HistoryInput,
        temporal::TemporalSpaces,
    },
    data::temporal::EpisodeStore,
    train::checkpoint::{load_learner, load_run, save_learner, save_run},
};
use serde::{Deserialize, Serialize};

const ACTIONS: ActionSpace = ActionSpace::Discrete { count: 2 };
const OBSERVATIONS: ObservationSpace =
    ObservationSpace::Discrete(DiscreteObservationSpace { symbols: 4 });

#[derive(Clone, Serialize, Deserialize)]
struct DelayedCue {
    cue: u64,
    phase: u64,
}

impl DelayedCue {
    fn observation(&self) -> Observation {
        let (symbol, mask) = match self.phase {
            0 => (self.cue, [true, false]),
            1 => (2, [true, true]),
            _ => (3, [false, false]),
        };
        Observation::discrete(symbol, Some(mask.into()))
    }
}

impl Env for DelayedCue {
    fn observation_space(&self) -> ObservationSpace {
        OBSERVATIONS
    }

    fn action_space(&self) -> &ActionSpace {
        &ACTIONS
    }

    fn reset(&mut self, _seed: Option<u64>) -> Result<EpisodeStart> {
        self.phase = 0;
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.observation().check_action(&ACTIONS, action)?;
        let terminated = self.phase == 1;
        let reward = if terminated {
            if action == &Action::Discrete(self.cue as usize) {
                1.0
            } else {
                -1.0
            }
        } else {
            0.0
        };
        self.phase += 1;
        Ok(Step {
            observation: self.observation(),
            reward: reward.into(),
            terminated,
            truncated: false,
        })
    }
}

impl Snapshot for DelayedCue {
    fn validate_snapshot(&self) -> Result<()> {
        if self.cue < 2 && self.phase <= 2 {
            Ok(())
        } else {
            Err(Error::State("invalid cue environment snapshot".into()))
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct RememberCue;

impl StateKeyBuilder for RememberCue {
    type Memory = Option<(u64, u64)>;
    type Key = (u64, u64);

    fn kind(&self) -> StateKeyKind {
        StateKeyKind::Exact
    }

    fn initial_memory(&self) -> Result<Self::Memory> {
        Ok(None)
    }

    fn observe(&self, previous: &Self::Memory, input: HistoryInput<'_>) -> Result<Self::Memory> {
        let symbol = input
            .observation
            .as_discrete()
            .ok_or_else(|| Error::State("expected a discrete cue".into()))?
            .symbol;
        match (previous, input.previous) {
            (None, None) => Ok(Some((symbol, symbol))),
            (Some((_, cue)), Some(_)) => Ok(Some((symbol, *cue))),
            _ => Err(Error::State("cue history does not begin at reset".into())),
        }
    }

    fn key(&self, memory: &Self::Memory) -> Result<Self::Key> {
        memory.ok_or_else(|| Error::State("cue is unobserved".into()))
    }
}

#[test]
fn restored_policy_remembers_the_public_cue_across_an_aliased_observation() -> Result<()> {
    let mut table = QTable::new(0.0)?;
    let td = TdConfig {
        gamma: 0.5,
        step_size: 1.0,
    };
    for cue in 0..2 {
        for action in 0..2 {
            let mut env = DelayedCue { cue, phase: 0 };
            let start = env.reset(None)?;
            let memory = RememberCue.observe(
                &None,
                HistoryInput {
                    observation: &start.observation,
                    previous: None,
                },
            )?;
            let first = env.step(&Action::Discrete(0))?;
            let memory = RememberCue.observe(
                &memory,
                HistoryInput {
                    observation: &first.observation,
                    previous: Some(rl::core::stateful::PreviousInput::from_transition(
                        &Action::Discrete(0),
                        &first.reward,
                        &start.observation,
                    )),
                },
            )?;
            let step = env.step(&Action::Discrete(action))?;
            table.q_learning(
                &RememberCue.key(&memory)?,
                &FiniteActionKey::Discrete(action),
                &step.reward,
                None,
                td,
            )?;
        }
    }

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("cue.learner");
    save_learner(&path, &table)?;
    let mut restored = QTable::new(0.0)?;
    load_learner(&path, &mut restored)?;
    let actor = cue_actor(restored, 2, 0.1)?;
    let env = SyncVectorEnv::new(vec![
        DelayedCue { cue: 0, phase: 0 },
        DelayedCue { cue: 1, phase: 0 },
    ])?;
    let mut collector = StatefulCollector::new(env, actor, 73)?;
    let mut replay = EpisodeStore::new(16, OBSERVATIONS, ACTIONS)?;
    collector.collect(2, ActMode::Deterministic, &mut replay)?;
    let episodes: Vec<_> = replay.episodes().collect();
    assert_eq!(episodes.len(), 2);
    assert_eq!(episodes[0].observations[1], episodes[1].observations[1]);
    assert_eq!(episodes[0].steps[1].action, Action::Discrete(0));
    assert_eq!(episodes[1].steps[1].action, Action::Discrete(1));
    assert!(episodes.iter().all(|episode| {
        episode.steps[1].terminated
            && episode.steps[1].reward.total() == 1.0
            && episode.steps[1].behavior.log_prob.is_none()
    }));
    let archived = serde_json::to_string(&replay)?;
    let restored_replay: EpisodeStore = serde_json::from_str(&archived)?;
    assert!(restored_replay.episodes().eq(replay.episodes()));
    Ok(())
}

type CueTable = QTable<(u64, u64), FiniteActionKey>;

fn cue_actor(table: CueTable, revision: u64, epsilon: f64) -> Result<TabularActor<RememberCue>> {
    TabularActor::new(
        RememberCue,
        table,
        TemporalSpaces {
            observations: OBSERVATIONS,
            actions: ACTIONS,
        },
        revision,
        epsilon,
    )
}

#[derive(Serialize, Deserialize)]
struct CueRun {
    collector: StatefulCollector<SyncVectorEnv<DelayedCue>, TabularActor<RememberCue>>,
    replay: EpisodeStore,
}

impl Snapshot for CueRun {
    fn validate_snapshot(&self) -> Result<()> {
        self.collector.validate_snapshot()?;
        if self.replay.observation_space() != &self.collector.observation_space()
            || self.replay.action_space() != self.collector.action_space()
        {
            return Err(Error::State(
                "cue replay spaces differ from collector".into(),
            ));
        }
        for lane in 0..self.collector.len() {
            let episode = self.collector.lane_episode(lane).unwrap();
            let expected = if self.collector.lane_begun(lane) == Some(true) {
                self.collector
                    .lane_timestep(lane)
                    .and_then(|timestep| usize::try_from(timestep).ok())
            } else {
                None
            };
            if self.replay.active_timestep(episode) != expected {
                return Err(Error::State(
                    "cue replay and collector clocks differ".into(),
                ));
            }
        }
        Ok(())
    }
}

#[test]
fn run_checkpoint_resumes_sampled_lanes_and_factual_replay_exactly() -> Result<()> {
    let td = TdConfig {
        gamma: 0.5,
        step_size: 1.0,
    };
    let mut table = CueTable::new(0.0)?;
    table.q_learning(
        &(2, 1),
        &FiniteActionKey::Discrete(1),
        &1.0.into(),
        None,
        td,
    )?;
    let env = SyncVectorEnv::new(vec![
        DelayedCue { cue: 0, phase: 0 },
        DelayedCue { cue: 1, phase: 0 },
    ])?;
    let mut live = CueRun {
        collector: StatefulCollector::new(env, cue_actor(table.clone(), 7, 0.4)?, 73)?,
        replay: EpisodeStore::new(128, OBSERVATIONS, ACTIONS)?,
    };
    // Take several sampled decisions before the checkpoint so restoring RNGs
    // to their initial seeds would change the subsequent trajectory.
    live.collector
        .collect(7, ActMode::Sample, &mut live.replay)?;
    let lane = 1;
    let in_flight = live.collector.lane_episode(lane).unwrap();
    assert_eq!(live.replay.active_timestep(in_flight), Some(1));
    table.q_learning(
        &(2, 1),
        &FiniteActionKey::Discrete(0),
        &2.0.into(),
        None,
        td,
    )?;
    live.collector.publish(cue_actor(table.clone(), 8, 0.8)?)?;

    let path = tempfile::tempdir()?;
    let checkpoint = path.path().join("cue.run");
    save_run(&checkpoint, &table, &live)?;
    let mut restored_table = CueTable::new(0.0)?;
    let mut resumed: CueRun = load_run(&checkpoint, &mut restored_table)?;
    assert_eq!(
        restored_table.checkpoint()?.metadata,
        table.checkpoint()?.metadata
    );
    assert_eq!(resumed.collector.lane_state(lane), Some(&Some((2, 1))));
    assert_eq!(resumed.collector.lane_revision(lane), Some(7));
    assert_eq!(resumed.collector.latest_revision(), 8);

    live.collector
        .collect(21, ActMode::Sample, &mut live.replay)?;
    resumed
        .collector
        .collect(21, ActMode::Sample, &mut resumed.replay)?;
    assert_eq!(resumed.collector.lane_revision(lane), Some(8));
    assert!(
        resumed.replay.episode(in_flight).unwrap().steps[1]
            .behavior
            .log_prob
            .is_some()
    );
    assert!(resumed.replay.episodes().eq(live.replay.episodes()));
    assert_eq!(
        serde_json::to_value(&resumed)?,
        serde_json::to_value(&live)?
    );
    Ok(())
}

#[test]
fn actor_snapshot_rejects_table_keys_outside_its_recorded_action_space() -> Result<()> {
    let mut table = CueTable::new(0.0)?;
    let td = TdConfig {
        gamma: 0.5,
        step_size: 1.0,
    };
    table.q_learning(
        &(2, 1),
        &FiniteActionKey::Discrete(1),
        &1.0.into(),
        None,
        td,
    )?;
    let mut snapshot = serde_json::to_value(cue_actor(table, 2, 0.2)?)?;
    snapshot["table"]["metadata"]["rows"][0]["action"] =
        serde_json::to_value(FiniteActionKey::Discrete(2))?;
    assert!(serde_json::from_value::<TabularActor<RememberCue>>(snapshot).is_err());
    Ok(())
}
