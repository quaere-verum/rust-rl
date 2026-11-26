use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    collect::{OnPolicyCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, EpisodeActor, EpisodeBehavior, RolloutActor, RolloutBehavior},
        env::{
            Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step, TimeLimit,
        },
        reward::Reward,
        transition::Transition,
    },
    data::replay::{NStepAccumulator, ReplayBuffer, ReplayItem},
};

#[derive(Clone)]
struct Counter {
    time: usize,
    horizon: usize,
    space: ActionSpace,
}
impl Counter {
    fn new(horizon: usize) -> Self {
        Self {
            time: 0,
            horizon,
            space: ActionSpace::Discrete { count: 1 },
        }
    }
}
impl Env for Counter {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, _: &Action) -> Result<Step> {
        self.time += 1;
        Ok(Step {
            observation: Observation::new(vec![self.time as f32]),
            reward: 1.0_f32.into(),
            terminated: self.time == self.horizon,
            truncated: false,
        })
    }
}
struct Fixed(ActionSpace);
impl Actor for Fixed {
    fn action_space(&self) -> &ActionSpace {
        &self.0
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        out.resize(obs.len(), Action::Discrete(0));
        Ok(())
    }
}
impl RolloutActor for Fixed {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        b: &mut RolloutBehavior,
    ) -> Result<()> {
        b.actions.resize(obs.len(), Action::Discrete(0));
        b.log_probs.resize(obs.len(), 0.);
        b.values.resize(obs.len(), 0.);
        b.latents.resize(obs.len(), None);
        Ok(())
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        *out = obs.iter().map(|o| o.dense_values()[0]).collect();
        Ok(())
    }
}
impl EpisodeActor for Fixed {
    fn sample_episode_into(
        &self,
        obs: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        b: &mut EpisodeBehavior,
    ) -> Result<()> {
        b.actions.resize(obs.len(), Action::Discrete(0));
        b.latents.resize(obs.len(), None);
        b.baseline_values = None;
        Ok(())
    }
}
use rl::data::rollout::RolloutBuffer;
fn transition(r: f32, terminal: bool, truncated: bool) -> Transition {
    Transition {
        observation: Observation::new(vec![r]),
        action: Action::Discrete(0),
        reward: r.into(),
        next_observation: Observation::new(vec![r + 1.]),
        terminated: terminal,
        truncated,
    }
}

#[test]
fn mixed_lanes_reset_without_replacing_final_observations() {
    let env = SyncVectorEnv::new(vec![Counter::new(1), Counter::new(3)]).unwrap();
    let mut collector = OnPolicyCollector::new(env, 4).unwrap();
    let mut roll = RolloutBuffer::default();
    let delta = collector
        .collect_steps_into(&Fixed(ActionSpace::Discrete { count: 1 }), 4, &mut roll)
        .unwrap();
    assert_eq!(
        roll.row(0)
            .unwrap()
            .transition
            .next_observation
            .dense_values(),
        vec![1.]
    );
    assert_eq!(
        roll.row(2).unwrap().transition.observation.dense_values(),
        vec![0.]
    );
    assert_eq!(
        roll.row(5)
            .unwrap()
            .transition
            .next_observation
            .dense_values(),
        vec![3.]
    );
    assert_eq!(
        roll.row(7).unwrap().transition.observation.dense_values(),
        vec![0.]
    );
    assert_eq!(delta.env_steps, 8);
    assert_eq!(delta.episodes, 5);
}

#[test]
fn complete_episode_collection_does_not_pad_with_fake_transitions() {
    let env = SyncVectorEnv::new(vec![Counter::new(1), Counter::new(3)]).unwrap();
    let mut collector = OnPolicyCollector::new(env, 4).unwrap();
    let mut roll = RolloutBuffer::default();
    collector
        .collect_episodes_into(&Fixed(ActionSpace::Discrete { count: 1 }), 5, &mut roll)
        .unwrap();
    assert_eq!(roll.valid_len(), 4);
    assert_eq!(
        roll.to_batch(1., None, false).unwrap().returns,
        vec![1., 3., 2., 1.]
    );
}

#[test]
fn monte_carlo_rejects_external_truncation_and_collection_guard() {
    let env = SyncVectorEnv::new(vec![TimeLimit::new(Counter::new(10), 2).unwrap()]).unwrap();
    let mut c = OnPolicyCollector::new(env, 1).unwrap();
    assert!(
        c.collect_episodes_into(
            &Fixed(ActionSpace::Discrete { count: 1 }),
            5,
            &mut RolloutBuffer::default()
        )
        .is_err()
    );
    let mut c =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![Counter::new(10)]).unwrap(), 1).unwrap();
    assert!(
        c.collect_episodes_into(
            &Fixed(ActionSpace::Discrete { count: 1 }),
            2,
            &mut RolloutBuffer::default()
        )
        .is_err()
    );
}

#[test]
fn nstep_stops_at_truncation_and_preserves_bootstrap_discount() {
    let mut acc = NStepAccumulator::new(3, 0.5, 1).unwrap();
    assert!(
        acc.push(0, transition(2., false, false))
            .unwrap()
            .is_empty()
    );
    let items = acc.push(0, transition(4., false, true)).unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].transition.reward.total(), 4.);
    assert_eq!(items[0].discount, 0.25);
    assert_eq!(
        items[0].transition.next_observation.dense_values(),
        vec![5.]
    );
    assert_eq!(
        acc.push(0, transition(9., true, false)).unwrap()[0].discount,
        0.
    );
}

#[test]
fn wrapped_nstep_windows_preserve_primitive_order_and_bootstrap() {
    let mut accumulator = NStepAccumulator::new(3, 0.5, 1).unwrap();
    let rewards = [
        Reward::from_primitives(vec![2., 4.]).unwrap(),
        (-3.).into(),
        5.0.into(),
        Reward::from_primitives(vec![8., -4.]).unwrap(),
        (-2.).into(),
    ];
    let mut items = Vec::new();
    for (step, reward) in rewards.into_iter().enumerate() {
        items.extend(
            accumulator
                .push(
                    0,
                    Transition {
                        observation: Observation::new(vec![step as f32]),
                        action: Action::Discrete(step),
                        reward,
                        next_observation: Observation::new(vec![(step + 1) as f32]),
                        terminated: false,
                        truncated: step == 4,
                    },
                )
                .unwrap(),
        );
    }
    let expected = [
        (3.875, 0.0625, 3.),
        (1., 0.0625, 4.),
        (7.75, 0.0625, 5.),
        (5.5, 0.125, 5.),
        (-2., 0.5, 5.),
    ];
    assert_eq!(items.len(), expected.len());
    for (step, (item, (reward, discount, successor))) in items.iter().zip(expected).enumerate() {
        assert_eq!(item.transition.action, Action::Discrete(step));
        assert_eq!(item.transition.observation.dense_values(), [step as f32]);
        assert_eq!(item.transition.reward.total(), reward);
        assert_eq!(item.discount, discount);
        assert_eq!(item.transition.next_observation.dense_values(), [successor]);
    }
}

#[test]
fn replay_wraparound_and_rng_serialization_preserve_sampling() {
    let mut replay = ReplayBuffer::new(2, 1, ActionSpace::Discrete { count: 1 }).unwrap();
    for reward in [1., 2., 3.] {
        replay
            .push(&ReplayItem {
                transition: transition(reward, true, false),
                discount: 0.,
            })
            .unwrap();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let mut restored: ChaCha8Rng =
        serde_json::from_str(&serde_json::to_string(&rng).unwrap()).unwrap();
    assert_eq!(rng.random::<u64>(), restored.random::<u64>());
    let batch = replay.sample(64, &mut rng).unwrap();
    assert!(
        batch
            .items
            .iter()
            .all(|x| [2., 3.].contains(&x.transition.reward.total()))
    );
    assert_eq!(replay.len(), 2);
}
