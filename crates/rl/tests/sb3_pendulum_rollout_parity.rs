#![cfg(feature = "neural")]

use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{OnPolicyCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, RolloutActor, RolloutBehavior},
        env::{
            ClipAction, Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step,
            TimeLimit,
        },
    },
    data::rollout::RolloutBuffer,
    envs::pendulum::Pendulum,
    nn::{distributions::Distribution, matrix, to_vec},
};
use serde::Deserialize;
use std::cell::Cell;

#[derive(Deserialize)]
struct Fixture {
    initial_state: [f64; 2],
    actor_weight: [f32; 3],
    actor_bias: f32,
    value_weight: [f32; 3],
    value_bias: f32,
    log_std: f32,
    gamma: f32,
    gae_lambda: f32,
    events: Vec<Event>,
}
#[derive(Deserialize)]
struct Event {
    observation: Vec<f32>,
    next_observation: Vec<f32>,
    raw_action: f32,
    physical_action: f32,
    reward: f32,
    buffer_reward: f32,
    terminated: bool,
    truncated: bool,
    log_prob: f32,
    advantage: f32,
    target: f32,
    value: f32,
}
struct Controlled {
    env: TimeLimit<Pendulum>,
    initial: [f64; 2],
}
impl Env for Controlled {
    fn observation_space(&self) -> ObservationSpace {
        self.env.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.env.action_space()
    }
    fn reset(&mut self, seed: Option<u64>) -> rl::Result<EpisodeStart> {
        self.env.reset(seed)?;
        self.env.inner = Pendulum::from_state(7, self.initial)?;
        Ok(EpisodeStart::fresh(Observation::new(vec![
            self.initial[0].cos() as f32,
            self.initial[0].sin() as f32,
            self.initial[1] as f32,
        ])))
    }
    fn step(&mut self, action: &Action) -> rl::Result<Step> {
        self.env.step(action)
    }
}
struct ReferenceActor<'a> {
    fixture: &'a Fixture,
    index: Cell<usize>,
    space: ActionSpace,
}
impl Actor for ReferenceActor<'_> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        _: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        _: &mut Vec<Action>,
    ) -> rl::Result<()> {
        unreachable!("fixture replays SB3's policy samples")
    }
}
impl RolloutActor for ReferenceActor<'_> {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> rl::Result<()> {
        assert_eq!(obs.len(), 1);
        let i = self.index.get();
        self.index.set(i + 1);
        let observation = obs.get(0).unwrap().dense_values();
        let mean = self
            .fixture
            .actor_weight
            .iter()
            .zip(observation)
            .map(|(a, b)| a * b)
            .sum::<f32>()
            + self.fixture.actor_bias;
        let distribution = Distribution::Gaussian {
            mean: matrix(vec![mean], 1, 1),
            log_std: matrix(vec![self.fixture.log_std], 1, 1),
            bounds: None,
        };
        out.actions = vec![Action::Continuous(vec![self.fixture.events[i].raw_action])];
        out.latents = vec![None];
        out.log_probs = to_vec(distribution.log_prob(&out.actions, &out.latents)?)?;
        self.values_into(obs, &mut out.values)
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> rl::Result<()> {
        *out = obs
            .iter()
            .map(|o| {
                self.fixture
                    .value_weight
                    .iter()
                    .zip(o.dense_values())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + self.fixture.value_bias
            })
            .collect();
        Ok(())
    }
}

#[test]
fn sb3_ppo_pendulum_rollout_matches_collection_and_gae() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_pendulum_rollout.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fixture.events.len(), 204);
    let actor = ReferenceActor {
        fixture: &fixture,
        index: Cell::new(0),
        space: ActionSpace::Unbounded { dimensions: 1 },
    };
    let env = Controlled {
        env: Pendulum::v1(7).unwrap(),
        initial: fixture.initial_state,
    };
    let mut collector = OnPolicyCollector::new(
        SyncVectorEnv::new(vec![ClipAction::new(env).unwrap()]).unwrap(),
        1,
    )
    .unwrap();
    let mut rollout = RolloutBuffer::default();
    let delta = collector
        .collect_steps_into(&actor, fixture.events.len(), &mut rollout)
        .unwrap();
    assert_eq!(delta.episodes, 1);
    assert!(
        fixture
            .events
            .iter()
            .any(|e| e.raw_action != e.physical_action)
    );
    let mut max_transition_error = 0_f32;
    for (i, event) in fixture.events.iter().enumerate() {
        let row = rollout.row(i).unwrap();
        assert_eq!(
            row.transition.action,
            Action::Continuous(vec![event.raw_action])
        );
        assert_eq!(event.raw_action.clamp(-2., 2.), event.physical_action);
        assert_eq!(
            (row.transition.terminated, row.transition.truncated),
            (event.terminated, event.truncated)
        );
        for (&a, &b) in row
            .transition
            .observation
            .dense_values()
            .iter()
            .zip(&event.observation)
            .chain(
                row.transition
                    .next_observation
                    .dense_values()
                    .iter()
                    .zip(&event.next_observation),
            )
            .chain(std::iter::once((
                &row.transition.reward.total(),
                &event.reward,
            )))
        {
            let error = (a - b).abs();
            max_transition_error = max_transition_error.max(error);
            assert!(error <= 2e-6 * (1. + b.abs()), "transition {i}: {a} != {b}");
        }
        if event.truncated {
            let bootstrap = rollout.next_value(i).unwrap();
            assert!(bootstrap.abs() > 0.1);
            assert!(
                (event.buffer_reward - (event.reward + fixture.gamma * bootstrap)).abs() < 5e-6
            );
        } else {
            assert_eq!(event.buffer_reward, event.reward);
        }
    }
    assert!(fixture.events[199].truncated);
    let batch = rollout
        .to_batch(fixture.gamma, Some(fixture.gae_lambda), false)
        .unwrap();
    let metadata = batch.bootstrapped().unwrap();
    let mut max_errors = [0_f32; 4];
    for (i, event) in fixture.events.iter().enumerate() {
        for (j, (actual, expected)) in [
            (metadata.log_probs[i], event.log_prob),
            (metadata.values[i], event.value),
            (metadata.advantages[i], event.advantage),
            (batch.returns[i], event.target),
        ]
        .into_iter()
        .enumerate()
        {
            let error = (actual - expected).abs();
            max_errors[j] = max_errors[j].max(error);
            let tolerance = if j < 2 {
                3e-6 * (1. + expected.abs())
            } else {
                5e-5 + 1e-6 * expected.abs()
            };
            assert!(
                error <= tolerance,
                "field {j} row {i}: {actual} != {expected}"
            );
        }
    }
    eprintln!(
        "Pendulum SB3 rollout max abs errors: transition={max_transition_error:e}, logp={:e}, value={:e}, advantage={:e}, return={:e}",
        max_errors[0], max_errors[1], max_errors[2], max_errors[3]
    );
}
