use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::Action,
        env::{Env, TimeLimit},
    },
    envs::cartpole::CartPole,
};
use serde::Deserialize;
use std::cell::Cell;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    sb3_time_limit_flags: [bool; 2],
}
#[derive(Deserialize)]
struct Case {
    name: String,
    initial_state: [String; 4],
    transitions: Vec<Transition>,
}
#[derive(Deserialize)]
struct Transition {
    action: usize,
    observation: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
    override_state: Option<[String; 4]>,
}

fn state(raw: &[String; 4]) -> [f64; 4] {
    raw.each_ref().map(|x| x.parse().unwrap())
}

#[test]
fn cartpole_v1_matches_gymnasium_and_sb3_boundary_convention() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_cartpole_env.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fixture.sb3_time_limit_flags, [true, false]);
    let mut max_error = 0_f32;
    let mut transitions = 0;
    let (mut saw_limit, mut saw_both, mut saw_failure) = (false, false, false);
    for case in fixture.cases {
        let mut env = TimeLimit::new(CartPole::new(7), 500).unwrap();
        env.reset(Some(7)).unwrap();
        env.inner = CartPole::from_state(7, state(&case.initial_state)).unwrap();
        for (index, expected) in case.transitions.iter().enumerate() {
            if let Some(override_state) = &expected.override_state {
                env.inner = CartPole::from_state(7, state(override_state)).unwrap();
            }
            let actual = env.step(&Action::Discrete(expected.action)).unwrap();
            assert_eq!(
                actual.observation.dense_values().len(),
                expected.observation.len()
            );
            for (a, b) in actual
                .observation
                .dense_values()
                .iter()
                .zip(&expected.observation)
            {
                let error = (a - b).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 2e-6 * (1. + b.abs()),
                    "{} step {index}: {a} != {b}",
                    case.name
                );
            }
            assert_eq!(actual.reward.total(), expected.reward);
            assert_eq!(
                actual.terminated, expected.terminated,
                "{} step {index}",
                case.name
            );
            // SB3 DummyVecEnv reports TimeLimit.truncated only when not terminated.
            assert_eq!(
                actual.truncated,
                expected.truncated && !expected.terminated,
                "{} step {index}",
                case.name
            );
            saw_limit |= expected.truncated && !expected.terminated;
            saw_both |= expected.truncated && expected.terminated;
            saw_failure |= expected.terminated && !expected.truncated;
            transitions += 1;
        }
        if case
            .transitions
            .last()
            .is_some_and(|step| step.terminated || step.truncated)
        {
            assert!(env.step(&Action::Discrete(0)).is_err());
        }
    }
    assert!(saw_limit && saw_both && saw_failure);
    assert!(transitions >= 1000);
    eprintln!(
        "CartPole compared {transitions} transitions; maximum absolute observation error {max_error:e}"
    );
}

#[test]
fn cartpole_time_limit_bootstraps_and_resets_through_collector() {
    use rl::{
        collect::{OnPolicyCollector, vector_env::SyncVectorEnv},
        core::{
            action::ActionSpace,
            actor::{ActMode, Actor, RolloutActor, RolloutBehavior},
            env::{EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step},
        },
        data::rollout::RolloutBuffer,
    };
    struct Controlled {
        env: TimeLimit<CartPole>,
        initial: [f64; 4],
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
            self.env.inner = CartPole::from_state(7, self.initial)?;
            Ok(EpisodeStart::fresh(Observation::new(
                self.initial.iter().map(|x| *x as f32).collect(),
            )))
        }
        fn step(&mut self, action: &Action) -> rl::Result<Step> {
            self.env.step(action)
        }
    }
    struct ActorWithRecordedActions {
        actions: Vec<usize>,
        index: Cell<usize>,
        space: ActionSpace,
    }
    impl Actor for ActorWithRecordedActions {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }
        fn act_into(
            &self,
            obs: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            out: &mut Vec<Action>,
        ) -> rl::Result<()> {
            assert_eq!(obs.len(), 1);
            let index = self.index.get();
            self.index.set(index + 1);
            *out = vec![Action::Discrete(self.actions[index % self.actions.len()])];
            Ok(())
        }
    }
    impl RolloutActor for ActorWithRecordedActions {
        fn sample_rollout_into(
            &self,
            obs: ObservationBatch<'_>,
            rng: &mut ChaCha8Rng,
            out: &mut RolloutBehavior,
        ) -> rl::Result<()> {
            self.act_into(obs, ActMode::Sample, rng, &mut out.actions)?;
            self.values_into(obs, &mut out.values)?;
            out.log_probs = vec![0.];
            out.latents = vec![None];
            Ok(())
        }
        fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> rl::Result<()> {
            *out = obs
                .iter()
                .map(|o| 2. + 1000. * o.dense_values()[2])
                .collect();
            Ok(())
        }
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_cartpole_env.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let balanced = fixture
        .cases
        .iter()
        .find(|c| c.name == "balanced_500")
        .unwrap();
    let actor = ActorWithRecordedActions {
        actions: balanced.transitions.iter().map(|x| x.action).collect(),
        index: Cell::new(0),
        space: ActionSpace::Discrete { count: 2 },
    };
    let env = Controlled {
        env: CartPole::v1(7).unwrap(),
        initial: [0.; 4],
    };
    let mut collector = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 5).unwrap();
    let mut rollout = RolloutBuffer::default();
    let delta = collector
        .collect_steps_into(&actor, 501, &mut rollout)
        .unwrap();
    assert_eq!(delta.episodes, 1);
    let last = rollout.row(499).unwrap();
    assert!(!last.transition.terminated && last.transition.truncated);
    for (actual, expected) in last
        .transition
        .next_observation
        .dense_values()
        .iter()
        .zip(&balanced.transitions[499].observation)
    {
        assert!((actual - expected).abs() <= 2e-6 * (1. + expected.abs()));
    }
    let terminal_value = 2. + 1000. * last.transition.next_observation.dense_values()[2];
    assert!(
        (terminal_value - 2.).abs() > 0.1,
        "final and reset observations must yield distinct values"
    );
    assert_eq!(rollout.next_value(499), Some(terminal_value));
    assert_eq!(
        rollout
            .row(500)
            .unwrap()
            .transition
            .observation
            .dense_values(),
        [0.; 4]
    );
    let batch = rollout.to_batch(0.9, Some(0.), false).unwrap();
    assert!((batch.returns[499] - (1. + 0.9 * terminal_value)).abs() < 1e-6);

    let terminal = Controlled {
        env: TimeLimit::new(CartPole::new(7), 1).unwrap(),
        initial: [2.399, 0.1, 0., 0.],
    };
    let mut collector =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![terminal]).unwrap(), 5).unwrap();
    collector
        .collect_steps_into(&actor, 1, &mut rollout)
        .unwrap();
    let row = rollout.row(0).unwrap();
    assert!(row.transition.terminated && !row.transition.truncated);
    assert_eq!(rollout.next_value(0), Some(0.));
    assert_eq!(
        rollout.to_batch(0.9, Some(0.), false).unwrap().returns[0],
        1.
    );
}
