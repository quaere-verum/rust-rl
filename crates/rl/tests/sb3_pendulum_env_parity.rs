use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::Action,
        env::{ClipAction, Env, TimeLimit},
    },
    envs::pendulum::Pendulum,
};
use serde::Deserialize;
use std::cell::Cell;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    sb3_truncation: bool,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    initial_state: [String; 2],
    initial_observation: Vec<f32>,
    transitions: Vec<Transition>,
}
#[derive(Deserialize)]
struct Transition {
    action: f32,
    physical_action: f32,
    observation: Vec<f32>,
    reward: f32,
    terminated: bool,
    truncated: bool,
}
fn parse_state(raw: &[String; 2]) -> [f64; 2] {
    raw.each_ref().map(|x| x.parse().unwrap())
}

#[test]
fn pendulum_v1_matches_gymnasium_through_clipping_adapter() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_pendulum_env.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert!(fixture.sb3_truncation);
    let mut max_observation_error = 0_f32;
    let mut max_reward_error = 0_f32;
    let (mut transitions, mut clipped, mut limits) = (0, 0, 0);
    for case in fixture.cases {
        let state = parse_state(&case.initial_state);
        let mut timed = TimeLimit::new(Pendulum::new(7), 200).unwrap();
        timed.reset(Some(7)).unwrap();
        timed.inner = Pendulum::from_state(7, state).unwrap();
        let mut env = ClipAction::new(timed).unwrap();
        let initial = [
            state[0].cos() as f32,
            state[0].sin() as f32,
            state[1] as f32,
        ];
        for (&a, &b) in initial.iter().zip(&case.initial_observation) {
            assert!((a - b).abs() <= 2e-6 * (1. + b.abs()));
        }
        for (index, expected) in case.transitions.iter().enumerate() {
            let raw = Action::Continuous(vec![expected.action]);
            assert_eq!(
                env.clipped_action(&raw).unwrap(),
                Action::Continuous(vec![expected.physical_action])
            );
            clipped += usize::from(expected.action != expected.physical_action);
            let actual = env.step(&raw).unwrap();
            assert_eq!(
                actual.observation.dense_values().len(),
                expected.observation.len()
            );
            for (&a, &b) in actual
                .observation
                .dense_values()
                .iter()
                .zip(&expected.observation)
            {
                let error = (a - b).abs();
                max_observation_error = max_observation_error.max(error);
                assert!(
                    error <= 2e-6 * (1. + b.abs()),
                    "{} step {index}: {a} != {b}",
                    case.name
                );
            }
            let error = (actual.reward.total() - expected.reward).abs();
            max_reward_error = max_reward_error.max(error);
            assert!(
                error <= 2e-6 * (1. + expected.reward.abs()),
                "{} step {index}: reward {} != {}",
                case.name,
                actual.reward.total(),
                expected.reward
            );
            assert_eq!(
                (actual.terminated, actual.truncated),
                (expected.terminated, expected.truncated)
            );
            transitions += 1;
            limits += usize::from(expected.truncated);
        }
        if case.transitions.last().is_some_and(|step| step.truncated) {
            assert!(env.step(&Action::Continuous(vec![0.])).is_err());
        }
    }
    assert!(transitions >= 800 && clipped >= 100 && limits >= 4);
    eprintln!(
        "Pendulum compared {transitions} transitions ({clipped} clipped actions); maximum f32 observation error {max_observation_error:e}, reward error {max_reward_error:e}"
    );
}

#[test]
fn collector_preserves_raw_torque_and_bootstraps_final_pendulum_observation() {
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
    struct RecordedActor {
        actions: Vec<f32>,
        index: Cell<usize>,
        space: ActionSpace,
    }
    impl Actor for RecordedActor {
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
            *out = vec![Action::Continuous(vec![
                self.actions[index % self.actions.len()],
            ])];
            Ok(())
        }
    }
    impl RolloutActor for RecordedActor {
        fn sample_rollout_into(
            &self,
            obs: ObservationBatch<'_>,
            rng: &mut ChaCha8Rng,
            out: &mut RolloutBehavior,
        ) -> rl::Result<()> {
            self.act_into(obs, ActMode::Sample, rng, &mut out.actions)?;
            self.values_into(obs, &mut out.values)?;
            let Action::Continuous(raw) = &out.actions[0] else {
                unreachable!()
            };
            out.log_probs = vec![raw[0]];
            out.latents = vec![None];
            Ok(())
        }
        fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> rl::Result<()> {
            *out = obs.iter().map(|o| 3. + 0.1 * o.dense_values()[2]).collect();
            Ok(())
        }
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sb3_pendulum_env.json"
    );
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let case = fixture.cases.iter().find(|c| c.name == "random_0").unwrap();
    let actor = RecordedActor {
        actions: case.transitions.iter().map(|x| x.action).collect(),
        index: Cell::new(0),
        space: ActionSpace::Unbounded { dimensions: 1 },
    };
    let initial = parse_state(&case.initial_state);
    let env = Controlled {
        env: Pendulum::v1(7).unwrap(),
        initial,
    };
    let mut collector = OnPolicyCollector::new(
        SyncVectorEnv::new(vec![ClipAction::new(env).unwrap()]).unwrap(),
        5,
    )
    .unwrap();
    let mut rollout = RolloutBuffer::default();
    let delta = collector
        .collect_steps_into(&actor, 201, &mut rollout)
        .unwrap();
    assert_eq!(delta.episodes, 1);
    for (i, expected) in case.transitions.iter().enumerate() {
        assert_eq!(
            rollout.row(i).unwrap().transition.action,
            Action::Continuous(vec![expected.action])
        );
    }
    assert!(
        case.transitions
            .iter()
            .any(|e| e.action != e.physical_action)
    );
    let last = rollout.row(199).unwrap();
    assert!(!last.transition.terminated && last.transition.truncated);
    for (&actual, &expected) in last
        .transition
        .next_observation
        .dense_values()
        .iter()
        .zip(&case.transitions[199].observation)
    {
        assert!((actual - expected).abs() <= 2e-6 * (1. + expected.abs()));
    }
    let terminal_value = 3. + 0.1 * last.transition.next_observation.dense_values()[2];
    let reset_value = 3. + 0.1 * case.initial_observation[2];
    assert!((terminal_value - reset_value).abs() > 0.01);
    assert_eq!(rollout.next_value(199), Some(terminal_value));
    for (&actual, &expected) in rollout
        .row(200)
        .unwrap()
        .transition
        .observation
        .dense_values()
        .iter()
        .zip(&case.initial_observation)
    {
        assert!((actual - expected).abs() <= 2e-6 * (1. + expected.abs()));
    }
    let reward = last.transition.reward.total();
    let batch = rollout.to_batch(0.9, Some(0.), false).unwrap();
    for (i, expected) in case.transitions.iter().enumerate() {
        assert_eq!(batch.bootstrapped().unwrap().log_probs[i], expected.action);
    }
    let target = batch.returns[199];
    assert!((target - (reward + 0.9 * terminal_value)).abs() < 2e-6);
}
