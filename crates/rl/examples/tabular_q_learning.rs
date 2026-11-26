//! Train Q-learning from environment steps with legal actions and primitive rewards.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::tabular::{QTable, TdConfig},
    core::{
        action::{Action, ActionSpace, FiniteActionKey},
        env::{DiscreteObservationSpace, Env, EpisodeStart, Observation, ObservationSpace, Step},
        reward::Reward,
    },
};

const ACTIONS: ActionSpace = ActionSpace::Discrete { count: 2 };

#[derive(Clone, Copy)]
enum Position {
    Start,
    Passage,
    End,
}

impl Position {
    fn observe(self) -> Observation {
        let mask = match self {
            Self::Start => [true, true],
            Self::Passage => [false, true],
            Self::End => [false, false],
        };
        Observation::discrete(self as u64, Some(mask.into()))
    }
}

impl Env for Position {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Discrete(DiscreteObservationSpace { symbols: 3 })
    }

    fn action_space(&self) -> &ActionSpace {
        &ACTIONS
    }

    fn reset(&mut self, _seed: Option<u64>) -> Result<EpisodeStart> {
        *self = Self::Start;
        Ok(EpisodeStart::fresh(self.observe()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.observe().check_action(self.action_space(), action)?;
        let (next, reward) = match (*self, action) {
            (Self::Start, Action::Discrete(0)) => (Self::End, 1.0.into()),
            (Self::Start, Action::Discrete(1)) => (Self::Passage, 0.0.into()),
            (Self::Passage, Action::Discrete(1)) => {
                (Self::End, Reward::from_primitives(vec![2.0, 4.0])?)
            }
            _ => unreachable!("the public action mask was checked"),
        };
        *self = next;
        Ok(Step {
            observation: next.observe(),
            reward,
            terminated: matches!(next, Self::End),
            truncated: false,
        })
    }
}

fn state(observation: &Observation) -> u64 {
    observation.as_discrete().expect("corridor symbol").symbol
}

fn keys(observation: &Observation, actions: &[Action]) -> Result<Vec<FiniteActionKey>> {
    actions
        .iter()
        .map(|action| observation.finite_action_key(action))
        .collect()
}

fn main() -> Result<()> {
    let mut env = Position::Start;
    let mut table = QTable::new(0.0)?;
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let update = TdConfig {
        gamma: 0.5,
        step_size: 1.0,
    };

    for _ in 0..128 {
        let mut observation = env.reset(None)?.observation;
        loop {
            let actions = observation.finite_actions(env.action_space())?;
            let action_keys = keys(&observation, &actions)?;
            let choice = table.epsilon_greedy(&state(&observation), &action_keys, 0.2, &mut rng)?;
            let step = env.step(&actions[choice.index])?;
            let next = if step.terminated {
                None
            } else {
                let actions = step.observation.finite_actions(env.action_space())?;
                Some((state(&step.observation), keys(&step.observation, &actions)?))
            };
            table.q_learning(
                &state(&observation),
                &action_keys[choice.index],
                &step.reward,
                next.as_ref()
                    .map(|(state, actions)| (state, actions.as_slice())),
                update,
            )?;
            if step.ended() {
                break;
            }
            observation = step.observation;
        }
    }

    let start = Position::Start as u64;
    let stop = table.value(&start, &FiniteActionKey::Discrete(0));
    let continue_on = table.value(&start, &FiniteActionKey::Discrete(1));
    assert_eq!((stop, continue_on), (1.0, 2.0));
    println!("Start action values are {stop} for stopping and {continue_on} for continuing.");
    Ok(())
}
