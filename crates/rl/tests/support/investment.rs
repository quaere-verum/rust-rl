use std::num::{NonZeroU16, NonZeroU32};

use rl::{
    Result,
    core::reward::PotentialValues,
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, ReturnObjective,
        Transition, WeightedTransition,
    },
    planning::finite::FiniteModel,
};

pub const HORIZON: NonZeroU16 = NonZeroU16::new(5).unwrap();
pub const MAX_WORKERS: u8 = 4;
pub const INITIAL: State = State {
    remaining: HORIZON.get(),
    workers: 1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct State {
    pub remaining: u16,
    pub workers: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Action {
    Research,
    Invest,
}

impl State {
    pub fn actions(self) -> &'static [Action] {
        if self.remaining == 0 {
            &[]
        } else if self.workers == MAX_WORKERS {
            &[Action::Research]
        } else {
            &[Action::Research, Action::Invest]
        }
    }

    /// Current metal production is either consumed by research or spent on an
    /// additional worker who starts producing on the next tick.
    pub fn step(self, action: Action) -> (f32, Self) {
        let invest = action == Action::Invest;
        let reward = if invest { 0. } else { f32::from(self.workers) };
        (
            reward,
            Self {
                remaining: self.remaining - 1,
                workers: self.workers + u8::from(invest),
            },
        )
    }

    pub fn potential(self) -> f64 {
        f64::from(self.remaining) * f64::from(self.workers)
    }
}

#[derive(Clone, Copy)]
pub enum Objective {
    Natural,
    Shaped,
    IncomeBonus,
}

impl Objective {
    pub fn prepare(self) -> Result<FiniteModel<State, Action>> {
        let contexts = (1..=HORIZON.get())
            .flat_map(|remaining| {
                (1..=MAX_WORKERS).map(move |workers| State { remaining, workers })
            })
            .collect::<Vec<_>>();
        FiniteModel::for_horizon(&self, &contexts, NonZeroU32::MIN)
    }
}

impl PlanningModel for Objective {
    type Context = State;
    type Action = Action;
    type Observation = State;

    fn actions(&self, state: &State, actions: &mut Vec<Action>) {
        actions.extend_from_slice(state.actions());
    }

    fn planning_returns(&self) -> PlanningReturns {
        let objective = match self {
            Self::Natural => ReturnObjective::Physical,
            Self::Shaped => ReturnObjective::Surrogate([1; 32]),
            Self::IncomeBonus => ReturnObjective::Surrogate([2; 32]),
        };
        PlanningReturns {
            objective,
            gamma: 1.,
        }
    }
}

impl EnumerableModel for Objective {
    fn enumerate(
        &self,
        state: &State,
        action: &Action,
        _: NonZeroU32,
    ) -> Result<OutcomeDistribution<State, State>> {
        let (reward, observation) = state.step(*action);
        let reward = f64::from(reward);
        let next = (observation.remaining > 0).then_some(observation);
        let reward = match self {
            Self::Natural => reward,
            Self::Shaped => PotentialValues {
                current: state.potential(),
                next: next.map(State::potential),
            }
            .shape(reward, 1.)?,
            Self::IncomeBonus => reward + f64::from(state.workers),
        };
        Ok(OutcomeDistribution {
            outcomes: vec![WeightedTransition {
                probability: 1.,
                transition: Transition {
                    observation,
                    reward,
                    duration: NonZeroU16::MIN,
                    next,
                },
            }],
            complete: true,
        })
    }
}
