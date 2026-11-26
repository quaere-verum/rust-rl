//! Two-decision public information-state problem shared by controlled tests.
#![allow(dead_code)]
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, PredictiveModel,
        ReturnObjective, Transition, WeightedTransition,
    },
};
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Context {
    Root,
    Signal(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Commit,
    Investigate,
    Guess(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Signal(u16),
    Terminal(bool),
    ExpectedTerminal,
}

pub struct InvestigateOrCommit {
    signals: NonZeroU16,
    expected_terminal_rewards: bool,
}

impl InvestigateOrCommit {
    pub const ROOT_ACTIONS: [Action; 2] = [Action::Commit, Action::Investigate];

    /// Use Bernoulli success with the prescribed action-specific probabilities.
    pub fn new(signals: NonZeroU16) -> Self {
        Self {
            signals,
            expected_terminal_rewards: false,
        }
    }

    /// Keep sampled signals and replace terminal reward noise by its exact
    /// conditional expectation. Matching rollout guesses then return exact leaf values.
    pub fn with_expected_terminal_rewards(mut self) -> Self {
        self.expected_terminal_rewards = true;
        self
    }

    pub fn contexts(&self) -> Vec<Context> {
        std::iter::once(Context::Root)
            .chain((0..self.signals.get()).map(Context::Signal))
            .collect()
    }

    pub fn success_probability(context: Context, action: Action) -> f64 {
        match (context, action) {
            (Context::Root, Action::Commit) => 0.6,
            (Context::Signal(signal), Action::Guess(guess)) if signal == guess => 0.9,
            (Context::Signal(_), Action::Guess(_)) => 0.5,
            _ => panic!("expected a legal terminal action"),
        }
    }

    fn signal(signal: u16) -> Transition<Context, Event> {
        Transition {
            observation: Event::Signal(signal),
            reward: 0.,
            duration: NonZeroU16::MIN,
            next: Some(Context::Signal(signal)),
        }
    }

    fn terminal(observation: Event, reward: f64) -> Transition<Context, Event> {
        Transition {
            observation,
            reward,
            duration: NonZeroU16::MIN,
            next: None,
        }
    }
}

impl PlanningModel for InvestigateOrCommit {
    type Context = Context;
    type Action = Action;
    type Observation = Event;

    fn actions(&self, context: &Context, actions: &mut Vec<Action>) {
        match context {
            Context::Root => actions.extend(Self::ROOT_ACTIONS),
            Context::Signal(_) => actions.extend((0..self.signals.get()).map(Action::Guess)),
        }
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 1.,
        }
    }
}

impl PredictiveModel for InvestigateOrCommit {
    type Hidden = ();

    fn sample_root(&self, _: &Context, _: &mut ChaCha8Rng) -> Result<()> {
        Ok(())
    }

    fn rollout(&self, context: &Context) -> Option<Action> {
        Some(match *context {
            Context::Root => Action::Commit,
            Context::Signal(signal) => Action::Guess(signal),
        })
    }

    fn simulate(
        &self,
        context: &Context,
        _: &mut (),
        action: &Action,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Context, Event>> {
        if *action == Action::Investigate {
            return Ok(Self::signal(rng.random_range(0..self.signals.get())));
        }
        let p = Self::success_probability(*context, *action);
        if self.expected_terminal_rewards {
            return Ok(Self::terminal(Event::ExpectedTerminal, p));
        }
        let success = rng.random_bool(p);
        Ok(Self::terminal(Event::Terminal(success), f64::from(success)))
    }
}

impl EnumerableModel for InvestigateOrCommit {
    fn enumerate(
        &self,
        context: &Context,
        action: &Action,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<Context, Event>> {
        let (outcomes, complete) = if *action == Action::Investigate {
            let total = u32::from(self.signals.get());
            let outcomes = (0..total.min(cap.get()))
                .map(|signal| WeightedTransition {
                    probability: 1. / f64::from(total),
                    transition: Self::signal(signal as u16),
                })
                .collect();
            (outcomes, cap.get() >= total)
        } else {
            let p = Self::success_probability(*context, *action);
            if self.expected_terminal_rewards {
                (
                    vec![WeightedTransition {
                        probability: 1.,
                        transition: Self::terminal(Event::ExpectedTerminal, p),
                    }],
                    true,
                )
            } else {
                let outcomes = [(false, 1. - p), (true, p)]
                    .into_iter()
                    .take(cap.get() as usize)
                    .map(|(success, probability)| WeightedTransition {
                        probability,
                        transition: Self::terminal(Event::Terminal(success), f64::from(success)),
                    })
                    .collect();
                (outcomes, cap.get() >= 2)
            }
        };
        Ok(OutcomeDistribution { outcomes, complete })
    }
}
