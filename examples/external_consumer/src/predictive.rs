use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::action::ConditionalAction,
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, PredictiveModel,
        ReturnObjective, Transition, WeightedTransition,
    },
    planning::{PlanningBudget, finite::FiniteModel, pomcp},
};
use std::num::{NonZeroU16, NonZeroU32};

use crate::hall::{ARRIVAL_CHANCE, GAMMA, PASSAGE, Side, State, target, use_door};

struct HallModel;

impl HallModel {
    fn outcome(
        context: State,
        action: ConditionalAction,
        arrived: bool,
    ) -> Result<Transition<State, State>> {
        let (successor, reward) = context.outcome(action, arrived)?;
        Ok(Transition {
            observation: successor,
            reward: reward.discounted_f64(GAMMA)?.0,
            duration: NonZeroU16::new(reward.duration() as u16).expect("one or two ticks"),
            next: (successor != State::Done).then_some(successor),
        })
    }
}

impl PlanningModel for HallModel {
    type Context = State;
    type Action = ConditionalAction;
    type Observation = State;

    fn actions(&self, context: &State, actions: &mut Vec<ConditionalAction>) {
        actions.extend(context.menu().iter().copied().map(use_door));
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: GAMMA,
        }
    }
}

impl EnumerableModel for HallModel {
    fn enumerate(
        &self,
        context: &State,
        action: &ConditionalAction,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<State, State>> {
        let law: &[(bool, f64)] = if *action == use_door(PASSAGE) {
            &[(true, ARRIVAL_CHANCE), (false, 1. - ARRIVAL_CHANCE)]
        } else {
            &[(true, 1.)]
        };
        let outcomes = law
            .iter()
            .take(cap.get() as usize)
            .map(|&(arrived, probability)| {
                Ok(WeightedTransition {
                    probability,
                    transition: Self::outcome(*context, *action, arrived)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(OutcomeDistribution {
            outcomes,
            complete: law.len() <= cap.get() as usize,
        })
    }
}

impl PredictiveModel for HallModel {
    type Hidden = ();

    fn sample_root(&self, _context: &State, _rng: &mut ChaCha8Rng) -> Result<()> {
        Ok(())
    }

    fn rollout(&self, context: &State) -> Option<ConditionalAction> {
        context.menu().first().copied().map(use_door)
    }

    fn simulate(
        &self,
        context: &State,
        _hidden: &mut (),
        action: &ConditionalAction,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<State, State>> {
        let arrived = *action == use_door(PASSAGE) && rng.random_bool(ARRIVAL_CHANCE);
        Self::outcome(*context, *action, arrived)
    }
}

pub fn plan(root: State) -> Result<(u64, f64, u64)> {
    let model = HallModel;
    let contexts = [
        State::Entrance(Side::Left),
        State::Entrance(Side::Right),
        State::Fork(Side::Left),
        State::Fork(Side::Right),
    ];
    let solution = FiniteModel::from_model(&model, &contexts, NonZeroU32::new(2).unwrap())?
        .value_iteration(NonZeroU32::new(16).unwrap(), 1e-10)?;
    let best = solution
        .states
        .iter()
        .find(|state| state.context == root)
        .ok_or_else(|| Error::State("root is absent from the public model".into()))?;
    let mut menu = Vec::new();
    model.actions(&root, &mut menu);
    let search = pomcp::search(
        &model,
        &root,
        &menu,
        pomcp::Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: GAMMA,
            exploration: 8.,
        },
        PlanningBudget::Simulations {
            trajectories: NonZeroU32::new(512).unwrap(),
            transitions: NonZeroU32::new(1024).unwrap(),
        },
        &mut ChaCha8Rng::seed_from_u64(18),
    )?;
    let sampled = search
        .action
        .ok_or_else(|| Error::State("sampled planner found no action".into()))?;
    Ok((target(*best.greedy_action())?, best.value, target(sampled)?))
}
