//! Exact public belief-state model for the two-stage relay.

use super::{INSPECT, RelayKey, route_key};
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::{
    Error, Result,
    core::action::ConditionalAction,
    models::predictive::{EnumerableModel, OutcomeDistribution, Transition, WeightedTransition},
    planning::finite::{FiniteModel, Solution},
};
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Hint(bool),
    Delivered(bool),
}

struct RelayModel;

type Row = WeightedTransition<RelayKey, Outcome>;

impl RelayModel {
    fn law(context: RelayKey, action: ConditionalAction) -> Result<[Row; 2]> {
        let outcome = |probability, observation, reward, next| WeightedTransition {
            probability,
            transition: Transition {
                observation,
                reward,
                duration: NonZeroU16::MIN,
                next,
            },
        };
        match context {
            // A symmetric prior makes the two hints equally likely.
            RelayKey::Inspect if action == INSPECT => Ok([false, true].map(|hint_right| {
                outcome(
                    0.5,
                    Outcome::Hint(hint_right),
                    0.0,
                    Some(RelayKey::Route { hint_right }),
                )
            })),
            RelayKey::Route { hint_right }
                if action == route_key(false) || action == route_key(true) =>
            {
                // Bayes' rule gives 3/4 posterior mass to the hinted port.
                let success = if action == route_key(hint_right) {
                    0.75
                } else {
                    0.25
                };
                Ok([
                    outcome(success, Outcome::Delivered(true), 1.0, None),
                    outcome(1.0 - success, Outcome::Delivered(false), 0.0, None),
                ])
            }
            _ => Err(Error::Action(
                "illegal action in the public relay model".into(),
            )),
        }
    }
}

impl PlanningModel for RelayModel {
    type Context = RelayKey;
    type Action = ConditionalAction;
    type Observation = Outcome;

    fn actions(&self, context: &RelayKey, actions: &mut Vec<ConditionalAction>) {
        match context {
            RelayKey::Inspect => actions.push(INSPECT),
            RelayKey::Route { .. } => actions.extend([route_key(false), route_key(true)]),
            RelayKey::End => {}
        }
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 1.0,
        }
    }
}

impl EnumerableModel for RelayModel {
    fn enumerate(
        &self,
        context: &RelayKey,
        action: &ConditionalAction,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<RelayKey, Outcome>> {
        let rows = Self::law(*context, *action)?;
        Ok(OutcomeDistribution {
            outcomes: rows.into_iter().take(cap.get() as usize).collect(),
            complete: cap.get() >= 2,
        })
    }
}

pub fn exact_plan() -> Result<Solution<RelayKey, ConditionalAction>> {
    let contexts = [
        RelayKey::Inspect,
        RelayKey::Route { hint_right: false },
        RelayKey::Route { hint_right: true },
    ];
    FiniteModel::from_model(&RelayModel, &contexts, NonZeroU32::new(2).unwrap())?
        .value_iteration(NonZeroU32::MIN, 1e-12)
}
