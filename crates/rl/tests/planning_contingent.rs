use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::planning::{PlanStatus, PlanningBudget, PlanningRequest};
use rl::{
    Result,
    models::predictive::{EnumerableModel, OutcomeDistribution, Transition, WeightedTransition},
    planning::{
        contingent::{self, Config, Options, ReturnBounds},
        guidance::PolicyGuide,
    },
};
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Clone, Copy)]
enum Context {
    Root,
    Reading(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Inspect,
    Guess(bool),
}

const ROOT: [Action; 3] = [Action::Inspect, Action::Guess(false), Action::Guess(true)];

struct Diagnosis;

impl PlanningModel for Diagnosis {
    type Context = Context;
    type Action = Action;
    type Observation = bool;

    fn actions(&self, context: &Context, actions: &mut Vec<Action>) {
        actions.extend_from_slice(match context {
            Context::Root => &ROOT,
            Context::Reading(_) => &ROOT[1..],
        });
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 0.9,
        }
    }
}

impl EnumerableModel for Diagnosis {
    fn enumerate(
        &self,
        context: &Context,
        action: &Action,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<Context, bool>> {
        let (probability, duration, next) = match (context, action) {
            (Context::Root, Action::Inspect) => (
                0.5,
                NonZeroU16::new(2).unwrap(),
                [Some(Context::Reading(true)), Some(Context::Reading(false))],
            ),
            (Context::Root, Action::Guess(_)) => (0.5, NonZeroU16::MIN, [None; 2]),
            (Context::Reading(reading), Action::Guess(guess)) => (
                if reading == guess { 0.9 } else { 0.1 },
                NonZeroU16::MIN,
                [None; 2],
            ),
            (Context::Reading(_), Action::Inspect) => panic!("inspection cannot repeat"),
        };
        let outcomes = [true, false]
            .into_iter()
            .zip([probability, 1. - probability])
            .zip(next)
            .take(cap.get() as usize)
            .map(|((observation, probability), next)| WeightedTransition {
                probability,
                transition: Transition {
                    observation,
                    reward: f64::from(observation && next.is_none()),
                    duration,
                    next,
                },
            })
            .collect();
        Ok(OutcomeDistribution {
            outcomes,
            complete: cap.get() >= 2,
        })
    }
}

struct PreferWrongGuess;

impl PolicyGuide<Context, Action> for PreferWrongGuess {
    fn prior(&self, _: &Context, feasible: &[Action]) -> Result<Vec<f64>> {
        Ok(feasible
            .iter()
            .map(|action| f64::from(*action == Action::Guess(false)))
            .collect())
    }
}

fn config(depth: u16, transitions: u32) -> Config {
    Config {
        depth: NonZeroU16::new(depth).unwrap(),
        max_transitions: NonZeroU32::new(transitions).unwrap(),
        return_bounds: ReturnBounds {
            lower: 0.,
            upper: 1.,
        },
    }
}

fn request(expansions: u32) -> PlanningRequest<'static, Context, Action> {
    PlanningRequest {
        context: &Context::Root,
        feasible: &ROOT,
        budget: PlanningBudget::Expansions(NonZeroU32::new(expansions).unwrap()),
    }
}

#[test]
fn contingent_backup_changes_guesses_after_opposite_readings() -> Result<()> {
    let report = contingent::search(
        &Diagnosis,
        [4; 32],
        request(64),
        config(2, 128),
        Options {
            guide: Some(&PreferWrongGuess),
            leaf_bounds: None,
        },
    )?;
    let action = &report.decision.estimates[0];
    assert_eq!(report.decision.selected, Some(Action::Inspect));
    assert_eq!(report.decision.status, PlanStatus::Exact);
    assert_eq!(action.action, Action::Inspect);
    assert!((action.value - 0.729).abs() < 1e-12);
    assert_eq!(action.upper_bound, Some(action.value));
    assert!(report.guide_evaluations > 0);
    Ok(())
}

#[test]
fn depth_cutoff_keeps_a_nonterminal_probability_interval() -> Result<()> {
    let report = contingent::search(
        &Diagnosis,
        [4; 32],
        request(64),
        config(1, 128),
        Options::default(),
    )?;
    assert_eq!(report.decision.status, PlanStatus::Bounded);
    assert_eq!(report.decision.selected, Some(Action::Guess(false)));
    assert_eq!(report.decision.work.expansions, 1);
    let scan = &report.decision.estimates[0];
    assert_eq!(scan.value, 0.);
    assert!((scan.upper_bound.unwrap() - 0.81).abs() < 1e-12);
    assert_eq!(report.decision.estimates[1].value, 0.5);
    assert_eq!(report.cutoffs, 2);
    Ok(())
}

#[test]
fn incomplete_probability_law_consumes_work_without_changing_bounds() -> Result<()> {
    let report = contingent::search(
        &Diagnosis,
        [4; 32],
        request(1),
        config(2, 1),
        Options::default(),
    )?;
    assert_eq!(report.decision.status, PlanStatus::Bounded);
    assert_eq!(report.decision.work.transitions, 1);
    assert_eq!(report.enumerations, 1);
    assert!(
        report
            .decision
            .estimates
            .iter()
            .all(|estimate| { estimate.value == 0. && estimate.upper_bound == Some(1.) })
    );
    Ok(())
}

#[test]
fn root_support_must_match_the_model_in_order() {
    let mut keys = ROOT;
    keys.swap(1, 2);
    let invalid = PlanningRequest {
        feasible: &keys,
        ..request(4)
    };
    assert!(
        contingent::search(
            &Diagnosis,
            [4; 32],
            invalid,
            config(2, 16),
            Options::default()
        )
        .is_err()
    );
}
