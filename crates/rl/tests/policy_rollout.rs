use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::planning::{Deliberation, PlanStatus, PlanningBudget, PlanningRequest};
use rl::{
    Result,
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PredictiveModel, Transition, WeightedTransition,
    },
    planning::{
        guidance::PolicyGuide,
        rollout::{
            Depth, ExactConfig, Horizon, RolloutPolicy, SampledConfig, Trigger, exact, sampled,
        },
    },
};
use std::num::{NonZeroU16, NonZeroU32};

const RETURNS: PlanningReturns = PlanningReturns {
    objective: ReturnObjective::Physical,
    gamma: 0.8,
};
const MODEL: [u8; 32] = [7; 32];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Root,
    Signal(bool),
    Tail,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Choice {
    Probe,
    Safe,
    Repair(bool),
    Baseline,
    TooOptimistic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Event {
    Signal(bool),
    Tail,
    Done,
}

const ROOT: [Choice; 2] = [Choice::Probe, Choice::Safe];

struct Model;
impl Model {
    fn transition(state: State, choice: Choice, hidden: bool) -> Transition<State, Event> {
        let (observation, reward, duration, next) = match (state, choice) {
            (State::Root, Choice::Probe) => {
                (Event::Signal(hidden), 0., 2, Some(State::Signal(hidden)))
            }
            (State::Root, Choice::Safe) => (Event::Done, 0.3, 1, None),
            (State::Signal(signal), Choice::Repair(guess)) if guess == signal => {
                (Event::Tail, 0., 1, Some(State::Tail))
            }
            (State::Signal(_), Choice::Repair(_)) => (Event::Done, 0., 1, None),
            (State::Tail, Choice::Baseline) => (Event::Done, 0.7, 1, None),
            (State::Tail, Choice::TooOptimistic) => (Event::Done, 1.5, 1, None),
            _ => panic!("illegal hypothetical action"),
        };
        Transition {
            observation,
            reward,
            duration: NonZeroU16::new(duration).unwrap(),
            next,
        }
    }
}

impl PlanningModel for Model {
    type Context = State;
    type Action = Choice;
    type Observation = Event;

    fn actions(&self, state: &State, out: &mut Vec<Choice>) {
        out.extend(match state {
            State::Root => ROOT,
            State::Signal(false) => [Choice::Repair(false), Choice::Repair(true)],
            State::Signal(true) => [Choice::Repair(true), Choice::Repair(false)],
            State::Tail => [Choice::Baseline, Choice::TooOptimistic],
        });
    }

    fn planning_returns(&self) -> PlanningReturns {
        RETURNS
    }
}

impl PredictiveModel for Model {
    type Hidden = bool;

    fn sample_root(&self, _: &State, rng: &mut ChaCha8Rng) -> Result<bool> {
        Ok(rng.random())
    }
    fn rollout(&self, _: &State) -> Option<Choice> {
        panic!("rollout must use the frozen policy")
    }
    fn simulate(
        &self,
        state: &State,
        hidden: &mut bool,
        action: &Choice,
        _: &mut ChaCha8Rng,
    ) -> Result<Transition<State, Event>> {
        Ok(Self::transition(*state, *action, *hidden))
    }
}

impl EnumerableModel for Model {
    fn enumerate(
        &self,
        state: &State,
        action: &Choice,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<State, Event>> {
        let random = (*state, *action) == (State::Root, Choice::Probe);
        let probabilities = if random {
            &[(false, 0.5), (true, 0.5)][..]
        } else {
            &[(false, 1.)][..]
        };
        Ok(OutcomeDistribution {
            complete: probabilities.len() <= cap.get() as usize,
            outcomes: probabilities
                .iter()
                .take(cap.get() as usize)
                .map(|&(signal, probability)| WeightedTransition {
                    probability,
                    transition: Self::transition(*state, *action, signal),
                })
                .collect(),
        })
    }
}

struct Policy;
impl PolicyGuide<State, Choice> for Policy {
    fn prior(&self, state: &State, _: &[Choice]) -> Result<Vec<f64>> {
        Ok(match state {
            State::Root => vec![0., 1.],
            State::Signal(_) => vec![0.5, 0.5],
            State::Tail => vec![1., 0.],
        })
    }
}

fn policy() -> RolloutPolicy<'static, State, Choice> {
    RolloutPolicy {
        guide: &Policy,
        returns: RETURNS,
        revision: 14,
    }
}

fn cap(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}

fn horizon(depth: Depth, bound: u16) -> Horizon {
    Horizon {
        depth,
        termination_bound: NonZeroU16::new(bound).unwrap(),
    }
}

fn request(budget: PlanningBudget) -> PlanningRequest<'static, State, Choice> {
    PlanningRequest {
        context: &State::Root,
        feasible: &ROOT,
        budget,
    }
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

#[test]
fn exact_two_layers_adapt_to_each_signal_and_finish_with_the_frozen_policy() -> Result<()> {
    let run = |depth| {
        exact(
            &Model,
            MODEL,
            request(PlanningBudget::Expansions(cap(100))),
            policy(),
            ExactConfig {
                horizon: horizon(depth, 3),
                max_transitions: cap(100),
            },
        )
    };
    let one = run(Depth::One)?;
    let two = run(Depth::Two)?;
    let expected = 0.8_f64.powi(3) * 0.7;
    assert_eq!(one.decision.selected.unwrap().action, Choice::Safe);
    close(one.decision.estimates[0].value, expected / 2.);
    assert_eq!(two.decision.status, PlanStatus::Exact);
    close(two.decision.estimates[0].value, expected);
    assert_eq!(
        two.decision.estimates[0].upper_bound,
        Some(two.decision.estimates[0].value)
    );
    assert_eq!(two.baseline_revision, 14);
    assert_eq!(two.returns, RETURNS);
    let selected = two.decision.selected.unwrap();
    assert_eq!(selected.action, Choice::Probe);
    assert_eq!(
        selected.continuations,
        [
            (Event::Signal(false), Choice::Repair(false)),
            (Event::Signal(true), Choice::Repair(true)),
        ]
    );
    assert_eq!(two.decision.work.expansions, 5);
    assert_eq!(two.decision.work.transitions, 9);
    assert_eq!(two.policy_evaluations, 3);
    Ok(())
}

#[test]
fn incomplete_search_returns_the_public_baseline_without_partial_action_values() -> Result<()> {
    for (nodes, transitions, expected) in [(1, 30, 2), (30, 1, 1)] {
        let result = exact(
            &Model,
            MODEL,
            request(PlanningBudget::Expansions(cap(nodes))),
            policy(),
            ExactConfig {
                horizon: horizon(Depth::Two, 3),
                max_transitions: cap(transitions),
            },
        )?;
        assert_eq!(result.decision.status, PlanStatus::Exhausted);
        assert_eq!(result.decision.selected.unwrap().action, Choice::Safe);
        assert!(result.decision.estimates.is_empty());
        assert_eq!(result.decision.work.transitions, expected);
        assert_eq!(result.policy_evaluations, 1);
    }
    let budget = PlanningBudget::Simulations {
        trajectories: cap(8),
        transitions: cap(1),
    };
    let result = sampled(
        &Model,
        MODEL,
        request(budget),
        policy(),
        SampledConfig {
            horizon: horizon(Depth::Two, 3),
            max_expansions: cap(10),
            exploration: 0.1,
        },
        &mut ChaCha8Rng::seed_from_u64(5),
    )?;
    assert_eq!(result.decision.status, PlanStatus::Exhausted);
    assert_eq!(result.decision.selected.unwrap().action, Choice::Safe);
    assert!(result.decision.estimates.is_empty());
    assert_eq!(result.decision.work.transitions, 1);
    assert_eq!(result.decision.work.simulations, 0);
    assert_eq!(
        Trigger::Never.deliberate(&ROOT, budget),
        Deliberation::Execute
    );
    assert_eq!(
        Trigger::WhenChoices.deliberate(&ROOT[..1], budget),
        Deliberation::Execute
    );
    assert_eq!(
        Trigger::WhenChoices.deliberate(&ROOT, budget),
        Deliberation::Plan(budget)
    );
    Ok(())
}

#[test]
fn sampled_rollout_bounds_the_tree_and_samples_the_actual_public_policy_law() -> Result<()> {
    let budget = PlanningBudget::Simulations {
        trajectories: cap(8_000),
        transitions: cap(30_000),
    };
    let run = |depth| {
        sampled(
            &Model,
            MODEL,
            request(budget),
            policy(),
            SampledConfig {
                horizon: horizon(depth, 3),
                max_expansions: cap(30),
                exploration: 0.35,
            },
            &mut ChaCha8Rng::seed_from_u64(818),
        )
    };
    let one = run(Depth::One)?;
    let two = run(Depth::Two)?;
    assert_eq!(one.decision.status, PlanStatus::Approximate);
    assert_eq!(one.decision.work.expansions, 1);
    assert_eq!(one.decision.selected.unwrap().action, Choice::Safe);
    assert!((one.decision.estimates[0].value - 0.1792).abs() < 0.035);
    assert_eq!(two.decision.work.expansions, 3);
    assert!(two.policy_evaluations > 1);
    let selected = two.decision.selected.unwrap();
    assert_eq!(selected.action, Choice::Probe);
    assert!(
        selected
            .continuations
            .contains(&(Event::Signal(false), Choice::Repair(false)))
    );
    assert!(
        selected
            .continuations
            .contains(&(Event::Signal(true), Choice::Repair(true)))
    );
    assert!((two.decision.estimates[0].value - 0.3584).abs() < 0.03);
    Ok(())
}

#[test]
fn live_cutoff_is_a_failed_model_guarantee_and_objective_mismatch_is_rejected() -> Result<()> {
    let exact_config = ExactConfig {
        horizon: horizon(Depth::One, 1),
        max_transitions: cap(30),
    };
    let exact_budget = PlanningBudget::Expansions(cap(30));
    let failed = exact(&Model, MODEL, request(exact_budget), policy(), exact_config)?;
    assert_eq!(failed.decision.status, PlanStatus::Failed);
    assert_eq!(failed.decision.selected.unwrap().action, Choice::Safe);
    assert!(
        failed
            .decision
            .failure
            .unwrap()
            .contains("termination bound")
    );
    let budget = PlanningBudget::Simulations {
        trajectories: cap(8),
        transitions: cap(20),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut before = rng.clone();
    let failed = sampled(
        &Model,
        MODEL,
        request(budget),
        policy(),
        SampledConfig {
            horizon: horizon(Depth::One, 1),
            max_expansions: cap(5),
            exploration: 0.,
        },
        &mut rng,
    )?;
    assert_eq!(failed.decision.status, PlanStatus::Failed);
    assert_eq!(failed.decision.selected.unwrap().action, Choice::Safe);
    assert_eq!(rng.random::<u64>(), before.random::<u64>());
    let mut incompatible = policy();
    incompatible.returns.gamma = 1.;
    assert!(
        exact(
            &Model,
            MODEL,
            request(exact_budget),
            incompatible,
            exact_config
        )
        .is_err()
    );
    Ok(())
}
