use std::num::{NonZeroU16, NonZeroU32};

use rl::{
    Result,
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, ReturnObjective,
        Transition, WeightedTransition,
    },
    planning::{
        finite::{FiniteModel, Status},
        guidance::PolicyGuide,
    },
};

type Action = &'static str;

#[derive(Clone, Copy)]
struct Row {
    observation: u8,
    probability: f64,
    reward: f64,
    duration: u16,
    next: Option<u8>,
}

impl Row {
    fn certain(reward: f64, next: Option<u8>) -> Self {
        Self {
            observation: 0,
            probability: 1.,
            reward,
            duration: 1,
            next,
        }
    }
}

struct Model {
    gamma: f64,
    states: Vec<Vec<(Action, Vec<Row>)>>,
    obey_cap: bool,
}

impl Model {
    fn new(gamma: f64, states: Vec<Vec<(Action, Vec<Row>)>>) -> Self {
        Self {
            gamma,
            states,
            obey_cap: true,
        }
    }
}

impl PlanningModel for Model {
    type Context = u8;
    type Action = Action;
    type Observation = u8;

    fn actions(&self, context: &u8, actions: &mut Vec<Action>) {
        actions.extend(
            self.states[usize::from(*context)]
                .iter()
                .map(|(key, _)| key),
        );
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: self.gamma,
        }
    }
}

impl EnumerableModel for Model {
    fn enumerate(
        &self,
        context: &u8,
        action: &Action,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<u8, u8>> {
        let (_, rows) = self.states[usize::from(*context)]
            .iter()
            .find(|(key, _)| key == action)
            .expect("the planner only enumerates a supplied action");
        let supplied = if self.obey_cap {
            rows.len().min(cap.get() as usize)
        } else {
            rows.len()
        };
        let outcomes = rows[..supplied]
            .iter()
            .map(|row| WeightedTransition {
                probability: row.probability,
                transition: Transition {
                    observation: row.observation,
                    reward: row.reward,
                    duration: NonZeroU16::new(row.duration).unwrap(),
                    next: row.next,
                },
            })
            .collect();
        Ok(OutcomeDistribution {
            outcomes,
            complete: supplied == rows.len(),
        })
    }
}

struct FrozenPolicy(Vec<Vec<f64>>);

impl PolicyGuide<u8, Action> for FrozenPolicy {
    fn prior(&self, context: &u8, _feasible: &[Action]) -> Result<Vec<f64>> {
        Ok(self.0[usize::from(*context)].clone())
    }
}

fn positive(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-10, "{actual} != {expected}");
}

fn cyclic_model() -> Model {
    Model::new(
        0.5,
        vec![
            vec![
                ("stay", vec![Row::certain(1., Some(0))]),
                ("go", vec![Row::certain(0.25, Some(1))]),
                ("also_go", vec![Row::certain(0.25, Some(1))]),
            ],
            vec![
                ("back", vec![Row::certain(1., Some(0))]),
                ("exit", vec![Row::certain(4., None)]),
            ],
        ],
    )
}

#[test]
fn cyclic_value_iteration_solves_bellman_equations_in_model_action_order() -> Result<()> {
    let table = FiniteModel::from_model(&cyclic_model(), &[0, 1], positive(2))?;
    let solution = table.value_iteration(positive(100), 1e-11)?;
    assert_eq!(solution.status, Status::Converged);
    assert!(solution.residual <= 5e-12);
    let [root, other] = solution.states.as_slice() else {
        panic!("the solver must return a row for each live context")
    };
    assert_eq!((root.context, other.context), (0, 1));
    close(root.value, 2.25);
    close(other.value, 4.);
    for (actual, (action, value)) in
        root.action_values()
            .iter()
            .zip([("stay", 2.125), ("go", 2.25), ("also_go", 2.25)])
    {
        assert_eq!(actual.action, action);
        close(actual.value, value);
    }
    assert_eq!(*root.greedy_action(), "go");
    assert_eq!(*other.greedy_action(), "exit");
    Ok(())
}

#[test]
fn cyclic_policy_evaluation_uses_the_frozen_probabilities() -> Result<()> {
    let table = FiniteModel::from_model(&cyclic_model(), &[0, 1], positive(2))?;
    let policy = FrozenPolicy(vec![vec![0.5, 0.5, 0.], vec![1., 0.]]);
    let evaluated = table.evaluate_policy(&policy, positive(100), 1e-11)?;
    assert_eq!(evaluated.status, Status::Converged);
    close(evaluated.states[0].value, 1.4);
    close(evaluated.states[1].value, 1.7);
    close(evaluated.states[0].action_values()[2].value, 1.1);
    Ok(())
}

#[test]
fn iteration_limit_reports_residual_on_the_returned_iterate() -> Result<()> {
    let table = FiniteModel::from_model(&cyclic_model(), &[0, 1], positive(2))?;
    let limited = table.value_iteration(positive(1), 1e-11)?;
    assert_eq!(limited.status, Status::IterationLimit);
    assert_eq!(limited.sweeps, 1);
    close(limited.states[0].value, 1.);
    close(limited.states[1].value, 4.);
    close(limited.residual, 1.25);
    Ok(())
}

fn two_tick_model(gamma: f64, action_reward: f64) -> Model {
    Model::new(
        gamma,
        vec![
            vec![
                (
                    "move",
                    vec![Row {
                        duration: 2,
                        ..Row::certain(action_reward, Some(1))
                    }],
                ),
                ("stop", vec![Row::certain(9., None)]),
            ],
            vec![(
                "result",
                vec![
                    Row {
                        probability: 0.75,
                        ..Row::certain(4., None)
                    },
                    Row {
                        observation: 1,
                        probability: 0.25,
                        ..Row::certain(-4., None)
                    },
                ],
            )],
        ],
    )
}

#[test]
fn unit_discount_solves_acyclic_models_successor_first() -> Result<()> {
    let table = FiniteModel::from_model(&two_tick_model(1., 8.), &[0, 1], positive(2))?;
    let solution = table.value_iteration(positive(1), 1e-11)?;
    assert_eq!(solution.status, Status::Converged);
    close(solution.states[1].value, 2.);
    close(solution.states[0].value, 10.);
    assert_eq!(*solution.states[0].greedy_action(), "move");
    let policy = FrozenPolicy(vec![vec![0.25, 0.75], vec![1.]]);
    let evaluated = table.evaluate_policy(&policy, positive(1), 1e-11)?;
    close(evaluated.states[0].value, 9.25);
    Ok(())
}

#[test]
fn extended_actions_discount_only_continuation_over_their_duration() -> Result<()> {
    // The model already discounted the physical trace [2, 6] to 5.
    let table = FiniteModel::from_model(&two_tick_model(0.5, 5.), &[0, 1], positive(2))?;
    let solution = table.value_iteration(positive(100), 1e-11)?;
    close(solution.states[0].action_values()[0].value, 5.5);
    assert_eq!(*solution.states[0].greedy_action(), "stop");
    Ok(())
}

#[test]
fn public_belief_states_choose_opposite_actions_after_opposite_signals() -> Result<()> {
    const ROOT: u8 = 0;
    const SIGNAL_LEFT: u8 = 1;
    const SIGNAL_RIGHT: u8 = 2;
    let guess = |key, accuracy| {
        (
            key,
            vec![
                Row {
                    probability: accuracy,
                    ..Row::certain(1., None)
                },
                Row {
                    observation: 1,
                    probability: 1. - accuracy,
                    ..Row::certain(-1., None)
                },
            ],
        )
    };
    let signal = |side| Row {
        observation: side,
        probability: 0.5,
        duration: 2,
        ..Row::certain(-0.125, Some(side))
    };
    let model = Model::new(
        0.75,
        vec![
            vec![
                guess("left", 0.5),
                guess("right", 0.5),
                ("inspect", vec![signal(SIGNAL_LEFT), signal(SIGNAL_RIGHT)]),
            ],
            vec![guess("left", 0.75), guess("right", 0.25)],
            vec![guess("left", 0.25), guess("right", 0.75)],
        ],
    );
    let table = FiniteModel::from_model(&model, &[ROOT, SIGNAL_LEFT, SIGNAL_RIGHT], positive(2))?;
    let solution = table.value_iteration(positive(100), 1e-11)?;
    assert_eq!(solution.status, Status::Converged);
    assert_eq!(*solution.states[0].greedy_action(), "inspect");
    assert_eq!(*solution.states[1].greedy_action(), "left");
    assert_eq!(*solution.states[2].greedy_action(), "right");
    close(solution.states[0].value, 5. / 32.);
    close(solution.states[1].value, 0.5);
    close(solution.states[2].value, 0.5);

    let ignores_signal = FrozenPolicy(vec![vec![0., 0., 1.], vec![0., 1.], vec![0., 1.]]);
    let evaluated = table.evaluate_policy(&ignores_signal, positive(100), 1e-11)?;
    close(evaluated.states[0].value, -0.125);
    close(evaluated.states[1].value, -0.5);
    close(evaluated.states[2].value, 0.5);
    Ok(())
}

#[test]
fn unit_discount_rejects_live_cycles() {
    let mut model = cyclic_model();
    model.gamma = 1.;
    assert!(FiniteModel::from_model(&model, &[0, 1], positive(2)).is_err());
}

#[test]
fn capped_or_oversized_enumeration_cannot_supply_a_probability_law() {
    let mut model = two_tick_model(0.5, 5.);
    assert!(FiniteModel::from_model(&model, &[0, 1], positive(1)).is_err());
    model.obey_cap = false;
    assert!(FiniteModel::from_model(&model, &[0, 1], positive(1)).is_err());
}

#[test]
fn malformed_transition_laws_fail_before_solving() {
    let terminal = Row::certain(1., None);
    let invalid = [
        vec![],
        vec![Row {
            next: Some(1),
            ..terminal
        }],
        vec![Row {
            reward: f64::NAN,
            ..terminal
        }],
        vec![Row {
            probability: f64::INFINITY,
            ..terminal
        }],
        vec![Row {
            probability: 0.,
            ..terminal
        }],
        vec![Row {
            probability: -1.,
            ..terminal
        }],
        vec![Row {
            probability: 0.9,
            ..terminal
        }],
        vec![Row {
            probability: 1. + 5e-10,
            ..terminal
        }],
        vec![
            Row {
                probability: 0.5,
                ..terminal
            };
            2
        ],
    ];
    for rows in invalid {
        let model = Model::new(0.5, vec![vec![("exit", rows)]]);
        assert!(FiniteModel::from_model(&model, &[0], positive(2)).is_err());
    }
}

#[test]
fn ambiguous_or_empty_live_support_is_rejected() {
    let mut model = cyclic_model();
    assert!(FiniteModel::from_model(&model, &[0, 0, 1], positive(2)).is_err());
    model.states[0].push(("stay", vec![Row::certain(1., None)]));
    assert!(FiniteModel::from_model(&model, &[0, 1], positive(2)).is_err());
    model.states[0].clear();
    assert!(FiniteModel::from_model(&model, &[0, 1], positive(2)).is_err());
}

#[test]
fn policy_must_supply_a_finite_normalized_row_over_the_whole_menu() -> Result<()> {
    let table = FiniteModel::from_model(&cyclic_model(), &[0, 1], positive(2))?;
    for invalid in [
        vec![1.],
        vec![f64::NAN, 0., 0.],
        vec![-0.5, 1.5, 0.],
        vec![0.2, 0.3, 0.4],
    ] {
        let policy = FrozenPolicy(vec![invalid, vec![1., 0.]]);
        assert!(
            table
                .evaluate_policy(&policy, positive(100), 1e-11)
                .is_err()
        );
    }
    Ok(())
}

#[test]
fn finite_horizon_solves_an_undiscounted_loop_with_a_zero_final_value() -> Result<()> {
    let model = Model::new(
        1.,
        vec![vec![
            ("loop", vec![Row::certain(1., Some(0))]),
            ("stop", vec![Row::certain(1.5, None)]),
        ]],
    );
    let prepared = FiniteModel::for_horizon(&model, &[0], positive(1))?;
    let horizon = NonZeroU16::new(2).unwrap();
    let optimal = prepared.backward_induction(horizon)?;
    assert!(optimal.states(0).is_none());
    assert!(optimal.states(3).is_none());
    let last = &optimal.states(1).unwrap()[0];
    assert_eq!(*last.greedy_action(), "stop");
    close(last.value, 1.5);
    let first = &optimal.states(2).unwrap()[0];
    assert_eq!(*first.greedy_action(), "loop");
    close(first.value, 2.5);
    let policy = prepared.evaluate_horizon_policy(horizon, |remaining, _, _| {
        Ok(if remaining == 1 {
            vec![0.25, 0.75]
        } else {
            vec![0.5, 0.5]
        })
    })?;
    close(policy.states(1).unwrap()[0].value, 1.375);
    close(policy.states(2).unwrap()[0].value, 1.9375);
    assert!(prepared.value_iteration(positive(10), 1e-8).is_err());
    Ok(())
}

#[test]
fn finite_horizon_discounts_by_duration_and_keeps_legal_action_order() -> Result<()> {
    let model = two_tick_model(0.5, 5.);
    let prepared = FiniteModel::for_horizon(&model, &[0, 1], positive(2))?;
    let horizon = NonZeroU16::new(2).unwrap();
    let optimal = prepared.backward_induction(horizon)?;
    let action_value = |remaining| optimal.states(remaining).unwrap()[0].action_values()[0].value;
    close(action_value(1), 5.);
    close(action_value(2), 5.5);
    assert!(
        prepared
            .evaluate_horizon_policy(horizon, |_, _, _| Ok(vec![1.]))
            .is_err()
    );
    Ok(())
}
