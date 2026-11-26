use super::*;
use crate::models::predictive::{
    PlanningModel, PlanningReturns, PredictiveModel, ReturnObjective, Transition,
};
use crate::planning::guidance::{
    self, PolicyGuide, SearchOptions, TreeSelection, ValueCutoff, ValueGuide,
};
use rand::SeedableRng;
use std::cell::Cell;
use std::num::NonZeroU32;

#[path = "strategy_tests.rs"]
mod strategy_tests;

struct Advice {
    calls: Cell<u32>,
    misleading: bool,
}
impl PolicyGuide<Context, Action> for Advice {
    fn prior(&self, _: &Context, actions: &[Action]) -> Result<Vec<f64>> {
        self.calls.set(self.calls.get() + 1);
        let weights: Vec<_> = actions
            .iter()
            .map(|a| {
                if self.misleading && *a == Action::Scan {
                    0.05
                } else {
                    1.
                }
            })
            .collect();
        let sum: f64 = weights.iter().sum();
        Ok(weights.iter().map(|p| p / sum).collect())
    }
}
impl ValueGuide<Context> for Advice {
    fn value(&self, context: &Context, remaining: u16, _: &PlanningReturns) -> Result<f64> {
        assert!(matches!(context, Context::Reading(_)));
        assert_eq!(remaining, 1);
        Ok(0.85)
    }
}
fn options() -> SearchOptions<'static, Context, Action> {
    SearchOptions::new(TwoDoors.planning_returns())
}

#[test]
fn optional_controls_preserve_legacy_results_and_rng() {
    for seed in 0..8 {
        let mut old_rng = ChaCha8Rng::seed_from_u64(seed);
        let mut new_rng = old_rng.clone();
        let old = search(
            &TwoDoors,
            &Context::Root,
            &root(),
            config(),
            budget(256, 512),
            &mut old_rng,
        )
        .unwrap();
        let new = guidance::search_with_options(
            &TwoDoors,
            &Context::Root,
            &root(),
            config(),
            budget(256, 512),
            &mut new_rng,
            options(),
        )
        .unwrap();
        assert_eq!(old, new);
        assert_eq!(old_rng, new_rng);
    }
}

#[test]
fn puct_caches_expansion_priors_and_overcomes_misleading_positive_mass() {
    let mut correct = 0;
    for seed in 0..16 {
        let advice = Advice {
            calls: Cell::new(0),
            misleading: true,
        };
        let plan = guidance::search_with_options(
            &TwoDoors,
            &Context::Root,
            &root(),
            config(),
            budget(4096, 8192),
            &mut ChaCha8Rng::seed_from_u64(seed),
            SearchOptions {
                selection: TreeSelection::Puct { exploration: 2. },
                policy: Some(&advice),
                ..options()
            },
        )
        .unwrap();
        assert_eq!(advice.calls.get(), plan.work.expansions);
        assert_eq!(plan.work.simulations, 4096);
        correct += usize::from(plan.action == Some(Action::Scan));
    }
    assert!(correct >= 14, "correct={correct}/16");
}

#[test]
fn public_rollout_can_be_enabled_without_tree_or_value_guidance() {
    struct PublicRolloutOnly;
    impl PlanningModel for PublicRolloutOnly {
        type Context = Context;
        type Action = Action;
        type Observation = bool;

        fn actions(&self, context: &Context, actions: &mut Vec<Action>) {
            TwoDoors.actions(context, actions);
        }

        fn planning_returns(&self) -> PlanningReturns {
            TwoDoors.planning_returns()
        }
    }

    impl PredictiveModel for PublicRolloutOnly {
        type Hidden = bool;
        fn sample_root(&self, context: &Context, rng: &mut ChaCha8Rng) -> Result<bool> {
            TwoDoors.sample_root(context, rng)
        }
        fn rollout(&self, _: &Context) -> Option<Action> {
            panic!("public policy must replace model rollout")
        }
        fn simulate(
            &self,
            context: &Context,
            hidden: &mut bool,
            action: &Action,
            rng: &mut ChaCha8Rng,
        ) -> Result<Transition<Context, bool>> {
            TwoDoors.simulate(context, hidden, action, rng)
        }
    }
    let advice = Advice {
        calls: Cell::new(0),
        misleading: false,
    };
    let mut expected_rng = ChaCha8Rng::seed_from_u64(81);
    let mut actual_rng = expected_rng.clone();
    let expected = search_with_rollout(
        &PublicRolloutOnly,
        &Context::Root,
        &root(),
        config(),
        budget(64, 128),
        &mut expected_rng,
        |context, _| {
            let mut actions = Vec::new();
            TwoDoors.actions(context, &mut actions);
            Ok(actions
                .first()
                .copied()
                .map_or(RolloutDecision::End, RolloutDecision::Act))
        },
    )
    .unwrap();
    let actual = guidance::search_with_options(
        &PublicRolloutOnly,
        &Context::Root,
        &root(),
        config(),
        budget(64, 128),
        &mut actual_rng,
        SearchOptions {
            policy: Some(&advice),
            rollout_policy: true,
            ..options()
        },
    )
    .unwrap();
    assert!(advice.calls.get() > 0);
    assert_eq!(expected, actual);
    assert_eq!(expected_rng, actual_rng);
}

#[test]
fn value_cutoff_applies_to_revisited_tree_nodes_and_not_terminals() {
    let advice = Advice {
        calls: Cell::new(0),
        misleading: false,
    };
    let plan = guidance::search_with_options(
        &TwoDoors,
        &Context::Root,
        &[Action::Scan],
        config(),
        budget(32, 32),
        &mut ChaCha8Rng::seed_from_u64(5),
        SearchOptions {
            value: Some(ValueCutoff {
                depth: NonZeroU16::MIN,
                guide: &advice,
            }),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(plan.work.transitions, 32);
    assert_eq!(plan.work.simulations, 32);
    assert!((plan.estimates[0].value - 0.6885).abs() < 1e-12);
    // The guide asserts if called outside a reading; guesses terminate directly.
    let terminal = guidance::search_with_options(
        &TwoDoors,
        &Context::Root,
        &[Action::Guess(true)],
        config(),
        budget(32, 32),
        &mut ChaCha8Rng::seed_from_u64(5),
        SearchOptions {
            value: Some(ValueCutoff {
                depth: NonZeroU16::MIN,
                guide: &advice,
            }),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(terminal.work.simulations, 32);
}

#[test]
fn expansion_limit_discards_the_incomplete_return() {
    let plan = guidance::search_with_options(
        &TwoDoors,
        &Context::Root,
        &[Action::Guess(true), Action::Scan],
        config(),
        budget(32, 64),
        &mut ChaCha8Rng::seed_from_u64(5),
        SearchOptions {
            max_expansions: Some(NonZeroU32::MIN),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Exhausted);
    assert_eq!(plan.work.expansions, 1);
    assert_eq!(plan.work.transitions, 2);
    assert_eq!(plan.work.simulations, 1);
    assert_eq!(plan.estimates.len(), 1);
}

#[test]
fn guided_errors_do_not_install_rng_or_invalid_estimates() {
    struct Invalid;
    impl PolicyGuide<Context, Action> for Invalid {
        fn prior(&self, context: &Context, actions: &[Action]) -> Result<Vec<f64>> {
            Ok(vec![
                if matches!(context, Context::Root) {
                    1. / actions.len() as f64
                } else {
                    f64::NAN
                };
                actions.len()
            ])
        }
    }
    impl ValueGuide<Context> for Invalid {
        fn value(&self, _: &Context, _: u16, _: &PlanningReturns) -> Result<f64> {
            Ok(f64::INFINITY)
        }
    }
    for leaf in [false, true] {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let before = rng.clone();
        let mut configured = options();
        if leaf {
            configured.value = Some(ValueCutoff {
                depth: NonZeroU16::MIN,
                guide: &Invalid,
            });
        } else {
            configured.selection = TreeSelection::Puct { exploration: 1. };
            configured.policy = Some(&Invalid);
        }
        let plan = guidance::search_with_options(
            &TwoDoors,
            &Context::Root,
            &root(),
            config(),
            budget(16, 32),
            &mut rng,
            configured,
        )
        .unwrap();
        assert_eq!(plan.status, PlanStatus::Failed);
        assert_eq!(plan.work.transitions, 1);
        assert_eq!(plan.work.simulations, 0);
        assert!(plan.action.is_none() && plan.estimates.is_empty());
        assert_eq!(rng, before);
    }
    for coefficient in [0., -1., f64::NAN, f64::INFINITY] {
        let mut rng = ChaCha8Rng::seed_from_u64(6);
        let before = rng.clone();
        assert!(
            guidance::search_with_options(
                &TwoDoors,
                &Context::Root,
                &root(),
                config(),
                budget(16, 32),
                &mut rng,
                SearchOptions {
                    selection: TreeSelection::Puct {
                        exploration: coefficient
                    },
                    policy: Some(&Invalid),
                    ..options()
                }
            )
            .is_err()
        );
        assert_eq!(rng, before);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Scan,
    Guess(bool),
}

#[derive(Clone, Copy)]
enum Context {
    Root,
    Reading(bool),
    ChainStart,
    FirstOfTwo,
    SecondReading(bool),
}

struct TwoDoors;

impl PlanningModel for TwoDoors {
    type Context = Context;
    type Action = Action;
    type Observation = bool;

    fn actions(&self, context: &Context, actions: &mut Vec<Action>) {
        match context {
            Context::Root | Context::ChainStart | Context::FirstOfTwo => {
                actions.extend([Action::Scan, Action::Guess(false), Action::Guess(true)])
            }
            Context::Reading(_) | Context::SecondReading(_) => {
                actions.extend([Action::Guess(false), Action::Guess(true)])
            }
        }
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: 0.9,
        }
    }
}

impl PredictiveModel for TwoDoors {
    type Hidden = bool;

    fn sample_root(&self, _: &Context, rng: &mut ChaCha8Rng) -> Result<bool> {
        Ok(rng.random())
    }

    fn rollout(&self, context: &Context) -> Option<Action> {
        Some(Action::Guess(match context {
            Context::Root | Context::ChainStart | Context::FirstOfTwo => true,
            Context::Reading(sign) | Context::SecondReading(sign) => *sign,
        }))
    }

    fn simulate(
        &self,
        context: &Context,
        hidden: &mut bool,
        action: &Action,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Context, bool>> {
        Ok(match action {
            Action::Scan => {
                let sign = if rng.random_bool(0.85) {
                    *hidden
                } else {
                    !*hidden
                };
                Transition {
                    observation: sign,
                    reward: 0.,
                    duration: NonZeroU16::new(2).unwrap(),
                    next: Some(match context {
                        Context::ChainStart => Context::FirstOfTwo,
                        Context::FirstOfTwo => Context::SecondReading(sign),
                        _ => Context::Reading(sign),
                    }),
                }
            }
            Action::Guess(which) => Transition {
                observation: *which == *hidden,
                reward: f64::from(*which == *hidden),
                duration: NonZeroU16::MIN,
                next: None,
            },
        })
    }
}

fn config() -> Config {
    Config {
        max_depth: NonZeroU16::new(2).unwrap(),
        gamma: 0.9,
        exploration: 1.0,
    }
}

fn budget(n: u32, transitions: u32) -> PlanningBudget {
    PlanningBudget::Simulations {
        trajectories: NonZeroU32::new(n).unwrap(),
        transitions: NonZeroU32::new(transitions).unwrap(),
    }
}

fn root() -> [Action; 3] {
    [Action::Scan, Action::Guess(false), Action::Guess(true)]
}

#[test]
fn analytic_leaf_backs_up_public_value_at_the_physical_discount() {
    let mut rng = ChaCha8Rng::seed_from_u64(1027);
    let mut seen = [false; 2];
    for _ in 0..16 {
        let plan = search_with_rollout(
            &TwoDoors,
            &Context::Root,
            &[Action::Scan],
            config(),
            budget(1, 1),
            &mut rng,
            |context, step| {
                let Context::Reading(sign) = context else {
                    panic!("continuation starts at a newly observed public history");
                };
                assert_eq!(
                    step,
                    RolloutStep {
                        index: 0,
                        remaining: 1
                    }
                );
                seen[usize::from(*sign)] = true;
                Ok(RolloutDecision::Value(0.85))
            },
        )
        .unwrap();
        assert_eq!(plan.action, Some(Action::Scan));
        assert_eq!(plan.work.simulations, 1);
        assert_eq!(plan.work.transitions, 1);
        assert!((plan.estimates[0].value - 0.85 * 0.9_f64.powi(2)).abs() < 1e-15);
    }
    assert_eq!(seen, [true, true]);
}

#[test]
fn continuation_limit_discards_incomplete_trajectory_and_keeps_completed_work() {
    let mut rng = ChaCha8Rng::seed_from_u64(225);
    let plan = search_with_rollout(
        &TwoDoors,
        &Context::Root,
        &[Action::Guess(false), Action::Scan],
        config(),
        budget(16, 32),
        &mut rng,
        |_, _| Ok(RolloutDecision::Exhausted),
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Exhausted);
    assert_eq!(plan.action, Some(Action::Guess(false)));
    assert_eq!(plan.work.transitions, 2);
    assert_eq!(plan.work.simulations, 1);
    assert_eq!(plan.estimates.len(), 1);
}

#[test]
fn rollout_steps_advance_with_observed_successors() {
    let mut rng = ChaCha8Rng::seed_from_u64(1041);
    let mut expected_rng = rng.clone();
    let _: bool = expected_rng.random();
    expected_rng.random_bool(0.85);
    let correct = expected_rng.random_bool(0.85);
    let mut calls = 0;
    let plan = search_with_rollout(
        &TwoDoors,
        &Context::ChainStart,
        &[Action::Scan],
        Config {
            max_depth: NonZeroU16::new(3).unwrap(),
            ..config()
        },
        budget(1, 3),
        &mut rng,
        |context, step| {
            calls += 1;
            match (step.index, step.remaining, context) {
                (0, 2, Context::FirstOfTwo) => Ok(RolloutDecision::Act(Action::Scan)),
                (1, 1, Context::SecondReading(sign)) => {
                    Ok(RolloutDecision::Act(Action::Guess(*sign)))
                }
                _ => panic!("unexpected rollout depth or public successor"),
            }
        },
    )
    .unwrap();
    assert_eq!(calls, 2);
    assert_eq!(plan.work.transitions, 3);
    assert_eq!(plan.work.simulations, 1);
    let expected = f64::from(correct) * 0.9_f64.powi(4);
    assert!((plan.estimates[0].value - expected).abs() < 1e-15);
    assert_eq!(rng, expected_rng);
}

#[test]
fn custom_rollout_cannot_leave_public_action_support() {
    let mut rng = ChaCha8Rng::seed_from_u64(225);
    let before = rng.clone();
    let plan = search_with_rollout(
        &TwoDoors,
        &Context::Root,
        &[Action::Scan],
        config(),
        budget(16, 32),
        &mut rng,
        |_, _| Ok(RolloutDecision::Act(Action::Scan)),
    )
    .unwrap();
    assert_eq!(plan.status, PlanStatus::Failed);
    assert_eq!(plan.work.transitions, 1);
    assert_eq!(plan.work.simulations, 0);
    assert_eq!(rng, before);
}

#[test]
fn contingent_root_value_approaches_independently_enumerated_solution() {
    // Direct guesses have 1/2 success. Scan reveals the cue with probability
    // 0.85 and delays the guessing reward by two physical ticks.
    let optimal = 0.85 * 0.9_f64.powi(2);
    for n in [64, 256, 1024, 4096] {
        let mut good = 0;
        let mut scan_error = 0.;
        let mut guess_error = 0.;
        for seed in 0..16 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let search = search(
                &TwoDoors,
                &Context::Root,
                &root(),
                config(),
                budget(n, 2 * n),
                &mut rng,
            )
            .unwrap();
            let [scan, left, right] = search.estimates.as_slice() else {
                panic!("every candidate must have a root estimate");
            };
            assert_eq!([scan.action, left.action, right.action], root());
            good += usize::from(search.action == Some(Action::Scan));
            scan_error += (scan.value - optimal).abs();
            guess_error += (left.value - 0.5).abs() + (right.value - 0.5).abs();
            assert_eq!(search.work.simulations, n);
            assert_eq!(
                scan.visits.unwrap() + left.visits.unwrap() + right.visits.unwrap(),
                n
            );
            assert!(search.work.transitions > n && search.work.transitions <= 2 * n);
            if n == 4096 && search.action == Some(Action::Scan) {
                assert!(
                    search
                        .continuations
                        .contains(&(false, Action::Guess(false)))
                );
                assert!(search.continuations.contains(&(true, Action::Guess(true))));
            }
        }
        eprintln!(
            "{n} simulations: optimum {good}/16, scan MAE {:.6}, guess MAE {:.6}",
            scan_error / 16.0,
            guess_error / 32.0,
        );
        if n == 4096 {
            assert!(good >= 14);
            assert!(scan_error / 16.0 < 0.05);
            assert!(guess_error / 32.0 < 0.08);
        }
    }
}

#[test]
fn transition_budget_discards_incomplete_returns_and_same_seed_replays_work() {
    let mut rng = ChaCha8Rng::seed_from_u64(91);
    let mut replay = rng.clone();
    let first = search(
        &TwoDoors,
        &Context::Root,
        &root(),
        config(),
        budget(64, 1),
        &mut rng,
    )
    .unwrap();
    assert_eq!(first.action, None);
    assert_eq!(first.work.transitions, 1);
    assert_eq!(first.work.simulations, 0);
    assert_eq!(first.status, PlanStatus::Exhausted);
    let clone = search(
        &TwoDoors,
        &Context::Root,
        &root(),
        config(),
        budget(64, 1),
        &mut replay,
    )
    .unwrap();
    assert_eq!(first, clone);
    assert_eq!(rng, replay);

    let guess_only = [Action::Guess(false)];
    let search = search(
        &TwoDoors,
        &Context::Root,
        &guess_only,
        config(),
        budget(64, 64),
        &mut rng,
    )
    .unwrap();
    assert_eq!(search.action, Some(Action::Guess(false)));
    assert_eq!(search.work.transitions, 64);
    assert_eq!(search.work.simulations, 64);
    assert_eq!(search.status, PlanStatus::Approximate);
}

#[test]
fn supplied_support_and_root_mass_are_enforced() {
    let mut rng = ChaCha8Rng::seed_from_u64(784);
    let original = rng.clone();
    for weights in [[0., 0., 0.], [0., f64::NAN, 1.], [-1., 0., 1.]] {
        assert!(sample_weighted(weights, &mut rng).is_err());
        assert_eq!(rng, original);
    }
    for _ in 0..64 {
        assert_eq!(sample_weighted([0., 1., 0.], &mut rng).unwrap(), 1);
    }
    let duplicate = [Action::Guess(false), Action::Guess(false)];
    let original = rng.clone();
    assert!(
        search(
            &TwoDoors,
            &Context::Root,
            &duplicate,
            config(),
            budget(4, 8),
            &mut rng
        )
        .is_err()
    );
    assert_eq!(rng, original);
}

#[test]
fn uct_visits_each_legal_edge_before_using_its_empirical_value() {
    let mut node = Node::<_, bool>::new(&root(), None);
    assert_eq!(node.uct(1.0), Some(0));
    node.visits = 1;
    node.edges[0].visits = 1;
    node.edges[0].value = 1.;
    assert_eq!(node.uct(1.0), Some(1));
    node.visits = 102;
    node.edges[0].visits = 100;
    node.edges[1].visits = 1;
    node.edges[2].visits = 1;
    assert_eq!(node.uct(0.), Some(0));
    assert_eq!(node.uct(1.), Some(1));
}

#[test]
fn failed_model_retains_attempted_work_and_does_not_install_rng_or_a_decision() {
    struct Fails(TwoDoors);
    impl PlanningModel for Fails {
        type Context = Context;
        type Action = Action;
        type Observation = bool;

        fn actions(&self, context: &Context, actions: &mut Vec<Action>) {
            self.0.actions(context, actions);
        }

        fn planning_returns(&self) -> PlanningReturns {
            self.0.planning_returns()
        }
    }

    impl PredictiveModel for Fails {
        type Hidden = bool;

        fn sample_root(&self, context: &Context, rng: &mut ChaCha8Rng) -> Result<bool> {
            self.0.sample_root(context, rng)
        }
        fn rollout(&self, context: &Context) -> Option<Action> {
            self.0.rollout(context)
        }
        fn simulate(
            &self,
            context: &Context,
            hidden: &mut bool,
            action: &Action,
            rng: &mut ChaCha8Rng,
        ) -> Result<Transition<Context, bool>> {
            if matches!(context, Context::Reading(_)) {
                return Err(Error::State("injected hypothetical step failure".into()));
            }
            self.0.simulate(context, hidden, action, rng)
        }
    }
    let mut rng = ChaCha8Rng::seed_from_u64(724);
    let prior = rng.clone();
    let result = search(
        &Fails(TwoDoors),
        &Context::Root,
        &root(),
        config(),
        budget(64, 128),
        &mut rng,
    )
    .unwrap();
    assert_eq!(result.status, PlanStatus::Failed);
    assert!(result.failure.unwrap().contains("injected"));
    assert_eq!(result.work.expansions, 2);
    assert_eq!(result.work.transitions, 2);
    assert_eq!(result.work.simulations, 0);
    assert!(result.action.is_none());
    assert_eq!(rng, prior);
}

#[test]
fn direct_sampled_search_validates_the_declared_gamma_before_using_rng() {
    let root = Context::Root;
    let config = Config {
        gamma: 0.8,
        ..config()
    };
    let mut rng = ChaCha8Rng::seed_from_u64(19);
    let before = serde_json::to_value(&rng).unwrap();
    let feasible = &[Action::Scan];
    assert!(search(&TwoDoors, &root, feasible, config, budget(1, 3), &mut rng).is_err());
    assert!(
        search_with_rollout(
            &TwoDoors,
            &root,
            feasible,
            config,
            budget(1, 3),
            &mut rng,
            |_, _| panic!("the rollout callback must not run for a rejected request"),
        )
        .is_err()
    );
    assert_eq!(serde_json::to_value(&rng).unwrap(), before);
}
