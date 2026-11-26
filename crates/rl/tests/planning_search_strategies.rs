#[path = "support/investigate_or_commit.rs"]
mod investigate_or_commit;

use investigate_or_commit::{Action, Context, InvestigateOrCommit};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    models::predictive::PlanningModel,
    planning::{
        PlanStatus, PlanningBudget,
        finite::{FiniteModel, StateValue},
        guidance::{
            Backup, PolicyGuide, RootSelection, SearchOptions, SearchStrategy, search_with_strategy,
        },
        pomcp,
    },
};
use std::{
    num::{NonZeroU16, NonZeroU32},
    time::Instant,
};

const STRATEGIES: [(&str, SearchStrategy); 4] = [
    (
        "uct_mean",
        SearchStrategy {
            backup: Backup::MonteCarloMean,
            root_selection: RootSelection::TreePolicy,
        },
    ),
    (
        "uct_bellman",
        SearchStrategy {
            backup: Backup::Bellman,
            root_selection: RootSelection::TreePolicy,
        },
    ),
    (
        "halving_mean",
        SearchStrategy {
            backup: Backup::MonteCarloMean,
            root_selection: RootSelection::SequentialHalving,
        },
    ),
    (
        "halving_bellman",
        SearchStrategy {
            backup: Backup::Bellman,
            root_selection: RootSelection::SequentialHalving,
        },
    ),
];

fn positive(x: u32) -> NonZeroU32 {
    NonZeroU32::new(x).unwrap()
}

fn problem(signals: u16) -> InvestigateOrCommit {
    InvestigateOrCommit::new(NonZeroU16::new(signals).unwrap())
}

fn exact_root(model: &InvestigateOrCommit, signals: u16) -> Result<StateValue<Context, Action>> {
    let finite = FiniteModel::from_model(model, &model.contexts(), positive(u32::from(signals)))?;
    Ok(finite
        .value_iteration(NonZeroU32::MIN, 1e-12)?
        .states
        .remove(0))
}

#[test]
fn investigate_or_commit_exact_solution_and_uniform_guess_values() -> Result<()> {
    struct UniformGuesses;
    impl PolicyGuide<Context, Action> for UniformGuesses {
        fn prior(&self, context: &Context, actions: &[Action]) -> Result<Vec<f64>> {
            if *context == Context::Root {
                return Ok(vec![0., 1.]);
            }
            Ok(vec![1. / actions.len() as f64; actions.len()])
        }
    }

    for (signals, uniform_value) in [(2, 0.7), (4, 0.6), (8, 0.55), (16, 0.525)] {
        let model = problem(signals);
        let root = exact_root(&model, signals)?;
        assert_eq!(*root.greedy_action(), Action::Investigate);
        assert!((root.action_values()[0].value - 0.6).abs() < 1e-12);
        assert!((root.action_values()[1].value - 0.9).abs() < 1e-12);
        let policy =
            FiniteModel::from_model(&model, &model.contexts(), positive(u32::from(signals)))?
                .evaluate_policy(&UniformGuesses, NonZeroU32::MIN, 1e-12)?;
        assert!((policy.states[0].value - uniform_value).abs() < 1e-12);
    }
    Ok(())
}

#[test]
#[ignore = "seeded search and exact-reference comparison"]
fn investigate_or_commit_root_regret() -> Result<()> {
    const SEEDS: u64 = 16;
    for signals in [2, 4, 8, 16] {
        let model = problem(signals);
        let exact = exact_root(&model, signals)?;
        let model = model.with_expected_terminal_rewards();
        let config = pomcp::Config {
            max_depth: NonZeroU16::new(2).unwrap(),
            gamma: 1.,
            exploration: 1.,
        };
        for trajectories in [50, 200, 800, 3200] {
            let budget = PlanningBudget::Simulations {
                trajectories: positive(trajectories),
                transitions: positive(2 * trajectories),
            };
            for (label, strategy) in STRATEGIES {
                let start = Instant::now();
                let mut regret = 0.;
                let mut optimal = 0;
                let mut transitions = 0;
                for seed in 0..SEEDS {
                    let search = search_with_strategy(
                        &model,
                        &Context::Root,
                        &InvestigateOrCommit::ROOT_ACTIONS,
                        config,
                        budget,
                        &mut ChaCha8Rng::seed_from_u64(seed),
                        SearchOptions::new(model.planning_returns()).with_strategy(strategy),
                    )?;
                    assert_eq!(search.status, PlanStatus::Approximate);
                    assert_eq!(search.work.simulations, trajectories);
                    let selected = search.action.unwrap();
                    let estimate = exact
                        .action_values()
                        .iter()
                        .find(|row| row.action == selected)
                        .unwrap();
                    regret += (exact.value - estimate.value).max(0.);
                    optimal += u64::from(selected == *exact.greedy_action());
                    transitions += u64::from(search.work.transitions);
                }
                println!(
                    "signals={signals} trajectories={trajectories} strategy={label} optimal={optimal}/{SEEDS} mean_regret={:.5} transitions={transitions} wall_ms={:.2}",
                    regret / SEEDS as f64,
                    start.elapsed().as_secs_f64() * 1000.,
                );
            }
        }
    }
    Ok(())
}
