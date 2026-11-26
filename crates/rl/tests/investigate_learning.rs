#![cfg(feature = "neural")]

//! Controlled study with one terminal Bernoulli reward and unit discount.
//! Every optimizer step uses a fresh collection of complete factual episodes.
//! Report the stochastic actor's exact environmental return without a collection floor.
#[path = "support/investigate_learning.rs"]
mod experiment;
#[path = "support/investigate_or_commit.rs"]
mod investigate_or_commit;

use experiment::{EPISODES_PER_UPDATE, Problem, Run, SUCCESS, run_mpo, run_reinforce};
use rl::{
    Result,
    algorithms::recurrent_mpo::{MpoBackup, RecurrentMpoConfig},
    nn::critics::CriticInitialization,
};

const SEEDS: [u64; 3] = [11, 27, 43];
const UPDATES: usize = 48;

#[test]
fn factual_batches_drive_both_learners_and_exact_policy_evaluation() -> Result<()> {
    let problem = Problem::new(2);
    let mpo = run_mpo(
        &problem,
        11,
        CriticInitialization::Constant { value: 0.6 },
        MpoBackup::PolicyEvaluation,
        1,
    )?;
    let reinforce = run_reinforce(&problem, 11, 0.05, 0.1, 1)?;
    for run in [mpo, reinforce] {
        assert_eq!(run.episodes, EPISODES_PER_UPDATE);
        assert!(run.transitions > run.episodes && run.transitions <= 2 * run.episodes);
        assert!(run.final_policy.investigate > 0. && run.final_policy.investigate < 1.);
        assert!(run.final_policy.competence > 0. && run.final_policy.competence < 1.);
        let expected = 0.6 * (1. - run.final_policy.investigate)
            + (0.5 + 0.4 * run.final_policy.competence) * run.final_policy.investigate;
        assert!((run.final_policy.value - expected).abs() < 1e-12);
    }
    Ok(())
}

fn announce(algorithm: &str) {
    let episodes = UPDATES * EPISODES_PER_UPDATE;
    eprintln!(
        "{algorithm} seeds={SEEDS:?} updates={UPDATES} episodes_per_run={episodes} transition_cap={} gamma=1 success_exact_value={SUCCESS}",
        2 * episodes,
    );
}

fn compare(label: &str, signals: u16, simulate: impl Fn(u64) -> Result<Run>) -> Result<()> {
    let mut totals = [0.; 3];
    let mut reached = 0;
    eprintln!("start k={signals} method={label}");
    for seed in SEEDS {
        let run = simulate(seed)?;
        let after = run.final_policy;
        let before = run.initial_policy;
        let observed_investigate = (run.transitions - run.episodes) as f64 / run.episodes as f64;
        totals[0] += after.investigate;
        totals[1] += after.competence;
        totals[2] += after.value;
        reached += usize::from(run.first_success.is_some());
        assert_eq!(run.episodes, UPDATES * EPISODES_PER_UPDATE);
        assert!((run.episodes..=2 * run.episodes).contains(&run.transitions));
        println!(
            "k={signals} method={label} seed={seed} initial_investigate={:.4} initial_competence={:.4} initial_value={:.4} investigate={:.4} competence={:.4} expected_value={:.4} collection_investigate={observed_investigate:.4} first_threshold_episode={} episodes={} transitions={} wall_ms={:.1}",
            before.investigate,
            before.competence,
            before.value,
            after.investigate,
            after.competence,
            after.value,
            run.first_success
                .map_or_else(|| "none".to_owned(), |ep| ep.to_string()),
            run.episodes,
            run.transitions,
            run.wall.as_secs_f64() * 1000.,
        );
    }
    let [investigate, competence, expected_value] = totals.map(|total| total / SEEDS.len() as f64);
    println!(
        "summary k={signals} method={label} reached={reached}/{} investigate={investigate:.4} competence={competence:.4} expected_value={expected_value:.4}",
        SEEDS.len(),
    );
    Ok(())
}

#[test]
#[ignore = "bounded multi-seed learning study"]
fn mpo_backup_initialization_and_current_policy_study() -> Result<()> {
    announce("MPO");
    let default_backup = RecurrentMpoConfig::default().backup;
    assert_eq!(default_backup, MpoBackup::PolicyEvaluation);
    eprintln!(
        "MPO default already uses the frozen current actor. Existing and bootstrap-only share a row. Optimistic initialization-only and combined also share a row. Actor lr=.03 critic lr=.02 target tau=.25; no tuning sweep."
    );
    let optimistic = CriticInitialization::Constant { value: 0.6 };
    for signals in [2, 4, 8, 16, 32, 64] {
        let problem = Problem::new(signals);
        for (label, initialization, backup) in [
            (
                "mpo_existing_or_bootstrap_only",
                CriticInitialization::Existing,
                default_backup,
            ),
            (
                "mpo_neutral_current",
                CriticInitialization::Constant { value: 0. },
                default_backup,
            ),
            ("mpo_optimistic_or_combined", optimistic, default_backup),
            (
                "mpo_optimistic_maximum",
                optimistic,
                MpoBackup::MaximumOptimality,
            ),
            (
                "mpo_optimistic_double_q",
                optimistic,
                MpoBackup::DoubleQOptimality,
            ),
        ] {
            compare(label, signals, |seed| {
                run_mpo(&problem, seed, initialization, backup, UPDATES)
            })?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "bounded multi-seed learning study"]
fn reinforce_entropy_and_root_exploration_study() -> Result<()> {
    announce("REINFORCE");
    eprintln!(
        "REINFORCE actor lr=.06, baseline lr=.05. Actor entropy is averaged over factual decisions under the unmodified policy's probabilities. The root floor is part of the collection law. No tuning sweep."
    );
    for signals in [2, 4, 8, 10, 16] {
        let problem = Problem::new(signals);
        for (label, entropy, epsilon) in [
            ("reinforce_baseline", 0., 0.),
            ("reinforce_entropy_0.01", 0.01, 0.),
            ("reinforce_entropy_0.05", 0.05, 0.),
            ("reinforce_root_floor_0.10", 0., 0.1),
        ] {
            compare(label, signals, |seed| {
                run_reinforce(&problem, seed, entropy, epsilon, UPDATES)
            })?;
        }
    }
    Ok(())
}
