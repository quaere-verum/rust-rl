use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::tabular::{EpisodeEnd, EpisodeStep, QTable, TdConfig},
    core::{checkpoint::Checkpoint, reward::Reward},
    train::checkpoint::{load_learner, save_learner},
};
use std::collections::BTreeSet;

type Table = QTable<u8, u8>;

const TD: TdConfig = TdConfig {
    gamma: 0.5,
    step_size: 1.0,
};

#[test]
fn unseen_pairs_use_configured_initial_value_without_creating_rows() -> Result<()> {
    let table = Table::new(4.5)?;
    let before = table.checkpoint()?.metadata;
    assert_eq!(table.value(&44, &9), 4.5);
    assert_eq!(table.visits(&44, &9), 0);
    assert_eq!(table.greedy_index(&44, &[9, 2, 4])?, 1);
    assert_eq!(table.checkpoint()?.metadata, before);
    assert!(Table::new(f64::NAN).is_err());
    Ok(())
}

#[test]
fn epsilon_greedy_respects_semantic_order_and_reports_its_executed_probability() -> Result<()> {
    let table = Table::new(0.0)?;
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let mut same_seed = rng.clone();
    let choices = [9, 2, 4];
    let shuffled = [2, 4, 9];
    let greedy = table.epsilon_greedy(&0, &choices, 0.0, &mut rng)?;
    assert_eq!(choices[greedy.index], 2);
    assert_eq!(greedy.probability, 1.0);
    let tiny = table.epsilon_greedy(&0, &choices, f64::from_bits(1), &mut rng)?;
    assert_eq!(choices[tiny.index], 2);
    assert_eq!(tiny.probability, 1.0);
    assert_eq!(rng.random::<u64>(), same_seed.random::<u64>());
    let mut explored = BTreeSet::new();
    for _ in 0..64 {
        let first = table.epsilon_greedy(&0, &choices, 1.0, &mut rng)?;
        let other = table.epsilon_greedy(&0, &shuffled, 1.0, &mut same_seed)?;
        let action = choices[first.index];
        assert_eq!(action, shuffled[other.index]);
        assert_eq!(first.probability, 1.0 / 3.0);
        explored.insert(action);
    }
    assert_eq!(explored, BTreeSet::from([2, 4, 9]));
    let mixed = table.epsilon_greedy(&0, &choices, 0.3, &mut rng)?;
    let expected_prob = if choices[mixed.index] == 2 { 0.8 } else { 0.1 };
    assert!((mixed.probability - expected_prob).abs() < 1e-14);
    assert_eq!(
        table.epsilon_greedy(&0, &[7], 0.3, &mut rng)?.probability,
        1.0
    );
    Ok(())
}

#[test]
fn q_learning_uses_only_legal_successors_and_discounts_over_physical_ticks() -> Result<()> {
    let mut table = Table::new(0.0)?;
    table.q_learning(&1, &4, &4.0.into(), None, TD)?;
    table.q_learning(&1, &9, &12.0.into(), None, TD)?;
    let rewards = Reward::from_primitives(vec![1.0, 2.0])?;
    let half_step = TdConfig {
        step_size: 0.5,
        ..TD
    };
    table.q_learning(&0, &2, &rewards, Some((&1, &[4, 5])), half_step)?;
    assert_eq!(table.value(&0, &2), 1.5); // Target 1 + .5*2 + .25*4.
    assert_eq!(table.visits(&0, &2), 1);
    table.q_learning(&0, &2, &rewards, None, half_step)?;
    assert_eq!(table.value(&0, &2), 1.75);
    assert_eq!(table.visits(&0, &2), 2);
    Ok(())
}

#[test]
fn sarsa_uses_the_selected_successor_even_when_it_is_suboptimal() -> Result<()> {
    let mut table = Table::new(0.0)?;
    table.sarsa(&1, &4, &4.0.into(), None, TD)?;
    table.sarsa(&1, &9, &12.0.into(), None, TD)?;
    let rewards = Reward::from_primitives(vec![1.0, 2.0])?;
    table.sarsa(&0, &2, &rewards, Some((&1, &4)), TD)?;
    assert_eq!(table.value(&0, &2), 3.0);
    table.sarsa(&0, &2, &rewards, None, TD)?;
    assert_eq!(table.value(&0, &2), 2.0);
    Ok(())
}

#[test]
fn q_learning_converges_on_a_known_discounted_cycle() -> Result<()> {
    // A → B gives 1 and B → A gives 2. The exact values are 8/3 and 10/3.
    let mut table = Table::new(0.0)?;
    for _ in 0..32 {
        table.q_learning(&0, &7, &1.0.into(), Some((&1, &[7])), TD)?;
        table.q_learning(&1, &7, &2.0.into(), Some((&0, &[7])), TD)?;
    }
    assert!((table.value(&0, &7) - 8.0 / 3.0).abs() < 1e-12);
    assert!((table.value(&1, &7) - 10.0 / 3.0).abs() < 1e-12);
    Ok(())
}

fn step(action: u8, reward: impl Into<Reward>) -> EpisodeStep<u8, u8> {
    EpisodeStep {
        state: 0,
        action,
        reward: reward.into(),
    }
}

#[test]
fn monte_carlo_averages_the_first_visit_return_of_each_pair() -> Result<()> {
    let mut table = Table::new(19.0)?;
    let episode = [
        step(2, 1.0),
        step(4, Reward::from_primitives(vec![2.0, 4.0])?),
        step(2, 8.0),
    ];
    table.monte_carlo_first_visit(&episode, EpisodeEnd::Terminated, 0.5)?;
    assert_eq!(table.value(&0, &2), 4.0);
    assert_eq!(table.value(&0, &4), 6.0);
    assert_eq!(table.visits(&0, &2), 1);
    table.monte_carlo_first_visit(&[step(2, 2.0)], EpisodeEnd::Terminated, 0.5)?;
    assert_eq!(table.value(&0, &2), 3.0);
    assert_eq!(table.visits(&0, &2), 2);
    assert_eq!(table.greedy_index(&0, &[2, 4])?, 1);
    Ok(())
}

#[test]
fn monte_carlo_uses_explicit_truncation_bootstrap() -> Result<()> {
    let mut table = Table::new(0.0)?;
    let trace = Reward::from_primitives(vec![1.0, 2.0])?;
    table.monte_carlo_first_visit(
        &[step(2, trace.clone())],
        EpisodeEnd::Truncated { bootstrap: 6.0 },
        0.5,
    )?;
    table.monte_carlo_first_visit(&[step(4, trace)], EpisodeEnd::Terminated, 0.5)?;
    assert_eq!(table.value(&0, &2), 3.5);
    assert_eq!(table.value(&0, &4), 2.0);
    Ok(())
}

#[test]
fn invalid_data_does_not_change_values_or_counts() -> Result<()> {
    let mut table = Table::new(0.0)?;
    table.q_learning(&0, &2, &3.0.into(), None, TD)?;
    let before = table.checkpoint()?.metadata;
    assert!(
        table
            .q_learning(&0, &2, &8.0.into(), Some((&1, &[])), TD)
            .is_err()
    );
    assert!(
        table
            .q_learning(&0, &2, &8.0.into(), Some((&1, &[4, 4])), TD)
            .is_err()
    );
    assert!(
        table
            .sarsa(
                &0,
                &2,
                &8.0.into(),
                None,
                TdConfig {
                    step_size: f64::NAN,
                    ..TD
                }
            )
            .is_err()
    );
    assert!(
        table
            .monte_carlo_first_visit(
                &[step(2, f32::NAN), step(4, 8.0)],
                EpisodeEnd::Terminated,
                0.5
            )
            .is_err()
    );
    assert!(
        table
            .monte_carlo_first_visit(&[], EpisodeEnd::Terminated, 0.5)
            .is_err()
    );
    let mut rng = ChaCha8Rng::seed_from_u64(16);
    let mut same_seed = rng.clone();
    assert!(table.epsilon_greedy(&0, &[2, 4, 2], 0.5, &mut rng).is_err());
    assert!(
        table
            .epsilon_greedy(&0, &[2, 4], f64::INFINITY, &mut rng)
            .is_err()
    );
    assert_eq!(rng.random::<u64>(), same_seed.random::<u64>());
    assert_eq!(table.checkpoint()?.metadata, before);
    Ok(())
}

#[test]
fn checkpoints_round_trip_and_reject_duplicates_without_mutation() -> Result<()> {
    let mut table = Table::new(4.0)?;
    table.q_learning(&4, &2, &3.0.into(), None, TD)?;
    table.q_learning(&1, &7, &8.0.into(), None, TD)?;
    let mut reversed = Table::new(4.0)?;
    reversed.q_learning(&1, &7, &8.0.into(), None, TD)?;
    reversed.q_learning(&4, &2, &3.0.into(), None, TD)?;
    let mut corrupt = table.checkpoint()?;
    let before = corrupt.metadata.clone();
    assert_eq!(before, reversed.checkpoint()?.metadata);
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("learner.json");
    save_learner(&path, &table)?;
    let mut restored = Table::new(0.0)?;
    load_learner(&path, &mut restored)?;
    assert_eq!(restored.value(&4, &2), 3.0);
    assert_eq!(restored.visits(&1, &7), 1);
    assert_eq!(restored.value(&9, &6), 4.0);
    let rows = corrupt.metadata["rows"].as_array_mut().unwrap();
    rows.push(rows[0].clone());
    assert!(restored.restore(&corrupt).is_err());
    corrupt.metadata = before.clone();
    corrupt.metadata["rows"][0]["value"] = "NaN".into();
    assert!(restored.restore(&corrupt).is_err());
    assert_eq!(restored.checkpoint()?.metadata, before);
    Ok(())
}

#[test]
fn monte_carlo_checks_every_new_visit_before_committing_any_rows() -> Result<()> {
    let mut table = Table::new(0.0)?;
    table.q_learning(&0, &4, &3.0.into(), None, TD)?;
    let mut record = table.checkpoint()?;
    record.metadata["rows"][0]["visits"] = u64::MAX.into();
    table.restore(&record)?;
    assert!(
        table
            .monte_carlo_first_visit(&[step(2, 5.0), step(4, 8.0)], EpisodeEnd::Terminated, 0.5)
            .is_err()
    );
    assert_eq!(table.visits(&0, &2), 0);
    assert_eq!(table.value(&0, &4), 3.0);
    assert_eq!(table.visits(&0, &4), u64::MAX);
    Ok(())
}
