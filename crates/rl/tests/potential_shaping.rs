use rl::{
    Result,
    algorithms::tabular::{QTable, TdConfig},
    core::reward::{PotentialValues, Reward},
};

#[path = "support/investment.rs"]
mod investment;

#[test]
fn shaping_uses_the_discounted_trace_and_terminal_potential_is_zero() -> Result<()> {
    let trace = Reward::from_primitives(vec![2., 6.])?;
    let (reward, discount) = trace.discounted_f64(0.5)?;
    let live = PotentialValues {
        current: 3.,
        next: Some(8.),
    };
    let terminal = PotentialValues {
        current: 3.,
        next: None,
    };
    assert_eq!(live.shape(reward, discount)?, 4.);
    assert_eq!(terminal.shape(reward, discount)?, 2.);
    assert_eq!(trace.primitives(), &[2., 6.]);
    assert!(live.shape(reward, f64::NAN).is_err());
    assert!(
        PotentialValues {
            current: 0.,
            next: Some(f64::INFINITY)
        }
        .shape(0., 0.)
        .is_err()
    );
    Ok(())
}

#[test]
fn shaped_tabular_updates_use_the_same_continuation_as_the_learning_target() -> Result<()> {
    let mut table = QTable::new(16.)?;
    let config = TdConfig {
        gamma: 0.5,
        step_size: 1.,
    };
    let trace = Reward::from_primitives(vec![2., 6.])?;
    let live = PotentialValues {
        current: 3.,
        next: Some(8.),
    };
    let terminal = PotentialValues {
        current: 3.,
        next: None,
    };
    let legal = [0];
    let next = Some((&1, legal.as_slice()));
    table.q_learning_with_potential(&0, &0, &trace, next, live, config)?;
    assert_eq!(table.value(&0, &0), 8.);
    assert!(
        table
            .q_learning_with_potential(&0, &0, &trace, next, terminal, config)
            .is_err()
    );
    assert_eq!(table.visits(&0, &0), 1);
    table.q_learning_with_potential(&0, &1, &trace, None, terminal, config)?;
    assert_eq!(table.value(&0, &1), 2.);
    Ok(())
}

#[test]
fn investment_shaping_preserves_all_optimal_actions_and_income_bonus_changes_the_policy()
-> Result<()> {
    use investment::{Action, HORIZON, INITIAL, Objective};

    let natural = Objective::Natural.prepare()?;
    let optimal = natural.backward_induction(HORIZON)?;
    let shaped = Objective::Shaped.prepare()?.backward_induction(HORIZON)?;
    let proxy = Objective::IncomeBonus
        .prepare()?
        .backward_induction(HORIZON)?;
    for (raw, shaped) in optimal
        .states(HORIZON.get())
        .unwrap()
        .iter()
        .zip(shaped.states(HORIZON.get()).unwrap())
    {
        assert_eq!(raw.context, shaped.context);
        assert_eq!(raw.greedy_action(), shaped.greedy_action());
        for (raw_action, shaped_action) in raw.action_values().iter().zip(shaped.action_values()) {
            assert_eq!(raw_action.action, shaped_action.action);
            assert_eq!(
                raw_action.value,
                shaped_action.value + raw.context.potential()
            );
        }
    }
    let initial = optimal
        .states(HORIZON.get())
        .unwrap()
        .iter()
        .position(|row| row.context == INITIAL)
        .unwrap();
    assert_eq!(optimal.states(HORIZON.get()).unwrap()[initial].value, 9.);
    assert_eq!(shaped.states(HORIZON.get()).unwrap()[initial].value, 4.);
    assert_eq!(proxy.states(HORIZON.get()).unwrap()[initial].value, 22.);
    let achieved = natural.evaluate_horizon_policy(HORIZON, |remaining, state, actions| {
        let stage = proxy.states(remaining).unwrap();
        let picked = stage
            .iter()
            .find(|row| row.context == *state)
            .unwrap()
            .greedy_action();
        Ok(actions
            .iter()
            .map(|action| f64::from(action == picked))
            .collect())
    })?;
    assert_eq!(achieved.states(HORIZON.get()).unwrap()[initial].value, 8.);
    let never_invest = natural.evaluate_horizon_policy(HORIZON, |_, _, actions| {
        Ok(actions
            .iter()
            .map(|action| f64::from(*action == Action::Research))
            .collect())
    })?;
    assert_eq!(
        never_invest.states(HORIZON.get()).unwrap()[initial].value,
        5.
    );
    Ok(())
}
