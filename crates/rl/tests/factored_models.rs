use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    models::factored::{DiscreteTransition, FactoredModel, FitConfig, FitLimits, Structure},
};

#[path = "support/factored_dynamics.rs"]
mod factored_dynamics;

fn config(pseudocount: f64) -> FitConfig {
    FitConfig {
        pseudocount,
        limits: FitLimits {
            max_parents: 2,
            max_candidates: 32,
            max_total_cells: 256,
            max_work: 8192,
        },
    }
}

#[test]
fn known_mixed_cardinality_tables_score_unobserved_rows_and_smooth_separately() -> Result<()> {
    let make = |state, action, next| DiscreteTransition {
        state,
        action,
        next,
    };
    let data = [
        make(&[0, 0], 0, &[0, 0]),
        make(&[0, 0], 0, &[0, 0]),
        make(&[0, 0], 0, &[0, 1]),
        make(&[1, 0], 1, &[0, 2]),
    ];
    let structure = [vec![], vec![0]];
    let fit = |alpha| {
        FactoredModel::fit(
            &[2, 3],
            2,
            &data,
            Structure::Known(&structure),
            config(alpha),
        )
    };
    let mle = fit(0.)?;
    let smoothed = fit(1.)?;
    assert_eq!(mle.predict_factors(&[0, 0], 0)?[1], &[2. / 3., 1. / 3., 0.]);
    assert_eq!(
        smoothed.predict_factors(&[0, 1], 0)?[1],
        &[0.5, 1. / 3., 1. / 6.]
    );
    assert_eq!(mle.predict_factors(&[1, 2], 0)?[1], &[1. / 3.; 3]);
    assert_eq!(mle.predict_factors(&[1, 0], 1)?[1], &[0., 0., 1.]);
    let expected = -2. * (2. * (2.0_f64 / 3.).ln() + (1.0_f64 / 3.).ln()) + 8. * 4.0_f64.ln();
    assert!((mle.bic_score(1).unwrap() - expected).abs() < 1e-12);
    assert_eq!(mle.bic_score(1), smoothed.bic_score(1));
    assert_eq!(
        smoothed.probability(&[0, 0], 0, &[0, 1])?,
        (4. / 5.) * (2. / 6.)
    );
    assert_eq!(
        mle.sample(&[1, 0], 1, &mut ChaCha8Rng::seed_from_u64(4))?,
        [0, 2]
    );
    Ok(())
}

#[test]
fn bic_recovers_a_two_parent_interaction_with_no_single_parent_signal() -> Result<()> {
    let mut transitions = Vec::new();
    for _ in 0..4 {
        for first in 0..2 {
            for second in 0..2 {
                for irrelevant in 0..2 {
                    for action in 0..2 {
                        transitions.push((
                            [first, second, irrelevant],
                            action,
                            [first ^ second ^ action, 0, 0],
                        ));
                    }
                }
            }
        }
    }
    let data = transitions
        .iter()
        .map(|(state, action, next)| DiscreteTransition {
            state,
            action: *action,
            next,
        })
        .collect::<Vec<_>>();
    let pools = [vec![2, 1, 0], vec![0, 1, 2], vec![0, 1, 2]];
    let fit = |alpha| {
        FactoredModel::fit(
            &[2; 3],
            2,
            &data,
            Structure::LearnBic(&pools),
            config(alpha),
        )
    };
    let model = fit(0.)?;
    assert_eq!(model.parents(0), Some([0, 1].as_slice()));
    assert_eq!(model.parents(1), Some([].as_slice()));
    assert_eq!(model.parents(2), Some([].as_slice()));
    assert_eq!(model.parents(0), fit(500.)?.parents(0));
    assert_eq!(model.predict_factors(&[1, 1, 1], 1)?[0], &[0., 1.]);
    Ok(())
}

#[test]
fn fitting_rejects_invalid_factual_rows_and_checks_all_cost_caps() {
    let good = DiscreteTransition {
        state: &[0],
        action: 0,
        next: &[1],
    };
    let bad = DiscreteTransition { next: &[2], ..good };
    let pools = [vec![0]];
    let fit = |data: &[DiscreteTransition<'_>], limits| {
        FactoredModel::fit(
            &[2],
            1,
            data,
            Structure::LearnBic(&pools),
            FitConfig {
                limits,
                ..config(0.)
            },
        )
    };
    let limits = config(0.).limits;
    assert!(fit(&[good], limits).is_ok());
    assert!(fit(&[bad], limits).is_err());
    assert!(
        fit(
            &[good],
            FitLimits {
                max_candidates: 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        fit(
            &[good],
            FitLimits {
                max_total_cells: 5,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        fit(
            &[good],
            FitLimits {
                max_work: 1,
                ..limits
            }
        )
        .is_err()
    );
    assert!(
        FactoredModel::fit(
            &[2],
            1,
            &[good],
            Structure::Known(&[vec![0, 0]]),
            config(0.)
        )
        .is_err()
    );
}

#[test]
fn oversized_count_tables_return_an_error_before_allocating() {
    let good = DiscreteTransition {
        state: &[0],
        action: 0,
        next: &[0],
    };
    let mut config = config(0.);
    config.limits.max_total_cells = usize::MAX;
    let result = FactoredModel::fit(
        &[usize::MAX],
        1,
        &[good],
        Structure::Known(&[vec![]]),
        config,
    );
    assert!(result.is_err());
}

#[test]
fn joint_sampling_draws_the_factors_independently() -> Result<()> {
    let data = [DiscreteTransition {
        state: &[0, 0],
        action: 0,
        next: &[0, 0],
    }];
    let fit = FactoredModel::fit(
        &[2; 2],
        2,
        &data,
        Structure::Known(&[vec![], vec![]]),
        config(0.),
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let mut counts = [0; 4];
    for _ in 0..4096 {
        let next = fit.sample(&[1, 1], 1, &mut rng)?;
        counts[2 * next[0] + next[1]] += 1;
    }
    assert!(
        counts.iter().all(|count| (900..=1148).contains(count)),
        "{counts:?}"
    );
    Ok(())
}

#[test]
fn learned_eight_factor_model_supports_exact_finite_horizon_decisions() -> Result<()> {
    use factored_dynamics::{ACTIONS, DenseLaw, HORIZON, STATES, VARIABLES, bits, encode};

    let dynamics = |state: usize, action: usize| {
        let state = bits(state);
        std::array::from_fn(|i| state[i] ^ state[(i + 1) % VARIABLES] ^ (action & 1))
    };
    let outcomes = (0..STATES)
        .flat_map(|state| {
            (0..ACTIONS).map(move |action| (bits(state), action, dynamics(state, action)))
        })
        .collect::<Vec<_>>();
    let data = outcomes
        .iter()
        .map(|(state, action, next)| DiscreteTransition {
            state,
            action: *action,
            next,
        })
        .collect::<Vec<_>>();
    let pools = vec![(0..VARIABLES).collect(); VARIABLES];
    let fitted = FactoredModel::fit(
        &[2; VARIABLES],
        ACTIONS,
        &data,
        Structure::LearnBic(&pools),
        factored_dynamics::config(0.),
    )?;
    for child in 0..VARIABLES {
        let mut expected = [child, (child + 1) % VARIABLES];
        expected.sort_unstable();
        assert_eq!(fitted.parents(child), Some(expected.as_slice()));
    }
    let fit = DenseLaw::from_factored(&fitted)?
        .prepare()?
        .backward_induction(HORIZON)?;
    assert_eq!(*fit.states(HORIZON.get()).unwrap()[0].greedy_action(), 1);
    let truth = DenseLaw::from_rows(|state, action| {
        let next = encode(&dynamics(state, action));
        let mut row = [0.; STATES];
        row[next] = 1.;
        Ok(row)
    })?
    .prepare()?;
    let evaluated = truth.evaluate_horizon_policy(HORIZON, |remaining, state, actions| {
        let picked = fit.states(remaining).unwrap()[*state].greedy_action();
        Ok(actions
            .iter()
            .map(|action| f64::from(action == picked))
            .collect())
    })?;
    assert_eq!(evaluated.states(HORIZON.get()).unwrap()[0].value, 4.);
    Ok(())
}
