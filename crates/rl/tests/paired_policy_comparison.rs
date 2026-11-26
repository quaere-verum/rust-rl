use rl::train::policy_comparison::{
    ReturnBounds, compare_paired_returns, paired_return_lower_bound,
};

#[test]
fn each_matched_group_gets_one_vote_in_the_summary_and_bound() -> rl::Result<()> {
    let pairs = [(0., 1.), (0.25, 1.), (0.5, 0.5), (1., 0.75)];
    let summary = compare_paired_returns(&pairs)?;
    assert_eq!(summary.pairs, 4);
    assert_eq!((summary.wins, summary.losses, summary.ties), (2, 1, 1));
    assert!((summary.incumbent_mean - 0.4375).abs() < 1e-15);
    assert!((summary.challenger_mean - 0.8125).abs() < 1e-15);
    assert!((summary.mean_difference - 0.375).abs() < 1e-15);
    assert!((summary.difference_standard_error.unwrap() - (1.0625_f64 / 12.).sqrt()).abs() < 1e-15);
    let bound = paired_return_lower_bound(
        &pairs,
        ReturnBounds {
            minimum: 0.,
            maximum: 1.,
        },
        0.05,
    )?;
    assert!((bound - (0.375 - (-2. * 0.05_f64.ln() / 4.).sqrt())).abs() < 1e-15);
    Ok(())
}

#[test]
fn finite_differences_and_bounds_survive_extreme_return_scales() -> rl::Result<()> {
    let tiny = f64::MIN_POSITIVE;
    let one = compare_paired_returns(&[(0., tiny)])?;
    assert_eq!(one.mean_difference, tiny);
    assert_eq!(one.difference_standard_deviation, None);
    let large = 1.0e308;
    let result = paired_return_lower_bound(
        &[(0., large)],
        ReturnBounds {
            minimum: 0.,
            maximum: large,
        },
        0.1,
    )?;
    assert!((result / large - (-1.1459660262893472)).abs() < 1e-14);
    Ok(())
}

#[test]
fn invalid_pairs_and_unrepresentable_bounds_fail() {
    let bounds = ReturnBounds {
        minimum: 0.,
        maximum: 1.,
    };
    assert!(compare_paired_returns(&[]).is_err());
    assert!(compare_paired_returns(&[(f64::NAN, 1.)]).is_err());
    assert!(compare_paired_returns(&[(-f64::MAX, f64::MAX)]).is_err());
    assert!(paired_return_lower_bound(&[(0., 2.)], bounds, 0.1).is_err());
    assert!(paired_return_lower_bound(&[(0., 1.)], bounds, 1.).is_err());
    assert!(
        paired_return_lower_bound(
            &[(0., 1.)],
            ReturnBounds {
                minimum: -f64::MAX,
                maximum: f64::MAX,
            },
            0.1
        )
        .is_err()
    );
}
