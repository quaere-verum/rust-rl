use rl::data::{gae::gae, returns::discounted_returns};

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} != {expected:?}");
    }
}

#[test]
fn terminal_truncation_and_collection_cut_have_distinct_semantics() {
    // The large reward starts a new episode and must not flow through truncation.
    let r = [1., 100., 2.];
    let term = [false, true, false];
    let trunc = [true, false, false];
    close(
        &discounted_returns(&r, &[10., 999., 4.], &term, &trunc, 0.5).unwrap(),
        &[6., 100., 4.],
    );
    let (a, targets) = gae(
        &r,
        &[1., 20., 1.],
        &[10., 999., 4.],
        &term,
        &trunc,
        0.5,
        0.9,
    )
    .unwrap();
    close(&a, &[5., 80., 3.]);
    close(&targets, &[6., 100., 4.]);
}

#[test]
fn gae_matches_two_step_hand_calculation() {
    let (a, _) = gae(
        &[1., 2.],
        &[0., 0.],
        &[0., 0.],
        &[false, true],
        &[false, false],
        0.9,
        0.95,
    )
    .unwrap();
    close(&a, &[2.71, 2.]);
}

#[test]
fn trajectory_math_rejects_mismatched_lengths_and_bad_discounts() {
    assert!(discounted_returns(&[1.], &[], &[true], &[false], 0.9).is_err());
    assert!(gae(&[1.], &[0.], &[0.], &[true], &[false], f32::NAN, 1.).is_err());
}
