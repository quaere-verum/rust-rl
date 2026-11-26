#[path = "support/controlled/reference.rs"]
mod reference;
fn close(got: f64, want: f64) {
    assert!((got - want).abs() < 1e-10, "got={got} want={want}");
}
#[test]
fn coincident_support_is_one_uniform() {
    for context in [-1, 1] {
        for tau in [0., 0.05, 0.5, 0.95, 1.] {
            close(reference::quantile(context, 0., tau), 0.5 + tau);
        }
        let (mean, variance) = reference::moments(context, 0.);
        close(mean, 1.);
        close(variance, 1. / 12.);
        close(reference::cvar(context, 0., 0.1), 0.55);
    }
}
#[test]
fn separated_support_generalized_inverse_and_tail_integral() {
    let probabilities = [
        0.05, 0.15, 0.2, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
    ];
    for (context, want, mean, variance, tail) in [
        (
            1,
            [
                -1.25, -0.75, -0.5, 1.5625, 1.6875, 1.8125, 1.9375, 2.0625, 2.1875, 2.3125, 2.4375,
            ],
            1.4,
            1.44 + 1. / 12.,
            -0.49375,
        ),
        (
            -1,
            [
                -1.25, -0.75, -0.5, 0.8125, 0.9375, 1.0625, 1.1875, 1.3125, 1.4375, 1.5625, 1.6875,
            ],
            0.8,
            0.81 + 1. / 12.,
            -0.64375,
        ),
    ] {
        for (tau, q) in probabilities.into_iter().zip(want) {
            close(reference::quantile(context, 0.5, tau), q);
        }
        let actual = reference::moments(context, 0.5);
        close(actual.0, mean);
        close(actual.1, variance);
        close(reference::cvar(context, 0.5, 0.1), -1.25);
        close(reference::cvar(context, 0.5, 0.2), -1.);
        close(reference::cvar(context, 0.5, 0.25), tail);
    }
}
#[test]
fn overlapping_support_has_three_density_segments() {
    let want = [
        0.35, 0.73, 0.83, 0.93, 1.03, 1.1375, 1.2625, 1.3875, 1.5125, 1.6375,
    ];
    for (j, q) in want.into_iter().enumerate() {
        close(reference::quantile(1, 0.1, (j as f64 + 0.5) / 10.), q);
    }
    let actual = reference::moments(1, 0.1);
    close(actual.0, 1.08);
    close(actual.1, 0.0576 + 1. / 12.);
    close(reference::cvar(1, 0.1, 0.1), 0.35);
    close(reference::cvar(1, 0.1, 0.2), 0.536);
}
