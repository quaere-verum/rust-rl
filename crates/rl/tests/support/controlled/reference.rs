//! Exact two-uniform return law; independent literals live in the tests.
fn supports(context: i8, exposure: f64) -> [f64; 4] {
    assert!(matches!(context, -1 | 1) && (0. ..=1.).contains(&exposure));
    let low = 1. - 4. * exposure;
    let high = 1. + if context == 1 { 2. } else { 0.5 } * exposure;
    [low - 0.5, low + 0.5, high - 0.5, high + 0.5]
}
pub fn cdf(context: i8, exposure: f64, x: f64) -> f64 {
    let [low, _, high, _] = supports(context, exposure);
    0.2 * (x - low).clamp(0., 1.) + 0.8 * (x - high).clamp(0., 1.)
}
fn endpoints(context: i8, exposure: f64) -> Vec<f64> {
    let mut points = supports(context, exposure).to_vec();
    points.sort_by(f64::total_cmp);
    points.dedup();
    points
}
pub fn quantile(context: i8, exposure: f64, tau: f64) -> f64 {
    assert!((0. ..=1.).contains(&tau));
    let points = endpoints(context, exposure);
    if tau == 0. {
        return points[0];
    }
    for pair in points.windows(2) {
        let f0 = cdf(context, exposure, pair[0]);
        let f1 = cdf(context, exposure, pair[1]);
        // The leftmost inverse at a flat CDF gap is the previous segment end.
        if tau <= f1 && f1 > f0 {
            return pair[0] + (pair[1] - pair[0]) * (tau - f0) / (f1 - f0);
        }
    }
    *points.last().unwrap()
}
pub fn moments(context: i8, exposure: f64) -> (f64, f64) {
    let b = if context == 1 { 2. } else { 0.5 };
    let _ = supports(context, exposure);
    (
        1. + (0.8 * b - 0.8) * exposure,
        0.16 * (4. + b) * (4. + b) * exposure * exposure + 1. / 12.,
    )
}
pub fn cvar(context: i8, exposure: f64, alpha: f64) -> f64 {
    assert!(alpha > 0. && alpha <= 1.);
    let mut integral = 0.;
    for pair in endpoints(context, exposure).windows(2) {
        let f0 = cdf(context, exposure, pair[0]);
        let f1 = cdf(context, exposure, pair[1]);
        if f0 >= alpha {
            break;
        }
        if f1 > f0 {
            let mass = alpha.min(f1) - f0;
            let end = pair[0] + (pair[1] - pair[0]) * mass / (f1 - f0);
            integral += mass * (pair[0] + end) / 2.;
        }
    }
    integral / alpha
}
