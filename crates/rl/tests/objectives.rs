#[cfg(feature = "neural")]
use burn::tensor::{Device, Tensor};
use rl::objectives::bellman::moment_targets;
#[cfg(feature = "neural")]
use rl::{
    nn::critics::quantile::QuantileGrid,
    objectives::{
        quantile::{QuantileLoss, quantile_loss},
        risk::{Risk, RiskObjective},
    },
};
#[test]
fn raw_second_moment_contains_the_cross_term() {
    let (a, b) = moment_targets(&[2.], &[0.5], &[3.], &[13.]).unwrap();
    assert_eq!(a, [3.5]);
    assert_eq!(b, [13.25]);
}
#[test]
#[cfg(feature = "neural")]
fn quantile_absolute_loss_has_the_correct_value_and_gradient() {
    let d = Device::flex().autodiff();
    let p = Tensor::<2>::from_data([[0.0_f32]], &d).require_grad();
    let y = Tensor::<2>::from_data([[2.0_f32]], &d);
    let loss = quantile_loss(
        p.clone(),
        y,
        &QuantileGrid::uniform(1).unwrap(),
        &QuantileLoss::Absolute,
    )
    .unwrap();
    assert!((loss.clone().into_scalar::<f32>() - 1.).abs() < 1e-6);
    let grads = loss.backward();
    assert_eq!(p.grad(&grads).unwrap().into_scalar::<f32>(), -0.5);
}
#[test]
#[cfg(feature = "neural")]
fn cvar_accounts_for_a_partial_quantile_cell() {
    let q = Tensor::<2>::from_data([[-4.0_f32, 4.]], &Device::flex().autodiff());
    let risk = Risk::Cvar { alpha: 0.75 };
    let value = risk
        .quantiles(q, &QuantileGrid::uniform(2).unwrap())
        .unwrap()
        .into_scalar::<f32>();
    assert!((value + 4. / 3.).abs() < 1e-6);
}

#[test]
#[cfg(feature = "neural")]
fn generalized_quantile_gradient_matches_independent_normal_integral() {
    let scale = 0.7_f32;
    let u = 1.3_f32;
    let p = Tensor::<2>::from_data([[0.0_f32]], &Device::flex().autodiff()).require_grad();
    let target = Tensor::<2>::from_data([[u]], &Device::flex().autodiff());
    let kind = QuantileLoss::Generalized {
        scale,
        ema: 0.9,
        minimum: 1e-4,
    };
    let loss = quantile_loss(p.clone(), target, &QuantileGrid::uniform(1).unwrap(), &kind).unwrap();
    let z = u as f64 / (scale as f64 * 2_f64.sqrt());
    let expected = 0.5
        * (u as f64 * libm::erf(z)
            + scale as f64 * (2. / std::f64::consts::PI).sqrt() * ((-z * z).exp() - 1.));
    assert!((loss.clone().into_scalar::<f32>() as f64 - expected).abs() < 1e-6);
    let gradient = p.grad(&loss.backward()).unwrap().into_scalar::<f32>();
    assert!((gradient as f64 + 0.5 * libm::erf(z)).abs() < 1e-6);
    let reference = |prediction: f64| {
        let residual = u as f64 - prediction;
        let z = residual / (scale as f64 * 2_f64.sqrt());
        0.5 * (residual * libm::erf(z)
            + scale as f64 * (2. / std::f64::consts::PI).sqrt() * ((-z * z).exp() - 1.))
    };
    assert!((gradient as f64 - (reference(1e-5) - reference(-1e-5)) / 2e-5).abs() < 1e-6);
}

#[test]
#[cfg(feature = "neural")]
fn adaptive_scale_uses_the_interpolated_seventieth_percentile() {
    let mut loss = QuantileLoss::Generalized {
        scale: 1.,
        ema: 0.,
        minimum: 0.001,
    };
    loss.adapt(vec![-4., 1., -2., 3.]).unwrap();
    let QuantileLoss::Generalized { scale, .. } = loss else {
        unreachable!()
    };
    assert!((scale - 3.1).abs() < 1e-6);
}

#[cfg(feature = "neural")]
fn adapted_scale(residuals: Vec<f32>, ema: f32, minimum: f32) -> f32 {
    let mut loss = QuantileLoss::Generalized {
        scale: 4.,
        ema,
        minimum,
    };
    loss.adapt(residuals).unwrap();
    let QuantileLoss::Generalized { scale, .. } = loss else {
        unreachable!()
    };
    scale
}

#[test]
#[cfg(feature = "neural")]
fn adaptive_percentile_handles_interpolation_singletons_duplicates_and_floor() {
    assert!((adapted_scale(vec![0., 1., 2., 3.], 0., 0.001) - 2.1).abs() < 1e-6);
    assert_eq!(adapted_scale(vec![-2.], 0., 0.001), 2.);
    assert_eq!(adapted_scale(vec![2., -2., 2., -2.], 0., 0.001), 2.);
    assert_eq!(adapted_scale(vec![0.; 4], 0., 0.5), 0.5);
    assert_eq!(adapted_scale(vec![0.; 4], 0.5, 0.001), 2.);
}

#[test]
#[cfg(feature = "neural")]
fn adaptive_percentile_rejects_nonfinite_or_empty_input_without_changing_scale() {
    for residuals in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
    ] {
        let mut loss = QuantileLoss::Generalized {
            scale: 3.,
            ema: 0.,
            minimum: 0.001,
        };
        assert!(loss.adapt(residuals).is_err());
        let QuantileLoss::Generalized { scale, .. } = loss else {
            unreachable!()
        };
        assert_eq!(scale, 3.);
    }
}

#[test]
#[cfg(feature = "neural")]
fn adaptive_percentile_matches_independent_sorted_oracle() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(71);
    for len in [1, 2, 3, 4, 7, 11, 16, 31, 100, 1024] {
        for _ in 0..8 {
            let residuals: Vec<f32> = (0..len).map(|_| rng.random_range(-10.0..10.0)).collect();
            let mut sorted: Vec<f64> = residuals.iter().map(|x| f64::from(x.abs())).collect();
            sorted.sort_by(f64::total_cmp);
            let position = 0.7 * (len - 1) as f64;
            let lower = position.floor() as usize;
            let fraction = position.fract();
            let oracle =
                sorted[lower] * (1. - fraction) + sorted[position.ceil() as usize] * fraction;
            let actual = adapted_scale(residuals, 0., 0.000001);
            assert!(
                (f64::from(actual) - oracle).abs() < 2e-6,
                "len={len}: {actual} vs {oracle}"
            );
        }
    }
}
