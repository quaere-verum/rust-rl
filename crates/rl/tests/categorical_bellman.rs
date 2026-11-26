use rl::objectives::categorical::CategoricalSupport;

#[test]
fn projection_conserves_mass_and_in_support_mean() {
    let support = CategoricalSupport::uniform(-2., 2., 5).unwrap();
    let source = [0.1, 0.2, 0.4, 0.2, 0.1];
    let result = support.project(0.25, 0.5, &source).unwrap();
    for (mass, expected) in result
        .probabilities
        .iter()
        .zip([0., 0.125, 0.525, 0.325, 0.025])
    {
        assert!((mass - expected).abs() < 1e-12, "{mass} != {expected}");
    }
    assert!((result.probabilities.iter().sum::<f64>() - 1.).abs() < 1e-12);
    let mean: f64 = result
        .probabilities
        .iter()
        .zip(support.atoms())
        .map(|(p, &z)| p * f64::from(z))
        .sum();
    assert!((mean - 0.25).abs() < 1e-12);
    assert_eq!(result.clipped_below, 0.);
    assert_eq!(result.clipped_above, 0.);
}

#[test]
fn projection_handles_terminal_rewards_exact_hits_and_clipping() {
    let support = CategoricalSupport::new(vec![-3., -1., 2.]).unwrap();
    let terminal = support.project(0.5, 0., &[]).unwrap();
    assert_eq!(terminal.probabilities, [0., 0.5, 0.5]);
    let exact = support.project(-1., 0., &[]).unwrap();
    assert_eq!(exact.probabilities, [0., 1., 0.]);
    let high = support.project(3., 0., &[]).unwrap();
    assert_eq!(high.probabilities, [0., 0., 1.]);
    assert_eq!((high.clipped_below, high.clipped_above), (0., 1.));
    let low = support.project(-1., 1., &[0.5, 0.5, 0.]).unwrap();
    assert_eq!(low.probabilities, [0.75, 0.25, 0.]);
    assert_eq!((low.clipped_below, low.clipped_above), (0.5, 0.));
}

#[test]
fn malformed_support_and_continuation_are_rejected() {
    assert!(CategoricalSupport::new(vec![1., 1.]).is_err());
    assert!(CategoricalSupport::new(vec![0., f32::INFINITY]).is_err());
    assert!(CategoricalSupport::uniform(1., f32::from_bits(1_f32.to_bits() + 1), 3).is_err());
    let support = CategoricalSupport::uniform(-1., 1., 2).unwrap();
    assert!(support.project(0., -0.5, &[0.5, 0.5]).is_err());
    assert!(support.project(0., 0.5, &[0.25, 0.25]).is_err());
    assert!(support.project(0., 0.5, &[-0.5, 1.5]).is_err());
    assert!(support.project(0., 0.5, &[0.5, f64::NAN]).is_err());
    assert!(support.project(0., 0.5, &[]).is_err());
}
