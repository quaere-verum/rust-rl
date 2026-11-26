use rl::approx::ReturnWeightedRetrieval;

#[test]
fn retrieval_uses_factual_success_and_matches_semantics_in_current_menu() {
    let mut retrieval = ReturnWeightedRetrieval::new(300, 1).unwrap();
    let menu = vec![vec![2.], vec![0.]];
    retrieval.observe(&[(vec![0.], vec![2.])], 0.).unwrap();
    assert_eq!(
        retrieval.probabilities(&[0.], &menu).unwrap(),
        vec![0.5, 0.5]
    );
    assert!(retrieval.is_empty());

    retrieval.observe(&[(vec![0.], vec![0.])], 1.).unwrap();
    retrieval.observe(&[(vec![1.], vec![2.])], 1.).unwrap();
    let actual = retrieval.probabilities(&[0.], &menu).unwrap();
    let kernel = (-4.0_f64).exp();
    assert!((actual[0] - kernel / (1. + kernel)).abs() < 1e-12);
    assert!((actual[1] - 1. / (1. + kernel)).abs() < 1e-12);
    assert_eq!(
        retrieval.probabilities(&[0.], &[vec![2.]]).unwrap(),
        vec![1.]
    );
}

#[test]
fn retrieval_distributes_each_episodes_return_over_its_executed_decisions() {
    let mut retrieval = ReturnWeightedRetrieval::new(300, 7).unwrap();
    let long_episode = vec![(vec![0.], vec![0.]); 4];
    retrieval.observe(&long_episode, 1.).unwrap();
    retrieval.observe(&[(vec![0.], vec![1.])], 1.).unwrap();

    let actual = retrieval
        .probabilities(&[0.], &[vec![0.], vec![1.]])
        .unwrap();
    assert!((actual[0] - 0.5).abs() < 1e-12);
    assert!((actual[1] - 0.5).abs() < 1e-12);
}

#[test]
fn retrieval_fifo_and_neighbor_ties_survive_checkpoint_round_trip() {
    let mut retrieval = ReturnWeightedRetrieval::new(2, 1).unwrap();
    for selected in [0., 1., 2.] {
        retrieval
            .observe(&[(vec![0.], vec![selected])], 1.)
            .unwrap();
    }
    assert_eq!(retrieval.len(), 2);
    let restored: ReturnWeightedRetrieval =
        serde_json::from_slice(&serde_json::to_vec(&retrieval).unwrap()).unwrap();
    let menu = vec![vec![0.], vec![1.], vec![2.]];
    let p = retrieval.probabilities(&[0.], &menu).unwrap();
    assert!(p[1] > p[0] && p[1] > p[2]);
    assert_eq!(restored.probabilities(&[0.], &menu).unwrap(), p);
}

#[test]
fn retrieval_rejects_a_malformed_episode_before_modifying_the_bank() {
    let mut retrieval = ReturnWeightedRetrieval::new(2, 1).unwrap();
    retrieval.observe(&[(vec![0.], vec![0.])], 1.).unwrap();
    let malformed = [(vec![0.], vec![1.]), (vec![0.], vec![1., 0.])];
    assert!(retrieval.observe(&malformed, 1.).is_err());
    assert_eq!(retrieval.len(), 1);
    assert!(retrieval.probabilities(&[0., 1.], &[vec![1.]]).is_err());

    let mut serialized = serde_json::to_value(&retrieval).unwrap();
    serialized["capacity"] = 0.into();
    assert!(serde_json::from_value::<ReturnWeightedRetrieval>(serialized).is_err());
}
