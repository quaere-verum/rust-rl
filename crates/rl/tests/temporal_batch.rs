#![cfg(feature = "neural")]

use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
        stateful::HistorySpec,
    },
    data::temporal::{
        BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep,
    },
    nn::temporal_batch::encode_history_batch,
};

fn step(action: Action, reward: f32, terminated: bool, truncated: bool) -> TemporalStep {
    TemporalStep {
        action,
        reward: reward.into(),
        terminated,
        truncated,
        behavior: BehaviorMetadata {
            revision: 1,
            log_prob: None,
        },
    }
}

#[test]
fn canonical_history_channels_use_the_previous_transition_and_zero_initial_presence() {
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 1,
        n_step: 2,
    };
    let batch = SequenceBatch {
        config,
        samples: vec![SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            episode_id: 7,
            learn_start: 1,
            burn_start: 0,
            observations: vec![
                Observation::new(vec![10., 20.]),
                Observation::new(vec![11., 21.]),
                Observation::new(vec![12., 22.]),
                Observation::new(vec![13., 23.]),
            ],
            steps: vec![
                step(Action::Discrete(2), 1.25, false, false),
                step(Action::Discrete(0), -2., false, false),
                step(Action::Discrete(1), 3.5, true, false),
            ],
            loss_mask: vec![false, true, false],
        }],
    };
    let spec = HistorySpec::new(2, ActionSpace::Discrete { count: 3 }).unwrap();
    let encoded = encode_history_batch(&batch, &spec).unwrap();
    assert_eq!((encoded.batch, encoded.time, encoded.input_dim), (1, 4, 7));
    assert_eq!(
        encoded.inputs,
        [
            10., 20., 0., 0., 0., 0., 0., 11., 21., 0., 0., 1., 1.25, 1., 12., 22., 1., 0., 0.,
            -2., 1., 13., 23., 0., 1., 0., 3.5, 1.,
        ]
    );
    assert_eq!(encoded.learning, [false, true, false, false]);
    assert_eq!(encoded.masks.detach, [true, true, true, true]);
    assert!(encoded.masks.detach[batch.samples[0].learn_start]);
}

fn unequal_batch() -> SequenceBatch {
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 2,
    };
    let long = SequenceSample {
        trajectory_start: 0,
        history_start: 0,
        previous: None,
        origin: Default::default(),
        episode_id: 1,
        learn_start: 2,
        burn_start: 1,
        observations: (0..=5).map(|t| Observation::new(vec![t as f32])).collect(),
        steps: (0..5)
            .map(|t| step(Action::Discrete(t % 2), (t + 1) as f32, t == 4, false))
            .collect(),
        loss_mask: vec![false, false, true, true, false],
    };
    let short = SequenceSample {
        trajectory_start: 0,
        history_start: 0,
        previous: None,
        origin: Default::default(),
        episode_id: 2,
        learn_start: 0,
        burn_start: 0,
        observations: (100..=102)
            .map(|t| Observation::new(vec![t as f32]))
            .collect(),
        steps: vec![
            step(Action::Discrete(0), 1., false, false),
            step(Action::Discrete(1), 2., false, true),
        ],
        loss_mask: vec![true, true],
    };
    SequenceBatch {
        samples: vec![long, short],
        config,
    }
}

#[test]
fn unequal_tapes_pack_batch_major_with_aligned_valid_learning_reset_and_detach_masks() {
    let batch = unequal_batch();
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let encoded = encode_history_batch(&batch, &spec).unwrap();
    assert_eq!((encoded.batch, encoded.time, encoded.input_dim), (2, 6, 5));
    assert_eq!(
        encoded.inputs,
        [
            0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 2., 0., 1., 2., 1., 3., 1., 0., 3., 1., 4., 0.,
            1., 4., 1., 5., 1., 0., 5., 1., 100., 0., 0., 0., 0., 101., 1., 0., 1., 1., 102., 0.,
            1., 2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]
    );
    assert_eq!(
        encoded.masks.valid,
        [
            true, true, true, true, true, true, true, true, true, false, false, false
        ]
    );
    assert_eq!(
        encoded.masks.reset,
        [
            true, false, false, false, false, false, true, false, false, false, false, false
        ]
    );
    assert_eq!(
        encoded.masks.detach,
        [
            true, true, true, false, true, true, true, false, true, false, false, false
        ]
    );
    assert_eq!(
        encoded.learning,
        [
            false, false, true, true, false, false, true, true, false, false, false, false
        ]
    );
    assert_eq!((encoded.masks.batch, encoded.masks.time), (2, 6));
    encoded.masks.validate().unwrap();
}

#[test]
fn both_terminal_and_truncated_final_tokens_receive_the_actual_last_action_and_reward() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    for terminated in [true, false] {
        let batch = SequenceBatch {
            config: SequenceConfig {
                burn_in: 0,
                learn_len: 1,
                n_step: 1,
            },
            samples: vec![SequenceSample {
                trajectory_start: 0,
                history_start: 0,
                previous: None,
                origin: Default::default(),
                episode_id: 1,
                learn_start: 0,
                burn_start: 0,
                observations: vec![
                    Observation::dense(vec![2.], Some(vec![true, true])),
                    Observation::dense(vec![9.], Some(vec![!terminated, !terminated])),
                ],
                steps: vec![step(Action::Discrete(1), -4., terminated, !terminated)],
                loss_mask: vec![true],
            }],
        };
        let encoded = encode_history_batch(&batch, &spec).unwrap();
        assert_eq!(encoded.inputs, [2., 0., 0., 0., 0., 9., 0., 1., -4., 1.]);
        assert_eq!(encoded.learning, [true, false]);
        assert_eq!(encoded.masks.reset, [true, false]);
    }
}

#[test]
fn continuous_history_uses_declared_bounds_and_keeps_raw_previous_action_channels() {
    let batch = SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 1,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            episode_id: 1,
            learn_start: 0,
            burn_start: 0,
            observations: vec![Observation::new(vec![0.2]), Observation::new(vec![0.4])],
            steps: vec![step(Action::Continuous(vec![0.5, -0.75]), 4., false, true)],
            loss_mask: vec![true],
        }],
    };
    let spec = HistorySpec::new(
        1,
        ActionSpace::Box {
            low: vec![-1., -1.],
            high: vec![1., 1.],
        },
    )
    .unwrap();
    let encoded = encode_history_batch(&batch, &spec).unwrap();
    assert_eq!(
        encoded.inputs,
        [0.2, 0., 0., 0., 0., 0.4, 0.5, -0.75, 4., 1.]
    );
    let tight = HistorySpec::new(
        1,
        ActionSpace::Box {
            low: vec![-0.1, -0.1],
            high: vec![0.1, 0.1],
        },
    )
    .unwrap();
    assert!(encode_history_batch(&batch, &tight).is_err());
}

#[test]
fn anchored_suffix_keeps_actual_previous_input_and_never_sets_reset() {
    use rl::core::{anchors::*, stateful::PreviousStep};
    use rl::data::temporal::SequenceOrigin;
    let mut sample = unequal_batch().samples.remove(1);
    sample.history_start = 9;
    sample.previous = Some(PreviousStep {
        action: Action::Discrete(1),
        reward: 7.,
    });
    sample.origin = SequenceOrigin::Anchored(AnchorSet {
        version: ANCHOR_VERSION,
        anchors: [(
            RepresentationId(0),
            StateAnchor {
                version: ANCHOR_VERSION,
                episode_id: sample.episode_id,
                observation_index: 9,
                phase: AnchorPhase::BeforeObservation,
                representation_id: RepresentationId(0),
                model_identity: [1; 32],
                generation: 0,
                schema: AnchorSchema {
                    codec: 1,
                    codec_version: 1,
                    representation_identity: [2; 32],
                    dimensions: vec![1],
                },
                payload: vec![0],
                exact_ancestry: true,
                source: AnchorSource::ResetReconstruction,
            },
        )]
        .into_iter()
        .collect(),
    });
    let batch = SequenceBatch {
        samples: vec![sample],
        config: unequal_batch().config,
    };
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let encoded = encode_history_batch(&batch, &spec).unwrap();
    assert_eq!(&encoded.inputs[..5], &[100., 0., 1., 7., 1.]);
    assert_eq!(encoded.masks.reset, vec![false; 3]);
    assert!(encoded.masks.detach[0]);
}

#[test]
fn invalid_declared_spaces_and_malformed_tapes_fail_before_neural_conversion() {
    let batch = unequal_batch();
    for spec in [
        HistorySpec {
            observation_dim: 2,
            action_space: ActionSpace::Discrete { count: 2 },
        },
        HistorySpec {
            observation_dim: 1,
            action_space: ActionSpace::Discrete { count: 1 },
        },
        HistorySpec {
            observation_dim: 1,
            action_space: ActionSpace::Unbounded { dimensions: 1 },
        },
        HistorySpec {
            observation_dim: usize::MAX,
            action_space: ActionSpace::Discrete { count: 2 },
        },
    ] {
        assert!(encode_history_batch(&batch, &spec).is_err());
    }
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut malformed = batch.clone();
    malformed.samples[0].loss_mask[0] = true;
    assert!(encode_history_batch(&malformed, &spec).is_err());
    let mut malformed = batch.clone();
    malformed.samples[0].observations[0]
        .as_dense_mut()
        .unwrap()
        .action_mask = Some(vec![false, true]);
    assert!(encode_history_batch(&malformed, &spec).is_err());
    let mut malformed = batch.clone();
    malformed.samples[0].steps[0].reward = f32::NAN.into();
    assert!(encode_history_batch(&malformed, &spec).is_err());
}
