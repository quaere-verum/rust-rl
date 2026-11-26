#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        anchors::{AnchorSource, RepresentationId},
        env::Observation,
        reconstruction::{
            ReconstructionConfig, ReconstructionMode, ReconstructionModel, advance_anchor,
            decode_anchor, reset_anchor,
        },
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
    },
    nn::{
        network::Mlp,
        reconstruction::{GRU_F32_LE_CODEC, GRU_F32_LE_CODEC_VERSION, GruRepresentation},
        recurrent_actor::RecurrentActor,
        temporal::{GruEncoder, GruQ, RecurrentQ},
    },
};

const ONLINE: RepresentationId = RepresentationId(1);
const TARGET: RepresentationId = RepresentationId(2);

fn spec() -> HistorySpec {
    HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap()
}
fn q(spec: &HistorySpec, hidden: usize, seed: u64) -> GruQ {
    let ActionSpace::Discrete { count } = spec.action_space else {
        panic!("discrete fixture")
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), hidden, &mut rng).unwrap(),
        Mlp::new(hidden, &[4], count, &mut rng).unwrap(),
    )
    .unwrap()
}
fn actor(seed: u64) -> RecurrentActor {
    RecurrentActor::new(q(&spec(), 4, seed), spec(), 0.35, 7).unwrap()
}
fn model(seed: u64, generation: u64) -> GruRepresentation {
    GruRepresentation::new(actor(seed), generation).unwrap()
}
fn anchored(max_generation_lag: Option<u64>) -> ReconstructionConfig {
    ReconstructionConfig {
        version: 1,
        mode: ReconstructionMode::Anchored { max_generation_lag },
    }
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn little_endian_codec_preserves_every_f32_bit_including_negative_zero() {
    let model = model(17, 0);
    let state = vec![0., -0., f32::from_bits(1), -1.25];
    let expected = vec![0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 0, 0, 160, 191];
    assert_eq!(model.encode_state(&state).unwrap(), expected);
    assert_eq!(
        bits(&model.decode_state(&expected).unwrap()),
        vec![0, 0x8000_0000, 1, 0xbfa0_0000]
    );
    assert_eq!(model.schema().codec, GRU_F32_LE_CODEC);
    assert_eq!(model.schema().codec_version, GRU_F32_LE_CODEC_VERSION);
    assert_eq!(model.schema().dimensions, vec![4]);
}

#[test]
fn codec_rejects_wrong_shapes_partial_words_and_nonfinite_values() {
    let model = model(17, 0);
    for state in [
        vec![],
        vec![0.; 3],
        vec![0.; 5],
        vec![f32::NAN; 4],
        vec![f32::INFINITY; 4],
        vec![f32::NEG_INFINITY; 4],
    ] {
        assert!(model.validate_state(&state).is_err());
        assert!(model.encode_state(&state).is_err());
    }
    for bytes in [vec![], vec![0; 4], vec![0; 15], vec![0; 17], vec![0; 20]] {
        assert!(model.decode_state(&bytes).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = vec![0; 16];
        bytes[4..8].copy_from_slice(&value.to_le_bytes());
        assert!(model.decode_state(&bytes).is_err());
    }
}

#[test]
fn reconstruction_assimilates_the_same_public_history_as_the_frozen_actor() {
    let actor = actor(17);
    let representation = GruRepresentation::new(actor.clone(), 12).unwrap();
    let observations = [
        Observation::new(vec![0.25]),
        Observation::new(vec![-0.5]),
        Observation::new(vec![0.75]),
    ];
    let actions = [Action::Discrete(1), Action::Discrete(0)];
    let rewards = [0.2, -0.3];
    let mut expected = actor.initial_state().unwrap();
    let mut actual = representation.initial_state().unwrap();
    assert_eq!(actual, vec![0.; 4]);
    for (t, observation) in observations.iter().enumerate() {
        let previous = t.checked_sub(1).map(|i| PreviousInput {
            factual: None,
            action: &actions[i],
            reward: rewards[i],
        });
        let input = HistoryInput {
            observation,
            previous,
        };
        let prior = actual.clone();
        expected = actor.assimilate(&expected, input).unwrap();
        actual = representation.assimilate(&actual, input).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        assert_eq!(
            bits(&actual),
            bits(&representation.assimilate(&prior, input).unwrap())
        );
        assert_eq!(
            bits(&actual),
            bits(
                &representation
                    .decode_state(&representation.encode_state(&actual).unwrap())
                    .unwrap()
            )
        );
    }
    assert_eq!(representation.generation(), 12);
    assert_eq!(representation.model_identity(), actor.model_identity());
}

#[test]
fn parameter_and_head_changes_keep_schema_compatible_but_have_distinct_model_identity() {
    let old = model(17, 3);
    let changed_parameters = model(19, 4);
    assert_eq!(old.schema(), changed_parameters.schema());
    assert_ne!(old.model_identity(), changed_parameters.model_identity());
    let mut changed_head = q(&spec(), 4, 17);
    changed_head.head = Mlp::new(4, &[3, 5], 2, &mut ChaCha8Rng::seed_from_u64(41)).unwrap();
    let changed_head = GruRepresentation::new(
        RecurrentActor::new(changed_head, spec(), 0., 99).unwrap(),
        4,
    )
    .unwrap();
    assert_eq!(old.schema(), changed_head.schema());
    assert_ne!(old.model_identity(), changed_head.model_identity());
    let observation = Observation::new(vec![0.4]);
    let input = HistoryInput {
        observation: &observation,
        previous: None,
    };
    assert_eq!(
        old.assimilate(&old.initial_state().unwrap(), input)
            .unwrap(),
        changed_head
            .assimilate(&changed_head.initial_state().unwrap(), input)
            .unwrap()
    );
    let original_anchor = reset_anchor(&old, ONLINE, 9).unwrap();
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &changed_head,
            ONLINE,
            9,
            0,
            &original_anchor
        )
        .is_err()
    );
    assert!(
        decode_anchor(
            anchored(None),
            &changed_head,
            ONLINE,
            9,
            0,
            &original_anchor
        )
        .unwrap()
        .1
        .approximate
    );
}

#[test]
fn schema_hash_includes_history_layout_dimensions_and_gru_controls() {
    let base = model(17, 0);
    for (reset_after, clip) in [(false, None), (true, Some(0.25))] {
        let mut q = q(&spec(), 4, 17);
        q.encoder.gru.reset_after = reset_after;
        q.encoder.gru.clip = clip;
        let changed =
            GruRepresentation::new(RecurrentActor::new(q, spec(), 0.35, 7).unwrap(), 0).unwrap();
        assert_ne!(
            base.schema().representation_identity,
            changed.schema().representation_identity
        );
    }
    let wider = GruRepresentation::new(
        RecurrentActor::new(q(&spec(), 5, 17), spec(), 0.35, 7).unwrap(),
        0,
    )
    .unwrap();
    assert_ne!(base.schema(), wider.schema());
    // Both encoders have input width6 and hidden width4, but channel boundaries
    // differ: observation1+onehot3 vs observation2+onehot2.
    let a = HistorySpec::new(1, ActionSpace::Discrete { count: 3 }).unwrap();
    let b = HistorySpec::new(2, ActionSpace::Discrete { count: 2 }).unwrap();
    let a =
        GruRepresentation::new(RecurrentActor::new(q(&a, 4, 17), a, 0., 0).unwrap(), 0).unwrap();
    let b =
        GruRepresentation::new(RecurrentActor::new(q(&b, 4, 17), b, 0., 0).unwrap(), 0).unwrap();
    assert_ne!(
        a.schema().representation_identity,
        b.schema().representation_identity
    );
}

#[test]
fn anchor_envelope_rejects_unknown_codecs_versions_schemas_and_exact_model_mismatch() {
    let model = model(17, 2);
    let original = reset_anchor(&model, ONLINE, 9).unwrap();
    for corruption in 0..7 {
        let mut anchor = original.clone();
        match corruption {
            0 => anchor.schema.codec ^= 1,
            1 => anchor.schema.codec_version = 0,
            2 => anchor.schema.codec_version = 99,
            3 => anchor.schema.representation_identity[0] ^= 1,
            4 => anchor.schema.dimensions = vec![2, 2],
            5 => anchor.version = 99,
            _ => anchor.model_identity[0] ^= 1,
        }
        assert!(
            decode_anchor(
                ReconstructionConfig::default(),
                &model,
                ONLINE,
                9,
                0,
                &anchor
            )
            .is_err(),
            "corruption {corruption}"
        );
        if corruption < 6 {
            assert!(decode_anchor(anchored(None), &model, ONLINE, 9, 0, &anchor).is_err());
        }
    }
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &model,
            TARGET,
            9,
            0,
            &original
        )
        .is_err()
    );
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &model,
            ONLINE,
            10,
            0,
            &original
        )
        .is_err()
    );
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &model,
            ONLINE,
            9,
            1,
            &original
        )
        .is_err()
    );
}

#[test]
fn generation_lag_and_approximate_ancestry_remain_explicit_across_advancement() {
    let old = model(17, 3);
    let current = model(19, 5);
    let old_anchor = reset_anchor(&old, ONLINE, 9).unwrap();
    assert!(decode_anchor(anchored(Some(1)), &current, ONLINE, 9, 0, &old_anchor).is_err());
    let (_, decision) =
        decode_anchor(anchored(Some(2)), &current, ONLINE, 9, 0, &old_anchor).unwrap();
    assert!(decision.approximate);
    assert_eq!(decision.generation_lag, 2);
    assert_eq!(decision.source_model_identity, old.model_identity());
    assert_eq!(decision.current_model_identity, current.model_identity());
    let observation = Observation::new(vec![0.4]);
    let (next, decision) = advance_anchor(
        anchored(Some(2)),
        &current,
        &old_anchor,
        HistoryInput {
            observation: &observation,
            previous: None,
        },
    )
    .unwrap();
    assert!(decision.approximate);
    assert!(!next.exact_ancestry);
    assert_eq!(next.generation, 5);
    assert!(matches!(
        &next.source,
        AnchorSource::Continued {
            parent_generation: 3,
            parent_exact: true,
            ..
        }
    ));
    let (_, inherited) = decode_anchor(anchored(Some(0)), &current, ONLINE, 9, 1, &next).unwrap();
    assert!(inherited.approximate);
    assert_eq!(inherited.generation_lag, 0);
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &current,
            ONLINE,
            9,
            1,
            &next
        )
        .is_err()
    );
    let future = GruRepresentation::new(old.actor().clone(), 6).unwrap();
    let future_anchor = reset_anchor(&future, ONLINE, 9).unwrap();
    assert!(decode_anchor(anchored(None), &current, ONLINE, 9, 0, &future_anchor).is_err());
}

#[test]
fn current_online_and_target_anchors_remain_distinct_even_with_identical_initial_weights() {
    let online = model(17, 4);
    let target = model(17, 4);
    let online_anchor = reset_anchor(&online, ONLINE, 9).unwrap();
    let target_anchor = reset_anchor(&target, TARGET, 9).unwrap();
    assert_eq!(online.model_identity(), target.model_identity());
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &online,
            ONLINE,
            9,
            0,
            &online_anchor
        )
        .is_ok()
    );
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &target,
            TARGET,
            9,
            0,
            &target_anchor
        )
        .is_ok()
    );
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &target,
            TARGET,
            9,
            0,
            &online_anchor
        )
        .is_err()
    );
    assert!(
        decode_anchor(
            ReconstructionConfig::default(),
            &online,
            ONLINE,
            9,
            0,
            &target_anchor
        )
        .is_err()
    );
}
