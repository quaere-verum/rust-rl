#![cfg(feature = "probes")]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::{ppo::PpoConfig, recurrent_ppo::RecurrentPpo},
    core::{action::ActionSpace, checkpoint::Checkpoint, stateful::HistorySpec},
    data::temporal::SequenceConfig,
    envs::delayed_bit::DelayedBit,
    nn::{temporal_policy::TemporalPolicy, temporal_value::TemporalValue},
    prelude::{OnPolicyConfig, SyncVectorEnv},
    profiling::{self, Phase},
    train::temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
};
fn ppo(seed: u64) -> RecurrentPpo {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentPpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[4], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[4], &mut rng).unwrap(),
        PpoConfig {
            epochs: 1,
            minibatch_size: 32,
            target_kl: None,
            ..Default::default()
        },
        seed + 1,
    )
    .unwrap()
}
fn run_config(pathwise: bool) -> TemporalOnPolicyConfig {
    TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 4,
            gae_lambda: if pathwise { None } else { Some(0.95) },
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 0,
            learn_len: 4,
            n_step: 1,
        },
        ..Default::default()
    }
}
fn canonical_record(record: &rl::core::checkpoint::LearnerRecord) -> serde_json::Value {
    use std::collections::BTreeMap;
    fn pack(bytes: Vec<u8>) -> serde_json::Value {
        let reader =
            burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes)).unwrap();
        let scalars = reader
            .scalars()
            .iter()
            .map(|(key, value)| (key.clone(), format!("{value:?}")))
            .collect::<BTreeMap<_, _>>();
        let metadata = reader
            .metadata()
            .iter()
            .map(|(key, value)| (key.clone(), format!("{value:?}")))
            .collect::<BTreeMap<_, _>>();
        let tensors=reader.into_tensors().unwrap().into_iter().map(|tensor| {
            let (name,dtype,shape,id,bytes)=tensor.into_parts().unwrap();
            (name,serde_json::json!({"dtype":format!("{dtype:?}"),"shape":shape,"id":id,"bytes":bytes}))
        }).collect::<BTreeMap<_,_>>();
        serde_json::json!({"scalars":scalars,"metadata":metadata,"tensors":tensors})
    }
    let mut value = serde_json::to_value(record).unwrap();
    for model in value["models"].as_object_mut().unwrap().values_mut() {
        model["bytes"] = pack(serde_json::from_value(model["bytes"].clone()).unwrap());
    }
    for optimizer in value["optimizers"].as_object_mut().unwrap().values_mut() {
        *optimizer = pack(serde_json::from_value(optimizer.clone()).unwrap());
    }
    value
}

#[test]
fn profiling_preserves_exact_optimizer_model_rng_continuation_and_disabled_is_empty() {
    let mut plain = ppo(71);
    let mut measured = ppo(92);
    measured.restore(&plain.checkpoint().unwrap()).unwrap();
    let make = |learner: &RecurrentPpo| {
        TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(vec![DelayedBit::new(2, None, false, 703).unwrap()]).unwrap(),
            learner,
            run_config(false),
            704,
        )
        .unwrap()
    };
    let mut left = make(&plain);
    let mut right = make(&measured);
    for _ in 0..3 {
        profiling::set_enabled(false);
        profiling::reset();
        left.cycle(&mut plain).unwrap();
        assert!(profiling::snapshot().is_empty());
        assert_eq!(profiling::recurrent_work().gru_calls, 0);
        profiling::set_enabled(true);
        right.cycle(&mut measured).unwrap();
        profiling::set_enabled(false);
        let snapshot = profiling::snapshot();
        for phase in [
            Phase::HistoryReconstruction,
            Phase::HistoryUnroll,
            Phase::PolicyEvaluation,
            Phase::ValueEvaluation,
            Phase::ActorOptimization,
            Phase::CriticOptimization,
        ] {
            assert!(
                snapshot
                    .iter()
                    .any(|sample| sample.phase == phase && sample.calls > 0),
                "missing {phase:?}"
            );
        }
        assert!(
            canonical_record(&plain.checkpoint().unwrap())
                == canonical_record(&measured.checkpoint().unwrap()),
            "profiling changed exact model/optimizer/RNG continuation"
        );
    }
    profiling::reset();
    assert!(profiling::snapshot().is_empty());
}

#[test]
fn profile_counts_prefix_learning_lookahead_and_padding_in_actual_ppo_unrolls() {
    let mut learner = ppo(71);
    let mut config = run_config(false);
    config.collection.rollout_steps = 8;
    let mut trainer = TemporalOnPolicyTrainer::new(
        SyncVectorEnv::new(vec![DelayedBit::new(20, None, false, 703).unwrap()]).unwrap(),
        &learner,
        config,
        704,
    )
    .unwrap();

    profiling::reset();
    profiling::set_enabled(true);
    trainer.cycle(&mut learner).unwrap();
    profiling::set_enabled(false);

    // Five passes (three frozen audits and two training forwards) of
    // two windows with learning intervals 0..4 and 4..8.
    let work = profiling::recurrent_work();
    assert_eq!(work.gru_calls, 45);
    assert_eq!(work.prefix_rows, 20);
    assert_eq!(work.learning_rows, 40);
    assert_eq!(work.lookahead_rows, 10);
    assert_eq!(work.padded_rows, 20);
    let snapshot = profiling::snapshot();
    for (phase, calls) in [
        (Phase::MixedStep, 25),
        (Phase::LearningStep, 15),
        (Phase::LookaheadStep, 5),
        (Phase::BehaviorPreflight, 1),
        (Phase::MinibatchSelection, 1),
        (Phase::MinibatchPreparation, 1),
        (Phase::HistoryEncoding, 5),
        (Phase::HistoryGather, 5),
    ] {
        assert_eq!(
            snapshot
                .iter()
                .find(|sample| sample.phase == phase)
                .unwrap()
                .calls,
            calls,
            "{phase:?}",
        );
    }
    profiling::reset();
    assert_eq!(profiling::recurrent_work().gru_calls, 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn optimized_ppo_reuses_shared_prefix_under_each_current_encoder() {
    use rl::algorithms::recurrent_critic::RecurrentUpdateMode;
    if !std::arch::is_x86_feature_detected!("fma") {
        return;
    }
    let mut reference = ppo(71);
    let mut optimized = ppo(92);
    optimized.restore(&reference.checkpoint().unwrap()).unwrap();
    optimized.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    let make = |learner: &RecurrentPpo| {
        let mut config = run_config(false);
        config.collection.rollout_steps = 16;
        TemporalOnPolicyTrainer::new(
            SyncVectorEnv::new(vec![DelayedBit::new(70, None, false, 703).unwrap()]).unwrap(),
            learner,
            config,
            704,
        )
        .unwrap()
    };
    let mut left = make(&reference);
    let mut right = make(&optimized);
    for _ in 0..2 {
        left.cycle(&mut reference).unwrap();
        right.cycle(&mut optimized).unwrap();
    }
    let measure = |run: &mut TemporalOnPolicyTrainer<_>, learner: &mut RecurrentPpo| {
        profiling::reset();
        profiling::set_enabled(true);
        let report = run.cycle(learner).unwrap();
        profiling::set_enabled(false);
        (report, profiling::recurrent_work(), profiling::snapshot())
    };
    let (expected, reference_work, reference_phases) = measure(&mut left, &mut reference);
    let (actual, optimized_work, phases) = measure(&mut right, &mut optimized);
    assert_eq!(expected.metrics, actual.metrics);
    assert_eq!(expected.updates, actual.updates);
    let mut record = optimized.checkpoint().unwrap();
    record
        .metadata
        .as_object_mut()
        .unwrap()
        .remove("update_mode");
    assert!(canonical_record(&reference.checkpoint().unwrap()) == canonical_record(&record));
    assert_eq!(
        left.collector().lane_state(0),
        right.collector().lane_state(0)
    );

    assert_eq!(reference_work.prefix_rows, 760);
    assert_eq!(optimized_work.prefix_rows, 568);
    assert_eq!(reference_work.learning_rows, optimized_work.learning_rows);
    assert_eq!(reference_work.lookahead_rows, optimized_work.lookahead_rows);
    assert!(
        !reference_phases
            .iter()
            .any(|sample| sample.phase == Phase::EncodedLaneReconstruction)
    );
    assert_eq!(
        phases
            .iter()
            .find(|sample| sample.phase == Phase::EncodedLaneReconstruction)
            .unwrap()
            .calls,
        1,
    );
    let physical_rows = |work: profiling::RecurrentWork| {
        work.prefix_rows + work.learning_rows + work.lookahead_rows + work.padded_rows
    };
    assert_eq!(
        physical_rows(reference_work) - physical_rows(optimized_work),
        128
    );
    for phase in [Phase::DetachedPrefix, Phase::LearningWindows] {
        assert_eq!(
            phases
                .iter()
                .find(|sample| sample.phase == phase)
                .unwrap()
                .calls,
            2
        );
    }
    profiling::reset();
}
