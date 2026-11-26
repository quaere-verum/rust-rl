//! Experimental replay estimator through ordinary collection and recorded batches.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::experimental::recurrent_acer::{AcerBatch, RecurrentAcer, RecurrentAcerConfig},
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::ActionSpace,
        actor::ActMode,
        checkpoint::Snapshot,
        stateful::{BehaviorPublisher, HistorySpec},
    },
    data::{
        categorical_replay::CategoricalReplay,
        temporal::{SequenceConfig, SequenceSampler},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_categorical_actor::{CategoricalMixtureSettings, TemporalCategoricalActor},
        temporal_discrete_q::TemporalDiscreteQ,
        temporal_policy::TemporalPolicy,
    },
    train::checkpoint::{load_learner, save_learner},
};

fn learner(seed: u64) -> RecurrentAcer {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    RecurrentAcer::new(
        TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(seed),
        )
        .unwrap(),
        TemporalDiscreteQ::new(
            spec,
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(seed ^ 91),
        )
        .unwrap(),
        RecurrentAcerConfig::default(),
    )
    .unwrap()
}
#[test]
fn recorded_replay_and_learner_restore_reproduce_the_next_updates() {
    let mut learner = learner(17);
    let actor = learner
        .behavior_actor(&CategoricalMixtureSettings { epsilon: 0.1 }, 40)
        .unwrap();
    let mut collector = StatefulCollector::new(
        SyncVectorEnv::new(vec![DelayedBit::new(2, None, false, 82).unwrap()]).unwrap(),
        actor,
        73,
    )
    .unwrap();
    let mut replay = CategoricalReplay::new(64, 5, ActionSpace::Discrete { count: 2 }, 1).unwrap();
    let mut sampler = SequenceSampler::new(71);
    let config = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 1,
    };
    for _ in 0..12 {
        collector
            .tick_recorded(&[true], ActMode::Sample, &mut replay)
            .unwrap();
    }
    for _ in 0..3 {
        let batch = replay
            .batch(sampler.sample(replay.store(), &config, 3).unwrap())
            .unwrap();
        learner.update(&AcerBatch::from_recorded(batch)).unwrap();
    }
    let dir = tempfile::tempdir().unwrap();
    save_learner(dir.path().join("learner"), &learner).unwrap();
    let mut resumed = self::learner(999);
    load_learner(dir.path().join("learner"), &mut resumed).unwrap();
    let mut restored_collector: StatefulCollector<
        SyncVectorEnv<DelayedBit>,
        TemporalCategoricalActor,
    > = serde_json::from_value(serde_json::to_value(&collector).unwrap()).unwrap();
    let mut restored_replay: CategoricalReplay =
        serde_json::from_value(serde_json::to_value(&replay).unwrap()).unwrap();
    let mut restored_sampler: SequenceSampler =
        serde_json::from_value(serde_json::to_value(&sampler).unwrap()).unwrap();
    restored_collector.validate_snapshot().unwrap();
    restored_replay.validate_snapshot().unwrap();
    restored_sampler.validate_snapshot().unwrap();
    for revision in 41..47 {
        let settings = CategoricalMixtureSettings { epsilon: 0.1 };
        collector
            .publish(learner.behavior_actor(&settings, revision).unwrap())
            .unwrap();
        restored_collector
            .publish(resumed.behavior_actor(&settings, revision).unwrap())
            .unwrap();
        collector
            .tick_recorded(&[true], ActMode::Sample, &mut replay)
            .unwrap();
        restored_collector
            .tick_recorded(&[true], ActMode::Sample, &mut restored_replay)
            .unwrap();
        let a = replay
            .batch(sampler.sample(replay.store(), &config, 3).unwrap())
            .unwrap();
        let b = restored_replay
            .batch(
                restored_sampler
                    .sample(restored_replay.store(), &config, 3)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(a.behavior, b.behavior);
        assert_eq!(a.sources, b.sources);
        assert_eq!(
            serde_json::to_value(&a.sequences).unwrap(),
            serde_json::to_value(&b.sequences).unwrap()
        );
        let x = learner.update(&AcerBatch::from_recorded(a)).unwrap();
        let y = resumed.update(&AcerBatch::from_recorded(b)).unwrap();
        assert_eq!(
            serde_json::to_value(x).unwrap(),
            serde_json::to_value(y).unwrap()
        );
        assert_eq!(collector.observations(), restored_collector.observations());
        assert_eq!(
            serde_json::to_value(&replay).unwrap(),
            serde_json::to_value(&restored_replay).unwrap()
        );
    }
}
