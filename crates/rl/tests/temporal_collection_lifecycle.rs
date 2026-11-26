#![cfg(feature = "neural")]

//! A metadata-only learner isolates the generic collection-start transaction.
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::data::temporal_on_policy::TemporalOnPolicyLearner;
use rl::{
    Error, Result,
    algorithms::{ppo::PpoConfig, recurrent_ppo::RecurrentPpo},
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        learner::UpdateReport,
        stateful::{BehaviorPublisher, HistorySpec, StatefulLearner},
    },
    data::{
        temporal::SequenceConfig,
        temporal_on_policy::{TemporalBatchKind, TemporalOnPolicyBatch},
    },
    envs::delayed_bit::DelayedBit,
    nn::{
        temporal_actor::TemporalActor, temporal_policy::TemporalPolicy,
        temporal_value::TemporalValue,
    },
    prelude::*,
    train::temporal_on_policy::{TemporalOnPolicyConfig, TemporalOnPolicyTrainer},
};
use std::cell::Cell;

struct LifecycleLearner {
    actor: TemporalActor,
    revision: Cell<u64>,
    starts: Cell<u64>,
    reject: Cell<bool>,
}
impl BehaviorPublisher for LifecycleLearner {
    type Actor = TemporalActor;
    type Settings = ();
    fn behavior_actor(&self, _: &(), revision: u64) -> Result<TemporalActor> {
        if revision != self.revision.get() {
            return Err(Error::State("wrong fixture revision".into()));
        }
        Ok(self.actor.with_revision(revision))
    }
}
impl StatefulLearner for LifecycleLearner {
    type Batch = TemporalOnPolicyBatch<TemporalActor>;
    fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
        assert_eq!(batch.revision, self.revision.get());
        batch.validate()?;
        self.revision.set(self.revision.get() + 1);
        Ok(UpdateReport::default()) // No optimization: this fixture tests lifecycle only.
    }
}
impl Checkpoint for LifecycleLearner {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        LearnerRecord::new("lifecycle_fixture", ())
    }
    fn restore(&mut self, _: &LearnerRecord) -> Result<()> {
        Err(Error::State("fixture has no checkpoint".into()))
    }
}
impl TemporalOnPolicyLearner<TemporalActor> for LifecycleLearner {
    fn collection_kind(&self) -> TemporalBatchKind {
        TemporalBatchKind::Bootstrapped
    }
    fn revision(&self) -> u64 {
        self.revision.get()
    }
    fn begin_collection(&self) -> Result<()> {
        if self.reject.get() {
            return Err(Error::State("rejected fixture start".into()));
        }
        self.revision.set(self.revision.get() + 1);
        self.starts.set(self.starts.get() + 1);
        Ok(())
    }
}
fn build() -> (
    LifecycleLearner,
    TemporalOnPolicyTrainer<SyncVectorEnv<DelayedBit>>,
) {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let ppo = RecurrentPpo::new(
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
        PpoConfig::default(),
        23,
    )
    .unwrap();
    let learner = LifecycleLearner {
        actor: ppo.behavior_actor(&(), 0).unwrap(),
        revision: Cell::new(0),
        starts: Cell::new(0),
        reject: Cell::new(false),
    };
    let env = SyncVectorEnv::new(vec![DelayedBit::new(19, Some(8), false, 47).unwrap()]).unwrap();
    let config = TemporalOnPolicyConfig {
        collection: OnPolicyConfig {
            rollout_steps: 2,
            normalize_advantages: false,
            ..Default::default()
        },
        sequence: SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
        ..Default::default()
    };
    let trainer = TemporalOnPolicyTrainer::new(env, &learner, config, 71).unwrap();
    (learner, trainer)
}
#[test]
fn collection_start_revision_is_adopted_once_without_reset_or_rng_change() {
    let (mut learner, mut trainer) = build();
    let id = trainer.collector().lane_episode(0);
    trainer.collect_tick(&learner).unwrap();
    trainer.collect_tick(&learner).unwrap();
    assert_eq!(learner.starts.get(), 1);
    assert_eq!(trainer.collector().lane_revision(0), Some(1));
    trainer.update(&mut learner).unwrap();
    let state = trainer.collector().lane_state(0).unwrap().clone();
    trainer.collect_tick(&learner).unwrap();
    assert_eq!(learner.starts.get(), 2);
    assert_eq!(trainer.collector().lane_revision(0), Some(3));
    assert_eq!(trainer.collector().lane_episode(0), id);
    assert_eq!(trainer.collector().lane_timestep(0), Some(3));
    assert_ne!(trainer.collector().lane_state(0), Some(&state));
    assert_eq!(
        trainer
            .rollout()
            .fresh_recordings()
            .next()
            .unwrap()
            .revision,
        3
    );
}
#[test]
fn rejected_collection_start_leaves_environment_and_streams_untouched() {
    let (learner, mut trainer) = build();
    learner.reject.set(true);
    let before = serde_json::to_value(&trainer).unwrap();
    assert!(trainer.collect_tick(&learner).is_err());
    assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    assert_eq!(learner.revision.get(), 0);
    assert_eq!(learner.starts.get(), 0);
    learner.reject.set(false);
    trainer.collect_tick(&learner).unwrap();
    assert_eq!(trainer.counters().env_steps, 1);
}
