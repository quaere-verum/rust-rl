use rand_chacha::ChaCha8Rng;
use rl::{core::temporal::ReturnSpec, envs::delayed_bit::DelayedBit, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
enum PlanningSettings {
    Reactive,
    Search { depth: usize, remember: bool },
}

struct RecordingLearner {
    updates: u64,
}
impl StatefulLearner for RecordingLearner {
    type Batch = SequenceBatch;
    fn update(&mut self, batch: &SequenceBatch) -> Result<UpdateReport> {
        batch.validate()?;
        self.updates += 1;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        })
    }
}
impl TemporalOffPolicyLearner for RecordingLearner {
    fn return_contract(&self) -> ReturnSpec {
        ReturnSpec {
            gamma: 1.,
            n_step: 1,
        }
    }
}
impl BehaviorPublisher for RecordingLearner {
    type Actor = PlanningActor;
    type Settings = PlanningSettings;
    fn behavior_actor(&self, settings: &Self::Settings, revision: u64) -> Result<PlanningActor> {
        Ok(PlanningActor {
            revision,
            settings: settings.clone(),
        })
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct PlanningActor {
    revision: u64,
    settings: PlanningSettings,
}
impl StatefulActor for PlanningActor {
    type State = (usize, u64);
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(5)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<Self::State> {
        Ok((0, 0))
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        if state.0 > 1 {
            return Err(Error::State("invalid public cue memory".into()));
        }
        Ok(())
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        input.observation.validate(5, self.action_space(), false)?;
        Ok((
            if input.previous.is_none() {
                usize::from(input.observation.dense_values()[0] > 0.)
            } else {
                prior.0
            },
            prior.1 + 1,
        ))
    }
    fn decide(
        &self,
        state: &Self::State,
        obs: &Observation,
        _: ActMode,
        _: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let action = if obs.dense_values()[2] > 0.
            && matches!(
                self.settings,
                PlanningSettings::Search { remember: true, .. }
            ) {
            state.0
        } else {
            0
        };
        Ok(StatefulDecision {
            action: Action::Discrete(action),
            log_prob: None,
        })
    }
}
impl Snapshot for PlanningActor {
    fn validate_snapshot(&self) -> Result<()> {
        Ok(())
    }
}

#[test]
fn typed_non_epsilon_behavior_publishes_without_coupling_the_learning_contract() {
    let mut learner = RecordingLearner { updates: 0 };
    let settings = PlanningSettings::Search {
        depth: 3,
        remember: true,
    };
    let actor = learner.behavior_actor(&settings, 0).unwrap();
    let config = TemporalOffPolicyConfig {
        replay_capacity: 64,
        batch_size: 2,
        warmup_transitions: 0,
        n_step: 1,
        gamma: 1.,
        updates_per_transition: 0.5,
    };
    let env = SyncVectorEnv::new(vec![
        DelayedBit::new(3, Some(1), false, 11).unwrap(),
        DelayedBit::new(4, Some(2), false, 23).unwrap(),
    ])
    .unwrap();
    let sequence = SequenceConfig {
        burn_in: 1,
        learn_len: 3,
        n_step: 1,
    };
    let mut trainer =
        TemporalTrainer::new(env, actor, config, sequence, FixedBehavior(settings), 47).unwrap();
    trainer.run(&mut learner, Budget::vector_steps(8)).unwrap();
    assert_eq!(learner.updates, 8);
    assert!(trainer.collector().latest_revision() > 0);
    assert!(trainer.counters().episodes >= 2);
    trainer.validate_snapshot().unwrap();
    assert!(!serde_json::to_string(&trainer).unwrap().contains("epsilon"));
}

#[test]
fn update_only_learner_needs_no_actor_or_behavior_implementation() {
    struct Sum;
    impl StatefulLearner for Sum {
        type Batch = Vec<f32>;
        fn update(&mut self, batch: &Self::Batch) -> Result<UpdateReport> {
            Ok(UpdateReport::default().metric("sum", batch.iter().map(|&x| f64::from(x)).sum()))
        }
    }
    assert_eq!(
        StatefulLearner::update(&mut Sum, &vec![1., 2.])
            .unwrap()
            .metrics["sum"],
        3.
    );
}

#[test]
fn optional_temporal_checkpoint_metadata_does_not_change_ordinary_records() {
    let record = rl::core::checkpoint::LearnerRecord::new("ordinary", ()).unwrap();
    let value = serde_json::to_value(&record).unwrap();
    assert!(value.get("temporal_returns").is_none());
    let restored: rl::core::checkpoint::LearnerRecord = serde_json::from_value(value).unwrap();
    assert!(restored.temporal_returns.is_none());
}

#[test]
fn return_identity_uses_f32_bits_without_tolerance() {
    let a = ReturnSpec {
        gamma: 0.99,
        n_step: 3,
    };
    a.check(a).unwrap();
    assert!(
        a.check(ReturnSpec {
            gamma: f32::from_bits(a.gamma.to_bits() + 1),
            ..a
        })
        .is_err()
    );
    assert!(a.check(ReturnSpec { n_step: 1, ..a }).is_err());
    assert!(
        ReturnSpec {
            gamma: 0.,
            n_step: 1
        }
        .check(ReturnSpec {
            gamma: -0.,
            n_step: 1
        })
        .is_err()
    );
}
