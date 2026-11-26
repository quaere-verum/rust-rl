#![cfg(feature = "neural")]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{prelude::*, train::observer::UpdateBoundary};
use serde::{Deserialize, Serialize};
use std::cell::Cell;

#[derive(Clone, Serialize, Deserialize)]
struct OptionClock {
    duration: u64,
    physical_steps: u64,
    time: usize,
    horizon: usize,
    truncated: bool,
    actions: Vec<Action>,
    space: ActionSpace,
}
impl Env for OptionClock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.physical_steps += self.duration;
        self.time += 1;
        self.actions.push(action.clone());
        Ok(Step {
            observation: Observation::new(vec![self.time as f32]),
            reward: 1.0_f32.into(),
            terminated: self.time == self.horizon && !self.truncated,
            truncated: self.time == self.horizon && self.truncated,
        })
    }
}
type Options = SyncVectorEnv<OptionClock>;
fn environment() -> Options {
    SyncVectorEnv::new(
        [(2, 3, false), (5, 2, true)]
            .map(|(duration, horizon, truncated)| OptionClock {
                duration,
                physical_steps: 0,
                time: 0,
                horizon,
                truncated,
                actions: vec![],
                space: ActionSpace::Discrete { count: 2 },
            })
            .to_vec(),
    )
    .unwrap()
}
fn physical(env: &Options) -> u64 {
    env.environments()
        .iter()
        .map(|lane| lane.physical_steps)
        .sum()
}
#[derive(Clone, Copy)]
enum Kind {
    Bootstrapped,
    Pathwise,
    Episodic,
}
struct Recorded {
    kind: Kind,
    space: ActionSpace,
    batches: Vec<OnPolicyBatch>,
    samples: Cell<usize>,
    fail_bootstrap: Cell<bool>,
}
impl Recorded {
    fn new(kind: Kind) -> Self {
        Self {
            kind,
            space: ActionSpace::Discrete { count: 2 },
            batches: vec![],
            samples: Cell::new(0),
            fail_bootstrap: Cell::new(false),
        }
    }
}
impl Actor for Recorded {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        rng: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        self.samples.set(self.samples.get() + obs.len());
        out.clear();
        out.extend(obs.iter().map(|_| Action::Discrete(rng.random_range(0..2))));
        Ok(())
    }
}
impl RolloutActor for Recorded {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()> {
        self.act_into(obs, ActMode::Sample, rng, &mut out.actions)?;
        out.values = obs.iter().map(|o| o.dense_values()[0]).collect();
        out.log_probs.resize(obs.len(), -std::f32::consts::LN_2);
        out.latents.resize(obs.len(), None);
        Ok(())
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        if self.fail_bootstrap.get() {
            return Err(Error::State("bootstrap failure".into()));
        }
        *out = obs.iter().map(|o| o.dense_values()[0]).collect();
        Ok(())
    }
}
impl EpisodeActor for Recorded {
    fn sample_episode_into(
        &self,
        _: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        _: &mut EpisodeBehavior,
    ) -> Result<()> {
        panic!("preflight must reject")
    }
}
impl PathwiseActor for Recorded {
    fn revision(&self) -> u64 {
        0
    }
    fn sample_pathwise_into(
        &self,
        _: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        _: &mut PathwiseBehavior,
    ) -> Result<()> {
        panic!("preflight must reject")
    }
}
impl Learner for Recorded {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        self
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        batch.validate()?;
        self.batches.push(batch.clone());
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        })
    }
}
impl OnPolicyLearner for Recorded {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        match self.kind {
            Kind::Bootstrapped => OnPolicyActorRef::Bootstrapped(self),
            Kind::Pathwise => OnPolicyActorRef::Pathwise(self),
            Kind::Episodic => OnPolicyActorRef::Episodic(self),
        }
    }
}
fn trainer() -> OnPolicyTrainer<Options> {
    OnPolicyTrainer::new(
        OnPolicyCollector::new(environment(), 11).unwrap(),
        OnPolicyConfig {
            rollout_steps: 32,
            gamma: 0.5,
            gae_lambda: Some(1.),
            normalize_advantages: false,
            ..Default::default()
        },
    )
    .unwrap()
}
fn budget() -> Budget {
    Budget {
        env_steps: Some(128),
        ..Default::default()
    }
}

fn assert_checkpoint_equal(left: &impl Checkpoint, right: &impl Checkpoint) {
    let left = serde_json::to_value(left.checkpoint().unwrap()).unwrap();
    let right = serde_json::to_value(right.checkpoint().unwrap()).unwrap();
    assert_eq!(left["metadata"], right["metadata"]); // Includes learner RNG and counters.
    assert_eq!(left["algorithm"], right["algorithm"]);
    for category in ["models", "optimizers"] {
        assert_eq!(
            left[category]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            right[category]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>()
        );
        for (name, value) in left[category].as_object().unwrap() {
            let other = &right[category][name];
            let (value, other) = if category == "models" {
                assert_eq!(value["signature"], other["signature"]);
                assert_eq!(value["parameter_shapes"], other["parameter_shapes"]);
                (&value["bytes"], &other["bytes"])
            } else {
                (value, other)
            };
            let read = |value: &serde_json::Value| {
                let bytes = serde_json::from_value::<Vec<u8>>(value.clone()).unwrap();
                burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes)).unwrap()
            };
            let a = read(value);
            let b = read(other);
            assert!(
                a.scalars() == b.scalars(),
                "{category}/{name}: scalar state"
            );
            assert!(a.metadata() == b.metadata(), "{category}/{name}: metadata");
            // Pack tensor ordering is not canonical. Compare every exact tensor
            // byte, ID, dtype and shape by name, as existing checkpoint tests do.
            let tensors = |reader: burn_pack::Reader| {
                reader
                    .into_tensors()
                    .unwrap()
                    .into_iter()
                    .map(|tensor| {
                        let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                        (name, (dtype, shape.to_vec(), id, bytes.to_vec()))
                    })
                    .collect::<std::collections::BTreeMap<_, _>>()
            };
            assert!(
                tensors(a) == tensors(b),
                "{category}/{name}: exact tensor state"
            );
        }
    }
}

#[test]
fn physical_stop_is_after_each_complete_vector_step_and_preserves_bootstraps() {
    let mut collector = OnPolicyCollector::new(environment(), 11).unwrap();
    let actor = Recorded::new(Kind::Bootstrapped);
    let mut rollout = RolloutBuffer::default();
    let mut inspected = vec![];
    let delta = collector
        .collect_steps_until_into(&actor, 32, &mut rollout, &mut |env| {
            inspected.push(physical(env));
            physical(env) >= 23
        })
        .unwrap();
    assert_eq!(inspected, [7, 14, 21, 28]);
    assert_eq!(
        (delta.vector_steps, delta.env_steps, delta.episodes),
        (4, 8, 3)
    );
    assert_eq!(rollout.len(), 8);
    assert_eq!(
        (0..8)
            .map(|i| rollout.next_value(i).unwrap())
            .collect::<Vec<_>>(),
        [1., 1., 2., 2., 0., 1., 1., 2.]
    );
    for i in 0..8 {
        let row = rollout.row(i).unwrap();
        assert_eq!(row.transition.terminated, i == 4);
        assert_eq!(row.transition.truncated, i == 3 || i == 7);
    }
    // Continuing lane 0 is not reset by the external budget. Lane 1's true
    // truncation does reset before the next physical vector step.
    collector
        .collect_steps_into(&actor, 1, &mut rollout)
        .unwrap();
    assert_eq!(
        rollout
            .row(0)
            .unwrap()
            .transition
            .observation
            .dense_values(),
        [1.]
    );
    assert_eq!(
        rollout
            .row(1)
            .unwrap()
            .transition
            .observation
            .dense_values(),
        [0.]
    );
    assert_eq!(physical(collector.environment()), 35);
}

#[test]
fn partial_rollout_gets_one_ordinary_update_and_honest_decision_counters() {
    let mut trainer = trainer();
    let mut learner = Recorded::new(Kind::Bootstrapped);
    let mut boundaries = vec![];
    let mut inspected = vec![];
    let summary = trainer
        .train_bootstrapped_observed_until(
            &mut learner,
            budget(),
            &mut |learner: &Recorded, boundary: UpdateBoundary<'_>| {
                boundaries.push((
                    learner.batches.len(),
                    boundary.learner_call,
                    boundary.collection_cycle,
                ));
                Ok(())
            },
            &mut |env| {
                inspected.push(physical(env));
                physical(env) >= 23
            },
        )
        .unwrap();
    assert_eq!(
        (
            summary.counters.vector_steps,
            summary.counters.env_steps,
            summary.counters.episodes
        ),
        (4, 8, 3)
    );
    assert_eq!(summary.counters.updates, 1);
    assert_eq!(boundaries, [(1, 1, 1)]);
    assert_eq!(inspected, [0, 7, 14, 21, 28]);
    assert_eq!(learner.batches.len(), 1);
    assert_eq!(
        trainer.last_batch().unwrap().returns,
        [1.75, 2., 1.5, 2., 1., 2., 1.5, 2.]
    );
    assert_eq!(trainer.last_batch().unwrap().len(), 8);
}

#[test]
fn already_stopped_is_a_noop_and_wrong_actor_is_rejected_before_callback_or_transition() {
    for kind in [Kind::Bootstrapped, Kind::Pathwise, Kind::Episodic] {
        let mut trainer = trainer();
        let before = serde_json::to_value(&trainer).unwrap();
        let mut learner = Recorded::new(kind);
        let mut stop_calls = 0;
        let result = trainer.train_bootstrapped_observed_until(
            &mut learner,
            budget(),
            &mut |_: &Recorded, _: UpdateBoundary<'_>| -> Result<()> { panic!("no update") },
            &mut |_| {
                stop_calls += 1;
                true
            },
        );
        match kind {
            Kind::Bootstrapped => {
                assert!(result.is_ok());
                assert_eq!(stop_calls, 1);
            }
            _ => {
                assert!(matches!(result, Err(Error::Config(_))));
                assert_eq!(stop_calls, 0);
            }
        }
        assert_eq!(before, serde_json::to_value(&trainer).unwrap());
        assert!(learner.batches.is_empty());
        assert_eq!(learner.samples.get(), 0);
    }
}

#[test]
fn a_one_shot_stop_is_not_repolled_and_resumed_after_the_partial_update() {
    let mut trainer = trainer();
    let mut learner = Recorded::new(Kind::Bootstrapped);
    let mut checks = 0;
    let summary = trainer
        .train_bootstrapped_observed_until(
            &mut learner,
            Budget::vector_steps(6),
            &mut |_: &Recorded, _: UpdateBoundary<'_>| Ok(()),
            &mut |_| {
                checks += 1;
                checks == 2
            },
        )
        .unwrap();
    assert_eq!(summary.counters.vector_steps, 1);
    assert_eq!(summary.counters.env_steps, 2);
    assert_eq!(summary.counters.updates, 1);
    assert_eq!(checks, 2);
}

#[test]
fn bootstrap_failure_after_stop_invalidates_and_poisoning_requires_explicit_reset() {
    let mut collector = OnPolicyCollector::new(environment(), 11).unwrap();
    let actor = Recorded::new(Kind::Bootstrapped);
    actor.fail_bootstrap.set(true);
    let mut rollout = RolloutBuffer::default();
    assert!(
        collector
            .collect_steps_until_into(&actor, 32, &mut rollout, &mut |_| true)
            .is_err()
    );
    assert!(rollout.is_empty());
    assert_eq!(physical(collector.environment()), 7);
    actor.fail_bootstrap.set(false);
    assert!(
        collector
            .collect_steps_until_into(&actor, 32, &mut rollout, &mut |_| true)
            .is_err()
    );
    assert_eq!(physical(collector.environment()), 7);
    collector.reset_all().unwrap();
    assert_eq!(
        collector
            .collect_steps_until_into(&actor, 32, &mut rollout, &mut |_| true)
            .unwrap()
            .env_steps,
        2
    );
}

#[test]
fn always_false_matches_default_actions_rng_models_optimizers_and_continuation() {
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let models = ActorCritic {
        policy: Policy::new(1, ActionSpace::Discrete { count: 2 }, &[4], &mut rng).unwrap(),
        value: Value::new(1, &[4], &mut rng).unwrap(),
    };
    let config = PpoConfig {
        epochs: 1,
        minibatch_size: 8,
        ..Default::default()
    };
    let mut standard = Ppo::new(models.clone(), config.clone(), 19).unwrap();
    let mut stopped = Ppo::new(models, config, 19).unwrap();
    let mut standard_trainer = trainer();
    let mut stopped_trainer = trainer();
    for steps in [35, 3] {
        standard_trainer
            .train(&mut standard, Budget::vector_steps(steps), None)
            .unwrap();
        stopped_trainer
            .train_bootstrapped_observed_until(
                &mut stopped,
                Budget::vector_steps(steps),
                &mut |_: &Ppo, _: UpdateBoundary<'_>| Ok(()),
                &mut |_| false,
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(&standard_trainer).unwrap(),
            serde_json::to_value(&stopped_trainer).unwrap()
        );
        assert_eq!(
            serde_json::to_value(standard_trainer.last_batch()).unwrap(),
            serde_json::to_value(stopped_trainer.last_batch()).unwrap()
        );
        assert_checkpoint_equal(&standard, &stopped);
    }
}
