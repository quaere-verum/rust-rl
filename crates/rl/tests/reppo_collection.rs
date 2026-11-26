use rand_chacha::ChaCha8Rng;
use rl::{
    core::actor::{PathwiseActor, PathwiseBehavior},
    prelude::*,
};
use std::cell::Cell;

#[derive(Clone)]
struct Counter {
    time: usize,
    horizon: usize,
    truncate: bool,
    space: ActionSpace,
}
impl Counter {
    fn new(horizon: usize, truncate: bool) -> Self {
        Self {
            time: 0,
            horizon,
            truncate,
            space: ActionSpace::Discrete { count: 2 },
        }
    }
    fn observation(&self, terminal: bool) -> Observation {
        Observation::dense(vec![self.time as f32], Some(vec![!terminal, false]))
    }
}
impl Env for Counter {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        Ok(EpisodeStart::fresh(self.observation(false)))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        self.space
            .check(action, self.observation(false).dense_mask())?;
        self.time += 1;
        let ended = self.time == self.horizon;
        Ok(Step {
            observation: self.observation(ended && !self.truncate),
            reward: (self.time as f32).into(),
            terminated: ended && !self.truncate,
            truncated: ended && self.truncate,
        })
    }
}

struct Fixed {
    space: ActionSpace,
    revision: Cell<u64>,
    corrupt: bool,
    change_revision: bool,
    samples: Cell<usize>,
}
impl Fixed {
    fn new() -> Self {
        Self {
            space: ActionSpace::Discrete { count: 2 },
            revision: Cell::new(7),
            corrupt: false,
            change_revision: false,
            samples: Cell::new(0),
        }
    }
}
impl Actor for Fixed {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        out.resize(obs.len(), Action::Discrete(0));
        Ok(())
    }
}
impl PathwiseActor for Fixed {
    fn revision(&self) -> u64 {
        self.revision.get()
    }
    fn sample_pathwise_into(
        &self,
        obs: ObservationBatch<'_>,
        _: &mut ChaCha8Rng,
        out: &mut PathwiseBehavior,
    ) -> Result<()> {
        self.samples.set(self.samples.get() + 1);
        out.actions.resize(obs.len(), Action::Discrete(0));
        out.log_probs
            .resize(obs.len(), if self.corrupt { f32::NAN } else { 0. });
        out.log_probs.fill(if self.corrupt { f32::NAN } else { 0. });
        out.latents.resize(obs.len(), None);
        if self.change_revision {
            self.revision.set(self.revision.get() + 1);
        }
        Ok(())
    }
}
fn collector() -> OnPolicyCollector<SyncVectorEnv<Counter>> {
    OnPolicyCollector::new(
        SyncVectorEnv::new(vec![Counter::new(2, false), Counter::new(1, true)]).unwrap(),
        3,
    )
    .unwrap()
}

#[test]
fn pathwise_collector_retains_time_lanes_final_observations_and_masks_without_values() {
    let mut c = collector();
    let mut rollout = RolloutBuffer::default();
    let actor = Fixed::new();
    let delta = c.collect_pathwise_into(&actor, 3, &mut rollout).unwrap();
    assert_eq!(delta.env_steps, 6);
    assert_eq!(actor.samples.get(), 3);
    let batch = rollout.to_batch(0.5, Some(0.1), true).unwrap();
    assert!(rollout.is_empty());
    assert!(batch.returns.is_empty());
    assert!(batch.discount_weights.is_empty());
    assert!(batch.bootstrapped().is_err());
    let p = batch.pathwise().unwrap();
    assert_eq!(p.gamma, 0.5);
    assert_eq!(p.lanes, 2);
    assert_eq!(p.revision, 7);
    assert_eq!(p.row_ids, [0, 1, 2, 3, 4, 5]);
    assert_eq!(p.episode_steps, [0, 0, 1, 0, 0, 0]);
    assert_eq!(p.rewards, [1., 1., 2., 1., 1., 1.]);
    assert_eq!(p.terminated, [false, false, true, false, false, false]);
    assert_eq!(p.truncated, [false, true, false, true, false, true]);
    assert_eq!(p.next_observations[2].dense_values(), [2.]);
    assert_eq!(
        p.next_observations[2].dense_mask(),
        Some([false, false].as_slice())
    );
    assert_eq!(batch.observations[4].dense_values(), [0.]);
    p.validate_complete().unwrap();
    batch.validate().unwrap();
}

#[test]
fn pathwise_selection_keeps_original_ids_and_cannot_reconstruct_a_fake_trajectory() {
    let mut c = collector();
    let mut rollout = RolloutBuffer::default();
    c.collect_pathwise_into(&Fixed::new(), 3, &mut rollout)
        .unwrap();
    let batch = rollout.to_batch(0.9, None, false).unwrap();
    let selected = batch.select(&[5, 0, 2]).unwrap();
    selected.validate().unwrap();
    let p = selected.pathwise().unwrap();
    assert_eq!(p.row_ids, [5, 0, 2]);
    assert_eq!(p.rollout_len, 6);
    assert_eq!(p.rewards, [1., 1., 2.]);
    assert!(p.validate_complete().is_err());
    assert_eq!(p.next_observations[2].dense_values(), [2.]);
    let decoded: OnPolicyBatch =
        serde_json::from_value(serde_json::to_value(&batch).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.pathwise().unwrap().row_ids, [0, 1, 2, 3, 4, 5]);
}

#[test]
fn malformed_metadata_and_behavior_revision_changes_are_rejected() {
    let mut c = collector();
    let mut rollout = RolloutBuffer::default();
    let mut actor = Fixed::new();
    actor.corrupt = true;
    assert!(c.collect_pathwise_into(&actor, 1, &mut rollout).is_err());
    assert!(rollout.is_empty());
    c.reset_all().unwrap();
    actor.corrupt = false;
    actor.change_revision = true;
    assert!(c.collect_pathwise_into(&actor, 1, &mut rollout).is_err());
    c.reset_all().unwrap();
    actor.change_revision = false;
    c.collect_pathwise_into(&actor, 2, &mut rollout).unwrap();
    let batch = rollout.to_batch(0.9, None, false).unwrap();
    let mut malformed = batch.clone();
    if let OnPolicyMetadata::Pathwise(p) = &mut malformed.metadata {
        p.row_ids[1] = 99;
    }
    assert!(malformed.validate().is_err());
    let mut malformed = batch.clone();
    if let OnPolicyMetadata::Pathwise(p) = &mut malformed.metadata {
        p.next_observations[0].as_dense_mut().unwrap().values[0] = 99.;
    }
    assert!(malformed.validate().is_err());
    let mut malformed = batch;
    if let OnPolicyMetadata::Pathwise(p) = &mut malformed.metadata {
        p.episode_steps[2] = 9;
    }
    assert!(malformed.validate().is_err());
}

struct InspectLearner {
    actor: Fixed,
    calls: usize,
}
impl Learner for InspectLearner {
    type Batch = OnPolicyBatch;
    fn actor(&self) -> &dyn Actor {
        &self.actor
    }
    fn update(&mut self, batch: &OnPolicyBatch) -> Result<UpdateReport> {
        batch.validate()?;
        assert_eq!(batch.pathwise()?.gamma, 0.37);
        self.calls += 1;
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        })
    }
}
impl OnPolicyLearner for InspectLearner {
    fn collection_actor(&self) -> OnPolicyActorRef<'_> {
        OnPolicyActorRef::Pathwise(&self.actor)
    }
}
#[test]
fn ordinary_trainer_uses_pathwise_capability_and_preserves_budget_and_callbacks() {
    let mut trainer = OnPolicyTrainer::new(
        collector(),
        OnPolicyConfig {
            rollout_steps: 3,
            gamma: 0.37,
            ..Default::default()
        },
    )
    .unwrap();
    let mut learner = InspectLearner {
        actor: Fixed::new(),
        calls: 0,
    };
    let mut callbacks = 0;
    let summary = trainer
        .train(
            &mut learner,
            Budget::vector_steps(5),
            Some(&mut |_| callbacks += 1),
        )
        .unwrap();
    assert_eq!(summary.counters.env_steps, 10);
    assert_eq!(summary.counters.vector_steps, 5);
    assert_eq!(summary.counters.updates, 2);
    assert_eq!(learner.calls, 2);
    assert_eq!(callbacks, 2);
}

#[test]
fn repeated_pathwise_collection_recycles_both_observation_payloads() {
    let mut c = collector();
    let mut rollout = RolloutBuffer::default();
    let actor = Fixed::new();
    let mut batch = OnPolicyBatch::default();
    let mut first = None;
    for cycle in 0..3 {
        c.collect_pathwise_into(&actor, 3, &mut rollout).unwrap();
        rollout.to_batch_into(0.9, None, false, &mut batch).unwrap();
        let pointers = (
            batch.observations[0].dense_values().as_ptr(),
            batch.pathwise().unwrap().next_observations[0]
                .dense_values()
                .as_ptr(),
        );
        if cycle == 0 {
            first = Some(pointers);
        }
        if cycle == 2 {
            assert_eq!(Some(pointers), first);
        }
    }
}
