use rand_chacha::ChaCha8Rng;
use rl::{Result, prelude::*};
use std::cell::Cell;
#[derive(Clone)]
struct Clock {
    t: usize,
    horizon: usize,
    truncated: bool,
    space: ActionSpace,
}
impl Env for Clock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.t = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, _: &Action) -> Result<Step> {
        self.t += 1;
        Ok(Step {
            observation: Observation::new(vec![self.t as f32]),
            reward: 1.0_f32.into(),
            terminated: self.t == self.horizon && !self.truncated,
            truncated: self.t == self.horizon && self.truncated,
        })
    }
}
struct Counted {
    values: Cell<usize>,
    actions: Cell<usize>,
    revision: Cell<f32>,
    baseline: bool,
    space: ActionSpace,
}
impl Counted {
    fn new(baseline: bool) -> Self {
        Self {
            values: Cell::new(0),
            actions: Cell::new(0),
            revision: Cell::new(0.),
            baseline,
            space: ActionSpace::Discrete { count: 1 },
        }
    }
}
impl Actor for Counted {
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
        self.actions.set(self.actions.get() + obs.len());
        out.resize(obs.len(), Action::Discrete(0));
        Ok(())
    }
}
impl RolloutActor for Counted {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        b: &mut RolloutBehavior,
    ) -> Result<()> {
        self.act_into(obs, ActMode::Sample, rng, &mut b.actions)?;
        self.values_into(obs, &mut b.values)?;
        b.log_probs.resize(obs.len(), 0.);
        b.latents.resize(obs.len(), None);
        Ok(())
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        self.values.set(self.values.get() + obs.len());
        *out = obs
            .iter()
            .map(|o| o.dense_values()[0] + self.revision.get())
            .collect();
        Ok(())
    }
}
impl EpisodeActor for Counted {
    fn sample_episode_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        b: &mut EpisodeBehavior,
    ) -> Result<()> {
        self.act_into(obs, ActMode::Sample, rng, &mut b.actions)?;
        b.latents.resize(obs.len(), None);
        if self.baseline {
            self.values_into(obs, b.baseline_values.get_or_insert_with(Vec::new))?
        } else {
            b.baseline_values = None
        };
        Ok(())
    }
}
fn env(horizon: usize, truncated: bool) -> Clock {
    Clock {
        t: 0,
        horizon,
        truncated,
        space: ActionSpace::Discrete { count: 1 },
    }
}
#[test]
fn continuing_values_are_evaluated_once_per_current_state_and_fragment_tail() {
    let actor = Counted::new(false);
    let mut c =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env(10, false); 2]).unwrap(), 3).unwrap();
    let mut r = RolloutBuffer::default();
    c.collect_steps_into(&actor, 3, &mut r).unwrap();
    assert_eq!(actor.actions.get(), 6);
    assert_eq!(actor.values.get(), 8);
    assert_eq!(r.next_value(0), Some(1.));
    let b = r.to_batch(0.5, Some(1.), false).unwrap();
    assert_eq!(b.returns, vec![2.125, 2.125, 2.25, 2.25, 2.5, 2.5]);
    actor.revision.set(100.);
    c.collect_steps_into(&actor, 1, &mut r).unwrap();
    let b = r.to_batch(0.5, Some(0.), false).unwrap();
    assert_eq!(b.bootstrapped().unwrap().values, vec![103., 103.]);
    assert_eq!(b.returns, vec![53., 53.]);
}
#[test]
fn terminal_states_need_no_bootstrap_and_truncated_tails_are_evaluated_once() {
    let actor = Counted::new(false);
    let mut c = OnPolicyCollector::new(
        SyncVectorEnv::new(vec![env(1, false), env(1, true)]).unwrap(),
        3,
    )
    .unwrap();
    let mut r = RolloutBuffer::default();
    let d = c.collect_steps_into(&actor, 3, &mut r).unwrap();
    assert_eq!(d.episodes, 6);
    assert_eq!(actor.actions.get(), 6);
    assert_eq!(actor.values.get(), 9);

    assert_eq!(
        r.row(2).unwrap().transition.observation.dense_values(),
        vec![0.]
    );
    assert_eq!(
        r.row(1).unwrap().transition.next_observation.dense_values(),
        vec![1.]
    );
    assert_eq!(
        r.to_batch(0.5, Some(1.), false).unwrap().returns,
        vec![1., 1.5, 1., 1.5, 1., 1.5]
    );
}
#[test]
fn episodic_inactive_lanes_and_optional_baseline_do_only_requested_work() {
    for baseline in [false, true] {
        let actor = Counted::new(baseline);
        let mut c = OnPolicyCollector::new(
            SyncVectorEnv::new(vec![env(1, false), env(3, false)]).unwrap(),
            3,
        )
        .unwrap();
        let mut r = RolloutBuffer::default();
        let d = c.collect_episodes_into(&actor, 3, &mut r).unwrap();
        assert_eq!(d.env_steps, 4);
        assert_eq!(actor.actions.get(), 4);
        assert_eq!(actor.values.get(), if baseline { 4 } else { 0 });
        let b = r.to_batch(1., None, false).unwrap();
        assert_eq!(b.returns, vec![1., 3., 2., 1.]);
        assert_eq!(b.baseline_values().unwrap().is_some(), baseline);
        assert!(b.bootstrapped().is_err());
    }
}
#[test]
fn malformed_behavior_never_advances_environment() {
    struct Bad(Counted);
    impl Actor for Bad {
        fn action_space(&self) -> &ActionSpace {
            self.0.action_space()
        }
        fn act_into(
            &self,
            o: ObservationBatch<'_>,
            m: ActMode,
            r: &mut ChaCha8Rng,
            a: &mut Vec<Action>,
        ) -> Result<()> {
            self.0.act_into(o, m, r, a)
        }
    }
    impl RolloutActor for Bad {
        fn sample_rollout_into(
            &self,
            o: ObservationBatch<'_>,
            r: &mut ChaCha8Rng,
            b: &mut RolloutBehavior,
        ) -> Result<()> {
            self.0.sample_rollout_into(o, r, b)?;
            b.log_probs.clear();
            Ok(())
        }
        fn values_into(&self, o: ObservationBatch<'_>, v: &mut Vec<f32>) -> Result<()> {
            self.0.values_into(o, v)
        }
    }
    let mut c =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env(3, false)]).unwrap(), 3).unwrap();
    let mut r = RolloutBuffer::default();
    assert!(
        c.collect_steps_into(&Bad(Counted::new(false)), 1, &mut r)
            .is_err()
    );
    assert!(r.is_empty());
    assert_eq!(c.environment().environments()[0].t, 0);
}
#[test]
#[cfg(feature = "neural")]
fn flattening_transfers_observation_action_and_latent_buffers() {
    use rand::SeedableRng;
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let space = ActionSpace::Box {
        low: vec![-1.],
        high: vec![1.],
    };
    let models = ActorCritic {
        policy: Policy::new(1, space.clone(), &[], &mut rng).unwrap(),
        value: Value::new(1, &[], &mut rng).unwrap(),
    };
    let mut env = env(3, false);
    env.space = space;
    let mut c = OnPolicyCollector::new(SyncVectorEnv::new(vec![env]).unwrap(), 3).unwrap();
    let mut r = RolloutBuffer::default();
    c.collect_steps_into(&models, 2, &mut r).unwrap();
    let row = r.row(0).unwrap();
    let observation_ptr = row.transition.observation.dense_values().as_ptr();
    let Action::Continuous(a) = &row.transition.action else {
        panic!()
    };
    let action_ptr = a.as_ptr();
    let latent_ptr = row.latent.as_ref().unwrap().as_ptr();
    let b = r.to_batch(0.9, Some(0.95), false).unwrap();
    let Action::Continuous(a) = &b.actions[0] else {
        panic!()
    };
    assert_eq!(observation_ptr, b.observations[0].dense_values().as_ptr());
    assert_eq!(action_ptr, a.as_ptr());
    assert_eq!(latent_ptr, b.latents[0].as_ref().unwrap().as_ptr());
    assert!(r.is_empty());
}
#[test]
fn episode_guard_is_a_safety_limit_not_an_upfront_storage_request() {
    let actor = Counted::new(false);
    let mut c =
        OnPolicyCollector::new(SyncVectorEnv::new(vec![env(1, false); 2]).unwrap(), 3).unwrap();
    let mut r = RolloutBuffer::default();
    let d = c.collect_episodes_into(&actor, usize::MAX, &mut r).unwrap();
    assert_eq!(d.env_steps, 2);
    assert_eq!(r.valid_len(), 2);
}
