use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{Result, prelude::*};
#[derive(Clone)]
struct Trace {
    time: usize,
    horizon: usize,
    truncated: bool,
    space: ActionSpace,
}
impl Env for Trace {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::dense(
            vec![0.],
            Some(vec![true, false]),
        )))
    }
    fn step(&mut self, a: &Action) -> Result<Step> {
        assert_eq!(a, &Action::Discrete(0));
        self.time += 1;
        let ended = self.time == self.horizon;
        Ok(Step {
            observation: Observation::dense(
                vec![self.time as f32],
                Some(vec![!ended || self.truncated, false]),
            ),
            reward: (2. * self.time as f32).into(),
            terminated: ended && !self.truncated,
            truncated: ended && self.truncated,
        })
    }
}
struct Spy {
    space: ActionSpace,
    seen: std::cell::RefCell<Vec<(f32, u64)>>,
}
impl Actor for Spy {
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
        out.clear();
        for o in obs.iter() {
            self.seen
                .borrow_mut()
                .push((o.dense_values()[0], rng.random()));
            out.push(Action::Discrete(0));
        }
        Ok(())
    }
}
fn setup(
    h: usize,
    truncated: bool,
    lanes: usize,
) -> (OffPolicyCollector<SyncVectorEnv<Trace>>, ReplayBuffer, Spy) {
    let space = ActionSpace::Discrete { count: 2 };
    (
        OffPolicyCollector::new(
            SyncVectorEnv::new(vec![
                Trace {
                    time: 0,
                    horizon: h,
                    truncated,
                    space: space.clone()
                };
                lanes
            ])
            .unwrap(),
            9,
        )
        .unwrap(),
        ReplayBuffer::new(5, 1, space.clone()).unwrap(),
        Spy {
            space,
            seen: Default::default(),
        },
    )
}
#[test]
fn direct_replay_collection_preserves_masks_resets_wraparound_and_rng() {
    let (mut c, mut replay, a) = setup(1, false, 2);
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    for _ in 0..2 {
        let _: u64 = rng.random();
    }
    let d = c
        .collect_into(
            &a,
            4,
            &mut ReplayWriter::new(&mut replay, None, 0.5, 2).unwrap(),
        )
        .unwrap();
    assert_eq!((d.env_steps, d.vector_steps, d.episodes), (8, 4, 8));
    assert_eq!(replay.len(), 5);
    for (o, r) in a.seen.borrow().iter() {
        assert_eq!(*o, 0.);
        assert_eq!(*r, rng.random::<u64>());
    }
    let b = replay.sample(32, &mut rng).unwrap();
    for x in b.items {
        assert_eq!(x.transition.reward.total(), 2.);
        assert_eq!(x.discount, 0.);
        assert_eq!(x.transition.observation.dense_values(), vec![0.]);
        assert_eq!(x.transition.next_observation.dense_values(), vec![1.]);
        assert_eq!(
            x.transition.next_observation.dense_mask(),
            Some([false, false].as_slice())
        );
    }
}
#[test]
fn pending_spans_calls_then_flushes_every_lane_without_crossing_resets() {
    for truncated in [false, true] {
        let (mut c, mut replay, a) = setup(2, truncated, 2);
        let mut pending = NStepAccumulator::new(3, 0.5, 2).unwrap();
        c.collect_into(
            &a,
            1,
            &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(replay.len(), 0);
        c.collect_into(
            &a,
            1,
            &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(replay.len(), 4);
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for item in replay.sample(128, &mut rng).unwrap().items {
            assert_eq!(item.transition.reward.total(), 4.);
            assert_eq!(item.transition.next_observation.dense_values(), vec![2.]);
            let d = if truncated {
                if item.transition.observation.dense_values() == vec![0.] {
                    0.25
                } else {
                    0.5
                }
            } else {
                0.
            };
            assert_eq!(item.discount, d);
        }
        c.collect_into(
            &a,
            1,
            &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 2).unwrap(),
        )
        .unwrap();
        assert_eq!(replay.len(), 4);
    }
}
#[test]
fn writer_configuration_is_checked_before_environment_steps() {
    let (mut c, mut replay, a) = setup(3, false, 2);
    let mut pending = NStepAccumulator::new(3, 0.9, 2).unwrap();
    assert!(ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 2).is_err());
    let mut wrong = ReplayBuffer::new(8, 2, a.space.clone()).unwrap();
    assert!(
        c.collect_into(
            &a,
            1,
            &mut ReplayWriter::new(&mut wrong, None, 0.5, 2).unwrap()
        )
        .is_err()
    );
    assert!(a.seen.borrow().is_empty());
}
#[test]
fn explicit_recovery_discards_pending_transitions_before_reset_episodes() {
    let (mut c, mut replay, a) = setup(2, false, 1);
    let mut pending = NStepAccumulator::new(3, 0.5, 1).unwrap();
    c.collect_into(
        &a,
        1,
        &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(replay.len(), 0);
    struct Invalid(ActionSpace);
    impl Actor for Invalid {
        fn action_space(&self) -> &ActionSpace {
            &self.0
        }
        fn act_into(
            &self,
            o: ObservationBatch<'_>,
            _: ActMode,
            _: &mut ChaCha8Rng,
            a: &mut Vec<Action>,
        ) -> Result<()> {
            a.clear();
            a.resize(o.len(), Action::Discrete(1));
            Ok(())
        }
    }
    assert!(
        c.collect_into(
            &Invalid(a.space.clone()),
            1,
            &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 1).unwrap()
        )
        .is_err()
    );
    c.reset_all().unwrap();
    pending.clear();
    c.collect_into(
        &a,
        2,
        &mut ReplayWriter::new(&mut replay, Some(&mut pending), 0.5, 1).unwrap(),
    )
    .unwrap();
    assert_eq!(replay.len(), 2);
    for item in replay
        .sample(16, &mut ChaCha8Rng::seed_from_u64(1))
        .unwrap()
        .items
    {
        assert_eq!(item.transition.reward.total(), 4.);
        assert_eq!(item.discount, 0.);
    }
}
