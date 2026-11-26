use rl::{
    Result,
    collect::{OnPolicyCollector, vector_env::VectorEnv},
    core::{
        action::{Action, ActionSpace},
        env::{EpisodeStart, Observation, ObservationSpace, Step},
    },
};

#[derive(Clone)]
struct BrokenReset {
    entries: usize,
    width: usize,
    space: ActionSpace,
}
impl VectorEnv for BrokenReset {
    fn len(&self) -> usize {
        2
    }
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, _: &[bool], _: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
        Ok((0..self.entries)
            .map(|_| Some(EpisodeStart::fresh(Observation::new(vec![0.; self.width]))))
            .collect())
    }
    fn step(&mut self, _: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
        unreachable!()
    }
}
#[test]
fn initial_reset_cardinality_is_checked_before_lane_state_is_created() {
    for entries in [1, 3] {
        let env = BrokenReset {
            entries,
            width: 1,
            space: ActionSpace::Discrete { count: 1 },
        };
        assert!(
            OnPolicyCollector::new(env, 3).is_err(),
            "reset length {entries} accepted for two lanes"
        );
    }
}
#[test]
fn custom_reset_observation_shape_is_validated() {
    let env = BrokenReset {
        entries: 2,
        width: 2,
        space: ActionSpace::Discrete { count: 1 },
    };
    assert_eq!(env.dense_observation_dim(), Some(1));
    assert_eq!(env.observation_space().require_dense_width().unwrap(), 1);
    assert!(OnPolicyCollector::new(env, 3).is_err());
}

use rand_chacha::ChaCha8Rng;
use rl::prelude::{
    ActMode, Actor, EpisodeActor, EpisodeBehavior, ObservationBatch, OffPolicyCollector,
    OnPolicyCollector as OnCollector, ReplayBuffer, ReplayWriter, RolloutBuffer,
};
#[derive(Clone, Copy)]
enum Fault {
    None,
    MissingReset,
    ExtraReset,
    MissingStep,
    InactiveStep,
    Width,
    Reward,
    LiveMask,
    TruncatedMask,
    TerminalMask,
    Partial,
}
struct Adversarial {
    fault: Fault,
    space: ActionSpace,
    t: [usize; 2],
    resets: usize,
}
impl Adversarial {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            space: ActionSpace::Discrete { count: 1 },
            t: [0; 2],
            resets: 0,
        }
    }
}
impl VectorEnv for Adversarial {
    fn len(&self) -> usize {
        2
    }
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, m: &[bool], _: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
        self.resets += 1;
        let mut out = vec![None; 2];
        for i in 0..2 {
            if m[i] {
                self.t[i] = 0;
                out[i] = Some(EpisodeStart::fresh(Observation::new(vec![0.])));
            }
        }
        if self.resets > 1 && !m.iter().all(|x| *x) {
            match self.fault {
                Fault::MissingReset => out[0] = None,
                Fault::ExtraReset => out[1] = Some(EpisodeStart::fresh(Observation::new(vec![0.]))),
                _ => {}
            }
        }
        Ok(out)
    }
    fn step(&mut self, a: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
        let mut out = vec![None; 2];
        for i in 0..2 {
            if a[i].is_some() {
                self.t[i] += 1;
                out[i] = Some(Step {
                    observation: Observation::new(vec![self.t[i] as f32]),
                    reward: 1.0_f32.into(),
                    terminated: self.t[i] == if i == 0 { 1 } else { 3 },
                    truncated: false,
                });
            }
        }
        match self.fault {
            Fault::MissingStep => out[0] = None,
            Fault::InactiveStep if a[0].is_none() => {
                out[0] = Some(Step {
                    observation: Observation::new(vec![8.]),
                    reward: 1.0_f32.into(),
                    terminated: false,
                    truncated: false,
                })
            }
            Fault::Width => out[0]
                .as_mut()
                .unwrap()
                .observation
                .as_dense_mut()
                .unwrap()
                .values
                .push(3.),
            Fault::Reward => out[0].as_mut().unwrap().reward = f32::NAN.into(),
            Fault::LiveMask | Fault::TruncatedMask | Fault::TerminalMask => {
                let s = out[0].as_mut().unwrap();
                s.observation.as_dense_mut().unwrap().action_mask = Some(vec![false]);
                s.terminated = matches!(self.fault, Fault::TerminalMask);
                s.truncated = matches!(self.fault, Fault::TruncatedMask);
            }
            Fault::Partial => {
                self.fault = Fault::None;
                return Err(rl::Error::State("advanced then failed".into()));
            }
            _ => {}
        }
        Ok(out)
    }
}
struct Fixed(ActionSpace);
impl Actor for Fixed {
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
        a.resize(o.len(), Action::Discrete(0));
        Ok(())
    }
}
impl EpisodeActor for Fixed {
    fn sample_episode_into(
        &self,
        o: ObservationBatch<'_>,
        r: &mut ChaCha8Rng,
        b: &mut EpisodeBehavior,
    ) -> Result<()> {
        self.act_into(o, ActMode::Sample, r, &mut b.actions)?;
        b.latents.resize(o.len(), None);
        b.baseline_values = None;
        Ok(())
    }
}
#[test]
fn malformed_step_and_selective_reset_batches_are_structured_errors() {
    for fault in [
        Fault::MissingReset,
        Fault::ExtraReset,
        Fault::MissingStep,
        Fault::Width,
        Fault::Reward,
        Fault::LiveMask,
        Fault::TruncatedMask,
    ] {
        let mut c = OffPolicyCollector::new(Adversarial::new(fault), 1).unwrap();
        let mut replay = ReplayBuffer::new(16, 1, ActionSpace::Discrete { count: 1 }).unwrap();
        let mut w = ReplayWriter::new(&mut replay, None, 0.9, 2).unwrap();
        let e = c
            .collect_into(&Fixed(ActionSpace::Discrete { count: 1 }), 2, &mut w)
            .unwrap_err();
        assert!(matches!(e, rl::Error::VectorEnvContract { .. }), "{e:?}");
    }
    let mut c = OnCollector::new(Adversarial::new(Fault::InactiveStep), 1).unwrap();
    let mut b = RolloutBuffer::default();
    assert!(matches!(
        c.collect_episodes_into(&Fixed(ActionSpace::Discrete { count: 1 }), 3, &mut b),
        Err(rl::Error::VectorEnvContract {
            operation: "step",
            lane: Some(0),
            ..
        })
    ));
    assert!(b.is_empty());
}
#[test]
fn terminal_empty_mask_is_allowed_and_partial_failure_requires_full_reset() {
    for fault in [Fault::TerminalMask, Fault::Partial] {
        let mut c = OffPolicyCollector::new(Adversarial::new(fault), 1).unwrap();
        let mut replay = ReplayBuffer::new(8, 1, ActionSpace::Discrete { count: 1 }).unwrap();
        let actor = Fixed(ActionSpace::Discrete { count: 1 });
        let mut w = ReplayWriter::new(&mut replay, None, 0.9, 2).unwrap();
        if matches!(fault, Fault::Partial) {
            assert!(c.collect_into(&actor, 1, &mut w).is_err());
            assert!(c.collect_into(&actor, 1, &mut w).is_err());
            assert_eq!(c.environment().t, [1, 1]);
            c.reset_all().unwrap();
        }
        c.collect_into(&actor, 1, &mut w).unwrap();
        assert_eq!(c.environment().t, [1, 1]);
    }
}
#[test]
fn reset_cannot_change_declared_shape_before_inference() {
    struct ShapeShift {
        inner: Adversarial,
        width: usize,
        on_reset: usize,
    }
    impl VectorEnv for ShapeShift {
        fn len(&self) -> usize {
            2
        }
        fn observation_space(&self) -> ObservationSpace {
            ObservationSpace::Dense(self.width)
        }
        fn action_space(&self) -> &ActionSpace {
            self.inner.action_space()
        }
        fn reset(&mut self, m: &[bool], s: Option<&[u64]>) -> Result<Vec<Option<EpisodeStart>>> {
            let result = self.inner.reset(m, s)?;
            if self.inner.resets == self.on_reset {
                self.width = 2;
            }
            Ok(result)
        }
        fn step(&mut self, a: &[Option<Action>]) -> Result<Vec<Option<Step>>> {
            self.inner.step(a)
        }
    }
    assert!(matches!(
        OffPolicyCollector::new(
            ShapeShift {
                inner: Adversarial::new(Fault::None),
                width: 1,
                on_reset: 1
            },
            1
        ),
        Err(rl::Error::VectorEnvContract {
            operation: "reset",
            ..
        })
    ));
    let mut c = OffPolicyCollector::new(
        ShapeShift {
            inner: Adversarial::new(Fault::None),
            width: 1,
            on_reset: 2,
        },
        1,
    )
    .unwrap();
    let mut replay = ReplayBuffer::new(8, 1, ActionSpace::Discrete { count: 1 }).unwrap();
    let mut writer = ReplayWriter::new(&mut replay, None, 0.9, 2).unwrap();
    assert!(matches!(
        c.collect_into(&Fixed(ActionSpace::Discrete { count: 1 }), 2, &mut writer),
        Err(rl::Error::VectorEnvContract {
            operation: "reset",
            ..
        })
    ));
    assert_eq!(c.environment().inner.t, [0, 1]);
}
