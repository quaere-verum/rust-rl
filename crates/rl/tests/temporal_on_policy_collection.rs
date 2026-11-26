#![cfg(feature = "neural")]

use burn::{module::Param, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{
        stateful::{InferenceMode, StatefulCollector},
        temporal_on_policy::OnPolicyRecording,
        vector_env::SyncVectorEnv,
    },
    core::{checkpoint::Snapshot, stateful::HistorySpec},
    data::{
        temporal::SequenceConfig,
        temporal_on_policy::{TemporalBatchKind, TemporalRollout},
    },
    nn::{
        device, temporal_actor::TemporalActor, temporal_policy::TemporalPolicy,
        temporal_value::TemporalValue, to_vec,
    },
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
struct Clock {
    time: usize,
    truncate: bool,
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
        self.time = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, a: &Action) -> Result<Step> {
        if self.time == 2 {
            return Err(Error::State("ended".into()));
        }
        self.space.check(a, None)?;
        self.time += 1;
        let mut observation = Observation::new(vec![self.time as f32]);
        if self.time == 2 && !self.truncate && matches!(self.space, ActionSpace::Discrete { .. }) {
            observation.as_dense_mut().unwrap().action_mask = Some(vec![false; 2]);
        }
        Ok(Step {
            observation,
            reward: (self.time as f32).into(),
            terminated: self.time == 2 && !self.truncate,
            truncated: self.time == 2 && self.truncate,
        })
    }
}
impl Snapshot for Clock {
    fn validate_snapshot(&self) -> Result<()> {
        self.space.validate()
    }
}
fn actor(space: ActionSpace, saturate: bool) -> TemporalActor {
    let spec = HistorySpec::new(1, space).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(43);
    let mut policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
    if saturate {
        policy.head.layers[0].weight = Param::from_tensor(Tensor::zeros([3, 1], &device()));
        policy.head.layers[0].bias = Some(Param::from_tensor(Tensor::from_data([40.], &device())));
        policy.log_std = Param::from_tensor(Tensor::from_data([-20.], &device()));
    }
    TemporalActor::new(
        policy,
        Some(TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap()),
        0,
    )
    .unwrap()
}
fn config() -> SequenceConfig {
    SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 1,
    }
}

#[test]
fn collection_preserves_actual_latents_and_successor_value_before_reset() {
    for truncate in [false, true] {
        let space = ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        };
        let behavior = actor(space.clone(), true);
        let mut runtime = StatefulCollector::new(
            SyncVectorEnv::new(vec![Clock {
                time: 0,
                truncate,
                space: space.clone(),
            }])
            .unwrap(),
            behavior.clone(),
            9,
        )
        .unwrap();
        let mut rollout = TemporalRollout::new(128, 1, space, 32, 1 << 20).unwrap();
        for _ in 0..2 {
            runtime
                .tick_recorded(
                    &[true],
                    ActMode::Sample,
                    &mut OnPolicyRecording::new(&mut rollout),
                )
                .unwrap();
        }
        let batch = rollout
            .seal(
                behavior,
                TemporalBatchKind::Bootstrapped,
                0.9,
                Some(0.8),
                false,
                &config(),
            )
            .unwrap();
        assert_eq!(batch.len(), 2);
        for row in &batch.rows {
            assert_eq!(row.latent, Some(vec![40.]));
            assert_eq!(
                batch.sequences.samples[row.sample].steps[row.time].action,
                Action::Continuous(vec![2.])
            );
            assert!(row.log_prob.is_finite());
            let lp = to_vec(
                row.distribution
                    .to_distribution(&device())
                    .unwrap()
                    .log_prob(
                        &[Action::Continuous(vec![2.])],
                        std::slice::from_ref(&row.latent),
                    )
                    .unwrap(),
            )
            .unwrap()[0];
            assert!((lp - row.log_prob).abs() < 1e-5);
        }
        assert_eq!(batch.rows[0].next_value, batch.rows[1].value);
        if truncate {
            assert!(batch.rows[1].next_value.unwrap().is_finite());
        } else {
            assert_eq!(batch.rows[1].next_value, Some(0.));
        }
    }
}

#[test]
fn detailed_batched_collection_matches_scalar_streams_and_terminal_empty_masks() {
    let space = ActionSpace::Discrete { count: 2 };
    let behavior = actor(space.clone(), false);
    let make = |mode| {
        StatefulCollector::new_with_inference_mode(
            SyncVectorEnv::new(vec![
                Clock {
                    time: 0,
                    truncate: false,
                    space: space.clone()
                };
                4
            ])
            .unwrap(),
            behavior.clone(),
            17,
            mode,
        )
        .unwrap()
    };
    let mut scalar = make(InferenceMode::Scalar);
    let mut batched = make(InferenceMode::Batched);
    let mut left = TemporalRollout::new(256, 1, space.clone(), 64, 1 << 20).unwrap();
    let mut right = TemporalRollout::new(256, 1, space, 64, 1 << 20).unwrap();
    for _ in 0..4 {
        scalar
            .tick_recorded(
                &[true; 4],
                ActMode::Sample,
                &mut OnPolicyRecording::new(&mut left),
            )
            .unwrap();
        batched
            .tick_recorded(
                &[true; 4],
                ActMode::Sample,
                &mut OnPolicyRecording::new(&mut right),
            )
            .unwrap();
        for lane in 0..4 {
            assert_eq!(scalar.lane_state(lane), batched.lane_state(lane));
        }
    }
    let a = left
        .seal(
            behavior.clone(),
            TemporalBatchKind::Bootstrapped,
            1.,
            Some(1.),
            false,
            &config(),
        )
        .unwrap();
    let b = right
        .seal(
            behavior,
            TemporalBatchKind::Bootstrapped,
            1.,
            Some(1.),
            false,
            &config(),
        )
        .unwrap();
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.sequences, b.sequences);
    assert!(
        a.rows
            .iter()
            .filter(|r| a.sequences.samples[r.sample].steps[r.time].terminated)
            .all(|r| r.next_value == Some(0.))
    );
}

#[test]
fn partial_collection_restore_preserves_future_details_and_agent_memories() {
    let space = ActionSpace::Discrete { count: 2 };
    let behavior = actor(space.clone(), false);
    let mut runtime = StatefulCollector::new(
        SyncVectorEnv::new(vec![Clock {
            time: 0,
            truncate: true,
            space: space.clone(),
        }])
        .unwrap(),
        behavior.clone(),
        47,
    )
    .unwrap();
    let mut rollout = TemporalRollout::new(128, 1, space, 32, 1 << 20).unwrap();
    runtime
        .tick_recorded(
            &[true],
            ActMode::Sample,
            &mut OnPolicyRecording::new(&mut rollout),
        )
        .unwrap();
    runtime.validate_snapshot().unwrap();
    rollout.validate_snapshot().unwrap();
    let mut restored: StatefulCollector<SyncVectorEnv<Clock>, TemporalActor> =
        serde_json::from_value(serde_json::to_value(&runtime).unwrap()).unwrap();
    let mut other: TemporalRollout =
        serde_json::from_value(serde_json::to_value(&rollout).unwrap()).unwrap();
    for _ in 0..4 {
        runtime
            .tick_recorded(
                &[true],
                ActMode::Sample,
                &mut OnPolicyRecording::new(&mut rollout),
            )
            .unwrap();
        restored
            .tick_recorded(
                &[true],
                ActMode::Sample,
                &mut OnPolicyRecording::new(&mut other),
            )
            .unwrap();
        assert_eq!(runtime.lane_state(0), restored.lane_state(0));
    }
    let a = rollout
        .seal(
            behavior.clone(),
            TemporalBatchKind::Bootstrapped,
            0.99,
            Some(0.95),
            false,
            &config(),
        )
        .unwrap();
    let b = other
        .seal(
            behavior,
            TemporalBatchKind::Bootstrapped,
            0.99,
            Some(0.95),
            false,
            &config(),
        )
        .unwrap();
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.sequences, b.sequences);
}
