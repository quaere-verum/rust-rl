#![cfg(feature = "neural")]

use burn::{
    module::Param,
    nn::Linear,
    tensor::{Device, Tensor, TensorData},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{stateful::StatefulCollector, vector_env::SyncVectorEnv},
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
        temporal::{TemporalSink, TemporalTransition},
    },
    nn::{
        device,
        network::Mlp,
        recurrent_actor::RecurrentActor,
        temporal::{GruEncoder, GruQ, RecurrentQ, TemporalEncoder},
        to_vec,
    },
};
use serde::{Deserialize, Serialize};

fn model(spec: &HistorySpec, hidden: usize, seed: u64) -> GruQ {
    let ActionSpace::Discrete { count } = spec.action_space else {
        panic!("test requires discrete actions")
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), hidden, &mut rng).unwrap(),
        Mlp::new(hidden, &[], count, &mut rng).unwrap(),
    )
    .unwrap()
}

fn actor(epsilon: f32, revision: u64) -> RecurrentActor {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 3 }).unwrap();
    let mut q = model(&spec, 3, 17);
    q.head.layers[0].weight = Param::from_tensor(Tensor::zeros([3, 3], &device()));
    q.head.layers[0].bias = Some(Param::from_tensor(Tensor::from_data(
        [1., 99., 2.],
        &device(),
    )));
    RecurrentActor::new(q, spec, epsilon, revision).unwrap()
}

fn masked_observation() -> Observation {
    Observation::dense(vec![0.4], Some(vec![true, false, true]))
}

#[test]
fn assimilates_literal_history_channels_once_and_decides_from_stored_memory() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut q = model(&spec, 1, 23);
    fn affine(weights: Vec<f32>, input: usize) -> Linear {
        Linear {
            weight: Param::from_tensor(Tensor::from_data(
                TensorData::new(weights, [input, 1]),
                &device(),
            )),
            bias: Some(Param::from_tensor(Tensor::zeros([1], &device()))),
        }
    }
    for gate in [
        &mut q.encoder.gru.update_gate,
        &mut q.encoder.gru.reset_gate,
    ] {
        gate.input_transform = affine(vec![0.; 5], 5);
        gate.hidden_transform = affine(vec![0.], 1);
    }
    q.encoder.gru.new_gate.input_transform = affine(vec![0.11, 0.21, 0.31, 0.41, 0.51], 5);
    q.encoder.gru.new_gate.hidden_transform = affine(vec![0.23], 1);
    q.head.layers[0].weight = Param::from_tensor(Tensor::from_data([[-1., 1.]], &device()));
    let actor = RecurrentActor::new(q, spec, 0., 8).unwrap();
    let initial = actor.initial_state().unwrap();
    assert_eq!(initial, [0.]);
    let first = actor
        .assimilate(
            &initial,
            HistoryInput {
                observation: &Observation::new(vec![2.]),
                previous: None,
            },
        )
        .unwrap();
    let expected_first = 0.5 * 0.22_f32.tanh();
    assert!((first[0] - expected_first).abs() < 2e-6);
    let observation = Observation::new(vec![4.]);
    let next = actor
        .assimilate(
            &first,
            HistoryInput {
                observation: &observation,
                previous: Some(PreviousInput {
                    factual: None,
                    action: &Action::Discrete(1),
                    reward: 3.,
                }),
            },
        )
        .unwrap();
    let expected_next = 0.5 * (2.49 + 0.115 * expected_first).tanh() + 0.5 * expected_first;
    assert!((next[0] - expected_next).abs() < 2e-6);
    let plain_device = Device::flex();
    let (_, direct) = actor
        .model()
        .encoder
        .step(
            Tensor::from_data([[4., 0., 1., 3., 1.]], &plain_device),
            Tensor::from_data([[first[0]]], &plain_device),
        )
        .unwrap();
    assert_eq!(next, to_vec(direct).unwrap());
    let saved = serde_json::to_value(&actor).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let original_rng = rng.clone();
    for value in [-100., 100., 4.] {
        let decision = actor
            .decide(
                &next,
                &Observation::new(vec![value]),
                ActMode::Deterministic,
                &mut rng,
            )
            .unwrap();
        assert_eq!(decision.action, Action::Discrete(1));
    }
    assert_eq!(
        serde_json::to_value(&rng).unwrap(),
        serde_json::to_value(&original_rng).unwrap()
    );
    assert_eq!(serde_json::to_value(&actor).unwrap(), saved);
    assert_eq!(initial, [0.]);
    assert_eq!(first, [expected_first]);
}

#[test]
fn seeded_epsilon_greedy_respects_masks_and_deterministic_mode_consumes_no_rng() {
    let stochastic = actor(1., 3);
    let state = stochastic.initial_state().unwrap();
    let observation = masked_observation();
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let mut expected_rng = rng.clone();
    let mut seen = [false; 2];
    for _ in 0..64 {
        let _ = expected_rng.random::<f32>();
        let selected = expected_rng.random_range(0..2);
        seen[selected] = true;
        let decision = stochastic
            .decide(&state, &observation, ActMode::Sample, &mut rng)
            .unwrap();
        assert_eq!(decision.action, Action::Discrete([0, 2][selected]));
    }
    assert_eq!(seen, [true, true]);
    assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
    let before = rng.clone();
    assert_eq!(
        stochastic
            .decide(&state, &observation, ActMode::Deterministic, &mut rng)
            .unwrap()
            .action,
        Action::Discrete(2)
    );
    assert_eq!(
        serde_json::to_value(&rng).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
    let greedy = actor(0., 4);
    assert_eq!(
        greedy
            .decide(&state, &observation, ActMode::Sample, &mut rng)
            .unwrap()
            .action,
        Action::Discrete(2)
    );
    assert_eq!(
        serde_json::to_value(&rng).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
    let mut none = observation;
    none.as_dense_mut().unwrap().action_mask = Some(vec![false; 3]);
    assert!(
        stochastic
            .assimilate(
                &state,
                HistoryInput {
                    observation: &none,
                    previous: None
                }
            )
            .is_ok()
    );
    assert!(
        stochastic
            .decide(&state, &none, ActMode::Sample, &mut rng)
            .is_err()
    );
    assert_eq!(
        serde_json::to_value(&rng).unwrap(),
        serde_json::to_value(&before).unwrap()
    );
}

#[test]
fn snapshot_and_clone_keep_frozen_weights_and_continue_the_same_history() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let mut q = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 3, &mut rng).unwrap(),
        Mlp::new(3, &[4, 2], 2, &mut rng).unwrap(),
    )
    .unwrap();
    q.encoder.gru.reset_after = false;
    q.encoder.gru.clip = Some(0.7);
    q.encoder.gru.new_gate.hidden_transform.bias = None;
    q.head.layers[1].bias = None;
    let actor = RecurrentActor::new(q.clone(), spec, 0.35, 7).unwrap();
    let bytes = serde_json::to_vec(&actor).unwrap();
    let restored: RecurrentActor = serde_json::from_slice(&bytes).unwrap();
    restored.validate_snapshot().unwrap();
    assert_eq!(restored.epsilon(), 0.35);
    assert_eq!(restored.revision(), 7);
    assert_eq!(restored.spec(), actor.spec());
    assert!(!restored.model().encoder.gru.reset_after);
    assert_eq!(restored.model().encoder.gru.clip, Some(0.7));
    assert!(
        restored
            .model()
            .encoder
            .gru
            .new_gate
            .hidden_transform
            .bias
            .is_none()
    );
    assert!(restored.model().head.layers[1].bias.is_none());
    let clone = actor.clone();
    q.encoder.gru.new_gate.input_transform.weight =
        Param::from_tensor(Tensor::zeros([5, 3], &device()));
    let input = HistoryInput {
        observation: &Observation::new(vec![0.3]),
        previous: None,
    };
    let state = actor
        .assimilate(&actor.initial_state().unwrap(), input)
        .unwrap();
    for frozen in [&actor, &restored, &clone] {
        assert!(
            !frozen
                .model()
                .encoder
                .gru
                .new_gate
                .input_transform
                .weight
                .val()
                .is_autodiff()
        );
        assert!(!frozen.model().head.layers[0].weight.val().is_autodiff());
        assert_eq!(
            state,
            frozen
                .assimilate(&frozen.initial_state().unwrap(), input)
                .unwrap()
        );
    }
    assert_eq!(serde_json::to_vec(&restored).unwrap(), bytes);
}

#[test]
fn rejects_malformed_states_history_model_and_snapshot_metadata() {
    let actor = actor(0.2, 1);
    for state in [
        vec![],
        vec![0.; 4],
        vec![f32::NAN; 3],
        vec![f32::INFINITY; 3],
    ] {
        assert!(actor.validate_state(&state).is_err());
        assert!(
            actor
                .assimilate(
                    &state,
                    HistoryInput {
                        observation: &masked_observation(),
                        previous: None
                    }
                )
                .is_err()
        );
        assert!(
            actor
                .decide(
                    &state,
                    &masked_observation(),
                    ActMode::Sample,
                    &mut ChaCha8Rng::seed_from_u64(9)
                )
                .is_err()
        );
    }
    assert!(
        actor
            .assimilate(
                &actor.initial_state().unwrap(),
                HistoryInput {
                    observation: &masked_observation(),
                    previous: Some(PreviousInput {
                        factual: None,
                        action: &Action::Discrete(99),
                        reward: 0.
                    })
                }
            )
            .is_err()
    );
    assert!(
        actor
            .assimilate(
                &actor.initial_state().unwrap(),
                HistoryInput {
                    observation: &masked_observation(),
                    previous: Some(PreviousInput {
                        factual: None,
                        action: &Action::Discrete(0),
                        reward: f32::NAN
                    })
                }
            )
            .is_err()
    );
    let original = serde_json::to_value(&actor).unwrap();
    for (field, value) in [
        ("version", serde_json::json!(2)),
        ("epsilon", serde_json::json!(-0.1)),
        ("hidden_dim", serde_json::json!(0)),
        ("hidden_dim", serde_json::json!(1_000_000_000_usize)),
        ("head_hidden", serde_json::json!([0])),
    ] {
        let mut corrupted = original.clone();
        corrupted["metadata"][field] = value;
        assert!(
            serde_json::from_value::<RecurrentActor>(corrupted).is_err(),
            "accepted corrupt {field}"
        );
    }
    let mut corrupted = original.clone();
    corrupted["models"]["q"]["signature"] = serde_json::json!("wrong");
    assert!(serde_json::from_value::<RecurrentActor>(corrupted).is_err());
    let mut corrupted = original.clone();
    *corrupted["models"]["q"]["parameter_shapes"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap() = serde_json::json!([999]);
    assert!(serde_json::from_value::<RecurrentActor>(corrupted).is_err());
    let mut corrupted = original;
    corrupted["models"]["q"]["bytes"] = serde_json::json!([0, 1, 2]);
    assert!(serde_json::from_value::<RecurrentActor>(corrupted).is_err());
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    assert!(RecurrentActor::new(model(&spec, 2, 9), spec.clone(), 1.1, 0).is_err());
    let mut malformed = model(&spec, 2, 9);
    malformed.head.layers[0].bias = Some(Param::from_tensor(Tensor::from_data(
        [f32::NAN, 0.],
        &device(),
    )));
    assert!(RecurrentActor::new(malformed, spec.clone(), 0., 0).is_err());
    let mut malformed = model(&spec, 2, 9);
    malformed.head.layers[0].weight = Param::from_tensor(Tensor::zeros([3, 2], &device()));
    assert!(RecurrentActor::new(malformed, spec, 0., 0).is_err());
}

#[derive(Clone, Serialize, Deserialize)]
struct Clock {
    tick: usize,
    limit: usize,
}
impl Env for Clock {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 3 }
    }
    fn reset(&mut self, _: Option<u64>) -> rl::Result<EpisodeStart> {
        self.tick = 0;
        Ok(EpisodeStart::fresh(Observation::new(vec![0.])))
    }
    fn step(&mut self, action: &Action) -> rl::Result<Step> {
        let Action::Discrete(action) = action else {
            panic!("discrete test clock")
        };
        self.tick += 1;
        Ok(Step {
            observation: Observation::new(vec![self.tick as f32]),
            reward: (self.tick as f32 + *action as f32).into(),
            terminated: self.tick == self.limit,
            truncated: false,
        })
    }
}
impl Snapshot for Clock {
    fn validate_snapshot(&self) -> rl::Result<()> {
        assert!(self.tick <= self.limit);
        Ok(())
    }
}
#[derive(Clone, Default)]
struct Sink(Vec<TemporalTransition>);
impl TemporalSink for Sink {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> rl::Result<()> {
        Ok(())
    }
    fn push(&mut self, transition: &TemporalTransition) -> rl::Result<()> {
        self.0.push(transition.clone());
        Ok(())
    }
}

#[test]
fn actor_and_lane_memory_roundtrip_inside_the_existing_stateful_runtime() {
    type Runtime = StatefulCollector<SyncVectorEnv<Clock>, RecurrentActor>;
    let env = SyncVectorEnv::new(vec![
        Clock { tick: 0, limit: 4 },
        Clock { tick: 0, limit: 2 },
    ])
    .unwrap();
    let mut runtime = Runtime::new(env, actor(0.8, 1), 53).unwrap();
    let mut sink = Sink::default();
    runtime.collect(1, ActMode::Sample, &mut sink).unwrap();
    runtime.publish(actor(0.4, 2)).unwrap();
    runtime.collect(1, ActMode::Sample, &mut sink).unwrap();
    let snapshot = serde_json::to_value(&runtime).unwrap();
    let mut restored: Runtime = serde_json::from_value(snapshot.clone()).unwrap();
    restored.validate_snapshot().unwrap();
    let mut other_sink = sink.clone();
    for _ in 0..7 {
        runtime.collect(1, ActMode::Sample, &mut sink).unwrap();
        restored
            .collect(1, ActMode::Sample, &mut other_sink)
            .unwrap();
    }
    assert_eq!(sink.0, other_sink.0);
    assert_eq!(
        serde_json::to_value(&runtime).unwrap(),
        serde_json::to_value(&restored).unwrap()
    );
    let mut malformed = snapshot;
    malformed["agents"][0]["state"] = serde_json::json!([0.]);
    let malformed: Runtime = serde_json::from_value(malformed).unwrap();
    assert!(malformed.validate_snapshot().is_err());
}
