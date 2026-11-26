#[cfg(feature = "neural")]
use burn::{
    module::Param,
    tensor::{Tensor, TensorData},
};
use rand::Rng;
#[cfg(feature = "neural")]
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    collect::{
        stateful::{InferenceMode, StatefulCollector},
        vector_env::SyncVectorEnv,
    },
    core::{
        stateful::{HistoryInput, StatefulActor},
        temporal::{TemporalSink, TemporalTransition},
    },
    envs::delayed_bit::DelayedBit,
    prelude::*,
};
#[cfg(feature = "neural")]
#[path = "support/resume_fixture.rs"]
mod resume_fixture;

#[cfg(feature = "neural")]
use resume_fixture::ResumeFixture;
#[cfg(feature = "neural")]
use rl::{
    core::stateful::HistorySpec,
    nn::{
        device,
        network::Mlp,
        recurrent_actor::RecurrentActor,
        temporal::{GruEncoder, GruQ, RecurrentQ},
    },
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "neural")]
use std::{cell::Cell, rc::Rc};

#[cfg(feature = "neural")]
fn model(seed: u64) -> GruQ {
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    // Legal greedy decisions have a fixed, large margin; hidden states still
    // exercise the seeded real GRU and canonical history input.
    model.head.layers[1].weight = Param::from_tensor(Tensor::zeros([4, 2], &device()));
    model.head.layers[1].bias = Some(Param::from_tensor(Tensor::from_data([1., 2.], &device())));
    model
}
#[cfg(feature = "neural")]
fn actor(seed: u64, epsilon: f32, revision: u64) -> RecurrentActor {
    RecurrentActor::new(
        model(seed),
        HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap(),
        epsilon,
        revision,
    )
    .unwrap()
}

#[test]
#[cfg(feature = "neural")]
fn actor_clone_shares_frozen_weights_without_sharing_training_mutations() {
    let mut trainable = model(17);
    let actor = RecurrentActor::new(
        trainable.clone(),
        HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap(),
        0.4,
        3,
    )
    .unwrap();
    let cloned = actor.clone();
    assert!(
        std::ptr::eq(actor.model(), cloned.model()),
        "cloning a pinned actor must share its immutable model allocation"
    );
    let observation = Observation::new(vec![0.2, 1., 0., 0., 0.]);
    let input = HistoryInput {
        observation: &observation,
        previous: None,
    };
    let before = actor
        .assimilate(&actor.initial_state().unwrap(), input)
        .unwrap();
    trainable.encoder.gru.new_gate.input_transform.weight = Param::from_tensor(Tensor::from_data(
        TensorData::new(vec![0.; 36], [9, 4]),
        &device(),
    ));
    assert_eq!(
        before,
        cloned
            .assimilate(&cloned.initial_state().unwrap(), input)
            .unwrap()
    );
}

#[test]
#[cfg(feature = "neural")]
fn behavior_settings_change_without_copying_or_changing_frozen_representation() {
    let original = actor(17, 0.35, 3);
    let before = serde_json::to_value(&original).unwrap();
    let published = original.with_behavior(0.8, 11).unwrap();
    assert!(std::ptr::eq(original.model(), published.model()));
    assert_eq!(original.model_identity(), published.model_identity());
    assert_eq!(original.spec(), published.spec());
    assert_eq!(published.epsilon(), 0.8);
    assert_eq!(published.revision(), 11);
    assert!(!original.batch_compatible(&published));
    assert_eq!(
        before["models"],
        serde_json::to_value(&published).unwrap()["models"]
    );
    for invalid in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(original.with_behavior(invalid, 4).is_err());
    }
    assert_eq!(before, serde_json::to_value(&original).unwrap());
}

#[derive(Clone, Default)]
struct Tape(Vec<TemporalTransition>);
impl TemporalSink for Tape {
    fn begin_episode(&mut self, _: u64, _: &EpisodeStart) -> Result<()> {
        Ok(())
    }
    fn push(&mut self, t: &TemporalTransition) -> Result<()> {
        self.0.push(t.clone());
        Ok(())
    }
}
#[cfg(feature = "neural")]
type Runtime = StatefulCollector<SyncVectorEnv<DelayedBit>, RecurrentActor>;
fn environments() -> SyncVectorEnv<DelayedBit> {
    SyncVectorEnv::new(
        [3, 5, 7]
            .into_iter()
            .enumerate()
            .map(|(lane, delay)| DelayedBit::new(delay, Some(2), false, 501 + lane as u64).unwrap())
            .collect(),
    )
    .unwrap()
}
#[cfg(feature = "neural")]
fn runtime(mode: InferenceMode) -> Runtime {
    StatefulCollector::new_with_inference_mode(environments(), actor(17, 0.35, 0), 73, mode)
        .unwrap()
}
#[cfg(feature = "neural")]
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "hidden element {index}: {actual} != {expected}"
        );
    }
}
#[cfg(feature = "neural")]
fn assert_lanes(a: &Runtime, b: &Runtime) {
    assert_eq!(a.observations(), b.observations());
    for lane in 0..a.len() {
        assert_eq!(a.lane_revision(lane), b.lane_revision(lane));
        assert_eq!(a.lane_episode(lane), b.lane_episode(lane));
        assert_eq!(a.lane_timestep(lane), b.lane_timestep(lane));
        assert_eq!(a.lane_needs_reset(lane), b.lane_needs_reset(lane));
        close(a.lane_state(lane).unwrap(), b.lane_state(lane).unwrap());
    }
    let a = serde_json::to_value(a).unwrap();
    let b = serde_json::to_value(b).unwrap();
    for lane in 0..a["agents"].as_array().unwrap().len() {
        assert_eq!(a["agents"][lane]["rng"], b["agents"][lane]["rng"]);
    }
}

#[test]
#[cfg(feature = "neural")]
fn model_content_identity_survives_restore_and_rejects_revision_only_compatibility() {
    let original = actor(17, 0.35, 3);
    let restored: RecurrentActor =
        serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    assert_eq!(original.model_identity(), restored.model_identity());
    assert!(original.batch_compatible(&restored));
    let different_model = actor(19, 0.35, 3);
    assert_ne!(original.model_identity(), different_model.model_identity());
    assert!(!original.batch_compatible(&different_model));
    let different_epsilon = actor(17, 0.65, 3);
    assert_eq!(
        original.model_identity(),
        different_epsilon.model_identity()
    );
    assert!(!original.batch_compatible(&different_epsilon));
    assert!(!original.batch_compatible(&actor(17, 0.35, 4)));
    let mut controls = model(17);
    controls.encoder.gru.reset_after = false;
    let controls = RecurrentActor::new(controls, original.spec().clone(), 0.35, 3).unwrap();
    assert_ne!(original.model_identity(), controls.model_identity());
    assert!(!original.batch_compatible(&controls));
}

#[test]
#[cfg(feature = "neural")]
fn direct_batched_neural_operations_preserve_masks_and_independent_rng_streams() {
    let mut q = model(17);
    q.head = Mlp::new(4, &[], 2, &mut ChaCha8Rng::seed_from_u64(29)).unwrap();
    q.head.layers[0].weight = Param::from_tensor(Tensor::from_data(
        [[2., -2.], [-2., 2.], [0., 0.], [0., 0.]],
        &device(),
    ));
    let actor = RecurrentActor::new(
        q,
        HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap(),
        0.4,
        0,
    )
    .unwrap();
    let observations = [
        Observation::dense(vec![1., 1., 0., 0., 0.], Some(vec![true, true])),
        Observation::dense(vec![-1., 0., 1., 0., 1.], Some(vec![true, true])),
        Observation::dense(vec![0., 0., 1., 1., 1.], Some(vec![false, true])),
    ];
    let priors = [
        vec![0.; 4],
        vec![0.1, -0.2, 0.3, -0.4],
        vec![0.4, 0.3, -0.2, -0.1],
    ];
    let actions = [
        Action::Discrete(0),
        Action::Discrete(1),
        Action::Discrete(0),
    ];
    let inputs = observations
        .iter()
        .enumerate()
        .map(|(lane, observation)| HistoryInput {
            observation,
            previous: Some(rl::core::stateful::PreviousInput {
                factual: None,
                action: &actions[lane],
                reward: lane as f32 * 0.2,
            }),
        })
        .collect::<Vec<_>>();
    let scalar = priors
        .iter()
        .zip(&inputs)
        .map(|(prior, input)| actor.assimilate(prior, *input).unwrap())
        .collect::<Vec<_>>();
    let batched = actor
        .assimilate_batch(&priors.iter().collect::<Vec<_>>(), &inputs)
        .unwrap();
    for (a, b) in scalar.iter().zip(&batched) {
        close(a, b);
    }
    let mut changed_priors = priors.clone();
    changed_priors[1][0] += 0.75;
    let changed = actor
        .assimilate_batch(&changed_priors.iter().collect::<Vec<_>>(), &inputs)
        .unwrap();
    assert_eq!(changed[0], batched[0]);
    assert_eq!(changed[2], batched[2]);
    assert_ne!(changed[1], batched[1]);
    // Nonconstant literal Q rows: [2,-2], [-2,2], [4,-4]. The last
    // observation masks out its greedy action, so expected actions are 0,1,1.
    let decision_states = [
        vec![1., 0., 0., 0.],
        vec![0., 1., 0., 0.],
        vec![2., 0., 0., 0.],
    ];
    let mut scalar_rngs = (31..34).map(ChaCha8Rng::seed_from_u64).collect::<Vec<_>>();
    let mut batched_rngs = scalar_rngs.clone();
    let refs = observations.iter().collect::<Vec<_>>();
    let states = decision_states.iter().collect::<Vec<_>>();
    for mode in [ActMode::Sample, ActMode::Deterministic, ActMode::Sample] {
        let expected = decision_states
            .iter()
            .zip(&observations)
            .zip(&mut scalar_rngs)
            .map(|((state, observation), rng)| {
                actor.decide(state, observation, mode, rng).unwrap().action
            })
            .collect::<Vec<_>>();
        if mode == ActMode::Deterministic {
            assert_eq!(
                expected,
                vec![
                    Action::Discrete(0),
                    Action::Discrete(1),
                    Action::Discrete(1)
                ]
            );
        }
        let actual = actor
            .decide_batch(&states, &refs, mode, &mut batched_rngs)
            .unwrap();
        assert_eq!(
            actual.into_iter().map(|d| d.action).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            serde_json::to_value(&scalar_rngs).unwrap(),
            serde_json::to_value(&batched_rngs).unwrap()
        );
    }
    assert!(
        actor
            .assimilate_batch(&priors.iter().collect::<Vec<_>>(), &inputs[..2])
            .is_err()
    );
    let before = serde_json::to_value(&batched_rngs).unwrap();
    assert!(
        actor
            .decide_batch(&states, &refs[..2], ActMode::Sample, &mut batched_rngs)
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&batched_rngs).unwrap());
    let mut invalid = decision_states.clone();
    invalid[1][0] = f32::NAN;
    assert!(
        actor
            .decide_batch(
                &invalid.iter().collect::<Vec<_>>(),
                &refs,
                ActMode::Sample,
                &mut batched_rngs
            )
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&batched_rngs).unwrap());
    let mut invalid_observations = observations.clone();
    invalid_observations[2].as_dense_mut().unwrap().action_mask = Some(vec![false, false]);
    assert!(
        actor
            .decide_batch(
                &states,
                &invalid_observations.iter().collect::<Vec<_>>(),
                ActMode::Sample,
                &mut batched_rngs
            )
            .is_err()
    );
    assert_eq!(before, serde_json::to_value(&batched_rngs).unwrap());
}

#[test]
#[cfg(feature = "neural")]
fn batched_collector_preserves_mixed_revisions_and_staggered_episode_streams() {
    let mut scalar = runtime(InferenceMode::Scalar);
    let mut batch = runtime(InferenceMode::Batched);
    let mut scalar_tape = Tape::default();
    let mut batch_tape = Tape::default();
    for call in 0..36 {
        if call == 3 || call == 13 {
            let candidate = actor(19 + call as u64, 0.65, if call == 3 { 1 } else { 2 });
            scalar.publish(candidate.clone()).unwrap();
            batch.publish(candidate).unwrap();
        }
        let active = match call % 7 {
            1 => [true, false, true],
            3 => [false, true, false],
            _ => [true; 3],
        };
        let a = scalar
            .tick(&active, ActMode::Sample, &mut scalar_tape)
            .unwrap();
        let b = batch
            .tick(&active, ActMode::Sample, &mut batch_tape)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(scalar_tape.0, batch_tape.0);
        assert_lanes(&scalar, &batch);
        scalar.validate_snapshot().unwrap();
        batch.validate_snapshot().unwrap();
    }
    assert!(batch_tape.0.iter().any(|t| t.behavior.revision == 0));
    assert!(batch_tape.0.iter().any(|t| t.behavior.revision == 2));
}

#[test]
#[cfg(feature = "neural")]
fn same_revision_different_parameters_are_not_coalesced_by_collector() {
    let mut snapshot = serde_json::to_value(runtime(InferenceMode::Scalar)).unwrap();
    snapshot["agents"][1]["actor"] = serde_json::to_value(actor(19, 0.35, 0)).unwrap();
    let mut scalar: Runtime = serde_json::from_value(snapshot.clone()).unwrap();
    let mut batch: Runtime = serde_json::from_value(snapshot).unwrap();
    batch.set_inference_mode(InferenceMode::Batched).unwrap();
    let mut a = Tape::default();
    let mut b = Tape::default();
    for _ in 0..12 {
        scalar.collect(1, ActMode::Sample, &mut a).unwrap();
        batch.collect(1, ActMode::Sample, &mut b).unwrap();
        assert_eq!(a.0, b.0);
        assert_lanes(&scalar, &batch);
    }
}

#[test]
#[cfg(feature = "neural")]
fn batched_snapshot_restores_compatible_groups_and_exact_future() {
    let mut uninterrupted = runtime(InferenceMode::Batched);
    let mut a = Tape::default();
    uninterrupted.collect(3, ActMode::Sample, &mut a).unwrap();
    uninterrupted.publish(actor(23, 0.6, 1)).unwrap();
    uninterrupted.collect(2, ActMode::Sample, &mut a).unwrap();
    let mut restored: Runtime =
        serde_json::from_slice(&serde_json::to_vec(&uninterrupted).unwrap()).unwrap();
    assert_eq!(restored.inference_mode(), InferenceMode::Batched);
    let mut b = a.clone();
    for _ in 0..18 {
        uninterrupted.collect(1, ActMode::Sample, &mut a).unwrap();
        restored.collect(1, ActMode::Sample, &mut b).unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(
            serde_json::to_value(&uninterrupted).unwrap(),
            serde_json::to_value(&restored).unwrap()
        );
    }
}

#[test]
#[cfg(feature = "neural")]
fn canonical_groups_keep_scalar_history_and_persist_the_execution_contract() {
    for lanes in [1, 4, 16] {
        let make = |mode| {
            StatefulCollector::new_with_inference_mode(
                SyncVectorEnv::new(
                    (0..lanes)
                        .map(|i| {
                            DelayedBit::new(6 + i % 3, Some(2), false, 100 + i as u64).unwrap()
                        })
                        .collect(),
                )
                .unwrap(),
                actor(17, 0.35, 0),
                73,
                mode,
            )
            .unwrap()
        };
        let mut scalar = make(InferenceMode::Scalar);
        let mut canonical = make(InferenceMode::CanonicalBatched);
        let mut a = Tape::default();
        let mut b = Tape::default();
        for tick in 0..23 {
            if tick == 5 {
                scalar.publish(actor(23, 0.6, 1)).unwrap();
                canonical.publish(actor(23, 0.6, 1)).unwrap();
            }
            let active = (0..scalar.len())
                .map(|i| i == 0 || (i + tick) % 3 != 0)
                .collect::<Vec<_>>();
            scalar.tick(&active, ActMode::Sample, &mut a).unwrap();
            canonical.tick(&active, ActMode::Sample, &mut b).unwrap();
            if tick == 5 && scalar.lane_revision(0) == Some(0) {
                let id = scalar.lane_episode(0).unwrap();
                let tape =
                    a.0.iter()
                        .filter(|t| t.episode_id == id)
                        .collect::<Vec<_>>();
                let mut observations = vec![tape[0].observation.clone()];
                observations.extend(tape.iter().map(|t| t.next_observation.clone()));
                let actions = tape.iter().map(|t| t.action.clone()).collect::<Vec<_>>();
                let rewards = tape.iter().map(|t| t.reward.total()).collect::<Vec<_>>();
                for runtime in [&mut scalar, &mut canonical] {
                    runtime
                        .reconcile_lane(
                            0,
                            rl::core::stateful::ReconstructionHistory {
                                observations: &observations,
                                actions: &actions,
                                rewards: rl::core::stateful::HistoryRewards::Totals(&rewards),
                            },
                        )
                        .unwrap();
                }
            }
            assert_eq!(a.0, b.0);
            assert_lanes(&scalar, &canonical);
            let saved = serde_json::to_value(&canonical).unwrap();
            assert_eq!(saved["version"], 5);
            assert_eq!(saved["inference_mode"], "canonical_batched_v1");
            let restored: Runtime = serde_json::from_value(saved.clone()).unwrap();
            restored.validate_snapshot().unwrap();
            assert_eq!(serde_json::to_value(&restored).unwrap(), saved);
            canonical = restored;
        }
        let before = serde_json::to_value(&canonical).unwrap();
        assert!(canonical.set_inference_mode(InferenceMode::Scalar).is_err());
        assert_eq!(serde_json::to_value(&canonical).unwrap(), before);
        // Entering via legacy->scalar must not bypass constructor-only selection.
        scalar.set_inference_mode(InferenceMode::Batched).unwrap();
        scalar.set_inference_mode(InferenceMode::Scalar).unwrap();
        assert!(
            scalar
                .set_inference_mode(InferenceMode::CanonicalBatched)
                .is_err()
        );
    }
}

#[test]
#[cfg(feature = "neural")]
fn inactive_lanes_keep_memory_and_rng_and_later_consume_their_own_stream() {
    let mut selected = runtime(InferenceMode::Batched);
    let mut direct = runtime(InferenceMode::Batched);
    let before = serde_json::to_value(&selected).unwrap();
    let hidden = selected.lane_state(1).unwrap().clone();
    let mut a = Tape::default();
    let mut b = Tape::default();
    selected
        .tick(&[true, false, true], ActMode::Sample, &mut a)
        .unwrap();
    assert_eq!(selected.lane_state(1).unwrap(), &hidden);
    assert_eq!(
        serde_json::to_value(&selected).unwrap()["agents"][1]["rng"],
        before["agents"][1]["rng"]
    );
    selected
        .tick(&[false, true, false], ActMode::Sample, &mut a)
        .unwrap();
    direct
        .tick(&[false, true, false], ActMode::Sample, &mut b)
        .unwrap();
    assert_eq!(a.0.last(), b.0.last());
    assert_eq!(selected.lane_state(1), direct.lane_state(1));
}

#[derive(Clone, Serialize, Deserialize)]
struct Structured;
impl StatefulActor for Structured {
    type State = (Vec<Vec<f32>>, usize);
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(5)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn revision(&self) -> u64 {
        0
    }
    fn initial_state(&self) -> Result<Self::State> {
        Ok((Vec::new(), 0))
    }
    fn validate_state(&self, _: &Self::State) -> Result<()> {
        Ok(())
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        let mut next = prior.clone();
        next.0.push(input.observation.dense_values().to_vec());
        next.1 += usize::from(input.previous.is_some());
        Ok(next)
    }
    fn decide(
        &self,
        _: &Self::State,
        observation: &Observation,
        _: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<rl::core::stateful::StatefulDecision> {
        let choices = (0..2)
            .filter(|&a| observation.dense_mask().is_none_or(|m| m[a]))
            .collect::<Vec<_>>();
        Ok(rl::core::stateful::StatefulDecision {
            action: Action::Discrete(choices[rng.random_range(0..choices.len())]),
            log_prob: None,
        })
    }
}
#[test]
fn optional_batch_mode_keeps_scalar_fallback_for_arbitrary_typed_states() {
    let mut scalar = StatefulCollector::new(environments(), Structured, 73).unwrap();
    let mut batch = StatefulCollector::new_with_inference_mode(
        environments(),
        Structured,
        73,
        InferenceMode::Batched,
    )
    .unwrap();
    let mut a = Tape::default();
    let mut b = Tape::default();
    for _ in 0..18 {
        scalar.collect(1, ActMode::Sample, &mut a).unwrap();
        batch.collect(1, ActMode::Sample, &mut b).unwrap();
    }
    assert_eq!(a.0, b.0);
    for lane in 0..3 {
        assert_eq!(scalar.lane_state(lane), batch.lane_state(lane));
    }
}

#[derive(Clone)]
#[cfg(feature = "neural")]
struct Counted {
    inner: RecurrentActor,
    decisions: Rc<Cell<usize>>,
    assimilations: Rc<Cell<usize>>,
}
#[cfg(feature = "neural")]
fn require_delayed_bit_previous(input: HistoryInput<'_>) -> Result<()> {
    if (input.observation.dense_values()[1] == 0.) != input.previous.is_some() {
        return Err(rl::Error::State(
            "delayed-bit observation has incorrect previous input".into(),
        ));
    }
    Ok(())
}
#[cfg(feature = "neural")]
impl StatefulActor for Counted {
    type State = Vec<f32>;
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.inner.action_space()
    }
    fn revision(&self) -> u64 {
        self.inner.revision()
    }
    fn initial_state(&self) -> Result<Self::State> {
        self.inner.initial_state()
    }
    fn validate_state(&self, s: &Self::State) -> Result<()> {
        self.inner.validate_state(s)
    }
    fn assimilate(&self, s: &Self::State, i: HistoryInput<'_>) -> Result<Self::State> {
        require_delayed_bit_previous(i)?;
        self.inner.assimilate(s, i)
    }
    fn decide(
        &self,
        s: &Self::State,
        o: &Observation,
        m: ActMode,
        r: &mut ChaCha8Rng,
    ) -> Result<rl::core::stateful::StatefulDecision> {
        self.inner.decide(s, o, m, r)
    }
    fn batch_compatible(&self, other: &Self) -> bool {
        self.inner.batch_compatible(&other.inner)
    }
    fn assimilate_batch(
        &self,
        s: &[&Self::State],
        i: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        for &input in i {
            require_delayed_bit_previous(input)?;
        }
        self.assimilations.set(self.assimilations.get() + 1);
        self.inner.assimilate_batch(s, i)
    }
    fn decide_batch(
        &self,
        s: &[&Self::State],
        o: &[&Observation],
        m: ActMode,
        r: &mut [ChaCha8Rng],
    ) -> Result<Vec<rl::core::stateful::StatefulDecision>> {
        self.decisions.set(self.decisions.get() + 1);
        self.inner.decide_batch(s, o, m, r)
    }
}
#[test]
#[cfg(feature = "neural")]
fn compatible_active_lanes_use_one_batch_callback_per_phase() {
    let decisions = Rc::new(Cell::new(0));
    let assimilations = Rc::new(Cell::new(0));
    let actor = Counted {
        inner: actor(17, 0.4, 0),
        decisions: decisions.clone(),
        assimilations: assimilations.clone(),
    };
    let mut runtime = StatefulCollector::new_with_inference_mode(
        environments(),
        actor,
        73,
        InferenceMode::Batched,
    )
    .unwrap();
    let initialized = assimilations.get();
    let mut tape = Tape::default();
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    assert_eq!(decisions.get(), 1);
    assert_eq!(assimilations.get() - initialized, 1);
    assert_eq!(tape.0.len(), 3);
}

#[test]
#[cfg(feature = "neural")]
fn mixed_checkpointed_resets_only_batch_genuine_reset_inputs() {
    for mode in [InferenceMode::Batched, InferenceMode::CanonicalBatched] {
        let envs = [None, Some(2), None]
            .into_iter()
            .map(|depth| ResumeFixture::new(DelayedBit::new(2, None, false, 17).unwrap(), depth))
            .collect();
        let mut runtime = StatefulCollector::new_with_inference_mode(
            SyncVectorEnv::new(envs).unwrap(),
            Counted {
                inner: actor(17, 0.4, 0),
                decisions: Rc::default(),
                assimilations: Rc::default(),
            },
            73,
            mode,
        )
        .unwrap();
        let mut tape = Tape::default();
        runtime.collect(4, ActMode::Sample, &mut tape).unwrap();
        assert_eq!(runtime.lane_timestep(0), Some(1));
        assert_eq!(runtime.lane_timestep(1), Some(3));
        assert_eq!(runtime.lane_timestep(2), Some(1));
        assert_eq!(tape.0.len(), 12);
    }
}

#[test]
#[cfg(feature = "neural")]
fn corrupt_loaded_runtime_is_audited_before_tick_publish_or_reset_mutation() {
    let snapshot = serde_json::to_value(runtime(InferenceMode::Batched)).unwrap();
    for corruption in 0..5 {
        let mut value = snapshot.clone();
        match corruption {
            0 => value["version"] = 99.into(),
            1 => value["agents"][1]["episode_id"] = 0.into(),
            2 => value["next_episode_id"] = 0.into(),
            3 => value["agents"][1]["state"] = serde_json::json!([]),
            _ => value["driver"]["episode_steps"][1] = 3.into(),
        }
        let mut loaded: Runtime = serde_json::from_value(value).unwrap();
        let before = serde_json::to_value(&loaded).unwrap();
        let mut tape = Tape::default();
        assert!(loaded.validate_local().is_err(), "corruption {corruption}");
        assert!(loaded.tick(&[true; 3], ActMode::Sample, &mut tape).is_err());
        assert!(loaded.publish(actor(19, 0.5, 1)).is_err());
        assert!(loaded.reset_all(&mut tape).is_err());
        assert!(tape.0.is_empty());
        assert_eq!(before, serde_json::to_value(&loaded).unwrap());
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[cfg(feature = "neural")]
struct MalformedBatch {
    inner: RecurrentActor,
    fail_decision: bool,
}
#[cfg(feature = "neural")]
impl StatefulActor for MalformedBatch {
    type State = Vec<f32>;
    fn observation_space(&self) -> ObservationSpace {
        self.inner.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.inner.action_space()
    }
    fn revision(&self) -> u64 {
        self.inner.revision()
    }
    fn initial_state(&self) -> Result<Self::State> {
        self.inner.initial_state()
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.inner.validate_state(state)
    }
    fn assimilate(&self, state: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        self.inner.assimilate(state, input)
    }
    fn decide(
        &self,
        state: &Self::State,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<rl::core::stateful::StatefulDecision> {
        self.inner.decide(state, observation, mode, rng)
    }
    fn batch_compatible(&self, other: &Self) -> bool {
        self.fail_decision == other.fail_decision && self.inner.batch_compatible(&other.inner)
    }
    fn assimilate_batch(
        &self,
        states: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        let states = self.inner.assimilate_batch(states, inputs)?;
        if !self.fail_decision && inputs.iter().any(|i| i.previous.is_some()) {
            Ok(Vec::new())
        } else {
            Ok(states)
        }
    }
    fn decide_batch(
        &self,
        states: &[&Self::State],
        observations: &[&Observation],
        mode: ActMode,
        rngs: &mut [ChaCha8Rng],
    ) -> Result<Vec<rl::core::stateful::StatefulDecision>> {
        let decisions = self.inner.decide_batch(states, observations, mode, rngs)?;
        if self.fail_decision {
            Ok(Vec::new())
        } else {
            Ok(decisions)
        }
    }
}
#[cfg(feature = "neural")]
impl Snapshot for MalformedBatch {
    fn validate_snapshot(&self) -> Result<()> {
        self.inner.validate_snapshot()
    }
}
#[test]
#[cfg(feature = "neural")]
fn malformed_batch_candidates_do_not_commit_lane_memory_or_rng_and_require_recovery() {
    for fail_decision in [true, false] {
        let actor = MalformedBatch {
            inner: actor(17, 0.4, 0),
            fail_decision,
        };
        let mut runtime = StatefulCollector::new_with_inference_mode(
            environments(),
            actor,
            73,
            InferenceMode::Batched,
        )
        .unwrap();
        let before = serde_json::to_value(&runtime).unwrap();
        let mut tape = Tape::default();
        assert!(runtime.collect(1, ActMode::Sample, &mut tape).is_err());
        let failed = serde_json::to_value(&runtime).unwrap();
        for lane in 0..3 {
            assert_eq!(before["agents"][lane]["rng"], failed["agents"][lane]["rng"]);
            assert_eq!(
                before["agents"][lane]["state"],
                failed["agents"][lane]["state"]
            );
        }
        if fail_decision {
            assert_eq!(before["driver"]["env"], failed["driver"]["env"]);
        } else {
            assert_ne!(before["driver"]["env"], failed["driver"]["env"]);
        }
        assert!(tape.0.is_empty());
        assert!(runtime.validate_local().is_err());
        assert!(runtime.validate_snapshot().is_err());
        assert!(runtime.collect(1, ActMode::Sample, &mut tape).is_err());
        runtime.reset_all(&mut tape).unwrap();
        runtime.validate_local().unwrap();
        runtime.validate_snapshot().unwrap();
        assert_eq!(runtime.lane_timestep(0), Some(0));
    }
}

#[test]
#[cfg(feature = "neural")]
fn preflight_mask_errors_leave_a_valid_collector_usable_and_old_snapshots_default_to_scalar() {
    let mut runtime = runtime(InferenceMode::Scalar);
    let mut tape = Tape::default();
    let snapshot = serde_json::to_value(&runtime).unwrap();
    assert!(snapshot.get("inference_mode").is_none());
    assert!(runtime.tick(&[true], ActMode::Sample, &mut tape).is_err());
    assert_eq!(snapshot, serde_json::to_value(&runtime).unwrap());
    runtime.validate_local().unwrap();
    runtime.collect(1, ActMode::Sample, &mut tape).unwrap();
    let restored: Runtime = serde_json::from_value(snapshot).unwrap();
    assert_eq!(restored.inference_mode(), InferenceMode::Scalar);
    restored.validate_local().unwrap();
}

#[test]
#[cfg(feature = "neural")]
fn inference_mode_version_pairs_are_audited_before_any_mutation() {
    for (mode, wrong_version) in [(InferenceMode::Batched, 1), (InferenceMode::Scalar, 2)] {
        let mut value = serde_json::to_value(runtime(mode)).unwrap();
        value["version"] = wrong_version.into();
        let mut loaded: Runtime = serde_json::from_value(value).unwrap();
        let before = serde_json::to_value(&loaded).unwrap();
        let mut tape = Tape::default();
        assert!(
            loaded.validate_local().is_err(),
            "accepted version {wrong_version} for {mode:?}"
        );
        assert!(loaded.tick(&[true; 3], ActMode::Sample, &mut tape).is_err());
        assert!(loaded.reset_all(&mut tape).is_err());
        assert_eq!(before, serde_json::to_value(&loaded).unwrap());
        assert!(tape.0.is_empty());
    }
    let mut collector = runtime(InferenceMode::Scalar);
    let original = serde_json::to_value(&collector).unwrap();
    assert_eq!(original["version"], 1);
    collector
        .set_inference_mode(InferenceMode::Batched)
        .unwrap();
    let batched = serde_json::to_value(&collector).unwrap();
    assert_eq!(batched["version"], 2);
    assert_eq!(batched["inference_mode"], "batched");
    collector.set_inference_mode(InferenceMode::Scalar).unwrap();
    assert_eq!(serde_json::to_value(&collector).unwrap(), original);
}
