#![cfg(feature = "neural")]

use burn::{
    module::Param,
    tensor::{Tensor, TensorData},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::core::stateful_on_policy::OnPolicyIdentity;
use rl::{
    Result,
    collect::{
        stateful::StatefulCollector, temporal_on_policy::OnPolicyRecording,
        vector_env::SyncVectorEnv,
    },
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
        stateful_on_policy::StatefulOnPolicyActor,
    },
    data::{
        temporal::SequenceConfig,
        temporal_on_policy::{TemporalBatchKind, TemporalRollout},
    },
    nn::{
        device, temporal::TemporalEncoder, temporal_actor::TemporalActor,
        temporal_pathwise_actor::TemporalPathwiseActor, temporal_policy::TemporalPolicy,
        temporal_q::TemporalQ, to_vec,
    },
};

fn models(seed: u64, space: ActionSpace) -> (TemporalPolicy, TemporalQ) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let spec = HistorySpec::new(1, space).unwrap();
    (
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalQ::new(spec, Some(5), &[4], 7, &mut rng).unwrap(),
    )
}
fn actor(seed: u64, space: ActionSpace) -> TemporalPathwiseActor {
    let (policy, critic) = models(seed, space);
    TemporalPathwiseActor::new(policy, critic, 4).unwrap()
}

#[test]
fn portable_import_audits_independent_q_memory_for_discrete_and_continuous_actors() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Unbounded { dimensions: 1 },
    ] {
        let actor = actor(711, space);
        let native = actor
            .assimilate(
                &actor.initial_state().unwrap(),
                HistoryInput {
                    observation: &Observation::new(vec![0.25]),
                    previous: None,
                },
            )
            .unwrap();
        for ulps in [1, 4, 5] {
            let mut saved = native.clone();
            saved.critic[0] = f32::from_bits(saved.critic[0].to_bits() + ulps);
            let result = actor.audit_portable_memory(&saved, &native);
            if ulps <= 4 {
                let audits = result.unwrap();
                assert_eq!(audits[0].adjusted_values, 0);
                assert_eq!(audits[1].role, "q");
                assert_eq!(audits[1].adjusted_values, 1);
                assert_eq!(audits[1].max_ulp_difference, ulps);
            } else {
                assert!(result.is_err(), "Q memory escaped independent audit");
            }
        }
        let mut nonfinite = native.clone();
        nonfinite.critic[0] = f32::NAN;
        assert!(actor.audit_portable_memory(&nonfinite, &native).is_err());
    }
}

#[test]
fn independent_compound_memories_and_sampling_match_scalar_and_batched_streams() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
    ] {
        let actor = actor(451, space);
        let initial = actor.initial_state().unwrap();
        assert_eq!(initial.policy.policy.len(), 3);
        assert!(initial.policy.value.is_none());
        assert_eq!(initial.critic.len(), 5);
        for lanes in [1, 4, 16] {
            let mut scalar = vec![initial.clone(); lanes];
            let mut batched = scalar.clone();
            let mut scalar_rngs: Vec<_> = (0..lanes)
                .map(|i| ChaCha8Rng::seed_from_u64(700 + i as u64))
                .collect();
            let mut batch_rngs = scalar_rngs.clone();
            let mut previous: Vec<Option<Action>> = vec![None; lanes];
            for tick in 0..5 {
                let observations: Vec<_> = (0..lanes)
                    .map(|lane| Observation::new(vec![0.1 * lane as f32 - 0.2 * tick as f32]))
                    .collect();
                let inputs: Vec<_> = observations
                    .iter()
                    .enumerate()
                    .map(|(lane, observation)| HistoryInput {
                        observation,
                        previous: previous[lane].as_ref().map(|action| PreviousInput {
                            factual: None,
                            action,
                            reward: 0.3 * tick as f32,
                        }),
                    })
                    .collect();
                let states = actor
                    .assimilate_batch(&batched.iter().collect::<Vec<_>>(), &inputs)
                    .unwrap();
                for lane in 0..lanes {
                    scalar[lane] = actor.assimilate(&scalar[lane], inputs[lane]).unwrap();
                }
                assert_eq!(scalar, states);
                batched = states;
                let expected: Vec<_> = (0..lanes)
                    .map(|lane| {
                        actor
                            .sample_on_policy(
                                &scalar[lane],
                                &observations[lane],
                                ActMode::Sample,
                                &mut scalar_rngs[lane],
                            )
                            .unwrap()
                    })
                    .collect();
                let actual = actor
                    .sample_on_policy_batch(
                        &batched.iter().collect::<Vec<_>>(),
                        &observations.iter().collect::<Vec<_>>(),
                        ActMode::Sample,
                        &mut batch_rngs,
                    )
                    .unwrap();
                assert_eq!(scalar_rngs, batch_rngs);
                for (lane, (a, b)) in expected.iter().zip(actual).enumerate() {
                    assert_eq!(a.decision.action, b.decision.action);
                    assert_eq!(a.decision.log_prob, b.decision.log_prob);
                    assert_eq!(a.latent, b.latent);
                    assert_eq!(a.distribution, b.distribution);
                    assert!(a.value.is_none() && b.value.is_none());
                    previous[lane] = Some(a.decision.action.clone());
                }
            }
        }
    }
}

#[test]
fn critic_assimilates_each_actual_input_once_and_sampling_never_queries_q_head() {
    let (policy, mut critic) = models(452, ActionSpace::Discrete { count: 2 });
    // A finite but enormous Q head is irrelevant to policy sampling and no-value
    // queries. Any accidental Q evaluation would produce unusable logits.
    critic.head.layers[0].bias = Some(Param::from_tensor(Tensor::full([4], f32::MAX, &device())));
    let actor = TemporalPathwiseActor::new(policy, critic, 0).unwrap();
    let policy_only = TemporalActor::new(actor.policy().clone(), None, 0).unwrap();
    let mut compound = actor.initial_state().unwrap();
    let mut expected_policy = policy_only.initial_state().unwrap();
    let mut expected_q = actor.critic().encoder.initial_state(1).unwrap();
    let action = Action::Discrete(1);
    for (time, value) in [0.8, -0.2, 0.4].into_iter().enumerate() {
        let observation = Observation::new(vec![value]);
        let input = HistoryInput {
            observation: &observation,
            previous: (time > 0).then_some(PreviousInput {
                factual: None,
                action: &action,
                reward: -0.3,
            }),
        };
        compound = actor.assimilate(&compound, input).unwrap();
        expected_policy = policy_only.assimilate(&expected_policy, input).unwrap();
        let encoded = actor.spec().encode(input).unwrap();
        let (_, next) = actor
            .critic()
            .encoder
            .step(
                Tensor::from_data(
                    TensorData::new(encoded, [1, actor.spec().input_dim()]),
                    &actor.critic().device(),
                ),
                expected_q,
            )
            .unwrap();
        expected_q = next;
        assert_eq!(compound.policy, expected_policy);
        assert_eq!(compound.critic, to_vec(expected_q.clone()).unwrap());
        let before = compound.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(100 + time as u64);
        let sample = actor
            .sample_on_policy(&compound, &observation, ActMode::Sample, &mut rng)
            .unwrap();
        assert!(sample.decision.log_prob.unwrap().is_finite());
        assert_eq!(actor.state_value(&compound, &observation).unwrap(), None);
        actor
            .validate_recording(
                &compound,
                &observation,
                &sample.decision.action,
                sample.latent.as_deref(),
                &sample.distribution,
                sample.decision.log_prob.unwrap(),
                None,
            )
            .unwrap();
        assert_eq!(compound, before);
    }
}

#[test]
fn identity_is_numeric_revision_independent_and_includes_the_critic_encoder() {
    let a = actor(453, ActionSpace::Discrete { count: 2 });
    let b = actor(453, ActionSpace::Discrete { count: 2 });
    assert_ne!(
        a.policy().head.layers[0].weight.id,
        b.policy().head.layers[0].weight.id
    );
    assert_eq!(a.model_identity(), b.model_identity());
    assert!(a.batch_compatible(&b));
    let revised = a.with_revision(5);
    assert_eq!(a.model_identity(), revised.model_identity());
    assert!(!a.batch_compatible(&revised));
    let cloned = a.clone();
    assert!(std::ptr::eq(a.policy(), cloned.policy()));
    assert!(std::ptr::eq(a.critic(), cloned.critic()));
    let mut critic = a.critic().clone();
    let gate = &mut critic
        .encoder
        .gru
        .as_mut()
        .unwrap()
        .gru
        .new_gate
        .input_transform;
    gate.weight = Param::from_tensor(gate.weight.val() + 0.125);
    let changed = TemporalPathwiseActor::new(a.policy().clone(), critic, 4).unwrap();
    assert_ne!(a.model_identity(), changed.model_identity());
    assert!(!a.batch_compatible(&changed));
}

#[test]
fn serde_round_trip_preserves_actions_memories_and_rejects_untrusted_wire() {
    let actor = actor(
        454,
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![2.],
        },
    );
    actor.validate_snapshot().unwrap();
    let wire = serde_json::to_value(&actor).unwrap();
    let restored: TemporalPathwiseActor = serde_json::from_value(wire.clone()).unwrap();
    restored.validate_snapshot().unwrap();
    assert_eq!(actor.model_identity(), restored.model_identity());
    let observation = Observation::new(vec![0.6]);
    let input = HistoryInput {
        observation: &observation,
        previous: None,
    };
    let state = actor
        .assimilate(&actor.initial_state().unwrap(), input)
        .unwrap();
    assert_eq!(
        state,
        restored
            .assimilate(&restored.initial_state().unwrap(), input)
            .unwrap()
    );
    let mut ar = ChaCha8Rng::seed_from_u64(455);
    let mut br = ar.clone();
    let a = actor
        .sample_on_policy(&state, &observation, ActMode::Sample, &mut ar)
        .unwrap();
    let b = restored
        .sample_on_policy(&state, &observation, ActMode::Sample, &mut br)
        .unwrap();
    assert_eq!(a.decision.action, b.decision.action);
    assert_eq!(a.latent, b.latent);
    assert_eq!(ar, br);
    let mut bad_version = wire.clone();
    bad_version["version"] = 2.into();
    let mut bad_arch = wire.clone();
    bad_arch["critic"]["metadata"]["architecture"]["model"]["hidden_dim"] = usize::MAX.into();
    let mut extra_optimizer = wire.clone();
    extra_optimizer["critic"]["optimizers"]["unexpected"] = serde_json::json!([]);
    let mut wrong_schema = wire;
    wrong_schema["critic"]["algorithm"] = "temporal_actor".into();
    for corrupt in [bad_version, bad_arch, extra_optimizer, wrong_schema] {
        assert!(serde_json::from_value::<TemporalPathwiseActor>(corrupt).is_err());
    }
}

#[test]
fn malformed_compound_memory_and_invented_baseline_fail_before_rng_consumption() {
    let actor = actor(456, ActionSpace::Discrete { count: 2 });
    let observation = Observation::new(vec![0.2]);
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &observation,
                previous: None,
            },
        )
        .unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(457);
    let before = rng.clone();
    let mut malformed = state.clone();
    malformed.critic.pop();
    assert!(
        actor
            .sample_on_policy(&malformed, &observation, ActMode::Sample, &mut rng)
            .is_err()
    );
    assert_eq!(rng, before);
    let mut rngs = vec![before.clone(), before.clone()];
    let expected_rngs = rngs.clone();
    assert!(
        actor
            .sample_on_policy_batch(
                &[&state, &malformed],
                &[&observation, &observation],
                ActMode::Sample,
                &mut rngs,
            )
            .is_err()
    );
    assert_eq!(rngs, expected_rngs);
    malformed = state.clone();
    malformed.policy.value = Some(vec![0.]);
    assert!(actor.validate_state(&malformed).is_err());
    let sample = actor
        .sample_on_policy(&state, &observation, ActMode::Sample, &mut rng)
        .unwrap();
    assert!(
        actor
            .validate_recording(
                &state,
                &observation,
                &sample.decision.action,
                None,
                &sample.distribution,
                sample.decision.log_prob.unwrap(),
                Some(0.)
            )
            .is_err()
    );
}

#[test]
fn constructor_rejects_shared_parameters_and_mismatched_history_schemas() {
    let mut rng = ChaCha8Rng::seed_from_u64(460);
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let policy = TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap();
    let mut critic = TemporalQ::new(spec, Some(3), &[], 1, &mut rng).unwrap();
    critic.encoder = policy.encoder.clone();
    assert!(TemporalPathwiseActor::new(policy.clone(), critic, 0).is_err());
    let other_spec = HistorySpec::new(2, ActionSpace::Discrete { count: 2 }).unwrap();
    let critic = TemporalQ::new(other_spec, Some(3), &[], 1, &mut rng).unwrap();
    assert!(TemporalPathwiseActor::new(policy, critic, 0).is_err());
}

#[derive(Clone)]
struct Terminal;
impl Env for Terminal {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }
    fn action_space(&self) -> &ActionSpace {
        &ActionSpace::Discrete { count: 2 }
    }
    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        Ok(EpisodeStart::fresh(Observation::new(vec![0.3])))
    }
    fn step(&mut self, _: &Action) -> Result<Step> {
        Ok(Step {
            observation: Observation::dense(vec![0.9], Some(vec![false, false])),
            reward: 1.0_f32.into(),
            terminated: true,
            truncated: false,
        })
    }
}
#[test]
fn terminal_empty_masks_are_assimilated_without_bootstrap_actions_or_q_values() {
    let actor = actor(458, ActionSpace::Discrete { count: 2 });
    let mut collector = StatefulCollector::new(
        SyncVectorEnv::new(vec![Terminal]).unwrap(),
        actor.clone(),
        459,
    )
    .unwrap();
    let mut rollout =
        TemporalRollout::new(16, 1, actor.action_space().clone(), 16, 1 << 20).unwrap();
    collector
        .tick_recorded(
            &[true],
            ActMode::Sample,
            &mut OnPolicyRecording::new(&mut rollout),
        )
        .unwrap();
    let state = collector.lane_state(0).unwrap();
    let observation = &collector.observations()[0];
    assert_eq!(actor.state_value(state, observation).unwrap(), None);
    let batch = rollout
        .seal(
            actor,
            TemporalBatchKind::Pathwise,
            0.9,
            None,
            false,
            &SequenceConfig {
                burn_in: 0,
                learn_len: 1,
                n_step: 1,
            },
        )
        .unwrap();
    assert_eq!(batch.len(), 1);
    assert!(batch.rows[0].value.is_none());
    assert!(batch.rows[0].next_value.is_none());
}
