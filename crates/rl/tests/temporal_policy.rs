#![cfg(feature = "neural")]

use burn::{module::Param, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        checkpoint::Snapshot,
        env::Observation,
        stateful::{HistoryInput, HistorySpec, PreviousInput, StatefulActor},
        stateful_on_policy::{DistributionSnapshot, OnPolicyIdentity, StatefulOnPolicyActor},
    },
    data::temporal::{
        BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep,
    },
    nn::{
        device,
        temporal::GruEncoder,
        temporal_actor::TemporalActor,
        temporal_policy::{TemporalPolicy, TemporalStochasticPolicy},
        temporal_value::{TemporalStateValue, TemporalValue},
        to_vec,
    },
};

fn models(space: ActionSpace) -> (TemporalPolicy, TemporalValue) {
    let spec = HistorySpec::new(1, space).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(71);
    (
        TemporalPolicy::new(spec.clone(), Some(3), &[], &mut rng).unwrap(),
        TemporalValue::new(spec, Some(5), &[], &mut rng).unwrap(),
    )
}

#[test]
fn portable_import_never_relaxes_observation_only_policy_or_value_memory() {
    for (policy_hidden, value_hidden) in [(Some(3), None), (None, Some(5))] {
        let mut rng = ChaCha8Rng::seed_from_u64(791);
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let actor = TemporalActor::new(
            TemporalPolicy::new(spec.clone(), policy_hidden, &[], &mut rng).unwrap(),
            Some(TemporalValue::new(spec, value_hidden, &[], &mut rng).unwrap()),
            1,
        )
        .unwrap();
        let native = actor
            .assimilate(
                &actor.initial_state().unwrap(),
                HistoryInput {
                    observation: &Observation::new(vec![0.25]),
                    previous: None,
                },
            )
            .unwrap();
        let mut saved = native.clone();
        let observation_memory = if policy_hidden.is_none() {
            &mut saved.policy[0]
        } else {
            &mut saved.value.as_mut().unwrap()[0]
        };
        *observation_memory = f32::from_bits(observation_memory.to_bits() + 1);
        assert!(actor.audit_portable_memory(&saved, &native).is_err());
    }
}

fn history() -> SequenceBatch {
    SequenceBatch {
        config: SequenceConfig {
            burn_in: 0,
            learn_len: 3,
            n_step: 1,
        },
        samples: vec![SequenceSample {
            trajectory_start: 0,
            history_start: 0,
            previous: None,
            origin: Default::default(),
            episode_id: 1,
            learn_start: 0,
            burn_start: 0,
            observations: [0.8, -0.3, 0.4, 0.]
                .map(|x| Observation::new(vec![x]))
                .to_vec(),
            steps: (0..3)
                .map(|i| TemporalStep {
                    action: Action::Discrete(i % 2),
                    reward: (0.2 * i as f32).into(),
                    terminated: i == 2,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision: 1,
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 3],
        }],
    }
}

// A saturated tanh action cannot reconstruct its original latent with atanh.
#[test]
fn bounded_sampling_retains_true_saturated_latent_and_exact_distribution() {
    let (mut policy, _) = models(ActionSpace::Box {
        low: vec![-2.],
        high: vec![2.],
    });
    policy.head.layers[0].weight = Param::from_tensor(Tensor::zeros([3, 1], &device()));
    policy.head.layers[0].bias = Some(Param::from_tensor(Tensor::from_data([40.], &device())));
    policy.log_std = Param::from_tensor(Tensor::from_data([-20.], &device()));
    let actor = TemporalActor::new(policy, None, 4).unwrap();
    let obs = Observation::new(vec![0.3]);
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let sample = actor
        .sample_on_policy(
            &state,
            &obs,
            ActMode::Sample,
            &mut ChaCha8Rng::seed_from_u64(9),
        )
        .unwrap();
    assert_eq!(sample.decision.action, Action::Continuous(vec![2.]));
    assert_eq!(sample.latent, Some(vec![40.]));
    assert!(sample.decision.log_prob.unwrap().is_finite());
    assert!(sample.value.is_none());
    assert_eq!(
        sample.distribution,
        DistributionSnapshot::Gaussian {
            mean: vec![40.],
            log_std: vec![-20.],
            bounds: Some((vec![-2.], vec![2.])),
        }
    );
    let dist = sample.distribution.to_distribution(&device()).unwrap();
    let lp = to_vec(
        dist.log_prob(&[sample.decision.action], &[sample.latent])
            .unwrap(),
    )
    .unwrap()[0];
    assert!((lp - sample.decision.log_prob.unwrap()).abs() < 1e-5);
}

#[test]
fn actor_and_value_keep_independent_memory_and_read_only_value_queries() {
    let (policy, value) = models(ActionSpace::Discrete { count: 2 });
    let actor = TemporalActor::new(policy, Some(value), 8).unwrap();
    let initial = actor.initial_state().unwrap();
    assert_eq!(initial.policy.len(), 3);
    assert_eq!(initial.value.as_ref().unwrap().len(), 5);
    let obs = Observation::new(vec![0.7]);
    let state = actor
        .assimilate(
            &initial,
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let before = state.clone();
    let v = actor.state_value(&state, &obs).unwrap().unwrap();
    assert!(v.is_finite());
    assert_eq!(state, before);
    assert_eq!(actor.state_value(&state, &obs).unwrap(), Some(v));
    let mut malformed = state;
    malformed.value = None;
    assert!(actor.validate_state(&malformed).is_err());
}

#[test]
fn batched_inference_preserves_scalar_states_samples_and_lane_rngs() {
    for space in [
        ActionSpace::Discrete { count: 2 },
        ActionSpace::Box {
            low: vec![-2.],
            high: vec![3.],
        },
    ] {
        let (policy, value) = models(space);
        let actor = TemporalActor::new(policy, Some(value), 12).unwrap();
        let initial = actor.initial_state().unwrap();
        let obs = [Observation::new(vec![0.2]), Observation::new(vec![-0.5])];
        let inputs: Vec<_> = obs
            .iter()
            .map(|observation| HistoryInput {
                observation,
                previous: None,
            })
            .collect();
        let scalar: Vec<_> = inputs
            .iter()
            .map(|&input| actor.assimilate(&initial, input).unwrap())
            .collect();
        let batch = actor
            .assimilate_batch(&[&initial, &initial], &inputs)
            .unwrap();
        assert_eq!(scalar, batch);
        let mut scalar_rngs = [ChaCha8Rng::seed_from_u64(3), ChaCha8Rng::seed_from_u64(6)];
        let mut batch_rngs = scalar_rngs.clone();
        let expected: Vec<_> = (0..2)
            .map(|i| {
                actor
                    .sample_on_policy(&scalar[i], &obs[i], ActMode::Sample, &mut scalar_rngs[i])
                    .unwrap()
            })
            .collect();
        let actual = actor
            .sample_on_policy_batch(
                &[&batch[0], &batch[1]],
                &[&obs[0], &obs[1]],
                ActMode::Sample,
                &mut batch_rngs,
            )
            .unwrap();
        assert_eq!(scalar_rngs, batch_rngs);
        for (a, b) in expected.iter().zip(actual) {
            assert_eq!(a.decision.action, b.decision.action);
            assert_eq!(a.decision.log_prob, b.decision.log_prob);
            assert_eq!(a.latent, b.latent);
            assert_eq!(a.distribution, b.distribution);
            assert_eq!(a.value, b.value);
        }
    }
}

#[test]
fn selected_likelihood_unroll_keeps_recurrent_gradients_and_token_order() {
    let (policy, value) = models(ActionSpace::Discrete { count: 2 });
    let batch = history();
    let dist = policy.distribution(&batch, &[(0, 2), (0, 0)]).unwrap();
    let lp = dist
        .log_prob(&[Action::Discrete(1), Action::Discrete(0)], &[None, None])
        .unwrap();
    let loss = lp.sum().neg();
    let gradients = loss.backward();
    let recurrent = policy
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .hidden_transform
        .weight
        .val();
    let grad = to_vec(recurrent.grad(&gradients).unwrap()).unwrap();
    assert!(grad.iter().all(|x| x.is_finite()));
    assert!(grad.iter().any(|x| x.abs() > 1e-7));
    assert!(value.head.layers[0].weight.val().grad(&gradients).is_none());
    assert!(policy.distribution(&batch, &[(0, 4)]).is_err());
    assert_eq!(value.values(&batch, &[(0, 2), (0, 0)]).unwrap().dims(), [2]);
}

#[test]
fn observation_only_models_ignore_previous_inputs_and_have_no_recurrent_state() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(10);
    let policy = TemporalPolicy::new(spec.clone(), None, &[], &mut rng).unwrap();
    let value = TemporalValue::new(spec, None, &[], &mut rng).unwrap();
    let actor = TemporalActor::new(policy, Some(value), 1).unwrap();
    let obs = Observation::new(vec![0.4]);
    let prior = actor.initial_state().unwrap();
    let action = Action::Discrete(1);
    let a = actor
        .assimilate(
            &prior,
            HistoryInput {
                observation: &obs,
                previous: None,
            },
        )
        .unwrap();
    let b = actor
        .assimilate(
            &a,
            HistoryInput {
                observation: &obs,
                previous: Some(PreviousInput {
                    factual: None,
                    action: &action,
                    reward: 17.,
                }),
            },
        )
        .unwrap();
    assert_eq!(a, b);
    assert!(actor.policy().encoder.gru.is_none());
    assert!(actor.value().unwrap().encoder.gru.is_none());
}

#[test]
fn frozen_actor_round_trip_preserves_history_rng_and_rejects_corrupt_architecture() {
    let (policy, value) = models(ActionSpace::Discrete { count: 2 });
    let actor = TemporalActor::new(policy, Some(value), 19).unwrap();
    actor.validate_snapshot().unwrap();
    let wire = serde_json::to_value(&actor).unwrap();
    let restored: TemporalActor = serde_json::from_value(wire.clone()).unwrap();
    assert!(actor.batch_compatible(&restored));
    let obs = Observation::new(vec![0.2]);
    let input = HistoryInput {
        observation: &obs,
        previous: None,
    };
    assert_eq!(
        actor
            .assimilate(&actor.initial_state().unwrap(), input)
            .unwrap(),
        restored
            .assimilate(&restored.initial_state().unwrap(), input)
            .unwrap()
    );
    let mut corrupt = wire;
    corrupt["metadata"]["policy"]["hidden_dim"] = usize::MAX.into();
    assert!(serde_json::from_value::<TemporalActor>(corrupt).is_err());
}

#[test]
fn snapshots_reject_incompatible_supports_and_nonfinite_parameters() {
    let snapshot = DistributionSnapshot::Categorical {
        logits: vec![0., 1.],
        masks: vec![false, true],
    };
    let dist = snapshot.to_distribution(&device()).unwrap();
    assert_eq!(
        to_vec(dist.categorical_terms().unwrap().probabilities).unwrap(),
        [0., 1.]
    );
    assert!(
        DistributionSnapshot::Categorical {
            logits: vec![0., 1.],
            masks: vec![false, false]
        }
        .to_distribution(&device())
        .is_err()
    );
    assert!(
        DistributionSnapshot::Gaussian {
            mean: vec![0.],
            log_std: vec![f32::NAN],
            bounds: None
        }
        .to_distribution(&device())
        .is_err()
    );
    let (mut policy, _) = models(ActionSpace::Discrete { count: 2 });
    policy.head.layers[0].weight = Param::from_tensor(Tensor::full([3, 2], f32::NAN, &device()));
    assert!(TemporalActor::new(policy, None, 1).is_err());
}

#[test]
fn replaced_backbone_width_is_rejected_before_behavior_construction() {
    let (mut policy, _) = models(ActionSpace::Discrete { count: 2 });
    policy.encoder.gru = Some(
        GruEncoder::new(
            policy.spec().input_dim(),
            4,
            &mut ChaCha8Rng::seed_from_u64(37),
        )
        .unwrap(),
    );
    assert!(TemporalActor::new(policy, None, 1).is_err());
}

#[test]
fn snapshots_reject_unrepresentable_box_width_before_tensor_conversion() {
    let snapshot = DistributionSnapshot::Gaussian {
        mean: vec![0.],
        log_std: vec![0.],
        bounds: Some((vec![-f32::MAX], vec![f32::MAX])),
    };
    assert!(snapshot.to_distribution(&device()).is_err());
}

#[test]
fn selected_training_distributions_match_full_scalar_history_before_selection() {
    let (policy, _) = models(ActionSpace::Discrete { count: 2 });
    let actor = TemporalActor::new(policy.clone(), None, 3).unwrap();
    let mut batch = history();
    // The selected token is preceded by a detached two-observation prefix.
    batch.samples[0].learn_start = 2;
    batch.samples[0].burn_start = 2;
    batch.samples[0].loss_mask = vec![false, false, true];
    let distribution = policy.distribution(&batch, &[(0, 2), (0, 0)]).unwrap();
    let sample = &batch.samples[0];
    let mut state = actor.initial_state().unwrap();
    let mut expected = Vec::new();
    for (t, observation) in sample.observations.iter().take(3).enumerate() {
        let previous = t.checked_sub(1).map(|i| PreviousInput {
            factual: None,
            action: &sample.steps[i].action,
            reward: sample.steps[i].reward.total(),
        });
        state = actor
            .assimilate(
                &state,
                HistoryInput {
                    observation,
                    previous,
                },
            )
            .unwrap();
        expected.push(
            actor
                .sample_on_policy(
                    &state,
                    observation,
                    ActMode::Deterministic,
                    &mut ChaCha8Rng::seed_from_u64(11),
                )
                .unwrap()
                .distribution,
        );
    }
    assert_eq!(
        DistributionSnapshot::from_distribution(&distribution, 0).unwrap(),
        expected[2]
    );
    assert_eq!(
        DistributionSnapshot::from_distribution(&distribution, 1).unwrap(),
        expected[0]
    );
}

#[test]
fn final_value_query_accepts_empty_action_mask_without_sampling_or_state_change() {
    let (policy, value) = models(ActionSpace::Discrete { count: 2 });
    let actor = TemporalActor::new(policy, Some(value), 2).unwrap();
    let mut observation = Observation::new(vec![0.4]);
    observation.as_dense_mut().unwrap().action_mask = Some(vec![false, false]);
    let action = Action::Discrete(1);
    let state = actor
        .assimilate(
            &actor.initial_state().unwrap(),
            HistoryInput {
                observation: &observation,
                previous: Some(PreviousInput {
                    factual: None,
                    action: &action,
                    reward: 2.,
                }),
            },
        )
        .unwrap();
    let before = state.clone();
    assert!(
        actor
            .state_value(&state, &observation)
            .unwrap()
            .unwrap()
            .is_finite()
    );
    assert_eq!(state, before);
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let prior_rng = rng.clone();
    assert!(
        actor
            .sample_on_policy(&state, &observation, ActMode::Sample, &mut rng)
            .is_err()
    );
    assert_eq!(rng, prior_rng);
}
