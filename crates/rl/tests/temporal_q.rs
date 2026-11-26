#![cfg(feature = "neural")]

use burn::{module::Param, optim::GradientsParams, tensor::Tensor};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    core::{
        action::{Action, ActionSpace},
        checkpoint::LearnerRecord,
        env::Observation,
        stateful::{HistoryInput, HistorySpec, PreviousInput},
    },
    data::temporal::{
        BehaviorMetadata, SequenceBatch, SequenceConfig, SequenceSample, TemporalStep,
    },
    nn::{
        checkpoint::RecordBurnExt,
        matrix,
        network::{Mlp, Network},
        target::frozen_copy,
        temporal::{GruEncoder, TemporalEncoder},
        temporal_policy::TemporalPolicy,
        temporal_q::{TemporalActionValue, TemporalQ},
        temporal_value::TemporalValue,
        to_vec, vector,
    },
    objectives::reppo::self_predictive_loss,
};

fn spec(space: ActionSpace) -> HistorySpec {
    HistorySpec::new(1, space).unwrap()
}
fn q(bins: usize) -> TemporalQ {
    TemporalQ::new(
        spec(ActionSpace::Unbounded { dimensions: 1 }),
        Some(3),
        &[4],
        bins,
        &mut ChaCha8Rng::seed_from_u64(401),
    )
    .unwrap()
}
fn history(space: &ActionSpace) -> SequenceBatch {
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
            observations: [0.7, -0.3, 0.2, 0.]
                .map(|x| Observation::new(vec![x]))
                .to_vec(),
            steps: (0..3)
                .map(|i| TemporalStep {
                    action: match space {
                        ActionSpace::Discrete { .. } => Action::Discrete(i % 2),
                        _ => Action::Continuous(vec![[-0.4, 0.1, 0.2][i]]),
                    },
                    reward: [0.6_f32, -0.2, 1.][i].into(),
                    terminated: i == 2,
                    truncated: false,
                    behavior: BehaviorMetadata {
                        revision: 0,
                        log_prob: None,
                    },
                })
                .collect(),
            loss_mask: vec![true; 3],
        }],
    }
}

#[test]
fn scalar_history_matches_sequence_features_and_candidate_queries_do_not_advance_it() {
    let q = q(3);
    let batch = history(&q.spec().action_space);
    let sample = &batch.samples[0];
    let mut state = q.encoder.initial_state(1).unwrap();
    let mut states = Vec::new();
    for (t, observation) in sample.observations.iter().take(3).enumerate() {
        let features = q
            .spec()
            .encode(HistoryInput {
                observation,
                previous: t.checked_sub(1).map(|i| PreviousInput {
                    factual: None,
                    action: &sample.steps[i].action,
                    reward: sample.steps[i].reward.total(),
                }),
            })
            .unwrap();
        let (_, next) = q
            .encoder
            .step(matrix(features, 1, q.spec().input_dim()), state)
            .unwrap();
        state = next;
        states.push(state.clone());
    }
    let selected = q.history_features(&batch, &[(0, 2), (0, 0)]).unwrap();
    assert_eq!(
        to_vec(selected.clone()).unwrap(),
        to_vec(Tensor::cat(vec![states[2].clone(), states[0].clone()], 0)).unwrap()
    );
    let before = to_vec(selected.clone()).unwrap();
    let expected = q
        .values_from_features(selected.clone(), matrix(vec![0.5, -0.6], 2, 1))
        .unwrap();
    let actual = q
        .values(&batch, &[(0, 2), (0, 0)], matrix(vec![0.5, -0.6], 2, 1))
        .unwrap();
    assert_eq!(to_vec(expected).unwrap(), to_vec(actual).unwrap());
    q.values_from_features(selected.clone(), matrix(vec![-0.9, 0.8], 2, 1))
        .unwrap();
    assert_eq!(before, to_vec(selected).unwrap());
}

#[test]
fn temporal_q_backpropagates_through_recurrent_history_and_action_conditioned_head() {
    let q = q(5);
    let batch = history(&q.spec().action_space);
    let loss = q
        .values(
            &batch,
            &[(0, 0), (0, 1), (0, 2)],
            matrix(vec![0.2, -0.4, 0.8], 3, 1),
        )
        .unwrap()
        .powf_scalar(2.)
        .sum();
    let gradients = loss.backward();
    let recurrent = q
        .encoder
        .gru
        .as_ref()
        .unwrap()
        .gru
        .new_gate
        .hidden_transform
        .weight
        .val();
    assert!(
        to_vec(recurrent.grad(&gradients).unwrap())
            .unwrap()
            .iter()
            .any(|g| g.abs() > 1e-7)
    );
    assert!(
        to_vec(q.head.layers[0].weight.val().grad(&gradients).unwrap())
            .unwrap()
            .iter()
            .any(|g| g.abs() > 1e-7)
    );
    assert!(
        q.predictor.as_ref().unwrap().layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
}

#[test]
fn frozen_q_preserves_exact_action_gradient_without_any_parameter_gradient() {
    let mut q = TemporalQ::new(
        spec(ActionSpace::Unbounded { dimensions: 1 }),
        Some(2),
        &[],
        1,
        &mut ChaCha8Rng::seed_from_u64(402),
    )
    .unwrap();
    q.head.layers[0].weight = Param::from_tensor(matrix(vec![0., 0., 2.], 3, 1));
    q.head.layers[0].bias = Some(Param::from_tensor(vector(&[0.])));
    let frozen = frozen_copy(&q);
    let action = matrix(vec![0.7], 1, 1).require_grad();
    let values = frozen
        .values_from_features(
            matrix(vec![0.4, -0.3], 1, 2).without_autodiff(),
            action.clone(),
        )
        .unwrap();
    assert!((to_vec(values.clone()).unwrap()[0] - 1.4).abs() < 1e-7);
    let mut gradients = values.sum().backward();
    assert_eq!(to_vec(action.grad(&gradients).unwrap()).unwrap(), vec![2.]);
    assert!(
        GradientsParams::from_module(&mut gradients, &frozen).is_empty(),
        "frozen critic parameters received gradients"
    );
}

#[test]
fn auxiliary_target_is_next_history_action_conditioned_and_detached() {
    let online = q(3);
    let mut rng = ChaCha8Rng::seed_from_u64(403);
    let target = TemporalQ::new(online.spec().clone(), Some(3), &[4], 3, &mut rng).unwrap();
    let batch = history(&online.spec().action_space);
    let next = target
        .representation(&batch, &[(0, 1)], matrix(vec![0.7], 1, 1))
        .unwrap();
    let current = target
        .representation(&batch, &[(0, 0)], matrix(vec![0.7], 1, 1))
        .unwrap();
    assert_ne!(
        to_vec(next.clone()).unwrap(),
        to_vec(current).unwrap(),
        "fixture must distinguish successor context"
    );
    let prediction = online
        .predict_representation(&batch, &[(0, 0)], matrix(vec![-0.4], 1, 1))
        .unwrap();
    let loss = self_predictive_loss(prediction, next, vector(&[1.])).unwrap();
    let gradients = loss.backward();
    assert!(
        target.head.layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    assert!(
        target
            .encoder
            .gru
            .as_ref()
            .unwrap()
            .gru
            .new_gate
            .input_transform
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
    assert!(
        online.predictor.as_ref().unwrap().layers[0]
            .weight
            .val()
            .grad(&gradients)
            .is_some()
    );
}

#[test]
fn categorical_one_hot_queries_and_optional_auxiliary_work_through_generic_capability() {
    fn output<Q: TemporalActionValue>(q: &Q, batch: &SequenceBatch) -> Tensor<2> {
        assert_eq!(q.action_dim(), 2);
        q.values(batch, &[(0, 1)], matrix(vec![0., 1.], 1, 2))
            .unwrap()
    }
    let s = spec(ActionSpace::Discrete { count: 2 });
    let mut rng = ChaCha8Rng::seed_from_u64(404);
    let full = TemporalQ::new(s.clone(), Some(3), &[4], 7, &mut rng).unwrap();
    assert_eq!(output(&full, &history(&s.action_space)).dims(), [1, 7]);
    let no_aux = TemporalQ::from_parts(
        full.encoder.clone(),
        full.head.clone(),
        None::<Mlp>,
        s,
        false,
    )
    .unwrap();
    assert!(!no_aux.supports_auxiliary());
    assert!(
        no_aux
            .predict_representation(
                &history(&no_aux.spec().action_space),
                &[(0, 1)],
                matrix(vec![1., 0.], 1, 2)
            )
            .is_err()
    );
    assert!(
        no_aux
            .representation(
                &history(&no_aux.spec().action_space),
                &[(0, 1)],
                matrix(vec![1., 0.], 1, 2)
            )
            .is_err()
    );
    assert_eq!(no_aux.architecture().unwrap().predictor_bias, None);
    let mut record = LearnerRecord::new("no_aux_q", ()).unwrap();
    record.model("q", &no_aux).unwrap();
    let restored =
        TemporalQ::restore_from_record(&record, "q", &no_aux.architecture().unwrap()).unwrap();
    assert!(!restored.supports_auxiliary());
    assert_eq!(
        to_vec(output(&no_aux, &history(&no_aux.spec().action_space))).unwrap(),
        to_vec(output(&restored, &history(&restored.spec().action_space))).unwrap()
    );
}

#[derive(burn::module::Module, Debug)]
struct CustomHead {
    inner: Mlp,
}
impl Network for CustomHead {
    fn input_dim(&self) -> usize {
        self.inner.input_dim()
    }
    fn output_dim(&self) -> usize {
        self.inner.output_dim()
    }
    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.inner.forward(input)
    }
}
#[test]
fn custom_encoder_and_non_auxiliary_head_need_only_the_generic_q_capability() {
    let mut rng = ChaCha8Rng::seed_from_u64(407);
    let spec = spec(ActionSpace::Discrete { count: 2 });
    let encoder = GruEncoder::new(spec.input_dim(), 2, &mut rng).unwrap();
    let head = CustomHead {
        inner: Mlp::new(4, &[], 1, &mut rng).unwrap(),
    };
    let q = TemporalQ::from_parts(encoder, head, None::<Mlp>, spec, false).unwrap();
    let values = TemporalActionValue::values(
        &q,
        &history(&q.spec().action_space),
        &[(0, 2)],
        matrix(vec![1., 0.], 1, 2),
    )
    .unwrap();
    assert_eq!(values.dims(), [1, 1]);
    assert!(to_vec(values).unwrap()[0].is_finite());
}

#[test]
fn q_record_restores_exact_values_and_rejects_corrupt_allocation_descriptors() {
    let q = q(5);
    let architecture = q.architecture().unwrap();
    assert_eq!(architecture.model.version, 2);
    assert_eq!(architecture.model.head_input_extra, 1);
    let mut record = LearnerRecord::new("temporal_q_test", ()).unwrap();
    record.model("q", &q).unwrap();
    let restored = TemporalQ::restore_from_record(&record, "q", &architecture).unwrap();
    let batch = history(&q.spec().action_space);
    let query = |q: &TemporalQ| {
        to_vec(
            q.values(&batch, &[(0, 2)], matrix(vec![0.4], 1, 1))
                .unwrap(),
        )
        .unwrap()
    };
    assert_eq!(query(&q), query(&restored));
    assert_eq!(q.architecture().unwrap(), restored.architecture().unwrap());
    let mut corrupt = architecture.clone();
    corrupt.model.hidden_dim = Some(usize::MAX);
    assert!(TemporalQ::restore_from_record(&record, "q", &corrupt).is_err());
    let mut corrupt = architecture.clone();
    corrupt.model.head_input_extra = 2;
    assert!(TemporalQ::restore_from_record(&record, "q", &corrupt).is_err());
    let mut corrupt = architecture;
    corrupt.predictor_bias = None;
    assert!(TemporalQ::restore_from_record(&record, "q", &corrupt).is_err());
}

#[test]
fn policy_and_value_v1_wire_is_unchanged_and_rejects_action_conditioned_v2() {
    let s = spec(ActionSpace::Unbounded { dimensions: 1 });
    let mut rng = ChaCha8Rng::seed_from_u64(405);
    let policy = TemporalPolicy::new(s.clone(), Some(3), &[], &mut rng).unwrap();
    let value = TemporalValue::new(s.clone(), Some(3), &[], &mut rng).unwrap();
    for architecture in [
        policy.architecture().unwrap(),
        value.architecture().unwrap(),
    ] {
        assert_eq!(architecture.version, 1);
        assert!(
            serde_json::to_value(&architecture)
                .unwrap()
                .get("head_input_extra")
                .is_none()
        );
        assert_eq!(architecture.head_input_extra, 0);
    }
    let architecture = TemporalQ::new(s, Some(3), &[], 1, &mut rng)
        .unwrap()
        .architecture()
        .unwrap();
    assert!(TemporalPolicy::from_architecture(&architecture.model, &mut rng).is_err());
    assert!(TemporalValue::from_architecture(&architecture.model, &mut rng).is_err());
}

#[test]
fn malformed_head_and_action_widths_are_rejected_without_querying_history() {
    let mut q = q(2);
    let batch = history(&q.spec().action_space);
    assert!(
        q.values(&batch, &[(0, 1)], matrix(vec![1., 0.], 1, 2))
            .is_err()
    );
    assert!(
        q.values_from_features(matrix(vec![1., 0.], 1, 2), matrix(vec![0.], 1, 1))
            .is_err()
    );
    q.encoder.gru = Some(
        GruEncoder::new(q.spec().input_dim(), 6, &mut ChaCha8Rng::seed_from_u64(406)).unwrap(),
    );
    assert!(q.architecture().is_err());
}
