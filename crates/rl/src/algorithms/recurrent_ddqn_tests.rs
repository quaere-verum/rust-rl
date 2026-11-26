use super::*;
use crate::nn::{
    network::Mlp,
    temporal::{GruEncoder, RecurrentQ},
};
use burn::optim::GradientsParams;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn frozen() -> serde_json::Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/recurrent-ddqn-pr12.json"
    )))
    .unwrap()
}
fn fixture() -> (RecurrentDdqn, SequenceBatch, serde_json::Value) {
    let frozen = frozen();
    let data = &frozen["numerical"]["learning"];
    let config: RecurrentDdqnConfig = serde_json::from_value(data["config"].clone()).unwrap();
    let batch = serde_json::from_value(data["selected_sequences"].clone()).unwrap();
    let spec =
        HistorySpec::new(5, crate::core::action::ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let q = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    (
        RecurrentDdqn::new(q, spec, config).unwrap(),
        batch,
        data.clone(),
    )
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-6 + 2e-6 * b.abs(),
            "float{i}: {a} != {b}"
        );
    }
}
fn numbers(value: &serde_json::Value) -> Vec<f32> {
    serde_json::from_value(value.clone()).unwrap()
}

#[test]
fn optimized_objective_gradients_and_updates_match_frozen_pr12() {
    let (mut learner, batch, data) = fixture();
    learner.set_update_mode(RecurrentUpdateMode::OptimizedExact);
    let objective = learner.critic.optimized_objective(&batch).unwrap();
    let expected_targets: Vec<f32> = data["updates"][0]["double_q_targets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["value"].as_f64().unwrap() as f32)
        .collect();
    close(
        &objective
            .targets
            .iter()
            .map(|p| p.value)
            .collect::<Vec<_>>(),
        &expected_targets,
    );
    let gradients = GradientsParams::from_grads(objective.loss.backward(), &learner.critic.online);
    struct Collect<'a> {
        gradients: &'a GradientsParams,
        values: Vec<Vec<f32>>,
    }
    impl ModuleVisitor for Collect<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.values.push(
                to_vec(
                    self.gradients
                        .get::<D>(p.id)
                        .expect("gradient for every recurrent/head parameter"),
                )
                .unwrap(),
            );
        }
    }
    let mut actual = Collect {
        gradients: &gradients,
        values: Vec::new(),
    };
    learner.critic.online.visit(&mut actual);
    let expected = data["first_update_parameter_arrays_and_raw_gradients"]
        .as_array()
        .unwrap();
    assert_eq!(actual.values.len(), expected.len());
    for (actual, expected) in actual.values.iter().zip(expected) {
        close(actual, &numbers(&expected["gradient"]));
    }
    for expected in data["updates"].as_array().unwrap() {
        let report = learner.update(&batch).unwrap();
        assert!(
            (report.metrics["critic_loss"]
                - expected["report"]["metrics"]["critic_loss"]
                    .as_f64()
                    .unwrap())
            .abs()
                < 2e-6
        );
        close(
            &learner.parameter_values().unwrap(),
            &numbers(&expected["online_parameters_after"]),
        );
        close(
            &parameter_values(&learner.critic.target).unwrap(),
            &numbers(&expected["target_parameters_after"]),
        );
    }
}

#[test]
fn behavior_weight_cache_is_invalidated_before_updates_and_on_restore() {
    let (mut learner, batch, _) = fixture();
    let first = learner.policy(0.1, 0).unwrap();
    let scheduled = learner.policy(0.9, 1).unwrap();
    assert!(std::ptr::eq(first.model(), scheduled.model()));
    learner.update(&batch).unwrap();
    let changed = learner.policy(0.1, 2).unwrap();
    assert_ne!(first.model_identity(), changed.model_identity());
    assert!(!std::ptr::eq(first.model(), changed.model()));
    let mut original = fixture().0;
    original.restore(&learner.checkpoint().unwrap()).unwrap();
    assert_eq!(
        changed.model_identity(),
        original.policy(0.1, 2).unwrap().model_identity()
    );
}

fn mixed_long_batch() -> SequenceBatch {
    use crate::core::{
        action::{Action, ActionSpace},
        env::{EpisodeStart, Observation},
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    };
    use crate::data::temporal::{EpisodeStore, SequenceConfig, sample_at};
    let mut store = EpisodeStore::new(256, 5, ActionSpace::Discrete { count: 2 }).unwrap();
    for episode in 0..2 {
        let observation = |t: usize| {
            Observation::new(vec![
                if t == 0 {
                    if episode == 0 { -1. } else { 1. }
                } else {
                    0.
                },
                f32::from(t == 0),
                0.,
                f32::from(t.is_multiple_of(16)),
                t as f32 / 96.,
            ])
        };
        store
            .begin_episode(episode, &EpisodeStart::fresh(observation(0)))
            .unwrap();
        for t in 0..96 {
            store
                .push(&TemporalTransition {
                    episode_id: episode,
                    timestep: t,
                    observation: observation(t),
                    action: Action::Discrete(t % 2),
                    reward: (if t == 95 { 1.0_f32 } else { 0. }).into(),
                    next_observation: observation(t + 1),
                    terminated: episode == 0 && t == 95,
                    truncated: episode == 1 && t == 95,
                    behavior: BehaviorMetadata::default(),
                })
                .unwrap();
        }
    }
    let config = SequenceConfig {
        burn_in: 16,
        learn_len: 4,
        n_step: 3,
    };
    SequenceBatch {
        samples: vec![
            sample_at(&store, 0, 0, &config).unwrap(),
            sample_at(&store, 1, 48, &config).unwrap(),
        ],
        config,
    }
}

#[test]
fn optimized_mixed_zero_and_long_prefixes_preserve_gradients_and_bootstrap_alignment() {
    let (model, _, _) = fixture();
    let batch = mixed_long_batch();
    let reference = model.critic.reference_objective(&batch).unwrap();
    let optimized = model.critic.optimized_objective(&batch).unwrap();
    let targets = |objective: &crate::algorithms::recurrent_critic::Objective| {
        objective
            .targets
            .iter()
            .map(|t| t.value)
            .collect::<Vec<_>>()
    };
    report_differences("targets", &targets(&reference), &targets(&optimized));
    report_differences(
        "loss",
        &to_vec(reference.loss.clone().without_autodiff()).unwrap(),
        &to_vec(optimized.loss.clone().without_autodiff()).unwrap(),
    );
    fn report_differences(stage: &str, left: &[f32], right: &[f32]) {
        close(left, right);
        let ordered = |value: f32| {
            let bits = value.to_bits();
            if value.is_sign_negative() {
                !bits
            } else {
                bits | 0x8000_0000
            }
        };
        let (mut max_abs, mut max_rel, mut max_ulp) = (0.0f64, 0.0f64, 0);
        for (&left, &right) in left.iter().zip(right) {
            let absolute = (f64::from(left) - f64::from(right)).abs();
            let scale = f64::from(left.abs().max(right.abs())).max(f64::MIN_POSITIVE);
            max_abs = max_abs.max(absolute);
            max_rel = max_rel.max(absolute / scale);
            max_ulp = max_ulp.max(ordered(left).abs_diff(ordered(right)));
        }
        println!("{stage} max_abs={max_abs:e} max_relative={max_rel:e} max_ulp={max_ulp}");
    }
    fn gradients(model: &GruQ, loss: Tensor<1>) -> Vec<f32> {
        let gradients = GradientsParams::from_grads(loss.backward(), model);
        struct Collect<'a> {
            g: &'a GradientsParams,
            v: Vec<f32>,
        }
        impl ModuleVisitor for Collect<'_> {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                self.v
                    .extend(to_vec(self.g.get::<D>(p.id).unwrap()).unwrap());
            }
        }
        let mut c = Collect {
            g: &gradients,
            v: Vec::new(),
        };
        model.visit(&mut c);
        c.v
    }
    let a = gradients(&model.critic.online, reference.loss);
    let b = gradients(&model.critic.online, optimized.loss);
    report_differences("raw gradients", &a, &b);
    assert!(b.iter().any(|v| v.abs() > 1e-5));
}

#[test]
fn using_online_warm_state_for_target_creates_obviously_wrong_targets() {
    let (mut model, _, _) = fixture();
    let batch = mixed_long_batch();
    let plain = Device::flex();
    for gate in [
        &mut model.critic.target.encoder.gru.update_gate,
        &mut model.critic.target.encoder.gru.reset_gate,
        &mut model.critic.target.encoder.gru.new_gate,
    ] {
        let shape = gate.input_transform.weight.val().dims();
        gate.input_transform.weight = Param::from_tensor(Tensor::zeros(shape, &plain));
        gate.input_transform.bias = Some(Param::from_tensor(Tensor::zeros([4], &plain)));
        gate.hidden_transform.weight = Param::from_tensor(Tensor::zeros([4, 4], &plain));
        gate.hidden_transform.bias = Some(Param::from_tensor(Tensor::zeros([4], &plain)));
    }
    model
        .critic
        .target
        .encoder
        .gru
        .new_gate
        .input_transform
        .bias = Some(Param::from_tensor(Tensor::ones([4], &plain)));
    for layer in &mut model.critic.target.head.layers {
        let shape = layer.weight.val().dims();
        layer.weight = Param::from_tensor(Tensor::ones(shape, &plain));
        layer.bias = Some(Param::from_tensor(Tensor::zeros([shape[1]], &plain)));
    }
    let prepared = prepare_exact_window(&batch, &model.critic.spec).unwrap();
    let p = &prepared.prefix;
    let s = &prepared.suffix;
    let online = model.critic.online.valid();
    let target = model.critic.target.valid();
    let input = Tensor::<3>::from_data(
        TensorData::new(p.inputs.clone(), [p.batch, p.time, p.input_dim]),
        &plain,
    );
    let online_state = warm_prefix(
        &online.encoder,
        input,
        online.initial_state(p.batch).unwrap(),
        &prepared.starts,
        &p.masks,
    )
    .unwrap();
    let input = Tensor::<3>::from_data(
        TensorData::new(s.inputs.clone(), [s.batch, s.time, s.input_dim]),
        &plain,
    );
    let (selection, _) = online
        .unroll(input.clone(), online_state.clone(), &s.masks)
        .unwrap();
    let (wrong_target, _) = target.unroll(input, online_state, &s.masks).unwrap();
    let wrong = construct_targets_layout(
        &batch,
        &to_vec(selection).unwrap(),
        &to_vec(wrong_target).unwrap(),
        2,
        model.config.gamma,
        QLayout {
            time: s.time,
            width: 5,
            offsets: &prepared.starts,
        },
    )
    .unwrap();
    let right = model.critic.optimized_objective(&batch).unwrap();
    let reference = model.critic.reference_objective(&batch).unwrap();
    close(
        &right.targets.iter().map(|p| p.value).collect::<Vec<_>>(),
        &reference
            .targets
            .iter()
            .map(|p| p.value)
            .collect::<Vec<_>>(),
    );
    assert!(
        right
            .targets
            .iter()
            .zip(wrong)
            .any(|(a, b)| (a.value - b.value).abs() > 0.05)
    );
}
