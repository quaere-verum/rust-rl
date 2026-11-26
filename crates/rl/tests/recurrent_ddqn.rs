#![cfg(feature = "neural")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::recurrent_ddqn::*,
    core::{
        stateful::HistorySpec,
        temporal::{BehaviorMetadata, TemporalExecutionMode, TemporalSink, TemporalTransition},
    },
    data::temporal::{EpisodeStore, SequenceBatch, SequenceConfig, sample_at},
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
    },
    prelude::*,
};

fn sample(terminated: bool, truncated: bool) -> SequenceBatch {
    let mut s = EpisodeStore::new(20, 1, ActionSpace::Discrete { count: 2 }).unwrap();
    s.begin_episode(0, &EpisodeStart::fresh(Observation::new(vec![1.])))
        .unwrap();
    for i in 0..3 {
        s.push(&TemporalTransition {
            episode_id: 0,
            timestep: i,
            observation: Observation::new(vec![(i + 1) as f32]),
            action: Action::Discrete(i % 2),
            reward: (if i == 2 { 1.0_f32 } else { 0.0_f32 }).into(),
            next_observation: Observation::new(vec![(i + 2) as f32]),
            terminated: terminated && i == 2,
            truncated: truncated && i == 2,
            behavior: BehaviorMetadata::default(),
        })
        .unwrap();
    }
    let c = SequenceConfig {
        burn_in: 1,
        learn_len: 2,
        n_step: 2,
    };
    SequenceBatch {
        samples: vec![sample_at(&s, 0, 1, &c).unwrap()],
        config: c,
    }
}
fn learner() -> RecurrentDdqn {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    RecurrentDdqn::new(
        model,
        spec,
        RecurrentDdqnConfig {
            gamma: 0.5,
            n_step: 2,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn temporal_execution_mode_preserves_ddqn_wire_and_legacy_imports() {
    let mut model = learner();
    let default_mode: TemporalExecutionMode = model.update_mode();
    assert_eq!(default_mode, TemporalExecutionMode::Reference);
    let legacy = model.checkpoint().unwrap();
    assert!(legacy.metadata.get("update_mode").is_none());

    let critic_mode: rl::algorithms::recurrent_critic::RecurrentUpdateMode =
        TemporalExecutionMode::OptimizedExact;
    let ddqn_mode: rl::algorithms::recurrent_ddqn::RecurrentUpdateMode = critic_mode;
    let prelude_mode: rl::prelude::RecurrentUpdateMode = ddqn_mode;
    model.set_update_mode(prelude_mode);
    let record = model.checkpoint().unwrap();
    assert_eq!(record.metadata["update_mode"], "OptimizedExact");

    let mut restored = learner();
    restored.restore(&record).unwrap();
    let restored_mode: rl::prelude::TemporalExecutionMode = restored.update_mode();
    assert_eq!(restored_mode, TemporalExecutionMode::OptimizedExact);
    restored.restore(&legacy).unwrap();
    assert_eq!(restored.update_mode(), TemporalExecutionMode::Reference);
    assert_eq!(
        serde_json::to_value(TemporalExecutionMode::Reference).unwrap(),
        "Reference"
    );
}

#[test]
fn optimized_exact_matches_reference_targets_losses_updates_and_restore() {
    for (terminated, truncated) in [(false, false), (true, false), (false, true)] {
        let data = sample(terminated, truncated);
        let mut reference = learner();
        let mut optimized = learner();
        optimized.set_update_mode(RecurrentUpdateMode::OptimizedExact);
        for _ in 0..4 {
            let a = reference.update(&data).unwrap();
            let b = optimized.update(&data).unwrap();
            for name in ["critic_loss", "target_mean"] {
                assert!((a.metrics[name] - b.metrics[name]).abs() < 2e-6, "{name}");
            }
            for (a, b) in reference
                .parameter_values()
                .unwrap()
                .iter()
                .zip(optimized.parameter_values().unwrap())
            {
                assert!(
                    (a - b).abs() <= 2e-6 + 2e-6 * a.abs(),
                    "optimized parameter drift {a} != {b}"
                );
            }
        }
        let record = optimized.checkpoint().unwrap();
        let mut restored = learner();
        restored.restore(&record).unwrap();
        assert_eq!(restored.update_mode(), RecurrentUpdateMode::OptimizedExact);
        let a = optimized.update(&data).unwrap();
        let b = restored.update(&data).unwrap();
        assert_eq!(a.metrics, b.metrics);
        assert_eq!(
            optimized.parameter_values().unwrap(),
            restored.parameter_values().unwrap()
        );
    }
}

#[test]
fn exact_double_q_targets_use_online_selection_target_evaluation_and_endings() {
    let b = sample(false, true);
    let online = vec![0., 0., 0., 0., 0., 0., 9., 1.];
    let target = vec![0., 0., 0., 0., 0., 0., 2., 100.];
    let points = construct_targets(&b, &online, &target, 2, 0.5).unwrap();
    assert_eq!(points.len(), 2);
    assert!((points[0].value - 1.0).abs() < 1e-6); // 0 + .5*1 + .25*2
    assert!((points[1].value - 2.0).abs() < 1e-6); // 1 + .5*2
    let terminal = construct_targets(&sample(true, false), &online, &target, 2, 0.5).unwrap();
    assert_eq!(terminal[0].value, 0.5);
    assert_eq!(terminal[1].value, 1.0);
}

#[test]
fn recurrent_representation_updates_and_checkpoint_numerical_continuation_is_exact() {
    let mut a = learner();
    let data = sample(true, false);
    let before = a.parameter_values().unwrap();
    let encoder_before = rl::nn::to_vec(
        a.online()
            .encoder
            .gru
            .update_gate
            .input_transform
            .weight
            .val(),
    )
    .unwrap();
    for _ in 0..3 {
        let report = a.update(&data).unwrap();
        assert!(report.metrics["critic_loss"].is_finite());
    }
    assert_ne!(before, a.parameter_values().unwrap());
    assert_ne!(
        encoder_before,
        rl::nn::to_vec(
            a.online()
                .encoder
                .gru
                .update_gate
                .input_transform
                .weight
                .val()
        )
        .unwrap()
    );
    let record = a.checkpoint().unwrap();
    let mut b = learner();
    b.restore(&record).unwrap();
    assert_eq!(a.parameter_values().unwrap(), b.parameter_values().unwrap());
    a.update(&data).unwrap();
    b.update(&data).unwrap();
    assert_eq!(a.parameter_values().unwrap(), b.parameter_values().unwrap());
    assert_eq!(a.updates(), b.updates());
    let mut invalid = record;
    invalid.metadata["version"] = 99.into();
    let before = b.parameter_values().unwrap();
    assert!(b.restore(&invalid).is_err());
    assert_eq!(before, b.parameter_values().unwrap());
}

#[test]
fn bootstrap_masks_override_online_argmax_and_termination_eliminates_bootstrap() {
    let mut b = sample(false, true);
    b.samples[0].observations[3]
        .as_dense_mut()
        .unwrap()
        .action_mask = Some(vec![false, true]);
    let online = vec![0., 0., 0., 0., 0., 0., 9., 1.];
    let target = vec![0., 0., 0., 0., 0., 0., 2., 100.];
    let p = construct_targets(&b, &online, &target, 2, 0.5).unwrap();
    assert_eq!(p[0].value, 25.5);
    assert_eq!(p[1].value, 51.);
    b.samples[0].steps[2].terminated = true;
    b.samples[0].steps[2].truncated = false;
    b.samples[0].observations[3]
        .as_dense_mut()
        .unwrap()
        .action_mask = Some(vec![false, false]);
    let p = construct_targets(&b, &online, &target, 2, 0.5).unwrap();
    assert_eq!(p[0].value, 0.5);
    assert_eq!(p[1].value, 1.);
}

#[test]
fn target_reconstructs_its_own_hidden_history_not_the_online_hidden_state() {
    use burn::module::Module;
    let mut a = learner();
    let b = sample(true, false);
    for _ in 0..20 {
        a.update(&b).unwrap();
    }
    let (online, actual) = a.reconstructed_q_values(&b).unwrap();
    let encoded = rl::nn::temporal_batch::encode_history_batch(&b, a.spec()).unwrap();
    let model = a.target().valid();
    let online_model = a.online().valid();
    let mut own = model.initial_state(1).unwrap();
    let mut other = online_model.initial_state(1).unwrap();
    let mut expected = Vec::new();
    let mut wrong = Vec::new();
    for t in 0..encoded.time {
        let x = burn::tensor::Tensor::<2>::from_data(
            burn::tensor::TensorData::new(
                encoded.inputs[t * encoded.input_dim..(t + 1) * encoded.input_dim].to_vec(),
                [1, encoded.input_dim],
            ),
            &burn::tensor::Device::flex(),
        );
        let (q, next) = model.step(x.clone(), own).unwrap();
        expected.extend(rl::nn::to_vec(q).unwrap());
        own = next;
        let (q, _) = model.step(x.clone(), other.clone()).unwrap();
        wrong.extend(rl::nn::to_vec(q).unwrap());
        other = online_model.step(x, other).unwrap().1;
    }
    assert_eq!(actual, expected);
    assert_ne!(actual, wrong);
    assert_ne!(actual, online);
}

#[test]
fn target_parameters_do_not_receive_gradients_or_move_before_synchronization() {
    let template = learner();
    let mut config = template.config().clone();
    config.ddqn.target_interval = 100;
    config.ddqn.target_tau = 1.;
    let spec = template.spec().clone();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    let mut a = RecurrentDdqn::new(model, spec, config).unwrap();
    let b = sample(true, false);
    let initial = a.reconstructed_q_values(&b).unwrap().1;
    for _ in 0..99 {
        a.update(&b).unwrap();
    }
    let (online, target) = a.reconstructed_q_values(&b).unwrap();
    assert_eq!(initial, target);
    assert_ne!(online, target);
    a.update(&b).unwrap();
    let (online, target) = a.reconstructed_q_values(&b).unwrap();
    assert_eq!(online, target);
}

#[test]
fn inference_only_models_are_rejected_before_any_backward() {
    let a = learner();
    assert!(RecurrentDdqn::new(a.online(), a.spec().clone(), a.config().clone()).is_err());
}

#[test]
fn learner_checkpoint_restores_nonparameter_gru_controls_and_future_updates() {
    let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let mut encoder = GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap();
    encoder.gru.reset_after = false;
    encoder.gru.clip = Some(0.01);
    let q = RecurrentQ::new(encoder, Mlp::new(4, &[4], 2, &mut rng).unwrap()).unwrap();
    let mut a = RecurrentDdqn::new(
        q,
        spec,
        RecurrentDdqnConfig {
            gamma: 0.5,
            n_step: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let data = sample(true, false);
    a.update(&data).unwrap();
    let record = a.checkpoint().unwrap();
    let mut b = learner();
    b.restore(&record).unwrap();
    assert_eq!(a.parameter_values().unwrap(), b.parameter_values().unwrap());
    assert_eq!(
        a.reconstructed_q_values(&data).unwrap(),
        b.reconstructed_q_values(&data).unwrap()
    );
    a.update(&data).unwrap();
    b.update(&data).unwrap();
    assert_eq!(a.parameter_values().unwrap(), b.parameter_values().unwrap());
}
