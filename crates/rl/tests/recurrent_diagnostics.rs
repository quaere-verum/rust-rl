#![cfg(feature = "neural")]

use burn::{
    module::{Module, ModuleVisitor, Param},
    tensor::Tensor,
};
use burn_pack::{Bytes, Reader};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    algorithms::recurrent_ddqn::*,
    core::{
        action::ActionSpace,
        anchors::{ANCHOR_VERSION, AnchorSet},
        checkpoint::{Checkpoint, LearnerRecord},
        reconstruction::{
            ReconstructionConfig, ReconstructionMode, ReplayRepresentations, advance_anchor,
            reset_anchor,
        },
        stateful::{HistoryInput, HistorySpec, PreviousInput, PreviousStep},
    },
    data::temporal::{SequenceBatch, SequenceOrigin},
    nn::{
        network::Mlp,
        temporal::{GruEncoder, RecurrentQ},
    },
};
use std::collections::BTreeMap;

fn fixture() -> (RecurrentDdqn, SequenceBatch) {
    let frozen: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/recurrent-ddqn-pr12.json"
    )))
    .unwrap();
    let learning = &frozen["numerical"]["learning"];
    let mut config: RecurrentDdqnConfig =
        serde_json::from_value(learning["config"].clone()).unwrap();
    // Diagnostics must expose the raw norm even when an actual optimizer step
    // would clip this fixture's gradient by a large factor.
    config.ddqn.optimizer.max_grad_norm = Some(0.01);
    let spec = HistorySpec::new(5, ActionSpace::Discrete { count: 2 }).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(17);
    let model = RecurrentQ::new(
        GruEncoder::new(spec.input_dim(), 4, &mut rng).unwrap(),
        Mlp::new(4, &[4], 2, &mut rng).unwrap(),
    )
    .unwrap();
    (
        RecurrentDdqn::new(model, spec, config).unwrap(),
        serde_json::from_value(learning["selected_sequences"].clone()).unwrap(),
    )
}

fn close(actual: f64, expected: f64) {
    assert!(
        actual.is_finite() && (actual - expected).abs() < 2e-6,
        "{actual} differs from {expected}"
    );
}

fn serialized(learner: &RecurrentDdqn) -> Vec<u8> {
    serde_json::to_vec(&learner.checkpoint().unwrap()).unwrap()
}

fn same_bytes(left: &[u8], right: &[u8], context: &str) {
    assert!(
        left == right,
        "{context}: bytes differ (lengths {} and {}, first difference {:?})",
        left.len(),
        right.len(),
        left.iter().zip(right).position(|(a, b)| a != b),
    );
}

fn checkpoint_parts(record: &LearnerRecord) -> (serde_json::Value, BTreeMap<String, Vec<u8>>) {
    let mut envelope = serde_json::to_value(record).unwrap();
    let mut packs = BTreeMap::new();
    for (name, model) in envelope["models"].as_object_mut().unwrap() {
        packs.insert(
            format!("model {name}"),
            serde_json::from_value(model["bytes"].take()).unwrap(),
        );
    }
    for (name, optimizer) in envelope["optimizers"].as_object_mut().unwrap() {
        packs.insert(
            format!("optimizer {name}"),
            serde_json::from_value(optimizer.take()).unwrap(),
        );
    }
    (envelope, packs)
}

fn same_pack(left: &[u8], right: &[u8], context: &str) {
    let left = Reader::from_bytes(Bytes::from_bytes_vec(left.to_vec())).unwrap();
    let right = Reader::from_bytes(Bytes::from_bytes_vec(right.to_vec())).unwrap();
    assert!(
        left.scalars() == right.scalars(),
        "{context}: scalars changed"
    );
    assert!(
        left.metadata() == right.metadata(),
        "{context}: optimizer parameter paths changed"
    );
    let tensors = |reader: Reader| {
        reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                (name, (dtype, shape.to_vec(), id, bytes.to_vec()))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let left = tensors(left);
    let right = tensors(right);
    assert!(
        left.keys().eq(right.keys()),
        "{context}: tensor names changed"
    );
    for (name, (dtype, shape, id, bytes)) in left {
        let (other_dtype, other_shape, other_id, other_bytes) = &right[&name];
        assert_eq!(dtype, *other_dtype, "{context} {name}: dtype changed");
        assert_eq!(shape, *other_shape, "{context} {name}: shape changed");
        assert_eq!(id, *other_id, "{context} {name}: parameter ID changed");
        same_bytes(&bytes, other_bytes, &format!("{context} {name} payload"));
    }
}

fn same_record_contents(left: &LearnerRecord, right: &LearnerRecord, context: &str) {
    let (left_envelope, left_packs) = checkpoint_parts(left);
    let (right_envelope, right_packs) = checkpoint_parts(right);
    assert!(
        left_envelope == right_envelope,
        "{context}: checkpoint configuration, counters, or parameter shapes changed"
    );
    assert!(
        left_packs.keys().eq(right_packs.keys()),
        "{context}: checkpoint pack names changed"
    );
    let mut different_containers = Vec::new();
    for (name, bytes) in left_packs {
        let other = &right_packs[&name];
        if &bytes != other {
            different_containers.push(name.clone());
        }
        same_pack(&bytes, other, &format!("{context} {name}"));
    }
    eprintln!(
        "{context}: equal named payloads; different raw containers: {different_containers:?}"
    );
}

fn model_parameters(model: &impl Module) -> Vec<f32> {
    #[derive(Default)]
    struct Collect(Vec<f32>);
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
            self.0
                .extend(rl::nn::to_vec(parameter.val().without_autodiff()).unwrap());
        }
    }
    let mut collector = Collect::default();
    model.visit(&mut collector);
    collector.0
}

fn same_parameters(left: &RecurrentDdqn, right: &RecurrentDdqn) {
    assert!(
        left.parameter_values().unwrap() == right.parameter_values().unwrap(),
        "online parameters differ"
    );
    assert!(
        model_parameters(&left.target()) == model_parameters(&right.target()),
        "target parameters differ"
    );
}

#[test]
fn diagnostics_report_the_frozen_raw_encoder_and_head_gradients() {
    // Frozen PR12 raw gradients independently establish the expected norms and
    // sparsity. An empty visitor or post-clipping norm must fail this test.
    for mode in [
        RecurrentUpdateMode::Reference,
        RecurrentUpdateMode::OptimizedExact,
    ] {
        let (mut learner, batch) = fixture();
        learner.set_update_mode(mode);
        let diagnostics = learner.diagnose_update(&batch).unwrap();
        close(diagnostics.raw_gradient_l2, 0.626906888312317);
        close(diagnostics.encoder_gradient_l2, 0.381021307659519);
        close(diagnostics.head_gradient_l2, 0.497830302134033);
        close(diagnostics.max_abs_gradient, 0.319380611181259);
        assert_eq!(diagnostics.gradient_elements, 210);
        assert_eq!(diagnostics.nonzero_gradient_elements, 133);
        close(diagnostics.critic_loss, 0.12664568424224854);
        close(diagnostics.target_mean, 0.37876900359988214);
        assert_eq!(diagnostics.learning_positions, 5);
        assert_eq!(diagnostics.approximate_online_anchor_rows, 0);
        assert_eq!(diagnostics.approximate_target_anchor_rows, 0);
        assert_eq!(diagnostics.max_direct_generation_lag, 0);
        let encoded = serde_json::to_vec(&diagnostics).unwrap();
        let decoded: RecurrentDiagnostics = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(diagnostics, decoded);
    }
}

#[test]
fn diagnostics_preserve_checkpoint_state_and_exact_optimizer_continuation() {
    for mode in [
        RecurrentUpdateMode::Reference,
        RecurrentUpdateMode::OptimizedExact,
    ] {
        let (mut learner, batch) = fixture();
        learner.set_update_mode(mode);
        // Existing Adam moments and a pending target synchronization make an
        // accidental optimizer step or counter increment observable.
        learner.update(&batch).unwrap();
        let record = learner.checkpoint().unwrap();
        let before = serde_json::to_vec(&record).unwrap();
        same_bytes(&before, &serialized(&learner), "before any diagnostics");
        let mut control = fixture().0;
        control.restore(&record).unwrap();
        same_record_contents(&record, &control.checkpoint().unwrap(), "restored control");
        same_parameters(&learner, &control);
        let first = learner.diagnose_update(&batch).unwrap();
        let second = learner.diagnose_update(&batch).unwrap();
        assert_eq!(first, second);
        same_bytes(&before, &serialized(&learner), "after two diagnostics");
        same_parameters(&learner, &control);
        assert_eq!(learner.updates(), 1);
        for update in 0..2 {
            let actual = learner.update(&batch).unwrap();
            let expected = control.update(&batch).unwrap();
            assert_eq!(actual.metrics, expected.metrics);
            same_parameters(&learner, &control);
            // Burn's optimizer record walks a HashMap, rebuilt on restore;
            // burn-pack writes tensor offsets in that iteration order. Compare
            // all named payloads/IDs and scalar state, rather than container
            // offsets, across independent restored optimizer instances.
            same_record_contents(
                &learner.checkpoint().unwrap(),
                &control.checkpoint().unwrap(),
                &format!("{mode:?} continuation update {update}"),
            );
            if update == 0 {
                close(first.critic_loss, actual.metrics["critic_loss"]);
                close(first.target_mean, actual.metrics["target_mean"]);
            }
        }
    }
}

fn anchor_first_row(learner: &RecurrentDdqn, batch: &SequenceBatch) -> SequenceBatch {
    let mut out = batch.clone();
    let original = &batch.samples[0];
    let cut = 2;
    let mut anchors = std::collections::BTreeMap::new();
    for (id, model) in learner.replay_representations().unwrap() {
        let mut anchor = reset_anchor(&model, id, original.episode_id).unwrap();
        for t in 0..cut {
            anchor = advance_anchor(
                ReconstructionConfig::default(),
                &model,
                &anchor,
                HistoryInput {
                    observation: &original.observations[t],
                    previous: t.checked_sub(1).map(|i| PreviousInput {
                        factual: None,
                        action: &original.steps[i].action,
                        reward: original.steps[i].reward.total(),
                    }),
                },
            )
            .unwrap()
            .0;
        }
        anchors.insert(id, anchor);
    }
    let previous = &original.steps[cut - 1];
    let sample = &mut out.samples[0];
    sample.history_start = cut;
    sample.previous = Some(PreviousStep {
        action: previous.action.clone(),
        reward: previous.reward.total(),
    });
    sample.origin = SequenceOrigin::Anchored(AnchorSet {
        version: ANCHOR_VERSION,
        anchors,
    });
    sample.observations = original.observations[cut..].to_vec();
    sample.steps = original.steps[cut..].to_vec();
    sample.loss_mask = original.loss_mask[cut..].to_vec();
    sample.learn_start -= cut;
    sample.burn_start = sample.learn_start.saturating_sub(out.config.burn_in);
    out.validate().unwrap();
    out
}

#[test]
fn diagnostics_report_stale_independent_anchors_in_a_padded_masked_batch() {
    for mode in [
        RecurrentUpdateMode::Reference,
        RecurrentUpdateMode::OptimizedExact,
    ] {
        let (mut learner, full) = fixture();
        learner.set_update_mode(mode);
        let anchored = anchor_first_row(&learner, &full);
        learner.update(&full).unwrap();
        let before_rejection = serialized(&learner);
        assert!(learner.diagnose_update(&anchored).is_err());
        same_bytes(
            &before_rejection,
            &serialized(&learner),
            "rejected stale anchors",
        );
        learner
            .set_reconstruction_config(ReconstructionConfig {
                version: 1,
                mode: ReconstructionMode::Anchored {
                    max_generation_lag: Some(2),
                },
            })
            .unwrap();
        let record = learner.checkpoint().unwrap();
        let before = serde_json::to_vec(&record).unwrap();
        same_bytes(
            &before,
            &serialized(&learner),
            "before anchored diagnostics",
        );
        let mut control = fixture().0;
        control.restore(&record).unwrap();
        same_record_contents(
            &record,
            &control.checkpoint().unwrap(),
            "restored anchored control",
        );
        same_parameters(&learner, &control);
        let diagnostics = learner.diagnose_update(&anchored).unwrap();
        assert_eq!(diagnostics.approximate_online_anchor_rows, 1);
        // The target has not synchronized at update one; its identity is exact
        // even though the anchor's direct generation lag is one.
        assert_eq!(diagnostics.approximate_target_anchor_rows, 0);
        assert_eq!(diagnostics.max_direct_generation_lag, 1);
        assert_eq!(diagnostics.learning_positions, 5);
        assert!(
            diagnostics.encoder_gradient_l2.is_finite() && diagnostics.encoder_gradient_l2 > 0.
        );
        assert!(diagnostics.head_gradient_l2.is_finite() && diagnostics.head_gradient_l2 > 0.);
        same_bytes(&before, &serialized(&learner), "after anchored diagnostics");
        same_parameters(&learner, &control);
        for update in 0..2 {
            let actual = learner.update(&anchored).unwrap();
            let expected = control.update(&anchored).unwrap();
            assert_eq!(actual.metrics, expected.metrics);
            same_parameters(&learner, &control);
            same_record_contents(
                &learner.checkpoint().unwrap(),
                &control.checkpoint().unwrap(),
                &format!("{mode:?} anchored continuation update {update}"),
            );
            if update == 0 {
                close(diagnostics.critic_loss, actual.metrics["critic_loss"]);
            }
        }
    }
}

#[test]
fn diagnostic_validation_rejects_invalid_batches_without_mutation() {
    let (learner, mut batch) = fixture();
    let before = serialized(&learner);
    batch.config.n_step += 1;
    assert!(learner.diagnose_update(&batch).is_err());
    same_bytes(&before, &serialized(&learner), "n-step mismatch rejection");
    batch.samples.clear();
    assert!(learner.diagnose_update(&batch).is_err());
    same_bytes(&before, &serialized(&learner), "empty batch rejection");
}
