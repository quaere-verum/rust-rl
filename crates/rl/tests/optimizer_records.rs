#![cfg(feature = "neural")]

use burn_pack::{Bytes, Reader, Scalar, Tensor, Writer};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{core::checkpoint::LearnerRecord, prelude::*};
use std::collections::BTreeMap;

#[test]
fn inference_record_loading_preserves_column_layout_and_validation() {
    use burn::{
        module::Module,
        tensor::{Device, Tensor as BurnTensor},
    };
    use rl::nn::checkpoint::RecordBurnExt;
    let device = Device::flex();
    let template = burn::nn::LinearConfig::new(2, 1)
        .with_layout(burn::nn::LinearLayout::Col)
        .with_initializer(burn::nn::Initializer::Constant { value: 0.1 })
        .init(&device);
    let mut record = LearnerRecord::new("inference", ()).unwrap();
    record.model("linear", &template).unwrap();
    let restored = record.load_model_inference("linear", &template).unwrap();
    assert_eq!(restored.weight.val().dims(), [2, 1]);
    assert!(!restored.weight.val().is_autodiff());
    assert!(
        !restored.clone().train().weight.val().is_require_grad(),
        "restored parameters stay frozen"
    );
    assert!(
        template.clone().train().weight.val().is_require_grad(),
        "inference restore does not freeze the caller's template"
    );
    let input = BurnTensor::<2>::from_data([[2., 3.]], &device);
    assert_eq!(
        template.forward(input.clone()).into_scalar::<f32>(),
        restored.forward(input).into_scalar::<f32>()
    );
    assert!(record.load_model_inference("missing", &template).is_err());
    for corruption in 0..3 {
        let mut json = serde_json::to_value(&record).unwrap();
        let model = &mut json["models"]["linear"];
        match corruption {
            0 => model["signature"] = "wrong".into(),
            1 => model["parameter_shapes"]["weight"] = serde_json::json!([99, 1]),
            _ => model["bytes"] = serde_json::json!([]),
        }
        let bad: LearnerRecord = serde_json::from_value(json).unwrap();
        assert!(bad.load_model_inference("linear", &template).is_err());
    }
}

#[derive(burn::module::Module, Debug)]
struct ColumnPolicy {
    linear: burn::nn::Linear,
    #[module(skip)]
    space: ActionSpace,
}
impl rl::nn::policy::DeterministicPolicy for ColumnPolicy {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn actions(&self, observations: ObservationBatch<'_>) -> rl::Result<burn::tensor::Tensor<2>> {
        Ok(self.linear.forward(rl::nn::observations(observations, 2)?))
    }
}

#[test]
fn column_layout_model_and_optimizer_restore_deterministically() {
    let mut rng = ChaCha8Rng::seed_from_u64(107);
    let policy = ColumnPolicy {
        linear: burn::nn::LinearConfig::new(2, 1)
            .with_layout(burn::nn::LinearLayout::Col)
            .with_initializer(burn::nn::Initializer::Constant { value: 0.1 })
            .init(&rl::nn::device()),
        space: ActionSpace::Unbounded { dimensions: 1 },
    };
    let critic = QValue::new(2, 1, 1, &[], &mut rng).unwrap();
    let mut original = Ddpg::new(policy.clone(), critic.clone(), Default::default(), 109).unwrap();
    let mut restored = Ddpg::new(policy, critic, Default::default(), 113).unwrap();
    original.update(&batch()).unwrap();
    restored.restore(&original.checkpoint().unwrap()).unwrap();
    assert_eq!(restored.policy().linear.weight.val().dims(), [2, 1]);
    assert_eq!(
        original.update(&batch()).unwrap().metrics,
        restored.update(&batch()).unwrap().metrics
    );
    let a = original.checkpoint().unwrap();
    let b = restored.checkpoint().unwrap();
    for (name, bytes) in &a.optimizers {
        same_pack(bytes, &b.optimizers[name]);
    }
}

fn learner() -> Ddpg {
    let mut rng = ChaCha8Rng::seed_from_u64(101);
    let policy = Deterministic::new(
        2,
        ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        },
        &[],
        &mut rng,
    )
    .unwrap();
    let critic = QValue::new(2, 1, 1, &[], &mut rng).unwrap();
    Ddpg::new(policy, critic, Default::default(), 103).unwrap()
}

fn batch() -> ReplayBatch {
    ReplayBatch {
        items: vec![rl::data::replay::ReplayItem {
            transition: rl::core::transition::Transition {
                observation: Observation::new(vec![1., 2.]),
                action: Action::Continuous(vec![0.]),
                reward: 1.0_f32.into(),
                next_observation: Observation::new(vec![0., 0.]),
                terminated: true,
                truncated: false,
            },
            discount: 0.,
        }],
    }
}

fn rewrite(
    record: &mut LearnerRecord,
    edit: impl FnOnce(&mut Vec<Tensor>, &mut BTreeMap<String, Scalar>),
) {
    let reader =
        Reader::from_bytes(Bytes::from_bytes_vec(record.optimizers["actor"].clone())).unwrap();
    let mut scalars = reader.scalars().clone();
    let metadata = reader.metadata().clone();
    let mut tensors = reader.into_tensors().unwrap();
    edit(&mut tensors, &mut scalars);
    let mut writer = Writer::new(tensors);
    for (name, value) in scalars {
        writer = writer.with_scalar(&name, value);
    }
    for (name, value) in metadata {
        writer = writer.with_metadata(&name, &value);
    }
    record
        .optimizers
        .insert("actor".into(), writer.into_bytes().unwrap().to_vec());
}

fn rejects(edit: impl FnOnce(&mut Vec<Tensor>, &mut BTreeMap<String, Scalar>)) {
    let mut live = learner();
    live.update(&batch()).unwrap();
    let good = live.checkpoint().unwrap();
    let mut broken = good.clone();
    rewrite(&mut broken, edit);
    assert!(
        live.restore(&broken).is_err(),
        "malformed optimizer was accepted"
    );
    assert_eq!(
        serde_json::to_value(live.checkpoint().unwrap()).unwrap(),
        serde_json::to_value(good).unwrap(),
        "failed restoration changed live learner"
    );
}

fn same_pack(left: &[u8], right: &[u8]) {
    let a = Reader::from_bytes(Bytes::from_bytes_vec(left.to_vec())).unwrap();
    let b = Reader::from_bytes(Bytes::from_bytes_vec(right.to_vec())).unwrap();
    assert_eq!(a.scalars(), b.scalars());
    assert_eq!(a.metadata(), b.metadata());
    let tensors = |reader: Reader| {
        reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|t| {
                let (name, dtype, shape, id, bytes) = t.into_parts().unwrap();
                (name, (dtype, shape, id, bytes.to_vec()))
            })
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(tensors(a), tensors(b));
}

#[test]
fn missing_optimizer_moment_is_rejected_atomically() {
    rejects(|tensors, _| {
        let i = tensors
            .iter()
            .position(|t| t.name.ends_with(".moment_1"))
            .unwrap();
        tensors.remove(i);
    });
}

#[test]
fn missing_optimizer_parameter_id_returns_error_without_panicking() {
    rejects(|tensors, _| tensors[0].param_id = None);
}

#[test]
fn inconsistent_moment_shapes_are_rejected_atomically() {
    rejects(|tensors, _| {
        let tensor = tensors
            .iter_mut()
            .find(|t| t.shape == [2, 1].into())
            .unwrap();
        tensor.shape = [1, 2].into();
    });
}

#[test]
fn matching_moments_with_wrong_parameter_shape_are_rejected_atomically() {
    rejects(|tensors, _| {
        for tensor in tensors.iter_mut().filter(|t| t.shape == [2, 1].into()) {
            tensor.shape = [1, 2].into();
        }
    });
}

#[test]
fn invalid_optimizer_rank_returns_error_without_panicking() {
    rejects(|_, scalars| {
        let key = scalars
            .keys()
            .find(|key| key.ends_with(".__rank"))
            .unwrap()
            .clone();
        scalars.insert(key, Scalar::from(99usize));
    });
}

#[test]
fn missing_optimizer_step_counter_is_rejected_atomically() {
    rejects(|_, scalars| {
        let key = scalars
            .keys()
            .find(|key| key.ends_with(".time"))
            .unwrap()
            .clone();
        scalars.remove(&key);
    });
}

#[test]
fn empty_and_initialized_optimizer_records_continue_deterministically() {
    for initialized in [false, true] {
        let mut original = learner();
        if initialized {
            original.update(&batch()).unwrap();
        }
        let good = original.checkpoint().unwrap();
        let mut restored = learner();
        restored.restore(&good).unwrap();
        let a = original.update(&batch()).unwrap();
        let b = restored.update(&batch()).unwrap();
        assert_eq!(a.metrics, b.metrics);
        let a = original.checkpoint().unwrap();
        let b = restored.checkpoint().unwrap();
        assert_eq!(a.metadata, b.metadata);
        // Burn packs tensor data in HashMap iteration order, so compare entries
        // rather than byte offsets/padding in the serialized container.
        for (name, bytes) in &a.optimizers {
            same_pack(bytes, &b.optimizers[name]);
        }
        let a = serde_json::to_value(a).unwrap();
        let b = serde_json::to_value(b).unwrap();
        for (name, model) in a["models"].as_object().unwrap() {
            let bytes = |value: &serde_json::Value| {
                serde_json::from_value::<Vec<u8>>(value["bytes"].clone()).unwrap()
            };
            same_pack(&bytes(model), &bytes(&b["models"][name]));
        }
    }
}
