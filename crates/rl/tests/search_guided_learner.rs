#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;

use burn::{
    module::Param,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::{
        OptimizerConfig,
        search_guided::{
            PolicyValueBatch, PolicyValueRow, SearchGuidedConfig, SearchGuidedPolicyValue,
        },
    },
    core::{checkpoint::Checkpoint, stateful::StatefulLearner},
    data::temporal::{SequenceConfig, sample_at},
    nn::structured::{LightweightPolicy, LightweightValue},
};

fn models() -> Result<(LightweightPolicy, LightweightValue)> {
    let config = doors::model_config().model;
    let device = Device::flex().autodiff();
    Ok((
        LightweightPolicy::new(config.clone(), &mut ChaCha8Rng::seed_from_u64(37), &device)?,
        LightweightValue::new(config, &mut ChaCha8Rng::seed_from_u64(53), &device)?,
    ))
}

#[test]
fn versioned_physical_objective_is_incompatible_with_legacy_publications_and_records() -> Result<()>
{
    use rl::models::predictive::ReturnObjective;

    let mut legacy = learner()?;
    let (policy, value) = models()?;
    let objective = ReturnObjective::PhysicalVersioned([91; 32]);
    let mut experimental =
        SearchGuidedPolicyValue::new_with_objective(policy, value, config(), objective)?;
    let previous = legacy.publish()?;
    let current = experimental.publish()?;
    assert_eq!(previous.returns().gamma, current.returns().gamma);
    assert_ne!(previous.identity(), current.identity());
    assert!(previous.returns().check(&current.returns()).is_err());
    assert_eq!(current.returns().objective, objective);
    let legacy_record = legacy.checkpoint()?;
    assert!(legacy_record.metadata.get("objective").is_none());
    assert!(experimental.restore(&legacy_record).is_err());
    let experimental_record = experimental.checkpoint()?;
    assert!(legacy.restore(&experimental_record).is_err());
    let (policy, value) = models()?;
    let mut compatible =
        SearchGuidedPolicyValue::new_with_objective(policy, value, config(), objective)?;
    compatible.restore(&experimental_record)?;
    assert_eq!(compatible.publish()?.identity(), current.identity());
    Ok(())
}

fn learner() -> Result<SearchGuidedPolicyValue> {
    let (policy, value) = models()?;
    SearchGuidedPolicyValue::new(policy, value, config())
}
fn config() -> SearchGuidedConfig {
    SearchGuidedConfig {
        gamma: 0.9,
        policy_optimizer: OptimizerConfig {
            learning_rate: 0.003,
            ..Default::default()
        },
        value_optimizer: OptimizerConfig {
            learning_rate: 0.003,
            ..Default::default()
        },
        max_pack_elements: 1024 * 1024,
    }
}
fn batch() -> Result<PolicyValueBatch> {
    let sample = sample_at(
        &doors::factual_store()?,
        7,
        0,
        &SequenceConfig {
            burn_in: 0,
            learn_len: 2,
            n_step: 1,
        },
    )?;
    Ok(PolicyValueBatch {
        rows: vec![
            PolicyValueRow {
                sample: sample.clone(),
                step: 0,
                actions: vec![doors::SCAN, doors::guess(false), doors::guess(true)],
                probabilities: vec![0.8, 0.1, 0.1],
            },
            PolicyValueRow {
                sample,
                step: 1,
                actions: vec![doors::guess(false), doors::guess(true)],
                probabilities: vec![0.05, 0.95],
            },
        ],
    })
}
fn record(learner: &SearchGuidedPolicyValue) -> Result<serde_json::Value> {
    // Burn-pack tensor order is unspecified after loading. Compare exact tensor
    // bits, IDs, shapes, scalar clocks and metadata rather than container order.
    fn packed(value: &mut serde_json::Value) -> Result<()> {
        let bytes: Vec<u8> = serde_json::from_value(value.take())?;
        let reader =
            burn_pack::Reader::from_bytes(burn::tensor::Bytes::from_bytes_vec(bytes)).unwrap();
        let scalars = format!("{:?}", reader.scalars());
        let metadata = reader.metadata().clone();
        let tensors = reader
            .into_tensors()
            .unwrap()
            .into_iter()
            .map(|tensor| {
                let (name, dtype, shape, id, bytes) = tensor.into_parts().unwrap();
                (
                    name,
                    serde_json::json!([format!("{dtype:?}"), shape, id, bytes.to_vec()]),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        *value = serde_json::json!({"scalars": scalars, "metadata": metadata, "tensors": tensors});
        Ok(())
    }
    let mut record = serde_json::to_value(learner.checkpoint()?)?;
    for model in record["models"].as_object_mut().unwrap().values_mut() {
        packed(&mut model["bytes"])?;
    }
    for optimizer in record["optimizers"].as_object_mut().unwrap().values_mut() {
        packed(optimizer)?;
    }
    Ok(record)
}

#[test]
fn fits_neutral_complete_distributions_and_factual_values_then_resumes_exactly() -> Result<()> {
    let mut learner = learner()?;
    let batch = batch()?;
    let before = learner.losses(&batch)?;
    for _ in 0..32 {
        learner.update(&batch)?;
    }
    let after = learner.losses(&batch)?;
    eprintln!(
        "paired supervision: CE {} -> {}; MC MSE {} -> {}",
        before.0, after.0, before.1, after.1
    );
    assert!(after.0 < before.0 * 0.8);
    assert!(after.1 < before.1 * 0.5);
    assert_eq!(learner.updates(), 32);
    let saved = learner.checkpoint()?;
    let (p, v) = models()?;
    let mut restored = SearchGuidedPolicyValue::new(p, v, config())?;
    restored.restore(&saved)?;
    let a = learner.update(&batch)?;
    let b = restored.update(&batch)?;
    assert_eq!(a.metrics, b.metrics);
    let actual = record(&restored)?;
    let expected = record(&learner)?;
    assert!(
        actual["metadata"] == expected["metadata"],
        "restored learner metadata differs"
    );
    assert!(
        actual["models"] == expected["models"],
        "restored parameters differ"
    );
    assert!(
        actual["optimizers"] == expected["optimizers"],
        "restored moments or clocks differ"
    );
    Ok(())
}

#[test]
fn malformed_rows_and_records_leave_both_models_optimizers_and_clocks_intact() -> Result<()> {
    let mut learner = learner()?;
    learner.update(&batch()?)?;
    let before = record(&learner)?;
    let mut invalid = vec![PolicyValueBatch { rows: vec![] }];
    for change in 0..5 {
        let mut b = batch()?;
        match change {
            0 => b.rows[1].actions.swap(0, 1),
            1 => b.rows[1].probabilities[0] = -0.05,
            2 => b.rows[1].probabilities[0] = f64::NAN,
            3 => b.rows[1].sample.steps[1].terminated = false,
            _ => b.rows[1].sample.loss_mask[1] = false,
        }
        invalid.push(b);
    }
    for b in invalid {
        assert!(learner.update(&b).is_err());
        assert_eq!(record(&learner)?, before);
    }
    for change in 0..4 {
        let mut bad = learner.checkpoint()?;
        match change {
            0 => bad.metadata["updates"] = 999.into(),
            1 => {
                bad.optimizers.remove("value");
            }
            2 => bad.metadata["config"]["gamma"] = 0.5.into(),
            _ => bad.metadata["policy_optimizer_ids"] = serde_json::json!([]),
        }
        assert!(learner.restore(&bad).is_err());
        assert_eq!(record(&learner)?, before);
    }
    Ok(())
}

#[test]
fn bad_value_gradient_cannot_partially_install_a_valid_policy_update() -> Result<()> {
    let (policy, mut value) = models()?;
    let device = Device::flex().autodiff();
    // Finite zero predictions, but d(loss)/d(final weight) is enormous. Fresh
    // leaves keep this an actual optimizer guard test, not a detached graph.
    let first = &mut value.head.layers[0];
    first.weight = Param::from_tensor(Tensor::zeros(first.weight.val().dims(), &device));
    let bias = first.bias.as_mut().unwrap();
    *bias = Param::from_tensor(Tensor::full(bias.val().dims(), 1e20, &device));
    let last = value.head.layers.last_mut().unwrap();
    last.weight = Param::from_tensor(Tensor::zeros(last.weight.val().dims(), &device));
    let bias = last.bias.as_mut().unwrap();
    *bias = Param::from_tensor(Tensor::zeros(bias.val().dims(), &device));
    let mut learner = SearchGuidedPolicyValue::new(policy, value, config())?;
    assert!(learner.losses(&batch()?)?.1.is_finite());
    let before = record(&learner)?;
    assert!(learner.update(&batch()?).is_err());
    assert_eq!(record(&learner)?, before);
    Ok(())
}

#[test]
fn independent_heads_must_not_share_encoder_parameters() -> Result<()> {
    let (policy, mut value) = models()?;
    value.encoder = policy.encoder.clone();
    assert!(SearchGuidedPolicyValue::new(policy, value, config()).is_err());
    Ok(())
}

#[test]
fn publications_are_frozen_identifiable_and_restore_for_inference() -> Result<()> {
    use burn::module::{Module, ModuleVisitor};
    use rl::{
        algorithms::search_guided::PolicyValuePublication,
        core::{actor::ActMode, stateful::StatefulActor},
    };
    let mut learner = learner()?;
    let observation = doors::frame(doors::Public { reading: None });
    let original = learner.publish()?;
    let law = original.action_probabilities(&observation)?;
    let value = original.value_prediction(&observation)?;
    for _ in 0..4 {
        learner.update(&batch()?)?;
    }
    assert_eq!(law, original.action_probabilities(&observation)?);
    assert_eq!(value, original.value_prediction(&observation)?);
    let current = learner.publish()?;
    assert_eq!(current.revision(), original.revision() + 1);
    assert_eq!(current.source_updates(), 4);
    assert_ne!(current.identity(), original.identity());
    let record = serde_json::to_value(&current)?;
    let restored: PolicyValuePublication = serde_json::from_value(record.clone())?;
    assert_eq!(restored.identity(), current.identity());
    assert_eq!(
        restored.action_probabilities(&observation)?,
        current.action_probabilities(&observation)?
    );
    assert_eq!(
        restored.value_prediction(&observation)?,
        current.value_prediction(&observation)?
    );
    struct Inference;
    impl ModuleVisitor for Inference {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            assert!(!p.val().is_require_grad());
            assert!(!p.val().is_tracked());
        }
    }
    restored.policy().visit(&mut Inference);
    restored.value().visit(&mut Inference);
    let greedy = restored.decide(
        &(),
        &observation,
        ActMode::Deterministic,
        &mut ChaCha8Rng::seed_from_u64(4),
    )?;
    assert_eq!(greedy.log_prob, Some(0.));
    let mut bad = record;
    bad["metadata"]["identity"][0] = (u64::from(current.identity()[0]) ^ 1).into();
    assert!(serde_json::from_value::<PolicyValuePublication>(bad).is_err());
    Ok(())
}
