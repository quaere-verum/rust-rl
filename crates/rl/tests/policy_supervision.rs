#![cfg(feature = "neural")]

use burn::tensor::activation::log_softmax;
#[path = "support/planning_doors.rs"]
mod doors;
use rl::{
    nn::{to_vec, vector},
    objectives::policy_supervision::distribution_loss,
};

#[test]
fn ragged_rows_have_equal_weight_and_independent_logit_gradients() {
    let first = vector(&[0., 3f32.ln()]).require_grad();
    let second = vector(&[0., 0., 0.]).require_grad();
    let loss = distribution_loss(
        vec![
            log_softmax(first.clone(), 0),
            log_softmax(second.clone(), 0),
        ],
        &[vec![0.75, 0.25], vec![0., 0.25, 0.75]],
    )
    .unwrap();
    let expected = -(0.75 * 0.25f64.ln() + 0.25 * 0.75f64.ln() + (1f64 / 3.).ln()) / 2.;
    assert!((f64::from(loss.clone().into_scalar::<f32>()) - expected).abs() < 1e-6);
    let grads = loss.backward();
    for (actual, expected) in to_vec(first.grad(&grads).unwrap())
        .unwrap()
        .iter()
        .zip([-0.25, 0.25])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in
        to_vec(second.grad(&grads).unwrap())
            .unwrap()
            .iter()
            .zip([1. / 6., 1. / 24., -5. / 24.])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn malformed_targets_and_nonfinite_likelihoods_reject() {
    for target in [
        vec![],
        vec![1.],
        vec![0.2, 0.2],
        vec![-0.1, 1.1],
        vec![f32::NAN, 1.],
        vec![f32::INFINITY, 0.],
    ] {
        assert!(distribution_loss(vec![vector(&[-1., -1.])], &[target]).is_err());
    }
    assert!(distribution_loss(vec![], &[]).is_err());
    assert!(distribution_loss(vec![vector(&[-1., -1.])], &[]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(distribution_loss(vec![vector(&[0., bad])], &[vec![1., 0.]]).is_err());
    }
}

#[test]
fn complete_factual_support_and_learning_mask_are_required() -> rl::Result<()> {
    use rl::data::{
        policy_supervision::PolicySupervision,
        temporal::{SequenceConfig, sample_at},
    };
    let store = doors::factual_store()?;
    let mut sample = sample_at(
        &store,
        7,
        1,
        &SequenceConfig {
            burn_in: 1,
            learn_len: 1,
            n_step: 1,
        },
    )?;
    let keys = [doors::guess(false), doors::guess(true)];
    PolicySupervision::new(&sample, 1, &keys, &[0.2, 0.8])?;
    assert!(
        PolicySupervision::new(&sample, 0, &[doors::SCAN, keys[0], keys[1]], &[1., 0., 0.])
            .is_err()
    );
    for bad in [
        vec![keys[1], keys[0]],
        vec![keys[0]],
        vec![keys[0], keys[0]],
        vec![doors::SCAN, keys[0], keys[1]],
    ] {
        let mass = vec![1. / bad.len() as f64; bad.len()];
        assert!(PolicySupervision::new(&sample, 1, &bad, &mass).is_err());
    }
    for mass in [[-0.1, 1.1], [0.1, 0.1], [f64::NAN, 1.]] {
        assert!(PolicySupervision::new(&sample, 1, &keys, &mass).is_err());
    }
    sample.loss_mask[1] = false;
    assert!(PolicySupervision::new(&sample, 1, &keys, &[0.2, 0.8]).is_err());
    Ok(())
}

#[test]
fn nonfinite_gradient_rejects_before_model_and_adam_mutation() -> rl::Result<()> {
    use burn::module::Param;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        algorithms::OptimizerConfig,
        core::checkpoint::LearnerRecord,
        nn::{
            checkpoint::RecordBurnExt,
            matrix,
            network::{Mlp, Network},
            optimization::optimize_validated,
        },
    };
    let mut model = Mlp::new(1, &[], 1, &mut ChaCha8Rng::seed_from_u64(8))?;
    model.layers[0].weight = Param::from_tensor(matrix(vec![0.], 1, 1));
    model.layers[0].bias = None;
    let mut adam = OptimizerConfig::default().build()?;
    let record = |model: &Mlp, adam: &burn::optim::ModuleOptimizer| -> rl::Result<Vec<u8>> {
        let mut record = LearnerRecord::new("guard", ())?;
        record.model("model", model)?;
        record.optimizer("adam", adam)?;
        Ok(serde_json::to_vec(&record)?)
    };
    let before = record(&model, &adam)?;
    // Finite log likelihood (zero), but d sqrt(x)/dx at x=0 is infinite.
    let logp = model
        .forward(matrix(vec![1.], 1, 1))
        .sqrt()
        .reshape([1])
        .neg();
    let loss = distribution_loss(vec![logp], &[vec![1.]])?;
    assert!(loss.clone().into_scalar::<f32>().is_finite());
    assert!(optimize_validated(&mut model, &mut adam, 0.001, loss).is_err());
    assert_eq!(before, record(&model, &adam)?);
    Ok(())
}
