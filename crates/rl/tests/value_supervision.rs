#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;
use rl::{
    Result,
    core::reward::Reward,
    data::{
        temporal::{SequenceConfig, SequenceSample, sample_at},
        value_supervision::ValueSupervision,
    },
    nn::{to_vec, vector},
    objectives::value_supervision::squared_loss,
};

fn sequence(learn_start: usize) -> Result<SequenceSample> {
    sample_at(
        &doors::factual_store()?,
        7,
        learn_start,
        &SequenceConfig {
            burn_in: learn_start,
            learn_len: 2 - learn_start,
            n_step: 1,
        },
    )
}

#[test]
fn mean_squared_loss_and_gradients_match_independent_scalar_reference() -> Result<()> {
    let predictions = vector(&[2., -1., 0.5]).require_grad();
    let targets = [1., 1., 2.];
    let loss = squared_loss(predictions.clone(), &targets)?;
    let reference = (1. + 4. + 2.25) / 3.;
    assert!((f64::from(loss.clone().into_scalar::<f32>()) - reference).abs() < 2e-7);
    for (actual, expected) in to_vec(predictions.grad(&loss.backward()).unwrap())?
        .iter()
        .zip([2. / 3., -4. / 3., -1.])
    {
        assert!((actual - expected).abs() < 1e-7);
    }
    Ok(())
}

#[test]
fn invalid_regression_values_shapes_and_overflow_reject() {
    for values in [
        vec![],
        vec![1.],
        vec![f32::NAN, 2.],
        vec![f32::INFINITY, 2.],
    ] {
        assert!(squared_loss(vector(&[1., 2.]), &values).is_err());
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(squared_loss(vector(&[bad]), &[0.]).is_err());
    }
}

#[test]
fn nonfinite_value_gradient_rejects_before_any_parameter_or_adam_mutation() -> Result<()> {
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
    let record = |model: &Mlp, adam: &burn::optim::ModuleOptimizer| -> Result<Vec<u8>> {
        let mut record = LearnerRecord::new("value-guard", ())?;
        record.model("value", model)?;
        record.optimizer("value", adam)?;
        Ok(serde_json::to_vec(&record)?)
    };
    let before = record(&model, &adam)?;
    let predictions = model.forward(matrix(vec![1.], 1, 1)).sqrt().reshape([1]);
    let loss = squared_loss(predictions, &[1.])?;
    assert!(optimize_validated(&mut model, &mut adam, 0.001, loss).is_err());
    assert_eq!(before, record(&model, &adam)?);
    Ok(())
}

#[test]
fn monte_carlo_targets_discount_actual_primitives_within_and_between_actions() -> Result<()> {
    let mut sample = sequence(0)?;
    sample.steps[0].reward = Reward::from_primitives(vec![1., 2.])?;
    sample.steps[1].reward = Reward::from_primitives(vec![3., 4., 5.])?;
    for gamma in [0_f64, 0.5, 0.9, 1.] {
        let first = ValueSupervision::monte_carlo(&sample, 0, gamma)?;
        let reference = [1., 2., 3., 4., 5.]
            .into_iter()
            .enumerate()
            .map(|(tick, reward)| reward * gamma.powi(tick as i32))
            .sum::<f64>();
        assert_eq!(first.target(), reference as f32);
        assert_eq!(first.gamma(), gamma);
        assert_eq!(first.step(), 0);
        assert_eq!(first.sample(), &sample);
        let second = ValueSupervision::monte_carlo(&sample, 1, gamma)?;
        assert_eq!(
            second.target(),
            (3. + gamma * 4. + gamma * gamma * 5.) as f32
        );
    }
    Ok(())
}

#[test]
fn incomplete_truncated_nonfinite_and_malformed_factual_tails_reject() -> Result<()> {
    let original = sequence(0)?;
    for mutation in 0..11 {
        let mut sample = original.clone();
        match mutation {
            0 => sample.steps[1].terminated = false,
            1 => {
                sample.steps[1].terminated = false;
                sample.steps[1].truncated = true;
            }
            2 => sample.steps[0].terminated = true,
            3 => sample.steps[0].truncated = true,
            4 => sample.steps[1].reward = f32::NAN.into(),
            5 => {
                sample.observations.pop();
            }
            6 => sample.loss_mask[0] = false,
            7 => sample.learn_start = 1,
            8 => sample.trajectory_start = 1,
            9 => sample.history_start = usize::MAX,
            _ => {
                sample.loss_mask.pop();
            }
        }
        assert!(
            ValueSupervision::monte_carlo(&sample, 0, 0.9).is_err(),
            "mutation {mutation}"
        );
    }
    for gamma in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(ValueSupervision::monte_carlo(&original, 0, gamma).is_err());
    }
    assert!(ValueSupervision::monte_carlo(&original, usize::MAX, 0.9).is_err());
    let mut both = original;
    both.steps[1].truncated = true;
    assert_eq!(ValueSupervision::monte_carlo(&both, 0, 0.9)?.target(), 0.81);
    let masked = sequence(1)?;
    assert_eq!(masked.loss_mask, [false, true]);
    assert!(ValueSupervision::monte_carlo(&masked, 0, 0.9).is_err());
    ValueSupervision::monte_carlo(&masked, 1, 0.9)?;
    Ok(())
}
