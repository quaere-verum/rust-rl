#![cfg(feature = "neural")]

#[path = "support/planning_doors.rs"]
mod doors;
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{Device, Tensor},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Result,
    algorithms::OptimizerConfig,
    core::{action::Action, checkpoint::LearnerRecord},
    data::{
        temporal::{SequenceConfig, SequenceOrigin, SequenceSample, sample_at},
        value_supervision::ValueSupervision,
    },
    nn::{
        checkpoint::RecordBurnExt,
        optimization::optimize_validated,
        structured::{
            LightweightValue, PackLimits, RecurrentLightweightValue, StructuredHistoryInput,
            recurrent_value_supervision_loss, value_supervision_loss,
        },
        to_vec,
    },
    objectives::value_supervision::squared_loss,
};
use std::sync::Arc;

fn finite_loss(loss: &Tensor<1>) -> Result<f64> {
    let value = f64::from(loss.clone().into_scalar::<f32>());
    assert!(value.is_finite());
    Ok(value)
}

fn sequence(start: usize) -> Result<SequenceSample> {
    sample_at(
        &doors::factual_store()?,
        7,
        start,
        &SequenceConfig {
            burn_in: start,
            learn_len: 2 - start,
            n_step: 1,
        },
    )
}

#[test]
fn existing_feedforward_value_and_adam_fit_real_complete_returns() -> Result<()> {
    let sample = sequence(0)?;
    let rows = [
        ValueSupervision::monte_carlo(&sample, 0, 0.9)?,
        ValueSupervision::monte_carlo(&sample, 1, 0.9)?,
    ];
    let mut model = LightweightValue::new(
        doors::model_config().model,
        &mut ChaCha8Rng::seed_from_u64(37),
        &Device::flex().autodiff(),
    )?;
    let limits = PackLimits::default();
    let before = finite_loss(&value_supervision_loss(&model, &rows, limits)?)?;
    let mut adam = OptimizerConfig::default().build()?;
    for _ in 0..24 {
        let loss = value_supervision_loss(&model, &rows, limits)?;
        optimize_validated(&mut model, &mut adam, 0.003, loss)?;
    }
    let after = finite_loss(&value_supervision_loss(&model, &rows, limits)?)?;
    eprintln!("feedforward factual MSE {before:.9} -> {after:.9}");
    assert!(after < before * 0.5);
    // Model and Adam persist through the existing record machinery.
    let mut record = LearnerRecord::new("value-fit-test", ())?;
    record.model("value", &model)?;
    record.optimizer("value", &adam)?;
    let restored = record.load_model("value", &model)?;
    assert_eq!(
        after,
        finite_loss(&value_supervision_loss(&restored, &rows, limits)?)?
    );
    let mut restored_adam = record.load_optimizer("value", OptimizerConfig::default().build()?)?;
    let mut restored = restored;
    for (value, optimizer) in [(&mut model, &mut adam), (&mut restored, &mut restored_adam)] {
        let loss = value_supervision_loss(value, &rows, limits)?;
        optimize_validated(value, optimizer, 0.003, loss)?;
    }
    assert_eq!(
        finite_loss(&value_supervision_loss(&model, &rows, limits)?)?,
        finite_loss(&value_supervision_loss(&restored, &rows, limits)?)?
    );
    assert!(value_supervision_loss(&model, &[], limits).is_err());
    let other_gamma = ValueSupervision::monte_carlo(&sample, 1, 0.5)?;
    assert!(value_supervision_loss(&model, &[rows[0], other_gamma], limits).is_err());
    Ok(())
}

fn gradients(model: &RecurrentLightweightValue, loss: Tensor<1>) -> Result<Vec<f32>> {
    struct Collect(GradientsParams, Vec<f32>);
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            if let Some(gradient) = self.0.get::<D>(p.id) {
                self.1.extend(to_vec(gradient).unwrap());
            } else {
                self.1.extend(vec![0.; p.val().dims().iter().product()]);
            }
        }
    }
    let mut collect = Collect(GradientsParams::from_grads(loss.backward(), model), vec![]);
    model.visit(&mut collect);
    Ok(collect.1)
}

#[test]
fn recurrent_value_fitting_is_causal_masked_and_detached_at_the_learning_boundary() -> Result<()> {
    let sample = sequence(1)?;
    assert_eq!(sample.loss_mask, [false, true]);
    let row = ValueSupervision::monte_carlo(&sample, 1, 0.9)?;
    assert!(ValueSupervision::monte_carlo(&sample, 0, 0.9).is_err());
    let limits = PackLimits::default();
    let mut model = RecurrentLightweightValue::new(
        doors::model_config(),
        &mut ChaCha8Rng::seed_from_u64(53),
        &Device::flex().autodiff(),
    )?;
    let loss = recurrent_value_supervision_loss(&model, &[row], limits)?;
    let before = finite_loss(&loss)?;
    let actual = gradients(&model, loss)?;
    let history = sample.exact_history_inputs()?.take(2).collect::<Vec<_>>();
    let reference = |learn_start| -> Result<Tensor<1>> {
        squared_loss(
            model.evaluate_selected(
                &[StructuredHistoryInput {
                    inputs: &history,
                    learn_start,
                }],
                &[(0, 1)],
                limits,
            )?,
            &[1.],
        )
    };
    assert_eq!(actual, gradients(&model, reference(1)?)?);
    assert!(
        actual
            .iter()
            .zip(gradients(&model, reference(0)?)?)
            .any(|(a, b)| (a - b).abs() > 1e-7)
    );
    let mut future = sample.clone();
    future.observations[2] = doors::frame(doors::Public {
        reading: Some(false),
    });
    future.steps[1].action = Action::Conditional(Arc::new(doors::guess(false)));
    let future_row = ValueSupervision::monte_carlo(&future, 1, 0.9)?;
    assert_eq!(
        before,
        finite_loss(&recurrent_value_supervision_loss(
            &model,
            &[future_row],
            limits
        )?)?
    );
    let mut changed_past = sample.clone();
    changed_past.steps[0].reward = 0.7.into();
    let changed_row = ValueSupervision::monte_carlo(&changed_past, 1, 0.9)?;
    assert_ne!(
        before,
        finite_loss(&recurrent_value_supervision_loss(
            &model,
            &[changed_row],
            limits
        )?)?
    );
    let mut adam = OptimizerConfig::default().build()?;
    for _ in 0..24 {
        let loss = recurrent_value_supervision_loss(&model, &[row], limits)?;
        optimize_validated(&mut model, &mut adam, 0.003, loss)?;
    }
    let after = finite_loss(&recurrent_value_supervision_loss(&model, &[row], limits)?)?;
    eprintln!("recurrent factual MSE {before:.9} -> {after:.9}");
    assert!(after < before * 0.5);
    let mut missing_history = sample;
    missing_history.origin = SequenceOrigin::Stateless;
    let row = ValueSupervision::monte_carlo(&missing_history, 1, 0.9)?;
    assert!(recurrent_value_supervision_loss(&model, &[row], limits).is_err());
    Ok(())
}

#[test]
fn ragged_recurrent_rows_match_independent_selected_losses_and_gradients() -> Result<()> {
    let early = sequence(0)?;
    let late = sequence(1)?;
    let a = ValueSupervision::monte_carlo(&early, 0, 0.9)?;
    let b = ValueSupervision::monte_carlo(&late, 1, 0.9)?;
    assert_eq!(a.target(), 0.81);
    assert_eq!(b.target(), 1.);
    let model = RecurrentLightweightValue::new(
        doors::model_config(),
        &mut ChaCha8Rng::seed_from_u64(59),
        &Device::flex().autodiff(),
    )?;
    let limits = PackLimits::default();
    let independent = (recurrent_value_supervision_loss(&model, &[a], limits)?
        + recurrent_value_supervision_loss(&model, &[b], limits)?)
        / 2.;
    let expected_loss = finite_loss(&independent)?;
    let expected_gradients = gradients(&model, independent)?;
    for rows in [[a, b], [b, a]] {
        let combined = recurrent_value_supervision_loss(&model, &rows, limits)?;
        assert!((finite_loss(&combined)? - expected_loss).abs() < 1e-6);
        for (actual, expected) in gradients(&model, combined)?.iter().zip(&expected_gradients) {
            assert!((actual - expected).abs() <= 2e-6 * (1. + expected.abs()));
        }
    }
    Ok(())
}

#[test]
fn stateless_mid_episode_value_row_uses_factual_frame_without_claiming_recurrent_history()
-> Result<()> {
    let mut sample = sequence(1)?;
    sample.observations.remove(0);
    sample.steps.remove(0);
    sample.loss_mask.remove(0);
    sample.trajectory_start = 1;
    sample.learn_start = 0;
    sample.burn_start = 0;
    sample.history_start = 1;
    sample.origin = SequenceOrigin::Stateless;
    let row = ValueSupervision::monte_carlo(&sample, 0, 0.9)?;
    assert_eq!(row.target(), 1.);
    let model = LightweightValue::new(
        doors::model_config().model,
        &mut ChaCha8Rng::seed_from_u64(61),
        &Device::flex().autodiff(),
    )?;
    assert!(
        finite_loss(&value_supervision_loss(
            &model,
            &[row],
            PackLimits::default()
        )?)?
        .is_finite()
    );
    let recurrent = RecurrentLightweightValue::new(
        doors::model_config(),
        &mut ChaCha8Rng::seed_from_u64(61),
        &Device::flex().autodiff(),
    )?;
    assert!(recurrent_value_supervision_loss(&recurrent, &[row], PackLimits::default()).is_err());
    Ok(())
}
