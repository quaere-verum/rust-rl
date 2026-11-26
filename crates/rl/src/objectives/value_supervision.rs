//! Scalar regression mathematics, independent of target construction or updates.
use crate::{
    Result,
    core::ensure,
    nn::{finite_loss, to_vec},
};
use burn::tensor::Tensor;

/// Mean squared error over selected learning rows, each weighted 1/B. Targets
/// are detached host constants. No one-half factor, clipping or bootstrapping.
pub fn squared_loss(predictions: Tensor<1>, targets: &[f32]) -> Result<Tensor<1>> {
    ensure(
        !targets.is_empty() && predictions.dims()[0] == targets.len(),
        "value supervision batch shape differs",
    )?;
    ensure(
        targets.iter().all(|x| x.is_finite())
            && to_vec(predictions.clone())?.iter().all(|x| x.is_finite()),
        "nonfinite value supervision input",
    )?;
    let device = predictions.device();
    let loss = (predictions - Tensor::from_data(targets, &device))
        .powf_scalar(2.)
        .mean();
    finite_loss(&loss)?;
    Ok(loss)
}
