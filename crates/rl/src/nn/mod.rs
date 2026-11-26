//! Burn modules and reusable policy/value interfaces.
pub mod canonical;
pub mod critics;
pub mod distributions;
pub mod exploration;
pub mod gaussian_policy;
pub mod network;
pub mod optimization;
pub mod policy;
pub mod reconstruction;
pub mod recurrent_actor;
pub mod structured;
pub mod target;
pub mod temporal;
pub mod temporal_actor;
pub mod temporal_batch;
pub mod temporal_categorical_actor;
#[cfg(feature = "experimental-acer")]
pub mod temporal_discrete_q;
pub mod temporal_models;
pub mod temporal_pathwise_actor;
pub mod temporal_policy;
pub mod temporal_q;
pub mod temporal_value;
pub(crate) mod temporal_work;

use crate::{Error, Result, core::env::ObservationBatch};
use burn::tensor::{Device, Tensor, TensorData};

pub fn device() -> Device {
    Device::flex().autodiff()
}
pub fn vector(values: &[f32]) -> Tensor<1> {
    Tensor::from_data(values, &device())
}
pub fn matrix(values: Vec<f32>, rows: usize, cols: usize) -> Tensor<2> {
    matrix_on_device(values, rows, cols, &device())
}
pub(crate) fn matrix_on_device(
    values: Vec<f32>,
    rows: usize,
    cols: usize,
    device: &Device,
) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, [rows, cols]), device)
}
pub fn to_vec<const D: usize>(t: Tensor<D>) -> Result<Vec<f32>> {
    t.into_data()
        .try_to_vec::<f32>()
        .map_err(|e| Error::Numerical(format!("tensor conversion: {e:?}")))
}
pub fn observations(obs: ObservationBatch<'_>, width: usize) -> Result<Tensor<2>> {
    observations_on_device(obs, width, &device())
}
pub(crate) fn observations_on_device(
    obs: ObservationBatch<'_>,
    width: usize,
    device: &Device,
) -> Result<Tensor<2>> {
    crate::core::ensure(
        !obs.is_empty()
            && obs.iter().all(|o| {
                o.as_dense().is_some_and(|dense| {
                    dense.values.len() == width && dense.values.iter().all(|v| v.is_finite())
                })
            }),
        "invalid network observations",
    )?;
    let count = obs
        .len()
        .checked_mul(width)
        .ok_or_else(|| Error::Shape("network observation size overflow".into()))?;
    let mut values = Vec::with_capacity(count);
    for observation in obs.iter() {
        values.extend_from_slice(observation.dense_values());
    }
    Ok(matrix_on_device(values, obs.len(), width, device))
}
pub(crate) fn finite_loss(loss: &Tensor<1>) -> Result<f64> {
    let value = loss.clone().into_scalar::<f32>() as f64;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::Numerical("nonfinite loss".into()))
    }
}
pub mod checkpoint;
#[cfg(any(feature = "probes", all(feature = "neural", feature = "diagnostics")))]
pub mod probes;
