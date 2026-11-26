//! Scalar-value evaluation at selected factual rows; callers own Adam and updates.
use super::{
    LightweightValue, PackLimits, RecurrentLightweightValue, StructuredHistoryInput,
    StructuredSample,
};
use crate::{
    Result, core::ensure, data::value_supervision::ValueSupervision,
    objectives::value_supervision::squared_loss,
};
use burn::tensor::Tensor;

fn targets(rows: &[ValueSupervision<'_>]) -> Result<Vec<f32>> {
    ensure(!rows.is_empty(), "empty value supervision batch")?;
    ensure(
        rows.iter().all(|row| row.gamma() == rows[0].gamma()),
        "value supervision batch mixes return discounts",
    )?;
    Ok(rows.iter().map(ValueSupervision::target).collect())
}

/// Mean squared error against complete physical Monte Carlo returns at the
/// selected public frames. Burn-in, padding and nonlearning rows are excluded.
pub fn value_supervision_loss(
    value: &LightweightValue,
    rows: &[ValueSupervision<'_>],
    limits: PackLimits,
) -> Result<Tensor<1>> {
    let targets = targets(rows)?;
    LightweightValue::parameter_count(value.config())?;
    let samples = rows
        .iter()
        .map(|row| {
            StructuredSample::from_observation_with_validated_schema(
                row.observation(),
                &value.config().input,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    squared_loss(value.evaluate(&samples, limits)?, &targets)
}

/// Reconstruct only the exact causal prefix through each selected decision.
/// The selected action/reward and subsequent observations are target data, never
/// input to this prediction. Detach at the sample's existing learning boundary.
pub fn recurrent_value_supervision_loss(
    value: &RecurrentLightweightValue,
    rows: &[ValueSupervision<'_>],
    limits: PackLimits,
) -> Result<Tensor<1>> {
    let targets = targets(rows)?;
    let histories = rows
        .iter()
        .map(|row| {
            Ok(row
                .sample()
                .exact_history_inputs()?
                .take(row.step() + 1)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    let inputs = histories
        .iter()
        .zip(rows)
        .map(|(inputs, row)| StructuredHistoryInput {
            inputs,
            learn_start: row.sample().learn_start,
        })
        .collect::<Vec<_>>();
    let tokens = rows
        .iter()
        .enumerate()
        .map(|(lane, row)| (lane, row.step()))
        .collect::<Vec<_>>();
    squared_loss(value.evaluate_selected(&inputs, &tokens, limits)?, &targets)
}
