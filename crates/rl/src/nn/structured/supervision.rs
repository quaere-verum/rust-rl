//! Thin evaluation adapters; callers retain model/optimizer ownership.
use super::{
    CandidatePolicy, PackLimits, RecurrentLightweightPolicy, StructuredHistoryInput,
    StructuredSample, candidate_log_probabilities_checked,
};
use crate::{
    Result, core::ensure, data::policy_supervision::PolicySupervision,
    objectives::policy_supervision::distribution_loss,
};
use burn::tensor::Tensor;

fn targets(rows: &[PolicySupervision<'_>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|r| r.probabilities().iter().map(|&q| q as f32).collect())
        .collect()
}

/// Mean complete-action cross-entropy using each selected factual frame.
pub fn candidate_supervision_loss<P: CandidatePolicy>(
    policy: &P,
    rows: &[PolicySupervision<'_>],
    limits: PackLimits,
) -> Result<Tensor<1>> {
    ensure(!rows.is_empty(), "empty policy supervision batch")?;
    P::validate_configuration(policy.configuration())?;
    let samples = rows
        .iter()
        .map(|row| {
            StructuredSample::from_observation_with_validated_schema(
                row.observation(),
                policy.input_schema(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let keys = rows
        .iter()
        .map(|r| r.actions().to_vec())
        .collect::<Vec<_>>();
    distribution_loss(
        candidate_log_probabilities_checked(policy, &samples, &keys, limits)?,
        &targets(rows),
    )
}

/// Reconstruct each exact causal prefix; detach at its existing learning
/// boundary. Only the selected learning row contributes loss. Neither the
/// selected action/reward nor any future observation is passed to the policy.
pub fn recurrent_supervision_loss(
    policy: &RecurrentLightweightPolicy,
    rows: &[PolicySupervision<'_>],
    limits: PackLimits,
) -> Result<Tensor<1>> {
    ensure(!rows.is_empty(), "empty recurrent supervision batch")?;
    let histories = rows
        .iter()
        .map(|r| {
            Ok(r.sample()
                .exact_history_inputs()?
                .take(r.step() + 1)
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
    let evaluations = policy.evaluate_histories(&inputs, limits)?;
    let likelihoods = evaluations
        .iter()
        .zip(rows)
        .map(|(evaluation, row)| evaluation.log_prob_many(row.step(), row.actions()))
        .collect::<Result<Vec<_>>>()?;
    distribution_loss(likelihoods, &targets(rows))
}
