use super::returns::Discounts;
use crate::Result;

/// Scalar-lane GAE. A fragment boundary is implicit at the end of the slices.
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    next_values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    gamma: f32,
    lambda: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    gae_with(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        Discounts::Constant(gamma),
        lambda,
    )
}

/// Lambda decays per decision. The continuation discount decays per physical tick.
pub(crate) fn gae_with(
    rewards: &[f32],
    values: &[f32],
    next_values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    discounts: Discounts<'_>,
    lambda: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    super::returns::validate(rewards, next_values, terminated, truncated, discounts)?;
    crate::core::ensure(
        values.len() == rewards.len() && values.iter().all(|x| x.is_finite()),
        "invalid value estimates",
    )?;
    crate::core::ensure(
        lambda.is_finite() && (0.0..=1.0).contains(&lambda),
        "lambda must be in [0,1]",
    )?;
    let mut advantages = vec![0.0; rewards.len()];
    let mut tail = 0.0;
    for t in (0..rewards.len()).rev() {
        let gamma = discounts.at(t);
        let next = if terminated[t] { 0.0 } else { next_values[t] };
        let continuation = if terminated[t] || truncated[t] {
            0.0
        } else {
            tail
        };
        tail = rewards[t] + gamma * next - values[t] + gamma * lambda * continuation;
        advantages[t] = tail;
    }
    let targets = advantages.iter().zip(values).map(|(a, v)| a + v).collect();
    Ok((advantages, targets))
}
