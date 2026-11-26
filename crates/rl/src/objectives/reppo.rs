//! Fixed soft TD-lambda action-value targets for complete time-major fragments.
use crate::{
    Result,
    core::ensure,
    data::returns::{self, Discounts},
};
use burn::tensor::Tensor;

/// Exact discrete maximum-entropy objective for each observation. Frozen Q
/// values are detached; zero-probability actions make no contribution.
pub fn categorical_objective(
    probabilities: Tensor<2>,
    log_probs: Tensor<2>,
    q: Tensor<2>,
    alpha: f32,
) -> Result<Tensor<1>> {
    let [n, k] = probabilities.dims();
    ensure(
        n > 0
            && k > 0
            && log_probs.dims() == [n, k]
            && q.dims() == [n, k]
            && alpha.is_finite()
            && alpha > 0.,
        "invalid discrete REPPO objective",
    )?;
    let zero = probabilities.clone().equal_elem(0.);
    let q = q
        .without_autodiff()
        .to_device(&probabilities.device())
        .mask_fill(zero.clone(), 0.);
    let log_probs = log_probs.mask_fill(zero, 0.);
    Ok((probabilities * (log_probs * alpha - q))
        .sum_dim(1)
        .squeeze_dim(1))
}

/// Latent self-prediction MSE averaged over all rows/features. A zero
/// continuation excludes episode endings; target-side gradients stop.
pub fn self_predictive_loss(
    prediction: Tensor<2>,
    target: Tensor<2>,
    continuation: Tensor<1>,
) -> Result<Tensor<1>> {
    let [n, k] = prediction.dims();
    ensure(
        n > 0 && k > 0 && target.dims() == [n, k] && continuation.dims() == [n],
        "invalid REPPO auxiliary shapes",
    )?;
    let device = prediction.device();
    let error = prediction - target.without_autodiff().to_device(&device);
    Ok((error.powf_scalar(2.).mean_dim(1).squeeze_dim(1)
        * continuation.without_autodiff().to_device(&device))
    .mean())
}

/// Minimize the pathwise objective below the per-observation KL bound and a
/// positive KL penalty at or above it. Gate decisions do not carry gradients.
pub fn kl_gated_loss(
    ordinary: Tensor<1>,
    kl: Tensor<1>,
    beta: f32,
    limit: f32,
) -> Result<Tensor<1>> {
    ensure(
        ordinary.dims() == kl.dims()
            && ordinary.dims()[0] > 0
            && beta.is_finite()
            && beta > 0.
            && limit.is_finite()
            && limit > 0.,
        "invalid REPPO KL gate",
    )?;
    let below = kl.clone().lower_elem(limit);
    Ok((kl * beta).mask_where(below, ordinary).mean())
}

/// Gradient descent raises a positive multiplier on negative slack and lowers
/// it on positive slack. Use H-target for alpha, and KL_limit-KL for beta.
pub fn constraint_dual_loss(log_multiplier: Tensor<1>, slack: Tensor<1>) -> Result<Tensor<1>> {
    ensure(
        log_multiplier.dims() == [1] && slack.dims()[0] > 0,
        "invalid REPPO dual shapes",
    )?;
    Ok(log_multiplier.exp() * slack.without_autodiff().mean())
}

/// Next-state values/entropies come from a single frozen behavior/critic snapshot.
/// Entropy entries already include alpha, but do not include gamma.
pub struct SoftTrajectory<'a> {
    pub rewards: &'a [f32],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
    pub bootstrap: &'a [f32],
    pub next_entropy: &'a [f32],
    pub lanes: usize,
}

/// Compute targets once per rollout, before shuffling or changing the critic.
/// Truncations and fragment tails bootstrap; true terminals have no next entropy.
pub fn soft_td_lambda(t: SoftTrajectory<'_>, gamma: f32, lambda: f32) -> Result<Vec<f32>> {
    soft_td_lambda_with(t, Discounts::Constant(gamma), lambda)
}

/// Apply per-decision continuation discounts to already discounted rewards.
pub(crate) fn soft_td_lambda_with(
    t: SoftTrajectory<'_>,
    discounts: Discounts<'_>,
    lambda: f32,
) -> Result<Vec<f32>> {
    let n = t.rewards.len();
    returns::validate(t.rewards, t.bootstrap, t.terminated, t.truncated, discounts)?;
    ensure(
        n > 0
            && t.lanes > 0
            && n.is_multiple_of(t.lanes)
            && t.next_entropy.len() == n
            && t.next_entropy.iter().all(|x| x.is_finite())
            && lambda.is_finite()
            && (0.0..=1.0).contains(&lambda),
        "invalid soft TD-lambda trajectory or settings",
    )?;
    let mut targets = vec![0.; n];
    for lane in 0..t.lanes {
        let mut following = None;
        for i in (lane..n).step_by(t.lanes).rev() {
            let value = if t.terminated[i] {
                f64::from(t.rewards[i])
            } else {
                let bootstrap = f64::from(t.bootstrap[i]);
                let continuation = if t.truncated[i] { None } else { following };
                let value = continuation.map_or(bootstrap, |next| {
                    (1. - f64::from(lambda)) * bootstrap + f64::from(lambda) * next
                });
                f64::from(t.rewards[i])
                    + f64::from(discounts.at(i)) * (f64::from(t.next_entropy[i]) + value)
            };
            ensure(
                value.is_finite() && (value as f32).is_finite(),
                "soft TD-lambda target overflow",
            )?;
            targets[i] = value as f32;
            following = Some(value);
        }
    }
    Ok(targets)
}
