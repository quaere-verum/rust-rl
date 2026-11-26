//! Continuous soft actor-critic objectives. Densities use physical action units.
use crate::{Result, core::ensure, nn::to_vec};
use burn::tensor::Tensor;

fn temperature(alpha: f32) -> Result<()> {
    ensure(alpha.is_finite() && alpha >= 0., "invalid SAC temperature")
}
/// `min(Q1,Q2) - alpha log pi`. Validate both estimates before taking a minimum.
pub fn soft_values(
    first: &[f32],
    second: &[f32],
    log_probs: &[f32],
    alpha: f32,
) -> Result<Vec<f32>> {
    temperature(alpha)?;
    ensure(
        first.len() == second.len()
            && first.len() == log_probs.len()
            && first
                .iter()
                .chain(second)
                .chain(log_probs)
                .all(|x| x.is_finite()),
        "invalid SAC soft value inputs",
    )?;
    let values: Vec<_> = first
        .iter()
        .zip(second)
        .zip(log_probs)
        .map(|((&a, &b), &lp)| a.min(b) - alpha * lp)
        .collect();
    ensure(
        values.iter().all(|x| x.is_finite()),
        "nonfinite SAC soft value",
    )?;
    Ok(values)
}
/// Critics must be frozen by the caller while retaining derivatives through actions.
pub fn actor_loss(
    log_probs: Tensor<1>,
    first: Tensor<1>,
    second: Tensor<1>,
    alpha: f32,
) -> Result<Tensor<1>> {
    temperature(alpha)?;
    ensure(
        log_probs.dims()[0] > 0
            && first.dims() == log_probs.dims()
            && second.dims() == log_probs.dims(),
        "SAC actor batch shape mismatch",
    )?;
    for t in [&log_probs, &first, &second] {
        ensure(
            to_vec(t.clone())?.iter().all(|x| x.is_finite()),
            "nonfinite SAC actor input",
        )?;
    }
    Ok((log_probs * alpha - first.min_pair(second)).mean())
}
/// Common log-temperature surrogate, not the exponential dual's gradient scale.
/// Descent increases alpha when sampled entropy is below target. No policy gradient.
pub fn temperature_loss(
    log_alpha: Tensor<1>,
    log_probs: Tensor<1>,
    target_entropy: f32,
) -> Result<Tensor<1>> {
    ensure(
        log_alpha.dims() == [1] && log_probs.dims()[0] > 0 && target_entropy.is_finite(),
        "invalid SAC temperature inputs",
    )?;
    ensure(
        to_vec(log_probs.clone())?.iter().all(|x| x.is_finite()),
        "nonfinite SAC log density",
    )?;
    Ok(-(log_alpha * (log_probs.detach() + target_entropy).mean()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{to_vec, vector};

    #[test]
    fn soft_values_use_both_critics_and_reject_nonfinite_inputs_before_minimum() {
        assert_eq!(
            soft_values(&[3., -2.], &[1., 4.], &[-2., 0.5], 0.25).unwrap(),
            [1.5, -2.125]
        );
        assert_eq!(
            soft_values(&[3., -2.], &[1., 4.], &[-2., 0.5], 0.).unwrap(),
            [1., -2.]
        );
        assert!(soft_values(&[f32::NAN], &[1.], &[0.], 0.).is_err());
        assert!(soft_values(&[1.], &[f32::NAN], &[0.], 0.).is_err());
        assert!(soft_values(&[1.], &[2.], &[0.], -0.1).is_err());
    }

    #[test]
    fn actor_objective_has_expected_minimum_and_density_gradients() {
        let lp = vector(&[-2., -1.]).require_grad();
        let first = vector(&[3., -2.]).require_grad();
        let second = vector(&[1., 4.]).require_grad();
        let loss = actor_loss(lp.clone(), first.clone(), second.clone(), 0.25).unwrap();
        assert_eq!(to_vec(loss.clone()).unwrap(), [0.125]);
        let grads = loss.backward();
        assert_eq!(to_vec(lp.grad(&grads).unwrap()).unwrap(), [0.125, 0.125]);
        assert_eq!(to_vec(first.grad(&grads).unwrap()).unwrap(), [0., -0.5]);
        assert_eq!(to_vec(second.grad(&grads).unwrap()).unwrap(), [-0.5, 0.]);
    }

    #[test]
    fn log_temperature_surrogate_sign_and_policy_detachment() {
        for (log_density, expected_gradient) in [(2., -1.), (0., 1.)] {
            let log_alpha = vector(&[-2.]).require_grad();
            let log_probs = vector(&[log_density]).require_grad();
            let loss = temperature_loss(log_alpha.clone(), log_probs.clone(), -1.).unwrap();
            let grads = loss.backward();
            assert_eq!(
                to_vec(log_alpha.grad(&grads).unwrap()).unwrap(),
                [expected_gradient]
            );
            assert!(log_probs.grad(&grads).is_none());
        }
    }
}
