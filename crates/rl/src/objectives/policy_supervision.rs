//! Source-independent fitting of complete discrete action distributions.
//! Support identities and learning-row selection are checked by the caller's
//! data adapter. Every selected row has weight 1/B, regardless of its width.
use crate::{Result, core::ensure, nn::to_vec};
use burn::tensor::Tensor;

/// Mean cross-entropy over ragged, complete legal-action rows. Targets are
/// detached host constants; gradients flow only through log probabilities.
/// No renormalization, probability flooring, or algorithm-specific trust region.
pub fn distribution_loss(likelihoods: Vec<Tensor<1>>, targets: &[Vec<f32>]) -> Result<Tensor<1>> {
    ensure(
        !likelihoods.is_empty() && likelihoods.len() == targets.len(),
        "policy supervision likelihood row count differs",
    )?;
    let mut losses = Vec::with_capacity(likelihoods.len());
    for (likelihood, target) in likelihoods.into_iter().zip(targets) {
        ensure(
            !target.is_empty() && likelihood.dims()[0] == target.len(),
            "policy supervision likelihood width differs",
        )?;
        ensure(
            target.iter().all(|p| p.is_finite() && *p >= 0.)
                && (target.iter().map(|&p| f64::from(p)).sum::<f64>() - 1.).abs() <= 1e-6,
            "policy supervision target must be a normalized distribution",
        )?;
        ensure(
            to_vec(likelihood.clone())?.iter().all(|p| p.is_finite()),
            "nonfinite policy supervision log probability",
        )?;
        let device = likelihood.device();
        // Keep Candidate MPO's original arithmetic and reduction order.
        losses.push(-(likelihood * Tensor::from_data(target.as_slice(), &device)).sum());
    }
    Ok(Tensor::cat(losses, 0).mean())
}
