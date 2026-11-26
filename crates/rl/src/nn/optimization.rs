//! Opt-in pre-Adam validation without changing the optimizer or clipping rule.
use crate::{
    Error, Result,
    core::ensure,
    nn::{finite_loss, to_vec},
};
use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::{GradientsParams, ModuleOptimizer},
    tensor::Tensor,
};
use std::collections::HashSet;

#[derive(Default)]
struct Ids {
    values: HashSet<ParamId>,
    duplicate: bool,
}
impl ModuleVisitor for Ids {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.duplicate |= !self.values.insert(param.id);
    }
}

pub(crate) fn validate_disjoint<P: Module, V: Module>(policy: &P, value: &V) -> Result<()> {
    let mut p = Ids::default();
    let mut v = Ids::default();
    policy.visit(&mut p);
    value.visit(&mut v);
    ensure(
        !p.duplicate && !v.duplicate && p.values.is_disjoint(&v.values),
        "optimization requires untied, independent parameter groups",
    )
}

/// Return loss and the unclipped f64 gradient norm. Unsafe gradients reject before
/// either the model or optimizer changes; ordinary Adam arithmetic is unchanged.
pub fn optimize_validated<M: Module>(
    model: &mut M,
    opt: &mut ModuleOptimizer,
    lr: f64,
    loss: Tensor<1>,
) -> Result<(f64, f64)> {
    optimize_with_global_clip(model, opt, lr, loss, None)
}

/// Explicit global clipping for algorithms whose reference optimizers clip the
/// concatenated parameter gradient. The optimizer itself must use no clipping.
pub(crate) fn optimize_with_global_clip<M: Module>(
    model: &mut M,
    opt: &mut ModuleOptimizer,
    lr: f64,
    loss: Tensor<1>,
    max_norm: Option<f32>,
) -> Result<(f64, f64)> {
    let value = finite_loss(&loss)?;
    crate::core::ensure(loss.is_tracked(), "optimization loss has no autodiff graph")?;
    let mut grads = GradientsParams::from_grads(loss.backward(), model);
    let mut visitor = NormVisitor {
        grads: Some(&grads),
        sum: 0.,
        error: None,
        ids: Default::default(),
        tensors: 0,
    };
    model.visit(&mut visitor);
    crate::core::ensure(
        visitor.tensors > 0,
        "optimization loss has no owned parameter gradients",
    )?;
    let norm = visitor.finish()?;
    // Adam and Burn norm clipping square/reduce in f32. The margin also
    // covers f32 reduction rounding; this does not modify valid gradients.
    if norm * norm > f32::MAX as f64 * 0.5 {
        return Err(Error::Numerical(
            "gradients exceed safe f32 Adam/norm-clipping arithmetic".into(),
        ));
    }
    if let Some(max_norm) = max_norm {
        let scale = f64::from(max_norm) / (norm + 1e-6);
        if scale < 1. {
            struct Scale<'a> {
                grads: &'a mut GradientsParams,
                seen: std::collections::HashSet<ParamId>,
                factor: f32,
            }
            impl ModuleVisitor for Scale<'_> {
                fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                    if self.seen.insert(param.id)
                        && let Some(gradient) = self.grads.remove::<D>(param.id)
                    {
                        self.grads.register(param.id, gradient * self.factor);
                    }
                }
            }
            model.visit(&mut Scale {
                grads: &mut grads,
                seen: Default::default(),
                factor: scale as f32,
            });
        }
    }
    *model = opt.step(lr, model.clone(), grads);
    Ok((value, norm))
}

/// Loss-only form for callers that do not report gradient diagnostics.
pub(crate) fn optimize<M: Module>(
    model: &mut M,
    opt: &mut ModuleOptimizer,
    lr: f64,
    loss: Tensor<1>,
) -> Result<f64> {
    Ok(optimize_validated(model, opt, lr, loss)?.0)
}

/// Stable host accumulation, deduplicating tied parameter IDs.
pub fn parameter_norm<M: Module>(model: &M) -> Result<f64> {
    let mut visitor = NormVisitor {
        grads: None,
        sum: 0.,
        error: None,
        ids: Default::default(),
        tensors: 0,
    };
    model.visit(&mut visitor);
    visitor.finish()
}
struct NormVisitor<'a> {
    grads: Option<&'a GradientsParams>,
    sum: f64,
    error: Option<Error>,
    ids: std::collections::HashSet<ParamId>,
    tensors: usize,
}
impl NormVisitor<'_> {
    fn finish(self) -> Result<f64> {
        if let Some(error) = self.error {
            Err(error)
        } else {
            Ok(self.sum.sqrt())
        }
    }
}
impl ModuleVisitor for NormVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if !self.ids.insert(param.id) {
            return;
        }
        let tensor = if let Some(grads) = self.grads {
            grads.get::<D>(param.id)
        } else {
            Some(param.val().without_autodiff())
        };
        if let Some(tensor) = tensor {
            self.tensors += 1;
            match to_vec(tensor) {
                Ok(values) => {
                    for x in values {
                        if !x.is_finite() {
                            self.error =
                                Some(Error::Numerical("nonfinite parameter or gradient".into()));
                            return;
                        }
                        self.sum += (x as f64).powi(2);
                    }
                }
                Err(error) => self.error = Some(error),
            }
        }
    }
}

/// Require at least one trainable parameter; partially frozen custom networks are valid.
pub(crate) fn validate_trainable<M: Module>(model: &M) -> Result<()> {
    struct Trainable(bool);
    impl ModuleVisitor for Trainable {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            self.0 |= p.val().is_require_grad();
        }
    }
    let mut visitor = Trainable(false);
    model.visit(&mut visitor);
    crate::core::ensure(visitor.0, "model has no trainable parameters")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::common::OptimizerConfig,
        core::checkpoint::LearnerRecord,
        nn::{
            checkpoint::RecordBurnExt,
            matrix,
            network::{Mlp, Network},
        },
    };
    use burn::{module::Param, optim::ModuleOptimizer};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[derive(burn::module::Module, Debug)]
    struct Tied {
        first: burn::nn::Linear,
        second: burn::nn::Linear,
    }

    #[test]
    fn rejects_tied_parameter_ids_within_a_module() {
        let device = crate::nn::device();
        let first = burn::nn::LinearConfig::new(1, 1).init(&device);
        let tied = Tied {
            second: first.clone(),
            first,
        };
        let independent = burn::nn::LinearConfig::new(1, 1).init(&device);
        assert!(validate_disjoint(&tied, &independent).is_err());
        assert!(validate_disjoint(&independent, &tied).is_err());
    }
    fn record(model: &Mlp, optimizer: &ModuleOptimizer) -> serde_json::Value {
        let mut record = LearnerRecord::new("guard_fixture", ()).unwrap();
        record.model("model", model).unwrap();
        record.optimizer("optimizer", optimizer).unwrap();
        serde_json::to_value(record).unwrap()
    }
    #[test]
    fn finite_loss_with_huge_finite_gradient_is_rejected_before_adam_mutation() {
        let mut model = Mlp::new(1, &[], 1, &mut ChaCha8Rng::seed_from_u64(671)).unwrap();
        model.layers[0].weight = Param::from_tensor(matrix(vec![1e-30], 1, 1));
        model.layers[0].bias = None;
        let mut optimizer = OptimizerConfig {
            max_grad_norm: None,
            ..Default::default()
        }
        .build()
        .unwrap();
        let before = record(&model, &optimizer);
        let loss = model.forward(matrix(vec![1e30], 1, 1)).sum();
        assert!(loss.clone().into_scalar::<f32>().is_finite());
        assert!(optimize_validated(&mut model, &mut optimizer, 0.001, loss).is_err());
        assert!(
            record(&model, &optimizer) == before,
            "guard mutated model or Adam before rejecting finite overflow-risk gradient"
        );
    }
    #[test]
    fn validated_optimizer_preserves_ordinary_adam_step_and_reports_f64_norm() {
        let mut model = Mlp::new(1, &[], 1, &mut ChaCha8Rng::seed_from_u64(672)).unwrap();
        model.layers[0].weight = Param::from_tensor(matrix(vec![0.25], 1, 1));
        model.layers[0].bias = None;
        let mut reference = model.clone();
        let mut optimizer = OptimizerConfig {
            max_grad_norm: None,
            ..Default::default()
        }
        .build()
        .unwrap();
        let mut reference_optimizer = OptimizerConfig {
            max_grad_norm: None,
            ..Default::default()
        }
        .build()
        .unwrap();
        let loss = model.forward(matrix(vec![3.], 1, 1)).sum();
        let (number, norm) = optimize_validated(&mut model, &mut optimizer, 0.001, loss).unwrap();
        assert_eq!(number, 0.75);
        assert_eq!(norm, 3.);
        let loss = reference.forward(matrix(vec![3.], 1, 1)).sum();
        crate::algorithms::common::optimize(&mut reference, &mut reference_optimizer, 0.001, loss)
            .unwrap();
        assert_eq!(
            crate::nn::to_vec(model.layers[0].weight.val()).unwrap(),
            crate::nn::to_vec(reference.layers[0].weight.val()).unwrap()
        );
        assert_eq!(
            parameter_norm(&model).unwrap(),
            crate::nn::to_vec(model.layers[0].weight.val()).unwrap()[0].abs() as f64
        );
    }

    #[test]
    fn detached_or_unowned_losses_reject_without_adam_mutation() {
        let mut model = Mlp::new(1, &[], 1, &mut ChaCha8Rng::seed_from_u64(678)).unwrap();
        let mut optimizer = OptimizerConfig::default().build().unwrap();
        let before = record(&model, &optimizer);
        for loss in [
            crate::nn::vector(&[1.]),
            crate::nn::vector(&[1.]).require_grad().powf_scalar(2.),
        ] {
            assert!(optimize_validated(&mut model, &mut optimizer, 0.001, loss).is_err());
            assert_eq!(record(&model, &optimizer), before);
        }
    }
}
