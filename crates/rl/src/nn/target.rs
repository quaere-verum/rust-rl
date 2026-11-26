use crate::{Result, core::ensure};

/// Synchronize and independently copy module parameters and running-statistic buffers.
/// Burn's `clone().fork()` alone shares RunningState storage.
pub fn frozen_copy<M: Module>(model: &M) -> M {
    model.valid().fork(&super::device()).freeze()
}
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::Tensor,
};

/// target <- (1-tau)*target + tau*online. Parameters are matched in module traversal order,
/// with shape/count validation; both modules must have the same Rust type and structure.
pub fn polyak_update<M: Module>(target: M, online: &M, tau: f32) -> Result<M> {
    let target = frozen_copy(&target);
    let online = online.valid();
    ensure(
        tau.is_finite() && (0.0..=1.0).contains(&tau),
        "tau must be in [0,1]",
    )?;
    #[derive(Default)]
    struct Collect {
        values: Vec<(Vec<usize>, Tensor<1>)>,
    }
    impl ModuleVisitor for Collect {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let t = p.val().without_autodiff();
            self.values.push((
                t.dims().to_vec(),
                t.clone().reshape([t.dims().iter().product::<usize>()]),
            ));
        }
    }
    let mut source = Collect::default();
    online.visit(&mut source);
    let mut dest = Collect::default();
    target.visit(&mut dest);
    ensure(
        source.values.len() == dest.values.len()
            && source
                .values
                .iter()
                .zip(&dest.values)
                .all(|(a, b)| a.0 == b.0),
        "target module structures differ",
    )?;
    struct Blend {
        values: std::vec::IntoIter<(Vec<usize>, Tensor<1>)>,
        tau: f32,
    }
    impl ModuleMapper for Blend {
        fn map_float<const D: usize>(&mut self, p: Param<Tensor<D>>) -> Param<Tensor<D>> {
            let (_, new) = self.values.next().expect("validated parameter count");
            p.map(|old| {
                let shape = old.dims();
                old.without_autodiff() * (1. - self.tau) + new.reshape(shape) * self.tau
            })
        }
    }
    Ok(target
        .map(&mut Blend {
            values: source.values.into_iter(),
            tau,
        })
        .freeze())
}
