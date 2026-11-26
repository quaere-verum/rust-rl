//! Structured models accept an explicit device, so restoration preserves the
//! destination's execution mode rather than implicitly turning inference on.
use crate::{Result, core::checkpoint::LearnerRecord, nn::checkpoint::RecordBurnExt};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::{Device, Tensor},
};
use std::collections::VecDeque;

#[derive(Default)]
struct Execution(VecDeque<(Device, bool)>);
impl ModuleVisitor for Execution {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let value = param.val();
        self.0.push_back((value.device(), value.is_require_grad()));
    }
}
impl ModuleMapper for Execution {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        // Same typed module, validated by RecordBurnExt's shape/signature checks.
        let (device, trainable) = self.0.pop_front().expect("same module traversal");
        // Tensor::to_device deliberately preserves the source autodiff mode.
        // Recreate the restored leaf on the destination's explicit execution
        // device (including its gradient-checkpointing strategy), then restore
        // trainability. This host copy is checkpoint-only, never an inference path.
        param.map(|value| Tensor::from_data(value.into_data(), &device).set_require_grad(trainable))
    }
}
pub(crate) fn restore<M: Module>(record: &LearnerRecord, name: &str, template: &M) -> Result<M> {
    let mut execution = Execution::default();
    template.visit(&mut execution);
    // Loading on plain Flex avoids even temporary trainable autodiff roots for
    // inference templates. Activate only the destination's declared parameters.
    let model = record.load_model_inference(name, &template.valid())?;
    let restored = model.map(&mut execution);
    debug_assert!(execution.0.is_empty());
    Ok(restored)
}
