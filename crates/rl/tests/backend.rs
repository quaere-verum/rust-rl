#![cfg(feature = "neural")]

use burn::{
    module::{Module, Param},
    optim::{AdamConfig, GradientsParams, OptimizerRecord},
    tensor::{Device, Tensor},
};

#[derive(Module, Debug)]
struct Model {
    weight: Param<Tensor<2>>,
}
fn model() -> Model {
    Model {
        weight: Param::from_tensor(Tensor::from_data([[2.0_f32]], &Device::flex().autodiff())),
    }
}
fn update(m: Model, opt: &mut burn::optim::ModuleOptimizer) -> Model {
    let loss = m.weight.val().powf_scalar(2.0).mean();
    let grads = GradientsParams::from_grads(loss.backward(), &m);
    opt.step(0.01, m, grads)
}
fn value(m: &Model) -> f32 {
    m.weight.val().into_scalar()
}

#[test]
fn optimizer_updates_online_parameters_without_mutating_target() {
    let online = model();
    let target = online.clone().fork(&Device::flex()).freeze();
    let mut opt = AdamConfig::new().init();
    let online = update(online, &mut opt);
    assert!(value(&online) < 2.0);
    assert_eq!(value(&target), 2.0);
}

#[test]
fn frozen_critic_preserves_action_gradient() {
    let critic = model().freeze();
    let action = Tensor::<2>::from_data([[3.0_f32]], &Device::flex().autodiff()).require_grad();
    let loss = action.clone().matmul(critic.weight.val()).sum();
    let grads = loss.backward();
    assert_eq!(action.grad(&grads).unwrap().into_scalar::<f32>(), 2.0_f32);
    assert!(critic.weight.val().grad(&grads).is_none());
}

#[test]
fn model_and_optimizer_restore_the_next_update() {
    let mut opt = AdamConfig::new().init();
    let m = update(model(), &mut opt);
    let model_bytes = m.clone().into_record().into_bytes().unwrap();
    let optim_bytes = opt.to_record().into_bytes().unwrap();
    let restored = model()
        .try_load_record(burn::store::ModuleRecord::from_bytes(model_bytes).unwrap())
        .unwrap();
    let mut restored_opt = AdamConfig::new()
        .init()
        .load_record(OptimizerRecord::from_bytes(optim_bytes).unwrap());
    assert!(
        (value(&update(m, &mut opt)) - value(&update(restored, &mut restored_opt))).abs() < 1e-7
    );
}
