#![cfg(feature = "neural")]

use burn::{
    module::{Module, RunningState},
    tensor::Tensor,
};
use rl::{
    Result,
    algorithms::ddqn::{Ddqn, DdqnConfig},
    core::{checkpoint::Checkpoint, env::ObservationBatch},
    nn::{checkpoint::RecordBurnExt, critics::scalar::DiscreteActionValue, matrix},
    train::checkpoint::LearnerRecord,
};
#[derive(Module, Debug)]
struct BufferedQ {
    running: RunningState<Tensor<1>>,
}
impl DiscreteActionValue for BufferedQ {
    fn action_count(&self) -> usize {
        1
    }
    fn action_values(&self, o: ObservationBatch<'_>) -> Result<Tensor<2>> {
        Ok(self
            .running
            .value_sync()
            .reshape([1, 1])
            .repeat_dim(0, o.len()))
    }
}
fn model() -> BufferedQ {
    BufferedQ {
        running: RunningState::new(rl::nn::vector(&[0.])),
    }
}
#[test]
fn target_running_statistics_are_independent_and_soft_updated() {
    let mut learner = Ddqn::new(
        model(),
        DdqnConfig {
            target_tau: 0.,
            ..Default::default()
        },
    )
    .unwrap();
    let online = model();
    online.running.update(rl::nn::vector(&[4.]));
    let mut record = learner.checkpoint().unwrap();
    record.model("online", &online).unwrap();
    learner.restore(&record).unwrap();
    assert_eq!(
        learner.policy().running.value_sync().into_scalar::<f32>(),
        4.
    );
    assert_eq!(
        learner.target().running.value_sync().into_scalar::<f32>(),
        0.
    );
    let target = rl::nn::target::polyak_update(learner.target(), &learner.policy(), 0.25).unwrap();
    assert_eq!(target.running.value_sync().into_scalar::<f32>(), 1.);
    assert_eq!(
        learner.policy().running.value_sync().into_scalar::<f32>(),
        4.
    );
    assert_eq!(
        learner.target().running.value_sync().into_scalar::<f32>(),
        0.
    );
}
#[test]
fn checkpoint_flushes_pending_running_statistics() {
    let source = model();
    source.running.update(rl::nn::vector(&[7.]));
    let mut record = LearnerRecord::new("test", ()).unwrap();
    record.model("model", &source).unwrap();
    let restored = record.load_model("model", &model()).unwrap();
    assert_eq!(restored.running.value_sync().into_scalar::<f32>(), 7.);
}
#[test]
fn batchnorm_online_restore_retains_training_behavior() {
    let original = burn::nn::BatchNormConfig::new(1)
        .with_momentum(1.)
        .init(&rl::nn::device());
    let _ = original.forward(matrix(vec![2., 4.], 2, 1));
    let mut r = LearnerRecord::new("test", ()).unwrap();
    r.model("bn", &original).unwrap();
    let restored = r
        .load_model(
            "bn",
            &burn::nn::BatchNormConfig::new(1)
                .with_momentum(1.)
                .init(&rl::nn::device()),
        )
        .unwrap();
    assert_eq!(restored.running_mean.value_sync().into_scalar::<f32>(), 3.);
    let _ = restored.forward(matrix(vec![4., 6.], 2, 1));
    assert_eq!(restored.running_mean.value_sync().into_scalar::<f32>(), 5.);
}
