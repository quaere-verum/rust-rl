//! Private statically dispatched critic contract and shared model mechanics.
use super::{ContinuousConfig, OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        checkpoint::LearnerRecord,
        ensure,
        env::{Observation, ObservationBatch},
    },
    data::replay::{ReplayBatch, ReplayItem},
    nn::{
        checkpoint::RecordBurnExt,
        critics::{residual::ResidualQ, scalar::ActionValue},
        matrix,
        policy::Deterministic,
        target::frozen_copy,
        to_vec,
    },
};
use burn::{module::ModuleDisplay, optim::ModuleOptimizer, tensor::Tensor};
use serde::{Serialize, de::DeserializeOwned};

pub(crate) struct PreparedReplayBatch<'a> {
    pub items: &'a [ReplayItem],
    observations: Vec<&'a Observation>,
    pub actions: Tensor<2>,
}
impl<'a> PreparedReplayBatch<'a> {
    pub fn new(batch: &'a ReplayBatch, space: &ActionSpace) -> Result<Self> {
        batch.validate()?;
        let mut actions = Vec::with_capacity(batch.items.len() * space.dimensions());
        for item in &batch.items {
            space.check(&item.transition.action, None)?;
            if let Action::Continuous(values) = &item.transition.action {
                actions.extend_from_slice(values);
            }
        }
        Ok(Self {
            items: &batch.items,
            observations: batch
                .items
                .iter()
                .map(|item| &item.transition.observation)
                .collect(),
            actions: matrix(actions, batch.items.len(), space.dimensions()),
        })
    }
    pub fn observations(&self) -> ObservationBatch<'_> {
        ObservationBatch::from_refs(&self.observations)
    }
}
pub(crate) struct TargetActionBatch<'a> {
    pub indices: Vec<usize>,
    pub observations: Vec<&'a Observation>,
    pub actions: Option<Tensor<2>>,
}
impl TargetActionBatch<'_> {
    pub fn observations(&self) -> ObservationBatch<'_> {
        ObservationBatch::from_refs(&self.observations)
    }
    pub fn validate(&self, len: usize, dimensions: usize) -> Result<()> {
        ensure(
            self.indices.len() == self.observations.len()
                && self.indices.iter().all(|&i| i < len)
                && self.indices.windows(2).all(|pair| pair[0] < pair[1])
                && match &self.actions {
                    None => self.indices.is_empty(),
                    Some(actions) => {
                        !self.indices.is_empty()
                            && actions.dims() == [self.indices.len(), dimensions]
                    }
                },
            "invalid target action batch",
        )
    }
}
pub(crate) trait CriticRule {
    type Targets;
    fn action_dim(&self) -> usize;
    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets>;
    fn update(&mut self, batch: &PreparedReplayBatch<'_>, targets: Self::Targets) -> Result<f64>;
    fn actor_utility(
        &self,
        observations: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<1>>;
    fn update_targets(&mut self, tau: f32) -> Result<()>;
    fn validate(&self) -> Result<()>;
    fn validate_training(&self, config: &ContinuousConfig) -> Result<()> {
        config.validate()
    }
}
pub(crate) trait CriticCheckpoint: CriticRule + Sized {
    type Metadata: Serialize + DeserializeOwned;
    fn metadata(&self) -> Self::Metadata;
    fn record(&self, record: &mut LearnerRecord) -> Result<()>;
    fn restored(
        &self,
        record: &LearnerRecord,
        metadata: Self::Metadata,
        config: &OptimizerConfig,
    ) -> Result<Self>;
}
pub(crate) struct CriticModel<Q: ActionValue> {
    pub online: Q,
    pub target: Q,
    optimizer: ModuleOptimizer,
    learning_rate: f64,
}
impl<Q: ActionValue> CriticModel<Q> {
    pub fn new(online: Q, config: &OptimizerConfig) -> Result<Self> {
        Ok(Self {
            target: frozen_copy(&online),
            online,
            optimizer: config.build()?,
            learning_rate: config.learning_rate,
        })
    }
    pub fn fit(&mut self, loss: Tensor<1>) -> Result<f64> {
        optimize(
            &mut self.online,
            &mut self.optimizer,
            self.learning_rate,
            loss,
        )
    }
    /// New learners can opt into finite-gradient checks without changing legacy steps.
    pub fn fit_validated(&mut self, loss: Tensor<1>) -> Result<f64> {
        let value = crate::nn::optimization::optimize_validated(
            &mut self.online,
            &mut self.optimizer,
            self.learning_rate,
            loss,
        )?
        .0;
        crate::nn::optimization::parameter_norm(&self.online.valid())?;
        Ok(value)
    }
    pub fn update_target(&mut self, tau: f32) -> Result<()> {
        self.target = self.target.updated_target(&self.online, tau)?;
        Ok(())
    }
    pub fn future(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Vec<f32>> {
        let width = self.online.output_dim();
        let mut values = vec![0.; batch.items.len() * width];
        if let Some(actions) = &next.actions {
            let live = to_vec(
                self.target
                    .valid()
                    .values(next.observations(), actions.clone())?,
            )?;
            for (row, &i) in next.indices.iter().enumerate() {
                values[i * width..(i + 1) * width]
                    .copy_from_slice(&live[row * width..(row + 1) * width]);
            }
        }
        Ok(values)
    }
    pub fn validate(&self, width: usize) -> Result<()> {
        ensure(
            width > 0
                && self.online.output_dim() == width
                && self.target.output_dim() == width
                && self.online.action_dim() > 0
                && self.target.action_dim() == self.online.action_dim(),
            "critic dimensions do not match algorithm",
        )?;
        ensure(
            self.online.checkpoint_coordinates()? == self.target.checkpoint_coordinates()?,
            "online and target critic input coordinates differ",
        )
    }
    pub fn record(&self, record: &mut LearnerRecord, online: &str, target: &str) -> Result<()> {
        record.model(online, &self.online)?;
        record.model(target, &self.target)?;
        for (name, model) in [(online, &self.online), (target, &self.target)] {
            if let Some(coordinates) = model.checkpoint_coordinates()? {
                record.metadata["critic_coordinates"][name] = coordinates;
            }
        }
        record.optimizer(online, &self.optimizer)
    }
    pub fn restored(
        &self,
        record: &LearnerRecord,
        online: &str,
        target: &str,
        config: &OptimizerConfig,
    ) -> Result<Self> {
        let online_model = record.load_model(online, &self.online)?;
        let target_model = record.load_model(target, &self.target)?;
        for (name, model) in [(online, &online_model), (target, &target_model)] {
            let recorded = record
                .metadata
                .get("critic_coordinates")
                .and_then(|map| map.get(name));
            ensure(
                recorded == model.checkpoint_coordinates()?.as_ref(),
                "checkpoint critic input coordinates differ from the restored model",
            )?;
        }
        Ok(Self {
            online: online_model,
            target: target_model,
            optimizer: record.load_optimizer(online, config.build()?)?,
            learning_rate: config.learning_rate,
        })
    }
}
impl<Q: ActionValue + ModuleDisplay> CriticModel<ResidualQ<Q>> {
    pub fn frozen_replacement(&self, anchor: &Deterministic) -> Result<Deterministic> {
        self.online.frozen_replacement(anchor)
    }

    pub fn install_anchor(&mut self, anchor: Deterministic) {
        self.online.install_anchor(anchor.clone());
        self.target.install_anchor(anchor);
    }
}
pub(crate) fn bellman(
    batch: &PreparedReplayBatch<'_>,
    mut future: Vec<f32>,
    width: usize,
) -> Tensor<2> {
    for (row, item) in batch.items.iter().enumerate() {
        for value in &mut future[row * width..(row + 1) * width] {
            *value = item.transition.reward.total() + item.discount * *value;
        }
    }
    matrix(future, batch.items.len(), width)
}
