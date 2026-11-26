use super::{
    common::OptimizerConfig,
    continuous::{
        ContinuousConfig, ContinuousLearner,
        critic::{
            CriticCheckpoint, CriticModel, CriticRule, PreparedReplayBatch, TargetActionBatch,
        },
        delegate_learner,
    },
};
use crate::{
    Result,
    core::{
        checkpoint::{Checkpoint, LearnerRecord},
        ensure,
        env::ObservationBatch,
    },
    nn::{
        critics::residual::ResidualQ,
        critics::{
            quantile::QuantileGrid,
            scalar::{ActionValue, QValue},
        },
        matrix,
        policy::{Deterministic, DeterministicPolicy},
        target::frozen_copy,
        to_vec,
    },
    objectives::{
        quantile::{QuantileLoss, quantile_loss},
        risk::{Risk, RiskObjective},
    },
};
use burn::{module::ModuleDisplay, tensor::Tensor};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct D4pgConfig<R = Risk> {
    pub training: ContinuousConfig,
    pub grid: QuantileGrid,
    pub loss: QuantileLoss,
    pub risk: R,
}
impl<P: DeterministicPolicy, Q: ActionValue + ModuleDisplay, R: RiskObjective>
    D4pg<P, ResidualQ<Q>, R>
{
    /// Explicitly replace the online and target critics' shared frozen anchor.
    pub fn refresh_residual_anchor(&mut self, anchor: &Deterministic) -> Result<()> {
        let model = &mut self.inner.critic.model;
        let anchor = model.frozen_replacement(anchor)?;
        model.install_anchor(anchor);
        Ok(())
    }
}
impl D4pgConfig {
    pub fn new(quantiles: usize) -> Result<Self> {
        Ok(Self {
            training: Default::default(),
            grid: QuantileGrid::uniform(quantiles)?,
            loss: Default::default(),
            risk: Risk::Expected,
        })
    }
}
pub struct D4pg<
    P: DeterministicPolicy = Deterministic,
    Q: ActionValue = QValue,
    R: RiskObjective = Risk,
> {
    inner: ContinuousLearner<P, QuantileCritic<Q, R>>,
}
impl<P: DeterministicPolicy, Q: ActionValue, R: RiskObjective> D4pg<P, Q, R> {
    pub fn new(policy: P, critic: Q, config: D4pgConfig<R>, seed: u64) -> Result<Self> {
        let critic = QuantileCritic {
            model: CriticModel::new(critic, &config.training.critic)?,
            grid: config.grid,
            loss: config.loss,
            risk: config.risk,
            residuals: Vec::new(),
        };
        Ok(Self {
            inner: ContinuousLearner::new(policy, critic, config.training, seed)?,
        })
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn training_config(&self) -> &ContinuousConfig {
        self.inner.training_config()
    }
    #[cfg(feature = "diagnostics")]
    pub(crate) fn risk(&self) -> &R {
        &self.inner.critic.risk
    }
    /// Frozen critic with raw regression heads; use `ordered_quantiles` to interpret them as quantiles.
    pub fn critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.online)
    }
    /// Predict ascending quantiles per row for inference, diagnostics, and visualisation.
    /// Sorting retains gradients with respect to actions; critic parameters are frozen.
    pub fn ordered_quantiles(
        &self,
        obs: ObservationBatch<'_>,
        actions: Tensor<2>,
    ) -> Result<Tensor<2>> {
        Ok(self.critic().values(obs, actions)?.sort(1))
    }
    pub fn target_critic(&self) -> Q {
        frozen_copy(&self.inner.critic.model.target)
    }
    pub fn grid(&self) -> &QuantileGrid {
        &self.inner.critic.grid
    }
    pub fn loss(&self) -> &QuantileLoss {
        &self.inner.critic.loss
    }
}
pub(crate) struct QuantileCritic<Q: ActionValue, R: RiskObjective> {
    model: CriticModel<Q>,
    grid: QuantileGrid,
    loss: QuantileLoss,
    risk: R,
    // Transient scratch is excluded from persistent metadata.
    residuals: Vec<f32>,
}
pub(crate) struct QuantileTargets {
    tensor: Tensor<2>,
    host: Option<Vec<f32>>,
}
#[cfg(test)]
thread_local! { static QUANTILE_HOST_PREPARATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
impl<Q: ActionValue, R: RiskObjective> CriticRule for QuantileCritic<Q, R> {
    type Targets = QuantileTargets;
    fn action_dim(&self) -> usize {
        self.model.online.action_dim()
    }
    fn targets(
        &self,
        batch: &PreparedReplayBatch<'_>,
        next: &TargetActionBatch<'_>,
    ) -> Result<Self::Targets> {
        let k = self.grid.len();
        let mut future = self.model.future(batch, next)?;
        for (row, item) in batch.items.iter().enumerate() {
            for value in &mut future[row * k..(row + 1) * k] {
                *value = item.transition.reward.total() + item.discount * *value;
            }
        }
        let host = self.loss.requires_adaptation().then(|| future.clone());
        Ok(QuantileTargets {
            tensor: matrix(future, batch.items.len(), k),
            host,
        })
    }
    fn update(&mut self, batch: &PreparedReplayBatch<'_>, target: Self::Targets) -> Result<f64> {
        let QuantileTargets {
            tensor: target,
            host,
        } = target;
        let pred = self
            .model
            .online
            .values(batch.observations(), batch.actions.clone())?;
        if self.loss.requires_adaptation() {
            #[cfg(test)]
            QUANTILE_HOST_PREPARATIONS.with(|count| count.set(count.get() + 1));
            let p = to_vec(pred.clone())?;
            let t = host.expect("adaptive targets retain detached host values");
            let k = self.grid.len();
            self.residuals.clear();
            self.residuals.reserve(batch.items.len() * k * k);
            for i in 0..batch.items.len() {
                for j in 0..k {
                    for h in 0..k {
                        self.residuals.push(t[i * k + h] - p[i * k + j]);
                    }
                }
            }
            self.loss.adapt_in_place(&mut self.residuals)?;
        }
        self.model
            .fit(quantile_loss(pred, target, &self.grid, &self.loss)?)
    }
    fn actor_utility(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<1>> {
        self.risk.quantiles(
            // Interpret heads as ascending quantiles only at the actor boundary.
            self.model
                .online
                .valid()
                .freeze()
                .values(obs, actions)?
                .sort(1),
            &self.grid,
        )
    }
    fn update_targets(&mut self, tau: f32) -> Result<()> {
        self.model.update_target(tau)
    }
    fn validate(&self) -> Result<()> {
        self.grid.validate()?;
        self.loss.validate()?;
        self.model.validate(self.grid.len())?;
        let _ = self.risk.quantiles(
            matrix(vec![0.; self.grid.len()], 1, self.grid.len()),
            &self.grid,
        )?;
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
pub(crate) struct QuantileMeta<R> {
    grid: QuantileGrid,
    loss: QuantileLoss,
    risk: R,
}
impl<Q: ActionValue, R: RiskObjective + Serialize + DeserializeOwned> CriticCheckpoint
    for QuantileCritic<Q, R>
{
    type Metadata = QuantileMeta<R>;
    fn metadata(&self) -> Self::Metadata {
        QuantileMeta {
            grid: self.grid.clone(),
            loss: self.loss.clone(),
            risk: self.risk.clone(),
        }
    }
    fn record(&self, record: &mut LearnerRecord) -> Result<()> {
        self.model.record(record, "critic", "target_critic")
    }
    fn restored(
        &self,
        record: &LearnerRecord,
        metadata: Self::Metadata,
        config: &OptimizerConfig,
    ) -> Result<Self> {
        metadata.grid.validate()?;
        metadata.loss.validate()?;
        ensure(
            metadata.grid.edges == self.grid.edges,
            "checkpoint quantile grid mismatch",
        )?;
        Ok(Self {
            model: self
                .model
                .restored(record, "critic", "target_critic", config)?,
            grid: metadata.grid,
            loss: metadata.loss,
            risk: metadata.risk,
            residuals: Vec::new(),
        })
    }
}
impl<P: DeterministicPolicy, Q: ActionValue, R: RiskObjective + Serialize + DeserializeOwned>
    Checkpoint for D4pg<P, Q, R>
{
    fn checkpoint(&self) -> Result<LearnerRecord> {
        self.inner.checkpoint("d4pg")
    }
    fn restore(&mut self, record: &LearnerRecord) -> Result<()> {
        self.inner.restore(record, "d4pg")
    }
}
delegate_learner!(D4pg, R);

#[cfg(test)]
mod quantile_preparation_tests {
    use super::*;
    use crate::{
        core::{
            action::{Action, ActionSpace},
            env::Observation,
            learner::Learner,
            transition::Transition,
        },
        data::replay::{ReplayBatch, ReplayItem},
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn crossed_learner() -> D4pg {
        use burn::module::Param;
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let policy = Deterministic::new(
            1,
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            &[],
            &mut rng,
        )
        .unwrap();
        let mut critic = QValue::new(1, 1, 3, &[], &mut rng).unwrap();
        critic.network.layers[0].weight =
            Param::from_tensor(matrix(vec![0., 0., 0., -1., 2., 3.], 2, 3));
        critic.network.layers[0].bias = Some(Param::from_tensor(crate::nn::vector(&[3., -1., 2.])));
        D4pg::new(policy, critic, D4pgConfig::new(3).unwrap(), 17).unwrap()
    }

    #[test]
    fn adaptive_targets_keep_host_values_for_mixed_terminal_truncated_and_continuing_rows() {
        let mut learner = crossed_learner();
        learner.inner.critic.loss = QuantileLoss::Generalized {
            scale: 1.,
            ema: 0.9,
            minimum: 0.001,
        };
        let batch = ReplayBatch {
            items: [
                (2., 0., true, false),
                (1., 0.5, false, true),
                (-1., 0.25, false, false),
            ]
            .into_iter()
            .map(|(reward, discount, terminated, truncated)| ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![0.]),
                    action: Action::Continuous(vec![0.]),
                    reward: reward.into(),
                    next_observation: Observation::new(vec![0.]),
                    terminated,
                    truncated,
                },
                discount,
            })
            .collect(),
        };
        let space = ActionSpace::Box {
            low: vec![-1.],
            high: vec![1.],
        };
        let prepared = PreparedReplayBatch::new(&batch, &space).unwrap();
        let next = TargetActionBatch {
            indices: vec![1, 2],
            observations: vec![
                &batch.items[1].transition.next_observation,
                &batch.items[2].transition.next_observation,
            ],
            actions: Some(matrix(vec![0., 1.], 2, 1)),
        };
        let targets = learner.inner.critic.targets(&prepared, &next).unwrap();
        let expected = [2., 2., 2., 2.5, 0.5, 2., -0.5, -0.75, 0.25];
        assert_eq!(targets.host.as_deref().unwrap(), expected);
        assert_eq!(to_vec(targets.tensor.clone()).unwrap(), expected);
        assert!(!targets.tensor.is_require_grad());
    }

    #[test]
    fn actor_orders_crossed_quantiles_and_preserves_action_gradients() {
        let mut learner = crossed_learner();
        let obs = [Observation::new(vec![0.]), Observation::new(vec![0.])];
        for (risk, grid, values, gradients) in [
            (
                Risk::Cvar { alpha: 0.5 },
                QuantileGrid::uniform(3).unwrap(),
                [0., 4. / 3.],
                [7. / 3., 1.],
            ),
            (
                Risk::Expected,
                QuantileGrid {
                    edges: vec![0., 0.25, 0.5, 1.],
                },
                [1.75, 3.25],
                [0.75, 1.75],
            ),
        ] {
            learner.inner.critic.risk = risk;
            learner.inner.critic.grid = grid;
            let actions = matrix(vec![0., 1.], 2, 1).require_grad();
            let utility = learner
                .inner
                .critic
                .actor_utility((&obs).into(), actions.clone())
                .unwrap();
            for (actual, expected) in to_vec(utility.clone()).unwrap().iter().zip(values) {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
            let grads = utility.sum().backward();
            for (actual, expected) in to_vec(actions.grad(&grads).unwrap())
                .unwrap()
                .iter()
                .zip(gradients)
            {
                assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
            }
        }
    }

    #[test]
    fn ordered_predictions_preserve_raw_critic_heads() {
        let learner = crossed_learner();
        let obs = [Observation::new(vec![0.]), Observation::new(vec![0.])];
        let actions = matrix(vec![0., 1.], 2, 1);
        let raw = || {
            to_vec(
                learner
                    .critic()
                    .values((&obs).into(), actions.clone())
                    .unwrap(),
            )
            .unwrap()
        };
        assert_eq!(raw(), [3., -1., 2., 2., 1., 5.]);
        assert_eq!(
            to_vec(
                learner
                    .ordered_quantiles((&obs).into(), actions.clone())
                    .unwrap()
            )
            .unwrap(),
            [-1., 2., 3., 1., 2., 5.]
        );
        assert_eq!(raw(), [3., -1., 2., 2., 1., 5.]);
    }

    #[test]
    fn critic_regression_keeps_crossed_head_locations() {
        let mut learner = crossed_learner();
        learner.inner.critic.loss = QuantileLoss::Absolute;
        let batch = ReplayBatch {
            items: vec![ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![0.]),
                    action: Action::Continuous(vec![0.]),
                    reward: 0.0_f32.into(),
                    next_observation: Observation::new(vec![0.]),
                    terminated: true,
                    truncated: false,
                },
                discount: 0.,
            }],
        };
        let report = learner.update(&batch).unwrap();
        assert!((report.metrics["critic_loss"] - 10. / 9.).abs() < 1e-6);
    }

    fn update_twice(loss: QuantileLoss) -> usize {
        let mut rng = ChaCha8Rng::seed_from_u64(13);
        let policy = Deterministic::new(
            1,
            ActionSpace::Box {
                low: vec![-1.],
                high: vec![1.],
            },
            &[],
            &mut rng,
        )
        .unwrap();
        let critic = QValue::new(1, 1, 4, &[], &mut rng).unwrap();
        let mut config = D4pgConfig::new(4).unwrap();
        config.loss = loss;
        let mut learner = D4pg::new(policy, critic, config, 17).unwrap();
        let mut batch = ReplayBatch {
            items: vec![ReplayItem {
                transition: Transition {
                    observation: Observation::new(vec![1.]),
                    action: Action::Continuous(vec![0.]),
                    reward: 1.0_f32.into(),
                    next_observation: Observation::new(vec![2.]),
                    terminated: true,
                    truncated: false,
                },
                discount: 0.,
            }],
        };
        QUANTILE_HOST_PREPARATIONS.with(|count| count.set(0));
        learner.update(&batch).unwrap();
        let capacity = learner.inner.critic.residuals.capacity();
        let pointer = learner.inner.critic.residuals.as_ptr();
        batch.items[0].transition.terminated = false;
        batch.items[0].discount = 0.9;
        learner.update(&batch).unwrap();
        assert_eq!(learner.inner.critic.residuals.capacity(), capacity);
        assert_eq!(learner.inner.critic.residuals.as_ptr(), pointer);
        QUANTILE_HOST_PREPARATIONS.with(std::cell::Cell::get)
    }
    #[test]
    fn fixed_quantile_losses_skip_host_residual_preparation() {
        for loss in [QuantileLoss::Absolute, QuantileLoss::Huber { kappa: 1. }] {
            assert_eq!(update_twice(loss), 0);
        }
    }
    #[test]
    fn generalized_prepares_host_residuals_once_per_update() {
        assert_eq!(
            update_twice(QuantileLoss::Generalized {
                scale: 1.,
                ema: 0.9,
                minimum: 0.001
            }),
            2
        );
    }
}
