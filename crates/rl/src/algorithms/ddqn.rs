use super::common::{OptimizerConfig, optimize};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, ObservationBatch},
        learner::{Learner, OffPolicyLearner, UpdateReport},
    },
    data::replay::ReplayBatch,
    nn::{
        critics::scalar::{DiscreteActionValue, DiscreteQ},
        device,
        target::polyak_update,
        to_vec, vector,
    },
};
use burn::{
    optim::ModuleOptimizer,
    tensor::{Int, Tensor, TensorData},
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DdqnConfig {
    pub optimizer: OptimizerConfig,
    pub epsilon: f32,
    pub target_tau: f32,
    pub target_interval: u64,
}
impl Default for DdqnConfig {
    fn default() -> Self {
        Self {
            optimizer: Default::default(),
            epsilon: 0.1,
            target_tau: 0.01,
            target_interval: 1,
        }
    }
}
#[derive(Debug, Clone)]
struct QActor<Q: DiscreteActionValue> {
    online: Q,
    space: ActionSpace,
    pub epsilon: f32,
}
fn greedy(values: &[f32], o: &Observation) -> Result<usize> {
    ensure(values.iter().all(|x| x.is_finite()), "nonfinite Q values")?;
    let valid = (0..values.len()).filter(|&i| o.dense_mask().is_none_or(|m| m[i]));
    valid
        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
        .ok_or_else(|| crate::Error::Action("all-invalid action mask on a live state".into()))
}
impl<Q: DiscreteActionValue> Actor for QActor<Q> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        for o in obs.iter() {
            o.validate(o.dense_values().len(), &self.space, true)?;
        }
        let k = self.online.action_count();
        let values = to_vec(self.online.valid().action_values(obs)?)?;
        ensure(values.len() == obs.len() * k, "Q output shape mismatch")?;
        actions.clear();
        for (i, o) in obs.iter().enumerate() {
            let a = if mode == ActMode::Sample
                && self.epsilon > 0.
                && rng.random::<f32>() < self.epsilon
            {
                let mut valid = (0..k).filter(|&a| o.dense_mask().is_none_or(|m| m[a]));
                let count = valid.clone().count();
                valid
                    .nth(rng.random_range(0..count))
                    .expect("validated live mask")
            } else {
                greedy(&values[i * k..(i + 1) * k], o)?
            };
            actions.push(Action::Discrete(a));
        }
        Ok(())
    }
}
#[cfg(feature = "diagnostics")]
pub(crate) fn diagnostic_greedy_actor<Q: DiscreteActionValue>(online: Q) -> impl Actor {
    let space = ActionSpace::Discrete {
        count: online.action_count(),
    };
    QActor {
        online,
        space,
        epsilon: 0.,
    }
}
pub struct Ddqn<Q: DiscreteActionValue = DiscreteQ> {
    actor: QActor<Q>,
    target: Q,
    optimizer: ModuleOptimizer,
    config: DdqnConfig,
    updates: u64,
}
impl<Q: DiscreteActionValue> Ddqn<Q> {
    pub fn new(online: Q, config: DdqnConfig) -> Result<Self> {
        ensure(
            online.action_count() > 0
                && config.epsilon.is_finite()
                && (0.0..=1.0).contains(&config.epsilon)
                && config.target_interval > 0
                && config.target_tau.is_finite()
                && (0.0..=1.0).contains(&config.target_tau),
            "invalid DDQN configuration",
        )?;
        let target = crate::nn::target::frozen_copy(&online);
        let space = ActionSpace::Discrete {
            count: online.action_count(),
        };
        Ok(Self {
            actor: QActor {
                online,
                space,
                epsilon: config.epsilon,
            },
            target,
            optimizer: config.optimizer.build()?,
            config,
            updates: 0,
        })
    }
}
impl<Q: DiscreteActionValue> Learner for Ddqn<Q> {
    type Batch = ReplayBatch;
    fn actor(&self) -> &dyn Actor {
        &self.actor
    }
    fn update(&mut self, b: &ReplayBatch) -> Result<UpdateReport> {
        b.validate()?;
        let n = b.items.len();
        let k = self.actor.online.action_count();
        let mut y: Vec<_> = b
            .items
            .iter()
            .map(|x| x.transition.reward.total())
            .collect();
        let live: Vec<_> = b
            .items
            .iter()
            .enumerate()
            .filter(|(_, x)| x.discount > 0.)
            .map(|(i, _)| i)
            .collect();
        if !live.is_empty() {
            let obs: Vec<_> = live
                .iter()
                .map(|&i| &b.items[i].transition.next_observation)
                .collect();
            for o in &obs {
                o.validate(o.dense_values().len(), &self.actor.space, true)?;
            }
            let online = to_vec(
                self.actor
                    .online
                    .valid()
                    .action_values(ObservationBatch::from_refs(&obs))?,
            )?;
            let target = to_vec(
                self.target
                    .valid()
                    .action_values(ObservationBatch::from_refs(&obs))?,
            )?;
            for (j, &i) in live.iter().enumerate() {
                let a = greedy(&online[j * k..(j + 1) * k], obs[j])?;
                y[i] += b.items[i].discount * target[j * k + a];
            }
        }
        let obs: Vec<_> = b.items.iter().map(|x| &x.transition.observation).collect();
        let mut indices = Vec::with_capacity(n);
        for item in &b.items {
            self.actor.space.check(
                &item.transition.action,
                item.transition.observation.dense_mask(),
            )?;
            let Action::Discrete(a) = item.transition.action else {
                unreachable!()
            };
            indices.push(a as i64);
        }
        let index = Tensor::<2, Int>::from_data(TensorData::new(indices, [n, 1]), &device());
        let residual = self
            .actor
            .online
            .action_values(ObservationBatch::from_refs(&obs))?
            .gather(1, index)
            .squeeze_dim(1)
            - vector(&y);
        let abs = residual.abs();
        let quadratic = abs.clone().clamp(0., 1.);
        let loss = (quadratic.clone().powf_scalar(2.) * 0.5 + abs - quadratic).mean();
        let number = optimize(
            &mut self.actor.online,
            &mut self.optimizer,
            self.config.optimizer.learning_rate,
            loss,
        )?;
        self.updates += 1;
        if self.updates.is_multiple_of(self.config.target_interval) {
            self.target = polyak_update(
                self.target.clone(),
                &self.actor.online,
                self.config.target_tau,
            )?;
        }
        Ok(UpdateReport {
            updates: 1,
            ..Default::default()
        }
        .metric("critic_loss", number))
    }
}
impl<Q: DiscreteActionValue> OffPolicyLearner for Ddqn<Q> {
    fn set_exploration(&mut self, x: f32) -> Result<()> {
        ensure(
            x.is_finite() && (0.0..=1.0).contains(&x),
            "epsilon must be in [0,1]",
        )?;
        self.actor.epsilon = x;
        Ok(())
    }
}
fn validate_config(config: &DdqnConfig) -> Result<()> {
    ensure(
        config.epsilon.is_finite()
            && (0.0..=1.0).contains(&config.epsilon)
            && config.target_interval > 0
            && config.target_tau.is_finite()
            && (0.0..=1.0).contains(&config.target_tau),
        "invalid DDQN configuration",
    )?;
    config.optimizer.build()?;
    Ok(())
}
impl<Q: DiscreteActionValue> Ddqn<Q> {
    pub fn config(&self) -> &DdqnConfig {
        &self.config
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    pub fn policy(&self) -> Q {
        crate::nn::target::frozen_copy(&self.actor.online)
    }
    pub fn target(&self) -> Q {
        crate::nn::target::frozen_copy(&self.target)
    }
    pub fn exploration(&self) -> f32 {
        self.actor.epsilon
    }
}
use crate::{
    core::checkpoint::{Checkpoint, LearnerRecord},
    nn::checkpoint::RecordBurnExt,
};
#[derive(Serialize, Deserialize)]
struct DdqnMeta {
    config: DdqnConfig,
    updates: u64,
    epsilon: f32,
    space: ActionSpace,
}
impl<Q: DiscreteActionValue> Checkpoint for Ddqn<Q> {
    fn checkpoint(&self) -> Result<LearnerRecord> {
        let mut r = LearnerRecord::new(
            "ddqn",
            DdqnMeta {
                config: self.config.clone(),
                updates: self.updates,
                epsilon: self.actor.epsilon,
                space: self.actor.space.clone(),
            },
        )?;
        r.model("online", &self.actor.online)?;
        r.model("target", &self.target)?;
        r.optimizer("critic", &self.optimizer)?;
        Ok(r)
    }
    fn restore(&mut self, r: &LearnerRecord) -> Result<()> {
        r.check("ddqn")?;
        let m: DdqnMeta = serde_json::from_value(r.metadata.clone())?;
        ensure(
            m.space == self.actor.space,
            "checkpoint action space mismatch",
        )?;
        validate_config(&m.config)?;
        ensure(
            m.epsilon.is_finite() && (0.0..=1.0).contains(&m.epsilon),
            "invalid restored epsilon",
        )?;
        let online = r.load_model("online", &self.actor.online)?;
        let target = r.load_model("target", &self.target)?;
        let optimizer = r.load_optimizer("critic", m.config.optimizer.build()?)?;
        self.actor.online = online;
        self.target = target;
        self.optimizer = optimizer;
        self.config = m.config;
        self.updates = m.updates;
        self.actor.epsilon = m.epsilon;
        Ok(())
    }
}
