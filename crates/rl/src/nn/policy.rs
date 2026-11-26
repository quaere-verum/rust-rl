use super::{
    critics::scalar::{StateValue, Value},
    device,
    distributions::Distribution,
    network::{Mlp, Network},
    observations,
};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor, RolloutActor, RolloutBehavior},
        ensure,
        env::ObservationBatch,
    },
};
use burn::{
    module::{Module, Param},
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;

pub trait StochasticPolicy: Module {
    fn action_space(&self) -> &ActionSpace;
    fn distribution(&self, obs: ObservationBatch<'_>) -> Result<Distribution>;
}

/// Deploy a stochastic policy without inventing a value function or retaining a trainer.
pub struct PolicyActor<P: StochasticPolicy = Policy> {
    policy: P,
}
impl<P: StochasticPolicy> PolicyActor<P> {
    pub fn new(policy: P) -> Result<Self> {
        policy.action_space().validate()?;
        if !matches!(policy.action_space(), ActionSpace::Discrete { .. }) {
            super::gaussian_policy::validate_gaussian_space(policy.action_space())?;
        }
        let policy = super::target::frozen_copy(&policy);
        super::optimization::parameter_norm(&policy)?;
        Ok(Self { policy })
    }
}
impl<P: StochasticPolicy> Actor for PolicyActor<P> {
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        sample_checked(&self.policy.valid(), obs, mode, rng, actions)
    }
}

/// Opt-in structural validation for new policy-only consumers; legacy inference
/// paths retain their arithmetic and established validation costs.
pub(crate) fn checked_distribution<P: StochasticPolicy>(
    policy: &P,
    obs: ObservationBatch<'_>,
) -> Result<Distribution> {
    ensure(!obs.is_empty(), "empty policy observation batch")?;
    ensure(
        !matches!(policy.action_space(), ActionSpace::Conditional(_)),
        "dense policy requires a fixed action space",
    )?;
    if !matches!(policy.action_space(), ActionSpace::Discrete { .. }) {
        ensure(
            obs.iter().all(|o| {
                o.as_dense()
                    .is_some_and(|dense| dense.action_mask.is_none())
            }),
            "continuous policy requires dense observations without action masks",
        )?;
    }
    let distribution = policy.distribution(obs)?;
    let valid = match (&distribution, policy.action_space()) {
        (Distribution::Categorical { logits, masks }, ActionSpace::Discrete { count }) => {
            logits.dims() == [obs.len(), *count]
                && masks.len() == obs.len() * count
                && obs.iter().zip(masks.chunks(*count)).all(|(o, m)| {
                    o.as_dense().is_some()
                        && o.dense_mask()
                            .map_or_else(|| m.iter().all(|v| *v), |expected| m == expected)
                })
        }
        (
            Distribution::Gaussian {
                mean,
                log_std,
                bounds,
            },
            space,
        ) => {
            mean.dims() == [obs.len(), space.dimensions()]
                && log_std.dims() == mean.dims()
                && match (bounds, space) {
                    (Some((l, h)), ActionSpace::Box { low, high }) => l == low && h == high,
                    (None, ActionSpace::Unbounded { .. }) => true,
                    _ => false,
                }
        }
        _ => false,
    };
    ensure(
        valid,
        "policy distribution differs from its public space or observation mask",
    )?;
    Ok(distribution)
}
pub(crate) fn sample_checked<P: StochasticPolicy>(
    policy: &P,
    obs: ObservationBatch<'_>,
    mode: ActMode,
    rng: &mut ChaCha8Rng,
    actions: &mut Vec<Action>,
) -> Result<()> {
    sample_executed_actions(
        &checked_distribution(policy, obs)?,
        policy.action_space(),
        obs,
        mode,
        rng,
        actions,
    )
}

pub(crate) fn sample_executed_actions(
    distribution: &Distribution,
    space: &ActionSpace,
    obs: ObservationBatch<'_>,
    mode: ActMode,
    rng: &mut ChaCha8Rng,
    actions: &mut Vec<Action>,
) -> Result<()> {
    distribution.sample_actions_into(mode, rng, actions)?;
    if let ActionSpace::Box { low, high } = space {
        for action in actions.iter_mut() {
            if let Action::Continuous(values) = action {
                for ((x, &l), &h) in values.iter_mut().zip(low).zip(high) {
                    *x = x.clamp(l, h);
                }
            }
        }
    }
    for (action, observation) in actions.iter().zip(obs.iter()) {
        space.check(action, observation.dense_mask())?;
    }
    Ok(())
}
#[derive(Module, Debug)]
pub struct Policy<M: Network = Mlp> {
    pub network: M,
    pub log_std: Param<Tensor<1>>,
    #[module(skip)]
    pub space: ActionSpace,
}
impl Policy<Mlp> {
    pub fn new(
        width: usize,
        space: ActionSpace,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        space.validate()?;
        let output = match space {
            ActionSpace::Discrete { count } => count,
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. } => space.dimensions(),
            ActionSpace::Conditional(_) => {
                return Err(crate::Error::Config(
                    "dense policy requires a fixed action space".into(),
                ));
            }
        };
        Self::from_network(Mlp::new(width, hidden, output, rng)?, space)
    }
}
impl<M: Network> Policy<M> {
    pub fn from_network(network: M, space: ActionSpace) -> Result<Self> {
        Self::from_network_on_device(network, space, &device())
    }
    /// The stochastic log-standard-deviation template is inference-only too.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn from_network_inference(network: M, space: ActionSpace) -> Result<Self> {
        Self::from_network_on_device(network, space, &Device::flex())
    }
    fn from_network_on_device(network: M, space: ActionSpace, device: &Device) -> Result<Self> {
        space.validate()?;
        let output = match space {
            ActionSpace::Discrete { count } => count,
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. } => space.dimensions(),
            ActionSpace::Conditional(_) => {
                return Err(crate::Error::Config(
                    "dense policy requires a fixed action space".into(),
                ));
            }
        };
        ensure(
            network.output_dim() == output,
            "policy output width mismatch",
        )?;
        let log_std = Param::from_tensor(Tensor::zeros([space.dimensions()], device));
        Ok(Self {
            network,
            log_std,
            space,
        })
    }
}
impl<M: Network> StochasticPolicy for Policy<M> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn distribution(&self, obs: ObservationBatch<'_>) -> Result<Distribution> {
        for o in obs.iter() {
            o.validate(self.network.input_dim(), &self.space, true)?;
        }
        let output = self
            .network
            .forward(observations(obs, self.network.input_dim())?);
        match &self.space {
            ActionSpace::Discrete { count } => {
                let mut masks = Vec::with_capacity(obs.len() * count);
                for o in obs.iter() {
                    if let Some(mask) = o.dense_mask() {
                        masks.extend_from_slice(mask);
                    } else {
                        masks.resize(masks.len() + count, true);
                    }
                }
                Ok(Distribution::Categorical {
                    logits: output,
                    masks,
                })
            }
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. } => {
                let d = self.space.dimensions();
                let log_std = self
                    .log_std
                    .val()
                    .clamp(-20., 2.)
                    .reshape([1, d])
                    .repeat_dim(0, obs.len());
                let bounds = if let ActionSpace::Box { low, high } = &self.space {
                    Some((low.clone(), high.clone()))
                } else {
                    None
                };
                Ok(Distribution::Gaussian {
                    mean: output,
                    log_std,
                    bounds,
                })
            }
            ActionSpace::Conditional(_) => Err(crate::Error::Config(
                "dense policy requires a fixed action space".into(),
            )),
        }
    }
}

pub trait DeterministicPolicy: Module {
    fn action_space(&self) -> &ActionSpace;
    fn actions(&self, obs: ObservationBatch<'_>) -> Result<Tensor<2>>;
}
#[derive(Module, Debug)]
pub struct Deterministic<M: Network = Mlp> {
    pub network: M,
    #[module(skip)]
    pub space: ActionSpace,
}
impl Deterministic<Mlp> {
    pub fn new(
        width: usize,
        space: ActionSpace,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let network = Mlp::new(width, hidden, space.dimensions(), rng)?;
        Self::from_network(network, space)
    }
}
impl<M: Network> Deterministic<M> {
    pub fn from_network(network: M, space: ActionSpace) -> Result<Self> {
        space.validate()?;
        ensure(
            matches!(
                space,
                ActionSpace::Box { .. } | ActionSpace::Unbounded { .. }
            ) && network.output_dim() == space.dimensions(),
            "deterministic policy requires continuous matching outputs",
        )?;
        Ok(Self { network, space })
    }

    pub(crate) fn actions_on_device(
        &self,
        obs: ObservationBatch<'_>,
        device: &burn::tensor::Device,
    ) -> Result<Tensor<2>> {
        let a = self.network.forward(super::observations_on_device(
            obs,
            self.network.input_dim(),
            device,
        )?);
        if let ActionSpace::Box { low, high } = &self.space {
            let d = low.len();
            let n = obs.len();
            let scale = super::matrix_on_device(
                low.iter().zip(high).map(|(l, h)| (h - l) * 0.5).collect(),
                1,
                d,
                &a.device(),
            )
            .repeat_dim(0, n);
            let center = super::matrix_on_device(
                low.iter().zip(high).map(|(l, h)| (h + l) * 0.5).collect(),
                1,
                d,
                &a.device(),
            )
            .repeat_dim(0, n);
            Ok(a.tanh() * scale + center)
        } else {
            Ok(a)
        }
    }
}
impl<M: Network> DeterministicPolicy for Deterministic<M> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn actions(&self, obs: ObservationBatch<'_>) -> Result<Tensor<2>> {
        self.actions_on_device(obs, &super::device())
    }
}

#[derive(Debug, Clone)]
pub struct ActorCritic<P: StochasticPolicy = Policy, V: StateValue = Value> {
    pub policy: P,
    pub value: V,
}
impl<P: StochasticPolicy, V: StateValue> Actor for ActorCritic<P, V> {
    fn action_space(&self) -> &ActionSpace {
        self.policy.action_space()
    }
    fn act_into(
        &self,
        obs: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        self.policy
            .valid()
            .distribution(obs)?
            .sample_actions_into(mode, rng, actions)
    }
}
impl<P: StochasticPolicy, V: StateValue> RolloutActor for ActorCritic<P, V> {
    fn sample_rollout_into(
        &self,
        obs: ObservationBatch<'_>,
        rng: &mut ChaCha8Rng,
        out: &mut RolloutBehavior,
    ) -> Result<()> {
        self.policy
            .valid()
            .distribution(obs)?
            .sample_behavior_into(ActMode::Sample, rng, out)?;
        self.values_into(obs, &mut out.values)
    }
    fn values_into(&self, obs: ObservationBatch<'_>, out: &mut Vec<f32>) -> Result<()> {
        let values = super::to_vec(self.value.valid().values(obs)?)?;
        out.clear();
        out.extend_from_slice(&values);
        Ok(())
    }
}
