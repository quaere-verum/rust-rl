use super::CriticInitialization;
use crate::nn::{
    network::{Mlp, Network},
    observations, observations_on_device,
};
use crate::{
    Result,
    core::{ensure, env::ObservationBatch},
};
use burn::{module::Module, tensor::Tensor};
use rand_chacha::ChaCha8Rng;

pub trait StateValue: Module {
    fn values(&self, obs: ObservationBatch<'_>) -> Result<Tensor<1>>;
}
#[derive(Module, Debug)]
pub struct Value<M: Network = Mlp> {
    pub network: M,
}
impl Value<Mlp> {
    pub fn new(width: usize, hidden: &[usize], rng: &mut ChaCha8Rng) -> Result<Self> {
        Ok(Self {
            network: Mlp::new(width, hidden, 1, rng)?,
        })
    }
    pub fn with_initialization(mut self, initialization: CriticInitialization) -> Result<Self> {
        initialization.apply(&mut self.network)?;
        Ok(self)
    }
}
impl<M: Network> StateValue for Value<M> {
    fn values(&self, obs: ObservationBatch<'_>) -> Result<Tensor<1>> {
        ensure(
            self.network.output_dim() == 1,
            "state value output must be scalar",
        )?;
        Ok(self
            .network
            .forward(observations(obs, self.network.input_dim())?)
            .squeeze_dim(1))
    }
}

pub trait ActionValue: Module {
    fn action_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>>;
    /// Update target parameters while preserving any declared input coordinates.
    #[doc(hidden)]
    fn updated_target(&self, online: &Self, tau: f32) -> Result<Self> {
        crate::nn::target::polyak_update(self.clone(), online, tau)
    }
    /// Fixed coordinate metadata to validate alongside the recorded parameters.
    #[doc(hidden)]
    fn checkpoint_coordinates(&self) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }
    /// Custom critics retain actor inference side effects unless they opt in.
    fn actor_second_forward_skippable(&self) -> bool {
        false
    }
}
#[derive(Module, Debug)]
pub struct QValue<M: Network = Mlp> {
    pub network: M,
    pub observation_dim: usize,
    pub action_dim: usize,
}
impl QValue<Mlp> {
    pub fn new(
        width: usize,
        actions: usize,
        outputs: usize,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        ensure(actions > 0, "action dimension must be positive")?;
        Ok(Self {
            network: Mlp::new(width + actions, hidden, outputs, rng)?,
            observation_dim: width,
            action_dim: actions,
        })
    }
    /// Configure the scalar output before constructing the learner and its target.
    pub fn with_initialization(mut self, initialization: CriticInitialization) -> Result<Self> {
        ensure(
            initialization == CriticInitialization::Existing || self.network.output_dim() == 1,
            "constant scalar initialization requires a single critic output",
        )?;
        initialization.apply(&mut self.network)?;
        Ok(self)
    }
}
impl<M: Network> ActionValue for QValue<M> {
    fn actor_second_forward_skippable(&self) -> bool {
        self.network.actor_second_forward_skippable()
    }
    fn action_dim(&self) -> usize {
        self.action_dim
    }
    fn output_dim(&self) -> usize {
        self.network.output_dim()
    }
    fn values(&self, obs: ObservationBatch<'_>, actions: Tensor<2>) -> Result<Tensor<2>> {
        ensure(
            actions.dims() == [obs.len(), self.action_dim]
                && self.network.input_dim() == self.observation_dim + self.action_dim,
            "critic dimensions mismatch",
        )?;
        Ok(self.network.forward(Tensor::cat(
            // Constants follow the action operand. Autodiff actions retain their
            // actor graph; plain inspection actions need no autodiff cat step.
            vec![
                observations_on_device(obs, self.observation_dim, &actions.device())?,
                actions,
            ],
            1,
        )))
    }
}

#[cfg(test)]
mod actor_capability_tests {
    use super::*;
    use rand::SeedableRng;

    #[derive(Module, Debug)]
    struct WrappedMlp {
        inner: Mlp,
    }
    impl Network for WrappedMlp {
        fn input_dim(&self) -> usize {
            self.inner.input_dim()
        }
        fn output_dim(&self) -> usize {
            self.inner.output_dim()
        }
        fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            self.inner.forward(input)
        }
    }

    #[test]
    fn only_opted_in_network_allows_qvalue_second_actor_forward_omission() {
        let mut rng = ChaCha8Rng::seed_from_u64(91);
        let built_in = QValue::new(1, 1, 1, &[], &mut rng).unwrap();
        assert!(built_in.actor_second_forward_skippable());
        let custom = QValue {
            network: WrappedMlp {
                inner: Mlp::new(2, &[], 1, &mut rng).unwrap(),
            },
            observation_dim: 1,
            action_dim: 1,
        };
        assert!(!custom.actor_second_forward_skippable());
    }
}
pub trait DiscreteActionValue: Module {
    fn action_values(&self, obs: ObservationBatch<'_>) -> Result<Tensor<2>>;
    fn action_count(&self) -> usize;
}
#[derive(Module, Debug)]
pub struct DiscreteQ<M: Network = Mlp> {
    pub network: M,
}
impl DiscreteQ<Mlp> {
    pub fn new(
        width: usize,
        actions: usize,
        hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        Ok(Self {
            network: Mlp::new(width, hidden, actions, rng)?,
        })
    }
    pub fn with_initialization(mut self, initialization: CriticInitialization) -> Result<Self> {
        initialization.apply(&mut self.network)?;
        Ok(self)
    }
}
impl<M: Network> DiscreteActionValue for DiscreteQ<M> {
    fn action_count(&self) -> usize {
        self.network.output_dim()
    }
    fn action_values(&self, obs: ObservationBatch<'_>) -> Result<Tensor<2>> {
        Ok(self
            .network
            .forward(observations(obs, self.network.input_dim())?))
    }
}
