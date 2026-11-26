//! All discrete action values from an independently reconstructed public history.
use super::{
    network::{Mlp, Network},
    temporal::TemporalEncoder,
    temporal_models::{self, TemporalBackbone, TemporalModelArchitecture},
};
use crate::{
    Result,
    core::{action::ActionSpace, ensure, stateful::HistorySpec},
    data::temporal::SequenceBatch,
};
use burn::{
    module::Module,
    tensor::{Device, Tensor},
};
use rand_chacha::ChaCha8Rng;

#[derive(Module, Debug)]
pub struct TemporalDiscreteQ<E: TemporalEncoder = TemporalBackbone, H: Network = Mlp> {
    pub encoder: E,
    pub head: H,
    #[module(skip)]
    spec: HistorySpec,
    #[module(skip)]
    observation_only: bool,
}
impl TemporalDiscreteQ {
    pub fn with_initialization(
        mut self,
        initialization: super::critics::CriticInitialization,
    ) -> Result<Self> {
        initialization.apply(&mut self.head)?;
        Ok(self)
    }

    pub fn new(
        spec: HistorySpec,
        hidden: Option<usize>,
        head_hidden: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        spec.validate()?;
        let ActionSpace::Discrete { count } = spec.action_space else {
            return Err(crate::Error::Action(
                "temporal discrete Q requires discrete actions".into(),
            ));
        };
        let encoder = TemporalBackbone::for_history(&spec, hidden, rng, false)?;
        let head = Mlp::new(encoder.output_dim(), head_hidden, count, rng)?;
        Self::from_parts(encoder, head, spec, hidden.is_none())
    }
    pub fn architecture(&self) -> Result<TemporalModelArchitecture> {
        let a = temporal_models::architecture(
            &self.encoder,
            &self.head,
            &self.spec,
            self.observation_only,
        )?;
        ensure(
            a.output_dim == self.spec.action_width(),
            "temporal discrete Q action width differs",
        )?;
        Ok(a)
    }
    pub fn from_architecture(a: &TemporalModelArchitecture, rng: &mut ChaCha8Rng) -> Result<Self> {
        a.validate()?;
        ensure(
            a.version == 1
                && a.head_input_extra == 0
                && matches!(a.spec.action_space, ActionSpace::Discrete { .. })
                && a.output_dim == a.spec.action_width(),
            "invalid all-action temporal Q architecture",
        )?;
        let (encoder, head) = a.parts(rng, false)?;
        Self::from_parts(encoder, head, a.spec.clone(), a.hidden_dim.is_none())
    }
}
impl<E: TemporalEncoder, H: Network> TemporalDiscreteQ<E, H> {
    pub fn from_parts(
        encoder: E,
        head: H,
        spec: HistorySpec,
        observation_only: bool,
    ) -> Result<Self> {
        spec.validate()?;
        ensure(
            matches!(spec.action_space, ActionSpace::Discrete { .. })
                && encoder.input_dim()
                    == if observation_only {
                        spec.observation_dim
                    } else {
                        spec.input_dim()
                    }
                && head.input_dim() == encoder.output_dim()
                && head.output_dim() == spec.action_width(),
            "invalid all-action temporal Q encoder/head/history shape",
        )?;
        Ok(Self {
            encoder,
            head,
            spec,
            observation_only,
        })
    }
    pub fn spec(&self) -> &HistorySpec {
        &self.spec
    }
    pub fn device(&self) -> Device {
        temporal_models::module_device(self)
    }
    pub fn values(&self, batch: &SequenceBatch, tokens: &[(usize, usize)]) -> Result<Tensor<2>> {
        let features = temporal_models::selected_features(
            &self.encoder,
            batch,
            &self.spec,
            self.observation_only,
            &self.device(),
            tokens,
        )?;
        Ok(self.head.forward(features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::{
            ddqn::DdqnConfig,
            recurrent_ddqn::{RecurrentDdqn, RecurrentDdqnConfig},
        },
        core::{
            action::{Action, ActionSpace},
            env::Observation,
            temporal::BehaviorMetadata,
        },
        data::temporal::{SequenceConfig, SequenceOrigin, SequenceSample, TemporalStep},
        nn::{temporal::RecurrentQ, temporal_policy::TemporalPolicy, to_vec},
    };
    use rand::SeedableRng;
    #[test]
    fn all_action_history_values_match_native_gru_q_with_identical_weights_and_burn() {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 2 }).unwrap();
        let model = TemporalDiscreteQ::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(561),
        )
        .unwrap();
        let old = TemporalPolicy::new(
            spec.clone(),
            Some(4),
            &[],
            &mut ChaCha8Rng::seed_from_u64(561),
        )
        .unwrap();
        let native = RecurrentDdqn::new(
            RecurrentQ::new(old.encoder.gru.unwrap(), old.head).unwrap(),
            spec,
            RecurrentDdqnConfig {
                ddqn: DdqnConfig::default(),
                gamma: 1.,
                n_step: 1,
            },
        )
        .unwrap();
        let o = |x| Observation::dense(vec![x], Some(vec![true; 2]));
        let b = SequenceBatch {
            config: SequenceConfig {
                burn_in: 1,
                learn_len: 1,
                n_step: 1,
            },
            samples: vec![SequenceSample {
                episode_id: 1,
                trajectory_start: 0,
                learn_start: 1,
                burn_start: 0,
                history_start: 0,
                previous: None,
                origin: SequenceOrigin::Reset,
                observations: vec![o(1.), o(-1.), o(2.)],
                steps: vec![
                    TemporalStep {
                        action: Action::Discrete(1),
                        reward: 0.25_f32.into(),
                        terminated: false,
                        truncated: false,
                        behavior: BehaviorMetadata::default(),
                    },
                    TemporalStep {
                        action: Action::Discrete(0),
                        reward: 1.0_f32.into(),
                        terminated: false,
                        truncated: true,
                        behavior: BehaviorMetadata::default(),
                    },
                ],
                loss_mask: vec![false, true],
            }],
        };
        let (q, _) = native.reconstructed_q_values(&b).unwrap();
        let actual = to_vec(model.values(&b, &[(0, 2), (0, 1), (0, 0)]).unwrap()).unwrap();
        for (x, y) in actual.iter().zip([q[4], q[5], q[2], q[3], q[0], q[1]]) {
            assert!((x - y).abs() < 1e-6);
        }
        assert_eq!(
            crate::nn::optimization::parameter_norm(&model).unwrap(),
            crate::nn::optimization::parameter_norm(&native.online()).unwrap()
        );
    }
}
