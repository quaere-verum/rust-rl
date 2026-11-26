//! Typed behavior scheduling, separate from learning and interaction.
use super::counters::RunCounters;
#[cfg(feature = "neural")]
use super::schedule::Schedule;
use crate::Result;
use serde::{Deserialize, Serialize};

/// Settings queries are deterministic and side-effect-free. Random inference is
/// owned by the actor's lane RNG, not by this publication schedule.
pub trait BehaviorSchedule: Clone {
    type Settings: Clone + PartialEq;
    fn settings(&self, counters: &RunCounters) -> Result<Self::Settings>;
    fn validate(&self) -> Result<()> {
        self.settings(&RunCounters::default())?;
        self.settings(&RunCounters {
            env_steps: u64::MAX,
            vector_steps: u64::MAX,
            episodes: u64::MAX,
            updates: u64::MAX,
            additional_physical_steps: 0,
        })?;
        Ok(())
    }
}

/// Constant typed settings also support planners, stochastic policies and other
/// consumers without an epsilon or a generic substitute scalar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedBehavior<C>(pub C);
impl<C: Clone + PartialEq> BehaviorSchedule for FixedBehavior<C> {
    type Settings = C;
    fn settings(&self, _: &RunCounters) -> Result<C> {
        Ok(self.0.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "neural")]
pub struct EpsilonGreedySchedule(pub Schedule);
#[cfg(feature = "neural")]
impl BehaviorSchedule for EpsilonGreedySchedule {
    type Settings = crate::algorithms::recurrent_ddqn::EpsilonGreedySettings;
    fn settings(&self, counters: &RunCounters) -> Result<Self::Settings> {
        let value = self
            .0
            .probability(counters, "invalid epsilon-greedy behavior schedule")?;
        Ok(Self::Settings {
            epsilon: value as f32,
        })
    }
}

/// A stochastic categorical policy mixed with masked-uniform exploration.
/// Unlike epsilon-greedy control, the unperturbed component remains stochastic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg(feature = "neural")]
pub struct CategoricalMixtureSchedule(pub Schedule);
#[cfg(feature = "neural")]
impl BehaviorSchedule for CategoricalMixtureSchedule {
    type Settings = crate::nn::temporal_categorical_actor::CategoricalMixtureSettings;
    fn settings(&self, counters: &RunCounters) -> Result<Self::Settings> {
        let epsilon = self
            .0
            .probability(counters, "invalid categorical mixture schedule")?;
        Ok(Self::Settings { epsilon })
    }
}

/// Convenience composition for existing DDQN settings. The generic trainer
/// never interprets exploration settings or dispatches on learner identity.
#[cfg(feature = "neural")]
pub fn ddqn_trainer<V: crate::collect::vector_env::VectorEnv>(
    env: V,
    learner: &crate::algorithms::recurrent_ddqn::RecurrentDdqn,
    config: super::config::OffPolicyConfig,
    sequence: crate::data::temporal::SequenceConfig,
    seed: u64,
) -> Result<
    super::recurrent::TemporalTrainer<
        V,
        crate::nn::recurrent_actor::RecurrentActor,
        EpsilonGreedySchedule,
    >,
> {
    use crate::core::stateful::{BehaviorPublisher, TemporalOffPolicyLearner};
    let temporal = super::temporal_config::TemporalOffPolicyConfig::from(&config);
    temporal.returns().check(learner.return_contract())?;
    let behavior = EpsilonGreedySchedule(config.exploration);
    behavior.validate()?;
    let settings = behavior.settings(&RunCounters::default())?;
    let actor = learner.behavior_actor(&settings, 0)?;
    super::recurrent::TemporalTrainer::new(env, actor, temporal, sequence, behavior, seed)
}
