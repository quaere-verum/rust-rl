//! Common public contracts and built-in components.
#[cfg(feature = "neural")]
pub use crate::algorithms::recurrent_ddqn::{
    RecurrentDdqn, RecurrentDdqnConfig, RecurrentUpdateMode,
};
#[cfg(feature = "neural")]
pub use crate::algorithms::{
    OptimizerConfig,
    d4pg::{D4pg, D4pgConfig},
    ddpg::{Ddpg, DdpgConfig},
    ddqn::{Ddqn, DdqnConfig},
    md4pg::{Md4pg, Md4pgConfig},
    mdpo::{Mdpo, MdpoConfig, MdpoWeighting, MirrorStepSchedule},
    ppo::{Ppo, PpoConfig, PpoUpdateSettings},
    reinforce::{Reinforce, ReinforceConfig},
    reppo::{CategoricalQ, Reppo, ReppoConfig, ReppoCritic},
    sac::{Sac, SacConfig},
    td3::{Td3, Td3Config},
    vmpo::{Vmpo, VmpoConfig},
};
pub use crate::collect::stateful::StatefulCollector;
pub use crate::collect::{
    vector_env::{SyncVectorEnv, VectorEnv},
    {CollectionDelta, OffPolicyCollector, OnPolicyCollector},
};
pub use crate::core::stateful::{
    BehaviorPublisher, HistoryInput, HistorySpec, PreviousInput, ReconstructionHistory,
    StatefulActor, StatefulDecision, StatefulLearner, StatelessActor, TemporalOffPolicyLearner,
};
pub use crate::core::{
    action::{Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction},
    actor::{
        ActMode, Actor, EpisodeActor, EpisodeBehavior, OnPolicyActorRef, PathwiseActor,
        PathwiseBehavior, RolloutActor, RolloutBehavior,
    },
    env::{
        ClipAction, Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace,
        PublicHistory, ResumeContext, Step, TimeLimit,
    },
    learner::{Learner, OffPolicyLearner, OnPolicyLearner, UpdateReport},
    reward::Reward,
    temporal::TemporalExecutionMode,
};
pub use crate::data::temporal::{EpisodeStore, SequenceBatch, SequenceConfig, SequenceSampler};
pub use crate::data::{
    replay::{NStepAccumulator, ReplayBatch, ReplayBuffer, ReplayWriter},
    rollout::{BootstrappedBatch, OnPolicyBatch, OnPolicyMetadata, PathwiseBatch, RolloutBuffer},
};
#[cfg(feature = "neural")]
pub use crate::nn::recurrent_actor::RecurrentActor;
#[cfg(feature = "neural")]
pub use crate::nn::temporal::{GruEncoder, RecurrentQ, TemporalEncoder, TemporalMasks};
#[cfg(feature = "neural")]
pub use crate::nn::{
    critics::scalar::{DiscreteQ, QValue, Value},
    gaussian_policy::GaussianPolicy,
    network::{Mlp, Network},
    policy::{
        ActorCritic, Deterministic, DeterministicPolicy, Policy, PolicyActor, StochasticPolicy,
    },
};
pub use crate::train::behavior::{BehaviorSchedule, FixedBehavior};
#[cfg(feature = "neural")]
pub use crate::train::behavior::{EpsilonGreedySchedule, ddqn_trainer};
pub use crate::train::recurrent::{TemporalProgress, TemporalTrainer};
pub use crate::train::temporal_config::TemporalOffPolicyConfig;
pub use crate::train::{
    checkpoint::{Checkpoint, Snapshot},
    config::{Budget, OffPolicyConfig, OnPolicyConfig},
    counters::RunCounters,
    schedule::{Counter, Schedule},
    trainer::{OffPolicyTrainer, OnPolicyTrainer, Progress},
};
pub use crate::{Error, Result};
