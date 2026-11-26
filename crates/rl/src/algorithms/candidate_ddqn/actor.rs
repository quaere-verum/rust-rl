use super::{model_frame, model_sample};
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::{LearnerRecord, Snapshot},
        ensure,
        env::{Observation, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
        structured::ActionCandidate,
    },
    nn::{
        optimization::parameter_norm,
        structured::{
            BuildStructuredActionValue, PackLimits, StructuredActionValue, StructuredQ,
            evaluate_checked,
        },
        to_vec,
    },
    train::{behavior::BehaviorSchedule, counters::RunCounters, schedule::Schedule},
};
use burn::tensor::Device;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationDistribution {
    #[default]
    CompleteActions,
    Operations,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateExploration {
    pub epsilon: f64,
    pub distribution: ExplorationDistribution,
}

impl Default for CandidateExploration {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            distribution: ExplorationDistribution::CompleteActions,
        }
    }
}

impl CandidateExploration {
    pub(in crate::algorithms) fn validate(self) -> Result<()> {
        ensure(
            self.epsilon.is_finite() && (0. ..=1.).contains(&self.epsilon),
            "invalid candidate epsilon-greedy settings",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateEpsilonSchedule {
    pub epsilon: Schedule,
    pub distribution: ExplorationDistribution,
}

impl BehaviorSchedule for CandidateEpsilonSchedule {
    type Settings = CandidateExploration;

    fn settings(&self, counters: &RunCounters) -> Result<Self::Settings> {
        let settings = CandidateExploration {
            epsilon: self.epsilon.value(counters)?,
            distribution: self.distribution,
        };
        settings.validate()?;
        Ok(settings)
    }
}

/// Immutable, stateless behavior published to the shared temporal collector.
/// Like other temporal actors, each lane retains its publication until reset.
#[derive(Clone)]
pub struct CandidateQActor<Q: StructuredActionValue = StructuredQ> {
    model: Arc<Q>,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: CandidateExploration,
    revision: u64,
    limits: PackLimits,
}

#[cfg(feature = "diagnostics")]
pub(crate) struct CandidateQPrediction {
    pub action: Action,
    pub candidates: Vec<(ConditionalAction, f32, f64)>,
}

impl<Q: StructuredActionValue> CandidateQActor<Q> {
    pub(super) fn new(
        model: Q,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        settings: CandidateExploration,
        revision: u64,
        limits: PackLimits,
    ) -> Result<Self> {
        let actor = Self {
            model: Arc::new(model),
            observation_space,
            action_space,
            settings,
            revision,
            limits,
        };
        actor.validate_config()?;
        Ok(actor)
    }

    fn validate_config(&self) -> Result<()> {
        self.settings.validate()?;
        ensure(
            self.limits.max_elements > 0,
            "candidate actor pack limit is zero",
        )?;
        Q::validate_configuration(self.model.configuration())?;
        self.model
            .validate_spaces(&self.observation_space, &self.action_space)
    }

    /// Read the frozen Q scores for enabled semantic keys without sampling.
    pub fn action_values(
        &self,
        observation: &Observation,
    ) -> Result<Vec<(ConditionalAction, f32)>> {
        Ok(model_frame(observation)?
            .candidates
            .iter()
            .zip(self.values(observation)?)
            .filter(|(candidate, _)| candidate.enabled)
            .map(|(candidate, value)| (candidate.action, value))
            .collect())
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_prediction(
        &self,
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<CandidateQPrediction> {
        let candidates = &model_frame(observation)?.candidates;
        let values = self.values(observation)?;
        let probabilities = mode_probabilities(candidates, &values, self.settings, mode)?;
        let decision = sample_behavior(candidates, &probabilities, rng);
        Ok(CandidateQPrediction {
            action: decision.action,
            candidates: candidates
                .iter()
                .zip(values.into_iter().zip(probabilities))
                .filter(|(candidate, _)| candidate.enabled)
                .map(|(candidate, (value, probability))| (candidate.action, value, probability))
                .collect(),
        })
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn diagnostic_global(
        &self,
        observation: &Observation,
        max_width: usize,
    ) -> Result<Option<Vec<f32>>> {
        observation.validate_in(&self.observation_space, &self.action_space, true)?;
        self.model
            .diagnostic_global(observation, self.limits, max_width)
    }

    fn values(&self, observation: &Observation) -> Result<Vec<f32>> {
        let sample = model_sample(observation, self.model.as_ref())?;
        let rows = evaluate_checked(
            self.model.as_ref(),
            std::slice::from_ref(&sample),
            observation.as_graph().as_slice(),
            self.limits,
        )?;
        to_vec(
            rows.into_iter()
                .next()
                .expect("validated single row")
                .values,
        )
    }
}

impl<Q: StructuredActionValue> StatefulActor for CandidateQActor<Q> {
    type State = ();

    fn observation_space(&self) -> ObservationSpace {
        self.observation_space.clone()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }
    fn revision(&self) -> u64 {
        self.revision
    }
    fn initial_state(&self) -> Result<()> {
        Ok(())
    }
    fn validate_state(&self, _: &()) -> Result<()> {
        Ok(())
    }

    fn assimilate(&self, _: &(), input: HistoryInput<'_>) -> Result<()> {
        input
            .observation
            .validate_in(&self.observation_space, &self.action_space, false)
    }

    fn decide(
        &self,
        _: &(),
        observation: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let values = self.values(observation)?;
        epsilon_decision(
            &model_frame(observation)?.candidates,
            &values,
            self.settings,
            mode,
            rng,
        )
    }
}

pub(in crate::algorithms) fn epsilon_decision(
    candidates: &[ActionCandidate],
    values: &[f32],
    settings: CandidateExploration,
    mode: ActMode,
    rng: &mut ChaCha8Rng,
) -> Result<StatefulDecision> {
    let probabilities = mode_probabilities(candidates, values, settings, mode)?;
    Ok(sample_behavior(candidates, &probabilities, rng))
}

fn mode_probabilities(
    candidates: &[ActionCandidate],
    values: &[f32],
    mut settings: CandidateExploration,
    mode: ActMode,
) -> Result<Vec<f64>> {
    if mode == ActMode::Deterministic {
        settings.epsilon = 0.;
    }
    behavior_probabilities(candidates, values, settings)
}

fn sample_behavior(
    candidates: &[ActionCandidate],
    probabilities: &[f64],
    rng: &mut ChaCha8Rng,
) -> StatefulDecision {
    // Greedy deployment integrates the same uniform tie law used in the backup.
    let draw = rng.random::<f64>();
    let mut cumulative = 0.;
    let mut selected = None;
    for (index, &p) in probabilities.iter().enumerate() {
        if p > 0. {
            cumulative += p;
            selected = Some(index);
            if draw < cumulative {
                break;
            }
        }
    }
    let index = selected.expect("validated live distribution");
    StatefulDecision {
        action: Action::Conditional(Arc::new(candidates[index].action)),
        log_prob: Some(probabilities[index].ln() as f32),
    }
}

fn behavior_probabilities(
    candidates: &[ActionCandidate],
    values: &[f32],
    settings: CandidateExploration,
) -> Result<Vec<f64>> {
    ensure(
        values.len() == candidates.len() && values.iter().all(|value| value.is_finite()),
        "invalid candidate behavior values",
    )?;
    let enabled = candidates
        .iter()
        .zip(values)
        .filter(|(candidate, _)| candidate.enabled);
    let maximum = enabled
        .clone()
        .map(|(_, &value)| value)
        .max_by(f32::total_cmp)
        .ok_or_else(|| crate::Error::Action("no enabled behavior candidate".into()))?;
    let enabled_count = enabled.clone().count();
    let ties = enabled
        .clone()
        .filter(|(_, value)| **value == maximum)
        .count();
    let mut operations = HashMap::new();
    if settings.distribution == ExplorationDistribution::Operations {
        for (candidate, _) in enabled {
            *operations
                .entry(candidate.action.operation)
                .or_insert(0usize) += 1;
        }
    }
    Ok(candidates
        .iter()
        .zip(values)
        .map(|(candidate, &value)| {
            if !candidate.enabled {
                return 0.;
            }
            let exploration = match settings.distribution {
                ExplorationDistribution::CompleteActions => settings.epsilon / enabled_count as f64,
                ExplorationDistribution::Operations => {
                    settings.epsilon
                        / (operations.len() as f64 * operations[&candidate.action.operation] as f64)
                }
            };
            exploration
                + if value == maximum {
                    (1. - settings.epsilon) / ties as f64
                } else {
                    0.
                }
        })
        .collect())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord<C> {
    version: u32,
    configuration: C,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: CandidateExploration,
    revision: u64,
    max_pack_elements: usize,
    model: LearnerRecord,
}

impl<Q: StructuredActionValue> Serialize for CandidateQActor<Q> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let model = self.model.checkpoint().map_err(serde::ser::Error::custom)?;
        ActorRecord {
            version: 1,
            configuration: self.model.configuration(),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
            settings: self.settings,
            revision: self.revision,
            max_pack_elements: self.limits.max_elements,
            model,
        }
        .serialize(serializer)
    }
}

impl<Q: BuildStructuredActionValue> CandidateQActor<Q> {
    fn from_record(record: ActorRecord<Q::Configuration>) -> Result<Self> {
        ensure(
            record.version == 1,
            "unsupported candidate actor snapshot version",
        )?;
        Q::validate_configuration(&record.configuration)?;
        let mut model = Q::from_configuration(
            record.configuration.clone(),
            &mut ChaCha8Rng::seed_from_u64(0),
            &Device::flex(),
        )?;
        model.restore(&record.model)?;
        ensure(
            *model.configuration() == record.configuration,
            "candidate actor model configuration differs",
        )?;
        let actor = Self::new(
            model.valid().freeze(),
            record.observation_space,
            record.action_space,
            record.settings,
            record.revision,
            PackLimits {
                max_elements: record.max_pack_elements,
            },
        )?;
        actor.validate_snapshot()?;
        Ok(actor)
    }
}

impl<'de, Q: BuildStructuredActionValue> Deserialize<'de> for CandidateQActor<Q> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(ActorRecord::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl<Q: BuildStructuredActionValue> Snapshot for CandidateQActor<Q> {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_config()?;
        parameter_norm(self.model.as_ref())?;
        Ok(())
    }
}
