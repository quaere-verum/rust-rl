use super::super::candidate_ddqn::{candidate_frame, candidate_sample};
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
            BuildCandidatePolicy, CandidatePolicy, PackLimits, StructuredPolicy,
            candidate_probabilities_checked,
        },
    },
    train::{behavior::BehaviorSchedule, counters::RunCounters, schedule::Schedule},
};
use burn::tensor::Device;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateMpoBehavior {
    pub epsilon: f64,
}

impl CandidateMpoBehavior {
    pub(in crate::algorithms) fn validate(self) -> Result<()> {
        ensure(
            self.epsilon.is_finite() && (0. ..=1.).contains(&self.epsilon),
            "invalid candidate MPO exploration",
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMpoSchedule(pub Schedule);

impl BehaviorSchedule for CandidateMpoSchedule {
    type Settings = CandidateMpoBehavior;
    fn settings(&self, counters: &RunCounters) -> Result<Self::Settings> {
        let settings = CandidateMpoBehavior {
            epsilon: self.0.value(counters)?,
        };
        settings.validate()?;
        Ok(settings)
    }
}

#[derive(Clone)]
pub struct CandidateMpoActor<P: CandidatePolicy = StructuredPolicy> {
    policy: Arc<P>,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: CandidateMpoBehavior,
    revision: u64,
    limits: PackLimits,
}

impl<P: CandidatePolicy> CandidateMpoActor<P> {
    pub(super) fn new(
        policy: P,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        settings: CandidateMpoBehavior,
        revision: u64,
        limits: PackLimits,
    ) -> Result<Self> {
        let actor = Self {
            policy: Arc::new(policy),
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
            "candidate MPO actor pack limit is zero",
        )?;
        P::validate_configuration(self.policy.configuration())?;
        self.policy
            .input_schema()
            .validate_spaces(&self.observation_space, &self.action_space)
    }

    /// Complete learned policy before any exploration mixture.
    pub fn action_probabilities(
        &self,
        observation: &Observation,
    ) -> Result<Vec<(ConditionalAction, f64)>> {
        Ok(candidate_frame(observation)?
            .candidates
            .iter()
            .zip(self.probabilities(observation)?)
            .filter(|(candidate, _)| candidate.enabled)
            .map(|(candidate, probability)| (candidate.action, probability))
            .collect())
    }

    fn probabilities(&self, observation: &Observation) -> Result<Vec<f64>> {
        let sample = candidate_sample(observation, self.policy.input_schema())?;
        Ok(candidate_probabilities_checked(
            self.policy.as_ref(),
            std::slice::from_ref(&sample),
            self.limits,
        )?
        .remove(0))
    }
}

impl<P: CandidatePolicy> StatefulActor for CandidateMpoActor<P> {
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
        let schema = self.policy.input_schema();
        if mode == ActMode::Deterministic {
            let sample = candidate_sample(observation, schema)?;
            let key = self.policy.greedy_action(&sample, self.limits)?;
            return Ok(StatefulDecision {
                action: Action::Conditional(Arc::new(key)),
                // The deployed behavior is a point mass. The learned law is
                // separately available through action_probabilities.
                log_prob: Some(0.),
            });
        }
        let joint = self.probabilities(observation)?;
        conditional_decision(
            &candidate_frame(observation)?.candidates,
            schema.operations.len(),
            joint,
            self.settings,
            rng,
        )
    }
}

pub(in crate::algorithms) fn conditional_decision(
    candidates: &[ActionCandidate],
    operation_count: usize,
    mut joint: Vec<f64>,
    settings: CandidateMpoBehavior,
    rng: &mut ChaCha8Rng,
) -> Result<StatefulDecision> {
    ensure(
        !joint.is_empty() && joint.len() == candidates.len(),
        "invalid conditional behavior row",
    )?;
    if settings.epsilon > 0. {
        let count = candidates
            .iter()
            .filter(|candidate| candidate.enabled)
            .count();
        for (p, candidate) in joint.iter_mut().zip(candidates) {
            if candidate.enabled {
                *p = (1. - settings.epsilon) * *p + settings.epsilon / count as f64;
            }
        }
    }
    let mut operations = vec![0.; operation_count];
    for (candidate, &p) in candidates.iter().zip(&joint) {
        operations[candidate.action.operation] += p;
    }
    let (operation, operation_probability) = select(&operations, rng);
    let conditional: Vec<_> = candidates
        .iter()
        .zip(&joint)
        .map(|(candidate, &p)| {
            if candidate.action.operation == operation {
                p
            } else {
                0.
            }
        })
        .collect();
    let (index, conditional_probability) = select(&conditional, rng);
    let probability = operation_probability * conditional_probability;
    Ok(StatefulDecision {
        action: Action::Conditional(Arc::new(candidates[index].action)),
        log_prob: Some(probability.ln().min(0.) as f32),
    })
}

fn select(weights: &[f64], rng: &mut ChaCha8Rng) -> (usize, f64) {
    let index = draw(weights, rng);
    (index, weights[index] / weights.iter().sum::<f64>())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ActorRecord<C> {
    version: u32,
    config: C,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    settings: CandidateMpoBehavior,
    revision: u64,
    max_pack_elements: usize,
    policy: LearnerRecord,
}

impl<P: CandidatePolicy> Serialize for CandidateMpoActor<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        ActorRecord {
            version: 1,
            config: self.policy.configuration(),
            observation_space: self.observation_space.clone(),
            action_space: self.action_space.clone(),
            settings: self.settings,
            revision: self.revision,
            max_pack_elements: self.limits.max_elements,
            policy: self
                .policy
                .checkpoint()
                .map_err(serde::ser::Error::custom)?,
        }
        .serialize(serializer)
    }
}

impl<P: BuildCandidatePolicy> CandidateMpoActor<P> {
    fn from_record(record: ActorRecord<P::Configuration>) -> Result<Self> {
        ensure(
            record.version == 1,
            "unsupported candidate MPO actor snapshot",
        )?;
        P::validate_configuration(&record.config)?;
        let mut policy = P::from_configuration(
            record.config.clone(),
            &mut ChaCha8Rng::seed_from_u64(0),
            &Device::flex(),
        )?;
        policy.restore(&record.policy)?;
        ensure(
            *policy.configuration() == record.config,
            "candidate MPO actor configuration differs from its saved model",
        )?;
        let actor = Self::new(
            policy.valid().freeze(),
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

impl<'de, P: BuildCandidatePolicy> Deserialize<'de> for CandidateMpoActor<P> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        Self::from_record(ActorRecord::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl<P: BuildCandidatePolicy> Snapshot for CandidateMpoActor<P> {
    fn validate_snapshot(&self) -> Result<()> {
        self.validate_config()?;
        parameter_norm(&*self.policy)?;
        Ok(())
    }
}

fn draw(p: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let mut threshold = rng.random::<f64>() * p.iter().sum::<f64>();
    for (i, &v) in p.iter().enumerate() {
        if v > 0. && threshold < v {
            return i;
        }
        threshold -= v;
    }
    p.iter()
        .rposition(|p| *p > 0.)
        .expect("nonempty validated distribution")
}
