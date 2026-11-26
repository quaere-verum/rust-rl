//! A stochastic recurrent actor with an explicit masked-uniform exploration mixture.
use crate::{
    Result,
    core::{
        action::{Action, ActionSpace},
        actor::ActMode,
        categorical_behavior::{
            BehaviorSource, CategoricalBehavior, CategoricalDecision, CategoricalStatefulActor,
        },
        checkpoint::Snapshot,
        ensure,
        env::{Observation, ObservationBatch, ObservationSpace},
        stateful::{HistoryInput, StatefulActor, StatefulDecision},
    },
    nn::{
        temporal_actor::{TemporalActor, TemporalActorState},
        temporal_policy::TemporalPolicy,
        to_vec,
    },
};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Probability of masked-uniform exploration mixed with the stochastic policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CategoricalMixtureSettings {
    pub epsilon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCategoricalActor {
    actor: TemporalActor,
    epsilon: f64,
}
impl TemporalCategoricalActor {
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
    /// Frozen policy identity, excluding publication revision and exploration.
    pub fn model_identity(&self) -> [u8; 32] {
        self.actor.model_identity()
    }

    pub fn new(policy: TemporalPolicy, epsilon: f64, revision: u64) -> Result<Self> {
        let actor = Self {
            actor: TemporalActor::new(policy, None, revision)?,
            epsilon,
        };
        actor.validate_snapshot()?;
        Ok(actor)
    }
    pub fn probabilities(
        &self,
        state: &TemporalActorState,
        observation: &Observation,
    ) -> Result<Vec<f64>> {
        ensure(
            self.epsilon.is_finite() && (0. ..=1.).contains(&self.epsilon),
            "invalid categorical exploration probability",
        )?;
        self.validate_state(state)?;
        observation.validate(self.actor.spec().observation_dim, self.action_space(), true)?;
        let policy = self.actor.policy();
        let features = burn::tensor::Tensor::from_data(
            burn::tensor::TensorData::new(state.policy.clone(), [1, state.policy.len()]),
            &policy.device(),
        );
        let distribution = policy
            .distribution_from_features(features, ObservationBatch::from_refs(&[observation]))?;
        let super::distributions::Distribution::Categorical { logits, masks } = distribution else {
            return Err(crate::Error::Action(
                "exploration mixture requires discrete policy".into(),
            ));
        };
        let logits = to_vec(logits)?;
        ensure(
            logits.iter().all(|p| p.is_finite()),
            "nonfinite categorical logits",
        )?;
        let maximum = logits
            .iter()
            .zip(&masks)
            .filter(|(_, m)| **m)
            .map(|(p, _)| *p as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut p = logits
            .iter()
            .zip(&masks)
            .map(|(l, m)| if *m { (*l as f64 - maximum).exp() } else { 0. })
            .collect::<Vec<_>>();
        let total = p.iter().sum::<f64>();
        let count = masks.iter().filter(|m| **m).count() as f64;
        for (p, m) in p.iter_mut().zip(&masks) {
            *p = if *m {
                (1. - self.epsilon) * (*p / total) + self.epsilon / count
            } else {
                0.
            };
        }
        // Normalize host arithmetic once; this exact vector defines sampling,
        // likelihood recording and subsequent off-policy provenance checks.
        let sum = p.iter().sum::<f64>();
        for p in &mut p {
            *p /= sum;
        }
        Ok(p)
    }
}
impl Snapshot for TemporalCategoricalActor {
    fn validate_snapshot(&self) -> Result<()> {
        self.actor.validate_snapshot()?;
        ensure(
            matches!(self.actor.action_space(), ActionSpace::Discrete { .. })
                && self.epsilon.is_finite()
                && (0. ..=1.).contains(&self.epsilon),
            "invalid discrete exploration mixture",
        )
    }
}
impl StatefulActor for TemporalCategoricalActor {
    type State = TemporalActorState;
    fn observation_space(&self) -> ObservationSpace {
        self.actor.observation_space()
    }
    fn action_space(&self) -> &ActionSpace {
        self.actor.action_space()
    }
    fn revision(&self) -> u64 {
        self.actor.revision()
    }
    fn initial_state(&self) -> Result<Self::State> {
        self.actor.initial_state()
    }
    fn validate_state(&self, state: &Self::State) -> Result<()> {
        self.actor.validate_state(state)
    }
    fn assimilate(&self, prior: &Self::State, input: HistoryInput<'_>) -> Result<Self::State> {
        self.actor.assimilate(prior, input)
    }
    fn assimilate_canonical_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        self.actor.assimilate_canonical_batch(priors, inputs)
    }
    fn assimilate_batch(
        &self,
        priors: &[&Self::State],
        inputs: &[HistoryInput<'_>],
    ) -> Result<Vec<Self::State>> {
        self.actor.assimilate_batch(priors, inputs)
    }
    fn batch_compatible(&self, other: &Self) -> bool {
        self.epsilon.to_bits() == other.epsilon.to_bits()
            && self.actor.batch_compatible(&other.actor)
    }
    fn decide(
        &self,
        state: &Self::State,
        o: &Observation,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
    ) -> Result<StatefulDecision> {
        let p = self.probabilities(state, o)?;
        let action = if mode == ActMode::Sample {
            sample_index(&p, rng)
        } else {
            let mut action = p.iter().position(|p| *p > 0.).unwrap();
            for (a, &probability) in p.iter().enumerate() {
                if probability > p[action] {
                    action = a;
                }
            }
            action
        };
        Ok(StatefulDecision {
            action: Action::Discrete(action),
            log_prob: Some(p[action].ln() as f32),
        })
    }
}

// The historical sampler arithmetic is shared unchanged by ordinary and rich
// recording. Source hashing happens only on the opt-in path and hashes the
// cached model identity, not all network weights on every decision.
fn sample_index(p: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let mut action = p.iter().position(|p| *p > 0.).unwrap();
    let u = rng.random::<f64>();
    let mut sum = 0.;
    for (a, &probability) in p.iter().enumerate() {
        if probability > 0. {
            action = a;
            sum += probability;
            if u < sum {
                break;
            }
        }
    }
    action
}
impl CategoricalStatefulActor for TemporalCategoricalActor {
    fn behavior_source(&self) -> BehaviorSource {
        let mut hash = Sha256::new();
        hash.update(b"rust-rl/categorical-mixture/f64-v1");
        hash.update(self.model_identity());
        hash.update(self.epsilon.to_bits().to_le_bytes());
        BehaviorSource(hash.finalize().into())
    }
    fn sample_with_behavior(
        &self,
        state: &Self::State,
        observation: &Observation,
        rng: &mut ChaCha8Rng,
    ) -> Result<CategoricalDecision> {
        let probabilities = self.probabilities(state, observation)?;
        let action = sample_index(&probabilities, rng);
        Ok(CategoricalDecision {
            decision: StatefulDecision {
                action: Action::Discrete(action),
                log_prob: Some(probabilities[action].ln() as f32),
            },
            behavior: CategoricalBehavior {
                source: self.behavior_source(),
                revision: self.revision(),
                probabilities,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::stateful::HistorySpec, nn::vector};
    use burn::module::Param;
    use rand::SeedableRng;
    #[test]
    fn mixture_records_actual_masked_distribution_and_resumes_its_samples() {
        let spec = HistorySpec::new(1, ActionSpace::Discrete { count: 3 }).unwrap();
        let mut policy =
            TemporalPolicy::new(spec, Some(4), &[], &mut ChaCha8Rng::seed_from_u64(28)).unwrap();
        policy.head.layers[0].weight = Param::from_tensor(crate::nn::matrix(vec![0.; 12], 4, 3));
        policy.head.layers[0].bias = Some(Param::from_tensor(vector(&[0., 3f32.ln(), 99.])));
        let actor = TemporalCategoricalActor::new(policy, 0.2, 7).unwrap();
        let o = Observation::dense(vec![1.], Some(vec![true, true, false]));
        let s = actor
            .assimilate(
                &actor.initial_state().unwrap(),
                HistoryInput {
                    observation: &o,
                    previous: None,
                },
            )
            .unwrap();
        let p = actor.probabilities(&s, &o).unwrap();
        assert!((p[0] - 0.3).abs() < 1e-7 && (p[1] - 0.7).abs() < 1e-7 && p[2] == 0.);
        let copy: TemporalCategoricalActor =
            serde_json::from_slice(&serde_json::to_vec(&actor).unwrap()).unwrap();
        let mut x = ChaCha8Rng::seed_from_u64(182);
        let mut y = x.clone();
        let mut count = 0;
        for _ in 0..10000 {
            let a = actor.decide(&s, &o, ActMode::Sample, &mut x).unwrap();
            let b = copy.decide(&s, &o, ActMode::Sample, &mut y).unwrap();
            assert_eq!(a.action, b.action);
            assert_eq!(a.log_prob, b.log_prob);
            let Action::Discrete(index) = a.action else {
                unreachable!()
            };
            assert!(index < 2);
            assert_eq!(a.log_prob, Some(p[index].ln() as f32));
            count += usize::from(index == 0);
        }
        assert!((count as f64 / 10000. - 0.3).abs() < 0.025);
        assert_eq!(
            actor
                .decide(&s, &o, ActMode::Deterministic, &mut x)
                .unwrap()
                .action,
            Action::Discrete(1)
        );
        let mut invalid = actor.clone();
        invalid.epsilon = f64::NAN;
        assert!(invalid.validate_snapshot().is_err());
    }
}
