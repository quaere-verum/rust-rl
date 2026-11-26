//! Opt-in historical categorical distributions from the actual sampling call.
use super::{
    action::{Action, ActionSpace},
    ensure,
    env::Observation,
    stateful::{StatefulActor, StatefulDecision},
};
use crate::Result;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Immutable sampling-law identity. Include model, input contract, exploration
/// and sampler version; a publication revision alone is not a source identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BehaviorSource(pub [u8; 32]);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalBehavior {
    pub source: BehaviorSource,
    pub revision: u64,
    pub probabilities: Vec<f64>,
}
impl CategoricalBehavior {
    pub fn validate(
        &self,
        space: &ActionSpace,
        observation: &Observation,
        decision: &StatefulDecision,
    ) -> Result<()> {
        validate_probabilities(
            space,
            observation,
            &decision.action,
            decision.log_prob,
            &self.probabilities,
        )
    }
}

/// Validate borrowed historical mass without allocating a row during replay.
pub fn validate_probabilities(
    space: &ActionSpace,
    observation: &Observation,
    action: &Action,
    log_prob: Option<f32>,
    probabilities: &[f64],
) -> Result<()> {
    let ActionSpace::Discrete { count } = space else {
        return Err(crate::Error::Action(
            "categorical recording needs discrete actions".into(),
        ));
    };
    let mask = observation.action_mask();
    space.check(action, mask)?;
    ensure(
        probabilities.len() == *count,
        "categorical behavior width differs",
    )?;
    ensure(
        probabilities
            .iter()
            .enumerate()
            .all(|(a, p)| p.is_finite() && *p >= 0. && (mask.is_none_or(|m| m[a]) || *p == 0.))
            && (probabilities.iter().sum::<f64>() - 1.).abs() <= 1e-10,
        "invalid categorical behavior mass or support",
    )?;
    let Action::Discrete(a) = *action else {
        unreachable!()
    };
    ensure(
        probabilities[a] > 0.
            && log_prob.is_some_and(|p| p.to_bits() == (probabilities[a].ln() as f32).to_bits()),
        "recorded likelihood differs from the sampled distribution",
    )
}

pub struct CategoricalDecision {
    pub decision: StatefulDecision,
    pub behavior: CategoricalBehavior,
}

/// Companion capability: ordinary `StatefulActor` users do no extra work.
/// Implementations return probabilities from the same stochastic draw, never a
/// second forward pass or a reconstruction from a sampled-action likelihood.
pub trait CategoricalStatefulActor: StatefulActor {
    fn behavior_source(&self) -> BehaviorSource;
    fn sample_with_behavior(
        &self,
        state: &Self::State,
        observation: &Observation,
        rng: &mut ChaCha8Rng,
    ) -> Result<CategoricalDecision>;
}
