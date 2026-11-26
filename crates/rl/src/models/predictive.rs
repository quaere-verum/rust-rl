//! Public predictive states and coherent hypothetical trajectories.
//!
//! `Context` is the caller's public observation/history or belief summary. It
//! may include known physical costs, resources and a partially known map; it
//! must not expose the factual simulator's hidden state. Belief updates are
//! model-owned and returned as the next public context. Finite exact beliefs
//! and approximate filters can use the same contract.
//!
//! `Hidden` is one hypothetical root particle carried and updated along a
//! simulated trajectory. It is not resampled at each transition or handed to
//! policy/value guidance. A deployed model needs only the public context and
//! its own predictive law, not a privileged simulator snapshot.
//!
//! Transitions here are predictions, never factual replay records. Observation
//! keys distinguish all public outcomes relevant to future choices. `next=None`
//! denotes a terminal prediction; reaching a planner's horizon is a separate
//! computation cutoff and must not be encoded as physical termination.
//!
//! Reward is f64 and already discounted *within* the action using the model's
//! declared physical-time gamma. A planner discounts continuation by
//! `gamma.powi(duration)`. Costs/resources are represented in the public context
//! and outcome key when they constrain continuation. Domain utility must be
//! identified separately from physical environment rewards. Exact and guided
//! planners validate the declared convention at their boundary.
use crate::{Result, core::ensure};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ReturnObjective {
    /// The environment's chronological primitive reward trace.
    #[default]
    Physical,
    /// A domain-owned version of the environment's actual primitive rewards.
    /// The identity distinguishes incompatible physical reward conventions.
    PhysicalVersioned([u8; 32]),
    /// A caller-owned identity for a domain utility, distinct from factual reward.
    Surrogate([u8; 32]),
}

impl ReturnObjective {
    pub fn is_physical(self) -> bool {
        matches!(self, Self::Physical | Self::PhysicalVersioned(_))
    }

    #[cfg(feature = "neural")]
    pub(crate) fn is_legacy_physical(&self) -> bool {
        *self == Self::Physical
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanningReturns {
    pub objective: ReturnObjective,
    /// Discount per physical tick, including ticks inside each semantic action.
    pub gamma: f64,
}

impl PlanningReturns {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.gamma.is_finite() && (0. ..=1.).contains(&self.gamma),
            "invalid planning physical-time discount",
        )
    }

    pub fn check(&self, other: &Self) -> Result<()> {
        self.validate()?;
        other.validate()?;
        ensure(self == other, "planning objective or discount mismatch")
    }
}

/// Hypothetical transition. The reward is discounted within the action,
/// and a live context is returned only when further actions are possible.
pub struct Transition<C, O> {
    pub observation: O,
    pub reward: f64,
    pub duration: NonZeroU16,
    pub next: Option<C>,
}

/// An observation key identifies the complete public successor of a given
/// action at a fixed public history. Include stochastic duration, public
/// reward or resource changes in the key when they affect available actions
/// or the next predictive state. The tree does not inspect hidden particles.
pub trait PlanningModel {
    type Context: Clone;
    type Action: Clone + Eq;
    type Observation: Copy + Eq;

    /// Append the legal complete actions for this public context in stable order.
    fn actions(&self, context: &Self::Context, actions: &mut Vec<Self::Action>);

    /// Declare the reward objective and the discount used within decisions.
    fn planning_returns(&self) -> PlanningReturns;
}

/// Models that can generate a coherent simulated trajectory.
pub trait PredictiveModel: PlanningModel {
    type Hidden;

    fn sample_root(&self, context: &Self::Context, rng: &mut ChaCha8Rng) -> Result<Self::Hidden>;
    fn rollout(&self, context: &Self::Context) -> Option<Self::Action>;
    fn simulate(
        &self,
        context: &Self::Context,
        hidden: &mut Self::Hidden,
        action: &Self::Action,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Self::Context, Self::Observation>>;
}

/// One positive-probability public outcome, marginalized over hidden state.
pub struct WeightedTransition<C, O> {
    pub probability: f64,
    pub transition: Transition<C, O>,
}

/// A bounded enumeration. An incomplete answer carries work only and cannot
/// be used as a probability law.
pub struct OutcomeDistribution<C, O> {
    pub outcomes: Vec<WeightedTransition<C, O>>,
    pub complete: bool,
}

impl<C, O: Eq> OutcomeDistribution<C, O> {
    pub(crate) fn validated_mass(&self) -> Result<f64> {
        ensure(
            self.complete && !self.outcomes.is_empty(),
            "expected a complete, nonempty model outcome law",
        )?;
        let mut mass = 0.;
        for (index, row) in self.outcomes.iter().enumerate() {
            ensure(
                row.probability > 0.
                    && row.probability <= 1.
                    && row.transition.reward.is_finite()
                    && !self.outcomes[..index]
                        .iter()
                        .any(|prior| prior.transition.observation == row.transition.observation),
                "invalid model outcome probability, reward or duplicate observation key",
            )?;
            mass += row.probability;
        }
        ensure(
            (mass - 1.).abs() <= 1e-9,
            "model outcome law is not normalized",
        )?;
        Ok(mass)
    }
}

/// Exact conditional laws over finite public outcomes.
pub trait EnumerableModel: PlanningModel {
    /// Return at most `cap` transition outcome rows in stable order. The cap
    /// applies at the model interface. Model-internal legality and probability
    /// construction are part of the measured latency. A complete law has unique
    /// public observation keys and total probability one.
    /// Marginalize hidden branches that yield the same public successor.
    fn enumerate(
        &self,
        context: &Self::Context,
        action: &Self::Action,
        cap: NonZeroU32,
    ) -> Result<OutcomeDistribution<Self::Context, Self::Observation>>;
}
