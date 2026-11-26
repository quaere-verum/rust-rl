//! Complete forced-first Monte Carlo evaluation of a frozen public policy.
//!
//! Predictions are generated from independent root particles and never enter
//! factual replay. A root receives a teacher only when every ordered legal
//! action has all its requested samples, each ending in true termination.
use super::{
    guidance::{PolicyGuide, validate_prior, validate_support},
    pomcp::sample_weighted,
    targets::{PolicyTarget, SearchRevision},
};
use crate::models::predictive::{PlanningReturns, PredictiveModel};
use crate::planning::PlanningWork;
use crate::{Result, core::ensure, objectives::mpo::e_step};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::{NonZeroU16, NonZeroU32};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyEvaluationConfig {
    pub samples_per_action: NonZeroU32,
    /// Total calls to the predictive transition law at this root, across all actions.
    pub max_transitions: NonZeroU32,
    /// The model/domain's declared upper bound on semantic decisions to true
    /// termination from this root, including the forced action.
    pub termination_bound: NonZeroU16,
    /// Per-root reverse KL budget, KL(teacher || reference).
    pub epsilon: f64,
}

impl PolicyEvaluationConfig {
    fn validate(self) -> Result<()> {
        ensure(
            self.epsilon.is_finite() && self.epsilon >= 0.,
            "invalid policy improvement KL budget",
        )
    }

    /// Planner identity for target provenance. The caller binds model and
    /// reference publication identities and the return objective independently.
    pub fn revision(self) -> Result<SearchRevision> {
        self.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"rust_rl.forced_first_policy_evaluation.v1");
        digest.update(self.samples_per_action.get().to_le_bytes());
        digest.update(self.max_transitions.get().to_le_bytes());
        digest.update(self.termination_bound.get().to_le_bytes());
        digest.update((self.epsilon + 0.).to_bits().to_le_bytes());
        Ok(SearchRevision {
            identity: digest.finalize().into(),
            revision: 1,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum PolicyEvaluationIneligibility {
    #[error("no legal root actions")]
    NoLegalActions,
    #[error("root actions differ from the model's ordered legal support")]
    RootSupport,
    #[error("live continuation has an empty or invalid legal support")]
    ContinuationSupport,
    #[error("root transition budget exhausted before every sample finished")]
    TransitionCap,
    #[error("live prediction at the declared bound on decisions to termination")]
    TerminationBound,
    #[error("predictive model failed: {0}")]
    Model(String),
    #[error("public policy failed: {0}")]
    Guide(String),
    #[error("prediction produced a nonfinite return or action statistic")]
    InvalidReturn,
    #[error("policy improvement failed: {0}")]
    Improvement(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyActionValue<A> {
    pub action: A,
    pub mean: f64,
    /// Sample standard error of the mean. Absent with fewer than two samples.
    pub standard_error: Option<f64>,
    pub samples: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyImprovement {
    pub target: PolicyTarget,
    /// (teacher - reference) dot Q, in the objective's return units.
    pub predicted_gain: f64,
    pub reverse_kl: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletedPolicyEvaluation<A> {
    pub actions: Vec<PolicyActionValue<A>>,
    pub reference: Vec<f64>,
    pub improvement: PolicyImprovement,
    /// Plug-in standard error of the gain contrast for a fixed teacher. It
    /// omits teacher selection on these samples. Use `gain_lower_bound` for
    /// a simultaneous bound that remains valid for the fitted teacher.
    pub gain_standard_error: Option<f64>,
}

impl<A> CompletedPolicyEvaluation<A> {
    /// One-root lower confidence bound using simultaneous Hoeffding intervals.
    /// The caller supplies valid bounds on all complete trajectory returns.
    /// Over repeated root evaluations, the probability of reporting an invalid
    /// bound is at most `failure_probability`, including attempts that exhaust
    /// the budget without reporting a bound. This also covers choosing the
    /// teacher on these samples. A nonpositive bound leaves gain unresolved.
    pub fn gain_lower_bound(
        &self,
        minimum: f64,
        maximum: f64,
        failure_probability: f64,
    ) -> Result<f64> {
        let span = maximum - minimum;
        ensure(
            span.is_finite() && span >= 0. && failure_probability > 0. && failure_probability < 1.,
            "invalid return range or confidence level",
        )?;
        let PolicyTarget::Probabilities(teacher) = &self.improvement.target else {
            return Err(crate::Error::State(
                "policy evaluation teacher requires probabilities".into(),
            ));
        };
        let log_factor = (2. * self.actions.len() as f64).ln() - failure_probability.ln();
        let uncertainty: f64 = teacher
            .iter()
            .zip(&self.reference)
            .zip(&self.actions)
            .map(|((&q, &p), estimate)| {
                let radius = span
                    * (log_factor / (2. * f64::from(estimate.samples)))
                        .sqrt()
                        .min(1.);
                (q - p).abs() * radius
            })
            .sum();
        let lower = self.improvement.predicted_gain - uncertainty;
        ensure(
            lower.is_finite(),
            "nonfinite policy improvement lower bound",
        )?;
        Ok(lower)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvaluation<A> {
    pub returns: PlanningReturns,
    /// Attempted hypothetical work, including discarded and failed trajectories.
    pub work: PlanningWork,
    /// Samples ending in true termination, including successful samples at
    /// a root that later became ineligible.
    pub completed_trajectories: u32,
    pub outcome: std::result::Result<CompletedPolicyEvaluation<A>, PolicyEvaluationIneligibility>,
}

/// Improve against an exact or otherwise already evaluated ordered Q row. A
/// legal action with zero reference mass still needs a finite value, and the
/// teacher retains zero mass there. This method supplies no sampling guarantee.
pub fn improve_reference_policy<A: PartialEq>(
    feasible: &[A],
    reference: &[f64],
    values: &[f64],
    epsilon: f64,
) -> Result<PolicyImprovement> {
    validate_prior(feasible, reference)?;
    improve(feasible, &normalized_prior(reference), values, epsilon)
}

fn normalized_prior(prior: &[f64]) -> Vec<f64> {
    let total: f64 = prior.iter().sum();
    prior.iter().map(|p| p / total).collect()
}

fn improve<A: PartialEq>(
    feasible: &[A],
    reference: &[f64],
    values: &[f64],
    epsilon: f64,
) -> Result<PolicyImprovement> {
    let improved = e_step(
        reference,
        values,
        &vec![true; feasible.len()],
        feasible.len(),
        epsilon,
    )?;
    let anchor = reference
        .iter()
        .zip(values)
        .find(|(p, _)| **p > 0.)
        .map(|(_, &value)| value)
        .expect("validated reference has positive mass");
    let predicted_gain = improved
        .probabilities
        .iter()
        .zip(reference)
        .zip(values)
        .filter(|((q, p), _)| q != p)
        .map(|((&q, &p), &value)| {
            let difference = value - anchor;
            if difference.is_finite() {
                (q - p) * difference
            } else {
                (q - p) * value - (q - p) * anchor
            }
        })
        .sum::<f64>();
    ensure(
        predicted_gain.is_finite() && improved.reverse_kl.is_finite(),
        "nonfinite policy improvement receipt",
    )?;
    let target = PolicyTarget::Probabilities(improved.probabilities);
    target.validate(feasible)?;
    Ok(PolicyImprovement {
        target,
        predicted_gain,
        reverse_kl: improved.reverse_kl.max(0.),
    })
}

/// Force each legal action at the root, then sample the fixed reference's
/// public probabilities at every subsequent decision. The guide must remain
/// frozen for the whole call. The supplied model RNG seeds a separate action
/// stream once per evaluation, then advances with attempted predictive work.
pub fn evaluate_reference_policy<G, P>(
    model: &G,
    root: &G::Context,
    feasible: &[G::Action],
    guide: &P,
    config: PolicyEvaluationConfig,
    rng: &mut ChaCha8Rng,
) -> Result<PolicyEvaluation<G::Action>>
where
    G: PredictiveModel,
    P: PolicyGuide<G::Context, G::Action>,
{
    config.validate()?;
    let returns = model.planning_returns();
    returns.validate()?;
    let choice_rng = ChaCha8Rng::from_rng(&mut *rng);
    let mut evaluator = Evaluator {
        model,
        guide,
        config,
        gamma: returns.gamma,
        rng,
        choice_rng,
        work: PlanningWork::default(),
        completed_trajectories: 0,
        actions: Vec::new(),
    };
    let outcome = evaluator.evaluate(root, feasible);
    Ok(PolicyEvaluation {
        returns,
        work: evaluator.work,
        completed_trajectories: evaluator.completed_trajectories,
        outcome,
    })
}

use PolicyEvaluationIneligibility as Ineligible;
type Eligible<T> = std::result::Result<T, Ineligible>;

struct Evaluator<'a, G: PredictiveModel, P> {
    model: &'a G,
    guide: &'a P,
    config: PolicyEvaluationConfig,
    gamma: f64,
    rng: &'a mut ChaCha8Rng,
    choice_rng: ChaCha8Rng,
    work: PlanningWork,
    completed_trajectories: u32,
    actions: Vec<G::Action>,
}

impl<G, P> Evaluator<'_, G, P>
where
    G: PredictiveModel,
    P: PolicyGuide<G::Context, G::Action>,
{
    fn evaluate(
        &mut self,
        root: &G::Context,
        feasible: &[G::Action],
    ) -> Eligible<CompletedPolicyEvaluation<G::Action>> {
        self.model.actions(root, &mut self.actions);
        if self.actions != feasible || validate_support(feasible).is_err() {
            return Err(Ineligible::RootSupport);
        }
        if feasible.is_empty() {
            return Err(Ineligible::NoLegalActions);
        }
        let reference = normalized_prior(&self.prior(root, feasible)?);
        let mut actions = Vec::with_capacity(feasible.len());
        let mut values = Vec::with_capacity(feasible.len());
        let samples = self.config.samples_per_action.get();
        for action in feasible {
            let mut mean = 0.;
            let mut deviation = 0_f64;
            for n in 1..=samples {
                let value = self.trajectory(root, action)?;
                self.completed_trajectories += 1;
                let delta = value - mean;
                mean += delta / f64::from(n);
                // Update the square root of Welford's sum of squared deviations
                // to keep the standard error representable at small reward scales.
                deviation = deviation.hypot(delta * (f64::from(n - 1) / f64::from(n)).sqrt());
            }
            if !mean.is_finite() || !deviation.is_finite() {
                return Err(Ineligible::InvalidReturn);
            }
            let standard_error = (samples > 1)
                .then(|| deviation / f64::from(samples).sqrt() / f64::from(samples - 1).sqrt());
            actions.push(PolicyActionValue {
                action: action.clone(),
                mean,
                standard_error,
                samples,
            });
            values.push(mean);
        }
        let improvement = improve(feasible, &reference, &values, self.config.epsilon)
            .map_err(|error| Ineligible::Improvement(error.to_string()))?;
        let PolicyTarget::Probabilities(teacher) = &improvement.target else {
            unreachable!()
        };
        let gain_standard_error = teacher
            .iter()
            .zip(&reference)
            .zip(&actions)
            .try_fold(0_f64, |se, ((&q, &p), estimate)| {
                Some(se.hypot((q - p).abs() * estimate.standard_error?))
            });
        Ok(CompletedPolicyEvaluation {
            actions,
            reference,
            improvement,
            gain_standard_error,
        })
    }

    fn prior(&self, context: &G::Context, actions: &[G::Action]) -> Eligible<Vec<f64>> {
        let prior = self
            .guide
            .prior(context, actions)
            .map_err(|error| Ineligible::Guide(error.to_string()))?;
        validate_prior(actions, &prior).map_err(|error| Ineligible::Guide(error.to_string()))?;
        Ok(prior)
    }

    fn trajectory(&mut self, root: &G::Context, first: &G::Action) -> Eligible<f64> {
        if self.work.transitions == self.config.max_transitions.get() {
            return Err(Ineligible::TransitionCap);
        }
        self.work.simulations += 1;
        let mut hidden = self
            .model
            .sample_root(root, self.rng)
            .map_err(|error| Ineligible::Model(error.to_string()))?;
        let mut context = root;
        let mut action = first;
        let mut continuation;
        let mut reward: f64 = 0.;
        let mut discount = 1.;
        for step in 1..=self.config.termination_bound.get() {
            self.work.transitions += 1;
            let transition = self
                .model
                .simulate(context, &mut hidden, action, self.rng)
                .map_err(|error| Ineligible::Model(error.to_string()))?;
            reward += discount * transition.reward;
            if !reward.is_finite() {
                return Err(Ineligible::InvalidReturn);
            }
            let Some(next) = transition.next else {
                return Ok(reward);
            };
            if step == self.config.termination_bound.get() {
                return Err(Ineligible::TerminationBound);
            }
            if self.work.transitions == self.config.max_transitions.get() {
                return Err(Ineligible::TransitionCap);
            }
            discount *= self.gamma.powi(i32::from(transition.duration.get()));
            continuation = next;
            context = &continuation;
            self.actions.clear();
            self.model.actions(context, &mut self.actions);
            if self.actions.is_empty() || validate_support(&self.actions).is_err() {
                return Err(Ineligible::ContinuationSupport);
            }
            let prior = self.prior(context, &self.actions)?;
            let selected = sample_weighted(prior, &mut self.choice_rng)
                .map_err(|error| Ineligible::Guide(error.to_string()))?;
            action = &self.actions[selected];
        }
        Err(Ineligible::TerminationBound)
    }
}
