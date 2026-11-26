//! Search supervision joined to existing factual replay, never imagined replay.
//!
//! A caller owns the immutable run/tape identity and model/planner/policy
//! revisions. Joins check that scope, the exact causal public prefix, and a
//! public projection of semantic actions. They do not authenticate arbitrary
//! caller-provided identities or certify a planner's predictions. Conditional
//! continuations remain hypothetical supervision until independently bound to
//! their own factual decisions. There is no `TemporalSink` implementation here.
use super::guidance::{validate_prior, validate_support};
use crate::models::predictive::{PlanningReturns, ReturnObjective};
use crate::planning::{ActionEstimate, PlanStatus, PlanningBudget, PlanningWork};
use crate::{
    Error, Result,
    core::{
        action::ConditionalAction,
        ensure,
        env::Observation,
        stateful::{PreviousInput, public_history_identity},
    },
    data::temporal::{EpisodeStore, TemporalStep},
};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU16;
mod improvement;
mod supervision;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactualBinding {
    /// Caller-owned immutable run/tape identity; episode numbers are store-local.
    pub tape: [u8; 32],
    pub episode_id: u64,
    pub observation_index: usize,
    pub history: [u8; 32],
}

/// Causal borrowed view: no actions/rewards after the selected observation are
/// exposed to the semantic/public-state projection.
pub struct PublicDecision<'a> {
    observations: &'a [Observation],
    steps: &'a [TemporalStep],
}
impl<'a> PublicDecision<'a> {
    pub fn observation(&self) -> &'a Observation {
        &self.observations[self.steps.len()]
    }
    pub fn observations(&self) -> &'a [Observation] {
        self.observations
    }
    pub fn steps(&self) -> &'a [TemporalStep] {
        self.steps
    }
}
fn public_decision(store: &EpisodeStore, id: u64, index: usize) -> Result<PublicDecision<'_>> {
    let episode = store
        .episode(id)
        .ok_or_else(|| Error::State("search target episode is not retained".into()))?;
    ensure(
        episode.base_timestep == 0 && episode.previous.is_none(),
        "search target needs an authenticated complete public prefix; compacted history cannot be guessed",
    )?;
    ensure(
        index >= episode.learning_start
            && index <= episode.steps.len()
            && episode.steps.len().checked_add(1) == Some(episode.observations.len())
            && !(index == episode.steps.len() && episode.ended()),
        "search target does not identify a factual decision boundary",
    )?;
    Ok(PublicDecision {
        observations: &episode.observations[..=index],
        steps: &episode.steps[..index],
    })
}
fn history(id: u64, public: &PublicDecision<'_>) -> [u8; 32] {
    public_history_identity(
        id,
        0,
        None,
        public.observations,
        public.steps.iter().enumerate().map(|(index, step)| {
            PreviousInput::from_transition(&step.action, &step.reward, &public.observations[index])
        }),
    )
}
impl FactualBinding {
    /// Capture before or after execution; a later join requires the actual
    /// transition to have been collected. Only a complete retained prefix is
    /// supported. Supporting compacted history needs a preserved authenticated
    /// prefix digest, not a digest of a different retained suffix.
    pub fn capture(
        tape: [u8; 32],
        store: &EpisodeStore,
        episode_id: u64,
        observation_index: usize,
    ) -> Result<Self> {
        let public = public_decision(store, episode_id, observation_index)?;
        Ok(Self {
            tape,
            episode_id,
            observation_index,
            history: history(episode_id, &public),
        })
    }
    pub fn verify<'a>(
        &self,
        tape: [u8; 32],
        store: &'a EpisodeStore,
    ) -> Result<PublicDecision<'a>> {
        ensure(
            self.tape == tape,
            "search target belongs to a different factual tape",
        )?;
        let public = public_decision(store, self.episode_id, self.observation_index)?;
        ensure(
            self.history == history(self.episode_id, &public),
            "search target public history differs from factual replay",
        )?;
        Ok(public)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PolicyTarget {
    Probabilities(Vec<f64>),
    /// Same ordered support; unevaluated actions are absent, never fabricated zeros.
    ActionValues(Vec<Option<f64>>),
}
impl PolicyTarget {
    pub fn validate<A: PartialEq>(&self, feasible: &[A]) -> Result<()> {
        validate_support(feasible)?;
        match self {
            Self::Probabilities(prior) => validate_prior(feasible, prior),
            Self::ActionValues(values) => ensure(
                !feasible.is_empty()
                    && values.len() == feasible.len()
                    && values.iter().any(Option::is_some)
                    && values.iter().flatten().all(|value| value.is_finite()),
                "invalid or entirely unevaluated search action values",
            ),
        }
    }
    pub fn from_estimates<A: PartialEq>(
        feasible: &[A],
        estimates: &[ActionEstimate<A>],
    ) -> Result<Self> {
        let values = improvement::ordered_estimates(feasible, estimates)?
            .into_iter()
            .map(|e| e.map(|e| e.value))
            .collect();
        let result = Self::ActionValues(values);
        result.validate(feasible)?;
        Ok(result)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRevision {
    /// Content/configuration identity, including the algorithm for a planner.
    pub identity: [u8; 32],
    pub revision: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchProvenance {
    pub model: SearchRevision,
    pub planner: SearchRevision,
    pub policy: Option<SearchRevision>,
}

/// Observed physical return over an explicitly named span of factual decisions.
/// This span is not automatically the search's semantic-action horizon: one
/// semantic option may execute several primitive replay decisions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactualReturn {
    /// Exclusive absolute transition index.
    pub through: usize,
    pub gamma: f64,
    pub reward: f64,
    pub bootstrap_discount: f64,
    pub physical_ticks: u64,
    pub terminated: bool,
    pub truncated: bool,
}

pub fn factual_return(
    binding: &FactualBinding,
    tape: [u8; 32],
    store: &EpisodeStore,
    through: usize,
    gamma: f64,
) -> Result<FactualReturn> {
    binding.verify(tape, store)?;
    PlanningReturns {
        objective: ReturnObjective::Physical,
        gamma,
    }
    .validate()?;
    let episode = store
        .episode(binding.episode_id)
        .expect("verified retained episode");
    ensure(
        through > binding.observation_index && through <= episode.steps.len(),
        "invalid factual return span",
    )?;
    let (reward, discount, ticks) = crate::data::returns::accumulate_rewards(
        episode.steps[binding.observation_index..through]
            .iter()
            .map(|step| &step.reward),
        gamma,
    )?;
    let last = &episode.steps[through - 1];
    Ok(FactualReturn {
        through,
        gamma,
        reward,
        bootstrap_discount: if last.terminated { 0. } else { discount },
        physical_ticks: ticks,
        terminated: last.terminated,
        truncated: last.truncated,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalTarget<H, A, O> {
    pub after: A,
    pub outcome: O,
    pub public: H,
    pub feasible: Vec<A>,
    pub policy: Option<PolicyTarget>,
    /// Continuation after one semantic action, at horizon minus one.
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTarget<H, A, O> {
    pub version: u32,
    /// Caller-defined public belief/history identity, checked by the join projection.
    pub public: H,
    pub binding: FactualBinding,
    pub feasible: Vec<A>,
    pub policy: Option<PolicyTarget>,
    /// Search estimate in `returns` units; never implicitly a factual return.
    pub value: Option<f64>,
    pub returns: PlanningReturns,
    pub horizon: NonZeroU16,
    pub factual_return: Option<FactualReturn>,
    pub provenance: SearchProvenance,
    pub budget: PlanningBudget,
    pub work: PlanningWork,
    pub status: PlanStatus,
    pub continuations: Vec<ConditionalTarget<H, A, O>>,
}
fn validate_supervision<A: PartialEq>(
    feasible: &[A],
    policy: Option<&PolicyTarget>,
    value: Option<f64>,
) -> Result<()> {
    validate_support(feasible)?;
    ensure(
        !feasible.is_empty()
            && (policy.is_some() || value.is_some())
            && value.is_none_or(f64::is_finite),
        "search target needs finite supervision over nonempty feasible support",
    )?;
    if let Some(policy) = policy {
        policy.validate(feasible)?;
    }
    Ok(())
}
impl<H: PartialEq, A: PartialEq, O: PartialEq> SearchTarget<H, A, O> {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.version == 1 && !matches!(self.status, PlanStatus::Failed | PlanStatus::NoAction),
            "unsupported or unsuccessful search target",
        )?;
        self.returns.validate()?;
        validate_supervision(&self.feasible, self.policy.as_ref(), self.value)?;
        match self.budget {
            PlanningBudget::Expansions(limit) => ensure(
                self.work.expansions <= limit.get(),
                "search target exceeds expansion budget",
            )?,
            PlanningBudget::Simulations {
                trajectories,
                transitions,
            } => ensure(
                self.work.simulations <= trajectories.get()
                    && self.work.transitions <= transitions.get(),
                "search target exceeds simulation budget",
            )?,
        }
        for (index, branch) in self.continuations.iter().enumerate() {
            ensure(
                self.horizon.get() > 1
                    && self.feasible.contains(&branch.after)
                    && !self.continuations[..index]
                        .iter()
                        .any(|old| old.after == branch.after && old.outcome == branch.outcome),
                "invalid or duplicate observation-contingent search branch",
            )?;
            validate_supervision(&branch.feasible, branch.policy.as_ref(), branch.value)?;
        }
        if let Some(factual) = &self.factual_return {
            PlanningReturns {
                objective: ReturnObjective::Physical,
                gamma: factual.gamma,
            }
            .validate()?;
            ensure(
                factual.through > self.binding.observation_index
                    && factual.physical_ticks > 0
                    && factual.reward.is_finite()
                    && factual.bootstrap_discount.is_finite()
                    && (0. ..=1.).contains(&factual.bootstrap_discount)
                    && (!factual.terminated || factual.bootstrap_discount == 0.),
                "invalid factual return metadata",
            )?;
            if self.returns.objective.is_physical() {
                ensure(
                    self.returns.gamma == factual.gamma,
                    "physical search and factual return discounts differ",
                )?;
            }
        }
        Ok(())
    }
    /// Validate hypothetical branch supports against a caller's public predictive
    /// projection. These are not claims that the outcomes occurred in replay.
    pub fn validate_continuations(
        &self,
        mut support: impl FnMut(&A, &O, &H) -> Result<Vec<A>>,
    ) -> Result<()> {
        self.validate()?;
        for branch in &self.continuations {
            ensure(
                support(&branch.after, &branch.outcome, &branch.public)? == branch.feasible,
                "conditional search support differs from public predictive projection",
            )?;
        }
        Ok(())
    }
    /// Join to a collected transition. `project` reconstructs public predictive
    /// identity and ordered feasible *semantic* actions from only the causal
    /// prefix. It need not equate an option identity with the primitive action
    /// stored in replay. `expected` expresses the consumer's revision policy.
    pub fn join<'a>(
        &self,
        tape: [u8; 32],
        store: &'a EpisodeStore,
        expected: &SearchProvenance,
        project: impl FnOnce(&PublicDecision<'_>) -> Result<(H, Vec<A>)>,
    ) -> Result<&'a TemporalStep> {
        self.validate()?;
        ensure(
            &self.provenance == expected,
            "stale or foreign search model/planner/policy revision",
        )?;
        let public = self.binding.verify(tape, store)?;
        let (identity, feasible) = project(&public)?;
        ensure(
            identity == self.public && feasible == self.feasible,
            "search public identity or feasible support differs from factual projection",
        )?;
        if let Some(saved) = &self.factual_return {
            let actual = factual_return(&self.binding, tape, store, saved.through, saved.gamma)?;
            ensure(
                actual == *saved,
                "search factual return differs from actual physical rewards",
            )?;
        }
        store
            .episode(self.binding.episode_id)
            .and_then(|episode| episode.steps.get(self.binding.observation_index))
            .ok_or_else(|| Error::State("search decision has not been factually executed".into()))
    }
}
impl<H: PartialEq, O: PartialEq> SearchTarget<H, ConditionalAction, O> {
    /// Direct conditional-action joins use exactly the enabled complete keys in
    /// the recorded frame's order. Other semantic action spaces use `join`.
    pub fn join_conditional<'a>(
        &self,
        tape: [u8; 32],
        store: &'a EpisodeStore,
        expected: &SearchProvenance,
        identity: impl FnOnce(&PublicDecision<'_>) -> Result<H>,
    ) -> Result<&'a TemporalStep> {
        self.join(tape, store, expected, |public| {
            let Observation::Structured(frame) = public.observation() else {
                return Err(Error::State(
                    "conditional search target needs a structured factual frame".into(),
                ));
            };
            Ok((
                identity(public)?,
                frame
                    .candidates
                    .iter()
                    .filter(|candidate| candidate.enabled)
                    .map(|candidate| candidate.action)
                    .collect(),
            ))
        })
    }
}
