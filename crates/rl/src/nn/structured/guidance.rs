//! Neural adapters for the existing public-context planning interfaces.
use super::{
    CandidatePolicy, LightweightValue, PackLimits, StructuredSample,
    candidate_probabilities_checked,
};
use crate::models::predictive::PlanningReturns;
use crate::{
    Result,
    core::{action::ConditionalAction, ensure, structured::StructuredObservation},
    planning::guidance::{PolicyGuide, ValueGuide, validate_prior},
};
use burn::module::Module;
use std::num::NonZeroU16;

/// Domain-owned public projection shared with factual neural encoding. Do not
/// fabricate notebook events or capture a hidden simulator in the projection.
pub struct PublicValueInput {
    pub observation: StructuredObservation,
    /// Conservative upper bound on semantic decisions until genuine termination
    /// along EVERY legal continuation, not the length of a realized trajectory.
    pub terminal_within: NonZeroU16,
}

/// Immutable guide pair over existing independent models. The caller owns
/// publication identity and freezes the models used to produce each target.
/// No search tree, replay, optimizer or environment is owned by this adapter.
pub struct NeuralGuidance<'a, P, F> {
    policy: &'a P,
    value: &'a LightweightValue,
    returns: PlanningReturns,
    limits: PackLimits,
    project: F,
}

impl<'a, P: CandidatePolicy, F> NeuralGuidance<'a, P, F> {
    pub fn new(
        policy: &'a P,
        value: &'a LightweightValue,
        returns: PlanningReturns,
        limits: PackLimits,
        project: F,
    ) -> Result<Self> {
        returns.validate()?;
        ensure(
            returns.objective.is_physical(),
            "neural Monte Carlo guidance requires the physical-return objective",
        )?;
        P::validate_configuration(policy.configuration())?;
        LightweightValue::parameter_count(value.config())?;
        ensure(
            policy.input_schema() == &value.config().input,
            "neural policy/value guidance input schemas differ",
        )?;
        Ok(Self {
            policy,
            value,
            returns,
            limits,
            project,
        })
    }
}

impl<C, P: CandidatePolicy, F: Fn(&C) -> Result<PublicValueInput>> PolicyGuide<C, ConditionalAction>
    for NeuralGuidance<'_, P, F>
{
    fn prior(&self, context: &C, feasible: &[ConditionalAction]) -> Result<Vec<f64>> {
        let input = (self.project)(context)?;
        ensure(
            input
                .observation
                .candidates
                .iter()
                .filter(|row| row.enabled)
                .map(|row| row.action)
                .eq(feasible.iter().copied()),
            "neural guidance support differs from ordered complete legal actions",
        )?;
        let sample = StructuredSample::from_observation_with_validated_schema(
            &input.observation,
            self.policy.input_schema(),
        )?;
        let all = candidate_probabilities_checked(self.policy, &[sample], self.limits)?
            .pop()
            .expect("one validated neural policy row");
        let probabilities = all
            .into_iter()
            .zip(&input.observation.candidates)
            .filter_map(|(p, row)| row.enabled.then_some(p))
            .collect::<Vec<_>>();
        validate_prior(feasible, &probabilities)?;
        Ok(probabilities)
    }
}

impl<C, P, F: Fn(&C) -> Result<PublicValueInput>> ValueGuide<C> for NeuralGuidance<'_, P, F> {
    fn value(&self, context: &C, remaining: u16, returns: &PlanningReturns) -> Result<f64> {
        self.returns.check(returns)?;
        let input = (self.project)(context)?;
        ensure(
            remaining >= input.terminal_within.get(),
            "full-return neural value exceeds the remaining planning horizon",
        )?;
        let sample = StructuredSample::from_observation_with_validated_schema(
            &input.observation,
            &self.value.config().input,
        )?;
        let value = f64::from(
            self.value
                .valid()
                .evaluate(&[sample], self.limits)?
                .into_scalar::<f32>(),
        );
        ensure(value.is_finite(), "nonfinite neural planning value")?;
        Ok(value)
    }
}
