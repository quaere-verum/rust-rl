//! Borrowed complete-action supervision at an existing factual learning row.
//! No replay ownership, policy provenance assertion, or optimizer lives here.
use super::temporal::SequenceSample;
use crate::{
    Result,
    core::{
        action::ConditionalAction, ensure, env::Observation, structured::StructuredObservation,
    },
};

#[derive(Clone, Copy)]
pub struct PolicySupervision<'a> {
    sample: &'a SequenceSample,
    step: usize,
    actions: &'a [ConditionalAction],
    probabilities: &'a [f64],
}
impl<'a> PolicySupervision<'a> {
    pub fn new(
        sample: &'a SequenceSample,
        step: usize,
        actions: &'a [ConditionalAction],
        probabilities: &'a [f64],
    ) -> Result<Self> {
        sample.validate_learning_row(step)?;
        let Observation::Structured(frame) = &sample.observations[step] else {
            return Err(crate::Error::Shape(
                "policy supervision needs a structured decision".into(),
            ));
        };
        ensure(
            !actions.is_empty()
                && actions
                    .iter()
                    .enumerate()
                    .all(|(i, a)| !actions[..i].contains(a))
                && frame
                    .candidates
                    .iter()
                    .filter(|c| c.enabled)
                    .map(|c| c.action)
                    .eq(actions.iter().copied()),
            "policy supervision support differs from ordered complete legal actions",
        )?;
        ensure(
            probabilities.len() == actions.len()
                && probabilities.iter().all(|p| p.is_finite() && *p >= 0.)
                && (probabilities.iter().sum::<f64>() - 1.).abs() <= 1e-9,
            "policy supervision target must be normalized over complete actions",
        )?;
        Ok(Self {
            sample,
            step,
            actions,
            probabilities,
        })
    }
    pub fn sample(&self) -> &'a SequenceSample {
        self.sample
    }
    pub fn step(&self) -> usize {
        self.step
    }
    pub fn actions(&self) -> &'a [ConditionalAction] {
        self.actions
    }
    pub fn probabilities(&self) -> &'a [f64] {
        self.probabilities
    }
    pub fn observation(&self) -> &'a StructuredObservation {
        let Observation::Structured(frame) = &self.sample.observations[self.step] else {
            unreachable!("validated structured supervision")
        };
        frame
    }
}
