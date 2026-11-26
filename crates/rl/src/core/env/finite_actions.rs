use super::Observation;
use crate::{
    Error, Result,
    core::{
        action::{Action, ActionSpace, FiniteActionKey},
        ensure,
    },
};

impl Observation {
    /// Legal complete actions in stable order by their published identity.
    ///
    /// Supply an observation and action space validated at the interaction boundary.
    /// A live frame must supply at least one action. Continuous spaces require a
    /// policy with a compatible sampling or optimization procedure.
    pub fn finite_actions(&self, space: &ActionSpace) -> Result<Vec<Action>> {
        let actions: Vec<_> = match space {
            ActionSpace::Discrete { count } => {
                let mask = self.action_mask();
                ensure(
                    mask.is_none_or(|mask| mask.len() == *count),
                    "finite action mask length mismatch",
                )?;
                (0..*count)
                    .filter(|&index| mask.is_none_or(|mask| mask[index]))
                    .map(Action::Discrete)
                    .collect()
            }
            ActionSpace::Conditional(_) => {
                let frame = self.as_structured().ok_or_else(|| {
                    Error::Shape("conditional actions require a structured observation".into())
                })?;
                let mut actions: Vec<_> = frame
                    .candidates
                    .iter()
                    .filter(|row| row.enabled)
                    .map(|row| row.action)
                    .collect();
                actions.sort_unstable();
                ensure(
                    actions.windows(2).all(|pair| pair[0] != pair[1]),
                    "duplicate legal action",
                )?;
                actions
                    .into_iter()
                    .map(|action| Action::Conditional(action.into()))
                    .collect()
            }
            ActionSpace::Box { .. } | ActionSpace::Unbounded { .. } => {
                return Err(Error::Action("expected a finite legal action space".into()));
            }
        };
        if actions.is_empty() {
            return Err(Error::Action(
                "no legal action at the public decision".into(),
            ));
        }
        Ok(actions)
    }

    /// The stable key of a finite action relative to this public frame.
    ///
    /// For an available choice, pass the raw action returned by
    /// `finite_actions`. Use `check_action` to validate other actions.
    pub fn finite_action_key(&self, action: &Action) -> Result<FiniteActionKey> {
        match action {
            Action::Discrete(index) => Ok(FiniteActionKey::Discrete(*index)),
            Action::Conditional(action) => {
                let frame = self.as_structured().ok_or_else(|| {
                    Error::Shape("conditional actions require a structured observation".into())
                })?;
                Ok(FiniteActionKey::Conditional(
                    action.in_region(frame.primary_region),
                ))
            }
            Action::Continuous(_) => Err(Error::Action("expected a finite action".into())),
        }
    }
}
