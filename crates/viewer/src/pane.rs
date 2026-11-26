//! Checked selected policy data and deferred selection changes.
use crate::{
    data::{RunIndex, SelectedInput, Selection},
    policy_change::{PolicyBatch, admission::checked_batch},
    resolver::ResolvedRecord,
};
use rl::core::action::ActionSpace;

pub struct PolicySource;
pub struct CheckedPolicy<'a> {
    pub frame: &'a ResolvedRecord,
    pub batch: PolicyBatch<'a>,
}
impl PolicySource {
    pub fn check_frame<'a>(
        frame: &'a ResolvedRecord,
        space: &'a ActionSpace,
        scratch_budget: usize,
    ) -> rl::Result<CheckedPolicy<'a>> {
        let batch = checked_batch(&frame.chunk, &frame.probes, space, scratch_budget)?;
        if frame.row >= batch.rows() {
            return Err(rl::Error::Shape(
                "selected policy row is out of bounds".into(),
            ));
        }
        Ok(CheckedPolicy { frame, batch })
    }
}
pub enum SelectionIntent {
    Input(SelectedInput),
    ModelRole(String),
    Revision(u64),
}
impl SelectionIntent {
    /// Apply to the caller's pending next selection, after readers release this pass.
    pub fn apply(self, run: &RunIndex, next: &mut Selection) -> rl::Result<()> {
        match self {
            Self::Input(input) => next.input = input,
            Self::Revision(revision) => next.select_revision(run, revision)?,
            Self::ModelRole(role) => {
                next.model_role = role;
                next.select_revision(run, next.revision)?;
            }
        }
        Ok(())
    }
}
