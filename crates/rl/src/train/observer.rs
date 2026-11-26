//! Optional immutable observation at completed learner-update boundaries.
//! No diagnostic traits are required of learners, environments or networks.
use super::counters::RunCounters;
use crate::{Result, core::learner::UpdateReport};

#[derive(Clone, Copy)]
pub struct UpdateBoundary<'a> {
    pub counters: &'a RunCounters,
    /// Completed calls to `Learner::update`, persistent across train calls.
    pub learner_call: u64,
    /// Completed collector calls, distinct from vector steps and revisions.
    pub collection_cycle: u64,
    pub report: &'a UpdateReport,
    /// Cumulative within this train call, excluding observer work.
    pub collection_seconds: f64,
    pub learning_seconds: f64,
}
pub trait UpdateObserver<L: ?Sized> {
    /// An error stops training after the completed update. Models, counters and
    /// replay credit remain committed; observer failures never roll back learning.
    fn after_update(&mut self, learner: &L, boundary: UpdateBoundary<'_>) -> Result<()>;
}
impl<L: ?Sized, F> UpdateObserver<L> for F
where
    F: FnMut(&L, UpdateBoundary<'_>) -> Result<()>,
{
    fn after_update(&mut self, learner: &L, boundary: UpdateBoundary<'_>) -> Result<()> {
        self(learner, boundary)
    }
}
pub(crate) struct NoObserver;
impl<L: ?Sized> UpdateObserver<L> for NoObserver {
    #[inline(always)]
    fn after_update(&mut self, _: &L, _: UpdateBoundary<'_>) -> Result<()> {
        Ok(())
    }
}
