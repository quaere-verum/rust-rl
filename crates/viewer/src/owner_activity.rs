//! Small screenshot-build ownership witnesses, never process/allocator estimates.
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
pub(crate) struct Activity {
    busy: AtomicUsize,
    started: AtomicUsize,
    outstanding: AtomicUsize,
    executing: AtomicUsize,
    executions: AtomicUsize,
}
impl Activity {
    pub(crate) fn busy(self: &Arc<Self>) -> Busy {
        self.busy.fetch_add(1, Ordering::AcqRel);
        self.started.fetch_add(1, Ordering::AcqRel);
        Busy(self.clone())
    }
    pub(crate) fn own(self: &Arc<Self>) -> Ownership {
        self.outstanding.fetch_add(1, Ordering::AcqRel);
        Ownership(self.clone())
    }
    pub(crate) fn execute(self: &Arc<Self>) -> Execution {
        self.executing.fetch_add(1, Ordering::AcqRel);
        self.executions.fetch_add(1, Ordering::AcqRel);
        Execution(self.clone())
    }
    pub(crate) fn snapshot(&self) -> ActivitySnapshot {
        ActivitySnapshot {
            busy: self.busy.load(Ordering::Acquire),
            started: self.started.load(Ordering::Acquire),
            outstanding: self.outstanding.load(Ordering::Acquire),
            executing: self.executing.load(Ordering::Acquire),
            executions: self.executions.load(Ordering::Acquire),
        }
    }
}
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ActivitySnapshot {
    pub busy: usize,
    pub started: usize,
    /// Requests, local answers and channel answers through their final drop.
    pub outstanding: usize,
    pub executing: usize,
    pub executions: usize,
}
#[cfg(test)]
impl ActivitySnapshot {
    pub(crate) fn idle(self) -> bool {
        self.busy == 0 && self.outstanding == 0
    }
}
pub(crate) struct Busy(Arc<Activity>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.busy.fetch_sub(1, Ordering::AcqRel);
    }
}
pub(crate) struct Ownership(Arc<Activity>);
impl Drop for Ownership {
    fn drop(&mut self) {
        self.0.outstanding.fetch_sub(1, Ordering::AcqRel);
    }
}
pub(crate) struct Execution(Arc<Activity>);
impl Drop for Execution {
    fn drop(&mut self) {
        self.0.executing.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn queued_answer_ownership_outlives_busy_and_dies_after_payload() {
        struct Payload(Arc<Activity>);
        impl Drop for Payload {
            fn drop(&mut self) {
                assert_eq!(self.0.snapshot().outstanding, 1);
            }
        }
        struct Envelope {
            _payload: Payload,
            _owner: Ownership,
        }
        let activity = Arc::new(Activity::default());
        let owner = activity.own();
        let busy = activity.busy();
        let result = Envelope {
            _payload: Payload(activity.clone()),
            _owner: owner,
        };
        drop(busy);
        assert!(!activity.snapshot().idle());
        drop(result);
        assert!(activity.snapshot().idle());
    }
}
