//! Opt-in, thread-local wall timings for recurrent training.
//!
//! Available only with `probes`; callers explicitly enable and reset a session.
//! Timings are inclusive. History reconstruction contains encoding, unroll and
//! gathering. The unroll's step buckets are disjoint: a single batched call can
//! process prefix, learning and lookahead rows simultaneously. Padding is counted
//! as executed work but does not determine the bucket. Head evaluation excludes
//! history reconstruction. Optimization includes backward, clipping and Adam.
//! No diagnostic enters a checkpoint.
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap, marker::PhantomData, rc::Rc, time::Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    RolloutStorage,
    BehaviorPreflight,
    BatchValidation,
    MinibatchSelection,
    MinibatchPreparation,
    HistoryReconstruction,
    HistoryEncoding,
    HistoryInputCache,
    HistoryPacking,
    HistoryUnroll,
    HistoryGather,
    DetachedPrefix,
    LearningWindows,
    LaneReconstruction,
    EncodedLaneReconstruction,
    PrefixStep,
    LearningStep,
    LookaheadStep,
    MixedStep,
    PolicyEvaluation,
    ValueEvaluation,
    QEvaluation,
    ActorOptimization,
    CriticOptimization,
    JointOptimization,
    DualOptimization,
    OnlineQHistory,
    TargetQHistory,
    CoalescedReplayHistory,
    MpoUpdate,
    MpoReferencePolicy,
    MpoEStepQ,
    MpoEStep,
    MpoPolicyBackup,
    MpoProjectionLoss,
    MpoCriticFit,
    MpoBacktracking,
}

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub phase: Phase,
    pub calls: u64,
    pub inclusive_seconds: f64,
}

/// Physical row evaluations in profiled GRU histories, including frozen audits.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RecurrentWork {
    pub gru_calls: u64,
    pub prefix_rows: u64,
    pub learning_rows: u64,
    pub lookahead_rows: u64,
    pub padded_rows: u64,
}

#[derive(Clone, Copy, Default)]
struct StepRows {
    prefix: u64,
    learning: u64,
    lookahead: u64,
    padded: u64,
}

#[derive(Default)]
struct Session {
    enabled: bool,
    generation: u64,
    samples: BTreeMap<Phase, Sample>,
    work: RecurrentWork,
    history_steps: Vec<StepRows>,
}
thread_local! {
    static SESSION: RefCell<Session> = RefCell::new(Session::default());
}

/// Enable/disable recording on this thread without clearing completed samples.
pub fn set_enabled(enabled: bool) {
    SESSION.with_borrow_mut(|session| session.enabled = enabled);
}
pub fn enabled() -> bool {
    SESSION.with_borrow(|session| session.enabled)
}
/// Clear this thread's samples and discard any guards predating the reset.
pub fn reset() {
    SESSION.with_borrow_mut(|session| {
        session.samples.clear();
        session.work = RecurrentWork::default();
        session.generation = session.generation.wrapping_add(1);
    });
}
pub fn snapshot() -> Vec<Sample> {
    SESSION.with_borrow(|session| session.samples.values().cloned().collect())
}

pub fn recurrent_work() -> RecurrentWork {
    SESSION.with_borrow(|session| session.work)
}

/// Scope the logical row kinds to the matching encoder call. Classifying ahead
/// of the unroll keeps the instrumentation out of its per-row mask construction.
pub(crate) fn with_history_steps<T>(
    time_steps: usize,
    valid: &[bool],
    learning: &[bool],
    prefix_ends: impl IntoIterator<Item = usize>,
    unroll: impl FnOnce() -> T,
) -> T {
    if !enabled() {
        return unroll();
    }
    let mut steps = vec![StepRows::default(); time_steps];
    for (row, prefix_end) in prefix_ends.into_iter().enumerate() {
        for (time, counts) in steps.iter_mut().enumerate() {
            let token = row * time_steps + time;
            if !valid[token] {
                counts.padded += 1;
            } else if time < prefix_end {
                counts.prefix += 1;
            } else if learning[token] {
                counts.learning += 1;
            } else {
                counts.lookahead += 1;
            }
        }
    }
    struct RestoreSteps(Vec<StepRows>);
    impl Drop for RestoreSteps {
        fn drop(&mut self) {
            SESSION.with_borrow_mut(|session| {
                session.history_steps = std::mem::take(&mut self.0);
            });
        }
    }
    let _restore = RestoreSteps(
        SESSION.with_borrow_mut(|session| std::mem::replace(&mut session.history_steps, steps)),
    );
    unroll()
}

pub(crate) fn enter_history_step(time: usize) -> Option<Guard> {
    SESSION.with_borrow_mut(|session| {
        if !session.enabled {
            return None;
        }
        let rows = *session.history_steps.get(time)?;
        Some(record_step(session, rows))
    })
}

pub(crate) fn enter_detached_step(active: &[bool]) -> Option<Guard> {
    SESSION.with_borrow_mut(|session| {
        session.enabled.then(|| {
            let prefix = active.iter().filter(|&&valid| valid).count();
            record_step(
                session,
                StepRows {
                    prefix: prefix as u64,
                    padded: (active.len() - prefix) as u64,
                    ..Default::default()
                },
            )
        })
    })
}

fn record_step(session: &mut Session, rows: StepRows) -> Guard {
    let phase = match (rows.prefix > 0, rows.learning > 0, rows.lookahead > 0) {
        (true, false, false) => Phase::PrefixStep,
        (false, true, false) => Phase::LearningStep,
        (false, false, true) => Phase::LookaheadStep,
        _ => Phase::MixedStep,
    };
    session.work.gru_calls += 1;
    session.work.prefix_rows += rows.prefix;
    session.work.learning_rows += rows.learning;
    session.work.lookahead_rows += rows.lookahead;
    session.work.padded_rows += rows.padded;
    Guard {
        active: Some((phase, session.generation, Instant::now())),
        same_thread: PhantomData,
    }
}

/// A guard stays on its creating thread. Disabled entry never reads the clock.
#[must_use]
pub struct Guard {
    active: Option<(Phase, u64, Instant)>,
    same_thread: PhantomData<Rc<()>>,
}
pub fn enter(phase: Phase) -> Guard {
    Guard {
        active: SESSION.with_borrow(|session| {
            session
                .enabled
                .then(|| (phase, session.generation, Instant::now()))
        }),
        same_thread: PhantomData,
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        let Some((phase, generation, start)) = self.active else {
            return;
        };
        SESSION.with_borrow_mut(|session| {
            if session.enabled && generation == session.generation {
                let sample = session.samples.entry(phase).or_insert(Sample {
                    phase,
                    calls: 0,
                    inclusive_seconds: 0.,
                });
                sample.calls = sample.calls.saturating_add(1);
                sample.inclusive_seconds += start.elapsed().as_secs_f64();
            }
        });
    }
}
