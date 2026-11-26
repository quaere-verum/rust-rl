//! Scheduled recording around an immutable completed learner revision.
use super::{
    schedule::{Due, RecordingSchedule},
    schema::*,
    storage::Recorder,
};
use crate::{
    Result,
    train::observer::{UpdateBoundary, UpdateObserver},
};
use std::{collections::BTreeMap, time::Instant};

pub struct DiagnosticObserver<F> {
    recorder: Recorder,
    schedule: RecordingSchedule,
    callback: F,
    previous_revision: Option<u64>,
    seconds: f64,
}
impl<F> DiagnosticObserver<F> {
    pub fn new(recorder: Recorder, schedule: RecordingSchedule, callback: F) -> Self {
        Self {
            recorder,
            schedule,
            callback,
            previous_revision: None,
            seconds: 0.,
        }
    }
    pub fn recorder(&self) -> &Recorder {
        &self.recorder
    }
    pub fn diagnostic_seconds(&self) -> f64 {
        self.seconds
    }
    pub fn finish(&mut self) -> Result<()> {
        let start = Instant::now();
        self.recorder.finish()?;
        self.seconds += start.elapsed().as_secs_f64();
        Ok(())
    }
}
impl<L: ?Sized, F> UpdateObserver<L> for DiagnosticObserver<F>
where
    F: FnMut(&L, &RecordContext, Due, &mut Recorder) -> Result<()>,
{
    fn after_update(&mut self, learner: &L, boundary: UpdateBoundary<'_>) -> Result<()> {
        let start = Instant::now();
        let revision = self
            .recorder
            .manifest()
            .starting_revision
            .checked_add(boundary.counters.updates)
            .ok_or_else(|| crate::Error::State("diagnostic revision overflow".into()))?;
        let mut context = RecordContext::new(&self.recorder.manifest().run_id, revision);
        context.counters = *boundary.counters;
        context.learner_call = boundary.learner_call;
        context.collection_cycle = boundary.collection_cycle;
        let due = self.schedule.due(self.previous_revision, revision)?;
        // User-selected adapters retain static dispatch. A custom network without
        // probes can use metrics alone and requires no diagnostic implementation.
        if due.probes || due.monte_carlo || due.snapshots {
            (self.callback)(learner, &context, due, &mut self.recorder)?;
        }
        if due.metrics {
            append_metrics(
                &mut self.recorder,
                &context,
                boundary,
                self.seconds + start.elapsed().as_secs_f64(),
            )?;
        }
        self.seconds += start.elapsed().as_secs_f64();
        self.previous_revision = Some(revision);
        Ok(())
    }
}

pub(crate) fn append_metrics(
    recorder: &mut Recorder,
    context: &RecordContext,
    boundary: UpdateBoundary<'_>,
    diagnostic_seconds: f64,
) -> Result<()> {
    let mut values = BTreeMap::new();
    let mut unavailable = BTreeMap::new();
    for (name, &value) in &boundary.report.metrics {
        if value.is_finite() {
            values.insert(name.clone(), value);
        } else {
            unavailable.insert(name.clone(), format!("nonfinite: {value}"));
            recorder.append(&Chunk {
                context: context.clone(),
                record: Record::Event(EventRecord {
                    category: "numerical".into(),
                    severity: "warning".into(),
                    message: format!("nonfinite metric {name}: {value}"),
                }),
                arrays: BTreeMap::new(),
            })?;
        }
    }
    values.insert(
        "time.collection_seconds_in_call".into(),
        boundary.collection_seconds,
    );
    values.insert(
        "time.learning_seconds_in_call".into(),
        boundary.learning_seconds,
    );
    values.insert(
        "time.diagnostics_seconds_before_metric_write".into(),
        diagnostic_seconds,
    );
    values.insert(
        "updates.in_learner_call".into(),
        boundary.report.updates as f64,
    );
    values.insert(
        "storage.committed_bytes_before_metric_write".into(),
        recorder.status().committed_bytes as f64,
    );
    values.insert(
        "storage.skipped_records".into(),
        recorder.status().skipped_records as f64,
    );
    recorder.append(&Chunk {
        context: context.clone(),
        record: Record::Metrics(MetricRecord {
            values,
            unavailable,
        }),
        arrays: BTreeMap::new(),
    })?;
    Ok(())
}
