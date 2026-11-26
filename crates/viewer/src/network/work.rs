//! Bounded asynchronous preparation for one selected historical MLP.
//!
//! The worker retains one detached numerical model and one bounded reference.
//! Restore temporarily overlaps the chunk and f32 model, then the f32 model and
//! selected f64 weights; old numerical/reference caches are dropped first.
//! ModelLimits bounds restoration/parameters, and configuration/name allocations
//! retain the same validated snapshot limits. Metadata/choices are shared with
//! published frames. One active and one queued request bound request ownership;
//! generation checks and last-request-wins publication reject obsolete work.
use super::{
    Frame, Preparation, Request, current,
    selection::{NetworkChoice, choices},
};
use crate::snapshot_catalog::SnapshotEntry;
use rl::{
    Result,
    diagnostics::{
        models::{ModelBundle, ModelLimits, SnapshotConfiguration},
        representations::Pca,
        storage::ChunkSource,
    },
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender, TrySendError},
    },
    thread,
};
struct Job {
    serial: u64,
    population_scope: String,
    snapshot: SnapshotEntry,
    source: ChunkSource,
    request: Request,
}
enum Command {
    Prepare(Box<Job>),
    Retire,
}
struct Cached {
    snapshot: SnapshotEntry,
    configuration: Arc<SnapshotConfiguration>,
    choices: Arc<Vec<NetworkChoice>>,
    preparation: Preparation,
    population_scope: String,
}
#[derive(Default)]
struct WorkerState {
    cached: Option<Cached>,
}
impl WorkerState {
    fn retire(&mut self) {
        self.cached = None;
    }
    fn run(&mut self, job: Job, check: &impl Fn() -> bool) -> Result<PublishedFrame> {
        current(check)?;
        if let Err(error) = job.source.check_available() {
            self.retire();
            return Err(error);
        }
        if job.request.network != job.snapshot.name {
            return Err(rl::Error::State("Network snapshot name mismatch".into()));
        }
        if self.cached.as_ref().is_none_or(|cached| {
            cached.snapshot.key != job.snapshot.key || cached.snapshot.name != job.snapshot.name
        }) {
            // Release the old reference and detached weights before restore.
            self.retire();
            let chunk = job.source.load()?;
            current(check)?;
            if chunk.context.identity.learner_update_revision != job.snapshot.revision
                || chunk.context.identity.model_role != job.snapshot.role
            {
                return Err(rl::Error::State(
                    "Network snapshot identity mismatch".into(),
                ));
            }
            let model = ModelBundle::load(&chunk, ModelLimits::default())?;
            drop(chunk);
            current(check)?;
            let network = model.numerical_network(&job.request.network)?;
            let configuration = Arc::new(model.configuration().clone());
            // Do not retain both restored tensors and their f64 copy.
            drop(model);
            current(check)?;
            self.cached = Some(Cached {
                snapshot: job.snapshot.clone(),
                choices: Arc::new(choices(&configuration)),
                configuration,
                preparation: Preparation::new(network),
                population_scope: job.population_scope.clone(),
            });
        }
        let cached = self.cached.as_mut().expect("model restored");
        if cached.population_scope != job.population_scope {
            cached.preparation.reference = None;
            cached.preparation.neighborhood = None;
            cached.population_scope = job.population_scope;
        }
        if cached.snapshot.revision != job.snapshot.revision
            || cached.snapshot.role != job.snapshot.role
        {
            return Err(rl::Error::State(
                "Cached Network snapshot identity mismatch".into(),
            ));
        }
        let geometry = cached
            .preparation
            .prepare_cancellable(&job.request, check)?;
        current(check)?;
        if let Err(error) = job.source.check_available() {
            self.retire();
            return Err(error);
        }
        let cached = self.cached.as_ref().expect("model prepared");
        let choice = cached
            .choices
            .iter()
            .find(|c| c.name == job.snapshot.name)
            .ok_or_else(|| rl::Error::State("Network output definition missing".into()))?;
        let output = super::output::prepare(
            &cached.configuration.models[&choice.model].kind,
            &job.snapshot.name,
            &cached.configuration.action_space,
            job.request.action_mask.as_deref(),
            &geometry.output_values,
            &geometry.finite_output_values,
        );
        current(check)?;
        if let Err(error) = job.source.check_available() {
            self.retire();
            return Err(error);
        }
        let cached = self.cached.as_ref().expect("model output prepared");
        Ok(PublishedFrame {
            snapshot: job.snapshot,
            configuration: cached.configuration.clone(),
            choices: cached.choices.clone(),
            geometry,
            output,
            preparation_retained_bytes: cached.preparation.retained_bytes(),
        })
    }
}
struct Answer {
    serial: u64,
    result: Result<PublishedFrame>,
}
/// Geometry and the exact immutable weights from which it was computed.
pub struct PublishedFrame {
    pub snapshot: SnapshotEntry,
    /// Small model metadata, not weights, retained with the geometry it describes.
    pub configuration: Arc<SnapshotConfiguration>,
    pub choices: Arc<Vec<NetworkChoice>>,
    pub geometry: Frame,
    pub output: std::result::Result<super::output::OutputComparison, String>,
    pub preparation_retained_bytes: usize,
}
pub struct NetworkWorker {
    send: SyncSender<Command>,
    receive: Receiver<Answer>,
    queued: Option<Job>,
    running: bool,
    serial: u64,
    generation: Arc<AtomicU64>,
    retirement: Arc<AtomicU64>,
    wanted: Option<String>,
    published: Option<String>,
    frame: Option<Arc<PublishedFrame>>,
    error: Option<String>,
    basis_scope: Option<String>,
    basis: Vec<Option<Pca>>,
}
impl NetworkWorker {
    pub fn new(wake: impl Fn() + Send + 'static) -> Self {
        let (send, jobs) = mpsc::sync_channel::<Command>(1);
        let (results, receive) = mpsc::channel::<Answer>();
        let generation = Arc::new(AtomicU64::new(0));
        let retirement = Arc::new(AtomicU64::new(0));
        let worker_generation = generation.clone();
        let worker_retirement = retirement.clone();
        thread::spawn(move || {
            let mut state = WorkerState::default();
            let mut retired = 0;
            while let Ok(command) = jobs.recv() {
                let epoch = worker_retirement.load(Ordering::Acquire);
                if retired != epoch {
                    state.retire();
                    retired = epoch;
                }
                let Command::Prepare(job) = command else {
                    state.retire();
                    // A newer request may have been deferred behind this
                    // bounded retirement message. Wake only in that case;
                    // repainting an unavailable source can itself retire again.
                    if worker_generation.load(Ordering::Acquire)
                        != worker_retirement.load(Ordering::Acquire)
                    {
                        wake();
                    }
                    continue;
                };
                let serial = job.serial;
                let result = state.run(*job, &|| {
                    worker_generation.load(Ordering::Acquire) == serial
                });
                let epoch = worker_retirement.load(Ordering::Acquire);
                if retired != epoch {
                    state.retire();
                    retired = epoch;
                }
                if results.send(Answer { serial, result }).is_err() {
                    break;
                }
                wake();
            }
        });
        Self {
            send,
            receive,
            queued: None,
            running: false,
            serial: 0,
            generation,
            retirement,
            wanted: None,
            published: None,
            frame: None,
            error: None,
            basis_scope: None,
            basis: Vec::new(),
        }
    }
    pub fn reset(&mut self) {
        self.invalidate_frame();
        self.basis_scope = None;
        self.basis.clear();
    }
    pub fn invalidate_frame(&mut self) {
        self.serial = self.serial.wrapping_add(1);
        self.generation.store(self.serial, Ordering::Release);
        self.retirement.store(self.serial, Ordering::Release);
        // Wake an idle worker as well as cancelling an active one. If the
        // bounded channel is full, its queued command observes the epoch.
        let _ = self.send.try_send(Command::Retire);
        self.wanted = None;
        self.published = None;
        self.queued = None;
        self.frame = None;
        self.error = None;
    }
    pub fn frame(&self) -> Option<&PublishedFrame> {
        self.frame.as_deref()
    }
    pub fn pending(&self) -> bool {
        self.running || self.queued.is_some()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn submitted(&self) -> Option<&str> {
        self.wanted.as_deref()
    }
    fn dispatch(&mut self) {
        if self.running {
            return;
        }
        let Some(job) = self.queued.take() else {
            return;
        };
        match self.send.try_send(Command::Prepare(Box::new(job))) {
            Ok(()) => self.running = true,
            Err(TrySendError::Full(Command::Prepare(job))) => self.queued = Some(*job),
            Err(TrySendError::Full(Command::Retire)) => unreachable!(),
            Err(TrySendError::Disconnected(_)) => {
                self.error = Some("Network worker disconnected".into());
            }
        }
    }
    pub fn poll(&mut self) {
        while let Ok(answer) = self.receive.try_recv() {
            self.running = false;
            if answer.serial == self.serial {
                match answer.result {
                    Ok(frame) => {
                        // prepare reuses compatible bases and refits only layers
                        // whose dimensions changed. Keep those replacements too.
                        self.basis = frame.geometry.pca_bases.clone();
                        self.frame = Some(Arc::new(frame));
                        self.error = None;
                        self.published = self.wanted.clone();
                    }
                    Err(error) => {
                        self.error = Some(error.to_string());
                        self.published = None;
                        self.frame = None;
                    }
                }
            }
        }
        self.dispatch();
    }
    /// Caller keys every relevant selection, including the exact immutable
    /// model source and original input source. No unrelated row is substituted.
    pub fn request(
        &mut self,
        signature: String,
        basis_scope: String,
        snapshot: SnapshotEntry,
        source: ChunkSource,
        mut request: Request,
    ) {
        if self.basis_scope.as_deref() != Some(&basis_scope) {
            self.basis_scope = Some(basis_scope);
            self.basis.clear();
            self.frame = None;
            self.published = None;
            self.wanted = None;
        }
        if self.wanted.as_deref() == Some(&signature) {
            return;
        }
        request.previous_basis = self.basis.clone();
        self.serial = self.serial.wrapping_add(1);
        self.generation.store(self.serial, Ordering::Release);
        self.wanted = Some(signature.clone());
        self.error = None;
        self.queued = Some(Job {
            serial: self.serial,
            population_scope: self.basis_scope.clone().unwrap_or_default(),
            snapshot,
            source,
            request,
        });
        self.dispatch();
    }
    pub fn ready(&self) -> bool {
        self.published.as_ref() == self.wanted.as_ref()
            && self.published.is_some()
            && self.frame.is_some()
    }
}
impl Drop for NetworkWorker {
    fn drop(&mut self) {
        let serial = self
            .generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        self.retirement.store(serial, Ordering::Release);
        let _ = self.send.try_send(Command::Retire);
    }
}

#[cfg(test)]
use crate::network_test_fixture as fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::RunIndex, snapshot_catalog::SnapshotStatus};
    use rl::diagnostics::storage::ReadLimits;

    fn job(run: &RunIndex, index: usize) -> Job {
        let probes = fixtures::observations();
        let entry = &run.recording().entries()[index];
        Job {
            serial: 1,
            population_scope: "fixed input source".into(),
            snapshot: SnapshotEntry {
                key: run.key(index).unwrap(),
                revision: entry.revision,
                role: entry.model_role.clone(),
                name: "actor".into(),
                status: SnapshotStatus::Eligible,
            },
            source: run.recording().source(index).unwrap(),
            request: Request {
                network: "actor".into(),
                anchor: probes[3].clone(),
                endpoint: probes[8].clone(),
                probes,
                axes: vec![0, 1],
                axis_scales: vec![1., 1.],
                ranges: [[0., 1.], [-1., 1.]],
                direction_angle: 0.4,
                t: 1.,
                radius: 0.2,
                action_mask: None,
                layer: 1,
                grid: [3, 3],
                previous_basis: vec![],
            },
        }
    }

    #[test]
    fn detached_model_reuse_tracks_artifact_and_population_and_releases_retired_state() {
        let (dir, mut writer) = fixtures::recorder();
        writer
            .append(&fixtures::snapshot(0, "online", &["actor"]))
            .unwrap();
        writer
            .append(&fixtures::snapshot(0, "online", &["actor"]))
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut state = WorkerState::default();
        state.run(job(&run, 1), &|| true).unwrap();
        let first_model = Arc::downgrade(&state.cached.as_ref().unwrap().preparation.network);
        let first_reference = Arc::downgrade(
            state
                .cached
                .as_ref()
                .unwrap()
                .preparation
                .reference
                .as_ref()
                .unwrap(),
        );
        let mut edited = job(&run, 1);
        edited.request.direction_angle = 1.2;
        edited.request.endpoint = edited.request.probes[4].clone();
        edited.request.layer = 0;
        state.run(edited, &|| true).unwrap();
        assert!(first_model.upgrade().is_some());
        assert!(first_reference.upgrade().is_some());
        let mut population = job(&run, 1);
        population.request.probes[0][0] += 0.2;
        state.run(population, &|| true).unwrap();
        assert!(first_model.upgrade().is_some());
        assert!(first_reference.upgrade().is_none());
        let changed_values = Arc::downgrade(
            state
                .cached
                .as_ref()
                .unwrap()
                .preparation
                .reference
                .as_ref()
                .unwrap(),
        );
        let mut replacement_source = job(&run, 1);
        replacement_source.request.probes[0][0] += 0.2;
        replacement_source.population_scope = "different source with identical values".into();
        state.run(replacement_source, &|| true).unwrap();
        assert!(
            changed_values.upgrade().is_none(),
            "reference identity includes the immutable population source"
        );
        state.run(job(&run, 2), &|| true).unwrap();
        assert!(
            first_model.upgrade().is_none(),
            "same-revision sibling is a new model"
        );
        let last = Arc::downgrade(&state.cached.as_ref().unwrap().preparation.network);
        state.retire();
        assert!(last.upgrade().is_none());
    }

    #[test]
    fn cancelled_or_deleted_source_never_uses_cached_model() {
        let (dir, mut writer) = fixtures::recorder();
        writer
            .append(&fixtures::snapshot(0, "online", &["actor"]))
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut state = WorkerState::default();
        assert!(state.run(job(&run, 1), &|| false).is_err());
        assert!(state.cached.is_none());
        state.run(job(&run, 1), &|| true).unwrap();
        std::fs::remove_file(run.root().join(&run.recording().entries()[1].file)).unwrap();
        assert!(state.run(job(&run, 1), &|| true).is_err());
        assert!(
            state.cached.is_none(),
            "deleted source releases retained weights"
        );
    }

    #[test]
    fn source_pruned_at_the_final_output_publication_boundary_is_not_published() {
        use std::cell::Cell;
        let (dir, mut writer) = fixtures::recorder();
        writer
            .append(&fixtures::snapshot(0, "online", &["actor"]))
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let mut state = WorkerState::default();
        state.run(job(&run, 1), &|| true).unwrap();
        let calls = Cell::new(0);
        state
            .run(job(&run, 1), &|| {
                calls.set(calls.get() + 1);
                true
            })
            .unwrap();
        let boundary = calls.get();
        calls.set(0);
        let path = run.root().join(&run.recording().entries()[1].file);
        let result = state.run(job(&run, 1), &|| {
            calls.set(calls.get() + 1);
            if calls.get() == boundary {
                std::fs::remove_file(&path).unwrap();
            }
            true
        });
        assert!(!path.exists());
        assert!(
            result.is_err(),
            "newly mapped output published a pruned snapshot"
        );
        assert!(
            state.cached.is_none(),
            "a pruned source kept detached weights/reference/map caches"
        );
    }

    #[test]
    fn idle_reset_releases_worker_owned_metadata_without_another_request() {
        let (dir, mut writer) = fixtures::recorder();
        writer
            .append(&fixtures::snapshot(0, "online", &["actor"]))
            .unwrap();
        writer.finish().unwrap();
        let run = RunIndex::open(dir.path().join("run"), ReadLimits::default()).unwrap();
        let (wakes, notifications) = mpsc::channel();
        let mut worker = NetworkWorker::new(move || {
            let _ = wakes.send(());
        });
        let job = job(&run, 1);
        worker.request(
            "first".into(),
            "population".into(),
            job.snapshot,
            job.source,
            job.request,
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while worker.pending() {
            worker.poll();
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(worker.ready(), "{:?}", worker.error());
        notifications
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        let metadata = Arc::downgrade(&worker.frame().unwrap().configuration);
        worker.reset();
        assert!(worker.frame().is_none());
        while metadata.upgrade().is_some() {
            assert!(
                std::time::Instant::now() < deadline,
                "idle worker retained retired cache"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(
            matches!(
                notifications.recv_timeout(std::time::Duration::from_millis(20)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ),
            "idle retirement must not start an unavailable-source repaint loop"
        );
    }
}
