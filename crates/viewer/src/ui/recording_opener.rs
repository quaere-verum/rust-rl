//! Perform potentially slow catalogue scans without blocking the interface.
use super::{CatalogEntry, ReadLimits, RunIndex, catalog, egui};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

pub(super) struct OpenedRecording {
    pub path: PathBuf,
    pub entries: Vec<CatalogEntry>,
    pub run: RunIndex,
    pub warnings: Vec<String>,
}

pub(super) fn prepare(path: &Path) -> Result<OpenedRecording, String> {
    let (entries, mut errors) = catalog(path).map_err(|error| error.to_string())?;
    prepare_entries(path, entries, &mut errors, &|| false)
}

fn prepare_interactive(
    path: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<OpenedRecording, String> {
    let (entries, mut errors) =
        crate::data::interactive_catalog(path, cancelled).map_err(|error| error.to_string())?;
    prepare_entries(path, entries, &mut errors, cancelled)
}

fn prepare_entries(
    path: &Path,
    entries: Vec<CatalogEntry>,
    errors: &mut Vec<String>,
    cancelled: &dyn Fn() -> bool,
) -> Result<OpenedRecording, String> {
    for entry in &entries {
        if cancelled() {
            return Err("Recording opening cancelled".into());
        }
        match RunIndex::open(&entry.root, ReadLimits::default()) {
            Ok(run) => {
                return Ok(OpenedRecording {
                    path: path.to_owned(),
                    entries,
                    run,
                    warnings: std::mem::take(errors),
                });
            }
            Err(error) => errors.push(format!("{}: {error}", entry.root.display())),
        }
    }
    Err(if errors.is_empty() {
        "No recordings found in this directory".into()
    } else {
        errors.join("\n")
    })
}

pub(super) struct RecordingOpener {
    requests: mpsc::SyncSender<(u64, PathBuf)>,
    results: mpsc::Receiver<(u64, Result<OpenedRecording, String>)>,
    generation: Arc<AtomicU64>,
    pending: Option<PathBuf>,
    active: bool,
}

impl RecordingOpener {
    pub fn new(ctx: egui::Context) -> std::io::Result<Self> {
        Self::with_prepare(ctx, prepare_interactive)
    }

    pub(super) fn with_prepare(
        ctx: egui::Context,
        prepare: impl Fn(&Path, &dyn Fn() -> bool) -> Result<OpenedRecording, String> + Send + 'static,
    ) -> std::io::Result<Self> {
        let (requests, input) = mpsc::sync_channel::<(u64, PathBuf)>(1);
        let (output, results) = mpsc::sync_channel(1);
        let generation = Arc::new(AtomicU64::new(0));
        let worker_generation = generation.clone();
        std::thread::Builder::new()
            .name("rl-recording-opener".into())
            .spawn(move || {
                while let Ok((generation, path)) = input.recv() {
                    if worker_generation.load(Ordering::Acquire) != generation {
                        continue;
                    }
                    let cancelled = || worker_generation.load(Ordering::Acquire) != generation;
                    let result = prepare(&path, &cancelled);
                    if cancelled() {
                        continue;
                    }
                    if output.send((generation, result)).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })?;
        Ok(Self {
            requests,
            results,
            generation,
            pending: None,
            active: false,
        })
    }

    pub fn request(&mut self, path: PathBuf) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.pending = Some(path);
        self.active = true;
    }

    pub fn cancel(&mut self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.pending = None;
        self.active = false;
    }

    pub fn poll(&mut self, ctx: &egui::Context) -> Option<Result<OpenedRecording, String>> {
        let mut latest = None;
        while let Ok((generation, result)) = self.results.try_recv() {
            if generation == self.generation.load(Ordering::Acquire) {
                self.active = false;
                latest = Some(result);
            }
        }
        if let Some(path) = self.pending.take() {
            match self
                .requests
                .try_send((self.generation.load(Ordering::Acquire), path))
            {
                Ok(()) => (),
                Err(mpsc::TrySendError::Full((_, path))) => self.pending = Some(path),
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.active = false;
                    latest = Some(Err("Recording opener unavailable".into()));
                }
            }
        }
        if self.active {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        latest
    }

    pub fn active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
#[test]
fn slow_open_is_nonblocking_and_latest_request_wins() {
    use std::time::{Duration, Instant};
    let ctx = egui::Context::default();
    let (entered, wait_entered) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let mut worker = RecordingOpener::with_prepare(ctx.clone(), move |path, _cancelled| {
        assert_ne!(
            path,
            Path::new("obsolete"),
            "superseded request was scanned"
        );
        if path == Path::new("slow") {
            entered.send(()).unwrap();
            gate.recv().unwrap();
        }
        Err(path.display().to_string())
    })
    .unwrap();
    worker.request("slow".into());
    assert!(worker.poll(&ctx).is_none());
    wait_entered.recv_timeout(Duration::from_secs(2)).unwrap();
    worker.request("obsolete".into());
    assert!(worker.poll(&ctx).is_none());
    worker.request("new".into());
    assert!(worker.poll(&ctx).is_none());
    release.send(()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(result) = worker.poll(&ctx) {
            assert!(matches!(result, Err(error) if error == "new"));
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
}
