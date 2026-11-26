//! Bounded selected-state preparation for numerical and rendered episodes.
//! The paint pass consumes one ready frame per active episode.
use crate::data::CacheKey;
use eframe::egui::{Pos2, Rect};
use rl::{
    Error, Result,
    diagnostics::{
        render::{RenderBounds, RenderLimits, RenderPrimitive, RenderScene, ValidatedRenderTrack},
        schema::{Chunk, DType, Record},
    },
};
use std::sync::{
    Arc, Condvar, Mutex, Weak,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

mod playback;
pub(crate) use playback::FrameTrace;
pub use playback::{EpisodeEnd, ExecutedAction, ObservedRow, PlaybackState};

/// Persistent worker/mailbox, bounded episode traces, identity and error metadata.
pub const CONTROL_BYTES: usize = 64 * 1024;
pub const MAX_RAW_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_SCENE_BYTES: usize = 512 * 1024;
/// One retained, shared set of bounded episode telemetry traces.
pub const MAX_TELEMETRY_BYTES: usize = 192 * 1024;
pub const MAX_OWNER_BYTES: usize = 72 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneIdentity {
    pub session: u64,
    pub key: CacheKey,
    pub revision: u64,
    pub role: String,
    pub step: usize,
}
#[derive(Clone, Debug)]
pub struct ReadoutSpec {
    pub key: String,
    pub component: usize,
    pub label: String,
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct SceneMetrics {
    pub generation: u64,
    pub decodes: u64,
    pub cancellations: u64,
    pub owned_bytes: usize,
    pub peak_bytes: usize,
}
#[derive(Default)]
struct Accounting {
    bytes: AtomicUsize,
    peak: AtomicUsize,
    decodes: AtomicU64,
    cancellations: AtomicU64,
}
struct Lease {
    owner: Arc<Accounting>,
    bytes: usize,
}
impl Lease {
    fn new(owner: &Arc<Accounting>, bytes: usize) -> Result<Self> {
        let old = owner
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes).filter(|n| *n <= MAX_OWNER_BYTES)
            })
            .map_err(|_| {
                Error::Shape("Environment retained scene ownership exceeds budget".into())
            })?;
        owner.peak.fetch_max(old + bytes, Ordering::AcqRel);
        Ok(Self {
            owner: owner.clone(),
            bytes,
        })
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.owner.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub struct PreparedScene {
    pub identity: SceneIdentity,
    pub scene: Option<RenderScene>,
    pub playback: PlaybackState,
    pub readouts: Vec<(String, f64)>,
    pub(crate) frame_readouts: Vec<(String, usize, f64)>,
    pub(crate) telemetry: Option<Arc<Telemetry>>,
    pub(crate) rewards: Option<Arc<[egui_plot::PlotPoint]>>,
    source: Weak<Chunk>,
    _lease: Lease,
}
impl PreparedScene {
    pub(crate) fn step_at_time(&self, time: f64) -> Option<usize> {
        if !time.is_finite()
            || !self
                .telemetry
                .as_ref()
                .is_some_and(|telemetry| telemetry.physical_time)
        {
            return None;
        }
        let chunk = self.source.upgrade()?;
        let times = playback::recorded_times(&chunk)?;
        let upper = times.partition_point(time);
        if upper == 0 {
            return Some(0);
        }
        if upper == times.len() {
            return Some(times.len() - 1);
        }
        Some(
            if time - times.get(upper - 1)? <= times.get(upper)? - time {
                upper - 1
            } else {
                upper
            },
        )
    }
}
pub(crate) struct Telemetry {
    pub observations: Vec<Vec<egui_plot::PlotPoint>>,
    pub actions: Vec<Vec<egui_plot::PlotPoint>>,
    pub frames: Vec<playback::FrameTrace>,
    pub physical_time: bool,
    bytes: usize,
    _lease: Lease,
}
struct PreparedTrack {
    render: ValidatedRenderTrack,
    rewards: Option<Arc<[egui_plot::PlotPoint]>>,
    telemetry: Weak<Telemetry>,
}
struct Job {
    generation: u64,
    identity: SceneIdentity,
    chunk: Arc<Chunk>,
    readouts: Vec<ReadoutSpec>,
    _raw: Lease,
    scene: Lease,
}
#[derive(Default)]
struct Mailbox {
    desired: Option<Job>,
    answer: Option<(u64, std::result::Result<Arc<PreparedScene>, String>)>,
    check: Option<(u64, std::path::PathBuf)>,
    checked: Option<(u64, bool)>,
    shutdown: bool,
}
struct Shared {
    mailbox: Mutex<Mailbox>,
    changed: Condvar,
    generation: AtomicU64,
    owner: Arc<Accounting>,
}
pub struct SceneWorker {
    shared: Arc<Shared>,
    _control: Lease,
    desired: Option<SceneIdentity>,
    desired_owner: Weak<Chunk>,
    frame: Option<Arc<PreparedScene>>,
    error: Option<String>,
    pending: bool,
}
impl SceneWorker {
    pub fn new(wake: impl Fn() + Send + 'static) -> Result<Self> {
        let mut track = None;
        Self::with_preparer(wake, move |job| prepare(job, &mut track))
    }
    fn with_preparer(
        wake: impl Fn() + Send + 'static,
        prepare: impl FnMut(Job) -> Result<PreparedScene> + Send + 'static,
    ) -> Result<Self> {
        Self::with_owner(Arc::new(Accounting::default()), wake, prepare)
    }

    pub(crate) fn peer(&self, wake: impl Fn() + Send + 'static) -> Result<Self> {
        let mut track = None;
        Self::with_owner(self.shared.owner.clone(), wake, move |job| {
            prepare(job, &mut track)
        })
    }

    fn with_owner(
        owner: Arc<Accounting>,
        wake: impl Fn() + Send + 'static,
        mut prepare: impl FnMut(Job) -> Result<PreparedScene> + Send + 'static,
    ) -> Result<Self> {
        let shared = Arc::new(Shared {
            mailbox: Mutex::new(Mailbox::default()),
            changed: Condvar::new(),
            generation: AtomicU64::new(0),
            owner,
        });
        let control = Lease::new(&shared.owner, CONTROL_BYTES)?;
        let worker = shared.clone();
        std::thread::Builder::new()
            .name("rl-environment-scene".into())
            .spawn(move || {
                loop {
                    let job = {
                        let mut mailbox = worker.mailbox.lock().unwrap_or_else(|e| e.into_inner());
                        while mailbox.desired.is_none()
                            && mailbox.check.is_none()
                            && !mailbox.shutdown
                        {
                            mailbox = worker
                                .changed
                                .wait(mailbox)
                                .unwrap_or_else(|e| e.into_inner());
                        }
                        if mailbox.shutdown {
                            return;
                        }
                        if let Some(job) = mailbox.desired.take() {
                            Some(job)
                        } else {
                            let (generation, path) =
                                mailbox.check.take().expect("availability request");
                            drop(mailbox);
                            let available = path.is_file();
                            let mut mailbox =
                                worker.mailbox.lock().unwrap_or_else(|e| e.into_inner());
                            if generation == worker.generation.load(Ordering::Acquire)
                                && !mailbox.shutdown
                            {
                                mailbox.checked = Some((generation, available));
                            }
                            drop(mailbox);
                            wake();
                            None
                        }
                    };
                    let Some(job) = job else {
                        continue;
                    };
                    let generation = job.generation;
                    if worker.generation.load(Ordering::Acquire) != generation {
                        worker.owner.cancellations.fetch_add(1, Ordering::AcqRel);
                        continue;
                    }
                    worker.owner.decodes.fetch_add(1, Ordering::AcqRel);
                    let result = prepare(job)
                        .map(Arc::new)
                        .map_err(|e| e.to_string().chars().take(1024).collect::<String>());
                    let mut mailbox = worker.mailbox.lock().unwrap_or_else(|e| e.into_inner());
                    if worker.generation.load(Ordering::Acquire) != generation || mailbox.shutdown {
                        worker.owner.cancellations.fetch_add(1, Ordering::AcqRel);
                    } else {
                        mailbox.answer = Some((generation, result));
                    }
                    drop(mailbox);
                    wake();
                }
            })?;
        Ok(Self {
            shared,
            _control: control,
            desired: None,
            desired_owner: Weak::new(),
            frame: None,
            error: None,
            pending: false,
        })
    }
    /// Admit the raw pin and maximum decoded result before cloning the chunk.
    /// Raw Loader cache residency does not waive this independent ownership.
    pub fn submit(
        &mut self,
        identity: SceneIdentity,
        chunk: &Arc<Chunk>,
        readouts: Vec<ReadoutSpec>,
    ) -> Result<()> {
        if self.desired.as_ref() == Some(&identity)
            && self.desired_owner.as_ptr() == Arc::as_ptr(chunk)
        {
            return Ok(());
        }
        if identity.role.len() > 128
            || identity.role.capacity() > 1024
            || identity.key.root.capacity() > 8 * 1024
            || identity.key.root.as_os_str().len() > 4096
            || identity.revision != chunk.context.identity.learner_update_revision
            || identity.role != chunk.context.identity.model_role
        {
            return Err(Error::State(
                "Environment scene identity differs from selected checkpoint/role".into(),
            ));
        }
        let readout_bytes = readouts
            .capacity()
            .saturating_mul(std::mem::size_of::<ReadoutSpec>())
            .saturating_add(
                readouts
                    .iter()
                    .map(|r| r.key.capacity().saturating_add(r.label.capacity()))
                    .sum::<usize>(),
            );
        if readout_bytes > 16 * 1024
            || readouts.len() > 6
            || readouts
                .iter()
                .any(|r| r.key.len() > 1024 || r.label.len() > 2048)
        {
            return Err(Error::Shape(
                "Environment readout metadata exceeds budget".into(),
            ));
        }
        let charge = chunk.retained_charge();
        if charge > MAX_RAW_BYTES {
            return Err(Error::Shape(
                "Environment raw trajectory exceeds 32 MiB admission".into(),
            ));
        }
        // Keep a published scene only while preparing another state from the
        // same immutable episode. Its lease stays charged until replacement.
        let previous = self.frame.take().filter(|frame| {
            let old = &frame.identity;
            old.session == identity.session
                && old.key == identity.key
                && old.revision == identity.revision
                && old.role == identity.role
                && frame.source.as_ptr() == Arc::as_ptr(chunk)
        });
        self.retire();
        let raw = Lease::new(&self.shared.owner, charge + 32 * 1024)?;
        let scene = Lease::new(&self.shared.owner, MAX_SCENE_BYTES)?;
        self.frame = previous;
        let generation = self.shared.generation.load(Ordering::Acquire);
        let job = Job {
            generation,
            identity: identity.clone(),
            chunk: chunk.clone(),
            readouts,
            _raw: raw,
            scene,
        };
        self.shared
            .mailbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .desired = Some(job);
        self.desired = Some(identity);
        self.desired_owner = Arc::downgrade(chunk);
        self.pending = true;
        self.shared.changed.notify_one();
        Ok(())
    }
    /// Check only the already selected immutable index path on this worker.
    /// No payload decode or UI-thread filesystem operation is required.
    pub fn check_path(&mut self, path: std::path::PathBuf) -> Result<()> {
        if path.capacity() > 8192 {
            return Err(Error::Shape(
                "Environment source path exceeds admission".into(),
            ));
        }
        if self.desired.is_some() {
            self.shared
                .mailbox
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .check = Some((self.shared.generation.load(Ordering::Acquire), path));
            self.shared.changed.notify_one();
        }
        Ok(())
    }
    pub fn poll(&mut self) {
        let checked = self
            .shared
            .mailbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .checked
            .take();
        if let Some((generation, false)) = checked
            && generation == self.shared.generation.load(Ordering::Acquire)
        {
            self.retire();
            self.error = Some("Selected immutable render source file is missing".into());
            return;
        }
        let answer = self
            .shared
            .mailbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .answer
            .take();
        if let Some((generation, result)) = answer
            && generation == self.shared.generation.load(Ordering::Acquire)
        {
            self.pending = false;
            match result {
                Ok(frame) => {
                    if self.desired.as_ref() == Some(&frame.identity)
                        && self.desired_owner.ptr_eq(&frame.source)
                    {
                        self.frame = Some(frame);
                        self.error = None;
                    }
                }
                Err(error) => {
                    self.frame = None;
                    self.error = Some(error);
                }
            }
        }
    }
    pub fn retire(&mut self) {
        if self.desired.is_none() && self.frame.is_none() && !self.pending && self.error.is_none() {
            return;
        }
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        let mut mailbox = self
            .shared
            .mailbox
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if mailbox.desired.take().is_some() {
            self.shared
                .owner
                .cancellations
                .fetch_add(1, Ordering::AcqRel);
        }
        mailbox.answer = None;
        mailbox.check = None;
        mailbox.checked = None;
        drop(mailbox);
        self.desired = None;
        self.desired_owner = Weak::new();
        self.frame = None;
        self.error = None;
        self.pending = false;
    }
    pub fn frame(&self) -> Option<&Arc<PreparedScene>> {
        self.frame.as_ref()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn pending(&self) -> bool {
        self.pending
    }
    pub fn metrics(&self) -> SceneMetrics {
        let a = &self.shared.owner;
        SceneMetrics {
            generation: self.shared.generation.load(Ordering::Acquire),
            decodes: a.decodes.load(Ordering::Acquire),
            cancellations: a.cancellations.load(Ordering::Acquire),
            owned_bytes: a.bytes.load(Ordering::Acquire),
            peak_bytes: a.peak.load(Ordering::Acquire),
        }
    }
}
impl Drop for SceneWorker {
    fn drop(&mut self) {
        self.retire();
        self.shared
            .mailbox
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .shutdown = true;
        self.shared.changed.notify_one();
    }
}

fn unique_frame_channel<'a>(
    frames: &'a std::collections::BTreeMap<String, String>,
    key: &str,
) -> Option<&'a str> {
    let mut channels = frames
        .iter()
        .filter(|(name, source)| name.len() <= 128 && source.as_str() == key);
    let (name, _) = channels.next()?;
    channels.next().is_none().then_some(name.as_str())
}

fn prepare(job: Job, track: &mut Option<PreparedTrack>) -> Result<PreparedScene> {
    if track
        .as_ref()
        .is_none_or(|proof| !proof.render.matches(&job.chunk))
    {
        *track = None;
        *track = Some(PreparedTrack {
            render: ValidatedRenderTrack::new(&job.chunk, RenderLimits::default())?,
            rewards: None,
            telemetry: Weak::new(),
        });
    }
    let track = track.as_mut().expect("validated track");
    let mut scene = track.render.scene_at(&job.chunk, job.identity.step)?;
    if scene.is_none() && track.rewards.is_none() {
        track.rewards = Some(playback::episode_rewards(&job.chunk)?.into());
    }
    // The base charge covers the bounded numeric rows and descriptor readouts.
    let mut charge = 32 * 1024;
    if let Some(scene) = &mut scene {
        scene.items.sort_by_key(|item| item.layer);
        charge += std::mem::size_of::<RenderScene>()
            + scene.items.capacity() * std::mem::size_of::<rl::diagnostics::render::RenderItem>();
        for item in &scene.items {
            charge = charge.saturating_add(match &item.primitive {
                RenderPrimitive::Polyline { points, .. }
                | RenderPrimitive::Polygon { points, .. } => {
                    points.capacity() * std::mem::size_of::<[f64; 2]>()
                }
                RenderPrimitive::Text { text, .. } => text.capacity(),
                _ => 0,
            });
        }
    }
    if charge > MAX_SCENE_BYTES {
        return Err(Error::Shape(
            "Environment decoded scene exceeds ownership admission".into(),
        ));
    }
    // Retain one bounded, separately charged set of extrema across nearby
    // episode states. The weak track cache releases it with the last scene.
    let telemetry = {
        let cached = track.telemetry.upgrade();
        let prepared = if let Some(cached) = cached {
            Some(cached)
        } else if let Ok(lease) = Lease::new(&job.scene.owner, MAX_TELEMETRY_BYTES) {
            let mut observations = playback::observation_traces(&job.chunk)?;
            let mut actions = playback::action_traces(&job.chunk)?;
            let mut frames = playback::frame_traces(&job.chunk)?;
            let times = playback::simulation_times(&job.chunk);
            if let Some(times) = times {
                for trace in observations.iter_mut().chain(&mut actions) {
                    playback::convert_to_time(trace, times);
                }
                for trace in &mut frames {
                    playback::convert_to_time(&mut trace.points, times);
                }
            }
            let bytes = (observations.capacity() + actions.capacity())
                * std::mem::size_of::<Vec<egui_plot::PlotPoint>>()
                + observations
                    .iter()
                    .chain(&actions)
                    .map(|trace| trace.capacity() * std::mem::size_of::<egui_plot::PlotPoint>())
                    .sum::<usize>()
                + frames.capacity() * std::mem::size_of::<playback::FrameTrace>()
                + frames
                    .iter()
                    .map(|trace| {
                        trace.name.capacity()
                            + trace.points.capacity() * std::mem::size_of::<egui_plot::PlotPoint>()
                    })
                    .sum::<usize>();
            if bytes <= MAX_TELEMETRY_BYTES {
                let telemetry = Arc::new(Telemetry {
                    observations,
                    actions,
                    frames,
                    physical_time: times.is_some(),
                    bytes,
                    _lease: lease,
                });
                track.telemetry = Arc::downgrade(&telemetry);
                Some(telemetry)
            } else {
                None
            }
        } else {
            None
        };
        prepared.filter(|telemetry| charge.saturating_add(telemetry.bytes) <= MAX_SCENE_BYTES)
    };
    let mut readouts = Vec::with_capacity(job.readouts.len());
    let mut frame_readouts = Vec::with_capacity(job.readouts.len());
    if let Record::Trajectory(record) = &job.chunk.record {
        for spec in job.readouts {
            if !record.frames.values().any(|key| key == &spec.key) {
                continue;
            }
            let Some(array) = job.chunk.arrays.get(&spec.key) else {
                continue;
            };
            let width = if array.shape().len() == 2 {
                array.shape()[1]
            } else {
                1
            };
            if spec.component >= width {
                continue;
            }
            let Some(index) = job
                .identity
                .step
                .checked_mul(width)
                .and_then(|i| i.checked_add(spec.component))
            else {
                continue;
            };
            let value = match array.dtype() {
                DType::F32 => array.as_f32()?.get(index).map(|v| f64::from(*v)),
                DType::F64 => array.as_f64()?.get(index).copied(),
                _ => None,
            };
            if let Some(value) = value.filter(|v| v.is_finite()) {
                if let Some(name) = unique_frame_channel(&record.frames, &spec.key) {
                    frame_readouts.push((name.to_owned(), spec.component, value));
                }
                readouts.push((spec.label, value));
            }
        }
    }
    let playback = PlaybackState::new(&job.chunk, job.identity.step)?;
    Ok(PreparedScene {
        identity: job.identity,
        scene,
        playback,
        readouts,
        frame_readouts,
        telemetry,
        rewards: track.rewards.clone(),
        source: Arc::downgrade(&job.chunk),
        _lease: job.scene,
    })
}
/// Uniform world-to-screen transform with y increasing upward. All returned
/// coordinates are bounded before conversion to egui f32 geometry.
pub struct SceneTransform {
    center: [f64; 2],
    screen: [f64; 2],
    scale: f64,
}
impl SceneTransform {
    pub fn new(bounds: &RenderBounds, rect: Rect) -> Option<Self> {
        let width = bounds.max[0] - bounds.min[0];
        let height = bounds.max[1] - bounds.min[1];
        if !width.is_finite()
            || !height.is_finite()
            || width <= 0.
            || height <= 0.
            || !rect.is_finite()
            || rect.width() <= 0.
            || rect.height() <= 0.
        {
            return None;
        }
        let scale = (f64::from(rect.width()) / width).min(f64::from(rect.height()) / height);
        if !scale.is_finite() || scale <= 0. {
            return None;
        }
        Some(Self {
            center: [bounds.min[0] + width / 2., bounds.min[1] + height / 2.],
            screen: [f64::from(rect.center().x), f64::from(rect.center().y)],
            scale,
        })
    }
    fn projected(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.screen[0] + (p[0] - self.center[0]) * self.scale,
            self.screen[1] - (p[1] - self.center[1]) * self.scale,
        ]
    }
    pub fn point(&self, p: [f64; 2]) -> Pos2 {
        let [x, y] = self.projected(p);
        Pos2::new(x as f32, y as f32)
    }
    pub fn radius(&self, r: f64) -> f32 {
        (r * self.scale) as f32
    }
    /// Reject rather than clamp geometry whose projection exceeds the bounded
    /// drawing range. Ordinary offscreen geometry is clipped by the painter.
    pub fn for_scene(scene: &RenderScene, rect: Rect) -> Result<Self> {
        let t = Self::new(&scene.bounds, rect).ok_or_else(|| {
            Error::Shape("Environment projection scale is outside supported range".into())
        })?;
        let check = |p| -> Result<()> {
            if t.projected(p)
                .iter()
                .any(|v| !v.is_finite() || v.abs() > 1e6)
            {
                Err(Error::Shape(
                    "Environment geometry projects outside the supported drawing range".into(),
                ))
            } else {
                Ok(())
            }
        };
        for item in &scene.items {
            match &item.primitive {
                RenderPrimitive::Line { from, to, .. } => {
                    check(*from)?;
                    check(*to)?;
                }
                RenderPrimitive::Polyline { points, .. }
                | RenderPrimitive::Polygon { points, .. } => {
                    for point in points {
                        check(*point)?;
                    }
                }
                RenderPrimitive::Rectangle { min, max, .. } => {
                    check(*min)?;
                    check(*max)?;
                }
                RenderPrimitive::Text { position, .. } => check(*position)?,
                RenderPrimitive::Circle { center, radius, .. } => {
                    check(*center)?;
                    let r = radius * t.scale;
                    if !r.is_finite() || r > 1e6 {
                        return Err(Error::Shape(
                            "Environment circle projects outside the supported drawing range"
                                .into(),
                        ));
                    }
                }
            }
        }
        Ok(t)
    }
}

#[cfg(test)]
mod worker_tests {
    use super::*;
    use rl::diagnostics::schema::{MetricRecord, RecordContext};
    use std::{
        collections::BTreeMap,
        sync::mpsc,
        time::{Duration, Instant},
    };
    #[test]
    fn named_readout_never_substitutes_an_ambiguous_frame_source() {
        let mut frames = BTreeMap::from([("horizontal".to_string(), "source".to_string())]);
        assert_eq!(unique_frame_channel(&frames, "source"), Some("horizontal"));
        frames.insert("alias".into(), "source".into());
        assert_eq!(unique_frame_channel(&frames, "source"), None);
        assert_eq!(unique_frame_channel(&frames, "missing"), None);
    }
    #[test]
    fn cancelled_active_read_retains_its_pin_after_cache_eviction_then_releases() {
        let (started, starts) = mpsc::sync_channel(1);
        let (release, releases) = mpsc::sync_channel(1);
        let mut worker = SceneWorker::with_preparer(
            || {},
            move |_job| {
                started.send(()).unwrap();
                releases.recv().unwrap();
                Err(Error::State("cancelled test job".into()))
            },
        )
        .unwrap();
        let chunk = Arc::new(Chunk {
            context: RecordContext::new("test", 1),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::new(),
        });
        let weak = Arc::downgrade(&chunk);
        let identity = SceneIdentity {
            session: 0,
            key: CacheKey {
                root: Arc::new("test".into()),
                sequence: 1,
            },
            revision: 1,
            role: "online".into(),
            step: 0,
        };
        worker.submit(identity, &chunk, vec![]).unwrap();
        starts.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(chunk);
        worker.retire();
        assert!(weak.upgrade().is_some());
        assert!(worker.metrics().owned_bytes >= MAX_SCENE_BYTES);
        release.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while worker.metrics().owned_bytes > CONTROL_BYTES {
            worker.poll();
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(1));
        }
        worker.poll();
        assert!(weak.upgrade().is_none());
        assert!(worker.frame().is_none());
        assert!(worker.error().is_none());
        assert_eq!(worker.metrics().cancellations, 1);
    }
}
