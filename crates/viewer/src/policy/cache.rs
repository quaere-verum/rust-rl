//! Prepared-only background provider. This module has no model restoration or
//! inference entry point. Catalog integrity, disk reads, decoding, projections
//! and selected-distribution preparation all happen on its worker.
use super::prepared::{PreparedGrid, PreparedPolicyFrame, cached_frame};
use crate::data::RunIndex;
use rl::{
    Error, Result,
    diagnostics::{
        policy_cache::{CacheEntry, CacheLimits, PolicyCache},
        policy_surface::PolicySurface,
    },
};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
pub struct PreparedLimits {
    pub max_bytes: usize,
    pub max_frames: usize,
    pub prefetch: usize,
    pub read: CacheLimits,
}
impl Default for PreparedLimits {
    fn default() -> Self {
        Self {
            max_bytes: 128 * 1024 * 1024,
            max_frames: 8,
            prefetch: 2,
            read: CacheLimits {
                max_frame_bytes: 64 * 1024 * 1024,
                max_chunk_bytes: 16 * 1024 * 1024,
                max_surface_bytes: 32 * 1024 * 1024,
                max_index_bytes: 32 * 1024 * 1024,
                ..Default::default()
            },
        }
    }
}
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct PreparedMetrics {
    pub loads: u64,
    pub cache_hits: u64,
    pub preparations: u64,
    pub load_micros: u64,
    pub read_micros: u64,
    pub decode_micros: u64,
    pub preparation_micros: u64,
    pub catalog_micros: u64,
    pub catalog_checked: usize,
    pub catalog_total: usize,
    pub retained_bytes: usize,
    pub peak_retained_bytes: usize,
    pub prefetch_failures: u64,
}
#[derive(Default)]
struct Accounting {
    retained: AtomicUsize,
    peak: AtomicUsize,
    loads: AtomicU64,
    hits: AtomicU64,
    preparations: AtomicU64,
    load_micros: AtomicU64,
    read_micros: AtomicU64,
    decode_micros: AtomicU64,
    preparation_micros: AtomicU64,
    catalog_micros: AtomicU64,
    catalog_checked: AtomicUsize,
    catalog_total: AtomicUsize,
    prefetch_failures: AtomicU64,
}
pub(crate) struct CacheLease {
    accounting: Arc<Accounting>,
    bytes: usize,
}
impl CacheLease {
    fn acquire(accounting: &Arc<Accounting>, bytes: usize, limit: usize) -> Result<Self> {
        let mut old = accounting.retained.load(Ordering::Acquire);
        loop {
            let Some(next) = old.checked_add(bytes).filter(|n| *n <= limit) else {
                return Err(Error::Config(
                    "Prepared Policy retained memory budget exhausted".into(),
                ));
            };
            match accounting.retained.compare_exchange_weak(
                old,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    accounting.peak.fetch_max(next, Ordering::AcqRel);
                    return Ok(Self {
                        accounting: accounting.clone(),
                        bytes,
                    });
                }
                Err(current) => old = current,
            }
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}
impl Drop for CacheLease {
    fn drop(&mut self) {
        self.accounting
            .retained
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub(crate) struct LoadedSurface {
    pub entry: Arc<CacheEntry>,
    pub surface: PolicySurface,
    _lease: CacheLease,
}
#[derive(Debug)]
pub struct PreparedCatalog {
    pub entries: Vec<Arc<CacheEntry>>,
    pub issues: Vec<String>,
    by_key: BTreeMap<String, usize>,
}
impl PreparedCatalog {
    pub fn get(&self, key: &str) -> Option<&Arc<CacheEntry>> {
        self.by_key.get(key).map(|i| &self.entries[*i])
    }
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FrameRequest {
    pub key: String,
    pub reference: Option<String>,
    pub action: usize,
    pub row: usize,
    pub domain: Option<[f64; 2]>,
}
impl FrameRequest {
    fn same_grid(&self, other: &Self) -> bool {
        self.key == other.key && self.reference == other.reference && self.action == other.action
    }
}
struct Job {
    generation: u64,
    request: FrameRequest,
    prefetch: Vec<String>,
}
#[derive(Default)]
struct Answer {
    catalog: Option<Result<Arc<PreparedCatalog>>>,
    frame: Option<(u64, Result<PreparedPolicyFrame>)>,
}
struct Shared {
    job: Mutex<Option<Job>>,
    wake: Condvar,
    answer: Mutex<Answer>,
    generation: AtomicU64,
    stop: AtomicBool,
    retired: AtomicBool,
}
impl Shared {
    fn cancelled(&self, generation: u64) -> bool {
        self.stop.load(Ordering::Acquire) || self.generation.load(Ordering::Acquire) != generation
    }
}
pub struct PreparedProvider {
    shared: Arc<Shared>,
    accounting: Arc<Accounting>,
    catalog: Option<Arc<PreparedCatalog>>,
    catalog_pending: bool,
    desired: Option<FrameRequest>,
    frame: Option<PreparedPolicyFrame>,
    error: Option<String>,
    pending: bool,
    ready: bool,
    limits: PreparedLimits,
}
impl PreparedProvider {
    pub fn new(
        run: RunIndex,
        path: PathBuf,
        limits: PreparedLimits,
        wake: impl Fn() + Send + 'static,
    ) -> Result<Self> {
        if limits.max_bytes == 0 || limits.max_frames == 0 {
            return Err(Error::Config(
                "Prepared Policy memory and frame limits must be positive".into(),
            ));
        }
        let shared = Arc::new(Shared {
            job: Mutex::new(None),
            wake: Condvar::new(),
            answer: Mutex::new(Answer::default()),
            generation: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            retired: AtomicBool::new(false),
        });
        let accounting = Arc::new(Accounting::default());
        let worker_shared = shared.clone();
        let worker_accounting = accounting.clone();
        std::thread::Builder::new()
            .name("prepared-policy".into())
            .spawn(move || {
                worker(
                    &run,
                    path,
                    limits,
                    &worker_shared,
                    &worker_accounting,
                    &wake,
                );
                worker_shared.retired.store(true, Ordering::Release);
                wake();
            })?;
        Ok(Self {
            shared,
            accounting,
            catalog: None,
            catalog_pending: true,
            desired: None,
            frame: None,
            error: None,
            pending: false,
            ready: false,
            limits,
        })
    }
    pub fn poll(&mut self) {
        let mut answer = self.shared.answer.lock().unwrap();
        if let Some(catalog) = answer.catalog.take() {
            self.catalog_pending = false;
            match catalog {
                Ok(c) => self.catalog = Some(c),
                Err(e) => self.error = Some(e.to_string()),
            }
        }
        if let Some((generation, frame)) = answer.frame.take()
            && !self.shared.cancelled(generation)
        {
            self.pending = false;
            match frame {
                Ok(frame) => {
                    self.frame = Some(frame);
                    self.ready = true;
                    self.error = None;
                }
                Err(e) => {
                    self.error = Some(e.to_string());
                    self.ready = false;
                }
            }
        }
    }
    pub fn catalog(&self) -> Option<&PreparedCatalog> {
        self.catalog.as_deref()
    }
    pub fn catalog_pending(&self) -> bool {
        self.catalog_pending
    }
    pub fn frame(&self) -> Option<&PreparedPolicyFrame> {
        self.frame.as_ref()
    }
    pub fn ready(&self) -> bool {
        self.ready
    }
    pub fn pending(&self) -> bool {
        self.pending
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    /// The initial Gaussian display domain is already prepared in the frame.
    /// Retaining that exact domain changes no numerical or plotted values and
    /// must not schedule another identical point preparation on an idle paint.
    pub(crate) fn retain_domain(&mut self, domain: [f64; 2]) {
        if self.ready
            && self
                .frame
                .as_ref()
                .and_then(|f| f.selected.as_ref())
                .and_then(|p| p.plot.as_ref())
                .is_some_and(|p| p.domain == domain)
            && let Some(desired) = &mut self.desired
        {
            desired.domain = Some(domain);
        }
    }
    pub fn request(
        &mut self,
        key: &str,
        action: usize,
        row: usize,
        reference: Option<&str>,
        domain: Option<[f64; 2]>,
        prefetch: &[String],
    ) -> Result<()> {
        let valid = self
            .catalog
            .as_ref()
            .and_then(|c| c.get(key))
            .is_some_and(|e| row < e.rows)
            && reference.is_none_or(|r| self.catalog.as_ref().is_some_and(|c| c.get(r).is_some()))
            && domain.is_none_or(|[l, h]| l.is_finite() && h.is_finite() && l < h);
        if !valid || self.shared.stop.load(Ordering::Acquire) {
            self.cancel();
            self.error =
                Some("Requested prepared slice/checkpoint/grid point is unavailable".into());
            return Err(Error::State(self.error.clone().unwrap()));
        }
        let request = FrameRequest {
            key: key.into(),
            reference: reference.map(str::to_owned),
            action,
            row,
            domain,
        };
        if self.desired.as_ref() == Some(&request)
            && (self.pending || self.ready || self.error.is_some())
        {
            return Ok(());
        }
        let generation = self.shared.generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.desired = Some(request.clone());
        self.pending = true;
        self.ready = false;
        self.error = None;
        *self.shared.job.lock().unwrap() = Some(Job {
            generation,
            request,
            prefetch: prefetch
                .iter()
                .filter(|key| self.catalog.as_ref().is_some_and(|c| c.get(key).is_some()))
                .take(self.limits.prefetch.min(self.limits.max_frames))
                .cloned()
                .collect(),
        });
        self.shared.wake.notify_one();
        Ok(())
    }
    pub fn cancel(&mut self) {
        self.shared.generation.fetch_add(1, Ordering::AcqRel);
        *self.shared.job.lock().unwrap() = None;
        self.shared.answer.lock().unwrap().frame = None;
        self.desired = None;
        self.pending = false;
        self.ready = false;
    }
    pub fn retire(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.cancel();
        self.frame = None;
        self.catalog = None;
        self.catalog_pending = false;
        self.shared.answer.lock().unwrap().catalog = None;
        self.shared.wake.notify_one();
    }
    pub fn retired(&self) -> bool {
        self.shared.retired.load(Ordering::Acquire)
    }
    pub fn metrics(&self) -> PreparedMetrics {
        let a = &self.accounting;
        PreparedMetrics {
            loads: a.loads.load(Ordering::Acquire),
            cache_hits: a.hits.load(Ordering::Acquire),
            preparations: a.preparations.load(Ordering::Acquire),
            load_micros: a.load_micros.load(Ordering::Acquire),
            read_micros: a.read_micros.load(Ordering::Acquire),
            decode_micros: a.decode_micros.load(Ordering::Acquire),
            preparation_micros: a.preparation_micros.load(Ordering::Acquire),
            catalog_micros: a.catalog_micros.load(Ordering::Acquire),
            catalog_checked: a.catalog_checked.load(Ordering::Acquire),
            catalog_total: a.catalog_total.load(Ordering::Acquire),
            retained_bytes: a.retained.load(Ordering::Acquire),
            peak_retained_bytes: a.peak.load(Ordering::Acquire),
            prefetch_failures: a.prefetch_failures.load(Ordering::Acquire),
        }
    }
}
impl Drop for PreparedProvider {
    fn drop(&mut self) {
        self.retire();
    }
}

fn evict_unused(
    lru: &mut Vec<Arc<LoadedSurface>>,
    accounting: &Accounting,
    bytes: usize,
    limit: usize,
) {
    while accounting
        .retained
        .load(Ordering::Acquire)
        .saturating_add(bytes)
        > limit
    {
        let Some(index) = lru
            .iter()
            .position(|surface| Arc::strong_count(surface) == 1)
        else {
            break;
        };
        lru.remove(index);
    }
}
fn load_surface(
    key: &str,
    cache: &PolicyCache,
    catalog: &PreparedCatalog,
    lru: &mut Vec<Arc<LoadedSurface>>,
    accounting: &Arc<Accounting>,
    limits: PreparedLimits,
) -> Result<Arc<LoadedSurface>> {
    if let Some(i) = lru.iter().position(|s| s.entry.key == key) {
        let surface = lru.remove(i);
        lru.push(surface.clone());
        accounting.hits.fetch_add(1, Ordering::AcqRel);
        return Ok(surface);
    }
    let entry = catalog
        .get(key)
        .ok_or_else(|| {
            Error::State("Prepared checkpoint is absent from the verified catalog".into())
        })?
        .clone();
    while !lru.is_empty()
        && (lru.len() >= limits.max_frames
            || accounting
                .retained
                .load(Ordering::Acquire)
                .saturating_add(entry.retained_bytes)
                > limits.max_bytes)
    {
        lru.remove(0);
    }
    let started = Instant::now();
    let (surface, timing) = cache.load_profiled(&entry)?;
    accounting
        .read_micros
        .fetch_add(timing.read_micros, Ordering::AcqRel);
    accounting
        .decode_micros
        .fetch_add(timing.decode_micros, Ordering::AcqRel);
    accounting
        .load_micros
        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
    accounting.loads.fetch_add(1, Ordering::AcqRel);
    let bytes = surface.owned_bytes();
    while !lru.is_empty()
        && accounting
            .retained
            .load(Ordering::Acquire)
            .saturating_add(bytes)
            > limits.max_bytes
    {
        lru.remove(0);
    }
    let lease = CacheLease::acquire(accounting, bytes, limits.max_bytes)?;
    let loaded = Arc::new(LoadedSurface {
        entry,
        surface,
        _lease: lease,
    });
    lru.push(loaded.clone());
    Ok(loaded)
}
fn worker(
    run: &RunIndex,
    path: PathBuf,
    limits: PreparedLimits,
    shared: &Shared,
    accounting: &Arc<Accounting>,
    wake: &impl Fn(),
) {
    let start = Instant::now();
    let opened = (|| -> Result<_> {
        let cache = PolicyCache::open(path, run.recording(), limits.read)?;
        let catalog = cache.catalog()?;
        accounting
            .catalog_total
            .store(catalog.entries.len(), Ordering::Release);
        let mut entries = Vec::new();
        let mut issues = catalog.issues;
        for entry in catalog.entries {
            if shared.stop.load(Ordering::Acquire) {
                return Err(Error::State("Prepared catalog cancelled".into()));
            }
            match cache.verify_payload(&entry) {
                Ok(()) => entries.push(Arc::new(entry)),
                Err(e) => issues.push(format!("{}: {e}", entry.key)),
            }
            accounting.catalog_checked.fetch_add(1, Ordering::AcqRel);
        }
        let by_key = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.key.clone(), i))
            .collect();
        Ok((
            cache,
            Arc::new(PreparedCatalog {
                entries,
                issues,
                by_key,
            }),
        ))
    })();
    accounting
        .catalog_micros
        .store(start.elapsed().as_micros() as u64, Ordering::Release);
    if shared.stop.load(Ordering::Acquire) {
        return;
    }
    let (cache, catalog) = match opened {
        Ok(pair) => pair,
        Err(e) => {
            shared.answer.lock().unwrap().catalog = Some(Err(e));
            wake();
            return;
        }
    };
    {
        let mut answer = shared.answer.lock().unwrap();
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        answer.catalog = Some(Ok(catalog.clone()));
    }
    wake();
    let mut lru = Vec::new();
    let mut last_grid: Option<(FrameRequest, Arc<PreparedGrid>)> = None;
    loop {
        let job = {
            let mut next = shared.job.lock().unwrap();
            while next.is_none() && !shared.stop.load(Ordering::Acquire) {
                next = shared.wake.wait(next).unwrap();
            }
            if shared.stop.load(Ordering::Acquire) {
                break;
            }
            next.take().unwrap()
        };
        if shared.cancelled(job.generation) {
            continue;
        }
        if last_grid
            .as_ref()
            .is_some_and(|(old, _)| !old.same_grid(&job.request))
        {
            last_grid = None;
        }
        let result = (|| {
            let current = load_surface(
                &job.request.key,
                &cache,
                &catalog,
                &mut lru,
                accounting,
                limits,
            )?;
            let reference = job
                .request
                .reference
                .as_ref()
                .map(|key| load_surface(key, &cache, &catalog, &mut lru, accounting, limits))
                .transpose()?;
            if shared.cancelled(job.generation) {
                return Err(Error::State("Prepared frame superseded".into()));
            }
            let start = Instant::now();
            let mut reserve = |bytes| {
                evict_unused(&mut lru, accounting, bytes, limits.max_bytes);
                CacheLease::acquire(accounting, bytes, limits.max_bytes)
            };
            let frame = cached_frame(
                run,
                &job.request,
                current,
                reference,
                &mut reserve,
                last_grid.as_ref().map(|(_, grid)| grid),
            )?;
            accounting
                .preparation_micros
                .fetch_add(start.elapsed().as_micros() as u64, Ordering::AcqRel);
            accounting.preparations.fetch_add(1, Ordering::AcqRel);
            last_grid = Some((job.request.clone(), frame.grid.clone()));
            Ok(frame)
        })();
        {
            let mut answer = shared.answer.lock().unwrap();
            if !shared.cancelled(job.generation) {
                answer.frame = Some((job.generation, result));
                wake();
            }
        }
        for key in job.prefetch {
            if shared.cancelled(job.generation) {
                break;
            }
            if load_surface(&key, &cache, &catalog, &mut lru, accounting, limits).is_err() {
                accounting.prefetch_failures.fetch_add(1, Ordering::AcqRel);
            }
        }
    }
}
