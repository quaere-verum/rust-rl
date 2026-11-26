//! Bounded historical inference shared by the Policy, Value and Network workspaces.
use crate::data::*;
use crate::policy::prepared::{
    self, Accounting, Completion, PolicyRequest, PreparedPolicyFrame, RequestOwner, Work,
};
use crate::value::prepared as value_prepared;
pub mod decision;
use rl::{
    Error, Result,
    diagnostics::{models::*, storage::ChunkSource},
};
use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
    mpsc::{self, Receiver, SyncSender},
};
#[cfg(test)]
mod grid_admission_tests;
#[cfg(test)]
pub(crate) mod value_worker_tests;
#[cfg(test)]
mod warmup_tests;

// Queued requests retain lightweight descriptors, not historical weights.
// The worker loads raw bytes only on a two-model-cache miss, one model at a time.
enum Task {
    Validate(CacheKey, Box<ChunkSource>, Option<String>),
    VerifySnapshot(
        CacheKey,
        Box<ChunkSource>,
        Weak<rl::diagnostics::schema::Chunk>,
    ),
    Warm(
        CacheKey,
        Box<ChunkSource>,
        Vec<CacheKey>,
        prepared::PolicyViewLease,
    ),
    Policy(Work),
    Value(value_prepared::Work),
    Decision(Arc<decision::RequestOwner>),
}
enum TaskAnswer {
    Validated(CacheKey),
    Warmed,
    Policy(Completion),
    Value(value_prepared::Completion),
    Decision(Arc<decision::Frame>),
}

struct Worker {
    send: SyncSender<WorkerJob>,
    receive: Receiver<WorkerResult>,
    generation: Arc<AtomicU64>,
    retirement: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
    active: Option<u64>,
    desired: Option<(u64, Task)>,
    answer: Option<Result<TaskAnswer>>,
    policy_accounting: Arc<Accounting>,
    #[cfg(feature = "screenshots")]
    activity: Arc<crate::owner_activity::Activity>,
    #[cfg(feature = "screenshots")]
    models: Arc<ModelOwners>,
    #[cfg(feature = "screenshots")]
    completed: Arc<CompletedAnswers>,
}
struct WorkerJob {
    version: u64,
    job: Option<Task>,
    #[cfg(feature = "screenshots")]
    ownership: crate::owner_activity::Ownership,
}
struct WorkerResult {
    version: u64,
    answer: Result<Option<TaskAnswer>>,
    #[cfg(feature = "screenshots")]
    _completed: CompletedAnswer,
    #[cfg(feature = "screenshots")]
    _ownership: crate::owner_activity::Ownership,
}
struct CachedBundle {
    key: CacheKey,
    encoded_identity: (u64, u32),
    model: ModelBundle,
    admitted_charge: usize,
    parameter_charge: usize,
    _reference_lease: Option<prepared::Lease>,
    #[cfg(feature = "screenshots")]
    source_charge: usize,
}
/// Compare every part of a cached snapshot used to establish scientific
/// identity or panel capability. Model payloads are validated independently
/// from the checksummed source; the cached payload is never used for inference.
fn same_snapshot_metadata(
    a: &rl::diagnostics::schema::Chunk,
    b: &rl::diagnostics::schema::Chunk,
) -> Result<bool> {
    use rl::diagnostics::{reference, schema::Record, value};
    let (Record::ModelSnapshot(a_model), Record::ModelSnapshot(b_model)) = (&a.record, &b.record)
    else {
        return Ok(false);
    };
    let contexts = a.context.identity == b.context.identity
        && a.context.counters == b.context.counters
        && a.context.learner_call == b.context.learner_call
        && a.context.collection_cycle == b.context.collection_cycle
        && a.context.probe_set == b.context.probe_set
        && a.context.trajectory == b.context.trajectory
        && a.context.environment_step == b.context.environment_step
        && a.context.simulation_time.map(f64::to_bits)
            == b.context.simulation_time.map(f64::to_bits)
        && a.context.wall_elapsed_seconds.map(f64::to_bits)
            == b.context.wall_elapsed_seconds.map(f64::to_bits);
    let models = a_model.models.len() == b_model.models.len()
        && a_model.models.iter().all(|(name, a)| {
            b_model.models.get(name).is_some_and(|b| {
                a.bytes == b.bytes
                    && a.signature == b.signature
                    && a.parameter_shapes == b.parameter_shapes
            })
        });
    Ok(contexts
        && a_model.id == b_model.id
        && a_model.architecture == b_model.architecture
        && a_model.configuration == b_model.configuration
        && a_model.capabilities == b_model.capabilities
        && models
        && value::declaration(a)? == value::declaration(b)?
        && value::return_target(a)? == value::return_target(b)?
        && reference::declaration(a)? == reference::declaration(b)?
        && reference::provider_specification(a)? == reference::provider_specification(b)?)
}
struct RecentPolicy {
    frame: PreparedPolicyFrame,
    // The selected point can be newer than the grid's original request.
    owner: Arc<RequestOwner>,
}
struct RecentValue {
    frame: value_prepared::PreparedValueFrame,
    owner: Arc<value_prepared::RequestOwner>,
}
struct PolicySpeculation {
    owner: Arc<RequestOwner>,
    grid: Option<Arc<prepared::PreparedGrid>>,
}
fn cached_bundle<'a>(
    bundles: &'a mut Vec<CachedBundle>,
    key: &CacheKey,
    source: &ChunkSource,
    accounting: Option<&Arc<Accounting>>,
    max_charge: Option<usize>,
    #[cfg(feature = "screenshots")] models: &ModelOwners,
) -> Result<&'a ModelBundle> {
    if let Some(position) = bundles.iter().position(|bundle| {
        &bundle.key == key && bundle.encoded_identity != source.encoded_identity()
    }) {
        bundles.remove(position);
    }
    let bundle = if let Some(position) = bundles.iter().position(|b| &b.key == key) {
        if max_charge.is_some_and(|limit| {
            bundles[position].admitted_charge > limit || bundles[position].parameter_charge > limit
        }) {
            return Err(Error::Config(
                "selected model exceeds read admission".into(),
            ));
        }
        bundles.remove(position)
    } else {
        if bundles.len() == 2 {
            bundles.remove(0);
        }
        #[cfg(feature = "screenshots")]
        {
            models.count.store(bundles.len(), Ordering::Release);
            models.source_charge.store(
                bundles.iter().map(|b| b.source_charge).sum(),
                Ordering::Release,
            );
        }
        let started = std::time::Instant::now();
        let chunk = if let Some(limit) = max_charge {
            source
                .load_if_within_charge(limit)?
                .ok_or_else(|| Error::Config("selected model exceeds read admission".into()))?
        } else {
            source.load()?
        };
        if let (Some(limit), rl::diagnostics::schema::Record::ModelSnapshot(snapshot)) =
            (max_charge, &chunk.record)
        {
            snapshot_configuration(
                snapshot,
                ModelLimits {
                    max_model_bytes: limit,
                    max_parameters: limit / 4,
                    ..ModelLimits::default()
                },
            )?;
        }
        let model = ModelBundle::load(&chunk, ModelLimits::default())?;
        let parameter_charge = decision::parameter_charge(model.configuration())?;
        let reference_bytes = model
            .reference_declaration()
            .map_or(0, |d| d.retained_charge());
        let reference_lease = accounting
            .filter(|_| reference_bytes > 0)
            .map(|a| prepared::Lease::acquire(a, reference_bytes))
            .transpose()?;
        let bundle = CachedBundle {
            key: key.clone(),
            encoded_identity: source.encoded_identity(),
            model,
            admitted_charge: chunk.retained_charge(),
            parameter_charge,
            _reference_lease: reference_lease,
            #[cfg(feature = "screenshots")]
            source_charge: chunk.retained_charge() + reference_bytes,
        };
        if let Some(a) = accounting {
            a.restores.fetch_add(1, Ordering::AcqRel);
            a.restore_micros
                .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
        }
        bundle
    };
    bundles.push(bundle);
    #[cfg(feature = "screenshots")]
    {
        models.count.store(bundles.len(), Ordering::Release);
        models.source_charge.store(
            bundles.iter().map(|b| b.source_charge).sum(),
            Ordering::Release,
        );
    }
    Ok(&bundles.last().unwrap().model)
}
#[cfg(feature = "screenshots")]
#[derive(Default)]
struct ModelOwners {
    count: std::sync::atomic::AtomicUsize,
    source_charge: std::sync::atomic::AtomicUsize,
    warmup_restores: AtomicU64,
}
#[cfg(feature = "screenshots")]
#[derive(Default)]
struct CompletedAnswers {
    count: std::sync::atomic::AtomicUsize,
}
#[cfg(feature = "screenshots")]
struct CompletedAnswer {
    owners: Arc<CompletedAnswers>,
}
#[cfg(feature = "screenshots")]
impl CompletedAnswer {
    fn new(owners: &Arc<CompletedAnswers>) -> Self {
        owners.count.fetch_add(1, Ordering::AcqRel);
        Self {
            owners: owners.clone(),
        }
    }
}
#[cfg(feature = "screenshots")]
impl Drop for CompletedAnswer {
    fn drop(&mut self) {
        self.owners.count.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Worker {
    fn new(wake: impl Fn() + Send + 'static) -> Result<Self> {
        let mut bundles: Vec<CachedBundle> = vec![];
        let accounting = Arc::new(Accounting::default());
        let worker_accounting = accounting.clone();
        #[cfg(feature = "screenshots")]
        let models = Arc::new(ModelOwners::default());
        #[cfg(feature = "screenshots")]
        let observed_models = models.clone();
        let mut worker = Self::spawn_tasks(wake, move |job, generation, current| {
            let Some(job) = job else {
                bundles.clear();
                #[cfg(feature = "screenshots")]
                {
                    observed_models.count.store(0, Ordering::Release);
                    observed_models.source_charge.store(0, Ordering::Release);
                }
                return Ok(None);
            };
            let cancelled = || generation != current.load(Ordering::Relaxed);
            if cancelled() {
                return Ok(None);
            }
            if let Task::Warm(key, source, protected, _lease) = job {
                // Admission is enforced in the cache owner, before any eviction.
                let mut needed = protected.clone();
                if !needed.contains(key) {
                    needed.push(key.clone());
                }
                if needed.len() > 2 {
                    return Ok(Some(TaskAnswer::Warmed));
                }
                source.check_available()?;
                if !bundles.iter().any(|b| &b.key == key) && bundles.len() == 2 {
                    let Some(victim) = bundles.iter().position(|b| !protected.contains(&b.key))
                    else {
                        return Ok(Some(TaskAnswer::Warmed));
                    };
                    bundles.swap(0, victim);
                }
                #[cfg(feature = "screenshots")]
                let before = worker_accounting.snapshot().restores;
                cached_bundle(
                    &mut bundles,
                    key,
                    source,
                    Some(&worker_accounting),
                    None,
                    #[cfg(feature = "screenshots")]
                    &observed_models,
                )?;
                #[cfg(feature = "screenshots")]
                observed_models.warmup_restores.fetch_add(
                    worker_accounting.snapshot().restores - before,
                    Ordering::AcqRel,
                );
                source.check_available()?;
                return Ok((!cancelled()).then_some(TaskAnswer::Warmed));
            }
            if let Task::VerifySnapshot(key, source, cached) = job {
                source.check_available()?;
                worker_accounting
                    .snapshot_verifications
                    .fetch_add(1, Ordering::AcqRel);
                // Re-read and checksum the immutable source. A weak cached
                // witness alone cannot detect same-size, same-time edits.
                let current = source.load()?;
                if let Some(cached) = cached.upgrade() {
                    if !same_snapshot_metadata(&cached, &current)? {
                        return Err(Error::State(
                            "cached snapshot metadata differs from the checksummed source".into(),
                        ));
                    }
                } else {
                    // Eviction while the worker was queued requires ordinary
                    // restoration rather than treating a vanished proof as valid.
                    cached_bundle(
                        &mut bundles,
                        key,
                        source,
                        Some(&worker_accounting),
                        None,
                        #[cfg(feature = "screenshots")]
                        &observed_models,
                    )?;
                }
                return Ok((!cancelled()).then(|| TaskAnswer::Validated(key.clone())));
            }
            if let Task::Validate(key, source, network) = job {
                source.check_available()?;
                if bundles.iter().any(|bundle| {
                    &bundle.key == key && bundle.encoded_identity == source.encoded_identity()
                }) {
                    // A catalogue validation must check current encoded bytes
                    // even if this immutable bundle is already resident.
                    source.load()?;
                }
                let bundle = cached_bundle(
                    &mut bundles,
                    key,
                    source,
                    Some(&worker_accounting),
                    None,
                    #[cfg(feature = "screenshots")]
                    &observed_models,
                )?;
                if let Some(name) = network {
                    bundle.numerical_network(name)?;
                }
                return Ok((!cancelled()).then(|| TaskAnswer::Validated(key.clone())));
            }
            if let Task::Value(work) = job {
                return value_prepared::execute(
                    work,
                    &worker_accounting,
                    cancelled,
                    &mut |key, source, operation| {
                        source.check_available()?;
                        let model = cached_bundle(
                            &mut bundles,
                            key,
                            source,
                            Some(&worker_accounting),
                            None,
                            #[cfg(feature = "screenshots")]
                            &observed_models,
                        )?;
                        if cancelled() {
                            return Err(Error::State("Value request superseded".into()));
                        }
                        operation(model)
                    },
                )
                .map(|answer| answer.map(TaskAnswer::Value));
            }
            if let Task::Decision(owner) = job {
                return decision::execute(
                    owner,
                    &worker_accounting,
                    cancelled,
                    &mut |key, source, operation| {
                        source.check_available()?;
                        let model = cached_bundle(
                            &mut bundles,
                            key,
                            source,
                            Some(&worker_accounting),
                            Some(decision::MAX_MODEL_READ),
                            #[cfg(feature = "screenshots")]
                            &observed_models,
                        )?;
                        operation(model)
                    },
                )
                .map(|answer| answer.map(TaskAnswer::Decision));
            }
            let Task::Policy(work) = job else {
                unreachable!("all other tasks handled above");
            };
            let mut answer = prepared::execute(
                work,
                &worker_accounting,
                cancelled,
                |source, observations| {
                    let address = source.model_source().ok_or_else(|| {
                        Error::State("Recorded source has no arbitrary-input adapter".into())
                    })?;
                    source.check_available()?;
                    let model = cached_bundle(
                        &mut bundles,
                        &source.identity.key,
                        address,
                        Some(&worker_accounting),
                        None,
                        #[cfg(feature = "screenshots")]
                        &observed_models,
                    )?;
                    if cancelled() {
                        return Err(Error::State(
                            "historical inference request superseded".into(),
                        ));
                    }
                    let configuration = model.configuration();
                    if configuration.observation_dim != source.semantics().observation_dim()
                        || &configuration.action_space != source.semantics().action_space()
                    {
                        return Err(Error::Config(
                            "Snapshot input/action configuration differs from recording".into(),
                        ));
                    }
                    let name = prepared::selected_name(configuration, source.name())?;
                    let started = std::time::Instant::now();
                    let result = rl::diagnostics::policy_surface::evaluate_policy_batch(
                        model,
                        &name,
                        observations,
                    );
                    worker_accounting
                        .inference_micros
                        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
                    result.map(|result| (result, name))
                },
            )?;
            if let (Work::Point { request, grid, .. }, Some(Completion::Point(point))) =
                (work, &mut answer)
                && let Some(address) = request.request.current.model_source()
                && request.request.current.action_value_critic().is_some()
                && !cancelled()
            {
                let source = &request.request.current;
                let started = std::time::Instant::now();
                let inference_before = worker_accounting.inference_micros.load(Ordering::Acquire);
                let restore_before = worker_accounting.restore_micros.load(Ordering::Acquire);
                if let Ok(model) = cached_bundle(
                    &mut bundles,
                    &source.identity.key,
                    address,
                    Some(&worker_accounting),
                    None,
                    #[cfg(feature = "screenshots")]
                    &observed_models,
                ) {
                    // The extra critic plot is optional; Policy remains useful
                    // if the declared adapter is unavailable or exceeds budget.
                    let _ = prepared::attach_action_value_curve(
                        point,
                        request,
                        grid,
                        model,
                        &worker_accounting,
                        &cancelled,
                    );
                }
                let inference =
                    worker_accounting.inference_micros.load(Ordering::Acquire) - inference_before;
                let restore =
                    worker_accounting.restore_micros.load(Ordering::Acquire) - restore_before;
                worker_accounting.preparation_micros.fetch_add(
                    (started.elapsed().as_micros() as u64).saturating_sub(inference + restore),
                    Ordering::AcqRel,
                );
                source.check_available()?;
                if cancelled() {
                    return Ok(None);
                }
            }
            Ok(answer.map(TaskAnswer::Policy))
        })?;
        worker.policy_accounting = accounting;
        #[cfg(feature = "screenshots")]
        let worker = {
            let mut worker = worker;
            worker.models = models;
            worker
        };
        Ok(worker)
    }
    fn spawn_tasks(
        wake: impl Fn() + Send + 'static,
        mut process: impl FnMut(Option<&Task>, u64, &AtomicU64) -> Result<Option<TaskAnswer>>
        + Send
        + 'static,
    ) -> Result<Self> {
        let (send, jobs) = mpsc::sync_channel::<WorkerJob>(1);
        let (results, receive) = mpsc::sync_channel(1);
        let generation = Arc::new(AtomicU64::new(0));
        let retirement = Arc::new(AtomicU64::new(0));
        let retired = Arc::new(AtomicU64::new(0));
        let current = generation.clone();
        let requested = retirement.clone();
        let acknowledged = retired.clone();
        #[cfg(feature = "screenshots")]
        let activity = Arc::new(crate::owner_activity::Activity::default());
        #[cfg(feature = "screenshots")]
        let worker_activity = activity.clone();
        #[cfg(feature = "screenshots")]
        let completed = Arc::new(CompletedAnswers::default());
        #[cfg(feature = "screenshots")]
        let worker_completed = completed.clone();
        std::thread::Builder::new()
            .name("historical-inference".into())
            .spawn(move || {
                while let Ok(message) = jobs.recv() {
                    #[cfg(feature = "screenshots")]
                    let _busy = worker_activity.busy();
                    let WorkerJob {
                        version,
                        job,
                        #[cfg(feature = "screenshots")]
                        ownership,
                    } = message;
                    let evaluated = job.is_some();
                    let mut answer = if evaluated && version == current.load(Ordering::Acquire) {
                        #[cfg(feature = "screenshots")]
                        let _execution = worker_activity.execute();
                        process(job.as_ref(), version, &current)
                    } else {
                        Ok(None)
                    };
                    drop(job); // Release source records before publishing completion.
                    if version != current.load(Ordering::Acquire) {
                        answer = Ok(None); // Drop stale local outputs before acknowledging retirement.
                    }
                    let retirement = requested.load(Ordering::Acquire);
                    if retirement != acknowledged.load(Ordering::Relaxed) {
                        if version != current.load(Ordering::Acquire) {
                            answer = Ok(None);
                        }
                        // Load/inspect is not forcibly interrupted. Once its current
                        // backend step returns, release every closure-owned bundle.
                        drop(process(None, retirement, &current));
                        acknowledged.store(retirement, Ordering::Release);
                    }
                    if evaluated {
                        #[cfg(feature = "screenshots")]
                        let completed = CompletedAnswer::new(&worker_completed);
                        if results
                            .send(WorkerResult {
                                version,
                                answer,
                                #[cfg(feature = "screenshots")]
                                _completed: completed,
                                #[cfg(feature = "screenshots")]
                                _ownership: ownership,
                            })
                            .is_err()
                        {
                            break;
                        }
                    } else {
                        drop(answer);
                        #[cfg(feature = "screenshots")]
                        drop(ownership);
                    }
                    wake();
                }
            })?;
        Ok(Self {
            send,
            receive,
            generation,
            retirement,
            retired,
            active: None,
            desired: None,
            answer: None,
            policy_accounting: Arc::new(Accounting::default()),
            #[cfg(feature = "screenshots")]
            activity,
            #[cfg(feature = "screenshots")]
            models: Arc::new(ModelOwners::default()),
            #[cfg(feature = "screenshots")]
            completed,
        })
    }
    fn invalidate(&mut self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.desired = None;
        self.answer = None;
    }
    fn retire_session(&mut self) {
        self.invalidate();
        let generation = self.generation.load(Ordering::Relaxed);
        self.retirement.store(generation, Ordering::Release);
        // A full channel already contains a job or retirement wake, either of
        // which checks the latest request. An idle worker needs this explicit wake.
        let _ = self.send.try_send(WorkerJob {
            version: generation,
            job: None,
            #[cfg(feature = "screenshots")]
            ownership: self.activity.own(),
        });
        self.drain();
    }
    fn retirement_status(&self) -> (u64, u64) {
        (
            self.retirement.load(Ordering::Acquire),
            self.retired.load(Ordering::Acquire),
        )
    }
    fn submit_task(&mut self, job: Task) -> Result<()> {
        self.invalidate();
        self.desired = Some((self.generation.load(Ordering::Relaxed), job));
        self.dispatch()
    }
    fn dispatch(&mut self) -> Result<()> {
        if self.active.is_none()
            && self.retirement_status().0 == self.retirement_status().1
            && let Some((version, job)) = self.desired.take()
        {
            match self.send.try_send(WorkerJob {
                version,
                job: Some(job),
                #[cfg(feature = "screenshots")]
                ownership: self.activity.own(),
            }) {
                Ok(()) => self.active = Some(version),
                Err(mpsc::TrySendError::Full(WorkerJob {
                    version,
                    job: Some(job),
                    ..
                })) => {
                    self.desired = Some((version, job));
                }
                Err(_) => {
                    return Err(Error::State(
                        "historical inference worker unavailable".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn poll(&mut self) -> Result<()> {
        self.drain();
        self.dispatch()
    }
    fn drain(&mut self) {
        if let Ok(WorkerResult {
            version,
            answer,
            #[cfg(feature = "screenshots")]
            _completed,
            #[cfg(feature = "screenshots")]
            _ownership,
        }) = self.receive.try_recv()
        {
            self.active = None;
            if version == self.generation.load(Ordering::Relaxed) {
                self.answer = match answer {
                    Ok(answer) => answer.map(Ok),
                    Err(error) => Some(Err(error)),
                };
            } else {
                drop(answer);
            }
        }
    }
    fn pending(&self) -> bool {
        self.active.is_some()
            || self.desired.is_some()
            || self.retirement_status().0 != self.retirement_status().1
    }
}
pub struct Interactive {
    decision: decision::State,
    validating_snapshot: bool,
    warm_attempt: Option<(CacheKey, Vec<CacheKey>)>,
    warming: bool,
    policy_speculation: Option<PolicySpeculation>,
    policy_prefetch_attempt: Option<CacheKey>,
    worker: Worker,
    value_desired: Option<Arc<value_prepared::RequestOwner>>,
    value_frame: Option<value_prepared::PreparedValueFrame>,
    value_published: Option<Arc<value_prepared::RequestOwner>>,
    value_recent: Vec<RecentValue>,
    value_requested_revision: u64,
    value_staged: Option<value_prepared::PreparedValueFrame>,
    // Presentation metadata follows the latest request even when identical
    // numerical work is already running under an older immutable RequestOwner.
    value_reference_requested_revision: Option<u64>,
    value_error: Option<String>,
    value_point_running: bool,
    policy_desired: Option<Arc<RequestOwner>>,
    policy_frame: Option<PreparedPolicyFrame>,
    policy_published: Option<Arc<RequestOwner>>,
    // Recently published historical frames share their original charged leases.
    // Keep the bound small; old entries are evicted before foreground admission.
    policy_recent: Vec<RecentPolicy>,
    policy_requested_revision: u64,
    policy_staged: Option<PreparedPolicyFrame>,
    policy_error: Option<String>,
    policy_point_running: bool,
}
impl Interactive {
    /// The grid is more expensive to regenerate than the selected point. An
    /// optional suspended point may be discarded first under memory pressure.
    fn discard_suspended_policy_points(&mut self) {
        for entry in &mut self.policy_recent {
            entry.frame.selected = None;
        }
    }
    pub(crate) fn policy_prefetch_ready(&self, request: &PolicyRequest) -> bool {
        self.policy_recent.iter().any(|entry| {
            entry.owner.request.same_grid(request) && entry.owner.request.same_pin(request)
        })
    }
    pub(crate) fn prefetch_policy(&mut self, request: PolicyRequest) -> Result<bool> {
        if self.worker.pending()
            || self.warming
            || self.validating_snapshot
            || request.visits.is_some()
            || request.current.identity().recorded
            || self.policy_frame.is_none()
            || self.policy_prefetch_attempt.as_ref() == Some(&request.current.identity().key)
            || self.policy_prefetch_ready(&request)
        {
            return Ok(false);
        }
        request.validate()?;
        request.current.check_available()?;
        request.check_inputs()?;
        if let Some(reference) = &request.reference {
            reference.check_available()?;
        }
        let needed = prepared::publication_charge(&request)?;
        while !self.policy_recent.is_empty()
            && self
                .worker
                .policy_accounting
                .snapshot()
                .prepared_bytes
                .saturating_add(needed)
                .saturating_add(512 * 1024)
                > crate::budgets::POLICY_PREPARED
        {
            self.policy_recent.remove(0);
        }
        if self
            .worker
            .policy_accounting
            .snapshot()
            .prepared_bytes
            .saturating_add(needed)
            .saturating_add(512 * 1024)
            > crate::budgets::POLICY_PREPARED
        {
            return Ok(false);
        }
        self.policy_prefetch_attempt = Some(request.current.identity().key.clone());
        let owner = RequestOwner::new(request, &self.worker.policy_accounting)?;
        self.worker
            .submit_task(Task::Policy(Work::Grid(owner.clone())))?;
        self.policy_speculation = Some(PolicySpeculation { owner, grid: None });
        self.warming = true;
        Ok(true)
    }
    /// One model-only successor attempt per foreground context. Never publishes.
    pub(crate) fn warm_snapshot(
        &mut self,
        run: &RunIndex,
        index: usize,
        protected: Vec<CacheKey>,
    ) -> Result<bool> {
        if self.worker.pending() || self.validating_snapshot {
            return Ok(false);
        }
        let key = run.key(index)?;
        let mut unique = Vec::with_capacity(3);
        for p in protected {
            if !unique.contains(&p) {
                unique.push(p);
            }
        }
        let attempt = (key.clone(), unique);
        if self.warm_attempt.as_ref() == Some(&attempt) {
            return Ok(false);
        }
        self.warm_attempt = Some(attempt.clone());
        if attempt.1.len() + usize::from(!attempt.1.contains(&key)) > 2 {
            return Ok(false);
        }
        // Descriptors retain no weights; missing sources settle this one attempt.
        let Ok(source) = run.recording().source(index) else {
            return Ok(false);
        };
        let Ok(lease) = self.reserve_policy_view(4096) else {
            return Ok(false);
        };
        self.worker
            .submit_task(Task::Warm(key, Box::new(source), attempt.1, lease))?;
        self.warming = true;
        Ok(true)
    }
    pub(crate) fn cancel_warmup(&mut self) {
        if self.warming {
            self.worker.invalidate();
        }
        self.warming = false;
        self.warm_attempt = None;
        self.policy_speculation = None;
        self.policy_prefetch_attempt = None;
    }
    fn poll_warmup(&mut self) -> Result<bool> {
        if !self.warming {
            return Ok(false);
        }
        self.worker.poll()?;
        if let Some(answer) = self.worker.answer.take() {
            if let Some(mut speculation) = self.policy_speculation.take() {
                match answer {
                    Ok(TaskAnswer::Policy(Completion::Grid(grid))) => {
                        speculation.grid = Some(grid.clone());
                        self.worker.submit_task(Task::Policy(Work::Point {
                            request: speculation.owner.clone(),
                            grid,
                            previous: None,
                        }))?;
                        self.policy_speculation = Some(speculation);
                        return Ok(true);
                    }
                    Ok(TaskAnswer::Policy(Completion::Point(point))) => {
                        if let Some(grid) = speculation.grid.take()
                            && speculation.owner.request.current.check_available().is_ok()
                            && speculation.owner.request.check_inputs().is_ok()
                            && speculation
                                .owner
                                .request
                                .reference
                                .as_ref()
                                .is_none_or(|s| s.check_available().is_ok())
                        {
                            let frame = PreparedPolicyFrame {
                                grid,
                                selected: Some(point),
                                requested_revision: speculation.owner.request.requested_revision,
                            };
                            if self.policy_recent.len() == 2 {
                                self.policy_recent.remove(0);
                            }
                            self.policy_recent.push(RecentPolicy {
                                frame,
                                owner: speculation.owner,
                            });
                        }
                    }
                    _ => {}
                }
            }
            self.warming = false;
        } else if !self.worker.pending() {
            // Failure is speculative only: never overwrite the held frame/error.
            self.warming = false;
            self.policy_speculation = None;
        }
        Ok(true)
    }
    pub fn new(wake: impl Fn() + Send + 'static) -> Result<Self> {
        Ok(Self {
            decision: decision::State::default(),
            worker: Worker::new(wake)?,
            validating_snapshot: false,
            warm_attempt: None,
            warming: false,
            policy_speculation: None,
            policy_prefetch_attempt: None,
            value_desired: None,
            value_frame: None,
            value_published: None,
            value_recent: Vec::new(),
            value_requested_revision: 0,
            value_staged: None,
            value_reference_requested_revision: None,
            value_error: None,
            value_point_running: false,
            policy_desired: None,
            policy_frame: None,
            policy_published: None,
            policy_recent: Vec::new(),
            policy_requested_revision: 0,
            policy_staged: None,
            policy_error: None,
            policy_point_running: false,
        })
    }
    /// Validate an exact snapshot asynchronously through the shared model cache.
    pub(crate) fn validate_snapshot(&mut self, run: &RunIndex, index: usize) -> Result<()> {
        self.validate_snapshot_for(run, index, None)
    }
    pub(crate) fn verify_snapshot(
        &mut self,
        run: &RunIndex,
        index: usize,
        chunk: &Arc<rl::diagnostics::schema::Chunk>,
    ) -> Result<()> {
        self.cancel_warmup();
        self.worker.submit_task(Task::VerifySnapshot(
            run.key(index)?,
            Box::new(run.recording().source(index)?),
            Arc::downgrade(chunk),
        ))?;
        self.validating_snapshot = true;
        Ok(())
    }
    pub(crate) fn validate_network_snapshot(
        &mut self,
        run: &RunIndex,
        index: usize,
        name: String,
    ) -> Result<()> {
        self.validate_snapshot_for(run, index, Some(name))
    }
    fn validate_snapshot_for(
        &mut self,
        run: &RunIndex,
        index: usize,
        network: Option<String>,
    ) -> Result<()> {
        self.cancel_warmup();
        let task = Task::Validate(
            run.key(index)?,
            Box::new(run.recording().source(index)?),
            network,
        );
        self.worker.submit_task(task)?;
        self.validating_snapshot = true;
        Ok(())
    }
    pub(crate) fn poll_snapshot_validation(&mut self) -> Result<Option<Result<CacheKey>>> {
        self.worker.poll()?;
        if !self.validating_snapshot {
            return Ok(None);
        }
        let Some(answer) = self.worker.answer.take() else {
            return Ok(None);
        };
        self.validating_snapshot = false;
        Ok(Some(match answer {
            Ok(TaskAnswer::Validated(key)) => Ok(key),
            Err(e) => Err(e),
            _ => Err(Error::State("snapshot validation superseded".into())),
        }))
    }
    pub(crate) fn cancel_snapshot_validation(&mut self) {
        if self.validating_snapshot {
            self.worker.invalidate();
            self.validating_snapshot = false;
        }
    }
    pub(crate) fn snapshot_validation_pending(&self) -> bool {
        self.validating_snapshot
    }

    /// Suspend a Policy session during a short visit to Training. Completed
    /// frames remain in the same charged, bounded cache; unfinished work is
    /// invalidated and must never publish after the workspace changes.
    pub(crate) fn suspend_policy(&mut self) {
        self.value_recent.clear();
        self.clear_decision();
        self.cancel_warmup();
        self.cancel_snapshot_validation();
        self.worker.invalidate();
        self.policy_desired = None;
        self.policy_staged = None;
        self.policy_error = None;
        self.policy_point_running = false;
        if let (Some(frame), Some(owner)) = (self.policy_frame.take(), self.policy_published.take())
        {
            if self.policy_recent.len() == 2 {
                self.policy_recent.remove(0);
            }
            self.policy_recent.push(RecentPolicy { frame, owner });
        }
    }

    /// Retain at most the currently published Policy frame for a compatible
    /// Value visit. The old model bundles still retire before Value work.
    pub(crate) fn suspend_policy_for_value(&mut self) {
        self.policy_recent.clear();
        let value_recent = std::mem::take(&mut self.value_recent);
        self.suspend_policy();
        self.value_recent = value_recent;
        self.worker.retire_session();
    }

    /// Keep a completed Value frame only when a complete Policy frame and
    /// validation headroom already fit in the same bounded owner accounting.
    pub(crate) fn leave_value_for_policy(&mut self) {
        self.clear_decision();
        self.cancel_warmup();
        self.cancel_snapshot_validation();
        self.worker.invalidate();
        self.value_desired = None;
        self.value_recent.clear();
        if let (Some(frame), Some(owner)) = (self.value_frame.take(), self.value_published.take())
            && self
                .policy_recent
                .iter()
                .any(|entry| entry.frame.selected.is_some())
            && self
                .worker
                .policy_accounting
                .snapshot()
                .prepared_bytes
                .saturating_add(512 * 1024)
                <= crate::budgets::POLICY_PREPARED
        {
            self.value_recent.push(RecentValue { frame, owner });
        }
        self.value_staged = None;
        self.value_reference_requested_revision = None;
        self.value_error = None;
        self.value_point_running = false;
    }

    /// Keep only completed, already charged Value frames across Training.
    /// Returning through ordinary submission revalidates every source and pin.
    pub(crate) fn suspend_value(&mut self) {
        self.policy_recent.clear();
        self.clear_decision();
        self.cancel_warmup();
        self.cancel_snapshot_validation();
        self.worker.invalidate();
        self.value_desired = None;
        self.value_staged = None;
        self.value_error = None;
        self.value_point_running = false;
        if let (Some(frame), Some(owner)) = (self.value_frame.take(), self.value_published.take()) {
            if self.value_recent.len() == 2 {
                self.value_recent.remove(0);
            }
            self.value_recent.push(RecentValue { frame, owner });
        }
    }

    /// Retire a run/mode session without blocking the UI on a backend operation.
    /// Ordinary input edits deliberately keep the two-model reuse cache.
    pub fn retire_session(&mut self) {
        self.decision = decision::State::default();
        self.cancel_warmup();
        self.validating_snapshot = false;
        self.value_desired = None;
        self.value_frame = None;
        self.value_published = None;
        self.value_recent.clear();
        self.value_staged = None;
        self.value_reference_requested_revision = None;
        self.value_error = None;
        self.value_point_running = false;
        self.policy_desired = None;
        self.policy_frame = None;
        self.policy_published = None;
        self.policy_recent.clear();
        self.policy_staged = None;
        self.policy_error = None;
        self.policy_point_running = false;
        self.worker.retire_session();
    }
    /// Cheap hidden-view housekeeping: discard completions without dispatching.
    pub fn housekeeping(&mut self) {
        self.worker.drain();
    }
    /// Requested and acknowledged retirement generations. Equality certifies
    /// closure-cache release after any uncancellable load/inspect step finishes.
    /// These are ownership acknowledgements, not process-memory measurements.
    pub fn retirement_status(&self) -> (u64, u64) {
        self.worker.retirement_status()
    }
    pub fn pending(&self) -> bool {
        self.worker.pending()
    }
    /// Submit only on a keyed input change. Same-grid pin edits replace one
    /// desired point without invalidating any unfinished grid generation.
    pub fn submit_policy(&mut self, request: PolicyRequest) -> Result<()> {
        self.clear_decision();
        if self.value_desired.is_some() || self.value_frame.is_some() {
            self.worker.invalidate();
            self.value_desired = None;
            self.value_frame = None;
            self.value_published = None;
            self.value_recent.clear();
            self.value_staged = None;
            self.value_error = None;
            self.value_point_running = false;
        }
        let result = self.submit_policy_inner(request);
        if let Err(error) = &result {
            self.worker.invalidate();
            self.policy_desired = None;
            self.policy_staged = None;
            self.policy_point_running = false;
            self.policy_error = Some(error.to_string());
        }
        result
    }
    pub(crate) fn fail_policy_context(&mut self, error: &Error) {
        self.invalidate_policy_context();
        self.validating_snapshot = false;
        self.policy_error = Some(error.to_string());
    }
    pub(crate) fn fail_value_context(&mut self, error: &Error) {
        self.invalidate_value_context();
        self.validating_snapshot = false;
        self.value_error = Some(error.to_string());
    }
    pub(crate) fn reserve_policy_view(&self, bytes: usize) -> Result<prepared::PolicyViewLease> {
        prepared::PolicyViewLease::new(&self.worker.policy_accounting, bytes)
    }
    /// Validation and grid preparation are serialized by the single worker.
    /// The peak is max(held + staged + validation, next + successor,
    /// next + validation), including a predecessor retained for cache reuse.
    /// Only proven exclusive retired allocations leave the steady baseline.
    fn grid_fits(
        &self,
        incremental: usize,
        complete: usize,
        successor: usize,
        retired: usize,
    ) -> bool {
        let current = self.worker.policy_accounting.snapshot().prepared_bytes;
        Self::grid_fits_at(current, incremental, complete, successor, retired)
    }
    fn grid_fits_at(
        current: usize,
        incremental: usize,
        complete: usize,
        successor: usize,
        retired: usize,
    ) -> bool {
        let baseline = current.saturating_sub(retired);
        current
            .saturating_add(incremental)
            .saturating_add(512 * 1024)
            <= crate::budgets::POLICY_PREPARED
            && baseline.saturating_add(complete).saturating_add(successor)
                <= crate::budgets::POLICY_PREPARED
            && baseline.saturating_add(complete).saturating_add(512 * 1024)
                <= crate::budgets::POLICY_PREPARED
    }
    /// Some optional evidence can fit with validation headroom even when two
    /// complete frames cannot. In that case the next foreground request must
    /// release its old frame before starting, rather than retain it as cache.
    fn single_value_grid_fits(&self, incremental: usize, complete: usize, retired: usize) -> bool {
        let current = self.worker.policy_accounting.snapshot().prepared_bytes;
        Self::single_value_grid_fits_at(current, incremental, complete, retired)
    }
    fn single_value_grid_fits_at(
        current: usize,
        incremental: usize,
        complete: usize,
        retired: usize,
    ) -> bool {
        let baseline = current.saturating_sub(retired);
        current
            .saturating_add(incremental)
            .saturating_add(512 * 1024)
            <= crate::budgets::POLICY_PREPARED
            && baseline.saturating_add(complete).saturating_add(512 * 1024)
                <= crate::budgets::POLICY_PREPARED
    }
    fn has_complete_policy(&self, request: &PolicyRequest) -> bool {
        self.policy_recent.iter().any(|entry| {
            entry.frame.selected.is_some()
                && entry.owner.request.same_grid(request)
                && entry.owner.request.same_pin(request)
        }) || self
            .policy_frame
            .as_ref()
            .zip(self.policy_published.as_ref())
            .is_some_and(|(frame, owner)| {
                frame.selected.is_some()
                    && owner.request.same_grid(request)
                    && owner.request.same_pin(request)
            })
    }
    pub(crate) fn automatic_policy_resolution(
        &mut self,
        request: &mut PolicyRequest,
    ) -> Result<()> {
        if !self.value_recent.is_empty() {
            if request.slice.mode == crate::policy::slice::SliceMode::TwoDimensional
                && !request.current.identity().recorded
                && request.input_population.is_none()
            {
                let original = request.slice.resolution;
                let candidates = self
                    .policy_recent
                    .iter()
                    .map(|e| (&e.frame, &e.owner))
                    .chain(
                        self.policy_frame
                            .as_ref()
                            .zip(self.policy_published.as_ref()),
                    );
                for (frame, owner) in candidates {
                    Arc::make_mut(&mut request.slice).resolution = owner.request.slice.resolution;
                    if frame.selected.is_some()
                        && owner.request.same_grid(request)
                        && owner.request.same_pin(request)
                    {
                        return Ok(());
                    }
                }
                Arc::make_mut(&mut request.slice).resolution = original;
            }
            if !self.has_complete_policy(request) {
                self.value_recent.clear();
            }
        }
        if request.slice.mode != crate::policy::slice::SliceMode::TwoDimensional
            || request.current.identity().recorded
            || request.input_population.is_some()
        {
            return Ok(());
        }
        if self
            .policy_frame
            .as_ref()
            .is_some_and(|f| f.grid.matches_request(request))
            || self
                .policy_recent
                .iter()
                .any(|f| f.frame.grid.matches_request(request))
        {
            return Ok(());
        }
        // Speculative retention may not force a lower foreground resolution.
        while !self.policy_recent.is_empty() {
            let Ok(complete) = prepared::publication_charge(request) else {
                break;
            };
            let fits = {
                let incremental = complete - request.metadata_bytes()
                    + RequestOwner::metadata_charge(request, &self.worker.policy_accounting)
                        .unwrap_or(complete);
                let retired = self
                    .policy_frame
                    .as_ref()
                    .map_or(0, |f| f.retirement_bytes(self.policy_desired.as_ref()));
                let successor = complete
                    - request
                        .reference
                        .as_ref()
                        .map_or(0, |r| r.shared_metadata().1);
                self.grid_fits(incremental, complete, successor, retired)
            };
            if fits {
                break;
            }
            self.policy_recent.remove(0);
        }
        let retired = self
            .policy_frame
            .as_ref()
            .map_or(0, |f| f.retirement_bytes(self.policy_desired.as_ref()));
        let mut slice = (*request.slice).clone();
        for n in [128, 96, 64, 48, 32] {
            slice.resolution = [n, n];
            request.slice = Arc::new(slice.clone());
            if let Ok(complete) = prepared::publication_charge(request) {
                let base = complete - request.metadata_bytes();
                let incremental =
                    base + RequestOwner::metadata_charge(request, &self.worker.policy_accounting)?;
                let successor = complete
                    - request
                        .reference
                        .as_ref()
                        .map_or(0, |r| r.shared_metadata().1);
                if self.grid_fits(incremental, complete, successor, retired) {
                    return Ok(());
                }
            }
        }
        Err(Error::Config("Automatic Policy grid cannot fit retained/replacement frames and validation headroom; reduce explicit resolution or release retained frames".into()))
    }
    pub(crate) fn automatic_value_resolution(
        &mut self,
        request: &mut value_prepared::ValueRequest,
        pending_catalog_bytes: usize,
    ) -> Result<()> {
        for entry in &self.value_recent {
            let mut reusable = request.clone();
            Arc::make_mut(&mut reusable.slice).resolution = entry.owner.request.slice.resolution;
            reusable.withdraw_unadmitted_empirical();
            let old = &entry.owner.request;
            if old.same_grid(&reusable)
                && reusable.same_grid(old)
                && old.pin.same(&reusable.pin)
                && old.empirical_condition == reusable.empirical_condition
            {
                *request = reusable;
                return Ok(());
            }
        }
        if !self.policy_recent.is_empty()
            && request.slice.mode == crate::policy::slice::SliceMode::TwoDimensional
            && !request.current.identity().recorded
        {
            let mut reusable = request.clone();
            reusable.withdraw_unadmitted_empirical();
            if self
                .value_frame
                .as_ref()
                .is_some_and(|f| f.grid.matches_request(&reusable))
                || self
                    .value_recent
                    .iter()
                    .any(|f| f.frame.grid.matches_request(&reusable))
            {
                *request = reusable;
                return Ok(());
            }
            // The retained Policy frame is optional. It must not reduce the
            // resolution normally selected for a Value request. A changed
            // Value frame also needs the usual replacement-frame headroom.
            if self.value_frame.is_some()
                || self.value_staged.is_some()
                || !self.value_recent.is_empty()
            {
                self.policy_recent.clear();
            } else if self.policy_recent.len() == 1 {
                let recent = &self.policy_recent[0];
                let retired = recent.frame.retirement_bytes(Some(&recent.owner));
                let retiring = recent
                    .frame
                    .retiring_owner(Some(&recent.owner))
                    .map_or(&[][..], |owner| owner.shared_leases());
                let accounting = &self.worker.policy_accounting;
                let without_policy = accounting.snapshot().prepared_bytes.saturating_sub(retired);
                let mut preferred = None;
                let mut slice = (*request.slice).clone();
                for single in [false, true] {
                    for n in [128, 96, 64, 48, 32] {
                        slice.resolution = [n, n];
                        request.slice = Arc::new(slice.clone());
                        let Ok(complete) = value_prepared::publication_charge(request) else {
                            continue;
                        };
                        let base = complete - request.metadata_bytes();
                        let incremental = base
                            + value_prepared::RequestOwner::metadata_charge_after(
                                request, accounting, retiring,
                            )?;
                        let standalone = base
                            + value_prepared::RequestOwner::standalone_metadata_charge(request);
                        let successor = complete
                            - request
                                .reference_value
                                .as_ref()
                                .map_or(0, |r| r.metadata_bytes)
                            - request
                                .reference_policy
                                .as_ref()
                                .map_or(0, |p| p.shared_metadata().1);
                        let fits = if single {
                            Self::single_value_grid_fits_at(
                                without_policy,
                                incremental.saturating_add(pending_catalog_bytes),
                                standalone.saturating_add(pending_catalog_bytes),
                                0,
                            )
                        } else {
                            Self::grid_fits_at(
                                without_policy,
                                incremental.saturating_add(pending_catalog_bytes),
                                complete.saturating_add(pending_catalog_bytes),
                                successor,
                                0,
                            )
                        };
                        if fits {
                            preferred = Some(n);
                            break;
                        }
                    }
                    if preferred.is_some() {
                        break;
                    }
                }
                if let Some(n) = preferred {
                    slice.resolution = [n, n];
                    request.slice = Arc::new(slice);
                    // A verified empirical grid determines the selected CDF's
                    // actual sample bound. Admit the grid first; the point is
                    // checked separately before the worker allocates it.
                    let complete = value_prepared::deferred_publication_charge(request)?;
                    let base = complete - request.metadata_bytes();
                    let incremental =
                        base + value_prepared::RequestOwner::metadata_charge(request, accounting)?;
                    let standalone =
                        base + value_prepared::RequestOwner::standalone_metadata_charge(request);
                    if self.single_value_grid_fits(
                        incremental.saturating_add(pending_catalog_bytes),
                        standalone.saturating_add(pending_catalog_bytes),
                        0,
                    ) {
                        return Ok(());
                    }
                    self.discard_suspended_policy_points();
                    if self.single_value_grid_fits(
                        incremental.saturating_add(pending_catalog_bytes),
                        standalone.saturating_add(pending_catalog_bytes),
                        0,
                    ) {
                        return Ok(());
                    }
                }
                self.policy_recent.clear();
            } else {
                self.policy_recent.clear();
            }
        }
        if !self.policy_recent.is_empty()
            && (request.slice.mode != crate::policy::slice::SliceMode::TwoDimensional
                || request.current.identity().recorded)
        {
            let fits = value_prepared::publication_charge(request)
                .ok()
                .is_some_and(|complete| {
                    let incremental = complete - request.metadata_bytes()
                        + value_prepared::RequestOwner::metadata_charge(
                            request,
                            &self.worker.policy_accounting,
                        )
                        .unwrap_or(complete);
                    let standalone = complete - request.metadata_bytes()
                        + value_prepared::RequestOwner::standalone_metadata_charge(request);
                    self.single_value_grid_fits(
                        incremental.saturating_add(pending_catalog_bytes),
                        standalone.saturating_add(pending_catalog_bytes),
                        0,
                    )
                });
            if !fits {
                self.policy_recent.clear();
            }
        }
        let original = (!self.policy_recent.is_empty()).then(|| request.clone());
        let result = self.automatic_value_resolution_inner(request, pending_catalog_bytes);
        if result.is_err()
            && let Some(original) = original
        {
            self.policy_recent.clear();
            *request = original;
            return self.automatic_value_resolution_inner(request, pending_catalog_bytes);
        }
        result
    }
    fn automatic_value_resolution_inner(
        &mut self,
        request: &mut value_prepared::ValueRequest,
        pending_catalog_bytes: usize,
    ) -> Result<()> {
        if request.slice.mode != crate::policy::slice::SliceMode::TwoDimensional
            || request.current.identity().recorded
        {
            return Ok(());
        }
        // Submission withdraws optional evidence that cannot fit the current
        // slice. Compare the same effective request before evicting a frame.
        let mut reusable = request.clone();
        reusable.withdraw_unadmitted_empirical();
        if self
            .value_frame
            .as_ref()
            .is_some_and(|f| f.grid.matches_request(&reusable))
            || self
                .value_recent
                .iter()
                .any(|f| f.frame.grid.matches_request(&reusable))
        {
            *request = reusable;
            return Ok(());
        }
        while !self.value_recent.is_empty() {
            let Ok(complete) = value_prepared::publication_charge(request) else {
                break;
            };
            let fits = {
                let incremental = complete - request.metadata_bytes()
                    + value_prepared::RequestOwner::metadata_charge(
                        request,
                        &self.worker.policy_accounting,
                    )
                    .unwrap_or(complete);
                let retired = self
                    .value_frame
                    .as_ref()
                    .map_or(0, |f| f.retirement_bytes(self.value_desired.as_ref()));
                let successor = complete
                    - request
                        .reference_value
                        .as_ref()
                        .map_or(0, |r| r.metadata_bytes)
                    - request
                        .reference_policy
                        .as_ref()
                        .map_or(0, |p| p.shared_metadata().1);
                self.grid_fits(
                    incremental.saturating_add(pending_catalog_bytes),
                    complete.saturating_add(pending_catalog_bytes),
                    successor,
                    retired,
                )
            };
            if fits {
                break;
            }
            self.value_recent.remove(0);
        }
        let retired = self
            .value_frame
            .as_ref()
            .map_or(0, |f| f.retirement_bytes(self.value_desired.as_ref()));
        let mut slice = (*request.slice).clone();
        for n in [128, 96, 64, 48, 32] {
            slice.resolution = [n, n];
            request.slice = Arc::new(slice.clone());
            if let Ok(complete) = value_prepared::publication_charge(request) {
                let base = complete - request.metadata_bytes();
                let incremental = base
                    + value_prepared::RequestOwner::metadata_charge(
                        request,
                        &self.worker.policy_accounting,
                    )?;
                let successor = complete
                    - request
                        .reference_value
                        .as_ref()
                        .map_or(0, |r| r.metadata_bytes)
                    - request
                        .reference_policy
                        .as_ref()
                        .map_or(0, |p| p.shared_metadata().1);
                if self.grid_fits(
                    incremental.saturating_add(pending_catalog_bytes),
                    complete.saturating_add(pending_catalog_bytes),
                    successor,
                    retired,
                ) {
                    return Ok(());
                }
            }
        }
        if request.empirical.is_some()
            || (self.value_frame.is_none() && self.value_staged.is_none())
        {
            for n in [128, 96, 64, 48, 32] {
                slice.resolution = [n, n];
                request.slice = Arc::new(slice.clone());
                if let Ok(complete) = value_prepared::publication_charge(request) {
                    let incremental = complete - request.metadata_bytes()
                        + value_prepared::RequestOwner::metadata_charge(
                            request,
                            &self.worker.policy_accounting,
                        )?;
                    let standalone = complete - request.metadata_bytes()
                        + value_prepared::RequestOwner::standalone_metadata_charge(request);
                    if self.single_value_grid_fits(
                        incremental.saturating_add(pending_catalog_bytes),
                        standalone.saturating_add(pending_catalog_bytes),
                        retired,
                    ) {
                        return Ok(());
                    }
                }
            }
        }
        Err(Error::Config("Automatic Value grid cannot fit retained/replacement frames and validation headroom; reduce explicit resolution or release retained frames".into()))
    }
    /// Cancel obsolete preparation while holding the last complete publication
    /// and existing two-model cache during discovery of its replacement.
    pub(crate) fn invalidate_policy_context(&mut self) {
        self.cancel_warmup();
        self.worker.invalidate();
        self.policy_desired = None;
        self.policy_staged = None;
        self.policy_error = None;
        self.policy_point_running = false;
    }
    fn submit_policy_inner(&mut self, request: PolicyRequest) -> Result<()> {
        request.validate()?;
        if !self.value_recent.is_empty() && !self.has_complete_policy(&request) {
            self.value_recent.clear();
        }
        self.policy_requested_revision = request.requested_revision;
        if let Some(index) = self.policy_recent.iter().position(|entry| {
            entry.owner.request.same_grid(&request) && entry.owner.request.same_pin(&request)
        }) {
            request.current.check_available()?;
            request.check_inputs()?;
            if request
                .reference
                .as_ref()
                .is_none_or(|source| source.check_available().is_ok())
            {
                self.cancel_warmup();
                self.worker.invalidate();
                self.policy_desired = None;
                self.policy_staged = None;
                self.policy_point_running = false;
                self.policy_error = None;
                let mut entry = self.policy_recent.remove(index);
                entry.frame.requested_revision = request.requested_revision;
                if entry.frame.selected.is_none() {
                    self.worker
                        .policy_accounting
                        .reused_grids
                        .fetch_add(1, Ordering::AcqRel);
                    self.policy_desired = Some(entry.owner);
                    self.policy_staged = Some(entry.frame);
                    self.submit_policy_point()?;
                    return Ok(());
                }
                if let (Some(frame), Some(owner)) = (
                    self.policy_frame.replace(entry.frame),
                    self.policy_published.replace(entry.owner),
                ) {
                    self.policy_recent.push(RecentPolicy { frame, owner });
                }
                return Ok(());
            }
            self.policy_recent.remove(index);
        }
        let same_grid = self
            .policy_staged
            .as_ref()
            .or(self.policy_frame.as_ref())
            .is_some_and(|frame| frame.grid.matches_request(&request))
            || (self
                .policy_desired
                .as_ref()
                .is_some_and(|old| old.request.same_grid(&request))
                && self.worker.pending()
                && !self.policy_point_running);
        if same_grid
            && self.policy_desired.as_ref().is_some_and(|old| {
                old.request.same_grid(&request) && old.request.same_pin(&request)
            })
        {
            if let Some(frame) = &mut self.policy_frame
                && frame.grid.matches_request(&request)
            {
                frame.requested_revision = request.requested_revision;
            }
            return Ok(());
        }
        self.cancel_warmup();
        if !same_grid && let Ok(needed) = prepared::publication_charge(&request) {
            // Retention is opportunistic: never let old cached frames consume
            // the foreground request's own bounded publication allowance.
            while !self.policy_recent.is_empty()
                && self
                    .worker
                    .policy_accounting
                    .snapshot()
                    .prepared_bytes
                    .saturating_add(needed)
                    .saturating_add(512 * 1024)
                    > crate::budgets::POLICY_PREPARED
            {
                self.policy_recent.remove(0);
            }
            if self
                .worker
                .policy_accounting
                .snapshot()
                .prepared_bytes
                .saturating_add(needed)
                .saturating_add(512 * 1024)
                <= crate::budgets::POLICY_PREPARED
                && let (Some(frame), Some(owner)) = (&self.policy_frame, &self.policy_published)
                && !self.policy_recent.iter().any(|old| {
                    Arc::ptr_eq(&old.frame.grid, &frame.grid) && Arc::ptr_eq(&old.owner, owner)
                })
            {
                if self.policy_recent.len() == 2 {
                    self.policy_recent.remove(0);
                }
                self.policy_recent.push(RecentPolicy {
                    frame: frame.clone(),
                    owner: owner.clone(),
                });
            }
        }
        let owner = RequestOwner::new(request, &self.worker.policy_accounting)?;
        self.policy_error = None;
        if same_grid {
            self.worker
                .policy_accounting
                .reused_grids
                .fetch_add(1, Ordering::AcqRel);
            self.policy_staged = self
                .policy_staged
                .take()
                .filter(|f| f.grid.matches_request(&owner.request))
                .or_else(|| {
                    self.policy_frame
                        .as_ref()
                        .filter(|f| f.grid.matches_request(&owner.request))
                        .cloned()
                });
            self.policy_desired = Some(owner);
            if self.policy_staged.is_some() {
                self.submit_policy_point()?;
            }
        } else {
            self.policy_staged = None;
            self.policy_desired = Some(owner.clone());
            self.policy_point_running = false;
            self.worker.submit_task(Task::Policy(Work::Grid(owner)))?;
        }
        Ok(())
    }
    fn submit_policy_point(&mut self) -> Result<()> {
        let (Some(request), Some(frame)) = (&self.policy_desired, &self.policy_staged) else {
            return Ok(());
        };
        self.policy_point_running = true;
        self.worker.submit_task(Task::Policy(Work::Point {
            request: request.clone(),
            grid: frame.grid.clone(),
            previous: frame.selected.clone(),
        }))
    }
    pub fn poll_policy(&mut self) -> Result<()> {
        if self.poll_warmup()? {
            return Ok(());
        }
        if self.validating_snapshot {
            return Ok(());
        }
        self.worker.poll()?;
        if let Some(answer) = self.worker.answer.take() {
            if let Some(request) = &self.policy_desired
                && let Err(error) = request.request.current.check_available()
            {
                self.policy_error = Some(error.to_string());
                self.policy_point_running = false;
                self.policy_staged = None;
                self.policy_desired = None;
                return Ok(());
            }
            // A completed answer may wait while retention removes its optional
            // source. Settle that loss before either staging or publication.
            // Do not reclassify an admission/input error as optional failure.
            let answer = if matches!(&answer, Ok(TaskAnswer::Policy(_)))
                && self.policy_desired.as_ref().is_some_and(|r| {
                    r.request
                        .reference
                        .as_ref()
                        .is_some_and(|s| s.check_available().is_err())
                }) {
                drop(answer);
                Ok(TaskAnswer::Policy(Completion::ReferenceUnavailable))
            } else {
                answer
            };
            match answer {
                Ok(TaskAnswer::Policy(Completion::Grid(grid))) => {
                    if self.policy_desired.is_some() {
                        self.policy_staged = Some(PreparedPolicyFrame {
                            grid,
                            selected: None,
                            requested_revision: self.policy_requested_revision,
                        });
                        self.submit_policy_point()?;
                    }
                }
                Ok(TaskAnswer::Policy(Completion::Point(point))) => {
                    self.policy_point_running = false;
                    if let Some(mut frame) = self.policy_staged.take() {
                        frame.requested_revision = self.policy_requested_revision;
                        frame.selected = Some(point);
                        self.policy_frame = Some(frame);
                        self.policy_published = self.policy_desired.clone();
                    }
                }
                Ok(TaskAnswer::Policy(Completion::ReferenceUnavailable)) => {
                    if let Some(desired) = &self.policy_desired
                        && desired.request.reference.is_some()
                    {
                        let mut request = desired.request.clone();
                        request.reference = None;
                        self.policy_staged = None;
                        self.policy_desired = None;
                        self.policy_point_running = false;
                        self.submit_policy_inner(request)?;
                        return Ok(());
                    }
                }
                Err(error) => {
                    self.policy_error = Some(error.to_string());
                    self.policy_point_running = false;
                    self.policy_desired = None;
                    self.policy_staged = None;
                }
                Ok(
                    TaskAnswer::Value(_)
                    | TaskAnswer::Validated(_)
                    | TaskAnswer::Warmed
                    | TaskAnswer::Decision(_),
                ) => {}
            }
        }
        Ok(())
    }
    pub fn policy_frame(&self) -> Option<&PreparedPolicyFrame> {
        self.policy_frame.as_ref()
    }
    pub fn policy_error(&self) -> Option<&str> {
        self.policy_error.as_deref()
    }
    pub fn policy_pending(&self) -> bool {
        self.policy_desired.is_some()
            && self.policy_error.is_none()
            && ((!self.warming && self.worker.pending()) || self.policy_point_running)
    }
    pub fn policy_metrics(&self) -> prepared::PolicyMetrics {
        self.worker.policy_accounting.snapshot()
    }
    pub(crate) fn invalidate_value_context(&mut self) {
        self.cancel_warmup();
        self.worker.invalidate();
        self.value_desired = None;
        self.value_staged = None;
        self.value_reference_requested_revision = None;
        self.value_error = None;
        self.value_point_running = false;
    }
    pub(crate) fn release_value_frames(&mut self) {
        self.invalidate_value_context();
        self.value_frame = None;
        self.value_published = None;
        self.value_recent.clear();
    }
    pub fn submit_value(&mut self, mut request: value_prepared::ValueRequest) -> Result<()> {
        request.withdraw_unadmitted_empirical();
        self.clear_decision();
        if self.policy_desired.is_some() || self.policy_frame.is_some() {
            self.invalidate_policy_context();
            self.policy_frame = None;
            self.policy_published = None;
            self.policy_recent.clear();
        }
        let cached_value = self.value_recent.iter().any(|entry| {
            let old = &entry.owner.request;
            old.same_grid(&request)
                && request.same_grid(old)
                && old.pin.same(&request.pin)
                && old.empirical_condition == request.empirical_condition
        });
        if !self.policy_recent.is_empty() && !cached_value {
            let charge = if self.value_frame.is_none()
                && self.value_staged.is_none()
                && self.value_recent.is_empty()
            {
                value_prepared::deferred_publication_charge(&request)
            } else {
                value_prepared::publication_charge(&request)
            };
            let fits = |this: &Self, complete| {
                let incremental = complete - request.metadata_bytes()
                    + value_prepared::RequestOwner::metadata_charge(
                        &request,
                        &this.worker.policy_accounting,
                    )
                    .unwrap_or(complete);
                let standalone = complete - request.metadata_bytes()
                    + value_prepared::RequestOwner::standalone_metadata_charge(&request);
                this.single_value_grid_fits(incremental, standalone, 0)
            };
            if !charge.as_ref().is_ok_and(|&complete| fits(self, complete)) {
                self.discard_suspended_policy_points();
                if !charge.is_ok_and(|complete| fits(self, complete)) {
                    self.policy_recent.clear();
                }
            }
        }
        let derived = request.output == rl::diagnostics::value::ValueOutput::DecisionLoss;
        let result = self.submit_value_inner(request);
        if let Err(e) = &result {
            if derived {
                self.value_frame = None;
            }
            self.worker.invalidate();
            self.value_desired = None;
            self.value_staged = None;
            self.value_point_running = false;
            self.value_error = Some(e.to_string());
        }
        result
    }
    fn submit_value_inner(&mut self, mut request: value_prepared::ValueRequest) -> Result<()> {
        request.validate()?;
        if request.output == rl::diagnostics::value::ValueOutput::DecisionLoss {
            value_prepared::check_sources(&request)?;
        }
        self.value_requested_revision = request.requested_revision;
        self.value_reference_requested_revision = request.reference_requested_revision;
        if let Some(index) = self.value_recent.iter().position(|entry| {
            let old = &entry.owner.request;
            old.same_grid(&request)
                && request.same_grid(old)
                && old.pin.same(&request.pin)
                && old.empirical_condition == request.empirical_condition
        }) {
            value_prepared::check_sources(&request)?;
            let optional_available = request
                .reference_value
                .as_ref()
                .is_none_or(|s| s.check_available().is_ok())
                && request
                    .reference_policy
                    .as_ref()
                    .is_none_or(|s| s.check_available().is_ok())
                && request
                    .current_policy
                    .as_ref()
                    .is_none_or(|s| s.check_available().is_ok())
                && request
                    .empirical
                    .as_ref()
                    .is_none_or(|s| s.available().is_ok())
                && request.calibration.as_ref().is_none_or(|c| {
                    c.source.check_available().is_ok()
                        && c.reference
                            .as_ref()
                            .is_none_or(|s| s.check_available().is_ok())
                });
            if optional_available {
                self.cancel_warmup();
                self.worker.invalidate();
                self.value_desired = None;
                self.value_staged = None;
                self.value_point_running = false;
                self.value_error = None;
                let mut entry = self.value_recent.remove(index);
                entry.frame.requested_revision = request.requested_revision;
                entry.frame.reference_requested_revision = request.reference_requested_revision;
                if let (Some(frame), Some(owner)) = (
                    self.value_frame.replace(entry.frame),
                    self.value_published.replace(entry.owner),
                ) {
                    self.value_recent.push(RecentValue { frame, owner });
                }
                return Ok(());
            }
            self.value_recent.remove(index);
        }
        let same_grid = self
            .value_staged
            .as_ref()
            .or(self.value_frame.as_ref())
            .is_some_and(|f| f.grid.matches_request(&request))
            || (self
                .value_desired
                .as_ref()
                .is_some_and(|r| r.request.same_grid(&request))
                && self.worker.pending()
                && !self.value_point_running);
        if same_grid
            && self.value_desired.as_ref().is_some_and(|r| {
                r.request.same_grid(&request)
                    && r.request.pin.same(&request.pin)
                    && r.request.empirical_condition == request.empirical_condition
            })
        {
            if let Some(frame) = &mut self.value_frame
                && frame.grid.matches_request(&request)
            {
                frame.reference_requested_revision = self.value_reference_requested_revision;
                frame.requested_revision = request.requested_revision;
            }
            return Ok(());
        }
        if let Some(candidate) = &request.empirical {
            let retained = self
                .value_staged
                .as_ref()
                .into_iter()
                .chain(self.value_frame.as_ref())
                .filter_map(|f| f.grid.empirical_source())
                .chain(
                    self.value_desired
                        .as_ref()
                        .and_then(|r| r.request.empirical.as_ref()),
                )
                .find(|source| source.same(candidate))
                .cloned();
            if let Some(retained) = retained {
                request.empirical = Some(retained);
            }
        }
        self.cancel_warmup();
        if !same_grid && let Ok(needed) = value_prepared::publication_charge(&request) {
            while !self.value_recent.is_empty()
                && self
                    .worker
                    .policy_accounting
                    .snapshot()
                    .prepared_bytes
                    .saturating_add(needed)
                    .saturating_add(512 * 1024)
                    > crate::budgets::POLICY_PREPARED
            {
                self.value_recent.remove(0);
            }
            if self
                .worker
                .policy_accounting
                .snapshot()
                .prepared_bytes
                .saturating_add(needed)
                .saturating_add(512 * 1024)
                <= crate::budgets::POLICY_PREPARED
                && let (Some(frame), Some(owner)) = (&self.value_frame, &self.value_published)
                && !self.value_recent.iter().any(|old| {
                    Arc::ptr_eq(&old.frame.grid, &frame.grid) && Arc::ptr_eq(&old.owner, owner)
                })
            {
                if self.value_recent.len() == 2 {
                    self.value_recent.remove(0);
                }
                self.value_recent.push(RecentValue {
                    frame: frame.clone(),
                    owner: owner.clone(),
                });
            }
        }
        let owner = value_prepared::RequestOwner::new(request, &self.worker.policy_accounting)?;
        self.value_error = None;
        if same_grid {
            self.worker
                .policy_accounting
                .reused_grids
                .fetch_add(1, Ordering::AcqRel);
            self.value_staged = self
                .value_staged
                .take()
                .filter(|f| f.grid.matches_request(&owner.request))
                .or_else(|| {
                    self.value_frame
                        .as_ref()
                        .filter(|f| f.grid.matches_request(&owner.request))
                        .cloned()
                });
            self.value_desired = Some(owner);
            if self.value_staged.is_some() {
                self.submit_value_point()?;
            }
        } else {
            self.value_staged = None;
            self.value_desired = Some(owner.clone());
            self.value_point_running = false;
            self.worker
                .submit_task(Task::Value(value_prepared::Work::Grid(owner)))?;
        }
        Ok(())
    }
    fn submit_value_point(&mut self) -> Result<()> {
        if !self.policy_recent.is_empty()
            && let (Some(request), Some(frame)) = (&self.value_desired, &self.value_staged)
        {
            let point = value_prepared::point_charge_for_grid(&request.request, &frame.grid)?;
            if self
                .worker
                .policy_accounting
                .snapshot()
                .prepared_bytes
                .saturating_add(point)
                .saturating_add(512 * 1024)
                > crate::budgets::POLICY_PREPARED
            {
                self.discard_suspended_policy_points();
                if self
                    .worker
                    .policy_accounting
                    .snapshot()
                    .prepared_bytes
                    .saturating_add(point)
                    .saturating_add(512 * 1024)
                    > crate::budgets::POLICY_PREPARED
                {
                    self.policy_recent.clear();
                }
            }
        }
        let (Some(request), Some(frame)) = (&self.value_desired, &self.value_staged) else {
            return Ok(());
        };
        self.value_point_running = true;
        self.worker
            .submit_task(Task::Value(value_prepared::Work::Point {
                request: request.clone(),
                grid: frame.grid.clone(),
            }))
    }
    pub fn poll_value(&mut self) -> Result<()> {
        if self.poll_warmup()? {
            return Ok(());
        }
        if self.validating_snapshot {
            return Ok(());
        }
        let result = self.poll_value_inner();
        if (result.is_err() || self.value_error.is_some())
            && self.value_frame.as_ref().is_some_and(|f| {
                f.grid.output() == &rl::diagnostics::value::ValueOutput::DecisionLoss
            })
        {
            self.value_frame = None;
        }
        if let Err(error) = &result {
            self.worker.invalidate();
            self.value_desired = None;
            self.value_staged = None;
            self.value_point_running = false;
            self.value_error = Some(error.to_string());
        }
        result
    }
    fn poll_value_inner(&mut self) -> Result<()> {
        self.worker.poll()?;
        if let Some(answer) = self.worker.answer.take() {
            if let Some(r) = &self.value_desired
                && let Err(e) = value_prepared::check_sources(&r.request)
            {
                self.value_error = Some(e.to_string());
                self.value_staged = None;
                self.value_desired = None;
                self.value_point_running = false;
                return Ok(());
            }
            match answer {
                Ok(TaskAnswer::Value(value_prepared::Completion::Grid(mut grid))) => {
                    if let Some(r) = &self.value_desired {
                        value_prepared::sanitize_optional(
                            &mut grid,
                            None,
                            &r.request,
                            &self.worker.policy_accounting,
                        )?;
                        self.value_staged = Some(value_prepared::PreparedValueFrame {
                            grid,
                            selected: None,
                            requested_revision: self.value_requested_revision,
                            reference_requested_revision: self.value_reference_requested_revision,
                        });
                        self.submit_value_point()?;
                    }
                }
                Ok(TaskAnswer::Value(value_prepared::Completion::Point(mut point))) => {
                    self.value_point_running = false;
                    if let (Some(mut f), Some(r)) = (self.value_staged.take(), &self.value_desired)
                    {
                        value_prepared::sanitize_optional(
                            &mut f.grid,
                            Some(&mut point),
                            &r.request,
                            &self.worker.policy_accounting,
                        )?;
                        f.requested_revision = self.value_requested_revision;
                        f.reference_requested_revision = self.value_reference_requested_revision;
                        f.selected = Some(point);
                        self.value_frame = Some(f);
                        self.value_published = self.value_desired.clone();
                    }
                }
                Err(e) => {
                    self.value_error = Some(e.to_string());
                    self.value_desired = None;
                    self.value_staged = None;
                    self.value_point_running = false;
                }
                _ => {}
            }
        }
        Ok(())
    }
    pub fn value_frame(&self) -> Option<&value_prepared::PreparedValueFrame> {
        self.value_frame.as_ref()
    }
    pub fn value_error(&self) -> Option<&str> {
        self.value_error.as_deref()
    }
    pub fn value_pending(&self) -> bool {
        self.value_desired.is_some()
            && self.value_error.is_none()
            && ((!self.warming && self.worker.pending()) || self.value_point_running)
    }
    pub fn value_metrics(&self) -> value_prepared::ValueMetrics {
        self.worker.policy_accounting.snapshot()
    }
    #[cfg(feature = "screenshots")]
    pub(crate) fn owner_snapshot(&self) -> InferenceOwners {
        InferenceOwners {
            activity: self.worker.activity.snapshot(),
            pending: self.worker.pending(),
            retirement: self.retirement_status(),
            models: self.worker.models.count.load(Ordering::Acquire),
            model_source_charge: self.worker.models.source_charge.load(Ordering::Acquire),
            warmup_restores: self.worker.models.warmup_restores.load(Ordering::Acquire),
            completed_results: self.worker.completed.count.load(Ordering::Acquire),
            completed_result_charge: 0,
            ui_answers: usize::from(self.worker.answer.is_some())
                + usize::from(self.policy_staged.is_some())
                + usize::from(self.value_staged.is_some())
                + usize::from(self.policy_frame.is_some())
                + usize::from(self.policy_desired.is_some())
                + usize::from(self.value_frame.is_some())
                + usize::from(self.value_desired.is_some())
                + self.decision.owners(),
            ui_answer_charge: 0,
            policy: self.policy_metrics(),
        }
    }
}
#[cfg(feature = "screenshots")]
#[derive(Debug, serde::Serialize)]
pub(crate) struct InferenceOwners {
    pub activity: crate::owner_activity::ActivitySnapshot,
    pub pending: bool,
    pub retirement: (u64, u64),
    pub models: usize,
    /// Validated source charge, not runtime ModelBundle/allocator heap size.
    pub model_source_charge: usize,
    /// Cumulative model restores performed by speculative Warm tasks only.
    pub warmup_restores: u64,
    pub completed_results: usize,
    pub completed_result_charge: usize,
    pub ui_answers: usize,
    pub ui_answer_charge: usize,
    /// Includes queued, completed, externally aliased and committed Policy
    /// leases once; numerical source chunks retain existing resolver charges.
    pub policy: prepared::PolicyMetrics,
}
#[cfg(test)]
mod worker_tests {
    use super::*;
    use std::time::Duration;

    fn job(run: &RunIndex, value: u64) -> Task {
        Task::Validate(
            CacheKey {
                root: run.root().clone(),
                sequence: value,
            },
            Box::new(run.recording().source(0).unwrap()),
            None,
        )
    }

    #[test]
    fn retirement_discards_queued_and_ui_answers_without_hidden_dispatch() {
        let (_dir, run) = value_worker_tests::fixture();
        for poll_first in [false, true] {
            let (wake, wakes) = mpsc::channel();
            let mut interactive = Interactive::new(|| {}).unwrap();
            interactive.worker = Worker::spawn_tasks(
                move || {
                    let _ = wake.send(());
                },
                move |job, _, _| match job {
                    Some(Task::Validate(key, _, _)) => Ok(Some(TaskAnswer::Validated(key.clone()))),
                    _ => Ok(None),
                },
            )
            .unwrap();
            interactive.worker.submit_task(job(&run, 1)).unwrap();
            wakes.recv_timeout(Duration::from_secs(2)).unwrap();
            #[cfg(feature = "screenshots")]
            assert_eq!(interactive.owner_snapshot().completed_results, 1);
            if poll_first {
                interactive.worker.poll().unwrap();
            }
            let source = prepared::PolicySource::model(&run, 0, Some("actor")).unwrap();
            let input = rl::core::env::Observation::new(vec![0.]);
            let slice =
                crate::policy::slice::SliceConfig::new(source.semantics(), &input, &[]).unwrap();
            let request = PolicyRequest::new(
                source,
                None,
                0,
                slice,
                prepared::PolicyPin::Synthetic(input),
                None,
            )
            .unwrap();
            let owner = RequestOwner::new(request, &interactive.worker.policy_accounting).unwrap();
            let retained = Arc::downgrade(&owner);
            interactive.policy_desired = Some(owner);
            interactive.retire_session();
            assert!(interactive.policy_desired.is_none());
            assert!(retained.upgrade().is_none());
            assert_eq!(interactive.policy_metrics().prepared_bytes, 0);
            assert!(interactive.worker.answer.is_none());
            while interactive.pending() {
                wakes.recv_timeout(Duration::from_secs(2)).unwrap();
                interactive.housekeeping();
            }
            assert!(interactive.worker.receive.try_recv().is_err());
            #[cfg(feature = "screenshots")]
            {
                let owners = interactive.owner_snapshot();
                assert_eq!(owners.completed_results, 0);
                assert_eq!(owners.ui_answers, 0);
                // Retirement acknowledgement can precede dropping the idle
                // wake's activity guard; channel/UI ownership is already gone.
                assert_eq!(owners.retirement.0, owners.retirement.1);
            }
        }
    }

    #[test]
    fn retirement_wakes_idle_worker_and_releases_its_cached_models() {
        let (_dir, run) = value_worker_tests::fixture();
        let (wake, wakes) = mpsc::channel();
        let model = Arc::new(());
        let owned = Arc::downgrade(&model);
        let mut cache = Some(model);
        let mut worker = Worker::spawn_tasks(
            move || {
                let _ = wake.send(());
            },
            move |job, _, _| {
                if job.is_none() {
                    cache = None;
                }
                let _keep_alive = &cache;
                Ok(None)
            },
        )
        .unwrap();
        worker.submit_task(job(&run, 1)).unwrap();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.poll().unwrap();
        worker.invalidate();
        assert!(
            owned.upgrade().is_some(),
            "editing must preserve model reuse"
        );
        worker.retire_session();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.poll().unwrap();
        assert!(owned.upgrade().is_none());
        assert!(!worker.pending());
        assert_eq!(worker.retirement_status().0, worker.retirement_status().1);
    }

    #[test]
    fn policy_suspension_discards_an_in_flight_answer_without_retiring_models() {
        let (_dir, run) = value_worker_tests::fixture();
        let (wake, wakes) = mpsc::channel();
        let (started, starts) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let model = Arc::new(());
        let retained = Arc::downgrade(&model);
        let mut cache = Some(model);
        let mut interactive = Interactive::new(|| {}).unwrap();
        interactive.worker = Worker::spawn_tasks(
            move || {
                let _ = wake.send(());
            },
            move |job, _, _| {
                let Some(Task::Validate(key, _, _)) = job else {
                    cache = None;
                    return Ok(None);
                };
                let _keep_alive = &cache;
                started.send(()).unwrap();
                blocked.recv().unwrap();
                Ok(Some(TaskAnswer::Validated(key.clone())))
            },
        )
        .unwrap();
        interactive.worker.submit_task(job(&run, 1)).unwrap();
        starts.recv_timeout(Duration::from_secs(2)).unwrap();
        interactive.suspend_policy();
        release.send(()).unwrap();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        interactive.housekeeping();
        assert!(interactive.worker.answer.is_none());
        assert!(!interactive.pending());
        assert!(retained.upgrade().is_some());
        interactive.retire_session();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        interactive.housekeeping();
        assert!(retained.upgrade().is_none());
    }

    #[test]
    fn retirement_after_in_flight_step_drops_cache_and_cancels_latest_job_before_reentry() {
        let (_dir, run) = value_worker_tests::fixture();
        let (wake, wakes) = mpsc::channel();
        let (resume_wake, wake_blocked) = mpsc::channel();
        let first_wake = std::cell::Cell::new(true);
        let (started, starts) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let model = Arc::new(());
        let owned = Arc::downgrade(&model);
        let mut cache = Some(model);
        let mut worker = Worker::spawn_tasks(
            move || {
                let _ = wake.send(());
                if first_wake.replace(false) {
                    // Keep the retirement wake queued after the stale result
                    // arrives, so reentry must retry dispatch after that wake.
                    wake_blocked.recv().unwrap();
                }
            },
            move |job, _, _| {
                let Some(Task::Validate(key, _, _)) = job else {
                    cache = None;
                    return Ok(None);
                };
                let _keep_alive = &cache;
                started.send(key.sequence).unwrap();
                if key.sequence == 1 {
                    blocked.recv().unwrap();
                }
                Ok(Some(TaskAnswer::Validated(key.clone())))
            },
        )
        .unwrap();
        worker.submit_task(job(&run, 1)).unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(2)).unwrap(), 1);
        #[cfg(feature = "screenshots")]
        {
            assert_eq!(worker.activity.snapshot().executing, 1);
            assert_eq!(worker.activity.snapshot().outstanding, 1);
        }
        worker.submit_task(job(&run, 2)).unwrap();
        worker.retire_session();
        assert!(
            owned.upgrade().is_some(),
            "a backend step may be uncancellable"
        );
        assert_ne!(worker.retirement_status().0, worker.retirement_status().1);
        release.send(()).unwrap();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        #[cfg(feature = "screenshots")]
        {
            assert_eq!(worker.retirement_status().0, worker.retirement_status().1);
            assert!(
                !worker.activity.snapshot().idle(),
                "acknowledgement must not hide the queued stale completion"
            );
        }
        worker.poll().unwrap();
        assert!(owned.upgrade().is_none());
        assert!(worker.answer.take().is_none());
        assert!(
            starts.try_recv().is_err(),
            "latest old-session job must never run"
        );
        assert_eq!(worker.retirement_status().0, worker.retirement_status().1);
        worker.submit_task(job(&run, 3)).unwrap();
        resume_wake.send(()).unwrap();
        // The queued retirement wake prevented immediate dispatch. Pump the
        // normal wake/poll protocol before waiting for the new job to start.
        while worker.pending() {
            wakes.recv_timeout(Duration::from_secs(2)).unwrap();
            worker.poll().unwrap();
        }
        assert_eq!(starts.recv_timeout(Duration::from_secs(2)).unwrap(), 3);
        assert!(matches!(
            worker.answer.take(),
            Some(Ok(TaskAnswer::Validated(key))) if key.sequence == 3
        ));
    }

    #[test]
    fn queued_inference_is_replaced_and_stale_answer_never_commits() {
        let (started, starts) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let (_dir, run) = value_worker_tests::fixture();
        let (wake, wakes) = mpsc::channel();
        let mut worker = Worker::spawn_tasks(
            move || {
                let _ = wake.send(());
            },
            move |job, _, _| {
                let Some(Task::Validate(key, _, _)) = job else {
                    return Ok(None);
                };
                started.send(key.sequence).unwrap();
                if key.sequence == 1 {
                    blocked.recv().unwrap();
                }
                Err(Error::State(format!("answer {}", key.sequence)))
            },
        )
        .unwrap();
        worker.submit_task(job(&run, 1)).unwrap();
        assert_eq!(starts.recv_timeout(Duration::from_secs(2)).unwrap(), 1);
        worker.submit_task(job(&run, 2)).unwrap();
        worker.submit_task(job(&run, 3)).unwrap();
        assert!(worker.pending());
        release.send(()).unwrap();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.poll().unwrap();
        assert!(worker.answer.take().is_none());
        assert_eq!(starts.recv_timeout(Duration::from_secs(2)).unwrap(), 3);
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.poll().unwrap();
        let Some(Err(error)) = worker.answer.take() else {
            panic!("latest answer missing")
        };
        assert!(error.to_string().contains("answer 3"));
        assert!(!worker.pending());
        assert!(starts.try_recv().is_err());
    }

    #[test]
    fn edited_inputs_invalidate_even_a_response_from_the_same_snapshot() {
        let (started, starts) = mpsc::channel();
        let (release, blocked) = mpsc::channel();
        let (_dir, run) = value_worker_tests::fixture();
        let (wake, wakes) = mpsc::channel();
        let mut worker = Worker::spawn_tasks(
            move || {
                let _ = wake.send(());
            },
            move |job, _, _| {
                let Some(Task::Validate(key, _, _)) = job else {
                    return Ok(None);
                };
                started.send(()).unwrap();
                blocked.recv().unwrap();
                Err(Error::State(format!("input {}", key.sequence)))
            },
        )
        .unwrap();
        worker.submit_task(job(&run, 1)).unwrap();
        starts.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.submit_task(job(&run, 1)).unwrap(); // Same weights, a new explicit validation.
        worker.invalidate(); // The source context is then invalidated without dispatching.
        release.send(()).unwrap();
        wakes.recv_timeout(Duration::from_secs(2)).unwrap();
        worker.poll().unwrap();
        assert!(worker.answer.take().is_none());
        assert!(!worker.pending());
        assert!(starts.try_recv().is_err());
    }
}
