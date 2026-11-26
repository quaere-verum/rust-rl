//! Exact episode mode on the existing Dynamics worker. Numerical projection and
//! unconditional population pooling are independent capabilities.
use super::*;
use rl::diagnostics::schema::RecordKind;

#[derive(Clone)]
pub struct EpisodeRequest {
    key: CacheKey,
    source: ChunkSource,
    manifest: Arc<RunManifest>,
    revision: u64,
    role: String,
    checkpoint: Option<String>,
    trajectory: Option<String>,
    bytes: usize,
    action: ActionAxis,
}
impl EpisodeRequest {
    pub fn new(run: &RunIndex, index: usize) -> Result<Self> {
        let entry = run
            .recording()
            .entries()
            .get(index)
            .ok_or_else(|| Error::State("missing episode record".into()))?;
        checked(
            entry.kind == RecordKind::Trajectory,
            "exact episode request requires a recorded trajectory",
        )?;
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(bytes.len());
                if self.0 > 128 * 1024 {
                    return Err(std::io::Error::other(
                        "episode manifest exceeds metadata admission",
                    ));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut encoded = Count(0);
        serde_json::to_writer(&mut encoded, run.recording().manifest())?;
        let bytes = encoded
            .0
            .saturating_mul(8)
            .saturating_add(64 * 1024)
            // One queued/active boxed job is part of this request's owner.
            .saturating_add(std::mem::size_of::<EpisodeJob>());
        let action = if matches!(
            run.recording().manifest().action_space,
            rl::core::action::ActionSpace::Discrete { .. }
        ) {
            ActionAxis::Discrete
        } else {
            ActionAxis::Continuous(0)
        };
        Ok(Self {
            key: run.key(index)?,
            source: run.recording().source(index)?,
            manifest: Arc::new(run.recording().manifest().clone()),
            revision: entry.revision,
            role: entry.model_role.clone(),
            checkpoint: entry.checkpoint_id.clone(),
            trajectory: entry.trajectory.clone(),
            bytes,
            action,
        })
    }
    pub fn key(&self) -> &CacheKey {
        &self.key
    }
    pub fn with_action(mut self, action: ActionAxis) -> Result<Self> {
        let valid = match (&self.manifest.action_space, action) {
            (rl::core::action::ActionSpace::Discrete { .. }, ActionAxis::Discrete) => true,
            (space, ActionAxis::Continuous(i))
                if !matches!(space, rl::core::action::ActionSpace::Discrete { .. }) =>
            {
                i < space.dimensions()
            }
            _ => false,
        };
        checked(valid, "invalid exact episode action component")?;
        self.action = action;
        Ok(self)
    }
}
pub struct EpisodeFrame {
    pub key: CacheKey,
    pub episode: Episode,
    pub retained_bytes: usize,
    pub action: ActionAxis,
    pub reward_trace: Vec<[f64; 2]>,
    pub action_trace: Vec<ActionSample>,
    request: EpisodeRequest,
}
pub(super) struct EpisodeJob {
    pub serial: u64,
    request: EpisodeRequest,
    cached: Option<Weak<EpisodeFrame>>,
    check_only: bool,
}
pub(super) struct EpisodeAnswer {
    pub serial: u64,
    pub result: Result<Arc<EpisodeFrame>>,
}
impl DynamicsWorker {
    pub fn episode_frame(&self) -> Option<&Arc<EpisodeFrame>> {
        self.episode_frame.as_ref()
    }
    pub fn request_episode(&mut self, request: EpisodeRequest) -> Result<bool> {
        if self
            .episode_request
            .as_ref()
            .is_some_and(|r| r.key == request.key && r.action == request.action)
        {
            return Ok(false);
        }
        self.episode_leases.retain(|w| w.strong_count() > 0);
        let old_bytes = self
            .episode_leases
            .iter()
            .filter_map(Weak::upgrade)
            .map(|f| f.retained_bytes)
            .sum::<usize>()
            + self
                .leases
                .iter()
                .filter_map(Weak::upgrade)
                .map(|f| {
                    f.retained_bytes
                        + f.selected.retained_bytes
                        + f.state.retained_bytes
                        + f.background.retained_bytes
                })
                .sum::<usize>();
        checked(
            self.episode_leases.len() < 3 && old_bytes <= 2 * self.limits.max_frame_bytes,
            "release old episode/frame leases before preparing another episode",
        )?;
        checked(
            request.bytes <= self.limits.max_source_set_bytes,
            "episode request metadata exceeds ownership admission",
        )?;
        let serial = self
            .shared
            .serial
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        cancel_availability(&self.shared, &mut mailbox);
        self.shared.hit_serial.fetch_add(1, Ordering::AcqRel);
        let cached = self
            .episode_frame
            .as_ref()
            .filter(|f| f.key == request.key)
            .map(Arc::downgrade);
        mailbox.desired = Some(WorkJob::Episode(Box::new(EpisodeJob {
            serial,
            request: request.clone(),
            cached,
            check_only: false,
        })));
        mailbox.answer = None;
        self.episode_request = Some(request);
        self.wanted = None;
        self.frame = None;
        self.error = None;
        self.submitted += 1;
        self.shared.changed.notify_one();
        Ok(true)
    }
    pub fn recheck_episode(&mut self) {
        let Some(frame) = &self.episode_frame else {
            return;
        };
        let mut mailbox = self.shared.mailbox.lock().unwrap();
        if mailbox.active || mailbox.desired.is_some() || mailbox.answer.is_some() {
            return;
        }
        mailbox.desired = Some(WorkJob::Episode(Box::new(EpisodeJob {
            serial: self.shared.serial.load(Ordering::Acquire),
            request: frame.request.clone(),
            cached: Some(Arc::downgrade(frame)),
            check_only: true,
        })));
        self.shared.changed.notify_one();
    }
    pub(super) fn poll_episode(&mut self) {
        let answer = {
            let mut mailbox = self.shared.mailbox.lock().unwrap();
            if matches!(mailbox.answer, Some(WorkAnswer::Episode(_))) {
                match mailbox.answer.take() {
                    Some(WorkAnswer::Episode(a)) => Some(a),
                    _ => unreachable!(),
                }
            } else {
                None
            }
        };
        let Some(answer) =
            answer.filter(|a| a.serial == self.shared.serial.load(Ordering::Acquire))
        else {
            return;
        };
        match answer.result {
            Ok(frame) => {
                if !self
                    .episode_leases
                    .iter()
                    .any(|w| w.ptr_eq(&Arc::downgrade(&frame)))
                {
                    self.episode_leases.push(Arc::downgrade(&frame));
                }
                self.episode_frame = Some(frame);
                self.error = None;
            }
            Err(error) => {
                self.episode_frame = None;
                self.error = Some(error.to_string().chars().take(512).collect());
            }
        }
    }
}
pub(super) fn execute(
    job: &EpisodeJob,
    shared: &Shared,
    limits: &WorkLimits,
    reader: &Reader,
) -> Result<Arc<EpisodeFrame>> {
    let request = &job.request;
    let check = || {
        checked(
            shared.serial.load(Ordering::Acquire) == job.serial,
            "episode request superseded",
        )
    };
    check()?;
    request.source.check_available()?;
    let episode = if let Some(cached) = job.cached.as_ref().and_then(Weak::upgrade) {
        check()?;
        if job.check_only {
            return Ok(cached);
        }
        cached.episode.clone()
    } else {
        let _decode = shared.decode_gate.lock.lock().unwrap();
        check()?;
        shared.counts.source_reads.fetch_add(1, Ordering::Relaxed);
        let chunk = reader(&request.source, limits.max_read_bytes)?
            .ok_or_else(|| Error::Shape("exact episode exceeds read admission".into()))?;
        shared
            .counts
            .source_read_bytes
            .fetch_add(chunk.retained_charge() as u64, Ordering::Relaxed);
        let identity = &chunk.context.identity;
        checked(
            identity.run_id == request.manifest.run_id
                && identity.learner_update_revision == request.revision
                && identity.model_role == request.role
                && identity.checkpoint_id == request.checkpoint
                && chunk.context.trajectory == request.trajectory,
            "exact episode source identity differs from requested record",
        )?;
        prepare_episode(&chunk, &request.manifest, &limits.science())?
    };
    check()?;
    request.source.check_available()?;
    let retained_bytes = episode
        .retained_bytes()
        .saturating_add(request.bytes)
        .saturating_add(4096)
        .saturating_add(
            episode
                .transitions()
                .min(limits.max_trace_points)
                .saturating_mul(16 + std::mem::size_of::<ActionSample>()),
        );
    checked(
        retained_bytes <= limits.max_frame_bytes,
        "exact episode frame exceeds ownership admission",
    )?;
    let indices = samples(episode.transitions(), limits.max_trace_points);
    let mut reward_trace = Vec::with_capacity(indices.len());
    let mut action_trace = Vec::with_capacity(indices.len());
    for t in indices {
        check()?;
        reward_trace.push([t as f64, episode.step(t)?.reward.expect("decision reward")]);
        action_trace.push(ActionSample {
            t,
            value: episode
                .action_value(t, request.action)?
                .expect("decision action"),
        });
    }
    shared
        .counts
        .selected_builds
        .fetch_add(1, Ordering::Relaxed);
    Ok(Arc::new(EpisodeFrame {
        key: request.key.clone(),
        episode,
        retained_bytes,
        action: request.action,
        reward_trace,
        action_trace,
        request: request.clone(),
    }))
}
