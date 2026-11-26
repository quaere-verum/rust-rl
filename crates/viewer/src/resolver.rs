//! Sparse policy resolution of immutable scientific inputs.
//!
//! Search uses metadata first and checks complete inputs only on candidate chunks.
//! A pending replacement may keep a compatible older frame, always with its own
//! revision and probe definition. It never relabels that frame as a new measurement.
use crate::{
    data::*,
    plot_data::{array, matching_probe_input},
};
use rl::{Error, Result, diagnostics::schema::*};
use std::{
    collections::VecDeque,
    ops::Deref,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveStatus {
    Ready,
    Loading,
    Unavailable,
}
#[derive(Clone)]
pub struct ResolvedRecord {
    pub chunk: LeasedChunk,
    pub probes: LeasedChunk,
    pub key: CacheKey,
    pub probe_key: CacheKey,
    pub row: usize,
}
impl ResolvedRecord {
    pub fn revision(&self) -> u64 {
        self.chunk.context.identity.learner_update_revision
    }
}
/// An immutable chunk retaining its frame's conservative memory charge. Cloning
/// this handle cannot detach numerical data from the resolver's accounting.
/// The whole frame remains charged until its last chunk/probe handle is dropped.
#[derive(Clone)]
pub struct LeasedChunk {
    chunk: Arc<Chunk>,
    _lease: Arc<FrameLease>,
}
impl Deref for LeasedChunk {
    type Target = Chunk;
    fn deref(&self) -> &Chunk {
        &self.chunk
    }
}
impl AsRef<Chunk> for LeasedChunk {
    fn as_ref(&self) -> &Chunk {
        &self.chunk
    }
}
struct FrameLease {
    account: Arc<AtomicUsize>,
    charge: usize,
}
impl Drop for FrameLease {
    fn drop(&mut self) {
        self.account.fetch_sub(self.charge, Ordering::Relaxed);
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveRejection {
    NoSelection,
    MissingSource,
    InvalidSource,
    Unreadable,
    BudgetExceeded,
    PopulationMismatch,
    InvalidRecord,
    InputMismatch,
}
pub struct ResolvedView {
    pub rejection: Option<ResolveRejection>,
    pub status: ResolveStatus,
    pub requested_revision: u64,
    pub record: Option<ResolvedRecord>,
    pub warnings: Vec<String>,
}
#[derive(PartialEq, Eq)]
struct Identity {
    root: Arc<PathBuf>,
    input: SelectedInput,
    role: String,
}
struct Slot {
    rejection: Option<ResolveRejection>,
    identity: Identity,
    revision: u64,
    frontier: (usize, Option<(u64, u64)>),
    cursor: Option<(u64, u64)>,
    finished: bool,
    frame: Option<ResolvedRecord>,
    source: Option<Pinned>,
    candidate: Option<Pinned>,
    warnings: Vec<String>,
}
struct Pinned {
    index: usize,
    chunk: Arc<Chunk>,
    charge: usize,
}
impl Pinned {
    fn new(index: usize, chunk: Arc<Chunk>) -> Self {
        let charge = ChunkCache::charge(&chunk);
        Self {
            index,
            chunk,
            charge,
        }
    }
}
impl Slot {
    fn pending_bytes(&self) -> usize {
        self.source
            .as_ref()
            .map_or(0, |p| p.charge)
            .saturating_add(self.candidate.as_ref().map_or(0, |p| p.charge))
    }
    fn clear_pending(&mut self) {
        self.source = None;
        self.candidate = None;
    }
    fn finish(&mut self) -> ResolveStatus {
        self.clear_pending();
        self.finished = true;
        if self.frame.is_some() {
            ResolveStatus::Ready
        } else {
            ResolveStatus::Unavailable
        }
    }
    fn reject(&mut self, cursor: (u64, u64)) {
        self.cursor = Some(cursor);
        self.candidate = None;
    }
    fn warn(&mut self, message: String) {
        if self.warnings.len() < 8 {
            self.warnings.push(message.chars().take(1024).collect());
        }
    }
    fn view(&self, status: ResolveStatus) -> ResolvedView {
        ResolvedView {
            status,
            rejection: if self.frame.is_some() {
                None
            } else {
                self.rejection
            },
            requested_revision: self.revision,
            record: self.frame.clone(),
            warnings: self.warnings.clone(),
        }
    }
}
/// Bounded pins for displayed frames, separate from the raw chunk LRU.
/// At most eight semantic queries and 64 candidate checks per call are retained.
/// Partial loads belong to one selected run/input/role; changing that selection
/// retires their pins. Completed frames remain available in the bounded cache.
/// Caller-owned views should only be retained for rendering the current frame.
pub struct ProbeResolver {
    slots: VecDeque<Slot>,
    budget: usize,
    leased_bytes: Arc<AtomicUsize>,
}
impl ProbeResolver {
    pub fn new(budget: usize) -> Self {
        Self {
            slots: VecDeque::new(),
            budget,
            leased_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn resident_bytes(&self) -> usize {
        self.leased_bytes
            .load(Ordering::Relaxed)
            .saturating_add(self.slots.iter().map(Slot::pending_bytes).sum::<usize>())
    }
    /// Retain a Loader-validated visitation record in this resolver's existing
    /// byte envelope. Invoke once per immutable key, then clone the returned
    /// lease. This does not scan arrays or create a second raw cache.
    pub fn lease_visitation(
        &mut self,
        run: &RunIndex,
        key: &CacheKey,
        chunk: Arc<Chunk>,
    ) -> Result<LeasedChunk> {
        if !matches!(&chunk.record, Record::Trajectory(_))
            && !matches!(&chunk.record, Record::ProbeSet(p) if p.population == ProbePopulation::PolicyVisited)
        {
            return Err(Error::Config(
                "Visitation lease requires a real trajectory or PolicyVisited population".into(),
            ));
        }
        self.lease_record(run, key, chunk)
    }
    /// The same immutable identity check and shared byte envelope for a Value
    /// critic/probe pair selected by the existing asynchronous source search.
    pub(crate) fn lease_record(
        &mut self,
        run: &RunIndex,
        key: &CacheKey,
        chunk: Arc<Chunk>,
    ) -> Result<LeasedChunk> {
        let index = run
            .by_sequence(key.sequence)
            .ok_or_else(|| Error::State("Missing visitation source identity".into()))?;
        let entry = &run.recording().entries()[index];
        let manifest = run.recording().manifest();
        let identity = &chunk.context.identity;
        if run.key(index)? != *key
            || identity.run_id != manifest.run_id
            || manifest
                .starting_revision
                .checked_add(chunk.context.counters.updates)
                != Some(entry.revision)
            || identity.learner_update_revision != entry.revision
            || chunk.record.kind() != entry.kind
            || identity.model_role != entry.model_role
            || chunk.context.probe_set != entry.probe_set
            || chunk.context.trajectory != entry.trajectory
            || identity.checkpoint_id != entry.checkpoint_id
        {
            return Err(Error::Config(
                "Visitation key and loaded chunk identity differ".into(),
            ));
        }
        let charge = ChunkCache::charge(&chunk);
        if self.resident_bytes().saturating_add(charge) > self.budget {
            return Err(Error::Config(
                "Visitation source exceeds resolver byte budget".into(),
            ));
        }
        self.leased_bytes.fetch_add(charge, Ordering::Relaxed);
        Ok(LeasedChunk {
            chunk,
            _lease: Arc::new(FrameLease {
                account: self.leased_bytes.clone(),
                charge,
            }),
        })
    }
    /// Release cached searches. Returned frames and chunk handles retain their
    /// charges until their owners drop them; clearing cannot reclaim that memory.
    pub fn clear(&mut self) {
        self.slots.clear();
    }
    pub fn resolve_policy(
        &mut self,
        run: &RunIndex,
        selection: &Selection,
        mut load: impl FnMut(usize) -> Result<Option<Arc<Chunk>>>,
    ) -> Result<ResolvedView> {
        let identity = Identity {
            root: run.root().clone(),
            input: selection.input.clone(),
            role: selection.model_role.clone(),
        };
        let set = match &selection.input {
            SelectedInput::ProbeRow { set, .. } => Some(set.as_str()),
            _ => None,
        };
        let frontier = run.candidate_frontier(
            RecordKind::Policy,
            &selection.model_role,
            set,
            selection.revision,
        );
        for previous in &mut self.slots {
            if previous.identity.root != identity.root
                || previous.identity.input != identity.input
                || previous.identity.role != identity.role
            {
                previous.clear_pending();
            }
        }
        let mut slot = self
            .slots
            .iter()
            .position(|s| s.identity == identity)
            .and_then(|i| self.slots.remove(i))
            .unwrap_or(Slot {
                identity,
                rejection: None,
                revision: selection.revision,
                frontier,
                cursor: None,
                finished: false,
                frame: None,
                source: None,
                candidate: None,
                warnings: vec![],
            });
        slot.revision = selection.revision;
        if slot.frontier != frontier {
            slot.frontier = frontier;
            slot.cursor = None;
            slot.finished = false;
            slot.warnings.clear();
            slot.rejection = None;
            if slot
                .frame
                .as_ref()
                .is_some_and(|f| f.revision() > selection.revision)
            {
                slot.frame = None;
            }
        }
        let status = if slot.finished {
            if slot.frame.is_some() {
                ResolveStatus::Ready
            } else {
                ResolveStatus::Unavailable
            }
        } else {
            Self::search(
                run,
                selection,
                &mut slot,
                &mut load,
                &mut self.slots,
                self.budget,
                &self.leased_bytes,
            )?
        };
        while self.slots.len() >= 8 {
            self.slots.pop_front();
        }
        let view = slot.view(status);
        self.slots.push_back(slot);
        Ok(view)
    }
    // Borrow the query and shared accounting without cloning either per search.
    #[allow(clippy::too_many_arguments)]
    fn search(
        run: &RunIndex,
        selection: &Selection,
        slot: &mut Slot,
        load: &mut impl FnMut(usize) -> Result<Option<Arc<Chunk>>>,
        other_slots: &mut VecDeque<Slot>,
        budget: usize,
        leased_bytes: &Arc<AtomicUsize>,
    ) -> Result<ResolveStatus> {
        let (sequence, source_row, set) = match &selection.input {
            SelectedInput::ProbeRow { sequence, row, set } => (*sequence, *row, Some(set.as_str())),
            SelectedInput::TrajectoryStep { sequence, step } => (*sequence, *step, None),
            SelectedInput::None => {
                slot.rejection = Some(ResolveRejection::NoSelection);
                return Ok(slot.finish());
            }
        };
        let indices: Vec<usize> = if let Some(cursor) = slot.cursor {
            run.candidates_before(RecordKind::Policy, &selection.model_role, set, cursor)
                .take(64)
                .collect()
        } else {
            run.candidates(
                RecordKind::Policy,
                &selection.model_role,
                set,
                selection.revision,
            )
            .take(64)
            .collect()
        };
        if indices.is_empty() {
            return Ok(slot.finish());
        }
        let Some(source_index) = run.by_sequence(sequence) else {
            slot.rejection = Some(ResolveRejection::MissingSource);
            slot.warn("Selected input source is missing from the run index".into());
            slot.frame = None;
            return Ok(slot.finish());
        };
        if slot.source.is_none() {
            let source = match load(source_index) {
                Ok(Some(source)) => {
                    if let Err(error) = validate_source(&source, &selection.input, source_row) {
                        slot.rejection = Some(ResolveRejection::InvalidSource);
                        slot.warn(format!("Selected input is invalid: {error}"));
                        slot.frame = None;
                        return Ok(slot.finish());
                    }
                    Pinned::new(source_index, source)
                }
                Ok(None) => return Ok(ResolveStatus::Loading),
                Err(error) => {
                    slot.rejection = Some(ResolveRejection::Unreadable);
                    slot.warn(format!("Selected input is unreadable: {error}"));
                    slot.frame = None;
                    return Ok(slot.finish());
                }
            };
            if !make_room(slot, other_slots, source.charge, budget, leased_bytes) {
                slot.rejection = Some(ResolveRejection::BudgetExceeded);
                slot.warn("Selected input exceeds the resolver working budget".into());
                if source.charge <= budget {
                    return Ok(ResolveStatus::Loading);
                }
                return Ok(slot.finish());
            }
            slot.source = Some(source);
        }
        let source = slot.source.as_ref().unwrap().chunk.clone();
        for index in indices {
            let entry = &run.recording().entries()[index];
            let cursor = (entry.revision, entry.sequence);
            let Some(probe_index) = run.probe_for_record(index) else {
                slot.rejection = Some(ResolveRejection::MissingSource);
                slot.warn(format!(
                    "Record {} has no preceding probe definition",
                    entry.sequence
                ));
                slot.reject(cursor);
                continue;
            };
            if slot.candidate.as_ref().is_some_and(|p| p.index != index) {
                slot.candidate = None;
            }
            if slot.candidate.as_ref().is_none_or(|p| p.index != index) {
                slot.candidate = None;
                let chunk = match load(index) {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => return Ok(ResolveStatus::Loading),
                    Err(error) => {
                        slot.rejection = Some(ResolveRejection::Unreadable);
                        slot.warn(format!("Skipped record {}: {error}", entry.sequence));
                        slot.reject(cursor);
                        continue;
                    }
                };
                let candidate = Pinned::new(index, chunk);
                if !make_room(slot, other_slots, candidate.charge, budget, leased_bytes) {
                    slot.rejection = Some(ResolveRejection::BudgetExceeded);
                    slot.warn(format!(
                        "Record {} and selected input exceed the resolver working budget",
                        entry.sequence
                    ));
                    if slot.pending_bytes().saturating_add(candidate.charge) <= budget {
                        return Ok(ResolveStatus::Loading);
                    }
                    slot.reject(cursor);
                    continue;
                }
                slot.candidate = Some(candidate);
            }
            let chunk = slot.candidate.as_ref().unwrap().chunk.clone();
            let chunk_charge = slot.candidate.as_ref().unwrap().charge;
            let probes = match if probe_index == source_index {
                Ok(Some(source.clone()))
            } else {
                load(probe_index)
            } {
                Ok(Some(probes)) => probes,
                Ok(None) => return Ok(ResolveStatus::Loading),
                Err(error) => {
                    slot.rejection = Some(ResolveRejection::Unreadable);
                    slot.warn(format!(
                        "Skipped probe definition for {}: {error}",
                        entry.sequence
                    ));
                    slot.reject(cursor);
                    continue;
                }
            };
            if let Err(error) = validate_policy_population(&chunk, &probes) {
                slot.rejection = Some(ResolveRejection::PopulationMismatch);
                slot.warn(format!(
                    "Skipped inconsistent record {}: {error}",
                    entry.sequence
                ));
                slot.reject(cursor);
                continue;
            }
            let row = match matching_probe_input(&source, source_row, &probes) {
                Ok(row) => {
                    if row.is_none() {
                        slot.rejection = Some(ResolveRejection::InputMismatch);
                    }
                    row
                }
                Err(error) => {
                    slot.rejection = Some(ResolveRejection::InvalidRecord);
                    slot.warn(format!(
                        "Invalid probe inputs for {}: {error}",
                        entry.sequence
                    ));
                    None
                }
            };
            if let Some(row) = row {
                let charge = chunk_charge.saturating_add(ChunkCache::charge(&probes));
                if charge > budget {
                    slot.rejection = Some(ResolveRejection::BudgetExceeded);
                    slot.warn(format!(
                        "Record {} exceeds the resolved-frame budget",
                        entry.sequence
                    ));
                    slot.reject(cursor);
                    continue;
                }
                drop(source);
                slot.clear_pending();
                slot.frame = None;
                if !make_room(slot, other_slots, charge, budget, leased_bytes) {
                    slot.rejection = Some(ResolveRejection::BudgetExceeded);
                    slot.warn(
                        "Waiting for returned diagnostic frames to release the memory budget"
                            .into(),
                    );
                    return Ok(ResolveStatus::Loading);
                }
                leased_bytes.fetch_add(charge, Ordering::Relaxed);
                let lease = Arc::new(FrameLease {
                    account: leased_bytes.clone(),
                    charge,
                });
                slot.frame = Some(ResolvedRecord {
                    chunk: LeasedChunk {
                        chunk,
                        _lease: lease.clone(),
                    },
                    probes: LeasedChunk {
                        chunk: probes,
                        _lease: lease,
                    },
                    key: run.key(index)?,
                    probe_key: run.key(probe_index)?,
                    row,
                });
                return Ok(slot.finish());
            }
            slot.reject(cursor);
        }
        // Continue the bounded metadata search on the next frame. This also lets
        // the application react to a new selection before scanning old history.
        let more = slot.cursor.is_some_and(|c| {
            run.candidates_before(RecordKind::Policy, &selection.model_role, set, c)
                .next()
                .is_some()
        });
        if more {
            Ok(ResolveStatus::Loading)
        } else {
            Ok(slot.finish())
        }
    }
}

fn validate_source(source: &Chunk, input: &SelectedInput, row: usize) -> Result<()> {
    let observations = match (input, &source.record) {
        (SelectedInput::ProbeRow { set, .. }, Record::ProbeSet(p)) if *set == p.id => {
            &p.observations
        }
        (SelectedInput::TrajectoryStep { .. }, Record::Trajectory(t)) => &t.observations,
        _ => {
            return Err(Error::State(
                "selection and immutable input source differ".into(),
            ));
        }
    };
    if array(source, observations)?
        .shape()
        .first()
        .is_none_or(|&rows| row >= rows)
    {
        return Err(Error::State(
            "selected row is outside its immutable input source".into(),
        ));
    }
    Ok(())
}

/// Reserve pins before retaining them, including all other semantic queries.
/// A frame is sacrificed only when the new dependency can itself fit the budget.
fn make_room(
    slot: &mut Slot,
    others: &mut VecDeque<Slot>,
    additional: usize,
    budget: usize,
    leased_bytes: &AtomicUsize,
) -> bool {
    if slot.pending_bytes().saturating_add(additional) > budget {
        return false;
    }
    while !others.is_empty()
        && others
            .iter()
            .map(Slot::pending_bytes)
            .sum::<usize>()
            .saturating_add(slot.pending_bytes())
            .saturating_add(leased_bytes.load(Ordering::Relaxed))
            .saturating_add(additional)
            > budget
    {
        others.pop_front();
    }
    if leased_bytes
        .load(Ordering::Relaxed)
        .saturating_add(slot.pending_bytes())
        .saturating_add(additional)
        > budget
        && slot.frame.is_some()
    {
        slot.frame = None;
        slot.warn(
            "Held frame released to keep pending diagnostic loads within the memory budget".into(),
        );
    }
    leased_bytes
        .load(Ordering::Relaxed)
        .saturating_add(slot.pending_bytes())
        .saturating_add(additional)
        <= budget
}

fn validate_policy_population(chunk: &Chunk, probes: &Chunk) -> Result<()> {
    let Record::ProbeSet(set) = &probes.record else {
        return Err(Error::State("missing probe definition".into()));
    };
    let Record::Policy(policy) = &chunk.record else {
        return Err(Error::State("expected policy measurement".into()));
    };
    let rows = array(probes, &set.observations)?.shape()[0];
    if array(chunk, &policy.actions)?.shape().first() != Some(&rows) {
        return Err(Error::Shape(
            "measurement and probe population row counts differ".into(),
        ));
    }
    Ok(())
}
