//! Shared-worker preparation for critic values.
mod decision;
mod domain;
mod empirical;
pub use super::profile::PreparedValueProfile;
use super::profile::display_scalar as scalar_at;
pub use super::source::ValueSource;
use super::{
    ensure,
    slice::{ActionConditioning, ValueSlice},
    source::Data,
    visitation::{self, PreparedValueVisits},
};
use crate::policy::slice::SliceMode;
use crate::{
    budgets::{POLICY_FRAME, POLICY_SCRATCH},
    data::CacheKey,
    pane::SelectionIntent,
    policy::prepared::{Accounting, Lease, PolicySource, SharedLease, VisitSource},
};
pub use domain::{CalibrationRequest, CalibrationResult, ColorRanges};
use rl::{
    Error, Result,
    core::{action::Action, env::Observation},
    diagnostics::{
        models::{CriticInspectionRequest, InspectionRequest, ModelBundle},
        storage::ChunkSource,
        value::*,
    },
};
use std::{
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[derive(Clone)]
pub enum ValuePin {
    /// Preserve the state; reevaluate its current actor at each checkpoint.
    State(Observation),
    /// Preserve the complete first input, including all unplotted action components.
    Input {
        observation: Observation,
        action: Option<Action>,
    },
    /// Real immutable probe row or trajectory decision, retained under its lease.
    Recorded { source: VisitSource, row: usize },
    Probe {
        source: VisitSource,
        row: usize,
        action: Option<Action>,
    },
}
impl ValuePin {
    pub fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::State(a), Self::State(b)) => a == b,
            (
                Self::Input {
                    observation: a,
                    action: x,
                },
                Self::Input {
                    observation: b,
                    action: y,
                },
            ) => a == b && x == y,
            (Self::Recorded { source: a, row: x }, Self::Recorded { source: b, row: y }) => {
                a.key == b.key && x == y
            }
            (
                Self::Probe {
                    source: a,
                    row: x,
                    action: aa,
                },
                Self::Probe {
                    source: b,
                    row: y,
                    action: ba,
                },
            ) => a.key == b.key && x == y && aa == ba,
            _ => false,
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Self::State(o) => input_bytes(o, None),
            Self::Input {
                observation,
                action,
            } => input_bytes(observation, action.as_ref()),
            Self::Recorded { .. } => 512,
            Self::Probe { action, .. } => {
                512 + action.as_ref().map_or(0, |a| match a {
                    Action::Continuous(v) => v.capacity() * 4,
                    _ => 0,
                })
            }
        }
    }
}
#[derive(Clone)]
pub struct ValueRequest {
    pub current: ValueSource,
    pub output: ValueOutput,
    pub slice: Arc<ValueSlice>,
    pub requested_revision: u64,
    /// Independent Comparison lookup time; absent for callers without a reference request.
    pub reference_requested_revision: Option<u64>,
    pub reference_value: Option<ValueSource>,
    /// Recorded conditional evidence, bound to this exact current subject/revision.
    pub empirical: Option<Arc<super::empirical::EmpiricalSource>>,
    pub reference_empirical: bool,
    pub empirical_condition: Option<CacheKey>,
    pub empirical_unavailable: Option<String>,
    pub reference_policy: Option<PolicySource>,
    /// Actual learner policy at the requested checkpoint, never a reference overlay.
    pub current_policy: Option<PolicySource>,
    pub(crate) capability_only: bool,
    pub condition: ComparisonCondition,
    pub pin: ValuePin,
    pub visits: Vec<VisitSource>,
    pub calibration: Option<CalibrationRequest>,
}
impl ValueRequest {
    /// Reuse existing metadata owners only for the same immutable scientific
    /// source. The worker still checks source availability before publication.
    pub(crate) fn coalesce_calibration_sources(&mut self) {
        let Some(calibration) = &mut self.calibration else {
            return;
        };
        if calibration.source.same(&self.current) {
            calibration.source = self.current.clone();
        }
        if let (Some(calibration_reference), Some(reference)) =
            (&mut calibration.reference, &self.reference_value)
            && calibration_reference.same(reference)
        {
            *calibration_reference = reference.clone();
        }
    }
    pub(crate) fn withdraw_empirical(&mut self, reason: &str) {
        self.empirical = None;
        self.empirical_condition = None;
        self.empirical_unavailable = Some(reason.into());
    }
    /// Optional evidence must not reject an otherwise admitted ordinary prediction.
    pub(crate) fn withdraw_unadmitted_empirical(&mut self) {
        if self.empirical.is_some() && publication_charge(self).is_err() {
            self.withdraw_empirical("Conditional evidence exceeds the prepared Value frame admission; ordinary critic prediction retained");
        }
    }
    pub(crate) fn decision_loss_available(&self) -> bool {
        decision::compatible(self).is_ok()
            && self
                .reference_value
                .as_ref()
                .is_some_and(|s| s.reference_info.prepared())
    }
    fn controller_state_pin(&self) -> bool {
        if matches!(self.pin, ValuePin::State(_)) {
            return true;
        }
        let (
            ValuePin::Probe {
                source,
                action: None,
                ..
            },
            Data::Recorded(frames),
        ) = (&self.pin, &self.current.data)
        else {
            return false;
        };
        let [frame] = frames.as_slice() else {
            return false;
        };
        let unconditioned = |chunk: &rl::diagnostics::schema::Chunk| matches!(&chunk.record,rl::diagnostics::schema::Record::ProbeSet(p) if p.histories.is_empty() && p.fixed_actions.is_none());
        self.current.kind == ValueCriticKind::StateValue
            && source.key == frame.probe_key
            && unconditioned(&source.chunk)
            && unconditioned(&frame.probes)
            && matches!(&frame.chunk.record,rl::diagnostics::schema::Record::Critic(c) if c.actions.is_none())
    }
    pub fn new(
        current: ValueSource,
        output: ValueOutput,
        slice: ValueSlice,
        pin: ValuePin,
    ) -> Result<Self> {
        let request = Self {
            requested_revision: current.identity.revision,
            reference_requested_revision: None,
            current,
            output,
            slice: Arc::new(slice),
            reference_value: None,
            empirical: None,
            reference_empirical: false,
            empirical_condition: None,
            empirical_unavailable: None,
            reference_policy: None,
            current_policy: None,
            capability_only: false,
            condition: ComparisonCondition::SameInput,
            pin,
            visits: vec![],
            calibration: None,
        };
        request.validate()?;
        Ok(request)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        self.slice.validate(self.current.semantics())?;
        if self.output == ValueOutput::DecisionLoss {
            ensure(
                self.slice.state_only()
                    && matches!(self.slice.conditioning, ActionConditioning::Policy),
                "Decision loss requires state coordinates and current-policy conditioning",
            )?;
        }
        ensure(
            self.current.kind != ValueCriticKind::StateValue || self.slice.state_only(),
            "V requires observation coordinates",
        )?;
        ensure(
            self.condition != ComparisonCondition::ControllerValues
                || self.slice.state_only()
                    && matches!(
                        self.output,
                        ValueOutput::ExpectedReturn | ValueOutput::Q1 | ValueOutput::Q2
                    )
                    && matches!(self.slice.conditioning, ActionConditioning::Policy)
                    && self.controller_state_pin(),
            "ControllerValues requires an explicitly policy-conditioned state-only expected return",
        )?;
        ensure(self.visits.len() <= 16, "too many visit sources")?;
        ensure(
            self.empirical_unavailable
                .as_ref()
                .is_none_or(|e| e.len() <= 512),
            "Empirical unavailable reason exceeds metadata admission",
        )?;
        ensure(
            !self.reference_empirical || self.reference_value.is_none(),
            "Choose one Value reference source",
        )?;
        for source in &self.visits {
            ensure(
                source.key.root == self.current.identity.key.root,
                "visits belong to another recording",
            )?;
        }
        if let ValuePin::Recorded { source, .. } | ValuePin::Probe { source, .. } = &self.pin {
            ensure(
                source.key.root == self.current.identity.key.root,
                "recorded pin belongs to another recording",
            )?;
        }
        if let ValuePin::State(o) | ValuePin::Input { observation: o, .. } = &self.pin {
            o.validate(
                self.current.manifest().observation_dim,
                &self.current.manifest().action_space,
                true,
            )?;
            ensure(
                self.pin.bytes() <= 256 * 1024,
                "Value pin exceeds admission",
            )?;
        }
        if let ValuePin::Input {
            observation,
            action: Some(a),
        } = &self.pin
        {
            self.current
                .manifest()
                .action_space
                .check(a, observation.dense_mask())?;
        }
        batch_capacity(&self.current)?;
        Ok(())
    }
    pub(crate) fn same_grid(&self, b: &Self) -> bool {
        self.current.same(&b.current)
            && self.reference_empirical == b.reference_empirical
            && self.empirical_unavailable == b.empirical_unavailable
            && match (&self.empirical, &b.empirical) {
                (Some(a), Some(b)) => a.same(b),
                (None, None) => true,
                _ => false,
            }
            && self.capability_only == b.capability_only
            && (self.output != ValueOutput::DecisionLoss || self.current.same_payload(&b.current))
            && match (&self.current_policy, &b.current_policy) {
                (Some(a), Some(b)) => {
                    a.same(b)
                        && (self.output != ValueOutput::DecisionLoss || a.same_decision_payload(b))
                }
                (None, None) => true,
                _ => false,
            }
            && self.output == b.output
            && self.slice == b.slice
            && self.condition == b.condition
            && b.calibration.as_ref().is_none_or(|b| {
                self.calibration.as_ref().is_some_and(|a| {
                    a.serial == b.serial
                        && a.source.same(&b.source)
                        && match (&a.reference, &b.reference) {
                            (Some(a), Some(b)) => a.same(b),
                            (None, None) => true,
                            _ => false,
                        }
                })
            })
            && match (&self.reference_value, &b.reference_value) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.same(b) && (self.output != ValueOutput::DecisionLoss || a.same_payload(b))
                }
                _ => false,
            }
            && self.reference_policy.as_ref().map(PolicySource::identity)
                == b.reference_policy.as_ref().map(PolicySource::identity)
            && self
                .visits
                .iter()
                .map(|v| &v.key)
                .eq(b.visits.iter().map(|v| &v.key))
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.slice.owned_bytes()
            + self.empirical.as_ref().map_or(0, |s| s.metadata_bytes)
            + self.empirical_condition.as_ref().map_or(0, |_| 512)
            + self
                .empirical_unavailable
                .as_ref()
                .map_or(0, String::capacity)
            + self.current.metadata_bytes
            + self
                .reference_value
                .as_ref()
                .map_or(0, |s| s.metadata_bytes)
            + self
                .reference_policy
                .as_ref()
                .map_or(0, |s| s.shared_metadata().1)
            + self
                .current_policy
                .as_ref()
                .map_or(0, |s| s.shared_metadata().1)
            + self.pin.bytes()
            + self.calibration.as_ref().map_or(0, |c| {
                c.source.metadata_bytes + c.reference.as_ref().map_or(0, |s| s.metadata_bytes)
            })
            + self.visits.capacity() * 512
            + 4096
    }
    pub(crate) fn view_owned_bytes(&self, charges: &mut super::source::ViewSourceCharges) -> usize {
        let mut bytes = self.metadata_bytes();
        for source in std::iter::once(&self.current)
            .chain(self.reference_value.as_ref())
            .chain(self.calibration.as_ref().map(|c| &c.source))
            .chain(self.calibration.as_ref().and_then(|c| c.reference.as_ref()))
        {
            bytes = bytes - source.metadata_bytes + charges.charge(source);
        }
        bytes
    }
}
pub type ValueMetrics = crate::policy::prepared::PolicyMetrics;
pub(crate) struct RequestOwner {
    pub request: ValueRequest,
    _lease: Lease,
    shared: Vec<SharedLease>,
}
impl RequestOwner {
    fn metadata_layout(request: &ValueRequest) -> (usize, Vec<(usize, usize)>) {
        let mut allocations = vec![(
            Arc::as_ptr(&request.slice) as usize,
            request.slice.owned_bytes(),
        )];
        let mut repeated_private = 0usize;
        for source in std::iter::once(&request.current)
            .chain(request.reference_value.as_ref())
            .chain(request.calibration.as_ref().map(|c| &c.source))
            .chain(
                request
                    .calibration
                    .as_ref()
                    .and_then(|c| c.reference.as_ref()),
            )
        {
            if allocations
                .iter()
                .any(|(key, _)| *key == Arc::as_ptr(&source.manifest) as usize)
            {
                repeated_private = repeated_private.saturating_add(source.private_metadata_bytes());
            }
            allocations.push((
                Arc::as_ptr(&source.manifest) as usize,
                source.metadata_bytes,
            ));
        }
        for policy in request
            .reference_policy
            .iter()
            .chain(request.current_policy.iter())
        {
            allocations.push(policy.shared_metadata());
        }
        if let Some(source) = &request.empirical {
            allocations.push((Arc::as_ptr(source) as usize, source.metadata_bytes));
        }
        let private = request.metadata_bytes()
            - allocations.iter().map(|(_, bytes)| bytes).sum::<usize>()
            + repeated_private;
        (private, allocations)
    }
    /// Charge each actual shared allocation once, without assuming that any
    /// owner held by a predecessor will survive its retirement.
    pub(crate) fn standalone_metadata_charge(request: &ValueRequest) -> usize {
        let (private, allocations) = Self::metadata_layout(request);
        private
            + allocations
                .iter()
                .enumerate()
                .filter(|(index, (key, _))| {
                    !allocations[..*index]
                        .iter()
                        .any(|(previous, _)| previous == key)
                })
                .map(|(_, (_, bytes))| *bytes)
                .sum::<usize>()
    }
    pub(crate) fn metadata_charge(
        request: &ValueRequest,
        accounting: &Accounting,
    ) -> Result<usize> {
        let (private, allocations) = Self::metadata_layout(request);
        Ok(private + SharedLease::additional_charge(accounting, &allocations)?)
    }

    pub(crate) fn metadata_charge_after(
        request: &ValueRequest,
        accounting: &Accounting,
        retiring: &[SharedLease],
    ) -> Result<usize> {
        let (private, allocations) = Self::metadata_layout(request);
        Ok(private + SharedLease::additional_charge_after(accounting, &allocations, retiring)?)
    }

    pub(crate) fn new(request: ValueRequest, accounting: &Arc<Accounting>) -> Result<Arc<Self>> {
        request.validate()?;
        let (bytes, allocations) = Self::metadata_layout(&request);
        let shared = allocations
            .into_iter()
            .map(|(key, bytes)| SharedLease::acquire(accounting, key, bytes))
            .collect::<Result<Vec<_>>>()?;
        let lease = Lease::acquire(accounting, bytes)?;
        Ok(Arc::new(Self {
            request,
            _lease: lease,
            shared,
        }))
    }
    fn additional_bytes(&self, retained: &Self) -> usize {
        self._lease.bytes
            + self
                .shared
                .iter()
                .filter(|s| !retained.shared.iter().any(|r| r.key == s.key))
                .map(|s| s.bytes)
                .sum::<usize>()
    }
}
#[derive(Clone)]
pub struct PreparedValueFrame {
    pub grid: Arc<PreparedValueGrid>,
    pub selected: Option<Arc<PreparedValuePoint>>,
    pub requested_revision: u64,
    pub reference_requested_revision: Option<u64>,
}
pub struct PreparedValueGrid {
    /// Robust limits of the displayed slice for Adaptive or provisional use.
    /// Frozen historical limits, when requested, are returned in `calibration`.
    pub color_domain: Option<[f64; 2]>,
    pub reference_color_domain: Option<[f64; 2]>,
    pub difference_color_domain: Option<[f64; 2]>,
    pub calibration: Option<CalibrationResult>,
    pub population_count: usize,
    /// Visible coordinate count, capped at five for sparse presentation.
    pub distinct_positions: usize,
    pub slice: Arc<ValueSlice>,
    pub values: Vec<f64>,
    pub reference_values: Vec<Option<f64>>,
    pub reference_actions: Vec<Option<Action>>,
    pub positions: Vec<[f32; 2]>,
    pub actions: Vec<Option<Action>>,
    pub policy_overlay: Vec<Option<Action>>,
    pub reference_policy_overlay: Vec<Option<Action>>,
    pub reference_policy_identity: Option<PreparedPolicyIdentity>,
    pub rows: Vec<usize>,
    pub valid: Vec<bool>,
    pub categories: Vec<usize>,
    pub reference_series: Option<PreparedValueSeries>,
    pub empirical: super::empirical::EmpiricalBatch,
    /// Sparse differences evaluated at the original empirical inputs/actions.
    pub empirical_differences: Option<PreparedValueSeries>,
    pub empirical_error: Option<String>,
    pub reference_policy_series: Option<PreparedPolicySeries>,
    pub reference_error: Option<String>,
    pub reference_policy_error: Option<String>,
    pub visits: Option<PreparedValueVisits>,
    owner: Arc<RequestOwner>,
    _lease: Lease,
    charge: usize,
}
#[derive(Clone)]
pub struct PreparedValueSeries {
    pub values: Vec<f64>,
    pub positions: Vec<[f32; 2]>,
    pub rows: Vec<usize>,
    pub actions: Vec<Option<Action>>,
}
#[derive(Clone)]
pub struct PreparedPolicyIdentity {
    pub name: String,
    pub role: String,
    pub stochastic: bool,
}
#[derive(Clone)]
pub struct PreparedPolicySeries {
    pub positions: Vec<[f32; 2]>,
    pub actions: Vec<Action>,
    pub rows: Vec<usize>,
    pub stochastic: bool,
}
impl PreparedValueGrid {
    pub fn promote_selected(&self) -> bool {
        self.population_count > 0
            && (self.slice.mode == SliceMode::SamplesOnly || self.distinct_positions <= 4)
    }
    pub fn recorded_population(&self) -> Option<&rl::diagnostics::schema::ProbeSetRecord> {
        let Data::Recorded(frames) = &self.current().data else {
            return None;
        };
        let rl::diagnostics::schema::Record::ProbeSet(p) = &frames[0].probes.record else {
            return None;
        };
        Some(p)
    }
    pub fn population_selection(&self, row: usize) -> Option<SelectionIntent> {
        if row >= self.population_count {
            return None;
        }
        let Data::Recorded(frames) = &self.current().data else {
            return None;
        };
        let rl::diagnostics::schema::Record::ProbeSet(p) = &frames[0].probes.record else {
            return None;
        };
        Some(SelectionIntent::Input(
            crate::data::SelectedInput::ProbeRow {
                set: p.id.clone(),
                row,
                sequence: frames[0].probe_key.sequence,
            },
        ))
    }
    pub fn output(&self) -> &ValueOutput {
        &self.owner.request.output
    }
    pub fn comparison_condition(&self) -> ComparisonCondition {
        self.owner.request.condition
    }
    pub fn current(&self) -> &ValueSource {
        &self.owner.request.current
    }
    pub fn reference(&self) -> Option<&ValueSource> {
        self.owner.request.reference_value.as_ref()
    }
    pub fn empirical_reference(&self) -> bool {
        self.owner.request.reference_empirical
    }
    pub(crate) fn empirical_source(&self) -> Option<&Arc<super::empirical::EmpiricalSource>> {
        self.owner.request.empirical.as_ref()
    }
    pub fn regular(&self) -> bool {
        self.slice.mode != SliceMode::SamplesOnly && matches!(self.current().data, Data::Model(_))
    }
    pub fn reference_policy(&self) -> Option<&PolicySource> {
        self.owner.request.reference_policy.as_ref()
    }
    pub fn input(&self, index: usize) -> Result<Observation> {
        ensure(index < self.values.len(), "Value cell out of bounds")?;
        if let Data::Recorded(frames) = &self.current().data {
            crate::policy::prepared::exact_observation(&frames[0].probes, self.rows[index])
        } else {
            self.slice.observation_at(index)
        }
    }
    pub fn difference_at(&self, index: usize) -> Option<f64> {
        Some(*self.values.get(index)? - self.reference_values.get(index).copied().flatten()?)
    }
    pub fn pin_at(&self, index: usize) -> Result<(ValuePin, ValueSlice)> {
        let observation = self.input(index)?;
        let mut slice = (*self.slice).clone();
        let action = self.actions.get(index).cloned().flatten();
        if let Data::Recorded(frames) = &self.current().data {
            let mut state = crate::policy::slice::SliceConfig::new(
                self.current().semantics(),
                &observation,
                &[],
            )?;
            state.ranges.clone_from(&slice.state.ranges);
            state.range_sources.clone_from(&slice.state.range_sources);
            state.tolerance = slice.state.tolerance;
            if slice.state.grid_eligibility() == crate::policy::slice::GridEligibility::HistoryBound
            {
                state = state.with_history();
            }
            slice.state = state;
            if let Some(a) = &action
                && self.output() != &ValueOutput::DecisionLoss
            {
                slice.conditioning = ActionConditioning::Fixed(a.clone());
            }
            return Ok((
                ValuePin::Probe {
                    source: VisitSource {
                        key: frames[0].probe_key.clone(),
                        chunk: frames[0].probes.clone(),
                    },
                    row: self.rows[index],
                    action,
                },
                slice,
            ));
        }
        if slice.state_only() && matches!(slice.conditioning, ActionConditioning::Policy) {
            Ok((ValuePin::State(observation), slice))
        } else {
            if let Some(a) = &action {
                slice.conditioning = ActionConditioning::Fixed(a.clone());
            }
            Ok((
                ValuePin::Input {
                    observation,
                    action,
                },
                slice,
            ))
        }
    }
    pub fn select_visit(&self, index: usize) -> Result<(ValuePin, ValueSlice, SelectionIntent)> {
        self.visits
            .as_ref()
            .ok_or_else(|| Error::State("no visits".into()))?
            .select_visit(
                index,
                &self.slice,
                self.current().kind != ValueCriticKind::StateValue,
            )
    }
    pub(crate) fn matches_request(&self, r: &ValueRequest) -> bool {
        self.owner.request.same_grid(r)
    }
}
pub struct PreparedValuePoint {
    pub input: Observation,
    pub action: Option<Action>,
    /// Selected-state critic values along one bounded continuous action component.
    pub action_curve: Option<PreparedActionCurve>,
    pub prediction: ValuePrediction,
    pub decision_loss: Option<rl::diagnostics::reference::DecisionLoss>,
    pub value: f64,
    pub reference_prediction: Option<ValuePrediction>,
    pub profile: Option<PreparedValueProfile>,
    pub reference_profile: Option<PreparedValueProfile>,
    pub reference_value: Option<f64>,
    pub reference_action: Option<Action>,
    pub reference_error: Option<String>,
    pub empirical: Option<PreparedEmpirical>,
    pub empirical_error: Option<String>,
    pub observed_return: Option<f64>,
    pub final_observation: bool,
    pub recorded_revision: Option<u64>,
    pub recorded_key: Option<CacheKey>,
    pub recorded_row: Option<usize>,
    pub policy_conditioned: bool,
    pub neighborhood_count: u32,
    _owner: Arc<RequestOwner>,
    _lease: Lease,
}
pub struct PreparedActionCurve {
    pub component: usize,
    pub points: Vec<[f64; 2]>,
}
pub struct PreparedEmpirical {
    pub condition: super::empirical::EmpiricalCondition,
    pub cdf: Vec<[f64; 2]>,
    pub histogram: Option<rl::diagnostics::analysis::Histogram>,
    pub wasserstein: Option<f64>,
    pub mean_crps: Option<f64>,
}
impl PreparedValueFrame {
    pub(crate) fn retirement_bytes(&self, desired: Option<&Arc<RequestOwner>>) -> usize {
        let grid_retires = Arc::strong_count(&self.grid) == 1;
        let point_retires = self
            .selected
            .as_ref()
            .is_some_and(|p| Arc::strong_count(p) == 1);
        let mut bytes = if grid_retires { self.grid.charge } else { 0 }
            + self
                .selected
                .as_ref()
                .filter(|_| point_retires)
                .map_or(0, |p| p._lease.bytes);
        // A pin edit can give Selected a different RequestOwner from its grid.
        // Count every known retiring internal Arc; external holders remain charged.
        let mut owners: Vec<(&Arc<RequestOwner>, usize)> =
            vec![(&self.grid.owner, usize::from(grid_retires))];
        for (owner, count) in self
            .selected
            .iter()
            .map(|p| (&p._owner, usize::from(point_retires)))
            .chain(desired.map(|d| (d, 1)))
        {
            if let Some((_, known)) = owners.iter_mut().find(|(old, _)| Arc::ptr_eq(old, owner)) {
                *known += count;
            } else {
                owners.push((owner, count));
            }
        }
        let retired: Vec<_> = owners
            .iter()
            .filter(|(owner, known)| Arc::strong_count(owner) == *known)
            .map(|(o, _)| *o)
            .collect();
        bytes += retired.iter().map(|o| o._lease.bytes).sum::<usize>();
        let shared: Vec<_> = retired.iter().flat_map(|o| &o.shared).collect();
        bytes + SharedLease::exclusive_charge(&shared)
    }
}
impl PreparedValuePoint {
    pub(crate) fn pin(&self) -> &ValuePin {
        &self._owner.request.pin
    }
    pub fn current(&self) -> &ValueSource {
        &self._owner.request.current
    }
    pub fn current_policy(&self) -> Option<&PolicySource> {
        self._owner.request.current_policy.as_ref()
    }
    pub fn reference(&self) -> Option<&ValueSource> {
        self._owner.request.reference_value.as_ref()
    }
    pub fn recorded_evaluation(&self) -> Option<&rl::diagnostics::schema::EvaluationMetadata> {
        if let ValuePin::Recorded { source, .. } = &self._owner.request.pin
            && let rl::diagnostics::schema::Record::Trajectory(t) = &source.chunk.record
        {
            Some(&t.evaluation)
        } else {
            None
        }
    }
    pub fn recorded_episode(&self) -> Option<&str> {
        if let ValuePin::Recorded { source, .. } = &self._owner.request.pin
            && let rl::diagnostics::schema::Record::Trajectory(t) = &source.chunk.record
        {
            Some(&t.id)
        } else {
            None
        }
    }
}
pub(crate) enum Work {
    Grid(Arc<RequestOwner>),
    Point {
        request: Arc<RequestOwner>,
        grid: Arc<PreparedValueGrid>,
    },
}
pub(crate) enum Completion {
    Grid(Arc<PreparedValueGrid>),
    Point(Arc<PreparedValuePoint>),
}
pub(crate) type WithBundle<'a> = dyn FnMut(&CacheKey, &ChunkSource, &mut dyn FnMut(&ModelBundle) -> Result<()>) -> Result<()>
    + 'a;
pub(crate) fn check_sources(r: &ValueRequest) -> Result<()> {
    r.current.check_available()?;
    if r.output == ValueOutput::DecisionLoss {
        decision::compatible(r)?;
    }
    Ok(())
}
/// Recheck optional capabilities before publishing a completion. An externally
/// retained immutable grid is copied only after a second shared lease is admitted.
pub(crate) fn sanitize_optional(
    grid: &mut Arc<PreparedValueGrid>,
    point: Option<&mut Arc<PreparedValuePoint>>,
    request: &ValueRequest,
    accounting: &Arc<Accounting>,
) -> Result<()> {
    if request.output == ValueOutput::DecisionLoss {
        return check_sources(request);
    }
    let empirical_error = request
        .empirical
        .as_ref()
        .and_then(|s| s.available().err())
        .map(super::empirical::error_text);
    let value_error = request
        .reference_value
        .as_ref()
        .and_then(|s| s.check_available().err())
        .map(|e| e.to_string())
        .or_else(|| {
            request
                .reference_empirical
                .then(|| empirical_error.clone())
                .flatten()
        });
    let policy_error = request
        .reference_policy
        .as_ref()
        .and_then(|s| s.check_available().err())
        .map(|e| e.to_string());
    if value_error.is_none() && policy_error.is_none() && empirical_error.is_none() {
        return Ok(());
    }
    let value_changed = value_error.as_ref().is_some_and(|error| {
        grid.reference_error.as_ref() != Some(error)
            || !grid.reference_values.is_empty()
            || !grid.reference_actions.is_empty()
            || grid.reference_series.is_some()
    });
    let policy_changed = policy_error.as_ref().is_some_and(|error| {
        grid.reference_policy_error.as_ref() != Some(error)
            || !grid.reference_policy_overlay.is_empty()
            || grid.reference_policy_series.is_some()
    });
    if value_changed || policy_changed || empirical_error.is_some() {
        if Arc::get_mut(grid).is_none() {
            let lease = Lease::acquire(accounting, grid.charge)?;
            *grid = Arc::new(PreparedValueGrid {
                color_domain: grid.color_domain,
                difference_color_domain: if value_error.is_some() {
                    None
                } else {
                    grid.difference_color_domain
                },
                calibration: grid.calibration.clone(),
                reference_color_domain: if value_error.is_some() {
                    None
                } else {
                    grid.reference_color_domain
                },
                population_count: grid.population_count,
                distinct_positions: grid.distinct_positions,
                slice: grid.slice.clone(),
                values: grid.values.clone(),
                reference_values: if value_error.is_some() {
                    vec![]
                } else {
                    grid.reference_values.clone()
                },
                reference_actions: if value_error.is_some() {
                    vec![]
                } else {
                    grid.reference_actions.clone()
                },
                positions: grid.positions.clone(),
                actions: grid.actions.clone(),
                policy_overlay: grid.policy_overlay.clone(),
                reference_policy_overlay: if policy_error.is_some() {
                    vec![]
                } else {
                    grid.reference_policy_overlay.clone()
                },
                reference_policy_identity: if policy_error.is_some() {
                    None
                } else {
                    grid.reference_policy_identity.clone()
                },
                rows: grid.rows.clone(),
                valid: grid.valid.clone(),
                categories: grid.categories.clone(),
                reference_series: if value_error.is_some() {
                    None
                } else {
                    grid.reference_series.clone()
                },
                empirical: if empirical_error.is_some() {
                    Default::default()
                } else {
                    grid.empirical.clone()
                },
                empirical_differences: if value_error.is_some() {
                    None
                } else {
                    grid.empirical_differences.clone()
                },
                empirical_error: grid.empirical_error.clone(),
                reference_policy_series: if policy_error.is_some() {
                    None
                } else {
                    grid.reference_policy_series.clone()
                },
                reference_error: grid.reference_error.clone(),
                reference_policy_error: grid.reference_policy_error.clone(),
                visits: grid.visits.clone(),
                owner: grid.owner.clone(),
                _lease: lease,
                charge: grid.charge,
            });
        }
        let grid = Arc::get_mut(grid)
            .ok_or_else(|| Error::State("Value completion ownership changed".into()))?;
        if let Some(error) = &value_error {
            grid.reference_color_domain = None;
            grid.difference_color_domain = None;
            grid.reference_values.clear();
            grid.reference_actions.clear();
            grid.reference_series = None;
            grid.empirical_differences = None;
            grid.reference_error = Some(error.clone());
        }
        if let Some(error) = &empirical_error {
            grid.empirical = Default::default();
            grid.empirical_error = Some(error.clone());
        }
        if let Some(error) = policy_error {
            grid.reference_policy_overlay.clear();
            grid.reference_policy_identity = None;
            grid.reference_policy_series = None;
            grid.reference_policy_error = Some(error);
        }
    }
    if let Some(point) = point.filter(|_| value_error.is_some() || empirical_error.is_some()) {
        let point = Arc::get_mut(point)
            .ok_or_else(|| Error::State("Value point completion ownership changed".into()))?;
        if let Some(error) = value_error {
            point.reference_prediction = None;
            point.reference_profile = None;
            point.reference_value = None;
            point.reference_action = None;
            point.reference_error = Some(error);
        }
        if let Some(error) = empirical_error {
            point.empirical = None;
            point.empirical_error = Some(error);
        }
    }
    Ok(())
}
pub(crate) fn execute(
    work: &Work,
    accounting: &Arc<Accounting>,
    cancelled: impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<Completion>> {
    let start = Instant::now();
    let inference = accounting.inference_micros.load(Ordering::Acquire);
    let restore = accounting.restore_micros.load(Ordering::Acquire);
    let result = match work {
        Work::Grid(owner) => prepare_grid(owner, accounting, &cancelled, with_bundle)
            .map(|g| g.map(Completion::Grid)),
        Work::Point { request, grid } => {
            prepare_point(request, grid, accounting, &cancelled, with_bundle)
                .map(|p| p.map(Completion::Point))
        }
    };
    accounting.preparation_micros.fetch_add(
        (start.elapsed().as_micros() as u64)
            .saturating_sub(accounting.inference_micros.load(Ordering::Acquire) - inference)
            .saturating_sub(accounting.restore_micros.load(Ordering::Acquire) - restore),
        Ordering::AcqRel,
    );
    result
}
fn input_bytes(o: &Observation, a: Option<&Action>) -> usize {
    128 + o.as_dense().map_or(0, |dense| {
        dense.values.capacity() * 4 + dense.action_mask.as_ref().map_or(0, Vec::capacity)
    }) + a.map_or(0, |a| match a {
        Action::Continuous(a) => a.capacity() * 4,
        Action::Discrete(_) | Action::Conditional(_) => 0,
    })
}
pub fn batch_capacity(source: &ValueSource) -> Result<usize> {
    let row = source
        .manifest()
        .observation_dim
        .checked_mul(64)
        .and_then(|n| {
            source
                .manifest()
                .action_space
                .dimensions()
                .checked_mul(96)
                .and_then(|a| n.checked_add(a))
        })
        .and_then(|n| {
            source
                .heads
                .checked_mul(96 * source.names.len())
                .and_then(|h| n.checked_add(h))
        })
        .and_then(|n| n.checked_add(2048))
        .ok_or_else(|| Error::Config("Value batch overflow".into()))?;
    let n = ((POLICY_SCRATCH - 256 * 1024) / row).min(1024);
    ensure(n > 0, "Value row exceeds scratch admission")?;
    Ok(n)
}

fn reference_compatibility(r: &ValueRequest) -> Result<()> {
    let reference = r
        .reference_value
        .as_ref()
        .ok_or_else(|| Error::Config("no value reference".into()))?;
    reference.check_available()?;
    r.current
        .reference_info
        .compatible(&reference.reference_info)?;
    ensure(
        crate::policy::prepared::compatible_encoding(
            r.current.identity.key.root == reference.identity.key.root,
            r.current.semantics(),
            reference.semantics(),
        ),
        "reference input encodings unknown or different",
    )?;
    let current = r
        .current
        .declaration
        .as_ref()
        .ok_or_else(|| Error::Config("current value conventions unknown".into()))?;
    let other = reference
        .declaration
        .as_ref()
        .ok_or_else(|| Error::Config("reference value conventions unknown".into()))?;
    let output = match r.output {
        ValueOutput::Q1 | ValueOutput::Q2 => ValueOutput::ExpectedReturn,
        _ => r.output.clone(),
    };
    let name = if r.output == ValueOutput::Q2 && r.current.names.len() == 2 {
        &r.current.names[1]
    } else {
        &r.current.names[0]
    };
    let reference_name = if r.output == ValueOutput::Q2 && reference.names.len() == 2 {
        &reference.names[1]
    } else {
        &reference.names[0]
    };
    if r.condition == ComparisonCondition::ControllerValues {
        for (source, declaration, name) in [
            (&r.current, &**current, name),
            (reference, &**other, reference_name),
        ] {
            if !matches!(
                source.kind,
                ValueCriticKind::StateValue | ValueCriticKind::DiscreteQ
            ) {
                ensure(
                    declaration
                        .critics
                        .get(name)
                        .and_then(|c| c.actor.as_ref())
                        .is_some_and(|a| !a.stochastic)
                        && source.actor.is_some()
                        && !source.actor_stochastic,
                    "controller-value Q requires a known deterministic actor",
                )?;
            }
        }
    }
    compatible_values(current, name, other, reference_name, &output, r.condition)
}
fn reference_output(r: &ValueRequest) -> ValueOutput {
    if matches!(r.output, ValueOutput::Q1 | ValueOutput::Q2)
        && r.reference_value
            .as_ref()
            .is_some_and(|s| s.names.len() == 1)
    {
        ValueOutput::ExpectedReturn
    } else {
        r.output.clone()
    }
}
fn stop(cancelled: &impl Fn() -> bool, a: &Accounting) -> bool {
    if cancelled() {
        a.cancellations.fetch_add(1, Ordering::AcqRel);
        true
    } else {
        false
    }
}
fn scalar_predictions(
    predictions: &[ValuePrediction],
    source: &ValueSource,
    output: &ValueOutput,
    actions: &[Option<Action>],
) -> Result<Vec<f64>> {
    if *output == ValueOutput::Utility && source.kind == ValueCriticKind::MomentQ {
        let mut means = Vec::with_capacity(predictions.len());
        let mut seconds = Vec::with_capacity(predictions.len());
        for p in predictions {
            let ValuePrediction::Moments { mean, raw_second } = p else {
                return Err(Error::Config("not moments".into()));
            };
            means.push(*mean);
            seconds.push(*raw_second);
        }
        return moment_utilities(
            &means,
            &seconds,
            source
                .utility()
                .ok_or_else(|| Error::Config("unknown moment utility".into()))?,
        );
    }
    predictions
        .iter()
        .zip(actions)
        .map(|(p, a)| scalar_at(p, a.as_ref(), output, source.utility()))
        .collect()
}
struct Batch {
    predictions: Vec<ValuePrediction>,
    actions: Vec<Option<Action>>,
    overlay: Vec<Option<Action>>,
}
fn actor_actions(
    accounting: &Accounting,
    source: &ValueSource,
    model: &ModelBundle,
    observations: &[Observation],
    required: bool,
) -> Result<Vec<Option<Action>>> {
    let Some(name) = &source.actor else {
        ensure(
            !required,
            "action-conditioned Q requires an explicit or unambiguous actor",
        )?;
        return Ok(vec![None; observations.len()]);
    };
    ensure(
        !required || !source.actor_stochastic,
        "stochastic Q expectation is unavailable; choose an explicit complete action",
    )?;
    let request = InspectionRequest::new(name, observations.to_vec());
    let started = Instant::now();
    let result = model.inspect(&request);
    accounting
        .inference_micros
        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
    let result = result?;
    Ok(result.actions.into_iter().map(Some).collect())
}
fn evaluate_model(
    accounting: &Accounting,
    source: &ValueSource,
    model: &ModelBundle,
    observations: &[Observation],
    actions: &[Option<Action>],
    overlay: Option<Vec<Option<Action>>>,
) -> Result<Batch> {
    ensure(
        model.configuration().observation_dim == source.manifest().observation_dim
            && model.configuration().action_space == source.manifest().action_space,
        "restored model encoding mismatch",
    )?;
    let action_conditioned = matches!(
        source.kind,
        ValueCriticKind::ScalarQ
            | ValueCriticKind::QuantileQ
            | ValueCriticKind::MomentQ
            | ValueCriticKind::CategoricalQ
            | ValueCriticKind::CategoricalReturnQ
    );
    let overlay = match overlay {
        Some(v) => v,
        None => actor_actions(
            accounting,
            source,
            model,
            observations,
            action_conditioned && actions.iter().any(Option::is_none),
        )?,
    };
    let mut actions: Vec<_> = actions
        .iter()
        .zip(&overlay)
        .map(|(a, p)| {
            if action_conditioned {
                a.clone().or_else(|| p.clone())
            } else {
                a.clone()
            }
        })
        .collect();
    let fixed_actions = if action_conditioned {
        Some(
            actions
                .iter()
                .map(|a| {
                    a.clone()
                        .ok_or_else(|| Error::Config("missing Q first action".into()))
                })
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };
    let request = CriticInspectionRequest {
        critics: source.names.clone(),
        observations: observations.to_vec(),
        fixed_actions,
    };
    let started = Instant::now();
    let result = model.inspect_critics(&request);
    accounting
        .inference_micros
        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
    let result = result?;
    let mut predictions = Vec::with_capacity(observations.len());
    for (row, o) in observations.iter().enumerate() {
        let legal = if source.kind == ValueCriticKind::DiscreteQ {
            Some(
                o.dense_mask()
                    .map(<[bool]>::to_vec)
                    .unwrap_or_else(|| vec![true; source.heads]),
            )
        } else {
            None
        };
        let mut p =
            ValuePrediction::from_critic(&result.critics[&source.names[0]], row, legal.as_deref())?;
        if source.names.len() == 2 {
            p = ValuePrediction::Twin {
                q1: Box::new(p),
                q2: Box::new(ValuePrediction::from_critic(
                    &result.critics[&source.names[1]],
                    row,
                    None,
                )?),
            };
        }
        if let ValuePrediction::Discrete { values, legal } = &p {
            let action = match actions[row] {
                Some(Action::Discrete(a)) => a,
                Some(_) => return Err(Error::Action("continuous action for DDQN".into())),
                None => values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| legal[*i])
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .ok_or_else(|| Error::Action("no legal action".into()))?,
            };
            p.select_discrete(action)?;
            actions[row] = Some(Action::Discrete(action));
        }
        predictions.push(p);
    }
    Ok(Batch {
        predictions,
        actions,
        overlay,
    })
}
fn prepared_charge(r: &ValueRequest, n: usize) -> Result<usize> {
    let width = r.current.manifest().action_space.dimensions();
    let row = 48usize
        .saturating_add(4usize.saturating_mul(32usize.saturating_add(width.saturating_mul(4))));
    let sparse_reference = r
        .reference_value
        .as_ref()
        .filter(|s| s.identity.recorded && !s.finite_reference())
        .map(recorded_rows)
        .transpose()?
        .unwrap_or(0);
    ensure(
        sparse_reference <= 8192,
        "sparse reference rows exceed admission",
    )?;
    let sparse_reference = sparse_reference.saturating_mul(
        r.reference_value
            .as_ref()
            .filter(|s| s.kind == ValueCriticKind::DiscreteQ && discrete_axis(&r.slice))
            .map_or(1, |s| s.heads),
    );
    let sparse_policy = r
        .reference_policy
        .as_ref()
        .filter(|s| !s.finite_reference())
        .and_then(PolicySource::recorded_frame)
        .map(|f| visitation::input_rows(&f.probes))
        .transpose()?
        .unwrap_or(0);
    ensure(sparse_policy <= 8192, "sparse policy rows exceed admission")?;
    // Visitation retains one globally capped marker vector, not 1024 markers
    // per source. Admit all source rows before preparation and include the
    // vector's geometric capacity growth without dropping any admitted row.
    let visit_rows = r.visits.iter().try_fold(0usize, |n, source| {
        Ok::<_, Error>(n.saturating_add(visitation::input_rows(&source.chunk)?))
    })?;
    ensure(
        visit_rows <= 8192,
        "visit population exceeds 8192 decisions",
    )?;
    let marker_capacity = if visit_rows == 0 {
        0
    } else {
        visit_rows.min(1024).next_power_of_two().max(4)
    };
    let visit_counts = if r.visits.is_empty() {
        0
    } else {
        r.slice.len()? * std::mem::size_of::<u32>()
    };
    let bytes = n
        .checked_mul(row)
        .and_then(|b| {
            b.checked_add(
                marker_capacity * 96
                    + visit_counts
                    + 4096
                    + sparse_reference * (64 + width * 4)
                    + r.current.heads * 8
                    + sparse_policy * (48 + width * 4),
            )
        })
        .and_then(|b| b.checked_add(empirical::grid_charge(r)))
        .ok_or_else(|| Error::Config("Value grid byte overflow".into()))?;
    ensure(
        bytes.saturating_add(r.metadata_bytes()) <= POLICY_FRAME,
        "Value frame exceeds 2 MiB; reduce resolution or dimensions",
    )?;
    Ok(bytes)
}
/// Preallocation estimate using the worker's actual grid and selected charges.
pub(crate) fn publication_charge(r: &ValueRequest) -> Result<usize> {
    let bytes = prepared_charge(r, r.slice.len()?)? + point_charge(r) + r.metadata_bytes();
    ensure(bytes <= POLICY_FRAME, "selected Value frame exceeds 2 MiB")?;
    Ok(bytes)
}
/// Before the empirical grid is prepared, retain only its fixed selected-point
/// allowance. The CDF is admitted separately from verified grid conditions.
pub(crate) fn deferred_publication_charge(r: &ValueRequest) -> Result<usize> {
    Ok(publication_charge(r)? - empirical::point_charge(r)
        + empirical::point_charge_for_samples(r, 0))
}
pub(crate) fn point_charge_for_grid(r: &ValueRequest, grid: &PreparedValueGrid) -> Result<usize> {
    ensure(
        r.empirical.is_none() || grid.matches_request(r),
        "Empirical point reservation requires the matching Value grid",
    )?;
    let samples = grid
        .empirical
        .conditions
        .iter()
        .map(|c| c.summary.count)
        .max()
        .unwrap_or(0);
    ensure(
        samples <= rl::diagnostics::empirical::MAX_EMPIRICAL_SAMPLES,
        "Empirical condition exceeds sample admission",
    )?;
    Ok(point_charge(r) - empirical::point_charge(r)
        + empirical::point_charge_for_samples(r, samples))
}
fn point_charge(r: &ValueRequest) -> usize {
    4096 + r.pin.bytes()
        + r.current.heads * 64
        + if action_curve_eligible(r) { 65 * 16 } else { 0 }
        + r.reference_value.as_ref().map_or(0, |s| s.heads * 64)
        + empirical::point_charge(r)
}

fn action_curve_eligible(r: &ValueRequest) -> bool {
    r.slice.mode == SliceMode::TwoDimensional
        && r.slice.state_only()
        && r.output != ValueOutput::DecisionLoss
        && matches!(&r.current.data, Data::Model(_))
        && matches!(
            &r.current.manifest().action_space,
            rl::core::action::ActionSpace::Box { .. }
        )
        && matches!(
            r.current.kind,
            ValueCriticKind::ScalarQ
                | ValueCriticKind::QuantileQ
                | ValueCriticKind::MomentQ
                | ValueCriticKind::CategoricalQ
                | ValueCriticKind::CategoricalReturnQ
        )
}

fn prepare_action_curve(
    r: &ValueRequest,
    input: &Observation,
    action: Option<&Action>,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<PreparedActionCurve>> {
    let (Data::Model(address), Some(Action::Continuous(base))) = (&r.current.data, action) else {
        return Ok(None);
    };
    let Some(&[lo, hi]) = r.slice.action_ranges.first() else {
        return Ok(None);
    };
    if base.is_empty() || !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Ok(None);
    }
    let count = batch_capacity(&r.current)?.min(65);
    if count < 2 {
        return Ok(None);
    }
    let observations = vec![input.clone(); count];
    let actions: Vec<_> = (0..count)
        .map(|index| {
            let mut values = base.clone();
            values[0] = (f64::from(lo)
                + (f64::from(hi) - f64::from(lo)) * index as f64 / (count - 1) as f64)
                as f32;
            Some(Action::Continuous(values))
        })
        .collect();
    let mut points = None;
    with_bundle(&r.current.identity.key, address, &mut |model| {
        if stop(cancelled, accounting) {
            return Ok(());
        }
        let batch = evaluate_model(
            accounting,
            &r.current,
            model,
            &observations,
            &actions,
            Some(vec![None; count]),
        )?;
        let values = scalar_predictions(&batch.predictions, &r.current, &r.output, &batch.actions)?;
        accounting.point_batches.fetch_add(1, Ordering::AcqRel);
        points = Some(
            actions
                .iter()
                .zip(values)
                .map(|(action, value)| {
                    let Some(Action::Continuous(action)) = action else {
                        unreachable!()
                    };
                    [f64::from(action[0]), value]
                })
                .collect(),
        );
        Ok(())
    })?;
    if points
        .as_ref()
        .is_some_and(|points: &Vec<[f64; 2]>| points.iter().any(|point| !point[1].is_finite()))
    {
        return Ok(None);
    }
    Ok(points.map(|points| PreparedActionCurve {
        component: 0,
        points,
    }))
}
fn prepare_grid(
    owner: &Arc<RequestOwner>,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<Arc<PreparedValueGrid>>> {
    let r = &owner.request;
    check_sources(r)?;
    if stop(cancelled, accounting) {
        return Ok(None);
    };
    if r.capability_only {
        // The ordinary source search found no usable current prediction. Only
        // establish independent policy/oracle admission; never publish a critic
        // frame or evaluate a learned prediction to advertise the derived output.
        if decision::compatible(r).is_ok() {
            let _ = r
                .reference_value
                .as_ref()
                .unwrap()
                .prepare_reference(accounting, cancelled);
        }
        if stop(cancelled, accounting) {
            return Ok(None);
        }
        return Err(Error::Config(
            "Current value unavailable for the selected source and input".into(),
        ));
    }
    let n = match &r.current.data {
        Data::Model(_) => r.slice.len()?,
        Data::Recorded(frames) => {
            let n = visitation::input_rows(&frames[0].probes)?;
            ensure(n <= 8192, "recorded Value exceeds row admission")?;
            if r.slice.mode == SliceMode::SamplesOnly {
                0
            } else {
                n.saturating_mul(
                    if r.slice.x == super::slice::ValueCoordinate::DiscreteAction
                        || r.slice.y == Some(super::slice::ValueCoordinate::DiscreteAction)
                    {
                        r.current.heads
                    } else {
                        1
                    },
                )
            }
        }
    };
    let calibration = if r.output == ValueOutput::DecisionLoss {
        None
    } else {
        domain::calibrate(r, accounting, cancelled, with_bundle)
    };
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    let charge = prepared_charge(r, n)?;
    let lease = Lease::acquire(accounting, charge)?;
    let mut grid = PreparedValueGrid {
        color_domain: None,
        reference_color_domain: None,
        difference_color_domain: None,
        calibration,
        distinct_positions: 0,
        population_count: match &r.current.data {
            Data::Recorded(frames) => visitation::input_rows(&frames[0].probes)?,
            _ => 0,
        },
        slice: r.slice.clone(),
        values: Vec::with_capacity(n),
        reference_values: Vec::with_capacity(if r.reference_value.is_some() { n } else { 0 }),
        reference_actions: Vec::with_capacity(if r.reference_value.is_some() { n } else { 0 }),
        positions: Vec::with_capacity(n),
        actions: Vec::with_capacity(n),
        policy_overlay: Vec::with_capacity(n),
        reference_policy_overlay: Vec::with_capacity(if r.reference_policy.is_some() {
            n
        } else {
            0
        }),
        reference_policy_identity: None,
        rows: Vec::with_capacity(if r.current.identity.recorded { n } else { 0 }),
        valid: Vec::with_capacity(n),
        categories: vec![],
        reference_series: None,
        empirical: Default::default(),
        empirical_differences: None,
        empirical_error: None,
        reference_policy_series: None,
        reference_error: None,
        reference_policy_error: None,
        visits: None,
        owner: owner.clone(),
        _lease: lease,
        charge,
    };
    let capacity = batch_capacity(&r.current)?;
    accounting.scratch(POLICY_SCRATCH)?;
    if r.output != ValueOutput::DecisionLoss && decision::compatible(r).is_ok() {
        let _ = r
            .reference_value
            .as_ref()
            .unwrap()
            .prepare_reference(accounting, cancelled);
    }
    if r.output == ValueOutput::DecisionLoss {
        return decision::grid(grid, accounting, cancelled, with_bundle);
    }
    let mut twin_samples = Vec::new();
    match &r.current.data {
        Data::Model(source) => {
            with_bundle(&r.current.identity.key, source, &mut |model| {
                // At most 256 unique states for mixed/action slices; state-only batches
                // contain no repeated states and need no retained actor cache.
                let mut actor_cache = std::collections::BTreeMap::<usize, Option<Action>>::new();
                for start in (0..n).step_by(capacity) {
                    if stop(cancelled, accounting) {
                        return Ok(());
                    };
                    let end = (start + capacity).min(n);
                    let observations = (start..end)
                        .map(|i| r.slice.observation_at(i))
                        .collect::<Result<Vec<_>>>()?;
                    let policy_required = !r.slice.has_discrete_axis()
                        && matches!(r.slice.conditioning, ActionConditioning::Policy)
                        && matches!(
                            r.current.kind,
                            ValueCriticKind::ScalarQ
                                | ValueCriticKind::QuantileQ
                                | ValueCriticKind::MomentQ
                                | ValueCriticKind::CategoricalQ
                                | ValueCriticKind::CategoricalReturnQ
                        );
                    let overlay = if r.slice.state_only() {
                        actor_actions(
                            accounting,
                            &r.current,
                            model,
                            &observations,
                            policy_required,
                        )?
                    } else {
                        let nx = r.slice.axis_len(r.slice.x, 0)?;
                        let mut out = Vec::with_capacity(observations.len());
                        for (i, o) in (start..end).zip(&observations) {
                            let key = if matches!(
                                r.slice.x,
                                super::slice::ValueCoordinate::Observation(_)
                            ) {
                                i % nx
                            } else if r.slice.y.is_some_and(|a| {
                                matches!(a, super::slice::ValueCoordinate::Observation(_))
                            }) {
                                i / nx
                            } else {
                                0
                            };
                            if let std::collections::btree_map::Entry::Vacant(e) =
                                actor_cache.entry(key)
                            {
                                e.insert(
                                    actor_actions(
                                        accounting,
                                        &r.current,
                                        model,
                                        std::slice::from_ref(o),
                                        policy_required,
                                    )?
                                    .pop()
                                    .flatten(),
                                );
                            }
                            out.push(actor_cache[&key].clone());
                        }
                        out
                    };
                    let actions = (start..end)
                        .zip(&overlay)
                        .map(|(i, a)| {
                            if r.current.kind == ValueCriticKind::StateValue {
                                Ok(None)
                            } else {
                                r.slice.action_at(i, a.as_ref())
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let batch = evaluate_model(
                        accounting,
                        &r.current,
                        model,
                        &observations,
                        &actions,
                        Some(overlay),
                    )?;
                    grid.values.extend(scalar_predictions(
                        &batch.predictions,
                        &r.current,
                        &r.output,
                        &batch.actions,
                    )?);
                    for (offset, prediction) in batch.predictions.iter().enumerate() {
                        domain::twin_sample(
                            &mut twin_samples,
                            prediction,
                            &r.output,
                            start + offset,
                            n,
                        )?;
                    }
                    grid.actions.extend(batch.actions);
                    grid.policy_overlay.extend(batch.overlay);
                    for i in start..end {
                        grid.positions.push(r.slice.position(i)?);
                    }
                    accounting.grid_batches.fetch_add(1, Ordering::AcqRel);
                }
                if !r.visits.is_empty() {
                    grid.visits = Some(visitation::prepare_visits(
                        &r.visits,
                        &r.slice,
                        r.current.kind != ValueCriticKind::StateValue,
                        accounting,
                        cancelled,
                        capacity,
                        &mut |observations| {
                            if r.current.kind == ValueCriticKind::DiscreteQ {
                                let batch = evaluate_model(
                                    accounting,
                                    &r.current,
                                    model,
                                    observations,
                                    &vec![None; observations.len()],
                                    None,
                                )?;
                                Ok(batch.actions)
                            } else {
                                actor_actions(accounting, &r.current, model, observations, true)
                            }
                        },
                    )?);
                }
                Ok(())
            })?;
        }
        Data::Recorded(frames) => {
            if r.slice.mode != SliceMode::SamplesOnly {
                let utilities = recorded_utilities(&r.current, &r.output, accounting, cancelled)?;
                let mut distinct = [[0.; 2]; 5];
                for row in 0..visitation::input_rows(&frames[0].probes)? {
                    if stop(cancelled, accounting) {
                        return Ok(None);
                    };
                    let o = crate::policy::prepared::exact_observation(&frames[0].probes, row)?;
                    let action = visitation::critic_action(
                        &frames[0].chunk,
                        row,
                        &r.current.manifest().action_space,
                    )?;
                    let p = recorded_prediction(&r.current, row, &o, action.as_ref())?;
                    let actions = recorded_actions(&p, &r.slice, action)?;
                    for action in actions {
                        if let Some(position) =
                            visitation::position_in_slice(&r.slice, &o, action.as_ref(), None)?
                        {
                            domain::twin_sample(
                                &mut twin_samples,
                                &p,
                                &r.output,
                                grid.values.len(),
                                n,
                            )?;
                            grid.values.push(match &utilities {
                                Some(values) => values[row],
                                None => {
                                    scalar_at(&p, action.as_ref(), &r.output, r.current.utility())?
                                }
                            });
                            if grid.distinct_positions < distinct.len()
                                && !distinct[..grid.distinct_positions].contains(&position)
                            {
                                distinct[grid.distinct_positions] = position;
                                grid.distinct_positions += 1;
                            }
                            grid.positions.push(position);
                            grid.actions.push(action);
                            grid.rows.push(row);
                            grid.policy_overlay.push(None);
                        }
                    }
                }
            }
        }
    }
    if stop(cancelled, accounting) {
        return Ok(None);
    };
    grid.valid
        .extend(grid.values.iter().map(|value| value.is_finite()));
    if let rl::core::action::ActionSpace::Discrete { count } = &r.current.manifest().action_space {
        grid.categories = (0..*count)
            .filter(|&a| r.slice.state.anchor_mask().is_none_or(|m| m[a]))
            .collect();
    }
    if r.slice.mode != SliceMode::SamplesOnly
        && let Some(reference) = &r.reference_value
    {
        match reference_compatibility(r) {
            Err(e) => grid.reference_error = Some(e.to_string()),
            Ok(()) => {
                let result = prepare_reference_grid(
                    reference,
                    r,
                    &mut grid,
                    accounting,
                    cancelled,
                    with_bundle,
                );
                if let Err(e) = result {
                    grid.reference_color_domain = None;
                    grid.reference_values.clear();
                    grid.reference_actions.clear();
                    grid.reference_series = None;
                    grid.reference_error = Some(e.to_string());
                }
            }
        }
    }
    if r.slice.mode != SliceMode::SamplesOnly
        && let Some(reference) = &r.reference_policy
    {
        let result = (|| -> Result<()> {
            reference.check_available()?;
            r.current
                .reference_info
                .compatible(&reference.reference_info)?;
            reference.prepare_reference(accounting, cancelled)?;
            ensure(
                crate::policy::prepared::compatible_encoding(
                    reference.identity().key.root == r.current.identity.key.root,
                    r.current.semantics(),
                    reference.semantics(),
                ),
                "policy reference encoding unavailable",
            )?;
            if let Some(frame) = reference.recorded_frame() {
                return prepare_recorded_policy(frame, r, &mut grid, accounting, cancelled);
            }
            let source = reference
                .model_source()
                .ok_or_else(|| Error::Config("policy reference unavailable".into()))?;
            with_bundle(&reference.identity().key, source, &mut |model| {
                let name = crate::policy::prepared::selected_name(
                    model.configuration(),
                    reference.name(),
                )?;
                for start in (0..grid.values.len()).step_by(capacity) {
                    if stop(cancelled, accounting) {
                        return Ok(());
                    };
                    let observations = (start..(start + capacity).min(grid.values.len()))
                        .map(|i| grid.input(i))
                        .collect::<Result<Vec<_>>>()?;
                    let request = InspectionRequest::new(&name, observations);
                    let started = Instant::now();
                    let result = model.inspect(&request);
                    accounting
                        .inference_micros
                        .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
                    let result = result?;
                    if grid.reference_policy_identity.is_none() {
                        ensure(
                            name.len() <= 1024
                                && result.policy.context.identity.model_role.len() <= 1024,
                            "controller identity exceeds metadata admission",
                        )?;
                        grid.reference_policy_identity = Some(PreparedPolicyIdentity {
                            name: name.clone(),
                            role: result.policy.context.identity.model_role.clone(),
                            stochastic: matches!(&result.policy.record, rl::diagnostics::schema::Record::Policy(p) if p.stochastic),
                        });
                    }
                    grid.reference_policy_overlay
                        .extend(result.actions.into_iter().map(Some));
                    accounting.grid_batches.fetch_add(1, Ordering::AcqRel);
                }
                Ok(())
            })
        })();
        if let Err(e) = result {
            grid.reference_policy_overlay.clear();
            grid.reference_policy_identity = None;
            grid.reference_policy_series = None;
            grid.reference_policy_error = Some(e.to_string());
        }
    }
    if r.slice.mode != SliceMode::SamplesOnly && !r.visits.is_empty() && grid.visits.is_none() {
        grid.visits = Some(visitation::prepare_visits(
            &r.visits,
            &r.slice,
            r.current.kind != ValueCriticKind::StateValue,
            accounting,
            cancelled,
            capacity,
            &mut |observations| Ok(vec![None; observations.len()]),
        )?);
    }
    if stop(cancelled, accounting) {
        return Ok(None);
    };
    empirical::prepare_grid(r, &mut grid, accounting, cancelled, with_bundle)?;
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    domain::finish(&mut grid, twin_samples);
    Ok(Some(Arc::new(grid)))
}
fn prepare_recorded_policy(
    frame: &crate::resolver::ResolvedRecord,
    r: &ValueRequest,
    grid: &mut PreparedValueGrid,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
) -> Result<()> {
    use rl::diagnostics::schema::Record;
    let Record::Policy(policy) = &frame.chunk.record else {
        return Err(Error::Config("not recorded policy".into()));
    };
    let batch = crate::policy_change::admission::checked_batch(
        &frame.chunk,
        &frame.probes,
        &r.current.manifest().action_space,
        POLICY_SCRATCH,
    )?;
    let n = batch.rows();
    let source = r.reference_policy.as_ref().unwrap();
    if source.finite_reference() {
        source.prepare_reference(accounting, cancelled)?;
        grid.reference_policy_identity = Some(PreparedPolicyIdentity {
            name: source.identity().name.clone(),
            role: frame.chunk.context.identity.model_role.clone(),
            stochastic: policy.stochastic,
        });
        for cell in 0..grid.values.len() {
            if stop(cancelled, accounting) {
                return Ok(());
            }
            let input = grid.input(cell)?;
            let action = source
                .reference_info
                .lookup(&frame.probes, &input)?
                .map(|row| {
                    visitation::action_row(
                        &frame.chunk,
                        &policy.actions,
                        row,
                        &r.current.manifest().action_space,
                    )
                })
                .transpose()?;
            grid.reference_policy_overlay.push(action);
        }
        return Ok(());
    }
    ensure(
        n <= 8192
            && n.saturating_mul(grid.values.len())
                .saturating_mul(r.current.manifest().observation_dim)
                <= 1024 * 1024,
        "sparse policy matching admission",
    )?;
    let mut series = PreparedPolicySeries {
        positions: Vec::with_capacity(n),
        actions: Vec::with_capacity(n),
        rows: Vec::with_capacity(n),
        stochastic: policy.stochastic,
    };
    grid.reference_policy_identity = Some(PreparedPolicyIdentity {
        name: r.reference_policy.as_ref().unwrap().identity().name.clone(),
        role: frame.chunk.context.identity.model_role.clone(),
        stochastic: policy.stochastic,
    });
    for row in 0..n {
        if stop(cancelled, accounting) {
            return Ok(());
        };
        let o = crate::policy::prepared::recorded_input(&batch, row)?;
        let action = visitation::action_row(
            &frame.chunk,
            &policy.actions,
            row,
            &r.current.manifest().action_space,
        )?;
        if let Some(position) = visitation::position_in_slice(&r.slice, &o, Some(&action), None)? {
            series.positions.push(position);
            series.actions.push(action);
            series.rows.push(row);
        }
    }
    for cell in 0..grid.values.len() {
        if stop(cancelled, accounting) {
            return Ok(());
        };
        let input = grid.input(cell)?;
        let mut action = None;
        for row in 0..n {
            if crate::policy::prepared::recorded_input(&batch, row)? == input {
                action = Some(visitation::action_row(
                    &frame.chunk,
                    &policy.actions,
                    row,
                    &r.current.manifest().action_space,
                )?);
                break;
            }
        }
        grid.reference_policy_overlay.push(action);
    }
    grid.reference_policy_series = Some(series);
    Ok(())
}
fn prepare_reference_grid(
    source: &ValueSource,
    r: &ValueRequest,
    grid: &mut PreparedValueGrid,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<()> {
    let capacity = batch_capacity(source)?;
    let output = reference_output(r);
    match &source.data {
        Data::Model(address) => with_bundle(&source.identity.key, address, &mut |model| {
            for start in (0..grid.values.len()).step_by(capacity) {
                if stop(cancelled, accounting) {
                    return Ok(());
                };
                let end = (start + capacity).min(grid.values.len());
                let observations = (start..end)
                    .map(|i| grid.input(i))
                    .collect::<Result<Vec<_>>>()?;
                let actions = if r.condition == ComparisonCondition::SameInput {
                    grid.actions[start..end].to_vec()
                } else {
                    vec![None; end - start]
                };
                let overlay = if r.condition == ComparisonCondition::SameInput {
                    Some(vec![None; observations.len()])
                } else {
                    None
                };
                let batch =
                    evaluate_model(accounting, source, model, &observations, &actions, overlay)?;
                grid.reference_values.extend(
                    scalar_predictions(&batch.predictions, source, &output, &batch.actions)?
                        .into_iter()
                        .map(|value| value.is_finite().then_some(value)),
                );
                grid.reference_actions.extend(batch.actions);
                accounting.grid_batches.fetch_add(1, Ordering::AcqRel);
            }
            Ok(())
        }),
        Data::Recorded(frames) => {
            if source.finite_reference() {
                source.prepare_reference(accounting, cancelled)?;
                for cell in 0..grid.values.len() {
                    if stop(cancelled, accounting) {
                        return Ok(());
                    }
                    let input = grid.input(cell)?;
                    let action = if r.condition == ComparisonCondition::SameInput {
                        grid.actions[cell].clone()
                    } else {
                        None
                    };
                    let evaluated = source
                        .reference_info
                        .lookup(&frames[0].probes, &input)?
                        .map(|row| {
                            let p = recorded_prediction(source, row, &input, action.as_ref())?;
                            let action = discrete_first_action(&p, action);
                            Ok::<_, Error>((
                                scalar_at(&p, action.as_ref(), &output, source.utility())?,
                                action,
                            ))
                        })
                        .transpose()?;
                    grid.reference_values.push(
                        evaluated
                            .as_ref()
                            .and_then(|(v, _)| v.is_finite().then_some(*v)),
                    );
                    grid.reference_actions.push(evaluated.and_then(|(_, a)| a));
                }
                return Ok(());
            }
            let rows = recorded_rows(source)?;
            let utilities = recorded_utilities(source, &output, accounting, cancelled)?;
            ensure(
                grid.values
                    .len()
                    .saturating_mul(rows)
                    .saturating_mul(source.manifest().observation_dim)
                    <= 1024 * 1024,
                "sparse reference match budget exceeded",
            )?;
            let capacity = rows.saturating_mul(
                if source.kind == ValueCriticKind::DiscreteQ && discrete_axis(&r.slice) {
                    source.heads
                } else {
                    1
                },
            );
            let mut series = PreparedValueSeries {
                values: Vec::with_capacity(capacity),
                positions: Vec::with_capacity(capacity),
                rows: Vec::with_capacity(capacity),
                actions: Vec::with_capacity(capacity),
            };
            for row in 0..rows {
                if stop(cancelled, accounting) {
                    return Ok(());
                };
                let o = crate::policy::prepared::exact_observation(&frames[0].probes, row)?;
                let action = visitation::critic_action(
                    &frames[0].chunk,
                    row,
                    &source.manifest().action_space,
                )?;
                let p = recorded_prediction(source, row, &o, action.as_ref())?;
                for action in recorded_actions(&p, &r.slice, action)? {
                    if let Some(position) =
                        visitation::position_in_slice(&r.slice, &o, action.as_ref(), None)?
                    {
                        let value = match &utilities {
                            Some(values) => values[row],
                            None => scalar_at(&p, action.as_ref(), &output, source.utility())?,
                        };
                        if !value.is_finite() {
                            continue;
                        }
                        series.values.push(value);
                        series.positions.push(position);
                        series.rows.push(row);
                        series.actions.push(action);
                    }
                }
            }
            grid.reference_series = Some(series);
            for i in 0..grid.values.len() {
                if stop(cancelled, accounting) {
                    return Ok(());
                };
                let o = grid.input(i)?;
                let action = if r.condition == ComparisonCondition::SameInput {
                    grid.actions[i].clone()
                } else {
                    None
                };
                let row = matching_recorded_row(source, &o, action.as_ref())?;
                let evaluated = row
                    .map(|row| {
                        let p = recorded_prediction(source, row, &o, action.as_ref())?;
                        let action = discrete_first_action(&p, action);
                        let value = match &utilities {
                            Some(values) => values[row],
                            None => scalar_at(&p, action.as_ref(), &output, source.utility())?,
                        };
                        Ok::<_, Error>((value, action))
                    })
                    .transpose()?;
                grid.reference_values.push(
                    evaluated
                        .as_ref()
                        .and_then(|(v, _)| v.is_finite().then_some(*v)),
                );
                grid.reference_actions.push(evaluated.and_then(|(_, a)| a));
            }
            Ok(())
        }
    }
}
fn recorded_rows(source: &ValueSource) -> Result<usize> {
    let Data::Recorded(f) = &source.data else {
        return Err(Error::State("not recorded".into()));
    };
    visitation::input_rows(&f[0].probes)
}
fn discrete_axis(slice: &ValueSlice) -> bool {
    slice.x == super::slice::ValueCoordinate::DiscreteAction
        || slice.y == Some(super::slice::ValueCoordinate::DiscreteAction)
}
fn recorded_actions(
    prediction: &ValuePrediction,
    slice: &ValueSlice,
    recorded: Option<Action>,
) -> Result<Vec<Option<Action>>> {
    let ValuePrediction::Discrete { values, legal } = prediction else {
        return Ok(vec![recorded]);
    };
    if discrete_axis(slice) {
        return Ok(legal
            .iter()
            .enumerate()
            .filter(|(_, valid)| **valid)
            .map(|(a, _)| Some(Action::Discrete(a)))
            .collect());
    }
    let action = match &slice.conditioning {
        ActionConditioning::Fixed(Action::Discrete(a)) => *a,
        ActionConditioning::Fixed(_) => {
            return Err(Error::Action("continuous action for DDQN".into()));
        }
        ActionConditioning::Policy => values
            .iter()
            .enumerate()
            .filter(|(a, _)| legal[*a])
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(a, _)| a)
            .ok_or_else(|| Error::Action("no legal action".into()))?,
    };
    prediction.select_discrete(action)?;
    Ok(vec![Some(Action::Discrete(action))])
}
fn recorded_prediction(
    source: &ValueSource,
    row: usize,
    o: &Observation,
    action: Option<&Action>,
) -> Result<ValuePrediction> {
    let Data::Recorded(frames) = &source.data else {
        return Err(Error::State("not recorded".into()));
    };
    if !matches!(
        source.kind,
        ValueCriticKind::StateValue | ValueCriticKind::DiscreteQ
    ) {
        let recorded_action =
            visitation::critic_action(&frames[0].chunk, row, &source.manifest().action_space)?;
        ensure(
            action.is_some() && recorded_action.as_ref() == action,
            "recorded action-conditioned Q requires its exact first action",
        )?;
    }
    let legal = if source.kind == ValueCriticKind::DiscreteQ {
        Some(
            o.dense_mask()
                .map(<[bool]>::to_vec)
                .unwrap_or_else(|| vec![true; source.heads]),
        )
    } else {
        None
    };
    let mut p = ValuePrediction::from_critic(&frames[0].chunk, row, legal.as_deref())?;
    if frames.len() == 2 {
        ensure(
            visitation::critic_action(&frames[0].chunk, row, &source.manifest().action_space)?
                == visitation::critic_action(
                    &frames[1].chunk,
                    row,
                    &source.manifest().action_space,
                )?,
            "recorded twin actions differ",
        )?;
        p = ValuePrediction::Twin {
            q1: Box::new(p),
            q2: Box::new(ValuePrediction::from_critic(&frames[1].chunk, row, None)?),
        };
    }
    if matches!(p, ValuePrediction::Discrete { .. })
        && let Some(Action::Discrete(a)) = action
    {
        p.select_discrete(*a)?;
    }
    Ok(p)
}
fn matching_recorded(
    source: &ValueSource,
    o: &Observation,
    action: Option<&Action>,
) -> Result<Option<ValuePrediction>> {
    matching_recorded_row(source, o, action)?
        .map(|row| recorded_prediction(source, row, o, action))
        .transpose()
}
fn matching_recorded_row(
    source: &ValueSource,
    o: &Observation,
    action: Option<&Action>,
) -> Result<Option<usize>> {
    let Data::Recorded(f) = &source.data else {
        return Err(Error::State("not recorded".into()));
    };
    if source.finite_reference() {
        return source.reference_info.lookup(&f[0].probes, o);
    }
    let rows = recorded_rows(source)?;
    ensure(
        rows <= 8192 && rows.saturating_mul(source.manifest().observation_dim) <= 1024 * 1024,
        "recorded reference row admission",
    )?;
    ensure(
        !matches!(&f[0].probes.record,rl::diagnostics::schema::Record::ProbeSet(p)if !p.histories.is_empty()),
        "history comparison needs the same immutable source-row pin",
    )?;
    for row in 0..rows {
        let actual = crate::policy::prepared::exact_observation(&f[0].probes, row)?;
        if &actual != o {
            continue;
        };
        let recorded_action =
            visitation::critic_action(&f[0].chunk, row, &source.manifest().action_space)?;
        if source.kind != ValueCriticKind::StateValue
            && source.kind != ValueCriticKind::DiscreteQ
            && recorded_action.as_ref() != action
        {
            continue;
        };
        return Ok(Some(row));
    }
    Ok(None)
}
// Retain only one scalar per sparse row; full predictions remain batch/point scoped.
fn recorded_utilities(
    source: &ValueSource,
    output: &ValueOutput,
    accounting: &Accounting,
    cancelled: &impl Fn() -> bool,
) -> Result<Option<Vec<f64>>> {
    if source.kind != ValueCriticKind::MomentQ || *output != ValueOutput::Utility {
        return Ok(None);
    }
    let Data::Recorded(frames) = &source.data else {
        return Ok(None);
    };
    let utility = source
        .utility()
        .ok_or_else(|| Error::Config("unknown moment utility".into()))?;
    let rows = recorded_rows(source)?;
    ensure(rows <= 8192, "recorded moment row admission")?;
    let mut values = Vec::with_capacity(rows);
    for start in (0..rows).step_by(batch_capacity(source)?) {
        if stop(cancelled, accounting) {
            return Err(Error::State("Value request superseded".into()));
        }
        let end = (start + batch_capacity(source)?).min(rows);
        let mut means = Vec::with_capacity(end - start);
        let mut seconds = Vec::with_capacity(end - start);
        for row in start..end {
            let ValuePrediction::Moments { mean, raw_second } =
                ValuePrediction::from_critic(&frames[0].chunk, row, None)?
            else {
                return Err(Error::Config(
                    "recorded moment representation mismatch".into(),
                ));
            };
            means.push(mean);
            seconds.push(raw_second);
        }
        values.extend(moment_utilities(&means, &seconds, utility)?);
    }
    Ok(Some(values))
}
fn prepare_point(
    owner: &Arc<RequestOwner>,
    grid: &Arc<PreparedValueGrid>,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<Arc<PreparedValuePoint>>> {
    let r = &owner.request;
    check_sources(r)?;
    if stop(cancelled, accounting) {
        return Ok(None);
    };
    let bytes = point_charge_for_grid(r, grid)?;
    ensure(
        grid.charge
            + grid.owner.request.metadata_bytes()
            + (if Arc::ptr_eq(owner, &grid.owner) {
                0
            } else {
                owner.additional_bytes(&grid.owner)
            })
            + bytes
            <= POLICY_FRAME,
        "selected Value frame exceeds 2 MiB",
    )?;
    let lease = Lease::acquire(accounting, bytes)?;
    accounting.scratch(POLICY_SCRATCH)?;
    let final_observation = match &r.pin {
        ValuePin::Recorded { source, row }
            if matches!(
                &source.chunk.record,
                rl::diagnostics::schema::Record::Trajectory(_)
            ) =>
        {
            *row == visitation::input_rows(&source.chunk)?
        }
        _ => false,
    };
    let (input, action) = match &r.pin {
        ValuePin::State(o) => (o.clone(), None),
        ValuePin::Input {
            observation,
            action,
        } => (observation.clone(), action.clone()),
        ValuePin::Recorded { source, row } => {
            if final_observation {
                ensure(
                    r.current.kind == ValueCriticKind::StateValue,
                    "final observation has no recorded action for an action-value critic",
                )?;
                (
                    crate::policy::prepared::exact_observation(&source.chunk, *row)?,
                    None,
                )
            } else {
                visitation::exact_input(source, *row, &r.current.manifest().action_space)?
            }
        }
        ValuePin::Probe {
            source,
            row,
            action,
        } => {
            if matches!(r.current.data, Data::Model(_)) {
                ensure(
                    !matches!(&source.chunk.record,rl::diagnostics::schema::Record::ProbeSet(p)if !p.histories.is_empty()),
                    "history-bound pin has no model inference adapter",
                )?;
            }
            (
                crate::policy::prepared::exact_observation(&source.chunk, *row)?,
                action.clone(),
            )
        }
    };
    input.validate(
        r.current.manifest().observation_dim,
        &r.current.manifest().action_space,
        true,
    )?;
    if r.output == ValueOutput::DecisionLoss {
        return decision::point(owner, input, lease, accounting, cancelled, with_bundle);
    }
    let action = if action.is_none()
        && r.current.return_target.is_some_and(ReturnTarget::is_soft)
        && let ActionConditioning::Fixed(first) = &r.slice.conditioning
    {
        Some(first.clone())
    } else {
        action
    };
    let (prediction, action) =
        if let (ValuePin::Probe { source, row, .. }, Data::Recorded(frames)) =
            (&r.pin, &r.current.data)
        {
            ensure(
                source.key == frames[0].probe_key,
                "recorded pin probe identity differs",
            )?;
            (
                recorded_prediction(&r.current, *row, &input, action.as_ref())?,
                action,
            )
        } else {
            evaluate_point(
                &r.current,
                &input,
                action,
                &r.output,
                accounting,
                with_bundle,
            )?
        };
    let action = if action.is_none() {
        if let ValuePrediction::Discrete { values, legal } = &prediction {
            values
                .iter()
                .enumerate()
                .filter(|(i, _)| legal[*i])
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| Action::Discrete(i))
        } else {
            None
        }
    } else {
        action
    };
    let (value, profile) = super::profile::selected_profile(
        &prediction,
        action.as_ref(),
        &r.output,
        r.current.utility(),
    )?;
    let mut point = PreparedValuePoint {
        input,
        action,
        action_curve: None,
        prediction,
        decision_loss: None,
        value,
        reference_prediction: None,
        profile,
        reference_profile: None,
        reference_value: None,
        reference_action: None,
        reference_error: None,
        empirical: None,
        empirical_error: None,
        observed_return: None,
        final_observation,
        recorded_revision: None,
        recorded_key: None,
        recorded_row: None,
        policy_conditioned: r.controller_state_pin(),
        neighborhood_count: 0,
        _owner: owner.clone(),
        _lease: lease,
    };
    if action_curve_eligible(r) {
        point.action_curve = prepare_action_curve(
            r,
            &point.input,
            point.action.as_ref(),
            accounting,
            cancelled,
            with_bundle,
        )
        .ok()
        .flatten();
    }
    if r.reference_value.is_some() {
        let result = (|| -> Result<()> {
            reference_compatibility(r)?;
            let source = r.reference_value.as_ref().unwrap();
            source.prepare_reference(accounting, cancelled)?;
            let action = if r.condition == ComparisonCondition::SameInput {
                point.action.clone()
            } else {
                None
            };
            let (p, a) = evaluate_point(
                source,
                &point.input,
                action,
                &reference_output(r),
                accounting,
                with_bundle,
            )?;
            let (reference_value, reference_profile) = super::profile::selected_profile(
                &p,
                a.as_ref(),
                &reference_output(r),
                source.utility(),
            )?;
            point.reference_value = reference_value.is_finite().then_some(reference_value);
            let distribution_compatible = r.condition == ComparisonCondition::SameInput
                && r.current
                    .declaration()
                    .zip(source.declaration())
                    .is_some_and(|(c, d)| {
                        compatible_values(
                            c,
                            &r.current.critic_names()[0],
                            d,
                            &source.critic_names()[0],
                            &ValueOutput::Quantile { probability: 0.5 },
                            r.condition,
                        )
                        .is_ok()
                    });
            point.reference_profile = match reference_profile {
                Some(profile) if profile.cdf().is_some() && !distribution_compatible => None,
                profile => profile,
            };
            point.reference_prediction = Some(p);
            point.reference_action = a;
            Ok(())
        })();
        if let Err(e) = result {
            point.reference_error = Some(e.to_string());
        }
    }
    if let ValuePin::Recorded { source, row } = &r.pin {
        point.recorded_key = Some(source.key.clone());
        point.recorded_row = Some(*row);
        point.recorded_revision = Some(source.chunk.context.identity.learner_update_revision);
        if !final_observation {
            point.observed_return = visitation::observed_return(
                source,
                *row,
                &point.input,
                point.action.as_ref(),
                r.current.kind != ValueCriticKind::StateValue,
                &r.current.manifest().action_space,
            )?;
        }
    }
    if let ValuePin::Probe { source, row, .. } = &r.pin {
        point.recorded_key = Some(source.key.clone());
        point.recorded_row = Some(*row);
        point.recorded_revision = Some(source.chunk.context.identity.learner_update_revision);
    }
    empirical::prepare_point(r, grid, &mut point)?;
    if let Some(visits) = &grid.visits {
        point.neighborhood_count =
            visits.neighborhood_count(&r.slice, &point.input, point.action.as_ref())?;
    }
    if stop(cancelled, accounting) {
        return Ok(None);
    };
    Ok(Some(Arc::new(point)))
}
fn evaluate_point(
    source: &ValueSource,
    input: &Observation,
    action: Option<Action>,
    _output: &ValueOutput,
    accounting: &Accounting,
    with_bundle: &mut WithBundle<'_>,
) -> Result<(ValuePrediction, Option<Action>)> {
    match &source.data {
        Data::Model(address) => {
            let mut result = None;
            with_bundle(&source.identity.key, address, &mut |model| {
                let overlay = if action.is_some()
                    || matches!(
                        source.kind,
                        ValueCriticKind::StateValue | ValueCriticKind::DiscreteQ
                    ) {
                    Some(vec![None])
                } else {
                    None
                };
                let mut batch = evaluate_model(
                    accounting,
                    source,
                    model,
                    std::slice::from_ref(input),
                    std::slice::from_ref(&action),
                    overlay,
                )?;
                result = Some((batch.predictions.remove(0), batch.actions.remove(0)));
                accounting.point_batches.fetch_add(1, Ordering::AcqRel);
                Ok(())
            })?;
            result.ok_or_else(|| Error::State("no point prediction".into()))
        }
        Data::Recorded(_) => matching_recorded(source, input, action.as_ref())?
            .map(|p| {
                let action = discrete_first_action(&p, action);
                (p, action)
            })
            .ok_or_else(|| Error::State("no exact recorded critic input/action".into())),
    }
}
fn discrete_first_action(prediction: &ValuePrediction, action: Option<Action>) -> Option<Action> {
    action.or_else(|| {
        let ValuePrediction::Discrete { values, legal } = prediction else {
            return None;
        };
        values
            .iter()
            .enumerate()
            .filter(|(i, _)| legal[*i])
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| Action::Discrete(i))
    })
}
