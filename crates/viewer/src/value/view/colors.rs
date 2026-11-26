use super::*;
use crate::{
    data::{Loader, RunIndex},
    resolver::ProbeResolver,
};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ColorMode {
    #[default]
    FinalSnapshot,
    Adaptive,
    Manual,
}

pub(crate) struct ScaleInput<'a> {
    pub current: &'a ValueSource,
    pub slice: &'a ValueSlice,
    pub output: &'a ValueOutput,
    pub reference: Option<&'a ValueSource>,
    pub condition: ComparisonCondition,
}
struct Key {
    slice: ValueSlice,
    output: ValueOutput,
    reference: Option<ValueSource>,
    condition: ComparisonCondition,
}

#[derive(Default)]
pub(crate) struct Colors {
    pub mode: ColorMode,
    pub manual: ColorRanges,
    pub source: Option<ValueSource>,
    pub target: Option<u64>,
    pub message: Option<String>,
    family: Option<ValueSource>,
    search: Option<ModelSearch>,
    key: Option<Key>,
    serial: u64,
    queued: bool,
    finished: bool,
    unavailable: bool,
    ranges: ColorRanges,
    provisional: ColorRanges,
}
impl Colors {
    /// Retain a completed calibration identity when an otherwise identical
    /// prepared view is resubmitted. A refresh or mode change must not reuse it.
    pub(crate) fn completed_calibration(
        &self,
        previous: &ValueRequest,
    ) -> Option<CalibrationRequest> {
        if self.mode != ColorMode::FinalSnapshot || !self.finished {
            return None;
        }
        previous
            .calibration
            .as_ref()
            .filter(|calibration| {
                calibration.serial == self.serial
                    && self
                        .source
                        .as_ref()
                        .is_some_and(|source| source.same(&calibration.source))
            })
            .cloned()
    }
    pub fn owned_bytes(&self, charges: &mut crate::value::source::ViewSourceCharges) -> usize {
        self.source.as_ref().map_or(0, |s| charges.charge(s))
            + self.family.as_ref().map_or(0, |s| charges.charge(s))
            + self
                .search
                .as_ref()
                .map_or(0, |s| s.view_owned_bytes_with(charges))
            + self.key.as_ref().map_or(0, |k| {
                k.slice.owned_bytes() + k.reference.as_ref().map_or(0, |s| charges.charge(s))
            })
            + self.message.as_ref().map_or(0, String::capacity)
    }
    pub fn refresh(&mut self, target: Option<u64>) {
        self.target = target;
        self.source = None;
        self.search = None;
        self.unavailable = false;
        self.invalidate();
    }
    fn invalidate(&mut self) {
        self.serial = self.serial.wrapping_add(1);
        self.queued = false;
        self.finished = false;
        self.ranges = ColorRanges::default();
        self.provisional = ColorRanges::default();
        self.message = Some("Waiting for slice calibration".into());
    }
    pub fn prepare(
        &mut self,
        input: ScaleInput<'_>,
        run: &RunIndex,
        loader: &mut Loader,
        resolver: &mut ProbeResolver,
    ) -> rl::Result<Option<CalibrationRequest>> {
        if self.family.as_ref().is_none_or(|s| {
            s.identity().key.root != input.current.identity().key.root
                || s.model_role() != input.current.model_role()
                || s.critic_names() != input.current.critic_names()
                || s.kind() != input.current.kind()
        }) {
            self.family = Some(input.current.clone());
            self.key = None;
            self.refresh(None);
        }
        let changed = self.key.as_ref().is_none_or(|k| {
            !same_slice(&k.slice, input.slice)
                || k.output != *input.output
                || k.condition != input.condition
                || !same_source(k.reference.as_ref(), input.reference)
        });
        if changed {
            self.key = Some(Key {
                slice: input.slice.clone(),
                output: input.output.clone(),
                reference: input.reference.cloned(),
                condition: input.condition,
            });
            self.invalidate();
        }
        if self.mode != ColorMode::FinalSnapshot || self.unavailable || self.finished {
            return Ok(None);
        }
        if self.source.is_none() {
            if self.search.is_none() {
                // Freeze this ceiling; appended checkpoints require an explicit refresh.
                let revision = self
                    .target
                    .unwrap_or_else(|| run.revisions().last().copied().unwrap_or(0));
                self.search = Some(ModelSearch::new(
                    input.current.model_role(),
                    revision,
                    input.current.critic_names().first().map(String::as_str),
                )?);
            }
            let search = self.search.as_mut().unwrap();
            let candidate =
                search.advance_value_for_slice(run, loader, resolver, Some(input.slice))?;
            if let Some(mut candidate) = candidate {
                if candidate.identity().recorded {
                    self.unavailable = true;
                    self.message = Some(
                        "Compatible model weights unavailable; provisional slice scale".into(),
                    );
                    return Ok(None);
                }
                if input.current.critic_names().len() == 2 {
                    let group = input
                        .current
                        .declaration()
                        .and_then(|d| d.critics.get(&input.current.critic_names()[0]))
                        .and_then(|c| c.twin.as_ref())
                        .map(|t| t.group.as_str());
                    match group.map(|g| candidate.clone().with_twin(g)) {
                        Some(Ok(twin)) => candidate = twin,
                        _ => {
                            search.reject_value_candidate("Calibration twin unavailable");
                            return Ok(None);
                        }
                    }
                }
                if candidate.kind() != input.current.kind()
                    || candidate.critic_names() != input.current.critic_names()
                    || (*input.output == ValueOutput::Utility
                        && !matches!(
                            candidate.utility(),
                            Some(
                                MomentUtility::ExpectedReturn | MomentUtility::MeanVariance { .. }
                            )
                        ))
                    || (matches!(input.slice.conditioning, ActionConditioning::Policy)
                        && matches!(
                            candidate.kind(),
                            ValueCriticKind::ScalarQ
                                | ValueCriticKind::QuantileQ
                                | ValueCriticKind::MomentQ
                                | ValueCriticKind::CategoricalQ
                                | ValueCriticKind::CategoricalReturnQ
                        )
                        && (candidate.actor_name().is_none() || candidate.actor_stochastic()))
                {
                    search.reject_value_candidate("Incompatible calibration critic");
                    return Ok(None);
                }
                self.source = Some(candidate);
                self.search = None;
            } else if !search.pending() {
                self.unavailable = true;
                self.message = Some("Calibration unavailable; provisional slice scale".into());
            }
        }
        Ok(self.source.as_ref().map(|source| CalibrationRequest {
            serial: self.serial,
            source: source.clone(),
            reference: input.reference.cloned(),
        }))
    }
    pub fn submitted(&mut self, request: &ValueRequest) {
        if request
            .calibration
            .as_ref()
            .is_some_and(|c| c.serial == self.serial)
        {
            self.queued = true;
        }
    }
    pub fn unsubmitted(&self) -> bool {
        !self.queued
    }
    fn matches(&self, grid: &PreparedValueGrid) -> bool {
        self.family.as_ref().is_some_and(|s| {
            s.identity().key.root == grid.current().identity().key.root
                && s.model_role() == grid.current().model_role()
                && s.critic_names() == grid.current().critic_names()
        }) && self.key.as_ref().is_some_and(|k| {
            same_slice(&k.slice, &grid.slice)
                && k.output == *grid.output()
                && k.condition == grid.comparison_condition()
                && same_source(k.reference.as_ref(), grid.reference())
        })
    }
    pub fn accept(&mut self, grid: &PreparedValueGrid) {
        if let Some(result) = &grid.calibration
            && result.serial == self.serial
            && self
                .source
                .as_ref()
                .is_some_and(|s| s.identity().key == result.source)
            && self.matches(grid)
        {
            self.ranges = result.ranges;
            self.message.clone_from(&result.error);
            self.finished = true;
        }
    }
    pub fn provisional(&self, difference: bool) -> bool {
        self.mode == ColorMode::FinalSnapshot
            && (if difference {
                self.ranges.difference
            } else {
                self.ranges.value
            })
            .is_none()
    }
    pub fn scale(&mut self, grid: &PreparedValueGrid, difference: bool) -> [f64; 2] {
        self.accept(grid);
        let adaptive = ColorRanges {
            value: grid.color_domain,
            difference: grid.difference_color_domain,
        };
        if self.matches(grid) && self.provisional.value.is_none() {
            self.provisional.value = adaptive.value;
        }
        if self.matches(grid) && self.provisional.difference.is_none() {
            self.provisional.difference = adaptive.difference;
        }
        let ranges = match self.mode {
            ColorMode::Adaptive => adaptive,
            ColorMode::Manual => self.manual,
            ColorMode::FinalSnapshot => self.ranges,
        };
        let get = |ranges: ColorRanges| {
            if difference {
                ranges.difference
            } else {
                ranges.value
            }
        };
        get(ranges)
            .or_else(|| get(self.provisional))
            .or_else(|| get(adaptive))
            .unwrap_or(if difference { [-1., 1.] } else { [0., 1.] })
    }
}

fn same_source(a: Option<&ValueSource>, b: Option<&ValueSource>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.same(b),
        (None, None) => true,
        _ => false,
    }
}
// Only plotted ranges and actual fixed inputs define a slice. Resolution,
// hidden ranges and the pinned point must not change a frozen calibration.
fn same_slice(a: &ValueSlice, b: &ValueSlice) -> bool {
    if a.mode != b.mode
        || a.x != b.x
        || a.y != b.y
        || a.state.anchor_mask() != b.state.anchor_mask()
        || a.state.grid_eligibility() != b.state.grid_eligibility()
        || a.state.tolerance != b.state.tolerance
        || a.state.fixed.len() != b.state.fixed.len()
    {
        return false;
    }
    let axes: Vec<_> = std::iter::once(a.x).chain(a.y).collect();
    if a.state
        .fixed
        .iter()
        .zip(&b.state.fixed)
        .enumerate()
        .any(|(i, (a, b))| !axes.contains(&ValueCoordinate::Observation(i)) && a != b)
    {
        return false;
    }
    for axis in &axes {
        let equal = match *axis {
            ValueCoordinate::Observation(i) => a.state.ranges[i] == b.state.ranges[i],
            ValueCoordinate::Action(i) => a.action_ranges[i] == b.action_ranges[i],
            ValueCoordinate::DiscreteAction => true,
        };
        if !equal {
            return false;
        }
    }
    match (&a.conditioning, &b.conditioning) {
        (
            ActionConditioning::Fixed(Action::Continuous(a)),
            ActionConditioning::Fixed(Action::Continuous(b)),
        ) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .enumerate()
                    .all(|(i, (a, b))| axes.contains(&ValueCoordinate::Action(i)) || a == b)
        }
        _ => a.conditioning == b.conditioning,
    }
}
