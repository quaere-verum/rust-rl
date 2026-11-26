//! Immutable visit identities and complete observed suffixes.
use super::{
    ensure,
    prepared::ValuePin,
    slice::{ActionConditioning, ValueCoordinate, ValueSlice},
};
use crate::{
    data::SelectedInput,
    pane::SelectionIntent,
    plot_data::array,
    policy::prepared::{Accounting, VisitSource, exact_observation},
};
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{schema::*, value::trajectory_return_to_go},
};
use std::sync::atomic::Ordering;

#[derive(Clone)]
pub struct ValueVisit {
    pub source: VisitSource,
    pub row: usize,
    pub position: [f32; 2],
}
#[derive(Clone)]
pub struct PreparedValueVisits {
    pub markers: Vec<ValueVisit>,
    pub filtered_count: usize,
    pub counts: Vec<u32>,
}
pub fn input_rows(chunk: &Chunk) -> Result<usize> {
    match &chunk.record {
        Record::ProbeSet(p) => Ok(array(chunk, &p.observations)?.shape()[0]),
        Record::Trajectory(t) => {
            let n = array(chunk, &t.actions)?.shape()[0];
            ensure(
                array(chunk, &t.observations)?.shape()[0] == n + 1,
                "trajectory needs T decisions and T+1 states",
            )?;
            Ok(n)
        }
        _ => Err(Error::Config("not genuine inputs".into())),
    }
}
pub(crate) fn action_row(
    chunk: &Chunk,
    key: &str,
    row: usize,
    space: &ActionSpace,
) -> Result<Action> {
    let a = array(chunk, key)?;
    ensure(row < a.shape()[0], "action row outside source")?;
    let action = match space {
        ActionSpace::Discrete { .. } => {
            let v = match a.dtype() {
                DType::U64 => a.as_u64()?[row],
                _ => return Err(Error::Shape("discrete actions must be u64".into())),
            };
            Action::Discrete(
                usize::try_from(v).map_err(|_| Error::Shape("discrete action overflow".into()))?,
            )
        }
        _ => {
            ensure(
                a.shape() == [a.shape()[0], space.dimensions()],
                "recorded action width mismatch",
            )?;
            let width = space.dimensions();
            let range = row * width..(row + 1) * width;
            let values = match a.dtype() {
                DType::F32 => a.as_f32()?[range].to_vec(),
                DType::F64 => a.as_f64()?[range]
                    .iter()
                    .map(|&v| {
                        let f = v as f32;
                        ensure(
                            f.is_finite() && f64::from(f) == v,
                            "action is not exactly representable as f32",
                        )?;
                        Ok(f)
                    })
                    .collect::<Result<Vec<_>>>()?,
                _ => return Err(Error::Shape("continuous action type".into())),
            };
            Action::Continuous(values)
        }
    };
    space.check(&action, None)?;
    Ok(action)
}
pub(crate) fn critic_action(
    chunk: &Chunk,
    row: usize,
    space: &ActionSpace,
) -> Result<Option<Action>> {
    let Record::Critic(c) = &chunk.record else {
        return Err(Error::Config("not critic".into()));
    };
    c.actions
        .as_ref()
        .map(|key| action_row(chunk, key, row, space))
        .transpose()
}
pub fn exact_input(
    source: &VisitSource,
    row: usize,
    space: &ActionSpace,
) -> Result<(Observation, Option<Action>)> {
    ensure(
        row < input_rows(&source.chunk)?,
        "visit row is not a decision",
    )?;
    let o = exact_observation(&source.chunk, row)?;
    let action = match &source.chunk.record {
        Record::Trajectory(t) => Some(action_row(&source.chunk, &t.actions, row, space)?),
        Record::ProbeSet(p) => p
            .fixed_actions
            .as_ref()
            .map(|k| action_row(&source.chunk, k, row, space))
            .transpose()?,
        _ => unreachable!(),
    };
    if let Some(a) = &action {
        space.check(a, o.dense_mask())?;
    }
    Ok((o, action))
}
pub fn observed_return(
    source: &VisitSource,
    row: usize,
    input: &Observation,
    action: Option<&Action>,
    q: bool,
    space: &ActionSpace,
) -> Result<Option<f64>> {
    let Record::Trajectory(t) = &source.chunk.record else {
        return Ok(None);
    };
    if t.evaluation.continuation_policy.trim().is_empty() {
        return Ok(None);
    };
    let (o, a) = exact_input(source, row, space)?;
    if &o != input || q && a.as_ref() != action {
        return Ok(None);
    };
    trajectory_return_to_go(&source.chunk, row)
}
pub(crate) fn position_in_slice(
    slice: &ValueSlice,
    o: &Observation,
    action: Option<&Action>,
    policy_action: Option<&Action>,
) -> Result<Option<[f32; 2]>> {
    if slice.mode == crate::policy::slice::SliceMode::SamplesOnly {
        return Ok(None);
    }
    ensure(
        o.dense_values().len() == slice.state.fixed.len(),
        "visit observation width",
    )?;
    if o.dense_mask() != slice.state.anchor_mask() {
        return Ok(None);
    };
    let axes = [Some(slice.x), slice.y];
    for (i, (&v, &fixed)) in o.dense_values().iter().zip(&slice.state.fixed).enumerate() {
        if !axes.contains(&Some(ValueCoordinate::Observation(i)))
            && if slice.state.numeric(i) {
                (f64::from(v) - f64::from(fixed)).abs() > f64::from(slice.state.tolerance)
            } else {
                v != fixed
            }
        {
            return Ok(None);
        }
    }
    let base = match &slice.conditioning {
        ActionConditioning::Fixed(a) => Some(a),
        ActionConditioning::Policy => policy_action,
    };
    if let (Some(Action::Continuous(a)), Some(Action::Continuous(b))) = (action, base) {
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            if !axes.contains(&Some(ValueCoordinate::Action(i)))
                && (f64::from(a) - f64::from(b)).abs() > f64::from(slice.state.tolerance)
            {
                return Ok(None);
            }
        }
    }
    if let (Some(Action::Discrete(a)), Some(Action::Discrete(b))) = (action, base)
        && !axes.contains(&Some(ValueCoordinate::DiscreteAction))
        && a != b
    {
        return Ok(None);
    };
    let mut position = [0.; 2];
    for (index, axis) in axes.into_iter().enumerate() {
        let Some(axis) = axis else { continue };
        let (value, range) = match axis {
            ValueCoordinate::Observation(i) => (o.dense_values()[i], slice.state.ranges[i]),
            ValueCoordinate::Action(i) => {
                let Some(Action::Continuous(a)) = action else {
                    return Ok(None);
                };
                (a[i], slice.action_ranges[i])
            }
            ValueCoordinate::DiscreteAction => {
                let Some(Action::Discrete(a)) = action else {
                    return Ok(None);
                };
                slice
                    .action_space()
                    .check(&Action::Discrete(*a), o.dense_mask())?;
                position[index] = *a as f32;
                continue;
            }
        };
        if value < range[0] || value > range[1] {
            return Ok(None);
        };
        position[index] = value;
    }
    Ok(Some(position))
}
fn bin(slice: &ValueSlice, position: [f32; 2]) -> Result<usize> {
    let nx = slice.axis_len(slice.x, 0)?;
    let mut ids = [0; 2];
    for (index, axis) in std::iter::once(slice.x).chain(slice.y).enumerate() {
        let n = slice.axis_len(axis, index)?;
        ids[index] = if axis == ValueCoordinate::DiscreteAction {
            (0..n)
                .find(|&i| slice.coordinate_at(axis, i, n).ok() == Some(position[index]))
                .ok_or_else(|| Error::Action("invalid category bin".into()))?
        } else {
            let range = match axis {
                ValueCoordinate::Observation(i) => slice.state.ranges[i],
                ValueCoordinate::Action(i) => slice.action_ranges[i],
                _ => unreachable!(),
            };
            (((f64::from(position[index]) - f64::from(range[0]))
                / (f64::from(range[1]) - f64::from(range[0]))
                * n as f64)
                .floor() as usize)
                .min(n - 1)
        };
    }
    Ok(ids[1] * nx + ids[0])
}
type PolicyActions<'a> = dyn FnMut(&[Observation]) -> Result<Vec<Option<Action>>> + 'a;
pub(crate) fn prepare_visits(
    sources: &[VisitSource],
    slice: &ValueSlice,
    q: bool,
    accounting: &Accounting,
    cancelled: &impl Fn() -> bool,
    capacity: usize,
    policy_actions: &mut PolicyActions<'_>,
) -> Result<PreparedValueVisits> {
    let mut visits = PreparedValueVisits {
        markers: Vec::new(),
        filtered_count: 0,
        counts: vec![0; slice.len()?],
    };
    let total = sources.iter().try_fold(0usize, |n, s| {
        Ok::<_, Error>(n.saturating_add(input_rows(&s.chunk)?))
    })?;
    ensure(total <= 8192, "visit population exceeds 8192 decisions")?;
    for source in sources {
        let n = input_rows(&source.chunk)?;
        for start in (0..n).step_by(capacity) {
            if cancelled() {
                return Ok(visits);
            };
            let inputs = (start..(start + capacity).min(n))
                .map(|row| exact_input(source, row, slice.action_space()))
                .collect::<Result<Vec<_>>>()?;
            let observations: Vec<_> = inputs.iter().map(|(o, _)| o.clone()).collect();
            let policies = if q && matches!(slice.conditioning, ActionConditioning::Policy) {
                policy_actions(&observations)?
            } else {
                vec![None; inputs.len()]
            };
            ensure(policies.len() == inputs.len(), "visit policy row mismatch")?;
            for (offset, ((o, a), policy)) in inputs.into_iter().zip(policies).enumerate() {
                if q && a.is_none() {
                    continue;
                };
                if q && matches!(slice.conditioning, ActionConditioning::Policy) && policy.is_none()
                {
                    continue;
                };
                if let Some(position) = position_in_slice(
                    slice,
                    &o,
                    if q { a.as_ref() } else { None },
                    policy.as_ref(),
                )? {
                    visits.filtered_count += 1;
                    visits.counts[bin(slice, position)?] += 1;
                    if visits.markers.len() < 1024 {
                        visits.markers.push(ValueVisit {
                            source: source.clone(),
                            row: start + offset,
                            position,
                        });
                    }
                }
            }
        }
    }
    accounting
        .source_rows
        .fetch_add(total as u64, Ordering::AcqRel);
    Ok(visits)
}
impl PreparedValueVisits {
    pub fn select_visit(
        &self,
        index: usize,
        slice: &ValueSlice,
        q: bool,
    ) -> Result<(ValuePin, ValueSlice, SelectionIntent)> {
        let visit = self
            .markers
            .get(index)
            .ok_or_else(|| Error::State("visit index".into()))?;
        let (o, a) = exact_input(&visit.source, visit.row, slice.action_space())?;
        let mut slice = slice.clone();
        slice.state.fixed = o.dense_values().to_vec();
        if q {
            slice.conditioning = ActionConditioning::Fixed(
                a.ok_or_else(|| Error::Config("Q visit has no exact recorded action".into()))?,
            );
        }
        let input = match &visit.source.chunk.record {
            Record::Trajectory(_) => SelectedInput::TrajectoryStep {
                sequence: visit.source.key.sequence,
                step: visit.row,
            },
            Record::ProbeSet(p) => SelectedInput::ProbeRow {
                set: p.id.clone(),
                sequence: visit.source.key.sequence,
                row: visit.row,
            },
            _ => unreachable!(),
        };
        Ok((
            ValuePin::Recorded {
                source: visit.source.clone(),
                row: visit.row,
            },
            slice,
            SelectionIntent::Input(input),
        ))
    }
    pub fn neighborhood_count(
        &self,
        slice: &ValueSlice,
        input: &Observation,
        action: Option<&Action>,
    ) -> Result<u32> {
        Ok(position_in_slice(slice, input, action, action)?
            .map(|p| bin(slice, p))
            .transpose()?
            .and_then(|i| self.counts.get(i).copied())
            .unwrap_or(0))
    }
}
