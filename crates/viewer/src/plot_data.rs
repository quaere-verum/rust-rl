//! Checked presentation adapters. Scientific calculations live in `rl`.
use rl::{
    Error, Result,
    diagnostics::{representations::equal_row, schema::*},
};

pub use rl::diagnostics::policy_batch::{array, gaussian_log_std, numeric_row, row};
fn masks_equal(a: Option<&[f64]>, b: Option<&[f64]>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        (Some(a), None) | (None, Some(a)) => a.iter().all(|&v| v == 1.),
        (None, None) => true,
    }
}
/// Resolve the original input without requiring a probe at the selected model revision.
pub fn source_observation(source: &Chunk, index: usize) -> Result<rl::core::env::Observation> {
    let (observations, masks) = match &source.record {
        Record::ProbeSet(p) => (&p.observations, &p.action_masks),
        Record::Trajectory(t) => (&t.observations, &t.action_masks),
        _ => {
            return Err(Error::State(
                "selected source is not an input record".into(),
            ));
        }
    };
    Ok(rl::core::env::Observation::dense(
        row(source, observations, index)?
            .into_iter()
            .map(|v| v as f32)
            .collect(),
        masks
            .as_ref()
            .map(|key| row(source, key, index).map(|r| r.into_iter().map(|v| v != 0.).collect()))
            .transpose()?,
    ))
}
/// Match complete recorded inputs, including fixed actions and imported histories.
/// Row positions and reused probe IDs alone never establish input equivalence.
pub fn matching_input(source: &Chunk, source_row: usize, target: &Chunk) -> Result<Option<usize>> {
    matching_input_where(source, source_row, target, false, |_| Ok(true))
}
/// Match a stable probe population, including its construction and explicit row identity.
/// A trajectory selection has no population identity and matches its recorded input instead.
pub fn matching_probe_input(
    source: &Chunk,
    source_row: usize,
    target: &Chunk,
) -> Result<Option<usize>> {
    matching_input_where(source, source_row, target, true, |_| Ok(true))
}
/// Additional checked conditioning may reject a candidate without hiding a
/// later matching row. Ordinary matching retains its existing exact rules.
pub(crate) fn matching_input_where(
    source: &Chunk,
    source_row: usize,
    target: &Chunk,
    strict: bool,
    mut accept: impl FnMut(usize) -> Result<bool>,
) -> Result<Option<usize>> {
    let Record::ProbeSet(p) = &target.record else {
        return Err(Error::State("expected target probe set".into()));
    };
    if strict
        && let Record::ProbeSet(original) = &source.record
        && (original.id != p.id
            || original.population != p.population
            || original.methodology != p.methodology)
    {
        return Ok(None);
    }
    let (observations, masks) = match &source.record {
        Record::ProbeSet(p) => (&p.observations, &p.action_masks),
        Record::Trajectory(t) => (&t.observations, &t.action_masks),
        _ => {
            return Err(Error::State(
                "selected source is not an input record".into(),
            ));
        }
    };
    let observed = array(source, observations)?;
    let mask = masks
        .as_ref()
        .map(|key| row(source, key, source_row))
        .transpose()?;
    for i in 0..array(target, &p.observations)?.shape()[0] {
        if !equal_row(observed, array(target, &p.observations)?, source_row, i) {
            continue;
        }
        let target_mask = p
            .action_masks
            .as_ref()
            .map(|key| row(target, key, i))
            .transpose()?;
        if !masks_equal(target_mask.as_deref(), mask.as_deref()) {
            continue;
        }
        match &source.record {
            Record::ProbeSet(original) => {
                if strict
                    && !equal_row(
                        array(source, &original.row_ids)?,
                        array(target, &p.row_ids)?,
                        source_row,
                        i,
                    )
                {
                    continue;
                }
                let fixed_match = match (&original.fixed_actions, &p.fixed_actions) {
                    (Some(a), Some(b)) => {
                        equal_row(array(source, a)?, array(target, b)?, source_row, i)
                    }
                    (None, None) => true,
                    _ => false,
                };
                if !fixed_match || original.histories.keys().ne(p.histories.keys()) {
                    continue;
                }
                let mut matched = true;
                for (name, key) in &original.histories {
                    matched &= equal_row(
                        array(source, key)?,
                        array(target, &p.histories[name])?,
                        source_row,
                        i,
                    );
                }
                if !matched {
                    continue;
                }
            }
            Record::Trajectory(t) => {
                if !p.histories.is_empty() {
                    continue;
                }
                if let Some(key) = &p.fixed_actions
                    && (source_row >= array(source, &t.actions)?.shape()[0]
                        || !equal_row(
                            array(source, &t.actions)?,
                            array(target, key)?,
                            source_row,
                            i,
                        ))
                {
                    continue;
                }
            }
            _ => unreachable!(),
        }
        if accept(i)? {
            return Ok(Some(i));
        }
    }
    Ok(None)
}
pub fn trajectory_points(a: &Array, times: Option<&[f64]>, column: usize) -> Result<Vec<[f64; 2]>> {
    let rows = a.shape()[0];
    let width = a.len().checked_div(rows).unwrap_or(0);
    if column >= width || times.is_some_and(|t| t.len() != rows) {
        return Err(Error::State(
            "trajectory clock or column dimension mismatch".into(),
        ));
    }
    let values = a.to_f64();
    Ok((0..rows)
        .map(|i| [times.map_or(i as f64, |t| t[i]), values[i * width + column]])
        .collect())
}
