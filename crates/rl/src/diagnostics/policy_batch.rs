//! Checked recorded policy batches shared by the current policy workspace.

use crate::{Error, Result, core::action::ActionSpace, diagnostics::schema::*};

pub struct PolicyBatch<'a> {
    chunk: &'a Chunk,
    meta: &'a PolicyRecord,
    probes: Option<&'a Chunk>,
    space: &'a ActionSpace,
    rows: usize,
}
fn check(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Config(reason.into()))
    }
}
impl<'a> PolicyBatch<'a> {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn parameters(&self) -> &'a PolicyParameters {
        &self.meta.parameters
    }
    pub fn actions(&self) -> &'a Array {
        &self.chunk.arrays[&self.meta.actions]
    }
    pub fn action_space(&self) -> &'a ActionSpace {
        self.space
    }
    /// Genuine input arrays already certified with this policy batch.
    pub fn recorded_probes(&self) -> Option<&'a Chunk> {
        self.probes
    }
    /// Parameter arrays borrow the checked source; full widths are preserved.
    pub fn parameter_array(&self, key: &str) -> Option<&'a Array> {
        self.chunk.arrays.get(key)
    }
    /// Full effective mask of a checked row. Admission must precede this
    /// allocation; independent row validity is not an input-matching certificate.
    pub fn effective_mask(&self, row: usize) -> Result<Option<Vec<bool>>> {
        check(row < self.rows, "policy mask row outside recording")?;
        self.mask(row)
    }
    /// Same checked effective mask, with an explicit allowance for every mask
    /// capacity alive during construction (probe and declared policy masks).
    pub fn effective_mask_with_budget(
        &self,
        row: usize,
        scratch_budget: usize,
    ) -> Result<Option<Vec<bool>>> {
        check(row < self.rows, "policy mask row outside recording")?;
        self.mask_with_budget(row, scratch_budget)
    }
    /// Checks dimensions and all recorded values once. Optional probes supply
    /// effective masks; callers associating fixed states must supply them.
    pub fn new(
        chunk: &'a Chunk,
        probes: Option<&'a Chunk>,
        space: &'a ActionSpace,
    ) -> Result<Self> {
        chunk.validate()?;
        space.validate()?;
        let Record::Policy(meta) = &chunk.record else {
            return Err(Error::Config("expected policy record".into()));
        };
        let rows = array(chunk, &meta.actions)?.shape()[0];
        let batch = Self {
            chunk,
            meta,
            probes,
            space,
            rows,
        };
        if let Some(probes) = probes {
            probes.validate()?;
            let Record::ProbeSet(p) = &probes.record else {
                return Err(Error::Config("policy requires probe definition".into()));
            };
            check(
                chunk.context.identity.run_id == probes.context.identity.run_id
                    && chunk.context.probe_set.as_deref() == Some(&p.id)
                    && array(probes, &p.observations)?.shape()[0] == rows,
                "policy/probe identity or rows differ",
            )?;
        }
        check(
            !matches!(meta.parameters, PolicyParameters::Deterministic) || !meta.stochastic,
            "sampled actions do not define deterministic-policy change",
        )?;
        for row in 0..rows {
            batch.validate_row(row)?;
        }
        Ok(batch)
    }
    fn mask(&self, row: usize) -> Result<Option<Vec<bool>>> {
        self.mask_with_budget(row, usize::MAX)
    }
    fn mask_with_budget(&self, row: usize, scratch_budget: usize) -> Result<Option<Vec<bool>>> {
        let prepare = |a: &Array, retained: usize| -> Result<Vec<bool>> {
            let width = a.shape()[1];
            check(
                retained
                    .checked_add(width)
                    .is_some_and(|b| b <= scratch_budget),
                "effective policy masks exceed scratch byte budget",
            )?;
            let mut values = Vec::with_capacity(width);
            check(
                retained
                    .checked_add(values.capacity())
                    .is_some_and(|b| b <= scratch_budget),
                "effective policy mask capacities exceed scratch byte budget",
            )?;
            values.extend(
                a.as_u8()?[row * width..(row + 1) * width]
                    .iter()
                    .map(|&v| v != 0),
            );
            Ok(values)
        };
        let mut mask = None;
        if let Some(probes) = self.probes
            && let Record::ProbeSet(p) = &probes.record
            && let Some(key) = &p.action_masks
        {
            let a = array(probes, key)?;
            mask = Some(prepare(a, 0)?);
        }
        if let Some(key) = &self.meta.masks {
            let a = array(self.chunk, key)?;
            let declared = prepare(a, mask.as_ref().map_or(0, Vec::capacity))?;
            check(
                mask.as_ref().is_none_or(|m| *m == declared),
                "policy and probe action masks differ",
            )?;
            mask = Some(declared);
        }
        Ok(mask)
    }
    fn validate_row(&self, row: usize) -> Result<()> {
        let actions = array(self.chunk, &self.meta.actions)?;
        let mask = self.mask(row)?;
        match self.space {
            ActionSpace::Discrete { count } => {
                check(
                    actions.dtype() == DType::U64 && actions.shape() == [self.rows],
                    "discrete policy action shape differs",
                )?;
                let action = actions.as_u64()?[row];
                check(
                    mask.as_ref()
                        .is_none_or(|m| m.len() == *count && m.iter().any(|&v| v)),
                    "invalid policy support",
                )?;
                check(
                    action < *count as u64 && mask.as_ref().is_none_or(|m| m[action as usize]),
                    "recorded discrete action violates support",
                )?;
            }
            _ => {
                check(
                    actions.shape() == [self.rows, self.space.dimensions()]
                        && matches!(actions.dtype(), DType::F32 | DType::F64),
                    "continuous policy action shape differs",
                )?;
                let values = numeric_row(actions, row)?;
                check(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite policy action",
                )?;
                if let ActionSpace::Box { low, high } = self.space {
                    check(
                        values
                            .iter()
                            .zip(low.iter().zip(high))
                            .all(|(&v, (&l, &h))| v >= f64::from(l) && v <= f64::from(h)),
                        "policy action outside bounds",
                    )?;
                }
            }
        }
        match &self.meta.parameters {
            PolicyParameters::Deterministic => {}
            PolicyParameters::Categorical { probabilities } => {
                let ActionSpace::Discrete { count } = self.space else {
                    return Err(Error::Config(
                        "categorical policy requires discrete action space".into(),
                    ));
                };
                let p = numeric_row(array(self.chunk, probabilities)?, row)?;
                check(
                    p.len() == *count
                        && p.iter().all(|v| v.is_finite() && *v >= 0.)
                        && (p.iter().sum::<f64>() - 1.).abs() < 1e-6,
                    "invalid categorical probabilities",
                )?;
                check(
                    mask.as_ref()
                        .is_none_or(|m| p.iter().zip(m).all(|(&p, &valid)| valid || p == 0.)),
                    "probability mass on masked action",
                )?;
            }
            PolicyParameters::Gaussian {
                mean,
                log_std,
                tanh_bounded,
            } => {
                check(
                    matches!(
                        (tanh_bounded, self.space),
                        (true, ActionSpace::Box { .. }) | (false, ActionSpace::Unbounded { .. })
                    ),
                    "unsupported Gaussian action transform",
                )?;
                let mean = numeric_row(array(self.chunk, mean)?, row)?;
                let std = gaussian_log_std(self.chunk, log_std, row)?;
                check(
                    mean.len() == self.space.dimensions()
                        && std.len() == mean.len()
                        && mean.iter().chain(&std).all(|v| v.is_finite()),
                    "invalid Gaussian parameters",
                )?;
            }
        }
        Ok(())
    }
}
pub fn array<'a>(chunk: &'a Chunk, key: &str) -> Result<&'a Array> {
    chunk
        .arrays
        .get(key)
        .ok_or_else(|| Error::State(format!("missing array {key}")))
}
pub fn numeric_row(a: &Array, index: usize) -> Result<Vec<f64>> {
    let rows = a.shape()[0];
    if index >= rows {
        return Err(Error::State("row outside recorded array".into()));
    }
    let width = a.len() / rows;
    let range = index * width..(index + 1) * width;
    Ok(match a.dtype() {
        DType::F32 => a.as_f32()?[range].iter().map(|&v| v as f64).collect(),
        DType::F64 => a.as_f64()?[range].to_vec(),
        DType::U64 => a.as_u64()?[range].iter().map(|&v| v as f64).collect(),
        DType::U8 => a.as_u8()?[range].iter().map(|&v| v as f64).collect(),
    })
}
pub fn row(chunk: &Chunk, key: &str, index: usize) -> Result<Vec<f64>> {
    numeric_row(array(chunk, key)?, index)
}
/// Gaussian log standard deviations may be shared `[D]` or per-input `[N,D]`.
pub fn gaussian_log_std(chunk: &Chunk, key: &str, index: usize) -> Result<Vec<f64>> {
    let values = array(chunk, key)?;
    if values.shape().len() == 1 {
        Ok(values.to_f64())
    } else {
        numeric_row(values, index)
    }
}
