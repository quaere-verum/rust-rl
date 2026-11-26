//! Worker-only conditional evidence preparation. Sparse evidence never becomes a surface.
use super::*;

fn condition_charge(r: &ValueRequest) -> usize {
    8192 + r.current.manifest().observation_dim.saturating_mul(8)
        + r.current
            .manifest()
            .action_space
            .dimensions()
            .saturating_mul(8)
}
pub(super) fn grid_charge(r: &ValueRequest) -> usize {
    r.empirical.as_ref().map_or(0, |s| {
        8192 + s
            .capacity()
            .next_power_of_two()
            .max(4)
            .saturating_mul(condition_charge(r) + 512)
    })
}
pub(super) fn point_charge(r: &ValueRequest) -> usize {
    r.empirical.as_ref().map_or(0, |_| {
        point_charge_for_samples(r, rl::diagnostics::empirical::MAX_EMPIRICAL_SAMPLES)
    })
}
pub(super) fn point_charge_for_samples(r: &ValueRequest, samples: usize) -> usize {
    r.empirical.as_ref().map_or(0, |_| {
        condition_charge(r) + 4096 + 2 * samples * std::mem::size_of::<[f64; 2]>()
    })
}
fn mean_output(output: &ValueOutput) -> bool {
    matches!(
        output,
        ValueOutput::ExpectedReturn | ValueOutput::Q1 | ValueOutput::Q2
    )
}
/// Slice tolerances are useful for visitation, but never certify an empirical match.
fn on_exact_slice(slice: &ValueSlice, input: &Observation, action: Option<&Action>) -> bool {
    use super::super::slice::ValueCoordinate;
    let plotted = |c| slice.x == c || slice.y == Some(c);
    if input.dense_mask() != slice.state.anchor_mask()
        || input
            .dense_values()
            .iter()
            .enumerate()
            .any(|(i, &x)| !plotted(ValueCoordinate::Observation(i)) && x != slice.state.fixed[i])
    {
        return false;
    }
    match (&slice.conditioning, action) {
        (
            ActionConditioning::Fixed(Action::Continuous(fixed)),
            Some(Action::Continuous(actual)),
        ) => {
            fixed.len() == actual.len()
                && fixed
                    .iter()
                    .zip(actual)
                    .enumerate()
                    .all(|(i, (a, b))| plotted(ValueCoordinate::Action(i)) || a == b)
        }
        (ActionConditioning::Fixed(Action::Discrete(fixed)), Some(Action::Discrete(actual))) => {
            plotted(ValueCoordinate::DiscreteAction) || fixed == actual
        }
        (ActionConditioning::Fixed(_), None) => true, // V has no first action.
        (ActionConditioning::Policy, _) => true,
        _ => false,
    }
}
pub(super) fn prepare_grid(
    r: &ValueRequest,
    grid: &mut PreparedValueGrid,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<()> {
    let Some(source) = &r.empirical else {
        grid.empirical_error = r.empirical_unavailable.clone();
        if r.reference_empirical {
            grid.reference_error = Some(r.empirical_unavailable.clone().unwrap_or_else(|| {
                "No conditional empirical evidence for this actual critic revision".into()
            }));
        }
        return Ok(());
    };
    match source.prepare(&r.current, &r.output, cancelled) {
        Ok(batch) => grid.empirical = batch,
        Err(error) => {
            grid.empirical_error = Some(super::super::empirical::error_text(error));
        }
    }
    if !r.reference_empirical {
        return Ok(());
    }
    let result = (|| -> Result<()> {
        ensure(
            mean_output(&r.output),
            "Empirical mean reference requires expected return, Q1 or Q2; selected empirical CDF remains available",
        )?;
        ensure(
            r.condition == ComparisonCondition::SameInput,
            "Empirical reference compares the same complete input/action only",
        )?;
        if let Some(error) = &grid.empirical_error {
            return Err(Error::Config(error.clone()));
        }
        let mut reference = PreparedValueSeries {
            values: Vec::new(),
            positions: Vec::new(),
            rows: Vec::new(),
            actions: Vec::new(),
        };
        let mut differences = reference.clone();
        for (index, condition) in grid.empirical.conditions.iter().enumerate() {
            if cancelled() {
                return Err(Error::State("Empirical comparison superseded".into()));
            }
            if condition.summary.count < 2
                || !on_exact_slice(&r.slice, &condition.input, condition.action.as_ref())
            {
                continue;
            }
            // Multiple separately recorded batches are not silently pooled or selected.
            if grid
                .empirical
                .exact(&condition.input, condition.action.as_ref())
                .is_err()
            {
                continue;
            }
            let Some(position) = visitation::position_in_slice(
                &r.slice,
                &condition.input,
                condition.action.as_ref(),
                None,
            )?
            else {
                continue;
            };
            let requested_action = if matches!(r.slice.conditioning, ActionConditioning::Policy)
                && condition.action.is_some()
            {
                None
            } else {
                condition.action.clone()
            };
            let (prediction, actual_action) = evaluate_point(
                &r.current,
                &condition.input,
                requested_action,
                &r.output,
                accounting,
                with_bundle,
            )?;
            if actual_action != condition.action {
                continue;
            }
            let value = scalar_at(
                &prediction,
                actual_action.as_ref(),
                &r.output,
                r.current.utility(),
            )?;
            reference.positions.push(position);
            reference.values.push(condition.summary.mean);
            reference.rows.push(index);
            reference.actions.push(condition.action.clone());
            if value.is_finite() {
                differences.positions.push(position);
                differences.values.push(value - condition.summary.mean);
                differences.rows.push(index);
                differences.actions.push(condition.action.clone());
            }
        }
        grid.reference_values = (0..grid.values.len())
            .map(|i| {
                let input = grid.input(i).ok()?;
                grid.empirical
                    .exact(&input, grid.actions[i].as_ref())
                    .ok()
                    .flatten()
                    .filter(|c| c.summary.count >= 2)
                    .map(|c| c.summary.mean)
            })
            .collect();
        if reference.values.is_empty() {
            return Err(Error::Config(grid.empirical.rejections.first().cloned().unwrap_or_else(|| "No supported empirical conditions on this exact slice/action; unmeasured inputs remain empty".into())));
        }
        grid.reference_series = Some(reference);
        grid.empirical_differences = Some(differences);
        Ok(())
    })();
    if let Err(error) = result {
        grid.reference_error = Some(super::super::empirical::error_text(error));
    }
    Ok(())
}
pub(super) fn prepare_point(
    r: &ValueRequest,
    grid: &PreparedValueGrid,
    point: &mut PreparedValuePoint,
) -> Result<()> {
    let Some(source) = &r.empirical else {
        point.empirical_error = grid.empirical_error.clone();
        return Ok(());
    };
    let action = if r.current.kind == ValueCriticKind::StateValue {
        None
    } else {
        point.action.as_ref()
    };
    let result = (|| -> Result<Option<PreparedEmpirical>> {
        let Some(condition) =
            grid.empirical
                .selected(&point.input, action, r.empirical_condition.as_ref())?
        else {
            return Ok(None);
        };
        let empirical = source.distribution(condition)?;
        let forecast = point.prediction.weighted_atoms().ok();
        let cdf = empirical.cdf_points()?;
        ensure(
            cdf.capacity() <= condition.summary.count.saturating_mul(2),
            "Empirical CDF allocation exceeds the verified sample reservation",
        )?;
        Ok(Some(PreparedEmpirical {
            condition: condition.clone(),
            cdf,
            histogram: empirical.histogram(24).ok(),
            wasserstein: forecast
                .as_ref()
                .and_then(|q| q.wasserstein1(&empirical).ok()),
            mean_crps: forecast.as_ref().and_then(|q| q.mean_crps(&empirical).ok()),
        }))
    })();
    match result {
        Ok(evidence) => point.empirical = evidence,
        Err(error) => point.empirical_error = Some(super::super::empirical::error_text(error)),
    }
    if point.empirical.is_none() && point.empirical_error.is_none() {
        point.empirical_error = grid
            .empirical_error
            .clone()
            .or_else(|| if grid.empirical.conditions.is_empty() { grid.empirical.rejections.first().cloned() }
                else {Some("No conditional evidence at this exact observation, mask and complete first action".into())});
    }
    if r.reference_empirical {
        if let Some(e) = &point.empirical
            && e.condition.summary.count >= 2
            && mean_output(&r.output)
            && r.condition == ComparisonCondition::SameInput
        {
            point.reference_value = Some(e.condition.summary.mean);
            point.reference_action = e.condition.action.clone();
        } else {
            point.reference_error = Some(if r.condition != ComparisonCondition::SameInput {
                "Empirical mean differences require SameInput conditioning".into()
            } else {
                "No matched conditional mean estimate at this complete input/action".into()
            });
        }
    }
    Ok(())
}
