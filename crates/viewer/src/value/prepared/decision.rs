//! One-decision loss: genuine state population, actual learner policy and exact Q* only.
use super::*;
use crate::{plot_data::numeric_row, policy_change::PolicyBatch};
use rl::diagnostics::{
    reference::{
        DecisionLoss, DecisionPolicy, compatible_contexts, compatible_decision_loss, decision_loss,
    },
    schema::{PolicyParameters, Record},
};

pub(super) fn compatible(r: &ValueRequest) -> Result<()> {
    let policy = r.current_policy.as_ref().ok_or_else(|| {
        Error::Config(
            "Decision loss: current learner policy unavailable at selected checkpoint".into(),
        )
    })?;
    let oracle = r.reference_value.as_ref().ok_or_else(|| {
        Error::Config("Decision loss requires a verified optimal Q reference".into())
    })?;
    policy.check_available()?;
    oracle.check_available()?;
    ensure(
        policy.identity().revision == r.requested_revision
            && policy.identity().key.root == r.current.identity.key.root
            && policy
                .reference_info
                .declaration()
                .is_some_and(|d| d.model_role == r.current.role),
        "Decision loss policy does not belong to the selected learner checkpoint",
    )?;
    let context = policy
        .reference_info
        .context()
        .ok_or_else(|| Error::Config("Current policy scientific context unknown".into()))?;
    let population_context = r
        .current
        .reference_info
        .context()
        .ok_or_else(|| Error::Config("Value population scientific context unknown".into()))?;
    compatible_contexts(context, population_context)?;
    ensure(
        policy.decision_supported(),
        "Current policy is not an admitted Markov discrete policy",
    )?;
    let declaration = oracle
        .reference_info
        .declaration()
        .ok_or_else(|| Error::Config("Oracle declaration missing".into()))?;
    policy.reference_info.compatible(&oracle.reference_info)?;
    compatible_decision_loss(context, declaration, &oracle.names[0])?;
    ensure(
        matches!(&oracle.data, Data::Recorded(f) if f.len() == 1)
            && oracle.kind == ValueCriticKind::DiscreteQ,
        "Decision loss requires recorded complete optimal Q rows",
    )?;
    ensure(
        r.slice.state.grid_eligibility() != crate::policy::slice::GridEligibility::HistoryBound,
        "History-dependent policies are unavailable",
    )?;
    Ok(())
}

fn row_loss(
    batch: &PolicyBatch<'_>,
    row: usize,
    q: &[f64],
    legal: &[bool],
) -> Result<DecisionLoss> {
    ensure(
        batch.effective_mask(row)?.as_deref().map_or_else(
            || legal.iter().all(|&allowed| allowed),
            |mask| mask == legal,
        ),
        "Policy and exact oracle legal masks differ",
    )?;
    match batch.parameters() {
        PolicyParameters::Categorical { probabilities } => {
            let p = numeric_row(
                batch
                    .parameter_array(probabilities)
                    .ok_or_else(|| Error::Config("Categorical probabilities absent".into()))?,
                row,
            )?;
            decision_loss(q, legal, DecisionPolicy::Categorical(&p))
        }
        PolicyParameters::Deterministic => {
            let a = usize::try_from(batch.actions().as_u64()?[row])
                .map_err(|_| Error::Action("Policy action overflow".into()))?;
            decision_loss(q, legal, DecisionPolicy::Deterministic(a))
        }
        _ => Err(Error::Config(
            "Decision loss requires categorical probabilities or a deterministic discrete action"
                .into(),
        )),
    }
}

fn oracle_loss(
    r: &ValueRequest,
    input: &Observation,
    batch: &PolicyBatch<'_>,
    row: usize,
) -> Result<Option<DecisionLoss>> {
    let oracle = r.reference_value.as_ref().unwrap();
    let Data::Recorded(frames) = &oracle.data else {
        unreachable!()
    };
    let Some(index) = oracle.reference_info.lookup(&frames[0].probes, input)? else {
        return Ok(None);
    };
    let ValuePrediction::Discrete { values, legal } =
        recorded_prediction(oracle, index, input, None)?
    else {
        return Err(Error::Config(
            "Oracle does not contain complete Q values".into(),
        ));
    };
    row_loss(batch, row, &values, &legal).map(Some)
}

fn losses(
    r: &ValueRequest,
    inputs: &[Observation],
    rows: Option<&[usize]>,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Vec<Option<DecisionLoss>>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let policy = r.current_policy.as_ref().unwrap();
    let mut results = Vec::with_capacity(inputs.len());
    if let Some(frame) = policy.recorded_frame() {
        let batch = crate::policy_change::admission::checked_batch(
            &frame.chunk,
            &frame.probes,
            &r.current.manifest().action_space,
            POLICY_SCRATCH,
        )?;
        ensure(
            batch.rows() <= 8192,
            "Current policy population exceeds admission",
        )?;
        for (i, input) in inputs.iter().enumerate() {
            if stop(cancelled, accounting) {
                return Ok(results);
            }
            let row = if let Some(rows) = rows {
                Some(rows[i])
            } else {
                let mut found = None;
                for candidate in 0..batch.rows() {
                    if candidate.is_multiple_of(128) && stop(cancelled, accounting) {
                        return Ok(results);
                    }
                    let mut actual =
                        crate::policy::prepared::exact_observation(&frame.probes, candidate)?;
                    actual
                        .as_dense_mut()
                        .expect("dense diagnostic input")
                        .action_mask = batch.effective_mask(candidate)?;
                    if actual == *input {
                        ensure(
                            found.is_none(),
                            "Duplicate current-policy state rows are ambiguous",
                        )?;
                        found = Some(candidate);
                    }
                }
                found
            };
            results.push(
                row.map(|row| oracle_loss(r, input, &batch, row))
                    .transpose()?
                    .flatten(),
            );
        }
    } else {
        let source = policy
            .model_source()
            .ok_or_else(|| Error::Config("Current policy source unavailable".into()))?;
        with_bundle(&policy.identity().key, source, &mut |model| {
            let name =
                crate::policy::prepared::selected_name(model.configuration(), policy.name())?;
            ensure(
                name == policy.identity().name,
                "Current policy model subject changed",
            )?;
            let request = InspectionRequest::new(name, inputs.to_vec());
            let started = Instant::now();
            let result = model.inspect(&request);
            accounting
                .inference_micros
                .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
            let result = result?;
            if stop(cancelled, accounting) {
                return Ok(());
            }
            let batch = crate::policy_change::admission::checked_batch(
                &result.policy,
                &result.probes,
                &r.current.manifest().action_space,
                POLICY_SCRATCH,
            )?;
            ensure(
                batch.rows() == inputs.len(),
                "Policy inspection row count changed",
            )?;
            for (row, input) in inputs.iter().enumerate() {
                ensure(
                    crate::policy::prepared::exact_observation(&result.probes, row)? == *input,
                    "Inspected policy row differs from requested complete state",
                )?;
                results.push(oracle_loss(r, input, &batch, row)?);
            }
            Ok(())
        })?;
    }
    Ok(results)
}

fn exact_position(slice: &ValueSlice, input: &Observation) -> Result<Option<[f32; 2]>> {
    for (i, (&value, &fixed)) in input
        .dense_values()
        .iter()
        .zip(&slice.state.fixed)
        .enumerate()
    {
        if slice.x != super::super::slice::ValueCoordinate::Observation(i)
            && slice.y != Some(super::super::slice::ValueCoordinate::Observation(i))
            && value != fixed
        {
            return Ok(None);
        }
    }
    visitation::position_in_slice(slice, input, None, None)
}

pub(super) fn grid(
    mut grid: PreparedValueGrid,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<Arc<PreparedValueGrid>>> {
    let owner = grid.owner.clone();
    let r = &owner.request;
    r.reference_value
        .as_ref()
        .unwrap()
        .prepare_reference(accounting, cancelled)?;
    // Only genuine recorded inputs populate the finite diagnostic. Selected
    // states remain independently evaluable even when no such plot exists.
    if let Data::Recorded(frames) = &r.current.data {
        let rows = recorded_rows(&r.current)?;
        let policy = r.current_policy.as_ref().unwrap();
        let shared_rows = policy.recorded_frame().is_some_and(|p| {
            p.probe_key == frames[0].probe_key
                && std::ptr::eq(p.probes.as_ref(), frames[0].probes.as_ref())
        });
        if let Some(p) = policy.recorded_frame() {
            ensure(
                shared_rows
                    || rows
                        .saturating_mul(visitation::input_rows(&p.probes)?)
                        .saturating_mul(r.current.manifest().observation_dim)
                        <= 1024 * 1024,
                "Current policy matching exceeds bounded work admission",
            )?;
        }
        let capacity = batch_capacity(&r.current)?.min(crate::policy::prepared::batch_capacity(
            r.current.manifest().observation_dim,
            r.current.manifest().action_space.dimensions(),
            true,
        )?);
        let mut distinct = [[0.; 2]; 5];
        for start in (0..rows).step_by(capacity) {
            if stop(cancelled, accounting) {
                return Ok(None);
            }
            let mut inputs = Vec::with_capacity(capacity);
            let mut selected_rows = Vec::with_capacity(capacity);
            for row in start..(start + capacity).min(rows) {
                let input = crate::policy::prepared::exact_observation(&frames[0].probes, row)?;
                if let Some(position) = exact_position(&r.slice, &input)? {
                    inputs.push(input);
                    selected_rows.push(row);
                    grid.positions.push(position);
                    if grid.distinct_positions < distinct.len()
                        && !distinct[..grid.distinct_positions].contains(&position)
                    {
                        distinct[grid.distinct_positions] = position;
                        grid.distinct_positions += 1;
                    }
                }
            }
            let values = losses(
                r,
                &inputs,
                shared_rows.then_some(selected_rows.as_slice()),
                accounting,
                cancelled,
                with_bundle,
            )?;
            if stop(cancelled, accounting) {
                return Ok(None);
            }
            for value in values {
                grid.valid.push(value.is_some());
                grid.values.push(value.map_or(0., |v| v.loss));
                grid.actions.push(None);
                grid.policy_overlay.push(None);
            }
            grid.rows.extend(selected_rows);
            accounting.grid_batches.fetch_add(1, Ordering::AcqRel);
        }
    }
    grid.reference_values.clear();
    grid.reference_actions.clear();
    domain::finish(&mut grid, vec![]);
    grid.reference_color_domain = None;
    Ok(Some(Arc::new(grid)))
}

pub(super) fn point(
    owner: &Arc<RequestOwner>,
    input: Observation,
    lease: Lease,
    accounting: &Arc<Accounting>,
    cancelled: &impl Fn() -> bool,
    with_bundle: &mut WithBundle<'_>,
) -> Result<Option<Arc<PreparedValuePoint>>> {
    let r = &owner.request;
    if let ValuePin::Probe { source, .. } = &r.pin {
        ensure(
            !matches!(&source.chunk.record, Record::ProbeSet(p) if !p.histories.is_empty()),
            "History-dependent selected input unavailable",
        )?;
    }
    r.reference_value
        .as_ref()
        .unwrap()
        .prepare_reference(accounting, cancelled)?;
    let values = losses(
        r,
        std::slice::from_ref(&input),
        None,
        accounting,
        cancelled,
        with_bundle,
    )?;
    if stop(cancelled, accounting) {
        return Ok(None);
    }
    let loss = values.into_iter().next().flatten().ok_or_else(|| Error::Config("Decision loss unavailable: exact live state, decision time, legal mask or current-policy row missing (terminal decisions excluded)".into()))?;
    let (recorded_key, recorded_row, recorded_revision) = match &r.pin {
        ValuePin::Recorded { source, row } | ValuePin::Probe { source, row, .. } => (
            Some(source.key.clone()),
            Some(*row),
            Some(source.chunk.context.identity.learner_update_revision),
        ),
        _ => (None, None, None),
    };
    Ok(Some(Arc::new(PreparedValuePoint {
        input,
        action: None,
        action_curve: None,
        prediction: ValuePrediction::Scalar(loss.loss),
        decision_loss: Some(loss),
        value: loss.loss,
        reference_prediction: None,
        profile: None,
        reference_profile: None,
        reference_value: None,
        reference_action: None,
        reference_error: None,
        empirical: None,
        empirical_error: None,
        observed_return: None,
        final_observation: false,
        recorded_revision,
        recorded_key,
        recorded_row,
        policy_conditioned: true,
        neighborhood_count: 0,
        _owner: owner.clone(),
        _lease: lease,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::diagnostics::schema::{Array, Chunk, PolicyRecord, RecordContext};
    #[test]
    fn unmasked_categorical_policy_uses_all_legal_actions() {
        let chunk = Chunk {
            context: RecordContext::new("test", 0),
            record: Record::Policy(PolicyRecord {
                actions: "a".into(),
                parameters: PolicyParameters::Categorical {
                    probabilities: "p".into(),
                },
                entropy: None,
                entropy_method: None,
                masks: None,
                stochastic: false,
            }),
            arrays: std::collections::BTreeMap::from([
                ("a".into(), Array::u64(vec![1], vec![1]).unwrap()),
                (
                    "p".into(),
                    Array::f64(vec![1, 2], vec![0.25, 0.75]).unwrap(),
                ),
            ]),
        };
        let space = rl::core::action::ActionSpace::Discrete { count: 2 };
        let batch = PolicyBatch::new(&chunk, None, &space).unwrap();
        assert_eq!(
            row_loss(&batch, 0, &[-2., -5.], &[true, true])
                .unwrap()
                .loss,
            2.25
        );
        assert!(row_loss(&batch, 0, &[-2., -5.], &[true, false]).is_err());
    }
}
