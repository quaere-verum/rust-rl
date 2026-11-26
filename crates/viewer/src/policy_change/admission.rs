use super::PolicyBatch;
use rl::{
    Error,
    core::action::ActionSpace,
    diagnostics::schema::{Chunk, CriticRepresentation, PolicyParameters, Record},
};
// Borrowed inspection only: validation errors and role captions may copy these
// strings. Bound their combined bytes before any checked-reader allocation.
fn metadata_text_bytes(c: &Chunk) -> usize {
    let mut bytes = 0usize;
    let mut add = |s: &str| bytes = bytes.saturating_add(s.len());
    let identity = &c.context.identity;
    for s in [&identity.run_id, &identity.model_role] {
        add(s);
    }
    for s in [
        &identity.checkpoint_id,
        &c.context.probe_set,
        &c.context.trajectory,
    ]
    .into_iter()
    .flatten()
    {
        add(s);
    }
    for key in c.arrays.keys() {
        add(key);
    }
    match &c.record {
        Record::Policy(p) => {
            add(&p.actions);
            for s in [&p.entropy, &p.entropy_method, &p.masks]
                .into_iter()
                .flatten()
            {
                add(s);
            }
            match &p.parameters {
                PolicyParameters::Categorical { probabilities } => add(probabilities),
                PolicyParameters::Gaussian { mean, log_std, .. } => {
                    add(mean);
                    add(log_std);
                }
                PolicyParameters::Deterministic => {}
            }
        }
        Record::ProbeSet(p) => {
            for s in [&p.id, &p.methodology, &p.observations, &p.row_ids] {
                add(s);
            }
            for s in [&p.action_masks, &p.fixed_actions].into_iter().flatten() {
                add(s);
            }
            for (key, value) in &p.histories {
                add(key);
                add(value);
            }
        }
        Record::Critic(c) => {
            for s in [&c.actions, &c.continuation_policy].into_iter().flatten() {
                add(s);
            }
            match &c.representation {
                CriticRepresentation::Scalar { values }
                | CriticRepresentation::Discrete { values } => add(values),
                CriticRepresentation::Quantiles {
                    values,
                    locations,
                    masses,
                } => {
                    add(values);
                    add(locations);
                    add(masses);
                }
                CriticRepresentation::RawMoments { mean, raw_second } => {
                    add(mean);
                    add(raw_second);
                }
                CriticRepresentation::CategoricalRegression {
                    logits,
                    weights,
                    centers,
                    values,
                } => {
                    add(logits);
                    add(weights);
                    add(centers);
                    add(values);
                }
                CriticRepresentation::CategoricalReturn {
                    atoms,
                    probabilities,
                } => {
                    add(atoms);
                    add(probabilities);
                }
            }
        }
        // No other kind reaches whole-record validation through this pane.
        _ => return usize::MAX,
    }
    bytes
}
pub(crate) fn admit_source(
    chunk: &Chunk,
    probes: &Chunk,
    space: &ActionSpace,
    budget: usize,
) -> rl::Result<()> {
    if metadata_text_bytes(chunk).saturating_add(metadata_text_bytes(probes)) > 1024 {
        return Err(Error::Config(
            "Decision metadata text exceeds bounded preparation allowance".into(),
        ));
    }
    let mut width = match space {
        ActionSpace::Discrete { count } => *count,
        _ => space.dimensions(),
    };
    let mut metadata = 0usize;
    let mut validation_copy = 0usize;
    for c in [chunk, probes] {
        metadata = metadata.saturating_add(c.arrays.len());
        if let Record::ProbeSet(p) = &c.record {
            metadata = metadata.saturating_add(p.histories.len());
            if let Some(ids) = c.arrays.get(&p.row_ids)
                && let Ok(values) = ids.as_u64()
                && !values.is_sorted()
            {
                // Schema uniqueness validation sorts a temporary copy for
                // unsorted stable IDs, even when each scientific row is narrow.
                validation_copy = validation_copy.saturating_add(values.len().saturating_mul(8));
            }
        }
        for a in c.arrays.values() {
            let rows = a.shape().first().copied().unwrap_or(0);
            if let Some(row_width) = a.len().checked_div(rows) {
                width = width.max(row_width);
            }
        }
        if let Record::Policy(p) = &c.record
            && let PolicyParameters::Gaussian { log_std, .. } = &p.parameters
            && let Some(a) = c.arrays.get(log_std)
            && a.shape().len() == 1
        {
            width = width.max(a.len());
        }
    }
    // A conservative simultaneous-owner bound: full-vector KL may hold four
    // f64 rows; validation/matching adds masks, observations and references.
    // 128 bytes/maximum component covers both endpoints and growth capacity;
    // 8 KiB covers temporary formatting and fixed container overhead. Reference
    // vectors from schema validation are bounded by the borrowed metadata count.
    let bytes = 8192usize
        .saturating_add(width.saturating_mul(128))
        .saturating_add(metadata.saturating_mul(32))
        .saturating_add(validation_copy)
        // Eight extreme f64 fixed-point previews can temporarily coexist as
        // component strings, joined text and the final String (with capacity).
        // That separate formatting phase needs no full-row comparison vectors.
        .max(12 * 1024);
    if bytes > budget {
        return Err(Error::Config(format!(
            "Decision temporary workspace exceeds {} KiB; wide imported input/action data unavailable",
            budget / 1024
        )));
    }
    Ok(())
}
pub fn checked_batch<'a>(
    chunk: &'a Chunk,
    probes: &'a Chunk,
    space: &'a ActionSpace,
    budget: usize,
) -> rl::Result<PolicyBatch<'a>> {
    admit_source(chunk, probes, space, budget)?;
    PolicyBatch::new(chunk, Some(probes), space)
}
