//! Semantic checks before committing or exposing decoded records to analysis/UI.
use super::{
    evaluation::{
        DISCOUNTED_REWARDS_KEY, EvaluationHorizon, PHYSICAL_DURATIONS_KEY, evaluation_horizon,
        trajectory_scenario_seed,
    },
    schema::*,
};
use crate::{Result, core::ensure};
fn floating(a: &Array) -> bool {
    matches!(a.dtype(), DType::F32 | DType::F64)
}
fn finite(a: &Array) -> bool {
    match &a.values {
        Values::F32(v) => v.iter().all(|x| x.is_finite()),
        Values::F64(v) => v.iter().all(|x| x.is_finite()),
        _ => true,
    }
}
fn positive(a: &Array) -> bool {
    a.shape().iter().all(|&n| n > 0)
}
fn categorical_regression(
    logits: &Array,
    weights: &Array,
    centers: &Array,
    values: &Array,
) -> Result<usize> {
    let rows = matrix(logits)?;
    let bins = logits.shape()[1];
    ensure(
        bins > 1
            && bins <= super::value::MAX_VALUE_HEADS
            && matrix(weights)? == rows
            && weights.shape() == logits.shape()
            && scalar(values)? == rows
            && floating(centers)
            && centers.shape() == [bins]
            && finite(logits)
            && finite(weights)
            && finite(centers)
            && finite(values),
        "invalid categorical regression arrays",
    )?;
    let read = |a: &Array, i: usize| match &a.values {
        Values::F32(v) => f64::from(v[i]),
        Values::F64(v) => v[i],
        _ => unreachable!("floating arrays admitted above"),
    };
    let centers = centers.to_f64();
    ensure(
        centers.windows(2).all(|pair| pair[0] < pair[1]),
        "categorical regression centers must increase",
    )?;
    for row in 0..rows {
        let offset = row * bins;
        let row_logits = || (offset..offset + bins).map(|i| read(logits, i));
        let peak = row_logits().fold(f64::NEG_INFINITY, f64::max);
        let scale: f64 = row_logits().map(|x| (x - peak).exp()).sum();
        let mut predicted = 0.;
        let mut magnitude = 0.;
        let mut total = 0.;
        for (bin, (logit, &center)) in row_logits().zip(&centers).enumerate() {
            let weight = read(weights, offset + bin);
            ensure(
                weight >= 0. && (weight - (logit - peak).exp() / scale).abs() <= 2e-5,
                "categorical regression weights differ from softmax logits",
            )?;
            predicted += weight * center;
            magnitude += (weight * center).abs();
            total += weight;
        }
        ensure(
            (total - 1.).abs() <= 2e-5
                && (predicted - read(values, row)).abs() <= 2e-5 * magnitude.max(1.),
            "categorical regression weights or decoded value are inconsistent",
        )?;
    }
    Ok(rows)
}

fn matrix(a: &Array) -> Result<usize> {
    ensure(
        floating(a) && a.shape().len() == 2 && positive(a),
        "expected nonempty floating matrix",
    )?;
    Ok(a.shape()[0])
}
fn categorical_return(atoms: &Array, probabilities: &Array) -> Result<usize> {
    let rows = matrix(probabilities)?;
    let width = probabilities.shape()[1];
    ensure(
        (2..=super::value::MAX_VALUE_HEADS).contains(&width)
            && floating(atoms)
            && atoms.shape() == [width]
            && finite(atoms)
            && atoms.to_f64().windows(2).all(|pair| pair[0] < pair[1]),
        "invalid categorical return support",
    )?;
    for row in 0..rows {
        let start = row * width;
        let mut total = 0.;
        for index in start..start + width {
            let mass = match &probabilities.values {
                Values::F32(values) => f64::from(values[index]),
                Values::F64(values) => values[index],
                _ => unreachable!("floating probability matrices admitted above"),
            };
            ensure(
                mass.is_finite() && mass >= 0.,
                "invalid categorical return mass",
            )?;
            total += mass;
        }
        ensure(
            (total - 1.).abs() <= 1e-6,
            "categorical return masses must sum to one",
        )?;
    }
    Ok(rows)
}
fn scalar(a: &Array) -> Result<usize> {
    ensure(
        floating(a)
            && positive(a)
            && (a.shape().len() == 1 || a.shape().len() == 2 && a.shape()[1] == 1),
        "expected nonempty scalar values",
    )?;
    Ok(a.shape()[0])
}
fn actions(a: &Array, n: usize) -> Result<()> {
    ensure(
        a.shape()[0] == n
            && positive(a)
            && finite(a)
            && ((a.dtype() == DType::U64 && a.shape().len() == 1)
                || (floating(a) && a.shape().len() == 2)),
        "invalid recorded action shape/dtype/values",
    )
}
fn mask(a: &Array, n: usize, live: bool) -> Result<()> {
    ensure(
        a.dtype() == DType::U8 && a.shape().len() == 2 && positive(a) && a.shape()[0] == n,
        "invalid recorded action-mask shape/dtype",
    )?;
    let width = a.shape()[1];
    ensure(
        a.as_u8()?.iter().all(|&x| x <= 1)
            && (!live || a.as_u8()?.chunks_exact(width).all(|r| r.contains(&1))),
        "invalid boolean mask or all-invalid live state",
    )
}
fn unique(values: &[u64]) -> bool {
    if values.is_sorted() {
        values.windows(2).all(|w| w[0] != w[1])
    } else {
        let mut copy = values.to_vec();
        copy.sort_unstable();
        copy.windows(2).all(|w| w[0] != w[1])
    }
}
pub(super) fn evaluation(e: &EvaluationMetadata) -> Result<()> {
    ensure(
        e.discount.is_finite()
            && (0. ..=1.).contains(&e.discount)
            && !e.continuation_policy.trim().is_empty()
            && !e.seed_methodology.trim().is_empty()
            && !e.truncation_handling.trim().is_empty(),
        "invalid evaluation metadata",
    )
}
pub(super) fn finite_initial_evaluation(e: &EvaluationMetadata) -> Result<()> {
    ensure(
        e.conditioning == Conditioning::InitialScenario
            && e.initial_action == ActionCondition::CurrentPolicy,
        "finite decision evaluation requires an unconditional initial scenario",
    )
}

pub(super) fn physical_durations(durations: impl IntoIterator<Item = u64>) -> Result<()> {
    ensure(
        durations
            .into_iter()
            .try_fold(0u64, |total, duration| {
                total.checked_add(duration).filter(|_| duration != 0)
            })
            .is_some(),
        "invalid physical trajectory durations",
    )
}
pub(super) fn validate(c: &Chunk) -> Result<()> {
    ensure(
        !c.context.identity.run_id.is_empty() && !c.context.identity.model_role.is_empty(),
        "empty diagnostic identity",
    )?;
    let get = |key: &str| &c.arrays[key]; // Reference existence is checked by Chunk first.
    match &c.record {
        Record::Metrics(_) => {}
        Record::Typed(record) => {
            let data = get(&record.payload);
            let kind = record.kind.record_kind();
            ensure(
                data.dtype() == DType::U8
                    && data.shape() == [data.byte_len()]
                    && data.byte_len() <= super::typed::MAX_TYPED_BYTES
                    && data.byte_len() > 0
                    && if kind == RecordKind::TypedEpisode {
                        c.context
                            .trajectory
                            .as_ref()
                            .is_some_and(|id| !id.is_empty())
                    } else {
                        c.context
                            .probe_set
                            .as_ref()
                            .is_some_and(|id| !id.is_empty())
                    },
                "invalid typed diagnostic chunk or identity",
            )?;
        }
        Record::Event(e) => ensure(
            !e.category.is_empty() && !e.severity.is_empty() && !e.message.is_empty(),
            "empty diagnostic event",
        )?,
        Record::ProbeSet(p) => {
            let x = get(&p.observations);
            let n = matrix(x)?;
            ensure(
                finite(x)
                    && !p.id.is_empty()
                    && !p.methodology.is_empty()
                    && c.context.probe_set.as_ref().is_none_or(|id| *id == p.id),
                "invalid probe identity or observation",
            )?;
            let ids = get(&p.row_ids);
            ensure(
                ids.dtype() == DType::U64 && ids.shape() == [n] && unique(ids.as_u64()?),
                "invalid/duplicate probe row identities",
            )?;
            if let Some(k) = &p.action_masks {
                mask(get(k), n, true)?;
            }
            if let Some(k) = &p.fixed_actions {
                actions(get(k), n)?;
            }
            for k in p.histories.values() {
                ensure(get(k).shape()[0] == n, "history/probe population mismatch")?;
            }
        }
        Record::Policy(p) => {
            let a = get(&p.actions);
            let n = a.shape()[0];
            actions(a, n)?;
            if let Some(k) = &p.entropy {
                ensure(scalar(get(k))? == n, "policy entropy row count differs")?;
            }
            if let Some(k) = &p.masks {
                let m = get(k);
                mask(m, n, true)?;
                ensure(
                    a.dtype() == DType::U64,
                    "continuous actions cannot use discrete masks",
                )?;
                for (i, &action) in a.as_u64()?.iter().enumerate() {
                    ensure(
                        action < m.shape()[1] as u64
                            && m.as_u8()?[i * m.shape()[1] + action as usize] == 1,
                        "recorded policy action violates mask",
                    )?;
                }
            }
            match &p.parameters {
                PolicyParameters::Categorical { probabilities } => {
                    let probs = get(probabilities);
                    ensure(
                        matrix(probs)? == n && a.dtype() == DType::U64,
                        "categorical prediction dimensions differ",
                    )?;
                    ensure(
                        a.as_u64()?
                            .iter()
                            .all(|&action| action < probs.shape()[1] as u64),
                        "categorical action out of range",
                    )?;
                    if let Some(k) = &p.masks {
                        ensure(
                            get(k).shape() == probs.shape(),
                            "probability/mask dimensions differ",
                        )?;
                    }
                }
                PolicyParameters::Gaussian { mean, log_std, .. } => {
                    let mean = get(mean);
                    let std = get(log_std);
                    ensure(
                        matrix(mean)? == n
                            && a.shape() == mean.shape()
                            && floating(std)
                            && (std.shape() == mean.shape() || std.shape() == [mean.shape()[1]]),
                        "Gaussian parameter dimensions differ",
                    )?;
                }
                PolicyParameters::Deterministic => {}
            }
        }
        Record::Critic(p) => {
            let n = match &p.representation {
                CriticRepresentation::Scalar { values } => scalar(get(values))?,
                CriticRepresentation::Discrete { values } => matrix(get(values))?,
                CriticRepresentation::RawMoments { mean, raw_second } => {
                    let n = scalar(get(mean))?;
                    ensure(
                        scalar(get(raw_second))? == n,
                        "moment prediction dimensions differ",
                    )?;
                    n
                }
                CriticRepresentation::CategoricalRegression {
                    logits,
                    weights,
                    centers,
                    values,
                } => {
                    ensure(
                        matches!(
                            p.action_condition,
                            ActionCondition::Fixed | ActionCondition::CurrentPolicy
                        ),
                        "categorical regression requires selected actions",
                    )?;
                    categorical_regression(get(logits), get(weights), get(centers), get(values))?
                }
                CriticRepresentation::CategoricalReturn {
                    atoms,
                    probabilities,
                } => {
                    ensure(
                        matches!(
                            p.action_condition,
                            ActionCondition::Fixed | ActionCondition::CurrentPolicy
                        ),
                        "categorical return prediction requires selected actions",
                    )?;
                    categorical_return(get(atoms), get(probabilities))?
                }
                CriticRepresentation::Quantiles {
                    values,
                    locations,
                    masses,
                } => {
                    let values = get(values);
                    let n = matrix(values)?;
                    let q = values.shape()[1];
                    let locations = get(locations);
                    let masses = get(masses);
                    ensure(
                        floating(locations)
                            && floating(masses)
                            && locations.shape() == [q]
                            && masses.shape() == [q]
                            && finite(locations)
                            && finite(masses),
                        "invalid quantile grid dimensions",
                    )?;
                    let locations = locations.to_f64();
                    let masses = masses.to_f64();
                    ensure(
                        locations.iter().all(|&t| t > 0. && t < 1.)
                            && locations.windows(2).all(|w| w[0] < w[1])
                            && masses.iter().all(|&m| m > 0.)
                            && (masses.iter().sum::<f64>() - 1.).abs() < 1e-5,
                        "invalid quantile locations/probability masses",
                    )?;
                    n
                }
            };
            if let Some(k) = &p.actions {
                actions(get(k), n)?;
            }
            ensure(
                !matches!(
                    p.action_condition,
                    ActionCondition::Fixed | ActionCondition::CurrentPolicy
                ) || p.actions.is_some(),
                "action-conditioned critic lacks evaluated actions",
            )?;
        }
        Record::Activations(p) => {
            ensure(!p.network.is_empty(), "empty activation network identity")?;
            let mut rows = None;
            for (name, key) in &p.layers {
                let a = get(key);
                let n = matrix(a)?;
                ensure(
                    rows.is_none_or(|r| r == n),
                    "activation layers have different probe populations",
                )?;
                rows = Some(n);
                if let Some(s) = p.summaries.get(name) {
                    ensure(
                        s.count == a.len() as u64,
                        "activation summary population differs from its matrix",
                    )?;
                }
                if let Some(key) = p.selected_neurons.get(name) {
                    let ids = get(key);
                    ensure(
                        ids.dtype() == DType::U64
                            && ids.shape() == [a.shape()[1]]
                            && unique(ids.as_u64()?),
                        "invalid selected-neuron identities",
                    )?;
                    let width = p.source_widths.get(name).copied().unwrap_or(0);
                    ensure(
                        ids.as_u64()?.iter().all(|&i| i < width as u64),
                        "selected neuron exceeds source width",
                    )?;
                } else if let Some(width) = p.source_widths.get(name) {
                    ensure(*width == a.shape()[1], "activation source width differs")?;
                }
            }
            ensure(
                p.selected_neurons.keys().all(|k| p.layers.contains_key(k)),
                "selected neurons lack an activation matrix",
            )?;
            if let ActivationInputs::ObservationsAndActions {
                actions: key,
                source,
            } = &p.inputs
            {
                actions(
                    get(key),
                    rows.ok_or_else(|| {
                        crate::Error::Config("action-input activation lacks rows".into())
                    })?,
                )?;
                ensure(
                    matches!(get(key).dtype(), DType::F32 | DType::F64 | DType::U64),
                    "activation inputs require floating/u64 actions",
                )?;
                if let ActivationActionSource::Policy {
                    identity, model, ..
                } = source
                {
                    ensure(
                        !model.trim().is_empty()
                            && !identity.run_id.trim().is_empty()
                            && !identity.model_role.trim().is_empty(),
                        "empty activation policy input identity",
                    )?;
                }
            }
            for s in p.summaries.values() {
                ensure(
                    s.finite_count <= s.count
                        && s.zero_fraction.is_finite()
                        && (0. ..=1.).contains(&s.zero_fraction)
                        && [s.mean, s.standard_deviation, s.minimum, s.maximum]
                            .into_iter()
                            .flatten()
                            .all(f64::is_finite)
                        && s.standard_deviation.is_none_or(|v| v >= 0.),
                    "invalid activation summary",
                )?;
            }
        }
        Record::Empirical(p) => {
            evaluation(&p.evaluation)?;
            if matches!(
                evaluation_horizon(c)?,
                EvaluationHorizon::FiniteDecisions(_)
            ) {
                finite_initial_evaluation(&p.evaluation)?;
                ensure(
                    p.probe_row.is_none()
                        && p.initial_action.is_none()
                        && p.conditioning_observation.is_none()
                        && p.conditioning_mask.is_none()
                        && p.simulator_state.is_none(),
                    "finite decision evaluation cannot be used as conditional critic evidence",
                )?;
            }
            let n = scalar(get(&p.returns))?;
            ensure(finite(get(&p.returns)), "nonfinite empirical return")?;
            let seeds = get(&p.seeds);
            ensure(
                seeds.dtype() == DType::U64
                    && seeds.shape() == [n]
                    && (!p.evaluation.independent_future_streams || unique(seeds.as_u64()?)),
                "empirical seeds do not describe independent samples",
            )?;
            if let Some(k) = &p.initial_action {
                actions(get(k), 1)?;
            }
            if let Some(k) = &p.conditioning_observation {
                ensure(
                    matrix(get(k))? == 1 && finite(get(k)),
                    "invalid conditioning observation",
                )?;
            }
            if let Some(k) = &p.conditioning_mask {
                mask(get(k), 1, true)?;
            }
            if let Some(k) = &p.simulator_state {
                ensure(
                    floating(get(k)) && positive(get(k)) && finite(get(k)),
                    "invalid full simulator state",
                )?;
            }
        }
        Record::Trajectory(p) => {
            super::render::validate_track(c)?;
            evaluation(&p.evaluation)?;
            trajectory_scenario_seed(c)?;
            let n = scalar(get(&p.rewards))?;
            ensure(
                finite(get(&p.rewards))
                    && matrix(get(&p.observations))? == n + 1
                    && finite(get(&p.observations)),
                "trajectory observations must have T+1 finite rows",
            )?;
            match (
                c.arrays.get(PHYSICAL_DURATIONS_KEY),
                c.arrays.get(DISCOUNTED_REWARDS_KEY),
            ) {
                (Some(durations), Some(discounted)) => {
                    ensure(
                        durations.shape() == [n]
                            && discounted.shape() == [n]
                            && discounted.as_f64()?.iter().all(|r| r.is_finite()),
                        "invalid physical trajectory reward shape or values",
                    )?;
                    physical_durations(durations.as_u64()?.iter().copied())?;
                }
                (None, None) => {}
                _ => {
                    return Err(crate::Error::Shape(
                        "incomplete physical trajectory rewards".into(),
                    ));
                }
            }
            actions(get(&p.actions), n)?;
            for key in [&p.terminated, &p.truncated] {
                let a = get(key);
                ensure(
                    a.dtype() == DType::U8
                        && a.shape() == [n]
                        && a.as_u8()?.iter().all(|&b| b <= 1),
                    "invalid trajectory boundary flags",
                )?;
            }
            let terminated = get(&p.terminated).as_u8()?;
            let truncated = get(&p.truncated).as_u8()?;
            ensure(
                terminated
                    .iter()
                    .zip(truncated)
                    .enumerate()
                    .all(|(i, (&a, &b))| a + b <= 1 && (i + 1 == n || a + b == 0)),
                "trajectory crosses an episode boundary or has contradictory flags",
            )?;
            let horizon = evaluation_horizon(c)?;
            let complete = horizon.completes(n, terminated[n - 1] == 1);
            if let EvaluationHorizon::FiniteDecisions(limit) = horizon {
                finite_initial_evaluation(&p.evaluation)?;
                ensure(
                    n <= limit && p.evaluation.complete_returns == complete,
                    "finite trajectory does not match its declared decision objective",
                )?;
            } else {
                ensure(
                    !p.evaluation.complete_returns || complete,
                    "complete episodic trajectory must end in true termination",
                )?;
            }
            if let Some(k) = &p.action_masks {
                mask(get(k), n + 1, false)?;
            }
            if let Some(k) = &p.simulation_times {
                let a = get(k);
                ensure(
                    scalar(a)? == n + 1 && finite(a) && a.to_f64().windows(2).all(|w| w[0] <= w[1]),
                    "invalid/nonmonotone trajectory clock",
                )?;
            }
            for k in p.frames.values() {
                ensure(
                    get(k).shape()[0] == n + 1,
                    "environment frame count differs from trajectory",
                )?;
            }
        }
        Record::ModelSnapshot(p) => {
            ensure(
                !p.id.is_empty()
                    && !p.architecture.is_empty()
                    && !p.models.is_empty()
                    && c.context
                        .identity
                        .checkpoint_id
                        .as_ref()
                        .is_none_or(|id| *id == p.id),
                "invalid snapshot identity",
            )?;
            for m in p.models.values() {
                let a = get(&m.bytes);
                ensure(
                    a.dtype() == DType::U8
                        && a.shape().len() == 1
                        && positive(a)
                        && !m.signature.is_empty(),
                    "invalid model artifact bytes/signature",
                )?;
            }
        }
    }
    super::reference::declaration(c)?;
    super::value::return_target(c)?;
    Ok(())
}
