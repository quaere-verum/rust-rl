//! Graph-free recurrent discrete backups on factual public histories.
use crate::{
    Result,
    core::{action::Action, ensure},
    data::temporal::SequenceBatch,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TargetPoint {
    pub row: usize,
    pub timestep: usize,
    pub action: usize,
    pub value: f32,
    pub bootstrap_index: usize,
    pub discount: f32,
}

pub(in crate::algorithms) struct QLayout<'a> {
    pub(in crate::algorithms) time: usize,
    pub(in crate::algorithms) width: usize,
    pub(in crate::algorithms) offsets: &'a [usize],
}

/// Numeric, graph-free Double-Q target construction. Arrays are `[B,T,A]`.
/// Online values select an action; target values evaluate that same legal action.
pub fn construct_targets(
    batch: &SequenceBatch,
    online: &[f32],
    target: &[f32],
    actions: usize,
    gamma: f32,
) -> Result<Vec<TargetPoint>> {
    batch.validate()?;
    let time = batch
        .samples
        .iter()
        .map(|s| s.observations.len())
        .max()
        .expect("validated batch");
    let offsets = vec![0; batch.samples.len()];
    construct_targets_layout(
        batch,
        online,
        target,
        actions,
        gamma,
        QLayout {
            time,
            width: batch.samples[0].observations[0].dense_values().len(),
            offsets: &offsets,
        },
    )
}

/// One-step expected-policy targets under supplied frozen policy probabilities.
/// Arrays use the full padded `[batch,time,action]` history layout.
pub fn construct_policy_targets(
    batch: &SequenceBatch,
    probabilities: &[f32],
    target: &[f32],
    actions: usize,
    gamma: f32,
) -> Result<Vec<TargetPoint>> {
    batch.validate()?;
    ensure(
        batch.config.n_step == 1,
        "expected-policy targets require one step; multistep behavior rewards need off-policy correction",
    )?;
    ensure(
        gamma.is_finite() && (0. ..=1.).contains(&gamma),
        "invalid expected-policy discount",
    )?;
    let time = batch
        .samples
        .iter()
        .map(|s| s.observations.len())
        .max()
        .expect("validated batch");
    let size = batch
        .samples
        .len()
        .checked_mul(time)
        .and_then(|n| n.checked_mul(actions))
        .ok_or_else(|| crate::Error::Shape("policy target size overflow".into()))?;
    ensure(
        actions > 0
            && probabilities.len() == size
            && target.len() == size
            && probabilities.iter().all(|p| p.is_finite() && *p >= 0.)
            && target.iter().all(|q| q.is_finite()),
        "invalid expected-policy arrays",
    )?;
    let space = crate::core::action::ActionSpace::Discrete { count: actions };
    let mut points = Vec::new();
    for (row, sample) in batch.samples.iter().enumerate() {
        // Validate every real policy history, including burn-in. True terminal
        // observations have no action distribution; padded rows are unused zeros.
        for (t, obs) in sample.observations.iter().enumerate() {
            let terminal = t > 0 && sample.steps[t - 1].terminated;
            obs.validate(obs.dense_values().len(), &space, !terminal)?;
            let p = &probabilities[(row * time + t) * actions..(row * time + t + 1) * actions];
            if terminal {
                ensure(
                    p.iter().all(|&p| p == 0.),
                    "terminal policy row must be zero",
                )?;
            } else {
                ensure(
                    p.iter()
                        .enumerate()
                        .all(|(a, &p)| p == 0. || obs.dense_mask().is_none_or(|m| m[a])),
                    "target policy assigns probability to masked action",
                )?;
                let mass: f64 = p.iter().map(|&p| f64::from(p)).sum();
                ensure(
                    (mass - 1.).abs() <= 1e-6,
                    "target policy probabilities must sum to one",
                )?;
            }
        }
        for (t, &learn) in sample.loss_mask.iter().enumerate() {
            if !learn {
                continue;
            }
            let step = &sample.steps[t];
            space.check(&step.action, sample.observations[t].dense_mask())?;
            let Action::Discrete(action) = step.action else {
                unreachable!()
            };
            let n = sample.target_at(t, gamma, 1)?;
            let mut value = n.reward;
            if n.discount > 0. {
                let base = (row * time + n.bootstrap_index) * actions;
                let expected: f64 = probabilities[base..base + actions]
                    .iter()
                    .zip(&target[base..base + actions])
                    .map(|(&p, &q)| f64::from(p) * f64::from(q))
                    .sum();
                value += n.discount * expected as f32;
            }
            ensure(value.is_finite(), "nonfinite expected-policy target")?;
            points.push(TargetPoint {
                row,
                timestep: t,
                action,
                value,
                bootstrap_index: n.bootstrap_index,
                discount: n.discount,
            });
        }
    }
    ensure(
        !points.is_empty(),
        "no policy-evaluation learning positions",
    )?;
    Ok(points)
}

pub(in crate::algorithms) fn construct_targets_layout(
    batch: &SequenceBatch,
    online: &[f32],
    target: &[f32],
    actions: usize,
    gamma: f32,
    layout: QLayout<'_>,
) -> Result<Vec<TargetPoint>> {
    let size = batch
        .samples
        .len()
        .checked_mul(layout.time)
        .and_then(|n| n.checked_mul(actions))
        .ok_or_else(|| crate::Error::Shape("temporal Q size overflow".into()))?;
    ensure(
        actions > 0
            && online.len() == size
            && target.len() == size
            && online.iter().chain(target).all(|x| x.is_finite()),
        "invalid recurrent Q values",
    )?;
    let space = crate::core::action::ActionSpace::Discrete { count: actions };
    let mut points = Vec::new();
    for (row, sample) in batch.samples.iter().enumerate() {
        for (t, &learn) in sample.loss_mask.iter().enumerate() {
            if !learn {
                continue;
            }
            let step = &sample.steps[t];
            space.check(&step.action, sample.observations[t].dense_mask())?;
            let Action::Discrete(action) = step.action else {
                unreachable!()
            };
            let n = sample.target_at(t, gamma, batch.config.n_step)?;
            let mut value = n.reward;
            if n.discount > 0. {
                let obs = &sample.observations[n.bootstrap_index];
                obs.validate(layout.width, &space, true)?;
                let local = n
                    .bootstrap_index
                    .checked_sub(layout.offsets[row])
                    .ok_or_else(|| {
                        crate::Error::Shape("bootstrap precedes exact learning window".into())
                    })?;
                ensure(local < layout.time, "bootstrap exceeds learning lookahead")?;
                let base = (row * layout.time + local) * actions;
                let a = (0..actions)
                    .filter(|&a| obs.dense_mask().is_none_or(|m| m[a]))
                    .max_by(|&a, &b| online[base + a].total_cmp(&online[base + b]))
                    .ok_or_else(|| {
                        crate::Error::Action("all-invalid recurrent bootstrap mask".into())
                    })?;
                value += n.discount * target[base + a];
            }
            ensure(value.is_finite(), "nonfinite recurrent TD target")?;
            points.push(TargetPoint {
                row,
                timestep: t,
                action,
                value,
                bootstrap_index: n.bootstrap_index,
                discount: n.discount,
            });
        }
    }
    ensure(!points.is_empty(), "no recurrent learning positions")?;
    Ok(points)
}
