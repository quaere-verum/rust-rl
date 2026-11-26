//! Supported built-in output mappings, prepared off the UI thread.
use rl::{core::action::ActionSpace, diagnostics::models::ModelKind};

const MAX_COMPONENTS: usize = 64;
#[derive(Clone, Debug)]
pub struct OutputComparison {
    pub label: String,
    pub anchor: Vec<f64>,
    pub point: Vec<f64>,
    pub components: usize,
    pub note: String,
}
impl OutputComparison {
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.label.capacity()
            + self.note.capacity()
            + (self.anchor.capacity() + self.point.capacity()) * 8
    }
}

pub(super) fn prepare(
    kind: &ModelKind,
    network: &str,
    space: &ActionSpace,
    mask: Option<&[bool]>,
    anchor: &[f64],
    point: &[f64],
) -> std::result::Result<OutputComparison, String> {
    if anchor.is_empty()
        || anchor.len() != point.len()
        || anchor.iter().chain(point).any(|v| !v.is_finite())
    {
        return Err("Nonfinite or inconsistent actual output".into());
    }
    if let ActionSpace::Discrete { count } = space
        && mask.is_some_and(|m| m.len() != *count || !m.iter().any(|v| *v))
    {
        return Err("The recorded A action mask is invalid".into());
    }
    let map = |values: &[f64]| -> std::result::Result<Vec<f64>, String> {
        match kind {
            ModelKind::StochasticPolicy if matches!(space, ActionSpace::Discrete { .. }) => {
                let ActionSpace::Discrete { count } = space else {
                    unreachable!()
                };
                if values.len() != *count {
                    return Err("Policy probability width mismatch".into());
                }
                let shift = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask.is_none_or(|m| m[*i]))
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask.is_none_or(|m| m[*i]))
                    .map(|(_, v)| (v - shift).exp())
                    .sum();
                Ok(values
                    .iter()
                    .enumerate()
                    .take(MAX_COMPONENTS)
                    .map(|(i, v)| {
                        if mask.is_none_or(|m| m[i]) {
                            (v - shift).exp() / z
                        } else {
                            0.
                        }
                    })
                    .collect())
            }
            ModelKind::StateGaussianPolicy
            | ModelKind::StochasticPolicy
            | ModelKind::DeterministicPolicy => {
                let dimension = space.dimensions();
                let outputs = if matches!(kind, ModelKind::StateGaussianPolicy) {
                    2 * dimension
                } else {
                    dimension
                };
                if values.len() != outputs {
                    return Err("Policy action width mismatch".into());
                }
                Ok(values[..dimension]
                    .iter()
                    .enumerate()
                    .take(MAX_COMPONENTS)
                    .map(|(i, &v)| {
                        if let ActionSpace::Box { low, high } = space {
                            let (low, high) = (f64::from(low[i]), f64::from(high[i]));
                            v.tanh() * ((high - low) / 2.) + (high + low) / 2.
                        } else {
                            v
                        }
                    })
                    .collect())
            }
            ModelKind::QuantileQ { grid } => {
                if values.len() != grid.len() {
                    return Err("Quantile output width mismatch".into());
                }
                let forecast = rl::diagnostics::analysis::QuantileForecast::ordered(
                    values.to_vec(),
                    grid.locations().into_iter().map(f64::from).collect(),
                    grid.masses().into_iter().map(f64::from).collect(),
                )
                .map_err(|e| e.to_string())?;
                Ok(vec![forecast.mean()])
            }
            ModelKind::DiscreteQ => {
                let index = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask.is_none_or(|m| m.get(*i).copied().unwrap_or(false)))
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .ok_or("No admissible greedy action")?;
                Ok(vec![index as f64])
            }
            ModelKind::CategoricalQ { .. }
            | ModelKind::CategoricalReturnQ { .. }
            | ModelKind::ResidualQ { .. }
            | ModelKind::Opaque { .. } => Err("No supported dense output mapping".into()),
            _ => Ok(values.iter().copied().take(MAX_COMPONENTS).collect()),
        }
    };
    let (label, note, components) = match kind {
        ModelKind::StochasticPolicy if matches!(space, ActionSpace::Discrete { .. }) => (
            "Action probabilities",
            "The recorded A mask is held fixed at x.",
            anchor.len(),
        ),
        ModelKind::StochasticPolicy if matches!(space, ActionSpace::Box { .. }) => (
            "Deterministic location action",
            "Declared tanh/Box transform of latent mean; not the expected action under policy noise.",
            anchor.len(),
        ),
        ModelKind::StochasticPolicy => (
            "Policy mean action",
            "Unbounded Gaussian action mean; noise scale is separate.",
            anchor.len(),
        ),
        ModelKind::StateGaussianPolicy => (
            "Deterministic location action",
            "Uses the latent mean and the declared action transform. Learned log-scale heads are available with raw layer outputs.",
            space.dimensions(),
        ),
        ModelKind::DeterministicPolicy => (
            "Action",
            "Uses the built-in policy's declared action transform.",
            anchor.len(),
        ),
        ModelKind::DiscreteQ => (
            "Greedy action ID",
            "Greedy Q choice with the recorded A mask held fixed at x.",
            1,
        ),
        ModelKind::QuantileQ { .. } => (
            "Ordered forecast expected return",
            "Uses ordered quantiles with masses kept in probability order; raw propagation heads remain unchanged.",
            1,
        ),
        ModelKind::MomentQ { .. } if network.ends_with(".second") => (
            "Raw second moment E[G²]",
            "Not return variance.",
            anchor.len(),
        ),
        ModelKind::MomentQ { .. } => ("Expected return E[G]", "First-moment critic.", anchor.len()),
        ModelKind::StateValue => ("V(s)", "State-value prediction.", anchor.len()),
        ModelKind::ScalarQ => (
            "Q(s,a)",
            "Conditioned on the input action coordinates at A and x.",
            anchor.len(),
        ),
        ModelKind::CategoricalQ { .. }
        | ModelKind::CategoricalReturnQ { .. }
        | ModelKind::ResidualQ { .. }
        | ModelKind::Opaque { .. } => {
            return Err("No supported dense output mapping".into());
        }
    };
    let anchor = map(anchor)?;
    let point = map(point)?;
    if anchor.iter().chain(&point).any(|v| !v.is_finite()) {
        return Err("Nonfinite actual output mapping".into());
    }
    Ok(OutputComparison {
        label: label.into(),
        anchor,
        point,
        components,
        note: note.into(),
    })
}

// These expectations are hand checked against the public policy contracts:
// masked softmax, tanh followed by Box affine bounds, and quantile masses.
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_output_is_masked_probability_or_actual_transformed_action() {
        let p = prepare(
            &ModelKind::StochasticPolicy,
            "policy",
            &ActionSpace::Discrete { count: 3 },
            Some(&[true, false, true]),
            &[0., 100., 0.],
            &[0., 100., 3_f64.ln()],
        )
        .unwrap();
        assert_eq!(p.anchor, [0.5, 0., 0.5]);
        assert!((p.point[2] - 0.75).abs() < 1e-12);
        let space = ActionSpace::Box {
            low: vec![-2., 10.],
            high: vec![2., 14.],
        };
        let a = prepare(
            &ModelKind::DeterministicPolicy,
            "actor",
            &space,
            None,
            &[0., 0.],
            &[0.5_f64.atanh(), -0.5_f64.atanh()],
        )
        .unwrap();
        assert_eq!(a.anchor, [0., 12.]);
        assert_eq!(a.point, [1., 11.]);
        let s = prepare(
            &ModelKind::StochasticPolicy,
            "policy",
            &space,
            None,
            &[0., 0.],
            &[0., 0.],
        )
        .unwrap();
        assert!(s.label.contains("location"));
        assert!(s.note.contains("not the expected action"));
    }
    #[test]
    fn actual_output_rejects_unknown_masks_and_nonfinite_values() {
        let kind = ModelKind::StochasticPolicy;
        let actions = ActionSpace::Discrete { count: 2 };
        assert!(
            prepare(
                &kind,
                "policy",
                &actions,
                Some(&[false, false]),
                &[0., 0.],
                &[0., 1.]
            )
            .is_err()
        );
        assert!(
            prepare(
                &kind,
                "policy",
                &actions,
                Some(&[true]),
                &[0., 0.],
                &[0., 1.]
            )
            .is_err()
        );
        assert!(prepare(&kind, "policy", &actions, None, &[0., 0.], &[f64::NAN, 1.]).is_err());
    }
    #[test]
    fn critic_output_uses_quantile_masses_and_keeps_raw_second_moment() {
        let kind = ModelKind::QuantileQ {
            grid: rl::nn::critics::quantile::QuantileGrid {
                edges: vec![0., 0.25, 1.],
            },
        };
        let actions = ActionSpace::Unbounded { dimensions: 1 };
        let q = prepare(&kind, "critic", &actions, None, &[0., 4.], &[4., 8.]).unwrap();
        assert_eq!(q.anchor, [3.]);
        assert_eq!(q.point, [7.]);
        let kind = ModelKind::MomentQ {
            second_key: "moment.second".into(),
        };
        let m = prepare(&kind, "moment.second", &actions, None, &[9.], &[16.]).unwrap();
        assert_eq!(m.point, [16.]);
        assert_eq!(m.label, "Raw second moment E[G²]");
        assert!(m.note.contains("Not return variance"));
    }
    #[test]
    fn quantile_output_orders_crossing_heads_without_reassigning_probability_masses() {
        let kind = ModelKind::QuantileQ {
            grid: rl::nn::critics::quantile::QuantileGrid {
                edges: vec![0., 0.25, 1.],
            },
        };
        let output = prepare(
            &kind,
            "critic",
            &ActionSpace::Unbounded { dimensions: 1 },
            None,
            &[4., 0.],
            &[0., 4.],
        )
        .unwrap();
        assert_eq!(
            output.anchor,
            [3.],
            "Expected return must use the same ordered forecast as Value and D4PG's actor boundary"
        );
        assert_eq!(output.point, [3.]);
        assert!(output.note.contains("ordered"));
    }
}
