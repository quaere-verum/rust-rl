//! Named built-in networks, shared by snapshot admission and the model selector.
use rl::core::action::ActionSpace;
use rl::diagnostics::models::{ModelKind, SnapshotConfiguration};

#[derive(Clone)]
pub struct NetworkChoice {
    pub name: String,
    pub label: String,
    pub kind: &'static str,
    pub model: String,
    pub input_width: usize,
    pub layers: usize,
}
pub fn choices(config: &SnapshotConfiguration) -> Vec<NetworkChoice> {
    let mut result = Vec::new();
    for (name, spec) in &config.models {
        let Some(layout) = &spec.network else {
            continue;
        };
        let (label, kind) = match &spec.kind {
            ModelKind::StochasticPolicy => (
                "Policy".into(),
                match config.action_space {
                    ActionSpace::Discrete { .. } => "Policy logits",
                    ActionSpace::Box { .. } => "Policy latent Gaussian mean (pre-tanh)",
                    ActionSpace::Unbounded { .. } => "Policy Gaussian mean action",
                    ActionSpace::Conditional(_) => continue,
                },
            ),
            ModelKind::StateGaussianPolicy => (
                "Gaussian policy".into(),
                "Policy latent mean and log-scale heads",
            ),
            ModelKind::DeterministicPolicy => (
                "Actor".into(),
                if matches!(config.action_space, ActionSpace::Box { .. }) {
                    "Pre-squash action"
                } else {
                    "Action"
                },
            ),
            ModelKind::StateValue => ("Value".into(), "V(s)"),
            ModelKind::DiscreteQ => ("Q".into(), "Discrete Q vector"),
            ModelKind::ScalarQ => ("Critic".into(), "Q(s,a)"),
            ModelKind::QuantileQ { .. } => ("Quantile critic".into(), "Quantile values"),
            ModelKind::MomentQ { .. } => ("Mean critic".into(), "E[G]"),
            ModelKind::CategoricalQ { .. }
            | ModelKind::CategoricalReturnQ { .. }
            | ModelKind::ResidualQ { .. }
            | ModelKind::Opaque { .. } => continue,
        };
        if matches!(&spec.kind, ModelKind::MomentQ { .. }) {
            result.push(NetworkChoice {
                name: format!("{name}.first"),
                label: format!("{name} · Mean critic"),
                kind,
                model: name.clone(),
                input_width: layout.widths[0],
                layers: layout.widths.len() - 1,
            });
            if let Some(second) = &spec.second {
                result.push(NetworkChoice {
                    name: format!("{name}.second"),
                    label: format!("{name} · Second-moment critic"),
                    kind: "E[G²] (raw second moment)",
                    model: name.clone(),
                    input_width: second.widths[0],
                    layers: second.widths.len() - 1,
                });
            }
        } else {
            result.push(NetworkChoice {
                name: name.clone(),
                label: if name == "policy" || name == "critic" || name == "value" {
                    label
                } else {
                    format!("{name} · {label}")
                },
                kind,
                model: name.clone(),
                input_width: layout.widths[0],
                layers: layout.widths.len() - 1,
            });
        }
    }
    result.sort_by_key(|choice| {
        let p = if choice.kind.starts_with("Policy")
            || choice.kind == "Pre-squash action"
            || choice.kind == "Action"
        {
            0
        } else if choice.kind == "V(s)" {
            1
        } else {
            2
        };
        (p, choice.label.clone())
    });
    result
}
