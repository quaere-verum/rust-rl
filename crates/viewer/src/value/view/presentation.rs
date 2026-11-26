//! Concise provenance shared by the plot and Selected tooltips.
use super::*;

pub(super) fn action(e: EnvironmentSemantics<'_>, action: Option<&Action>) -> String {
    match action {
        Some(Action::Continuous(values)) => values
            .iter()
            .enumerate()
            .map(|(i, v)| format!("{} = {v}", coordinate_label(e, ValueCoordinate::Action(i))))
            .collect::<Vec<_>>()
            .join(", "),
        Some(Action::Discrete(i)) => format!(
            "{} (index {i})",
            e.action(*i)
                .map_or_else(|| format!("action[{i}]"), |a| a.label.into_owned())
        ),
        Some(Action::Conditional(_)) => "Conditional action unavailable".into(),
        None => "No first action (state value)".into(),
    }
}
pub(super) fn value_source(
    label: &str,
    source: &ValueSource,
    requested: Option<u64>,
    first: Option<&Action>,
) -> String {
    let identity = source.identity();
    let requested = requested.map_or_else(|| "unknown".into(), |k| k.to_string());
    let continuation = source
        .critic_names()
        .iter()
        .map(|name| {
            let c = source
                .declaration()
                .and_then(|d| d.critics.get(name))
                .and_then(|c| c.continuation.as_ref());
            c.map_or_else(
                || format!("{name}: unknown"),
                |c| {
                    format!(
                        "{name}: {} ({})",
                        c.identity,
                        if c.stochastic {
                            "stochastic"
                        } else {
                            "deterministic"
                        }
                    )
                },
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "{label}: {} · {} · actual k={} / requested k={requested}\nRun {} · record #{}\nContinuation {continuation}\n{label} first action: {}",
        source.critic_names().join(", "),
        source.model_role(),
        identity.revision,
        source.manifest().run_id,
        identity.key.sequence,
        action(source.semantics(), first)
    )
}
pub(super) fn comparison(condition: ComparisonCondition) -> &'static str {
    match condition {
        ComparisonCondition::SameInput => "Same first input",
        ComparisonCondition::ControllerValues => "Each controller's first action",
    }
}
pub(super) fn selected_input(frame: &PreparedValueFrame, point: &PreparedValuePoint) -> String {
    let mut text = format!(
        "Complete observation {:?}\nMask {:?}\n{}\n{}",
        point.input.dense_values(),
        point.input.dense_mask(),
        comparison(frame.grid.comparison_condition()),
        value_source(
            "Current",
            point.current(),
            Some(frame.requested_revision),
            point.action.as_ref()
        )
    );
    if point.reference_value.is_some()
        && let Some(source) = frame.grid.reference()
    {
        text.push('\n');
        text.push_str(&value_source(
            "Reference",
            source,
            frame.reference_requested_revision,
            point.reference_action.as_ref(),
        ));
    }
    text
}
pub(super) fn independent_controller_samples(g: &PreparedValueGrid) -> bool {
    g.current().kind() == ValueCriticKind::DiscreteQ
        && g.slice.state_only()
        && matches!(g.slice.conditioning, ActionConditioning::Policy)
}
pub(super) fn reference_series_label(g: &PreparedValueGrid) -> &'static str {
    if g.empirical_reference() {
        "Empirical conditional means · measured inputs only"
    } else if independent_controller_samples(g) {
        "Reference controller samples · independent inputs"
    } else {
        "Reference samples · independent inputs"
    }
}
pub(super) fn projected(slice: &ValueSlice, actor: &Action, conditioned: Option<&Action>) -> bool {
    let (Action::Continuous(actor), Some(Action::Continuous(conditioned))) = (actor, conditioned)
    else {
        return false;
    };
    actor
        .iter()
        .zip(conditioned)
        .enumerate()
        .any(|(i, (a, b))| {
            !std::iter::once(slice.x)
                .chain(slice.y)
                .any(|c| c == ValueCoordinate::Action(i))
                && a != b
        })
}
pub(super) fn overlay_qualifiers(
    slice: &ValueSlice,
    actor: &Action,
    conditioned: Option<&Action>,
    stochastic: bool,
) -> String {
    let mut text = String::new();
    if projected(slice, actor, conditioned) {
        text.push_str("\nPolicy component projection; fixed remaining action components differ");
    }
    if stochastic {
        text.push_str("\nStochastic representative action");
    }
    text
}
