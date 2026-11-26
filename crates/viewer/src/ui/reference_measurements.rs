use super::*;
use rl::diagnostics::{reference::provider_specification, schema::Array};
use std::{collections::BTreeSet, sync::Weak};

const PREFIX: &str = "reference.";
const MAX_INDEX_CANDIDATES: usize = 32;
const MAX_RECORDS: usize = 8;
const MAX_RECORD_CHARGE: usize = 512 * 1024;
const MAX_DISPLAYED_METRICS: usize = 32;

/// These measurements were evaluated by the session at the requested learner
/// checkpoint. The separate comparison controls select historical overlays.
pub(super) fn show(
    ui: &mut egui::Ui,
    run: &RunIndex,
    selection: &Selection,
    loader: &mut Loader,
    pointwise: bool,
) {
    let entries = run.recording().entries();
    let selected_set = match &selection.input {
        SelectedInput::ProbeRow { set, .. } => Some(set.as_str()),
        _ => None,
    };
    let mut candidates: Vec<_> = run
        .candidates(
            RecordKind::Metrics,
            &selection.model_role,
            None,
            selection.revision,
        )
        .take_while(|&index| entries[index].revision == selection.revision)
        .take(MAX_INDEX_CANDIDATES)
        .collect();
    // Run-wide evaluation and the selected population remain visible when a
    // checkpoint contains several independent diagnostic populations.
    candidates.sort_by_key(|&index| match entries[index].probe_set.as_deref() {
        None => 0,
        Some(set) if selected_set == Some(set) => 1,
        Some(_) => 2,
    });
    let records: Vec<_> = candidates
        .into_iter()
        .take(MAX_RECORDS)
        .filter_map(|index| {
            let chunk = loader
                .request_admitted(run, index, MAX_RECORD_CHARGE)
                .ok()??;
            let Record::Metrics(metrics) = &chunk.record else {
                return None;
            };
            metrics
                .values
                .keys()
                .chain(metrics.unavailable.keys())
                .any(|key| {
                    key.strip_prefix(PREFIX)
                        .is_some_and(|name| !name.is_empty())
                })
                .then_some((index, chunk))
        })
        .collect();
    if records.is_empty() {
        return;
    }
    crate::landscape::popover(ui, "Recorded diagnostics", |ui| {
        ui.set_max_width(520.);
        ui.strong(format!("Reference diagnostics · k {}", selection.revision));
        if let Some(provider) = provider_caption(ui, &records[0].1) {
            ui.weak(provider);
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .id_salt("recorded-reference-diagnostics")
            .max_height(310.)
            .show(ui, |ui| {
                show_metrics(ui, run, selection, loader, pointwise, &records)
            });
    });
}

fn show_metrics(
    ui: &mut egui::Ui,
    run: &RunIndex,
    selection: &Selection,
    loader: &mut Loader,
    pointwise: bool,
    records: &[(usize, Arc<Chunk>)],
) {
    let entries = run.recording().entries();
    let mut displayed = BTreeSet::new();
    for (index, chunk) in records {
        let index = *index;
        let Record::Metrics(metrics) = &chunk.record else {
            continue;
        };
        let probe = run.probe_for_record(index);
        let population = entries[index].probe_set.as_ref();
        let scope = population.map(|_| entries[probe.unwrap_or(index)].sequence);
        let selected = if pointwise && !chunk.arrays.is_empty() {
            probe.and_then(|probe| selected_row(run, selection, loader, probe))
        } else {
            None
        };
        for (key, value) in metrics
            .values
            .iter()
            .map(|(key, value)| (key, Ok(*value)))
            .chain(
                metrics
                    .unavailable
                    .iter()
                    .map(|(key, reason)| (key, Err(reason))),
            )
        {
            let Some(name) = key.strip_prefix(PREFIX).filter(|name| !name.is_empty()) else {
                continue;
            };
            if displayed.len() == MAX_DISPLAYED_METRICS {
                return;
            }
            if !displayed.insert((scope, key.clone())) {
                continue;
            }
            let mut label = metric_label(name);
            if let Some(set) = population {
                label.push_str(&format!(" ({})", shortened(set, 40)));
            }
            match value {
                Ok(value) => {
                    let value = metric_number(value);
                    let array = chunk.arrays.get(key);
                    let input = selected
                        .zip(array)
                        .and_then(|((row, rows), array)| input_value(array, row, rows))
                        .map(metric_number);
                    let statistic = if array.is_some() { " mean" } else { "" };
                    let text = match input {
                        Some(input) => format!("{label}{statistic} {value} · Anchor row {input}"),
                        None => format!("{label}{statistic} {value}"),
                    };
                    ui.small(text);
                }
                Err(reason) => {
                    ui.weak(format!("{label} unavailable"))
                        .on_hover_text(shortened(reason, 512));
                }
            }
        }
    }
}

#[derive(Clone)]
struct ProviderCaption {
    source: Weak<Chunk>,
    text: Option<String>,
}

fn provider_caption(ui: &egui::Ui, chunk: &Arc<Chunk>) -> Option<String> {
    let id = ui.scope_id().with("provider-caption");
    let source = Arc::downgrade(chunk);
    if let Some(old) = ui.ctx().data(|data| data.get_temp::<ProviderCaption>(id))
        && old.source.ptr_eq(&source)
    {
        return old.text;
    }
    let text = match provider_specification(chunk) {
        Ok(Some(provider)) => Some(format!(
            "{} · {}",
            provider.id,
            crate::reference::information_label(provider.information)
        )),
        Ok(None) => None,
        Err(_) => Some("Reference provenance unavailable".into()),
    };
    ui.ctx().data_mut(|data| {
        data.insert_temp(
            id,
            ProviderCaption {
                source,
                text: text.clone(),
            },
        );
    });
    text
}

fn selected_row(
    run: &RunIndex,
    selection: &Selection,
    loader: &mut Loader,
    index: usize,
) -> Option<(usize, usize)> {
    let SelectedInput::ProbeRow { sequence, row, .. } = selection.input else {
        return None;
    };
    // The immutable definition includes the full observation and legal mask.
    // A row index from a different definition does not identify this input.
    if run.recording().entries()[index].sequence != sequence {
        return None;
    }
    let chunk = loader.request(run, index).ok()??;
    let Record::ProbeSet(p) = &chunk.record else {
        return None;
    };
    let rows = chunk.arrays.get(&p.row_ids)?.len();
    (row < rows).then_some((row, rows))
}

fn input_value(array: &Array, row: usize, rows: usize) -> Option<f64> {
    if array.shape() != [rows] {
        return None;
    }
    let value = array
        .as_f64()
        .map(|values| values[row])
        .or_else(|_| array.as_f32().map(|values| f64::from(values[row])))
        .ok()?;
    value.is_finite().then_some(value)
}

fn metric_label(name: &str) -> String {
    let name = shortened(name, 80).replace(['.', '_'], " ");
    let mut chars = name.chars();
    chars
        .next()
        .into_iter()
        .flat_map(char::to_uppercase)
        .chain(chars)
        .collect()
}

fn shortened(text: &str, max: usize) -> String {
    let mut chars = text.chars();
    let mut prefix = chars.by_ref().take(max).collect::<String>();
    if chars.next().is_some() {
        prefix.push('…');
    }
    prefix
}

fn metric_number(value: f64) -> String {
    if value != 0. && !(0.0001..100_000.).contains(&value.abs()) {
        format!("{value:.4e}")
    } else {
        format!("{value:.6}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .into()
    }
}
