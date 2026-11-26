//! Report matched recorded quantile and conditional-return evidence.
//! Usage: cargo run -p rl-viewer --example conditional_calibration -- RECORDING
use rl::diagnostics::{
    analysis::{EmpiricalDistribution, match_conditional_rows},
    schema::{Record, RecordKind},
    storage::ReadLimits,
    value::ValuePrediction,
};
use rl_viewer::data::RunIndex;

fn main() -> rl::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| rl::Error::Config("recording path required".into()))?;
    let run = RunIndex::open(path, ReadLimits::default())?;
    let recording = run.recording();
    let entries = recording.entries();
    println!(
        "revision,row,futures,predicted_mean,predicted_sd,empirical_mean,empirical_sd,empirical_mean_se,continuation"
    );
    let mut matched_count = 0;
    for (index, entry) in entries.iter().enumerate().filter(|(_, entry)| {
        entry.kind == RecordKind::Empirical
            && entry.model_role == "online"
            && entry.probe_set.as_deref() == Some("conditional-alpha")
    }) {
        let empirical = recording.load(index)?;
        let Record::Empirical(e) = &empirical.record else {
            unreachable!()
        };
        let Some(row) = e
            .probe_row
            .map(usize::try_from)
            .transpose()
            .map_err(|_| rl::Error::Shape("probe row overflow".into()))?
        else {
            continue;
        };
        if rl::diagnostics::empirical::declaration(&empirical)?.is_none() {
            return Err(rl::Error::State(
                "conditional evidence lacks its scientific declaration".into(),
            ));
        }
        let empirical_probe = run
            .probe_for_record(index)
            .ok_or_else(|| rl::Error::State("empirical probe missing".into()))?;
        let probes = recording.load(empirical_probe)?;
        let mut matched = None;
        for (ci, _) in entries.iter().enumerate().filter(|(_, candidate)| {
            candidate.kind == RecordKind::Critic
                && candidate.revision == entry.revision
                && candidate.model_role == entry.model_role
                && candidate.probe_set == entry.probe_set
        }) {
            let critic = recording.load(ci)?;
            let Some(critic_probe) = run.probe_for_record(ci) else {
                continue;
            };
            let critic_probes = recording.load(critic_probe)?;
            if match_conditional_rows(
                &critic,
                &critic_probes,
                row,
                &empirical,
                &probes,
                recording.manifest(),
            )
            .is_ok()
                && let Ok(ValuePrediction::Quantiles {
                    raw,
                    locations,
                    masses,
                }) = ValuePrediction::from_critic(&critic, row, None)
            {
                let forecast =
                    rl::diagnostics::analysis::QuantileForecast::ordered(raw, locations, masses)?;
                let mean = forecast.mean();
                let variance: f64 = forecast
                    .values()
                    .iter()
                    .zip(forecast.masses())
                    .map(|(v, p)| p * (v - mean).powi(2))
                    .sum();
                matched = Some((mean, variance.sqrt()));
                break;
            }
        }
        let Some((mean, sd)) = matched else {
            return Err(rl::Error::State(format!(
                "no compatible critic for revision {} row {row}",
                entry.revision
            )));
        };
        let summary =
            EmpiricalDistribution::new(empirical.arrays[&e.returns].as_f64()?)?.summary()?;
        let variance = summary.sample_variance.ok_or_else(|| {
            rl::Error::State("at least two independent futures are required".into())
        })?;
        let mean_se = summary
            .mean_standard_error
            .ok_or_else(|| rl::Error::State("empirical mean uncertainty unavailable".into()))?;
        println!(
            "{},{row},{},{mean:.6},{sd:.6},{:.6},{:.6},{mean_se:.6},\"{}\"",
            entry.revision,
            summary.count,
            summary.mean,
            variance.sqrt(),
            e.evaluation.continuation_policy.replace('"', "\"\"")
        );
        matched_count += 1;
    }
    if matched_count == 0 {
        return Err(rl::Error::State(
            "no conditional-alpha calibration evidence found".into(),
        ));
    }
    Ok(())
}
