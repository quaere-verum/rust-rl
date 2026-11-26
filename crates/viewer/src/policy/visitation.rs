//! Real visits filtered to the same slice used for local-bin counts and markers.
//! Sources deliberately exclude fixed reference probes. Every admitted matching
//! row contributes to counts; markers deterministically sample that population.
use super::{
    ensure,
    slice::{SliceConfig, SliceMode, admit_population},
};
use rl::{Result, core::env::Observation, diagnostics::visualization::EnvironmentSemantics};
pub const MAX_VISIT_MARKERS: usize = 1024;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitationSource {
    EvaluationTrajectory,
    PolicyVisited,
}
#[derive(Clone, Copy)]
pub struct VisitationEvidence<'a> {
    pub source: VisitationSource,
    pub revision: u64,
    pub observations: &'a [Observation],
}
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedVisitation {
    pub source: VisitationSource,
    pub revision: u64,
    pub tolerance: f32,
    pub filtered_count: usize,
    /// Uniform local bins over the declared plot range, x fastest. The final
    /// bin includes the high endpoint. Outside-range visits are excluded.
    pub counts: Vec<u32>,
    /// `[x,y]` in actual model-input units; y=0 in a 1D slice.
    pub markers: Vec<[f32; 2]>,
}
pub fn prepare_visitation(
    environment: EnvironmentSemantics<'_>,
    config: &SliceConfig,
    evidence: Option<VisitationEvidence<'_>>,
    marker_limit: usize,
) -> Result<Option<PreparedVisitation>> {
    let Some(evidence) = evidence else {
        return Ok(None);
    };
    if evidence.observations.is_empty() {
        return Ok(None);
    }
    config.validate(environment)?;
    admit_population(config.fixed.len(), evidence.observations)?;
    ensure(
        marker_limit <= MAX_VISIT_MARKERS,
        "visitation marker count exceeds budget",
    )?;
    let two_dimensional = config.mode == SliceMode::TwoDimensional;
    let mut counts = vec![0; config.len()?];
    // Retain bounded row indices, not copies of complete wide observations.
    let mut matched = Vec::with_capacity(evidence.observations.len());
    for (index, observation) in evidence.observations.iter().enumerate() {
        observation.validate(config.fixed.len(), environment.action_space(), true)?;
        let Some(bin) = config.visit_bin(observation)? else {
            continue;
        };
        counts[bin] += 1;
        matched.push(index);
    }
    let marker_count = matched.len().min(marker_limit);
    let markers = (0..marker_count)
        .map(|i| {
            let row = &evidence.observations[matched[i * matched.len() / marker_count]];
            [
                row.dense_values()[config.x],
                if two_dimensional {
                    row.dense_values()[config.y.expect("validated y")]
                } else {
                    0.
                },
            ]
        })
        .collect();
    Ok(Some(PreparedVisitation {
        source: evidence.source,
        revision: evidence.revision,
        tolerance: config.tolerance,
        filtered_count: matched.len(),
        counts,
        markers,
    }))
}
