//! Optional scientific provenance and reference-only, exact finite input lookup.
//! Numeric arrays stay with the resolver; the index owns only sorted row IDs.
use crate::policy::prepared::{Accounting, Lease};
use eframe::egui;
use rl::{
    Error, Result,
    core::env::Observation,
    diagnostics::{
        reference::*,
        schema::{Chunk, Record, RunManifest},
    },
};
use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

pub(crate) const MAX_INDEX_BYTES: usize = 512 * 1024;

pub(crate) fn provider_role(role: &str) -> bool {
    role.strip_prefix("reference:")
        .is_some_and(|id| !id.is_empty())
}

pub(crate) struct ReferenceInfo {
    declaration: Option<ReferenceDeclaration>,
    provider: Option<ReferenceSpec>,
    anchor: Option<ReferenceContext>,
    index: Mutex<Option<Arc<ExactIndex>>>,
    bytes: usize,
}
impl ReferenceInfo {
    pub(crate) fn new(manifest: &RunManifest, chunk: Option<&Chunk>) -> Result<Arc<Self>> {
        let declaration = chunk.map(declaration).transpose()?.flatten();
        if let Some(d) = &declaration {
            d.validate_manifest(manifest)?;
        }
        let provider = chunk.map(provider_specification).transpose()?.flatten();
        if let Some((d, p)) = declaration.as_ref().zip(provider.as_ref()) {
            compatible_contexts(&d.context, &p.context)?;
        }
        let anchor = if declaration.is_none() && provider.is_none() {
            manifest_context(manifest)?
        } else {
            None
        };
        let bytes = std::mem::size_of::<Self>()
            + 64
            + declaration
                .as_ref()
                .map_or(0, ReferenceDeclaration::retained_charge)
            + provider
                .as_ref()
                .map(|p| -> Result<usize> {
                    Ok(serde_json::to_vec(p)?.len() * 8
                        + p.context.environment_config.len() * 192
                        + 4096)
                })
                .transpose()?
                .unwrap_or(0)
            + anchor
                .as_ref()
                .map(|c| -> Result<usize> {
                    Ok(serde_json::to_vec(c)?.len() * 8 + c.environment_config.len() * 192 + 2048)
                })
                .transpose()?
                .unwrap_or(0);
        Ok(Arc::new(Self {
            declaration,
            provider,
            anchor,
            index: Mutex::new(None),
            bytes,
        }))
    }
    pub(crate) fn declaration(&self) -> Option<&ReferenceDeclaration> {
        self.declaration.as_ref()
    }
    pub(crate) fn prepared(&self) -> bool {
        self.index.try_lock().is_ok_and(|i| i.is_some())
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn context(&self) -> Option<&ReferenceContext> {
        self.declaration
            .as_ref()
            .map(|d| &d.context)
            .or_else(|| self.provider.as_ref().map(|p| &p.context))
            .or(self.anchor.as_ref())
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        self.declaration == other.declaration
            && self.provider == other.provider
            && self.context() == other.context()
    }
    pub(crate) fn finite(&self) -> bool {
        self.declaration
            .as_ref()
            .is_some_and(|d| d.finite_table.is_some())
    }
    pub(crate) fn compatible(&self, other: &Self) -> Result<()> {
        if self
            .provider
            .iter()
            .chain(&other.provider)
            .any(|p| p.information == ReferenceInformation::PrivilegedSimulatorState)
        {
            return Err(Error::Config(
                "This reference uses privileged simulator information".into(),
            ));
        }
        match (self.context(), other.context()) {
            (Some(a), Some(b)) => compatible_contexts(a, b),
            _ if self.provider.is_some() || other.provider.is_some() => Err(Error::Config(
                "Recorded provider requires matching authored scientific contexts".into(),
            )),
            _ if self.finite() || other.finite() => Err(Error::Config(
                "Exact reference requires matching authored scientific contexts".into(),
            )),
            _ => Ok(()), // Preserve ordinary legacy comparisons, without claiming optimality.
        }
    }
    pub(crate) fn prepare(
        &self,
        table: &Chunk,
        probes: &Chunk,
        accounting: &Arc<Accounting>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        if !self.finite() {
            return Ok(());
        }
        if cancelled() {
            return Err(Error::State("Reference index request superseded".into()));
        }
        let mut cached = self
            .index
            .lock()
            .map_err(|_| Error::State("Reference index lock poisoned".into()))?;
        if cached.is_none() {
            validate_table(table, probes)?;
            let index = ExactIndex::build(probes, accounting, cancelled)?;
            if cancelled() {
                return Err(Error::State("Reference index request superseded".into()));
            }
            *cached = Some(Arc::new(index));
        }
        Ok(())
    }
    pub(crate) fn lookup(&self, probes: &Chunk, input: &Observation) -> Result<Option<usize>> {
        let cached = self
            .index
            .lock()
            .map_err(|_| Error::State("Reference index lock poisoned".into()))?;
        cached
            .as_ref()
            .ok_or_else(|| Error::State("Exact reference index is not prepared".into()))?
            .lookup(probes, input)
    }
    pub(crate) fn ui(&self, ui: &mut egui::Ui, name: &str, recorded: bool, _capture: bool) {
        let quality = self.declaration.as_ref().and_then(|d| {
            d.sources.get(name).or_else(|| {
                (recorded && d.sources.len() == 1)
                    .then(|| d.sources.values().next())
                    .flatten()
            })
        });
        let quality = self.provider.as_ref().map(|p| &p.quality).or(quality);
        let quality_text = quality_label(quality.map(|q| q.kind));
        let response = if self
            .provider
            .as_ref()
            .is_some_and(|p| p.information == ReferenceInformation::PrivilegedSimulatorState)
        {
            ui.small(format!("{quality_text} · Privileged simulator state"))
        } else {
            ui.small(quality_text)
        };
        if let (Some(q), Some(c)) = (quality, self.context()) {
            let show = |ui: &mut egui::Ui| {
                ui.set_max_width(480.);
                egui::ScrollArea::vertical().max_height(360.).show(ui,|ui| {
                    if let Some(p) = &self.provider {
                        ui.label(format!("{} · {}", p.id, information_label(p.information)));
                        ui.label(match p.target {
                            ReferenceTarget::BenchmarkPolicy => "Benchmark policy",
                            ReferenceTarget::VerifiedOptimal => "Verified optimal target",
                            ReferenceTarget::LearnerPolicy => "Evaluation of the learner's policy",
                            ReferenceTarget::CriticContinuation => "Evaluation of the critic's continuation policy",
                        });
                        if let Some(uncertainty) = &p.uncertainty {
                            ui.label(uncertainty);
                        }
                    }
                    ui.label(format!("Method: {}",q.method));
                    ui.label("Objective: expected discounted return (larger is better)");
                    ui.label(format!("Horizon: {:?}; discount: {}",c.horizon,c.conventions.discount));
                    ui.label(format!("Environment: {}",c.environment));
                    ui.label(format!("Assumptions: {}",q.assumptions.join("; ")));
                    if self.finite() { ui.label("Exact complete state, decision time and legal mask only. Action-value entries are Q*(s,a); V*(s) is the maximum legal Q, independently of policy tie breaking."); }
                    if let Ok(config) = serde_json::to_string_pretty(c) { ui.add(egui::Label::new(egui::RichText::new(config).monospace()).wrap()); }
                });
            };
            #[cfg(feature = "screenshots")]
            if _capture && std::env::var("RL_VIEWER_REFERENCE_HOVER").as_deref() == Ok("true") {
                egui::Tooltip::always_open(
                    ui.ctx().clone(),
                    response.layer_id,
                    response.id.with("reference-quality"),
                    egui::PopupAnchor::from(&response),
                )
                .show(show);
                return;
            }
            response.on_hover_ui(show);
        }
    }
}
pub(crate) fn information_label(information: ReferenceInformation) -> &'static str {
    match information {
        ReferenceInformation::LearnerObservation => "Learner observations",
        ReferenceInformation::PrivilegedSimulatorState => "Privileged simulator state",
    }
}
pub(crate) fn quality_label(kind: Option<ReferenceKind>) -> &'static str {
    match kind {
        None => "Unknown reference quality",
        Some(ReferenceKind::Learned) => "Learned reference",
        Some(ReferenceKind::Heuristic) => "Heuristic benchmark",
        Some(ReferenceKind::Analytical) => "Analytical benchmark",
        Some(ReferenceKind::ApproximateNumerical) => "Approximate numerical benchmark",
        Some(ReferenceKind::VerifiedOptimal) => "Verified optimal reference",
    }
}

struct Rows<'a> {
    values: &'a [f32],
    masks: &'a [u8],
    width: usize,
    actions: usize,
    count: usize,
}
impl<'a> Rows<'a> {
    fn new(probes: &'a Chunk) -> Result<Self> {
        let Record::ProbeSet(p) = &probes.record else {
            return Err(Error::Config("Exact table needs probes".into()));
        };
        let x = crate::plot_data::array(probes, &p.observations)?;
        let m = crate::plot_data::array(
            probes,
            p.action_masks
                .as_ref()
                .ok_or_else(|| Error::Config("Exact table needs explicit masks".into()))?,
        )?;
        if !p.histories.is_empty()
            || p.fixed_actions.is_some()
            || x.shape().len() != 2
            || m.shape().len() != 2
            || x.shape()[0] != m.shape()[0]
            || x.shape()[1] == 0
            || m.shape()[1] == 0
            || x.shape()[1] > 64
            || m.shape()[1] > 64
        {
            return Err(Error::Config(
                "Exact table input shape/conditioning invalid".into(),
            ));
        }
        Ok(Self {
            values: x.as_f32()?,
            masks: m.as_u8()?,
            width: x.shape()[1],
            actions: m.shape()[1],
            count: x.shape()[0],
        })
    }
    fn values(&self, row: usize) -> &[f32] {
        &self.values[row * self.width..(row + 1) * self.width]
    }
    fn masks(&self, row: usize) -> &[u8] {
        &self.masks[row * self.actions..(row + 1) * self.actions]
    }
    fn compare(&self, a: usize, b: usize) -> Ordering {
        self.values(a)
            .iter()
            .zip(self.values(b))
            .map(|(a, b)| a.partial_cmp(b).unwrap())
            .find(|c| *c != Ordering::Equal)
            .unwrap_or_else(|| self.masks(a).cmp(self.masks(b)))
    }
    fn compare_input(&self, row: usize, input: &Observation) -> Ordering {
        self.values(row)
            .iter()
            .zip(input.dense_values())
            .map(|(a, b)| a.partial_cmp(b).unwrap())
            .find(|c| *c != Ordering::Equal)
            .unwrap_or_else(|| {
                self.masks(row)
                    .iter()
                    .map(|v| *v != 0)
                    .cmp(input.dense_mask().unwrap().iter().copied())
            })
    }
}
pub(crate) struct ExactIndex {
    rows: Vec<u32>,
    _lease: Lease,
}
impl ExactIndex {
    fn build(
        probes: &Chunk,
        accounting: &Arc<Accounting>,
        cancelled: &impl Fn() -> bool,
    ) -> Result<Self> {
        let data = Rows::new(probes)?;
        if data.count == 0 || data.count > MAX_REFERENCE_TABLE_ROWS {
            return Err(Error::Config("Exact reference exceeds 65536 rows".into()));
        }
        let charge = data.count * 4 + std::mem::size_of::<Self>() + 64;
        if charge > MAX_INDEX_BYTES {
            return Err(Error::Config("Exact index exceeds 512 KiB".into()));
        }
        let lease = Lease::acquire(accounting, charge)?;
        accounting.scratch(data.count * 4 + 4096)?;
        let stop = || {
            if cancelled() {
                Err(Error::State("Reference index request superseded".into()))
            } else {
                Ok(())
            }
        };
        for row in 0..data.count {
            if row % 256 == 0 {
                stop()?;
            }
            if data.values(row).iter().any(|v| !v.is_finite())
                || data.masks(row).iter().any(|v| *v > 1)
                || !data.masks(row).contains(&1)
            {
                return Err(Error::Config(
                    "Exact reference has nonfinite or terminal input".into(),
                ));
            }
        }
        let mut rows = (0..data.count as u32).collect::<Vec<_>>();
        let mut scratch = vec![0u32; data.count];
        // Bottom-up merge sort bounds cancellation latency and scratch precisely.
        let mut width = 1;
        while width < data.count {
            for first in (0..data.count).step_by(width * 2) {
                let mid = (first + width).min(data.count);
                let end = (first + width * 2).min(data.count);
                let (mut a, mut b) = (first, mid);
                for (offset, out) in scratch[first..end].iter_mut().enumerate() {
                    if offset % 256 == 0 {
                        stop()?;
                    }
                    if a < mid
                        && (b == end
                            || data.compare(rows[a] as usize, rows[b] as usize)
                                != Ordering::Greater)
                    {
                        *out = rows[a];
                        a += 1;
                    } else {
                        *out = rows[b];
                        b += 1;
                    }
                }
            }
            std::mem::swap(&mut rows, &mut scratch);
            width *= 2;
        }
        for (n, pair) in rows.windows(2).enumerate() {
            if n % 256 == 0 {
                stop()?;
            }
            if data.compare(pair[0] as usize, pair[1] as usize) == Ordering::Equal {
                return Err(Error::Config(
                    "Duplicate exact reference input and mask".into(),
                ));
            }
        }
        stop()?;
        Ok(Self {
            rows,
            _lease: lease,
        })
    }
    #[cfg(test)]
    fn charge(&self) -> usize {
        self._lease.bytes
    }
    fn lookup(&self, probes: &Chunk, input: &Observation) -> Result<Option<usize>> {
        let data = Rows::new(probes)?;
        if input.dense_values().len() != data.width
            || input.dense_values().iter().any(|v| !v.is_finite())
            || input
                .dense_mask()
                .is_none_or(|m| m.len() != data.actions || !m.contains(&true))
        {
            return Ok(None);
        }
        Ok(self
            .rows
            .binary_search_by(|row| data.compare_input(*row as usize, input))
            .ok()
            .map(|at| self.rows[at] as usize))
    }
}
#[cfg(test)]
pub(crate) mod tests;
