//! First-class Network workspace. One actual network is inspected at a time.
use super::*;
use crate::network::{
    self,
    representations::{
        Query as RepresentationQuery, RepresentationFrame, Worker as RepresentationWorker,
    },
    selection::{NetworkChoice, choices},
    work::NetworkWorker,
};
use egui_plot::{Line as PlotLine, Plot as EguiPlot, PlotBounds, PlotPoint, Points as PlotPoints};
use nalgebra::{DMatrix, SymmetricEigen};
use rl::{
    core::action::ActionSpace,
    diagnostics::{
        models::{ModelKind, ModelLimits, SnapshotConfiguration, snapshot_configuration},
        visualization::{EnvironmentSemantics, ObservationType},
    },
};
use std::sync::Weak;

const NORMALIZATION_HELP: &str = "Hidden-layer sensitivities use a regularized inverse activation-covariance metric, estimated from a fixed reference population. Ridge = 0.001 × max(trace(covariance) / width, 1e-12). With fewer than three probes the metric is the identity. PCA bases are fitted once for this network and input population, then reused across checkpoints.";
const INTERPOLATION_HELP: &str = "A and B use recorded observations. Critic action coordinates use recorded actions where available, otherwise the Box midpoint or unbounded zero default. Representation paths map the straight segment A -> B through each layer. This assumes a convex admissible input domain; intermediate points are not guaranteed to be valid environment inputs. The fixed-reference PCA projection reports its reference variance and path-step energy fidelity per layer in Details and on each layer heading.";
const GEOMETRY_HELP: &str = "Restricted pullback metric in the selected slice. Long ellipse axes indicate near-kernel directions. Colour indicates log sensitivity, scaled within this field. The 3D view projects the local ellipsoid at A; yaw and pitch control its orientation.";
const FINITE_HELP: &str = "Solid cyan: actual Δ = h(A+t(B−A))−h(A). Dashed lavender: tangent L = t Dh(A)(B−A). Each independently scaled canonical plane preserves this pair's metric lengths and angle; planes are not shared neuron/PCA coordinates. The aligned bars share one metric-norm scale across every layer. E = ||Δ−L||/(||Δ||+||L||+1e-12), computed in the full original representation using the fixed-population covariance metric. A zero tangent has no distinguished direction; a ring replaces its directional comparison.";
const NONLINEARITY_HELP: &str = "A-centered 16×16 neural input interventions on the same X/Y axes as Local geometry. Each E uses the full original selected-layer representation and its covariance metric, not PCA. Colour stays on one fixed 0…1 scale across layers and checkpoints. Crossed cells are invalid numerical evaluations, never zero error. Descriptor/plot ranges do not prove environmental admissibility. A is an outlined circle; x_t is a diamond; the light arrow is the slice projection of A -> B.";
#[path = "network_representation_view.rs"]
mod representation_view;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NetworkMode {
    Representation,
    Geometry,
}

struct RepresentationControls {
    worker: RepresentationWorker,
    network: Option<String>,
    layer: Option<String>,
    historical: bool,
    heatmap: bool,
    colour: bool,
    feature: usize,
    painted: bool,
}
impl RepresentationControls {
    fn new(ctx: egui::Context) -> Self {
        Self {
            worker: RepresentationWorker::new(move || ctx.request_repaint()),
            network: None,
            layer: None,
            historical: false,
            heatmap: false,
            colour: false,
            feature: 0,
            painted: false,
        }
    }
}
fn control_label(ui: &mut egui::Ui, label: &str, help: &str) {
    ui.horizontal(|ui| {
        ui.label(label);
        info_icon(ui, help);
    });
}
fn panel_title(ui: &mut egui::Ui, title: &str, help: &str) {
    ui.horizontal(|ui| {
        ui.heading(title);
        info_icon(ui, help);
    });
}
fn info_icon(ui: &mut egui::Ui, help: &str) {
    // Draw the icon: the bundled fonts do not contain a circled-i glyph.
    let (rect, response) = ui.allocate_exact_size(egui::vec2(16., 18.), egui::Sense::hover());
    let center = rect.center();
    ui.painter()
        .circle_stroke(center, 6., egui::Stroke::new(1., crate::theme::MUTED));
    ui.painter().text(
        center,
        egui::Align2::CENTER_CENTER,
        "i",
        egui::FontId::proportional(11.),
        crate::theme::MUTED,
    );
    response.on_hover_text(help);
}

pub(super) struct NetworkPane {
    pub worker: NetworkWorker,
    mode: Option<NetworkMode>,
    mode_automatic: bool,
    representation: RepresentationControls,
    network: String,
    pub(super) layer: usize,
    axes: [usize; 3],
    three_d: bool,
    angle: f64,
    pub(super) t: f64,
    radius: f64,
    endpoint_row: usize,
    endpoint_explicit: bool,
    yaw: f64,
    pitch: f64,
    context: Option<(Arc<std::path::PathBuf>, String)>,
    available: Vec<NetworkChoice>,
    requested: Option<crate::snapshot_catalog::SnapshotEntry>,
    painted: Option<String>,
    painted_scope: Option<String>,
    probe_population: Option<PreparedProbes>,
    metadata: Option<PreparedMetadata>,
    capture_error: Option<String>,
    capture_layer: Option<usize>,
    capture_axes: Option<Vec<usize>>,
    capture_endpoint: Option<usize>,
}
struct PreparedMetadata {
    key: crate::data::CacheKey,
    source: Weak<Chunk>,
    configuration: Arc<SnapshotConfiguration>,
    choices: Arc<Vec<NetworkChoice>>,
}

/// Only the bounded numerical population is retained. A weak source key reserves
/// allocation identity without keeping the recorded arrays alive.
struct PreparedProbes {
    source: Weak<Chunk>,
    input_width: usize,
    observation_dim: usize,
    action_space: ActionSpace,
    values: Arc<Vec<Vec<f64>>>,
}

fn same_action_space(a: &ActionSpace, b: &ActionSpace) -> bool {
    match (a, b) {
        (ActionSpace::Discrete { count: a }, ActionSpace::Discrete { count: b }) => a == b,
        (ActionSpace::Unbounded { dimensions: a }, ActionSpace::Unbounded { dimensions: b }) => {
            a == b
        }
        (ActionSpace::Box { low: al, high: ah }, ActionSpace::Box { low: bl, high: bh }) => {
            let same = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
            };
            same(al, bl) && same(ah, bh)
        }
        _ => false,
    }
}
impl NetworkPane {
    pub fn new(ctx: egui::Context) -> Self {
        let representation = RepresentationControls::new(ctx.clone());
        let mut pane = Self {
            worker: NetworkWorker::new(move || ctx.request_repaint()),
            mode: None,
            mode_automatic: false,
            representation,
            network: String::new(),
            layer: 2,
            axes: [0, 1, 2],
            three_d: cfg!(feature = "screenshots")
                && std::env::var("RL_VIEWER_NETWORK_3D").as_deref() == Ok("true"),
            angle: 0.,
            t: 1.,
            radius: 0.2,
            endpoint_row: 1,
            endpoint_explicit: false,
            yaw: 0.55,
            pitch: 0.3,
            context: None,
            available: Vec::new(),
            requested: None,
            painted: None,
            painted_scope: None,
            probe_population: None,
            metadata: None,
            capture_error: None,
            capture_layer: None,
            capture_axes: None,
            capture_endpoint: None,
        };
        pane.apply_capture_options();
        pane
    }
    fn apply_capture_options(&mut self) {
        #[cfg(feature = "screenshots")]
        {
            self.capture_error = self.read_capture_options().err().map(|e| e.to_string());
        }
    }
    #[cfg(feature = "screenshots")]
    fn read_capture_options(&mut self) -> Result<()> {
        self.representation.heatmap =
            std::env::var("RL_VIEWER_NETWORK_HEATMAP").as_deref() == Ok("true");
        if let Ok(source) = std::env::var("RL_VIEWER_NETWORK_SOURCE") {
            self.mode = Some(match source.as_str() {
                "historical" | "recorded" => NetworkMode::Representation,
                "geometry" => NetworkMode::Geometry,
                _ => return Err(rl::Error::Config("invalid RL_VIEWER_NETWORK_SOURCE".into())),
            });
            self.representation.historical = source == "historical";
        }
        let number = |name: &str| -> Result<Option<usize>> {
            std::env::var(name)
                .ok()
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| rl::Error::Config(format!("invalid {name}")))
                })
                .transpose()
        };
        let scalar = |name: &str, low: f64, high: f64| -> Result<Option<f64>> {
            std::env::var(name)
                .ok()
                .map(|value| {
                    let value = value
                        .parse::<f64>()
                        .map_err(|_| rl::Error::Config(format!("invalid {name}")))?;
                    if !value.is_finite() || !(low..=high).contains(&value) {
                        return Err(rl::Error::Config(format!("{name} must be {low}..={high}")));
                    }
                    Ok(value)
                })
                .transpose()
        };
        if let Ok(name) = std::env::var("RL_VIEWER_NETWORK_NAME") {
            if name.is_empty() {
                return Err(rl::Error::Config("empty RL_VIEWER_NETWORK_NAME".into()));
            }
            self.network = name;
            self.representation.network = Some(self.network.clone());
        }
        if let Some(t) = scalar("RL_VIEWER_NETWORK_T", 0., 1.)? {
            self.t = t;
        }
        if let Some(radius) = scalar("RL_VIEWER_NETWORK_RADIUS", 0.001, 1.)? {
            self.radius = radius;
        }
        self.capture_layer = number("RL_VIEWER_NETWORK_LAYER")?;
        if let Some(layer) = self.capture_layer {
            self.layer = layer;
        }
        self.capture_endpoint = number("RL_VIEWER_NETWORK_B")?;
        if let Some(endpoint) = self.capture_endpoint {
            self.endpoint_row = endpoint;
            self.endpoint_explicit = true;
        }
        if let Ok(axes) = std::env::var("RL_VIEWER_NETWORK_AXES") {
            let axes = axes
                .split(',')
                .map(|a| a.trim().parse::<usize>())
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| rl::Error::Config("invalid RL_VIEWER_NETWORK_AXES".into()))?;
            if !(2..=3).contains(&axes.len())
                || axes.iter().collect::<std::collections::BTreeSet<_>>().len() != axes.len()
            {
                return Err(rl::Error::Config(
                    "RL_VIEWER_NETWORK_AXES needs 2 or 3 distinct indices".into(),
                ));
            }
            self.axes[..axes.len()].copy_from_slice(&axes);
            self.three_d = axes.len() == 3;
            self.capture_axes = Some(axes);
        }
        if self.mode.is_none()
            && (self.capture_layer.is_some()
                || self.capture_axes.is_some()
                || self.capture_endpoint.is_some()
                || self.three_d
                || !self.network.is_empty()
                || std::env::var_os("RL_VIEWER_NETWORK_T").is_some()
                || std::env::var_os("RL_VIEWER_NETWORK_RADIUS").is_some())
        {
            self.mode = Some(NetworkMode::Geometry);
        }
        Ok(())
    }
    fn prepare_metadata(
        &mut self,
        key: &crate::data::CacheKey,
        snapshot: &Arc<Chunk>,
    ) -> Result<(Arc<SnapshotConfiguration>, Arc<Vec<NetworkChoice>>)> {
        if let Some(cached) = &self.metadata
            && cached.key == *key
            && std::ptr::eq(cached.source.as_ptr(), Arc::as_ptr(snapshot))
        {
            return Ok((cached.configuration.clone(), cached.choices.clone()));
        }
        self.metadata = None;
        let Record::ModelSnapshot(meta) = &snapshot.record else {
            return Err(rl::Error::State(
                "expected historical model snapshot".into(),
            ));
        };
        let configuration = Arc::new(snapshot_configuration(meta, ModelLimits::default())?);
        let choices = Arc::new(choices(&configuration));
        self.metadata = Some(PreparedMetadata {
            key: key.clone(),
            source: Arc::downgrade(snapshot),
            configuration: configuration.clone(),
            choices: choices.clone(),
        });
        Ok((configuration, choices))
    }
    fn prepare_probes(
        &mut self,
        source: &Arc<Chunk>,
        input_width: usize,
        observation_dim: usize,
        action_space: &ActionSpace,
    ) -> Result<Arc<Vec<Vec<f64>>>> {
        if let Some(cached) = &self.probe_population
            && std::ptr::eq(cached.source.as_ptr(), Arc::as_ptr(source))
            && cached.input_width == input_width
            && cached.observation_dim == observation_dim
            && same_action_space(&cached.action_space, action_space)
        {
            return Ok(cached.values.clone());
        }
        // Failed replacement preparation must not leave a stale population.
        self.probe_population = None;
        let count = input_rows(source)?.min(network::MAX_PROBES);
        let mut probes = Vec::with_capacity(count);
        for row_index in 0..count {
            // Preserve the observation f32 roundtrip, recorded/default actions,
            // and mask checks performed by the original per-paint adapter.
            probes.push(input_row(
                source,
                row_index,
                input_width,
                observation_dim,
                action_space,
            )?);
        }
        let values = Arc::new(probes);
        self.probe_population = Some(PreparedProbes {
            source: Arc::downgrade(source),
            input_width,
            observation_dim,
            action_space: action_space.clone(),
            values: values.clone(),
        });
        Ok(values)
    }
    pub fn clear(&mut self) {
        self.worker.reset();
        self.retire_representations();
        self.mode = None;
        self.mode_automatic = false;
        self.representation.network = None;
        self.representation.layer = None;
        self.representation.historical = false;
        self.representation.feature = 0;
        self.network.clear();
        self.endpoint_row = 1;
        self.endpoint_explicit = false;
        self.context = None;
        self.available.clear();
        self.requested = None;
        self.painted = None;
        self.painted_scope = None;
        self.probe_population = None;
        self.metadata = None;
        self.apply_capture_options();
    }
    pub(super) fn retire_representations(&mut self) {
        self.representation.worker.clear();
        self.representation.painted = false;
    }
}
fn append_action(
    source: &Chunk,
    row_index: usize,
    width: usize,
    action_space: &ActionSpace,
) -> Result<Vec<f64>> {
    if width == 0 {
        return Ok(Vec::new());
    }
    let recorded = match &source.record {
        Record::ProbeSet(probes) => probes
            .fixed_actions
            .as_deref()
            .map(|key| row(source, key, row_index))
            .transpose()?,
        Record::Trajectory(trajectory) => {
            let rows = array(source, &trajectory.actions)?.shape()[0];
            if row_index < rows {
                Some(row(source, &trajectory.actions, row_index)?)
            } else {
                None
            }
        }
        _ => None,
    };
    if let Some(values) = recorded
        && values.len() == width
    {
        return Ok(values);
    }
    let default = match action_space {
        ActionSpace::Box { low, high } => low
            .iter()
            .zip(high)
            .map(|(lo, hi)| f64::from((lo + hi) / 2.))
            .collect::<Vec<_>>(),
        ActionSpace::Unbounded { dimensions } => vec![0.; *dimensions],
        ActionSpace::Discrete { .. } => vec![],
        ActionSpace::Conditional(_) => {
            return Err(rl::Error::Config(
                "network input requires a numeric action space".into(),
            ));
        }
    };
    if default.len() != width {
        return Err(rl::Error::Shape(
            "network action-input width mismatch".into(),
        ));
    }
    Ok(default)
}
fn input_row(
    source: &Chunk,
    row_index: usize,
    input_width: usize,
    observation_dim: usize,
    action_space: &ActionSpace,
) -> Result<Vec<f64>> {
    let observation = source_observation(source, row_index)?;
    let mut values: Vec<f64> = observation
        .dense_values()
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    if values.len() != observation_dim || input_width < values.len() {
        return Err(rl::Error::Shape(
            "network observation-input mismatch".into(),
        ));
    }
    values.extend(append_action(
        source,
        row_index,
        input_width - observation_dim,
        action_space,
    )?);
    Ok(values)
}
fn input_rows(source: &Chunk) -> Result<usize> {
    match &source.record {
        Record::ProbeSet(p) => Ok(array(source, &p.observations)?.shape()[0]),
        Record::Trajectory(t) => Ok(array(source, &t.observations)?.shape()[0]),
        _ => Err(rl::Error::Config(
            "Network requires a recorded input population".into(),
        )),
    }
}
fn axis_name(e: EnvironmentSemantics<'_>, i: usize) -> String {
    if i < e.observation_dim() {
        crate::landscape::observation_axis_label(e, i)
    } else {
        format!("a[{}]", i - e.observation_dim())
    }
}
fn axis_range(e: EnvironmentSemantics<'_>, inputs: &[Vec<f64>], index: usize) -> [f64; 2] {
    if index < e.observation_dim()
        && let Some(obs) = e.observation(index)
        && let Some(range) = obs.range
        && range[0] < range[1]
    {
        return [f64::from(range[0]), f64::from(range[1])];
    }
    if index >= e.observation_dim()
        && let ActionSpace::Box { low, high } = e.action_space()
        && let Some((&lo, &hi)) = low
            .get(index - e.observation_dim())
            .zip(high.get(index - e.observation_dim()))
        && lo < hi
    {
        return [f64::from(lo), f64::from(hi)];
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for row in inputs {
        lo = lo.min(row[index]);
        hi = hi.max(row[index]);
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        lo = inputs[0][index] - 0.5;
        hi = inputs[0][index] + 0.5;
    }
    let padding = 0.08 * (hi - lo);
    [lo - padding, hi + padding]
}
fn valid_axis(e: EnvironmentSemantics<'_>, index: usize) -> bool {
    if index >= e.observation_dim() {
        return !matches!(e.action_space(), ActionSpace::Discrete { .. });
    }
    e.observation(index).is_none_or(|v| {
        v.kind
            .is_none_or(|kind| matches!(kind, ObservationType::Numerical))
    })
}
fn has_representations(run: &RunIndex, role: &str) -> bool {
    run.candidates(RecordKind::Activations, role, None, u64::MAX)
        .next()
        .is_some()
}

fn default_network_mode(run: &RunIndex, role: &str) -> NetworkMode {
    if !run.snapshot_revisions(role).is_empty() || !has_representations(run, role) {
        NetworkMode::Geometry
    } else {
        NetworkMode::Representation
    }
}

fn selected_population(selection: &Selection) -> Option<&str> {
    match &selection.input {
        SelectedInput::ProbeRow { set, .. } => Some(set),
        _ => None,
    }
}

impl Laboratory {
    pub(super) fn network_has_representation_source(&self, run: &RunIndex) -> bool {
        match self.network_pane.mode {
            Some(mode) => mode == NetworkMode::Representation,
            None => {
                default_network_mode(run, &self.selection.model_role) == NetworkMode::Representation
            }
        }
    }

    /// The actual published geometry, with immutable snapshot provenance.
    pub fn network_frame(&self) -> Option<&network::work::PublishedFrame> {
        self.network_pane.worker.frame()
    }
    /// Complete, selected population published by the representation worker.
    pub fn network_representation_frame(&self) -> Option<&RepresentationFrame> {
        self.network_pane.representation.worker.frame.as_ref()
    }
    pub fn network_ready(&self) -> bool {
        if self.network_pane.mode == Some(NetworkMode::Representation) {
            let pane = &self.network_pane.representation;
            return self.network_pane.capture_error.is_none()
                && pane.painted
                && !pane.worker.pending()
                && pane.worker.frame.as_ref().is_some_and(|frame| {
                    frame.requested_revision == self.selection.revision
                        && frame.historical == pane.historical
                });
        }
        self.network_pane.capture_error.is_none()
            && self.network_pane.worker.ready()
            && self.network_pane.painted.as_deref() == self.network_pane.worker.submitted()
            && self.network_frame().is_some_and(|frame| {
                frame.snapshot.revision == self.selection.revision
                    && frame.snapshot.name == self.network_pane.network
                    && frame.snapshot.role == self.selection.model_role
            })
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn network_capture_error(&self) -> Option<String> {
        if self.network_pane.mode == Some(NetworkMode::Representation) {
            return self
                .network_pane
                .capture_error
                .clone()
                .or_else(|| self.network_pane.representation.worker.error.clone());
        }
        self.network_pane
            .capture_error
            .clone()
            .or_else(|| self.network_pane.worker.error().map(str::to_owned))
    }
    #[cfg(feature = "screenshots")]
    pub(super) fn network_capture_evidence(&self) -> serde_json::Value {
        if self.network_pane.mode == Some(NetworkMode::Representation) {
            let Some(frame) = self.network_representation_frame() else {
                return serde_json::json!({"mode":"representation","ready":false});
            };
            return serde_json::json!({
                "mode":"representation", "ready":self.network_ready(),
                "historical":frame.historical, "historical_fixed_reference":frame.historical_fixed_reference,
                "name":frame.network, "layer":frame.layer,
                "role":self.selection.model_role, "revision":frame.revision,
                "requested_revision":frame.requested_revision, "basis_revision":frame.basis_revision,
                "basis_recording":frame.basis_recording, "population":frame.population,
                "source_sequence":frame.probe_sequence, "input":format!("{:?}", self.selection.input),
                "selected_row":frame.row_for_input(&self.selection.input),
                "rows":frame.row_ids.len(), "width":frame.width,
                "source_width":frame.source_width, "cka":frame.cka,
                "explained":frame.explained, "input_lineage":frame.input_lineage,
                "comparison":frame.comparison_note, "projection":frame.projection_note,
            });
        }
        let Some(published) = self.network_frame() else {
            return serde_json::json!({"ready": false});
        };
        let frame = &published.geometry;
        let layer = self.network_pane.layer;
        let output = match &published.output {
            Ok(output) => {
                serde_json::json!({"label":output.label,"A":output.anchor,"x":output.point,
                "components":output.components,"note":output.note,"bytes":output.retained_bytes()})
            }
            Err(reason) => serde_json::json!({"unavailable":reason}),
        };
        serde_json::json!({
            "ready":self.network_ready(), "name":published.snapshot.name,
            "revision":published.snapshot.revision, "role":published.snapshot.role,
            "source_sequence":published.snapshot.key.sequence, "input":format!("{:?}",self.selection.input),
            "signature":self.network_pane.worker.submitted(), "B_row":self.network_pane.endpoint_row,
            "t":frame.t,"radius":frame.radius,"axes":self.network_pane.axes,
            "layer":layer,"error_scale":network::finite::ERROR_SCALE,
            "grid_points":frame.neighborhood.positions.len(),"invalid":frame.neighborhood.invalid_count(layer),
            "bounds":frame.neighborhood.bounds,"errors":frame.neighborhood.errors.get(layer),
            "slice_A":frame.slice_anchor,"slice_x_t":frame.slice_position,"slice_B":frame.slice_endpoint,"position_on_slice":frame.position_on_slice,
            "comparisons":frame.comparisons.iter().map(|p|serde_json::json!({"layer":p.name,
                "linear":p.linear,"actual":p.actual,"linear_norm":p.linear_norm,"actual_norm":p.actual_norm,
                "residual_norm":p.residual_norm,"error":p.error})).collect::<Vec<_>>(),
            "actual_output":output,"preparation_retained_bytes":published.preparation_retained_bytes,
            "added_frame_bytes":frame.finite_retained_bytes(),
        })
    }
    pub(super) fn network_snapshot_transport(
        &mut self,
        ui: &mut egui::Ui,
        run: &RunIndex,
    ) -> Result<()> {
        use crate::{
            snapshot_catalog::{SnapshotCatalog, SnapshotPanel},
            timeline_ui::SnapshotPresentation,
        };
        let context = (run.root().clone(), self.selection.model_role.clone());
        if self.network_pane.context.as_ref() != Some(&context) {
            self.network_pane.clear();
            self.network_pane.context = Some(context);
            self.transport.clear_snapshot_state();
        }
        if self.network_pane.mode.is_none() {
            self.network_pane.mode = Some(default_network_mode(run, &self.selection.model_role));
            self.network_pane.mode_automatic = true;
        }
        let mode = self.network_pane.mode.as_ref().unwrap();
        if *mode == NetworkMode::Representation {
            return self.representation_transport(ui, run);
        }
        self.network_pane.worker.poll();
        let panel = SnapshotPanel::Network {
            name: (!self.network_pane.network.is_empty())
                .then(|| self.network_pane.network.clone()),
        };
        if self
            .snapshot_catalog
            .as_ref()
            .is_none_or(|c| !c.matches(run, &self.selection.model_role, &panel))
        {
            self.interactive.cancel_snapshot_validation();
            self.snapshot_catalog = None;
            self.snapshot_catalog = Some(SnapshotCatalog::new(
                run,
                &self.selection.model_role,
                panel,
                &self.interactive,
            )?);
            self.transport.clear_snapshot_state();
        }
        let catalog = self.snapshot_catalog.as_mut().unwrap();
        let priority = run
            .snapshot_revisions(&self.selection.model_role)
            .iter()
            .rev()
            .find(|&&k| k <= self.selection.revision)
            .copied()
            .unwrap_or(self.selection.revision);
        catalog.prioritize_revision(priority);
        if self.network_pane.worker.pending() {
            catalog.yield_to_foreground(&mut self.interactive);
        } else {
            catalog.advance(run, &mut self.loader, &mut self.interactive)?;
        }
        if self.network_pane.mode_automatic
            && !catalog.pending()
            && catalog.positions().is_empty()
            && has_representations(run, &self.selection.model_role)
        {
            self.network_pane.mode = Some(NetworkMode::Representation);
            self.interactive.cancel_snapshot_validation();
            self.snapshot_catalog = None;
            self.transport.clear_snapshot_state();
            self.network_pane.worker.reset();
            return self.representation_transport(ui, run);
        }
        for choice in catalog.networks() {
            if !self
                .network_pane
                .available
                .iter()
                .any(|old| old.name == choice.name)
            {
                self.network_pane.available.push(choice.clone());
            }
        }
        if self.network_pane.network.is_empty()
            && let Some(choice) = self.network_pane.available.first()
        {
            self.network_pane.network = choice.name.clone();
            ui.weak("Checking model snapshots…");
            ui.ctx().request_repaint();
            return Ok(());
        }
        let unavailable = self.network_pane.requested.as_ref().is_some_and(|entry| {
            entry.revision == self.selection.revision
                && run
                    .exact_snapshot_index(&entry.key, entry.revision, &entry.role)
                    .is_err()
        });
        if unavailable {
            self.network_pane.metadata = None;
            catalog.recheck_availability(run);
            self.network_pane.worker.invalidate_frame();
            self.transport
                .hold_unavailable_snapshot(Some(self.selection.revision));
        } else {
            self.transport.hold_unavailable_snapshot(None);
        }
        let published = self
            .network_pane
            .worker
            .frame()
            .filter(|frame| {
                frame.snapshot.name == self.network_pane.network
                    && catalog
                        .selected(frame.snapshot.revision)
                        .is_some_and(|entry| entry.key == frame.snapshot.key)
            })
            .map(|frame| frame.snapshot.revision);
        // Preparation alone is insufficient: at high playback speeds a fresh
        // completion must be painted once before another snapshot is requested.
        let ready = self.network_pane.worker.ready()
            && self.network_pane.painted.as_deref() == self.network_pane.worker.submitted()
            && !catalog.pending()
            && !unavailable;
        let error = if unavailable {
            Some("Checkpoint unavailable; choose another snapshot")
        } else {
            self.network_pane.worker.error()
        };
        if catalog.limited() {
            ui.weak("Snapshot catalog limited to 256 candidates");
        }
        let retry = self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            SnapshotPresentation {
                positions: &catalog.positions(),
                published,
                ready,
                checking: catalog.pending(),
                error,
            },
        )?;
        if retry {
            self.network_pane.metadata = None;
            catalog.retry_rejected();
            self.network_pane.worker.invalidate_frame();
            // Retry is an explicit user choice to resolve the current revision
            // again, including a replacement artifact if the original was pruned.
            self.network_pane.requested = None;
        }
        Ok(())
    }

    fn representation_transport(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        use crate::timeline_ui::SnapshotPresentation;
        let pane = &mut self.network_pane.representation;
        pane.worker.poll();
        let role = &self.selection.model_role;
        let positions = if pane.historical {
            run.snapshot_revisions(role)
                .iter()
                .rev()
                .take(256)
                .copied()
                .collect::<Vec<_>>()
        } else {
            run.candidates(
                RecordKind::Activations,
                role,
                selected_population(&self.selection),
                u64::MAX,
            )
            .take(256)
            .map(|index| run.recording().entries()[index].revision)
            .collect()
        };
        let mut positions = positions;
        positions.sort_unstable();
        positions.dedup();
        let revision = self.selection.revision;
        let published = pane
            .worker
            .frame
            .as_ref()
            .filter(|frame| {
                frame.requested_revision == revision && frame.historical == pane.historical
            })
            .map(|frame| frame.revision);
        let checking =
            pane.worker.pending() || (published.is_none() && pane.worker.error.is_none());
        if self.transport.show_snapshots(
            ui,
            run,
            &mut self.selection,
            SnapshotPresentation {
                positions: &positions,
                published,
                ready: pane.painted && !checking,
                checking,
                error: pane.worker.error.as_deref(),
            },
        )? {
            pane.worker.clear();
            pane.painted = false;
        }
        Ok(())
    }

    fn representation_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        let pane = &mut self.network_pane.representation;
        pane.worker.request(RepresentationQuery {
            run: run.clone(),
            role: self.selection.model_role.clone(),
            population: selected_population(&self.selection).map(str::to_owned),
            anchor_sequence: match &self.selection.input {
                SelectedInput::ProbeRow { sequence, .. } => Some(*sequence),
                _ => None,
            },
            revision: self.selection.revision,
            network: pane.network.clone(),
            layer: pane.layer.clone(),
            reference: self.comparison.reference_run().map(|reference| {
                (
                    reference.clone(),
                    self.comparison.reference_role().to_owned(),
                    self.comparison.reference_revision(),
                )
            }),
            historical: pane.historical,
        });
        if let Some(input) = representation_view::show(
            ui,
            pane,
            &self.selection.input,
            &run.recording().manifest().model_id,
        ) {
            crate::pane::SelectionIntent::Input(input).apply(run, &mut self.selection)?;
            self.workspace_defaults = None;
        }
        Ok(())
    }

    pub(super) fn network_workspace(&mut self, ui: &mut egui::Ui, run: &RunIndex) -> Result<()> {
        if let Some(error) = &self.network_pane.capture_error {
            return Err(rl::Error::Config(error.clone()));
        }
        let old = self.network_pane.mode.unwrap_or(NetworkMode::Geometry);
        let mut mode = old;
        let mut explicitly_selected = false;
        if has_representations(run, &self.selection.model_role) {
            ui.horizontal(|ui| {
                explicitly_selected |= ui
                    .selectable_value(&mut mode, NetworkMode::Representation, "Representations")
                    .clicked();
                if run
                    .snapshot_revisions(&self.selection.model_role)
                    .is_empty()
                {
                    ui.add_enabled(false, egui::Button::new("Geometry"))
                        .on_disabled_hover_text(
                            "This recording has no compatible dense intervention model",
                        );
                } else {
                    explicitly_selected |= ui
                        .selectable_value(&mut mode, NetworkMode::Geometry, "Geometry")
                        .on_hover_text("Advanced interventions for compatible dense models")
                        .clicked();
                }
            });
        }
        if explicitly_selected {
            self.network_pane.mode_automatic = false;
        }
        if mode != old {
            self.network_pane.mode = Some(mode);
            self.transport.clear_snapshot_state();
            self.interactive.cancel_snapshot_validation();
            self.snapshot_catalog = None;
            self.network_pane.retire_representations();
            self.network_pane.worker.reset();
        }
        if mode == NetworkMode::Representation {
            return self.representation_workspace(ui, run);
        }
        // The header is the only publication point. A completion arriving here
        // waits for the next paint, keeping its counter and geometry atomic.
        let Some((source, anchor_row)) = self.selected_source(run)? else {
            self.network_pane.probe_population = None;
            ui.heading("Network");
            ui.weak("Select a fixed probe state or an episode step to inspect the network.");
            return Ok(());
        };
        let role = self.selection.model_role.clone();
        if self.network_pane.requested.as_ref().is_some_and(|entry| {
            entry.revision == self.selection.revision
                && run
                    .exact_snapshot_index(&entry.key, entry.revision, &entry.role)
                    .is_err()
        }) {
            network_model_selector(
                ui,
                &mut self.network_pane,
                run,
                &mut self.selection.model_role,
            );
            ui.weak("Selected network checkpoint unavailable");
            return Ok(());
        }
        let Some(entry) = self
            .snapshot_catalog
            .as_ref()
            .and_then(|catalog| catalog.selected(self.selection.revision))
            .filter(|entry| entry.name == self.network_pane.network)
            .cloned()
        else {
            network_model_selector(
                ui,
                &mut self.network_pane,
                run,
                &mut self.selection.model_role,
            );
            ui.weak(
                if self.snapshot_catalog.as_ref().is_some_and(|c| c.pending()) {
                    "Checking selected network snapshots…"
                } else {
                    "Selected network checkpoint unavailable"
                },
            );
            return Ok(());
        };
        let snapshot_index = match run.exact_snapshot_index(&entry.key, entry.revision, &role) {
            Ok(index) => index,
            Err(_) => {
                ui.weak("Selected network checkpoint unavailable");
                return Ok(());
            }
        };
        let snapshot = self.loader.request(run, snapshot_index)?;
        let metadata_ready = snapshot.is_some();
        let (config, snapshot_choices) = if let Some(snapshot) = snapshot {
            self.network_pane.prepare_metadata(&entry.key, &snapshot)?
        } else if let Some(frame) = self.network_pane.worker.frame() {
            // Keep controls and plots in place even while the next metadata
            // chunk loads. Never submit inference using this older metadata.
            (frame.configuration.clone(), frame.choices.clone())
        } else {
            ui.spinner();
            ui.weak("Loading historical network metadata…");
            return Ok(());
        };
        let pane = &mut self.network_pane;
        let Some(choice) = snapshot_choices.iter().find(|c| c.name == pane.network) else {
            ui.weak("Selected network absent from this checkpoint");
            return Ok(());
        };
        let rows = input_rows(&source)?;
        if rows == 0 || anchor_row >= rows {
            return Err(rl::Error::Shape(
                "selected Network row outside input source".into(),
            ));
        }
        if pane
            .capture_layer
            .is_some_and(|layer| layer > choice.layers)
            || pane.capture_endpoint.is_some_and(|row| row >= rows)
        {
            return Err(rl::Error::Config(
                "Network capture layer or B row outside the selected source".into(),
            ));
        }
        if !pane.endpoint_explicit {
            pane.endpoint_row = (anchor_row + rows / 3).min(rows - 1);
        }
        pane.endpoint_row = pane.endpoint_row.min(rows - 1);
        let e = EnvironmentSemantics::new(run.recording().manifest());
        let selected_mask = source_observation(&source, anchor_row)?
            .dense_mask()
            .map(<[bool]>::to_vec);
        let probes = pane.prepare_probes(
            &source,
            choice.input_width,
            config.observation_dim,
            &config.action_space,
        )?;
        let anchor = input_row(
            &source,
            anchor_row,
            choice.input_width,
            config.observation_dim,
            &config.action_space,
        )?;
        let endpoint = input_row(
            &source,
            pane.endpoint_row,
            choice.input_width,
            config.observation_dim,
            &config.action_space,
        )?;
        let numeric_axes: Vec<usize> = (0..choice.input_width)
            .filter(|&i| valid_axis(e, i))
            .collect();
        if pane
            .capture_axes
            .as_ref()
            .is_some_and(|axes| axes.iter().any(|axis| !numeric_axes.contains(axis)))
        {
            return Err(rl::Error::Config(
                "Network capture axes are not numerical inputs of the selected network".into(),
            ));
        }
        if numeric_axes.len() < 2 {
            ui.heading("Network");
            ui.weak("The geometric 2D/3D workspace needs at least two numerical input directions.");
            return Ok(());
        }
        for j in 0..3 {
            if (!numeric_axes.contains(&pane.axes[j])
                || (0..j).any(|k| pane.axes[k] == pane.axes[j]))
                && let Some(&replacement) = numeric_axes
                    .iter()
                    .find(|&&i| !(0..j).any(|k| pane.axes[k] == i))
            {
                pane.axes[j] = replacement;
            }
        }
        if numeric_axes.len() < 3 {
            pane.three_d = false;
        }
        pane.layer = pane.layer.min(choice.layers);
        let mut ranges = [
            axis_range(e, &probes, pane.axes[0]),
            axis_range(e, &probes, pane.axes[1]),
        ];
        let actual_revision = run.recording().entries()[snapshot_index].revision;
        let source_key = run.key(snapshot_index)?;

        let mut pending_anchor: Option<usize> = None;
        let result = egui::ScrollArea::vertical()
            .id_salt("network-workspace")
            .show(ui, |ui| -> Result<()> {
                ui.horizontal_wrapped(|ui| {
                                network_model_selector(ui, pane, run, &mut self.selection.model_role);
                                ui.label("Layer");
                                egui::ComboBox::from_id_salt("network-layer")
                                    .selected_text(network::layer_name(pane.layer, choice.layers))
                                    .width(110.)
                                    .show_ui(ui, |ui| {
                                        for layer in 0..=choice.layers {
                                            ui.selectable_value(&mut pane.layer, layer,
                                                network::layer_name(layer, choice.layers));
                                        }
                                    });
                                ui.label("A").on_hover_text("The selected recorded observation, including recorded or default critic action coordinates. Checkpoint navigation preserves this input and its source population.");
                                let mut chosen_row = anchor_row;
                                if ui.add(egui::DragValue::new(&mut chosen_row).range(0..=rows-1).prefix("row ")).changed() {
                                    pending_anchor = Some(chosen_row);
                                }
                                ui.label("B").on_hover_text(INTERPOLATION_HELP);
                                if ui.add(egui::DragValue::new(&mut pane.endpoint_row).range(0..=rows-1).prefix("row ")).changed() {
                                    pane.endpoint_explicit = true;
                                }
                                crate::landscape::popover(ui, "Slice", |ui| {
                                ui.set_max_width(340.);
                                ui.label("Vary these coordinates; all others stay fixed at A.");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(&mut pane.three_d, false, "2D");
                                    ui.add_enabled_ui(numeric_axes.len() > 2, |ui| {
                                        ui.selectable_value(&mut pane.three_d, true, "3D");
                                    });
                                });
                                for axis in 0..(2 + usize::from(pane.three_d)) {
                                    ui.horizontal(|ui| {
                                        ui.label(["X", "Y", "Z"][axis]);
                                        egui::ComboBox::from_id_salt(("network-axis", axis))
                                            .selected_text(axis_name(e, pane.axes[axis]))
                                            .width(174.)
                                            .show_ui(ui, |ui| {
                                                for &index in &numeric_axes {
                                                    if !(0..2 + usize::from(pane.three_d)).any(|j| j != axis && pane.axes[j] == index) {
                                                        ui.selectable_value(&mut pane.axes[axis], index, axis_name(e, index));
                                                    }
                                                }
                                            });
                                    });
                                }
                                if pane.three_d {
                                    ui.add(egui::Slider::new(&mut pane.yaw, -std::f64::consts::PI..=std::f64::consts::PI).text("yaw"));
                                    ui.add(egui::Slider::new(&mut pane.pitch, -1.5..=1.5).text("pitch"));
                                }
                                control_label(ui, "Slice direction", "Direction used by the additional output directional-sensitivity readout. Finite propagation uses the complete displacement B−A instead.");
                                ui.add(egui::Slider::new(&mut pane.angle, 0. ..=std::f64::consts::TAU).suffix(" rad"));
                                ui.small("A unit direction spans one complete input coordinate range. The 3D view is an orthographic projection at A.");
                                ui.small(format!("{} probes · normalized", probes.len())).on_hover_text(NORMALIZATION_HELP);
                                });
                });
                ui.add_space(8.);
                    ui.vertical(|ui| -> Result<()> {
                        if pane.network != entry.name || self.selection.model_role != role {
                            ui.ctx().request_repaint();
                            return Ok(());
                        }
                        // Input coordinates are normalized by the slice ranges,
                        // so a unit tangent means one complete coordinate range.
                        ranges[0] = axis_range(e, &probes, pane.axes[0]);
                        ranges[1] = axis_range(e, &probes, pane.axes[1]);
                        let axes = pane.axes[..2 + usize::from(pane.three_d)].to_vec();
                        let view_scope = format!(
                            "{}:{role}:{}:{anchor_row}:{}:{:?}:{:?}:{}:{}:{}:{}:{}:{:?}",
                            run.root().display(),
                            pane.network,
                            pane.endpoint_row,
                            axes,
                            ranges,
                            pane.layer,
                            pane.angle.to_bits(),
                            pane.three_d,
                            pane.t.to_bits(),
                            pane.radius.to_bits(),
                            (&anchor, &endpoint, &self.selection.input),
                        );
                        let signature = format!("{actual_revision}:{}:{view_scope}", source_key.sequence);
                        // The worker already deduplicates this exact signature.
                        // Avoid building owned numerical work that it would discard.
                        if metadata_ready && pane.worker.submitted() != Some(signature.as_str()) {
                            let axis_scales = axes.iter()
                                .map(|&i| {
                                    let r = axis_range(e, &probes, i);
                                    r[1] - r[0]
                                })
                                .collect::<Vec<_>>();
                            let request = network::Request {
                                network: pane.network.clone(),
                                anchor,
                                endpoint,
                                probes: probes.as_ref().clone(),
                                axes,
                                axis_scales,
                                ranges,
                                direction_angle: pane.angle,
                                t: pane.t,
                                radius: pane.radius,
                                action_mask: selected_mask.clone(),
                                layer: pane.layer,
                                grid: [12, 8],
                                previous_basis: Vec::new(),
                            };
                            let basis_scope = format!(
                                "{}:{role}:{}:{}",
                                run.root().display(),
                                pane.network,
                                match &self.selection.input {
                                    SelectedInput::ProbeRow { sequence, .. }
                                    | SelectedInput::TrajectoryStep { sequence, .. } => *sequence,
                                    SelectedInput::None => 0,
                                },
                            );
                            pane.worker.request(
                                signature,
                                basis_scope,
                                entry.clone(),
                                run.recording().source(snapshot_index)?,
                                request,
                            );
                        }
                        pane.requested = Some(entry.clone());
                        let ready = metadata_ready && pane.worker.ready()
                            && pane.worker.frame().is_some_and(|frame| frame.snapshot.key == entry.key);
                        // Only checkpoint changes may reuse a painted view; an
                        // edited input, layer or slice must not relabel old geometry.
                        let visible = pane.worker.frame().filter(|frame| {
                            ready || (pane.painted_scope.as_deref() == Some(&view_scope)
                                && run.exact_snapshot_index(&frame.snapshot.key, frame.snapshot.revision, &frame.snapshot.role).is_ok())
                        });
                        if let Some(published) = visible {
                            let frame = &published.geometry;
                            let displayed_choices = &published.choices;
                            let displayed_choice = displayed_choices.iter()
                                .find(|choice| choice.name == published.snapshot.name)
                                .expect("validated published network");
                            ui.columns(2, |columns| {
                            crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    ui.heading("Local geometry");
                                    info_icon(ui, GEOMETRY_HELP);
                                    ui.weak(format!("{} · {}", published.snapshot.name, network::layer_name(pane.layer, displayed_choice.layers)));
                                    crate::landscape::popover(ui, "Details", |ui| {
                                        ui.set_width(440_f32.min(ui.ctx().content_rect().width() - 48.));
                                        egui::ScrollArea::vertical().max_height(440.).show(ui, |ui| {
                                            ui.strong(format!("Shown k {} · {} · {}", published.snapshot.revision, published.snapshot.role, published.snapshot.name));
                                            ui.small(format!("Snapshot source sequence {}", published.snapshot.key.sequence));
                                            for (title, help) in [("Geometry", GEOMETRY_HELP), ("Metric & reference population", NORMALIZATION_HELP), ("Finite propagation", FINITE_HELP), ("Local nonlinearity", NONLINEARITY_HELP), ("Representation paths", INTERPOLATION_HELP)] {
                                                ui.strong(title);
                                                ui.label(help);
                                            }
                                            if frame.kink_count > 0 {
                                                ui.colored_label(crate::theme::WARNING, format!("{} exact ReLU kinks at A. Derivatives use the existing zero-derivative convention.", frame.kink_count));
                                            }
                                            ui.separator();
                                            ui.strong("Exact finite comparisons");
                                            for pair in &frame.comparisons {
                                                ui.label(format!("{}: E={} · ||L||={} · ||Δ||={} · ||Δ−L||={}", pair.name, pair.error, pair.linear_norm, pair.actual_norm, pair.residual_norm));
                                            }
                                            ui.label(format!("Slice A {:?}; B {:?}; x_t {:?}; t={}; radius={} coordinate ranges", frame.slice_anchor, frame.slice_endpoint, frame.slice_position, frame.t, frame.radius));
                                            for path in &frame.paths {
                                                ui.label(format!("{}: fixed-reference PCA variance {}; projected path-step energy {}", path.layer, path.variance, path.projected_energy));
                                            }
                                            ui.strong(format!("Raw output: {}", displayed_choice.kind));
                                            match &published.output {
                                                Ok(output) => {
                                                    ui.label(&output.label);
                                                    ui.label(format!("A {:?} -> x_t {:?}", output.anchor, output.point));
                                                    ui.label(format!("{} {} components; at most the first 64 are retained for this readout.", output.note, output.components));
                                                }
                                                Err(reason) => { ui.label(format!("Actual output unavailable: {reason}")); }
                                            }
                                            let (label, sensitivity) = head_sensitivity(frame, &published.configuration.models[&displayed_choice.model].kind, &published.configuration.action_space, selected_mask.as_deref());
                                            ui.label(format!("{label}: {sensitivity}"));
                                            ui.small("This output-head directional statistic uses Slice direction, independently of the complete finite displacement B−A and the covariance-normalized representation metric.");
                                        });
                                    });
                                });
                                if frame.kink_count > 0 {
                                    ui.colored_label(crate::theme::WARNING, format!("{} ReLU kinks at A", frame.kink_count))
                                        .on_hover_text("The network is not differentiable at these exact ReLU kinks. Derivatives use the existing zero-derivative convention.");
                                }
                                if pane.three_d {
                                    show_ellipsoid(ui, frame, pane.yaw, pane.pitch);
                                } else {
                                    show_ellipse_field(ui, frame, &ranges, [axis_name(e, pane.axes[0]), axis_name(e, pane.axes[1])]);
                                }
                            });
                              crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    panel_title(ui, "Propagation by layer", FINITE_HELP);
                                    ui.add(egui::Slider::new(&mut pane.t, 0. ..=1.).text("t").fixed_decimals(3))
                                        .on_hover_text("x_t = A+t(B−A). Compare the actual layer displacement with the tangent prediction at A.");
                                    if pane.t.to_bits() != frame.t.to_bits() {
                                        ui.weak(format!("Requested {:.3} · shown {:.3} · pending", pane.t, frame.t));
                                    }
                                });
                                show_propagation(ui, frame);
                              });
                            });
                            ui.add_space(8.);
                            ui.columns(2, |columns| {
                              crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    panel_title(ui, "Local nonlinearity", NONLINEARITY_HELP);
                                    ui.add(egui::Slider::new(&mut pane.radius, 0.001..=1.).logarithmic(true).text("radius"))
                                        .on_hover_text("Half-width in units of the fixed X/Y coordinate ranges, centered at A. Other inputs stay at A. Neural interventions are not environment transitions or guaranteed admissible states.");
                                    if pane.radius.to_bits() != frame.radius.to_bits() {
                                        ui.weak(format!("Requested {:.3} · shown {:.3} · pending", pane.radius, frame.radius));
                                    }
                                });
                                show_nonlinearity(ui, frame, pane.layer,
                                    [axis_name(e, pane.axes[0]), axis_name(e, pane.axes[1])]);
                              });
                              crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                ui.horizontal_wrapped(|ui| {
                                    panel_title(ui, "Representation paths", INTERPOLATION_HELP);
                                });
                                show_paths(ui, frame);
                              });
                            });
                            if ready && pane.painted.as_deref() != pane.worker.submitted() {
                                pane.painted = pane.worker.submitted().map(str::to_owned);
                                pane.painted_scope = Some(view_scope);
                            }
                        } else {
                            let reason = pane.worker.error().unwrap_or("Preparing selected geometry…");
                            ui.columns(2, |columns| {
                            crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                panel_title(ui, "Local geometry", GEOMETRY_HELP);
                                ui.weak(reason);
                                ui.add_space(168.);
                            });
                                crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        panel_title(ui, "Propagation by layer", FINITE_HELP);
                                        ui.add(egui::Slider::new(&mut pane.t, 0. ..=1.).text("t").fixed_decimals(3));
                                    });
                                    ui.weak(format!("Requested t={:.3} · awaiting selected frame", pane.t));
                                    ui.add_space(212.);
                                });
                            });
                            ui.add_space(8.);
                            ui.columns(2, |columns| {
                                crate::theme::panel_frame().show(&mut columns[0], |ui| {
                                    ui.horizontal_wrapped(|ui| {
                                        panel_title(ui, "Local nonlinearity", NONLINEARITY_HELP);
                                        ui.add(egui::Slider::new(&mut pane.radius, 0.001..=1.).logarithmic(true).text("radius"));
                                    });
                                    ui.weak(format!("Requested radius={:.3} · awaiting selected frame", pane.radius));
                                    ui.add_space(212.);
                                });
                            crate::theme::panel_frame().show(&mut columns[1], |ui| {
                                panel_title(ui, "Representation paths", INTERPOLATION_HELP);
                                ui.weak("Awaiting selected geometry; incompatible paths are hidden.");
                                ui.add_space(132.);
                            });
                            });
                            if let Some(error) = pane.worker.error() {
                                ui.colored_label(WARM, error);
                                if ui.button("Retry geometry").clicked() {
                                    pane.worker.reset();
                                }
                            }
                        }
                        Ok(())
                    }).inner
            })
            .inner;
        if let Some(row) = pending_anchor {
            match &mut self.selection.input {
                SelectedInput::ProbeRow { row: selected, .. } => *selected = row,
                SelectedInput::TrajectoryStep { step, .. } => *step = row,
                SelectedInput::None => {}
            }
        }
        result
    }
}
fn network_model_selector(
    ui: &mut egui::Ui,
    pane: &mut NetworkPane,
    run: &RunIndex,
    role: &mut String,
) {
    ui.label("Model");
    if run.snapshot_roles().count() > 1 {
        egui::ComboBox::from_id_salt("network-model-role")
            .selected_text(role.as_str())
            .width(80.)
            .show_ui(ui, |ui| {
                for option in run.snapshot_roles() {
                    ui.selectable_value(role, option.into(), option);
                }
            });
    }
    let label = pane
        .available
        .iter()
        .find(|c| c.name == pane.network)
        .map_or("Choose model", |c| c.label.as_str());
    egui::ComboBox::from_id_salt("network-role")
        .selected_text(label)
        .width(195.)
        .show_ui(ui, |ui| {
            for option in &pane.available {
                if ui
                    .selectable_value(&mut pane.network, option.name.clone(), &option.label)
                    .changed()
                {
                    pane.worker.reset();
                    pane.requested = None;
                }
            }
        }).response.on_hover_text("Inspect one retained historical network at a time. Critic inputs use recorded actions, otherwise zero or the action-space midpoint. Complete definitions and output semantics are in Details.");
}
fn ellipse_coordinates(
    grid: &[network::Glyph],
    scales: [f64; 2],
    multiplier: f64,
) -> Vec<PlotPoint> {
    // Keep separate cos/sin calls and the original final endpoint's bits.
    let circle: [[f64; 2]; 33] = std::array::from_fn(|i| {
        let t = std::f64::consts::TAU * i as f64 / 32.;
        [t.cos(), t.sin()]
    });
    // This paint-local allocation replaces the individually owned Line inputs.
    let mut coordinates = Vec::with_capacity(grid.len() * circle.len());
    for glyph in grid {
        let position = glyph.position;
        let [a, b] = glyph.gram;
        let angle = (2. * a[1]).atan2(a[0] - b[1]) / 2.;
        let trace = a[0] + b[1];
        let discriminant = ((a[0] - b[1]).powi(2) + 4. * a[1] * a[1]).sqrt();
        let eig1 = (0.5 * (trace + discriminant)).max(1e-12);
        let eig2 = (0.5 * (trace - discriminant)).max(1e-12);
        let long = (multiplier / eig2.sqrt()).min(0.4);
        let short = (multiplier / eig1.sqrt()).min(0.4);
        coordinates.extend(circle.iter().map(|&[cos, sin]| {
            let u = short * cos;
            let v = long * sin;
            PlotPoint::new(
                position[0] + scales[0] * (angle.cos() * u - angle.sin() * v),
                position[1] + scales[1] * (angle.sin() * u + angle.cos() * v),
            )
        }));
    }
    coordinates
}
fn ellipse_lines<'a>(
    coordinates: &'a [PlotPoint],
    log_strengths: &'a [f64],
    lo: f64,
    hi: f64,
) -> impl Iterator<Item = PlotLine<'a>> {
    coordinates
        .chunks_exact(33)
        .enumerate()
        .map(move |(index, points)| {
            let strength = if hi > lo {
                (log_strengths[index] - lo) / (hi - lo)
            } else {
                0.5
            };
            let color = ellipse_color(strength);
            PlotLine::new(format!("ellipse-{index}"), points)
                .color(color)
                .width(1.5)
        })
}
fn ellipse_color(strength: f64) -> egui::Color32 {
    // A sequential sensitivity scale; lift the dark end so thin glyphs
    // remain visible on the dark plot background.
    let base = nonlinearity_color(strength);
    let lift = |channel: u8| ((u16::from(channel) * 3 + 255 + 2) / 4) as u8;
    egui::Color32::from_rgb(lift(base.r()), lift(base.g()), lift(base.b()))
}
fn network_plot_height(ui: &egui::Ui, base: f32) -> f32 {
    // The four geometry panels should use the available desktop height while
    // remaining usable on smaller windows.
    let extra = ui.ctx().content_rect().height() - 720.;
    base + (extra * if extra < 0. { 0.6 } else { 0.55 }).clamp(-48., 200.)
}

fn show_ellipse_field(
    ui: &mut egui::Ui,
    frame: &network::Frame,
    ranges: &[[f64; 2]; 2],
    axes: [String; 2],
) {
    let (dx, dy) = (
        (ranges[0][1] - ranges[0][0]) / 11.,
        (ranges[1][1] - ranges[1][0]) / 7.,
    );
    let log_strengths = frame
        .grid
        .iter()
        .map(|g| (g.strength + 1e-12).log10())
        .collect::<Vec<_>>();
    let lo = log_strengths.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = log_strengths
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || !hi.is_finite() {
        ui.weak("Geometry unavailable");
        return;
    }
    let max_sensitivity = frame
        .grid
        .iter()
        .map(|g| g.strength)
        .fold(0., f64::max)
        .max(1e-12);
    let coordinates = ellipse_coordinates(&frame.grid, [dx, dy], 0.15 * max_sensitivity);
    let height = network_plot_height(ui, 188.);
    ui.horizontal_top(|ui| {
        EguiPlot::new("network-local-geometry")
            .width((ui.available_width() - 70.).max(120.))
            .height(height)
            .show_grid(false)
            .x_axis_label(&axes[0])
            .y_axis_label(&axes[1])
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max(
                    [ranges[0][0] - dx, ranges[1][0] - dy],
                    [ranges[0][1] + dx, ranges[1][1] + dy],
                ));
                for line in ellipse_lines(&coordinates, &log_strengths, lo, hi) {
                    plot.line(line);
                }
                plot.points(
                    PlotPoints::new("A", vec![frame.slice_anchor])
                        .radius(4.5)
                        .filled(false)
                        .color(crate::theme::SELECTED),
                );
                plot.points(
                    PlotPoints::new("B", vec![frame.slice_endpoint])
                        .radius(5.)
                        .shape(egui_plot::MarkerShape::Diamond)
                        .filled(false)
                        .color(crate::theme::REFERENCE),
                );
                plot.points(
                    PlotPoints::new(
                        if frame.position_on_slice {
                            "xₜ"
                        } else {
                            "xₜ projected"
                        },
                        vec![frame.slice_position],
                    )
                    .radius(4.)
                    .shape(egui_plot::MarkerShape::Cross)
                    .color(crate::theme::CURRENT),
                );
                for (name, position, label, color) in [
                    (
                        "geometry-a-label",
                        frame.slice_anchor,
                        "A",
                        crate::theme::SELECTED,
                    ),
                    (
                        "geometry-b-label",
                        frame.slice_endpoint,
                        "B",
                        crate::theme::REFERENCE,
                    ),
                    (
                        "geometry-x-label",
                        frame.slice_position,
                        if frame.position_on_slice {
                            "x_t"
                        } else {
                            "x_t*"
                        },
                        crate::theme::CURRENT,
                    ),
                ] {
                    plot.text(
                        egui_plot::Text::new(name, position.into(), label)
                            .anchor(egui::Align2::LEFT_BOTTOM)
                            .color(color),
                    );
                }
            });
        ui.vertical(|ui| {
            ui.set_min_width(58.);
            ui.small("log s")
                .on_hover_text("Log sensitivity, scaled within the displayed field.");
            let (bar, _) = ui.allocate_exact_size(
                egui::vec2(12., (height - 34.).max(60.)),
                egui::Sense::hover(),
            );
            for i in 0..100 {
                let y = bar.bottom() - (i + 1) as f32 * bar.height() / 100.;
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(bar.left(), y),
                        egui::vec2(bar.width(), bar.height() / 100. + 1.),
                    ),
                    0.,
                    ellipse_color(if hi > lo { i as f64 / 99. } else { 0.5 }),
                );
            }
            for (value, y) in [(hi, bar.top()), (lo, bar.bottom())] {
                ui.painter().text(
                    egui::pos2(bar.right() + 4., y),
                    egui::Align2::LEFT_CENTER,
                    format!("{value:.2}"),
                    egui::FontId::proportional(11.),
                    crate::theme::MUTED,
                );
            }
        });
    });
}
fn show_ellipsoid(ui: &mut egui::Ui, frame: &network::Frame, yaw: f64, pitch: f64) {
    let Some(metric) = frame.metric_3d else {
        ui.weak("No 3D metric");
        return;
    };
    let matrix = DMatrix::<f64>::from_fn(3, 3, |i, j| metric[i][j]);
    let eigen = SymmetricEigen::new(matrix);
    let scales = (0..3)
        .map(|i| 1. / eigen.eigenvalues[i].max(1e-8).sqrt())
        .collect::<Vec<_>>();
    let smax = scales.iter().copied().fold(0., f64::max).max(1e-12);
    let project = |point: [f64; 3]| {
        let r = std::array::from_fn::<_, 3, _>(|i| {
            (0..3)
                .map(|j| eigen.eigenvectors[(i, j)] * scales[j] / smax * point[j])
                .sum::<f64>()
        });
        let x = yaw.cos() * r[0] - yaw.sin() * r[1];
        let y = yaw.sin() * r[0] + yaw.cos() * r[1];
        [x, pitch.cos() * y - pitch.sin() * r[2]]
    };
    EguiPlot::new("network-ellipsoid")
        .height(network_plot_height(ui, 188.))
        .show_grid(false)
        .data_aspect(1.)
        .show(ui, |plot| {
            for i in 0..8 {
                let phi = std::f64::consts::TAU * i as f64 / 8.;
                let meridian = (0..=48)
                    .map(|j| {
                        let theta = std::f64::consts::PI * j as f64 / 48.;
                        project([
                            theta.sin() * phi.cos(),
                            theta.sin() * phi.sin(),
                            theta.cos(),
                        ])
                    })
                    .collect::<Vec<_>>();
                plot.line(
                    PlotLine::new(format!("meridian-{i}"), meridian)
                        .color(crate::theme::CURRENT.gamma_multiply(0.7)),
                );
            }
            for i in 1..8 {
                let theta = std::f64::consts::PI * i as f64 / 8.;
                let latitude = (0..=64)
                    .map(|j| {
                        let phi = std::f64::consts::TAU * j as f64 / 64.;
                        project([
                            theta.sin() * phi.cos(),
                            theta.sin() * phi.sin(),
                            theta.cos(),
                        ])
                    })
                    .collect::<Vec<_>>();
                plot.line(
                    PlotLine::new(format!("latitude-{i}"), latitude)
                        .color(crate::theme::REFERENCE.gamma_multiply(0.7)),
                );
            }
        });
}
fn show_propagation(ui: &mut egui::Ui, frame: &network::Frame) {
    if ui.available_width() >= 440. {
        show_propagation_table(ui, frame);
        return;
    }
    let linear_color = crate::theme::REFERENCE;
    let actual_color = crate::theme::CURRENT;
    ui.horizontal_wrapped(|ui| {
        let (key, _) = ui.allocate_exact_size(egui::vec2(24., 12.), egui::Sense::hover());
        ui.painter().line_segment(
            [key.left_center(), key.right_center()],
            egui::Stroke::new(2., actual_color),
        );
        ui.small("Actual Δ");
        let (key, _) = ui.allocate_exact_size(egui::vec2(24., 12.), egui::Sense::hover());
        ui.painter().extend(egui::Shape::dashed_line(
            &[key.left_center(), key.right_center()],
            egui::Stroke::new(2., linear_color),
            4.,
            3.,
        ));
        ui.small("Tangent L");
        ui.small(format!("t={:.3}", frame.t));
    });
    let common_max = frame
        .comparisons
        .iter()
        .map(|pair| pair.linear_norm.max(pair.actual_norm))
        .fold(0_f64, f64::max);
    ui.small(format!("Shared norm scale: 0…{common_max:.3}"))
        .on_hover_text("Arrows use independently scaled canonical comparison planes. Bars use the same prepared metric norms on one scale across every layer; compare their lengths to see normalized amplification. Exact norms and E are in Details.");
    let columns = (((ui.available_width() + ui.spacing().item_spacing.x)
        / (112. + ui.spacing().item_spacing.x))
        .floor()
        .max(1.) as usize)
        .min(frame.comparisons.len().max(1));
    for row in frame.comparisons.chunks(columns) {
        ui.columns(columns, |uis| {
            for (pair, ui) in row.iter().zip(uis) {
                ui.strong(&pair.name);
                let width = ui.available_width();
                let (rect, response) =
                    ui.allocate_exact_size(egui::vec2(width, 76.), egui::Sense::hover());
                let origin = rect.center_bottom() - egui::vec2(0., 10.);
                let max = pair.linear_norm.max(pair.actual_norm);
                let length = (width / 2. - 10.).clamp(24., 58.);
                let vector = |p: [f64; 2]| {
                    if max > 0. {
                        egui::vec2((p[0] / max) as f32 * length, (-p[1] / max) as f32 * length)
                    } else {
                        egui::Vec2::ZERO
                    }
                };
                ui.painter().circle_filled(origin, 2.5, crate::theme::MUTED);
                if pair.linear_norm > 0. {
                    let tip = origin + vector(pair.linear);
                    ui.painter().extend(egui::Shape::dashed_line(
                        &[origin, tip],
                        egui::Stroke::new(3.5, linear_color),
                        5.,
                        3.,
                    ));
                    for offset in [egui::vec2(-7., -5.), egui::vec2(-7., 5.)] {
                        ui.painter()
                            .line_segment([tip + offset, tip], egui::Stroke::new(2., linear_color));
                    }
                } else {
                    ui.painter()
                        .circle_stroke(origin, 5., egui::Stroke::new(2., linear_color));
                }
                if pair.actual_norm > 0. && pair.linear_norm > 0. {
                    ui.painter().arrow(
                        origin,
                        vector(pair.actual),
                        egui::Stroke::new(2., actual_color),
                    );
                } else {
                    // No direction is invented when the tangent or actual
                    // displacement is zero. The bars still show exact norms.
                    ui.painter()
                        .circle_stroke(origin, 8., egui::Stroke::new(2., actual_color));
                    ui.painter().text(
                        rect.center_top(),
                        egui::Align2::CENTER_TOP,
                        if pair.linear_norm == 0. && pair.actual_norm == 0. {
                            "Both zero"
                        } else if pair.linear_norm == 0. {
                            "No tangent direction"
                        } else {
                            "Actual Δ = 0"
                        },
                        egui::FontId::proportional(11.),
                        crate::theme::MUTED,
                    );
                }
                response.on_hover_text(format!(
                    "{}\n{}\n||L||={}\n||Δ||={}\n||Δ−L||={}\nE={}\ncos θ={}",
                    pair.name,
                    FINITE_HELP,
                    pair.linear_norm,
                    pair.actual_norm,
                    pair.residual_norm,
                    pair.error,
                    pair.cosine()
                        .map_or_else(|| "undefined".into(), |value| format!("{value:.4}"))
                ));
                let (bars, _) =
                    ui.allocate_exact_size(egui::vec2(width, 16.), egui::Sense::hover());
                for (norm, y, color) in [
                    (pair.linear_norm, bars.top() + 4., linear_color),
                    (pair.actual_norm, bars.top() + 11., actual_color),
                ] {
                    ui.painter().line_segment(
                        [egui::pos2(bars.left(), y), egui::pos2(bars.right(), y)],
                        egui::Stroke::new(3., crate::theme::GRID),
                    );
                    let fraction = if common_max > 0. {
                        (norm / common_max) as f32
                    } else {
                        0.
                    };
                    let end = egui::pos2(bars.left() + width * fraction, y);
                    if color == linear_color {
                        ui.painter().extend(egui::Shape::dashed_line(
                            &[egui::pos2(bars.left(), y), end],
                            egui::Stroke::new(3., color),
                            5.,
                            2.,
                        ));
                    } else {
                        ui.painter().line_segment(
                            [egui::pos2(bars.left(), y), end],
                            egui::Stroke::new(3., color),
                        );
                    }
                }
                ui.small(format!("E={:.4}", pair.error));
                ui.small(
                    pair.cosine()
                        .map_or_else(|| "cos θ —".into(), |value| format!("cos θ {value:.3}")),
                );
                ui.small(format!(
                    "|L| {:.3} · |Δ| {:.3}",
                    pair.linear_norm, pair.actual_norm
                ));
            }
        });
        ui.add_space(6.);
    }
}
fn show_propagation_table(ui: &mut egui::Ui, frame: &network::Frame) {
    let previous_spacing = ui.spacing().item_spacing.y;
    if ui.ctx().content_rect().height() < 720. {
        ui.spacing_mut().item_spacing.y = 2.;
    }
    let width = ui.available_width();
    let rows = frame.comparisons.len().max(1) as f32;
    let compact = ui.ctx().content_rect().height() < 720.;
    let row_height = ((network_plot_height(ui, 188.) - 18. - rows * 14.) / rows)
        .clamp(if compact { 24. } else { 32. }, 72.);
    let layer_width = (width * 0.15).clamp(70., 110.);
    let vector_width = (width * 0.15).clamp(72., 110.);
    let metric_width =
        ((width - layer_width - vector_width - 6. * ui.spacing().item_spacing.x) / 5.).max(44.);
    ui.horizontal(|ui| {
        for (label, width) in [
            ("Layer", layer_width),
            ("Δ / L", vector_width),
            ("||Δ||", metric_width),
            ("||L||", metric_width),
            ("||Δ−L||", metric_width),
            ("cos θ", metric_width),
            ("E", metric_width),
        ] {
            ui.add_sized(
                [width, 18.],
                egui::Label::new(egui::RichText::new(label).small().strong()),
            );
        }
    });
    let scale = frame
        .comparisons
        .iter()
        .map(|p| p.actual_norm.max(p.linear_norm).max(p.residual_norm))
        .fold(0_f64, f64::max);
    for pair in &frame.comparisons {
        ui.separator();
        ui.horizontal(|ui| {
            ui.add_sized([layer_width, row_height], egui::Label::new(&pair.name));
            let (rect, response) = ui.allocate_exact_size(egui::vec2(vector_width, row_height), egui::Sense::hover());
            let center = rect.center();
            let painter = ui.painter();
            painter.line_segment([egui::pos2(rect.left()+5., center.y), egui::pos2(rect.right()-5., center.y)], egui::Stroke::new(1., crate::theme::GRID));
            painter.line_segment([egui::pos2(center.x, rect.top()+3.), egui::pos2(center.x, rect.bottom()-3.)], egui::Stroke::new(1., crate::theme::GRID));
            let max = pair.linear_norm.max(pair.actual_norm);
            let extent = (rect.height() * 0.40).min(rect.width() * 0.36);
            if max > 0. {
                let vector = |p: [f64; 2]| egui::vec2((p[0] / max) as f32 * extent, -(p[1] / max) as f32 * extent);
                if pair.linear_norm > 0. {
                    painter.extend(egui::Shape::dashed_line(&[center, center + vector(pair.linear)], egui::Stroke::new(2., crate::theme::REFERENCE), 4., 2.));
                }
                if pair.actual_norm > 0. && pair.linear_norm > 0. {
                    painter.arrow(center, vector(pair.actual), egui::Stroke::new(2., crate::theme::CURRENT));
                }
            }
            response.on_hover_text("Cyan: actual displacement Δ. Dashed amber: tangent prediction L. Directions use the canonical comparison plane.");
            for (value, color) in [(pair.actual_norm, crate::theme::CURRENT), (pair.linear_norm, crate::theme::REFERENCE), (pair.residual_norm, crate::theme::EMPIRICAL)] {
                ui.allocate_ui_with_layout(egui::vec2(metric_width, row_height), egui::Layout::top_down(egui::Align::Min), |ui| {
                    if compact {
                        ui.spacing_mut().item_spacing.y = 1.;
                    }
                    ui.label(format!("{value:.3}"));
                    let (bar, _) = ui.allocate_exact_size(egui::vec2((metric_width - 4.).max(1.), 4.), egui::Sense::hover());
                    ui.painter().rect_filled(bar, 0., crate::theme::GRID);
                    let fraction = if scale > 0. { (value / scale).clamp(0., 1.) as f32 } else { 0. };
                    ui.painter().rect_filled(egui::Rect::from_min_size(bar.min, egui::vec2(bar.width() * fraction, bar.height())), 0., color);
                });
            }
            ui.add_sized([metric_width, row_height], egui::Label::new(pair.cosine().map_or_else(|| "—".into(), |value| format!("{value:.3}"))));
            ui.add_sized([metric_width, row_height], egui::Label::new(format!("{:.3}", pair.error)));
        });
    }
    ui.spacing_mut().item_spacing.y = previous_spacing;
}
fn nonlinearity_color(error: f64) -> egui::Color32 {
    // Fixed perceptually ordered viridis anchors; invalid cells are rendered separately.
    const STOPS: [[f64; 3]; 5] = [
        [68., 1., 84.],
        [59., 82., 139.],
        [33., 145., 140.],
        [94., 201., 98.],
        [253., 231., 37.],
    ];
    let position = error.clamp(0., 1.) * 4.;
    let index = (position as usize).min(3);
    let (low, high, t) = (STOPS[index], STOPS[index + 1], position - index as f64);
    let rgb: [u8; 3] = std::array::from_fn(|i| (low[i] + (high[i] - low[i]) * t) as u8);
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}
fn show_nonlinearity(ui: &mut egui::Ui, frame: &network::Frame, layer: usize, axes: [String; 2]) {
    let map = &frame.neighborhood;
    let Some(values) = map.errors.get(layer) else {
        ui.weak("Selected layer unavailable");
        return;
    };
    let dx = (map.bounds[0][1] - map.bounds[0][0]) / 15.;
    let dy = (map.bounds[1][1] - map.bounds[1][0]) / 15.;
    if map.invalid_count(layer) > 0 {
        ui.colored_label(
            crate::theme::WARNING,
            format!("{} invalid cells", map.invalid_count(layer)),
        );
    }
    let inside = (0..2).all(|i| {
        frame.slice_position[i] >= map.bounds[i][0] && frame.slice_position[i] <= map.bounds[i][1]
    });
    if !inside || !frame.position_on_slice {
        ui.weak(if !inside {
            "x_t outside neighborhood"
        } else {
            "x_t projected"
        })
        .on_hover_text("Other input coordinates can differ from the displayed slice.");
    }
    let plot_height = network_plot_height(ui, 176.);
    ui.horizontal_top(|ui| {
        EguiPlot::new("network-nonlinearity")
            .width((ui.available_width() - 55.).max(120.))
            .height(plot_height)
            .show_grid(false)
            .grid_spacing(30.0..=60.0)
            .y_grid_spacer(egui_plot::log_grid_spacer(2))
            .x_axis_label(&axes[0])
            .y_axis_label(&axes[1])
            .y_axis_formatter(crate::landscape::axis_tick)
            .show(ui, |plot| {
                plot.set_plot_bounds(PlotBounds::from_min_max(
                    [map.bounds[0][0] - dx / 2., map.bounds[1][0] - dy / 2.],
                    [map.bounds[0][1] + dx / 2., map.bounds[1][1] + dy / 2.],
                ));
                for (i, (&p, value)) in map.positions.iter().zip(values).enumerate() {
                    let corners = vec![
                        [p[0] - dx / 2., p[1] - dy / 2.],
                        [p[0] + dx / 2., p[1] - dy / 2.],
                        [p[0] + dx / 2., p[1] + dy / 2.],
                        [p[0] - dx / 2., p[1] + dy / 2.],
                    ];
                    plot.polygon(
                        egui_plot::Polygon::new(
                            value.map_or_else(
                                || format!("Invalid evaluation · cell {i}"),
                                |value| format!("E={value:.8} · cell {i}"),
                            ),
                            corners,
                        )
                        .fill_color(value.map_or(crate::theme::INSET, nonlinearity_color))
                        .stroke(egui::Stroke::NONE),
                    );
                    if value.is_none() {
                        plot.line(
                            PlotLine::new(
                                format!("invalid-{i}"),
                                vec![
                                    [p[0] - dx / 2., p[1] - dy / 2.],
                                    [p[0] + dx / 2., p[1] + dy / 2.],
                                ],
                            )
                            .color(crate::theme::MUTED),
                        );
                        plot.line(
                            PlotLine::new(
                                format!("invalid-cross-{i}"),
                                vec![
                                    [p[0] - dx / 2., p[1] + dy / 2.],
                                    [p[0] + dx / 2., p[1] - dy / 2.],
                                ],
                            )
                            .color(crate::theme::MUTED),
                        );
                    }
                }
                // Clip only the drawing of the ray, never its numerical input or errors.
                let direction = [
                    frame.slice_endpoint[0] - frame.slice_anchor[0],
                    frame.slice_endpoint[1] - frame.slice_anchor[1],
                ];
                let mut fraction = 1.0_f64;
                for (i, &d) in direction.iter().enumerate() {
                    let edge = if d > 0. {
                        map.bounds[i][1]
                    } else {
                        map.bounds[i][0]
                    };
                    if d != 0. {
                        fraction =
                            fraction.min(0.95 * ((edge - frame.slice_anchor[i]) / d).max(0.));
                    }
                }
                if direction != [0., 0.] {
                    let tip = [
                        frame.slice_anchor[0] + fraction * direction[0],
                        frame.slice_anchor[1] + fraction * direction[1],
                    ];
                    plot.arrows(
                        egui_plot::Arrows::new(
                            "A -> B (slice projection)",
                            vec![frame.slice_anchor],
                            vec![tip],
                        )
                        .color(crate::theme::SELECTED),
                    );
                }
                if inside {
                    plot.points(
                        PlotPoints::new(
                            if frame.position_on_slice {
                                "Current x_t"
                            } else {
                                "Projected x_t (other inputs differ)"
                            },
                            vec![frame.slice_position],
                        )
                        .radius(4.5)
                        .shape(egui_plot::MarkerShape::Diamond)
                        .color(crate::theme::CURRENT),
                    );
                }
                plot.points(
                    PlotPoints::new("A", vec![frame.slice_anchor])
                        .radius(5.)
                        .filled(false)
                        .color(crate::theme::SELECTED),
                );
            });
        ui.vertical(|ui| {
            ui.set_min_width(42.);
            ui.strong("E").on_hover_text(NONLINEARITY_HELP);
            let (bar, _) = ui.allocate_exact_size(
                egui::vec2(12., (plot_height - 34.).max(60.)),
                egui::Sense::hover(),
            );
            for i in 0..100 {
                let y = bar.bottom() - (i + 1) as f32 * bar.height() / 100.;
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(bar.left(), y),
                        egui::vec2(bar.width(), bar.height() / 100. + 1.),
                    ),
                    0.,
                    nonlinearity_color(i as f64 / 99.),
                );
            }
            for (value, y) in [(1., bar.top()), (0.5, bar.center().y), (0., bar.bottom())] {
                ui.painter().text(
                    egui::pos2(bar.right() + 4., y),
                    egui::Align2::LEFT_CENTER,
                    format!("{value:.1}"),
                    egui::FontId::proportional(11.),
                    crate::theme::MUTED,
                );
            }
        });
    });
}
fn path_coordinates(points: &[[f64; 2]]) -> Vec<PlotPoint> {
    if points.len() < 2 {
        return Vec::new();
    }
    points.iter().copied().map(PlotPoint::from).collect()
}
fn path_segments(index: usize, coordinates: &[PlotPoint]) -> impl Iterator<Item = PlotLine<'_>> {
    coordinates.windows(2).enumerate().map(move |(i, pair)| {
        let t = i as f64 / (coordinates.len() - 1) as f64;
        let color = crate::theme::CURRENT.gamma_multiply((0.5 + 0.5 * t) as f32);
        PlotLine::new(format!("segment-{index}-{i}"), pair)
            .color(color)
            .width(2.)
    })
}
fn show_paths(ui: &mut egui::Ui, frame: &network::Frame) {
    if frame.paths.is_empty() {
        ui.weak("PCA unavailable: the fixed probe population has no usable variance.");
        return;
    }
    let min_width = if ui.ctx().content_rect().height() < 720. {
        105.
    } else {
        125.
    };
    let count = (((ui.available_width() + ui.spacing().item_spacing.x)
        / (min_width + ui.spacing().item_spacing.x))
        .floor()
        .max(1.) as usize)
        .min(frame.paths.len());
    for (row_index, paths) in frame.paths.chunks(count).enumerate() {
        ui.columns(count, |columns| {
            for (column, path) in paths.iter().enumerate() {
                let index = row_index * count + column;
                let ui = &mut columns[column];
                ui.strong(&path.layer).on_hover_text(format!(
                    "Independent fixed-reference PCA plane. Reference variance: {:.2}%. Projected path-step energy fidelity: {:.2}%. Exact values and definitions are also in Details.",
                    100. * path.variance,
                    100. * path.projected_energy
                ));
                // Each plot consumes its borrowed segment inputs before the
                // next path's buffer is allocated.
                let coordinates = path_coordinates(&path.points);
                EguiPlot::new(("network-path", index))
                    .height(network_plot_height(ui, 140.))
                    .show_grid(false)
                    .grid_spacing(30.0..=60.0)
                    .y_grid_spacer(egui_plot::log_grid_spacer(2))
                    .x_axis_label("PC 1")
                    .y_axis_label("PC 2")
                    .x_axis_formatter(crate::landscape::axis_tick)
                    .y_axis_formatter(crate::landscape::axis_tick)
                    .show(ui, |plot| {
                        for line in path_segments(index, &coordinates) {
                            plot.line(line);
                        }
                        plot.points(
                            PlotPoints::new("A", vec![path.points[0]])
                                .radius(4.)
                                .filled(false)
                                .color(crate::theme::SELECTED),
                        );
                        plot.points(
                            PlotPoints::new("B", vec![*path.points.last().unwrap()])
                                .radius(4.)
                                .shape(egui_plot::MarkerShape::Diamond)
                                .filled(false)
                                .color(crate::theme::SELECTED),
                        );
                        plot.text(egui_plot::Text::new("endpoint-a-label", path.points[0].into(), "A")
                            .anchor(egui::Align2::LEFT_BOTTOM).color(crate::theme::SELECTED));
                        plot.text(egui_plot::Text::new("endpoint-b-label", (*path.points.last().unwrap()).into(), "B")
                            .anchor(egui::Align2::RIGHT_TOP).color(crate::theme::SELECTED));
                    });
                ui.small(format!("variance {:.3}", path.variance));
            }
        });
        ui.add_space(6.);
    }
}

/// Report an output quantity that has a well-defined meaning for this model
/// family. Output-head statistics are deliberately separate from the common
/// covariance-normalized representation metric.
fn head_sensitivity(
    frame: &network::Frame,
    kind: &ModelKind,
    actions: &ActionSpace,
    mask: Option<&[bool]>,
) -> (String, f64) {
    let values = &frame.output_values;
    let d = &frame.output_tangent;
    let norm = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>().sqrt();
    match kind {
        ModelKind::StochasticPolicy if matches!(actions, ActionSpace::Discrete { .. }) => {
            let valid: Vec<usize> = (0..values.len())
                .filter(|&i| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false)))
                .collect();
            if valid.is_empty() {
                return ("No valid categorical actions".into(), 0.);
            }
            let shift = valid
                .iter()
                .map(|&i| values[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<_> = valid.iter().map(|&i| (values[i] - shift).exp()).collect();
            let z = w.iter().sum::<f64>();
            let mean = valid
                .iter()
                .zip(&w)
                .map(|(&i, &weight)| weight / z * d[i])
                .sum::<f64>();
            let fisher = valid
                .iter()
                .zip(&w)
                .map(|(&i, &weight)| weight / z * (d[i] - mean).powi(2))
                .sum::<f64>()
                .sqrt();
            ("Policy Fisher directional sensitivity".into(), fisher)
        }
        ModelKind::StochasticPolicy => (
            "Latent mean directional sensitivity (noise scale excluded)".into(),
            norm(d),
        ),
        ModelKind::StateGaussianPolicy => (
            "Latent mean directional sensitivity".into(),
            norm(&d[..actions.dimensions().min(d.len())]),
        ),
        ModelKind::DeterministicPolicy => {
            if let ActionSpace::Box { low, high } = actions {
                let derivatives = d
                    .iter()
                    .enumerate()
                    .map(|(i, &value)| {
                        let transformed = values[i].tanh();
                        let scale = (f64::from(high[i]) - f64::from(low[i])) / 2.;
                        value * scale * (1. - transformed * transformed)
                    })
                    .collect::<Vec<_>>();
                ("Action directional sensitivity".into(), norm(&derivatives))
            } else {
                ("Action directional sensitivity".into(), norm(d))
            }
        }
        ModelKind::DiscreteQ => {
            let mut admissible = (0..values.len())
                .filter(|&i| mask.is_none_or(|m| m.get(i).copied().unwrap_or(false)))
                .collect::<Vec<_>>();
            admissible.sort_by(|&i, &j| values[j].total_cmp(&values[i]));
            if admissible.len() >= 2 {
                let (a, b) = (admissible[0], admissible[1]);
                (format!("D_v [Q(a{a}) − Q(a{b})]"), d[a] - d[b])
            } else {
                ("No valid action pair".into(), 0.)
            }
        }
        ModelKind::QuantileQ { grid } => {
            let masses = grid.masses();
            let expected = d.iter().zip(masses).map(|(v, w)| v * f64::from(w)).sum();
            (
                "Raw weighted-head directional derivative (unordered)".into(),
                expected,
            )
        }
        ModelKind::StateValue => ("V(s) directional derivative".into(), d[0]),
        ModelKind::ScalarQ => ("Q(s,a) directional derivative".into(), d[0]),
        ModelKind::MomentQ { .. } => ("Moment directional derivative".into(), d[0]),
        ModelKind::CategoricalQ { .. }
        | ModelKind::CategoricalReturnQ { .. }
        | ModelKind::ResidualQ { .. }
        | ModelKind::Opaque { .. } => ("Unsupported dense output".into(), 0.),
    }
}

#[cfg(test)]
mod cached_input_tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn metadata_reuse_is_exact_and_replacement_validation_cannot_be_skipped() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rl::{
            diagnostics::models::SnapshotBuilder,
            nn::{network::Mlp, policy::Deterministic},
        };
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut builder = SnapshotBuilder::new(
            "metadata",
            RecordContext::new("metadata", 0),
            3,
            space.clone(),
        )
        .unwrap();
        builder
            .deterministic(
                "actor",
                &Deterministic::from_network(Mlp::new(3, &[4], 1, &mut rng).unwrap(), space)
                    .unwrap(),
            )
            .unwrap();
        let snapshot = Arc::new(builder.finish().unwrap());
        let mut pane = NetworkPane::new(egui::Context::default());
        let key = crate::data::CacheKey {
            root: Arc::new("metadata".into()),
            sequence: 1,
        };
        let (first, first_choices) = pane.prepare_metadata(&key, &snapshot).unwrap();
        let (same, same_choices) = pane.prepare_metadata(&key, &snapshot).unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        assert!(Arc::ptr_eq(&first_choices, &same_choices));
        let sibling = crate::data::CacheKey {
            sequence: 2,
            ..key.clone()
        };
        let (other, _) = pane.prepare_metadata(&sibling, &snapshot).unwrap();
        assert!(!Arc::ptr_eq(&first, &other));
        let mut replacement = (*snapshot).clone();
        if let Record::ModelSnapshot(meta) = &mut replacement.record {
            meta.configuration["observation_dim"] = 0.into();
        }
        assert!(
            pane.prepare_metadata(&sibling, &Arc::new(replacement))
                .is_err()
        );
        assert!(pane.metadata.is_none());
        pane.prepare_metadata(&key, &snapshot).unwrap();
        pane.clear();
        assert!(pane.metadata.is_none());
    }

    fn source(rows: usize, fixed_actions: bool) -> Arc<Chunk> {
        Arc::new(Chunk {
            context: RecordContext::new("cached-inputs", 0),
            record: Record::ProbeSet(ProbeSetRecord {
                id: "fixed".into(),
                population: ProbePopulation::FixedReference,
                methodology: "cache equivalence".into(),
                observations: "x".into(),
                action_masks: Some("m".into()),
                fixed_actions: fixed_actions.then(|| "a".into()),
                row_ids: "ids".into(),
                histories: BTreeMap::new(),
            }),
            arrays: BTreeMap::from([
                (
                    "x".into(),
                    Array::f64(
                        vec![rows, 3],
                        (0..rows)
                            .flat_map(|i| [i as f64 + 0.10000000001, -0.20000000001, -0.0])
                            .collect(),
                    )
                    .unwrap(),
                ),
                (
                    "m".into(),
                    Array::u8(vec![rows, 2], vec![1; rows * 2]).unwrap(),
                ),
                (
                    "a".into(),
                    Array::f64(vec![rows, 1], (0..rows).map(|i| i as f64 + 0.125).collect())
                        .unwrap(),
                ),
            ]),
        })
    }

    #[test]
    fn cached_probes_preserve_bounded_order_exact_values_and_fixed_actions() {
        let source = source(network::MAX_PROBES + 4, true);
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let mut pane = NetworkPane::new(egui::Context::default());
        let probes = pane.prepare_probes(&source, 4, 3, &space).unwrap();
        assert_eq!(probes.len(), network::MAX_PROBES);
        for (i, actual) in probes.iter().enumerate() {
            let expected = input_row(&source, i, 4, 3, &space).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(actual[0], f64::from((i as f64 + 0.10000000001) as f32));
            assert_eq!(actual[3], i as f64 + 0.125);
        }
        let cached = pane.prepare_probes(&source, 4, 3, &space).unwrap();
        assert!(Arc::ptr_eq(&probes, &cached));
    }

    #[test]
    fn cached_probes_key_source_allocation_width_and_observation_dimension() {
        let source = source(3, true);
        let replacement = Arc::new((*source).clone());
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let mut pane = NetworkPane::new(egui::Context::default());
        let first = pane.prepare_probes(&source, 4, 3, &space).unwrap();
        let replaced = pane.prepare_probes(&replacement, 4, 3, &space).unwrap();
        assert!(!Arc::ptr_eq(&first, &replaced));
        assert_eq!(*first, *replaced);
        let narrow = pane.prepare_probes(&replacement, 3, 3, &space).unwrap();
        assert!(!Arc::ptr_eq(&replaced, &narrow));
        assert_eq!(narrow[0].len(), 3);
        assert!(pane.prepare_probes(&replacement, 3, 2, &space).is_err());
        assert!(pane.probe_population.is_none());
    }

    #[test]
    fn cached_probes_key_complete_action_space_and_preserve_defaults() {
        let source = source(3, false);
        let mut pane = NetworkPane::new(egui::Context::default());
        let first = pane
            .prepare_probes(
                &source,
                4,
                3,
                &ActionSpace::Box {
                    low: vec![-1.],
                    high: vec![3.],
                },
            )
            .unwrap();
        assert_eq!(first[0][3], 1.);
        let bounds = pane
            .prepare_probes(
                &source,
                4,
                3,
                &ActionSpace::Box {
                    low: vec![-1.],
                    high: vec![5.],
                },
            )
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &bounds));
        assert_eq!(bounds[0][3], 2.);
        let zeros = pane
            .prepare_probes(&source, 4, 3, &ActionSpace::Unbounded { dimensions: 1 })
            .unwrap();
        assert_eq!(zeros[0][3], 0.);
        assert!(
            pane.prepare_probes(&source, 4, 3, &ActionSpace::Unbounded { dimensions: 2 })
                .is_err()
        );
        assert!(pane.probe_population.is_none());
        let discrete = pane
            .prepare_probes(&source, 3, 3, &ActionSpace::Discrete { count: 2 })
            .unwrap();
        let count = pane
            .prepare_probes(&source, 3, 3, &ActionSpace::Discrete { count: 3 })
            .unwrap();
        assert!(!Arc::ptr_eq(&discrete, &count));
        let positive_zero = pane
            .prepare_probes(
                &source,
                3,
                3,
                &ActionSpace::Box {
                    low: vec![0.],
                    high: vec![1.],
                },
            )
            .unwrap();
        let negative_zero = pane
            .prepare_probes(
                &source,
                3,
                3,
                &ActionSpace::Box {
                    low: vec![-0.],
                    high: vec![1.],
                },
            )
            .unwrap();
        assert!(!Arc::ptr_eq(&positive_zero, &negative_zero));
    }

    #[test]
    fn cached_probes_preserve_trajectory_terminal_action_default() {
        let mut source = (*source(3, true)).clone();
        source.record = Record::Trajectory(TrajectoryRecord {
            render: None,
            id: "episode".into(),
            environment: "test".into(),
            observations: "x".into(),
            actions: "a".into(),
            rewards: "r".into(),
            terminated: "t".into(),
            truncated: "u".into(),
            simulation_times: None,
            action_masks: Some("m".into()),
            frames: BTreeMap::new(),
            evaluation: EvaluationMetadata {
                conditioning: Conditioning::InitialScenario,
                initial_action: ActionCondition::CurrentPolicy,
                continuation_policy: "fixture".into(),
                stochastic_policy: false,
                discount: 0.9,
                independent_future_streams: true,
                seed_methodology: "explicit".into(),
                truncation_handling: "terminal".into(),
                complete_returns: true,
            },
        });
        source.arrays.insert(
            "a".into(),
            Array::f64(vec![2, 1], vec![0.125, 1.125]).unwrap(),
        );
        let source = Arc::new(source);
        let space = ActionSpace::Box {
            low: vec![-1.],
            high: vec![5.],
        };
        let mut pane = NetworkPane::new(egui::Context::default());
        let probes = pane.prepare_probes(&source, 4, 3, &space).unwrap();
        assert_eq!(
            probes.iter().map(|r| r[3]).collect::<Vec<_>>(),
            [0.125, 1.125, 2.]
        );
        for (i, actual) in probes.iter().enumerate() {
            assert_eq!(*actual, input_row(&source, i, 4, 3, &space).unwrap());
        }
    }

    #[test]
    fn cached_probes_validate_masks_and_drop_invalid_previous_population() {
        let valid = source(3, false);
        let space = ActionSpace::Unbounded { dimensions: 1 };
        let mut pane = NetworkPane::new(egui::Context::default());
        let old = pane.prepare_probes(&valid, 4, 3, &space).unwrap();
        let mut malformed = (*valid).clone();
        malformed
            .arrays
            .insert("m".into(), Array::u8(vec![1, 2], vec![1, 0]).unwrap());
        assert!(
            pane.prepare_probes(&Arc::new(malformed), 4, 3, &space)
                .is_err()
        );
        assert!(pane.probe_population.is_none());
        let rebuilt = pane.prepare_probes(&valid, 4, 3, &space).unwrap();
        assert!(!Arc::ptr_eq(&old, &rebuilt));
    }

    #[test]
    fn cached_probes_do_not_retain_raw_source_and_clear_releases_population() {
        let source = source(3, false);
        let source_lifetime = Arc::downgrade(&source);
        let mut pane = NetworkPane::new(egui::Context::default());
        let probes = pane
            .prepare_probes(&source, 4, 3, &ActionSpace::Unbounded { dimensions: 1 })
            .unwrap();
        let population_lifetime = Arc::downgrade(&probes);
        drop(probes);
        drop(source);
        assert!(source_lifetime.upgrade().is_none());
        assert!(population_lifetime.upgrade().is_some());
        pane.clear();
        assert!(pane.probe_population.is_none());
        assert!(population_lifetime.upgrade().is_none());
    }
}

#[cfg(test)]
#[path = "network_plot_tests.rs"]
mod plot_tests;
