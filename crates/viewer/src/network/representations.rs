//! Bounded preparation of recorded and restored representations.
use crate::{
    data::{RunIndex, SelectedInput},
    plot_data::source_observation,
    policy::prepared::compatible_encoding,
    value::visitation::action_row,
};
use rl::{
    Error, Result,
    diagnostics::{
        activation_inputs::{InputEquality, compare_activation_inputs, validate_activation_inputs},
        models::{InspectionRequest, ModelBundle, ModelKind, ModelLimits},
        representations::{Pca, RepresentationLimits, equal_row, linear_cka, match_reference_sets},
        schema::*,
        visualization::EnvironmentSemantics,
    },
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

const RECORD_BYTES: usize = 4 * 1024 * 1024;
const MODEL_BYTES: usize = 16 * 1024 * 1024;
const MAX_CANDIDATES: usize = 32;
const NORMALIZED_MODEL: &str = "rust-rl/reppo/categorical-q/1";
const LIMITS: RepresentationLimits = RepresentationLimits {
    max_samples: super::MAX_PROBES,
    max_features: 2048,
    max_work_elements: 4 * 1024 * 1024,
};

#[derive(Clone)]
pub struct RepresentationFrame {
    pub revision: u64,
    pub requested_revision: u64,
    pub basis_revision: Option<u64>,
    pub basis_recording: Option<String>,
    pub layer: String,
    pub network: String,
    pub choices: BTreeMap<String, Vec<String>>,
    pub population: String,
    pub methodology: String,
    pub input_lineage: &'static str,
    pub row_ids: Vec<u64>,
    pub width: usize,
    pub source_width: usize,
    pub feature_ids: Vec<u64>,
    pub values: Vec<f64>,
    pub absolute_max: f64,
    pub points: Vec<[f64; 2]>,
    pub reference_points: Option<Vec<[f64; 2]>>,
    pub explained: Vec<f64>,
    pub cka: Option<f64>,
    pub observation_coordinate: Vec<f64>,
    pub actions: Vec<String>,
    pub historical: bool,
    pub historical_fixed_reference: bool,
    pub historical_available: bool,
    pub comparison_note: Option<String>,
    pub projection_note: Option<String>,
    pub(crate) probe_sequence: u64,
    anchor_rows: Option<(u64, Vec<Option<usize>>)>,
}
impl RepresentationFrame {
    pub(crate) fn input(&self, row: usize) -> SelectedInput {
        SelectedInput::ProbeRow {
            sequence: self.probe_sequence,
            set: self.population.clone(),
            row,
        }
    }
    /// Resolve an immutable recorded selection into this population's row order.
    pub fn row_for_input(&self, input: &SelectedInput) -> Option<usize> {
        let SelectedInput::ProbeRow { sequence, set, row } = input else {
            return None;
        };
        if set != &self.population {
            return None;
        }
        if *sequence == self.probe_sequence {
            return (*row < self.row_ids.len()).then_some(*row);
        }
        let (source, map) = self.anchor_rows.as_ref()?;
        (*sequence == *source)
            .then(|| map.get(*row).copied().flatten())
            .flatten()
    }
}

pub(crate) struct Query {
    pub run: RunIndex,
    pub role: String,
    pub population: Option<String>,
    pub anchor_sequence: Option<u64>,
    pub revision: u64,
    pub network: Option<String>,
    pub layer: Option<String>,
    pub reference: Option<(RunIndex, String, u64)>,
    pub historical: bool,
}

pub(crate) fn normalized_model(run: &RunIndex) -> bool {
    run.recording().manifest().model_id == NORMALIZED_MODEL
}

#[derive(PartialEq, Eq, Clone)]
struct Scope {
    run: Arc<std::path::PathBuf>,
    role: String,
    population: Option<String>,
    network: Option<String>,
    layer: Option<String>,
    reference: Option<(Arc<std::path::PathBuf>, String, u64, usize)>,
}
#[derive(PartialEq, Eq)]
struct Key {
    scope: Scope,
    revision: u64,
    frontier: usize,
    exact_snapshot: Option<(u64, bool)>,
    anchor_sequence: Option<u64>,
    historical: bool,
}
impl Query {
    fn key(&self) -> Key {
        let count = |run: &RunIndex, role: &str, revision| {
            run.candidate_frontier(
                RecordKind::Activations,
                role,
                self.population.as_deref(),
                revision,
            )
            .0
        };
        Key {
            scope: Scope {
                run: self.run.root().clone(),
                role: self.role.clone(),
                population: self.population.clone(),
                network: self.network.clone(),
                layer: self.layer.clone(),
                reference: self.reference.as_ref().map(|(run, role, k)| {
                    (run.root().clone(), role.clone(), *k, count(run, role, *k))
                }),
            },
            revision: self.revision,
            frontier: count(&self.run, &self.role, self.revision),
            exact_snapshot: exact_snapshot_index(&self.run, &self.role, self.revision).map(
                |index| {
                    (
                        self.run.recording().entries()[index].sequence,
                        self.run.recording().is_retained(index),
                    )
                },
            ),
            historical: self.historical,
            anchor_sequence: self.anchor_sequence,
        }
    }
}

pub(crate) struct Worker {
    sender: mpsc::SyncSender<Option<(u64, Query)>>,
    receiver: mpsc::Receiver<(u64, Result<RepresentationFrame>)>,
    pending: Option<Query>,
    generation: Arc<AtomicU64>,
    active: bool,
    wanted: Option<Key>,
    pub frame: Option<RepresentationFrame>,
    pub error: Option<String>,
}
impl Worker {
    pub fn new(wake: impl Fn() + Send + 'static) -> Self {
        let (sender, requests) = mpsc::sync_channel::<Option<(u64, Query)>>(1);
        let (answers, receiver) = mpsc::sync_channel(1);
        let generation = Arc::new(AtomicU64::new(0));
        let serial = generation.clone();
        std::thread::spawn(move || {
            let mut basis = None;
            while let Ok(command) = requests.recv() {
                let Some((id, query)) = command else {
                    basis = None;
                    wake();
                    continue;
                };
                let check = || serial.load(Ordering::Acquire) == id;
                let result = prepare(query, &mut basis, &check);
                let result = super::current(&check).and(result);
                if answers.send((id, result)).is_err() {
                    return;
                }
                wake();
            }
        });
        Self {
            sender,
            receiver,
            pending: None,
            generation,
            active: false,
            wanted: None,
            frame: None,
            error: None,
        }
    }
    pub fn clear(&mut self) {
        if self.wanted.take().is_some() {
            self.generation.fetch_add(1, Ordering::AcqRel);
            let _ = self.sender.try_send(None);
        }
        self.pending = None;
        while self.receiver.try_recv().is_ok() {
            self.active = false;
        }
        self.error = None;
        self.frame = None;
    }
    pub fn request(&mut self, query: Query) {
        let key = query.key();
        if self.wanted.as_ref() == Some(&key) {
            return;
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.wanted = Some(key);
        self.pending = Some(query);
        self.frame = None;
        self.error = None;
        self.dispatch();
    }
    pub fn poll(&mut self) {
        while let Ok((id, result)) = self.receiver.try_recv() {
            self.active = false;
            if id == self.generation.load(Ordering::Acquire) {
                match result {
                    Ok(frame) => self.frame = Some(frame),
                    Err(error) => self.error = Some(error.to_string()),
                }
            }
        }
        self.dispatch();
    }
    fn dispatch(&mut self) {
        if self.active {
            return;
        }
        if let Some(query) = self.pending.take() {
            let id = self.generation.load(Ordering::Acquire);
            match self.sender.try_send(Some((id, query))) {
                Ok(()) => self.active = true,
                Err(mpsc::TrySendError::Full(Some((_, query)))) => self.pending = Some(query),
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.error = Some("Representation worker disconnected".into());
                }
                Err(mpsc::TrySendError::Full(None)) => unreachable!(),
            }
        }
    }
    pub fn pending(&self) -> bool {
        self.active || self.pending.is_some()
    }
}

struct Dataset {
    revision: u64,
    probes: Chunk,
    activations: Chunk,
    sequence: u64,
    network: String,
    layer: String,
    rows: usize,
    cols: usize,
    source_width: usize,
    feature_ids: Vec<u64>,
    values: Vec<f64>,
}
impl Dataset {
    fn meta(&self) -> &ActivationRecord {
        let Record::Activations(meta) = &self.activations.record else {
            unreachable!()
        };
        meta
    }
    fn probe(&self) -> &ProbeSetRecord {
        let Record::ProbeSet(meta) = &self.probes.record else {
            unreachable!()
        };
        meta
    }
    fn fixed_population(&self) -> bool {
        matches!(
            self.probe().population,
            ProbePopulation::FixedReference
                | ProbePopulation::UserSupplied
                | ProbePopulation::ImportedHistory
        )
    }
    fn with_activations(mut self, activations: Chunk) -> Result<Self> {
        self.revision = activations.context.identity.learner_update_revision;
        self.activations = activations;
        let key = self.meta().layers.get(&self.layer).ok_or_else(|| {
            Error::Config("Selected representation layer was not recorded".into())
        })?;
        let array = &self.activations.arrays[key];
        ensure_dims(array)?;
        self.rows = array.shape()[0];
        self.cols = array.shape()[1];
        (self.source_width, self.feature_ids) =
            feature_ids(&self.activations, &self.layer, self.cols)?;
        self.values = array.to_f64();
        Ok(self)
    }
}

fn feature_ids(activations: &Chunk, layer: &str, cols: usize) -> Result<(usize, Vec<u64>)> {
    let Record::Activations(meta) = &activations.record else {
        unreachable!()
    };
    let ids = if let Some(key) = meta.selected_neurons.get(layer) {
        activations.arrays[key].as_u64()?.to_vec()
    } else {
        (0..cols as u64).collect()
    };
    Ok((meta.source_widths.get(layer).copied().unwrap_or(cols), ids))
}

fn aligned_anchor_rows(
    run: &RunIndex,
    data: &Dataset,
    sequence: u64,
) -> Option<Vec<Option<usize>>> {
    if !data.fixed_population() || sequence == data.sequence {
        return None;
    }
    let index = run.by_sequence(sequence)?;
    let anchor = load(run, index, RECORD_BYTES).ok()?;
    let Record::ProbeSet(a) = &anchor.record else {
        return None;
    };
    let b = data.probe();
    if a.id != b.id
        || a.population != b.population
        || a.methodology != b.methodology
        || !a.histories.keys().eq(b.histories.keys())
    {
        return None;
    }
    let left_ids = anchor.arrays[&a.row_ids].as_u64().ok()?;
    if left_ids.len() > LIMITS.max_samples {
        return None;
    }
    let right_ids = data.probes.arrays[&b.row_ids].as_u64().ok()?;
    let by_id = right_ids
        .iter()
        .enumerate()
        .map(|(row, &id)| (id, row))
        .collect::<BTreeMap<_, _>>();
    let equal = |left: &str, right: &str, row_a, row_b| {
        equal_row(
            &anchor.arrays[left],
            &data.probes.arrays[right],
            row_a,
            row_b,
        )
    };
    let same_optional = |left: Option<&String>, right: Option<&String>, a, b| match (left, right) {
        (Some(left), Some(right)) => equal(left, right, a, b),
        (None, None) => true,
        _ => false,
    };
    Some(
        left_ids
            .iter()
            .enumerate()
            .map(|(row, id)| {
                let &current = by_id.get(id)?;
                (equal(&a.observations, &b.observations, row, current)
                    && same_optional(
                        a.action_masks.as_ref(),
                        b.action_masks.as_ref(),
                        row,
                        current,
                    )
                    && same_optional(
                        a.fixed_actions.as_ref(),
                        b.fixed_actions.as_ref(),
                        row,
                        current,
                    )
                    && a.histories
                        .iter()
                        .all(|(name, key)| equal(key, &b.histories[name], row, current)))
                .then_some(current)
            })
            .collect(),
    )
}

struct Basis {
    scope: Scope,
    run_name: String,
    dataset: Dataset,
    pca: std::result::Result<Pca, String>,
}

fn ensure_dims(array: &Array) -> Result<()> {
    if matches!(array.shape(), [rows, width] if *rows > 0 && *rows <= LIMITS.max_samples
        && *width > 0 && *width <= LIMITS.max_features)
    {
        Ok(())
    } else {
        Err(Error::Shape(
            "Representation exceeds the admitted rows or features".into(),
        ))
    }
}

fn load(run: &RunIndex, index: usize, budget: usize) -> Result<Chunk> {
    run.recording()
        .source(index)?
        .load_if_within_charge(budget)?
        .ok_or_else(|| Error::Shape("Representation source exceeds the read budget".into()))
}

fn exact_snapshot_index(run: &RunIndex, role: &str, revision: u64) -> Option<usize> {
    run.candidates(RecordKind::ModelSnapshot, role, None, revision)
        .next()
        .filter(|&index| run.recording().entries()[index].revision == revision)
}

fn recorded(
    run: &RunIndex,
    role: &str,
    population: Option<&str>,
    revision: u64,
    network: Option<&str>,
    layer: Option<&str>,
    check: &impl Fn() -> bool,
) -> Result<(Dataset, BTreeMap<String, Vec<String>>)> {
    let entries = run.recording().entries();
    let mut selected = None;
    let mut choices = BTreeMap::new();
    let mut skipped = None;
    for index in run
        .candidates(RecordKind::Activations, role, population, revision)
        .take(MAX_CANDIDATES)
    {
        super::current(check)?;
        let entry = &entries[index];
        if selected
            .as_ref()
            .is_some_and(|(k, _, _, _)| entry.revision < *k)
        {
            break;
        }
        let chunk = match load(run, index, RECORD_BYTES) {
            Ok(chunk) => chunk,
            Err(error) => {
                skipped = Some(error);
                continue;
            }
        };
        let Record::Activations(meta) = &chunk.record else {
            continue;
        };
        let mut layers = meta
            .layers
            .iter()
            .filter(|(_, key)| ensure_dims(&chunk.arrays[*key]).is_ok())
            .map(|(name, _)| name)
            .take(64);
        let names = layers.by_ref().cloned().collect::<Vec<_>>();
        if names.is_empty() {
            continue;
        }
        choices
            .entry(meta.network.clone())
            .or_insert_with(|| names.clone());
        let prefer_normalized = network.is_none() && normalized_model(run);
        let normalized = names.iter().any(|name| name == "encoder.features");
        let use_choice = selected.as_ref().is_none_or(|(_, _, _, earlier): &(_, _, _, Chunk)| {
            prefer_normalized && normalized
                && matches!(&earlier.record, Record::Activations(record) if !record.layers.contains_key("encoder.features"))
        });
        if use_choice
            && network.is_none_or(|name| name == meta.network)
            && let Some(name) = layer.map(str::to_owned).or_else(|| {
                names
                    .iter()
                    .find(|name| name.as_str() == "encoder.features")
                    .or(names.first())
                    .cloned()
            })
            && names.contains(&name)
            && let Some(probe) = run.probe_for_record(index)
        {
            selected = Some((entry.revision, probe, name, chunk));
        }
    }
    let (_, index, layer, activations) = selected.ok_or_else(|| {
        skipped.unwrap_or_else(|| {
            Error::Config("No recorded representations at this checkpoint or earlier".into())
        })
    })?;
    let probes = load(run, index, RECORD_BYTES)?;
    let manifest = run.recording().manifest();
    validate_activation_inputs(
        &activations,
        &probes,
        manifest.observation_dim,
        &manifest.action_space,
    )?;
    let Record::Activations(meta) = &activations.record else {
        unreachable!()
    };
    let array = &activations.arrays[&meta.layers[&layer]];
    let (source_width, feature_ids) = feature_ids(&activations, &layer, array.shape()[1])?;
    let data = Dataset {
        revision: activations.context.identity.learner_update_revision,
        probes,
        sequence: entries[index].sequence,
        network: meta.network.clone(),
        layer,
        rows: array.shape()[0],
        cols: array.shape()[1],
        source_width,
        feature_ids,
        values: array.to_f64(),
        activations,
    };
    Ok((data, choices))
}

fn restore(
    run: &RunIndex,
    role: &str,
    revision: u64,
    data: Dataset,
    reference: Option<(&Dataset, &[usize])>,
    check: &impl Fn() -> bool,
) -> Result<Dataset> {
    let index = exact_snapshot_index(run, role, revision)
        .ok_or_else(|| Error::Config("No retained snapshot at this exact revision".into()))?;
    let snapshot = load(run, index, MODEL_BYTES)?;
    super::current(check)?;
    let bundle = ModelBundle::load(
        &snapshot,
        ModelLimits {
            max_model_bytes: MODEL_BYTES,
            max_batch: LIMITS.max_samples,
            ..Default::default()
        },
    )?;
    let definition = bundle.configuration().models.get(&data.network);
    if !definition.is_some_and(|def| matches!(def.kind, ModelKind::CategoricalQ { .. })) {
        return Err(Error::Config(
            "This model has no normalized historical representation adapter".into(),
        ));
    }
    let policy = bundle
        .configuration()
        .models
        .iter()
        .find(|(_, def)| def.kind.is_stochastic_policy())
        .map(|(name, _)| name)
        .ok_or_else(|| Error::Config("Snapshot lacks its policy model".into()))?;
    let action_dataset = reference.map_or(&data, |(dataset, _)| dataset);
    let ActivationInputs::ObservationsAndActions { actions, .. } = &action_dataset.meta().inputs
    else {
        return Err(Error::Config(
            "Historical representation requires recorded complete actions".into(),
        ));
    };
    let manifest = run.recording().manifest();
    let observations = (0..data.rows)
        .map(|row| source_observation(&data.probes, row))
        .collect::<Result<Vec<_>>>()?;
    let actions = (0..data.rows)
        .map(|row| {
            let action_row_index = reference.map_or(row, |(_, row_map)| row_map[row]);
            action_row(
                &action_dataset.activations,
                actions,
                action_row_index,
                &manifest.action_space,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let mut request = InspectionRequest::new(policy, observations);
    request.critics.push(data.network.clone());
    request
        .layers
        .insert(data.network.clone(), vec![data.layer.clone()]);
    request.fixed_actions = Some(actions);
    super::current(check)?;
    let mut result = bundle.inspect(&request)?;
    let activation = result.activations.remove(&data.network).ok_or_else(|| {
        Error::State("Historical model produced no selected representation".into())
    })?;
    validate_activation_inputs(
        &activation,
        &result.probes,
        manifest.observation_dim,
        &manifest.action_space,
    )?;
    super::current(check)?;
    run.recording().source(index)?.check_available()?;
    data.with_activations(activation)
}

fn projected(pca: &Pca, values: &[f64], rows: usize) -> Result<Vec<[f64; 2]>> {
    let k = pca.components();
    Ok(pca
        .project(values, rows)?
        .chunks_exact(k)
        .map(|row| [row[0], row.get(1).copied().unwrap_or(0.)])
        .collect())
}

fn prepare(
    query: Query,
    basis: &mut Option<Basis>,
    check: &impl Fn() -> bool,
) -> Result<RepresentationFrame> {
    let scope = query.key().scope;
    let (mut data, choices) = recorded(
        &query.run,
        &query.role,
        query.population.as_deref(),
        query.revision,
        query.network.as_deref(),
        query.layer.as_deref(),
        check,
    )?;
    let historical_available = normalized_model(&query.run)
        && data.meta().layers.contains_key("encoder.features")
        && matches!(
            data.meta().inputs,
            ActivationInputs::ObservationsAndActions { .. }
        )
        && exact_snapshot_index(&query.run, &query.role, query.revision)
            .is_some_and(|index| query.run.has_model_record(index));
    let manifest = query.run.recording().manifest();
    if basis.as_ref().is_none_or(|basis| {
        basis.scope != scope
            || basis.dataset.layer != data.layer
            || basis.dataset.network != data.network
            || basis.dataset.probe().id != data.probe().id
    }) {
        *basis = None;
        if let Some((reference, role, revision)) = &query.reference
            && data.fixed_population()
            && compatible_encoding(
                query.run.root() == reference.root(),
                EnvironmentSemantics::new(manifest),
                EnvironmentSemantics::new(reference.recording().manifest()),
            )
            && let Ok((candidate, _)) = recorded(
                reference,
                role,
                Some(&data.probe().id),
                *revision,
                Some(&data.network),
                Some(&data.layer),
                check,
            )
            && candidate.fixed_population()
        {
            super::current(check)?;
            let pca = Pca::fit(&candidate.values, candidate.rows, candidate.cols, 2, LIMITS)
                .map_err(|error| error.to_string());
            *basis = Some(Basis {
                scope,
                run_name: reference.recording().manifest().run_id.clone(),
                dataset: candidate,
                pca,
            });
        }
    }
    let mut historical_fixed_reference = false;
    if query.historical {
        let fixed_inputs = basis.as_ref().and_then(|basis| {
            matches!(
                basis.dataset.meta().inputs,
                ActivationInputs::ObservationsAndActions { .. }
            )
            .then(|| match_reference_sets(&data.probes, &basis.dataset.probes).ok())
            .flatten()
            .map(|row_map| (&basis.dataset, row_map))
        });
        historical_fixed_reference = fixed_inputs.is_some();
        let reference = fixed_inputs
            .as_ref()
            .map(|(dataset, map)| (*dataset, map.as_slice()));
        data = restore(
            &query.run,
            &query.role,
            query.revision,
            data,
            reference,
            check,
        )?;
    }
    let mut points = Vec::new();
    let mut reference_points = None;
    let mut basis_revision = None;
    let mut basis_recording = None;
    let mut explained = Vec::new();
    let mut cka = None;
    let mut comparison_note = None;
    let mut projection_note = None;
    if let Some(basis) = basis.as_ref() {
        let comparison = (|| {
            if data.source_width != basis.dataset.source_width
                || data.feature_ids != basis.dataset.feature_ids
            {
                return Err(Error::Config(
                    "Recorded feature identities or their order differ".into(),
                ));
            }
            let row_map = match_reference_sets(&data.probes, &basis.dataset.probes)?;
            let equality =
                compare_activation_inputs(&data.activations, &basis.dataset.activations, &row_map)?;
            if equality != InputEquality::Equal {
                return Err(Error::Config(equality.label().into()));
            }
            let reference = row_map
                .iter()
                .flat_map(|&row| {
                    let start = row * basis.dataset.cols;
                    basis.dataset.values[start..start + basis.dataset.cols]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            cka = linear_cka(
                &data.values,
                data.cols,
                &reference,
                basis.dataset.cols,
                data.rows,
                LIMITS,
            )
            .ok();
            let pca = basis
                .pca
                .as_ref()
                .map_err(|reason| Error::Config(reason.clone()))?;
            points = projected(pca, &data.values, data.rows)?;
            reference_points = Some(projected(pca, &reference, data.rows)?);
            explained = pca.explained_fraction();
            basis_revision = Some(basis.dataset.revision);
            basis_recording = Some(basis.run_name.clone());
            Ok(())
        })();
        comparison_note = comparison.err().map(|error| error.to_string());
    } else if query.reference.is_some() {
        comparison_note = Some(
            if data.fixed_population() {
                "No compatible recorded representation at the reference checkpoint"
            } else {
                "This population changes with the recorded experience"
            }
            .into(),
        );
    }
    if points.is_empty() {
        match Pca::fit(&data.values, data.rows, data.cols, 2, LIMITS) {
            Ok(pca) => {
                points = projected(&pca, &data.values, data.rows)?;
                explained = pca.explained_fraction();
                basis_revision = Some(data.revision);
                basis_recording = Some(manifest.run_id.clone());
            }
            Err(error) => projection_note = Some(error.to_string()),
        }
    }
    super::current(check)?;
    let probe = data.probe();
    let anchor_rows = query.anchor_sequence.and_then(|sequence| {
        aligned_anchor_rows(&query.run, &data, sequence).map(|rows| (sequence, rows))
    });
    let row_ids = data.probes.arrays[&probe.row_ids].as_u64()?.to_vec();
    let observation_coordinate = (0..data.rows)
        .map(|row| source_observation(&data.probes, row).map(|o| f64::from(o.dense_values()[0])))
        .collect::<Result<Vec<_>>>()?;
    let actions =
        if let ActivationInputs::ObservationsAndActions { actions, .. } = &data.meta().inputs {
            (0..data.rows)
                .map(|row| {
                    action_row(&data.activations, actions, row, &manifest.action_space)
                        .map(|action| format!("{action:?}"))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
    Ok(RepresentationFrame {
        revision: data.revision,
        requested_revision: query.revision,
        basis_revision,
        basis_recording,
        layer: data.layer.clone(),
        network: data.network.clone(),
        choices,
        population: probe.id.clone(),
        methodology: probe.methodology.clone(),
        input_lineage: rl::diagnostics::activation_inputs::input_source(&data.meta().inputs).0,
        row_ids,
        width: data.cols,
        source_width: data.source_width,
        feature_ids: data.feature_ids,
        absolute_max: data
            .values
            .iter()
            .fold(0_f64, |max, value| max.max(value.abs())),
        values: data.values,
        points,
        reference_points,
        explained,
        cka,
        observation_coordinate,
        actions,
        historical: query.historical,
        historical_fixed_reference,
        historical_available,
        comparison_note,
        projection_note,
        probe_sequence: data.sequence,
        anchor_rows,
    })
}
