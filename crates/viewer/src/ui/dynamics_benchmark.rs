//! Matched recorded benchmark paths in the learner's existing spatial domain.
use crate::{
    comparison::Comparison,
    data::{Loader, RunIndex},
    dynamics::{
        Axes, Episode,
        source::{RecordedScenario, TrajectorySource},
    },
    reference::{provider_role, quality_label},
};
use rl::diagnostics::{
    reference::{
        ReferenceKind, ReferenceLifetime, ReferenceTarget, compatible_contexts, manifest_context,
        provider_specification,
    },
    schema::{Array, Chunk, DType, Record, RecordKind},
};

const MAX_CANDIDATES: usize = 16;
const MAX_CHARGE: usize = 2 * 1024 * 1024;
const MAX_STATES: usize = 1025;

#[derive(PartialEq, Eq)]
struct Request {
    role: String,
    revision: u64,
    learner_sequence: u64,
    axes: Axes,
    frontier: (usize, Option<(u64, u64)>),
}

pub(super) struct Path {
    pub points: Vec<[f64; 2]>,
    pub revision: u64,
    pub sequence: u64,
    pub seed: u64,
    pub quality: ReferenceKind,
}

#[derive(Default)]
pub(super) struct Overlay {
    request: Option<Request>,
    checked: usize,
    exhausted: bool,
    pub path: Option<Path>,
}

impl Overlay {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn advance(
        &mut self,
        run: &RunIndex,
        comparison: &Comparison,
        source: &TrajectorySource,
        episode: &Episode,
        axes: Axes,
        loader: &mut Loader,
    ) {
        let role = comparison.reference_role();
        if !provider_role(role)
            || source.role == role
            || comparison
                .reference_run()
                .is_none_or(|reference| reference.root() != run.root())
        {
            self.clear();
            return;
        }
        let revision = comparison.reference_revision();
        let frontier = run.candidate_frontier(RecordKind::Trajectory, role, None, revision);
        let request = Request {
            role: role.into(),
            revision,
            learner_sequence: source.key.sequence,
            axes,
            frontier,
        };
        if self.request.as_ref() != Some(&request) {
            self.clear();
            self.request = Some(request);
        }
        let Some(scenario) = &source.scenario else {
            self.exhausted = true;
            return;
        };
        if self.exhausted || self.path.is_some() {
            return;
        }
        for index in run
            .candidates(RecordKind::Trajectory, role, None, revision)
            .take(MAX_CANDIDATES)
            .skip(self.checked)
        {
            match loader.request_admitted(run, index, MAX_CHARGE) {
                Ok(None) => return,
                Ok(Some(chunk)) => {
                    let sequence = run.recording().entries()[index].sequence;
                    if let Some(path) =
                        matching_path(run, role, sequence, scenario, episode, axes, &chunk)
                    {
                        self.path = Some(path);
                        return;
                    }
                }
                Err(_) => {}
            }
            self.checked += 1;
        }
        self.exhausted = true;
    }

    pub fn status(&self) -> Option<(String, String)> {
        self.request.as_ref()?;
        Some(match &self.path {
            Some(path) => (
                format!(
                    "{} path · k{} · same scenario",
                    quality_label(Some(path.quality)),
                    path.revision
                ),
                format!("Independent recorded rollout #{} from scenario seed {}, drawn in state coordinates.",
                    path.sequence, path.seed),
            ),
            None if self.exhausted => (
                "Benchmark path unavailable".into(),
                "No admitted benchmark matched the selected episode's scenario and evaluation horizon.".into(),
            ),
            None => (
                "Loading benchmark path…".into(),
                "Checking the selected provider's recorded trajectories.".into(),
            ),
        })
    }
}

fn float(array: &Array, index: usize) -> Option<f64> {
    match array.dtype() {
        DType::F32 => array
            .as_f32()
            .ok()?
            .get(index)
            .map(|&value| f64::from(value)),
        DType::F64 => array.as_f64().ok()?.get(index).copied(),
        _ => None,
    }
}

fn matching_path(
    run: &RunIndex,
    role: &str,
    sequence: u64,
    scenario: &RecordedScenario,
    episode: &Episode,
    axes: Axes,
    chunk: &Chunk,
) -> Option<Path> {
    let Record::Trajectory(record) = &chunk.record else {
        return None;
    };
    let identity = &chunk.context.identity;
    let manifest = run.recording().manifest();
    if record.environment != episode.record().environment
        || &RecordedScenario::for_trajectory(chunk).ok().flatten()? != scenario
        || identity.run_id != manifest.run_id
        || identity.model_role != role
        || identity.learner_update_revision
            != run.recording().entries()[run.by_sequence(sequence)?].revision
    {
        return None;
    }
    let spec = provider_specification(chunk).ok().flatten()?;
    let revision = identity.learner_update_revision;
    if spec.model_role() != role
        || !matches!(
            spec.target,
            ReferenceTarget::BenchmarkPolicy | ReferenceTarget::VerifiedOptimal
        )
        || (spec.lifetime == ReferenceLifetime::CheckpointDependent
            && revision != episode.context().identity.learner_update_revision)
        || spec.context.environment != manifest.environment
        || spec.context.observation_dim != manifest.observation_dim
        || spec.context.action_space != manifest.action_space
        || manifest_context(manifest)
            .ok()?
            .is_some_and(|context| compatible_contexts(&context, &spec.context).is_err())
    {
        return None;
    }
    let states = chunk.arrays.get(&record.observations)?;
    let &[rows, width] = states.shape() else {
        return None;
    };
    let initial = episode.step(0).ok()?;
    if !(2..=MAX_STATES).contains(&rows)
        || width != initial.observation.len()
        || [axes.x, axes.y].into_iter().any(|axis| axis >= width)
        || initial
            .observation
            .iter()
            .enumerate()
            .any(|(i, &value)| float(states, i) != Some(value))
    {
        return None;
    }
    let masks = record
        .action_masks
        .as_ref()
        .and_then(|key| chunk.arrays.get(key));
    match (initial.mask, masks) {
        (None, None) => {}
        (Some(mask), Some(values)) if values.as_u8().ok()?.get(..mask.len()) == Some(mask) => {}
        _ => return None,
    }
    let points = (0..rows)
        .map(|row| {
            Some([
                float(states, row * width + axes.x)?,
                float(states, row * width + axes.y)?,
            ])
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Path {
        points,
        revision,
        sequence,
        seed: scenario.seed,
        quality: spec.quality.kind,
    })
}
