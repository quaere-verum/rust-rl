//! One exact episode decision, evaluated on the existing historical worker.
//! Recorded first actions and inferred representative actions stay separate.
use super::*;
use crate::dynamics::{DecisionAction, Episode};
use rl::{
    core::{
        action::{Action, ActionSpace},
        env::Observation,
    },
    diagnostics::{
        schema::*,
        value::{ValueCriticKind, ValueOutput, ValuePrediction},
    },
};
use std::collections::BTreeMap;

pub(super) const MAX_MODEL_READ: usize = 2 * 1024 * 1024;
const MAX_FRAME: usize = 512 * 1024;
#[derive(Clone)]
pub struct Request {
    pub episode_key: CacheKey,
    pub step: usize,
    pub snapshot_key: CacheKey,
    pub observation: Observation,
    pub recorded_action: Option<Action>,
    pub episode_revision: u64,
    pub model_revision: u64,
    role: String,
    space: ActionSpace,
    source: ChunkSource,
    episode_source: ChunkSource,
    metadata_identity: usize,
    metadata_bytes: usize,
}
impl Request {
    pub fn new(
        run: &RunIndex,
        snapshot: usize,
        episode_key: CacheKey,
        episode: &Episode,
        step: usize,
    ) -> Result<Self> {
        if run
            .recording()
            .manifest()
            .metadata
            .get("inference_history")
            .is_some_and(|v| v != "stateless")
        {
            return Err(Error::Config("recurrent memory/history required for inference is not recorded; no hidden state is reconstructed".into()));
        }
        let row = episode.step(step)?;
        if row.action.is_none() {
            return Err(Error::Config(
                "final recorded observation has no live policy decision or recorded Q first action"
                    .into(),
            ));
        }
        if run.root().as_os_str().len() > 4096 || row.mask.is_some_and(|m| m.len() > 4096) {
            return Err(Error::Config(
                "selected source path or legal mask exceeds admission".into(),
            ));
        }
        let exact = |values: &[f64]| -> Result<Vec<f32>> {
            if values.len() > 64
                || values
                    .iter()
                    .any(|&v| !v.is_finite() || f64::from(v as f32) != v)
            {
                return Err(Error::Config(
                    "selected input exceeds width or exact f32 inference encoding".into(),
                ));
            }
            Ok(values.iter().map(|&v| v as f32).collect())
        };
        let observation = Observation::dense(
            exact(row.observation)?,
            row.mask.map(|m| m.iter().map(|&v| v != 0).collect()),
        );
        let recorded_action = match row.action {
            Some(DecisionAction::Discrete(v)) => Some(Action::Discrete(
                usize::try_from(v).map_err(|_| Error::Shape("action ID overflow".into()))?,
            )),
            Some(DecisionAction::Continuous(v)) => Some(Action::Continuous(exact(v)?)),
            None => None,
        };
        let manifest = run.recording().manifest();
        observation.validate(manifest.observation_dim, &manifest.action_space, true)?;
        if let Some(action) = &recorded_action {
            manifest
                .action_space
                .check(action, observation.dense_mask())?;
        }
        let index = run
            .by_sequence(episode_key.sequence)
            .ok_or_else(|| Error::State("selected episode identity missing".into()))?;
        if episode_key.root != *run.root()
            || run.recording().entries()[index].kind != RecordKind::Trajectory
        {
            return Err(Error::Config(
                "selected episode belongs to another source".into(),
            ));
        }
        let episode_entry = &run.recording().entries()[index];
        let actual = &episode.context().identity;
        if actual.run_id != manifest.run_id
            || actual.learner_update_revision != episode_entry.revision
            || actual.model_role != episode_entry.model_role
            || actual.checkpoint_id != episode_entry.checkpoint_id
            || episode.context().trajectory != episode_entry.trajectory
            || episode_entry
                .trajectory
                .as_ref()
                .is_some_and(|id| id != &episode.record().id)
        {
            return Err(Error::Config(
                "selected episode payload does not belong to its immutable source key".into(),
            ));
        }
        let entry = run
            .recording()
            .entries()
            .get(snapshot)
            .ok_or_else(|| Error::State("selected model missing".into()))?;
        if entry.kind != RecordKind::ModelSnapshot {
            return Err(Error::Config(
                "selected prediction requires exact retained weights".into(),
            ));
        }
        if entry.model_role.len() > 1024 {
            return Err(Error::Config(
                "selected model role exceeds metadata admission".into(),
            ));
        }
        let metadata_identity = manifest as *const RunManifest as usize;
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self.0.saturating_add(bytes.len());
                if self.0 > 48 * 1024 {
                    return Err(std::io::Error::other(
                        "selected manifest metadata exceeds admission",
                    ));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut encoded = Count(0);
        serde_json::to_writer(&mut encoded, manifest)?;
        let metadata_bytes = encoded.0.saturating_mul(8).saturating_add(128 * 1024);
        Ok(Self {
            episode_key,
            step,
            snapshot_key: run.key(snapshot)?,
            observation,
            recorded_action,
            episode_revision: episode.context().identity.learner_update_revision,
            model_revision: entry.revision,
            role: entry.model_role.clone(),
            space: manifest.action_space.clone(),
            source: run.recording().source(snapshot)?,
            episode_source: run.recording().source(index)?,
            metadata_identity,
            metadata_bytes,
        })
    }
    fn same(&self, b: &Self) -> bool {
        self.episode_key == b.episode_key
            && self.step == b.step
            && self.snapshot_key == b.snapshot_key
            && self.observation == b.observation
            && self.recorded_action == b.recorded_action
    }
    fn check(&self) -> Result<()> {
        self.source.check_available()?;
        self.episode_source.check_available()
    }
}
pub(super) struct RequestOwner {
    request: Request,
    _lease: prepared::Lease,
    _metadata: prepared::SharedLease,
}
pub struct Frame {
    pub episode_key: CacheKey,
    pub step: usize,
    pub snapshot_key: CacheKey,
    pub episode_revision: u64,
    pub model_revision: u64,
    pub recorded_action: Option<Action>,
    pub inferred_action: Option<Action>,
    pub representative_stochastic: bool,
    pub policy_name: Option<String>,
    pub critics: BTreeMap<String, ValuePrediction>,
    pub critic_summaries: BTreeMap<String, CriticSummary>,
    pub critic_note: Option<String>,
    _owner: Arc<RequestOwner>,
    _lease: prepared::Lease,
}
pub struct CriticSummary {
    pub expected_return: Option<f64>,
    pub details: Option<String>,
}
impl Frame {
    pub fn matches(&self, request: &Request) -> bool {
        self._owner.request.same(request)
    }
}
#[derive(Default)]
pub(super) struct State {
    desired: Option<Arc<RequestOwner>>,
    frame: Option<Arc<Frame>>,
    error: Option<String>,
    running: bool,
    failed: Option<(CacheKey, usize, CacheKey)>,
}
impl State {
    #[cfg(feature = "screenshots")]
    pub(super) fn owners(&self) -> usize {
        usize::from(self.desired.is_some()) + usize::from(self.frame.is_some())
    }
}
impl Interactive {
    pub fn submit_decision(&mut self, request: Request) -> Result<()> {
        let signature = (
            request.episode_key.clone(),
            request.step,
            request.snapshot_key.clone(),
        );
        if self.decision.failed.as_ref() == Some(&signature) {
            return Ok(());
        }
        let result = self.submit_decision_inner(request);
        if let Err(error) = &result {
            self.worker.invalidate();
            self.decision = State {
                error: Some(error.to_string()),
                failed: Some(signature),
                ..State::default()
            };
        }
        result
    }
    fn submit_decision_inner(&mut self, request: Request) -> Result<()> {
        if self
            .decision
            .desired
            .as_ref()
            .is_some_and(|r| r.request.same(&request))
        {
            return Ok(());
        }
        self.cancel_warmup();
        self.cancel_snapshot_validation();
        if self.policy_desired.is_some() || self.policy_frame.is_some() {
            self.invalidate_policy_context();
            self.policy_frame = None;
        }
        if self.value_desired.is_some() || self.value_frame.is_some() {
            self.invalidate_value_context();
            self.value_frame = None;
        }
        let owner = Arc::new(RequestOwner {
            _metadata: prepared::SharedLease::acquire(
                &self.worker.policy_accounting,
                request.metadata_identity,
                request.metadata_bytes,
            )?,
            _lease: prepared::Lease::acquire(
                &self.worker.policy_accounting,
                32 * 1024
                    + request.episode_key.root.as_os_str().len() * 16
                    + request.role.capacity(),
            )?,
            request,
        });
        self.worker.submit_task(Task::Decision(owner.clone()))?;
        self.decision.desired = Some(owner);
        self.decision.failed = None;
        self.decision.error = None;
        self.decision.running = true;
        Ok(())
    }
    pub fn poll_decision(&mut self) -> Result<()> {
        self.worker.poll()?;
        if !self.decision.running {
            return Ok(());
        }
        let Some(answer) = self.worker.answer.take() else {
            return Ok(());
        };
        self.decision.running = false;
        match answer {
            Ok(TaskAnswer::Decision(frame))
                if self
                    .decision
                    .desired
                    .as_ref()
                    .is_some_and(|r| frame.matches(&r.request)) =>
            {
                if let Err(e) = frame._owner.request.check() {
                    self.decision.frame = None;
                    self.decision.error = Some(e.to_string());
                } else {
                    self.decision.frame = Some(frame);
                }
            }
            Err(error) => {
                self.decision.frame = None;
                self.decision.error = Some(error.to_string());
            }
            _ => {}
        }
        Ok(())
    }
    pub fn decision_frame(&self) -> Option<&Frame> {
        self.decision.frame.as_deref()
    }
    pub fn decision_error(&self) -> Option<&str> {
        self.decision.error.as_deref()
    }
    pub fn decision_pending(&self) -> bool {
        self.decision.running
    }
    pub(crate) fn decision_requested_for(
        &self,
        episode: &CacheKey,
        step: usize,
        snapshot: &CacheKey,
    ) -> bool {
        self.decision.desired.as_ref().is_some_and(|owner| {
            &owner.request.episode_key == episode
                && owner.request.step == step
                && &owner.request.snapshot_key == snapshot
        }) || self
            .decision
            .failed
            .as_ref()
            .is_some_and(|(e, t, s)| e == episode && *t == step && s == snapshot)
    }
    pub fn clear_decision(&mut self) {
        if self.decision.running {
            self.worker.invalidate();
        }
        self.decision = State::default();
    }
}
pub(super) fn parameter_charge(config: &SnapshotConfiguration) -> Result<usize> {
    let mut parameters = 0usize;
    for definition in config.models.values() {
        for layout in definition.network.iter().chain(definition.second.iter()) {
            for (pair, &bias) in layout.widths.windows(2).zip(&layout.biases) {
                parameters = pair[0]
                    .checked_mul(pair[1])
                    .and_then(|n| n.checked_add(if bias { pair[1] } else { 0 }))
                    .and_then(|n| n.checked_add(parameters))
                    .ok_or_else(|| Error::Config("selected parameter admission overflow".into()))?;
            }
        }
        if definition.kind.is_stochastic_policy() {
            parameters = parameters
                .checked_add(config.action_space.dimensions())
                .ok_or_else(|| Error::Config("selected parameter admission overflow".into()))?;
        }
    }
    parameters
        .checked_mul(4)
        .ok_or_else(|| Error::Config("selected parameter byte overflow".into()))
}
pub(super) fn execute(
    owner: &Arc<RequestOwner>,
    accounting: &Arc<Accounting>,
    cancelled: impl Fn() -> bool,
    with_bundle: &mut value_prepared::WithBundle<'_>,
) -> Result<Option<Arc<Frame>>> {
    let r = &owner.request;
    r.check()?;
    let lease = prepared::Lease::acquire(accounting, MAX_FRAME)?;
    accounting.scratch(crate::budgets::POLICY_SCRATCH)?;
    let mut answer = None;
    with_bundle(&r.snapshot_key, &r.source, &mut |bundle| {
        let started = std::time::Instant::now();
        if cancelled() {
            return Ok(());
        }
        let config = bundle.configuration();
        if config.models.keys().any(|name| name.len() > 1024) {
            return Err(Error::Config(
                "selected model name exceeds answer metadata admission".into(),
            ));
        }
        let identity = &bundle.context().identity;
        if identity.learner_update_revision != r.model_revision || identity.model_role != r.role {
            return Err(Error::Config(
                "selected prediction model identity differs from exact request".into(),
            ));
        }
        if config.observation_dim != r.observation.dense_values().len()
            || config.action_space != r.space
        {
            return Err(Error::Config(
                "historical selected input encoding differs from recording".into(),
            ));
        }
        let actors = config
            .models
            .values()
            .filter(|d| {
                matches!(
                    d.kind,
                    ModelKind::StochasticPolicy
                        | ModelKind::StateGaussianPolicy
                        | ModelKind::DeterministicPolicy
                )
            })
            .count();
        let discrete = config
            .models
            .values()
            .filter(|d| matches!(d.kind, ModelKind::DiscreteQ))
            .count();
        if actors > 1 || actors == 0 && discrete > 1 {
            return Err(Error::Config("selected snapshot has multiple possible actors; an unambiguous stateless policy adapter is required".into()));
        }
        let policy = config
            .models
            .iter()
            .find(|(_, d)| {
                matches!(
                    d.kind,
                    ModelKind::StochasticPolicy
                        | ModelKind::StateGaussianPolicy
                        | ModelKind::DeterministicPolicy
                )
            })
            .or_else(|| {
                config
                    .models
                    .iter()
                    .find(|(_, d)| matches!(d.kind, ModelKind::DiscreteQ))
            })
            .map(|(name, d)| (name.clone(), d.kind.is_stochastic_policy()));
        let critics = config
            .models
            .iter()
            .filter(|(_, d)| {
                d.kind
                    .critic_kind()
                    .is_some_and(|kind| kind != ValueCriticKind::CategoricalQ)
            })
            .map(|(n, _)| n.clone())
            .collect::<Vec<_>>();
        if critics.len() > 8 {
            return Err(Error::Config(
                "selected critic count exceeds admission".into(),
            ));
        }
        let heads = critics
            .iter()
            .map(|name| {
                let kind = &config.models[name].kind;
                match kind.critic_kind() {
                    Some(ValueCriticKind::QuantileQ) => kind.quantile_grid().unwrap().len(),
                    Some(ValueCriticKind::CategoricalReturnQ) => config.models[name]
                        .network
                        .as_ref()
                        .and_then(|layout| layout.widths.last())
                        .copied()
                        .unwrap_or(0),
                    Some(ValueCriticKind::DiscreteQ) => match config.action_space {
                        ActionSpace::Discrete { count } => count,
                        _ => 0,
                    },
                    Some(ValueCriticKind::MomentQ) => 2,
                    _ => 1,
                }
            })
            .try_fold(0usize, |n, width| {
                n.checked_add(width)
                    .ok_or_else(|| Error::Config("selected head count overflow".into()))
            })?;
        if heads > 4096 {
            return Err(Error::Config(
                "selected total critic heads exceed frame admission".into(),
            ));
        }
        let (inferred_action, raw, critic_note) = if let Some((name, _)) = &policy {
            let request = InspectionRequest::new(name, vec![r.observation.clone()]);
            let result = bundle.inspect(&request)?;
            // Isolated critic inspection enforces disabled learned baselines;
            // keep a supported actor output if a critic is unavailable.
            let (raw, note) = if critics.is_empty() {
                (BTreeMap::new(), None)
            } else {
                match bundle.inspect_critics(&CriticInspectionRequest {
                    critics,
                    observations: vec![r.observation.clone()],
                    fixed_actions: r.recorded_action.clone().map(|a| vec![a]),
                }) {
                    Ok(result) => (result.critics, None),
                    Err(error) => (
                        BTreeMap::new(),
                        Some(error.to_string().chars().take(512).collect()),
                    ),
                }
            };
            (result.actions.into_iter().next(), raw, note)
        } else {
            if critics.is_empty() {
                return Err(Error::Config("no supported stateless prediction adapter; required recurrent memory/history is unavailable".into()));
            }
            let result = bundle.inspect_critics(&CriticInspectionRequest {
                critics,
                observations: vec![r.observation.clone()],
                fixed_actions: r.recorded_action.clone().map(|a| vec![a]),
            })?;
            (None, result.critics, None)
        };
        let mut predictions = BTreeMap::new();
        let unmasked_legal = match config.action_space {
            ActionSpace::Discrete { count }
                if r.observation.dense_mask().is_none() && count <= heads =>
            {
                Some(vec![true; count])
            }
            _ => None,
        };
        let legal = r.observation.dense_mask().or(unmasked_legal.as_deref());
        for (name, chunk) in raw {
            let prediction = ValuePrediction::from_critic(&chunk, 0, legal)?;
            predictions.insert(name, prediction);
        }
        r.check()?;
        accounting
            .inference_micros
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::AcqRel);
        answer = Some((inferred_action, predictions, policy, critic_note));
        Ok(())
    })?;
    let Some((inferred_action, critics, policy, critic_note)) = answer.filter(|_| !cancelled())
    else {
        return Ok(None);
    };
    let mut critic_summaries = BTreeMap::new();
    for (name, prediction) in &critics {
        let selected = match (prediction, &r.recorded_action) {
            (ValuePrediction::Discrete { .. }, Some(Action::Discrete(action))) => {
                Some(prediction.select_discrete(*action)?)
            }
            _ => None,
        };
        let p = selected.as_ref().unwrap_or(prediction);
        let expected_return = p.scalar(&ValueOutput::ExpectedReturn, None).ok();
        let details = match p {
            ValuePrediction::Quantiles { .. } => {
                let q = p.ordered_quantiles()?;
                Some(format!(
                    "{} ordered heads · {} raw crossings",
                    q.values().len(),
                    q.source_crossing_count()
                ))
            }
            ValuePrediction::Moments { .. } => {
                let m = p.moment_summary()?;
                Some(format!(
                    "Raw second moment {} · signed variance {}",
                    m.raw_second, m.variance
                ))
            }
            ValuePrediction::CategoricalReturn { atoms, .. } => {
                Some(format!("{} fixed return atoms", atoms.len()))
            }
            _ => None,
        };
        critic_summaries.insert(
            name.clone(),
            CriticSummary {
                expected_return,
                details,
            },
        );
    }
    Ok(Some(Arc::new(Frame {
        episode_key: r.episode_key.clone(),
        step: r.step,
        snapshot_key: r.snapshot_key.clone(),
        episode_revision: r.episode_revision,
        model_revision: r.model_revision,
        recorded_action: r.recorded_action.clone(),
        inferred_action,
        representative_stochastic: policy.as_ref().is_some_and(|(_, s)| *s),
        policy_name: policy.map(|(n, _)| n),
        critics,
        critic_summaries,
        critic_note,
        _owner: owner.clone(),
        _lease: lease,
    })))
}
