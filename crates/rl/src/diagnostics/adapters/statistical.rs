//! Diagnostic adapters for explicitly published statistical policies.

#[path = "statistical/retrieval.rs"]
mod retrieval;
#[path = "statistical/snapshot.rs"]
mod snapshot;

pub use retrieval::{
    RestoredRetrieval, RetrievalDiagnostics, RetrievalFeatureMap, restore_retrieval,
    restore_retrieval_with_features,
};

use super::{
    models::ModelLimits,
    probes::ProbeSet,
    schema::{
        ActionCondition, Array, Chunk, CriticRecord, CriticRepresentation, MetricRecord,
        PolicyParameters, PolicyRecord, Record, RecordContext, actions_array,
    },
    session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
    typed::{
        FiniteCandidate, FiniteDecision, FinitePredictions, FiniteValueKind, declare_action_values,
        legal_keys,
    },
    value::{self, ContinuationIdentity, CriticDeclaration, ValueCriticKind, ValueDeclaration},
};
use crate::{
    Error, Result,
    algorithms::{fqi, lspi},
    approx::{
        FeatureMap, FittedQ, StatisticalActor,
        batch::BatchCoverage,
        greedy_index,
        regression::{
            LinearModel, LogisticDiagnostics, Predictor, RidgeDiagnostics, SolveDiagnostics,
        },
    },
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, ObservationBatch, PublicHistory},
        reward::BatchObjective,
        stateful::StatefulActor,
    },
};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;

const FITTED_ARCHITECTURE: &str = "rust-rl/fitted-action-model/1";
const FITTED_MODEL: &str = "fitted_model";
const ACTION_VALUE: &str = "action_value";

/// Optional backend-specific metrics for the last FQI sweep at a publication.
pub trait FqiRegressionDiagnostics {
    fn record_metrics(&self, record: &mut MetricRecord);
}

impl FqiRegressionDiagnostics for () {
    fn record_metrics(&self, _: &mut MetricRecord) {}
}

impl FqiRegressionDiagnostics for RidgeDiagnostics {
    fn record_metrics(&self, record: &mut MetricRecord) {
        metric(
            record,
            "fit.fqi.regression.weighted_residual_norm",
            self.weighted_residual_norm,
        );
        solve_metrics(record, "fit.fqi.augmented_design", &self.solve);
    }
}

impl FqiRegressionDiagnostics for LogisticDiagnostics {
    fn record_metrics(&self, record: &mut MetricRecord) {
        metric(record, "fit.fqi.logistic.objective", self.objective);
        metric(
            record,
            "fit.fqi.logistic.initial_objective",
            self.initial_objective,
        );
        metric(record, "fit.fqi.logistic.gradient_norm", self.gradient_norm);
        metric(
            record,
            "fit.fqi.logistic.scaled_gradient_norm",
            self.scaled_gradient_norm,
        );
        metric(
            record,
            "fit.fqi.logistic.iterations",
            self.iterations as f64,
        );
        metric(
            record,
            "fit.fqi.logistic.backtracking_steps",
            self.backtracking_steps as f64,
        );
    }
}

type FittedEncoder<F, M> = fn(&StatisticalActor<F, M>, usize) -> Result<Vec<u8>>;

/// A completed FQI or LSPI publication borrowed from its owner.
///
/// Observation-only evaluation assumes the feature map is observation-sufficient.
/// Use explicit typed history probes to inspect history-dependent features.
pub struct FittedQDiagnostics<'a, F, M> {
    actor: &'a StatisticalActor<F, M>,
    algorithm: &'static str,
    metrics: MetricRecord,
    fqi: Option<fqi::TargetDescription>,
    continuation: Option<&'a StatisticalActor<F, M>>,
    snapshot: Option<(snapshot::Reconstruction, FittedEncoder<F, M>)>,
}

impl<'a, F: FeatureMap, M: Predictor> FittedQDiagnostics<'a, F, M> {
    pub fn fqi<D: FqiRegressionDiagnostics>(
        actor: &'a StatisticalActor<F, M>,
        sweeps: &[fqi::Sweep<D>],
        coverage: &BatchCoverage,
    ) -> Result<Self> {
        let mut metrics = dataset_metrics(actor.model(), coverage)?;
        metric(&mut metrics, "fit.fqi.sweeps", sweeps.len() as f64);
        if let Some(sweep) = sweeps.last() {
            metric(&mut metrics, "fit.fqi.rows", sweep.rows as f64);
            metric(
                &mut metrics,
                "fit.fqi.excluded_incomplete",
                sweep.excluded_incomplete as f64,
            );
            metric(
                &mut metrics,
                "fit.fqi.targets.minimum",
                sweep.targets.minimum,
            );
            metric(&mut metrics, "fit.fqi.targets.mean", sweep.targets.mean);
            metric(
                &mut metrics,
                "fit.fqi.targets.maximum",
                sweep.targets.maximum,
            );
            sweep.diagnostics.record_metrics(&mut metrics);
        }
        Ok(Self {
            actor,
            algorithm: "FQI",
            metrics,
            fqi: sweeps.last().map(|sweep| sweep.target_description),
            continuation: None,
            snapshot: None,
        })
    }

    /// Declare the planned target operator for an initial FQI publication.
    /// Completed sweeps carry the settings that actually produced the model.
    pub fn with_fqi_config(mut self, config: fqi::FqiConfig<'_>) -> Self {
        if self.algorithm == "FQI" && self.fqi.is_none() {
            self.fqi = Some(config.target_description());
        }
        self
    }

    /// Attach the frozen deployment policy used as the critic's continuation.
    /// For LSPI this may be the policy evaluated by the last solve, before improvement.
    pub fn with_critic_continuation(
        mut self,
        evaluated: &'a StatisticalActor<F, M>,
    ) -> Result<Self> {
        ensure(
            self.actor.model().spaces() == evaluated.model().spaces()
                && self.actor.model().objective() == evaluated.model().objective()
                && self.physical(),
            "incompatible fitted critic continuation",
        )?;
        self.continuation = Some(evaluated);
        Ok(self)
    }

    /// Store the predictor with its feature contract for restoration with a caller-supplied map.
    pub fn with_fitted_model(mut self) -> Self
    where
        M: Serialize,
    {
        self.snapshot = Some((
            snapshot::Reconstruction::NeedsFeatures,
            |actor, max_bytes| {
                super::codec::bounded_json(
                    &FittedPayload {
                        model: actor.model(),
                        epsilon: actor.sampling_epsilon(),
                    },
                    max_bytes,
                )
            },
        ));
        self
    }

    /// Store the predictor and its serializable feature mapping together.
    pub fn with_portable_model(mut self) -> Self
    where
        F: Serialize + DeserializeOwned,
        M: Serialize + DeserializeOwned,
    {
        self.snapshot = Some((
            snapshot::Reconstruction::Portable,
            super::codec::bounded_json,
        ));
        self
    }

    fn physical(&self) -> bool {
        matches!(
            self.actor.model().objective(),
            BatchObjective::Physical { .. }
        )
    }

    fn continuation(&self, context: &RecordContext) -> Option<ContinuationIdentity> {
        if let Some(evaluated) = self.continuation {
            return Some(ContinuationIdentity {
                identity: if std::ptr::eq(self.actor, evaluated) {
                    policy_id(context)
                } else {
                    format!(
                        "{}@{}:fit-continuation",
                        context.identity.run_id,
                        evaluated.revision()
                    )
                },
                stochastic: false,
            });
        }
        (self.physical()
            && self.fqi.as_ref().is_some_and(|target| {
                matches!(target.kind, fqi::TargetKind::Greedy) && target.horizon.get() == 1
            }))
        .then(|| ContinuationIdentity {
            identity: "optimality".into(),
            stochastic: false,
        })
    }

    fn rows(&self, set: &ProbeSet, max_bytes: usize) -> Result<Vec<FiniteRow>> {
        evaluate_rows(
            set,
            &self.actor.model().spaces().actions,
            max_bytes,
            |history| {
                let (actions, values): (Vec<_>, Vec<_>) =
                    self.actor.action_values(history)?.into_iter().unzip();
                let selected = greedy_index(&actions, &values)?;
                let probabilities = self.actor.probabilities(&actions, &values)?;
                Ok(FiniteRow {
                    actions,
                    selected,
                    probabilities,
                    values: Some(values),
                })
            },
        )
    }

    fn critic(
        &self,
        request: &CaptureRequest<'_>,
        set: &ProbeSet,
        rows: &[FiniteRow],
    ) -> Result<Chunk> {
        let ActionSpace::Discrete { count } = set.action_space() else {
            unreachable!()
        };
        let context = probe_context(request.context, set);
        let continuation = self.continuation(&context);
        let mut chunk = Chunk {
            record: Record::Critic(CriticRecord {
                representation: CriticRepresentation::Discrete {
                    values: "values".into(),
                },
                action_condition: ActionCondition::AllDiscreteActions,
                actions: None,
                continuation_policy: continuation.as_ref().map(|c| c.identity.clone()),
                continuation_stochastic: continuation.as_ref().map(|c| c.stochastic),
            }),
            arrays: BTreeMap::from([(
                "values".into(),
                Array::f64(
                    vec![rows.len(), *count],
                    rows.iter()
                        .flat_map(|row| {
                            row.values
                                .as_ref()
                                .expect("fitted predictions")
                                .iter()
                                .copied()
                        })
                        .collect(),
                )?,
            )]),
            context,
        };
        value::attach_declaration(
            &mut chunk,
            &ValueDeclaration {
                version: 1,
                run_id: request.context.identity.run_id.clone(),
                revision: self.actor.revision(),
                model_role: request.context.identity.model_role.clone(),
                conventions: request
                    .conventions
                    .filter(|c| c.discount == self.actor.model().objective().gamma())
                    .cloned(),
                critics: BTreeMap::from([(
                    ACTION_VALUE.into(),
                    CriticDeclaration {
                        kind: ValueCriticKind::DiscreteQ,
                        learned_baseline: None,
                        actor: Some(ContinuationIdentity {
                            identity: policy_id(request.context),
                            stochastic: request.evaluation_mode == ActMode::Sample,
                        }),
                        continuation,
                        twin: None,
                        utility: None,
                    },
                )]),
            },
        )?;
        Ok(chunk)
    }
}

impl<'a, F: FeatureMap> FittedQDiagnostics<'a, F, LinearModel> {
    pub fn lspi(
        actor: &'a StatisticalActor<F, LinearModel>,
        iterations: &[lspi::Iteration],
        coverage: &BatchCoverage,
        stable: bool,
    ) -> Result<Self> {
        let mut metrics = dataset_metrics(actor.model(), coverage)?;
        metric(&mut metrics, "fit.lspi.iterations", iterations.len() as f64);
        if let Some(iteration) = iterations.last() {
            metric(
                &mut metrics,
                "fit.lspi.stable_on_sampled_successors",
                u8::from(stable).into(),
            );
            metric(
                &mut metrics,
                "fit.lspi.changed_sampled_successors",
                iteration.changed_actions as f64,
            );
            solve_metrics(&mut metrics, "fit.lspi.moment_system", &iteration.solve);
        }
        Ok(Self {
            actor,
            algorithm: "LSPI",
            metrics,
            fqi: None,
            continuation: None,
            snapshot: None,
        })
    }
}

impl<F: FeatureMap, M: Predictor> DiagnosticLearner for FittedQDiagnostics<'_, F, M> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        let model = self.actor.model();
        Ok(LearnerInfo {
            algorithm: self.algorithm,
            model: "fitted action predictor",
            configuration: serde_json::json!({
                "objective": model.objective(),
                "features": model.feature_spec().namespace,
                "feature_version": model.feature_spec().version,
                "sampling_epsilon": self.actor.sampling_epsilon(),
                "fqi": self.fqi,
                "evaluation_inputs": "observation roots",
            }),
            discount: Some(model.objective().gamma()),
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.actor.revision()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(root_fitted_actor(self.actor))
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        Ok(self
            .continuation
            .zip(self.continuation(context))
            .map(|(actor, identity)| (root_fitted_actor(actor), identity)))
    }

    fn diagnostic_metrics(&self, context: &RecordContext) -> Result<Option<Chunk>> {
        Ok(Some(metric_chunk(context, self.metrics.clone())))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        if request.predictions {
            for set in request.probes {
                let rows = self.rows(set, request.max_chunk_bytes)?;
                let legacy_discrete = set.typed_inputs().is_none()
                    && matches!(set.action_space(), ActionSpace::Discrete { .. });
                let dense_values = legacy_discrete
                    && self.physical()
                    && matches!(set.action_space(),
                        ActionSpace::Discrete { count } if rows.iter().all(|row| row.actions.len() == *count)
                    );
                if legacy_discrete {
                    chunks.push(dense_policy(request, set, &rows)?);
                }
                if dense_values {
                    chunks.push(self.critic(request, set, &rows)?);
                } else {
                    if set.typed_inputs().is_none() {
                        chunks.push(
                            set.record_as_typed(request.context.clone(), request.max_chunk_bytes)?,
                        );
                    }
                    let physical = self.physical();
                    let kind = if physical {
                        FiniteValueKind::ActionValue
                    } else {
                        FiniteValueKind::PreferenceScore
                    };
                    let mut chunk = typed_policy(request, set, rows, Some(kind))?;
                    if physical {
                        declare_action_values(
                            &mut chunk,
                            request.conventions,
                            Some(ContinuationIdentity {
                                identity: policy_id(request.context),
                                stochastic: request.evaluation_mode == ActMode::Sample,
                            }),
                            self.continuation(request.context),
                        )?;
                    }
                    chunks.push(chunk);
                }
            }
        }
        if let Some((mode, encoder)) = self.snapshot.filter(|_| request.snapshots) {
            chunks.push(snapshot::capture(
                request,
                FITTED_ARCHITECTURE,
                FITTED_MODEL,
                mode,
                |limit| encoder(self.actor, limit),
            )?);
        }
        Ok(chunks)
    }
}

#[derive(Serialize, Deserialize)]
struct FittedPayload<T> {
    model: T,
    epsilon: f64,
}

/// Restore a complete statistical Q publication with its historical feature map.
pub fn restore_fitted_q<F, M>(chunk: &Chunk, limits: ModelLimits) -> Result<StatisticalActor<F, M>>
where
    F: FeatureMap + DeserializeOwned,
    M: Predictor + DeserializeOwned,
{
    let frozen: StatisticalActor<F, M> = snapshot::load(
        chunk,
        FITTED_ARCHITECTURE,
        FITTED_MODEL,
        snapshot::Reconstruction::Portable,
        limits,
    )?;
    ensure(
        frozen.revision() == chunk.context.identity.learner_update_revision,
        "fitted snapshot revision mismatch",
    )?;
    ensure(
        frozen.model().predictor().dimension() <= limits.max_parameters,
        "fitted snapshot exceeds parameter limit",
    )?;
    Ok(frozen)
}

/// Restore a predictor-only publication against its original feature contract.
pub fn restore_fitted_q_with_features<F, M>(
    chunk: &Chunk,
    features: F,
    limits: ModelLimits,
) -> Result<StatisticalActor<F, M>>
where
    F: FeatureMap,
    M: Predictor + DeserializeOwned,
{
    let fitted: FittedPayload<FittedQ<M>> = snapshot::load(
        chunk,
        FITTED_ARCHITECTURE,
        FITTED_MODEL,
        snapshot::Reconstruction::NeedsFeatures,
        limits,
    )?;
    ensure(
        fitted.model.predictor().dimension() <= limits.max_parameters,
        "fitted snapshot exceeds parameter limit",
    )?;
    StatisticalActor::new(
        features,
        fitted.model,
        chunk.context.identity.learner_update_revision,
        fitted.epsilon,
    )
}

fn root_fitted_actor<F: FeatureMap, M: Predictor>(
    actor: &StatisticalActor<F, M>,
) -> Box<dyn Actor + '_> {
    Box::new(RootActor {
        space: &actor.model().spaces().actions,
        decide: move |observation: &Observation, mode, rng: &mut ChaCha8Rng| {
            actor
                .decide(
                    &Some(PublicHistory::new(observation.clone())),
                    observation,
                    mode,
                    rng,
                )
                .map(|decision| decision.action)
        },
    })
}

fn dataset_metrics<M: Predictor>(
    model: &FittedQ<M>,
    coverage: &BatchCoverage,
) -> Result<MetricRecord> {
    ensure(
        coverage.objective == model.objective()
            && coverage.feature_dimension == model.feature_spec().dimension(),
        "diagnostic coverage differs from the fitted model contract",
    )?;
    let mut record = empty_metrics();
    metric(&mut record, "dataset.rows", coverage.rows as f64);
    metric(&mut record, "dataset.episodes", coverage.episodes as f64);
    metric(
        &mut record,
        "dataset.feature_dimension",
        coverage.feature_dimension as f64,
    );
    metric(
        &mut record,
        "dataset.mc_eligible_rows",
        coverage.mc_eligible_rows as f64,
    );
    metric(
        &mut record,
        "dataset.behavior_revisions",
        coverage.behavior_revisions.len() as f64,
    );
    metric(
        &mut record,
        "dataset.actions_offered",
        coverage.actions.len() as f64,
    );
    metric(
        &mut record,
        "dataset.actions_selected",
        coverage
            .actions
            .iter()
            .filter(|action| action.selected_rows > 0)
            .count() as f64,
    );
    if let Some(ess) = coverage.effective_sample_size {
        metric(&mut record, "dataset.weight_effective_sample_size", ess);
    }
    Ok(record)
}

fn solve_metrics(record: &mut MetricRecord, prefix: &str, solve: &SolveDiagnostics) {
    for (suffix, value) in [
        ("rows", solve.rows as f64),
        ("columns", solve.columns as f64),
        ("rank", solve.rank as f64),
        ("absolute_cutoff", solve.absolute_cutoff),
        ("residual_norm", solve.residual_norm),
        ("relative_residual", solve.relative_residual),
    ] {
        metric(record, &format!("{prefix}.{suffix}"), value);
    }
    let name = format!("{prefix}.condition_number");
    if let Some(value) = solve.condition_number {
        metric(record, &name, value);
    } else {
        record.unavailable.insert(
            name,
            "solve did not retain every coefficient direction".into(),
        );
    }
}

fn empty_metrics() -> MetricRecord {
    MetricRecord {
        values: BTreeMap::new(),
        unavailable: BTreeMap::new(),
    }
}

fn metric(record: &mut MetricRecord, name: &str, value: f64) {
    if value.is_finite() {
        record.values.insert(name.into(), value);
    } else {
        record
            .unavailable
            .insert(name.into(), format!("nonfinite: {value}"));
    }
}

fn metric_chunk(context: &RecordContext, record: MetricRecord) -> Chunk {
    Chunk {
        context: context.clone(),
        record: Record::Metrics(record),
        arrays: BTreeMap::new(),
    }
}

fn policy_id(context: &RecordContext) -> String {
    let identity = &context.identity;
    format!("{}@{}", identity.run_id, identity.learner_update_revision)
}

struct RootActor<'a, D> {
    space: &'a ActionSpace,
    decide: D,
}

impl<D> Actor for RootActor<'_, D>
where
    D: Fn(&Observation, ActMode, &mut ChaCha8Rng) -> Result<Action>,
{
    fn action_space(&self) -> &ActionSpace {
        self.space
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        mode: ActMode,
        rng: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            actions.push((self.decide)(observation, mode, rng)?);
        }
        Ok(())
    }
}

struct FiniteRow {
    actions: Vec<Action>,
    selected: usize,
    probabilities: Vec<f64>,
    values: Option<Vec<f64>>,
}

fn evaluate_rows(
    set: &ProbeSet,
    actions: &ActionSpace,
    max_bytes: usize,
    mut predict: impl FnMut(&PublicHistory) -> Result<FiniteRow>,
) -> Result<Vec<FiniteRow>> {
    ensure(
        set.action_space() == actions,
        "statistical model and probe action spaces differ",
    )?;
    let mut bytes = 0usize;
    set.observations()
        .iter()
        .enumerate()
        .map(|(index, observation)| {
            let count = legal_keys(observation, actions)?.len();
            let row_bytes = count.saturating_mul(size_of::<Action>() + 2 * size_of::<f64>());
            bytes = bytes
                .saturating_add(size_of::<FiniteRow>())
                .saturating_add(row_bytes);
            ensure(
                bytes <= max_bytes,
                "statistical probe predictions exceed capture budget",
            )?;
            if let Some(history) = set
                .typed_inputs()
                .and_then(|inputs| inputs[index].public_history())
            {
                predict(history)
            } else {
                predict(&PublicHistory::new(observation.clone()))
            }
        })
        .collect()
}

fn probe_context(context: &RecordContext, set: &ProbeSet) -> RecordContext {
    let mut context = context.clone();
    context.probe_set = Some(set.id().into());
    context
}

fn dense_policy(request: &CaptureRequest<'_>, set: &ProbeSet, rows: &[FiniteRow]) -> Result<Chunk> {
    let ActionSpace::Discrete { count } = set.action_space() else {
        unreachable!()
    };
    let entries = rows
        .len()
        .checked_mul(*count)
        .filter(|len| {
            len.saturating_mul(9)
                .saturating_add(rows.len() * size_of::<u64>())
                <= request.max_chunk_bytes
        })
        .ok_or_else(|| {
            Error::Config("statistical policy prediction exceeds capture budget".into())
        })?;
    let mut actions = Vec::with_capacity(rows.len());
    let mut probabilities = vec![0.0; entries];
    let mut masks = vec![0; entries];
    let mut stochastic = false;
    for (index, row) in rows.iter().enumerate() {
        actions.push(row.actions[row.selected].clone());
        stochastic |= row.probabilities[row.selected] < 1.0;
        for (action, probability) in row.actions.iter().zip(&row.probabilities) {
            let Action::Discrete(action) = action else {
                unreachable!()
            };
            probabilities[index * count + action] = *probability;
            masks[index * count + action] = 1;
        }
    }
    Ok(Chunk {
        context: probe_context(request.context, set),
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            parameters: PolicyParameters::Categorical {
                probabilities: "probabilities".into(),
            },
            entropy: None,
            entropy_method: None,
            masks: Some("masks".into()),
            stochastic,
        }),
        arrays: BTreeMap::from([
            ("actions".into(), actions_array(&actions)?),
            (
                "probabilities".into(),
                Array::f64(vec![rows.len(), *count], probabilities)?,
            ),
            ("masks".into(), Array::u8(vec![rows.len(), *count], masks)?),
        ]),
    })
}

fn typed_policy(
    request: &CaptureRequest<'_>,
    set: &ProbeSet,
    rows: Vec<FiniteRow>,
    value_kind: Option<FiniteValueKind>,
) -> Result<Chunk> {
    let rows = rows
        .into_iter()
        .zip(set.observations())
        .zip(set.row_ids())
        .map(|((row, observation), &row_id)| {
            let selected = observation.finite_action_key(&row.actions[row.selected])?;
            let candidates = row
                .actions
                .iter()
                .zip(row.probabilities)
                .enumerate()
                .map(|(index, (action, probability))| {
                    let deployed_probability = match request.evaluation_mode {
                        ActMode::Sample => probability,
                        ActMode::Deterministic => f64::from(u8::from(index == row.selected)),
                    };
                    Ok(FiniteCandidate {
                        action: observation.finite_action_key(action)?,
                        probability: Some(deployed_probability),
                        value: row.values.as_ref().map(|values| values[index]),
                        visits: None,
                        updates: None,
                    })
                })
                .collect::<Result<_>>()?;
            Ok(FiniteDecision {
                row_id,
                information_state: None,
                selected: Some(selected),
                candidates,
            })
        })
        .collect::<Result<_>>()?;
    FinitePredictions { value_kind, rows }.record(
        request.context.clone(),
        set,
        request.max_chunk_bytes,
    )
}
