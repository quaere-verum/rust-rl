use super::*;
use crate::diagnostics::{
    analysis::QuantileForecast,
    probes::actions_array,
    reference::{
        OptimalDecisionDomain, ReferenceCheckpoint, ReferenceDeclaration, ReferenceInformation,
        ReferenceLifetime, ReferenceTarget, action_gradient_comparison, attach_provider_spec,
        continuous_decision_loss, critic_error, decision_loss, performance_gap,
    },
    schema::{
        ActionCondition, Array, CriticRecord, CriticRepresentation, MetricRecord, PolicyParameters,
        PolicyRecord, decode_actions,
    },
    typed::{FiniteCandidate, FiniteDecision, FinitePredictions, FiniteValueKind, TypedPayload},
    value::{
        self, ContinuationIdentity, CriticDeclaration, ValueCriticKind, ValueDeclaration,
        ValueOutput, ValuePrediction,
    },
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub(super) struct ReferenceCapture<'a> {
    provider: Box<dyn ReferenceProvider + 'a>,
    spec: reference::ReferenceSpec,
    incompatible: Option<String>,
    trajectory_limit: usize,
    fixed_evaluation: Option<EvaluationResult>,
    fixed_probes: BTreeMap<String, FixedPredictions>,
}

struct RegretRows {
    losses: Vec<f64>,
    standard_errors: Option<Vec<f64>>,
}

struct FixedPredictions {
    state_values: Vec<Option<f64>>,
    discrete_action_values: Vec<Vec<Option<f64>>>,
}

impl<'a> ReferenceCapture<'a> {
    pub(super) fn new(
        provider: Box<dyn ReferenceProvider + 'a>,
        experiment: Option<&ReferenceContext>,
        trajectory_limit: usize,
    ) -> Result<Self> {
        let spec = provider.specification().clone();
        spec.validate()?;
        let incompatible = match experiment {
            Some(context) => reference::compatible_contexts(context, &spec.context)
                .err()
                .map(|error| error.to_string()),
            None => Some("The recording has no declared scientific reference context".into()),
        };
        Ok(Self {
            provider,
            spec,
            incompatible,
            trajectory_limit,
            fixed_evaluation: None,
            fixed_probes: BTreeMap::new(),
        })
    }

    pub(super) fn can_evaluate(&self) -> bool {
        self.incompatible.is_none()
            && !matches!(
                self.spec.target,
                ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation
            )
            && self.spec.information == ReferenceInformation::LearnerObservation
            && self.provider.actor().is_some()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "all evidence belongs to one immutable capture boundary"
    )]
    pub(super) fn record(
        &mut self,
        learner: &(impl DiagnosticLearner + ?Sized),
        actor: &dyn Actor,
        context: &RecordContext,
        probes: &[ProbeSet],
        learner_chunks: &[Chunk],
        evaluation: Option<&EvaluationResult>,
        config: &EvaluationConfig,
        horizon: EvaluationHorizon,
        runner: Option<&mut EvaluationRunner<'_>>,
        action_samples: usize,
        recorder: &mut Recorder,
    ) -> Result<()> {
        let mut summary = Summary::default();
        if let Some(reason) = &self.incompatible {
            summary.unavailable("compatibility", reason.clone());
            return self.write_summary(context.clone(), summary, recorder);
        }
        if self.spec.information == ReferenceInformation::PrivilegedSimulatorState {
            summary.unavailable(
                "compatibility",
                "The reference uses privileged simulator information",
            );
            return self.write_summary(context.clone(), summary, recorder);
        }
        let continuation = if self.spec.target == ReferenceTarget::CriticContinuation {
            let Some(continuation) = learner.diagnostic_conditional_actor(context)? else {
                summary.unavailable(
                    "critic_calibration",
                    "The learner supplies no frozen critic continuation",
                );
                return self.write_summary(context.clone(), summary, recorder);
            };
            Some(continuation)
        } else {
            None
        };
        let (policy, policy_id, mode) = if let Some((policy, identity)) = &continuation {
            (
                policy.as_ref(),
                identity.identity.as_str(),
                if identity.stochastic {
                    ActMode::Sample
                } else {
                    ActMode::Deterministic
                },
            )
        } else {
            (actor, config.policy_id.as_str(), config.mode)
        };
        let checkpoint = ReferenceCheckpoint {
            revision: context.identity.learner_update_revision,
            policy_id,
            policy,
            mode,
        };
        self.record_evaluation(
            context,
            evaluation,
            config,
            horizon,
            runner,
            recorder,
            &mut summary,
        )?;
        if probes.is_empty() {
            summary.unavailable("input_population", "No recorded diagnostic inputs");
        }
        self.write_summary(context.clone(), summary, recorder)?;
        for set in probes {
            let mut context = context.clone();
            context.probe_set = Some(set.id().into());
            let mut summary = Summary::default();
            self.record_probes(
                learner,
                &checkpoint,
                &context,
                set,
                learner_chunks,
                action_samples,
                recorder,
                &mut summary,
            )?;
            self.write_summary(context, summary, recorder)?;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "evaluation and recorder are independently optional capabilities"
    )]
    fn record_evaluation(
        &mut self,
        context: &RecordContext,
        evaluation: Option<&EvaluationResult>,
        config: &EvaluationConfig,
        horizon: EvaluationHorizon,
        runner: Option<&mut EvaluationRunner<'_>>,
        recorder: &mut Recorder,
        summary: &mut Summary,
    ) -> Result<()> {
        let (Some(actor), Some(learner), Some(run)) = (self.provider.actor(), evaluation, runner)
        else {
            summary.unavailable(
                "performance_gap",
                "A reference actor and independent learner evaluation are required",
            );
            return Ok(());
        };
        if matches!(
            self.spec.target,
            ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation
        ) {
            summary.unavailable(
                "performance_gap",
                "This provider evaluates the learner's policy",
            );
            return Ok(());
        }
        if let EvaluationHorizon::FiniteDecisions(limit) = horizon
            && !matches!(self.spec.context.horizon, reference::ReferenceHorizon::Finite { decisions } if decisions as usize == limit)
        {
            summary.unavailable(
                "performance_gap",
                "The explicit evaluation horizon differs from the reference horizon",
            );
            return Ok(());
        }
        let is_fixed = self.spec.lifetime == ReferenceLifetime::Fixed;
        let mut fresh = None;
        let reference = if let Some(cached) = &self.fixed_evaluation {
            cached
        } else {
            let mut config = config.clone();
            config.policy_id = self.spec.model_role();
            config.policy_seed ^= 0xeb0d_9a33_19a7_1d71;
            config.mode = self.provider.evaluation_mode(config.mode);
            config.trajectories = self.trajectory_limit.min(learner.trajectories.len());
            let mut result = run(actor, &config, horizon)?;
            let context = self.reference_context(context);
            let mut chunk = result.empirical_record(context.clone(), None)?;
            attach_provider_spec(&mut chunk, &self.spec)?;
            recorder.append(&chunk)?;
            let environment = recorder.manifest().environment.clone();
            for (i, trajectory) in result.trajectories.iter().enumerate() {
                ensure(
                    learner
                        .trajectories
                        .get(i)
                        .is_some_and(|other| other.seed == trajectory.seed),
                    "reference trajectory scenario differs from the recorded learner scenario",
                )?;
                let mut chunk = result.trajectory_record(
                    i,
                    context.clone(),
                    &format!(
                        "{}-{}-{}",
                        self.spec.model_role(),
                        context.identity.learner_update_revision,
                        i
                    ),
                    &environment,
                )?;
                attach_provider_spec(&mut chunk, &self.spec)?;
                recorder.append(&chunk)?;
            }
            // The fixed-reference cache retains returns and scenario provenance.
            result.trajectories = Vec::new();
            fresh.insert(result)
        };
        match performance_gap(learner, reference) {
            Ok(estimate) => {
                summary.value("learner_mean_return", estimate.incumbent_mean);
                summary.value("benchmark_mean_return", estimate.challenger_mean);
                let label = if self.spec.target == ReferenceTarget::VerifiedOptimal
                    && self.spec.context.markov_observation
                {
                    "optimality_gap"
                } else {
                    "benchmark_gap"
                };
                summary.value(label, estimate.mean_difference);
                summary.value("matched_scenarios", estimate.pairs as f64);
                if let Some(se) = estimate.difference_standard_error {
                    summary.value(&format!("{label}.standard_error"), se);
                }
            }
            Err(error) => summary.unavailable("performance_gap", error.to_string()),
        }
        if is_fixed && self.fixed_evaluation.is_none() {
            self.fixed_evaluation = fresh;
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "probe evidence is bound to its capture context"
    )]
    fn record_probes(
        &mut self,
        learner: &(impl DiagnosticLearner + ?Sized),
        checkpoint: &ReferenceCheckpoint<'_>,
        context: &RecordContext,
        probes: &ProbeSet,
        learner_chunks: &[Chunk],
        action_samples: usize,
        recorder: &mut Recorder,
        summary: &mut Summary,
    ) -> Result<()> {
        if let Some(inputs) = probes.typed_inputs() {
            return self.record_typed_probes(
                checkpoint,
                context,
                probes,
                inputs,
                learner_chunks,
                recorder,
                summary,
            );
        }
        let fixed = self.spec.lifetime == ReferenceLifetime::Fixed
            && probes.population() != &ProbePopulation::PolicyVisited;
        let mut prepared = None;
        let new_population = !fixed || !self.fixed_probes.contains_key(probes.id());
        let predictions = if let Some(saved) = self.fixed_probes.get(probes.id()).filter(|_| fixed)
        {
            saved
        } else {
            prepared.insert(self.prepare_rows(checkpoint, probes)?)
        };
        if new_population {
            let reference_context = self.reference_context(context);
            recorder.append(&probes.record(reference_context.clone())?)?;
            if let Some(actor) = self.provider.actor() {
                let policy = policy_record(actor, probes, reference_context.clone())?;
                self.append_declared(policy, "policy", recorder)?;
            }
            if let Some(values) = complete(&predictions.state_values) {
                let state = scalar_record(
                    reference_context.clone(),
                    values,
                    ActionCondition::StateValue,
                    None,
                )?;
                self.append_value(
                    state,
                    "state_value",
                    ValueCriticKind::StateValue,
                    checkpoint,
                    recorder,
                )?;
            }
            if let ActionSpace::Discrete { count } = probes.action_space()
                && !predictions.discrete_action_values.is_empty()
            {
                if let Some(values) = predictions
                    .discrete_action_values
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Option<Vec<_>>>()
                {
                    let chunk = Chunk {
                        context: reference_context.clone(),
                        record: Record::Critic(CriticRecord {
                            representation: CriticRepresentation::Discrete {
                                values: "values".into(),
                            },
                            action_condition: ActionCondition::AllDiscreteActions,
                            actions: None,
                            continuation_policy: None,
                            continuation_stochastic: None,
                        }),
                        arrays: BTreeMap::from([(
                            "values".into(),
                            Array::f64(vec![probes.observations().len(), *count], values)?,
                        )]),
                    };
                    self.append_value(
                        chunk,
                        "action_value",
                        ValueCriticKind::DiscreteQ,
                        checkpoint,
                        recorder,
                    )?;
                } else {
                    // Sparse (row, action, value) triples preserve admitted
                    // legal evidence without inventing values for masked actions.
                    let known: Vec<_> = predictions
                        .discrete_action_values
                        .iter()
                        .enumerate()
                        .flat_map(|(row, actions)| {
                            actions
                                .iter()
                                .enumerate()
                                .filter_map(move |(action, value)| {
                                    value.map(|value| [row as f64, action as f64, value])
                                })
                        })
                        .flatten()
                        .collect();
                    if !known.is_empty() {
                        let count = known.len() / 3;
                        summary.value("recorded_legal_action_values", count as f64);
                        summary.arrays.insert(
                            "reference.recorded_action_value_rows".into(),
                            Array::f64(vec![count, 3], known)?,
                        );
                    }
                }
            }
            if let Some(actions) = probes
                .fixed_actions()
                .filter(|_| !matches!(probes.action_space(), ActionSpace::Discrete { .. }))
            {
                let values = probes
                    .observations()
                    .iter()
                    .zip(actions)
                    .map(|(observation, action)| {
                        self.provider.action_value(checkpoint, observation, action)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if let Some(values) = complete(&values) {
                    let chunk = scalar_record(
                        reference_context.clone(),
                        values,
                        ActionCondition::Fixed,
                        Some(actions),
                    )?;
                    self.append_value(
                        chunk,
                        "action_value",
                        ValueCriticKind::ScalarQ,
                        checkpoint,
                        recorder,
                    )?;
                }
                let forecasts = probes
                    .observations()
                    .iter()
                    .zip(actions)
                    .map(|(observation, action)| {
                        self.provider
                            .return_distribution(checkpoint, observation, action)
                    })
                    .collect::<Result<Vec<_>>>()?;
                if let Some(forecasts) = forecasts.into_iter().collect::<Option<Vec<_>>>()
                    && let Some(chunk) = quantile_record(reference_context, &forecasts, actions)?
                {
                    self.append_value(
                        chunk,
                        "return_distribution",
                        ValueCriticKind::QuantileQ,
                        checkpoint,
                        recorder,
                    )?;
                }
            }
        }
        match self.regret(checkpoint, probes, predictions, learner_chunks, action_samples)? {
            Some(rows) => {
                let statistic = match self.provider.optimal_decision_domain() {
                    Some(OptimalDecisionDomain::ModelCoordinates) => "model_coordinate_decision_regret",
                    _ => "local_decision_regret",
                };
                summary.rows(statistic, rows.losses)?;
                if let Some(errors) = rows.standard_errors {
                    summary.rows(&format!("{statistic}.standard_error"), errors)?;
                }
            }
            None => summary.unavailable("local_decision_regret",
                "Verified optimal values and a supported policy are required for every recorded probe row"),
        }
        self.calibrate(
            learner,
            checkpoint,
            probes,
            predictions,
            learner_chunks,
            summary,
        )?;
        if fixed && new_population {
            self.fixed_probes.insert(
                probes.id().into(),
                prepared.expect("new population captured"),
            );
        }
        Ok(())
    }

    fn prepare_rows(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        set: &ProbeSet,
    ) -> Result<FixedPredictions> {
        let state_values = set
            .observations()
            .iter()
            .map(|obs| self.provider.state_value(checkpoint, obs))
            .collect::<Result<Vec<_>>>()?;
        let mut discrete_action_values = Vec::new();
        if let ActionSpace::Discrete { count } = set.action_space() {
            for obs in set.observations() {
                let row = (0..*count)
                    .map(|i| {
                        if obs.dense_mask().is_none_or(|mask| mask[i]) {
                            self.provider
                                .action_value(checkpoint, obs, &Action::Discrete(i))
                        } else {
                            Ok(None)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                discrete_action_values.push(row);
            }
        }
        Ok(FixedPredictions {
            state_values,
            discrete_action_values,
        })
    }

    fn regret(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        probes: &ProbeSet,
        predictions: &FixedPredictions,
        learner_chunks: &[Chunk],
        samples: usize,
    ) -> Result<Option<RegretRows>> {
        if self.spec.target != ReferenceTarget::VerifiedOptimal
            || self.provider.optimal_decision_domain().is_none()
        {
            return Ok(None);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0x0957_d12a_998b ^ checkpoint.revision);
        if let ActionSpace::Discrete { count } = probes.action_space() {
            let policy = learner_chunks.iter().find(|c| {
                c.context.identity.model_role == "online"
                    && c.context.probe_set.as_deref() == Some(probes.id())
                    && matches!(c.record, Record::Policy(_))
            });
            let Some(policy) = policy else {
                return Ok(None);
            };
            let Record::Policy(schema) = &policy.record else {
                unreachable!()
            };
            let probabilities = match &schema.parameters {
                PolicyParameters::Categorical { probabilities }
                    if checkpoint.mode == ActMode::Sample =>
                {
                    Some(policy.arrays[probabilities].to_f64())
                }
                _ => None,
            };
            let actions = if probabilities.is_none() {
                checkpoint
                    .policy
                    .act(probes.observations(), ActMode::Deterministic, &mut rng)?
            } else {
                Vec::new()
            };
            let mut loss = Vec::new();
            for (row, (obs, q)) in probes
                .observations()
                .iter()
                .zip(&predictions.discrete_action_values)
                .enumerate()
            {
                let legal: Vec<_> = (0..*count)
                    .filter(|&i| obs.dense_mask().is_none_or(|m| m[i]))
                    .collect();
                let Some(q): Option<Vec<_>> = legal.iter().map(|&i| q[i]).collect() else {
                    return Ok(None);
                };
                let all_legal = vec![true; q.len()];
                let result = if let Some(probabilities) = &probabilities {
                    let p = probabilities
                        .get(row * count..(row + 1) * count)
                        .ok_or_else(|| {
                            Error::Shape("reference policy probability row missing".into())
                        })?;
                    ensure(
                        (0..*count).all(|i| legal.contains(&i) || p[i] == 0.),
                        "learner policy places probability on a masked action",
                    )?;
                    let p: Vec<_> = legal.iter().map(|&i| p[i]).collect();
                    decision_loss(&q, &all_legal, reference::DecisionPolicy::Categorical(&p))?
                } else {
                    let Some(Action::Discrete(action)) = actions.get(row) else {
                        return Ok(None);
                    };
                    let Some(action) = legal.iter().position(|i| i == action) else {
                        return Ok(None);
                    };
                    decision_loss(
                        &q,
                        &all_legal,
                        reference::DecisionPolicy::Deterministic(action),
                    )?
                };
                loss.push(result.loss);
            }
            return Ok((!loss.is_empty()).then_some(RegretRows {
                losses: loss,
                standard_errors: None,
            }));
        }
        if !matches!(
            probes.action_space(),
            ActionSpace::Unbounded { .. } | ActionSpace::Box { .. }
        ) {
            return Ok(None);
        }
        let stochastic = checkpoint.mode == ActMode::Sample;
        let mut loss = Vec::new();
        let mut standard_error = Vec::new();
        for (observation, optimal) in probes.observations().iter().zip(&predictions.state_values) {
            let Some(optimal) = *optimal else {
                return Ok(None);
            };
            let mut values = Vec::new();
            for _ in 0..if stochastic { samples } else { 1 } {
                let actions = checkpoint.policy.act(
                    std::slice::from_ref(observation),
                    checkpoint.mode,
                    &mut rng,
                )?;
                let [action] = actions.as_slice() else {
                    return Err(Error::Shape(
                        "learner did not return one action per reference input".into(),
                    ));
                };
                probes
                    .action_space()
                    .check(action, observation.dense_mask())?;
                let Some(value) = self
                    .provider
                    .action_value(checkpoint, observation, action)?
                else {
                    return Ok(None);
                };
                values.push(value);
            }
            let row = continuous_decision_loss(optimal, &values)?;
            loss.push(row.loss);
            standard_error.extend(row.standard_error);
        }
        Ok((!loss.is_empty()).then_some(RegretRows {
            losses: loss,
            standard_errors: stochastic.then_some(standard_error),
        }))
    }

    fn calibrate(
        &self,
        learner: &(impl DiagnosticLearner + ?Sized),
        checkpoint: &ReferenceCheckpoint<'_>,
        probes: &ProbeSet,
        references: &FixedPredictions,
        chunks: &[Chunk],
        summary: &mut Summary,
    ) -> Result<()> {
        let mut compared = false;
        for chunk in chunks.iter().filter(|chunk| {
            chunk.context.identity.model_role == "online"
                && chunk.context.probe_set.as_deref() == Some(probes.id())
                && matches!(chunk.record, Record::Critic(_))
        }) {
            let Some(declaration) = value::declaration(chunk)? else {
                continue;
            };
            let Some((name, critic)) = declaration.critics.first_key_value() else {
                continue;
            };
            let prefix = format!("critic.{name}");
            if let Some(target) = value::return_target(chunk)?
                && !target.is_environmental()
            {
                summary.unavailable(
                    &format!("{prefix}.calibration"),
                    if target.is_soft() {
                        "Entropy-regularized critic requires a matched soft-return reference"
                    } else {
                        "Potential-shaped critic requires a reference in the same return convention"
                    },
                );
                continue;
            }
            if !self.compatible_calibration(checkpoint, &declaration, critic) {
                summary.unavailable(&format!("{prefix}.calibration"),
                    "Reference and learned critic have different or unknown continuation, target or return conventions");
                continue;
            }
            let Record::Critic(record) = &chunk.record else {
                unreachable!()
            };
            let actions = record
                .actions
                .as_ref()
                .map(|key| decode_actions(&chunk.arrays[key], probes.action_space()))
                .transpose()?;
            let mut expected = Vec::new();
            let mut predicted = Vec::new();
            let mut discrete = Vec::new();
            let mut discrete_weights = Vec::new();
            let mut discrete_pointwise = Vec::new();
            let mut distribution_crps = Vec::new();
            let mut gradient_error = Vec::new();
            let mut learned_gradient = Vec::new();
            let mut gradient_cosine = Vec::new();
            let mut gradient_method = None;
            for (row, observation) in probes.observations().iter().enumerate() {
                let mask = if let ActionSpace::Discrete { count } = probes.action_space() {
                    Some(
                        observation
                            .dense_mask()
                            .map_or_else(|| vec![true; *count], |m| m.to_vec()),
                    )
                } else {
                    None
                };
                let prediction = ValuePrediction::from_critic(chunk, row, mask.as_deref())?;
                match (
                    &prediction,
                    actions.as_ref().and_then(|actions| actions.get(row)),
                ) {
                    (ValuePrediction::Discrete { values, legal }, _) => {
                        let pairs = values
                            .iter()
                            .zip(legal)
                            .zip(&references.discrete_action_values[row])
                            .enumerate()
                            .filter(|(_, ((_, legal), _))| **legal)
                            .map(|(action, ((&prediction, _), target))| {
                                target.map(|target| (action, prediction, target))
                            })
                            .collect::<Option<Vec<_>>>();
                        let Some(pairs) = pairs else {
                            break;
                        };
                        let n = pairs.len() as f64;
                        predicted.push(pairs.iter().map(|(_, p, _)| p / n).sum());
                        expected.push(pairs.iter().map(|(_, _, q)| q / n).sum());
                        for (action, prediction, target) in pairs {
                            discrete.push((prediction, target));
                            discrete_weights.push(1. / n);
                            discrete_pointwise.extend([
                                row as f64,
                                action as f64,
                                prediction - target,
                            ]);
                        }
                    }
                    (_, Some(action)) => {
                        let reference_forecast = if matches!(
                            prediction,
                            ValuePrediction::Quantiles { .. }
                                | ValuePrediction::CategoricalReturn { .. }
                        ) {
                            self.provider
                                .return_distribution(checkpoint, observation, action)?
                        } else {
                            None
                        };
                        let reference_forecast =
                            reference_forecast.filter(|forecast| forecast.crossing_count() == 0);
                        let target = self
                            .provider
                            .action_value(checkpoint, observation, action)?
                            .or_else(|| {
                                reference_forecast.as_ref().map(|forecast| forecast.mean())
                            });
                        let Some(target) = target else {
                            break;
                        };
                        predicted.push(prediction.scalar(&ValueOutput::ExpectedReturn, None)?);
                        expected.push(target);
                        if let Some(reference) = reference_forecast {
                            let forecast = prediction.weighted_atoms()?;
                            match forecast.mean_crps_against(&reference.into_weighted_atoms()?) {
                                Ok(score) => distribution_crps.push(score),
                                Err(error) => summary.unavailable(
                                    &format!("{prefix}.reference_distribution_crps"),
                                    error.to_string(),
                                ),
                            }
                        }
                        if let (Some(a), Some(b)) = (
                            learner.diagnostic_action_gradient(name, observation, action)?,
                            self.provider
                                .action_gradient(checkpoint, observation, action)?,
                        ) {
                            let result = action_gradient_comparison(
                                &a.values, a.method, &b.values, b.method,
                            )?;
                            learned_gradient.push(result.magnitude_error);
                            gradient_error.push(result.vector_error);
                            gradient_cosine.extend(result.cosine);
                            gradient_method = Some((a.method, b.method));
                        }
                    }
                    (_, None) => {
                        let Some(target) = references.state_values[row] else {
                            break;
                        };
                        predicted.push(prediction.scalar(&ValueOutput::ExpectedReturn, None)?);
                        expected.push(target);
                    }
                }
            }
            if predicted.len() != probes.observations().len() {
                summary.unavailable(
                    &format!("{prefix}.calibration"),
                    "Reference does not cover every matched input",
                );
                continue;
            }
            compared = true;
            let error = if discrete.is_empty() {
                critic_error(&predicted, &expected, None)?
            } else {
                let (p, q): (Vec<_>, Vec<_>) = discrete.into_iter().unzip();
                let error = critic_error(&p, &q, Some(&discrete_weights))?;
                summary.arrays.insert(
                    format!("reference.{prefix}.legal_action_error"),
                    Array::f64(vec![p.len(), 3], discrete_pointwise)?,
                );
                summary.value(&format!("{prefix}.legal_state_actions"), p.len() as f64);
                error
            };
            summary.value(&format!("{prefix}.bias"), error.bias);
            summary.value(&format!("{prefix}.mae"), error.mae);
            summary.value(&format!("{prefix}.rmse"), error.rmse);
            summary.value(&format!("{prefix}.rows"), predicted.len() as f64);
            summary.rows(
                &format!("{prefix}.pointwise_error"),
                predicted
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| a - b)
                    .collect(),
            )?;
            summary.rows(&format!("{prefix}.reference_value"), expected)?;
            if distribution_crps.len() == probes.observations().len() {
                summary.rows(
                    &format!("{prefix}.reference_distribution_crps"),
                    distribution_crps,
                )?;
            }
            if learned_gradient.len() == probes.observations().len() {
                summary.rows(
                    &format!("{prefix}.action_gradient_vector_error"),
                    gradient_error,
                )?;
                let suffix =
                    gradient_method.map_or(String::new(), |(a, b)| format!(".{a:?}_vs_{b:?}"));
                summary.rows(
                    &format!("{prefix}.action_gradient_magnitude_error{suffix}"),
                    learned_gradient,
                )?;
                if gradient_cosine.len() == probes.observations().len() {
                    summary.rows(&format!("{prefix}.action_gradient_cosine"), gradient_cosine)?;
                }
            }
        }
        if !compared {
            summary.unavailable(
                "critic_calibration",
                "No compatible critic comparison on this population",
            );
        }
        Ok(())
    }

    fn compatible_calibration(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        declaration: &ValueDeclaration,
        critic: &CriticDeclaration,
    ) -> bool {
        if declaration.conventions.as_ref() != Some(&self.spec.context.conventions)
            || critic.learned_baseline == Some(false)
        {
            return false;
        }
        match self.spec.target {
            ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation => {
                critic.continuation.as_ref().is_some_and(|c| {
                    c.identity == checkpoint.policy_id
                        && c.stochastic == (checkpoint.mode == ActMode::Sample)
                })
            }
            ReferenceTarget::VerifiedOptimal => critic
                .continuation
                .as_ref()
                .is_some_and(|c| c.identity == "optimality"),
            ReferenceTarget::BenchmarkPolicy => false,
        }
    }

    fn reference_context(&self, context: &RecordContext) -> RecordContext {
        let mut reference = context.clone();
        reference.identity.model_role = self.spec.model_role();
        reference.identity.checkpoint_id = None;
        reference
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "typed comparison is checkpoint-bound"
    )]
    fn record_typed_probes(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        context: &RecordContext,
        probes: &ProbeSet,
        inputs: &[super::super::typed::DiagnosticInput],
        learner_chunks: &[Chunk],
        recorder: &mut Recorder,
        summary: &mut Summary,
    ) -> Result<()> {
        use crate::diagnostics::reference::DecisionPolicy;
        if !matches!(
            probes.action_space(),
            ActionSpace::Discrete { .. } | ActionSpace::Conditional(_)
        ) {
            summary.unavailable(
                "input_population",
                "Typed reference prediction requires finite legal actions",
            );
            return Ok(());
        }
        let optimal = self.spec.target == ReferenceTarget::VerifiedOptimal
            && self.spec.context.markov_observation
            && self.provider.optimal_decision_domain()
                == Some(OptimalDecisionDomain::MarkovObservation);
        let learner_chunk = learner_chunks.iter().find(|chunk| {
            chunk.record.kind() == crate::diagnostics::schema::RecordKind::FinitePredictions
                && chunk.context.probe_set == context.probe_set
                && chunk.context.identity == context.identity
        });
        let learner = match learner_chunk.map(TypedPayload::read).transpose()? {
            Some(TypedPayload::Predictions(predictions)) => Some(predictions),
            _ => None,
        };
        if let Some(learner) = &learner {
            learner.validate_support(
                probes.row_ids(),
                probes.observations(),
                probes.action_space(),
            )?;
        }
        let compatible_critic = if learner.as_ref().and_then(|frame| frame.value_kind)
            != Some(FiniteValueKind::ActionValue)
            || self.spec.target == ReferenceTarget::VerifiedOptimal && !optimal
        {
            false
        } else {
            learner_chunk
                .map(value::declaration)
                .transpose()?
                .flatten()
                .is_some_and(|declaration| {
                    declaration
                        .critics
                        .get("action_value")
                        .is_some_and(|critic| {
                            critic.kind == ValueCriticKind::DiscreteQ
                                && self.compatible_calibration(checkpoint, &declaration, critic)
                        })
                })
        };
        let mut losses = Vec::new();
        let (mut predicted_values, mut reference_values) = (Vec::new(), Vec::new());
        let mut action_count = 0usize;
        let mut rows = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let observation = input.observation();
            action_count +=
                super::super::typed::legal_keys(observation, probes.action_space())?.len();
            ensure(
                action_count <= super::super::typed::MAX_PREDICTIONS,
                "reference finite predictions exceed the action budget",
            )?;
            let legal = observation.finite_actions(probes.action_space())?;
            let mut candidates = Vec::with_capacity(legal.len());
            for action in &legal {
                candidates.push(FiniteCandidate {
                    action: observation.finite_action_key(action)?,
                    probability: None,
                    value: self
                        .provider
                        .action_value_for_input(checkpoint, input, action)?,
                    visits: None,
                    updates: None,
                });
            }
            let selected = if let Some(actor) = self.provider.actor() {
                let mut rng = ChaCha8Rng::seed_from_u64(probes.row_ids()[index]);
                let actions = actor.act(
                    std::slice::from_ref(observation),
                    ActMode::Deterministic,
                    &mut rng,
                )?;
                ensure(
                    actions.len() == 1,
                    "reference actor must return one action for a typed input",
                )?;
                observation.check_action(probes.action_space(), &actions[0])?;
                Some(observation.finite_action_key(&actions[0])?)
            } else {
                None
            };
            if let Some(learner) = &learner {
                let observed = &learner.rows[index].candidates;
                let learned: BTreeMap<_, _> =
                    observed.iter().map(|row| (row.action, row)).collect();
                if optimal {
                    let q: Option<Vec<_>> = candidates.iter().map(|c| c.value).collect();
                    let p: Option<Vec<_>> = candidates
                        .iter()
                        .map(|c| learned[&c.action].probability)
                        .collect();
                    if let (Some(q), Some(p)) = (q, p)
                        && let Ok(loss) = reference::finite_decision_loss(
                            &q,
                            &vec![true; q.len()],
                            DecisionPolicy::Categorical(&p),
                        )
                    {
                        losses.push(loss.loss);
                    }
                }
                if compatible_critic {
                    for candidate in &candidates {
                        if let (Some(learned), Some(reference)) =
                            (learned[&candidate.action].value, candidate.value)
                        {
                            predicted_values.push(learned);
                            reference_values.push(reference);
                        }
                    }
                }
            }
            rows.push(FiniteDecision {
                row_id: probes.row_ids()[index],
                information_state: None,
                selected,
                candidates,
            });
        }
        if rows
            .iter()
            .any(|row| row.selected.is_some() || row.candidates.iter().any(|c| c.value.is_some()))
        {
            let value_kind = Some(if optimal {
                FiniteValueKind::OptimalActionValue
            } else {
                FiniteValueKind::ActionValue
            });
            let mut chunk = FinitePredictions { value_kind, rows }.record(
                self.reference_context(context),
                probes,
                super::super::typed::MAX_TYPED_BYTES,
            )?;
            attach_provider_spec(&mut chunk, &self.spec)?;
            recorder.append(&chunk)?;
        }
        if !losses.is_empty() {
            let count = losses.len() as f64;
            summary.value(
                "decision_regret",
                losses.iter().map(|loss| loss / count).sum::<f64>(),
            );
            summary.value("decision_regret.rows", count);
        }
        if !predicted_values.is_empty() {
            let error = critic_error(&predicted_values, &reference_values, None)?;
            summary.value("critic.action_value.bias", error.bias);
            summary.value("critic.action_value.mae", error.mae);
            summary.value("critic.action_value.rmse", error.rmse);
            summary.value(
                "critic.action_value.legal_state_actions",
                error.count as f64,
            );
        }
        Ok(())
    }

    fn write_summary(
        &self,
        context: RecordContext,
        summary: Summary,
        recorder: &mut Recorder,
    ) -> Result<()> {
        let mut chunk = Chunk {
            context,
            record: Record::Metrics(summary.metrics),
            arrays: summary.arrays,
        };
        attach_provider_spec(&mut chunk, &self.spec)?;
        recorder.append(&chunk)?;
        Ok(())
    }

    fn append_declared(&self, mut chunk: Chunk, name: &str, recorder: &mut Recorder) -> Result<()> {
        let mut quality = self.spec.quality.clone();
        if matches!(chunk.record, Record::Policy(_)) && quality.assumptions.len() < 16 {
            quality.assumptions.push(
                "The overlay records deterministic representative actions; empirical evidence states its actual evaluation mode.".into()
            );
        }
        let declaration = ReferenceDeclaration {
            version: 1,
            run_id: chunk.context.identity.run_id.clone(),
            revision: chunk.context.identity.learner_update_revision,
            model_role: chunk.context.identity.model_role.clone(),
            context: self.spec.context.clone(),
            sources: BTreeMap::from([(name.to_owned(), quality)]),
            finite_table: None,
        };
        reference::attach_declaration(&mut chunk, &declaration)?;
        attach_provider_spec(&mut chunk, &self.spec)?;
        recorder.append(&chunk)?;
        Ok(())
    }

    fn append_value(
        &self,
        mut chunk: Chunk,
        name: &str,
        kind: ValueCriticKind,
        checkpoint: &ReferenceCheckpoint<'_>,
        recorder: &mut Recorder,
    ) -> Result<()> {
        let continuation = match self.spec.target {
            ReferenceTarget::VerifiedOptimal => ContinuationIdentity {
                identity: "optimality".into(),
                stochastic: false,
            },
            ReferenceTarget::BenchmarkPolicy => ContinuationIdentity {
                identity: self.spec.model_role(),
                stochastic: self.provider.evaluation_mode(checkpoint.mode) == ActMode::Sample,
            },
            ReferenceTarget::LearnerPolicy | ReferenceTarget::CriticContinuation => {
                ContinuationIdentity {
                    identity: checkpoint.policy_id.into(),
                    stochastic: checkpoint.mode == ActMode::Sample,
                }
            }
        };
        let declaration = ValueDeclaration {
            version: 1,
            run_id: chunk.context.identity.run_id.clone(),
            revision: chunk.context.identity.learner_update_revision,
            model_role: chunk.context.identity.model_role.clone(),
            conventions: Some(self.spec.context.conventions.clone()),
            critics: BTreeMap::from([(
                name.to_owned(),
                CriticDeclaration {
                    kind,
                    actor: None,
                    continuation: Some(continuation),
                    twin: None,
                    utility: None,
                    learned_baseline: None,
                },
            )]),
        };
        value::attach_declaration(&mut chunk, &declaration)?;
        self.append_declared(chunk, name, recorder)
    }
}

fn complete(values: &[Option<f64>]) -> Option<Vec<f64>> {
    values.iter().copied().collect()
}

fn policy_record(
    actor: &dyn Actor,
    probes: &ProbeSet,
    mut context: RecordContext,
) -> Result<Chunk> {
    ensure(
        actor.action_space() == probes.action_space(),
        "benchmark actor has a different action space",
    )?;
    let mut rng = ChaCha8Rng::seed_from_u64(0x59a7_83ce);
    let actions = actor.act(probes.observations(), ActMode::Deterministic, &mut rng)?;
    ensure(
        actions.len() == probes.observations().len(),
        "benchmark actor did not return one action per input",
    )?;
    for (action, observation) in actions.iter().zip(probes.observations()) {
        probes
            .action_space()
            .check(action, observation.dense_mask())?;
    }
    context.probe_set = Some(probes.id().into());
    let mut arrays = BTreeMap::from([("actions".into(), actions_array(&actions)?)]);
    let masks = if let ActionSpace::Discrete { count } = probes.action_space() {
        arrays.insert(
            "action_masks".into(),
            Array::u8(
                vec![actions.len(), *count],
                probes
                    .observations()
                    .iter()
                    .flat_map(|o| {
                        (0..*count).map(|i| u8::from(o.dense_mask().is_none_or(|m| m[i])))
                    })
                    .collect(),
            )?,
        );
        Some("action_masks".into())
    } else {
        None
    };
    Ok(Chunk {
        context,
        arrays,
        record: Record::Policy(PolicyRecord {
            actions: "actions".into(),
            masks,
            stochastic: false,
            parameters: PolicyParameters::Deterministic,
            entropy: None,
            entropy_method: None,
        }),
    })
}

fn scalar_record(
    context: RecordContext,
    values: Vec<f64>,
    condition: ActionCondition,
    actions: Option<&[Action]>,
) -> Result<Chunk> {
    let mut arrays = BTreeMap::new();
    if !values.is_empty() {
        arrays.insert("values".into(), Array::f64(vec![values.len()], values)?);
    }
    if let Some(actions) = actions {
        arrays.insert("actions".into(), actions_array(actions)?);
    }
    Ok(Chunk {
        context,
        arrays,
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Scalar {
                values: "values".into(),
            },
            action_condition: condition,
            actions: actions.map(|_| "actions".into()),
            continuation_policy: None,
            continuation_stochastic: None,
        }),
    })
}

fn quantile_record(
    context: RecordContext,
    forecasts: &[QuantileForecast],
    actions: &[Action],
) -> Result<Option<Chunk>> {
    let Some(first) = forecasts.first() else {
        return Ok(None);
    };
    let locations = first.locations();
    let masses = first.masses();
    if !forecasts.iter().all(|forecast| {
        forecast.locations() == locations
            && forecast.masses() == masses
            && forecast.crossing_count() == 0
    }) {
        return Ok(None);
    }
    ensure(
        locations.len() <= value::MAX_VALUE_HEADS
            && locations.len().saturating_mul(forecasts.len()) <= 1_048_576,
        "reference return distribution exceeds the diagnostic forecast budget",
    )?;
    Ok(Some(Chunk {
        context,
        record: Record::Critic(CriticRecord {
            representation: CriticRepresentation::Quantiles {
                values: "values".into(),
                locations: "locations".into(),
                masses: "masses".into(),
            },
            action_condition: ActionCondition::Fixed,
            actions: Some("actions".into()),
            continuation_policy: None,
            continuation_stochastic: None,
        }),
        arrays: BTreeMap::from([
            (
                "values".into(),
                Array::f64(
                    vec![forecasts.len(), locations.len()],
                    forecasts
                        .iter()
                        .flat_map(|f| f.values().iter().copied())
                        .collect(),
                )?,
            ),
            (
                "locations".into(),
                Array::f64(vec![locations.len()], locations.to_vec())?,
            ),
            (
                "masses".into(),
                Array::f64(vec![masses.len()], masses.to_vec())?,
            ),
            ("actions".into(), actions_array(actions)?),
        ]),
    }))
}

struct Summary {
    metrics: MetricRecord,
    arrays: BTreeMap<String, Array>,
}

impl Default for Summary {
    fn default() -> Self {
        Self {
            metrics: MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            },
            arrays: BTreeMap::new(),
        }
    }
}

impl Summary {
    fn value(&mut self, key: &str, value: f64) {
        self.metrics
            .values
            .insert(format!("reference.{key}"), value);
    }
    fn unavailable(&mut self, key: &str, reason: impl Into<String>) {
        self.metrics
            .unavailable
            .insert(format!("reference.{key}"), reason.into());
    }
    fn rows(&mut self, key: &str, values: Vec<f64>) -> Result<()> {
        let n = values.len();
        ensure(
            n > 0 && values.iter().all(|n| n.is_finite()),
            "invalid reference result rows",
        )?;
        self.value(key, values.iter().map(|value| value / n as f64).sum());
        self.arrays
            .insert(format!("reference.{key}"), Array::f64(vec![n], values)?);
        Ok(())
    }
}
