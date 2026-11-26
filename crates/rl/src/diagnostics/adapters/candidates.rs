//! Feedforward candidate diagnostics preserve the complete semantic decision menu.
use crate::{
    Result,
    algorithms::{
        candidate_ddqn::{CandidateDdqn, CandidateExploration, CandidateQActor},
        candidate_mpo::{CandidateMpo, CandidateMpoBackup, CandidateMpoBehavior},
    },
    core::{
        action::{Action, ActionSpace, ConditionalAction},
        actor::{ActMode, Actor},
        ensure,
        env::{Observation, ObservationBatch},
        stateful::{BehaviorPublisher, StatefulActor},
    },
    diagnostics::{
        probes::{ProbeSet, activation_summary},
        schema::{ActivationInputs, ActivationRecord, Array, Chunk, Record},
        session::{CaptureRequest, DiagnosticLearner, LearnerInfo},
        typed::{
            FiniteCandidate, FiniteDecision, FinitePredictions, FiniteValueKind, MAX_PREDICTIONS,
            legal_keys,
        },
        value::{self, ContinuationIdentity, CriticDeclaration, ValueCriticKind, ValueDeclaration},
    },
    nn::structured::{CandidatePolicy, StructuredActionValue},
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

const GREEDY: CandidateExploration = CandidateExploration {
    epsilon: 0.,
    distribution: crate::algorithms::candidate_ddqn::ExplorationDistribution::CompleteActions,
};

struct FeedforwardActor<A>(A);

impl<A: StatefulActor<State = ()>> Actor for FeedforwardActor<A> {
    fn action_space(&self) -> &ActionSpace {
        self.0.action_space()
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
            actions.push(self.0.decide(&(), observation, mode, rng)?.action);
        }
        Ok(())
    }
}

impl<Q: StructuredActionValue> DiagnosticLearner for CandidateDdqn<Q> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "Candidate DDQN",
            model: "semantic-candidate-q",
            configuration: serde_json::json!({
                "critic": self.config(), "architecture": std::any::type_name::<Q>(),
                "evaluation": "Greedy complete-action Q with uniform tie selection and zero exploration",
            }),
            discount: Some(f64::from(self.config().gamma)),
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(FeedforwardActor(
            self.behavior_actor(&GREEDY, self.updates())?,
        )))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        if !request.predictions && !request.representations {
            return Ok(Vec::new());
        }
        let actor = self.behavior_actor(&GREEDY, self.updates())?;
        capture(
            request,
            &actor,
            optimality(),
            |observation, row_id, mode, rng| {
                let prediction = actor.diagnostic_prediction(observation, mode, rng)?;
                decision(
                    observation,
                    row_id,
                    &prediction.action,
                    prediction.candidates,
                )
            },
        )
    }
}

impl<Q: StructuredActionValue, P: CandidatePolicy> DiagnosticLearner for CandidateMpo<Q, P> {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "Candidate MPO",
            model: "semantic-candidate-policy-and-q",
            configuration: serde_json::json!({
                "actor": self.config(), "critic": self.diagnostic_critic_config(),
                "actor_architecture": std::any::type_name::<P>(),
                "critic_architecture": std::any::type_name::<Q>(),
                "evaluation": "Published complete-action policy with zero exploration and hierarchical greedy deployment",
            }),
            discount: Some(f64::from(self.diagnostic_critic_config().gamma)),
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.updates()
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(FeedforwardActor(self.behavior_actor(
            &CandidateMpoBehavior::default(),
            self.updates(),
        )?)))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        if !request.predictions && !request.representations {
            return Ok(Vec::new());
        }
        let q = self.diagnostic_q_actor(GREEDY)?;
        let actor = self.behavior_actor(&CandidateMpoBehavior::default(), self.updates())?;
        let continuation = match self.config().backup {
            CandidateMpoBackup::PolicyEvaluation => {
                let identity = &request.context.identity;
                ContinuationIdentity {
                    identity: format!(
                        "{}@{}/pre-update-current-policy",
                        identity.run_id, identity.learner_update_revision
                    ),
                    stochastic: true,
                }
            }
            CandidateMpoBackup::MaximumOptimality | CandidateMpoBackup::DoubleQOptimality => {
                optimality()
            }
        };
        capture(
            request,
            &q,
            continuation,
            |observation, row_id, mode, rng| {
                let selected = actor.decide(&(), observation, mode, rng)?.action;
                let Action::Conditional(key) = &selected else {
                    unreachable!()
                };
                let values = q.action_values(observation)?;
                let policy = match mode {
                    ActMode::Deterministic => values
                        .iter()
                        .map(|(candidate, _)| {
                            (*candidate, f64::from(u8::from(candidate == key.as_ref())))
                        })
                        .collect(),
                    ActMode::Sample => actor
                        .action_probabilities(observation)?
                        .into_iter()
                        .collect::<BTreeMap<_, _>>(),
                };
                let scores = values
                    .into_iter()
                    .map(|(action, value)| {
                        let probability = policy.get(&action).copied().ok_or_else(|| {
                            crate::Error::State(
                                "candidate critic and policy legal menus differ".into(),
                            )
                        })?;
                        Ok((action, value, probability))
                    })
                    .collect::<Result<Vec<_>>>()?;
                decision(observation, row_id, &selected, scores)
            },
        )
    }
}

fn optimality() -> ContinuationIdentity {
    ContinuationIdentity {
        identity: "optimality".into(),
        stochastic: false,
    }
}

fn decision(
    observation: &Observation,
    row_id: u64,
    action: &Action,
    scores: Vec<(ConditionalAction, f32, f64)>,
) -> Result<FiniteDecision> {
    Ok(FiniteDecision {
        row_id,
        information_state: None,
        selected: Some(observation.finite_action_key(action)?),
        candidates: scores
            .into_iter()
            .map(|(action, value, probability)| {
                Ok(FiniteCandidate {
                    action: observation.finite_action_key(&Action::Conditional(action.into()))?,
                    probability: Some(probability),
                    value: Some(f64::from(value)),
                    visits: None,
                    updates: None,
                })
            })
            .collect::<Result<_>>()?,
    })
}

fn capture<Q: StructuredActionValue>(
    request: &CaptureRequest<'_>,
    critic: &CandidateQActor<Q>,
    continuation: ContinuationIdentity,
    mut predict: impl FnMut(&Observation, u64, ActMode, &mut ChaCha8Rng) -> Result<FiniteDecision>,
) -> Result<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut remaining = request.max_activation_values;
    for set in request.probes {
        ensure(
            set.observation_space() == &critic.observation_space()
                && set.action_space() == critic.action_space(),
            "candidate diagnostic probe space differs from its model",
        )?;
        if request.predictions {
            let mut count = 0;
            for observation in set.observations() {
                count += legal_keys(observation, set.action_space())?.len();
                ensure(
                    count <= MAX_PREDICTIONS,
                    "candidate diagnostic prediction count exceeds its limit",
                )?;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(request.policy_seed);
            let rows = set
                .observations()
                .iter()
                .zip(set.row_ids())
                .map(|(observation, &row_id)| {
                    predict(observation, row_id, request.evaluation_mode, &mut rng)
                })
                .collect::<Result<_>>()?;
            let mut chunk = FinitePredictions {
                value_kind: Some(FiniteValueKind::ActionValue),
                rows,
            }
            .record(request.context.clone(), set, request.max_chunk_bytes)?;
            let identity = &chunk.context.identity;
            let declaration = ValueDeclaration {
                version: 1,
                run_id: identity.run_id.clone(),
                revision: identity.learner_update_revision,
                model_role: identity.model_role.clone(),
                conventions: request.conventions.cloned(),
                critics: BTreeMap::from([(
                    "q".into(),
                    CriticDeclaration {
                        kind: ValueCriticKind::DiscreteQ,
                        learned_baseline: None,
                        actor: None,
                        continuation: Some(continuation.clone()),
                        twin: None,
                        utility: None,
                    },
                )]),
            };
            value::attach_declaration(&mut chunk, &declaration)?;
            chunks.push(chunk);
        }
        if request.representations
            && let Some(chunk) = representation(critic, set, request, &mut remaining)?
        {
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

fn representation<Q: StructuredActionValue>(
    critic: &CandidateQActor<Q>,
    set: &ProbeSet,
    request: &CaptureRequest<'_>,
    remaining: &mut usize,
) -> Result<Option<Chunk>> {
    let rows = set.observations().len();
    let max_width = (*remaining).min(request.max_chunk_bytes / size_of::<f32>()) / rows;
    if max_width == 0 {
        return Ok(None);
    }
    let Some(first) = critic.diagnostic_global(&set.observations()[0], max_width)? else {
        return Ok(None);
    };
    let width = first.len();
    ensure(
        (1..=max_width).contains(&width),
        "candidate encoder representation exceeds its width budget",
    )?;
    let mut values = Vec::with_capacity(rows * width);
    values.extend(first);
    for observation in &set.observations()[1..] {
        let row = critic
            .diagnostic_global(observation, max_width)?
            .filter(|row| row.len() == width)
            .ok_or_else(|| {
                crate::Error::Shape(
                    "candidate representation width changed across probe rows".into(),
                )
            })?;
        values.extend(row);
    }
    ensure(
        values.iter().all(|value| value.is_finite()),
        "candidate representation contains nonfinite values",
    )?;
    *remaining -= values.len();
    let summary = activation_summary(&values);
    let mut context = request.context.clone();
    context.probe_set = Some(set.id().into());
    Ok(Some(Chunk {
        context,
        record: Record::Activations(ActivationRecord {
            inputs: ActivationInputs::Observations,
            network: "candidate_q".into(),
            layers: BTreeMap::from([("global".into(), "global".into())]),
            summaries: BTreeMap::from([("global".into(), summary)]),
            ..Default::default()
        }),
        arrays: BTreeMap::from([("global".into(), Array::f32(vec![rows, width], values)?)]),
    }))
}
