#![cfg(feature = "diagnostics")]

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::{Env, EpisodeStart, Observation, ObservationBatch, ObservationSpace, Step},
        learner::UpdateReport,
    },
    diagnostics::{
        evaluation::{EvaluationHorizon, evaluation_horizon},
        probes::ProbeSet,
        reference::{
            BenchmarkActor, ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
            provider_specification,
        },
        schedule::Interval,
        schema::{
            ActionCondition, Array, Chunk, CriticRecord, CriticRepresentation, MetricRecord,
            PolicyParameters, PolicyRecord, ProbePopulation, Record,
        },
        session::{
            CaptureRequest, DiagnosticConfig, DiagnosticLearner, DiagnosticSession, LearnerInfo,
        },
        storage::{ReadLimits, Recording},
        value::{
            self, ContinuationIdentity, CriticDeclaration, ValueConventions, ValueCriticKind,
            ValueDeclaration,
        },
    },
    train::{
        counters::RunCounters,
        observer::{UpdateBoundary, UpdateObserver},
    },
};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};

const SPACE: ActionSpace = ActionSpace::Discrete { count: 3 };
const REWARDS: [[f32; 2]; 2] = [[1., 4.], [2., 6.]];
const PROBES: &str = "live-decisions";

fn observation(step: usize) -> Observation {
    Observation::dense(vec![step as f32], Some(vec![step < 2, step < 2, false]))
}

#[derive(Default)]
struct TwoDecisions(usize);

impl Env for TwoDecisions {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Dense(1)
    }

    fn action_space(&self) -> &ActionSpace {
        &SPACE
    }

    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.0 = 0;
        Ok(EpisodeStart::fresh(observation(0)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        let Action::Discrete(action) = action else {
            return Err(Error::Action("fixture requires discrete action".into()));
        };
        let reward = REWARDS[self.0][*action];
        self.0 += 1;
        Ok(Step {
            observation: observation(self.0),
            reward: reward.into(),
            terminated: self.0 == 2,
            truncated: false,
        })
    }
}

struct FixedActor {
    action: usize,
    observed_rows: Option<Rc<Cell<usize>>>,
}

impl Actor for FixedActor {
    fn action_space(&self) -> &ActionSpace {
        &SPACE
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        if let Some(rows) = &self.observed_rows {
            rows.set(rows.get() + observations.len());
        }
        out.clear();
        out.resize(observations.len(), Action::Discrete(self.action));
        Ok(())
    }
}

struct Learner {
    revision: u64,
    action: usize,
    mismatched_continuation: bool,
    target_continuation: bool,
}

struct TargetActor(usize);

impl Actor for TargetActor {
    fn action_space(&self) -> &ActionSpace {
        &SPACE
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        mode: ActMode,
        _: &mut ChaCha8Rng,
        out: &mut Vec<Action>,
    ) -> Result<()> {
        let action = if mode == ActMode::Sample {
            1 - self.0
        } else {
            self.0
        };
        out.clear();
        out.resize(observations.len(), Action::Discrete(action));
        Ok(())
    }
}

impl Learner {
    fn target_identity(&self, run_id: &str) -> ContinuationIdentity {
        ContinuationIdentity {
            identity: format!("{run_id}@{}/target-fixture", self.revision),
            stochastic: true,
        }
    }
}

impl DiagnosticLearner for Learner {
    fn diagnostic_info(&self) -> Result<LearnerInfo> {
        Ok(LearnerInfo {
            algorithm: "finite-fixture",
            model: "tabular-fixture",
            configuration: serde_json::json!({}),
            discount: Some(1.),
        })
    }

    fn diagnostic_revision(&self) -> u64 {
        self.revision
    }

    fn diagnostic_actor(&self) -> Result<Box<dyn Actor + '_>> {
        Ok(Box::new(FixedActor {
            action: self.action,
            observed_rows: None,
        }))
    }

    fn diagnostic_conditional_actor(
        &self,
        context: &rl::diagnostics::schema::RecordContext,
    ) -> Result<Option<(Box<dyn Actor + '_>, ContinuationIdentity)>> {
        if !self.target_continuation {
            return Ok(None);
        }
        Ok(Some((
            Box::new(TargetActor(self.action)),
            self.target_identity(&context.identity.run_id),
        )))
    }

    fn diagnostic_records(&self, request: &CaptureRequest<'_>) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        for probes in request.probes {
            let mut context = request.context.clone();
            context.probe_set = Some(probes.id().into());
            let n = probes.observations().len();
            let masks = probes
                .observations()
                .iter()
                .flat_map(|obs| obs.dense_mask().unwrap().iter().copied().map(u8::from))
                .collect();
            chunks.push(Chunk {
                context: context.clone(),
                arrays: BTreeMap::from([
                    (
                        "actions".into(),
                        Array::u64(vec![n], vec![self.action as u64; n])?,
                    ),
                    ("mask".into(), Array::u8(vec![n, 3], masks)?),
                ]),
                record: Record::Policy(PolicyRecord {
                    actions: "actions".into(),
                    parameters: PolicyParameters::Deterministic,
                    entropy: None,
                    entropy_method: None,
                    masks: Some("mask".into()),
                    stochastic: false,
                }),
            });

            let actor = ContinuationIdentity {
                identity: format!("{}@{}", context.identity.run_id, self.revision),
                stochastic: false,
            };
            let mut continuation = if self.target_continuation {
                self.target_identity(&context.identity.run_id)
            } else {
                actor.clone()
            };
            if self.mismatched_continuation {
                continuation.identity = "a-different-policy".into();
            }
            let continuation_action = if self.target_continuation {
                1 - self.action
            } else {
                self.action
            };
            let next_reward = f64::from(REWARDS[1][continuation_action]);
            // Each legal row has errors [+1, -1]. A large illegal head must
            // affect neither calibration nor the policy's decision regret.
            let values = vec![2. + next_reward, 3. + next_reward, 1000., 3., 5., 1000.];
            let declaration = ValueDeclaration {
                version: 1,
                run_id: context.identity.run_id.clone(),
                revision: self.revision,
                model_role: context.identity.model_role.clone(),
                conventions: request.conventions.cloned(),
                critics: BTreeMap::from([(
                    "policy_q".into(),
                    CriticDeclaration {
                        kind: ValueCriticKind::DiscreteQ,
                        learned_baseline: None,
                        actor: Some(actor),
                        continuation: Some(continuation.clone()),
                        twin: None,
                        utility: None,
                    },
                )]),
            };
            let mut critic = Chunk {
                context,
                arrays: BTreeMap::from([("values".into(), Array::f64(vec![n, 3], values)?)]),
                record: Record::Critic(CriticRecord {
                    representation: CriticRepresentation::Discrete {
                        values: "values".into(),
                    },
                    action_condition: ActionCondition::AllDiscreteActions,
                    actions: None,
                    continuation_policy: Some(continuation.identity),
                    continuation_stochastic: Some(continuation.stochastic),
                }),
            };
            value::attach_declaration(&mut critic, &declaration)?;
            chunks.push(critic);
        }
        Ok(chunks)
    }
}

fn context() -> ReferenceContext {
    ReferenceContext {
        environment: "fixture-two-decisions".into(),
        observation_dim: 1,
        typed_observation_space: None,
        action_space: SPACE,
        conventions: ValueConventions {
            dynamics: "two successive deterministic decision states".into(),
            observation: "fully observed integer decision time".into(),
            action: "discrete action index".into(),
            reward_units: "fixture reward".into(),
            reward_convention: "environment reward, larger is better".into(),
            discount: 1.,
            horizon: "two decisions".into(),
            terminal_treatment: "no continuation after the second decision".into(),
            action_constraints: "the third action is always masked".into(),
        },
        objective: ReferenceObjective::ExpectedDiscountedReturn,
        horizon: ReferenceHorizon::Finite { decisions: 2 },
        environment_config: BTreeMap::from([("horizon".into(), ConfigValue::Integer(2))]),
        markov_observation: true,
        decision_time_coordinate: Some(0),
    }
}

fn specification(target: ReferenceTarget) -> ReferenceSpec {
    use ReferenceTarget::*;
    ReferenceSpec {
        id: "fixture".into(),
        context: context(),
        quality: ReferenceQuality {
            kind: match target {
                BenchmarkPolicy => ReferenceKind::Heuristic,
                VerifiedOptimal => ReferenceKind::VerifiedOptimal,
                LearnerPolicy | CriticContinuation => ReferenceKind::Analytical,
            },
            method: "exact two-decision fixture".into(),
            assumptions: vec![],
        },
        target,
        lifetime: if matches!(target, LearnerPolicy | CriticContinuation) {
            ReferenceLifetime::CheckpointDependent
        } else {
            ReferenceLifetime::Fixed
        },
        information: ReferenceInformation::LearnerObservation,
        uncertainty: None,
    }
}

type FrozenRevisions = Rc<RefCell<Vec<(u64, String, usize)>>>;

struct ExactReference {
    spec: ReferenceSpec,
    revisions: FrozenRevisions,
}

impl ExactReference {
    fn policy_action(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        obs: &Observation,
    ) -> Result<usize> {
        if self.spec.target == ReferenceTarget::VerifiedOptimal {
            return Ok(1);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        match checkpoint
            .policy
            .act(std::slice::from_ref(obs), checkpoint.mode, &mut rng)?
            .as_slice()
        {
            [Action::Discrete(action)] => Ok(*action),
            _ => Err(Error::Action(
                "fixture policy must return one discrete action".into(),
            )),
        }
    }
}

impl ReferenceProvider for ExactReference {
    fn specification(&self) -> &ReferenceSpec {
        &self.spec
    }

    fn state_value(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        obs: &Observation,
    ) -> Result<Option<f64>> {
        let action = self.policy_action(checkpoint, obs)?;
        if obs.dense_values() == [0.] {
            self.revisions.borrow_mut().push((
                checkpoint.revision,
                checkpoint.policy_id.into(),
                action,
            ));
        }
        self.action_value(checkpoint, obs, &Action::Discrete(action))
    }

    fn action_value(
        &self,
        checkpoint: &ReferenceCheckpoint<'_>,
        obs: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        SPACE.check(action, obs.dense_mask())?;
        let Action::Discrete(action) = action else {
            unreachable!()
        };
        let step = obs.dense_values()[0] as usize;
        let mut total = f64::from(REWARDS[step][*action]);
        if step == 0 {
            total += f64::from(REWARDS[1][self.policy_action(checkpoint, &observation(1))?]);
        }
        Ok(Some(total))
    }
}

#[derive(Default)]
struct CaseOptions {
    evaluation_horizon: EvaluationHorizon,
    target_continuation: bool,
}

fn run_case(
    experiment: ReferenceContext,
    provider: impl ReferenceProvider,
    checkpoints: &[(u64, usize, bool)],
    options: CaseOptions,
) -> Result<Vec<Chunk>> {
    let temporary = tempfile::tempdir()?;
    let path = temporary.path().join("run");
    let mut learner = Learner {
        revision: 11,
        action: 1,
        mismatched_continuation: false,
        target_continuation: options.target_continuation,
    };
    let mut config = DiagnosticConfig::light();
    config.evaluation.mode = ActMode::Deterministic;
    config.evaluation.seeds = vec![4, 5];
    config.evaluation.max_steps = 2;
    config.evaluation.trajectories = 0;
    config.evaluation_horizon = options.evaluation_horizon;
    let every = Some(Interval { first: 1, every: 1 });
    config.schedule.evaluation = every;
    config.schedule.probes = every;
    config.schedule.reference = every;
    let probes = ProbeSet::new(
        PROBES,
        ProbePopulation::FixedReference,
        "both live states with exact masks",
        vec![observation(0), observation(1)],
        None,
        SPACE,
    )?
    .with_row_ids(vec![23, 17])?;
    let mut session = DiagnosticSession::builder(&path)
        .environment(&TwoDecisions::default())
        .evaluation_factory(|_| Ok(TwoDecisions::default()))
        .config(config)
        .probe_set(probes)
        .reference_context(experiment)
        .reference(provider)
        .build(&learner)?;
    for &(revision, action, mismatch) in checkpoints {
        let updates = revision - learner.revision;
        learner.revision = revision;
        learner.action = action;
        learner.mismatched_continuation = mismatch;
        session.after_update(
            &learner,
            UpdateBoundary {
                counters: &RunCounters {
                    updates: revision - 11,
                    ..Default::default()
                },
                learner_call: revision - 11,
                collection_cycle: revision - 11,
                report: &UpdateReport {
                    updates,
                    ..Default::default()
                },
                collection_seconds: 0.,
                learning_seconds: 0.,
            },
        )?;
    }
    session.finish(&learner)?;
    let recording = Recording::open(path, ReadLimits::default())?;
    assert!(recording.complete());
    (0..recording.entries().len())
        .map(|n| recording.load(n))
        .collect()
}

fn summary<'a>(
    chunks: &'a [Chunk],
    revision: u64,
    population: Option<&str>,
) -> (&'a MetricRecord, &'a BTreeMap<String, Array>) {
    chunks
        .iter()
        .find_map(|chunk| {
            if chunk.context.identity.learner_update_revision == revision
                && chunk.context.probe_set.as_deref() == population
                && let Record::Metrics(metrics) = &chunk.record
                && provider_specification(chunk).unwrap().is_some()
            {
                Some((metrics, &chunk.arrays))
            } else {
                None
            }
        })
        .expect("portable provider summary at requested checkpoint and population")
}

#[test]
fn policy_only_reference_reports_signed_gap_and_caches_fixed_outputs() -> Result<()> {
    let observed_rows = Rc::new(Cell::new(0));
    let benchmark = BenchmarkActor::new(
        specification(ReferenceTarget::BenchmarkPolicy),
        FixedActor {
            action: 0,
            observed_rows: Some(observed_rows.clone()),
        },
    )?;
    let chunks = run_case(
        context(),
        benchmark,
        &[(14, 0, false)],
        CaseOptions::default(),
    )?;
    let (first, _) = summary(&chunks, 11, None);
    assert_eq!(first.values["reference.learner_mean_return"], 10.);
    assert_eq!(first.values["reference.benchmark_mean_return"], 3.);
    assert_eq!(first.values["reference.benchmark_gap"], -7.);
    assert!(!first.values.contains_key("reference.optimality_gap"));
    assert_eq!(
        summary(&chunks, 14, None).0.values["reference.benchmark_gap"],
        0.
    );
    for revision in [11, 14] {
        let (metrics, arrays) = summary(&chunks, revision, Some(PROBES));
        assert!(
            metrics
                .unavailable
                .contains_key("reference.critic_calibration")
        );
        assert!(
            metrics
                .unavailable
                .contains_key("reference.local_decision_regret")
        );
        assert!(!arrays.contains_key("reference.local_decision_regret"));
    }
    let reference_chunks: Vec<_> = chunks
        .iter()
        .filter(|chunk| chunk.context.identity.model_role == "reference:fixture")
        .collect();
    assert_eq!(
        reference_chunks
            .iter()
            .filter(|chunk| matches!(chunk.record, Record::Empirical(_)))
            .count(),
        1
    );
    assert_eq!(
        reference_chunks
            .iter()
            .filter(|chunk| matches!(chunk.record, Record::Policy(_)))
            .count(),
        1
    );
    assert!(
        reference_chunks
            .iter()
            .all(|chunk| chunk.context.identity.learner_update_revision == 11)
    );
    assert_eq!(observed_rows.get(), 6); // Two two-step evaluation episodes and two fixed inputs.
    Ok(())
}

#[test]
fn checkpoint_dependent_reference_reads_frozen_policy_and_rejects_a_different_continuation()
-> Result<()> {
    let revisions = FrozenRevisions::default();
    let provider = ExactReference {
        spec: specification(ReferenceTarget::LearnerPolicy),
        revisions: revisions.clone(),
    };
    let chunks = run_case(
        context(),
        provider,
        &[(14, 0, false), (17, 1, true)],
        CaseOptions::default(),
    )?;
    assert_eq!(
        *revisions.borrow(),
        [
            (11, "run@11".into(), 1),
            (14, "run@14".into(), 0),
            (17, "run@17".into(), 1)
        ]
    );
    for revision in [11, 14] {
        let (metrics, _) = summary(&chunks, revision, Some(PROBES));
        assert_eq!(metrics.values["reference.critic.policy_q.bias"], 0.);
        assert_eq!(metrics.values["reference.critic.policy_q.mae"], 1.);
        assert_eq!(metrics.values["reference.critic.policy_q.rmse"], 1.);
    }
    let (mismatch, _) = summary(&chunks, 17, Some(PROBES));
    assert!(mismatch.unavailable["reference.critic.policy_q.calibration"].contains("continuation"));
    assert!(
        !mismatch
            .values
            .contains_key("reference.critic.policy_q.mae")
    );
    let value_at = |revision| {
        chunks
            .iter()
            .find_map(|chunk| {
                if chunk.context.identity.model_role != "reference:fixture"
                    || chunk.context.identity.learner_update_revision != revision
                {
                    return None;
                }
                match &chunk.record {
                    Record::Critic(record)
                        if record.action_condition == ActionCondition::StateValue =>
                    {
                        Some(chunk.arrays["values"].to_f64())
                    }
                    _ => None,
                }
            })
            .expect("recorded revision-specific reference state values")
    };
    assert_eq!(value_at(11), [10., 6.]);
    assert_eq!(value_at(14), [3., 2.]);
    assert_eq!(value_at(17), [10., 6.]);
    Ok(())
}

#[test]
fn verified_optimum_reports_masked_local_regret_without_scoring_policy_critic_against_q_star()
-> Result<()> {
    let revisions = FrozenRevisions::default();
    let provider = ExactReference {
        spec: specification(ReferenceTarget::VerifiedOptimal),
        revisions: revisions.clone(),
    };
    let chunks = run_case(
        context(),
        provider,
        &[(14, 0, false)],
        CaseOptions::default(),
    )?;
    let (metrics, arrays) = summary(&chunks, 14, Some(PROBES));
    assert_eq!(
        arrays["reference.local_decision_regret"].as_f64()?,
        [3., 4.]
    );
    assert!(
        metrics
            .unavailable
            .contains_key("reference.critic.policy_q.calibration")
    );
    assert!(!metrics.values.contains_key("reference.critic.policy_q.mae"));
    assert_eq!(revisions.borrow().len(), 1);
    Ok(())
}

#[test]
fn incompatible_reference_horizon_leaves_online_evaluation_available() -> Result<()> {
    let mut spec = specification(ReferenceTarget::BenchmarkPolicy);
    spec.context.horizon = ReferenceHorizon::Finite { decisions: 1 };
    spec.context.conventions.horizon = "one decision".into();
    spec.context
        .environment_config
        .insert("horizon".into(), ConfigValue::Integer(1));
    let observed_rows = Rc::new(Cell::new(0));
    let provider = BenchmarkActor::new(
        spec,
        FixedActor {
            action: 0,
            observed_rows: Some(observed_rows.clone()),
        },
    )?;
    let chunks = run_case(context(), provider, &[], CaseOptions::default())?;
    let (metrics, _) = summary(&chunks, 11, None);
    assert!(metrics.unavailable.contains_key("reference.compatibility"));
    assert!(!metrics.values.contains_key("reference.benchmark_gap"));
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk.context.identity.model_role == "online"
                && matches!(chunk.record, Record::Empirical(_)))
    );
    assert_eq!(observed_rows.get(), 0);
    Ok(())
}

#[test]
fn finite_runtime_evaluation_does_not_compare_to_a_continuing_reference() -> Result<()> {
    let mut spec = specification(ReferenceTarget::BenchmarkPolicy);
    spec.context.horizon = ReferenceHorizon::Continuing;
    spec.context.conventions.horizon = "continuing with an absorbing zero-reward state".into();
    spec.context.environment_config.remove("horizon");
    spec.context
        .environment_config
        .insert("absorbing_reward".into(), ConfigValue::Integer(0));
    let experiment = spec.context.clone();
    let observed_rows = Rc::new(Cell::new(0));
    let provider = BenchmarkActor::new(
        spec,
        FixedActor {
            action: 0,
            observed_rows: Some(observed_rows.clone()),
        },
    )?;
    let horizon = EvaluationHorizon::FiniteDecisions(1);
    let chunks = run_case(
        experiment,
        provider,
        &[],
        CaseOptions {
            evaluation_horizon: horizon,
            ..Default::default()
        },
    )?;
    let (metrics, _) = summary(&chunks, 11, None);
    assert!(metrics.unavailable["reference.performance_gap"].contains("horizon"));
    assert!(!metrics.unavailable.contains_key("reference.compatibility"));
    assert!(!metrics.values.contains_key("reference.benchmark_gap"));
    let evaluations: Vec<_> = chunks
        .iter()
        .filter(|c| matches!(c.record, Record::Empirical(_)))
        .collect();
    let [evaluation] = evaluations.as_slice() else {
        panic!("expected only the online evaluation")
    };
    assert_eq!(evaluation.context.identity.model_role, "online");
    assert_eq!(evaluation_horizon(evaluation)?, horizon);
    assert_eq!(evaluation.arrays["returns"].as_f64()?, [4., 4.]);
    assert_eq!(observed_rows.get(), 2); // Only the independent policy overlay ran.
    Ok(())
}

#[test]
fn critic_continuation_provider_receives_frozen_target_identity_and_sampling_mode() -> Result<()> {
    let revisions = FrozenRevisions::default();
    let provider = ExactReference {
        spec: specification(ReferenceTarget::CriticContinuation),
        revisions: revisions.clone(),
    };
    let chunks = run_case(
        context(),
        provider,
        &[(14, 0, false)],
        CaseOptions {
            target_continuation: true,
            ..Default::default()
        },
    )?;
    // The online policy chooses 1 then 0 across checkpoints. Only the target
    // actor in sampling mode makes the opposite decisions.
    assert_eq!(
        *revisions.borrow(),
        [
            (11, "run@11/target-fixture".into(), 0),
            (14, "run@14/target-fixture".into(), 1)
        ]
    );

    for (revision, expected_reference, online_return) in [(11, [3., 2.], 10.), (14, [10., 6.], 3.)]
    {
        let (metrics, _) = summary(&chunks, revision, Some(PROBES));
        assert_eq!(metrics.values["reference.critic.policy_q.bias"], 0.);
        assert_eq!(metrics.values["reference.critic.policy_q.mae"], 1.);
        assert_eq!(metrics.values["reference.critic.policy_q.rmse"], 1.);
        let state = chunks.iter().find(|chunk| {
            chunk.context.identity.learner_update_revision == revision
                && chunk.context.identity.model_role == "reference:fixture"
                && matches!(&chunk.record, Record::Critic(record) if record.action_condition == ActionCondition::StateValue)
        }).expect("recorded continuation state values");
        assert_eq!(state.arrays["values"].as_f64()?, expected_reference);
        let declaration = value::declaration(state)?.unwrap();
        assert_eq!(
            declaration.critics["state_value"].continuation,
            Some(ContinuationIdentity {
                identity: format!("run@{revision}/target-fixture"),
                stochastic: true,
            })
        );
        let online = chunks
            .iter()
            .find(|chunk| {
                chunk.context.identity.learner_update_revision == revision
                    && chunk.context.identity.model_role == "online"
                    && matches!(chunk.record, Record::Empirical(_))
            })
            .expect("recorded online return samples");
        assert_eq!(online.arrays["returns"].as_f64()?, [online_return; 2]);
    }
    Ok(())
}
