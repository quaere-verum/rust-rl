use super::workflow::{RelayKey, SignalGraph, exact_plan, neural};
use rand_chacha::ChaCha8Rng;
use rl::{
    core::temporal::TemporalTransition,
    diagnostics::{
        evaluation::EvaluationHorizon,
        probes::ProbeSet,
        reference::{
            ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
        },
        schema::ProbePopulation,
        session::{DiagnosticConfig, DiagnosticSession},
        value::ValueConventions,
    },
    planning::finite::Solution,
    prelude::*,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const SEED: u64 = 43;
const DEFAULT_STEPS: u64 = 32;

pub(super) fn cli(mut arguments: impl Iterator<Item = String>) -> Result<()> {
    let path = arguments.next().map(PathBuf::from).ok_or_else(|| {
        Error::Config("usage: structured_control --record <recording> [--steps=N]".into())
    })?;
    let steps = arguments.next().map_or(Ok(DEFAULT_STEPS), |argument| {
        argument
            .strip_prefix("--steps=")
            .and_then(|steps| steps.parse::<u64>().ok())
            .filter(|steps| (1..=1024).contains(steps))
            .ok_or_else(|| Error::Config("--steps must be an integer between 1 and 1024".into()))
    })?;
    if arguments.next().is_some() {
        return Err(Error::Config("unexpected structured control option".into()));
    }
    record(&path, steps)
}

fn record(path: &Path, steps: u64) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let first = SignalGraph::new(31);
    let mut learner = neural::build_learner(&first, SEED)?;
    let mut trainer = neural::build_trainer(
        vec![first.clone(), SignalGraph::new(37)],
        &learner,
        SEED + 4,
    )?;
    let discount = f64::from(learner.config().gamma);
    let reference = RelayReference::new(&first, discount)?;
    let alternatives = fixed_probes(first.clone())?;
    let mut config = DiagnosticConfig::detailed();
    config.evaluation.mode = ActMode::Deterministic;
    config.evaluation.max_steps = 2;
    config.evaluation_horizon = EvaluationHorizon::FiniteDecisions(2);
    config.schedule.snapshots = None;
    let mut session = DiagnosticSession::builder(path)
        .experiment("RL Visualization Laboratory")
        .environment(&first)
        .discount(discount)
        .config(config)
        .metadata("training.seed", SEED.to_string())
        .metadata("training.protocol", "Two SignalGraph lanes and Candidate DDQN with the maintained TemporalTrainer configuration")
        .reference_context(reference.specification().context.clone())
        .reference(reference)
        .reference_trajectories(1)
        .evaluation_factory(|seed| Ok(SignalGraph::new(seed)))
        .probe_set(alternatives)
        .build(&learner)?;
    for step in 1..=steps {
        trainer.step(&mut learner)?;
        if step % 8 == 0 || step == steps {
            session.checkpoint(&learner, *trainer.counters())?;
            println!(
                "SignalGraph decisions={} learner revision={}",
                trainer.counters().env_steps,
                learner.updates()
            );
        }
    }
    if let Some(episode) = trainer
        .store()
        .episodes()
        .filter(|episode| {
            episode.base_timestep == 0 && episode.steps.last().is_some_and(|step| step.ended())
        })
        .last()
    {
        let transitions: Vec<_> = episode
            .steps
            .iter()
            .enumerate()
            .map(|(timestep, step)| TemporalTransition {
                episode_id: episode.id,
                timestep,
                observation: episode.observations[timestep].clone(),
                action: step.action.clone(),
                reward: step.reward.clone(),
                next_observation: episode.observations[timestep + 1].clone(),
                terminated: step.terminated,
                truncated: step.truncated,
                behavior: step.behavior.clone(),
            })
            .collect();
        session.record_factual_episode(
            &learner,
            *trainer.counters(),
            &format!("training-episode-{}", episode.id),
            &transitions,
        )?;
    }
    session.finish(&learner)?;
    println!("SignalGraph recording saved at {}", path.display());
    Ok(())
}

fn fixed_probes(mut env: SignalGraph) -> Result<ProbeSet> {
    let mut observations = BTreeMap::new();
    for seed in 31..63 {
        let start = env.reset(Some(seed))?.observation;
        let mut actions = start.finite_actions(env.action_space())?;
        let route = env.step(&actions.remove(0))?.observation;
        observations.entry(public_key(&start)?).or_insert(start);
        observations.entry(public_key(&route)?).or_insert(route);
        if observations.len() == 3 {
            break;
        }
    }
    if observations.len() != 3 {
        return Err(Error::State(
            "seeded SignalGraph probes did not expose both hints".into(),
        ));
    }
    ProbeSet::new_in(
        "relay-public-beliefs",
        ProbePopulation::FixedReference,
        "Three public decision graphs sampled from fixed seeded resets and inspection steps",
        observations.into_values().map(Into::into).collect(),
        None,
        env.observation_space(),
        env.action_space().clone(),
    )
}

fn public_key(observation: &Observation) -> Result<RelayKey> {
    let graph = observation
        .as_graph()
        .ok_or_else(|| Error::Shape("relay requires a graph observation".into()))?;
    match (graph.tick, graph.observation.events.as_slice()) {
        (0, []) => Ok(RelayKey::Inspect),
        (1, [hint]) => match hint.features.as_ref() {
            [0.] => Ok(RelayKey::Route { hint_right: false }),
            [1.] => Ok(RelayKey::Route { hint_right: true }),
            _ => Err(Error::State("unrecognized relay hint".into())),
        },
        (2, _) => Ok(RelayKey::End),
        _ => Err(Error::State("unrecognized relay decision graph".into())),
    }
}

struct RelayReference {
    specification: ReferenceSpec,
    plan: Solution<RelayKey, rl::core::action::ConditionalAction>,
}

impl RelayReference {
    fn new(environment: &SignalGraph, discount: f64) -> Result<Self> {
        let specification = ReferenceSpec {
            id: "optimal-public-relay".into(),
            context: ReferenceContext {
                environment: "signal_graph".into(),
                observation_dim: 0,
                typed_observation_space: Some(environment.observation_space()),
                action_space: environment.action_space().clone(),
                conventions: ValueConventions {
                    dynamics: "Symmetric hidden destination and conditionally correct public route hint".into(),
                    observation: "Lossless public graph with inspect and route phases and an optional hint event".into(),
                    action: "Semantic inspect or route operation with a port identity".into(),
                    reward_units: "packet delivery reward".into(),
                    reward_convention: "One for a correct final route, zero otherwise".into(),
                    discount,
                    horizon: "Two decisions".into(),
                    terminal_treatment: "Route ends the episode".into(),
                    action_constraints: "Inspect at reset, then route to either observed port".into(),
                },
                objective: ReferenceObjective::ExpectedDiscountedReturn,
                horizon: ReferenceHorizon::Finite { decisions: 2 },
                environment_config: BTreeMap::from([
                    ("horizon".into(), ConfigValue::Integer(2)),
                    ("hint_accuracy".into(), ConfigValue::Real(0.75)),
                ]),
                markov_observation: true,
                decision_time_coordinate: None,
            },
            quality: ReferenceQuality {
                kind: ReferenceKind::VerifiedOptimal,
                method: "Exact dynamic programming over the three live public belief states".into(),
                assumptions: vec!["The unit-discount plan is rescaled by the configured discount on the inspect row because all delivery rewards arrive exactly one decision later.".into()],
            },
            target: ReferenceTarget::VerifiedOptimal,
            lifetime: ReferenceLifetime::Fixed,
            information: ReferenceInformation::LearnerObservation,
            uncertainty: None,
        };
        specification.validate()?;
        Ok(Self {
            specification,
            plan: exact_plan()?,
        })
    }
}

impl Actor for RelayReference {
    fn action_space(&self) -> &ActionSpace {
        &self.specification.context.action_space
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        actions: &mut Vec<Action>,
    ) -> Result<()> {
        actions.clear();
        for observation in observations.iter() {
            let key = public_key(observation)?;
            let state = self
                .plan
                .states
                .iter()
                .find(|row| row.context == key)
                .ok_or_else(|| Error::State("terminal graph has no optimal action".into()))?;
            actions.push(Action::Conditional((*state.greedy_action()).into()));
        }
        Ok(())
    }
}

impl ReferenceProvider for RelayReference {
    fn specification(&self) -> &ReferenceSpec {
        &self.specification
    }

    fn actor(&self) -> Option<&dyn Actor> {
        Some(self)
    }

    fn evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        let key = public_key(observation)?;
        let factor = if key == RelayKey::Inspect {
            self.specification.context.conventions.discount
        } else {
            1.
        };
        Ok(self
            .plan
            .states
            .iter()
            .find(|row| row.context == key)
            .and_then(|row| {
                let Action::Conditional(action) = action else {
                    return None;
                };
                row.action_values()
                    .iter()
                    .find(|row| row.action == **action)
                    .map(|row| factor * row.value)
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl::diagnostics::{
        activation_inputs::validate_activation_inputs,
        schema::{Record, RecordKind},
        storage::{ReadLimits, Recording},
        typed::{TypedEpisodeSource, TypedPayload},
    };

    #[test]
    fn recorded_graph_episode_and_candidate_evidence_preserve_semantic_keys() -> rl::Result<()> {
        let temp = tempfile::tempdir()?;
        let path = temp.path().join("signal-graph");
        record(&path, 8)?;
        let recording = Recording::open(path, ReadLimits::default())?;
        assert!(recording.complete());

        let mut saw_independent_episode = false;
        let mut saw_collected_episode = false;
        let mut saw_optimal_routes = false;
        let mut saw_graph_representations = false;
        let mut saw_valid_learner_menu = false;
        let mut fixed_probe = None;
        for (index, entry) in recording.entries().iter().enumerate() {
            let chunk = recording.load(index)?;
            match entry.kind {
                RecordKind::TypedEpisode if entry.model_role == "online" => {
                    let TypedPayload::Episode(episode) = TypedPayload::read(&chunk)? else {
                        unreachable!()
                    };
                    assert_eq!(episode.steps.len(), 2);
                    assert!(episode.observations[1].as_graph().is_some());
                    assert!(episode.steps[1].terminated && !episode.steps[1].truncated);
                    match episode.source {
                        TypedEpisodeSource::Evaluation { .. } => saw_independent_episode = true,
                        TypedEpisodeSource::Factual { .. } => {
                            assert!(
                                episode
                                    .steps
                                    .iter()
                                    .all(|step| step.behavior_revision.is_some()
                                        && step.factual_reward.is_some())
                            );
                            saw_collected_episode = true;
                        }
                    }
                }
                RecordKind::TypedProbes
                    if entry.model_role == "online"
                        && entry.probe_set.as_deref() == Some("relay-public-beliefs") =>
                {
                    fixed_probe = Some(chunk);
                }
                RecordKind::FinitePredictions
                    if entry.model_role == "online"
                        && entry.probe_set.as_deref() == Some("relay-public-beliefs") =>
                {
                    let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunk)? else {
                        unreachable!()
                    };
                    let TypedPayload::Probes(probes) = TypedPayload::read(
                        fixed_probe
                            .as_ref()
                            .expect("fixed probe was recorded first"),
                    )?
                    else {
                        unreachable!()
                    };
                    predictions.validate_support(
                        &probes.row_ids,
                        probes.inputs.iter().map(|input| input.observation()),
                        &recording.manifest().action_space,
                    )?;
                    assert!(
                        predictions
                            .rows
                            .iter()
                            .flat_map(|row| &row.candidates)
                            .all(|candidate| candidate.probability.is_some()
                                && candidate.value.is_some())
                    );
                    saw_valid_learner_menu = true;
                }
                RecordKind::FinitePredictions if entry.model_role.starts_with("reference:") => {
                    let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunk)? else {
                        unreachable!()
                    };
                    for row in predictions
                        .rows
                        .iter()
                        .filter(|row| row.candidates.len() == 2)
                    {
                        let values: Vec<_> = row
                            .candidates
                            .iter()
                            .map(|row| row.value.unwrap())
                            .collect();
                        assert!(values.contains(&0.25) && values.contains(&0.75));
                        assert!(row.selected.is_some());
                        saw_optimal_routes = true;
                    }
                }
                RecordKind::Activations
                    if entry.model_role == "online"
                        && entry.probe_set.as_deref() == Some("relay-public-beliefs") =>
                {
                    let Record::Activations(activations) = &chunk.record else {
                        unreachable!()
                    };
                    assert!(activations.layers.contains_key("global"));
                    validate_activation_inputs(
                        &chunk,
                        fixed_probe
                            .as_ref()
                            .expect("fixed probe was recorded first"),
                        0,
                        &recording.manifest().action_space,
                    )?;
                    saw_graph_representations = true;
                }
                _ => {}
            }
        }
        assert!(
            saw_independent_episode
                && saw_collected_episode
                && saw_optimal_routes
                && saw_graph_representations
                && saw_valid_learner_menu
        );
        Ok(())
    }
}
