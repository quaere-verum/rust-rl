#[path = "support/gridworld.rs"]
mod gridworld;

use gridworld::{
    ACTIONS, CellActionFeatures, ENVIRONMENT, ExactGrid, GAMMA, Grid, MAX_STEPS, START, cell,
    fixed_probes, legal, spaces, states,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    algorithms::{
        fqi, lspi,
        tabular::{QTable, TabularActor, TdConfig},
    },
    approx::{
        FeatureMap, FittedQ, StatisticalActor,
        batch::{BatchCoverage, BatchObjective, BatchOptions, DecisionSpan, FactualBatch},
        regression::{DiagonalPenalty, LinearModel, RidgeDiagnostics, WeightedRidge},
    },
    core::{
        action::{Action, FiniteActionKey},
        actor::ActMode,
        env::{Env, PublicHistory},
        state_key::DiscreteSymbolKey,
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::temporal::EpisodeStore,
    diagnostics::{
        evaluation::EvaluationHorizon,
        probes::ProbeSet,
        reference::ReferenceProvider,
        schedule::Interval,
        schema::ProbePopulation,
        session::{DiagnosticConfig, DiagnosticLearner, DiagnosticSession},
        statistical::FittedQDiagnostics,
        tabular::{TabularDiagnostics, TabularMethod},
        typed::DiagnosticInput,
    },
    train::counters::RunCounters,
};
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

type Table = QTable<u64, FiniteActionKey>;
type StatisticalPolicy = StatisticalActor<CellActionFeatures, LinearModel>;

struct Corpus {
    store: EpisodeStore,
    table: Table,
    counters: RunCounters,
    histories: Vec<PublicHistory>,
    actions: Vec<Action>,
    last_episode: Vec<TemporalTransition>,
    rng: ChaCha8Rng,
}

impl Corpus {
    fn new() -> Result<Self> {
        Ok(Self {
            store: EpisodeStore::new(128, spaces().observations, ACTIONS)?,
            table: Table::new(0.0)?,
            counters: RunCounters::default(),
            histories: Vec::new(),
            actions: Vec::new(),
            last_episode: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(17),
        })
    }

    fn collect_episode(&mut self, state: usize, first: usize) -> Result<()> {
        let mut env = Grid::starting_at(state);
        let start = env.reset(None)?;
        let episode_id = self.counters.episodes;
        self.store.begin_episode(episode_id, &start)?;
        self.last_episode.clear();
        let mut history = PublicHistory::new(start.observation);
        let mut action = Action::Discrete(first);
        loop {
            let observation = history.observation();
            let step = env.step(&action)?;
            let transition = TemporalTransition {
                episode_id,
                timestep: self.last_episode.len(),
                observation: observation.clone(),
                action: action.clone(),
                reward: step.reward.clone(),
                next_observation: step.observation.clone(),
                terminated: step.terminated,
                truncated: step.truncated,
                behavior: BehaviorMetadata {
                    revision: 0,
                    log_prob: None,
                },
            };
            self.store.push(&transition)?;
            self.last_episode.push(transition);
            self.histories.push(history.clone());
            self.actions.push(action.clone());
            self.counters.env_steps += 1;
            self.counters.vector_steps += 1;

            let next_state = cell(&step.observation)?;
            let next_keys: Vec<_> = legal(next_state).map(FiniteActionKey::Discrete).collect();
            let next_symbol = next_state as u64;
            let successor = (!step.terminated).then_some((&next_symbol, next_keys.as_slice()));
            self.table.q_learning(
                &(cell(observation)? as u64),
                &observation.finite_action_key(&action)?,
                &step.reward,
                successor,
                TdConfig {
                    gamma: GAMMA,
                    step_size: 1.0,
                },
            )?;
            if step.ended() {
                break;
            }
            history = history.append(action, step.reward, step.observation)?;
            let choices = history.observation().finite_actions(&ACTIONS)?;
            action = choices[self.rng.random_range(0..choices.len())].clone();
        }
        self.counters.episodes += 1;
        Ok(())
    }

    fn coverage_probe(&self) -> Result<Option<ProbeSet>> {
        if self.histories.is_empty() {
            return Ok(None);
        }
        ProbeSet::new_in(
            "factual-visitation",
            ProbePopulation::ReplayRetained,
            "All factual training decisions retained by this boundary, with their original history and executed action",
            self.histories.iter().cloned().map(DiagnosticInput::History).collect(),
            Some(self.actions.clone()),
            spaces().observations,
            ACTIONS,
        ).map(Some)
    }
}

struct Publication {
    tabular: TabularActor<DiscreteSymbolKey>,
    fqi: StatisticalPolicy,
    fqi_sweeps: Vec<fqi::Sweep<RidgeDiagnostics>>,
    lspi: StatisticalPolicy,
    lspi_iterations: Vec<lspi::Iteration>,
    lspi_stable: bool,
    coverage: BatchCoverage,
    counters: RunCounters,
    visitation: Option<ProbeSet>,
    factual_episode: Vec<TemporalTransition>,
}

fn fit_publications() -> Result<Vec<Publication>> {
    let mut corpus = Corpus::new()?;
    // Successors have larger cell symbols. This order makes the shared corpus
    // of one episode per legal opening action informative even at early fits.
    let openings: Vec<_> = states()
        .rev()
        .flat_map(|state| legal(state).map(move |action| (state, action)))
        .collect();
    let mut openings = openings.into_iter();
    let features = CellActionFeatures::new();
    let objective = BatchObjective::Physical { gamma: GAMMA };
    let batch_options = BatchOptions {
        objective,
        span: DecisionSpan::Single,
    };
    let ridge = WeightedRidge {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..Default::default()
    };
    let lspi_config = lspi::LspiConfig {
        penalty: DiagonalPenalty::Uniform(0.0),
        ..Default::default()
    };
    let mut publications = Vec::<Publication>::with_capacity(4);
    for (revision, (episodes, sweeps)) in [(0, 1), (4, 1), (4, 2), (usize::MAX, MAX_STEPS)]
        .into_iter()
        .enumerate()
    {
        for (state, action) in openings.by_ref().take(episodes) {
            corpus.collect_episode(state, action)?;
        }
        let batch = FactualBatch::from_store(&corpus.store, &features, batch_options)?;
        let (fqi_model, fqi_sweeps, lspi_model, lspi_iterations, lspi_stable) =
            if let Some(previous) = publications.last() {
                let fqi_config = fqi::FqiConfig {
                    sweeps: NonZeroUsize::new(sweeps).expect("positive sweep count"),
                    ..Default::default()
                };
                let fqi_fit = fqi::fit(
                    &batch,
                    &ridge,
                    fqi_config,
                    fqi::Initial::Fitted(previous.fqi.model()),
                )?;
                let lspi_fit = lspi::fit(&batch, &lspi_config, Some(&previous.lspi))?;
                (
                    fqi_fit.model,
                    fqi_fit.sweeps,
                    lspi_fit.model,
                    lspi_fit.iterations,
                    lspi_fit.stable,
                )
            } else {
                let initial = FittedQ::new(
                    features.spec().clone(),
                    spaces(),
                    objective,
                    LinearModel::new(vec![0.0; features.spec().dimension()])?,
                )?;
                (initial.clone(), Vec::new(), initial, Vec::new(), false)
            };
        let revision = revision as u64;
        publications.push(Publication {
            tabular: TabularActor::new(
                DiscreteSymbolKey,
                corpus.table.clone(),
                spaces(),
                revision,
                0.0,
            )?,
            fqi: StatisticalActor::new(features.clone(), fqi_model, revision, 0.0)?,
            fqi_sweeps,
            lspi: StatisticalActor::new(features.clone(), lspi_model, revision, 0.0)?,
            lspi_iterations,
            lspi_stable,
            coverage: batch.coverage()?,
            counters: corpus.counters,
            visitation: corpus.coverage_probe()?,
            factual_episode: corpus.last_episode.clone(),
        });
    }
    Ok(publications)
}

fn record_publication<L: DiagnosticLearner>(
    session: &mut DiagnosticSession<'_, L>,
    learner: &L,
    item: &Publication,
) -> Result<()> {
    session.checkpoint(learner, item.counters)?;
    if let Some(coverage) = &item.visitation {
        session.record_history_probe(learner, item.counters, coverage)?;
    }
    if let Some(first) = item.factual_episode.first() {
        session.record_factual_episode(
            learner,
            item.counters,
            &format!("factual-{}", first.episode_id),
            &item.factual_episode,
        )?;
    }
    Ok(())
}

fn write_run<'a, L: DiagnosticLearner>(
    root: &Path,
    name: &str,
    reference: &ExactGrid,
    publications: impl IntoIterator<Item = (L, &'a Publication)>,
) -> Result<()> {
    let mut publications = publications.into_iter();
    let (first, baseline) = publications.next().expect("Gridworld baseline publication");
    let mut configuration = DiagnosticConfig::standard();
    let each = Some(Interval { first: 0, every: 1 });
    configuration.schedule.probes = each;
    configuration.schedule.snapshots = each;
    configuration.schedule.reference = each;
    configuration.schedule.evaluation = each;
    configuration.schedule.initial = false;
    configuration.schedule.final_checkpoint = false;
    configuration.evaluation_horizon = EvaluationHorizon::FiniteDecisions(MAX_STEPS);
    configuration.evaluation.seeds = vec![0];
    configuration.evaluation.mode = ActMode::Deterministic;
    configuration.evaluation.max_steps = MAX_STEPS;
    configuration.evaluation.trajectories = 1;
    configuration.evaluation.allow_truncated_prefix = false;
    configuration.max_probe_rows = 128;
    let mut session = DiagnosticSession::builder(root.join(name))
        .environment(&Grid::starting_at(START))
        .environment_name(ENVIRONMENT)
        .experiment("Directed Gridworld learner comparison")
        .config(configuration)
        .probe_set(fixed_probes()?)
        .evaluation_factory(|_| Ok(Grid::starting_at(START)))
        .reference_context(reference.specification().context.clone())
        .reference(reference.clone())
        .reference_trajectories(1)
        .build(&first)?;
    record_publication(&mut session, &first, baseline)?;
    let mut latest = first;
    for (learner, item) in publications {
        record_publication(&mut session, &learner, item)?;
        latest = learner;
    }
    session.finish(&latest)
}

fn record_gridworld(root: &Path) -> Result<()> {
    let reference = ExactGrid::new()?;
    let publications = fit_publications()?;
    std::fs::create_dir_all(root)?;
    let tabular = publications.iter().map(|publication| {
        (
            TabularDiagnostics::discrete(&publication.tabular, TabularMethod::QLearning, GAMMA),
            publication,
        )
    });
    write_run(root, "tabular", &reference, tabular)?;
    let fitted = publications
        .iter()
        .map(|publication| {
            Ok((
                FittedQDiagnostics::fqi(
                    &publication.fqi,
                    &publication.fqi_sweeps,
                    &publication.coverage,
                )?
                .with_fqi_config(fqi::FqiConfig::default())
                .with_portable_model(),
                publication,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    write_run(root, "fqi", &reference, fitted)?;
    let lspi = publications
        .iter()
        .map(|publication| {
            Ok((
                FittedQDiagnostics::lspi(
                    &publication.lspi,
                    &publication.lspi_iterations,
                    &publication.coverage,
                    publication.lspi_stable,
                )?
                .with_portable_model(),
                publication,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    write_run(root, "lspi", &reference, lspi)
}

#[cfg(test)]
mod tests {
    use super::gridworld::{EAST, ExactGrid, SOUTH, START};
    use super::record_gridworld;
    use rl::{
        Result,
        core::action::FiniteActionKey,
        diagnostics::{
            reference::{DecisionPolicy, decision_loss},
            schema::{Record, RecordKind},
            storage::{ReadLimits, Recording},
            typed::{DiagnosticInput, TypedEpisodeSource, TypedPayload},
        },
    };

    #[test]
    fn exact_grid_reference_certifies_the_cost_of_the_upper_route() -> Result<()> {
        let grid = ExactGrid::new()?;
        let state = grid.solution(START).unwrap();
        assert_eq!(state.value, 1.0);
        assert_eq!(*state.greedy_action(), SOUTH);
        assert_eq!(state.action_values()[0].action, EAST);
        assert_eq!(state.action_values()[0].value, 0.5);
        assert_eq!(state.action_values()[1].value, 1.0);
        let wrong_turn = decision_loss(
            &[0.5, 1.0],
            &[true, true],
            DecisionPolicy::Deterministic(EAST),
        )?;
        assert_eq!(wrong_turn.loss, 0.5);
        Ok(())
    }

    #[test]
    fn three_learners_publish_comparable_checkpoints_with_their_factual_evidence() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let output = temporary.path().join("gridworld");
        record_gridworld(&output)?;
        for name in ["tabular", "fqi", "lspi"] {
            let recording = Recording::open(output.join(name), ReadLimits::default())?;
            assert!(recording.complete());
            let mut published = Vec::new();
            let mut optimality_gaps = Vec::new();
            let mut regret = Vec::new();
            let mut visited = false;
            let mut factual = false;
            let condition_metric = match name {
                "fqi" => Some("fit.fqi.augmented_design.condition_number"),
                "lspi" => Some("fit.lspi.moment_system.condition_number"),
                _ => None,
            };
            let mut conditioned = condition_metric.is_none();
            for (index, entry) in recording.entries().iter().enumerate() {
                match entry.kind {
                    RecordKind::FinitePredictions
                        if entry.model_role == "online"
                            && entry.probe_set.as_deref() == Some("grid-cells") =>
                    {
                        let chunk = recording.load(index)?;
                        let TypedPayload::Predictions(predictions) = TypedPayload::read(&chunk)?
                        else {
                            panic!("expected grid predictions")
                        };
                        let start = predictions
                            .rows
                            .iter()
                            .find(|row| row.row_id == START as u64)
                            .unwrap();
                        assert_eq!(start.candidates.len(), 2);
                        if entry.revision == 0 {
                            assert_eq!(start.selected, Some(FiniteActionKey::Discrete(EAST)));
                        } else if entry.revision == 3 {
                            assert_eq!(start.selected, Some(FiniteActionKey::Discrete(SOUTH)));
                        }
                        published.push(entry.revision);
                    }
                    RecordKind::Metrics => {
                        let chunk = recording.load(index)?;
                        let Record::Metrics(metrics) = &chunk.record else {
                            panic!("expected reference metrics")
                        };
                        if let Some(&gap) = metrics.values.get("reference.optimality_gap") {
                            optimality_gaps.push((entry.revision, gap));
                        }
                        if let Some(&loss) = metrics.values.get("reference.decision_regret") {
                            regret.push((entry.revision, loss));
                        }
                        if entry.revision == 3
                            && condition_metric.is_some_and(|key| metrics.values.contains_key(key))
                        {
                            conditioned = true;
                        }
                    }
                    RecordKind::TypedProbes
                        if entry.revision == 3
                            && entry.probe_set.as_deref() == Some("factual-visitation") =>
                    {
                        let chunk = recording.load(index)?;
                        let TypedPayload::Probes(probes) = TypedPayload::read(&chunk)? else {
                            panic!("expected factual probes")
                        };
                        visited = probes.fixed_actions.as_ref().is_some_and(|actions| actions.len() == probes.inputs.len())
                            && probes.inputs.iter().any(|input| matches!(input, DiagnosticInput::History(history) if history.timestep() > 0));
                    }
                    RecordKind::TypedEpisode
                        if entry.revision == 3
                            && entry
                                .trajectory
                                .as_deref()
                                .is_some_and(|id| id.starts_with("factual-")) =>
                    {
                        let chunk = recording.load(index)?;
                        let TypedPayload::Episode(episode) = TypedPayload::read(&chunk)? else {
                            panic!("expected factual episode")
                        };
                        factual = matches!(episode.source, TypedEpisodeSource::Factual { .. });
                    }
                    _ => {}
                }
            }
            assert_eq!(published, [0, 1, 2, 3], "{name}");
            assert_eq!(optimality_gaps.first(), Some(&(0, 2.5)), "{name}");
            assert_eq!(optimality_gaps.last(), Some(&(3, 0.0)), "{name}");
            assert!(
                matches!(regret.first(), Some((0, loss)) if *loss > 0.0),
                "{name}"
            );
            assert_eq!(regret.last(), Some(&(3, 0.0)), "{name}");
            assert!(visited && factual, "{name}");
            assert!(conditioned, "{name} has full coverage of its legal basis");
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let output = args
        .next()
        .map_or_else(|| PathBuf::from("runs/diagnostic-gridworld"), PathBuf::from);
    if args.next().is_some() {
        return Err(Error::Config(
            "usage: diagnostic_gridworld [output-directory]".into(),
        ));
    }
    record_gridworld(&output)?;
    for name in ["tabular", "fqi", "lspi"] {
        println!("{}", output.join(name).display());
    }
    Ok(())
}
