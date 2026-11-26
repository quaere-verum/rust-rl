#[path = "../examples/support/structured_control.rs"]
mod workflow;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl::approx::batch::{BatchOptions, DecisionSpan, FactualBatch};
use rl::{
    Result,
    algorithms::{fqi, lspi},
    approx::{
        FeatureMap, FeatureSpec, StatisticalActor,
        regression::{DiagonalPenalty, LinearModel, WeightedRidge},
    },
    core::{
        action::{Action, ActionArgument, ConditionalAction},
        actor::ActMode,
        env::{Env, EpisodeStart, Observation, PublicHistory, Step},
        reward::BatchObjective,
        stateful::{HistoryInput, PreviousInput, StatefulActor},
        structured::{EntityId, FeatureIdentity},
        temporal::{BehaviorMetadata, TemporalSink, TemporalTransition},
    },
    data::temporal::EpisodeStore,
    planning::finite::Status,
};
use std::{num::NonZeroUsize, sync::Arc};
use workflow::{RelayKey, SignalGraph, exact_plan, train_tabular};

fn inspect() -> Action {
    Action::Conditional(Arc::new(ConditionalAction {
        operation: 0,
        argument: None,
    }))
}

fn route(right: bool) -> Action {
    Action::Conditional(Arc::new(ConditionalAction {
        operation: 1,
        argument: Some(ActionArgument::Entity(EntityId {
            kind: 1,
            id: if right { 29 } else { 11 },
        })),
    }))
}

#[test]
fn routing_becomes_legal_only_after_inspection_adds_the_sensor_node() -> Result<()> {
    let mut env = SignalGraph::new(7);
    let initial = env.reset(None)?.observation;
    assert_eq!(initial.as_graph().unwrap().nodes.len(), 3);
    assert_eq!(initial.finite_actions(env.action_space())?, [inspect()]);
    assert!(env.step(&route(false)).is_err());

    let inspected = env.step(&inspect())?;
    let graph = inspected.observation.as_graph().unwrap();
    assert_eq!(
        (graph.tick, graph.nodes.len(), graph.edges.len()),
        (1, 4, 3)
    );
    assert!(!inspected.ended());
    assert_eq!(
        inspected.observation.finite_actions(env.action_space())?,
        [route(false), route(true)]
    );
    assert!(env.step(&inspect()).is_err());

    let finished = env.step(&route(false))?;
    assert!(finished.terminated);
    assert!(!finished.truncated);
    let terminal = finished.observation.as_graph().unwrap();
    assert_eq!(terminal.observation.events, graph.observation.events);
    assert!(terminal.observation.candidates.is_empty());
    Ok(())
}

fn routing_history(seed: u64) -> Result<(SignalGraph, Observation, Step)> {
    let mut env = SignalGraph::new(seed);
    let initial = env.reset(None)?.observation;
    let inspected = env.step(&inspect())?;
    Ok((env, initial, inspected))
}

fn hint_right(observation: &Observation) -> bool {
    observation.as_structured().unwrap().events[0].features[0] == 1.0
}

fn graph_episode_bank() -> Result<EpisodeStore> {
    let template = SignalGraph::new(0);
    let mut store = EpisodeStore::new(
        32,
        template.observation_space(),
        template.action_space().clone(),
    )?;
    let mut counts = [[[0; 2]; 2]; 2];
    let mut episode_id = 0;
    for seed in 0..64 {
        for right in [false, true] {
            let (mut env, reset, inspected) = routing_history(seed)?;
            let hint = hint_right(&inspected.observation);
            let action = route(right);
            let terminal = env.step(&action)?;
            let delivered = terminal.reward.total() == 1.0;
            let count = &mut counts[usize::from(hint)][usize::from(right)][usize::from(delivered)];
            // Keep a fixed factual bank with the exact three-to-one outcome frequencies.
            let quota = if delivered == (hint == right) { 3 } else { 1 };
            if *count == quota {
                continue;
            }
            *count += 1;
            store.begin_episode(episode_id, &EpisodeStart::fresh(reset.clone()))?;
            let route_frame = inspected.observation.clone();
            for (timestep, (observation, action, step)) in [
                (reset, inspect(), inspected),
                (route_frame, action, terminal),
            ]
            .into_iter()
            .enumerate()
            {
                store.push(&TemporalTransition {
                    episode_id,
                    timestep,
                    observation,
                    action,
                    reward: step.reward,
                    next_observation: step.observation,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    behavior: BehaviorMetadata {
                        revision: 0,
                        log_prob: None,
                    },
                })?;
            }
            episode_id += 1;
            if episode_id == 16 {
                return Ok(store);
            }
        }
    }
    panic!("the fixed relay seeds did not cover all public outcomes")
}

struct RelayBasis(FeatureSpec);

impl RelayBasis {
    fn new() -> Self {
        Self(FeatureSpec {
            namespace: "test.graph_relay".into(),
            version: 1,
            terms: ["inspect", "follow_hint", "oppose_hint"]
                .map(|name| FeatureIdentity::new(name, "indicator"))
                .into(),
            configuration: vec![],
            representation: None,
        })
    }
}

impl FeatureMap for RelayBasis {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let frame = history.observation();
        assert!(frame.as_graph().is_some());
        Ok(actions
            .iter()
            .map(|action| {
                let mut features = vec![0.0; 3];
                let term = if *action == inspect() {
                    0
                } else if *action == route(hint_right(frame)) {
                    1
                } else {
                    2
                };
                features[term] = 1.0;
                features
            })
            .collect())
    }
}

fn graph_batch<'data, 'map>(
    store: &'data EpisodeStore,
    features: &'map RelayBasis,
) -> Result<FactualBatch<'data, 'map>> {
    FactualBatch::from_store(
        store,
        features,
        BatchOptions {
            objective: BatchObjective::Physical { gamma: 1.0 },
            span: DecisionSpan::Single,
        },
    )
}

fn assert_statistical_relay_policy(
    actor: &StatisticalActor<RelayBasis, LinearModel>,
) -> Result<()> {
    let mut seen = [false; 2];
    for seed in 0..16 {
        let (mut env, reset, inspected) = routing_history(seed)?;
        let history = PublicHistory::new(reset);
        let initial = actor.action_values(&history)?;
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].0, inspect());
        assert!((initial[0].1 - 0.75).abs() < 1e-9);

        let hint = hint_right(&inspected.observation);
        seen[usize::from(hint)] = true;
        let history = history.append(inspect(), inspected.reward, inspected.observation)?;
        let values = actor.action_values(&history)?;
        assert_eq!(values.len(), 2);
        for ((action, value), right) in values.into_iter().zip([false, true]) {
            assert_eq!(action, route(right));
            let expected = if hint == right { 0.75 } else { 0.25 };
            assert!((value - expected).abs() < 1e-9, "{value} != {expected}");
        }
        let selected = actor.greedy(&history)?;
        assert_eq!(selected, route(hint));
        assert!(env.step(&selected)?.terminated);
    }
    assert_eq!(seen, [true, true]);
    Ok(())
}

#[test]
fn fitted_q_iteration_learns_and_executes_on_a_conditional_graph() -> Result<()> {
    let store = graph_episode_bank()?;
    let features = RelayBasis::new();
    let fit = fqi::fit(
        &graph_batch(&store, &features)?,
        &WeightedRidge {
            penalty: DiagonalPenalty::Uniform(0.0),
            ..Default::default()
        },
        fqi::FqiConfig {
            sweeps: NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        },
        fqi::Initial::Zero,
    )?;
    assert_statistical_relay_policy(&StatisticalActor::new(features, fit.model, 1, 0.0)?)
}

#[test]
fn least_squares_policy_iteration_learns_and_executes_on_a_conditional_graph() -> Result<()> {
    let store = graph_episode_bank()?;
    let features = RelayBasis::new();
    let fit = lspi::fit(
        &graph_batch(&store, &features)?,
        &lspi::LspiConfig {
            penalty: DiagonalPenalty::Uniform(0.0),
            ..Default::default()
        },
        None,
    )?;
    assert!(fit.stable);
    assert_statistical_relay_policy(&StatisticalActor::new(features, fit.model, 1, 0.0)?)
}

#[test]
fn the_same_public_history_can_have_opposite_factual_outcomes() -> Result<()> {
    let mut histories = Vec::new();
    for seed in 0..32 {
        let (mut env, initial, inspected) = routing_history(seed)?;
        assert_eq!(inspected.reward.total(), 0.0);
        let reward = env.step(&route(false))?.reward.total();
        histories.push((initial, inspected.observation, reward));
    }
    assert!(histories.iter().enumerate().any(|(index, (a, b, reward))| {
        histories[..index]
            .iter()
            .any(|(other_a, other_b, other_reward)| {
                a == other_a && b == other_b && (*reward - *other_reward).abs() == 1.0
            })
    }));
    Ok(())
}

fn reverse_storage_order(observation: &Observation) -> Observation {
    let mut graph = observation.as_graph().unwrap().clone();
    graph.nodes.reverse();
    graph.edges.reverse();
    graph.observation.entities.reverse();
    graph.observation.candidates.reverse();
    Observation::Graph(Arc::new(graph))
}

#[test]
fn learned_tabular_policy_matches_the_exact_plan_for_both_public_hints() -> Result<()> {
    let exact = exact_plan()?;
    assert_eq!(exact.status, Status::Converged);
    assert_eq!(exact.residual, 0.0);
    assert_eq!(exact.states.len(), 3);
    let value = |key| exact.states.iter().find(|row| row.context == key).unwrap();
    let inspect_value = value(RelayKey::Inspect);
    let optimal_inspection = Action::Conditional(Arc::new(*inspect_value.greedy_action()));
    assert_eq!(inspect_value.value, 0.75);
    assert_eq!(optimal_inspection, inspect());
    for (hint_right, expected) in [(false, [0.75, 0.25]), (true, [0.25, 0.75])] {
        let row = value(RelayKey::Route { hint_right });
        assert_eq!(row.value, 0.75);
        assert_eq!(
            row.action_values()
                .iter()
                .map(|action| (Action::Conditional(Arc::new(action.action)), action.value))
                .collect::<Vec<_>>(),
            [(route(false), expected[0]), (route(true), expected[1])]
        );
    }

    let actor = train_tabular(768)?;
    let mut rng = ChaCha8Rng::seed_from_u64(51);
    let mut hints = [false; 2];
    let inspection = inspect();
    for seed in 0..16 {
        let (env, initial, inspected) = routing_history(seed)?;
        let memory = actor.assimilate(
            &actor.initial_state()?,
            HistoryInput {
                observation: &initial,
                previous: None,
            },
        )?;
        assert_eq!(
            actor
                .decide(&memory, &initial, ActMode::Deterministic, &mut rng)?
                .action,
            optimal_inspection,
        );
        let right = hint_right(&inspected.observation);
        hints[usize::from(right)] = true;
        let optimal = Action::Conditional(Arc::new(
            *value(RelayKey::Route { hint_right: right }).greedy_action(),
        ));
        let permuted = reverse_storage_order(&inspected.observation);
        for observation in [&inspected.observation, &permuted] {
            let memory = actor.assimilate(
                &memory,
                HistoryInput {
                    observation,
                    previous: Some(PreviousInput::from_transition(
                        &inspection,
                        &inspected.reward,
                        &initial,
                    )),
                },
            )?;
            let decision = actor.decide(&memory, observation, ActMode::Deterministic, &mut rng)?;
            observation.check_action(env.action_space(), &decision.action)?;
            assert_eq!(decision.action, optimal);
        }
    }
    assert_eq!(hints, [true, true]);
    Ok(())
}

#[cfg(feature = "neural")]
#[test]
fn graph_ddqn_updates_from_the_same_environment_and_restores_its_values() -> Result<()> {
    use rl::{
        algorithms::candidate_ddqn::CandidateQActor,
        core::stateful::BehaviorPublisher,
        data::temporal::EpisodeStore,
        nn::structured::GraphStructuredQ,
        train::checkpoint::{load_run, save_run},
    };
    use workflow::neural::{EXPLORATION, Trainer, build_learner, build_trainer};

    let (env, _, inspected) = routing_history(8)?;
    let frame = &inspected.observation;
    let mut learner = build_learner(&env, 53)?;
    let actor = learner.behavior_actor(&EXPLORATION, 0)?;
    let before = actor.action_values(frame)?;
    let mut permuted = actor.action_values(&reverse_storage_order(frame))?;
    permuted.sort_unstable_by_key(|(action, _)| *action);
    assert_eq!(before.len(), 2);
    for ((key, value), (moved_key, moved_value)) in before.iter().zip(&permuted) {
        assert_eq!(key, moved_key);
        assert!((value - moved_value).abs() < 1e-5);
    }

    let mut rewired = frame.as_graph().unwrap().clone();
    rewired.edges[0].to = rewired.edges[1].to;
    assert_ne!(
        before,
        actor.action_values(&Observation::Graph(Arc::new(rewired)))?
    );

    let mut trainer = build_trainer(vec![env.clone(), SignalGraph::new(19)], &learner, 61)?;
    let mut reports = 0;
    // Seven ticks leave both lanes between inspection and the terminal reward.
    for _ in 0..7 {
        let progress = trainer.step(&mut learner)?;
        reports += progress.reports.len();
        assert!(progress.reports.iter().all(|report| {
            report.updates > 0 && report.metrics.values().all(|metric| metric.is_finite())
        }));
    }
    assert_eq!(trainer.counters().env_steps, 14);
    assert!(reports > 0 && learner.updates() > 0);
    assert!(trainer.last_update_batch().unwrap().samples.iter().all(|sample| {
        sample.observations.iter().all(
            |observation| matches!(observation, Observation::Graph(graph) if !graph.edges.is_empty()),
        )
    }));

    let replay: EpisodeStore = serde_json::from_slice(&serde_json::to_vec(trainer.store())?)?;
    assert!(replay.episodes().eq(trainer.store().episodes()));

    let revision = trainer.behavior_revision();
    let actor = learner.behavior_actor(&EXPLORATION, revision)?;
    let after = actor.action_values(frame)?;
    assert_ne!(before, after);
    let dir = tempfile::tempdir()?;
    let snapshot = dir.path().join("structured-control.run");
    save_run(&snapshot, &learner, &trainer)?;
    let mut restored = build_learner(&env, 67)?;
    let mut resumed: Trainer = load_run(&snapshot, &mut restored)?;
    assert_eq!(
        after,
        restored
            .behavior_actor(&EXPLORATION, revision)?
            .action_values(frame)?
    );
    let actor: CandidateQActor<GraphStructuredQ> =
        serde_json::from_slice(&serde_json::to_vec(&actor)?)?;
    assert_eq!(after, actor.action_values(frame)?);

    for _ in 0..4 {
        trainer.step(&mut learner)?;
        resumed.step(&mut restored)?;
        assert_eq!(trainer.counters(), resumed.counters());
        assert_eq!(trainer.last_windows(), resumed.last_windows());
        assert!(trainer.store().episodes().eq(resumed.store().episodes()));
    }
    let revision = trainer.behavior_revision();
    assert_eq!(
        learner
            .behavior_actor(&EXPLORATION, revision)?
            .action_values(frame)?,
        restored
            .behavior_actor(&EXPLORATION, revision)?
            .action_values(frame)?
    );
    Ok(())
}
