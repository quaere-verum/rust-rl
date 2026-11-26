use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    algorithms::tabular::{EpisodeEnd, EpisodeStep, QTable, TabularActor},
    core::{
        action::{
            Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation,
        },
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, GraphObservationSpace, Observation, ObservationSpace, Step},
        graph::{
            EdgeStatus, GRAPH_SCHEMA_VERSION, GraphEdge, GraphNode, GraphNodeId, GraphObservation,
            GraphSchema,
        },
        state_key::{StateKeyBuilder, StateKeyKind},
        stateful::{HistoryInput, PreviousInput},
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, EventStamp, FeatureIdentity,
            INCREMENTAL_EVENT_CONVENTION, ObservationSchema, PublicEvent, SpatialObservation,
            StructuredObservation,
        },
        temporal::TemporalSpaces,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[path = "structured_control/reference.rs"]
mod reference;
pub use reference::exact_plan;

const NAMESPACE: &str = "example.signal_graph";
const PORTS: [EntityId; 2] = [EntityId { kind: 1, id: 11 }, EntityId { kind: 1, id: 29 }];
const SENSOR: EntityId = EntityId { kind: 2, id: 47 };
const INSPECT: ConditionalAction = ConditionalAction {
    operation: 0,
    argument: None,
};

fn route_key(right: bool) -> ConditionalAction {
    ConditionalAction {
        operation: 1,
        argument: Some(ActionArgument::Entity(PORTS[usize::from(right)])),
    }
}

fn observation_schema() -> ObservationSchema {
    ObservationSchema {
        namespace: NAMESPACE.into(),
        version: 1,
        spatial: vec![FeatureIdentity::new("relay_station", "present")],
        entities: vec![FeatureIdentity::new("role", "left_right_sensor_index")],
        events: vec![FeatureIdentity::new("hint_right", "binary")],
        scalars: vec![FeatureIdentity::new("phase", "decision_index")],
        vocabularies: vec![],
        conventions: vec![INCREMENTAL_EVENT_CONVENTION.into()],
    }
}

fn action_grammar() -> ActionGrammar {
    ActionGrammar {
        namespace: NAMESPACE.into(),
        version: 1,
        operations: vec![
            Operation {
                name: "inspect".into(),
                argument: ArgumentKind::None,
            },
            Operation {
                name: "route".into(),
                argument: ArgumentKind::Entity,
            },
        ],
        candidate_features: vec![FeatureIdentity::new("duration", "primitive_steps")],
    }
}

fn graph_schema() -> GraphSchema {
    GraphSchema {
        namespace: NAMESPACE.into(),
        version: GRAPH_SCHEMA_VERSION,
        node_features: vec![FeatureIdentity::new(
            "role",
            "left_right_sensor_agent_index",
        )],
        edge_features: vec![],
        edge_types: vec!["route".into(), "reports".into()],
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Phase {
    Inspect,
    Route,
    End,
}

/// Inspect before routing to one of two public ports. The sensor hints at the
/// hidden live port with 75% accuracy. Its cue remains in subsequent frames.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignalGraph {
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    rng: ChaCha8Rng,
    phase: Phase,
    live_right: bool,
    hint_right: bool,
}

impl SignalGraph {
    pub fn new(seed: u64) -> Self {
        Self {
            observation_space: ObservationSpace::Graph(Arc::new(GraphObservationSpace {
                base: observation_schema(),
                graph: graph_schema(),
            })),
            action_space: ActionSpace::Conditional(Arc::new(action_grammar())),
            rng: ChaCha8Rng::seed_from_u64(seed),
            phase: Phase::End,
            live_right: false,
            hint_right: false,
        }
    }

    fn observation(&self) -> Observation {
        let mut entities = vec![
            EntityRow {
                id: PORTS[0],
                features: vec![0.0],
            },
            EntityRow {
                id: PORTS[1],
                features: vec![1.0],
            },
        ];
        let mut events = Vec::new();
        let mut edges: Vec<_> = PORTS
            .iter()
            .enumerate()
            .map(|(index, &port)| GraphEdge {
                id: index as u64,
                from: GraphNodeId::Agent,
                to: GraphNodeId::Entity(port),
                kind: 0,
                tick: 0,
                status: EdgeStatus::Observed,
                conditional: false,
                features: vec![],
            })
            .collect();
        if self.phase != Phase::Inspect {
            entities.push(EntityRow {
                id: SENSOR,
                features: vec![2.0],
            });
            edges.push(GraphEdge {
                id: 2,
                from: GraphNodeId::Entity(SENSOR),
                to: GraphNodeId::Agent,
                kind: 1,
                tick: 1,
                status: EdgeStatus::Observed,
                conditional: false,
                features: vec![],
            });
            events.push(PublicEvent {
                features: Arc::from([f32::from(u8::from(self.hint_right))]),
                entity: Some(SENSOR),
                context: Arc::from([SENSOR]),
                stamp: Some(EventStamp {
                    source: 0,
                    ordinal: 0,
                    tick: 1,
                }),
            });
        }
        let nodes = std::iter::once(GraphNode {
            id: GraphNodeId::Agent,
            features: vec![3.0],
        })
        .chain(entities.iter().map(|entity| GraphNode {
            id: GraphNodeId::Entity(entity.id),
            features: entity.features.clone(),
        }))
        .collect();
        let candidate = |action| ActionCandidate {
            action,
            features: vec![1.0],
            enabled: true,
        };
        let candidates = match self.phase {
            Phase::Inspect => vec![candidate(INSPECT)],
            Phase::Route => vec![candidate(route_key(false)), candidate(route_key(true))],
            Phase::End => vec![],
        };
        Observation::Graph(Arc::new(GraphObservation {
            namespace: NAMESPACE.into(),
            version: GRAPH_SCHEMA_VERSION,
            tick: self.phase as u64,
            observation: StructuredObservation {
                spatial: Some(SpatialObservation {
                    width: 1,
                    height: 1,
                    cells: vec![1.0],
                }),
                primary_region: 0,
                regions: vec![],
                entities,
                events,
                scalars: vec![self.phase as u8 as f32],
                candidates,
            },
            nodes,
            edges,
        }))
    }
}

impl Env for SignalGraph {
    fn observation_space(&self) -> ObservationSpace {
        self.observation_space.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.live_right = self.rng.random();
        self.hint_right = if self.rng.random_bool(0.75) {
            self.live_right
        } else {
            !self.live_right
        };
        self.phase = Phase::Inspect;
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.observation()
            .check_action(&self.action_space, action)?;
        let reward = match self.phase {
            Phase::Inspect => {
                self.phase = Phase::Route;
                0.0
            }
            Phase::Route => {
                self.phase = Phase::End;
                let delivered = matches!(
                    action,
                    Action::Conditional(key) if **key == route_key(self.live_right)
                );
                f32::from(u8::from(delivered))
            }
            Phase::End => unreachable!("a terminal frame has no legal action"),
        };
        Ok(Step {
            observation: self.observation(),
            reward: reward.into(),
            terminated: self.phase == Phase::End,
            truncated: false,
        })
    }
}

impl Snapshot for SignalGraph {
    fn validate_snapshot(&self) -> Result<()> {
        if !matches!(
            &self.observation_space,
            ObservationSpace::Graph(space)
                if space.base == observation_schema() && space.graph == graph_schema()
        ) || !matches!(
            &self.action_space,
            ActionSpace::Conditional(grammar) if **grammar == action_grammar()
        ) {
            return Err(Error::Config("relay snapshot schema differs".into()));
        }
        self.observation().validate_in(
            &self.observation_space,
            &self.action_space,
            self.phase != Phase::End,
        )
    }
}

/// The hint identifies an exact finite posterior under the symmetric prior.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RelayKey {
    Inspect,
    Route { hint_right: bool },
    End,
}

#[derive(Clone, Copy)]
pub struct RelayKeys;

impl StateKeyBuilder for RelayKeys {
    type Memory = Option<RelayKey>;
    type Key = RelayKey;

    fn kind(&self) -> StateKeyKind {
        StateKeyKind::Exact
    }

    fn initial_memory(&self) -> Result<Self::Memory> {
        Ok(None)
    }

    fn observe(&self, prior: &Self::Memory, input: HistoryInput<'_>) -> Result<Self::Memory> {
        let graph = input
            .observation
            .as_graph()
            .ok_or_else(|| Error::Shape("relay requires a sparse graph".into()))?;
        let state = match (prior, input.previous, graph.tick) {
            (None, None, 0) => RelayKey::Inspect,
            (Some(RelayKey::Inspect), Some(previous), 1) => {
                let previous = previous
                    .factual
                    .and_then(|factual| factual.observation.as_structured())
                    .ok_or_else(|| Error::State("missing public relay predecessor".into()))?;
                let introduced = graph.observation.introduced_events(Some(previous))?;
                let [event] = introduced else {
                    return Err(Error::State("inspection must introduce one hint".into()));
                };
                let hint_right = match event.features.as_ref() {
                    [0.0] => false,
                    [1.0] => true,
                    _ => return Err(Error::State("invalid relay hint".into())),
                };
                RelayKey::Route { hint_right }
            }
            (Some(RelayKey::Route { .. }), Some(_), 2) => RelayKey::End,
            _ => return Err(Error::State("invalid relay history".into())),
        };
        Ok(Some(state))
    }

    fn key(&self, memory: &Self::Memory) -> Result<Self::Key> {
        memory.ok_or_else(|| Error::State("relay has no public observation".into()))
    }
}

pub fn train_tabular(episodes: usize) -> Result<TabularActor<RelayKeys>> {
    let mut env = SignalGraph::new(17);
    let mut rng = ChaCha8Rng::seed_from_u64(23);
    let mut table = QTable::new(0.0)?;
    let keys = RelayKeys;
    for _ in 0..episodes {
        let mut observation = env.reset(None)?.observation;
        let mut memory = keys.observe(
            &keys.initial_memory()?,
            HistoryInput {
                observation: &observation,
                previous: None,
            },
        )?;
        let mut trajectory = Vec::with_capacity(2);
        loop {
            let state = keys.key(&memory)?;
            let actions = observation.finite_actions(env.action_space())?;
            let values = actions
                .iter()
                .map(|action| observation.finite_action_key(action))
                .collect::<Result<Vec<_>>>()?;
            let index = table.epsilon_greedy(&state, &values, 0.3, &mut rng)?.index;
            let step = env.step(&actions[index])?;
            let ended = step.ended();
            memory = keys.observe(
                &memory,
                HistoryInput {
                    observation: &step.observation,
                    previous: Some(PreviousInput::from_transition(
                        &actions[index],
                        &step.reward,
                        &observation,
                    )),
                },
            )?;
            trajectory.push(EpisodeStep {
                state,
                action: values[index],
                reward: step.reward,
            });
            if ended {
                break;
            }
            observation = step.observation;
        }
        table.monte_carlo_first_visit(&trajectory, EpisodeEnd::Terminated, 1.0)?;
    }
    TabularActor::new(
        keys,
        table,
        TemporalSpaces {
            observations: env.observation_space(),
            actions: env.action_space().clone(),
        },
        0,
        0.0,
    )
}

#[cfg(feature = "neural")]
pub mod neural {
    use super::{SignalGraph, action_grammar, graph_schema, observation_schema};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rl::{
        Result,
        algorithms::candidate_ddqn::{
            CandidateDdqn, CandidateDdqnConfig, CandidateEpsilonSchedule, CandidateExploration,
            CandidateQActor, ExplorationDistribution,
        },
        collect::vector_env::SyncVectorEnv,
        core::{env::Env, stateful::BehaviorPublisher},
        data::temporal::SequenceConfig,
        nn::{
            device,
            structured::{
                GraphStructuredQ, GraphStructuredQConfig, InputSchema, SemanticInput,
                StructuredModelConfig,
            },
        },
        train::{
            recurrent::TemporalTrainer, schedule::Schedule,
            temporal_config::TemporalOffPolicyConfig,
        },
    };

    pub const EXPLORATION: CandidateExploration = CandidateExploration {
        epsilon: 0.35,
        distribution: ExplorationDistribution::CompleteActions,
    };
    pub type Learner = CandidateDdqn<GraphStructuredQ>;
    pub type Trainer = TemporalTrainer<
        SyncVectorEnv<SignalGraph>,
        CandidateQActor<GraphStructuredQ>,
        CandidateEpsilonSchedule,
    >;

    pub fn build_learner(env: &SignalGraph, seed: u64) -> Result<Learner> {
        let observation = observation_schema();
        let actions = action_grammar();
        let input = InputSchema {
            version: 2,
            spatial_channels: observation.spatial.len(),
            entity_features: observation.entities.len(),
            event_features: observation.events.len(),
            scalar_features: observation.scalars.len(),
            candidate_features: actions.candidate_features.len(),
            operations: actions
                .operations
                .iter()
                .map(|operation| operation.argument)
                .collect(),
            semantics: SemanticInput {
                version: 1,
                namespace: observation.namespace,
                observation_version: observation.version,
                action_version: actions.version,
                spatial: observation.spatial,
                entities: observation.entities,
                events: observation.events,
                scalars: observation.scalars,
                candidates: actions.candidate_features,
                operations: actions
                    .operations
                    .into_iter()
                    .map(|operation| operation.name)
                    .collect(),
                vocabularies: observation.vocabularies,
                conventions: observation.conventions,
            },
        };
        let config = GraphStructuredQConfig {
            version: 1,
            base: StructuredModelConfig::small(input),
            graph: graph_schema(),
            graph_width: 8,
        };
        let model = GraphStructuredQ::new(config, &mut ChaCha8Rng::seed_from_u64(seed), &device())?;
        CandidateDdqn::new(
            model,
            CandidateDdqnConfig::default(),
            env.observation_space(),
            env.action_space().clone(),
        )
    }

    pub fn build_trainer(envs: Vec<SignalGraph>, learner: &Learner, seed: u64) -> Result<Trainer> {
        TemporalTrainer::new(
            SyncVectorEnv::new(envs)?,
            learner.behavior_actor(&EXPLORATION, 0)?,
            TemporalOffPolicyConfig {
                n_step: 1,
                gamma: learner.config().gamma,
                replay_capacity: 64,
                batch_size: 2,
                warmup_transitions: 0,
                updates_per_transition: 0.5,
            },
            SequenceConfig {
                n_step: 1,
                learn_len: 1,
                burn_in: 0,
            },
            CandidateEpsilonSchedule {
                epsilon: Schedule::Constant(EXPLORATION.epsilon),
                distribution: EXPLORATION.distribution,
            },
            seed,
        )
    }
}
