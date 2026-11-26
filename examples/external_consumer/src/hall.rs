use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{
            Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation,
        },
        env::{Env, EpisodeStart, GraphObservationSpace, Observation, ObservationSpace, Step},
        graph::{
            EdgeStatus, GRAPH_SCHEMA_VERSION, GraphEdge, GraphNode, GraphNodeId, GraphObservation,
            GraphSchema,
        },
        reward::Reward,
        state_key::{StateKeyBuilder, StateKeyKind},
        stateful::HistoryInput,
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity, ObservationSchema,
            SpatialObservation, StructuredObservation,
        },
        temporal::TemporalSpaces,
    },
};

const NAMESPACE: &str = "external-consumer.door-hall";
pub const PASSAGE: u64 = 10;
pub const LEFT_DOOR: u64 = 20;
pub const RIGHT_DOOR: u64 = 30;
pub const SAFE_EXIT: u64 = 40;
pub const ARRIVAL_CHANCE: f64 = 0.75;
pub const GAMMA: f64 = 0.75;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Side {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum State {
    Entrance(Side),
    Fork(Side),
    Done,
}

impl State {
    pub fn menu(self) -> &'static [u64] {
        match self {
            Self::Entrance(_) => &[PASSAGE, SAFE_EXIT],
            Self::Fork(_) => &[LEFT_DOOR, RIGHT_DOOR],
            Self::Done => &[],
        }
    }

    pub fn outcome(self, action: ConditionalAction, arrived: bool) -> Result<(Self, Reward)> {
        match (self, target(action)?) {
            (Self::Entrance(side), PASSAGE) => Ok((
                if arrived {
                    Self::Fork(side)
                } else {
                    Self::Done
                },
                Reward::from_primitives(vec![0., 4.])?,
            )),
            (Self::Entrance(_), SAFE_EXIT) => Ok((Self::Done, 2.0.into())),
            (Self::Fork(Side::Left), LEFT_DOOR) | (Self::Fork(Side::Right), RIGHT_DOOR) => {
                Ok((Self::Done, 8.0.into()))
            }
            (Self::Fork(_), LEFT_DOOR | RIGHT_DOOR) => Ok((Self::Done, (-1.0).into())),
            _ => Err(Error::Action("door is absent at this decision".into())),
        }
    }

    fn observation(self, tick: u64) -> Observation {
        let cue = match self {
            Self::Entrance(Side::Left) => -1.,
            Self::Entrance(Side::Right) => 1.,
            Self::Fork(_) => 0.,
            Self::Done => 2.,
        };
        let doors = self.menu();
        let mut nodes = vec![GraphNode {
            id: GraphNodeId::Agent,
            features: vec![],
        }];
        nodes.extend(doors.iter().map(|&door| GraphNode {
            id: GraphNodeId::Entity(entity(door)),
            features: vec![],
        }));
        Observation::Graph(
            GraphObservation {
                namespace: NAMESPACE.into(),
                version: GRAPH_SCHEMA_VERSION,
                tick,
                observation: StructuredObservation {
                    spatial: Some(SpatialObservation {
                        width: 1,
                        height: 1,
                        cells: vec![1.],
                    }),
                    primary_region: 0,
                    regions: vec![],
                    entities: doors
                        .iter()
                        .map(|&door| EntityRow {
                            id: entity(door),
                            features: vec![door as f32 / 40.],
                        })
                        .collect(),
                    events: vec![],
                    scalars: vec![cue],
                    candidates: doors
                        .iter()
                        .map(|&door| ActionCandidate {
                            action: use_door(door),
                            features: vec![if door == PASSAGE { 2. } else { 1. }],
                            enabled: true,
                        })
                        .collect(),
                },
                nodes,
                edges: doors
                    .iter()
                    .map(|&door| GraphEdge {
                        id: door,
                        from: GraphNodeId::Agent,
                        to: GraphNodeId::Entity(entity(door)),
                        kind: 0,
                        tick,
                        status: EdgeStatus::Observed,
                        conditional: false,
                        features: vec![],
                    })
                    .collect(),
            }
            .into(),
        )
    }
}

pub struct DoorHall {
    side: Side,
    state: State,
    tick: u64,
    spaces: TemporalSpaces,
    rng: ChaCha8Rng,
}

impl DoorHall {
    pub fn new(side: Side) -> Self {
        Self {
            side,
            state: State::Done,
            tick: 0,
            spaces: spaces(),
            rng: ChaCha8Rng::seed_from_u64(0),
        }
    }
}

impl Env for DoorHall {
    fn observation_space(&self) -> ObservationSpace {
        self.spaces.observations.clone()
    }

    fn action_space(&self) -> &ActionSpace {
        &self.spaces.actions
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
        }
        self.state = State::Entrance(self.side);
        self.tick = 0;
        Ok(EpisodeStart::fresh(self.state.observation(self.tick)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.state
            .observation(self.tick)
            .check_action(self.action_space(), action)?;
        let Action::Conditional(action) = action else {
            return Err(Error::Action("expected a door action".into()));
        };
        let arrived = **action == use_door(PASSAGE) && self.rng.random_bool(ARRIVAL_CHANCE);
        let (state, reward) = self.state.outcome(**action, arrived)?;
        self.state = state;
        self.tick += reward.duration();
        Ok(Step {
            observation: state.observation(self.tick),
            reward,
            terminated: state == State::Done,
            truncated: false,
        })
    }
}

#[derive(Clone)]
pub struct RememberCue;

impl StateKeyBuilder for RememberCue {
    type Memory = Option<State>;
    type Key = State;

    fn kind(&self) -> StateKeyKind {
        StateKeyKind::Exact
    }

    fn initial_memory(&self) -> Result<Self::Memory> {
        Ok(None)
    }

    fn observe(&self, prior: &Self::Memory, input: HistoryInput<'_>) -> Result<Self::Memory> {
        let frame = input
            .observation
            .as_graph()
            .ok_or_else(|| Error::Shape("expected a door-hall graph".into()))?;
        let state = match (frame.observation.scalars.as_slice(), prior, input.previous) {
            ([-1.], None, None) => State::Entrance(Side::Left),
            ([1.], None, None) => State::Entrance(Side::Right),
            ([0.], Some(State::Entrance(side)), Some(_)) => State::Fork(*side),
            ([2.], Some(_), Some(_)) => State::Done,
            _ => {
                return Err(Error::State(
                    "door-hall history is missing its entrance cue".into(),
                ));
            }
        };
        Ok(Some(state))
    }

    fn key(&self, memory: &Self::Memory) -> Result<Self::Key> {
        memory.ok_or_else(|| Error::State("door-hall state is unobserved".into()))
    }
}

fn entity(door: u64) -> EntityId {
    EntityId { kind: 0, id: door }
}

pub fn use_door(door: u64) -> ConditionalAction {
    ConditionalAction {
        operation: 0,
        argument: Some(ActionArgument::Entity(entity(door))),
    }
}

pub fn target(action: ConditionalAction) -> Result<u64> {
    match action {
        ConditionalAction {
            operation: 0,
            argument: Some(ActionArgument::Entity(EntityId { kind: 0, id })),
        } => Ok(id),
        _ => Err(Error::Action("expected a door entity".into())),
    }
}

pub fn spaces() -> TemporalSpaces {
    let feature = |name| vec![FeatureIdentity::new(name, "identity")];
    let observation = GraphObservationSpace {
        base: ObservationSchema {
            namespace: NAMESPACE.into(),
            version: 1,
            spatial: feature("walkable"),
            entities: feature("door_marker"),
            events: feature("signal"),
            scalars: feature("cue_or_phase"),
            vocabularies: vec![],
            conventions: vec!["cue visible only at the entrance".into()],
        },
        graph: GraphSchema {
            namespace: NAMESPACE.into(),
            version: GRAPH_SCHEMA_VERSION,
            node_features: vec![],
            edge_features: vec![],
            edge_types: vec!["reachable".into()],
        },
    };
    let grammar = ActionGrammar {
        namespace: NAMESPACE.into(),
        version: 1,
        operations: vec![Operation {
            name: "use_door".into(),
            argument: ArgumentKind::Entity,
        }],
        candidate_features: feature("duration"),
    };
    TemporalSpaces {
        observations: ObservationSpace::Graph(observation.into()),
        actions: ActionSpace::Conditional(grammar.into()),
    }
}
