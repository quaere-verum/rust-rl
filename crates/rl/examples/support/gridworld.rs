use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    approx::{FeatureMap, FeatureSpec},
    core::{
        action::{Action, ActionSpace},
        actor::{ActMode, Actor},
        env::{
            DiscreteObservationSpace, Env, EpisodeStart, Observation, ObservationBatch,
            ObservationSpace, PublicHistory, Step,
        },
        structured::FeatureIdentity,
        temporal::TemporalSpaces,
    },
    diagnostics::{
        probes::ProbeSet,
        reference::{
            ConfigValue, ReferenceCheckpoint, ReferenceContext, ReferenceHorizon,
            ReferenceInformation, ReferenceKind, ReferenceLifetime, ReferenceObjective,
            ReferenceProvider, ReferenceQuality, ReferenceSpec, ReferenceTarget,
        },
        schema::ProbePopulation,
        typed::{DiagnosticInput, DisplayProjection},
        value::ValueConventions,
    },
    models::predictive::{
        EnumerableModel, OutcomeDistribution, PlanningModel, PlanningReturns, ReturnObjective,
        Transition, WeightedTransition,
    },
    planning::finite::{FiniteModel, Solution, StateValue, Status},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    num::{NonZeroU16, NonZeroU32},
};

const WIDTH: usize = 4;
const HEIGHT: usize = 3;
const CELLS: usize = WIDTH * HEIGHT;
pub(super) const START: usize = 0;
const TRAP: usize = 3;
const WALL: usize = 5;
const GOAL: usize = CELLS - 1;
const HIGH_ROUTE: usize = 1;
pub(super) const EAST: usize = 0;
pub(super) const SOUTH: usize = 1;
const DIRECTIONS: usize = 2;
pub(super) const MAX_STEPS: usize = WIDTH + HEIGHT - 2;
pub(super) const GAMMA: f64 = 1.0;
pub(super) const ACTIONS: ActionSpace = ActionSpace::Discrete { count: DIRECTIONS };
pub(super) const ENVIRONMENT: &str = "Directed Gridworld 4 × 3";

pub(super) fn states() -> impl DoubleEndedIterator<Item = usize> {
    (0..CELLS).filter(|state| live(*state))
}

fn live(state: usize) -> bool {
    state < CELLS && !matches!(state, WALL | GOAL | TRAP)
}

fn destination(state: usize, action: usize) -> Option<usize> {
    if !live(state) {
        return None;
    }
    let destination = match action {
        EAST if state % WIDTH + 1 < WIDTH => state + 1,
        SOUTH if state / WIDTH + 1 < HEIGHT => state + WIDTH,
        _ => return None,
    };
    (destination != WALL).then_some(destination)
}

fn arrival_reward(state: usize) -> f32 {
    match state {
        GOAL => 1.0,
        TRAP => -1.0,
        HIGH_ROUTE => -0.5,
        _ => 0.0,
    }
}

pub(super) fn legal(state: usize) -> impl Iterator<Item = usize> {
    (0..DIRECTIONS).filter(move |action| destination(state, *action).is_some())
}

fn legal_pairs() -> impl Iterator<Item = (usize, usize)> {
    states().flat_map(|state| legal(state).map(move |action| (state, action)))
}

pub(super) fn observe(state: usize) -> Observation {
    Observation::discrete(
        state as u64,
        Some(
            (0..DIRECTIONS)
                .map(|a| destination(state, a).is_some())
                .collect(),
        ),
    )
}

pub(super) fn cell(observation: &Observation) -> Result<usize> {
    observation
        .as_discrete()
        .and_then(|discrete| usize::try_from(discrete.symbol).ok())
        .filter(|&cell| cell < CELLS && cell != WALL)
        .ok_or_else(|| Error::Shape("expected a Gridworld cell symbol".into()))
}

pub(super) fn spaces() -> TemporalSpaces {
    TemporalSpaces {
        observations: ObservationSpace::Discrete(DiscreteObservationSpace {
            symbols: CELLS as u64,
        }),
        actions: ACTIONS,
    }
}

pub(super) struct Grid {
    initial: usize,
    current: usize,
}

impl Grid {
    pub(super) fn starting_at(initial: usize) -> Self {
        Self {
            initial,
            current: initial,
        }
    }
}

impl Env for Grid {
    fn observation_space(&self) -> ObservationSpace {
        spaces().observations
    }

    fn action_space(&self) -> &ActionSpace {
        &ACTIONS
    }

    fn reset(&mut self, _: Option<u64>) -> Result<EpisodeStart> {
        self.current = self.initial;
        Ok(EpisodeStart::fresh(observe(self.current)))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        observe(self.current).check_action(&ACTIONS, action)?;
        let Action::Discrete(action) = action else {
            unreachable!()
        };
        self.current = destination(self.current, *action).expect("legal Gridworld action");
        Ok(Step {
            observation: observe(self.current),
            reward: arrival_reward(self.current).into(),
            terminated: !live(self.current),
            truncated: false,
        })
    }
}

impl PlanningModel for Grid {
    type Context = usize;
    type Action = usize;
    type Observation = usize;

    fn actions(&self, state: &usize, actions: &mut Vec<usize>) {
        actions.extend(legal(*state));
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: GAMMA,
        }
    }
}

impl EnumerableModel for Grid {
    fn enumerate(
        &self,
        state: &usize,
        action: &usize,
        _: NonZeroU32,
    ) -> Result<OutcomeDistribution<usize, usize>> {
        let next = destination(*state, *action)
            .ok_or_else(|| Error::Action("illegal Gridworld model action".into()))?;
        Ok(OutcomeDistribution {
            outcomes: vec![WeightedTransition {
                probability: 1.0,
                transition: Transition {
                    observation: next,
                    reward: arrival_reward(next).into(),
                    duration: NonZeroU16::MIN,
                    next: live(next).then_some(next),
                },
            }],
            complete: true,
        })
    }
}

pub(super) fn fixed_probes() -> Result<ProbeSet> {
    let cells: Vec<_> = states().collect();
    ProbeSet::new_in(
        "grid-cells",
        ProbePopulation::FixedReference,
        "All live Gridworld cells in row order, with integer display coordinates",
        cells
            .iter()
            .map(|&state| DiagnosticInput::Projected {
                observation: observe(state),
                projection: DisplayProjection {
                    labels: vec!["X".into(), "Y".into()],
                    coordinates: vec![(state % WIDTH) as f64, (state / WIDTH) as f64],
                },
            })
            .collect(),
        None,
        spaces().observations,
        ACTIONS,
    )?
    .with_row_ids(cells.into_iter().map(|state| state as u64).collect())
}

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct CellActionFeatures(FeatureSpec);

impl CellActionFeatures {
    pub(super) fn new() -> Self {
        let terms = legal_pairs()
            .map(|(cell, action)| {
                FeatureIdentity::new(
                    format!("cell_{cell}_action_{action}"),
                    "one for this legal cell-action pair, zero otherwise",
                )
            })
            .collect();
        Self(FeatureSpec {
            namespace: "example.directed_gridworld.cell_action".into(),
            version: 1,
            terms,
            configuration: vec![
                format!("{WIDTH} columns by {HEIGHT} rows, row-major cells"),
                "action 0 is East and 1 is South".into(),
            ],
            representation: None,
        })
    }
}

impl FeatureMap for CellActionFeatures {
    fn spec(&self) -> &FeatureSpec {
        &self.0
    }

    fn validate_spaces(&self, bound: &TemporalSpaces) -> Result<()> {
        if bound != &spaces() || self.0 != Self::new().0 {
            return Err(Error::Config(
                "incompatible Gridworld cell-action feature contract".into(),
            ));
        }
        Ok(())
    }

    fn encode(&self, history: &PublicHistory, actions: &[Action]) -> Result<Vec<Vec<f64>>> {
        let observation = history.observation();
        let state = cell(observation)?;
        actions
            .iter()
            .map(|action| {
                observation.check_action(&ACTIONS, action)?;
                let Action::Discrete(index) = action else {
                    unreachable!()
                };
                let column = legal_pairs()
                    .position(|pair| pair == (state, *index))
                    .ok_or_else(|| {
                        Error::Action("Gridworld features require an available action".into())
                    })?;
                let mut row = vec![0.0; self.0.dimension()];
                row[column] = 1.0;
                Ok(row)
            })
            .collect()
    }
}

#[derive(Clone)]
pub(super) struct ExactGrid {
    spec: ReferenceSpec,
    optimal: Solution<usize, usize>,
}

impl ExactGrid {
    pub(super) fn new() -> Result<Self> {
        let model = FiniteModel::from_model(
            &Grid::starting_at(START),
            &states().collect::<Vec<_>>(),
            NonZeroU32::MIN,
        )?;
        let optimal = model.value_iteration(NonZeroU32::MIN, 1e-12)?;
        if optimal.status != Status::Converged || optimal.residual != 0.0 {
            return Err(Error::Numerical(
                "Gridworld dynamic programming did not give an exact fixed point".into(),
            ));
        }
        Ok(Self {
            optimal,
            spec: ReferenceSpec {
                id: "gridworld-exact".into(),
                context: ReferenceContext {
                    environment: ENVIRONMENT.into(),
                    observation_dim: 0,
                    typed_observation_space: Some(spaces().observations),
                    action_space: ACTIONS,
                    conventions: ValueConventions {
                        dynamics: "deterministic directed moves on a 4 by 3 grid".into(),
                        observation: "lossless row-major discrete cell symbol".into(),
                        action: "0 moves East and 1 moves South".into(),
                        reward_units: "grid points".into(),
                        reward_convention: "arrival reward in grid points, larger is better".into(),
                        discount: GAMMA,
                        horizon: "at most five moves to goal or trap".into(),
                        terminal_treatment: "goal and trap end the episode with no continuation"
                            .into(),
                        action_constraints:
                            "moves outside the grid or into the wall are unavailable".into(),
                    },
                    objective: ReferenceObjective::ExpectedDiscountedReturn,
                    horizon: ReferenceHorizon::Finite {
                        decisions: MAX_STEPS as u32,
                    },
                    environment_config: BTreeMap::from([
                        ("columns".into(), ConfigValue::Integer(WIDTH as i64)),
                        ("rows".into(), ConfigValue::Integer(HEIGHT as i64)),
                        ("start".into(), ConfigValue::Integer(START as i64)),
                        ("goal".into(), ConfigValue::Integer(GOAL as i64)),
                        ("trap".into(), ConfigValue::Integer(TRAP as i64)),
                        ("wall".into(), ConfigValue::Integer(WALL as i64)),
                        ("horizon".into(), ConfigValue::Integer(MAX_STEPS as i64)),
                    ]),
                    markov_observation: true,
                    decision_time_coordinate: None,
                },
                quality: ReferenceQuality {
                    kind: ReferenceKind::VerifiedOptimal,
                    method: "Exact successor-first dynamic programming over all live cells".into(),
                    assumptions: vec![
                        "Deterministic moves, acyclic support, and rewards of +1, 0, −0.5 or −1"
                            .into(),
                    ],
                },
                target: ReferenceTarget::VerifiedOptimal,
                lifetime: ReferenceLifetime::Fixed,
                information: ReferenceInformation::LearnerObservation,
                uncertainty: None,
            },
        })
    }

    pub(super) fn solution(&self, state: usize) -> Option<&StateValue<usize, usize>> {
        self.optimal.states.iter().find(|row| row.context == state)
    }
}

impl Actor for ExactGrid {
    fn action_space(&self) -> &ActionSpace {
        &ACTIONS
    }

    fn act_into(
        &self,
        observations: ObservationBatch<'_>,
        _: ActMode,
        _: &mut ChaCha8Rng,
        selected: &mut Vec<Action>,
    ) -> Result<()> {
        selected.clear();
        for observation in observations.iter() {
            let state = cell(observation)?;
            let row = self.solution(state).ok_or_else(|| {
                Error::State("Gridworld reference decision needs a live cell".into())
            })?;
            let choice = Action::Discrete(*row.greedy_action());
            observation.check_action(&ACTIONS, &choice)?;
            selected.push(choice);
        }
        Ok(())
    }
}

impl ReferenceProvider for ExactGrid {
    fn specification(&self) -> &ReferenceSpec {
        &self.spec
    }

    fn actor(&self) -> Option<&dyn Actor> {
        Some(self)
    }

    fn evaluation_mode(&self, _: ActMode) -> ActMode {
        ActMode::Deterministic
    }

    fn state_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
    ) -> Result<Option<f64>> {
        Ok(self.solution(cell(observation)?).map(|row| row.value))
    }

    fn action_value(
        &self,
        _: &ReferenceCheckpoint<'_>,
        observation: &Observation,
        action: &Action,
    ) -> Result<Option<f64>> {
        let Action::Discrete(action) = action else {
            return Ok(None);
        };
        Ok(self
            .solution(cell(observation)?)
            .and_then(|row| {
                row.action_values()
                    .iter()
                    .find(|choice| choice.action == *action)
            })
            .map(|choice| choice.value))
    }
}
