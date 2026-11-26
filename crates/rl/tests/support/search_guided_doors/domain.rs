//! Known finite physical model. This module contains no DP or training labels.
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
use rl::{
    Result,
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction},
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        reward::Reward,
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity,
            INCREMENTAL_EVENT_CONVENTION, StructuredObservation,
        },
    },
    models::predictive::{PredictiveModel, Transition},
    nn::structured::{InputSchema, LightweightConfig, SemanticInput},
};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU16, sync::Arc};

pub const GAMMA: f64 = 0.9;
fn ensure(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(rl::Error::State(message.into()))
    }
}
pub const SCAN: ConditionalAction = ConditionalAction {
    operation: 0,
    argument: None,
};
pub fn guess(right: bool) -> ConditionalAction {
    ConditionalAction {
        operation: 1,
        argument: Some(ActionArgument::Entity(EntityId {
            kind: 1,
            id: u64::from(right),
        })),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub prior_milli: u16,
    pub cost_milli: u16,
}
impl Task {
    pub fn new(prior_milli: u16, cost_milli: u16) -> Result<Self> {
        let task = Self {
            prior_milli,
            cost_milli,
        };
        task.validate()?;
        Ok(task)
    }
    pub fn validate(self) -> Result<()> {
        ensure(
            (1..1000).contains(&self.prior_milli) && self.cost_milli <= 1000,
            "invalid noisy-door public task",
        )
    }
    pub fn cost(self) -> f32 {
        self.cost_milli as f32 / 1000.
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Public {
    pub task: Task,
    pub reading: Option<bool>,
}
impl Public {
    pub fn root(task: Task) -> Self {
        Self {
            task,
            reading: None,
        }
    }
    /// Integer Bayes masses, exactly derived from the factual task and reading.
    pub fn masses(self) -> [u32; 2] {
        let right = u32::from(self.task.prior_milli);
        match self.reading {
            None => [1000 - right, right],
            Some(false) => [(1000 - right) * 850, right * 150],
            Some(true) => [(1000 - right) * 150, right * 850],
        }
    }
    pub fn probability_right(self) -> f64 {
        let [left, right] = self.masses();
        f64::from(right) / f64::from(left + right)
    }
}

pub fn network_config() -> LightweightConfig {
    let input = InputSchema {
        version: 2,
        spatial_channels: 0,
        entity_features: 1,
        event_features: 1,
        scalar_features: 4,
        candidate_features: 1,
        operations: vec![ArgumentKind::None, ArgumentKind::Entity],
        semantics: SemanticInput {
            version: 1,
            namespace: "example.search_guided_doors.v2".into(),
            observation_version: 2,
            action_version: 1,
            spatial: vec![],
            entities: vec![FeatureIdentity::new("door", "signed_unit")],
            events: vec![FeatureIdentity::new("unused", "empty")],
            scalars: vec![
                FeatureIdentity::new("prior_right", "milli_div_1000_f32"),
                FeatureIdentity::new("remaining_scan_cost", "milli_div_1000_f32_zero_after_scan"),
                FeatureIdentity::new("posterior_right", "integer_bayes_f32"),
                FeatureIdentity::new("reading", "signed_unit_or_zero"),
            ],
            candidates: vec![FeatureIdentity::new("duration", "ticks")],
            operations: vec!["scan".into(), "guess".into()],
            vocabularies: vec![],
            conventions: vec![INCREMENTAL_EVENT_CONVENTION.into()],
        },
    };
    let mut config = LightweightConfig::small(input);
    config.entity_width = 8;
    config.event_width = 8;
    config.scalar_width = 16;
    config.operation_width = 8;
    config
}
pub fn frame(public: Public) -> StructuredObservation {
    StructuredObservation {
        primary_region: 0,
        regions: vec![],
        spatial: None,
        events: vec![],
        entities: [false, true]
            .map(|r| EntityRow {
                id: EntityId {
                    kind: 1,
                    id: u64::from(r),
                },
                features: vec![if r { 1. } else { -1. }],
            })
            .to_vec(),
        scalars: vec![
            public.task.prior_milli as f32 / 1000.,
            if public.reading.is_none() {
                public.task.cost()
            } else {
                0.
            },
            public.probability_right() as f32,
            public.reading.map_or(0., |r| if r { 1. } else { -1. }),
        ],
        candidates: [SCAN, guess(false), guess(true)]
            .map(|action| ActionCandidate {
                action,
                features: vec![if action == SCAN { 2. } else { 1. }],
                enabled: action != SCAN || public.reading.is_none(),
            })
            .to_vec(),
    }
}
pub fn observation(public: Public) -> Observation {
    Observation::Structured(Arc::new(frame(public)))
}
pub fn decode(observation: &Observation) -> Result<Public> {
    let Observation::Structured(recorded) = observation else {
        return Err(rl::Error::Shape(
            "door observation is not structured".into(),
        ));
    };
    ensure(
        recorded.scalars.len() == 4 && recorded.scalars.iter().all(|v| v.is_finite()),
        "invalid public door features",
    )?;
    let task = Task::new(
        (recorded.scalars[0] * 1000.).round() as u16,
        (recorded.scalars[1] * 1000.).round() as u16,
    )?;
    let public = Public {
        task,
        reading: match recorded.scalars[3] {
            -1. => Some(false),
            0. => None,
            1. => Some(true),
            _ => return Err(rl::Error::State("invalid public door reading".into())),
        },
    };
    ensure(
        **recorded == frame(public),
        "door frame differs from exact public task/reading encoding",
    )?;
    Ok(public)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DoorsModel;
impl PlanningModel for DoorsModel {
    type Context = Public;
    type Action = ConditionalAction;
    type Observation = bool;

    fn actions(&self, context: &Public, out: &mut Vec<ConditionalAction>) {
        if context.reading.is_none() {
            out.push(SCAN);
        }
        out.extend([guess(false), guess(true)]);
    }

    fn planning_returns(&self) -> PlanningReturns {
        PlanningReturns {
            objective: ReturnObjective::Physical,
            gamma: GAMMA,
        }
    }
}

impl PredictiveModel for DoorsModel {
    type Hidden = bool;
    fn sample_root(&self, context: &Public, rng: &mut ChaCha8Rng) -> Result<bool> {
        context.task.validate()?;
        let [left, right] = context.masses();
        Ok(rng.random_range(0..left + right) >= left)
    }
    fn rollout(&self, context: &Public) -> Option<ConditionalAction> {
        Some(guess(context.probability_right() > 0.5))
    }
    fn simulate(
        &self,
        context: &Public,
        hidden: &mut bool,
        action: &ConditionalAction,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Public, bool>> {
        if *action == SCAN {
            ensure(context.reading.is_none(), "repeated imagined door scan")?;
            let reading = if rng.random_range(0..1000) < 850 {
                *hidden
            } else {
                !*hidden
            };
            Ok(Transition {
                observation: reading,
                reward: -f64::from(context.task.cost()),
                duration: NonZeroU16::new(2).unwrap(),
                next: Some(Public {
                    reading: Some(reading),
                    ..*context
                }),
            })
        } else {
            ensure(
                [guess(false), guess(true)].contains(action),
                "unknown imagined door action",
            )?;
            let correct = *action == guess(*hidden);
            Ok(Transition {
                observation: correct,
                reward: if correct { 1. } else { -1. },
                duration: NonZeroU16::MIN,
                next: None,
            })
        }
    }
}

/// Domain-separated SplitMix64 derivation; actual stream seeds are reported by
/// the study. Physical task, incident and sensor streams are independently owned.
pub fn stream_seed(seed: u64, domain: u64) -> u64 {
    let mut value = seed ^ domain;
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

/// Physical environment: stepping never calls the predictive model. Hidden
/// truth is used only to generate actual readings/rewards and to resume the env.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoorsEnv {
    tasks: Vec<Task>,
    public: Public,
    hidden_right: bool,
    ended: bool,
    task_rng: ChaCha8Rng,
    incident_rng: ChaCha8Rng,
    sensor_rng: ChaCha8Rng,
    space: ActionSpace,
}
impl DoorsEnv {
    pub fn new(tasks: Vec<Task>) -> Result<Self> {
        ensure(!tasks.is_empty(), "empty door task family")?;
        for task in &tasks {
            task.validate()?;
        }
        Ok(Self {
            public: Public::root(tasks[0]),
            tasks,
            hidden_right: false,
            ended: true,
            task_rng: ChaCha8Rng::seed_from_u64(0),
            incident_rng: ChaCha8Rng::seed_from_u64(0),
            sensor_rng: ChaCha8Rng::seed_from_u64(0),
            space: ActionSpace::Conditional(Arc::new(network_config().input.action_grammar())),
        })
    }
}
impl Env for DoorsEnv {
    fn observation_space(&self) -> ObservationSpace {
        ObservationSpace::Structured(Arc::new(network_config().input.observation_schema()))
    }
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }
    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        if let Some(seed) = seed {
            self.task_rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, 0x5441534b));
            self.incident_rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, 0x494e4349));
            self.sensor_rng = ChaCha8Rng::seed_from_u64(stream_seed(seed, 0x53454e53));
        }
        let task = self.tasks[self.task_rng.random_range(0..self.tasks.len())];
        self.public = Public::root(task);
        self.hidden_right = self.incident_rng.random_range(0..1000) < task.prior_milli;
        self.ended = false;
        Ok(EpisodeStart::fresh(observation(self.public)))
    }
    fn step(&mut self, action: &Action) -> Result<Step> {
        ensure(!self.ended, "door episode already ended")?;
        observation(self.public).check_action(&self.space, action)?;
        let Action::Conditional(action) = action else {
            unreachable!("checked conditional action")
        };
        let reward = if **action == SCAN {
            let reliable = self.sensor_rng.random_range(0..1000) < 850;
            self.public.reading = Some(if reliable {
                self.hidden_right
            } else {
                !self.hidden_right
            });
            Reward::from_primitives(vec![-self.public.task.cost(), 0.])?
        } else {
            self.ended = true;
            if **action == guess(self.hidden_right) {
                1f32.into()
            } else {
                (-1f32).into()
            }
        };
        Ok(Step {
            observation: observation(self.public),
            reward,
            terminated: self.ended,
            truncated: false,
        })
    }
}
impl Snapshot for DoorsEnv {
    fn validate_snapshot(&self) -> Result<()> {
        ensure(
            !self.tasks.is_empty()
                && self.tasks.contains(&self.public.task)
                && self.space
                    == ActionSpace::Conditional(Arc::new(network_config().input.action_grammar())),
            "invalid door environment snapshot",
        )?;
        for task in &self.tasks {
            task.validate()?;
        }
        self.public.task.validate()
    }
}
