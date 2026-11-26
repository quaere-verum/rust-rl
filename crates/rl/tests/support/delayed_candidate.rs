use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl::{
    Error, Result,
    core::{
        action::{Action, ActionArgument, ActionSpace, ConditionalAction},
        actor::ActMode,
        checkpoint::Snapshot,
        env::{Env, EpisodeStart, Observation, ObservationSpace, Step},
        reward::Reward,
        stateful::{HistoryInput, PreviousInput, StatefulActor},
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, EventStamp, FeatureIdentity,
            INCREMENTAL_EVENT_CONVENTION, PublicEvent, StructuredObservation,
        },
    },
    nn::structured::{InputSchema, RecurrentLightweightConfig, SemanticInput},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const DELAY: usize = 5;
const SIGNAL: EntityId = EntityId { kind: 0, id: 0 };
const CONTINUE: ConditionalAction = ConditionalAction {
    operation: 0,
    argument: None,
};

fn target(right: bool) -> EntityId {
    EntityId {
        kind: 1,
        id: u64::from(right),
    }
}

pub fn continue_action() -> Action {
    Action::Conditional(Arc::new(CONTINUE))
}

pub fn input_schema() -> InputSchema {
    InputSchema {
        version: 2,
        spatial_channels: 0,
        entity_features: 2,
        event_features: 1,
        scalar_features: 1,
        candidate_features: 1,
        operations: vec![ArgumentKind::None, ArgumentKind::Entity],
        semantics: SemanticInput {
            version: 1,
            namespace: "example.delayed_candidate".into(),
            observation_version: 1,
            action_version: 1,
            spatial: vec![],
            entities: vec![
                FeatureIdentity::new("signal_entity", "binary"),
                FeatureIdentity::new("sign", "signed_unit"),
            ],
            events: vec![FeatureIdentity::new("signal_available", "binary")],
            scalars: vec![FeatureIdentity::new("decision_index", "normalized_delay")],
            candidates: vec![FeatureIdentity::new("duration", "primitive_steps")],
            operations: vec!["continue".into(), "choose".into()],
            vocabularies: vec![],
            conventions: vec![INCREMENTAL_EVENT_CONVENTION.into()],
        },
    }
}

pub fn model_config() -> RecurrentLightweightConfig {
    let mut config = RecurrentLightweightConfig::small(input_schema());
    config.model.entity_width = 8;
    config.model.event_width = 8;
    config.model.scalar_width = 8;
    config.model.operation_width = 4;
    config.model.fused_width = 16;
    config.model.head_width = 16;
    config.hidden_width = 16;
    config
}

pub fn score<A: StatefulActor>(actor: &A, mode: ActMode, erase: bool) -> Result<f64> {
    score_env(actor, mode, erase, &DelayedCandidate::new())
}

pub fn score_env<A: StatefulActor>(
    actor: &A,
    mode: ActMode,
    erase: bool,
    template: &DelayedCandidate,
) -> Result<f64> {
    let mut rescued = 0;
    for pair in 0..32 {
        for cue in 0..2 {
            let mut env = template.clone();
            let mut observation = env.reset(Some(20_000 + pair * 2 + cue))?.observation;
            let mut state = actor.assimilate(
                &actor.initial_state()?,
                HistoryInput {
                    observation: &observation,
                    previous: None,
                },
            )?;
            let action = continue_action();
            let waits = env.waits();
            for decision in 0..waits {
                let step = env.step(&action)?;
                if erase && decision + 1 == waits {
                    state = actor.initial_state()?;
                }
                state = actor.assimilate(
                    &state,
                    HistoryInput {
                        observation: &step.observation,
                        previous: Some(PreviousInput::from_transition(
                            &action,
                            &step.reward,
                            &observation,
                        )),
                    },
                )?;
                observation = step.observation;
            }
            let mut rng = ChaCha8Rng::seed_from_u64(30_000 + pair);
            let choice = actor.decide(&state, &observation, mode, &mut rng)?;
            rescued += usize::from(env.step(&choice.action)?.reward.total() > 0.);
        }
    }
    Ok(rescued as f64 / 64.)
}

/// The signal is visible in one public entity at reset. The notebook retains
/// the event's identity and stable link after that entity leaves the table.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DelayedCandidate {
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    rng: ChaCha8Rng,
    decision: usize,
    right: bool,
    #[serde(default, skip_serializing_if = "CueSchedule::is_delayed")]
    pub(crate) cue_schedule: CueSchedule,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum CueSchedule {
    #[default]
    Delayed,
    VisibleAtChoice,
    Immediate,
}

impl CueSchedule {
    fn is_delayed(&self) -> bool {
        *self == Self::Delayed
    }
}

impl DelayedCandidate {
    pub fn new() -> Self {
        let input = input_schema();
        Self {
            observation_space: ObservationSpace::Structured(Arc::new(input.observation_schema())),
            action_space: ActionSpace::Conditional(Arc::new(input.action_grammar())),
            rng: ChaCha8Rng::seed_from_u64(0),
            decision: DELAY + 1,
            right: false,
            cue_schedule: CueSchedule::Delayed,
        }
    }

    pub(crate) fn waits(&self) -> usize {
        DELAY.saturating_sub(self.decision)
    }

    fn observation(&self) -> Observation {
        let mut entities = vec![
            EntityRow {
                id: target(false),
                features: vec![0., -1.],
            },
            EntityRow {
                id: target(true),
                features: vec![0., 1.],
            },
        ];
        if self.decision == 0 || (self.decision == DELAY && !self.cue_schedule.is_delayed()) {
            entities.push(EntityRow {
                id: SIGNAL,
                features: vec![1., if self.right { 1. } else { -1. }],
            });
        }
        let candidates = match self.decision {
            decision if decision < DELAY => vec![ActionCandidate {
                action: CONTINUE,
                features: vec![2.],
                enabled: true,
            }],
            DELAY => [false, true]
                .into_iter()
                .map(|right| ActionCandidate {
                    action: ConditionalAction {
                        operation: 1,
                        argument: Some(ActionArgument::Entity(target(right))),
                    },
                    features: vec![1.],
                    enabled: true,
                })
                .collect(),
            _ => vec![],
        };
        Observation::Structured(Arc::new(StructuredObservation {
            primary_region: 0,
            regions: vec![],
            spatial: None,
            entities,
            events: vec![PublicEvent {
                features: [1.].into(),
                entity: Some(SIGNAL),
                context: [].into(),
                stamp: Some(EventStamp {
                    source: 0,
                    ordinal: 0,
                    tick: 0,
                }),
            }],
            scalars: vec![self.decision as f32 / DELAY as f32],
            candidates,
        }))
    }
}

impl Env for DelayedCandidate {
    fn observation_space(&self) -> ObservationSpace {
        self.observation_space.clone()
    }
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn reset(&mut self, seed: Option<u64>) -> Result<EpisodeStart> {
        self.decision = if self.cue_schedule == CueSchedule::Immediate {
            DELAY
        } else {
            0
        };
        self.right = if let Some(seed) = seed {
            self.rng = ChaCha8Rng::seed_from_u64(seed);
            seed % 2 == 1
        } else {
            self.rng.random()
        };
        Ok(EpisodeStart::fresh(self.observation()))
    }

    fn step(&mut self, action: &Action) -> Result<Step> {
        self.observation()
            .check_action(&self.action_space, action)?;
        let Action::Conditional(action) = action else {
            return Err(Error::Action(
                "delayed task requires a complete action".into(),
            ));
        };
        let terminated = self.decision == DELAY;
        let reward = if terminated {
            let correct = action.argument == Some(ActionArgument::Entity(target(self.right)));
            Reward::One(if correct { 1. } else { -1. })
        } else {
            Reward::from_primitives(vec![0., 0.])?
        };
        self.decision += 1;
        Ok(Step {
            observation: self.observation(),
            reward,
            terminated,
            truncated: false,
        })
    }
}

impl Snapshot for DelayedCandidate {
    fn validate_snapshot(&self) -> Result<()> {
        if self.decision > DELAY + 1 {
            return Err(Error::State("invalid delayed task decision clock".into()));
        }
        input_schema().validate_spaces(&self.observation_space, &self.action_space)
    }
}
