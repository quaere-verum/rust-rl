//! Finite public-belief reference and a separate factual environment fixture.
#![allow(dead_code)]
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rl::models::predictive::{PlanningModel, PlanningReturns, ReturnObjective};
#[cfg(feature = "neural")]
use rl::nn::structured::{InputSchema, RecurrentLightweightConfig, SemanticInput};
use rl::{
    Result,
    core::{
        action::{
            Action, ActionArgument, ActionGrammar, ActionSpace, ConditionalAction, Operation,
        },
        env::{EpisodeStart, Observation, ObservationSpace},
        structured::{
            ActionCandidate, ArgumentKind, EntityId, EntityRow, FeatureIdentity,
            INCREMENTAL_EVENT_CONVENTION, ObservationSchema, StructuredObservation,
        },
        temporal::TemporalSpaces,
    },
    data::temporal::{BehaviorMetadata, EpisodeStore, TemporalSink, TemporalTransition},
    models::predictive::{PredictiveModel, Transition},
};
use std::{num::NonZeroU16, sync::Arc};

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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Public {
    pub reading: Option<bool>,
}
pub struct Doors;
pub fn returns() -> PlanningReturns {
    PlanningReturns {
        objective: ReturnObjective::Physical,
        gamma: 0.9,
    }
}
impl PlanningModel for Doors {
    type Context = Public;
    type Action = ConditionalAction;
    type Observation = bool;

    fn actions(&self, public: &Public, out: &mut Vec<ConditionalAction>) {
        if public.reading.is_none() {
            out.push(SCAN);
        }
        out.extend([guess(false), guess(true)]);
    }

    fn planning_returns(&self) -> PlanningReturns {
        returns()
    }
}

impl PredictiveModel for Doors {
    type Hidden = bool;
    fn sample_root(&self, public: &Public, rng: &mut ChaCha8Rng) -> Result<bool> {
        Ok(rng.random_bool(public.reading.map_or(0.5, |r| if r { 0.85 } else { 0.15 })))
    }
    fn rollout(&self, public: &Public) -> Option<ConditionalAction> {
        Some(guess(public.reading.unwrap_or(false)))
    }
    fn simulate(
        &self,
        _: &Public,
        hidden: &mut bool,
        action: &ConditionalAction,
        rng: &mut ChaCha8Rng,
    ) -> Result<Transition<Public, bool>> {
        if *action == SCAN {
            let reading = if rng.random_bool(0.85) {
                *hidden
            } else {
                !*hidden
            };
            Ok(Transition {
                observation: reading,
                reward: 0.,
                duration: NonZeroU16::new(2).unwrap(),
                next: Some(Public {
                    reading: Some(reading),
                }),
            })
        } else {
            let success = *action == guess(*hidden);
            Ok(Transition {
                observation: success,
                reward: f64::from(success),
                duration: NonZeroU16::MIN,
                next: None,
            })
        }
    }
}

/// Independent finite-belief DP: enumerate hidden worlds and observations,
/// update the posterior by Bayes' rule, then maximize continuation.
pub fn exact_values(prior_right: f64, scan_available: bool, horizon: u16) -> Vec<f64> {
    if horizon == 0 {
        return vec![];
    }
    let mut result = Vec::new();
    if scan_available {
        let mut value = 0.;
        if horizon > 1 {
            for reading in [false, true] {
                let right_joint = prior_right * if reading { 0.85 } else { 0.15 };
                let left_joint = (1. - prior_right) * if reading { 0.15 } else { 0.85 };
                let mass = right_joint + left_joint;
                let continuation = exact_values(right_joint / mass, false, horizon - 1)
                    .into_iter()
                    .fold(0., f64::max);
                value += mass * continuation;
            }
        }
        result.push(0.9f64.powi(2) * value);
    }
    result.extend([1. - prior_right, prior_right]);
    result
}

fn schemas() -> (ObservationSchema, ActionGrammar) {
    let namespace = "test.planning_doors";
    let observations = ObservationSchema {
        namespace: namespace.into(),
        version: 1,
        spatial: vec![],
        entities: vec![FeatureIdentity::new("door", "signed_unit")],
        events: vec![FeatureIdentity::new("reading", "signed_unit")],
        scalars: vec![FeatureIdentity::new("reading", "signed_unit_or_zero")],
        vocabularies: vec![],
        conventions: vec![INCREMENTAL_EVENT_CONVENTION.into()],
    };
    let actions = ActionGrammar {
        namespace: namespace.into(),
        version: 1,
        operations: vec![
            Operation {
                name: "scan".into(),
                argument: ArgumentKind::None,
            },
            Operation {
                name: "guess".into(),
                argument: ArgumentKind::Entity,
            },
        ],
        candidate_features: vec![FeatureIdentity::new("duration", "ticks")],
    };
    (observations, actions)
}

pub fn spaces() -> TemporalSpaces {
    let (observations, actions) = schemas();
    TemporalSpaces {
        observations: ObservationSpace::Structured(Arc::new(observations)),
        actions: ActionSpace::Conditional(Arc::new(actions)),
    }
}

#[cfg(feature = "neural")]
pub fn model_config() -> RecurrentLightweightConfig {
    let (observations, actions) = schemas();
    let input = InputSchema {
        version: 2,
        spatial_channels: 0,
        entity_features: observations.entities.len(),
        event_features: observations.events.len(),
        scalar_features: observations.scalars.len(),
        candidate_features: actions.candidate_features.len(),
        operations: actions.operations.iter().map(|op| op.argument).collect(),
        semantics: SemanticInput {
            version: 1,
            namespace: observations.namespace,
            observation_version: observations.version,
            action_version: actions.version,
            spatial: observations.spatial,
            entities: observations.entities,
            events: observations.events,
            scalars: observations.scalars,
            candidates: actions.candidate_features,
            operations: actions.operations.into_iter().map(|op| op.name).collect(),
            vocabularies: observations.vocabularies,
            conventions: observations.conventions,
        },
    };
    let mut c = RecurrentLightweightConfig::small(input);
    c.model.entity_width = 4;
    c.model.event_width = 4;
    c.model.scalar_width = 4;
    c.model.operation_width = 4;
    c.model.fused_width = 8;
    c.model.head_width = 8;
    c.hidden_width = 8;
    c
}
pub fn frame(public: Public) -> Observation {
    Observation::Structured(Arc::new(StructuredObservation {
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
        scalars: vec![public.reading.map_or(0., |r| if r { 1. } else { -1. })],
        candidates: [SCAN, guess(false), guess(true)]
            .map(|action| ActionCandidate {
                action,
                features: vec![if action == SCAN { 2. } else { 1. }],
                enabled: action != SCAN || public.reading.is_none(),
            })
            .to_vec(),
    }))
}

/// Actual deterministic episode: the right door is hidden, scanning observes a
/// positive reading, then the agent guesses. These steps never call simulate.
pub fn factual_store() -> Result<EpisodeStore> {
    let spaces = spaces();
    let mut store = EpisodeStore::new(16, spaces.observations, spaces.actions)?;
    let root = frame(Public { reading: None });
    let reading = frame(Public {
        reading: Some(true),
    });
    store.begin_episode(7, &EpisodeStart::fresh(root.clone()))?;
    for (timestep, observation, action, reward, next_observation, terminated) in [
        (
            0,
            root,
            SCAN,
            rl::core::reward::Reward::from_primitives(vec![0., 0.])?,
            reading.clone(),
            false,
        ),
        (1, reading.clone(), guess(true), 1f32.into(), reading, true),
    ] {
        store.push(&TemporalTransition {
            episode_id: 7,
            timestep,
            observation,
            action: Action::Conditional(Arc::new(action)),
            reward,
            next_observation,
            terminated,
            truncated: false,
            behavior: BehaviorMetadata::default(),
        })?;
    }
    Ok(store)
}
